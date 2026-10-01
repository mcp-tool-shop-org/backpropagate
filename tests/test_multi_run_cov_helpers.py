"""Behavioural tests for the small, mostly pure ``multi_run`` helpers:
liveness probing, HF callback bridges, constructor validation, OOM
classification, learning-rate / data-window arithmetic, result assembly and the
``main()`` CLI.

Mocked boundaries are named in each docstring. NOTE: ``os.kill(pid, 0)`` is never
called for real here -- on Windows signal 0 is ``CTRL_C_EVENT``, so a real probe
could deliver Ctrl+C to a process group. Every ``_pid_alive`` test therefore
patches ``os.kill``.
"""

from __future__ import annotations

import errno
import logging
import math
import os
import sys
import threading
import types
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from backpropagate import multi_run
from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy, RunHistoryManager
from backpropagate.exceptions import ConfigurationError
from backpropagate.multi_run import (
    MergeMode,
    MultiRunConfig,
    MultiRunResult,
    MultiRunTrainer,
    RunResult,
    _build_abort_callback,
    _build_multi_run_step_callback,
    _current_host,
    _pid_alive,
)
from backpropagate.slao import MergeResult

MR_LOGGER = "backpropagate.multi_run"


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    """CPU-only, deterministic regardless of the rig's GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _trainer(tmp_path=None, **cfg):
    if tmp_path is not None:
        cfg.setdefault("checkpoint_dir", str(tmp_path))
    return MultiRunTrainer(model="m", config=MultiRunConfig(**cfg))


# =============================================================================
# _pid_alive  (os.kill is the mocked OS boundary)
# =============================================================================


class TestPidAlive:
    @pytest.mark.parametrize("pid", [None, 0, -7])
    def test_missing_or_non_positive_pid_is_dead_without_probing(self, monkeypatch, pid):
        def forbidden(*_a):
            raise AssertionError("os.kill must not be called for an invalid pid")

        monkeypatch.setattr(os, "kill", forbidden)
        assert _pid_alive(pid) is False

    def test_live_when_the_existence_probe_succeeds(self, monkeypatch):
        calls = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append((pid, sig)))
        assert _pid_alive(4242) is True
        assert calls == [(4242, 0)]  # signal 0 = probe only

    def test_dead_when_no_such_process(self, monkeypatch):
        def gone(*_a):
            raise ProcessLookupError

        monkeypatch.setattr(os, "kill", gone)
        assert _pid_alive(4242) is False

    def test_alive_when_owned_by_another_user(self, monkeypatch):
        def denied(*_a):
            raise PermissionError

        monkeypatch.setattr(os, "kill", denied)
        assert _pid_alive(4242) is True

    def test_windows_invalid_parameter_means_dead(self, monkeypatch):
        class _WinInvalid(OSError):
            winerror = 87  # ERROR_INVALID_PARAMETER

        def invalid(*_a):
            raise _WinInvalid(errno.EINVAL, "invalid parameter")

        monkeypatch.setattr(os, "kill", invalid)
        assert _pid_alive(4242) is False

    @pytest.mark.parametrize(
        "code, expected",
        [(errno.ESRCH, False), (errno.EIO, True)],
    )
    def test_other_oserrors_are_judged_by_errno(self, monkeypatch, code, expected):
        def err(*_a):
            raise OSError(code, "probe failed")

        monkeypatch.setattr(os, "kill", err)
        # unknown errors fail safe: assume the holder is alive
        assert _pid_alive(4242) is expected


# =============================================================================
# _entry_is_live / _stamp_liveness
# =============================================================================


class TestEntryIsLive:
    """Mocked: ``_current_host`` and ``_pid_alive`` (the OS boundary)."""

    @pytest.fixture
    def host(self, monkeypatch):
        monkeypatch.setattr(multi_run, "_current_host", lambda: "this-box")
        monkeypatch.setattr(multi_run, "_pid_alive", lambda pid: pid == 111)

    def _hb(self, delta_seconds):
        return (datetime.now() - timedelta(seconds=delta_seconds)).isoformat()

    def test_live_pid_on_this_host_is_enough(self, host):
        assert MultiRunTrainer._entry_is_live({"host": "this-box", "pid": 111}) is True

    def test_dead_pid_falls_through_to_a_fresh_heartbeat(self, host):
        entry = {"host": "this-box", "pid": 222, "heartbeat_at": self._hb(60)}
        assert MultiRunTrainer._entry_is_live(entry) is True

    def test_dead_pid_and_stale_heartbeat_is_a_crashed_orphan(self, host):
        entry = {"host": "this-box", "pid": 222, "heartbeat_at": self._hb(3600)}
        assert MultiRunTrainer._entry_is_live(entry) is False

    def test_other_host_pid_is_ignored_even_if_the_number_matches(self, host):
        entry = {"host": "elsewhere", "pid": 111, "heartbeat_at": self._hb(3600)}
        assert MultiRunTrainer._entry_is_live(entry) is False

    def test_fresh_heartbeat_from_another_host_counts_as_live(self, host):
        entry = {"host": "elsewhere", "pid": 111, "heartbeat_at": self._hb(5)}
        assert MultiRunTrainer._entry_is_live(entry) is True

    def test_heartbeat_from_the_future_is_not_trusted(self, host):
        entry = {"heartbeat_at": (datetime.now() + timedelta(hours=2)).isoformat()}
        assert MultiRunTrainer._entry_is_live(entry) is False

    def test_timezone_aware_heartbeat_is_treated_as_unparseable(self, host):
        entry = {"heartbeat_at": datetime.now(timezone.utc).isoformat()}
        assert MultiRunTrainer._entry_is_live(entry) is False

    def test_garbage_heartbeat_is_not_live(self, host):
        assert MultiRunTrainer._entry_is_live({"heartbeat_at": "yesterday-ish"}) is False

    def test_legacy_entry_without_liveness_fields_is_resumable(self, host):
        assert MultiRunTrainer._entry_is_live({"run_id": "old"}) is False


class TestStampLiveness:
    def test_without_history_or_run_id_is_a_noop(self):
        mrt = _trainer()
        mrt._stamp_liveness()  # nothing bound yet: must not raise
        mrt._run_history = object()  # would blow up if touched
        mrt._stamp_liveness()  # still no run_id

    def test_records_host_pid_and_heartbeat_on_the_running_entry(self, tmp_path):
        history = RunHistoryManager(str(tmp_path))
        history.record_run_started(run_id="rid-1", model_name="m", session_kind="multi_run")
        mrt = _trainer(tmp_path)
        mrt._run_history, mrt._run_id = history, "rid-1"

        before = datetime.now()
        mrt._stamp_liveness()

        entry = history.get_run("rid-1")
        assert entry["host"] == _current_host()
        assert entry["pid"] == os.getpid()
        assert before <= datetime.fromisoformat(entry["heartbeat_at"]) <= datetime.now()

    def test_history_failure_is_swallowed(self, caplog):
        class _Broken:
            def update_run(self, *a, **k):
                raise OSError("locked")

        mrt = _trainer()
        mrt._run_history, mrt._run_id = _Broken(), "rid"
        with caplog.at_level(logging.DEBUG, logger=MR_LOGGER):
            mrt._stamp_liveness()
        assert any("_stamp_liveness failed (non-fatal)" in r.getMessage() for r in caplog.records)


# =============================================================================
# HF callback bridges
# =============================================================================


class TestCallbackBridges:
    def test_bridges_are_none_when_transformers_cannot_be_imported(self, monkeypatch, caplog):
        mrt = _trainer()
        mrt.on_step = lambda *a: None
        monkeypatch.setitem(sys.modules, "transformers", None)  # forces ImportError

        with caplog.at_level(logging.DEBUG, logger=MR_LOGGER):
            assert _build_abort_callback(mrt) is None
            assert _build_multi_run_step_callback(mrt, 3) is None

        msgs = [r.getMessage() for r in caplog.records]
        assert any("mid-run abort will fall back" in m for m in msgs)
        assert any("on_step will not fire for run 3" in m for m in msgs)

    def test_step_bridge_is_none_without_an_on_step_callback(self):
        assert _build_multi_run_step_callback(_trainer(), 1) is None

    def _bridge(self, on_step):
        mrt = _trainer()
        mrt.on_step = on_step
        return mrt, _build_multi_run_step_callback(mrt, 4)

    def _state(self, step=7, history=None):
        return types.SimpleNamespace(global_step=step, log_history=history or [])

    def test_loss_in_logs_is_forwarded_with_run_index_and_step(self):
        seen = []
        _mrt, cb = self._bridge(lambda *a: seen.append(a))
        cb.on_log(None, self._state(step=7), None, logs={"loss": "1.25"})
        assert seen == [(4, 7, 1.25)]

    def test_falls_back_to_the_log_history_tail(self):
        seen = []
        _mrt, cb = self._bridge(lambda *a: seen.append(a))
        history = [{"loss": 9.0}, {"loss": "not-a-number"}, {"eval_loss": 1.0}]
        # logs carry no loss; walk the history newest-first, skipping junk
        cb.on_log(None, self._state(step=3, history=history), None, logs={"lr": 1e-4})
        assert seen == [(4, 3, 9.0)]

    def test_unparseable_logged_loss_also_falls_back_to_history(self):
        seen = []
        _mrt, cb = self._bridge(lambda *a: seen.append(a))
        cb.on_log(None, self._state(history=[{"loss": 2.5}]), None, logs={"loss": "nan?"})
        assert seen == [(4, 7, 2.5)]

    def test_no_usable_loss_means_no_callback(self):
        seen = []
        _mrt, cb = self._bridge(lambda *a: seen.append(a))
        cb.on_log(None, self._state(), None, logs={"eval_loss": 3.0})
        cb.on_log(None, self._state(history=[{"loss": object()}]), None, logs=None)
        assert seen == []

    def test_missing_global_step_defaults_to_zero(self):
        seen = []
        _mrt, cb = self._bridge(lambda *a: seen.append(a))
        state = types.SimpleNamespace(global_step=None, log_history=[])
        cb.on_log(None, state, None, logs={"loss": 1.0})
        assert seen == [(4, 0, 1.0)]

    def test_a_failing_user_callback_is_logged_not_propagated(self, caplog):
        def bad(*_a):
            raise RuntimeError("dashboard down")

        _mrt, cb = self._bridge(bad)
        with caplog.at_level(logging.WARNING, logger=MR_LOGGER):
            cb.on_log(None, self._state(step=9), None, logs={"loss": 0.5})
        assert any(
            "on_step callback raised error (run_index=4 step=9 loss=0.5000): dashboard down"
            in r.getMessage()
            for r in caplog.records
        )

    def test_callback_detached_after_construction_is_a_noop(self):
        mrt, cb = self._bridge(lambda *a: pytest.fail("must not be called"))
        mrt.on_step = None
        cb.on_log(None, self._state(), None, logs={"loss": 1.0})


# =============================================================================
# Constructor validation / precedence warnings
# =============================================================================


class TestConstructorPrecedence:
    def test_kwarg_overrides_explicit_config_with_a_warning(self, caplog):
        cfg = MultiRunConfig(num_runs=3, steps_per_run=10, samples_per_run=20, checkpoint_dir="a")
        with caplog.at_level(logging.WARNING, logger=MR_LOGGER):
            mrt = MultiRunTrainer(
                model="m", config=cfg, num_runs=7, steps_per_run=11, samples_per_run=21,
                checkpoint_dir="b", merge_mode="simple",
            )

        assert (cfg.num_runs, cfg.steps_per_run, cfg.samples_per_run) == (7, 11, 21)
        assert cfg.checkpoint_dir == "b" and cfg.merge_mode == MergeMode.SIMPLE
        assert mrt.config is cfg
        warned = [r.getMessage() for r in caplog.records if "overrides config." in r.getMessage()]
        for field_name in ("num_runs", "steps_per_run", "samples_per_run", "checkpoint_dir",
                           "merge_mode"):
            assert any(f"convenience kwarg {field_name}=" in m for m in warned), field_name

    def test_agreeing_or_implicit_values_do_not_warn(self, caplog):
        cfg = MultiRunConfig(num_runs=3)
        with caplog.at_level(logging.WARNING, logger=MR_LOGGER):
            MultiRunTrainer(model="m", config=cfg, num_runs=3, merge_mode=MergeMode.SLAO)
            MultiRunTrainer(model="m", num_runs=9, merge_mode="Simple")  # no explicit config
        assert not any("overrides config." in r.getMessage() for r in caplog.records)

    def test_merge_mode_enum_conflict_warns(self, caplog):
        cfg = MultiRunConfig()  # SLAO
        with caplog.at_level(logging.WARNING, logger=MR_LOGGER):
            MultiRunTrainer(model="m", config=cfg, merge_mode=MergeMode.SIMPLE)
        assert cfg.merge_mode == MergeMode.SIMPLE
        assert any("convenience kwarg merge_mode=" in r.getMessage() for r in caplog.records)

    def test_merge_mode_strings_are_case_insensitive(self):
        assert MultiRunTrainer(model="m", merge_mode="SLAO").config.merge_mode == MergeMode.SLAO
        assert MultiRunTrainer(model="m", merge_mode="Simple").config.merge_mode == MergeMode.SIMPLE

    def test_unknown_merge_mode_is_a_configuration_error(self):
        with pytest.raises(ConfigurationError) as exc_info:
            MultiRunTrainer(model="m", merge_mode="turbo")
        assert "Invalid merge mode: 'turbo'" in str(exc_info.value)
        assert exc_info.value.__cause__ is None  # raised ``from None``


# =============================================================================
# OOM classification
# =============================================================================


class TestOomClassification:
    def test_strict_matcher(self):
        assert MultiRunTrainer._is_oom_error(torch.cuda.OutOfMemoryError("x")) is True
        assert MultiRunTrainer._is_oom_error(RuntimeError("CUDA Out Of Memory")) is True
        assert MultiRunTrainer._is_oom_error(RuntimeError("something else")) is False
        assert MultiRunTrainer._is_oom_error(ValueError("out of memory")) is False  # not a RuntimeError

    def test_strict_matcher_survives_a_torch_without_oom_class(self, monkeypatch):
        monkeypatch.delattr(torch.cuda, "OutOfMemoryError")
        assert MultiRunTrainer._is_oom_error(RuntimeError("out of memory")) is True
        assert MultiRunTrainer._is_oom_error(KeyError("x")) is False

    def test_strict_matcher_survives_torch_being_unimportable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert MultiRunTrainer._is_oom_error(RuntimeError("out of memory")) is True

    @pytest.mark.parametrize(
        "marker",
        ["cublas_status_alloc_failed", "cudnn_status_not_initialized", "memory_allocator"],
    )
    def test_adjacent_markers_are_matched_case_insensitively(self, marker):
        exc = RuntimeError(f"boom: {marker.upper()} while running")
        assert MultiRunTrainer._is_oom_adjacent(exc) == (True, marker)

    def test_strict_oom_reports_the_canonical_marker(self):
        assert MultiRunTrainer._is_oom_adjacent(torch.cuda.OutOfMemoryError("x")) == (
            True, "out of memory",
        )

    def test_unrelated_errors_do_not_match(self):
        assert MultiRunTrainer._is_oom_adjacent(RuntimeError("shape mismatch")) == (False, None)
        # a marker in a non-RuntimeError is not an OOM signal
        assert MultiRunTrainer._is_oom_adjacent(ValueError("CUBLAS_STATUS_ALLOC_FAILED")) == (
            False, None,
        )


# =============================================================================
# Accessors, learning rate, data windows
# =============================================================================


class TestCheckpointAccessors:
    def test_none_until_a_manager_exists_then_live(self, tmp_path):
        mrt = _trainer(tmp_path)
        assert mrt.get_checkpoint_manager() is None
        assert mrt.get_checkpoint_stats() is None

        mrt._checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(tmp_path), policy=CheckpointPolicy()
        )
        (tmp_path / "ck").mkdir()
        mrt._checkpoint_manager.register(run_index=1, checkpoint_path=str(tmp_path / "ck"))

        assert mrt.get_checkpoint_manager() is mrt._checkpoint_manager
        assert mrt.get_checkpoint_stats().total_count == 1


class TestLearningRateSchedule:
    def _lrs(self, decay, runs=5):
        mrt = _trainer(num_runs=runs, initial_lr=1e-3, final_lr=1e-4, lr_decay=decay)
        return [mrt._get_learning_rate(i) for i in range(1, runs + 1)]

    def test_linear(self):
        assert self._lrs("linear") == pytest.approx([1e-3, 7.75e-4, 5.5e-4, 3.25e-4, 1e-4])

    def test_cosine(self):
        # final + 0.5 * (initial - final) * (1 + cos(pi * progress)), progress = (i-1)/4
        expected = [1e-4 + 0.5 * 9e-4 * (1 + math.cos(math.pi * p)) for p in (0, .25, .5, .75, 1)]
        assert self._lrs("cosine") == pytest.approx(expected)

    def test_constant_and_unknown_schedules_hold_the_initial_rate(self):
        assert self._lrs("constant") == [1e-3] * 5
        assert self._lrs("no-such-decay") == [1e-3] * 5

    def test_single_run_uses_the_initial_rate(self):
        assert self._lrs("linear", runs=1) == [1e-3]


class _RowsDataset:
    """Real ``datasets.Dataset`` of ``row<i>`` strings."""

    @staticmethod
    def make(n):
        from datasets import Dataset

        return Dataset.from_dict({"text": [f"row{i}" for i in range(n)]})


class TestDataWindows:
    def test_window_wraps_inside_the_train_pool_only(self):
        mrt = _trainer(samples_per_run=8, validate_every_run=True, shuffle_data=False)
        ds = _RowsDataset.make(20)  # holdout = last 2 rows, pool = 18

        run1 = mrt._get_data_chunk(ds, 1)["text"]
        run3 = mrt._get_data_chunk(ds, 3)["text"]  # starts at 16: rows 16,17 then wraps

        assert run1 == [f"row{i}" for i in range(8)]
        assert run3 == ["row16", "row17"] + [f"row{i}" for i in range(6)]
        assert "row18" not in run3 and "row19" not in run3  # validation rows never trained on

    def test_empty_train_pool_is_rejected(self):
        mrt = _trainer(samples_per_run=1, validate_every_run=True)
        with pytest.raises(ConfigurationError) as exc_info:
            mrt._get_data_chunk(_RowsDataset.make(1), 1)  # 10% holdout swallows the only row
        assert "Training pool is empty" in str(exc_info.value)
        assert exc_info.value.details["train_pool_size"] == 0

    def test_empty_pool_has_no_fresh_window(self):
        assert _trainer()._fresh_window_indices(1, 0) == []

    def test_chunk_larger_than_dataset_without_validation_is_rejected(self):
        mrt = _trainer(samples_per_run=50, validate_every_run=False)
        with pytest.raises(ConfigurationError) as exc_info:
            mrt._get_data_chunk(_RowsDataset.make(10), 1)
        assert "exceeds total dataset size (10)" in str(exc_info.value)
        assert exc_info.value.details == {"samples_per_run": 50, "total_samples": 10}

    def test_chunk_larger_than_the_pool_with_validation_is_rejected(self):
        mrt = _trainer(samples_per_run=10, validate_every_run=True)
        with pytest.raises(ConfigurationError) as exc_info:
            mrt._get_data_chunk(_RowsDataset.make(10), 1)  # pool = 9
        assert exc_info.value.details["holdout_size"] == 1

    def test_shuffle_reorders_deterministically_without_changing_membership(self):
        ds = _RowsDataset.make(30)
        plain = _trainer(samples_per_run=10, shuffle_data=False)._get_data_chunk(ds, 2)["text"]
        a = _trainer(samples_per_run=10, shuffle_data=True)._get_data_chunk(ds, 2)["text"]
        b = _trainer(samples_per_run=10, shuffle_data=True)._get_data_chunk(ds, 2)["text"]

        assert a == b  # seeded by (training seed + run index)
        assert a != plain and sorted(a) == sorted(plain)

    def test_replay_has_nothing_to_sample_when_no_prior_window_exists(self):
        mrt = _trainer(samples_per_run=0, replay_fraction=0.3)
        assert mrt._get_replay_samples(_RowsDataset.make(10), 2, 3) is None


# =============================================================================
# Result assembly + merge-history serialisation
# =============================================================================


class TestResultAssembly:
    def test_result_with_no_runs_is_empty_and_checkpointless(self, tmp_path):
        mrt = _trainer(tmp_path)
        mrt._run_id = "rid"
        result = mrt._create_result(1.5)

        assert (result.total_runs, result.total_steps, result.total_samples) == (0, 0, 0)
        assert result.final_loss == 0.0 and result.final_checkpoint_path is None
        assert result.checkpoint_stats is None
        assert result.total_duration_seconds == 1.5 and result.run_id == "rid"
        assert result.merge_mode == "slao" and result.aborted is False

    def test_abort_result_carries_reason_and_run_id(self):
        mrt = _trainer()
        mrt._run_id = "rid"
        result = mrt._create_abort_result("because")
        assert result.aborted is True and result.abort_reason == "because"
        assert result.total_runs == 0 and result.run_id == "rid"

    def test_merge_history_serialises_dataclasses_objects_and_unserialisable_results(self):
        mrt = _trainer()

        @dataclass
        class _Unclonable:
            lock: object = field(default_factory=threading.Lock)  # deepcopy fails

        def run(i, merge_result):
            return RunResult(run_index=i, steps=1, samples=1, final_loss=0.0,
                             merge_result=merge_result)

        plain = types.SimpleNamespace(a=1, _private=2)
        mrt._runs = [
            run(1, None),
            run(2, MergeResult(run_index=2, scale_factor=0.5, a_matrices_merged=1,
                               b_matrices_merged=1, total_params_merged=4,
                               merge_time_seconds=0.1)),
            run(3, plain),
            run(4, _Unclonable()),
        ]

        history = mrt._collect_merge_history()

        assert [h["run_index"] for h in history] == [2, 3, 4]  # run 1 had no merge
        assert history[0]["result"]["scale_factor"] == 0.5
        assert history[0]["result"]["strategy"] == "qiao_mahdavi"
        assert history[1]["result"] == {"a": 1}  # private attrs dropped
        assert isinstance(history[2]["result"], str)  # last-resort str() fallback


# =============================================================================
# main() CLI  (mocked: the trainer -- running it would train a model)
# =============================================================================


class TestMainCli:
    def test_main_builds_the_config_runs_and_prints_a_summary(self, monkeypatch, capsys):
        captured = {}

        class _FakeTrainer:
            def __init__(self, model=None, config=None):
                captured["model"], captured["config"] = model, config

            def run(self, dataset):
                captured["dataset"] = dataset
                return MultiRunResult(
                    total_runs=2, total_steps=14, total_samples=18,
                    total_duration_seconds=120.0, final_loss=0.5,
                    final_checkpoint_path="out/run_002/lora",
                )

        monkeypatch.setattr(multi_run, "SpeedrunTrainer", _FakeTrainer)
        monkeypatch.setattr(
            sys, "argv",
            ["speedrun", "--model", "tiny", "--dataset", "d.jsonl", "--runs", "2",
             "--steps", "7", "--samples", "9", "--mode", "simple", "--output", "out",
             "--lr", "1e-3", "--lr-final", "1e-4"],
        )

        multi_run.main()

        cfg = captured["config"]
        assert captured["model"] == "tiny" and captured["dataset"] == "d.jsonl"
        assert (cfg.num_runs, cfg.steps_per_run, cfg.samples_per_run) == (2, 7, 9)
        assert cfg.merge_mode == MergeMode.SIMPLE and cfg.checkpoint_dir == "out"
        assert (cfg.initial_lr, cfg.final_lr) == (1e-3, 1e-4)
        out = capsys.readouterr().out
        assert "SPEEDRUN COMPLETE" in out
        assert "Runs: 2" in out and "Total steps: 14" in out and "Total samples: 18" in out
        assert "Final loss: 0.5000" in out and "Duration: 2.0 minutes" in out
        assert "Checkpoint: out/run_002/lora" in out
