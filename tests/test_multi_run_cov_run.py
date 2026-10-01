"""Behavioural tests for the ``MultiRunTrainer.run()`` loop.

``run()`` executes for real here: real checkpoint manager + run history on
``tmp_path``, real SLAO merger, real tiny PEFT adapter, real ``SFTConfig`` and
a real ``datasets.Dataset``. Mocked at the true boundaries only:

* ``trl.SFTTrainer`` (the inner training call) -> ``make_fake_sft``;
* ``backpropagate.trainer.Trainer`` (HF-Hub / GPU model load) -> ``FakeInnerTrainer``
  carrying a real PEFT model;
* ``get_gpu_status`` / ``wait_for_safe_gpu`` (no GPU on CI).
"""

from __future__ import annotations

import logging
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("trl")

from backpropagate.checkpoints import RunHistoryManager
from backpropagate.exceptions import BackpropagateError, InvalidSettingError
from backpropagate.gpu_safety import GPUCondition, GPUStatus
from backpropagate.multi_run import MergeMode
from tests.test_multi_run_cov_support import (
    build_env,
    fill_adapter,
    safe_gpu_status,
    text_dataset,
)

MULTI_RUN_LOGGER = "backpropagate.multi_run"


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    """CPU-only and deterministic, whatever GPU the dev rig has."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def env(monkeypatch, tmp_path):
    return build_env(monkeypatch, tmp_path)


# =============================================================================
# End-to-end happy path
# =============================================================================


class TestRunLoopHappyPath:
    def test_three_slao_runs_end_to_end(self, env):
        mrt = env.build(num_runs=3, initial_lr=1e-4, final_lr=1e-5, lr_decay="linear")
        completed = []
        started = []
        mrt.on_run_start = started.append
        mrt.on_run_complete = completed.append

        result = mrt.run(text_dataset(60))

        # -- ordering & shape --------------------------------------------------
        assert [r.run_index for r in result.runs] == [1, 2, 3]
        assert started == [1, 2, 3]
        assert [r.run_index for r in completed] == [1, 2, 3]
        assert result.total_runs == 3
        assert result.total_steps == 6 and result.total_samples == 24
        assert result.aborted is False and result.abort_reason is None
        assert result.merge_mode == "slao"
        assert result.run_boundaries == [0, 2, 4]
        assert result.aggregate_loss_history == [2.5, 1.5] * 3
        assert result.final_loss == 1.5
        assert result.final_checkpoint_path == str(env.tmp_path / "run_003" / "lora")
        assert result.checkpoint_stats is not None
        # -- learning-rate schedule (linear 1e-4 -> 1e-5 over 3 runs) ----------
        lrs = [s.args.learning_rate for s in env.fake.created]
        assert lrs == [pytest.approx(1e-4), pytest.approx(5.5e-5), pytest.approx(1e-5)]
        # -- the inner Trainer was built from the config ------------------------
        kw = env.inner.ctor_kwargs
        assert kw["model"] == "tiny-test"
        assert kw["learning_rate"] == pytest.approx(1e-4)
        assert kw["train_on_responses"] is True
        assert kw["mode"] == "lora"
        assert env.inner.load_model_calls == 1
        # -- SLAO: merged B = EMA of the per-run fills 1, 2, 3 -------------------
        lam2, lam3 = 1 / math.sqrt(2), 1 / math.sqrt(3)
        b2 = 1.0 + lam2 * (2.0 - 1.0)
        b3 = b2 + lam3 * (3.0 - b2)
        acc = mrt._slao_merger.get_merged_lora()
        for k, v in acc.items():
            if ".lora_B." in k:
                assert torch.allclose(v, torch.full_like(v, b3), atol=1e-5)
        assert [m.run_index for m in mrt._slao_merger.merge_history] == [2, 3]
        # -- the history entry was started, stamped and completed ---------------
        entry = RunHistoryManager(str(env.tmp_path)).get_run(result.run_id)
        assert entry["status"] == "completed"
        assert entry["host"] and entry["pid"] and entry["heartbeat_at"]
        assert entry["final_loss"] == pytest.approx(1.5)
        assert [m["run_index"] for m in entry["merge_history"]] == [1, 2, 3]
        # cleanup
        assert mrt._is_running is False

    def test_dataset_loader_path_and_wrap_around_warning(self, env, caplog):
        """Asking for more samples than exist logs a cycle warning; tiny dataset."""
        mrt = env.build(num_runs=2, samples_per_run=12, merge_mode=MergeMode.SIMPLE)

        with caplog.at_level(logging.WARNING, logger=MULTI_RUN_LOGGER):
            result = mrt.run(text_dataset(20))

        assert result.total_runs == 2
        assert any("Requested 24 samples but only 20 available" in r.getMessage()
                   for r in caplog.records)

    def test_inner_trainer_receives_wave6b_overrides(self, env):
        mrt = env.build(
            num_runs=1, merge_mode=MergeMode.SIMPLE, use_dora=True, packing=True,
            init_lora_weights="pissa", lora_preset="quality", optim="adamw_torch",
        )

        mrt.run(text_dataset(30))

        kw = env.inner.ctor_kwargs
        assert kw["use_dora"] is True and kw["packing"] is True
        assert kw["init_lora_weights"] == "pissa"
        assert kw["lora_preset"] == "quality" and kw["optim"] == "adamw_torch"

    def test_session_oom_summary_is_logged_after_a_survived_oom(self, env, caplog):
        def oom(_sft):
            raise RuntimeError("CUDA out of memory")

        mrt = env.build([oom], num_runs=1, merge_mode=MergeMode.SIMPLE)

        with caplog.at_level(logging.INFO, logger=MULTI_RUN_LOGGER):
            result = mrt.run(text_dataset(30))

        assert result.runs[0].oom_retries == 1
        assert any("session_oom_summary" in r.getMessage()
                   and "total_oom_events=1" in r.getMessage() for r in caplog.records)


# =============================================================================
# Validation, early stopping, abort, cooldown
# =============================================================================


class TestRunLoopControlFlow:
    def test_early_stopping_requires_validation(self, env):
        mrt = env.build(early_stopping=True, validate_every_run=False)

        with pytest.raises(InvalidSettingError) as exc_info:
            mrt.run(text_dataset(30))

        assert exc_info.value.setting_name == "early_stopping"
        assert env.fake.created == []  # failed before any model/data work

    def test_early_stopping_ends_the_session_before_the_next_run(self, env):
        """Mocked: ``_compute_validation_loss`` returns a scripted loss curve so the
        stop point is deterministic (real validation is exercised elsewhere)."""
        mrt = env.build(
            num_runs=5, merge_mode=MergeMode.SIMPLE, validate_every_run=True,
            early_stopping=True, early_stopping_patience=1, validation_samples=2,
        )
        losses = iter([2.0, 3.0, 1.0, 1.0, 1.0])
        mrt._compute_validation_loss = lambda *_a: next(losses)

        result = mrt.run(text_dataset(40))

        # run 1 sets the best (2.0); run 2 (3.0) fails to improve with patience 1
        assert len(env.fake.created) == 2  # runs 3..5 never start
        assert result.abort_reason == "Early stopping triggered"
        assert mrt._best_val_loss == 2.0 and mrt._early_stop_counter == 1

    def test_validation_loss_is_recorded_per_run(self, env):
        mrt = env.build(
            num_runs=2, merge_mode=MergeMode.SIMPLE, validate_every_run=True, validation_samples=2
        )

        result = mrt.run(text_dataset(40))

        for r in result.runs:
            assert r.validation_loss is not None and math.isfinite(r.validation_loss)

    def test_abort_between_runs_stops_the_loop_and_skips_cooldown(self, env):
        mrt = env.build(num_runs=4, merge_mode=MergeMode.SIMPLE, pause_on_overheat=True)
        mrt.on_run_complete = lambda r: mrt.abort("operator stop")

        result = mrt.run(text_dataset(40))

        assert result.aborted is True and result.abort_reason == "operator stop"
        assert result.total_runs == 1
        assert len(env.fake.created) == 1
        # preflight (1 call) only: the between-run cooldown check was skipped
        assert env.gpu_calls == 1

    def test_cooldown_waits_when_the_gpu_is_too_hot_between_runs(self, env, monkeypatch):
        """Mocked: ``get_gpu_status`` reports a hot GPU after run 1; ``wait_for_safe_gpu``
        (a polling sleep loop) is replaced with a recorder."""
        mrt = env.build(
            num_runs=2, merge_mode=MergeMode.SIMPLE, pause_on_overheat=True,
            max_temp_c=80.0, cooldown_seconds=12.0,
        )
        hot = GPUStatus(
            available=True, device_name="Fake", vram_total_gb=24.0,
            temperature_c=95.0, condition=GPUCondition.WARM,
        )
        statuses = iter([safe_gpu_status(), hot, hot])
        monkeypatch.setattr(
            "backpropagate.multi_run.get_gpu_status", lambda *a, **k: next(statuses)
        )
        waits = []
        monkeypatch.setattr(
            "backpropagate.multi_run.wait_for_safe_gpu", lambda **kw: waits.append(kw) or True
        )

        mrt.run(text_dataset(40))

        assert waits == [
            {"max_wait_seconds": 12.0, "check_interval": 5.0},
            {"max_wait_seconds": 12.0, "check_interval": 5.0},
        ]


class TestRunLoopGpuPause:
    def _arm_pause(self, mrt):
        mrt._start_gpu_monitor = lambda: mrt._gpu_pause_event.set()

    def test_pause_ceiling_aborts_with_a_structured_error(self, env, monkeypatch):
        mrt = env.build(
            merge_mode=MergeMode.SIMPLE, enable_gpu_monitoring=True, max_pause_seconds=0.05
        )
        self._arm_pause(mrt)
        monkeypatch.setattr("backpropagate.multi_run.time.sleep", lambda _s: None)

        with pytest.raises(BackpropagateError) as exc_info:
            mrt.run(text_dataset(40))

        err = exc_info.value
        assert err.code == "RUNTIME_GPU_TEMPERATURE_CRITICAL"
        assert err.details["max_pause_seconds"] == 0.05
        assert err.details["next_run_index"] == 1
        assert env.fake.created == []  # never trained while the GPU was hot
        # the failure is persisted in the run history
        entry = RunHistoryManager(str(env.tmp_path)).get_run(mrt._run_id)
        assert entry["status"] == "failed"
        assert "BackpropagateError" in entry["failure_reason"]

    def test_run_resumes_once_the_pause_event_clears(self, env, monkeypatch):
        mrt = env.build(
            num_runs=1, merge_mode=MergeMode.SIMPLE, enable_gpu_monitoring=True,
            max_pause_seconds=60.0,
        )
        self._arm_pause(mrt)
        sleeps = []

        def clearing_sleep(seconds):
            sleeps.append(seconds)
            mrt._gpu_pause_event.clear()

        monkeypatch.setattr("backpropagate.multi_run.time.sleep", clearing_sleep)

        result = mrt.run(text_dataset(40))

        assert sleeps == [1.0]  # polled once, then the event cleared
        assert result.total_runs == 1 and len(env.fake.created) == 1

    def test_preflight_failure_returns_abort_result_even_if_history_write_fails(
        self, env, monkeypatch, caplog
    ):
        mrt = env.build(merge_mode=MergeMode.SIMPLE)
        monkeypatch.setattr(
            "backpropagate.multi_run.get_gpu_status",
            lambda *a, **k: GPUStatus(available=False, condition=GPUCondition.UNKNOWN),
        )

        def boom(self, **kwargs):
            raise OSError("history disk full")

        monkeypatch.setattr(RunHistoryManager, "record_run_failed", boom)

        with caplog.at_level(logging.WARNING, logger=MULTI_RUN_LOGGER):
            result = mrt.run(text_dataset(40))

        assert result.aborted is True
        assert result.abort_reason == "GPU safety check failed"
        assert result.total_runs == 0
        assert env.inner.load_model_calls == 0  # never loaded a model
        assert any("record_run_failed failed" in r.getMessage() for r in caplog.records)


# =============================================================================
# History-bookkeeping failure modes (never allowed to gate training)
# =============================================================================


class TestRunLoopHistoryFailures:
    def test_history_start_failure_is_non_fatal(self, env, monkeypatch, caplog):
        mrt = env.build(num_runs=1, merge_mode=MergeMode.SIMPLE)

        def boom(self, **kwargs):
            raise OSError("cannot write history")

        monkeypatch.setattr(RunHistoryManager, "record_run_started", boom)

        with caplog.at_level(logging.WARNING, logger=MULTI_RUN_LOGGER):
            result = mrt.run(text_dataset(30))

        assert result.total_runs == 1  # training went ahead
        assert any("record_run_started failed" in r.getMessage() for r in caplog.records)

    def test_dataset_hash_failure_records_the_run_without_a_hash(self, env, monkeypatch):
        mrt = env.build(num_runs=1, merge_mode=MergeMode.SIMPLE)

        def boom(_dataset):
            raise RuntimeError("cannot hash")

        monkeypatch.setattr("backpropagate.trainer._compute_dataset_hash", boom)

        result = mrt.run(text_dataset(30))

        entry = RunHistoryManager(str(env.tmp_path)).get_run(result.run_id)
        assert entry["status"] == "completed"
        assert not entry.get("dataset_hash")

    def test_loop_failure_is_persisted_with_merge_history_and_reraised(self, env):
        """Run 2 trains to NaN B matrices -> SLAO_MERGE_DIVERGED escapes ``run()``;
        history keeps run 1's merge record and flags the session failed."""

        def nan_run(sft):
            fill_adapter(sft.model, run=2, b_value=float("nan"))
            return type("R", (), {"training_loss": 1.0})()

        def ok_run(sft):
            fill_adapter(sft.model, run=1)
            return type("R", (), {"training_loss": 1.0})()

        mrt = env.build([ok_run, nan_run], num_runs=3)

        with pytest.raises(BackpropagateError) as exc_info:
            mrt.run(text_dataset(60))

        assert exc_info.value.code == "SLAO_MERGE_DIVERGED"
        assert mrt._is_running is False
        entry = RunHistoryManager(str(env.tmp_path)).get_run(mrt._run_id)
        assert entry["status"] == "failed"
        assert entry["failure_reason"].startswith("BackpropagateError:")
        assert len(env.fake.created) == 2  # run 3 never started

    def test_failure_record_error_does_not_mask_the_original_exception(
        self, env, monkeypatch, caplog
    ):
        def nan_run(sft):
            fill_adapter(sft.model, run=1, b_value=float("nan"))
            return type("R", (), {"training_loss": 1.0})()

        mrt = env.build([lambda s: _ok(s, 1), nan_run], num_runs=2)

        def boom(self, **kwargs):
            raise OSError("history locked")

        monkeypatch.setattr(RunHistoryManager, "record_run_failed", boom)

        with caplog.at_level(logging.WARNING, logger=MULTI_RUN_LOGGER):
            with pytest.raises(BackpropagateError) as exc_info:
                mrt.run(text_dataset(40))

        assert exc_info.value.code == "SLAO_MERGE_DIVERGED"
        assert any("record_run_failed failed" in r.getMessage() for r in caplog.records)

    def test_completion_write_failure_with_missing_entry_cannot_be_repaired(
        self, env, monkeypatch, caplog
    ):
        mrt = env.build(num_runs=1, merge_mode=MergeMode.SIMPLE)

        def boom(self, *a, **k):
            raise OSError("history locked")

        monkeypatch.setattr(RunHistoryManager, "record_run_completed", boom)
        monkeypatch.setattr(RunHistoryManager, "update_run", lambda self, *a, **k: None)

        with caplog.at_level(logging.INFO, logger=MULTI_RUN_LOGGER):
            result = mrt.run(text_dataset(30))

        assert result.total_runs == 1  # the successful session is not turned into an error
        assert any("record_run_completed failed" in r.getMessage() for r in caplog.records)
        assert not any("run_status_repaired" in r.getMessage() for r in caplog.records)

    def test_completion_repair_failure_is_logged_not_raised(self, env, monkeypatch, caplog):
        mrt = env.build(num_runs=1, merge_mode=MergeMode.SIMPLE)

        def boom(self, *a, **k):
            raise OSError("history locked")

        monkeypatch.setattr(RunHistoryManager, "record_run_completed", boom)
        original_update = RunHistoryManager.update_run

        def update(self, run_id, **fields):
            if fields.get("status") == "completed":
                raise OSError("still locked")
            return original_update(self, run_id, **fields)

        monkeypatch.setattr(RunHistoryManager, "update_run", update)

        with caplog.at_level(logging.WARNING, logger=MULTI_RUN_LOGGER):
            result = mrt.run(text_dataset(30))

        assert result.total_runs == 1
        assert any("Independent status flip to 'completed' also failed" in r.getMessage()
                   for r in caplog.records)


def _ok(sft, run):
    fill_adapter(sft.model, run=run)
    return type("R", (), {"training_loss": 1.0})()


class TestRunLoopStartIndex:
    def test_a_preset_start_index_without_a_checkpoint_skips_the_earlier_runs(self, env, caplog):
        """``_resume_start_run_idx`` > 1 but no checkpoint path (nothing to hydrate):
        the loop still starts at that run and never tries to load weights."""
        mrt = env.build(num_runs=3, merge_mode=MergeMode.SIMPLE)
        mrt._resume_start_run_idx = 2

        with caplog.at_level(logging.INFO, logger=MULTI_RUN_LOGGER):
            result = mrt.run(text_dataset(40))

        assert [r.run_index for r in result.runs] == [2, 3]
        assert any("Resuming multi-run from run 2/3" in r.getMessage() for r in caplog.records)
        assert not any("Resumed LoRA weights" in r.getMessage() for r in caplog.records)
