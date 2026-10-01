"""Behavioural tests for ``MultiRunTrainer._execute_run`` and its validation wrapper.

What is real: the tiny PEFT/LoRA adapter, adapter extraction + loading, the SLAO
merger and its on-disk checkpoints, ``CheckpointManager`` / ``RunHistoryManager``,
the real ``SFTConfig`` built by ``trainer._build_sft_config``, real HF datasets
and the real tiny tokenizer. What is mocked (and said so per test): the inner
per-run ``trl.SFTTrainer`` (``make_fake_sft`` -- it records what it was given and
writes known values into the real adapter instead of running gradient steps),
and the inner ``Trainer`` instance (model loading is the HF-Hub/GPU boundary).
"""

from __future__ import annotations

import logging
import math
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("trl")

from backpropagate.config import settings
from backpropagate.exceptions import BackpropagateError
from backpropagate.multi_run import MergeMode
from tests.test_multi_run_cov_support import (
    N_A,
    N_B,
    FakeInnerTrainer,
    fill_adapter,
    install_fake_sft,
    lora_params,
    make_fake_sft,
    make_trainer,
    text_dataset,
)

MULTI_RUN_LOGGER = "backpropagate.multi_run"


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    """CPU-only and deterministic, whatever GPU the dev rig has."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _b_tensors(params):
    return {k: v for k, v in params.items() if ".lora_B." in k}


def _a_tensors(params):
    return {k: v for k, v in params.items() if ".lora_A." in k}


# =============================================================================
# Happy path: two SLAO runs through the real adapter
# =============================================================================



def _cuda_available_to_multi_run_only() -> bool:
    """``torch.cuda.is_available`` that says True only to ``backpropagate.multi_run``.

    The tests reach multi_run's GPU-cleanup branches on a machine without a GPU.
    A plain ``lambda: True`` also convinces transformers' TrainingArguments, which
    then initialises the real CUDA runtime and fails ("No CUDA GPUs are
    available"); on a rig with a GPU it silently used the real card instead.
    """
    import sys

    return sys._getframe(1).f_globals.get("__name__") == "backpropagate.multi_run"

class TestExecuteRunSlaoHappyPath:
    """Mocked: ``trl.SFTTrainer`` (``make_fake_sft``) and the inner Trainer."""

    def test_run_one_trains_merges_and_checkpoints(self, tmp_path, monkeypatch):
        fake = make_fake_sft()
        install_fake_sft(monkeypatch, fake)
        mrt = make_trainer(tmp_path, steps_per_run=3, samples_per_run=8, initial_lr=2e-4)
        ds = text_dataset(40)

        result = mrt._execute_run(1, ds, tmp_path)

        # --- what the inner trainer was handed -------------------------------
        assert len(fake.created) == 1
        sft = fake.created[0]
        assert sft.args.output_dir == str(tmp_path / "run_001")
        assert sft.args.max_steps == 3
        assert sft.args.learning_rate == pytest.approx(2e-4)
        assert sft.args.per_device_train_batch_size == 2
        assert sft.args.gradient_accumulation_steps == 1
        assert sft.args.seed == settings.training.seed + 1  # + run_idx + 0 * 1000
        assert sft.args.warmup_steps == mrt.config.warmup_steps_per_run
        assert sft.model is mrt._trainer._model
        assert len(sft.train_dataset) == 8  # one fresh chunk, no replay
        # abort bridge is always installed; the step bridge only when on_step set
        assert [type(cb).__name__ for cb in sft.callbacks] == ["_AbortCallback"]

        # --- the RunResult ----------------------------------------------------
        assert result.run_index == 1
        assert result.steps == 3
        assert result.samples == 8
        assert result.final_loss == pytest.approx(1.5)  # result.training_loss wins
        assert result.loss_history == [2.5, 1.5]  # eval-only log entry skipped
        assert result.learning_rate == pytest.approx(2e-4)
        assert result.failed is False and result.failure_reason is None
        assert result.oom_retries == 0
        assert result.run_id == "abcdef0123456789"
        assert result.checkpoint_path == str(tmp_path / "run_001" / "lora")
        assert mrt._aggregate_loss == [2.5, 1.5]
        assert mrt._run_boundaries == [0]

        # --- SLAO: run 1 seeds the accumulator with the trained adapter -------
        assert result.merge_result.run_index == 1
        assert result.merge_result.a_matrices_merged == 0  # seed run merges nothing
        trained = lora_params(mrt._trainer._model)
        acc = mrt._slao_merger.get_merged_lora()
        assert set(acc) == set(trained)
        for k, v in trained.items():
            assert torch.equal(acc[k], v)

        # --- on-disk: real adapter, manifest entry, SLAO accumulator ---------
        assert (tmp_path / "run_001" / "lora" / "adapter_config.json").exists()
        assert (tmp_path / "run_001" / "slao" / "merged_lora.pt").exists()
        cp = mrt._checkpoint_manager.find_latest_for_run_id("abcdef0123456789")
        assert cp.run_index == 1 and cp.path == result.checkpoint_path
        # Trainer.save() must NOT register in its own manifest (multi-run owns it)
        path, kwargs = mrt._trainer.save_calls[0]
        assert kwargs == {"run_id": "abcdef0123456789", "register_in_manifest": False}

    def test_run_two_starts_from_slao_init_and_merges_with_time_aware_scale(
        self, tmp_path, monkeypatch
    ):
        fake = make_fake_sft()
        install_fake_sft(monkeypatch, fake)
        mrt = make_trainer(tmp_path, num_runs=2, initial_lr=2e-4, final_lr=5e-5)
        ds = text_dataset(40)

        mrt._execute_run(1, ds, tmp_path)
        run1_params = lora_params(mrt._trainer._model)
        result2 = mrt._execute_run(2, ds, tmp_path)

        # -- Before run 2 trained, the model was re-initialised from the
        #    accumulator: B copied (all 1.0), A orthogonally re-initialised.
        start = fake.created[1].start_params
        for k, v in _b_tensors(start).items():
            assert torch.equal(v, run1_params[k])
        for k, a in _a_tensors(start).items():
            assert torch.allclose(a @ a.T, torch.eye(a.shape[0]), atol=1e-5), k
            assert not torch.allclose(a, run1_params[k])  # genuinely re-initialised

        # -- After run 2 (B filled with 2.0, A random seed 1002) the accumulator is
        #    B = 1 + (1/sqrt(2)) * (2 - 1)  and A = run-2 A (hard replace).
        lam = 1.0 / math.sqrt(2.0)
        run2_params = lora_params(mrt._trainer._model)
        acc = mrt._slao_merger.get_merged_lora()
        for v in _b_tensors(acc).values():
            assert torch.allclose(v, torch.full_like(v, 1.0 + lam * (2.0 - 1.0)), atol=1e-6)
        for k, v in _a_tensors(acc).items():
            assert torch.equal(v, run2_params[k])

        mr = result2.merge_result
        assert mr.run_index == 2
        assert mr.scale_factor == pytest.approx(lam)
        assert mr.a_matrices_merged == N_A and mr.b_matrices_merged == N_B
        assert mr.new_keys_added == 0
        assert result2.learning_rate == pytest.approx(5e-5)  # end of linear decay
        # accumulator persisted for resume
        assert (tmp_path / "run_002" / "slao" / "merged_lora.pt").exists()
        # boundaries record where each run's losses start in the aggregate
        assert mrt._run_boundaries == [0, 2]
        assert mrt._aggregate_loss == [2.5, 1.5, 2.5, 1.5]

    def test_simple_mode_has_no_merge_and_keeps_current_weights(self, tmp_path, monkeypatch):
        fake = make_fake_sft()
        install_fake_sft(monkeypatch, fake)
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)
        ds = text_dataset(40)

        mrt._execute_run(1, ds, tmp_path)
        run1_params = lora_params(mrt._trainer._model)
        result2 = mrt._execute_run(2, ds, tmp_path)

        assert result2.merge_result is None
        # SIMPLE continues from the live weights: nothing re-initialised run 2
        for k, v in fake.created[1].start_params.items():
            assert torch.equal(v, run1_params[k])
        assert not (tmp_path / "run_002" / "slao").exists()
        assert (tmp_path / "run_002" / "lora" / "adapter_model.safetensors").exists()

    def test_save_every_run_false_writes_no_checkpoint(self, tmp_path, monkeypatch):
        install_fake_sft(monkeypatch, make_fake_sft())
        mrt = make_trainer(tmp_path, save_every_run=False)

        result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert result.checkpoint_path is None
        assert mrt._trainer.save_calls == []
        assert mrt._checkpoint_manager.find_latest_for_run_id("abcdef0123456789") is None
        assert not (tmp_path / "run_001").exists()

    def test_final_loss_falls_back_to_last_logged_loss(self, tmp_path, monkeypatch):
        """``train()`` returned an object without ``training_loss``."""

        def train(sft):
            fill_adapter(sft.model, run=1)
            sft.state.log_history = [{"loss": 4.0}, {"loss": 3.25}]
            return object()

        install_fake_sft(monkeypatch, make_fake_sft([train]))
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)

        result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert result.final_loss == 3.25
        assert result.loss_history == [4.0, 3.25]

    def test_final_loss_is_zero_with_no_loss_signal_at_all(self, tmp_path, monkeypatch):
        def train(sft):
            fill_adapter(sft.model, run=1)
            return object()  # no training_loss and an empty log history

        install_fake_sft(monkeypatch, make_fake_sft([train]))
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)

        result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert result.final_loss == 0.0
        assert result.loss_history == []


# =============================================================================
# Callbacks, run naming, platform branches
# =============================================================================


class TestExecuteRunWiring:
    def test_on_run_start_fires_with_the_run_index(self, tmp_path, monkeypatch):
        install_fake_sft(monkeypatch, make_fake_sft())
        seen = []
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)
        mrt.on_run_start = seen.append

        mrt._execute_run(2, text_dataset(40), tmp_path)

        assert seen == [2]

    def test_on_step_installs_the_step_bridge_after_the_abort_bridge(self, tmp_path, monkeypatch):
        fake = make_fake_sft()
        install_fake_sft(monkeypatch, fake)
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)
        mrt.on_step = lambda run, step, loss: None

        mrt._execute_run(1, text_dataset(40), tmp_path)

        names = [type(cb).__name__ for cb in fake.created[0].callbacks]
        assert names == ["_AbortCallback", "_StepCallback"]

    def test_run_name_is_set_only_when_experiment_tracking_is_on(self, tmp_path, monkeypatch):
        fake = make_fake_sft()
        install_fake_sft(monkeypatch, fake)
        inner = FakeInnerTrainer(report_to="wandb")
        mrt = make_trainer(tmp_path, inner=inner, merge_mode=MergeMode.SIMPLE)

        mrt._execute_run(3, text_dataset(40), tmp_path)

        assert fake.created[0].args.run_name == "backprop-abcdef012345-run-003"
        assert fake.created[0].args.report_to == ["wandb"]

    @pytest.mark.parametrize("pre_tokenize", [True, False])
    def test_pre_tokenize_only_on_windows_with_the_setting_on(
        self, tmp_path, monkeypatch, pre_tokenize
    ):
        monkeypatch.setattr(settings.windows, "pre_tokenize", pre_tokenize)
        install_fake_sft(monkeypatch, make_fake_sft())
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)

        mrt._execute_run(1, text_dataset(40), tmp_path)

        expected_calls = 1 if (os.name == "nt" and pre_tokenize) else 0
        assert len(mrt._trainer.pre_tokenize_calls) == expected_calls

    def test_vram_cleanup_branch_runs_when_cuda_is_reported_available(
        self, tmp_path, monkeypatch
    ):
        """Mocked: the CUDA runtime (``is_available`` / ``empty_cache`` /
        ``memory_allocated``) -- there is no GPU on CI."""
        emptied = []
        monkeypatch.setattr(torch.cuda, "is_available", _cuda_available_to_multi_run_only)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 2_000_000_000)
        install_fake_sft(monkeypatch, make_fake_sft())
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)

        result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert result.failed is False
        assert emptied  # allocator cache reclaimed after the run

    def test_gpu_peaks_are_reset_before_the_run(self, tmp_path, monkeypatch):
        install_fake_sft(monkeypatch, make_fake_sft())
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)
        mrt._gpu_max_temp = 99.0
        mrt._gpu_max_vram = 99.0

        result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert result.gpu_max_temp == 0.0 and result.gpu_max_vram_percent == 0.0


# =============================================================================
# OOM recovery
# =============================================================================


def _oom(_sft):
    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


class TestExecuteRunOomRecovery:
    """Mocked: ``SFTTrainer.train`` raising RuntimeError OOM strings (no GPU)."""

    def test_oom_halves_batch_doubles_accumulation_and_retries(self, tmp_path, monkeypatch):
        fake = make_fake_sft([_oom])
        install_fake_sft(monkeypatch, fake)
        inner = FakeInnerTrainer(batch_size=4, grad_accum=2)
        mrt = make_trainer(tmp_path, inner=inner, merge_mode=MergeMode.SIMPLE)

        result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert len(fake.created) == 2  # a fresh SFTTrainer per attempt
        first, second = (s.args for s in fake.created)
        assert (first.per_device_train_batch_size, first.gradient_accumulation_steps) == (4, 2)
        assert (second.per_device_train_batch_size, second.gradient_accumulation_steps) == (2, 4)
        # effective batch size (batch * accum) preserved; seed shifted by 1000
        assert 2 * 4 == 4 * 2
        assert second.seed == first.seed + 1000
        assert inner.batch_size == 2 and inner.gradient_accumulation == 4
        assert result.failed is False
        assert result.oom_retries == 1
        assert mrt._session_oom_count == 1
        assert mrt._oom_consecutive_at_min_batch == 0

    def test_cublas_alloc_failure_is_treated_as_oom_with_a_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        def cublas(_sft):
            raise RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate")

        fake = make_fake_sft([cublas])
        install_fake_sft(monkeypatch, fake)
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)

        with caplog.at_level(logging.WARNING, logger=MULTI_RUN_LOGGER):
            result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert len(fake.created) == 2
        assert result.oom_retries == 1
        assert any("RUNTIME_OOM_ADJACENT" in r.getMessage() for r in caplog.records)

    def test_grad_accumulation_ceiling_aborts_instead_of_retrying(self, tmp_path, monkeypatch):
        fake = make_fake_sft([_oom])
        install_fake_sft(monkeypatch, fake)
        mrt = make_trainer(
            tmp_path, inner=FakeInnerTrainer(batch_size=2, grad_accum=1), max_grad_accumulation=1
        )

        with pytest.raises(BackpropagateError) as exc_info:
            mrt._execute_run(1, text_dataset(40), tmp_path)

        err = exc_info.value
        assert err.code == "RUNTIME_OOM_RECOVERY_EXHAUSTED"
        assert err.details["attempted_grad_accumulation"] == 2
        assert err.details["max_grad_accumulation"] == 1
        assert err.details["current_batch_size"] == 2
        assert len(fake.created) == 1  # no retry
        assert isinstance(err.__cause__, RuntimeError)
        assert mrt._trainer.batch_size == 2  # operator's pinned values untouched

    def test_floor_ooms_fail_runs_then_abort_the_session(self, tmp_path, monkeypatch):
        fake = make_fake_sft([_oom, _oom, _oom])
        install_fake_sft(monkeypatch, fake)
        inner = FakeInnerTrainer(batch_size=1, grad_accum=1)
        mrt = make_trainer(tmp_path, inner=inner, merge_mode=MergeMode.SIMPLE)
        ds = text_dataset(40)

        for n in (1, 2):
            r = mrt._execute_run(n, ds, tmp_path)
            assert r.failed is True and "out of memory" in r.failure_reason
            assert r.checkpoint_path is None  # failed runs are never registered
            assert r.oom_retries == 1
            assert mrt._oom_consecutive_at_min_batch == n
        assert inner.save_calls == []
        assert mrt._checkpoint_manager.find_latest_for_run_id("abcdef0123456789") is None

        with pytest.raises(BackpropagateError) as exc_info:
            mrt._execute_run(3, ds, tmp_path)

        err = exc_info.value
        assert err.code == "RUNTIME_OOM_RECOVERY_EXHAUSTED"
        assert err.details["consecutive_oom_at_min_batch"] == 3
        assert err.details["run_index"] == 3
        assert mrt._session_oom_count == 3

    def test_without_recovery_oom_is_recorded_with_the_runtime_gpu_oom_code(
        self, tmp_path, monkeypatch
    ):
        fake = make_fake_sft([_oom])
        install_fake_sft(monkeypatch, fake)
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)
        mrt.oom_recovery = False

        result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert len(fake.created) == 1  # no retry
        assert result.failed is True
        assert result.failure_reason.startswith("RUNTIME_GPU_OOM: RuntimeError:")
        assert result.oom_retries == 0
        assert mrt._trainer.batch_size == 2  # untouched

    def test_non_oom_failure_keeps_partial_losses_and_skips_merge_and_save(
        self, tmp_path, monkeypatch
    ):
        def explode(sft):
            sft.state.log_history = [{"loss": 3.0}, {"loss": 2.0}]
            raise ValueError("bad batch")

        install_fake_sft(monkeypatch, make_fake_sft([explode]))
        mrt = make_trainer(tmp_path)  # SLAO mode

        result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert result.failed is True
        assert result.failure_reason == "ValueError: bad batch"  # no OOM prefix
        assert result.loss_history == [3.0, 2.0]  # partial losses are salvaged...
        assert mrt._aggregate_loss == [3.0, 2.0]  # ...into the session aggregate too
        assert result.final_loss == 2.0
        assert result.merge_result is None
        assert mrt._slao_merger.get_merged_lora() is None  # nothing merged
        assert mrt._trainer.save_calls == []

    def test_non_oom_failure_with_empty_log_history_has_zero_loss(self, tmp_path, monkeypatch):
        def explode(_sft):
            raise ValueError("early")

        install_fake_sft(monkeypatch, make_fake_sft([explode]))
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)

        result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert result.failed is True and result.final_loss == 0.0
        assert mrt._aggregate_loss == []

    @pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
    def test_interrupts_are_never_swallowed(self, tmp_path, monkeypatch, exc_type):
        def interrupt(_sft):
            raise exc_type()

        install_fake_sft(monkeypatch, make_fake_sft([interrupt]))
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)

        with pytest.raises(exc_type):
            mrt._execute_run(1, text_dataset(40), tmp_path)


# =============================================================================
# Merge outcomes: divergence, drift gate, save failures
# =============================================================================


class TestExecuteRunMergeOutcomes:
    def test_diverged_merge_propagates_and_is_not_retried_as_oom(self, tmp_path, monkeypatch):
        """Run 2 trains to NaN B matrices: the SLAO finite scan raises, and the
        error escapes ``_execute_run`` (no batch-halving retry)."""
        install_fake_sft(monkeypatch, make_fake_sft())  # run 1 default
        mrt = make_trainer(tmp_path)
        ds = text_dataset(40)
        mrt._execute_run(1, ds, tmp_path)

        def nan_run(sft):
            fill_adapter(sft.model, run=2, b_value=float("nan"))
            return type("R", (), {"training_loss": 1.0})()

        fake2 = make_fake_sft([nan_run])
        install_fake_sft(monkeypatch, fake2)
        with pytest.raises(BackpropagateError) as exc_info:
            mrt._execute_run(2, ds, tmp_path)

        assert exc_info.value.code == "SLAO_MERGE_DIVERGED"
        assert exc_info.value.details["run_id"] == "abcdef0123456789"
        assert len(fake2.created) == 1  # exactly one training attempt
        assert mrt._trainer.batch_size == 2  # OOM recovery was never entered

    def test_drift_gate_branches_a_run_and_leaves_the_accumulator_alone(
        self, tmp_path, monkeypatch
    ):
        """Run 1 B=+1, run 2 B=-1: cosine similarity -1 < threshold 0.5, so run 2
        is kept as a sibling instead of merged."""

        def run_two(sft):
            fill_adapter(sft.model, run=2, b_value=-1.0)
            return type("R", (), {"training_loss": 1.0})()

        install_fake_sft(monkeypatch, make_fake_sft([lambda s: _train_b(s, 1.0), run_two]))
        mrt = make_trainer(tmp_path, drift_gate=True, drift_threshold=0.5)
        ds = text_dataset(40)

        mrt._execute_run(1, ds, tmp_path)
        before = {k: v.clone() for k, v in mrt._slao_merger.get_merged_lora().items()}
        result2 = mrt._execute_run(2, ds, tmp_path)

        assert result2.branched is True
        assert mrt._branched_runs == [2]
        assert result2.merge_result.branched is True
        assert result2.merge_result.task_similarity == pytest.approx(-1.0, abs=1e-6)
        assert mrt._slao_merger.run_index == 1  # not advanced
        for k, v in mrt._slao_merger.get_merged_lora().items():
            assert torch.equal(v, before[k])
        # the sibling is still saved on disk
        assert (tmp_path / "run_002" / "lora" / "adapter_config.json").exists()

    def test_checkpoint_save_failure_is_logged_and_run_still_succeeds(
        self, tmp_path, monkeypatch, caplog
    ):
        """Mocked: the filesystem write inside ``Trainer.save`` (raises OSError)."""
        install_fake_sft(monkeypatch, make_fake_sft())
        inner = FakeInnerTrainer()
        inner.fail_save_with = OSError("No space left on device")
        mrt = make_trainer(tmp_path, inner=inner)

        with caplog.at_level(logging.ERROR, logger=MULTI_RUN_LOGGER):
            result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert result.failed is False
        assert result.checkpoint_path is None
        assert result.merge_result is not None  # the merge itself succeeded
        assert any("Failed to save checkpoint for run 1" in r.getMessage() for r in caplog.records)
        assert mrt._checkpoint_manager.find_latest_for_run_id("abcdef0123456789") is None

    def test_a_merge_exception_is_logged_with_context_and_reraised(
        self, tmp_path, monkeypatch, caplog
    ):
        def nan_run(sft):
            fill_adapter(sft.model, run=1, b_value=1.0)
            with torch.no_grad():
                for n, p in sft.model.named_parameters():
                    if ".lora_A." in n:
                        p.fill_(float("inf"))
            return type("R", (), {"training_loss": 1.0})()

        install_fake_sft(monkeypatch, make_fake_sft([nan_run]))
        mrt = make_trainer(tmp_path)
        # Seed an accumulator so run 1 is treated as a real merge, not a seed.
        mrt._slao_merger.initialize(lora_params(mrt._trainer._model))

        with caplog.at_level(logging.ERROR, logger=MULTI_RUN_LOGGER):
            with pytest.raises(BackpropagateError):
                mrt._execute_run(1, text_dataset(40), tmp_path)

        assert any("merge_failed run_id=abcdef0123456789 run_index=1" in r.getMessage()
                   for r in caplog.records)


def _train_b(sft, b_value):
    fill_adapter(sft.model, run=1, b_value=b_value)
    return type("R", (), {"training_loss": 1.0})()


# =============================================================================
# _execute_run_with_validation (real validation loss on the tiny model)
# =============================================================================


class TestExecuteRunWithValidation:
    """Validation loss is computed for real: forward passes of the tiny Llama
    over a reserved held-out slice. Mocked: ``SFTTrainer`` only."""

    def test_validation_loss_is_real_finite_and_backfilled_into_the_manifest(
        self, tmp_path, monkeypatch
    ):
        install_fake_sft(monkeypatch, make_fake_sft())
        mrt = make_trainer(
            tmp_path, validate_every_run=True, validation_samples=3, samples_per_run=8,
            merge_mode=MergeMode.SIMPLE,
        )
        ds = text_dataset(40)

        result, val_loss = mrt._execute_run_with_validation(1, ds, tmp_path)

        assert val_loss is not None and math.isfinite(val_loss) and val_loss > 0
        assert result.validation_loss == val_loss
        # Untrained random Llama over a 64-way vocab: loss ~ ln(64) ~ 4.16
        assert 2.0 < val_loss < 7.0
        cp = mrt._checkpoint_manager.find_latest_for_run_id("abcdef0123456789")
        assert cp.validation_loss == pytest.approx(val_loss)
        # the model was returned to training mode by the validator
        assert mrt._trainer._model.training is True

    def test_failed_run_skips_validation(self, tmp_path, monkeypatch, caplog):
        def explode(_sft):
            raise ValueError("boom")

        install_fake_sft(monkeypatch, make_fake_sft([explode]))
        mrt = make_trainer(tmp_path, validate_every_run=True, merge_mode=MergeMode.SIMPLE)
        called = []
        monkeypatch.setattr(mrt, "_compute_validation_loss", lambda *a: called.append(a) or 1.0)

        with caplog.at_level(logging.WARNING, logger=MULTI_RUN_LOGGER):
            result, val_loss = mrt._execute_run_with_validation(1, text_dataset(40), tmp_path)

        assert result.failed is True
        assert val_loss is None and result.validation_loss is None
        assert called == []  # the indeterminate model was never evaluated
        assert any("skipping validation loss computation" in r.getMessage() for r in caplog.records)

    def test_non_finite_validation_loss_is_recorded_but_not_propagated(
        self, tmp_path, monkeypatch, caplog
    ):
        install_fake_sft(monkeypatch, make_fake_sft())
        mrt = make_trainer(tmp_path, validate_every_run=True, merge_mode=MergeMode.SIMPLE)
        monkeypatch.setattr(mrt, "_compute_validation_loss", lambda *a: float("inf"))

        with caplog.at_level(logging.WARNING, logger=MULTI_RUN_LOGGER):
            result, val_loss = mrt._execute_run_with_validation(1, text_dataset(40), tmp_path)

        assert val_loss is None  # not fed into early-stop / best-checkpoint scoring
        assert result.validation_loss == float("inf")  # but visible in the record
        cp = mrt._checkpoint_manager.find_latest_for_run_id("abcdef0123456789")
        assert cp.validation_loss is None
        assert any("non-finite" in r.getMessage() for r in caplog.records)

    def test_validation_disabled_returns_no_loss(self, tmp_path, monkeypatch):
        install_fake_sft(monkeypatch, make_fake_sft())
        mrt = make_trainer(tmp_path, validate_every_run=False, merge_mode=MergeMode.SIMPLE)

        result, val_loss = mrt._execute_run_with_validation(1, text_dataset(40), tmp_path)

        assert val_loss is None and result.validation_loss is None

    def test_missing_manifest_entry_is_logged_not_fatal(self, tmp_path, monkeypatch, caplog):
        """With ``save_every_run=False`` no manifest entry exists to receive the
        validation loss; the metric is dropped with a DEBUG breadcrumb."""
        install_fake_sft(monkeypatch, make_fake_sft())
        mrt = make_trainer(
            tmp_path, validate_every_run=True, save_every_run=False, merge_mode=MergeMode.SIMPLE
        )
        monkeypatch.setattr(mrt, "_compute_validation_loss", lambda *a: 2.5)

        with caplog.at_level(logging.DEBUG, logger=MULTI_RUN_LOGGER):
            result, val_loss = mrt._execute_run_with_validation(1, text_dataset(40), tmp_path)

        assert val_loss == 2.5 and result.validation_loss == 2.5
        assert any("was not backfilled into any manifest entry" in r.getMessage()
                   for r in caplog.records)

    def test_no_checkpoint_manager_skips_the_manifest_backfill(self, tmp_path, monkeypatch):
        install_fake_sft(monkeypatch, make_fake_sft())
        mrt = make_trainer(
            tmp_path, validate_every_run=True, save_every_run=False, merge_mode=MergeMode.SIMPLE
        )
        mrt._checkpoint_manager = None
        monkeypatch.setattr(mrt, "_compute_validation_loss", lambda *a: 1.25)

        result, val_loss = mrt._execute_run_with_validation(1, text_dataset(40), tmp_path)

        assert val_loss == 1.25 and result.validation_loss == 1.25

    def test_unlocked_manifest_write_still_patches_the_in_memory_entry(
        self, tmp_path, monkeypatch
    ):
        """If the cross-process lock is unavailable (``locked=False``) the entry is
        patched from the in-memory snapshot rather than a fresh disk read."""
        from contextlib import contextmanager

        install_fake_sft(monkeypatch, make_fake_sft())
        mrt = make_trainer(tmp_path, validate_every_run=True, merge_mode=MergeMode.SIMPLE)
        monkeypatch.setattr(mrt, "_compute_validation_loss", lambda *a: 0.75)

        @contextmanager
        def no_lock(_operation):
            yield False

        monkeypatch.setattr(mrt._checkpoint_manager, "_locked_manifest_write", no_lock)

        mrt._execute_run_with_validation(1, text_dataset(40), tmp_path)

        entry = mrt._checkpoint_manager._checkpoints[0]
        assert entry.run_index == 1 and entry.validation_loss == 0.75

    def test_already_scored_entry_is_skipped_when_backfilling(self, tmp_path, monkeypatch):
        """Entries for a different run, or already carrying a loss, are left alone."""
        install_fake_sft(monkeypatch, make_fake_sft())
        mrt = make_trainer(tmp_path, validate_every_run=True, merge_mode=MergeMode.SIMPLE)
        monkeypatch.setattr(mrt, "_compute_validation_loss", lambda *a: 0.5)
        mgr = mrt._checkpoint_manager
        (tmp_path / "x").mkdir()
        mgr.register(run_index=2, checkpoint_path=str(tmp_path / "x"), validation_loss=None,
                     training_loss=1.0, is_run_boundary=False, protected=True,
                     run_id="abcdef0123456789")

        mrt._execute_run_with_validation(1, text_dataset(40), tmp_path)

        by_run = {c.run_index: c for c in mgr._checkpoints}
        assert by_run[1].validation_loss == 0.5
        assert by_run[2].validation_loss is None


class TestExecuteRunDefensivePaths:
    def test_cache_reclaim_failure_during_oom_recovery_is_non_fatal(self, tmp_path, monkeypatch):
        """Mocked: the CUDA runtime -- reported available, but the first ``empty_cache`` raises."""
        emptied = []

        def broken_empty_cache():
            emptied.append(1)
            if len(emptied) == 1:  # the OOM-recovery reclaim; the end-of-run cleanup works
                raise RuntimeError("CUDA context lost")

        monkeypatch.setattr(torch.cuda, "is_available", _cuda_available_to_multi_run_only)
        monkeypatch.setattr(torch.cuda, "empty_cache", broken_empty_cache)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
        fake = make_fake_sft([_oom])
        install_fake_sft(monkeypatch, fake)
        mrt = make_trainer(tmp_path, merge_mode=MergeMode.SIMPLE)

        result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert emptied  # the reclaim was attempted ...
        assert result.failed is False and result.oom_retries == 1  # ... and the retry went on
        assert len(fake.created) == 2

    def test_run_is_checkpointed_even_without_a_checkpoint_manager(self, tmp_path, monkeypatch):
        install_fake_sft(monkeypatch, make_fake_sft())
        mrt = make_trainer(tmp_path)
        mrt._checkpoint_manager = None

        result = mrt._execute_run(1, text_dataset(40), tmp_path)

        assert result.checkpoint_path == str(tmp_path / "run_001" / "lora")
        assert (tmp_path / "run_001" / "lora" / "adapter_config.json").exists()
        assert (tmp_path / "run_001" / "slao" / "merged_lora.pt").exists()
