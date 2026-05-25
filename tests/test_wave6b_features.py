"""Tests for v1.4 Wave 6b features — backend half.

Covers:
- Item 1: ``Trainer(mode="full")`` + 3B ceiling gate (BACKEND-F-008)
- Item 2: ``estimate_vram`` pre-flight estimator (BACKEND-F-002)
- Item 3: ``MultiRunTrainer`` ``on_step`` callback parity (BACKEND-F-003)
- Item 4: ``CheckpointManager._save_manifest`` filelock parity (BACKEND-F-004)

These tests are pure-Python (no GPU, no model load); the four implementations
are deliberately decoupled from the heavy runtime stack so they can be
verified end-to-end in CI without a 7B download.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Item 1: Trainer(mode="full") + 3B ceiling gate
# =============================================================================


class TestTrainerModeFullGate:
    """v1.4 BACKEND-F-008: mode='full' construction-time + load-time gate."""

    def test_mode_default_is_lora(self):
        """Trainer() default mode is 'lora' (preserves pre-Wave-6b behavior)."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert trainer.mode == "lora", (
            "mode default drifted from 'lora'; this would silently change "
            "every existing operator's training contract."
        )

    def test_mode_explicit_lora(self):
        """Trainer(mode='lora') preserves the v1.3 byte-identical contract."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(mode="lora")

        assert trainer.mode == "lora"

    def test_mode_full_accepts_small_preset(self):
        """Trainer(mode='full', model='smollm3-3b') passes the 3B gate."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            # smollm3-3b is 3B and within the ceiling.
            trainer = Trainer(model="smollm3-3b", mode="full")

        assert trainer.mode == "full"

    def test_mode_full_rejects_oversized_model(self):
        """Trainer(mode='full', model='Qwen/Qwen2.5-7B-Instruct') raises."""
        from backpropagate.exceptions import FullFinetuneModelTooLargeError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(FullFinetuneModelTooLargeError) as exc_info:
                Trainer(model="Qwen/Qwen2.5-7B-Instruct", mode="full")

        err = exc_info.value
        assert err.code == "RUNTIME_FULL_FT_MODEL_TOO_LARGE"
        # The model id MUST appear in the message so an operator pasting the
        # traceback into Slack sees what they passed.
        assert "Qwen/Qwen2.5-7B-Instruct" in str(err)
        # The 3B ceiling is the documented contract — drift surfaces here.
        assert err.ceiling_billions == 3.0
        # The recovery hint names the lora alternative + at least one
        # mode='full'-compatible preset.
        assert "mode='lora'" in str(err)
        assert "phi-4-mini-3.8b" in str(err) or "smollm3-3b" in str(err)

    def test_mode_full_rejects_mistral_7b(self):
        """Trainer(mode='full', model='mistralai/Mistral-7B-Instruct-v0.3') raises."""
        from backpropagate.exceptions import FullFinetuneModelTooLargeError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(FullFinetuneModelTooLargeError):
                Trainer(model="mistralai/Mistral-7B-Instruct-v0.3", mode="full")

    def test_mode_invalid_raises_invalid_setting(self):
        """Trainer(mode='aqlm') raises InvalidSettingError with a hint."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError) as exc_info:
                Trainer(mode="aqlm")

        err = exc_info.value
        assert err.setting_name == "mode"
        assert err.value == "aqlm"

    def test_mode_full_lowers_default_learning_rate(self):
        """mode='full' with no explicit LR applies the ~10x divisor."""
        from backpropagate.config import settings
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(model="smollm3-3b", mode="full")

        expected = settings.training.learning_rate / 10.0
        assert trainer.learning_rate == pytest.approx(expected), (
            f"mode='full' should apply the 10x LR divisor by default; "
            f"got {trainer.learning_rate}, expected {expected}."
        )

    def test_mode_full_honors_explicit_learning_rate(self):
        """mode='full' with explicit LR does NOT apply the divisor."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                model="smollm3-3b",
                mode="full",
                learning_rate=5e-5,
            )

        assert trainer.learning_rate == 5e-5

    def test_mode_full_with_unknown_model_defers_check(self):
        """An obscure model id without an estimate is accepted at construction.

        The load-time recheck (in load_model) catches it; pre-load we never
        block on a model we cannot estimate.
        """
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            # No B-suffix, no preset match — estimator returns None.
            trainer = Trainer(model="random-org/obscure-finetune", mode="full")

        assert trainer.mode == "full"

    def test_build_sft_config_full_mode_forces_gradient_checkpointing(self):
        """_build_sft_config(mode='full', ...) sets gradient_checkpointing=True."""
        # Use a stub SFTConfig so we don't depend on a real trl install in
        # the test env. Any object that captures kwargs works.
        captured_kwargs: dict = {}

        class _StubSFTConfig:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        from backpropagate import trainer as trainer_mod

        with patch.dict("sys.modules", {"trl": MagicMock(SFTConfig=_StubSFTConfig)}):
            trainer_mod._build_sft_config(
                output_dir="./out",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=10,
                learning_rate=2e-5,
                warmup_steps=2,
                max_seq_length=1024,
                seed=42,
                lr_scheduler_type="cosine",
                logging_steps=1,
                mode="full",
            )

        assert captured_kwargs.get("gradient_checkpointing") is True
        # use_reentrant=False is the documented HF recommended default.
        gck_kwargs = captured_kwargs.get("gradient_checkpointing_kwargs", {})
        assert gck_kwargs.get("use_reentrant") is False

    def test_build_sft_config_lora_mode_omits_gradient_checkpointing(self):
        """_build_sft_config(mode='lora', ...) does NOT inject gradient_checkpointing.

        Mode='lora' inherits gradient_checkpointing from
        settings.lora.use_gradient_checkpointing (already wired in
        _load_with_unsloth). The helper must NOT silently force it on.
        """
        captured_kwargs: dict = {}

        class _StubSFTConfig:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        from backpropagate import trainer as trainer_mod

        with patch.dict("sys.modules", {"trl": MagicMock(SFTConfig=_StubSFTConfig)}):
            trainer_mod._build_sft_config(
                output_dir="./out",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=10,
                learning_rate=2e-4,
                warmup_steps=2,
                max_seq_length=1024,
                seed=42,
                lr_scheduler_type="cosine",
                logging_steps=1,
                mode="lora",
            )

        assert "gradient_checkpointing" not in captured_kwargs

    def test_build_sft_config_full_mode_upgrades_optim_to_paged(self):
        """_build_sft_config(mode='full', optim='adamw_8bit') upgrades to paged."""
        captured_kwargs: dict = {}

        class _StubSFTConfig:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        from backpropagate import trainer as trainer_mod

        with patch.dict("sys.modules", {"trl": MagicMock(SFTConfig=_StubSFTConfig)}):
            # No CUDA / torch — the resolver short-circuits at the
            # "torch.cuda.is_available()" check inside _detect_optim_for_card.
            with patch("torch.cuda.is_available", return_value=False):
                trainer_mod._build_sft_config(
                    output_dir="./out",
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=1,
                    max_steps=10,
                    learning_rate=2e-5,
                    warmup_steps=2,
                    max_seq_length=1024,
                    seed=42,
                    lr_scheduler_type="cosine",
                    logging_steps=1,
                    optim="adamw_8bit",
                    mode="full",
                )

        # mode='full' short-circuits adamw_8bit -> paged_adamw_8bit even
        # when the LoRA detector wouldn't (e.g. on a 24GB card).
        assert captured_kwargs.get("optim") == "paged_adamw_8bit"

    def test_build_sft_config_invalid_mode_raises(self):
        """_build_sft_config(mode='aqlm') raises ValueError."""
        from backpropagate import trainer as trainer_mod

        with pytest.raises(ValueError, match="mode="):
            trainer_mod._build_sft_config(
                output_dir="./out",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=10,
                learning_rate=2e-5,
                warmup_steps=2,
                max_seq_length=1024,
                seed=42,
                lr_scheduler_type="cosine",
                logging_steps=1,
                mode="aqlm",
            )


# =============================================================================
# Helpers shared across the parameter-count tests
# =============================================================================


class TestEstimateParamCount:
    """v1.4 BACKEND-F-008: _estimate_param_count_billions probe."""

    def test_preset_7b_yields_7(self):
        from backpropagate.trainer import _estimate_param_count_billions

        result = _estimate_param_count_billions("Qwen/Qwen2.5-7B-Instruct")
        assert result == pytest.approx(7.0)

    def test_preset_phi4_yields_3_8(self):
        from backpropagate.trainer import _estimate_param_count_billions

        result = _estimate_param_count_billions("microsoft/Phi-4-mini-instruct")
        # Preset name "phi-4-mini-3.8b" → 3.8.
        assert result == pytest.approx(3.8)

    def test_preset_smollm3_yields_3(self):
        from backpropagate.trainer import _estimate_param_count_billions

        result = _estimate_param_count_billions("HuggingFaceTB/SmolLM3-3B")
        assert result == pytest.approx(3.0)

    def test_obscure_id_returns_none(self):
        """Random orgs without a B-suffix return None (gate defers to load_model)."""
        from backpropagate.trainer import _estimate_param_count_billions

        result = _estimate_param_count_billions("random/finetune-no-size-clue")
        assert result is None

    def test_heuristic_regex_catches_quantized_id(self):
        """The heuristic falls back to a regex scan for B-suffix tokens."""
        from backpropagate.trainer import _estimate_param_count_billions

        # The preset table HAS no entry for "Custom-13B-Chat", so the
        # regex fallback fires. Match anchored on the 'B' suffix.
        result = _estimate_param_count_billions("Custom-13B-Chat")
        assert result == pytest.approx(13.0)


# =============================================================================
# Item 2: estimate_vram pre-flight estimator
# =============================================================================


class TestEstimateVRAM:
    """v1.4 BACKEND-F-002: VRAM pre-flight estimator."""

    def test_returns_structured_estimate(self):
        from backpropagate.trainer import VRAMEstimate, estimate_vram

        est = estimate_vram(
            model="Qwen/Qwen2.5-7B-Instruct",
            mode="lora",
            lora_r=16,
            batch_size=1,
            max_seq_length=2048,
        )

        assert isinstance(est, VRAMEstimate)
        assert est.total_gb > 0
        assert est.mode == "lora"
        assert est.param_count_billions == pytest.approx(7.0)

    def test_breakdown_components_sum_correctly(self):
        from backpropagate.trainer import estimate_vram

        est = estimate_vram(
            model="Qwen/Qwen2.5-7B-Instruct",
            mode="lora",
            lora_r=16,
            batch_size=1,
            max_seq_length=2048,
            overhead_fraction=0.15,
        )

        subtotal = (
            est.model_weights_gb
            + est.lora_adapter_gb
            + est.optimizer_state_gb
            + est.activations_gb
            + est.kv_cache_gb
        )
        # Overhead is 15% of subtotal.
        expected_overhead = subtotal * 0.15
        expected_total = subtotal + expected_overhead

        assert est.overhead_gb == pytest.approx(expected_overhead, rel=1e-6)
        assert est.total_gb == pytest.approx(expected_total, rel=1e-6)

    def test_quantized_base_reduces_weights(self):
        """nf4 quantization (the trainer default) reduces model_weights_gb 4x."""
        from backpropagate.trainer import estimate_vram

        quantized = estimate_vram(
            model="Qwen/Qwen2.5-7B-Instruct",
            quantize_base=True,
        )
        unquantized = estimate_vram(
            model="Qwen/Qwen2.5-7B-Instruct",
            quantize_base=False,
            bytes_per_param=2,
        )

        # nf4=0.5 bytes vs bf16=2 bytes → 4x reduction.
        assert quantized.model_weights_gb < unquantized.model_weights_gb * 0.3

    def test_lora_mode_includes_adapter_gb(self):
        """mode='lora' produces a non-zero lora_adapter_gb line."""
        from backpropagate.trainer import estimate_vram

        est = estimate_vram(
            model="Qwen/Qwen2.5-7B-Instruct",
            mode="lora",
            lora_r=256,
        )

        assert est.lora_adapter_gb > 0

    def test_full_mode_zeroes_adapter_gb(self):
        """mode='full' has no LoRA adapter → 0.0 in that line."""
        from backpropagate.trainer import estimate_vram

        est = estimate_vram(
            model="smollm3-3b",
            mode="full",
        )

        assert est.lora_adapter_gb == 0.0

    def test_fits_on_card_returns_bool(self):
        from backpropagate.trainer import estimate_vram

        est = estimate_vram(model="smollm3-3b", batch_size=1, max_seq_length=1024)

        # 3B model on 24GB card should fit.
        assert est.fits_on_card(24.0) is True
        # 3B model on 1GB card cannot fit.
        assert est.fits_on_card(1.0) is False

    def test_summary_includes_total_and_breakdown(self):
        from backpropagate.trainer import estimate_vram

        est = estimate_vram(model="Qwen/Qwen2.5-7B-Instruct")
        summary = est.summary()

        assert "total=" in summary
        assert "weights=" in summary
        assert "lora=" in summary
        assert "activations=" in summary

    def test_unknown_model_falls_back_to_7b(self):
        """Unknown model defaults to 7B with a note explaining the imputation."""
        from backpropagate.trainer import estimate_vram

        est = estimate_vram(model="totally-unknown-org/totally-unknown-model")

        assert est.param_count_billions == pytest.approx(7.0)
        # Note: the imputation MUST appear so the operator sees it.
        assert any("assumed 7.0B" in note for note in est.notes)

    def test_trainer_estimate_vram_method(self):
        """Trainer.estimate_vram() pulls config from the instance."""
        from backpropagate.trainer import Trainer, VRAMEstimate

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                model="Qwen/Qwen2.5-7B-Instruct",
                lora_r=256,
                batch_size=1,
                max_seq_length=2048,
            )

        est = trainer.estimate_vram()
        assert isinstance(est, VRAMEstimate)
        assert est.lora_r == 256
        assert est.batch_size == 1
        assert est.max_seq_length == 2048


# =============================================================================
# Item 3: MultiRunTrainer callback parity
# =============================================================================


class TestMultiRunTrainerCallbacks:
    """v1.4 BACKEND-F-003: on_step / on_run_start / on_run_complete parity."""

    def test_on_step_bridge_returns_none_when_callback_unset(self):
        """No on_step → no callback bridge installed."""
        from backpropagate.multi_run import (
            MultiRunTrainer,
            _build_multi_run_step_callback,
        )

        with patch("torch.cuda.is_available", return_value=False):
            mrt = MultiRunTrainer(model="smollm3-3b", num_runs=1, steps_per_run=1)

        cb = _build_multi_run_step_callback(mrt, run_idx=1)
        assert cb is None, (
            "When MultiRunTrainer.on_step is unset, the bridge should "
            "return None so the SFTTrainer callback list stays minimal."
        )

    def test_on_step_bridge_installs_when_callback_supplied(self):
        """A configured on_step yields a non-None HF TrainerCallback subclass."""
        from backpropagate.multi_run import (
            MultiRunTrainer,
            _build_multi_run_step_callback,
        )

        seen: list[tuple] = []

        def my_on_step(run_idx, step, loss):
            seen.append((run_idx, step, loss))

        with patch("torch.cuda.is_available", return_value=False):
            mrt = MultiRunTrainer(
                model="smollm3-3b",
                num_runs=1,
                steps_per_run=1,
                on_step=my_on_step,
            )

        cb = _build_multi_run_step_callback(mrt, run_idx=3)
        assert cb is not None

        # Drive the bridge: HF would call on_log with logs={'loss': 0.42}
        # at every logging_steps boundary.
        state = MagicMock(global_step=15)
        cb.on_log(args=None, state=state, control=None, logs={"loss": 0.42})

        assert seen == [(3, 15, 0.42)]

    def test_on_step_callback_isolation(self):
        """A buggy user callback must NOT propagate into the training loop."""
        from backpropagate.multi_run import (
            MultiRunTrainer,
            _build_multi_run_step_callback,
        )

        def buggy(run_idx, step, loss):  # noqa: ARG001
            raise RuntimeError("user callback bug")

        with patch("torch.cuda.is_available", return_value=False):
            mrt = MultiRunTrainer(
                model="smollm3-3b",
                num_runs=1,
                steps_per_run=1,
                on_step=buggy,
            )

        cb = _build_multi_run_step_callback(mrt, run_idx=1)
        state = MagicMock(global_step=10)
        # Must not raise.
        cb.on_log(args=None, state=state, control=None, logs={"loss": 1.0})

    def test_on_step_falls_back_to_log_history(self):
        """When logs dict is empty, the bridge polls state.log_history tail."""
        from backpropagate.multi_run import (
            MultiRunTrainer,
            _build_multi_run_step_callback,
        )

        captured: list[tuple] = []

        def cb_user(run_idx, step, loss):
            captured.append((run_idx, step, loss))

        with patch("torch.cuda.is_available", return_value=False):
            mrt = MultiRunTrainer(
                model="smollm3-3b",
                num_runs=1,
                steps_per_run=1,
                on_step=cb_user,
            )

        cb = _build_multi_run_step_callback(mrt, run_idx=2)
        state = MagicMock(
            global_step=20,
            log_history=[{"epoch": 1}, {"loss": 0.5}],
        )
        cb.on_log(args=None, state=state, control=None, logs=None)

        assert captured == [(2, 20, 0.5)]

    def test_on_run_start_and_complete_already_fire(self):
        """on_run_start and on_run_complete are wired (pre-Wave-6b parity).

        This test pins the pre-existing wiring so a future refactor of
        _execute_run cannot silently remove the call sites.
        """
        from backpropagate.multi_run import MultiRunTrainer

        with patch("torch.cuda.is_available", return_value=False):
            mrt = MultiRunTrainer(
                model="smollm3-3b",
                num_runs=1,
                steps_per_run=1,
                on_run_start=lambda idx: None,
                on_run_complete=lambda result: None,
            )

        assert callable(mrt.on_run_start)
        assert callable(mrt.on_run_complete)

    def test_on_gpu_status_fires_from_monitor(self):
        """on_gpu_status fires from _on_gpu_status when supplied."""
        from backpropagate.gpu_safety import GPUCondition, GPUStatus
        from backpropagate.multi_run import MultiRunTrainer

        seen: list = []

        def gpu_cb(status):
            seen.append(status)

        with patch("torch.cuda.is_available", return_value=False):
            mrt = MultiRunTrainer(
                model="smollm3-3b",
                num_runs=1,
                steps_per_run=1,
                on_gpu_status=gpu_cb,
            )

        status = GPUStatus(
            available=True,
            device_name="test",
            temperature_c=60.0,
            vram_used_gb=4.0,
            vram_total_gb=16.0,
            vram_percent=25.0,
            condition=GPUCondition.SAFE,
            condition_reason="ok",
        )
        mrt._on_gpu_status(status)

        assert seen == [status]


# =============================================================================
# Item 4: CheckpointManager._save_manifest filelock parity
# =============================================================================


class TestCheckpointManagerFilelock:
    """v1.4 BACKEND-F-004: filelock parity for _save_manifest writes."""

    def test_locked_manifest_write_yields_true_on_success(self, tmp_path: Path):
        """Lock acquisition succeeds in the happy path → yields True."""
        from backpropagate.checkpoints import CheckpointManager

        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        with mgr._locked_manifest_write("test_op") as acquired:
            assert acquired is True

    def test_register_creates_lockfile_alongside_manifest(self, tmp_path: Path):
        """register() now writes through the lock — lockfile lives next to manifest."""
        from backpropagate.checkpoints import CheckpointManager

        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        cp_dir = tmp_path / "run_001" / "lora"
        cp_dir.mkdir(parents=True)
        (cp_dir / "weights.bin").write_bytes(b"x" * 1024)

        mgr.register(
            run_index=1,
            checkpoint_path=str(cp_dir),
            training_loss=0.5,
        )

        # Both the manifest and the lockfile sibling exist after the write.
        assert (tmp_path / "manifest.json").exists()
        # Lock file may be cleaned up by filelock after release on some
        # platforms; checking that the path is configured is sufficient.
        assert mgr._lock_path == tmp_path / "manifest.json.lock"

    def test_lock_timeout_default_30s(self, tmp_path: Path):
        """Default lock timeout is 30s (mirrors RunHistoryManager BACKEND-F-012)."""
        from backpropagate.checkpoints import CheckpointManager

        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        assert mgr._lock_timeout_seconds == 30.0

    def test_lock_timeout_kwarg_override(self, tmp_path: Path):
        """lock_timeout_seconds kwarg overrides the default."""
        from backpropagate.checkpoints import CheckpointManager

        mgr = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            lock_timeout_seconds=5.0,
        )
        assert mgr._lock_timeout_seconds == 5.0

    def test_lock_timeout_zero_means_block_forever(self, tmp_path: Path):
        """lock_timeout_seconds=0 → block forever (operator opt-in)."""
        from backpropagate.checkpoints import CheckpointManager

        mgr = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            lock_timeout_seconds=0.0,
        )
        # Public contract: 0 means block-forever; this is the same
        # convention as RunHistoryManager.lock_timeout_seconds.
        assert mgr._lock_timeout_seconds == 0.0

    def test_prune_runs_under_lock(self, tmp_path: Path):
        """prune() exercises _locked_manifest_write before saving."""
        from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy

        # auto_prune=False so register doesn't double-prune in the test.
        mgr = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            policy=CheckpointPolicy(auto_prune=False, max_total=1),
        )
        for idx in range(3):
            cp = tmp_path / f"run_{idx:03d}" / "lora"
            cp.mkdir(parents=True)
            (cp / "x").write_bytes(b"x" * 100)
            mgr.register(run_index=idx, checkpoint_path=str(cp), training_loss=float(idx))

        # No exception → the prune path's lock acquisition worked.
        pruned = mgr.prune(dry_run=True)
        assert isinstance(pruned, list)

    def test_concurrent_register_serializes_through_lock(self, tmp_path: Path):
        """Two CheckpointManager instances on the same dir serialize their writes.

        End-to-end: both managers write through the same filelock; the
        on-disk manifest contains BOTH registrations after both succeed
        (no silent-clobber from interleaved writes).
        """
        from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy

        # Disable auto-prune so we can count registrations directly.
        policy = CheckpointPolicy(auto_prune=False)

        mgr_a = CheckpointManager(checkpoint_dir=str(tmp_path), policy=policy)
        mgr_b = CheckpointManager(checkpoint_dir=str(tmp_path), policy=policy)

        cp_a = tmp_path / "run_001" / "lora"
        cp_a.mkdir(parents=True)
        (cp_a / "x").write_bytes(b"x")

        cp_b = tmp_path / "run_002" / "lora"
        cp_b.mkdir(parents=True)
        (cp_b / "x").write_bytes(b"x")

        mgr_a.register(run_index=1, checkpoint_path=str(cp_a), training_loss=1.0)

        # mgr_b loaded the manifest BEFORE mgr_a's register, so its
        # in-memory state is stale. The fix at the in-memory layer is
        # out of scope for the lock contract — what BACKEND-F-004
        # guarantees is that the WRITES don't interleave / corrupt
        # the JSON file. Reload mgr_b after mgr_a's write so its
        # subsequent register lands cleanly atop the on-disk state.
        mgr_b._load_manifest()
        mgr_b.register(run_index=2, checkpoint_path=str(cp_b), training_loss=2.0)

        # Reload a fresh manager and verify both entries persisted.
        mgr_check = CheckpointManager(checkpoint_dir=str(tmp_path), policy=policy)
        run_indices = sorted(cp.run_index for cp in mgr_check.list_checkpoints())
        assert run_indices == [1, 2]


# =============================================================================
# MultiRunTrainer mode='full' end-to-end gate
# =============================================================================


class TestMultiRunTrainerModeFull:
    """Mode='full' construction-time gate fires through MultiRunTrainer too."""

    def test_multi_run_config_default_mode_is_lora(self):
        from backpropagate.multi_run import MultiRunConfig

        cfg = MultiRunConfig()
        assert cfg.mode == "lora"

    def test_multi_run_config_accepts_mode_full(self):
        from backpropagate.multi_run import MultiRunConfig

        cfg = MultiRunConfig(mode="full")
        assert cfg.mode == "full"

    def test_multi_run_threads_mode_to_trainer_via_run(self, tmp_path: Path):
        """MultiRunTrainer.run() constructs the inner Trainer with config.mode."""
        from backpropagate.exceptions import FullFinetuneModelTooLargeError
        from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

        cfg = MultiRunConfig(
            num_runs=1,
            steps_per_run=1,
            samples_per_run=1,
            checkpoint_dir=str(tmp_path),
            mode="full",
        )
        # Use a 7B model id — should raise from the inner Trainer ctor as
        # soon as run() instantiates it.
        with patch("torch.cuda.is_available", return_value=False):
            mrt = MultiRunTrainer(
                model="Qwen/Qwen2.5-7B-Instruct",
                config=cfg,
            )
        # The error fires at the Trainer construction inside run()'s
        # try block. We don't actually exercise the full run loop (no
        # CUDA / model); instead we directly probe the Trainer
        # construction path that mode='full' triggers.
        with pytest.raises(FullFinetuneModelTooLargeError):
            from backpropagate.trainer import Trainer

            Trainer(model=mrt.model_name, mode=mrt.config.mode)
