"""Tests for v1.5 T1.2 — ORPO (reference-free preference training).

ORPO Wave 2 is the trainer-integration half of the v1.5 ORPO feature (config
+ dataset halves landed in Wave 1). These tests pin the trainer wiring:

- ``Trainer(method=...)`` resolution + validation + the ORPO+full guard
  (BACKEND / ORPO Wave 2).
- ``Trainer`` ORPO learning-rate default (8e-6) vs explicit override.
- ``_build_orpo_config`` — ORPO-specific kwargs (``beta`` / ``max_length``),
  the absence of SFT-only ``packing`` / forced ``gradient_checkpointing``, and
  the Stage-A CPU-runner regression guard (CPU → ``adamw_torch`` + fp32) plus
  the consumer-GPU paged/bf16 path.
- ``train()`` objective dispatch — ORPO builds an ``ORPOTrainer`` (NOT an
  ``SFTTrainer``), skips ``_apply_train_on_responses_only`` + the Windows
  pre-tokenize, tags run-history with ``method``, and the OOM-retry path
  rebuilds an ``ORPOTrainer`` via the shared ``_build_trainer`` helper (the
  anti-drift contract).
- ``_load_dataset(method='orpo')`` — preference-shaped output, the
  ``DatasetFormatError`` on an SFT dataset, and the symmetric SFT-on-preference
  WARN.

All tests are pure-Python / CPU / trl-MOCKED (the real-train smoke is a later
agent's). They follow the ``patch.dict("sys.modules", {"trl": MagicMock(...)})``
+ ``patch("torch.cuda.is_available", ...)`` patterns established in
test_trainer.py / test_wave6b_features.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Shared stubs / helpers
# =============================================================================


class _StubORPOConfig:
    """Stub ORPOConfig that records the kwargs it was constructed with.

    Mirrors the ``_StubSFTConfig`` pattern in test_wave6b_features.py — any
    object that captures **kwargs works; using a stub means the test does not
    depend on a real trl install for the config-shape assertions.
    """

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


def _build_orpo_config_capturing(**call_kwargs) -> dict:
    """Call ``_build_orpo_config`` with trl mocked to ``_StubORPOConfig``.

    Returns the captured ORPOConfig constructor kwargs. The caller wraps this
    in the appropriate ``torch.cuda.is_available`` patch for the CPU vs GPU
    path under test.
    """
    from backpropagate import trainer as trainer_mod

    captured: dict = {}

    class _Capture(_StubORPOConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    with patch.dict("sys.modules", {"trl": MagicMock(ORPOConfig=_Capture)}):
        trainer_mod._build_orpo_config(**call_kwargs)
    return captured


_ORPO_CFG_BASE: dict = {
    "output_dir": "./out",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_steps": 10,
    "learning_rate": 8e-6,
    "warmup_steps": 2,
    "max_seq_length": 1024,
    "orpo_beta": 0.1,
    "seed": 42,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
}


class _ORPOOOMScript:
    """ORPOTrainer mock factory whose .train() OOMs N times then succeeds.

    Mirrors test_trainer.py::_OOMScript but for ORPO: the OOM-recovery loop
    re-instantiates the inner trainer on each retry, so we need a fresh mock
    instance per construction that still shares the failure counter. We also
    count how many times the factory fired so a test can assert the OOM-retry
    REBUILT an ORPOTrainer (not an SFTTrainer) — the BACKEND-A-003 anti-drift
    contract extended to ORPO.
    """

    def __init__(self, oom_count: int):
        self._oom_remaining = oom_count
        self.train_calls = 0
        self.construct_calls = 0

    def factory(self, *args, **kwargs):
        self.construct_calls += 1
        instance = MagicMock()

        def train_impl(*a, **k):
            self.train_calls += 1
            if self._oom_remaining > 0:
                self._oom_remaining -= 1
                # OOM-shaped RuntimeError — the trainer's detector matches the
                # "out of memory" substring when isinstance of
                # torch.cuda.OutOfMemoryError fails (it does in this CPU rig).
                raise RuntimeError("CUDA out of memory at batch_size attempt")
            result = MagicMock()
            result.training_loss = 0.42
            return result

        instance.train.side_effect = train_impl
        instance.state.log_history = [{"loss": 0.42}]
        return instance


def _orpo_trainer_ready(temp_dir, **kwargs):
    """Construct a CPU Trainer(method='orpo') with a fake loaded model."""
    from backpropagate.trainer import Trainer

    with patch("torch.cuda.is_available", return_value=False):
        trainer = Trainer(
            method="orpo",
            output_dir=str(temp_dir),
            use_unsloth=False,
            **kwargs,
        )
    trainer._model = MagicMock()
    trainer._tokenizer = MagicMock()
    trainer._is_loaded = True
    return trainer


# =============================================================================
# Trainer construction: method resolution + guards + LR default
# =============================================================================


class TestTrainerMethodResolution:
    """ORPO Wave 2: Trainer(method=...) resolution, validation, guards."""

    def test_method_default_is_sft(self):
        """Trainer() default method is 'sft' (preserves pre-v1.5 behavior)."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert trainer.method == "sft", (
            "method default drifted from 'sft'; this would silently change "
            "every existing operator's training contract."
        )

    def test_method_orpo_sets_attribute(self):
        """Trainer(method='orpo') resolves self.method='orpo'."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="orpo")

        assert trainer.method == "orpo"

    def test_method_orpo_default_beta(self):
        """Trainer(method='orpo') resolves orpo_beta from settings (0.1)."""
        from backpropagate.config import settings
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="orpo")

        assert trainer.orpo_beta == pytest.approx(settings.training.orpo_beta)

    def test_method_orpo_explicit_beta(self):
        """Trainer(method='orpo', orpo_beta=0.3) honors the explicit weight."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="orpo", orpo_beta=0.3)

        assert trainer.orpo_beta == pytest.approx(0.3)

    @pytest.mark.parametrize("bad_beta", [0.0, -1.0, -0.001])
    def test_method_orpo_nonpositive_beta_raises(self, bad_beta):
        """Re-audit #6 (trainer half): a DIRECT Trainer(method='orpo',
        orpo_beta<=0) raises InvalidSettingError (CONFIG_INVALID_SETTING).

        config.py validates the SETTINGS path; the direct kwarg bypasses it and
        would flow unclamped into ORPOConfig(beta=...) — beta=0 degenerates ORPO
        to SFT, negative trains toward the REJECTED response. The constructor
        re-validates (defense in depth, mirroring the method= guard).
        """
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError) as exc_info:
                Trainer(method="orpo", orpo_beta=bad_beta)

        err = exc_info.value
        assert err.setting_name == "orpo_beta"
        assert err.value == pytest.approx(bad_beta)
        assert err.code == "CONFIG_INVALID_SETTING"

    def test_nonpositive_beta_harmless_for_sft(self):
        """beta is inert for SFT, so a stray orpo_beta<=0 must NOT block a
        method='sft' run (the guard is gated on method=='orpo')."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            # No raise — SFT never builds an ORPOConfig.
            trainer = Trainer(method="sft", orpo_beta=0.0)
        assert trainer.method == "sft"

    def test_method_invalid_raises_invalid_setting(self):
        """Trainer(method='dpo') raises InvalidSettingError (CONFIG_INVALID_SETTING).

        Defense in depth: config.py validates the settings path, but a DIRECT
        Trainer(method=...) call bypasses the settings layer, so the
        constructor must re-validate.
        """
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError) as exc_info:
                Trainer(method="dpo")

        err = exc_info.value
        assert err.setting_name == "method"
        assert err.value == "dpo"
        assert err.code == "CONFIG_INVALID_SETTING"

    def test_method_orpo_with_full_mode_raises(self):
        """Trainer(method='orpo', mode='full') raises InvalidSettingError.

        ORPO is supported with mode='lora' ONLY in v1.5; the combination is
        refused at construction (after BOTH self.mode and self.method resolve).
        """
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError) as exc_info:
                # smollm3-3b clears the 3B full-FT ceiling, so the ONLY thing
                # that can raise here is the ORPO+full guard.
                Trainer(method="orpo", mode="full", model="smollm3-3b")

        err = exc_info.value
        assert err.setting_name == "method+mode"
        assert "mode='lora'" in str(err)

    def test_method_orpo_lowers_default_learning_rate(self):
        """method='orpo' with no explicit LR applies the 8e-6 ORPO default."""
        from backpropagate.trainer import _ORPO_DEFAULT_LR, Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="orpo")

        assert trainer.learning_rate == pytest.approx(_ORPO_DEFAULT_LR), (
            f"method='orpo' should default learning_rate to {_ORPO_DEFAULT_LR}; "
            f"got {trainer.learning_rate}."
        )
        assert pytest.approx(8e-6) == _ORPO_DEFAULT_LR

    def test_method_orpo_honors_explicit_learning_rate(self):
        """method='orpo' with explicit LR does NOT apply the ORPO default."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="orpo", learning_rate=1e-5)

        assert trainer.learning_rate == pytest.approx(1e-5)

    def test_method_sft_does_not_lower_learning_rate(self):
        """method='sft' (default) keeps the LoRA SFT learning-rate default.

        Guards against the ORPO LR branch accidentally firing for SFT (the
        two LR branches are mutually exclusive and gated on self.method).
        """
        from backpropagate.config import settings
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()  # method defaults to sft, mode defaults to lora

        assert trainer.learning_rate == pytest.approx(
            settings.training.learning_rate
        )


# =============================================================================
# _build_orpo_config
# =============================================================================


class TestBuildORPOConfig:
    """ORPO Wave 2: the module-level _build_orpo_config helper."""

    def test_passes_beta_and_max_length(self):
        """beta == orpo_beta and max_length == the passed max_seq_length."""
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_orpo_config_capturing(
                **{**_ORPO_CFG_BASE, "orpo_beta": 0.2, "max_seq_length": 768}
            )

        assert captured.get("beta") == pytest.approx(0.2)
        assert captured.get("max_length") == 768

    def test_omits_packing_and_gradient_checkpointing(self):
        """ORPOConfig must NOT receive packing or forced gradient_checkpointing.

        ORPOConfig has no ``packing`` field (it rejects it) and ORPO is
        mode='lora'-only in v1.5 (no full-FT activation-memory contract).
        """
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_orpo_config_capturing(**_ORPO_CFG_BASE)

        assert "packing" not in captured, (
            "packing must be ABSENT — ORPOConfig rejects it."
        )
        assert "gradient_checkpointing" not in captured, (
            "gradient_checkpointing must not be force-injected for ORPO."
        )

    def test_optional_truncation_knobs_added_when_set(self):
        """max_prompt_length / max_completion_length appear only when not None."""
        with patch("torch.cuda.is_available", return_value=False):
            with_knobs = _build_orpo_config_capturing(
                **{
                    **_ORPO_CFG_BASE,
                    "max_prompt_length": 256,
                    "max_completion_length": 200,
                }
            )
            without_knobs = _build_orpo_config_capturing(**_ORPO_CFG_BASE)

        assert with_knobs.get("max_prompt_length") == 256
        assert with_knobs.get("max_completion_length") == 200
        assert "max_prompt_length" not in without_knobs
        assert "max_completion_length" not in without_knobs

    def test_cpu_runner_downgrades_optim_and_forces_fp32(self):
        """CPU path → optim downgraded to adamw_torch AND bf16==fp16==False.

        This is the Stage-A CPU-runner regression guard for ORPO: because
        _build_orpo_config reuses _detect_optim_for_card (bnb-8bit → adamw_torch
        on CPU) and _detect_optimal_dtype (fp32 on CPU), the ORPO config is
        CPU-constructible for free. If either detector stops being reused, this
        assertion catches the divergence.
        """
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_orpo_config_capturing(
                **{**_ORPO_CFG_BASE, "optim": "adamw_8bit"}
            )

        assert captured.get("optim") == "adamw_torch", (
            "CPU runner must downgrade the bnb 8-bit optim to adamw_torch "
            "(bitsandbytes 8-bit optimizers are CUDA-only)."
        )
        assert captured.get("bf16") is False
        assert captured.get("fp16") is False

    def test_consumer_gpu_uses_paged_optim_and_bf16(self):
        """16GB Ada consumer GPU → paged_adamw_8bit + bf16 (parity with SFT)."""
        gb16 = 16 * (1024**3)
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_properties",
            return_value=MagicMock(total_memory=gb16),
        ), patch("torch.cuda.get_device_capability", return_value=(8, 9)):
            captured = _build_orpo_config_capturing(
                **{**_ORPO_CFG_BASE, "optim": "adamw_8bit"}
            )

        assert captured.get("optim") == "paged_adamw_8bit", (
            "consumer card (<24GB) should upgrade adamw_8bit -> paged_adamw_8bit."
        )
        assert captured.get("bf16") is True
        assert captured.get("fp16") is False

    def test_real_orpoconfig_is_cpu_constructible(self):
        """The REAL trl ORPOConfig builds on CPU (no bf16/packing rejection).

        Belt-and-braces against the real trl install: the Stage-A fix means a
        bare CPU ORPO config must construct without TrainingArguments rejecting
        bf16 or ORPOConfig rejecting an unknown kwarg.
        """
        pytest.importorskip("trl")
        from backpropagate import trainer as trainer_mod

        with patch("torch.cuda.is_available", return_value=False):
            cfg = trainer_mod._build_orpo_config(
                **{**_ORPO_CFG_BASE, "optim": "adamw_8bit"}
            )

        assert cfg.beta == pytest.approx(0.1)
        assert cfg.max_length == 1024
        assert bool(cfg.bf16) is False
        assert bool(cfg.fp16) is False


# =============================================================================
# train() objective dispatch
# =============================================================================


class TestTrainORPODispatch:
    """ORPO Wave 2: train() routes to ORPOTrainer and varies only what it must."""

    def _mock_orpo_instance(self, training_loss: float = 0.33) -> MagicMock:
        inst = MagicMock()
        inst.train.return_value = MagicMock(training_loss=training_loss)
        inst.state.log_history = [{"loss": training_loss}]
        return inst

    def test_orpo_builds_orpo_trainer_not_sft_trainer(self, temp_dir):
        """method='orpo' instantiates ORPOTrainer, never SFTTrainer."""
        trainer = _orpo_trainer_ready(temp_dir)
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)
        orpo_inst = self._mock_orpo_instance()

        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch(
            "trl.ORPOTrainer", return_value=orpo_inst
        ) as m_orpo, patch("trl.SFTTrainer") as m_sft, patch("trl.ORPOConfig"):
            run = trainer.train("dummy", steps=5)

        assert m_orpo.called, "ORPOTrainer should be constructed for method='orpo'."
        assert not m_sft.called, "SFTTrainer must NOT be constructed for ORPO."
        assert run is not None

    def test_orpo_skips_train_on_responses_only(self, temp_dir):
        """method='orpo' does NOT apply train_on_responses_only (paired data)."""
        from backpropagate import trainer as trainer_mod

        trainer = _orpo_trainer_ready(temp_dir)
        # use_unsloth would normally be a precondition; force it on to prove
        # the gate is the method check, not the unsloth check.
        trainer.use_unsloth = True
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)
        orpo_inst = self._mock_orpo_instance()

        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch(
            "trl.ORPOTrainer", return_value=orpo_inst
        ), patch("trl.SFTTrainer"), patch("trl.ORPOConfig"), patch.object(
            trainer_mod, "_apply_train_on_responses_only"
        ) as m_apply:
            trainer.train("dummy", steps=5)

        assert not m_apply.called, (
            "train_on_responses_only is meaningless for ORPO paired data and "
            "must be skipped."
        )

    def test_orpo_skips_windows_pre_tokenize(self, temp_dir):
        """On Windows, method='orpo' must NOT pre-tokenize (no text column)."""
        trainer = _orpo_trainer_ready(temp_dir)
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)
        orpo_inst = self._mock_orpo_instance()

        # Force the Windows branch ON via os.name + the setting; assert
        # _pre_tokenize is still NOT called because method='orpo' gates it.
        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch(
            "trl.ORPOTrainer", return_value=orpo_inst
        ), patch("trl.SFTTrainer"), patch("trl.ORPOConfig"), patch(
            "backpropagate.trainer.os.name", "nt"
        ), patch(
            "backpropagate.config.settings.windows.pre_tokenize", True
        ), patch.object(
            trainer, "_pre_tokenize"
        ) as m_pre:
            trainer.train("dummy", steps=5)

        assert not m_pre.called, (
            "ORPO must not pre-tokenize on Windows — ORPOTrainer tokenizes its "
            "own paired rows."
        )

    def test_orpo_records_method_in_run_history(self, temp_dir):
        """run-history hyperparameters carry method='orpo' + orpo_beta."""
        from backpropagate.checkpoints import RunHistoryManager

        trainer = _orpo_trainer_ready(temp_dir, orpo_beta=0.25)
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)
        orpo_inst = self._mock_orpo_instance()

        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch(
            "trl.ORPOTrainer", return_value=orpo_inst
        ), patch("trl.SFTTrainer"), patch("trl.ORPOConfig"):
            run = trainer.train("dummy", steps=5)

        record = RunHistoryManager(str(temp_dir)).get_run(run.run_id)
        assert record is not None
        hp = record.get("hyperparameters", {})
        assert hp.get("method") == "orpo"
        assert hp.get("orpo_beta") == pytest.approx(0.25)

    def test_orpo_oom_retry_rebuilds_orpo_trainer(self, temp_dir):
        """One OOM → the retry REBUILDS an ORPOTrainer (not an SFTTrainer).

        This is the BACKEND-A-003 anti-drift contract extended to ORPO: the
        first-attempt construction and the OOM-retry rebuild both route through
        the shared _build_trainer helper, so the retry cannot silently fall
        back to SFTTrainer. SFTTrainer is patched to raise if ever constructed.
        """
        trainer = _orpo_trainer_ready(
            temp_dir, batch_size=4, gradient_accumulation=1, oom_recovery=True
        )
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)
        script = _ORPOOOMScript(oom_count=1)  # one OOM then success

        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch(
            "trl.ORPOTrainer", side_effect=script.factory
        ), patch(
            "trl.SFTTrainer",
            side_effect=AssertionError(
                "SFTTrainer must NOT be constructed on an ORPO OOM-retry"
            ),
        ), patch("trl.ORPOConfig"):
            run = trainer.train("dummy", steps=5)

        assert script.train_calls == 2, (
            f"Expected 2 .train() invocations (initial OOM + 1 retry); "
            f"got {script.train_calls}."
        )
        assert script.construct_calls == 2, (
            f"Expected the inner ORPOTrainer to be constructed twice (first + "
            f"retry rebuild); got {script.construct_calls}."
        )
        assert run is not None


# =============================================================================
# Cross-version warnings_issued shim (trl 0.24 x transformers 5.x)
# =============================================================================


class TestORPOWarningsIssuedShim:
    """``_build_trainer`` provides ``model.warnings_issued`` for trl's ORPOTrainer.

    trl 0.24's ``ORPOTrainer.__init__`` writes
    ``model.warnings_issued["estimate_tokens"] = True``; transformers 5.x
    REMOVED that attribute from ``PreTrainedModel``, so the write raised
    ``AttributeError`` at constructor time on the project's own target stack
    (trl 0.24 + transformers 5.5) — before a single step. ``_build_trainer``
    now provides an inert dict when the attribute is absent. The gated
    ``test_orpo_smoke.py`` is the end-to-end proof on a real model; THIS is the
    fast-lane unit regression so the shim can't silently rot.
    """

    class _PlainModel:
        """Model stand-in with NO ``warnings_issued`` (a MagicMock would
        auto-create the attribute and make the assertion vacuous)."""

    def test_build_trainer_adds_warnings_issued_when_absent(self, temp_dir):
        trainer = _orpo_trainer_ready(temp_dir)
        model = self._PlainModel()
        trainer._model = model
        assert not hasattr(model, "warnings_issued")  # precondition

        with patch("trl.ORPOTrainer", return_value=MagicMock()) as m_orpo:
            trainer._build_trainer(MagicMock(), MagicMock(), [])

        assert m_orpo.called
        assert getattr(model, "warnings_issued", None) == {}, (
            "the shim must provide an inert warnings_issued dict so trl's "
            "ORPOTrainer can write estimate_tokens on transformers 5.x."
        )

    def test_build_trainer_does_not_clobber_existing_warnings_issued(self, temp_dir):
        trainer = _orpo_trainer_ready(temp_dir)
        model = self._PlainModel()
        model.warnings_issued = {"estimate_tokens": True, "preexisting": 1}
        trainer._model = model

        with patch("trl.ORPOTrainer", return_value=MagicMock()):
            trainer._build_trainer(MagicMock(), MagicMock(), [])

        assert model.warnings_issued == {"estimate_tokens": True, "preexisting": 1}, (
            "the shim must NOT overwrite an existing warnings_issued "
            "(transformers 4.x populates it via PreTrainedModel.__init__)."
        )

    def test_sft_build_trainer_does_not_touch_warnings_issued(self, temp_dir):
        """The shim is ORPO-scoped — SFT construction never adds the attribute."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                method="sft", output_dir=str(temp_dir), use_unsloth=False
            )
        model = self._PlainModel()
        trainer._model = model
        trainer._tokenizer = MagicMock()

        with patch("trl.SFTTrainer", return_value=MagicMock()) as m_sft:
            trainer._build_trainer(MagicMock(), MagicMock(), [])

        assert m_sft.called
        assert not hasattr(model, "warnings_issued"), (
            "the SFT path must not add warnings_issued — the shim is ORPO-only."
        )


# =============================================================================
# _load_dataset(method='orpo')
# =============================================================================


class TestLoadDatasetORPO:
    """ORPO Wave 2: _load_dataset preference path + format guards."""

    def _write_jsonl(self, path: Path, rows: list[dict]) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return str(path)

    def test_pair_jsonl_returns_preference_dataset(self, temp_dir):
        """A {chosen, rejected, prompt} JSONL yields a preference-shaped Dataset."""
        trainer = _orpo_trainer_ready(temp_dir)
        path = self._write_jsonl(
            temp_dir / "pairs.jsonl",
            [
                {"prompt": "Hi", "chosen": "Hello there!", "rejected": "no"},
                {"prompt": "Bye", "chosen": "Goodbye!", "rejected": "k"},
            ],
        )

        ds = trainer._load_dataset(path, method="orpo")

        assert "chosen" in ds.column_names
        assert "rejected" in ds.column_names
        assert len(ds) == 2

    def test_sft_jsonl_with_orpo_raises_format_error(self, temp_dir):
        """An SFT (messages) JSONL + method='orpo' raises DatasetFormatError."""
        from backpropagate.exceptions import DatasetFormatError

        trainer = _orpo_trainer_ready(temp_dir)
        path = self._write_jsonl(
            temp_dir / "sft.jsonl",
            [
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "yo"},
                    ]
                }
            ],
        )

        with pytest.raises(DatasetFormatError) as exc_info:
            trainer._load_dataset(path, method="orpo")

        assert exc_info.value.code == "INPUT_DATASET_FORMAT_UNSUPPORTED"

    def test_in_memory_dataset_without_pairs_raises_for_orpo(self, temp_dir):
        """An in-memory Dataset with no chosen/rejected columns raises for ORPO."""
        from datasets import Dataset

        from backpropagate.exceptions import DatasetFormatError

        trainer = _orpo_trainer_ready(temp_dir)
        ds = Dataset.from_list([{"text": "hello"}, {"text": "world"}])

        with pytest.raises(DatasetFormatError) as exc_info:
            trainer._load_dataset(ds, method="orpo")

        assert exc_info.value.code == "INPUT_DATASET_FORMAT_UNSUPPORTED"

    def test_in_memory_preference_dataset_passes_for_orpo(self, temp_dir):
        """An in-memory Dataset WITH chosen/rejected is accepted for ORPO."""
        from datasets import Dataset

        trainer = _orpo_trainer_ready(temp_dir)
        ds = Dataset.from_list(
            [
                {"prompt": "p", "chosen": "good", "rejected": "bad"},
                {"prompt": "q", "chosen": "nice", "rejected": "meh"},
            ]
        )

        out = trainer._load_dataset(ds, method="orpo")

        assert "chosen" in out.column_names and "rejected" in out.column_names
        assert len(out) == 2

    def test_sft_method_on_preference_data_warns(self, temp_dir, caplog):
        """method='sft' on a preference JSONL WARNs and points at --method orpo."""
        import logging

        trainer = _orpo_trainer_ready(temp_dir)
        path = self._write_jsonl(
            temp_dir / "pairs.jsonl",
            [
                {"prompt": "Hi", "chosen": "Hello!", "rejected": "no"},
                {"prompt": "Bye", "chosen": "Goodbye!", "rejected": "k"},
            ],
        )

        with caplog.at_level(logging.WARNING, logger="backpropagate.trainer"):
            trainer._load_dataset(path, method="sft")

        warned = any(
            "preference" in rec.message.lower()
            and "method orpo" in rec.message.lower()
            for rec in caplog.records
        )
        assert warned, (
            "SFT run on a preference-shaped dataset should WARN and suggest "
            "--method orpo."
        )
