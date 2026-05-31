"""Tests for v1.5 T2.1 — FP8 compute path (torchao float8, experimental opt-in).

FP8 is the RTX-50-series (Blackwell sm_120) / Hopper (sm_90) headroom lever:
the BASE projection linears are converted to torchao ``Float8Linear`` AFTER the
LoRA adapter is attached, with the adapter's rank-``r`` sub-linears + ``lm_head``
+ embeddings EXCLUDED (converting the rank-``r`` linears crashes on backward —
the load-bearing gotcha, dogfood-verified on this box). It layers on top of bf16
and degrades gracefully to bf16 (one WARN, no raise) on a non-CUDA / pre-Hopper
card or when torchao is absent.

These tests are pure-Python / CPU / torchao+torch MOCKED — the real-train proof
is ``tests/test_fp8_smoke.py``. They pin:

- Config round-trip (BOTH the pydantic and dataclass-fallback branches) + env
  override (``BACKPROPAGATE_TRAINING__FP8``).
- The constructor GATE LADDER: fp8+full / fp8+orpo / fp8+explicit-4bit raise
  ``InvalidSettingError`` (CONFIG_INVALID_SETTING); fp8+default-4bit flips 4-bit
  off with an INFO log (no raise).
- GRACEFUL FALLBACK: fp8=True on a CUDA-absent / pre-sm90 / torchao-absent host
  sets ``_fp8_effective is False``, logs ONE WARN, and does NOT raise.
- ``_fp8_supported()`` truth table across the three environment axes.
- The ``_fp8_module_filter`` predicate on a tiny mocked module tree (base linear
  → convert; ``lora_``-FQN → exclude; ``lm_head`` / embedding → exclude;
  non-Linear → exclude).
- ``_apply_fp8_to_base`` behavior: no-op when inactive; bf16-degrade on a
  conversion failure; the RUNTIME_FP8_UNSUPPORTED hard error on a broken
  torchao import.

The patterns (``patch.dict("sys.modules", ...)`` + ``patch("torch.cuda.*")``)
follow test_orpo.py / test_wave6b_features.py.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Helpers
# =============================================================================


def _cpu_trainer(**kwargs):
    """Construct a CPU Trainer with use_unsloth=False (no GPU needed).

    Patches ``torch.cuda.is_available`` False so the gate ladder takes the
    CPU branch unless a test overrides the patch itself.
    """
    from backpropagate.trainer import Trainer

    with patch("torch.cuda.is_available", return_value=False):
        return Trainer(use_unsloth=False, **kwargs)


# =============================================================================
# Config round-trip — pydantic branch
# =============================================================================


class TestFp8ConfigPydantic:
    """v1.5 T2.1: the ``fp8`` field on the pydantic TrainingConfig branch."""

    def test_fp8_default_false(self):
        """fp8 defaults False — existing runs stay byte-identical (bf16)."""
        from backpropagate.config import TrainingConfig

        assert TrainingConfig().fp8 is False

    def test_fp8_accepts_true(self):
        """fp8=True round-trips on the field."""
        from backpropagate.config import TrainingConfig

        assert TrainingConfig(fp8=True).fp8 is True

    def test_fp8_via_env(self, monkeypatch):
        """BACKPROPAGATE_TRAINING__FP8=true round-trips (pydantic env prefix)."""
        from backpropagate.config import TrainingConfig

        monkeypatch.setenv("BACKPROPAGATE_TRAINING__FP8", "true")
        assert TrainingConfig().fp8 is True

    def test_fp8_env_false_default(self, monkeypatch):
        """An unset env var leaves fp8 at its False default."""
        from backpropagate.config import TrainingConfig

        monkeypatch.delenv("BACKPROPAGATE_TRAINING__FP8", raising=False)
        assert TrainingConfig().fp8 is False


# =============================================================================
# Config round-trip — dataclass fallback branch
# =============================================================================


class TestFp8ConfigDataclassFallback:
    """v1.5 T2.1: ``fp8`` must match byte-for-byte on the dataclass fallback.

    Mirrors ``TestOrpoDataclassFallback`` — re-exec config.py with
    pydantic_settings blocked so the dataclass branch materialises.
    """

    def _load_dataclass_config_module(self):
        import importlib
        import types

        import backpropagate.config as cfg

        source = Path(cfg.__file__).read_text(encoding="utf-8")
        fake = types.ModuleType("backpropagate._config_dataclass_probe_fp8")
        fake.__dict__["__file__"] = cfg.__file__
        fake.__dict__["__name__"] = "backpropagate.config"
        sys.modules.setdefault(
            "backpropagate", importlib.import_module("backpropagate")
        )
        blocked = {"pydantic_settings": None}
        with patch.dict(sys.modules, blocked):
            try:
                exec(compile(source, cfg.__file__, "exec"), fake.__dict__)  # noqa: S102
            except Exception:
                return None
        if fake.__dict__.get("PYDANTIC_SETTINGS_AVAILABLE", True):
            return None
        return fake

    def test_dataclass_fp8_default_false(self):
        mod = self._load_dataclass_config_module()
        if mod is None:
            pytest.skip("dataclass fallback branch not materialisable here")
        cfg_cls = mod.__dict__["TrainingConfig"]
        assert cfg_cls().fp8 is False

    def test_dataclass_fp8_accepts_true(self):
        mod = self._load_dataclass_config_module()
        if mod is None:
            pytest.skip("dataclass fallback branch not materialisable here")
        cfg_cls = mod.__dict__["TrainingConfig"]
        assert cfg_cls(fp8=True).fp8 is True


# =============================================================================
# _fp8_supported() truth table
# =============================================================================


class TestFp8Supported:
    """v1.5 T2.1: the ``_fp8_supported()`` environment-axis truth table.

    Three AND-ed axes: CUDA present, torchao present, compute capability >= 9.
    Each returns ``(False, reason)`` on its own when it is the failing axis.
    """

    def test_no_cuda_unsupported(self):
        """CUDA absent → (False, reason mentioning CUDA). No torchao needed."""
        trainer = _cpu_trainer()
        with patch("torch.cuda.is_available", return_value=False):
            ok, reason = trainer._fp8_supported()
        assert ok is False
        assert reason is not None
        assert "cuda" in reason.lower()

    def test_torchao_absent_unsupported(self):
        """CUDA + sm_90 present but torchao absent → (False, torchao reason)."""
        trainer = _cpu_trainer()
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ), patch("backpropagate.trainer.check_feature", return_value=False):
            ok, reason = trainer._fp8_supported()
        assert ok is False
        assert reason is not None
        assert "torchao" in reason.lower()

    def test_pre_sm90_unsupported(self):
        """CUDA + torchao present but sm_89 (Ada) → (False, capability reason)."""
        trainer = _cpu_trainer()
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(8, 9)
        ), patch("backpropagate.trainer.check_feature", return_value=True), patch(
            "torch.cuda.get_device_name", return_value="RTX 4090"
        ):
            ok, reason = trainer._fp8_supported()
        assert ok is False
        assert reason is not None
        # Names the sm level + the fix.
        assert "sm_" in reason.lower() or "sm9" in reason.lower()

    def test_hopper_supported(self):
        """CUDA + torchao + sm_90 (Hopper) → (True, None)."""
        trainer = _cpu_trainer()
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ), patch("backpropagate.trainer.check_feature", return_value=True):
            ok, reason = trainer._fp8_supported()
        assert ok is True
        assert reason is None

    def test_blackwell_supported(self):
        """CUDA + torchao + sm_120 (Blackwell, the verified card) → (True, None)."""
        trainer = _cpu_trainer()
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(12, 0)
        ), patch("backpropagate.trainer.check_feature", return_value=True):
            ok, reason = trainer._fp8_supported()
        assert ok is True
        assert reason is None


# =============================================================================
# Constructor gate ladder — MISCONFIGURATION (raise)
# =============================================================================


class TestFp8GateLadderRaises:
    """v1.5 T2.1: combinations that can never work raise at construction."""

    def test_fp8_plus_full_raises(self):
        """fp8=True + mode='full' → InvalidSettingError (CONFIG_INVALID_SETTING)."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError) as exc_info:
                # smollm3-3b clears the 3B full-FT ceiling, so the ONLY thing
                # that can raise is the FP8+full gate.
                Trainer(
                    fp8=True, mode="full", model="smollm3-3b", use_unsloth=False
                )
        err = exc_info.value
        assert err.code == "CONFIG_INVALID_SETTING"
        assert err.setting_name == "fp8+mode"
        assert "mode='lora'" in str(err)

    def test_fp8_plus_orpo_raises(self):
        """fp8=True + method='orpo' → InvalidSettingError (CONFIG_INVALID_SETTING)."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError) as exc_info:
                Trainer(fp8=True, method="orpo", use_unsloth=False)
        err = exc_info.value
        assert err.code == "CONFIG_INVALID_SETTING"
        assert err.setting_name == "fp8+method"
        assert "method='sft'" in str(err)

    def test_fp8_plus_explicit_4bit_raises(self):
        """fp8=True + EXPLICIT load_in_4bit=True → InvalidSettingError."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError) as exc_info:
                Trainer(fp8=True, load_in_4bit=True, use_unsloth=False)
        err = exc_info.value
        assert err.code == "CONFIG_INVALID_SETTING"
        assert err.setting_name == "fp8+load_in_4bit"
        assert "not stackable" in str(err)

    def test_fp8_orpo_takes_priority_over_capability(self):
        """The MISCONFIG gates fire even on a fully FP8-capable host.

        A misconfiguration is an operator error regardless of hardware — it must
        raise even when CUDA + torchao + sm_90 would otherwise support FP8.
        """
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ), patch("backpropagate.trainer.check_feature", return_value=True):
            with pytest.raises(InvalidSettingError):
                Trainer(fp8=True, method="orpo", use_unsloth=False)


# =============================================================================
# Constructor gate ladder — default 4-bit flip (INFO, no raise)
# =============================================================================


class TestFp8Default4bitFlip:
    """v1.5 T2.1: when 4-bit is only the default, FP8 flips it off (no raise)."""

    def test_fp8_disables_default_4bit_no_raise(self, caplog):
        """fp8=True with NO explicit 4-bit → _load_in_4bit flipped False + INFO."""
        from backpropagate.trainer import Trainer

        with caplog.at_level(logging.INFO, logger="backpropagate.trainer"):
            with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.get_device_capability", return_value=(12, 0)
            ), patch("backpropagate.trainer.check_feature", return_value=True):
                trainer = Trainer(fp8=True, use_unsloth=False)
        # Default 4-bit was flipped off (FP8 keeps the base in float8).
        assert trainer._load_in_4bit is False
        assert trainer._load_in_4bit_explicit is False
        # And it was NOT an explicit request, so no raise — we got a Trainer.
        assert trainer.fp8 is True
        # The flip was announced at INFO.
        assert any(
            "disabling the default 4-bit" in rec.message for rec in caplog.records
        )

    def test_no_fp8_keeps_default_4bit(self):
        """Without fp8 the default 4-bit stays on (byte-identical pre-v1.5)."""
        trainer = _cpu_trainer()
        assert trainer._load_in_4bit is True
        assert trainer._fp8_effective is False


# =============================================================================
# Constructor — graceful fallback (WARN, no raise, _fp8_effective False)
# =============================================================================


class TestFp8GracefulFallback:
    """v1.5 T2.1: fp8 requested on an unsupported host degrades to bf16.

    The contract: ONE WARN naming the reason + fix, ``_fp8_effective is False``,
    NO raise — mirrors the unsloth→transformers fallback (a missing capability
    is an environment fact, not an operator error).
    """

    def test_cuda_absent_degrades(self, caplog):
        """No CUDA → _fp8_effective False + WARN, no raise."""
        from backpropagate.trainer import Trainer

        with caplog.at_level(logging.WARNING, logger="backpropagate.trainer"):
            with patch("torch.cuda.is_available", return_value=False):
                trainer = Trainer(fp8=True, use_unsloth=False)
        assert trainer._fp8_effective is False
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("unavailable on this host" in r.message for r in warns)
        assert any("bf16" in r.message for r in warns)

    def test_pre_sm90_degrades(self):
        """Ada (sm_89) → _fp8_effective False, no raise."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(8, 9)
        ), patch("backpropagate.trainer.check_feature", return_value=True), patch(
            "torch.cuda.get_device_name", return_value="RTX 4090"
        ):
            trainer = Trainer(fp8=True, use_unsloth=False)
        assert trainer._fp8_effective is False

    def test_torchao_absent_degrades(self):
        """torchao missing on a capable card → _fp8_effective False, no raise."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(12, 0)
        ), patch("backpropagate.trainer.check_feature", return_value=False):
            trainer = Trainer(fp8=True, use_unsloth=False)
        assert trainer._fp8_effective is False

    def test_supported_sets_effective_and_warns_experimental(self, caplog):
        """A fully capable host → _fp8_effective True + ONE experimental WARN."""
        from backpropagate.trainer import Trainer

        with caplog.at_level(logging.WARNING, logger="backpropagate.trainer"):
            with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.get_device_capability", return_value=(12, 0)
            ), patch("backpropagate.trainer.check_feature", return_value=True):
                trainer = Trainer(fp8=True, use_unsloth=False)
        assert trainer._fp8_effective is True
        assert any(
            "EXPERIMENTAL" in r.message for r in caplog.records
        ), "the one-shot experimental WARN should fire when FP8 is effective"


# =============================================================================
# _fp8_module_filter predicate
# =============================================================================


class TestFp8ModuleFilter:
    """v1.5 T2.1: the load-bearing convert/exclude predicate.

    Base projection linears → convert (True). LoRA adapter sub-linears,
    lm_head, embeddings, and any non-Linear → exclude (False). This is the fix
    for the naive-conversion backward crash.
    """

    @staticmethod
    def _linear():
        import torch.nn as nn

        return nn.Linear(8, 8)

    @staticmethod
    def _embedding():
        import torch.nn as nn

        return nn.Embedding(10, 8)

    def test_base_projection_linear_converts(self):
        from backpropagate.trainer import Trainer

        assert (
            Trainer._fp8_module_filter(
                self._linear(), "model.layers.0.self_attn.q_proj"
            )
            is True
        )

    def test_base_mlp_linear_converts(self):
        from backpropagate.trainer import Trainer

        assert (
            Trainer._fp8_module_filter(
                self._linear(), "model.layers.5.mlp.gate_proj"
            )
            is True
        )

    def test_lora_a_excluded(self):
        """A ``lora_A`` sub-linear is excluded — the load-bearing exclusion."""
        from backpropagate.trainer import Trainer

        assert (
            Trainer._fp8_module_filter(
                self._linear(),
                "base_model.model.layers.0.self_attn.q_proj.lora_A.default",
            )
            is False
        )

    def test_lora_b_excluded(self):
        from backpropagate.trainer import Trainer

        assert (
            Trainer._fp8_module_filter(
                self._linear(),
                "base_model.model.layers.0.self_attn.q_proj.lora_B.default",
            )
            is False
        )

    def test_lm_head_excluded(self):
        from backpropagate.trainer import Trainer

        assert Trainer._fp8_module_filter(self._linear(), "lm_head") is False

    def test_embed_named_linear_excluded(self):
        from backpropagate.trainer import Trainer

        assert (
            Trainer._fp8_module_filter(self._linear(), "model.embed_tokens")
            is False
        )

    def test_embedding_module_excluded(self):
        """A real nn.Embedding is excluded (not an nn.Linear at all)."""
        from backpropagate.trainer import Trainer

        assert (
            Trainer._fp8_module_filter(self._embedding(), "model.embed_tokens")
            is False
        )

    def test_non_linear_excluded(self):
        """A non-Linear module at a base FQN is still excluded (isinstance gate)."""
        from backpropagate.trainer import Trainer

        assert (
            Trainer._fp8_module_filter(
                self._embedding(), "model.layers.0.self_attn.q_proj"
            )
            is False
        )


# =============================================================================
# _apply_fp8_to_base
# =============================================================================


class TestApplyFp8ToBase:
    """v1.5 T2.1: the conversion driver called from load_model() post-LoRA."""

    def test_noop_when_not_effective(self):
        """Inactive FP8 → no torchao import, no model mutation."""
        trainer = _cpu_trainer(fp8=False)
        assert trainer._fp8_effective is False
        sentinel = MagicMock()
        trainer._model = sentinel
        # Should return immediately without touching torchao or the model.
        trainer._apply_fp8_to_base()
        assert trainer._model is sentinel  # untouched

    def test_conversion_failure_degrades_to_bf16(self, caplog):
        """A convert_to_float8_training exception → bf16 degrade + WARN, no raise."""
        trainer = _cpu_trainer()
        trainer._fp8_effective = True  # force the active path
        trainer._model = MagicMock()

        # Mock torchao.float8 so the import succeeds but convert raises.
        fake_float8 = MagicMock()
        fake_float8.Float8LinearConfig = MagicMock(return_value=MagicMock())
        fake_float8.convert_to_float8_training = MagicMock(
            side_effect=RuntimeError("scaled_mm not supported")
        )
        with caplog.at_level(logging.WARNING, logger="backpropagate.trainer"):
            with patch.dict(
                sys.modules, {"torchao.float8": fake_float8, "torchao": MagicMock()}
            ):
                # No raise — the conversion failure is a graceful degrade.
                trainer._apply_fp8_to_base()
        assert trainer._fp8_effective is False
        assert any(
            "falling back" in r.message.lower()
            or "fall back" in r.message.lower()
            for r in caplog.records
        )

    def test_zero_converted_degrades(self):
        """convert succeeds but matches 0 base linears → honest bf16 degrade."""
        trainer = _cpu_trainer()
        trainer._fp8_effective = True
        # A model whose named_modules has nothing the Float8Linear isinstance
        # check matches → converted count 0.
        model = MagicMock()
        model.named_modules.return_value = [("a", MagicMock()), ("b", MagicMock())]
        trainer._model = model

        fake_float8 = MagicMock()
        fake_float8.Float8LinearConfig = MagicMock(return_value=MagicMock())
        fake_float8.convert_to_float8_training = MagicMock()
        fake_linear_mod = MagicMock()
        # Float8Linear is a class nothing in named_modules is an instance of.
        fake_linear_mod.Float8Linear = type("Float8Linear", (), {})
        with patch.dict(
            sys.modules,
            {
                "torchao.float8": fake_float8,
                "torchao.float8.float8_linear": fake_linear_mod,
                "torchao": MagicMock(),
            },
        ):
            trainer._apply_fp8_to_base()
        assert trainer._fp8_effective is False

    def test_broken_torchao_import_raises_runtime_fp8_unsupported(self):
        """import torchao.float8 failing (broken install) → RUNTIME_FP8_UNSUPPORTED.

        The gate already promised FP8 (find_spec succeeded), so an import that
        then fails is a contradictory, unrecoverable state — a structured hard
        error, NOT a silent bf16 degrade.
        """
        from backpropagate.exceptions import TrainingError

        trainer = _cpu_trainer()
        trainer._fp8_effective = True
        trainer._model = MagicMock()

        # Make the torchao.float8 import raise ImportError.
        real_import = __import__

        def _boom(name, *args, **kwargs):
            if name == "torchao.float8" or name.startswith("torchao.float8"):
                raise ImportError("DLL load failed: torchao C++ ext missing")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_boom):
            with pytest.raises(TrainingError) as exc_info:
                trainer._apply_fp8_to_base()
        assert exc_info.value.code == "RUNTIME_FP8_UNSUPPORTED"
        # And the effective flag is cleared on the way out.
        assert trainer._fp8_effective is False


# =============================================================================
# Hyperparameter provenance + run-history honesty
# =============================================================================


class TestFp8Provenance:
    """v1.5 T2.1: _fp8_effective (not the request) is the persisted truth."""

    def test_degraded_fp8_resolves_effective_false(self):
        """fp8=True that degraded → self._fp8_effective is the False that persists.

        The hyperparameters dict stamps ``self._fp8_effective``; a CPU run that
        requested fp8 but degraded must report False (honest provenance), even
        though ``self.fp8`` (the request) is True.
        """
        trainer = _cpu_trainer(fp8=True)
        assert trainer.fp8 is True  # the request
        assert trainer._fp8_effective is False  # what actually ran

    def test_use_rslora_threads_to_attribute(self):
        """use_rslora resolves onto the instance for the LoraConfig wiring + history."""
        trainer = _cpu_trainer(use_rslora=True)
        assert trainer.use_rslora is True
