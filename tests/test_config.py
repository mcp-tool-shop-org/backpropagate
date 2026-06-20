"""Tests for configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# =============================================================================
# BASIC IMPORTS AND SETTINGS
# =============================================================================

def test_settings_import():
    """Test that settings can be imported."""
    from backpropagate import Settings, settings
    assert settings is not None
    assert isinstance(settings, Settings)


def test_settings_defaults_pin_user_facing_contract():
    """Pin the operator-facing default configuration.

    These defaults are what a user gets when they run ``Trainer()``
    with no overrides. They are reproduced in the README, the
    handbook's Quick Start, and the v1.3 release notes. A silent drift
    here is a documentation drift everywhere — the assertion messages
    name the doctrine each default carries so a failed diff reads as
    "what changed and why it matters" instead of just two numbers.
    """
    from backpropagate import settings

    # Model defaults — Qwen 2.5 is the default model family
    assert "Qwen" in settings.model.name or "qwen" in settings.model.name.lower(), (
        f"Default model family drifted from Qwen: settings.model.name="
        f"{settings.model.name!r}. The Quick Start README + handbook "
        f"document Qwen as the default; switching families is a "
        f"user-visible behavior change."
    )
    assert settings.model.load_in_4bit is True, (
        f"settings.model.load_in_4bit default flipped from True to "
        f"{settings.model.load_in_4bit}. 4-bit load is what makes 7B "
        f"trainable on 16GB VRAM (RTX 5080 baseline); flipping it "
        f"breaks the headline 'one-line 7B fine-tune' promise."
    )
    assert settings.model.max_seq_length == 2048, (
        f"settings.model.max_seq_length default drifted: expected 2048, "
        f"got {settings.model.max_seq_length}. This is the documented "
        f"context length; bumping it raises VRAM cost for every default "
        f"install."
    )

    # Training defaults
    assert settings.training.learning_rate == 2e-4, (
        f"settings.training.learning_rate default drifted: expected "
        f"2e-4, got {settings.training.learning_rate}. The v1.3 LoRA "
        f"preset story (BACKEND-1) calibrates around 2e-4 as the "
        f"'quality' default; changing it without updating the preset "
        f"matrix breaks the documented behavior."
    )
    assert settings.training.per_device_train_batch_size == 2, (
        f"per_device_train_batch_size default drifted: expected 2, got "
        f"{settings.training.per_device_train_batch_size}. The 'fits "
        f"7B on 16GB' contract assumes batch=2 + grad-accum=4."
    )
    assert settings.training.gradient_accumulation_steps == 4, (
        f"gradient_accumulation_steps default drifted: expected 4, got "
        f"{settings.training.gradient_accumulation_steps}. Effective "
        f"batch is 2 * 4 = 8; changing one half without the other "
        f"breaks documented convergence."
    )

    # LoRA defaults — v1.3 BACKEND-1: bumped from rank 16 (q+v target,
    # 1x LR) to rank 256 (all-linear target). Operators who want the
    # old behavior pass --lora-preset=fast.
    assert settings.lora.r == 256, (
        f"settings.lora.r default drifted from the v1.3 BACKEND-1 "
        f"'quality' contract: expected 256, got {settings.lora.r}. v1.3 "
        f"bumped from rank 16 (q+v target) to rank 256 (all-linear "
        f"target) to ship a stronger out-of-box result. If you're "
        f"reverting to rank 16, that's a v2.0-grade default change — "
        f"update the handbook lora_presets.md + RUNTIME tab simultaneously."
    )
    assert settings.lora.lora_alpha == 512, (
        f"settings.lora.lora_alpha default drifted: expected 512 (2x r), "
        f"got {settings.lora.lora_alpha}. The 2:1 alpha:r ratio is the "
        f"documented effective-LR coupling; breaking it changes "
        f"convergence behavior without notice."
    )


def test_feature_flags():
    """Test feature flag detection."""
    from backpropagate import FEATURES

    assert isinstance(FEATURES, dict)
    assert "unsloth" in FEATURES
    assert "ui" in FEATURES
    assert "validation" in FEATURES


def test_get_gpu_info():
    """Test GPU info retrieval."""
    from backpropagate import get_gpu_info

    info = get_gpu_info()
    assert isinstance(info, dict)
    assert "available" in info


def test_get_system_info():
    """Test system info retrieval."""
    from backpropagate import get_system_info

    info = get_system_info()
    assert isinstance(info, dict)
    assert "python_version" in info
    assert "platform" in info
    assert "features" in info


def test_training_args():
    """Test training arguments generation."""
    from backpropagate import get_training_args

    args = get_training_args()
    assert isinstance(args, dict)
    assert "learning_rate" in args
    assert "per_device_train_batch_size" in args
    assert "bf16" in args


def test_version():
    """Test version is defined."""
    from backpropagate import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)
    assert "." in __version__


# =============================================================================
# PYDANTIC SETTINGS AVAILABILITY
# =============================================================================

class TestPydanticSettingsAvailability:
    """Tests for PYDANTIC_SETTINGS_AVAILABLE flag."""

    def test_pydantic_settings_available_is_bool(self):
        """Test that PYDANTIC_SETTINGS_AVAILABLE is boolean."""
        from backpropagate.config import PYDANTIC_SETTINGS_AVAILABLE
        assert isinstance(PYDANTIC_SETTINGS_AVAILABLE, bool)

    def test_pydantic_settings_import_handling(self):
        """Test that config module handles import errors gracefully.

        This tests lines 44-49:
            try:
                from pydantic import Field
                from pydantic_settings import BaseSettings, SettingsConfigDict
                PYDANTIC_SETTINGS_AVAILABLE = True
            except ImportError:
                PYDANTIC_SETTINGS_AVAILABLE = False
        """
        # We can't truly mock the import, but we can verify the flag exists
        from backpropagate import config
        assert hasattr(config, "PYDANTIC_SETTINGS_AVAILABLE")


# =============================================================================
# SETTINGS SUB-CONFIGURATIONS
# =============================================================================

class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        from backpropagate.config import ModelConfig

        config = ModelConfig()
        assert config.name == "Qwen/Qwen2.5-7B-Instruct"
        assert config.load_in_4bit is True
        assert config.max_seq_length == 2048
        assert config.dtype is None
        assert config.trust_remote_code is True

    def test_model_config_attributes(self):
        """Test ModelConfig has all expected attributes."""
        from backpropagate.config import ModelConfig

        config = ModelConfig()
        assert hasattr(config, "name")
        assert hasattr(config, "load_in_4bit")
        assert hasattr(config, "max_seq_length")
        assert hasattr(config, "dtype")
        assert hasattr(config, "trust_remote_code")


class TestLoRAConfig:
    """Tests for LoRAConfig class."""

    def test_lora_config_defaults(self):
        """Test LoRAConfig default values.

        v1.3 BACKEND-1: defaults bumped from rank 16 (q+v target, 1x
        LR) to rank 256 (all-linear target, 10x LR) per
        Biderman 2024 + Thinking Machines 2025 — the rank-16 defaults
        left ~15-20% post-training quality on the table. Operators who
        want the speed-tilted old defaults pass --lora-preset=fast or
        construct a LoRAConfig with the v1.2 shape explicitly.
        """
        from backpropagate.config import LoRAConfig

        config = LoRAConfig()
        assert config.r == 256, (
            f"LoRAConfig.r default drifted from v1.3 BACKEND-1 'quality' "
            f"contract: expected 256, got {config.r}. The rank 16 → 256 "
            f"bump is sourced from Biderman 2024 + Thinking Machines "
            f"2025; reverting requires updating the handbook + cli "
            f"parser default + trainer init test simultaneously."
        )
        assert config.lora_alpha == 512, (
            f"LoRAConfig.lora_alpha default drifted: expected 512 (2x r), "
            f"got {config.lora_alpha}. The 2:1 ratio is documented "
            f"effective-LR coupling."
        )
        assert config.lora_dropout == 0.05, (
            f"LoRAConfig.lora_dropout default drifted: expected 0.05, "
            f"got {config.lora_dropout}."
        )
        assert config.use_gradient_checkpointing == "unsloth", (
            f"LoRAConfig.use_gradient_checkpointing default drifted from "
            f"'unsloth' (the optimized path) to "
            f"{config.use_gradient_checkpointing!r}."
        )
        assert config.random_state == 42, (
            f"LoRAConfig.random_state default drifted from 42 to "
            f"{config.random_state}. Determinism on first-run results "
            f"breaks when this changes."
        )
        # v1.3 BACKEND-3 / BACKEND-6: new fields default OFF for
        # backward-compat with v1.2 behavior. Operators opt in via
        # --use-dora / --init-lora-weights {pissa,loftq}.
        assert config.use_dora is False, (
            f"LoRAConfig.use_dora default flipped from False to "
            f"{config.use_dora}. v1.3 BACKEND-3 chose opt-in for DoRA "
            f"to preserve v1.2 behavior; flipping the default is a "
            f"breaking change without a CHANGELOG note."
        )
        assert config.init_lora_weights == "default", (
            f"LoRAConfig.init_lora_weights drifted from 'default' to "
            f"{config.init_lora_weights!r}. v1.3 BACKEND-6 added "
            f"pissa/loftq as opt-in; flipping the default breaks "
            f"backward-compat with v1.2 first-run weights."
        )

    def test_lora_config_target_modules(self):
        """Test LoRAConfig target_modules default.

        v1.3 BACKEND-1: default flipped from a hand-curated 7-module
        list to PEFT's ``"all-linear"`` wildcard (matches every linear/
        Conv1D module except the LM head). Wider adaptation surface =
        better quality at ~2-3x LoRA parameter count.
        """
        from backpropagate.config import LoRAConfig

        config = LoRAConfig()
        assert config.target_modules == "all-linear"


class TestUseRsloraConfig:
    """v1.5 T2.3 (rsLoRA, finding 19): the ``use_rslora`` LoRAConfig field.

    rsLoRA scales the adapter by alpha/sqrt(r) instead of alpha/r; its benefit
    grows with rank (relevant at the rank-256 default) at zero inference cost
    and is merge-safe. Default OFF for backward-compat; opt in via the field,
    ``BACKPROPAGATE_LORA__USE_RSLORA``, or ``Trainer(use_rslora=True)``.
    """

    def test_use_rslora_default_false(self):
        """use_rslora defaults False — adapter scaling stays alpha/r (pre-v1.5)."""
        from backpropagate.config import LoRAConfig

        assert LoRAConfig().use_rslora is False

    def test_use_rslora_accepts_true(self):
        """use_rslora=True round-trips on the field."""
        from backpropagate.config import LoRAConfig

        assert LoRAConfig(use_rslora=True).use_rslora is True

    def test_use_rslora_via_env(self, monkeypatch):
        """BACKPROPAGATE_LORA__USE_RSLORA=true round-trips (pydantic env prefix)."""
        from backpropagate.config import LoRAConfig

        monkeypatch.setenv("BACKPROPAGATE_LORA__USE_RSLORA", "true")
        assert LoRAConfig().use_rslora is True

    def test_use_rslora_dataclass_fallback_parity(self):
        """The dataclass-fallback LoRAConfig must default use_rslora False too.

        Re-execs config.py with pydantic_settings blocked so the dataclass
        branch materialises (mirrors TestOrpoDataclassFallback). Guards a
        byte-identical default across the two installs.
        """
        import importlib
        import sys
        import types

        import backpropagate.config as cfg

        source = Path(cfg.__file__).read_text(encoding="utf-8")
        fake = types.ModuleType("backpropagate._config_dataclass_probe_rslora")
        fake.__dict__["__file__"] = cfg.__file__
        fake.__dict__["__name__"] = "backpropagate.config"
        sys.modules.setdefault(
            "backpropagate", importlib.import_module("backpropagate")
        )
        with patch.dict(sys.modules, {"pydantic_settings": None}):
            try:
                exec(compile(source, cfg.__file__, "exec"), fake.__dict__)  # noqa: S102
            except Exception:
                pytest.skip("dataclass fallback branch not materialisable here")
        if fake.__dict__.get("PYDANTIC_SETTINGS_AVAILABLE", True):
            pytest.skip("dataclass fallback branch not materialisable here")
        lora_cls = fake.__dict__["LoRAConfig"]
        assert lora_cls().use_rslora is False
        assert lora_cls(use_rslora=True).use_rslora is True


class TestTrainingConfig:
    """Tests for TrainingConfig class."""

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        from backpropagate.config import TrainingConfig

        config = TrainingConfig()
        assert config.per_device_train_batch_size == 2
        assert config.gradient_accumulation_steps == 4
        assert config.max_steps == 100
        assert config.num_train_epochs == 1
        assert config.learning_rate == 2e-4
        assert config.weight_decay == 0.01
        assert config.warmup_steps == 10
        assert config.warmup_ratio == 0.0
        assert config.optim == "adamw_8bit"
        assert config.lr_scheduler_type == "cosine"
        assert config.logging_steps == 10
        assert config.save_steps == 100
        assert config.bf16 is True
        assert config.fp16 is False
        assert config.seed == 42
        assert config.output_dir == "./output"
        assert config.overwrite_output_dir is True

    def test_training_config_rejects_bf16_and_fp16(self):
        """DATA-A-007: bf16=True AND fp16=True must raise at construction.

        transformers rejects the contradiction deep inside SFTConfig
        mid-run; the pydantic validator must surface a structured
        InvalidSettingError (CONFIG_INVALID_SETTING) up front instead.
        """
        from backpropagate.config import TrainingConfig
        from backpropagate.exceptions import InvalidSettingError

        with pytest.raises(InvalidSettingError) as exc_info:
            TrainingConfig(bf16=True, fp16=True)
        assert exc_info.value.code == "CONFIG_INVALID_SETTING"

    def test_training_config_allows_single_precision_flag(self):
        """DATA-A-007 negative side: exactly-one / neither must construct."""
        from backpropagate.config import TrainingConfig

        # bf16 only (the default)
        assert TrainingConfig(bf16=True, fp16=False).bf16 is True
        # fp16 only (older cards)
        assert TrainingConfig(bf16=False, fp16=True).fp16 is True
        # neither (fp32)
        neither = TrainingConfig(bf16=False, fp16=False)
        assert neither.bf16 is False
        assert neither.fp16 is False

    def test_training_config_rejects_bf16_fp16_via_env(self, monkeypatch):
        """DATA-A-007: the contradiction is caught even when set via env vars.

        BaseSettings runs the after-validator on env-driven construction too,
        so a misconfigured ``BACKPROPAGATE_TRAINING__{BF16,FP16}=true`` pair
        fails fast rather than crashing transformers later in the run.
        """
        from backpropagate.config import TrainingConfig
        from backpropagate.exceptions import InvalidSettingError

        monkeypatch.setenv("BACKPROPAGATE_TRAINING__BF16", "true")
        monkeypatch.setenv("BACKPROPAGATE_TRAINING__FP16", "true")
        with pytest.raises(InvalidSettingError):
            TrainingConfig()

    # -------------------------------------------------------------------------
    # v1.5 T1.2 (ORPO) — method selector + orpo_beta
    # -------------------------------------------------------------------------

    def test_training_config_method_default_is_sft(self):
        """v1.5 T1.2: method defaults to 'sft' (byte-identical v1.4 behavior)."""
        from backpropagate.config import TrainingConfig

        config = TrainingConfig()
        assert config.method == "sft", (
            f"TrainingConfig.method default = {config.method!r}; expected "
            f"'sft' so the default training path is unchanged from v1.4."
        )

    def test_training_config_orpo_beta_default(self):
        """v1.5 T1.2: orpo_beta defaults to 0.1 (the ORPO paper headline)."""
        from backpropagate.config import TrainingConfig

        config = TrainingConfig()
        assert config.orpo_beta == 0.1, (
            f"TrainingConfig.orpo_beta default = {config.orpo_beta}; "
            f"expected 0.1."
        )

    def test_training_config_accepts_method_orpo(self):
        """v1.5 T1.2: method='orpo' is accepted."""
        from backpropagate.config import TrainingConfig

        config = TrainingConfig(method="orpo")
        assert config.method == "orpo"

    def test_training_config_rejects_unknown_method(self):
        """v1.5 T1.2: a method outside {'sft','orpo'} raises InvalidSettingError.

        The selector must fail fast with a structured CONFIG_INVALID_SETTING
        so an operator typing --method dpo (not implemented in v1.5) gets an
        actionable code/hint up front rather than a deep dispatch crash.
        """
        from backpropagate.config import TrainingConfig
        from backpropagate.exceptions import InvalidSettingError

        with pytest.raises(InvalidSettingError) as exc_info:
            TrainingConfig(method="dpo")
        assert exc_info.value.code == "CONFIG_INVALID_SETTING"
        assert exc_info.value.setting_name == "method"

    def test_training_config_method_orpo_via_env(self, monkeypatch):
        """v1.5 T1.2: BACKPROPAGATE_TRAINING__METHOD=orpo round-trips.

        Mirrors the existing bf16/fp16 env-override pattern — the pydantic
        prefix wires the env var for free, so the value lands on the field.
        """
        from backpropagate.config import TrainingConfig

        monkeypatch.setenv("BACKPROPAGATE_TRAINING__METHOD", "orpo")
        config = TrainingConfig()
        assert config.method == "orpo"

    def test_training_config_orpo_beta_via_env(self, monkeypatch):
        """v1.5 T1.2: BACKPROPAGATE_TRAINING__ORPO_BETA env override round-trips."""
        from backpropagate.config import TrainingConfig

        monkeypatch.setenv("BACKPROPAGATE_TRAINING__ORPO_BETA", "0.25")
        config = TrainingConfig()
        assert config.orpo_beta == pytest.approx(0.25)

    def test_training_config_rejects_unknown_method_via_env(self, monkeypatch):
        """v1.5 T1.2: a bad method via env var raises the SAME structured error.

        The field is typed ``str`` so the ``_reject_invalid_method``
        after-validator (not pydantic's type machinery) is the gate. A bad
        ``BACKPROPAGATE_TRAINING__METHOD`` therefore surfaces the same
        ``InvalidSettingError`` / ``CONFIG_INVALID_SETTING`` as a bad kwarg,
        rather than a generic ``ValidationError``.
        """
        from backpropagate.config import TrainingConfig
        from backpropagate.exceptions import InvalidSettingError

        monkeypatch.setenv("BACKPROPAGATE_TRAINING__METHOD", "ppo")
        with pytest.raises(InvalidSettingError) as exc_info:
            TrainingConfig()
        assert exc_info.value.code == "CONFIG_INVALID_SETTING"

    def test_training_config_accepts_positive_orpo_beta(self):
        """v1.5 T1.2: a positive orpo_beta constructs (the default 0.1 and a
        custom value both round-trip)."""
        from backpropagate.config import TrainingConfig

        assert TrainingConfig(orpo_beta=0.1).orpo_beta == pytest.approx(0.1)
        assert TrainingConfig(orpo_beta=0.5).orpo_beta == pytest.approx(0.5)

    @pytest.mark.parametrize("bad_beta", [0.0, -1.0, -0.05])
    def test_training_config_rejects_nonpositive_orpo_beta(self, bad_beta):
        """v1.5 T1.2: orpo_beta <= 0 raises InvalidSettingError.

        The field comment promises "Keep > 0 — a non-positive weight
        degenerates ORPO back to plain SFT," but nothing enforced it.
        ``orpo_beta=0`` silently zeroes the odds-ratio term (SFT wearing an
        ORPO label) and a negative beta trains toward the REJECTED
        completion. Both are silent correctness bugs; the validator surfaces
        a structured CONFIG_INVALID_SETTING up front instead.
        """
        from backpropagate.config import TrainingConfig
        from backpropagate.exceptions import InvalidSettingError

        with pytest.raises(InvalidSettingError) as exc_info:
            TrainingConfig(orpo_beta=bad_beta)
        assert exc_info.value.code == "CONFIG_INVALID_SETTING"
        assert exc_info.value.setting_name == "orpo_beta"

    def test_training_config_rejects_nonpositive_orpo_beta_via_env(self, monkeypatch):
        """v1.5 T1.2: a non-positive orpo_beta via env var raises the SAME
        structured error.

        The field is a plain ``float`` so the ``_reject_invalid_orpo_beta``
        after-validator (not pydantic's type machinery) is the gate; a bad
        ``BACKPROPAGATE_TRAINING__ORPO_BETA`` therefore surfaces the same
        ``InvalidSettingError`` / ``CONFIG_INVALID_SETTING`` as a bad kwarg.
        """
        from backpropagate.config import TrainingConfig
        from backpropagate.exceptions import InvalidSettingError

        monkeypatch.setenv("BACKPROPAGATE_TRAINING__ORPO_BETA", "0")
        with pytest.raises(InvalidSettingError) as exc_info:
            TrainingConfig()
        assert exc_info.value.code == "CONFIG_INVALID_SETTING"
        assert exc_info.value.setting_name == "orpo_beta"


class TestOrpoDataclassFallback:
    """v1.5 T1.2: the dataclass-fallback TrainingConfig must match the pydantic
    branch's method/orpo_beta contract byte-for-byte.

    pydantic-settings is installed in CI, so the fallback dataclass branch
    isn't the one ``from backpropagate.config import TrainingConfig`` returns.
    We exercise it directly by re-importing config with PYDANTIC_SETTINGS
    forced off, the same shape the existing fallback-targeting tests use
    (guarded so a future hard-dep on pydantic doesn't break the suite).
    """

    def _load_dataclass_config_module(self):
        """Import a fresh config module with the dataclass fallback active.

        Returns the reloaded module, or ``None`` (caller skips) if the
        fallback can't be materialised in this environment.
        """
        import importlib
        import sys

        import backpropagate.config as cfg

        # Force the fallback branch: patch the module's availability flag and
        # re-exec the dataclass definitions in an isolated module namespace so
        # we don't disturb the already-imported pydantic classes other tests
        # rely on. We compile the source under a patched global.
        source = Path(cfg.__file__).read_text(encoding="utf-8")
        import types

        fake = types.ModuleType("backpropagate._config_dataclass_probe")
        fake.__dict__["__file__"] = cfg.__file__
        # Pre-seed the availability flag to False so the `if
        # PYDANTIC_SETTINGS_AVAILABLE:` branch is skipped and the dataclass
        # fallback executes. The try/except import block re-sets it, so we
        # instead stub the imports to force the except path.
        fake.__dict__["__name__"] = "backpropagate.config"
        sys.modules.setdefault("backpropagate", importlib.import_module("backpropagate"))
        # Block pydantic_settings so the top-level try/except sets
        # PYDANTIC_SETTINGS_AVAILABLE = False and the dataclass branch runs.
        blocked = {"pydantic_settings": None}
        with patch.dict(sys.modules, blocked):
            try:
                exec(compile(source, cfg.__file__, "exec"), fake.__dict__)
            except Exception:
                return None
        if fake.__dict__.get("PYDANTIC_SETTINGS_AVAILABLE", True):
            # Couldn't force the fallback; skip.
            return None
        return fake

    def test_dataclass_method_default_and_orpo_beta(self):
        mod = self._load_dataclass_config_module()
        if mod is None:
            pytest.skip("dataclass fallback branch not materialisable here")
        cfg_cls = mod.__dict__["TrainingConfig"]
        config = cfg_cls()
        assert config.method == "sft"
        assert config.orpo_beta == 0.1

    def test_dataclass_accepts_method_orpo(self):
        mod = self._load_dataclass_config_module()
        if mod is None:
            pytest.skip("dataclass fallback branch not materialisable here")
        cfg_cls = mod.__dict__["TrainingConfig"]
        assert cfg_cls(method="orpo").method == "orpo"

    def test_dataclass_rejects_unknown_method(self):
        mod = self._load_dataclass_config_module()
        if mod is None:
            pytest.skip("dataclass fallback branch not materialisable here")
        cfg_cls = mod.__dict__["TrainingConfig"]
        exc_cls = mod.__dict__["InvalidSettingError"] if "InvalidSettingError" in mod.__dict__ else None
        from backpropagate.exceptions import InvalidSettingError

        with pytest.raises(InvalidSettingError) as exc_info:
            cfg_cls(method="dpo")
        assert exc_info.value.code == "CONFIG_INVALID_SETTING"
        # The fallback raises the same exception class from backpropagate.exceptions.
        if exc_cls is not None:
            assert exc_cls is InvalidSettingError

    def test_dataclass_accepts_positive_orpo_beta(self):
        mod = self._load_dataclass_config_module()
        if mod is None:
            pytest.skip("dataclass fallback branch not materialisable here")
        cfg_cls = mod.__dict__["TrainingConfig"]
        assert cfg_cls(orpo_beta=0.1).orpo_beta == pytest.approx(0.1)

    @pytest.mark.parametrize("bad_beta", [0.0, -1.0])
    def test_dataclass_rejects_nonpositive_orpo_beta(self, bad_beta):
        """The dataclass fallback's __post_init__ must reject orpo_beta <= 0
        with the same structured CONFIG_INVALID_SETTING the pydantic
        _reject_invalid_orpo_beta validator raises — byte-for-byte parity so a
        pydantic-settings-less install can't silently run SFT under an ORPO
        label (beta=0) or train toward the rejected completion (beta<0)."""
        mod = self._load_dataclass_config_module()
        if mod is None:
            pytest.skip("dataclass fallback branch not materialisable here")
        cfg_cls = mod.__dict__["TrainingConfig"]
        from backpropagate.exceptions import InvalidSettingError

        with pytest.raises(InvalidSettingError) as exc_info:
            cfg_cls(orpo_beta=bad_beta)
        assert exc_info.value.code == "CONFIG_INVALID_SETTING"
        assert exc_info.value.setting_name == "orpo_beta"


class TestDataConfig:
    """Tests for DataConfig class."""

    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        from backpropagate.config import DataConfig

        config = DataConfig()
        assert config.dataset_name == "HuggingFaceH4/ultrachat_200k"
        assert config.dataset_split == "train_sft"
        assert config.max_samples == 1000
        assert config.text_column == "text"
        assert config.chat_format == "chatml"
        assert config.pre_tokenize is True
        assert config.shuffle is True
        # v1.3 BACKEND-4: packing default flipped False -> True (single
        # biggest wall-clock lever for SFT: 1.7-3x throughput, attention-
        # backend agnostic). Opt out via --no-packing or
        # BACKPROPAGATE_DATA__PACKING=false.
        assert config.packing is True


class TestUIConfig:
    """Tests for UIConfig class."""

    def test_ui_config_defaults(self):
        """Test UIConfig default values."""
        from backpropagate.config import UIConfig

        config = UIConfig()
        assert config.port == 7862
        assert config.host == "127.0.0.1"
        assert config.share is False
        assert config.auto_open is True


class TestWindowsConfig:
    """Tests for WindowsConfig class."""

    def test_windows_config_defaults(self):
        """Test WindowsConfig default values."""
        from backpropagate.config import WindowsConfig

        config = WindowsConfig()
        assert config.dataloader_num_workers == 0
        assert config.tokenizers_parallelism is False
        assert config.xformers_disabled is True
        assert config.cuda_launch_blocking is False
        assert config.pre_tokenize is True


# =============================================================================
# MAIN SETTINGS CLASS
# =============================================================================

class TestSettings:
    """Tests for main Settings class."""

    def test_settings_has_nested_configs(self):
        """Test Settings has all nested config objects."""
        from backpropagate.config import Settings

        s = Settings()
        assert hasattr(s, "model")
        assert hasattr(s, "training")
        assert hasattr(s, "lora")
        assert hasattr(s, "data")
        assert hasattr(s, "ui")
        assert hasattr(s, "windows")
        assert hasattr(s, "multi_run")

    def test_settings_version_and_name(self):
        """Test Settings version and name."""
        from backpropagate.config import Settings

        s = Settings()
        assert s.version  # dynamic version from package metadata
        assert s.name == "backpropagate"

    def test_settings_to_dict(self):
        """Test Settings.to_dict() method.

        This tests the to_dict method (lines 267-291 in pydantic version
        or 407-408 in dataclass version).
        """
        from backpropagate.config import Settings

        s = Settings()
        d = s.to_dict()

        assert isinstance(d, dict)
        assert "version" in d

        # If pydantic settings, check full dict
        if "model" in d:
            assert "name" in d["model"]
            assert "training" in d
            assert "lora" in d
            assert "data" in d

    def test_settings_apply_windows_fixes_on_windows(self, monkeypatch):
        """Test Settings.apply_windows_fixes() on Windows.

        This tests lines 293-300 (pydantic) or 410-414 (dataclass):
            def apply_windows_fixes(self) -> None:
                if os.name == "nt":
                    os.environ["TOKENIZERS_PARALLELISM"] = ...

        Uses monkeypatch so the env vars set by apply_windows_fixes (which
        mutates os.environ directly) cannot leak past the test boundary —
        monkeypatch.delenv on teardown reverts to the captured prior value.
        """
        from backpropagate.config import Settings

        s = Settings()

        with patch("os.name", "nt"):
            # monkeypatch captures pre-test value; teardown restores it.
            monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

            s.apply_windows_fixes()

            # Check environment variables were set
            assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"

    def test_settings_apply_windows_fixes_xformers_disabled(self, monkeypatch):
        """Test that xformers is disabled on Windows."""
        from backpropagate.config import Settings

        s = Settings()
        # Ensure xformers_disabled is True for this test
        s.windows.xformers_disabled = True

        with patch("os.name", "nt"):
            monkeypatch.delenv("XFORMERS_DISABLED", raising=False)

            s.apply_windows_fixes()

            assert os.environ.get("XFORMERS_DISABLED") == "1"

    def test_settings_apply_windows_fixes_not_on_linux(self, monkeypatch):
        """Test that Windows fixes don't apply on Linux."""
        from backpropagate.config import Settings

        s = Settings()

        with patch("os.name", "posix"):
            # Use monkeypatch.delenv to clear safely — teardown restores.
            monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

            s.apply_windows_fixes()

            # On non-Windows the helper must not set TOKENIZERS_PARALLELISM.
            assert "TOKENIZERS_PARALLELISM" not in os.environ, (
                "apply_windows_fixes() set TOKENIZERS_PARALLELISM on a non-nt "
                "platform — the os.name == 'nt' guard regressed."
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings(self):
        """Test get_settings returns Settings instance."""
        from backpropagate.config import Settings, get_settings

        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_is_cached(self):
        """Test get_settings returns cached instance."""
        from backpropagate.config import get_settings

        s1 = get_settings()
        s2 = get_settings()

        # Should be the exact same object
        assert s1 is s2


class TestReloadSettings:
    """Tests for reload_settings function."""

    def test_reload_settings_clears_cache(self):
        """Test reload_settings clears the cache.

        This tests lines 440-445:
            def reload_settings() -> Settings:
                get_settings.cache_clear()
                global settings
                settings = get_settings()
                return settings
        """
        from backpropagate.config import Settings, get_settings, reload_settings

        s1 = get_settings()
        s2 = reload_settings()

        # Both should be Settings instances
        assert isinstance(s1, Settings)
        assert isinstance(s2, Settings)

    def test_reload_settings_returns_new_instance(self):
        """Test reload_settings returns a (potentially) new instance."""
        from backpropagate.config import Settings, reload_settings

        s = reload_settings()
        assert isinstance(s, Settings)


class TestGetOutputDir:
    """Tests for get_output_dir function."""

    def test_get_output_dir_returns_path(self):
        """Test get_output_dir returns Path.

        This tests lines 448-452:
            def get_output_dir() -> Path:
                output_dir = Path(settings.training.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                return output_dir
        """
        from backpropagate.config import get_output_dir

        output_dir = get_output_dir()
        assert isinstance(output_dir, Path)

    def test_get_output_dir_creates_directory(self, tmp_path):
        """Test get_output_dir creates the directory if it doesn't exist."""
        from backpropagate.config import get_output_dir, settings

        # Temporarily change output_dir
        orig = settings.training.output_dir
        test_dir = tmp_path / "test_output"
        settings.training.output_dir = str(test_dir)

        try:
            result = get_output_dir()
            assert result.exists()
            assert result.is_dir()
        finally:
            settings.training.output_dir = orig


class TestGetCacheDir:
    """Tests for get_cache_dir function."""

    def test_get_cache_dir_returns_path(self):
        """Test get_cache_dir returns Path.

        This tests lines 455-459:
            def get_cache_dir() -> Path:
                cache_dir = Path.home() / ".cache" / "backpropagate"
                cache_dir.mkdir(parents=True, exist_ok=True)
                return cache_dir
        """
        from backpropagate.config import get_cache_dir

        cache_dir = get_cache_dir()
        assert isinstance(cache_dir, Path)

    def test_get_cache_dir_in_home_directory(self):
        """Test get_cache_dir is in home directory."""
        from backpropagate.config import get_cache_dir

        cache_dir = get_cache_dir()
        assert cache_dir.name == "backpropagate"
        assert ".cache" in str(cache_dir)

    def test_get_cache_dir_creates_directory(self):
        """Test get_cache_dir creates the directory."""
        from backpropagate.config import get_cache_dir

        cache_dir = get_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()


class TestGetTrainingArgs:
    """Tests for get_training_args function."""

    def test_get_training_args_returns_dict(self):
        """Test get_training_args returns dict."""
        from backpropagate.config import get_training_args

        args = get_training_args()
        assert isinstance(args, dict)

    def test_get_training_args_has_expected_keys(self):
        """Test get_training_args has all expected keys.

        This tests lines 462-489.
        """
        from backpropagate.config import get_training_args

        args = get_training_args()

        expected_keys = [
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "num_train_epochs",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "warmup_ratio",
            "optim",
            "lr_scheduler_type",
            "logging_steps",
            "save_steps",
            "bf16",
            "fp16",
            "seed",
            "output_dir",
            "overwrite_output_dir",
            "dataloader_num_workers",
        ]

        for key in expected_keys:
            assert key in args, f"Missing key: {key}"

    def test_get_training_args_max_steps_handling(self):
        """Test max_steps is -1 when max_steps is 0."""
        from backpropagate.config import get_training_args, settings

        # Save original
        orig = settings.training.max_steps

        try:
            # Test when max_steps > 0
            settings.training.max_steps = 100
            args = get_training_args()
            assert args["max_steps"] == 100

            # Note: max_steps = 0 would set to -1, but we can't easily test
            # without reloading settings
        finally:
            settings.training.max_steps = orig

    def test_get_training_args_dataloader_workers_on_windows(self):
        """Test dataloader_num_workers varies by OS."""
        from backpropagate.config import get_training_args

        with patch("os.name", "nt"):
            args = get_training_args()
            # On Windows, should use windows config value (0)
            assert args["dataloader_num_workers"] == 0

    def test_get_training_args_dataloader_workers_on_linux(self):
        """Test dataloader_num_workers is 4 on non-Windows."""
        from backpropagate.config import get_training_args

        with patch("os.name", "posix"):
            args = get_training_args()
            assert args["dataloader_num_workers"] == 4


# =============================================================================
# WINDOWS DEFAULTS
# =============================================================================

class TestWindowsDefaults:
    """Tests for WINDOWS_DEFAULTS constant."""

    def test_windows_defaults_exists(self):
        """Test WINDOWS_DEFAULTS dict exists."""
        from backpropagate.config import WINDOWS_DEFAULTS

        assert isinstance(WINDOWS_DEFAULTS, dict)

    def test_windows_defaults_values(self):
        """Test WINDOWS_DEFAULTS has expected values."""
        from backpropagate.config import WINDOWS_DEFAULTS

        assert WINDOWS_DEFAULTS["dataloader_num_workers"] == 0
        assert WINDOWS_DEFAULTS["tokenizers_parallelism"] is False
        assert WINDOWS_DEFAULTS["xformers_disabled"] is True
        assert WINDOWS_DEFAULTS["cuda_launch_blocking"] is False
        assert WINDOWS_DEFAULTS["pre_tokenize"] is True

    def test_windows_defaults_match_dataclass_defaults(self):
        """CONFIG-A-003: WINDOWS_DEFAULTS must not contradict the live
        WindowsConfig dataclass defaults. Pre-fix the dict pinned
        cuda_launch_blocking=True while the dataclass default is False,
        a silent contradiction since no production path reads the dict.
        """
        from backpropagate.config import WINDOWS_DEFAULTS, WindowsConfig

        config = WindowsConfig()
        assert WINDOWS_DEFAULTS["dataloader_num_workers"] == config.dataloader_num_workers
        assert WINDOWS_DEFAULTS["tokenizers_parallelism"] == config.tokenizers_parallelism
        assert WINDOWS_DEFAULTS["xformers_disabled"] == config.xformers_disabled
        assert WINDOWS_DEFAULTS["cuda_launch_blocking"] == config.cuda_launch_blocking
        assert WINDOWS_DEFAULTS["pre_tokenize"] == config.pre_tokenize


# =============================================================================
# MODULE EXPORTS
# =============================================================================

class TestModuleExports:
    """Tests for config module exports."""

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from backpropagate import config

        expected = [
            "Settings",
            "settings",
            "get_settings",
            "reload_settings",
            "get_output_dir",
            "get_cache_dir",
            "ModelConfig",
            "TrainingConfig",
            "LoRAConfig",
            "DataConfig",
            "UIConfig",
            "WindowsConfig",
            "PYDANTIC_SETTINGS_AVAILABLE",
        ]

        for name in expected:
            assert name in config.__all__, f"{name} should be in __all__"

    def test_imports_from_package(self):
        """Test configs can be imported from backpropagate package."""
        from backpropagate import (
            PYDANTIC_SETTINGS_AVAILABLE,
            DataConfig,
            LoRAConfig,
            ModelConfig,
            Settings,
            TrainingConfig,
            get_cache_dir,
            get_output_dir,
            get_settings,
            get_training_args,
            reload_settings,
            settings,
        )

        assert Settings is not None
        assert settings is not None
        assert callable(get_settings)
        assert callable(reload_settings)
        assert callable(get_output_dir)
        assert callable(get_cache_dir)
        assert callable(get_training_args)
        assert ModelConfig is not None
        assert TrainingConfig is not None
        assert LoRAConfig is not None
        assert DataConfig is not None
        assert isinstance(PYDANTIC_SETTINGS_AVAILABLE, bool)


# =============================================================================
# TRAINING PRESETS TESTS
# =============================================================================

class TestTrainingPresets:
    """Tests for training presets (Phase 1.2)."""

    def test_training_presets_exist(self):
        """Test TRAINING_PRESETS dict exists."""
        from backpropagate.config import TRAINING_PRESETS

        assert isinstance(TRAINING_PRESETS, dict)
        assert "fast" in TRAINING_PRESETS
        assert "balanced" in TRAINING_PRESETS
        assert "quality" in TRAINING_PRESETS

    def test_training_preset_dataclass(self):
        """Test TrainingPreset dataclass structure."""
        from backpropagate.config import TrainingPreset

        preset = TrainingPreset(
            name="test",
            description="Test preset",
            lora_r=16,
            lora_alpha=32,
            batch_size=2,
            gradient_accumulation=4,
            learning_rate=2e-4,
            warmup_steps=10,
            steps_per_run=100,
            num_runs=5,
        )

        assert preset.name == "test"
        assert preset.description == "Test preset"
        assert preset.lora_r == 16
        assert preset.lora_alpha == 32
        assert preset.batch_size == 2
        assert preset.gradient_accumulation == 4
        assert preset.learning_rate == 2e-4
        assert preset.warmup_steps == 10
        assert preset.steps_per_run == 100
        assert preset.num_runs == 5

    def test_effective_batch_size_property(self):
        """Test TrainingPreset.effective_batch_size property."""
        from backpropagate.config import TrainingPreset

        preset = TrainingPreset(
            name="test",
            description="Test",
            lora_r=16,
            lora_alpha=32,
            batch_size=2,
            gradient_accumulation=8,
            learning_rate=2e-4,
            warmup_steps=10,
            steps_per_run=100,
            num_runs=5,
        )

        assert preset.effective_batch_size == 16  # 2 * 8

    def test_get_preset_fast(self):
        """Test get_preset for 'fast' preset."""
        from backpropagate.config import get_preset

        preset = get_preset("fast")

        assert preset.name == "fast"
        assert preset.lora_r == 8
        assert preset.lora_alpha == 16
        assert preset.learning_rate == 5e-4
        assert preset.steps_per_run == 50
        assert preset.num_runs == 3

    def test_get_preset_balanced(self):
        """Test get_preset for 'balanced' preset."""
        from backpropagate.config import get_preset

        preset = get_preset("balanced")

        assert preset.name == "balanced"
        assert preset.lora_r == 16
        assert preset.lora_alpha == 32
        assert preset.learning_rate == 2e-4
        assert preset.steps_per_run == 100
        assert preset.num_runs == 5

    def test_get_preset_quality(self):
        """Test get_preset for 'quality' preset."""
        from backpropagate.config import get_preset

        preset = get_preset("quality")

        assert preset.name == "quality"
        assert preset.lora_r == 32
        assert preset.lora_alpha == 64
        assert preset.learning_rate == 1e-4
        assert preset.steps_per_run == 200
        assert preset.num_runs == 10
        assert preset.replay_fraction == 0.1
        assert preset.validate_every_run is True

    def test_get_preset_invalid(self):
        """Test get_preset raises for unknown preset."""
        from backpropagate.config import get_preset

        with pytest.raises(ValueError) as exc_info:
            get_preset("nonexistent")

        assert "Unknown preset" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_preset_optional_fields(self):
        """Test TrainingPreset optional fields have defaults."""
        from backpropagate.config import TrainingPreset

        preset = TrainingPreset(
            name="minimal",
            description="Minimal",
            lora_r=8,
            lora_alpha=16,
            batch_size=2,
            gradient_accumulation=4,
            learning_rate=2e-4,
            warmup_steps=5,
            steps_per_run=50,
            num_runs=3,
        )

        # Optional fields should have defaults
        assert preset.samples_per_run == 1000
        assert preset.replay_fraction == 0.0
        assert preset.validate_every_run is False


# =============================================================================
# LR SCALING TESTS (Phase 1.3)
# =============================================================================

class TestLRScaling:
    """Tests for learning rate scaling helpers."""

    def test_get_recommended_lr_small_dataset(self):
        """Test LR recommendation for small datasets (<1K)."""
        from backpropagate.config import get_recommended_lr

        lr = get_recommended_lr(500)
        assert lr == 5e-4  # Higher LR for small datasets

    def test_get_recommended_lr_medium_dataset(self):
        """Test LR recommendation for medium datasets (1K-10K)."""
        from backpropagate.config import get_recommended_lr

        lr = get_recommended_lr(5000)
        assert lr == 2e-4  # Standard LR

    def test_get_recommended_lr_large_dataset(self):
        """Test LR recommendation for large datasets (>10K)."""
        from backpropagate.config import get_recommended_lr

        lr = get_recommended_lr(50000)
        assert lr == 1e-4  # Lower LR for stability

    def test_get_recommended_lr_boundary_cases(self):
        """Test LR recommendation at boundary values."""
        from backpropagate.config import get_recommended_lr

        # Exactly at 1000 boundary
        lr_999 = get_recommended_lr(999)
        lr_1000 = get_recommended_lr(1000)
        assert lr_999 == 5e-4  # < 1000, high LR
        assert lr_1000 == 2e-4  # >= 1000, standard LR

        # Exactly at 10000 boundary
        lr_9999 = get_recommended_lr(9999)
        lr_10000 = get_recommended_lr(10000)
        assert lr_9999 == 2e-4  # < 10000, standard LR
        assert lr_10000 == 1e-4  # >= 10000, low LR

    def test_get_recommended_lr_custom_base(self):
        """Test LR recommendation with custom base_lr."""
        from backpropagate.config import get_recommended_lr

        # Custom base_lr affects medium dataset return value
        lr = get_recommended_lr(5000, base_lr=3e-4)
        assert lr == 3e-4

    def test_get_recommended_lr_medium_scales_with_base(self):
        """DATA-A-008: the medium branch must honor the ``scale`` multiplier.

        Pre-fix the medium tier returned bare ``base_lr`` while small/large
        used ``* scale``. For the default base that is numerically identical,
        but a custom base must scale the medium anchor (2e-4 * scale) exactly
        like the others — and the small > medium > large ordering must hold.
        """
        from backpropagate.config import get_recommended_lr

        for base in (2e-4, 3e-4, 5e-4, 1e-4):
            scale = base / 2e-4
            small = get_recommended_lr(500, base_lr=base)
            medium = get_recommended_lr(5000, base_lr=base)
            large = get_recommended_lr(50000, base_lr=base)
            assert medium == pytest.approx(2e-4 * scale)
            assert small == pytest.approx(5e-4 * scale)
            assert large == pytest.approx(1e-4 * scale)
            # Ladder is strictly monotone for any positive base.
            assert small > medium > large

    def test_get_recommended_lr_method_sft_unchanged(self):
        """v1.5 T1.2: method='sft' returns the existing SFT ladder UNCHANGED.

        Explicitly passing the default must be byte-identical to omitting it,
        so the ORPO param can never silently shift the SFT recommendation.
        """
        from backpropagate.config import get_recommended_lr

        for size in (500, 5000, 50000):
            assert get_recommended_lr(size, method="sft") == get_recommended_lr(size)

    def test_get_recommended_lr_method_orpo_ladder(self):
        """v1.5 T1.2: method='orpo' returns the ORPO ladder (2e-5/1e-5/5e-6)."""
        from backpropagate.config import get_recommended_lr

        assert get_recommended_lr(500, method="orpo") == 2e-5  # small (<1K)
        assert get_recommended_lr(5000, method="orpo") == 1e-5  # medium (1K-10K)
        assert get_recommended_lr(50000, method="orpo") == 5e-6  # large (>10K)

    def test_get_recommended_lr_orpo_ladder_monotone(self):
        """v1.5 T1.2: the ORPO ladder is strictly monotone small > medium > large."""
        from backpropagate.config import get_recommended_lr

        small = get_recommended_lr(500, method="orpo")
        medium = get_recommended_lr(5000, method="orpo")
        large = get_recommended_lr(50000, method="orpo")
        assert small > medium > large

    def test_get_recommended_lr_orpo_below_sft(self):
        """v1.5 T1.2: ORPO LRs sit an order of magnitude below the SFT ladder.

        ORPO's odds-ratio loss is unstable at SFT LR magnitudes; pin that the
        ladder is materially lower at every tier so a future edit can't
        accidentally re-anchor ORPO onto the SFT values.
        """
        from backpropagate.config import get_recommended_lr

        for size in (500, 5000, 50000):
            assert get_recommended_lr(size, method="orpo") < get_recommended_lr(
                size, method="sft"
            )

    def test_get_recommended_lr_orpo_ignores_base_lr(self):
        """v1.5 T1.2: the ORPO ladder is fixed — base_lr does not scale it."""
        from backpropagate.config import get_recommended_lr

        # Same anchors regardless of base_lr (which only governs the SFT ladder).
        for base in (1e-4, 2e-4, 5e-4):
            assert get_recommended_lr(500, base_lr=base, method="orpo") == 2e-5
            assert get_recommended_lr(5000, base_lr=base, method="orpo") == 1e-5
            assert get_recommended_lr(50000, base_lr=base, method="orpo") == 5e-6


class TestWarmupScaling:
    """Tests for warmup steps scaling helpers."""

    def test_get_recommended_warmup_small_dataset(self):
        """Test warmup recommendation for small datasets (<1K)."""
        from backpropagate.config import get_recommended_warmup

        warmup = get_recommended_warmup(500, num_steps=100)
        assert warmup == 15  # 15% of steps

    def test_get_recommended_warmup_medium_dataset(self):
        """Test warmup recommendation for medium datasets (1K-10K)."""
        from backpropagate.config import get_recommended_warmup

        warmup = get_recommended_warmup(5000, num_steps=100)
        assert warmup == 10  # 10% of steps

    def test_get_recommended_warmup_large_dataset(self):
        """Test warmup recommendation for large datasets (>10K)."""
        from backpropagate.config import get_recommended_warmup

        warmup = get_recommended_warmup(50000, num_steps=100)
        assert warmup == 5  # 5% of steps

    def test_get_recommended_warmup_minimum_one(self):
        """Test warmup is at least 1."""
        from backpropagate.config import get_recommended_warmup

        warmup = get_recommended_warmup(50000, num_steps=10)
        assert warmup >= 1

    def test_get_recommended_warmup_boundary_cases(self):
        """Test warmup at boundary values."""
        from backpropagate.config import get_recommended_warmup

        # At 1000 boundary
        warmup_999 = get_recommended_warmup(999, num_steps=100)
        warmup_1000 = get_recommended_warmup(1000, num_steps=100)
        assert warmup_999 == 15  # 15% for < 1000
        assert warmup_1000 == 10  # 10% for >= 1000

        # At 10000 boundary
        warmup_9999 = get_recommended_warmup(9999, num_steps=100)
        warmup_10000 = get_recommended_warmup(10000, num_steps=100)
        assert warmup_9999 == 10  # 10% for < 10000
        assert warmup_10000 == 5  # 5% for >= 10000


# =============================================================================
# WINDOWS CONFIG ADDITIONAL TESTS
# =============================================================================

class TestWindowsConfigAdvanced:
    """Additional tests for WindowsConfig settings."""

    def test_apply_windows_fixes_cuda_launch_blocking(self, monkeypatch):
        """Test cuda_launch_blocking is applied when True."""
        from backpropagate.config import Settings

        s = Settings()
        s.windows.cuda_launch_blocking = True

        with patch("os.name", "nt"):
            monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)

            s.apply_windows_fixes()

            assert os.environ.get("CUDA_LAUNCH_BLOCKING") == "1"

    def test_apply_windows_fixes_xformers_not_disabled(self, monkeypatch):
        """Test xformers is not disabled when setting is False."""
        from backpropagate.config import Settings

        s = Settings()
        s.windows.xformers_disabled = False

        with patch("os.name", "nt"):
            monkeypatch.delenv("XFORMERS_DISABLED", raising=False)

            s.apply_windows_fixes()

            # Should NOT set XFORMERS_DISABLED when xformers_disabled is False.
            # The code only sets it when True, never unsets when False — so a
            # cleared env at the call site must remain cleared after.
            assert "XFORMERS_DISABLED" not in os.environ, (
                "apply_windows_fixes() set XFORMERS_DISABLED even though "
                "settings.windows.xformers_disabled is False — regression."
            )


class TestMultiRunConfigSettings:
    """Tests for MultiRunConfig in Settings."""

    def test_multi_run_config_exists(self):
        """Test Settings has multi_run config."""
        from backpropagate.config import Settings

        s = Settings()
        assert hasattr(s, "multi_run")

    def test_multi_run_config_defaults(self):
        """Test MultiRunConfig default values."""
        from backpropagate.config import Settings

        s = Settings()
        assert s.multi_run.num_runs == 5
        assert s.multi_run.steps_per_run == 100
        assert s.multi_run.samples_per_run == 1000
        assert s.multi_run.continue_from_previous is True
        assert s.multi_run.save_intermediate is True


# =============================================================================
# PRESET EXPORTS TESTS
# =============================================================================

class TestPresetExports:
    """Tests for preset module exports."""

    def test_exports_in_all(self):
        """Test presets are in __all__."""
        from backpropagate import config

        assert "TrainingPreset" in config.__all__
        assert "TRAINING_PRESETS" in config.__all__
        assert "get_preset" in config.__all__
        assert "get_recommended_lr" in config.__all__
        assert "get_recommended_warmup" in config.__all__

    def test_imports_from_config(self):
        """Test presets can be imported from config."""
        from backpropagate.config import (
            TRAINING_PRESETS,
            TrainingPreset,
            get_preset,
            get_recommended_lr,
            get_recommended_warmup,
        )

        assert TrainingPreset is not None
        assert isinstance(TRAINING_PRESETS, dict)
        assert callable(get_preset)
        assert callable(get_recommended_lr)
        assert callable(get_recommended_warmup)

    def test_imports_from_package(self):
        """Test presets can be imported from backpropagate.config module.

        Note: These are not exported from the top-level backpropagate package,
        but should be accessible from backpropagate.config.
        """
        from backpropagate.config import (
            TRAINING_PRESETS,
            TrainingPreset,
            get_preset,
            get_recommended_lr,
            get_recommended_warmup,
        )

        assert TrainingPreset is not None
        assert TRAINING_PRESETS is not None
        assert get_preset is not None
        assert get_recommended_lr is not None
        assert get_recommended_warmup is not None


# =============================================================================
# STAGE C PROACTIVE AMEND — DATA-B-009 (deprecated env-var scan)
# =============================================================================


class TestStageCDeprecatedEnvVarScan:
    """DATA-B-009: a renamed BACKPROPAGATE_* env var that ``extra="ignore"``
    would silently drop now produces a WARN naming the replacement."""

    def test_deprecated_env_var_detected(self, monkeypatch):
        from backpropagate import config

        # Use a known-deprecated name from the map.
        old = next(iter(config._DEPRECATED_ENV_VARS))
        monkeypatch.setenv(old, "5")
        found = config._warn_deprecated_env_vars()
        assert old in found

    def test_deprecated_env_var_warns_with_replacement(self, monkeypatch, capsys):
        from backpropagate import config

        old, new = next(iter(config._DEPRECATED_ENV_VARS.items()))
        monkeypatch.setenv(old, "5")
        config._warn_deprecated_env_vars()
        # The project routes warnings through structlog, which renders to
        # stdout/stderr; assert the deprecated + replacement names surface.
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert old in combined
        if new:
            assert new in combined

    def test_clean_env_returns_empty(self, monkeypatch):
        from backpropagate import config

        for name in config._DEPRECATED_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        assert config._warn_deprecated_env_vars() == []

    def test_get_settings_runs_scan(self, monkeypatch):
        from backpropagate import config

        old = next(iter(config._DEPRECATED_ENV_VARS))
        monkeypatch.setenv(old, "5")
        config.get_settings.cache_clear()
        called = {}
        real = config._warn_deprecated_env_vars

        def spy():
            called["hit"] = True
            return real()

        monkeypatch.setattr(config, "_warn_deprecated_env_vars", spy)
        config.get_settings()
        assert called.get("hit") is True
        config.get_settings.cache_clear()
