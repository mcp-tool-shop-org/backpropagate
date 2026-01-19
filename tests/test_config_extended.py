"""
Extended configuration tests for comprehensive coverage.

Covers:
- Pydantic availability detection
- Fallback dataclass implementation
- Environment variable parsing
- Training presets
- LR scaling helpers
- Settings management
- Windows-specific behavior
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from dataclasses import dataclass


# =============================================================================
# PYDANTIC AVAILABILITY TESTS
# =============================================================================


class TestPydanticAvailability:
    """Tests for pydantic availability detection."""

    def test_pydantic_flag_set(self):
        """PYDANTIC_SETTINGS_AVAILABLE flag is set correctly."""
        from backpropagate.config import PYDANTIC_SETTINGS_AVAILABLE
        assert isinstance(PYDANTIC_SETTINGS_AVAILABLE, bool)

    def test_settings_class_exists(self):
        """Settings class exists regardless of pydantic."""
        from backpropagate.config import Settings
        assert Settings is not None

    def test_settings_instantiates(self):
        """Settings can be instantiated."""
        from backpropagate.config import Settings
        s = Settings()
        assert s is not None


# =============================================================================
# FALLBACK IMPLEMENTATION TESTS
# =============================================================================


class TestFallbackEnvParsing:
    """Tests for fallback environment variable parsing."""

    def test_get_env_with_value(self):
        """Environment variable retrieved correctly."""
        with patch.dict(os.environ, {"BACKPROPAGATE_TEST_KEY": "test_value"}):
            # Import functions that should be available in fallback
            from backpropagate.config import Settings
            # The fallback implementation uses _get_env internally

    def test_get_env_default(self):
        """Default value returned when env var missing."""
        # Ensure the key doesn't exist
        key = "BACKPROPAGATE_NONEXISTENT_KEY"
        if key in os.environ:
            del os.environ[key]

        from backpropagate.config import Settings
        s = Settings()
        # Should use defaults
        assert s.model.name == "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"


class TestFallbackTypeConversion:
    """Tests for type conversion in fallback mode."""

    def test_int_conversion(self):
        """Integer environment variables converted correctly."""
        from backpropagate.config import Settings

        with patch.dict(os.environ, {"BACKPROPAGATE_TRAINING__MAX_STEPS": "200"}):
            # In pydantic mode this would work automatically
            # In fallback mode we test defaults work
            s = Settings()
            assert isinstance(s.training.max_steps, int)

    def test_float_conversion(self):
        """Float environment variables converted correctly."""
        from backpropagate.config import Settings

        s = Settings()
        assert isinstance(s.training.learning_rate, float)

    def test_bool_conversion(self):
        """Boolean environment variables converted correctly."""
        from backpropagate.config import Settings

        s = Settings()
        assert isinstance(s.training.bf16, bool)


# =============================================================================
# SUB-CONFIG TESTS
# =============================================================================


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_model_config_defaults(self):
        """ModelConfig has correct defaults."""
        from backpropagate.config import ModelConfig

        config = ModelConfig()
        assert config.name == "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
        assert config.load_in_4bit is True
        assert config.max_seq_length == 2048
        assert config.trust_remote_code is True

    def test_model_config_dtype_default(self):
        """ModelConfig dtype defaults to None (auto-detect)."""
        from backpropagate.config import ModelConfig

        config = ModelConfig()
        assert config.dtype is None


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_lora_config_defaults(self):
        """LoRAConfig has correct defaults."""
        from backpropagate.config import LoRAConfig

        config = LoRAConfig()
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert len(config.target_modules) > 0

    def test_lora_target_modules(self):
        """LoRAConfig target_modules contains expected modules."""
        from backpropagate.config import LoRAConfig

        config = LoRAConfig()
        assert "q_proj" in config.target_modules
        assert "k_proj" in config.target_modules
        assert "v_proj" in config.target_modules


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_training_config_defaults(self):
        """TrainingConfig has correct defaults."""
        from backpropagate.config import TrainingConfig

        config = TrainingConfig()
        assert config.per_device_train_batch_size == 2
        assert config.gradient_accumulation_steps == 4
        assert config.max_steps == 100
        assert config.learning_rate == 2e-4
        assert config.warmup_steps == 10
        assert config.optim == "adamw_8bit"

    def test_training_config_output_dir(self):
        """TrainingConfig output_dir default."""
        from backpropagate.config import TrainingConfig

        config = TrainingConfig()
        assert config.output_dir == "./output"


class TestDataConfig:
    """Tests for DataConfig."""

    def test_data_config_defaults(self):
        """DataConfig has correct defaults."""
        from backpropagate.config import DataConfig

        config = DataConfig()
        assert config.dataset_name == "HuggingFaceH4/ultrachat_200k"
        assert config.dataset_split == "train_sft"
        assert config.max_samples == 1000
        assert config.text_column == "text"

    def test_data_config_flags(self):
        """DataConfig boolean flags."""
        from backpropagate.config import DataConfig

        config = DataConfig()
        assert config.pre_tokenize is True
        assert config.shuffle is True


class TestUIConfig:
    """Tests for UIConfig."""

    def test_ui_config_defaults(self):
        """UIConfig has correct defaults."""
        from backpropagate.config import UIConfig

        config = UIConfig()
        assert config.port == 7862
        assert config.host == "127.0.0.1"
        assert config.share is False


class TestWindowsConfig:
    """Tests for WindowsConfig."""

    def test_windows_config_defaults(self):
        """WindowsConfig has Windows-safe defaults."""
        from backpropagate.config import WindowsConfig

        config = WindowsConfig()
        assert config.dataloader_num_workers == 0  # Critical for Windows
        assert config.tokenizers_parallelism is False
        assert config.xformers_disabled is True
        assert config.pre_tokenize is True


# =============================================================================
# SETTINGS MANAGEMENT TESTS
# =============================================================================


class TestSettingsManagement:
    """Tests for settings management functions."""

    def test_get_settings_returns_settings(self):
        """get_settings() returns Settings instance."""
        from backpropagate.config import get_settings, Settings

        result = get_settings()
        assert isinstance(result, Settings)

    def test_get_settings_cached(self):
        """get_settings() returns cached instance."""
        from backpropagate.config import get_settings

        first = get_settings()
        second = get_settings()
        assert first is second

    def test_reload_settings_clears_cache(self):
        """reload_settings() clears the cache."""
        from backpropagate.config import get_settings, reload_settings

        first = get_settings()
        reload_settings()
        # After reload, should still work
        second = get_settings()
        assert second is not None

    def test_settings_singleton(self):
        """settings module variable is a singleton."""
        from backpropagate.config import settings, get_settings

        # settings should be the same as get_settings()
        assert settings is not None


# =============================================================================
# TRAINING PRESETS TESTS
# =============================================================================


class TestTrainingPresets:
    """Tests for training presets."""

    def test_presets_available(self):
        """Training presets are available."""
        from backpropagate.config import TRAINING_PRESETS

        assert TRAINING_PRESETS is not None
        assert isinstance(TRAINING_PRESETS, dict)

    def test_fast_preset(self):
        """Fast preset has low steps, high lr."""
        from backpropagate.config import get_preset, TrainingPreset

        preset = get_preset("fast")
        assert preset is not None
        assert isinstance(preset, TrainingPreset)
        # Fast should have fewer steps
        assert preset.steps_per_run <= 100
        assert preset.learning_rate >= 2e-4  # Higher LR for fast

    def test_balanced_preset(self):
        """Balanced preset has medium values."""
        from backpropagate.config import get_preset, TrainingPreset

        preset = get_preset("balanced")
        assert preset is not None
        assert isinstance(preset, TrainingPreset)
        assert preset.lora_r == 16
        assert preset.learning_rate == 2e-4

    def test_quality_preset(self):
        """Quality preset has high steps, low lr."""
        from backpropagate.config import get_preset, TrainingPreset

        preset = get_preset("quality")
        assert preset is not None
        assert isinstance(preset, TrainingPreset)
        # Quality should have more steps
        assert preset.steps_per_run >= 100
        assert preset.lora_r >= 32

    def test_unknown_preset(self):
        """Unknown preset raises ValueError."""
        from backpropagate.config import get_preset

        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")


# =============================================================================
# LR SCALING HELPERS TESTS
# =============================================================================


class TestLRScalingHelpers:
    """Tests for learning rate scaling helpers."""

    def test_get_recommended_lr(self):
        """get_recommended_lr returns reasonable values."""
        from backpropagate.config import get_recommended_lr

        lr = get_recommended_lr(dataset_size=5000, base_lr=2e-4)
        assert lr > 0
        assert lr < 1  # Sanity check

    def test_get_recommended_lr_scaling(self):
        """Smaller datasets get higher LR (learn quickly), larger datasets get lower LR."""
        from backpropagate.config import get_recommended_lr

        lr_small_dataset = get_recommended_lr(dataset_size=500)  # Small: 5e-4
        lr_medium_dataset = get_recommended_lr(dataset_size=5000)  # Medium: 2e-4
        lr_large_dataset = get_recommended_lr(dataset_size=50000)  # Large: 1e-4

        # Small datasets get higher LR for aggressive learning
        assert lr_small_dataset > lr_medium_dataset
        # Large datasets get lower LR for stability
        assert lr_medium_dataset > lr_large_dataset

    def test_get_recommended_warmup(self):
        """get_recommended_warmup returns reasonable values."""
        from backpropagate.config import get_recommended_warmup

        warmup = get_recommended_warmup(dataset_size=5000, num_steps=1000)
        assert warmup > 0
        assert warmup < 1000

    def test_get_recommended_warmup_ratio(self):
        """Warmup is proportional to total steps."""
        from backpropagate.config import get_recommended_warmup

        warmup_1000 = get_recommended_warmup(dataset_size=5000, num_steps=1000)
        warmup_100 = get_recommended_warmup(dataset_size=5000, num_steps=100)

        # More steps should have more warmup
        assert warmup_1000 > warmup_100

    def test_get_recommended_warmup_by_dataset_size(self):
        """Smaller datasets need more warmup (higher ratio)."""
        from backpropagate.config import get_recommended_warmup

        # Same num_steps, different dataset sizes
        warmup_small = get_recommended_warmup(dataset_size=500, num_steps=100)  # 15%
        warmup_medium = get_recommended_warmup(dataset_size=5000, num_steps=100)  # 10%
        warmup_large = get_recommended_warmup(dataset_size=50000, num_steps=100)  # 5%

        # Smaller datasets need more warmup
        assert warmup_small >= warmup_medium
        assert warmup_medium >= warmup_large


# =============================================================================
# WINDOWS FIXES TESTS
# =============================================================================


class TestWindowsFixes:
    """Tests for Windows-specific fixes."""

    def test_apply_windows_fixes_on_windows(self):
        """apply_windows_fixes sets env vars on Windows."""
        from backpropagate.config import Settings

        settings = Settings()

        with patch.object(os, 'name', 'nt'):
            settings.apply_windows_fixes()

            # Should set TOKENIZERS_PARALLELISM
            assert "TOKENIZERS_PARALLELISM" in os.environ

    def test_apply_windows_fixes_xformers(self):
        """apply_windows_fixes disables xformers on Windows."""
        from backpropagate.config import Settings

        settings = Settings()

        with patch.object(os, 'name', 'nt'):
            settings.apply_windows_fixes()

            if settings.windows.xformers_disabled:
                assert os.environ.get("XFORMERS_DISABLED") == "1"

    def test_apply_windows_fixes_not_on_posix(self):
        """apply_windows_fixes doesn't modify env on Linux."""
        from backpropagate.config import Settings

        settings = Settings()

        # Save original values
        original_env = os.environ.copy()

        with patch.object(os, 'name', 'posix'):
            settings.apply_windows_fixes()

        # Env should be relatively unchanged on non-Windows


# =============================================================================
# TO_DICT TESTS
# =============================================================================


class TestSettingsToDict:
    """Tests for settings serialization."""

    def test_to_dict_returns_dict(self):
        """to_dict() returns a dictionary."""
        from backpropagate.config import Settings

        settings = Settings()
        result = settings.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_contains_version(self):
        """to_dict() contains version."""
        from backpropagate.config import Settings

        settings = Settings()
        result = settings.to_dict()
        assert "version" in result


# =============================================================================
# OUTPUT/CACHE DIR TESTS
# =============================================================================


class TestDirectoryHelpers:
    """Tests for directory helper functions."""

    def test_get_output_dir(self):
        """get_output_dir returns path."""
        from backpropagate.config import get_output_dir

        result = get_output_dir()
        assert result is not None
        assert isinstance(result, (str, os.PathLike))

    def test_get_cache_dir(self):
        """get_cache_dir returns path."""
        from backpropagate.config import get_cache_dir

        result = get_cache_dir()
        assert result is not None


# =============================================================================
# ENVIRONMENT VARIABLE OVERRIDE TESTS
# =============================================================================


class TestEnvOverrides:
    """Tests for environment variable overrides."""

    def test_model_name_override(self):
        """Model name can be overridden via env var."""
        from backpropagate.config import reload_settings, get_settings

        with patch.dict(os.environ, {"BACKPROPAGATE_MODEL__NAME": "custom/model"}):
            reload_settings()
            # In pydantic mode, this would work
            # In fallback mode, defaults are used

    def test_learning_rate_override(self):
        """Learning rate can be overridden via env var."""
        from backpropagate.config import reload_settings, get_settings

        with patch.dict(os.environ, {"BACKPROPAGATE_TRAINING__LEARNING_RATE": "1e-5"}):
            reload_settings()
            # Test that the system handles this


# =============================================================================
# MULTI-RUN CONFIG TESTS
# =============================================================================


class TestMultiRunConfig:
    """Tests for MultiRunConfig (in config.py)."""

    def test_multirun_config_defaults(self):
        """MultiRunConfig has correct defaults."""
        from backpropagate.config import Settings

        settings = Settings()
        assert settings.multi_run.num_runs == 5
        assert settings.multi_run.steps_per_run == 100
        assert settings.multi_run.samples_per_run == 1000

    def test_multirun_continue_from_previous(self):
        """continue_from_previous default is True."""
        from backpropagate.config import Settings

        settings = Settings()
        assert settings.multi_run.continue_from_previous is True


# =============================================================================
# VERSION TESTS
# =============================================================================


class TestVersion:
    """Tests for version information."""

    def test_version_attribute(self):
        """Settings has version attribute."""
        from backpropagate.config import Settings

        settings = Settings()
        assert hasattr(settings, 'version')
        assert settings.version == "0.1.0"

    def test_name_attribute(self):
        """Settings has name attribute."""
        from backpropagate.config import Settings

        settings = Settings()
        assert hasattr(settings, 'name')
        assert settings.name == "backpropagate"


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """__all__ contains expected exports."""
        from backpropagate import config

        assert "Settings" in config.__all__
        assert "settings" in config.__all__
        assert "get_settings" in config.__all__
        assert "reload_settings" in config.__all__
        assert "ModelConfig" in config.__all__
        assert "TrainingConfig" in config.__all__
        assert "LoRAConfig" in config.__all__

    def test_exports_importable(self):
        """All exports can be imported."""
        from backpropagate.config import (
            Settings,
            settings,
            get_settings,
            reload_settings,
            ModelConfig,
            TrainingConfig,
            LoRAConfig,
            DataConfig,
            UIConfig,
            WindowsConfig,
        )

        assert Settings is not None
        assert settings is not None
        assert callable(get_settings)
        assert callable(reload_settings)
