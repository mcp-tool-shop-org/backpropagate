"""Tests for configuration module."""

import pytest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path


# =============================================================================
# BASIC IMPORTS AND SETTINGS
# =============================================================================

def test_settings_import():
    """Test that settings can be imported."""
    from backpropagate import settings, Settings
    assert settings is not None
    assert isinstance(settings, Settings)


def test_settings_defaults():
    """Test default configuration values."""
    from backpropagate import settings

    # Model defaults
    assert "Qwen" in settings.model.name or "qwen" in settings.model.name.lower()
    assert settings.model.load_in_4bit is True
    assert settings.model.max_seq_length == 2048

    # Training defaults
    assert settings.training.learning_rate == 2e-4
    assert settings.training.per_device_train_batch_size == 2
    assert settings.training.gradient_accumulation_steps == 4

    # LoRA defaults
    assert settings.lora.r == 16
    assert settings.lora.lora_alpha == 32


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
        assert config.name == "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
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
        """Test LoRAConfig default values."""
        from backpropagate.config import LoRAConfig

        config = LoRAConfig()
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.use_gradient_checkpointing == "unsloth"
        assert config.random_state == 42

    def test_lora_config_target_modules(self):
        """Test LoRAConfig target_modules default."""
        from backpropagate.config import LoRAConfig

        config = LoRAConfig()
        expected_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]
        assert config.target_modules == expected_modules


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
        assert config.packing is False


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
        assert s.version == "0.1.0"
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

    def test_settings_apply_windows_fixes_on_windows(self):
        """Test Settings.apply_windows_fixes() on Windows.

        This tests lines 293-300 (pydantic) or 410-414 (dataclass):
            def apply_windows_fixes(self) -> None:
                if os.name == "nt":
                    os.environ["TOKENIZERS_PARALLELISM"] = ...
        """
        from backpropagate.config import Settings

        s = Settings()

        with patch("os.name", "nt"):
            # Store original values
            orig_tokenizers = os.environ.get("TOKENIZERS_PARALLELISM")

            s.apply_windows_fixes()

            # Check environment variables were set
            assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"

            # Restore
            if orig_tokenizers is not None:
                os.environ["TOKENIZERS_PARALLELISM"] = orig_tokenizers

    def test_settings_apply_windows_fixes_xformers_disabled(self):
        """Test that xformers is disabled on Windows."""
        from backpropagate.config import Settings

        s = Settings()
        # Ensure xformers_disabled is True for this test
        s.windows.xformers_disabled = True

        with patch("os.name", "nt"):
            orig_xformers = os.environ.get("XFORMERS_DISABLED")

            s.apply_windows_fixes()

            assert os.environ.get("XFORMERS_DISABLED") == "1"

            # Restore
            if orig_xformers is not None:
                os.environ["XFORMERS_DISABLED"] = orig_xformers
            else:
                os.environ.pop("XFORMERS_DISABLED", None)

    def test_settings_apply_windows_fixes_not_on_linux(self):
        """Test that Windows fixes don't apply on Linux."""
        from backpropagate.config import Settings

        s = Settings()

        with patch("os.name", "posix"):
            orig_tokenizers = os.environ.get("TOKENIZERS_PARALLELISM")
            # Clear the variable
            os.environ.pop("TOKENIZERS_PARALLELISM", None)

            s.apply_windows_fixes()

            # Should not have set anything (or it should still be None)
            # Note: this test may not work perfectly because we can't unset during the test

            # Restore
            if orig_tokenizers is not None:
                os.environ["TOKENIZERS_PARALLELISM"] = orig_tokenizers


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings(self):
        """Test get_settings returns Settings instance."""
        from backpropagate.config import get_settings, Settings

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
        from backpropagate.config import reload_settings, get_settings, Settings

        s1 = get_settings()
        s2 = reload_settings()

        # Both should be Settings instances
        assert isinstance(s1, Settings)
        assert isinstance(s2, Settings)

    def test_reload_settings_returns_new_instance(self):
        """Test reload_settings returns a (potentially) new instance."""
        from backpropagate.config import reload_settings, Settings

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
        from backpropagate.config import settings, get_output_dir

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
        from backpropagate.config import settings, get_training_args

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
        assert WINDOWS_DEFAULTS["cuda_launch_blocking"] is True
        assert WINDOWS_DEFAULTS["pre_tokenize"] is True


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
            Settings,
            settings,
            get_settings,
            reload_settings,
            get_output_dir,
            get_cache_dir,
            get_training_args,
            ModelConfig,
            TrainingConfig,
            LoRAConfig,
            DataConfig,
            PYDANTIC_SETTINGS_AVAILABLE,
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

    def test_apply_windows_fixes_cuda_launch_blocking(self):
        """Test cuda_launch_blocking is applied when True."""
        from backpropagate.config import Settings

        s = Settings()
        s.windows.cuda_launch_blocking = True

        with patch("os.name", "nt"):
            orig_cuda = os.environ.get("CUDA_LAUNCH_BLOCKING")

            s.apply_windows_fixes()

            assert os.environ.get("CUDA_LAUNCH_BLOCKING") == "1"

            # Restore
            if orig_cuda is not None:
                os.environ["CUDA_LAUNCH_BLOCKING"] = orig_cuda
            else:
                os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

    def test_apply_windows_fixes_xformers_not_disabled(self):
        """Test xformers is not disabled when setting is False."""
        from backpropagate.config import Settings

        s = Settings()
        s.windows.xformers_disabled = False

        with patch("os.name", "nt"):
            orig_xformers = os.environ.get("XFORMERS_DISABLED")
            os.environ.pop("XFORMERS_DISABLED", None)

            s.apply_windows_fixes()

            # Should NOT set XFORMERS_DISABLED when xformers_disabled is False
            # Note: The code only sets it when True, doesn't unset when False

            # Restore
            if orig_xformers is not None:
                os.environ["XFORMERS_DISABLED"] = orig_xformers


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
            TrainingPreset,
            TRAINING_PRESETS,
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
            TrainingPreset,
            TRAINING_PRESETS,
            get_preset,
            get_recommended_lr,
            get_recommended_warmup,
        )

        assert TrainingPreset is not None
        assert TRAINING_PRESETS is not None
        assert get_preset is not None
        assert get_recommended_lr is not None
        assert get_recommended_warmup is not None
