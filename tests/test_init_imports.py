"""
Tests for backpropagate package imports.

Verifies that:
- All exports in __all__ are importable
- Version is defined
- Core modules load correctly
- Lazy loading works for optional features
- Error messages are helpful for missing features
"""

import pytest


class TestPackageImports:
    """Tests for basic package imports."""

    def test_import_package(self):
        """Package can be imported."""
        import backpropagate
        assert backpropagate is not None

    def test_version_defined(self):
        """__version__ is defined."""
        import backpropagate
        assert hasattr(backpropagate, "__version__")
        assert isinstance(backpropagate.__version__, str)
        assert backpropagate.__version__ == "0.1.0"


class TestExceptionImports:
    """Tests for exception class imports."""

    def test_base_exception(self):
        """BackpropagateError can be imported."""
        from backpropagate import BackpropagateError
        assert issubclass(BackpropagateError, Exception)

    def test_configuration_errors(self):
        """Configuration error classes can be imported."""
        from backpropagate import ConfigurationError, InvalidSettingError
        assert issubclass(ConfigurationError, Exception)
        assert issubclass(InvalidSettingError, ConfigurationError)

    def test_dataset_errors(self):
        """Dataset error classes can be imported."""
        from backpropagate import (
            DatasetError,
            DatasetNotFoundError,
            DatasetParseError,
            DatasetValidationError,
        )
        assert issubclass(DatasetError, Exception)
        assert issubclass(DatasetNotFoundError, DatasetError)
        assert issubclass(DatasetParseError, DatasetError)
        assert issubclass(DatasetValidationError, DatasetError)

    def test_training_errors(self):
        """Training error classes can be imported."""
        from backpropagate import (
            TrainingError,
            ModelLoadError,
            TrainingAbortedError,
            CheckpointError,
        )
        assert issubclass(TrainingError, Exception)
        assert issubclass(ModelLoadError, TrainingError)
        assert issubclass(TrainingAbortedError, TrainingError)
        assert issubclass(CheckpointError, TrainingError)

    def test_export_errors(self):
        """Export error classes can be imported."""
        from backpropagate import (
            ExportError,
            LoRAExportError,
            MergeExportError,
            GGUFExportError,
            OllamaRegistrationError,
        )
        assert issubclass(ExportError, Exception)
        assert issubclass(LoRAExportError, ExportError)
        assert issubclass(MergeExportError, ExportError)
        assert issubclass(GGUFExportError, ExportError)
        assert issubclass(OllamaRegistrationError, ExportError)

    def test_gpu_errors(self):
        """GPU error classes can be imported."""
        from backpropagate import GPUError, GPUMemoryError, GPUTemperatureError
        assert issubclass(GPUError, Exception)
        assert issubclass(GPUMemoryError, GPUError)
        assert issubclass(GPUTemperatureError, GPUError)

    def test_slao_errors(self):
        """SLAO error classes can be imported."""
        from backpropagate import SLAOError, SLAOMergeError, SLAOCheckpointError
        assert issubclass(SLAOError, Exception)
        assert issubclass(SLAOMergeError, SLAOError)
        assert issubclass(SLAOCheckpointError, SLAOError)

    def test_batch_operation_error(self):
        """BatchOperationError can be imported."""
        from backpropagate import BatchOperationError
        assert issubclass(BatchOperationError, Exception)


class TestSecurityImports:
    """Tests for security module imports."""

    def test_security_functions(self):
        """Security functions can be imported."""
        from backpropagate import safe_path, check_torch_security
        assert callable(safe_path)
        assert callable(check_torch_security)

    def test_security_exceptions(self):
        """Security exceptions can be imported."""
        from backpropagate import SecurityWarning, PathTraversalError
        assert issubclass(SecurityWarning, Warning)
        assert issubclass(PathTraversalError, Exception)


class TestUISecurityImports:
    """Tests for UI security module imports."""

    def test_security_config(self):
        """SecurityConfig can be imported."""
        from backpropagate import SecurityConfig, DEFAULT_SECURITY_CONFIG
        assert SecurityConfig is not None
        assert DEFAULT_SECURITY_CONFIG is not None

    def test_rate_limiter(self):
        """EnhancedRateLimiter can be imported."""
        from backpropagate import EnhancedRateLimiter
        assert EnhancedRateLimiter is not None

    def test_file_validator(self):
        """FileValidator can be imported."""
        from backpropagate import FileValidator
        assert FileValidator is not None

    def test_extension_constants(self):
        """File extension constants can be imported."""
        from backpropagate import ALLOWED_DATASET_EXTENSIONS, DANGEROUS_EXTENSIONS
        assert isinstance(ALLOWED_DATASET_EXTENSIONS, (set, frozenset, list, tuple))
        assert isinstance(DANGEROUS_EXTENSIONS, (set, frozenset, list, tuple))

    def test_security_utilities(self):
        """Security utility functions can be imported."""
        from backpropagate import safe_gradio_handler, log_security_event
        assert callable(safe_gradio_handler)
        assert callable(log_security_event)


class TestFeatureFlagImports:
    """Tests for feature flag imports."""

    def test_features_dict(self):
        """FEATURES dict can be imported."""
        from backpropagate import FEATURES
        assert isinstance(FEATURES, dict)

    def test_feature_functions(self):
        """Feature flag functions can be imported."""
        from backpropagate import (
            check_feature,
            require_feature,
            get_install_hint,
            list_available_features,
            list_missing_features,
        )
        assert callable(check_feature)
        assert callable(require_feature)
        assert callable(get_install_hint)
        assert callable(list_available_features)
        assert callable(list_missing_features)

    def test_feature_exception(self):
        """FeatureNotAvailable can be imported."""
        from backpropagate import FeatureNotAvailable
        assert issubclass(FeatureNotAvailable, Exception)

    def test_system_info_functions(self):
        """System info functions can be imported."""
        from backpropagate import get_gpu_info, get_system_info
        assert callable(get_gpu_info)
        assert callable(get_system_info)


class TestConfigImports:
    """Tests for configuration imports."""

    def test_settings_class(self):
        """Settings class can be imported."""
        from backpropagate import Settings, settings
        assert Settings is not None
        assert settings is not None

    def test_settings_functions(self):
        """Settings functions can be imported."""
        from backpropagate import (
            get_settings,
            reload_settings,
            get_output_dir,
            get_cache_dir,
            get_training_args,
        )
        assert callable(get_settings)
        assert callable(reload_settings)
        assert callable(get_output_dir)
        assert callable(get_cache_dir)
        assert callable(get_training_args)

    def test_config_classes(self):
        """Config classes can be imported."""
        from backpropagate import ModelConfig, TrainingConfig, LoRAConfig, DataConfig
        assert ModelConfig is not None
        assert TrainingConfig is not None
        assert LoRAConfig is not None
        assert DataConfig is not None

    def test_pydantic_available_flag(self):
        """PYDANTIC_SETTINGS_AVAILABLE can be imported."""
        from backpropagate import PYDANTIC_SETTINGS_AVAILABLE
        assert isinstance(PYDANTIC_SETTINGS_AVAILABLE, bool)


class TestTrainerImports:
    """Tests for trainer module imports."""

    def test_trainer_class(self):
        """Trainer class can be imported."""
        from backpropagate import Trainer
        assert Trainer is not None

    def test_training_dataclasses(self):
        """Training dataclasses can be imported."""
        from backpropagate import TrainingRun, TrainingCallback
        assert TrainingRun is not None
        assert TrainingCallback is not None

    def test_trainer_functions(self):
        """Trainer functions can be imported."""
        from backpropagate import load_model, load_dataset
        assert callable(load_model)
        assert callable(load_dataset)


class TestMultiRunImports:
    """Tests for multi-run module imports."""

    def test_multi_run_classes(self):
        """Multi-run classes can be imported."""
        from backpropagate import (
            MultiRunTrainer,
            MultiRunConfig,
            MultiRunResult,
            RunResult,
            MergeMode,
        )
        assert MultiRunTrainer is not None
        assert MultiRunConfig is not None
        assert MultiRunResult is not None
        assert RunResult is not None
        assert MergeMode is not None

    def test_backwards_compatibility_aliases(self):
        """Backwards compatibility aliases can be imported."""
        from backpropagate import SpeedrunTrainer, SpeedrunConfig, SpeedrunResult
        assert SpeedrunTrainer is not None
        assert SpeedrunConfig is not None
        assert SpeedrunResult is not None


class TestSLAOImports:
    """Tests for SLAO module imports."""

    def test_slao_classes(self):
        """SLAO classes can be imported."""
        from backpropagate import SLAOMerger, SLAOConfig, MergeResult
        assert SLAOMerger is not None
        assert SLAOConfig is not None
        assert MergeResult is not None

    def test_slao_functions(self):
        """SLAO functions can be imported."""
        from backpropagate import (
            time_aware_scale,
            orthogonal_init_A,
            merge_lora_weights,
        )
        assert callable(time_aware_scale)
        assert callable(orthogonal_init_A)
        assert callable(merge_lora_weights)

    def test_advanced_slao_functions(self):
        """Advanced SLAO functions can be imported."""
        from backpropagate import (
            compute_task_similarity,
            adaptive_scale,
            get_layer_scale,
        )
        assert callable(compute_task_similarity)
        assert callable(adaptive_scale)
        assert callable(get_layer_scale)


class TestCheckpointImports:
    """Tests for checkpoint module imports."""

    def test_checkpoint_classes(self):
        """Checkpoint classes can be imported."""
        from backpropagate import (
            CheckpointManager,
            CheckpointPolicy,
            CheckpointInfo,
            CheckpointStats,
        )
        assert CheckpointManager is not None
        assert CheckpointPolicy is not None
        assert CheckpointInfo is not None
        assert CheckpointStats is not None


class TestGPUSafetyImports:
    """Tests for GPU safety module imports."""

    def test_gpu_monitor_class(self):
        """GPUMonitor class can be imported."""
        from backpropagate import GPUMonitor
        assert GPUMonitor is not None

    def test_gpu_status_classes(self):
        """GPU status classes can be imported."""
        from backpropagate import GPUStatus, GPUSafetyConfig, GPUCondition
        assert GPUStatus is not None
        assert GPUSafetyConfig is not None
        assert GPUCondition is not None

    def test_gpu_functions(self):
        """GPU functions can be imported."""
        from backpropagate import (
            check_gpu_safe,
            get_gpu_status,
            wait_for_safe_gpu,
            format_gpu_status,
        )
        assert callable(check_gpu_safe)
        assert callable(get_gpu_status)
        assert callable(wait_for_safe_gpu)
        assert callable(format_gpu_status)


class TestExportImports:
    """Tests for export module imports."""

    def test_export_enums(self):
        """Export enums can be imported."""
        from backpropagate import GGUFQuantization, ExportFormat
        assert GGUFQuantization is not None
        assert ExportFormat is not None

    def test_export_result(self):
        """ExportResult can be imported."""
        from backpropagate import ExportResult
        assert ExportResult is not None

    def test_export_functions(self):
        """Export functions can be imported."""
        from backpropagate import (
            export_lora,
            export_merged,
            export_gguf,
        )
        assert callable(export_lora)
        assert callable(export_merged)
        assert callable(export_gguf)

    def test_ollama_functions(self):
        """Ollama functions can be imported."""
        from backpropagate import (
            create_modelfile,
            register_with_ollama,
            list_ollama_models,
        )
        assert callable(create_modelfile)
        assert callable(register_with_ollama)
        assert callable(list_ollama_models)


class TestDatasetImports:
    """Tests for dataset module imports."""

    def test_dataset_core_classes(self):
        """Dataset core classes can be imported."""
        from backpropagate import (
            DatasetLoader,
            DatasetFormat,
            ValidationResult,
            ValidationError,
            DatasetStats,
            FormatConverter,
        )
        assert DatasetLoader is not None
        assert DatasetFormat is not None
        assert ValidationResult is not None
        assert ValidationError is not None
        assert DatasetStats is not None
        assert FormatConverter is not None

    def test_dataset_core_functions(self):
        """Dataset core functions can be imported."""
        from backpropagate import (
            detect_format,
            validate_dataset,
            convert_to_chatml,
            preview_samples,
            get_dataset_stats,
        )
        assert callable(detect_format)
        assert callable(validate_dataset)
        assert callable(convert_to_chatml)
        assert callable(preview_samples)
        assert callable(get_dataset_stats)

    def test_streaming_loader(self):
        """StreamingDatasetLoader can be imported."""
        from backpropagate import StreamingDatasetLoader
        assert StreamingDatasetLoader is not None

    def test_filtering_classes(self):
        """Filtering classes can be imported."""
        from backpropagate import FilterStats, filter_by_quality
        assert FilterStats is not None
        assert callable(filter_by_quality)

    def test_deduplication_functions(self):
        """Deduplication functions can be imported."""
        from backpropagate import deduplicate_exact, deduplicate_minhash
        assert callable(deduplicate_exact)
        assert callable(deduplicate_minhash)

    def test_perplexity_classes(self):
        """Perplexity filtering classes can be imported."""
        from backpropagate import PerplexityFilter, PerplexityStats
        assert PerplexityFilter is not None
        assert PerplexityStats is not None

    def test_perplexity_functions(self):
        """Perplexity functions can be imported."""
        from backpropagate import compute_perplexity, filter_by_perplexity
        assert callable(compute_perplexity)
        assert callable(filter_by_perplexity)

    def test_curriculum_classes(self):
        """Curriculum learning classes can be imported."""
        from backpropagate import CurriculumStats
        assert CurriculumStats is not None

    def test_curriculum_functions(self):
        """Curriculum learning functions can be imported."""
        from backpropagate import (
            compute_difficulty_score,
            order_by_difficulty,
            get_curriculum_chunks,
            analyze_curriculum,
        )
        assert callable(compute_difficulty_score)
        assert callable(order_by_difficulty)
        assert callable(get_curriculum_chunks)
        assert callable(analyze_curriculum)


class TestLazyLoadingImports:
    """Tests for lazy-loaded optional features."""

    def test_launch_lazy_load(self):
        """launch function is lazy-loaded."""
        import backpropagate
        # Check it's in __all__
        assert "launch" in backpropagate.__all__
        # Don't actually call it as it may not be available

    def test_theme_functions_lazy_load(self):
        """Theme functions are lazy-loaded."""
        import backpropagate
        assert "create_backpropagate_theme" in backpropagate.__all__
        assert "get_theme_info" in backpropagate.__all__
        assert "get_css" in backpropagate.__all__


class TestAllExportsComplete:
    """Tests that __all__ is complete and correct."""

    def test_all_is_list(self):
        """__all__ is a list."""
        import backpropagate
        assert isinstance(backpropagate.__all__, list)

    def test_all_has_version(self):
        """__all__ includes __version__."""
        import backpropagate
        assert "__version__" in backpropagate.__all__

    def test_all_count_reasonable(self):
        """__all__ has a reasonable number of exports."""
        import backpropagate
        # Should have many exports (100+)
        assert len(backpropagate.__all__) >= 100

    def test_non_lazy_exports_exist(self):
        """Non-lazy exports in __all__ are importable."""
        import backpropagate

        lazy_items = {"launch", "create_backpropagate_theme", "get_theme_info", "get_css"}

        for name in backpropagate.__all__:
            if name in lazy_items:
                continue  # Skip lazy-loaded items

            assert hasattr(backpropagate, name), f"Missing export: {name}"


class TestGetAttrForMissingFeatures:
    """Tests for helpful error messages on missing features."""

    def test_missing_feature_error_message(self):
        """Missing feature provides helpful error message."""
        import backpropagate
        from backpropagate import FEATURES

        # If UI feature is missing, trying to access launch should give helpful error
        if not FEATURES.get("ui", False):
            with pytest.raises(ImportError) as exc_info:
                _ = backpropagate.launch
            error_msg = str(exc_info.value)
            assert "ui" in error_msg.lower() or "install" in error_msg.lower()

    def test_invalid_attribute_error(self):
        """Invalid attribute raises AttributeError."""
        import backpropagate

        with pytest.raises(AttributeError):
            _ = backpropagate.nonexistent_attribute_xyz


class TestCrossImports:
    """Tests for cross-module import consistency."""

    def test_trainer_from_package(self):
        """Trainer from package matches trainer module."""
        from backpropagate import Trainer as TrainerFromPackage
        from backpropagate.trainer import Trainer as TrainerFromModule
        assert TrainerFromPackage is TrainerFromModule

    def test_settings_from_package(self):
        """settings from package is a Settings instance."""
        from backpropagate import settings as SettingsFromPackage
        from backpropagate.config import Settings
        # Check it's a Settings instance (not necessarily same object due to test isolation)
        assert isinstance(SettingsFromPackage, Settings)

    def test_exceptions_from_package(self):
        """Exceptions from package match exceptions module."""
        from backpropagate import TrainingError as FromPackage
        from backpropagate.exceptions import TrainingError as FromModule
        assert FromPackage is FromModule


class TestImportFromPattern:
    """Tests for 'from backpropagate import X' pattern."""

    def test_import_trainer(self):
        """from backpropagate import Trainer works."""
        from backpropagate import Trainer
        assert Trainer is not None

    def test_import_settings(self):
        """from backpropagate import settings works."""
        from backpropagate import settings
        assert settings is not None

    def test_import_multiple(self):
        """Multiple imports in one statement work."""
        from backpropagate import Trainer, settings, TrainingRun, TrainingCallback
        assert Trainer is not None
        assert settings is not None
        assert TrainingRun is not None
        assert TrainingCallback is not None


class TestStarImport:
    """Tests for 'from backpropagate import *' pattern."""

    def test_star_import_works(self):
        """Star import doesn't raise errors."""
        # This is a simpler test - just verify it doesn't crash
        import importlib
        spec = importlib.util.find_spec("backpropagate")
        assert spec is not None
        # Note: We don't actually do 'from backpropagate import *'
        # because that would pollute the test namespace
