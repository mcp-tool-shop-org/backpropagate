"""
Tests for custom exception hierarchy.

Tests cover:
- Base exception class (BackpropagateError)
- Configuration errors
- Dataset errors
- Training errors
- Export errors
- GPU errors
- SLAO errors
- Batch operation errors
"""

import pytest
from backpropagate.exceptions import (
    # Base
    BackpropagateError,
    # Configuration
    ConfigurationError,
    InvalidSettingError,
    # Dataset
    DatasetError,
    DatasetNotFoundError,
    DatasetParseError,
    DatasetValidationError,
    DatasetFormatError,
    # Training
    TrainingError,
    ModelLoadError,
    TrainingAbortedError,
    CheckpointError,
    # Export
    ExportError,
    LoRAExportError,
    GGUFExportError,
    MergeExportError,
    OllamaRegistrationError,
    # GPU
    GPUError,
    GPUNotAvailableError,
    GPUMemoryError,
    GPUTemperatureError,
    GPUMonitoringError,
    # SLAO
    SLAOError,
    SLAOMergeError,
    SLAOCheckpointError,
    # Batch
    BatchOperationError,
)


# =============================================================================
# BASE EXCEPTION TESTS
# =============================================================================

class TestBackpropagateError:
    """Tests for base BackpropagateError class."""

    def test_basic_creation(self):
        """Should create exception with message."""
        err = BackpropagateError("Something went wrong")
        assert err.message == "Something went wrong"
        assert str(err) == "Something went wrong"

    def test_with_details(self):
        """Should store details dict."""
        err = BackpropagateError(
            "Error occurred",
            details={"key": "value", "count": 42}
        )
        assert err.details == {"key": "value", "count": 42}

    def test_with_suggestion(self):
        """Should include suggestion in full message."""
        err = BackpropagateError(
            "Error occurred",
            suggestion="Try doing X instead"
        )
        assert err.suggestion == "Try doing X instead"
        assert "Suggestion: Try doing X instead" in str(Exception.__str__(err))

    def test_repr(self):
        """Should have proper repr."""
        err = BackpropagateError("Test error")
        assert repr(err) == "BackpropagateError('Test error')"

    def test_empty_details_default(self):
        """Should default to empty dict for details."""
        err = BackpropagateError("Error")
        assert err.details == {}

    def test_exception_hierarchy(self):
        """Should inherit from Exception."""
        err = BackpropagateError("Error")
        assert isinstance(err, Exception)


# =============================================================================
# CONFIGURATION ERROR TESTS
# =============================================================================

class TestConfigurationError:
    """Tests for ConfigurationError class."""

    def test_basic_creation(self):
        """Should create configuration error."""
        err = ConfigurationError("Invalid configuration")
        assert isinstance(err, BackpropagateError)
        assert str(err) == "Invalid configuration"

    def test_hierarchy(self):
        """Should be catchable as BackpropagateError."""
        err = ConfigurationError("Config error")
        assert isinstance(err, BackpropagateError)


class TestInvalidSettingError:
    """Tests for InvalidSettingError class."""

    def test_basic_creation(self):
        """Should create with setting details."""
        err = InvalidSettingError(
            setting_name="batch_size",
            value=-1,
            expected="positive integer"
        )
        assert err.setting_name == "batch_size"
        assert err.value == -1
        assert err.expected == "positive integer"
        assert "batch_size" in str(err)
        assert "-1" in str(err)

    def test_with_suggestion(self):
        """Should include suggestion."""
        err = InvalidSettingError(
            setting_name="learning_rate",
            value="invalid",
            expected="float",
            suggestion="Use a value like 2e-4"
        )
        assert err.suggestion == "Use a value like 2e-4"

    def test_details_populated(self):
        """Should populate details dict."""
        err = InvalidSettingError(
            setting_name="r",
            value=0,
            expected="positive integer > 0"
        )
        assert err.details["setting"] == "r"
        assert err.details["value"] == 0
        assert err.details["expected"] == "positive integer > 0"

    def test_hierarchy(self):
        """Should be catchable as ConfigurationError."""
        err = InvalidSettingError("test", "val", "str")
        assert isinstance(err, ConfigurationError)
        assert isinstance(err, BackpropagateError)


# =============================================================================
# DATASET ERROR TESTS
# =============================================================================

class TestDatasetError:
    """Tests for DatasetError class."""

    def test_basic_creation(self):
        """Should create dataset error."""
        err = DatasetError("Dataset problem")
        assert isinstance(err, BackpropagateError)

    def test_hierarchy(self):
        """Should be catchable as BackpropagateError."""
        err = DatasetError("Error")
        assert isinstance(err, BackpropagateError)


class TestDatasetNotFoundError:
    """Tests for DatasetNotFoundError class."""

    def test_basic_creation(self):
        """Should create with path."""
        err = DatasetNotFoundError("/path/to/data.jsonl")
        assert err.path == "/path/to/data.jsonl"
        assert "data.jsonl" in str(err)

    def test_default_suggestion(self):
        """Should have default suggestion."""
        err = DatasetNotFoundError("/some/path.json")
        assert err.suggestion is not None
        assert "path.json" in err.suggestion

    def test_custom_suggestion(self):
        """Should accept custom suggestion."""
        err = DatasetNotFoundError(
            "/path/to/file",
            suggestion="Download the dataset first"
        )
        assert err.suggestion == "Download the dataset first"

    def test_details_populated(self):
        """Should populate details."""
        err = DatasetNotFoundError("/data/file.jsonl")
        assert err.details["path"] == "/data/file.jsonl"

    def test_hierarchy(self):
        """Should be catchable as DatasetError."""
        err = DatasetNotFoundError("/path")
        assert isinstance(err, DatasetError)
        assert isinstance(err, BackpropagateError)


class TestDatasetParseError:
    """Tests for DatasetParseError class."""

    def test_basic_creation(self):
        """Should create with message."""
        err = DatasetParseError("Invalid JSON")
        assert "Invalid JSON" in str(err)

    def test_with_path(self):
        """Should store path."""
        err = DatasetParseError("Parse error", path="/data/file.jsonl")
        assert err.path == "/data/file.jsonl"
        assert err.details["path"] == "/data/file.jsonl"

    def test_with_line_number(self):
        """Should include line number in message."""
        err = DatasetParseError(
            "Invalid syntax",
            path="/data/file.jsonl",
            line_number=42
        )
        assert err.line_number == 42
        assert "line 42" in str(err)
        assert err.details["line_number"] == 42

    def test_default_suggestion(self):
        """Should have default suggestion."""
        err = DatasetParseError("Error")
        assert "JSON/CSV" in err.suggestion

    def test_custom_suggestion(self):
        """Should accept custom suggestion."""
        err = DatasetParseError(
            "Parse error",
            suggestion="Check file encoding"
        )
        assert err.suggestion == "Check file encoding"

    def test_hierarchy(self):
        """Should be catchable as DatasetError."""
        err = DatasetParseError("Error")
        assert isinstance(err, DatasetError)


class TestDatasetValidationError:
    """Tests for DatasetValidationError class."""

    def test_basic_creation(self):
        """Should create with message and empty errors list."""
        err = DatasetValidationError("Validation failed", errors=[])
        assert "Validation failed" in str(err)
        assert err.errors == []

    def test_with_errors_list(self):
        """Should store and format error list."""
        errors = [
            "Row 1: missing 'text' field",
            "Row 5: empty content",
            "Row 10: invalid role"
        ]
        err = DatasetValidationError("Validation failed", errors=errors)
        assert err.errors == errors
        assert "Row 1" in str(err)
        assert "Row 5" in str(err)

    def test_truncates_many_errors(self):
        """Should truncate when more than 10 errors."""
        errors = [f"Error {i}" for i in range(15)]
        err = DatasetValidationError("Many errors", errors=errors)
        # Should show only first 10 and a count
        assert "5 more" in str(err)

    def test_details_populated(self):
        """Should populate details."""
        errors = ["e1", "e2", "e3"]
        err = DatasetValidationError("Failed", errors=errors)
        assert err.details["error_count"] == 3
        assert err.details["errors"] == errors

    def test_hierarchy(self):
        """Should be catchable as DatasetError."""
        err = DatasetValidationError("Error", errors=[])
        assert isinstance(err, DatasetError)


class TestDatasetFormatError:
    """Tests for DatasetFormatError class."""

    def test_basic_creation(self):
        """Should create with message."""
        err = DatasetFormatError("Unknown format")
        assert "Unknown format" in str(err)

    def test_with_detected_format(self):
        """Should store detected format."""
        err = DatasetFormatError(
            "Cannot process format",
            detected_format="weird_format"
        )
        assert err.detected_format == "weird_format"
        assert err.details["detected_format"] == "weird_format"

    def test_with_supported_formats(self):
        """Should list supported formats in suggestion."""
        err = DatasetFormatError(
            "Unsupported format",
            supported_formats=["sharegpt", "alpaca", "openai"]
        )
        assert err.supported_formats == ["sharegpt", "alpaca", "openai"]
        assert "sharegpt" in err.suggestion
        assert "alpaca" in err.suggestion

    def test_no_suggestion_without_formats(self):
        """Should have no suggestion if no supported formats provided."""
        err = DatasetFormatError("Error")
        assert err.suggestion is None

    def test_hierarchy(self):
        """Should be catchable as DatasetError."""
        err = DatasetFormatError("Error")
        assert isinstance(err, DatasetError)


# =============================================================================
# TRAINING ERROR TESTS
# =============================================================================

class TestTrainingError:
    """Tests for TrainingError class."""

    def test_basic_creation(self):
        """Should create training error."""
        err = TrainingError("Training failed")
        assert isinstance(err, BackpropagateError)

    def test_hierarchy(self):
        """Should be catchable as BackpropagateError."""
        err = TrainingError("Error")
        assert isinstance(err, BackpropagateError)


class TestModelLoadError:
    """Tests for ModelLoadError class."""

    def test_basic_creation(self):
        """Should create with model name and reason."""
        err = ModelLoadError(
            model_name="unsloth/Qwen2.5-7B",
            reason="Model not found"
        )
        assert err.model_name == "unsloth/Qwen2.5-7B"
        assert err.reason == "Model not found"
        assert "Qwen2.5-7B" in str(err)
        assert "not found" in str(err)

    def test_default_suggestion(self):
        """Should have default suggestion."""
        err = ModelLoadError("model", "reason")
        assert "model name" in err.suggestion.lower()

    def test_custom_suggestion(self):
        """Should accept custom suggestion."""
        err = ModelLoadError(
            "model",
            "reason",
            suggestion="Check your HuggingFace token"
        )
        assert err.suggestion == "Check your HuggingFace token"

    def test_details_populated(self):
        """Should populate details."""
        err = ModelLoadError("my-model", "network error")
        assert err.details["model_name"] == "my-model"
        assert err.details["reason"] == "network error"

    def test_hierarchy(self):
        """Should be catchable as TrainingError."""
        err = ModelLoadError("m", "r")
        assert isinstance(err, TrainingError)
        assert isinstance(err, BackpropagateError)


class TestTrainingAbortedError:
    """Tests for TrainingAbortedError class."""

    def test_basic_creation(self):
        """Should create with reason."""
        err = TrainingAbortedError("User interrupted")
        assert err.reason == "User interrupted"
        assert "User interrupted" in str(err)

    def test_with_steps_completed(self):
        """Should include steps in message."""
        err = TrainingAbortedError(
            "GPU overheat",
            steps_completed=150
        )
        assert err.steps_completed == 150
        assert "150 steps" in str(err)

    def test_with_checkpoint_path(self):
        """Should include checkpoint in message."""
        err = TrainingAbortedError(
            "Out of memory",
            steps_completed=100,
            checkpoint_path="/checkpoints/step-100"
        )
        assert err.checkpoint_path == "/checkpoints/step-100"
        assert "step-100" in str(err)

    def test_details_populated(self):
        """Should populate details."""
        err = TrainingAbortedError(
            "reason",
            steps_completed=50,
            checkpoint_path="/path"
        )
        assert err.details["reason"] == "reason"
        assert err.details["steps_completed"] == 50
        assert err.details["checkpoint_path"] == "/path"

    def test_hierarchy(self):
        """Should be catchable as TrainingError."""
        err = TrainingAbortedError("reason")
        assert isinstance(err, TrainingError)


class TestCheckpointError:
    """Tests for CheckpointError class."""

    def test_save_operation(self):
        """Should create for save operation."""
        err = CheckpointError(
            operation="save",
            path="/checkpoints/model",
            reason="Disk full"
        )
        assert err.operation == "save"
        assert err.path == "/checkpoints/model"
        assert err.reason == "Disk full"
        assert "save" in str(err)
        assert "Disk full" in str(err)

    def test_load_operation(self):
        """Should create for load operation."""
        err = CheckpointError(
            operation="load",
            path="/checkpoints/model",
            reason="File corrupted"
        )
        assert "load" in str(err)

    def test_details_populated(self):
        """Should populate details."""
        err = CheckpointError("save", "/path", "reason")
        assert err.details["operation"] == "save"
        assert err.details["path"] == "/path"
        assert err.details["reason"] == "reason"

    def test_hierarchy(self):
        """Should be catchable as TrainingError."""
        err = CheckpointError("op", "/path", "reason")
        assert isinstance(err, TrainingError)


# =============================================================================
# EXPORT ERROR TESTS
# =============================================================================

class TestExportError:
    """Tests for ExportError class."""

    def test_basic_creation(self):
        """Should create export error."""
        err = ExportError("Export failed")
        assert isinstance(err, BackpropagateError)

    def test_hierarchy(self):
        """Should be catchable as BackpropagateError."""
        err = ExportError("Error")
        assert isinstance(err, BackpropagateError)


class TestLoRAExportError:
    """Tests for LoRAExportError class."""

    def test_basic_creation(self):
        """Should create with reason."""
        err = LoRAExportError("No LoRA adapters found")
        assert err.reason == "No LoRA adapters found"
        assert "LoRA export failed" in str(err)

    def test_with_output_path(self):
        """Should store output path."""
        err = LoRAExportError(
            "Write failed",
            output_path="/models/lora"
        )
        assert err.output_path == "/models/lora"
        assert err.details["output_path"] == "/models/lora"

    def test_default_suggestion(self):
        """Should have default suggestion."""
        err = LoRAExportError("reason")
        assert "LoRA adapters" in err.suggestion

    def test_custom_suggestion(self):
        """Should accept custom suggestion."""
        err = LoRAExportError("reason", suggestion="Check permissions")
        assert err.suggestion == "Check permissions"

    def test_hierarchy(self):
        """Should be catchable as ExportError."""
        err = LoRAExportError("reason")
        assert isinstance(err, ExportError)


class TestGGUFExportError:
    """Tests for GGUFExportError class."""

    def test_basic_creation(self):
        """Should create with reason."""
        err = GGUFExportError("Conversion failed")
        assert err.reason == "Conversion failed"
        assert "GGUF export failed" in str(err)

    def test_with_output_path(self):
        """Should store output path."""
        err = GGUFExportError(
            "Write error",
            output_path="/models/model.gguf"
        )
        assert err.output_path == "/models/model.gguf"

    def test_with_quantization(self):
        """Should store quantization type."""
        err = GGUFExportError(
            "Quant failed",
            output_path="/model.gguf",
            quantization="q4_k_m"
        )
        assert err.quantization == "q4_k_m"
        assert err.details["quantization"] == "q4_k_m"

    def test_default_suggestion(self):
        """Should have default suggestion."""
        err = GGUFExportError("reason")
        assert "Unsloth" in err.suggestion or "llama.cpp" in err.suggestion

    def test_hierarchy(self):
        """Should be catchable as ExportError."""
        err = GGUFExportError("reason")
        assert isinstance(err, ExportError)


class TestMergeExportError:
    """Tests for MergeExportError class."""

    def test_basic_creation(self):
        """Should create with reason."""
        err = MergeExportError("Merge failed")
        assert "Merge export failed" in str(err)

    def test_with_suggestion(self):
        """Should accept suggestion."""
        err = MergeExportError("reason", suggestion="Try again")
        assert err.suggestion == "Try again"

    def test_hierarchy(self):
        """Should be catchable as ExportError."""
        err = MergeExportError("reason")
        assert isinstance(err, ExportError)


class TestOllamaRegistrationError:
    """Tests for OllamaRegistrationError class."""

    def test_basic_creation(self):
        """Should create with model name and reason."""
        err = OllamaRegistrationError(
            model_name="my-model",
            reason="Ollama not running"
        )
        assert err.model_name == "my-model"
        assert "my-model" in str(err)
        assert "Ollama" in str(err)

    def test_default_suggestion(self):
        """Should have default suggestion."""
        err = OllamaRegistrationError("model", "reason")
        assert "ollama" in err.suggestion.lower()

    def test_custom_suggestion(self):
        """Should accept custom suggestion."""
        err = OllamaRegistrationError(
            "model",
            "reason",
            suggestion="Start Ollama with 'ollama serve'"
        )
        assert err.suggestion == "Start Ollama with 'ollama serve'"

    def test_details_populated(self):
        """Should populate details."""
        err = OllamaRegistrationError("my-model", "reason")
        assert err.details["model_name"] == "my-model"

    def test_hierarchy(self):
        """Should be catchable as ExportError."""
        err = OllamaRegistrationError("m", "r")
        assert isinstance(err, ExportError)


# =============================================================================
# GPU ERROR TESTS
# =============================================================================

class TestGPUError:
    """Tests for GPUError class."""

    def test_basic_creation(self):
        """Should create GPU error."""
        err = GPUError("GPU problem")
        assert isinstance(err, BackpropagateError)

    def test_hierarchy(self):
        """Should be catchable as BackpropagateError."""
        err = GPUError("Error")
        assert isinstance(err, BackpropagateError)


class TestGPUNotAvailableError:
    """Tests for GPUNotAvailableError class."""

    def test_basic_creation(self):
        """Should create with default message."""
        err = GPUNotAvailableError()
        assert "CUDA" in str(err) or "GPU" in str(err)

    def test_default_suggestion(self):
        """Should have default suggestion."""
        err = GPUNotAvailableError()
        assert "CUDA" in err.suggestion

    def test_custom_suggestion(self):
        """Should accept custom suggestion."""
        err = GPUNotAvailableError(suggestion="Install NVIDIA drivers")
        assert err.suggestion == "Install NVIDIA drivers"

    def test_hierarchy(self):
        """Should be catchable as GPUError."""
        err = GPUNotAvailableError()
        assert isinstance(err, GPUError)


class TestGPUMemoryError:
    """Tests for GPUMemoryError class."""

    def test_basic_creation(self):
        """Should create with default message."""
        err = GPUMemoryError()
        assert "memory" in str(err).lower()

    def test_with_memory_info(self):
        """Should include memory details in message."""
        err = GPUMemoryError(
            required_gb=12.0,
            available_gb=8.0
        )
        assert err.required_gb == 12.0
        assert err.available_gb == 8.0
        assert "12.0" in str(err)
        assert "8.0" in str(err)

    def test_default_suggestion(self):
        """Should have default suggestion."""
        err = GPUMemoryError()
        assert "batch size" in err.suggestion.lower() or "gradient checkpointing" in err.suggestion.lower()

    def test_custom_suggestion(self):
        """Should accept custom suggestion."""
        err = GPUMemoryError(suggestion="Use a smaller model")
        assert err.suggestion == "Use a smaller model"

    def test_details_populated(self):
        """Should populate details."""
        err = GPUMemoryError(required_gb=12.0, available_gb=8.0)
        assert err.details["required_gb"] == 12.0
        assert err.details["available_gb"] == 8.0

    def test_hierarchy(self):
        """Should be catchable as GPUError."""
        err = GPUMemoryError()
        assert isinstance(err, GPUError)


class TestGPUTemperatureError:
    """Tests for GPUTemperatureError class."""

    def test_basic_creation(self):
        """Should create with temperature info."""
        err = GPUTemperatureError(
            temperature=95.0,
            threshold=85.0
        )
        assert err.temperature == 95.0
        assert err.threshold == 85.0
        assert "95" in str(err)
        assert "85" in str(err)

    def test_default_suggestion(self):
        """Should have default suggestion."""
        err = GPUTemperatureError(90.0, 85.0)
        assert "cool" in err.suggestion.lower()

    def test_custom_suggestion(self):
        """Should accept custom suggestion."""
        err = GPUTemperatureError(
            90.0,
            85.0,
            suggestion="Clean the GPU fans"
        )
        assert err.suggestion == "Clean the GPU fans"

    def test_details_populated(self):
        """Should populate details."""
        err = GPUTemperatureError(92.0, 85.0)
        assert err.details["temperature"] == 92.0
        assert err.details["threshold"] == 85.0

    def test_hierarchy(self):
        """Should be catchable as GPUError."""
        err = GPUTemperatureError(90.0, 85.0)
        assert isinstance(err, GPUError)


class TestGPUMonitoringError:
    """Tests for GPUMonitoringError class."""

    def test_basic_creation(self):
        """Should create with reason."""
        err = GPUMonitoringError("pynvml not available")
        assert "monitoring failed" in str(err).lower()
        assert "pynvml" in str(err)

    def test_default_suggestion(self):
        """Should have default suggestion."""
        err = GPUMonitoringError("reason")
        assert "pynvml" in err.suggestion.lower()

    def test_custom_suggestion(self):
        """Should accept custom suggestion."""
        err = GPUMonitoringError("reason", suggestion="Check NVIDIA drivers")
        assert err.suggestion == "Check NVIDIA drivers"

    def test_hierarchy(self):
        """Should be catchable as GPUError."""
        err = GPUMonitoringError("reason")
        assert isinstance(err, GPUError)


# =============================================================================
# SLAO ERROR TESTS
# =============================================================================

class TestSLAOError:
    """Tests for SLAOError class."""

    def test_basic_creation(self):
        """Should create SLAO error."""
        err = SLAOError("SLAO problem")
        assert isinstance(err, BackpropagateError)

    def test_hierarchy(self):
        """Should be catchable as BackpropagateError."""
        err = SLAOError("Error")
        assert isinstance(err, BackpropagateError)


class TestSLAOMergeError:
    """Tests for SLAOMergeError class."""

    def test_basic_creation(self):
        """Should create with reason."""
        err = SLAOMergeError("Merge weights incompatible")
        assert "SLAO merge failed" in str(err)
        assert "incompatible" in str(err)

    def test_with_run_index(self):
        """Should include run index in message."""
        err = SLAOMergeError(
            "Shape mismatch",
            run_index=3
        )
        assert err.run_index == 3
        assert "run 3" in str(err)
        assert err.details["run_index"] == 3

    def test_without_run_index(self):
        """Should work without run index."""
        err = SLAOMergeError("General error")
        assert err.run_index is None
        assert "run" not in str(err).lower() or "run_index" in str(err).lower()

    def test_with_suggestion(self):
        """Should accept suggestion."""
        err = SLAOMergeError("reason", suggestion="Check layer dimensions")
        assert err.suggestion == "Check layer dimensions"

    def test_hierarchy(self):
        """Should be catchable as SLAOError."""
        err = SLAOMergeError("reason")
        assert isinstance(err, SLAOError)


class TestSLAOCheckpointError:
    """Tests for SLAOCheckpointError class."""

    def test_save_operation(self):
        """Should create for save operation."""
        err = SLAOCheckpointError(
            operation="save",
            path="/checkpoints/slao",
            reason="Permission denied"
        )
        assert err.operation == "save"
        assert err.path == "/checkpoints/slao"
        assert "save" in str(err)
        assert "Permission denied" in str(err)

    def test_load_operation(self):
        """Should create for load operation."""
        err = SLAOCheckpointError(
            operation="load",
            path="/checkpoints/slao",
            reason="File not found"
        )
        assert "load" in str(err)

    def test_details_populated(self):
        """Should populate details."""
        err = SLAOCheckpointError("save", "/path", "reason")
        assert err.details["operation"] == "save"
        assert err.details["path"] == "/path"

    def test_hierarchy(self):
        """Should be catchable as SLAOError."""
        err = SLAOCheckpointError("op", "/path", "reason")
        assert isinstance(err, SLAOError)


# =============================================================================
# BATCH OPERATION ERROR TESTS
# =============================================================================

class TestBatchOperationError:
    """Tests for BatchOperationError class."""

    def test_basic_creation(self):
        """Should create with basic info."""
        errors = [(0, ValueError("bad")), (2, TypeError("wrong"))]
        err = BatchOperationError(
            operation="validate",
            total_items=10,
            failed_items=2,
            errors=errors
        )
        assert err.operation == "validate"
        assert err.total_items == 10
        assert err.failed_items == 2
        assert len(err.errors) == 2

    def test_success_rate_in_message(self):
        """Should include success rate."""
        err = BatchOperationError(
            operation="process",
            total_items=100,
            failed_items=20,
            errors=[]
        )
        assert "80" in str(err)  # 80% success rate

    def test_error_summary_truncation(self):
        """Should truncate errors if more than 5."""
        errors = [(i, ValueError(f"error {i}")) for i in range(10)]
        err = BatchOperationError(
            operation="test",
            total_items=10,
            failed_items=10,
            errors=errors
        )
        # Should show first 5 and indicate more
        assert "5 more" in str(err)

    def test_success_count_property(self):
        """Should compute success count."""
        err = BatchOperationError(
            operation="test",
            total_items=100,
            failed_items=30,
            errors=[]
        )
        assert err.success_count == 70

    def test_success_rate_property(self):
        """Should compute success rate."""
        err = BatchOperationError(
            operation="test",
            total_items=100,
            failed_items=25,
            errors=[]
        )
        assert err.success_rate == 75.0

    def test_success_rate_zero_total(self):
        """Should handle zero total items."""
        err = BatchOperationError(
            operation="test",
            total_items=0,
            failed_items=0,
            errors=[]
        )
        assert err.success_rate == 0.0

    def test_details_populated(self):
        """Should populate details."""
        errors = [(0, ValueError("err"))]
        err = BatchOperationError(
            operation="test",
            total_items=10,
            failed_items=1,
            errors=errors
        )
        assert err.details["operation"] == "test"
        assert err.details["total_items"] == 10
        assert err.details["failed_items"] == 1
        assert err.details["error_count"] == 1

    def test_with_suggestion(self):
        """Should accept suggestion."""
        err = BatchOperationError(
            operation="test",
            total_items=10,
            failed_items=5,
            errors=[],
            suggestion="Check input data"
        )
        assert err.suggestion == "Check input data"

    def test_hierarchy(self):
        """Should be catchable as BackpropagateError."""
        err = BatchOperationError("test", 10, 1, [])
        assert isinstance(err, BackpropagateError)


# =============================================================================
# EXCEPTION CATCHING TESTS
# =============================================================================

class TestExceptionCatching:
    """Tests for catching exceptions at different levels."""

    def test_catch_all_with_base(self):
        """Should catch all errors with BackpropagateError."""
        exceptions = [
            ConfigurationError("config"),
            InvalidSettingError("s", "v", "e"),
            DatasetError("dataset"),
            DatasetNotFoundError("/path"),
            TrainingError("training"),
            ModelLoadError("m", "r"),
            ExportError("export"),
            GGUFExportError("reason"),
            GPUError("gpu"),
            GPUMemoryError(),
            SLAOError("slao"),
            SLAOMergeError("reason"),
            BatchOperationError("op", 10, 1, []),
        ]

        for exc in exceptions:
            # All should be catchable as BackpropagateError
            try:
                raise exc
            except BackpropagateError as e:
                assert e is exc
            except Exception:
                pytest.fail(f"{type(exc).__name__} not caught as BackpropagateError")

    def test_catch_category(self):
        """Should catch by category (e.g., all dataset errors)."""
        dataset_errors = [
            DatasetError("base"),
            DatasetNotFoundError("/path"),
            DatasetParseError("parse"),
            DatasetValidationError("validation", errors=[]),
            DatasetFormatError("format"),
        ]

        for exc in dataset_errors:
            try:
                raise exc
            except DatasetError as e:
                assert e is exc
            except Exception:
                pytest.fail(f"{type(exc).__name__} not caught as DatasetError")

    def test_catch_specific(self):
        """Should catch specific exception types."""
        try:
            raise GPUTemperatureError(95.0, 85.0)
        except GPUTemperatureError as e:
            assert e.temperature == 95.0
        except GPUError:
            pytest.fail("Caught as GPUError instead of GPUTemperatureError")


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================

class TestExportsFromModule:
    """Tests for module exports."""

    def test_all_exceptions_exported(self):
        """Should export all exception classes."""
        from backpropagate import exceptions

        expected = [
            "BackpropagateError",
            "ConfigurationError",
            "InvalidSettingError",
            "DatasetError",
            "DatasetNotFoundError",
            "DatasetParseError",
            "DatasetValidationError",
            "DatasetFormatError",
            "TrainingError",
            "ModelLoadError",
            "TrainingAbortedError",
            "CheckpointError",
            "ExportError",
            "LoRAExportError",
            "GGUFExportError",
            "MergeExportError",
            "OllamaRegistrationError",
            "GPUError",
            "GPUNotAvailableError",
            "GPUMemoryError",
            "GPUTemperatureError",
            "GPUMonitoringError",
            "SLAOError",
            "SLAOMergeError",
            "SLAOCheckpointError",
            "BatchOperationError",
        ]

        for name in expected:
            assert hasattr(exceptions, name), f"Missing export: {name}"

    def test_in_all(self):
        """Should be in __all__ list."""
        from backpropagate import exceptions

        for name in exceptions.__all__:
            assert hasattr(exceptions, name), f"{name} in __all__ but not defined"

    def test_importable_from_package(self):
        """Should be importable from backpropagate package."""
        from backpropagate import (
            BackpropagateError,
            ConfigurationError,
            DatasetError,
            TrainingError,
            ExportError,
            GPUError,
            SLAOError,
        )

        assert BackpropagateError is not None
        assert ConfigurationError is not None
        assert DatasetError is not None
        assert TrainingError is not None
        assert ExportError is not None
        assert GPUError is not None
        assert SLAOError is not None
