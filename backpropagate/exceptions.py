"""
Backpropagate - Custom Exception Hierarchy
==========================================

Production-ready exception classes for clear error handling and debugging.

Exception Hierarchy:
    BackpropagateError (base)
    ├── ConfigurationError
    │   └── InvalidSettingError
    ├── DatasetError
    │   ├── DatasetNotFoundError
    │   ├── DatasetParseError
    │   ├── DatasetValidationError
    │   └── DatasetFormatError
    ├── TrainingError
    │   ├── ModelLoadError
    │   ├── TrainingAbortedError
    │   └── CheckpointError
    ├── ExportError
    │   ├── GGUFExportError
    │   ├── MergeExportError
    │   └── OllamaRegistrationError
    ├── GPUError
    │   ├── GPUNotAvailableError
    │   ├── GPUMemoryError
    │   ├── GPUTemperatureError
    │   └── GPUMonitoringError
    └── SLAOError
        ├── SLAOMergeError
        └── SLAOCheckpointError

Usage:
    from backpropagate.exceptions import DatasetNotFoundError, TrainingError

    try:
        trainer.train(dataset="missing.jsonl")
    except DatasetNotFoundError as e:
        print(f"Dataset not found: {e.path}")
    except TrainingError as e:
        print(f"Training failed: {e}")
"""

from typing import Optional, Any, List
from pathlib import Path


__all__ = [
    # Base
    "BackpropagateError",
    # Configuration
    "ConfigurationError",
    "InvalidSettingError",
    # Dataset
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetParseError",
    "DatasetValidationError",
    "DatasetFormatError",
    # Training
    "TrainingError",
    "ModelLoadError",
    "TrainingAbortedError",
    "CheckpointError",
    # Export
    "ExportError",
    "LoRAExportError",
    "GGUFExportError",
    "MergeExportError",
    "OllamaRegistrationError",
    # GPU
    "GPUError",
    "GPUNotAvailableError",
    "GPUMemoryError",
    "GPUTemperatureError",
    "GPUMonitoringError",
    # SLAO
    "SLAOError",
    "SLAOMergeError",
    "SLAOCheckpointError",
    # Utilities
    "BatchOperationError",
]


# =============================================================================
# BASE EXCEPTION
# =============================================================================

class BackpropagateError(Exception):
    """
    Base exception for all Backpropagate errors.

    All custom exceptions inherit from this class, allowing users to catch
    all library errors with a single except clause if desired.

    Attributes:
        message: Human-readable error description
        details: Optional dict with additional context for debugging
        suggestion: Optional suggestion for how to fix the error
    """

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
        suggestion: Optional[str] = None,
    ):
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion

        # Build full message
        full_message = message
        if suggestion:
            full_message += f"\n\nSuggestion: {suggestion}"

        super().__init__(full_message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(BackpropagateError):
    """Invalid configuration or settings."""
    pass


class InvalidSettingError(ConfigurationError):
    """A specific setting has an invalid value."""

    def __init__(
        self,
        setting_name: str,
        value: Any,
        expected: str,
        suggestion: Optional[str] = None,
    ):
        self.setting_name = setting_name
        self.value = value
        self.expected = expected

        message = f"Invalid value for '{setting_name}': got {value!r}, expected {expected}"
        super().__init__(
            message,
            details={"setting": setting_name, "value": value, "expected": expected},
            suggestion=suggestion,
        )


# =============================================================================
# DATASET ERRORS
# =============================================================================

class DatasetError(BackpropagateError):
    """Base class for dataset-related errors."""
    pass


class DatasetNotFoundError(DatasetError):
    """Dataset file or resource not found."""

    def __init__(self, path: str, suggestion: Optional[str] = None):
        self.path = path
        super().__init__(
            f"Dataset not found: {path}",
            details={"path": str(path)},
            suggestion=suggestion or f"Check that the file exists at: {path}",
        )


class DatasetParseError(DatasetError):
    """Failed to parse dataset content."""

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        line_number: Optional[int] = None,
        suggestion: Optional[str] = None,
    ):
        self.path = path
        self.line_number = line_number

        details = {}
        if path:
            details["path"] = str(path)
        if line_number is not None:
            details["line_number"] = line_number
            message = f"{message} (line {line_number})"

        super().__init__(
            message,
            details=details,
            suggestion=suggestion or "Check that the file contains valid JSON/CSV data",
        )


class DatasetValidationError(DatasetError):
    """Dataset validation failed."""

    def __init__(
        self,
        message: str,
        errors: Optional[List[str]] = None,
        suggestion: Optional[str] = None,
    ):
        self.errors = errors or []

        full_message = message
        if errors:
            full_message += "\n\nValidation errors:\n" + "\n".join(f"  - {e}" for e in errors[:10])
            if len(errors) > 10:
                full_message += f"\n  ... and {len(errors) - 10} more"

        super().__init__(
            full_message,
            details={"error_count": len(self.errors), "errors": errors[:20]},
            suggestion=suggestion,
        )


class DatasetFormatError(DatasetError):
    """Dataset format is unsupported or cannot be detected."""

    def __init__(
        self,
        message: str,
        detected_format: Optional[str] = None,
        supported_formats: Optional[List[str]] = None,
    ):
        self.detected_format = detected_format
        self.supported_formats = supported_formats or []

        suggestion = None
        if supported_formats:
            suggestion = f"Supported formats: {', '.join(supported_formats)}"

        super().__init__(
            message,
            details={
                "detected_format": detected_format,
                "supported_formats": supported_formats,
            },
            suggestion=suggestion,
        )


# =============================================================================
# TRAINING ERRORS
# =============================================================================

class TrainingError(BackpropagateError):
    """Base class for training-related errors."""
    pass


class ModelLoadError(TrainingError):
    """Failed to load model or tokenizer."""

    def __init__(
        self,
        model_name: str,
        reason: str,
        suggestion: Optional[str] = None,
    ):
        self.model_name = model_name
        self.reason = reason

        super().__init__(
            f"Failed to load model '{model_name}': {reason}",
            details={"model_name": model_name, "reason": reason},
            suggestion=suggestion or "Check that the model name is correct and you have network access",
        )


class TrainingAbortedError(TrainingError):
    """Training was aborted (user interrupt, GPU issue, etc.)."""

    def __init__(
        self,
        reason: str,
        steps_completed: int = 0,
        checkpoint_path: Optional[str] = None,
    ):
        self.reason = reason
        self.steps_completed = steps_completed
        self.checkpoint_path = checkpoint_path

        message = f"Training aborted: {reason}"
        if steps_completed > 0:
            message += f" (completed {steps_completed} steps)"
        if checkpoint_path:
            message += f"\nLast checkpoint: {checkpoint_path}"

        super().__init__(
            message,
            details={
                "reason": reason,
                "steps_completed": steps_completed,
                "checkpoint_path": checkpoint_path,
            },
        )


class CheckpointError(TrainingError):
    """Failed to save or load checkpoint."""

    def __init__(
        self,
        operation: str,  # "save" or "load"
        path: str,
        reason: str,
    ):
        self.operation = operation
        self.path = path
        self.reason = reason

        super().__init__(
            f"Failed to {operation} checkpoint at '{path}': {reason}",
            details={"operation": operation, "path": path, "reason": reason},
        )


# =============================================================================
# EXPORT ERRORS
# =============================================================================

class ExportError(BackpropagateError):
    """Base class for model export errors."""
    pass


class LoRAExportError(ExportError):
    """Failed to export LoRA adapter."""

    def __init__(
        self,
        reason: str,
        output_path: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.reason = reason
        self.output_path = output_path

        message = f"LoRA export failed: {reason}"

        super().__init__(
            message,
            details={"output_path": output_path},
            suggestion=suggestion or "Check that the model has LoRA adapters attached",
        )


class GGUFExportError(ExportError):
    """Failed to export model to GGUF format."""

    def __init__(
        self,
        reason: str,
        output_path: Optional[str] = None,
        quantization: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.reason = reason
        self.output_path = output_path
        self.quantization = quantization

        message = f"GGUF export failed: {reason}"

        super().__init__(
            message,
            details={
                "output_path": output_path,
                "quantization": quantization,
            },
            suggestion=suggestion or "Ensure Unsloth is installed or llama.cpp convert script is available",
        )


class MergeExportError(ExportError):
    """Failed to merge and export model."""

    def __init__(self, reason: str, suggestion: Optional[str] = None):
        super().__init__(
            f"Merge export failed: {reason}",
            suggestion=suggestion,
        )


class OllamaRegistrationError(ExportError):
    """Failed to register model with Ollama."""

    def __init__(
        self,
        model_name: str,
        reason: str,
        suggestion: Optional[str] = None,
    ):
        self.model_name = model_name

        super().__init__(
            f"Failed to register '{model_name}' with Ollama: {reason}",
            details={"model_name": model_name},
            suggestion=suggestion or "Ensure Ollama is installed and running (https://ollama.ai)",
        )


# =============================================================================
# GPU ERRORS
# =============================================================================

class GPUError(BackpropagateError):
    """Base class for GPU-related errors."""
    pass


class GPUNotAvailableError(GPUError):
    """No GPU available or CUDA not configured."""

    def __init__(self, suggestion: Optional[str] = None):
        super().__init__(
            "No CUDA GPU available",
            suggestion=suggestion or "Ensure CUDA is installed and a compatible GPU is present",
        )


class GPUMemoryError(GPUError):
    """GPU memory (VRAM) error."""

    def __init__(
        self,
        required_gb: Optional[float] = None,
        available_gb: Optional[float] = None,
        suggestion: Optional[str] = None,
    ):
        self.required_gb = required_gb
        self.available_gb = available_gb

        message = "Insufficient GPU memory"
        if required_gb and available_gb:
            message = f"Insufficient GPU memory: need {required_gb:.1f}GB, have {available_gb:.1f}GB"

        super().__init__(
            message,
            details={"required_gb": required_gb, "available_gb": available_gb},
            suggestion=suggestion or "Try reducing batch size, using gradient checkpointing, or a smaller model",
        )


class GPUTemperatureError(GPUError):
    """GPU temperature exceeded safe limits."""

    def __init__(
        self,
        temperature: float,
        threshold: float,
        suggestion: Optional[str] = None,
    ):
        self.temperature = temperature
        self.threshold = threshold

        super().__init__(
            f"GPU temperature critical: {temperature}°C (threshold: {threshold}°C)",
            details={"temperature": temperature, "threshold": threshold},
            suggestion=suggestion or "Wait for GPU to cool down or improve cooling",
        )


class GPUMonitoringError(GPUError):
    """Failed to monitor GPU status."""

    def __init__(self, reason: str, suggestion: Optional[str] = None):
        super().__init__(
            f"GPU monitoring failed: {reason}",
            suggestion=suggestion or "Install pynvml for GPU monitoring: pip install pynvml",
        )


# =============================================================================
# SLAO ERRORS
# =============================================================================

class SLAOError(BackpropagateError):
    """Base class for SLAO merging errors."""
    pass


class SLAOMergeError(SLAOError):
    """Failed to merge LoRA weights using SLAO."""

    def __init__(
        self,
        reason: str,
        run_index: Optional[int] = None,
        suggestion: Optional[str] = None,
    ):
        self.run_index = run_index

        message = f"SLAO merge failed: {reason}"
        if run_index is not None:
            message = f"SLAO merge failed at run {run_index}: {reason}"

        super().__init__(
            message,
            details={"run_index": run_index},
            suggestion=suggestion,
        )


class SLAOCheckpointError(SLAOError):
    """Failed to save or load SLAO checkpoint."""

    def __init__(
        self,
        operation: str,
        path: str,
        reason: str,
    ):
        self.operation = operation
        self.path = path

        super().__init__(
            f"SLAO checkpoint {operation} failed at '{path}': {reason}",
            details={"operation": operation, "path": path},
        )


# =============================================================================
# BATCH OPERATION ERROR (for error aggregation)
# =============================================================================

class BatchOperationError(BackpropagateError):
    """
    Multiple errors occurred during a batch operation.

    Used for error aggregation pattern where we want to continue processing
    even when some items fail, then report all errors together.
    """

    def __init__(
        self,
        operation: str,
        total_items: int,
        failed_items: int,
        errors: List[tuple],  # List of (index, exception) tuples
        suggestion: Optional[str] = None,
    ):
        self.operation = operation
        self.total_items = total_items
        self.failed_items = failed_items
        self.errors = errors

        success_rate = ((total_items - failed_items) / total_items * 100) if total_items > 0 else 0

        message = (
            f"{operation} partially failed: {failed_items}/{total_items} items failed "
            f"({success_rate:.1f}% success rate)"
        )

        # Add first few errors
        if errors:
            message += "\n\nFirst errors:"
            for idx, err in errors[:5]:
                message += f"\n  [{idx}]: {type(err).__name__}: {err}"
            if len(errors) > 5:
                message += f"\n  ... and {len(errors) - 5} more errors"

        super().__init__(
            message,
            details={
                "operation": operation,
                "total_items": total_items,
                "failed_items": failed_items,
                "success_rate": success_rate,
                "error_count": len(errors),
            },
            suggestion=suggestion,
        )

    @property
    def success_count(self) -> int:
        return self.total_items - self.failed_items

    @property
    def success_rate(self) -> float:
        return (self.success_count / self.total_items * 100) if self.total_items > 0 else 0.0
