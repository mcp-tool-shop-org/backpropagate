"""
Backpropagate - Custom Exception Hierarchy
==========================================

Production-ready exception classes for clear error handling and debugging.

Exception Hierarchy:
    BackpropagateError (base)
    ├── UserInputError                 (user-actionable input/validation errors)
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
    │   ├── LoRAExportError
    │   ├── GGUFExportError
    │   ├── MergeExportError
    │   └── OllamaRegistrationError
    ├── GPUError
    │   ├── GPUNotAvailableError
    │   ├── GPUMemoryError
    │   ├── GPUTemperatureError
    │   └── GPUMonitoringError
    ├── SLAOError
    │   ├── SLAOMergeError
    │   └── SLAOCheckpointError
    └── Utilities
        ├── BatchOperationError        (aggregate of failures across N items)
        └── PartialSuccess             (operation completed with some failures)

Structured shape (Ship Gate B1):
    Every exception carries optional ``code`` (machine-readable identifier),
    ``cause`` (chained exception via ``__cause__``), and ``retryable`` (bool).
    Defaults preserve existing call-site behavior; subclasses may set sensible
    defaults (e.g. GPU temperature errors are retryable, auth errors are not).

Usage:
    from backpropagate.exceptions import DatasetNotFoundError, TrainingError

    try:
        trainer.train(dataset="missing.jsonl")
    except DatasetNotFoundError as e:
        print(f"Dataset not found: {e.path}  [{e.code}]")
    except TrainingError as e:
        print(f"Training failed: {e}")
"""

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Cause categories for ModelLoadError (C-CLI-006). When a caller knows the
# nature of the underlying HF Hub failure it can pass ``cause_category`` so
# the user gets a remediation hint tailored to the failure mode instead of
# the generic "check model name + network" line. Backend raise sites in
# ``trainer.py`` are expected to pattern-match HTTP status / exception text
# and pass the appropriate category; today this is a forward-compatible
# extension point (default behavior preserved when the kwarg is omitted).
ModelLoadCauseCategory = Literal[
    "auth", "not_found", "network", "version", "unknown"
]

__all__ = [
    # Base
    "BackpropagateError",
    "UserInputError",
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
    "ModelLoadCauseCategory",
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
    "PartialSuccess",
    # Catalog
    "ERROR_CODES",
    "print_error_code_catalog",
]


# =============================================================================
# ERROR CODE CATALOG (C-CLI-005, C-CLI-014)
# =============================================================================
# Single source of truth for the machine-readable codes Ship Gate B1 introduced.
# Each entry carries:
#   description: one-liner explaining the failure category
#   default_hint: fallback suggestion when the raise site does not pass an
#                 explicit ``suggestion`` argument
#   retryable:   "yes" / "no" / "sometimes" — operators automating retry loops
#                can scrape this to decide whether backoff is meaningful
#
# Adding a new exception subclass? Add a row here FIRST so the discipline of
# "central catalog → raise site" stays intact. Operators reading a code in
# their terminal can run ``backprop info --error-codes`` to look it up.

ERROR_CODES: dict[str, dict[str, str]] = {
    # User input
    "INPUT_VALIDATION_FAILED": {
        "description": "A CLI argument failed validation (wrong type, out of range, malformed).",
        "default_hint": "Re-read the relevant --help and pass a valid value.",
        "retryable": "no",
    },
    "INPUT_AUTH_REQUIRED": {
        "description": "An operation required --auth credentials but they were not supplied.",
        "default_hint": "Pass --auth user:pass on the CLI, or set BACKPROPAGATE_UI_AUTH=user:pass in the environment. See handbook/security.md for the full auth contract.",
        "retryable": "no",
    },
    "INPUT_AUTH_INVALID_SHAPE": {
        "description": "BACKPROPAGATE_UI_AUTH credentials malformed (expected user:pass).",
        "default_hint": "Pass --auth user:pass with a non-empty username and password separated by a single colon.",
        "retryable": "no",
    },
    "INPUT_PATH_TRAVERSAL": {
        "description": "A user-supplied path escaped the allowed base directory or contained a rejected '..' pattern.",
        "default_hint": "Pass an absolute path inside the sandbox base, or remove the '..' segments.",
        "retryable": "no",
    },
    # Dataset
    "INPUT_DATASET_INVALID": {
        "description": "A dataset reference is invalid in a way the loader cannot self-describe.",
        "default_hint": "Inspect the dataset path and format; pass --verbose for the full traceback.",
        "retryable": "no",
    },
    "INPUT_DATASET_NOT_FOUND": {
        "description": "The dataset file or HuggingFace name could not be located.",
        "default_hint": "Verify the path is reachable from the current working directory, or that the HuggingFace dataset exists.",
        "retryable": "no",
    },
    "INPUT_DATASET_PARSE_FAILED": {
        "description": "The dataset could not be parsed as JSONL / CSV / supported format.",
        "default_hint": "Open the offending line in a text editor; check for unescaped quotes, BOM, or wrong encoding.",
        "retryable": "no",
    },
    "INPUT_DATASET_VALIDATION_FAILED": {
        "description": "Dataset structure failed semantic validation (missing fields, empty rows, etc.).",
        "default_hint": "Run with --verbose to see the validation error list.",
        "retryable": "no",
    },
    "INPUT_DATASET_FORMAT_UNSUPPORTED": {
        "description": "Dataset format is unsupported or could not be auto-detected.",
        "default_hint": "Convert to one of the supported formats (JSONL, ShareGPT, Alpaca, OpenAI).",
        "retryable": "no",
    },
    # Configuration
    "CONFIG_INVALID": {
        "description": "Configuration object is invalid or malformed.",
        "default_hint": "Re-check the BACKPROPAGATE_* env vars or your .env file.",
        "retryable": "no",
    },
    "CONFIG_INVALID_SETTING": {
        "description": "A specific configuration field has an invalid value.",
        "default_hint": "Compare the value against the expected type described in the error.",
        "retryable": "no",
    },
    # Training / runtime
    "RUNTIME_TRAINING_FAILED": {
        "description": "Training crashed for an internal reason (model bug, library mismatch, etc.).",
        "default_hint": "Run with --verbose for the full traceback; check transformers / unsloth versions.",
        "retryable": "sometimes",
    },
    "RUNTIME_UI_AUTH_NOT_ENFORCED": {
        "description": "Web UI authentication was requested but the runtime does not yet enforce it.",
        "default_hint": "Until middleware lands, do not pass --auth or --share. Use SSH port-forwarding for remote access: ssh -L 7860:localhost:7860 <host>",
        "retryable": "no",
    },
    "UI_OUTPUT_DIR_FORBIDDEN": {
        "description": "BACKPROPAGATE_UI__OUTPUT_DIR points at a forbidden base path (e.g., /etc, ~/.ssh).",
        "default_hint": "Set BACKPROPAGATE_UI__OUTPUT_DIR to a writable directory under your home or workspace.",
        "retryable": "no",
    },
    "DEP_MODEL_LOAD_FAILED": {
        "description": "Failed to load the model or tokenizer from disk or HuggingFace Hub.",
        "default_hint": "Verify model name, HF token, and network access to huggingface.co.",
        "retryable": "yes",
    },
    "RUNTIME_TRAINING_ABORTED": {
        "description": "Training stopped early (user interrupt, GPU fault, watchdog).",
        "default_hint": "Inspect the abort reason; the last checkpoint may be usable.",
        "retryable": "sometimes",
    },
    "STATE_CHECKPOINT_INVALID": {
        "description": "Checkpoint save or load failed (disk full, corrupt file, permission).",
        "default_hint": "Verify free disk space and write permission on the output directory.",
        "retryable": "sometimes",
    },
    "PEFT_API_INCOMPATIBLE": {
        "description": "Installed peft version does not expose the API Backpropagate needs (lora_A/lora_B parameter access OR get_adapter_state_dict/load_adapter_state_dict) — SLAO startup invariant failed.",
        "default_hint": "Upgrade peft: `pip install -U 'peft>=0.7.0'`. Verify with `python -c 'import peft; print(peft.__version__)'`.",
        "retryable": "no",
    },
    # Export
    "RUNTIME_EXPORT_FAILED": {
        "description": "Generic export failure not covered by a more specific code.",
        "default_hint": "Run with --verbose for the full traceback.",
        "retryable": "no",
    },
    "RUNTIME_LORA_EXPORT_FAILED": {
        "description": "LoRA adapter export failed.",
        "default_hint": "Confirm the model has LoRA adapters attached and the output dir is writable.",
        "retryable": "no",
    },
    "RUNTIME_GGUF_EXPORT_FAILED": {
        "description": "GGUF quantization / export failed.",
        "default_hint": "Ensure Unsloth is installed or llama.cpp convert script is on PATH.",
        "retryable": "no",
    },
    "RUNTIME_MERGE_EXPORT_FAILED": {
        "description": "Merge-and-export step failed.",
        "default_hint": "Re-run with --verbose; verify enough VRAM for the merge.",
        "retryable": "no",
    },
    "DEP_OLLAMA_REGISTRATION_FAILED": {
        "description": "Failed to register the exported model with the local Ollama daemon.",
        "default_hint": "Ensure Ollama is installed and running (`ollama --version`); see https://ollama.ai.",
        "retryable": "yes",
    },
    "HUB_PUSH_INVALID_REPO": {
        "description": "Hugging Face repo_id failed pre-network validation (shape, control chars, '..' segment, length > 96).",
        "default_hint": "Pass --repo owner/name with only [A-Za-z0-9._-] in each segment; total length <= 96.",
        "retryable": "no",
    },
    "HUB_PUSH_NOT_FOUND": {
        "description": "Hugging Face Hub returned 404 for the push target — repo does not exist or token cannot see it.",
        "default_hint": "Verify the repo_id spelling, OR pass --create-repo so backpropagate creates it, OR confirm the token has write scope.",
        "retryable": "no",
    },
    "HUB_PUSH_NETWORK": {
        "description": "Hub upload failed due to a network or 5xx error (transient).",
        "default_hint": "Retry in a few minutes; check https://status.huggingface.co. If persistent, set HTTPS_PROXY for corporate proxies.",
        "retryable": "yes",
    },
    "HUB_PUSH_UNKNOWN": {
        "description": "Hub upload failed for a reason backpropagate could not categorise.",
        "default_hint": "Re-run with --verbose for the full traceback; check huggingface_hub version; file a bug if the signature repeats.",
        "retryable": "sometimes",
    },
    # GPU
    "RUNTIME_GPU_ERROR": {
        "description": "Generic GPU failure not covered by a more specific code.",
        "default_hint": "Check `nvidia-smi`; restart the process if VRAM is wedged.",
        "retryable": "sometimes",
    },
    "DEP_GPU_NOT_AVAILABLE": {
        "description": "No CUDA-capable GPU detected.",
        "default_hint": "Install CUDA-enabled PyTorch; verify `torch.cuda.is_available()` returns True.",
        "retryable": "no",
    },
    "RUNTIME_GPU_OOM": {
        "description": "GPU ran out of memory (VRAM).",
        "default_hint": "Reduce --batch-size, enable gradient checkpointing, or use a smaller model.",
        "retryable": "no",
    },
    "RUNTIME_OOM_RECOVERY_EXHAUSTED": {
        "description": "OOM auto-recovery ran out of options (hit batch_size=1 with no further degradation path, or the effective-batch ceiling has been reached).",
        "default_hint": "Use a smaller model, reduce max_seq_length, enable gradient_checkpointing, or quantize. Set oom_recovery=False to make OOMs hard-fail immediately.",
        "retryable": "no",
    },
    "RUNTIME_OOM_ADJACENT": {
        "description": "A CUDA error that looks adjacent to OOM (CUBLAS_STATUS_ALLOC_FAILED / CUDNN_STATUS_NOT_INITIALIZED / NCCL post-VRAM-exhaustion) bypassed the strict OOM matcher.",
        "default_hint": "Same remediation as RUNTIME_GPU_OOM. If this signature appears frequently, file a bug so the OOM matcher can learn it.",
        "retryable": "sometimes",
    },
    "RUNTIME_GPU_TEMPERATURE_CRITICAL": {
        "description": "GPU temperature exceeded the safety threshold.",
        "default_hint": "Wait for the GPU to cool; improve case airflow; lower --batch-size.",
        "retryable": "yes",
    },
    "RUNTIME_GPU_MONITORING_FAILED": {
        "description": "GPU monitoring (pynvml) probe failed.",
        "default_hint": "Install pynvml: `pip install pynvml`.",
        "retryable": "sometimes",
    },
    # SLAO
    "RUNTIME_SLAO_ERROR": {
        "description": "Generic SLAO multi-run merge error.",
        "default_hint": "Run with --verbose; inspect the merge_history.json for the failing run.",
        "retryable": "no",
    },
    "RUNTIME_SLAO_MERGE_FAILED": {
        "description": "SLAO merge step failed for a specific run.",
        "default_hint": "Check the run's LoRA dir for shape mismatches.",
        "retryable": "no",
    },
    "SLAO_MERGE_DIVERGED": {
        "description": "SLAO merge produced non-finite (NaN/inf) weights — accumulator corrupted, likely bf16 underflow or exploding gradients in the upstream run.",
        "default_hint": "Rewind to the previous healthy checkpoint; inspect the latest run for bf16 underflow, exploding gradients, or LR-too-high. Lower learning_rate or switch optim to AdamW.",
        "retryable": "no",
    },
    "STATE_SLAO_CHECKPOINT_INVALID": {
        "description": "SLAO checkpoint save or load failed.",
        "default_hint": "Verify free disk space and write permission on the checkpoint directory.",
        "retryable": "sometimes",
    },
    # Aggregate / partial
    "PARTIAL_BATCH_OPERATION": {
        "description": "A batched operation completed with per-item failures (aggregate).",
        "default_hint": "Inspect the error list in the exception details; retry only failed items.",
        "retryable": "sometimes",
    },
    "PARTIAL_SUCCESS": {
        "description": "Operation completed but some sub-steps failed (exit code 3).",
        "default_hint": "Outputs may be usable; review the warning summary above.",
        "retryable": "no",
    },
}


# Cause-category specific hints for ModelLoadError (C-CLI-006). Each maps a
# coarse failure category to a concrete next step. ``unknown`` falls back to
# today's generic hint so behavior is preserved when the category is omitted.
_MODEL_LOAD_HINTS: dict[str, str] = {
    "auth": (
        "Check your HuggingFace token: `huggingface-cli login`. If the model is "
        "gated, accept the terms on its model page."
    ),
    "not_found": (
        "Verify the model name spelling (HF model names are case-sensitive). "
        "Common: 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit'. Search: "
        "https://huggingface.co/models"
    ),
    "network": (
        "Check internet access to huggingface.co. If you are behind a proxy, "
        "set HTTPS_PROXY and consider HF_HUB_DISABLE_TELEMETRY=1."
    ),
    "version": (
        "Update transformers / unsloth / peft: "
        "`pip install -U backpropagate[unsloth]`."
    ),
    "unknown": (
        "Check that the model name is correct and you have network access"
    ),
}


def print_error_code_catalog() -> None:
    """Print the ``ERROR_CODES`` registry as a human-readable table.

    Called from ``backprop info --error-codes`` for operator reference. The
    output is tab-delimited so it composes with grep / awk in CI scripts but
    stays readable in a terminal.
    """
    # Header
    print("code\tretryable\tdescription")
    print("----\t---------\t-----------")
    for code in sorted(ERROR_CODES.keys()):
        entry = ERROR_CODES[code]
        print(f"{code}\t{entry['retryable']}\t{entry['description']}")
        # Indent the default hint on a continuation line so the column shape
        # stays scannable.
        print(f"\t\t  hint: {entry['default_hint']}")


# =============================================================================
# BASE EXCEPTION
# =============================================================================

class BackpropagateError(Exception):
    """
    Base exception for all Backpropagate errors.

    All custom exceptions inherit from this class, allowing users to catch
    all library errors with a single except clause if desired.

    Attributes:
        message: Human-readable error description.
        details: Optional dict with additional context for debugging.
        suggestion: Optional suggestion for how to fix the error.
        code: Machine-readable identifier (Ship Gate B1). Intended to be
              STABLE across class renames — every subclass sets an explicit
              value drawn from the Ship Gate registry prefixes
              (``IO_``, ``CONFIG_``, ``PERM_``, ``DEP_``, ``RUNTIME_``,
              ``PARTIAL_``, ``INPUT_``, ``STATE_``). When ``code`` is left
              ``None`` we emit a debug log so missing codes stay visible
              (the previous auto-derived ``cls.__name__.upper()`` default
              silently drifted on every class rename).
        cause: The underlying exception, if any. Also chained via
               ``raise X from cause`` semantics (``self.__cause__``).
        retryable: Whether a caller can retry the operation without user
                   intervention (e.g. GPU temperature throttling -> True,
                   missing dataset file -> False). Defaults to ``False``.

    Backward compatibility:
        ``code``, ``cause``, and ``retryable`` are optional keyword arguments.
        Existing callers that only pass ``message`` / ``details`` /
        ``suggestion`` work unchanged; ``code`` will be ``None`` (with a
        debug log line) instead of a derived upper-case class name.
    """

    def __init__(
        self,
        message: str,
        details: dict | None = None,
        suggestion: str | None = None,
        code: str | None = None,
        cause: Exception | None = None,
        retryable: bool = False,
    ):
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion
        # B-004: drop the auto-derived `cls.__name__.upper()` default — it
        # silently broke the documented "stable across renames" contract on
        # every subclass rename. Subclasses now set an explicit code; a
        # missing code stays visible as `None` (debug-logged) instead of
        # invisibly drifting.
        self.code = code
        if code is None:
            # Debug-level (not warn) — base BackpropagateError is intentionally
            # constructed in many user-facing codepaths without a stable code.
            logger.debug(
                "BackpropagateError instance has no stable `code`: "
                "class=%s message=%r",
                self.__class__.__name__,
                message,
            )
        self.cause = cause
        self.retryable = retryable

        # Chain the underlying cause so traceback machinery picks it up.
        if cause is not None:
            self.__cause__ = cause

        # Build full message
        full_message = message
        if suggestion:
            full_message += f"\n\nSuggestion: {suggestion}"

        super().__init__(full_message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"

    def to_dict(self) -> dict:
        """Serialize the structured envelope for logging or transport."""
        envelope: dict[str, Any] = {
            "type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.suggestion:
            envelope["suggestion"] = self.suggestion
        if self.details:
            envelope["details"] = self.details
        if self.cause is not None:
            envelope["cause"] = f"{type(self.cause).__name__}: {self.cause}"
        return envelope


# =============================================================================
# USER INPUT ERROR (CLI exit code 1 — user-actionable)
# =============================================================================

class UserInputError(BackpropagateError):
    """
    Raised when a user-supplied argument, path, or value is invalid.

    Maps to Ship Gate B2 exit code ``1`` (user error). Distinct from
    ConfigurationError (which covers persisted settings) — UserInputError
    represents per-invocation argument or flag problems that the user can
    fix by adjusting their command line.
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        hint: str | None = None,
        details: dict | None = None,
    ):
        super().__init__(
            message,
            details=details,
            suggestion=hint,
            code=code or "INPUT_VALIDATION_FAILED",
            retryable=False,
        )


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(BackpropagateError):
    """Invalid configuration or settings."""

    def __init__(
        self,
        message: str,
        details: dict | None = None,
        suggestion: str | None = None,
        code: str | None = None,
        cause: Exception | None = None,
        retryable: bool = False,
    ):
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            code=code or "CONFIG_INVALID",
            cause=cause,
            retryable=retryable,
        )


class InvalidSettingError(ConfigurationError):
    """A specific setting has an invalid value."""

    def __init__(
        self,
        setting_name: str,
        value: Any,
        expected: str,
        suggestion: str | None = None,
    ):
        self.setting_name = setting_name
        self.value = value
        self.expected = expected

        message = f"Invalid value for '{setting_name}': got {value!r}, expected {expected}"
        super().__init__(
            message,
            details={"setting": setting_name, "value": value, "expected": expected},
            suggestion=suggestion,
            code="CONFIG_INVALID_SETTING",
        )


# =============================================================================
# DATASET ERRORS
# =============================================================================

class DatasetError(BackpropagateError):
    """Base class for dataset-related errors."""

    def __init__(
        self,
        message: str,
        details: dict | None = None,
        suggestion: str | None = None,
        code: str | None = None,
        cause: Exception | None = None,
        retryable: bool = False,
    ):
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            code=code or "INPUT_DATASET_INVALID",
            cause=cause,
            retryable=retryable,
        )


class DatasetNotFoundError(DatasetError):
    """Dataset file or resource not found."""

    def __init__(self, path: str, suggestion: str | None = None):
        self.path = path
        super().__init__(
            f"Dataset not found: {path}",
            details={"path": str(path)},
            suggestion=suggestion or f"Check that the file exists at: {path}",
            code="INPUT_DATASET_NOT_FOUND",
            retryable=False,
        )


class DatasetParseError(DatasetError):
    """Failed to parse dataset content."""

    def __init__(
        self,
        message: str,
        path: str | None = None,
        line_number: int | None = None,
        suggestion: str | None = None,
    ):
        self.path = path
        self.line_number = line_number

        details: dict[str, str | int] = {}
        if path:
            details["path"] = str(path)
        if line_number is not None:
            details["line_number"] = line_number
            message = f"{message} (line {line_number})"

        super().__init__(
            message,
            details=details,
            suggestion=suggestion or "Check that the file contains valid JSON/CSV data",
            code="INPUT_DATASET_PARSE_FAILED",
        )


class DatasetValidationError(DatasetError):
    """Dataset validation failed."""

    def __init__(
        self,
        message: str,
        errors: list[str] | None = None,
        suggestion: str | None = None,
    ):
        self.errors = errors or []

        full_message = message
        if errors:
            full_message += "\n\nValidation errors:\n" + "\n".join(f"  - {e}" for e in errors[:10])
            if len(errors) > 10:
                full_message += f"\n  ... and {len(errors) - 10} more"

        super().__init__(
            full_message,
            details={"error_count": len(self.errors), "errors": self.errors[:20]},
            suggestion=suggestion,
            code="INPUT_DATASET_VALIDATION_FAILED",
        )


class DatasetFormatError(DatasetError):
    """Dataset format is unsupported or cannot be detected."""

    def __init__(
        self,
        message: str,
        detected_format: str | None = None,
        supported_formats: list[str] | None = None,
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
            code="INPUT_DATASET_FORMAT_UNSUPPORTED",
        )


# =============================================================================
# TRAINING ERRORS
# =============================================================================

class TrainingError(BackpropagateError):
    """Base class for training-related errors."""

    def __init__(
        self,
        message: str,
        details: dict | None = None,
        suggestion: str | None = None,
        code: str | None = None,
        cause: Exception | None = None,
        retryable: bool = False,
    ):
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            code=code or "RUNTIME_TRAINING_FAILED",
            cause=cause,
            retryable=retryable,
        )


class ModelLoadError(TrainingError):
    """Failed to load model or tokenizer.

    The optional ``cause_category`` argument lets the raise site classify the
    underlying failure (auth / not_found / network / version / unknown). When
    set, the default suggestion is drawn from the per-category remediation
    table instead of the generic "check name + network" line — addresses the
    Stage C audit gap that 401, 404, and network errors all got the same hint.

    Backend raise sites in ``trainer.py`` are responsible for inspecting the
    underlying transformers / HF Hub exception and passing the right category.
    Today this is forward-compatible: callers that omit ``cause_category``
    behave byte-identically to pre-Stage-C ModelLoadError.
    """

    def __init__(
        self,
        model_name: str,
        reason: str,
        suggestion: str | None = None,
        cause_category: ModelLoadCauseCategory | None = None,
    ):
        self.model_name = model_name
        self.reason = reason
        self.cause_category = cause_category

        # Hint precedence:
        #   1. explicit ``suggestion`` arg wins (caller knows best)
        #   2. otherwise, look up by cause_category if provided
        #   3. otherwise, fall back to the pre-Stage-C generic line
        if suggestion is None and cause_category is not None:
            effective_hint = _MODEL_LOAD_HINTS.get(
                cause_category, _MODEL_LOAD_HINTS["unknown"]
            )
        else:
            effective_hint = suggestion or _MODEL_LOAD_HINTS["unknown"]

        details = {"model_name": model_name, "reason": reason}
        if cause_category is not None:
            details["cause_category"] = cause_category

        super().__init__(
            f"Failed to load model '{model_name}': {reason}",
            details=details,
            suggestion=effective_hint,
            code="DEP_MODEL_LOAD_FAILED",
            # Most ModelLoadError instances come from transient network
            # failures (HF Hub 503 / timeout). Callers may inspect
            # ``details['reason']`` or ``cause_category`` to decide whether
            # a retry is worth attempting (e.g. auth/not_found ⇒ don't retry).
            retryable=True,
        )


class TrainingAbortedError(TrainingError):
    """Training was aborted (user interrupt, GPU issue, etc.)."""

    def __init__(
        self,
        reason: str,
        steps_completed: int = 0,
        checkpoint_path: str | None = None,
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
            code="RUNTIME_TRAINING_ABORTED",
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
            code="STATE_CHECKPOINT_INVALID",
        )


# =============================================================================
# EXPORT ERRORS
# =============================================================================

class ExportError(BackpropagateError):
    """Base class for model export errors."""

    def __init__(
        self,
        message: str,
        details: dict | None = None,
        suggestion: str | None = None,
        code: str | None = None,
        cause: Exception | None = None,
        retryable: bool = False,
    ):
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            code=code or "RUNTIME_EXPORT_FAILED",
            cause=cause,
            retryable=retryable,
        )


class LoRAExportError(ExportError):
    """Failed to export LoRA adapter."""

    def __init__(
        self,
        reason: str,
        output_path: str | None = None,
        suggestion: str | None = None,
    ):
        self.reason = reason
        self.output_path = output_path

        message = f"LoRA export failed: {reason}"

        super().__init__(
            message,
            details={"output_path": output_path},
            suggestion=suggestion or "Check that the model has LoRA adapters attached",
            code="RUNTIME_LORA_EXPORT_FAILED",
        )


class GGUFExportError(ExportError):
    """Failed to export model to GGUF format."""

    def __init__(
        self,
        reason: str,
        output_path: str | None = None,
        quantization: str | None = None,
        suggestion: str | None = None,
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
            code="RUNTIME_GGUF_EXPORT_FAILED",
        )


class MergeExportError(ExportError):
    """Failed to merge and export model."""

    def __init__(self, reason: str, suggestion: str | None = None):
        super().__init__(
            f"Merge export failed: {reason}",
            suggestion=suggestion,
            code="RUNTIME_MERGE_EXPORT_FAILED",
        )


class OllamaRegistrationError(ExportError):
    """Failed to register model with Ollama."""

    def __init__(
        self,
        model_name: str,
        reason: str,
        suggestion: str | None = None,
    ):
        self.model_name = model_name

        super().__init__(
            f"Failed to register '{model_name}' with Ollama: {reason}",
            details={"model_name": model_name},
            suggestion=suggestion or "Ensure Ollama is installed and running (https://ollama.ai)",
            code="DEP_OLLAMA_REGISTRATION_FAILED",
            # Ollama daemon failures are typically transient (service
            # starting, restarting, or temporarily unreachable). Callers
            # implementing backoff should treat this as retryable.
            retryable=True,
        )


# =============================================================================
# GPU ERRORS
# =============================================================================

class GPUError(BackpropagateError):
    """Base class for GPU-related errors."""

    def __init__(
        self,
        message: str,
        details: dict | None = None,
        suggestion: str | None = None,
        code: str | None = None,
        cause: Exception | None = None,
        retryable: bool = False,
    ):
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            code=code or "RUNTIME_GPU_ERROR",
            cause=cause,
            retryable=retryable,
        )


class GPUNotAvailableError(GPUError):
    """No GPU available or CUDA not configured."""

    def __init__(self, suggestion: str | None = None):
        super().__init__(
            "No CUDA GPU available",
            suggestion=suggestion or "Ensure CUDA is installed and a compatible GPU is present",
            code="DEP_GPU_NOT_AVAILABLE",
        )


class GPUMemoryError(GPUError):
    """GPU memory (VRAM) error."""

    def __init__(
        self,
        required_gb: float | None = None,
        available_gb: float | None = None,
        suggestion: str | None = None,
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
            code="RUNTIME_GPU_OOM",
            # Not retryable by default — if the model literally won't fit, a
            # retry just OOMs again. Callers who know they're dealing with a
            # transient peak-allocation can construct a subclass or rebuild
            # with retryable=True.
            retryable=False,
        )


class GPUTemperatureError(GPUError):
    """GPU temperature exceeded safe limits."""

    def __init__(
        self,
        temperature: float,
        threshold: float,
        suggestion: str | None = None,
    ):
        self.temperature = temperature
        self.threshold = threshold

        super().__init__(
            f"GPU temperature critical: {temperature}°C (threshold: {threshold}°C)",
            details={"temperature": temperature, "threshold": threshold},
            suggestion=suggestion or "Wait for GPU to cool down or improve cooling",
            code="RUNTIME_GPU_TEMPERATURE_CRITICAL",
            # Retryable: temperature naturally recovers as the GPU cools.
            retryable=True,
        )


class GPUMonitoringError(GPUError):
    """Failed to monitor GPU status."""

    def __init__(self, reason: str, suggestion: str | None = None):
        super().__init__(
            f"GPU monitoring failed: {reason}",
            suggestion=suggestion or "Install pynvml for GPU monitoring: pip install pynvml",
            code="RUNTIME_GPU_MONITORING_FAILED",
        )


# =============================================================================
# SLAO ERRORS
# =============================================================================

class SLAOError(BackpropagateError):
    """Base class for SLAO merging errors."""

    def __init__(
        self,
        message: str,
        details: dict | None = None,
        suggestion: str | None = None,
        code: str | None = None,
        cause: Exception | None = None,
        retryable: bool = False,
    ):
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            code=code or "RUNTIME_SLAO_ERROR",
            cause=cause,
            retryable=retryable,
        )


class SLAOMergeError(SLAOError):
    """Failed to merge LoRA weights using SLAO."""

    def __init__(
        self,
        reason: str,
        run_index: int | None = None,
        suggestion: str | None = None,
    ):
        self.run_index = run_index

        message = f"SLAO merge failed: {reason}"
        if run_index is not None:
            message = f"SLAO merge failed at run {run_index}: {reason}"

        super().__init__(
            message,
            details={"run_index": run_index},
            suggestion=suggestion,
            code="RUNTIME_SLAO_MERGE_FAILED",
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
            code="STATE_SLAO_CHECKPOINT_INVALID",
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
        errors: list[tuple],  # List of (index, exception) tuples
        suggestion: str | None = None,
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
            code="PARTIAL_BATCH_OPERATION",
        )

    @property
    def success_count(self) -> int:
        return self.total_items - self.failed_items

    @property
    def success_rate(self) -> float:
        return (self.success_count / self.total_items * 100) if self.total_items > 0 else 0.0


# =============================================================================
# PARTIAL SUCCESS (Ship Gate B2 exit code 3)
# =============================================================================

class PartialSuccess(BackpropagateError):
    """
    Raised when an operation completed with both successes and failures.

    Maps to Ship Gate B2 exit code ``3`` (partial success). Distinct from
    :class:`BatchOperationError` in intent: ``BatchOperationError`` is the
    raw aggregate from a per-item loop; ``PartialSuccess`` is the CLI-facing
    signal that "the work ran to completion, but not every unit succeeded".

    A handler that catches ``BatchOperationError`` may choose to re-raise as
    ``PartialSuccess`` once it has decided the overall operation succeeded
    enough to warrant exit code 3 instead of exit code 2.
    """

    def __init__(
        self,
        message: str,
        total_items: int,
        succeeded: int,
        failed: int,
        suggestion: str | None = None,
        details: dict | None = None,
    ):
        self.total_items = total_items
        self.succeeded = succeeded
        self.failed = failed

        merged_details = {
            "total_items": total_items,
            "succeeded": succeeded,
            "failed": failed,
        }
        if details:
            merged_details.update(details)

        super().__init__(
            message,
            details=merged_details,
            suggestion=suggestion,
            code="PARTIAL_SUCCESS",
            retryable=False,
        )
