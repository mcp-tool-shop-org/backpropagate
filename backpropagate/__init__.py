"""
Backpropagate - Headless LLM Fine-Tuning
=========================================

A clean, production-ready interface for LLM fine-tuning that makes
complex training accessible through simple parameters and smart defaults.

v1.1.1: hotfix CI action SHAs (no user-facing changes vs 1.1.0)

Installation:
    pip install backpropagate              # Core only (minimal)
    pip install backpropagate[unsloth]     # + Unsloth 2x faster training
    pip install backpropagate[ui]          # + Reflex web UI (migrated from Gradio in v1.1.0)
    pip install backpropagate[standard]    # unsloth + ui (recommended)
    pip install backpropagate[full]        # Everything

Features (Core):
- QLoRA fine-tuning with smart defaults
- Windows-safe multiprocessing
- Auto VRAM detection for batch size
- Multiple export formats (LoRA, merged, GGUF)

Features (Optional):
- [unsloth] 2x faster training with 50% less VRAM
- [ui] Reflex web interface for training management (Radix UI, WebSocket state)
- [validation] Pydantic configuration validation
- [export] GGUF export for Ollama/llama.cpp
- [monitoring] WandB logging and system monitoring

Usage:
    # Core functionality (always available)
    from backpropagate import Trainer

    trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    trainer.train("my_data.jsonl", steps=100)
    trainer.save("./my-model")

    # Export to GGUF for Ollama
    trainer.export("gguf", quantization="q4_k_m")

    # UI (requires [ui] extra). The Reflex UI runs as a subprocess managed
    # by the CLI rather than a Python function, so launch via the CLI:
    #     backprop ui --port 7862
    # programmatically:
    #     from backpropagate.cli import cmd_ui
"""

from typing import Any

# Exceptions (custom error hierarchy)
from .exceptions import (
    # Base
    BackpropagateError,
    # Batch operations
    BatchOperationError,
    CheckpointError,
    # Configuration
    ConfigurationError,
    # Dataset
    DatasetError,
    DatasetFormatError,
    DatasetNotFoundError,
    DatasetParseError,
    DatasetValidationError,
    # Export
    ExportError,
    GGUFExportError,
    # GPU
    GPUError,
    GPUMemoryError,
    GPUMonitoringError,
    GPUNotAvailableError,
    GPUTemperatureError,
    InvalidSettingError,
    LoRAExportError,
    MergeExportError,
    ModelLoadError,
    OllamaRegistrationError,
    SLAOCheckpointError,
    # SLAO
    SLAOError,
    SLAOMergeError,
    TrainingAbortedError,
    # Training
    TrainingError,
)

# Security utilities
from .security import (
    PathTraversalError,
    SecurityWarning,
    check_torch_security,
    safe_path,
)

# UI Security utilities (production-hardened, framework-agnostic). The Reflex
# UI (canonical from v1.1.0) consumes these helpers via subprocess. The
# try/except guards against the [ui] extra being absent.
try:
    from .ui_security import (
        ALLOWED_DATASET_EXTENSIONS,
        DANGEROUS_EXTENSIONS,
        DEFAULT_SECURITY_CONFIG,
        EnhancedRateLimiter,
        FileValidator,
        SecurityConfig,
        log_security_event,
    )
except ImportError:
    # UI extra not installed — UI security helpers unavailable. Set every
    # exported name to None so downstream ``from backpropagate import X``
    # still resolves (caller must None-check) instead of AttributeError.
    SecurityConfig = None  # type: ignore
    DEFAULT_SECURITY_CONFIG = None  # type: ignore
    EnhancedRateLimiter = None  # type: ignore
    FileValidator = None  # type: ignore
    ALLOWED_DATASET_EXTENSIONS = None  # type: ignore
    DANGEROUS_EXTENSIONS = None  # type: ignore
    log_security_event = None  # type: ignore

# Feature flags (detect available optional features)
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Checkpoint management (Phase 5.3)
from .checkpoints import (
    CheckpointInfo,
    CheckpointManager,
    CheckpointPolicy,
    CheckpointStats,
    RunHistoryManager,
)

# Configuration
#
# BRIDGE-B-010 (Stage C): promote the public preset surface +
# get_recommended_lr / get_recommended_warmup helpers from config.__all__
# into backpropagate's top-level namespace so the documented imports
# resolve directly. UIConfig and WindowsConfig stay internal to Settings —
# operators who need them can `from backpropagate.config import UIConfig`
# explicitly.
#
# The multi-run-loop preset dict is canonically ``MULTI_RUN_PRESETS``.
from .config import (
    LORA_PRESETS,
    MODEL_PRESETS,
    MULTI_RUN_PRESETS,
    PYDANTIC_SETTINGS_AVAILABLE,
    DataConfig,
    LoRAConfig,
    LoRAPreset,
    ModelConfig,
    ModelPreset,
    Settings,
    TrainingConfig,
    TrainingPreset,
    get_cache_dir,
    get_lora_preset,
    get_model_preset,
    get_output_dir,
    get_preset,
    get_recommended_lr,
    get_recommended_warmup,
    get_settings,
    get_training_args,
    lookup_model_preset_by_id,
    reload_settings,
    settings,
)

# Dataset-quality report (v1.5 T1.1 — the moat). Pure-stdlib + numpy-light
# analysis surface; does NOT pull torch, so re-exporting these names keeps
# the eval-path re-export below cheap. Powers `backprop data report <jsonl>`.
from .dataset_report import (
    ContaminationResult,
    DataQualityReport,
    analyze_dataset,
)

# Datasets
from .datasets import (
    # Curriculum learning (Phase 3.3)
    CurriculumStats,
    DatasetFormat,
    # Core classes
    DatasetLoader,
    DatasetStats,
    # Filtering
    FilterStats,
    FormatConverter,
    # Perplexity filtering
    PerplexityFilter,
    PerplexityStats,
    # Streaming
    StreamingDatasetLoader,
    ValidationError,
    ValidationResult,
    analyze_curriculum,
    compute_difficulty_score,
    compute_perplexity,
    convert_to_chatml,
    # Deduplication
    deduplicate_exact,
    deduplicate_minhash,
    # Core functions
    detect_format,
    filter_by_perplexity,
    filter_by_quality,
    get_curriculum_chunks,
    get_dataset_stats,
    order_by_difficulty,
    preview_samples,
    validate_dataset,
)

# Lightweight eval harness (v1.5 T1.1 — the moat). eval.py keeps torch LAZY
# (imported inside evaluate_run, not at module top), so re-exporting these
# public names is cheap — ``import backpropagate`` stays torch-free. Powers
# `backprop eval <run_id>`.
from .eval import (
    EvalDiff,
    EvalGateDecision,
    EvalResult,
    diff_evals,
    eval_gate,
    evaluate_run,
)

# Export
from .export import (
    ExportFormat,
    ExportResult,
    GGUFQuantization,
    create_modelfile,
    export_gguf,
    export_lora,
    export_merged,
    list_ollama_models,
    push_to_hub,
    register_with_ollama,
    remove_ollama_model,
    write_model_card,
)
from .feature_flags import (
    FEATURES,
    INSTALL_HINTS,
    FeatureNotAvailable,
    check_feature,
    ensure_feature,
    get_gpu_info,
    get_install_hint,
    get_system_info,
    list_available_features,
    list_missing_features,
    refresh_features,
    require_feature,
)

# GPU safety
from .gpu_safety import (
    GPUCondition,
    GPUMonitor,
    GPUSafetyConfig,
    GPUStatus,
    check_gpu_safe,
    format_gpu_status,
    get_gpu_status,
    wait_for_safe_gpu,
)
from .model_card import (
    generate_model_card,
)

# Multi-Run (SLAO training)
from .multi_run import (
    MergeMode,
    MultiRunConfig,
    MultiRunResult,
    MultiRunTrainer,
    RunResult,
    SpeedrunConfig,
    SpeedrunResult,
    # Backwards compatibility aliases
    SpeedrunTrainer,
)

# SLAO merging
from .slao import (
    MergeResult,
    SLAOConfig,
    SLAOMerger,
    adaptive_scale,
    # Phase 4: Advanced SLAO
    compute_task_similarity,
    get_layer_scale,
    merge_lora_weights,
    orthogonal_init_A,
    time_aware_scale,
)

# Core trainer
from .trainer import (
    Trainer,
    TrainingCallback,
    TrainingRun,
    load_dataset,
    load_model,
)

# BRIDGE-A-015 (Stage C amend): wrap _pkg_version in a try/except so importing
# the package from a source tree without `pip install -e .` (CI checkouts,
# container builds, dev rebases) does NOT raise PackageNotFoundError at import
# time. Falling back to a PEP 440 +local sentinel keeps __version__ a string
# so callers that do ``backpropagate.__version__`` never AttributeError.
try:
    __version__ = _pkg_version("backpropagate")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


# =============================================================================
# DEPRECATED LAZY-LOADED UI ATTRIBUTES
# =============================================================================
#
# v1.1.0 migrated the Web UI from Gradio (in-process Python launch) to Reflex
# (subprocess via the CLI). The Python-level ``launch()`` entry point is gone;
# operators run ``backprop ui --port N`` instead. For backwards compatibility
# we keep ``__getattr__`` so callers that import the old names get a clear
# deprecation error pointing at the CLI.

# BRIDGE-B-015 (Stage C): each entry now carries (message, removed_in)
# so the warning + ImportError can name the version where the hard removal
# will happen. The deprecation handler emits a DeprecationWarning BEFORE
# raising ImportError so code wrapped in ``try: ... except ImportError:``
# still sees the warning via stderr (the default DeprecationWarning filter
# behavior is "default once per location" which gives operators a single
# heads-up without spamming long-running notebooks). Industry convention
# per PEP 562 examples and numpy / pandas precedent.
#
# CLI-A-005 (v1.4 Wave A2): the BRIDGE-B-015 note said the shim would
# "become a hard AttributeError in v1.4", but at v1.4.0 the shim still
# raises ImportError (the transition was deferred — the back-compat
# contract callers wrap as ``except ImportError`` was kept one more cycle).
# Bumped to the next planned removal (v1.6) so the warning text no longer
# names a version ahead of the current release. The actual swap to
# AttributeError happens in __getattr__ at the marker's cut, not here.
# (History: the marker was v1.4 → v1.5 → v1.6 → v1.7; each cycle the ImportError
# grace period was extended one more minor release rather than hard-breaking
# callers. v1.6 shipped the gradio purge + the gradio-named-alias removal but
# kept these v1.1.0-removed attrs on the ImportError grace one more cycle.)
_REMOVED_IN_VERSION = "v1.7"

_DEPRECATED_UI_ATTRS = {
    "launch": (
        "backpropagate.launch() is removed in v1.1.0. The Web UI migrated "
        "from Gradio to Reflex and is now subprocess-launched via the CLI. "
        "Run `backprop ui --port 7862` instead."
    ),
    "create_backpropagate_theme": (
        "create_backpropagate_theme() is removed in v1.1.0. The Reflex UI "
        "uses Radix theme tokens at backpropagate.ui_theme.RADIX_THEME."
    ),
    "get_theme_info": (
        "get_theme_info() is removed in v1.1.0. Inspect "
        "backpropagate.ui_theme.THEME_TOKENS / LIGHT_TOKENS directly."
    ),
    "get_css": (
        "get_css() is removed in v1.1.0. The Reflex CSS lives at "
        "backpropagate.ui_theme.TOKENS_CSS."
    ),
}


def __getattr__(name: str) -> Any:
    """
    Lazy loading for optional features with helpful error messages.

    BRIDGE-B-015 (Stage C): when a user touches a removed Gradio-era
    attribute we (1) emit a DeprecationWarning naming the future removal
    version (currently ``v1.7`` — see CLI-A-005) so callers wrapping the
    access in ``try: ... except ImportError: pass`` still see the heads-up
    in stderr, then (2) raise ``ImportError`` with the migration hint so the
    existing contract is preserved. At the v1.7 cut the ImportError can be
    swapped for a plain AttributeError to match the rest of
    ``__getattr__``'s contract.
    """
    if name in _DEPRECATED_UI_ATTRS:
        import warnings as _warnings
        base_msg = _DEPRECATED_UI_ATTRS[name]
        full_msg = (
            f"{base_msg} This shim will become a hard AttributeError in "
            f"{_REMOVED_IN_VERSION}."
        )
        _warnings.warn(full_msg, DeprecationWarning, stacklevel=2)
        raise ImportError(full_msg)

    raise AttributeError(f"module 'backpropagate' has no attribute '{name}'")


__all__ = [
    # Version
    "__version__",

    # Security
    "safe_path",
    "check_torch_security",
    "SecurityWarning",
    "PathTraversalError",

    # UI Security (production-hardened)
    "SecurityConfig",
    "DEFAULT_SECURITY_CONFIG",
    "EnhancedRateLimiter",
    "FileValidator",
    "ALLOWED_DATASET_EXTENSIONS",
    "DANGEROUS_EXTENSIONS",
    "log_security_event",

    # Exceptions
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
    "MergeExportError",
    "GGUFExportError",
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

    # Feature flags
    "FEATURES",
    "INSTALL_HINTS",
    "check_feature",
    "require_feature",
    "ensure_feature",
    "refresh_features",
    "get_install_hint",
    "list_available_features",
    "list_missing_features",
    "FeatureNotAvailable",
    "get_gpu_info",
    "get_system_info",

    # Configuration
    "Settings",
    "settings",
    "get_settings",
    "reload_settings",
    "get_output_dir",
    "get_cache_dir",
    "get_training_args",
    "ModelConfig",
    "TrainingConfig",
    "LoRAConfig",
    "DataConfig",
    "PYDANTIC_SETTINGS_AVAILABLE",
    # BRIDGE-B-010 (Stage C): training presets are documented public surface
    # (see `Research-backed presets based on SLAO paper` in config.py) and
    # now resolve via `from backpropagate import MULTI_RUN_PRESETS`.
    "MULTI_RUN_PRESETS",
    "TrainingPreset",
    "get_preset",
    "get_recommended_lr",
    "get_recommended_warmup",
    # v1.3 BACKEND-1 / BACKEND-8/9/10: LoRA + model presets
    "LORA_PRESETS",
    "LoRAPreset",
    "get_lora_preset",
    "MODEL_PRESETS",
    "ModelPreset",
    "get_model_preset",
    "lookup_model_preset_by_id",

    # Core trainer
    "Trainer",
    "TrainingRun",
    "TrainingCallback",
    "load_model",
    "load_dataset",

    # Multi-Run (SLAO)
    "MultiRunTrainer",
    "MultiRunConfig",
    "MultiRunResult",
    "RunResult",
    "MergeMode",
    # Backwards compatibility
    "SpeedrunTrainer",
    "SpeedrunConfig",
    "SpeedrunResult",

    # SLAO merging
    "SLAOMerger",
    "SLAOConfig",
    "MergeResult",
    "time_aware_scale",
    "orthogonal_init_A",
    "merge_lora_weights",
    # Phase 4: Advanced SLAO
    "compute_task_similarity",
    "adaptive_scale",
    "get_layer_scale",

    # Checkpoint management (Phase 5.3)
    "CheckpointManager",
    "CheckpointPolicy",
    "CheckpointInfo",
    "CheckpointStats",
    "RunHistoryManager",

    # GPU safety
    "GPUMonitor",
    "GPUStatus",
    "GPUSafetyConfig",
    "GPUCondition",
    "check_gpu_safe",
    "get_gpu_status",
    "wait_for_safe_gpu",
    "format_gpu_status",

    # Export
    "GGUFQuantization",
    "ExportFormat",
    "ExportResult",
    "export_lora",
    "export_merged",
    "export_gguf",
    "create_modelfile",
    "register_with_ollama",
    "list_ollama_models",
    "remove_ollama_model",
    "push_to_hub",
    "write_model_card",
    "generate_model_card",

    # Datasets - Core
    "DatasetLoader",
    "DatasetFormat",
    "ValidationResult",
    "ValidationError",
    "DatasetStats",
    "FormatConverter",
    "detect_format",
    "validate_dataset",
    "convert_to_chatml",
    "preview_samples",
    "get_dataset_stats",
    # Datasets - Streaming
    "StreamingDatasetLoader",
    # Datasets - Filtering
    "FilterStats",
    "filter_by_quality",
    # Datasets - Deduplication
    "deduplicate_exact",
    "deduplicate_minhash",
    # Datasets - Perplexity filtering
    "PerplexityFilter",
    "PerplexityStats",
    "compute_perplexity",
    "filter_by_perplexity",
    # Datasets - Curriculum learning
    "CurriculumStats",
    "compute_difficulty_score",
    "order_by_difficulty",
    "get_curriculum_chunks",
    "analyze_curriculum",

    # Dataset-quality report (v1.5 T1.1)
    "analyze_dataset",
    "DataQualityReport",
    "ContaminationResult",

    # Eval harness (v1.5 T1.1)
    "evaluate_run",
    "diff_evals",
    "eval_gate",
    "EvalResult",
    "EvalGateDecision",
    "EvalDiff",

]
