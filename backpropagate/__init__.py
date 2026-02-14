"""
Backpropagate - Headless LLM Fine-Tuning
=========================================

A clean, production-ready interface for LLM fine-tuning that makes
complex training accessible through simple parameters and smart defaults.

v0.1.0: Modular architecture with optional features

Installation:
    pip install backpropagate              # Core only (minimal)
    pip install backpropagate[unsloth]     # + Unsloth 2x faster training
    pip install backpropagate[ui]          # + Gradio web UI
    pip install backpropagate[standard]    # unsloth + ui (recommended)
    pip install backpropagate[full]        # Everything

Features (Core):
- QLoRA fine-tuning with smart defaults
- Windows-safe multiprocessing
- Auto VRAM detection for batch size
- Multiple export formats (LoRA, merged, GGUF)

Features (Optional):
- [unsloth] 2x faster training with 50% less VRAM
- [ui] Gradio web interface for training management
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

    # UI (requires [ui] extra)
    from backpropagate import launch
    launch()
"""

# Exceptions (custom error hierarchy)
from .exceptions import (
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
    MergeExportError,
    GGUFExportError,
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
    # Batch operations
    BatchOperationError,
)

# Security utilities
from .security import (
    safe_path,
    check_torch_security,
    SecurityWarning,
    PathTraversalError,
)

# UI Security utilities (production-hardened) - requires gradio
try:
    from .ui_security import (
        SecurityConfig,
        DEFAULT_SECURITY_CONFIG,
        EnhancedRateLimiter,
        FileValidator,
        ALLOWED_DATASET_EXTENSIONS,
        DANGEROUS_EXTENSIONS,
        safe_gradio_handler,
        log_security_event,
    )
except ImportError:
    # Gradio not installed - UI security features unavailable
    SecurityConfig = None  # type: ignore
    DEFAULT_SECURITY_CONFIG = None  # type: ignore
    EnhancedRateLimiter = None  # type: ignore
    FileValidator = None  # type: ignore
    ALLOWED_DATASET_EXTENSIONS = None  # type: ignore
    DANGEROUS_EXTENSIONS = None  # type: ignore
    safe_gradio_handler = None  # type: ignore
    log_security_event = None  # type: ignore

# Feature flags (detect available optional features)
from .feature_flags import (
    FEATURES,
    check_feature,
    require_feature,
    get_install_hint,
    list_available_features,
    list_missing_features,
    FeatureNotAvailable,
    get_gpu_info,
    get_system_info,
)

# Configuration
from .config import (
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

# Core trainer
from .trainer import (
    Trainer,
    TrainingRun,
    TrainingCallback,
    load_model,
    load_dataset,
)

# Multi-Run (SLAO training)
from .multi_run import (
    MultiRunTrainer,
    MultiRunConfig,
    MultiRunResult,
    RunResult,
    MergeMode,
    # Backwards compatibility aliases
    SpeedrunTrainer,
    SpeedrunConfig,
    SpeedrunResult,
)

# SLAO merging
from .slao import (
    SLAOMerger,
    SLAOConfig,
    MergeResult,
    time_aware_scale,
    orthogonal_init_A,
    merge_lora_weights,
    # Phase 4: Advanced SLAO
    compute_task_similarity,
    adaptive_scale,
    get_layer_scale,
)

# Checkpoint management (Phase 5.3)
from .checkpoints import (
    CheckpointManager,
    CheckpointPolicy,
    CheckpointInfo,
    CheckpointStats,
)

# GPU safety
from .gpu_safety import (
    GPUMonitor,
    GPUStatus,
    GPUSafetyConfig,
    GPUCondition,
    check_gpu_safe,
    get_gpu_status,
    wait_for_safe_gpu,
    format_gpu_status,
)

# Export
from .export import (
    GGUFQuantization,
    ExportFormat,
    ExportResult,
    export_lora,
    export_merged,
    export_gguf,
    create_modelfile,
    register_with_ollama,
    list_ollama_models,
)

# Datasets
from .datasets import (
    # Core classes
    DatasetLoader,
    DatasetFormat,
    ValidationResult,
    ValidationError,
    DatasetStats,
    FormatConverter,
    # Core functions
    detect_format,
    validate_dataset,
    convert_to_chatml,
    preview_samples,
    get_dataset_stats,
    # Streaming
    StreamingDatasetLoader,
    # Filtering
    FilterStats,
    filter_by_quality,
    # Deduplication
    deduplicate_exact,
    deduplicate_minhash,
    # Perplexity filtering
    PerplexityFilter,
    PerplexityStats,
    compute_perplexity,
    filter_by_perplexity,
    # Curriculum learning (Phase 3.3)
    CurriculumStats,
    compute_difficulty_score,
    order_by_difficulty,
    get_curriculum_chunks,
    analyze_curriculum,
)

from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("backpropagate")


# =============================================================================
# LAZY LOADING FOR OPTIONAL FEATURES
# =============================================================================

_LAZY_IMPORTS = {
    # UI features
    "launch": ("ui", None),  # Special handling
    "create_backpropagate_theme": ("ui", ".theme"),
    "get_theme_info": ("ui", ".theme"),
    "get_css": ("ui", ".theme"),
}


def __getattr__(name: str):
    """
    Lazy loading for optional features with helpful error messages.

    When a user tries to import an optional feature that isn't installed,
    this provides a clear error message with installation instructions.
    """
    if name in _LAZY_IMPORTS:
        feature, module = _LAZY_IMPORTS[name]
        if not FEATURES.get(feature, False):
            hint = get_install_hint(feature)
            raise ImportError(
                f"'{name}' requires the [{feature}] feature. "
                f"Install with: {hint}"
            )

        # Special handling for launch
        if name == "launch":
            def _launch(port: int = 7862, share: bool = False):
                from .ui import create_ui
                app = create_ui()
                app.launch(
                    server_name="0.0.0.0",
                    server_port=port,
                    share=share,
                    inbrowser=True,
                )
            return _launch

        # Import the actual object
        if module:
            import importlib
            mod = importlib.import_module(module, __package__)
            return getattr(mod, name)

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
    "safe_gradio_handler",
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
    "check_feature",
    "require_feature",
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

    # Lazy-loaded (UI)
    "launch",
    "create_backpropagate_theme",
    "get_theme_info",
    "get_css",
]
