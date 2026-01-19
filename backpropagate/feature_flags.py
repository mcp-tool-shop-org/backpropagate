"""
Backpropagate - Feature Flags
==============================

Detects which optional features are available based on installed dependencies.
Provides decorators and utilities for graceful degradation.

Usage:
    from backpropagate.feature_flags import FEATURES, require_feature

    # Check if feature is available
    if FEATURES["unsloth"]:
        from unsloth import FastLanguageModel

    # Decorator to require a feature
    @require_feature("ui")
    def launch_ui():
        ...

Installation commands for each feature:
    pip install backpropagate              # Core only
    pip install backpropagate[unsloth]     # + Unsloth 2x faster training
    pip install backpropagate[ui]          # + Gradio web UI
    pip install backpropagate[validation]  # + Pydantic config validation
    pip install backpropagate[export]      # + GGUF export
    pip install backpropagate[monitoring]  # + WandB & system monitoring
    pip install backpropagate[standard]    # unsloth + ui (recommended)
    pip install backpropagate[full]        # Everything
"""

import functools
import logging
import warnings
from typing import Dict, Callable, TypeVar, Any

# Use standard logging to avoid circular import with logging_config
logger = logging.getLogger(__name__)

__all__ = [
    "FEATURES",
    "check_feature",
    "require_feature",
    "get_install_hint",
    "list_available_features",
    "list_missing_features",
    "FeatureNotAvailable",
]

# Feature detection results
FEATURES: Dict[str, bool] = {
    "unsloth": False,
    "ui": False,
    "validation": False,
    "export": False,
    "monitoring": False,
    "observability": False,
    "flash_attention": False,
    "triton": False,
}

# Installation hints for each feature
INSTALL_HINTS: Dict[str, str] = {
    "unsloth": "pip install backpropagate[unsloth]",
    "ui": "pip install backpropagate[ui]",
    "validation": "pip install backpropagate[validation]",
    "export": "pip install backpropagate[export]",
    "monitoring": "pip install backpropagate[monitoring]",
    "observability": "pip install backpropagate[observability]",
    "flash_attention": "pip install flash-attn --no-build-isolation",
    "triton": "pip install triton",
}

# Feature descriptions
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "unsloth": "Unsloth for 2x faster training with 50% less VRAM",
    "ui": "Gradio web interface for training management",
    "validation": "Pydantic configuration validation",
    "export": "GGUF export for Ollama/llama.cpp deployment",
    "monitoring": "WandB logging and system monitoring (psutil)",
    "observability": "OpenTelemetry distributed tracing",
    "flash_attention": "Flash Attention 2 for faster attention",
    "triton": "Triton kernels for optimized operations",
}


# =============================================================================
# FEATURE DETECTION
# =============================================================================

def _detect_features() -> None:
    """Detect which optional features are available."""
    global FEATURES

    # Unsloth feature
    # Note: Unsloth uses torch.compile which isn't supported on Python 3.14+
    # Suppress the import order warning - it's expected when checking availability
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Unsloth should be imported before.*")
            import unsloth  # noqa: F401
        FEATURES["unsloth"] = True
        logger.debug("Feature 'unsloth' available")
    except ImportError:
        logger.debug("Feature 'unsloth' unavailable: unsloth not installed")
    except RuntimeError as e:
        # torch.compile not supported on Python 3.14+
        logger.warning(f"Feature 'unsloth' unavailable (Python 3.14+ incompatibility): {e}")
    except Exception as e:
        logger.warning(f"Feature 'unsloth' failed to load: {e}")

    # UI feature (Gradio)
    try:
        import gradio  # noqa: F401
        FEATURES["ui"] = True
        logger.debug("Feature 'ui' available: gradio installed")
    except ImportError:
        logger.debug("Feature 'ui' unavailable: gradio not installed")

    # Validation feature (Pydantic)
    try:
        import pydantic  # noqa: F401
        import pydantic_settings  # noqa: F401
        FEATURES["validation"] = True
        logger.debug("Feature 'validation' available: pydantic installed")
    except ImportError:
        logger.debug("Feature 'validation' unavailable: pydantic not installed")

    # Export feature (llama-cpp-python)
    try:
        import llama_cpp  # noqa: F401
        FEATURES["export"] = True
        logger.debug("Feature 'export' available: llama-cpp-python installed")
    except ImportError:
        logger.debug("Feature 'export' unavailable: llama-cpp-python not installed")

    # Monitoring feature (WandB + psutil)
    try:
        import wandb  # noqa: F401
        import psutil  # noqa: F401
        FEATURES["monitoring"] = True
        logger.debug("Feature 'monitoring' available: wandb and psutil installed")
    except ImportError:
        logger.debug("Feature 'monitoring' unavailable")

    # Observability feature (OpenTelemetry)
    try:
        import opentelemetry  # noqa: F401
        FEATURES["observability"] = True
        logger.debug("Feature 'observability' available: opentelemetry installed")
    except ImportError:
        logger.debug("Feature 'observability' unavailable")

    # Flash Attention feature
    try:
        import flash_attn  # noqa: F401
        FEATURES["flash_attention"] = True
        logger.debug("Feature 'flash_attention' available")
    except ImportError:
        logger.debug("Feature 'flash_attention' unavailable")

    # Triton feature
    try:
        import triton  # noqa: F401
        FEATURES["triton"] = True
        logger.debug("Feature 'triton' available")
    except ImportError:
        logger.debug("Feature 'triton' unavailable")

    # Log summary of detected features
    available = [name for name, enabled in FEATURES.items() if enabled]
    missing = [name for name, enabled in FEATURES.items() if not enabled]
    logger.debug(f"Feature detection complete: {len(available)} available, {len(missing)} missing")


# Run detection on module load
_detect_features()


# =============================================================================
# PUBLIC API
# =============================================================================

def check_feature(feature: str) -> bool:
    """
    Check if a feature is available.

    Args:
        feature: Feature name (unsloth, ui, validation, export, monitoring, etc.)

    Returns:
        True if the feature is installed
    """
    return FEATURES.get(feature, False)


def get_install_hint(feature: str) -> str:
    """
    Get installation command for a feature.

    Args:
        feature: Feature name

    Returns:
        pip install command string
    """
    return INSTALL_HINTS.get(feature, f"pip install backpropagate[{feature}]")


def list_available_features() -> Dict[str, str]:
    """
    List all installed features with descriptions.

    Returns:
        Dict of feature name -> description for installed features
    """
    return {
        name: FEATURE_DESCRIPTIONS.get(name, "")
        for name, available in FEATURES.items()
        if available
    }


def list_missing_features() -> Dict[str, str]:
    """
    List all missing features with install hints.

    Returns:
        Dict of feature name -> install hint for missing features
    """
    return {
        name: INSTALL_HINTS.get(name, "")
        for name, available in FEATURES.items()
        if not available
    }


# Type variable for generic function preservation
F = TypeVar("F", bound=Callable[..., Any])


def require_feature(feature: str) -> Callable[[F], F]:
    """
    Decorator to require a feature for a function.

    Raises ImportError with install hint if feature is not available.

    Args:
        feature: Feature name to require

    Usage:
        @require_feature("unsloth")
        def fast_train(model, dataset):
            # Uses Unsloth internally
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not FEATURES.get(feature, False):
                hint = INSTALL_HINTS.get(feature, f"pip install backpropagate[{feature}]")
                raise ImportError(
                    f"Feature '{feature}' is required but not installed. "
                    f"Install with: {hint}"
                )
            return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


class FeatureNotAvailable(ImportError):
    """Raised when trying to use a feature that isn't installed."""

    def __init__(self, feature: str, message: str = ""):
        self.feature = feature
        self.install_hint = get_install_hint(feature)
        if not message:
            message = (
                f"Feature '{feature}' is not available. "
                f"Install with: {self.install_hint}"
            )
        super().__init__(message)


def ensure_feature(feature: str) -> None:
    """
    Ensure a feature is available, raising FeatureNotAvailable if not.

    Args:
        feature: Feature name to check

    Raises:
        FeatureNotAvailable: If the feature is not installed
    """
    if not FEATURES.get(feature, False):
        raise FeatureNotAvailable(feature)


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information if available.

    Returns:
        Dict with GPU info or empty dict if not available
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(0),
                "memory_reserved": torch.cuda.memory_reserved(0),
                "compute_capability": torch.cuda.get_device_capability(0),
            }
    except Exception:  # nosec B110 - intentional silent fallback for GPU detection
        pass
    return {"available": False}


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.

    Returns:
        Dict with system info
    """
    import platform
    import sys

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "features": {k: v for k, v in FEATURES.items()},
        "gpu": get_gpu_info(),
    }

    # Add memory info if psutil available
    if FEATURES["monitoring"]:
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["memory"] = {
                "total": mem.total,
                "available": mem.available,
                "percent": mem.percent,
            }
        except Exception:  # nosec B110 - intentional silent fallback for optional monitoring
            pass

    return info
