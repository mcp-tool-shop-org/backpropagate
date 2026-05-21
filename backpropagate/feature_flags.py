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
import importlib.util
import logging
import os
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

# Use standard logging to avoid circular import with logging_config
logger = logging.getLogger(__name__)

__all__ = [
    "FEATURES",
    "check_feature",
    "require_feature",
    "get_install_hint",
    "list_available_features",
    "list_missing_features",
    "refresh_features",
    "FeatureNotAvailable",
]

# Feature detection results
FEATURES: dict[str, bool] = {
    "unsloth": False,
    "ui": False,
    "validation": False,
    "export": False,
    "monitoring": False,
    "observability": False,
    "flash_attention": False,
    "triton": False,
    # F-005: per-tracker availability so the trainer's report_to resolver can
    # decide whether to wire wandb / tensorboard / mlflow into SFTConfig.
    # ``monitoring`` is the [monitoring] extra (covers wandb + psutil); the
    # entries below give fine-grained per-tracker visibility.
    "wandb": False,
    "tensorboard": False,
    "mlflow": False,
}

# Installation hints for each feature
INSTALL_HINTS: dict[str, str] = {
    "unsloth": "pip install backpropagate[unsloth]",
    "ui": "pip install backpropagate[ui]",
    "validation": "pip install backpropagate[validation]",
    "export": "pip install backpropagate[export]",
    "monitoring": "pip install backpropagate[monitoring]",
    "observability": "pip install backpropagate[observability]",
    "flash_attention": "pip install flash-attn --no-build-isolation",
    "triton": "pip install triton",
    # F-005 per-tracker hints.
    "wandb": "pip install backpropagate[monitoring]  # bundles wandb + psutil",
    "tensorboard": "pip install tensorboard",
    "mlflow": "pip install mlflow",
}

# Feature descriptions
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "unsloth": "Unsloth for 2x faster training with 50% less VRAM",
    "ui": "Gradio web interface for training management",
    "validation": "Pydantic configuration validation",
    "export": "GGUF export for Ollama/llama.cpp deployment",
    "monitoring": "WandB logging and system monitoring (psutil)",
    "observability": "OpenTelemetry distributed tracing",
    "flash_attention": "Flash Attention 2 for faster attention",
    "triton": "Triton kernels for optimized operations",
    # F-005 per-tracker descriptions.
    "wandb": "Weights & Biases experiment tracking",
    "tensorboard": "TensorBoard local experiment logs",
    "mlflow": "MLflow experiment tracking and model registry",
}


# =============================================================================
# FEATURE DETECTION
# =============================================================================

def _has_module(name: str) -> bool:
    """Check whether a module is importable WITHOUT executing it.

    Uses :func:`importlib.util.find_spec`, which inspects the import system's
    finders without running module-level code. This avoids the side effects
    of eagerly importing heavy / opinionated packages just to flip a boolean:

    * ``unsloth`` registers ``torch.compile`` hooks and emits warnings
    * ``wandb`` may probe network on import (rare, but observed in some envs)
    * ``opentelemetry`` registers a global tracer provider
    * ``flash_attn`` triggers a multi-second CUDA capability probe

    ``find_spec`` returns a :class:`ModuleSpec` or ``None`` and is roughly
    100x faster than the corresponding ``try: import X`` for a missing
    package — the difference compounds across 8 optional features and is
    visible on ``backprop --help`` startup latency.

    Returns ``False`` if any error occurs during the lookup (e.g. a
    half-installed package whose ``__init__.py`` cannot even be located).
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        # Some packages register pathological finders that throw on lookup
        # rather than returning None — treat any failure as "not available".
        return False


def _detect_features() -> None:
    """Detect which optional features are available.

    Honors the ``BACKPROPAGATE_DEFER_FEATURE_DETECTION`` env var as an opt-out
    for power users who want the absolute-fastest CLI startup; when set, all
    features remain ``False`` until :func:`refresh_features` is called.
    """
    global FEATURES

    if os.environ.get("BACKPROPAGATE_DEFER_FEATURE_DETECTION"):
        logger.debug(
            "Feature detection deferred via BACKPROPAGATE_DEFER_FEATURE_DETECTION"
        )
        return

    # Unsloth feature — find_spec only, never actually imports unsloth
    # (which would register torch.compile hooks and emit warnings).
    # Note: Unsloth uses torch.compile which isn't supported on Python 3.14+,
    # but the runtime check there happens only when unsloth is actually used.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Unsloth should be imported before.*")
        if _has_module("unsloth"):
            FEATURES["unsloth"] = True
            logger.debug("Feature 'unsloth' available (spec found, not imported)")
        else:
            logger.debug("Feature 'unsloth' unavailable: unsloth not installed")

    # UI feature (Gradio)
    if _has_module("gradio"):
        FEATURES["ui"] = True
        logger.debug("Feature 'ui' available: gradio installed")
    else:
        logger.debug("Feature 'ui' unavailable: gradio not installed")

    # Validation feature (Pydantic) — needs BOTH pydantic and pydantic_settings
    if _has_module("pydantic") and _has_module("pydantic_settings"):
        FEATURES["validation"] = True
        logger.debug("Feature 'validation' available: pydantic installed")
    else:
        logger.debug("Feature 'validation' unavailable: pydantic not installed")

    # Export feature (llama-cpp-python)
    if _has_module("llama_cpp"):
        FEATURES["export"] = True
        logger.debug("Feature 'export' available: llama-cpp-python installed")
    else:
        logger.debug("Feature 'export' unavailable: llama-cpp-python not installed")

    # Monitoring feature (WandB + psutil) — needs both
    if _has_module("psutil") and _has_module("wandb"):
        FEATURES["monitoring"] = True
        logger.debug("Feature 'monitoring' available: wandb and psutil installed")
    else:
        logger.debug("Feature 'monitoring' unavailable")

    # Observability feature (OpenTelemetry)
    if _has_module("opentelemetry"):
        FEATURES["observability"] = True
        logger.debug("Feature 'observability' available: opentelemetry installed")
    else:
        logger.debug("Feature 'observability' unavailable")

    # Flash Attention feature
    if _has_module("flash_attn"):
        FEATURES["flash_attention"] = True
        logger.debug("Feature 'flash_attention' available")
    else:
        logger.debug("Feature 'flash_attention' unavailable")

    # Triton feature
    if _has_module("triton"):
        FEATURES["triton"] = True
        logger.debug("Feature 'triton' available")
    else:
        logger.debug("Feature 'triton' unavailable")

    # F-005: per-tracker detection — each one toggles a different report_to
    # branch in the trainer's auto-resolver. Detect via find_spec only so we
    # don't trigger wandb's network probe or mlflow's import side effects.
    if _has_module("wandb"):
        FEATURES["wandb"] = True
        logger.debug("Feature 'wandb' available")
    else:
        logger.debug("Feature 'wandb' unavailable: wandb not installed")

    # TensorBoard ships as either `tensorboard` (full) or via `tensorboardX`.
    if _has_module("tensorboard") or _has_module("tensorboardX"):
        FEATURES["tensorboard"] = True
        logger.debug("Feature 'tensorboard' available")
    else:
        logger.debug("Feature 'tensorboard' unavailable")

    if _has_module("mlflow"):
        FEATURES["mlflow"] = True
        logger.debug("Feature 'mlflow' available")
    else:
        logger.debug("Feature 'mlflow' unavailable")

    # Log summary of detected features
    available = [name for name, enabled in FEATURES.items() if enabled]
    missing = [name for name, enabled in FEATURES.items() if not enabled]
    logger.debug(f"Feature detection complete: {len(available)} available, {len(missing)} missing")


def check_unsloth_runtime() -> tuple[bool, str | None]:
    """
    Lazy runtime probe for unsloth's actual usability.

    ``_detect_features`` only checks whether the ``unsloth`` package is
    importable; it does NOT import the module (to avoid torch.compile hook
    registration). Callers that are about to actually use unsloth should
    call this helper first to surface Python-version incompatibilities
    (unsloth currently requires Python <3.14 because it depends on
    ``torch.compile``) or other init-time failures BEFORE the import fans
    out across the training stack.

    Returns:
        ``(ok, error_message)`` — ``ok=True`` means import succeeded;
        otherwise ``error_message`` describes why.
    """
    if not FEATURES.get("unsloth"):
        return False, "unsloth package not installed"

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Unsloth should be imported before.*"
            )
            import unsloth  # noqa: F401
        return True, None
    except RuntimeError as e:
        # torch.compile not supported on Python 3.14+
        return False, f"Python 3.14+ incompatibility: {e}"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"unsloth failed to load: {e}"


# Run detection on module load
_detect_features()


def refresh_features() -> dict[str, bool]:
    """
    Re-run feature detection and update the FEATURES dict.

    Useful after installing or uninstalling optional dependencies at runtime
    (e.g., in a notebook or long-running process) without restarting the interpreter.

    Returns:
        Updated FEATURES dict.
    """
    # Reset all flags before re-detection
    for key in FEATURES:
        FEATURES[key] = False
    _detect_features()
    return dict(FEATURES)


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


def list_available_features() -> dict[str, str]:
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


def list_missing_features() -> dict[str, str]:
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
                desc = FEATURE_DESCRIPTIONS.get(feature, "")
                desc_suffix = f" ({desc})" if desc else ""
                raise ImportError(
                    f"Feature '{feature}'{desc_suffix} is required but not installed. "
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


def get_gpu_info() -> dict[str, Any]:
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


def get_system_info() -> dict[str, Any]:
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
        "features": dict(FEATURES.items()),
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
