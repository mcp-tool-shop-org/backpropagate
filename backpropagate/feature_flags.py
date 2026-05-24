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
    pip install backpropagate[ui]          # + Reflex web UI (migrated from Gradio in v1.1.0)
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
    "ensure_feature",
    "get_install_hint",
    "list_available_features",
    "list_missing_features",
    "refresh_features",
    "FeatureNotAvailable",
    "INSTALL_HINTS",
    "FEATURE_DESCRIPTIONS",
]

# Feature detection results
FEATURES: dict[str, bool] = {
    "unsloth": False,
    "ui": False,
    "validation": False,
    "export": False,
    "monitoring": False,
    "flash_attention": False,
    "triton": False,
    # F-005: per-tracker availability so the trainer's report_to resolver can
    # decide whether to wire wandb / tensorboard / mlflow into SFTConfig.
    # ``monitoring`` is the [monitoring] extra (covers wandb + psutil); the
    # entries below give fine-grained per-tracker visibility.
    "wandb": False,
    "tensorboard": False,
    "mlflow": False,
    # BRIDGE-B-009 (Stage C): expose psutil as its own primitive flag so a
    # wandb-only OR psutil-only install can still reach get_system_info()'s
    # memory section. ``monitoring`` is preserved as the AND of psutil+wandb
    # for back-compat with callers that gate on the umbrella flag.
    "psutil": False,
}

# Installation hints for each feature.
#
# BRIDGE-B-003 (Stage C humanization): each hint names the install command, what
# the operator gains by adding it, and the common failure mode they will hit if
# they DON'T install it. The goal is "next action plus consequence" so the
# operator can triage a missing-feature stderr line without leaving the terminal
# — Norman 1988 affordance framing applied to optional-extra install hints.
INSTALL_HINTS: dict[str, str] = {
    "unsloth": (
        "pip install 'backpropagate[unsloth]' — enables 2x faster QLoRA "
        "training with ~50% less VRAM (Qwen / Llama 7B fits 16 GB). "
        "Without it: training falls back to plain transformers + peft (~2x "
        "slower; may OOM on a 16 GB card for 7B + LoRA r=32)."
    ),
    "ui": (
        "pip install 'backpropagate[ui]' — installs Reflex (Radix UI) web "
        "interface for training / export / run history. "
        "Without it: `backprop ui` exits with 'UI dependencies not installed'."
    ),
    "validation": (
        "pip install 'backpropagate[validation]' — enables Pydantic config "
        "validation so BACKPROPAGATE_* env vars are type-checked at startup. "
        "Without it: malformed env vars (e.g. non-numeric learning_rate) "
        "surface as runtime ValueError in the trainer, not a clear startup error."
    ),
    "export": (
        "pip install 'backpropagate[export]' — installs llama-cpp-python for "
        "in-process GGUF export. "
        "Without it: GGUF export falls back to a subprocess call to "
        "~/llama.cpp/convert_hf_to_gguf.py (or shutil.which) and errors if "
        "neither is present."
    ),
    "monitoring": (
        "pip install 'backpropagate[monitoring]' — installs wandb + psutil "
        "so the trainer can wire experiment tracking and `backprop info` "
        "shows host memory. "
        "Without it: training runs unobserved (no wandb dashboard) and "
        "`backprop info` omits the Memory section."
    ),
    "flash_attention": (
        "pip install flash-attn --no-build-isolation — enables Flash "
        "Attention 2 (typically 1.3-1.7x training speedup on Ampere+). "
        "Without it: training uses standard SDPA attention (correct but slower)."
    ),
    "triton": (
        "pip install triton — enables Unsloth's hand-tuned Triton kernels. "
        "Without it: Unsloth falls back to its pure-PyTorch path (correct "
        "but loses the kernel-fusion speedup)."
    ),
    # F-005 per-tracker hints.
    "wandb": (
        "pip install 'backpropagate[monitoring]' — bundles wandb + psutil. "
        "Without it: `report_to=\"wandb\"` is silently dropped by the trainer's "
        "auto-resolver (no error, no dashboard)."
    ),
    "tensorboard": (
        "pip install tensorboard — enables local TensorBoard event-file "
        "logging. "
        "Without it: `report_to=\"tensorboard\"` is silently dropped."
    ),
    "mlflow": (
        "pip install mlflow — enables MLflow tracking + model-registry "
        "integration. "
        "Without it: `report_to=\"mlflow\"` is silently dropped."
    ),
}

# Feature descriptions
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "unsloth": "Unsloth for 2x faster training with 50% less VRAM",
    "ui": "Reflex (Radix UI) web interface for training management",
    "validation": "Pydantic configuration validation",
    "export": "GGUF export for Ollama/llama.cpp deployment",
    "monitoring": "WandB logging and system monitoring (psutil)",
    "flash_attention": "Flash Attention 2 for faster attention",
    "triton": "Triton kernels for optimized operations",
    # F-005 per-tracker descriptions.
    "wandb": "Weights & Biases experiment tracking",
    "tensorboard": "TensorBoard local experiment logs",
    "mlflow": "MLflow experiment tracking and model registry",
    "psutil": "Host-memory introspection (used by `backprop info`)",
}


# BRIDGE-B-009 (Stage C): psutil install hint as a primitive (separate from
# the wandb+psutil monitoring bundle). Operators who only want memory
# introspection can pip install just psutil.
INSTALL_HINTS["psutil"] = (
    "pip install psutil — enables host-memory introspection in `backprop info`. "
    "Without it: `backprop info` omits the Memory section (training continues "
    "fine; only the introspection surface degrades)."
)


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

    # UI feature (Reflex; migrated from Gradio in v1.1.0). The probe stays
    # importlib.util.find_spec-only so detection costs ~0ms even though Reflex
    # itself has a non-trivial import-time cost (FastAPI + WebSocket stack).
    if _has_module("reflex"):
        FEATURES["ui"] = True
        logger.debug("Feature 'ui' available: reflex installed")
    else:
        logger.debug("Feature 'ui' unavailable: reflex not installed")

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

    # BRIDGE-B-009 (Stage C): detect psutil + wandb independently so each
    # primitive flag can drive its own code path (memory introspection vs.
    # experiment-tracking auto-wire). ``monitoring`` remains the AND of the
    # two so existing callers gating on the umbrella flag keep behaving.
    has_psutil = _has_module("psutil")
    has_wandb = _has_module("wandb")
    FEATURES["psutil"] = has_psutil
    if has_psutil:
        logger.debug("Feature 'psutil' available")
    else:
        logger.debug("Feature 'psutil' unavailable: psutil not installed")

    if has_psutil and has_wandb:
        FEATURES["monitoring"] = True
        logger.debug("Feature 'monitoring' available: wandb and psutil installed")
    else:
        logger.debug(
            "Feature 'monitoring' unavailable (requires both psutil and wandb)"
        )

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

    # Add memory info if psutil available.
    #
    # BRIDGE-B-009 (Stage C): gate on the psutil primitive flag rather than
    # the monitoring umbrella. Pre-fix this required BOTH wandb AND psutil
    # to surface memory info — an operator with `pip install psutil` alone
    # would see Memory missing because wandb wasn't installed, even though
    # psutil itself was reachable. Now the memory path fires whenever psutil
    # is importable.
    if FEATURES.get("psutil") or FEATURES.get("monitoring"):
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
