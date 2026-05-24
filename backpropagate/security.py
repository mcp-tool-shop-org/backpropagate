"""
Backpropagate - Security Utilities
===================================

Security utilities for safe path handling, version validation, and audit logging.

Usage:
    from backpropagate.security import safe_path, check_torch_security

    # Validate user-provided paths
    path = safe_path("/user/provided/path", must_exist=True)

    # Check PyTorch version for security features
    check_torch_security()
"""

import logging
import warnings
from pathlib import Path
from typing import Any

__all__ = [
    "safe_path",
    "check_torch_security",
    "SecurityWarning",
    "PathTraversalError",
]

logger = logging.getLogger(__name__)

# Minimum PyTorch version for proper weights_only=True enforcement
MINIMUM_TORCH_VERSION = "2.0.0"


class SecurityWarning(UserWarning):
    """Warning for security-related concerns."""
    pass


class PathTraversalError(ValueError):
    """Error raised when path traversal is detected.

    Carries a stable Ship Gate B1 ``code`` (``INPUT_PATH_TRAVERSAL``) for
    structured logging / machine-readable handling parallel to the
    ``BackpropagateError`` hierarchy, even though this class inherits from
    :class:`ValueError` (not :class:`BackpropagateError`) for compatibility
    with callers that already catch ``ValueError``.
    """

    code: str = "INPUT_PATH_TRAVERSAL"

    def __init__(self, path: str, allowed_base: str | None = None):
        self.path = path
        self.allowed_base = allowed_base

        if allowed_base:
            message = f"Path '{path}' escapes allowed directory '{allowed_base}'"
        else:
            message = f"Path traversal detected in: {path}"

        super().__init__(message)


def safe_path(
    user_path: str | Path,
    must_exist: bool = False,
    allowed_base: str | Path | None = None,
    allow_relative: bool = True,
) -> Path:
    """
    Safely resolve and validate a user-provided path.

    Prevents path traversal attacks by ensuring the resolved path stays within
    allowed boundaries.

    Default behavior (no ``allowed_base``):
        - Relative paths containing ``..`` that resolve **inside the current
          working directory** are accepted with a WARNING log line. This is the
          backward-compatible path for callers that just want normalization
          (e.g. ``safe_path("subdir/../other")``).
        - Any other ``..`` pattern — an absolute path containing ``..``, or a
          relative path that resolves OUTSIDE the current working directory —
          is REJECTED with :class:`PathTraversalError`. Callers that previously
          relied on the warn-only behavior for these cases must now pass an
          explicit ``allowed_base`` to declare the legitimate scope.

    With ``allowed_base``:
        The resolved path must be inside ``allowed_base``. Anything that escapes
        raises :class:`PathTraversalError` (this behavior is unchanged).

    Args:
        user_path: The user-provided path to validate.
        must_exist: If True, raise :class:`FileNotFoundError` if the resolved
            path does not exist.
        allowed_base: If provided, ensure the resolved path is within this
            directory. Recommended at every sink that touches the filesystem.
        allow_relative: If False, reject paths that are not absolute up front.

    Returns:
        Resolved, validated :class:`Path` object.

    Raises:
        PathTraversalError: If the path escapes ``allowed_base``, or — when
            no ``allowed_base`` is supplied — if it contains ``..`` AND is
            absolute, OR resolves outside the current working directory.
        FileNotFoundError: If ``must_exist=True`` and the path does not exist.
        ValueError: If ``allow_relative=False`` and the path is relative.

    Examples:
        >>> safe_path("/models/my_model", must_exist=True)
        PosixPath('/models/my_model')

        >>> safe_path("../../etc/passwd", allowed_base="/models")
        PathTraversalError: Path '../../etc/passwd' escapes allowed directory '/models'

        >>> safe_path("/etc/../etc/passwd")
        PathTraversalError: Path traversal detected in: /etc/../etc/passwd
    """
    path = Path(user_path)

    # Check for relative paths if not allowed
    if not allow_relative and not path.is_absolute():
        raise ValueError(f"Relative paths not allowed: {user_path}")

    # Resolve to absolute path
    resolved = path.resolve()

    # Check path traversal against allowed base
    if allowed_base is not None:
        base_resolved = Path(allowed_base).resolve()

        try:
            # This will raise ValueError if resolved is not relative to base_resolved
            resolved.relative_to(base_resolved)
        except ValueError:
            raise PathTraversalError(str(user_path), str(allowed_base))

    # When no allowed_base is supplied, defend at the sink for obvious
    # traversal patterns. Previously this only warned, which made `safe_path`
    # a paper tiger at every CLI call site that did not pass `allowed_base`
    # (see F-002). Now: ".." in an absolute path, or in a relative path that
    # resolves outside the current working directory, is rejected outright.
    # Relative ".." that normalizes back inside cwd keeps the legacy
    # warn-only behavior so simple normalization callers are unaffected.
    path_str = str(user_path)
    if ".." in path_str and allowed_base is None:
        cwd_resolved = Path.cwd().resolve()
        is_absolute_input = path.is_absolute()

        try:
            resolved.relative_to(cwd_resolved)
            resolves_inside_cwd = True
        except ValueError:
            resolves_inside_cwd = False

        if is_absolute_input or not resolves_inside_cwd:
            logger.warning(
                f"Path traversal pattern detected and rejected: {user_path}"
            )
            raise PathTraversalError(str(user_path))

        # Legacy warn-only path for relative cwd-bound normalization.
        logger.warning(f"Path traversal pattern detected in: {user_path}")

    # Check existence if required
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    return resolved


def check_torch_security() -> bool:
    """
    Check PyTorch version for security features.

    PyTorch < 2.0 may not fully enforce weights_only=True parameter
    in torch.load(), which could allow arbitrary code execution
    through malicious pickle payloads.

    Returns:
        True if PyTorch version is secure, False otherwise

    Warns:
        SecurityWarning: If PyTorch version is below recommended minimum
    """
    try:
        import torch
        from packaging import version

        current_version = version.parse(torch.__version__.split("+")[0])  # Handle versions like "2.0.0+cu118"
        minimum_version = version.parse(MINIMUM_TORCH_VERSION)

        if current_version < minimum_version:
            warnings.warn(
                f"PyTorch {torch.__version__} may not fully enforce weights_only=True. "
                f"Upgrade to >= {MINIMUM_TORCH_VERSION} for improved security against "
                f"pickle deserialization attacks.",
                SecurityWarning,
                stacklevel=2,
            )
            logger.warning(
                f"Security: PyTorch {torch.__version__} < {MINIMUM_TORCH_VERSION}. "
                f"Consider upgrading for better protection against malicious model files."
            )
            return False

        return True

    except ImportError as e:
        logger.debug(f"Could not check PyTorch security: {e}")
        return True  # Assume safe if we can't check


# Stage C amend BACKEND-B-013: thread-safe module-init security check.
# Previously this was a lazy first-call cache inside ``safe_torch_load``,
# which:
#   (a) raced across threads in the Reflex UI (benign — N warnings instead
#       of one — but messy);
#   (b) NEVER fired when a .safetensors path was loaded because the lazy
#       check happened AFTER the early-return branch. A torch-old env
#       silently missed the warning on every safetensors load.
# Now the check fires once at module import, with a threading.Lock guarding
# the bookkeeping flag for symmetry with the rest of the module.
_torch_security_checked: bool = False
_torch_security_lock = __import__("threading").Lock()


def _ensure_torch_security_checked() -> None:
    """Stage C amend BACKEND-B-013: run :func:`check_torch_security` once
    per process under a lock so the warning fires before ANY load path
    (safetensors OR torch.load) ever returns.

    Idempotent and thread-safe. The lock contention is negligible — the
    flag is set once and every subsequent call short-circuits on the
    initial read before grabbing the lock.
    """
    global _torch_security_checked
    if _torch_security_checked:
        return
    with _torch_security_lock:
        if _torch_security_checked:
            return
        check_torch_security()
        _torch_security_checked = True


# Eagerly run the check at module import so the warning fires before any
# load happens, even on a pure-safetensors workflow that never touches
# torch.load. Wrapped in try/except so a bad torch env doesn't kill
# import — the lazy fallback in safe_torch_load still covers that case.
try:
    _ensure_torch_security_checked()
except Exception as _bootstrap_exc:  # pragma: no cover - defensive
    logger.debug(
        f"Eager torch-security check at import failed: {_bootstrap_exc}; "
        f"will retry lazily inside safe_torch_load."
    )


def safe_torch_load(
    path: str | Path,
    weights_only: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Safely load PyTorch weights with security checks.

    Prefers safetensors format when available, falls back to
    torch.load with weights_only=True.

    Stage C amend BACKEND-B-013: the torch-version security check is now
    invoked BEFORE the safetensors early-return so a torch-old environment
    loading .safetensors files still surfaces the warning. The check is
    thread-safe via :func:`_ensure_torch_security_checked`.

    Args:
        path: Path to the weights file
        weights_only: Enforce weights_only mode (default True)
        **kwargs: Additional arguments passed to torch.load

    Returns:
        Loaded state dict

    Raises:
        FileNotFoundError: If path doesn't exist
        RuntimeError: If loading fails
    """
    import torch

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")

    # Stage C amend BACKEND-B-013: run the security check BEFORE branching
    # on file suffix. Pre-fix, .safetensors paths returned without ever
    # firing the check; now both paths share the same security floor.
    _ensure_torch_security_checked()

    # Prefer safetensors format (more secure)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            logger.debug(f"Loading safetensors file: {path}")
            return load_file(str(path))
        except ImportError:
            logger.warning(
                "safetensors not installed. Install with: pip install safetensors"
            )

    # Fall back to torch.load with security enabled
    logger.debug(f"Loading PyTorch file with weights_only={weights_only}: {path}")
    result: dict[str, Any] = torch.load(path, weights_only=weights_only, **kwargs)  # nosec B614 - weights_only=True by default
    return result


def audit_log(
    operation: str,
    path: str | None = None,
    user: str | None = None,
    success: bool = True,
    details: dict | None = None,
) -> None:
    """
    Log security-sensitive operations for audit trail.

    Args:
        operation: Name of the operation (e.g., "model_load", "export")
        path: File path involved in the operation
        user: User performing the operation (if known)
        success: Whether the operation succeeded
        details: Additional context details
    """
    audit_logger = logging.getLogger("backpropagate.security.audit")

    log_data = {
        "operation": operation,
        "success": success,
    }

    if path:
        log_data["path"] = str(path)
    if user:
        log_data["user"] = user
    if details:
        log_data.update(details)

    level = logging.INFO if success else logging.WARNING
    audit_logger.log(level, f"AUDIT: {operation}", extra=log_data)
