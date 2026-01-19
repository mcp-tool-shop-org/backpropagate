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
from typing import Optional, Union

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
    """Error raised when path traversal is detected."""

    def __init__(self, path: str, allowed_base: Optional[str] = None):
        self.path = path
        self.allowed_base = allowed_base

        if allowed_base:
            message = f"Path '{path}' escapes allowed directory '{allowed_base}'"
        else:
            message = f"Path traversal detected in: {path}"

        super().__init__(message)


def safe_path(
    user_path: Union[str, Path],
    must_exist: bool = False,
    allowed_base: Optional[Union[str, Path]] = None,
    allow_relative: bool = True,
) -> Path:
    """
    Safely resolve and validate a user-provided path.

    Prevents path traversal attacks by ensuring the resolved path
    stays within allowed boundaries.

    Args:
        user_path: The user-provided path to validate
        must_exist: If True, raise FileNotFoundError if path doesn't exist
        allowed_base: If provided, ensure path is within this directory
        allow_relative: If False, reject relative paths

    Returns:
        Resolved, validated Path object

    Raises:
        PathTraversalError: If path escapes allowed directory
        FileNotFoundError: If must_exist=True and path doesn't exist
        ValueError: If allow_relative=False and path is relative

    Examples:
        >>> safe_path("/models/my_model", must_exist=True)
        PosixPath('/models/my_model')

        >>> safe_path("../../etc/passwd", allowed_base="/models")
        PathTraversalError: Path '../../etc/passwd' escapes allowed directory '/models'
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

    # Check for suspicious path components even without base restriction
    # This catches obvious traversal attempts like "../../"
    path_str = str(user_path)
    if ".." in path_str:
        # Log the attempt for security monitoring
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


def safe_torch_load(
    path: Union[str, Path],
    weights_only: bool = True,
    **kwargs
) -> dict:
    """
    Safely load PyTorch weights with security checks.

    Prefers safetensors format when available, falls back to
    torch.load with weights_only=True.

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

    # Check PyTorch security on first load
    check_torch_security()

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
    return torch.load(path, weights_only=weights_only, **kwargs)  # nosec B614 - weights_only=True by default


def audit_log(
    operation: str,
    path: Optional[str] = None,
    user: Optional[str] = None,
    success: bool = True,
    details: Optional[dict] = None,
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
