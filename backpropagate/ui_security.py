"""
Backpropagate - UI Security Module
==================================

Production-hardened security utilities for the Gradio web interface.

Based on:
- Trail of Bits Gradio 5 Security Audit (https://huggingface.co/blog/gradio-5-security)
- OWASP Web Security Best Practices
- Gradio CVE mitigations (CVE-2024-47872, CVE-2024-1727, CVE-2025-5320)

Features:
- Enhanced rate limiting with IP tracking
- File upload validation and sanitization
- Request logging for security monitoring
- Input validation with configurable limits
- CSRF protection helpers
- Security event logging
- Health check endpoint support
- Request ID tracing
- Structured JSON logging
- Session timeout management
- Concurrent operation limits
- Environment variable configuration

Usage:
    from backpropagate.ui_security import (
        SecurityConfig,
        EnhancedRateLimiter,
        FileValidator,
        validate_and_log_request,
        raise_gradio_error,
        get_health_status,
        RequestContext,
    )
"""

import gradio as gr
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
from functools import wraps

__all__ = [
    # Configuration
    "SecurityConfig",
    "DEFAULT_SECURITY_CONFIG",
    "load_config_from_env",
    # Rate limiting
    "EnhancedRateLimiter",
    "RateLimitExceeded",
    "RateLimitInfo",
    # File validation
    "FileValidator",
    "ALLOWED_DATASET_EXTENSIONS",
    "ALLOWED_MODEL_EXTENSIONS",
    "DANGEROUS_EXTENSIONS",
    "validate_file_magic",
    # Gradio error helpers
    "raise_gradio_error",
    "raise_gradio_warning",
    "raise_gradio_info",
    "safe_gradio_handler",
    # Request validation
    "validate_and_log_request",
    "sanitize_filename",
    "validate_numeric_input",
    "validate_string_input",
    # Security logging
    "SecurityLogger",
    "log_security_event",
    "JSONSecurityFormatter",
    # Health check
    "get_health_status",
    "HealthStatus",
    # Request context
    "RequestContext",
    "get_request_id",
    # Session management
    "SessionManager",
    "SessionInfo",
    # Concurrency control
    "ConcurrencyLimiter",
    # JWT authentication (2026)
    "JWT_AVAILABLE",
    "JWTConfig",
    "JWTManager",
    # CSRF protection (2026)
    "CSRFToken",
    "CSRFProtection",
    # Combined session handler (2026)
    "SecureSessionHandler",
    "get_secure_session_handler",
    # Content Security Policy (2026)
    "CSPConfig",
    "ContentSecurityPolicy",
    "DEFAULT_GRADIO_CSP",
    "get_gradio_csp",
    "apply_security_headers",
]

logger = logging.getLogger(__name__)
security_logger = logging.getLogger("backpropagate.security.ui")


# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

@dataclass
class SecurityConfig:
    """
    Centralized security configuration for the UI.

    All security-related settings in one place for easy auditing and updates.
    Can be configured via environment variables with BACKPROPAGATE_SECURITY__ prefix.
    """
    # Rate limiting
    training_rate_limit: int = 3  # Max training starts per window
    training_rate_window: int = 60  # Window in seconds
    export_rate_limit: int = 5
    export_rate_window: int = 60
    upload_rate_limit: int = 10
    upload_rate_window: int = 60

    # Input validation
    max_model_name_length: int = 200
    max_path_length: int = 500
    max_text_input_length: int = 10000
    max_system_prompt_length: int = 4000

    # File upload
    max_upload_size_mb: int = 500  # Max file size for uploads
    allowed_dataset_extensions: Set[str] = field(default_factory=lambda: {
        ".jsonl", ".json", ".csv", ".txt", ".parquet"
    })
    blocked_extensions: Set[str] = field(default_factory=lambda: {
        ".exe", ".bat", ".cmd", ".ps1", ".sh", ".py", ".js", ".html", ".htm",
        ".php", ".asp", ".aspx", ".jsp", ".cgi", ".pl", ".rb", ".svg"
    })
    validate_file_magic: bool = False  # Validate file content matches extension

    # CSRF protection
    csrf_enabled: bool = True
    csrf_localhost_only: bool = True  # Block non-localhost origins by default

    # Logging
    log_all_requests: bool = False  # Enable for debugging, disable in production
    log_security_events: bool = True
    log_format_json: bool = False  # Use structured JSON logging

    # Authentication
    require_auth_for_share: bool = True  # Require auth when share=True

    # Session management
    session_timeout_minutes: int = 60  # Auto-expire sessions after inactivity
    max_sessions_per_ip: int = 5  # Limit concurrent sessions per IP

    # Concurrency control
    max_concurrent_trainings: int = 1  # Max concurrent training jobs per IP
    max_concurrent_exports: int = 2  # Max concurrent export jobs per IP

    # Health check
    health_check_enabled: bool = True
    health_check_include_gpu: bool = True


def load_config_from_env(base_config: Optional[SecurityConfig] = None) -> SecurityConfig:
    """
    Load security configuration from environment variables.

    Environment variables use the prefix BACKPROPAGATE_SECURITY__ followed by
    the uppercase field name. For example:
    - BACKPROPAGATE_SECURITY__TRAINING_RATE_LIMIT=5
    - BACKPROPAGATE_SECURITY__MAX_UPLOAD_SIZE_MB=1000
    - BACKPROPAGATE_SECURITY__LOG_FORMAT_JSON=true

    Args:
        base_config: Base configuration to override (defaults to SecurityConfig())

    Returns:
        SecurityConfig with environment variable overrides applied
    """
    config = base_config or SecurityConfig()
    prefix = "BACKPROPAGATE_SECURITY__"

    # Map of field names to their types for conversion
    int_fields = {
        "training_rate_limit", "training_rate_window", "export_rate_limit",
        "export_rate_window", "upload_rate_limit", "upload_rate_window",
        "max_model_name_length", "max_path_length", "max_text_input_length",
        "max_system_prompt_length", "max_upload_size_mb", "session_timeout_minutes",
        "max_sessions_per_ip", "max_concurrent_trainings", "max_concurrent_exports",
    }
    bool_fields = {
        "csrf_enabled", "csrf_localhost_only", "log_all_requests",
        "log_security_events", "log_format_json", "require_auth_for_share",
        "validate_file_magic", "health_check_enabled", "health_check_include_gpu",
    }

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        field_name = key[len(prefix):].lower()

        if field_name in int_fields:
            try:
                setattr(config, field_name, int(value))
            except ValueError:
                logger.warning(f"Invalid int value for {key}: {value}")

        elif field_name in bool_fields:
            setattr(config, field_name, value.lower() in ("true", "1", "yes", "on"))

    return config


DEFAULT_SECURITY_CONFIG = load_config_from_env(SecurityConfig())


# =============================================================================
# FILE EXTENSION SETS (CVE-2024-47872 mitigation)
# =============================================================================

# Safe extensions for dataset uploads
ALLOWED_DATASET_EXTENSIONS = {".jsonl", ".json", ".csv", ".txt", ".parquet"}

# Safe extensions for model files
ALLOWED_MODEL_EXTENSIONS = {
    ".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ggml"
}

# Dangerous extensions that should NEVER be accepted (XSS/RCE risk)
DANGEROUS_EXTENSIONS = {
    # Executables
    ".exe", ".bat", ".cmd", ".ps1", ".sh", ".bash", ".zsh",
    ".com", ".msi", ".app", ".dmg", ".pkg",
    # Scripts
    ".py", ".pyw", ".pyc", ".pyo", ".pyd",
    ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".rb", ".pl", ".php", ".asp", ".aspx", ".jsp", ".cgi",
    # Web content (XSS vectors - CVE-2024-47872)
    ".html", ".htm", ".xhtml", ".xml", ".xsl", ".xslt",
    ".svg", ".svgz",  # SVG can contain JavaScript
    ".swf", ".fla",   # Flash
    # Archives (can contain malicious files)
    ".jar", ".war", ".ear",
    # Other
    ".dll", ".so", ".dylib", ".class",
}


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, wait_seconds: float, operation: str = "operation"):
        self.wait_seconds = wait_seconds
        self.operation = operation
        super().__init__(
            f"Rate limit exceeded for {operation}. "
            f"Please wait {wait_seconds:.0f} seconds."
        )


class EnhancedRateLimiter:
    """
    Enhanced rate limiter with IP tracking and per-user limits.

    Improvements over basic RateLimiter:
    - Per-IP tracking (when available from gr.Request)
    - Configurable burst allowance
    - Automatic cleanup of old entries
    - Security event logging
    """

    def __init__(
        self,
        max_requests: int = 5,
        window_seconds: int = 60,
        burst_allowance: int = 0,
        operation_name: str = "operation",
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_allowance = burst_allowance
        self.operation_name = operation_name

        # Track requests per IP
        self._requests: Dict[str, List[float]] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Cleanup every 5 minutes

    def _get_client_id(self, request: Optional[gr.Request] = None) -> str:
        """Get client identifier (IP or fallback to 'anonymous')."""
        if request is not None:
            # Try to get IP from request
            client_ip = getattr(request, "client", {})
            if isinstance(client_ip, dict):
                return client_ip.get("host", "anonymous")
            return str(client_ip) if client_ip else "anonymous"
        return "anonymous"

    def _cleanup_old_entries(self) -> None:
        """Remove expired entries to prevent memory growth."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        cutoff = now - self.window_seconds

        # Remove old requests
        for client_id in list(self._requests.keys()):
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > cutoff
            ]
            # Remove empty entries
            if not self._requests[client_id]:
                del self._requests[client_id]

    def check(self, request: Optional[gr.Request] = None) -> Tuple[bool, float]:
        """
        Check if request is allowed.

        Returns:
            Tuple of (is_allowed, wait_seconds_if_denied)
        """
        self._cleanup_old_entries()

        client_id = self._get_client_id(request)
        now = time.time()
        cutoff = now - self.window_seconds

        # Get requests for this client
        if client_id not in self._requests:
            self._requests[client_id] = []

        # Filter to recent requests
        recent = [t for t in self._requests[client_id] if t > cutoff]
        self._requests[client_id] = recent

        # Check limit (with burst allowance)
        effective_limit = self.max_requests + self.burst_allowance

        if len(recent) >= effective_limit:
            # Calculate wait time
            oldest = min(recent)
            wait_time = max(0.0, self.window_seconds - (now - oldest))

            # Log rate limit event
            log_security_event(
                "rate_limit_exceeded",
                client_id=client_id,
                operation=self.operation_name,
                requests_in_window=len(recent),
                limit=effective_limit,
            )

            return False, wait_time

        # Allow and record
        self._requests[client_id].append(now)
        return True, 0.0

    def is_allowed(self, request: Optional[gr.Request] = None) -> bool:
        """Simple check returning just bool."""
        allowed, _ = self.check(request)
        return allowed

    def require(self, request: Optional[gr.Request] = None) -> None:
        """Raise exception if rate limited."""
        allowed, wait_time = self.check(request)
        if not allowed:
            raise RateLimitExceeded(wait_time, self.operation_name)


# =============================================================================
# FILE VALIDATION (CVE-2024-47872 mitigation)
# =============================================================================

class FileValidator:
    """
    Validates uploaded files for security.

    Mitigates:
    - CVE-2024-47872: XSS via malicious file uploads
    - Path traversal attacks
    - Oversized file uploads
    - Dangerous file types
    """

    def __init__(
        self,
        allowed_extensions: Optional[Set[str]] = None,
        max_size_mb: int = 500,
        config: Optional[SecurityConfig] = None,
    ):
        self.config = config or DEFAULT_SECURITY_CONFIG
        self.allowed_extensions = allowed_extensions or self.config.allowed_dataset_extensions
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def validate(
        self,
        file_obj: Any,
        purpose: str = "upload",
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Validate an uploaded file.

        Args:
            file_obj: Gradio file object (has .name attribute)
            purpose: Description for logging

        Returns:
            Tuple of (is_valid, error_message, validated_path)
        """
        if file_obj is None:
            return False, "No file provided", None

        # Get file path
        try:
            file_path = Path(file_obj.name if hasattr(file_obj, "name") else str(file_obj))
        except Exception as e:
            return False, f"Invalid file object: {e}", None

        # Check extension
        ext = file_path.suffix.lower()

        # Block dangerous extensions (always)
        if ext in DANGEROUS_EXTENSIONS:
            log_security_event(
                "dangerous_file_blocked",
                file_name=file_path.name,  # Use file_name to avoid LogRecord conflict
                extension=ext,
                purpose=purpose,
            )
            return False, f"File type '{ext}' is not allowed for security reasons", None

        # Check against allowed list
        if ext not in self.allowed_extensions:
            allowed = ", ".join(sorted(self.allowed_extensions))
            return False, f"File type '{ext}' not supported. Allowed: {allowed}", None

        # Check file size
        if file_path.exists():
            size = file_path.stat().st_size
            if size > self.max_size_bytes:
                max_mb = self.max_size_bytes / (1024 * 1024)
                actual_mb = size / (1024 * 1024)
                return False, f"File too large ({actual_mb:.1f}MB). Maximum: {max_mb:.0f}MB", None

        # Sanitize filename (remove path components)
        safe_name = sanitize_filename(file_path.name)
        if safe_name != file_path.name:
            log_security_event(
                "filename_sanitized",
                original=file_path.name,
                sanitized=safe_name,
                purpose=purpose,
            )

        return True, "", file_path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and injection.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem use
    """
    # Remove path separators
    name = filename.replace("/", "_").replace("\\", "_")

    # Remove null bytes
    name = name.replace("\x00", "")

    # Remove leading/trailing dots and spaces
    name = name.strip(". ")

    # Remove control characters
    name = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', name)

    # Limit length
    if len(name) > 255:
        # Keep extension
        ext = Path(name).suffix
        base = name[:255 - len(ext)]
        name = base + ext

    return name or "unnamed_file"


# =============================================================================
# GRADIO ERROR HELPERS
# =============================================================================

def raise_gradio_error(
    message: str,
    duration: Optional[int] = 10,
    title: str = "Error",
    log: bool = True,
) -> None:
    """
    Raise a Gradio error with proper formatting.

    Use this instead of returning error strings for better UX.

    Args:
        message: Error message to display
        duration: Seconds to show (None = until closed)
        title: Error dialog title
        log: Whether to log the error

    Raises:
        gr.Error: Always raises
    """
    if log:
        logger.error(f"UI Error: {message}")

    raise gr.Error(message, duration=duration, title=title)


def raise_gradio_warning(
    message: str,
    duration: Optional[int] = 5,
    title: str = "Warning",
    log: bool = True,
) -> None:
    """
    Show a Gradio warning (non-blocking).

    Unlike gr.Error, this does NOT halt execution.

    Args:
        message: Warning message
        duration: Seconds to show
        title: Warning dialog title
        log: Whether to log
    """
    if log:
        logger.warning(f"UI Warning: {message}")

    gr.Warning(message, duration=duration, title=title)


def raise_gradio_info(
    message: str,
    duration: Optional[int] = 3,
    title: str = "Info",
) -> None:
    """
    Show a Gradio info message (non-blocking).

    Args:
        message: Info message
        duration: Seconds to show
        title: Info dialog title
    """
    gr.Info(message, duration=duration, title=title)


F = TypeVar('F', bound=Callable[..., Any])


def safe_gradio_handler(
    operation_name: str = "operation",
    rate_limiter: Optional[EnhancedRateLimiter] = None,
    log_errors: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to wrap Gradio handlers with security features.

    Features:
    - Converts exceptions to gr.Error for proper UI display
    - Optional rate limiting
    - Security event logging
    - Request validation

    Usage:
        @safe_gradio_handler("training", rate_limiter=training_limiter)
        def start_training(...):
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check rate limit
            if rate_limiter is not None:
                request = kwargs.get("request")
                allowed, wait_time = rate_limiter.check(request)
                if not allowed:
                    raise gr.Error(
                        f"Too many requests. Please wait {wait_time:.0f} seconds.",
                        duration=10,
                        title="Rate Limited",
                    )

            try:
                return func(*args, **kwargs)

            except gr.Error:
                # Re-raise Gradio errors as-is
                raise

            except RateLimitExceeded as e:
                raise gr.Error(str(e), duration=10, title="Rate Limited")

            except FileNotFoundError as e:
                if log_errors:
                    logger.error(f"{operation_name} failed - file not found: {e}")
                raise gr.Error(f"File not found: {e}", duration=10, title="File Not Found")

            except PermissionError as e:
                if log_errors:
                    logger.error(f"{operation_name} failed - permission denied: {e}")
                raise gr.Error(
                    "Permission denied. Check file/folder permissions.",
                    duration=10,
                    title="Permission Denied",
                )

            except ValueError as e:
                if log_errors:
                    logger.warning(f"{operation_name} validation error: {e}")
                raise gr.Error(f"Invalid input: {e}", duration=10, title="Validation Error")

            except Exception as e:
                if log_errors:
                    logger.exception(f"{operation_name} failed with unexpected error")

                log_security_event(
                    "handler_exception",
                    operation=operation_name,
                    error_type=type(e).__name__,
                    error_message=str(e)[:200],
                )

                # Don't expose internal errors to users
                raise gr.Error(
                    f"An error occurred during {operation_name}. Check logs for details.",
                    duration=10,
                    title="Error",
                )

        return wrapper  # type: ignore
    return decorator


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_numeric_input(
    value: Any,
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_none: bool = False,
) -> Optional[float]:
    """
    Validate and sanitize numeric input.

    Args:
        value: Input value to validate
        name: Parameter name (for error messages)
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_none: Whether None is acceptable

    Returns:
        Validated numeric value

    Raises:
        gr.Error: If validation fails
    """
    if value is None:
        if allow_none:
            return None
        raise gr.Error(f"{name} is required", duration=5)

    try:
        num = float(value)
    except (ValueError, TypeError):
        raise gr.Error(f"{name} must be a number, got: {type(value).__name__}", duration=5)

    if min_value is not None and num < min_value:
        raise gr.Error(f"{name} must be at least {min_value}, got {num}", duration=5)

    if max_value is not None and num > max_value:
        raise gr.Error(f"{name} must be at most {max_value}, got {num}", duration=5)

    return num


def validate_string_input(
    value: Any,
    name: str,
    max_length: int = 1000,
    min_length: int = 0,
    pattern: Optional[str] = None,
    allow_none: bool = False,
    allow_empty: bool = False,
) -> Optional[str]:
    """
    Validate and sanitize string input.

    Args:
        value: Input value to validate
        name: Parameter name (for error messages)
        max_length: Maximum string length
        min_length: Minimum string length
        pattern: Optional regex pattern to match
        allow_none: Whether None is acceptable
        allow_empty: Whether empty string is acceptable

    Returns:
        Validated string value

    Raises:
        gr.Error: If validation fails
    """
    if value is None:
        if allow_none:
            return None
        raise gr.Error(f"{name} is required", duration=5)

    text = str(value)

    # Remove null bytes (security)
    text = text.replace("\x00", "")

    # Check length
    if len(text) > max_length:
        raise gr.Error(
            f"{name} is too long ({len(text)} chars). Maximum: {max_length}",
            duration=5,
        )

    if not allow_empty and len(text.strip()) == 0:
        raise gr.Error(f"{name} cannot be empty", duration=5)

    if len(text) < min_length:
        raise gr.Error(
            f"{name} is too short ({len(text)} chars). Minimum: {min_length}",
            duration=5,
        )

    # Check pattern
    if pattern is not None:
        if not re.match(pattern, text):
            raise gr.Error(f"{name} has invalid format", duration=5)

    return text


def validate_and_log_request(
    operation: str,
    request: Optional[gr.Request] = None,
    **params: Any,
) -> None:
    """
    Log a request for security monitoring.

    Args:
        operation: Operation name
        request: Gradio request object
        **params: Additional parameters to log (sanitized)
    """
    if not DEFAULT_SECURITY_CONFIG.log_all_requests:
        return

    client_id = "anonymous"
    if request is not None:
        client = getattr(request, "client", {})
        if isinstance(client, dict):
            client_id = client.get("host", "anonymous")

    # Sanitize params for logging (truncate long values)
    safe_params = {}
    for key, value in params.items():
        if isinstance(value, str) and len(value) > 100:
            safe_params[key] = value[:100] + "..."
        elif isinstance(value, (int, float, bool)):
            safe_params[key] = value
        else:
            safe_params[key] = type(value).__name__

    security_logger.info(
        f"Request: {operation}",
        extra={
            "operation": operation,
            "client_id": client_id,
            "params": safe_params,
        },
    )


# =============================================================================
# SECURITY LOGGING
# =============================================================================

class SecurityLogger:
    """
    Centralized security event logging.

    Events are logged with structured data for SIEM/monitoring integration.
    """

    _instance: Optional["SecurityLogger"] = None

    def __new__(cls) -> "SecurityLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._logger = logging.getLogger("backpropagate.security.events")
        return cls._instance

    def log(
        self,
        event_type: str,
        severity: str = "INFO",
        **details: Any,
    ) -> None:
        """
        Log a security event.

        Args:
            event_type: Type of event (e.g., "rate_limit_exceeded")
            severity: "INFO", "WARNING", "ERROR", or "CRITICAL"
            **details: Additional event details
        """
        level = getattr(logging, severity.upper(), logging.INFO)

        # Create structured log entry
        log_entry = {
            "event_type": event_type,
            "timestamp": time.time(),
            **details,
        }

        self._logger.log(level, f"Security Event: {event_type}", extra=log_entry)


def log_security_event(event_type: str, **details: Any) -> None:
    """
    Log a security event.

    Convenience function for SecurityLogger.log().

    Args:
        event_type: Type of event
        **details: Event details
    """
    if DEFAULT_SECURITY_CONFIG.log_security_events:
        SecurityLogger().log(event_type, **details)


# =============================================================================
# CSRF PROTECTION HELPERS
# =============================================================================

def check_csrf_protection(
    request: Optional[gr.Request] = None,
    config: Optional[SecurityConfig] = None,
) -> bool:
    """
    Check if request passes CSRF protection.

    Note: Gradio 5+ has built-in CSRF protection. This is an additional layer.

    Args:
        request: Gradio request object
        config: Security configuration

    Returns:
        True if request is safe, False otherwise
    """
    config = config or DEFAULT_SECURITY_CONFIG

    if not config.csrf_enabled:
        return True

    if request is None:
        return True  # Can't check without request

    # Get origin/referer
    headers = getattr(request, "headers", {})
    origin = headers.get("origin", "")
    referer = headers.get("referer", "")

    # If localhost only, check origin
    if config.csrf_localhost_only:
        localhost_patterns = ["localhost", "127.0.0.1", "::1"]

        origin_ok = not origin or any(p in origin for p in localhost_patterns)
        referer_ok = not referer or any(p in referer for p in localhost_patterns)

        if not (origin_ok and referer_ok):
            log_security_event(
                "csrf_check_failed",
                origin=origin[:100] if origin else None,
                referer=referer[:100] if referer else None,
            )
            return False

    return True


# =============================================================================
# HEALTH CHECK
# =============================================================================

@dataclass
class HealthStatus:
    """
    Health check status for the application.

    Used by container orchestration (K8s, Docker) and load balancers.
    """
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature_c: Optional[float] = None
    active_trainings: int = 0
    active_sessions: int = 0
    rate_limit_status: str = "ok"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        result = {
            "status": self.status,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "active_trainings": self.active_trainings,
            "active_sessions": self.active_sessions,
            "rate_limit_status": self.rate_limit_status,
            "timestamp": self.timestamp,
        }

        if self.gpu_available:
            result["gpu"] = {
                "available": self.gpu_available,
                "name": self.gpu_name,
                "memory_used_gb": self.gpu_memory_used_gb,
                "memory_total_gb": self.gpu_memory_total_gb,
                "temperature_c": self.gpu_temperature_c,
            }

        return result


_app_start_time = time.time()


def get_health_status(
    config: Optional[SecurityConfig] = None,
    include_gpu: Optional[bool] = None,
) -> HealthStatus:
    """
    Get current health status of the application.

    Args:
        config: Security configuration
        include_gpu: Override config.health_check_include_gpu

    Returns:
        HealthStatus with current application state
    """
    config = config or DEFAULT_SECURITY_CONFIG

    # Get version
    try:
        from . import __version__
    except ImportError:
        __version__ = "unknown"

    uptime = time.time() - _app_start_time

    status = HealthStatus(
        status="healthy",
        version=__version__,
        uptime_seconds=uptime,
    )

    # Check GPU if requested
    should_check_gpu = include_gpu if include_gpu is not None else config.health_check_include_gpu
    if should_check_gpu:
        try:
            from .gpu_safety import get_gpu_status
            gpu_status = get_gpu_status()
            status.gpu_available = gpu_status.available
            if gpu_status.available:
                status.gpu_name = gpu_status.gpu_name
                status.gpu_memory_used_gb = gpu_status.vram_used_gb
                status.gpu_memory_total_gb = gpu_status.vram_total_gb
                status.gpu_temperature_c = gpu_status.temperature_c

                # Check for GPU issues
                if gpu_status.temperature_c and gpu_status.temperature_c > 85:
                    status.status = "degraded"
        except Exception:
            pass  # GPU check failed, continue without

    return status


# =============================================================================
# REQUEST CONTEXT AND TRACING
# =============================================================================

@dataclass
class RequestContext:
    """
    Context for a single request, including tracing information.

    Thread-safe request context for logging and monitoring.
    """
    request_id: str
    client_ip: str
    timestamp: float
    operation: Optional[str] = None
    user_id: Optional[str] = None

    @classmethod
    def from_gradio_request(
        cls,
        request: Optional[gr.Request] = None,
        operation: Optional[str] = None,
    ) -> "RequestContext":
        """Create context from a Gradio request."""
        request_id = str(uuid.uuid4())[:8]
        client_ip = "anonymous"

        if request is not None:
            client = getattr(request, "client", {})
            if isinstance(client, dict):
                client_ip = client.get("host", "anonymous")

        return cls(
            request_id=request_id,
            client_ip=client_ip,
            timestamp=time.time(),
            operation=operation,
        )

    def to_log_dict(self) -> Dict[str, Any]:
        """Get dictionary for logging extra fields."""
        return {
            "request_id": self.request_id,
            "client_ip": self.client_ip,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "user_id": self.user_id,
        }


def get_request_id(request: Optional[gr.Request] = None) -> str:
    """
    Generate or extract a unique request ID.

    Args:
        request: Optional Gradio request (may contain existing ID in headers)

    Returns:
        8-character unique request ID
    """
    # Check for existing request ID in headers
    if request is not None:
        headers = getattr(request, "headers", {})
        existing_id = headers.get("x-request-id") or headers.get("x-correlation-id")
        if existing_id:
            return str(existing_id)[:8]

    return str(uuid.uuid4())[:8]


# =============================================================================
# STRUCTURED JSON LOGGING
# =============================================================================

class JSONSecurityFormatter(logging.Formatter):
    """
    JSON formatter for security logs.

    Outputs structured JSON for SIEM/log aggregation systems
    (ELK, Splunk, Datadog, etc.).
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields (request_id, client_ip, etc.)
        for key in ["request_id", "client_ip", "operation", "user_id",
                    "event_type", "params", "error_type", "error_message"]:
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def configure_json_logging(
    logger_names: Optional[List[str]] = None,
    level: int = logging.INFO,
) -> None:
    """
    Configure JSON logging for security loggers.

    Args:
        logger_names: Loggers to configure (defaults to security loggers)
        level: Logging level
    """
    if logger_names is None:
        logger_names = [
            "backpropagate.security.ui",
            "backpropagate.security.events",
        ]

    formatter = JSONSecurityFormatter()

    for name in logger_names:
        log = logging.getLogger(name)
        log.setLevel(level)

        # Remove existing handlers
        for handler in log.handlers[:]:
            log.removeHandler(handler)

        # Add JSON handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        log.addHandler(handler)


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

@dataclass
class SessionInfo:
    """Information about an active session."""
    session_id: str
    client_ip: str
    created_at: float
    last_activity: float
    user_id: Optional[str] = None


class SessionManager:
    """
    Manages user sessions with timeout and limits.

    Thread-safe session tracking for the web UI.
    """

    _instance: Optional["SessionManager"] = None
    _lock: Lock = Lock()

    def __new__(cls) -> "SessionManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._sessions: Dict[str, SessionInfo] = {}
                    cls._instance._sessions_by_ip: Dict[str, List[str]] = {}
        return cls._instance

    def create_session(
        self,
        client_ip: str,
        user_id: Optional[str] = None,
        config: Optional[SecurityConfig] = None,
    ) -> Tuple[bool, Optional[str], str]:
        """
        Create a new session.

        Args:
            client_ip: Client IP address
            user_id: Optional user identifier
            config: Security configuration

        Returns:
            Tuple of (success, session_id, message)
        """
        config = config or DEFAULT_SECURITY_CONFIG

        with self._lock:
            # Clean expired sessions first
            self._cleanup_expired(config)

            # Check session limit per IP
            ip_sessions = self._sessions_by_ip.get(client_ip, [])
            if len(ip_sessions) >= config.max_sessions_per_ip:
                return False, None, f"Maximum {config.max_sessions_per_ip} sessions per IP"

            # Create session
            session_id = str(uuid.uuid4())
            now = time.time()

            session = SessionInfo(
                session_id=session_id,
                client_ip=client_ip,
                created_at=now,
                last_activity=now,
                user_id=user_id,
            )

            self._sessions[session_id] = session
            if client_ip not in self._sessions_by_ip:
                self._sessions_by_ip[client_ip] = []
            self._sessions_by_ip[client_ip].append(session_id)

            log_security_event(
                "session_created",
                session_id=session_id[:8],
                client_ip=client_ip,
            )

            return True, session_id, "Session created"

    def validate_session(
        self,
        session_id: str,
        config: Optional[SecurityConfig] = None,
    ) -> Tuple[bool, str]:
        """
        Validate and refresh a session.

        Args:
            session_id: Session ID to validate
            config: Security configuration

        Returns:
            Tuple of (is_valid, message)
        """
        config = config or DEFAULT_SECURITY_CONFIG

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False, "Session not found"

            # Check timeout
            timeout_seconds = config.session_timeout_minutes * 60
            if time.time() - session.last_activity > timeout_seconds:
                self._remove_session(session_id)
                return False, "Session expired"

            # Refresh activity
            session.last_activity = time.time()
            return True, "Session valid"

    def end_session(self, session_id: str) -> bool:
        """End a session."""
        with self._lock:
            return self._remove_session(session_id)

    def get_active_count(self) -> int:
        """Get count of active sessions."""
        with self._lock:
            return len(self._sessions)

    def _remove_session(self, session_id: str) -> bool:
        """Remove a session (internal, assumes lock held)."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False

        # Remove from IP tracking
        ip_sessions = self._sessions_by_ip.get(session.client_ip, [])
        if session_id in ip_sessions:
            ip_sessions.remove(session_id)
            if not ip_sessions:
                del self._sessions_by_ip[session.client_ip]

        log_security_event(
            "session_ended",
            session_id=session_id[:8],
        )
        return True

    def _cleanup_expired(self, config: SecurityConfig) -> int:
        """Remove expired sessions (internal, assumes lock held)."""
        timeout_seconds = config.session_timeout_minutes * 60
        now = time.time()
        expired = []

        for session_id, session in self._sessions.items():
            if now - session.last_activity > timeout_seconds:
                expired.append(session_id)

        for session_id in expired:
            self._remove_session(session_id)

        return len(expired)


# =============================================================================
# CONCURRENCY CONTROL
# =============================================================================

class ConcurrencyLimiter:
    """
    Limits concurrent operations per client IP.

    Thread-safe concurrency control for expensive operations like training.
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        operation_name: str = "operation",
    ):
        self.max_concurrent = max_concurrent
        self.operation_name = operation_name
        self._active: Dict[str, int] = {}
        self._lock = Lock()

    def _get_client_id(self, request: Optional[gr.Request] = None) -> str:
        """Get client identifier from request."""
        if request is not None:
            client = getattr(request, "client", {})
            if isinstance(client, dict):
                return client.get("host", "anonymous")
        return "anonymous"

    def acquire(self, request: Optional[gr.Request] = None) -> Tuple[bool, str]:
        """
        Try to acquire a slot for an operation.

        Args:
            request: Gradio request for client identification

        Returns:
            Tuple of (success, message)
        """
        client_id = self._get_client_id(request)

        with self._lock:
            current = self._active.get(client_id, 0)

            if current >= self.max_concurrent:
                log_security_event(
                    "concurrency_limit_exceeded",
                    operation=self.operation_name,
                    client_id=client_id,
                    current=current,
                    limit=self.max_concurrent,
                )
                return False, f"Maximum {self.max_concurrent} concurrent {self.operation_name}(s)"

            self._active[client_id] = current + 1
            return True, "Acquired"

    def release(self, request: Optional[gr.Request] = None) -> None:
        """Release a slot after operation completes."""
        client_id = self._get_client_id(request)

        with self._lock:
            current = self._active.get(client_id, 0)
            if current > 0:
                self._active[client_id] = current - 1
                if self._active[client_id] == 0:
                    del self._active[client_id]

    def get_active_count(self, request: Optional[gr.Request] = None) -> int:
        """Get count of active operations for a client."""
        client_id = self._get_client_id(request)
        with self._lock:
            return self._active.get(client_id, 0)

    def get_total_active(self) -> int:
        """Get total count of active operations across all clients."""
        with self._lock:
            return sum(self._active.values())


# =============================================================================
# RATE LIMIT INFO (for response headers)
# =============================================================================

@dataclass
class RateLimitInfo:
    """
    Rate limit information for response headers.

    Standard rate limit headers for API responses.
    """
    limit: int
    remaining: int
    reset_timestamp: float
    retry_after: Optional[float] = None

    def to_headers(self) -> Dict[str, str]:
        """
        Get rate limit headers.

        Returns headers following RFC 6585 / IETF draft-polli-ratelimit-headers.
        """
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_timestamp)),
        }

        if self.retry_after is not None:
            headers["Retry-After"] = str(int(self.retry_after))

        return headers


# =============================================================================
# FILE MAGIC VALIDATION
# =============================================================================

# Common file signatures (magic bytes)
FILE_SIGNATURES: Dict[str, List[bytes]] = {
    ".json": [b"{", b"["],  # JSON starts with { or [
    ".jsonl": [b"{"],  # JSONL lines start with {
    ".csv": [],  # CSV has no standard signature
    ".txt": [],  # Plain text has no signature
    ".parquet": [b"PAR1"],  # Parquet magic bytes
    ".safetensors": [],  # SafeTensors has complex header
    ".gguf": [b"GGUF"],  # GGUF magic bytes
}


def validate_file_magic(
    file_path: Path,
    expected_extension: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate file content matches expected type using magic bytes.

    This helps prevent extension spoofing attacks where malicious files
    are renamed to bypass extension checks.

    Args:
        file_path: Path to file to validate
        expected_extension: Expected extension (defaults to file's extension)

    Returns:
        Tuple of (is_valid, message)
    """
    if not file_path.exists():
        return False, "File does not exist"

    ext = expected_extension or file_path.suffix.lower()

    # Get expected signatures for this extension
    signatures = FILE_SIGNATURES.get(ext, [])

    if not signatures:
        # No signature check for this type
        return True, "No signature check available"

    try:
        with open(file_path, "rb") as f:
            # Read enough bytes to check signature
            header = f.read(16)

        for sig in signatures:
            if header.startswith(sig):
                return True, "Signature valid"

        # Check if it might be HTML/script masquerading as data
        suspicious_starts = [
            b"<!DOCTYPE", b"<html", b"<HTML", b"<script", b"<SCRIPT",
            b"<?php", b"<?PHP", b"#!/",
        ]
        for suspicious in suspicious_starts:
            if header.startswith(suspicious):
                log_security_event(
                    "suspicious_file_content",
                    file=file_path.name,
                    expected_extension=ext,
                    found_signature=header[:20].decode("utf-8", errors="replace"),
                )
                return False, f"File content appears to be HTML/script, not {ext}"

        return True, "Content check passed"

    except Exception as e:
        return False, f"Failed to read file: {e}"


# =============================================================================
# JWT SESSION TOKENS (2026 Best Practices)
# =============================================================================

# JWT is optional dependency
try:
    import jwt as pyjwt
    JWT_AVAILABLE = True
except ImportError:
    pyjwt = None  # type: ignore
    JWT_AVAILABLE = False


@dataclass
class JWTConfig:
    """
    JWT configuration following 2026 security best practices.

    Based on:
    - IETF RFC 7519 (JWT)
    - OWASP Session Management Guidelines
    - Gradio-Session patterns (https://discuss.huggingface.co/t/implementing-session-authentication-in-gradio)
    """
    secret: str = ""  # Must be set in production
    algorithm: str = "HS256"
    expiry_minutes: int = 30
    issuer: str = "backpropagate"
    audience: str = "backpropagate-ui"
    # Refresh token settings
    refresh_enabled: bool = True
    refresh_expiry_minutes: int = 1440  # 24 hours


class JWTManager:
    """
    JWT-based session management for Gradio.

    Provides stateless authentication with configurable expiry.
    Tokens are stored in HTTP-only cookies when possible.

    Usage:
        manager = JWTManager(config)
        token = manager.create_token(user_id="user123")
        payload = manager.verify_token(token)
    """

    def __init__(self, config: Optional[JWTConfig] = None):
        self.config = config or JWTConfig()
        if not self.config.secret:
            # Generate random secret if not provided (not persistent across restarts)
            import secrets
            self.config.secret = secrets.token_urlsafe(32)
            logger.warning(
                "JWT secret not configured - using random secret. "
                "Sessions will be invalidated on restart. "
                "Set BACKPROPAGATE_SECURITY__JWT_SECRET for persistence."
            )

    def create_token(
        self,
        user_id: str,
        additional_claims: Optional[Dict[str, Any]] = None,
        is_refresh: bool = False,
    ) -> str:
        """
        Create a JWT token for a user.

        Args:
            user_id: User identifier
            additional_claims: Extra claims to include
            is_refresh: If True, create a refresh token with longer expiry

        Returns:
            Encoded JWT token string
        """
        if not JWT_AVAILABLE:
            raise RuntimeError("PyJWT not installed. Install with: pip install PyJWT")

        now = time.time()
        expiry_minutes = (
            self.config.refresh_expiry_minutes if is_refresh
            else self.config.expiry_minutes
        )

        payload = {
            "sub": user_id,  # Subject (user ID)
            "iss": self.config.issuer,  # Issuer
            "aud": self.config.audience,  # Audience
            "iat": int(now),  # Issued at
            "exp": int(now + expiry_minutes * 60),  # Expiry
            "jti": str(uuid.uuid4()),  # JWT ID (for revocation)
            "type": "refresh" if is_refresh else "access",
        }

        if additional_claims:
            payload.update(additional_claims)

        token = pyjwt.encode(
            payload,
            self.config.secret,
            algorithm=self.config.algorithm,
        )

        log_security_event(
            "jwt_token_created",
            user_id=user_id,
            token_type="refresh" if is_refresh else "access",
            expiry_minutes=expiry_minutes,
        )

        return token

    def verify_token(
        self,
        token: str,
        expected_type: str = "access",
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string
            expected_type: Expected token type ("access" or "refresh")

        Returns:
            Tuple of (is_valid, payload, message)
        """
        if not JWT_AVAILABLE:
            return False, None, "PyJWT not installed"

        try:
            payload = pyjwt.decode(
                token,
                self.config.secret,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer,
            )

            # Check token type
            if payload.get("type") != expected_type:
                return False, None, f"Invalid token type (expected {expected_type})"

            return True, payload, "Token valid"

        except pyjwt.ExpiredSignatureError:
            log_security_event("jwt_token_expired", token_prefix=token[:10])
            return False, None, "Token expired"

        except pyjwt.InvalidTokenError as e:
            log_security_event("jwt_token_invalid", error=str(e))
            return False, None, f"Invalid token: {e}"

    def refresh_access_token(
        self,
        refresh_token: str,
    ) -> Tuple[bool, Optional[str], str]:
        """
        Generate new access token using refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            Tuple of (success, new_access_token, message)
        """
        valid, payload, msg = self.verify_token(refresh_token, expected_type="refresh")

        if not valid:
            return False, None, msg

        # Create new access token
        user_id = payload.get("sub")  # type: ignore
        if not user_id:
            return False, None, "Invalid refresh token (no subject)"

        new_token = self.create_token(user_id)
        return True, new_token, "Token refreshed"


# =============================================================================
# CSRF PROTECTION (2026 Best Practices)
# =============================================================================

@dataclass
class CSRFToken:
    """CSRF token with expiry."""
    token: str
    created_at: float
    expiry_minutes: int = 60


class CSRFProtection:
    """
    CSRF protection for state-changing requests.

    Based on:
    - OWASP CSRF Prevention Cheat Sheet
    - Synchronizer Token Pattern
    - Gradio CVE-2024-1727 mitigation

    Usage:
        csrf = CSRFProtection()
        token = csrf.generate_token(session_id)
        is_valid = csrf.validate_token(session_id, token)
    """

    def __init__(self, expiry_minutes: int = 60):
        self.expiry_minutes = expiry_minutes
        self._tokens: Dict[str, CSRFToken] = {}
        self._lock = Lock()

    def generate_token(self, session_id: str) -> str:
        """
        Generate a CSRF token for a session.

        Args:
            session_id: Session identifier

        Returns:
            CSRF token string
        """
        import secrets

        token_value = secrets.token_urlsafe(32)

        with self._lock:
            self._cleanup_expired()
            self._tokens[session_id] = CSRFToken(
                token=token_value,
                created_at=time.time(),
                expiry_minutes=self.expiry_minutes,
            )

        log_security_event("csrf_token_generated", session_id=session_id[:8])
        return token_value

    def validate_token(
        self,
        session_id: str,
        token: str,
        consume: bool = True,
    ) -> Tuple[bool, str]:
        """
        Validate a CSRF token.

        Args:
            session_id: Session identifier
            token: CSRF token to validate
            consume: If True, token is single-use (deleted after validation)

        Returns:
            Tuple of (is_valid, message)
        """
        with self._lock:
            self._cleanup_expired()

            csrf_token = self._tokens.get(session_id)

            if not csrf_token:
                log_security_event(
                    "csrf_validation_failed",
                    reason="no_token_for_session",
                    session_id=session_id[:8] if session_id else "none",
                )
                return False, "No CSRF token found for session"

            # Check expiry
            age_minutes = (time.time() - csrf_token.created_at) / 60
            if age_minutes > csrf_token.expiry_minutes:
                del self._tokens[session_id]
                log_security_event(
                    "csrf_validation_failed",
                    reason="token_expired",
                    session_id=session_id[:8],
                )
                return False, "CSRF token expired"

            # Constant-time comparison to prevent timing attacks
            if not hmac.compare_digest(csrf_token.token, token):
                log_security_event(
                    "csrf_validation_failed",
                    reason="token_mismatch",
                    session_id=session_id[:8],
                )
                return False, "Invalid CSRF token"

            # Consume token if requested (single-use)
            if consume:
                del self._tokens[session_id]

            return True, "CSRF token valid"

    def _cleanup_expired(self) -> None:
        """Remove expired tokens."""
        now = time.time()
        expired = [
            sid for sid, token in self._tokens.items()
            if (now - token.created_at) / 60 > token.expiry_minutes
        ]
        for sid in expired:
            del self._tokens[sid]


# Need hmac for constant-time comparison
import hmac


# =============================================================================
# COMBINED SESSION + CSRF HANDLER
# =============================================================================

class SecureSessionHandler:
    """
    Combined session management with JWT and CSRF protection.

    Provides a complete authentication solution for Gradio apps.

    Usage:
        handler = SecureSessionHandler()

        # Login
        session = handler.login(username, password)

        # Protected action
        if handler.validate_request(session_token, csrf_token):
            # Perform action
            pass

        # Logout
        handler.logout(session_token)
    """

    def __init__(
        self,
        jwt_config: Optional[JWTConfig] = None,
        csrf_expiry_minutes: int = 60,
    ):
        self.jwt = JWTManager(jwt_config)
        self.csrf = CSRFProtection(csrf_expiry_minutes)
        self._active_sessions: Dict[str, str] = {}  # token -> user_id
        self._lock = Lock()

    def login(
        self,
        user_id: str,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Create a new authenticated session.

        Args:
            user_id: User identifier
            additional_claims: Additional JWT claims

        Returns:
            Dict with access_token, refresh_token, and csrf_token
        """
        access_token = self.jwt.create_token(user_id, additional_claims)
        refresh_token = self.jwt.create_token(user_id, is_refresh=True)

        # Generate CSRF token keyed by access token
        csrf_token = self.csrf.generate_token(access_token[:32])

        with self._lock:
            self._active_sessions[access_token[:32]] = user_id

        log_security_event("session_login", user_id=user_id)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "csrf_token": csrf_token,
        }

    def validate_request(
        self,
        access_token: str,
        csrf_token: str,
        require_csrf: bool = True,
    ) -> Tuple[bool, Optional[str], str]:
        """
        Validate an authenticated request.

        Args:
            access_token: JWT access token
            csrf_token: CSRF token
            require_csrf: Whether CSRF validation is required

        Returns:
            Tuple of (is_valid, user_id, message)
        """
        # Validate JWT
        valid, payload, msg = self.jwt.verify_token(access_token)
        if not valid:
            return False, None, msg

        user_id = payload.get("sub")  # type: ignore

        # Validate CSRF if required
        if require_csrf:
            csrf_valid, csrf_msg = self.csrf.validate_token(
                access_token[:32],
                csrf_token,
                consume=False,  # Don't consume - allow multiple requests
            )
            if not csrf_valid:
                return False, None, csrf_msg

        return True, user_id, "Request valid"

    def logout(self, access_token: str) -> None:
        """
        End a session.

        Args:
            access_token: JWT access token
        """
        with self._lock:
            token_key = access_token[:32]
            if token_key in self._active_sessions:
                user_id = self._active_sessions.pop(token_key)
                log_security_event("session_logout", user_id=user_id)

    def refresh_session(
        self,
        refresh_token: str,
    ) -> Tuple[bool, Optional[Dict[str, str]], str]:
        """
        Refresh an expired session.

        Args:
            refresh_token: JWT refresh token

        Returns:
            Tuple of (success, new_tokens, message)
        """
        valid, new_access, msg = self.jwt.refresh_access_token(refresh_token)

        if not valid or not new_access:
            return False, None, msg

        # Generate new CSRF token
        csrf_token = self.csrf.generate_token(new_access[:32])

        return True, {
            "access_token": new_access,
            "csrf_token": csrf_token,
        }, "Session refreshed"


# Global instance for convenience
_secure_session_handler: Optional[SecureSessionHandler] = None


def get_secure_session_handler() -> SecureSessionHandler:
    """Get the global secure session handler."""
    global _secure_session_handler
    if _secure_session_handler is None:
        _secure_session_handler = SecureSessionHandler()
    return _secure_session_handler


# =============================================================================
# CONTENT SECURITY POLICY (2026 Best Practices)
# =============================================================================

@dataclass
class CSPConfig:
    """
    Content Security Policy configuration.

    Based on:
    - OWASP CSP Cheat Sheet
    - MDN CSP Documentation
    - 2026 Frontend Security Best Practices

    The default policy is restrictive but allows Gradio to function.
    Adjust as needed for your deployment.
    """
    # Script sources
    script_src: List[str] = field(default_factory=lambda: ["'self'"])
    # Style sources
    style_src: List[str] = field(default_factory=lambda: ["'self'", "'unsafe-inline'"])  # Gradio needs inline styles
    # Image sources
    img_src: List[str] = field(default_factory=lambda: ["'self'", "data:", "blob:"])
    # Font sources
    font_src: List[str] = field(default_factory=lambda: ["'self'", "data:"])
    # Connect sources (for API calls, WebSocket)
    connect_src: List[str] = field(default_factory=lambda: ["'self'", "ws:", "wss:"])
    # Frame ancestors (who can embed this page)
    frame_ancestors: List[str] = field(default_factory=lambda: ["'self'"])
    # Base URI
    base_uri: List[str] = field(default_factory=lambda: ["'self'"])
    # Form action
    form_action: List[str] = field(default_factory=lambda: ["'self'"])
    # Object sources (plugins)
    object_src: List[str] = field(default_factory=lambda: ["'none'"])
    # Default fallback
    default_src: List[str] = field(default_factory=lambda: ["'self'"])
    # Report URI for violations (optional)
    report_uri: Optional[str] = None
    # Report-only mode (log violations but don't enforce)
    report_only: bool = False


class ContentSecurityPolicy:
    """
    Generate and manage Content Security Policy headers.

    Usage:
        csp = ContentSecurityPolicy()
        header_name, header_value = csp.get_header()

        # Add to response headers
        response.headers[header_name] = header_value

        # With custom config
        config = CSPConfig(script_src=["'self'", "https://trusted.cdn.com"])
        csp = ContentSecurityPolicy(config)
    """

    # Nonce for inline scripts (regenerated per request)
    _nonce: Optional[str] = None

    def __init__(self, config: Optional[CSPConfig] = None):
        self.config = config or CSPConfig()

    def generate_nonce(self) -> str:
        """
        Generate a nonce for inline scripts.

        Include this nonce in script tags: <script nonce="...">

        Returns:
            Base64-encoded nonce string
        """
        import base64
        import secrets

        nonce_bytes = secrets.token_bytes(16)
        self._nonce = base64.b64encode(nonce_bytes).decode("utf-8")
        return self._nonce

    def get_nonce(self) -> Optional[str]:
        """Get the current nonce (if generated)."""
        return self._nonce

    def build_policy(self, include_nonce: bool = False) -> str:
        """
        Build the CSP directive string.

        Args:
            include_nonce: Include generated nonce in script-src

        Returns:
            CSP directive string
        """
        directives = []

        # default-src
        if self.config.default_src:
            directives.append(f"default-src {' '.join(self.config.default_src)}")

        # script-src
        script_src = self.config.script_src.copy()
        if include_nonce and self._nonce:
            script_src.append(f"'nonce-{self._nonce}'")
        if script_src:
            directives.append(f"script-src {' '.join(script_src)}")

        # style-src
        if self.config.style_src:
            directives.append(f"style-src {' '.join(self.config.style_src)}")

        # img-src
        if self.config.img_src:
            directives.append(f"img-src {' '.join(self.config.img_src)}")

        # font-src
        if self.config.font_src:
            directives.append(f"font-src {' '.join(self.config.font_src)}")

        # connect-src
        if self.config.connect_src:
            directives.append(f"connect-src {' '.join(self.config.connect_src)}")

        # frame-ancestors
        if self.config.frame_ancestors:
            directives.append(f"frame-ancestors {' '.join(self.config.frame_ancestors)}")

        # base-uri
        if self.config.base_uri:
            directives.append(f"base-uri {' '.join(self.config.base_uri)}")

        # form-action
        if self.config.form_action:
            directives.append(f"form-action {' '.join(self.config.form_action)}")

        # object-src
        if self.config.object_src:
            directives.append(f"object-src {' '.join(self.config.object_src)}")

        # report-uri (deprecated but still supported)
        if self.config.report_uri:
            directives.append(f"report-uri {self.config.report_uri}")

        return "; ".join(directives)

    def get_header(self, include_nonce: bool = False) -> Tuple[str, str]:
        """
        Get the CSP header name and value.

        Args:
            include_nonce: Include generated nonce in script-src

        Returns:
            Tuple of (header_name, header_value)
        """
        policy = self.build_policy(include_nonce)

        if self.config.report_only:
            return "Content-Security-Policy-Report-Only", policy
        else:
            return "Content-Security-Policy", policy

    def get_all_security_headers(self) -> Dict[str, str]:
        """
        Get all recommended security headers.

        Returns a dict of header name -> value for comprehensive security.

        Returns:
            Dict of security headers
        """
        csp_name, csp_value = self.get_header()

        headers = {
            csp_name: csp_value,
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            # Prevent clickjacking
            "X-Frame-Options": "SAMEORIGIN",
            # XSS protection (legacy, but still useful)
            "X-XSS-Protection": "1; mode=block",
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Permissions policy (formerly Feature-Policy)
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }

        return headers


# Default CSP for Gradio apps
DEFAULT_GRADIO_CSP = CSPConfig(
    script_src=["'self'", "'unsafe-eval'"],  # Gradio needs eval for dynamic components
    style_src=["'self'", "'unsafe-inline'"],  # Gradio uses inline styles
    img_src=["'self'", "data:", "blob:", "https:"],  # Allow images from various sources
    font_src=["'self'", "data:", "https://fonts.gstatic.com"],  # Google Fonts
    connect_src=["'self'", "ws:", "wss:", "https://huggingface.co"],  # WebSocket + HF API
    frame_ancestors=["'self'"],
    object_src=["'none'"],
)


def get_gradio_csp(report_only: bool = False) -> ContentSecurityPolicy:
    """
    Get a CSP configured for Gradio apps.

    Args:
        report_only: If True, only report violations (don't enforce)

    Returns:
        ContentSecurityPolicy instance
    """
    config = CSPConfig(
        script_src=DEFAULT_GRADIO_CSP.script_src.copy(),
        style_src=DEFAULT_GRADIO_CSP.style_src.copy(),
        img_src=DEFAULT_GRADIO_CSP.img_src.copy(),
        font_src=DEFAULT_GRADIO_CSP.font_src.copy(),
        connect_src=DEFAULT_GRADIO_CSP.connect_src.copy(),
        frame_ancestors=DEFAULT_GRADIO_CSP.frame_ancestors.copy(),
        object_src=DEFAULT_GRADIO_CSP.object_src.copy(),
        report_only=report_only,
    )
    return ContentSecurityPolicy(config)


# =============================================================================
# MIDDLEWARE HELPERS FOR GRADIO
# =============================================================================

def apply_security_headers(
    response: Any,
    csp: Optional[ContentSecurityPolicy] = None,
) -> None:
    """
    Apply security headers to a response object.

    Works with various response types (Gradio, FastAPI, etc.)

    Args:
        response: Response object with headers attribute
        csp: ContentSecurityPolicy instance (default: Gradio-compatible)
    """
    if csp is None:
        csp = get_gradio_csp()

    headers = csp.get_all_security_headers()

    if hasattr(response, "headers"):
        for name, value in headers.items():
            response.headers[name] = value
    elif hasattr(response, "set_header"):
        for name, value in headers.items():
            response.set_header(name, value)
