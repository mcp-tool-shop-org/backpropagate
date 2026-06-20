"""
Backpropagate - UI Security Module
==================================

Production-hardened security utilities used by the Reflex web interface
(canonical from v1.1.0). The helpers were originally authored against Gradio
in v1.0; several still carry ``gradio`` in their names for back-compatibility
(``safe_gradio_handler``, ``DEFAULT_GRADIO_CSP``, etc.) and the underlying
behavior is framework-agnostic.

**v1.4 symbol rename (FRONTEND, Wave 6a foundation).** The Gradio-prefixed
public names were renamed to framework-agnostic canonical names. The legacy
names continue to resolve via module-level ``__getattr__`` and emit a
``DeprecationWarning`` pointing operators at the new symbol. The
deprecation cycle (advisor 2026-05-25 Q4, removal version revised
2026-06-20 to track the actual ship schedule):

* **v1.4 → present** — ``DeprecationWarning`` on legacy name access.
* **future release (v1.7 or later)** — legacy names removed; access
  raises ``AttributeError``.

| Legacy (v1.0 Gradio era) | Canonical (v1.4+) |
|---|---|
| ``safe_gradio_handler`` | ``safe_ui_handler`` |
| ``raise_gradio_error`` | ``raise_ui_error`` |
| ``raise_gradio_warning`` | ``raise_ui_warning`` |
| ``raise_gradio_info`` | ``raise_ui_info`` |
| ``RequestContext.from_gradio_request`` | ``RequestContext.from_request`` |
| ``DEFAULT_GRADIO_CSP`` | ``DEFAULT_REFLEX_CSP`` (see FRONTEND-A-003 Wave 2) |
| ``get_gradio_csp`` | ``get_reflex_csp`` (see FRONTEND-A-003 Wave 2) |

See ``site/src/content/docs/handbook/migrations.md`` for the full operator
migration narrative.

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
        raise_ui_error,
        get_health_status,
        RequestContext,
    )
"""

import json
import logging
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Optional, TypeVar
from urllib.parse import urlparse

# Gradio import is conditional. The Web UI migrated from Gradio to Reflex in
# v1.1.0; many helpers in this module are framework-agnostic (rate limiter,
# file validator, auth-shape, error sanitization, path/IP extraction) and
# should stay importable when neither Gradio nor Reflex is installed.
# Gradio-specific surfaces (gr.Error wrappers, gr.Request type hints) check
# GRADIO_AVAILABLE before use; type-hint references to ``gr.Request`` resolve
# to the lightweight stub below when Gradio is absent so the module body still
# evaluates without ImportError under [ui] = reflex-only.
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised when [ui] uses reflex only
    GRADIO_AVAILABLE = False

    class _GradioShim:
        """Stand-in for the ``gradio`` module when it isn't installed.

        Provides a ``Request`` placeholder type for annotations and an
        ``Error`` subclass that mirrors Gradio's user-facing exception so
        ``raise gr.Error(...)`` keeps working in unit tests / mocks. The
        ``Warning`` / ``Info`` shims are no-ops.

        **Transitional during the v1.4 → v1.6 ui_security rename
        deprecation cycle.** Wave 6b features attempted deletion of this
        class; the cascade survey found 44 ``gr.*`` reference sites in
        this module (18 ``gr.Request`` type hints + 24 ``gr.Error``
        raises/handlers + 1 ``gr.Warning`` + 1 ``gr.Info``), 13 of
        which sit inside the legacy ``safe_ui_handler`` decorator's
        ``except gr.Error:`` block that the v1.4 → v1.6 deprecation
        cycle keeps alive for real-Gradio consumers. Replacing the
        shim cleanly requires either (a) wrapping every ``gr.*`` call
        site behind ``if GRADIO_AVAILABLE:`` (not a mechanical rename —
        each call site has different fallback semantics), or (b)
        splitting this module into ``ui_security_core.py`` (framework-
        agnostic helpers the Reflex UI uses) and
        ``ui_security_legacy.py`` (Gradio-era classes preserved for
        backward-compat). Both paths are v1.5 candidates; see
        WAVE_6A_TODO.md → "FRONTEND symbol-rename gating → STILL
        DEFERRED (later wave / v1.5 candidate)". The shim's continued
        presence is NOT dead code — it's load-bearing whenever Gradio
        isn't installed (the [ui] extra ships Reflex only).
        """

        class Request:  # type: ignore[no-redef]
            pass

        class Error(Exception):  # type: ignore[no-redef]
            def __init__(self, message: str, *, duration: int = 10, title: str = "Error") -> None:
                super().__init__(message)
                self.duration = duration
                self.title = title

        @staticmethod
        def Warning(message: str, *, duration: int = 5, title: str = "Warning") -> None:  # noqa: N802, ARG004
            _ = (message, duration, title)
            return None

        @staticmethod
        def Info(message: str, *, duration: int = 5, title: str = "Info") -> None:  # noqa: N802, ARG004
            _ = (message, duration, title)
            return None

    gr = _GradioShim()  # type: ignore[assignment]

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
    # UI error helpers — canonical v1.4+ names (framework-agnostic).
    # Legacy Gradio-prefixed names continue to resolve via module-level
    # __getattr__ + emit DeprecationWarning (v1.4 → present) → AttributeError
    # (future release, v1.7 or later). See module docstring for the rename
    # table.
    "raise_ui_error",
    "raise_ui_warning",
    "raise_ui_info",
    "safe_ui_handler",
    # Request validation
    "validate_and_log_request",
    "sanitize_filename",
    "validate_numeric_input",
    "validate_string_input",
    # UI output directory (F-002)
    "get_ui_output_dir",
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
    "DEFAULT_GRADIO_CSP",  # deprecated in v1.4 — use DEFAULT_REFLEX_CSP for new wiring
    "DEFAULT_REFLEX_CSP",  # FRONTEND-A-003 (v1.4 Wave 2): Reflex/Next.js-tuned default
    "get_gradio_csp",  # deprecated in v1.4 — use get_reflex_csp for new wiring
    "get_reflex_csp",  # FRONTEND-A-003 (v1.4 Wave 2): factory used by security_headers_middleware
    "apply_security_headers",
    "security_headers_dict",  # FRONTEND-A-003 (v1.4 Wave 2): ASGI-shape helper
    # Framework-agnostic helpers (moved from ui.py during Reflex migration v1.1.0)
    "safe_markdown_fence",
    "sanitize_error_for_user",
    "validate_auth_shape",
]

logger = logging.getLogger(__name__)
security_logger = logging.getLogger("backpropagate.security.ui")


# =============================================================================
# CLIENT IP EXTRACTION (F-001 fix)
# =============================================================================

def _extract_client_ip(request: gr.Request | None) -> str:
    """
    Robustly extract a client IP from a Gradio/Starlette request.

    Handles all shapes that ``request.client`` can take in practice:

    - ``Address`` namedtuple (Starlette/FastAPI's default — what Gradio passes
      in production): exposes a ``host`` attribute (and a separate ``port``).
      We deliberately use ONLY ``host`` so per-IP rate-limit buckets don't
      degenerate into per-TCP-connection buckets (different source port per
      connection = effectively no rate limiting).
    - ``None`` (e.g. unit tests, some ASGI middleware): we return
      ``"unknown"`` so rate limiting still applies as a single shared bucket
      rather than failing open with a unique id per call.
    - ``dict`` with a ``"host"`` key: preserved for legacy/test callers that
      construct a plain dict.
    - bare ``str``: treated as the host directly. Note: we do NOT honor
      ``X-Forwarded-For`` headers by default; trusting them requires a
      separate trusted-proxy configuration and is intentionally out of scope.

    Default on unknown shape: ``"unknown"`` (fail-closed under a shared
    bucket rather than fail-open with per-connection buckets).
    """
    if request is None:
        return "unknown"

    client = getattr(request, "client", None)
    if client is None:
        return "unknown"

    # 1. Address-like namedtuple (the real Starlette/FastAPI shape).
    host = getattr(client, "host", None)
    if isinstance(host, str) and host:
        return host

    # 2. Dict (legacy/test shape).
    if isinstance(client, dict):
        dict_host = client.get("host")
        if isinstance(dict_host, str) and dict_host:
            return dict_host
        return "unknown"

    # 3. Bare string fallback.
    if isinstance(client, str) and client:
        return client

    # 4. Unknown shape — single shared bucket.
    return "unknown"


# =============================================================================
# UI OUTPUT DIRECTORY (F-002 fix)
# =============================================================================

# Categories of system / credential paths an operator should never accidentally
# point ``BACKPROPAGATE_UI__OUTPUT_DIR`` at. We err on the side of "obviously
# dangerous" rather than enumerating every plausible-but-unusual path. Each
# entry is expanded with ``~`` resolution and ``Path.resolve()`` before being
# compared with ``Path.is_relative_to``.
#
# Categories (used by the hint surfaced on rejection):
#   - System root + system bin/lib trees (Linux/macOS)
#   - Boot / kernel / device pseudo-filesystems (Linux)
#   - Per-host runtime/state trees (``/var/run``, ``/var/lib``)
#   - Root user's home (``/root``)
#   - User credential directories under any home (``~/.ssh``, ``~/.aws``,
#     ``~/.kube``, ``~/.docker``, ``~/.gnupg``, ``~/.config``)
#   - Windows system + program-files trees + ``ProgramData``
#   - Windows per-user credential dirs (``%USERPROFILE%\.ssh``, AppData crypto)
_FORBIDDEN_OUTPUT_BASE_CATEGORIES = (
    "system trees (/etc, /usr, /bin, /sbin, /boot, /sys, /proc, /dev, /var/run, /var/lib)",
    "root home (/root)",
    "user credential dirs (~/.ssh, ~/.aws, ~/.kube, ~/.docker, ~/.gnupg, ~/.config)",
    "Windows system roots (C:\\Windows, C:\\Program Files, C:\\Program Files (x86), C:\\ProgramData)",
    "Windows per-user credential dirs (%USERPROFILE%\\.ssh, %APPDATA%\\Microsoft\\Crypto)",
)


def _forbidden_output_bases() -> list[Path]:
    """
    Build the resolved list of forbidden output bases for the current host.

    Resolved at call time (not module import) so test code can monkeypatch
    ``Path.home`` / ``os.environ`` and see the result reflected. Paths that
    don't exist on this host are still included — ``Path.is_relative_to``
    works on lexical comparison after ``resolve()``, so a Linux denylist
    entry on a Windows host is harmless (just never matches).
    """
    # Exact path roots ("/" and "C:\\") are intentionally NOT in this list:
    # `is_relative_to` against the root matches every absolute path on the
    # platform, which would also reject legitimate non-home bases like /tmp,
    # /opt, /srv, C:\Temp, etc. The enumerated system trees below cover the
    # actual dangerous surface — anything outside both home AND these trees
    # is the operator's choice to make.
    #
    # Note: bare "/var" is intentionally absent. On macOS, /var resolves to
    # /private/var (symlink), and the system per-user temp tree lives at
    # /var/folders/<hash>/T/... — which is where pytest tmp_path, NSTemporaryDirectory,
    # and most legitimate macOS tooling write. Blanket-denying /var blocks
    # those. The dangerous subtrees we actually care about (/var/run for
    # PID files and sockets, /var/lib for service state) are listed
    # individually.
    raw: list[str] = [
        "/etc",
        "/proc",
        "/sys",
        "/dev",
        "/boot",
        "/var/run",
        "/var/lib",
        "/usr",
        "/bin",
        "/sbin",
        "/root",
        "C:\\Windows",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        "C:\\ProgramData",
    ]
    # Per-user credential directories — expand to the current user's home.
    home = Path.home()
    home_credential_subdirs = (
        ".ssh",
        ".aws",
        ".kube",
        ".docker",
        ".gnupg",
        ".config",
    )
    for sub in home_credential_subdirs:
        raw.append(str(home / sub))

    # Windows-specific per-user credential roots.
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        raw.append(str(Path(appdata) / "Microsoft" / "Crypto"))

    resolved: list[Path] = []
    seen: set[str] = set()
    for entry in raw:
        try:
            p = Path(entry).expanduser().resolve()
        except (OSError, RuntimeError):
            # Resolving a non-existent Windows drive on Linux (or vice versa)
            # can raise; that just means this entry can't apply here.
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(p)
    return resolved


def _is_forbidden_output_base(path: Path) -> bool:
    """
    Return True if ``path`` resolves to (or under) any denylisted category.

    The check uses ``Path.is_relative_to`` (Python 3.9+) for ancestor
    matching, plus an exact-equality check so that, e.g., ``Path('/etc')``
    itself is rejected. ``path`` is expected to already be resolved by the
    caller; we resolve defensively in case it isn't.

    Liberal-by-design: when the candidate is strictly inside the current
    user's home directory, system-tree rejections (``/var``, ``/usr``,
    etc.) are skipped — a service account whose home happens to be
    ``/var/lib/jenkins`` should still be allowed to use the default
    ``~/.backpropagate/ui-outputs``. Credential-dir matches (``~/.ssh``,
    ``%APPDATA%\\Microsoft\\Crypto``, etc.) are still enforced even
    inside the home, because those are the actual foot-guns we care
    about.

    UI-A-007 (Wave A2) symlink hardening: the credential-dir denylist is
    enforced against the *resolved* candidate, so a symlink whose target is a
    credential dir is caught. But the "strictly inside home → allow" fast path
    (below) returns before the system-tree loop runs — so a symlink placed
    *under home* that points at a non-credential location outside home could
    previously slip through by masking its true target behind ``.resolve()``.
    We now reject the candidate if any path component STRICTLY BELOW the home
    root (i.e. the user-controlled tail, not the home prefix itself) is a
    symlink. A legitimate UI output dir under home never needs to traverse a
    symlink; refusing one closes the "symlink under $HOME hides its target"
    class without breaking setups where home itself (or a system mount like
    macOS ``/var`` → ``/private/var``) is the symlink. Operators who
    deliberately want a symlinked output dir should point the env var at the
    link's target directly (the env-var-must-not-traverse-symlinks-below-home
    constraint).
    """
    try:
        candidate = path.expanduser().resolve()
    except (OSError, RuntimeError):
        # If we can't even resolve it, treat as forbidden (fail-closed).
        return True

    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        home = None

    # UI-A-007: reject a symlink in the user-controlled tail below home.
    # Walk the PRE-resolved path's ancestry from the leaf up; if any element
    # that lives strictly below the home root is a symlink, treat the base as
    # forbidden. We use the un-resolved expansion so the symlink itself is
    # observable (``.resolve()`` would have already collapsed it).
    if home is not None:
        try:
            pre = path.expanduser()
        except (OSError, RuntimeError):
            pre = None
        if pre is not None:
            for ancestor in (pre, *pre.parents):
                # Boundary check on the UNRESOLVED ancestor — resolving here
                # would collapse a symlink to its (out-of-home) target and hide
                # the very component we must catch. Stop once we climb to or
                # above the home root (the home prefix and anything above it is
                # out of the operator's per-output-dir control and may
                # legitimately be a symlink, e.g. macOS /var -> /private/var).
                try:
                    below_home = ancestor.is_relative_to(home) and ancestor != home
                except (ValueError, OSError):
                    below_home = False
                if not below_home:
                    break
                try:
                    if ancestor.is_symlink():
                        log_security_event(
                            "ui_output_dir_symlink_below_home",
                            requested=str(pre),
                            symlink_component=str(ancestor),
                        )
                        return True
                except OSError:
                    # If we can't stat it, fail closed for this component.
                    return True

    # Set of "credential" forbidden roots, computed identically here to the
    # main list so we can selectively enforce them inside the home directory.
    credential_subdirs = (".ssh", ".aws", ".kube", ".docker", ".gnupg", ".config")
    credential_roots: set[Path] = set()
    if home is not None:
        for sub in credential_subdirs:
            try:
                credential_roots.add((home / sub).resolve())
            except (OSError, RuntimeError):
                continue
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        try:
            credential_roots.add((Path(appdata) / "Microsoft" / "Crypto").resolve())
        except (OSError, RuntimeError):
            pass

    # If the candidate is strictly under the home directory and is NOT under
    # one of the credential roots, allow it — covers /var/lib/jenkins/.* and
    # similar service-account homes that happen to live under a system root.
    if home is not None:
        try:
            if candidate.is_relative_to(home) and candidate != home:
                for cred in credential_roots:
                    try:
                        if candidate == cred or candidate.is_relative_to(cred):
                            return True
                    except (ValueError, OSError):
                        continue
                return False
        except (ValueError, OSError):
            pass

    for forbidden in _forbidden_output_bases():
        try:
            if candidate == forbidden or candidate.is_relative_to(forbidden):
                return True
        except (ValueError, OSError):
            # Cross-drive comparison on Windows can raise ValueError; that
            # just means this entry can't match.
            continue
    return False


def get_ui_output_dir() -> Path:
    """
    Return the single allowed-base directory for ALL UI-initiated filesystem
    writes (saved LoRA adapters, GGUF exports, converted datasets, Modelfiles,
    Ollama registrations, etc.).

    Resolution order:

    1. ``BACKPROPAGATE_UI__OUTPUT_DIR`` env var (operator override) — if set,
       it is resolved to an absolute path and created on first use.
    2. ``~/.backpropagate/ui-outputs`` (default) — created on first use.

    Why this exists: ``validate_path_input`` previously called the underlying
    ``safe_path`` without ``allowed_base``, which meant any absolute or
    relative path was accepted (``safe_path`` only logged a warning for
    literal ``..`` substrings without a base). Under the documented
    ``--share + --auth`` flow that exposed filesystem-wide write access to
    any authenticated remote user. Every UI sink now passes this directory
    as ``allowed_base`` so user-supplied paths must resolve inside it.

    System-path guard (FB-003):
        After resolving the override (or the default), the path is checked
        against a small denylist of obviously dangerous bases — system
        roots, credential directories, etc. — and rejected with a
        ``BackpropagateError(code="UI_OUTPUT_DIR_FORBIDDEN")`` if it
        matches. The denylist is intentionally conservative (system trees
        and well-known credential directories only); operators with niche
        legitimate use cases should pick a non-system directory and the
        guard will pass.

    The directory is created (with parents) on first call so the path is
    immediately writable. The directory's permissions are NOT chmod'd —
    that's the operator's responsibility.
    """
    override = os.environ.get("BACKPROPAGATE_UI__OUTPUT_DIR", "").strip()
    if override:
        base = Path(override).expanduser().resolve()
    else:
        base = (Path.home() / ".backpropagate" / "ui-outputs").resolve()

    # FB-003: refuse to mkdir into a system / credential directory.
    if _is_forbidden_output_base(base):
        from .exceptions import BackpropagateError

        categories_hint = "; ".join(_FORBIDDEN_OUTPUT_BASE_CATEGORIES)
        log_security_event(
            "ui_output_dir_forbidden",
            requested=str(base),
            source="env" if override else "default",
        )
        raise BackpropagateError(
            message=(
                f"BACKPROPAGATE_UI__OUTPUT_DIR resolves to a forbidden system "
                f"or credential path: {base}"
            ),
            code="UI_OUTPUT_DIR_FORBIDDEN",
            suggestion=(
                "Pick a non-system directory under your home (e.g. "
                "~/.backpropagate/ui-outputs or ~/work/backprop-out). The "
                f"following categories are refused: {categories_hint}."
            ),
            details={
                "requested": str(base),
                "source": "env" if override else "default",
            },
        )

    base.mkdir(parents=True, exist_ok=True)
    return base


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
    allowed_dataset_extensions: set[str] = field(default_factory=lambda: {
        ".jsonl", ".json", ".csv", ".txt", ".parquet"
    })
    blocked_extensions: set[str] = field(default_factory=lambda: {
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


def load_config_from_env(base_config: SecurityConfig | None = None) -> SecurityConfig:
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
        self._max_clients = 10000

        # Track requests per IP
        self._requests: dict[str, list[float]] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Cleanup every 5 minutes
        self._lock = Lock()

    def _get_client_id(self, request: gr.Request | None = None) -> str:
        """Get client identifier (IP or fallback to 'unknown')."""
        return _extract_client_ip(request)

    def _cleanup_old_entries(self) -> None:
        """Remove expired entries to prevent memory growth."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        with self._lock:
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

    def _enforce_client_cap(self) -> None:
        """Hard cap on tracked client IDs to prevent unbounded memory growth."""
        if len(self._requests) <= self._max_clients:
            return

        # Force immediate cleanup regardless of interval
        now = time.time()
        self._last_cleanup = now
        cutoff = now - self.window_seconds
        for client_id in list(self._requests.keys()):
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > cutoff
            ]
            if not self._requests[client_id]:
                del self._requests[client_id]

        # If still over cap, evict oldest entries
        if len(self._requests) > self._max_clients:
            # Sort by most recent request timestamp (oldest first)
            by_age = sorted(
                self._requests.items(),
                key=lambda item: max(item[1]) if item[1] else 0.0,
            )
            excess = len(self._requests) - self._max_clients
            for client_id, _ in by_age[:excess]:
                del self._requests[client_id]

    def check(self, request: gr.Request | None = None) -> tuple[bool, float]:
        """
        Check if request is allowed.

        Returns:
            Tuple of (is_allowed, wait_seconds_if_denied)
        """
        self._cleanup_old_entries()
        self._enforce_client_cap()

        client_id = self._get_client_id(request)
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
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

    def is_allowed(self, request: gr.Request | None = None) -> bool:
        """Simple check returning just bool."""
        allowed, _ = self.check(request)
        return allowed

    def require(self, request: gr.Request | None = None) -> None:
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
        allowed_extensions: set[str] | None = None,
        max_size_mb: int = 500,
        config: SecurityConfig | None = None,
    ):
        self.config = config or DEFAULT_SECURITY_CONFIG
        self.allowed_extensions = allowed_extensions or self.config.allowed_dataset_extensions
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def validate(
        self,
        file_obj: Any,
        purpose: str = "upload",
    ) -> tuple[bool, str, Path | None]:
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

        # FB-010: Magic-bytes validation — wire the config flag end-to-end.
        # When BACKPROPAGATE_SECURITY__VALIDATE_FILE_MAGIC=true (or the
        # SecurityConfig is constructed with validate_file_magic=True), every
        # accepted-extension upload is also content-sniffed. This catches the
        # "rename .html → .jsonl" extension-spoof case Trail of Bits flagged
        # against Gradio 5 (CVE-2024-47872 family). For extensions with no
        # canonical signature (.csv/.txt/.safetensors), validate_file_magic
        # still runs the suspicious-starts heuristic (HTML / shell / PHP),
        # so even no-signature types get a minimum defense.
        if self.config.validate_file_magic and file_path.exists():
            is_ok, magic_msg = validate_file_magic(file_path, expected_extension=ext)
            if not is_ok:
                log_security_event(
                    "file_magic_rejected",
                    file_name=file_path.name,
                    extension=ext,
                    purpose=purpose,
                    reason=magic_msg,
                )
                return (
                    False,
                    (
                        f"File content does not match {ext} (magic-bytes check failed): "
                        f"{magic_msg}"
                    ),
                    None,
                )

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
# UI ERROR HELPERS (framework-agnostic; v1.4 rename of gradio_* names)
# =============================================================================
#
# Authored against Gradio in v1.0; v1.1.0 migrated the canonical UI to Reflex
# but the helpers stayed framework-agnostic — they wrap whatever ``gr.*`` the
# module's import-time fallback resolved to (real Gradio when installed,
# ``_GradioShim`` otherwise). The v1.4 rename drops the ``gradio_`` prefix
# from the public names while preserving the legacy names as deprecation
# aliases (see module-level ``__getattr__`` near the file footer).

def raise_ui_error(
    message: str,
    duration: int | None = 10,
    title: str = "Error",
    log: bool = True,
) -> None:
    """
    Raise a UI error with proper formatting.

    Use this instead of returning error strings for better UX. The
    function name lost the ``gradio_`` prefix in v1.4; the
    ``raise_gradio_error`` legacy alias keeps working with a
    ``DeprecationWarning`` until v1.6.

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


def raise_ui_warning(
    message: str,
    duration: int | None = 5,
    title: str = "Warning",
    log: bool = True,
) -> None:
    """
    Show a UI warning (non-blocking).

    Unlike gr.Error, this does NOT halt execution. The function name lost
    the ``gradio_`` prefix in v1.4; the ``raise_gradio_warning`` legacy
    alias keeps working with a ``DeprecationWarning`` until v1.6.

    Args:
        message: Warning message
        duration: Seconds to show
        title: Warning dialog title
        log: Whether to log
    """
    if log:
        logger.warning(f"UI Warning: {message}")

    gr.Warning(message, duration=duration, title=title)


def raise_ui_info(
    message: str,
    duration: int | None = 3,
    title: str = "Info",
) -> None:
    """
    Show a UI info message (non-blocking).

    The function name lost the ``gradio_`` prefix in v1.4; the
    ``raise_gradio_info`` legacy alias keeps working with a
    ``DeprecationWarning`` until v1.6.

    Args:
        message: Info message
        duration: Seconds to show
        title: Info dialog title
    """
    gr.Info(message, duration=duration, title=title)


F = TypeVar('F', bound=Callable[..., Any])


def safe_ui_handler(
    operation_name: str = "operation",
    rate_limiter: EnhancedRateLimiter | None = None,
    log_errors: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to wrap UI handlers with security features.

    Features:
    - Converts exceptions to gr.Error for proper UI display
    - Optional rate limiting
    - Security event logging
    - Request validation

    The decorator was renamed from ``safe_gradio_handler`` in v1.4
    (Reflex is canonical from v1.1.0 — the Gradio prefix no longer
    matched the framework). The ``safe_gradio_handler`` legacy alias
    keeps working with a ``DeprecationWarning`` until v1.6.

    Usage:
        @safe_ui_handler("training", rate_limiter=training_limiter)
        def start_training(...):
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
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
                    f"An unexpected error occurred during {operation_name} ({type(e).__name__}). Check the terminal/logs for full details.",
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
    min_value: float | None = None,
    max_value: float | None = None,
    allow_none: bool = False,
) -> float | None:
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
    pattern: str | None = None,
    allow_none: bool = False,
    allow_empty: bool = False,
) -> str | None:
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
    request: gr.Request | None = None,
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

    client_id = _extract_client_ip(request)

    # Sanitize params for logging (truncate long values)
    safe_params: dict[str, str | int | float | bool] = {}
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
    _logger: logging.Logger

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
    request: gr.Request | None = None,
    config: SecurityConfig | None = None,
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
        localhost_hostnames = {"localhost", "127.0.0.1", "::1"}

        def _is_localhost(url: str) -> bool:
            """Check if a URL's hostname is exactly a localhost address."""
            if not url:
                return True  # Empty origin/referer is allowed
            try:
                parsed = urlparse(url)
                hostname = parsed.hostname or ""
                return hostname in localhost_hostnames
            except Exception:
                return False

        origin_ok = _is_localhost(origin)
        referer_ok = _is_localhost(referer)

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
    gpu_name: str | None = None
    gpu_memory_used_gb: float | None = None
    gpu_memory_total_gb: float | None = None
    gpu_temperature_c: float | None = None
    active_trainings: int = 0
    active_sessions: int = 0
    rate_limit_status: str = "ok"
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
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
    config: SecurityConfig | None = None,
    include_gpu: bool | None = None,
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
                status.gpu_name = gpu_status.device_name
                status.gpu_memory_used_gb = gpu_status.vram_used_gb
                status.gpu_memory_total_gb = gpu_status.vram_total_gb
                status.gpu_temperature_c = gpu_status.temperature_c

                # Check for GPU issues
                if gpu_status.temperature_c and gpu_status.temperature_c > 85:
                    status.status = "degraded"
        except Exception as e:
            status.status = "degraded"
            logger.error("GPU health check failed, marking status degraded: %s", e)

    return status


# =============================================================================
# AUTH BADGE CONTEXT (FRONTEND-F-FOOTER-AUTH-BADGE, Stage C humanization)
# =============================================================================
#
# Read the same env-var surface that ``ui_app/auth.py::basic_auth_transformer``
# consumes and produce a small, JSON-serializable description of the current
# operator-facing posture. The footer auth badge in the Reflex UI reads this
# at server boot and shows a colored chip with hover-text so the operator can
# tell at a glance whether the instance is loopback-only / shared / network
# and whether auth is wired.
#
# Load-bearing invariant: this function is READ-ONLY w.r.t. the auth decision.
# It introspects env vars; it does NOT participate in middleware enforcement.
# All four GHSA-f65r-h4g3-3h9h contracts (pre-accept WS cookie validation,
# 4-mode resolution, constant-time compares, Host/Origin allowlists) remain
# the exclusive responsibility of ``ui_app/auth.py``.


@dataclass
class AuthBadgeContext:
    """Operator-facing auth posture summary for the footer badge.

    Fields:

    - ``mode_key``: machine identifier (``no_auth_local`` / ``token_local`` /
      ``basic_local`` / ``basic_shared`` / ``basic_network`` / ``insecure``).
      Stable across launches so the badge component can match on it.
    - ``mode_color``: ``green`` / ``amber`` / ``red`` — drives the chip tint.
    - ``mode_text``: short human label that appears in the chip (with the
      leading emoji so it's still recognizable in a screen-reader). Limited
      to ~24 chars so the chip fits the 32px footer height comfortably.
    - ``hover_text``: full operator context ("bound to <host>:<port>;
      reachable on <reach>; <auth>") — surfaces as both the tooltip and the
      ``aria-label`` so vision-impaired operators get the same info.
    - ``bind_host`` / ``bind_port``: resolved bind address (defaults to
      ``localhost`` / ``7860`` when env vars are unset). Mirrored to the
      hover-text but exposed as separate fields in case future surfaces want
      them.
    - ``reachable_from``: short description of who can reach this UI
      (``loopback-only`` / ``LAN`` / ``public network``).
    - ``auth_user``: the resolved username for Basic-auth modes (empty
      string in token / no-auth modes). Read from the same
      ``BACKPROPAGATE_UI_AUTH`` env var the middleware consumes, with the
      password half DROPPED — only the username is suitable for surfacing.
    """

    mode_key: str
    mode_color: str
    mode_text: str
    hover_text: str
    bind_host: str
    bind_port: str
    reachable_from: str
    auth_user: str


def _classify_reach(bind_host: str, share_host: str) -> str:
    """Bucket the bind into loopback / LAN / public for the hover text."""
    if share_host:
        return "public network"
    if not bind_host or bind_host in ("localhost", "127.0.0.1", "::1"):
        return "loopback-only"
    if bind_host in ("0.0.0.0", "::"):
        return "any local interface (LAN)"
    return "LAN"


def get_auth_badge_context(env: dict[str, str] | None = None) -> AuthBadgeContext:
    """Compute the footer auth-badge context from the current env.

    The mode-resolution logic mirrors ``ui_app/auth.py::_detect_mode`` so the
    badge cannot disagree with the middleware. We re-derive (rather than
    import) because ``ui_security`` predates ``ui_app`` in the dep DAG —
    importing the auth module from here would create a circular dependency
    and pull Reflex into modules that should stay framework-agnostic.

    Args:
        env: Optional env-var dict (mostly for testing); defaults to
            ``os.environ`` snapshot at call time.

    Returns:
        An ``AuthBadgeContext`` with the 6 fields the badge component reads.
    """
    env = env if env is not None else dict(os.environ)
    auth_creds = env.get("BACKPROPAGATE_UI_AUTH", "").strip()
    share_host = env.get("BACKPROPAGATE_UI_SHARE_HOST", "").strip()
    host_bind = env.get("BACKPROPAGATE_UI_HOST_BIND", "").strip().lower()
    launch_token = env.get("BACKPROPAGATE_UI_LAUNCH_TOKEN", "").strip()
    port = env.get("BACKPROPAGATE_UI_PORT", "").strip() or "7860"

    # Resolve effective bind host for display. If --host wasn't passed and
    # we're not in share mode, we report loopback ("localhost") — that's what
    # the operator sees in the browser bar.
    if share_host:
        display_host = share_host
    elif host_bind and host_bind not in ("", "localhost", "127.0.0.1", "::1"):
        display_host = host_bind
    else:
        display_host = "localhost"

    reach = _classify_reach(host_bind, share_host)

    # Extract the username half of BACKPROPAGATE_UI_AUTH (NEVER the password).
    auth_user = ""
    if auth_creds and ":" in auth_creds:
        auth_user = auth_creds.split(":", 1)[0]

    # Map (auth_creds, share_host, host_bind, launch_token) → posture.
    # Order mirrors AuthMode resolution; the badge layers on the LAN /
    # loopback distinction since the middleware folds both into PRODUCTION.
    if auth_creds:
        if share_host:
            mode_key = "basic_shared"
            mode_color = "amber"
            mode_text = "Shared - Basic"
            hover_text = (
                f"bound to {display_host}:{port}; reachable on public network; "
                f"HTTP Basic auth (user '{auth_user}')"
            )
        elif host_bind and host_bind not in ("", "localhost", "127.0.0.1", "::1"):
            mode_key = "basic_network"
            mode_color = "amber"
            mode_text = "Network - Basic"
            hover_text = (
                f"bound to {display_host}:{port}; reachable on LAN; "
                f"HTTP Basic auth (user '{auth_user}')"
            )
        else:
            mode_key = "basic_local"
            mode_color = "green"
            mode_text = "Local - Basic"
            hover_text = (
                f"bound to {display_host}:{port}; loopback-only; "
                f"HTTP Basic auth (user '{auth_user}')"
            )
    elif launch_token:
        mode_key = "token_local"
        mode_color = "green"
        mode_text = "Local - token"
        hover_text = (
            f"bound to {display_host}:{port}; loopback-only; "
            "URL-token authentication"
        )
    elif share_host or (host_bind and host_bind not in (
        "", "localhost", "127.0.0.1", "::1"
    )):
        # Non-loopback bind with no auth — this state should be rejected by
        # cli.py's refuse-to-start rails (FRONTEND-B-001 / GHSA-f65r-h4g3-3h9h),
        # but if the operator bypassed via direct ``reflex run``, the badge
        # screams red so the misconfiguration is visible.
        mode_key = "insecure"
        mode_color = "red"
        mode_text = "INSECURE - no auth"
        hover_text = (
            f"bound to {display_host}:{port}; reachable on {reach}; "
            "NO AUTH - anyone with the URL has full access"
        )
    else:
        mode_key = "no_auth_local"
        mode_color = "green"
        mode_text = "Local - no auth"
        hover_text = (
            f"bound to {display_host}:{port}; loopback-only; "
            "no authentication"
        )

    return AuthBadgeContext(
        mode_key=mode_key,
        mode_color=mode_color,
        mode_text=mode_text,
        hover_text=hover_text,
        bind_host=display_host,
        bind_port=port,
        reachable_from=reach,
        auth_user=auth_user,
    )


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
    operation: str | None = None
    user_id: str | None = None

    @classmethod
    def from_request(
        cls,
        request: gr.Request | None = None,
        operation: str | None = None,
    ) -> "RequestContext":
        """Create context from a UI request.

        Renamed from ``from_gradio_request`` in v1.4 — the helper is
        framework-agnostic (the ``gr.Request`` type hint is satisfied by
        any object with a ``client`` attribute, which Starlette / FastAPI
        / Reflex requests all provide). The legacy
        ``RequestContext.from_gradio_request`` keeps working with a
        ``DeprecationWarning`` until v1.6.
        """
        request_id = str(uuid.uuid4())[:8]
        client_ip = _extract_client_ip(request)

        return cls(
            request_id=request_id,
            client_ip=client_ip,
            timestamp=time.time(),
            operation=operation,
        )

    # Legacy alias for the v1.0-era ``from_gradio_request`` classmethod.
    # The classmethod's __getattr__ contract is preserved on the dataclass
    # itself — ``RequestContext.from_gradio_request(...)`` still resolves
    # but emits a DeprecationWarning. We can't use module-level __getattr__
    # for class attributes, so the deprecation shim lives here as a
    # classmethod that warns + delegates.
    @classmethod
    def from_gradio_request(
        cls,
        request: gr.Request | None = None,
        operation: str | None = None,
    ) -> "RequestContext":
        """DEPRECATED v1.4 alias for :meth:`from_request`.

        Emits ``DeprecationWarning`` and delegates to the canonical
        ``from_request`` classmethod. The legacy name will be removed in a
        future release (v1.7 or later) per the rename cycle (advisor
        2026-05-25 Q4).
        """
        import warnings as _warnings

        _warnings.warn(
            "'RequestContext.from_gradio_request' is deprecated in v1.4; "
            "use 'RequestContext.from_request' instead. The legacy name will "
            "be removed in a future release (v1.7 or later).",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_request(request=request, operation=operation)

    def to_log_dict(self) -> dict[str, Any]:
        """Get dictionary for logging extra fields."""
        return {
            "request_id": self.request_id,
            "client_ip": self.client_ip,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "user_id": self.user_id,
        }


def get_request_id(request: gr.Request | None = None) -> str:
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
    logger_names: list[str] | None = None,
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
    user_id: str | None = None


class SessionManager:
    """
    Manages user sessions with timeout and limits.

    Thread-safe session tracking for the web UI.
    """

    _instance: Optional["SessionManager"] = None
    _class_lock: Lock = Lock()  # Only used for singleton construction
    _lock: Lock
    _sessions: dict[str, SessionInfo]
    _sessions_by_ip: dict[str, list[str]]

    def __new__(cls) -> "SessionManager":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._lock = Lock()
                    cls._instance._sessions = {}
                    cls._instance._sessions_by_ip = {}
        return cls._instance

    def create_session(
        self,
        client_ip: str,
        user_id: str | None = None,
        config: SecurityConfig | None = None,
    ) -> tuple[bool, str | None, str]:
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
                return False, None, (
                    f"Maximum {config.max_sessions_per_ip} sessions per IP. "
                    "Close unused browser tabs running this app, or wait for inactive sessions to expire."
                )

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
        config: SecurityConfig | None = None,
    ) -> tuple[bool, str]:
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
        self._active: dict[str, int] = {}
        self._lock = Lock()

    def _get_client_id(self, request: gr.Request | None = None) -> str:
        """Get client identifier from request."""
        return _extract_client_ip(request)

    def acquire(self, request: gr.Request | None = None) -> tuple[bool, str]:
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

    def release(self, request: gr.Request | None = None) -> None:
        """Release a slot after operation completes."""
        client_id = self._get_client_id(request)

        with self._lock:
            current = self._active.get(client_id, 0)
            if current > 0:
                self._active[client_id] = current - 1
                if self._active[client_id] == 0:
                    del self._active[client_id]

    def get_active_count(self, request: gr.Request | None = None) -> int:
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
    retry_after: float | None = None

    def to_headers(self) -> dict[str, str]:
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
FILE_SIGNATURES: dict[str, list[bytes]] = {
    ".json": [b"{", b"["],  # JSON starts with { or [
    ".jsonl": [b"{"],  # JSONL lines start with {
    ".csv": [],  # CSV has no standard signature
    ".txt": [],  # Plain text has no signature
    ".parquet": [b"PAR1"],  # Parquet magic bytes
    ".safetensors": [],  # SafeTensors has complex header
    ".gguf": [b"GGUF"],  # GGUF magic bytes
}


# Hand-rolled "this is HTML/PHP/shell, not data" header heuristic. Used both
# inside the signature-positive path (signature present but not matched) and
# the signature-absent path (.csv/.txt/.safetensors have no canonical magic
# bytes but should still never start with these). Kept module-level so other
# callers (e.g. a future trainer-side upload sink) can reuse it.
_SUSPICIOUS_FILE_HEADERS: tuple[bytes, ...] = (
    b"<!DOCTYPE",
    b"<html",
    b"<HTML",
    b"<script",
    b"<SCRIPT",
    b"<?php",
    b"<?PHP",
    b"#!/",
    b"MZ",            # Windows PE/DOS executable
    b"\x7fELF",       # Linux ELF executable
    b"\xca\xfe\xba\xbe",  # macOS Mach-O fat binary (also Java class — both bad here)
    b"PK\x03\x04",    # ZIP/JAR/DOCX — never a legitimate dataset format here
)


def validate_file_magic(
    file_path: Path,
    expected_extension: str | None = None,
) -> tuple[bool, str]:
    """
    Validate file content matches expected type using magic bytes.

    This helps prevent extension spoofing attacks where malicious files
    are renamed to bypass extension checks.

    Behavior by extension category:
        - Signature-defined (``.json``, ``.jsonl``, ``.parquet``, ``.gguf``):
          must match a known signature, otherwise we check the suspicious-
          headers list and reject on match (HTML/PHP/shell/binary).
        - Signature-absent (``.csv``, ``.txt``, ``.safetensors``): no
          positive signature exists, but we still apply the suspicious-
          headers heuristic so a renamed ``index.html`` cannot slide in
          as a ``.csv`` upload (FB-010).

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

    try:
        with open(file_path, "rb") as f:
            # Read enough bytes to check signature
            header = f.read(16)
    except Exception as e:
        return False, f"Failed to read file: {e}"

    # First: always run the suspicious-headers heuristic so no-signature
    # extensions (.csv/.txt/.safetensors) still get a minimum defense.
    # Categorize the rejection so the message stays useful while remaining
    # backward-compatible with existing "HTML/script" assertions.
    html_script_signatures = (
        b"<!DOCTYPE", b"<html", b"<HTML", b"<script", b"<SCRIPT",
        b"<?php", b"<?PHP", b"#!/",
    )
    binary_signatures = (
        b"MZ", b"\x7fELF", b"\xca\xfe\xba\xbe", b"PK\x03\x04",
    )
    for suspicious in html_script_signatures:
        if header.startswith(suspicious):
            log_security_event(
                "suspicious_file_content",
                file=file_path.name,
                expected_extension=ext,
                found_signature=header[:20].decode("utf-8", errors="replace"),
            )
            return False, f"File content appears to be HTML/script, not {ext}"
    for suspicious in binary_signatures:
        if header.startswith(suspicious):
            log_security_event(
                "suspicious_file_content",
                file=file_path.name,
                expected_extension=ext,
                found_signature=header[:20].hex(),
            )
            return False, (
                f"File content appears to be a binary executable or archive, not {ext}"
            )

    if not signatures:
        # No positive signature for this extension; the suspicious-headers
        # check above is the only defense available. Message preserves the
        # legacy "No signature check available" wording so downstream
        # tooling that grep'd that string still matches.
        return True, "No signature check available"

    # Positive signature defined — at least one must match.
    for sig in signatures:
        if header.startswith(sig):
            return True, "Signature valid"

    # Reaches here when a signature is defined but none match AND no
    # suspicious header tripped — flag as content mismatch.
    return False, f"File content does not match expected {ext} signature"


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

    def __init__(self, config: JWTConfig | None = None):
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
        additional_claims: dict[str, Any] | None = None,
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
    ) -> tuple[bool, dict[str, Any] | None, str]:
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
    ) -> tuple[bool, str | None, str]:
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

    def __init__(self, expiry_minutes: int = 60, max_tokens: int = 10000):
        self.expiry_minutes = expiry_minutes
        self._max_tokens = max_tokens
        self._tokens: dict[str, CSRFToken] = {}
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
            self._enforce_token_cap()
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
    ) -> tuple[bool, str]:
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

    def _enforce_token_cap(self) -> None:
        """Evict oldest tokens when dict exceeds max_tokens cap."""
        if len(self._tokens) <= self._max_tokens:
            return
        # Sort by created_at ascending (oldest first), evict excess
        by_age = sorted(
            self._tokens.items(),
            key=lambda item: item[1].created_at,
        )
        excess = len(self._tokens) - self._max_tokens
        for sid, _ in by_age[:excess]:
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
        jwt_config: JWTConfig | None = None,
        csrf_expiry_minutes: int = 60,
    ):
        self.jwt = JWTManager(jwt_config)
        self.csrf = CSRFProtection(csrf_expiry_minutes)
        self._active_sessions: dict[str, str] = {}  # token -> user_id
        self._lock = Lock()

    def login(
        self,
        user_id: str,
        additional_claims: dict[str, Any] | None = None,
    ) -> dict[str, str]:
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
    ) -> tuple[bool, str | None, str]:
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
    ) -> tuple[bool, dict[str, str] | None, str]:
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
_secure_session_handler: SecureSessionHandler | None = None


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
    script_src: list[str] = field(default_factory=lambda: ["'self'"])
    # Style sources
    style_src: list[str] = field(default_factory=lambda: ["'self'", "'unsafe-inline'"])  # Gradio needs inline styles
    # Image sources
    img_src: list[str] = field(default_factory=lambda: ["'self'", "data:", "blob:"])
    # Font sources
    font_src: list[str] = field(default_factory=lambda: ["'self'", "data:"])
    # Connect sources (for API calls, WebSocket)
    connect_src: list[str] = field(default_factory=lambda: ["'self'", "ws:", "wss:"])
    # Frame ancestors (who can embed this page)
    frame_ancestors: list[str] = field(default_factory=lambda: ["'self'"])
    # Base URI
    base_uri: list[str] = field(default_factory=lambda: ["'self'"])
    # Form action
    form_action: list[str] = field(default_factory=lambda: ["'self'"])
    # Object sources (plugins)
    object_src: list[str] = field(default_factory=lambda: ["'none'"])
    # Default fallback
    default_src: list[str] = field(default_factory=lambda: ["'self'"])
    # Report URI for violations (optional)
    report_uri: str | None = None
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
    _nonce: str | None = None

    def __init__(self, config: CSPConfig | None = None):
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

    def get_nonce(self) -> str | None:
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

    def get_header(self, include_nonce: bool = False) -> tuple[str, str]:
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

    def get_all_security_headers(self) -> dict[str, str]:
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


# DEPRECATED (v1.4): the Reflex UI no longer uses this — it's preserved as
# a back-compat alias for downstream callers that ``from ui_security import
# DEFAULT_GRADIO_CSP``. The v1.4 rename in V1_4_BRIEF item 7 will retire the
# "gradio" prefix across this module; until then, prefer ``DEFAULT_REFLEX_CSP``
# for any new wiring (see ``backpropagate/ui_app/middleware/security_headers.py``).
DEFAULT_GRADIO_CSP = CSPConfig(
    script_src=["'self'", "'unsafe-eval'"],  # Gradio needs eval for dynamic components
    style_src=["'self'", "'unsafe-inline'"],  # Gradio uses inline styles
    img_src=["'self'", "data:", "blob:", "https:"],  # Allow images from various sources
    font_src=["'self'", "data:", "https://fonts.gstatic.com"],  # Google Fonts
    connect_src=["'self'", "ws:", "wss:", "https://huggingface.co"],  # WebSocket + HF API
    frame_ancestors=["'self'"],
    object_src=["'none'"],
)


# FRONTEND-A-003 (v1.4 Wave 2): Reflex-tuned CSP for the production
# middleware wired into ``ui_app/app.py::rx.App(api_transformer=...)``.
#
# Differences from ``DEFAULT_GRADIO_CSP`` and the load-bearing reasons:
#
# - ``script_src=['self', 'unsafe-eval', 'unsafe-inline']`` — Reflex's
#   Next.js shell ships an inline bootstrap ``<script>`` block per page
#   (the standard ``__NEXT_DATA__`` hydration payload + Reflex's own
#   per-page Var-binding script). Without ``'unsafe-inline'`` the entire
#   UI is blocked before the WS connects. ``'unsafe-eval'`` is still
#   needed for Reflex's runtime Var → DOM expression machinery (same
#   as Gradio, different mechanism).
# - ``style_src`` unchanged from Gradio — Reflex uses inline styles for
#   the Radix theme + per-component CSS variables (the TOKENS_CSS
#   ``<style>`` tag rx.el.style injects on every page).
# - ``img_src=['self', 'data:', 'blob:', 'https:']`` — Reflex serves
#   bundled SVG icons from /icons (self) and the chrome's auth-badge
#   chip can carry inline SVG data URIs. ``blob:`` is needed for
#   client-rendered chart exports / file-download affordances.
# - ``connect_src=['self', 'ws:', 'wss:']`` — Reflex's /_event WS
#   endpoint is on the same origin. ``ws:`` covers loopback; ``wss:``
#   covers the documented cloudflared-tunnel / reverse-proxy deploy
#   shapes. HF API is NOT included — the UI never directly calls
#   huggingface.co from the browser (model downloads happen server-
#   side in the trainer process; the UI only consumes WS frames).
# - ``font_src=['self', 'data:']`` — Reflex's stylesheet bundle inlines
#   the Inter / JetBrains Mono webfonts (Wave 5.5 design-token doc), so
#   we don't need Google Fonts. Dropping the Google Fonts allowance
#   tightens the surface and removes a third-party network dependency
#   the UI doesn't actually use.
# - ``object-src='none'`` and ``frame-ancestors='self'`` — same as
#   Gradio. No plugin objects; no iframe-embedding outside same-origin.
DEFAULT_REFLEX_CSP = CSPConfig(
    script_src=["'self'", "'unsafe-eval'", "'unsafe-inline'"],
    style_src=["'self'", "'unsafe-inline'"],
    img_src=["'self'", "data:", "blob:", "https:"],
    font_src=["'self'", "data:"],
    connect_src=["'self'", "ws:", "wss:"],
    frame_ancestors=["'self'"],
    object_src=["'none'"],
)


def get_gradio_csp(report_only: bool = False) -> ContentSecurityPolicy:
    """
    Get a CSP configured for Gradio apps.

    DEPRECATED (v1.4): the Reflex UI uses ``get_reflex_csp`` instead. This
    helper is preserved for back-compat with any downstream caller that
    imported the Gradio name. The v1.4 rename in V1_4_BRIEF item 7 will
    retire the "gradio" prefix across this module.

    Args:
        report_only: If True, only report violations (don't enforce)

    Returns:
        ContentSecurityPolicy instance
    """
    # FRONTEND-B-007 (v1.4 Wave 4 Stage C humanization): emit a
    # DeprecationWarning so any downstream caller importing the Gradio
    # name sees an audit-trail rather than silent grandfathering onto
    # the historical Gradio-era allowlist. ``stacklevel=2`` points the
    # warning at the caller of ``get_gradio_csp``, not this function.
    # Wave 6a will retire ``DEFAULT_GRADIO_CSP`` / ``get_gradio_csp``
    # outright per ``V1_4_BRIEF`` item 7; until then, this warning is
    # the operator-facing migration nudge.
    import warnings as _warnings

    _warnings.warn(
        "get_gradio_csp / DEFAULT_GRADIO_CSP are deprecated as of v1.4; "
        "use get_reflex_csp / DEFAULT_REFLEX_CSP for the Reflex UI. "
        "Wave 6a will remove the gradio-named symbols.",
        DeprecationWarning,
        stacklevel=2,
    )
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


def get_reflex_csp(report_only: bool = False) -> ContentSecurityPolicy:
    """Get a CSP configured for the Reflex UI (FRONTEND-A-003, v1.4 Wave 2).

    Returns a ``ContentSecurityPolicy`` seeded from ``DEFAULT_REFLEX_CSP``.
    See that constant's docstring for the per-directive rationale.

    Args:
        report_only: If True, only report violations (don't enforce). Use
            during a deployment shakeout to log violations to console
            (``Content-Security-Policy-Report-Only`` header) without
            breaking the UI on a missing directive.

    Returns:
        ContentSecurityPolicy instance — wired into the new
        ``security_headers_middleware`` (sibling of ``request_logging``,
        AFTER auth, BEFORE Reflex in the chain).
    """
    config = CSPConfig(
        script_src=DEFAULT_REFLEX_CSP.script_src.copy(),
        style_src=DEFAULT_REFLEX_CSP.style_src.copy(),
        img_src=DEFAULT_REFLEX_CSP.img_src.copy(),
        font_src=DEFAULT_REFLEX_CSP.font_src.copy(),
        connect_src=DEFAULT_REFLEX_CSP.connect_src.copy(),
        frame_ancestors=DEFAULT_REFLEX_CSP.frame_ancestors.copy(),
        object_src=DEFAULT_REFLEX_CSP.object_src.copy(),
        report_only=report_only,
    )
    return ContentSecurityPolicy(config)


# =============================================================================
# MIDDLEWARE HELPERS FOR GRADIO
# =============================================================================

def apply_security_headers(
    response: Any,
    csp: ContentSecurityPolicy | None = None,
) -> None:
    """
    Apply security headers to a response object.

    Works with various response types (Gradio, FastAPI, etc.).

    Args:
        response: Response object with headers attribute
        csp: ContentSecurityPolicy instance. When ``None``, defaults to
            :func:`get_reflex_csp` — the v1.4 Reflex-tuned CSP (FRONTEND-A-003).
            Pre-v1.4 the default was :func:`get_gradio_csp`, which permitted
            ``https://huggingface.co`` and ``https://fonts.gstatic.com``
            (Gradio era); those are no longer needed in the Reflex era.
            Callers that need the legacy default can pass
            ``csp=get_gradio_csp()`` explicitly.
    """
    if csp is None:
        csp = get_reflex_csp()

    headers = csp.get_all_security_headers()

    if hasattr(response, "headers"):
        for name, value in headers.items():
            response.headers[name] = value
    elif hasattr(response, "set_header"):
        for name, value in headers.items():
            response.set_header(name, value)


def security_headers_dict(csp: ContentSecurityPolicy | None = None) -> dict[str, str]:
    """Return the full security-header dict for ASGI-level emission.

    FRONTEND-A-003 (v1.4 Wave 2): ``apply_security_headers`` mutates a
    response object (Starlette / FastAPI / Gradio shape). ASGI middlewares
    instead manipulate the ``headers`` list on the
    ``http.response.start`` message — they have no response object to
    mutate. This helper exposes the same set of headers (CSP +
    X-Content-Type-Options + X-Frame-Options + X-XSS-Protection +
    Referrer-Policy + Permissions-Policy) as a plain dict so the ASGI
    middleware can encode them as ``(name_bytes, value_bytes)`` tuples.

    Args:
        csp: ContentSecurityPolicy instance. ``None`` uses the v1.4
            Reflex-tuned default (same as ``apply_security_headers``).

    Returns:
        Dict of header name -> value. All values are plain ASCII; the
        ASGI middleware is responsible for ``.encode("ascii")`` on both
        sides.
    """
    if csp is None:
        csp = get_reflex_csp()
    return csp.get_all_security_headers()


# =============================================================================
# FRAMEWORK-AGNOSTIC HELPERS (moved from ui.py during Reflex migration v1.1.0)
#
# These helpers were authored against the Gradio surface in Stage B/C but their
# logic is framework-independent — pure string/path/auth manipulation that the
# Reflex pages call the same way. The moves preserve them as a stable surface
# across the UI framework migration.
# =============================================================================


def safe_markdown_fence(content: str, language: str = "") -> str:
    """Wrap user content in a fenced Markdown code block safely (C-UX-005).

    Dataset / Modelfile / Ollama previews wrap user-supplied content in
    triple-backtick fences for display. If the content itself contains the
    literal sequence ``` (extremely common — ChatML samples literally include
    code fences as training data, and Modelfile bodies read from disk can
    contain arbitrary characters), the outer fence terminates early. The rest
    renders as raw markdown / creates a Markdown injection vector.

    Defense: count the longest backtick run inside the content and use a
    fence one tick longer. CommonMark explicitly allows this. We always
    use at least three backticks so the output renders correctly in
    historical viewers too.
    """
    if content is None:
        content = ""
    # Find the longest run of consecutive backticks in the content.
    longest = 0
    current = 0
    for ch in content:
        if ch == "`":
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    fence_len = max(3, longest + 1)
    fence = "`" * fence_len
    lang = language or ""
    return f"{fence}{lang}\n{content}\n{fence}"


# FB-011: user-facing error sanitization.
# Raw exception messages from torch / transformers / huggingface_hub frequently
# embed absolute filesystem paths (home directory, HF cache, internal model
# layout). Surfacing those into the UI toast leaks both PII (the operator's
# username) and an internal-map of the deployment that attackers shouldn't get
# from a UI handler. The patterns below redact:
#   - Unix-style absolute paths starting with /home/, /Users/, /root/
#   - Windows-style absolute paths under C:\Users\<name>\, including UNC roots
#   - The HF cache location (~/.cache/huggingface/...)
#   - Whole tempdirs surfaced by tempfile / shutil
# Truncate to a fixed limit so a 10 KB pyarrow traceback can't blow up the
# toast component.
_REDACTED = "<redacted-path>"
_PATH_REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/(?:home|Users|root)/[^\s'\":]+"),
    re.compile(r"[A-Za-z]:\\Users\\[^\s'\":]+"),
    re.compile(r"\\\\[^\\\s'\":]+\\[^\s'\":]+"),  # UNC \\server\share\...
    re.compile(r"/tmp/[^\s'\":]+"),  # nosec B108 — regex pattern matches /tmp paths for REDACTION in error messages; not an actual /tmp file write
    re.compile(r"[A-Za-z]:\\Windows\\Temp\\[^\s'\":]+"),
    re.compile(r"~?/?\.cache/huggingface/[^\s'\":]*"),
)


def _redact_paths(text: str) -> str:
    """
    Replace absolute filesystem paths with ``<redacted-path>`` for UI display.

    Used by ``sanitize_error_for_user`` so trainer / transformers errors that
    embed the operator's home directory don't leak into the UI toast.
    """
    if not text:
        return text
    for pat in _PATH_REDACTION_PATTERNS:
        text = pat.sub(_REDACTED, text)
    return text


def sanitize_error_for_user(
    exc: BaseException,
    operation: str,
    max_length: int = 240,
) -> tuple[str, str | None]:
    """
    Convert a backend exception into a user-safe display message (FB-011).

    Returns a ``(message, suggestion)`` tuple:

    - For ``BackpropagateError`` subclasses, return the structured
      ``{code}: {message}`` plus the suggestion (already user-facing — these
      are authored to be safe to display).
    - For any other exception type, return a generic "Internal error
      (RUNTIME_UI). Check server logs for details." — the full traceback is
      still logged server-side via ``logger.exception`` at the call site,
      so operators see the real cause; users see something actionable
      without internal paths leaking.

    The ``operation`` argument is folded into the user-facing message
    ("during training", "during export") so the toast tells the user which
    handler failed even when the underlying exception is hidden.
    """
    from .exceptions import BackpropagateError

    # BackpropagateError: structured, user-authored — surface code + message.
    if isinstance(exc, BackpropagateError):
        code = exc.code or "BACKPROPAGATE_ERROR"
        message = _redact_paths(exc.message or str(exc))
        if len(message) > max_length:
            message = message[: max_length - 1] + "…"
        suggestion = exc.suggestion
        if suggestion:
            suggestion = _redact_paths(suggestion)
            if len(suggestion) > max_length:
                suggestion = suggestion[: max_length - 1] + "…"
        return f"{code}: {message}", suggestion

    # Anything else: hide the type + the full message; surface a stable
    # operator-actionable string with the operation name only.
    generic = (
        f"Internal error during {operation} (RUNTIME_UI). "
        "Check the server logs for full details."
    )
    return generic, None


def validate_auth_shape(auth: Any) -> None:
    """
    FB-012: Validate the shape of the ``auth`` kwarg before passing it to the
    UI server (Gradio originally; Reflex/FastAPI in v1.1.0+).

    Frameworks' own validation is permissive in ways that can grant
    unintended access — e.g. an empty list silently disables auth, a
    one-element tuple raises a confusing internal TypeError, and a list
    containing a non-tuple element gets coerced. Validate at the launch
    boundary so misconfigured auth fails loudly rather than degrading to "no
    auth" or "auth-but-not-in-the-shape-you-think".

    Accepts:
        - ``None`` (caller's responsibility to enforce share+auth)
        - A 2-element tuple ``(username, password)`` — both non-empty strings.
        - A non-empty list of such 2-element tuples (multi-user).
        - A ``callable`` (Gradio supports custom auth functions; preserved
          for backward compat though Reflex doesn't use it).

    Rejects everything else with ``BackpropagateError(
    code="INPUT_AUTH_INVALID_SHAPE")`` and a hint enumerating the accepted
    shapes.
    """
    from .exceptions import BackpropagateError

    if auth is None:
        return

    if callable(auth):
        # Callable auth (legacy Gradio) — defer validation to the framework.
        return

    def _is_credential_pair(value: Any) -> bool:
        if not isinstance(value, tuple):
            return False
        if len(value) != 2:
            return False
        username, password = value
        if not isinstance(username, str) or not isinstance(password, str):
            return False
        # Empty strings produce a silently-broken login flow.
        return bool(username) and bool(password)

    accepted_shapes_hint = (
        "Accepted shapes: (username, password) tuple of non-empty strings; "
        "list of such tuples; or a callable (username, password) -> bool. "
        "Empty strings, empty lists, and non-tuple elements are rejected."
    )

    if isinstance(auth, tuple):
        if not _is_credential_pair(auth):
            raise BackpropagateError(
                message="auth tuple must be (username, password) of non-empty strings.",
                code="INPUT_AUTH_INVALID_SHAPE",
                suggestion=accepted_shapes_hint,
                details={"shape": "tuple", "length": len(auth)},
            )
        return

    if isinstance(auth, list):
        if not auth:
            raise BackpropagateError(
                message="auth list must contain at least one (username, password) tuple — empty lists silently disable auth.",
                code="INPUT_AUTH_INVALID_SHAPE",
                suggestion=accepted_shapes_hint,
                details={"shape": "list", "length": 0},
            )
        for index, entry in enumerate(auth):
            if not _is_credential_pair(entry):
                raise BackpropagateError(
                    message=(
                        f"auth[{index}] must be a (username, password) tuple of non-empty strings."
                    ),
                    code="INPUT_AUTH_INVALID_SHAPE",
                    suggestion=accepted_shapes_hint,
                    details={"shape": "list", "bad_index": index},
                )
        return

    # Anything else (dict, str, int, ...) — reject.
    raise BackpropagateError(
        message=(
            f"auth must be None, a (username, password) tuple, a list of such tuples, "
            f"or a callable. Got: {type(auth).__name__}"
        ),
        code="INPUT_AUTH_INVALID_SHAPE",
        suggestion=accepted_shapes_hint,
        details={"shape": type(auth).__name__},
    )


# Internal alias preserved for any code that imported the leading-underscore
# name from ui.py — those imports will be updated, but the alias prevents
# transient breakage during the migration.
_validate_auth_shape = validate_auth_shape


# =============================================================================
# LEGACY ALIAS SHIM (v1.4 rename — see module docstring for the cycle)
# =============================================================================
#
# Wave 6a foundation (V1_4_BRIEF item 7) — drop the ``gradio_`` prefix from
# the public UI-error helpers + classmethod, keeping the legacy names alive
# via module-level ``__getattr__``. The shim emits a ``DeprecationWarning``
# pointing operators at the new symbol so downstream consumers see an
# audit-trail rather than silent grandfathering.
#
# Deprecation cycle (advisor 2026-05-25 Q4; removal version revised
# 2026-06-20 to track the actual ship schedule):
#   v1.4 → present → DeprecationWarning (silent by default; -W default shows it)
#   future release (v1.7 or later) → AttributeError (legacy names removed)
#
# ``DEFAULT_GRADIO_CSP`` + ``get_gradio_csp`` are NOT in this table — they
# keep the existing in-place ``DeprecationWarning`` shape from Wave 2
# FRONTEND-A-003 + Stage C FRONTEND-B-007 (the canonical replacements are
# ``DEFAULT_REFLEX_CSP`` + ``get_reflex_csp``, not a third ``DEFAULT_UI_*``
# name). The constant continues to live in module globals because it's
# referenced by ``get_gradio_csp()``'s internal ``.copy()`` calls.
#
# ``RequestContext.from_gradio_request`` is NOT in this table either — it's
# a classmethod, not a module-level symbol, so the deprecation shim lives
# on the dataclass itself (see :meth:`RequestContext.from_gradio_request`
# above).

_LEGACY_ALIASES: dict[str, str] = {
    "safe_gradio_handler": "safe_ui_handler",
    "raise_gradio_error": "raise_ui_error",
    "raise_gradio_warning": "raise_ui_warning",
    "raise_gradio_info": "raise_ui_info",
}


def __getattr__(name: str) -> Any:
    """Module-level legacy-alias resolver (PEP 562).

    Resolves a v1.0-era ``gradio_``-prefixed symbol to its v1.4 canonical
    name while emitting a ``DeprecationWarning`` so the operator sees the
    migration nudge in stderr. ``stacklevel=2`` points the warning at the
    caller of the import / attribute access, not this function.

    Anything not in ``_LEGACY_ALIASES`` raises ``AttributeError`` per the
    PEP 562 contract (so ``hasattr`` / ``getattr`` with default still work
    naturally).
    """
    if name in _LEGACY_ALIASES:
        new_name = _LEGACY_ALIASES[name]
        import warnings as _warnings

        _warnings.warn(
            f"{name!r} is deprecated in v1.4; use {new_name!r} instead. "
            f"The legacy name will be removed in a future release "
            f"(v1.7 or later).",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(
        f"module 'backpropagate.ui_security' has no attribute {name!r}"
    )
