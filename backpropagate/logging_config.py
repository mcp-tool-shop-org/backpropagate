"""
Backpropagate - Structured Logging Configuration
=================================================

Production-ready logging using structlog for JSON output in production
and human-readable output in development.

Based on 2026 best practices:
- https://www.structlog.org/en/stable/logging-best-practices.html
- https://betterstack.com/community/guides/logging/structlog/
- https://signoz.io/guides/structlog/

Features:
- Automatic JSON logging in production (non-TTY)
- Pretty console logging in development (TTY)
- Request ID tracing
- Structured exception tracebacks
- Log level configuration via environment
- Integration with standard library logging

Usage:
    from backpropagate.logging_config import get_logger, configure_logging

    # Configure once at startup
    configure_logging(level="INFO", json_logs=True)

    # Get logger
    logger = get_logger(__name__)
    logger.info("Training started", model="qwen", batch_size=4)

Environment Variables:
    BACKPROPAGATE_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO)
    BACKPROPAGATE_LOG_JSON: true/false (default: auto-detect from TTY)
    BACKPROPAGATE_LOG_FILE: Path to log file (optional)
"""

import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "configure_logging",
    "get_logger",
    "get_standard_logger",
    "add_request_context",
    "clear_request_context",
    "bind_run_context",
    "unbind_run_context",
    "run_context",
    "LogContext",
    "STRUCTLOG_AVAILABLE",
    "redact_secrets",
    "SECRET_PATTERNS",
]

# Check if structlog is available
try:
    import structlog
    from structlog.types import Processor
    STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None  # type: ignore
    STRUCTLOG_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

def _get_log_level() -> str:
    """Get log level from environment."""
    return os.environ.get("BACKPROPAGATE_LOG_LEVEL", "INFO").upper()


def _should_use_json() -> bool:
    """Determine if JSON logging should be used."""
    env_json = os.environ.get("BACKPROPAGATE_LOG_JSON", "").lower()
    if env_json == "true":
        return True
    if env_json == "false":
        return False
    # Auto-detect: use JSON if not running in a TTY
    return not sys.stderr.isatty()


def _get_log_file() -> str | None:
    """Get log file path from environment."""
    return os.environ.get("BACKPROPAGATE_LOG_FILE")


# =============================================================================
# SECRET REDACTION (CLI-A-002)
# =============================================================================
#
# Single source of truth for credential redaction, shared by:
#   - the CLI error-output paths (``backpropagate.cli`` re-exports these), and
#   - the structured-log / file pipeline below (a structlog processor for the
#     structlog path + a stdlib ``logging.Filter`` for the fallback path).
#
# Pre-CLI-A-002 the patterns lived only in ``cli.py`` and the log pipeline
# emitted events verbatim — a token bound into log context (or interpolated
# into a log message) reached the JSON/file stream un-redacted. Moving the
# patterns here lets both surfaces redact through ONE definition so they can
# never drift apart.
#
# Pattern shape rationale (carried over from the original cli.py home):
# - High-signal prefixes (Bearer, sk-, hf_, AKIA, ghp_/glpat-, JWT,
#   url-embedded creds) trigger on their own — low false-positive by
#   construction.
# - The keyword-prefixed pattern requires ``key=value`` / ``key:value`` with
#   a high-entropy value (>=8 non-space chars incl. >=1 digit-or-special) so
#   prose like "the token: abc is wrong" is left intact. The separator is
#   captured in group 2 and re-emitted so ``token: foo`` stays
#   ``token: <REDACTED>`` (not ``token=<REDACTED>``).
SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"), "Bearer <REDACTED>"),
    (re.compile(r"sk-[A-Za-z0-9]{8,}"), "sk-<REDACTED>"),
    (re.compile(r"hf_[A-Za-z0-9]{8,}"), "hf_<REDACTED>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA<REDACTED>"),
    # URL-embedded credentials (https://user:token@host/...). Match the
    # canonical scheme://user:secret@host form (NOT the user@host SSH-style
    # address, which carries no secret). The capture preserves scheme + host
    # so the operator still sees which service was contacted.
    (
        re.compile(r"(https?://)[^/\s:@]+:[^/\s:@]+@([^/\s]+)"),
        r"\1<REDACTED>@\2",
    ),
    # JWT tokens (3 base64url segments separated by `.`, RFC 7519). The
    # `eyJ` prefix is the base64 of `{"`, present in essentially every real
    # JWT, which keeps false-positive risk low against prose.
    (
        re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
        "<JWT_REDACTED>",
    ),
    # GitHub PATs (ghp_/ghs_/gho_/ghu_/ghr_): canonical prefix per GitHub's
    # secret-scanning docs; 36+ base62 suffix (modern, post-2021). The legacy
    # 40-hex format is intentionally NOT matched (high false-positive rate).
    (
        re.compile(r"\b(ghp|ghs|gho|ghu|ghr)_[A-Za-z0-9]{36,}\b"),
        r"\1_<REDACTED>",
    ),
    # GitLab PATs (glpat-...): 20+ base62 / hyphen suffix per GitLab docs.
    (
        re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
        "glpat-<REDACTED>",
    ),
    # Keyword-prefixed credentials: key=value OR key:value with a
    # high-entropy value (>=8 non-space chars incl. >=1 digit-or-special).
    # The separator (group 2) is preserved in the replacement.
    (
        re.compile(
            r"(password|passwd|pwd|secret|token|api[_-]?key)"
            r"(\s*[=:]\s*)"
            r'(?=[^\s]*[\d!@#$%^&*()+\-./?_=])'  # lookahead: at least one digit/special
            r"[^\s]{8,}",
            re.IGNORECASE,
        ),
        r"\1\2<REDACTED>",
    ),
]


def redact_secrets(text: str) -> str:
    """
    Best-effort redaction of common secret patterns in arbitrary text.

    Single source of truth shared by the CLI error paths and the log
    pipeline (CLI-A-002). Designed to leave human-readable prose intact
    (e.g. "the token: abc is wrong" is NOT redacted because "abc" is too
    short / has no digit) while catching realistic credential leaks (e.g.
    "api_key=EXAMPLE-NOT-A-REAL-KEY" → "api_key=<REDACTED>"). The separator
    character (``:`` vs ``=``) is preserved to avoid silently rewriting the
    input shape.
    """
    if not text:
        return text
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _redact_event_processor(
    _logger: Any, _method_name: str, event_dict: dict
) -> dict:
    """structlog processor: redact secrets in the rendered event + string values.

    CLI-A-002: runs LATE in the processor chain (after merge_contextvars,
    so context-bound tokens are present) but BEFORE the final renderer
    (JSONRenderer / ConsoleRenderer), so it scrubs both the primary
    ``event`` message and any string-valued bound field (e.g. a ``token``
    bound via ``logger.bind(token=...)`` or run-context) before the bytes
    leave the process.
    """
    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            event_dict[key] = redact_secrets(value)
    return event_dict


class _SecretRedactingFilter(logging.Filter):
    """stdlib logging.Filter that redacts secrets in the rendered message.

    CLI-A-002 fallback-path counterpart to :func:`_redact_event_processor`.
    Attached to every handler on the structlog-less path so the file /
    stream output is scrubbed regardless of which logging backend is live.
    Mutates ``record.msg`` / ``record.args`` so the redaction survives
    downstream formatting (``%`` interpolation included).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Render the final message (applies %-args) then redact, and
            # neutralize args so the formatter doesn't re-interpolate the
            # raw (un-redacted) values.
            rendered = record.getMessage()
            record.msg = redact_secrets(rendered)
            record.args = None
        except Exception:  # noqa: BLE001 — redaction must never drop a log line
            pass
        return True


# =============================================================================
# STRUCTLOG CONFIGURATION
# =============================================================================

def _configure_structlog(
    level: str = "INFO",
    json_logs: bool = False,
    log_file: str | None = None,
) -> None:
    """
    Configure structlog for production use.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Whether to output JSON format
        log_file: Optional file path for logging
    """
    if not STRUCTLOG_AVAILABLE:
        return

    # Shared processors for all logging
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,  # Add context vars
        structlog.processors.add_log_level,  # Add level
        structlog.processors.StackInfoRenderer(),  # Add stack info
        structlog.dev.set_exc_info,  # Add exception info
        structlog.processors.TimeStamper(fmt="iso"),  # ISO timestamp
    ]

    if json_logs:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,  # Structured tracebacks
            # CLI-A-002: scrub credentials from event + string fields before
            # the renderer serializes them to the (file/stdout) stream.
            _redact_event_processor,
            structlog.processors.JSONRenderer(),  # JSON output
        ]
    else:
        # Development: Pretty console output
        processors = shared_processors + [
            # CLI-A-002: scrub credentials before the console renderer.
            _redact_event_processor,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to use structlog
    # Clear existing root handlers first — basicConfig is a no-op when handlers
    # already exist, which would silently skip our format/stream/level settings.
    log_level = getattr(logging, level, logging.INFO)
    root = logging.getLogger()
    # BRIDGE-A-016 (Stage C amend): explicitly .close() each handler BEFORE
    # dropping the reference. The previous root.handlers.clear() left every
    # FileHandler open (its OS file descriptor pinned), so a long-running
    # test session calling configure_logging(..., force=True) repeatedly
    # leaked one fd per call — eventually hitting EMFILE on POSIX. The
    # try/except guards against handlers whose close() raises (rare; the
    # debug log keeps the diagnostic visible without aborting setup).
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:  # noqa: BLE001 — best-effort handler cleanup
            logging.getLogger(__name__).debug(
                "configure_logging: handler.close() raised", exc_info=True
            )
    root.handlers.clear()
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level, logging.INFO),
    )

    # CLI-A-002: attach the redaction filter to the stdlib handlers that
    # basicConfig just created so logs routed through the stdlib `logging`
    # module directly (e.g. audit_log, third-party libraries) are scrubbed
    # too — not only the structlog-rendered events.
    _redact_filter = _SecretRedactingFilter()
    for _h in logging.getLogger().handlers:
        _h.addFilter(_redact_filter)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level, logging.INFO))
        if json_logs:
            # JSON format for file
            file_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
        file_handler.addFilter(_redact_filter)
        logging.getLogger().addHandler(file_handler)


# =============================================================================
# FALLBACK STANDARD LOGGING
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for standard library logging.

    Used when structlog is not available.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        import json

        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)  # type: ignore

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def _configure_standard_logging(
    level: str = "INFO",
    json_logs: bool = False,
    log_file: str | None = None,
) -> None:
    """
    Configure standard library logging (fallback when structlog unavailable).

    Args:
        level: Log level
        json_logs: Whether to use JSON format
        log_file: Optional file path
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Clear existing handlers
    root = logging.getLogger()
    # BRIDGE-A-016 (Stage C amend): close each handler explicitly before
    # dropping it. Same fd-leak rationale as the structlog branch above.
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:  # noqa: BLE001 — best-effort handler cleanup
            logging.getLogger(__name__).debug(
                "_configure_standard_logging: handler.close() raised",
                exc_info=True,
            )
    root.handlers.clear()
    root.setLevel(log_level)

    # Create handler
    handler: logging.Handler
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(log_level)

    # Set formatter
    if json_logs:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    # CLI-A-002: redact credentials on the structlog-less fallback path too,
    # so the file/stream output is scrubbed regardless of backend.
    handler.addFilter(_SecretRedactingFilter())

    root.addHandler(handler)


# =============================================================================
# PUBLIC API
# =============================================================================

_configured = False


def configure_logging(
    level: str | None = None,
    json_logs: bool | None = None,
    log_file: str | None = None,
    force: bool = False,
) -> None:
    """
    Configure logging for the application.

    Should be called once at application startup. Subsequent calls
    are ignored unless force=True.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Default: from env or INFO
        json_logs: Use JSON format. Default: auto-detect from TTY
        log_file: Path to log file. Default: from env or None

    Environment Variables:
        BACKPROPAGATE_LOG_LEVEL: Override level
        BACKPROPAGATE_LOG_JSON: Override json_logs (true/false)
        BACKPROPAGATE_LOG_FILE: Override log_file
    """
    global _configured

    if _configured and not force:
        return

    # Apply defaults from environment
    level = level or _get_log_level()
    json_logs = json_logs if json_logs is not None else _should_use_json()
    log_file = log_file or _get_log_file()

    if STRUCTLOG_AVAILABLE:
        _configure_structlog(level, json_logs, log_file)
    else:
        _configure_standard_logging(level, json_logs, log_file)

    _configured = True


def get_logger(name: str | None = None) -> Any:
    """
    Get a logger instance.

    Returns a structlog logger if available, otherwise a standard library logger.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance with .info(), .debug(), .warning(), .error() methods

    Example:
        logger = get_logger(__name__)
        logger.info("Processing started", items=100)
    """
    # Ensure logging is configured
    if not _configured:
        configure_logging()

    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


def get_standard_logger(name: str) -> logging.Logger:
    """
    Get a standard library logger.

    Use this when you need the standard logging.Logger interface.

    Args:
        name: Logger name

    Returns:
        Standard library Logger
    """
    if not _configured:
        configure_logging()
    return logging.getLogger(name)


# =============================================================================
# CONTEXT MANAGEMENT
# =============================================================================

class LogContext:
    """
    Context manager for adding structured context to logs.

    Usage:
        with LogContext(request_id="abc123", user="john"):
            logger.info("Processing")  # Includes request_id and user
    """

    def __init__(self, **context: Any):
        self.context = context
        self._token: Any | None = None

    def __enter__(self) -> "LogContext":
        if STRUCTLOG_AVAILABLE:
            self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        if STRUCTLOG_AVAILABLE and self._token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


def add_request_context(**context: Any) -> None:
    """
    Add context to all subsequent log messages in this context.

    Useful for adding request ID, user ID, etc. at the start of a request.

    Args:
        **context: Key-value pairs to add to log context

    Example:
        add_request_context(request_id="abc123", user="john")
        logger.info("Processing")  # Includes request_id and user
    """
    if STRUCTLOG_AVAILABLE:
        structlog.contextvars.bind_contextvars(**context)


def clear_request_context() -> None:
    """Clear all context vars."""
    if STRUCTLOG_AVAILABLE:
        structlog.contextvars.clear_contextvars()


# =============================================================================
# RUN CORRELATION CONTEXT (B-001)
# =============================================================================

def bind_run_context(run_id: str, **kwargs: Any) -> None:
    """
    Bind a stable ``run_id`` (and optional session_id / extra fields) to the
    structured-logger context so every log line emitted by this thread carries
    a single correlation token.

    This is the load-bearing primitive behind B-001: an operator triaging a
    multi-hour overnight job can grep one identifier across logs, checkpoint
    manifests, run_history.json, and SLAO merge_history.json without
    cross-referencing wall-clock timestamps across machines.

    Safe to call even when structlog is not installed — becomes a no-op so
    callers don't need a structlog-availability check at every site.

    Args:
        run_id: UUID4-derived (or otherwise stable) correlation token. Should
            be persisted into any artefact saved during this run.
        **kwargs: Optional additional context (e.g. ``session_id``,
            ``model_name``) bound alongside ``run_id``.

    Example:
        >>> import uuid
        >>> run_id = uuid.uuid4().hex
        >>> bind_run_context(run_id=run_id, model="qwen2.5-7b")
        >>> get_logger(__name__).info("run_started run_id=%s" % run_id)
    """
    if not STRUCTLOG_AVAILABLE:
        return
    payload: dict[str, Any] = {"run_id": run_id}
    payload.update(kwargs)
    structlog.contextvars.bind_contextvars(**payload)


def unbind_run_context(*keys: str) -> None:
    """
    Unbind run-correlation context fields previously set by
    :func:`bind_run_context`.

    When called with no arguments, removes the default ``run_id`` key only —
    use :func:`clear_request_context` to wipe the entire context. Safe when
    structlog is unavailable (no-op).

    Args:
        *keys: Names of context fields to unbind. Defaults to ``("run_id",)``.
    """
    if not STRUCTLOG_AVAILABLE:
        return
    target_keys = keys or ("run_id",)
    structlog.contextvars.unbind_contextvars(*target_keys)


# Stage C amend BACKEND-B-019: context-manager wrapper that pairs
# bind_run_context with an automatic unbind on exit. The legacy
# bind/unbind primitives are still exported because:
#   - existing call sites use them across try/finally blocks where
#     refactoring to ``with`` would balloon the diff; and
#   - structlog's contextvars API is itself imperative, so callers who
#     need fine-grained control (e.g. binding mid-flow without an
#     enclosing scope) still need the imperative form.
# But for new call sites the context-manager form makes the bind/unbind
# invariant automatic: a future contributor who adds a third bind key
# inside ``with run_context(...)`` no longer needs to remember to also
# add it to a matching unbind call; the cm tracks every key it bound
# and unbinds exactly those on exit.
class _RunContext:
    """Stage C amend BACKEND-B-019: see :func:`run_context`."""

    def __init__(self, run_id: str, **kwargs: Any) -> None:
        self.run_id = run_id
        self.extra = dict(kwargs)
        self._bound_keys: tuple[str, ...] = ()

    def __enter__(self) -> "_RunContext":
        bind_run_context(self.run_id, **self.extra)
        self._bound_keys = ("run_id",) + tuple(self.extra.keys())
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._bound_keys:
            unbind_run_context(*self._bound_keys)


def run_context(run_id: str, **kwargs: Any) -> _RunContext:
    """Stage C amend BACKEND-B-019: context-manager wrapper around
    :func:`bind_run_context` / :func:`unbind_run_context`.

    Pairs bind on enter with unbind on exit so the bind/unbind invariant
    is automatic — a future contributor adding a third bind key no
    longer needs to remember to update a matching unbind call.

    Example:
        with run_context(run_id=run_id, session_kind="multi_run"):
            do_work()
        # run_id and session_kind unbound here, even if do_work raised.

    Safe when structlog is unavailable (the inner bind/unbind become
    no-ops; the context manager itself still works).
    """
    return _RunContext(run_id, **kwargs)


# =============================================================================
# TRAINING-SPECIFIC LOGGING
# =============================================================================

class TrainingLogger:
    """
    Specialized logger for training progress.

    Provides consistent structured logging for ML training loops.
    Works with both structlog (structured kwargs) and standard logging (formatted string).

    Usage:
        tlog = TrainingLogger("qwen-finetune")
        tlog.log_step(step=100, loss=1.23, lr=2e-4)
        tlog.log_epoch(epoch=1, train_loss=1.1, val_loss=1.2)
    """

    def __init__(self, run_name: str):
        self.run_name = run_name
        self.logger = get_logger(f"training.{run_name}")
        self._use_structlog = STRUCTLOG_AVAILABLE

    def _log(self, level: str, event: str, **data: Any) -> None:
        """Log with structlog or standard logging."""
        log_method = getattr(self.logger, level)
        if self._use_structlog:
            log_method(event, **data)
        else:
            # Format as a readable string for standard logging
            parts = [f"{k}={v}" for k, v in data.items()]
            msg = f"{event}: {', '.join(parts)}" if parts else event
            log_method(msg)

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float | None = None,
        grad_norm: float | None = None,
        **extras: Any,
    ) -> None:
        """Log a training step."""
        data = {
            "run": self.run_name,
            "step": step,
            "loss": round(loss, 4),
        }
        if lr is not None:
            data["lr"] = f"{lr:.2e}"
        if grad_norm is not None:
            data["grad_norm"] = round(grad_norm, 4)
        data.update(extras)

        self._log("info", "train_step", **data)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        **extras: Any,
    ) -> None:
        """Log epoch completion."""
        data = {
            "run": self.run_name,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
        }
        if val_loss is not None:
            data["val_loss"] = round(val_loss, 4)
        data.update(extras)

        self._log("info", "epoch_complete", **data)

    def log_run_start(
        self,
        model: str,
        dataset: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Log training run start."""
        self._log(
            "info",
            "run_started",
            run=self.run_name,
            model=model,
            dataset=dataset,
            config=config or {},
        )

    def log_run_end(
        self,
        final_loss: float,
        total_steps: int,
        duration_seconds: float,
    ) -> None:
        """Log training run completion."""
        self._log(
            "info",
            "run_complete",
            run=self.run_name,
            final_loss=round(final_loss, 4),
            total_steps=total_steps,
            duration_seconds=round(duration_seconds, 2),
        )

    def log_checkpoint(self, path: str, step: int) -> None:
        """Log checkpoint save."""
        self._log(
            "info",
            "checkpoint_saved",
            run=self.run_name,
            path=path,
            step=step,
        )
