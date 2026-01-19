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
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TextIO, Union

__all__ = [
    "configure_logging",
    "get_logger",
    "get_standard_logger",
    "add_request_context",
    "clear_request_context",
    "LogContext",
    "STRUCTLOG_AVAILABLE",
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


def _get_log_file() -> Optional[str]:
    """Get log file path from environment."""
    return os.environ.get("BACKPROPAGATE_LOG_FILE")


# =============================================================================
# STRUCTLOG CONFIGURATION
# =============================================================================

def _configure_structlog(
    level: str = "INFO",
    json_logs: bool = False,
    log_file: Optional[str] = None,
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
    shared_processors: List[Processor] = [
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
            structlog.processors.JSONRenderer(),  # JSON output
        ]
    else:
        # Development: Pretty console output
        processors = shared_processors + [
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
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level, logging.INFO),
    )

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
    log_file: Optional[str] = None,
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

    root.addHandler(handler)


# =============================================================================
# PUBLIC API
# =============================================================================

_configured = False


def configure_logging(
    level: Optional[str] = None,
    json_logs: Optional[bool] = None,
    log_file: Optional[str] = None,
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


def get_logger(name: Optional[str] = None) -> Any:
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
        self._token: Optional[Any] = None

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
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
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
        val_loss: Optional[float] = None,
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
        config: Optional[Dict[str, Any]] = None,
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
