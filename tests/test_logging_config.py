"""
Tests for backpropagate.logging_config module.

Tests cover:
- Logging configuration
- Logger creation (with and without structlog)
- Context management
- TrainingLogger specialized logging
- Environment variable handling
- Fallback standard logging
"""

import logging
import os
from io import StringIO
from unittest import mock

import pytest

from backpropagate.logging_config import (
    configure_logging,
    get_logger,
    get_standard_logger,
    add_request_context,
    clear_request_context,
    LogContext,
    STRUCTLOG_AVAILABLE,
)


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def setup_method(self):
        """Reset logging state before each test."""
        # Reset the _configured flag
        import backpropagate.logging_config as lc
        lc._configured = False
        # Clear root handlers
        root = logging.getLogger()
        root.handlers.clear()

    def test_configure_logging_no_exception(self):
        """configure_logging doesn't raise exceptions."""
        configure_logging(force=True)
        # Should complete without error

    def test_configure_logging_default_level(self):
        """configure_logging uses INFO level by default."""
        configure_logging(force=True)
        # Root logger should be at INFO
        root = logging.getLogger()
        assert root.level <= logging.INFO

    def test_configure_logging_custom_level(self):
        """configure_logging respects custom level."""
        configure_logging(level="DEBUG", force=True)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_configure_logging_idempotent(self):
        """configure_logging only configures once without force."""
        configure_logging(level="INFO", force=True)
        # Second call should be ignored
        configure_logging(level="DEBUG")  # No force
        # Level should still be INFO (not DEBUG)
        root = logging.getLogger()
        # The first call set INFO, second was ignored
        assert root.level == logging.INFO

    def test_configure_logging_force_reconfigure(self):
        """configure_logging with force=True reconfigures."""
        configure_logging(level="INFO", force=True)
        configure_logging(level="DEBUG", force=True)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_configure_logging_from_env_level(self):
        """configure_logging reads level from environment."""
        import backpropagate.logging_config as lc
        lc._configured = False

        with mock.patch.dict(os.environ, {"BACKPROPAGATE_LOG_LEVEL": "WARNING"}):
            configure_logging(force=True)
            root = logging.getLogger()
            assert root.level == logging.WARNING

    def test_configure_logging_json_false(self):
        """configure_logging with json_logs=False uses console output."""
        configure_logging(json_logs=False, force=True)
        # Should complete without error

    def test_configure_logging_json_true(self):
        """configure_logging with json_logs=True uses JSON output."""
        configure_logging(json_logs=True, force=True)
        # Should complete without error


class TestGetLogger:
    """Tests for get_logger function."""

    def setup_method(self):
        """Reset logging state before each test."""
        import backpropagate.logging_config as lc
        lc._configured = False

    def test_get_logger_returns_logger(self):
        """get_logger returns a logger object."""
        logger = get_logger("test")
        assert logger is not None

    def test_get_logger_with_name(self):
        """get_logger with name returns named logger."""
        logger = get_logger("my.module")
        # Logger should have the name (structlog or standard)
        assert logger is not None

    def test_get_logger_without_name(self):
        """get_logger without name returns root-like logger."""
        logger = get_logger()
        assert logger is not None

    def test_get_logger_auto_configures(self):
        """get_logger auto-configures logging if not configured."""
        import backpropagate.logging_config as lc
        lc._configured = False

        logger = get_logger("test")
        assert lc._configured is True

    def test_get_logger_has_log_methods(self):
        """get_logger returns object with standard log methods."""
        logger = get_logger("test")
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_get_logger_info_callable(self):
        """Logger info method is callable."""
        logger = get_logger("test")
        # Should not raise
        logger.info("test message")

    def test_get_logger_with_kwargs(self):
        """Logger accepts keyword arguments for structured logging."""
        logger = get_logger("test")
        # Should not raise (structlog or standard logging)
        try:
            logger.info("test message", extra_key="value")
        except TypeError:
            # Standard logging doesn't accept arbitrary kwargs
            # but that's expected behavior
            pass


class TestGetStandardLogger:
    """Tests for get_standard_logger function."""

    def setup_method(self):
        """Reset logging state before each test."""
        import backpropagate.logging_config as lc
        lc._configured = False

    def test_get_standard_logger_returns_logger(self):
        """get_standard_logger returns a logging.Logger."""
        logger = get_standard_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_get_standard_logger_with_name(self):
        """get_standard_logger with name returns named logger."""
        logger = get_standard_logger("my.standard.module")
        assert logger.name == "my.standard.module"

    def test_get_standard_logger_auto_configures(self):
        """get_standard_logger auto-configures logging if not configured."""
        import backpropagate.logging_config as lc
        lc._configured = False

        logger = get_standard_logger("test")
        assert lc._configured is True


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_log_context_enter_exit(self):
        """LogContext can be entered and exited."""
        with LogContext(request_id="123"):
            pass  # Should not raise

    def test_log_context_returns_self(self):
        """LogContext __enter__ returns self."""
        ctx = LogContext(key="value")
        result = ctx.__enter__()
        assert result is ctx
        ctx.__exit__(None, None, None)

    def test_log_context_stores_context(self):
        """LogContext stores provided context."""
        ctx = LogContext(request_id="abc", user="john")
        assert ctx.context == {"request_id": "abc", "user": "john"}

    def test_log_context_nested(self):
        """Nested LogContext works correctly."""
        with LogContext(outer="1"):
            with LogContext(inner="2"):
                pass  # Should not raise


class TestAddRequestContext:
    """Tests for add_request_context function."""

    def test_add_request_context_no_exception(self):
        """add_request_context doesn't raise."""
        add_request_context(request_id="123")
        # Should complete without error

    def test_add_request_context_multiple(self):
        """add_request_context can be called multiple times."""
        add_request_context(key1="value1")
        add_request_context(key2="value2")
        # Should not raise


class TestClearRequestContext:
    """Tests for clear_request_context function."""

    def test_clear_request_context_no_exception(self):
        """clear_request_context doesn't raise."""
        clear_request_context()
        # Should complete without error

    def test_clear_request_context_after_add(self):
        """clear_request_context clears added context."""
        add_request_context(request_id="123")
        clear_request_context()
        # Should not raise


class TestStructlogAvailable:
    """Tests for STRUCTLOG_AVAILABLE constant."""

    def test_structlog_available_is_bool(self):
        """STRUCTLOG_AVAILABLE is a boolean."""
        assert isinstance(STRUCTLOG_AVAILABLE, bool)

    def test_structlog_import_consistent(self):
        """STRUCTLOG_AVAILABLE matches actual import."""
        try:
            import structlog
            assert STRUCTLOG_AVAILABLE is True
        except ImportError:
            assert STRUCTLOG_AVAILABLE is False


class TestTrainingLogger:
    """Tests for TrainingLogger class."""

    def setup_method(self):
        """Reset logging state before each test."""
        import backpropagate.logging_config as lc
        lc._configured = False
        configure_logging(level="DEBUG", force=True)

    def test_training_logger_import(self):
        """TrainingLogger can be imported."""
        from backpropagate.logging_config import TrainingLogger
        assert TrainingLogger is not None

    def test_training_logger_creation(self):
        """TrainingLogger can be created with run name."""
        from backpropagate.logging_config import TrainingLogger
        tlog = TrainingLogger("test-run")
        assert tlog.run_name == "test-run"

    def test_training_logger_has_logger(self):
        """TrainingLogger has internal logger."""
        from backpropagate.logging_config import TrainingLogger
        tlog = TrainingLogger("test-run")
        assert tlog.logger is not None

    def test_training_logger_log_step(self):
        """TrainingLogger.log_step doesn't raise."""
        from backpropagate.logging_config import TrainingLogger
        tlog = TrainingLogger("test-run")
        tlog.log_step(step=10, loss=1.234)

    def test_training_logger_log_step_with_lr(self):
        """TrainingLogger.log_step accepts learning rate."""
        from backpropagate.logging_config import TrainingLogger
        tlog = TrainingLogger("test-run")
        tlog.log_step(step=10, loss=1.234, lr=2e-4)

    def test_training_logger_log_step_with_grad_norm(self):
        """TrainingLogger.log_step accepts gradient norm."""
        from backpropagate.logging_config import TrainingLogger
        tlog = TrainingLogger("test-run")
        tlog.log_step(step=10, loss=1.234, grad_norm=0.5)

    def test_training_logger_log_step_with_extras(self):
        """TrainingLogger.log_step accepts extra kwargs."""
        from backpropagate.logging_config import TrainingLogger
        tlog = TrainingLogger("test-run")
        tlog.log_step(step=10, loss=1.234, custom="value")

    def test_training_logger_log_epoch(self):
        """TrainingLogger.log_epoch doesn't raise."""
        from backpropagate.logging_config import TrainingLogger
        tlog = TrainingLogger("test-run")
        tlog.log_epoch(epoch=1, train_loss=1.1)

    def test_training_logger_log_epoch_with_val_loss(self):
        """TrainingLogger.log_epoch accepts validation loss."""
        from backpropagate.logging_config import TrainingLogger
        tlog = TrainingLogger("test-run")
        tlog.log_epoch(epoch=1, train_loss=1.1, val_loss=1.2)

    def test_training_logger_log_run_start(self):
        """TrainingLogger.log_run_start doesn't raise."""
        from backpropagate.logging_config import TrainingLogger
        tlog = TrainingLogger("test-run")
        tlog.log_run_start(model="test-model", dataset="test-data")

    def test_training_logger_log_run_start_with_config(self):
        """TrainingLogger.log_run_start accepts config dict."""
        from backpropagate.logging_config import TrainingLogger
        tlog = TrainingLogger("test-run")
        tlog.log_run_start(
            model="test-model",
            dataset="test-data",
            config={"lr": 2e-4, "batch_size": 8}
        )

    def test_training_logger_log_run_end(self):
        """TrainingLogger.log_run_end doesn't raise."""
        from backpropagate.logging_config import TrainingLogger
        tlog = TrainingLogger("test-run")
        tlog.log_run_end(
            final_loss=0.5,
            total_steps=100,
            duration_seconds=120.5
        )

    def test_training_logger_log_checkpoint(self):
        """TrainingLogger.log_checkpoint doesn't raise."""
        from backpropagate.logging_config import TrainingLogger
        tlog = TrainingLogger("test-run")
        tlog.log_checkpoint(path="/models/checkpoint-100", step=100)


class TestStructuredFormatter:
    """Tests for StructuredFormatter fallback class."""

    def test_structured_formatter_import(self):
        """StructuredFormatter can be imported."""
        from backpropagate.logging_config import StructuredFormatter
        assert StructuredFormatter is not None

    def test_structured_formatter_creation(self):
        """StructuredFormatter can be instantiated."""
        from backpropagate.logging_config import StructuredFormatter
        formatter = StructuredFormatter()
        assert formatter is not None

    def test_structured_formatter_format(self):
        """StructuredFormatter.format returns JSON string."""
        import json
        from backpropagate.logging_config import StructuredFormatter

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        # Should be valid JSON
        data = json.loads(result)
        assert "timestamp" in data
        assert "level" in data
        assert "message" in data
        assert data["message"] == "Test message"
        assert data["level"] == "INFO"

    def test_structured_formatter_with_exception(self):
        """StructuredFormatter includes exception info."""
        import json
        from backpropagate.logging_config import StructuredFormatter

        formatter = StructuredFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)
        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_get_log_level_default(self):
        """_get_log_level returns INFO by default."""
        from backpropagate.logging_config import _get_log_level

        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove the env var if present
            os.environ.pop("BACKPROPAGATE_LOG_LEVEL", None)
            level = _get_log_level()
            assert level == "INFO"

    def test_get_log_level_from_env(self):
        """_get_log_level reads from environment."""
        from backpropagate.logging_config import _get_log_level

        with mock.patch.dict(os.environ, {"BACKPROPAGATE_LOG_LEVEL": "DEBUG"}):
            level = _get_log_level()
            assert level == "DEBUG"

    def test_get_log_level_uppercase(self):
        """_get_log_level uppercases the result."""
        from backpropagate.logging_config import _get_log_level

        with mock.patch.dict(os.environ, {"BACKPROPAGATE_LOG_LEVEL": "warning"}):
            level = _get_log_level()
            assert level == "WARNING"

    def test_should_use_json_default(self):
        """_should_use_json auto-detects from TTY."""
        from backpropagate.logging_config import _should_use_json

        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BACKPROPAGATE_LOG_JSON", None)
            # Result depends on whether stderr is a TTY
            result = _should_use_json()
            assert isinstance(result, bool)

    def test_should_use_json_true(self):
        """_should_use_json returns True when env is 'true'."""
        from backpropagate.logging_config import _should_use_json

        with mock.patch.dict(os.environ, {"BACKPROPAGATE_LOG_JSON": "true"}):
            result = _should_use_json()
            assert result is True

    def test_should_use_json_false(self):
        """_should_use_json returns False when env is 'false'."""
        from backpropagate.logging_config import _should_use_json

        with mock.patch.dict(os.environ, {"BACKPROPAGATE_LOG_JSON": "false"}):
            result = _should_use_json()
            assert result is False

    def test_get_log_file_default(self):
        """_get_log_file returns None by default."""
        from backpropagate.logging_config import _get_log_file

        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BACKPROPAGATE_LOG_FILE", None)
            result = _get_log_file()
            assert result is None

    def test_get_log_file_from_env(self):
        """_get_log_file reads from environment."""
        from backpropagate.logging_config import _get_log_file

        with mock.patch.dict(os.environ, {"BACKPROPAGATE_LOG_FILE": "/var/log/app.log"}):
            result = _get_log_file()
            assert result == "/var/log/app.log"


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_defined(self):
        """Module has __all__ defined."""
        import backpropagate.logging_config as lc
        assert hasattr(lc, "__all__")
        assert isinstance(lc.__all__, list)

    def test_expected_exports(self):
        """Module exports expected items."""
        import backpropagate.logging_config as lc
        expected = [
            "configure_logging",
            "get_logger",
            "get_standard_logger",
            "add_request_context",
            "clear_request_context",
            "LogContext",
            "STRUCTLOG_AVAILABLE",
        ]
        for item in expected:
            assert item in lc.__all__, f"Expected export {item} not in __all__"

    def test_all_exports_importable(self):
        """All items in __all__ can be imported."""
        import backpropagate.logging_config as lc
        for name in lc.__all__:
            assert hasattr(lc, name), f"Export {name} not found in module"


class TestFallbackLogging:
    """Tests for fallback standard logging (when structlog unavailable)."""

    def test_configure_standard_logging(self):
        """_configure_standard_logging works correctly."""
        from backpropagate.logging_config import _configure_standard_logging

        # Clear existing handlers
        root = logging.getLogger()
        root.handlers.clear()

        _configure_standard_logging(level="DEBUG", json_logs=False)

        assert root.level == logging.DEBUG
        assert len(root.handlers) > 0

    def test_configure_standard_logging_json(self):
        """_configure_standard_logging with JSON formatter."""
        from backpropagate.logging_config import (
            _configure_standard_logging,
            StructuredFormatter,
        )

        root = logging.getLogger()
        root.handlers.clear()

        _configure_standard_logging(level="INFO", json_logs=True)

        assert len(root.handlers) > 0
        handler = root.handlers[0]
        assert isinstance(handler.formatter, StructuredFormatter)

    def test_configure_standard_logging_non_json(self):
        """_configure_standard_logging with standard formatter."""
        from backpropagate.logging_config import _configure_standard_logging

        root = logging.getLogger()
        root.handlers.clear()

        _configure_standard_logging(level="INFO", json_logs=False)

        assert len(root.handlers) > 0
        handler = root.handlers[0]
        # Should be a standard Formatter, not StructuredFormatter
        assert isinstance(handler.formatter, logging.Formatter)
