"""Coverage-raising tests for ``backpropagate.logging_config``.

The structlog-less code paths cannot be reached in an environment where
structlog is installed, so ``_load_without_structlog`` executes a fresh copy of
the module with ``sys.modules['structlog'] = None`` (which makes the import
raise ``ImportError``) and registers it under the real dotted name for the
duration of the test; ``monkeypatch`` puts the original module back.

Nothing is mocked beyond that import block. Tests that mutate the root logger
restore its handlers/level afterwards.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

import backpropagate
import backpropagate.logging_config as lc_real

_LC_PATH = Path(backpropagate.__file__).parent / "logging_config.py"

# Process-wide root-logger / structlog mutation: pin to one xdist worker like test_logging_config.
pytestmark = pytest.mark.serial


@pytest.fixture
def restore_root_logging():
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    saved_cfg = lc_real._configured
    yield
    for h in list(root.handlers):
        if h not in saved_handlers:
            try:
                h.close()
            except Exception:  # noqa: BLE001
                logging.getLogger(__name__).debug("close failed", exc_info=True)
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    lc_real._configured = saved_cfg


@pytest.fixture
def lc_nostructlog(monkeypatch, restore_root_logging):
    """A fresh ``logging_config`` module object imported as if structlog were absent."""
    monkeypatch.setitem(sys.modules, "structlog", None)
    monkeypatch.setitem(sys.modules, "structlog.types", None)
    spec = importlib.util.spec_from_file_location("backpropagate.logging_config", _LC_PATH)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "backpropagate.logging_config", mod)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Module imported without structlog
# =============================================================================


class TestWithoutStructlog:
    def test_flag_and_module_attribute(self, lc_nostructlog):
        assert lc_nostructlog.STRUCTLOG_AVAILABLE is False
        assert lc_nostructlog.structlog is None

    def test_configure_uses_standard_logging_and_sets_flag(self, lc_nostructlog):
        lc_nostructlog.configure_logging(level="WARNING", json_logs=False, force=True)
        assert lc_nostructlog._configured is True
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert any(isinstance(f, lc_nostructlog._SecretRedactingFilter)
                   for h in root.handlers for f in h.filters)

    def test_configure_structlog_is_a_noop(self, lc_nostructlog):
        before = list(logging.getLogger().handlers)
        assert lc_nostructlog._configure_structlog("INFO", False, None) is None
        assert logging.getLogger().handlers == before

    def test_second_configure_without_force_is_ignored(self, lc_nostructlog):
        lc_nostructlog.configure_logging(level="ERROR", force=True)
        lc_nostructlog.configure_logging(level="DEBUG")
        assert logging.getLogger().level == logging.ERROR

    def test_get_logger_returns_adapter_that_folds_fields_into_message(
        self, lc_nostructlog, caplog
    ):
        lc_nostructlog._configured = True  # skip auto-configure
        log = lc_nostructlog.get_logger("cov.adapter")
        assert isinstance(log, lc_nostructlog._StdlibStructuredLogger)
        with caplog.at_level(logging.INFO, logger="cov.adapter"):
            log.info("hello", run="r1", step=3, extra={"k": "v"})
        rec = caplog.records[-1]
        assert rec.getMessage() == "hello run=r1 step=3"
        assert rec.k == "v"  # stdlib-reserved kwarg ``extra`` still reaches logging

    def test_get_logger_auto_configures(self, lc_nostructlog):
        lc_nostructlog._configured = False
        lc_nostructlog.get_logger("cov.auto")
        assert lc_nostructlog._configured is True

    def test_get_standard_logger_auto_configures_and_returns_logger(self, lc_nostructlog):
        lc_nostructlog._configured = False
        log = lc_nostructlog.get_standard_logger("cov.std")
        assert isinstance(log, logging.Logger) and log.name == "cov.std"
        assert lc_nostructlog._configured is True

    def test_context_helpers_are_noops(self, lc_nostructlog):
        lc_nostructlog.add_request_context(a=1)
        lc_nostructlog.clear_request_context()
        lc_nostructlog.bind_run_context("rid", model="m")
        lc_nostructlog.unbind_run_context()
        lc_nostructlog.unbind_run_context("a", "b")
        with lc_nostructlog.LogContext(x=1) as ctx:
            assert ctx.context == {"x": 1} and ctx._token is None
        with lc_nostructlog.run_context("rid", extra="e") as rc:
            assert rc.run_id == "rid" and rc.extra == {"extra": "e"}

    def test_training_logger_formats_key_value_strings(self, lc_nostructlog, caplog):
        lc_nostructlog._configured = True
        tlog = lc_nostructlog.TrainingLogger("run-x")
        assert tlog._use_structlog is False
        with caplog.at_level(logging.INFO, logger="training.run-x"):
            tlog.log_step(step=5, loss=1.23456, lr=2e-4, grad_norm=0.5, custom="c")
            tlog.log_epoch(epoch=2, train_loss=1.0, val_loss=0.9)
            tlog.log_run_start(model="m", dataset="d")
            tlog.log_run_end(final_loss=0.1234567, total_steps=9, duration_seconds=3.14159)
            tlog.log_checkpoint("/tmp/ck", 7)  # noqa: S108
        msgs = [r.getMessage() for r in caplog.records]
        assert msgs[0] == ("train_step: run=run-x, step=5, loss=1.2346, lr=2.00e-04, "
                           "grad_norm=0.5, custom=c")
        assert msgs[1].startswith("epoch_complete: run=run-x, epoch=2, train_loss=1.0, val_loss=0.9")
        assert msgs[2].startswith("run_started: run=run-x, model=m, dataset=d, config={}")
        assert "final_loss=0.1235" in msgs[3] and "duration_seconds=3.14" in msgs[3]
        assert msgs[4] == "checkpoint_saved: run=run-x, path=/tmp/ck, step=7"  # noqa: S108

    def test_training_logger_event_without_data_is_bare(self, lc_nostructlog, caplog):
        lc_nostructlog._configured = True
        tlog = lc_nostructlog.TrainingLogger("bare")
        with caplog.at_level(logging.INFO, logger="training.bare"):
            tlog._log("info", "just_event")
        assert caplog.records[-1].getMessage() == "just_event"


# =============================================================================
# _configure_standard_logging / StructuredFormatter detail
# =============================================================================


class TestStandardConfigure:
    def test_log_file_handler_receives_redacted_output(self, tmp_path, restore_root_logging):
        path = tmp_path / "out.log"
        lc_real._configure_standard_logging("INFO", json_logs=False, log_file=str(path))
        logging.getLogger("cov.file").info("token=hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ok")
        for h in logging.getLogger().handlers:
            h.flush()
        text = path.read_text(encoding="utf-8")
        assert "ok" in text and "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" not in text
        assert " | INFO     | cov.file | " in text

    def test_json_formatter_applied_when_json_logs(self, restore_root_logging):
        lc_real._configure_standard_logging("DEBUG", json_logs=True)
        handler = logging.getLogger().handlers[-1]
        assert isinstance(handler.formatter, lc_real.StructuredFormatter)
        assert logging.getLogger().level == logging.DEBUG

    def test_unknown_level_falls_back_to_info(self, restore_root_logging):
        lc_real._configure_standard_logging("NOT_A_LEVEL")
        assert logging.getLogger().level == logging.INFO

    def test_handler_close_failure_does_not_abort_setup(self, restore_root_logging):
        class Boom(logging.Handler):
            def emit(self, record):
                pass

            def close(self):
                raise RuntimeError("close exploded")

        logging.getLogger().addHandler(Boom())
        lc_real._configure_standard_logging("INFO")
        assert not any(isinstance(h, Boom) for h in logging.getLogger().handlers)

    def test_structured_formatter_includes_extra_dict(self):
        import json

        rec = logging.LogRecord("n", logging.INFO, __file__, 1, "msg %s", ("a",), None)
        rec.extra = {"run_id": "r1"}
        out = json.loads(lc_real.StructuredFormatter().format(rec))
        assert out["message"] == "msg a" and out["run_id"] == "r1" and out["level"] == "INFO"


class TestStructlogConfigure:
    def test_json_with_file_handler_writes_redacted_json(self, tmp_path, restore_root_logging):
        path = tmp_path / "s.log"
        lc_real._configure_structlog("INFO", json_logs=True, log_file=str(path))
        root = logging.getLogger()
        assert any(isinstance(h, logging.FileHandler) for h in root.handlers)
        logging.getLogger("cov.s").warning("api_key=sk-abcdefghijklmnopqrstuvwxyz123456 done")
        for h in root.handlers:
            h.flush()
        text = path.read_text(encoding="utf-8")
        assert "done" in text and "sk-abcdefghijklmnopqrstuvwxyz123456" not in text

    def test_pretty_console_file_handler_uses_timestamp_format(self, tmp_path, restore_root_logging):
        path = tmp_path / "p.log"
        lc_real._configure_structlog("DEBUG", json_logs=False, log_file=str(path))
        fh = next(h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler))
        assert " - " in fh.formatter._fmt and "%(asctime)s" in fh.formatter._fmt
        assert fh.level == logging.DEBUG

    def test_handler_close_failure_does_not_abort_structlog_setup(self, restore_root_logging):
        class Boom(logging.Handler):
            def emit(self, record):
                pass

            def close(self):
                raise RuntimeError("close exploded")

        logging.getLogger().addHandler(Boom())
        lc_real._configure_structlog("INFO", json_logs=False)
        assert not any(isinstance(h, Boom) for h in logging.getLogger().handlers)


class TestRedactionEdges:
    def test_filter_returns_true_when_message_rendering_fails(self):
        class BadMsg:
            def __str__(self):
                raise RuntimeError("cannot render")

        rec = logging.LogRecord("n", logging.INFO, __file__, 1, BadMsg(), None, None)
        assert lc_real._SecretRedactingFilter().filter(rec) is True  # record kept, not dropped

    def test_redact_value_handles_tuples_and_plain_scalars(self):
        out = lc_real._redact_value(("token=abcd1234efgh5678", 3, None))
        assert isinstance(out, tuple) and out[1] == 3 and out[2] is None
        assert "abcd1234efgh5678" not in out[0]
        assert lc_real._redact_value(3.5) == 3.5

    def test_redact_value_mutates_mapping_in_place_and_returns_it(self):
        d = {"a": "password=hunter2hunter2", "b": [{"hf_token": "x"}]}
        assert lc_real._redact_value(d) is d
        assert "hunter2hunter2" not in d["a"]


class TestRunContextWithStructlog:
    def test_bind_unbind_roundtrip_and_default_key(self, restore_root_logging):
        import structlog

        structlog.contextvars.clear_contextvars()
        lc_real.bind_run_context("rid-1", model="m")
        assert structlog.contextvars.get_contextvars() == {"run_id": "rid-1", "model": "m"}
        lc_real.unbind_run_context()  # default: only run_id
        assert structlog.contextvars.get_contextvars() == {"model": "m"}
        lc_real.unbind_run_context("model")
        assert structlog.contextvars.get_contextvars() == {}

    def test_run_context_manager_unbinds_on_exit_even_on_error(self, restore_root_logging):
        import structlog

        structlog.contextvars.clear_contextvars()
        with pytest.raises(ValueError, match="boom"), lc_real.run_context("rid-2", phase="train"):
            assert structlog.contextvars.get_contextvars()["phase"] == "train"
            raise ValueError("boom")
        assert structlog.contextvars.get_contextvars() == {}

    def test_log_context_binds_and_unbinds(self):
        import structlog

        structlog.contextvars.clear_contextvars()
        with lc_real.LogContext(request_id="abc"):
            assert structlog.contextvars.get_contextvars() == {"request_id": "abc"}
        assert structlog.contextvars.get_contextvars() == {}

    def test_training_logger_uses_structured_kwargs(self, monkeypatch):
        calls = []

        class Rec:
            def info(self, event, **data):
                calls.append((event, data))

        monkeypatch.setattr(lc_real, "get_logger", lambda name=None: Rec())
        t = lc_real.TrainingLogger("r")
        t.log_step(1, 2.0)
        assert calls == [("train_step", {"run": "r", "step": 1, "loss": 2.0})]
