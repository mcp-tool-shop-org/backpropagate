"""Coverage tests for ``cli.main`` (entry point, logging wiring, last-resort
exception net / sysexits mapping) and the ``ollama`` bare-verb handler.

Real: the real parser, real ``main`` flow, exit codes, stderr text.
Mocked (real boundaries only where noted): ``cmd_*`` handlers are swapped for
tiny functions that raise the exception under test (they are the injected
fault, not the code being tested), and ``configure_logging`` / log binding are
made to fail to exercise the logging-failure warning.
"""

from __future__ import annotations

import sys

import pytest

from backpropagate import cli
from backpropagate.exceptions import BackpropagateError, DatasetError, UserInputError


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """main() writes BACKPROPAGATE_LOG_* env vars; make monkeypatch restore them."""
    for var in ("BACKPROPAGATE_LOG_LEVEL", "BACKPROPAGATE_LOG_JSON", "BACKPROPAGATE_LOG_FILE",
                "BACKPROPAGATE_DEBUG"):
        # setenv-then-delenv (not a bare delenv) so monkeypatch records "absent" as the original
        # state and removes whatever main() writes into os.environ when the test ends.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)


def _raising_config(monkeypatch, exc):
    def boom(args):
        raise exc

    monkeypatch.setattr(cli, "cmd_config", boom)


class TestNoSubcommand:
    def test_bare_invocation_prints_help(self, capsys):
        assert cli.main([]) == cli.EXIT_USER_ERROR
        err = capsys.readouterr().err
        assert "usage:" in err and "train" in err

    def test_bare_ollama_prints_ollama_help(self, capsys):
        assert cli.main(["ollama"]) == cli.EXIT_USER_ERROR
        err = capsys.readouterr().err
        assert "register" in err and "shelf" in err

    def test_help_flag_propagates_system_exit(self):
        with pytest.raises(SystemExit) as ei:
            cli.main(["--help"])
        assert ei.value.code == 0

    def test_subcommand_system_exit_propagates(self, monkeypatch):
        _raising_config(monkeypatch, SystemExit(7))
        with pytest.raises(SystemExit) as ei:
            cli.main(["config"])
        assert ei.value.code == 7


class TestArgcomplete:
    def test_missing_argcomplete_is_ignored(self, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "argcomplete", None)
        assert cli.main(["config"]) == cli.EXIT_OK
        assert "Backpropagate Configuration" in capsys.readouterr().out


class TestParseInterrupt:
    def test_ctrl_c_during_parse(self, monkeypatch, capsys):
        class P:
            def parse_args(self, argv):
                raise KeyboardInterrupt

        monkeypatch.setattr(cli, "create_parser", lambda: P())
        assert cli.main(["config"]) == cli.EXIT_INTERRUPTED
        assert "Interrupted (run_id=" in capsys.readouterr().err


class TestLoggingFlags:
    def test_root_flags_overwrite_env_and_configure_logging(self, monkeypatch, capsys):
        seen = {}

        def fake_configure(level=None, force=False, **kw):
            seen["level"] = level
            seen["force"] = force

        monkeypatch.setattr("backpropagate.logging_config.configure_logging", fake_configure)
        argv = ["--log-level", "WARNING", "--log-format", "json", "--log-file", "x.log", "config"]
        assert cli.main(argv) == cli.EXIT_OK
        import os

        assert os.environ["BACKPROPAGATE_LOG_LEVEL"] == "WARNING"
        assert os.environ["BACKPROPAGATE_LOG_JSON"] == "true"
        assert os.environ["BACKPROPAGATE_LOG_FILE"] == "x.log"
        assert seen == {"level": None, "force": True}

    def test_console_format_and_verbose_after_subcommand(self, monkeypatch):
        seen = {}
        monkeypatch.setattr("backpropagate.logging_config.configure_logging",
                            lambda level=None, force=False, **kw: seen.update(level=level))
        assert cli.main(["config", "--log-format", "console", "--verbose"]) == cli.EXIT_OK
        import os

        assert os.environ["BACKPROPAGATE_LOG_JSON"] == "false"
        assert seen["level"] == "DEBUG"

    def test_configure_logging_failure_is_warned_not_fatal(self, monkeypatch, capsys):
        def boom(**kw):
            raise OSError("log dir missing")

        monkeypatch.setattr("backpropagate.logging_config.configure_logging", boom)
        assert cli.main(["config"]) == cli.EXIT_OK
        captured = capsys.readouterr()
        assert "[WARN] structured logging setup failed: OSError: log dir missing" in captured.err
        assert "Backpropagate Configuration" in captured.out

    def test_configure_logging_failure_with_debug_prints_traceback(self, monkeypatch, capsys):
        def boom(**kw):
            raise OSError("log dir missing")

        monkeypatch.setattr("backpropagate.logging_config.configure_logging", boom)
        monkeypatch.setenv("BACKPROPAGATE_DEBUG", "1")
        assert cli.main(["config"]) == cli.EXIT_OK
        assert "Traceback" in capsys.readouterr().err

    def test_bind_context_failure_is_warned(self, monkeypatch, capsys):
        def boom(**kw):
            raise RuntimeError("no contextvars")

        monkeypatch.setattr("backpropagate.logging_config.bind_run_context", boom)
        assert cli.main(["config"]) == cli.EXIT_OK
        assert "bind_run_context: RuntimeError: no contextvars" in capsys.readouterr().err

    def test_cli_invoked_emit_failure_is_warned(self, monkeypatch, capsys):
        class BadLogger:
            def info(self, *a, **k):
                raise RuntimeError("sink closed")

        monkeypatch.setattr("backpropagate.logging_config.get_logger", lambda name=None: BadLogger())
        assert cli.main(["config"]) == cli.EXIT_OK
        assert "cli_invoked emit: RuntimeError: sink closed" in capsys.readouterr().err

    def test_only_first_logging_failure_is_reported(self, monkeypatch, capsys):
        def boom(**kw):
            raise OSError("first failure")

        class BadLogger:
            def info(self, *a, **k):
                raise RuntimeError("second failure")

        monkeypatch.setattr("backpropagate.logging_config.configure_logging", boom)
        monkeypatch.setattr("backpropagate.logging_config.bind_run_context", boom)
        monkeypatch.setattr("backpropagate.logging_config.get_logger", lambda name=None: BadLogger())
        assert cli.main(["config"]) == cli.EXIT_OK
        err = capsys.readouterr().err
        assert "first failure" in err and "second failure" not in err


class TestDeprecation:
    def test_deprecated_subcommand_prints_hint_and_still_runs(self, tmp_path, capsys):
        assert cli.SUBCOMMAND_TIERS["list-runs"].startswith("deprecated")
        assert cli.main(["list-runs", "--output", str(tmp_path / "none")]) == cli.EXIT_OK
        err = capsys.readouterr().err
        assert "[deprecation] `backprop list-runs` is deprecated; prefer `backprop runs`." in err

    def test_deprecated_tier_without_replacement_name(self, monkeypatch, capsys):
        monkeypatch.setitem(cli.SUBCOMMAND_TIERS, "config", "deprecated")
        assert cli.main(["config"]) == cli.EXIT_OK
        assert "prefer `backprop the replacement subcommand`" in capsys.readouterr().err


class TestLastResortNet:
    def test_keyboard_interrupt(self, monkeypatch, capsys):
        _raising_config(monkeypatch, KeyboardInterrupt())
        assert cli.main(["config"]) == cli.EXIT_INTERRUPTED
        assert "Interrupted (run_id=" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "exc, expected",
        [
            (UserInputError("bad flag"), cli.EXIT_USAGE),
            (DatasetError("bad data"), cli.EXIT_DATA_ERR),
            (BackpropagateError("oom", code="RUNTIME_GPU_OOM"), cli.EXIT_OOM_KILLED),
            (BackpropagateError("oom", code="RUNTIME_OOM_RECOVERY_EXHAUSTED"), cli.EXIT_OOM_KILLED),
            (BackpropagateError("net", code="HUB_PUSH_NETWORK"), cli.EXIT_UNAVAILABLE),
            (BackpropagateError("gone", code="DEP_OLLAMA_REGISTRATION_FAILED"), cli.EXIT_UNAVAILABLE),
            (BackpropagateError("misc", code="RUNTIME_TRAINING_FAILED"), cli.EXIT_RUNTIME_ERROR),
            (BackpropagateError("nocode"), cli.EXIT_RUNTIME_ERROR),
        ],
    )
    def test_structured_error_mapping(self, monkeypatch, capsys, exc, expected):
        _raising_config(monkeypatch, exc)
        assert cli.main(["config"]) == expected
        err = capsys.readouterr().err
        assert "[ERROR]" in err and str(exc.message) in err

    def test_structured_error_message_is_redacted_and_debug_prints_traceback(self, monkeypatch, capsys):
        _raising_config(monkeypatch, BackpropagateError("Authorization: Bearer abcdef1234567890", code="RUNTIME_TRAINING_FAILED"))
        assert cli.main(["config"]) == cli.EXIT_RUNTIME_ERROR
        err = capsys.readouterr().err
        error_line = next(line for line in err.splitlines() if line.startswith("[ERROR]"))
        assert "RUNTIME_TRAINING_FAILED:" in error_line and "abcdef1234567890" not in error_line
        assert "Traceback" not in err

        monkeypatch.setenv("BACKPROPAGATE_DEBUG", "1")
        assert cli.main(["config"]) == cli.EXIT_RUNTIME_ERROR
        assert "Traceback" in capsys.readouterr().err

    def _plain_os_error(self, errno_value):
        e = OSError("odd failure")
        e.errno = errno_value  # a bare OSError (not auto-promoted to PermissionError)
        return e

    @pytest.mark.parametrize(
        "exc, expected",
        [
            (RuntimeError("boom"), cli.EXIT_SOFTWARE),
            (PermissionError("denied"), cli.EXIT_NO_PERM),
            (ConnectionError("down"), cli.EXIT_UNAVAILABLE),
            (TimeoutError("slow"), cli.EXIT_UNAVAILABLE),
            (type("OutOfMemoryError", (RuntimeError,), {})("cuda oom"), cli.EXIT_OOM_KILLED),
            (type("CUDAOutOfMemoryError", (RuntimeError,), {})("cuda oom"), cli.EXIT_OOM_KILLED),
            (type("HTTPError", (RuntimeError,), {})("503"), cli.EXIT_UNAVAILABLE),
            (type("ReadTimeoutThing", (RuntimeError,), {})("slow"), cli.EXIT_UNAVAILABLE),
            (type("HostUnreachable", (RuntimeError,), {})("nope"), cli.EXIT_UNAVAILABLE),
        ],
    )
    def test_unexpected_exception_mapping(self, monkeypatch, capsys, exc, expected):
        _raising_config(monkeypatch, exc)
        assert cli.main(["config"]) == expected
        err = capsys.readouterr().err
        assert f"Unexpected error: {type(exc).__name__}" in err
        assert "BACKPROPAGATE_DEBUG=1" in err

    @pytest.mark.parametrize("errno_value", [13, 1])
    def test_bare_oserror_with_permission_errno(self, monkeypatch, errno_value):
        _raising_config(monkeypatch, self._plain_os_error(errno_value))
        assert cli.main(["config"]) == cli.EXIT_NO_PERM

    def test_bare_oserror_with_other_errno_is_software(self, monkeypatch):
        _raising_config(monkeypatch, self._plain_os_error(5))
        assert cli.main(["config"]) == cli.EXIT_SOFTWARE

    def test_unexpected_exception_redacted_and_debug_traceback(self, monkeypatch, capsys):
        _raising_config(monkeypatch, RuntimeError("Authorization: Bearer abcdef1234567890"))
        assert cli.main(["config"]) == cli.EXIT_SOFTWARE
        err = capsys.readouterr().err
        assert "abcdef1234567890" not in err

        monkeypatch.setenv("BACKPROPAGATE_DEBUG", "1")
        assert cli.main(["config"]) == cli.EXIT_SOFTWARE
        err = capsys.readouterr().err
        assert "Traceback" in err and "BACKPROPAGATE_DEBUG=1 for the full traceback" not in err


class TestParserWiring:
    def test_subcommands_all_registered_with_handlers(self):
        parser = cli.create_parser()
        for argv in (["train", "--data", "x"], ["multi-run", "--data", "x"], ["export"], ["info"], ["config"],
                     ["ui"], ["validate", "f"], ["estimate-vram", "m"], ["runs"], ["list-runs"],
                     ["show-run", "r"], ["diff-runs", "a", "b"], ["replay", "r"], ["export-runs"],
                     ["data", "report", "f"], ["data", "split", "f"], ["eval", "r"], ["generate", "a", "p"],
                     ["ollama", "list"], ["push", "p", "--repo", "a/b"], ["resume", "r"]):
            ns = parser.parse_args(argv)
            assert callable(ns.func), argv

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as ei:
            cli.create_parser().parse_args(["--version"])
        assert ei.value.code == 0
        assert cli.__version__ in capsys.readouterr().out

    def test_invalid_numeric_flag_exits_2(self, capsys):
        with pytest.raises(SystemExit) as ei:
            cli.create_parser().parse_args(["train", "--data", "x", "--steps", "0"])
        assert ei.value.code == 2
        assert "must be positive" in capsys.readouterr().err

    def test_common_flags_backfilled_when_absent(self, monkeypatch):
        seen = {}

        def capture(args):
            seen.update(vars(args))
            return cli.EXIT_OK

        monkeypatch.setattr(cli, "cmd_config", capture)
        assert cli.main(["config"]) == cli.EXIT_OK
        for dest, default in cli._COMMON_FLAG_DEFAULTS.items():
            assert seen[dest] == default
        assert isinstance(seen["cli_run_id"], str) and len(seen["cli_run_id"]) == 32
