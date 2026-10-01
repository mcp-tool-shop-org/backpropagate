"""Coverage tests for ``cmd_ui`` in cli.py.

Mocked (real boundaries): ``cli._run_reflex`` (the Reflex server child),
``_spawn_cloudflared_tunnel`` (network tunnel; the helper itself is covered in
``test_cli_cov_ui_support.py``) and the port pre-flight (``_find_port_in_use``,
also covered for real in that file). Real: argument parsing, the auth-file
resolver, auth/host gates, lock-file write + cleanup (into ``tmp_path`` via
``XDG_RUNTIME_DIR``), env handed to the child, exit codes and output.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from backpropagate import cli
from backpropagate.exceptions import BackpropagateError, UserInputError
from tests.helpers.cli_cov_support import parse
from tests.helpers.ui_auth import assert_child_env_has_verifier


class _ExplodingLogger:
    def info(self, *a, **k):
        raise RuntimeError("logger down")

    debug = info


@pytest.fixture
def ui(tmp_path, monkeypatch):
    """Harness: a fake Reflex child, free ports, lock files redirected to tmp_path."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "xdg"))
    monkeypatch.setenv("BACKPROPAGATE_UI_QUIET", "1")
    monkeypatch.delenv("BACKPROPAGATE_UI_AUTH", raising=False)
    monkeypatch.setattr(cli, "_find_port_in_use", lambda host, ports: None)
    calls: dict = {"run": []}

    def fake_run(cmd, env=None, cwd=None):
        calls["run"].append(SimpleNamespace(cmd=cmd, env=env, cwd=cwd))
        # While the "UI" runs, the lock file (if any) must exist.
        lock_dir = tmp_path / "xdg" / "backpropagate"
        calls["locks_during"] = sorted(p.name for p in lock_dir.glob("session-*.lock")) if lock_dir.exists() else []
        return SimpleNamespace(returncode=calls.get("returncode", 0))

    monkeypatch.setattr(cli, "_run_reflex", fake_run)
    calls["lock_dir"] = tmp_path / "xdg" / "backpropagate"
    return calls


class TestPrerequisites:
    def test_reflex_missing(self, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "reflex", None)
        args = parse(["ui"])
        args.verbose = True
        assert cli.cmd_ui(args) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "UI dependencies not installed" in captured.err
        assert "pip install backpropagate[ui]" in captured.out

    def test_ui_security_not_importable(self, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "backpropagate.ui_security", None)
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_USER_ERROR
        assert "UI security helpers not importable" in capsys.readouterr().err

    def test_enforcement_unavailable_without_auth_still_launches(self, monkeypatch, ui):
        monkeypatch.setitem(sys.modules, "backpropagate.ui_app.auth", None)
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_OK
        assert len(ui["run"]) == 1

    def test_enforcement_unavailable_with_auth_refuses(self, monkeypatch, ui):
        monkeypatch.setitem(sys.modules, "backpropagate.ui_app.auth", None)
        with pytest.raises(BackpropagateError) as ei:
            cli.cmd_ui(parse(["ui", "--auth", "alice:secret123"]))
        assert ei.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
        assert "refusing to launch an unprotected UI" in ei.value.message
        assert ui["run"] == []


class TestAuthFile:
    def _file(self, tmp_path, text):
        f = tmp_path / "auth.txt"
        f.write_text(text, encoding="utf-8")
        return str(f)

    def test_mutex_with_inline_auth(self, tmp_path, ui):
        args = parse(["ui", "--auth", "a:b", "--auth-file", self._file(tmp_path, "a:b")])
        with pytest.raises(UserInputError, match="mutually exclusive") as ei:
            cli.cmd_ui(args)
        assert ei.value.code == "INPUT_VALIDATION_FAILED"

    def test_missing_file(self, tmp_path, ui):
        with pytest.raises(UserInputError, match="path does not exist"):
            cli.cmd_ui(parse(["ui", "--auth-file", str(tmp_path / "nope")]))

    def test_empty_file(self, tmp_path, ui):
        with pytest.raises(UserInputError, match="is empty"):
            cli.cmd_ui(parse(["ui", "--auth-file", self._file(tmp_path, "\n")]))

    def test_invalid_shape(self, tmp_path, ui):
        with pytest.raises(UserInputError, match="failed validation") as ei:
            cli.cmd_ui(parse(["ui", "--auth-file", self._file(tmp_path, "no-colon-here")]))
        assert ei.value.code == "INPUT_AUTH_INVALID_SHAPE"

    def test_unreadable_file(self, tmp_path, monkeypatch, ui):
        path = self._file(tmp_path, "a:b")

        def deny(self, *a, **k):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "read_text", deny)
        with pytest.raises(UserInputError, match="could not be read: denied"):
            cli.cmd_ui(parse(["ui", "--auth-file", path]))

    def test_valid_file_feeds_child_env_and_lock_file(self, tmp_path, ui, capsys):
        args = parse(["ui", "--port", "7900", "--auth-file", self._file(tmp_path, " alice:s3cret \n")])
        assert cli.cmd_ui(args) == cli.EXIT_OK
        run = ui["run"][0]
        assert_child_env_has_verifier(run.env, "alice", "s3cret")
        assert run.env["BACKPROPAGATE_UI_PORT"] == "7900"
        assert run.env["BACKPROPAGATE_UI_HOST_BIND"] == "127.0.0.1"
        assert run.cmd[-2:] == ["--backend-host", "127.0.0.1"]
        # production mode on ONE port: the dev backend cannot start on this package layout
        assert run.cmd[run.cmd.index("--env") + 1] == "prod"
        assert run.cmd[run.cmd.index("--frontend-port") + 1] == "7900"
        assert run.cmd[run.cmd.index("--backend-port") + 1] == "7900"
        assert "7901" not in run.cmd
        out = capsys.readouterr().out
        assert "lock-file" not in out  # credentials are never persisted, so no lock file
        assert "inline" not in out  # the --auth-file path never prints the inline-credential warning
        assert ui["locks_during"] == []
        assert list(ui["lock_dir"].glob("session-*.lock")) == []

    def test_wide_posix_mode_warns(self, tmp_path, monkeypatch, ui, capsys):
        """On POSIX a group/other-readable credential file triggers a warning (Windows stat reports 0o666)."""

        class PosixOs:
            name = "posix"

            def __getattr__(self, item):
                import os

                return getattr(os, item)

        monkeypatch.setattr(cli, "os", PosixOs())
        monkeypatch.setattr(Path, "stat", lambda self, **k: SimpleNamespace(st_mode=0o100644))
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr(Path, "read_text", lambda self, **k: "alice:pw")
        monkeypatch.setattr(cli, "write_launch_token_lock", lambda port, payload: tmp_path / "lock")
        monkeypatch.setattr(Path, "unlink", lambda self, **k: None)
        assert cli.cmd_ui(parse(["ui", "--auth-file", "creds"])) == cli.EXIT_OK
        assert "mode is 0o644" in capsys.readouterr().out

    def test_posix_stat_error_is_ignored(self, tmp_path, monkeypatch, ui):
        class PosixOs:
            name = "posix"

            def __getattr__(self, item):
                import os

                return getattr(os, item)

        path = self._file(tmp_path, "alice:pw")
        real_stat = Path.stat
        calls = {"n": 0}

        def flaky(self, **k):
            calls["n"] += 1
            if calls["n"] == 2:  # exists() uses the first; the mode probe is the second
                raise OSError("stat failed")
            return real_stat(self, **k)

        monkeypatch.setattr(cli, "os", PosixOs())
        monkeypatch.setattr(Path, "stat", flaky)
        monkeypatch.setattr(cli, "write_launch_token_lock", lambda port, payload: tmp_path / "lock")
        monkeypatch.setattr(Path, "unlink", lambda self, **k: None)
        assert cli.cmd_ui(parse(["ui", "--auth-file", path])) == cli.EXIT_OK
        assert_child_env_has_verifier(ui["run"][0].env, "alice", "pw")


class TestGates:
    def test_inline_auth_warns(self, ui, capsys):
        assert cli.cmd_ui(parse(["ui", "--auth", "bob:pw"])) == cli.EXIT_OK
        assert "--auth was passed inline" in capsys.readouterr().out

    def test_share_requires_auth(self, ui):
        with pytest.raises(BackpropagateError) as ei:
            cli.cmd_ui(parse(["ui", "--share"]))
        assert ei.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
        assert "--share requires --auth" in ei.value.message

    def test_non_loopback_host_requires_auth(self, ui):
        with pytest.raises(BackpropagateError) as ei:
            cli.cmd_ui(parse(["ui", "--host", "0.0.0.0"]))
        assert ei.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
        assert "'0.0.0.0'" in ei.value.message
        assert ui["run"] == []

    def test_non_loopback_host_with_auth_sets_bind(self, ui):
        assert cli.cmd_ui(parse(["ui", "--host", "0.0.0.0", "--auth", "u:p"])) == cli.EXIT_OK
        run = ui["run"][0]
        assert run.env["BACKPROPAGATE_UI_HOST_BIND"] == "0.0.0.0"
        assert run.cmd[-2:] == ["--backend-host", "0.0.0.0"]

    def test_auth_without_colon_direct_call(self, ui, capsys):
        args = parse(["ui"])
        args.auth = "nocolon"
        assert cli.cmd_ui(args) == cli.EXIT_USER_ERROR
        assert "Invalid auth format" in capsys.readouterr().err

    def test_auth_shape_rejected_by_validator(self, ui, capsys):
        args = parse(["ui"])
        args.auth = ":password-without-user"
        assert cli.cmd_ui(args) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "[INPUT_AUTH_INVALID_SHAPE]" in captured.err or "non-empty" in captured.err
        assert ui["run"] == []

    def test_ambient_auth_env_is_stripped_without_flag(self, ui, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "ambient:bypass")
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH_VERIFIER", "scrypt$ambient")
        monkeypatch.setenv("BACKPROPAGATE_UI_LAUNCH_TOKEN", "ambient-token")
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_OK
        env = ui["run"][0].env
        assert "BACKPROPAGATE_UI_AUTH" not in env
        assert "BACKPROPAGATE_UI_AUTH_VERIFIER" not in env
        assert env["BACKPROPAGATE_UI_LAUNCH_TOKEN"] != "ambient-token"  # a fresh per-launch token
        assert ui["locks_during"] == ["session-7862.lock"]  # the token lock file, not a credential

    def test_rxconfig_missing(self, ui, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "__file__", str(tmp_path / "pkg" / "cli.py"))
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "rxconfig.py not found" in captured.err
        assert "reinstall the [ui] extra" in captured.out
        assert ui["run"] == []

    def test_port_in_use_raises(self, ui, monkeypatch):
        probed: list[list[int]] = []

        def busy(host, ports):
            probed.append(list(ports))
            return ports[0]

        monkeypatch.setattr(cli, "_find_port_in_use", busy)
        with pytest.raises(BackpropagateError) as ei:
            cli.cmd_ui(parse(["ui", "--port", "7950"]))
        assert ei.value.code == "RUNTIME_UI_PORT_IN_USE"
        assert "7950" in ei.value.message
        assert probed == [[7950]]  # one port: there is no N+1 backend any more
        assert "+ 1" not in ei.value.message and "N+1" not in (ei.value.suggestion or "")

    def test_reflex_runs_in_prod_mode_on_a_single_port(self, ui):
        assert cli.cmd_ui(parse(["ui", "--port", "7940"])) == cli.EXIT_OK
        cmd = ui["run"][0].cmd
        assert cmd[1:4] == ["-m", "reflex", "run"]
        assert cmd[cmd.index("--env") + 1] == "prod"
        assert cmd[cmd.index("--frontend-port") + 1] == cmd[cmd.index("--backend-port") + 1] == "7940"
        assert "7941" not in cmd


class _FakeTunnelProc:
    def __init__(self, wait_behaviour=("ok",), terminate_exc=None):
        self.terminated = False
        self.killed = False
        self.returncode = 0
        self._wait = list(wait_behaviour)
        self._terminate_exc = terminate_exc

    def terminate(self):
        self.terminated = True
        if self._terminate_exc:
            raise self._terminate_exc

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        step = self._wait.pop(0) if self._wait else "ok"
        if step == "timeout":
            raise subprocess.TimeoutExpired("cloudflared", timeout)
        return 0


class TestShare:
    def test_spawn_failure_is_user_error(self, ui, monkeypatch):
        monkeypatch.setattr(cli, "_spawn_cloudflared_tunnel", lambda port: None)
        assert cli.cmd_ui(parse(["ui", "--share", "--auth", "u:p"])) == cli.EXIT_USER_ERROR
        assert ui["run"] == []

    def test_tunnel_host_exported_and_proc_terminated(self, ui, monkeypatch, capsys):
        proc = _FakeTunnelProc()
        monkeypatch.setattr(cli, "_spawn_cloudflared_tunnel",
                            lambda port: (proc, "https://Cool-Name-1.trycloudflare.com/path"))
        assert cli.cmd_ui(parse(["ui", "--share", "--auth", "u:p"])) == cli.EXIT_OK
        assert ui["run"][0].env["BACKPROPAGATE_UI_SHARE_HOST"] == "cool-name-1.trycloudflare.com"
        assert "Tunnel ready: https://Cool-Name-1.trycloudflare.com/path" in capsys.readouterr().out
        assert proc.terminated and not proc.killed

    def test_unparsable_tunnel_url_fails_closed(self, ui, monkeypatch, capsys):
        proc = _FakeTunnelProc(terminate_exc=OSError("already dead"))
        monkeypatch.setattr(cli, "_spawn_cloudflared_tunnel", lambda port: (proc, "not a url"))
        assert cli.cmd_ui(parse(["ui", "--share", "--auth", "u:p"])) == cli.EXIT_RUNTIME_ERROR
        assert "Could not parse the trycloudflare.com hostname" in capsys.readouterr().err
        assert ui["run"] == []

    def test_urlsplit_failure_fails_closed(self, ui, monkeypatch, capsys):
        proc = _FakeTunnelProc()
        monkeypatch.setattr(cli, "_spawn_cloudflared_tunnel",
                            lambda port: (proc, "https://x.trycloudflare.com"))

        def boom(url):
            raise ValueError("bad url")

        monkeypatch.setattr("urllib.parse.urlsplit", boom)
        assert cli.cmd_ui(parse(["ui", "--share", "--auth", "u:p"])) == cli.EXIT_RUNTIME_ERROR
        assert proc.terminated

    def test_sigkill_escalation_when_terminate_hangs(self, ui, monkeypatch):
        proc = _FakeTunnelProc(wait_behaviour=("timeout", "ok"))
        monkeypatch.setattr(cli, "_spawn_cloudflared_tunnel",
                            lambda port: (proc, "https://x.trycloudflare.com"))
        assert cli.cmd_ui(parse(["ui", "--share", "--auth", "u:p"])) == cli.EXIT_OK
        assert proc.killed

    def test_zombie_tunnel_does_not_propagate_timeout(self, ui, monkeypatch):
        proc = _FakeTunnelProc(wait_behaviour=("timeout", "timeout"))
        monkeypatch.setattr(cli, "_spawn_cloudflared_tunnel",
                            lambda port: (proc, "https://x.trycloudflare.com"))
        assert cli.cmd_ui(parse(["ui", "--share", "--auth", "u:p"])) == cli.EXIT_OK
        assert proc.killed

    def test_exploding_logger_during_cleanup_is_swallowed(self, ui, monkeypatch):
        monkeypatch.setattr("backpropagate.logging_config.get_logger", lambda name: _ExplodingLogger())
        for waits in (("ok",), ("timeout", "ok"), ("timeout", "timeout")):
            proc = _FakeTunnelProc(wait_behaviour=waits)
            monkeypatch.setattr(cli, "_spawn_cloudflared_tunnel",
                                lambda port, p=proc: (p, "https://x.trycloudflare.com"))
            assert cli.cmd_ui(parse(["ui", "--share", "--auth", "u:p"])) == cli.EXIT_OK
            assert proc.terminated


class TestLaunchOutcomes:
    def test_child_exit_code_is_propagated(self, ui):
        ui["returncode"] = 7
        assert cli.cmd_ui(parse(["ui"])) == 7

    def test_none_returncode_means_ok(self, ui):
        ui["returncode"] = None
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_OK

    def test_lock_file_failure_is_a_warning_only(self, ui, monkeypatch, capsys):
        def boom(port, payload):
            raise OSError("disk full")

        monkeypatch.setattr(cli, "write_launch_token_lock", boom)
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Could not write launch lock-file (disk full)" in out
        assert len(ui["run"]) == 1

    def test_exploding_logger_never_blocks_launch(self, ui, monkeypatch, tmp_path):
        monkeypatch.setattr("backpropagate.logging_config.get_logger", lambda name: _ExplodingLogger())
        assert cli.cmd_ui(parse(["ui", "--auth", "u:p"])) == cli.EXIT_OK
        assert list(ui["lock_dir"].glob("session-*.lock")) == []

    def test_keyboard_interrupt(self, ui, monkeypatch, capsys):
        def interrupted(cmd, env=None, cwd=None):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_run_reflex", interrupted)
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_OK
        assert "UI stopped" in capsys.readouterr().out

    def test_keyboard_interrupt_with_exploding_logger(self, ui, monkeypatch):
        monkeypatch.setattr("backpropagate.logging_config.get_logger", lambda name: _ExplodingLogger())

        def interrupted(cmd, env=None, cwd=None):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_run_reflex", interrupted)
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_OK

    def test_interpreter_missing(self, ui, monkeypatch, capsys):
        def missing(cmd, env=None, cwd=None):
            raise FileNotFoundError("python")

        monkeypatch.setattr(cli, "_run_reflex", missing)
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "interpreter not found" in captured.err
        assert "pip install backpropagate[ui]" in captured.out

    def test_interpreter_missing_with_exploding_logger(self, ui, monkeypatch):
        monkeypatch.setattr("backpropagate.logging_config.get_logger", lambda name: _ExplodingLogger())

        def missing(cmd, env=None, cwd=None):
            raise FileNotFoundError("python")

        monkeypatch.setattr(cli, "_run_reflex", missing)
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_USER_ERROR

    def test_user_input_error_from_launch(self, ui, monkeypatch, capsys):
        def bad(cmd, env=None, cwd=None):
            raise UserInputError("bad flag", hint="fix the flag")

        monkeypatch.setattr(cli, "_run_reflex", bad)
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "bad flag" in captured.err and "Suggestion: fix the flag" in captured.out

    @pytest.mark.parametrize("verbose", [False, True])
    def test_backpropagate_error_from_launch(self, ui, monkeypatch, capsys, verbose):
        def bad(cmd, env=None, cwd=None):
            raise BackpropagateError("reflex exploded", suggestion="reinstall")

        monkeypatch.setattr(cli, "_run_reflex", bad)
        args = parse(["ui"])
        args.verbose = verbose
        assert cli.cmd_ui(args) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "reflex exploded" in captured.err and "Suggestion: reinstall" in captured.out

    def test_unexpected_error_redacted_then_verbose(self, ui, monkeypatch, capsys):
        def bad(cmd, env=None, cwd=None):
            raise RuntimeError("Authorization: Bearer abcdef1234567890")

        monkeypatch.setattr(cli, "_run_reflex", bad)
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "abcdef1234567890" not in captured.err
        assert "Run with --verbose" in captured.out

        args = parse(["ui"])
        args.verbose = True
        assert cli.cmd_ui(args) == cli.EXIT_RUNTIME_ERROR
        assert "Traceback" in capsys.readouterr().err
