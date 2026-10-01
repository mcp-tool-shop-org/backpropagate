"""Unit tests for ``cli._run_reflex`` / ``cli._kill_process_tree`` (the ``backprop ui`` child).

Why these exist: ``subprocess.run`` blocks in ``WaitForSingleObject(INFINITE)`` on
Windows, which a console Ctrl+C cannot interrupt, so a Reflex whose shutdown hung
left ``backprop ui`` unstoppable. ``_run_reflex`` polls instead, and after a
Ctrl+C gives the Reflex tree a bounded grace period before killing it.

Mocked: ``subprocess.Popen`` (a scripted fake child, so timing is deterministic)
and, for the orchestration tests, ``_kill_process_tree``. Real: the polling loop,
the grace deadline, the exception flow, and (in ``TestKillProcessTree``) a real
parent + grandchild process pair that must both die.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from backpropagate import cli
from tests.helpers.cli_cov_support import parse


class _FakeChild:
    """A scripted ``Popen``: ``wait`` consumes ``script`` (int = exit, 'timeout', 'interrupt')."""

    def __init__(self, script, exits_after_interrupt=None):
        self.pid = 4242
        self._script = list(script)
        self.returncode = None
        self.killed = False
        self.wait_calls = 0
        self._interrupted = False
        self._exits_after_interrupt = exits_after_interrupt  # exit code once Ctrl+C has arrived

    def wait(self, timeout=None):
        self.wait_calls += 1
        step = self._script.pop(0) if self._script else "timeout"
        if step == "timeout":
            if self._exits_after_interrupt is not None and self._interrupted:
                self.returncode = self._exits_after_interrupt
                return self.returncode
            raise subprocess.TimeoutExpired("reflex", timeout or 0)
        if step == "interrupt":
            self._interrupted = True
            raise KeyboardInterrupt
        self.returncode = step
        return step

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(cli, "_UI_POLL_SECONDS", 0.01)
    monkeypatch.setattr(cli, "_UI_STOP_GRACE_SECONDS", 0.2)
    tree_kills: list = []
    monkeypatch.setattr(cli, "_kill_process_tree", lambda proc: tree_kills.append(proc.pid) or proc.kill())
    return tree_kills


def _popen_returning(child, seen):
    def fake_popen(cmd, **kwargs):
        seen.append(SimpleNamespace(cmd=cmd, kwargs=kwargs))
        return child

    return fake_popen


class TestPollingWait:
    def test_returns_the_childs_exit_code(self, monkeypatch, fast):
        child = _FakeChild(["timeout", "timeout", 3])
        seen: list = []
        monkeypatch.setattr(cli.subprocess, "Popen", _popen_returning(child, seen))
        result = cli._run_reflex(["reflex", "run"], env={"A": "1"}, cwd="/tmp/x")
        assert result.returncode == 3
        assert child.wait_calls == 3  # polled, never one unbounded wait
        assert seen[0].cmd == ["reflex", "run"]
        assert seen[0].kwargs == {"env": {"A": "1"}, "cwd": "/tmp/x"}
        assert not child.killed and fast == []

    def test_every_wait_has_a_timeout(self, monkeypatch, fast):
        timeouts: list = []

        class _Child(_FakeChild):
            def wait(self, timeout=None):
                timeouts.append(timeout)
                return super().wait(timeout)

        child = _Child(["timeout", 0])
        monkeypatch.setattr(cli.subprocess, "Popen", _popen_returning(child, []))
        cli._run_reflex(["reflex"])
        assert timeouts and all(t is not None and t <= 1.0 for t in timeouts)


class TestCtrlC:
    def test_hung_child_is_killed_after_the_grace_period(self, monkeypatch, fast):
        child = _FakeChild(["timeout", "interrupt"])  # then 'timeout' forever: shutdown hangs
        monkeypatch.setattr(cli.subprocess, "Popen", _popen_returning(child, []))
        t0 = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            cli._run_reflex(["reflex"])
        elapsed = time.monotonic() - t0
        assert fast == [child.pid]
        assert child.killed
        assert elapsed >= 0.2  # the whole grace period was honoured before killing

    def test_child_that_exits_during_grace_is_not_killed(self, monkeypatch, fast):
        child = _FakeChild(["interrupt"], exits_after_interrupt=0)
        monkeypatch.setattr(cli.subprocess, "Popen", _popen_returning(child, []))
        t0 = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            cli._run_reflex(["reflex"])
        assert fast == [] and not child.killed
        assert time.monotonic() - t0 < 0.2  # returned as soon as the child exited

    def test_second_ctrl_c_skips_the_grace_period(self, monkeypatch, fast):
        monkeypatch.setattr(cli, "_UI_STOP_GRACE_SECONDS", 30.0)
        child = _FakeChild(["interrupt", "interrupt"])
        monkeypatch.setattr(cli.subprocess, "Popen", _popen_returning(child, []))
        t0 = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            cli._run_reflex(["reflex"])
        assert fast == [child.pid]
        assert time.monotonic() - t0 < 5.0

    def test_unexpected_error_never_orphans_the_server(self, monkeypatch, fast):
        class _Boom(_FakeChild):
            def wait(self, timeout=None):
                raise RuntimeError("boom")

        child = _Boom([])
        monkeypatch.setattr(cli.subprocess, "Popen", _popen_returning(child, []))
        with pytest.raises(RuntimeError, match="boom"):
            cli._run_reflex(["reflex"])
        assert fast == [child.pid]


class TestCmdUiOnCtrlC:
    """End to end through ``cmd_ui``: Ctrl+C on a hung Reflex still ends cleanly."""

    @staticmethod
    def _harness(monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "xdg"))
        monkeypatch.setenv("BACKPROPAGATE_UI_QUIET", "1")
        monkeypatch.delenv("BACKPROPAGATE_UI_AUTH", raising=False)
        monkeypatch.setattr(cli, "_find_port_in_use", lambda host, ports: None)

    def test_ui_stopped_lock_removed_and_tree_killed(self, monkeypatch, tmp_path, capsys, fast):
        self._harness(monkeypatch, tmp_path)
        child = _FakeChild(["timeout", "interrupt"])
        monkeypatch.setattr(cli.subprocess, "Popen", _popen_returning(child, []))
        assert cli.cmd_ui(parse(["ui"])) == cli.EXIT_OK
        assert "UI stopped" in capsys.readouterr().out
        assert fast == [child.pid]
        lock_dir = tmp_path / "xdg" / "backpropagate"
        assert not lock_dir.exists() or list(lock_dir.glob("session-*.lock")) == []

    def test_normal_exit_returns_the_exit_code(self, monkeypatch, tmp_path, fast):
        self._harness(monkeypatch, tmp_path)
        child = _FakeChild(["timeout", 7])
        monkeypatch.setattr(cli.subprocess, "Popen", _popen_returning(child, []))
        assert cli.cmd_ui(parse(["ui"])) == 7
        assert fast == []


_PARENT_WITH_GRANDCHILD = textwrap.dedent(
    """
    import subprocess, sys, time
    kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    print(kid.pid, flush=True)
    time.sleep(120)
    """
)


class TestKillProcessTree:
    def test_kills_the_child_and_its_grandchild(self):
        psutil = pytest.importorskip("psutil")
        parent = subprocess.Popen(
            [sys.executable, "-c", _PARENT_WITH_GRANDCHILD], stdout=subprocess.PIPE, text=True
        )
        try:
            grandchild_pid = int(parent.stdout.readline())
            assert psutil.pid_exists(grandchild_pid)
            cli._kill_process_tree(parent)
            parent.wait(timeout=15)
            deadline = time.monotonic() + 10
            while psutil.pid_exists(grandchild_pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            assert not psutil.pid_exists(grandchild_pid), "grandchild (the server) outlived the kill"
        finally:
            if parent.poll() is None:
                parent.kill()
            parent.stdout.close()

    def test_already_dead_child_is_fine(self):
        parent = subprocess.Popen([sys.executable, "-c", "pass"])
        parent.wait(timeout=30)
        cli._kill_process_tree(parent)  # must not raise
