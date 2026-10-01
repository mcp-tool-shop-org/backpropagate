"""Liveness probe used by multi-run auto-resume (``multi_run._pid_alive``).

On Windows ``signal.CTRL_C_EVENT == 0``, so the old ``os.kill(pid, 0)`` probe
SENT Ctrl+C (GenerateConsoleCtrlEvent) instead of checking existence. Windows
now queries the process handle; the POSIX probe is unchanged.
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import time

import pytest

from backpropagate import multi_run
from backpropagate.multi_run import _pid_alive, _pid_alive_posix

windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows process-handle path")


@pytest.mark.parametrize("pid", [None, 0, -7])
def test_missing_or_non_positive_pid_is_dead_without_probing(monkeypatch, pid):
    def forbidden(*_a):
        raise AssertionError("no probe for an invalid pid")

    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(multi_run, "_pid_alive_windows", forbidden)
    assert _pid_alive(pid) is False


@windows_only
def test_windows_never_calls_os_kill(monkeypatch):
    def forbidden(*_a):
        raise AssertionError("os.kill(pid, 0) sends CTRL_C_EVENT on Windows")

    monkeypatch.setattr(os, "kill", forbidden)
    assert _pid_alive(os.getpid()) is True


@windows_only
def test_windows_real_processes():
    """Real processes, nothing mocked: a live child, the same child after exit, a bogus PID."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert _pid_alive(child.pid) is True
    finally:
        child.terminate()
        child.wait()
    deadline = time.time() + 5
    while _pid_alive(child.pid) and time.time() < deadline:
        time.sleep(0.1)
    assert _pid_alive(child.pid) is False
    assert _pid_alive(2**31 - 4) is False


class TestPosixProbe:
    """``os.kill`` is the mocked boundary, so these run on every OS."""

    def test_live_when_the_probe_succeeds(self, monkeypatch):
        calls = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append((pid, sig)))
        assert _pid_alive_posix(4242) is True
        assert calls == [(4242, 0)]

    def test_dead_when_no_such_process(self, monkeypatch):
        def gone(*_a):
            raise ProcessLookupError

        monkeypatch.setattr(os, "kill", gone)
        assert _pid_alive_posix(4242) is False

    def test_alive_when_owned_by_another_user(self, monkeypatch):
        def denied(*_a):
            raise PermissionError

        monkeypatch.setattr(os, "kill", denied)
        assert _pid_alive_posix(4242) is True

    @pytest.mark.parametrize("code, expected", [(errno.ESRCH, False), (errno.EIO, True)])
    def test_other_oserrors_are_judged_by_errno(self, monkeypatch, code, expected):
        def err(*_a):
            raise OSError(code, "probe failed")

        monkeypatch.setattr(os, "kill", err)
        assert _pid_alive_posix(4242) is expected


@pytest.mark.skipif(os.name == "nt", reason="POSIX dispatch")
def test_posix_dispatches_to_the_kill_probe():
    assert _pid_alive(os.getpid()) is True
