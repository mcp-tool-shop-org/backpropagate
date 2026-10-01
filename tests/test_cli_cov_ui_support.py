"""Coverage tests for the ``backprop ui`` support helpers in cli.py.

Covers ``_spawn_cloudflared_tunnel``, ``_find_port_in_use`` and the
launch-token lock-file helpers.

Boundary handling: ``cloudflared`` is an external tool, so ``subprocess.Popen``
is redirected to a *real* short-lived ``python -c`` child with the same
stdout-pipe contract (real process, real pipe, real reader thread); only the
executable is substituted. Sockets are real (a genuinely occupied port on
127.0.0.1); only the EACCES case uses a stub socket. Lock files are written for
real into ``tmp_path``.
"""

from __future__ import annotations

import errno
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from backpropagate import cli

_REAL_POPEN = subprocess.Popen


def _patch_cloudflared(monkeypatch, script: str):
    """Make cli's Popen spawn ``python -c script`` instead of cloudflared."""
    spawned: list = []

    def fake_popen(cmd, **kw):
        spawned.append(cmd)
        return _REAL_POPEN([sys.executable, "-u", "-c", script], **kw)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/cloudflared")
    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    return spawned


class _PosixOs:
    """Proxy for the ``os`` module as seen by cli.py only: reports ``name == "posix"``
    and records chmod calls (delegating to the real chmod), everything else real."""

    name = "posix"

    def __init__(self):
        import os as _os

        self._os = _os
        self.chmods: list = []

    def chmod(self, path, mode):
        self.chmods.append((Path(path), mode))
        self._os.chmod(path, mode)

    def __getattr__(self, item):
        return getattr(self._os, item)


class _ExplodingLogger:
    def info(self, *a, **k):
        raise RuntimeError("logger down")

    debug = info


class TestSpawnCloudflared:
    def test_not_on_path(self, monkeypatch, capsys):
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert cli._spawn_cloudflared_tunnel(7860) is None
        captured = capsys.readouterr()
        assert "requires `cloudflared` on PATH" in captured.err
        assert "ssh -L 7860:localhost:7860" in captured.out

    def test_spawn_oserror(self, monkeypatch, capsys):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/cloudflared")

        def boom(*a, **k):
            raise OSError("exec format error")

        monkeypatch.setattr(cli.subprocess, "Popen", boom)
        assert cli._spawn_cloudflared_tunnel(7861) is None
        captured = capsys.readouterr()
        assert "Failed to spawn cloudflared: exec format error" in captured.err
        assert "cloudflared --version" in captured.out

    def test_url_parsed_and_post_url_lines_drained(self, monkeypatch):
        script = (
            "import sys, time\n"
            "print('INF starting', flush=True)\n"
            "print('INF |  https://quiet-river-1234.trycloudflare.com  |', flush=True)\n"
            "time.sleep(0.4)\n"
            "print('INF later chatter', flush=True)\n"
            "time.sleep(0.4)\n"
        )
        spawned = _patch_cloudflared(monkeypatch, script)
        result = cli._spawn_cloudflared_tunnel(7862)
        assert result is not None
        proc, url = result
        try:
            assert url == "https://quiet-river-1234.trycloudflare.com"
            assert spawned[0][-2:] == ["--url", "http://localhost:7862"]
            time.sleep(0.9)  # let the reader drain post-URL output without blocking
            assert proc.poll() is not None
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            proc.stdout.close()

    def test_dead_process_without_url_prints_nothing_extra(self, monkeypatch, capsys):
        _patch_cloudflared(monkeypatch, "import sys\nsys.exit(3)\n")
        assert cli._spawn_cloudflared_tunnel(7863) is None
        captured = capsys.readouterr()
        assert "cloudflared exited before publishing a tunnel URL." in captured.err
        assert "cloudflared output (tail)" not in captured.out

    def test_timeout_with_bad_env_values_uses_default(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_CLOUDFLARED_DEFAULT_TIMEOUT_SECONDS", 1)
        for bad in ("not-a-number", "0", "-4"):
            monkeypatch.setenv("BACKPROPAGATE_CLOUDFLARED_TIMEOUT", bad)
            _patch_cloudflared(monkeypatch, "import time\ntime.sleep(30)\n")
            start = time.monotonic()
            assert cli._spawn_cloudflared_tunnel(7864) is None
            assert time.monotonic() - start < 6
            captured = capsys.readouterr()
            assert "did not surface a tunnel URL within 1s" in captured.err
            assert "BACKPROPAGATE_CLOUDFLARED_TIMEOUT" in captured.out

    def test_timeout_terminate_oserror_is_swallowed(self, monkeypatch, capsys):
        monkeypatch.setenv("BACKPROPAGATE_CLOUDFLARED_TIMEOUT", "1")
        holder: dict = {}

        class WrapProc:
            """Real child process whose terminate() raises, to exercise the guard."""

            def __init__(self, *a, **k):
                self._p = _REAL_POPEN([sys.executable, "-u", "-c", "import time\ntime.sleep(30)\n"],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                holder["p"] = self._p
                self.stdout = self._p.stdout
                self.pid = self._p.pid

            def poll(self):
                return self._p.poll()

            def terminate(self):
                raise OSError("already gone")

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/cloudflared")
        monkeypatch.setattr(cli.subprocess, "Popen", WrapProc)
        try:
            assert cli._spawn_cloudflared_tunnel(7865) is None
        finally:
            holder["p"].kill()
            holder["p"].wait(timeout=5)
            holder["p"].stdout.close()
        assert "did not surface a tunnel URL within 1s" in capsys.readouterr().err

    def test_exited_pre_url_with_queued_tail(self, monkeypatch, capsys):
        """Process is observed dead while the reader still has unread output.

        A stub process makes the sequence deterministic: the 0.5s queue wait
        times out, poll() reports exit, and the tail text arrives on the queue
        just before the drain.
        """
        release = threading.Event()
        reads = {"n": 0}

        class Pipe:
            def readline(self):
                reads["n"] += 1
                if reads["n"] == 1:
                    release.wait(timeout=10)
                    return "ERR tunnel registration failed: no route to host\n"
                return ""

            def close(self):
                pass

        class StubProc:
            pid = 4242
            returncode = 1
            stdout = Pipe()

            def poll(self):
                release.set()
                time.sleep(0.3)  # let the reader enqueue the tail line + EOF sentinel
                return 1

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/cloudflared")
        monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: StubProc())
        assert cli._spawn_cloudflared_tunnel(7866) is None
        captured = capsys.readouterr()
        assert "cloudflared exited before publishing a tunnel URL." in captured.err
        assert "cloudflared output (tail):" in captured.out
        assert "no route to host" in captured.out

    def test_exploding_logger_never_aborts_launch(self, monkeypatch, capsys):
        """Structured-log failures are observability-only on every lifecycle path."""
        monkeypatch.setattr("backpropagate.logging_config.get_logger", lambda name: _ExplodingLogger())

        # not-on-path
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert cli._spawn_cloudflared_tunnel(7867) is None

        # spawn failure
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/cloudflared")

        def boom(*a, **k):
            raise OSError("nope")

        monkeypatch.setattr(cli.subprocess, "Popen", boom)
        assert cli._spawn_cloudflared_tunnel(7867) is None

        # success (spawn_started + debug + url_parsed all raise inside the guard)
        script = "import time\nprint('https://still-works.trycloudflare.com', flush=True)\ntime.sleep(0.3)\n"
        _patch_cloudflared(monkeypatch, script)
        result = cli._spawn_cloudflared_tunnel(7867)
        assert result is not None and result[1] == "https://still-works.trycloudflare.com"
        result[0].wait(timeout=5)
        result[0].stdout.close()

        # timeout path + dead-process path with a raising logger
        monkeypatch.setenv("BACKPROPAGATE_CLOUDFLARED_TIMEOUT", "1")
        _patch_cloudflared(monkeypatch, "import time\ntime.sleep(30)\n")
        assert cli._spawn_cloudflared_tunnel(7867) is None
        _patch_cloudflared(monkeypatch, "import sys\nsys.exit(1)\n")
        assert cli._spawn_cloudflared_tunnel(7867) is None
        capsys.readouterr()


class TestFindPortInUse:
    def test_real_occupied_port_is_returned(self):
        import socket

        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            holder.bind(("127.0.0.1", 0))
            holder.listen(1)
            busy = holder.getsockname()[1]
            assert cli._find_port_in_use("127.0.0.1", [busy]) == busy
        finally:
            holder.close()

    def test_free_ports_return_none(self):
        import socket

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
        probe.close()
        assert cli._find_port_in_use("127.0.0.1", [free]) is None

    def test_eacces_counts_as_in_use(self, monkeypatch):
        class Sock:
            def __init__(self, *a, **k):
                pass

            def bind(self, addr):
                raise OSError(errno.EACCES, "permission denied")

            def close(self):
                pass

        monkeypatch.setattr("socket.socket", Sock)
        assert cli._find_port_in_use("127.0.0.1", [80, 81]) == 80

    def test_unrelated_bind_errors_are_skipped_and_empty_host_defaults(self, monkeypatch):
        seen = []

        class Sock:
            def __init__(self, *a, **k):
                pass

            def bind(self, addr):
                seen.append(addr)
                raise OSError(errno.EADDRNOTAVAIL, "not local")

            def close(self):
                pass

        monkeypatch.setattr("socket.socket", Sock)
        assert cli._find_port_in_use("", [7000, 7001]) is None
        assert seen == [("127.0.0.1", 7000), ("127.0.0.1", 7001)]


class TestLockFileDir:
    def test_xdg_runtime_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        d = cli._lock_file_dir()
        assert d == tmp_path / "backpropagate" and d.is_dir()

    def test_macos_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setattr(cli.sys, "platform", "darwin")
        d = cli._lock_file_dir()
        assert d == tmp_path / "Library" / "Application Support" / "backpropagate"
        assert d.is_dir()

    def test_windows_localappdata_and_home_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.setattr(cli.sys, "platform", "win32")
        local = tmp_path / "local"
        monkeypatch.setenv("LOCALAPPDATA", str(local))
        assert cli._lock_file_dir() == local / "backpropagate"

        monkeypatch.delenv("LOCALAPPDATA")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        assert cli._lock_file_dir() == tmp_path / "AppData" / "Local" / "backpropagate"

    def test_linux_tmp_fallback_and_posix_chmod(self, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.setattr(cli.sys, "platform", "linux")
        made = []
        fake_os = _PosixOs()
        fake_os.chmod = lambda p, m: fake_os.chmods.append((Path(p), m))  # no real chmod on a fake dir
        monkeypatch.setattr(Path, "mkdir", lambda self, **k: made.append(self))
        monkeypatch.setattr(cli, "os", fake_os)
        d = cli._lock_file_dir()
        assert d == Path("/tmp") / "backpropagate"
        assert made == [d]
        assert fake_os.chmods == [(d, 0o700)]

    def test_posix_chmod_oserror_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        fake_os = _PosixOs()

        def deny(p, m):
            raise OSError("fs does not support chmod")

        fake_os.chmod = deny
        monkeypatch.setattr(cli, "os", fake_os)
        assert cli._lock_file_dir() == tmp_path / "backpropagate"


class TestWriteLaunchTokenLock:
    @pytest.fixture(autouse=True)
    def _xdg(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        self.base = tmp_path / "backpropagate"

    def test_writes_token_atomically_and_overwrites(self):
        p = cli.write_launch_token_lock(7860, "tok-one")
        assert p == self.base / "session-7860.lock"
        assert p.read_text(encoding="utf-8") == "tok-one"
        p2 = cli.write_launch_token_lock(7860, "tok-two")
        assert p2 == p and p.read_text(encoding="utf-8") == "tok-two"
        assert [f.name for f in self.base.iterdir()] == ["session-7860.lock"]  # no temp leftovers

    def test_distinct_ports_do_not_collide(self):
        a = cli.write_launch_token_lock(7001, "a")
        b = cli.write_launch_token_lock(7002, "b")
        assert a != b and a.read_text(encoding="utf-8") == "a" and b.read_text(encoding="utf-8") == "b"

    def test_posix_mode_is_restricted(self, monkeypatch):
        fake_os = _PosixOs()
        monkeypatch.setattr(cli, "os", fake_os)
        p = cli.write_launch_token_lock(7003, "secret")
        assert any(m == 0o600 for _, m in fake_os.chmods)
        assert any(m == 0o700 and path == self.base for path, m in fake_os.chmods)
        assert p.read_text(encoding="utf-8") == "secret"

    def test_replace_failure_cleans_up_temp_and_reraises(self, monkeypatch):
        def deny(src, dst):
            raise PermissionError("locked")

        monkeypatch.setattr(cli.os, "replace", deny)
        with pytest.raises(PermissionError, match="locked"):
            cli.write_launch_token_lock(7004, "x")
        assert list(self.base.iterdir()) == []

    def test_cleanup_unlink_failure_does_not_mask_original_error(self, monkeypatch):
        monkeypatch.setattr(cli.os, "replace", lambda s, d: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(Path, "unlink", lambda self, **k: (_ for _ in ()).throw(OSError("stuck")))
        with pytest.raises(RuntimeError, match="boom"):
            cli.write_launch_token_lock(7005, "x")
