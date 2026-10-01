"""End-to-end check of ``backprop ui`` against a REAL Reflex run.

Every other UI auth test drives ``cmd_ui`` with a fake Reflex child, or drives the
ASGI middleware in isolation. Neither can notice that the real UI never starts:
Reflex's dev backend refuses this package layout (``get_reload_paths`` raises
``There should not be an __init__.py file in your app root directory``), and in
dev mode the auth middleware sits on a different port than the page the user
opens. This module launches the actual CLI (``python -m backpropagate ui``), lets
it build and serve the real app, and talks to it over real sockets.

CPU only: nothing here imports torch or touches the GPU. It does need the ``[ui]``
extra, node/bun (Reflex fetches bun itself) and a first-start production build
(about a minute; later starts reuse the per-user UI working directory's
``.web`` build, v1.8.2+). That is why it is marked ``integration`` and excluded from the fast suite; run it by hand::

    pytest tests/test_ui_e2e_real_reflex.py -m integration -p no:cacheprovider --timeout=900

Set ``BACKPROPAGATE_UI_E2E_PORT`` to pin the base port (the ``--auth`` run uses
base + 10); otherwise a free port is picked.

The record of the first manual run (commands and status codes) lives in
``docs/ui-e2e-check-2026-10-01.md``.
"""

from __future__ import annotations

import base64
import http.client
import importlib.util
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from backpropagate.ui_security import verify_password

pytestmark = [
    pytest.mark.integration,
    pytest.mark.serial,
    pytest.mark.timeout(900),
    pytest.mark.skipif(
        importlib.util.find_spec("reflex") is None,
        reason="needs the [ui] extra (pip install backpropagate[ui])",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[1]
READY_TIMEOUT_S = 600
STOP_TIMEOUT_S = 90

# Boot shim: make a console CTRL_BREAK behave like Ctrl+C (KeyboardInterrupt) so the
# CLI's own ``finally`` (lock-file cleanup) runs on Windows exactly as it does when a
# user presses Ctrl+C in a terminal. On POSIX the test sends a real SIGINT instead.
_BOOT = (
    "import signal, sys\n"
    "if hasattr(signal, 'SIGBREAK'):\n"
    "    signal.signal(signal.SIGBREAK, signal.default_int_handler)\n"
    "from backpropagate.cli import main\n"
    "sys.exit(main(sys.argv[1:]))\n"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _base_port() -> int:
    pinned = os.environ.get("BACKPROPAGATE_UI_E2E_PORT", "").strip()
    return int(pinned) if pinned.isdigit() else _free_port()


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _http(
    port: int,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    method: str = "GET",
) -> tuple[int, dict[str, str], bytes]:
    """One raw HTTP exchange (no redirects followed, Host header overridable)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(method, path, headers=headers or {})
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, body
    finally:
        conn.close()


def _ws_upgrade_status(port: int, extra: dict[str, str], query: str = "") -> str:
    """Open the ``/_event`` WebSocket handshake and return the HTTP status line."""
    key = base64.b64encode(os.urandom(16)).decode()
    headers = {
        "Host": f"127.0.0.1:{port}",
        "Origin": f"http://127.0.0.1:{port}",
        "Upgrade": "websocket",
        "Connection": "Upgrade",
        "Sec-WebSocket-Key": key,
        "Sec-WebSocket-Version": "13",
    }
    headers.update(extra)
    path = "/_event/?EIO=4&transport=websocket" + query
    request = f"GET {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
        sock.sendall(request.encode())
        data = sock.recv(4096)
    return data.split(b"\r\n", 1)[0].decode("latin-1", "replace")


def _session_cookie(set_cookie: str) -> str:
    match = re.match(r"(backprop_sess=[^;]+)", set_cookie)
    assert match, f"no backprop_sess cookie in {set_cookie!r}"
    return match.group(1)


class _UiLaunch:
    """A real ``backprop ui`` process plus everything needed to inspect and stop it."""

    def __init__(self, tmp_path: Path, port: int, extra_args: list[str]) -> None:
        self.port = port
        self.run_dir = tmp_path / "run"
        self.run_dir.mkdir()
        self.out_path = tmp_path / "ui.out"
        self.err_path = tmp_path / "ui.err"
        self.lock_path = self.run_dir / "backpropagate" / f"session-{port}.lock"
        env = {k: v for k, v in os.environ.items() if not k.startswith("BACKPROPAGATE_UI_")}
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")]))
        env["XDG_RUNTIME_DIR"] = str(self.run_dir)  # keeps the lock file out of the real profile
        # v1.8.2 routes Reflex's cwd to a per-user workdir; pin it under
        # tmp_path so the real run stays hermetic (stub rxconfig + the .web
        # build tree land here instead of the user profile).
        env["BACKPROPAGATE_UI_WORKDIR"] = str(self.run_dir / "ui-workdir")
        cmd = [sys.executable, "-c", _BOOT, "ui", "--port", str(port), *extra_args]
        kwargs: dict[str, object] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self._out = self.out_path.open("wb")
        self._err = self.err_path.open("wb")
        self.proc = subprocess.Popen(  # noqa: S603  # nosec B603 - argv is built here
            cmd, env=env, stdout=self._out, stderr=self._err, cwd=str(REPO_ROOT), **kwargs  # type: ignore[arg-type]
        )
        self.token = ""
        self.cookie = ""
        self.user = ""
        self.password = ""
        self.returncode: int | None = None
        self.leftovers: list[int] = []
        self.hard_killed = False
        self._known_pids: set[int] = {self.proc.pid}

    # -- output ----------------------------------------------------------
    def output(self) -> str:
        parts = []
        for path in (self.out_path, self.err_path):
            try:
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                parts.append("")
        return "\n".join(parts)

    # -- process tree ----------------------------------------------------
    def descendants(self) -> list[psutil.Process]:
        try:
            kids = psutil.Process(self.proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            return []
        self._known_pids.update(p.pid for p in kids)
        return kids

    def reflex_children(self) -> list[psutil.Process]:
        """The Reflex server tree. Excludes the CLI itself: on Windows the venv launcher
        re-executes the interpreter, so the CLI shows up as its own descendant."""
        found = []
        for proc in self.descendants():
            try:
                if "backpropagate.cli import main" not in " ".join(proc.cmdline()):
                    found.append(proc)
            except psutil.Error:
                continue
        return found

    # -- lifecycle -------------------------------------------------------
    def wait_ready(self) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"backprop ui exited early (rc={self.proc.returncode}):\n{self.output()[-3000:]}"
                )
            try:
                _http(self.port, "/")
                break
            except OSError:
                time.sleep(1.0)
        else:
            raise AssertionError(f"UI not serving after {READY_TIMEOUT_S}s:\n{self.output()[-3000:]}")
        self.descendants()

    def stop(self) -> None:
        """Ctrl+C the CLI like a user would, then reap anything left behind."""
        if self.returncode is None and self.proc.poll() is None:
            self.descendants()
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.hard_killed = True
                self.proc.kill()
                self.proc.wait(timeout=30)
        self.returncode = self.proc.returncode
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            alive = [pid for pid in self._known_pids if psutil.pid_exists(pid) and pid != self.proc.pid]
            if not alive:
                break
            time.sleep(0.5)
        self.leftovers = [pid for pid in self._known_pids if pid != self.proc.pid and psutil.pid_exists(pid)]
        for pid in self.leftovers:  # never leave servers behind, whatever the assertions say
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass
        self._out.close()
        self._err.close()


@pytest.fixture(scope="class")
def default_ui(tmp_path_factory):
    launch = _UiLaunch(tmp_path_factory.mktemp("ui-default"), _base_port(), [])
    try:
        launch.wait_ready()
        launch.token = _banner_token(launch)
        yield launch
    finally:
        launch.stop()


@pytest.fixture(scope="class")
def auth_ui(tmp_path_factory):
    user = "e2e-user"
    password = secrets.token_urlsafe(18)
    launch = _UiLaunch(
        tmp_path_factory.mktemp("ui-auth"),
        _free_port(),
        ["--auth", f"{user}:{password}"],
    )
    launch.user = user
    launch.password = password
    try:
        launch.wait_ready()
        yield launch
    finally:
        launch.stop()


def _banner_token(launch: _UiLaunch) -> str:
    match = re.search(r"/\?token=([A-Za-z0-9_\-]+)", launch.output())
    assert match, "banner did not print a ?token= URL"
    return match.group(1)


class TestDefaultLaunchToken:
    """``backprop ui`` with no flags: per-launch token, then a session cookie."""

    def test_banner_url_is_the_one_port_the_server_listens_on(self, default_ui):
        out = default_ui.output()
        assert f"http://127.0.0.1:{default_ui.port}/?token=" in out
        assert f"App running at: http://127.0.0.1:{default_ui.port}/" in out
        # production mode: a single listener, no N+1 backend port
        assert not _port_open(default_ui.port + 1)

    def test_page_without_token_is_refused(self, default_ui):
        status, headers, _ = _http(default_ui.port, "/")
        assert status == 401
        assert "www-authenticate" in headers

    def test_wrong_token_is_refused(self, default_ui):
        status, _, _ = _http(default_ui.port, "/?token=" + "x" * 43)
        assert status == 401

    def test_token_url_is_accepted_and_sets_the_session_cookie(self, default_ui):
        status, headers, _ = _http(default_ui.port, f"/?token={default_ui.token}")
        assert status == 302
        assert headers["location"] == "/"
        cookie = headers["set-cookie"]
        assert cookie.startswith("backprop_sess=")
        assert "HttpOnly" in cookie
        assert "SameSite=Lax" in cookie
        default_ui.cookie = _session_cookie(cookie)

    def test_cookie_alone_gets_the_real_app(self, default_ui):
        status, _, body = _http(default_ui.port, "/", headers={"Cookie": default_ui.cookie})
        assert status == 200
        assert b"<html" in body
        # the compiled frontend is served by the same guarded port
        asset = re.search(rb'(?:src|href)="(/assets/[^"]+\.js)"', body)
        assert asset, "no compiled asset referenced by the page"
        asset_path = asset.group(1).decode()
        assert _http(default_ui.port, asset_path)[0] == 401
        assert _http(default_ui.port, asset_path, headers={"Cookie": default_ui.cookie})[0] == 200

    def test_websocket_needs_the_cookie(self, default_ui):
        assert " 101 " in _ws_upgrade_status(default_ui.port, {"Cookie": default_ui.cookie})
        assert " 101 " not in _ws_upgrade_status(default_ui.port, {})
        # the token is a page credential only; it never opens the socket
        token_only = _ws_upgrade_status(default_ui.port, {}, query=f"&token={default_ui.token}")
        assert " 101 " not in token_only
        evil = _ws_upgrade_status(default_ui.port, {"Cookie": default_ui.cookie, "Origin": "http://evil.example"})
        assert " 101 " not in evil

    def test_foreign_host_header_is_421(self, default_ui):
        status, _, _ = _http(default_ui.port, "/", headers={"Host": "evil.example"})
        assert status == 421
        status, _, _ = _http(
            default_ui.port,
            f"/?token={default_ui.token}",
            headers={"Host": "evil.example"},
        )
        assert status == 421
        assert " 101 " not in _ws_upgrade_status(
            default_ui.port, {"Cookie": default_ui.cookie, "Host": "evil.example"}
        )

    def test_lock_file_holds_the_token_while_running(self, default_ui):
        assert default_ui.lock_path.is_file()
        assert default_ui.lock_path.read_text(encoding="utf-8").strip() == default_ui.token
        if os.name == "posix":
            assert default_ui.lock_path.stat().st_mode & 0o777 == 0o600

    def test_child_process_tree_holds_the_token_not_a_password(self, default_ui):
        seen = 0
        readable = 0
        report = []  # what each process looked like, for the failure message
        for proc in default_ui.reflex_children():
            try:
                label = f"{proc.pid} {' '.join(proc.cmdline())[:90]!r}"
            except psutil.Error as exc:
                label = f"{proc.pid} <cmdline: {type(exc).__name__}>"
            try:
                env = proc.environ()
            except psutil.Error as exc:
                report.append(f"{label}: environ {type(exc).__name__}")
                continue
            has_token = "BACKPROPAGATE_UI_LAUNCH_TOKEN" in env
            report.append(f"{label}: {len(env)} vars, token={has_token}")
            readable += bool(env)
            if has_token:
                seen += 1
                assert env["BACKPROPAGATE_UI_LAUNCH_TOKEN"] == default_ui.token
            assert "BACKPROPAGATE_UI_AUTH" not in env
        if not readable:
            # Granian (Reflex's prod server when uvicorn/gunicorn are absent)
            # rewrites its process title, which on Linux overwrites the memory
            # /proc/<pid>/environ reads: every process then shows 0 variables.
            # The token tests above already prove the server holds the token.
            pytest.skip("no Reflex process exposes a readable environment:\n" + "\n".join(report))
        assert seen, "no descendant carried BACKPROPAGATE_UI_LAUNCH_TOKEN:\n" + (
            "\n".join(report) or "(no Reflex descendants found)"
        )

    def test_ctrl_c_stops_everything_and_deletes_the_lock_file(self, default_ui):
        default_ui.stop()
        assert not default_ui.hard_killed, f"Ctrl+C did not stop the CLI within {STOP_TIMEOUT_S}s"
        assert default_ui.returncode == 0
        assert "UI stopped" in default_ui.output()
        assert not default_ui.lock_path.exists()
        assert default_ui.leftovers == [], "server processes outlived the CLI"
        assert not _port_open(default_ui.port)


class TestExplicitAuth:
    """``backprop ui --auth user:pass``: HTTP Basic against a scrypt verifier."""

    @staticmethod
    def _basic(user: str, password: str) -> dict[str, str]:
        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}

    def test_no_credentials_is_401_with_a_challenge(self, auth_ui):
        status, headers, _ = _http(auth_ui.port, "/")
        assert status == 401
        assert headers["www-authenticate"].startswith("Basic")

    def test_wrong_password_and_wrong_user_are_refused(self, auth_ui):
        assert _http(auth_ui.port, "/", headers=self._basic(auth_ui.user, auth_ui.password + "x"))[0] == 401
        assert _http(auth_ui.port, "/", headers=self._basic("someone-else", auth_ui.password))[0] == 401

    def test_a_launch_token_is_not_a_credential_in_this_mode(self, auth_ui):
        assert _http(auth_ui.port, "/?token=" + "x" * 43)[0] == 401

    def test_right_password_is_accepted_and_sets_a_session_cookie(self, auth_ui):
        status, headers, body = _http(auth_ui.port, "/", headers=self._basic(auth_ui.user, auth_ui.password))
        assert status == 200
        assert b"<html" in body
        auth_ui.cookie = _session_cookie(headers["set-cookie"])
        status, _, _ = _http(auth_ui.port, "/", headers={"Cookie": auth_ui.cookie})
        assert status == 200

    def test_websocket_needs_the_session_cookie(self, auth_ui):
        assert " 101 " in _ws_upgrade_status(auth_ui.port, {"Cookie": auth_ui.cookie})
        assert " 101 " not in _ws_upgrade_status(auth_ui.port, {})
        assert " 101 " not in _ws_upgrade_status(auth_ui.port, self._basic(auth_ui.user, auth_ui.password))

    def test_foreign_host_header_is_421(self, auth_ui):
        headers = {**self._basic(auth_ui.user, auth_ui.password), "Host": "evil.example"}
        assert _http(auth_ui.port, "/", headers=headers)[0] == 421

    def test_no_plaintext_password_in_child_env_cmdline_lock_dir_or_output(self, auth_ui):
        password = auth_ui.password
        children = auth_ui.reflex_children()
        assert children, "no Reflex child process found"
        verified = 0
        readable = 0
        for proc in children:
            try:
                env = proc.environ()
                cmdline = " ".join(proc.cmdline())
            except psutil.Error:
                continue
            readable += bool(env)
            assert "BACKPROPAGATE_UI_AUTH" not in env
            assert "BACKPROPAGATE_UI_LAUNCH_TOKEN" not in env
            assert password not in cmdline
            assert all(password not in value for value in env.values())
            verifier = env.get("BACKPROPAGATE_UI_AUTH_VERIFIER")
            if verifier:
                verified += 1
                assert env.get("BACKPROPAGATE_UI_AUTH_USER") == auth_ui.user
                assert verifier.startswith("scrypt$")
                assert verify_password(password, verifier)
                assert not verify_password(password + "x", verifier)
        # Granian's process-title rewrite can blank /proc/<pid>/environ (see the
        # token test); the verifier is only checkable where an environment reads.
        assert verified or not readable, "no descendant carried the scrypt verifier"
        # nothing on disk: auth mode writes no lock file, and no file under the run dir holds it
        assert not auth_ui.lock_path.exists()
        for path in auth_ui.run_dir.rglob("*"):
            if path.is_file():
                assert password not in path.read_text(encoding="utf-8", errors="replace")
        assert password not in auth_ui.output()

    def test_ctrl_c_stops_everything(self, auth_ui):
        auth_ui.stop()
        assert not auth_ui.hard_killed, f"Ctrl+C did not stop the CLI within {STOP_TIMEOUT_S}s"
        assert auth_ui.returncode == 0
        assert auth_ui.leftovers == [], "server processes outlived the CLI"
        assert not _port_open(auth_ui.port)
