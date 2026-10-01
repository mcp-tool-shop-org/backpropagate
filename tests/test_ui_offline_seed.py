"""OFFLINE first/second-launch proof for the bundled UI frontend (1.8.2, PR B).

The Store package must run the UI with NO network from first launch (Microsoft
policy + the S4/S4b spike evidence: a naive first launch runs ``bun add`` and
dies on dead networks). This test drives the real CLI (``python -m
backpropagate ui``) through the production seeding path with:

* all proxies pointed at a dead loopback port (bun/curl/urllib honor them),
* an EMPTY bun global cache (BUN_INSTALL_CACHE_DIR) — fresh-machine stand-in,
* a FRESH Reflex state dir (REFLEX_DIR) — bun must be installed by the seeder,
* the workdir + runtime dir pinned under tmp_path.

Expectations per launch (first AND second): port serves, no-token request is
401, the banner token URL is 200, and the logs contain no registry access
("bun add" / "Downloading package manifest" / "ConnectionRefused").

Manual/integration — needs a real payload (~290 MB; build once):

    python scripts/build_ui_frontend.py --out E:/AI/spikes/s4b/real-payload
    set BACKPROPAGATE_UI_TEST_PAYLOAD=E:\\AI\\spikes\\s4b\\real-payload\\ui_frontend_payload
    pytest tests/test_ui_offline_seed.py -m integration -p no:cacheprovider --timeout=1200
"""

from __future__ import annotations

import http.client
import importlib.util
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.serial,
    pytest.mark.timeout(1200),
    pytest.mark.skipif(
        importlib.util.find_spec("reflex") is None,
        reason="needs the [ui] extra (pip install backpropagate[ui])",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_ENV = "BACKPROPAGATE_UI_TEST_PAYLOAD"
OFFLINE_SINS = ("bun add", "downloading package manifest", "connectionrefused", "registry")


def _payload_root() -> Path:
    raw = os.environ.get(PAYLOAD_ENV, "").strip()
    if not raw:
        pytest.skip(f"no payload: set {PAYLOAD_ENV} (see module docstring)")
    root = Path(raw)
    if not (root / "payload.json").is_file() or not (root / "web.zip").is_file():
        pytest.skip(f"{PAYLOAD_ENV}={root} is not a built payload (run scripts/build_ui_frontend.py)")
    return root


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _http_status(port: int, path: str) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("GET", path)
        return conn.getresponse().status
    finally:
        conn.close()


def _kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(  # noqa: S603  # nosec B603 — OS taskkill on our own child
            ["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=30
        )
    else:
        import signal

        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _launch(tmp_path: Path, port: int, payload: Path, label: str) -> tuple[subprocess.Popen, Path]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("BACKPROPAGATE_UI_")}
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")])
            ),
            # offline rig
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "http_proxy": "http://127.0.0.1:9",
            "https_proxy": "http://127.0.0.1:9",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "BUN_INSTALL_CACHE_DIR": str(tmp_path / "empty-bun-cache"),
            # hermetic per-test state
            "REFLEX_DIR": str(tmp_path / "reflex-state"),
            "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
            "BACKPROPAGATE_UI_WORKDIR": str(tmp_path / "workdir"),
            "BACKPROPAGATE_UI_PAYLOAD_DIR": str(payload),
        }
    )
    (tmp_path / "empty-bun-cache").mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / f"{label}.log"
    out = log_path.open("wb")
    proc = subprocess.Popen(  # noqa: S603  # nosec B603 — argv is internally constructed
        [sys.executable, "-m", "backpropagate", "ui", "--port", str(port)],
        env=env,
        cwd=str(tmp_path),  # neutral cwd: PYTHONPATH must resolve the package
        stdout=out,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        **({"start_new_session": True} if os.name != "nt" else {}),
    )
    return proc, log_path


def _wait_ready(proc: subprocess.Popen, port: int, timeout: float) -> float:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if proc.poll() is not None:
            return -1.0
        if _port_open(port):
            return time.monotonic() - start
        time.sleep(2)
    return -1.0


def _token_from_log(log_path: Path) -> str:
    match = re.search(
        r"/\?token=([A-Za-z0-9_\-]+)", log_path.read_text(errors="replace")
    )
    assert match, f"banner did not print a ?token= URL in {log_path}"
    return match.group(1)


def _staging_root(tmp_path: Path) -> Path:
    """Short-path staging root for the heavy trees (payload copy is ~290 MB).

    The payload's deepest file path is ~182 chars, and this rig (like the
    Store target) runs with LongPathsEnabled=0 — under pytest's deep
    %LOCALAPPDATA%\\Temp\\pytest-of-*\\... prefix the seed copy EXCEEDS
    MAX_PATH (260) and dies mid-copy (measured). Stage at a drive-root short
    path instead; override with BACKPROPAGATE_UI_TEST_STAGING.
    """
    override = os.environ.get("BACKPROPAGATE_UI_TEST_STAGING", "").strip()
    if override:
        root = Path(override)
    elif os.name == "nt":
        root = Path(r"C:\bp-uitest")
    else:
        root = Path("/tmp/bp-uitest")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    return root


def test_offline_first_and_second_launch(tmp_path):
    payload = _payload_root()
    staging = _staging_root(tmp_path)
    port = _free_port()
    timings: dict[str, float] = {}

    for label, budget in (("first", 420.0), ("second", 180.0)):
        launch_dir = staging / label
        launch_dir.mkdir()
        proc, log_path = _launch(launch_dir, port, payload, label)
        try:
            boot = _wait_ready(proc, port, budget)
            timings[label] = boot
            assert boot >= 0, (
                f"{label} launch never served; log tail:\n"
                + log_path.read_text(errors="replace")[-3000:]
            )
            assert _http_status(port, "/") == 401
            token = _token_from_log(log_path)
            assert _http_status(port, f"/?token={token}") == 302
        finally:
            _kill_tree(proc.pid)
            proc.wait(timeout=30)

        log = log_path.read_text(errors="replace").lower()
        for sin in OFFLINE_SINS:
            assert sin not in log, f"{label} launch hit the network: {sin!r} in log"

    # the seeder really ran: payload web tree + seed record in the workdir
    assert (staging / "first" / "workdir" / ".web" / "build").is_dir()
    assert (staging / "first" / "workdir" / ui_frontend_seed_record()).is_file()
    # bundled bun was installed into the hermetic REFLEX_DIR
    bun = staging / "first" / "reflex-state" / "bun" / "bin"
    assert any(bun.iterdir()), "seeder did not install the bundled bun"


def ui_frontend_seed_record() -> str:
    from backpropagate import ui_frontend

    return ui_frontend.SEED_RECORD
