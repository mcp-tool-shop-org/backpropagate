#!/usr/bin/env python3
"""Build the offline UI frontend payload (1.8.2 MSIX track).

Produces ``ui_frontend_payload/`` — the artifact ``build_msix.py`` copies into
the package and ``backpropagate/ui_frontend.py`` seeds into the per-user
workdir at first launch::

    ui_frontend_payload/
        payload.json   schema, reflex/backpropagate/bun versions, bun hash
        web/           the complete Reflex .web tree WITHOUT the machine-local
                       install-cache marker (regenerated at seed time)
        bun/bun.exe    pinned bun-windows-x64 build, SHA-256 verified

How: stage a throwaway dir holding a stub rxconfig (rendered from the shipped
template, same as the runtime workdir), run a real production build
(``python -m reflex run --env prod``), wait for the server to come up, stop
the tree, then harvest ``.web``. bun is fetched from the pinned GitHub release
with SHA-256 verification (same doctrine as docker/fetch_bun.py).

Requires the [ui] extra in the active interpreter and PYTHONPATH (or an
installed backpropagate) pointing at the sources under build. Network: one
github.com fetch for bun; the frontend build itself resolves all npm packages
(takes 1-3 min online).

Usage: python scripts/build_ui_frontend.py [--out DIR] [--keep-staging]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BUN_VERSION = "1.3.13"
# https://github.com/oven-sh/bun/releases/download/bun-v1.3.13/SHASUMS256.txt
BUN_WINDOWS_SHA256 = "85b14f3e0584218e9b63407b3aa6b90c4835ec5c32435c1f12cb6fc13667c7c9"

BUILD_TIMEOUT_SECONDS = 360


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(  # nosec B603 — the OS taskkill on our own child pid
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            timeout=30,
        )
    else:
        import signal

        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _fetch_bun(dest_dir: Path) -> str:
    name = "bun-windows-x64"
    url = (
        "https://github.com/oven-sh/bun/releases/download/"
        f"bun-v{BUN_VERSION}/{name}.zip"
    )
    print(f"[1/4] fetching {url}", flush=True)
    with urllib.request.urlopen(url, timeout=180) as resp:  # noqa: S310 — fixed https URL with pinned hash below  # nosec B310 — pinned github release + SHA-256 verified right after
        data = resp.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != BUN_WINDOWS_SHA256:
        raise RuntimeError(
            f"bun SHA-256 mismatch: {actual} != {BUN_WINDOWS_SHA256}"
        )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        binary = archive.read(f"{name}/bun.exe")
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "bun.exe"
    target.write_bytes(binary)
    return hashlib.sha256(binary).hexdigest()


def _build_web(staging: Path) -> None:
    """Run a production build in the staging dir and stop once serving."""
    from backpropagate.ui_workdir import render_stub_rxconfig

    (staging / "rxconfig.py").write_text(
        render_stub_rxconfig(REPO_ROOT / "backpropagate" / "rxconfig.py"),
        encoding="utf-8",
    )
    port = _free_port()
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    env["BACKPROPAGATE_UI_PORT"] = str(port)
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"[2/4] production build in {staging} (port {port})...", flush=True)
    log_path = staging / "build.log"
    with log_path.open("wb") as log:
        proc = subprocess.Popen(  # nosec B603 — internally constructed argv
            [
                sys.executable,
                "-m",
                "reflex",
                "run",
                "--env",
                "prod",
                "--frontend-port",
                str(port),
                "--backend-port",
                str(port),
                "--backend-host",
                "127.0.0.1",
            ],
            cwd=str(staging),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **({"start_new_session": True} if os.name != "nt" else {}),
        )
        try:
            deadline = time.monotonic() + BUILD_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    tail = log_path.read_text(errors="replace")[-3000:]
                    raise RuntimeError(
                        f"reflex build exited early (code {proc.returncode}):\n{tail}"
                    )
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(2)
            else:
                raise RuntimeError(f"reflex build did not serve within {BUILD_TIMEOUT_SECONDS}s")
        finally:
            _kill_tree(proc.pid)

    web = staging / ".web"
    if not (web / "build").is_dir():
        raise RuntimeError(f"production build output missing: {web / 'build'}")
    cache_dir = os.environ.get("BUN_INSTALL_CACHE_DIR")  # informational only
    print(f"      build OK (bun cache override: {cache_dir or 'none'})", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "dist" / "ui-frontend-payload",
        help="payload output dir (default: dist/ui-frontend-payload)",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="keep the staging workdir for debugging",
    )
    args = parser.parse_args(argv)

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    try:
        import reflex  # noqa: F401

        reflex_version = _dist_version("reflex")
        subprocess.run(  # nosec B603 — fixed module argv
            [sys.executable, "-c", "import reflex_base"],
            check=True,
            capture_output=True,
        )
    except (ImportError, PackageNotFoundError, subprocess.CalledProcessError):
        print("error: reflex not importable — run inside the [ui] venv", file=sys.stderr)
        return 1

    from backpropagate import __version__ as _dist_backpropagate_version

    # Version of the sources under build: pyproject.toml is the source of
    # truth (dist metadata is stale in editable dev installs).
    try:
        import tomllib

        pkg_version = tomllib.loads(
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["version"]
    except Exception:  # noqa: BLE001 — any parse trouble: fall back to dist metadata
        pkg_version = _dist_backpropagate_version

    payload_root = args.out / "ui_frontend_payload"
    staging = args.out / "build-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if payload_root.exists():
        shutil.rmtree(payload_root)

    bun_sha = _fetch_bun(payload_root / "bun")
    _build_web(staging)

    print(f"[3/4] harvesting .web -> {payload_root / 'web'}", flush=True)
    shutil.copytree(
        staging / ".web",
        payload_root / "web",
        # The install-cache marker embeds build-machine paths/ports — it is
        # regenerated on the user's machine by ui_frontend's warmup step.
        ignore=shutil.ignore_patterns("reflex.install_frontend_packages.cached"),
    )

    # Windows MAX_PATH gate (LongPathsEnabled=0 machines): the runtime
    # workdir prefix is ~60-90 chars (LOCALAPPDATA + username + versioning),
    # so a payload with deep node_modules tails must leave headroom or the
    # 1.2 GB-class seed copy dies with ENOENT mid-copy.
    deepest = max(
        (len(str(p.relative_to(payload_root))) for p in payload_root.rglob("*") if p.is_file()),
        default=0,
    )
    budget = 258 - 90
    if deepest > budget:
        raise RuntimeError(
            f"payload's deepest file path is {deepest} chars; with a 90-char "
            f"runtime prefix budget that breaks MAX_PATH (260). Shorten the "
            "tree (fewer node_modules nesting levels) before shipping."
        )
    print(f"      deepest payload path: {deepest} chars (budget {budget})", flush=True)

    meta = {
        "schema": 1,
        "reflex_version": reflex_version,
        "backpropagate_version": pkg_version,
        "bun_version": BUN_VERSION,
        "bun_sha256": hashlib.sha256((payload_root / "bun" / "bun.exe").read_bytes()).hexdigest(),
        "bun_note": f"bun.exe bytes verified against pinned zip hash {BUN_WINDOWS_SHA256}",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (payload_root / "payload.json").write_text(
        json.dumps(meta, indent=1) + "\n", encoding="utf-8"
    )
    print(f"[4/4] payload.json: {meta}", flush=True)

    total = sum(f.stat().st_size for f in payload_root.rglob("*") if f.is_file())
    print(f"payload ready: {payload_root} ({total / 1024**2:.0f} MiB)", flush=True)

    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
