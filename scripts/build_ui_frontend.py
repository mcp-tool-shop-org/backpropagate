#!/usr/bin/env python3
"""Build the offline UI frontend payload (1.8.2 MSIX track).

Produces ``ui_frontend_payload/`` — the artifact ``build_msix.py`` copies into
the package and ``backpropagate/ui_frontend.py`` seeds into the per-user
workdir at first launch::

    ui_frontend_payload/
        payload.json   schema, reflex/backpropagate/bun versions, bun hash,
                       web.zip hash
        web.zip        the complete Reflex .web tree, zipped WITHOUT the
                       machine-local install-cache marker (regenerated at
                       seed time). Zipped because the raw node_modules tails
                       exceed MAX_PATH under the WindowsApps install prefix
                       (LongPathsEnabled=0) — extraction happens in the
                       per-user workdir where the depth budget holds.
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

# Windows MAX_PATH model (LongPathsEnabled=0): the runtime extraction target
# is %LOCALAPPDATA%\backpropagate\ui\<version>-<hash8>\.web\<member>. Worst
# realistic prefix: 15 + U_MAX ("C:\Users\<name>\AppData\Local"; profile names
# are <= 20 for local accounts, we model 24 for Entra-linked outliers) + 16
# ("backpropagate\ui\") + 20 (version up to "10.20.30" + '-abcdefgh') + 6
# ("\\.web\\") = 81 chars. node_modules MUST ship: reflex's prod path runs
# `react-router build` unconditionally at every launch (setup_frontend_prod →
# build(), verified in reflex 0.9.5 source); bun install runs in "isolated"
# linker mode, which nests co-dependency copies ~8 levels deep under each
# @radix-ui root package — that nesting is the depth driver and pruning
# .map/.d.ts does NOT collapse it (the compiled .mjs sits at the same depth).
_U_MAX = 24
_VERSION_DIR_MAX = 20
RUNTIME_PREFIX_MODEL = 15 + _U_MAX + 16 + _VERSION_DIR_MAX + 6  # = 81
WEB_DEPTH_BUDGET = 259 - RUNTIME_PREFIX_MODEL


def web_depth_budget() -> int:
    """Max chars a .web member path may occupy (modeled worst runtime prefix)."""
    return WEB_DEPTH_BUDGET


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
    from backpropagate.ui_workdir import render_stub_rxconfig, sync_ui_assets

    (staging / "rxconfig.py").write_text(
        render_stub_rxconfig(REPO_ROOT / "backpropagate" / "rxconfig.py"),
        encoding="utf-8",
    )
    # The logo and icons: without them the bundled frontend 404s every image.
    sync_ui_assets(REPO_ROOT / "backpropagate", staging)
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

    print(f"[3/4] zipping .web -> {payload_root / 'web.zip'}", flush=True)

    # Ship zipped: the raw tree's node_modules tails (~180 chars relative to
    # the payload root) would exceed MAX_PATH under the ~86-char WindowsApps
    # install prefix. The seed step extracts into the per-user workdir, where
    # the producer-side depth budget (below) holds.
    zip_path = payload_root / "web.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file in sorted((staging / ".web").rglob("*")):
            if file.name == "reflex.install_frontend_packages.cached":
                continue  # machine-local install marker — regenerated at seed
            if file.is_file():
                zf.write(file, file.relative_to(staging / ".web").as_posix())
    web_zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    deepest_member = max(
        (str(p.relative_to(staging / ".web")).replace("\\", "/") for p in (staging / ".web").rglob("*") if p.is_file()),
        key=len,
        default="",
    )
    deepest = len(deepest_member)
    budget = web_depth_budget()
    if deepest > budget:
        raise RuntimeError(
            f"payload's deepest .web member is {deepest} chars ({deepest_member!r}); "
            f"with the modeled worst runtime prefix ({RUNTIME_PREFIX_MODEL} chars: "
            f"LOCALAPPDATA, u<=24, version dir <=20) the run would need "
            f"{deepest + RUNTIME_PREFIX_MODEL} > 259. If this "
            "trips, shorten OUR segments (workdir naming) — the node_modules "
            "nesting itself is reflex+bun's, not ours to prune."
        )
    print(f"      deepest .web member: {deepest} chars (budget {budget})", flush=True)

    meta = {
        "schema": 1,
        "reflex_version": reflex_version,
        "backpropagate_version": pkg_version,
        "bun_version": BUN_VERSION,
        "bun_sha256": hashlib.sha256((payload_root / "bun" / "bun.exe").read_bytes()).hexdigest(),
        "bun_note": f"bun.exe bytes verified against pinned zip hash {BUN_WINDOWS_SHA256}",
        "web_zip_sha256": web_zip_sha,
        "deepest_web_member": deepest,
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
