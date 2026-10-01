"""Offline-capable UI frontend seeding (v1.8.2 MSIX track).

The Microsoft Store package ships a prebuilt Reflex frontend so ``backprop ui``
works fully offline from first launch (the Store forbids pulling code from the
network at runtime, and a plain Reflex first launch fetches npm manifests —
spike S4/S4b, 2026-10-01). The payload lives next to the installed package as
``ui_frontend_payload/``::

    payload.json   {"schema": 1, "reflex_version": ..., "backpropagate_version": ...,
                    "bun_version": ..., "bun_sha256": ..., "built_utc": ...}
    web/           the complete .web tree (node_modules, bun.lock, build/, ...)
                   EXCLUDING the machine-specific install marker
    bun/bun.exe    the pinned Windows bun binary

What ``prepare_offline_frontend`` does for a launch (no-ops without a payload,
so pip installs are unchanged):

1. Copies the payload ``.web`` into the per-user working directory when it is
   missing or built from a different payload (tracked by the seed record; a
   version bump / package rebuild reseeds once).
2. Installs the pinned bun binary at Reflex's probe path when absent
   (``$REFLEX_DIR/bun/bin/bun.exe``; Reflex only ``which()``-falls back or
   errors offline otherwise).
3. Runs the marker warmup (``python -m backpropagate.ui_marker_warmup``) when
   the launch parameters changed since the last warm: Reflex gates
   ``bun add`` behind a fingerprint marker that embeds absolute tool paths,
   the port and pydantic's set-ordered config JSON, so the marker must be
   computed ON the end-user machine with the SAME ``PYTHONHASHSEED`` the real
   run will use.

The shared seed is a SECRET, generated per install
(``secrets.randbelow(2**32 - 1) + 1``) and stored in the per-user seed record:
a public constant seed on a network-reachable Reflex server (``--share`` /
``--host``) would hand attackers the hash table's randomization — the classic
hash-flooding DoS a randomized hash seed exists to prevent. The seed is
regenerated whenever the warmup re-runs, stored beside ``warmed_for``, and
reused verbatim on warm launches. This function returns the seed so the
caller can pin the real run to the same value (never ``"0"``).

If the payload's Reflex version differs from the installed one the seed is
skipped with a warning: the fingerprint shape is private API, so we never
guess across versions (worst case is today's online behavior).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

PAYLOAD_DIRNAME = "ui_frontend_payload"
PAYLOAD_META = "payload.json"
SEED_RECORD = ".backpropagate-ui-seed.json"
_WARMUP_TIMEOUT_SECONDS = 420


def _new_hash_seed() -> str:
    """A fresh secret PYTHONHASHSEED for Reflex child processes.

    Range 1..2**32-1 — 0 is excluded because it DISABLES hash randomization.
    """
    return str(secrets.randbelow(2**32 - 1) + 1)


def payload_dir(package_dir: Path) -> Path:
    """Where the bundled frontend payload lives (``ui_frontend_payload/``).

    ``BACKPROPAGATE_UI_PAYLOAD_DIR`` overrides the location — a testing and
    packaging-staging knob (the offline seed integration test points it at a
    fixture/payload outside the package tree).
    """
    override = os.environ.get("BACKPROPAGATE_UI_PAYLOAD_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(package_dir) / PAYLOAD_DIRNAME


def payload_present(package_dir: Path) -> bool:
    return (payload_dir(package_dir) / PAYLOAD_META).is_file()


def _load_payload_meta(package_dir: Path) -> dict | None:
    try:
        meta = json.loads(
            (payload_dir(package_dir) / PAYLOAD_META).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    return meta if isinstance(meta, dict) and "reflex_version" in meta else None


def _payload_id(package_dir: Path) -> str:
    return hashlib.sha256(
        (payload_dir(package_dir) / PAYLOAD_META).read_bytes()
    ).hexdigest()


def _load_seed_record(workdir: Path) -> dict:
    try:
        record = json.loads((workdir / SEED_RECORD).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return record if isinstance(record, dict) else {}


def _write_seed_record(workdir: Path, record: dict) -> None:
    (workdir / SEED_RECORD).write_text(
        json.dumps(record, indent=1, sort_keys=True), encoding="utf-8"
    )


def _reflex_bun_probe_path() -> Path:
    from reflex_base.environment import environment

    return (
        environment.REFLEX_DIR.get()
        / "bun"
        / "bin"
        / ("bun.exe" if os.name == "nt" else "bun")
    )


def _ensure_bun(
    package_dir: Path,
    *,
    warn: Callable[[str], None],
    info: Callable[[str], None] | None = None,
) -> None:
    probe = _reflex_bun_probe_path()
    if probe.is_file():
        return
    bundled = payload_dir(package_dir) / "bun" / probe.name
    meta = _load_payload_meta(package_dir) or {}
    expected = meta.get("bun_sha256")
    # Never install an unverified binary: no checksum in the manifest or no
    # bundled file both mean the payload is not what we built.
    if not expected:
        raise OSError(
            "payload.json carries no bun_sha256; refusing to install an "
            "unverified bun binary"
        )
    if not bundled.is_file():
        raise OSError(f"bundled bun missing from the payload at {bundled}")
    actual = hashlib.sha256(bundled.read_bytes()).hexdigest()
    if actual != expected:
        raise OSError(
            f"bundled bun failed its SHA-256 check ({actual} != {expected}); "
            "the package payload is corrupt"
        )
    probe.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundled, probe)
    if os.name != "nt":  # copyfile does not preserve +x; bun must be executable
        os.chmod(probe, 0o700)  # owner-only execute: REFLEX_DIR is per-user
    (info or warn)(f"Installed bundled bun {meta.get('bun_version', '?')} to {probe}")


def _warm_install_marker(
    workdir: Path,
    *,
    port: int,
    backend_host: str,
    hash_seed: str,
    child_env: dict[str, str],
    warn: Callable[[str], None],
) -> bool:
    """Regenerate Reflex's install-cache marker for this machine + port.

    Returns True when the marker was written. Failure only means the launch
    falls back to today's behavior (an online "bun add" attempt); it never
    blocks the launch.
    """
    env = dict(child_env)
    env["PYTHONHASHSEED"] = hash_seed
    try:
        result = subprocess.run(  # nosec B603 — argv is internally constructed
            [
                sys.executable,
                "-m",
                "backpropagate.ui_marker_warmup",
                str(port),
                backend_host,
            ],
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=_WARMUP_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        warn(f"UI frontend cache warmup did not complete ({exc}); the first launch may need network access.")
        return False
    if result.returncode == 0:
        return True
    tail = (result.stderr or "").strip().splitlines()[-3:]
    warn(
        "UI frontend cache warmup was skipped "
        f"(exit {result.returncode}: {'; '.join(tail) if tail else 'no output'}); "
        "the first launch may need network access."
    )
    return False


def prepare_offline_frontend(
    workdir: Path,
    package_dir: Path,
    *,
    port: int,
    backend_host: str,
    child_env: dict[str, str],
    warn: Callable[[str], None],
    info: Callable[[str], None] | None = None,
) -> tuple[bool, str | None]:
    """Seed the bundled frontend into the workdir.

    Returns ``(ready, hash_seed)``: ``ready`` is True when the launch is fully
    offline-capable; ``hash_seed`` is the per-install secret PYTHONHASHSEED the
    real Reflex run must share with the warmup (None when no seeding applied).

    Raises OSError for real filesystem failures (caller falls back with a
    warning, matching the workdir fallback doctrine).
    """
    meta = _load_payload_meta(package_dir)
    if meta is None:
        return (False, None)

    from importlib.metadata import version as _dist_version

    payload_reflex = meta["reflex_version"]
    installed_reflex = _dist_version("reflex")
    if payload_reflex != installed_reflex:
        warn(
            f"The bundled UI frontend was built for Reflex {payload_reflex} but "
            f"{installed_reflex} is installed; skipping the offline seed "
            "(the first launch may need network access)."
        )
        return (False, None)

    workdir = Path(workdir)
    payload_id = _payload_id(package_dir)
    record = _load_seed_record(workdir)
    web_target = workdir / ".web"

    if record.get("payload_id") != payload_id or not web_target.is_dir():
        if info is not None:
            info("Seeding the bundled UI frontend (~250 MB; once per version)...")
        if web_target.exists():
            shutil.rmtree(web_target)
        shutil.copytree(payload_dir(package_dir) / "web", web_target)
        record = {"payload_id": payload_id}
        _write_seed_record(workdir, record)

    _ensure_bun(package_dir, warn=warn, info=info)

    hash_seed = record.get("hash_seed")
    # A record without hash_seed means the stored marker was computed under an
    # unknown seed (pre-1.8.2 record or tampering) — rewarm to re-pair them.
    needs_warm = (
        record.get("warmed_for") != [port, backend_host]
        or not (web_target / "reflex.install_frontend_packages.cached").is_file()
        or not isinstance(hash_seed, str)
    )
    if needs_warm:
        if info is not None:
            info("Warming the offline frontend cache (once per port change)...")
        hash_seed = _new_hash_seed()  # ALWAYS regenerate when the warmup re-runs
        if _warm_install_marker(
            web_target.parent,
            port=port,
            backend_host=backend_host,
            hash_seed=hash_seed,
            child_env=child_env,
            warn=warn,
        ):
            record["warmed_for"] = [port, backend_host]
            record["hash_seed"] = hash_seed
            _write_seed_record(workdir, record)
        else:
            return (False, hash_seed)
    return (True, hash_seed)
