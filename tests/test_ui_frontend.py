"""Unit tests for backpropagate/ui_frontend.py (offline frontend seeding, 1.8.2).

Payload fixtures are tiny fakes (a real payload is ~250 MB); the heavy
end-to-end offline proof with a real payload lives in
tests/test_ui_offline_seed.py (integration, manual).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from backpropagate import ui_frontend

PAYLOAD_ENV = "BACKPROPAGATE_UI_PAYLOAD_DIR"
BUN_NAME = "bun.exe" if sys.platform == "win32" else "bun"
MAX_HASH_SEED = 2**32 - 1


@pytest.fixture
def payload(tmp_path, monkeypatch):
    """A minimal fake payload pointed at by BACKPROPAGATE_UI_PAYLOAD_DIR."""
    root = tmp_path / "payload"
    (root / "bun").mkdir(parents=True)
    bun_bytes = b"fake-bun-binary"
    # The seeder looks up the PLATFORM bun name (bun.exe on nt, bun elsewhere).
    (root / "bun" / BUN_NAME).write_bytes(bun_bytes)
    # The web tree ships ZIPPED (raw node_modules tails exceed MAX_PATH under
    # the WindowsApps install prefix).
    web_zip = root / "web.zip"
    with zipfile.ZipFile(web_zip, "w") as zf:
        zf.writestr("package.json", "{}")
        zf.writestr("bun.lock", "lock")
        zf.writestr("build/index.js", "bundle")
    from importlib.metadata import version as _dist_version

    meta = {
        "schema": 1,
        "reflex_version": _dist_version("reflex"),
        "backpropagate_version": "0.0.0",
        "bun_version": "1.3.13",
        "bun_sha256": hashlib.sha256(bun_bytes).hexdigest(),
        "web_zip_sha256": _sha256(web_zip),
        "built_utc": "2026-10-01T00:00:00+00:00",
    }
    (root / "payload.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setenv(PAYLOAD_ENV, str(root))
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def reflex_dir(tmp_path, monkeypatch):
    """Isolate the Reflex state dir (bun probe path lives under it)."""
    target = tmp_path / "reflex-state"
    monkeypatch.setenv("REFLEX_DIR", str(target))
    return target


@pytest.fixture
def no_warmup(monkeypatch):
    """Replace the marker warmup with a recording fake; default = success."""
    calls: list[dict] = []

    def fake(workdir, *, port, backend_host, hash_seed, child_env, warn):
        calls.append(
            {
                "workdir": Path(workdir),
                "port": port,
                "backend_host": backend_host,
                "hash_seed": hash_seed,
                "env": dict(child_env),
            }
        )
        (Path(workdir) / ".web" / "reflex.install_frontend_packages.cached").write_bytes(b"x")
        return True

    monkeypatch.setattr(ui_frontend, "_warm_install_marker", fake)
    return calls


def _prepare(workdir, package_dir, **kwargs):
    kwargs.setdefault("port", 7862)
    kwargs.setdefault("backend_host", "127.0.0.1")
    kwargs.setdefault("child_env", {})
    kwargs.setdefault("warn", lambda m: None)
    return ui_frontend.prepare_offline_frontend(workdir, package_dir, **kwargs)


def _read_record(workdir) -> dict:
    return json.loads((workdir / ui_frontend.SEED_RECORD).read_text())


class TestNoPayload:
    def test_returns_not_ready_and_touches_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv(PAYLOAD_ENV, str(tmp_path / "absent"))
        workdir = tmp_path / "wd"
        workdir.mkdir()
        warnings: list[str] = []
        ready, hash_seed = ui_frontend.prepare_offline_frontend(
            workdir, tmp_path, port=7862, backend_host="127.0.0.1",
            child_env={}, warn=warnings.append,
        )
        assert ready is False
        assert hash_seed is None
        assert not (workdir / ".web").exists()
        assert warnings == []


class TestSeeding:
    def test_first_launch_seeds_web_installs_bun_and_warms(
        self, payload, reflex_dir, no_warmup, tmp_path
    ):
        workdir = tmp_path / "wd"
        workdir.mkdir()
        ready, hash_seed = _prepare(workdir, tmp_path, child_env={"A": "1"})
        assert ready is True
        assert (workdir / ".web" / "build" / "index.js").read_text() == "bundle"
        bun = reflex_dir / "bun" / "bin" / BUN_NAME
        assert bun.is_file() and bun.read_bytes() == b"fake-bun-binary"
        assert no_warmup[0]["port"] == 7862 and no_warmup[0]["backend_host"] == "127.0.0.1"
        record = _read_record(workdir)
        assert record["warmed_for"] == [7862, "127.0.0.1"]
        assert record["hash_seed"] == hash_seed

    def test_warm_launch_copies_nothing_and_skips_warmup(
        self, payload, reflex_dir, no_warmup, tmp_path
    ):
        workdir = tmp_path / "wd"
        workdir.mkdir()
        ready1, seed1 = _prepare(workdir, tmp_path)
        assert ready1 is True
        sentinel = workdir / ".web" / "user-state.txt"
        sentinel.write_text("keep me", encoding="utf-8")
        first_mtime = (workdir / ".web" / "bun.lock").stat().st_mtime_ns
        no_warmup.clear()

        ready2, seed2 = _prepare(workdir, tmp_path)
        assert ready2 is True
        assert no_warmup == []  # no second warmup for the same port+host
        assert (workdir / ".web" / "bun.lock").stat().st_mtime_ns == first_mtime
        assert sentinel.read_text() == "keep me"
        assert seed2 == seed1  # warm launches reuse the stored seed verbatim

    def test_port_change_rewarms_regenerates_seed_without_reseeding(
        self, payload, reflex_dir, no_warmup, tmp_path, monkeypatch
    ):
        workdir = tmp_path / "wd"
        workdir.mkdir()
        # Deterministic seed stream: 100 → 200. A randbelow call per warmup.
        seeds = iter([99, 199])
        monkeypatch.setattr(
            ui_frontend.secrets, "randbelow", lambda n: next(seeds)
        )
        ready1, seed1 = _prepare(workdir, tmp_path)
        assert ready1 is True and seed1 == "100"
        (workdir / ".web" / "user-state.txt").write_text("keep me", encoding="utf-8")
        no_warmup.clear()

        ready2, seed2 = _prepare(workdir, tmp_path, port=8000)
        assert ready2 is True
        assert len(no_warmup) == 1 and no_warmup[0]["port"] == 8000
        assert no_warmup[0]["hash_seed"] == "200"
        assert seed2 == "200" and seed2 != seed1  # warmup re-run regenerates
        assert (workdir / ".web" / "user-state.txt").read_text() == "keep me"
        assert _read_record(workdir)["hash_seed"] == "200"

    def test_new_payload_reseeds_web(self, payload, reflex_dir, no_warmup, tmp_path):
        workdir = tmp_path / "wd"
        workdir.mkdir()
        _prepare(workdir, tmp_path)
        (workdir / ".web" / "stale.txt").write_text("old", encoding="utf-8")
        # payload changes (version bump) → reseed
        meta = json.loads((payload / "payload.json").read_text())
        meta["built_utc"] = "2026-10-02T00:00:00+00:00"
        (payload / "payload.json").write_text(json.dumps(meta), encoding="utf-8")
        no_warmup.clear()

        ready, hash_seed = _prepare(workdir, tmp_path)
        assert ready is True
        assert not (workdir / ".web" / "stale.txt").exists()
        assert len(no_warmup) == 1  # record reset → warmup runs again
        assert _read_record(workdir)["hash_seed"] == hash_seed

    def test_reflex_version_mismatch_skips_with_warning(
        self, payload, reflex_dir, tmp_path
    ):
        meta = json.loads((payload / "payload.json").read_text())
        meta["reflex_version"] = "0.0.0-not-installed"
        (payload / "payload.json").write_text(json.dumps(meta), encoding="utf-8")
        workdir = tmp_path / "wd"
        workdir.mkdir()
        warnings: list[str] = []
        ready, hash_seed = ui_frontend.prepare_offline_frontend(
            workdir, tmp_path, port=1, backend_host="h", child_env={},
            warn=warnings.append,
        )
        assert ready is False
        assert hash_seed is None
        assert any("Reflex 0.0.0-not-installed" in w for w in warnings)
        assert not (workdir / ".web").exists()


class TestHashSeedContract:
    """Requirement 1 review pin: the seed is a per-install secret, never "0",
    and the warmup and the real run always share the stored value."""

    def test_seed_is_secret_shaped_and_never_zero(
        self, payload, reflex_dir, no_warmup, tmp_path
    ):
        ready, hash_seed = _prepare(tmp_path / "wd", tmp_path)
        assert ready is True
        assert hash_seed != "0"
        assert 1 <= int(hash_seed) <= MAX_HASH_SEED
        # warmup saw exactly the same value the caller will pin on the real run
        assert no_warmup[0]["hash_seed"] == hash_seed
        assert _read_record(tmp_path / "wd")["hash_seed"] == hash_seed

    def test_warmup_failure_returns_fresh_seed_without_persisting(
        self, payload, reflex_dir, tmp_path, monkeypatch
    ):
        workdir = tmp_path / "wd"
        monkeypatch.setattr(
            ui_frontend.secrets, "randbelow", lambda n: 424243
        )
        monkeypatch.setattr(
            ui_frontend, "_warm_install_marker", lambda *a, **k: False
        )
        ready, hash_seed = _prepare(workdir, tmp_path)
        assert ready is False
        assert hash_seed == "424244"  # the launch still runs on the fresh secret
        assert "hash_seed" not in _read_record(workdir)  # next launch rewarms

    def test_record_without_hash_seed_forces_rewarm(
        self, payload, reflex_dir, no_warmup, tmp_path
    ):
        workdir = tmp_path / "wd"
        _prepare(workdir, tmp_path)
        # Simulate a pre-upgrade (or tampered) record: warmed_for matches and
        # the marker exists, but the seed that produced it is unknown.
        record = _read_record(workdir)
        record.pop("hash_seed")
        (workdir / ui_frontend.SEED_RECORD).write_text(json.dumps(record))
        no_warmup.clear()

        ready, hash_seed = _prepare(workdir, tmp_path)
        assert ready is True
        assert len(no_warmup) == 1  # re-paired marker with a known seed
        assert _read_record(workdir)["hash_seed"] == hash_seed != "0"


class TestWebZipVerification:
    def test_missing_web_zip_sha256_refuses(
        self, payload, reflex_dir, no_warmup, tmp_path
    ):
        meta = json.loads((payload / "payload.json").read_text())
        del meta["web_zip_sha256"]
        (payload / "payload.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(OSError, match="no web_zip_sha256"):
            ui_frontend.prepare_offline_frontend(
                tmp_path / "wd", tmp_path, port=1, backend_host="h",
                child_env={}, warn=lambda m: None,
            )

    def test_corrupt_web_zip_refuses(self, payload, reflex_dir, no_warmup, tmp_path):
        (payload / "web.zip").write_bytes(b"tampered")
        with pytest.raises(OSError, match="SHA-256"):
            ui_frontend.prepare_offline_frontend(
                tmp_path / "wd", tmp_path, port=1, backend_host="h",
                child_env={}, warn=lambda m: None,
            )

    def test_web_zip_member_traversal_refused(
        self, payload, reflex_dir, no_warmup, tmp_path
    ):
        evil = payload / "web.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("build/index.js", "bundle")
            zf.writestr("../evil.txt", "x")
        # Keep the manifest hash CONSISTENT so only the traversal guard fires.
        meta = json.loads((payload / "payload.json").read_text())
        meta["web_zip_sha256"] = hashlib.sha256(evil.read_bytes()).hexdigest()
        (payload / "payload.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(OSError, match="escapes the extract root"):
            ui_frontend.prepare_offline_frontend(
                tmp_path / "wd", tmp_path, port=1, backend_host="h",
                child_env={}, warn=lambda m: None,
            )
        assert not (tmp_path / "evil.txt").exists()


class TestBunVerification:
    def test_corrupt_bundled_bun_raises(self, payload, reflex_dir, no_warmup, tmp_path):
        meta = json.loads((payload / "payload.json").read_text())
        meta["bun_sha256"] = "0" * 64
        (payload / "payload.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(OSError, match="SHA-256"):
            ui_frontend.prepare_offline_frontend(
                tmp_path / "wd", tmp_path, port=1, backend_host="h",
                child_env={}, warn=lambda m: None,
            )

    def test_missing_bun_sha256_refuses_to_install(
        self, payload, reflex_dir, no_warmup, tmp_path
    ):
        meta = json.loads((payload / "payload.json").read_text())
        del meta["bun_sha256"]
        (payload / "payload.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(OSError, match="no bun_sha256"):
            ui_frontend.prepare_offline_frontend(
                tmp_path / "wd", tmp_path, port=1, backend_host="h",
                child_env={}, warn=lambda m: None,
            )

    def test_missing_bundled_bun_refuses_to_install(
        self, payload, reflex_dir, no_warmup, tmp_path
    ):
        (payload / "bun" / BUN_NAME).unlink()
        with pytest.raises(OSError, match="missing from the payload"):
            ui_frontend.prepare_offline_frontend(
                tmp_path / "wd", tmp_path, port=1, backend_host="h",
                child_env={}, warn=lambda m: None,
            )

    def test_existing_bun_is_not_overwritten(self, payload, reflex_dir, no_warmup, tmp_path):
        probe = reflex_dir / "bun" / "bin" / BUN_NAME
        probe.parent.mkdir(parents=True)
        probe.write_bytes(b"operator bun")
        ui_frontend.prepare_offline_frontend(
            tmp_path / "wd", tmp_path, port=1, backend_host="h",
            child_env={}, warn=lambda m: None,
        )
        assert probe.read_bytes() == b"operator bun"


class TestDepthPreflight:
    def test_short_workdir_is_silent(self):
        warn = ui_frontend._preflight_web_depth(Path(r"C:\Users\u\AppData\Local\backpropagate\ui\1.8.2-a1b2c3d4"), 178)
        assert warn is None

    def test_deep_workdir_warns_with_exact_numbers(self):
        deep = Path("C:") / ("u" * 100) / "wd"
        warn = ui_frontend._preflight_web_depth(deep, 178)
        assert warn is not None
        assert "BACKPROPAGATE_UI_WORKDIR" in warn
        assert "259" in warn

    def test_missing_metadata_is_silent(self):
        assert ui_frontend._preflight_web_depth(Path("C:/x"), None) is None


class TestWarmupRunner:
    def test_invokes_child_module_with_caller_seed(self, payload, tmp_path, monkeypatch):
        seen: dict = {}

        def fake_run(cmd, **kwargs):
            seen.update(kwargs)
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(ui_frontend.subprocess, "run", fake_run)
        ok = ui_frontend._warm_install_marker(
            tmp_path, port=7862, backend_host="127.0.0.1",
            hash_seed="424242", child_env={"A": "1"}, warn=lambda m: None,
        )
        assert ok is True
        assert Path(seen["cwd"]) == tmp_path
        assert seen["env"]["PYTHONHASHSEED"] == "424242"
        assert seen["env"]["PYTHONHASHSEED"] != "0"  # never the public constant
        assert seen["env"]["A"] == "1"
        assert seen["cmd"][1:4] == ["-m", "backpropagate.ui_marker_warmup", "7862"]
        assert seen["cmd"][4] == "127.0.0.1"

    def test_failure_warns_and_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ui_frontend.subprocess,
            "run",
            lambda cmd, **k: subprocess.CompletedProcess(cmd, 3, "", "moved"),
        )
        warnings: list[str] = []
        ok = ui_frontend._warm_install_marker(
            tmp_path, port=1, backend_host="h", hash_seed="7",
            child_env={}, warn=warnings.append,
        )
        assert ok is False
        assert any("warmup was skipped" in w for w in warnings)

    def test_timeout_warns_and_returns_false(self, tmp_path, monkeypatch):
        def slow(cmd, **k):
            raise subprocess.TimeoutExpired(cmd, 1)

        monkeypatch.setattr(ui_frontend.subprocess, "run", slow)
        warnings: list[str] = []
        assert ui_frontend._warm_install_marker(
            tmp_path, port=1, backend_host="h", hash_seed="7",
            child_env={}, warn=warnings.append,
        ) is False
        assert any("did not complete" in w for w in warnings)


class TestWarmupModule:
    def test_bad_argv(self):
        from backpropagate import ui_marker_warmup

        assert ui_marker_warmup.main(["only-one"]) == 2

    def test_reflex_internals_moved_returns_3(self, monkeypatch):
        import reflex.utils.js_runtimes as jr

        from backpropagate import ui_marker_warmup

        monkeypatch.setattr(jr, "_frontend_packages_cache_payload", None, raising=False)
        assert ui_marker_warmup.main(["prog", "7862", "127.0.0.1"]) == 3
