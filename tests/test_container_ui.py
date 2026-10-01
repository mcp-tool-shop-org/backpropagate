"""The container UI path: compose.yaml, the pinned bun fetcher, Reflex telemetry.

compose.yaml could never start the UI before 1.8.1 (no --auth with a
non-loopback listener, no Reflex in the image). These tests hold the pieces
that make it start, without needing Docker.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import stat
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_fetch_bun():
    spec = importlib.util.spec_from_file_location("fetch_bun", ROOT / "docker" / "fetch_bun.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zip_with(name: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(f"{name}/bun", payload)
    return buf.getvalue()


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestFetchBun:
    def test_unknown_arch_refused(self, tmp_path, capsys):
        fetch_bun = _load_fetch_bun()
        assert fetch_bun.main("s390x", str(tmp_path)) == 1
        assert "no pinned bun build" in capsys.readouterr().err
        assert not (tmp_path / "bun").exists()

    def test_checksum_mismatch_installs_nothing(self, tmp_path, monkeypatch, capsys):
        fetch_bun = _load_fetch_bun()
        data = _zip_with("bun-linux-x64", b"tampered")
        monkeypatch.setattr(fetch_bun.urllib.request, "urlopen", lambda url, timeout: _FakeResponse(data))
        assert fetch_bun.main("amd64", str(tmp_path)) == 1
        assert "SHA-256 mismatch" in capsys.readouterr().err
        assert not (tmp_path / "bun").exists()

    def test_verified_archive_installs_executable(self, tmp_path, monkeypatch):
        fetch_bun = _load_fetch_bun()
        data = _zip_with("bun-linux-aarch64", b"\x7fELF-fake-bun")
        monkeypatch.setitem(
            fetch_bun.ASSETS, "arm64", ("bun-linux-aarch64", hashlib.sha256(data).hexdigest())
        )
        seen = {}

        def fake_urlopen(url, timeout):
            seen["url"] = url
            return _FakeResponse(data)

        monkeypatch.setattr(fetch_bun.urllib.request, "urlopen", fake_urlopen)
        assert fetch_bun.main("arm64", str(tmp_path / "bin")) == 0
        target = tmp_path / "bin" / "bun"
        assert target.read_bytes() == b"\x7fELF-fake-bun"
        assert seen["url"] == (
            f"https://github.com/oven-sh/bun/releases/download/bun-v{fetch_bun.VERSION}/bun-linux-aarch64.zip"
        )
        if sys.platform != "win32":
            assert target.stat().st_mode & stat.S_IXUSR

    def test_pins_are_sha256_hex(self):
        fetch_bun = _load_fetch_bun()
        assert set(fetch_bun.ASSETS) == {"amd64", "arm64"}
        for _name, digest in fetch_bun.ASSETS.values():
            assert len(digest) == 64 and int(digest, 16) >= 0


class TestComposeFile:
    @pytest.fixture()
    def service(self):
        yaml = pytest.importorskip("yaml")
        doc = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
        return doc, doc["services"]["ui"]

    def test_credential_comes_from_a_secret_file(self, service):
        doc, ui = service
        command = [str(part) for part in ui["command"]]
        assert command[0] == "ui"
        # A non-loopback listener needs credentials, or backprop ui refuses.
        assert command[command.index("--host") + 1] == "0.0.0.0"
        assert command[command.index("--auth-file") + 1] == "/run/secrets/ui_auth"
        assert "--auth" not in command
        assert ui["secrets"] == ["ui_auth"]
        assert doc["secrets"]["ui_auth"]["file"] == "./ui-auth.txt"

    def test_published_port_is_loopback_only(self, service):
        _doc, ui = service
        assert ui["ports"] == ["127.0.0.1:7860:7860"]

    def test_no_ambient_credential_env(self, service):
        # backprop ui strips these, so setting them would only mislead.
        _doc, ui = service
        env = " ".join(ui.get("environment") or [])
        assert "BACKPROPAGATE_UI_AUTH" not in env
        assert "BACKPROPAGATE_UI_LAUNCH_TOKEN" not in env

    def test_credential_file_is_ignored(self):
        for ignore in (".gitignore", ".dockerignore"):
            lines = (ROOT / ignore).read_text(encoding="utf-8").splitlines()
            assert "ui-auth.txt" in lines, ignore


class TestDockerfile:
    def test_image_carries_the_ui(self):
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "--extra ui" in text
        assert "fetch_bun.py" in text
        owned = next(line for line in text.splitlines() if line.lstrip().startswith("&& chown appuser"))
        for path in (".web", ".states", "reflex.lock", "uploaded_files", ".gitignore", "requirements.txt"):
            assert path in owned.split(), path


def test_reflex_telemetry_is_off():
    pytest.importorskip("reflex")
    spec = importlib.util.spec_from_file_location("_bp_rxconfig", ROOT / "backpropagate" / "rxconfig.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.config.telemetry_enabled is False
