"""Sandbox and handler fixes for the UI output folder, uploads, and deletions.

Each test pins one behaviour the pages rely on: a link is not followed, a
token-file path stays off the page, a finished run does not show the home
folder, a second upload keeps the first file, and a cache delete stays inside
the cache.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("reflex", reason="reflex is required (install backpropagate[ui])")

from backpropagate import ui_security as sec  # noqa: E402
from backpropagate import ui_state as us  # noqa: E402
from backpropagate.exceptions import BackpropagateError  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)
    return SimpleNamespace(
        home=home.resolve(),
        out=(home / ".backpropagate" / "ui-outputs").resolve(),
        cache=(home / ".cache" / "huggingface" / "hub").resolve(),
    )


def _junction(link: Path, target: Path) -> bool:
    """Create a Windows junction. False when this host cannot."""
    if os.name != "nt":
        return False
    link.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and link.exists()


def _dir_link(link: Path, target: Path) -> str:
    """A junction when Windows can make one, otherwise a directory symlink."""
    if _junction(link, target):
        return "junction"
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create a directory link: {exc}")
    return "symlink"


class TestOutputDirLinks:
    def test_junction_ui_outputs_is_forbidden_and_creates_nothing(self, sandbox, tmp_path):
        outside = tmp_path / "outside-secret"
        outside.mkdir()
        link = sandbox.home / ".backpropagate" / "ui-outputs"
        if not _junction(link, outside):
            pytest.skip("platform cannot create a junction")
        with pytest.raises(BackpropagateError) as exc:
            sec.get_ui_output_dir()
        assert exc.value.code == "UI_OUTPUT_DIR_FORBIDDEN"
        assert list(outside.iterdir()) == []
        assert str(outside) not in exc.value.message
        assert str(outside) not in str(exc.value.details)

    def test_symlink_ui_outputs_is_forbidden_and_creates_nothing(
        self, sandbox, tmp_path, monkeypatch
    ):
        outside = tmp_path / "outside-link"
        outside.mkdir()
        link = sandbox.home / ".backpropagate" / "linked-outputs"
        try:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this host")
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(link))
        with pytest.raises(BackpropagateError) as exc:
            sec.get_ui_output_dir()
        assert exc.value.code == "UI_OUTPUT_DIR_FORBIDDEN"
        assert list(outside.iterdir()) == []
        assert str(outside) not in exc.value.message


class TestJunctionDetectionFallbacks:
    def test_is_junction_errors(self, tmp_path, monkeypatch):
        path = tmp_path / "dir"
        path.mkdir()

        def denied(self):
            raise OSError("denied")

        # 3.10 and 3.11 have no Path.is_junction; add it there so every
        # interpreter exercises the same branch.
        monkeypatch.setattr(Path, "is_junction", denied, raising=False)
        assert sec._path_is_junction(path) is True

        def missing(self):
            raise FileNotFoundError("gone")

        monkeypatch.setattr(Path, "is_junction", missing, raising=False)
        assert sec._path_is_junction(path) is False

    def test_os_isjunction_errors(self, tmp_path, monkeypatch):
        # On 3.13 Path.is_junction is inherited from a base class, so a
        # delattr on Path does nothing; a None attribute is not callable and
        # sends the helper to the os.path branch on every interpreter.
        monkeypatch.setattr(Path, "is_junction", None, raising=False)
        monkeypatch.setattr(os.path, "isjunction", lambda _path: True, raising=False)
        assert sec._path_is_junction(tmp_path) is True

        def denied(_path):
            raise OSError("denied")

        monkeypatch.setattr(os.path, "isjunction", denied)
        assert sec._path_is_junction(tmp_path) is True

        def missing(_path):
            raise FileNotFoundError("gone")

        monkeypatch.setattr(os.path, "isjunction", missing)
        assert sec._path_is_junction(tmp_path) is False

    def test_reparse_attribute_fallback(self, tmp_path, monkeypatch):
        plain = tmp_path / "plain"
        plain.mkdir()
        assert sec._reparse_point(plain) is False
        assert sec._reparse_point(tmp_path / "missing") is False

        def denied(_path):
            raise OSError("denied")

        monkeypatch.setattr(os, "lstat", denied)
        assert sec._reparse_point(plain) is True

        class _Stat:
            st_file_attributes = 0x400

        monkeypatch.setattr(os, "lstat", lambda _path: _Stat())
        assert sec._reparse_point(plain) is True

    def test_lstat_fallback_is_used_when_no_helper_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "is_junction", None, raising=False)
        monkeypatch.setattr(os.path, "isjunction", None, raising=False)
        plain = tmp_path / "plain-dir"
        plain.mkdir()
        assert sec._path_is_junction(plain) is False


class TestHubSourceRecheck:
    def test_source_swapped_for_a_link_is_refused_at_push(self, sandbox, monkeypatch):
        import backpropagate.export as export_mod

        calls: list[dict] = []

        def _record(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(export_mod, "push_to_hub", _record)
        adapter = sandbox.out / "adapter"
        adapter.mkdir(parents=True)
        outside = sandbox.home.parent / "outside-model"
        outside.mkdir()
        (outside / "keep.txt").write_text("stay", encoding="utf-8")
        state = us.ExportState()
        state.set_source_model_path(str(adapter))
        assert state.source_model_path_error == ""
        stored = Path(state.source_model_path)
        if stored.is_dir() and not stored.is_symlink():
            stored.rmdir()
        _dir_link(stored, outside)
        state.set_hub_repo_id("owner/repo")
        state.set_hub_token("hf_" + "a" * 37)
        state.push_to_hub()
        assert calls == []
        assert state.hub_status == "error"
        assert "UI output folder" in state.hub_message
        assert str(outside) not in state.hub_message
        public = {
            k: str(getattr(state, k)) for k in state.get_fields() if not k.startswith("_")
        }
        assert all(str(outside) not in value for value in public.values())


class TestClientVisiblePaths:
    def test_failed_job_hides_the_home_folder(self, sandbox):
        log = sandbox.out / "jobs" / "x" / "output.log"
        log.parent.mkdir(parents=True)
        home = str(sandbox.home)
        log.write_text(f"failed at {home}\\secret\\weights\n", encoding="utf-8")
        state = us.TrainState()
        output = sandbox.out / "runs" / "job1"
        state._finalize_job(
            {"status": "failed", "output_path": str(output), "log_path": str(log)}
        )
        state.refuse(f"refused {home}\\secret")
        visible = " ".join([state.job_output_path, *state.job_log_tail, state.job_refusal])
        assert home not in visible
        assert state._job_output_path == str(output)
        assert "job1" in state.job_output_path
        assert not Path(state.job_output_path).is_absolute()

    def test_refusal_is_capped(self):
        state = us.TrainState()
        state.set_job_refusal("x" * 1000)
        assert len(state.job_refusal) == us._REFUSAL_MAX
        state.refuse("")
        assert state.job_refusal == ""
        state.refuse("y" * 1000)
        assert len(state.job_refusal) == us._REFUSAL_MAX


class TestDeleteModelLinks:
    def _cache(self, sandbox, monkeypatch):
        sandbox.cache.mkdir(parents=True)
        return sandbox.cache

    def test_junction_is_refused_and_the_target_stays(self, sandbox, tmp_path, monkeypatch):
        cache = self._cache(sandbox, monkeypatch)
        outside = tmp_path / "precious"
        outside.mkdir()
        (outside / "keep.txt").write_text("do not delete", encoding="utf-8")
        link = cache / "models--evil--junction"
        if not _junction(link, outside):
            pytest.skip("platform cannot create a junction")
        monkeypatch.setattr(us.ModelsState, "load_models", lambda self: None)
        state = us.ModelsState()
        state.delete_model("models--evil--junction")
        assert state.error
        assert "junction" in state.error.lower() or "symlink" in state.error.lower()
        assert (outside / "keep.txt").read_text(encoding="utf-8") == "do not delete"
        assert link.exists()

    def test_resolved_cache_root_is_not_deleted(self, sandbox, monkeypatch):
        cache = self._cache(sandbox, monkeypatch)
        (cache / "keep.txt").write_text("stay", encoding="utf-8")
        victim = cache / "models--org--model"
        victim.mkdir()
        real = Path.resolve

        def as_root(self, *args, **kwargs):
            if self.name.startswith("models--"):
                return cache.resolve()
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", as_root)
        monkeypatch.setattr(us.ModelsState, "load_models", lambda self: None)
        state = us.ModelsState()
        state.delete_model("models--org--model")
        assert state.error
        assert (cache / "keep.txt").exists()
        assert victim.exists()


class TestExclusiveWrite:
    def test_a_failed_write_removes_the_partial_file(self, tmp_path, monkeypatch):
        target = tmp_path / "partial.jsonl"

        def boom(_fd, _data):
            raise OSError("disk full")

        monkeypatch.setattr(us.os, "write", boom)
        with pytest.raises(OSError, match="disk full"):
            us._exclusive_write(target, b"abc")
        assert not target.exists()

    def test_a_stuck_unlink_still_raises_the_write_error(self, tmp_path, monkeypatch):
        target = tmp_path / "partial.jsonl"

        def boom(_fd, _data):
            raise OSError("disk full")

        def stuck(self):
            raise OSError("busy")

        monkeypatch.setattr(us.os, "write", boom)
        monkeypatch.setattr(Path, "unlink", stuck)
        with pytest.raises(OSError, match="disk full"):
            us._exclusive_write(target, b"abc")

    def test_a_zero_length_write_stops(self, tmp_path, monkeypatch):
        target = tmp_path / "empty.jsonl"
        monkeypatch.setattr(us.os, "write", lambda _fd, _data: 0)
        assert us._exclusive_write(target, b"abc") is True
        assert target.exists()


class TestHomeRedaction:
    def _portable_home(self, tmp_path: Path) -> Path:
        if os.name == "nt":
            return tmp_path / "portable-home"
        return Path("/opt/backprop-portable-home")

    def test_a_home_outside_users_is_redacted(self, monkeypatch, tmp_path):
        home = self._portable_home(tmp_path)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        out = sec._redact_paths(f"failed at {home / 'secret'} and out/run_x stays")
        assert str(home) not in out
        assert "<redacted-path>" in out
        assert "out/run_x stays" in out
        assert sec._redact_paths("Dry-run OK") == "Dry-run OK"
        assert sec._redact_paths(out) == out

    def test_a_resolved_home_that_differs_is_also_redacted(self, monkeypatch, tmp_path):
        home = tmp_path / "link-home"
        real_home = self._portable_home(tmp_path)
        real = Path.resolve

        def follow(self, *args, **kwargs):
            if self == home:
                return real_home
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        monkeypatch.setattr(Path, "resolve", follow)
        out = sec._redact_paths(f"at {real_home / 'secret'}")
        assert str(real_home) not in out and "<redacted-path>" in out
        linked = sec._redact_paths(f"at {home / 'secret'}")
        assert str(home) not in linked

    def test_an_unresolvable_home_still_redacts_the_raw_folder(self, monkeypatch, tmp_path):
        home = self._portable_home(tmp_path)
        real = Path.resolve

        def boom(self, *args, **kwargs):
            if self == home:
                raise OSError("resolve denied")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        monkeypatch.setattr(Path, "resolve", boom)
        out = sec._redact_paths(f"failed at {home / 'secret'}")
        assert str(home) not in out and "<redacted-path>" in out

    def test_a_missing_home_still_redacts_known_prefixes(self, monkeypatch):
        def boom(cls):
            raise OSError("no home")

        monkeypatch.setattr(Path, "home", classmethod(boom))
        out = sec._redact_paths("see /home/alice/weights")
        assert "alice" not in out and "<redacted-path>" in out

    def test_a_relative_home_is_not_a_prefix(self, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("relative-home")))
        assert sec._redact_paths("relative-home/secret stays") == "relative-home/secret stays"

    def test_a_root_home_is_not_a_prefix(self, monkeypatch):
        root = Path("C:/") if os.name == "nt" else Path("/")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: root))
        assert sec._redact_paths("keep this sentence intact") == "keep this sentence intact"
