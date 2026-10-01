"""Coverage tests for the helper layer of ``backpropagate.export``.

Input validators, the interruptible subprocess runner, Hugging Face token
resolution, ``push_to_hub`` error mapping and README mirroring, model-card
plumbing, disk-space pre-flight, bf16 down-cast and llama.cpp discovery.

What is mocked (and nothing else):

* ``huggingface_hub`` (``HfApi`` / ``create_repo`` / ``HfHubHTTPError``) -
  the network boundary of ``push_to_hub``. Calls are recorded and asserted.
* ``subprocess.Popen`` for the Ctrl+C / timeout teardown tests (real child
  processes are used for the happy path and the non-zero-exit path).
* ``unsloth_zoo.llama_cpp`` / ``unsloth`` module stand-ins (the GGUF build
  probe), and ``torch.cuda`` for the load-dtype probe (GPU boundary).
* ``shutil.disk_usage`` and a few ``pathlib`` calls to force OS errors.

Real tiny torch models (``tests/helpers/tiny_models.py``) are used wherever a
model is needed.
"""

from __future__ import annotations

import errno
import logging
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

from backpropagate import export
from backpropagate.exceptions import (
    ExportError,
    GGUFExportError,
    MergeExportError,
)
from tests.helpers.tiny_models import tiny_llama

LOGGER = "backpropagate.export"


# =============================================================================
# Input validators
# =============================================================================


class TestValidateModelName:
    @pytest.mark.parametrize("name", ["my-model", "my-model:latest", "a", "Qwen2.5_7b.v1:q4", "A" * 128])
    def test_accepts_legal_names(self, name):
        export._validate_model_name(name)  # no raise

    @pytest.mark.parametrize("bad", ["", None, 42])
    def test_rejects_empty_and_non_string(self, bad):
        with pytest.raises(ExportError) as exc:
            export._validate_model_name(bad)  # type: ignore[arg-type]
        assert exc.value.code == "INPUT_VALIDATION_FAILED"
        assert "non-empty string" in exc.value.message

    @pytest.mark.parametrize("bad", ["a\nb", "a\rb", "a\x00b"])
    def test_rejects_control_characters(self, bad):
        with pytest.raises(ExportError) as exc:
            export._validate_model_name(bad)
        assert exc.value.code == "INPUT_VALIDATION_FAILED"
        assert "control character" in exc.value.message

    def test_rejects_leading_dash(self):
        with pytest.raises(ExportError, match="starts with '-'") as exc:
            export._validate_model_name("-h")
        assert exc.value.code == "INPUT_VALIDATION_FAILED"

    @pytest.mark.parametrize("bad", ["has space", "a/b", "a\\b", "x" * 129, "semi;colon", "ünï"])
    def test_rejects_characters_outside_allowlist(self, bad):
        with pytest.raises(ExportError, match="invalid characters") as exc:
            export._validate_model_name(bad)
        assert exc.value.code == "INPUT_VALIDATION_FAILED"

    def test_drive_prefixed_name_hits_path_separator_defense_on_windows(self):
        """``C:foo`` passes the regex; ``Path('C:foo').name`` differs only on Windows."""
        if sys.platform == "win32":
            with pytest.raises(ExportError, match="path separators") as exc:
                export._validate_model_name("C:foo")
            assert exc.value.code == "INPUT_VALIDATION_FAILED"
        else:
            export._validate_model_name("C:foo")  # a legal name elsewhere


class TestValidateRepoId:
    @pytest.mark.parametrize("repo", ["alice/qwen-finetune", "a/b", "org.name/model_v1.0"])
    def test_accepts_owner_slash_name(self, repo):
        export._validate_repo_id(repo)

    @pytest.mark.parametrize(
        ("repo", "needle"),
        [
            ("", "non-empty string"),
            (None, "non-empty string"),
            ("alice\\qwen", "backslash"),
            ("alice/qwen\n", "control character"),
            ("alice/\x00x", "control character"),
            ("../qwen", "'..' or '.' segment"),
            ("alice/..", "'..' or '.' segment"),
            ("./x", "'..' or '.' segment"),
            ("noslash", "owner/name"),
            ("-bad/name", "owner/name"),
            ("a/b/c", "owner/name"),
            ("a/" + "x" * 97, "owner/name"),
        ],
    )
    def test_rejects_with_stable_code(self, repo, needle):
        with pytest.raises(ExportError) as exc:
            export._validate_repo_id(repo)  # type: ignore[arg-type]
        assert exc.value.code == "HUB_PUSH_INVALID_REPO"
        assert needle in exc.value.message
        assert exc.value.suggestion  # every rejection carries a next step


# =============================================================================
# _run_subprocess_interruptible
# =============================================================================


class TestRunSubprocessReal:
    """Real child processes (the current interpreter)."""

    def test_captures_stdout_and_returns_completed_process(self):
        done = export._run_subprocess_interruptible(
            [sys.executable, "-c", "print('hello'); import sys; print('warn', file=sys.stderr)"],
            timeout=60,
        )
        assert isinstance(done, subprocess.CompletedProcess)
        assert done.returncode == 0
        assert done.stdout.strip() == "hello"
        assert done.stderr.strip() == "warn"

    def test_nonzero_exit_raises_with_stderr(self):
        with pytest.raises(subprocess.CalledProcessError) as exc:
            export._run_subprocess_interruptible(
                [sys.executable, "-c", "import sys; sys.stderr.write('bad news'); sys.exit(3)"],
                timeout=60,
            )
        assert exc.value.returncode == 3
        assert "bad news" in exc.value.stderr

    def test_check_false_returns_the_failure(self):
        done = export._run_subprocess_interruptible(
            [sys.executable, "-c", "import sys; sys.exit(4)"], timeout=60, check=False
        )
        assert done.returncode == 4

    def test_capture_output_false_leaves_streams_unset(self):
        done = export._run_subprocess_interruptible(
            [sys.executable, "-c", "pass"], timeout=60, capture_output=False
        )
        assert done.returncode == 0
        assert done.stdout is None and done.stderr is None


class _FakePopen:
    """Stand-in for ``subprocess.Popen`` driving the teardown paths."""

    instances: list[_FakePopen] = []

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.returncode = 0
        self.calls: list[tuple[str, tuple]] = []
        _FakePopen.instances.append(self)

    # behaviour switches set by the tests via the class attributes below
    communicate_exc: BaseException | None = None
    wait_raises: bool = False

    def communicate(self, timeout=None):
        self.calls.append(("communicate", (timeout,)))
        if type(self).communicate_exc is not None:
            raise type(self).communicate_exc
        return ("out", "err")

    def send_signal(self, sig):
        self.calls.append(("send_signal", (sig,)))

    def terminate(self):
        self.calls.append(("terminate", ()))

    def wait(self, timeout=None):
        self.calls.append(("wait", (timeout,)))
        if type(self).wait_raises:
            raise RuntimeError("child ignored the signal")

    def kill(self):
        self.calls.append(("kill", ()))


@pytest.fixture
def fake_popen(monkeypatch):
    _FakePopen.instances = []
    _FakePopen.communicate_exc = None
    _FakePopen.wait_raises = False
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    # Both ``creationflags`` and ``CTRL_BREAK_EVENT`` are Windows-only names.
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", 21, raising=False)
    return _FakePopen


def _platform(monkeypatch, name: str) -> None:
    """Make ``export`` believe it runs on ``name`` (only its ``sys`` view changes)."""
    stub = types.SimpleNamespace(platform=name, executable=sys.executable)
    monkeypatch.setattr(export, "sys", stub)


class TestRunSubprocessTeardown:
    """Mocks: ``subprocess.Popen`` so no real process is signalled."""

    @pytest.mark.parametrize("platform", ["win32", "linux"])
    def test_new_process_group_flags_per_platform(self, fake_popen, monkeypatch, platform):
        _platform(monkeypatch, platform)
        export._run_subprocess_interruptible(["tool", "--x"], timeout=5)
        kwargs = fake_popen.instances[0].kwargs
        if platform == "win32":
            assert kwargs["creationflags"] == 512
            assert "start_new_session" not in kwargs
        else:
            assert kwargs["start_new_session"] is True
            assert "creationflags" not in kwargs
        assert kwargs["text"] is True

    def test_caller_supplied_group_flags_are_respected(self, fake_popen):
        export._run_subprocess_interruptible(["t"], timeout=5, creationflags=1)
        assert fake_popen.instances[0].kwargs["creationflags"] == 1

    @pytest.mark.parametrize("platform", ["win32", "linux"])
    def test_timeout_signals_child_then_reraises(self, fake_popen, monkeypatch, platform):
        _platform(monkeypatch, platform)
        fake_popen.communicate_exc = subprocess.TimeoutExpired(["tool"], 5)
        with pytest.raises(subprocess.TimeoutExpired):
            export._run_subprocess_interruptible(["tool"], timeout=5)
        names = [c[0] for c in fake_popen.instances[0].calls]
        assert names == ["communicate", "send_signal" if platform == "win32" else "terminate", "wait"]
        assert "kill" not in names
        sent = dict(fake_popen.instances[0].calls)
        assert sent["wait"] == (10,)
        if platform == "win32":
            assert sent["send_signal"] == (21,)

    @pytest.mark.parametrize("platform", ["win32", "linux"])
    def test_timeout_falls_back_to_kill_when_child_ignores_signal(self, fake_popen, monkeypatch, platform):
        _platform(monkeypatch, platform)
        fake_popen.communicate_exc = subprocess.TimeoutExpired(["tool"], 5)
        fake_popen.wait_raises = True
        with pytest.raises(subprocess.TimeoutExpired):
            export._run_subprocess_interruptible(["tool"], timeout=5)
        assert fake_popen.instances[0].calls[-1] == ("kill", ())

    @pytest.mark.parametrize("platform", ["win32", "linux"])
    def test_keyboard_interrupt_stops_child_and_propagates(self, fake_popen, monkeypatch, platform):
        _platform(monkeypatch, platform)
        fake_popen.communicate_exc = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            export._run_subprocess_interruptible(["tool"], timeout=5)
        names = [c[0] for c in fake_popen.instances[0].calls]
        assert names == ["communicate", "send_signal" if platform == "win32" else "terminate", "wait"]

    def test_keyboard_interrupt_kills_stubborn_child(self, fake_popen):
        fake_popen.communicate_exc = KeyboardInterrupt()
        fake_popen.wait_raises = True
        with pytest.raises(KeyboardInterrupt):
            export._run_subprocess_interruptible(["tool"], timeout=5)
        assert fake_popen.instances[0].calls[-1] == ("kill", ())


# =============================================================================
# Hugging Face token resolution
# =============================================================================


@pytest.fixture
def clean_hf_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


class TestResolveHfToken:
    def test_explicit_wins_over_env(self, clean_hf_env, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "from-env")
        assert export._resolve_hf_token("explicit") == "explicit"

    def test_env_order(self, clean_hf_env, monkeypatch):
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy")
        assert export._resolve_hf_token(None) == "legacy"
        monkeypatch.setenv("HF_TOKEN", "modern")
        assert export._resolve_hf_token(None) == "modern"

    def test_cached_token_file_first_line_only(self, clean_hf_env):
        cache = clean_hf_env / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        (cache / "token").write_text("  hf_cached  \nsecond line ignored\n", encoding="utf-8")
        assert export._resolve_hf_token(None) == "hf_cached"

    def test_blank_cached_file_resolves_to_none(self, clean_hf_env):
        cache = clean_hf_env / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        (cache / "token").write_text("   \n", encoding="utf-8")
        assert export._resolve_hf_token(None) is None

    def test_unreadable_cache_resolves_to_none(self, clean_hf_env):
        # a directory sitting where the token file should be: exists() but unreadable
        (clean_hf_env / ".cache" / "huggingface" / "token").mkdir(parents=True)
        assert export._resolve_hf_token(None) is None

    def test_no_sources_is_none(self, clean_hf_env):
        assert export._resolve_hf_token(None) is None


# =============================================================================
# push_to_hub
# =============================================================================


class _FakeHubError(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.response = types.SimpleNamespace(status_code=status) if status is not None else None


class _FakeApi:
    def __init__(self, token=None):
        self.token = token
        self.calls: list[tuple[str, dict]] = []
        self.raise_on_upload: BaseException | None = None

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        if self.raise_on_upload:
            raise self.raise_on_upload

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))
        if self.raise_on_upload:
            raise self.raise_on_upload


@pytest.fixture
def hub(monkeypatch, clean_hf_env):
    """Install a recording ``huggingface_hub`` (the network boundary)."""
    state = types.SimpleNamespace(api=None, repos=[], apis=[])

    def make_api(token=None):
        state.api = _FakeApi(token)
        state.apis.append(state.api)
        return state.api

    def create_repo(repo_id, **kwargs):
        state.repos.append((repo_id, kwargs))

    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = make_api  # type: ignore[attr-defined]
    mod.create_repo = create_repo  # type: ignore[attr-defined]
    utils = types.ModuleType("huggingface_hub.utils")
    utils.HfHubHTTPError = _FakeHubError  # type: ignore[attr-defined]
    mod.utils = utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return state


def _adapter_dir(tmp_path: Path, *, card: str | None = "# Card\n") -> Path:
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"w" * 64)
    (d / "weights-big.bin").write_bytes(b"b" * 64)
    if card is not None:
        (d / "model_card.md").write_text(card, encoding="utf-8")
    return d


class TestPushToHubHappyPaths:
    def test_invalid_repo_is_rejected_before_any_import(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(ExportError) as exc:
            export.push_to_hub(tmp_path, "not a repo")
        assert exc.value.code == "HUB_PUSH_INVALID_REPO"

    def test_missing_dependency_is_reported(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(ExportError, match="huggingface_hub is not installed"):
            export.push_to_hub(tmp_path, "alice/m")

    def test_missing_path(self, hub, tmp_path):
        with pytest.raises(ExportError, match="Local path does not exist"):
            export.push_to_hub(tmp_path / "gone", "alice/m")

    def test_adapter_dir_uploads_only_adapter_patterns(self, hub, tmp_path):
        d = _adapter_dir(tmp_path)
        url = export.push_to_hub(
            d, "alice/m", token="tok", private=True, commit_message="msg",
            revision="dev", repo_type="model",
        )
        assert url == "https://huggingface.co/alice/m"
        assert hub.api.token == "tok"
        assert hub.repos == [
            ("alice/m", {"token": "tok", "private": True, "exist_ok": True, "repo_type": "model"})
        ]
        (name, kwargs), = hub.api.calls
        assert name == "upload_folder"
        assert kwargs["folder_path"] == str(d.resolve())
        assert kwargs["repo_id"] == "alice/m"
        assert kwargs["commit_message"] == "msg"
        assert kwargs["revision"] == "dev"
        assert kwargs["token"] == "tok"
        assert "adapter_*" in kwargs["allow_patterns"] and "*.md" in kwargs["allow_patterns"]
        # the weights file would NOT match the allow-list
        import fnmatch

        assert not any(fnmatch.fnmatch("weights-big.bin", p) for p in kwargs["allow_patterns"])

    def test_include_base_uploads_everything(self, hub, tmp_path):
        d = _adapter_dir(tmp_path)
        export.push_to_hub(d, "alice/m", include_base=True)
        assert hub.api.calls[0][1]["allow_patterns"] is None
        assert hub.api.calls[0][1]["commit_message"] == "Upload via backpropagate"

    def test_nested_directories_are_walked_but_not_counted_as_files(self, hub, tmp_path, caplog):
        d = _adapter_dir(tmp_path, card=None)
        (d / "nested").mkdir()
        (d / "nested" / "extra.json").write_text("{}", encoding="utf-8")
        with caplog.at_level(logging.INFO, logger=LOGGER):
            export.push_to_hub(d, "alice/m")
        started = [r.getMessage() for r in caplog.records if r.getMessage().startswith("hub_push_started")]
        assert len(started) == 1
        assert "file_count=4" in started[0]  # 3 adapter-dir files + nested/extra.json, no dir entries
        done = [r.getMessage() for r in caplog.records if r.getMessage().startswith("hub_push_complete")]
        assert len(done) == 1 and "url=https://huggingface.co/alice/m" in done[0]

    def test_dir_without_adapter_files_has_no_filter(self, hub, tmp_path):
        d = tmp_path / "merged"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"x")
        export.push_to_hub(d, "alice/m")
        assert hub.api.calls[0][1]["allow_patterns"] is None

    def test_create_repo_false_skips_creation(self, hub, tmp_path):
        d = _adapter_dir(tmp_path, card=None)
        export.push_to_hub(d, "alice/m", create_repo=False)
        assert hub.repos == []
        assert hub.api.calls[0][0] == "upload_folder"

    def test_single_file_goes_through_upload_file(self, hub, tmp_path):
        f = tmp_path / "model-q4.gguf"
        f.write_bytes(b"g" * 32)
        export.push_to_hub(f, "alice/gguf", revision="r1")
        (name, kwargs), = hub.api.calls
        assert name == "upload_file"
        assert kwargs["path_in_repo"] == "model-q4.gguf"
        assert kwargs["path_or_fileobj"] == str(f.resolve())
        assert kwargs["revision"] == "r1"

    def test_token_resolved_from_env_when_not_passed(self, hub, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_TOKEN", "env-token")
        export.push_to_hub(_adapter_dir(tmp_path, card=None), "alice/m")
        assert hub.api.token == "env-token"
        assert hub.repos[0][1]["token"] == "env-token"


class TestPushToHubReadmeMirror:
    def test_model_card_is_mirrored_to_readme_and_kept_on_success(self, hub, tmp_path):
        d = _adapter_dir(tmp_path, card="# My card\n")
        export.push_to_hub(d, "alice/m")
        assert (d / "README.md").read_text(encoding="utf-8") == "# My card\n"

    def test_existing_readme_is_not_overwritten(self, hub, tmp_path):
        d = _adapter_dir(tmp_path, card="# card\n")
        (d / "README.md").write_text("hand written", encoding="utf-8")
        export.push_to_hub(d, "alice/m")
        assert (d / "README.md").read_text(encoding="utf-8") == "hand written"

    def test_symlinked_card_is_refused(self, hub, tmp_path, monkeypatch, caplog):
        d = _adapter_dir(tmp_path, card="# card\n")
        real = Path.is_symlink
        monkeypatch.setattr(
            Path, "is_symlink", lambda self: True if self.name == "model_card.md" else real(self)
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            export.push_to_hub(d, "alice/m")
        assert not (d / "README.md").exists()
        assert any("Refusing to mirror symlinked model_card.md" in r.getMessage() for r in caplog.records)

    def test_oversized_card_is_refused(self, hub, tmp_path, caplog):
        d = _adapter_dir(tmp_path, card=None)
        (d / "model_card.md").write_bytes(b"x" * 1_000_001)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            export.push_to_hub(d, "alice/m")
        assert not (d / "README.md").exists()
        assert any("1000001 bytes" in r.getMessage() for r in caplog.records)

    def test_card_at_exactly_the_cap_is_mirrored(self, hub, tmp_path):
        d = _adapter_dir(tmp_path, card=None)
        (d / "model_card.md").write_bytes(b"x" * 1_000_000)
        export.push_to_hub(d, "alice/m")
        assert (d / "README.md").stat().st_size == 1_000_000

    def test_unstat_able_card_is_skipped(self, hub, tmp_path, monkeypatch, caplog):
        d = _adapter_dir(tmp_path, card="# card\n")
        real = Path.stat

        def flaky(self, *a, **k):
            if self.name == "model_card.md" and sys._getframe(1).f_code.co_name == "push_to_hub":
                raise OSError(errno.EIO, "disk hiccup")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "stat", flaky)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            export.push_to_hub(d, "alice/m")
        assert not (d / "README.md").exists()
        assert any("Could not stat model_card.md" in r.getMessage() for r in caplog.records)

    def test_unwritable_readme_is_skipped_but_push_proceeds(self, hub, tmp_path, monkeypatch, caplog):
        d = _adapter_dir(tmp_path, card="# card\n")
        real = Path.write_text

        def refuse(self, *a, **k):
            if self.name == "README.md":
                raise OSError(errno.EACCES, "read-only")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "write_text", refuse)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            url = export.push_to_hub(d, "alice/m")
        assert url.endswith("alice/m")
        assert hub.api.calls[0][0] == "upload_folder"
        assert any("Could not mirror model_card.md" in r.getMessage() for r in caplog.records)

    def test_failed_upload_rolls_back_the_mirror(self, hub, tmp_path):
        d = _adapter_dir(tmp_path, card="# card\n")
        # arm the next HfApi instance to fail on upload
        real_make = sys.modules["huggingface_hub"].HfApi

        def failing_api(token=None):
            api = real_make(token)
            api.raise_on_upload = _FakeHubError("server down", 503)
            return api

        sys.modules["huggingface_hub"].HfApi = failing_api
        with pytest.raises(ExportError):
            export.push_to_hub(d, "alice/m")
        assert not (d / "README.md").exists()
        assert (d / "model_card.md").exists()

    def test_rollback_failure_does_not_mask_the_real_error(self, hub, tmp_path, monkeypatch, caplog):
        d = _adapter_dir(tmp_path, card="# card\n")
        real_make = sys.modules["huggingface_hub"].HfApi

        def failing_api(token=None):
            api = real_make(token)
            api.raise_on_upload = _FakeHubError("nope", 404)
            return api

        sys.modules["huggingface_hub"].HfApi = failing_api
        real_unlink = Path.unlink

        def stuck(self, *a, **k):
            if self.name == "README.md":
                raise OSError(errno.EBUSY, "locked")
            return real_unlink(self, *a, **k)

        monkeypatch.setattr(Path, "unlink", stuck)
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            with pytest.raises(ExportError) as exc:
                export.push_to_hub(d, "alice/m")
        assert exc.value.code == "HUB_PUSH_NOT_FOUND"
        assert any("README rollback cleanup failed" in r.getMessage() for r in caplog.records)


class TestPushToHubErrorMapping:
    def _fail_with(self, hub, exc: BaseException) -> None:
        mod = sys.modules["huggingface_hub"]
        real_make = mod.HfApi

        def failing_api(token=None):
            api = real_make(token)
            api.raise_on_upload = exc
            return api

        mod.HfApi = failing_api

    @pytest.mark.parametrize(
        ("status", "code", "needle"),
        [
            (401, "INPUT_AUTH_REQUIRED", "authentication failed (HTTP 401)"),
            (403, "INPUT_AUTH_REQUIRED", "authentication failed (HTTP 403)"),
            (404, "HUB_PUSH_NOT_FOUND", "repo not found (HTTP 404)"),
            (500, "HUB_PUSH_NETWORK", "push failed (HTTP 500)"),
            (503, "HUB_PUSH_NETWORK", "push failed (HTTP 503)"),
            (418, "HUB_PUSH_UNKNOWN", "push failed (HTTP 418)"),
            (None, "HUB_PUSH_UNKNOWN", "push failed (HTTP None)"),
        ],
    )
    def test_http_status_maps_to_stable_code(self, hub, tmp_path, status, code, needle):
        self._fail_with(hub, _FakeHubError("hub said no", status))
        with pytest.raises(ExportError) as exc:
            export.push_to_hub(_adapter_dir(tmp_path, card=None), "alice/m")
        assert exc.value.code == code
        assert needle in exc.value.message
        assert isinstance(exc.value.__cause__, _FakeHubError)
        assert exc.value.suggestion

    @pytest.mark.parametrize("raised", [ConnectionError("reset"), TimeoutError("slow")])
    def test_network_errors(self, hub, tmp_path, raised):
        self._fail_with(hub, raised)
        with pytest.raises(ExportError) as exc:
            export.push_to_hub(_adapter_dir(tmp_path, card=None), "alice/m")
        assert exc.value.code == "HUB_PUSH_NETWORK"
        assert "Network error contacting Hugging Face Hub" in exc.value.message
        assert exc.value.__cause__ is raised

    def test_unexpected_exception_is_wrapped(self, hub, tmp_path):
        self._fail_with(hub, ValueError("weird"))
        with pytest.raises(ExportError) as exc:
            export.push_to_hub(_adapter_dir(tmp_path, card=None), "alice/m")
        assert exc.value.code == "HUB_PUSH_UNKNOWN"
        assert "weird" in exc.value.message

    def test_export_error_from_inside_the_push_passes_through_unchanged(self, hub, tmp_path):
        original = ExportError("already structured", code="INPUT_VALIDATION_FAILED")
        self._fail_with(hub, original)
        with pytest.raises(ExportError) as exc:
            export.push_to_hub(_adapter_dir(tmp_path, card=None), "alice/m")
        assert exc.value is original

    def test_create_repo_failure_is_mapped_too(self, hub, tmp_path):
        mod = sys.modules["huggingface_hub"]

        def boom(repo_id, **kwargs):
            raise _FakeHubError("quota", 403)

        mod.create_repo = boom
        with pytest.raises(ExportError) as exc:
            export.push_to_hub(_adapter_dir(tmp_path, card=None), "alice/m")
        assert exc.value.code == "INPUT_AUTH_REQUIRED"


class _ExplodingInfoLogger:
    """A logger whose ``info`` always fails (observability must never block a push)."""

    def info(self, *a, **k):
        raise RuntimeError("log sink exploded")

    def warning(self, *a, **k):
        pass

    debug = warning


class TestPushToHubObservabilityIsBestEffort:
    def test_failing_log_sink_does_not_fail_the_push(self, hub, tmp_path, monkeypatch):
        monkeypatch.setattr(export, "logger", _ExplodingInfoLogger())
        d = _adapter_dir(tmp_path, card=None)
        url = export.push_to_hub(d, "alice/m")
        assert url == "https://huggingface.co/alice/m"
        assert hub.api.calls[0][0] == "upload_folder"

    def test_inventory_survives_unstatable_files(self, hub, tmp_path, monkeypatch):
        f = tmp_path / "weights.gguf"
        f.write_bytes(b"x" * 8)
        real = Path.stat

        def flaky(self, *a, **k):
            if sys._getframe(1).f_code.co_name == "_inventory":
                raise OSError(errno.EIO, "stat failed")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "stat", flaky)
        assert export.push_to_hub(f, "alice/m").endswith("alice/m")
        assert hub.api.calls[0][0] == "upload_file"

    def test_inventory_survives_unstatable_directory_entries(self, hub, tmp_path, monkeypatch):
        d = _adapter_dir(tmp_path, card=None)
        real = Path.stat

        def flaky(self, *a, **k):
            if sys._getframe(1).f_code.co_name == "_inventory":
                raise OSError(errno.EIO, "stat failed")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "stat", flaky)
        assert export.push_to_hub(d, "alice/m").endswith("alice/m")


# =============================================================================
# write_model_card / _maybe_write_model_card
# =============================================================================


class TestModelCardPlumbing:
    def test_write_model_card_wrapper_writes_the_file(self, tmp_path):
        out = tmp_path / "card-out"
        path = export.write_model_card(out, run_id="run-1", base_model="org/base", final_loss=0.5)
        assert path == out / "model_card.md"
        text = path.read_text(encoding="utf-8")
        assert "run-1" in text and "org/base" in text

    def test_disabled_returns_none_and_writes_nothing(self, tmp_path):
        assert export._maybe_write_model_card(
            tmp_path, enabled=False, run_id=None, base_model=None, output_root=None
        ) is None
        assert not (tmp_path / "model_card.md").exists()

    def test_single_file_export_writes_card_beside_it(self, tmp_path):
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"g")
        card = export._maybe_write_model_card(
            gguf, enabled=True, run_id=None, base_model="org/base", output_root=None,
            extra_card_fields={"export_format": "gguf", "quantization": None},
        )
        assert card == tmp_path / "model_card.md"
        text = card.read_text(encoding="utf-8")
        assert "org/base" in text
        assert "Incomplete provenance" in text

    def test_write_failure_is_logged_not_raised(self, tmp_path, monkeypatch, caplog):
        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr("backpropagate.model_card.write_model_card_for_export", boom)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = export._maybe_write_model_card(
                tmp_path, enabled=True, run_id=None, base_model=None, output_root=None
            )
        assert result is None
        assert any("model card emission failed: disk full" in r.getMessage() for r in caplog.records)

    def test_run_record_found_in_parent_of_export_dir(self, tmp_path):
        from backpropagate.checkpoints import RunHistoryManager

        manager = RunHistoryManager(str(tmp_path))
        manager.record_run_started(
            run_id="parent-run", model_name="org/trained", dataset_info="d.jsonl",
            hyperparameters={"lora_r": 8, "lora_alpha": 16, "seed": 7, "method": "sft"},
        )
        manager.record_run_completed(
            run_id="parent-run", final_loss=0.25, loss_history=[1.0, 0.25], steps=10
        )
        export_dir = tmp_path / "lora"
        export_dir.mkdir()
        card = export._maybe_write_model_card(
            export_dir, enabled=True, run_id="parent-run", base_model="fallback/base",
            output_root=None, extra_card_fields={"export_format": "lora"},
        )
        text = card.read_text(encoding="utf-8")
        assert "org/trained" in text  # run record wins over the fallback
        assert "fallback/base" not in text
        assert "| LoRA rank | 8 |" in text
        assert "Incomplete provenance" not in text


# =============================================================================
# llama.cpp / unsloth probes
# =============================================================================


@pytest.fixture
def no_auto_install(monkeypatch):
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising=False)


def _install_fake_unsloth_zoo(monkeypatch, check):
    pkg = types.ModuleType("unsloth_zoo")
    sub = types.ModuleType("unsloth_zoo.llama_cpp")
    sub.LLAMA_CPP_DEFAULT_DIR = "/home/u/.unsloth/llama.cpp"  # type: ignore[attr-defined]
    sub.check_llama_cpp = check  # type: ignore[attr-defined]
    pkg.llama_cpp = sub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unsloth_zoo", pkg)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.llama_cpp", sub)
    return sub


class TestUnslothGgufReady:
    """Mocks: ``unsloth_zoo.llama_cpp`` (the build probe)."""

    @pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
    def test_auto_install_env_means_ready_without_probing(self, monkeypatch, value):
        monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", value)
        monkeypatch.setitem(sys.modules, "unsloth_zoo", None)  # would fail if probed
        assert export._unsloth_gguf_ready() == (True, "")

    def test_unknown_unsloth_zoo_layout_lets_unsloth_try(self, monkeypatch, no_auto_install):
        monkeypatch.setitem(sys.modules, "unsloth_zoo", None)
        monkeypatch.setitem(sys.modules, "unsloth_zoo.llama_cpp", None)  # may already be cached
        assert export._unsloth_gguf_ready() == (True, "")

    def test_built_llama_cpp_is_ready(self, monkeypatch, no_auto_install):
        seen = []
        _install_fake_unsloth_zoo(monkeypatch, lambda d: seen.append(d))
        assert export._unsloth_gguf_ready() == (True, "")
        assert seen == ["/home/u/.unsloth/llama.cpp"]

    def test_missing_build_explains_and_names_the_escape_hatches(self, monkeypatch, no_auto_install):
        def missing(_dir):
            raise RuntimeError("llama-quantize not found\nsecond line")

        _install_fake_unsloth_zoo(monkeypatch, missing)
        ok, why = export._unsloth_gguf_ready()
        assert ok is False
        assert "/home/u/.unsloth/llama.cpp" in why
        assert "llama-quantize not found" in why and "second line" not in why
        assert "BACKPROPAGATE_UNSLOTH_AUTO_INSTALL=1" in why
        assert "UNSLOTH_LLAMA_CPP_PATH" in why

    def test_blank_error_message_falls_back_to_exception_type(self, monkeypatch, no_auto_install):
        def missing(_dir):
            raise FileNotFoundError()

        _install_fake_unsloth_zoo(monkeypatch, missing)
        ok, why = export._unsloth_gguf_ready()
        assert ok is False and "FileNotFoundError" in why


class TestFindLlamaQuantize:
    def _script(self, tmp_path: Path) -> Path:
        root = tmp_path / "llama.cpp"
        root.mkdir()
        script = root / "convert_hf_to_gguf.py"
        script.write_text("# stub", encoding="utf-8")
        return script

    def test_found_next_to_converter(self, tmp_path):
        script = self._script(tmp_path)
        exe = script.parent / ("llama-quantize.exe" if sys.platform == "win32" else "llama-quantize")
        exe.write_bytes(b"")
        assert export._find_llama_quantize(script) == exe

    @pytest.mark.parametrize("sub", [("build", "bin"), ("build", "bin", "Release")])
    def test_found_in_build_dirs(self, tmp_path, sub):
        script = self._script(tmp_path)
        d = script.parent.joinpath(*sub)
        d.mkdir(parents=True)
        exe = d / ("llama-quantize.exe" if sys.platform == "win32" else "llama-quantize")
        exe.write_bytes(b"")
        assert export._find_llama_quantize(script) == exe

    def test_falls_back_to_path_lookup(self, tmp_path, monkeypatch):
        script = self._script(tmp_path)
        monkeypatch.setattr(export.shutil, "which", lambda name: "/opt/bin/llama-quantize")
        assert export._find_llama_quantize(script) == Path("/opt/bin/llama-quantize")

    def test_nothing_found(self, tmp_path, monkeypatch):
        script = self._script(tmp_path)
        monkeypatch.setattr(export.shutil, "which", lambda name: None)
        assert export._find_llama_quantize(script) is None


class TestHasUnslothAndPeftProbes:
    def test_has_unsloth_false_when_import_fails(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "unsloth", None)
        assert export._has_unsloth() is False

    def test_has_unsloth_true_with_stand_in(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "unsloth", types.ModuleType("unsloth"))
        assert export._has_unsloth() is True

    def test_is_peft_model_without_peft(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "peft", None)
        assert export._is_peft_model(object()) is False

    def test_is_peft_model_with_real_peft(self):
        from peft import LoraConfig, get_peft_model

        base = tiny_llama(layers=1)
        peft_model = get_peft_model(base, LoraConfig(r=2, target_modules=["q_proj"]))
        assert export._is_peft_model(peft_model) is True
        assert export._is_peft_model(base) is False


class TestExportLoadDtype:
    """Mocks: ``torch.cuda`` (GPU boundary) so no device is touched."""

    def test_cpu_is_float32(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert export._export_load_dtype() is torch.float32

    def test_cuda_prefers_bf16_then_fp16(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        assert export._export_load_dtype() is torch.bfloat16
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        assert export._export_load_dtype() is torch.float16


# =============================================================================
# Parameter counting, float32 detection, bf16 cast, disk guard (real tiny model)
# =============================================================================


class _RaisingParams:
    def parameters(self):
        raise RuntimeError("no parameters here")


class _BadIterModel:
    def parameters(self):
        def gen():
            yield types.SimpleNamespace(numel=lambda: 3)
            raise RuntimeError("iterator died")

        return gen()


class TestParamCountAndDtypeProbes:
    def test_param_count_of_real_model(self):
        model = tiny_llama(layers=1)
        expected = sum(p.numel() for p in model.parameters())
        assert export._estimate_param_count(model) == expected > 0

    def test_param_count_degrades_to_none(self):
        assert export._estimate_param_count(object()) is None  # no .parameters()
        assert export._estimate_param_count(_RaisingParams()) is None
        assert export._estimate_param_count(_BadIterModel()) is None
        assert export._estimate_param_count(types.SimpleNamespace(parameters=lambda: [])) is None

    def test_float32_detection(self):
        import torch

        assert export._model_has_float32_params(tiny_llama(layers=1)) is True
        assert export._model_has_float32_params(tiny_llama(layers=1, dtype=torch.bfloat16)) is False

    def test_float32_detection_never_raises(self, monkeypatch):
        assert export._model_has_float32_params(object()) is False
        assert export._model_has_float32_params(_RaisingParams()) is False
        assert export._model_has_float32_params(_BadIterModel()) is False
        monkeypatch.setitem(sys.modules, "torch", None)
        # torch missing in a docs-only install: the model is simply not inspected
        assert export._model_has_float32_params(types.SimpleNamespace(parameters=lambda: [])) is False


class TestCastMergedModelToBf16:
    def test_float32_model_is_downcast_in_place(self):
        import torch

        model = tiny_llama(layers=1)
        before = sum(p.numel() for p in model.parameters())
        out = export._cast_merged_model_to_bf16(model)
        assert all(p.dtype == torch.bfloat16 for p in out.parameters())
        assert sum(p.numel() for p in out.parameters()) == before

    def test_bf16_model_is_returned_untouched(self):
        import torch

        model = tiny_llama(layers=1, dtype=torch.bfloat16)
        assert export._cast_merged_model_to_bf16(model) is model

    def test_model_without_callable_to_is_returned_untouched(self):
        import torch

        class NoTo:
            to = None

            def parameters(self):
                return [torch.zeros(2, dtype=torch.float32)]

        m = NoTo()
        assert export._cast_merged_model_to_bf16(m) is m

    def test_cast_failure_degrades_with_warning(self, caplog):
        import torch

        class Stubborn:
            def parameters(self):
                return [torch.zeros(2, dtype=torch.float32)]

            def to(self, dtype):
                raise RuntimeError("cannot cast")

        m = Stubborn()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert export._cast_merged_model_to_bf16(m) is m
        assert any("Could not down-cast merged model to bf16" in r.getMessage() for r in caplog.records)

    def test_to_returning_none_keeps_the_original(self):
        import torch

        class InPlace:
            def __init__(self):
                self.called_with = None

            def parameters(self):
                return [torch.zeros(2, dtype=torch.float32)]

            def to(self, dtype):
                self.called_with = dtype

        m = InPlace()
        assert export._cast_merged_model_to_bf16(m) is m
        assert m.called_with is torch.bfloat16


class TestCheckExportDiskSpace:
    """Mocks: ``shutil.disk_usage`` (OS boundary)."""

    def _usage(self, free: int):
        return types.SimpleNamespace(total=free * 2, used=free, free=free)

    def test_no_param_estimate_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(export.shutil, "disk_usage", lambda p: pytest.fail("must not probe"))
        export._check_export_disk_space(tmp_path, object(), error_cls=MergeExportError)

    def test_insufficient_space_raises_before_the_merge(self, tmp_path, monkeypatch):
        model = tiny_llama(layers=1)
        params = export._estimate_param_count(model)
        monkeypatch.setattr(export.shutil, "disk_usage", lambda p: self._usage(1024))
        with pytest.raises(MergeExportError) as exc:
            export._check_export_disk_space(tmp_path, model, error_cls=MergeExportError, multiplier=1.2)
        assert "Insufficient disk space for export" in exc.value.message
        assert f"~{params / 1e9:.1f}B params" in exc.value.message
        assert exc.value.suggestion

    def test_error_kwargs_reach_the_error_class(self, tmp_path, monkeypatch):
        model = tiny_llama(layers=1)
        monkeypatch.setattr(export.shutil, "disk_usage", lambda p: self._usage(0))
        with pytest.raises(GGUFExportError) as exc:
            export._check_export_disk_space(
                tmp_path, model, error_cls=GGUFExportError,
                output_path=str(tmp_path), quantization="q4_k_m",
            )
        assert exc.value.code == "RUNTIME_GGUF_EXPORT_FAILED"

    def test_ample_space_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(export.shutil, "disk_usage", lambda p: self._usage(10**12))
        export._check_export_disk_space(tmp_path, tiny_llama(layers=1), error_cls=MergeExportError)

    def test_unreadable_volume_skips_the_guard_with_a_warning(self, tmp_path, monkeypatch, caplog):
        def boom(p):
            raise OSError("no such volume")

        monkeypatch.setattr(export.shutil, "disk_usage", boom)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            export._check_export_disk_space(tmp_path, tiny_llama(layers=1), error_cls=MergeExportError)
        assert any("Could not check free disk space" in r.getMessage() for r in caplog.records)
