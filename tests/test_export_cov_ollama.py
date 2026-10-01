"""Coverage tests for the Ollama half of ``backpropagate.export``.

Modelfile generation, ``register_with_ollama`` / ``remove_ollama_model`` /
``list_ollama_models`` and the adapter-native ("ollama-adapter") export plus
the adapter shelf listing.

What is mocked (and nothing else): the boundary to the ``ollama`` binary.

* ``shutil.which("ollama")`` is patched to report the CLI present / absent.
* ``export._run_subprocess_interruptible`` (``ollama create``) and
  ``subprocess.run`` (``ollama list`` / ``ollama rm``) are replaced by
  recorders that assert the exact argv, kwargs and - for ``create`` - the
  Modelfile contents on disk AT THE MOMENT the CLI would run.

Modelfiles, adapter directories and GGUF stubs are real files on ``tmp_path``.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from backpropagate import export
from backpropagate.exceptions import (
    ExportError,
    GGUFExportError,
    OllamaRegistrationError,
)

LOGGER = "backpropagate.export"


@pytest.fixture
def ollama_present(monkeypatch):
    monkeypatch.setattr(export.shutil, "which", lambda name: "/usr/bin/ollama" if name == "ollama" else None)


@pytest.fixture
def ollama_absent(monkeypatch):
    monkeypatch.setattr(export.shutil, "which", lambda name: None)


class CreateRecorder:
    """Records ``ollama create`` and snapshots the Modelfile it was handed."""

    def __init__(self):
        self.calls: list[dict] = []
        self.raises: BaseException | None = None

    def __call__(self, cmd, **kwargs):
        modelfile = Path(cmd[cmd.index("-f") + 1])
        self.calls.append({
            "cmd": list(cmd),
            "kwargs": kwargs,
            "modelfile": modelfile,
            "modelfile_text": modelfile.read_text(encoding="utf-8") if modelfile.exists() else None,
        })
        if self.raises:
            raise self.raises
        return subprocess.CompletedProcess(cmd, 0, "success", "")


@pytest.fixture
def create(monkeypatch):
    rec = CreateRecorder()
    monkeypatch.setattr(export, "_run_subprocess_interruptible", rec)
    return rec


def _gguf(tmp_path: Path, name: str = "m.gguf") -> Path:
    p = tmp_path / name
    p.write_bytes(b"GGUF")
    return p


# =============================================================================
# create_modelfile
# =============================================================================


class TestCreateModelfile:
    def test_defaults_land_next_to_the_gguf(self, tmp_path):
        gguf = _gguf(tmp_path)
        path = export.create_modelfile(gguf)
        assert path == gguf.resolve().parent / "Modelfile"
        escaped = str(gguf.resolve()).replace("\\", "\\\\")
        assert path.read_text() == (
            f'FROM "{escaped}"\n\nPARAMETER temperature 0.7\nPARAMETER num_ctx 4096'
        )

    def test_custom_parameters_output_path_and_system_prompt(self, tmp_path):
        gguf = _gguf(tmp_path)
        out = tmp_path / "custom" / "Modelfile.alt"
        out.parent.mkdir()
        path = export.create_modelfile(
            gguf, out, system_prompt='Say "hi" \\ politely\tplease', temperature=0.2, context_length=8192
        )
        assert path == out
        text = out.read_text()
        assert "PARAMETER temperature 0.2" in text and "PARAMETER num_ctx 8192" in text
        # backslash escaped first, then the quote
        assert text.endswith('SYSTEM "Say \\"hi\\" \\\\ politely\tplease"')

    def test_empty_system_prompt_adds_no_system_line(self, tmp_path):
        text = export.create_modelfile(_gguf(tmp_path), system_prompt="").read_text()
        assert "SYSTEM" not in text

    @pytest.mark.parametrize(
        ("bad", "label"),
        [
            ("a\nb", "newline"), ("a\rb", "CR"), ("a\x00b", "NUL"),
            ("a\x0cb", "form-feed"), ("a\x0bb", "vertical-tab"), ("a\x01b", "control char U+0001"),
        ],
    )
    def test_system_prompt_control_characters_are_rejected(self, tmp_path, bad, label):
        with pytest.raises(ExportError) as exc:
            export.create_modelfile(_gguf(tmp_path), system_prompt=bad)
        assert exc.value.code == "INPUT_VALIDATION_FAILED"
        assert f"system_prompt contains a {label} character" in exc.value.message
        assert not (tmp_path / "Modelfile").exists()

    def test_control_characters_in_the_gguf_path_are_rejected(self, tmp_path):
        with pytest.raises(ExportError, match="gguf_path contains a newline") as exc:
            export.create_modelfile(tmp_path / "bad\nname.gguf")
        assert exc.value.code == "INPUT_VALIDATION_FAILED"


# =============================================================================
# register_with_ollama
# =============================================================================


class TestRegisterWithOllama:
    def test_success_runs_ollama_create_with_the_modelfile_then_cleans_up(self, tmp_path, ollama_present, create):
        gguf = _gguf(tmp_path)
        assert export.register_with_ollama(gguf, "my-model:v1", system_prompt="Be brief") is True
        (call,) = create.calls
        assert call["cmd"] == ["ollama", "create", "my-model:v1", "-f", str(gguf.resolve().parent / "Modelfile")]
        assert call["kwargs"] == {"capture_output": True, "text": True, "check": True, "timeout": 600}
        assert call["modelfile_text"].startswith("FROM ")
        assert 'SYSTEM "Be brief"' in call["modelfile_text"]
        assert not call["modelfile"].exists()  # temp Modelfile removed afterwards

    def test_quantize_adds_the_ollama_flag(self, tmp_path, ollama_present, create):
        export.register_with_ollama(_gguf(tmp_path), "m", quantize="q4_K_M")
        assert create.calls[0]["cmd"][-2:] == ["--quantize", "q4_K_M"]

    @pytest.mark.parametrize("bad", ["q4 K_M", "q4-K", "x" * 17, ";rm"])
    def test_invalid_quantize_level_is_rejected_and_cleaned_up(self, tmp_path, ollama_present, create, bad):
        with pytest.raises(OllamaRegistrationError, match="Invalid quantization level"):
            export.register_with_ollama(_gguf(tmp_path), "m", quantize=bad)
        assert create.calls == []
        assert not (tmp_path / "Modelfile").exists()

    def test_invalid_model_name_fails_before_touching_anything(self, tmp_path, ollama_present, create):
        with pytest.raises(ExportError) as exc:
            export.register_with_ollama(_gguf(tmp_path), "-evil")
        assert exc.value.code == "INPUT_VALIDATION_FAILED"
        assert create.calls == []

    def test_missing_gguf(self, tmp_path, ollama_present, create):
        with pytest.raises(OllamaRegistrationError, match="GGUF file not found") as exc:
            export.register_with_ollama(tmp_path / "gone.gguf", "m")
        assert exc.value.code == "DEP_OLLAMA_REGISTRATION_FAILED" and exc.value.retryable is True
        assert exc.value.model_name == "m"

    def test_missing_cli(self, tmp_path, ollama_absent, create):
        with pytest.raises(OllamaRegistrationError, match="Ollama CLI not found in PATH") as exc:
            export.register_with_ollama(_gguf(tmp_path), "m")
        assert "https://ollama.ai" in (exc.value.suggestion or "")
        assert create.calls == []

    def test_modelfile_creation_failure_is_wrapped(self, tmp_path, ollama_present, create):
        with pytest.raises(OllamaRegistrationError, match="Failed to create Modelfile") as exc:
            export.register_with_ollama(_gguf(tmp_path), "m", system_prompt="bad\nprompt")
        assert isinstance(exc.value.__cause__, ExportError)
        assert create.calls == []

    def test_timeout(self, tmp_path, ollama_present, create):
        create.raises = subprocess.TimeoutExpired(["ollama"], 600)
        with pytest.raises(OllamaRegistrationError, match="timed out after 10 minutes") as exc:
            export.register_with_ollama(_gguf(tmp_path), "m")
        assert "ollama serve" in (exc.value.suggestion or "")
        assert not (tmp_path / "Modelfile").exists()

    def test_nonzero_exit_includes_truncated_stderr(self, tmp_path, ollama_present, create):
        create.raises = subprocess.CalledProcessError(1, ["ollama"], stderr="E" * 600)
        with pytest.raises(OllamaRegistrationError) as exc:
            export.register_with_ollama(_gguf(tmp_path), "m")
        assert "ollama create failed: " + "E" * 500 in exc.value.message
        assert "E" * 501 not in exc.value.message
        assert isinstance(exc.value.__cause__, subprocess.CalledProcessError)
        assert not (tmp_path / "Modelfile").exists()

    def test_nonzero_exit_without_stderr(self, tmp_path, ollama_present, create):
        create.raises = subprocess.CalledProcessError(1, ["ollama"], stderr=None)
        with pytest.raises(OllamaRegistrationError, match="Unknown error"):
            export.register_with_ollama(_gguf(tmp_path), "m")

    def test_modelfile_cleanup_failure_is_only_a_warning(self, tmp_path, ollama_present, create, monkeypatch, caplog):
        real_unlink = Path.unlink

        def stuck(self, *a, **k):
            if self.name == "Modelfile":
                raise OSError("locked")
            return real_unlink(self, *a, **k)

        monkeypatch.setattr(Path, "unlink", stuck)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert export.register_with_ollama(_gguf(tmp_path), "m") is True
        assert any("Failed to clean up Modelfile" in r.getMessage() for r in caplog.records)


# =============================================================================
# list_ollama_models / remove_ollama_model (boundary: subprocess.run)
# =============================================================================


class RunRecorder:
    def __init__(self, result=None, raises=None):
        self.result = result
        self.raises = raises
        self.calls: list[tuple[list, dict]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        if self.raises:
            raise self.raises
        return self.result


LIST_OUTPUT = (
    "NAME              ID            SIZE     MODIFIED\n"
    "my-finetune:latest  abc123      4.1 GB   2 days ago\n"
    "\n"
    "qwen:7b           def456        4.4 GB   3 weeks ago\n"
)


class TestListOllamaModels:
    def test_parses_names_and_skips_header_and_blank_lines(self, ollama_present, monkeypatch):
        rec = RunRecorder(subprocess.CompletedProcess(["ollama", "list"], 0, LIST_OUTPUT, ""))
        monkeypatch.setattr(export.subprocess, "run", rec)
        assert export.list_ollama_models() == ["my-finetune:latest", "qwen:7b"]
        (cmd, kwargs), = rec.calls
        assert cmd == ["ollama", "list"]
        assert kwargs == {"capture_output": True, "text": True, "check": True, "timeout": 30}

    def test_header_only_output(self, ollama_present, monkeypatch):
        monkeypatch.setattr(
            export.subprocess, "run",
            RunRecorder(subprocess.CompletedProcess([], 0, "NAME ID SIZE MODIFIED\n", "")),
        )
        assert export.list_ollama_models() == []

    def test_missing_cli_warns_and_returns_empty(self, ollama_absent, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert export.list_ollama_models() == []
        assert any("not on PATH" in r.getMessage() for r in caplog.records)

    def test_daemon_error_warns_with_stderr(self, ollama_present, monkeypatch, caplog):
        monkeypatch.setattr(
            export.subprocess, "run",
            RunRecorder(raises=subprocess.CalledProcessError(2, ["ollama"], stderr="connection refused")),
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert export.list_ollama_models() == []
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "exit 2" in msg and "connection refused" in msg and "ollama serve" in msg

    def test_timeout_warns(self, ollama_present, monkeypatch, caplog):
        monkeypatch.setattr(export.subprocess, "run", RunRecorder(raises=subprocess.TimeoutExpired(["o"], 30)))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert export.list_ollama_models() == []
        assert any("timed out after 30s" in r.getMessage() for r in caplog.records)


class TestRemoveOllamaModel:
    def test_invalid_name(self, ollama_present):
        with pytest.raises(ExportError) as exc:
            export.remove_ollama_model("bad name")
        assert exc.value.code == "INPUT_VALIDATION_FAILED"

    def test_missing_cli_returns_false(self, ollama_absent):
        assert export.remove_ollama_model("m") is False

    def test_success_runs_ollama_rm(self, ollama_present, monkeypatch):
        rec = RunRecorder(subprocess.CompletedProcess([], 0, "", ""))
        monkeypatch.setattr(export.subprocess, "run", rec)
        assert export.remove_ollama_model("my-model:v1") is True
        (cmd, kwargs), = rec.calls
        assert cmd == ["ollama", "rm", "my-model:v1"]
        assert kwargs["timeout"] == 30 and kwargs["check"] is True

    def test_timeout(self, ollama_present, monkeypatch):
        monkeypatch.setattr(export.subprocess, "run", RunRecorder(raises=subprocess.TimeoutExpired(["o"], 30)))
        with pytest.raises(OllamaRegistrationError, match="ollama rm timed out after 30s") as exc:
            export.remove_ollama_model("m")
        assert "pkill ollama" in (exc.value.suggestion or "")

    @pytest.mark.parametrize(
        ("stderr", "needle"),
        [
            ("Error: model 'm' not found", "Tag-sensitive"),
            ("dial tcp: connection refused", "does not appear to be running"),
            ("open /x: no such file", "does not appear to be running"),
            ("something else broke", "--verbose"),
            ("", "--verbose"),
            (None, "--verbose"),
        ],
    )
    def test_failure_suggestion_depends_on_the_error(self, ollama_present, monkeypatch, stderr, needle):
        monkeypatch.setattr(
            export.subprocess, "run",
            RunRecorder(raises=subprocess.CalledProcessError(1, ["ollama"], stderr=stderr)),
        )
        with pytest.raises(OllamaRegistrationError) as exc:
            export.remove_ollama_model("m")
        assert needle in (exc.value.suggestion or "")
        assert "ollama rm failed: " in exc.value.message
        if not stderr:
            assert "Unknown error" in exc.value.message


# =============================================================================
# Adapter-native export
# =============================================================================


def _safetensors_adapter(root: Path, name: str = "my-adapter") -> Path:
    d = root / name
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"st")
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    return d


class TestResolveAdapterArtifact:
    def test_single_gguf_and_safetensors_files(self, tmp_path):
        g = tmp_path / "x.GGUF"
        g.write_bytes(b"g")
        s = tmp_path / "y.safetensors"
        s.write_bytes(b"s")
        assert export._resolve_adapter_artifact(g) == g
        assert export._resolve_adapter_artifact(s) == s

    def test_unrecognised_file_suffix(self, tmp_path):
        f = tmp_path / "adapter.bin"
        f.write_bytes(b"b")
        with pytest.raises(GGUFExportError, match=r"got suffix '\.bin'") as exc:
            export._resolve_adapter_artifact(f)
        assert exc.value.code == "RUNTIME_GGUF_EXPORT_FAILED"
        nosuffix = tmp_path / "adapter"
        nosuffix.write_bytes(b"b")
        with pytest.raises(GGUFExportError, match="<none>"):
            export._resolve_adapter_artifact(nosuffix)

    def test_nonexistent_path(self, tmp_path):
        with pytest.raises(GGUFExportError, match="adapter path does not exist"):
            export._resolve_adapter_artifact(tmp_path / "ghost")

    def test_safetensors_directory_is_returned_as_is(self, tmp_path):
        d = _safetensors_adapter(tmp_path)
        assert export._resolve_adapter_artifact(d) == d

    def test_gguf_adapter_inside_directory_wins_when_no_safetensors(self, tmp_path):
        d = tmp_path / "gg"
        d.mkdir()
        (d / "z-lora.gguf").write_bytes(b"1")
        (d / "a-lora.gguf").write_bytes(b"2")
        assert export._resolve_adapter_artifact(d) == d / "a-lora.gguf"  # sorted, first

    def test_bin_only_and_empty_directories_degrade_with_a_named_fix(self, tmp_path):
        b = tmp_path / "binonly"
        b.mkdir()
        (b / "adapter_model.bin").write_bytes(b"x")
        with pytest.raises(GGUFExportError, match="legacy adapter_model.bin") as exc:
            export._resolve_adapter_artifact(b)
        assert "convert_lora_to_gguf.py" in (exc.value.suggestion or "")
        e = tmp_path / "empty"
        e.mkdir()
        with pytest.raises(GGUFExportError, match="no adapter_model.safetensors and no .gguf adapter"):
            export._resolve_adapter_artifact(e)


class TestDeriveAdapterTag:
    @pytest.mark.parametrize(
        ("dirname", "tag"),
        [
            ("my-adapter", "my-adapter"),
            ("Run 2026 v2", "Run-2026-v2"),
            ("a  b!!c", "a-b-c"),
            ("__x__", "x"),
            ("!!!", "adapter"),
            ("v1.2_final", "v1.2_final"),
        ],
    )
    def test_directory_names(self, tmp_path, dirname, tag):
        d = tmp_path / dirname
        d.mkdir()
        assert export._derive_adapter_tag(d) == tag

    def test_single_file_uses_the_stem(self, tmp_path):
        f = tmp_path / "task one.gguf"
        f.write_bytes(b"g")
        assert export._derive_adapter_tag(f) == "task-one"


class TestCreateAdapterModelfile:
    def test_bare_base_name_is_unquoted(self, tmp_path):
        d = _safetensors_adapter(tmp_path)
        path = export.create_adapter_modelfile("llama3.2", d, system_prompt='be "nice"')
        assert path == d.resolve() / "Modelfile"
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines[0] == "FROM llama3.2"
        assert lines[1] == 'ADAPTER "' + str(d.resolve()).replace("\\", "\\\\") + '"'
        assert "PARAMETER temperature 0.7" in text and "PARAMETER num_ctx 4096" in text
        assert text.endswith('SYSTEM "be \\"nice\\""\n')

    @pytest.mark.parametrize(
        ("base", "quoted"),
        [
            ("mistral:7b", False),
            ("llama3.2", False),
            ("/models/base.gguf", True),
            ("C:\\models\\base.gguf", True),
            ("my base", True),
            ("C:base", True),
        ],
    )
    def test_from_line_quotes_only_path_like_bases(self, tmp_path, base, quoted):
        d = _safetensors_adapter(tmp_path)
        first = export.create_adapter_modelfile(base, d).read_text(encoding="utf-8").splitlines()[0]
        escaped = base.replace("\\", "\\\\").replace('"', '\\"')
        assert first == (f'FROM "{escaped}"' if quoted else f"FROM {base}")

    def test_output_path_creates_parent_directories(self, tmp_path):
        d = _safetensors_adapter(tmp_path)
        out = tmp_path / "deep" / "er" / "Modelfile"
        assert export.create_adapter_modelfile("llama3.2", d, out) == out
        assert out.is_file()

    def test_single_file_adapter_defaults_next_to_the_file(self, tmp_path):
        g = tmp_path / "x-lora.gguf"
        g.write_bytes(b"g")
        path = export.create_adapter_modelfile("llama3.2", g)
        assert path == tmp_path.resolve() / "Modelfile"
        adapter_line = path.read_text(encoding="utf-8").splitlines()[1]
        assert adapter_line == 'ADAPTER "' + str(g.resolve()).replace("\\", "\\\\") + '"'

    def test_degrades_for_adapterless_directory(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(GGUFExportError):
            export.create_adapter_modelfile("llama3.2", d)
        assert not (d / "Modelfile").exists()

    def test_control_character_in_base_model_is_rejected(self, tmp_path):
        d = _safetensors_adapter(tmp_path)
        with pytest.raises(ExportError) as exc:
            export.create_adapter_modelfile("llama3.2\x01", d)
        assert exc.value.code == "INPUT_VALIDATION_FAILED"
        assert "base_model contains a control char U+0001 character" in exc.value.message
        assert not (d / "Modelfile").exists()

    def test_control_character_in_system_prompt_is_rejected(self, tmp_path):
        d = _safetensors_adapter(tmp_path)
        with pytest.raises(ExportError) as exc:
            export.create_adapter_modelfile("llama3.2", d, system_prompt="bad\x01value")
        assert "system_prompt contains a control char U+0001 character" in exc.value.message
        assert not (d / "Modelfile").exists()

    def test_tab_is_allowed_in_the_system_prompt(self, tmp_path):
        d = _safetensors_adapter(tmp_path)
        text = export.create_adapter_modelfile("llama3.2", d, system_prompt="a\tb").read_text(encoding="utf-8")
        assert 'SYSTEM "a\tb"' in text

    @pytest.mark.parametrize("ch", ["\n", "\r", "\x00", "\x0c", "\x0b"])
    def test_named_control_characters(self, tmp_path, ch):
        d = _safetensors_adapter(tmp_path)
        with pytest.raises(ExportError) as exc:
            export.create_adapter_modelfile("llama3.2", d, system_prompt=f"a{ch}b")
        names = {"\n": "newline", "\r": "CR", "\x00": "NUL", "\x0c": "form-feed", "\x0b": "vertical-tab"}
        assert names[ch] in exc.value.message


class TestExportOllamaAdapter:
    def test_modelfile_only_never_calls_ollama(self, tmp_path, create, ollama_absent):
        d = _safetensors_adapter(tmp_path)
        result = export.export_ollama_adapter(d, base_model="llama3.2:7b", modelfile_only=True)
        assert create.calls == []
        assert result.format is export.ExportFormat.OLLAMA_ADAPTER
        assert result.path == d.resolve() / "Modelfile" and result.path.is_file()
        assert result.quantization == "llama3.2:my-adapter"  # base tag + derived adapter tag
        assert result.size_mb > 0

    def test_registration_runs_ollama_create_and_removes_the_modelfile(self, tmp_path, create, ollama_present):
        d = _safetensors_adapter(tmp_path)
        result = export.export_ollama_adapter(
            d, base_model="llama3.2", tag="taskA", system_prompt="Be terse", temperature=0.1, context_length=2048
        )
        (call,) = create.calls
        assert call["cmd"] == ["ollama", "create", "llama3.2:taskA", "-f", str(d.resolve() / "Modelfile")]
        assert call["kwargs"]["timeout"] == 600 and call["kwargs"]["check"] is True
        text = call["modelfile_text"]
        assert text.splitlines()[0] == "FROM llama3.2"
        assert "PARAMETER temperature 0.1" in text and "PARAMETER num_ctx 2048" in text
        assert 'SYSTEM "Be terse"' in text
        assert result.quantization == "llama3.2:taskA"
        assert not result.path.exists() and result.size_mb == 0.0

    def test_keep_modelfile(self, tmp_path, create, ollama_present):
        d = _safetensors_adapter(tmp_path)
        result = export.export_ollama_adapter(d, base_model="llama3.2", keep_modelfile=True)
        assert result.path.exists() and result.size_mb > 0

    @pytest.mark.parametrize("tag", ["has space", "a/b", "semi;colon", "tag\nnewline"])
    def test_invalid_tag_fails_before_any_file_is_written(self, tmp_path, create, ollama_present, tag):
        d = _safetensors_adapter(tmp_path)
        with pytest.raises(ExportError) as exc:
            export.export_ollama_adapter(d, base_model="llama3.2", tag=tag)
        assert exc.value.code == "INPUT_VALIDATION_FAILED"
        assert not (d / "Modelfile").exists() and create.calls == []

    def test_unloadable_adapter_degrades_before_registration(self, tmp_path, create, ollama_present):
        d = tmp_path / "nothing"
        d.mkdir()
        with pytest.raises(GGUFExportError):
            export.export_ollama_adapter(d, base_model="llama3.2")
        assert create.calls == []

    def test_missing_cli_keeps_the_modelfile_for_manual_registration(self, tmp_path, create, ollama_absent):
        d = _safetensors_adapter(tmp_path)
        with pytest.raises(OllamaRegistrationError, match="Ollama CLI not found") as exc:
            export.export_ollama_adapter(d, base_model="llama3.2", tag="t")
        assert (d / "Modelfile").is_file()
        assert f"ollama create llama3.2:t -f {d.resolve() / 'Modelfile'}" in (exc.value.suggestion or "")

    def test_timeout(self, tmp_path, create, ollama_present):
        create.raises = subprocess.TimeoutExpired(["ollama"], 600)
        d = _safetensors_adapter(tmp_path)
        with pytest.raises(OllamaRegistrationError, match="timed out after 10 minutes") as exc:
            export.export_ollama_adapter(d, base_model="llama3.2:7b")
        assert "'llama3.2'" in (exc.value.suggestion or "")
        assert not (d / "Modelfile").exists()

    @pytest.mark.parametrize(("stderr", "tail"), [("manifest missing", "manifest missing"), ("", "Unknown error"), (None, "Unknown error")])
    def test_registration_failure(self, tmp_path, create, ollama_present, stderr, tail):
        create.raises = subprocess.CalledProcessError(1, ["ollama"], stderr=stderr)
        d = _safetensors_adapter(tmp_path)
        with pytest.raises(OllamaRegistrationError) as exc:
            export.export_ollama_adapter(d, base_model="llama3.2")
        assert f"ollama create failed: {tail}" in exc.value.message
        assert "ollama pull llama3.2" in (exc.value.suggestion or "")
        assert not (d / "Modelfile").exists()

    def test_modelfile_already_gone_after_registration_is_fine(self, tmp_path, ollama_present, monkeypatch):
        def consuming(cmd, **kwargs):
            Path(cmd[cmd.index("-f") + 1]).unlink()  # the CLI consumed and removed it
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(export, "_run_subprocess_interruptible", consuming)
        d = _safetensors_adapter(tmp_path)
        result = export.export_ollama_adapter(d, base_model="llama3.2", tag="t")
        assert result.size_mb == 0.0 and result.quantization == "llama3.2:t"

    def test_modelfile_cleanup_failure_is_a_warning(self, tmp_path, create, ollama_present, monkeypatch, caplog):
        real_unlink = Path.unlink

        def stuck(self, *a, **k):
            if self.name == "Modelfile":
                raise OSError("locked")
            return real_unlink(self, *a, **k)

        monkeypatch.setattr(Path, "unlink", stuck)
        d = _safetensors_adapter(tmp_path)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = export.export_ollama_adapter(d, base_model="llama3.2")
        assert result.quantization.startswith("llama3.2:")
        assert any("Failed to clean up adapter Modelfile" in r.getMessage() for r in caplog.records)


SHELF_OUTPUT = (
    "NAME                 ID           SIZE     MODIFIED\n"
    "llama3.2:taskA       aaa111       2.0 GB   2 days ago\n"
    "llama3.2:latest      bbb222       2.0 GB   3 days ago\n"
    "llama3.2             ccc333       2.0 GB   3 days ago\n"
    "mistral:taskA        ddd444       4.0 GB   1 day ago\n"
    "\n"
    "llama3.2:short       eee555       1 GB\n"
    "llama3.2:tiny\n"
)


class TestListAdapterShelf:
    def test_lists_only_tagged_variants_of_the_base(self, ollama_present, monkeypatch):
        rec = RunRecorder(subprocess.CompletedProcess([], 0, SHELF_OUTPUT, ""))
        monkeypatch.setattr(export.subprocess, "run", rec)
        shelf = export.list_adapter_shelf("llama3.2:7b")
        assert [(e.model_name, e.base, e.tag) for e in shelf] == [
            ("llama3.2:taskA", "llama3.2", "taskA"),
            ("llama3.2:short", "llama3.2", "short"),
            ("llama3.2:tiny", "llama3.2", "tiny"),
        ]
        first, short, tiny = shelf
        assert (first.size, first.modified) == ("2.0 GB", "2 days ago")
        assert (short.size, short.modified) == ("1 GB", "")
        assert (tiny.size, tiny.modified) == ("", "")
        assert rec.calls[0][0] == ["ollama", "list"]

    def test_missing_cli(self, ollama_absent, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert export.list_adapter_shelf("llama3.2") == []
        assert any("empty shelf" in r.getMessage() and "llama3.2" in r.getMessage() for r in caplog.records)

    def test_daemon_error(self, ollama_present, monkeypatch, caplog):
        monkeypatch.setattr(
            export.subprocess, "run",
            RunRecorder(raises=subprocess.CalledProcessError(1, ["ollama"], stderr="daemon down")),
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert export.list_adapter_shelf("llama3.2") == []
        assert any("daemon down" in r.getMessage() for r in caplog.records)
        monkeypatch.setattr(
            export.subprocess, "run",
            RunRecorder(raises=subprocess.CalledProcessError(1, ["ollama"], stderr="")),
        )
        assert export.list_adapter_shelf("llama3.2") == []

    def test_timeout(self, ollama_present, monkeypatch, caplog):
        monkeypatch.setattr(export.subprocess, "run", RunRecorder(raises=subprocess.TimeoutExpired(["o"], 30)))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert export.list_adapter_shelf("llama3.2") == []
        assert any("timed out after 30s" in r.getMessage() for r in caplog.records)

