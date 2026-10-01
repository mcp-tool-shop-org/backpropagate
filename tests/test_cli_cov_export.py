"""Coverage tests for ``cmd_export`` and its HF-token helpers in cli.py.

Real: the LoRA export (``export_lora`` copies a real adapter directory on
tmp_path), path sandboxing, argument parsing, exit codes, output text.
Mocked (real boundaries): model loading for merge/GGUF, the llama.cpp / merge
writers (``export_merged`` / ``export_gguf``), ``export_ollama_adapter`` and
``register_with_ollama`` (subprocess to ``ollama``), and ``push_to_hub``
(network).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from backpropagate import cli
from backpropagate.exceptions import (
    BackpropagateError,
    ExportError,
    PartialSuccess,
    UserInputError,
)
from tests.helpers.cli_cov_support import parse


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    """A real on-disk adapter dir (the files export_lora copies) under cwd."""
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "lora"
    d.mkdir()
    (d / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"\x00" * 2048)
    return d


def _result(path: Path, **kw):
    return SimpleNamespace(
        path=path, size_mb=1.5, export_time_seconds=2.0, quantization=kw.get("quantization"),
        deferred_quantization=kw.get("deferred_quantization"),
    )



class _PosixOs:
    """``os`` as cli sees it, reporting ``name == "posix"``; everything else is the real module.

    Patching the global ``os.name`` instead makes pathlib build ``PosixPath`` on
    Windows for every caller, pytest's own cache included (INTERNALERROR on CI).
    """

    name = "posix"

    def __getattr__(self, attr):
        return getattr(os, attr)

class TestExportPreflight:
    def test_no_model_path(self, capsys):
        assert cli.cmd_export(parse(["export"])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "No model path specified." in captured.err
        assert "backprop export ./output/lora --format lora" in captured.out

    def test_nul_byte_in_path(self, capsys):
        args = parse(["export"])
        args.model_path = "a\x00b"
        assert cli.cmd_export(args) == cli.EXIT_USER_ERROR
        assert "embedded NUL" in capsys.readouterr().err

    def test_ollama_requires_gguf(self):
        with pytest.raises(UserInputError, match="--ollama requires --format=gguf"):
            cli.cmd_export(parse(["export", "x", "--ollama"]))

    def test_ollama_adapter_requires_base_model(self, capsys):
        assert cli.cmd_export(parse(["export", "x", "--format", "ollama-adapter"])) == cli.EXIT_USER_ERROR
        assert "--format ollama-adapter requires --base-model" in capsys.readouterr().err

    def test_missing_model_path_on_disk(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert cli.cmd_export(parse(["export", "nope"])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "Model path not found: nope" in captured.err
        assert "huggingface-cli download" in captured.out

    def test_relative_path_escaping_cwd(self, tmp_path, monkeypatch, capsys):
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        assert cli.cmd_export(parse(["export", "../lora"])) == cli.EXIT_USER_ERROR
        assert "Security error" in capsys.readouterr().err

    def test_invalid_model_path_oserror_from_path(self, monkeypatch, capsys):
        def bad_expand(self):
            raise OSError("cannot expand ~")

        monkeypatch.setattr(Path, "expanduser", bad_expand)
        assert cli.cmd_export(parse(["export", "~x"])) == cli.EXIT_USER_ERROR
        err = capsys.readouterr()
        assert "Invalid model path: cannot expand ~" in err.err
        assert "~ left literal" in err.out

    def test_invalid_model_path_from_safe_path(self, monkeypatch, capsys, tmp_path):
        monkeypatch.chdir(tmp_path)

        def bad(*a, **k):
            raise ValueError("bad drive")

        monkeypatch.setattr(cli, "safe_path", bad)
        assert cli.cmd_export(parse(["export", "x"])) == cli.EXIT_USER_ERROR
        err = capsys.readouterr()
        assert "Invalid model path: bad drive" in err.err
        assert "invalid drive letter" in err.out

    def test_output_traversal_rejected(self, adapter, capsys):
        assert cli.cmd_export(parse(["export", "lora", "--output", "../escape"])) == cli.EXIT_USER_ERROR
        assert "output path escapes its allowed base" in capsys.readouterr().err

    def test_invalid_output_path(self, adapter, monkeypatch, capsys):
        real = cli.safe_path
        calls = {"n": 0}

        def flaky(p, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("read-only mount")
            return real(p, **kw)

        monkeypatch.setattr(cli, "safe_path", flaky)
        assert cli.cmd_export(parse(["export", "lora", "--output", "out"])) == cli.EXIT_USER_ERROR
        assert "Invalid output path: read-only mount" in capsys.readouterr().err


class TestExportLora:
    def test_real_lora_export_relative(self, adapter, capsys):
        assert cli.cmd_export(parse(["export", "lora", "--output", "out_lora", "--no-model-card"])) == cli.EXIT_OK
        captured = capsys.readouterr()
        assert "Export complete!" in captured.out
        assert (adapter.parent / "out_lora" / "adapter_config.json").exists()
        assert (adapter.parent / "out_lora" / "adapter_model.safetensors").stat().st_size == 2048
        assert "Absolute" not in captured.out

    def test_absolute_paths_warn_but_work(self, adapter, tmp_path, capsys):
        out = tmp_path / "abs_out"
        args = parse(["export", str(adapter), "--output", str(out)])
        assert cli.cmd_export(args) == cli.EXIT_OK
        text = capsys.readouterr().out
        assert "Absolute --model_path supplied" in text
        assert "Absolute --output supplied" in text
        assert (out / "adapter_config.json").exists()

    def test_default_output_next_to_model(self, adapter, tmp_path, capsys):
        assert cli.cmd_export(parse(["export", str(adapter)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Absolute --output supplied" not in out
        assert (tmp_path / "lora" / "lora" / "adapter_config.json").exists() or \
            (adapter.parent / "lora").exists()

    def test_observability_failures_ignored(self, adapter, monkeypatch, capsys):
        def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr("backpropagate.logging_config.bind_run_context", boom)
        monkeypatch.setattr("backpropagate.logging_config.get_logger", boom)
        assert cli.cmd_export(parse(["export", "lora", "--output", "o2", "--no-model-card"])) == cli.EXIT_OK


class TestExportFormats:
    def test_merged(self, adapter, monkeypatch, capsys):
        calls = {}
        monkeypatch.setattr("backpropagate.export.load_model_for_export", lambda p: ("M", "T"))

        def fake_merged(**kw):
            calls.update(kw)
            return _result(adapter.parent / "m")

        monkeypatch.setattr("backpropagate.export.export_merged", fake_merged)
        assert cli.cmd_export(parse(["export", "lora", "--format", "merged", "--output", "m"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Loading model for merge" in out and "Merging adapters" in out
        assert calls["model"] == "M" and calls["tokenizer"] == "T"
        assert calls["emit_model_card"] is True

    def test_gguf_with_ollama_registration_success_and_deferred(self, adapter, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr("backpropagate.export.load_model_for_export", lambda p: ("M", "T"))

        def fake_gguf(**kw):
            seen["gguf"] = kw
            return _result(adapter.parent / "g.gguf", deferred_quantization="q4_k_m")

        def fake_register(path, name, quantize=None):
            seen["register"] = (path, name, quantize)
            return True

        monkeypatch.setattr("backpropagate.export.export_gguf", fake_gguf)
        monkeypatch.setattr("backpropagate.export.register_with_ollama", fake_register)
        argv = ["export", "lora", "--format", "gguf", "--output", "g", "--ollama", "--ollama-name", "mymodel"]
        assert cli.cmd_export(parse(argv)) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Quantization: q4_k_m" in out
        assert "Ollama will quantize the f16 GGUF to q4_k_m" in out
        assert "Registered with Ollama: mymodel" in out
        assert "ollama run mymodel" in out
        assert seen["gguf"]["defer_quantization_to_ollama"] is True
        assert seen["register"][1:] == ("mymodel", "q4_k_m")

    def test_gguf_registration_failure_is_partial_success(self, adapter, monkeypatch, capsys):
        monkeypatch.setattr("backpropagate.export.load_model_for_export", lambda p: ("M", "T"))
        monkeypatch.setattr("backpropagate.export.export_gguf",
                            lambda **kw: _result(adapter.parent / "g.gguf"))
        monkeypatch.setattr("backpropagate.export.register_with_ollama", lambda *a, **k: False)
        argv = ["export", "lora", "--format", "gguf", "--output", "g", "--ollama"]
        assert cli.cmd_export(parse(argv)) == cli.EXIT_PARTIAL_SUCCESS
        captured = capsys.readouterr()
        assert "'lora' was NOT created" in captured.err
        assert "ollama serve" in captured.out

    def test_ollama_adapter(self, adapter, monkeypatch, capsys):
        seen = {}

        def fake_adapter(path, base_model, tag):
            seen.update(path=path, base_model=base_model, tag=tag)
            return _result(adapter.parent / "oa", quantization="llama3.2:taskA")

        monkeypatch.setattr("backpropagate.export.export_ollama_adapter", fake_adapter)
        argv = ["export", "lora", "--format", "ollama-adapter", "--base-model", "llama3.2",
                "--adapter-tag", "taskA", "--output", "oa"]
        assert cli.cmd_export(parse(argv)) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Registered with Ollama: llama3.2:taskA" in out
        assert "ollama run llama3.2:taskA" in out
        assert seen["base_model"] == "llama3.2" and seen["tag"] == "taskA"

    def test_ollama_adapter_without_registered_name(self, adapter, monkeypatch, capsys):
        monkeypatch.setattr("backpropagate.export.export_ollama_adapter",
                            lambda *a, **k: _result(adapter.parent / "oa"))
        argv = ["export", "lora", "--format", "ollama-adapter", "--base-model", "llama3.2",
                "--output", "oa"]
        assert cli.cmd_export(parse(argv)) == cli.EXIT_OK
        assert "Registered with Ollama" not in capsys.readouterr().out

    def test_unknown_format_defensive_branch(self, adapter, capsys):
        args = parse(["export", "lora", "--output", "o"])
        args.format = "weird"
        assert cli.cmd_export(args) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "Unknown format: 'weird'" in captured.err
        assert "Supported: --format lora" in captured.out


class TestExportHubPush:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("BACKPROPAGATE_QUIET_TOKEN_HINT", raising=False)
        monkeypatch.setattr(cli, "_ENV_TOKEN_CALIBRATION_WARNED", False)

    def _argv(self, *extra):
        return ["export", "lora", "--output", "o", "--no-model-card", "--push-to-hub", "me/repo", *extra]

    def test_push_with_token_file(self, adapter, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr("backpropagate.export.push_to_hub",
                            lambda **kw: seen.update(kw) or "https://hf.co/me/repo")
        tok = adapter.parent / "tok.txt"
        tok.write_text("hf_filetoken\n", encoding="utf-8")
        argv = self._argv("--hub-token-file", str(tok), "--hub-private")
        assert cli.cmd_export(parse(argv)) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Pushed to Hub: https://hf.co/me/repo" in out
        assert seen["token"] == "hf_filetoken" and seen["private"] is True
        assert seen["repo_id"] == "me/repo"
        assert Path(seen["local_root"] if "local_root" in seen else seen["local_path"]).is_dir()

    def test_push_inline_token_warns(self, adapter, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr("backpropagate.export.push_to_hub",
                            lambda **kw: seen.update(kw) or "https://hf.co/x")
        assert cli.cmd_export(parse(self._argv("--hub-token", "hf_inline"))) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "--hub-token was passed on the command line" in out
        assert seen["token"] == "hf_inline"

    def test_push_env_token_calibration_note(self, adapter, monkeypatch, capsys):
        monkeypatch.setenv("HF_TOKEN", "hf_env")
        monkeypatch.setattr("backpropagate.export.push_to_hub", lambda **kw: "https://hf.co/x")
        assert cli.cmd_export(parse(self._argv())) == cli.EXIT_OK
        assert "Using HF_TOKEN from the environment" in capsys.readouterr().out

    def test_token_and_file_mutex(self, adapter, capsys):
        argv = self._argv("--hub-token", "t", "--hub-token-file", "f")
        assert cli.cmd_export(parse(argv)) == cli.EXIT_USER_ERROR
        err = capsys.readouterr().err
        assert "mutually exclusive" in err
        assert "[INPUT_VALIDATION_FAILED]" in err

    def test_push_failure_is_partial_success(self, adapter, monkeypatch, capsys):
        def fail(**kw):
            raise BackpropagateError("upload denied", suggestion="check token")

        monkeypatch.setattr("backpropagate.export.push_to_hub", fail)
        assert cli.cmd_export(parse(self._argv())) == cli.EXIT_PARTIAL_SUCCESS
        captured = capsys.readouterr()
        assert "Hub push failed: upload denied" in captured.err
        assert "Suggestion: check token" in captured.out


class TestExportErrorMapping:
    @pytest.fixture(autouse=True)
    def _boom(self):
        self.exc = None

    def _run(self, adapter, monkeypatch, exc, verbose=False):
        def raiser(**kw):
            raise exc

        monkeypatch.setattr("backpropagate.export.load_model_for_export", lambda p: ("M", "T"))
        monkeypatch.setattr("backpropagate.export.export_merged", raiser)
        args = parse(["export", "lora", "--format", "merged", "--output", "m"])
        args.verbose = verbose
        return cli.cmd_export(args)

    @pytest.mark.parametrize("verbose", [False, True])
    def test_export_error(self, adapter, monkeypatch, capsys, verbose):
        code = self._run(adapter, monkeypatch, ExportError("merge failed", suggestion="free disk"), verbose)
        assert code == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "Export error: merge failed" in captured.err
        assert "Suggestion: free disk" in captured.out

    def test_user_input_error(self, adapter, monkeypatch, capsys):
        code = self._run(adapter, monkeypatch, UserInputError("nope", hint="try again"))
        assert code == cli.EXIT_USER_ERROR
        assert "Suggestion: try again" in capsys.readouterr().out

    def test_partial_success(self, adapter, monkeypatch, capsys):
        exc = PartialSuccess("some", total_items=2, succeeded=1, failed=1, suggestion="retry")
        assert self._run(adapter, monkeypatch, exc) == cli.EXIT_PARTIAL_SUCCESS
        assert "Suggestion: retry" in capsys.readouterr().out

    @pytest.mark.parametrize("verbose", [False, True])
    def test_generic_backpropagate_error(self, adapter, monkeypatch, capsys, verbose):
        code = self._run(adapter, monkeypatch, BackpropagateError("generic", suggestion="s"), verbose)
        assert code == cli.EXIT_RUNTIME_ERROR
        assert "Suggestion: s" in capsys.readouterr().out

    def test_unexpected_redacted(self, adapter, monkeypatch, capsys):
        code = self._run(adapter, monkeypatch, RuntimeError("Authorization: Bearer abcdef1234567890"))
        assert code == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "abcdef1234567890" not in captured.err
        assert "Run with --verbose" in captured.out

    def test_unexpected_verbose(self, adapter, monkeypatch, capsys):
        code = self._run(adapter, monkeypatch, RuntimeError("kaboom"), verbose=True)
        assert code == cli.EXIT_RUNTIME_ERROR
        assert "Traceback" in capsys.readouterr().err


class TestHubTokenFile:
    def test_missing_file(self, tmp_path):
        with pytest.raises(UserInputError, match="path does not exist") as ei:
            cli._read_hub_token_file(str(tmp_path / "nope"), flag_name="--hub-token-file")
        assert ei.value.code == "INPUT_VALIDATION_FAILED"

    def test_empty_file(self, tmp_path):
        f = tmp_path / "t"
        f.write_text("  \n", encoding="utf-8")
        with pytest.raises(UserInputError, match="is empty"):
            cli._read_hub_token_file(str(f), flag_name="--hub-token-file")

    def test_reads_and_strips(self, tmp_path):
        f = tmp_path / "t"
        f.write_text(" hf_abc \n", encoding="utf-8")
        assert cli._read_hub_token_file(str(f), flag_name="--x") == "hf_abc"

    def test_unreadable_file(self, tmp_path, monkeypatch):
        f = tmp_path / "t"
        f.write_text("hf_abc", encoding="utf-8")

        def deny(self, *a, **k):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "read_text", deny)
        with pytest.raises(UserInputError, match="could not be read: denied"):
            cli._read_hub_token_file(str(f), flag_name="--x")

    def test_posix_wide_mode_warns(self, tmp_path, monkeypatch, capsys):
        f = tmp_path / "t"
        f.write_text("hf_abc", encoding="utf-8")
        monkeypatch.setattr(cli, "os", _PosixOs())
        fake_stat = SimpleNamespace(st_mode=0o100644)
        monkeypatch.setattr(Path, "stat", lambda self, **k: fake_stat)
        assert cli._read_hub_token_file(str(f), flag_name="--hub-token-file") == "hf_abc"
        assert "mode is 0o644" in capsys.readouterr().out

    def test_posix_stat_oserror_is_ignored(self, tmp_path, monkeypatch):
        f = tmp_path / "t"
        f.write_text("hf_abc", encoding="utf-8")
        monkeypatch.setattr(cli, "os", _PosixOs())

        real_stat = Path.stat
        calls = {"n": 0}

        def bad_stat(self, **k):
            calls["n"] += 1
            if calls["n"] == 2:  # 1st call is exists(); 2nd is the mode probe
                raise OSError("stat failed")
            return real_stat(self, **k)

        monkeypatch.setattr(Path, "stat", bad_stat)
        assert cli._read_hub_token_file(str(f), flag_name="--x") == "hf_abc"


class TestEnvTokenCalibration:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("BACKPROPAGATE_QUIET_TOKEN_HINT", raising=False)
        monkeypatch.setattr(cli, "_ENV_TOKEN_CALIBRATION_WARNED", False)

    def test_no_env_token_is_silent(self, capsys):
        cli._warn_env_token_calibration()
        assert capsys.readouterr().out == ""
        assert cli._ENV_TOKEN_CALIBRATION_WARNED is False

    def test_quiet_env_suppresses(self, monkeypatch, capsys):
        monkeypatch.setenv("HF_TOKEN", "x")
        monkeypatch.setenv("BACKPROPAGATE_QUIET_TOKEN_HINT", "1")
        cli._warn_env_token_calibration()
        assert capsys.readouterr().out == ""
        assert cli._ENV_TOKEN_CALIBRATION_WARNED is True

    def test_warns_once(self, monkeypatch, capsys):
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "x")
        cli._warn_env_token_calibration()
        first = capsys.readouterr().out
        cli._warn_env_token_calibration()
        second = capsys.readouterr().out
        assert "Using HF_TOKEN from the environment" in first
        assert second == ""
