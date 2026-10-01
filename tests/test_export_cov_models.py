"""Coverage tests for the model-export paths in ``backpropagate.export``.

LoRA / merged / GGUF export and ``load_model_for_export`` run against a REAL
tiny Llama + LoRA adapter (random weights built in-process, CPU only, see
``tests/helpers/tiny_models.py``). Real files are written to ``tmp_path`` and
re-read to assert what landed on disk (for example, a merged export is
reloaded and its logits compared with the adapter-active model).

What is mocked (and nothing else):

* ``export._run_subprocess_interruptible`` - the boundary to llama.cpp's
  ``convert_hf_to_gguf.py`` / ``llama-quantize``. The stand-in records the
  exact argv + kwargs and writes the output file the real tool would.
* ``export._has_unsloth`` / ``export._unsloth_gguf_ready`` and a stub
  ``unsloth.FastLanguageModel`` (Unsloth is GPU-only).
* ``torch.cuda.is_available`` (always False: nothing here touches a GPU).
* ``LlamaForCausalLM.push_to_hub`` (network) in the merged-export push test.
* A handful of ``pathlib`` / ``shutil`` calls to force OS errors.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model

from backpropagate import export
from backpropagate.exceptions import (
    ExportError,
    GGUFExportError,
    InvalidSettingError,
    MergeExportError,
)
from tests.helpers.tiny_models import tiny_llama, tiny_tokenizer

LOGGER = "backpropagate.export"


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """GPU boundary: report no CUDA so nothing here can touch a device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def make_peft(*, layers: int = 2, dtype: torch.dtype = torch.float32):
    torch.manual_seed(1234)  # LoRA A init draws from the global RNG
    base = tiny_llama(layers=layers, dtype=dtype)
    model = get_peft_model(
        base,
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0.0),
    )
    with torch.no_grad():  # non-zero B so the merge actually changes the weights
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0.0, 1.0)
    return model


IDS = torch.tensor([[1, 5, 6, 7, 8]])


def _logits(model) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(IDS).logits.clone()


# =============================================================================
# export_lora
# =============================================================================


class TestExportLora:
    def test_peft_model_adapter_is_saved_and_reloadable(self, tmp_path):
        model = make_peft()
        out = tmp_path / "adapters" / "lora"
        result = export.export_lora(model, out, emit_model_card=False)
        assert result.format is export.ExportFormat.LORA
        assert result.path == out
        assert result.size_mb > 0 and result.export_time_seconds >= 0
        assert (out / "adapter_config.json").is_file()
        assert (out / "adapter_model.safetensors").is_file()
        assert not (tmp_path / "adapters" / "lora.partial").exists()

        from peft import PeftModel

        reloaded = PeftModel.from_pretrained(tiny_llama(layers=2), str(out))
        # same adapter -> same function (base weights equal because same seed)
        assert torch.allclose(_logits(reloaded), _logits(model), atol=1e-5)

    def test_re_export_keeps_other_files_and_wipes_stale_partial(self, tmp_path):
        out = tmp_path / "lora"
        out.mkdir()
        (out / "old.txt").write_text("stale", encoding="utf-8")
        stale = tmp_path / "lora.partial"
        stale.mkdir()
        (stale / "junk").write_text("junk", encoding="utf-8")
        export.export_lora(make_peft(), out, emit_model_card=False)
        # Only the adapter files are replaced; the operator's file survives.
        assert (out / "old.txt").read_text(encoding="utf-8") == "stale"
        assert (out / "adapter_config.json").is_file()
        assert not stale.exists()

    def test_copy_from_saved_adapter_directory(self, tmp_path):
        src = tmp_path / "src"
        make_peft().save_pretrained(src)
        (src / "README.md").write_text("not copied", encoding="utf-8")
        out = tmp_path / "out"
        export.export_lora(src, out, emit_model_card=False)
        assert sorted(p.name for p in out.iterdir()) == ["adapter_config.json", "adapter_model.safetensors"]

    def test_missing_source_path(self, tmp_path):
        with pytest.raises(ExportError, match="Source model path does not exist"):
            export.export_lora(tmp_path / "nope", tmp_path / "out")
        assert not (tmp_path / "out.partial").exists()

    def test_directory_without_adapter_files(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "other.txt").write_text("x", encoding="utf-8")
        with pytest.raises(ExportError, match="No adapter files found"):
            export.export_lora(src, tmp_path / "out")
        assert not (tmp_path / "out.partial").exists()

    def test_non_peft_object_is_rejected(self, tmp_path):
        with pytest.raises(ExportError, match="Cannot export LoRA from Linear"):
            export.export_lora(torch.nn.Linear(2, 2), tmp_path / "out")

    def test_save_failure_is_wrapped_and_partial_cleaned(self, tmp_path):
        model = make_peft()

        def boom(path, **kw):
            raise RuntimeError("disk exploded")

        model.save_pretrained = boom  # type: ignore[method-assign]
        with pytest.raises(ExportError, match="LoRA export failed: disk exploded") as exc:
            export.export_lora(model, tmp_path / "out")
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert not (tmp_path / "out.partial").exists()
        assert not (tmp_path / "out").exists()

    def test_parent_directory_errors_are_structured(self, tmp_path, monkeypatch):
        real_mkdir = Path.mkdir

        def deny(self, *a, **k):
            if self.name == "locked":
                raise PermissionError("no write access")
            if self.name == "weird":
                raise OSError("device busy")
            return real_mkdir(self, *a, **k)

        monkeypatch.setattr(Path, "mkdir", deny)
        with pytest.raises(ExportError, match="Cannot create parent directory") as exc:
            export.export_lora(make_peft(), tmp_path / "locked" / "out")
        assert "Check write permissions" in (exc.value.suggestion or "")
        with pytest.raises(ExportError, match="Failed to create parent directory"):
            export.export_lora(make_peft(), tmp_path / "weird" / "out")

    def test_partial_directory_creation_failure(self, tmp_path, monkeypatch):
        real_mkdir = Path.mkdir

        def deny(self, *a, **k):
            if self.name.endswith(".partial"):
                raise OSError("cannot create scratch dir")
            return real_mkdir(self, *a, **k)

        monkeypatch.setattr(Path, "mkdir", deny)
        with pytest.raises(ExportError, match="Failed to create partial directory"):
            export.export_lora(make_peft(), tmp_path / "out")


# =============================================================================
# export_merged
# =============================================================================


class _StubTokenizer:
    def __init__(self, fail_save=False, fail_push=False):
        self.saved_to = None
        self.pushed_to = None
        self.fail_save = fail_save
        self.fail_push = fail_push

    def save_pretrained(self, path):
        if self.fail_save:
            raise OSError("tokenizer disk full")
        self.saved_to = Path(path)
        (Path(path) / "tokenizer.stub").write_text("tok", encoding="utf-8")

    def push_to_hub(self, repo_id):
        if self.fail_push:
            raise RuntimeError("hub rejected tokenizer")
        self.pushed_to = repo_id


class TestExportMerged:
    def test_merged_checkpoint_matches_the_adapter_model(self, tmp_path):
        model = make_peft()
        expected = _logits(model)
        out = tmp_path / "merged"
        result = export.export_merged(model, tiny_tokenizer(), out, emit_model_card=False)
        assert result.format is export.ExportFormat.MERGED
        assert result.path == out and result.size_mb > 0
        assert (out / "config.json").is_file()
        assert list(out.glob("*.safetensors"))
        assert (out / "tokenizer.json").is_file()

        from transformers import AutoModelForCausalLM

        merged = AutoModelForCausalLM.from_pretrained(str(out))
        assert not export._is_peft_model(merged)
        # the adapter genuinely changes the function ...
        assert (expected - _logits(tiny_llama(layers=2))).abs().max() > 0.2  # 4x the tolerance below
        # ... and the merged checkpoint reproduces it (saved as bf16, so loosely)
        assert next(merged.parameters()).dtype == torch.bfloat16
        assert torch.allclose(_logits(merged).float(), expected, atol=0.05, rtol=0.05)

    def test_default_emits_model_card(self, tmp_path):
        out = tmp_path / "merged"
        export.export_merged(make_peft(), tiny_tokenizer(), out, base_model="org/base")
        card = (out / "model_card.md").read_text(encoding="utf-8")
        assert "org/base" in card and "Incomplete provenance" in card

    def test_float32_merge_is_saved_as_bf16(self, tmp_path):
        """The FP8-trained-model finding: an f32 merge is down-cast before saving."""
        from safetensors.torch import load_file

        out = tmp_path / "merged"
        export.export_merged(make_peft(dtype=torch.float32), tiny_tokenizer(), out, emit_model_card=False)
        tensors = load_file(str(next(out.glob("*.safetensors"))))
        assert {t.dtype for t in tensors.values()} == {torch.bfloat16}

    def test_non_peft_model_rejected(self, tmp_path):
        with pytest.raises(MergeExportError, match="Cannot merge non-PeftModel"):
            export.export_merged(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "m")

    def test_output_directory_errors(self, tmp_path, monkeypatch):
        real_mkdir = Path.mkdir

        def deny(self, *a, **k):
            if self.name == "locked":
                raise PermissionError("denied")
            if self.name == "weird":
                raise OSError("busy")
            return real_mkdir(self, *a, **k)

        monkeypatch.setattr(Path, "mkdir", deny)
        with pytest.raises(MergeExportError, match="Cannot create output directory"):
            export.export_merged(make_peft(), tiny_tokenizer(), tmp_path / "locked")
        with pytest.raises(MergeExportError, match="Failed to create output directory"):
            export.export_merged(make_peft(), tiny_tokenizer(), tmp_path / "weird")

    def test_insufficient_disk_space_fails_before_merging(self, tmp_path, monkeypatch):
        merged = []
        model = make_peft()
        monkeypatch.setattr(model, "merge_and_unload", lambda: merged.append(1))
        monkeypatch.setattr(
            export.shutil, "disk_usage", lambda p: types.SimpleNamespace(total=1, used=1, free=0)
        )
        with pytest.raises(MergeExportError, match="Insufficient disk space"):
            export.export_merged(model, tiny_tokenizer(), tmp_path / "m")
        assert merged == []

    def test_merge_failure_is_wrapped(self, tmp_path, monkeypatch):
        model = make_peft()

        def boom():
            raise RuntimeError("adapter mismatch")

        monkeypatch.setattr(model, "merge_and_unload", boom)
        with pytest.raises(MergeExportError, match="Failed to merge LoRA adapters: adapter mismatch"):
            export.export_merged(model, tiny_tokenizer(), tmp_path / "m")

    def test_save_failure_is_wrapped(self, tmp_path):
        with pytest.raises(MergeExportError, match="Failed to save merged model: tokenizer disk full"):
            export.export_merged(make_peft(), _StubTokenizer(fail_save=True), tmp_path / "m")

    def test_push_requires_repo_id(self, tmp_path):
        with pytest.raises(MergeExportError, match="repo_id required when push_to_hub=True") as exc:
            export.export_merged(make_peft(), _StubTokenizer(), tmp_path / "m", push_to_hub=True)
        assert "username/model-name" in (exc.value.suggestion or "")

    def test_push_pushes_model_and_tokenizer(self, tmp_path, monkeypatch):
        """Mocks: ``LlamaForCausalLM.push_to_hub`` (network)."""
        from transformers import LlamaForCausalLM

        pushed = []
        monkeypatch.setattr(LlamaForCausalLM, "push_to_hub", lambda self, repo: pushed.append(repo))
        tok = _StubTokenizer()
        result = export.export_merged(
            make_peft(), tok, tmp_path / "m", push_to_hub=True, repo_id="alice/merged",
            emit_model_card=False,
        )
        assert pushed == ["alice/merged"]
        assert tok.pushed_to == "alice/merged"
        assert result.path == tmp_path / "m"

    def test_push_failure_is_wrapped(self, tmp_path, monkeypatch):
        from transformers import LlamaForCausalLM

        def refuse(self, repo):
            raise RuntimeError("401 unauthorized")

        monkeypatch.setattr(LlamaForCausalLM, "push_to_hub", refuse)
        with pytest.raises(MergeExportError, match="Failed to push to HuggingFace Hub: 401 unauthorized") as exc:
            export.export_merged(
                make_peft(), _StubTokenizer(), tmp_path / "m", push_to_hub=True, repo_id="a/b"
            )
        assert "token" in (exc.value.suggestion or "")


# =============================================================================
# export_gguf - llama.cpp fallback (boundary: the converter / quantizer binaries)
# =============================================================================


class ToolRunner:
    """Stands in for ``export._run_subprocess_interruptible``.

    Records every invocation and writes the artefact the external tool would
    have produced. ``fail_converter`` / ``fail_quantize`` raise the failure a
    non-zero exit or a hang would.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.fail_converter: BaseException | None = None
        self.fail_quantize: BaseException | None = None
        self.write_output = True
        self.merged_listing: list[str] = []

    def __call__(self, cmd, **kwargs):
        record = {"cmd": [str(c) for c in cmd], "kwargs": kwargs}
        self.calls.append(record)
        if "--outfile" in cmd:  # convert_hf_to_gguf.py
            merged_dir = Path(cmd[2])
            self.merged_listing = sorted(p.name for p in merged_dir.iterdir())
            if self.fail_converter:
                raise self.fail_converter
            if self.write_output:
                Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF" + b"\0" * 2048)
        else:  # llama-quantize <in> <out> <TYPE>
            if self.fail_quantize:
                raise self.fail_quantize
            Path(cmd[2]).write_bytes(b"GGUF" + b"\1" * 512)
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def runner(monkeypatch):
    r = ToolRunner()
    monkeypatch.setattr(export, "_run_subprocess_interruptible", r)
    return r


@pytest.fixture
def llama_cpp(tmp_path, monkeypatch):
    """A fake llama.cpp checkout discovered through BACKPROPAGATE_LLAMA_CPP_PATH."""
    root = tmp_path / "llama.cpp"
    root.mkdir()
    script = root / "convert_hf_to_gguf.py"
    script.write_text("# stub converter\n", encoding="utf-8")
    monkeypatch.setenv("BACKPROPAGATE_LLAMA_CPP_PATH", str(script))
    monkeypatch.setattr(export, "_has_unsloth", lambda: False)
    monkeypatch.setattr(export.shutil, "which", lambda name: None)
    return script


def _add_quantizer(script: Path) -> Path:
    exe = script.parent / "llama-quantize"
    exe.write_bytes(b"")
    return exe


class TestExportGgufLlamaCppFallback:
    def test_q8_0_converts_directly_and_names_the_model(self, tmp_path, llama_cpp, runner):
        out = tmp_path / "gguf"
        result = export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), out, "q8_0", model_name="mymodel")
        assert result.format is export.ExportFormat.GGUF
        assert result.path == out / "mymodel-q8_0.gguf"
        assert result.quantization == "q8_0" and result.deferred_quantization is None
        assert result.size_mb > 0
        (call,) = runner.calls
        assert call["cmd"] == [
            sys.executable, str(llama_cpp), str(out / "merged_temp"),
            "--outfile", str(out / "mymodel-q8_0.gguf"),
            "--outtype", "q8_0", "--model-name", "mymodel",
        ]
        assert call["kwargs"]["timeout"] == 1800
        assert call["kwargs"]["check"] is True and call["kwargs"]["encoding"] == "utf-8"
        # the converter saw a complete HF checkpoint ...
        assert "config.json" in runner.merged_listing
        assert any(n.endswith(".safetensors") for n in runner.merged_listing)
        assert "tokenizer.json" in runner.merged_listing
        # ... and the scratch merge is cleaned up afterwards
        assert not (out / "merged_temp").exists()
        assert "q8_0" in (out / "model_card.md").read_text(encoding="utf-8")

    def test_peft_model_is_merged_before_conversion(self, tmp_path, llama_cpp, runner):
        out = tmp_path / "gguf"
        export.export_gguf(make_peft(), tiny_tokenizer(), out, export.GGUFQuantization.F16)
        (call,) = runner.calls
        assert call["cmd"][call["cmd"].index("--outtype") + 1] == "f16"
        # a merged checkpoint has no adapter files in it
        assert not any(n.startswith("adapter_") for n in runner.merged_listing)
        assert "config.json" in runner.merged_listing

    def test_quantization_is_case_insensitive_and_validated(self, tmp_path, llama_cpp, runner):
        r = export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "a", "Q8_0")
        assert r.quantization == "q8_0"
        with pytest.raises(InvalidSettingError) as exc:
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "b", "q9_z")
        assert exc.value.code == "CONFIG_INVALID_SETTING"
        assert "q4_k_m" in exc.value.expected or "q4_k_m" in (exc.value.suggestion or "")

    def test_model_name_path_components_are_stripped(self, tmp_path, llama_cpp, runner, caplog):
        out = tmp_path / "gguf"
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            r = export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), out, "f16", model_name="../../etc/evil")
        assert r.path == out / "evil-f16.gguf"
        assert any("path components" in rec.getMessage() for rec in caplog.records)
        r2 = export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "g2", "f16", model_name="..")
        assert r2.path == tmp_path / "g2" / "model-f16.gguf"

    def test_quantizing_level_converts_to_f16_then_runs_llama_quantize(self, tmp_path, llama_cpp, runner):
        quantizer = _add_quantizer(llama_cpp)
        out = tmp_path / "gguf"
        result = export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), out, "q4_k_m", model_name="m")
        convert, quant = runner.calls
        assert convert["cmd"][convert["cmd"].index("--outtype") + 1] == "f16"
        assert convert["cmd"][convert["cmd"].index("--outfile") + 1] == str(out / "m-f16.gguf")
        assert quant["cmd"] == [str(quantizer), str(out / "m-f16.gguf"), str(out / "m-q4_k_m.gguf"), "Q4_K_M"]
        assert result.path == out / "m-q4_k_m.gguf"
        assert result.quantization == "q4_k_m" and result.deferred_quantization is None
        assert not (out / "m-f16.gguf").exists()  # intermediate removed

    @pytest.mark.parametrize(
        ("level", "tool_type"), [("q5_k_m", "Q5_K_M"), ("q4_0", "Q4_0"), ("q2_k", "Q2_K")]
    )
    def test_every_quantize_level_maps_to_a_llama_quantize_type(self, tmp_path, llama_cpp, runner, level, tool_type):
        _add_quantizer(llama_cpp)
        export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "o", level)
        assert runner.calls[1]["cmd"][-1] == tool_type

    def test_quantize_failure_keeps_the_f16_and_reports_exit_code(self, tmp_path, llama_cpp, runner):
        _add_quantizer(llama_cpp)
        runner.fail_quantize = subprocess.CalledProcessError(2, ["q"], stderr="bad type\n" * 5)
        out = tmp_path / "gguf"
        with pytest.raises(GGUFExportError, match=r"llama-quantize failed \(exit code 2\)") as exc:
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), out, "q4_0", model_name="m")
        assert "bad type" in exc.value.message
        assert str(out / "m-f16.gguf") in (exc.value.suggestion or "")
        assert (out / "m-f16.gguf").exists()

    def test_quantize_failure_without_stderr(self, tmp_path, llama_cpp, runner):
        _add_quantizer(llama_cpp)
        runner.fail_quantize = subprocess.CalledProcessError(9, ["q"], stderr=None)
        with pytest.raises(GGUFExportError, match="No error output"):
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "o", "q4_0")

    def test_quantize_timeout(self, tmp_path, llama_cpp, runner):
        _add_quantizer(llama_cpp)
        runner.fail_quantize = subprocess.TimeoutExpired(["q"], 1800)
        with pytest.raises(GGUFExportError, match="llama-quantize timed out after 30 minutes") as exc:
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "o", "q4_0")
        assert exc.value.quantization == "q4_0"

    def test_failing_to_remove_the_f16_intermediate_only_warns(self, tmp_path, llama_cpp, runner, monkeypatch, caplog):
        _add_quantizer(llama_cpp)
        real_unlink = Path.unlink

        def stuck(self, *a, **k):
            if self.name.endswith("-f16.gguf"):
                raise OSError("locked by antivirus")
            return real_unlink(self, *a, **k)

        monkeypatch.setattr(Path, "unlink", stuck)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            r = export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "o", "q4_k_m", model_name="m")
        assert r.path.name == "m-q4_k_m.gguf" and r.path.exists()
        assert any("Failed to remove intermediate" in rec.getMessage() for rec in caplog.records)

    def test_unsupported_level_without_quantizer_refuses_before_merging(self, tmp_path, llama_cpp, runner):
        with pytest.raises(GGUFExportError, match="cannot produce 'q5_k_m'") as exc:
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "o", "q5_k_m")
        assert runner.calls == []
        assert not (tmp_path / "o" / "merged_temp").exists()
        assert "--quantization q8_0" in (exc.value.suggestion or "")
        # even with deferral requested, Ollama cannot make q5_k_m
        with pytest.raises(GGUFExportError, match="cannot produce 'q5_k_m'"):
            export.export_gguf(
                tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "o2", "q5_k_m",
                defer_quantization_to_ollama=True,
            )

    def test_q4_k_m_can_be_deferred_to_ollama(self, tmp_path, llama_cpp, runner, caplog):
        out = tmp_path / "gguf"
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            r = export.export_gguf(
                tiny_llama(layers=1), tiny_tokenizer(), out, "q4_k_m", model_name="m",
                defer_quantization_to_ollama=True,
            )
        assert len(runner.calls) == 1  # converter only; no llama-quantize
        assert r.path == out / "m-f16.gguf"
        assert r.quantization == "f16" and r.deferred_quantization == "q4_K_M"
        assert any("quantize it to q4_K_M at registration" in rec.getMessage() for rec in caplog.records)
        assert "f16" in (out / "model_card.md").read_text(encoding="utf-8")

    def test_converter_failure_reports_stderr_tail_and_cleans_up(self, tmp_path, llama_cpp, runner):
        runner.fail_converter = subprocess.CalledProcessError(
            1, ["c"], stderr="x" * 3000 + "ModuleNotFoundError: No module named 'sentencepiece'"
        )
        out = tmp_path / "gguf"
        with pytest.raises(GGUFExportError, match=r"conversion failed \(exit code 1\)") as exc:
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), out, "q8_0")
        assert "sentencepiece" in exc.value.message  # the tail, not the head
        assert len(exc.value.message) < 1700
        assert not (out / "merged_temp").exists()

    def test_converter_failure_without_stderr(self, tmp_path, llama_cpp, runner):
        runner.fail_converter = subprocess.CalledProcessError(1, ["c"], stderr="")
        with pytest.raises(GGUFExportError, match="No error output"):
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "o", "f16")

    def test_converter_timeout(self, tmp_path, llama_cpp, runner):
        runner.fail_converter = subprocess.TimeoutExpired(["c"], 1800)
        out = tmp_path / "o"
        with pytest.raises(GGUFExportError, match="timed out after 30 minutes") as exc:
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), out, "f16")
        assert not (out / "merged_temp").exists()
        assert "too large" in (exc.value.suggestion or "")

    def test_missing_output_after_a_clean_exit(self, tmp_path, llama_cpp, runner):
        runner.write_output = False
        with pytest.raises(GGUFExportError, match="GGUF file was not created"):
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "o", "q8_0", model_name="m")

    def test_merge_save_failure_cleans_scratch(self, tmp_path, llama_cpp, runner):
        class Broken:
            def save_pretrained(self, path):
                raise OSError("no space left")

        out = tmp_path / "o"
        with pytest.raises(GGUFExportError, match="Failed to prepare model for GGUF conversion: no space left"):
            export.export_gguf(Broken(), tiny_tokenizer(), out, "q8_0")
        assert not (out / "merged_temp").exists()
        assert runner.calls == []

    def test_scratch_directory_creation_failure(self, tmp_path, llama_cpp, runner, monkeypatch):
        real_mkdir = Path.mkdir

        def deny(self, *a, **k):
            if self.name == "merged_temp":
                raise OSError("read-only volume")
            return real_mkdir(self, *a, **k)

        monkeypatch.setattr(Path, "mkdir", deny)
        with pytest.raises(GGUFExportError, match="Failed to create temp directory for merge"):
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "o", "q8_0")

    def test_scratch_cleanup_failure_only_warns(self, tmp_path, llama_cpp, runner, monkeypatch, caplog):
        real_rmtree = export.shutil.rmtree

        def stuck(path, *a, **k):
            if Path(path).name == "merged_temp":
                raise OSError("handle still open")
            return real_rmtree(path, *a, **k)

        monkeypatch.setattr(export.shutil, "rmtree", stuck)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            r = export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "o", "q8_0")
        assert r.path.exists()
        assert any("Failed to clean up temp directory" in rec.getMessage() for rec in caplog.records)

    def test_output_directory_errors(self, tmp_path, llama_cpp, runner, monkeypatch):
        real_mkdir = Path.mkdir

        def deny(self, *a, **k):
            if self.name == "locked":
                raise PermissionError("denied")
            if self.name == "weird":
                raise OSError("busy")
            return real_mkdir(self, *a, **k)

        monkeypatch.setattr(Path, "mkdir", deny)
        with pytest.raises(GGUFExportError, match="Cannot create output directory"):
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "locked", "q8_0")
        with pytest.raises(GGUFExportError, match="Failed to create output directory"):
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "weird", "q8_0")

    def test_disk_guard_runs_before_any_work(self, tmp_path, llama_cpp, runner, monkeypatch):
        monkeypatch.setattr(
            export.shutil, "disk_usage", lambda p: types.SimpleNamespace(total=1, used=1, free=0)
        )
        with pytest.raises(GGUFExportError, match="Insufficient disk space") as exc:
            export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "o", "q4_k_m")
        assert exc.value.quantization == "q4_k_m"
        assert runner.calls == []


class TestConverterDiscovery:
    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch, tmp_path, runner):
        monkeypatch.delenv("BACKPROPAGATE_LLAMA_CPP_PATH", raising=False)
        monkeypatch.setattr(export, "_has_unsloth", lambda: False)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        monkeypatch.setattr(export.shutil, "which", lambda name: None)
        self.home = home

    def _convert(self, tmp_path):
        return export.export_gguf(tiny_llama(layers=1), tiny_tokenizer(), tmp_path / "out", "q8_0")

    def test_env_var_may_name_the_checkout_directory(self, tmp_path, monkeypatch, runner):
        root = tmp_path / "checkout"
        root.mkdir()
        (root / "convert_hf_to_gguf.py").write_text("#", encoding="utf-8")
        monkeypatch.setenv("BACKPROPAGATE_LLAMA_CPP_PATH", str(root))
        self._convert(tmp_path)
        assert runner.calls[0]["cmd"][1] == str(root / "convert_hf_to_gguf.py")

    def test_path_lookup(self, tmp_path, monkeypatch, runner):
        script = tmp_path / "bin" / "convert_hf_to_gguf.py"
        script.parent.mkdir()
        script.write_text("#", encoding="utf-8")
        monkeypatch.setattr(export.shutil, "which", lambda name: str(script))
        self._convert(tmp_path)
        assert runner.calls[0]["cmd"][1] == str(script)

    def test_home_directory_fallback(self, tmp_path, runner):
        script = self.home / "llama.cpp" / "convert_hf_to_gguf.py"
        script.parent.mkdir()
        script.write_text("#", encoding="utf-8")
        self._convert(tmp_path)
        assert runner.calls[0]["cmd"][1] == str(script)

    def test_stale_env_var_falls_through_to_other_sources(self, tmp_path, monkeypatch, runner):
        monkeypatch.setenv("BACKPROPAGATE_LLAMA_CPP_PATH", str(tmp_path / "does-not-exist.py"))
        script = self.home / "llama.cpp" / "convert_hf_to_gguf.py"
        script.parent.mkdir()
        script.write_text("#", encoding="utf-8")
        self._convert(tmp_path)
        assert runner.calls[0]["cmd"][1] == str(script)

    def test_nothing_found_lists_every_probed_path(self, tmp_path, monkeypatch, runner):
        monkeypatch.setenv("BACKPROPAGATE_LLAMA_CPP_PATH", str(tmp_path / "missing.py"))
        monkeypatch.setattr(export.shutil, "which", lambda name: str(tmp_path / "ghost" / "convert_hf_to_gguf.py"))
        with pytest.raises(GGUFExportError, match="requires either Unsloth or llama.cpp") as exc:
            self._convert(tmp_path)
        hint = exc.value.suggestion or ""
        assert "missing.py" in hint and "ghost" in hint
        assert str(self.home / "llama.cpp" / "convert_hf_to_gguf.py") in hint
        assert "BACKPROPAGATE_LLAMA_CPP_PATH" in hint
        assert runner.calls == []

    def test_error_names_the_default_locations_when_nothing_else_is_configured(self, tmp_path, runner):
        with pytest.raises(GGUFExportError) as exc:
            self._convert(tmp_path)
        assert str(self.home / "llama.cpp" / "convert_hf_to_gguf.py") in (exc.value.suggestion or "")


# =============================================================================
# export_gguf - Unsloth path (boundary: Unsloth's save_pretrained_gguf)
# =============================================================================


class _UnslothModel:
    """Stub of an Unsloth model: ``save_pretrained_gguf`` writes into the dir given."""

    def __init__(self, *, produce="model-q4_k_m.gguf", payload=b"GGUF" + b"\0" * 4096, error=None):
        self.produce = produce
        self.payload = payload
        self.error = error
        self.calls: list[tuple] = []

    def save_pretrained_gguf(self, directory, tokenizer, quantization_method):
        self.calls.append((directory, tokenizer, quantization_method))
        if self.error:
            raise self.error
        if self.produce:
            (Path(directory) / self.produce).write_bytes(self.payload)

    def save_pretrained(self, path):  # used only by the llama.cpp fallback's scratch merge
        (Path(path) / "config.json").write_text("{}", encoding="utf-8")

    def parameters(self):
        return []


@pytest.fixture
def unsloth_ready(monkeypatch):
    monkeypatch.setattr(export, "_has_unsloth", lambda: True)
    monkeypatch.setattr(export, "_unsloth_gguf_ready", lambda: (True, ""))


class TestExportGgufUnsloth:
    def test_unsloth_output_is_promoted_atomically(self, tmp_path, unsloth_ready, capsys):
        out = tmp_path / "gguf"
        model = _UnslothModel()
        tok = tiny_tokenizer()
        result = export.export_gguf(model, tok, out, "q4_k_m", base_model="org/base", run_id="r1")
        assert result.path == out / "model-q4_k_m.gguf"
        assert result.quantization == "q4_k_m" and result.size_mb > 0
        (directory, passed_tok, quant), = model.calls
        assert directory == str(out / "_unsloth_partial")
        assert passed_tok is tok and quant == "q4_k_m"
        assert not (out / "_unsloth_partial").exists()
        assert "Exporting to GGUF format" in capsys.readouterr().out
        card = (out / "model_card.md").read_text(encoding="utf-8")
        assert "org/base" in card and "q4_k_m" in card

    def test_existing_target_and_stale_partial_are_replaced(self, tmp_path, unsloth_ready):
        out = tmp_path / "gguf"
        (out / "_unsloth_partial").mkdir(parents=True)
        (out / "_unsloth_partial" / "stale.gguf").write_bytes(b"old")
        (out / "model-q4_k_m.gguf").write_bytes(b"previous export")
        result = export.export_gguf(_UnslothModel(), tiny_tokenizer(), out, "q4_k_m", emit_model_card=False)
        assert result.path.read_bytes().startswith(b"GGUF")
        assert not (out / "_unsloth_partial").exists()

    def test_no_gguf_produced_is_a_structured_failure_not_a_fallback(self, tmp_path, unsloth_ready, runner):
        out = tmp_path / "gguf"
        with pytest.raises(GGUFExportError, match="was not created in partial directory"):
            export.export_gguf(_UnslothModel(produce=None), tiny_tokenizer(), out, "q4_k_m")
        assert runner.calls == []  # no silent fall-through to llama.cpp
        assert not (out / "_unsloth_partial").exists()

    def test_empty_gguf_is_rejected(self, tmp_path, unsloth_ready):
        with pytest.raises(GGUFExportError, match="GGUF file is empty"):
            export.export_gguf(_UnslothModel(payload=b""), tiny_tokenizer(), tmp_path / "o", "q4_k_m")

    def test_unexpected_unsloth_error_falls_back_to_llama_cpp(self, tmp_path, monkeypatch, llama_cpp, runner, caplog, capsys):
        # the llama_cpp fixture disables Unsloth; this test wants it present but failing
        monkeypatch.setattr(export, "_has_unsloth", lambda: True)
        monkeypatch.setattr(export, "_unsloth_gguf_ready", lambda: (True, ""))
        out = tmp_path / "gguf"
        model = _UnslothModel(error=RuntimeError("CUDA required"))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = export.export_gguf(model, tiny_tokenizer(), out, "q8_0", model_name="m", emit_model_card=False)
        assert len(model.calls) == 1
        assert result.path == out / "m-q8_0.gguf"
        assert len(runner.calls) == 1
        assert any("Unsloth GGUF export failed: CUDA required" in r.getMessage() for r in caplog.records)
        assert "falling back to llama.cpp" in capsys.readouterr().out
        assert not (out / "_unsloth_partial").exists()

    def test_unready_llama_cpp_build_skips_unsloth_with_explanation(self, tmp_path, monkeypatch, llama_cpp, runner, caplog, capsys):
        monkeypatch.setattr(export, "_has_unsloth", lambda: True)
        monkeypatch.setattr(export, "_unsloth_gguf_ready", lambda: (False, "no llama.cpp build"))
        model = _UnslothModel()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = export.export_gguf(model, tiny_tokenizer(), tmp_path / "o", "q8_0", model_name="m")
        assert model.calls == []  # Unsloth never invoked
        assert result.path.name == "m-q8_0.gguf"
        assert any("no llama.cpp build" in r.getMessage() for r in caplog.records)
        assert "WARNING: no llama.cpp build" in capsys.readouterr().out

    def test_failure_cleanup_tolerates_a_scratch_dir_that_vanished(self, tmp_path, unsloth_ready):
        """Unsloth removing its own scratch directory must not mask the real error."""
        import shutil

        class Vanishing(_UnslothModel):
            def save_pretrained_gguf(self, directory, tokenizer, quantization_method):
                shutil.rmtree(directory)

        with pytest.raises(GGUFExportError, match="was not created in partial directory"):
            export.export_gguf(Vanishing(), tiny_tokenizer(), tmp_path / "o", "q4_k_m")

    def test_fallback_cleanup_tolerates_a_scratch_dir_that_vanished(self, tmp_path, monkeypatch, llama_cpp, runner):
        import shutil

        monkeypatch.setattr(export, "_has_unsloth", lambda: True)
        monkeypatch.setattr(export, "_unsloth_gguf_ready", lambda: (True, ""))

        class VanishThenFail(_UnslothModel):
            def save_pretrained_gguf(self, directory, tokenizer, quantization_method):
                shutil.rmtree(directory)
                raise RuntimeError("unsloth crashed")

        result = export.export_gguf(
            VanishThenFail(), tiny_tokenizer(), tmp_path / "o", "q8_0", model_name="m", emit_model_card=False
        )
        assert result.path.name == "m-q8_0.gguf"
        assert len(runner.calls) == 1

    def test_partial_directory_creation_failure(self, tmp_path, unsloth_ready, monkeypatch):
        real_mkdir = Path.mkdir

        def deny(self, *a, **k):
            if self.name == "_unsloth_partial":
                raise OSError("read-only")
            return real_mkdir(self, *a, **k)

        monkeypatch.setattr(Path, "mkdir", deny)
        with pytest.raises(GGUFExportError, match="Failed to create partial directory") as exc:
            export.export_gguf(_UnslothModel(), tiny_tokenizer(), tmp_path / "o", "q4_k_m")
        assert exc.value.quantization == "q4_k_m"


# =============================================================================
# load_model_for_export (real transformers + peft on CPU)
# =============================================================================


@pytest.fixture(scope="module")
def saved_base(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("export_base")
    tiny_llama(layers=2).save_pretrained(d)
    tiny_tokenizer().save_pretrained(d)
    return d


def _adapter_dir(tmp_path: Path, base: Path, *, tokenizer: bool = False) -> Path:
    model = get_peft_model(
        tiny_llama(layers=2),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]),
    )
    d = tmp_path / "adapter"
    model.save_pretrained(d)
    cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = str(base)
    (d / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if tokenizer:
        tiny_tokenizer().save_pretrained(d)
    return d


class TestLoadModelForExport:
    def test_plain_model_directory_loads_without_an_adapter(self, saved_base):
        model, tok = export.load_model_for_export(saved_base)
        assert not export._is_peft_model(model)
        assert type(model).__name__ == "LlamaForCausalLM"
        assert next(model.parameters()).dtype == torch.float32  # CPU -> fp32
        assert tok.pad_token == "<pad>"

    def test_adapter_directory_loads_base_plus_adapter_via_transformers(self, tmp_path, saved_base, monkeypatch):
        monkeypatch.setattr(export, "_has_unsloth", lambda: False)
        adapter = _adapter_dir(tmp_path, saved_base, tokenizer=True)
        model, tok = export.load_model_for_export(adapter)
        assert export._is_peft_model(model)
        assert "default" in model.peft_config
        assert tok.pad_token == "<pad>"

    def test_tokenizer_falls_back_to_the_base_when_adapter_has_none(self, tmp_path, saved_base, monkeypatch):
        monkeypatch.setattr(export, "_has_unsloth", lambda: False)
        adapter = _adapter_dir(tmp_path, saved_base, tokenizer=False)
        _, tok = export.load_model_for_export(adapter)
        assert tok.eos_token == "</s>"

    def test_missing_pad_token_is_set_to_eos(self, tmp_path, saved_base, monkeypatch):
        monkeypatch.setattr(export, "_has_unsloth", lambda: False)
        adapter = _adapter_dir(tmp_path, saved_base, tokenizer=False)
        base_copy = tmp_path / "base_nopad"
        base_copy.mkdir()
        tiny_llama(layers=2).save_pretrained(base_copy)
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        tk = Tokenizer(WordLevel({"<s>": 0, "</s>": 1, "<unk>": 2, "cat": 3}, unk_token="<unk>"))
        tk.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        ).save_pretrained(base_copy)
        cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        cfg["base_model_name_or_path"] = str(base_copy)
        (adapter / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        _, tok = export.load_model_for_export(adapter)
        assert tok.pad_token == tok.eos_token == "</s>"

    def test_unreadable_adapter_config(self, tmp_path):
        d = tmp_path / "adapter"
        d.mkdir()
        (d / "adapter_config.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(MergeExportError, match="Cannot read adapter config") as exc:
            export.load_model_for_export(d)
        assert "adapter_model.safetensors" in (exc.value.suggestion or "")

    def test_adapter_config_without_a_base_model(self, tmp_path):
        d = tmp_path / "adapter"
        d.mkdir()
        (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": ""}), encoding="utf-8")
        with pytest.raises(MergeExportError, match="names no base model"):
            export.load_model_for_export(d)

    def _fake_unsloth(self, monkeypatch, factory):
        monkeypatch.setattr(export, "_has_unsloth", lambda: True)
        mod = types.ModuleType("unsloth")
        calls = []

        class FastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                calls.append(kwargs)
                return factory()

        mod.FastLanguageModel = FastLanguageModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "unsloth", mod)
        return calls

    def test_unsloth_peft_result_is_used_directly(self, tmp_path, saved_base, monkeypatch):
        """Mocks: ``unsloth.FastLanguageModel`` (GPU-only)."""
        adapter = _adapter_dir(tmp_path, saved_base)
        peft_model = make_peft()
        tok = tiny_tokenizer()
        calls = self._fake_unsloth(monkeypatch, lambda: (peft_model, tok))
        model, tokenizer = export.load_model_for_export(adapter)
        assert model is peft_model and tokenizer is tok
        assert calls == [{
            "model_name": str(adapter), "load_in_4bit": False, "dtype": None, "trust_remote_code": False,
        }]

    def test_unsloth_non_peft_result_falls_back_to_transformers(self, tmp_path, saved_base, monkeypatch, caplog):
        adapter = _adapter_dir(tmp_path, saved_base, tokenizer=True)
        self._fake_unsloth(monkeypatch, lambda: (tiny_llama(layers=1), tiny_tokenizer()))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            model, _ = export.load_model_for_export(adapter)
        assert export._is_peft_model(model)
        assert any("not a PEFT model" in r.getMessage() for r in caplog.records)

    def test_unsloth_failure_falls_back_to_transformers(self, tmp_path, saved_base, monkeypatch, caplog):
        adapter = _adapter_dir(tmp_path, saved_base, tokenizer=True)

        def explode():
            raise RuntimeError("no accelerator")

        self._fake_unsloth(monkeypatch, explode)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            model, _ = export.load_model_for_export(adapter)
        assert export._is_peft_model(model)
        assert any("Unsloth could not load" in r.getMessage() and "no accelerator" in r.getMessage()
                   for r in caplog.records)


def test_get_dir_size_of_file_and_directory(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 1024 * 1024)
    assert export._get_dir_size_mb(f) == pytest.approx(1.0)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"x" * 1024 * 1024)
    assert export._get_dir_size_mb(tmp_path) == pytest.approx(2.0)


def test_export_result_summary_lines():
    r = export.ExportResult(
        format=export.ExportFormat.GGUF, path=Path("/x/m.gguf"), size_mb=12.34,
        quantization="q4_k_m", export_time_seconds=3.21,
    )
    text = r.summary()
    assert "Format: gguf" in text and "Size: 12.3 MB" in text
    assert "Quantization: q4_k_m" in text and "Time: 3.2s" in text
    bare = export.ExportResult(export.ExportFormat.LORA, Path("/x"), 1.0).summary()
    assert "Quantization" not in bare and "Time:" not in bare

