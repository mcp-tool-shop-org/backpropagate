"""Coverage tests for ``cmd_generate``, ``cmd_estimate_vram`` and the
``ollama`` register / list / rm / shelf handlers in cli.py.

Real: ``generate`` runs end to end on CPU against a tiny random-weight Llama
(``tests/helpers/tiny_models.py``) saved to ``tmp_path`` together with a real
PEFT LoRA adapter; ``estimate-vram`` uses the real estimator.
Mocked (real boundaries): CUDA queries in the VRAM probe, and the ``ollama``
daemon / CLI helpers in ``backpropagate.export`` (subprocess to an external
tool) plus ``shutil.which("ollama")``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backpropagate import cli
from backpropagate.exceptions import BackpropagateError
from tests.helpers.cli_cov_support import last_json, parse

# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_adapter(tmp_path_factory):
    """A saved tiny Llama base (+tokenizer) and a real LoRA adapter whose
    adapter_config.json records that base."""
    peft = pytest.importorskip("peft")
    from tests.helpers.tiny_models import tiny_llama, tiny_tokenizer

    root = tmp_path_factory.mktemp("tiny_gen")
    base = root / "base"
    tiny_llama(layers=2).save_pretrained(base)
    tiny_tokenizer().save_pretrained(base)
    cfg = peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    adapter = root / "adapter"
    peft.get_peft_model(tiny_llama(layers=2), cfg).save_pretrained(adapter)
    cfg_path = adapter / "adapter_config.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    data["base_model_name_or_path"] = str(base)
    cfg_path.write_text(json.dumps(data), encoding="utf-8")
    return SimpleNamespace(base=base, adapter=adapter, root=root)


class TestGeneratePreflight:
    def test_adapter_missing(self, tmp_path, capsys):
        assert cli.cmd_generate(parse(["generate", str(tmp_path / "no"), "hi"])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "Adapter path not found" in captured.err and "adapter_config.json" in captured.out

    def test_adapter_is_a_file(self, tmp_path, capsys):
        f = tmp_path / "adapter.bin"
        f.write_bytes(b"x")
        assert cli.cmd_generate(parse(["generate", str(f), "hi"])) == cli.EXIT_USER_ERROR
        assert "is a file, expected a directory" in capsys.readouterr().err

    def test_base_cannot_be_inferred(self, tmp_path, capsys):
        d = tmp_path / "adapter"
        d.mkdir()
        assert cli.cmd_generate(parse(["generate", str(d), "hi"])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "Could not infer the base model" in captured.err and "--base" in captured.out

    def test_log_failures_are_ignored_on_error_paths(self, tmp_path, monkeypatch, capsys):
        def boom(*a, **k):
            raise RuntimeError("log down")

        monkeypatch.setattr("backpropagate.logging_config.get_logger", boom)
        monkeypatch.setattr("backpropagate.logging_config.bind_run_context", boom)
        f = tmp_path / "f.bin"
        f.write_bytes(b"x")
        d = tmp_path / "d"
        d.mkdir()
        for path in (tmp_path / "missing", f, d):
            assert cli.cmd_generate(parse(["generate", str(path), "hi"])) == cli.EXIT_USER_ERROR
        capsys.readouterr()


class TestInferBase:
    def test_reads_base_from_adapter_config(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text('{"base_model_name_or_path": " org/base "}', encoding="utf-8")
        assert cli._infer_base_model_from_adapter(tmp_path) == "org/base"

    @pytest.mark.parametrize(
        "content",
        ["not json", "[1, 2]", '{"base_model_name_or_path": ""}', '{"base_model_name_or_path": 5}', "{}"],
    )
    def test_unusable_configs_return_none(self, tmp_path, content):
        (tmp_path / "adapter_config.json").write_text(content, encoding="utf-8")
        assert cli._infer_base_model_from_adapter(tmp_path) is None

    def test_missing_file_and_unreadable(self, tmp_path, monkeypatch):
        assert cli._infer_base_model_from_adapter(tmp_path) is None
        (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
        import builtins

        def deny(*a, **k):
            raise PermissionError("denied")

        monkeypatch.setattr(builtins, "open", deny)
        assert cli._infer_base_model_from_adapter(tmp_path) is None


class TestGenerateEndToEnd:
    def test_inferred_base_single_sample(self, tiny_adapter, capsys):
        args = parse(["generate", str(tiny_adapter.adapter), "the cat sat",
                      "--max-new-tokens", "4", "--temperature", "0"])
        assert cli.cmd_generate(args) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert f"Base model: {tiny_adapter.base} (inferred)" in out
        assert "Prompt: the cat sat" in out and "Samples: 1" in out
        assert "Generation complete" in out and "Generation 1/" not in out

    def test_explicit_base_multiple_samples(self, tiny_adapter, capsys):
        args = parse(["generate", str(tiny_adapter.adapter), "the dog ran", "--base", str(tiny_adapter.base),
                      "--num", "2", "--max-new-tokens", "3", "--temperature", "0"])
        assert cli.cmd_generate(args) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "(inferred)" not in out
        assert "Generation 1/2" in out and "Generation 2/2" in out

    def test_bad_base_surfaces_structured_runtime_error(self, tiny_adapter, tmp_path, capsys):
        args = parse(["generate", str(tiny_adapter.adapter), "hi", "--base", str(tmp_path / "no-such-model")])
        assert cli.cmd_generate(args) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "[RUNTIME_EVAL_FAILED]" in captured.err and "Generation error:" in captured.err

    def test_bad_base_verbose(self, tiny_adapter, tmp_path, capsys):
        args = parse(["generate", str(tiny_adapter.adapter), "hi", "--base", str(tmp_path / "no-such-model")])
        args.verbose = True
        assert cli.cmd_generate(args) == cli.EXIT_RUNTIME_ERROR


class TestGenerateErrorMapping:
    """Error mapping around the (mocked) generation boundary."""

    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        d = tmp_path / "adapter"
        d.mkdir()

        def _run(exc, verbose=False):
            def fake_load(synthetic):
                raise exc

            monkeypatch.setattr("backpropagate.eval._load_model_and_tokenizer", fake_load)
            args = parse(["generate", str(d), "hi", "--base", "some/base"])
            args.verbose = verbose
            return cli.cmd_generate(args)

        return _run

    def test_keyboard_interrupt(self, run, capsys):
        assert run(KeyboardInterrupt()) == cli.EXIT_INTERRUPTED
        assert "Generation interrupted by user" in capsys.readouterr().out

    @pytest.mark.parametrize("verbose", [False, True])
    def test_user_input_error(self, run, capsys, verbose):
        from backpropagate.exceptions import UserInputError

        assert run(UserInputError("bad prompt", hint="shorten it"), verbose) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "bad prompt" in captured.err and "Suggestion: shorten it" in captured.out

    @pytest.mark.parametrize("verbose", [False, True])
    def test_structured_error(self, run, capsys, verbose):
        assert run(BackpropagateError("load failed", suggestion="check path"), verbose) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "Generation error: load failed" in captured.err and "Suggestion: check path" in captured.out

    def test_unexpected_error_redacted_then_verbose(self, run, capsys):
        exc = RuntimeError("Authorization: Bearer abcdef1234567890")
        assert run(exc) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "abcdef1234567890" not in captured.err and "Run with --verbose" in captured.out
        assert run(exc, verbose=True) == cli.EXIT_RUNTIME_ERROR
        assert "Traceback" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# estimate-vram
# ---------------------------------------------------------------------------


class TestEstimateVram:
    def test_explicit_vram_table(self, capsys):
        assert cli.cmd_estimate_vram(parse(["estimate-vram", "Qwen/Qwen2.5-7B-Instruct", "--vram-gb", "24"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "VRAM: 24.0 GB (user (--vram-gb))" in out
        assert "Recommended --batch-size: 4" in out
        assert "Tier table" in out and out.count("<- this card") == 1
        assert "Per-config estimate" not in out

    @pytest.mark.parametrize(
        "vram, batch",
        [(80, 8), (31.8, 6), (23.6, 4), (15.5, 2), (11.0, 1), (0.5, 1)],
    )
    def test_tier_selection_with_tolerance(self, capsys, vram, batch):
        argv = ["estimate-vram", "m", "--vram-gb", str(vram), "--json"]
        assert cli.cmd_estimate_vram(parse(argv)) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        assert payload["recommended_batch_size"] == batch
        assert payload["schema_version"] == cli.CLI_JSON_SCHEMA_VERSION
        assert payload["per_config_estimate"] is None
        assert len(payload["tiers"]) == len(cli._VRAM_BATCH_SIZE_TIERS)

    @pytest.mark.parametrize("bad", [0.0, -4.0, 600.0])
    def test_implausible_override(self, capsys, bad):
        args = parse(["estimate-vram", "m", "--vram-gb", "16"])
        args.vram_gb = bad  # direct call: the parser itself only rejects <= 0
        assert cli.cmd_estimate_vram(args) == cli.EXIT_USER_ERROR
        assert "outside the plausible range" in capsys.readouterr().err

    @pytest.mark.parametrize("bad", ["0", "-4"])
    def test_parser_rejects_non_positive_vram(self, capsys, bad):
        with pytest.raises(SystemExit) as ei:
            parse(["estimate-vram", "m", "--vram-gb", bad])
        assert ei.value.code == 2
        assert "must be positive" in capsys.readouterr().err

    def test_parser_accepts_oversized_vram_and_handler_rejects(self, capsys):
        assert cli.cmd_estimate_vram(parse(["estimate-vram", "m", "--vram-gb", "600"])) == cli.EXIT_USER_ERROR
        assert "outside the plausible range" in capsys.readouterr().err

    def test_per_config_estimate_lora(self, capsys):
        argv = ["estimate-vram", "Qwen/Qwen2.5-7B-Instruct", "--vram-gb", "48", "--batch-size", "2", "--lora-r", "32"]
        assert cli.cmd_estimate_vram(parse(argv)) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Per-config estimate (lora)" in out
        assert "LoRA adapter" in out and "Total VRAM (est.)" in out and "Optimizer state" in out

    def test_per_config_estimate_json_matches_real_estimator(self, capsys):
        from backpropagate.trainer import estimate_vram

        argv = ["estimate-vram", "Qwen/Qwen2.5-7B-Instruct", "--vram-gb", "48", "--batch-size", "2", "--json"]
        assert cli.cmd_estimate_vram(parse(argv)) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        expected = estimate_vram(model="Qwen/Qwen2.5-7B-Instruct", mode="lora", lora_r=16, batch_size=2,
                                 offload=False)
        assert payload["per_config_estimate"]["total_gb"] == pytest.approx(expected.total_gb)

    def test_full_mode_hides_lora_row_and_warns_on_oom(self, capsys):
        argv = ["estimate-vram", "Qwen/Qwen2.5-7B-Instruct", "--vram-gb", "8", "--mode", "full"]
        assert cli.cmd_estimate_vram(parse(argv)) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Per-config estimate (full)" in out
        assert "LoRA adapter" not in out
        assert "likely OOM" in out

    def test_offload_flag_adds_host_ram(self, capsys):
        argv = ["estimate-vram", "Qwen/Qwen2.5-7B-Instruct", "--vram-gb", "32", "--mode", "full",
                "--full-ft-offload", "--json"]
        assert cli.cmd_estimate_vram(parse(argv)) == cli.EXIT_OK
        est = last_json(capsys.readouterr().out)["per_config_estimate"]
        assert est["host_ram_gb"] and est["host_ram_gb"] > 0

    def test_dict_shaped_estimate_is_accepted(self, monkeypatch, capsys):
        monkeypatch.setattr("backpropagate.trainer.estimate_vram",
                            lambda **kw: {"total_gb": 1.5, "model_weights_gb": "n/a"})
        argv = ["estimate-vram", "m", "--vram-gb", "16", "--batch-size", "1"]
        assert cli.cmd_estimate_vram(parse(argv)) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Total VRAM (est.): 1.50 GB" in out and "Model weights: -" in out
        assert "likely OOM" not in out

    def test_estimator_failure_degrades_gracefully(self, monkeypatch, capsys):
        def boom(**kw):
            raise RuntimeError("estimator down")

        monkeypatch.setattr("backpropagate.trainer.estimate_vram", boom)
        argv = ["estimate-vram", "m", "--vram-gb", "16", "--batch-size", "1", "--json"]
        assert cli.cmd_estimate_vram(parse(argv)) == cli.EXIT_OK
        assert last_json(capsys.readouterr().out)["per_config_estimate"] is None


class TestEstimateVramProbe:
    """The no-override path asks torch for the device. CUDA is faked (hardware boundary)."""

    def test_detected_via_torch(self, monkeypatch, capsys):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_properties",
                            lambda i: SimpleNamespace(total_memory=24 * 1024 ** 3))
        assert cli.cmd_estimate_vram(parse(["estimate-vram", "m", "--json"])) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        assert payload["vram_gb"] == 24.0 and payload["detected_via"] == "torch.cuda"

    def test_no_cuda_requires_override(self, monkeypatch, capsys):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert cli.cmd_estimate_vram(parse(["estimate-vram", "m"])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "Could not detect VRAM" in captured.err and "--vram-gb" in captured.out

    def test_probe_error_is_swallowed(self, monkeypatch, capsys):
        import torch

        def boom():
            raise RuntimeError("driver mismatch")

        monkeypatch.setattr(torch.cuda, "is_available", boom)
        assert cli.cmd_estimate_vram(parse(["estimate-vram", "m"])) == cli.EXIT_USER_ERROR
        assert "Could not detect VRAM" in capsys.readouterr().err

    def test_torch_not_installed(self, monkeypatch, capsys):
        import sys

        monkeypatch.setitem(sys.modules, "torch", None)
        assert cli.cmd_estimate_vram(parse(["estimate-vram", "m"])) == cli.EXIT_USER_ERROR
        assert "Could not detect VRAM" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# ollama
# ---------------------------------------------------------------------------


class TestOllamaRegister:
    def test_path_missing(self, tmp_path, capsys):
        assert cli.cmd_ollama_register(parse(["ollama", "register", str(tmp_path / "no.gguf")])) == cli.EXIT_USER_ERROR
        assert "GGUF path does not exist" in capsys.readouterr().err

    def test_directory_without_gguf(self, tmp_path, capsys):
        assert cli.cmd_ollama_register(parse(["ollama", "register", str(tmp_path)])) == cli.EXIT_USER_ERROR
        assert "No *.gguf file found" in capsys.readouterr().err

    def test_directory_with_multiple_ggufs_uses_first(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "b.gguf").write_bytes(b"x")
        (tmp_path / "a.gguf").write_bytes(b"x")
        seen = {}
        monkeypatch.setattr("backpropagate.export.register_with_ollama",
                            lambda path, name: seen.update(path=path, name=name) or True)
        assert cli.cmd_ollama_register(parse(["ollama", "register", str(tmp_path)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Multiple GGUF files" in out and "using a.gguf" in out
        assert seen["path"].name == "a.gguf" and seen["name"] == "a"
        assert "Registered with Ollama: a" in out and "ollama run a" in out

    def test_directory_with_single_gguf_has_no_warning(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "only.gguf").write_bytes(b"x")
        monkeypatch.setattr("backpropagate.export.register_with_ollama", lambda path, name: True)
        assert cli.cmd_ollama_register(parse(["ollama", "register", str(tmp_path)])) == cli.EXIT_OK
        assert "Multiple GGUF files" not in capsys.readouterr().out

    def test_explicit_name(self, tmp_path, monkeypatch):
        g = tmp_path / "model.gguf"
        g.write_bytes(b"x")
        seen = {}
        monkeypatch.setattr("backpropagate.export.register_with_ollama",
                            lambda path, name: seen.update(name=name) or True)
        assert cli.cmd_ollama_register(parse(["ollama", "register", str(g), "--name", "mine"])) == cli.EXIT_OK
        assert seen["name"] == "mine"

    def test_cli_missing_is_unavailable(self, tmp_path, monkeypatch, capsys):
        g = tmp_path / "model.gguf"
        g.write_bytes(b"x")
        monkeypatch.setattr("backpropagate.export.register_with_ollama", lambda path, name: False)
        assert cli.cmd_ollama_register(parse(["ollama", "register", str(g)])) == cli.EXIT_UNAVAILABLE
        captured = capsys.readouterr()
        assert "Ollama CLI not found on PATH." in captured.err and "ollama serve" in captured.out

    @pytest.mark.parametrize(
        "code, expected",
        [
            ("DEP_OLLAMA_REGISTRATION_FAILED", cli.EXIT_UNAVAILABLE),
            ("INPUT_VALIDATION_FAILED", cli.EXIT_USER_ERROR),
            ("RUNTIME_ANYTHING", cli.EXIT_RUNTIME_ERROR),
        ],
    )
    def test_structured_error_codes(self, tmp_path, monkeypatch, capsys, code, expected):
        g = tmp_path / "model.gguf"
        g.write_bytes(b"x")

        def fail(path, name):
            raise BackpropagateError("daemon said no", suggestion="start it", code=code)

        monkeypatch.setattr("backpropagate.export.register_with_ollama", fail)
        assert cli.cmd_ollama_register(parse(["ollama", "register", str(g)])) == expected
        captured = capsys.readouterr()
        assert "Registration failed: daemon said no" in captured.err and "Suggestion: start it" in captured.out

    def test_unexpected_error(self, tmp_path, monkeypatch, capsys):
        g = tmp_path / "model.gguf"
        g.write_bytes(b"x")

        def fail(path, name):
            raise RuntimeError("Authorization: Bearer abcdef1234567890")

        monkeypatch.setattr("backpropagate.export.register_with_ollama", fail)
        assert cli.cmd_ollama_register(parse(["ollama", "register", str(g)])) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "abcdef1234567890" not in captured.err and "Run with --verbose" in captured.out
        args = parse(["ollama", "register", str(g)])
        args.verbose = True
        assert cli.cmd_ollama_register(args) == cli.EXIT_RUNTIME_ERROR
        assert "Traceback" in capsys.readouterr().err


class TestOllamaList:
    def test_lists_models(self, monkeypatch, capsys):
        monkeypatch.setattr("backpropagate.export.list_ollama_models", lambda: ["a:latest", "b:q4"])
        assert cli.cmd_ollama_list(parse(["ollama", "list"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "MODEL" in out and "a:latest" in out and "b:q4" in out and "Listed 2 model(s)." in out

    def test_empty_with_cli_present(self, monkeypatch, capsys):
        monkeypatch.setattr("backpropagate.export.list_ollama_models", lambda: [])
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ollama")
        assert cli.cmd_ollama_list(parse(["ollama", "list"])) == cli.EXIT_OK
        assert "No models registered with Ollama." in capsys.readouterr().out

    def test_empty_with_cli_absent(self, monkeypatch, capsys):
        monkeypatch.setattr("backpropagate.export.list_ollama_models", lambda: [])
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert cli.cmd_ollama_list(parse(["ollama", "list"])) == cli.EXIT_UNAVAILABLE
        assert "Ollama CLI not found on PATH." in capsys.readouterr().out


class TestOllamaRm:
    def test_removed(self, monkeypatch, capsys):
        monkeypatch.setattr("backpropagate.export.remove_ollama_model", lambda name: True)
        assert cli.cmd_ollama_rm(parse(["ollama", "rm", "old"])) == cli.EXIT_OK
        assert "Removed from Ollama: old" in capsys.readouterr().out

    def test_cli_missing(self, monkeypatch, capsys):
        monkeypatch.setattr("backpropagate.export.remove_ollama_model", lambda name: False)
        assert cli.cmd_ollama_rm(parse(["ollama", "rm", "old"])) == cli.EXIT_UNAVAILABLE
        assert "Ollama CLI not found on PATH." in capsys.readouterr().err

    @pytest.mark.parametrize(
        "code, expected",
        [
            ("DEP_OLLAMA_REGISTRATION_FAILED", cli.EXIT_UNAVAILABLE),
            ("INPUT_VALIDATION_FAILED", cli.EXIT_USER_ERROR),
            ("RUNTIME_ANYTHING", cli.EXIT_RUNTIME_ERROR),
        ],
    )
    def test_structured_error_codes(self, monkeypatch, capsys, code, expected):
        def fail(name):
            raise BackpropagateError("no such model", suggestion="check the name", code=code)

        monkeypatch.setattr("backpropagate.export.remove_ollama_model", fail)
        assert cli.cmd_ollama_rm(parse(["ollama", "rm", "old"])) == expected
        captured = capsys.readouterr()
        assert "Removal failed: no such model" in captured.err and "Suggestion: check the name" in captured.out

    def test_unexpected_error(self, monkeypatch, capsys):
        def fail(name):
            raise RuntimeError("Authorization: Bearer abcdef1234567890")

        monkeypatch.setattr("backpropagate.export.remove_ollama_model", fail)
        assert cli.cmd_ollama_rm(parse(["ollama", "rm", "old"])) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "abcdef1234567890" not in captured.err and "Run with --verbose" in captured.out
        args = parse(["ollama", "rm", "old"])
        args.verbose = True
        assert cli.cmd_ollama_rm(args) == cli.EXIT_RUNTIME_ERROR
        assert "Traceback" in capsys.readouterr().err


class TestOllamaShelf:
    def test_lists_adapters(self, monkeypatch, capsys):
        entries = [SimpleNamespace(model_name="llama3.2:taskA", tag="taskA", size="1.2 GB", modified="today")]
        monkeypatch.setattr("backpropagate.export.list_adapter_shelf", lambda base: entries)
        assert cli.cmd_ollama_shelf(parse(["ollama", "shelf", "llama3.2"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "llama3.2:taskA\ttaskA\t1.2 GB\ttoday" in out and "Listed 1 adapter(s) on base 'llama3.2'." in out

    def test_empty_with_cli_present(self, monkeypatch, capsys):
        monkeypatch.setattr("backpropagate.export.list_adapter_shelf", lambda base: [])
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ollama")
        assert cli.cmd_ollama_shelf(parse(["ollama", "shelf", "llama3.2"])) == cli.EXIT_OK
        assert "No adapters on the shelf for base 'llama3.2'." in capsys.readouterr().out

    def test_empty_with_cli_absent(self, monkeypatch, capsys):
        monkeypatch.setattr("backpropagate.export.list_adapter_shelf", lambda base: [])
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert cli.cmd_ollama_shelf(parse(["ollama", "shelf", "llama3.2"])) == cli.EXIT_UNAVAILABLE
        assert "Ollama CLI not found on PATH." in capsys.readouterr().out
