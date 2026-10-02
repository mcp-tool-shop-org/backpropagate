# ui-v2 P3: CLI parity on the training forms + the inline VRAM estimate.
"""Every Single run / Multi-run form field reaches a real CLI flag.

Pins: the form -> JobSpec mapping (mode, method knobs, LoRA shape, advanced
flags), the argv the JobManager builds from it, the server-side refusals
for the new knobs, the "fits / tight / won't fit" verdict (which must be
exactly ``backprop estimate-vram``'s numbers), and the estimator fixes the
inline estimate exposed (sub-billion ids, full mode is 16-bit, gradient
checkpointing, the model's own shape).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backpropagate import ui_jobs
from backpropagate.trainer import _estimate_param_count_billions, estimate_vram
from backpropagate.ui_jobs import JobSpec, JobValidationError, vram_verdict


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    out = tmp_path / "ui-outputs"
    out.mkdir()
    monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(out))
    data = out / "d.jsonl"
    data.write_text('{"text": "hi"}\n', encoding="utf-8")
    return out, data


def _argv(spec: JobSpec, tmp_path: Path) -> list[str]:
    return ui_jobs._build_argv(spec, tmp_path / "run")


def _flag(argv, name):
    return argv[argv.index(name) + 1]


# ---- argv ------------------------------------------------------------------------


def test_train_argv_carries_every_p3_knob(tmp_path):
    spec = JobSpec(
        kind="sft", model="org/m", dataset_path="d.jsonl", lora_r=16,
        lora_alpha=32, lora_dropout=0.1, target_modules="q_proj,v_proj",
        base_4bit=False, method="simpo",
        method_params={"simpo_beta": 2.5, "simpo_gamma": 0.5},
        run_name="exp-1", gpu_max_temp=88, gradient_checkpointing=False,
    )
    argv = _argv(spec, tmp_path)
    assert _flag(argv, "--lora-alpha") == "32"
    assert _flag(argv, "--lora-dropout") == "0.1"
    assert _flag(argv, "--target-modules") == "q_proj,v_proj"
    assert "--no-4bit" in argv
    assert _flag(argv, "--method") == "simpo"
    assert _flag(argv, "--simpo-beta") == "2.5"
    assert _flag(argv, "--simpo-gamma") == "0.5"
    assert _flag(argv, "--run-name") == "exp-1"
    assert _flag(argv, "--gpu-max-temp") == "88"
    assert "--no-gradient-checkpointing" in argv


def test_default_spec_sends_only_cli_defaults(tmp_path):
    argv = _argv(JobSpec(kind="sft", model="org/m", dataset_path="d.jsonl"), tmp_path)
    for flag in ("--method", "--no-4bit", "--run-name", "--gpu-max-temp",
                 "--no-gradient-checkpointing", "--lora-alpha", "--target-modules"):
        assert flag not in argv


def test_full_mode_drops_lora_only_flags(tmp_path):
    spec = JobSpec(
        kind="sft", model="org/m", dataset_path="d.jsonl", mode="full",
        lora_alpha=32, target_modules="q_proj", base_4bit=False,
        gradient_checkpointing=False,
    )
    argv = _argv(spec, tmp_path)
    assert _flag(argv, "--mode") == "full"
    for flag in ("--lora-alpha", "--target-modules", "--no-4bit", "--no-gradient-checkpointing"):
        assert flag not in argv


def test_multi_run_argv_carries_lr_rank_batch_and_knobs(tmp_path):
    spec = JobSpec(
        kind="multi_run", model="org/m", dataset_path="d.jsonl", lr=1e-4,
        lora_r=64, batch="4", method="kto",
        method_params={"kto_beta": 0.2}, run_name="sweep",
    )
    argv = _argv(spec, tmp_path)
    assert argv[3] == "multi-run"
    assert _flag(argv, "--lr") == "0.0001"
    assert _flag(argv, "--lora-r") == "64"
    assert _flag(argv, "--batch-size") == "4"
    assert _flag(argv, "--method") == "kto"
    assert _flag(argv, "--kto-beta") == "0.2"
    assert _flag(argv, "--run-name") == "sweep"


# ---- server-side validation ---------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"method": "dpo"}, "Unknown method"),
        ({"mode": "full", "method": "orpo"}, "SFT only"),
        ({"method": "orpo", "method_params": {"simpo_beta": 1.0}}, "does not apply"),
        ({"method": "orpo", "method_params": {"orpo_beta": 0}}, "orpo_beta"),
        ({"lora_alpha": 0}, "lora_alpha"),
        ({"lora_dropout": 1.0}, "lora_dropout"),
        ({"target_modules": "q_proj;rm -rf"}, "target_modules"),
        ({"run_name": "a b"}, "Run name"),
        ({"gpu_max_temp": 30}, "temperature"),
    ],
)
def test_validation_refuses_bad_knobs(sandbox, overrides, needle):
    _out, data = sandbox
    spec = JobSpec(kind="sft", model="org/m", dataset_path=str(data), **overrides)
    with pytest.raises(JobValidationError, match=needle):
        ui_jobs._validate_spec(spec)


def test_full_fine_tune_is_single_run_only(sandbox):
    _out, data = sandbox
    spec = JobSpec(kind="multi_run", model="org/m", dataset_path=str(data), mode="full")
    with pytest.raises(JobValidationError, match="single run"):
        ui_jobs._validate_spec(spec)


def test_full_sft_and_all_linear_are_accepted(sandbox):
    _out, data = sandbox
    ui_jobs._validate_spec(JobSpec(kind="sft", model="org/m", dataset_path=str(data), mode="full"))
    ui_jobs._validate_spec(
        JobSpec(kind="sft", model="org/m", dataset_path=str(data), target_modules="all-linear")
    )


# ---- the inline estimate == `backprop estimate-vram` ----------------------------------


def test_verdict_matches_estimate_vram(capsys):
    from backpropagate.cli import main

    got = vram_verdict("meta-llama/Llama-3.2-3B-Instruct", lora_r=64, batch="4",
                       base_4bit=False, gradient_checkpointing=False, card_gb=24.0)
    rc = main(["estimate-vram", "meta-llama/Llama-3.2-3B-Instruct", "--lora-r", "64",
               "--batch-size", "4", "--no-4bit", "--no-gradient-checkpointing",
               "--vram-gb", "24", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    # stdout can carry structured log lines before the payload: take the
    # JSON object that has the per-config estimate.
    decoder, cli, i = json.JSONDecoder(), None, 0
    while cli is None and (i := out.find("{", i)) != -1:
        try:
            obj, _end = decoder.raw_decode(out, i)
        except json.JSONDecodeError:
            obj = None
        cli = obj if isinstance(obj, dict) and "per_config_estimate" in obj else None
        i += 1
    assert cli is not None, out
    assert got["total_gb"] == pytest.approx(cli["per_config_estimate"]["total_gb"], abs=0.01)


def test_auto_batch_uses_the_cli_tier_for_the_card():
    assert vram_verdict("org/m-7B", card_gb=31.8)["batch"] == 6  # 32 GB tier
    assert vram_verdict("org/m-7B", card_gb=16.0)["batch"] == 2


@pytest.mark.parametrize(
    ("card", "verdict"),
    [(80.0, "fits"), (None, "unknown")],
)
def test_verdict_buckets(card, verdict):
    assert vram_verdict("org/m-1B", lora_r=16, batch="1", card_gb=card)["verdict"] == verdict


def test_verdict_tight_and_wont_fit():
    total = vram_verdict("org/m-7B", lora_r=16, batch="1", card_gb=1000.0)["total_gb"]
    assert vram_verdict("org/m-7B", lora_r=16, batch="1", card_gb=total / 0.95)["verdict"] == "tight"
    assert vram_verdict("org/m-7B", lora_r=16, batch="1", card_gb=total * 0.5)["verdict"] == "wont_fit"


# ---- estimator fixes ---------------------------------------------------------------------


def test_sub_billion_ids_parse_in_millions():
    assert _estimate_param_count_billions("HuggingFaceTB/SmolLM2-135M-Instruct") == pytest.approx(0.135)
    assert _estimate_param_count_billions("org/thing-360m") == pytest.approx(0.36)
    assert _estimate_param_count_billions("org/Qwen2.5-7B-Instruct") == pytest.approx(7.0)


def test_full_mode_prices_a_16_bit_base():
    full = estimate_vram("org/m-1B", mode="full", batch_size=1)
    assert full.model_weights_gb == pytest.approx(1e9 * 2 / 1024**3, rel=1e-6)
    assert any("16-bit base" in n for n in full.notes)


def test_gradient_checkpointing_off_raises_lora_activations():
    on = estimate_vram("org/m-7B", lora_r=16, batch_size=2)
    off = estimate_vram("org/m-7B", lora_r=16, batch_size=2, gradient_checkpointing=False)
    assert off.activations_gb > on.activations_gb * 4


def test_small_models_use_their_size_class_shape():
    small = estimate_vram("org/m-1B", lora_r=16, batch_size=4)
    big = estimate_vram("org/m-7B", lora_r=16, batch_size=4)
    assert small.activations_gb < big.activations_gb


def test_shape_comes_from_a_local_config(tmp_path):
    model_dir = tmp_path / "tiny"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 4,
        "intermediate_size": 1024, "vocab_size": 1000, "tie_word_embeddings": True,
    }), encoding="utf-8")
    est = estimate_vram(str(model_dir), lora_r=8, batch_size=1)
    assert est.param_count_billions < 0.01
    assert any("config.json" in n for n in est.notes)


# ---- the form -> spec mapping --------------------------------------------------------------


reflex = pytest.importorskip("reflex", reason="reflex is required (install backpropagate[ui])")


def test_form_fields_map_to_spec_fields():
    from backpropagate import ui_state as us

    s = us.TrainState()
    s.set_method("orpo")
    s.set_method_param("orpo_beta", "0.3")
    s.apply_lora_shape("fast")
    s.set_train_mode("lora")
    fields = us._training_spec_fields(s)
    assert fields["method"] == "orpo" and fields["method_params"] == {"orpo_beta": 0.3}
    assert (fields["lora_r"], fields["lora_alpha"], fields["target_modules"]) == (16, 32, "q_proj,v_proj")
    assert fields["mode"] == "lora" and fields["base_4bit"] is False
    assert fields["gpu_max_temp"] == 90.0
    JobSpec(kind="sft", **fields)  # every key is a JobSpec field


def test_preset_fills_model_and_rank():
    from backpropagate import ui_state as us

    s = us.TrainState()
    s.set_preset("llama-3.2-1b")
    assert s.model == "meta-llama/Llama-3.2-1B-Instruct"
    assert (s.lora_r, s.lora_alpha) == (64, 128)
    assert s.lora_shape == "custom"
    s.set_model("someone/else")
    assert s.preset == "custom"


def test_full_mode_needs_sft():
    from backpropagate import ui_state as us

    s = us.TrainState()
    s.set_method("kto")
    s.set_train_mode("full")
    assert s.train_mode == "qlora" and "SFT" in s.job_refusal
    s.set_method("sft")
    s.set_train_mode("full")
    s.set_method("simpo")  # leaving SFT drops full back to QLoRA
    assert s.train_mode == "qlora"


def test_safety_event_names_the_stop():
    from backpropagate import ui_state as us

    s = us.TrainState()
    s.job_kind = "sft"
    s._apply_job_event({"kind": "safety", "reason": "GPU at 91 °C, above the 90 °C limit"})
    s._finalize_job({"status": "stopped"})
    assert s.run_state == "stopped"
    assert "temperature limit" in s.events[-1]["msg"]
    assert "91 °C" in s.events[-1]["msg"]
