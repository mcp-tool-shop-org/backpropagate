# The default LoRA shape and automatic batch follow the GPU.
"""``resolve_lora_shape`` and ``Trainer._fit_auto_batch``.

The default used to be rank 256 on every linear layer whatever the card. On a
7B model that needs about 17 GB at batch 1 (measured: 16.61 GiB at batch 2 on
an RTX 5090), so a 16 GB card could not run it and halving the batch could
not help. ``--lora-preset fast`` was also stored and never applied.

Now: quality, balanced (rank 64) and fast are real shapes, and the default
picks the largest that is estimated to fit the GPU memory that is free.
"""

from __future__ import annotations

import logging

import pytest

import backpropagate.trainer as trainer_mod
from backpropagate.cli import create_parser
from backpropagate.config import LORA_PRESET_ORDER, LORA_PRESETS, get_lora_preset
from backpropagate.exceptions import InvalidSettingError
from backpropagate.trainer import Trainer, resolve_lora_shape

# Real shapes (config.json), so the numbers match the handbook.
QWEN_7B = {"hidden_size": 3584, "num_hidden_layers": 28, "num_attention_heads": 28,
           "num_key_value_heads": 4, "intermediate_size": 18944, "vocab_size": 152064,
           "tie_word_embeddings": False}
LLAMA_3B = {"hidden_size": 3072, "num_hidden_layers": 28, "num_attention_heads": 24,
            "num_key_value_heads": 8, "intermediate_size": 8192, "vocab_size": 128256,
            "tie_word_embeddings": True}


@pytest.fixture
def qwen7b(monkeypatch):
    monkeypatch.setattr(trainer_mod, "_cached_model_config", lambda model: dict(QWEN_7B))
    return "org/qwen-7b"


@pytest.fixture
def llama3b(monkeypatch):
    monkeypatch.setattr(trainer_mod, "_cached_model_config", lambda model: dict(LLAMA_3B))
    return "org/llama-3b"


def _free(monkeypatch, gib):
    monkeypatch.setattr(trainer_mod, "_free_vram_gib", lambda: gib)


# ---- the presets ------------------------------------------------------------------------


def test_three_presets_from_largest_to_smallest():
    assert LORA_PRESET_ORDER == ("quality", "balanced", "fast")
    assert [LORA_PRESETS[n].r for n in LORA_PRESET_ORDER] == [256, 64, 16]
    balanced = get_lora_preset("balanced")
    assert (balanced.lora_alpha, balanced.target_modules) == (128, "all-linear")


@pytest.mark.parametrize(
    ("name", "r", "alpha", "targets"),
    [
        ("quality", 256, 512, "all-linear"),
        ("balanced", 64, 128, "all-linear"),
        ("fast", 16, 32, ["q_proj", "v_proj"]),
    ],
)
def test_a_named_preset_is_that_shape_whatever_is_free(qwen7b, name, r, alpha, targets):
    shape = resolve_lora_shape(qwen7b, preset=name, free_gib=4.0)
    assert (shape.preset, shape.r, shape.lora_alpha, shape.target_modules) == (name, r, alpha, targets)
    assert shape.chosen_for_card is False and shape.note == ""


def test_an_explicit_field_replaces_that_field_of_a_named_preset(qwen7b):
    shape = resolve_lora_shape(qwen7b, preset="fast", lora_r=32, target_modules="all-linear")
    assert (shape.r, shape.lora_alpha, shape.target_modules) == (32, 32, "all-linear")


def test_an_unknown_preset_is_an_error(qwen7b):
    with pytest.raises(ValueError, match="auto, quality, balanced, fast"):
        resolve_lora_shape(qwen7b, preset="turbo")


# ---- the automatic choice ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("free", "expected"),
    [
        (30.5, "quality"),   # 32 GB card: 17.0 GB at batch 1 fits
        (22.5, "quality"),   # 24 GB card
        (14.5, "balanced"),  # 16 GB card: quality needs 17.0, balanced 11.0
        (13.0, "balanced"),
        (10.8, "fast"),      # 12 GB card: balanced needs 11.0, fast about 9
    ],
)
def test_a_7b_model_gets_the_largest_shape_that_fits(qwen7b, free, expected):
    shape = resolve_lora_shape(qwen7b, free_gib=free)
    assert shape.preset == expected and shape.chosen_for_card and shape.fits
    assert shape.r == LORA_PRESETS[expected].r
    if expected == "quality":
        assert shape.note == "" and shape.target_modules is None
    else:
        assert f"LoRA preset {expected}" in shape.note and "17.0 GB" in shape.note


def test_a_3b_model_keeps_quality_on_a_16gb_card(llama3b):
    # Measured: rank 256 all-linear on Llama 3.2 3B peaked at 9.0 GiB at batch 2.
    assert resolve_lora_shape(llama3b, free_gib=14.5).preset == "quality"


def test_nothing_fits_is_fast_with_a_warning(qwen7b):
    shape = resolve_lora_shape(qwen7b, free_gib=6.0)
    assert (shape.preset, shape.fits) == ("fast", False)
    assert "No LoRA preset is estimated to fit" in shape.note


def test_auto_is_the_same_as_no_preset(qwen7b):
    assert resolve_lora_shape(qwen7b, preset="auto", free_gib=14.5).preset == "balanced"
    assert resolve_lora_shape(qwen7b, preset=None, free_gib=14.5).preset == "balanced"


def test_no_gpu_reading_keeps_the_long_standing_default(qwen7b):
    shape = resolve_lora_shape(qwen7b, free_gib=None)
    assert (shape.preset, shape.r, shape.lora_alpha, shape.target_modules) == (
        "quality", 256, 512, None,
    )
    assert shape.chosen_for_card is False


@pytest.mark.parametrize(
    "explicit",
    [{"lora_r": 128}, {"lora_alpha": 64}, {"target_modules": ["q_proj"]}, {"lora_r": 256}],
)
def test_an_explicit_field_turns_the_automatic_choice_off(qwen7b, explicit):
    shape = resolve_lora_shape(qwen7b, free_gib=6.0, **explicit)
    assert shape.preset == "custom" and shape.chosen_for_card is False
    assert shape.r == explicit.get("lora_r", 256)
    assert shape.lora_alpha == explicit.get("lora_alpha", 512)
    assert shape.target_modules == explicit.get("target_modules")


def test_lora_settings_changed_by_the_user_are_respected(qwen7b, monkeypatch):
    monkeypatch.setattr(trainer_mod.settings.lora, "r", 32)
    shape = resolve_lora_shape(qwen7b, free_gib=6.0)
    assert (shape.preset, shape.r) == ("custom", 32)


def test_a_failing_estimate_keeps_the_default(qwen7b, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no estimate")

    monkeypatch.setattr(trainer_mod, "estimate_vram", boom)
    assert resolve_lora_shape(qwen7b, free_gib=14.5).preset == "quality"


def test_a_16bit_base_is_priced_as_one(qwen7b):
    # 7.6B parameters in 16-bit are 14 GiB before any adapter.
    assert resolve_lora_shape(qwen7b, free_gib=14.5, quantize_base=False).fits is False


# ---- the trainer ------------------------------------------------------------------------


def _trainer(model, **kw):
    kw.setdefault("use_unsloth", False)
    kw.setdefault("report_to", "none")
    return Trainer(model=model, **kw)


def test_the_fast_preset_is_applied(qwen7b):
    """It used to be stored and ignored: the run still trained rank 256."""
    t = _trainer(qwen7b, lora_preset="fast", batch_size=1)
    assert (t.lora_r, t.lora_alpha, t.lora_preset) == (16, 32, "fast")
    assert t._resolved_target_modules() == ["q_proj", "v_proj"]


def test_the_balanced_preset_is_applied(qwen7b):
    t = _trainer(qwen7b, lora_preset="balanced", batch_size=1)
    assert (t.lora_r, t.lora_alpha, t.lora_preset) == (64, 128, "balanced")
    assert t._resolved_target_modules() == "all-linear"


def test_default_trainer_without_a_gpu_reading_is_unchanged(qwen7b):
    t = _trainer(qwen7b, batch_size=1)
    assert (t.lora_r, t.lora_alpha, t.lora_preset) == (256, 512, "quality")
    assert t._target_modules_override is None


def test_default_trainer_on_a_16gb_card_picks_balanced_and_says_why(qwen7b, monkeypatch, caplog):
    _free(monkeypatch, 14.5)
    with caplog.at_level(logging.INFO, logger=trainer_mod.logger.name):
        t = _trainer(qwen7b, batch_size=1)
    assert (t.lora_r, t.lora_alpha, t.lora_preset) == (64, 128, "balanced")
    assert any("LoRA preset balanced" in r.getMessage() for r in caplog.records)


def test_an_explicit_rank_is_never_changed(qwen7b, monkeypatch):
    _free(monkeypatch, 14.5)
    t = _trainer(qwen7b, lora_r=256, batch_size=1)
    assert (t.lora_r, t.lora_alpha, t.lora_preset) == (256, 512, "quality")


def test_nothing_fits_warns(qwen7b, monkeypatch, caplog):
    _free(monkeypatch, 6.0)
    with caplog.at_level(logging.WARNING, logger=trainer_mod.logger.name):
        t = _trainer(qwen7b, batch_size=1)
    assert t.lora_r == 16
    assert any(
        r.levelno == logging.WARNING and "No LoRA preset is estimated to fit" in r.getMessage()
        for r in caplog.records
    )


def test_an_unknown_preset_is_a_setting_error(qwen7b):
    with pytest.raises(InvalidSettingError) as info:
        _trainer(qwen7b, lora_preset="turbo", batch_size=1)
    assert info.value.setting_name == "lora_preset"


def test_full_mode_has_no_lora_shape_to_choose(monkeypatch):
    monkeypatch.setattr(trainer_mod, "_cached_model_config", lambda model: None)
    _free(monkeypatch, 14.5)
    t = _trainer("HuggingFaceTB/SmolLM2-135M-Instruct", mode="full", batch_size=1)
    assert t.lora_preset == "quality"  # untouched


# ---- the automatic batch ----------------------------------------------------------------


def _tier(monkeypatch, batch):
    monkeypatch.setattr(Trainer, "_detect_batch_size", lambda self: batch)


def test_auto_batch_keeps_the_tier_when_it_fits(qwen7b, monkeypatch):
    _tier(monkeypatch, 6)
    _free(monkeypatch, 30.5)  # rank 256 at batch 6 is estimated at 23.7 GB
    t = _trainer(qwen7b)
    assert (t.lora_preset, t.batch_size) == ("quality", 6)


def test_auto_batch_is_lowered_to_what_fits(qwen7b, monkeypatch, caplog):
    _tier(monkeypatch, 6)
    _free(monkeypatch, 22.5)  # batch 6 needs 23.7, batch 4 needs 20.7 > 20.25, batch 2 fits
    with caplog.at_level(logging.INFO, logger=trainer_mod.logger.name):
        t = _trainer(qwen7b)
    assert (t.lora_preset, t.batch_size) == ("quality", 2)
    assert any("Automatic batch size 2" in r.getMessage() for r in caplog.records)


def test_auto_batch_is_never_raised_above_the_tier(llama3b, monkeypatch):
    _tier(monkeypatch, 2)
    _free(monkeypatch, 30.5)  # batch 8 would fit easily
    assert _trainer(llama3b).batch_size == 2


def test_a_16gb_card_and_a_7b_model_start_at_a_batch_that_fits(qwen7b, monkeypatch):
    """The case that used to fail: tier batch 2 at rank 256 needs 17.8 GB."""
    _tier(monkeypatch, 2)
    _free(monkeypatch, 14.5)
    t = _trainer(qwen7b)
    assert (t.lora_r, t.batch_size) == (64, 2)  # 11.7 GB


def test_batch_one_that_does_not_fit_warns_and_stays_at_one(qwen7b, monkeypatch, caplog):
    _tier(monkeypatch, 2)
    _free(monkeypatch, 14.5)
    with caplog.at_level(logging.WARNING, logger=trainer_mod.logger.name):
        t = _trainer(qwen7b, lora_preset="quality")
    assert t.batch_size == 1
    assert any("Even batch size 1" in r.getMessage() for r in caplog.records)


def test_an_explicit_batch_is_never_changed(qwen7b, monkeypatch):
    _free(monkeypatch, 6.0)
    assert _trainer(qwen7b, batch_size=4).batch_size == 4


def test_auto_batch_without_a_gpu_reading_is_the_tier(qwen7b, monkeypatch):
    _tier(monkeypatch, 4)
    assert _trainer(qwen7b).batch_size == 4


# ---- the CLI ----------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["train", "multi-run"])
def test_cli_defaults_are_automatic(command):
    args = create_parser().parse_args([command, "--data", "d.jsonl"])
    assert args.lora_preset == "auto"
    assert args.lora_r is None


@pytest.mark.parametrize("preset", ["auto", "quality", "balanced", "fast"])
def test_cli_accepts_every_preset(preset):
    args = create_parser().parse_args(["train", "--data", "d.jsonl", "--lora-preset", preset])
    assert args.lora_preset == preset
