# The web UI explains itself, and its LoRA shape follows the GPU.
"""The "i" tips, the shape that fits the card, and the estimate against free memory.

The web UI is for people who are curious, not only for people who already
know the vocabulary. These pin the pieces that make that true:

* every tip a page uses exists, is plain and is short enough to read;
* the tip button is a real, named button with its words in the page for
  screen readers;
* the form opens on the largest LoRA shape that fits the GPU, says why, and
  leaves the user's own choice alone;
* the estimate is compared with the memory that is free, and the batch shown
  is the batch the trainer will pick.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("reflex", reason="reflex is required (install backpropagate[ui])")

import backpropagate.trainer as trainer_mod  # noqa: E402
from backpropagate import ui_jobs  # noqa: E402
from backpropagate import ui_state as us  # noqa: E402
from backpropagate.config import LORA_PRESET_ORDER, LORA_PRESETS  # noqa: E402
from backpropagate.ui_app import help_text  # noqa: E402
from backpropagate.ui_app.help_text import TIPS, Tip  # noqa: E402

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


def _free(monkeypatch, gib):
    monkeypatch.setattr(trainer_mod, "_free_vram_gib", lambda: gib)


# ---- the words ---------------------------------------------------------------------------


def _rendered(page_path: str) -> str:
    """The page's component tree as text (what the browser is sent)."""
    import importlib

    module_name, func = page_path.split(":")
    component = getattr(importlib.import_module(module_name), func)()
    return str(component.render())


@pytest.mark.parametrize(
    "page",
    [
        "backpropagate.ui_app.pages.train:train_page",
        "backpropagate.ui_app.pages.multi_run:multi_run_page",
    ],
)
def test_every_tip_a_page_uses_exists(page):
    used = set(re.findall(r'data-tip[\\"\':= ]+([a-z_]+)', _rendered(page)))
    assert used, "the page renders no tips"
    assert used <= set(TIPS), used - set(TIPS)


def test_single_run_explains_every_area():
    used = set(re.findall(r'data-tip[\\"\':= ]+([a-z_]+)', _rendered(
        "backpropagate.ui_app.pages.train:train_page")))
    assert {
        "page_single_run", "model", "dataset", "method", "mode", "steps", "batch_size",
        "learning_rate", "lora", "rank", "alpha", "dropout", "target_modules", "gpu_temp",
        "run_name", "gradient_checkpointing", "vram", "measure", "loss", "gpu", "events",
    } <= used


def test_multi_run_explains_its_own_settings():
    used = set(re.findall(r'data-tip[\\"\':= ]+([a-z_]+)', _rendered(
        "backpropagate.ui_app.pages.multi_run:multi_run_page")))
    assert {"page_multi_run", "runs", "samples_per_run", "merge_mode", "lora", "rank"} <= used


@pytest.mark.parametrize("key", sorted(TIPS))
def test_a_tip_is_short_and_complete(key):
    tip = TIPS[key]
    assert tip.title and not tip.title.endswith(".")
    assert 1 <= len(tip.body) <= 3
    for paragraph in tip.body:
        assert 20 <= len(paragraph) <= 230, (key, len(paragraph))
        assert paragraph.endswith(".")
    assert len(tip.start) <= 120
    assert len(tip.text) <= 620  # a card, not an article


@pytest.mark.parametrize("key", sorted(TIPS))
def test_a_tip_does_not_lean_on_unexplained_shorthand(key):
    # Words a newcomer cannot be expected to know, unless the tip is the one
    # that explains them.
    text = TIPS[key].text.lower()
    for jargon, explained_in in (
        ("hyperparameter", ()),
        ("oom", ()),
        ("nf4", ()),
        ("bf16", ()),
        ("sdpa", ()),
        ("gradient accumulation", ()),
        ("tokenizer", ()),
        ("vram", ("vram",)),
    ):
        if key not in explained_in:
            assert jargon not in text, (key, jargon)


def test_tip_links_point_at_handbook_pages_that_exist():
    from pathlib import Path

    handbook = Path(__file__).resolve().parent.parent / "site/src/content/docs/handbook"
    for key, tip in TIPS.items():
        if not tip.link:
            continue
        page = tip.link.strip("/").split("/")[0]
        assert (handbook / f"{page}.md").is_file(), (key, tip.link)
        assert tip.url.startswith(help_text.HANDBOOK + "/")


def test_the_starting_point_is_read_out_with_the_tip():
    tip = Tip("A title", ("One sentence.",), start="Leave it.")
    assert tip.text == "A title. One sentence. Good starting point: Leave it."


def test_an_unknown_tip_is_a_programming_error():
    with pytest.raises(KeyError):
        help_text.tip("no_such_tip")


def test_the_tip_button_is_named_and_described():
    from backpropagate.ui_app.components.info_tip import info_tip

    html = str(info_tip("rank").render())
    assert "About: The size of the adapter" in html
    assert "bp-tip-rank" in html  # aria-describedby target, always in the page
    assert "bp-visually-hidden" in html
    assert "A higher rank lets the adapter capture more of your data" in html


# ---- which shape fits ---------------------------------------------------------------------


def test_the_ui_offers_exactly_the_trainer_presets():
    assert tuple(us.LORA_SHAPES) == LORA_PRESET_ORDER
    for name, (rank, alpha, targets) in us.LORA_SHAPES.items():
        preset = LORA_PRESETS[name]
        expected = (
            preset.target_modules
            if isinstance(preset.target_modules, str)
            else ", ".join(preset.target_modules)
        )
        assert (rank, alpha, targets) == (preset.r, preset.lora_alpha, expected)
        assert name in us.LORA_SHAPE_NOTES


@pytest.mark.parametrize(
    ("free", "recommended"), [(30.5, "quality"), (14.5, "balanced"), (10.8, "fast")]
)
def test_shape_options_name_the_shape_that_fits(qwen7b, monkeypatch, free, recommended):
    _free(monkeypatch, free)
    out = ui_jobs.lora_shape_options(qwen7b)
    assert out["shapes"] == {"quality": 17.0, "balanced": 11.0, "fast": 9.0}
    assert (out["recommended"], out["fits"], out["free_gb"]) == (recommended, True, free)


def test_shape_options_when_nothing_fits(qwen7b, monkeypatch):
    _free(monkeypatch, 6.0)
    out = ui_jobs.lora_shape_options(qwen7b)
    assert (out["recommended"], out["fits"]) == ("fast", False)


def test_shape_options_without_a_gpu_reading_recommend_nothing(qwen7b):
    out = ui_jobs.lora_shape_options(qwen7b)
    assert out["recommended"] == "" and out["free_gb"] is None
    assert set(out["shapes"]) == {"quality", "balanced", "fast"}


def test_shape_options_never_raise(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no estimate")

    monkeypatch.setattr(trainer_mod, "estimate_vram", boom)
    assert ui_jobs.lora_shape_options("org/x") == {
        "shapes": {}, "free_gb": None, "recommended": "", "fits": True,
    }


# ---- the estimate bar ---------------------------------------------------------------------


def test_the_verdict_is_against_free_memory_when_it_is_known(qwen7b, monkeypatch):
    """Another program on the card leaves less than its size. Against the
    whole card this setup read "Fits"; the run would have been refused."""
    _free(monkeypatch, 14.5)
    out = ui_jobs.vram_verdict(qwen7b, lora_r=256, batch="1", card_gb=31.8)
    assert out["total_gb"] == pytest.approx(17.0, abs=0.1)
    assert (out["against"], out["budget_gb"], out["verdict"]) == ("free", 14.5, "wont_fit")


def test_the_verdict_falls_back_to_the_card_without_a_reading(qwen7b):
    out = ui_jobs.vram_verdict(qwen7b, lora_r=256, batch="1", card_gb=31.8)
    assert (out["against"], out["budget_gb"], out["verdict"]) == ("card", 31.8, "fits")


def test_auto_batch_shown_is_the_batch_the_trainer_picks(qwen7b, monkeypatch):
    _free(monkeypatch, 14.5)
    out = ui_jobs.vram_verdict(qwen7b, lora_r=64, batch="auto", card_gb=31.8)
    # The 32 GB tier says 6; rank 64 at batch 2 (11.7 GB) is the largest that
    # fits 90% of 14.5 GB.
    assert (out["tier_batch"], out["batch"]) == (6, 2)
    assert out["verdict"] == "fits"


def test_an_explicit_batch_is_shown_as_given(qwen7b, monkeypatch):
    _free(monkeypatch, 14.5)
    out = ui_jobs.vram_verdict(qwen7b, lora_r=64, batch="4", card_gb=31.8)
    assert (out["tier_batch"], out["batch"]) == (None, 4)


def test_a_preference_method_never_reads_measured(qwen7b, monkeypatch):
    import backpropagate.vram_calibration as vc

    measured = vc.Calibration(
        model=qwen7b, mode="lora", base_4bit=True, machine={}, load_gib=6.7, floor_gib=2.0,
        quad_bytes=0.0, lin_bytes=650_000.0, probe_trainable_params=1, seq_min=1024, seq_max=2048,
    )
    monkeypatch.setattr(vc, "lookup", lambda *a, **k: measured)
    assert ui_jobs.vram_verdict(qwen7b, batch="2", card_gb=31.8)["source"] == "measured"
    assert ui_jobs.vram_verdict(qwen7b, batch="2", card_gb=31.8, method="orpo")["source"] == (
        "estimate"
    )


# ---- the form state -----------------------------------------------------------------------


def _options(recommended="balanced", fits=True, free=14.5):
    return {"shapes": {"quality": 17.0, "balanced": 11.0, "fast": 9.0},
            "free_gb": free, "recommended": recommended, "fits": fits}


def _apply(s, options):
    """What refresh_estimate does with the options, without the event loop."""
    s.lora_shape_gb = dict(options["shapes"])
    s.lora_free_gb = float(options["free_gb"] or 0.0)
    s.lora_recommended = options["recommended"]
    s.lora_recommended_fits = options["fits"]
    if s.lora_follow_gpu and s.lora_recommended in us.LORA_SHAPES:
        s.lora_r, s.lora_alpha, s.target_modules = us.LORA_SHAPES[s.lora_recommended]


@pytest.mark.parametrize("state_cls", [us.TrainState, us.MultiRunState])
def test_the_form_opens_on_the_shape_that_fits_and_says_why(state_cls):
    s = state_cls()
    assert s.lora_follow_gpu is True
    _apply(s, _options())
    assert (s.lora_r, s.lora_alpha, s.target_modules, s.lora_shape) == (
        64, 128, "all-linear", "balanced",
    )
    assert s.lora_caption == (
        "Chosen for your GPU: the largest shape that fits this model (about 11.0 GB of the "
        "14.5 GB free). Quality would need about 17.0 GB."
    )
    assert (s.lora_gb_quality, s.lora_gb_balanced, s.lora_gb_fast) == (
        "17.0 GB", "11.0 GB", "9.0 GB",
    )
    assert s.lora_can_follow_gpu is False


@pytest.mark.parametrize("state_cls", [us.TrainState, us.MultiRunState])
def test_a_shape_the_user_picks_stays(state_cls):
    s = state_cls()
    _apply(s, _options())
    s.apply_lora_shape("quality")
    assert s.lora_follow_gpu is False and s.lora_shape == "quality"
    _apply(s, _options())  # a later refresh must not take it back
    assert s.lora_shape == "quality"
    assert s.lora_caption.endswith("Your GPU fits Balanced for this model.")
    assert s.lora_can_follow_gpu is True
    s.follow_gpu_shape()
    assert s.lora_follow_gpu is True


@pytest.mark.parametrize("state_cls", [us.TrainState, us.MultiRunState])
@pytest.mark.parametrize(
    ("setter", "value"),
    [("set_lora_r", "32"), ("set_lora_alpha", "64"), ("set_target_modules", "q_proj")],
)
def test_editing_the_numbers_by_hand_is_also_the_users_choice(state_cls, setter, value):
    s = state_cls()
    getattr(s, setter)(value)
    assert s.lora_follow_gpu is False
    assert s.lora_shape == "custom"
    assert s.lora_caption == "Your own rank, alpha and target modules."


@pytest.mark.parametrize("state_cls", [us.TrainState, us.MultiRunState])
def test_retyping_the_same_value_changes_nothing(state_cls):
    s = state_cls()
    s.set_lora_r("256")
    s.set_lora_alpha("512")
    s.set_target_modules("all-linear")
    assert s.lora_follow_gpu is True


def test_nothing_fits_is_said_plainly():
    s = us.TrainState()
    _apply(s, _options(recommended="fast", fits=False, free=6.0))
    assert s.lora_shape == "fast"
    assert "No shape is estimated to fit this model in the 6.0 GB free" in s.lora_caption


def test_without_a_gpu_reading_the_form_keeps_quality():
    s = us.TrainState()
    _apply(s, {"shapes": {"quality": 17.0, "balanced": 11.0, "fast": 9.0},
               "free_gb": None, "recommended": "", "fits": True})
    assert s.lora_shape == "quality"
    assert s.lora_caption == us.LORA_SHAPE_NOTES["quality"]


def test_the_estimate_label_names_what_it_is_compared_with():
    s = us.TrainState()
    s._apply_verdict({"verdict": "wont_fit", "total_gb": 17.0, "batch": 1, "tier_batch": 6,
                      "budget_gb": 14.5, "against": "free", "source": "estimate"})
    assert s.vram_est_label == "17.0 GB of 14.5 GB free"
    assert s.vram_est_pct == "100.0%"
    s._apply_verdict({"verdict": "fits", "total_gb": 17.0, "batch": 1, "budget_gb": 31.8,
                      "against": "card", "source": "estimate"})
    assert s.vram_est_label == "17.0 GB of 31.8 GB"


def test_the_detail_line_says_when_the_batch_was_lowered():
    s = us.TrainState()
    s._apply_verdict({"verdict": "fits", "total_gb": 11.7, "batch": 2, "tier_batch": 6,
                      "budget_gb": 14.5, "against": "free", "source": "estimate"})
    assert "batch 2, chosen automatically for this model and GPU (lowered from 6 to fit)" in (
        s.vram_est_detail
    )
    s.method = "orpo"
    assert "ORPO compares two answers per example" in s.vram_est_detail


def test_a_setup_that_will_not_fit_offers_the_shape_that_does():
    s = us.TrainState()
    _apply(s, _options())
    s.apply_lora_shape("quality")
    s._apply_verdict({"verdict": "wont_fit", "total_gb": 17.0, "batch": 1, "budget_gb": 14.5,
                      "against": "free", "source": "estimate"})
    assert s.vram_fix_label == "Use Balanced"
    follow_up = s.apply_vram_fix()
    assert s.lora_follow_gpu is True and follow_up is us.TrainState.refresh_estimate


def test_an_explicit_batch_that_will_not_fit_offers_the_automatic_one():
    s = us.TrainState()
    _apply(s, _options())  # already on the recommended shape
    s.set_batch_size("8")
    s._apply_verdict({"verdict": "wont_fit", "total_gb": 19.0, "batch": 8, "budget_gb": 14.5,
                      "against": "free", "source": "estimate"})
    assert s.vram_fix_label == "Use automatic batch"
    s.apply_vram_fix()
    assert s.batch_size == "auto"


def test_no_fix_is_offered_when_it_fits_or_nothing_would_help():
    s = us.TrainState()
    _apply(s, _options())
    s._apply_verdict({"verdict": "fits", "total_gb": 11.7, "batch": 2, "budget_gb": 14.5,
                      "against": "free", "source": "estimate"})
    assert s.vram_fix_label == ""
    assert s.apply_vram_fix() is None
    s._apply_verdict({"verdict": "wont_fit", "total_gb": 20.0, "batch": 1, "budget_gb": 14.5,
                      "against": "free", "source": "estimate"})
    assert s.vram_fix_label == ""  # already the recommended shape on automatic batch


def test_full_fine_tune_offers_no_lora_fix():
    s = us.TrainState()
    _apply(s, _options())
    s.train_mode = "full"
    s._apply_verdict({"verdict": "wont_fit", "total_gb": 40.0, "batch": 1, "budget_gb": 14.5,
                      "against": "free", "source": "estimate"})
    assert s.vram_fix_label == ""
