# Four small things a new user trips over in the web UI.
"""Seen in real-browser screenshots on 2026-10-02:

* the Events panel showed UTC while the Runs page showed local time;
* the Runs page said "1 runs";
* a long dataset path hid the file name on the training forms;
* the note under the model field was written for insiders.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

pytest.importorskip("reflex")

import backpropagate.ui_state as us  # noqa: E402
from backpropagate.config import MODEL_PRESETS  # noqa: E402

# ---- one clock ------------------------------------------------------------------------


def test_event_times_are_local_time(monkeypatch):
    class _Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            # 16:17 on the wall; 20:17 in UTC.
            local = dt.datetime(2026, 10, 2, 16, 17, 50)
            if tz is None:
                return local
            return dt.datetime(2026, 10, 2, 20, 17, 50, tzinfo=dt.timezone.utc).astimezone(tz)

    monkeypatch.setattr(dt, "datetime", _Clock)
    assert us._ts_now() == "16:17:50"


def test_event_times_look_like_a_clock():
    assert re.fullmatch(r"\d\d:\d\d:\d\d", us._ts_now())


def test_a_start_time_that_names_its_zone_is_shown_in_local_time():
    stamp = "2026-10-02T20:17:50Z"
    local = dt.datetime(2026, 10, 2, 20, 17, 50, tzinfo=dt.timezone.utc).astimezone()
    assert us._fmt_started(stamp) == local.strftime("%Y-%m-%d %H:%M")
    assert us._fmt_started("2026-10-02T20:17:50+02:00") == (
        dt.datetime(2026, 10, 2, 18, 17, 50, tzinfo=dt.timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M")
    )


def test_a_start_time_without_a_zone_is_already_local():
    assert us._fmt_started("2026-10-02T04:45:42.269557") == "2026-10-02 04:45"


@pytest.mark.parametrize(
    ("value", "shown"),
    [(None, "-"), ("", "-"), ("-", "-"), ("yesterday at noon!", "yesterday at noo")],
)
def test_a_start_time_that_is_missing_or_unreadable_never_raises(value, shown):
    assert us._fmt_started(value) == shown


# ---- "1 run" ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "label"), [(0, "0 runs"), (1, "1 run"), (2, "2 runs"), (1200, "1,200 runs")]
)
def test_the_runs_line_counts_in_plain_english(rows, label):
    state = us.RunsState()
    state.runs = [{"run_id": str(i)} for i in range(rows)]
    assert state.runs_count_label == label


def test_the_runs_page_uses_that_label():
    from backpropagate.ui_app.pages.runs import runs_page

    rendered = str(runs_page().render())
    assert "runs_count_label" in rendered
    assert " runs · updated" not in rendered


# ---- which file is selected ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "name"),
    [
        ("C:\\Users\\someone\\.backpropagate\\ui-outputs\\datasets\\recipes-prepared.jsonl",
         "recipes-prepared.jsonl"),
        ("/home/someone/.backpropagate/ui-outputs/uploads/data.jsonl", "data.jsonl"),
        ("data.jsonl", "data.jsonl"),
        ("  /a/b/c.json  ", "c.json"),
        ("C:\\a\\folder\\", "folder"),
        ("", ""),
    ],
)
def test_the_file_name_is_the_end_of_the_path_on_any_system(path, name):
    assert us._file_name(path) == name


@pytest.mark.parametrize("state_cls", [us.TrainState, us.MultiRunState])
def test_both_training_forms_name_the_selected_file(state_cls):
    state = state_cls()
    assert state.dataset_file_name == ""
    state.dataset_path = "E:\\a-very\\long\\folder\\name\\datasets\\bakery-answers-prepared.jsonl"
    assert state.dataset_file_name == "bakery-answers-prepared.jsonl"


@pytest.mark.parametrize(
    "page", ["backpropagate.ui_app.pages.train:train_page",
             "backpropagate.ui_app.pages.multi_run:multi_run_page"]
)
def test_both_training_pages_show_the_file_name_once(page):
    import importlib

    module, func = page.split(":")
    rendered = str(getattr(importlib.import_module(module), func)().render())
    assert rendered.count("bp-dataset-file") == 1
    assert "dataset_file_name" in rendered


# ---- model notes in plain words -----------------------------------------------------------


def test_every_model_preset_has_a_plain_note():
    assert set(us.MODEL_NOTES) == set(MODEL_PRESETS)


@pytest.mark.parametrize("key", sorted(us.MODEL_NOTES))
def test_a_model_note_is_short_and_free_of_shorthand(key):
    note = us.MODEL_NOTES[key]
    assert 30 <= len(note) <= 170 and note.endswith(".")
    lowered = note.lower()
    for word in ("vram", "qlora", "smoke test", "post-training", "baseline", "mmlu",
                 "humaneval", "envelope", "adamw", "gib", "max_seq", "plumbing", "drop-in"):
        assert word not in lowered, (key, word)


def test_the_note_under_the_model_field_is_the_plain_one_plus_the_licence():
    options = {o["key"]: o for o in us.model_preset_options()}
    note = options["llama-3.2-1b"]["note"]
    assert note.startswith(us.MODEL_NOTES["llama-3.2-1b"])
    assert note.endswith("License: Llama-3.2-Community.")
    # A licence caveat the preset carries still follows the note.
    assert "non-commercial" in options["qwen2.5-3b"]["note"]


def test_a_preset_without_a_plain_note_falls_back_to_its_own(monkeypatch):
    monkeypatch.delitem(us.MODEL_NOTES, "mistral-7b")
    options = {o["key"]: o for o in us.model_preset_options()}
    assert options["mistral-7b"]["note"].startswith(MODEL_PRESETS["mistral-7b"].best_for)


@pytest.mark.parametrize("state_cls", [us.TrainState, us.MultiRunState])
def test_the_form_shows_the_plain_note_for_its_preset(state_cls):
    state = state_cls()
    state.preset = "qwen2.5-7b"
    assert state.preset_note.startswith(us.MODEL_NOTES["qwen2.5-7b"])
    state.preset = "custom"
    assert "Hugging Face model id" in state.preset_note
