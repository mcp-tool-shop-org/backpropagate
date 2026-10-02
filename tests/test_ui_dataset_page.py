# The Dataset page does what it shows (ui_state.DatasetState + pages/dataset.py).
"""The Dataset page's settings used to be stored and never applied.

These pin what the page does now: an upload fills the preview and the
statistics from the file, the clean-up settings change what would be kept as
they change, "Save a cleaned copy" writes exactly that, and "Use in ..." hands
the file to a training form. No training, no GPU, no network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("reflex")

import backpropagate.dataset_prep as prep  # noqa: E402
import backpropagate.ui_state as us  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    return SimpleNamespace(
        home=home.resolve(), out=(home / ".backpropagate" / "ui-outputs").resolve()
    )


class _Reader:
    """Chunked async reader shaped like ``rx.upload``'s file objects."""

    def __init__(self, data: bytes, filename: str):
        self.filename = filename
        self._data = data
        self._pos = 0

    async def read(self, n=-1):
        if n is None or n < 0:
            n = len(self._data) - self._pos
        chunk = self._data[self._pos:self._pos + n]
        self._pos += n
        return chunk


def _alpaca(instruction: str, output: str) -> dict:
    return {"instruction": instruction, "input": "", "output": output}


# Seven examples: one repeat, one with an empty answer, one long, one very short.
_ROWS = [
    _alpaca("What is the capital of France?", "Paris."),
    _alpaca("Name a prime number.", "Seven is a prime number."),
    _alpaca("What is the capital of France?", "Paris."),
    _alpaca("Say nothing.", ""),
    _alpaca("Tell a long story.", "word " * 400),
    _alpaca("Hi", "Yo"),
    _alpaca("What is two plus two?", "Four."),
]


def _jsonl(rows, *extra_lines: str) -> bytes:
    return ("\n".join([json.dumps(r) for r in rows] + list(extra_lines)) + "\n").encode()


def _upload(state, data: bytes, name: str = "data.jsonl"):
    asyncio.run(state.handle_upload([_Reader(data, name)]))
    assert state.upload_error == "", state.upload_error
    return state


@pytest.fixture
def loaded(sandbox):
    return _upload(us.DatasetState(), _jsonl(_ROWS))


def _public(state) -> dict[str, str]:
    return {k: str(getattr(state, k)) for k in state.get_fields() if not k.startswith("_")}


# ---- an upload fills the page from the file ------------------------------------------


def test_an_upload_shows_what_the_file_contains(loaded):
    s = loaded
    assert s.detected_format == "Alpaca" and s.inspect_note == ""
    assert (s.record_count, s.dedup_hits, s.skipped_lines) == (7, 1, 0)
    assert 0 < s.shortest_tokens < s.avg_tokens < s.longest_tokens
    assert len(s.preview_records) == 5
    first = s.preview_records[0]
    assert first["number"] == "1" and int(first["tokens"]) > 0
    assert first["text"] == "User: What is the capital of France?\nAssistant: Paris."


def test_the_page_starts_with_what_the_default_settings_would_do(loaded):
    s = loaded
    assert (s.kept_count, s.removed_duplicate, s.removed_empty) == (5, 1, 1)
    assert (s.removed_short, s.removed_long, s.removed_count) == (0, 0, 2)
    assert s.cleanup_summary == "5 of 7 examples are kept. Removed: 1 repeat, 1 empty."
    assert s.can_save_copy is True


def test_lines_that_are_not_examples_are_reported(sandbox):
    s = _upload(us.DatasetState(), _jsonl(_ROWS[:2], "{not json"))
    assert (s.record_count, s.skipped_lines) == (2, 1)
    assert s.skipped_note.startswith("1 line was skipped")
    s = _upload(us.DatasetState(), _jsonl(_ROWS[:2], "{not json", "42"), "two.jsonl")
    assert s.skipped_note.startswith("2 lines were skipped")
    assert _upload(us.DatasetState(), _jsonl(_ROWS[:2]), "clean.jsonl").skipped_note == ""


def test_the_stats_are_shown_with_thousands_separated(loaded):
    loaded.record_count, loaded.longest_tokens = 125_000, 2_775
    assert loaded.stat_text["examples"] == "125,000"
    assert loaded.stat_text["longest"] == "2,775"
    assert set(loaded.stat_text) == {"examples", "repeats", "average", "shortest", "longest"}


def test_the_upload_handler_is_one_the_framework_accepts():
    # The page's upload returned HTTP 500 on every drop while this parameter
    # was a bare ``list``: Reflex only routes uploads to a handler that takes
    # ``list[rx.UploadFile]``. Unit tests call the handler directly, so only
    # the framework's own check sees the difference.
    from reflex_base.event import resolve_upload_handler_param

    name, annotation = resolve_upload_handler_param(us.DatasetState.handle_upload)
    assert name == "files" and "UploadFile" in str(annotation)


@pytest.mark.parametrize(
    ("row", "name"),
    [
        ({"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]},
         "ShareGPT"),
        ({"messages": [{"role": "user", "content": "Q"},
                       {"role": "assistant", "content": "A"}]}, "OpenAI"),
        ({"prompt": "Q", "chosen": "good", "rejected": "bad"}, "Preference pairs"),
        ("Plain text with no structure at all.", "Plain text"),
    ],
)
def test_each_layout_gets_its_everyday_name(sandbox, row, name):
    assert _upload(us.DatasetState(), _jsonl([row])).detected_format == name


def test_a_layout_that_is_not_recognised_is_not_named(sandbox):
    s = _upload(us.DatasetState(), _jsonl([{"foo": 1, "bar": [2, 3]}]))
    assert s.detected_format == "" and s.record_count == 1


def test_no_public_value_carries_the_home_folder(loaded, sandbox):
    loaded.save_cleaned_copy()
    assert loaded.prepared_name
    assert all(str(sandbox.home) not in value for value in _public(loaded).values())


def test_a_new_upload_replaces_everything_from_the_last_one(loaded):
    loaded.save_cleaned_copy()
    _upload(loaded, _jsonl(_ROWS[:2]), "second.jsonl")
    assert (loaded.record_count, loaded.dedup_hits, loaded.kept_count) == (2, 0, 2)
    assert (loaded.prepared_name, loaded.training_file_name) == ("", "second.jsonl")


def test_the_same_name_uploaded_again_is_read_again(loaded):
    _upload(loaded, _jsonl(_ROWS[:3]))  # data.jsonl again, different content
    assert (loaded.record_count, loaded.dedup_hits) == (3, 1)


# ---- the settings change what would be kept, as they change -------------------------


def test_changing_a_setting_recounts_without_reading_the_file_again(loaded, monkeypatch):
    def no_second_read(*a, **k):
        raise AssertionError("the file was read again")

    monkeypatch.setattr(prep, "summarise_dataset", no_second_read)
    s = loaded
    s.set_dedup_enabled(False)
    assert (s.kept_count, s.removed_duplicate) == (6, 0)
    s.set_drop_empty(False)
    assert s.kept_count == 7
    assert s.cleanup_summary == "All 7 examples are kept. These settings remove nothing."
    assert s.can_save_copy is False  # a copy would be the same file
    s.set_apply_curriculum(True)
    assert s.can_save_copy is True  # same examples, different order
    s.set_max_tokens(100)
    s.set_min_tokens(20)
    # "Hi" / "Yo" and the one with no answer are both under 20 tokens.
    assert (s.removed_long, s.removed_short, s.kept_count) == (1, 2, 4)
    assert s.cleanup_summary == (
        "4 of 7 examples are kept. Removed: 2 shorter than 20 tokens, 1 longer than 100 tokens."
    )


def test_counts_survive_a_server_restart(loaded):
    us._SUMMARY_CACHE.clear()  # what a restart does
    loaded.set_dedup_enabled(False)
    assert loaded.kept_count == 6


def test_the_summary_cache_stays_small(sandbox):
    us._SUMMARY_CACHE.clear()
    s = us.DatasetState()
    for i in range(us._SUMMARY_CACHE_MAX + 1):
        _upload(s, _jsonl(_ROWS[: i + 1]), f"f{i}.jsonl")
        s.upload_count = 0  # stay under the per-session upload cap
    assert len(us._SUMMARY_CACHE) == us._SUMMARY_CACHE_MAX
    assert not any(key[0].endswith("f0.jsonl") for key in us._SUMMARY_CACHE)


def test_reading_the_file_as_another_layout_reads_it_again(loaded):
    loaded.set_format_hint("jsonl")
    assert loaded.detected_format == "ChatML" and loaded.record_count == 7
    loaded.set_format_hint("auto")
    assert loaded.detected_format == "Alpaca"
    loaded.set_format_hint("parquet")  # not offered: ignored
    assert loaded.format_hint == "auto" and loaded.detected_format == "Alpaca"


def test_settings_before_any_upload_change_nothing_and_never_raise():
    s = us.DatasetState()
    s.set_dedup_enabled(False)
    s.set_min_tokens(10)
    s.set_format_hint("alpaca")
    assert (s.kept_count, s.record_count, s.detected_format) == (0, 0, "")
    assert s.cleanup_summary == "Upload a file to see what these settings would remove."
    assert (s.can_save_copy, s.training_file_name, s.training_file_note) == (False, "", "")


def test_a_file_that_went_away_leaves_empty_counts_not_an_error(loaded):
    Path(loaded._uploaded_path).unlink()
    loaded.set_dedup_enabled(False)
    assert (loaded.kept_count, loaded.removed_count) == (0, 0)
    loaded.set_format_hint("alpaca")
    assert "could not be read for a preview" in loaded.inspect_note


def test_limits_set_out_of_order_by_hand_do_not_raise(loaded):
    loaded.min_tokens, loaded.max_tokens = 50, 10  # the setters never allow this
    loaded._recount()
    assert loaded.kept_count == 0


def test_a_file_the_page_cannot_look_inside_still_uploads(sandbox):
    s = _upload(us.DatasetState(), b"question,answer\nhi,hello\n", "pairs.csv")
    assert "Only .jsonl and .json files" in s.inspect_note
    assert (s.record_count, s.preview_records, s.cleanup_summary) == (0, [], "")
    assert s.can_save_copy is False
    assert s.training_file_name == "pairs.csv"
    assert s.training_file_note == "The file as you uploaded it."
    s.set_dedup_enabled(False)  # a setting change keeps the note and the zero counts
    assert s.inspect_note and s.kept_count == 0


# ---- the cleaned copy -----------------------------------------------------------------


def test_save_writes_what_the_settings_keep_and_leaves_the_upload_alone(loaded, sandbox):
    upload = Path(loaded._uploaded_path)
    before = upload.read_bytes()
    loaded.save_cleaned_copy()
    copy = sandbox.out / "datasets" / "data-prepared.jsonl"
    assert loaded.prepare_error == "" and loaded._prepared_path == str(copy)
    kept = [json.loads(line) for line in copy.read_text(encoding="utf-8").splitlines()]
    assert len(kept) == loaded.prepared_count == loaded.kept_count == 5
    assert all(row in _ROWS for row in kept)
    assert upload.read_bytes() == before
    assert loaded.prepared_name == "data-prepared.jsonl"
    assert loaded.training_file_name == "data-prepared.jsonl"
    assert loaded.training_file_note == "The cleaned copy: 5 examples."


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.set_dedup_enabled(False),
        lambda s: s.set_drop_empty(False),
        lambda s: s.set_apply_curriculum(True),
        lambda s: s.set_min_tokens(5),
        lambda s: s.set_max_tokens(100),
        lambda s: s.set_format_hint("alpaca"),
    ],
)
def test_a_changed_setting_stops_offering_a_copy_that_no_longer_matches(loaded, sandbox, change):
    loaded.save_cleaned_copy()
    change(loaded)
    assert (loaded.prepared_name, loaded.prepared_count) == ("", 0)
    assert loaded.training_file_name == "data.jsonl"
    assert loaded.training_file_note == "The file as you uploaded it: 7 examples."
    assert (sandbox.out / "datasets" / "data-prepared.jsonl").exists()  # not deleted


def test_a_number_that_cannot_be_read_keeps_the_copy(loaded):
    loaded.save_cleaned_copy()
    loaded.set_max_tokens("junk")
    loaded.set_min_tokens("junk")
    assert loaded.prepared_name == "data-prepared.jsonl"
    assert loaded.max_tokens_error and loaded.min_tokens_error


def test_save_without_an_upload_says_what_to_do():
    s = us.DatasetState()
    s.save_cleaned_copy()
    assert s.prepare_error == "Upload a file first." and s.prepared_name == ""


def test_save_with_nothing_left_writes_nothing(loaded, sandbox):
    loaded.set_min_tokens(100_000)
    assert loaded.can_save_copy is False
    loaded.save_cleaned_copy()  # a direct call, past the disabled button
    assert loaded.prepare_error == "No examples would be left with these settings."
    assert loaded.prepared_name == "" and not (sandbox.out / "datasets").exists()


def test_a_failed_save_is_reported_without_paths(loaded, sandbox, monkeypatch):
    def boom(*a, **k):
        raise PermissionError(f"[WinError 5] Access is denied: '{sandbox.home}\\\\x'")

    monkeypatch.setattr(prep, "prepare_dataset", boom)
    loaded.save_cleaned_copy()
    assert loaded.prepare_error and loaded.prepared_name == ""
    assert str(sandbox.home) not in loaded.prepare_error


def test_a_later_failure_clears_an_earlier_copy(loaded, monkeypatch):
    loaded.save_cleaned_copy()
    monkeypatch.setattr(prep, "prepare_dataset", lambda *a, **k: (_ for _ in ()).throw(
        prep.DatasetPrepError("The file has no examples.")))
    loaded.save_cleaned_copy()
    assert loaded.prepare_error == "The file has no examples."
    assert loaded.prepared_name == ""


# ---- handing the file to a training form -----------------------------------------------


@pytest.fixture
def forms(monkeypatch):
    made = {us.TrainState: us.TrainState(), us.MultiRunState: us.MultiRunState()}
    asked = []

    async def get_state(self, cls):
        asked.append(cls)
        return made[cls]

    monkeypatch.setattr(us.DatasetState, "get_state", get_state)
    return SimpleNamespace(single=made[us.TrainState], multi=made[us.MultiRunState], asked=asked)


def _route(event) -> str:
    return {str(k): v for k, v in event.args}["path"]._var_value


def test_use_in_single_run_fills_the_form_and_goes_there(loaded, forms):
    event = asyncio.run(loaded.use_in_single_run())
    assert _route(event) == "/"
    assert forms.single.dataset_path == str(Path(loaded._uploaded_path).resolve())
    assert forms.single.dataset_path_error == ""
    assert forms.multi.dataset_path == ""  # only the form that was asked for


def test_use_in_multi_run_fills_the_other_form(loaded, forms):
    event = asyncio.run(loaded.use_in_multi_run())
    assert _route(event) == "/multi-run"
    assert forms.multi.dataset_path == str(Path(loaded._uploaded_path).resolve())
    assert forms.single.dataset_path == ""


def test_the_cleaned_copy_is_what_gets_used_once_it_exists(loaded, forms):
    loaded.save_cleaned_copy()
    asyncio.run(loaded.use_in_single_run())
    assert Path(forms.single.dataset_path).name == "data-prepared.jsonl"
    loaded.set_dedup_enabled(False)  # the copy no longer matches the settings
    asyncio.run(loaded.use_in_single_run())
    assert Path(forms.single.dataset_path).name == "data.jsonl"


def test_the_training_form_accepts_the_path_it_is_given(loaded, forms):
    # The same check a typed path goes through: inside the UI's own folder.
    loaded.save_cleaned_copy()
    asyncio.run(loaded.use_in_multi_run())
    value, error = us._validate_ui_path(forms.multi.dataset_path)
    assert error == "" and value == forms.multi.dataset_path


def test_use_without_an_upload_does_nothing(forms):
    s = us.DatasetState()
    assert asyncio.run(s.use_in_single_run()) is None
    assert asyncio.run(s.use_in_multi_run()) is None
    assert forms.asked == [] and forms.single.dataset_path == ""


# ---- the page --------------------------------------------------------------------------


def _walk(component):
    yield component
    for child in getattr(component, "children", []) or []:
        yield from _walk(child)


def test_the_page_offers_each_action_once():
    from backpropagate.ui_app.pages.dataset import dataset_page

    ids = [c.id for c in _walk(dataset_page()) if getattr(c, "id", None)]
    for wanted in ("bp-save-copy", "bp-use-single", "bp-use-multi", "bp-cleanup-summary",
                   "bp-training-file", "bp-dataset-preview"):
        assert ids.count(wanted) == 1, (wanted, ids)


def test_counts_read_naturally():
    assert us._count(1, "example") == "1 example"
    assert us._count(0, "repeat") == "0 repeats"
    assert us._count(12_345, "example") == "12,345 examples"
