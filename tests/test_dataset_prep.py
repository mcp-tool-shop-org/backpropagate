# Inspect a dataset file and write a cleaned copy (backpropagate/dataset_prep.py).
"""What the web UI's Dataset page is built on.

The page used to store its clean-up settings and never apply them. These pin
what each setting does now, on the layouts the trainer reads, and that the
original file is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backpropagate import dataset_prep as dp
from backpropagate.dataset_prep import (
    DatasetPrepError,
    PrepSettings,
    inspect_dataset,
    load_records,
    prepare_dataset,
    summarise_dataset,
)


def _alpaca(instruction: str, output: str, **extra) -> dict:
    return {"instruction": instruction, "input": "", "output": output, **extra}


def _sharegpt(question: str, answer: str) -> dict:
    return {"conversations": [{"from": "human", "value": question},
                              {"from": "gpt", "value": answer}]}


def _openai(question: str, answer: str) -> dict:
    return {"messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]}


def _write(path: Path, rows: list, *, raw_lines: list[str] | None = None) -> Path:
    lines = [json.dumps(row) for row in rows] + list(raw_lines or [])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _kept(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def data(tmp_path):
    """Seven Alpaca examples: one repeat, one with an empty answer, one long."""
    rows = [
        _alpaca("What is the capital of France?", "Paris."),
        _alpaca("Name a prime number.", "Seven is a prime number."),
        _alpaca("What is the capital of France?", "Paris.", id="again"),  # same conversation
        _alpaca("Say nothing.", ""),  # empty answer
        _alpaca("Tell a long story.", "word " * 400),  # about 500 tokens
        _alpaca("Hi", "Yo"),  # very short
        _alpaca("What is two plus two?", "Four."),
    ]
    return _write(tmp_path / "data.jsonl", rows)


# ---- reading ------------------------------------------------------------------------


def test_jsonl_lines_that_are_not_examples_are_counted_and_skipped(tmp_path):
    path = _write(tmp_path / "d.jsonl", [_alpaca("a", "b"), "a plain string row"],
                  raw_lines=["{not json", "42", "", "[1, 2]"])
    records, malformed = load_records(path)
    assert len(records) == 2 and malformed == 3


def test_a_json_file_holds_a_list_or_one_example(tmp_path):
    many = tmp_path / "many.json"
    many.write_text(json.dumps([_alpaca("a", "b"), 7, _alpaca("c", "d")]), encoding="utf-8")
    assert [len(x) for x in load_records(many)[:1]] == [2]
    assert load_records(many)[1] == 1
    one = tmp_path / "one.json"
    one.write_text(json.dumps(_alpaca("a", "b")), encoding="utf-8")
    assert load_records(one) == ([_alpaca("a", "b")], 0)


def test_a_byte_order_mark_does_not_break_the_first_line(tmp_path):
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(_alpaca("a", "b")).encode() + b"\n")
    assert load_records(path) == ([_alpaca("a", "b")], 0)


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("data.csv", b"a,b\n1,2\n", "Only .jsonl and .json files"),
        ("data", b"{}", "Only .jsonl and .json files"),
        ("broken.json", b"{not json", "not valid JSON"),
        ("latin.jsonl", b'{"instruction": "caf\xe9"}\n', "not UTF-8"),
    ],
)
def test_files_that_cannot_be_inspected_say_why(tmp_path, name, content, message):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(DatasetPrepError, match=message):
        load_records(path)


def test_a_missing_file_says_so_without_the_path(tmp_path):
    with pytest.raises(DatasetPrepError) as info:
        load_records(tmp_path / "secret-folder" / "gone.jsonl")
    assert "could not be read" in str(info.value)
    assert "secret-folder" not in str(info.value)


# ---- what a file contains -------------------------------------------------------------


def test_the_report_describes_the_file(data):
    report = inspect_dataset(data, PrepSettings(dedup=False, drop_empty=False))
    assert (report.total, report.malformed, report.format) == (7, 0, "alpaca")
    assert report.duplicates == 1  # counted whatever the settings
    assert report.kept == 7 and report.removed == 0
    assert report.shortest_tokens < report.avg_tokens < report.longest_tokens
    assert report.longest_tokens > 450


def test_the_average_matches_the_trainer_side_statistics(data):
    from backpropagate.datasets import get_dataset_stats

    records, _ = load_records(data)
    stats = get_dataset_stats(records)
    report = inspect_dataset(data)
    assert report.avg_tokens == int(round(stats.avg_tokens_per_sample))
    assert (report.shortest_tokens, report.longest_tokens) == (stats.min_tokens, stats.max_tokens)


def test_the_preview_reads_like_a_conversation(data):
    preview = inspect_dataset(data).preview
    assert len(preview) == dp.PREVIEW_COUNT
    assert preview[0]["number"] == "1" and int(preview[0]["tokens"]) > 0
    assert preview[0]["text"] == "User: What is the capital of France?\nAssistant: Paris."
    long_one = preview[4]["text"]
    assert len(long_one) <= dp.PREVIEW_CHARS + 2 and long_one.endswith(" …")


@pytest.mark.parametrize(
    ("row", "name", "first_line"),
    [
        (_sharegpt("Why is the sky blue?", "Scattering."), "sharegpt", "User: Why is the sky blue?"),
        (_openai("Why is the sky blue?", "Scattering."), "openai", "User: Why is the sky blue?"),
        (_alpaca("Why is the sky blue?", "Scattering."), "alpaca", "User: Why is the sky blue?"),
    ],
)
def test_each_layout_is_recognised(tmp_path, row, name, first_line):
    report = inspect_dataset(_write(tmp_path / "d.jsonl", [row, row]))
    assert report.format == name
    assert report.preview[0]["text"].splitlines()[0] == first_line
    assert report.duplicates == 1


def test_an_empty_file_is_an_empty_report(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    report = inspect_dataset(path)
    assert (report.total, report.kept, report.avg_tokens, report.preview) == (0, 0, 0, [])
    assert report.format == "unknown"


def test_a_plain_text_row_is_shown_the_way_the_trainer_reads_it(tmp_path):
    # The trainer wraps a bare string as one user message, so the preview says so.
    report = inspect_dataset(_write(tmp_path / "d.jsonl", ["Once upon a time there was a model."]))
    assert report.preview[0]["text"] == "User: Once upon a time there was a model."
    assert (report.format, report.kept) == ("raw_text", 1)


# ---- what each setting removes --------------------------------------------------------


def test_the_defaults_remove_repeats_and_empty_examples(data):
    report = inspect_dataset(data)
    assert (report.removed_duplicate, report.removed_empty) == (1, 1)
    assert (report.removed_short, report.removed_long) == (0, 0)
    assert report.kept == 5 and report.removed == 2


def test_each_removed_example_is_counted_once(data):
    report = inspect_dataset(data, PrepSettings(min_tokens=20, max_tokens=100))
    assert report.kept + report.removed == report.total
    assert report.removed_long == 1  # the long story
    assert report.removed_short == 1  # "Hi" / "Yo": 16 tokens with the chat markers


def test_no_upper_limit_keeps_long_examples(data):
    assert inspect_dataset(data, PrepSettings(max_tokens=0)).removed_long == 0
    assert inspect_dataset(data, PrepSettings(max_tokens=100)).removed_long == 1


def test_a_repeat_differs_only_in_fields_that_are_not_the_conversation(tmp_path):
    rows = [_sharegpt("Q", "A"), {**_sharegpt("Q", "A"), "id": 7, "source": "scrape"}]
    assert inspect_dataset(_write(tmp_path / "d.jsonl", rows)).removed_duplicate == 1


def test_preference_pairs_with_different_rejected_answers_are_not_repeats(tmp_path):
    rows = [
        {"prompt": "Explain rain.", "chosen": "Water falls.", "rejected": "No idea."},
        {"prompt": "Explain rain.", "chosen": "Water falls.", "rejected": "Rain is dry."},
        {"prompt": "Explain rain.", "chosen": "Water falls.", "rejected": "No idea."},
    ]
    report = inspect_dataset(_write(tmp_path / "d.jsonl", rows))
    assert report.format == "preference"
    assert (report.duplicates, report.removed_duplicate, report.kept) == (1, 1, 2)


def test_feedback_rows_are_kept_not_treated_as_empty(tmp_path):
    rows = [
        {"prompt": "Is water wet?", "completion": "Yes.", "label": True},
        {"prompt": "Is fire cold?", "completion": "Yes.", "label": False},
    ]
    report = inspect_dataset(_write(tmp_path / "d.jsonl", rows))
    assert (report.kept, report.removed_empty) == (2, 0)


@pytest.mark.parametrize(
    "row",
    [
        {"instruction": "", "input": "", "output": ""},
        {"instruction": "   ", "output": "\n"},
        _alpaca("A question with no answer", ""),
        _sharegpt("", "An answer with no question"),
        "   ",
    ],
)
def test_what_counts_as_empty(tmp_path, row):
    path = _write(tmp_path / "d.jsonl", [row, _alpaca("Real question?", "Real answer.")])
    assert inspect_dataset(path).removed_empty == 1
    assert inspect_dataset(path, PrepSettings(drop_empty=False)).removed_empty == 0


def test_an_optional_empty_input_is_not_an_empty_example(tmp_path):
    path = _write(tmp_path / "d.jsonl", [_alpaca("Translate hello to French.", "Bonjour.")])
    assert inspect_dataset(path).removed_empty == 0


def test_the_layout_can_be_set_by_hand(tmp_path):
    path = _write(tmp_path / "d.jsonl", [_alpaca("Q?", "A.")])
    assert inspect_dataset(path, PrepSettings(format_hint="alpaca")).format == "alpaca"
    assert inspect_dataset(path, PrepSettings(format_hint="JSONL")).format == "chatml"
    with pytest.raises(DatasetPrepError, match="Unknown format"):
        inspect_dataset(path, PrepSettings(format_hint="parquet"))


def test_a_layout_set_wrongly_keeps_each_row_readable(tmp_path):
    # ShareGPT rows read as Alpaca: the converter finds no question or answer,
    # so the row falls back to its own text and is never silently blank.
    path = _write(tmp_path / "d.jsonl", [_sharegpt("Why?", "Because.")])
    report = inspect_dataset(path, PrepSettings(format_hint="alpaca", drop_empty=False))
    assert report.kept == 1 and report.preview[0]["text"].strip()


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (PrepSettings(min_tokens=-1), "cannot be negative"),
        (PrepSettings(max_tokens=-5), "cannot be negative"),
        (PrepSettings(min_tokens=50, max_tokens=10), "below the minimum"),
    ],
)
def test_limits_that_make_no_sense_are_refused(data, settings, message):
    with pytest.raises(DatasetPrepError, match=message):
        inspect_dataset(data, settings)


def test_one_read_answers_for_any_settings(data, monkeypatch):
    summary = summarise_dataset(data)
    monkeypatch.setattr(dp, "load_records", None)  # no second read
    assert summary.report().kept == 5
    assert summary.report(PrepSettings(dedup=False, drop_empty=False)).kept == 7
    report, kept = summary.select(PrepSettings(max_tokens=100, curriculum=True))
    assert report.kept == len(kept) == 4
    assert [summary.tokens[i] for i in kept] == sorted(summary.tokens[i] for i in kept)
    report.preview[0]["text"] = "changed"  # a report never edits the summary
    assert summary.preview[0]["text"].startswith("User:")


# ---- the cleaned copy -----------------------------------------------------------------


def test_the_copy_holds_the_kept_examples_unchanged(data, tmp_path):
    before = data.read_bytes()
    result = prepare_dataset(data, tmp_path / "datasets")
    assert result.path == tmp_path / "datasets" / "data-prepared.jsonl"
    kept = _kept(result.path)
    assert len(kept) == result.report.kept == 5
    originals, _ = load_records(data)
    assert all(row in originals for row in kept)
    assert kept[0] == originals[0]  # original order
    assert data.read_bytes() == before  # the source is never modified


def test_the_copy_is_read_by_the_trainer_like_the_original(data, tmp_path):
    from backpropagate.datasets import DatasetFormat, _detect_format_from_file

    result = prepare_dataset(data, tmp_path / "out")
    assert _detect_format_from_file(result.path) == DatasetFormat.ALPACA
    assert inspect_dataset(result.path, PrepSettings()).removed == 0  # already clean


def test_short_to_long_orders_the_copy(data, tmp_path):
    result = prepare_dataset(data, tmp_path / "out", PrepSettings(curriculum=True))
    lengths = [len(json.dumps(row)) for row in _kept(result.path)]
    assert lengths == sorted(lengths)
    plain = prepare_dataset(data, tmp_path / "plain")
    assert sorted(map(json.dumps, _kept(plain.path))) == sorted(map(json.dumps, _kept(result.path)))


def test_text_outside_ascii_survives(tmp_path):
    row = _alpaca("Traduis « bonjour » en japonais.", "こんにちは")
    result = prepare_dataset(_write(tmp_path / "d.jsonl", [row]), tmp_path / "out")
    assert "こんにちは" in result.path.read_text(encoding="utf-8")
    assert _kept(result.path) == [row]


def test_a_second_copy_replaces_the_first_and_leaves_no_temp_file(data, tmp_path):
    out = tmp_path / "out"
    first = prepare_dataset(data, out)
    second = prepare_dataset(data, out, PrepSettings(max_tokens=100))
    assert first.path == second.path
    assert len(_kept(second.path)) == second.report.kept == 4
    assert [p.name for p in out.iterdir()] == ["data-prepared.jsonl"]


def test_nothing_left_writes_nothing(data, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(DatasetPrepError, match="No examples would be left"):
        prepare_dataset(data, out, PrepSettings(min_tokens=10_000))
    assert not out.exists() or list(out.iterdir()) == []
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DatasetPrepError, match="has no examples"):
        prepare_dataset(empty, out)


def test_a_failed_write_leaves_no_temp_file(data, tmp_path, monkeypatch):
    out = tmp_path / "out"

    def refuse(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(dp.os, "replace", refuse)
    with pytest.raises(PermissionError):
        prepare_dataset(data, out)
    assert list(out.iterdir()) == []
