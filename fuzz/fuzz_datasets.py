#!/usr/bin/env python3
"""Atheris harness: dataset row detection, validation, conversion and filtering.

Target: ``backpropagate/datasets.py`` -- the code that decides what a training
row *is* (JSONL / ShareGPT / Alpaca / OpenAI / preference / KTO / ChatML / raw
text) and turns it into the ChatML the trainer consumes. Every row comes from a
file or Hub dataset the operator did not write, so this is the widest
untrusted-input surface in the library.

Two modes, chosen by the first input byte:

1. **Hostile rows.** The rest of the input is parsed as a JSON document (or kept
   as raw text if it is not JSON) and pushed through ``detect_format``,
   ``validate_dataset``, ``convert_to_chatml``, ``deduplicate_exact`` and
   ``filter_by_quality``. Properties: only documented behaviour escapes (no
   stray ``TypeError`` / ``AttributeError``), detection is deterministic and
   wrapper-stable, the validation result is internally consistent, batch
   conversion equals per-row conversion, an UNKNOWN row is always dropped, every
   converted row is a ``{"text": str}`` ChatML string, and the filters keep
   their accounting (kept + removed == total, kept is an ordered subset).
2. **Well-typed rows.** A conversation is *built* in one of the supported shapes
   and must round-trip: detected as that format, validated clean, and converted
   to ChatML whose parsed turns equal the roles and contents it was built from.

Run: ``python fuzz/fuzz_datasets.py -atheris_runs=200000 fuzz/corpus/datasets``
"""

from __future__ import annotations

import json
import sys
from typing import Any

from fuzz_common import Provider, instrument, main

with instrument(__name__ == "__main__"):
    from backpropagate.datasets import (
        _CHATML_TURN_RE,
        DatasetFormat,
        FormatConverter,
        _get_ngrams,
        convert_to_chatml,
        deduplicate_exact,
        detect_format,
        filter_by_quality,
        split_dataset,
        validate_dataset,
    )
    from backpropagate.exceptions import InvalidSettingError

ROLE_MAP = FormatConverter.ROLE_MAP_SHAREGPT
OPENAI_ROLES = ("system", "user", "assistant", "function", "tool")


def decode_rows(raw: bytes) -> list[Any]:
    """Parse ``raw`` as JSON if possible, else treat it as one raw-text row."""
    text = raw.decode("utf-8", errors="replace")
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        value = text
    return value if isinstance(value, list) else [value]


# --------------------------------------------------------------------------
# Mode 1: hostile rows
# --------------------------------------------------------------------------


def check_rows(rows: list[Any]) -> None:
    fmts = []
    for row in rows:
        fmt = detect_format(row)
        assert isinstance(fmt, DatasetFormat)
        assert detect_format(row) == fmt, "detect_format is not deterministic"
        if not isinstance(row, list):  # a list row is itself unwrapped one level
            assert detect_format([row]) == fmt, "list wrapper changed the detected format"
        fmts.append(fmt)

    # -- validation: must report problems, never crash on them --------------
    result = validate_dataset(rows)
    assert result.total_rows == len(rows)
    assert 0 <= result.valid_rows <= result.total_rows
    assert isinstance(result.format_detected, DatasetFormat)
    assert result.is_valid == (len(result.errors) == 0)
    if rows:
        assert result.format_detected == fmts[0]
    for err in [*result.errors, *result.warnings]:
        assert isinstance(err.error_type, str)
        assert 0 <= err.row_index < max(len(rows), 1), "error points outside the dataset"

    # -- conversion: never raises, drops what it cannot convert --------------
    converted = convert_to_chatml(rows)
    assert isinstance(converted, list) and len(converted) <= len(rows)
    for item in converted:
        assert set(item) == {"text"} and isinstance(item["text"], str)

    per_row: list[dict[str, str]] = []
    for row, fmt in zip(rows, fmts):
        single = convert_to_chatml([row])
        assert len(single) <= 1
        per_row.extend(single)
        if fmt == DatasetFormat.UNKNOWN:
            assert single == [], "an UNKNOWN row must be dropped, not converted"
        elif single and fmt != DatasetFormat.CHATML:
            text = single[0]["text"]
            assert text == "" or text.startswith("<|im_start|>"), text[:60]
    assert converted == per_row, "batch conversion differs from per-row conversion"

    # -- dedupe / quality filter accounting ---------------------------------
    unique, removed = deduplicate_exact(list(converted))
    assert len(unique) + removed == len(converted)
    texts = [u["text"] for u in unique]
    assert len(set(texts)) == len(texts), "deduplicate_exact left a duplicate"
    assert texts == list(dict.fromkeys(c["text"] for c in converted)), "order or content lost"
    assert deduplicate_exact(list(unique))[1] == 0, "deduplicate_exact is not idempotent"

    kept, stats = filter_by_quality(
        list(converted),
        min_tokens=0,
        max_tokens=10**9,
        min_turns=0,
        remove_empty=False,
        require_assistant=False,
    )
    assert kept == converted, "a no-op filter changed the rows"
    assert stats.total_before == len(converted) and stats.total_after == len(kept)

    kept, stats = filter_by_quality(list(converted))
    assert stats.total_before == len(converted) and stats.total_after == len(kept)
    assert stats.total_removed == len(converted) - len(kept)
    removed_buckets = (
        stats.removed_empty
        + stats.removed_too_short
        + stats.removed_too_long
        + stats.removed_few_turns
        + stats.removed_many_turns
        + stats.removed_no_assistant
        + stats.removed_custom
    )
    assert removed_buckets == stats.total_removed, "filter accounting does not add up"
    it = iter(converted)
    assert all(any(k is c for c in it) for k in kept), "kept rows are not an ordered subset"


def check_ngrams_and_split(p: Provider) -> None:
    text = p.text(30)
    grams = _get_ngrams(text, 3)
    text = text.lower()  # _get_ngrams lowercases, which can change the length
    if not text:
        assert grams == []
    else:
        assert len(grams) == max(len(text) - 2, 1)
        assert all(1 <= len(g) <= 3 for g in grams)

    records = [{"i": i} for i in range(p.int_in_range(0, 12))]
    ratio = p.int_in_range(0, 1000) / 1000
    seed = p.int_in_range(0, 2**16)
    snapshot = list(records)
    try:
        train, held = split_dataset(records, ratio, seed)
    except InvalidSettingError:
        assert len(records) < 2 or not (0.0 < ratio < 1.0)
        return
    assert records == snapshot, "split_dataset mutated its input"
    assert train and held, "a split came back empty"
    assert sorted(train + held, key=lambda r: r["i"]) == records, "rows lost or duplicated"
    assert split_dataset(records, ratio, seed) == (train, held), "split is not deterministic"


# --------------------------------------------------------------------------
# Mode 2: well-typed rows must round-trip
# --------------------------------------------------------------------------


def _content(p: Provider) -> str:
    # Real content may not contain ChatML control tokens (that is injection, not
    # round-tripping), but may contain any other text, newlines included.
    return p.text(24).replace("<|", "< |")


def _parse(chatml: str) -> list[tuple[str, str]]:
    return _CHATML_TURN_RE.findall(chatml)


def check_roundtrip(p: Provider) -> None:
    kind = p.pick(("sharegpt", "openai", "alpaca", "preference", "kto"))
    turns = [_content(p) for _ in range(p.int_in_range(1, 4))]

    if kind == "sharegpt":
        names = ("human", "user", "gpt", "assistant", "system", "Human", "GPT")
        convo = [{"from": p.pick(names), "value": c} for c in turns]
        row: dict[str, Any] = {"conversations": convo}
        expected = [(ROLE_MAP[t["from"].lower()], t["value"]) for t in convo]
        fmt = DatasetFormat.SHAREGPT
    elif kind == "openai":
        msgs = [{"role": p.pick(OPENAI_ROLES), "content": c} for c in turns]
        row = {"messages": msgs}
        expected = [(m["role"], m["content"]) for m in msgs]
        fmt = DatasetFormat.OPENAI
    elif kind == "alpaca":
        instruction = turns[0]
        output = _content(p)
        row = {"instruction": instruction, "output": output}
        parts = [instruction]
        if p.bool():
            row["input"] = _content(p)
            if row["input"]:
                parts.append(row["input"])
        expected = []
        if p.bool():
            row["system"] = _content(p)
            if row["system"]:
                expected.append(("system", row["system"]))
        expected += [("user", "\n\n".join(parts)), ("assistant", output)]
        fmt = DatasetFormat.ALPACA
    elif kind == "preference":
        prompt, chosen = _content(p), _content(p)
        row = {"chosen": chosen, "rejected": _content(p)}
        expected = []
        if p.bool():
            row["prompt"] = prompt
            if prompt.strip():
                expected.append(("user", prompt))
        expected.append(("assistant", chosen))
        fmt = DatasetFormat.PREFERENCE
    else:
        row = {"prompt": _content(p), "completion": _content(p), "label": p.bool()}
        expected = []
        fmt = DatasetFormat.KTO

    assert detect_format(row) == fmt, f"{kind} row detected as {detect_format(row)}"
    result = validate_dataset([row])
    assert result.is_valid, [str(e) for e in result.errors]

    if fmt == DatasetFormat.KTO:
        return  # KTO rows are kept raw for TRL; they have no ChatML rendering.
    converted = convert_to_chatml([row])
    assert len(converted) == 1, f"{kind} row was dropped"
    assert _parse(converted[0]["text"]) == expected, f"{kind} did not round-trip"


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def check_datasets(data: bytes) -> None:
    p = Provider(data)
    mode = p.int_in_range(0, 3)
    if mode == 0:
        check_roundtrip(p)
    elif mode == 1:
        check_ngrams_and_split(p)
    else:
        check_rows(decode_rows(p.rest()))


def TestOneInput(data: bytes) -> None:
    check_datasets(data)


if __name__ == "__main__":
    sys.exit(main(TestOneInput))
