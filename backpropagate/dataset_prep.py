"""Inspect a dataset file and write a cleaned copy of it.

The web UI's Dataset page is built on this module: what a file contains (its
layout, how long the examples are, how many repeat or are empty), what the
clean-up settings would remove, and the cleaned copy itself. It is plain
Python with no UI dependency, so the same answers are available from a
script::

    from backpropagate.dataset_prep import PrepSettings, inspect_dataset, prepare_dataset

    report = inspect_dataset("data.jsonl", PrepSettings(max_tokens=2048))
    print(report.kept, "of", report.total, "would be kept")
    result = prepare_dataset("data.jsonl", "cleaned/", PrepSettings(max_tokens=2048))
    print(result.path)

What the settings mean:

* **dedup**: keep the first of every set of exact repeats. Two examples
  repeat when their conversation text is identical (extra fields such as an
  id are ignored); for preference and feedback data, when the whole record is.
* **drop_empty**: remove examples with no text, or with an empty question or
  answer.
* **min_tokens / max_tokens**: remove examples shorter or longer than this.
  ``max_tokens=0`` means no upper limit. Tokens are approximated as four
  characters each, the same figure ``datasets.get_dataset_stats`` reports.
* **curriculum**: order the kept examples from short to long. A multi-run
  takes its rounds from the file in order, so earlier rounds then get the
  shorter examples. A single run shuffles its examples, so there the order
  changes nothing.

The source file is never modified. The cleaned copy holds the kept records
exactly as they were, one JSON object per line.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .datasets import _CHATML_TURN_RE, DatasetFormat, FormatConverter, detect_format

__all__ = [
    "DatasetPrepError",
    "DatasetReport",
    "DatasetSummary",
    "PrepResult",
    "PrepSettings",
    "inspect_dataset",
    "load_records",
    "prepare_dataset",
    "summarise_dataset",
]

#: Examples shown by the preview, and how much of each.
PREVIEW_COUNT = 5
PREVIEW_CHARS = 600
#: The formats whose conversation text identifies an example. For the others
#: (preference pairs, feedback rows, unknown) the whole record does.
_CONVERSATION_FORMATS = frozenset(
    {DatasetFormat.SHAREGPT, DatasetFormat.ALPACA, DatasetFormat.OPENAI, DatasetFormat.CHATML}
)
_FORMAT_HINTS = {
    "sharegpt": DatasetFormat.SHAREGPT,
    "alpaca": DatasetFormat.ALPACA,
    "openai": DatasetFormat.OPENAI,
    "jsonl": DatasetFormat.CHATML,
}
_ROLE_NAMES = {"system": "System", "user": "User", "assistant": "Assistant"}


class DatasetPrepError(ValueError):
    """The file cannot be inspected or prepared. The message is safe to show."""


@dataclass(frozen=True)
class PrepSettings:
    """What the clean-up does. The defaults remove repeats and empty examples."""

    dedup: bool = True
    drop_empty: bool = True
    min_tokens: int = 0
    max_tokens: int = 0  # 0: no upper limit
    curriculum: bool = False
    format_hint: str = "auto"  # auto | sharegpt | alpaca | openai | jsonl


@dataclass
class DatasetReport:
    """What a file contains, and what the settings would keep."""

    total: int = 0  # examples read
    malformed: int = 0  # lines that were not a JSON object or string
    format: str = "unknown"
    avg_tokens: int = 0
    shortest_tokens: int = 0
    longest_tokens: int = 0
    duplicates: int = 0  # exact repeats in the file, whatever the settings
    kept: int = 0
    removed_empty: int = 0
    removed_short: int = 0
    removed_long: int = 0
    removed_duplicate: int = 0
    preview: list[dict[str, str]] = field(default_factory=list)

    @property
    def removed(self) -> int:
        return self.removed_empty + self.removed_short + self.removed_long + self.removed_duplicate


@dataclass
class PrepResult:
    """A cleaned copy that was written, and the report it was made from."""

    path: Path
    report: DatasetReport


@dataclass
class _Row:
    record: dict[str, Any] | str
    text: str  # what the trainer reads (ChatML), or the record's own text
    key: bytes  # what makes two rows the same example
    tokens: int
    empty: bool


# ---- reading ------------------------------------------------------------------------


def load_records(path: str | os.PathLike[str]) -> tuple[list[dict[str, Any] | str], int]:
    """The examples in a ``.jsonl`` or ``.json`` file, and how many lines of
    a ``.jsonl`` were skipped because they were not a JSON object or string.

    Raises :class:`DatasetPrepError` for another file type, a file that
    cannot be read, or a ``.json`` file that is not valid JSON.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in (".jsonl", ".json"):
        raise DatasetPrepError(
            f"Only .jsonl and .json files can be inspected here (this is a {suffix or 'file'})."
        )
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DatasetPrepError(f"The file could not be read ({type(exc).__name__}).") from exc
    except UnicodeDecodeError as exc:
        raise DatasetPrepError("The file is not UTF-8 text.") from exc

    records: list[dict[str, Any] | str] = []
    malformed = 0
    if suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if isinstance(value, (dict, str)):
                records.append(value)
            else:
                malformed += 1
        return records, malformed

    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise DatasetPrepError("The file is not valid JSON.") from exc
    for value in parsed if isinstance(parsed, list) else [parsed]:
        if isinstance(value, (dict, str)):
            records.append(value)
        else:
            malformed += 1
    return records, malformed


# ---- one example ----------------------------------------------------------------------


def _plain_text(value: Any) -> str:
    """Every string in a record, in a stable order."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_plain_text(value[k]) for k in sorted(value, key=str))
    if isinstance(value, (list, tuple)):
        return "\n".join(_plain_text(v) for v in value)
    return ""


def _row(record: dict[str, Any] | str, pinned: DatasetFormat | None) -> _Row:
    fmt = pinned if pinned is not None else detect_format(record)
    chatml = ""
    try:
        chatml = str(FormatConverter.to_chatml(record, fmt))
    except Exception:  # noqa: BLE001 - a row the converter cannot read keeps its own text
        chatml = ""
    plain = _plain_text(record)
    text = chatml if chatml.strip() else plain
    turns = _CHATML_TURN_RE.findall(chatml)
    empty = not plain.strip() or any(
        role in ("user", "assistant") and not body.strip() for role, body in turns
    )
    if fmt in _CONVERSATION_FORMATS and chatml.strip():
        key_source = chatml
    else:
        key_source = (
            record if isinstance(record, str) else json.dumps(record, sort_keys=True, default=str)
        )
    key = hashlib.sha256(key_source.encode("utf-8", "surrogatepass")).digest()[:16]
    return _Row(record=record, text=text, key=key, tokens=len(text) // 4, empty=empty)


def _readable(row: _Row) -> str:
    """An example as a person reads it: "User: ..." / "Assistant: ..." lines."""
    turns = _CHATML_TURN_RE.findall(row.text)
    if turns:
        shown = "\n".join(
            f"{_ROLE_NAMES.get(role, role.capitalize())}: {body.strip()}" for role, body in turns
        )
    else:
        shown = row.text.strip()
    if len(shown) > PREVIEW_CHARS:
        shown = shown[:PREVIEW_CHARS].rstrip() + " …"
    return shown


def _pinned_format(hint: str) -> DatasetFormat | None:
    name = (hint or "auto").strip().lower()
    if name == "auto":
        return None
    if name not in _FORMAT_HINTS:
        raise DatasetPrepError(
            f"Unknown format {hint!r}. Use auto, sharegpt, alpaca, openai or jsonl."
        )
    return _FORMAT_HINTS[name]


# ---- the whole file -------------------------------------------------------------------


@dataclass
class DatasetSummary:
    """A file read once: enough about each example to answer for any settings.

    Changing a setting then costs a pass over three small lists, not another
    read of the file, which is what lets the Dataset page update its counts as
    the settings change. It holds no example text beyond the preview.
    """

    malformed: int = 0
    format: str = "unknown"
    tokens: list[int] = field(default_factory=list)
    empty: list[bool] = field(default_factory=list)
    keys: list[bytes] = field(default_factory=list)
    preview: list[dict[str, str]] = field(default_factory=list)

    def select(self, settings: PrepSettings | None = None) -> tuple[DatasetReport, list[int]]:
        """What ``settings`` would keep: the report, and the positions of the
        kept examples in the order they would be written."""
        settings = settings or PrepSettings()
        if settings.min_tokens < 0 or settings.max_tokens < 0:
            raise DatasetPrepError("Token limits cannot be negative.")
        if settings.max_tokens and settings.max_tokens < settings.min_tokens:
            raise DatasetPrepError("The maximum length is below the minimum length.")

        report = DatasetReport(
            total=len(self.tokens),
            malformed=self.malformed,
            format=self.format,
            preview=[dict(row) for row in self.preview],
        )
        if self.tokens:
            report.avg_tokens = int(round(sum(self.tokens) / len(self.tokens)))
            report.shortest_tokens, report.longest_tokens = min(self.tokens), max(self.tokens)
            report.duplicates = len(self.keys) - len(set(self.keys))

        kept: list[int] = []
        seen: set[bytes] = set()
        for index, (tokens, empty, key) in enumerate(zip(self.tokens, self.empty, self.keys)):
            if settings.drop_empty and empty:
                report.removed_empty += 1
            elif tokens < settings.min_tokens:
                report.removed_short += 1
            elif settings.max_tokens and tokens > settings.max_tokens:
                report.removed_long += 1
            elif settings.dedup and key in seen:
                report.removed_duplicate += 1
            else:
                seen.add(key)
                kept.append(index)
        if settings.curriculum:
            kept.sort(key=lambda index: self.tokens[index])  # stable: ties keep file order
        report.kept = len(kept)
        return report, kept

    def report(self, settings: PrepSettings | None = None) -> DatasetReport:
        return self.select(settings)[0]


def _summarise(
    records: list[dict[str, Any] | str], malformed: int, format_hint: str
) -> DatasetSummary:
    pinned = _pinned_format(format_hint)
    rows = [_row(record, pinned) for record in records]
    if pinned is not None:
        name = pinned.value
    elif records:
        name = detect_format(records).value
    else:
        name = "unknown"
    return DatasetSummary(
        malformed=malformed,
        format=name,
        tokens=[row.tokens for row in rows],
        empty=[row.empty for row in rows],
        keys=[row.key for row in rows],
        preview=[
            {"number": str(index + 1), "tokens": str(row.tokens), "text": _readable(row)}
            for index, row in enumerate(rows[:PREVIEW_COUNT])
        ],
    )


def summarise_dataset(path: str | os.PathLike[str], format_hint: str = "auto") -> DatasetSummary:
    """Read ``path`` once. ``format_hint`` pins the layout the examples are
    read as (``auto`` recognises it from the file)."""
    records, malformed = load_records(path)
    return _summarise(records, malformed, format_hint)


def inspect_dataset(
    path: str | os.PathLike[str], settings: PrepSettings | None = None
) -> DatasetReport:
    """What ``path`` contains and what ``settings`` would keep. Reads only."""
    settings = settings or PrepSettings()
    return summarise_dataset(path, settings.format_hint).report(settings)


def prepare_dataset(
    path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    settings: PrepSettings | None = None,
) -> PrepResult:
    """Write the examples ``settings`` keeps to ``<out_dir>/<name>-prepared.jsonl``.

    The records are written as they were, one JSON object per line, so the
    trainer reads the copy exactly as it would have read the original. The
    copy replaces an earlier one of the same name in one step (a temporary
    file, then a rename). Raises :class:`DatasetPrepError` when nothing would
    be left.
    """
    settings = settings or PrepSettings()
    source = Path(path)
    records, malformed = load_records(source)
    report, kept = _summarise(records, malformed, settings.format_hint).select(settings)
    if not kept:
        raise DatasetPrepError(
            "No examples would be left with these settings."
            if records
            else "The file has no examples."
        )
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.stem}-prepared.jsonl"
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for index in kept:
                handle.write(json.dumps(records[index], ensure_ascii=False) + "\n")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return PrepResult(path=target, report=report)
