#!/usr/bin/env python3
"""Atheris harness: dataset *files* through DatasetLoader and StreamingDatasetLoader.

Target: ``DatasetLoader`` / ``StreamingDatasetLoader`` in
``backpropagate/datasets.py`` -- the file-parsing front door (``.jsonl``,
``.json``, ``.txt`` / ``.md`` and the unknown-extension fallback to JSONL). A
dataset file is the most common thing an operator did not author.

The fuzzer writes arbitrary bytes to a file with a fuzzer-chosen extension and
checks the loaders against an independent reference parser:

* only documented exceptions escape: ``BackpropagateError`` (the structured
  hierarchy) or ``ValueError`` (the loader's documented wrapper for anything
  unstructured, which includes ``UnicodeDecodeError``). Anything else is a
  finding;
* the loaded samples equal the reference parse, line for line (JSONL skips
  undecodable lines, raises ``DatasetParseError`` when every line fails, and
  fails the whole load on a non-``JSONDecodeError`` such as a 5000-digit int);
* the streaming loader yields exactly the rows the non-streaming loader holds;
* a successfully loaded dataset is self-consistent (length, detected format,
  ChatML conversion) and validates without crashing.

CSV and Parquet are not fuzzed: they delegate straight to pandas / pyarrow, so
the parser under test would be someone else's.

Run: ``python fuzz/fuzz_dataset_files.py -atheris_runs=50000 fuzz/corpus/dataset_files``
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

from fuzz_common import Provider, instrument, main, scratch_dir

with instrument(__name__ == "__main__"):
    from backpropagate.datasets import (
        DatasetFormat,
        DatasetLoader,
        StreamingDatasetLoader,
        detect_format,
    )
    from backpropagate.exceptions import BackpropagateError, DatasetParseError

SUFFIXES = (".jsonl", ".json", ".txt", ".md", ".dat")
PARSE_ERROR = "parse_error"  # -> DatasetParseError
WRAPPED = "wrapped"  # -> ValueError ("Failed to load dataset")
UNDECODABLE = "undecodable"  # -> ValueError (UnicodeDecodeError wrapped)


def reference(suffix: str, raw: bytes) -> list[Any] | str:
    """Independent re-implementation of the loaders' documented contract."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return UNDECODABLE
    # open(..., encoding="utf-8") uses universal newlines: \r\n and \r -> \n.
    stream = io.StringIO(text, newline=None)

    if suffix == ".json":
        try:
            data = json.loads(stream.read())
        except json.JSONDecodeError:
            return PARSE_ERROR
        except (ValueError, RecursionError):
            return WRAPPED
        return data if isinstance(data, list) else [data]

    if suffix in (".txt", ".md"):
        content = stream.read()
        if "\n\n" in content:
            return [s.strip() for s in content.split("\n\n") if s.strip()]
        return [content]

    # .jsonl and every unknown extension
    samples: list[Any] = []
    total = 0
    for line in stream:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        except (ValueError, RecursionError):
            return WRAPPED
    if total and not samples:
        return PARSE_ERROR
    return samples


def _try(fn: Any) -> tuple[Any, Exception | None]:
    try:
        return fn(), None
    except (BackpropagateError, ValueError, RecursionError) as exc:
        return None, exc


def _check_failure(expected: str, exc: Exception, streaming: bool) -> None:
    if expected == PARSE_ERROR:
        assert isinstance(exc, DatasetParseError), f"expected DatasetParseError, got {exc!r}"
    elif expected == WRAPPED:
        # The loader wraps whatever json raised into a plain ValueError. The
        # streaming loader leaves a non-recursion ValueError alone and turns a
        # RecursionError (JSON nested too deeply) into a DatasetParseError.
        if streaming and isinstance(exc, DatasetParseError):
            assert isinstance(exc.__cause__, RecursionError), f"unexpected {exc!r}"
            return
        assert isinstance(exc, ValueError), f"expected ValueError, got {exc!r}"
        assert not isinstance(exc, BackpropagateError)
    else:  # UNDECODABLE
        assert isinstance(exc, ValueError), f"expected ValueError, got {exc!r}"


def check_dataset_files(data: bytes) -> None:
    p = Provider(data)
    suffix = p.pick(SUFFIXES)
    raw = p.rest()
    path = Path(scratch_dir()) / f"input{suffix}"
    path.write_bytes(raw)

    expected = reference(suffix, raw)

    # -- DatasetLoader -------------------------------------------------------
    loader, exc = _try(lambda: DatasetLoader(path, validate=False))
    if isinstance(expected, str):
        assert exc is not None, f"expected {expected} but the file loaded"
        _check_failure(expected, exc, streaming=False)
        assert not isinstance(exc, RecursionError), "RecursionError escaped DatasetLoader"
    else:
        assert exc is None, f"valid dataset failed to load: {exc!r}"
        assert loader.samples == expected, "loaded samples differ from the reference parse"
        assert len(loader) == len(expected) == len(list(loader))
        assert isinstance(loader.detected_format, DatasetFormat)
        if expected:
            assert loader.detected_format == detect_format(expected)
        for item in loader.to_chatml():
            assert set(item) == {"text"} and isinstance(item["text"], str)

        # The default constructor also validates; that step must not crash.
        checked = DatasetLoader(path)
        assert checked.validation_result.total_rows == len(expected)

    # -- StreamingDatasetLoader ---------------------------------------------
    rows, exc = _try(lambda: list(StreamingDatasetLoader(str(path))))
    if isinstance(expected, str):
        assert exc is not None, f"streaming loaded a file the loader rejected ({expected})"
        _check_failure(expected, exc, streaming=True)
    else:
        assert exc is None, f"streaming failed on a file the loader accepts: {exc!r}"
        assert rows == expected, "streaming rows differ from the loader's rows"


def TestOneInput(data: bytes) -> None:
    check_dataset_files(data)


if __name__ == "__main__":
    sys.exit(main(TestOneInput))
