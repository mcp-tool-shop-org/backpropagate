"""Shared helpers for the ``tests/test_cli_cov_*.py`` files (CLI coverage wave)."""

from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

from backpropagate import cli


def parse(argv: list[str]) -> Namespace:
    """Parse ``argv`` with the real parser and backfill the shared logging flags
    exactly the way ``cli.main`` does, so handlers see a complete namespace."""
    args = cli.create_parser().parse_args(argv)
    for dest, default in cli._COMMON_FLAG_DEFAULTS.items():
        if not hasattr(args, dest):
            setattr(args, dest, default)
    return args


def run(argv: list[str]) -> int:
    """Run the real ``cli.main`` in-process and return its exit code."""
    return cli.main(argv)


def seed_runs(output_dir, entries: list[dict]) -> None:
    """Persist ``entries`` through the real ``RunHistoryManager`` (no mocking)."""
    from backpropagate.checkpoints import RunHistoryManager

    manager = RunHistoryManager(str(output_dir))
    for entry in entries:
        assert manager.record_run(dict(entry)), f"record_run refused {entry!r}"


def last_json(text: str) -> Any:
    """Return the CLI JSON payload found in ``text``.

    Structured log lines can also render as JSON on stdout, so prefer the last
    top-level object carrying ``schema_version`` and fall back to the last object.
    """
    decoder = json.JSONDecoder()
    idx = 0
    objs: list[Any] = []
    while idx < len(text):
        brace = text.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        objs.append(obj)
        idx = end
    if not objs:
        raise AssertionError(f"no JSON object in: {text!r}")
    versioned = [o for o in objs if isinstance(o, dict) and "schema_version" in o]
    return (versioned or objs)[-1]
