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


def last_json(text: str) -> Any:
    """Return the last top-level JSON object found in ``text`` (skips log noise)."""
    decoder = json.JSONDecoder()
    idx, found = 0, None
    while idx < len(text):
        brace = text.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        found, idx = obj, end
    if found is None:
        raise AssertionError(f"no JSON object in: {text!r}")
    return found
