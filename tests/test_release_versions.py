"""The release version surfaces must agree with each other.

backpropagate ships from TWO manifests: `pyproject.toml` (the PyPI package) and
`package.json` (the `@mcptoolshop/backpropagate` npm launcher). Nothing in the
normal test run reads both, so a bump that touches one and not the other is
invisible until `release.yml`'s "Verify tag matches package.json version" step
runs -- which happens AFTER the tag has been pushed, i.e. after the release is
already cut and the only remedy is to delete a published tag.

Earned 2026-09-07. v1.7.1 bumped `pyproject.toml` to 1.7.1 and left
`package.json` at 1.7.0. CI was green, review was clean, shipcheck audit passed
100%, and the drift surfaced only when the tag was pushed and Release refused:

    Tag: v1.7.1 -> 1.7.1
    Package: 1.7.0
    ##[error]Tag 1.7.1 does not match package.json version 1.7.0

This test moves that check from release time to every push, which is the only
difference that matters: a red test costs a commit, a refused release costs a
retracted tag.
"""

from __future__ import annotations

import json
from pathlib import Path

# tomllib is 3.11+ stdlib; fall back to tomli on 3.10, matching the idiom
# scripts/check_doc_drift.py already uses.
try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - only taken on 3.10
    import tomli as tomllib  # type: ignore[no-redef,import-not-found]

REPO = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    with open(REPO / "pyproject.toml", "rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


def _package_json_version() -> str:
    with open(REPO / "package.json", encoding="utf-8") as fh:
        return str(json.load(fh)["version"])


def test_pyproject_and_package_json_versions_agree() -> None:
    """The PyPI package and the npm launcher ship as one release, under one number."""
    py = _pyproject_version()
    npm = _package_json_version()
    assert py == npm, (
        f"pyproject.toml is {py} and package.json is {npm}. These are two halves of "
        f"one release: release.yml verifies the git tag against package.json, so a "
        f"bump that moves only pyproject.toml refuses at tag time, after the tag is "
        f"already pushed. Bump both, or neither."
    )


def test_both_versions_are_plain_release_numbers() -> None:
    """No `v` prefix, no local segment -- the tag is built as `v` + this string.

    A `v1.7.1` in either manifest would produce a `vv1.7.1` tag comparison and fail
    a release for a reason nobody would look for in a version field.
    """
    for name, value in (("pyproject.toml", _pyproject_version()),
                        ("package.json", _package_json_version())):
        assert not value.startswith("v"), f"{name} version {value!r} carries a 'v' prefix"
        assert value.strip() == value, f"{name} version {value!r} has surrounding whitespace"
