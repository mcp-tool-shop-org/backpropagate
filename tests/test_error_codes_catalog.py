"""Mechanical regression test for the ERROR_CODES catalog (TESTS-F-002).

Background — CHANGELOG v1.2.0 "Known issues / tech debt":
    "8 emitted error codes are documented in cli.py:_BRIDGE_LOCAL_ERROR_CODES
    instead of exceptions.ERROR_CODES — the 4 HUB_PUSH_* codes plus
    SLAO_MERGE_DIVERGED, PEFT_API_INCOMPATIBLE, UI_OUTPUT_DIR_FORBIDDEN,
    and INPUT_AUTH_INVALID_SHAPE."

Stage B caught this drift by HAND across two waves; Wave 3.5 also caught it
by hand. Wave 6 mechanises it. The test walks ``backpropagate/**/*.py`` with
``ast.parse`` and finds every string literal passed as ``code=...`` to a
function call, then asserts the set is a subset of
``exceptions.ERROR_CODES.keys()`` (plus a tight allowlist of currently-known
drift that v1.3 will close).

A regression that emits a new ``code='SOMETHING_NEW'`` without adding the
key to ERROR_CODES will fail this test immediately.

The reverse direction — catalog entries that are never emitted — is also
checked (advisory: warn-only via the "stale_codes" data, not assert).
Stale codes are not a correctness bug; they are documentation debt.

See ``backpropagate/exceptions.py:115-128`` for the catalog discipline
("Adding a new exception subclass? Add a row here FIRST so the discipline
of 'central catalog → raise site' stays intact.")
"""

from __future__ import annotations

import ast
import pathlib

from backpropagate.exceptions import ERROR_CODES

# =============================================================================
# KNOWN-DRIFT ALLOWLIST (v1.2.0 CHANGELOG "Known issues / tech debt")
# =============================================================================
# Codes currently emitted from backpropagate/*.py that are NOT in
# exceptions.ERROR_CODES. The audit chain (Stage B → Wave 3.5 → Wave 5)
# flagged these as v1.3-promotion candidates. Each entry MUST cite the
# emitting file so it's easy to find and promote.
#
# When v1.3 promotes a code into ERROR_CODES, remove it from this list.
# The test will fail if a new (unlisted) code drifts in.
#
# Wave 6.5 (2026-05-23): the last two drift entries
# (``INPUT_AUTH_INVALID_SHAPE`` and ``UI_OUTPUT_DIR_FORBIDDEN``) were
# promoted into ``exceptions.ERROR_CODES``, so the allowlist is now
# empty. New literal ``code='X'`` emissions in ``backpropagate/`` must
# be added to the canonical catalog rather than re-populating this
# allowlist — the v1.3 milestone retired the drift surface.
KNOWN_DRIFT_TO_PROMOTE_IN_V1_3: frozenset[str] = frozenset()


def _scan_emitted_codes(source_root: pathlib.Path) -> dict[str, list[str]]:
    """Walk source_root/**/*.py and return every literal ``code='...'`` kwarg.

    Returns a mapping ``code -> [list of "file:lineno" sites]`` so a failed
    assertion can point the operator straight at the offending line.

    Skips:
    - Test files (``test_*.py``, ``*_test.py``)
    - Legacy modules (``*_legacy.py``)
    - The ``__pycache__`` tree (Path.rglob already skips it but be explicit)

    Only matches literal string constants in keyword position. Dynamic
    code strings (``code=some_var``) are NOT detected — they would defeat
    the catalog discipline anyway and the audit chain has not surfaced any.
    """
    sites: dict[str, list[str]] = {}
    for py in source_root.rglob("*.py"):
        name = py.name
        if name.startswith("test_") or name.endswith("_test.py"):
            continue
        if "_legacy" in name:
            continue
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover — defensive
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "code":
                    continue
                if not isinstance(kw.value, ast.Constant):
                    continue
                if not isinstance(kw.value.value, str):
                    continue
                code = kw.value.value
                sites.setdefault(code, []).append(
                    f"{py.as_posix()}:{node.lineno}"
                )
    return sites


SOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "backpropagate"


def test_every_emitted_code_is_in_catalog_or_known_drift():
    """Every literal ``code='X'`` in backpropagate/ must resolve to a catalog key.

    The only exceptions are the codes in ``KNOWN_DRIFT_TO_PROMOTE_IN_V1_3``,
    which the v1.2.0 CHANGELOG explicitly tracks as tech debt. v1.3 will
    promote them; until then, this list bounds the drift.
    """
    sites = _scan_emitted_codes(SOURCE_ROOT)
    emitted = set(sites.keys())
    catalog = set(ERROR_CODES.keys())

    missing = emitted - catalog - KNOWN_DRIFT_TO_PROMOTE_IN_V1_3
    if missing:
        # Build a precise failure message that names the offending sites
        # so the next contributor knows exactly where to look.
        site_lines = []
        for code in sorted(missing):
            for site in sites[code]:
                site_lines.append(f"    {code} @ {site}")
        site_block = "\n".join(site_lines)
        raise AssertionError(
            f"\nCodes emitted but missing from exceptions.ERROR_CODES "
            f"(and not in the KNOWN_DRIFT_TO_PROMOTE_IN_V1_3 allowlist):\n"
            f"{site_block}\n\n"
            f"Add the code to backpropagate/exceptions.py::ERROR_CODES "
            f"with description, default_hint, and retryable fields. "
            f"See the docstring in exceptions.py at line 115 for the "
            f"catalog discipline."
        )


def test_known_drift_allowlist_is_still_drifting():
    """Allowlist hygiene — every entry must still actually be drifting.

    If v1.3 lands a fix (promotes a code into ERROR_CODES) but forgets to
    remove the code from this allowlist, the allowlist would silently mask
    a future regression on that code. This test detects the stale allowlist
    entry and fails so the contributor knows to clean up.
    """
    catalog = set(ERROR_CODES.keys())
    stale = KNOWN_DRIFT_TO_PROMOTE_IN_V1_3 & catalog
    assert not stale, (
        f"KNOWN_DRIFT_TO_PROMOTE_IN_V1_3 contains codes that are NOW in "
        f"exceptions.ERROR_CODES: {sorted(stale)}. Remove these from the "
        f"allowlist so a future drift on the same code fails the test."
    )


def test_known_drift_allowlist_codes_are_actually_emitted():
    """Allowlist hygiene — every entry must still actually be emitted.

    If a deprecated code is removed from the source but stays in this
    allowlist, the allowlist hides a documentation gap. This test fails
    so the contributor knows to clean up after the removal.
    """
    sites = _scan_emitted_codes(SOURCE_ROOT)
    emitted = set(sites.keys())
    orphaned = KNOWN_DRIFT_TO_PROMOTE_IN_V1_3 - emitted
    assert not orphaned, (
        f"KNOWN_DRIFT_TO_PROMOTE_IN_V1_3 contains codes that are no "
        f"longer emitted from backpropagate/*.py: {sorted(orphaned)}. "
        f"Remove from the allowlist (the drift is closed)."
    )


def test_catalog_describes_stale_entries():
    """Advisory — report catalog entries that are never emitted (warn-only).

    This is NOT a hard assertion — some codes may be intentionally
    pre-declared for an upcoming raise site (e.g. PEFT_API_INCOMPATIBLE
    was added to the catalog before all the SLAO checks landed). But the
    list is useful documentation debt: a contributor adding a new code
    can see which existing entries might already cover their case.

    Run with ``pytest -s`` to see the stdout report.
    """
    sites = _scan_emitted_codes(SOURCE_ROOT)
    emitted = set(sites.keys())
    catalog = set(ERROR_CODES.keys())
    never_emitted = catalog - emitted

    # Stable assertion: the count must not silently double — bound it so
    # an explosion of new pre-declared codes shows up as a failure.
    assert len(never_emitted) <= 20, (
        f"Catalog has {len(never_emitted)} never-emitted entries — "
        f"this is documentation debt. Either remove the unused entries "
        f"or add the raise sites that need them. Never-emitted codes: "
        f"{sorted(never_emitted)}"
    )
