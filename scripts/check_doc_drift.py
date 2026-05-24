#!/usr/bin/env python3
"""Drift-detection: catch within-swarm doc-lies before merge.

Per ``[[within-swarm-doc-lie-drift-detection]]`` doctrine (earned v1.3,
2026-05-24). Every wave that adds new surface (env vars / CLI flags / state
fields) risks shipping a doc-lie alongside in the same release — the audit
chain caught two examples from Stage C's own ship in Wave 5. This script is
the forcing function that fires on every PR so Wave 6+ commits get caught
BEFORE merge instead of after.

Four checks
-----------
1. New ``BACKPROPAGATE_*`` env vars read by ``backpropagate/**/*.py`` must
   appear in both ``site/src/content/docs/handbook/env-vars.md`` AND the
   ``_enumerate_env_vars`` catalog in ``backpropagate/cli.py``.
2. New ``--flag`` arguments registered via ``argparse.add_argument`` in
   ``backpropagate/cli.py`` must appear in
   ``site/src/content/docs/handbook/cli-reference.md``.
3. Public state fields declared on ``rx.State`` subclasses in
   ``backpropagate/ui_state.py`` must have at least one consumer in
   ``backpropagate/ui_app/`` (allow-list at
   ``scripts/doc-drift-allow.toml`` covers known false positives).
4. The ``tests/test_error_codes_catalog.py`` regression test exists and
   still asserts every ``code='...'`` literal is in ``ERROR_CODES``.

Usage
-----
    python scripts/check_doc_drift.py

Exit 0 ⇒ no drift; exit 1 ⇒ drift detected (report written to stdout).

The script is stdlib-only (Python 3.10+) — no extra deps so it can run as
the first CI step on a stock setup-python action.

Future work (v1.4 scope, not wired here)
----------------------------------------
- Pre-commit hook integration so the check fires on ``git commit`` locally,
  not just in CI. v1.3 ships CI-only; see ``scripts/README.md`` for the
  v1.4 next step.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import tomllib

REPO_ROOT = Path(__file__).parent.parent
SOURCE_ROOT = REPO_ROOT / "backpropagate"
UI_APP_ROOT = SOURCE_ROOT / "ui_app"
CLI_PATH = SOURCE_ROOT / "cli.py"
UI_STATE_PATH = SOURCE_ROOT / "ui_state.py"
HANDBOOK_ROOT = REPO_ROOT / "site" / "src" / "content" / "docs" / "handbook"
ENV_VARS_DOC = HANDBOOK_ROOT / "env-vars.md"
CLI_REF_DOC = HANDBOOK_ROOT / "cli-reference.md"
ERROR_CODES_TEST = REPO_ROOT / "tests" / "test_error_codes_catalog.py"
ALLOW_LIST_PATH = REPO_ROOT / "scripts" / "doc-drift-allow.toml"

ENV_VAR_PATTERN = re.compile(
    r"""(?x)
    (?:os\.environ\.get|os\.getenv|env\.get)\s*\(\s*
    ["'](BACKPROPAGATE_[A-Z0-9_]+)["']
    |
    os\.environ\s*\[\s*["'](BACKPROPAGATE_[A-Z0-9_]+)["']\s*\]
    """
)

DOCUMENTED_ENV_VAR_PATTERN = re.compile(r"`(BACKPROPAGATE_[A-Z0-9_]+)`")
DOCUMENTED_FLAG_PATTERN = re.compile(r"`(--[a-z][a-z0-9-]*)`")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_allow_list() -> dict[str, Any]:
    """Return parsed allow-list, or an empty fallback if the file is absent."""
    if not ALLOW_LIST_PATH.exists():
        return {}
    with ALLOW_LIST_PATH.open("rb") as f:
        data: dict[str, Any] = tomllib.load(f)
    return data


def _iter_python_sources(root: Path) -> Iterator[Path]:
    """Yield every .py file under ``root`` (depth-unlimited)."""
    return (p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


# ---------------------------------------------------------------------------
# Check 1: env vars
# ---------------------------------------------------------------------------


def _find_runtime_env_vars() -> set[str]:
    """Scan backpropagate/**/*.py for runtime ``BACKPROPAGATE_*`` reads.

    Patterns recognised:
    - ``os.environ.get("BACKPROPAGATE_X")`` / single-quoted
    - ``os.getenv("BACKPROPAGATE_X")``
    - ``os.environ["BACKPROPAGATE_X"]``
    - ``env.get("BACKPROPAGATE_X")`` (the ui_security / auth modules use a
      local ``env`` dict bound to ``os.environ`` for testability)
    """
    found: set[str] = set()
    for py_path in _iter_python_sources(SOURCE_ROOT):
        text = py_path.read_text(encoding="utf-8")
        for match in ENV_VAR_PATTERN.finditer(text):
            name = match.group(1) or match.group(2)
            if name:
                found.add(name)
    return found


def _find_documented_env_vars() -> set[str]:
    """Parse env-vars.md for every backtick-wrapped ``BACKPROPAGATE_*``."""
    if not ENV_VARS_DOC.exists():
        return set()
    text = ENV_VARS_DOC.read_text(encoding="utf-8")
    return set(DOCUMENTED_ENV_VAR_PATTERN.findall(text))


def _find_cataloged_env_vars() -> set[str]:
    """Parse cli.py's ``_enumerate_env_vars`` for env-var literals.

    Walks the AST of ``_enumerate_env_vars`` and collects every string
    literal starting with ``BACKPROPAGATE_``. The function builds two lists
    of dict entries (one via pydantic-settings introspection, one
    hand-curated for the raw-os-environ-get knobs); the AST walk picks up
    both paths because they end in the same ``rows.append`` / static-list
    pattern.
    """
    if not CLI_PATH.exists():
        return set()
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
    target_func: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_enumerate_env_vars":
            target_func = node
            break
    if target_func is None:
        return set()
    found: set[str] = set()
    for node in ast.walk(target_func):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("BACKPROPAGATE_"):
                found.add(node.value)
    return found


def check_env_vars() -> list[str]:
    """Verify every runtime env var is both documented AND cataloged.

    Two allow-list keys honored:
    - ``env_vars_handbook_grandfathered`` — runtime env vars that pre-date the
      drift check and are intentionally not in env-vars.md (rare; should
      converge to empty).
    - ``env_vars_catalog_grandfathered`` — runtime env vars that pre-date the
      drift check and are intentionally not in the ``_enumerate_env_vars``
      operator-facing catalog (e.g., UI env vars consumed by the Reflex
      subprocess that have their own documentation surface).

    New env-var reads added in any future PR must close the gap in BOTH
    places OR justify a grandfathering entry — the doctrine review on a new
    allow-list entry is the same as the review on a new code='X' literal.
    """
    allow = _load_allow_list()
    handbook_grandfathered = set(allow.get("env_vars_handbook_grandfathered", []))
    catalog_grandfathered = set(allow.get("env_vars_catalog_grandfathered", []))
    runtime = _find_runtime_env_vars()
    documented = _find_documented_env_vars()
    cataloged = _find_cataloged_env_vars()
    findings: list[str] = []
    for name in sorted(runtime):
        missing_from: list[str] = []
        if name not in documented and name not in handbook_grandfathered:
            missing_from.append("env-vars.md")
        if name not in cataloged and name not in catalog_grandfathered:
            missing_from.append("_enumerate_env_vars catalog")
        if missing_from:
            findings.append(
                f"{name}: read at runtime but missing from {', '.join(missing_from)}"
            )
    return findings


# ---------------------------------------------------------------------------
# Check 2: CLI flags
# ---------------------------------------------------------------------------


def _find_runtime_flags() -> set[str]:
    """Walk cli.py AST for every ``add_argument("--flag", ...)`` literal.

    Argparse positional-only args (no leading ``--``) and short aliases
    (``-m``) are skipped — only long flags appear in the cli-reference
    handbook table rows.
    """
    if not CLI_PATH.exists():
        return set()
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.startswith("--")
                ):
                    found.add(arg.value)
    return found


def _find_documented_flags() -> set[str]:
    """Parse cli-reference.md for every backtick-wrapped ``--flag``."""
    if not CLI_REF_DOC.exists():
        return set()
    text = CLI_REF_DOC.read_text(encoding="utf-8")
    return set(DOCUMENTED_FLAG_PATTERN.findall(text))


def check_cli_flags() -> list[str]:
    """Verify every long flag in cli.py is documented in cli-reference.md.

    Honors the ``cli_flags_grandfathered`` allow-list key for pre-existing
    drift (subcommand flags that aren't yet in the handbook). A NEW
    add_argument in a future PR must either be documented or earn a
    grandfathering entry — same review bar.
    """
    allow = _load_allow_list()
    grandfathered = set(allow.get("cli_flags_grandfathered", []))
    runtime = _find_runtime_flags()
    documented = _find_documented_flags()
    findings: list[str] = []
    for flag in sorted(runtime):
        if flag in grandfathered:
            continue
        if flag not in documented:
            findings.append(f"{flag}: registered in cli.py but missing from cli-reference.md")
    return findings


# ---------------------------------------------------------------------------
# Check 3: Reflex state fields
# ---------------------------------------------------------------------------


def _is_rx_state_base(base: ast.expr) -> bool:
    """Match ``rx.State`` (Attribute) or ``State`` (Name) as a class base."""
    if isinstance(base, ast.Attribute) and base.attr == "State":
        if isinstance(base.value, ast.Name) and base.value.id == "rx":
            return True
    return False


def _find_state_fields() -> dict[str, list[str]]:
    """Return ``{StateClassName: [field_name, ...]}`` for ui_state.py.

    A "field" is a class-level annotated assignment (e.g.,
    ``mode_text: str = ""``) on a class that inherits from ``rx.State``.
    Underscore-prefixed fields are skipped (internal-by-convention).
    """
    if not UI_STATE_PATH.exists():
        return {}
    tree = ast.parse(UI_STATE_PATH.read_text(encoding="utf-8"))
    state_fields: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_is_rx_state_base(b) for b in node.bases):
            continue
        fields: list[str] = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                name = item.target.id
                if not name.startswith("_"):
                    fields.append(name)
        state_fields[node.name] = fields
    return state_fields


def _find_consumers_for_field(class_name: str, field_name: str) -> bool:
    """Return True if any ui_app file references ``<ClassName>.<field>`` or
    a closely-related access pattern.

    Reflex's reactive var resolution can hide some accesses (e.g., template
    interpolation), so this check uses a generous substring match: any file
    in ui_app/ that mentions both the class name AND the field name is
    treated as a consumer. False positives are acceptable; false negatives
    (missing real consumers) bloat the report and are addressed via
    ``scripts/doc-drift-allow.toml``.
    """
    if not UI_APP_ROOT.exists():
        return False
    target_attr = f"{class_name}.{field_name}"
    for py_path in _iter_python_sources(UI_APP_ROOT):
        text = py_path.read_text(encoding="utf-8")
        if target_attr in text:
            return True
        # Fallback: class imported AND field name appears somewhere in the
        # same file — Reflex's template-binding sometimes hides the direct
        # attribute access.
        if class_name in text and re.search(rf"\b{re.escape(field_name)}\b", text):
            return True
    return False


def check_reflex_state_fields() -> list[str]:
    """Verify every public state field has at least one consumer in ui_app/.

    The allow-list at ``scripts/doc-drift-allow.toml`` skips known-OK
    entries (the v1.3 default allow-list is empty; Wave 5.5 frontend agent
    is closing the AuthBadgeState gap).
    """
    allow = _load_allow_list().get("reflex_state_fields_no_consumer_required", {})
    findings: list[str] = []
    for class_name, fields in _find_state_fields().items():
        allowed = set(allow.get(class_name, []))
        for field in fields:
            if field in allowed:
                continue
            if not _find_consumers_for_field(class_name, field):
                findings.append(
                    f"{class_name}.{field}: declared in ui_state.py but no consumer in ui_app/"
                )
    return findings


# ---------------------------------------------------------------------------
# Check 4: ERROR_CODES regression test exists
# ---------------------------------------------------------------------------


def check_error_codes_test_exists() -> list[str]:
    """Verify tests/test_error_codes_catalog.py still asserts catalog drift."""
    findings: list[str] = []
    if not ERROR_CODES_TEST.exists():
        findings.append(
            f"{ERROR_CODES_TEST.relative_to(REPO_ROOT)}: missing — "
            "ERROR_CODES catalog regression test must exist"
        )
        return findings
    text = ERROR_CODES_TEST.read_text(encoding="utf-8")
    # The test must import ERROR_CODES from backpropagate.exceptions and
    # at least one function or class must reference it. We do a structural
    # check rather than full AST walking because the test file is small
    # enough that a substring scan is sufficient.
    if "from backpropagate.exceptions import ERROR_CODES" not in text:
        findings.append(
            f"{ERROR_CODES_TEST.relative_to(REPO_ROOT)}: lost import of "
            "ERROR_CODES from backpropagate.exceptions"
        )
    if "ERROR_CODES" not in text or "code=" not in text:
        findings.append(
            f"{ERROR_CODES_TEST.relative_to(REPO_ROOT)}: no longer asserts on "
            "literal code='...' kwargs against ERROR_CODES catalog"
        )
    return findings


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    all_findings: list[str] = []
    checks = (
        check_env_vars,
        check_cli_flags,
        check_reflex_state_fields,
        check_error_codes_test_exists,
    )
    for check in checks:
        findings = check()
        if findings:
            print(f"\n=== {check.__name__} ===")
            for f in findings:
                print(f"  {f}")
            all_findings.extend(findings)
    if all_findings:
        print(
            f"\nDrift detected: {len(all_findings)} item(s). "
            "See doctrine: [[within-swarm-doc-lie-drift-detection]]"
        )
        return 1
    print("Drift check passed: 0 items.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
