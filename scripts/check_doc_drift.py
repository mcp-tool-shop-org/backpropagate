#!/usr/bin/env python3
"""Drift-detection: catch within-swarm doc-lies before merge.

Per ``[[within-swarm-doc-lie-drift-detection]]`` doctrine (earned v1.3,
2026-05-24). Every wave that adds new surface (env vars / CLI flags / state
fields) risks shipping a doc-lie alongside in the same release — the audit
chain caught two examples from Stage C's own ship in Wave 5. This script is
the forcing function that fires on every PR so Wave 6+ commits get caught
BEFORE merge instead of after.

Ten checks (5-class extension landed v1.4 Wave 6a — was 4 in v1.3; Class 4
gained a 4b README variant in v1.4 Wave A1, bringing the wired total to ten)
----------------------------------------------------------------------------
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

Five-class extension (added v1.4 Wave 6a)
-----------------------------------------
5. CLASS 1 (argparse defaults vs handbook copy) — every ``default=N`` value
   in ``cli.py`` ``add_argument`` calls must match the documented default
   in the cli-reference.md row for that flag (catches the case where a
   CLI default bumps but the handbook table still names the old value).
6. CLASS 2 (env var NAMES advertised in llms.txt vs runtime) — every
   ``BACKPROPAGATE_*`` token in ``llms.txt`` must correspond to a real
   runtime read in ``backpropagate/**/*.py`` (catches the case where an
   env var is documented to LLM agents but no code path actually reads
   it — or vice versa, the converse).
7. CLASS 3 (env var DEFAULT values in env-vars.md vs config.py source-
   of-truth) — every ``BACKPROPAGATE_*`` row in env-vars.md whose env var
   maps cleanly to a ``Settings`` field must show the same default the
   Pydantic field declares (catches default-bump drift across the two
   surfaces).
8. CLASS 4 (error-codes.md "Fix" column cross-doc consistency) — every
   handbook reference to ``BACKPROPAGATE_*``, ``--flag``, or another
   error code inside the error-codes.md Fix column must point at a thing
   that still exists (env var read, registered flag, catalog entry).
9. CLASS 4b (README error-code refs, added v1.4 Wave A1) — every
   backtick-wrapped ``PREFIX_...`` token in README.md drawn from a live
   ERROR_CODES prefix family must be an actual key in
   ``backpropagate/exceptions.py:ERROR_CODES``. Catches the case where the
   README troubleshooting table names a stale / renamed / wrong code
   (TESTSCI-A-001: the --share/--auth row named INPUT_AUTH_REQUIRED instead
   of the runtime's RUNTIME_UI_AUTH_NOT_ENFORCED). Class 4 scanned only the
   handbook, leaving README drift invisible to the gate built to catch it.
10. CLASS 5 (CI workflow step NAMES / comments vs flag semantics) —
   every ``MEDIUM/MEDIUM`` / ``HIGH`` / ``LOW/LOW`` etc. severity-claim
   in a workflow step name or comment must match the actual flag
   semantics on the next command line (catches the Bandit
   ``-l -i`` = LOW/LOW vs ``MEDIUM/MEDIUM`` doc-lie that fired in v1.4
   Wave 5).

Usage
-----
    python scripts/check_doc_drift.py

Exit 0 ⇒ no drift; exit 1 ⇒ drift detected (report written to stdout).

The script is stdlib-only (Python 3.10+) — no extra deps so it can run as
the first CI step on a stock setup-python action.

Pre-commit
----------
A lightweight pre-commit hook config lives at ``.pre-commit-config.yaml``
(repo root) so the check fires on ``git commit`` locally as well as in
CI. CONTRIBUTING.md names the hook in the "Local development" section.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# tomllib is Python 3.11+ stdlib; fall back to tomli on 3.10 (tomli is a
# transitive dep via multiple packages in uv.lock with marker
# python_full_version < '3.11', so it's guaranteed available on 3.10).
try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef,import-not-found]

REPO_ROOT = Path(__file__).parent.parent
SOURCE_ROOT = REPO_ROOT / "backpropagate"
UI_APP_ROOT = SOURCE_ROOT / "ui_app"
CLI_PATH = SOURCE_ROOT / "cli.py"
CONFIG_PATH = SOURCE_ROOT / "config.py"
UI_STATE_PATH = SOURCE_ROOT / "ui_state.py"
EXCEPTIONS_PATH = SOURCE_ROOT / "exceptions.py"
HANDBOOK_ROOT = REPO_ROOT / "site" / "src" / "content" / "docs" / "handbook"
ENV_VARS_DOC = HANDBOOK_ROOT / "env-vars.md"
CLI_REF_DOC = HANDBOOK_ROOT / "cli-reference.md"
ERROR_CODES_DOC = HANDBOOK_ROOT / "error-codes.md"
README_PATH = REPO_ROOT / "README.md"
LLMS_TXT_PATH = REPO_ROOT / "llms.txt"
WORKFLOWS_ROOT = REPO_ROOT / ".github" / "workflows"
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

# Class 2: BACKPROPAGATE_* tokens advertised in llms.txt — covers both
# backticked and bare references because the LLMs-txt convention is to
# enumerate the env vars in prose.
LLMS_ENV_VAR_PATTERN = re.compile(r"\bBACKPROPAGATE_[A-Z0-9_]+\b")

# Class 5: Bandit / Trivy / pip-audit severity claims in CI step copy.
# The trigger word is one of the canonical CVE/CVSS severity labels.
SEVERITY_CLAIM_PATTERN = re.compile(
    r"\b(LOW|MEDIUM|HIGH|CRITICAL)\b(?:\s*/\s*\b(LOW|MEDIUM|HIGH|CRITICAL)\b)?",
    re.IGNORECASE,
)


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

    Pydantic-settings-bound env vars (the ``BACKPROPAGATE_GROUP__FIELD``
    shape) are intentionally NOT folded in here — they have their own
    catalog (``_enumerate_env_vars`` in cli.py) and their own check (the
    Class 1 / Class 3 entry-point), and Check 1 was tuned against the
    regex scan's narrower scope as the baseline.
    """
    found: set[str] = set()
    for py_path in _iter_python_sources(SOURCE_ROOT):
        text = py_path.read_text(encoding="utf-8")
        for match in ENV_VAR_PATTERN.finditer(text):
            name = match.group(1) or match.group(2)
            if name:
                found.add(name)
    return found


def _find_indirect_module_const_env_vars() -> set[str]:
    """Return env var strings stashed in module-level constants.

    Walks the AST of every Python source file for module-level
    assignments of the shape ``_FOO_ENV = "BACKPROPAGATE_X"``. Used by
    Class 2 (llms.txt cross-check) so middleware modules that read env
    via ``env.get(_FOO_ENV)`` instead of an inline literal don't
    surface as false-positive "advertised but not read."
    """
    found: set[str] = set()
    for py_path in _iter_python_sources(SOURCE_ROOT):
        text = py_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        # Module level only — class- and function-local constants are
        # different shapes (typing aliases, dispatch dicts) that don't
        # match the "stash an env var name in a constant" pattern.
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if not isinstance(node.value, ast.Constant):
                continue
            if not isinstance(node.value.value, str):
                continue
            if node.value.value.startswith("BACKPROPAGATE_"):
                found.add(node.value.value)
    return found


def _find_pydantic_bound_env_vars() -> set[str]:
    """Return ``BACKPROPAGATE_*`` env vars auto-bound by BaseSettings classes.

    Walks ``config.py`` for every ``BaseSettings`` subclass with
    ``env_prefix="BACKPROPAGATE_X__"`` and emits ``BACKPROPAGATE_X__<FIELD>``
    for each typed field. Used by Class 2 (llms.txt cross-check) so
    llms.txt entries for ``BACKPROPAGATE_LORA__R`` etc. aren't flagged
    as "advertised but not read" — pydantic-settings binds them.
    """
    if not CONFIG_PATH.exists():
        return set()
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        env_prefix: str | None = None
        for item in node.body:
            if not (isinstance(item, ast.Assign) and len(item.targets) == 1):
                continue
            target = item.targets[0]
            if not (isinstance(target, ast.Name) and target.id == "model_config"):
                continue
            if not isinstance(item.value, ast.Call):
                continue
            for kw in item.value.keywords:
                if kw.arg == "env_prefix" and isinstance(kw.value, ast.Constant):
                    env_prefix = kw.value.value
        if env_prefix is None or not env_prefix.startswith("BACKPROPAGATE_"):
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            if not isinstance(item.target, ast.Name):
                continue
            field_name = item.target.id
            if field_name.startswith("_"):
                continue
            found.add(f"{env_prefix}{field_name.upper()}")
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
# Class 1 (5-class extension v1.4 Wave 6a) — argparse defaults vs handbook
# ---------------------------------------------------------------------------


def _find_argparse_flag_defaults() -> dict[str, str]:
    """Return ``{flag_name: stringified_default}`` for ``add_argument`` calls.

    Walks the AST of ``cli.py`` for every ``parser.add_argument("--flag",
    ..., default=<value>, ...)`` shape. Skips flags whose ``default`` is
    not a literal (e.g. dynamic ``default=get_default()``) — those can't
    be cross-checked against a static handbook table without executing
    the function. The returned value is the Python ``repr`` of the
    default literal so callers can match against handbook strings of the
    form ``2e-4`` / ``256`` / ``"quality"``.
    """
    if not CLI_PATH.exists():
        return {}
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
    flag_defaults: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        flag_name: str | None = None
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ):
                flag_name = arg.value
                break
        if flag_name is None:
            continue
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            if isinstance(kw.value, ast.Constant):
                flag_defaults[flag_name] = repr(kw.value.value)
            # Skip dynamic defaults (Call / Name / Attribute) — not
            # cross-checkable without executing the function.
    return flag_defaults


def _parse_handbook_flag_table_defaults() -> dict[str, str]:
    """Return ``{flag_name: documented_default_text}`` from cli-reference.md.

    The handbook tables are Markdown rows shaped like::

        | `--flag` | `default` | Description |

    A small fraction of rows use sentinel words (``required``, ``auto``,
    ``unset``, ``off``) instead of a literal default — those are still
    captured as the documented value; the cross-check tolerates them
    via an allow-list of accepted sentinels.
    """
    if not CLI_REF_DOC.exists():
        return {}
    text = CLI_REF_DOC.read_text(encoding="utf-8")
    # Match a row of the form `| `--flag` | <default> | ...`.
    row_pattern = re.compile(
        r"^\|\s*`(--[a-z][a-z0-9-]*)`\s*\|\s*([^|]+?)\s*\|",
        re.MULTILINE,
    )
    documented: dict[str, str] = {}
    for match in row_pattern.finditer(text):
        flag = match.group(1)
        default_cell = match.group(2).strip()
        # Strip surrounding backticks and ``**bold**`` markup so the cell
        # comparison can match ``256`` / ``2e-4`` / ``required``.
        default_cell = default_cell.strip("`").strip("*").strip()
        documented[flag] = default_cell
    return documented


# Sentinels the handbook uses in lieu of a literal default. These are
# treated as "documented" even though they don't match the argparse
# literal repr — the table's job is to communicate the runtime shape,
# not to be a machine-readable mirror of the argparse Namespace.
#
# Prefix-match: any handbook cell that STARTS WITH one of these sentinels
# (e.g. ``unset (all)``, ``unset (auto-detect)``, ``unset (stdout)``)
# is treated as "documented sentinel for argparse default=None". The
# parenthesised suffix is operator-facing prose, not a literal default.
_HANDBOOK_DEFAULT_SENTINELS = frozenset(
    {
        "required",
        "auto",
        "unset",
        "off",
        "on",
        "(packing on by default)",  # --no-packing-style flags
        "false",
        "true",
        "none",
        "**required**",
    }
)

# Sentinels that map to argparse ``default=None`` — when the handbook
# says ``unset`` (or any ``unset (...)`` variant) and argparse has
# ``default=None``, that's the documented contract, not drift. Same for
# ``required``: argparse may not name a default at all (= None) when the
# flag is marked ``required=True``.
_HANDBOOK_NONE_SENTINELS = ("unset", "required", "none", "**required**", "auto")


def _normalise_default(value: str) -> str:
    """Lowercase + strip surrounding whitespace and quote-noise for matching."""
    return value.strip().strip("`").strip('"').strip("'").lower()


def check_argparse_default_drift() -> list[str]:
    """Class 1: argparse ``default=N`` vs handbook ``default: N`` strings.

    Honors the ``argparse_default_drift_allowlist`` allow-list key — a
    dict of ``flag_name: justification`` (the value is a free-form note
    so reviewers can see why the row is exempt without scrolling). NEW
    drift on an unlisted flag fails the gate.
    """
    allow = _load_allow_list()
    grandfathered = set(allow.get("argparse_default_drift_allowlist", {}).keys())
    runtime = _find_argparse_flag_defaults()
    documented = _parse_handbook_flag_table_defaults()
    findings: list[str] = []
    for flag in sorted(runtime):
        if flag in grandfathered:
            continue
        if flag not in documented:
            # Coverage gap — handbook doesn't have a row at all. That's a
            # Class 2 (cli-flag) problem, not a Class 1 default-drift
            # problem; skip here to avoid duplicate findings.
            continue
        argparse_default = _normalise_default(runtime[flag])
        # argparse repr-form: "'quality'" / "256" / "0.0" / "False" / "None"
        # — strip the outer quotes so string defaults can match the
        # handbook table's bare-text shape.
        argparse_default = argparse_default.strip("'").strip('"')
        handbook_default = _normalise_default(documented[flag])
        if handbook_default in _HANDBOOK_DEFAULT_SENTINELS:
            continue
        # argparse default=None ⇔ handbook ``unset`` / ``required`` /
        # ``auto`` (with any trailing parenthesised prose). The contract
        # is "no value supplied by argparse; runtime computes / requires
        # one." Not drift.
        if argparse_default == "none" and any(
            handbook_default == s or handbook_default.startswith(s + " ")
            for s in _HANDBOOK_NONE_SENTINELS
        ):
            continue
        if argparse_default == handbook_default:
            continue
        # Some handbook cells include trailing prose like "256 (must be
        # > 0)" — extract the leading token and re-compare.
        leading_token = handbook_default.split()[0] if handbook_default else ""
        if argparse_default == leading_token:
            continue
        # Float-shape match: argparse ``0.0002`` ⇔ handbook ``2e-4`` — both
        # are the same number, different notation. Round-trip through
        # float() and compare numerically.
        try:
            if float(argparse_default) == float(leading_token):
                continue
        except (ValueError, TypeError):
            pass
        findings.append(
            f"{flag}: argparse default={argparse_default!r} but handbook says "
            f"{handbook_default!r}"
        )
    return findings


# ---------------------------------------------------------------------------
# Class 2 (5-class extension v1.4 Wave 6a) — llms.txt env vars vs runtime
# ---------------------------------------------------------------------------


def _find_llms_txt_env_vars() -> set[str]:
    """Return the set of ``BACKPROPAGATE_*`` tokens that appear in llms.txt."""
    if not LLMS_TXT_PATH.exists():
        return set()
    text = LLMS_TXT_PATH.read_text(encoding="utf-8")
    return set(LLMS_ENV_VAR_PATTERN.findall(text))


def check_llms_txt_env_vars() -> list[str]:
    """Class 2: env vars advertised in llms.txt vs runtime reads.

    Two failure modes:
    - ``llms.txt`` names ``BACKPROPAGATE_FOO`` but no code path reads it
      (LLM agent gets a hallucinated knob — high-severity doc-lie).
    - The converse — ``BACKPROPAGATE_FOO`` is read at runtime but
      ``llms.txt`` doesn't mention it — is NOT a finding here because
      ``llms.txt`` is a curated highlights file, not a complete catalog
      (the full reference lives in env-vars.md and the
      ``_enumerate_env_vars`` catalog, both covered by Check 1).

    "Read at runtime" combines three signals:
    1. Direct ``os.environ.get`` / ``env.get`` literal (the regex scan).
    2. Module-level constant aliasing an env-var literal (the AST walk
       for middleware indirection).
    3. Pydantic-settings auto-bound field via ``env_prefix`` (the AST
       walk over config.py BaseSettings subclasses).

    Honors the ``llms_txt_env_var_grandfathered`` allow-list key for
    documented-but-not-read knobs (rare — should converge to empty).
    """
    allow = _load_allow_list()
    grandfathered = set(allow.get("llms_txt_env_var_grandfathered", []))
    runtime = (
        _find_runtime_env_vars()
        | _find_indirect_module_const_env_vars()
        | _find_pydantic_bound_env_vars()
    )
    llms = _find_llms_txt_env_vars()
    findings: list[str] = []
    for name in sorted(llms):
        if name in grandfathered:
            continue
        if name not in runtime:
            findings.append(
                f"{name}: advertised in llms.txt but not read at runtime in "
                "backpropagate/**/*.py"
            )
    return findings


# ---------------------------------------------------------------------------
# Class 3 (5-class extension v1.4 Wave 6a) — env-vars.md defaults vs config
# ---------------------------------------------------------------------------


def _find_config_field_defaults() -> dict[str, str]:
    """Return ``{BACKPROPAGATE_GROUP__FIELD: stringified_default}`` from config.py.

    Walks the AST of every ``BaseSettings`` subclass in ``config.py``,
    reads its ``model_config = SettingsConfigDict(env_prefix=...)`` to
    learn the prefix, and emits the full ``BACKPROPAGATE_<GROUP>__<FIELD>``
    name with the AST-stringified default value. Only literal defaults
    are captured — ``Field(default=...)`` calls with literal arguments
    AND bare ``name: type = literal`` assignments. Dynamic defaults (e.g.
    ``Field(default_factory=...)``) are skipped because the static
    repr doesn't generalise.
    """
    if not CONFIG_PATH.exists():
        return {}
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    defaults: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Find the env_prefix from model_config = SettingsConfigDict(...).
        env_prefix: str | None = None
        for item in node.body:
            if not (isinstance(item, ast.Assign) and len(item.targets) == 1):
                continue
            target = item.targets[0]
            if not (isinstance(target, ast.Name) and target.id == "model_config"):
                continue
            if not isinstance(item.value, ast.Call):
                continue
            for kw in item.value.keywords:
                if kw.arg == "env_prefix" and isinstance(kw.value, ast.Constant):
                    env_prefix = kw.value.value
        if env_prefix is None:
            continue
        # Now walk the class body for `name: type = literal` and
        # `name: type = Field(default=literal)` shapes.
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            if not isinstance(item.target, ast.Name):
                continue
            field_name = item.target.id
            if field_name.startswith("_"):
                continue
            default_value: ast.expr | None = item.value
            if default_value is None:
                continue
            literal_default: str | None = None
            if isinstance(default_value, ast.Constant):
                literal_default = str(default_value.value)
            elif (
                isinstance(default_value, ast.Call)
                and isinstance(default_value.func, ast.Name)
                and default_value.func.id == "Field"
            ):
                for kw in default_value.keywords:
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                        literal_default = str(kw.value.value)
            if literal_default is None:
                continue
            env_name = f"{env_prefix}{field_name.upper()}"
            defaults[env_name] = literal_default
    return defaults


def _parse_env_vars_md_defaults() -> dict[str, str]:
    """Return ``{BACKPROPAGATE_*: documented_default_text}`` from env-vars.md.

    Parses Markdown rows of shape ``| `BACKPROPAGATE_FOO` | <default> | ...``.
    """
    if not ENV_VARS_DOC.exists():
        return {}
    text = ENV_VARS_DOC.read_text(encoding="utf-8")
    row_pattern = re.compile(
        r"^\|\s*`(BACKPROPAGATE_[A-Z0-9_]+)`\s*\|\s*([^|]+?)\s*\|",
        re.MULTILINE,
    )
    documented: dict[str, str] = {}
    for match in row_pattern.finditer(text):
        env_var = match.group(1)
        default_cell = match.group(2).strip()
        default_cell = default_cell.strip("`").strip("*").strip()
        documented[env_var] = default_cell
    return documented


# Sentinels env-vars.md uses for "no default" / dynamic defaults.
_ENV_DEFAULT_SENTINELS = frozenset(
    {
        "unset",
        "auto-detect",
        "auto",
        "none",
        "unset (random)",
        "unset (stdout)",
        "unset (no restriction)",
    }
)


def check_env_var_default_drift() -> list[str]:
    """Class 3: env-vars.md default cells vs config.py source-of-truth.

    Honors the ``env_var_default_drift_allowlist`` allow-list key with the
    same shape as Class 1 — ``{env_var: justification}``. Only env vars
    that appear in BOTH config.py AND env-vars.md are checked; missing-
    row cases are covered by Check 1 (handbook-grandfathered).
    """
    allow = _load_allow_list()
    grandfathered = set(allow.get("env_var_default_drift_allowlist", {}).keys())
    runtime = _find_config_field_defaults()
    documented = _parse_env_vars_md_defaults()
    findings: list[str] = []
    for env_var in sorted(runtime):
        if env_var in grandfathered:
            continue
        if env_var not in documented:
            continue  # missing-row case handled elsewhere
        runtime_default = _normalise_default(runtime[env_var])
        handbook_default = _normalise_default(documented[env_var])
        if handbook_default in _ENV_DEFAULT_SENTINELS:
            continue
        # config.py ``None`` ⇔ env-vars.md ``unset`` / ``unset (...)`` /
        # ``auto-detect`` / ``unset (auto)`` etc. Same prefix-match shape
        # as Class 1.
        if runtime_default == "none" and any(
            handbook_default == s or handbook_default.startswith(s + " ")
            for s in _HANDBOOK_NONE_SENTINELS
        ):
            continue
        if runtime_default == handbook_default:
            continue
        # Handbook may include trailing prose like "100 (rolling 60s window)".
        leading_token = handbook_default.split()[0] if handbook_default else ""
        if runtime_default == leading_token:
            continue
        # Handle the common float-shape variants 2e-4 vs 0.0002 etc. by
        # round-tripping both through float() when both parse.
        try:
            if float(runtime_default) == float(leading_token):
                continue
        except (ValueError, TypeError):
            pass
        findings.append(
            f"{env_var}: config.py default={runtime_default!r} but env-vars.md "
            f"says {handbook_default!r}"
        )
    return findings


# ---------------------------------------------------------------------------
# Class 4 (5-class extension v1.4 Wave 6a) — error-codes.md Fix-column refs
# ---------------------------------------------------------------------------


# Class 4: looser env-var pattern — matches the env var name anywhere
# inside a backtick-wrapped span. Operator-facing prose often writes
# ``BACKPROPAGATE_DEBUG=1`` inside backticks where the backtick body
# carries the ``=value`` suffix; the docs-narrow pattern at
# ``DOCUMENTED_ENV_VAR_PATTERN`` only matches the exact-backtick form.
_DOC_REF_ENV_VAR_PATTERN = re.compile(r"\bBACKPROPAGATE_[A-Z0-9_]+\b")


def _find_error_codes() -> set[str]:
    """Return the live ``ERROR_CODES`` keys by AST-parsing exceptions.py.

    The catalog at ``backpropagate/exceptions.py:ERROR_CODES`` is the single
    source of truth for stable machine-readable codes. We read it via AST
    (rather than ``import backpropagate.exceptions``) to keep this script
    stdlib-only — the package import path can pull in torch / pydantic on a
    full install, and the drift gate is the first CI step on a stock
    setup-python action. Walks the module body for the
    ``ERROR_CODES: ... = {...}`` assignment and collects every string-literal
    dict key.
    """
    if not EXCEPTIONS_PATH.exists():
        return set()
    tree = ast.parse(EXCEPTIONS_PATH.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        # Match both ``ERROR_CODES = {...}`` (Assign) and the annotated
        # ``ERROR_CODES: dict[...] = {...}`` (AnnAssign) shapes.
        target_is_error_codes = False
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            target_is_error_codes = any(
                isinstance(t, ast.Name) and t.id == "ERROR_CODES"
                for t in node.targets
            )
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target_is_error_codes = (
                isinstance(node.target, ast.Name)
                and node.target.id == "ERROR_CODES"
            )
            value = node.value
        if not target_is_error_codes or not isinstance(value, ast.Dict):
            continue
        for key in value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                found.add(key.value)
    return found


def check_error_codes_doc_refs() -> list[str]:
    """Class 4: error-codes.md Fix column references must point at real things.

    For every error-codes.md mention of ``BACKPROPAGATE_FOO`` or
    ``--flag`` inside the Fix column, verify the named knob actually
    exists in the current runtime (Class 1 / Class 2 surfaces both feed
    in here). Catches the case where a recovery path names an env var
    that was renamed or a flag that was retired.

    Detection scope: env vars are matched with a permissive boundary-
    word pattern so ``BACKPROPAGATE_DEBUG=1`` inside backticks is
    captured as ``BACKPROPAGATE_DEBUG``. Flags use the narrower
    backtick-wrapped pattern (``--foo``) because operators rarely
    embed flags in larger tokens.

    Honors the ``error_codes_doc_ref_grandfathered`` allow-list key for
    references that intentionally point at future / external surfaces
    (e.g. an arXiv link, an environment variable from an external tool
    like ``HF_TOKEN``).
    """
    if not ERROR_CODES_DOC.exists():
        return []
    allow = _load_allow_list()
    grandfathered = set(allow.get("error_codes_doc_ref_grandfathered", []))
    text = ERROR_CODES_DOC.read_text(encoding="utf-8")
    runtime_env_vars = (
        _find_runtime_env_vars()
        | _find_indirect_module_const_env_vars()
        | _find_pydantic_bound_env_vars()
    )
    # Layer in the explicit doc allow-lists so a "documented but not in
    # this drift gate's runtime scan" reference doesn't false-positive
    # — env-vars.md is the authoritative documented surface.
    documented_env_vars = _find_documented_env_vars()
    runtime_flags = _find_runtime_flags()
    documented_flags = _find_documented_flags()
    findings: list[str] = []
    # Parse out every BACKPROPAGATE_* / `--flag` token in the body of
    # the Markdown.
    for env_match in _DOC_REF_ENV_VAR_PATTERN.finditer(text):
        name = env_match.group(0)
        if name in grandfathered:
            continue
        if name in runtime_env_vars or name in documented_env_vars:
            continue
        findings.append(
            f"error-codes.md references {name}, but no runtime read AND no "
            "env-vars.md row exists for it"
        )
    for flag_match in DOCUMENTED_FLAG_PATTERN.finditer(text):
        flag = flag_match.group(1)
        if flag in grandfathered:
            continue
        if flag in runtime_flags or flag in documented_flags:
            continue
        findings.append(
            f"error-codes.md references {flag}, but no add_argument site AND "
            "no cli-reference.md row exists for it"
        )
    return findings


# ---------------------------------------------------------------------------
# Class 4b (v1.4 Wave A1) — README error-code references vs ERROR_CODES catalog
# ---------------------------------------------------------------------------
#
# Sibling of Class 4: the same "named code must be a live key" invariant, but
# applied to README.md instead of error-codes.md. README's troubleshooting
# table is a load-bearing operator surface that names stable codes the user
# greps their logs for — a wrong code there is the exact doc-lie this gate
# exists to catch (TESTSCI-A-001: the --share/--auth row named the HF-push
# code INPUT_AUTH_REQUIRED instead of the code the runtime actually raises,
# RUNTIME_UI_AUTH_NOT_ENFORCED). Before this check the gate scanned only the
# handbook, so README error-code drift was invisible to the gate built to
# catch it.


def _readme_error_code_token_pattern(error_codes: set[str]) -> re.Pattern[str] | None:
    """Build a backtick-anchored matcher for ERROR_CODES-shaped README tokens.

    Conservative by construction: the matcher only fires on a fully
    backtick-wrapped, all-caps ``PREFIX_...`` token whose ``PREFIX_`` is one
    of the prefix families that actually appear in the live ``ERROR_CODES``
    catalog (``INPUT_``, ``CONFIG_``, ``DEP_``, ``RUNTIME_``, ``STATE_``,
    ``PARTIAL_``, ``HUB_``, ``PEFT_``, ``UI_``, ``SLAO_`` today). Anchoring
    on real families is what keeps prose / config snippets like ``mode="full"``
    or env vars like ``BACKPROPAGATE_DEBUG`` (a different family, covered by
    the env-var checks) from surfacing as false positives — a token has to
    look exactly like an error code from a known family to be considered.

    Returns ``None`` when the catalog yields no usable prefixes (defensive;
    in practice the catalog is always populated).
    """
    prefixes = sorted(
        {code.split("_", 1)[0] for code in error_codes if "_" in code}
    )
    if not prefixes:
        return None
    prefix_alt = "|".join(re.escape(p) for p in prefixes)
    # `(PREFIX_REST)` — one of the live families, an underscore, then one or
    # more uppercase / digit / underscore chars, the whole thing wrapped in
    # single backticks so prose mentions outside code spans never match.
    return re.compile(rf"`((?:{prefix_alt})_[A-Z0-9_]+)`")


def check_readme_error_code_refs() -> list[str]:
    """Class 4b: every error-code token named in README.md must be a live key.

    Scans README.md for backtick-wrapped ``PREFIX_...`` tokens drawn from the
    live ERROR_CODES prefix families and asserts each is an actual key in
    ``backpropagate/exceptions.py:ERROR_CODES``. Catches the case where the
    troubleshooting table (or any README prose) names a stale, renamed, or
    simply wrong code — the operator greps their logs for that string and
    finds nothing, or finds the wrong remediation.

    Honors the ``readme_error_code_ref_grandfathered`` allow-list key for
    tokens that intentionally aren't catalog keys (rare — e.g. a deliberately
    illustrative ``RUNTIME_GPU_OOO`` typo demo, should it ever be needed).
    """
    if not README_PATH.exists():
        return []
    error_codes = _find_error_codes()
    # If we couldn't read the catalog at all, stay silent rather than flag
    # every README code as "unknown" — a missing/unparseable exceptions.py is
    # a different, louder failure caught elsewhere.
    if not error_codes:
        return []
    pattern = _readme_error_code_token_pattern(error_codes)
    if pattern is None:
        return []
    allow = _load_allow_list()
    grandfathered = set(allow.get("readme_error_code_ref_grandfathered", []))
    text = README_PATH.read_text(encoding="utf-8")
    findings: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(text):
        token = match.group(1)
        if token in seen:
            continue
        seen.add(token)
        if token in grandfathered:
            continue
        if token not in error_codes:
            findings.append(
                f"README.md references error code `{token}`, but it is not a "
                "key in backpropagate/exceptions.py:ERROR_CODES (renamed, "
                "stale, or the wrong code for the documented symptom)"
            )
    return findings


# ---------------------------------------------------------------------------
# Class 5 (5-class extension v1.4 Wave 6a) — workflow severity-claim doc-lies
# ---------------------------------------------------------------------------


# Map a Bandit short-form severity flag to its English label. Bandit
# accepts ``-l`` / ``-ll`` / ``-lll`` for low / medium / high severity
# floors, and ``-i`` / ``-ii`` / ``-iii`` for confidence floors.
_BANDIT_SEVERITY_LEVELS = {
    "l": "LOW",
    "ll": "MEDIUM",
    "lll": "HIGH",
}
_BANDIT_CONFIDENCE_LEVELS = {
    "i": "LOW",
    "ii": "MEDIUM",
    "iii": "HIGH",
}


def _scan_workflow_for_severity_drift(path: Path) -> list[str]:
    """Walk a workflow YAML for step-name/comment severity claims vs flags.

    Two doc-lie patterns covered today:
    - Bandit step name or comment claims ``MEDIUM/MEDIUM`` (or any other
      severity pair) but the next ``bandit ...`` line uses ``-l -i``
      flags (= LOW/LOW). The cross-check pairs the closest preceding
      claim with the closest following bandit invocation.
    - Trivy / pip-audit step name or comment claims ``CRITICAL`` (or
      similar) but the next ``trivy`` / ``pip-audit`` line passes a
      mismatched ``--severity`` / ``--vulnerability-floor`` flag.

    NB: stdlib-only — no PyYAML. Treats the workflow as plain text
    because the load-bearing pattern is "line N claims X, line N+k
    invokes a tool with flags F." YAML structure doesn't change that
    relationship and the diff cost of pulling in a YAML parser as a
    drift-script dep is not earned by the improvement.
    """
    findings: list[str] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    # First pass: build a list of (line_idx, claim, source_line_text)
    # for every step name / comment that mentions a severity word.
    claims: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        # Step-name shape: ``      - name: Run Bandit (gating — MEDIUM ...)``
        # Comment shape:   ``      # ...MEDIUM severity / MEDIUM confidence...``
        if "name:" not in line and "#" not in line:
            continue
        m = SEVERITY_CLAIM_PATTERN.search(line)
        if m is None:
            continue
        sev_main = m.group(1).upper() if m.group(1) else ""
        sev_secondary = m.group(2).upper() if m.group(2) else ""
        claim = f"{sev_main}/{sev_secondary}" if sev_secondary else sev_main
        claims.append((idx, claim, line.strip()))
    # Second pass: for each claim, find the next ``bandit ...`` /
    # ``trivy ...`` / ``pip-audit ...`` invocation within the same step
    # (heuristic: within 30 lines).
    for claim_idx, claim, claim_line in claims:
        # Skip claims that aren't paired with a CLI invocation — the
        # surrounding documentation prose is allowed to mention
        # "CRITICAL" without triggering a finding.
        invocation_line: str | None = None
        invocation_tool: str | None = None
        for j in range(claim_idx + 1, min(claim_idx + 30, len(lines))):
            stripped = lines[j].strip()
            # Stop scanning when we leave the step (new ``- name:`` line)
            if stripped.startswith("- name:"):
                break
            for tool in ("bandit ", "trivy ", "pip-audit ", "bandit\\"):
                if tool in stripped:
                    invocation_line = stripped
                    invocation_tool = tool.strip().rstrip("\\")
                    break
            if invocation_line is not None:
                break
        if invocation_line is None or invocation_tool is None:
            continue
        # Now cross-check the claim against the actual flag semantics.
        if invocation_tool == "bandit":
            # Match `-l` / `-ll` / `-lll` and `-i` / `-ii` / `-iii`.
            sev_flag_match = re.search(r"-l{1,3}\b", invocation_line)
            conf_flag_match = re.search(r"-i{1,3}\b", invocation_line)
            if sev_flag_match is None or conf_flag_match is None:
                continue
            sev_flag = sev_flag_match.group(0).lstrip("-")
            conf_flag = conf_flag_match.group(0).lstrip("-")
            actual_sev = _BANDIT_SEVERITY_LEVELS.get(sev_flag, "UNKNOWN")
            actual_conf = _BANDIT_CONFIDENCE_LEVELS.get(conf_flag, "UNKNOWN")
            actual_pair = f"{actual_sev}/{actual_conf}"
            # claim is one of "X" or "X/Y" depending on whether the
            # step-name/comment captured a paired severity. Accept both
            # equivalent shapes as a match:
            #   - "X/Y" must match actual_pair exactly.
            #   - "X" matches when X is the severity AND the confidence
            #     half also matches (e.g. step says "LOW severity / LOW
            #     confidence" — first capture is LOW, claim is "LOW",
            #     and that's correct against actual "LOW/LOW").
            claim_matches = (
                claim == actual_pair
                or (claim == actual_sev and actual_sev == actual_conf)
            )
            if not claim_matches:
                findings.append(
                    f"{path.relative_to(REPO_ROOT)}: Bandit step claims "
                    f"{claim} but ``-{sev_flag} -{conf_flag}`` = {actual_pair} "
                    f"(claim line: {claim_line!r})"
                )
        # Trivy / pip-audit severity cross-check is a v1.5 extension —
        # the v1.4 Wave 6a closure covers Bandit only because that's the
        # one that fired. The scanner shape is reusable; add new tool
        # blocks here as they earn drift incidents.
    return findings


def check_workflow_severity_drift() -> list[str]:
    """Class 5: CI step name/comment severity claims vs actual flag semantics.

    Walks every ``.github/workflows/*.yml`` file looking for the Bandit
    pattern (``MEDIUM/MEDIUM`` claim vs ``-l -i`` flags).

    Honors the ``workflow_severity_drift_allowlist`` allow-list key —
    a list of file-path:line-number strings for known-OK exceptions
    (e.g. a comment that intentionally documents the historical claim
    inside a "we fixed this" block).
    """
    if not WORKFLOWS_ROOT.exists():
        return []
    allow = _load_allow_list()
    grandfathered = set(allow.get("workflow_severity_drift_allowlist", []))
    findings: list[str] = []
    for path in sorted(WORKFLOWS_ROOT.glob("*.yml")):
        for finding in _scan_workflow_for_severity_drift(path):
            # Allow-list match: substring-in-finding (file path + line
            # excerpt — easy to maintain since the finding includes the
            # full claim line).
            if any(allow_key in finding for allow_key in grandfathered):
                continue
            findings.append(finding)
    return findings


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    # Windows-first: this script prints doc cells verbatim (drift findings echo
    # the offending text). A non-cp1252 glyph in a doc — e.g. an arrow or a >=
    # sign — would otherwise crash the gate with UnicodeEncodeError on a legacy
    # console. Reconfigure to UTF-8 with errors="replace" so the gate reports
    # rather than crashes (mirrors the cli._reconfigure_stdio_utf8 fix).
    for _stream in (sys.stdout, sys.stderr):
        _reconfigure = getattr(_stream, "reconfigure", None)
        if _reconfigure is not None:
            try:
                _reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001  # nosec B110 — never fail the gate over stdio
                pass
    all_findings: list[str] = []
    checks = (
        check_env_vars,
        check_cli_flags,
        check_reflex_state_fields,
        check_error_codes_test_exists,
        # 5-class extension (v1.4 Wave 6a).
        check_argparse_default_drift,
        check_llms_txt_env_vars,
        check_env_var_default_drift,
        check_error_codes_doc_refs,
        # Class 4b extension (v1.4 Wave A1) — README error-code refs.
        check_readme_error_code_refs,
        check_workflow_severity_drift,
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
