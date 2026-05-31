"""Unit tests for the README error-code drift check (Class 4b).

v1.4 Wave A1 (TESTSCI-A-002). The drift gate at ``scripts/check_doc_drift.py``
historically scanned only the handbook for error-code references, so a wrong
code in README.md's troubleshooting table was invisible to the gate built to
catch it (TESTSCI-A-001: the ``--share``/``--auth`` row named the HF-push code
``INPUT_AUTH_REQUIRED`` instead of the code the runtime actually raises,
``RUNTIME_UI_AUTH_NOT_ENFORCED``). Class 4b extends the gate to scan README.md
for backtick-wrapped error-code-shaped tokens and assert each is a live key in
``backpropagate/exceptions.py:ERROR_CODES``.

Test approach mirrors ``tests/test_check_doc_drift.py``: the module is loaded
from ``scripts/check_doc_drift.py`` via ``importlib`` and its path constants
are monkeypatched at a ``tmp_path`` synthetic tree so the scanner runs the
same code path the production gate does, without depending on live repo state.

Scope note: the gate verifies "is a live key," not "is the *right* key for
this symptom." A wrong-but-real code (the literal TESTSCI-A-001 bug, since
``INPUT_AUTH_REQUIRED`` is a real catalog entry) is NOT caught — that would
need a symptom→code mapping the gate has no source of truth for. What IS
caught is the structural class the doctrine names: a renamed / stale / typo'd
code that no longer exists in the catalog.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from textwrap import dedent

import pytest


def _load_drift_module(monkeypatch: pytest.MonkeyPatch, **path_overrides):
    """Load scripts/check_doc_drift.py and override path constants.

    Same loader shape as ``tests/test_check_doc_drift.py`` — the production
    gate runs as a script, so we load it by file location rather than putting
    ``scripts/`` on PYTHONPATH.
    """
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "check_doc_drift.py"
    spec = importlib.util.spec_from_file_location("check_doc_drift_readme", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in path_overrides.items():
        monkeypatch.setattr(module, name, value)
    return module


# A trimmed ERROR_CODES catalog with the prefix families the README uses —
# enough to exercise prefix-anchored matching without copying the full
# exceptions.py catalog. The check derives valid prefixes from these keys.
_CATALOG_STUB = dedent(
    """
    ERROR_CODES: dict[str, dict[str, str]] = {
        "INPUT_AUTH_REQUIRED": {"description": "x", "default_hint": "y", "retryable": "no"},
        "RUNTIME_UI_AUTH_NOT_ENFORCED": {"description": "x", "default_hint": "y", "retryable": "no"},
        "RUNTIME_GPU_OOM": {"description": "x", "default_hint": "y", "retryable": "no"},
        "DEP_OLLAMA_REGISTRATION_FAILED": {"description": "x", "default_hint": "y", "retryable": "yes"},
    }
    """
).strip()


def _bind(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    """Load the drift module with README + exceptions + allow-list at tmp_path."""
    (tmp_path / "backpropagate").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "backpropagate" / "exceptions.py").write_text(
        _CATALOG_STUB, encoding="utf-8"
    )
    # Empty allow-list by default.
    (tmp_path / "scripts" / "doc-drift-allow.toml").write_text("\n", encoding="utf-8")
    return _load_drift_module(
        monkeypatch,
        REPO_ROOT=tmp_path,
        SOURCE_ROOT=tmp_path / "backpropagate",
        EXCEPTIONS_PATH=tmp_path / "backpropagate" / "exceptions.py",
        README_PATH=tmp_path / "README.md",
        ALLOW_LIST_PATH=tmp_path / "scripts" / "doc-drift-allow.toml",
    )


class TestReadmeErrorCodeRefs:
    """Class 4b: README error-code refs must be live ERROR_CODES keys."""

    def test_no_finding_when_all_codes_are_live(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = _bind(monkeypatch, tmp_path)
        (tmp_path / "README.md").write_text(
            dedent(
                """
                | Symptom | Error code | Fix |
                |---|---|---|
                | OOM | `RUNTIME_GPU_OOM` | reduce batch |
                | share rejected | `RUNTIME_UI_AUTH_NOT_ENFORCED` | pass --auth |
                | ollama down | `DEP_OLLAMA_REGISTRATION_FAILED` | start daemon |
                """
            ).strip(),
            encoding="utf-8",
        )
        assert module.check_readme_error_code_refs() == []

    def test_finding_when_code_is_renamed_or_stale(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A code-shaped token that is not a catalog key is a doc-lie.

        This is the structural class TESTSCI-A-002 closes: README naming a
        code that does not exist (renamed / stale / typo'd).
        """
        module = _bind(monkeypatch, tmp_path)
        (tmp_path / "README.md").write_text(
            dedent(
                """
                | Symptom | Error code | Fix |
                |---|---|---|
                | share rejected | `RUNTIME_UI_AUTH_NOT_ENFORCEDXYZ` | pass --auth |
                """
            ).strip(),
            encoding="utf-8",
        )
        findings = module.check_readme_error_code_refs()
        assert len(findings) == 1
        assert "RUNTIME_UI_AUTH_NOT_ENFORCEDXYZ" in findings[0]
        assert "ERROR_CODES" in findings[0]

    def test_real_but_wrong_code_is_not_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Documents the scope boundary: a real-but-semantically-wrong code passes.

        ``INPUT_AUTH_REQUIRED`` IS a real catalog key, so the "is it a live
        key" gate cannot know it's the wrong code for the --share row. This is
        the exact TESTSCI-A-001 bug, and the gate intentionally does not flag
        it — semantic symptom→code matching has no source of truth here.
        """
        module = _bind(monkeypatch, tmp_path)
        (tmp_path / "README.md").write_text(
            dedent(
                """
                | Symptom | Error code | Fix |
                |---|---|---|
                | share rejected | `INPUT_AUTH_REQUIRED` | pass --auth |
                """
            ).strip(),
            encoding="utf-8",
        )
        assert module.check_readme_error_code_refs() == []

    def test_prose_and_config_snippets_do_not_false_positive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backtick tokens that aren't error-code families must not trip the check.

        ``mode="full"`` / ``BACKPROPAGATE_DEBUG`` / ``Trainer(...)`` are all
        backtick-wrapped in the real README but are not error codes — the
        prefix-anchored matcher must skip them.
        """
        module = _bind(monkeypatch, tmp_path)
        (tmp_path / "README.md").write_text(
            dedent(
                """
                Pass `Trainer(..., mode="full")` or `--mode=full`. Set
                `BACKPROPAGATE_DEBUG=1` for verbose logs. Quantize with
                `quant_method="aqlm"`. Codes like `RUNTIME_GPU_OOM` are stable.
                """
            ).strip(),
            encoding="utf-8",
        )
        assert module.check_readme_error_code_refs() == []

    def test_allowlist_suppresses_intentional_non_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``readme_error_code_ref_grandfathered`` suppresses a known exception."""
        module = _bind(monkeypatch, tmp_path)
        (tmp_path / "README.md").write_text(
            "Illustrative typo demo: `RUNTIME_GPU_OOO` (note the typo).\n",
            encoding="utf-8",
        )
        (tmp_path / "scripts" / "doc-drift-allow.toml").write_text(
            dedent(
                """
                readme_error_code_ref_grandfathered = [
                    "RUNTIME_GPU_OOO",
                ]
                """
            ).strip(),
            encoding="utf-8",
        )
        assert module.check_readme_error_code_refs() == []
