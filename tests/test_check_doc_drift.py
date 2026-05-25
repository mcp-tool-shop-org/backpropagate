"""Unit tests for scripts/check_doc_drift.py — the 5-class extension.

v1.4 Wave 6a (CIDOCS-F / V1_4_BRIEF item 5). The drift gate is the
forcing function for [[within-swarm-doc-lie-drift-detection]]. v1.3
shipped four checks (env vars / CLI flags / state fields / error-codes
catalog regression). v1.4 Wave 6a adds five more:

- Class 1: argparse defaults vs handbook copy
- Class 2: env var names advertised in llms.txt vs runtime reads
- Class 3: env var DEFAULT values in env-vars.md vs config.py source-of-truth
- Class 4: error-codes.md "Fix" column cross-references
- Class 5: CI workflow step name/comment severity claims vs flag semantics

Test approach
-------------
Each scanner is imported directly from ``scripts.check_doc_drift`` so the
unit tests run the same code path the production gate does. Each test
constructs a tmp_path-anchored mini-tree with the load-bearing surfaces
(cli.py / config.py / handbook .md / llms.txt / workflows yml) so we
can assert on the finding behaviour without depending on the live
backpropagate state.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from textwrap import dedent

import pytest

# ---------------------------------------------------------------------------
# Module loader — load check_doc_drift.py as a module without putting
# scripts/ on PYTHONPATH (the production gate runs as a script).
# ---------------------------------------------------------------------------


def _load_drift_module(monkeypatch: pytest.MonkeyPatch, **path_overrides):
    """Load scripts/check_doc_drift.py with REPO_ROOT pointed at a tmp tree.

    ``path_overrides`` lets a test override REPO_ROOT (and the derived
    paths) so the scanners operate on a synthetic tree.
    """
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "check_doc_drift.py"
    spec = importlib.util.spec_from_file_location("check_doc_drift", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Now patch the module-level path constants per the overrides.
    for name, value in path_overrides.items():
        monkeypatch.setattr(module, name, value)
    return module


@pytest.fixture
def drift_tree(tmp_path: Path) -> Path:
    """Create a minimal synthetic repo tree under tmp_path.

    Provides:
    - tmp_path/backpropagate/cli.py     (argparse + env reads)
    - tmp_path/backpropagate/config.py  (Pydantic BaseSettings)
    - tmp_path/backpropagate/ui_state.py (rx.State subclass)
    - tmp_path/backpropagate/ui_app/    (consumer scan target)
    - tmp_path/site/src/content/docs/handbook/{env-vars,cli-reference,error-codes}.md
    - tmp_path/llms.txt
    - tmp_path/.github/workflows/ci.yml
    - tmp_path/scripts/doc-drift-allow.toml
    - tmp_path/tests/test_error_codes_catalog.py
    """
    (tmp_path / "backpropagate" / "ui_app").mkdir(parents=True)
    (tmp_path / "site" / "src" / "content" / "docs" / "handbook").mkdir(parents=True)
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    # Empty allow-list (no allow-listed entries by default).
    (tmp_path / "scripts" / "doc-drift-allow.toml").write_text("\n", encoding="utf-8")
    # Drop a placeholder ui_state so the original Reflex-state scanner
    # doesn't error on a missing file.
    (tmp_path / "backpropagate" / "ui_state.py").write_text("\n", encoding="utf-8")
    # Drop a placeholder error-codes regression test so Check 4 passes.
    (tmp_path / "tests" / "test_error_codes_catalog.py").write_text(
        dedent(
            """
            from backpropagate.exceptions import ERROR_CODES

            def test_catalog():
                # references code= literal
                assert 'INPUT_VALIDATION_FAILED' in ERROR_CODES
                _ = "code='X'"
            """
        ).strip(),
        encoding="utf-8",
    )
    return tmp_path


def _bind_drift_module(
    monkeypatch: pytest.MonkeyPatch, tree: Path
):
    """Helper — load the drift module against the synthetic tree."""
    return _load_drift_module(
        monkeypatch,
        REPO_ROOT=tree,
        SOURCE_ROOT=tree / "backpropagate",
        UI_APP_ROOT=tree / "backpropagate" / "ui_app",
        CLI_PATH=tree / "backpropagate" / "cli.py",
        CONFIG_PATH=tree / "backpropagate" / "config.py",
        UI_STATE_PATH=tree / "backpropagate" / "ui_state.py",
        HANDBOOK_ROOT=tree / "site" / "src" / "content" / "docs" / "handbook",
        ENV_VARS_DOC=tree / "site" / "src" / "content" / "docs" / "handbook" / "env-vars.md",
        CLI_REF_DOC=tree / "site" / "src" / "content" / "docs" / "handbook" / "cli-reference.md",
        ERROR_CODES_DOC=tree / "site" / "src" / "content" / "docs" / "handbook" / "error-codes.md",
        LLMS_TXT_PATH=tree / "llms.txt",
        WORKFLOWS_ROOT=tree / ".github" / "workflows",
        ERROR_CODES_TEST=tree / "tests" / "test_error_codes_catalog.py",
        ALLOW_LIST_PATH=tree / "scripts" / "doc-drift-allow.toml",
    )


# ===========================================================================
# Class 1 — argparse defaults vs handbook copy
# ===========================================================================


class TestArgparseDefaultDrift:
    """Class 1: argparse ``default=N`` vs handbook ``default: N`` strings."""

    def test_no_drift_when_defaults_match(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (drift_tree / "backpropagate" / "cli.py").write_text(
            dedent(
                """
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--steps", default=100)
                parser.add_argument("--lora-r", default=256)
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "cli-reference.md").write_text(
            dedent(
                """
                | Flag | Default | Description |
                |------|---------|-------------|
                | `--steps` | `100` | Number of steps. |
                | `--lora-r` | `256` | LoRA rank. |
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_argparse_default_drift() == []

    def test_drift_when_handbook_lags_argparse(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (drift_tree / "backpropagate" / "cli.py").write_text(
            dedent(
                """
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--lora-r", default=256)
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "cli-reference.md").write_text(
            dedent(
                """
                | Flag | Default | Description |
                |------|---------|-------------|
                | `--lora-r` | `16` | LoRA rank. |
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        findings = module.check_argparse_default_drift()
        assert len(findings) == 1
        assert "--lora-r" in findings[0]
        assert "256" in findings[0]
        assert "16" in findings[0]

    def test_float_shape_equivalence_2e4_vs_0_0002(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """argparse ``0.0002`` ⇔ handbook ``2e-4`` — same number, no drift."""
        (drift_tree / "backpropagate" / "cli.py").write_text(
            dedent(
                """
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--lr", default=2e-4)
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "cli-reference.md").write_text(
            dedent(
                """
                | Flag | Default | Description |
                |------|---------|-------------|
                | `--lr` | `2e-4` | Learning rate. |
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_argparse_default_drift() == []

    def test_none_sentinels_unset_required_auto(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """argparse ``default=None`` ⇔ handbook ``unset`` / ``unset (...)``."""
        (drift_tree / "backpropagate" / "cli.py").write_text(
            dedent(
                """
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--max-samples", default=None)
                parser.add_argument("--status", default=None)
                parser.add_argument("--vram-gb", default=None)
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "cli-reference.md").write_text(
            dedent(
                """
                | Flag | Default | Description |
                |------|---------|-------------|
                | `--max-samples` | `unset (all)` | Cap samples. |
                | `--status` | `unset (all)` | Filter. |
                | `--vram-gb` | `unset (auto-detect)` | VRAM. |
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_argparse_default_drift() == []

    def test_allowlist_suppresses_documented_fallthrough(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``argparse_default_drift_allowlist`` should suppress a known mismatch."""
        (drift_tree / "backpropagate" / "cli.py").write_text(
            dedent(
                """
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--host", default=None)
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "cli-reference.md").write_text(
            dedent(
                """
                | Flag | Default | Description |
                |------|---------|-------------|
                | `--host` | `127.0.0.1` | Bind host. |
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "scripts" / "doc-drift-allow.toml").write_text(
            dedent(
                """
                [argparse_default_drift_allowlist]
                "--host" = "argparse default=None; config.py supplies the runtime default"
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_argparse_default_drift() == []


# ===========================================================================
# Class 2 — llms.txt env vars vs runtime reads
# ===========================================================================


class TestLlmsTxtEnvVars:
    """Class 2: llms.txt env-var names vs runtime ``os.environ.get`` reads."""

    def test_no_finding_when_runtime_reads_match(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (drift_tree / "backpropagate" / "cli.py").write_text(
            dedent(
                """
                import os
                _ = os.environ.get("BACKPROPAGATE_FOO")
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "llms.txt").write_text(
            "Knobs: BACKPROPAGATE_FOO is the canonical foo toggle.\n",
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_llms_txt_env_vars() == []

    def test_finding_when_llms_advertises_unread_var(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (drift_tree / "backpropagate" / "cli.py").write_text(
            "# no env reads at all\n",
            encoding="utf-8",
        )
        (drift_tree / "llms.txt").write_text(
            "Knobs: BACKPROPAGATE_GHOST is documented but no code reads it.\n",
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        findings = module.check_llms_txt_env_vars()
        assert len(findings) == 1
        assert "BACKPROPAGATE_GHOST" in findings[0]
        assert "advertised in llms.txt" in findings[0]

    def test_pydantic_settings_field_counts_as_runtime_read(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pydantic BaseSettings env_prefix binding counts as a runtime read."""
        (drift_tree / "backpropagate" / "config.py").write_text(
            dedent(
                """
                from pydantic_settings import BaseSettings, SettingsConfigDict

                class ModelConfig(BaseSettings):
                    model_config = SettingsConfigDict(env_prefix="BACKPROPAGATE_MODEL__")
                    name: str = "Qwen/Qwen2.5-7B-Instruct"
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "llms.txt").write_text(
            "BACKPROPAGATE_MODEL__NAME bound via pydantic-settings.\n",
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_llms_txt_env_vars() == []

    def test_module_const_alias_counts_as_runtime_read(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Module-level ``_NAME = "BACKPROPAGATE_X"`` const counts as a read.

        The middleware modules stash the env-var name in a constant for
        testability. The Class 2 scanner needs to recognise that pattern
        so it doesn't false-positive llms.txt entries for those vars.
        """
        (drift_tree / "backpropagate" / "ui_app" / "rate_limit.py").write_text(
            dedent(
                """
                import os
                _HTTP_ENV = "BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN"

                def _resolve_cap(env_var: str, env: dict[str, str]) -> int:
                    return int(env.get(env_var, "100"))
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "llms.txt").write_text(
            "BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN is the per-IP rate cap.\n",
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_llms_txt_env_vars() == []


# ===========================================================================
# Class 3 — env-vars.md defaults vs config.py source-of-truth
# ===========================================================================


class TestEnvVarDefaultDrift:
    """Class 3: env-vars.md default cells vs config.py BaseSettings defaults."""

    def test_no_drift_when_defaults_match(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (drift_tree / "backpropagate" / "config.py").write_text(
            dedent(
                """
                from pydantic_settings import BaseSettings, SettingsConfigDict

                class LoRAConfig(BaseSettings):
                    model_config = SettingsConfigDict(env_prefix="BACKPROPAGATE_LORA__")
                    r: int = 256
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "env-vars.md").write_text(
            dedent(
                """
                | Variable | Default | What it does |
                |----------|---------|--------------|
                | `BACKPROPAGATE_LORA__R` | `256` | LoRA rank. |
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_env_var_default_drift() == []

    def test_drift_when_handbook_default_is_stale(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The packing default flipped True in config.py; env-vars.md was stale."""
        (drift_tree / "backpropagate" / "config.py").write_text(
            dedent(
                """
                from pydantic_settings import BaseSettings, SettingsConfigDict

                class DataConfig(BaseSettings):
                    model_config = SettingsConfigDict(env_prefix="BACKPROPAGATE_DATA__")
                    packing: bool = True
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "env-vars.md").write_text(
            dedent(
                """
                | Variable | Default | What it does |
                |----------|---------|--------------|
                | `BACKPROPAGATE_DATA__PACKING` | `false` | Combine short sequences. |
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        findings = module.check_env_var_default_drift()
        assert len(findings) == 1
        assert "BACKPROPAGATE_DATA__PACKING" in findings[0]
        assert "true" in findings[0].lower()
        assert "false" in findings[0].lower()

    def test_none_default_matches_unset_sentinel(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """config.py ``None`` ⇔ env-vars.md ``unset (auto)`` — no drift."""
        (drift_tree / "backpropagate" / "config.py").write_text(
            dedent(
                """
                from pydantic_settings import BaseSettings, SettingsConfigDict

                class ModelConfig(BaseSettings):
                    model_config = SettingsConfigDict(env_prefix="BACKPROPAGATE_MODEL__")
                    dtype: str | None = None
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "env-vars.md").write_text(
            dedent(
                """
                | Variable | Default | What it does |
                |----------|---------|--------------|
                | `BACKPROPAGATE_MODEL__DTYPE` | `unset (auto)` | Force bf16/fp16/fp32. |
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_env_var_default_drift() == []


# ===========================================================================
# Class 4 — error-codes.md "Fix" column cross-references
# ===========================================================================


class TestErrorCodesDocRefs:
    """Class 4: error-codes.md Fix column references must point at real things."""

    def test_no_finding_when_references_resolve(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (drift_tree / "backpropagate" / "cli.py").write_text(
            dedent(
                """
                import argparse, os
                _ = os.environ.get("BACKPROPAGATE_DEBUG")
                parser = argparse.ArgumentParser()
                parser.add_argument("--verbose", action="store_true")
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "env-vars.md").write_text(
            "| `BACKPROPAGATE_DEBUG` | unset | debug toggle |\n",
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "cli-reference.md").write_text(
            "| `--verbose` | off | verbose output |\n",
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "error-codes.md").write_text(
            dedent(
                """
                | Code | Fix |
                |------|-----|
                | RUNTIME_X | Re-run with `--verbose` or `BACKPROPAGATE_DEBUG=1`. |
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_error_codes_doc_refs() == []

    def test_finding_when_referenced_flag_does_not_exist(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (drift_tree / "backpropagate" / "cli.py").write_text(
            "# no add_argument anywhere\n",
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "cli-reference.md").write_text(
            "# empty handbook\n",
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "error-codes.md").write_text(
            dedent(
                """
                | Code | Fix |
                |------|-----|
                | RUNTIME_X | Run with `--ghost-flag` to enable recovery. |
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        findings = module.check_error_codes_doc_refs()
        assert len(findings) == 1
        assert "--ghost-flag" in findings[0]

    def test_finding_when_referenced_env_var_does_not_exist(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (drift_tree / "backpropagate" / "cli.py").write_text("\n", encoding="utf-8")
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "env-vars.md").write_text(
            "# empty handbook\n",
            encoding="utf-8",
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "error-codes.md").write_text(
            dedent(
                """
                | Code | Fix |
                |------|-----|
                | RUNTIME_X | Set `BACKPROPAGATE_HALLUCINATED_KNOB=1`. |
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        findings = module.check_error_codes_doc_refs()
        assert len(findings) == 1
        assert "BACKPROPAGATE_HALLUCINATED_KNOB" in findings[0]


# ===========================================================================
# Class 5 — CI workflow severity-claim doc-lies
# ===========================================================================


class TestWorkflowSeverityDrift:
    """Class 5: CI workflow step name/comment severity claims vs flag semantics."""

    def test_no_finding_when_claim_matches_flags(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (drift_tree / ".github" / "workflows" / "ci.yml").write_text(
            dedent(
                """
                jobs:
                  bandit:
                    steps:
                      - name: Run Bandit (LOW severity / LOW confidence)
                        run: |
                          bandit -r src/ -l -i -f txt
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_workflow_severity_drift() == []

    def test_finding_when_step_name_claims_medium_but_flags_are_low_low(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bandit step claims MEDIUM/MEDIUM but ``-l -i`` = LOW/LOW.

        This is exactly the doc-lie that fired in v1.4 Wave 5 and drove
        V1_4_BRIEF item 4. The scanner must catch it.
        """
        (drift_tree / ".github" / "workflows" / "ci.yml").write_text(
            dedent(
                """
                jobs:
                  bandit:
                    steps:
                      - name: Run Bandit (gating MEDIUM/MEDIUM threshold)
                        run: |
                          bandit -r src/ -l -i -f txt
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        findings = module.check_workflow_severity_drift()
        assert len(findings) == 1
        assert "MEDIUM/MEDIUM" in findings[0]
        assert "LOW/LOW" in findings[0]

    def test_no_finding_when_comment_severity_mention_is_unpaired(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A severity word in prose without a bandit invocation is not a doc-lie."""
        (drift_tree / ".github" / "workflows" / "ci.yml").write_text(
            dedent(
                """
                # CRITICAL is the floor we gate on at release; everything below is advisory.
                jobs:
                  example:
                    steps:
                      - name: Build
                        run: python -m build
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_workflow_severity_drift() == []

    def test_allowlist_suppresses_substring_match(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``workflow_severity_drift_allowlist`` entry suppresses a known doc-lie.

        Use case: a comment block that deliberately documents the
        historical bad claim inside a "we fixed this" passage.
        """
        (drift_tree / ".github" / "workflows" / "ci.yml").write_text(
            dedent(
                """
                jobs:
                  bandit:
                    steps:
                      - name: Run Bandit (gating MEDIUM/MEDIUM threshold)
                        run: |
                          bandit -r src/ -l -i -f txt
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "scripts" / "doc-drift-allow.toml").write_text(
            dedent(
                """
                workflow_severity_drift_allowlist = [
                    "Run Bandit (gating MEDIUM/MEDIUM threshold)",
                ]
                """
            ).strip(),
            encoding="utf-8",
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        assert module.check_workflow_severity_drift() == []


# ===========================================================================
# Class 1+ — driver smoke
# ===========================================================================


class TestDriver:
    """End-to-end smoke: the driver wires every scanner in."""

    def test_driver_returns_zero_on_clean_tree(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drop minimal-but-valid surfaces.
        (drift_tree / "backpropagate" / "cli.py").write_text(
            dedent(
                """
                def _enumerate_env_vars():
                    return []
                """
            ).strip(),
            encoding="utf-8",
        )
        (drift_tree / "llms.txt").write_text("# no env-var references\n", encoding="utf-8")
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "env-vars.md").write_text(
            "# empty\n", encoding="utf-8"
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "cli-reference.md").write_text(
            "# empty\n", encoding="utf-8"
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "error-codes.md").write_text(
            "# empty\n", encoding="utf-8"
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        # Mute the print noise during the test.
        result = module.main()
        assert result == 0

    def test_driver_returns_nonzero_on_any_drift(
        self, drift_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (drift_tree / "backpropagate" / "cli.py").write_text("\n", encoding="utf-8")
        (drift_tree / "llms.txt").write_text(
            "BACKPROPAGATE_HALLUCINATED is fake.\n", encoding="utf-8"
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "env-vars.md").write_text(
            "# empty\n", encoding="utf-8"
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "cli-reference.md").write_text(
            "# empty\n", encoding="utf-8"
        )
        (drift_tree / "site" / "src" / "content" / "docs" / "handbook" / "error-codes.md").write_text(
            "# empty\n", encoding="utf-8"
        )
        module = _bind_drift_module(monkeypatch, drift_tree)
        result = module.main()
        assert result == 1
