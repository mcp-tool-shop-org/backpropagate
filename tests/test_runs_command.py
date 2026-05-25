"""Tests for ``backprop runs`` command + ``_build_runs_payload`` + ``RunsState``.

TESTS-A-002 (v1.3 swarm Wave 1 Stage A): the BRIDGE-F-001 run-history data
API surface — the CLI ``runs`` subcommand, its underlying payload projector,
and the Reflex ``RunsState`` that consumes the same data path — had ZERO
regression tests after v1.2.0 shipped. A future schema-bump on
``RUNS_JSON_SCHEMA_VERSION`` or a renaming inside ``_build_runs_payload``
would have silently broken the Reflex /runs page without any CI signal.

This file pins:
1. ``_build_runs_payload`` projection — empty list, populated, status filter
   passthrough (the function itself doesn't filter; it projects whatever
   RunHistoryManager.list_runs returns), schema_version contract.
2. ``cmd_runs`` end-to-end — empty dir (early-return JSON path), populated
   dir, ``--status`` filter, ``--limit`` cap, ``--json`` machine output.
3. ``RunsState`` state-handler — empty state default, populated load_runs,
   set_status_filter triggers reload, set_output_dir_override path validation.

All tests are read-only (no actual training); they manipulate run_history.json
on disk under ``tmp_path`` and exercise the projection / state code directly.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

# =============================================================================
# _build_runs_payload — pure projection contract
# =============================================================================


class TestBuildRunsPayload:
    """Pin the BRIDGE-F-001 payload shape so a frontend rename doesn't break."""

    def test_empty_list_produces_well_formed_payload(self, tmp_path):
        """Empty runs list returns a well-formed payload with schema_version."""
        from backpropagate.cli import RUNS_JSON_SCHEMA_VERSION, _build_runs_payload

        payload = _build_runs_payload([], tmp_path)

        assert payload["schema_version"] == RUNS_JSON_SCHEMA_VERSION
        assert payload["runs"] == []
        assert payload["output_dir"] == str(tmp_path)
        # generated_at is an ISO8601 string — must be present and parseable
        assert "generated_at" in payload
        assert "T" in payload["generated_at"], "generated_at must be ISO8601 datetime"

    def test_schema_version_is_string_one(self):
        """The schema version is the literal string '1' (not int, not float).

        Downstream consumers compare against the string; a silent type drift
        to int would break ``payload['schema_version'] == '1'`` comparisons.
        """
        from backpropagate.cli import RUNS_JSON_SCHEMA_VERSION

        assert isinstance(RUNS_JSON_SCHEMA_VERSION, str)
        assert RUNS_JSON_SCHEMA_VERSION == "1"

    def test_populated_runs_projection_pins_field_names(self, tmp_path):
        """Each entry in the projected list has the documented field set.

        Field renames in ``RunHistoryManager`` MUST be re-projected here
        explicitly — that is the load-bearing decoupling between the
        on-disk format and the UI contract. This test pins every field.
        """
        from backpropagate.cli import _build_runs_payload

        raw = [{
            "run_id": "abc12345-deadbeef",
            "status": "completed",
            "session_kind": "single",
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "dataset_info": "ultrachat",
            "started_at": "2026-05-23T10:00:00+00:00",
            "completed_at": "2026-05-23T10:30:00+00:00",
            "duration_seconds": 1800.0,
            "final_loss": 0.5,
            "loss_history": [1.0, 0.8, 0.6, 0.5],
            "checkpoint_path": str(tmp_path / "ckpt"),
        }]

        payload = _build_runs_payload(raw, tmp_path)
        assert len(payload["runs"]) == 1
        proj = payload["runs"][0]

        # Every documented field must be present with the projected name.
        assert proj["run_id"] == "abc12345-deadbeef"
        assert proj["status"] == "completed"
        assert proj["session_kind"] == "single"
        assert proj["model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert proj["dataset"] == "ultrachat"
        assert proj["duration_seconds"] == 1800.0
        assert proj["started_at"] == "2026-05-23T10:00:00+00:00"
        assert proj["completed_at"] == "2026-05-23T10:30:00+00:00"
        assert proj["checkpoint_path"] == str(tmp_path / "ckpt")
        assert proj["loss"]["final"] == 0.5
        assert proj["loss"]["min"] == 0.5  # from loss_history

    def test_duration_computed_from_timestamps_when_missing(self, tmp_path):
        """When ``duration_seconds`` is absent, it is computed from started/completed."""
        from backpropagate.cli import _build_runs_payload

        raw = [{
            "run_id": "r1",
            "started_at": "2026-05-23T10:00:00+00:00",
            "completed_at": "2026-05-23T10:01:30+00:00",
        }]

        payload = _build_runs_payload(raw, tmp_path)
        assert payload["runs"][0]["duration_seconds"] == 90.0

    def test_min_loss_extracted_from_loss_history(self, tmp_path):
        """``loss.min`` reflects the smallest numeric value in ``loss_history``."""
        from backpropagate.cli import _build_runs_payload

        raw = [{
            "run_id": "r1",
            "loss_history": [2.0, 0.3, 1.0, 0.5, 1.5],
            "final_loss": 1.5,
        }]

        payload = _build_runs_payload(raw, tmp_path)
        assert payload["runs"][0]["loss"]["min"] == 0.3
        assert payload["runs"][0]["loss"]["final"] == 1.5

    def test_invalid_loss_history_entries_dont_crash(self, tmp_path):
        """Non-numeric entries in loss_history are skipped, not propagated as errors."""
        from backpropagate.cli import _build_runs_payload

        raw = [{
            "run_id": "r1",
            "loss_history": [None, "nan", 0.5, {"bad": True}, 0.2],
        }]

        # Must not raise.
        payload = _build_runs_payload(raw, tmp_path)
        # min should still come from the two numeric values
        assert payload["runs"][0]["loss"]["min"] == 0.2

    def test_missing_optional_fields_become_none(self, tmp_path):
        """Entries with no loss/duration data don't crash and produce None fields."""
        from backpropagate.cli import _build_runs_payload

        raw = [{"run_id": "minimal"}]
        payload = _build_runs_payload(raw, tmp_path)
        proj = payload["runs"][0]
        assert proj["loss"]["final"] is None
        assert proj["loss"]["min"] is None
        assert proj["duration_seconds"] is None


# =============================================================================
# cmd_runs — CLI end-to-end (no subprocess; calls cmd_runs directly)
# =============================================================================


def _seed_history(output_dir: Path, entries: list[dict]) -> None:
    """Write a run_history.json that RunHistoryManager can read back."""
    output_dir.mkdir(parents=True, exist_ok=True)
    history_file = output_dir / "run_history.json"
    history_file.write_text(json.dumps(entries), encoding="utf-8")


class TestCmdRuns:
    """End-to-end tests of ``backprop runs`` via cmd_runs(args)."""

    def test_empty_output_dir_emits_empty_json_payload(self, tmp_path):
        """``backprop runs --json`` against a non-existent dir emits empty payload."""
        from backpropagate.cli import EXIT_OK, cmd_runs

        nonexistent = tmp_path / "no-history"
        args = argparse.Namespace(
            output=str(nonexistent),
            status=None,
            limit=None,
            json=True,
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            result = cmd_runs(args)
        assert result == EXIT_OK

        payload = json.loads(buf.getvalue())
        assert payload["schema_version"] == "1"
        assert payload["runs"] == []

    def test_populated_runs_in_json_payload(self, tmp_path):
        """``backprop runs --json`` lists every entry from run_history.json."""
        from backpropagate.cli import EXIT_OK, cmd_runs

        _seed_history(tmp_path, [
            {
                "run_id": "alpha",
                "status": "completed",
                "model_name": "m1",
                "started_at": "2026-05-23T10:00:00+00:00",
                "final_loss": 0.1,
            },
            {
                "run_id": "beta",
                "status": "running",
                "model_name": "m2",
                "started_at": "2026-05-23T11:00:00+00:00",
            },
        ])
        args = argparse.Namespace(
            output=str(tmp_path),
            status=None,
            limit=None,
            json=True,
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            result = cmd_runs(args)
        assert result == EXIT_OK

        payload = json.loads(buf.getvalue())
        assert len(payload["runs"]) == 2
        ids = {r["run_id"] for r in payload["runs"]}
        assert ids == {"alpha", "beta"}

    def test_status_filter_narrows_results(self, tmp_path):
        """``--status completed`` filters out non-matching runs."""
        from backpropagate.cli import EXIT_OK, cmd_runs

        _seed_history(tmp_path, [
            {"run_id": "a", "status": "completed", "started_at": "2026-05-23T10:00:00+00:00"},
            {"run_id": "b", "status": "running", "started_at": "2026-05-23T11:00:00+00:00"},
            {"run_id": "c", "status": "completed", "started_at": "2026-05-23T09:00:00+00:00"},
        ])
        args = argparse.Namespace(
            output=str(tmp_path),
            status="completed",
            limit=None,
            json=True,
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            result = cmd_runs(args)
        assert result == EXIT_OK

        payload = json.loads(buf.getvalue())
        statuses = [r["status"] for r in payload["runs"]]
        assert statuses == ["completed", "completed"]

    def test_runs_sorted_newest_first(self, tmp_path):
        """RunHistoryManager.list_runs sorts newest-first; cmd_runs preserves order."""
        from backpropagate.cli import EXIT_OK, cmd_runs

        _seed_history(tmp_path, [
            {"run_id": "old", "status": "completed", "started_at": "2026-01-01T00:00:00+00:00"},
            {"run_id": "new", "status": "completed", "started_at": "2026-12-31T00:00:00+00:00"},
            {"run_id": "mid", "status": "completed", "started_at": "2026-06-15T00:00:00+00:00"},
        ])
        args = argparse.Namespace(
            output=str(tmp_path),
            status=None,
            limit=None,
            json=True,
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            result = cmd_runs(args)
        assert result == EXIT_OK

        payload = json.loads(buf.getvalue())
        run_order = [r["run_id"] for r in payload["runs"]]
        assert run_order == ["new", "mid", "old"]

    def test_limit_cap_applies_to_payload(self, tmp_path):
        """``--limit 2`` caps the result set to the 2 newest entries.

        TESTS-B-017 (v1.3 Wave 3 Stage C humanization): the module docstring
        claims '--limit cap' coverage but every prior test passed limit=None,
        so a regression in the limit-cap logic would not be caught. This
        test seeds 5 entries, calls with limit=2, and asserts the payload
        contains exactly the 2 newest IDs in the documented sort order.
        """
        from backpropagate.cli import EXIT_OK, cmd_runs

        _seed_history(tmp_path, [
            {"run_id": "alpha", "status": "completed", "started_at": "2026-01-01T00:00:00+00:00"},
            {"run_id": "bravo", "status": "completed", "started_at": "2026-02-01T00:00:00+00:00"},
            {"run_id": "charlie", "status": "completed", "started_at": "2026-03-01T00:00:00+00:00"},
            {"run_id": "delta", "status": "completed", "started_at": "2026-04-01T00:00:00+00:00"},
            {"run_id": "echo", "status": "completed", "started_at": "2026-05-01T00:00:00+00:00"},
        ])
        args = argparse.Namespace(
            output=str(tmp_path),
            status=None,
            limit=2,
            json=True,
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            result = cmd_runs(args)
        assert result == EXIT_OK

        payload = json.loads(buf.getvalue())
        assert len(payload["runs"]) == 2, (
            f"--limit 2 should cap result to 2 entries, got {len(payload['runs'])}"
        )
        # Newest-first: echo (May), delta (April).
        ids = [r["run_id"] for r in payload["runs"]]
        assert ids == ["echo", "delta"], (
            f"--limit 2 should keep the 2 newest, got {ids}"
        )


# =============================================================================
# RunsState — Reflex state handler
# =============================================================================


class TestRunsState:
    """Pin the FRONTEND-F-RUN-HISTORY-PAGE state contract."""

    def test_initial_state_defaults(self):
        """Fresh RunsState has empty/zero defaults."""
        reflex = pytest.importorskip(
            "reflex",
            reason="reflex is required for ui_state tests (install backpropagate[ui])",
        )
        # ``reflex`` import in pyproject is gated by the [ui] extra; importorskip
        # gracefully handles a headless install. ``reflex`` is referenced only to
        # trigger the importorskip; the actual import comes from ui_state itself.
        del reflex
        from backpropagate.ui_state import RunsState

        state = RunsState()
        assert state.runs == []
        assert state.loading is False
        assert state.error == ""
        assert state.status_filter == ""
        assert state.output_dir_override == ""
        assert state.last_loaded_at == ""

    def test_load_runs_against_empty_history_sets_error(self, tmp_path):
        """When the history directory is missing, ``load_runs`` populates ``error``."""
        pytest.importorskip(
            "reflex",
            reason="reflex is required for ui_state tests",
        )
        from backpropagate.ui_state import RunsState

        state = RunsState()
        state.output_dir_override = str(tmp_path / "nope")
        state.load_runs()
        assert state.runs == []
        assert "No run history" in state.error
        assert state.loading is False  # Always reset in finally:

    def test_load_runs_populates_from_history(self, tmp_path):
        """A populated history directory loads + projects entries into state."""
        pytest.importorskip(
            "reflex",
            reason="reflex is required for ui_state tests",
        )
        from backpropagate.ui_state import RunsState

        _seed_history(tmp_path, [
            {
                "run_id": "deadbeefdeadbeef",
                "status": "completed",
                "model": "qwen-7b",
                "dataset": "ultrachat",
                "started_at": "2026-05-23T10:00:00+00:00",
                "duration_seconds": 120.0,
                "final_loss": 0.42,
            },
        ])

        state = RunsState()
        state.output_dir_override = str(tmp_path)
        state.load_runs()

        assert state.error == ""
        assert state.loading is False
        assert len(state.runs) == 1
        row = state.runs[0]
        assert row["run_id"] == "deadbeefdeadbeef"
        assert row["run_id_short"] == "deadbeef"  # truncated to 8 chars
        assert row["model"] == "qwen-7b"
        assert row["dataset"] == "ultrachat"
        assert row["status"] == "completed"
        assert row["duration"] == "120s"
        assert row["final_loss"] == "0.4200"

    def test_set_status_filter_accepts_valid_values(self, tmp_path):
        """set_status_filter accepts the documented enum; rejects others silently."""
        pytest.importorskip(
            "reflex",
            reason="reflex is required for ui_state tests",
        )
        from backpropagate.ui_state import RunsState

        state = RunsState()
        # Point at a non-existent dir so load_runs short-circuits to the
        # error-banner path — we're not testing the load here, just the filter.
        state.output_dir_override = str(tmp_path / "no-history")
        state.set_status_filter("completed")
        assert state.status_filter == "completed"
        state.set_status_filter("running")
        assert state.status_filter == "running"

        # Invalid filter — should be silently rejected (state.status_filter
        # stays at the prior valid value).
        state.set_status_filter("garbage")
        assert state.status_filter == "running"

    def test_clear_error_dismisses_banner(self):
        """``clear_error`` resets the error string."""
        pytest.importorskip(
            "reflex",
            reason="reflex is required for ui_state tests",
        )
        from backpropagate.ui_state import RunsState

        state = RunsState()
        state.error = "something went wrong"
        state.clear_error()
        assert state.error == ""


# =============================================================================
# BACKEND-B-003 ENTRY-LEVEL schema_version MISMATCH WARN PATH
# =============================================================================


class TestRunHistoryEntrySchemaVersionWarn:
    """Pin BACKEND-B-003 entry-level schema_version mismatch contract.

    v1.3 added ``RunHistoryManager.CURRENT_ENTRY_SCHEMA_VERSION`` (string
    "1.0") plus a "warn-once-per-unique-version" loop in ``_load`` at
    backpropagate/checkpoints.py:983-1000. Coverage gap (Wave 3.5
    TESTS-B-004): the entry-level WARN was never exercised. Every
    existing test seeded entries WITHOUT a ``schema_version`` key OR
    with no entries at all, so the warn-once branch was dead-coverage.

    These tests pin:

      1. Legacy v1.0 entries lacking ``schema_version`` AND
         ``session_kind`` (the brief-named "session_kind=None" scenario)
         load without raising AND fire the missing-version WARN exactly
         once via the "0.0" implicit baseline.
      2. Future-version entries (``schema_version='99.0'``) emit the
         same WARN and load anyway (fail-loud-but-keep-going).
      3. Warn-once dedup works: two entries at the SAME mismatched
         version produce ONE WARN, not two (the ``seen_versions`` set
         in the source).
      4. Mixed-version entries (one legacy, one future) emit TWO WARNs
         — one per unique mismatched version.

    Why this matters: operators correlate weird ``backprop runs``
    output to schema age via these log lines. If someone silently
    drops the WARN (refactor, log-level demotion to DEBUG) the
    mixed-version diagnosability disappears and v1.2 → v1.3 migration
    bugs become invisible.
    """

    def _seed(self, output_dir: Path, entries: list[dict]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_history.json").write_text(
            json.dumps(entries), encoding="utf-8"
        )

    def test_legacy_entry_missing_schema_and_session_kind_loads_with_warn(
        self, tmp_path, caplog
    ):
        """BACKEND-B-003: a pre-v1.3 entry (no schema_version, no
        session_kind) must load AND fire the warn-once line.

        This is the brief's "legacy v1.0 run-history entry" scenario —
        an entry written by an older build that never grew the
        schema_version field. The source defaults missing version to
        "0.0" so the operator sees the gap on the next list_runs call.
        """
        from backpropagate.checkpoints import RunHistoryManager

        self._seed(tmp_path, [
            {
                # No schema_version, no session_kind — legacy v1.0 shape
                "run_id": "legacy-run-001",
                "status": "completed",
                "model_name": "qwen-7b",
                "started_at": "2026-01-01T00:00:00+00:00",
            }
        ])

        with caplog.at_level("WARNING", logger="backpropagate.checkpoints"):
            mgr = RunHistoryManager(str(tmp_path))
            runs = mgr.list_runs()

        assert len(runs) == 1, (
            "BACKEND-B-003 contract: legacy entries (no schema_version) "
            "MUST load non-fatally. Got "
            f"{len(runs)} runs instead of 1. If a future refactor wants "
            "to refuse legacy entries, ship a real migrator first."
        )
        # session_kind missing is tolerated — no crash on access through
        # the public list_runs surface
        assert runs[0]["run_id"] == "legacy-run-001"

        version_warns = [
            r for r in caplog.records
            if r.levelname == "WARNING"
            and "schema_version" in r.getMessage()
        ]
        assert len(version_warns) == 1, (
            "BACKEND-B-003 warn-once contract violated: legacy entry "
            "(missing schema_version, defaults to '0.0') must emit "
            f"exactly one WARN. Got {len(version_warns)} WARN records. "
            f"All warns: {[r.getMessage() for r in caplog.records]!r}"
        )
        msg = version_warns[0].getMessage()
        assert "0.0" in msg, (
            f"Legacy-entry WARN must surface the '0.0' fallback so "
            f"operators recognize a pre-anchor row; got: {msg!r}"
        )

    def test_future_version_entry_loads_with_warn(self, tmp_path, caplog):
        """BACKEND-B-003: future schema_version ('99.0') WARNs but loads.

        Forward-compat is best-effort: an entry written by a newer
        build than this one is parsed with field-defaults and surfaces
        a WARN so the operator knows downgrade-on-read happened.
        """
        from backpropagate.checkpoints import RunHistoryManager

        self._seed(tmp_path, [
            {
                "schema_version": "99.0",
                "run_id": "future-run-002",
                "status": "completed",
                "model_name": "qwen-7b",
                "started_at": "2026-05-01T00:00:00+00:00",
                "session_kind": "single",
            }
        ])

        with caplog.at_level("WARNING", logger="backpropagate.checkpoints"):
            mgr = RunHistoryManager(str(tmp_path))
            runs = mgr.list_runs()

        assert len(runs) == 1, (
            "BACKEND-B-003 forward-compat contract: future-version "
            f"entries must load. Got {len(runs)} instead of 1."
        )
        version_warns = [
            r for r in caplog.records
            if r.levelname == "WARNING"
            and "schema_version" in r.getMessage()
        ]
        assert len(version_warns) == 1, (
            "Future-version entry must emit exactly one mismatch WARN; "
            f"got {len(version_warns)}."
        )
        msg = version_warns[0].getMessage()
        assert "99.0" in msg, (
            f"Future-version WARN must name the on-disk version so "
            f"operators see the gap direction; got: {msg!r}"
        )

    def test_warn_once_dedup_two_entries_same_mismatched_version(
        self, tmp_path, caplog
    ):
        """BACKEND-B-003: ``seen_versions`` dedups WARNs per unique version.

        Pins the ``ver not in seen_versions: seen_versions.add(ver)``
        guard at checkpoints.py:992-993. Two legacy entries should
        produce ONE WARN, not two — otherwise an old history file with
        hundreds of legacy rows floods the log on every list call.
        """
        from backpropagate.checkpoints import RunHistoryManager

        self._seed(tmp_path, [
            {
                "run_id": "legacy-a",
                "status": "completed",
                "model_name": "m",
                "started_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "run_id": "legacy-b",
                "status": "completed",
                "model_name": "m",
                "started_at": "2026-01-02T00:00:00+00:00",
            },
        ])

        with caplog.at_level("WARNING", logger="backpropagate.checkpoints"):
            mgr = RunHistoryManager(str(tmp_path))
            mgr.list_runs()

        version_warns = [
            r for r in caplog.records
            if r.levelname == "WARNING"
            and "schema_version" in r.getMessage()
        ]
        assert len(version_warns) == 1, (
            "BACKEND-B-003 warn-once dedup violated: two legacy entries "
            "at the same implicit '0.0' version must produce ONE WARN "
            f"(the seen_versions set should dedup). Got {len(version_warns)}. "
            "Without dedup, a 100-row legacy history floods the log."
        )

    def test_mixed_version_entries_warn_per_unique_version(
        self, tmp_path, caplog
    ):
        """BACKEND-B-003: one legacy + one future ⇒ TWO distinct WARNs.

        Confirms the dedup is PER UNIQUE VERSION, not "warn at most
        once total." A mixed-version writer scenario (v1.2 wrote a
        row, v1.99 wrote another, v1.3 reads both) should surface
        BOTH version gaps so the operator sees the full picture.
        """
        from backpropagate.checkpoints import RunHistoryManager

        self._seed(tmp_path, [
            {
                # Legacy — no schema_version, will fall back to "0.0"
                "run_id": "legacy-run",
                "status": "completed",
                "model_name": "m",
                "started_at": "2026-01-01T00:00:00+00:00",
            },
            {
                # Future version
                "schema_version": "99.0",
                "run_id": "future-run",
                "status": "completed",
                "model_name": "m",
                "started_at": "2026-01-02T00:00:00+00:00",
                "session_kind": "single",
            },
        ])

        with caplog.at_level("WARNING", logger="backpropagate.checkpoints"):
            mgr = RunHistoryManager(str(tmp_path))
            mgr.list_runs()

        version_warns = [
            r for r in caplog.records
            if r.levelname == "WARNING"
            and "schema_version" in r.getMessage()
        ]
        msgs = [r.getMessage() for r in version_warns]
        assert len(version_warns) == 2, (
            "BACKEND-B-003: mixed-version entries must produce one WARN "
            f"PER UNIQUE mismatched version (here '0.0' and '99.0'). "
            f"Got {len(version_warns)} WARNs. Messages: {msgs!r}"
        )
        # Both unique versions named across the warn set
        combined = " ".join(msgs)
        assert "0.0" in combined, (
            f"Legacy version '0.0' missing from warn set; got: {msgs!r}"
        )
        assert "99.0" in combined, (
            f"Future version '99.0' missing from warn set; got: {msgs!r}"
        )

    def test_current_version_entry_is_silent_no_warn(
        self, tmp_path, caplog
    ):
        """BACKEND-B-003 happy-path: matching schema_version emits NO WARN.

        Pinning the silent half so a future refactor that fires WARN on
        every load is caught — flood = signal-blindness.
        """
        from backpropagate.checkpoints import RunHistoryManager

        self._seed(tmp_path, [
            {
                "schema_version": RunHistoryManager.CURRENT_ENTRY_SCHEMA_VERSION,
                "run_id": "current-run",
                "status": "completed",
                "model_name": "m",
                "started_at": "2026-05-01T00:00:00+00:00",
                "session_kind": "single",
            }
        ])

        with caplog.at_level("WARNING", logger="backpropagate.checkpoints"):
            mgr = RunHistoryManager(str(tmp_path))
            mgr.list_runs()

        version_warns = [
            r for r in caplog.records
            if r.levelname == "WARNING"
            and "schema_version" in r.getMessage()
        ]
        assert not version_warns, (
            "BACKEND-B-003 happy-path: an entry at the current "
            f"schema_version='{RunHistoryManager.CURRENT_ENTRY_SCHEMA_VERSION}' "
            f"must NOT emit a mismatch WARN. Got: "
            f"{[r.getMessage() for r in version_warns]!r}. "
            "Spurious WARNs on every load bury the real signal."
        )
