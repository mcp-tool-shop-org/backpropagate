"""Coverage tests for ``RunHistoryManager`` (backpropagate/checkpoints.py).

Real filesystem and real ``filelock``. Mock boundary: OS failure injection
(``os.replace`` / ``os.fsync`` / ``Path.unlink`` raising) for disk-full and
locked-file conditions; lock contention is produced by holding the manager's
``.lock`` file from a second ``FileLock`` handle.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from filelock import FileLock

from backpropagate import checkpoints as ckpt_mod
from backpropagate.checkpoints import RunHistoryManager

LOGGER = "backpropagate.checkpoints"


@pytest.fixture
def hist(tmp_path):
    return RunHistoryManager(str(tmp_path / "out"))


def on_disk(h: RunHistoryManager):
    return json.loads(h._history_path.read_text())


def fail_replace(monkeypatch):
    def fake(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(ckpt_mod.os, "replace", fake)


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

class TestHistoryLocking:
    def test_without_filelock_mutators_still_persist(self, hist, monkeypatch, caplog):
        monkeypatch.setattr(ckpt_mod, "_FILELOCK_AVAILABLE", False)
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            hist.record_run_started("r1", model_name="m")
        assert on_disk(hist)[0]["run_id"] == "r1"
        assert "RunHistoryManager._locked_mutate(record_run_started): filelock unavailable" in caplog.text

    def test_zero_timeout_blocks_forever_but_uncontended_works(self, tmp_path):
        h = RunHistoryManager(str(tmp_path / "z"), lock_timeout_seconds=0)
        h.record_run_started("r1")
        assert h._lock_timeout_seconds == 0.0
        assert [e["run_id"] for e in on_disk(h)] == ["r1"]

    def test_lock_timeout_logs_and_mutation_still_proceeds(self, tmp_path, caplog):
        h = RunHistoryManager(str(tmp_path / "t"), lock_timeout_seconds=0.05)
        with FileLock(str(h._lock_path)), caplog.at_level(logging.ERROR, logger=LOGGER):
            h.record_run_started("r1")
            h.record_run_completed("r1", final_loss=0.5)
            assert h.update_run("r1", note="x")["note"] == "x"
            assert h.delete_run("r1") is True
        assert caplog.text.count("lock acquisition timed out") == 4
        for op in ("record_run_started", "record_run_completed", "update_run", "delete_run"):
            assert f"_locked_mutate({op})" in caplog.text
        assert on_disk(h) == []


# ---------------------------------------------------------------------------
# _load / _save / downsample
# ---------------------------------------------------------------------------

class TestLoad:
    def test_missing_file_is_empty(self, hist):
        assert hist.get_history() == []

    def test_schema_mismatch_warned_once_per_version_and_non_dict_skipped(self, hist, caplog):
        hist._history_path.write_text(json.dumps([
            {"run_id": "a", "schema_version": "9.9"},
            {"run_id": "b", "schema_version": "9.9"},
            {"run_id": "c"},                      # implicit 0.0
            "not-a-dict",
            {"run_id": "d", "schema_version": "1.0"},
        ]))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            data = hist.get_history()
        assert [e["run_id"] for e in data if isinstance(e, dict)] == ["a", "b", "c", "d"]
        assert caplog.text.count("schema_version='9.9'") == 1
        assert caplog.text.count("schema_version='0.0'") == 1
        assert "schema_version='1.0'" not in caplog.text

    def test_non_array_json_resets_to_empty(self, hist, caplog):
        hist._history_path.write_text('{"a": 1}')
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert hist.get_history() == []
        assert "not a JSON array" in caplog.text

    def test_corrupt_json_resets_to_empty(self, hist, caplog):
        hist._history_path.write_text("[{broken")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert hist.get_history() == []
        assert "Failed to load run history" in caplog.text


class TestSave:
    def test_fsync_unsupported_is_tolerated(self, hist, monkeypatch, caplog):
        def no_fsync(fd):
            raise OSError("unsupported")

        monkeypatch.setattr(ckpt_mod.os, "fsync", no_fsync)
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert hist._save([{"run_id": "x"}]) is True
        assert "fsync skipped" in caplog.text
        assert on_disk(hist) == [{"run_id": "x"}]

    def test_replace_failure_returns_false_cleans_tmp(self, hist, monkeypatch, caplog):
        fail_replace(monkeypatch)
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            assert hist._save([{"run_id": "x"}]) is False
        assert "Failed to save run history: disk full" in caplog.text
        assert not hist._history_path.with_suffix(".json.tmp").exists()

    def test_tmp_cleanup_failure_is_swallowed(self, hist, monkeypatch):
        fail_replace(monkeypatch)
        real_unlink = Path.unlink

        def deny(self, *a, **k):
            if self.name.endswith(".tmp"):
                raise PermissionError("held")
            return real_unlink(self, *a, **k)

        monkeypatch.setattr(Path, "unlink", deny)
        assert hist._save([]) is False

    def test_unserialisable_payload_returns_false_without_tmp_leftover_error(self, hist, caplog):
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            assert hist._save([{"bad": object()}]) is False
        assert "Failed to save run history" in caplog.text
        assert not hist._history_path.exists()


class TestDownsample:
    def test_short_history_is_copied_unchanged(self):
        src = [1.0, 2.0, 3.0]
        out = RunHistoryManager._downsample_loss_history(src, 5)
        assert out == src and out is not src

    def test_long_history_sampled_uniformly_to_max_points(self):
        src = [float(i) for i in range(1000)]
        out = RunHistoryManager._downsample_loss_history(src, 100)
        assert len(out) == 100
        assert out[0] == 0.0 and out[1] == 10.0 and out[-1] == 990.0


# ---------------------------------------------------------------------------
# record_run + lookups
# ---------------------------------------------------------------------------

class TestRecordRun:
    def test_normalises_entry_adds_schema_and_timestamp(self, hist):
        entry = hist.record_run({"run_id": "r1", "final_loss": 0.4, "steps": 10})
        assert set(RunHistoryManager._EXPECTED_FIELDS) <= set(entry)
        assert entry["schema_version"] == "1.0"
        assert entry["timestamp"] is not None
        assert entry["model_name"] is None
        assert on_disk(hist) == [entry]

    def test_preserves_explicit_timestamp_and_downsamples_losses(self, hist):
        entry = hist.record_run({"run_id": "r1", "timestamp": "2026-01-01T00:00:00",
                                 "loss_history": [float(i) for i in range(500)]})
        assert entry["timestamp"] == "2026-01-01T00:00:00"
        assert len(entry["loss_history"]) == RunHistoryManager.MAX_LOSS_HISTORY_POINTS

    def test_non_list_loss_history_left_alone(self, hist):
        entry = hist.record_run({"run_id": "r1", "loss_history": None})
        assert entry["loss_history"] is None

    def test_save_failure_warns_but_returns_entry_and_fires_callback(
        self, tmp_path, monkeypatch, caplog
    ):
        seen = []
        h = RunHistoryManager(str(tmp_path / "cb"), on_record_callback=seen.append)
        fail_replace(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            entry = h.record_run({"run_id": "r1"})
        assert "Run history save failed" in caplog.text
        assert seen == [entry]

    def test_callback_failure_is_swallowed_with_warning(self, tmp_path, caplog):
        def bad(entry):
            raise ValueError("sink down")

        h = RunHistoryManager(str(tmp_path / "cb2"), on_record_callback=bad)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            entry = h.record_run({"run_id": "r1"})
        assert "on_record_callback raised (ValueError: sink down)" in caplog.text
        assert on_disk(h)[0]["run_id"] == entry["run_id"]  # persisted regardless

    def test_every_lifecycle_call_fires_callback(self, tmp_path):
        seen = []
        h = RunHistoryManager(str(tmp_path / "cb3"), on_record_callback=seen.append)
        h.record_run_started("a")
        h.record_run_completed("a", final_loss=1.0)
        h.record_run_started("b")
        h.record_run_failed("b", "oom")
        assert [(e["run_id"], e["status"]) for e in seen] == [
            ("a", "running"), ("a", "completed"), ("b", "running"), ("b", "failed")]


class TestBestRun:
    def test_none_when_no_runs_or_no_losses(self, hist):
        assert hist.get_best_run() is None
        hist.record_run({"run_id": "a"})
        assert hist.get_best_run() is None

    def test_returns_lowest_final_loss_skipping_none(self, hist):
        hist.record_run({"run_id": "a", "final_loss": 0.9})
        hist.record_run({"run_id": "b", "final_loss": 0.3})
        hist.record_run({"run_id": "c"})
        assert hist.get_best_run()["run_id"] == "b"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycleStarted:
    def test_started_entry_shape_and_idempotent_replace(self, hist):
        e1 = hist.record_run_started("r1", model_name="m", dataset_info={"n": 3},
                                    hyperparameters={"lr": 1e-4}, session_kind="multi_run",
                                    checkpoint_path="/c", dataset_hash="abc")
        assert e1["status"] == "running" and e1["session_kind"] == "multi_run"
        assert e1["hyperparameters"] == {"lr": 1e-4}
        assert e1["loss_history"] == [] and e1["merge_history"] == [] and e1["export_paths"] == []
        hist.record_run_started("r1", model_name="m2")
        rows = on_disk(hist)
        assert len(rows) == 1 and rows[0]["model_name"] == "m2"
        assert rows[0]["hyperparameters"] == {}

    def test_save_failure_warns(self, hist, monkeypatch, caplog):
        fail_replace(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            e = hist.record_run_started("r1")
        assert "record_run_started: save failed" in caplog.text
        assert e["run_id"] == "r1"


class TestLifecycleCompleted:
    def test_updates_every_provided_field(self, hist):
        hist.record_run_started("r1", model_name="m")
        e = hist.record_run_completed(
            "r1", final_loss=0.2, loss_history=[float(i) for i in range(300)], steps=300,
            duration_seconds=12.5, gpu_max_temp=71.0, checkpoint_path="/ckpt",
            merge_history=[{"merge": 1}], extra={"custom": "yes"},
        )
        row = on_disk(hist)[0]
        assert row == e
        assert row["status"] == "completed" and row["completed_at"]
        assert (row["final_loss"], row["steps"], row["duration_seconds"]) == (0.2, 300, 12.5)
        assert row["gpu_max_temp"] == 71.0 and row["checkpoint_path"] == "/ckpt"
        assert row["merge_history"] == [{"merge": 1}] and row["custom"] == "yes"
        assert len(row["loss_history"]) == 100
        assert row["model_name"] == "m"  # untouched

    def test_omitted_fields_keep_started_values(self, hist):
        hist.record_run_started("r1", checkpoint_path="/orig")
        hist.record_run_completed("r1")
        row = on_disk(hist)[0]
        assert row["status"] == "completed"
        assert row["checkpoint_path"] == "/orig"
        assert row["final_loss"] is None and row["steps"] is None

    def test_unmatched_run_is_synthesised_with_warning(self, hist, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            e = hist.record_run_completed("ghost", final_loss=0.7, steps=5,
                                          loss_history=[1.0, 0.7], extra={"k": "v"},
                                          merge_history=[{"m": 1}], checkpoint_path="/p",
                                          duration_seconds=1.0, gpu_max_temp=60.0)
        assert "no started record for run_id=ghost" in caplog.text
        assert e["status"] == "completed" and e["k"] == "v"
        assert e["loss_history"] == [1.0, 0.7] and e["merge_history"] == [{"m": 1}]
        assert e["schema_version"] == "1.0" and e["export_paths"] == []
        assert on_disk(hist) == [e]

    def test_unmatched_without_optional_inputs_gets_empty_lists(self, hist):
        e = hist.record_run_completed("ghost")
        assert e["loss_history"] == [] and e["merge_history"] == []

    def test_save_failure_warns(self, hist, monkeypatch, caplog):
        hist.record_run_started("r1")
        fail_replace(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            hist.record_run_completed("r1", final_loss=1.0)
        assert "record_run_completed: save failed" in caplog.text


class TestLifecycleFailed:
    def test_matched_failure_updates_fields(self, hist):
        hist.record_run_started("r1")
        e = hist.record_run_failed(
            "r1", "CUDA OOM", loss_history=[float(i) for i in range(250)],
            duration_seconds=3.0, checkpoint_path="/last", extra={"oom_at": 7},
        )
        row = on_disk(hist)[0]
        assert row == e
        assert row["status"] == "failed" and row["failure_reason"] == "CUDA OOM"
        assert row["duration_seconds"] == 3.0 and row["checkpoint_path"] == "/last"
        assert row["oom_at"] == 7 and len(row["loss_history"]) == 100
        assert row["completed_at"]

    def test_matched_failure_without_options_leaves_other_fields(self, hist):
        hist.record_run_started("r1", checkpoint_path="/orig")
        hist.record_run_failed("r1", "boom")
        row = on_disk(hist)[0]
        assert row["checkpoint_path"] == "/orig" and row["duration_seconds"] is None

    def test_unmatched_failure_synthesised(self, hist, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            e = hist.record_run_failed("ghost", "died", loss_history=[2.0],
                                       duration_seconds=1.5, checkpoint_path="/c",
                                       extra={"why": "power"})
        assert "no started record for run_id=ghost" in caplog.text
        assert e["status"] == "failed" and e["failure_reason"] == "died"
        assert e["why"] == "power" and e["loss_history"] == [2.0]
        assert on_disk(hist) == [e]

    def test_unmatched_failure_without_options(self, hist):
        e = hist.record_run_failed("ghost", "died")
        assert e["loss_history"] == [] and e["merge_history"] == []

    def test_save_failure_warns(self, hist, monkeypatch, caplog):
        hist.record_run_started("r1")
        fail_replace(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            hist.record_run_failed("r1", "x")
        assert "record_run_failed: save failed" in caplog.text


class TestUpdateRun:
    def test_no_fields_returns_current_entry(self, hist):
        hist.record_run_started("r1")
        assert hist.update_run("r1")["run_id"] == "r1"
        assert hist.update_run("nope") is None

    def test_patch_persists(self, hist):
        hist.record_run_started("r1")
        out = hist.update_run("r1", status="running", export_paths=["/a"])
        assert out["export_paths"] == ["/a"]
        assert on_disk(hist)[0]["export_paths"] == ["/a"]

    def test_unknown_run_warns_and_returns_none(self, hist, caplog):
        hist.record_run_started("r1")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert hist.update_run("zzz", status="failed") is None
        assert "update_run: run_id=zzz not found" in caplog.text

    def test_skips_non_matching_entries_before_match(self, hist):
        hist.record_run_started("a")
        hist.record_run_started("b")
        hist.update_run("b", note=1)
        rows = {r["run_id"]: r for r in on_disk(hist)}
        assert rows["b"]["note"] == 1 and "note" not in rows["a"]

    def test_save_failure_warns_but_returns_entry(self, hist, monkeypatch, caplog):
        hist.record_run_started("r1")
        fail_replace(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            out = hist.update_run("r1", note="x")
        assert out["note"] == "x"
        assert "update_run: save failed" in caplog.text


class TestGetRun:
    def test_exact_prefix_unique_ambiguous_and_missing(self, hist):
        for rid in ("abc12345", "abd99999", "zzz00000"):
            hist.record_run_started(rid)
        assert hist.get_run("abc12345")["run_id"] == "abc12345"
        assert hist.get_run("zzz")["run_id"] == "zzz00000"
        assert hist.get_run("ab") is None          # ambiguous prefix
        assert hist.get_run("nothing") is None

    def test_non_string_run_ids_never_prefix_match(self, hist):
        hist._save([{"run_id": 123}, {"run_id": None}, {"run_id": "x1"}])
        assert hist.get_run("x")["run_id"] == "x1"
        assert hist.get_run("1") is None


class TestListRuns:
    def _seed(self, hist):
        hist._save([
            {"run_id": "old", "status": "completed", "started_at": "2026-01-01T00:00:00"},
            {"run_id": "new", "status": "running", "started_at": "2026-03-01T00:00:00"},
            {"run_id": "mid", "status": "failed", "timestamp": "2026-02-01T00:00:00"},
            {"run_id": "undated", "status": "completed"},
        ])

    def test_newest_first_with_timestamp_fallback_and_undated_last(self, hist):
        self._seed(hist)
        assert [r["run_id"] for r in hist.list_runs()] == ["new", "mid", "old", "undated"]

    def test_status_filter_and_limit(self, hist):
        self._seed(hist)
        assert [r["run_id"] for r in hist.list_runs(status="completed")] == ["old", "undated"]
        assert [r["run_id"] for r in hist.list_runs(limit=2)] == ["new", "mid"]
        assert len(hist.list_runs(limit=0)) == 4  # non-positive limit = no cap

    def test_invalid_status_raises_value_error(self, hist):
        with pytest.raises(ValueError, match="Invalid status 'bogus'"):
            hist.list_runs(status="bogus")


class TestDeleteRun:
    def test_delete_existing(self, hist):
        hist.record_run_started("a")
        hist.record_run_started("b")
        assert hist.delete_run("a") is True
        assert [r["run_id"] for r in on_disk(hist)] == ["b"]

    def test_delete_missing_returns_false(self, hist, caplog):
        hist.record_run_started("a")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert hist.delete_run("zzz") is False
        assert "delete_run: run_id=zzz not found" in caplog.text
        assert len(on_disk(hist)) == 1

    def test_save_failure_returns_false(self, hist, monkeypatch, caplog):
        hist.record_run_started("a")
        fail_replace(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert hist.delete_run("a") is False
        assert "delete_run: save failed" in caplog.text
        assert len(on_disk(hist)) == 1  # on-disk state untouched


# ---------------------------------------------------------------------------
# in_progress_runs
# ---------------------------------------------------------------------------

def iso(delta: timedelta, *, tz=None) -> str:
    return (datetime.now(tz) - delta).isoformat()


class TestInProgressRuns:
    def test_empty_and_only_non_running(self, hist):
        assert hist.in_progress_runs() == []
        hist._save([{"run_id": "d", "status": "completed", "started_at": iso(timedelta(0))}])
        assert hist.in_progress_runs() == []

    def test_fresh_kept_stale_dropped_with_logs(self, hist, caplog):
        hist._save([
            {"run_id": "fresh", "status": "running", "started_at": iso(timedelta(hours=1))},
            {"run_id": "stale", "status": "running", "started_at": iso(timedelta(days=3))},
        ])
        with caplog.at_level(logging.INFO, logger=LOGGER):
            live = hist.in_progress_runs()
        assert [r["run_id"] for r in live] == ["fresh"]
        assert "skipping stale entry run_id='stale'" in caplog.text
        assert "filtered 1 stale entries" in caplog.text
        assert "1 live" in caplog.text

    def test_custom_threshold_applies(self, hist):
        hist._save([{"run_id": "r", "status": "running",
                     "started_at": iso(timedelta(minutes=10))}])
        assert hist.in_progress_runs(stale_threshold_seconds=60) == []
        assert len(hist.in_progress_runs(stale_threshold_seconds=3600)) == 1

    def test_zero_threshold_disables_filter(self, hist):
        hist._save([{"run_id": "ancient", "status": "running",
                     "started_at": iso(timedelta(days=900))}])
        assert [r["run_id"] for r in hist.in_progress_runs(stale_threshold_seconds=0.0)] == ["ancient"]

    def test_entries_without_usable_timestamps_are_kept_for_triage(self, hist):
        hist._save([
            {"run_id": "none", "status": "running"},
            {"run_id": "junk", "status": "running", "started_at": "not-a-date"},
            {"run_id": "tz", "status": "running",
             "started_at": iso(timedelta(days=30), tz=timezone.utc)},  # aware vs naive now
            {"run_id": "ts_only", "status": "running", "timestamp": iso(timedelta(hours=1))},
        ])
        assert {r["run_id"] for r in hist.in_progress_runs()} == {"none", "junk", "tz", "ts_only"}


def test_run_history_roundtrip_survives_new_instance(tmp_path):
    a = RunHistoryManager(str(tmp_path / "o"))
    a.record_run_started("r1", model_name="m")
    a.record_run_completed("r1", final_loss=0.1)
    b = RunHistoryManager(str(tmp_path / "o"))
    assert b.get_run("r1")["status"] == "completed"
    assert os.path.exists(b._history_path)


class TestHistoryCorners:
    def test_completed_matches_later_entry_skipping_earlier_ones(self, hist):
        hist.record_run_started("a")
        hist.record_run_started("b")
        hist.record_run_completed("b", final_loss=0.3)
        rows = {r["run_id"]: r for r in on_disk(hist)}
        assert rows["b"]["status"] == "completed" and rows["a"]["status"] == "running"

    def test_save_with_missing_output_dir_fails_before_tmp_exists(self, tmp_path, caplog):
        h = RunHistoryManager(str(tmp_path / "gone"))
        h.output_dir.rmdir()
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            assert h._save([{"run_id": "x"}]) is False
        assert "Failed to save run history" in caplog.text
