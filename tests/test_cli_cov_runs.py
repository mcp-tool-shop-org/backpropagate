"""Coverage tests for the run-history readers: ``list-runs``, ``runs``,
``show-run`` and ``diff-runs`` in cli.py.

Nothing is mocked: run history is written through the real
``RunHistoryManager`` into ``tmp_path`` and read back by the real handlers.
"""

from __future__ import annotations

import json

import pytest

from backpropagate import cli
from tests.helpers.cli_cov_support import last_json, parse, seed_runs


@pytest.fixture
def out_dir(tmp_path):
    out = tmp_path / "out"
    seed_runs(out, [
        {
            "run_id": "aaaaaaaa-1111", "status": "completed", "model_name": "tiny/model-a",
            "dataset_info": "train.jsonl", "session_kind": "single_run", "steps": 50,
            "final_loss": 0.4321, "loss_history": [2.0, 1.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.43],
            "duration_seconds": 12.34, "checkpoint_path": "/ckpt/a", "dataset_hash": "deadbeef12345678",
            "hyperparameters": {"lora_r": 8, "learning_rate": 0.0002},
            "started_at": "2026-05-21T13:42:18.123456", "completed_at": "2026-05-21T13:43:00",
            "merge_history": [{"run_index": 0, "result": {"merged": True}}],
            "export_paths": ["/exports/a"],
        },
        {
            "run_id": "bbbbbbbb-2222", "status": "failed", "model_name": "tiny/model-b",
            "dataset_info": "train.jsonl", "session_kind": "multi_run", "steps": 50,
            "final_loss": 1e7, "failure_reason": "CUDA OOM",
            "hyperparameters": {"lora_r": 16, "learning_rate": 0.0002, "extra": "x"},
            "started_at": "2026-05-22T09:00:00", "completed_at": "2026-05-22T09:00:30",
        },
        {
            "run_id": "cccccccc-3333", "status": "running", "model_name": "tiny/model-c",
            "started_at": "not-a-date", "completed_at": "also-not-a-date",
        },
    ])
    return out


class TestListRuns:
    def test_missing_dir_warns(self, tmp_path, capsys):
        assert cli.cmd_list_runs(parse(["list-runs", "--output", str(tmp_path / "x")])) == cli.EXIT_OK
        assert "No history found" in capsys.readouterr().out

    def test_invalid_status_filter(self, out_dir, capsys):
        args = parse(["list-runs", "--output", str(out_dir)])
        args.status = "bogus"
        assert cli.cmd_list_runs(args) == cli.EXIT_USER_ERROR
        assert "Invalid status 'bogus'" in capsys.readouterr().err

    def test_empty_with_status_filter(self, out_dir, capsys):
        seed_dir = out_dir.parent / "empty"
        seed_dir.mkdir()
        args = parse(["list-runs", "--output", str(seed_dir), "--status", "failed"])
        assert cli.cmd_list_runs(args) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "No training runs recorded." in out and "(Filter: status=failed)" in out

    def test_table(self, out_dir, capsys):
        assert cli.cmd_list_runs(parse(["list-runs", "--output", str(out_dir)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "RUN_ID" in out and "FINAL_LOSS" in out
        assert "aaaaaaaa-111" in out and "0.4321" in out
        assert "Listed 3 run(s)" in out

    def test_json_rows_are_versioned(self, out_dir, capsys):
        assert cli.cmd_list_runs(parse(["list-runs", "--output", str(out_dir), "--json"])) == cli.EXIT_OK
        rows = json.loads(capsys.readouterr().out)
        assert len(rows) == 3
        assert {r["schema_version"] for r in rows} == {cli.CLI_JSON_SCHEMA_VERSION}


class TestRuns:
    def test_missing_dir_human_and_json(self, tmp_path, capsys):
        missing = str(tmp_path / "nowhere")
        assert cli.cmd_runs(parse(["runs", "--output", missing])) == cli.EXIT_OK
        assert "No history found" in capsys.readouterr().out
        assert cli.cmd_runs(parse(["runs", "--output", missing, "--json"])) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        assert payload["runs"] == [] and payload["schema_version"] == cli.RUNS_JSON_SCHEMA_VERSION

    def test_invalid_status(self, out_dir, capsys):
        args = parse(["runs", "--output", str(out_dir)])
        args.status = "bogus"
        assert cli.cmd_runs(args) == cli.EXIT_USER_ERROR
        assert "Invalid status 'bogus'" in capsys.readouterr().err

    def test_empty_history_human(self, tmp_path, capsys):
        empty = tmp_path / "e"
        empty.mkdir()
        assert cli.cmd_runs(parse(["runs", "--output", str(empty), "--status", "completed"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "No training runs recorded." in out and "(Filter: status=completed)" in out

    def test_empty_history_without_filter(self, tmp_path, capsys):
        empty = tmp_path / "e"
        empty.mkdir()
        assert cli.cmd_runs(parse(["runs", "--output", str(empty)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "No training runs recorded." in out and "(Filter" not in out

    def test_human_listing(self, out_dir, capsys):
        assert cli.cmd_runs(parse(["runs", "--output", str(out_dir)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Training runs (schema v1)" in out
        assert "Run count: 3" in out
        assert "aaaaaaaa-111" in out and "loss=0.4321" in out
        assert "loss=-" in out  # the run without a final_loss
        assert "backprop list-runs" in out

    def test_json_payload_projection(self, out_dir, capsys):
        assert cli.cmd_runs(parse(["runs", "--output", str(out_dir), "--json"])) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        by_id = {r["run_id"]: r for r in payload["runs"]}
        assert by_id["aaaaaaaa-1111"]["loss"] == {"final": 0.4321, "min": 0.43}
        assert by_id["aaaaaaaa-1111"]["duration_seconds"] == 12.34
        # Computed from timestamps when no explicit duration was recorded.
        assert by_id["bbbbbbbb-2222"]["duration_seconds"] == 30.0
        # Unparseable timestamps degrade to None rather than crashing.
        assert by_id["cccccccc-3333"]["duration_seconds"] is None

    def test_runs_payload_ignores_unparseable_values(self, tmp_path):
        payload = cli._build_runs_payload(
            [{"run_id": "r", "loss_history": ["x", None], "started_at": 5, "completed_at": 6}], tmp_path)
        assert payload["runs"][0]["loss"] == {"final": None, "min": None}
        assert payload["runs"][0]["duration_seconds"] is None


class TestShowRun:
    def test_missing_dir(self, tmp_path, capsys):
        assert cli.cmd_show_run(parse(["show-run", "x", "--output", str(tmp_path / "no")])) == cli.EXIT_USER_ERROR
        assert "No history directory" in capsys.readouterr().err

    def test_unknown_run(self, out_dir, capsys):
        assert cli.cmd_show_run(parse(["show-run", "zzz", "--output", str(out_dir)])) == cli.EXIT_USER_ERROR
        assert "No run matching 'zzz'" in capsys.readouterr().err

    def test_json(self, out_dir, capsys):
        assert cli.cmd_show_run(parse(["show-run", "aaaaaaaa", "--output", str(out_dir), "--json"])) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        assert payload["run_id"] == "aaaaaaaa-1111"
        assert payload["schema_version"] == cli.CLI_JSON_SCHEMA_VERSION

    def test_full_human_view(self, out_dir, capsys):
        assert cli.cmd_show_run(parse(["show-run", "aaaaaaaa-1111", "--output", str(out_dir)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Run aaaaaaaa-111" in out
        assert "Dataset sha256 (16): deadbeef12345678" in out
        assert "Final loss: 0.4321" in out and "Steps: 50" in out and "Duration: 12.3s" in out
        assert "Checkpoint: /ckpt/a" in out
        assert "Hyperparameters" in out and "lora_r: 8" in out
        # >8 loss points -> head, ellipsis and tail are all shown
        assert "Loss history" in out and "(10 points)" in out and ", ..., " in out
        assert "SLAO merge history" in out and "run_0:" in out
        assert "Exports" in out and "/exports/a" in out
        assert "Started: 2026-05-21 13:42" in out

    def test_failed_run_shows_reason(self, out_dir, capsys):
        assert cli.cmd_show_run(parse(["show-run", "bbbbbbbb", "--output", str(out_dir)])) == cli.EXIT_OK
        assert "Failure reason: CUDA OOM" in capsys.readouterr().out

    def test_sparse_run_omits_optional_blocks(self, out_dir, capsys):
        assert cli.cmd_show_run(parse(["show-run", "cccccccc", "--output", str(out_dir)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        for absent in ("Loss history", "SLAO merge history", "Exports", "Hyperparameters",
                       "Failure reason", "Checkpoint", "Dataset sha256"):
            assert absent not in out

    def test_short_loss_history_has_no_tail(self, tmp_path, capsys):
        out_dir = tmp_path / "o"
        seed_runs(out_dir, [{"run_id": "short-1", "loss_history": [3.0, 2.0, 1.0]}])
        assert cli.cmd_show_run(parse(["show-run", "short-1", "--output", str(out_dir)])) == cli.EXIT_OK
        text = capsys.readouterr().out
        assert "(3 points)" in text and ", ..., " not in text and "3.0000, 2.0000, 1.0000" in text


class TestDiffRuns:
    def test_missing_dir(self, tmp_path, capsys):
        args = parse(["diff-runs", "a", "b", "--output", str(tmp_path / "no")])
        assert cli.cmd_diff_runs(args) == cli.EXIT_USER_ERROR
        assert "No history directory" in capsys.readouterr().err

    @pytest.mark.parametrize("which", ["a", "b"])
    def test_unknown_run(self, out_dir, capsys, which):
        ids = ["aaaaaaaa", "bbbbbbbb"]
        ids[0 if which == "a" else 1] = "nope"
        args = parse(["diff-runs", *ids, "--output", str(out_dir)])
        assert cli.cmd_diff_runs(args) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert f"run_id_{which}" in captured.err
        assert "Suggestion:" in captured.out

    def test_json(self, out_dir, capsys):
        args = parse(["diff-runs", "aaaaaaaa", "bbbbbbbb", "--output", str(out_dir), "--format", "json"])
        assert cli.cmd_diff_runs(args) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        rows = {r["field"]: r for r in payload["diff"]}
        assert rows["model_name"]["changed"] is True
        assert rows["dataset_info"]["changed"] is False
        assert rows["hp.lora_r"] == {"field": "hp.lora_r", "run_a": 8, "run_b": 16, "changed": True}
        assert rows["hp.extra"]["run_a"] is None  # field only present on run b
        assert payload["changed_count"] == sum(1 for r in payload["diff"] if r["changed"])
        assert payload["run_a"]["run_id"] == "aaaaaaaa-1111"

    def test_non_dict_hyperparameters_are_ignored(self, tmp_path, capsys):
        seed_runs(tmp_path, [
            {"run_id": "odd-1111", "hyperparameters": "not-a-dict", "final_loss": 1.0},
            {"run_id": "odd-2222", "hyperparameters": {"lora_r": 4}, "final_loss": 2.0},
        ])
        args = parse(["diff-runs", "odd-1111", "odd-2222", "--output", str(tmp_path), "--format", "json"])
        assert cli.cmd_diff_runs(args) == cli.EXIT_OK
        rows = {r["field"]: r for r in last_json(capsys.readouterr().out)["diff"]}
        assert rows["hp.lora_r"]["run_a"] is None and rows["hp.lora_r"]["run_b"] == 4
        assert rows["final_loss"]["changed"] is True

    def test_table(self, out_dir, capsys):
        args = parse(["diff-runs", "aaaaaaaa", "bbbbbbbb", "--output", str(out_dir)])
        assert cli.cmd_diff_runs(args) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Diff: aaaaaaaa-111 vs bbbbbbbb-222" in out
        assert "FIELD" in out
        assert "0.4321" in out          # float formatted to 4 places
        assert "1e+07" in out           # large float switches to %g
        assert "-" in out               # None rendered as '-'
        assert "field(s) differ." in out
