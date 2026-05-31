"""CLI smoke tests for the v1.5 T1.1 surface: `data report` + `eval`.

Build Agent C (v1.5 feature wave). Covers:
- parser wiring for `backprop data report` (nested) + `backprop eval` (flat),
  including the ``_unit_float`` validator on the gate flags;
- the CHEAP user-error exit paths that return BEFORE the torch-heavy lazy
  import (so they run without loading a model): missing file / dir,
  ``--against`` missing, missing output dir, run-not-found, ``--heldout`` /
  ``--prompts`` unresolved;
- the full ``cmd_data_report`` path end-to-end (analyze_dataset is torch-free)
  including the ``--fail-*`` gate → exit 65 contract + ``--json`` payload;
- the full ``cmd_eval`` single / diff / gate paths with ``evaluate_run`` /
  ``diff_evals`` / ``eval_gate`` mocked so no model is loaded.

The exit-code contract under test (per the v1.5 CLI brief):
- data report: 0 advisory/clean · 1 bad input · 65 gate tripped OR zero rows.
- eval:        0 ran (gate passed) · 1 run-not-found / heldout-unresolved /
               prompts unreadable · 65 --gate-against regression tripped.
"""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

import pytest

from backpropagate.cli import (
    EXIT_DATA_ERR,
    EXIT_OK,
    EXIT_USER_ERROR,
    cmd_data_report,
    cmd_eval,
    create_parser,
)

# ---------------------------------------------------------------------------
# Shared fixtures / builders
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_parser():
    return create_parser()


def _payload_from_stdout(out: str) -> dict:
    """Extract the CLI JSON payload from captured stdout.

    The handlers emit a structured ``*_invoked`` log line via the structured
    logger; depending on the test environment's logging config that line may
    render as a JSON object on stdout AHEAD of the real payload. Both the log
    line and the payload are valid JSON, so we iterate every top-level JSON
    object in the stream (via raw_decode) and return the one carrying
    ``schema_version`` — the CLI payload, never the log line.
    """
    decoder = json.JSONDecoder()
    idx = 0
    candidates: list[dict] = []
    n = len(out)
    while idx < n:
        # Skip whitespace between objects.
        while idx < n and out[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(out, idx)
        except json.JSONDecodeError:
            # Non-JSON noise (shouldn't happen on the --json path) — advance
            # to the next newline and retry.
            nl = out.find("\n", idx)
            if nl == -1:
                break
            idx = nl + 1
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
        idx = end
    for obj in candidates:
        if "schema_version" in obj:
            return obj
    raise AssertionError(
        f"no JSON object with schema_version found in stdout: {out!r}"
    )


def _sharegpt_jsonl(path, n: int = 6) -> None:
    """Write an n-row ShareGPT JSONL dataset."""
    path.write_text(
        "\n".join(
            json.dumps({
                "conversations": [
                    {"from": "human", "value": f"question number {i}"},
                    {"from": "gpt", "value": f"answer number {i}"},
                ],
            })
            for i in range(n)
        ),
        encoding="utf-8",
    )


def _data_report_ns(**overrides) -> Namespace:
    """Default Namespace for cmd_data_report; override per-test."""
    base = {
        "dataset": "data.jsonl",
        "format": "auto",
        "against": None,
        "max_samples": None,
        "dup_threshold": 0.9,
        "fail_on_dups": None,
        "fail_on_contamination": None,
        "max_outlier_rate": None,
        "strict": False,
        "json": False,
        "verbose": False,
    }
    base.update(overrides)
    return Namespace(**base)


def _eval_ns(**overrides) -> Namespace:
    """Default Namespace for cmd_eval; override per-test."""
    base = {
        "run_id": "abc123",
        "vs": None,
        "gate_against": None,
        "output": "./output",
        "heldout": None,
        "prompts": None,
        "num_samples": 5,
        "max_new_tokens": 128,
        "max_regression": 0.0,
        "seed": 0,
        "json": False,
        "verbose": False,
        "cli_run_id": "deadbeefcafe",
    }
    base.update(overrides)
    return Namespace(**base)


# ===========================================================================
# Parser wiring — `backprop data report`
# ===========================================================================


class TestDataReportParser:
    def test_data_report_minimal_parses(self, cli_parser):
        args = cli_parser.parse_args(["data", "report", "my.jsonl"])
        assert args.func.__name__ == "cmd_data_report"
        assert args.dataset == "my.jsonl"
        assert args.format == "auto"
        assert args.dup_threshold == 0.9
        assert args.fail_on_dups is None
        assert args.strict is False
        assert args.json is False

    def test_data_report_all_flags_parse(self, cli_parser):
        args = cli_parser.parse_args([
            "data", "report", "my.jsonl",
            "--format", "sharegpt",
            "--against", "held.jsonl",
            "--max-samples", "1000",
            "--dup-threshold", "0.8",
            "--fail-on-dups", "0.1",
            "--fail-on-contamination", "0.0",
            "--max-outlier-rate", "0.2",
            "--strict",
            "--json",
        ])
        assert args.format == "sharegpt"
        assert args.against == "held.jsonl"
        assert args.max_samples == 1000
        assert args.dup_threshold == 0.8
        assert args.fail_on_dups == 0.1
        assert args.fail_on_contamination == 0.0
        assert args.max_outlier_rate == 0.2
        assert args.strict is True
        assert args.json is True

    def test_unit_float_rejects_out_of_range(self, cli_parser):
        # 90 (a percent typo for 0.9) must be rejected by _unit_float.
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["data", "report", "x.jsonl", "--fail-on-dups", "90"])

    def test_unit_float_rejects_negative(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["data", "report", "x.jsonl", "--dup-threshold", "-0.1"])

    def test_format_choices_enforced(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["data", "report", "x.jsonl", "--format", "csv"])

    def test_data_no_subcommand_returns_nonzero(self):
        # `backprop data` with no subcommand prints help + returns non-zero.
        from backpropagate.cli import create_parser as _cp
        parser = _cp()
        args = parser.parse_args(["data"])
        assert args.func.__name__ == "_data_no_subcommand"
        rc = args.func(args)
        assert rc != EXIT_OK


# ===========================================================================
# Parser wiring — `backprop eval`
# ===========================================================================


class TestEvalParser:
    def test_eval_minimal_parses(self, cli_parser):
        args = cli_parser.parse_args(["eval", "abc123"])
        assert args.func.__name__ == "cmd_eval"
        assert args.run_id == "abc123"
        assert args.vs is None
        assert args.gate_against is None
        assert args.output == "./output"
        assert args.num_samples == 5
        assert args.max_new_tokens == 128
        assert args.max_regression == 0.0
        assert args.seed == 0
        assert args.json is False

    def test_eval_all_flags_parse(self, cli_parser):
        args = cli_parser.parse_args([
            "eval", "abc123",
            "--vs", "def456",
            "--gate-against", "base789",
            "--output", "/tmp/out",
            "--heldout", "held.jsonl",
            "--prompts", "prompts.txt",
            "-n", "3",
            "--max-new-tokens", "64",
            "--max-regression", "0.05",
            "--seed", "7",
            "--json",
        ])
        assert args.vs == "def456"
        assert args.gate_against == "base789"
        assert args.output == "/tmp/out"
        assert args.heldout == "held.jsonl"
        assert args.prompts == "prompts.txt"
        assert args.num_samples == 3
        assert args.max_new_tokens == 64
        assert args.max_regression == 0.05
        assert args.seed == 7
        assert args.json is True

    def test_num_samples_long_alias(self, cli_parser):
        args = cli_parser.parse_args(["eval", "abc", "--num-samples", "8"])
        assert args.num_samples == 8


# ===========================================================================
# Root --help epilog enumerates the two new verbs
# ===========================================================================


class TestEpilogEnumeratesNewVerbs:
    def test_epilog_lists_data_report_and_eval(self, cli_parser):
        epilog = cli_parser.epilog or ""
        assert "data report" in epilog
        assert "eval" in epilog


# ===========================================================================
# cmd_data_report — cheap user-error paths (no torch, no analysis module)
# ===========================================================================


class TestDataReportUserErrors:
    def test_missing_file_exits_user_error(self, tmp_path, capsys):
        ns = _data_report_ns(dataset=str(tmp_path / "missing.jsonl"), json=True)
        rc = cmd_data_report(ns)
        assert rc == EXIT_USER_ERROR
        payload = _payload_from_stdout(capsys.readouterr().out)
        assert payload["error"] == "dataset_not_found"
        assert payload["schema_version"] == "1"

    def test_directory_exits_user_error(self, tmp_path, capsys):
        ns = _data_report_ns(dataset=str(tmp_path), json=True)
        rc = cmd_data_report(ns)
        assert rc == EXIT_USER_ERROR
        payload = _payload_from_stdout(capsys.readouterr().out)
        assert payload["error"] == "dataset_path_is_directory"

    def test_against_missing_exits_user_error(self, tmp_path, capsys):
        data = tmp_path / "data.jsonl"
        _sharegpt_jsonl(data)
        ns = _data_report_ns(
            dataset=str(data),
            against=str(tmp_path / "nope.jsonl"),
            json=True,
        )
        rc = cmd_data_report(ns)
        assert rc == EXIT_USER_ERROR
        payload = _payload_from_stdout(capsys.readouterr().out)
        assert payload["error"] == "against_not_found"

    def test_zero_parseable_rows_exits_data_err(self, tmp_path, capsys):
        # File exists + readable but has no parseable JSON rows.
        empty = tmp_path / "empty.jsonl"
        empty.write_text("\n\n   \n", encoding="utf-8")
        ns = _data_report_ns(dataset=str(empty), json=True)
        rc = cmd_data_report(ns)
        assert rc == EXIT_DATA_ERR
        payload = _payload_from_stdout(capsys.readouterr().out)
        assert payload["error"] == "no_parseable_rows"


# ===========================================================================
# cmd_data_report — full path (analyze_dataset is torch-free, runs for real)
# ===========================================================================


class TestDataReportFullPath:
    def test_clean_dataset_advisory_exits_ok(self, tmp_path, capsys):
        data = tmp_path / "data.jsonl"
        _sharegpt_jsonl(data, n=8)
        ns = _data_report_ns(dataset=str(data), json=True)
        rc = cmd_data_report(ns)
        assert rc == EXIT_OK
        payload = _payload_from_stdout(capsys.readouterr().out)
        assert payload["schema_version"] == "1"
        # to_dict() fields are spread into the payload.
        assert "verdict" in payload
        assert "failed_thresholds" in payload

    def test_fail_on_dups_gate_trips_exit_65(self, tmp_path, capsys):
        # A dataset of identical rows trips --fail-on-dups 0.0 (any dup fails).
        data = tmp_path / "dupes.jsonl"
        row = json.dumps({
            "conversations": [
                {"from": "human", "value": "same question"},
                {"from": "gpt", "value": "same answer"},
            ],
        })
        data.write_text("\n".join([row] * 8), encoding="utf-8")
        ns = _data_report_ns(dataset=str(data), fail_on_dups=0.0, json=True)
        rc = cmd_data_report(ns)
        assert rc == EXIT_DATA_ERR
        payload = _payload_from_stdout(capsys.readouterr().out)
        assert payload["verdict"] == "FAIL"
        assert payload["failed_thresholds"]  # non-empty

    def test_gate_trip_stamps_catalog_code(self, tmp_path):
        # The non-zero return must be accompanied by a structured log line
        # carrying code="INPUT_DATASET_REPORT_THRESHOLD".
        data = tmp_path / "dupes.jsonl"
        row = json.dumps({
            "conversations": [
                {"from": "human", "value": "x"},
                {"from": "gpt", "value": "y"},
            ],
        })
        data.write_text("\n".join([row] * 8), encoding="utf-8")
        ns = _data_report_ns(dataset=str(data), fail_on_dups=0.0)
        with patch("backpropagate.logging_config.get_logger") as mock_get_logger:
            rc = cmd_data_report(ns)
        assert rc == EXIT_DATA_ERR
        warning = mock_get_logger.return_value.warning
        assert warning.called
        # code= is passed as a kwarg on the structured log call.
        _, kwargs = warning.call_args
        assert kwargs.get("code") == "INPUT_DATASET_REPORT_THRESHOLD"

    def test_human_output_renders_summary(self, tmp_path, capsys):
        data = tmp_path / "data.jsonl"
        _sharegpt_jsonl(data, n=6)
        ns = _data_report_ns(dataset=str(data), json=False)
        rc = cmd_data_report(ns)
        assert rc == EXIT_OK
        out = capsys.readouterr().out
        assert "Verdict" in out  # from DataQualityReport.summary()


# ===========================================================================
# cmd_eval — cheap user-error paths (return BEFORE the torch import)
# ===========================================================================


class TestEvalUserErrors:
    def test_missing_output_dir_exits_user_error(self, tmp_path, capsys):
        ns = _eval_ns(output=str(tmp_path / "no-such-dir"))
        rc = cmd_eval(ns)
        assert rc == EXIT_USER_ERROR
        err = capsys.readouterr().err
        assert "No output directory" in err

    def test_run_not_found_exits_user_error(self, tmp_path, capsys):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        ns = _eval_ns(run_id="ghost", output=str(out_dir))
        # RunHistoryManager.get_run returns None for the unknown id.
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value=None):
            rc = cmd_eval(ns)
        assert rc == EXIT_USER_ERROR
        err = capsys.readouterr().err
        assert "not found in run history" in err

    def test_heldout_unresolved_exits_user_error(self, tmp_path, capsys):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        ns = _eval_ns(
            run_id="real",
            output=str(out_dir),
            heldout=str(tmp_path / "missing-held.jsonl"),
        )
        # run_id resolves, but --heldout path does not exist.
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "real"}):
            rc = cmd_eval(ns)
        assert rc == EXIT_USER_ERROR
        err = capsys.readouterr().err
        assert "--heldout" in err

    def test_prompts_unresolved_exits_user_error(self, tmp_path, capsys):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        ns = _eval_ns(
            run_id="real",
            output=str(out_dir),
            prompts=str(tmp_path / "missing-prompts.txt"),
        )
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "real"}):
            rc = cmd_eval(ns)
        assert rc == EXIT_USER_ERROR
        err = capsys.readouterr().err
        assert "--prompts" in err

    def test_run_not_found_stamps_catalog_code(self, tmp_path):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        ns = _eval_ns(run_id="ghost", output=str(out_dir))
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value=None), \
             patch("backpropagate.logging_config.get_logger") as mock_get_logger:
            rc = cmd_eval(ns)
        assert rc == EXIT_USER_ERROR
        # At least one warning call carried the run-not-found code.
        codes = [
            kw.get("code")
            for _, kw in mock_get_logger.return_value.warning.call_args_list
        ]
        assert "INPUT_EVAL_RUN_NOT_FOUND" in codes


# ===========================================================================
# cmd_eval — full paths with evaluate_run / diff_evals / eval_gate MOCKED
# ===========================================================================


def _fake_eval_result(run_id="real", loss=1.0):
    """A lightweight stand-in for EvalResult with the fields the CLI reads."""
    from backpropagate.eval import EvalResult
    return EvalResult(
        run_id=run_id,
        model_name="fake/model",
        held_out_loss=loss,
        perplexity=2.0,
        generations=[],
        n_prompts=0,
    )


class TestEvalSingle:
    def test_single_eval_json_ok(self, tmp_path, capsys):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        ns = _eval_ns(run_id="real", output=str(out_dir), json=True)
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "real"}), \
             patch("backpropagate.eval.evaluate_run", return_value=_fake_eval_result()):
            rc = cmd_eval(ns)
        assert rc == EXIT_OK
        payload = _payload_from_stdout(capsys.readouterr().out)
        assert payload["schema_version"] == "1"
        assert payload["mode"] == "single"
        assert payload["result"]["run_id"] == "real"

    def test_single_eval_human_ok(self, tmp_path, capsys):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        ns = _eval_ns(run_id="real", output=str(out_dir), json=False)
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "real"}), \
             patch("backpropagate.eval.evaluate_run", return_value=_fake_eval_result()):
            rc = cmd_eval(ns)
        assert rc == EXIT_OK
        out = capsys.readouterr().out
        assert "Held-out loss" in out
        assert "Eval complete" in out


class TestEvalDiff:
    def test_vs_diff_json_ok(self, tmp_path, capsys):
        from backpropagate.eval import EvalDiff

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        ns = _eval_ns(run_id="a", vs="b", output=str(out_dir), json=True)
        diff = EvalDiff(run_id_a="a", run_id_b="b", rows=[("held_out_loss", "1.0000", "1.2000")])
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "x"}), \
             patch("backpropagate.eval.evaluate_run", side_effect=[_fake_eval_result("a", 1.0), _fake_eval_result("b", 1.2)]), \
             patch("backpropagate.eval.diff_evals", return_value=diff):
            rc = cmd_eval(ns)
        assert rc == EXIT_OK
        payload = _payload_from_stdout(capsys.readouterr().out)
        assert payload["mode"] == "diff"
        assert payload["run_a"] == "a"
        assert payload["run_b"] == "b"


class TestEvalGate:
    def test_gate_accept_exits_ok(self, tmp_path, capsys):
        from backpropagate.eval import EvalGateDecision

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        ns = _eval_ns(run_id="after", gate_against="before", output=str(out_dir), json=True)
        decision = EvalGateDecision(accept=True, reason="no regression", regression=-0.1)
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "x"}), \
             patch("backpropagate.eval.evaluate_run", side_effect=[_fake_eval_result("after", 0.9), _fake_eval_result("before", 1.0)]), \
             patch("backpropagate.eval.eval_gate", return_value=decision):
            rc = cmd_eval(ns)
        assert rc == EXIT_OK
        payload = _payload_from_stdout(capsys.readouterr().out)
        assert payload["mode"] == "gate"
        assert payload["accept"] is True

    def test_gate_reject_exits_data_err_and_stamps_code(self, tmp_path):
        from backpropagate.eval import EvalGateDecision

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        ns = _eval_ns(run_id="after", gate_against="before", output=str(out_dir), json=False)
        decision = EvalGateDecision(accept=False, reason="regressed", regression=0.3)
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "x"}), \
             patch("backpropagate.eval.evaluate_run", side_effect=[_fake_eval_result("after", 1.3), _fake_eval_result("before", 1.0)]), \
             patch("backpropagate.eval.eval_gate", return_value=decision), \
             patch("backpropagate.logging_config.get_logger") as mock_get_logger:
            rc = cmd_eval(ns)
        assert rc == EXIT_DATA_ERR
        codes = [
            kw.get("code")
            for _, kw in mock_get_logger.return_value.warning.call_args_list
        ]
        assert "RUNTIME_EVAL_GATE_REGRESSED" in codes
