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
    EVAL_METRIC_CHOICES,
    EXIT_DATA_ERR,
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    EXIT_USER_ERROR,
    cmd_data_report,
    cmd_data_split,
    cmd_eval,
    cmd_generate,
    cmd_train,
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
        # v1.6 C4: task-metric / reference-set / gate-metric flags.
        "metric": None,
        "references": None,
        "gate_metric": None,
    }
    base.update(overrides)
    return Namespace(**base)


def _generate_ns(**overrides) -> Namespace:
    """Default Namespace for cmd_generate (v1.6 C4); override per-test."""
    base = {
        "adapter_path": "./output",
        "prompt": "Hello",
        "base": None,
        "max_new_tokens": 128,
        "num": 1,
        "temperature": 0.7,
        "seed": 0,
        "verbose": False,
        "cli_run_id": "deadbeefcafe",
    }
    base.update(overrides)
    return Namespace(**base)


def _data_split_ns(**overrides) -> Namespace:
    """Default Namespace for cmd_data_split (v1.6 C4); override per-test."""
    base = {
        "dataset": "data.jsonl",
        "heldout_ratio": 0.1,
        "seed": 0,
        "out_train": None,
        "out_heldout": None,
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


# ===========================================================================
# v1.6 C4 — train / multi-run method + per-method hyperparameter flags
# ===========================================================================


class TestTrainMethodFlags:
    """`backprop train --method {sft,orpo,simpo,kto}` + SimPO/KTO knobs parse
    and bind to the namespace the cmd_train wave6b kwarg-bundle reads."""

    def test_method_choices_include_simpo_kto(self, cli_parser):
        for method in ("sft", "orpo", "simpo", "kto"):
            args = cli_parser.parse_args(["train", "--data", "d.jsonl", "--method", method])
            assert args.method == method

    def test_method_rejects_unknown(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["train", "--data", "d.jsonl", "--method", "dpo"])

    def test_simpo_flags_parse(self, cli_parser):
        args = cli_parser.parse_args([
            "train", "--data", "d.jsonl", "--method", "simpo",
            "--simpo-beta", "2.5", "--simpo-gamma", "1.2",
        ])
        assert args.simpo_beta == 2.5
        assert args.simpo_gamma == 1.2

    def test_kto_flags_parse(self, cli_parser):
        args = cli_parser.parse_args([
            "train", "--data", "d.jsonl", "--method", "kto",
            "--kto-beta", "0.2",
            "--kto-desirable-weight", "1.5",
            "--kto-undesirable-weight", "0.8",
        ])
        assert args.kto_beta == 0.2
        assert args.kto_desirable_weight == 1.5
        assert args.kto_undesirable_weight == 0.8

    def test_simpo_kto_flags_default_none(self, cli_parser):
        # Unset -> None so the config field defaults (2.0/1.0/0.1/1.0/1.0)
        # govern instead of being clobbered by an argparse default.
        args = cli_parser.parse_args(["train", "--data", "d.jsonl"])
        assert args.simpo_beta is None
        assert args.simpo_gamma is None
        assert args.kto_beta is None
        assert args.kto_desirable_weight is None
        assert args.kto_undesirable_weight is None

    def test_positive_float_rejects_zero(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["train", "--data", "d.jsonl", "--kto-beta", "0"])

    def test_positive_float_rejects_negative(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["train", "--data", "d.jsonl", "--simpo-beta", "-1"])


class TestMultiRunMethodFlags:
    """multi-run mirrors the train method surface (v1.6 C4)."""

    def test_method_choices_include_simpo_kto(self, cli_parser):
        for method in ("sft", "orpo", "simpo", "kto"):
            args = cli_parser.parse_args(["multi-run", "--data", "d.jsonl", "--method", method])
            assert args.method == method

    def test_simpo_kto_flags_parse(self, cli_parser):
        args = cli_parser.parse_args([
            "multi-run", "--data", "d.jsonl", "--method", "kto",
            "--kto-beta", "0.15", "--simpo-beta", "2.0",
        ])
        assert args.kto_beta == 0.15
        assert args.simpo_beta == 2.0


class TestTrainMethodReachesConfig:
    """End-to-end: --method simpo/kto + the hyperparameters reach the Trainer
    constructor as kwargs (the wave6b introspection bundle forwards them when
    the constructor accepts them). The trainer dispatch itself is built by the
    trainer agent; here we assert the CLI wires the values through."""

    def _run_cmd_train_capture_kwargs(self, cli_parser, argv):
        """Parse `argv`, run cmd_train with a fake Trainer, return the kwargs
        the Trainer constructor was called with."""
        from unittest.mock import MagicMock

        args = cli_parser.parse_args(argv)
        args.cli_run_id = "deadbeefcafe"
        # main() backfills the SUPPRESS-defaulted --verbose; mirror it here.
        args.verbose = False

        captured = {}

        def _fake_trainer_init(self, **kwargs):
            captured.update(kwargs)

        # A real class (not a MagicMock) so inspect.signature sees an explicit
        # **kwargs VAR_KEYWORD -> the CLI does NOT filter the new keys out.
        class _FakeTrainer:
            def __init__(self, **kwargs):
                _fake_trainer_init(self, **kwargs)

            def train(self, *a, **k):
                res = MagicMock()
                res.final_loss = 0.1
                res.duration_seconds = 1.0
                res.run_id = "rid"
                return res

            def save(self, out):
                return out

        with patch("backpropagate.trainer.Trainer", _FakeTrainer), \
             patch("backpropagate.trainer.TrainingCallback", MagicMock()):
            rc = cmd_train(args)
        return rc, captured

    def test_simpo_method_and_params_reach_trainer(self, cli_parser):
        rc, kwargs = self._run_cmd_train_capture_kwargs(
            cli_parser,
            ["train", "--data", "d.jsonl", "--method", "simpo",
             "--simpo-beta", "2.5", "--simpo-gamma", "1.0", "--steps", "1"],
        )
        assert rc == EXIT_OK
        assert kwargs.get("method") == "simpo"
        assert kwargs.get("simpo_beta") == 2.5
        assert kwargs.get("simpo_gamma") == 1.0

    def test_kto_method_and_params_reach_trainer(self, cli_parser):
        rc, kwargs = self._run_cmd_train_capture_kwargs(
            cli_parser,
            ["train", "--data", "d.jsonl", "--method", "kto",
             "--kto-beta", "0.2", "--kto-desirable-weight", "1.5", "--steps", "1"],
        )
        assert rc == EXIT_OK
        assert kwargs.get("method") == "kto"
        assert kwargs.get("kto_beta") == 0.2
        assert kwargs.get("kto_desirable_weight") == 1.5

    def test_unset_simpo_kto_params_not_forwarded(self, cli_parser):
        # When the operator doesn't set them, the None-valued keys are dropped
        # so the config defaults govern (not clobbered by None).
        rc, kwargs = self._run_cmd_train_capture_kwargs(
            cli_parser,
            ["train", "--data", "d.jsonl", "--method", "simpo", "--steps", "1"],
        )
        assert rc == EXIT_OK
        assert kwargs.get("method") == "simpo"
        assert "simpo_beta" not in kwargs
        assert "kto_beta" not in kwargs

    def test_config_validation_error_surfaces_as_user_error(self, cli_parser):
        # A bad value that the C2 config validators reject (raised as a
        # structured BackpropagateError from the Trainer constructor) is
        # surfaced cleanly via the structured-error path. We simulate the
        # constructor raising InvalidSettingError (CONFIG_INVALID_SETTING).
        from unittest.mock import MagicMock

        from backpropagate.exceptions import InvalidSettingError

        args = cli_parser.parse_args(
            ["train", "--data", "d.jsonl", "--method", "simpo", "--steps", "1"]
        )
        args.cli_run_id = "deadbeefcafe"
        # main() normally backfills the SUPPRESS-defaulted --verbose; mirror it
        # for the direct-handler call.
        args.verbose = False

        def _raise(**kwargs):
            raise InvalidSettingError("simpo_gamma", -1.0, "a positive float")

        with patch("backpropagate.trainer.Trainer", side_effect=_raise), \
             patch("backpropagate.trainer.TrainingCallback", MagicMock()):
            rc = cmd_train(args)
        # InvalidSettingError is a BackpropagateError (ConfigurationError) — the
        # cmd_train handler maps it to a runtime error (it isn't a
        # UserInputError subclass), but it must NOT crash / leak a stack.
        assert rc in (EXIT_USER_ERROR, EXIT_RUNTIME_ERROR)


# ===========================================================================
# v1.6 C4 — `backprop generate <adapter_path> "<prompt>"`
# ===========================================================================


class TestGenerateParser:
    def test_generate_minimal_parses(self, cli_parser):
        args = cli_parser.parse_args(["generate", "./adapter", "Say hi"])
        assert args.func.__name__ == "cmd_generate"
        assert args.adapter_path == "./adapter"
        assert args.prompt == "Say hi"
        assert args.base is None
        assert args.max_new_tokens == 128
        assert args.num == 1
        assert args.temperature == 0.7

    def test_generate_all_flags_parse(self, cli_parser):
        args = cli_parser.parse_args([
            "generate", "./adapter", "Say hi",
            "--base", "Qwen/Qwen2.5-7B-Instruct",
            "--max-new-tokens", "64",
            "-n", "3",
            "--temperature", "0.9",
            "--seed", "11",
        ])
        assert args.base == "Qwen/Qwen2.5-7B-Instruct"
        assert args.max_new_tokens == 64
        assert args.num == 3
        assert args.temperature == 0.9
        assert args.seed == 11

    def test_num_long_alias(self, cli_parser):
        args = cli_parser.parse_args(["generate", "./a", "p", "--num", "5"])
        assert args.num == 5

    def test_max_new_tokens_rejects_zero(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["generate", "./a", "p", "--max-new-tokens", "0"])

    def test_requires_prompt(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["generate", "./a"])


class TestGenerateErrors:
    """Cheap user-error paths return BEFORE the torch-heavy import."""

    def test_missing_adapter_path_user_error(self, tmp_path, capsys):
        ns = _generate_ns(adapter_path=str(tmp_path / "does_not_exist"))
        rc = cmd_generate(ns)
        assert rc == EXIT_USER_ERROR
        assert "not found" in capsys.readouterr().err.lower()

    def test_adapter_path_is_file_user_error(self, tmp_path, capsys):
        f = tmp_path / "afile"
        f.write_text("not a dir", encoding="utf-8")
        ns = _generate_ns(adapter_path=str(f))
        rc = cmd_generate(ns)
        assert rc == EXIT_USER_ERROR
        assert "directory" in capsys.readouterr().err.lower()

    def test_base_unresolved_user_error(self, tmp_path, capsys):
        # An adapter dir with NO adapter_config.json and no --base -> user error.
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        ns = _generate_ns(adapter_path=str(adapter_dir), base=None)
        rc = cmd_generate(ns)
        assert rc == EXIT_USER_ERROR
        err = capsys.readouterr().err.lower()
        assert "base" in err

    def test_missing_adapter_stamps_catalog_code(self, tmp_path):
        ns = _generate_ns(adapter_path=str(tmp_path / "nope"))
        with patch("backpropagate.logging_config.get_logger") as mock_get_logger:
            rc = cmd_generate(ns)
        assert rc == EXIT_USER_ERROR
        codes = [
            kw.get("code")
            for _, kw in mock_get_logger.return_value.warning.call_args_list
        ]
        assert "INPUT_VALIDATION_FAILED" in codes


class TestGenerateBaseInference:
    """--base is inferred from adapter_config.json when present."""

    def test_infers_base_from_adapter_config(self, tmp_path):
        from backpropagate.cli import _infer_base_model_from_adapter

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}),
            encoding="utf-8",
        )
        assert _infer_base_model_from_adapter(adapter_dir) == "Qwen/Qwen2.5-7B-Instruct"

    def test_returns_none_when_no_config(self, tmp_path):
        from backpropagate.cli import _infer_base_model_from_adapter

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        assert _infer_base_model_from_adapter(adapter_dir) is None

    def test_returns_none_on_malformed_config(self, tmp_path):
        from backpropagate.cli import _infer_base_model_from_adapter

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{not json", encoding="utf-8")
        assert _infer_base_model_from_adapter(adapter_dir) is None


class TestGenerateHappyPath:
    """Full handler path with the model load + generation MOCKED."""

    def test_generate_prints_completions(self, tmp_path, capsys):
        from backpropagate.eval import GenerationSample

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "fake/base"}),
            encoding="utf-8",
        )
        ns = _generate_ns(adapter_path=str(adapter_dir), prompt="hi", num=2)
        samples = [
            GenerationSample(prompt="hi", completion="hello there"),
            GenerationSample(prompt="hi", completion="greetings"),
        ]
        with patch(
            "backpropagate.eval._load_model_and_tokenizer",
            return_value=(object(), object()),
        ), patch("backpropagate.eval._generate", return_value=samples) as gen:
            rc = cmd_generate(ns)
        assert rc == EXIT_OK
        out = capsys.readouterr().out
        assert "hello there" in out
        assert "greetings" in out
        # n=2 completions were requested for the single prompt.
        _, gen_kwargs = gen.call_args
        prompts_arg = gen.call_args[0][2]
        assert prompts_arg == ["hi", "hi"]

    def test_explicit_base_overrides_inference(self, tmp_path, capsys):
        from backpropagate.eval import GenerationSample

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "fake/inferred"}),
            encoding="utf-8",
        )
        ns = _generate_ns(adapter_path=str(adapter_dir), base="explicit/base")
        captured_run = {}

        def _fake_load(run):
            captured_run.update(run)
            return object(), object()

        with patch("backpropagate.eval._load_model_and_tokenizer", side_effect=_fake_load), \
             patch("backpropagate.eval._generate", return_value=[GenerationSample(prompt="Hello", completion="ok")]):
            rc = cmd_generate(ns)
        assert rc == EXIT_OK
        assert captured_run.get("model_name") == "explicit/base"
        assert captured_run.get("checkpoint_path") == str(adapter_dir)

    def test_load_failure_returns_runtime_error(self, tmp_path):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "fake/base"}),
            encoding="utf-8",
        )
        from backpropagate.exceptions import TrainingError

        ns = _generate_ns(adapter_path=str(adapter_dir))
        with patch(
            "backpropagate.eval._load_model_and_tokenizer",
            side_effect=TrainingError("boom", code="RUNTIME_EVAL_FAILED"),
        ):
            rc = cmd_generate(ns)
        assert rc == EXIT_RUNTIME_ERROR


# ===========================================================================
# v1.6 C4 — `backprop data split <jsonl>`
# ===========================================================================


class TestDataSplitParser:
    def test_data_split_minimal_parses(self, cli_parser):
        args = cli_parser.parse_args(["data", "split", "d.jsonl"])
        assert args.func.__name__ == "cmd_data_split"
        assert args.dataset == "d.jsonl"
        assert args.heldout_ratio == 0.1
        assert args.seed == 0
        assert args.out_train is None
        assert args.out_heldout is None

    def test_data_split_all_flags_parse(self, cli_parser):
        args = cli_parser.parse_args([
            "data", "split", "d.jsonl",
            "--heldout-ratio", "0.2",
            "--seed", "7",
            "--out-train", "t.jsonl",
            "--out-heldout", "h.jsonl",
        ])
        assert args.heldout_ratio == 0.2
        assert args.seed == 7
        assert args.out_train == "t.jsonl"
        assert args.out_heldout == "h.jsonl"

    def test_heldout_ratio_rejects_out_of_range(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["data", "split", "d.jsonl", "--heldout-ratio", "2"])

    def test_seed_rejects_negative(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["data", "split", "d.jsonl", "--seed", "-1"])


class TestDataSplitHandler:
    def _write_rows(self, path, n=10):
        path.write_text(
            "\n".join(json.dumps({"text": f"row {i}"}) for i in range(n)),
            encoding="utf-8",
        )

    def test_round_trip_default_paths(self, tmp_path, capsys):
        src = tmp_path / "data.jsonl"
        self._write_rows(src, n=10)
        ns = _data_split_ns(dataset=str(src), heldout_ratio=0.2, seed=0)
        rc = cmd_data_split(ns)
        assert rc == EXIT_OK

        train_path = tmp_path / "data.train.jsonl"
        heldout_path = tmp_path / "data.heldout.jsonl"
        assert train_path.exists()
        assert heldout_path.exists()

        train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        heldout_rows = [json.loads(line) for line in heldout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        # 20% of 10 rows held out; both sides non-empty; total preserved.
        assert len(train_rows) + len(heldout_rows) == 10
        assert len(heldout_rows) == 2
        assert len(train_rows) == 8
        out = capsys.readouterr().out
        assert "n_train" in out
        assert "n_heldout" in out

    def test_explicit_out_paths(self, tmp_path):
        src = tmp_path / "data.jsonl"
        self._write_rows(src, n=10)
        out_train = tmp_path / "sub" / "mytrain.jsonl"
        out_heldout = tmp_path / "sub" / "myheld.jsonl"
        ns = _data_split_ns(
            dataset=str(src),
            heldout_ratio=0.1,
            out_train=str(out_train),
            out_heldout=str(out_heldout),
        )
        rc = cmd_data_split(ns)
        assert rc == EXIT_OK
        assert out_train.exists()
        assert out_heldout.exists()

    def test_deterministic_split_same_seed(self, tmp_path):
        src = tmp_path / "data.jsonl"
        self._write_rows(src, n=20)
        out1 = tmp_path / "h1.jsonl"
        out2 = tmp_path / "h2.jsonl"
        cmd_data_split(_data_split_ns(dataset=str(src), seed=42, out_heldout=str(out1), out_train=str(tmp_path / "t1.jsonl")))
        cmd_data_split(_data_split_ns(dataset=str(src), seed=42, out_heldout=str(out2), out_train=str(tmp_path / "t2.jsonl")))
        assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")

    def test_missing_file_user_error(self, tmp_path, capsys):
        ns = _data_split_ns(dataset=str(tmp_path / "nope.jsonl"))
        rc = cmd_data_split(ns)
        assert rc == EXIT_USER_ERROR
        assert "not found" in capsys.readouterr().err.lower()

    def test_directory_user_error(self, tmp_path, capsys):
        ns = _data_split_ns(dataset=str(tmp_path))
        rc = cmd_data_split(ns)
        assert rc == EXIT_USER_ERROR
        assert "directory" in capsys.readouterr().err.lower()

    def test_empty_dataset_user_error(self, tmp_path, capsys):
        src = tmp_path / "empty.jsonl"
        src.write_text("\n\n", encoding="utf-8")
        ns = _data_split_ns(dataset=str(src))
        rc = cmd_data_split(ns)
        assert rc == EXIT_USER_ERROR
        assert "no parseable rows" in capsys.readouterr().err.lower()

    def test_bad_ratio_too_few_rows_user_error(self, tmp_path, capsys):
        # split_dataset raises InvalidSettingError (CONFIG_INVALID_SETTING) when
        # the ratio leaves a side empty; surfaced as a user error, not a crash.
        src = tmp_path / "tiny.jsonl"
        src.write_text(json.dumps({"text": "only one row"}), encoding="utf-8")
        ns = _data_split_ns(dataset=str(src), heldout_ratio=0.5)
        rc = cmd_data_split(ns)
        assert rc == EXIT_USER_ERROR


# ===========================================================================
# v1.6 C4 — `backprop eval` task-metric / references / gate-metric flags
# ===========================================================================


class TestEvalMetricFlagsParser:
    def test_metric_repeatable(self, cli_parser):
        args = cli_parser.parse_args([
            "eval", "rid",
            "--metric", "token_f1",
            "--metric", "normalized_exact_match",
        ])
        assert args.metric == ["token_f1", "normalized_exact_match"]

    def test_references_and_eval_set_alias(self, cli_parser):
        a = cli_parser.parse_args(["eval", "rid", "--references", "refs.jsonl"])
        b = cli_parser.parse_args(["eval", "rid", "--eval-set", "refs.jsonl"])
        assert a.references == "refs.jsonl"
        assert b.references == "refs.jsonl"

    def test_gate_metric_repeatable(self, cli_parser):
        args = cli_parser.parse_args([
            "eval", "rid", "--gate-against", "base",
            "--gate-metric", "token_f1", "--gate-metric", "contains",
        ])
        assert args.gate_metric == ["token_f1", "contains"]

    def test_metric_rejects_unknown_name(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["eval", "rid", "--metric", "bleu"])

    def test_metric_choices_match_constant(self, cli_parser):
        # Every advertised metric name parses.
        for m in EVAL_METRIC_CHOICES:
            args = cli_parser.parse_args(["eval", "rid", "--metric", m])
            assert args.metric == [m]


class TestEvalReferencesThreaded:
    """References JSONL is loaded + passed to evaluate_run; metrics/gate_metrics
    thread to evaluate_run / eval_gate. evaluate_run is mocked."""

    def _refs_file(self, tmp_path):
        p = tmp_path / "refs.jsonl"
        p.write_text(
            "\n".join([
                json.dumps({"prompt": "2+2?", "reference": "4"}),
                json.dumps({"prompt": "cap of France?", "references": ["Paris"]}),
            ]),
            encoding="utf-8",
        )
        return p

    def test_references_loaded_and_metrics_passed(self, tmp_path):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        refs = self._refs_file(tmp_path)
        ns = _eval_ns(
            run_id="real", output=str(out_dir), json=True,
            references=str(refs), metric=["token_f1"],
        )
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "real"}), \
             patch("backpropagate.eval.evaluate_run", return_value=_fake_eval_result()) as ev:
            rc = cmd_eval(ns)
        assert rc == EXIT_OK
        _, kwargs = ev.call_args
        assert kwargs["metrics"] == ["token_f1"]
        # references parsed into a list of 2 dicts.
        assert isinstance(kwargs["references"], list)
        assert len(kwargs["references"]) == 2
        assert kwargs["references"][0]["prompt"] == "2+2?"

    def test_missing_references_file_user_error(self, tmp_path, capsys):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        ns = _eval_ns(run_id="real", output=str(out_dir), references=str(tmp_path / "nope.jsonl"))
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "real"}):
            rc = cmd_eval(ns)
        assert rc == EXIT_USER_ERROR
        assert "references" in capsys.readouterr().err.lower()

    def test_empty_references_file_user_error(self, tmp_path, capsys):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        empty = tmp_path / "empty.jsonl"
        empty.write_text("\n", encoding="utf-8")
        ns = _eval_ns(run_id="real", output=str(out_dir), references=str(empty))
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "real"}):
            rc = cmd_eval(ns)
        assert rc == EXIT_USER_ERROR
        assert "no parseable reference" in capsys.readouterr().err.lower()

    def test_gate_metric_threaded_to_eval_gate(self, tmp_path):
        from backpropagate.eval import EvalGateDecision

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        refs = self._refs_file(tmp_path)
        ns = _eval_ns(
            run_id="after", gate_against="before", output=str(out_dir), json=True,
            references=str(refs), gate_metric=["token_f1"],
        )
        decision = EvalGateDecision(accept=True, reason="ok", regression=-0.1)
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "x"}), \
             patch("backpropagate.eval.evaluate_run", side_effect=[_fake_eval_result("after", 0.9), _fake_eval_result("before", 1.0)]), \
             patch("backpropagate.eval.eval_gate", return_value=decision) as gate:
            rc = cmd_eval(ns)
        assert rc == EXIT_OK
        _, kwargs = gate.call_args
        assert kwargs.get("gated_metrics") == ["token_f1"]

    def test_no_references_keeps_metrics_none(self, tmp_path):
        # Backward-compat: when --references is absent, metrics/references both
        # pass through as None and behavior is unchanged.
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        ns = _eval_ns(run_id="real", output=str(out_dir), json=True)
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "real"}), \
             patch("backpropagate.eval.evaluate_run", return_value=_fake_eval_result()) as ev:
            rc = cmd_eval(ns)
        assert rc == EXIT_OK
        _, kwargs = ev.call_args
        assert kwargs["metrics"] is None
        assert kwargs["references"] is None

    def test_task_metrics_rendered_in_human_output(self, tmp_path, capsys):
        from backpropagate.eval import EvalResult

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        refs = self._refs_file(tmp_path)
        result = EvalResult(
            run_id="real",
            model_name="fake/model",
            held_out_loss=1.0,
            perplexity=2.0,
            generations=[],
            n_prompts=0,
            task_metrics={"token_f1": 0.75},
            eval_n=2,
            metric_ci={"token_f1": 0.05},
        )
        ns = _eval_ns(run_id="real", output=str(out_dir), json=False, references=str(refs), metric=["token_f1"])
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "real"}), \
             patch("backpropagate.eval.evaluate_run", return_value=result):
            rc = cmd_eval(ns)
        assert rc == EXIT_OK
        out = capsys.readouterr().out
        assert "token_f1" in out
        assert "Task metrics" in out
