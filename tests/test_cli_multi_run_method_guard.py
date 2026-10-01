"""`backprop multi-run --method` refuses objectives the backend cannot run.

MultiRunConfig / MultiRunTrainer take no ``method``, so through v1.7.2 the CLI's
signature filter dropped ``--method orpo|simpo|kto`` and every run trained SFT
without a word. The CLI now refuses a non-SFT method. MultiRunTrainer is replaced
by a recorder (no model load); MultiRunConfig and the parser are real.
"""

from __future__ import annotations

import pytest

import backpropagate.cli as cli
import backpropagate.multi_run as mr


def _parse(argv):
    return cli.create_parser().parse_args(argv)


@pytest.fixture
def recorder(monkeypatch):
    calls = []

    class _Rec:
        def __init__(self, model, config, on_run_complete=None, resume_from=None):
            calls.append({"model": model, "config": config})

        def run(self, data):
            raise SystemExit("stop after construction")

    monkeypatch.setattr(mr, "MultiRunTrainer", _Rec)
    return calls


@pytest.mark.parametrize("method", ["orpo", "simpo", "kto"])
def test_non_sft_method_is_refused(method, recorder, capsys, tmp_path):
    args = _parse(["multi-run", "--data", str(tmp_path / "d.jsonl"), "--method", method])
    assert cli.cmd_multi_run(args) == cli.EXIT_USER_ERROR
    out = capsys.readouterr()
    text = out.out + out.err
    assert f"--method {method} is not supported by multi-run" in text
    assert recorder == []  # refused before any trainer was built


def test_sft_method_still_builds_the_trainer(recorder, tmp_path):
    args = _parse(["multi-run", "--data", str(tmp_path / "d.jsonl"), "--method", "sft"])
    with pytest.raises(SystemExit, match="stop after construction"):
        cli.cmd_multi_run(args)
    assert len(recorder) == 1


def test_runs_payload_survives_huge_int_loss(tmp_path):
    """An int above float range in loss_history must not crash `runs --json`."""
    payload = cli._build_runs_payload(
        [{"run_id": "r1", "status": "completed", "loss_history": [10**400, 0.5],
          "final_loss": 10**400}],
        tmp_path,
    )
    (entry,) = [r for r in payload["runs"] if r.get("run_id") == "r1"]
    assert entry["loss"] == {"final": None, "min": None}
