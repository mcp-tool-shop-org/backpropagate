"""Coverage tests for ``cmd_eval`` in cli.py.

Real: run history on ``tmp_path`` (``RunHistoryManager``), heldout / prompts /
references files, the real ``EvalResult`` / ``diff_evals`` / ``eval_gate``,
exit codes and output. Mocked (real boundary: model load + generation):
``backpropagate.eval.evaluate_run`` returns canned ``EvalResult`` objects and
records the kwargs the CLI hands it.
"""

from __future__ import annotations

import builtins

import pytest

from backpropagate import cli
from backpropagate.eval import EvalResult
from tests.helpers.cli_cov_support import last_json, parse, seed_runs


@pytest.fixture
def out_dir(tmp_path):
    out = tmp_path / "out"
    seed_runs(out, [{"run_id": "run-aaaa-0001"}, {"run_id": "run-bbbb-0002"}])
    return out


@pytest.fixture
def evaluate(monkeypatch):
    calls: list = []
    results = {
        "run-aaaa-0001": EvalResult(run_id="run-aaaa-0001", model_name="tiny", held_out_loss=1.0,
                                    perplexity=2.718, n_prompts=3),
        "run-bbbb-0002": EvalResult(run_id="run-bbbb-0002", model_name="tiny", held_out_loss=1.5,
                                    perplexity=4.48, n_prompts=3),
    }

    def fake_evaluate_run(run_id, **kw):
        calls.append((run_id, kw))
        return results[next(k for k in results if k.startswith(run_id))]

    monkeypatch.setattr("backpropagate.eval.evaluate_run", fake_evaluate_run)
    return SimpleCalls(calls, results)


class SimpleCalls:
    def __init__(self, calls, results):
        self.calls = calls
        self.results = results


def _argv(out_dir, *extra, run="run-aaaa"):
    return ["eval", run, "--output", str(out_dir), *extra]


class TestEvalResolution:
    def test_missing_output_dir(self, tmp_path, capsys):
        args = parse(["eval", "x", "--output", str(tmp_path / "no")])
        assert cli.cmd_eval(args) == cli.EXIT_USER_ERROR
        assert "No output directory" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "extra, label",
        [((), "run_id"), (("--vs", "zzz"), "--vs"), (("--gate-against", "zzz"), "--gate-against")],
    )
    def test_unknown_run_ids(self, out_dir, capsys, extra, label):
        run = "zzz" if label == "run_id" else "run-aaaa"
        assert cli.cmd_eval(parse(_argv(out_dir, *extra, run=run))) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert f"{label}='zzz' not found" in captured.err
        assert "backprop runs --output" in captured.out

    def test_log_failures_are_ignored_on_error_paths(self, out_dir, tmp_path, monkeypatch, capsys):
        def boom(name):
            raise RuntimeError("log down")

        monkeypatch.setattr("backpropagate.logging_config.get_logger", boom)
        monkeypatch.setattr("backpropagate.logging_config.bind_run_context", boom)
        assert cli.cmd_eval(parse(["eval", "x", "--output", str(tmp_path / "no")])) == cli.EXIT_USER_ERROR
        assert cli.cmd_eval(parse(_argv(out_dir, run="zzz"))) == cli.EXIT_USER_ERROR
        assert cli.cmd_eval(parse(_argv(out_dir, "--heldout", str(tmp_path / "h.jsonl")))) == cli.EXIT_USER_ERROR
        assert cli.cmd_eval(parse(_argv(out_dir, "--prompts", str(tmp_path / "p.txt")))) == cli.EXIT_USER_ERROR
        assert cli.cmd_eval(parse(_argv(out_dir, "--references", str(tmp_path / "r.jsonl")))) == cli.EXIT_USER_ERROR
        capsys.readouterr()

    def test_heldout_unresolved(self, out_dir, tmp_path, capsys):
        args = parse(_argv(out_dir, "--heldout", str(tmp_path / "missing.jsonl")))
        assert cli.cmd_eval(args) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "--heldout dataset not found" in captured.err and "held-out split" in captured.out

    def test_heldout_directory(self, out_dir, tmp_path, capsys):
        assert cli.cmd_eval(parse(_argv(out_dir, "--heldout", str(tmp_path)))) == cli.EXIT_USER_ERROR

    def test_prompts_unresolved(self, out_dir, tmp_path, capsys):
        args = parse(_argv(out_dir, "--prompts", str(tmp_path / "missing.txt")))
        assert cli.cmd_eval(args) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "--prompts file not found" in captured.err and "one prompt per" in captured.out

    def test_references_unresolved(self, out_dir, tmp_path, capsys):
        args = parse(_argv(out_dir, "--references", str(tmp_path / "missing.jsonl")))
        assert cli.cmd_eval(args) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "--references/--eval-set file not found" in captured.err
        assert '"prompt"' in captured.out

    def test_references_not_utf8_unreadable_and_empty(self, out_dir, tmp_path, monkeypatch, capsys):
        latin = tmp_path / "latin.jsonl"
        latin.write_bytes(b'{"prompt": "caf\xe9"}\n')
        assert cli.cmd_eval(parse(_argv(out_dir, "--references", str(latin)))) == cli.EXIT_USER_ERROR
        assert "is not valid UTF-8" in capsys.readouterr().err

        empty = tmp_path / "empty.jsonl"
        empty.write_text("\n{bad json\n", encoding="utf-8")
        assert cli.cmd_eval(parse(_argv(out_dir, "--references", str(empty)))) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "has no parseable reference rows" in captured.err and "Each line must be" in captured.out

        good = tmp_path / "good.jsonl"
        good.write_text('{"prompt": "p", "reference": "r"}\n', encoding="utf-8")
        real_open = builtins.open

        def deny(path, *a, **k):
            if str(path) == str(good):
                raise PermissionError("denied")
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", deny)
        assert cli.cmd_eval(parse(_argv(out_dir, "--references", str(good)))) == cli.EXIT_USER_ERROR
        assert "Could not read --references/--eval-set: denied" in capsys.readouterr().err


class TestEvalSingle:
    def test_human_output_and_forwarded_kwargs(self, out_dir, tmp_path, evaluate, capsys):
        heldout = tmp_path / "held.jsonl"
        heldout.write_text('{"text": "x"}\n', encoding="utf-8")
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("hello\n", encoding="utf-8")
        refs = tmp_path / "refs.jsonl"
        refs.write_text('{"prompt": "p1", "reference": "r1"}\n\nnot json\n{"prompt": "p2", "reference": "r2"}\n',
                        encoding="utf-8")
        argv = _argv(out_dir, "--heldout", str(heldout), "--prompts", str(prompts), "--references", str(refs),
                     "--num-samples", "7", "--seed", "11", "--max-new-tokens", "33",
                     "--metric", "token_f1", "--metric", "contains")
        assert cli.cmd_eval(parse(argv)) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Held-out loss: 1.0000" in out and "Perplexity: 2.7180" in out and "Prompts: 3" in out
        assert "Eval complete" in out
        run_id, kw = evaluate.calls[0]
        assert run_id == "run-aaaa"
        assert kw["heldout"] == str(heldout) and kw["prompts"] == str(prompts)
        assert (kw["n"], kw["seed"], kw["max_new_tokens"]) == (7, 11, 33)
        assert kw["metrics"] == ["token_f1", "contains"]
        assert kw["references"] == [{"prompt": "p1", "reference": "r1"}, {"prompt": "p2", "reference": "r2"}]

    def test_task_metrics_block(self, out_dir, monkeypatch, capsys):
        result = EvalResult(run_id="run-aaaa-0001", model_name="tiny", held_out_loss=None, perplexity=None,
                            task_metrics={"token_f1": 0.5, "contains": 1.0}, eval_n=4,
                            metric_ci={"token_f1": 0.125})
        monkeypatch.setattr("backpropagate.eval.evaluate_run", lambda run_id, **kw: result)
        assert cli.cmd_eval(parse(_argv(out_dir))) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Held-out loss: n/a" in out and "Perplexity: n/a" in out
        assert "Task metrics:" in out
        assert "token_f1: 0.5000 (+/- 0.1250)" in out
        assert "contains: 1.0000" in out and "(+/-" not in out.split("contains:")[1].splitlines()[0]
        assert "Scored over: 4 held-out reference items" in out

    def test_task_metrics_without_eval_n_omit_scored_over(self, out_dir, monkeypatch, capsys):
        result = EvalResult(run_id="run-aaaa-0001", model_name="tiny", held_out_loss=0.5, perplexity=1.6,
                            task_metrics={"contains": 0.75}, eval_n=0)
        monkeypatch.setattr("backpropagate.eval.evaluate_run", lambda run_id, **kw: result)
        assert cli.cmd_eval(parse(_argv(out_dir))) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "contains: 0.7500" in out and "Scored over" not in out

    def test_json(self, out_dir, evaluate, capsys):
        assert cli.cmd_eval(parse(_argv(out_dir, "--json"))) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        assert payload["mode"] == "single" and payload["run_id"] == "run-aaaa"
        assert payload["result"]["held_out_loss"] == 1.0


class TestEvalVs:
    def test_table(self, out_dir, evaluate, capsys):
        assert cli.cmd_eval(parse(_argv(out_dir, "--vs", "run-bbbb"))) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Eval diff: run-aaaa vs run-bbbb" in out
        assert "METRIC" in out and "held_out_loss" in out
        assert [c[0] for c in evaluate.calls] == ["run-aaaa", "run-bbbb"]

    def test_json(self, out_dir, evaluate, capsys):
        assert cli.cmd_eval(parse(_argv(out_dir, "--vs", "run-bbbb", "--json"))) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        assert payload["mode"] == "diff"
        assert payload["run_a"] == "run-aaaa" and payload["run_b"] == "run-bbbb"
        assert any(row[0] == "held_out_loss" for row in payload["diff"]["rows"])

    def test_empty_diff_rows_fall_back_to_repr(self, out_dir, evaluate, monkeypatch, capsys):
        class EmptyDiff:
            rows: list = []

            def __str__(self):
                return "<empty diff>"

        monkeypatch.setattr("backpropagate.eval.diff_evals", lambda a, b: EmptyDiff())
        assert cli.cmd_eval(parse(_argv(out_dir, "--vs", "run-bbbb"))) == cli.EXIT_OK
        assert "<empty diff>" in capsys.readouterr().out


class TestEvalGate:
    def test_regression_trips_gate(self, out_dir, evaluate, capsys):
        # run-bbbb (loss 1.5) is the candidate, run-aaaa (loss 1.0) the baseline.
        argv = _argv(out_dir, "--gate-against", "run-aaaa", "--max-regression", "0.1", run="run-bbbb")
        assert cli.cmd_eval(parse(argv)) == cli.EXIT_DATA_ERR
        captured = capsys.readouterr()
        assert "Eval gate: REJECT" in captured.err
        assert "Regression: 0.5" in captured.out

    def test_improvement_is_accepted(self, out_dir, evaluate, capsys):
        argv = _argv(out_dir, "--gate-against", "run-bbbb", run="run-aaaa")
        assert cli.cmd_eval(parse(argv)) == cli.EXIT_OK
        assert "Eval gate: ACCEPT" in capsys.readouterr().out

    def test_gate_json_and_gate_metrics(self, out_dir, evaluate, capsys):
        argv = _argv(out_dir, "--gate-against", "run-aaaa", "--json", "--gate-metric", "token_f1", run="run-bbbb")
        assert cli.cmd_eval(parse(argv)) == cli.EXIT_DATA_ERR
        payload = last_json(capsys.readouterr().out)
        assert payload["mode"] == "gate" and payload["accept"] is False
        assert payload["baseline_run_id"] == "run-aaaa"
        assert payload["result"]["run_id"] == "run-bbbb-0002"

    def test_gate_log_failure_does_not_change_exit(self, out_dir, evaluate, monkeypatch):
        def boom(name):
            raise RuntimeError("log down")

        monkeypatch.setattr("backpropagate.logging_config.get_logger", boom)
        argv = _argv(out_dir, "--gate-against", "run-aaaa", run="run-bbbb")
        assert cli.cmd_eval(parse(argv)) == cli.EXIT_DATA_ERR
