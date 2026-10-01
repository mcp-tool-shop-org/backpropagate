"""pass_rate runs model-generated code: opt-in + out-of-process isolation.

Every snippet below is a FIXED, hand-written string. No test here ever feeds
model output to the harness.

Run against ``main``'s ``eval.py`` (in-process ``exec``, no opt-in), the opt-in,
timeout, environment-isolation and working-directory tests fail.
"""

from __future__ import annotations

import os
import time
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

import backpropagate.eval as ev
from backpropagate.cli import create_parser
from backpropagate.exceptions import UserInputError

ENV = "BACKPROPAGATE_ALLOW_CODE_EVAL"

# Hand-written escape from the restricted builtins: reaches the real ``os``
# through ``os._wrap_close.__init__.__globals__['__builtins__']``. It is here to
# prove the CHILD PROCESS (not the builtins) is what hides the parent's state.
GET_OS = (
    "os = [c for c in ().__class__.__base__.__subclasses__() if c.__name__ == '_wrap_close']"
    "[0].__init__.__globals__['__builtins__']['__import__']('os')\n"
)

ADD_OK = "def add(a, b):\n    return a + b\n"


@pytest.fixture(autouse=True)
def _no_ambient_opt_in(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


@pytest.fixture
def allowed(monkeypatch):
    monkeypatch.setenv(ENV, "1")


# ---------------------------------------------------------------------------
# Opt-in
# ---------------------------------------------------------------------------


class TestOptIn:
    def test_refused_without_opt_in(self):
        with patch.object(ev, "_run_code_eval_child", side_effect=AssertionError("code ran")):
            with pytest.raises(UserInputError) as exc:
                ev.pass_rate(ADD_OK, ["assert add(1, 2) == 3"])
        assert exc.value.code == "INPUT_VALIDATION_FAILED"
        hint = exc.value.suggestion or ""
        assert ENV in hint
        assert "--allow-code-exec" in hint
        assert "NOT a sandbox" in hint

    @pytest.mark.parametrize("value", ["", "0", "no", "false", "off"])
    def test_falsey_env_values_do_not_opt_in(self, monkeypatch, value):
        monkeypatch.setenv(ENV, value)
        with pytest.raises(UserInputError):
            ev.pass_rate(ADD_OK, ["assert True"])

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_env_values_opt_in(self, monkeypatch, value):
        monkeypatch.setenv(ENV, value)
        assert ev.pass_rate(ADD_OK, ["assert add(1, 2) == 3"]) == pytest.approx(1.0)

    def test_explicit_argument_opts_in_without_env(self):
        score = ev.pass_rate(ADD_OK, ["assert add(1, 2) == 3"], allow_code_execution=True)
        assert score == pytest.approx(1.0)

    def test_explicit_false_without_env_is_refused(self):
        with pytest.raises(UserInputError):
            ev.pass_rate(ADD_OK, ["assert True"], allow_code_execution=False)

    def test_compute_task_metric_refuses_without_opt_in(self):
        with pytest.raises(UserInputError) as exc:
            ev.compute_task_metric("pass_rate", ADD_OK, ["assert True"])
        assert exc.value.code == "INPUT_VALIDATION_FAILED"

    def test_compute_task_metric_threads_argument_and_timeout(self):
        calls = []

        def fake(prediction, snippets, *, allow_code_execution=None, timeout=None):
            calls.append((allow_code_execution, timeout))
            return 0.25

        with patch.dict(ev.TASK_METRICS, {"pass_rate": fake}):
            score = ev.compute_task_metric(
                "pass_rate", "x", ["y"], allow_code_execution=True, code_exec_timeout=3.0
            )
        assert score == 0.25
        assert calls == [(True, 3.0)]

    def test_other_metrics_ignore_the_flag(self):
        assert ev.compute_task_metric(
            "contains", "abc", ["b"], allow_code_execution=True
        ) == 1.0

    def test_non_positive_timeout_is_a_user_error(self, allowed):
        with pytest.raises(UserInputError) as exc:
            ev.pass_rate(ADD_OK, ["assert True"], timeout=0)
        assert exc.value.code == "INPUT_VALIDATION_FAILED"

    def test_evaluate_run_refuses_before_loading_the_model(self, tmp_path):
        history = MagicMock()
        history.get_run.return_value = {
            "run_id": "abc123def456",
            "model_name": "unsloth/Qwen2.5-3B",
            "dataset_info": "InMemoryDataset",
            "hyperparameters": {"max_seq_length": 256},
        }
        with patch.object(
            ev, "_load_model_and_tokenizer", side_effect=AssertionError("model loaded")
        ), patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
            with pytest.raises(UserInputError) as exc:
                ev.evaluate_run(
                    "abc123def456",
                    output_dir=str(tmp_path),
                    heldout_texts=["held out"],
                    metrics=["pass_rate"],
                    references=[{"prompt": "write add", "reference": "assert True"}],
                )
        assert exc.value.code == "INPUT_VALIDATION_FAILED"

    def test_task_metrics_scores_pass_rate_when_allowed(self):
        gens = [ev.GenerationSample(prompt="p", completion=ADD_OK)]
        refs = [{"prompt": "p", "references": ["assert add(1, 2) == 3", "assert add(1, 2) == 4"]}]
        metrics, n, _ci = ev._compute_task_metrics(
            gens, refs, ["pass_rate"], allow_code_execution=True
        )
        assert n == 1
        assert metrics["pass_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Scoring semantics are unchanged
# ---------------------------------------------------------------------------


class TestScoring:
    def test_passing_code_scores_one(self, allowed):
        assert ev.pass_rate(ADD_OK, ["assert add(2, 3) == 5"]) == pytest.approx(1.0)

    def test_failing_assert_scores_zero(self, allowed):
        assert ev.pass_rate("def add(a, b):\n    return a - b\n", ["assert add(2, 3) == 5"]) == 0.0

    def test_partial_pass_is_a_fraction(self, allowed):
        tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 4", "assert add(0, 0) == 0"]
        assert ev.pass_rate(ADD_OK, tests) == pytest.approx(2 / 3)

    def test_syntax_error_in_prediction_scores_zero(self, allowed):
        assert ev.pass_rate("def broken(:\n", ["assert True", "assert True"]) == 0.0

    def test_exception_while_defining_scores_zero(self, allowed):
        assert ev.pass_rate("raise RuntimeError('boom')\n", ["assert True"]) == 0.0

    def test_syntax_error_in_one_snippet_only_fails_that_snippet(self, allowed):
        tests = ["assert add(1, 2) == 3", "assert add(1, 2) ==", "assert add(2, 2) == 4"]
        assert ev.pass_rate(ADD_OK, tests) == pytest.approx(2 / 3)

    def test_snippets_do_not_share_state(self, allowed):
        not_shared = (
            "try:\n    leaked\nexcept Exception:\n    ok = True\n"
            "else:\n    raise AssertionError('state leaked between snippets')\n"
        )
        assert ev.pass_rate("x = 0\n", ["leaked = 1", not_shared]) == pytest.approx(1.0)

    def test_system_exit_in_a_snippet_is_a_failed_snippet_not_a_crash(self, allowed):
        tests = ["raise SystemExit(0)", "assert True"]
        assert ev.pass_rate("x = 0\n", tests) == pytest.approx(0.5)

    def test_stray_writes_to_fd1_do_not_corrupt_the_report(self, allowed):
        # The real report is the LAST line; earlier noise (here a forged
        # report written straight to fd 1) is ignored.
        noise = GET_OS + "os.write(1, b'{\"results\": [false]}\\n')\nassert True\n"
        assert ev.pass_rate("x = 0\n", [noise]) == pytest.approx(1.0)

    def test_empty_snippets_score_zero(self, allowed):
        assert ev.pass_rate(ADD_OK, []) == 0.0

    def test_restricted_builtins_still_block_import(self, allowed):
        assert ev.pass_rate("import os", ["assert True"]) == 0.0

    def test_bare_string_is_one_snippet(self, allowed):
        assert ev.pass_rate(ADD_OK, "assert add(5, 5) == 10") == 1.0

    def test_non_ascii_round_trips(self, allowed):
        code = "def f():\n    return 'café ☃'\n"
        assert ev.pass_rate(code, ["assert f() == 'café ☃'"]) == 1.0


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_infinite_loop_in_prediction_times_out_and_scores_zero(self, allowed):
        start = time.monotonic()
        score = ev.pass_rate("while True:\n    pass\n", ["assert True"], timeout=1.0)
        elapsed = time.monotonic() - start
        assert score == 0.0
        assert elapsed < 10.0

    def test_infinite_loop_in_a_snippet_times_out_and_scores_zero(self, allowed):
        start = time.monotonic()
        score = ev.pass_rate(ADD_OK, ["assert add(1, 2) == 3", "while True:\n    pass\n"], timeout=1.0)
        elapsed = time.monotonic() - start
        assert score == 0.0
        assert elapsed < 10.0

    def test_default_timeout_is_ten_seconds(self):
        assert ev.DEFAULT_CODE_EVAL_TIMEOUT == 10.0


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_escape_snippet_really_reaches_os(self, allowed):
        # Control: proves the escape works, so the isolation tests below are
        # not passing only because the snippet failed to get ``os``.
        assert ev.pass_rate("x = 0\n", [GET_OS + "assert os.getcwd()"]) == 1.0

    def test_parent_env_vars_are_not_visible_to_the_child(self, allowed, monkeypatch):
        monkeypatch.setenv("BP_TEST_SECRET_TOKEN", "super-secret")
        monkeypatch.setenv("HF_TOKEN", "hf_not_for_you")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-you")
        snippet = (
            GET_OS
            + "assert os.environ.get('BP_TEST_SECRET_TOKEN') is None\n"
            + "assert os.environ.get('HF_TOKEN') is None\n"
            + "assert os.environ.get('OPENAI_API_KEY') is None\n"
            + f"assert os.environ.get({ENV!r}) is None\n"
        )
        assert ev.pass_rate("x = 0\n", [snippet]) == 1.0

    def test_child_runs_in_a_fresh_temp_directory(self, allowed, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        marker = tmp_path.resolve().as_posix()
        snippet = (
            GET_OS
            + f"assert os.path.realpath(os.getcwd()).replace(os.sep, '/') != {marker!r}\n"
            + "assert 'bp_code_eval_' in os.getcwd()\n"
        )
        assert ev.pass_rate("x = 0\n", [snippet]) == 1.0

    def test_child_is_an_isolated_interpreter_with_closed_stdin(self, allowed):
        snippet = (
            GET_OS
            + "sys = os.sys\n"
            + "assert sys.flags.isolated == 1\n"
            + "assert sys.stdin.read() == ''\n"
        )
        assert ev.pass_rate("x = 0\n", [snippet]) == 1.0

    def test_temp_directory_is_removed_afterwards(self, allowed, tmp_path):
        seen = []
        real = ev._code_eval_child_env

        def spy(workdir):
            seen.append(workdir)
            return real(workdir)

        with patch.object(ev, "_code_eval_child_env", side_effect=spy):
            ev.pass_rate("x = 0\n", ["assert True"])
        assert seen and not os.path.exists(seen[0])

    def test_minimal_env_has_no_secrets(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_x")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
        env = ev._code_eval_child_env("/some/dir")
        assert "HF_TOKEN" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env
        assert env["TMPDIR"] == "/some/dir"


class TestReportHandling:
    """A malformed child report is a 0, never a crash or a wrong score."""

    @pytest.mark.parametrize(
        "program",
        [
            "import sys; sys.exit(3)",  # no report at all
            "print('not json')",  # not JSON
            "print('[1, 2]')",  # JSON, wrong shape
            "print('{\"results\": [true, true, true]}')",  # wrong length
            "print('{\"results\": [1]}')",  # non-bool entries
            "print('x' * 2000000)",  # over the output cap
        ],
    )
    def test_bad_reports_score_zero(self, allowed, program):
        with patch.object(ev, "_CODE_EVAL_CHILD", program):
            assert ev.pass_rate("x = 0\n", ["assert True"]) == 0.0

    def test_good_report_from_stub_child_is_used(self, allowed):
        with patch.object(ev, "_CODE_EVAL_CHILD", "print('{\"results\": [true, false]}')"):
            assert ev.pass_rate("x = 0\n", ["a", "b"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestCliSurface:
    def test_flags_parse(self):
        args = create_parser().parse_args(
            ["eval", "rid", "--metric", "pass_rate", "--allow-code-exec", "--code-exec-timeout", "2.5"]
        )
        assert args.allow_code_exec is True
        assert args.code_exec_timeout == 2.5

    def test_flags_default_off(self):
        args = create_parser().parse_args(["eval", "rid"])
        assert args.allow_code_exec is False
        assert args.code_exec_timeout is None

    def test_cmd_eval_forwards_opt_in_and_timeout(self, tmp_path):
        from backpropagate.cli import cmd_eval

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        refs = tmp_path / "refs.jsonl"
        refs.write_text('{"prompt": "p", "reference": "assert True"}\n', encoding="utf-8")
        ns = Namespace(
            run_id="real", vs=None, gate_against=None, output=str(out_dir), heldout=None,
            prompts=None, num_samples=5, max_new_tokens=128, max_regression=0.0, seed=0,
            json=True, verbose=False, cli_run_id="deadbeefcafe", metric=["pass_rate"],
            references=str(refs), gate_metric=None, allow_code_exec=True, code_exec_timeout=4.0,
        )
        result = ev.EvalResult(
            run_id="real", model_name="m", held_out_loss=1.0, perplexity=2.0,
            generations=[], n_prompts=0,
        )
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "real"}), \
             patch("backpropagate.eval.evaluate_run", return_value=result) as run:
            cmd_eval(ns)
        _, kwargs = run.call_args
        assert kwargs["allow_code_execution"] is True
        assert kwargs["code_exec_timeout"] == 4.0

    def test_cmd_eval_omits_the_kwargs_by_default(self, tmp_path):
        from backpropagate.cli import cmd_eval

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        ns = Namespace(
            run_id="real", vs=None, gate_against=None, output=str(out_dir), heldout=None,
            prompts=None, num_samples=5, max_new_tokens=128, max_regression=0.0, seed=0,
            json=True, verbose=False, cli_run_id="deadbeefcafe", metric=None,
            references=None, gate_metric=None, allow_code_exec=False, code_exec_timeout=None,
        )
        result = ev.EvalResult(
            run_id="real", model_name="m", held_out_loss=1.0, perplexity=2.0,
            generations=[], n_prompts=0,
        )
        with patch("backpropagate.checkpoints.RunHistoryManager.get_run", return_value={"run_id": "real"}), \
             patch("backpropagate.eval.evaluate_run", return_value=result) as run:
            cmd_eval(ns)
        _, kwargs = run.call_args
        assert "allow_code_execution" not in kwargs
        assert "code_exec_timeout" not in kwargs
