"""Tests for the v1.6 deterministic, judge-free eval task-metrics (Contract C3).

Covers the metric primitives (``normalized_exact_match``, ``token_f1``,
``contains``, ``regex``, ``pass_rate``), the SQuAD normalization helper, the
``compute_task_metric`` dispatcher, the bootstrap CI half-width helper, the
references -> ``task_metrics`` path threaded through ``evaluate_run``, and the
upgraded conjunction ``eval_gate`` (loss floor AND task-metric noise-band).

torch is MOCKED where ``evaluate_run`` runs (same fakes as test_eval.py) so no
real model download / CUDA. The metric primitives are pure and need no fakes.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import backpropagate.eval as ev
from backpropagate.eval import (
    EvalGateDecision,
    EvalResult,
    GenerationSample,
    bootstrap_ci_halfwidth,
    compute_task_metric,
    contains_match,
    eval_gate,
    evaluate_run,
    normalize_squad_text,
    normalized_exact_match,
    pass_rate,
    regex_match,
    token_f1,
)
from backpropagate.exceptions import UserInputError


# =============================================================================
# SQuAD normalization
# =============================================================================

class TestNormalizeSquadText:
    def test_lowercases(self):
        assert normalize_squad_text("HeLLo") == "hello"

    def test_strips_punctuation(self):
        assert normalize_squad_text("hello, world!") == "hello world"

    def test_drops_articles(self):
        # a / an / the are dropped as whole tokens.
        assert normalize_squad_text("the cat and a dog") == "cat and dog"
        assert normalize_squad_text("an apple") == "apple"

    def test_does_not_drop_article_substring(self):
        # "theme" must NOT lose its leading "the" (article drop is token-level).
        assert normalize_squad_text("theme") == "theme"

    def test_collapses_whitespace(self):
        assert normalize_squad_text("  hello   world  ") == "hello world"

    def test_combined(self):
        assert normalize_squad_text("The Quick, Brown FOX!") == "quick brown fox"


# =============================================================================
# normalized_exact_match
# =============================================================================

class TestNormalizedExactMatch:
    def test_exact_after_normalization(self):
        assert normalized_exact_match("The Paris.", ["paris"]) == 1.0

    def test_no_match(self):
        assert normalized_exact_match("London", ["paris"]) == 0.0

    def test_takes_best_over_multiple_references(self):
        # Any reference matching -> 1.0 (max over references).
        assert normalized_exact_match("paris", ["london", "Paris"]) == 1.0

    def test_empty_references_is_zero(self):
        assert normalized_exact_match("paris", []) == 0.0


# =============================================================================
# token_f1 (SQuAD-style token-overlap F1)
# =============================================================================

class TestTokenF1:
    def test_perfect_overlap(self):
        assert token_f1("the quick brown fox", ["quick brown fox"]) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # pred tokens: {cat, sat}; ref tokens: {cat, ran}
        # precision = 1/2, recall = 1/2, f1 = 0.5
        assert token_f1("the cat sat", ["a cat ran"]) == pytest.approx(0.5)

    def test_no_overlap(self):
        assert token_f1("apple", ["orange"]) == pytest.approx(0.0)

    def test_takes_best_over_multiple_references(self):
        score = token_f1("cat dog", ["bird fish", "cat dog"])
        assert score == pytest.approx(1.0)

    def test_empty_prediction_and_reference_is_one(self):
        # SQuAD convention: empty-vs-empty counts as a match.
        assert token_f1("", [""]) == pytest.approx(1.0)

    def test_empty_prediction_nonempty_reference_is_zero(self):
        assert token_f1("", ["something"]) == pytest.approx(0.0)


# =============================================================================
# contains / regex
# =============================================================================

class TestContains:
    def test_contains_case_insensitive_substring(self):
        assert contains_match("The answer is PARIS, France.", ["paris"]) == 1.0

    def test_contains_miss(self):
        assert contains_match("the answer is london", ["paris"]) == 0.0

    def test_contains_any_reference(self):
        assert contains_match("answer: 42", ["7", "42"]) == 1.0


class TestRegex:
    def test_regex_match(self):
        assert regex_match("the total is 1234 dollars", [r"\d{4}"]) == 1.0

    def test_regex_miss(self):
        assert regex_match("no digits here", [r"\d{4}"]) == 0.0

    def test_invalid_regex_raises_user_input_error(self):
        with pytest.raises(UserInputError) as exc:
            regex_match("anything", ["(unterminated"])
        assert exc.value.code == "INPUT_VALIDATION_FAILED"


# =============================================================================
# pass_rate (best-effort code execution against test snippets)
# =============================================================================

class TestPassRate:
    def test_passing_code_scores_one(self):
        code = "def add(a, b):\n    return a + b\n"
        tests = ["assert add(2, 3) == 5"]
        assert pass_rate(code, tests) == pytest.approx(1.0)

    def test_failing_assertion_scores_zero(self):
        code = "def add(a, b):\n    return a - b\n"
        tests = ["assert add(2, 3) == 5"]
        assert pass_rate(code, tests) == pytest.approx(0.0)

    def test_partial_pass(self):
        code = "def f(x):\n    return x\n"
        tests = ["assert f(1) == 1", "assert f(1) == 2"]
        assert pass_rate(code, tests) == pytest.approx(0.5)

    def test_no_tests_returns_zero(self):
        assert pass_rate("def f(): pass", []) == pytest.approx(0.0)

    def test_code_with_syntax_error_scores_zero_not_crash(self):
        # A model emitting invalid code must score 0, never crash the eval.
        assert pass_rate("def broken(:\n", ["assert True"]) == pytest.approx(0.0)


# =============================================================================
# compute_task_metric dispatcher
# =============================================================================

class TestComputeTaskMetric:
    def test_dispatches_normalized_exact_match(self):
        assert compute_task_metric("normalized_exact_match", "Paris", ["paris"]) == 1.0

    def test_dispatches_token_f1(self):
        assert compute_task_metric("token_f1", "cat dog", ["cat dog"]) == pytest.approx(1.0)

    def test_dispatches_contains(self):
        assert compute_task_metric("contains", "x paris y", ["paris"]) == 1.0

    def test_dispatches_regex(self):
        assert compute_task_metric("regex", "abc123", [r"\d+"]) == 1.0

    def test_dispatches_pass_rate(self):
        assert compute_task_metric(
            "pass_rate", "def f():\n    return 1\n", ["assert f() == 1"]
        ) == pytest.approx(1.0)

    def test_unknown_metric_raises_user_input_error(self):
        with pytest.raises(UserInputError) as exc:
            compute_task_metric("bleu_score_made_up", "x", ["y"])
        assert exc.value.code == "INPUT_VALIDATION_FAILED"


# =============================================================================
# bootstrap_ci_halfwidth
# =============================================================================

class TestBootstrapCI:
    def test_returns_none_for_too_few_samples(self):
        assert bootstrap_ci_halfwidth([1.0]) is None
        assert bootstrap_ci_halfwidth([]) is None

    def test_zero_variance_gives_zero_halfwidth(self):
        # All identical -> every resample mean equals the value -> hw 0.
        hw = bootstrap_ci_halfwidth([1.0, 1.0, 1.0, 1.0, 1.0], seed=0)
        assert hw == pytest.approx(0.0, abs=1e-9)

    def test_nonzero_variance_gives_positive_halfwidth(self):
        hw = bootstrap_ci_halfwidth([0.0, 1.0] * 50, seed=0)
        assert hw is not None and hw > 0.0

    def test_deterministic_for_fixed_seed(self):
        scores = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        a = bootstrap_ci_halfwidth(scores, seed=7)
        b = bootstrap_ci_halfwidth(scores, seed=7)
        assert a == b


# =============================================================================
# evaluate_run — references -> task_metrics path (mocked model/torch)
# =============================================================================

class _FakeTensor:
    def __init__(self, rows):
        self._rows = rows

    @property
    def shape(self):
        n_rows = len(self._rows)
        n_cols = len(self._rows[0]) if n_rows else 0
        return (n_rows, n_cols)

    def to(self, _device):
        return self

    def __getitem__(self, idx):
        return _FakeRow(self._rows[idx])


class _FakeRow:
    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, sl):
        return _FakeRow(self._values[sl])


class _FakeLoss:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeTorch:
    float16 = "float16"

    def __init__(self):
        self.manual_seed_calls = []

    def manual_seed(self, seed):
        self.manual_seed_calls.append(seed)

    @contextmanager
    def no_grad(self):
        yield


def _make_tokenizer():
    tok = MagicMock()

    def _tokenize(text, **kwargs):
        n = max(1, len(str(text).split()))
        return {
            "input_ids": _FakeTensor([list(range(n))]),
            "attention_mask": _FakeTensor([[1] * n]),
        }

    tok.side_effect = _tokenize
    tok.pad_token = "<pad>"
    tok.eos_token = "</s>"
    tok.decode = MagicMock(return_value="Paris")
    return tok


def _make_model(loss_value: float = 0.5):
    model = MagicMock()
    model.device = None
    model.eval = MagicMock()

    def _forward(**kwargs):
        return SimpleNamespace(loss=_FakeLoss(loss_value))

    model.side_effect = _forward
    model.generate = MagicMock(return_value=_FakeTensor([list(range(64))]))
    return model


def _run_entry(**overrides):
    base = {
        "run_id": "abc123def456",
        "model_name": "unsloth/Qwen2.5-3B",
        "dataset_info": "InMemoryDataset",
        "checkpoint_path": None,
        "hyperparameters": {"max_seq_length": 256},
    }
    base.update(overrides)
    return base


class TestEvaluateRunReferences:
    def test_references_populate_task_metrics_and_eval_n(self, tmp_path):
        """references=[{prompt, reference}] + metrics=[...] => task_metrics filled.

        The tokenizer.decode mock returns 'Paris' for every generation, so
        normalized_exact_match against reference 'paris' scores 1.0.
        """
        model = _make_model(loss_value=0.5)
        tokenizer = _make_tokenizer()
        history = MagicMock()
        history.get_run.return_value = _run_entry()
        history.record_run_completed.return_value = {}

        references = [
            {"prompt": "Capital of France?", "reference": "Paris"},
            {"prompt": "Capital of France again?", "references": ["paris", "PARIS"]},
        ]

        with patch.dict(sys.modules, {"torch": _FakeTorch()}), \
             patch.object(ev, "_load_model_and_tokenizer", return_value=(model, tokenizer)), \
             patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
            result = evaluate_run(
                "abc123def456",
                output_dir=str(tmp_path),
                heldout_texts=["a held out text"],
                metrics=["normalized_exact_match", "token_f1"],
                references=references,
            )

        assert isinstance(result, EvalResult)
        # Both references decoded to "Paris" -> EM 1.0, token_f1 1.0.
        assert result.task_metrics["normalized_exact_match"] == pytest.approx(1.0)
        assert result.task_metrics["token_f1"] == pytest.approx(1.0)
        # eval_n counts the scored reference items.
        assert result.eval_n == 2
        # metric_ci is a dict (CI computed) or None; with only 2 items it may be
        # present with small half-widths. When present it carries the metric keys.
        if result.metric_ci is not None:
            assert set(result.metric_ci).issubset({"normalized_exact_match", "token_f1"})

    def test_no_references_leaves_task_metrics_empty(self, tmp_path):
        """Backward-compatible: omitting references => task_metrics stays {}."""
        model = _make_model(loss_value=0.5)
        tokenizer = _make_tokenizer()
        history = MagicMock()
        history.get_run.return_value = _run_entry()
        history.record_run_completed.return_value = {}

        with patch.dict(sys.modules, {"torch": _FakeTorch()}), \
             patch.object(ev, "_load_model_and_tokenizer", return_value=(model, tokenizer)), \
             patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
            result = evaluate_run(
                "abc123def456",
                output_dir=str(tmp_path),
                heldout_texts=["a held out text"],
                n=2,
            )

        assert result.task_metrics == {}
        assert result.eval_n == 0
        assert result.metric_ci is None
        # The existing held-out loss path is unaffected.
        assert result.held_out_loss == pytest.approx(0.5)

    def test_references_default_metric_when_metrics_omitted(self, tmp_path):
        """references given but metrics omitted => defaults to the SQuAD pair
        (normalized_exact_match + token_f1)."""
        model = _make_model(loss_value=0.5)
        tokenizer = _make_tokenizer()
        history = MagicMock()
        history.get_run.return_value = _run_entry()
        history.record_run_completed.return_value = {}

        with patch.dict(sys.modules, {"torch": _FakeTorch()}), \
             patch.object(ev, "_load_model_and_tokenizer", return_value=(model, tokenizer)), \
             patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
            result = evaluate_run(
                "abc123def456",
                output_dir=str(tmp_path),
                heldout_texts=["a held out text"],
                references=[{"prompt": "q", "reference": "Paris"}],
            )

        assert "normalized_exact_match" in result.task_metrics
        assert "token_f1" in result.task_metrics

    def test_bad_reference_shape_raises_user_input_error(self, tmp_path):
        """A reference item missing both 'reference' and 'references' is a clear
        INPUT_ error, not a silent skip."""
        model = _make_model(loss_value=0.5)
        tokenizer = _make_tokenizer()
        history = MagicMock()
        history.get_run.return_value = _run_entry()

        with patch.dict(sys.modules, {"torch": _FakeTorch()}), \
             patch.object(ev, "_load_model_and_tokenizer", return_value=(model, tokenizer)), \
             patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
            with pytest.raises(UserInputError) as exc:
                evaluate_run(
                    "abc123def456",
                    output_dir=str(tmp_path),
                    heldout_texts=["a held out text"],
                    references=[{"prompt": "q"}],  # no reference/references
                )
        assert exc.value.code == "INPUT_VALIDATION_FAILED"


# =============================================================================
# eval_gate — conjunction (loss floor AND task-metric noise band)
# =============================================================================

def _result(loss, task_metrics=None, eval_n=0, metric_ci=None):
    return EvalResult(
        run_id="r",
        model_name="m",
        held_out_loss=loss,
        perplexity=None,
        generations=[],
        n_prompts=0,
        task_metrics=task_metrics or {},
        eval_n=eval_n,
        metric_ci=metric_ci,
    )


class TestEvalGateConjunction:
    def test_loss_only_unchanged_when_no_gated_metrics(self):
        """Backward-compatible: with no gated_metrics, behavior matches the old
        loss-only gate (a regression beyond tol rejects)."""
        before = _result(1.0)
        after = _result(1.5)
        decision = eval_gate(before, after, max_regression=0.0)
        assert isinstance(decision, EvalGateDecision)
        assert decision.accept is False

    def test_loss_floor_blocks_even_when_metric_improves(self):
        """Loss is a FLOOR: a loss regression beyond tol rejects even if the
        gated task-metric improved."""
        before = _result(1.0, task_metrics={"token_f1": 0.5})
        after = _result(2.0, task_metrics={"token_f1": 0.9})
        decision = eval_gate(
            before, after, max_regression=0.0, gated_metrics=["token_f1"]
        )
        assert decision.accept is False
        assert "loss" in decision.reason.lower()

    def test_metric_regression_beyond_band_rejects_even_when_loss_improves(self):
        """Loss improving does NOT save a real task-metric regression — the
        conjunction rejects. Message is contrastive (loss improved BUT metric)."""
        before = _result(
            1.0, task_metrics={"normalized_exact_match": 0.80}, eval_n=200,
            metric_ci={"normalized_exact_match": 0.02},
        )
        after = _result(
            0.5, task_metrics={"normalized_exact_match": 0.60}, eval_n=200,
            metric_ci={"normalized_exact_match": 0.02},
        )
        decision = eval_gate(
            before, after, max_regression=0.0,
            gated_metrics=["normalized_exact_match"],
        )
        assert decision.accept is False
        low = decision.reason.lower()
        assert "loss" in low and "exact_match" in low

    def test_metric_drop_within_noise_band_does_not_gate(self):
        """A metric drop smaller than the CI half-width is NOISE — not gating.
        The message frames it contrastively (dropped X +/- Y -> within noise)."""
        before = _result(
            1.0, task_metrics={"normalized_exact_match": 0.80}, eval_n=200,
            metric_ci={"normalized_exact_match": 0.08},
        )
        # drop of 0.04 < CI half-width 0.08 -> within noise.
        after = _result(
            0.9, task_metrics={"normalized_exact_match": 0.76}, eval_n=200,
            metric_ci={"normalized_exact_match": 0.08},
        )
        decision = eval_gate(
            before, after, max_regression=0.0,
            gated_metrics=["normalized_exact_match"],
        )
        assert decision.accept is True
        assert "noise" in decision.reason.lower()

    def test_metric_improvement_accepts(self):
        before = _result(1.0, task_metrics={"token_f1": 0.5}, eval_n=200)
        after = _result(0.9, task_metrics={"token_f1": 0.7}, eval_n=200)
        decision = eval_gate(
            before, after, max_regression=0.0, gated_metrics=["token_f1"]
        )
        assert decision.accept is True

    def test_default_tol_used_when_no_ci(self):
        """When metric_ci is absent, a sensible default noise tol is used so a
        tiny drop still passes but a large one gates."""
        before = _result(1.0, task_metrics={"token_f1": 0.80}, eval_n=200)
        # default metric_tol default is small; a 0.20 drop must gate.
        after = _result(0.9, task_metrics={"token_f1": 0.60}, eval_n=200)
        decision = eval_gate(
            before, after, max_regression=0.0, gated_metrics=["token_f1"]
        )
        assert decision.accept is False

    def test_explicit_metric_tol_overrides(self):
        """An explicit metric_tol widens the noise band for all gated metrics."""
        before = _result(1.0, task_metrics={"token_f1": 0.80}, eval_n=200)
        after = _result(0.9, task_metrics={"token_f1": 0.70}, eval_n=200)
        decision = eval_gate(
            before, after, max_regression=0.0, gated_metrics=["token_f1"],
            metric_tol=0.2,  # band wider than the 0.10 drop
        )
        assert decision.accept is True

    def test_missing_gated_metric_rejects_fail_safe(self):
        """A gated metric absent from one side cannot be proven non-regressed ->
        fail-safe reject."""
        before = _result(1.0, task_metrics={"token_f1": 0.8}, eval_n=200)
        after = _result(0.9, task_metrics={}, eval_n=200)  # missing token_f1
        decision = eval_gate(
            before, after, max_regression=0.0, gated_metrics=["token_f1"]
        )
        assert decision.accept is False

    def test_underpowered_warns_but_still_decides(self, caplog):
        """eval_n < 100 emits an underpowered WARNING but the gate still
        returns a decision (it does not block on power alone)."""
        import logging

        before = _result(1.0, task_metrics={"token_f1": 0.8}, eval_n=30)
        after = _result(0.9, task_metrics={"token_f1": 0.85}, eval_n=30)
        with caplog.at_level(logging.WARNING, logger="backpropagate.eval"):
            decision = eval_gate(
                before, after, max_regression=0.0, gated_metrics=["token_f1"]
            )
        assert decision.accept is True
        assert any(
            "underpower" in rec.message.lower() or "eval_n" in rec.message.lower()
            for rec in caplog.records
        )

    def test_missing_loss_still_fails_safe(self):
        """The fail-safe on a missing held-out loss is preserved."""
        decision = eval_gate(_result(None), _result(1.0))
        assert decision.accept is False
        assert decision.regression != decision.regression  # NaN
