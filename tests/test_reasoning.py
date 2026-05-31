"""
Tests for reasoning-trace SFT support (v1.5 T3.2).

Two halves:

1. **Think-preservation invariant** — ``<think>...</think>`` blocks are PLAIN
   TEXT and the converters pass assistant content verbatim, so a reasoning
   trace MUST survive ``convert_to_chatml`` intact for ShareGPT / OpenAI /
   Alpaca / ChatML. A think-only assistant turn must NOT be mis-flagged as an
   empty turn (the ``_CHATML_TURN_RE`` DOTALL safety). This locks the invariant
   the architect verified.
2. **Trace-length filter** — ``filter_by_trace_length`` keeps rows whose summed
   ``<think>`` token band is in range, dropping too-short / too-long /
   unbalanced / no-think rows with a per-reason ``TraceFilterStats`` tally;
   inverted bounds raise ``InvalidSettingError`` (``CONFIG_INVALID_SETTING``);
   an all-removed run WARNs loudly.

The filter-math tests pass ``token_counter=len`` for deterministic
char==token arithmetic (no tokenizer, no fuzz).
"""

import logging

import pytest

from backpropagate.datasets import (
    DatasetFormat,
    DatasetLoader,
    TraceFilterStats,
    _extract_think_spans,
    _warn_on_empty_turns,
    convert_to_chatml,
    dataset_has_leading_think,
    filter_by_trace_length,
    warn_on_doubled_think,
)
from backpropagate.exceptions import InvalidSettingError

# A reasoning trace: a multi-line <think> body followed by the final answer.
_TRACE = "<think>step1\nstep2</think>The answer is 42."


# =============================================================================
# THINK-PRESERVATION INVARIANT (the locked contract)
# =============================================================================


class TestThinkPreservation:
    """``<think>...</think>The answer...`` survives conversion for every format."""

    def _assert_survives(self, sample: dict) -> str:
        out = convert_to_chatml([sample])
        assert len(out) == 1
        text = out[0]["text"]
        # The FULL think-block + answer substring must be present verbatim.
        assert _TRACE in text, f"trace lost in conversion: {text!r}"
        return text

    def test_sharegpt_preserves_think(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "What is 6 times 7?"},
                {"from": "gpt", "value": _TRACE},
            ]
        }
        self._assert_survives(sample)

    def test_openai_preserves_think(self):
        sample = {
            "messages": [
                {"role": "user", "content": "What is 6 times 7?"},
                {"role": "assistant", "content": _TRACE},
            ]
        }
        self._assert_survives(sample)

    def test_alpaca_preserves_think(self):
        sample = {"instruction": "What is 6 times 7?", "output": _TRACE}
        self._assert_survives(sample)

    def test_chatml_preserves_think(self):
        sample = {
            "text": (
                "<|im_start|>user\nWhat is 6 times 7?<|im_end|>\n"
                f"<|im_start|>assistant\n{_TRACE}<|im_end|>"
            )
        }
        self._assert_survives(sample)

    def test_think_survives_with_explicit_format_hint(self):
        # Pinning the source format (not just auto-detect) preserves it too.
        sample = {"instruction": "q", "output": _TRACE}
        out = convert_to_chatml([sample], source_format=DatasetFormat.ALPACA)
        assert _TRACE in out[0]["text"]


class TestThinkOnlyTurnNotEmpty:
    """A think-only assistant body is real content, never an empty turn."""

    def test_think_only_turn_not_flagged_empty(self, caplog):
        # An assistant turn whose entire body is a <think> block (DOTALL spans
        # the newline) must NOT trigger the empty-turn WARN — the body is not
        # blank.
        chatml = (
            "<|im_start|>user\nsolve it<|im_end|>\n"
            "<|im_start|>assistant\n<think>reason\nmore reasoning</think><|im_end|>"
        )
        with caplog.at_level(logging.WARNING, logger="backpropagate.datasets"):
            _warn_on_empty_turns(chatml, 0)
        assert not any(
            "empty" in rec.getMessage().lower() for rec in caplog.records
        ), "think-only turn was mis-flagged as empty"

    def test_empty_assistant_still_flagged(self, caplog):
        # Control: a genuinely empty assistant body still warns (the DOTALL
        # change must not mask real blanks).
        chatml = (
            "<|im_start|>user\nq<|im_end|>\n"
            "<|im_start|>assistant\n<|im_end|>"
        )
        with caplog.at_level(logging.WARNING, logger="backpropagate.datasets"):
            _warn_on_empty_turns(chatml, 0)
        assert any(
            "empty" in rec.getMessage().lower() for rec in caplog.records
        )


# =============================================================================
# _extract_think_spans
# =============================================================================


class TestExtractThinkSpans:
    def test_extracts_multiple_spans(self):
        spans = _extract_think_spans("<think>a</think>mid<think>b\nc</think>z")
        assert spans == ["a", "b\nc"]

    def test_no_spans_returns_empty_list(self):
        assert _extract_think_spans("no reasoning here") == []

    def test_case_insensitive(self):
        assert _extract_think_spans("<THINK>x</THINK>") == ["x"]
        assert _extract_think_spans("<Think>y</Think>") == ["y"]

    def test_empty_span_body(self):
        assert _extract_think_spans("<think></think>") == [""]


# =============================================================================
# TRACE-LENGTH FILTER MATH (token_counter=len for determinism)
# =============================================================================


def _chatml_row(body: str) -> dict:
    """A minimal ChatML row whose assistant body is ``body``."""
    return {
        "text": (
            "<|im_start|>user\nq<|im_end|>\n"
            f"<|im_start|>assistant\n{body}<|im_end|>"
        )
    }


class TestTraceFilterMath:
    def test_empty_think_removed_too_short(self):
        rows = [_chatml_row("<think></think>The answer.")]
        out, stats = filter_by_trace_length(
            rows, min_trace_tokens=1, token_counter=len
        )
        assert out == []
        assert stats.removed_trace_too_short == 1
        assert stats.total_after == 0

    def test_long_trace_removed_too_long(self):
        # 5000-char trace, char==token via len, max=100 -> too long.
        rows = [_chatml_row("<think>" + ("x" * 5000) + "</think>ok")]
        out, stats = filter_by_trace_length(
            rows, min_trace_tokens=1, max_trace_tokens=100, token_counter=len
        )
        assert out == []
        assert stats.removed_trace_too_long == 1

    def test_unbalanced_think_removed(self):
        # An opener with no closer -> unbalanced, dropped before length checks.
        rows = [_chatml_row("<think>unclosed reasoning with no end")]
        out, stats = filter_by_trace_length(rows, token_counter=len)
        assert out == []
        assert stats.removed_unbalanced_think == 1

    def test_stray_close_also_unbalanced(self):
        rows = [_chatml_row("answer then a stray </think> tag")]
        out, stats = filter_by_trace_length(rows, token_counter=len)
        assert out == []
        assert stats.removed_unbalanced_think == 1

    def test_no_think_removed_when_required(self):
        rows = [_chatml_row("just a plain answer, no reasoning")]
        out, stats = filter_by_trace_length(
            rows, require_think=True, token_counter=len
        )
        assert out == []
        assert stats.removed_no_think == 1

    def test_no_think_kept_when_not_required(self):
        rows = [_chatml_row("just a plain answer, no reasoning")]
        out, stats = filter_by_trace_length(
            rows, require_think=False, token_counter=len
        )
        assert len(out) == 1
        assert stats.removed_no_think == 0
        assert stats.total_after == 1

    def test_good_trace_retained(self):
        # A 50-char trace, char==token, within [8, 8192] -> kept.
        rows = [_chatml_row("<think>" + ("a" * 50) + "</think>The answer.")]
        out, stats = filter_by_trace_length(
            rows, min_trace_tokens=8, max_trace_tokens=8192, token_counter=len
        )
        assert len(out) == 1
        assert stats.total_after == 1
        assert stats.total_removed == 0

    def test_full_mix_counts_and_retention(self):
        rows = [
            _chatml_row("<think>" + ("a" * 50) + "</think>good"),  # retained
            _chatml_row("<think></think>empty"),                  # too short
            _chatml_row("<think>" + ("y" * 5000) + "</think>x"),  # too long
            _chatml_row("<think>unclosed"),                       # unbalanced
            _chatml_row("plain no think"),                        # no think
        ]
        out, stats = filter_by_trace_length(
            rows,
            min_trace_tokens=8,
            max_trace_tokens=100,
            require_think=True,
            token_counter=len,
        )
        assert stats.total_before == 5
        assert stats.total_after == 1
        assert stats.removed_trace_too_short == 1
        assert stats.removed_trace_too_long == 1
        assert stats.removed_unbalanced_think == 1
        assert stats.removed_no_think == 1
        assert stats.total_removed == 4
        assert stats.retention_rate == pytest.approx(0.2)

    def test_multiple_spans_summed(self):
        # Two 30-char spans = 60 tokens via len. min=50 keeps it; min=70 drops.
        body = "<think>" + ("a" * 30) + "</think>mid<think>" + ("b" * 30) + "</think>end"
        kept, kstats = filter_by_trace_length(
            [_chatml_row(body)], min_trace_tokens=50, token_counter=len
        )
        assert kstats.total_after == 1
        dropped, dstats = filter_by_trace_length(
            [_chatml_row(body)], min_trace_tokens=70, token_counter=len
        )
        assert dstats.total_after == 0
        assert dstats.removed_trace_too_short == 1

    def test_default_token_counter_is_approx(self):
        # Without token_counter the ~4 chars/token estimate applies: a 40-char
        # trace ~= 10 tokens, comfortably above the default min of 8.
        rows = [_chatml_row("<think>" + ("a" * 40) + "</think>answer body")]
        out, stats = filter_by_trace_length(rows)
        assert stats.total_after == 1

    def test_approx_counter_undercounts_cjk_direction(self):
        """The default approx counter UNDER-counts CJK (Phase 8 direction fix).

        A 20-char CJK trace is ~20+ real tokens (CJK is ~1+ token/char), so a
        real tokenizer keeps it against min_trace_tokens=12. But the default
        ``len//4`` approximation sees only 20//4 == 5 tokens and WRONGLY drops
        it as trace_too_short. This is the concrete consequence of under-counting
        (the old docstrings claimed the opposite "over-count" direction).
        """
        cjk_trace = "我" * 20  # 20 CJK chars; real tokens >= ~20
        row = _chatml_row(f"<think>{cjk_trace}</think>答案")

        # Real tokenizer-like counter (~1 token/char) -> 20 tokens -> KEPT.
        _kept, real_stats = filter_by_trace_length(
            [row], min_trace_tokens=12, token_counter=len
        )
        assert real_stats.total_after == 1

        # Default approx counter (len//4 == 5 tokens) UNDER-counts -> dropped
        # as too-short even though the real trace clears the floor.
        _dropped, approx_stats = filter_by_trace_length(
            [row], min_trace_tokens=12
        )
        assert approx_stats.total_after == 0
        assert approx_stats.removed_trace_too_short == 1


class TestTraceFilterStatsShape:
    def test_retention_rate_zero_on_empty(self):
        stats = TraceFilterStats(total_before=0, total_after=0)
        assert stats.retention_rate == 0.0
        assert stats.total_removed == 0

    def test_summary_lists_reasons_and_cjk_caveat(self):
        _out, stats = filter_by_trace_length(
            [_chatml_row("<think></think>x")],
            min_trace_tokens=1,
            token_counter=len,
        )
        text = stats.summary()
        assert "Reasoning-Trace Filter Results" in text
        assert "Trace too short" in text
        assert "cjk" in text.lower()


# =============================================================================
# INVALID BOUNDS -> InvalidSettingError (CONFIG_INVALID_SETTING)
# =============================================================================


class TestTraceFilterBounds:
    def test_inverted_bounds_raise(self):
        with pytest.raises(InvalidSettingError) as exc:
            filter_by_trace_length(
                [], min_trace_tokens=100, max_trace_tokens=10
            )
        assert exc.value.code == "CONFIG_INVALID_SETTING"

    def test_negative_min_raises(self):
        with pytest.raises(InvalidSettingError) as exc:
            filter_by_trace_length([], min_trace_tokens=-1)
        assert exc.value.code == "CONFIG_INVALID_SETTING"

    def test_negative_max_raises(self):
        with pytest.raises(InvalidSettingError) as exc:
            filter_by_trace_length([], max_trace_tokens=-5)
        assert exc.value.code == "CONFIG_INVALID_SETTING"

    def test_equal_bounds_ok(self):
        # min == max is a valid (degenerate) band, not inverted.
        rows = [_chatml_row("<think>" + ("a" * 8) + "</think>x")]
        out, stats = filter_by_trace_length(
            rows, min_trace_tokens=8, max_trace_tokens=8, token_counter=len
        )
        assert stats.total_after == 1


# =============================================================================
# ALL-REMOVED -> LOUD WARN
# =============================================================================


class TestTraceFilterWarnings:
    def test_all_removed_emits_warning(self, caplog):
        rows = [
            _chatml_row("<think></think>a"),
            _chatml_row("<think></think>b"),
        ]
        with caplog.at_level(logging.WARNING, logger="backpropagate.datasets"):
            out, stats = filter_by_trace_length(
                rows, min_trace_tokens=8, token_counter=len
            )
        assert stats.total_after == 0
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("removed ALL" in m for m in messages)
        # The WARN must name the actionable knob.
        assert any("min_trace_tokens" in m for m in messages)

    def test_no_warning_on_healthy_retention(self, caplog):
        rows = [_chatml_row("<think>" + ("a" * 50) + "</think>ok")]
        with caplog.at_level(logging.WARNING, logger="backpropagate.datasets"):
            filter_by_trace_length(rows, min_trace_tokens=8, token_counter=len)
        assert not any(
            "removed ALL" in rec.getMessage() for rec in caplog.records
        )


# =============================================================================
# DatasetLoader.filter_by_trace_length
# =============================================================================


class TestLoaderTraceFilter:
    def test_loader_filters_and_returns_stats(self):
        loader = DatasetLoader(
            [
                {"instruction": "q1", "output": "<think>" + ("a" * 50) + "</think>good"},
                {"instruction": "q2", "output": "plain answer no reasoning"},
            ],
            validate=False,
        )
        new_loader, stats = loader.filter_by_trace_length(
            min_trace_tokens=8, require_think=True, token_counter=len
        )
        assert isinstance(stats, TraceFilterStats)
        assert isinstance(new_loader, DatasetLoader)
        assert stats.total_before == 2
        assert stats.total_after == 1
        assert stats.removed_no_think == 1

    def test_loader_inverted_bounds_raise(self):
        loader = DatasetLoader(
            [{"instruction": "q", "output": "<think>x</think>a"}], validate=False
        )
        with pytest.raises(InvalidSettingError):
            loader.filter_by_trace_length(
                min_trace_tokens=100, max_trace_tokens=1
            )


# =============================================================================
# DOUBLED-<think> ADVISORY HELPERS (used by WIRE's trainer)
# =============================================================================


class TestDoubledThinkAdvisory:
    def test_predicate_true_when_data_leads_with_think(self):
        rows = [_chatml_row("<think>reason</think>answer")]
        assert dataset_has_leading_think(rows) is True

    def test_predicate_false_when_no_leading_think(self):
        rows = [_chatml_row("answer then <think>late</think>")]
        # A <think> that does NOT open the assistant body is not a leading tag.
        assert dataset_has_leading_think(rows) is False

    def test_predicate_false_on_plain_data(self):
        assert dataset_has_leading_think([_chatml_row("just an answer")]) is False

    def test_predicate_case_insensitive(self):
        rows = [_chatml_row("<THINK>reason</THINK>answer")]
        assert dataset_has_leading_think(rows) is True

    def test_warn_when_template_injects_and_data_leads(self, caplog):
        rows = [_chatml_row("<think>reason</think>answer")]
        with caplog.at_level(logging.WARNING, logger="backpropagate.datasets"):
            warn_on_doubled_think(rows, template_injects_think=True)
        assert any(
            "doubled" in rec.getMessage().lower() for rec in caplog.records
        )

    def test_no_warn_when_template_does_not_inject(self, caplog):
        rows = [_chatml_row("<think>reason</think>answer")]
        with caplog.at_level(logging.WARNING, logger="backpropagate.datasets"):
            warn_on_doubled_think(rows, template_injects_think=False)
        assert not any(
            "doubled" in rec.getMessage().lower() for rec in caplog.records
        )

    def test_no_warn_when_data_has_no_leading_think(self, caplog):
        rows = [_chatml_row("a plain answer with no reasoning")]
        with caplog.at_level(logging.WARNING, logger="backpropagate.datasets"):
            warn_on_doubled_think(rows, template_injects_think=True)
        assert not any(
            "doubled" in rec.getMessage().lower() for rec in caplog.records
        )
        # never mutate
        assert rows == [_chatml_row("a plain answer with no reasoning")]
