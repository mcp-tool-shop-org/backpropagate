"""Coverage tests for ``backpropagate.datasets`` pure logic.

Format detection, converters, validators, quality/trace filters, dedup,
stats, curriculum and the HuggingFace transient-retry wrapper. Everything
runs against real in-memory samples or real files written to ``tmp_path``.

What is mocked (and nothing else):

* ``tenacity.nap.time.sleep`` so the retry back-off schedule does not block.
* ``sys.modules`` entries for optional libraries (``requests``,
  ``huggingface_hub.utils``, ``datasketch``) to simulate absent extras.
* A tiny exact-Jaccard stand-in for ``datasketch`` (not in any extra, so
  CI never has the real one) used only where MinHash dedup is exercised.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import pytest

from backpropagate import datasets as ds
from backpropagate.datasets import (
    DatasetFormat,
    FilterStats,
    FormatConverter,
    TraceFilterStats,
    ValidationError,
    ValidationResult,
    convert_to_chatml,
    deduplicate_exact,
    deduplicate_minhash,
    detect_format,
    filter_by_quality,
    validate_dataset,
    validate_sample,
)
from backpropagate.exceptions import InvalidSettingError

LOGGER = "backpropagate.datasets"


def _chatml(user: str, assistant: str) -> str:
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


def _error_types(errors: list[ValidationError]) -> list[str]:
    return [e.error_type for e in errors]


# =============================================================================
# HF transient retry wrapper
# =============================================================================


class TestHfTransientExceptions:
    def test_includes_builtin_requests_and_hub_classes(self):
        import requests
        from huggingface_hub.utils import HfHubHTTPError

        excs = ds._hf_transient_exceptions()
        assert ConnectionError in excs
        assert TimeoutError in excs
        assert requests.exceptions.ConnectionError in excs
        assert requests.exceptions.Timeout in excs
        assert requests.exceptions.HTTPError in excs
        assert HfHubHTTPError in excs

    def test_degrades_when_requests_and_hub_are_absent(self, monkeypatch):
        """Mocks: ``sys.modules`` entries so the optional imports fail."""
        monkeypatch.setitem(sys.modules, "requests", None)
        monkeypatch.setitem(sys.modules, "huggingface_hub.utils", None)
        excs = ds._hf_transient_exceptions()
        assert excs == (ConnectionError, TimeoutError)


class TestIsTransientHfException:
    def test_non_network_exception_is_not_transient(self):
        assert ds._is_transient_hf_exception(ValueError("bad")) is False

    def test_connection_error_without_response_is_transient(self):
        assert ds._is_transient_hf_exception(ConnectionError("reset")) is True

    @pytest.mark.parametrize(
        ("status", "expected"),
        [(429, True), (500, True), (503, True), (401, False), (403, False), (404, False)],
    )
    def test_http_status_decides(self, status, expected):
        import requests

        exc = requests.exceptions.HTTPError(
            "boom", response=types.SimpleNamespace(status_code=status)
        )
        assert ds._is_transient_hf_exception(exc) is expected


class TestRetryHfCall:
    def test_success_passes_args_and_kwargs_through(self, monkeypatch):
        monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
        seen = {}

        def fn(a, b, *, c):
            seen.update(a=a, b=b, c=c)
            return "ok"

        assert ds._retry_hf_call(fn, 1, 2, c=3, _label="t") == "ok"
        assert seen == {"a": 1, "b": 2, "c": 3}

    def test_transient_failure_retries_three_times_then_logs_and_raises(
        self, monkeypatch, caplog
    ):
        """Mocks: tenacity's sleep (records the back-off schedule)."""
        import requests

        slept: list[float] = []
        monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: slept.append(s))
        calls = []

        def always_down():
            calls.append(1)
            raise requests.exceptions.ConnectionError("down")

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with pytest.raises(requests.exceptions.ConnectionError):
                ds._retry_hf_call(always_down, _label="probe:label")

        assert len(calls) == ds._HF_RETRY_ATTEMPTS == 3
        # two sleeps between three attempts, never below the 5s floor
        assert len(slept) == 2
        assert all(s >= ds._HF_RETRY_BASE_SECONDS for s in slept)
        assert any(
            "HF transient retry exhausted" in r.message and "probe:label" in r.message
            for r in caplog.records
        )

    def test_non_transient_error_is_not_retried(self, monkeypatch):
        slept: list[float] = []
        monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: slept.append(s))
        calls = []

        def bad():
            calls.append(1)
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            ds._retry_hf_call(bad)
        assert calls == [1]
        assert slept == []

    def test_http_404_is_not_retried(self, monkeypatch):
        import requests

        monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
        calls = []

        def missing():
            calls.append(1)
            raise requests.exceptions.HTTPError(
                "nf", response=types.SimpleNamespace(status_code=404)
            )

        with pytest.raises(requests.exceptions.HTTPError):
            ds._retry_hf_call(missing)
        assert calls == [1]


# =============================================================================
# Summaries / dataclass helpers
# =============================================================================


class TestSummaries:
    def test_validation_result_summary_lists_first_errors_and_warnings(self):
        errors = [
            ValidationError(i, "f", "missing_field", f"msg{i}") for i in range(7)
        ]
        warnings = [ValidationError(9, "g", "empty_content", "blank")]
        result = ValidationResult(
            is_valid=False,
            total_rows=10,
            valid_rows=2,
            errors=errors,
            warnings=warnings,
            format_detected=DatasetFormat.ALPACA,
        )
        text = result.summary()
        assert "Format: alpaca" in text
        assert "Valid rows: 2 (20.0%)" in text
        assert "First 5 errors:" in text
        assert "Row 4: [missing_field] f - msg4" in text
        assert "msg5" not in text  # only the first five are listed
        assert "First 5 warnings:" in text
        assert "Row 9: [empty_content] g - blank" in text
        assert result.error_count == 7
        assert result.warning_count == 1
        assert result.error_rate == pytest.approx(0.8)

    def test_validation_result_error_rate_of_empty_is_zero(self):
        assert ValidationResult(True, 0, 0).error_rate == 0.0

    def test_filter_stats_summary_names_every_reason(self):
        stats = FilterStats(
            total_before=100,
            total_after=40,
            removed_too_short=10,
            removed_too_long=11,
            removed_few_turns=12,
            removed_many_turns=13,
            removed_empty=7,
            removed_no_assistant=5,
            removed_custom=2,
        )
        text = stats.summary()
        assert "After:  40 (40.0% retained)" in text
        assert "Removed: 60" in text
        for needle in (
            "Too short: 10",
            "Too long: 11",
            "Too few turns: 12",
            "Too many turns: 13",
            "Empty content: 7",
            "No assistant: 5",
            "Custom filter: 2",
        ):
            assert needle in text
        assert stats.total_removed == 60
        assert FilterStats(0, 0).retention_rate == 0.0

    def test_summaries_omit_zero_count_reasons(self):
        text = FilterStats(10, 8, removed_too_long=2).summary()
        assert "Too long: 2" in text
        for absent in ("Too short", "Too few turns", "Too many turns", "Empty content", "No assistant", "Custom filter"):
            assert absent not in text
        only_short = FilterStats(10, 9, removed_too_short=1).summary()
        assert "Too short: 1" in only_short and "Too long" not in only_short
        trace = TraceFilterStats(5, 4, removed_trace_too_long=1).summary()
        assert "Trace too long: 1" in trace
        assert "Trace too short" not in trace and "No <think>" not in trace and "Unbalanced" not in trace
        trace2 = TraceFilterStats(5, 4, removed_trace_too_short=1).summary()
        assert "Trace too short: 1" in trace2 and "Trace too long" not in trace2

    def test_perplexity_stats_summary_thresholds_are_optional(self):
        base = {
            "total_samples": 4, "samples_scored": 4, "samples_failed": 0,
            "mean_perplexity": 10.0, "median_perplexity": 9.0, "std_perplexity": 1.0,
            "min_perplexity": 8.0, "max_perplexity": 12.0,
            "filtered_count": 1, "retained_count": 3,
        }
        bare = ds.PerplexityStats(**base).summary()
        assert "Low threshold" not in bare and "High threshold" not in bare
        assert "Retained: 3 (75.0%)" in bare
        high_only = ds.PerplexityStats(**base, threshold_high=11.5).summary()
        assert "High threshold: 11.50" in high_only and "Low threshold" not in high_only
        low_only = ds.PerplexityStats(**base, threshold_low=8.5).summary()
        assert "Low threshold: 8.50" in low_only and "High threshold" not in low_only
        assert ds.PerplexityStats(**{**base, "total_samples": 0, "retained_count": 0}).retention_rate == 0.0

    def test_trace_filter_stats_summary_names_every_reason(self):
        stats = TraceFilterStats(
            total_before=10,
            total_after=1,
            removed_trace_too_short=2,
            removed_trace_too_long=3,
            removed_no_think=3,
            removed_unbalanced_think=1,
        )
        text = stats.summary()
        assert "Trace too short: 2" in text
        assert "Trace too long: 3" in text
        assert "No <think> span: 3" in text
        assert "Unbalanced <think> tags: 1" in text
        assert "CJK" in text  # the documented token-estimate caveat
        assert stats.total_removed == 9
        assert TraceFilterStats(0, 0).retention_rate == 0.0


# =============================================================================
# Format detection
# =============================================================================


class TestDetectFormatBranches:
    def test_empty_list_and_non_container_are_unknown(self):
        assert detect_format([]) == DatasetFormat.UNKNOWN
        assert detect_format(42) == DatasetFormat.UNKNOWN  # type: ignore[arg-type]

    def test_chatml_string_vs_raw_string(self):
        assert detect_format(_chatml("hi", "yo")) == DatasetFormat.CHATML
        assert detect_format("just some prose") == DatasetFormat.RAW_TEXT

    @pytest.mark.parametrize(
        "sample",
        [
            {"conversations": []},
            {"conversations": "not a list"},
            {"conversations": [{"from": "human"}]},  # no "value"
            {"conversations": ["bare string"]},
        ],
    )
    def test_malformed_sharegpt_falls_through_to_unknown(self, sample):
        assert detect_format(sample) == DatasetFormat.UNKNOWN

    @pytest.mark.parametrize(
        "sample",
        [
            {"messages": []},
            {"messages": "x"},
            {"messages": [{"role": "user"}]},  # no "content"
            {"messages": ["bare"]},
        ],
    )
    def test_malformed_openai_falls_through_to_unknown(self, sample):
        assert detect_format(sample) == DatasetFormat.UNKNOWN

    def test_text_without_chatml_markers_is_unknown(self):
        assert detect_format({"text": "plain"}) == DatasetFormat.UNKNOWN
        assert detect_format({"text": 5}) == DatasetFormat.UNKNOWN

    def test_text_with_chatml_markers_is_chatml(self):
        assert detect_format({"text": _chatml("a", "b")}) == DatasetFormat.CHATML

    def test_float_label_is_not_kto(self):
        sample = {"completion": "x", "label": 1.0}
        assert detect_format(sample) == DatasetFormat.UNKNOWN

    def test_int_zero_one_labels_are_kto(self):
        assert detect_format({"completion": "x", "label": 0}) == DatasetFormat.KTO
        assert detect_format({"completion": "x", "label": 1}) == DatasetFormat.KTO
        assert detect_format({"completion": "x", "label": 2}) == DatasetFormat.UNKNOWN

    def test_preference_wins_over_sibling_messages(self):
        sample = {"messages": [{"role": "user", "content": "q"}], "chosen": "a", "rejected": "b"}
        assert detect_format(sample) == DatasetFormat.PREFERENCE


class TestDetectFormatFromFile:
    def _write(self, tmp_path: Path, name: str, text: str) -> Path:
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        return p

    def test_jsonl_majority_vote_and_sample_cap(self, tmp_path):
        rows = [
            {"instruction": "a", "output": "b"},
            {"instruction": "c", "output": "d"},
            {"messages": [{"role": "user", "content": "x"}]},
        ]
        text = "\n\n".join(json.dumps(r) for r in rows)  # blank lines tolerated
        p = self._write(tmp_path, "d.jsonl", text)
        assert ds._detect_format_from_file(p) == DatasetFormat.ALPACA

    def test_jsonl_only_samples_first_n_lines(self, tmp_path):
        alpaca = {"instruction": "a", "output": "b"}
        openai = {"messages": [{"role": "user", "content": "x"}]}
        # first two lines alpaca, then many openai rows past the sample window
        lines = [json.dumps(alpaca)] * 2 + [json.dumps(openai)] * 10
        p = self._write(tmp_path, "d.jsonl", "\n".join(lines))
        # sample_size=2 sees only the alpaca rows
        assert ds._detect_format_from_file(p, sample_size=2) == DatasetFormat.ALPACA
        # sample_size=5 sees 2 alpaca + 3 openai -> openai wins
        assert ds._detect_format_from_file(p, sample_size=5) == DatasetFormat.OPENAI

    def test_json_list_and_json_dict(self, tmp_path):
        row = {"conversations": [{"from": "human", "value": "q"}]}
        p_list = self._write(tmp_path, "l.json", json.dumps([row, row]))
        p_dict = self._write(tmp_path, "o.json", json.dumps(row))
        assert ds._detect_format_from_file(p_list) == DatasetFormat.SHAREGPT
        assert ds._detect_format_from_file(p_dict) == DatasetFormat.SHAREGPT

    def test_text_files_detect_chatml_or_raw(self, tmp_path):
        p1 = self._write(tmp_path, "a.txt", _chatml("q", "a"))
        p2 = self._write(tmp_path, "b.md", "# heading\n\nprose")
        assert ds._detect_format_from_file(p1) == DatasetFormat.CHATML
        assert ds._detect_format_from_file(p2) == DatasetFormat.RAW_TEXT

    def test_unknown_extension_tries_jsonl(self, tmp_path):
        p = self._write(tmp_path, "d.dat", json.dumps({"instruction": "a", "output": "b"}))
        assert ds._detect_format_from_file(p) == DatasetFormat.ALPACA

    def test_unknown_extension_non_json_is_raw_text(self, tmp_path):
        p = self._write(tmp_path, "d.dat", "not json at all\n")
        assert ds._detect_format_from_file(p) == DatasetFormat.RAW_TEXT

    def test_empty_file_is_unknown(self, tmp_path):
        p = self._write(tmp_path, "e.jsonl", "")
        assert ds._detect_format_from_file(p) == DatasetFormat.UNKNOWN

    def test_corrupt_file_logs_and_returns_unknown(self, tmp_path, caplog):
        p = self._write(tmp_path, "bad.jsonl", "{not json}\n")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert ds._detect_format_from_file(p) == DatasetFormat.UNKNOWN
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "Error detecting format" in msg and "bad.jsonl" in msg

    def test_missing_file_logs_and_returns_unknown(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert ds._detect_format_from_file(tmp_path / "nope.jsonl") == DatasetFormat.UNKNOWN
        assert any("falling back to UNKNOWN" in r.getMessage() for r in caplog.records)


# =============================================================================
# Converters
# =============================================================================


class TestRenderFunctionCall:
    def test_name_and_arguments(self):
        out = ds._render_function_call({"name": "f", "arguments": '{"x":1}'})
        assert out == '[Function call: f({"x":1})]'

    def test_name_only(self):
        assert ds._render_function_call({"name": "f"}) == "[Function call: f]"

    def test_nameless_dict_and_non_dict_fall_back_to_repr(self):
        assert ds._render_function_call({"arguments": "a"}) == "[Function call: {'arguments': 'a'}]"
        assert ds._render_function_call("raw") == "[Function call: raw]"


class TestConverterEdges:
    def test_openai_function_call_keeps_reply_text(self):
        sample = {
            "messages": [
                {"role": "assistant", "content": "Checking.", "function_call": {"name": "get", "arguments": "{}"}},
                {"role": "assistant", "content": None, "function_call": {"name": "only"}},
            ]
        }
        out = FormatConverter.openai_to_chatml(sample)
        assert "Checking.\n[Function call: get({})]" in out  # reply preserved
        assert "<|im_start|>assistant\n[Function call: only]<|im_end|>" in out

    def test_preference_message_lists_normalise_roles_and_bare_strings(self):
        sample = {
            "prompt": [
                {"from": "human", "value": "hi"},
                "bare prompt string",
            ],
            "chosen": [
                {"role": "gpt", "content": "calling", "function_call": {"name": "tool", "arguments": "1"}},
                {"role": "assistant", "content": None, "function_call": {"name": "t2"}},
            ],
            "rejected": "dropped",
        }
        out = FormatConverter.preference_to_chatml(sample)
        assert out.startswith("<|im_start|>user\nhi<|im_end|>")
        # bare string in a prompt list takes the default role (user)
        assert "<|im_start|>user\nbare prompt string<|im_end|>" in out
        assert "calling\n[Function call: tool(1)]" in out
        assert "[Function call: t2]" in out
        assert "dropped" not in out  # rejected never reaches SFT text

    def test_preference_blank_prompt_string_omitted_and_missing_role_defaults(self):
        sample = {"prompt": "   ", "chosen": [{"content": "no role here"}], "rejected": "x"}
        out = FormatConverter.preference_to_chatml(sample)
        assert "<|im_start|>user" not in out
        assert out == "<|im_start|>assistant\nno role here<|im_end|>"

    def test_preference_string_chosen(self):
        out = FormatConverter.preference_to_chatml({"chosen": "yes", "rejected": "no"})
        assert out == "<|im_start|>assistant\nyes<|im_end|>"

    @pytest.mark.parametrize(
        ("fmt", "label"),
        [
            (DatasetFormat.SHAREGPT, "ShareGPT"),
            (DatasetFormat.ALPACA, "Alpaca"),
            (DatasetFormat.OPENAI, "OpenAI"),
            (DatasetFormat.PREFERENCE, "Preference"),
        ],
    )
    def test_structured_formats_reject_bare_strings(self, fmt, label):
        with pytest.raises(ValueError, match=f"{label} format requires dict"):
            FormatConverter.to_chatml("just text", fmt)

    def test_chatml_and_raw_text_dispatch(self):
        assert FormatConverter.to_chatml({"text": "t"}, DatasetFormat.CHATML) == "t"
        assert FormatConverter.to_chatml("s", DatasetFormat.CHATML) == "s"
        assert (
            FormatConverter.to_chatml({"text": "body"}, DatasetFormat.RAW_TEXT)
            == "<|im_start|>user\nbody<|im_end|>"
        )

    def test_unknown_and_kto_formats_cannot_convert(self):
        with pytest.raises(ValueError, match="Cannot convert format"):
            FormatConverter.to_chatml({"a": 1}, DatasetFormat.UNKNOWN)
        with pytest.raises(ValueError, match="Cannot convert format"):
            FormatConverter.to_chatml({"completion": "c", "label": True}, DatasetFormat.KTO)


class TestConvertToChatmlLoss:
    def test_unconvertible_rows_are_dropped_with_row_index_and_summary(self, caplog):
        samples = [
            {"unrecognised": "shape"},
            {"instruction": "do", "output": "done"},
            {"another": 1},
        ]
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            out = convert_to_chatml(samples)
        assert len(out) == 1
        assert "<|im_start|>assistant\ndone<|im_end|>" in out[0]["text"]
        messages = [r.getMessage() for r in caplog.records]
        assert any("Failed to convert sample 0" in m for m in messages)
        assert any("Failed to convert sample 2" in m for m in messages)
        assert any("dropped 2/3" in m for m in messages)

    def test_empty_turns_warn_with_row_index(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            out = convert_to_chatml([{"instruction": "", "output": "x"}], DatasetFormat.ALPACA)
        assert len(out) == 1
        assert any("sample 0 produced an empty user turn" in r.getMessage() for r in caplog.records)

    def test_empty_input_returns_empty(self):
        assert convert_to_chatml([]) == []

    def test_unknown_hint_triggers_per_sample_detection(self):
        rows = [
            {"instruction": "a", "output": "b"},
            {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "r"}]},
        ]
        out = convert_to_chatml(rows, DatasetFormat.UNKNOWN)
        assert "<|im_start|>user\nr" not in out[0]["text"]
        assert out[1]["text"] == (
            "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\nr<|im_end|>"
        )


# =============================================================================
# Validators
# =============================================================================


class TestChatmlValidator:
    def test_unbalanced_empty_and_bad_role(self):
        errs = ds._validate_chatml("<|im_start|>wizard\nhi", 3)
        assert set(_error_types(errs)) == {"unbalanced_tags", "invalid_role"}
        assert all(e.row_index == 3 for e in errs)
        empty = ds._validate_chatml("   ", 0)
        assert "empty_content" in _error_types(empty)

    def test_valid_chatml_has_no_errors(self):
        assert ds._validate_chatml(_chatml("q", "a"), 0) == []


class TestShareGptValidator:
    def test_missing_conversations(self):
        errs = ds._validate_sharegpt({}, 1)
        assert _error_types(errs) == ["missing_field"]
        assert errs[0].field == "conversations"

    def test_non_list_conversations(self):
        errs = ds._validate_sharegpt({"conversations": {"a": 1}}, 1)
        assert _error_types(errs) == ["invalid_type"]
        assert "got dict" in errs[0].message

    def test_empty_conversations(self):
        errs = ds._validate_sharegpt({"conversations": []}, 1)
        assert _error_types(errs) == ["empty_conversations"]

    def test_turn_level_errors_name_the_index(self):
        errs = ds._validate_sharegpt(
            {"conversations": ["bare", {"value": "v"}, {"from": "human"}]}, 5
        )
        fields = {e.field for e in errs}
        assert fields == {
            "conversations[0]",
            "conversations[1].from",
            "conversations[2].value",
        }
        assert errs[0].error_type == "invalid_type"


class TestAlpacaValidator:
    def test_missing_and_blank_fields(self):
        errs = ds._validate_alpaca({}, 0)
        types_ = _error_types(errs)
        assert types_.count("missing_field") == 2
        assert types_.count("empty_content") == 2

    def test_none_and_whitespace_are_blank(self):
        errs = ds._validate_alpaca({"instruction": None, "output": "  "}, 0)
        assert _error_types(errs) == ["empty_content", "empty_content"]

    def test_valid(self):
        assert ds._validate_alpaca({"instruction": "a", "output": "b"}, 0) == []


class TestIsBlank:
    def test_values(self):
        assert ds._is_blank("") and ds._is_blank("  \n")
        assert not ds._is_blank("x")
        assert ds._is_blank([]) and ds._is_blank({})
        assert not ds._is_blank([{"role": "user"}]) and not ds._is_blank({"k": 1})
        assert ds._is_blank(None) and ds._is_blank(float("nan")) and ds._is_blank(3)


class TestPreferenceValidator:
    def test_missing_pair_reports_missing_empty_and_implicit_prompt(self):
        errs = ds._validate_preference({}, 2)
        by_type = _error_types(errs)
        assert by_type.count("missing_field") == 2
        assert by_type.count("empty_content") == 2
        assert by_type.count("implicit_prompt") == 1

    def test_full_row_is_clean_and_message_lists_count_as_content(self):
        row = {
            "prompt": "q",
            "chosen": [{"role": "assistant", "content": "a"}],
            "rejected": "b",
        }
        assert ds._validate_preference(row, 0) == []

    def test_blank_chosen_flagged(self):
        errs = ds._validate_preference({"prompt": "q", "chosen": [], "rejected": "r"}, 0)
        assert _error_types(errs) == ["empty_content"]
        assert errs[0].field == "chosen"


class TestKtoValidator:
    def test_missing_everything(self):
        errs = ds._validate_kto({}, 0)
        assert _error_types(errs) == ["missing_field", "missing_field", "implicit_prompt"]

    def test_non_boolean_label_is_invalid_value(self):
        errs = ds._validate_kto({"prompt": "p", "completion": "c", "label": "positive"}, 0)
        assert _error_types(errs) == ["invalid_value"]
        assert errs[0].value == "positive"
        errs2 = ds._validate_kto({"prompt": "p", "completion": "c", "label": 2}, 0)
        assert _error_types(errs2) == ["invalid_value"]

    def test_blank_completion_is_empty_content(self):
        errs = ds._validate_kto({"prompt": "p", "completion": " ", "label": True}, 0)
        assert _error_types(errs) == ["empty_content"]

    def test_valid_rows_including_int_labels(self):
        assert ds._validate_kto({"prompt": "p", "completion": "c", "label": 0}, 0) == []
        assert ds._validate_kto({"prompt": "p", "completion": "c", "label": True}, 0) == []


class TestOpenAiValidator:
    def test_missing_messages(self):
        assert _error_types(ds._validate_openai({}, 0)) == ["missing_field"]

    def test_non_list_and_empty(self):
        assert _error_types(ds._validate_openai({"messages": "x"}, 0)) == ["invalid_type"]
        assert _error_types(ds._validate_openai({"messages": []}, 0)) == ["empty_messages"]

    def test_message_level_errors(self):
        errs = ds._validate_openai(
            {
                "messages": [
                    "bare",
                    {"content": "no role"},
                    {"role": "wizard", "content": "x"},
                    {"role": "user"},
                    {"role": "assistant", "function_call": {"name": "f"}},  # ok, no content
                ]
            },
            0,
        )
        pairs = [(e.field, e.error_type) for e in errs]
        assert ("messages[0]", "invalid_type") in pairs
        assert ("messages[1].role", "missing_field") in pairs
        assert ("messages[2].role", "invalid_role") in pairs
        assert ("messages[3].content", "missing_field") in pairs
        assert not any(f.startswith("messages[4]") for f, _ in pairs)


class TestValidateSample:
    @pytest.mark.parametrize(
        ("fmt", "label"),
        [
            (DatasetFormat.SHAREGPT, "ShareGPT"),
            (DatasetFormat.ALPACA, "Alpaca"),
            (DatasetFormat.PREFERENCE, "Preference"),
            (DatasetFormat.KTO, "KTO"),
            (DatasetFormat.OPENAI, "OpenAI"),
        ],
    )
    def test_structured_formats_reject_strings(self, fmt, label):
        errs = validate_sample("a string", 4, fmt)
        assert len(errs) == 1
        assert errs[0].error_type == "invalid_type"
        assert errs[0].row_index == 4
        assert f"{label} format requires dict" in errs[0].message

    def test_chatml_dict_and_string(self):
        assert validate_sample({"text": _chatml("q", "a")}, 0, DatasetFormat.CHATML) == []
        assert validate_sample(_chatml("q", "a"), 0, DatasetFormat.CHATML) == []
        assert _error_types(validate_sample({"text": "<|im_start|>user\nx"}, 0, DatasetFormat.CHATML))

    def test_raw_text(self):
        assert validate_sample("hello", 0, DatasetFormat.RAW_TEXT) == []
        errs = validate_sample("   ", 7, DatasetFormat.RAW_TEXT)
        assert _error_types(errs) == ["empty_content"] and errs[0].row_index == 7
        assert validate_sample({"text": "dict raw"}, 0, DatasetFormat.RAW_TEXT) == []

    def test_unknown_format(self):
        errs = validate_sample({"a": 1}, 0, DatasetFormat.UNKNOWN)
        assert _error_types(errs) == ["unknown_format"]


class TestValidateDataset:
    def test_empty_dataset(self):
        result = validate_dataset([])
        assert result.is_valid is False
        assert result.total_rows == 0
        assert _error_types(result.errors) == ["empty_dataset"]
        assert result.format_detected == DatasetFormat.UNKNOWN

    def test_warning_types_do_not_invalidate(self):
        rows = [
            {"prompt": "q", "chosen": "a", "rejected": "b"},
            {"chosen": "a", "rejected": "b"},  # implicit prompt -> warning only
        ]
        result = validate_dataset(rows)
        assert result.is_valid is True
        assert result.warning_count == 1
        assert result.warnings[0].error_type == "implicit_prompt"
        assert result.valid_rows == 1  # the warning row is not "clean"

    def test_mixed_formats_are_validated_per_sample(self):
        rows = [
            {"instruction": "a", "output": "b"},
            {"conversations": [{"from": "human", "value": "q"}]},
        ]
        result = validate_dataset(rows)
        assert result.is_valid is True
        assert result.format_detected == DatasetFormat.ALPACA  # first row's format

    def test_pinned_format_applies_to_every_row(self):
        rows = [{"instruction": "a", "output": "b"}, {"instruction": "c"}]
        result = validate_dataset(rows, DatasetFormat.ALPACA)
        assert result.is_valid is False
        assert [e.row_index for e in result.errors] == [1]
        assert result.format_detected == DatasetFormat.ALPACA

    def test_max_errors_stops_collection(self):
        rows = [{"instruction": f"q{i}"} for i in range(50)]  # each: 1 missing_field error
        result = validate_dataset(rows, DatasetFormat.ALPACA, max_errors=3)
        assert len(result.errors) == 3
        assert result.total_rows == 50
        assert result.is_valid is False


# =============================================================================
# Quality filter / trace filter / dedup
# =============================================================================


class TestFilterByQualityEdges:
    def test_near_total_wipeout_warns_with_percentage(self, caplog):
        good = _chatml("u" * 300, "a" * 300)
        samples = [{"text": good}] + [{"text": "tiny"} for _ in range(39)]
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            kept, stats = filter_by_quality(samples)
        assert len(kept) == 1
        assert stats.removed_too_short == 39
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "retained only 1/40" in msg and "2.5%" in msg

    def test_total_wipeout_warns_with_breakdown(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            kept, stats = filter_by_quality([{"text": "tiny"}, {"text": ""}])
        assert kept == []
        assert stats.removed_empty == 1 and stats.removed_too_short == 1
        assert any("removed ALL 2 samples" in r.getMessage() for r in caplog.records)

    def test_custom_filter_and_turn_bounds(self):
        long_ = "x" * 300
        one_turn = f"<|im_start|>assistant\n{long_}<|im_end|>"
        three = _chatml(long_, long_) + f"\n<|im_start|>user\n{long_}<|im_end|>"
        two = _chatml(long_, long_)
        kept, stats = filter_by_quality(
            [{"text": one_turn}, {"text": three}, {"text": two}],
            max_turns=2,
        )
        assert [s["text"] for s in kept] == [two]
        assert stats.removed_few_turns == 1
        assert stats.removed_many_turns == 1

        kept2, stats2 = filter_by_quality(
            [{"text": two, "keep": True}, {"text": two, "keep": False}],
            custom_filter=lambda s: s["keep"],
        )
        assert [s["keep"] for s in kept2] == [True]
        assert stats2.removed_custom == 1

    def test_too_long_and_no_assistant(self):
        long_user_only = "<|im_start|>user\n" + "z" * 400 + "<|im_end|>\n<|im_start|>user\nq<|im_end|>"
        kept, stats = filter_by_quality([{"text": long_user_only}], min_tokens=0, max_tokens=10)
        assert kept == [] and stats.removed_too_long == 1
        kept, stats = filter_by_quality([{"text": long_user_only}], min_tokens=0)
        assert kept == [] and stats.removed_no_assistant == 1

    def test_non_dict_samples_are_coerced_to_text(self):
        long_ = "x" * 300
        kept, _ = filter_by_quality([_chatml(long_, long_)])  # type: ignore[list-item]
        assert len(kept) == 1


class TestDedupHelpers:
    def test_get_text_content_handles_strings_dicts_and_missing_keys(self):
        assert ds._get_text_content("raw") == "raw"
        assert ds._get_text_content({"text": "t"}) == "t"
        assert ds._get_text_content({"other": "o"}, key="other") == "o"
        assert ds._get_text_content({}) == ""

    def test_exact_dedup_on_plain_strings_keeps_first(self):
        unique, removed = deduplicate_exact(["a", "b", "a", "c", "b"])
        assert unique == ["a", "b", "c"]
        assert removed == 2

    def test_exact_dedup_on_custom_key(self):
        rows = [{"k": "x", "id": 1}, {"k": "x", "id": 2}, {"k": "y", "id": 3}]
        unique, removed = deduplicate_exact(rows, key="k")
        assert [r["id"] for r in unique] == [1, 3]
        assert removed == 1

    def test_ngrams(self):
        assert ds._get_ngrams("") == []
        assert ds._get_ngrams("AB") == ["ab"]
        assert ds._get_ngrams("abcd") == ["abc", "bcd"]


# --- an exact-Jaccard stand-in for datasketch (see module docstring) ----------


class _FakeMinHash:
    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm
        self.items: set[bytes] = set()

    def update(self, data: bytes) -> None:
        self.items.add(data)

    def jaccard(self, other: _FakeMinHash) -> float:
        union = self.items | other.items
        return len(self.items & other.items) / len(union) if union else 1.0


class _FakeLSH:
    def __init__(self, threshold: float = 0.5, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.store: dict[str, _FakeMinHash] = {}

    def query(self, mh: _FakeMinHash) -> list[str]:
        return [k for k, v in self.store.items() if v.jaccard(mh) >= self.threshold]

    def insert(self, key: str, mh: _FakeMinHash) -> None:
        if key in self.store:
            raise ValueError("key exists")
        self.store[key] = mh


@pytest.fixture
def fake_datasketch(monkeypatch):
    mod = types.ModuleType("datasketch")
    mod.MinHash = _FakeMinHash  # type: ignore[attr-defined]
    mod.MinHashLSH = _FakeLSH  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasketch", mod)
    return mod


FOX = "the quick brown fox jumps over the lazy dog"


class TestDeduplicateMinhash:
    """Mocks: ``datasketch`` (exact-Jaccard stand-in); all dedup logic is real."""

    def test_near_duplicate_dropped_first_seen_kept(self, fake_datasketch):
        rows = [{"text": FOX + "."}, {"text": FOX}, {"text": "completely different words here"}]
        unique, removed = deduplicate_minhash(rows, threshold=0.9)
        assert removed == 1
        assert [r["text"] for r in unique] == [FOX + ".", "completely different words here"]

    def test_deterministic_across_calls(self, fake_datasketch):
        rows = [{"text": FOX}, {"text": FOX + "!"}, {"text": "another sentence entirely"}]
        first = deduplicate_minhash(rows, threshold=0.9)
        second = deduplicate_minhash(rows, threshold=0.9)
        assert first == second

    def test_low_threshold_collapses_more(self, fake_datasketch):
        rows = [{"text": "abcdefghij"}, {"text": "abcdefgxyz"}]
        kept_hi, _ = deduplicate_minhash(rows, threshold=0.9)
        kept_lo, removed_lo = deduplicate_minhash(rows, threshold=0.2)
        assert len(kept_hi) == 2
        assert len(kept_lo) == 1 and removed_lo == 1

    def test_empty_and_whitespace_rows_are_all_kept(self, fake_datasketch):
        rows = ["", "   ", "", FOX]
        unique, removed = deduplicate_minhash(rows)
        assert removed == 0
        assert unique == rows

    def test_plain_string_samples_supported(self, fake_datasketch):
        unique, removed = deduplicate_minhash([FOX, FOX, "zzz yyy xxx"], threshold=0.95)
        assert removed == 1
        assert unique == [FOX, "zzz yyy xxx"]

    def test_memory_advisories_for_large_inputs(self, fake_datasketch, caplog):
        samples = [""] * 100_000  # blank rows are kept; cheap to hash
        with caplog.at_level(logging.INFO, logger=LOGGER):
            unique, removed = deduplicate_minhash(samples, num_perm=1001)
        assert removed == 0 and len(unique) == 100_000
        messages = [(r.levelno, r.getMessage()) for r in caplog.records]
        assert any(lvl == logging.INFO and "estimated" in m and "100000 samples" in m for lvl, m in messages)
        assert any(lvl == logging.WARNING and "may" in m and "exhaust RAM" in m for lvl, m in messages)

    def test_lsh_insert_failure_keeps_the_sample_and_warns(self, fake_datasketch, monkeypatch, caplog):
        class Refusing(_FakeLSH):
            def insert(self, key, mh):
                raise ValueError("refused")

        monkeypatch.setattr(fake_datasketch, "MinHashLSH", Refusing)
        rows = ["alpha beta gamma delta", "wholly unrelated text goes here"]
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            unique, removed = deduplicate_minhash(rows)
        assert unique == rows and removed == 0
        assert sum("LSH insert raised" in r.getMessage() for r in caplog.records) == 2

    def test_missing_datasketch_raises_actionable_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "datasketch", None)
        with pytest.raises(ImportError, match="pip install datasketch"):
            deduplicate_minhash(["a", "b"])


# =============================================================================
# Stats / difficulty / curriculum
# =============================================================================


class TestStatsAndCurriculumEdges:
    def test_get_dataset_stats_empty(self):
        stats = ds.get_dataset_stats([])
        assert stats.total_samples == 0
        assert stats.format_detected == DatasetFormat.UNKNOWN

    def test_get_dataset_stats_counts_system_prompts(self):
        rows = [
            {"system": "be brief", "instruction": "a", "output": "b"},
            {"system": "be brief", "instruction": "c", "output": "d"},
            {"system": "be loud", "instruction": "e", "output": "f"},
        ]
        stats = ds.get_dataset_stats(rows)
        assert stats.format_detected == DatasetFormat.ALPACA
        assert stats.has_system_prompts is True
        assert stats.unique_system_prompts == 2
        assert stats.avg_turns_per_conversation == 3

    def test_difficulty_score_edges(self):
        assert ds.compute_difficulty_score("") == 0.0
        # whitespace-only: text is truthy but has no words -> length score only
        assert ds.compute_difficulty_score("   ") == pytest.approx(3 / 5000)
        easy = ds.compute_difficulty_score("a a a a a a")
        hard = ds.compute_difficulty_score("quantum chromodynamics describes hadronic interactions")
        assert 0.0 <= easy < hard <= 1.0
        assert ds.compute_difficulty_score({"text": "x"}) == ds.compute_difficulty_score("x")
        assert ds.compute_difficulty_score({"body": "word"}, key="body") > 0

    def test_order_by_difficulty_directions(self):
        hardest = " ".join(f"word{i}longer" for i in range(300))
        rows = ["a a", hardest, "mid sized text here"]
        asc = ds.order_by_difficulty(rows)
        desc = ds.order_by_difficulty(rows, ascending=False)
        assert asc == ["a a", "mid sized text here", hardest]
        assert desc == list(reversed(asc))

    def test_curriculum_chunks_cover_all_samples_easy_to_hard(self):
        rows = [{"text": "w " * n} for n in range(1, 12)]
        chunks = ds.get_curriculum_chunks(rows, num_chunks=3)
        assert [len(c) for c in chunks] == [3, 3, 5]  # remainder lands in the last chunk
        flat = [r["text"] for c in chunks for r in c]
        assert sorted(flat) == sorted(r["text"] for r in rows)
        scores = [ds.compute_difficulty_score(r) for c in chunks for r in c]
        assert scores == sorted(scores)

    def test_curriculum_chunks_clamp_non_positive_count(self):
        chunks = ds.get_curriculum_chunks(["a", "b", "c"], num_chunks=0)
        assert len(chunks) == 1 and len(chunks[0]) == 3

    def test_analyze_curriculum_empty(self):
        stats = ds.analyze_curriculum([])
        assert stats.total_samples == 0 and stats.num_chunks == 0
        assert stats.chunk_sizes == [] and stats.difficulty_ranges == []

    def test_analyze_curriculum_clamps_chunks_to_sample_count(self):
        stats = ds.analyze_curriculum(["one", "two words", "three whole words"], num_chunks=10)
        assert stats.num_chunks == 3
        assert stats.chunk_sizes == [1, 1, 1]
        ranges = stats.difficulty_ranges
        assert all(lo <= hi for lo, hi in ranges)
        assert ranges == sorted(ranges)

    def test_curriculum_stats_summary(self):
        stats = ds.analyze_curriculum(["a", "b c", "d e f", "g h i j"], num_chunks=2)
        text = stats.summary()
        assert "Total samples: 4" in text
        assert "Chunk 1: 2 samples" in text and "Chunk 2: 2 samples" in text


class TestSplitDatasetGuards:
    def test_ratio_must_be_open_interval(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(InvalidSettingError) as exc:
                ds.split_dataset([{"i": 1}, {"i": 2}], bad, seed=0)
            assert exc.value.code == "CONFIG_INVALID_SETTING"

    def test_needs_two_rows(self):
        with pytest.raises(InvalidSettingError) as exc:
            ds.split_dataset([{"i": 1}], 0.5, seed=0)
        assert exc.value.code == "CONFIG_INVALID_SETTING"

    def test_both_sides_non_empty_and_deterministic(self):
        rows = [{"i": i} for i in range(10)]
        train, held = ds.split_dataset(rows, 0.001, seed=3)
        assert len(held) == 1 and len(train) == 9
        train2, held2 = ds.split_dataset(rows, 0.999, seed=3)
        assert len(train2) == 1 and len(held2) == 9
        again = ds.split_dataset(rows, 0.001, seed=3)
        assert again == (train, held)
        assert rows == [{"i": i} for i in range(10)]  # caller's list untouched
        assert sorted(r["i"] for r in train + held) == list(range(10))
