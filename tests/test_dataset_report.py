"""
Tests for the dataset-quality report module (v1.5 T1.1).

Mirrors the class-based style of tests/test_datasets.py. Covers:
- a clean dataset -> PASS with zero flags
- injected exact + near duplicates -> cluster count + rate + DETERMINISM
- mixed-format file -> format_distribution shows both formats
- empty-turn rows counted
- histogram bins sum to parseable_rows; a planted 50k-token row flagged outlier
- contamination_overlap: 2 of 10 train rows duplicated from held-out -> 0.2
- threshold gate: fail_on_dups on a 30%-dup set -> FAIL naming dups
- to_dict() JSON-safety + summary() CJK caveat
- the module imports no torch
"""

import json

import pytest

from backpropagate.dataset_report import (
    ContaminationResult,
    DataQualityReport,
    analyze_dataset,
    contamination_overlap,
    find_duplicate_clusters,
    token_length_histogram,
    trace_length_histogram,
)

# =============================================================================
# FIXTURES / HELPERS
# =============================================================================


def _alpaca(instruction: str, output: str, input_text: str = "") -> dict:
    return {"instruction": instruction, "input": input_text, "output": output}


def _sharegpt(user: str, assistant: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": user},
            {"from": "gpt", "value": assistant},
        ]
    }


# Genuinely diverse prompt/answer pairs. NOT templated — two rows here share
# little trigram overlap, so a correct near-dup detector leaves them as
# singletons (an earlier fixture used a single template with only an integer
# swapped, which has Jaccard ~0.91 and is a REAL near-duplicate at threshold
# 0.9 — the detector was right and the fixture was wrong).
_DIVERSE_PAIRS = [
    ("How do tides work?", "Tides arise from the gravitational pull of the moon and sun on the oceans."),
    ("Name three jazz musicians.", "Miles Davis, John Coltrane, and Thelonious Monk shaped modern jazz."),
    ("What is photosynthesis?", "Plants convert sunlight, water, and carbon dioxide into glucose and oxygen."),
    ("Explain inflation briefly.", "Inflation is a sustained rise in the general price level over time."),
    ("Why is the sky blue?", "Shorter blue wavelengths scatter more in the atmosphere than red ones."),
    ("Describe a sourdough starter.", "A starter is a fermented culture of flour and water full of wild yeast."),
    ("What causes earthquakes?", "Tectonic plates slip along faults, releasing stored elastic energy."),
    ("Summarize the water cycle.", "Water evaporates, condenses into clouds, precipitates, and runs off."),
    ("How does a transistor switch?", "A small base current gates a much larger current between collector and emitter."),
    ("What is a black hole?", "A region where gravity is so strong not even light can escape it."),
    ("Define entropy simply.", "Entropy measures disorder; isolated systems trend toward higher entropy."),
    ("Explain a hash table.", "Keys map to buckets via a hash function for average constant-time lookup."),
    ("What is a monsoon?", "A seasonal reversal of prevailing winds bringing heavy regional rainfall."),
    ("Describe DNA structure.", "A double helix of paired nucleotide bases held by hydrogen bonds."),
    ("Why do we dream?", "Dreaming may consolidate memory and process emotion during REM sleep."),
    ("What is compound interest?", "Interest earned on both the principal and previously accrued interest."),
    ("Explain a peer-to-peer network.", "Nodes share resources directly without a central coordinating server."),
    ("How do vaccines work?", "They train the immune system using a harmless piece or version of a pathogen."),
    ("What is a glacier?", "A persistent mass of dense ice that slowly flows under its own weight."),
    ("Describe the Doppler effect.", "Observed wave frequency shifts as a source moves toward or away from you."),
    ("What is a binary search?", "Repeatedly halving a sorted range to locate a target in logarithmic time."),
    ("Explain ocean acidification.", "Absorbed carbon dioxide lowers seawater pH, stressing shelled marine life."),
]


def _clean_dataset(n: int = 12) -> list[dict]:
    """A clean, all-distinct Alpaca dataset with healthy, diverse turn content."""
    if n > len(_DIVERSE_PAIRS):
        raise ValueError(
            f"_clean_dataset supports up to {len(_DIVERSE_PAIRS)} distinct rows"
        )
    return [_alpaca(q, a) for q, a in _DIVERSE_PAIRS[:n]]


# =============================================================================
# CLEAN DATASET -> PASS
# =============================================================================


class TestCleanDataset:
    """A healthy dataset should pass with no flags."""

    def test_clean_dataset_passes(self):
        report = analyze_dataset(_clean_dataset())
        assert report.verdict == "PASS"
        assert report.failed_thresholds == []

    def test_clean_dataset_no_flags(self):
        report = analyze_dataset(_clean_dataset())
        assert report.exact_duplicates == 0
        assert report.duplicate_clusters == 0
        assert report.near_duplicate_rate == 0.0
        assert report.empty_turn_rows == 0
        assert report.no_assistant_rows == 0
        assert report.outlier_rows == 0
        assert report.parse_errors == 0

    def test_clean_dataset_parseable_equals_total(self):
        samples = _clean_dataset(10)
        report = analyze_dataset(samples)
        assert report.total_rows == 10
        assert report.parseable_rows == 10

    def test_format_distribution_single_format(self):
        report = analyze_dataset(_clean_dataset(8))
        assert report.format_distribution == {"alpaca": 8}


# =============================================================================
# DUPLICATES + DETERMINISM
# =============================================================================


class TestDuplicates:
    """Exact + near duplicates are detected, counted, and deterministic."""

    def _dup_dataset(self) -> list[dict]:
        # 6 distinct (diverse) base rows + 2 exact dups + 2 near dups.
        base = _clean_dataset(6)
        # exact duplicates of rows 0 and 1
        exact = [dict(base[0]), dict(base[1])]
        # near-duplicates of rows 2 and 3: same long answer with a few words
        # appended (Jaccard stays well above 0.9 -> a genuine near-dup).
        near0 = dict(base[2])
        near0["output"] = base[2]["output"] + " This is well established."
        near1 = dict(base[3])
        near1["output"] = base[3]["output"] + " That is the short of it."
        return base + exact + [near0, near1]

    def test_exact_duplicates_counted(self):
        report = analyze_dataset(self._dup_dataset())
        # The two exact copies of rows 0 and 1.
        assert report.exact_duplicates == 2

    def test_near_duplicate_clusters_detected(self):
        clusters, exact, rate = find_duplicate_clusters(self._dup_dataset())
        # rows 0,2 + their exact/near copies -> at least 2 multi-member clusters
        # (the exact pair for row0/row1 and the near pair for row2/row3).
        assert clusters >= 2
        assert exact == 2
        assert rate > 0.0

    def test_near_duplicate_rate_in_range(self):
        report = analyze_dataset(self._dup_dataset())
        assert 0.0 < report.near_duplicate_rate <= 1.0

    def test_determinism_identical_to_dict(self):
        samples = self._dup_dataset()
        r1 = analyze_dataset(samples)
        r2 = analyze_dataset(samples)
        assert r1.to_dict() == r2.to_dict()

    def test_determinism_repeated_clusters(self):
        samples = self._dup_dataset()
        a = find_duplicate_clusters(samples)
        b = find_duplicate_clusters(samples)
        c = find_duplicate_clusters(samples)
        assert a == b == c

    def test_determinism_across_fresh_call_order(self):
        # Run an unrelated analysis between the two identical calls to prove
        # there is no hidden mutable/global state leaking between calls.
        samples = self._dup_dataset()
        r1 = analyze_dataset(samples)
        _ = analyze_dataset(_clean_dataset())
        r2 = analyze_dataset(samples)
        assert r1.to_dict() == r2.to_dict()

    def test_empty_rows_not_collapsed_into_one_cluster(self):
        # Two genuinely blank-text rows must NOT fold into a near-dup cluster
        # via the MinHash path (mirrors the datasets._get_ngrams empty-content
        # contract: no grams -> no signature -> kept as singletons). They will
        # still be exact-duplicate of each other (both ""), which is a separate,
        # correct signal — what must NOT happen is a *near-dup cluster*.
        samples = [{"text": ""}, {"text": "   "}]
        clusters, _exact, _rate = find_duplicate_clusters(samples)
        assert clusters == 0


# =============================================================================
# MIXED FORMAT
# =============================================================================


class TestMixedFormat:
    """A file mixing Alpaca + ShareGPT shows both in the distribution."""

    def test_mixed_format_distribution(self):
        samples = (
            [_alpaca(f"q{i}", f"a{i} long enough answer here") for i in range(9)]
            + [_sharegpt("hello there friend", "general kenobi a fine reply")]
        )
        report = analyze_dataset(samples)
        assert report.format_distribution.get("alpaca") == 9
        assert report.format_distribution.get("sharegpt") == 1

    def test_distribution_sums_to_total(self):
        samples = (
            [_alpaca(f"q{i}", f"a{i} answer body text here") for i in range(5)]
            + [_sharegpt(f"u{i}", f"resp {i} content body") for i in range(3)]
        )
        report = analyze_dataset(samples)
        assert sum(report.format_distribution.values()) == report.total_rows


# =============================================================================
# EMPTY TURNS / NO ASSISTANT
# =============================================================================


class TestQualityFlags:
    def test_empty_turn_rows_counted(self):
        samples = _clean_dataset(6)
        # Inject two rows whose assistant body is blank.
        samples.append(_alpaca("A real question with content here", ""))
        samples.append(_alpaca("Another question with body text", "   "))
        report = analyze_dataset(samples)
        assert report.empty_turn_rows == 2

    def test_no_assistant_rows_counted(self):
        samples = _clean_dataset(5)
        # A ShareGPT row that has only a user turn -> no assistant.
        samples.append(
            {"conversations": [{"from": "human", "value": "lonely user turn"}]}
        )
        report = analyze_dataset(samples)
        assert report.no_assistant_rows >= 1


# =============================================================================
# TOKEN HISTOGRAM + OUTLIERS
# =============================================================================


class TestHistogram:
    def test_histogram_bins_sum_to_parseable(self):
        samples = _clean_dataset(15)
        report = analyze_dataset(samples)
        total_in_hist = sum(count for _upper, count in report.token_histogram)
        assert total_in_hist == report.parseable_rows

    def test_histogram_last_bucket_is_inf(self):
        hist = token_length_histogram(_clean_dataset(4))
        assert hist[-1][0] == -1  # -1 upper bound marks the "inf" bucket

    def test_histogram_custom_bins(self):
        hist = token_length_histogram(_clean_dataset(6), bins=(10, 100))
        # 2 explicit bins + 1 overflow bucket.
        assert len(hist) == 3
        assert hist[-1][0] == -1

    def test_planted_giant_row_is_outlier(self):
        # A planted ~50k-token row (≈200k chars at 4 chars/token) must be
        # flagged as a length outlier at sigma=3.
        samples = _clean_dataset(20)
        giant = _alpaca("Summarize this enormous document", "x" * 200_000)
        samples.append(giant)
        report = analyze_dataset(samples, outlier_sigma=3.0)
        assert report.outlier_rows >= 1

    def test_giant_row_lands_in_inf_bucket(self):
        samples = _clean_dataset(5)
        samples.append(_alpaca("huge", "y" * 200_000))
        hist = token_length_histogram(samples)
        # The overflow bucket should hold the one giant row.
        assert hist[-1][1] >= 1


# =============================================================================
# CONTAMINATION
# =============================================================================


class TestContamination:
    def test_two_of_ten_overlap_rate(self):
        # 10 train rows; 2 are exact duplicates of held-out rows -> rate 0.2.
        held_out = _clean_dataset(4)
        train = [
            _alpaca(q, a) for q, a in
            [(p[0], p[1]) for p in _DIVERSE_PAIRS[10:18]]  # 8 disjoint rows
        ]
        # Make 2 of the train rows identical to 2 held-out rows.
        train.append(dict(held_out[0]))
        train.append(dict(held_out[1]))
        assert len(train) == 10
        result = contamination_overlap(train, held_out)
        assert isinstance(result, ContaminationResult)
        assert result.overlap_rows == 2
        assert result.overlap_rate == pytest.approx(0.2)

    def test_no_contamination_when_disjoint(self):
        train = [_alpaca(q, a) for q, a in _DIVERSE_PAIRS[:6]]
        held_out = [_alpaca(q, a) for q, a in _DIVERSE_PAIRS[14:20]]
        result = contamination_overlap(train, held_out)
        assert result.overlap_rows == 0
        assert result.overlap_rate == 0.0

    def test_contamination_wired_into_report(self):
        held_out = [_alpaca(q, a) for q, a in _DIVERSE_PAIRS[14:18]]
        train = _clean_dataset(6) + [dict(held_out[0])]
        report = analyze_dataset(
            train, against=held_out, against_path="test.jsonl"
        )
        assert isinstance(report, DataQualityReport)
        assert report.contamination is not None
        assert report.contamination.against_path == "test.jsonl"
        assert report.contamination.overlap_rows >= 1

    def test_contamination_none_without_against(self):
        report = analyze_dataset(_clean_dataset(4))
        assert report.contamination is None

    def test_empty_inputs_safe(self):
        result = contamination_overlap([], [])
        assert result.overlap_rows == 0
        assert result.overlap_rate == 0.0


# =============================================================================
# THRESHOLD GATES / VERDICT
# =============================================================================


class TestGates:
    def _thirty_percent_dup_set(self) -> list[dict]:
        # 10 rows, 3 of them exact duplicates of row 0 -> 0.3 near-dup rate.
        base = _clean_dataset(7)
        dups = [dict(base[0]), dict(base[0]), dict(base[0])]
        return base + dups

    def test_fail_on_dups_trips_fail(self):
        samples = self._thirty_percent_dup_set()
        report = analyze_dataset(samples, fail_on_dups=0.1)
        assert report.verdict == "FAIL"

    def test_failed_thresholds_names_dups(self):
        samples = self._thirty_percent_dup_set()
        report = analyze_dataset(samples, fail_on_dups=0.1)
        assert any("dup" in gate.lower() for gate in report.failed_thresholds)

    def test_loose_threshold_does_not_fail(self):
        samples = self._thirty_percent_dup_set()
        report = analyze_dataset(samples, fail_on_dups=0.95)
        assert report.verdict != "FAIL"

    def test_fail_on_contamination_trips(self):
        held_out = [_alpaca(q, a) for q, a in _DIVERSE_PAIRS[12:17]]
        # 4 of 9 train rows contaminated -> ~0.44 contamination.
        train = [_alpaca(q, a) for q, a in _DIVERSE_PAIRS[:5]]
        train += [dict(held_out[j]) for j in range(4)]
        report = analyze_dataset(
            train,
            against=held_out,
            against_path="eval.jsonl",
            fail_on_contamination=0.01,
        )
        assert report.verdict == "FAIL"
        assert any(
            "contam" in gate.lower() for gate in report.failed_thresholds
        )

    def test_max_outlier_rate_gate(self):
        samples = _clean_dataset(10)
        samples.append(_alpaca("huge one", "z" * 200_000))
        report = analyze_dataset(samples, max_outlier_rate=0.0)
        assert report.verdict == "FAIL"
        assert any(
            "outlier" in gate.lower() for gate in report.failed_thresholds
        )

    def test_strict_preset_promotes_dups_to_hard_gate(self):
        samples = self._thirty_percent_dup_set()
        # ~30% near-dup rate exceeds the strict 0.25 dup gate.
        report = analyze_dataset(samples, strict=True)
        assert report.verdict == "FAIL"

    def test_strict_clean_dataset_still_passes(self):
        report = analyze_dataset(_clean_dataset(12), strict=True)
        assert report.verdict == "PASS"


# =============================================================================
# REPORT SERIALIZATION + SUMMARY
# =============================================================================


class TestSerialization:
    def test_to_dict_is_json_safe(self):
        report = analyze_dataset(
            _clean_dataset(6),
            against=[_alpaca("h", "held answer body here")],
            against_path="test.jsonl",
        )
        payload = report.to_dict()
        # Must round-trip through JSON without error.
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        assert decoded["verdict"] in ("PASS", "WARN", "FAIL")
        # v1.5 T3.2: the appended reasoning-trace keys must be present and
        # JSON-safe (histogram tuples normalised to lists).
        assert decoded["think_rows"] == 0
        assert decoded["think_pct"] == 0.0
        assert all(isinstance(b, list) for b in decoded["trace_histogram"])

    def test_to_dict_histogram_is_lists(self):
        report = analyze_dataset(_clean_dataset(4))
        payload = report.to_dict()
        assert all(isinstance(b, list) for b in payload["token_histogram"])

    def test_to_dict_contamination_nested_dict(self):
        report = analyze_dataset(
            _clean_dataset(4),
            against=[_alpaca("h", "held body")],
            against_path="x.jsonl",
        )
        payload = report.to_dict()
        assert isinstance(payload["contamination"], dict)
        assert payload["contamination"]["against_path"] == "x.jsonl"

    def test_to_dict_contamination_none(self):
        report = analyze_dataset(_clean_dataset(4))
        payload = report.to_dict()
        assert payload["contamination"] is None

    def test_summary_is_multiline_string(self):
        report = analyze_dataset(_clean_dataset(6))
        text = report.summary()
        assert isinstance(text, str)
        assert "\n" in text
        assert "Verdict:" in text

    def test_summary_mentions_cjk_caveat(self):
        report = analyze_dataset(_clean_dataset(6))
        text = report.summary().lower()
        assert "cjk" in text


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    def test_empty_dataset(self):
        report = analyze_dataset([])
        assert report.total_rows == 0
        assert report.parseable_rows == 0
        assert report.verdict in ("PASS", "WARN", "FAIL")

    def test_all_unparseable_rows_counted(self):
        # Rows with no recognizable format -> parse_errors.
        samples = [{"random_key": "value"}, {"another": "thing"}]
        report = analyze_dataset(samples)
        assert report.parse_errors == 2
        assert report.parseable_rows == 0

    def test_format_hint_applied_to_unknown(self):
        # A bare {"text": "..."} that isn't ChatML detects as UNKNOWN; the hint
        # is recorded but doesn't override real detection elsewhere.
        report = analyze_dataset(
            [{"instruction": "x", "output": "a body answer here"}],
            format_hint="alpaca",
        )
        assert "alpaca" in report.format_distribution

    def test_no_torch_imported(self):
        # The module must stay cheap: it must not import torch itself.
        import backpropagate.dataset_report as dr

        # We can't assert torch is wholly absent from sys.modules (the package
        # __init__ pulls it). Instead assert the module's own globals never bind
        # a torch reference, and its source never imports torch.
        assert "torch" not in vars(dr)
        import inspect

        source = inspect.getsource(dr)
        assert "import torch" not in source

    def test_single_row_dataset(self):
        report = analyze_dataset([_alpaca("hi", "a one-line answer body")])
        assert report.total_rows == 1
        assert report.parseable_rows == 1
        assert report.duplicate_clusters == 0


# =============================================================================
# REASONING-TRACE REPORT FIELDS (v1.5 T3.2)
# =============================================================================


def _think_alpaca(instruction: str, reasoning: str, answer: str) -> dict:
    """An Alpaca row whose output carries a <think> reasoning trace."""
    return _alpaca(instruction, f"<think>{reasoning}</think>{answer}")


class TestReasoningTraceFields:
    """think_rows / think_pct / trace_histogram on the report."""

    def _mixed_set(self) -> list[dict]:
        # 4 rows: 2 carry a <think> trace, 2 are plain.
        return [
            _think_alpaca("q1", "a" * 40, "The answer is one."),
            _think_alpaca("q2", "b" * 40, "The answer is two."),
            _alpaca("q3", "A plain answer with enough body text here."),
            _alpaca("q4", "Another plain answer with body text here."),
        ]

    def test_think_rows_counted(self):
        report = analyze_dataset(self._mixed_set())
        assert report.think_rows == 2

    def test_think_pct_is_half(self):
        report = analyze_dataset(self._mixed_set())
        assert report.think_pct == pytest.approx(0.5)

    def test_trace_histogram_sums_to_think_rows(self):
        report = analyze_dataset(self._mixed_set())
        total = sum(count for _upper, count in report.trace_histogram)
        assert total == report.think_rows == 2

    def test_trace_histogram_last_bucket_is_inf(self):
        report = analyze_dataset(self._mixed_set())
        assert report.trace_histogram[-1][0] == -1

    def test_non_reasoning_dataset_zero(self):
        # A plain (contamination-style) dataset stays at zero — and quiet.
        report = analyze_dataset(_clean_dataset(6))
        assert report.think_rows == 0
        assert report.think_pct == 0.0
        assert all(count == 0 for _u, count in report.trace_histogram)

    def test_summary_shows_block_only_when_traces_present(self):
        with_traces = analyze_dataset(self._mixed_set()).summary()
        without = analyze_dataset(_clean_dataset(6)).summary()
        assert "Reasoning traces:" in with_traces
        assert "Reasoning traces:" not in without

    def test_to_dict_roundtrips_new_keys(self):
        report = analyze_dataset(self._mixed_set())
        payload = report.to_dict()
        decoded = json.loads(json.dumps(payload))
        assert decoded["think_rows"] == 2
        assert decoded["think_pct"] == pytest.approx(0.5)
        # trace_histogram tuples must be normalised to lists for JSON.
        assert all(isinstance(b, list) for b in decoded["trace_histogram"])
        assert sum(b[1] for b in decoded["trace_histogram"]) == 2

    def test_empty_dataset_trace_fields_safe(self):
        report = analyze_dataset([])
        assert report.think_rows == 0
        assert report.think_pct == 0.0


class TestTraceLengthHistogram:
    """The standalone trace_length_histogram primitive."""

    def test_skips_non_reasoning_rows(self):
        samples = [
            _think_alpaca("q", "a" * 40, "ans"),
            _alpaca("q2", "plain body with no reasoning here"),
        ]
        hist = trace_length_histogram(samples)
        # Only the 1 reasoning row is counted.
        assert sum(count for _u, count in hist) == 1

    def test_custom_bins(self):
        hist = trace_length_histogram(
            [_think_alpaca("q", "a" * 40, "ans")], bins=(4, 16)
        )
        assert len(hist) == 3  # 2 explicit + 1 overflow
        assert hist[-1][0] == -1

    def test_empty_samples(self):
        hist = trace_length_histogram([])
        assert sum(count for _u, count in hist) == 0

    def test_long_trace_lands_in_inf_bucket(self):
        # A ~200k-char trace (≈50k tokens) overflows the default last bin (8192).
        hist = trace_length_histogram(
            [_think_alpaca("q", "x" * 200_000, "ans")]
        )
        assert hist[-1][1] == 1


# =============================================================================
# SINGLE-PASS CONVERSION (TRAINER-DATA-003)
# =============================================================================


class TestSinglePassConversion:
    """The module's "single pass" / "cheap, torch-free" contract (docstring
    lines 5, 11): ``analyze_dataset`` must convert each sample to ChatML ONCE.

    Regression guard for TRAINER-DATA-003: the format-distribution loop,
    ``token_length_histogram``, and ``trace_length_histogram`` each used to
    re-run ``convert_to_chatml`` over ALL samples, costing ~3x per report.
    ``convert_to_chatml`` is the single shared primitive every per-sample
    conversion routes through (via ``_to_chatml_text``), so counting its calls
    counts conversions. It must be invoked exactly once per sample.
    """

    def _spy(self, monkeypatch):
        from backpropagate import dataset_report as dr

        real = dr.convert_to_chatml
        counter = {"per_sample": 0, "batch": 0}

        def _counting(samples, *args, **kwargs):
            counter["batch"] += 1
            counter["per_sample"] += len(samples)
            return real(samples, *args, **kwargs)

        monkeypatch.setattr(dr, "convert_to_chatml", _counting)
        return counter

    def test_clean_dataset_converts_each_sample_once(self, monkeypatch):
        counter = self._spy(monkeypatch)
        samples = _clean_dataset(15)
        analyze_dataset(samples)
        # Exactly one conversion per sample — not 3x (format-dist + token-hist
        # + trace-hist). _to_chatml_text converts one sample per call, so the
        # per-sample tally equals the number of _to_chatml_text invocations.
        assert counter["per_sample"] == len(samples)

    def test_reasoning_dataset_converts_each_sample_once(self, monkeypatch):
        # Reasoning rows exercise the trace-histogram path too.
        counter = self._spy(monkeypatch)
        samples = [
            _think_alpaca("q1", "reasoning one here", "ans one"),
            _think_alpaca("q2", "reasoning two here", "ans two"),
            _alpaca("plain", "a plain answer with no trace"),
        ]
        analyze_dataset(samples)
        assert counter["per_sample"] == len(samples)

    def test_against_set_converts_each_held_out_row_once(self, monkeypatch):
        # The contamination path converts the held-out rows once each, on top
        # of the train-set single pass.
        counter = self._spy(monkeypatch)
        train = _clean_dataset(8)
        against = _clean_dataset(4)
        analyze_dataset(train, against=against)
        assert counter["per_sample"] == len(train) + len(against)

    def test_report_is_byte_identical_after_single_pass(self):
        # Output must not change: same input -> same report dict.
        samples = _clean_dataset(12)
        a = analyze_dataset(samples).to_dict()
        b = analyze_dataset(samples).to_dict()
        assert a == b
