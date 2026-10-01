"""Coverage tests for the ``backpropagate.datasets`` loaders and perplexity filter.

Real files (JSONL / JSON / text / CSV / parquet) are written to ``tmp_path``
and loaded through ``DatasetLoader`` / ``StreamingDatasetLoader``. The
perplexity filter runs a real tiny GPT-2 (random weights, built in-process
and saved under ``tmp_path``) on CPU.

What is mocked (and nothing else):

* ``datasets.load_dataset`` in the streaming-from-the-Hub tests (network).
* ``sys.modules`` entries (``pandas`` / ``pyarrow`` / ``datasets`` /
  ``transformers`` / ``torch``) to simulate absent optional extras.
* ``torch.cuda.is_available`` / ``torch.cuda.empty_cache`` (GPU boundary) so
  nothing here ever touches CUDA.
* The exact-Jaccard ``datasketch`` stand-in from ``test_datasets_cov_core``
  where ``DatasetLoader.deduplicate("minhash")`` is exercised.
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
    DatasetLoader,
    PerplexityFilter,
    StreamingDatasetLoader,
)
from backpropagate.exceptions import (
    BackpropagateError,
    DatasetError,
    DatasetFormatError,
    DatasetNotFoundError,
    DatasetParseError,
    InvalidSettingError,
)
from tests.helpers.tiny_models import tiny_gpt2, tiny_tokenizer
from tests.test_datasets_cov_core import _FakeLSH, _FakeMinHash

LOGGER = "backpropagate.datasets"

ALPACA = {"instruction": "Say hi", "output": "hi there"}


@pytest.fixture
def fake_datasketch(monkeypatch):
    """Install the exact-Jaccard ``datasketch`` stand-in (see module docstring)."""
    mod = types.ModuleType("datasketch")
    mod.MinHash = _FakeMinHash  # type: ignore[attr-defined]
    mod.MinHashLSH = _FakeLSH  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasketch", mod)
    return mod


def _write_jsonl(path: Path, rows: list, *, junk: int = 0, junk_first: bool = False) -> Path:
    lines = [json.dumps(r) for r in rows]
    bad = ["{not json" for _ in range(junk)]
    lines = bad + lines if junk_first else lines + bad
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _chatml(user: str, assistant: str) -> str:
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


# =============================================================================
# DatasetLoader: loading from disk
# =============================================================================


class TestLoaderJsonl:
    def test_loads_rows_and_detects_format(self, tmp_path):
        p = _write_jsonl(tmp_path / "d.jsonl", [ALPACA, ALPACA])
        loader = DatasetLoader(p)
        assert len(loader) == 2
        assert loader.detected_format == DatasetFormat.ALPACA
        assert loader.is_valid is True
        assert loader[0] == ALPACA
        assert list(iter(loader)) == [ALPACA, ALPACA]

    def test_missing_file_raises_structured_not_found(self, tmp_path):
        with pytest.raises(DatasetNotFoundError) as exc:
            DatasetLoader(tmp_path / "absent.jsonl")
        assert exc.value.code == "INPUT_DATASET_NOT_FOUND"
        assert "absent.jsonl" in exc.value.message

    def test_unknown_extension_detection_skips_blank_lines(self, tmp_path):
        p = tmp_path / "d.dat"
        p.write_text("\n" + json.dumps(ALPACA) + "\n\n" + json.dumps(ALPACA) + "\n", encoding="utf-8")
        assert ds._detect_format_from_file(p) == DatasetFormat.ALPACA
        assert len(DatasetLoader(p)) == 2

    def test_unknown_extension_detection_stops_at_sample_window(self, tmp_path):
        alpaca = json.dumps(ALPACA)
        openai = json.dumps({"messages": [{"role": "user", "content": "x"}]})
        p = tmp_path / "d.dat"
        p.write_text("\n".join([alpaca] * 2 + [openai] * 8), encoding="utf-8")
        assert ds._detect_format_from_file(p, sample_size=2) == DatasetFormat.ALPACA

    def test_unknown_extension_is_read_as_jsonl(self, tmp_path):
        p = _write_jsonl(tmp_path / "d.dat", [ALPACA])
        assert DatasetLoader(p).detected_format == DatasetFormat.ALPACA

    def test_few_bad_lines_are_skipped_with_per_line_warnings(self, tmp_path, caplog):
        p = _write_jsonl(tmp_path / "d.jsonl", [ALPACA] * 5, junk=2)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            loader = DatasetLoader(p)
        assert len(loader) == 5
        invalid = [r.getMessage() for r in caplog.records if "Invalid JSON on line" in r.getMessage()]
        assert len(invalid) == 2
        assert not any("High JSONL parse failure" in r.getMessage() for r in caplog.records)

    def test_warnings_are_capped_then_summarised(self, tmp_path, caplog):
        """25 bad lines out of 55: first 20 verbatim, one suppression note, one summary."""
        p = _write_jsonl(tmp_path / "d.jsonl", [ALPACA] * 30, junk=25)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            loader = DatasetLoader(p)
        assert len(loader) == 30
        msgs = [r.getMessage() for r in caplog.records]
        assert sum("Invalid JSON on line" in m for m in msgs) == 20
        assert sum("suppressing per-line warnings" in m for m in msgs) == 1
        summary = [m for m in msgs if m.startswith("JSONL load:")]
        assert len(summary) == 1 and "25/55" in summary[0] and "kept 30" in summary[0]

    def test_majority_bad_lines_warns_with_percentage(self, tmp_path, caplog):
        p = _write_jsonl(tmp_path / "d.jsonl", [ALPACA] * 2, junk=6)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            loader = DatasetLoader(p)
        assert len(loader) == 2
        high = [r.getMessage() for r in caplog.records if "High JSONL parse failure" in r.getMessage()]
        assert len(high) == 1
        assert "6/8 lines (75%)" in high[0]
        assert "the 2 surviving samples" in high[0]

    def test_all_lines_bad_raises_parse_error_with_code(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("{nope\n[also nope\n\n", encoding="utf-8")
        with pytest.raises(DatasetParseError) as exc:
            DatasetLoader(p)
        assert exc.value.code == "INPUT_DATASET_PARSE_FAILED"
        assert "All 2 non-empty lines" in exc.value.message
        assert exc.value.path == str(p)

    def test_undecodable_bytes_wrap_into_plain_value_error(self, tmp_path):
        p = tmp_path / "bin.jsonl"
        p.write_bytes(b"\xff\xfe\x00 not utf-8\n")
        with pytest.raises(ValueError, match="Failed to load dataset") as exc:
            DatasetLoader(p)
        assert not isinstance(exc.value, BackpropagateError)
        assert isinstance(exc.value.__cause__, UnicodeDecodeError)


class TestLoaderJsonAndText:
    def test_json_array(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text(json.dumps([ALPACA, ALPACA, ALPACA]), encoding="utf-8")
        assert len(DatasetLoader(p)) == 3

    def test_json_single_object_becomes_one_row(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text(json.dumps(ALPACA), encoding="utf-8")
        loader = DatasetLoader(p)
        assert loader.samples == [ALPACA]

    def test_malformed_json_raises_parse_error_with_line_number(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text('{"a": 1,\n "b": }', encoding="utf-8")
        with pytest.raises(DatasetParseError) as exc:
            DatasetLoader(p)
        assert exc.value.code == "INPUT_DATASET_PARSE_FAILED"
        assert exc.value.line_number == 2
        assert "(line 2)" in exc.value.message

    def test_text_split_on_blank_lines(self, tmp_path):
        p = tmp_path / "d.txt"
        p.write_text("first para\n\nsecond para\n\n\n\nthird", encoding="utf-8")
        loader = DatasetLoader(p)
        assert loader.samples == ["first para", "second para", "third"]
        assert loader.detected_format == DatasetFormat.RAW_TEXT

    def test_markdown_without_blank_lines_is_one_sample(self, tmp_path):
        p = tmp_path / "d.md"
        p.write_text("single block\nof text", encoding="utf-8")
        assert DatasetLoader(p).samples == ["single block\nof text"]


class TestLoaderTabular:
    def test_csv_empty_cells_become_none_and_warn(self, tmp_path, caplog):
        p = tmp_path / "d.csv"
        p.write_text("instruction,output\nhello,world\nfoo,\n", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            loader = DatasetLoader(p)
        assert loader.samples == [
            {"instruction": "hello", "output": "world"},
            {"instruction": "foo", "output": None},
        ]
        assert loader.detected_format == DatasetFormat.ALPACA
        # the blank cell is surfaced as a validation warning, not an error
        assert loader.is_valid is True
        assert loader.validation_result.warning_count == 1
        assert any("output=1" in r.getMessage() for r in caplog.records)

    def test_parquet_round_trip(self, tmp_path):
        import pandas as pd

        p = tmp_path / "d.parquet"
        pd.DataFrame(
            {
                "instruction": ["a", "b"],
                "output": ["x", None],
                "score": [1.5, float("nan")],
            }
        ).to_parquet(p)
        loader = DatasetLoader(p)
        assert loader.samples[0] == {"instruction": "a", "output": "x", "score": 1.5}
        assert loader.samples[1]["output"] is None
        assert loader.samples[1]["score"] is None
        assert loader.detected_format == DatasetFormat.ALPACA

    def test_records_from_df_survives_a_broken_null_probe(self, tmp_path, monkeypatch):
        import pandas as pd

        original = pd.DataFrame.isna
        calls = []

        def boom_once(self):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("isna exploded")
            return original(self)

        monkeypatch.setattr(pd.DataFrame, "isna", boom_once)
        p = tmp_path / "d.csv"
        p.write_text("instruction,output\na,b\n", encoding="utf-8")
        loader = DatasetLoader(p)  # diagnostics are best-effort; load still works
        assert loader.samples == [{"instruction": "a", "output": "b"}]

    def test_csv_without_pandas_names_the_dependency(self, tmp_path, monkeypatch):
        """Mocks: ``sys.modules['pandas'] = None`` (extra not installed)."""
        p = tmp_path / "d.csv"
        p.write_text("a,b\n1,2\n", encoding="utf-8")
        monkeypatch.setitem(sys.modules, "pandas", None)
        with pytest.raises(DatasetError) as exc:
            DatasetLoader(p)
        assert exc.value.code == "DEP_DATASET_ENGINE_MISSING"
        assert "pandas is required to load CSV" in exc.value.message
        assert "pip install pandas" in (exc.value.suggestion or "")

    def test_parquet_without_pyarrow_names_the_dependency(self, tmp_path, monkeypatch):
        """Mocks: ``sys.modules['pyarrow'] = None`` (engine not installed)."""
        p = tmp_path / "d.parquet"
        p.write_bytes(b"PAR1")
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        with pytest.raises(DatasetError) as exc:
            DatasetLoader(p)
        assert exc.value.code == "DEP_DATASET_ENGINE_MISSING"
        assert "pyarrow" in exc.value.message

    def test_parquet_without_pandas_names_the_dependency(self, tmp_path, monkeypatch):
        p = tmp_path / "d.parquet"
        p.write_bytes(b"PAR1")
        monkeypatch.setitem(sys.modules, "pandas", None)
        with pytest.raises(DatasetError) as exc:
            DatasetLoader(p)
        assert exc.value.code == "DEP_DATASET_ENGINE_MISSING"
        assert "pandas and pyarrow" in exc.value.message


class TestLoaderFromList:
    def test_list_source_detects_format_unless_pinned(self):
        rows = [{"instruction": "a", "output": "b"}]
        assert DatasetLoader(rows).detected_format == DatasetFormat.ALPACA
        pinned = DatasetLoader(rows, format_type=DatasetFormat.SHAREGPT)
        assert pinned.detected_format == DatasetFormat.SHAREGPT
        assert pinned.is_valid is False  # pinned format is enforced on every row

    def test_lazy_validation_when_skipped_on_construction(self):
        loader = DatasetLoader([{"instruction": "a"}], validate=False)
        assert loader._validation is None
        assert loader.is_valid is False  # triggers validation
        loader2 = DatasetLoader([{"instruction": "a", "output": "b"}], validate=False)
        assert loader2.validation_result.is_valid is True
        loader3 = DatasetLoader([{"instruction": "a"}], validate=False)
        report = loader3.validation_report()
        assert "Dataset Validation Report" in report and "Errors: 1" in report

    def test_from_local(self, tmp_path):
        p = _write_jsonl(tmp_path / "d.jsonl", [ALPACA])
        loader = DatasetLoader.from_local(p)
        assert len(loader) == 1
        with pytest.raises(FileNotFoundError, match="Local file not found"):
            DatasetLoader.from_local(tmp_path / "nope.jsonl")

    def test_from_streaming_returns_streaming_loader(self, tmp_path):
        p = _write_jsonl(tmp_path / "d.jsonl", [ALPACA])
        loader = DatasetLoader.from_streaming(str(p), buffer_size=7, split="train")
        assert isinstance(loader, StreamingDatasetLoader)
        assert loader.buffer_size == 7 and loader.split == "train"


# =============================================================================
# DatasetLoader: conversions and views
# =============================================================================


class TestLoaderConversions:
    def test_to_chatml_detects_per_row_unless_pinned(self):
        rows = [
            {"instruction": "a", "output": "b"},
            {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "r"}]},
        ]
        out = DatasetLoader(rows).to_chatml()
        assert out[0]["text"] == "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>"
        assert out[1]["text"] == "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\nr<|im_end|>"

    def test_to_hf_dataset_and_split_wrapper(self):
        loader = DatasetLoader([ALPACA, ALPACA])
        dataset = loader.to_hf_dataset()
        assert dataset.column_names == ["text"]
        assert len(dataset) == 2
        wrapped = loader.to_hf_dataset(split="train")
        assert set(wrapped) == {"train"} and len(wrapped["train"]) == 2

    def test_to_hf_dataset_requires_datasets_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "datasets", None)
        with pytest.raises(ImportError, match="pip install datasets"):
            DatasetLoader([ALPACA]).to_hf_dataset()

    def test_preference_dataset_keeps_raw_columns_and_normalises_prompt(self):
        msg = [{"role": "assistant", "content": "x"}]
        rows = [
            {"prompt": "q1", "chosen": msg, "rejected": msg},
            {"chosen": msg, "rejected": msg},
            "stray string row",
            {"instruction": "not a pair", "output": "o"},
        ]
        loader = DatasetLoader(rows, validate=False)
        dataset = loader.to_preference_dataset()
        assert set(dataset.column_names) == {"prompt", "chosen", "rejected"}
        assert len(dataset) == 2
        assert dataset[0]["prompt"] == "q1"
        assert dataset[1]["prompt"] is None  # implicit prompt gets a null
        # message-lists are preserved as-is (TRL renders them later)
        assert dataset[1]["chosen"] == msg
        wrapped = loader.to_preference_dataset(split="train")
        assert len(wrapped["train"]) == 2

    def test_preference_dataset_omits_prompt_column_when_no_row_has_one(self):
        loader = DatasetLoader([{"chosen": "a", "rejected": "b"}])
        assert set(loader.to_preference_dataset().column_names) == {"chosen", "rejected"}

    def test_preference_dataset_rejects_sft_data(self):
        loader = DatasetLoader([ALPACA, "plain"], validate=False)
        with pytest.raises(DatasetFormatError) as exc:
            loader.to_preference_dataset()
        assert exc.value.code == "INPUT_DATASET_FORMAT_UNSUPPORTED"
        assert "{prompt, chosen, rejected}" in exc.value.supported_formats
        assert exc.value.detected_format == "alpaca"

    def test_preference_dataset_requires_datasets_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "datasets", None)
        with pytest.raises(ImportError, match="pip install datasets"):
            DatasetLoader([{"chosen": "a", "rejected": "b"}]).to_preference_dataset()

    def test_kto_dataset_columns_and_label_coercion(self):
        rows = [
            {"prompt": "p1", "completion": "c1", "label": True},
            {"completion": "c2", "label": 0},
            {"prompt": "p3", "completion": "c3", "label": "yes"},  # not a KTO label
            {"prompt": "p4", "label": True},  # no completion
            "stray",
        ]
        loader = DatasetLoader(rows, validate=False)
        dataset = loader.to_kto_dataset()
        assert set(dataset.column_names) == {"prompt", "completion", "label"}
        assert len(dataset) == 2
        assert dataset[0] == {"prompt": "p1", "completion": "c1", "label": True}
        assert dataset[1] == {"prompt": "", "completion": "c2", "label": False}
        assert len(loader.to_kto_dataset(split="train")["train"]) == 2

    def test_kto_dataset_rejects_paired_preference_data(self):
        loader = DatasetLoader([{"chosen": "a", "rejected": "b"}])
        with pytest.raises(DatasetFormatError) as exc:
            loader.to_kto_dataset()
        assert exc.value.code == "INPUT_DATASET_FORMAT_UNSUPPORTED"
        assert "method='orpo'/'simpo'" in exc.value.message
        assert exc.value.detected_format == "preference"

    def test_kto_dataset_requires_datasets_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "datasets", None)
        with pytest.raises(ImportError, match="pip install datasets"):
            DatasetLoader([{"completion": "c", "label": True}]).to_kto_dataset()

    def test_preview_chatml_and_raw(self):
        loader = DatasetLoader([ALPACA, {"instruction": "x", "output": "y"}])
        chatml = loader.preview(1)
        assert len(chatml) == 1 and chatml[0].startswith("<|im_start|>user\nSay hi")
        raw = loader.preview(2, as_chatml=False)
        assert json.loads(raw[0]) == ALPACA
        text_loader = DatasetLoader(["plain one", "plain two"])
        assert text_loader.preview(2, as_chatml=False) == ["plain one", "plain two"]
        assert ds.preview_samples(["plain one"], n=1, as_chatml=False) == ["plain one"]

    def test_preview_with_pinned_format(self):
        loader = DatasetLoader([ALPACA], format_type=DatasetFormat.ALPACA)
        assert "Say hi" in loader.preview(1)[0]

    def test_stats(self):
        loader = DatasetLoader([ALPACA, ALPACA])
        stats = loader.stats()
        assert stats.total_samples == 2
        assert stats.format_detected == DatasetFormat.ALPACA
        pinned = DatasetLoader([ALPACA], format_type=DatasetFormat.ALPACA).stats()
        assert pinned.total_samples == 1


class TestLoaderTransformations:
    ROWS = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(20)]

    def test_shuffle_with_seed_is_deterministic_and_non_destructive(self):
        loader = DatasetLoader(list(self.ROWS))
        a = loader.shuffle(seed=7)
        b = loader.shuffle(seed=7)
        assert a.samples == b.samples
        assert sorted(s["instruction"] for s in a.samples) == sorted(r["instruction"] for r in self.ROWS)
        assert a.samples != self.ROWS  # seed 7 reorders 20 rows
        assert loader.samples == self.ROWS
        assert a.detected_format == DatasetFormat.ALPACA

    def test_shuffle_without_seed_keeps_every_row(self):
        loader = DatasetLoader(list(self.ROWS))
        shuffled = loader.shuffle()
        assert len(shuffled) == 20
        assert sorted(s["instruction"] for s in shuffled.samples) == sorted(
            r["instruction"] for r in self.ROWS
        )

    def test_split_ratio_and_disjoint(self):
        loader = DatasetLoader(list(self.ROWS))
        train, test = loader.split(train_ratio=0.75, seed=1)
        assert (len(train), len(test)) == (15, 5)
        seen = [s["instruction"] for s in train.samples + test.samples]
        assert sorted(seen) == sorted(r["instruction"] for r in self.ROWS)

    def test_filter_drops_by_token_turn_and_custom_rules(self):
        rows = [
            {"instruction": "short", "output": "x"},
            {"instruction": "long " * 100, "output": "answer " * 100},
        ]
        loader = DatasetLoader(rows)
        kept = loader.filter(min_tokens=50)
        assert len(kept) == 1
        assert kept.detected_format == DatasetFormat.CHATML
        assert len(loader.filter(max_tokens=20)) == 1
        assert len(loader.filter(min_turns=3)) == 0
        assert len(loader.filter(max_turns=1)) == 0
        assert len(loader.filter(custom_filter=lambda s: "long" in s["text"])) == 1

    def test_deduplicate_exact(self):
        loader = DatasetLoader([ALPACA, ALPACA, {"instruction": "z", "output": "w"}])
        deduped = loader.deduplicate()
        assert len(deduped) == 2
        assert deduped.detected_format == DatasetFormat.CHATML

    def test_deduplicate_minhash_uses_threshold(self, fake_datasketch):
        """Mocks: ``datasketch`` (exact-Jaccard stand-in)."""
        fox = "the quick brown fox jumps over the lazy dog"
        rows = [
            {"instruction": fox, "output": "ok"},
            {"instruction": fox + ".", "output": "ok"},
            {"instruction": "something else entirely different", "output": "no"},
        ]
        deduped = DatasetLoader(rows).deduplicate(method="minhash", threshold=0.9)
        assert len(deduped) == 2

    def test_deduplicate_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown deduplication method: fuzzy"):
            DatasetLoader([ALPACA]).deduplicate(method="fuzzy")

    def test_filter_by_trace_length_returns_loader_and_stats(self):
        rows = [
            {"instruction": "q1", "output": "<think>" + "reason " * 20 + "</think>answer"},
            {"instruction": "q2", "output": "no trace at all"},
            {"instruction": "q3", "output": "<think>x</think>answer"},
        ]
        loader = DatasetLoader(rows)
        kept, stats = loader.filter_by_trace_length(min_trace_tokens=8)
        assert len(kept) == 1
        assert stats.removed_no_think == 1
        assert stats.removed_trace_too_short == 1
        assert kept.detected_format == DatasetFormat.CHATML
        counted, stats2 = loader.filter_by_trace_length(
            min_trace_tokens=1, token_counter=lambda s: 1
        )
        assert len(counted) == 2 and stats2.removed_no_think == 1


# =============================================================================
# StreamingDatasetLoader
# =============================================================================


class TestStreamingLocal:
    def test_jsonl_stream_take_skip_batches(self, tmp_path):
        rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(7)]
        loader = StreamingDatasetLoader(str(_write_jsonl(tmp_path / "d.jsonl", rows)))
        assert loader.detected_format == DatasetFormat.UNKNOWN  # nothing read yet
        assert loader.take(3) == rows[:3]
        assert loader.detected_format == DatasetFormat.ALPACA
        assert list(loader.skip(5)) == rows[5:]
        batches = list(loader.batches(3))
        assert [len(b) for b in batches] == [3, 3, 1]
        assert list(loader) == rows

    def test_jsonl_skips_bad_lines_and_summarises(self, tmp_path, caplog):
        rows = [ALPACA] * 30
        p = _write_jsonl(tmp_path / "d.jsonl", rows, junk=22)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            streamed = list(StreamingDatasetLoader(str(p)))
        assert len(streamed) == 30
        msgs = [r.getMessage() for r in caplog.records]
        assert sum("Invalid JSON on line" in m for m in msgs) == 20
        assert sum("JSONL stream: more than 20 invalid lines" in m for m in msgs) == 1
        summary = [m for m in msgs if "JSONL stream summary" in m]
        assert len(summary) == 1 and "22/52" in summary[0]

    def test_jsonl_all_bad_raises_after_stream(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text("{x\n{y\n", encoding="utf-8")
        with pytest.raises(DatasetParseError) as exc:
            list(StreamingDatasetLoader(str(p)))
        assert exc.value.code == "INPUT_DATASET_PARSE_FAILED"
        assert "All 2 non-empty line" in exc.value.message

    def test_unknown_extension_streams_as_jsonl(self, tmp_path):
        p = _write_jsonl(tmp_path / "d.dat", [ALPACA])
        assert list(StreamingDatasetLoader(str(p))) == [ALPACA]

    def test_json_array_and_object(self, tmp_path):
        arr = tmp_path / "a.json"
        arr.write_text(json.dumps([ALPACA, ALPACA]), encoding="utf-8")
        obj = tmp_path / "o.json"
        obj.write_text(json.dumps(ALPACA), encoding="utf-8")
        loader = StreamingDatasetLoader(str(arr))
        assert list(loader) == [ALPACA, ALPACA]
        assert loader.detected_format == DatasetFormat.ALPACA
        assert list(StreamingDatasetLoader(str(obj))) == [ALPACA]

    def test_json_parse_error_is_structured(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text("[1, 2,\n", encoding="utf-8")
        with pytest.raises(DatasetParseError) as exc:
            list(StreamingDatasetLoader(str(p)))
        assert exc.value.code == "INPUT_DATASET_PARSE_FAILED"
        assert exc.value.line_number == 2

    def test_text_stream_chunks_on_blank_lines(self, tmp_path):
        p = tmp_path / "d.txt"
        p.write_text("one\n\ntwo\n\n\n\nthree", encoding="utf-8")
        loader = StreamingDatasetLoader(str(p))
        assert list(loader) == ["one", "two", "three"]
        assert loader.detected_format == DatasetFormat.RAW_TEXT
        single = tmp_path / "s.md"
        single.write_text("only block", encoding="utf-8")
        assert list(StreamingDatasetLoader(str(single))) == ["only block"]

    def test_to_chatml_all_and_first_n(self, tmp_path):
        rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(4)]
        loader = StreamingDatasetLoader(str(_write_jsonl(tmp_path / "d.jsonl", rows)))
        assert len(loader.to_chatml()) == 4
        two = loader.to_chatml(n=2)
        assert len(two) == 2 and "q1" in two[1]["text"]

    def test_to_chatml_with_pinned_format(self, tmp_path):
        p = _write_jsonl(tmp_path / "d.jsonl", [ALPACA])
        loader = StreamingDatasetLoader(
            str(p), format_type=DatasetFormat.ALPACA
        )
        assert "Say hi" in loader.to_chatml()[0]["text"]

    def test_filter_applies_every_rule(self, tmp_path):
        big = "word " * 200
        rows = [
            {"instruction": "tiny", "output": "x"},
            {"instruction": big, "output": big},
            {"instruction": "mid " * 30, "output": "answer " * 30},
            {"instruction": "", "output": ""},
        ]
        loader = StreamingDatasetLoader(str(_write_jsonl(tmp_path / "d.jsonl", rows)))
        assert len(list(loader.filter(min_tokens=20))) == 2
        assert len(list(loader.filter(max_tokens=100))) == 3  # all but the long row
        assert len(list(loader.filter(min_turns=3))) == 0
        assert len(list(loader.filter(max_turns=1))) == 0
        kept = list(loader.filter(custom_filter=lambda s: s["instruction"].startswith("mid")))
        assert len(kept) == 1 and "mid" in kept[0]["text"]
        # strings stream through untouched by custom_filter (it only sees dicts)
        text_p = tmp_path / "t.txt"
        text_p.write_text("plain text", encoding="utf-8")
        text_loader = StreamingDatasetLoader(str(text_p))
        kept_text = list(text_loader.filter(custom_filter=lambda s: False, require_assistant=False))
        assert len(kept_text) == 1 and "plain text" in kept_text[0]["text"]

    def test_filter_skips_blank_converted_text(self, tmp_path):
        p = tmp_path / "d.jsonl"
        rows = [{"text": "   "}, {"text": _chatml("q", "a")}]
        _write_jsonl(p, rows)
        loader = StreamingDatasetLoader(str(p), format_type=DatasetFormat.CHATML)
        kept = list(loader.filter())
        assert [k["text"] for k in kept] == [_chatml("q", "a")]

    def test_blank_lines_are_ignored_while_streaming(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text(json.dumps(ALPACA) + "\n\n   \n" + json.dumps(ALPACA) + "\n", encoding="utf-8")
        assert list(StreamingDatasetLoader(str(p))) == [ALPACA, ALPACA]

    def test_batches_with_exact_multiple_has_no_trailing_batch(self, tmp_path):
        p = _write_jsonl(tmp_path / "d.jsonl", [ALPACA] * 4)
        batches = list(StreamingDatasetLoader(str(p)).batches(2))
        assert [len(b) for b in batches] == [2, 2]

    def test_text_stream_keeps_pinned_format(self, tmp_path):
        p = tmp_path / "d.txt"
        p.write_text("a\n\nb", encoding="utf-8")
        loader = StreamingDatasetLoader(str(p), format_type=DatasetFormat.CHATML)
        assert list(loader) == ["a", "b"]
        assert loader.detected_format == DatasetFormat.CHATML

    def test_filter_requires_assistant_unless_disabled(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text(json.dumps({"text": "<|im_start|>user\nonly user<|im_end|>"}) + "\n")
        loader = StreamingDatasetLoader(str(p), format_type=DatasetFormat.CHATML)
        assert list(loader.filter()) == []
        assert len(list(loader.filter(require_assistant=False))) == 1


class TestStreamingHub:
    """Mocks: ``datasets.load_dataset`` (the network boundary)."""

    def test_streams_hub_dataset_once_with_retry_wrapper(self, monkeypatch):
        calls = []

        def fake_load(name, **kwargs):
            calls.append((name, kwargs))
            return iter([ALPACA, ALPACA])

        monkeypatch.setattr("datasets.load_dataset", fake_load)
        loader = StreamingDatasetLoader("org/ds", split="train")
        assert loader._is_hf_dataset is True
        rows = list(loader)
        assert rows == [ALPACA, ALPACA]
        assert calls == [("org/ds", {"split": "train", "streaming": True})]
        assert loader.detected_format == DatasetFormat.ALPACA

    def test_pinned_format_is_not_overwritten(self, monkeypatch):
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: iter([ALPACA]))
        loader = StreamingDatasetLoader("org/ds", format_type=DatasetFormat.SHAREGPT)
        list(loader)
        assert loader.detected_format == DatasetFormat.SHAREGPT

    def test_requires_datasets_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "datasets", None)
        with pytest.raises(ImportError, match="pip install datasets"):
            list(StreamingDatasetLoader("org/ds"))


# =============================================================================
# Perplexity filter (real tiny GPT-2 on CPU)
# =============================================================================


@pytest.fixture(scope="module")
def tiny_ppl_dir(tmp_path_factory) -> str:
    d = tmp_path_factory.mktemp("tiny_ppl")
    tiny_gpt2(vocab=64).save_pretrained(d)
    tiny_tokenizer().save_pretrained(d)
    return str(d)


@pytest.fixture
def cpu_only(monkeypatch):
    """Mocks the GPU boundary: report no CUDA so nothing touches a GPU."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


TEXTS = [
    "the cat sat on the mat and ran to the park",
    "what is two plus four yes",
    "a dog ran to the park and the cat sat",
    "user what is the cat assistant yes it is a dog",
    "the dog and the cat and the mat and the park",
    "two plus two is four and no is not yes",
    "assistant the mat is on the cat",
    "ran ran ran to the to the to the park",
]


def _samples(texts=TEXTS):
    return [{"text": t} for t in texts]


class TestPerplexityConstruction:
    def test_explicit_device_is_kept(self):
        pf = PerplexityFilter(model_name="x", device="cpu", batch_size=3, max_length=16)
        assert (pf._device, pf.batch_size, pf.max_length, pf.model_name) == ("cpu", 3, 16, "x")
        assert pf._model is None and pf._tokenizer is None

    def test_autodetect_follows_torch_cuda(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert PerplexityFilter()._device == "cuda"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert PerplexityFilter()._device == "cpu"

    def test_autodetect_without_torch_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert PerplexityFilter()._device == "cpu"

    def test_load_requires_transformers(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(ImportError, match="transformers and torch required"):
            PerplexityFilter(device="cpu")._load_model()


class TestPerplexityScoring:
    def test_load_scores_and_context_manager_unload(self, tiny_ppl_dir, cpu_only):
        with PerplexityFilter(model_name=tiny_ppl_dir, device="cpu") as pf:
            ppl = pf.score_text(TEXTS[0])
            assert pf._model is not None and pf._tokenizer is not None
            assert 1.0 < ppl < 1e6
            assert pf.score_text(TEXTS[0]) == pytest.approx(ppl)  # deterministic in eval
            assert pf._model.training is False
        assert pf._model is None and pf._tokenizer is None

    def test_single_token_text_scores_infinite(self, tiny_ppl_dir, cpu_only):
        pf = PerplexityFilter(model_name=tiny_ppl_dir, device="cpu")
        assert pf.score_text("cat") == float("inf")
        pf.unload()

    def test_missing_pad_token_falls_back_to_eos(self, tmp_path, cpu_only):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        tk = Tokenizer(WordLevel({"<s>": 0, "</s>": 1, "<unk>": 2, "the": 3, "cat": 4}, unk_token="<unk>"))
        tk.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(
            tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        assert tok.pad_token is None
        tok.save_pretrained(tmp_path)
        tiny_gpt2(vocab=64).save_pretrained(tmp_path)
        pf = PerplexityFilter(model_name=str(tmp_path), device="cpu")
        pf._load_model()
        assert pf._tokenizer.pad_token == "</s>"
        pf._load_model()  # second call is a no-op (model already resident)
        pf.unload()

    def test_score_batches_and_flags_unscorable_text(self, tiny_ppl_dir, cpu_only, caplog):
        pf = PerplexityFilter(model_name=tiny_ppl_dir, device="cpu", batch_size=3)
        samples = [{"text": t} for t in TEXTS[:4]] + [{"text": ""}, {"text": "short"}, "a plain string sample here"]
        with caplog.at_level(logging.INFO, logger=LOGGER):
            scores = pf.score(samples)
        assert len(scores) == 7
        assert scores[4] is None and scores[5] is None  # blank and <10 chars
        assert all(isinstance(s, float) for i, s in enumerate(scores) if i not in (4, 5))
        progress = [r.getMessage() for r in caplog.records if "Perplexity scoring" in r.getMessage()]
        assert progress[-1].startswith("Perplexity scoring: 7/7") and len(progress) == 3
        quiet = pf.score(samples[:2], show_progress=False)
        assert len(quiet) == 2
        pf.unload()

    def test_score_batch_swallows_per_text_failures(self, tiny_ppl_dir, cpu_only, monkeypatch, caplog):
        pf = PerplexityFilter(model_name=tiny_ppl_dir, device="cpu")

        def flaky(text):
            if "boom" in text:
                raise RuntimeError("scoring blew up")
            return 12.5

        monkeypatch.setattr(pf, "score_text", flaky)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            out = pf._score_batch(["fine text here ok", "this will boom badly"])
        assert out == [12.5, None]
        assert any("Failed to score text: scoring blew up" in r.getMessage() for r in caplog.records)

    def test_inf_score_becomes_none(self, tiny_ppl_dir, cpu_only, monkeypatch):
        pf = PerplexityFilter(model_name=tiny_ppl_dir, device="cpu")
        monkeypatch.setattr(pf, "score_text", lambda t: float("inf"))
        assert pf._score_batch(["long enough text here"]) == [None]


class TestPerplexityUnload:
    def test_unload_without_load_is_noop_and_repeatable(self):
        pf = PerplexityFilter(model_name="x", device="cpu")
        pf.unload()
        pf.unload()
        assert pf._model is None

    def test_unload_tolerates_missing_attributes(self, tiny_ppl_dir, cpu_only):
        class Stubborn(PerplexityFilter):
            def __delattr__(self, name):
                raise RuntimeError(f"cannot delete {name}")

        pf = Stubborn(model_name=tiny_ppl_dir, device="cpu")
        pf._load_model()
        assert pf._model is not None
        pf.unload()  # the failed deletes are swallowed; references still dropped
        assert pf._model is None and pf._tokenizer is None

    def test_unload_empties_cuda_cache_when_available(self, monkeypatch):
        """Mocks: ``torch.cuda.is_available`` / ``empty_cache`` (GPU boundary)."""
        import torch

        emptied = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
        pf = PerplexityFilter(model_name="x", device="cpu")
        pf._model = object()
        pf._tokenizer = object()
        pf.unload()
        assert emptied == [1]

    def test_unload_survives_empty_cache_failure(self, monkeypatch, caplog):
        import torch

        def broken():
            raise RuntimeError("cuda gone")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", broken)
        pf = PerplexityFilter(model_name="x", device="cpu")
        pf._model = object()
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            pf.unload()
        assert pf._model is None
        assert any("empty_cache skipped: cuda gone" in r.getMessage() for r in caplog.records)

    def test_unload_without_torch_is_fine(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        pf = PerplexityFilter(model_name="x", device="cpu")
        pf._model = object()
        pf.unload()
        assert pf._model is None


class TestPerplexityFilterMethod:
    @pytest.fixture
    def scored(self, tiny_ppl_dir, cpu_only):
        pf = PerplexityFilter(model_name=tiny_ppl_dir, device="cpu", batch_size=4)
        scores = pf.score(_samples(), show_progress=False)
        assert all(s is not None for s in scores)
        yield pf, scores
        pf.unload()

    def test_inverted_percentiles_rejected(self):
        pf = PerplexityFilter(model_name="x", device="cpu")
        with pytest.raises(InvalidSettingError) as exc:
            pf.filter(_samples(), min_percentile=90, max_percentile=10)
        assert exc.value.code == "CONFIG_INVALID_SETTING"
        assert exc.value.setting_name == "min_percentile"

    def test_inverted_absolute_bounds_rejected(self):
        pf = PerplexityFilter(model_name="x", device="cpu")
        with pytest.raises(InvalidSettingError) as exc:
            pf.filter(_samples(), min_percentile=None, max_percentile=None,
                      min_perplexity=500.0, max_perplexity=5.0)
        assert exc.value.setting_name == "min_perplexity"

    def test_percentile_thresholds_match_sorted_scores(self, scored):
        pf, scores = scored
        kept, stats = pf.filter(_samples(), min_percentile=25, max_percentile=75, show_progress=False)
        ordered = sorted(scores)
        lo, hi = ordered[int(8 * 25 / 100)], ordered[int(8 * 75 / 100)]
        assert stats.threshold_low == lo and stats.threshold_high == hi
        expected = [s for s, sc in zip(_samples(), scores) if lo <= sc <= hi]
        assert kept == expected
        assert stats.total_samples == 8
        assert stats.retained_count == len(kept)
        assert stats.filtered_count == 8 - len(kept)
        assert stats.samples_scored == 8 and stats.samples_failed == 0
        assert stats.min_perplexity == min(scores) and stats.max_perplexity == max(scores)
        assert stats.mean_perplexity == pytest.approx(sum(scores) / 8)
        assert "Perplexity Filter Results" in stats.summary()

    def test_absolute_thresholds_override_percentiles(self, scored):
        pf, scores = scored
        ordered = sorted(scores)
        lo, hi = ordered[2], ordered[5]
        kept, stats = pf.filter(
            _samples(), min_perplexity=lo, max_perplexity=hi, show_progress=False
        )
        assert (stats.threshold_low, stats.threshold_high) == (lo, hi)
        assert kept == [s for s, sc in zip(_samples(), scores) if lo <= sc <= hi]

    def test_only_low_threshold_leaves_high_open(self, scored):
        pf, scores = scored
        lo = sorted(scores)[3]
        kept, stats = pf.filter(
            _samples(), min_percentile=None, max_percentile=None,
            min_perplexity=lo, show_progress=False,
        )
        assert stats.threshold_high is None
        assert kept == [s for s, sc in zip(_samples(), scores) if sc >= lo]

    def test_unscorable_rows_removed_or_kept_per_flag(self, scored):
        pf, _ = scored
        samples = _samples() + [{"text": ""}, {"text": "tiny"}]
        kept, stats = pf.filter(
            samples, min_percentile=None, max_percentile=None, show_progress=False
        )
        assert len(kept) == 8 and stats.samples_failed == 2 and stats.filtered_count == 2
        kept2, stats2 = pf.filter(
            samples, min_percentile=None, max_percentile=None,
            remove_failed=False, show_progress=False,
        )
        assert len(kept2) == 10 and stats2.filtered_count == 0 and stats2.samples_failed == 2

    def test_no_valid_scores_returns_input_unchanged(self, tiny_ppl_dir, cpu_only, caplog):
        pf = PerplexityFilter(model_name=tiny_ppl_dir, device="cpu")
        samples = [{"text": ""}, {"text": "tiny"}]
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            kept, stats = pf.filter(samples, show_progress=False)
        assert kept == samples
        assert stats.samples_scored == 0 and stats.samples_failed == 2
        assert stats.retained_count == 2 and stats.filtered_count == 0
        assert any("No valid perplexity scores" in r.getMessage() for r in caplog.records)
        pf.unload()

    def test_single_valid_score_has_zero_std(self, tiny_ppl_dir, cpu_only):
        pf = PerplexityFilter(model_name=tiny_ppl_dir, device="cpu")
        kept, stats = pf.filter(
            [{"text": TEXTS[0]}, {"text": ""}],
            min_percentile=None, max_percentile=None, show_progress=False,
        )
        assert len(kept) == 1 and stats.std_perplexity == 0.0
        pf.unload()

    def test_filter_by_threshold_with_precomputed_scores(self):
        pf = PerplexityFilter(model_name="x", device="cpu")
        samples = ["a", "b", "c", "d", "e"]
        scores = [5.0, None, 50.0, 500.0, 20.0]
        assert pf.filter_by_threshold(samples, scores, min_perplexity=10, max_perplexity=100) == ["c", "e"]
        assert pf.filter_by_threshold(samples, scores, remove_failed=False) == samples
        assert pf.filter_by_threshold(samples, scores) == ["a", "c", "d", "e"]
        assert pf.filter_by_threshold(samples, scores, max_perplexity=20) == ["a", "e"]


class TestPerplexityConvenience:
    def test_compute_perplexity_matches_filter_score(self, tiny_ppl_dir, cpu_only):
        direct = ds.compute_perplexity(TEXTS[1], model_name=tiny_ppl_dir, device="cpu")
        pf = PerplexityFilter(model_name=tiny_ppl_dir, device="cpu")
        assert direct == pytest.approx(pf.score_text(TEXTS[1]))
        pf.unload()

    def test_filter_by_perplexity_function(self, tiny_ppl_dir, cpu_only):
        kept, stats = ds.filter_by_perplexity(
            _samples(), model_name=tiny_ppl_dir, device="cpu", batch_size=2,
            min_percentile=None, max_percentile=90, show_progress=False,
        )
        assert stats.total_samples == 8
        assert stats.threshold_low is None and stats.threshold_high is not None
        assert all(isinstance(s, dict) for s in kept) and len(kept) >= 7

    def test_loader_filter_perplexity_returns_chatml_loader(self, tiny_ppl_dir, cpu_only):
        rows = [{"instruction": t, "output": t} for t in TEXTS]
        loader = DatasetLoader(rows)
        new_loader, stats = loader.filter_perplexity(
            model_name=tiny_ppl_dir, device="cpu", batch_size=4,
            min_percentile=None, max_percentile=None, show_progress=False,
        )
        assert len(new_loader) == 8 and stats.retained_count == 8
        assert new_loader.detected_format == DatasetFormat.CHATML
        assert new_loader[0]["text"].startswith("<|im_start|>user\nthe cat sat")

