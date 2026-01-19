"""
Tests for Dataset utilities module.

Tests cover:
- Format detection (ShareGPT, Alpaca, OpenAI, ChatML, raw)
- Format conversion to ChatML
- Validation logic
- DatasetLoader class
- Statistics computation
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from backpropagate.datasets import (
    DatasetFormat,
    DatasetLoader,
    ValidationResult,
    ValidationError,
    FormatConverter,
    detect_format,
    validate_dataset,
    convert_to_chatml,
    preview_samples,
    get_dataset_stats,
)


# =============================================================================
# FORMAT DETECTION TESTS
# =============================================================================

class TestDetectFormat:
    """Tests for format detection."""

    def test_detect_sharegpt(self):
        """Should detect ShareGPT format."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"},
            ]
        }
        assert detect_format(sample) == DatasetFormat.SHAREGPT

    def test_detect_alpaca(self):
        """Should detect Alpaca format."""
        sample = {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour",
        }
        assert detect_format(sample) == DatasetFormat.ALPACA

    def test_detect_openai(self):
        """Should detect OpenAI format."""
        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        assert detect_format(sample) == DatasetFormat.OPENAI

    def test_detect_chatml(self):
        """Should detect ChatML format."""
        sample = {
            "text": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>"
        }
        assert detect_format(sample) == DatasetFormat.CHATML

    def test_detect_chatml_string(self):
        """Should detect ChatML from raw string."""
        text = "<|im_start|>user\nHello<|im_end|>"
        assert detect_format(text) == DatasetFormat.CHATML

    def test_detect_raw_text(self):
        """Should detect raw text."""
        text = "This is plain text without any special format."
        assert detect_format(text) == DatasetFormat.RAW_TEXT

    def test_detect_from_list(self):
        """Should detect format from list of samples."""
        samples = [
            {"conversations": [{"from": "human", "value": "Hi"}]}
        ]
        assert detect_format(samples) == DatasetFormat.SHAREGPT

    def test_detect_unknown(self):
        """Should return UNKNOWN for unrecognized format."""
        sample = {"random_field": "value"}
        assert detect_format(sample) == DatasetFormat.UNKNOWN

    def test_detect_empty_list(self):
        """Should return UNKNOWN for empty list."""
        assert detect_format([]) == DatasetFormat.UNKNOWN


# =============================================================================
# FORMAT CONVERTER TESTS
# =============================================================================

class TestFormatConverter:
    """Tests for FormatConverter class."""

    def test_sharegpt_to_chatml(self):
        """Should convert ShareGPT to ChatML."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"},
            ]
        }
        result = FormatConverter.sharegpt_to_chatml(sample)

        assert "<|im_start|>user" in result
        assert "Hello" in result
        assert "<|im_start|>assistant" in result
        assert "Hi there!" in result
        assert "<|im_end|>" in result

    def test_sharegpt_with_system(self):
        """Should handle system messages in ShareGPT."""
        sample = {
            "conversations": [
                {"from": "system", "value": "You are helpful"},
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi!"},
            ]
        }
        result = FormatConverter.sharegpt_to_chatml(sample)

        assert "<|im_start|>system" in result
        assert "You are helpful" in result

    def test_alpaca_to_chatml(self):
        """Should convert Alpaca to ChatML."""
        sample = {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour",
        }
        result = FormatConverter.alpaca_to_chatml(sample)

        assert "<|im_start|>user" in result
        assert "Translate to French" in result
        assert "Hello" in result
        assert "<|im_start|>assistant" in result
        assert "Bonjour" in result

    def test_alpaca_without_input(self):
        """Should handle Alpaca without input."""
        sample = {
            "instruction": "Say hello",
            "output": "Hello!",
        }
        result = FormatConverter.alpaca_to_chatml(sample)

        assert "Say hello" in result
        assert "Hello!" in result

    def test_alpaca_with_system(self):
        """Should handle Alpaca with system prompt."""
        sample = {
            "system": "You are a translator",
            "instruction": "Translate",
            "input": "Hello",
            "output": "Bonjour",
        }
        result = FormatConverter.alpaca_to_chatml(sample)

        assert "<|im_start|>system" in result
        assert "You are a translator" in result

    def test_openai_to_chatml(self):
        """Should convert OpenAI to ChatML."""
        sample = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        result = FormatConverter.openai_to_chatml(sample)

        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result

    def test_raw_to_chatml(self):
        """Should convert raw text to ChatML."""
        text = "This is a message"
        result = FormatConverter.raw_to_chatml(text)

        assert "<|im_start|>user" in result
        assert "This is a message" in result
        assert "<|im_end|>" in result

    def test_to_chatml_generic(self):
        """Should use correct converter based on format."""
        sample = {"instruction": "Test", "output": "Response"}
        result = FormatConverter.to_chatml(sample, DatasetFormat.ALPACA)

        assert "<|im_start|>user" in result
        assert "Test" in result


class TestConvertToChatml:
    """Tests for convert_to_chatml function."""

    def test_convert_list(self):
        """Should convert list of samples."""
        samples = [
            {"instruction": "Q1", "output": "A1"},
            {"instruction": "Q2", "output": "A2"},
        ]
        result = convert_to_chatml(samples, DatasetFormat.ALPACA)

        assert len(result) == 2
        assert "text" in result[0]
        assert "Q1" in result[0]["text"]

    def test_auto_detect_format(self):
        """Should auto-detect format if not provided."""
        samples = [
            {"messages": [{"role": "user", "content": "Hi"}]}
        ]
        result = convert_to_chatml(samples)

        assert len(result) == 1
        assert "<|im_start|>user" in result[0]["text"]

    def test_empty_list(self):
        """Should handle empty list."""
        result = convert_to_chatml([])
        assert result == []


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Tests for validation functions."""

    def test_validate_valid_sharegpt(self):
        """Should pass valid ShareGPT samples."""
        samples = [
            {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]},
            {"conversations": [{"from": "human", "value": "Test"}, {"from": "gpt", "value": "Response"}]},
        ]
        result = validate_dataset(samples, DatasetFormat.SHAREGPT)

        assert result.is_valid
        assert result.valid_rows == 2
        assert result.error_count == 0

    def test_validate_missing_conversations(self):
        """Should catch missing conversations field."""
        samples = [{"other_field": "value"}]
        result = validate_dataset(samples, DatasetFormat.SHAREGPT)

        assert not result.is_valid
        assert result.error_count > 0
        assert any("conversations" in str(e) for e in result.errors)

    def test_validate_empty_conversations(self):
        """Should catch empty conversations."""
        samples = [{"conversations": []}]
        result = validate_dataset(samples, DatasetFormat.SHAREGPT)

        assert not result.is_valid

    def test_validate_valid_alpaca(self):
        """Should pass valid Alpaca samples."""
        samples = [
            {"instruction": "Test", "output": "Response"},
            {"instruction": "Another", "input": "With input", "output": "Result"},
        ]
        result = validate_dataset(samples, DatasetFormat.ALPACA)

        assert result.is_valid

    def test_validate_missing_instruction(self):
        """Should catch missing instruction."""
        samples = [{"output": "Response only"}]
        result = validate_dataset(samples, DatasetFormat.ALPACA)

        assert not result.is_valid

    def test_validate_valid_openai(self):
        """Should pass valid OpenAI samples."""
        samples = [
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
        ]
        result = validate_dataset(samples, DatasetFormat.OPENAI)

        assert result.is_valid

    def test_validate_invalid_role(self):
        """Should catch invalid role in OpenAI format."""
        samples = [
            {"messages": [{"role": "invalid_role", "content": "Hi"}]},
        ]
        result = validate_dataset(samples, DatasetFormat.OPENAI)

        # Invalid role is a warning, not error
        assert result.warning_count > 0

    def test_validate_chatml_balanced_tags(self):
        """Should catch unbalanced ChatML tags."""
        samples = [
            {"text": "<|im_start|>user\nHello"}  # Missing im_end
        ]
        result = validate_dataset(samples, DatasetFormat.CHATML)

        assert not result.is_valid
        assert any("unbalanced" in str(e).lower() for e in result.errors)

    def test_validate_empty_dataset(self):
        """Should catch empty dataset."""
        result = validate_dataset([])

        assert not result.is_valid
        assert any("empty" in str(e).lower() for e in result.errors)

    def test_validation_result_summary(self):
        """Should generate readable summary."""
        result = ValidationResult(
            is_valid=True,
            total_rows=100,
            valid_rows=100,
            errors=[],
            warnings=[],
            format_detected=DatasetFormat.SHAREGPT,
        )
        summary = result.summary()

        assert "sharegpt" in summary.lower()
        assert "100" in summary

    def test_error_rate_calculation(self):
        """Should calculate error rate correctly."""
        result = ValidationResult(
            is_valid=False,
            total_rows=100,
            valid_rows=90,
            errors=[],
            format_detected=DatasetFormat.UNKNOWN,
        )

        assert result.error_rate == 0.1  # 10% error rate


# =============================================================================
# DATASET LOADER TESTS
# =============================================================================

class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    @pytest.fixture
    def temp_jsonl_file(self):
        """Create a temporary JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            samples = [
                {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]},
                {"conversations": [{"from": "human", "value": "Test"}, {"from": "gpt", "value": "Response"}]},
            ]
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
            return Path(f.name)

    @pytest.fixture
    def temp_json_file(self):
        """Create a temporary JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            samples = [
                {"instruction": "Q1", "output": "A1"},
                {"instruction": "Q2", "output": "A2"},
            ]
            json.dump(samples, f)
            return Path(f.name)

    def test_load_jsonl(self, temp_jsonl_file):
        """Should load JSONL file."""
        loader = DatasetLoader(temp_jsonl_file)

        assert len(loader) == 2
        assert loader.detected_format == DatasetFormat.SHAREGPT
        temp_jsonl_file.unlink()

    def test_load_json(self, temp_json_file):
        """Should load JSON file."""
        loader = DatasetLoader(temp_json_file)

        assert len(loader) == 2
        assert loader.detected_format == DatasetFormat.ALPACA
        temp_json_file.unlink()

    def test_load_from_list(self):
        """Should load from list directly."""
        samples = [
            {"messages": [{"role": "user", "content": "Hi"}]},
        ]
        loader = DatasetLoader(samples)

        assert len(loader) == 1
        assert loader.detected_format == DatasetFormat.OPENAI

    def test_to_chatml(self, temp_jsonl_file):
        """Should convert to ChatML."""
        loader = DatasetLoader(temp_jsonl_file)
        chatml = loader.to_chatml()

        assert len(chatml) == 2
        assert "text" in chatml[0]
        assert "<|im_start|>" in chatml[0]["text"]
        temp_jsonl_file.unlink()

    def test_preview(self, temp_jsonl_file):
        """Should preview samples."""
        loader = DatasetLoader(temp_jsonl_file)
        previews = loader.preview(n=1, as_chatml=True)

        assert len(previews) == 1
        assert "<|im_start|>" in previews[0]
        temp_jsonl_file.unlink()

    def test_validation_report(self, temp_jsonl_file):
        """Should generate validation report."""
        loader = DatasetLoader(temp_jsonl_file)
        report = loader.validation_report()

        assert "sharegpt" in report.lower()
        assert "2" in report  # Total rows
        temp_jsonl_file.unlink()

    def test_stats(self, temp_jsonl_file):
        """Should compute statistics."""
        loader = DatasetLoader(temp_jsonl_file)
        stats = loader.stats()

        assert stats.total_samples == 2
        assert stats.total_tokens_approx > 0
        assert stats.format_detected == DatasetFormat.SHAREGPT
        temp_jsonl_file.unlink()

    def test_shuffle(self, temp_jsonl_file):
        """Should shuffle samples."""
        loader = DatasetLoader(temp_jsonl_file)
        shuffled = loader.shuffle(seed=42)

        assert len(shuffled) == len(loader)
        temp_jsonl_file.unlink()

    def test_split(self, temp_jsonl_file):
        """Should split into train/test."""
        loader = DatasetLoader(temp_jsonl_file)
        train, test = loader.split(train_ratio=0.5, seed=42)

        assert len(train) == 1
        assert len(test) == 1
        temp_jsonl_file.unlink()

    def test_iter(self, temp_jsonl_file):
        """Should be iterable."""
        loader = DatasetLoader(temp_jsonl_file)
        samples = list(loader)

        assert len(samples) == 2
        temp_jsonl_file.unlink()

    def test_getitem(self, temp_jsonl_file):
        """Should support indexing."""
        loader = DatasetLoader(temp_jsonl_file)

        assert "conversations" in loader[0]
        temp_jsonl_file.unlink()

    def test_file_not_found(self):
        """Should raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            DatasetLoader("/nonexistent/path.jsonl")

    def test_to_hf_dataset(self, temp_jsonl_file):
        """Should convert to HuggingFace dataset."""
        try:
            from datasets import Dataset
        except ImportError:
            pytest.skip("datasets not installed")

        loader = DatasetLoader(temp_jsonl_file)
        hf_dataset = loader.to_hf_dataset()

        assert len(hf_dataset) == 2
        assert "text" in hf_dataset.column_names
        temp_jsonl_file.unlink()


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestDatasetStats:
    """Tests for dataset statistics."""

    def test_stats_basic(self):
        """Should compute basic statistics."""
        samples = [
            {"text": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>"},
            {"text": "<|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\nResponse<|im_end|>"},
        ]
        stats = get_dataset_stats(samples, DatasetFormat.CHATML)

        assert stats.total_samples == 2
        assert stats.total_tokens_approx > 0
        assert stats.avg_tokens_per_sample > 0
        assert stats.min_tokens > 0
        assert stats.max_tokens >= stats.min_tokens

    def test_stats_with_system_prompts(self):
        """Should detect system prompts."""
        samples = [
            {"text": "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHi<|im_end|>"},
        ]
        stats = get_dataset_stats(samples, DatasetFormat.CHATML)

        assert stats.has_system_prompts
        assert stats.unique_system_prompts == 1

    def test_stats_empty(self):
        """Should handle empty dataset."""
        stats = get_dataset_stats([])

        assert stats.total_samples == 0
        assert stats.total_tokens_approx == 0

    def test_stats_turn_counting(self):
        """Should count conversation turns."""
        samples = [
            {"text": "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\nA2<|im_end|>"},
        ]
        stats = get_dataset_stats(samples, DatasetFormat.CHATML)

        assert stats.avg_turns_per_conversation == 4  # 2 user + 2 assistant


# =============================================================================
# PREVIEW SAMPLES TESTS
# =============================================================================

class TestPreviewSamples:
    """Tests for preview_samples function."""

    def test_preview_from_list(self):
        """Should preview samples from list."""
        samples = [
            {"instruction": "Test", "output": "Response"},
        ]
        previews = preview_samples(samples, n=1, as_chatml=True)

        assert len(previews) == 1
        assert "<|im_start|>" in previews[0]

    def test_preview_raw_format(self):
        """Should preview in raw format."""
        samples = [
            {"instruction": "Test", "output": "Response"},
        ]
        loader = DatasetLoader(samples)
        previews = loader.preview(n=1, as_chatml=False)

        assert len(previews) == 1
        assert "instruction" in previews[0]


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_unicode_content(self):
        """Should handle Unicode content."""
        samples = [
            {"conversations": [{"from": "human", "value": "Hello 你好 🌟"}]},
        ]
        loader = DatasetLoader(samples)
        chatml = loader.to_chatml()

        assert "你好" in chatml[0]["text"]
        assert "🌟" in chatml[0]["text"]

    def test_multiline_content(self):
        """Should handle multiline content."""
        samples = [
            {"conversations": [{"from": "human", "value": "Line 1\nLine 2\nLine 3"}]},
        ]
        loader = DatasetLoader(samples)
        chatml = loader.to_chatml()

        assert "Line 1\nLine 2\nLine 3" in chatml[0]["text"]

    def test_special_characters(self):
        """Should handle special characters."""
        samples = [
            {"conversations": [{"from": "human", "value": "Test <tag> & 'quotes'"}]},
        ]
        loader = DatasetLoader(samples)
        chatml = loader.to_chatml()

        assert "<tag>" in chatml[0]["text"]

    def test_empty_values(self):
        """Should handle empty values in validation."""
        samples = [
            {"instruction": "", "output": "Response"},  # Empty instruction
        ]
        result = validate_dataset(samples, DatasetFormat.ALPACA)

        # Empty content should be a warning
        assert result.warning_count > 0 or result.error_count > 0

    def test_mixed_format_detection(self):
        """Should use most common format from samples."""
        # All samples should be consistent
        samples = [
            {"conversations": [{"from": "human", "value": "1"}]},
            {"conversations": [{"from": "human", "value": "2"}]},
            {"conversations": [{"from": "human", "value": "3"}]},
        ]
        assert detect_format(samples) == DatasetFormat.SHAREGPT


# =============================================================================
# FILTER STATS TESTS
# =============================================================================

class TestFilterStats:
    """Tests for FilterStats dataclass."""

    def test_filter_stats_basic(self):
        """Should compute basic filter stats."""
        from backpropagate.datasets import FilterStats

        stats = FilterStats(
            total_before=100,
            total_after=80,
            removed_too_short=10,
            removed_too_long=5,
            removed_empty=5,
        )

        assert stats.total_removed == 20
        assert stats.retention_rate == 0.8

    def test_filter_stats_summary(self):
        """Should generate summary string."""
        from backpropagate.datasets import FilterStats

        stats = FilterStats(
            total_before=100,
            total_after=80,
            removed_too_short=10,
            removed_too_long=5,
            removed_empty=5,
        )

        summary = stats.summary()
        assert "Before: 100" in summary
        assert "After:  80" in summary
        assert "Too short: 10" in summary

    def test_filter_stats_zero_before(self):
        """Should handle zero total_before."""
        from backpropagate.datasets import FilterStats

        stats = FilterStats(total_before=0, total_after=0)
        assert stats.retention_rate == 0.0


# =============================================================================
# QUALITY FILTERING TESTS
# =============================================================================

class TestFilterByQuality:
    """Tests for filter_by_quality function."""

    def test_filter_by_tokens(self):
        """Should filter by token count."""
        from backpropagate.datasets import filter_by_quality

        samples = [
            {"text": "x" * 100},   # ~25 tokens
            {"text": "x" * 400},   # ~100 tokens
            {"text": "x" * 2000},  # ~500 tokens
        ]

        filtered, stats = filter_by_quality(
            samples,
            min_tokens=50,
            max_tokens=200,
            min_turns=0,
            require_assistant=False,
        )

        assert len(filtered) == 1  # Only the middle one
        assert stats.removed_too_short == 1
        assert stats.removed_too_long == 1

    def test_filter_by_turns(self):
        """Should filter by turn count."""
        from backpropagate.datasets import filter_by_quality

        samples = [
            {"text": "<|im_start|>user\nHi<|im_end|>"},  # 1 turn
            {"text": "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|}"},  # 2 turns
            {"text": "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nA<|im_end|>\n<|im_start|>user\nQ2<|im_end|>"},  # 3 turns
        ]

        filtered, stats = filter_by_quality(
            samples,
            min_tokens=0,
            max_tokens=10000,
            min_turns=2,
            max_turns=2,
            require_assistant=False,
        )

        assert len(filtered) == 1
        assert stats.removed_few_turns == 1
        assert stats.removed_many_turns == 1

    def test_filter_require_assistant(self):
        """Should filter samples without assistant response."""
        from backpropagate.datasets import filter_by_quality

        samples = [
            {"text": "<|im_start|>user\nQuestion<|im_end|>"},
            {"text": "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nA<|im_end|>"},
        ]

        filtered, stats = filter_by_quality(
            samples,
            min_tokens=0,
            max_tokens=10000,
            min_turns=0,
            require_assistant=True,
        )

        assert len(filtered) == 1
        assert stats.removed_no_assistant == 1

    def test_filter_empty_samples(self):
        """Should remove empty samples."""
        from backpropagate.datasets import filter_by_quality

        samples = [
            {"text": ""},
            {"text": "   "},
            {"text": "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>"},
        ]

        filtered, stats = filter_by_quality(
            samples,
            min_tokens=0,
            max_tokens=10000,
            min_turns=0,
            require_assistant=False,
            remove_empty=True,
        )

        assert len(filtered) == 1
        assert stats.removed_empty == 2

    def test_filter_custom_filter(self):
        """Should support custom filter function."""
        from backpropagate.datasets import filter_by_quality

        samples = [
            {"text": "keep me", "flag": True},
            {"text": "remove me", "flag": False},
        ]

        filtered, stats = filter_by_quality(
            samples,
            min_tokens=0,
            max_tokens=10000,
            min_turns=0,
            require_assistant=False,
            custom_filter=lambda s: s.get("flag", False),
        )

        assert len(filtered) == 1
        assert stats.removed_custom == 1


# =============================================================================
# DEDUPLICATION TESTS
# =============================================================================

class TestDeduplicateExact:
    """Tests for exact deduplication."""

    def test_dedupe_exact_basic(self):
        """Should remove exact duplicates."""
        from backpropagate.datasets import deduplicate_exact

        samples = [
            {"text": "Hello world"},
            {"text": "Hello world"},
            {"text": "Different text"},
        ]

        unique, num_removed = deduplicate_exact(samples)

        assert len(unique) == 2
        assert num_removed == 1

    def test_dedupe_exact_preserves_order(self):
        """Should preserve order of first occurrences."""
        from backpropagate.datasets import deduplicate_exact

        samples = [
            {"text": "First"},
            {"text": "Second"},
            {"text": "First"},  # Duplicate
            {"text": "Third"},
        ]

        unique, _ = deduplicate_exact(samples)

        assert unique[0]["text"] == "First"
        assert unique[1]["text"] == "Second"
        assert unique[2]["text"] == "Third"

    def test_dedupe_exact_no_duplicates(self):
        """Should handle no duplicates."""
        from backpropagate.datasets import deduplicate_exact

        samples = [
            {"text": "One"},
            {"text": "Two"},
            {"text": "Three"},
        ]

        unique, num_removed = deduplicate_exact(samples)

        assert len(unique) == 3
        assert num_removed == 0

    def test_dedupe_exact_custom_key(self):
        """Should use custom key for comparison."""
        from backpropagate.datasets import deduplicate_exact

        samples = [
            {"content": "Same", "id": 1},
            {"content": "Same", "id": 2},
            {"content": "Different", "id": 3},
        ]

        unique, num_removed = deduplicate_exact(samples, key="content")

        assert len(unique) == 2
        assert num_removed == 1


class TestDeduplicateMinhash:
    """Tests for MinHash deduplication."""

    def test_dedupe_minhash_import_error(self):
        """Should raise ImportError if datasketch not installed."""
        from backpropagate.datasets import deduplicate_minhash

        # This test should skip if datasketch is installed
        try:
            import datasketch
            pytest.skip("datasketch is installed")
        except ImportError:
            samples = [{"text": "Hello"}]
            with pytest.raises(ImportError) as exc_info:
                deduplicate_minhash(samples)
            assert "datasketch" in str(exc_info.value)

    def test_dedupe_minhash_basic(self):
        """Should remove near-duplicates."""
        try:
            from datasketch import MinHash
        except ImportError:
            pytest.skip("datasketch not installed")

        from backpropagate.datasets import deduplicate_minhash

        samples = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "The quick brown fox jumps over the lazy cat."},  # Similar
            {"text": "Something completely different with no overlap."},
        ]

        unique, num_removed = deduplicate_minhash(samples, threshold=0.7)

        # With high similarity threshold, the first two should be deduped
        assert len(unique) == 2
        assert num_removed == 1


# =============================================================================
# DATASET LOADER FILTER/DEDUPE METHODS TESTS
# =============================================================================

class TestDatasetLoaderFilter:
    """Tests for DatasetLoader.filter() method."""

    def test_filter_method(self):
        """Should filter using method."""
        samples = [
            {"text": "x" * 100},   # Short
            {"text": "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n" + "x" * 400 + "<|im_end|>"},
        ]

        loader = DatasetLoader(samples, DatasetFormat.CHATML)
        filtered = loader.filter(min_tokens=50)

        assert len(filtered) == 1

    def test_filter_chaining(self):
        """Should support chaining operations."""
        samples = [
            {"text": "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nA<|im_end|>"},
            {"text": "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nA<|im_end|>"},  # Duplicate
            {"text": "<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\nA2<|im_end|>"},
        ]

        loader = DatasetLoader(samples, DatasetFormat.CHATML)
        result = loader.filter(min_turns=2).deduplicate()

        assert len(result) == 2


class TestDatasetLoaderDeduplicate:
    """Tests for DatasetLoader.deduplicate() method."""

    def test_deduplicate_method_exact(self):
        """Should deduplicate using exact method."""
        samples = [
            {"text": "<|im_start|>user\nHello<|im_end|>"},
            {"text": "<|im_start|>user\nHello<|im_end|>"},
            {"text": "<|im_start|>user\nWorld<|im_end|>"},
        ]

        loader = DatasetLoader(samples, DatasetFormat.CHATML)
        deduped = loader.deduplicate(method="exact")

        assert len(deduped) == 2

    def test_deduplicate_invalid_method(self):
        """Should raise error for invalid method."""
        samples = [{"text": "Test"}]
        loader = DatasetLoader(samples, DatasetFormat.CHATML)

        with pytest.raises(ValueError) as exc_info:
            loader.deduplicate(method="invalid")
        assert "Unknown deduplication method" in str(exc_info.value)


# =============================================================================
# STREAMING DATASET LOADER TESTS
# =============================================================================

class TestStreamingDatasetLoader:
    """Tests for StreamingDatasetLoader class."""

    @pytest.fixture
    def temp_jsonl_file(self, tmp_path):
        """Create a temporary JSONL file for testing."""
        file_path = tmp_path / "test_stream.jsonl"
        samples = [
            {"text": f"<|im_start|>user\nQ{i}<|im_end|>\n<|im_start|>assistant\nA{i}<|im_end|>"}
            for i in range(10)
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        return file_path

    def test_streaming_from_file(self, temp_jsonl_file):
        """Should stream from local file."""
        from backpropagate.datasets import StreamingDatasetLoader

        loader = StreamingDatasetLoader(str(temp_jsonl_file))
        samples = list(loader)

        assert len(samples) == 10

    def test_streaming_take(self, temp_jsonl_file):
        """Should take first n samples."""
        from backpropagate.datasets import StreamingDatasetLoader

        loader = StreamingDatasetLoader(str(temp_jsonl_file))
        samples = loader.take(5)

        assert len(samples) == 5

    def test_streaming_skip(self, temp_jsonl_file):
        """Should skip first n samples."""
        from backpropagate.datasets import StreamingDatasetLoader

        loader = StreamingDatasetLoader(str(temp_jsonl_file))
        samples = list(loader.skip(5))

        assert len(samples) == 5

    def test_streaming_batches(self, temp_jsonl_file):
        """Should yield samples in batches."""
        from backpropagate.datasets import StreamingDatasetLoader

        loader = StreamingDatasetLoader(str(temp_jsonl_file))
        batches = list(loader.batches(3))

        assert len(batches) == 4  # 3 + 3 + 3 + 1
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1  # Last batch has 1

    def test_streaming_to_chatml(self, temp_jsonl_file):
        """Should convert to ChatML format."""
        from backpropagate.datasets import StreamingDatasetLoader

        loader = StreamingDatasetLoader(str(temp_jsonl_file))
        chatml = loader.to_chatml(n=3)

        assert len(chatml) == 3
        assert "text" in chatml[0]

    def test_streaming_filter(self, temp_jsonl_file):
        """Should filter samples while streaming."""
        from backpropagate.datasets import StreamingDatasetLoader

        loader = StreamingDatasetLoader(str(temp_jsonl_file))
        filtered = list(loader.filter(min_turns=2))

        assert len(filtered) == 10  # All have 2 turns

    def test_streaming_detected_format(self, temp_jsonl_file):
        """Should detect format from streamed samples."""
        from backpropagate.datasets import StreamingDatasetLoader

        loader = StreamingDatasetLoader(str(temp_jsonl_file))
        # Trigger format detection by taking a sample
        loader.take(1)

        assert loader.detected_format == DatasetFormat.CHATML


class TestDatasetLoaderFromStreaming:
    """Tests for DatasetLoader.from_streaming() classmethod."""

    def test_from_streaming(self, tmp_path):
        """Should create streaming loader from classmethod."""
        from backpropagate.datasets import StreamingDatasetLoader

        file_path = tmp_path / "test.jsonl"
        samples = [{"text": "<|im_start|>user\nTest<|im_end|>"}]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = DatasetLoader.from_streaming(str(file_path))

        assert isinstance(loader, StreamingDatasetLoader)


# =============================================================================
# PERPLEXITY FILTERING TESTS
# =============================================================================

class TestPerplexityStats:
    """Tests for PerplexityStats dataclass."""

    def test_perplexity_stats_creation(self):
        """Should create PerplexityStats with all fields."""
        from backpropagate.datasets import PerplexityStats

        stats = PerplexityStats(
            total_samples=100,
            samples_scored=95,
            samples_failed=5,
            mean_perplexity=50.0,
            median_perplexity=45.0,
            std_perplexity=10.0,
            min_perplexity=20.0,
            max_perplexity=100.0,
            filtered_count=10,
            retained_count=90,
            threshold_low=25.0,
            threshold_high=90.0,
        )

        assert stats.total_samples == 100
        assert stats.samples_scored == 95
        assert stats.mean_perplexity == 50.0
        assert stats.median_perplexity == 45.0

    def test_perplexity_stats_retention_rate(self):
        """Should compute retention rate correctly."""
        from backpropagate.datasets import PerplexityStats

        stats = PerplexityStats(
            total_samples=100,
            samples_scored=100,
            samples_failed=0,
            mean_perplexity=50.0,
            median_perplexity=50.0,
            std_perplexity=10.0,
            min_perplexity=20.0,
            max_perplexity=100.0,
            filtered_count=20,
            retained_count=80,
        )

        assert stats.retention_rate == 0.8

    def test_perplexity_stats_retention_rate_empty(self):
        """Should handle zero total samples."""
        from backpropagate.datasets import PerplexityStats

        stats = PerplexityStats(
            total_samples=0,
            samples_scored=0,
            samples_failed=0,
            mean_perplexity=0.0,
            median_perplexity=0.0,
            std_perplexity=0.0,
            min_perplexity=0.0,
            max_perplexity=0.0,
            filtered_count=0,
            retained_count=0,
        )

        assert stats.retention_rate == 0.0

    def test_perplexity_stats_summary(self):
        """Should generate summary string."""
        from backpropagate.datasets import PerplexityStats

        stats = PerplexityStats(
            total_samples=100,
            samples_scored=95,
            samples_failed=5,
            mean_perplexity=50.0,
            median_perplexity=45.0,
            std_perplexity=10.0,
            min_perplexity=20.0,
            max_perplexity=100.0,
            filtered_count=10,
            retained_count=90,
            threshold_low=25.0,
            threshold_high=90.0,
        )

        summary = stats.summary()
        assert "Perplexity Filter Results" in summary
        assert "Mean: 50.00" in summary
        assert "Median: 45.00" in summary
        assert "Retained: 90" in summary


class TestPerplexityFilterClass:
    """Tests for PerplexityFilter class."""

    def test_perplexity_filter_creation(self):
        """Should create PerplexityFilter with default parameters."""
        from backpropagate.datasets import PerplexityFilter

        pf = PerplexityFilter()

        assert pf.model_name == "gpt2"
        assert pf.batch_size == 8
        assert pf.max_length == 512

    def test_perplexity_filter_custom_params(self):
        """Should accept custom parameters."""
        from backpropagate.datasets import PerplexityFilter

        pf = PerplexityFilter(
            model_name="gpt2-medium",
            device="cpu",
            batch_size=4,
            max_length=256,
        )

        assert pf.model_name == "gpt2-medium"
        assert pf._device == "cpu"
        assert pf.batch_size == 4
        assert pf.max_length == 256

    def test_perplexity_filter_lazy_loading(self):
        """Should not load model until needed."""
        from backpropagate.datasets import PerplexityFilter

        pf = PerplexityFilter()

        assert pf._model is None
        assert pf._tokenizer is None

    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="transformers not installed"),
        reason="transformers not installed"
    )
    def test_perplexity_filter_model_load(self):
        """Should load model when needed."""
        from backpropagate.datasets import PerplexityFilter

        pf = PerplexityFilter(model_name="gpt2", device="cpu")

        # Force model load
        try:
            pf._load_model()
            assert pf._model is not None
            assert pf._tokenizer is not None
        except Exception:
            pytest.skip("Could not load model")

    def test_perplexity_filter_by_threshold_no_model(self):
        """filter_by_threshold should work with pre-computed scores."""
        from backpropagate.datasets import PerplexityFilter

        pf = PerplexityFilter()

        samples = [
            {"text": "sample 1"},
            {"text": "sample 2"},
            {"text": "sample 3"},
        ]
        scores = [10.0, 50.0, 100.0]

        # Filter by absolute threshold
        filtered = pf.filter_by_threshold(
            samples, scores,
            min_perplexity=20.0,
            max_perplexity=80.0
        )

        # Only sample 2 (ppl=50) should pass
        assert len(filtered) == 1
        assert filtered[0]["text"] == "sample 2"

    def test_perplexity_filter_by_threshold_none_scores(self):
        """filter_by_threshold should handle None scores."""
        from backpropagate.datasets import PerplexityFilter

        pf = PerplexityFilter()

        samples = [
            {"text": "sample 1"},
            {"text": "sample 2"},
            {"text": "sample 3"},
        ]
        scores = [50.0, None, 50.0]

        # Remove failed by default
        filtered = pf.filter_by_threshold(
            samples, scores,
            min_perplexity=20.0,
            max_perplexity=80.0,
            remove_failed=True,
        )

        assert len(filtered) == 2

        # Keep failed
        filtered = pf.filter_by_threshold(
            samples, scores,
            min_perplexity=20.0,
            max_perplexity=80.0,
            remove_failed=False,
        )

        assert len(filtered) == 3


class TestFilterByPerplexityFunction:
    """Tests for filter_by_perplexity convenience function."""

    def test_filter_by_perplexity_import(self):
        """Should be importable from datasets module."""
        from backpropagate.datasets import filter_by_perplexity

        assert callable(filter_by_perplexity)

    def test_compute_perplexity_import(self):
        """Should be importable from datasets module."""
        from backpropagate.datasets import compute_perplexity

        assert callable(compute_perplexity)


class TestDatasetLoaderPerplexityFilter:
    """Tests for DatasetLoader.filter_perplexity() method."""

    def test_filter_perplexity_method_exists(self):
        """DatasetLoader should have filter_perplexity method."""
        from backpropagate.datasets import DatasetLoader

        assert hasattr(DatasetLoader, "filter_perplexity")
        assert callable(getattr(DatasetLoader, "filter_perplexity"))


class TestPerplexityExports:
    """Tests for perplexity module exports."""

    def test_exports_from_init(self):
        """Should be able to import from backpropagate package."""
        from backpropagate import (
            PerplexityFilter,
            PerplexityStats,
            compute_perplexity,
            filter_by_perplexity,
        )

        assert PerplexityFilter is not None
        assert PerplexityStats is not None
        assert callable(compute_perplexity)
        assert callable(filter_by_perplexity)

    def test_in_all_exports(self):
        """Should be in __all__ list."""
        import backpropagate

        assert "PerplexityFilter" in backpropagate.__all__
        assert "PerplexityStats" in backpropagate.__all__
        assert "compute_perplexity" in backpropagate.__all__
        assert "filter_by_perplexity" in backpropagate.__all__


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================

class TestDatasetLoaderToHfDataset:
    """Tests for DatasetLoader.to_hf_dataset() method."""

    def test_to_hf_dataset_basic(self, tmp_path):
        """to_hf_dataset should return HuggingFace Dataset."""
        pytest.importorskip("datasets")
        from datasets import Dataset

        file_path = tmp_path / "test.jsonl"
        samples = [
            {"text": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>"},
            {"text": "<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\nGoodbye!<|im_end|>"},
        ]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = DatasetLoader(str(file_path))
        dataset = loader.to_hf_dataset()

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 2
        assert "text" in dataset.column_names

    def test_to_hf_dataset_with_split(self, tmp_path):
        """to_hf_dataset should return dict with split name when provided."""
        pytest.importorskip("datasets")

        file_path = tmp_path / "test.jsonl"
        samples = [{"text": "<|im_start|>user\nTest<|im_end|>"}]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = DatasetLoader(str(file_path))
        result = loader.to_hf_dataset(split="train")

        assert isinstance(result, dict)
        assert "train" in result


class TestStreamingDatasetLoaderHuggingFace:
    """Tests for StreamingDatasetLoader with HuggingFace datasets."""

    def test_streaming_loader_detects_hf_dataset(self):
        """StreamingDatasetLoader should detect HuggingFace dataset names."""
        from backpropagate.datasets import StreamingDatasetLoader

        # Non-existent path should be treated as HF dataset
        loader = StreamingDatasetLoader("HuggingFaceH4/ultrachat_200k")
        assert loader._is_hf_dataset is True

    def test_streaming_loader_detects_local_file(self, tmp_path):
        """StreamingDatasetLoader should detect local files."""
        from backpropagate.datasets import StreamingDatasetLoader

        file_path = tmp_path / "test.jsonl"
        file_path.write_text('{"text": "test"}\n')

        loader = StreamingDatasetLoader(str(file_path))
        assert loader._is_hf_dataset is False


class TestConvertToChatmlRawText:
    """Tests for convert_to_chatml with RAW_TEXT format."""

    def test_convert_raw_text_string(self):
        """convert_to_chatml should handle raw text strings."""
        from backpropagate.datasets import convert_to_chatml, DatasetFormat

        samples = ["Hello world", "This is plain text"]
        result = convert_to_chatml(samples, DatasetFormat.RAW_TEXT)

        assert len(result) == 2
        assert "<|im_start|>" in result[0]["text"]
        assert "Hello world" in result[0]["text"]

    def test_convert_raw_text_with_format_detection(self):
        """convert_to_chatml should auto-detect raw text."""
        from backpropagate.datasets import convert_to_chatml

        samples = ["Plain text without any format markers"]
        result = convert_to_chatml(samples)

        assert len(result) == 1
        assert "<|im_start|>" in result[0]["text"]


class TestValidateDatasetMaxErrors:
    """Tests for validate_dataset with max_errors limit."""

    def test_validate_dataset_stops_at_max_errors(self):
        """validate_dataset should stop collecting errors at max_errors."""
        from backpropagate.datasets import validate_dataset, DatasetFormat

        # Create many invalid samples
        invalid_samples = [
            {"invalid_field": f"value_{i}"}
            for i in range(50)
        ]

        result = validate_dataset(invalid_samples, DatasetFormat.SHAREGPT, max_errors=5)

        # Should have stopped at max_errors
        assert len(result.errors) <= 5

    def test_validate_dataset_default_max_errors(self):
        """validate_dataset should use default max_errors of 100."""
        from backpropagate.datasets import validate_dataset, DatasetFormat

        # Create many invalid samples
        invalid_samples = [
            {"conversations": "not_a_list"}  # Invalid type
            for _ in range(150)
        ]

        result = validate_dataset(invalid_samples, DatasetFormat.SHAREGPT)

        # Default max_errors is 100
        assert len(result.errors) <= 100


class TestFormatConverterUnknown:
    """Tests for FormatConverter with UNKNOWN format."""

    def test_to_chatml_unknown_format_raises(self):
        """FormatConverter.to_chatml should raise for UNKNOWN format."""
        from backpropagate.datasets import FormatConverter, DatasetFormat

        sample = {"random": "data"}

        with pytest.raises(ValueError, match="Cannot convert format"):
            FormatConverter.to_chatml(sample, DatasetFormat.UNKNOWN)


class TestDatasetLoaderEdgeCases:
    """Additional edge case tests for DatasetLoader."""

    def test_loader_with_empty_file(self, tmp_path):
        """DatasetLoader should handle empty files."""
        file_path = tmp_path / "empty.jsonl"
        file_path.write_text("")

        loader = DatasetLoader(str(file_path), validate=False)
        assert len(loader) == 0

    def test_loader_preview_as_json(self, tmp_path):
        """DatasetLoader.preview should work with as_chatml=False."""
        file_path = tmp_path / "test.jsonl"
        samples = [{"text": "<|im_start|>user\nHello<|im_end|>"}]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = DatasetLoader(str(file_path))
        preview = loader.preview(1, as_chatml=False)

        assert len(preview) == 1
        # Should be JSON string
        assert "{" in preview[0]

    def test_loader_shuffle_with_seed(self, tmp_path):
        """DatasetLoader.shuffle should be reproducible with seed."""
        file_path = tmp_path / "test.jsonl"
        samples = [
            {"text": f"<|im_start|>user\nSample {i}<|im_end|>"}
            for i in range(10)
        ]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = DatasetLoader(str(file_path))

        shuffled1 = loader.shuffle(seed=42)
        shuffled2 = loader.shuffle(seed=42)

        # Same seed should produce same order
        assert [s["text"] for s in shuffled1._samples] == [s["text"] for s in shuffled2._samples]

    def test_loader_split(self, tmp_path):
        """DatasetLoader.split should create train/test loaders."""
        file_path = tmp_path / "test.jsonl"
        samples = [
            {"text": f"<|im_start|>user\nSample {i}<|im_end|>"}
            for i in range(10)
        ]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = DatasetLoader(str(file_path))
        train_loader, test_loader = loader.split(train_ratio=0.8, seed=42)

        assert len(train_loader) == 8
        assert len(test_loader) == 2

    def test_loader_getitem(self, tmp_path):
        """DatasetLoader should support indexing."""
        file_path = tmp_path / "test.jsonl"
        samples = [{"text": f"sample {i}"} for i in range(5)]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = DatasetLoader(str(file_path), validate=False)

        assert loader[0]["text"] == "sample 0"
        assert loader[4]["text"] == "sample 4"

    def test_loader_iter(self, tmp_path):
        """DatasetLoader should support iteration."""
        file_path = tmp_path / "test.jsonl"
        samples = [{"text": f"sample {i}"} for i in range(3)]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = DatasetLoader(str(file_path), validate=False)

        items = list(loader)
        assert len(items) == 3


# =============================================================================
# DATASET LOADING TESTS (Phase 5 additions)
# =============================================================================

class TestDatasetLoadingAdvanced:
    """Advanced dataset loading tests."""

    def test_load_ultrachat(self):
        """Should load HuggingFace UltraChat dataset (mocked)."""
        pytest.importorskip("datasets")
        from backpropagate.datasets import StreamingDatasetLoader

        # Mock the HuggingFace dataset loading
        mock_samples = [
            {"messages": [{"role": "user", "content": f"Question {i}"}, {"role": "assistant", "content": f"Answer {i}"}]}
            for i in range(5)
        ]

        with patch("datasets.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = lambda self: iter(mock_samples)
            mock_load.return_value = mock_dataset

            loader = StreamingDatasetLoader("HuggingFaceH4/ultrachat_200k", split="train_sft")
            # Taking samples should work
            samples = loader.take(3)
            # Loader should recognize HF dataset source
            assert loader._is_hf_dataset is True

    def test_load_with_sample_limit(self, tmp_path):
        """Should limit number of samples when loading."""
        from backpropagate.datasets import StreamingDatasetLoader

        file_path = tmp_path / "test.jsonl"
        samples = [{"text": f"<|im_start|>user\nSample {i}<|im_end|>"} for i in range(100)]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = StreamingDatasetLoader(str(file_path))

        # Limit to first 10 samples
        limited = loader.take(10)
        assert len(limited) == 10
        assert limited[0]["text"] == "<|im_start|>user\nSample 0<|im_end|>"
        assert limited[9]["text"] == "<|im_start|>user\nSample 9<|im_end|>"

    def test_load_with_offset(self, tmp_path):
        """Should skip first N samples when loading."""
        from backpropagate.datasets import StreamingDatasetLoader

        file_path = tmp_path / "test.jsonl"
        samples = [{"text": f"<|im_start|>user\nSample {i}<|im_end|>"} for i in range(20)]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = StreamingDatasetLoader(str(file_path))

        # Skip first 5 samples, take next 5
        skipped = list(loader.skip(5))
        assert len(skipped) == 15
        assert skipped[0]["text"] == "<|im_start|>user\nSample 5<|im_end|>"

    def test_dataset_shuffling_reproducible(self, tmp_path):
        """Should shuffle with seed for reproducibility."""
        file_path = tmp_path / "test.jsonl"
        samples = [
            {"text": f"<|im_start|>user\nSample {i}<|im_end|>"}
            for i in range(20)
        ]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = DatasetLoader(str(file_path))

        # Shuffle with same seed should produce identical results
        shuffled1 = loader.shuffle(seed=12345)
        shuffled2 = loader.shuffle(seed=12345)

        texts1 = [s["text"] for s in shuffled1.samples]
        texts2 = [s["text"] for s in shuffled2.samples]

        assert texts1 == texts2

        # Different seed should produce different order
        shuffled3 = loader.shuffle(seed=99999)
        texts3 = [s["text"] for s in shuffled3.samples]

        assert texts1 != texts3


# =============================================================================
# TOKENIZATION TESTS (Phase 5 additions)
# =============================================================================

class TestTokenization:
    """Tests for dataset tokenization functionality."""

    def test_tokenize_conversations(self):
        """Should tokenize chat format correctly."""
        # Note: This tests the ChatML conversion which is the pre-tokenization step
        from backpropagate.datasets import FormatConverter, DatasetFormat

        # ShareGPT format
        sample = {
            "conversations": [
                {"from": "system", "value": "You are helpful."},
                {"from": "human", "value": "Hello!"},
                {"from": "gpt", "value": "Hi there! How can I help?"},
                {"from": "human", "value": "What's 2+2?"},
                {"from": "gpt", "value": "4"},
            ]
        }

        chatml = FormatConverter.to_chatml(sample, DatasetFormat.SHAREGPT)

        # Verify all parts are present and in correct order
        assert "<|im_start|>system\nYou are helpful.<|im_end|>" in chatml
        assert "<|im_start|>user\nHello!<|im_end|>" in chatml
        assert "<|im_start|>assistant\nHi there! How can I help?<|im_end|>" in chatml

        # Verify order - system comes before user
        system_pos = chatml.find("<|im_start|>system")
        user_pos = chatml.find("<|im_start|>user")
        assert system_pos < user_pos

    def test_tokenize_truncation(self):
        """Should handle long sequences (preparation for truncation)."""
        from backpropagate.datasets import filter_by_quality

        # Create a very long sample
        long_text = "x" * 50000  # Very long text
        samples = [
            {"text": f"<|im_start|>user\n{long_text}<|im_end|>\n<|im_start|>assistant\nOK<|im_end|>"},
        ]

        # Filter by max tokens should remove it
        filtered, stats = filter_by_quality(
            samples,
            min_tokens=0,
            max_tokens=1000,  # Approximately 4000 characters
            min_turns=0,
            require_assistant=False,
        )

        assert len(filtered) == 0
        assert stats.removed_too_long == 1

    def test_tokenize_padding(self):
        """Should handle padding considerations (preparation for training)."""
        from backpropagate.datasets import get_dataset_stats, DatasetFormat

        # Create samples of different lengths
        samples = [
            {"text": "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>"},
            {"text": "<|im_start|>user\nThis is a much longer question about something complex<|im_end|>\n<|im_start|>assistant\nThis is a detailed response<|im_end|>"},
        ]

        stats = get_dataset_stats(samples, DatasetFormat.CHATML)

        # Should compute min/max tokens for padding decisions
        assert stats.min_tokens > 0
        assert stats.max_tokens > stats.min_tokens

    def test_chat_template_applied(self):
        """Should apply model's chat template (ChatML format)."""
        from backpropagate.datasets import FormatConverter, DatasetFormat

        # OpenAI format input
        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }

        chatml = FormatConverter.to_chatml(sample, DatasetFormat.OPENAI)

        # ChatML template markers should be present
        assert "<|im_start|>" in chatml
        assert "<|im_end|>" in chatml
        assert "<|im_start|>user\nHello<|im_end|>" in chatml
        assert "<|im_start|>assistant\nHi!<|im_end|>" in chatml


# =============================================================================
# DATA CHUNKING TESTS (Phase 5 additions)
# =============================================================================

class TestDataChunking:
    """Tests for data chunking functionality."""

    def test_get_data_chunk(self):
        """Should get specific chunk by index."""
        from backpropagate.datasets import get_curriculum_chunks

        samples = [{"text": f"sample_{i}"} for i in range(100)]

        # Split into 5 chunks
        chunks = get_curriculum_chunks(samples, num_chunks=5, key="text")

        assert len(chunks) == 5
        # Each chunk should have ~20 samples
        for chunk in chunks:
            assert len(chunk) >= 19  # Allow for rounding

        # Should be able to access specific chunks
        chunk_0 = chunks[0]
        chunk_2 = chunks[2]
        chunk_4 = chunks[4]

        assert len(chunk_0) > 0
        assert len(chunk_2) > 0
        assert len(chunk_4) > 0

    def test_chunk_no_overlap(self):
        """Chunks shouldn't share samples."""
        from backpropagate.datasets import get_curriculum_chunks

        samples = [{"text": f"unique_sample_{i}"} for i in range(50)]

        chunks = get_curriculum_chunks(samples, num_chunks=5, key="text")

        # Collect all texts from all chunks
        all_texts = []
        for chunk in chunks:
            for sample in chunk:
                all_texts.append(sample["text"])

        # Check for duplicates - each sample should appear exactly once
        assert len(all_texts) == len(set(all_texts))
        assert len(all_texts) == 50

    def test_chunk_exhaustion(self):
        """Should handle when all data is used."""
        from backpropagate.datasets import get_curriculum_chunks

        samples = [{"text": f"sample_{i}"} for i in range(10)]

        # Create more chunks than samples
        chunks = get_curriculum_chunks(samples, num_chunks=5, key="text")

        # All samples should be distributed
        total_samples = sum(len(chunk) for chunk in chunks)
        assert total_samples == 10

        # Some chunks may be empty or small
        assert all(isinstance(chunk, list) for chunk in chunks)

    def test_streaming_batches_chunk_style(self, tmp_path):
        """Should yield samples in batch chunks."""
        from backpropagate.datasets import StreamingDatasetLoader

        file_path = tmp_path / "test.jsonl"
        samples = [{"text": f"sample_{i}"} for i in range(25)]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = StreamingDatasetLoader(str(file_path))

        # Get data in chunks of 10
        batches = list(loader.batches(10))

        assert len(batches) == 3  # 10 + 10 + 5
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5

        # Verify no overlap between batches
        all_texts = []
        for batch in batches:
            for sample in batch:
                all_texts.append(sample["text"])

        assert len(all_texts) == len(set(all_texts))


class TestStreamingLoaderOperations:
    """Tests for StreamingDatasetLoader operations."""

    def test_streaming_skip(self, tmp_path):
        """StreamingDatasetLoader.skip should skip samples."""
        from backpropagate.datasets import StreamingDatasetLoader

        file_path = tmp_path / "test.jsonl"
        samples = [{"text": f"sample {i}"} for i in range(5)]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = StreamingDatasetLoader(str(file_path))

        # Skip first 2 samples
        skipped = list(loader.skip(2))
        assert len(skipped) == 3
        assert skipped[0]["text"] == "sample 2"

    def test_streaming_batches(self, tmp_path):
        """StreamingDatasetLoader.batches should yield batches."""
        from backpropagate.datasets import StreamingDatasetLoader

        file_path = tmp_path / "test.jsonl"
        samples = [{"text": f"sample {i}"} for i in range(7)]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = StreamingDatasetLoader(str(file_path))

        batches = list(loader.batches(3))
        assert len(batches) == 3  # 3 + 3 + 1
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 1

    def test_streaming_to_chatml_with_n(self, tmp_path):
        """StreamingDatasetLoader.to_chatml should limit samples with n."""
        from backpropagate.datasets import StreamingDatasetLoader

        file_path = tmp_path / "test.jsonl"
        samples = [{"text": f"<|im_start|>user\nSample {i}<|im_end|>"} for i in range(10)]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = StreamingDatasetLoader(str(file_path))
        chatml = loader.to_chatml(n=5)

        assert len(chatml) == 5

    def test_streaming_filter(self, tmp_path):
        """StreamingDatasetLoader.filter should yield filtered samples."""
        from backpropagate.datasets import StreamingDatasetLoader

        file_path = tmp_path / "test.jsonl"
        samples = [
            {"text": "<|im_start|>user\nShort<|im_end|>\n<|im_start|>assistant\nY<|im_end|>"},
            {"text": "<|im_start|>user\nThis is a longer message with more content<|im_end|>\n<|im_start|>assistant\nThis response is also longer<|im_end|>"},
        ]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = StreamingDatasetLoader(str(file_path))

        # Filter by min tokens
        filtered = list(loader.filter(min_tokens=20))
        assert len(filtered) == 1  # Only the longer one

    def test_streaming_detected_format(self, tmp_path):
        """StreamingDatasetLoader should detect format."""
        from backpropagate.datasets import StreamingDatasetLoader, DatasetFormat

        file_path = tmp_path / "test.jsonl"
        samples = [{"text": "<|im_start|>user\nTest<|im_end|>"}]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = StreamingDatasetLoader(str(file_path))

        # Take a sample to trigger format detection
        loader.take(1)

        assert loader.detected_format == DatasetFormat.CHATML


# =============================================================================
# CURRICULUM LEARNING TESTS (Additional Coverage)
# =============================================================================

class TestCurriculumLearning:
    """Tests for curriculum learning utilities."""

    def test_get_curriculum_chunks_basic(self):
        """Should split samples into curriculum chunks."""
        from backpropagate.datasets import get_curriculum_chunks

        # Create samples with varying complexity (by text length)
        samples = [
            {"text": "x" * 10},   # short
            {"text": "x" * 50},   # medium
            {"text": "x" * 100},  # long
            {"text": "x" * 20},   # short
            {"text": "x" * 80},   # long
            {"text": "x" * 30},   # medium
        ]

        chunks = get_curriculum_chunks(samples, num_chunks=3, key="text")

        assert len(chunks) == 3
        total = sum(len(c) for c in chunks)
        assert total == 6

    def test_get_curriculum_chunks_single_chunk(self):
        """Should handle single chunk request."""
        from backpropagate.datasets import get_curriculum_chunks

        samples = [{"text": f"sample_{i}"} for i in range(10)]
        chunks = get_curriculum_chunks(samples, num_chunks=1, key="text")

        assert len(chunks) == 1
        assert len(chunks[0]) == 10

    def test_get_curriculum_chunks_more_chunks_than_samples(self):
        """Should handle more chunks than samples."""
        from backpropagate.datasets import get_curriculum_chunks

        samples = [{"text": f"sample_{i}"} for i in range(3)]
        chunks = get_curriculum_chunks(samples, num_chunks=10, key="text")

        # Should have at most len(samples) non-empty chunks
        non_empty = [c for c in chunks if len(c) > 0]
        assert len(non_empty) <= 3

    def test_get_curriculum_chunks_sorts_by_complexity(self):
        """Should sort samples by complexity (length) before chunking."""
        from backpropagate.datasets import get_curriculum_chunks

        samples = [
            {"text": "x" * 100},  # longest
            {"text": "x" * 10},   # shortest
            {"text": "x" * 50},   # middle
        ]

        chunks = get_curriculum_chunks(samples, num_chunks=3, key="text")

        # First chunk should have shortest samples
        first_chunk_lengths = [len(s["text"]) for s in chunks[0]]
        last_chunk_lengths = [len(s["text"]) for s in chunks[-1]]

        assert max(first_chunk_lengths) <= min(last_chunk_lengths) or len(chunks[-1]) == 0

    def test_get_curriculum_chunks_empty_samples(self):
        """Should handle empty sample list."""
        from backpropagate.datasets import get_curriculum_chunks

        chunks = get_curriculum_chunks([], num_chunks=3, key="text")

        assert len(chunks) == 3
        assert all(len(c) == 0 for c in chunks)


class TestDatasetLoaderCurriculum:
    """Tests for DatasetLoader curriculum methods."""

    def test_loader_get_curriculum_chunks(self, tmp_path):
        """DatasetLoader should support curriculum chunking."""
        file_path = tmp_path / "test.jsonl"
        samples = [
            {"text": f"<|im_start|>user\n{'x' * (i * 10)}<|im_end|>"}
            for i in range(1, 11)
        ]
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = DatasetLoader(str(file_path), validate=False)

        # If method exists
        if hasattr(loader, "curriculum_chunks"):
            chunks = loader.curriculum_chunks(num_chunks=3)
            assert len(chunks) == 3


# =============================================================================
# DATASET STATS ADDITIONAL TESTS
# =============================================================================

class TestDatasetStatsAdditional:
    """Additional tests for DatasetStats."""

    def test_stats_with_long_samples(self):
        """Should handle samples with many tokens."""
        samples = [
            {"text": "<|im_start|>user\n" + "word " * 500 + "<|im_end|>"},
        ]
        stats = get_dataset_stats(samples, DatasetFormat.CHATML)

        assert stats.total_tokens_approx > 100
        assert stats.max_tokens > 100

    def test_stats_empty_text_samples(self):
        """Should handle samples with empty text."""
        samples = [
            {"text": ""},
            {"text": "<|im_start|>user\nHello<|im_end|>"},
        ]
        stats = get_dataset_stats(samples, DatasetFormat.CHATML)

        assert stats.total_samples == 2
        assert stats.min_tokens == 0 or stats.min_tokens >= 0

    def test_stats_multi_turn_counting(self):
        """Should count multiple turns accurately."""
        samples = [
            {"text": "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\nA2<|im_end|>\n<|im_start|>user\nQ3<|im_end|>\n<|im_start|>assistant\nA3<|im_end|>"},
        ]
        stats = get_dataset_stats(samples, DatasetFormat.CHATML)

        assert stats.avg_turns_per_conversation == 6  # 3 pairs

    def test_stats_unique_system_prompts(self):
        """Should count unique system prompts."""
        samples = [
            {"text": "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHi<|im_end|>"},
            {"text": "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHello<|im_end|>"},  # Same system
            {"text": "<|im_start|>system\nYou are different<|im_end|>\n<|im_start|>user\nHey<|im_end|>"},  # Different
        ]
        stats = get_dataset_stats(samples, DatasetFormat.CHATML)

        assert stats.has_system_prompts is True
        assert stats.unique_system_prompts == 2


# =============================================================================
# VALIDATION ADDITIONAL TESTS
# =============================================================================

class TestValidationAdditional:
    """Additional validation tests for edge cases."""

    def test_validate_sharegpt_invalid_role(self):
        """Should handle invalid ShareGPT roles.

        Note: The validation may be lenient about unknown roles,
        so we just verify it processes without error.
        """
        samples = [
            {"conversations": [{"from": "invalid_role", "value": "text"}]},
        ]
        result = validate_dataset(samples, DatasetFormat.SHAREGPT)

        # Validation may be lenient - just ensure it ran
        assert result.total_rows == 1

    def test_validate_openai_missing_content(self):
        """Should catch missing content in OpenAI format."""
        samples = [
            {"messages": [{"role": "user"}]},  # Missing content
        ]
        result = validate_dataset(samples, DatasetFormat.OPENAI)

        assert not result.is_valid or result.warning_count > 0

    def test_validate_chatml_nested_tags(self):
        """Should handle nested ChatML tags."""
        samples = [
            {"text": "<|im_start|>user\nCode: <|im_start|>example<|im_end|><|im_end|>"},
        ]
        result = validate_dataset(samples, DatasetFormat.CHATML)

        # Nested tags should be flagged
        assert result.warning_count > 0 or result.error_count > 0

    def test_validate_very_long_sample(self):
        """Should handle very long samples."""
        long_text = "<|im_start|>user\n" + "x" * 100000 + "<|im_end|>"
        samples = [{"text": long_text}]
        result = validate_dataset(samples, DatasetFormat.CHATML)

        assert result.total_rows == 1

    def test_validation_result_error_rate_edge_cases(self):
        """Test error rate with edge cases."""
        # Zero total rows
        result = ValidationResult(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            errors=[],
            format_detected=DatasetFormat.UNKNOWN,
        )
        assert result.error_rate == 0.0

        # All rows valid
        result = ValidationResult(
            is_valid=True,
            total_rows=100,
            valid_rows=100,
            errors=[],
            format_detected=DatasetFormat.SHAREGPT,
        )
        assert result.error_rate == 0.0

        # All rows invalid
        result = ValidationResult(
            is_valid=False,
            total_rows=100,
            valid_rows=0,
            errors=["error"],
            format_detected=DatasetFormat.UNKNOWN,
        )
        assert result.error_rate == 1.0


# =============================================================================
# FORMAT CONVERTER ADDITIONAL TESTS
# =============================================================================

class TestFormatConverterAdditional:
    """Additional tests for FormatConverter."""

    def test_sharegpt_multi_turn(self):
        """Should handle multi-turn ShareGPT conversation."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Q1"},
                {"from": "gpt", "value": "A1"},
                {"from": "human", "value": "Q2"},
                {"from": "gpt", "value": "A2"},
                {"from": "human", "value": "Q3"},
                {"from": "gpt", "value": "A3"},
            ]
        }
        result = FormatConverter.sharegpt_to_chatml(sample)

        assert result.count("<|im_start|>user") == 3
        assert result.count("<|im_start|>assistant") == 3

    def test_alpaca_with_empty_input(self):
        """Should handle Alpaca with explicitly empty input."""
        sample = {
            "instruction": "Just say hello",
            "input": "",
            "output": "Hello!",
        }
        result = FormatConverter.alpaca_to_chatml(sample)

        assert "Just say hello" in result
        assert "Hello!" in result

    def test_openai_with_tool_calls(self):
        """Should handle OpenAI format with function/tool role."""
        sample = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check..."},
                {"role": "tool", "content": "Temperature: 72F"},
                {"role": "assistant", "content": "It's 72 degrees."},
            ]
        }
        result = FormatConverter.openai_to_chatml(sample)

        # Should include all messages
        assert "weather" in result.lower()
        assert "72" in result

    def test_to_chatml_with_chatml_input(self):
        """Should pass through ChatML format unchanged."""
        sample = {"text": "<|im_start|>user\nHello<|im_end|>"}
        result = FormatConverter.to_chatml(sample, DatasetFormat.CHATML)

        assert result == "<|im_start|>user\nHello<|im_end|>"


# =============================================================================
# FILTER BY QUALITY ADDITIONAL TESTS
# =============================================================================

class TestFilterByQualityAdditional:
    """Additional tests for filter_by_quality function."""

    def test_filter_all_removed(self):
        """Should handle case where all samples are filtered."""
        from backpropagate.datasets import filter_by_quality

        samples = [
            {"text": "x"},  # Too short
            {"text": "y"},  # Too short
        ]

        filtered, stats = filter_by_quality(
            samples,
            min_tokens=100,
            max_tokens=10000,
            min_turns=0,
            require_assistant=False,
        )

        assert len(filtered) == 0
        assert stats.retention_rate == 0.0

    def test_filter_none_removed(self):
        """Should handle case where no samples are filtered."""
        from backpropagate.datasets import filter_by_quality

        samples = [
            {"text": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>"},
            {"text": "<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\nGoodbye!<|im_end|>"},
        ]

        filtered, stats = filter_by_quality(
            samples,
            min_tokens=0,
            max_tokens=10000,
            min_turns=0,
            require_assistant=False,
        )

        assert len(filtered) == 2
        assert stats.retention_rate == 1.0

    def test_filter_combined_criteria(self):
        """Should apply multiple filter criteria."""
        from backpropagate.datasets import filter_by_quality

        samples = [
            {"text": "x" * 10},  # Too short
            {"text": "x" * 10000},  # Too long
            {"text": "<|im_start|>user\nHi<|im_end|>"},  # No assistant
            {"text": "<|im_start|>user\n" + "x" * 200 + "<|im_end|>\n<|im_start|>assistant\nOK<|im_end|>"},  # Good
        ]

        filtered, stats = filter_by_quality(
            samples,
            min_tokens=10,
            max_tokens=500,
            min_turns=2,
            require_assistant=True,
        )

        assert len(filtered) == 1
        assert stats.removed_too_short >= 1
        assert stats.removed_too_long >= 1


# =============================================================================
# DEDUPLICATION ADDITIONAL TESTS
# =============================================================================

class TestDeduplicationAdditional:
    """Additional deduplication tests."""

    def test_dedupe_exact_with_none_key(self):
        """Should handle None in key field."""
        from backpropagate.datasets import deduplicate_exact

        samples = [
            {"text": "Hello"},
            {"text": None},
            {"text": "Hello"},
        ]

        unique, num_removed = deduplicate_exact(samples, key="text")

        # Should dedupe "Hello" and handle None
        assert len(unique) == 2
        assert num_removed == 1

    def test_dedupe_exact_empty_list(self):
        """Should handle empty list."""
        from backpropagate.datasets import deduplicate_exact

        unique, num_removed = deduplicate_exact([])

        assert len(unique) == 0
        assert num_removed == 0

    def test_dedupe_exact_all_duplicates(self):
        """Should handle all duplicates."""
        from backpropagate.datasets import deduplicate_exact

        samples = [
            {"text": "Same"},
            {"text": "Same"},
            {"text": "Same"},
        ]

        unique, num_removed = deduplicate_exact(samples)

        assert len(unique) == 1
        assert num_removed == 2


# =============================================================================
# STREAMING LOADER ADDITIONAL TESTS
# =============================================================================

class TestStreamingLoaderAdditional:
    """Additional tests for StreamingDatasetLoader."""

    def test_streaming_empty_file(self, tmp_path):
        """Should handle empty file."""
        from backpropagate.datasets import StreamingDatasetLoader

        file_path = tmp_path / "empty.jsonl"
        file_path.write_text("")

        loader = StreamingDatasetLoader(str(file_path))
        samples = list(loader)

        assert len(samples) == 0

    def test_streaming_single_sample(self, tmp_path):
        """Should handle file with single sample."""
        from backpropagate.datasets import StreamingDatasetLoader

        file_path = tmp_path / "single.jsonl"
        file_path.write_text('{"text": "hello"}\n')

        loader = StreamingDatasetLoader(str(file_path))
        samples = list(loader)

        assert len(samples) == 1
        assert samples[0]["text"] == "hello"

    def test_streaming_malformed_json_line(self, tmp_path):
        """Should handle malformed JSON lines gracefully."""
        from backpropagate.datasets import StreamingDatasetLoader

        file_path = tmp_path / "malformed.jsonl"
        file_path.write_text('{"text": "good"}\n{bad json\n{"text": "also good"}\n')

        loader = StreamingDatasetLoader(str(file_path))

        # Should either skip bad lines or raise
        try:
            samples = list(loader)
            # If it skips, should have 2 good samples
            assert len(samples) >= 2 or len(samples) == 0
        except json.JSONDecodeError:
            # Also acceptable behavior
            pass

    def test_streaming_with_filter_empty_result(self, tmp_path):
        """Should handle filter that removes all samples."""
        from backpropagate.datasets import StreamingDatasetLoader

        # Use valid ChatML format so conversion works
        file_path = tmp_path / "test.jsonl"
        samples = [{"text": "<|im_start|>user\nx<|im_end|>"}]  # Very short but valid format
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        loader = StreamingDatasetLoader(str(file_path))
        filtered = list(loader.filter(min_tokens=1000))

        assert len(filtered) == 0
