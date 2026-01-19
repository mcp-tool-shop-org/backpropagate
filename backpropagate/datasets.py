"""
Backpropagate - Dataset Utilities
=================================

Painless data preparation for LLM fine-tuning.

Features:
- Auto-detect format (ShareGPT, Alpaca, OpenAI, raw text)
- Convert any format to ChatML
- Validate datasets before training
- Preview and statistics

Supported Formats:
- ShareGPT: {"conversations": [{"from": "human/gpt", "value": "..."}]}
- Alpaca: {"instruction": "...", "input": "...", "output": "..."}
- OpenAI: {"messages": [{"role": "user/assistant", "content": "..."}]}
- ChatML: {"text": "<|im_start|>user\n...<|im_end|>\n..."}
- Raw: Plain text files

Usage:
    from backpropagate.datasets import DatasetLoader

    loader = DatasetLoader("my_data.jsonl")
    print(loader.detected_format)  # "sharegpt"
    print(loader.validation_report())

    # Convert and get HuggingFace dataset
    dataset = loader.to_chatml()
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)

__all__ = [
    # Core classes
    "DatasetFormat",
    "DatasetLoader",
    "ValidationResult",
    "ValidationError",
    "DatasetStats",
    "FormatConverter",
    # Core functions
    "detect_format",
    "validate_dataset",
    "convert_to_chatml",
    "preview_samples",
    "get_dataset_stats",
    # Streaming
    "StreamingDatasetLoader",
    # Filtering
    "FilterStats",
    "filter_by_quality",
    # Deduplication
    "deduplicate_exact",
    "deduplicate_minhash",
    # Perplexity filtering
    "PerplexityFilter",
    "PerplexityStats",
    "compute_perplexity",
    "filter_by_perplexity",
    # Curriculum learning (Phase 3.3)
    "CurriculumStats",
    "compute_difficulty_score",
    "order_by_difficulty",
    "get_curriculum_chunks",
    "analyze_curriculum",
]


# =============================================================================
# ENUMS AND DATACLASSES
# =============================================================================

class DatasetFormat(Enum):
    """Supported dataset formats."""
    SHAREGPT = "sharegpt"
    ALPACA = "alpaca"
    OPENAI = "openai"
    CHATML = "chatml"
    RAW_TEXT = "raw_text"
    UNKNOWN = "unknown"


@dataclass
class ValidationError:
    """A single validation error."""
    row_index: int
    field: str
    error_type: str
    message: str
    value: Optional[Any] = None

    def __str__(self) -> str:
        return f"Row {self.row_index}: [{self.error_type}] {self.field} - {self.message}"


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    total_rows: int
    valid_rows: int
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    format_detected: DatasetFormat = DatasetFormat.UNKNOWN

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def error_rate(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return (self.total_rows - self.valid_rows) / self.total_rows

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Dataset Validation Report",
            f"=" * 40,
            f"Format: {self.format_detected.value}",
            f"Total rows: {self.total_rows}",
            f"Valid rows: {self.valid_rows} ({100 * self.valid_rows / max(1, self.total_rows):.1f}%)",
            f"Errors: {self.error_count}",
            f"Warnings: {self.warning_count}",
        ]

        if self.errors:
            lines.append("\nFirst 5 errors:")
            for err in self.errors[:5]:
                lines.append(f"  - {err}")

        if self.warnings:
            lines.append("\nFirst 5 warnings:")
            for warn in self.warnings[:5]:
                lines.append(f"  - {warn}")

        return "\n".join(lines)


@dataclass
class DatasetStats:
    """Statistics about a dataset."""
    total_samples: int
    total_tokens_approx: int
    avg_tokens_per_sample: float
    min_tokens: int
    max_tokens: int
    format_detected: DatasetFormat
    has_system_prompts: bool
    avg_turns_per_conversation: float
    unique_system_prompts: int


@dataclass
class FilterStats:
    """Statistics from quality filtering."""
    total_before: int
    total_after: int
    removed_too_short: int = 0
    removed_too_long: int = 0
    removed_few_turns: int = 0
    removed_many_turns: int = 0
    removed_empty: int = 0
    removed_no_assistant: int = 0
    removed_custom: int = 0

    @property
    def total_removed(self) -> int:
        return self.total_before - self.total_after

    @property
    def retention_rate(self) -> float:
        if self.total_before == 0:
            return 0.0
        return self.total_after / self.total_before

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Filter Results",
            f"=" * 40,
            f"Before: {self.total_before}",
            f"After:  {self.total_after} ({100 * self.retention_rate:.1f}% retained)",
            f"Removed: {self.total_removed}",
        ]
        if self.removed_too_short > 0:
            lines.append(f"  - Too short: {self.removed_too_short}")
        if self.removed_too_long > 0:
            lines.append(f"  - Too long: {self.removed_too_long}")
        if self.removed_few_turns > 0:
            lines.append(f"  - Too few turns: {self.removed_few_turns}")
        if self.removed_many_turns > 0:
            lines.append(f"  - Too many turns: {self.removed_many_turns}")
        if self.removed_empty > 0:
            lines.append(f"  - Empty content: {self.removed_empty}")
        if self.removed_no_assistant > 0:
            lines.append(f"  - No assistant: {self.removed_no_assistant}")
        if self.removed_custom > 0:
            lines.append(f"  - Custom filter: {self.removed_custom}")
        return "\n".join(lines)


# =============================================================================
# FORMAT DETECTION
# =============================================================================

def detect_format(data: Union[Dict, List[Dict], str]) -> DatasetFormat:
    """
    Auto-detect the format of a dataset sample.

    Args:
        data: A single sample or list of samples

    Returns:
        Detected DatasetFormat
    """
    # Handle list - check first item
    if isinstance(data, list):
        if not data:
            return DatasetFormat.UNKNOWN
        data = data[0]

    # Handle string (raw text or file content)
    if isinstance(data, str):
        # Check if it's ChatML formatted
        if "<|im_start|>" in data and "<|im_end|>" in data:
            return DatasetFormat.CHATML
        return DatasetFormat.RAW_TEXT

    if not isinstance(data, dict):
        return DatasetFormat.UNKNOWN

    # Check for ShareGPT format
    if "conversations" in data:
        convos = data["conversations"]
        if isinstance(convos, list) and convos:
            first = convos[0]
            if isinstance(first, dict) and "from" in first and "value" in first:
                return DatasetFormat.SHAREGPT

    # Check for OpenAI format
    if "messages" in data:
        msgs = data["messages"]
        if isinstance(msgs, list) and msgs:
            first = msgs[0]
            if isinstance(first, dict) and "role" in first and "content" in first:
                return DatasetFormat.OPENAI

    # Check for Alpaca format
    if "instruction" in data and "output" in data:
        return DatasetFormat.ALPACA

    # Check for ChatML format (pre-formatted text)
    if "text" in data:
        text = data["text"]
        if isinstance(text, str) and "<|im_start|>" in text:
            return DatasetFormat.CHATML

    return DatasetFormat.UNKNOWN


def _detect_format_from_file(file_path: Path, sample_size: int = 5) -> DatasetFormat:
    """Detect format by sampling a file."""
    samples = []
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))

        elif suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data[:sample_size]
                else:
                    samples = [data]

        elif suffix in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return detect_format(content)

        else:
            # Try JSONL first
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            # Not JSON, treat as raw text
                            return DatasetFormat.RAW_TEXT

    except Exception as e:
        logger.warning(f"Error detecting format: {e}")
        return DatasetFormat.UNKNOWN

    if not samples:
        return DatasetFormat.UNKNOWN

    # Check consistency across samples
    formats = [detect_format(s) for s in samples]
    if formats:
        # Return most common format
        from collections import Counter
        return Counter(formats).most_common(1)[0][0]

    return DatasetFormat.UNKNOWN


# =============================================================================
# FORMAT CONVERTERS
# =============================================================================

class FormatConverter:
    """Convert between dataset formats."""

    # Role mappings for different formats
    ROLE_MAP_SHAREGPT = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }

    ROLE_MAP_OPENAI = {
        "user": "user",
        "assistant": "assistant",
        "system": "system",
    }

    @staticmethod
    def sharegpt_to_chatml(sample: Dict) -> str:
        """Convert ShareGPT format to ChatML."""
        conversations = sample.get("conversations", [])
        parts = []

        for turn in conversations:
            role_raw = turn.get("from", "").lower()
            role = FormatConverter.ROLE_MAP_SHAREGPT.get(role_raw, role_raw)
            content = turn.get("value", "")

            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        return "\n".join(parts)

    @staticmethod
    def alpaca_to_chatml(sample: Dict) -> str:
        """Convert Alpaca format to ChatML."""
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")
        system = sample.get("system", "")

        parts = []

        # Add system prompt if present
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>")

        # Combine instruction and input for user message
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        parts.append(f"<|im_start|>user\n{user_content}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{output}<|im_end|>")

        return "\n".join(parts)

    @staticmethod
    def openai_to_chatml(sample: Dict) -> str:
        """Convert OpenAI chat format to ChatML."""
        messages = sample.get("messages", [])
        parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle function calls (OpenAI format)
            if "function_call" in msg:
                content = f"[Function call: {msg['function_call']}]"

            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        return "\n".join(parts)

    @staticmethod
    def raw_to_chatml(text: str, default_role: str = "user") -> str:
        """Convert raw text to ChatML."""
        # Simple conversion - treat as single user message
        return f"<|im_start|>{default_role}\n{text}<|im_end|>"

    @classmethod
    def to_chatml(cls, sample: Union[Dict, str], format_type: DatasetFormat) -> str:
        """Convert any format to ChatML."""
        if format_type == DatasetFormat.CHATML:
            if isinstance(sample, dict):
                return sample.get("text", "")
            return sample

        if format_type == DatasetFormat.SHAREGPT:
            return cls.sharegpt_to_chatml(sample)

        if format_type == DatasetFormat.ALPACA:
            return cls.alpaca_to_chatml(sample)

        if format_type == DatasetFormat.OPENAI:
            return cls.openai_to_chatml(sample)

        if format_type == DatasetFormat.RAW_TEXT:
            text = sample if isinstance(sample, str) else sample.get("text", "")
            return cls.raw_to_chatml(text)

        raise ValueError(f"Cannot convert format: {format_type}")


def convert_to_chatml(
    samples: List[Union[Dict, str]],
    source_format: Optional[DatasetFormat] = None,
) -> List[Dict[str, str]]:
    """
    Convert a list of samples to ChatML format.

    Args:
        samples: List of samples in any supported format
        source_format: Optional format hint (auto-detected if not provided)

    Returns:
        List of dicts with "text" key containing ChatML
    """
    if not samples:
        return []

    if source_format is None:
        source_format = detect_format(samples[0])

    results = []
    for sample in samples:
        try:
            chatml = FormatConverter.to_chatml(sample, source_format)
            results.append({"text": chatml})
        except Exception as e:
            logger.warning(f"Failed to convert sample: {e}")
            continue

    return results


# =============================================================================
# VALIDATION
# =============================================================================

def _validate_chatml(text: str, row_index: int) -> List[ValidationError]:
    """Validate a ChatML formatted string."""
    errors = []

    # Check for balanced tags
    start_count = text.count("<|im_start|>")
    end_count = text.count("<|im_end|>")

    if start_count != end_count:
        errors.append(ValidationError(
            row_index=row_index,
            field="text",
            error_type="unbalanced_tags",
            message=f"Unbalanced ChatML tags: {start_count} starts, {end_count} ends",
        ))

    # Check for empty content
    if not text.strip():
        errors.append(ValidationError(
            row_index=row_index,
            field="text",
            error_type="empty_content",
            message="Empty text content",
        ))

    # Check for valid roles
    role_pattern = r"<\|im_start\|>(\w+)"
    roles = re.findall(role_pattern, text)
    valid_roles = {"system", "user", "assistant", "tool"}

    for role in roles:
        if role not in valid_roles:
            errors.append(ValidationError(
                row_index=row_index,
                field="role",
                error_type="invalid_role",
                message=f"Unknown role: {role}",
                value=role,
            ))

    return errors


def _validate_sharegpt(sample: Dict, row_index: int) -> List[ValidationError]:
    """Validate a ShareGPT formatted sample."""
    errors = []

    if "conversations" not in sample:
        errors.append(ValidationError(
            row_index=row_index,
            field="conversations",
            error_type="missing_field",
            message="Missing 'conversations' field",
        ))
        return errors

    convos = sample["conversations"]
    if not isinstance(convos, list):
        errors.append(ValidationError(
            row_index=row_index,
            field="conversations",
            error_type="invalid_type",
            message=f"Expected list, got {type(convos).__name__}",
        ))
        return errors

    if not convos:
        errors.append(ValidationError(
            row_index=row_index,
            field="conversations",
            error_type="empty_conversations",
            message="Empty conversations list",
        ))
        return errors

    for i, turn in enumerate(convos):
        if not isinstance(turn, dict):
            errors.append(ValidationError(
                row_index=row_index,
                field=f"conversations[{i}]",
                error_type="invalid_type",
                message=f"Expected dict, got {type(turn).__name__}",
            ))
            continue

        if "from" not in turn:
            errors.append(ValidationError(
                row_index=row_index,
                field=f"conversations[{i}].from",
                error_type="missing_field",
                message="Missing 'from' field",
            ))

        if "value" not in turn:
            errors.append(ValidationError(
                row_index=row_index,
                field=f"conversations[{i}].value",
                error_type="missing_field",
                message="Missing 'value' field",
            ))

    return errors


def _validate_alpaca(sample: Dict, row_index: int) -> List[ValidationError]:
    """Validate an Alpaca formatted sample."""
    errors = []

    if "instruction" not in sample:
        errors.append(ValidationError(
            row_index=row_index,
            field="instruction",
            error_type="missing_field",
            message="Missing 'instruction' field",
        ))

    if "output" not in sample:
        errors.append(ValidationError(
            row_index=row_index,
            field="output",
            error_type="missing_field",
            message="Missing 'output' field",
        ))

    # Check for empty values
    if sample.get("instruction", "").strip() == "":
        errors.append(ValidationError(
            row_index=row_index,
            field="instruction",
            error_type="empty_content",
            message="Empty instruction",
        ))

    if sample.get("output", "").strip() == "":
        errors.append(ValidationError(
            row_index=row_index,
            field="output",
            error_type="empty_content",
            message="Empty output",
        ))

    return errors


def _validate_openai(sample: Dict, row_index: int) -> List[ValidationError]:
    """Validate an OpenAI chat formatted sample."""
    errors = []

    if "messages" not in sample:
        errors.append(ValidationError(
            row_index=row_index,
            field="messages",
            error_type="missing_field",
            message="Missing 'messages' field",
        ))
        return errors

    messages = sample["messages"]
    if not isinstance(messages, list):
        errors.append(ValidationError(
            row_index=row_index,
            field="messages",
            error_type="invalid_type",
            message=f"Expected list, got {type(messages).__name__}",
        ))
        return errors

    if not messages:
        errors.append(ValidationError(
            row_index=row_index,
            field="messages",
            error_type="empty_messages",
            message="Empty messages list",
        ))
        return errors

    valid_roles = {"system", "user", "assistant", "function", "tool"}

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(ValidationError(
                row_index=row_index,
                field=f"messages[{i}]",
                error_type="invalid_type",
                message=f"Expected dict, got {type(msg).__name__}",
            ))
            continue

        if "role" not in msg:
            errors.append(ValidationError(
                row_index=row_index,
                field=f"messages[{i}].role",
                error_type="missing_field",
                message="Missing 'role' field",
            ))
        elif msg["role"] not in valid_roles:
            errors.append(ValidationError(
                row_index=row_index,
                field=f"messages[{i}].role",
                error_type="invalid_role",
                message=f"Invalid role: {msg['role']}",
                value=msg["role"],
            ))

        if "content" not in msg and "function_call" not in msg:
            errors.append(ValidationError(
                row_index=row_index,
                field=f"messages[{i}].content",
                error_type="missing_field",
                message="Missing 'content' field",
            ))

    return errors


def validate_sample(
    sample: Union[Dict, str],
    row_index: int,
    format_type: DatasetFormat,
) -> List[ValidationError]:
    """Validate a single sample."""
    if format_type == DatasetFormat.CHATML:
        text = sample if isinstance(sample, str) else sample.get("text", "")
        return _validate_chatml(text, row_index)

    if format_type == DatasetFormat.SHAREGPT:
        return _validate_sharegpt(sample, row_index)

    if format_type == DatasetFormat.ALPACA:
        return _validate_alpaca(sample, row_index)

    if format_type == DatasetFormat.OPENAI:
        return _validate_openai(sample, row_index)

    if format_type == DatasetFormat.RAW_TEXT:
        # Raw text has minimal validation
        if isinstance(sample, str) and not sample.strip():
            return [ValidationError(
                row_index=row_index,
                field="text",
                error_type="empty_content",
                message="Empty text content",
            )]
        return []

    return [ValidationError(
        row_index=row_index,
        field="format",
        error_type="unknown_format",
        message=f"Unknown format: {format_type}",
    )]


def validate_dataset(
    samples: List[Union[Dict, str]],
    format_type: Optional[DatasetFormat] = None,
    max_errors: int = 100,
) -> ValidationResult:
    """
    Validate an entire dataset.

    Args:
        samples: List of samples to validate
        format_type: Optional format hint (auto-detected if not provided)
        max_errors: Maximum number of errors to collect

    Returns:
        ValidationResult with all errors and warnings
    """
    if not samples:
        return ValidationResult(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            errors=[ValidationError(
                row_index=0,
                field="dataset",
                error_type="empty_dataset",
                message="Dataset is empty",
            )],
            format_detected=DatasetFormat.UNKNOWN,
        )

    if format_type is None:
        format_type = detect_format(samples[0])

    all_errors = []
    all_warnings = []
    valid_count = 0

    for i, sample in enumerate(samples):
        errors = validate_sample(sample, i, format_type)

        if errors:
            # Separate errors from warnings based on severity
            for err in errors:
                if err.error_type in ("empty_content", "invalid_role"):
                    all_warnings.append(err)
                else:
                    all_errors.append(err)

                if len(all_errors) >= max_errors:
                    break
        else:
            valid_count += 1

        if len(all_errors) >= max_errors:
            break

    return ValidationResult(
        is_valid=len(all_errors) == 0,
        total_rows=len(samples),
        valid_rows=valid_count,
        errors=all_errors,
        warnings=all_warnings,
        format_detected=format_type,
    )


# =============================================================================
# QUALITY FILTERING
# =============================================================================

def _count_tokens_approx(text: str) -> int:
    """Approximate token count (4 chars ≈ 1 token)."""
    return len(text) // 4


def _count_turns(text: str) -> int:
    """Count conversation turns in ChatML text."""
    return text.count("<|im_start|>")


def _has_assistant_response(text: str) -> bool:
    """Check if text contains an assistant response."""
    return "<|im_start|>assistant" in text


def filter_by_quality(
    samples: List[Dict],
    min_tokens: int = 50,
    max_tokens: int = 4096,
    min_turns: int = 2,
    max_turns: Optional[int] = None,
    remove_empty: bool = True,
    require_assistant: bool = True,
    custom_filter: Optional[Callable[[Dict], bool]] = None,
) -> Tuple[List[Dict], FilterStats]:
    """
    Filter samples by quality criteria.

    Args:
        samples: List of samples (should be ChatML format with "text" key)
        min_tokens: Minimum token count (approximate)
        max_tokens: Maximum token count (approximate)
        min_turns: Minimum conversation turns
        max_turns: Maximum conversation turns (None = no limit)
        remove_empty: Remove samples with empty content
        require_assistant: Require at least one assistant response
        custom_filter: Optional callable that returns True to keep sample

    Returns:
        Tuple of (filtered_samples, FilterStats)
    """
    stats = FilterStats(
        total_before=len(samples),
        total_after=0,
    )

    filtered = []

    for sample in samples:
        text = sample.get("text", "") if isinstance(sample, dict) else str(sample)

        # Check empty
        if remove_empty and not text.strip():
            stats.removed_empty += 1
            continue

        # Check token count
        token_count = _count_tokens_approx(text)
        if min_tokens is not None and token_count < min_tokens:
            stats.removed_too_short += 1
            continue
        if max_tokens is not None and token_count > max_tokens:
            stats.removed_too_long += 1
            continue

        # Check turn count
        turn_count = _count_turns(text)
        if min_turns is not None and turn_count < min_turns:
            stats.removed_few_turns += 1
            continue
        if max_turns is not None and turn_count > max_turns:
            stats.removed_many_turns += 1
            continue

        # Check for assistant response
        if require_assistant and not _has_assistant_response(text):
            stats.removed_no_assistant += 1
            continue

        # Custom filter
        if custom_filter is not None and not custom_filter(sample):
            stats.removed_custom += 1
            continue

        filtered.append(sample)

    stats.total_after = len(filtered)
    return filtered, stats


# =============================================================================
# DEDUPLICATION
# =============================================================================

def _get_text_content(sample: Union[Dict, str], key: str = "text") -> str:
    """Extract text content from sample."""
    if isinstance(sample, str):
        return sample
    return sample.get(key, "")


def deduplicate_exact(
    samples: List[Union[Dict, str]],
    key: str = "text",
) -> Tuple[List[Union[Dict, str]], int]:
    """
    Remove exact duplicates from samples.

    Args:
        samples: List of samples
        key: Field to check for duplicates (if samples are dicts)

    Returns:
        Tuple of (deduplicated_samples, num_removed)
    """
    seen = set()
    unique = []

    for sample in samples:
        text = _get_text_content(sample, key)
        text_hash = hash(text)

        if text_hash not in seen:
            seen.add(text_hash)
            unique.append(sample)

    num_removed = len(samples) - len(unique)
    return unique, num_removed


def deduplicate_minhash(
    samples: List[Union[Dict, str]],
    key: str = "text",
    threshold: float = 0.9,
    num_perm: int = 128,
) -> Tuple[List[Union[Dict, str]], int]:
    """
    Remove near-duplicates using MinHash LSH.

    Requires datasketch library: pip install datasketch

    Args:
        samples: List of samples
        key: Field to check for duplicates (if samples are dicts)
        threshold: Jaccard similarity threshold (0-1)
        num_perm: Number of permutation functions

    Returns:
        Tuple of (deduplicated_samples, num_removed)
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        raise ImportError(
            "datasketch required for minhash deduplication: pip install datasketch"
        )

    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Create MinHash for each sample and add to LSH
    minhashes = []
    for i, sample in enumerate(samples):
        text = _get_text_content(sample, key)

        # Create MinHash from character n-grams
        mh = MinHash(num_perm=num_perm)
        for ngram in _get_ngrams(text, n=3):
            mh.update(ngram.encode("utf-8"))

        minhashes.append(mh)

        # Try to insert - if similar document exists, skip
        try:
            lsh.insert(str(i), mh)
        except ValueError:
            # Duplicate detected by LSH
            pass

    # Get unique indices
    unique_indices = set()
    for i, mh in enumerate(minhashes):
        result = lsh.query(mh)
        if result:
            # Keep the first (lowest index) among similar documents
            min_idx = min(int(r) for r in result)
            unique_indices.add(min_idx)

    unique = [samples[i] for i in sorted(unique_indices)]
    num_removed = len(samples) - len(unique)

    return unique, num_removed


def _get_ngrams(text: str, n: int = 3) -> List[str]:
    """Generate character n-grams from text."""
    text = text.lower()
    return [text[i:i+n] for i in range(max(1, len(text) - n + 1))]


# =============================================================================
# PERPLEXITY-BASED FILTERING
# =============================================================================

@dataclass
class PerplexityStats:
    """Statistics from perplexity filtering."""
    total_samples: int
    samples_scored: int
    samples_failed: int
    mean_perplexity: float
    median_perplexity: float
    std_perplexity: float
    min_perplexity: float
    max_perplexity: float
    filtered_count: int
    retained_count: int
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None

    @property
    def retention_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.retained_count / self.total_samples

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Perplexity Filter Results",
            f"=" * 40,
            f"Total samples: {self.total_samples}",
            f"Scored: {self.samples_scored}",
            f"Failed to score: {self.samples_failed}",
            f"",
            f"Perplexity Statistics:",
            f"  Mean: {self.mean_perplexity:.2f}",
            f"  Median: {self.median_perplexity:.2f}",
            f"  Std: {self.std_perplexity:.2f}",
            f"  Range: [{self.min_perplexity:.2f}, {self.max_perplexity:.2f}]",
            f"",
            f"Filtering:",
        ]
        if self.threshold_low is not None:
            lines.append(f"  Low threshold: {self.threshold_low:.2f}")
        if self.threshold_high is not None:
            lines.append(f"  High threshold: {self.threshold_high:.2f}")
        lines.extend([
            f"  Filtered out: {self.filtered_count}",
            f"  Retained: {self.retained_count} ({100 * self.retention_rate:.1f}%)",
        ])
        return "\n".join(lines)


class PerplexityFilter:
    """
    Filter samples by perplexity using model inference.

    Perplexity measures how "surprised" a language model is by a text.
    - Low perplexity: predictable, potentially too simple or repetitive
    - Medium perplexity: natural, typical language
    - High perplexity: unusual, potentially noisy or low-quality

    Usage:
        filter = PerplexityFilter(model_name="gpt2")
        filtered, stats = filter.filter(samples, min_percentile=5, max_percentile=95)

    Or for more control:
        filter = PerplexityFilter(model_name="gpt2")
        scores = filter.score(samples)
        filtered = filter.filter_by_threshold(samples, scores, min_ppl=10, max_ppl=500)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 512,
    ):
        """
        Initialize perplexity filter.

        Args:
            model_name: HuggingFace model name (gpt2, gpt2-medium, etc.)
            device: Device to run on (None = auto-detect)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

        # Determine device
        if device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = device

    def _load_model(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch required for perplexity filtering: "
                "pip install transformers torch"
            )

        logger.info(f"Loading perplexity model: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        )
        self._model.to(self._device)
        self._model.eval()

        logger.info(f"Model loaded on {self._device}")

    def score_text(self, text: str) -> float:
        """
        Compute perplexity for a single text.

        Args:
            text: Text to score

        Returns:
            Perplexity score (lower = more predictable)
        """
        self._load_model()

        import torch

        # Tokenize
        encodings = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        input_ids = encodings.input_ids.to(self._device)

        if input_ids.size(1) < 2:
            # Too short to compute perplexity
            return float("inf")

        # Compute loss
        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            loss = outputs.loss

        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()

        return perplexity

    def score(
        self,
        samples: List[Union[Dict, str]],
        key: str = "text",
        show_progress: bool = True,
    ) -> List[Optional[float]]:
        """
        Compute perplexity scores for all samples.

        Args:
            samples: List of samples
            key: Key to extract text from (if samples are dicts)
            show_progress: Whether to show progress

        Returns:
            List of perplexity scores (None for failed samples)
        """
        self._load_model()

        import torch

        scores = []
        total = len(samples)

        # Process in batches
        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch_texts = []

            for i in range(batch_start, batch_end):
                sample = samples[i]
                text = sample.get(key, "") if isinstance(sample, dict) else str(sample)
                batch_texts.append(text)

            # Score batch
            batch_scores = self._score_batch(batch_texts)
            scores.extend(batch_scores)

            if show_progress:
                pct = 100 * len(scores) / total
                logger.info(f"Perplexity scoring: {len(scores)}/{total} ({pct:.1f}%)")

        return scores

    def _score_batch(self, texts: List[str]) -> List[Optional[float]]:
        """Score a batch of texts."""
        import torch

        scores = []

        for text in texts:
            try:
                if not text or len(text.strip()) < 10:
                    scores.append(None)
                    continue

                score = self.score_text(text)
                scores.append(score if score != float("inf") else None)
            except Exception as e:
                logger.warning(f"Failed to score text: {e}")
                scores.append(None)

        return scores

    def filter(
        self,
        samples: List[Union[Dict, str]],
        key: str = "text",
        min_percentile: Optional[float] = 5.0,
        max_percentile: Optional[float] = 95.0,
        min_perplexity: Optional[float] = None,
        max_perplexity: Optional[float] = None,
        remove_failed: bool = True,
        show_progress: bool = True,
    ) -> Tuple[List[Union[Dict, str]], PerplexityStats]:
        """
        Filter samples by perplexity.

        Can filter by percentile (relative to dataset) or absolute thresholds.
        Percentile-based filtering is recommended as it adapts to the dataset.

        Args:
            samples: List of samples
            key: Key to extract text from
            min_percentile: Remove samples below this percentile (0-100)
            max_percentile: Remove samples above this percentile (0-100)
            min_perplexity: Absolute minimum perplexity (overrides min_percentile)
            max_perplexity: Absolute maximum perplexity (overrides max_percentile)
            remove_failed: Remove samples that failed to score
            show_progress: Show progress during scoring

        Returns:
            Tuple of (filtered_samples, PerplexityStats)
        """
        # Score all samples
        scores = self.score(samples, key=key, show_progress=show_progress)

        # Compute statistics from valid scores
        valid_scores = [s for s in scores if s is not None]

        if not valid_scores:
            logger.warning("No valid perplexity scores computed")
            return samples, PerplexityStats(
                total_samples=len(samples),
                samples_scored=0,
                samples_failed=len(samples),
                mean_perplexity=0.0,
                median_perplexity=0.0,
                std_perplexity=0.0,
                min_perplexity=0.0,
                max_perplexity=0.0,
                filtered_count=0,
                retained_count=len(samples),
            )

        import statistics
        mean_ppl = statistics.mean(valid_scores)
        median_ppl = statistics.median(valid_scores)
        std_ppl = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
        min_ppl = min(valid_scores)
        max_ppl = max(valid_scores)

        # Determine thresholds
        threshold_low = min_perplexity
        threshold_high = max_perplexity

        if threshold_low is None and min_percentile is not None:
            sorted_scores = sorted(valid_scores)
            idx = int(len(sorted_scores) * min_percentile / 100)
            threshold_low = sorted_scores[min(idx, len(sorted_scores) - 1)]

        if threshold_high is None and max_percentile is not None:
            sorted_scores = sorted(valid_scores)
            idx = int(len(sorted_scores) * max_percentile / 100)
            threshold_high = sorted_scores[min(idx, len(sorted_scores) - 1)]

        # Filter
        filtered = []
        filtered_count = 0

        for sample, score in zip(samples, scores):
            # Handle failed scores
            if score is None:
                if remove_failed:
                    filtered_count += 1
                    continue
                else:
                    filtered.append(sample)
                    continue

            # Check thresholds
            if threshold_low is not None and score < threshold_low:
                filtered_count += 1
                continue

            if threshold_high is not None and score > threshold_high:
                filtered_count += 1
                continue

            filtered.append(sample)

        samples_failed = len([s for s in scores if s is None])

        stats = PerplexityStats(
            total_samples=len(samples),
            samples_scored=len(samples) - samples_failed,
            samples_failed=samples_failed,
            mean_perplexity=mean_ppl,
            median_perplexity=median_ppl,
            std_perplexity=std_ppl,
            min_perplexity=min_ppl,
            max_perplexity=max_ppl,
            filtered_count=filtered_count,
            retained_count=len(filtered),
            threshold_low=threshold_low,
            threshold_high=threshold_high,
        )

        return filtered, stats

    def filter_by_threshold(
        self,
        samples: List[Union[Dict, str]],
        scores: List[Optional[float]],
        min_perplexity: Optional[float] = None,
        max_perplexity: Optional[float] = None,
        remove_failed: bool = True,
    ) -> List[Union[Dict, str]]:
        """
        Filter samples using pre-computed scores and absolute thresholds.

        Args:
            samples: List of samples
            scores: Pre-computed perplexity scores
            min_perplexity: Minimum perplexity threshold
            max_perplexity: Maximum perplexity threshold
            remove_failed: Remove samples with None scores

        Returns:
            Filtered samples
        """
        filtered = []

        for sample, score in zip(samples, scores):
            if score is None:
                if not remove_failed:
                    filtered.append(sample)
                continue

            if min_perplexity is not None and score < min_perplexity:
                continue

            if max_perplexity is not None and score > max_perplexity:
                continue

            filtered.append(sample)

        return filtered


def compute_perplexity(
    text: str,
    model_name: str = "gpt2",
    device: Optional[str] = None,
) -> float:
    """
    Compute perplexity for a single text.

    Convenience function for one-off perplexity computation.
    For batch processing, use PerplexityFilter class.

    Args:
        text: Text to score
        model_name: HuggingFace model name
        device: Device to run on

    Returns:
        Perplexity score
    """
    filter_obj = PerplexityFilter(model_name=model_name, device=device)
    return filter_obj.score_text(text)


def filter_by_perplexity(
    samples: List[Union[Dict, str]],
    model_name: str = "gpt2",
    min_percentile: Optional[float] = 5.0,
    max_percentile: Optional[float] = 95.0,
    min_perplexity: Optional[float] = None,
    max_perplexity: Optional[float] = None,
    key: str = "text",
    device: Optional[str] = None,
    batch_size: int = 8,
    show_progress: bool = True,
) -> Tuple[List[Union[Dict, str]], PerplexityStats]:
    """
    Filter samples by perplexity score.

    This is a convenience function. For more control, use PerplexityFilter class.

    Perplexity measures how "surprised" a language model is by a text:
    - Low perplexity: predictable text (may be too simple/repetitive)
    - Medium perplexity: natural text
    - High perplexity: unusual text (may be noisy/low-quality)

    Args:
        samples: List of samples (dicts with "text" key or strings)
        model_name: HuggingFace model for perplexity (default: gpt2)
        min_percentile: Remove samples below this percentile
        max_percentile: Remove samples above this percentile
        min_perplexity: Absolute min threshold (overrides min_percentile)
        max_perplexity: Absolute max threshold (overrides max_percentile)
        key: Key to extract text from samples
        device: Device for model inference (None = auto)
        batch_size: Batch size for scoring
        show_progress: Show progress during scoring

    Returns:
        Tuple of (filtered_samples, PerplexityStats)

    Example:
        # Filter out the 5% most predictable and 5% most unusual samples
        filtered, stats = filter_by_perplexity(samples, min_percentile=5, max_percentile=95)
        print(stats.summary())

        # Filter by absolute thresholds
        filtered, stats = filter_by_perplexity(
            samples,
            min_perplexity=10.0,
            max_perplexity=500.0,
            min_percentile=None,
            max_percentile=None,
        )
    """
    filter_obj = PerplexityFilter(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    return filter_obj.filter(
        samples,
        key=key,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
        min_perplexity=min_perplexity,
        max_perplexity=max_perplexity,
        show_progress=show_progress,
    )


# =============================================================================
# DATASET LOADER
# =============================================================================

class DatasetLoader:
    """
    Unified dataset loader with format detection and validation.

    Usage:
        loader = DatasetLoader("data.jsonl")
        print(loader.detected_format)
        print(loader.validation_report())

        # Get as HuggingFace dataset
        dataset = loader.to_hf_dataset()

        # Or just get ChatML samples
        samples = loader.to_chatml()
    """

    def __init__(
        self,
        source: Union[str, Path, List[Dict]],
        format_type: Optional[DatasetFormat] = None,
        validate: bool = True,
    ):
        """
        Initialize the loader.

        Args:
            source: File path or list of samples
            format_type: Optional format override
            validate: Whether to validate on load
        """
        self.source = source
        self._samples: List[Union[Dict, str]] = []
        self._format: DatasetFormat = format_type or DatasetFormat.UNKNOWN
        self._validation: Optional[ValidationResult] = None
        self._loaded = False

        self._load()
        if validate:
            self._validate()

    def _load(self) -> None:
        """Load samples from source."""
        if isinstance(self.source, list):
            self._samples = self.source
            if self._format == DatasetFormat.UNKNOWN:
                self._format = detect_format(self._samples)
            self._loaded = True
            return

        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        suffix = path.suffix.lower()

        try:
            if suffix == ".jsonl":
                self._samples = self._load_jsonl(path)
            elif suffix == ".json":
                self._samples = self._load_json(path)
            elif suffix in (".txt", ".md"):
                self._samples = self._load_text(path)
            elif suffix == ".parquet":
                self._samples = self._load_parquet(path)
            elif suffix == ".csv":
                self._samples = self._load_csv(path)
            else:
                # Try JSONL
                self._samples = self._load_jsonl(path)

            if self._format == DatasetFormat.UNKNOWN:
                self._format = detect_format(self._samples)

            self._loaded = True

        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}") from e

    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load JSONL file."""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
        return samples

    def _load_json(self, path: Path) -> List[Dict]:
        """Load JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]

    def _load_text(self, path: Path) -> List[str]:
        """Load text file."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            # Split on double newlines for separate samples
            if "\n\n" in content:
                return [s.strip() for s in content.split("\n\n") if s.strip()]
            return [content]

    def _load_parquet(self, path: Path) -> List[Dict]:
        """Load Parquet file."""
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return df.to_dict("records")
        except ImportError:
            raise ImportError("pandas and pyarrow required for parquet: pip install pandas pyarrow")

    def _load_csv(self, path: Path) -> List[Dict]:
        """Load CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_dict("records")
        except ImportError:
            raise ImportError("pandas required for CSV: pip install pandas")

    def _validate(self) -> None:
        """Run validation."""
        self._validation = validate_dataset(self._samples, self._format)

    @property
    def detected_format(self) -> DatasetFormat:
        """Get the detected format."""
        return self._format

    @property
    def samples(self) -> List[Union[Dict, str]]:
        """Get raw samples."""
        return self._samples

    @property
    def is_valid(self) -> bool:
        """Check if dataset is valid."""
        if self._validation is None:
            self._validate()
        return self._validation.is_valid

    @property
    def validation_result(self) -> ValidationResult:
        """Get validation result."""
        if self._validation is None:
            self._validate()
        return self._validation

    def validation_report(self) -> str:
        """Get human-readable validation report."""
        return self.validation_result.summary()

    def to_chatml(self) -> List[Dict[str, str]]:
        """Convert all samples to ChatML format."""
        return convert_to_chatml(self._samples, self._format)

    def to_hf_dataset(self, split: Optional[str] = None):
        """
        Convert to HuggingFace Dataset.

        Args:
            split: Optional split name (e.g., "train", "test")

        Returns:
            datasets.Dataset
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets required: pip install datasets")

        chatml_samples = self.to_chatml()
        dataset = Dataset.from_list(chatml_samples)

        if split:
            return {split: dataset}
        return dataset

    def preview(self, n: int = 3, as_chatml: bool = True) -> List[str]:
        """
        Preview samples.

        Args:
            n: Number of samples to preview
            as_chatml: Whether to show as ChatML

        Returns:
            List of formatted preview strings
        """
        samples = self._samples[:n]

        if as_chatml:
            return [FormatConverter.to_chatml(s, self._format) for s in samples]
        else:
            return [json.dumps(s, indent=2) if isinstance(s, dict) else s for s in samples]

    def stats(self) -> DatasetStats:
        """Get dataset statistics."""
        return get_dataset_stats(self._samples, self._format)

    def shuffle(self, seed: Optional[int] = None) -> "DatasetLoader":
        """Return a new loader with shuffled samples."""
        shuffled = self._samples.copy()
        if seed is not None:
            random.seed(seed)
        random.shuffle(shuffled)
        return DatasetLoader(shuffled, self._format, validate=False)

    def split(
        self,
        train_ratio: float = 0.9,
        seed: Optional[int] = None,
    ) -> Tuple["DatasetLoader", "DatasetLoader"]:
        """Split into train/test loaders."""
        shuffled = self.shuffle(seed)
        n_train = int(len(shuffled._samples) * train_ratio)

        train_loader = DatasetLoader(shuffled._samples[:n_train], self._format, validate=False)
        test_loader = DatasetLoader(shuffled._samples[n_train:], self._format, validate=False)

        return train_loader, test_loader

    def filter(
        self,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_turns: Optional[int] = None,
        max_turns: Optional[int] = None,
        require_assistant: bool = True,
        custom_filter: Optional[Callable[[Dict], bool]] = None,
    ) -> "DatasetLoader":
        """
        Return new loader with filtered samples.

        Converts to ChatML before filtering, then filters based on criteria.

        Args:
            min_tokens: Minimum token count (approximate, None = no limit)
            max_tokens: Maximum token count (approximate, None = no limit)
            min_turns: Minimum conversation turns (None = no limit)
            max_turns: Maximum conversation turns (None = no limit)
            require_assistant: Require at least one assistant response
            custom_filter: Optional callable that returns True to keep sample

        Returns:
            New DatasetLoader with filtered samples
        """
        # Convert to ChatML first
        chatml_samples = self.to_chatml()

        # Apply filter
        filtered, stats = filter_by_quality(
            chatml_samples,
            min_tokens=min_tokens if min_tokens is not None else 0,
            max_tokens=max_tokens if max_tokens is not None else float("inf"),
            min_turns=min_turns if min_turns is not None else 0,
            max_turns=max_turns,
            remove_empty=True,
            require_assistant=require_assistant,
            custom_filter=custom_filter,
        )

        logger.info(f"Filter: {stats.total_before} -> {stats.total_after} samples")

        return DatasetLoader(filtered, DatasetFormat.CHATML, validate=False)

    def deduplicate(
        self,
        method: str = "exact",
        threshold: float = 0.9,
        key: str = "text",
    ) -> "DatasetLoader":
        """
        Return new loader with duplicates removed.

        Args:
            method: Deduplication method ("exact" or "minhash")
            threshold: Similarity threshold for fuzzy methods (0-1)
            key: Field to deduplicate on

        Returns:
            New DatasetLoader with duplicates removed
        """
        # Convert to ChatML first
        chatml_samples = self.to_chatml()

        if method == "exact":
            deduped, num_removed = deduplicate_exact(chatml_samples, key=key)
        elif method == "minhash":
            deduped, num_removed = deduplicate_minhash(
                chatml_samples, key=key, threshold=threshold
            )
        else:
            raise ValueError(f"Unknown deduplication method: {method}")

        logger.info(f"Deduplicate ({method}): removed {num_removed} duplicates")

        return DatasetLoader(deduped, DatasetFormat.CHATML, validate=False)

    def filter_perplexity(
        self,
        model_name: str = "gpt2",
        min_percentile: Optional[float] = 5.0,
        max_percentile: Optional[float] = 95.0,
        min_perplexity: Optional[float] = None,
        max_perplexity: Optional[float] = None,
        device: Optional[str] = None,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> Tuple["DatasetLoader", "PerplexityStats"]:
        """
        Return new loader with samples filtered by perplexity.

        Perplexity measures how "surprised" a language model is by a text:
        - Low perplexity: very predictable (may be too simple/repetitive)
        - Medium perplexity: natural text
        - High perplexity: unusual (may be noisy/low-quality)

        Args:
            model_name: HuggingFace model for perplexity (gpt2, gpt2-medium, etc.)
            min_percentile: Remove samples below this percentile (0-100)
            max_percentile: Remove samples above this percentile (0-100)
            min_perplexity: Absolute min threshold (overrides min_percentile)
            max_perplexity: Absolute max threshold (overrides max_percentile)
            device: Device for inference (None = auto)
            batch_size: Batch size for scoring
            show_progress: Show progress during scoring

        Returns:
            Tuple of (new DatasetLoader with filtered samples, PerplexityStats)

        Example:
            # Remove outliers (top/bottom 5%)
            loader = DatasetLoader("data.jsonl")
            filtered_loader, stats = loader.filter_perplexity(
                min_percentile=5,
                max_percentile=95,
            )
            print(stats.summary())
        """
        # Convert to ChatML first
        chatml_samples = self.to_chatml()

        # Filter by perplexity
        filtered, stats = filter_by_perplexity(
            chatml_samples,
            model_name=model_name,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
            min_perplexity=min_perplexity,
            max_perplexity=max_perplexity,
            device=device,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        logger.info(f"Perplexity filter: {stats.total_samples} -> {stats.retained_count} samples")

        new_loader = DatasetLoader(filtered, DatasetFormat.CHATML, validate=False)
        return new_loader, stats

    @classmethod
    def from_streaming(
        cls,
        source: str,
        buffer_size: int = 1000,
        split: Optional[str] = None,
    ) -> "StreamingDatasetLoader":
        """
        Load dataset in streaming mode for large files.

        Args:
            source: HuggingFace dataset name or file path
            buffer_size: Number of samples to buffer
            split: Dataset split to use (e.g., "train")

        Returns:
            StreamingDatasetLoader instance
        """
        return StreamingDatasetLoader(source, buffer_size=buffer_size, split=split)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Union[Dict, str]:
        return self._samples[idx]

    def __iter__(self):
        return iter(self._samples)


# =============================================================================
# STREAMING DATASET LOADER
# =============================================================================

class StreamingDatasetLoader:
    """
    Streaming dataset loader for large files.

    Yields samples one at a time without loading everything into memory.
    Works with HuggingFace datasets in streaming mode or local files.

    Usage:
        # From HuggingFace
        loader = StreamingDatasetLoader("HuggingFaceH4/ultrachat_200k", split="train_sft")
        for sample in loader.take(1000):
            print(sample)

        # From local file
        loader = StreamingDatasetLoader("large_data.jsonl")
        for batch in loader.batches(100):
            process(batch)
    """

    def __init__(
        self,
        source: str,
        buffer_size: int = 1000,
        split: Optional[str] = None,
        format_type: Optional[DatasetFormat] = None,
    ):
        """
        Initialize streaming loader.

        Args:
            source: HuggingFace dataset name or file path
            buffer_size: Number of samples to buffer for operations
            split: Dataset split to use (for HF datasets)
            format_type: Optional format override
        """
        self.source = source
        self.buffer_size = buffer_size
        self.split = split
        self._format = format_type or DatasetFormat.UNKNOWN
        self._iterator = None
        self._is_hf_dataset = False

        # Detect if this is a HuggingFace dataset or local file
        path = Path(source)
        self._is_hf_dataset = not path.exists()

    def _create_iterator(self):
        """Create the underlying iterator."""
        if self._is_hf_dataset:
            return self._stream_hf_dataset()
        else:
            return self._stream_local_file()

    def _stream_hf_dataset(self):
        """Stream from HuggingFace dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets required: pip install datasets")

        dataset = load_dataset(self.source, split=self.split, streaming=True)

        # Detect format from first sample
        first_sample = None
        for sample in dataset:
            first_sample = sample
            if self._format == DatasetFormat.UNKNOWN:
                self._format = detect_format(sample)
            yield sample
            break

        # Yield rest of samples
        for sample in dataset:
            yield sample

    def _stream_local_file(self):
        """Stream from local file."""
        path = Path(self.source)
        suffix = path.suffix.lower()

        if suffix == ".jsonl":
            yield from self._stream_jsonl(path)
        elif suffix == ".json":
            yield from self._stream_json(path)
        elif suffix in (".txt", ".md"):
            yield from self._stream_text(path)
        else:
            # Try JSONL
            yield from self._stream_jsonl(path)

    def _stream_jsonl(self, path: Path):
        """Stream JSONL file."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    if self._format == DatasetFormat.UNKNOWN:
                        self._format = detect_format(sample)
                    yield sample
                except json.JSONDecodeError:
                    continue

    def _stream_json(self, path: Path):
        """Stream JSON array file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for sample in data:
                    if self._format == DatasetFormat.UNKNOWN:
                        self._format = detect_format(sample)
                    yield sample
            else:
                yield data

    def _stream_text(self, path: Path):
        """Stream text file."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if self._format == DatasetFormat.UNKNOWN:
                self._format = detect_format(content)
            if "\n\n" in content:
                for chunk in content.split("\n\n"):
                    if chunk.strip():
                        yield chunk.strip()
            else:
                yield content

    def __iter__(self):
        """Iterate over all samples."""
        return self._create_iterator()

    def take(self, n: int) -> List[Union[Dict, str]]:
        """
        Take first n samples.

        Args:
            n: Number of samples to take

        Returns:
            List of samples
        """
        samples = []
        for i, sample in enumerate(self):
            if i >= n:
                break
            samples.append(sample)
        return samples

    def skip(self, n: int):
        """
        Skip first n samples and return iterator for rest.

        Args:
            n: Number of samples to skip

        Yields:
            Samples after the first n
        """
        for i, sample in enumerate(self):
            if i >= n:
                yield sample

    def batches(self, batch_size: int):
        """
        Yield samples in batches.

        Args:
            batch_size: Number of samples per batch

        Yields:
            Lists of samples
        """
        batch = []
        for sample in self:
            batch.append(sample)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def to_chatml(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Convert samples to ChatML format.

        Args:
            n: Number of samples to convert (None = all)

        Returns:
            List of ChatML formatted samples
        """
        samples = self.take(n) if n is not None else list(self)
        return convert_to_chatml(samples, self._format)

    def filter(
        self,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_turns: Optional[int] = None,
        max_turns: Optional[int] = None,
        require_assistant: bool = True,
        custom_filter: Optional[Callable[[Dict], bool]] = None,
    ):
        """
        Yield filtered samples.

        Args:
            min_tokens: Minimum token count
            max_tokens: Maximum token count
            min_turns: Minimum turns
            max_turns: Maximum turns
            require_assistant: Require assistant response
            custom_filter: Custom filter function

        Yields:
            Filtered samples
        """
        for sample in self:
            # Convert to ChatML for consistent filtering
            chatml = FormatConverter.to_chatml(sample, self._format)
            text = chatml if isinstance(chatml, str) else chatml

            # Check empty
            if not text.strip():
                continue

            # Check token count
            token_count = _count_tokens_approx(text)
            if min_tokens is not None and token_count < min_tokens:
                continue
            if max_tokens is not None and token_count > max_tokens:
                continue

            # Check turn count
            turn_count = _count_turns(text)
            if min_turns is not None and turn_count < min_turns:
                continue
            if max_turns is not None and turn_count > max_turns:
                continue

            # Check for assistant response
            if require_assistant and not _has_assistant_response(text):
                continue

            # Custom filter
            if custom_filter is not None and not custom_filter(sample):
                continue

            yield {"text": chatml}

    @property
    def detected_format(self) -> DatasetFormat:
        """Get the detected format."""
        return self._format


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def preview_samples(
    source: Union[str, Path, List[Dict]],
    n: int = 3,
    as_chatml: bool = True,
) -> List[str]:
    """
    Quick preview of dataset samples.

    Args:
        source: File path or samples
        n: Number to preview
        as_chatml: Convert to ChatML

    Returns:
        List of preview strings
    """
    loader = DatasetLoader(source, validate=False)
    return loader.preview(n, as_chatml)


def get_dataset_stats(
    samples: List[Union[Dict, str]],
    format_type: Optional[DatasetFormat] = None,
) -> DatasetStats:
    """
    Compute statistics for a dataset.

    Args:
        samples: List of samples
        format_type: Optional format hint

    Returns:
        DatasetStats with computed statistics
    """
    if not samples:
        return DatasetStats(
            total_samples=0,
            total_tokens_approx=0,
            avg_tokens_per_sample=0,
            min_tokens=0,
            max_tokens=0,
            format_detected=DatasetFormat.UNKNOWN,
            has_system_prompts=False,
            avg_turns_per_conversation=0,
            unique_system_prompts=0,
        )

    if format_type is None:
        format_type = detect_format(samples)

    # Convert to ChatML for consistent analysis
    chatml_samples = convert_to_chatml(samples, format_type)

    # Approximate token counts (4 chars ≈ 1 token)
    token_counts = []
    system_prompts = set()
    turn_counts = []

    for sample in chatml_samples:
        text = sample.get("text", "")
        tokens = len(text) // 4
        token_counts.append(tokens)

        # Count turns
        turns = text.count("<|im_start|>")
        turn_counts.append(turns)

        # Extract system prompts
        system_match = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", text, re.DOTALL)
        if system_match:
            system_prompts.add(system_match.group(1).strip())

    return DatasetStats(
        total_samples=len(samples),
        total_tokens_approx=sum(token_counts),
        avg_tokens_per_sample=sum(token_counts) / len(token_counts) if token_counts else 0,
        min_tokens=min(token_counts) if token_counts else 0,
        max_tokens=max(token_counts) if token_counts else 0,
        format_detected=format_type,
        has_system_prompts=len(system_prompts) > 0,
        avg_turns_per_conversation=sum(turn_counts) / len(turn_counts) if turn_counts else 0,
        unique_system_prompts=len(system_prompts),
    )


# =============================================================================
# CURRICULUM LEARNING (Phase 3.3)
# =============================================================================

def compute_difficulty_score(
    sample: Union[Dict, str],
    key: str = "text",
) -> float:
    """
    Compute difficulty score for a sample.

    Higher score = more difficult. Factors:
    - Token length (longer = harder)
    - Vocabulary complexity (more unique words = harder)
    - Average word length (proxy for vocabulary complexity)

    Args:
        sample: Sample dict or string
        key: Key to extract text from (if sample is dict)

    Returns:
        Difficulty score (0.0 to 1.0)
    """
    text = _get_text_content(sample, key)
    if not text:
        return 0.0

    # Length score (normalized by ~5000 chars)
    length_score = min(len(text) / 5000, 1.0)

    # Word-level complexity
    words = text.split()
    if not words:
        return length_score

    unique_words = set(w.lower() for w in words)
    vocab_ratio = len(unique_words) / len(words)

    # Average word length (proxy for vocabulary complexity)
    avg_word_len = sum(len(w) for w in words) / len(words)
    word_complexity = min(avg_word_len / 10, 1.0)

    # Combine scores (weighted)
    score = (length_score * 0.5) + (vocab_ratio * 0.25) + (word_complexity * 0.25)

    return min(max(score, 0.0), 1.0)


def order_by_difficulty(
    samples: List[Union[Dict, str]],
    key: str = "text",
    ascending: bool = True,
) -> List[Union[Dict, str]]:
    """
    Order samples by difficulty for curriculum learning.

    Args:
        samples: List of samples
        key: Key to extract text from (if samples are dicts)
        ascending: If True, easy samples first (recommended for training)

    Returns:
        Reordered list of samples

    Example:
        # Order easy to hard for curriculum learning
        ordered = order_by_difficulty(samples)

        # Or hard to easy
        ordered = order_by_difficulty(samples, ascending=False)
    """
    # Compute difficulty scores
    scored = [(sample, compute_difficulty_score(sample, key)) for sample in samples]

    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=not ascending)

    return [sample for sample, _ in scored]


def get_curriculum_chunks(
    samples: List[Union[Dict, str]],
    num_chunks: int = 5,
    key: str = "text",
) -> List[List[Union[Dict, str]]]:
    """
    Split samples into curriculum chunks (easy to hard).

    Useful for multi-run training where you want:
    - Run 1: Easy examples
    - Run 2: Medium-easy
    - ...
    - Run N: Hardest examples

    Args:
        samples: List of samples
        num_chunks: Number of difficulty chunks
        key: Key to extract text from

    Returns:
        List of sample chunks, ordered easy to hard

    Example:
        chunks = get_curriculum_chunks(samples, num_chunks=5)
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {len(chunk)} samples")
            trainer.train(chunk, steps=100)
    """
    # Order by difficulty
    ordered = order_by_difficulty(samples, key=key, ascending=True)

    # Split into chunks
    chunk_size = len(ordered) // num_chunks
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        if i == num_chunks - 1:
            # Last chunk gets remaining samples
            end = len(ordered)
        else:
            end = start + chunk_size
        chunks.append(ordered[start:end])

    return chunks


@dataclass
class CurriculumStats:
    """Statistics from curriculum ordering."""
    total_samples: int
    num_chunks: int
    chunk_sizes: List[int]
    difficulty_ranges: List[Tuple[float, float]]  # (min, max) per chunk

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Curriculum Learning Stats",
            f"=" * 40,
            f"Total samples: {self.total_samples}",
            f"Chunks: {self.num_chunks}",
            "",
            "Difficulty distribution:",
        ]
        for i, (size, (d_min, d_max)) in enumerate(zip(self.chunk_sizes, self.difficulty_ranges)):
            lines.append(f"  Chunk {i+1}: {size} samples, difficulty [{d_min:.3f} - {d_max:.3f}]")
        return "\n".join(lines)


def analyze_curriculum(
    samples: List[Union[Dict, str]],
    num_chunks: int = 5,
    key: str = "text",
) -> CurriculumStats:
    """
    Analyze curriculum distribution without reordering.

    Args:
        samples: List of samples
        num_chunks: Number of chunks to analyze
        key: Key to extract text from

    Returns:
        CurriculumStats with distribution info
    """
    # Get scores
    scores = [compute_difficulty_score(s, key) for s in samples]

    # Sort indices by score
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])

    # Compute chunk stats
    chunk_size = len(samples) // num_chunks
    chunk_sizes = []
    difficulty_ranges = []

    for i in range(num_chunks):
        start = i * chunk_size
        if i == num_chunks - 1:
            end = len(sorted_indices)
        else:
            end = start + chunk_size

        chunk_indices = sorted_indices[start:end]
        chunk_scores = [scores[j] for j in chunk_indices]

        chunk_sizes.append(len(chunk_indices))
        difficulty_ranges.append((min(chunk_scores), max(chunk_scores)))

    return CurriculumStats(
        total_samples=len(samples),
        num_chunks=num_chunks,
        chunk_sizes=chunk_sizes,
        difficulty_ranges=difficulty_ranges,
    )
