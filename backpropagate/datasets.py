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
import random
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

from .exceptions import (
    BackpropagateError,
    DatasetError,
    DatasetNotFoundError,
    DatasetParseError,
    InvalidSettingError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# B-017 HUGGINGFACE HUB TRANSIENT RETRY (mirror of trainer._retry_hf_call)
# =============================================================================
# Wraps load_dataset and from_pretrained calls with exponential backoff so a
# transient 5xx / 429 / connection blip on the Hub doesn't kill a long job.

_HF_RETRY_ATTEMPTS = 3
_HF_RETRY_BASE_SECONDS = 5
_HF_RETRY_MAX_SECONDS = 60
_HF_RETRY_MULTIPLIER = 2


def _hf_transient_exceptions() -> tuple[type[BaseException], ...]:
    """Candidate exception classes; status-code filtering in
    ``_is_transient_hf_exception`` decides whether to actually retry."""
    excs: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import requests  # type: ignore

        excs.extend([
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
        ])
    except ImportError:
        pass
    try:
        from huggingface_hub.utils import HfHubHTTPError  # type: ignore

        excs.append(HfHubHTTPError)
    except ImportError:
        pass
    return tuple(excs)


def _is_transient_hf_exception(exc: BaseException) -> bool:
    """Retry only on 429 / 5xx / connection / timeout — never on 401/403/404."""
    transient_excs = _hf_transient_exceptions()
    if not isinstance(exc, transient_excs):
        return False
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        return True
    return bool(status == 429 or status >= 500)


def _retry_hf_call(
    fn: Callable[..., Any],
    *args: Any,
    _label: str = "hf_call",
    **kwargs: Any,
) -> Any:
    """B-017: invoke ``fn`` with tenacity-based retry on transient HF errors."""
    from tenacity import (
        before_sleep_log,
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential,
    )

    transient_excs = _hf_transient_exceptions()

    @retry(
        stop=stop_after_attempt(_HF_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=_HF_RETRY_MULTIPLIER,
            min=_HF_RETRY_BASE_SECONDS,
            max=_HF_RETRY_MAX_SECONDS,
        ),
        retry=retry_if_exception(_is_transient_hf_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call() -> Any:
        return fn(*args, **kwargs)

    try:
        return _call()
    except transient_excs as exc:
        logger.error(
            f"HF transient retry exhausted: label={_label} "
            f"err={type(exc).__name__}: {exc}"
        )
        raise


def _dtype_kwarg(dtype: Any) -> dict[str, Any]:
    """DATA-B-003: build the load-dtype kwarg for ``from_pretrained`` using
    the name the installed transformers version expects.

    transformers renamed ``torch_dtype`` → ``dtype`` and deprecated the old
    name (it logs ``"torch_dtype is deprecated! Use dtype instead!"`` on
    every model load on >= 4.56, and is slated for removal in 5.0+). We pick
    the modern ``dtype`` key on >= 5.0 and fall back to ``torch_dtype`` on
    older releases that predate the rename, so the same call site is quiet on
    both. Detection failures degrade to the legacy key (always accepted —
    just deprecated — on every version that still ships it).
    """
    try:
        import transformers
        from packaging import version

        ver = version.parse(transformers.__version__.split("+")[0])
        if ver >= version.parse("5.0.0"):
            return {"dtype": dtype}
    except Exception:  # noqa: BLE001 — never let version detection break load
        pass  # nosec B110 — fall through to the legacy kwarg key below
    return {"torch_dtype": dtype}


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
    value: Any | None = None

    def __str__(self) -> str:
        return f"Row {self.row_index}: [{self.error_type}] {self.field} - {self.message}"


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    total_rows: int
    valid_rows: int
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
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
            "Dataset Validation Report",
            "=" * 40,
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
            "Filter Results",
            "=" * 40,
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

def detect_format(data: dict | list[dict | str] | str) -> DatasetFormat:
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
            with open(file_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))

        elif suffix == ".json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data[:sample_size]
                else:
                    samples = [data]

        elif suffix in (".txt", ".md"):
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                return detect_format(content)

        else:
            # Try JSONL first
            with open(file_path, encoding="utf-8") as f:
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

def _render_function_call(function_call: Any) -> str:
    """Render an OpenAI ``function_call`` as readable text for ChatML.

    Prefers a structured ``name(arguments)`` rendering over a raw ``str(dict)``
    so the converted training text reads like a tool invocation rather than a
    Python dict repr (DATA-A-003).
    """
    if isinstance(function_call, dict):
        name = function_call.get("name", "")
        arguments = function_call.get("arguments", "")
        if name:
            if arguments:
                return f"[Function call: {name}({arguments})]"
            return f"[Function call: {name}]"
    return f"[Function call: {function_call}]"


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
    def sharegpt_to_chatml(sample: dict) -> str:
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
    def alpaca_to_chatml(sample: dict) -> str:
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
    def openai_to_chatml(sample: dict) -> str:
        """Convert OpenAI chat format to ChatML."""
        messages = sample.get("messages", [])
        parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""

            # Handle function calls (OpenAI format). Preserve any natural-language
            # content the assistant produced alongside the call rather than
            # overwriting it (DATA-A-003): an assistant turn may carry both a
            # reply and a function_call, and dropping the reply silently trains
            # the model on a raw dict repr instead of the real answer.
            if msg.get("function_call"):
                call_repr = _render_function_call(msg["function_call"])
                content = f"{content}\n{call_repr}" if content else call_repr

            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        return "\n".join(parts)

    @staticmethod
    def raw_to_chatml(text: str, default_role: str = "user") -> str:
        """Convert raw text to ChatML."""
        # Simple conversion - treat as single user message
        return f"<|im_start|>{default_role}\n{text}<|im_end|>"

    @classmethod
    def to_chatml(cls, sample: dict | str, format_type: DatasetFormat) -> str:
        """Convert any format to ChatML."""
        if format_type == DatasetFormat.CHATML:
            if isinstance(sample, dict):
                return str(sample.get("text", ""))
            return sample

        if format_type == DatasetFormat.SHAREGPT:
            if not isinstance(sample, dict):
                raise ValueError(f"ShareGPT format requires dict, got {type(sample)}")
            return cls.sharegpt_to_chatml(sample)

        if format_type == DatasetFormat.ALPACA:
            if not isinstance(sample, dict):
                raise ValueError(f"Alpaca format requires dict, got {type(sample)}")
            return cls.alpaca_to_chatml(sample)

        if format_type == DatasetFormat.OPENAI:
            if not isinstance(sample, dict):
                raise ValueError(f"OpenAI format requires dict, got {type(sample)}")
            return cls.openai_to_chatml(sample)

        if format_type == DatasetFormat.RAW_TEXT:
            text = sample if isinstance(sample, str) else sample.get("text", "")
            return cls.raw_to_chatml(text)

        raise ValueError(f"Cannot convert format: {format_type}")


# Matches a ChatML turn header and its body up to the next <|im_end|>.
_CHATML_TURN_RE = re.compile(
    r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL
)


def _warn_on_empty_turns(chatml: str, row_index: int) -> None:
    """Emit a WARN when a converted sample yields an empty user or assistant body.

    Silent blank turns mean the model trains on empty prompts/answers — usually
    the symptom of a per-row format mismatch (DATA-A-001). Surfacing it keeps the
    content loss visible instead of training on blanks.
    """
    for role, body in _CHATML_TURN_RE.findall(chatml):
        if role in ("user", "assistant") and not body.strip():
            logger.warning(
                "Converted sample %d produced an empty %s turn "
                "(possible format mismatch — verify the source row's format)",
                row_index,
                role,
            )


def convert_to_chatml(
    samples: list[dict | str],
    source_format: DatasetFormat | None = None,
) -> list[dict[str, str]]:
    """
    Convert a list of samples to ChatML format.

    Args:
        samples: List of samples in any supported format. When ``source_format``
            is omitted, each sample's format is detected individually so a mixed
            file (e.g. an Alpaca-first file with a stray ShareGPT row) does not
            silently convert later rows to empty turns (DATA-A-001).
        source_format: Optional format hint. If provided, it is applied to every
            sample (auto-detected per-sample when not provided).

    Returns:
        List of dicts with "text" key containing ChatML
    """
    if not samples:
        return []

    # Per-sample detection when no explicit format is given: a single file may
    # carry rows in different formats, and detecting once from samples[0] would
    # convert every non-matching row to blank turns (silent content loss).
    per_sample_detect = source_format is None or source_format == DatasetFormat.UNKNOWN

    results = []
    dropped = 0
    for i, sample in enumerate(samples):
        # Structured so mypy narrows source_format to a concrete DatasetFormat
        # in the pinned-format branch (per_sample_detect ⇒ source_format is
        # None/UNKNOWN, so the else here is always a real format).
        if source_format is not None and not per_sample_detect:
            fmt = source_format
        else:
            fmt = detect_format(sample)
        try:
            chatml = FormatConverter.to_chatml(sample, fmt)
            _warn_on_empty_turns(chatml, i)
            results.append({"text": chatml})
        except Exception as e:
            # V2-b: DATA-A-001's fix left the convert-FAILS sub-case
            # unprotected — a malformed / UNKNOWN-format row was dropped
            # silently with no row index and no count. Thread the row index
            # ``i`` (matching the empty-turn WARN's visibility) and emit a
            # final dropped-count summary so wholesale content loss can't pass
            # unnoticed.
            dropped += 1
            logger.warning("Failed to convert sample %d: %s", i, e)
            continue

    if dropped:
        logger.warning(
            "convert_to_chatml dropped %d/%d sample(s) that failed to "
            "convert (see the per-row warnings above for the offending "
            "indices). The output has fewer rows than the input.",
            dropped,
            len(samples),
        )

    return results


# =============================================================================
# VALIDATION
# =============================================================================

def _validate_chatml(text: str, row_index: int) -> list[ValidationError]:
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


def _validate_sharegpt(sample: dict, row_index: int) -> list[ValidationError]:
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


def _validate_alpaca(sample: dict, row_index: int) -> list[ValidationError]:
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

    # Check for empty values.
    # DATA-B-002: a CSV/parquet source with an empty cell yields a None (or,
    # pre-coercion, a float NaN) for that field. ``sample.get(k, "")`` returns
    # the present-but-None value (the "" default only fires for ABSENT keys),
    # so a bare ``.strip()`` raised ``AttributeError: 'NoneType'/'float' object
    # has no attribute 'strip'`` and crashed validation of an otherwise-loadable
    # dataset. Treat any non-string (None, NaN, numeric) as empty content.
    def _is_blank(value: Any) -> bool:
        return not isinstance(value, str) or value.strip() == ""

    if _is_blank(sample.get("instruction")):
        errors.append(ValidationError(
            row_index=row_index,
            field="instruction",
            error_type="empty_content",
            message="Empty instruction",
        ))

    if _is_blank(sample.get("output")):
        errors.append(ValidationError(
            row_index=row_index,
            field="output",
            error_type="empty_content",
            message="Empty output",
        ))

    return errors


def _validate_openai(sample: dict, row_index: int) -> list[ValidationError]:
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
    sample: dict | str,
    row_index: int,
    format_type: DatasetFormat,
) -> list[ValidationError]:
    """Validate a single sample."""
    if format_type == DatasetFormat.CHATML:
        text = sample if isinstance(sample, str) else sample.get("text", "")
        return _validate_chatml(text, row_index)

    if format_type == DatasetFormat.SHAREGPT:
        if not isinstance(sample, dict):
            return [ValidationError(
                row_index=row_index,
                field="sample",
                error_type="invalid_type",
                message=f"ShareGPT format requires dict, got {type(sample).__name__}",
            )]
        return _validate_sharegpt(sample, row_index)

    if format_type == DatasetFormat.ALPACA:
        if not isinstance(sample, dict):
            return [ValidationError(
                row_index=row_index,
                field="sample",
                error_type="invalid_type",
                message=f"Alpaca format requires dict, got {type(sample).__name__}",
            )]
        return _validate_alpaca(sample, row_index)

    if format_type == DatasetFormat.OPENAI:
        if not isinstance(sample, dict):
            return [ValidationError(
                row_index=row_index,
                field="sample",
                error_type="invalid_type",
                message=f"OpenAI format requires dict, got {type(sample).__name__}",
            )]
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
    samples: list[dict | str],
    format_type: DatasetFormat | None = None,
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

    # V1-c / DATA-A-001 (validator side): when no format is pinned, detect
    # per-sample so a mixed file (e.g. an Alpaca-first file with a stray
    # ShareGPT row) is validated against each row's ACTUAL format instead of
    # the first row's. Validating every row against samples[0]'s format
    # produced false "errors" for the minority-format rows and masked the
    # real per-row problems. When a format IS pinned, apply it to every row
    # exactly as before.
    per_sample_detect = format_type is None
    # ``format_detected`` preserves the prior return contract (the single
    # detected format); for a mixed file it reports the first row's format.
    reported_format = format_type if format_type is not None else detect_format(samples[0])

    all_errors = []
    all_warnings = []
    valid_count = 0

    for i, sample in enumerate(samples):
        if format_type is not None and not per_sample_detect:
            row_format = format_type
        else:
            row_format = detect_format(sample)
        errors = validate_sample(sample, i, row_format)

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
        format_detected=reported_format,
    )


# =============================================================================
# QUALITY FILTERING
# =============================================================================

def _count_tokens_approx(text: str) -> int:
    """Approximate token count (4 chars ≈ 1 token).

    Stage C amend BACKEND-B-024: the 4-chars-per-token heuristic is
    calibrated for ASCII English. CJK datasets are roughly 1 char per
    token (so this heuristic over-counts by ~4x — the dataset will appear
    longer than it really is and ``min_tokens`` floors will be MUCH
    stricter than intended). Code datasets are closer to 2-3 chars per
    token. Operators filtering by ``min_tokens`` / ``max_tokens`` on a
    non-ASCII-English corpus should re-derive the cutoffs against their
    real tokenizer; v1.4 will accept an optional ``token_counter``
    callable to plug in ``tokenizer.encode`` directly. Until then, this
    function is a fast-path approximation only.
    """
    return len(text) // 4


def _count_turns(text: str) -> int:
    """Count conversation turns in ChatML text."""
    return text.count("<|im_start|>")


def _has_assistant_response(text: str) -> bool:
    """Check if text contains an assistant response."""
    return "<|im_start|>assistant" in text


def filter_by_quality(
    samples: list[dict],
    min_tokens: int = 50,
    max_tokens: int = 4096,
    min_turns: int = 2,
    max_turns: int | None = None,
    remove_empty: bool = True,
    require_assistant: bool = True,
    custom_filter: Callable[[dict], bool] | None = None,
) -> tuple[list[dict], FilterStats]:
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

    # DATA-A-005: the default min_tokens=50 combined with the coarse
    # 4-chars-per-token heuristic (_count_tokens_approx) can silently drop
    # an entire corpus of legitimately short or non-ASCII (e.g. CJK, where
    # the heuristic over-counts ~4x) chats. A run that filters to zero — or
    # near-zero — is almost always a mis-calibrated threshold, not a genuinely
    # empty dataset. Surface it loudly with the actionable cause + knob so the
    # operator doesn't discover the empty training set the hard way.
    if stats.total_before > 0 and stats.total_after == 0:
        logger.warning(
            "filter_by_quality removed ALL %d samples (0 retained). The "
            "most common cause is min_tokens=%s being too high for short or "
            "non-ASCII (e.g. CJK) content — the token estimate is ~4 chars/"
            "token and over-counts CJK ~4x. Lower min_tokens / raise "
            "max_tokens, or re-derive the cutoffs against your real "
            "tokenizer. Breakdown: too_short=%d too_long=%d few_turns=%d "
            "many_turns=%d no_assistant=%d empty=%d custom=%d.",
            stats.total_before,
            min_tokens,
            stats.removed_too_short,
            stats.removed_too_long,
            stats.removed_few_turns,
            stats.removed_many_turns,
            stats.removed_no_assistant,
            stats.removed_empty,
            stats.removed_custom,
        )
    elif stats.total_before >= 20 and stats.total_after / stats.total_before < 0.05:
        # Near-total wipe-out on a non-trivial input (kept < 5% of >= 20 rows).
        # Same likely cause; warn but don't imply a hard failure.
        logger.warning(
            "filter_by_quality retained only %d/%d samples (%.1f%%). If this "
            "is unexpected, check min_tokens=%s against your content length / "
            "tokenizer (the 4-chars/token estimate over-counts CJK ~4x).",
            stats.total_after,
            stats.total_before,
            100.0 * stats.total_after / stats.total_before,
            min_tokens,
        )

    return filtered, stats


# =============================================================================
# DEDUPLICATION
# =============================================================================

def _get_text_content(sample: dict | str, key: str = "text") -> str:
    """Extract text content from sample."""
    if isinstance(sample, str):
        return sample
    return str(sample.get(key, ""))


def deduplicate_exact(
    samples: list[dict | str],
    key: str = "text",
) -> tuple[list[dict | str], int]:
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
    samples: list[dict | str],
    key: str = "text",
    threshold: float = 0.9,
    num_perm: int = 128,
) -> tuple[list[dict | str], int]:
    """
    Remove near-duplicates using MinHash LSH.

    Requires datasketch library: pip install datasketch

    Stage C amend BACKEND-B-007: deterministic, single-pass algorithm.
    The prior implementation interleaved insert + query which made the
    dedup result depend on insertion order in a subtle way: the LSH
    ``insert`` raised ``ValueError`` on near-duplicates without indexing
    them, then the follow-up ``lsh.query(mh)`` for those skipped documents
    returned an empty result set, so they were quietly dropped from
    ``unique_indices`` even when they should have been the canonical
    representative for their cluster. Net effect: re-running the same
    dataset twice could yield different sample counts on near-duplicate
    boundaries.

    The fix is the canonical pattern: (1) build every MinHash up front
    without inserting, (2) iterate samples in document order, (3) for each
    sample ask LSH whether any already-inserted document matches; if not,
    insert + keep this one as the canonical representative; if yes, skip.
    This is deterministic given a deterministic ``num_perm`` seed (the
    datasketch default uses a fixed seed for permutations, so MinHash
    values are reproducible across runs).

    Determinism contract:
        ``MinHashLSH(threshold, num_perm)`` is deterministic for a given
        ``(threshold, num_perm)`` pair. The MinHash permutations are
        seeded from a fixed default in the datasketch library. Calling
        this function twice on the same ``samples`` list yields byte-
        identical ``unique`` output. Operators who shuffle ``samples``
        between calls will get different (but deterministic) outputs —
        sort or seed the upstream shuffle if you need stability across
        pipeline runs.

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

    # Stage C BACKEND-B-008 humanization: surface the expected memory
    # footprint before the dedup pass for large datasets. Each MinHash
    # holds ~``num_perm`` 64-bit ints plus Python object overhead, ~1KB
    # per MinHash at num_perm=128. A 1M-sample dataset thus costs ~1GB
    # just for the hashes; on a 16GB RAM workstation that can OOM the
    # host process well before training starts. We log at INFO for big
    # batches and WARN at the "this almost certainly won't fit" line.
    sample_count = len(samples)
    estimated_mb = sample_count * num_perm * 8 / (1024 * 1024)
    if sample_count >= 100_000:
        # The 100K-sample threshold is the "this is going to take a
        # noticeable amount of time and memory; the operator deserves a
        # heads-up" line. 1GB-of-MinHashes territory.
        logger.info(
            "deduplicate_minhash: estimated %.0fMB working memory "
            "for %d samples at num_perm=%d (MinHash storage; LSH index "
            "adds ~30%% on top). For dataset sizes >> 1M consider "
            "shuffling + chunking the dataset and running dedup per "
            "chunk if RAM is tight.",
            estimated_mb,
            sample_count,
            num_perm,
        )
    if sample_count * num_perm > 100_000_000:
        # Heuristic: when MinHash storage alone clears ~800MB we're in
        # workstation-RAM trouble. Warn loud + name the operator's
        # options (lower num_perm trades dedup precision for memory;
        # chunked dedup trades correctness-at-cluster-boundaries for
        # memory). Pre-fix this OOMed silently.
        logger.warning(
            "deduplicate_minhash: %d samples at num_perm=%d will "
            "allocate ~%.0fMB just for the MinHash array. This may "
            "exhaust RAM on a typical 16GB workstation. To proceed, "
            "either lower num_perm (e.g. 64 — reduces precision near "
            "the threshold cutoff), shuffle + chunk the dataset and "
            "run dedup per chunk, or run on a larger-RAM machine. To "
            "skip dedup entirely, pass dedup=False at the caller.",
            sample_count, num_perm, estimated_mb,
        )

    # Stage C amend BACKEND-B-007: phase 1 — build every MinHash up front.
    # No insert side effects in this pass so the per-sample minhash is
    # purely a function of its text. DATA-A-006: track which rows have no
    # grams (empty / whitespace-only content). An all-empty MinHash matches
    # every other all-empty MinHash, so without this flag every blank row
    # would collapse into one — we instead keep them verbatim.
    minhashes: list[Any] = []
    empty_content: list[bool] = []
    for sample in samples:
        text = _get_text_content(sample, key)
        grams = _get_ngrams(text, n=3)
        mh = MinHash(num_perm=num_perm)
        for ngram in grams:
            mh.update(ngram.encode("utf-8"))
        minhashes.append(mh)
        # Whitespace-only content is "empty" for dedup purposes too: a row
        # of all spaces would otherwise hash to one space-gram and collapse
        # with every other whitespace-only row.
        empty_content.append(not text.strip())

    # Stage C amend BACKEND-B-007: phase 2 — single-pass canonical dedup.
    # For each sample in document order, ask LSH whether any
    # already-inserted document is a near-duplicate. If not, this sample
    # IS the canonical representative for its cluster; insert + keep.
    # If yes, skip.
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept_indices: list[int] = []
    for i, mh in enumerate(minhashes):
        if empty_content[i]:
            # DATA-A-006: empty-content rows have no comparable signature;
            # never query/insert (which would fold them all together) — keep
            # each one as-is and let downstream quality filters drop blanks.
            kept_indices.append(i)
            continue
        matches = lsh.query(mh)
        if matches:
            # Near-duplicate of an already-kept canonical — skip.
            continue
        # No prior match — keep this one as the canonical representative
        # for its cluster.
        try:
            lsh.insert(str(i), mh)
        except ValueError as exc:
            # Defensive: LSH should never raise here because the query
            # above returned empty. If it does, we'd silently drop the
            # sample under the old code path; instead, log loud and keep
            # the sample (consistency over discard).
            logger.warning(
                f"deduplicate_minhash: LSH insert raised on index {i} "
                f"despite empty query result — keeping sample (degraded "
                f"to first-seen). Underlying error: {exc}"
            )
        kept_indices.append(i)

    unique = [samples[i] for i in kept_indices]
    num_removed = len(samples) - len(unique)

    return unique, num_removed


def _get_ngrams(text: str, n: int = 3) -> list[str]:
    """Generate character n-grams from text.

    DATA-A-006: the prior ``range(max(1, len(text) - n + 1))`` form
    degenerated badly on short / empty inputs:

    * Empty text produced ``['']`` — a SINGLE empty-string gram. Every
      empty / whitespace-only row therefore hashed identically, so MinHash
      LSH collapsed all of them into one "duplicate" cluster (a force
      multiplier with DATA-A-001's blank-conversion rows). We now return
      ``[]`` for empty text; callers (``deduplicate_minhash``) treat a
      no-gram row as having no comparable content and keep it as-is rather
      than folding distinct empties together.
    * Text shorter than ``n`` yielded one truncated gram via the
      ``max(1, ...)`` floor. That is still distinct per input, so we keep
      returning the whole short string as a single gram — but we do it
      explicitly instead of relying on the ``max(1, ...)`` clamp, so the
      intent (and the empty-text special case) is legible.
    """
    text = text.lower()
    if not text:
        # No content -> no grams. The dedup caller keeps these rows instead
        # of collapsing every empty row into a single representative.
        return []
    if len(text) < n:
        # Too short for a full n-gram: use the whole (distinct) string once.
        return [text]
    return [text[i:i + n] for i in range(len(text) - n + 1)]


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
    threshold_low: float | None = None
    threshold_high: float | None = None

    @property
    def retention_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.retained_count / self.total_samples

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "Perplexity Filter Results",
            "=" * 40,
            f"Total samples: {self.total_samples}",
            f"Scored: {self.samples_scored}",
            f"Failed to score: {self.samples_failed}",
            "",
            "Perplexity Statistics:",
            f"  Mean: {self.mean_perplexity:.2f}",
            f"  Median: {self.median_perplexity:.2f}",
            f"  Std: {self.std_perplexity:.2f}",
            f"  Range: [{self.min_perplexity:.2f}, {self.max_perplexity:.2f}]",
            "",
            "Filtering:",
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
        device: str | None = None,
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
        self._model: Any = None
        self._tokenizer: Any = None

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

        # B-017: retry on transient HF Hub failures (5xx, 429, timeouts).
        self._tokenizer = _retry_hf_call(
            AutoTokenizer.from_pretrained,
            self.model_name,
            _label=f"perplexity_tokenizer:{self.model_name}",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # DATA-B-003: transformers renamed torch_dtype -> dtype (deprecated on
        # 4.56+, removed on 5.x). Pick the kwarg the installed version wants.
        _dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = _retry_hf_call(
            AutoModelForCausalLM.from_pretrained,
            self.model_name,
            _label=f"perplexity_model:{self.model_name}",
            **_dtype_kwarg(_dtype),
        )
        self._model.to(self._device)
        self._model.eval()

        logger.info(f"Model loaded on {self._device}")

    def unload(self) -> None:
        """Stage C amend BACKEND-B-011: release the perplexity model from
        VRAM/RAM.

        ``PerplexityFilter`` lazy-loads the scoring model on first use and
        keeps the reference indefinitely. For a 1.3GB GPT-2 model on a
        constrained-VRAM rig, this competes with the main training model
        for VRAM — calling ``loader.filter_perplexity(...)`` then
        ``trainer.train(...)`` could OOM at training start because GPT-2
        weights were still resident.

        ``unload()`` drops both model + tokenizer, runs ``gc.collect()``,
        and calls ``torch.cuda.empty_cache()`` if CUDA is available. Safe
        to call multiple times; safe to call when the model was never
        loaded. The filter can be re-used after unload — the next score
        call will lazy-reload.

        For deterministic cleanup, prefer the context-manager form:

            with PerplexityFilter("gpt2") as pf:
                filtered, _ = pf.filter(samples)
            # model is released here
        """
        if self._model is None and self._tokenizer is None:
            return
        logger.info(
            f"PerplexityFilter.unload: releasing model={self.model_name!r} "
            f"from {self._device}"
        )
        try:
            del self._model
        except Exception:  # noqa: BLE001
            pass  # nosec B110 — cleanup; missing attr is fine
        try:
            del self._tokenizer
        except Exception:  # noqa: BLE001
            pass  # nosec B110 — cleanup; missing attr is fine
        self._model = None
        self._tokenizer = None
        import gc as _gc
        _gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except ImportError:
            pass
        except Exception as exc:
            logger.debug(f"PerplexityFilter.unload: empty_cache skipped: {exc}")

    def __enter__(self) -> "PerplexityFilter":
        """Stage C amend BACKEND-B-011: context-manager support.

        Enables ``with PerplexityFilter(...) as pf:`` for automatic
        ``unload()`` on exit — the recommended pattern when running
        perplexity scoring before fine-tuning to avoid competing for VRAM
        with the training model.
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Stage C amend BACKEND-B-011: auto-unload on context-manager exit."""
        self.unload()

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
        samples: list[dict | str],
        key: str = "text",
        show_progress: bool = True,
    ) -> list[float | None]:
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

        scores: list[float | None] = []
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

    def _score_batch(self, texts: list[str]) -> list[float | None]:
        """Score a batch of texts."""
        scores: list[float | None] = []

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
        samples: list[dict | str],
        key: str = "text",
        min_percentile: float | None = 5.0,
        max_percentile: float | None = 95.0,
        min_perplexity: float | None = None,
        max_perplexity: float | None = None,
        remove_failed: bool = True,
        show_progress: bool = True,
    ) -> tuple[list[dict | str], PerplexityStats]:
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
        # Stage C amend BACKEND-B-020: fail loud on inverted percentile /
        # threshold bounds. Pre-fix, ``min_percentile > max_percentile``
        # silently produced ``threshold_low > threshold_high`` which
        # filtered out nearly every sample — the operator saw an empty
        # result with no explanation. Same for absolute thresholds. The
        # message names the bad value and the fix.
        if (
            min_percentile is not None
            and max_percentile is not None
            and min_percentile > max_percentile
        ):
            raise InvalidSettingError(
                "min_percentile",
                min_percentile,
                f"value less than max_percentile (got "
                f"min_percentile={min_percentile}, "
                f"max_percentile={max_percentile})",
                suggestion=(
                    "Swap the two arguments — min_percentile is the LOW "
                    "cutoff (drop samples scoring below this percentile) "
                    "and max_percentile is the HIGH cutoff (drop samples "
                    "scoring above)."
                ),
            )
        if (
            min_perplexity is not None
            and max_perplexity is not None
            and min_perplexity > max_perplexity
        ):
            raise InvalidSettingError(
                "min_perplexity",
                min_perplexity,
                f"value less than max_perplexity (got "
                f"min_perplexity={min_perplexity}, "
                f"max_perplexity={max_perplexity})",
                suggestion=(
                    "Swap the two arguments — min_perplexity is the LOW "
                    "cutoff (drop samples easier than this) and "
                    "max_perplexity is the HIGH cutoff (drop samples "
                    "harder than this)."
                ),
            )

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
        samples: list[dict | str],
        scores: list[float | None],
        min_perplexity: float | None = None,
        max_perplexity: float | None = None,
        remove_failed: bool = True,
    ) -> list[dict | str]:
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
    device: str | None = None,
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
    samples: list[dict | str],
    model_name: str = "gpt2",
    min_percentile: float | None = 5.0,
    max_percentile: float | None = 95.0,
    min_perplexity: float | None = None,
    max_perplexity: float | None = None,
    key: str = "text",
    device: str | None = None,
    batch_size: int = 8,
    show_progress: bool = True,
) -> tuple[list[dict | str], PerplexityStats]:
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
        source: str | Path | list[dict | str],
        format_type: DatasetFormat | None = None,
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
        self._samples: list[dict[Any, Any] | str] = []
        self._format: DatasetFormat = format_type or DatasetFormat.UNKNOWN
        # Track whether the caller pinned a format. When they did NOT, conversion
        # detects per-sample so a mixed file's later rows aren't converted with
        # the first row's format (DATA-A-001 sibling).
        self._format_explicit = format_type is not None
        self._validation: ValidationResult | None = None
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
            raise DatasetNotFoundError(
                str(path),
                suggestion="Check the file path. Supported formats: .jsonl, .json, .txt, .md, .parquet, .csv",
            )

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

        except BackpropagateError:
            # DATA-A-011: the per-format loaders already raise structured
            # errors (e.g. DatasetParseError with INPUT_DATASET_PARSE_FAILED
            # + a remediation hint). Re-wrapping those in a bare ValueError
            # discarded the stable error code / suggestion that callers and
            # the CLI rely on. Let any BackpropagateError subclass propagate
            # unchanged; only genuinely unexpected (non-structured) failures
            # fall through to the generic wrapper below.
            raise
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}") from e

    def _load_jsonl(self, path: Path) -> list[dict[Any, Any] | str]:
        """Load JSONL file.

        Stage C BACKEND-B-015 humanization: pre-fix this emitted one
        ``logger.warning`` per bad line, which on a corrupt 1M-line file
        produces 500K WARN log lines and blows stderr buffers / log
        shipping. We now log the FIRST few per-line warnings verbatim (so
        the operator gets concrete diagnostic data on which lines failed
        and why), then suppress the rest while tracking the count. The
        post-loop summary line names the total skip count, the percentage,
        and the actionable next step. The function still raises
        DatasetParseError when 100% of non-empty lines fail (already
        load-bearing — preserved).
        """
        # Threshold for verbose per-line WARN. The first N failures emit a
        # full diagnostic line (line number + JSON error); subsequent
        # failures are counted but silent until the post-loop summary.
        # Tuned for "operator gets enough signal to find the bad rows" but
        # "tail -f never floods" — 20 is large enough to capture a
        # repeating pattern, small enough to fit in a single screen of
        # output.
        _VERBOSE_WARN_CEILING = 20

        samples: list[dict[Any, Any] | str] = []
        total_lines = 0
        skipped_lines = 0
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    skipped_lines += 1
                    if skipped_lines <= _VERBOSE_WARN_CEILING:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    elif skipped_lines == _VERBOSE_WARN_CEILING + 1:
                        # One-shot transition line so the operator knows
                        # subsequent failures are being counted but
                        # suppressed. Same shape as the curl/apt patterns.
                        logger.warning(
                            "JSONL parse: more than %d invalid lines in %s; "
                            "suppressing per-line warnings — final count "
                            "will be reported in the load summary.",
                            _VERBOSE_WARN_CEILING,
                            path.name,
                        )

        if total_lines > 0 and not samples:
            raise DatasetParseError(
                f"All {total_lines} non-empty lines in {path.name} failed JSON parsing. "
                "File may be corrupted or not in JSONL format.",
                path=str(path),
                suggestion=(
                    "Inspect the first failing line above for the JSON "
                    "decode error (unescaped quotes, BOM, wrong encoding, "
                    "or non-JSONL format are common causes). If the file "
                    "is one big JSON object/array, save it as .json (not "
                    ".jsonl) and reload — DatasetLoader auto-detects the "
                    "extension. To convert a CSV / parquet source into "
                    "JSONL, see the conversion examples in "
                    "handbook/recipes.md."
                ),
            )

        if total_lines > 0 and skipped_lines > total_lines * 0.5:
            # Stage C BACKEND-B-015 humanization: name the operator's next
            # step — verify encoding / format before training proceeds on
            # the surviving rows. Pre-fix the warning was diagnostic-only
            # ("high parse failure rate") with no actionable hint.
            logger.warning(
                "High JSONL parse failure rate: %d/%d lines (%d%%) failed in %s. "
                "Training will proceed on the %d surviving samples — verify "
                "this is the dataset you intended. Common causes: file is a "
                "single JSON document (rename .jsonl → .json), wrong encoding "
                "(re-save as UTF-8 without BOM), or trailing-comma / "
                "unescaped-quote rows that need cleanup. See "
                "handbook/troubleshooting.md for the JSONL recovery recipe.",
                skipped_lines,
                total_lines,
                skipped_lines * 100 // total_lines,
                path.name,
                len(samples),
            )
        elif skipped_lines > _VERBOSE_WARN_CEILING:
            # When skip count is below the 50%-failure ceiling but did
            # exceed the verbose-WARN ceiling, surface a single summary
            # so the suppressed-per-line count isn't invisible.
            logger.warning(
                "JSONL load: %d/%d lines failed parsing in %s; "
                "kept %d samples.",
                skipped_lines,
                total_lines,
                path.name,
                len(samples),
            )

        return samples

    def _load_json(self, path: Path) -> list[dict[Any, Any] | str]:
        """Load JSON file.

        DATA-B-001: the ``.jsonl`` path (``_load_jsonl``) gives a malformed
        file the full structured ``DatasetParseError`` treatment; the
        single-document ``.json`` path used a bare ``json.load`` whose
        ``JSONDecodeError`` surfaced as an opaque traceback with no recovery
        hint. Mirror the jsonl contract: catch the decode error, attach the
        offending byte position via ``line_number``, and point the operator
        at the common causes (trailing comma, BOM, wrong extension).
        """
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise DatasetParseError(
                    f"Failed to parse {path.name} as JSON: {e.msg}",
                    path=str(path),
                    line_number=e.lineno,
                    suggestion=(
                        "A .json file must be a single valid JSON document "
                        "(object or array). Common causes: a trailing comma, "
                        "an unescaped quote, a UTF-8 BOM, or a "
                        "newline-delimited file saved with a .json extension "
                        "(rename it .jsonl — DatasetLoader auto-detects the "
                        "extension). See handbook/troubleshooting.md."
                    ),
                ) from e
            if isinstance(data, list):
                return cast(list[dict[Any, Any] | str], data)
            return [data]

    def _load_text(self, path: Path) -> list[dict[Any, Any] | str]:
        """Load text file."""
        with open(path, encoding="utf-8") as f:
            content = f.read()
            # Split on double newlines for separate samples
            if "\n\n" in content:
                return [s.strip() for s in content.split("\n\n") if s.strip()]
            return [content]

    def _records_from_df(self, df: Any, source_name: str) -> list[dict[Any, Any] | str]:
        """DATA-B-002: convert a pandas DataFrame to records with NaN coerced
        to ``None``.

        Pandas represents an empty CSV/parquet cell as the float ``nan``.
        ``df.to_dict("records")`` then leaves those ``nan`` floats in place,
        so a downstream ``str(value)`` turns an empty cell into the literal
        three-character training token ``'nan'``, and a numeric filter (e.g.
        a min-length check that calls ``len``) raises ``TypeError`` on the
        float. Replacing NaN with ``None`` before ``to_dict`` yields a clean
        ``None`` the converters already skip. We also surface a single WARN
        naming the affected columns + counts so a silently-sparse source is
        visible (e.g. a header typo that produced an all-empty column).
        """
        try:
            null_counts = df.isna().sum()
            offenders = {
                str(col): int(n) for col, n in null_counts.items() if n > 0
            }
        except Exception:  # noqa: BLE001 — diagnostics only; never block load
            offenders = {}
        if offenders:
            logger.warning(
                "%s: %d column(s) contain empty/NaN cells (%s); these become "
                "None (skipped by the converters), not the literal string "
                "'nan'. Verify your column headers if this is unexpected.",
                source_name,
                len(offenders),
                ", ".join(f"{c}={n}" for c, n in sorted(offenders.items())),
            )
        # Coerce NaN -> None. The naive ``df.where(df.notna(), None)`` does
        # NOT work on modern pandas (>= 2.x): with the StringDtype/object
        # columns a read_csv produces, ``where`` re-introduces the float NaN
        # and the cell survives as ``nan``. Casting to ``object`` first makes
        # the substitution stick across both string and numeric columns.
        clean = df.astype(object).where(df.notna(), None)
        return cast(list[dict[Any, Any] | str], clean.to_dict("records"))

    def _load_parquet(self, path: Path) -> list[dict[Any, Any] | str]:
        """Load Parquet file.

        DATA-B-006: parquet needs BOTH pandas and a parquet engine
        (pyarrow). A missing engine doesn't fail at ``import pandas`` — it
        fails inside ``read_parquet`` with an ``ImportError`` whose message
        only the well-read recognize. Probe pyarrow explicitly and raise the
        structured :class:`DatasetError` (with the missing-dep code) so the
        operator gets a single actionable hint either way.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise DatasetError(
                "pandas and pyarrow are required to load parquet files.",
                suggestion="Install them: pip install pandas pyarrow",
                code="DEP_DATASET_ENGINE_MISSING",
                cause=e,
            ) from e
        try:
            import pyarrow  # noqa: F401
        except ImportError as e:
            raise DatasetError(
                "A parquet engine (pyarrow) is required to load parquet files.",
                suggestion="Install it: pip install pyarrow",
                code="DEP_DATASET_ENGINE_MISSING",
                cause=e,
            ) from e
        df = pd.read_parquet(path)
        return self._records_from_df(df, path.name)

    def _load_csv(self, path: Path) -> list[dict[Any, Any] | str]:
        """Load CSV file."""
        try:
            import pandas as pd
        except ImportError as e:
            raise DatasetError(
                "pandas is required to load CSV files.",
                suggestion="Install it: pip install pandas",
                code="DEP_DATASET_ENGINE_MISSING",
                cause=e,
            ) from e
        df = pd.read_csv(path)
        return self._records_from_df(df, path.name)

    def _validate(self) -> None:
        """Run validation.

        V1-c: this was the 5th (and last) ``self._format`` consumer left on
        the raw-format path after Wave A1 migrated the other four to the
        per-sample ``self._format if self._format_explicit else None``
        pattern. When the caller did NOT pin a format, pass ``None`` so the
        validator detects format per-sample instead of validating a mixed
        file against a possibly-wrong cached ``self._format`` (the same
        first-row assumption DATA-A-001 fixed for conversion).
        """
        fmt = self._format if self._format_explicit else None
        self._validation = validate_dataset(self._samples, fmt)

    @property
    def detected_format(self) -> DatasetFormat:
        """Get the detected format."""
        return self._format

    @property
    def samples(self) -> list[dict | str]:
        """Get raw samples."""
        return self._samples

    @property
    def is_valid(self) -> bool:
        """Check if dataset is valid."""
        if self._validation is None:
            self._validate()
        assert self._validation is not None
        return self._validation.is_valid

    @property
    def validation_result(self) -> ValidationResult:
        """Get validation result."""
        if self._validation is None:
            self._validate()
        assert self._validation is not None
        return self._validation

    def validation_report(self) -> str:
        """Get human-readable validation report."""
        return self.validation_result.summary()

    def to_chatml(self) -> list[dict[str, str]]:
        """Convert all samples to ChatML format."""
        fmt = self._format if self._format_explicit else None
        return convert_to_chatml(self._samples, fmt)

    def to_hf_dataset(self, split: str | None = None) -> Any:
        """
        Convert to HuggingFace Dataset.

        Args:
            split: Optional split name (e.g., "train", "test")

        Returns:
            datasets.Dataset or dict with split key
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

    def preview(self, n: int = 3, as_chatml: bool = True) -> list[str]:
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
            # Detect per-sample when the caller didn't pin a format so a mixed
            # file previews each row in its own format (DATA-A-001 sibling).
            return [
                FormatConverter.to_chatml(
                    s, self._format if self._format_explicit else detect_format(s)
                )
                for s in samples
            ]
        else:
            return [json.dumps(s, indent=2) if isinstance(s, dict) else s for s in samples]

    def stats(self) -> DatasetStats:
        """Get dataset statistics."""
        # Pass None when the format was auto-detected so get_dataset_stats detects
        # per-sample for a mixed file (DATA-A-001 sibling).
        fmt = self._format if self._format_explicit else None
        return get_dataset_stats(self._samples, fmt)

    def shuffle(self, seed: int | None = None) -> "DatasetLoader":
        """Return a new loader with shuffled samples."""
        shuffled = self._samples.copy()
        # Use a local Random instance instead of mutating the global Python RNG;
        # the global mutation pattern silently pollutes any other code that uses
        # `random` later in the same process (e.g. MinHash dedup seeds, user
        # callbacks). When no seed is provided, fall back to the module-level
        # random.shuffle so we still get a non-deterministic shuffle without
        # touching the global seed.
        if seed is not None:
            # nosec B311 — deterministic dataset shuffle, not crypto. Local
            # random.Random per B-002 (no global seed mutation).
            rng = random.Random(seed)  # nosec B311 — non-crypto dataset shuffle; see comment above
            rng.shuffle(shuffled)
        else:
            random.shuffle(shuffled)
        return DatasetLoader(shuffled, self._format, validate=False)

    def split(
        self,
        train_ratio: float = 0.9,
        seed: int | None = None,
    ) -> tuple["DatasetLoader", "DatasetLoader"]:
        """Split into train/test loaders."""
        shuffled = self.shuffle(seed)
        n_train = int(len(shuffled._samples) * train_ratio)

        train_loader = DatasetLoader(shuffled._samples[:n_train], self._format, validate=False)
        test_loader = DatasetLoader(shuffled._samples[n_train:], self._format, validate=False)

        return train_loader, test_loader

    def filter(
        self,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        min_turns: int | None = None,
        max_turns: int | None = None,
        require_assistant: bool = True,
        custom_filter: Callable[[dict], bool] | None = None,
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
            max_tokens=max_tokens if max_tokens is not None else 2**31,
            min_turns=min_turns if min_turns is not None else 0,
            max_turns=max_turns,
            remove_empty=True,
            require_assistant=require_assistant,
            custom_filter=custom_filter,
        )

        logger.info(f"Filter: {stats.total_before} -> {stats.total_after} samples")

        return DatasetLoader(
            cast(list[dict[Any, Any] | str], filtered),
            DatasetFormat.CHATML,
            validate=False,
        )

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
        chatml_samples: list[dict[Any, Any] | str] = cast(
            list[dict[Any, Any] | str], self.to_chatml()
        )

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
        min_percentile: float | None = 5.0,
        max_percentile: float | None = 95.0,
        min_perplexity: float | None = None,
        max_perplexity: float | None = None,
        device: str | None = None,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> tuple["DatasetLoader", "PerplexityStats"]:
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
        chatml_samples: list[dict[Any, Any] | str] = cast(
            list[dict[Any, Any] | str], self.to_chatml()
        )

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
    def from_local(
        cls,
        path: str | Path,
        format_type: DatasetFormat | None = None,
        validate: bool = True,
    ) -> "DatasetLoader":
        """
        Load dataset from a local file (JSONL, JSON, CSV, Parquet).

        Convenience method for loading local datasets with clear semantics.

        Args:
            path: Path to local file
            format_type: Optional format override (auto-detected if not provided)
            validate: Whether to validate on load

        Returns:
            DatasetLoader instance

        Example:
            >>> loader = DatasetLoader.from_local("F:/AI/data/perfect_pairs_chat.jsonl")
            >>> print(f"Loaded {len(loader)} samples")
            >>> dataset = loader.to_hf_dataset()
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {path}")
        return cls(path, format_type=format_type, validate=validate)

    @classmethod
    def from_streaming(
        cls,
        source: str,
        buffer_size: int = 1000,
        split: str | None = None,
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

    def __getitem__(self, idx: int) -> dict | str:
        return self._samples[idx]

    def __iter__(self) -> Iterator[dict[Any, Any] | str]:
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
        split: str | None = None,
        format_type: DatasetFormat | None = None,
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
        # When the caller did not pin a format, convert/filter detect per-sample
        # so a mixed stream's rows aren't all coerced to the first row's format
        # (DATA-A-001 sibling).
        self._format_explicit = format_type is not None
        self._iterator = None
        self._is_hf_dataset = False

        # Detect if this is a HuggingFace dataset or local file
        path = Path(source)
        self._is_hf_dataset = not path.exists()

    def _create_iterator(self) -> Iterator[dict[Any, Any] | str]:
        """Create the underlying iterator."""
        if self._is_hf_dataset:
            return self._stream_hf_dataset()
        else:
            return self._stream_local_file()

    def _stream_hf_dataset(self) -> Iterator[dict[Any, Any] | str]:
        """Stream from HuggingFace dataset.

        B-017: ``load_dataset`` is wrapped with the HF transient-retry
        decorator. The streaming iterator itself is NOT retried (mid-stream
        retries would replay samples and break determinism); only the
        initial connection is retried.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets required: pip install datasets")

        dataset = _retry_hf_call(
            load_dataset,
            self.source,
            split=self.split,
            streaming=True,
            _label=f"streaming_load_dataset:{self.source}",
        )

        # DATA-B-010: iterate the streaming dataset exactly ONCE. The prior
        # two-loop shape (a first loop that `break`s after one sample, then a
        # second `for sample in dataset`) re-entered ``__iter__`` on an
        # IterableDataset — which restarts the stream — so row 0 was yielded
        # twice ([0, 0, 1, 2, ...]). A single loop with a first-iteration
        # flag detects the format off the first sample without replaying it.
        first = True
        for sample in dataset:
            if first:
                if self._format == DatasetFormat.UNKNOWN:
                    self._format = detect_format(sample)
                first = False
            yield sample

    def _stream_local_file(self) -> Iterator[dict[Any, Any] | str]:
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

    def _stream_jsonl(self, path: Path) -> Iterator[dict[Any, Any] | str]:
        """Stream JSONL file.

        DATA-A-010: the prior body silently ``continue``d on every
        ``JSONDecodeError`` with no count, no log, and no raise. A file that
        was entirely malformed (wrong delimiter, a JSON array saved with a
        ``.jsonl`` suffix, a corrupt export) streamed ZERO samples and the
        operator got a silently-empty training run. We now mirror the
        non-streaming ``_load_jsonl`` contract: log the first few failures
        verbatim (capped so a corrupt 1M-line file can't flood stderr),
        count the rest, emit a post-stream summary, and raise
        ``DatasetParseError`` when every non-empty line failed.
        """
        _VERBOSE_WARN_CEILING = 20

        total_lines = 0
        skipped_lines = 0
        yielded = 0
        with open(path, encoding="utf-8") as f:
            for line_num, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    skipped_lines += 1
                    if skipped_lines <= _VERBOSE_WARN_CEILING:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    elif skipped_lines == _VERBOSE_WARN_CEILING + 1:
                        logger.warning(
                            "JSONL stream: more than %d invalid lines in %s; "
                            "suppressing per-line warnings — final count will "
                            "be reported in the stream summary.",
                            _VERBOSE_WARN_CEILING,
                            path,
                        )
                    continue
                if self._format == DatasetFormat.UNKNOWN:
                    self._format = detect_format(sample)
                yielded += 1
                yield sample

        # Every non-empty line failed to decode -> the stream produced nothing
        # usable. Raise rather than return an empty iterator silently.
        if total_lines > 0 and yielded == 0:
            raise DatasetParseError(
                f"All {total_lines} non-empty line(s) in {path} failed to "
                "parse as JSON; the stream produced zero samples.",
                path=str(path),
                suggestion=(
                    "Verify the file is newline-delimited JSON (one JSON "
                    "object per line). A JSON array saved with a .jsonl "
                    "suffix is the most common cause — use a .json extension "
                    "for array files."
                ),
            )
        if skipped_lines:
            pct = 100.0 * skipped_lines / total_lines if total_lines else 0.0
            logger.warning(
                "JSONL stream summary: skipped %d/%d non-empty line(s) "
                "(%.1f%%) in %s due to JSON parse errors.",
                skipped_lines,
                total_lines,
                pct,
                path,
            )

    def _stream_json(self, path: Path) -> Iterator[dict[Any, Any] | str]:
        """Stream JSON array file.

        DATA-B-001: mirror the structured ``DatasetParseError`` that
        ``_stream_jsonl`` raises on a fully-malformed file. ``json.load`` on
        a corrupt ``.json`` previously raised a bare ``JSONDecodeError`` mid
        ``__iter__`` with no recovery hint.
        """
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise DatasetParseError(
                    f"Failed to parse {path} as JSON: {e.msg}",
                    path=str(path),
                    line_number=e.lineno,
                    suggestion=(
                        "A .json file must be a single valid JSON document "
                        "(object or array). A newline-delimited file saved "
                        "with a .json extension is the most common cause — "
                        "rename it .jsonl so it streams line-by-line."
                    ),
                ) from e
            if isinstance(data, list):
                for sample in data:
                    if self._format == DatasetFormat.UNKNOWN:
                        self._format = detect_format(sample)
                    yield sample
            else:
                yield data

    def _stream_text(self, path: Path) -> Iterator[dict[Any, Any] | str]:
        """Stream text file."""
        with open(path, encoding="utf-8") as f:
            content = f.read()
            if self._format == DatasetFormat.UNKNOWN:
                self._format = detect_format(content)
            if "\n\n" in content:
                for chunk in content.split("\n\n"):
                    if chunk.strip():
                        yield chunk.strip()
            else:
                yield content

    def __iter__(self) -> Iterator[dict[Any, Any] | str]:
        """Iterate over all samples."""
        return self._create_iterator()

    def take(self, n: int) -> list[dict | str]:
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

    def skip(self, n: int) -> Iterator[dict[Any, Any] | str]:
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

    def batches(self, batch_size: int) -> Iterator[list[dict[Any, Any] | str]]:
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

    def to_chatml(self, n: int | None = None) -> list[dict[str, str]]:
        """
        Convert samples to ChatML format.

        Args:
            n: Number of samples to convert (None = all)

        Returns:
            List of ChatML formatted samples
        """
        samples = self.take(n) if n is not None else list(self)
        fmt = self._format if self._format_explicit else None
        return convert_to_chatml(samples, fmt)

    def filter(
        self,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        min_turns: int | None = None,
        max_turns: int | None = None,
        require_assistant: bool = True,
        custom_filter: Callable[[dict], bool] | None = None,
    ) -> Iterator[dict[str, str]]:
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
            # Convert to ChatML for consistent filtering. Detect per-sample when
            # the caller didn't pin a format so a mixed stream isn't coerced to
            # the first row's format (DATA-A-001 sibling).
            fmt = self._format if self._format_explicit else detect_format(sample)
            # FormatConverter.to_chatml always returns a str (see its
            # signature), so the prior ``chatml if isinstance(chatml, str)
            # else chatml`` ternary (DATA-A-010) was a no-op — both branches
            # returned ``chatml``. Assign directly.
            text = FormatConverter.to_chatml(sample, fmt)

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
            if custom_filter is not None and isinstance(sample, dict) and not custom_filter(sample):
                continue

            yield {"text": text}

    @property
    def detected_format(self) -> DatasetFormat:
        """Get the detected format."""
        return self._format


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def preview_samples(
    source: str | Path | list[dict[Any, Any] | str],
    n: int = 3,
    as_chatml: bool = True,
) -> list[str]:
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
    samples: list[dict | str],
    format_type: DatasetFormat | None = None,
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

    # When no format is given, detect once for the reported `format_detected`
    # summary but let convert_to_chatml detect per-sample so a mixed file's later
    # rows are not converted with the wrong format (DATA-A-001 sibling).
    auto_detected = format_type is None
    # Narrow on the value (not the bool alias) so mypy knows format_type is a
    # concrete DatasetFormat below (the `format_detected` summary field).
    if format_type is None:
        format_type = detect_format(samples)

    # Convert to ChatML for consistent analysis
    chatml_samples = convert_to_chatml(samples, None if auto_detected else format_type)

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
    sample: dict | str,
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

    unique_words = {w.lower() for w in words}
    vocab_ratio = len(unique_words) / len(words)

    # Average word length (proxy for vocabulary complexity)
    avg_word_len = sum(len(w) for w in words) / len(words)
    word_complexity = min(avg_word_len / 10, 1.0)

    # Combine scores (weighted)
    score = (length_score * 0.5) + (vocab_ratio * 0.25) + (word_complexity * 0.25)

    return min(max(score, 0.0), 1.0)


def order_by_difficulty(
    samples: list[dict | str],
    key: str = "text",
    ascending: bool = True,
) -> list[dict | str]:
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
    samples: list[dict | str],
    num_chunks: int = 5,
    key: str = "text",
) -> list[list[dict | str]]:
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
    # Guard against a non-positive chunk count: num_chunks=0 (or negative) would
    # raise a raw ZeroDivisionError below (DATA-A-002). Clamp to at least one
    # chunk; the more-chunks-than-samples case is already handled by integer
    # division producing trailing empty chunks.
    num_chunks = max(1, num_chunks)

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
    chunk_sizes: list[int]
    difficulty_ranges: list[tuple[float, float]]  # (min, max) per chunk

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "Curriculum Learning Stats",
            "=" * 40,
            f"Total samples: {self.total_samples}",
            f"Chunks: {self.num_chunks}",
            "",
            "Difficulty distribution:",
        ]
        for i, (size, (d_min, d_max)) in enumerate(zip(self.chunk_sizes, self.difficulty_ranges)):
            lines.append(f"  Chunk {i+1}: {size} samples, difficulty [{d_min:.3f} - {d_max:.3f}]")
        return "\n".join(lines)


def analyze_curriculum(
    samples: list[dict | str],
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
    # Empty input: report empty stats rather than crashing on the division /
    # min()/max()-on-empty paths below (DATA-A-002).
    if not samples:
        return CurriculumStats(
            total_samples=0,
            num_chunks=0,
            chunk_sizes=[],
            difficulty_ranges=[],
        )

    # Clamp to a sane chunk count: num_chunks=0 would raise ZeroDivisionError and
    # num_chunks > len(samples) would leave empty chunks whose min()/max() blow up
    # (DATA-A-002). Bounding to [1, len(samples)] guarantees every chunk is
    # non-empty. The effective count is what we report in the returned stats so
    # chunk_sizes / difficulty_ranges stay consistent (see CurriculumStats.summary).
    effective_chunks = max(1, min(num_chunks, len(samples)))

    # Get scores
    scores = [compute_difficulty_score(s, key) for s in samples]

    # Sort indices by score
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])

    # Compute chunk stats
    chunk_size = len(samples) // effective_chunks
    chunk_sizes = []
    difficulty_ranges = []

    for i in range(effective_chunks):
        start = i * chunk_size
        if i == effective_chunks - 1:
            end = len(sorted_indices)
        else:
            end = start + chunk_size

        chunk_indices = sorted_indices[start:end]
        chunk_scores = [scores[j] for j in chunk_indices]

        chunk_sizes.append(len(chunk_indices))
        # Defensive: an empty chunk yields a neutral (0.0, 0.0) range instead of
        # crashing on min()/max() of an empty sequence.
        if chunk_scores:
            difficulty_ranges.append((min(chunk_scores), max(chunk_scores)))
        else:
            difficulty_ranges.append((0.0, 0.0))

    return CurriculumStats(
        total_samples=len(samples),
        num_chunks=effective_chunks,
        chunk_sizes=chunk_sizes,
        difficulty_ranges=difficulty_ranges,
    )
