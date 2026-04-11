"""
Learning Contracts - Schema definitions for learning signals.

This module defines the data contracts for the backpropagation learning system,
ensuring type safety and validation for all learning signals.

P0: Schema versioning for stored data compatibility.
P0: Learning safety bounds (clamp deltas to [-1.0, +1.0]).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# Schema version for backwards compatibility
SCHEMA_VERSION = "1.0.0"

# Learning safety bounds
MIN_DELTA = -1.0
MAX_DELTA = 1.0


class SignalSource(Enum):
    """Source of the learning signal."""

    USER_FEEDBACK = "user_feedback"
    TOOL_RESULT = "tool_result"
    INFERENCE = "inference"
    CORRECTION = "correction"
    SYSTEM = "system"


@dataclass
class LearningSignal:
    """
    A learning signal representing feedback for a tool/action.

    Attributes:
        tool_id: Identifier for the tool that produced the result
        delta: Learning adjustment in range [-1.0, +1.0]
        confidence: Confidence in this signal (0.0 to 1.0)
        source: Where the signal originated from
        timestamp: When the signal was generated
        context: Optional context data for the signal
        input_pattern: Optional pattern to scope learning
        schema_version: Version of the schema for compatibility
    """

    tool_id: str
    delta: float
    confidence: float = 1.0
    source: SignalSource = SignalSource.USER_FEEDBACK
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: dict[str, Any] = field(default_factory=dict)
    input_pattern: str | None = None
    schema_version: str = field(default=SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Validate and clamp values after initialization."""
        self.delta = max(MIN_DELTA, min(MAX_DELTA, self.delta))
        self.confidence = max(0.0, min(1.0, self.confidence))
        if not self.tool_id or not self.tool_id.strip():
            raise ValueError("tool_id cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "tool_id": self.tool_id,
            "delta": self.delta,
            "confidence": self.confidence,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "input_pattern": self.input_pattern,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningSignal":
        """Deserialize from dictionary."""
        schema_version = data.get("schema_version", "1.0.0")
        return cls(
            tool_id=data["tool_id"],
            delta=data["delta"],
            confidence=data.get("confidence", 1.0),
            source=SignalSource(data.get("source", "user_feedback")),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data.get("timestamp"), str)
            else data.get("timestamp", datetime.now(timezone.utc)),
            context=data.get("context", {}),
            input_pattern=data.get("input_pattern"),
            schema_version=schema_version,
        )

    def weighted_delta(self) -> float:
        """Return delta weighted by confidence."""
        return self.delta * self.confidence


@dataclass
class TraceStep:
    """A single step in a tool execution trace."""

    tool_id: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    success: bool
    duration_ms: float = 0.0
    order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate step data."""
        if not self.tool_id or not self.tool_id.strip():
            raise ValueError("tool_id cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tool_id": self.tool_id,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "order": self.order,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceStep":
        """Deserialize from dictionary."""
        return cls(
            tool_id=data["tool_id"],
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data", {}),
            success=data.get("success", True),
            duration_ms=data.get("duration_ms", 0.0),
            order=data.get("order", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Trace:
    """A complete execution trace containing multiple steps."""

    trace_id: str
    steps: list[TraceStep] = field(default_factory=list)
    success: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Validate trace data."""
        if not self.trace_id or not self.trace_id.strip():
            raise ValueError("trace_id cannot be empty")

    def add_step(self, step: TraceStep) -> None:
        """Add a step to the trace with proper ordering."""
        step.order = len(self.steps)
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trace_id": self.trace_id,
            "steps": [s.to_dict() for s in self.steps],
            "success": self.success,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trace":
        """Deserialize from dictionary."""
        steps = [TraceStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            trace_id=data["trace_id"],
            steps=steps,
            success=data.get("success", True),
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at", datetime.now(timezone.utc)),
            metadata=data.get("metadata", {}),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    def get_tool_ids(self) -> list[str]:
        """Get list of tool IDs in execution order."""
        return [step.tool_id for step in sorted(self.steps, key=lambda s: s.order)]

    def is_valid(self) -> bool:
        """Check if trace has required data."""
        return bool(self.trace_id and len(self.steps) > 0)
