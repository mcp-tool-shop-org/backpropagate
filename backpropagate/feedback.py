"""
Feedback Scoring - Convert user feedback to learning signals.

P1: Confidence weighting based on feedback source and recency.
"""

from datetime import datetime, timezone, timedelta
from typing import Any

from .contracts import LearningSignal, SignalSource, MIN_DELTA, MAX_DELTA

FEEDBACK_DELTAS = {
    "thumbs_up": 0.5, "like": 0.5, "love": 0.8, "helpful": 0.6,
    "correct": 0.7, "perfect": 1.0, "good": 0.4, "yes": 0.3,
    "thumbs_down": -0.5, "dislike": -0.5, "wrong": -0.7, "unhelpful": -0.6,
    "incorrect": -0.8, "bad": -0.4, "no": -0.3, "error": -0.6,
    "neutral": 0.0, "skip": 0.0, "unknown": 0.0,
}

SOURCE_CONFIDENCE = {
    SignalSource.USER_FEEDBACK: 1.0, SignalSource.CORRECTION: 0.9,
    SignalSource.TOOL_RESULT: 0.7, SignalSource.INFERENCE: 0.5,
    SignalSource.SYSTEM: 0.8,
}

CONFIDENCE_DECAY_HALFLIFE_HOURS = 168


def score_feedback(
    feedback_type: str, tool_id: str,
    source: SignalSource = SignalSource.USER_FEEDBACK,
    context: dict[str, Any] | None = None,
    input_pattern: str | None = None,
    timestamp: datetime | None = None,
    custom_delta: float | None = None,
) -> LearningSignal:
    """Convert feedback into a learning signal."""
    if custom_delta is not None:
        delta = max(MIN_DELTA, min(MAX_DELTA, custom_delta))
    else:
        delta = FEEDBACK_DELTAS.get(feedback_type.lower(), 0.0)
    confidence = SOURCE_CONFIDENCE.get(source, 0.5)
    return LearningSignal(
        tool_id=tool_id, delta=delta, confidence=confidence, source=source,
        timestamp=timestamp or datetime.now(timezone.utc),
        context=context or {}, input_pattern=input_pattern,
    )


def score_binary_feedback(
    is_positive: bool, tool_id: str, intensity: float = 0.5,
    source: SignalSource = SignalSource.USER_FEEDBACK,
    context: dict[str, Any] | None = None,
) -> LearningSignal:
    """Convert binary (yes/no) feedback to a learning signal."""
    intensity = max(0.0, min(1.0, intensity))
    delta = intensity if is_positive else -intensity
    return LearningSignal(
        tool_id=tool_id, delta=delta,
        confidence=SOURCE_CONFIDENCE.get(source, 0.5),
        source=source, context=context or {},
    )


def score_numeric_feedback(
    rating: float, tool_id: str,
    min_rating: float = 1.0, max_rating: float = 5.0,
    source: SignalSource = SignalSource.USER_FEEDBACK,
    context: dict[str, Any] | None = None,
) -> LearningSignal:
    """Convert a numeric rating to a learning signal."""
    normalized = (rating - min_rating) / (max_rating - min_rating)
    normalized = max(0.0, min(1.0, normalized))
    delta = (normalized * 2) - 1
    return LearningSignal(
        tool_id=tool_id, delta=delta,
        confidence=SOURCE_CONFIDENCE.get(source, 0.5),
        source=source, context=context or {},
    )


def apply_time_decay(
    signal: LearningSignal,
    reference_time: datetime | None = None,
    halflife_hours: float = CONFIDENCE_DECAY_HALFLIFE_HOURS,
) -> LearningSignal:
    """Apply time-based confidence decay to a signal."""
    reference = reference_time or datetime.now(timezone.utc)
    age_hours = (reference - signal.timestamp).total_seconds() / 3600
    decay_factor = 0.5 ** (age_hours / halflife_hours)
    decayed_confidence = signal.confidence * decay_factor
    return LearningSignal(
        tool_id=signal.tool_id, delta=signal.delta,
        confidence=decayed_confidence, source=signal.source,
        timestamp=signal.timestamp, context=signal.context,
        input_pattern=signal.input_pattern,
        schema_version=signal.schema_version,
    )


def aggregate_signals(signals: list[LearningSignal], apply_decay: bool = True) -> float:
    """Aggregate multiple signals into a single weighted delta."""
    if not signals:
        return 0.0
    if apply_decay:
        signals = [apply_time_decay(s) for s in signals]
    total_weight = sum(s.confidence for s in signals)
    if total_weight == 0:
        return 0.0
    weighted_sum = sum(s.weighted_delta() for s in signals)
    result = weighted_sum / total_weight
    return max(MIN_DELTA, min(MAX_DELTA, result))


def is_positive_feedback(feedback_type: str) -> bool:
    """Check if a feedback type is positive."""
    return FEEDBACK_DELTAS.get(feedback_type.lower(), 0.0) > 0


def is_negative_feedback(feedback_type: str) -> bool:
    """Check if a feedback type is negative."""
    return FEEDBACK_DELTAS.get(feedback_type.lower(), 0.0) < 0


def is_neutral_feedback(feedback_type: str) -> bool:
    """Check if a feedback type is neutral."""
    return FEEDBACK_DELTAS.get(feedback_type.lower(), 0.0) == 0.0


def get_feedback_types() -> dict[str, float]:
    """Get all supported feedback types and their deltas."""
    return FEEDBACK_DELTAS.copy()
