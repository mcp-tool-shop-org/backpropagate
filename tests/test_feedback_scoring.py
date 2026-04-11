"""Tests for the feedback scoring module."""

import pytest
from datetime import datetime, timezone, timedelta

from backpropagate.feedback import (
    score_feedback,
    score_binary_feedback,
    score_numeric_feedback,
    apply_time_decay,
    aggregate_signals,
    is_positive_feedback,
    is_negative_feedback,
    is_neutral_feedback,
    get_feedback_types,
    FEEDBACK_DELTAS,
)
from backpropagate.contracts import LearningSignal, SignalSource


class TestScoreFeedback:
    """Tests for the score_feedback function."""

    def test_positive_feedback_thumbs_up(self):
        """Test that thumbs_up produces positive delta."""
        signal = score_feedback("thumbs_up", "test_tool")
        assert signal.delta == 0.5
        assert signal.tool_id == "test_tool"
        assert signal.source == SignalSource.USER_FEEDBACK

    def test_negative_feedback_thumbs_down(self):
        """Test that thumbs_down produces negative delta."""
        signal = score_feedback("thumbs_down", "test_tool")
        assert signal.delta == -0.5

    def test_neutral_feedback(self):
        """Test that neutral feedback produces zero delta."""
        signal = score_feedback("neutral", "test_tool")
        assert signal.delta == 0.0

    def test_unknown_feedback_type_returns_zero(self):
        """Test that unknown feedback types default to zero delta."""
        signal = score_feedback("unknown_type", "test_tool")
        assert signal.delta == 0.0

    def test_custom_delta_override(self):
        """Test that custom_delta overrides the default."""
        signal = score_feedback("thumbs_up", "test_tool", custom_delta=0.9)
        assert signal.delta == 0.9

    def test_custom_delta_clamped(self):
        """Test that custom_delta is clamped to [-1.0, +1.0]."""
        signal = score_feedback("thumbs_up", "test_tool", custom_delta=5.0)
        assert signal.delta == 1.0

        signal = score_feedback("thumbs_down", "test_tool", custom_delta=-5.0)
        assert signal.delta == -1.0

    def test_input_pattern_passed_through(self):
        """Test that input_pattern is included in the signal."""
        signal = score_feedback(
            "thumbs_up", "test_tool", input_pattern="query:*"
        )
        assert signal.input_pattern == "query:*"

    def test_context_passed_through(self):
        """Test that context is included in the signal."""
        context = {"user_id": "123", "session": "abc"}
        signal = score_feedback("thumbs_up", "test_tool", context=context)
        assert signal.context == context

    def test_source_respected(self):
        """Test that source parameter is respected."""
        signal = score_feedback(
            "correct", "test_tool", source=SignalSource.CORRECTION
        )
        assert signal.source == SignalSource.CORRECTION

    def test_timestamp_default_is_now(self):
        """Test that timestamp defaults to current time."""
        before = datetime.now(timezone.utc)
        signal = score_feedback("thumbs_up", "test_tool")
        after = datetime.now(timezone.utc)
        assert before <= signal.timestamp <= after

    def test_explicit_timestamp(self):
        """Test that explicit timestamp is used."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        signal = score_feedback("thumbs_up", "test_tool", timestamp=ts)
        assert signal.timestamp == ts


class TestScoreBinaryFeedback:
    """Tests for binary feedback scoring."""

    def test_positive_binary_feedback(self):
        """Test positive binary feedback."""
        signal = score_binary_feedback(True, "test_tool")
        assert signal.delta == 0.5  # default intensity
        assert signal.delta > 0

    def test_negative_binary_feedback(self):
        """Test negative binary feedback."""
        signal = score_binary_feedback(False, "test_tool")
        assert signal.delta == -0.5
        assert signal.delta < 0

    def test_intensity_scaling(self):
        """Test that intensity scales the delta."""
        weak = score_binary_feedback(True, "test_tool", intensity=0.2)
        strong = score_binary_feedback(True, "test_tool", intensity=0.8)
        assert weak.delta == 0.2
        assert strong.delta == 0.8

    def test_intensity_clamped(self):
        """Test that intensity is clamped to [0, 1]."""
        signal = score_binary_feedback(True, "test_tool", intensity=1.5)
        assert signal.delta == 1.0


class TestScoreNumericFeedback:
    """Tests for numeric rating feedback."""

    def test_max_rating_gives_max_delta(self):
        """Test that max rating produces +1.0 delta."""
        signal = score_numeric_feedback(5, "test_tool", min_rating=1, max_rating=5)
        assert signal.delta == 1.0

    def test_min_rating_gives_min_delta(self):
        """Test that min rating produces -1.0 delta."""
        signal = score_numeric_feedback(1, "test_tool", min_rating=1, max_rating=5)
        assert signal.delta == -1.0

    def test_mid_rating_gives_zero_delta(self):
        """Test that mid rating produces 0.0 delta."""
        signal = score_numeric_feedback(3, "test_tool", min_rating=1, max_rating=5)
        assert signal.delta == 0.0

    def test_custom_rating_range(self):
        """Test with custom rating range."""
        # 0-10 scale, 7.5 should be +0.5
        signal = score_numeric_feedback(7.5, "test_tool", min_rating=0, max_rating=10)
        assert 0.4 <= signal.delta <= 0.6


class TestTimeDecay:
    """Tests for time-based confidence decay."""

    def test_no_decay_for_current_signal(self):
        """Test that recent signals have minimal decay."""
        signal = LearningSignal(
            tool_id="test_tool",
            delta=0.5,
            confidence=1.0,
            timestamp=datetime.now(timezone.utc),
        )
        decayed = apply_time_decay(signal)
        assert decayed.confidence > 0.99

    def test_halflife_decay(self):
        """Test that confidence halves after halflife period."""
        halflife = 168  # 1 week in hours
        old_time = datetime.now(timezone.utc) - timedelta(hours=halflife)
        signal = LearningSignal(
            tool_id="test_tool",
            delta=0.5,
            confidence=1.0,
            timestamp=old_time,
        )
        decayed = apply_time_decay(signal, halflife_hours=halflife)
        assert 0.45 <= decayed.confidence <= 0.55  # ~0.5

    def test_double_halflife_decay(self):
        """Test decay after two half-life periods."""
        halflife = 168
        old_time = datetime.now(timezone.utc) - timedelta(hours=halflife * 2)
        signal = LearningSignal(
            tool_id="test_tool",
            delta=0.5,
            confidence=1.0,
            timestamp=old_time,
        )
        decayed = apply_time_decay(signal, halflife_hours=halflife)
        assert 0.2 <= decayed.confidence <= 0.3  # ~0.25


class TestAggregateSignals:
    """Tests for signal aggregation."""

    def test_empty_list_returns_zero(self):
        """Test that empty list returns zero."""
        result = aggregate_signals([])
        assert result == 0.0

    def test_single_signal(self):
        """Test aggregation of single signal."""
        signal = LearningSignal(tool_id="test", delta=0.5, confidence=1.0)
        result = aggregate_signals([signal], apply_decay=False)
        assert result == 0.5

    def test_multiple_signals_weighted_average(self):
        """Test weighted average of multiple signals."""
        signals = [
            LearningSignal(tool_id="test", delta=1.0, confidence=0.5),
            LearningSignal(tool_id="test", delta=0.0, confidence=0.5),
        ]
        # Weighted average: (1.0*0.5 + 0.0*0.5) / (0.5 + 0.5) = 0.5
        result = aggregate_signals(signals, apply_decay=False)
        assert result == 0.5

    def test_result_clamped(self):
        """Test that result is clamped to [-1, 1]."""
        signals = [
            LearningSignal(tool_id="test", delta=1.0, confidence=1.0),
            LearningSignal(tool_id="test", delta=1.0, confidence=1.0),
        ]
        result = aggregate_signals(signals, apply_decay=False)
        assert result <= 1.0


class TestFeedbackClassification:
    """Tests for feedback type classification."""

    def test_is_positive_feedback(self):
        """Test positive feedback detection."""
        assert is_positive_feedback("thumbs_up")
        assert is_positive_feedback("love")
        assert is_positive_feedback("correct")
        assert not is_positive_feedback("thumbs_down")
        assert not is_positive_feedback("neutral")

    def test_is_negative_feedback(self):
        """Test negative feedback detection."""
        assert is_negative_feedback("thumbs_down")
        assert is_negative_feedback("wrong")
        assert is_negative_feedback("error")
        assert not is_negative_feedback("thumbs_up")
        assert not is_negative_feedback("neutral")

    def test_is_neutral_feedback(self):
        """Test neutral feedback detection."""
        assert is_neutral_feedback("neutral")
        assert is_neutral_feedback("skip")
        assert not is_neutral_feedback("thumbs_up")
        assert not is_neutral_feedback("thumbs_down")

    def test_get_feedback_types(self):
        """Test getting all feedback types."""
        types = get_feedback_types()
        assert "thumbs_up" in types
        assert "thumbs_down" in types
        assert "neutral" in types
        assert types == FEEDBACK_DELTAS
