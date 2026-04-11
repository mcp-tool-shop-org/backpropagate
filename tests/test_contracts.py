"""Tests for learning contracts (schema definitions)."""

import pytest
from datetime import datetime, timezone

from backpropagate.contracts import (
    LearningSignal,
    TraceStep,
    Trace,
    SignalSource,
    SCHEMA_VERSION,
    MIN_DELTA,
    MAX_DELTA,
)


class TestLearningSignal:
    """Tests for LearningSignal class."""

    def test_create_basic_signal(self):
        """Test creating a basic learning signal."""
        signal = LearningSignal(tool_id="test_tool", delta=0.5)
        assert signal.tool_id == "test_tool"
        assert signal.delta == 0.5
        assert signal.confidence == 1.0  # default
        assert signal.source == SignalSource.USER_FEEDBACK  # default

    def test_delta_clamped_positive(self):
        """Test that delta > 1.0 is clamped to 1.0."""
        signal = LearningSignal(tool_id="test", delta=5.0)
        assert signal.delta == MAX_DELTA  # 1.0

    def test_delta_clamped_negative(self):
        """Test that delta < -1.0 is clamped to -1.0."""
        signal = LearningSignal(tool_id="test", delta=-5.0)
        assert signal.delta == MIN_DELTA  # -1.0

    def test_confidence_clamped(self):
        """Test that confidence is clamped to [0, 1]."""
        signal = LearningSignal(tool_id="test", delta=0.5, confidence=2.0)
        assert signal.confidence == 1.0

        signal = LearningSignal(tool_id="test", delta=0.5, confidence=-0.5)
        assert signal.confidence == 0.0

    def test_empty_tool_id_raises(self):
        """Test that empty tool_id raises ValueError."""
        with pytest.raises(ValueError, match="tool_id cannot be empty"):
            LearningSignal(tool_id="", delta=0.5)

    def test_whitespace_tool_id_raises(self):
        """Test that whitespace-only tool_id raises ValueError."""
        with pytest.raises(ValueError, match="tool_id cannot be empty"):
            LearningSignal(tool_id="   ", delta=0.5)

    def test_weighted_delta(self):
        """Test weighted delta calculation."""
        signal = LearningSignal(tool_id="test", delta=0.5, confidence=0.8)
        assert signal.weighted_delta() == 0.4  # 0.5 * 0.8

    def test_schema_version_default(self):
        """Test that schema version is set by default."""
        signal = LearningSignal(tool_id="test", delta=0.5)
        assert signal.schema_version == SCHEMA_VERSION

    def test_signal_sources(self):
        """Test all signal sources."""
        for source in SignalSource:
            signal = LearningSignal(tool_id="test", delta=0.5, source=source)
            assert signal.source == source


class TestLearningSignalSerialization:
    """Tests for LearningSignal serialization."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        signal = LearningSignal(
            tool_id="test_tool",
            delta=0.5,
            confidence=0.8,
            source=SignalSource.CORRECTION,
            context={"key": "value"},
            input_pattern="query:*",
        )
        data = signal.to_dict()

        assert data["tool_id"] == "test_tool"
        assert data["delta"] == 0.5
        assert data["confidence"] == 0.8
        assert data["source"] == "correction"
        assert data["context"] == {"key": "value"}
        assert data["input_pattern"] == "query:*"
        assert "timestamp" in data
        assert data["schema_version"] == SCHEMA_VERSION

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "tool_id": "test_tool",
            "delta": 0.5,
            "confidence": 0.8,
            "source": "user_feedback",
            "timestamp": "2024-01-01T12:00:00",
            "context": {"key": "value"},
            "input_pattern": None,
            "schema_version": SCHEMA_VERSION,
        }
        signal = LearningSignal.from_dict(data)

        assert signal.tool_id == "test_tool"
        assert signal.delta == 0.5
        assert signal.source == SignalSource.USER_FEEDBACK

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = LearningSignal(
            tool_id="test_tool",
            delta=0.5,
            confidence=0.8,
            source=SignalSource.TOOL_RESULT,
            context={"test": True},
        )
        data = original.to_dict()
        restored = LearningSignal.from_dict(data)

        assert restored.tool_id == original.tool_id
        assert restored.delta == original.delta
        assert restored.confidence == original.confidence
        assert restored.source == original.source


class TestTraceStep:
    """Tests for TraceStep class."""

    def test_create_basic_step(self):
        """Test creating a basic trace step."""
        step = TraceStep(
            tool_id="test_tool",
            input_data={"query": "test"},
            output_data={"result": "success"},
            success=True,
        )
        assert step.tool_id == "test_tool"
        assert step.success is True
        assert step.order == 0  # default

    def test_empty_tool_id_raises(self):
        """Test that empty tool_id raises ValueError."""
        with pytest.raises(ValueError, match="tool_id cannot be empty"):
            TraceStep(tool_id="", input_data={}, output_data={}, success=True)

    def test_step_with_duration(self):
        """Test step with duration."""
        step = TraceStep(
            tool_id="test",
            input_data={},
            output_data={},
            success=True,
            duration_ms=150.5,
        )
        assert step.duration_ms == 150.5


class TestTraceStepSerialization:
    """Tests for TraceStep serialization."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        step = TraceStep(
            tool_id="test_tool",
            input_data={"a": 1},
            output_data={"b": 2},
            success=True,
            duration_ms=100.0,
            order=0,
        )
        data = step.to_dict()

        assert data["tool_id"] == "test_tool"
        assert data["input_data"] == {"a": 1}
        assert data["output_data"] == {"b": 2}
        assert data["success"] is True
        assert data["duration_ms"] == 100.0
        assert data["order"] == 0

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "tool_id": "test_tool",
            "input_data": {"a": 1},
            "output_data": {"b": 2},
            "success": True,
        }
        step = TraceStep.from_dict(data)

        assert step.tool_id == "test_tool"
        assert step.success is True

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = TraceStep(
            tool_id="test",
            input_data={"x": 1},
            output_data={"y": 2},
            success=False,
            duration_ms=50.0,
            order=3,
        )
        data = original.to_dict()
        restored = TraceStep.from_dict(data)

        assert restored.tool_id == original.tool_id
        assert restored.success == original.success
        assert restored.order == original.order


class TestTrace:
    """Tests for Trace class."""

    def test_create_basic_trace(self):
        """Test creating a basic trace."""
        trace = Trace(trace_id="trace-001")
        assert trace.trace_id == "trace-001"
        assert trace.success is True  # default
        assert len(trace.steps) == 0

    def test_empty_trace_id_raises(self):
        """Test that empty trace_id raises ValueError."""
        with pytest.raises(ValueError, match="trace_id cannot be empty"):
            Trace(trace_id="")

    def test_add_step_sets_order(self):
        """Test that add_step sets correct order."""
        trace = Trace(trace_id="trace-001")
        step1 = TraceStep(tool_id="t1", input_data={}, output_data={}, success=True)
        step2 = TraceStep(tool_id="t2", input_data={}, output_data={}, success=True)

        trace.add_step(step1)
        trace.add_step(step2)

        assert step1.order == 0
        assert step2.order == 1

    def test_get_tool_ids(self, sample_trace):
        """Test getting tool IDs in order."""
        tool_ids = sample_trace.get_tool_ids()
        assert tool_ids == ["search_tool", "summarize_tool"]

    def test_is_valid(self, sample_trace):
        """Test trace validity check."""
        assert sample_trace.is_valid() is True

    def test_is_valid_empty_steps(self):
        """Test that trace with no steps is invalid."""
        trace = Trace(trace_id="trace-001")
        assert trace.is_valid() is False


class TestTraceSerialization:
    """Tests for Trace serialization."""

    def test_to_dict(self, sample_trace):
        """Test serialization to dictionary."""
        data = sample_trace.to_dict()

        assert data["trace_id"] == "test-trace-001"
        assert len(data["steps"]) == 2
        assert data["success"] is True
        assert "created_at" in data
        assert data["schema_version"] == SCHEMA_VERSION

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "trace_id": "trace-001",
            "steps": [
                {"tool_id": "t1", "input_data": {}, "output_data": {}, "success": True}
            ],
            "success": True,
            "created_at": "2024-01-01T12:00:00",
            "metadata": {},
            "schema_version": SCHEMA_VERSION,
        }
        trace = Trace.from_dict(data)

        assert trace.trace_id == "trace-001"
        assert len(trace.steps) == 1

    def test_roundtrip(self, sample_trace):
        """Test serialization roundtrip."""
        data = sample_trace.to_dict()
        restored = Trace.from_dict(data)

        assert restored.trace_id == sample_trace.trace_id
        assert len(restored.steps) == len(sample_trace.steps)
        assert restored.success == sample_trace.success


class TestSchemaVersioning:
    """Tests for schema versioning (P0)."""

    def test_schema_version_constant(self):
        """Test that SCHEMA_VERSION is defined."""
        assert SCHEMA_VERSION is not None
        assert len(SCHEMA_VERSION) > 0

    def test_signal_includes_version(self):
        """Test that signals include schema version."""
        signal = LearningSignal(tool_id="test", delta=0.5)
        assert signal.schema_version == SCHEMA_VERSION

    def test_trace_includes_version(self):
        """Test that traces include schema version."""
        trace = Trace(trace_id="test")
        assert trace.schema_version == SCHEMA_VERSION

    def test_from_dict_handles_missing_version(self):
        """Test deserialization handles missing version."""
        data = {
            "tool_id": "test",
            "delta": 0.5,
            "timestamp": "2024-01-01T12:00:00",
        }
        signal = LearningSignal.from_dict(data)
        assert signal.schema_version == "1.0.0"  # default


class TestSafetyBounds:
    """Tests for learning safety bounds (P0)."""

    def test_min_delta_constant(self):
        """Test MIN_DELTA is -1.0."""
        assert MIN_DELTA == -1.0

    def test_max_delta_constant(self):
        """Test MAX_DELTA is 1.0."""
        assert MAX_DELTA == 1.0

    def test_delta_clamping_applies(self):
        """Test that clamping is applied on creation."""
        signal = LearningSignal(tool_id="test", delta=100.0)
        assert signal.delta == 1.0

        signal = LearningSignal(tool_id="test", delta=-100.0)
        assert signal.delta == -1.0

    def test_valid_range_unchanged(self):
        """Test that valid delta values are unchanged."""
        for delta in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            signal = LearningSignal(tool_id="test", delta=delta)
            assert signal.delta == delta
