"""Tests for engine failure handling (P0: handle empty/malformed traces)."""

import pytest
from datetime import datetime, timezone

from backpropagate.engine import (
    BackpropagationEngine,
    PropagationResult,
    SimulationResult,
    create_engine,
)
from backpropagate.contracts import Trace, TraceStep, LearningSignal, SignalSource
from backpropagate.memory import MemoryUpdater


class TestEmptyTraceHandling:
    """Tests for handling empty traces."""

    def test_propagate_empty_trace(self, backprop_engine):
        """Test propagating through an empty trace."""
        trace = Trace(trace_id="empty-trace", steps=[])
        result = backprop_engine.propagate(trace, 0.5)

        assert result.success is False
        assert "no steps" in result.errors[0].lower()
        assert result.signals_generated == 0

    def test_simulate_empty_trace(self, backprop_engine):
        """Test simulating with an empty trace."""
        trace = Trace(trace_id="empty-trace", steps=[])
        result = backprop_engine.simulate(trace, 0.5)

        assert len(result.would_generate) == 0
        assert "validation" in result.explanations[0].lower()

    def test_process_feedback_empty_trace(self, backprop_engine):
        """Test processing feedback on empty trace."""
        trace = Trace(trace_id="empty-trace", steps=[])
        result = backprop_engine.process_feedback(trace, "thumbs_up")

        assert result.success is False


class TestMalformedTraceHandling:
    """Tests for handling malformed traces."""

    def test_trace_missing_id(self, backprop_engine):
        """Test trace with missing ID."""
        with pytest.raises(ValueError):
            Trace(trace_id="", steps=[])

    def test_step_missing_tool_id(self):
        """Test step with missing tool ID."""
        with pytest.raises(ValueError):
            TraceStep(
                tool_id="",
                input_data={},
                output_data={},
                success=True,
            )

    def test_step_whitespace_tool_id(self):
        """Test step with whitespace-only tool ID."""
        with pytest.raises(ValueError):
            TraceStep(
                tool_id="   ",
                input_data={},
                output_data={},
                success=True,
            )

    def test_propagate_handles_step_validation_error(self, backprop_engine):
        """Test that propagation handles malformed steps gracefully."""
        # Create a trace with a step that has an empty tool_id won't work
        # because TraceStep validates in __post_init__
        # So we test the engine's validation logic directly
        trace = Trace(
            trace_id="test-trace",
            steps=[],  # Empty - triggers validation
        )
        result = backprop_engine.propagate(trace, 0.5)
        assert result.success is False


class TestTraceValidation:
    """Tests for trace validation in engine."""

    def test_valid_trace_passes(self, backprop_engine, sample_trace):
        """Test that valid trace passes validation."""
        result = backprop_engine.propagate(sample_trace, 0.5)
        assert result.success is True
        assert len(result.errors) == 0

    def test_engine_logs_validation_errors(self, backprop_engine, caplog):
        """Test that validation errors are logged."""
        trace = Trace(trace_id="empty", steps=[])
        backprop_engine.propagate(trace, 0.5)
        # Check that warning was logged
        assert any("invalid trace" in record.message.lower() for record in caplog.records)


class TestFailedStepHandling:
    """Tests for handling failed steps in traces."""

    def test_failed_step_gets_negative_signal(self, backprop_engine, failed_trace):
        """Test that failed steps receive negative signals."""
        result = backprop_engine.simulate(failed_trace, 0.5)

        # Find the signal for the failed tool
        failed_signals = [s for s in result.would_generate if s.tool_id == "parse_tool"]
        assert len(failed_signals) == 1
        assert failed_signals[0].delta < 0

    def test_propagate_through_failure(self, backprop_engine, failed_trace):
        """Test propagating through a trace with failures."""
        result = backprop_engine.propagate(failed_trace, 0.5)

        assert result.success is True  # Propagation succeeded
        assert "parse_tool" in result.tools_affected

    def test_learn_from_failure(self, backprop_engine, failed_trace):
        """Test the learn_from_failure convenience method."""
        result = backprop_engine.learn_from_failure(failed_trace)

        assert result.success is True
        # Both tools should have learned
        assert len(result.tools_affected) == 2


class TestEngineStateOnError:
    """Tests for engine state after errors."""

    def test_memory_unchanged_on_validation_error(self, backprop_engine):
        """Test that memory is unchanged when validation fails."""
        initial_len = len(backprop_engine.memory)

        trace = Trace(trace_id="empty", steps=[])
        backprop_engine.propagate(trace, 0.5)

        assert len(backprop_engine.memory) == initial_len

    def test_memory_unchanged_on_simulate(self, backprop_engine, sample_trace):
        """Test that simulate doesn't modify memory."""
        initial_len = len(backprop_engine.memory)

        backprop_engine.simulate(sample_trace, 0.5)

        assert len(backprop_engine.memory) == initial_len


class TestEngineRecovery:
    """Tests for engine recovery from errors."""

    def test_engine_works_after_error(self, backprop_engine, sample_trace):
        """Test that engine continues working after error."""
        # First, cause an error
        empty_trace = Trace(trace_id="empty", steps=[])
        result1 = backprop_engine.propagate(empty_trace, 0.5)
        assert result1.success is False

        # Then, do a valid operation
        result2 = backprop_engine.propagate(sample_trace, 0.5)
        assert result2.success is True

    def test_reset_clears_state(self, backprop_engine, sample_trace):
        """Test that reset clears all learning state."""
        # Do some learning
        backprop_engine.propagate(sample_trace, 0.5)
        assert len(backprop_engine.memory) > 0

        # Reset
        backprop_engine.reset()
        assert len(backprop_engine.memory) == 0

    def test_reset_specific_tool(self, backprop_engine, sample_trace):
        """Test resetting a specific tool."""
        backprop_engine.propagate(sample_trace, 0.5)

        # Reset only one tool
        backprop_engine.reset(tool_id="search_tool")

        # The other tool should still have signals
        assert "summarize_tool" in backprop_engine.memory


class TestCreateEngine:
    """Tests for engine factory function."""

    def test_create_engine_defaults(self):
        """Test creating engine with defaults."""
        engine = create_engine()
        assert engine is not None
        assert engine.memory is not None
        assert engine.propagation_factor == 0.5

    def test_create_engine_custom_settings(self):
        """Test creating engine with custom settings."""
        engine = create_engine(
            max_signals_per_tool=50,
            propagation_factor=0.3,
            min_confidence=0.2,
        )
        assert engine.propagation_factor == 0.3
        assert engine.min_confidence == 0.2


class TestEnginePersistence:
    """Tests for engine save/load."""

    def test_save_and_load(self, backprop_engine, sample_trace, tmp_path):
        """Test saving and loading engine state."""
        # Do some learning
        backprop_engine.propagate(sample_trace, 0.5)

        # Save
        path = tmp_path / "engine.json"
        backprop_engine.save(str(path))

        # Load into new engine
        loaded = BackpropagationEngine.load(str(path))

        # Verify state was preserved
        assert len(loaded.memory) == len(backprop_engine.memory)
        for tool_id in backprop_engine.memory.iter_tools():
            assert tool_id in loaded.memory


class TestSignalHooks:
    """Tests for signal generation hooks."""

    def test_hook_called_on_propagation(self, backprop_engine, sample_trace):
        """Test that hooks are called when signals are generated."""
        signals_received = []

        def hook(signal):
            signals_received.append(signal)

        backprop_engine.add_signal_hook(hook)
        backprop_engine.propagate(sample_trace, 0.5)

        assert len(signals_received) > 0

    def test_hook_error_does_not_stop_propagation(self, backprop_engine, sample_trace):
        """Test that hook errors don't stop propagation."""
        def bad_hook(signal):
            raise RuntimeError("Hook error")

        backprop_engine.add_signal_hook(bad_hook)
        result = backprop_engine.propagate(sample_trace, 0.5)

        assert result.success is True
        assert result.signals_applied > 0

    def test_multiple_hooks(self, backprop_engine, sample_trace):
        """Test multiple hooks are all called."""
        calls1 = []
        calls2 = []

        backprop_engine.add_signal_hook(lambda s: calls1.append(s))
        backprop_engine.add_signal_hook(lambda s: calls2.append(s))
        backprop_engine.propagate(sample_trace, 0.5)

        assert len(calls1) > 0
        assert len(calls1) == len(calls2)
