"""
Backpropagation Engine - Core learning engine for tool feedback.

P0: Handle empty/malformed traces gracefully.
P2: Dry-run/analysis mode (engine.simulate).
P2: Explanation interface for learning decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
import logging

from .contracts import LearningSignal, Trace, TraceStep, SignalSource
from .feedback import score_feedback, score_binary_feedback, aggregate_signals
from .trace import TraceGraph, TraceGraphBuilder
from .memory import MemoryUpdater, MemoryStats

logger = logging.getLogger(__name__)


@dataclass
class PropagationResult:
    """Result of a backpropagation operation."""
    signals_generated: int
    signals_applied: int
    tools_affected: list[str]
    trace_id: str
    success: bool
    errors: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result of a dry-run simulation."""
    would_generate: list[LearningSignal]
    would_affect: list[str]
    trace_id: str
    explanations: list[str]


class BackpropagationEngine:
    """Core engine for processing feedback and updating learning memory."""

    def __init__(self, memory: MemoryUpdater | None = None,
                 propagation_factor: float = 0.5,
                 min_confidence: float = 0.1) -> None:
        self.memory = memory or MemoryUpdater()
        self.propagation_factor = propagation_factor
        self.min_confidence = min_confidence
        self._signal_hooks: list[Callable[[LearningSignal], None]] = []

    def add_signal_hook(self, hook: Callable[[LearningSignal], None]) -> None:
        self._signal_hooks.append(hook)

    def _validate_trace(self, trace: Trace) -> list[str]:
        errors: list[str] = []
        if not trace.trace_id:
            errors.append("Trace missing trace_id")
        if not trace.steps:
            errors.append("Trace has no steps")
        for i, step in enumerate(trace.steps):
            if not step.tool_id:
                errors.append(f"Step {i} missing tool_id")
        return errors

    def _generate_signals_from_trace(
        self, trace: Trace, feedback_delta: float,
        source: SignalSource = SignalSource.TOOL_RESULT,
    ) -> tuple[list[LearningSignal], list[str]]:
        signals: list[LearningSignal] = []
        explanations: list[str] = []
        if not trace.steps:
            return signals, explanations
        sorted_steps = sorted(trace.steps, key=lambda s: s.order, reverse=True)
        for i, step in enumerate(sorted_steps):
            propagation_decay = self.propagation_factor ** i
            confidence = propagation_decay
            if confidence < self.min_confidence:
                explanations.append(
                    f"Skipped {step.tool_id}: confidence {confidence:.2f} < min {self.min_confidence}")
                continue
            step_delta = feedback_delta
            if not step.success:
                step_delta = min(step_delta, -0.5)
                explanations.append(
                    f"Adjusted {step.tool_id}: failed step, delta clamped to {step_delta:.2f}")
            signal = LearningSignal(
                tool_id=step.tool_id, delta=step_delta, confidence=confidence,
                source=source, context={"trace_id": trace.trace_id, "step_order": step.order})
            signals.append(signal)
            explanations.append(
                f"Generated signal for {step.tool_id}: delta={step_delta:.2f}, confidence={confidence:.2f}")
        return signals, explanations

    def propagate(self, trace: Trace, feedback_delta: float,
                  source: SignalSource = SignalSource.USER_FEEDBACK,
                  apply_to_memory: bool = True) -> PropagationResult:
        errors = self._validate_trace(trace)
        if errors:
            logger.warning(f"Invalid trace {trace.trace_id}: {errors}")
            return PropagationResult(
                signals_generated=0, signals_applied=0, tools_affected=[],
                trace_id=trace.trace_id or "unknown", success=False, errors=errors)
        signals, explanations = self._generate_signals_from_trace(trace, feedback_delta, source)
        tools_affected = list(set(s.tool_id for s in signals))
        applied_count = 0
        if apply_to_memory:
            for signal in signals:
                self.memory.update(signal)
                applied_count += 1
                for hook in self._signal_hooks:
                    try:
                        hook(signal)
                    except Exception as e:
                        logger.warning(f"Signal hook error: {e}")
        return PropagationResult(
            signals_generated=len(signals), signals_applied=applied_count,
            tools_affected=tools_affected, trace_id=trace.trace_id,
            success=True, explanations=explanations)

    def propagate_from_graph(self, graph: TraceGraph, feedback_delta: float,
                             source: SignalSource = SignalSource.USER_FEEDBACK) -> PropagationResult:
        return self.propagate(graph.trace, feedback_delta, source)

    def simulate(self, trace: Trace, feedback_delta: float,
                 source: SignalSource = SignalSource.USER_FEEDBACK) -> SimulationResult:
        errors = self._validate_trace(trace)
        if errors:
            return SimulationResult(
                would_generate=[], would_affect=[],
                trace_id=trace.trace_id or "unknown",
                explanations=[f"Validation errors: {errors}"])
        signals, explanations = self._generate_signals_from_trace(trace, feedback_delta, source)
        return SimulationResult(
            would_generate=signals, would_affect=list(set(s.tool_id for s in signals)),
            trace_id=trace.trace_id, explanations=explanations)

    def process_feedback(self, trace: Trace, feedback_type: str,
                         source: SignalSource = SignalSource.USER_FEEDBACK) -> PropagationResult:
        signal = score_feedback(feedback_type, "dummy", source=source)
        return self.propagate(trace, signal.delta, source)

    def process_binary_feedback(self, trace: Trace, is_positive: bool,
                                intensity: float = 0.5) -> PropagationResult:
        signal = score_binary_feedback(is_positive, "dummy", intensity)
        return self.propagate(trace, signal.delta)

    def learn_from_success(self, trace: Trace) -> PropagationResult:
        return self.propagate(trace, 0.5, SignalSource.TOOL_RESULT)

    def learn_from_failure(self, trace: Trace) -> PropagationResult:
        return self.propagate(trace, -0.5, SignalSource.TOOL_RESULT)

    def get_tool_scores(self) -> dict[str, float]:
        return self.memory.get_all_scores()

    def get_tool_score(self, tool_id: str) -> float:
        return self.memory.get_tool_score(tool_id)

    def explain_tool_score(self, tool_id: str) -> dict[str, Any]:
        signals = self.memory.get_signals(tool_id=tool_id)
        score = self.memory.get_tool_score(tool_id)
        if not signals:
            return {"tool_id": tool_id, "current_score": 0.0, "signal_count": 0,
                    "explanation": "No learning signals recorded for this tool."}
        positive_signals = [s for s in signals if s.delta > 0]
        negative_signals = [s for s in signals if s.delta < 0]
        return {
            "tool_id": tool_id, "current_score": score, "signal_count": len(signals),
            "positive_signals": len(positive_signals), "negative_signals": len(negative_signals),
            "neutral_signals": len([s for s in signals if s.delta == 0]),
            "average_confidence": sum(s.confidence for s in signals) / len(signals),
            "sources": list(set(s.source.value for s in signals)),
            "explanation": (
                f"Score {score:.2f} computed from {len(signals)} signals: "
                f"{len(positive_signals)} positive, {len(negative_signals)} negative. "
                f"Signals are weighted by confidence and time decay."),
        }

    def get_stats(self) -> MemoryStats:
        return self.memory.get_stats()

    def reset(self, tool_id: str | None = None) -> int:
        return self.memory.clear(tool_id)

    def rollback_to(self, timestamp: datetime, tool_id: str | None = None) -> int:
        return self.memory.rollback(timestamp, tool_id)

    def save(self, path: str) -> None:
        self.memory.save(path)

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "BackpropagationEngine":
        memory = MemoryUpdater.load(path)
        return cls(memory=memory, **kwargs)


def create_engine(max_signals_per_tool: int = 1000,
                  propagation_factor: float = 0.5,
                  min_confidence: float = 0.1) -> BackpropagationEngine:
    """Create a new backpropagation engine with specified settings."""
    memory = MemoryUpdater(max_signals_per_tool=max_signals_per_tool)
    return BackpropagationEngine(
        memory=memory, propagation_factor=propagation_factor,
        min_confidence=min_confidence)
