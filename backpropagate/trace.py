"""
Trace Graph - Build and analyze tool execution traces.

P1: Explicit ordering to trace graph nodes.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator
import json

from .contracts import Trace, TraceStep, SCHEMA_VERSION


@dataclass
class TraceNode:
    """A node in the trace graph representing a tool execution."""
    step: TraceStep
    children: list["TraceNode"] = field(default_factory=list)
    parent: "TraceNode | None" = None

    @property
    def tool_id(self) -> str:
        return self.step.tool_id

    @property
    def success(self) -> bool:
        return self.step.success

    @property
    def order(self) -> int:
        return self.step.order

    def add_child(self, node: "TraceNode") -> None:
        node.parent = self
        self.children.append(node)

    def depth(self) -> int:
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth


class TraceGraph:
    """A graph representation of tool execution traces."""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self.nodes: dict[int, TraceNode] = {}
        self.root: TraceNode | None = None
        self._build_graph()

    def _build_graph(self) -> None:
        if not self.trace.steps:
            return
        sorted_steps = sorted(self.trace.steps, key=lambda s: s.order)
        for step in sorted_steps:
            self.nodes[step.order] = TraceNode(step=step)
        for i, step in enumerate(sorted_steps):
            node = self.nodes[step.order]
            if i == 0:
                self.root = node
            else:
                prev_order = sorted_steps[i - 1].order
                self.nodes[prev_order].add_child(node)

    @property
    def trace_id(self) -> str:
        return self.trace.trace_id

    @property
    def success(self) -> bool:
        return self.trace.success

    def get_node(self, order: int) -> TraceNode | None:
        return self.nodes.get(order)

    def get_tool_nodes(self, tool_id: str) -> list[TraceNode]:
        return [n for n in self.nodes.values() if n.tool_id == tool_id]

    def get_failed_nodes(self) -> list[TraceNode]:
        return [n for n in self.nodes.values() if not n.success]

    def get_successful_nodes(self) -> list[TraceNode]:
        return [n for n in self.nodes.values() if n.success]

    def iter_nodes(self) -> Iterator[TraceNode]:
        for order in sorted(self.nodes.keys()):
            yield self.nodes[order]

    def iter_depth_first(self) -> Iterator[TraceNode]:
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def iter_breadth_first(self) -> Iterator[TraceNode]:
        if self.root is None:
            return
        from collections import deque
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    def get_execution_path(self) -> list[str]:
        return [node.tool_id for node in self.iter_nodes()]

    def get_failure_path(self) -> list[str]:
        for node in self.iter_nodes():
            if not node.success:
                path = []
                current: TraceNode | None = node
                while current is not None:
                    path.insert(0, current.tool_id)
                    current = current.parent
                return path
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace.to_dict(),
            "node_count": len(self.nodes),
            "execution_path": self.get_execution_path(),
            "has_failures": len(self.get_failed_nodes()) > 0,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceGraph":
        trace = Trace.from_dict(data["trace"])
        return cls(trace)

    @classmethod
    def from_json(cls, json_str: str) -> "TraceGraph":
        return cls.from_dict(json.loads(json_str))

    def __len__(self) -> int:
        return len(self.nodes)

    def __bool__(self) -> bool:
        return len(self.nodes) > 0


class TraceGraphBuilder:
    """Builder for constructing trace graphs incrementally."""

    def __init__(self, trace_id: str, metadata: dict[str, Any] | None = None) -> None:
        self.trace_id = trace_id
        self.metadata = metadata or {}
        self.steps: list[TraceStep] = []
        self.success = True
        self.created_at = datetime.now(timezone.utc)

    def add_step(self, tool_id: str, input_data: dict[str, Any],
                 output_data: dict[str, Any], success: bool,
                 duration_ms: float = 0.0,
                 metadata: dict[str, Any] | None = None) -> TraceStep:
        step = TraceStep(
            tool_id=tool_id, input_data=input_data, output_data=output_data,
            success=success, duration_ms=duration_ms,
            order=len(self.steps), metadata=metadata or {})
        self.steps.append(step)
        if not success:
            self.success = False
        return step

    def build_trace(self) -> Trace:
        return Trace(
            trace_id=self.trace_id, steps=self.steps.copy(),
            success=self.success, created_at=self.created_at,
            metadata=self.metadata)

    def build(self) -> TraceGraph:
        return TraceGraph(self.build_trace())

    def __len__(self) -> int:
        return len(self.steps)


def build_trace_from_steps(trace_id: str, steps: list[dict[str, Any]],
                           metadata: dict[str, Any] | None = None) -> TraceGraph:
    """Build a trace graph from a list of step dictionaries."""
    builder = TraceGraphBuilder(trace_id, metadata)
    for step_data in steps:
        builder.add_step(
            tool_id=step_data["tool_id"],
            input_data=step_data.get("input_data", {}),
            output_data=step_data.get("output_data", {}),
            success=step_data.get("success", True),
            duration_ms=step_data.get("duration_ms", 0.0),
            metadata=step_data.get("metadata", {}))
    return builder.build()


def get_tool_statistics(graphs: list[TraceGraph]) -> dict[str, dict[str, Any]]:
    """Calculate statistics for tools across multiple traces."""
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total_calls": 0, "successful_calls": 0, "failed_calls": 0,
                 "total_duration_ms": 0.0, "success_rate": 0.0})
    for graph in graphs:
        for node in graph.iter_nodes():
            tool_stats = stats[node.tool_id]
            tool_stats["total_calls"] += 1
            tool_stats["total_duration_ms"] += node.step.duration_ms
            if node.success:
                tool_stats["successful_calls"] += 1
            else:
                tool_stats["failed_calls"] += 1
    for tool_stats in stats.values():
        if tool_stats["total_calls"] > 0:
            tool_stats["success_rate"] = tool_stats["successful_calls"] / tool_stats["total_calls"]
    return dict(stats)
