"""Tests for the trace graph module."""

import pytest
import json

from backpropagate.trace import (
    TraceGraph,
    TraceGraphBuilder,
    TraceNode,
    build_trace_from_steps,
    get_tool_statistics,
)
from backpropagate.contracts import Trace, TraceStep


class TestTraceGraph:
    """Tests for TraceGraph class."""

    def test_build_from_trace(self, sample_trace):
        """Test building a graph from a trace."""
        graph = TraceGraph(sample_trace)
        assert len(graph) == 2
        assert graph.trace_id == "test-trace-001"

    def test_empty_trace(self):
        """Test handling of empty trace."""
        trace = Trace(trace_id="empty", steps=[])
        graph = TraceGraph(trace)
        assert len(graph) == 0
        assert graph.root is None

    def test_get_node_by_order(self, sample_trace):
        """Test getting a node by its order."""
        graph = TraceGraph(sample_trace)
        node = graph.get_node(0)
        assert node is not None
        assert node.tool_id == "search_tool"

    def test_get_tool_nodes(self, sample_trace):
        """Test getting all nodes for a specific tool."""
        graph = TraceGraph(sample_trace)
        nodes = graph.get_tool_nodes("search_tool")
        assert len(nodes) == 1
        assert nodes[0].tool_id == "search_tool"

    def test_get_failed_nodes(self, failed_trace):
        """Test getting failed nodes."""
        graph = TraceGraph(failed_trace)
        failed = graph.get_failed_nodes()
        assert len(failed) == 1
        assert failed[0].tool_id == "parse_tool"

    def test_get_successful_nodes(self, sample_trace):
        """Test getting successful nodes."""
        graph = TraceGraph(sample_trace)
        successful = graph.get_successful_nodes()
        assert len(successful) == 2

    def test_iter_nodes_in_order(self, sample_trace):
        """Test iterating nodes in execution order."""
        graph = TraceGraph(sample_trace)
        tool_ids = [n.tool_id for n in graph.iter_nodes()]
        assert tool_ids == ["search_tool", "summarize_tool"]

    def test_get_execution_path(self, sample_trace):
        """Test getting the execution path."""
        graph = TraceGraph(sample_trace)
        path = graph.get_execution_path()
        assert path == ["search_tool", "summarize_tool"]

    def test_get_failure_path(self, failed_trace):
        """Test getting path to first failure."""
        graph = TraceGraph(failed_trace)
        path = graph.get_failure_path()
        assert path == ["search_tool", "parse_tool"]

    def test_no_failure_path_on_success(self, sample_trace):
        """Test that successful traces have empty failure path."""
        graph = TraceGraph(sample_trace)
        path = graph.get_failure_path()
        assert path == []


class TestTraceGraphSerialization:
    """Tests for trace graph serialization."""

    def test_to_dict(self, sample_trace):
        """Test serialization to dictionary."""
        graph = TraceGraph(sample_trace)
        data = graph.to_dict()
        assert "trace" in data
        assert data["node_count"] == 2
        assert data["execution_path"] == ["search_tool", "summarize_tool"]

    def test_to_json(self, sample_trace):
        """Test serialization to JSON."""
        graph = TraceGraph(sample_trace)
        json_str = graph.to_json()
        data = json.loads(json_str)
        assert "trace" in data

    def test_from_dict_roundtrip(self, sample_trace):
        """Test serialization roundtrip."""
        graph = TraceGraph(sample_trace)
        data = graph.to_dict()
        restored = TraceGraph.from_dict(data)
        assert len(restored) == len(graph)
        assert restored.trace_id == graph.trace_id

    def test_from_json_roundtrip(self, sample_trace):
        """Test JSON roundtrip."""
        graph = TraceGraph(sample_trace)
        json_str = graph.to_json()
        restored = TraceGraph.from_json(json_str)
        assert restored.trace_id == graph.trace_id


class TestTraceGraphBuilder:
    """Tests for TraceGraphBuilder class."""

    def test_build_empty_trace(self):
        """Test building an empty trace."""
        builder = TraceGraphBuilder("test-trace")
        graph = builder.build()
        assert len(graph) == 0

    def test_add_steps_sequentially(self):
        """Test adding steps in sequence."""
        builder = TraceGraphBuilder("test-trace")
        builder.add_step(
            tool_id="tool1",
            input_data={"a": 1},
            output_data={"b": 2},
            success=True,
        )
        builder.add_step(
            tool_id="tool2",
            input_data={"c": 3},
            output_data={"d": 4},
            success=True,
        )
        graph = builder.build()
        assert len(graph) == 2
        path = graph.get_execution_path()
        assert path == ["tool1", "tool2"]

    def test_failure_updates_success_flag(self):
        """Test that failed step updates overall success."""
        builder = TraceGraphBuilder("test-trace")
        builder.add_step("tool1", {}, {}, success=True)
        builder.add_step("tool2", {}, {}, success=False)
        trace = builder.build_trace()
        assert trace.success is False

    def test_step_ordering(self):
        """Test that steps are ordered correctly."""
        builder = TraceGraphBuilder("test-trace")
        step1 = builder.add_step("tool1", {}, {}, success=True)
        step2 = builder.add_step("tool2", {}, {}, success=True)
        assert step1.order == 0
        assert step2.order == 1

    def test_metadata_preserved(self):
        """Test that metadata is preserved."""
        metadata = {"session": "abc"}
        builder = TraceGraphBuilder("test-trace", metadata=metadata)
        trace = builder.build_trace()
        assert trace.metadata == metadata


class TestBuildTraceFromSteps:
    """Tests for build_trace_from_steps function."""

    def test_build_from_dict_steps(self):
        """Test building from dictionary step data."""
        steps = [
            {"tool_id": "tool1", "input_data": {}, "output_data": {}, "success": True},
            {"tool_id": "tool2", "input_data": {}, "output_data": {}, "success": True},
        ]
        graph = build_trace_from_steps("trace-001", steps)
        assert len(graph) == 2
        assert graph.get_execution_path() == ["tool1", "tool2"]

    def test_build_with_metadata(self):
        """Test building with trace metadata."""
        steps = [{"tool_id": "tool1", "input_data": {}, "output_data": {}, "success": True}]
        metadata = {"user": "test"}
        graph = build_trace_from_steps("trace-001", steps, metadata=metadata)
        assert graph.trace.metadata == metadata


class TestTraceNode:
    """Tests for TraceNode class."""

    def test_node_properties(self):
        """Test node property accessors."""
        step = TraceStep(
            tool_id="test_tool",
            input_data={},
            output_data={},
            success=True,
            order=0,
        )
        node = TraceNode(step=step)
        assert node.tool_id == "test_tool"
        assert node.success is True
        assert node.order == 0

    def test_node_depth(self):
        """Test node depth calculation."""
        step1 = TraceStep(tool_id="t1", input_data={}, output_data={}, success=True)
        step2 = TraceStep(tool_id="t2", input_data={}, output_data={}, success=True)
        step3 = TraceStep(tool_id="t3", input_data={}, output_data={}, success=True)

        root = TraceNode(step=step1)
        child = TraceNode(step=step2)
        grandchild = TraceNode(step=step3)

        root.add_child(child)
        child.add_child(grandchild)

        assert root.depth() == 0
        assert child.depth() == 1
        assert grandchild.depth() == 2


class TestToolStatistics:
    """Tests for tool statistics calculation."""

    def test_stats_from_single_trace(self, sample_trace):
        """Test calculating stats from a single trace."""
        graph = TraceGraph(sample_trace)
        stats = get_tool_statistics([graph])

        assert "search_tool" in stats
        assert "summarize_tool" in stats
        assert stats["search_tool"]["total_calls"] == 1
        assert stats["search_tool"]["success_rate"] == 1.0

    def test_stats_from_multiple_traces(self, sample_trace, failed_trace):
        """Test calculating stats from multiple traces."""
        graph1 = TraceGraph(sample_trace)
        graph2 = TraceGraph(failed_trace)
        stats = get_tool_statistics([graph1, graph2])

        # search_tool appears in both traces
        assert stats["search_tool"]["total_calls"] == 2
        assert stats["search_tool"]["successful_calls"] == 2

        # parse_tool failed in the second trace
        assert stats["parse_tool"]["failed_calls"] == 1
        assert stats["parse_tool"]["success_rate"] == 0.0

    def test_empty_graphs_list(self):
        """Test stats with empty list."""
        stats = get_tool_statistics([])
        assert stats == {}


class TestTraceValidation:
    """Tests for trace validation."""

    def test_trace_with_steps_is_valid(self, sample_trace):
        """Test that trace with steps is valid."""
        assert sample_trace.is_valid()

    def test_empty_trace_is_invalid(self):
        """Test that empty trace is invalid."""
        trace = Trace(trace_id="empty", steps=[])
        assert not trace.is_valid()

    def test_trace_requires_id(self):
        """Test that trace requires an ID."""
        with pytest.raises(ValueError):
            Trace(trace_id="", steps=[])

    def test_step_requires_tool_id(self):
        """Test that step requires a tool ID."""
        with pytest.raises(ValueError):
            TraceStep(
                tool_id="",
                input_data={},
                output_data={},
                success=True,
            )
