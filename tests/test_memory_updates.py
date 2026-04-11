"""Tests for the memory updater module."""

import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from backpropagate.memory import (
    MemoryUpdater,
    ToolMemory,
    MemoryStats,
    DEFAULT_MAX_SIGNALS_PER_TOOL,
)
from backpropagate.contracts import LearningSignal, SignalSource


class TestToolMemory:
    """Tests for ToolMemory class."""

    def test_add_signal(self):
        """Test adding a signal to tool memory."""
        memory = ToolMemory(tool_id="test_tool")
        signal = LearningSignal(tool_id="test_tool", delta=0.5)
        memory.add_signal(signal)
        assert len(memory.signals) == 1

    def test_get_current_score(self):
        """Test calculating current score."""
        memory = ToolMemory(tool_id="test_tool")
        memory.add_signal(LearningSignal(tool_id="test_tool", delta=0.6))
        memory.add_signal(LearningSignal(tool_id="test_tool", delta=0.4))
        score = memory.get_current_score(apply_decay=False)
        assert 0.4 <= score <= 0.6  # Weighted average

    def test_compaction_on_overflow(self):
        """Test that memory compacts when exceeding max."""
        memory = ToolMemory(tool_id="test_tool", max_signals=10)
        for i in range(15):
            memory.add_signal(LearningSignal(tool_id="test_tool", delta=0.1))
        assert len(memory.signals) <= 10

    def test_get_signals_for_pattern(self):
        """Test filtering signals by pattern."""
        memory = ToolMemory(tool_id="test_tool")
        memory.add_signal(LearningSignal(tool_id="test_tool", delta=0.5, input_pattern="query:*"))
        memory.add_signal(LearningSignal(tool_id="test_tool", delta=0.3, input_pattern="search:*"))
        memory.add_signal(LearningSignal(tool_id="test_tool", delta=0.2))

        query_signals = memory.get_signals_for_pattern("query:*")
        assert len(query_signals) == 1
        assert query_signals[0].input_pattern == "query:*"

    def test_rollback_to_timestamp(self):
        """Test rolling back to a specific time."""
        memory = ToolMemory(tool_id="test_tool")
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        new_time = datetime.now(timezone.utc)

        memory.add_signal(LearningSignal(tool_id="test_tool", delta=0.5, timestamp=old_time))
        memory.add_signal(LearningSignal(tool_id="test_tool", delta=0.3, timestamp=new_time))

        removed = memory.rollback_to(old_time)
        assert removed == 1
        assert len(memory.signals) == 1

    def test_clear_signals(self):
        """Test clearing all signals."""
        memory = ToolMemory(tool_id="test_tool")
        memory.add_signal(LearningSignal(tool_id="test_tool", delta=0.5))
        memory.add_signal(LearningSignal(tool_id="test_tool", delta=0.3))
        count = memory.clear()
        assert count == 2
        assert len(memory.signals) == 0

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        memory = ToolMemory(tool_id="test_tool")
        memory.add_signal(LearningSignal(tool_id="test_tool", delta=0.5))
        data = memory.to_dict()
        restored = ToolMemory.from_dict(data)
        assert restored.tool_id == memory.tool_id
        assert len(restored.signals) == len(memory.signals)


class TestMemoryUpdater:
    """Tests for MemoryUpdater class."""

    def test_update_creates_tool_memory(self, memory_updater):
        """Test that update creates tool memory if needed."""
        signal = LearningSignal(tool_id="new_tool", delta=0.5)
        memory_updater.update(signal)
        assert "new_tool" in memory_updater

    def test_update_batch(self, memory_updater):
        """Test batch update of signals."""
        signals = [
            LearningSignal(tool_id="tool1", delta=0.5),
            LearningSignal(tool_id="tool2", delta=0.3),
            LearningSignal(tool_id="tool1", delta=0.2),
        ]
        count = memory_updater.update_batch(signals)
        assert count == 3
        assert "tool1" in memory_updater
        assert "tool2" in memory_updater

    def test_get_tool_score(self, memory_updater):
        """Test getting tool score."""
        signal = LearningSignal(tool_id="test_tool", delta=0.5)
        memory_updater.update(signal)
        score = memory_updater.get_tool_score("test_tool", apply_decay=False)
        assert score == 0.5

    def test_get_tool_score_nonexistent(self, memory_updater):
        """Test getting score for non-existent tool."""
        score = memory_updater.get_tool_score("nonexistent")
        assert score == 0.0

    def test_get_all_scores(self, memory_updater):
        """Test getting all tool scores."""
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.5))
        memory_updater.update(LearningSignal(tool_id="tool2", delta=0.3))
        scores = memory_updater.get_all_scores(apply_decay=False)
        assert "tool1" in scores
        assert "tool2" in scores

    def test_get_signals_by_tool(self, memory_updater):
        """Test getting signals filtered by tool."""
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.5))
        memory_updater.update(LearningSignal(tool_id="tool2", delta=0.3))
        signals = memory_updater.get_signals(tool_id="tool1")
        assert len(signals) == 1
        assert signals[0].tool_id == "tool1"

    def test_get_signals_since(self, memory_updater):
        """Test getting signals since a timestamp."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.5, timestamp=old_time))
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.3))

        since = datetime.now(timezone.utc) - timedelta(hours=1)
        signals = memory_updater.get_signals(since=since)
        assert len(signals) == 1


class TestMemoryUpdaterRollback:
    """Tests for memory rollback functionality (P0: reversible learning)."""

    def test_rollback_all_tools(self, memory_updater):
        """Test rolling back all tools."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.5, timestamp=old_time))
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.3))
        memory_updater.update(LearningSignal(tool_id="tool2", delta=0.4))

        removed = memory_updater.rollback(old_time)
        assert removed == 2  # Two new signals removed

    def test_rollback_specific_tool(self, memory_updater):
        """Test rolling back a specific tool."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.5, timestamp=old_time))
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.3))
        memory_updater.update(LearningSignal(tool_id="tool2", delta=0.4))

        removed = memory_updater.rollback(old_time, tool_id="tool1")
        assert removed == 1  # Only tool1's new signal

    def test_apply_rolling_window(self, memory_updater):
        """Test applying rolling window expiration."""
        # Create a memory with a 1-day window
        memory = MemoryUpdater(rolling_window_days=1)

        # Add old signal (should be removed)
        old_time = datetime.now(timezone.utc) - timedelta(days=2)
        memory.update(LearningSignal(tool_id="tool1", delta=0.5, timestamp=old_time))

        # Add recent signal (should be kept)
        memory.update(LearningSignal(tool_id="tool1", delta=0.3))

        removed = memory.apply_rolling_window()
        assert removed == 1
        assert len(memory) == 1

    def test_apply_rolling_window_keeps_recent_removes_old(self):
        """Test that rolling window removes old signals and keeps recent ones."""
        memory = MemoryUpdater(rolling_window_days=7)
        now = datetime.now(timezone.utc)

        # Add 3 old signals (older than 7 days, should be removed)
        for i in range(3):
            old_time = now - timedelta(days=10 + i)
            memory.update(LearningSignal(
                tool_id="tool_a", delta=0.1 * i, timestamp=old_time))

        # Add 2 recent signals (within 7 days, should be kept)
        for i in range(2):
            recent_time = now - timedelta(days=i)
            memory.update(LearningSignal(
                tool_id="tool_a", delta=0.5 + 0.1 * i, timestamp=recent_time))

        assert len(memory) == 5

        removed = memory.apply_rolling_window()

        assert removed == 3, "Should remove 3 old signals"
        assert len(memory) == 2, "Should keep 2 recent signals"

        # Verify the kept signals are the recent ones
        remaining = memory.get_signals(tool_id="tool_a")
        for sig in remaining:
            age = now - sig.timestamp
            assert age.days <= 7, (
                f"Signal with age {age.days} days should have been removed")


class TestMemoryUpdaterNegativeSignals:
    """Tests for negative signal handling."""

    def test_negative_signal_stored(self, memory_updater):
        """Test that negative signals are stored correctly."""
        signal = LearningSignal(tool_id="test_tool", delta=-0.5)
        memory_updater.update(signal)
        signals = memory_updater.get_signals()
        assert len(signals) == 1
        assert signals[0].delta == -0.5

    def test_negative_signals_affect_score(self, memory_updater):
        """Test that negative signals reduce score."""
        memory_updater.update(LearningSignal(tool_id="test_tool", delta=0.5))
        memory_updater.update(LearningSignal(tool_id="test_tool", delta=-0.5))
        score = memory_updater.get_tool_score("test_tool", apply_decay=False)
        assert score == 0.0  # They cancel out

    def test_all_negative_signals(self, memory_updater):
        """Test tool with only negative signals."""
        memory_updater.update(LearningSignal(tool_id="test_tool", delta=-0.5))
        memory_updater.update(LearningSignal(tool_id="test_tool", delta=-0.3))
        score = memory_updater.get_tool_score("test_tool", apply_decay=False)
        assert score < 0


class TestMemoryStats:
    """Tests for memory statistics."""

    def test_get_stats(self, memory_updater):
        """Test getting memory statistics."""
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.5))
        memory_updater.update(LearningSignal(tool_id="tool2", delta=0.3))
        stats = memory_updater.get_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.total_signals == 2
        assert stats.total_tools == 2
        assert stats.signals_by_tool == {"tool1": 1, "tool2": 1}

    def test_stats_timestamps(self, memory_updater):
        """Test that stats include timestamp info."""
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.5))
        stats = memory_updater.get_stats()
        assert stats.oldest_signal is not None
        assert stats.newest_signal is not None


class TestMemoryPersistence:
    """Tests for memory persistence (save/load)."""

    def test_save_and_load(self, memory_updater, tmp_path):
        """Test saving and loading memory."""
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.5))
        memory_updater.update(LearningSignal(tool_id="tool2", delta=0.3))

        path = tmp_path / "memory.json"
        memory_updater.save(path)

        loaded = MemoryUpdater.load(path)
        assert len(loaded) == 2
        assert "tool1" in loaded
        assert "tool2" in loaded

    def test_save_creates_parent_dirs(self, memory_updater, tmp_path):
        """Test that save creates parent directories."""
        path = tmp_path / "nested" / "path" / "memory.json"
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.5))
        memory_updater.save(path)
        assert path.exists()

    def test_serialization_roundtrip(self, memory_updater):
        """Test serialization and deserialization."""
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.5))
        data = memory_updater.to_dict()
        restored = MemoryUpdater.from_dict(data)
        assert len(restored) == len(memory_updater)


class TestToolIsolation:
    """Tests for tool isolation (P1: scope learning per tool)."""

    def test_tools_isolated(self, memory_updater):
        """Test that tools have isolated memory."""
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.8))
        memory_updater.update(LearningSignal(tool_id="tool2", delta=-0.5))

        score1 = memory_updater.get_tool_score("tool1", apply_decay=False)
        score2 = memory_updater.get_tool_score("tool2", apply_decay=False)

        assert score1 == 0.8
        assert score2 == -0.5

    def test_pattern_isolation(self, memory_updater):
        """Test that patterns provide isolation within tools."""
        memory_updater.update(
            LearningSignal(tool_id="tool1", delta=0.8, input_pattern="pattern_a")
        )
        memory_updater.update(
            LearningSignal(tool_id="tool1", delta=-0.5, input_pattern="pattern_b")
        )

        score_a = memory_updater.get_tool_score("tool1", pattern="pattern_a")
        score_b = memory_updater.get_tool_score("tool1", pattern="pattern_b")

        assert score_a > 0
        assert score_b < 0

    def test_clear_specific_tool(self, memory_updater):
        """Test clearing only a specific tool."""
        memory_updater.update(LearningSignal(tool_id="tool1", delta=0.5))
        memory_updater.update(LearningSignal(tool_id="tool2", delta=0.3))

        memory_updater.clear(tool_id="tool1")

        assert "tool1" not in memory_updater
        assert "tool2" in memory_updater
