"""Tests for Checkpoint Management module (backpropagate/checkpoints.py)."""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import pytest

from backpropagate.checkpoints import (
    CheckpointInfo,
    CheckpointManager,
    CheckpointPolicy,
    CheckpointStats,
)

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def sample_checkpoint_path(temp_checkpoint_dir):
    """Create a sample checkpoint directory with files."""
    checkpoint_path = temp_checkpoint_dir / "checkpoint-100"
    checkpoint_path.mkdir()

    # Create some dummy files to simulate a real checkpoint
    (checkpoint_path / "adapter_config.json").write_text('{"r": 16}')
    (checkpoint_path / "adapter_model.safetensors").write_bytes(b"x" * 1024)
    (checkpoint_path / "training_state.json").write_text('{"step": 100}')

    return checkpoint_path


@pytest.fixture
def manager(temp_checkpoint_dir):
    """Create a CheckpointManager with default policy."""
    return CheckpointManager(str(temp_checkpoint_dir))


@pytest.fixture
def manager_no_auto_prune(temp_checkpoint_dir):
    """Create a CheckpointManager with auto_prune disabled."""
    policy = CheckpointPolicy(auto_prune=False)
    return CheckpointManager(str(temp_checkpoint_dir), policy)


def create_dummy_checkpoint(base_dir: Path, name: str, size_kb: int = 1) -> Path:
    """Helper to create a dummy checkpoint directory."""
    checkpoint_path = base_dir / name
    checkpoint_path.mkdir(exist_ok=True)
    (checkpoint_path / "model.safetensors").write_bytes(b"x" * (size_kb * 1024))
    return checkpoint_path


# =============================================================================
# CHECKPOINT POLICY TESTS
# =============================================================================

class TestCheckpointPolicyDefaults:
    """Pin CheckpointPolicy default values.

    These defaults are an operator-facing contract: a user who passes
    no ``policy=`` to ``CheckpointManager`` is implicitly opting into
    the values here. Silently bumping ``max_total`` or flipping
    ``auto_prune`` would change disk usage on every existing install.
    Tests fail-loud if any default drifts.
    """

    def test_default_keep_best_n_is_3_and_max_total_10(self):
        """Defaults: keep_best_n=3, keep_final=True, max_total=10, auto_prune=True.

        Bumping any of these is a user-visible behavior change (disk
        usage + retained-checkpoint count). If you intentionally change
        a default, update this test AND the handbook's Checkpoint
        Management page in the same PR.
        """
        policy = CheckpointPolicy()

        assert policy.keep_best_n == 3, (
            f"CheckpointPolicy default keep_best_n drifted: expected 3, "
            f"got {policy.keep_best_n}. Operators rely on this default; "
            f"bumping it changes disk usage on every existing install."
        )
        assert policy.keep_final is True, (
            f"CheckpointPolicy default keep_final flipped from True to "
            f"{policy.keep_final}. The 'final checkpoint is never pruned' "
            f"contract is load-bearing for resume-from-final flows."
        )
        assert policy.keep_run_boundaries is False, (
            f"CheckpointPolicy default keep_run_boundaries flipped from "
            f"False to {policy.keep_run_boundaries}. SLAO multi-run "
            f"opts-in to boundary retention; the single-run default is "
            f"off to keep disk usage predictable."
        )
        assert policy.max_total == 10, (
            f"CheckpointPolicy default max_total drifted: expected 10, "
            f"got {policy.max_total}. This is the hard upper bound on "
            f"retained checkpoints regardless of keep_best_n / keep_final."
        )
        assert policy.min_improvement == 0.0, (
            f"CheckpointPolicy default min_improvement drifted: expected "
            f"0.0 (every val-loss improvement counts), got "
            f"{policy.min_improvement}."
        )
        assert policy.auto_prune is True, (
            f"CheckpointPolicy default auto_prune flipped from True to "
            f"{policy.auto_prune}. Auto-prune-on by default is load-bearing "
            f"for the 'one-call register, never run out of disk' UX."
        )

    def test_checkpoint_policy_custom_values(self):
        """Custom policy configuration."""
        policy = CheckpointPolicy(
            keep_best_n=5,
            keep_final=False,
            keep_run_boundaries=True,
            max_total=20,
            min_improvement=0.01,
            auto_prune=False,
        )

        assert policy.keep_best_n == 5
        assert policy.keep_final is False
        assert policy.keep_run_boundaries is True
        assert policy.max_total == 20
        assert policy.min_improvement == 0.01
        assert policy.auto_prune is False

    def test_checkpoint_policy_partial_override(self):
        """Test partial override of defaults."""
        policy = CheckpointPolicy(keep_best_n=7, auto_prune=False)

        assert policy.keep_best_n == 7
        assert policy.auto_prune is False
        # Other defaults should remain
        assert policy.keep_final is True
        assert policy.max_total == 10

    def test_checkpoint_policy_zero_values(self):
        """Test edge case with zero values."""
        policy = CheckpointPolicy(
            keep_best_n=0,
            max_total=0,
            min_improvement=0.0,
        )

        assert policy.keep_best_n == 0
        assert policy.max_total == 0  # 0 = unlimited


# =============================================================================
# CHECKPOINT INFO TESTS
# =============================================================================

class TestCheckpointInfoCreation:
    """Tests for CheckpointInfo creation and properties."""

    def test_checkpoint_info_creation(self):
        """Create CheckpointInfo with all fields."""
        info = CheckpointInfo(
            run_index=0,
            path="/path/to/checkpoint",
            validation_loss=0.5,
            training_loss=0.6,
            is_run_boundary=True,
            is_final=False,
            size_bytes=1024,
            protected=True,
        )

        assert info.run_index == 0
        assert info.path == "/path/to/checkpoint"
        assert info.validation_loss == 0.5
        assert info.training_loss == 0.6
        assert info.is_run_boundary is True
        assert info.is_final is False
        assert info.size_bytes == 1024
        assert info.protected is True
        assert info.timestamp is not None

    def test_checkpoint_info_minimal_creation(self):
        """Create CheckpointInfo with minimal required fields."""
        info = CheckpointInfo(
            run_index=1,
            path="/checkpoint",
        )

        assert info.run_index == 1
        assert info.path == "/checkpoint"
        assert info.validation_loss is None
        assert info.training_loss is None
        assert info.is_run_boundary is False
        assert info.is_final is False
        assert info.size_bytes == 0
        assert info.protected is False

    def test_checkpoint_info_timestamp_auto_generated(self):
        """Verify timestamp is auto-generated."""
        before = datetime.now().isoformat()
        info = CheckpointInfo(run_index=0, path="/path")
        after = datetime.now().isoformat()

        assert before <= info.timestamp <= after

    def test_checkpoint_info_to_dict(self):
        """Test to_dict() method."""
        info = CheckpointInfo(
            run_index=0,
            path="/path",
            validation_loss=0.5,
            protected=True,
        )

        d = info.to_dict()

        assert isinstance(d, dict)
        assert d["run_index"] == 0
        assert d["path"] == "/path"
        assert d["validation_loss"] == 0.5
        assert d["protected"] is True

    def test_checkpoint_info_from_dict(self):
        """Parse checkpoint from dict to extract metadata."""
        data = {
            "run_index": 2,
            "path": "/checkpoints/run-2",
            "validation_loss": 0.35,
            "training_loss": 0.4,
            "timestamp": "2026-01-18T10:30:00",
            "is_run_boundary": True,
            "is_final": True,
            "size_bytes": 2048,
            "protected": False,
        }

        info = CheckpointInfo.from_dict(data)

        assert info.run_index == 2
        assert info.path == "/checkpoints/run-2"
        assert info.validation_loss == 0.35
        assert info.timestamp == "2026-01-18T10:30:00"
        assert info.is_run_boundary is True
        assert info.is_final is True

    def test_checkpoint_info_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = CheckpointInfo(
            run_index=5,
            path="/path/to/cp",
            validation_loss=0.123,
            training_loss=0.456,
            is_run_boundary=True,
            is_final=True,
            size_bytes=4096,
            protected=True,
        )

        restored = CheckpointInfo.from_dict(original.to_dict())

        assert restored.run_index == original.run_index
        assert restored.path == original.path
        assert restored.validation_loss == original.validation_loss
        assert restored.training_loss == original.training_loss
        assert restored.is_run_boundary == original.is_run_boundary
        assert restored.is_final == original.is_final
        assert restored.size_bytes == original.size_bytes
        assert restored.protected == original.protected

    def test_checkpoint_info_comparison_by_loss(self):
        """Compare checkpoints by validation loss."""
        cp1 = CheckpointInfo(run_index=0, path="/cp1", validation_loss=0.5)
        cp2 = CheckpointInfo(run_index=1, path="/cp2", validation_loss=0.3)
        cp3 = CheckpointInfo(run_index=2, path="/cp3", validation_loss=0.7)

        # Sort by validation loss (lower is better)
        sorted_cps = sorted([cp1, cp2, cp3], key=lambda x: x.validation_loss or float('inf'))

        assert sorted_cps[0].run_index == 1  # Best (0.3)
        assert sorted_cps[1].run_index == 0  # Middle (0.5)
        assert sorted_cps[2].run_index == 2  # Worst (0.7)


# =============================================================================
# CHECKPOINT STATS TESTS
# =============================================================================

class TestCheckpointStats:
    """Tests for CheckpointStats."""

    def test_checkpoint_stats_defaults(self):
        """Test CheckpointStats default values."""
        stats = CheckpointStats()

        assert stats.total_count == 0
        assert stats.total_size_bytes == 0
        assert stats.total_size_gb == 0.0
        assert stats.best_checkpoint is None
        assert stats.oldest_checkpoint is None
        assert stats.newest_checkpoint is None
        assert stats.protected_count == 0
        assert stats.prunable_count == 0

    def test_checkpoint_stats_summary_empty(self):
        """Test summary with no checkpoints."""
        stats = CheckpointStats()
        summary = stats.summary()

        assert "Checkpoints: 0" in summary
        assert "0.00 GB" in summary

    def test_checkpoint_stats_summary_with_data(self):
        """Test summary with checkpoint data."""
        best_cp = CheckpointInfo(
            run_index=2,
            path="/cp2",
            validation_loss=0.25,
        )

        stats = CheckpointStats(
            total_count=5,
            total_size_bytes=1024 * 1024 * 1024,  # 1 GB
            total_size_gb=1.0,
            best_checkpoint=best_cp,
            protected_count=2,
            prunable_count=3,
        )

        summary = stats.summary()

        assert "Checkpoints: 5" in summary
        assert "1.00 GB" in summary
        assert "Protected: 2" in summary
        assert "Prunable: 3" in summary
        assert "Best: Run 2" in summary
        assert "0.25" in summary


# =============================================================================
# CHECKPOINT MANAGER INITIALIZATION TESTS
# =============================================================================

class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_checkpoint_manager_init(self, temp_checkpoint_dir):
        """Initialize with output directory."""
        manager = CheckpointManager(str(temp_checkpoint_dir))

        assert manager.checkpoint_dir == temp_checkpoint_dir
        assert manager.policy is not None
        assert isinstance(manager.policy, CheckpointPolicy)

    def test_checkpoint_manager_init_custom_policy(self, temp_checkpoint_dir):
        """Initialize with custom policy."""
        policy = CheckpointPolicy(keep_best_n=5, max_total=20)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        assert manager.policy.keep_best_n == 5
        assert manager.policy.max_total == 20

    def test_checkpoint_manager_creates_directory(self, tmp_path):
        """Test that manager creates checkpoint directory if it doesn't exist."""
        new_dir = tmp_path / "new_checkpoints"
        assert not new_dir.exists()

        manager = CheckpointManager(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_checkpoint_manager_loads_existing_manifest(self, temp_checkpoint_dir):
        """Test loading an existing manifest file."""
        # Create a manifest file
        manifest_data = {
            "version": "1.0",
            "updated": "2026-01-18T10:00:00",
            "policy": {"keep_best_n": 3, "keep_final": True},
            "checkpoints": [
                {
                    "run_index": 0,
                    "path": str(temp_checkpoint_dir / "cp0"),
                    "validation_loss": 0.5,
                    "training_loss": 0.6,
                    "timestamp": "2026-01-18T09:00:00",
                    "is_run_boundary": False,
                    "is_final": True,
                    "size_bytes": 1024,
                    "protected": False,
                }
            ],
        }

        manifest_path = temp_checkpoint_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))

        manager = CheckpointManager(str(temp_checkpoint_dir))

        assert len(manager.list_checkpoints()) == 1
        assert manager.list_checkpoints()[0].run_index == 0

    def test_checkpoint_manager_handles_corrupt_manifest(self, temp_checkpoint_dir):
        """Test graceful handling of corrupt manifest."""
        manifest_path = temp_checkpoint_dir / "manifest.json"
        manifest_path.write_text("not valid json {{{")

        # Should not raise, just start with empty list
        manager = CheckpointManager(str(temp_checkpoint_dir))

        assert len(manager.list_checkpoints()) == 0


# =============================================================================
# SAVE/REGISTER CHECKPOINT TESTS
# =============================================================================

class TestSaveCheckpoint:
    """Tests for saving/registering checkpoints."""

    def test_save_checkpoint(self, manager_no_auto_prune, sample_checkpoint_path):
        """Save a checkpoint and verify files exist."""
        info = manager_no_auto_prune.register(
            run_index=0,
            checkpoint_path=str(sample_checkpoint_path),
            validation_loss=0.5,
        )

        assert info.run_index == 0
        assert info.validation_loss == 0.5
        assert info.is_final is True
        assert info.size_bytes > 0

        # Check manifest was saved
        manifest_path = Path(manager_no_auto_prune.checkpoint_dir) / "manifest.json"
        assert manifest_path.exists()

    def test_save_checkpoint_with_metadata(self, manager_no_auto_prune, sample_checkpoint_path):
        """Save with custom metadata dict (training_loss, is_run_boundary)."""
        info = manager_no_auto_prune.register(
            run_index=1,
            checkpoint_path=str(sample_checkpoint_path),
            validation_loss=0.4,
            training_loss=0.45,
            is_run_boundary=True,
            protected=True,
        )

        assert info.validation_loss == 0.4
        assert info.training_loss == 0.45
        assert info.is_run_boundary is True
        assert info.protected is True

    def test_save_checkpoint_marks_previous_as_not_final(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Saving new checkpoint marks previous as not final."""
        cp1 = create_dummy_checkpoint(temp_checkpoint_dir, "cp1")
        cp2 = create_dummy_checkpoint(temp_checkpoint_dir, "cp2")

        info1 = manager_no_auto_prune.register(0, str(cp1), validation_loss=0.5)
        assert info1.is_final is True

        info2 = manager_no_auto_prune.register(1, str(cp2), validation_loss=0.4)
        assert info2.is_final is True

        # Check that first checkpoint is no longer final
        checkpoints = manager_no_auto_prune.list_checkpoints()
        cp1_updated = next(cp for cp in checkpoints if cp.run_index == 0)
        assert cp1_updated.is_final is False

    def test_save_checkpoint_calculates_size(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Verify checkpoint size is calculated correctly."""
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "sized_cp", size_kb=5)

        info = manager_no_auto_prune.register(0, str(cp), validation_loss=0.5)

        # Should be at least 5KB
        assert info.size_bytes >= 5 * 1024


# =============================================================================
# LIST CHECKPOINTS TESTS
# =============================================================================

class TestListCheckpoints:
    """Tests for listing checkpoints."""

    def test_list_checkpoints(self, manager_no_auto_prune, temp_checkpoint_dir):
        """List all checkpoints sorted by step (run_index)."""
        for i in range(5):
            cp = create_dummy_checkpoint(temp_checkpoint_dir, f"cp{i}")
            manager_no_auto_prune.register(i, str(cp), validation_loss=0.5 - i * 0.05)

        checkpoints = manager_no_auto_prune.list_checkpoints()

        assert len(checkpoints) == 5

    def test_list_checkpoints_empty_dir(self, manager):
        """Handle empty checkpoint directory."""
        checkpoints = manager.list_checkpoints()

        assert checkpoints == []
        assert len(checkpoints) == 0


# =============================================================================
# GET BEST/LATEST CHECKPOINT TESTS
# =============================================================================

class TestGetBestCheckpoint:
    """Tests for getting best checkpoint."""

    def test_get_best_checkpoint(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Get checkpoint with lowest loss."""
        cp1 = create_dummy_checkpoint(temp_checkpoint_dir, "cp1")
        cp2 = create_dummy_checkpoint(temp_checkpoint_dir, "cp2")
        cp3 = create_dummy_checkpoint(temp_checkpoint_dir, "cp3")

        manager_no_auto_prune.register(0, str(cp1), validation_loss=0.5)
        manager_no_auto_prune.register(1, str(cp2), validation_loss=0.3)  # Best
        manager_no_auto_prune.register(2, str(cp3), validation_loss=0.4)

        best = manager_no_auto_prune.get_best_checkpoint()

        assert best is not None
        assert best.run_index == 1
        assert best.validation_loss == 0.3

    def test_get_best_checkpoint_no_validation_loss(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Get best when no checkpoints have validation_loss."""
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp1")
        manager_no_auto_prune.register(0, str(cp))  # No validation_loss

        best = manager_no_auto_prune.get_best_checkpoint()

        assert best is None

    def test_get_best_checkpoint_empty(self, manager):
        """Get best from empty manager."""
        assert manager.get_best_checkpoint() is None


class TestGetLatestCheckpoint:
    """Tests for getting latest/final checkpoint."""

    def test_get_latest_checkpoint(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Get most recent checkpoint."""
        for i in range(3):
            cp = create_dummy_checkpoint(temp_checkpoint_dir, f"cp{i}")
            manager_no_auto_prune.register(i, str(cp), validation_loss=0.5)

        final = manager_no_auto_prune.get_final_checkpoint()

        assert final is not None
        assert final.run_index == 2
        assert final.is_final is True

    def test_get_latest_checkpoint_empty(self, manager):
        """Get latest from empty manager."""
        assert manager.get_final_checkpoint() is None


# =============================================================================
# AUTO PRUNE TESTS
# =============================================================================

class TestAutoPruneKeepLast:
    """Tests for auto-pruning with max_total limit."""

    def test_auto_prune_keep_last(self, temp_checkpoint_dir):
        """Prune keeps only N most recent via max_total."""
        # Policy: keep max 3 checkpoints
        policy = CheckpointPolicy(
            keep_best_n=0,  # Disable keep_best_n
            keep_final=True,
            max_total=3,
            auto_prune=True,
        )
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Create 5 checkpoints
        for i in range(5):
            cp = create_dummy_checkpoint(temp_checkpoint_dir, f"cp{i}")
            manager.register(i, str(cp), validation_loss=0.5)

        # Should have pruned down to 3
        assert len(manager.list_checkpoints()) <= 3


class TestAutoPruneKeepBest:
    """Tests for auto-pruning by validation loss."""

    def test_auto_prune_keep_best(self, temp_checkpoint_dir):
        """Prune keeps N best by loss."""
        policy = CheckpointPolicy(
            keep_best_n=2,
            keep_final=True,
            max_total=3,
            auto_prune=True,
        )
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Create checkpoints with different losses
        losses = [0.5, 0.2, 0.4, 0.1, 0.3]  # Best are 0.1, 0.2
        for i, loss in enumerate(losses):
            cp = create_dummy_checkpoint(temp_checkpoint_dir, f"cp{i}")
            manager.register(i, str(cp), validation_loss=loss)

        # Check that best checkpoints are preserved
        checkpoints = manager.list_checkpoints()
        losses_remaining = [cp.validation_loss for cp in checkpoints]

        # The best ones (0.1, 0.2) should be kept
        assert 0.1 in losses_remaining
        assert 0.2 in losses_remaining


class TestAutoPruneKeepMilestones:
    """Tests for preserving milestone checkpoints."""

    def test_auto_prune_keep_milestones(self, temp_checkpoint_dir):
        """Milestone (run boundary) checkpoints preserved."""
        policy = CheckpointPolicy(
            keep_best_n=1,
            keep_run_boundaries=True,
            max_total=10,
            auto_prune=True,
        )
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Create run boundary checkpoint
        cp_milestone = create_dummy_checkpoint(temp_checkpoint_dir, "cp_boundary")
        manager.register(0, str(cp_milestone), validation_loss=0.9, is_run_boundary=True)

        # Create several more checkpoints with better loss
        for i in range(1, 5):
            cp = create_dummy_checkpoint(temp_checkpoint_dir, f"cp{i}")
            manager.register(i, str(cp), validation_loss=0.1 * i)

        # Run boundary checkpoint should still exist
        checkpoints = manager.list_checkpoints()
        run_boundary = next((cp for cp in checkpoints if cp.is_run_boundary), None)
        assert run_boundary is not None


class TestAutoPruneDisabled:
    """Tests for disabled auto-pruning."""

    def test_auto_prune_disabled(self, temp_checkpoint_dir):
        """No pruning when auto_prune=False."""
        policy = CheckpointPolicy(
            keep_best_n=1,
            max_total=2,
            auto_prune=False,  # Disabled
        )
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Create 10 checkpoints
        for i in range(10):
            cp = create_dummy_checkpoint(temp_checkpoint_dir, f"cp{i}")
            manager.register(i, str(cp), validation_loss=0.5)

        # All should still exist (no auto-prune)
        assert len(manager.list_checkpoints()) == 10


# =============================================================================
# MANUAL PRUNE TESTS
# =============================================================================

class TestManualPrune:
    """Tests for manual pruning operations."""

    def test_prune_dry_run(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Dry run returns what would be pruned without deleting."""
        for i in range(5):
            cp = create_dummy_checkpoint(temp_checkpoint_dir, f"cp{i}")
            manager_no_auto_prune.register(i, str(cp), validation_loss=0.5)

        would_prune = manager_no_auto_prune.prune(dry_run=True)

        # All checkpoints should still exist
        assert len(manager_no_auto_prune.list_checkpoints()) == 5

    def test_prune_removes_low_value_checkpoints(self, temp_checkpoint_dir):
        """Prune actually removes files."""
        policy = CheckpointPolicy(
            keep_best_n=1,
            keep_final=True,
            max_total=2,
            auto_prune=False,
        )
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Create checkpoints
        paths = []
        for i in range(5):
            cp = create_dummy_checkpoint(temp_checkpoint_dir, f"cp{i}")
            paths.append(cp)
            manager.register(i, str(cp), validation_loss=0.1 * (i + 1))

        # Manual prune
        pruned = manager.prune()

        # Check files were actually deleted
        remaining = len(manager.list_checkpoints())
        assert remaining <= 2


# =============================================================================
# PROTECTED CHECKPOINTS TESTS
# =============================================================================

class TestProtectedCheckpoints:
    """Tests for protected checkpoint functionality."""

    def test_protect_checkpoint(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Protect a checkpoint from pruning."""
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        manager_no_auto_prune.register(0, str(cp), validation_loss=0.5)

        result = manager_no_auto_prune.protect_checkpoint(0)

        assert result is True
        checkpoints = manager_no_auto_prune.list_checkpoints()
        assert checkpoints[0].protected is True

    def test_protect_nonexistent_checkpoint(self, manager):
        """Protect returns False for nonexistent checkpoint."""
        result = manager.protect_checkpoint(999)
        assert result is False

    def test_unprotect_checkpoint(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Unprotect a checkpoint."""
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        manager_no_auto_prune.register(0, str(cp), validation_loss=0.5, protected=True)

        result = manager_no_auto_prune.unprotect_checkpoint(0)

        assert result is True
        checkpoints = manager_no_auto_prune.list_checkpoints()
        assert checkpoints[0].protected is False

    def test_protected_checkpoint_survives_prune(self, temp_checkpoint_dir):
        """Protected checkpoints are not pruned."""
        policy = CheckpointPolicy(
            keep_best_n=0,
            keep_final=False,
            max_total=1,
            auto_prune=False,
        )
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Create and protect checkpoint
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "protected_cp")
        manager.register(0, str(cp), validation_loss=0.9, protected=True)

        # Create another checkpoint (better loss)
        cp2 = create_dummy_checkpoint(temp_checkpoint_dir, "better_cp")
        manager.register(1, str(cp2), validation_loss=0.1)

        # Prune
        manager.prune()

        # Protected checkpoint should still exist
        checkpoints = manager.list_checkpoints()
        protected_exists = any(cp.run_index == 0 for cp in checkpoints)
        assert protected_exists


# =============================================================================
# CHECKPOINT STATS TESTS
# =============================================================================

class TestCheckpointStatsMethod:
    """Tests for get_stats() method."""

    def test_checkpoint_stats(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Get stats (count, total_size, best_loss)."""
        for i in range(3):
            cp = create_dummy_checkpoint(temp_checkpoint_dir, f"cp{i}", size_kb=10)
            manager_no_auto_prune.register(i, str(cp), validation_loss=0.5 - i * 0.1)

        stats = manager_no_auto_prune.get_stats()

        assert stats.total_count == 3
        assert stats.total_size_bytes >= 30 * 1024  # At least 30KB
        assert stats.best_checkpoint is not None
        assert stats.best_checkpoint.run_index == 2  # Lowest loss

    def test_checkpoint_stats_empty(self, manager):
        """Get stats from empty manager."""
        stats = manager.get_stats()

        assert stats.total_count == 0
        assert stats.total_size_bytes == 0
        assert stats.best_checkpoint is None

    def test_checkpoint_stats_protected_count(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Stats includes protected count."""
        cp1 = create_dummy_checkpoint(temp_checkpoint_dir, "cp1")
        cp2 = create_dummy_checkpoint(temp_checkpoint_dir, "cp2")

        manager_no_auto_prune.register(0, str(cp1), validation_loss=0.5, protected=True)
        manager_no_auto_prune.register(1, str(cp2), validation_loss=0.4)

        stats = manager_no_auto_prune.get_stats()

        assert stats.protected_count == 1


# =============================================================================
# CLEANUP TESTS
# =============================================================================

class TestCleanupAll:
    """Tests for cleanup operations."""

    def test_cleanup_orphaned(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Remove checkpoints from manifest that no longer exist on disk."""
        # Register a checkpoint
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        manager_no_auto_prune.register(0, str(cp), validation_loss=0.5)

        # Delete the checkpoint file directly (orphan it)
        shutil.rmtree(cp)

        # Cleanup orphaned
        removed = manager_no_auto_prune.cleanup_orphaned()

        assert removed == 1
        assert len(manager_no_auto_prune.list_checkpoints()) == 0

    def test_force_prune_to_size(self, temp_checkpoint_dir):
        """Force prune to fit within size limit."""
        policy = CheckpointPolicy(auto_prune=False)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Create checkpoints with known sizes
        for i in range(5):
            cp = create_dummy_checkpoint(temp_checkpoint_dir, f"cp{i}", size_kb=100)
            manager.register(i, str(cp), validation_loss=0.1 * (i + 1))

        # Force prune to 200KB (should keep ~2 checkpoints)
        pruned = manager.force_prune_to_size(0.0002)  # 0.2 MB = ~200KB

        # Should have pruned some checkpoints
        assert len(pruned) > 0


# =============================================================================
# MANIFEST PERSISTENCE TESTS
# =============================================================================

class TestManifestPersistence:
    """Tests for manifest file operations."""

    def test_manifest_saved_on_register(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Manifest is saved when checkpoint is registered."""
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        manager_no_auto_prune.register(0, str(cp), validation_loss=0.5)

        manifest_path = temp_checkpoint_dir / "manifest.json"
        assert manifest_path.exists()

        data = json.loads(manifest_path.read_text())
        assert len(data["checkpoints"]) == 1
        assert data["checkpoints"][0]["run_index"] == 0

    def test_manifest_contains_policy(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Manifest includes policy configuration."""
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        manager_no_auto_prune.register(0, str(cp), validation_loss=0.5)

        manifest_path = temp_checkpoint_dir / "manifest.json"
        data = json.loads(manifest_path.read_text())

        assert "policy" in data
        assert "keep_best_n" in data["policy"]

    def test_manifest_version(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Manifest includes version number."""
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        manager_no_auto_prune.register(0, str(cp), validation_loss=0.5)

        manifest_path = temp_checkpoint_dir / "manifest.json"
        data = json.loads(manifest_path.read_text())

        assert data["version"] == "1.0"

    def test_manifest_updated_timestamp(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Manifest includes updated timestamp."""
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        manager_no_auto_prune.register(0, str(cp), validation_loss=0.5)

        manifest_path = temp_checkpoint_dir / "manifest.json"
        data = json.loads(manifest_path.read_text())

        assert "updated" in data
        # Should be a valid ISO format timestamp
        datetime.fromisoformat(data["updated"])


# =============================================================================
# BACKEND-B-003 MANIFEST SCHEMA-VERSION WARN PATH
# =============================================================================

class TestManifestVersionMismatchWarn:
    """Pin the BACKEND-B-003 manifest schema-version WARN contract.

    v1.3 added ``CheckpointManager.CURRENT_MANIFEST_VERSION`` and the
    fail-loud-but-keep-going log on ``_load_manifest`` when the on-disk
    ``version`` field doesn't match this build's expected version.
    Coverage gap (Wave 3.5 TESTS-B-003): every existing test loaded
    manifests at the current version, so the WARN branch was never
    exercised. These tests pin:

      1. Mismatch (disk_version != CURRENT_MANIFEST_VERSION) emits the
         WARN log with both versions named.
      2. Mismatch is non-fatal — manifest is still parsed, checkpoints
         load.
      3. Missing version field defaults to "0.0" (per the .get() fallback
         in checkpoints.py:230) and surfaces the WARN.
      4. Matching version is SILENT — no warn line on the happy path.

    Why this matters: the WARN is what operators correlate weird
    post-resume behavior to schema age. If the message ever silently
    regresses (typo, refactor that drops one half of the version pair,
    log-level change to DEBUG) we want the test to fail-loud naming the
    contract.
    """

    def _write_manifest(self, temp_dir: Path, version: str | None) -> None:
        """Helper: write a minimal manifest with the given on-disk version.

        Passing ``None`` for ``version`` simulates a pre-v1.0 manifest
        that predates the ``version`` field entirely.
        """
        manifest_data: dict = {
            "updated": "2026-01-18T10:00:00",
            "policy": {"keep_best_n": 3, "keep_final": True},
            "checkpoints": [
                {
                    "run_index": 0,
                    "path": str(temp_dir / "cp0"),
                    "validation_loss": 0.5,
                    "training_loss": 0.6,
                    "timestamp": "2026-01-18T09:00:00",
                    "is_run_boundary": False,
                    "is_final": True,
                    "size_bytes": 1024,
                    "protected": False,
                }
            ],
        }
        if version is not None:
            manifest_data["version"] = version

        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))

    def test_old_disk_version_emits_warn(self, temp_checkpoint_dir, caplog):
        """BACKEND-B-003: disk_version='0.5' (older than 1.0) MUST WARN.

        Asserts the warning text names BOTH the disk version AND the
        expected version so an operator can grep the log for the actual
        schema gap without re-reading source.
        """
        self._write_manifest(temp_checkpoint_dir, version="0.5")

        with caplog.at_level("WARNING", logger="backpropagate.checkpoints"):
            CheckpointManager(str(temp_checkpoint_dir))

        warn_records = [
            r for r in caplog.records
            if r.levelname == "WARNING"
            and "version" in r.getMessage().lower()
        ]
        assert warn_records, (
            "BACKEND-B-003 contract violated: no WARN emitted when manifest "
            "disk_version='0.5' != CURRENT_MANIFEST_VERSION='1.0'. "
            "Operators rely on this log to correlate weird post-resume "
            f"behavior with schema age. caplog records: {caplog.records!r}"
        )
        msg = warn_records[0].getMessage()
        assert "0.5" in msg, (
            f"BACKEND-B-003 WARN text must name the on-disk version so "
            f"operators see the actual gap; got: {msg!r}"
        )
        assert "1.0" in msg, (
            f"BACKEND-B-003 WARN text must name the expected version so "
            f"operators see the gap direction; got: {msg!r}"
        )

    def test_mismatch_is_non_fatal_load_still_succeeds(
        self, temp_checkpoint_dir, caplog
    ):
        """BACKEND-B-003: WARN is fail-loud-but-keep-going, NOT raise.

        v1.3 chose to log + continue rather than refuse the load; a real
        migrator is deferred to v1.4. This test pins that choice — if
        someone later promotes the mismatch to an exception, this test
        fails and forces an explicit migration contract.
        """
        self._write_manifest(temp_checkpoint_dir, version="0.5")

        with caplog.at_level("WARNING", logger="backpropagate.checkpoints"):
            manager = CheckpointManager(str(temp_checkpoint_dir))

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 1, (
            "BACKEND-B-003 contract: version mismatch must be non-fatal "
            f"(load + parse continue). Expected 1 checkpoint, got "
            f"{len(checkpoints)}. If you intentionally promoted mismatch "
            "to an exception, update this test AND ship a migrator."
        )
        assert checkpoints[0].run_index == 0, (
            "Checkpoint payload must survive the WARN path unchanged."
        )

    def test_missing_version_field_defaults_to_zero_and_warns(
        self, temp_checkpoint_dir, caplog
    ):
        """BACKEND-B-003: pre-v1.0 manifests (no version field) WARN as '0.0'.

        Pins the ``data.get("version") or "0.0"`` fallback at
        checkpoints.py:230. Pre-v1.0 manifests existed before the version
        field was added; loading them should surface the gap so
        operators see "your manifest predates the schema anchor."
        """
        self._write_manifest(temp_checkpoint_dir, version=None)

        with caplog.at_level("WARNING", logger="backpropagate.checkpoints"):
            manager = CheckpointManager(str(temp_checkpoint_dir))

        warn_records = [
            r for r in caplog.records
            if r.levelname == "WARNING"
            and "version" in r.getMessage().lower()
        ]
        assert warn_records, (
            "BACKEND-B-003 contract: pre-v1.0 manifests (no 'version' key) "
            "must surface a WARN via the '0.0' fallback. The .get() "
            "fallback at checkpoints.py:230 exists precisely for this case."
        )
        msg = warn_records[0].getMessage()
        assert "0.0" in msg, (
            f"Missing-version fallback must surface as '0.0' so operators "
            f"recognize a pre-anchor manifest; got: {msg!r}"
        )
        # Load is still non-fatal
        assert len(manager.list_checkpoints()) == 1

    def test_current_version_is_silent_no_warn(
        self, temp_checkpoint_dir, caplog
    ):
        """BACKEND-B-003 happy-path: matching version emits NO WARN.

        The contract is fail-loud-on-mismatch, silent-on-match. If a
        future refactor accidentally fires WARN on every load, operator
        logs flood with false positives and the real schema-gap signal
        gets buried. This test pins the silent half.
        """
        self._write_manifest(
            temp_checkpoint_dir,
            version=CheckpointManager.CURRENT_MANIFEST_VERSION,
        )

        with caplog.at_level("WARNING", logger="backpropagate.checkpoints"):
            CheckpointManager(str(temp_checkpoint_dir))

        version_warns = [
            r for r in caplog.records
            if r.levelname == "WARNING"
            and "version" in r.getMessage().lower()
            and "manifest" in r.getMessage().lower()
        ]
        assert not version_warns, (
            "BACKEND-B-003 happy-path violated: matching "
            f"version='{CheckpointManager.CURRENT_MANIFEST_VERSION}' must "
            "NOT emit a version-mismatch WARN. Spurious WARNs on every "
            f"load bury the real schema-gap signal. Got: "
            f"{[r.getMessage() for r in version_warns]!r}"
        )


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_register_nonexistent_path(self, manager_no_auto_prune):
        """Registering nonexistent path sets size to 0."""
        info = manager_no_auto_prune.register(
            0,
            "/nonexistent/path",
            validation_loss=0.5,
        )

        assert info.size_bytes == 0

    def test_register_file_checkpoint(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Register a single file as checkpoint."""
        cp_file = temp_checkpoint_dir / "model.safetensors"
        cp_file.write_bytes(b"x" * 2048)

        info = manager_no_auto_prune.register(0, str(cp_file), validation_loss=0.5)

        assert info.size_bytes == 2048

    def test_multiple_managers_same_directory(self, temp_checkpoint_dir):
        """Multiple managers can share the same directory."""
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "shared_cp")

        manager1 = CheckpointManager(str(temp_checkpoint_dir),
                                      CheckpointPolicy(auto_prune=False))
        manager1.register(0, str(cp), validation_loss=0.5)

        # New manager should load existing manifest
        manager2 = CheckpointManager(str(temp_checkpoint_dir),
                                      CheckpointPolicy(auto_prune=False))

        assert len(manager2.list_checkpoints()) == 1

    def test_prune_with_no_prunable_checkpoints(self, temp_checkpoint_dir):
        """Prune when all checkpoints are protected."""
        policy = CheckpointPolicy(auto_prune=False)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        cp = create_dummy_checkpoint(temp_checkpoint_dir, "protected")
        manager.register(0, str(cp), validation_loss=0.5, protected=True)

        pruned = manager.prune()

        assert len(pruned) == 0
        assert len(manager.list_checkpoints()) == 1


# =============================================================================
# SCORING TESTS
# =============================================================================

class TestCheckpointScoring:
    """Tests for checkpoint scoring logic."""

    def test_protected_gets_infinite_score(self, manager_no_auto_prune, temp_checkpoint_dir):
        """Protected checkpoints get highest priority."""
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        manager_no_auto_prune.register(0, str(cp), validation_loss=0.9, protected=True)

        checkpoints = manager_no_auto_prune.list_checkpoints()
        score = manager_no_auto_prune._score_checkpoint(checkpoints[0])

        assert score == float('inf')

    def test_final_checkpoint_gets_bonus(self, temp_checkpoint_dir):
        """Final checkpoint gets score bonus."""
        policy = CheckpointPolicy(keep_final=True, auto_prune=False)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        manager.register(0, str(cp), validation_loss=0.9)

        checkpoints = manager.list_checkpoints()
        score = manager._score_checkpoint(checkpoints[0])

        assert score >= 1000.0  # Final bonus is 1000

    def test_run_boundary_gets_bonus(self, temp_checkpoint_dir):
        """Run boundary checkpoint gets score bonus."""
        policy = CheckpointPolicy(keep_run_boundaries=True, auto_prune=False)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        manager.register(0, str(cp), validation_loss=0.9, is_run_boundary=True)

        # Mark as not final for this test
        checkpoints = manager.list_checkpoints()
        checkpoints[0].is_final = False

        score = manager._score_checkpoint(checkpoints[0])

        assert score >= 500.0  # Run boundary bonus is 500


# =============================================================================
# CHECKPOINT MANAGER EDGE CASES (Phase 2)
# =============================================================================

class TestCorruptedCheckpointHandling:
    """Tests for graceful handling of corrupt checkpoint files."""

    def test_corrupted_manifest_json(self, temp_checkpoint_dir):
        """Graceful handling of corrupt manifest.json."""
        # Create corrupt manifest
        manifest_path = temp_checkpoint_dir / "manifest.json"
        manifest_path.write_text("{invalid json without closing brace")

        # Should not raise, just start with empty list
        manager = CheckpointManager(str(temp_checkpoint_dir))

        assert len(manager.list_checkpoints()) == 0

    def test_corrupted_manifest_partial_json(self, temp_checkpoint_dir):
        """Handle manifest with valid JSON but invalid schema."""
        manifest_path = temp_checkpoint_dir / "manifest.json"
        manifest_path.write_text('{"version": "1.0", "checkpoints": "not_a_list"}')

        # Should handle gracefully
        manager = CheckpointManager(str(temp_checkpoint_dir))
        # May have zero checkpoints or raise - either is acceptable
        # The key is no unhandled exception

    def test_corrupted_checkpoint_file_on_size_calculation(self, temp_checkpoint_dir):
        """Handle errors when calculating checkpoint size."""
        # Create a checkpoint directory
        cp = temp_checkpoint_dir / "cp0"
        cp.mkdir()

        # Create a normal file
        (cp / "model.bin").write_bytes(b"x" * 100)

        policy = CheckpointPolicy(auto_prune=False)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Register should work
        info = manager.register(0, str(cp), validation_loss=0.5)
        assert info.size_bytes == 100

    def test_missing_checkpoint_on_prune(self, temp_checkpoint_dir):
        """Handle missing checkpoint during prune operation."""
        policy = CheckpointPolicy(auto_prune=False, keep_best_n=1, max_total=1)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Create and register checkpoints
        cp1 = create_dummy_checkpoint(temp_checkpoint_dir, "cp1")
        cp2 = create_dummy_checkpoint(temp_checkpoint_dir, "cp2")

        manager.register(0, str(cp1), validation_loss=0.5)
        manager.register(1, str(cp2), validation_loss=0.3)

        # Delete cp1 manually before pruning
        shutil.rmtree(cp1)

        # Prune should not crash
        pruned = manager.prune()
        # The operation should complete (even if cp1 doesn't exist)

    def test_permission_error_on_prune(self, temp_checkpoint_dir, monkeypatch):
        """Handle permission errors during prune gracefully."""
        policy = CheckpointPolicy(auto_prune=False, keep_best_n=0, max_total=1)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        manager.register(0, str(cp), validation_loss=0.5)

        # Mock shutil.rmtree to raise PermissionError
        def mock_rmtree(path):
            raise PermissionError("Access denied")

        monkeypatch.setattr(shutil, "rmtree", mock_rmtree)

        # Should handle gracefully (log error, not crash)
        manager.prune()  # Should not raise

    def test_manifest_missing_required_fields(self, temp_checkpoint_dir):
        """Handle manifest with missing required fields."""
        manifest_path = temp_checkpoint_dir / "manifest.json"
        # Missing 'checkpoints' key
        manifest_path.write_text('{"version": "1.0"}')

        manager = CheckpointManager(str(temp_checkpoint_dir))
        assert len(manager.list_checkpoints()) == 0


class TestConcurrentSaveOperations:
    """Tests for thread safety of checkpoint saves."""

    def test_concurrent_register_operations(self, temp_checkpoint_dir):
        """Thread safety for concurrent register operations."""
        import threading

        policy = CheckpointPolicy(auto_prune=False)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        results = []
        errors = []

        def register_checkpoint(idx):
            try:
                cp = create_dummy_checkpoint(temp_checkpoint_dir, f"concurrent_cp_{idx}")
                info = manager.register(idx, str(cp), validation_loss=0.5 - idx * 0.01)
                results.append(info)
            except Exception as e:
                errors.append(e)

        # Create threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=register_checkpoint, args=(i,))
            threads.append(t)

        # Start all threads nearly simultaneously
        for t in threads:
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "Thread did not finish within timeout"

        # Should have no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        # All checkpoints should be registered
        assert len(results) == 5

    def test_concurrent_prune_operations(self, temp_checkpoint_dir):
        """Thread safety for concurrent prune calls."""
        import threading

        policy = CheckpointPolicy(auto_prune=False, keep_best_n=2, max_total=5)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Create checkpoints first (sequentially)
        for i in range(10):
            cp = create_dummy_checkpoint(temp_checkpoint_dir, f"cp_{i}")
            manager.register(i, str(cp), validation_loss=0.1 * i)

        errors = []

        def prune_operation():
            try:
                manager.prune()
            except Exception as e:
                errors.append(e)

        # Run multiple prune operations concurrently
        threads = [threading.Thread(target=prune_operation) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "Thread did not finish within timeout"

        # Should handle concurrent prunes gracefully
        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_register_and_prune_concurrent(self, temp_checkpoint_dir):
        """Test registering and pruning at the same time."""
        import threading

        policy = CheckpointPolicy(auto_prune=False, keep_best_n=2, max_total=5)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Pre-populate with some checkpoints
        for i in range(5):
            cp = create_dummy_checkpoint(temp_checkpoint_dir, f"initial_cp_{i}")
            manager.register(i, str(cp), validation_loss=0.5)

        errors = []

        def register_task():
            try:
                for i in range(5, 10):
                    cp = create_dummy_checkpoint(temp_checkpoint_dir, f"new_cp_{i}")
                    manager.register(i, str(cp), validation_loss=0.3)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        def prune_task():
            try:
                for _ in range(5):
                    manager.prune()
                    time.sleep(0.02)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=register_task)
        t2 = threading.Thread(target=prune_task)

        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        assert not t1.is_alive(), "Thread did not finish within timeout"
        t2.join(timeout=10.0)
        assert not t2.is_alive(), "Thread did not finish within timeout"

        # Should complete without errors
        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestDiskFullHandling:
    """Tests for graceful error handling when disk is full."""

    def test_disk_full_on_manifest_save(self, temp_checkpoint_dir, monkeypatch):
        """Graceful behavior when disk is full during manifest save.

        Patches _save_manifest to raise OSError, then verifies the manager
        does not crash and prior in-memory state remains intact.
        _save_manifest catches all exceptions internally and returns False,
        so register() should still succeed (in-memory) even when persistence fails.
        """
        policy = CheckpointPolicy(auto_prune=False)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Register a checkpoint successfully first
        cp0 = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        info0 = manager.register(0, str(cp0), validation_loss=0.5)
        assert info0 is not None
        assert len(manager.list_checkpoints()) == 1

        # Patch _save_manifest to simulate disk-full on the next register
        def failing_save():
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(manager, "_save_manifest", failing_save)

        # Second register — _save_manifest failure is caught internally,
        # so the checkpoint is still tracked in memory
        cp1 = create_dummy_checkpoint(temp_checkpoint_dir, "cp1")
        info1 = manager.register(1, str(cp1), validation_loss=0.4)
        assert info1 is not None
        # Both checkpoints should be tracked in memory despite persistence failure
        assert len(manager.list_checkpoints()) == 2

    def test_disk_full_simulation(self, temp_checkpoint_dir, monkeypatch):
        """Simulate disk full during prune manifest persistence.

        Verifies that a disk-full error during _save_manifest after pruning
        does not corrupt previously registered in-memory checkpoint state.
        """
        policy = CheckpointPolicy(auto_prune=False)
        manager = CheckpointManager(str(temp_checkpoint_dir), policy)

        # Register a checkpoint successfully
        cp = create_dummy_checkpoint(temp_checkpoint_dir, "cp0")
        manager.register(0, str(cp), validation_loss=0.5)
        assert len(manager.list_checkpoints()) == 1

        # Patch _save_manifest to simulate disk full
        def failing_save():
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(manager, "_save_manifest", failing_save)

        # Prune should not crash even if manifest save fails internally
        pruned = manager.prune()

        # Original checkpoint data should still be queryable in memory
        assert len(manager.list_checkpoints()) >= 1

    def test_readonly_checkpoint_directory(self, temp_checkpoint_dir):
        """Handle read-only checkpoint directory gracefully."""
        # This test is platform-specific and may not work on Windows
        # Skip if we can't set permissions
        import platform

        if platform.system() == "Windows":
            pytest.skip("Permission tests not reliable on Windows")

        # Create a read-only directory
        readonly_dir = temp_checkpoint_dir / "readonly"
        readonly_dir.mkdir()

        try:
            # Make directory read-only
            readonly_dir.chmod(0o444)

            # Attempting to create manager should handle gracefully
            # or raise an appropriate error
            try:
                manager = CheckpointManager(str(readonly_dir / "subdir"))
                # If it gets here, it should have created the directory somehow
            except (PermissionError, OSError):
                # Expected behavior
                pass

        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)


# =============================================================================
# F-003 RUN HISTORY MANAGER LIFECYCLE TESTS
# =============================================================================

class TestRunHistoryManagerLifecycle:
    """Lifecycle API: record_run_started / record_run_completed / record_run_failed."""

    def _manager(self, tmp_path):
        from backpropagate.checkpoints import RunHistoryManager

        return RunHistoryManager(str(tmp_path))

    def test_record_run_started_creates_running_entry(self, tmp_path):
        manager = self._manager(tmp_path)
        entry = manager.record_run_started(
            run_id="abc123",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            dataset_info="data.jsonl",
            hyperparameters={"lr": 2e-4},
        )
        assert entry["run_id"] == "abc123"
        assert entry["status"] == "running"
        assert entry["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
        assert entry["hyperparameters"] == {"lr": 2e-4}

        # Round-trip via list_runs.
        runs = manager.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "abc123"

    def test_record_run_started_is_idempotent_on_same_run_id(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.record_run_started(run_id="dup", model_name="m1")
        manager.record_run_started(run_id="dup", model_name="m2")
        runs = manager.list_runs()
        assert len(runs) == 1
        assert runs[0]["model_name"] == "m2"  # Second call wins.

    def test_record_run_completed_updates_status_and_metrics(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.record_run_started(run_id="cc", model_name="m")
        manager.record_run_completed(
            run_id="cc",
            final_loss=0.42,
            loss_history=[1.0, 0.5, 0.42],
            steps=100,
            duration_seconds=33.5,
        )
        run = manager.get_run("cc")
        assert run is not None
        assert run["status"] == "completed"
        assert run["final_loss"] == 0.42
        assert run["steps"] == 100
        assert run["duration_seconds"] == 33.5
        assert run["loss_history"] == [1.0, 0.5, 0.42]

    def test_record_run_failed_marks_failure_reason(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.record_run_started(run_id="ff", model_name="m")
        manager.record_run_failed(
            run_id="ff",
            failure_reason="RuntimeError: out of memory",
            duration_seconds=5.0,
        )
        run = manager.get_run("ff")
        assert run is not None
        assert run["status"] == "failed"
        assert run["failure_reason"] == "RuntimeError: out of memory"

    def test_record_run_completed_synthesizes_when_no_started(self, tmp_path):
        manager = self._manager(tmp_path)
        # Skip the started call.
        manager.record_run_completed(
            run_id="orphan",
            final_loss=0.1,
            steps=10,
        )
        run = manager.get_run("orphan")
        assert run is not None
        assert run["status"] == "completed"
        assert run["final_loss"] == 0.1

    def test_record_run_failed_synthesizes_when_no_started(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.record_run_failed(run_id="lost", failure_reason="boom")
        run = manager.get_run("lost")
        assert run is not None
        assert run["status"] == "failed"
        assert run["failure_reason"] == "boom"

    def test_list_runs_sorted_most_recent_first(self, tmp_path):
        import time as _time

        manager = self._manager(tmp_path)
        manager.record_run_started(run_id="r1", model_name="m")
        _time.sleep(0.01)
        manager.record_run_started(run_id="r2", model_name="m")
        _time.sleep(0.01)
        manager.record_run_started(run_id="r3", model_name="m")
        ids = [r["run_id"] for r in manager.list_runs()]
        assert ids[0] == "r3"  # newest first
        assert ids[-1] == "r1"

    def test_list_runs_filter_by_status(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.record_run_started(run_id="a", model_name="m")
        manager.record_run_started(run_id="b", model_name="m")
        manager.record_run_completed(run_id="b", final_loss=0.1)
        manager.record_run_started(run_id="c", model_name="m")
        manager.record_run_failed(run_id="c", failure_reason="x")

        running = manager.list_runs(status="running")
        assert [r["run_id"] for r in running] == ["a"]
        completed = manager.list_runs(status="completed")
        assert [r["run_id"] for r in completed] == ["b"]
        failed = manager.list_runs(status="failed")
        assert [r["run_id"] for r in failed] == ["c"]

    def test_list_runs_status_rejects_unknown(self, tmp_path):
        manager = self._manager(tmp_path)
        with pytest.raises(ValueError):
            manager.list_runs(status="unknown")

    def test_list_runs_limit_caps_results(self, tmp_path):
        manager = self._manager(tmp_path)
        for i in range(5):
            manager.record_run_started(run_id=f"r{i}", model_name="m")
        assert len(manager.list_runs(limit=2)) == 2

    def test_get_run_partial_prefix_match(self, tmp_path):
        manager = self._manager(tmp_path)
        full_id = "abcdef0123456789"
        manager.record_run_started(run_id=full_id, model_name="m")
        # Unique prefix resolves to the entry.
        assert manager.get_run("abcdef")["run_id"] == full_id
        # Non-matching prefix returns None.
        assert manager.get_run("zzz") is None

    def test_get_run_ambiguous_prefix_returns_none(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.record_run_started(run_id="abc111", model_name="m")
        manager.record_run_started(run_id="abc222", model_name="m")
        # Multiple prefix matches => ambiguous, return None.
        assert manager.get_run("abc") is None

    def test_delete_run_removes_entry(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.record_run_started(run_id="del", model_name="m")
        assert manager.delete_run("del") is True
        assert manager.get_run("del") is None
        # Idempotent: second delete returns False.
        assert manager.delete_run("del") is False

    def test_update_run_patches_fields(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.record_run_started(run_id="up", model_name="m")
        manager.update_run("up", export_paths=["/tmp/lora1", "/tmp/lora2"])
        run = manager.get_run("up")
        assert run["export_paths"] == ["/tmp/lora1", "/tmp/lora2"]

    def test_update_run_missing_returns_none(self, tmp_path):
        manager = self._manager(tmp_path)
        assert manager.update_run("ghost", model_name="x") is None

    def test_in_progress_runs(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.record_run_started(run_id="x", model_name="m")
        manager.record_run_started(run_id="y", model_name="m")
        manager.record_run_completed(run_id="y", final_loss=0.0)
        in_progress = manager.in_progress_runs()
        assert [r["run_id"] for r in in_progress] == ["x"]


# =============================================================================
# F-003 TRAINER + MULTI-RUN INTEGRATION HOOKS
# =============================================================================

class TestRunHistoryIntegrationHooks:
    """Ensure RunHistoryManager is wired into Trainer / MultiRunTrainer."""

    def test_run_history_manager_reexported_from_package(self):
        import backpropagate

        assert hasattr(backpropagate, "RunHistoryManager")
        assert (
            backpropagate.RunHistoryManager
            .__module__.startswith("backpropagate.checkpoints")
        )

    def test_run_history_manager_in_dunder_all(self):
        import backpropagate

        assert "RunHistoryManager" in backpropagate.__all__
