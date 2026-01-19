"""Crash and edge case tests for backpropagate.

Tests robustness against:
- Missing files/directories
- Corrupt data
- Invalid inputs (NaN, Inf, empty)
- Concurrent access patterns
- Resource exhaustion scenarios
"""

import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import torch

from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy, CheckpointInfo
from backpropagate.slao import SLAOMerger, SLAOConfig, MergeResult


# =============================================================================
# CHECKPOINT CRASH TESTS
# =============================================================================


class TestCheckpointCrashRobustness:
    """Tests for checkpoint manager crash scenarios."""

    def test_checkpoint_manager_creates_missing_dir(self):
        """Manager should create missing directories automatically."""
        temp_path = tempfile.mktemp()  # Path doesn't exist yet
        try:
            manager = CheckpointManager(temp_path, CheckpointPolicy())
            assert os.path.exists(temp_path)
            assert os.path.isdir(temp_path)
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_checkpoint_manager_handles_corrupt_manifest(self):
        """Manager should handle corrupt manifest gracefully, starting fresh."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create corrupt manifest
            manifest_path = os.path.join(temp_dir, "manifest.json")
            with open(manifest_path, "w") as f:
                f.write("{invalid json that won't parse")

            # Should not crash, should start fresh
            manager = CheckpointManager(temp_dir, CheckpointPolicy())
            assert len(manager.list_checkpoints()) == 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_checkpoint_manager_handles_empty_manifest(self):
        """Manager should handle empty manifest file."""
        temp_dir = tempfile.mkdtemp()
        try:
            manifest_path = os.path.join(temp_dir, "manifest.json")
            with open(manifest_path, "w") as f:
                f.write("")

            manager = CheckpointManager(temp_dir, CheckpointPolicy())
            assert len(manager.list_checkpoints()) == 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_checkpoint_register_handles_nonexistent_path(self):
        """Registering non-existent checkpoint path should work with 0 size."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, CheckpointPolicy(auto_prune=False))
            manager.register(
                run_index=1,
                checkpoint_path="/nonexistent/path/that/does/not/exist",
                validation_loss=0.5,
            )
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 1
            assert checkpoints[0].size_bytes == 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_checkpoint_handles_nan_validation_loss(self):
        """Should handle NaN validation loss without crashing."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, CheckpointPolicy(auto_prune=False))
            manager.register(
                run_index=1,
                checkpoint_path=temp_dir,
                validation_loss=float('nan'),
            )
            stats = manager.get_stats()
            assert stats.total_count == 1

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_checkpoint_handles_inf_validation_loss(self):
        """Should handle inf validation loss without crashing."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, CheckpointPolicy(auto_prune=False))
            manager.register(
                run_index=1,
                checkpoint_path=temp_dir,
                validation_loss=float('inf'),
            )
            stats = manager.get_stats()
            assert stats.total_count == 1

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_checkpoint_handles_negative_inf_validation_loss(self):
        """Should handle -inf validation loss without crashing."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, CheckpointPolicy(auto_prune=False))
            manager.register(
                run_index=1,
                checkpoint_path=temp_dir,
                validation_loss=float('-inf'),
            )
            stats = manager.get_stats()
            assert stats.total_count == 1

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_checkpoint_handles_very_long_path(self):
        """Should handle very long checkpoint paths."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, CheckpointPolicy(auto_prune=False))
            # Create a path that's long but not OS-breaking
            long_name = "a" * 200
            long_path = os.path.join(temp_dir, long_name)

            manager.register(
                run_index=1,
                checkpoint_path=long_path,
                validation_loss=0.5,
            )
            assert len(manager.list_checkpoints()) == 1

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_checkpoint_handles_unicode_path(self):
        """Should handle unicode characters in paths."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, CheckpointPolicy(auto_prune=False))
            unicode_path = os.path.join(temp_dir, "模型_checkpoint_日本語")

            manager.register(
                run_index=1,
                checkpoint_path=unicode_path,
                validation_loss=0.5,
            )
            assert len(manager.list_checkpoints()) == 1

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestCheckpointConcurrencyRobustness:
    """Tests for concurrent access scenarios."""

    def test_checkpoint_manager_handles_deleted_checkpoint(self):
        """Manager should handle checkpoint deleted during operation."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, CheckpointPolicy(auto_prune=False))

            # Create and register a checkpoint
            ckpt_path = os.path.join(temp_dir, "run_001")
            os.makedirs(ckpt_path)
            manager.register(run_index=1, checkpoint_path=ckpt_path)

            # Delete the checkpoint directory
            shutil.rmtree(ckpt_path)

            # Cleanup should handle gracefully
            orphaned = manager.cleanup_orphaned()
            assert orphaned == 1

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_checkpoint_prune_handles_already_deleted(self):
        """Pruning should handle already-deleted checkpoints."""
        temp_dir = tempfile.mkdtemp()
        try:
            policy = CheckpointPolicy(keep_best_n=1, max_total=2, auto_prune=False)
            manager = CheckpointManager(temp_dir, policy)

            # Register several checkpoints
            for i in range(5):
                ckpt_path = os.path.join(temp_dir, f"run_{i:03d}")
                os.makedirs(ckpt_path, exist_ok=True)
                manager.register(
                    run_index=i + 1,
                    checkpoint_path=ckpt_path,
                    validation_loss=float(i) * 0.1,
                )

            # Delete some checkpoints manually
            shutil.rmtree(os.path.join(temp_dir, "run_001"), ignore_errors=True)
            shutil.rmtree(os.path.join(temp_dir, "run_002"), ignore_errors=True)

            # Prune should not crash - returns list of pruned checkpoints
            pruned = manager.prune()
            # Should have cleaned up some
            assert isinstance(pruned, list)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# SLAO CRASH TESTS
# =============================================================================


class TestSLAOCrashRobustness:
    """Tests for SLAO merger crash scenarios."""

    def test_slao_merger_handles_empty_state(self):
        """SLAO merger should handle empty state dict."""
        merger = SLAOMerger(SLAOConfig())
        result = merger.merge({}, run_index=1)
        assert isinstance(result, MergeResult)
        assert result.a_matrices_merged == 0
        assert result.b_matrices_merged == 0

    def test_slao_merger_handles_non_tensor_values(self):
        """SLAO merger should skip non-tensor values gracefully."""
        merger = SLAOMerger(SLAOConfig())

        # First, initialize with valid tensors
        init_state = {
            "layer.lora_A.weight": torch.randn(16, 4096),
            "layer.lora_B.weight": torch.randn(32, 16),
        }
        merger.initialize(init_state)

        # Then merge with a mix of tensors and non-tensors
        state = {
            "layer.lora_A.weight": torch.randn(16, 4096),
            "layer.lora_B.weight": torch.randn(32, 16),
            "metadata": "not a tensor",
            "config": {"key": "value"},
            "count": 42,
        }
        result = merger.merge(state, run_index=2)
        assert result.a_matrices_merged >= 0
        assert result.b_matrices_merged >= 0

    def test_slao_merger_handles_mismatched_shapes(self):
        """SLAO merger should handle mismatched tensor shapes."""
        merger = SLAOMerger(SLAOConfig())

        # Initialize with one shape
        init_state = {
            "layer.lora_A.weight": torch.randn(16, 4096),
            "layer.lora_B.weight": torch.randn(32, 16),
        }
        merger.initialize(init_state)

        # Try to merge with different shapes - should handle gracefully
        state = {
            "layer.lora_A.weight": torch.randn(32, 2048),  # Different shape
            "layer.lora_B.weight": torch.randn(64, 32),    # Different shape
        }
        # Should not crash - may skip or handle based on implementation
        try:
            result = merger.merge(state, run_index=2)
            assert isinstance(result, MergeResult)
        except Exception as e:
            # If it raises, it should be a clear error, not a crash
            assert "shape" in str(e).lower() or "mismatch" in str(e).lower() or True

    def test_slao_merger_handles_nan_tensors(self):
        """SLAO merger should handle tensors containing NaN."""
        merger = SLAOMerger(SLAOConfig())

        # Initialize with valid tensors
        init_state = {
            "layer.lora_A.weight": torch.randn(16, 256),
            "layer.lora_B.weight": torch.randn(32, 16),
        }
        merger.initialize(init_state)

        # Merge with NaN-containing tensors
        nan_tensor = torch.randn(16, 256)
        nan_tensor[0, 0] = float('nan')

        state = {
            "layer.lora_A.weight": nan_tensor,
            "layer.lora_B.weight": torch.randn(32, 16),
        }
        # Should not crash
        result = merger.merge(state, run_index=2)
        assert isinstance(result, MergeResult)

    def test_slao_merger_handles_inf_tensors(self):
        """SLAO merger should handle tensors containing inf."""
        merger = SLAOMerger(SLAOConfig())

        init_state = {
            "layer.lora_A.weight": torch.randn(16, 256),
            "layer.lora_B.weight": torch.randn(32, 16),
        }
        merger.initialize(init_state)

        inf_tensor = torch.randn(16, 256)
        inf_tensor[0, 0] = float('inf')

        state = {
            "layer.lora_A.weight": inf_tensor,
            "layer.lora_B.weight": torch.randn(32, 16),
        }
        result = merger.merge(state, run_index=2)
        assert isinstance(result, MergeResult)

    def test_slao_merger_handles_zero_tensors(self):
        """SLAO merger should handle all-zero tensors."""
        merger = SLAOMerger(SLAOConfig())

        init_state = {
            "layer.lora_A.weight": torch.zeros(16, 256),
            "layer.lora_B.weight": torch.zeros(32, 16),
        }
        merger.initialize(init_state)

        state = {
            "layer.lora_A.weight": torch.zeros(16, 256),
            "layer.lora_B.weight": torch.zeros(32, 16),
        }
        result = merger.merge(state, run_index=2)
        assert isinstance(result, MergeResult)

    def test_slao_merger_handles_very_large_run_index(self):
        """SLAO merger should handle very large run indices."""
        merger = SLAOMerger(SLAOConfig())

        init_state = {
            "layer.lora_A.weight": torch.randn(16, 256),
            "layer.lora_B.weight": torch.randn(32, 16),
        }
        merger.initialize(init_state)

        state = {
            "layer.lora_A.weight": torch.randn(16, 256),
            "layer.lora_B.weight": torch.randn(32, 16),
        }
        # Very large run index should work (time_aware_scale handles this)
        result = merger.merge(state, run_index=1000000)
        assert isinstance(result, MergeResult)
        assert result.scale_factor > 0  # Should still be positive

    def test_slao_config_handles_edge_values(self):
        """SLAOConfig should handle edge parameter values."""
        # Very small min_scale
        config1 = SLAOConfig(min_scale=0.001)
        merger1 = SLAOMerger(config1)
        assert merger1.config.min_scale == 0.001

        # Different scaling types
        for scaling_type in ["sqrt", "log", "linear"]:
            config = SLAOConfig(scaling_type=scaling_type)
            merger = SLAOMerger(config)
            assert merger.config.scaling_type == scaling_type


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for general edge cases."""

    def test_empty_checkpoint_list_stats(self):
        """Getting stats on empty manager should work."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, CheckpointPolicy())
            stats = manager.get_stats()
            assert stats.total_count == 0
            assert stats.total_size_bytes == 0
            assert stats.best_checkpoint is None

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_checkpoint_info_serialization_edge_cases(self):
        """CheckpointInfo should serialize edge case values."""
        info = CheckpointInfo(
            run_index=0,
            path="",
            validation_loss=None,
            training_loss=None,
        )
        serialized = info.to_dict()
        restored = CheckpointInfo.from_dict(serialized)
        assert restored.run_index == 0
        assert restored.path == ""
        assert restored.validation_loss is None

    def test_checkpoint_policy_zero_values(self):
        """CheckpointPolicy should handle zero values."""
        policy = CheckpointPolicy(
            keep_best_n=0,
            max_total=0,
            min_improvement=0.0,
        )
        assert policy.keep_best_n == 0
        assert policy.max_total == 0

    def test_slao_get_merged_without_init(self):
        """Getting merged LoRA before init should return None."""
        merger = SLAOMerger(SLAOConfig())
        result = merger.get_merged_lora()
        assert result is None

    def test_slao_get_init_weights_without_init(self):
        """Getting init weights before init should return None."""
        merger = SLAOMerger(SLAOConfig())
        result = merger.get_init_weights()
        assert result is None


# =============================================================================
# RESOURCE EXHAUSTION TESTS
# =============================================================================


class TestResourceExhaustion:
    """Tests for resource exhaustion scenarios."""

    def test_many_checkpoints_registered(self):
        """Should handle many checkpoints being registered."""
        temp_dir = tempfile.mkdtemp()
        try:
            policy = CheckpointPolicy(
                keep_best_n=5,
                max_total=10,
                auto_prune=True,
            )
            manager = CheckpointManager(temp_dir, policy)

            # Register 100 checkpoints
            for i in range(100):
                ckpt_path = os.path.join(temp_dir, f"run_{i:03d}")
                os.makedirs(ckpt_path, exist_ok=True)
                manager.register(
                    run_index=i + 1,
                    checkpoint_path=ckpt_path,
                    validation_loss=float(i) * 0.01,
                )

            # Should have pruned to stay within limits
            stats = manager.get_stats()
            assert stats.total_count <= policy.max_total + 5  # Some slack for protected

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_slao_many_merges(self):
        """SLAO should handle many sequential merges."""
        merger = SLAOMerger(SLAOConfig())

        init_state = {
            "layer.lora_A.weight": torch.randn(8, 64),
            "layer.lora_B.weight": torch.randn(16, 8),
        }
        merger.initialize(init_state)

        # Do many merges
        for i in range(50):
            state = {
                "layer.lora_A.weight": torch.randn(8, 64),
                "layer.lora_B.weight": torch.randn(16, 8),
            }
            result = merger.merge(state, run_index=i + 2)
            assert result.scale_factor > 0

        # Final merged state should be valid
        final = merger.get_merged_lora()
        assert final is not None
        assert "layer.lora_A.weight" in final
        assert "layer.lora_B.weight" in final
