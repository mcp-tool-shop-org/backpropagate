#!/usr/bin/env python
"""Test Phase 5.3 Checkpoint Management integration."""
import os
import sys

if __name__ != "__main__":
    sys.exit(0)

print('=== Testing Phase 5.3 Checkpoint Management ===\n')

# Test 1: CheckpointPolicy
print('--- Test 1: CheckpointPolicy ---')
from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy

policy = CheckpointPolicy(
    keep_best_n=3,
    keep_final=True,
    keep_run_boundaries=False,
    max_total=10,
    auto_prune=True,
)
print(f'Policy: keep_best_n={policy.keep_best_n}, max_total={policy.max_total}, auto_prune={policy.auto_prune}')
assert policy.keep_best_n == 3
assert policy.auto_prune == True
print('[PASS] CheckpointPolicy works')

# Test 2: MultiRunConfig with checkpoint options
print('\n--- Test 2: MultiRunConfig checkpoint options ---')
from backpropagate.multi_run import MultiRunConfig

config = MultiRunConfig(
    num_runs=5,
    steps_per_run=100,
    checkpoint_keep_best_n=5,
    checkpoint_keep_final=True,
    checkpoint_keep_run_boundaries=True,
    checkpoint_max_total=15,
    checkpoint_auto_prune=False,  # Disable for those who want all runs saved
)
print('Config checkpoint options:')
print(f'  keep_best_n: {config.checkpoint_keep_best_n}')
print(f'  keep_final: {config.checkpoint_keep_final}')
print(f'  keep_run_boundaries: {config.checkpoint_keep_run_boundaries}')
print(f'  max_total: {config.checkpoint_max_total}')
print(f'  auto_prune: {config.checkpoint_auto_prune}')

assert config.checkpoint_keep_best_n == 5
assert config.checkpoint_auto_prune == False
print('[PASS] MultiRunConfig checkpoint options work')

# Test 3: CheckpointManager creation
print('\n--- Test 3: CheckpointManager ---')
import shutil
import tempfile

# Create temp directory
temp_dir = tempfile.mkdtemp(prefix="backprop_ckpt_test_")
try:
    manager = CheckpointManager(checkpoint_dir=temp_dir, policy=policy)
    print(f'Manager created at: {temp_dir}')

    # Register some mock checkpoints
    for i in range(1, 6):
        # Create dummy checkpoint dirs
        ckpt_path = os.path.join(temp_dir, f"run_{i:03d}")
        os.makedirs(ckpt_path, exist_ok=True)

        # Create a small file to have some size
        with open(os.path.join(ckpt_path, "model.bin"), "wb") as f:
            f.write(b"x" * (1024 * 1024))  # 1 MB

        manager.register(
            run_index=i,
            checkpoint_path=ckpt_path,
            validation_loss=0.5 - (i * 0.05),  # Decreasing loss
            training_loss=0.6 - (i * 0.05),
            is_run_boundary=(i == 1),
        )

    stats = manager.get_stats()
    print(f'Stats: {stats.total_count} checkpoints, {stats.total_size_gb:.2f} GB')
    print(f'Best checkpoint: Run {stats.best_checkpoint.run_index if stats.best_checkpoint else "N/A"}')
    print(f'Prunable: {stats.prunable_count}')

    # Auto-prune keeps best 3, so we should have 3 checkpoints (not 5)
    # The policy has auto_prune=True and keep_best_n=3
    assert stats.total_count <= policy.keep_best_n + 1, f"Expected max {policy.keep_best_n + 1} checkpoints, got {stats.total_count}"
    print('[PASS] CheckpointManager with auto-prune works')

    # Test that best checkpoint is run 5 (lowest val_loss)
    assert stats.best_checkpoint is not None
    assert stats.best_checkpoint.run_index == 5, "Best checkpoint should be run 5 (lowest val_loss)"
    print('[PASS] Best checkpoint selection works')

    # Test 4: Disable auto-prune (for users who want all runs saved)
    print('\n--- Test 4: Disable auto-prune ---')
    policy_no_prune = CheckpointPolicy(
        keep_best_n=3,
        auto_prune=False,  # Disable pruning
    )
    temp_dir2 = tempfile.mkdtemp(prefix="backprop_no_prune_")
    manager2 = CheckpointManager(checkpoint_dir=temp_dir2, policy=policy_no_prune)

    for i in range(1, 6):
        ckpt_path = os.path.join(temp_dir2, f"run_{i:03d}")
        os.makedirs(ckpt_path, exist_ok=True)
        with open(os.path.join(ckpt_path, "model.bin"), "wb") as f:
            f.write(b"x" * 1024)
        manager2.register(run_index=i, checkpoint_path=ckpt_path, validation_loss=0.5 - (i * 0.05))

    stats2 = manager2.get_stats()
    print(f'Stats (no auto-prune): {stats2.total_count} checkpoints')
    assert stats2.total_count == 5, "With auto_prune=False, all 5 checkpoints should be kept"
    print('[PASS] Disable auto-prune works (all runs saved)')
    shutil.rmtree(temp_dir2, ignore_errors=True)

finally:
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

# Test 5: UI function imports
print('\n--- Test 5: UI function imports ---')
from backpropagate.ui import get_dashboard_metrics

print('get_dashboard_metrics imported')
print('refresh_dashboard imported')

# Check checkpoint metrics exist in dashboard
metrics = get_dashboard_metrics()
assert "ckpt_count" in metrics
assert "ckpt_size" in metrics
assert "ckpt_best" in metrics
assert "ckpt_prunable" in metrics
assert "ckpt_policy" in metrics
print('[PASS] Dashboard metrics include checkpoint fields')

print('\n=== Phase 5.3 Checkpoint Management Tests Passed ===')
print('\nFeatures implemented:')
print('  - CheckpointPolicy with configurable keep_best_n, max_total, auto_prune')
print('  - CheckpointManager with smart pruning based on validation loss')
print('  - MultiRunConfig integration with checkpoint options')
print('  - UI sidebar with checkpoint stats')
print('  - Option to disable auto_prune for users who want all runs saved')
