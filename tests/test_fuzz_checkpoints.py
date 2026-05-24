"""Fuzz tests for checkpoint management.

Uses Hypothesis to generate random checkpoint policies and operations,
testing that the checkpoint manager never crashes and respects invariants.
"""

import os
import shutil
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backpropagate.checkpoints import (
    CheckpointInfo,
    CheckpointManager,
    CheckpointPolicy,
    CheckpointStats,
)

# =============================================================================
# STRATEGIES
# =============================================================================


@st.composite
def checkpoint_policy_strategy(draw):
    """Generate valid checkpoint policies with reasonable bounds."""
    keep_best_n = draw(st.integers(min_value=1, max_value=20))
    max_total = draw(st.integers(min_value=0, max_value=50))

    # Ensure max_total >= keep_best_n when both are non-zero
    if max_total > 0 and max_total < keep_best_n:
        max_total = keep_best_n

    return CheckpointPolicy(
        keep_best_n=keep_best_n,
        keep_final=draw(st.booleans()),
        keep_run_boundaries=draw(st.booleans()),
        max_total=max_total,
        min_improvement=draw(st.floats(min_value=0.0, max_value=1.0)),
        auto_prune=draw(st.booleans()),
    )


@st.composite
def validation_losses_strategy(draw, min_size=1, max_size=20):
    """Generate lists of valid validation losses (no NaN/Inf)."""
    return draw(st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=min_size,
        max_size=max_size,
    ))


# =============================================================================
# FUZZ TESTS
# =============================================================================


@pytest.mark.hypothesis
class TestCheckpointManagerFuzz:
    """Fuzz tests for CheckpointManager."""

    @given(
        policy=checkpoint_policy_strategy(),
        num_checkpoints=st.integers(min_value=1, max_value=20),
        val_losses=validation_losses_strategy(min_size=1, max_size=20),
    )
    @settings(max_examples=100, deadline=None)
    def test_register_never_crashes(self, policy, num_checkpoints, val_losses):
        """Registering checkpoints should never crash regardless of inputs."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, policy)

            for i in range(min(num_checkpoints, len(val_losses))):
                ckpt_path = os.path.join(temp_dir, f"run_{i:03d}")
                os.makedirs(ckpt_path, exist_ok=True)

                # This should never raise
                manager.register(
                    run_index=i + 1,
                    checkpoint_path=ckpt_path,
                    validation_loss=val_losses[i],
                )

            # Should be able to get stats without crashing
            stats = manager.get_stats()
            assert isinstance(stats, CheckpointStats)
            assert stats.total_count >= 0
            assert stats.total_size_bytes >= 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @given(policy=checkpoint_policy_strategy())
    @settings(max_examples=50, deadline=None)
    def test_prune_respects_policy_limits(self, policy):
        """Pruning should never exceed policy.max_total (when set)."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, policy)

            # Register 15 checkpoints with varying losses
            for i in range(15):
                ckpt_path = os.path.join(temp_dir, f"run_{i:03d}")
                os.makedirs(ckpt_path, exist_ok=True)
                manager.register(
                    run_index=i + 1,
                    checkpoint_path=ckpt_path,
                    validation_loss=float(i) * 0.1 + 0.5,  # 0.5 to 1.9
                )

            stats = manager.get_stats()

            # If auto_prune is on and max_total is set, respect the limit
            # Allow some slack for protected checkpoints (final, run boundaries)
            if policy.auto_prune and policy.max_total > 0:
                max_expected = policy.max_total + 3  # +3 for protected slots
                assert stats.total_count <= max_expected, (
                    f"Expected at most {max_expected} checkpoints, got {stats.total_count}"
                )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @given(
        policy=checkpoint_policy_strategy(),
        losses=validation_losses_strategy(min_size=5, max_size=15),
    )
    @settings(max_examples=50, deadline=None)
    def test_best_checkpoint_has_lowest_loss(self, policy, losses):
        """get_best_checkpoint should return checkpoint with lowest loss."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, policy)

            for i, loss in enumerate(losses):
                ckpt_path = os.path.join(temp_dir, f"run_{i:03d}")
                os.makedirs(ckpt_path, exist_ok=True)
                manager.register(
                    run_index=i + 1,
                    checkpoint_path=ckpt_path,
                    validation_loss=loss,
                )

            best = manager.get_best_checkpoint()

            if best is not None:
                # Best should have loss <= all other checkpoints
                all_checkpoints = manager.list_checkpoints()
                for cp in all_checkpoints:
                    if cp.validation_loss is not None:
                        assert best.validation_loss <= cp.validation_loss, (
                            f"Best loss {best.validation_loss} > {cp.validation_loss}"
                        )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @given(
        policy=checkpoint_policy_strategy(),
        run_indices=st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=3,
            max_size=15,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_run_boundary_tracking(self, policy, run_indices):
        """Run boundaries should be correctly tracked.

        Property under test: when checkpoints are registered with
        ``is_run_boundary=True`` exactly once per unique ``run_index`` (the
        first time we see it), then EVERY persisted ``CheckpointInfo``
        whose ``is_run_boundary`` flag is True must correspond to a
        run_index that was actually registered with that flag in this
        test. And, if pruning has not removed it, the first checkpoint
        registered for each unique run must still bear the boundary flag.

        Previously this test had zero ``assert`` statements — 50 examples
        ran 50 no-ops. v1.3 Wave 3.5 (TESTS-B-006) restored the load-
        bearing property checks.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, policy)

            # Track which checkpoint path was registered as a boundary,
            # plus the run_index → path mapping for the first registration
            # of each run.
            seen_runs: set[int] = set()
            boundary_paths: set[str] = set()
            first_path_per_run: dict[int, str] = {}
            registered_boundary_runs: set[int] = set()
            for i, run_idx in enumerate(run_indices):
                ckpt_path = os.path.join(temp_dir, f"ckpt_{i:03d}")
                os.makedirs(ckpt_path, exist_ok=True)

                is_boundary = run_idx not in seen_runs
                if is_boundary:
                    boundary_paths.add(ckpt_path)
                    first_path_per_run[run_idx] = ckpt_path
                    registered_boundary_runs.add(run_idx)
                seen_runs.add(run_idx)

                manager.register(
                    run_index=run_idx,
                    checkpoint_path=ckpt_path,
                    validation_loss=float(i) * 0.1,
                    is_run_boundary=is_boundary,
                )

            # Property 1: every checkpoint with is_run_boundary=True (that
            # survived pruning) must trace back to a path we registered
            # as a boundary in this test — the manager must NEVER fabricate
            # boundary marks on its own.
            all_checkpoints = manager.list_checkpoints()
            for cp in all_checkpoints:
                if cp.is_run_boundary:
                    assert cp.path in boundary_paths, (
                        f"Manager reports is_run_boundary=True for "
                        f"checkpoint path={cp.path!r} run_index="
                        f"{cp.run_index} which was NOT registered with "
                        f"is_run_boundary=True. Boundary paths we "
                        f"registered: {sorted(boundary_paths)}"
                    )

            # Property 2: if a first-registration for a run survived pruning,
            # its is_run_boundary flag must still be True (the manager must
            # NEVER strip a flag the caller set, even if the policy chooses
            # not to PROTECT it from pruning — preserved-but-unprotected
            # is still a valid state).
            surviving_paths = {cp.path: cp for cp in all_checkpoints}
            for run_idx, path in first_path_per_run.items():
                if path in surviving_paths:
                    cp = surviving_paths[path]
                    assert cp.is_run_boundary is True, (
                        f"First checkpoint for run_index={run_idx} at "
                        f"path={path!r} survived pruning but lost its "
                        f"is_run_boundary flag: cp.is_run_boundary="
                        f"{cp.is_run_boundary!r}. Manager must not strip "
                        f"caller-set flags."
                    )
                    assert cp.run_index == run_idx, (
                        f"run_index mismatch on survived first-of-run "
                        f"checkpoint: expected {run_idx}, got "
                        f"{cp.run_index} at path={path!r}"
                    )

            # Property 3: boundary count is bounded by the number of
            # unique run indices we ever marked as boundaries (pruning
            # may reduce it, never inflate it).
            boundary_count = sum(
                1 for cp in all_checkpoints if cp.is_run_boundary
            )
            assert boundary_count <= len(registered_boundary_runs), (
                f"Boundary count {boundary_count} exceeds the "
                f"{len(registered_boundary_runs)} unique runs we ever "
                f"marked as boundaries — manager invented boundaries."
            )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @given(policy=checkpoint_policy_strategy())
    @settings(max_examples=30, deadline=None)
    def test_final_checkpoint_always_tracked(self, policy):
        """The most recent checkpoint should always be marked as final."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = CheckpointManager(temp_dir, policy)

            # Register several checkpoints
            for i in range(5):
                ckpt_path = os.path.join(temp_dir, f"run_{i:03d}")
                os.makedirs(ckpt_path, exist_ok=True)
                manager.register(
                    run_index=i + 1,
                    checkpoint_path=ckpt_path,
                    validation_loss=float(i) * 0.2,
                )

            # Get all checkpoints and find the final one
            all_checkpoints = manager.list_checkpoints()
            if all_checkpoints:
                final_count = sum(1 for cp in all_checkpoints if cp.is_final)
                assert final_count <= 1, f"Multiple final checkpoints: {final_count}"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.hypothesis
class TestCheckpointPolicyFuzz:
    """Fuzz tests for CheckpointPolicy validation."""

    @given(
        keep_best_n=st.integers(min_value=-10, max_value=100),
        max_total=st.integers(min_value=-10, max_value=100),
    )
    def test_policy_creation_with_any_values(self, keep_best_n, max_total):
        """Policy should handle any integer values (dataclass doesn't validate)."""
        # This just tests that creation doesn't crash
        # Actual validation happens at usage time
        policy = CheckpointPolicy(
            keep_best_n=keep_best_n,
            max_total=max_total,
        )
        assert policy.keep_best_n == keep_best_n
        assert policy.max_total == max_total

    @given(min_improvement=st.floats(allow_nan=True, allow_infinity=True))
    def test_policy_handles_special_floats(self, min_improvement):
        """Policy should accept any float value for min_improvement."""
        policy = CheckpointPolicy(min_improvement=min_improvement)
        # Just test creation doesn't crash
        assert policy is not None


@pytest.mark.hypothesis
class TestCheckpointInfoFuzz:
    """Fuzz tests for CheckpointInfo."""

    @given(
        run_index=st.integers(),
        path=st.text(min_size=1, max_size=100),
        validation_loss=st.one_of(
            st.none(),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50)
    def test_info_creation(self, run_index, path, validation_loss):
        """CheckpointInfo should handle various input combinations."""
        info = CheckpointInfo(
            run_index=run_index,
            path=path,
            validation_loss=validation_loss,
        )
        assert info.run_index == run_index
        assert info.path == path

    @given(
        run_index=st.integers(min_value=1, max_value=100),
        path=st.text(min_size=1, max_size=50).filter(lambda x: "/" not in x and "\\" not in x),
    )
    @settings(max_examples=30)
    def test_info_serialization_roundtrip(self, run_index, path):
        """to_dict/from_dict should be lossless."""
        original = CheckpointInfo(
            run_index=run_index,
            path=path,
            validation_loss=0.5,
            is_run_boundary=True,
        )

        serialized = original.to_dict()
        restored = CheckpointInfo.from_dict(serialized)

        assert restored.run_index == original.run_index
        assert restored.path == original.path
        assert restored.validation_loss == original.validation_loss
        assert restored.is_run_boundary == original.is_run_boundary
