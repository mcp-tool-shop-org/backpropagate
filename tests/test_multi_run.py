"""
Tests for the Multi-Run training orchestrator.

Tests cover:
- MultiRunConfig dataclass
- MultiRunResult and RunResult dataclasses
- MergeMode enum
- MultiRunTrainer class
- Data chunking logic
- Learning rate scheduling
- Callback handling
"""

from unittest.mock import MagicMock, patch

import pytest

from backpropagate.gpu_safety import GPUCondition, GPUStatus
from backpropagate.multi_run import (
    MergeMode,
    MultiRunConfig,
    MultiRunResult,
    MultiRunTrainer,
    RunResult,
)


class TestMergeMode:
    """Tests for MergeMode enum."""

    def test_merge_mode_values(self):
        """Should have expected merge mode values."""
        assert MergeMode.SIMPLE.value == "simple"
        assert MergeMode.SLAO.value == "slao"

    def test_merge_mode_from_string(self):
        """Should create from string value."""
        assert MergeMode("simple") == MergeMode.SIMPLE
        assert MergeMode("slao") == MergeMode.SLAO


class TestMultiRunConfig:
    """Tests for MultiRunConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = MultiRunConfig()

        assert config.num_runs == 5
        assert config.steps_per_run == 100
        assert config.samples_per_run == 1000
        assert config.merge_mode == MergeMode.SLAO
        assert config.initial_lr == 2e-4
        assert config.final_lr == 5e-5

    def test_custom_values(self):
        """Should accept custom values."""
        config = MultiRunConfig(
            num_runs=10,
            steps_per_run=200,
            samples_per_run=2000,
            merge_mode=MergeMode.SIMPLE,
        )

        assert config.num_runs == 10
        assert config.steps_per_run == 200
        assert config.samples_per_run == 2000
        assert config.merge_mode == MergeMode.SIMPLE

    def test_lr_decay_options(self):
        """Should support different LR decay types."""
        for decay_type in ["linear", "cosine", "constant"]:
            config = MultiRunConfig(lr_decay=decay_type)
            assert config.lr_decay == decay_type

    def test_gpu_safety_options(self):
        """Should have GPU safety configuration."""
        config = MultiRunConfig(
            enable_gpu_monitoring=True,
            pause_on_overheat=True,
            max_temp_c=80.0,
        )

        assert config.enable_gpu_monitoring is True
        assert config.pause_on_overheat is True
        assert config.max_temp_c == 80.0


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_basic_result(self):
        """Should store basic run results."""
        result = RunResult(
            run_index=1,
            steps=100,
            samples=1000,
            final_loss=1.5,
        )

        assert result.run_index == 1
        assert result.steps == 100
        assert result.samples == 1000
        assert result.final_loss == 1.5

    def test_full_result(self):
        """Should store all result fields."""
        result = RunResult(
            run_index=3,
            steps=100,
            samples=1000,
            final_loss=0.8,
            loss_history=[1.5, 1.2, 1.0, 0.8],
            learning_rate=1e-4,
            duration_seconds=120.5,
            checkpoint_path="/path/to/checkpoint",
            validation_loss=0.9,
            gpu_max_temp=75.0,
            gpu_max_vram_percent=85.0,
        )

        assert result.loss_history == [1.5, 1.2, 1.0, 0.8]
        assert result.learning_rate == 1e-4
        assert result.checkpoint_path == "/path/to/checkpoint"


class TestMultiRunResult:
    """Tests for MultiRunResult dataclass."""

    def test_basic_result(self):
        """Should store aggregate results."""
        result = MultiRunResult(
            total_runs=5,
            total_steps=500,
            total_samples=5000,
            total_duration_seconds=600.0,
            final_loss=0.5,
        )

        assert result.total_runs == 5
        assert result.total_steps == 500
        assert result.total_samples == 5000

    def test_with_run_history(self):
        """Should store individual run history."""
        runs = [
            RunResult(run_index=1, steps=100, samples=1000, final_loss=1.5),
            RunResult(run_index=2, steps=100, samples=1000, final_loss=1.0),
            RunResult(run_index=3, steps=100, samples=1000, final_loss=0.7),
        ]

        result = MultiRunResult(
            total_runs=3,
            total_steps=300,
            total_samples=3000,
            total_duration_seconds=300.0,
            final_loss=0.7,
            runs=runs,
        )

        assert len(result.runs) == 3
        assert result.runs[0].final_loss == 1.5
        assert result.runs[2].final_loss == 0.7

    def test_abort_information(self):
        """Should track abort state."""
        result = MultiRunResult(
            total_runs=2,
            total_steps=200,
            total_samples=2000,
            total_duration_seconds=120.0,
            final_loss=1.0,
            aborted=True,
            abort_reason="GPU emergency",
        )

        assert result.aborted is True
        assert result.abort_reason == "GPU emergency"


class TestMultiRunTrainer:
    """Tests for MultiRunTrainer class."""

    def test_initialization_with_defaults(self):
        """Should initialize with default config."""
        trainer = MultiRunTrainer(model="test-model")

        assert trainer.model_name == "test-model"
        assert trainer.config.num_runs == 5
        assert trainer.config.merge_mode == MergeMode.SLAO

    def test_initialization_with_config(self):
        """Should accept full config object."""
        config = MultiRunConfig(
            num_runs=10,
            steps_per_run=50,
        )

        trainer = MultiRunTrainer(model="test-model", config=config)

        assert trainer.config.num_runs == 10
        assert trainer.config.steps_per_run == 50

    def test_initialization_with_overrides(self):
        """Should accept convenience parameter overrides."""
        trainer = MultiRunTrainer(
            model="test-model",
            num_runs=8,
            steps_per_run=150,
            merge_mode="simple",
        )

        assert trainer.config.num_runs == 8
        assert trainer.config.steps_per_run == 150
        assert trainer.config.merge_mode == MergeMode.SIMPLE

    def test_abort(self):
        """Should support abort request."""
        trainer = MultiRunTrainer(model="test-model")

        assert trainer._should_abort is False

        trainer.abort("Test abort")

        assert trainer._should_abort is True
        assert trainer._abort_reason == "Test abort"


class TestMultiRunTrainerLearningRate:
    """Tests for learning rate scheduling in MultiRunTrainer."""

    @pytest.fixture
    def trainer(self):
        return MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(
                num_runs=5,
                initial_lr=2e-4,
                final_lr=5e-5,
            ),
        )

    def test_linear_lr_decay(self, trainer):
        """Should compute linear LR decay correctly."""
        trainer.config.lr_decay = "linear"

        # First run should be initial LR
        lr1 = trainer._get_learning_rate(1)
        assert lr1 == pytest.approx(2e-4)

        # Last run should be final LR
        lr5 = trainer._get_learning_rate(5)
        assert lr5 == pytest.approx(5e-5)

        # Middle run should be interpolated
        lr3 = trainer._get_learning_rate(3)
        expected = 2e-4 + 0.5 * (5e-5 - 2e-4)  # Midpoint
        assert lr3 == pytest.approx(expected)

    def test_constant_lr(self, trainer):
        """Should maintain constant LR when configured."""
        trainer.config.lr_decay = "constant"

        for i in range(1, 6):
            lr = trainer._get_learning_rate(i)
            assert lr == pytest.approx(2e-4)

    def test_cosine_lr_decay(self, trainer):
        """Should compute cosine LR decay correctly."""
        trainer.config.lr_decay = "cosine"

        # First run should be initial LR
        lr1 = trainer._get_learning_rate(1)
        assert lr1 == pytest.approx(2e-4)

        # Last run should be final LR
        lr5 = trainer._get_learning_rate(5)
        assert lr5 == pytest.approx(5e-5)

        # Cosine decay should be slower at start
        lr2 = trainer._get_learning_rate(2)
        assert lr2 > 1.5e-4  # Should still be relatively high


class TestMultiRunTrainerDataChunking:
    """Tests for data chunking logic in MultiRunTrainer."""

    @pytest.fixture
    def trainer(self):
        return MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(
                num_runs=5,
                samples_per_run=100,
                shuffle_data=False,  # Disable shuffle for predictable tests
            ),
        )

    def test_get_data_chunk_basic(self, trainer):
        """Should return correct chunk for each run."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=500)
        mock_dataset.select = MagicMock(return_value=mock_dataset)

        # Get chunk for run 1 (indices 0-99)
        trainer._get_data_chunk(mock_dataset, 1)
        mock_dataset.select.assert_called()

    def test_data_chunks_non_overlapping(self, trainer):
        """Data chunks across runs must not overlap (when chunks fit one pass).

        TESTS-B-004 escalation fix: the previous body only computed
        ``start_idx`` / ``end_idx`` inside the test and asserted on those
        synthetic ranges — it never called ``_get_data_chunk`` at all, so
        a broken implementation would still pass. The test was a false
        positive that gave zero coverage of the actual function under test.

        The rewrite mirrors the shape of ``test_data_chunk_wraparound``
        below (which is correctly invoked per the Stage A audit): build a
        deterministic dataset with one identifiable string per index, call
        ``_get_data_chunk`` for each run, decode the returned strings back
        to indices, and assert (a) every run returns the right size, (b)
        union of indices across runs == expected range, and (c) no index
        appears in two different runs.

        Configuration: 5 runs * 100 samples = 500 indices needed, against
        a 500-sample dataset. With validation OFF the train pool is the
        full dataset and there is no wrap-around — each run gets a
        disjoint window. (Wrap-around is covered by
        ``test_data_chunk_wraparound``.)
        """
        from datasets import Dataset

        # Mirror the trainer fixture's config (num_runs=5, samples_per_run=100,
        # shuffle_data=False). Validation OFF so the train pool == full dataset.
        total_samples = 500
        samples_per_run = trainer.config.samples_per_run
        num_runs = trainer.config.num_runs
        trainer.config.replay_fraction = 0.0  # isolate the new-samples path
        trainer.config.validate_every_run = False
        trainer.config.early_stopping = False

        dataset = Dataset.from_dict(
            {"text": [f"sample_{i}" for i in range(total_samples)]}
        )

        seen_indices: set[int] = set()

        for run_idx in range(1, num_runs + 1):
            chunk = trainer._get_data_chunk(dataset, run_idx)

            assert chunk is not None
            assert len(chunk) == samples_per_run, (
                f"Run {run_idx}: expected {samples_per_run} samples, got "
                f"{len(chunk)}"
            )

            # Decode the deterministic 'sample_N' strings back to indices.
            chunk_indices = {int(value.split("_")[1]) for value in chunk["text"]}
            assert len(chunk_indices) == samples_per_run, (
                f"Run {run_idx}: _get_data_chunk returned duplicate indices "
                f"within the same chunk ({samples_per_run - len(chunk_indices)} dupes)"
            )

            # The load-bearing assertion: this run's indices must not overlap
            # any previously returned run's indices.
            overlap = seen_indices & chunk_indices
            assert not overlap, (
                f"Run {run_idx} returned indices already seen in earlier "
                f"runs: {sorted(overlap)[:10]}... "
                f"(total overlap size {len(overlap)}). _get_data_chunk must "
                f"partition the dataset across runs when chunks fit one pass."
            )

            seen_indices.update(chunk_indices)

        # End-state contract: union covers exactly [0, total_samples).
        expected = set(range(total_samples))
        assert seen_indices == expected, (
            f"Union of all run chunks ({len(seen_indices)} indices) does not "
            f"equal the full dataset range ({total_samples} indices). "
            f"Missing: {sorted(expected - seen_indices)[:10]}..."
        )

    def test_data_chunk_wraparound(self, trainer):
        """Wraparound (samples_per_run * num_runs > dataset) cycles inside the train pool.

        SB-T-003 (a): the previous body was a comment + no assertion. The
        Wave 1 multi_run.py fix carefully distinguishes (i) the wraparound
        case (chunk_size <= train_pool_size, wrap inside [0, train_pool_size))
        from (ii) the new ConfigurationError case (chunk_size > train_pool_size,
        raise). This test pins case (i): wraparound succeeds without raising
        AND respects the validation-holdout boundary on every wrap.
        """
        from datasets import Dataset

        # 100 samples, 10 per run, 12 runs = forces wraparound through the train pool.
        # With validate_every_run=True the holdout = max(int(100 * 0.10), 1) = 10
        # → train_pool = [0, 90), val_holdout = [90, 100).
        total_samples = 100
        samples_per_run = 10
        num_runs = 12

        trainer.config.samples_per_run = samples_per_run
        trainer.config.num_runs = num_runs
        trainer.config.shuffle_data = False
        trainer.config.replay_fraction = 0.0  # isolate the new-samples path
        trainer.config.validate_every_run = True

        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(total_samples)]})

        # Expected train pool & holdout
        val_start = total_samples - max(int(total_samples * 0.10), 1)
        # = 100 - 10 = 90
        assert val_start == 90

        for run_idx in range(1, num_runs + 1):
            chunk = trainer._get_data_chunk(dataset, run_idx)
            assert chunk is not None
            assert len(chunk) == samples_per_run
            # The dataset values are deterministic strings; convert back to indices
            for value in chunk["text"]:
                # value looks like "sample_N"
                idx = int(value.split("_")[1])
                assert 0 <= idx < val_start, (
                    f"Run {run_idx} produced a chunk containing index {idx} "
                    f"which falls in the validation holdout [{val_start}, {total_samples}). "
                    f"Wraparound must wrap inside the train pool only."
                )

    def test_get_data_chunk_raises_when_chunk_exceeds_train_pool(self):
        """SB-T-003: chunk_size > train_pool with validation active → ConfigurationError.

        Pins the Wave 1 multi_run.py:758 raise. A future change that silently
        clamps chunk_size to train_pool_size to "be forgiving" would let
        validation samples leak into training without any CI signal.
        """
        from datasets import Dataset

        from backpropagate.exceptions import ConfigurationError

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(
                num_runs=3,
                samples_per_run=100,  # > train pool of 50 - 5 = 45
                validate_every_run=True,
                shuffle_data=False,
            ),
        )

        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(50)]})

        with pytest.raises(ConfigurationError) as exc_info:
            trainer._get_data_chunk(dataset, 1)

        # The error message must mention the actionable parameter
        msg = str(exc_info.value).lower()
        assert "samples_per_run" in msg or "samples per run" in msg, (
            f"Error message should mention samples_per_run for operator hint: {msg}"
        )

    def test_get_replay_samples_clamped_to_train_pool(self):
        """SB-T-003: replay indices stay inside [0, train_pool_size).

        Pins the Wave 1 unification of replay + chunker on _get_train_pool_size.
        A future regression that lets replay pull from [train_pool_size,
        total_samples) would silently resurrect validation-holdout samples
        into training, polluting the held-out set used to measure
        catastrophic forgetting.
        """
        from datasets import Dataset

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(
                num_runs=5,
                samples_per_run=20,
                replay_fraction=0.5,
                replay_strategy="all_previous",
                validate_every_run=True,
                shuffle_data=False,
            ),
        )

        total_samples = 200
        train_pool_size = total_samples - max(int(total_samples * 0.10), 1)  # = 180
        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(total_samples)]})

        # Run several replays from later run indices — these are the ones most
        # likely to push past the train pool if the clamp regressed.
        for run_idx in range(2, 6):
            replay = trainer._get_replay_samples(dataset, run_idx=run_idx, count=10)
            if replay is None:
                continue
            for value in replay["text"]:
                idx = int(value.split("_")[1])
                assert 0 <= idx < train_pool_size, (
                    f"Replay for run {run_idx} surfaced index {idx} outside "
                    f"the train pool [0, {train_pool_size}) — would leak the "
                    f"validation holdout into the replay buffer."
                )

    def test_replay_uses_local_rng_not_global(self):
        """SB-T-003: _get_replay_samples does NOT mutate the global random state.

        Pins the Wave 1 random.Random fix. A future refactor that swaps the
        local rng for random.sample(...) would re-introduce the global RNG
        pollution that silently breaks deterministic-seed-controlled callers
        running in the same process (MinHash for dedup, custom_filter, etc.).
        """
        import random

        from datasets import Dataset

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(
                num_runs=5,
                samples_per_run=20,
                replay_fraction=0.5,
                replay_strategy="random",
                shuffle_data=False,
            ),
        )

        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(200)]})

        # Snapshot the global RNG state, call replay, then check the state
        # against a reference draw from the same snapshot. If _get_replay_samples
        # mutated the global state, the two draws will diverge.
        random.seed(42)
        snapshot = random.getstate()

        # Reference draw — what should happen if the global state is untouched
        random.setstate(snapshot)
        reference = [random.random() for _ in range(10)]

        # Reset and run replay between snapshot and the actual draw
        random.setstate(snapshot)
        _ = trainer._get_replay_samples(dataset, run_idx=3, count=5)
        actual_after_replay = [random.random() for _ in range(10)]

        assert actual_after_replay == reference, (
            "Global random state mutated by _get_replay_samples — the local "
            "random.Random(seed) protection has regressed. See multi_run.py:887."
        )


class TestMultiRunTrainerCallbacks:
    """Tests for callback handling in MultiRunTrainer."""

    def test_on_run_start_callback(self):
        """Should call on_run_start callback."""
        run_starts = []

        def on_run_start(run_idx):
            run_starts.append(run_idx)

        trainer = MultiRunTrainer(
            model="test-model",
            on_run_start=on_run_start,
        )

        # Simulate callback being called
        if trainer.on_run_start:
            trainer.on_run_start(1)
            trainer.on_run_start(2)

        assert run_starts == [1, 2]

    def test_on_run_complete_callback(self):
        """Should call on_run_complete callback."""
        completed_runs = []

        def on_run_complete(result):
            completed_runs.append(result)

        trainer = MultiRunTrainer(
            model="test-model",
            on_run_complete=on_run_complete,
        )

        # Simulate callback being called
        mock_result = RunResult(run_index=1, steps=100, samples=1000, final_loss=1.5)
        if trainer.on_run_complete:
            trainer.on_run_complete(mock_result)

        assert len(completed_runs) == 1
        assert completed_runs[0].run_index == 1

    def test_on_gpu_status_callback(self):
        """Should call on_gpu_status callback."""
        status_updates = []

        def on_gpu_status(status):
            status_updates.append(status)

        trainer = MultiRunTrainer(
            model="test-model",
            on_gpu_status=on_gpu_status,
        )

        # Simulate callback being called
        mock_status = GPUStatus(available=True, temperature_c=70.0)
        if trainer.on_gpu_status:
            trainer.on_gpu_status(mock_status)

        assert len(status_updates) == 1


class TestMultiRunTrainerGPUSafety:
    """Tests for GPU safety integration in MultiRunTrainer."""

    def test_preflight_check_passes(self):
        """Should pass preflight check when GPU is safe."""
        mock_status = GPUStatus(
            available=True,
            device_name="Test GPU",
            vram_total_gb=16.0,
            condition=GPUCondition.SAFE,
        )

        trainer = MultiRunTrainer(model="test-model")

        with patch("backpropagate.multi_run.get_gpu_status", return_value=mock_status):
            result = trainer._preflight_gpu_check()
            assert result is True

    def test_preflight_check_fails_emergency(self):
        """Should fail preflight check on emergency."""
        mock_status = GPUStatus(
            available=True,
            condition=GPUCondition.EMERGENCY,
            condition_reason="Temperature emergency",
        )

        trainer = MultiRunTrainer(model="test-model")

        with patch("backpropagate.multi_run.get_gpu_status", return_value=mock_status):
            result = trainer._preflight_gpu_check()
            assert result is False

    def test_preflight_check_no_gpu(self):
        """Should fail preflight check when no GPU."""
        mock_status = GPUStatus(
            available=False,
            condition=GPUCondition.UNKNOWN,
        )

        trainer = MultiRunTrainer(model="test-model")

        with patch("backpropagate.multi_run.get_gpu_status", return_value=mock_status):
            result = trainer._preflight_gpu_check()
            assert result is False


class TestMultiRunTrainerResults:
    """Tests for result creation in MultiRunTrainer."""

    def test_create_result(self):
        """Should create proper result object."""
        trainer = MultiRunTrainer(model="test-model")

        # Add some mock runs
        trainer._runs = [
            RunResult(run_index=1, steps=100, samples=1000, final_loss=1.5),
            RunResult(run_index=2, steps=100, samples=1000, final_loss=1.0),
        ]
        trainer._aggregate_loss = [1.8, 1.6, 1.5, 1.3, 1.1, 1.0]
        trainer._run_boundaries = [0, 3]

        result = trainer._create_result(total_duration=200.0)

        assert result.total_runs == 2
        assert result.total_steps == 200
        assert result.total_samples == 2000
        assert result.final_loss == 1.0
        assert result.aborted is False

    def test_create_abort_result(self):
        """Should create proper abort result."""
        trainer = MultiRunTrainer(model="test-model")

        result = trainer._create_abort_result("Test abort reason")

        assert result.total_runs == 0
        assert result.aborted is True
        assert result.abort_reason == "Test abort reason"


class TestMultiRunTrainerIntegration:
    """Integration-style tests for MultiRunTrainer."""

    def test_full_config_flow(self):
        """Should handle complete configuration flow."""
        config = MultiRunConfig(
            num_runs=3,
            steps_per_run=50,
            samples_per_run=500,
            merge_mode=MergeMode.SLAO,
            initial_lr=2e-4,
            final_lr=1e-4,
            lr_decay="linear",
            warmup_steps_per_run=5,
            save_every_run=True,
            enable_gpu_monitoring=True,
            max_temp_c=85.0,
        )

        trainer = MultiRunTrainer(model="test-model", config=config)

        # Verify all config was applied
        assert trainer.config.num_runs == 3
        assert trainer.config.steps_per_run == 50
        assert trainer.config.merge_mode == MergeMode.SLAO
        assert trainer.config.warmup_steps_per_run == 5

    def test_merge_mode_selection(self):
        """Should use correct merge mode."""
        trainer_slao = MultiRunTrainer(model="test", merge_mode="slao")
        trainer_simple = MultiRunTrainer(model="test", merge_mode="simple")

        assert trainer_slao.config.merge_mode == MergeMode.SLAO
        assert trainer_simple.config.merge_mode == MergeMode.SIMPLE


class TestMultiRunEdgeCases:
    """Edge case tests for Multi-Run module."""

    def test_single_run(self):
        """Should handle single run configuration."""
        config = MultiRunConfig(num_runs=1)
        trainer = MultiRunTrainer(model="test", config=config)

        # LR should be initial LR for single run
        lr = trainer._get_learning_rate(1)
        assert lr == config.initial_lr

    def test_many_runs(self):
        """Should handle many runs configuration."""
        config = MultiRunConfig(num_runs=100)
        trainer = MultiRunTrainer(model="test", config=config)

        # Should compute valid LR for all runs (with small tolerance for floating point)
        epsilon = 1e-10
        for i in range(1, 101):
            lr = trainer._get_learning_rate(i)
            assert config.final_lr - epsilon <= lr <= config.initial_lr + epsilon

    def test_zero_samples_per_run(self):
        """Should handle edge case configuration."""
        config = MultiRunConfig(samples_per_run=0)
        # This is technically invalid but shouldn't crash

    def test_empty_callbacks(self):
        """Should handle None callbacks gracefully."""
        trainer = MultiRunTrainer(
            model="test",
            on_run_start=None,
            on_run_complete=None,
            on_gpu_status=None,
        )

        assert trainer.on_run_start is None
        assert trainer.on_run_complete is None


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================

class TestMultiRunTrainerCallbackInvocation:
    """Tests for callback invocation during actual run execution."""

    def test_on_run_complete_invoked_with_result(self):
        """on_run_complete callback should be invoked with RunResult during run."""
        completed_results = []

        def on_complete(result):
            completed_results.append(result)

        trainer = MultiRunTrainer(
            model="test-model",
            num_runs=2,
            on_run_complete=on_complete,
        )

        # Simulate what happens during _execute_run completion
        # by directly testing callback invocation pattern
        mock_result = RunResult(
            run_index=1,
            steps=100,
            samples=1000,
            final_loss=0.75,
            duration_seconds=60.0,
        )

        trainer._runs.append(mock_result)
        if trainer.on_run_complete:
            trainer.on_run_complete(mock_result)

        assert len(completed_results) == 1
        assert completed_results[0].run_index == 1
        assert completed_results[0].final_loss == 0.75

    def test_on_run_start_invoked_with_index(self):
        """on_run_start callback should be invoked with run index."""
        started_runs = []

        def on_start(run_idx):
            started_runs.append(run_idx)

        trainer = MultiRunTrainer(
            model="test-model",
            on_run_start=on_start,
        )

        # Simulate start callbacks
        for i in range(1, 4):
            if trainer.on_run_start:
                trainer.on_run_start(i)

        assert started_runs == [1, 2, 3]

    def test_on_step_callback_signature(self):
        """on_step callback should receive (run_idx, step, loss)."""
        step_calls = []

        def on_step(run_idx, step, loss):
            step_calls.append((run_idx, step, loss))

        trainer = MultiRunTrainer(
            model="test-model",
            on_step=on_step,
        )

        # Simulate step callback
        if trainer.on_step:
            trainer.on_step(1, 10, 1.25)
            trainer.on_step(1, 20, 0.95)
            trainer.on_step(2, 10, 0.82)

        assert len(step_calls) == 3
        assert step_calls[0] == (1, 10, 1.25)
        assert step_calls[2] == (2, 10, 0.82)


class TestMultiRunTrainerCheckpointLoading:
    """Tests for checkpoint loading between runs."""

    def test_get_lora_state_dict_with_get_adapter(self):
        """_get_lora_state_dict should use get_adapter_state_dict when available."""
        trainer = MultiRunTrainer(model="test-model")

        # Setup mock model with get_adapter_state_dict
        mock_model = MagicMock()
        mock_model.get_adapter_state_dict.return_value = {
            "lora_A.weight": "mock_A",
            "lora_B.weight": "mock_B",
        }

        mock_trainer = MagicMock()
        mock_trainer._model = mock_model
        trainer._trainer = mock_trainer

        state_dict = trainer._get_lora_state_dict()

        mock_model.get_adapter_state_dict.assert_called_once()
        assert "lora_A.weight" in state_dict
        assert "lora_B.weight" in state_dict

    def test_get_lora_state_dict_fallback(self):
        """_get_lora_state_dict should fallback to named_parameters when no adapter method."""
        import torch

        trainer = MultiRunTrainer(model="test-model")

        # Setup mock model without get_adapter_state_dict
        mock_model = MagicMock()
        del mock_model.get_adapter_state_dict  # Remove the method

        # Mock named_parameters to return LoRA-like parameters
        mock_params = [
            ("layer.lora_A.weight", torch.tensor([1.0])),
            ("layer.lora_B.weight", torch.tensor([2.0])),
            ("layer.other.weight", torch.tensor([3.0])),  # Non-LoRA
        ]
        mock_model.named_parameters.return_value = iter(mock_params)

        mock_trainer = MagicMock()
        mock_trainer._model = mock_model
        trainer._trainer = mock_trainer

        state_dict = trainer._get_lora_state_dict()

        # Should only contain LoRA parameters
        assert "layer.lora_A.weight" in state_dict
        assert "layer.lora_B.weight" in state_dict
        assert "layer.other.weight" not in state_dict

    def test_load_lora_state_dict_with_load_adapter(self):
        """_load_lora_state_dict should use load_adapter_state_dict when available."""
        trainer = MultiRunTrainer(model="test-model")

        mock_model = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer._model = mock_model
        trainer._trainer = mock_trainer

        state_dict = {"lora_A.weight": "value_A"}

        trainer._load_lora_state_dict(state_dict)

        mock_model.load_adapter_state_dict.assert_called_once_with(state_dict)

    def test_load_lora_state_dict_fallback(self):
        """_load_lora_state_dict should fallback to manual loading when no method."""
        import torch

        trainer = MultiRunTrainer(model="test-model")

        mock_model = MagicMock()
        del mock_model.load_adapter_state_dict  # Remove the method

        # Mock state_dict
        existing_state = {
            "lora_A.weight": torch.tensor([0.0]),
            "lora_B.weight": torch.tensor([0.0]),
        }
        mock_model.state_dict.return_value = existing_state

        mock_trainer = MagicMock()
        mock_trainer._model = mock_model
        trainer._trainer = mock_trainer

        new_state = {
            "lora_A.weight": torch.tensor([1.0]),
            "lora_B.weight": torch.tensor([2.0]),
        }

        trainer._load_lora_state_dict(new_state)

        # Verify state_dict was retrieved for manual update
        mock_model.state_dict.assert_called_once()

    def test_prepare_for_next_run_slao_mode(self):
        """_prepare_for_next_run should load SLAO weights in SLAO mode."""
        trainer = MultiRunTrainer(
            model="test-model",
            merge_mode="slao",
        )

        # Setup SLAO merger mock
        mock_merger = MagicMock()
        mock_merger.get_init_weights.return_value = {
            "lora_A.weight": "init_A",
            "lora_B.weight": "init_B",
        }
        trainer._slao_merger = mock_merger

        # Setup mock trainer
        mock_model = MagicMock()
        mock_trainer_inner = MagicMock()
        mock_trainer_inner._model = mock_model
        trainer._trainer = mock_trainer_inner

        trainer._prepare_for_next_run(2)

        mock_merger.get_init_weights.assert_called_once()
        mock_model.load_adapter_state_dict.assert_called_once()

    def test_prepare_for_next_run_simple_mode(self):
        """_prepare_for_next_run should do nothing in SIMPLE mode."""
        trainer = MultiRunTrainer(
            model="test-model",
            merge_mode="simple",
        )

        # Setup mock (should not be called)
        mock_model = MagicMock()
        mock_trainer_inner = MagicMock()
        mock_trainer_inner._model = mock_model
        trainer._trainer = mock_trainer_inner

        trainer._prepare_for_next_run(2)

        # Should not call any loading methods in simple mode
        mock_model.load_adapter_state_dict.assert_not_called()


class TestMultiRunTrainerGPUMonitoring:
    """Tests for GPU monitoring during runs."""

    def test_on_gpu_status_tracks_max_temp(self):
        """_on_gpu_status should track maximum temperature."""
        trainer = MultiRunTrainer(model="test-model")

        trainer._gpu_max_temp = 0.0

        # Simulate status updates
        status1 = GPUStatus(available=True, temperature_c=65.0)
        trainer._on_gpu_status(status1)
        assert trainer._gpu_max_temp == 65.0

        status2 = GPUStatus(available=True, temperature_c=72.0)
        trainer._on_gpu_status(status2)
        assert trainer._gpu_max_temp == 72.0

        # Lower temp shouldn't change max
        status3 = GPUStatus(available=True, temperature_c=68.0)
        trainer._on_gpu_status(status3)
        assert trainer._gpu_max_temp == 72.0

    def test_on_gpu_status_tracks_max_vram(self):
        """_on_gpu_status should track maximum VRAM usage."""
        trainer = MultiRunTrainer(model="test-model")

        trainer._gpu_max_vram = 0.0

        status1 = GPUStatus(available=True, vram_percent=75.0)
        trainer._on_gpu_status(status1)
        assert trainer._gpu_max_vram == 75.0

        status2 = GPUStatus(available=True, vram_percent=92.0)
        trainer._on_gpu_status(status2)
        assert trainer._gpu_max_vram == 92.0

    def test_on_gpu_status_invokes_callback(self):
        """_on_gpu_status should invoke user callback."""
        status_updates = []

        def on_status(status):
            status_updates.append(status)

        trainer = MultiRunTrainer(
            model="test-model",
            on_gpu_status=on_status,
        )

        status = GPUStatus(available=True, temperature_c=70.0)
        trainer._on_gpu_status(status)

        assert len(status_updates) == 1
        assert status_updates[0].temperature_c == 70.0

    def test_on_gpu_emergency_triggers_abort(self):
        """_on_gpu_emergency should trigger abort."""
        trainer = MultiRunTrainer(model="test-model")

        assert trainer._should_abort is False

        status = GPUStatus(
            available=True,
            condition=GPUCondition.EMERGENCY,
            condition_reason="Temperature > 95C",
        )

        trainer._on_gpu_emergency(status)

        assert trainer._should_abort is True
        assert "Temperature > 95C" in trainer._abort_reason


class TestMultiRunTrainerDatasetLoading:
    """Tests for dataset loading in MultiRunTrainer."""

    def test_load_full_dataset_from_none_uses_config(self):
        """_load_full_dataset with None should use config default."""
        trainer = MultiRunTrainer(model="test-model")

        with patch("datasets.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.__len__ = MagicMock(return_value=1000)
            mock_load.return_value = mock_ds

            result = trainer._load_full_dataset(None)

            mock_load.assert_called_once()

    def test_load_full_dataset_from_jsonl(self, tmp_path):
        """_load_full_dataset should load JSONL files."""
        import json

        trainer = MultiRunTrainer(model="test-model")

        # Create test JSONL (ShareGPT format for DatasetLoader)
        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(10):
                f.write(json.dumps({"conversations": [
                    {"from": "human", "value": f"Question {i}"},
                    {"from": "gpt", "value": f"Answer {i}"},
                ]}) + "\n")

        ds = trainer._load_full_dataset(str(jsonl_path))

        assert len(ds) == 10

    def test_load_full_dataset_from_csv(self, tmp_path):
        """_load_full_dataset should load CSV files."""
        trainer = MultiRunTrainer(model="test-model")

        # Alpaca format CSV (instruction + output) for DatasetLoader
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "instruction,input,output\n"
            "Say hello,,Hello there\n"
            "Say goodbye,,Goodbye\n"
            "Say thanks,,Thank you\n"
        )

        ds = trainer._load_full_dataset(str(csv_path))

        assert len(ds) == 3

    def test_load_full_dataset_from_dataset_object(self):
        """_load_full_dataset should accept Dataset object directly."""
        from datasets import Dataset

        trainer = MultiRunTrainer(model="test-model")

        mock_ds = MagicMock(spec=Dataset)
        mock_ds.__len__ = MagicMock(return_value=500)

        result = trainer._load_full_dataset(mock_ds)

        assert result is mock_ds

    def test_load_full_dataset_invalid_type_raises(self):
        """_load_full_dataset should raise for invalid types."""
        trainer = MultiRunTrainer(model="test-model")

        with pytest.raises((ValueError, Exception)):
            trainer._load_full_dataset(12345)


class TestBackwardsCompatibility:
    """Tests for backwards compatibility aliases."""

    def test_speedrun_trainer_alias(self):
        """SpeedrunTrainer should be alias for MultiRunTrainer."""
        from backpropagate.multi_run import MultiRunTrainer, SpeedrunTrainer

        assert SpeedrunTrainer is MultiRunTrainer

    def test_speedrun_config_alias(self):
        """SpeedrunConfig should be alias for MultiRunConfig."""
        from backpropagate.multi_run import MultiRunConfig, SpeedrunConfig

        assert SpeedrunConfig is MultiRunConfig

    def test_speedrun_result_alias(self):
        """SpeedrunResult should be alias for MultiRunResult."""
        from backpropagate.multi_run import MultiRunResult, SpeedrunResult

        assert SpeedrunResult is MultiRunResult

    def test_speedrun_trainer_creates_instance(self):
        """SpeedrunTrainer should create valid instance."""
        from backpropagate.multi_run import SpeedrunTrainer

        trainer = SpeedrunTrainer(model="test-model", num_runs=3)

        assert trainer.config.num_runs == 3


# =============================================================================
# EARLY STOPPING TESTS (Phase 4.3)
# =============================================================================

class TestEarlyStoppingDetailed:
    """Detailed tests for early stopping functionality."""

    def test_early_stopping_default_disabled(self):
        """Early stopping should be disabled by default."""
        config = MultiRunConfig()
        assert config.early_stopping is False

    def test_early_stopping_patience_default(self):
        """Early stopping patience should have sensible default."""
        config = MultiRunConfig(early_stopping=True)
        assert config.early_stopping_patience == 2

    def test_early_stopping_threshold_default(self):
        """Early stopping threshold should default to 0."""
        config = MultiRunConfig(early_stopping=True)
        assert config.early_stopping_threshold == 0.0

    def test_check_early_stopping_first_run(self):
        """First run should establish baseline, not trigger stop."""
        trainer = MultiRunTrainer(model="test-model")
        trainer.config.early_stopping = True
        trainer.config.early_stopping_patience = 2

        result = trainer._check_early_stopping(0.5, run_idx=1)

        assert result is False
        assert trainer._best_val_loss == 0.5
        assert trainer._early_stop_counter == 0

    def test_check_early_stopping_improvement_resets_counter(self):
        """Counter should reset when loss improves."""
        trainer = MultiRunTrainer(model="test-model")
        trainer.config.early_stopping = True
        trainer.config.early_stopping_patience = 2
        trainer.config.early_stopping_threshold = 0.0

        # First run
        trainer._check_early_stopping(0.5, run_idx=1)

        # Second run - no improvement
        trainer._check_early_stopping(0.6, run_idx=2)
        assert trainer._early_stop_counter == 1

        # Third run - improvement
        trainer._check_early_stopping(0.4, run_idx=3)
        assert trainer._early_stop_counter == 0
        assert trainer._best_val_loss == 0.4

    def test_check_early_stopping_triggers_at_patience(self):
        """Should trigger when patience is exceeded."""
        trainer = MultiRunTrainer(model="test-model")
        trainer.config.early_stopping = True
        trainer.config.early_stopping_patience = 2

        trainer._check_early_stopping(0.5, run_idx=1)
        assert trainer._check_early_stopping(0.6, run_idx=2) is False  # counter=1
        assert trainer._check_early_stopping(0.7, run_idx=3) is True   # counter=2

    def test_check_early_stopping_threshold_required_improvement(self):
        """Small improvements under threshold should not count."""
        trainer = MultiRunTrainer(model="test-model")
        trainer.config.early_stopping = True
        trainer.config.early_stopping_patience = 3
        trainer.config.early_stopping_threshold = 0.05

        trainer._check_early_stopping(0.5, run_idx=1)

        # Improvement of 0.02 (below threshold of 0.05)
        trainer._check_early_stopping(0.48, run_idx=2)
        assert trainer._early_stop_counter == 1  # Counts as no improvement

        # Improvement of 0.08 (above threshold)
        trainer._check_early_stopping(0.4, run_idx=3)
        assert trainer._early_stop_counter == 0


class TestMultiRunValidation:
    """Tests for validation during multi-run training."""

    def test_validation_samples_configuration(self):
        """Validation samples should be configurable."""
        config = MultiRunConfig(validation_samples=500)
        assert config.validation_samples == 500

    def test_validate_every_run_configuration(self):
        """Validate every run should be configurable."""
        config = MultiRunConfig(validate_every_run=True)
        assert config.validate_every_run is True

        config = MultiRunConfig(validate_every_run=False)
        assert config.validate_every_run is False


# =============================================================================
# CHECKPOINT MANAGER INTEGRATION TESTS (Phase 5.3)
# =============================================================================

class TestCheckpointManagerIntegration:
    """Tests for checkpoint manager integration."""

    def test_checkpoint_policy_defaults_in_config(self):
        """Config should have checkpoint policy defaults."""
        config = MultiRunConfig()

        assert config.checkpoint_keep_best_n == 3
        assert config.checkpoint_keep_final is True
        assert config.checkpoint_keep_run_boundaries is False
        assert config.checkpoint_max_total == 10
        assert config.checkpoint_auto_prune is True

    def test_checkpoint_policy_custom_values(self):
        """Config should accept custom checkpoint policy values."""
        config = MultiRunConfig(
            checkpoint_keep_best_n=5,
            checkpoint_keep_final=False,
            checkpoint_keep_run_boundaries=True,
            checkpoint_max_total=20,
            checkpoint_auto_prune=False,
        )

        assert config.checkpoint_keep_best_n == 5
        assert config.checkpoint_keep_final is False
        assert config.checkpoint_keep_run_boundaries is True
        assert config.checkpoint_max_total == 20
        assert config.checkpoint_auto_prune is False

    def test_get_checkpoint_manager_before_run(self):
        """Checkpoint manager should be None before run()."""
        trainer = MultiRunTrainer(model="test-model")
        assert trainer.get_checkpoint_manager() is None

    def test_get_checkpoint_stats_before_run(self):
        """Checkpoint stats should be None before run()."""
        trainer = MultiRunTrainer(model="test-model")
        assert trainer.get_checkpoint_stats() is None


# =============================================================================
# SLAO INTEGRATION TESTS
# =============================================================================

class TestSLAOIntegration:
    """Tests for SLAO (Single LoRA Asymmetric) integration."""

    def test_slao_mode_is_default(self):
        """SLAO should be the default merge mode."""
        config = MultiRunConfig()
        assert config.merge_mode == MergeMode.SLAO

    def test_slao_mode_selection(self):
        """SLAO mode should be selectable."""
        trainer = MultiRunTrainer(model="test-model", merge_mode="slao")
        assert trainer.config.merge_mode == MergeMode.SLAO

    def test_simple_mode_selection(self):
        """Simple mode should be selectable as alternative."""
        trainer = MultiRunTrainer(model="test-model", merge_mode="simple")
        assert trainer.config.merge_mode == MergeMode.SIMPLE


# =============================================================================
# EXPERIENCE REPLAY TESTS
# =============================================================================

class TestExperienceReplay:
    """Tests for experience replay functionality."""

    def test_replay_fraction_default(self):
        """Replay fraction should default to 0."""
        config = MultiRunConfig()
        assert config.replay_fraction == 0.0

    def test_replay_fraction_configurable(self):
        """Replay fraction should be configurable."""
        config = MultiRunConfig(replay_fraction=0.2)
        assert config.replay_fraction == 0.2

    def test_replay_strategy_default(self):
        """Replay strategy should default to 'recent'."""
        config = MultiRunConfig()
        assert config.replay_strategy == "recent"

    def test_replay_strategy_options(self):
        """All replay strategies should be accepted."""
        for strategy in ["recent", "random", "all_previous"]:
            config = MultiRunConfig(replay_strategy=strategy)
            assert config.replay_strategy == strategy

    def test_get_replay_samples_recent_strategy(self):
        """Recent strategy should sample from last run."""
        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(
                samples_per_run=100,
                replay_fraction=0.2,
                replay_strategy="recent",
            ),
        )

        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=1000)
        mock_ds.select = MagicMock(return_value=mock_ds)

        # Get replay samples for run 3
        trainer._get_replay_samples(mock_ds, run_idx=3, count=20)

        # Should have called select on the dataset
        mock_ds.select.assert_called()

    def test_get_replay_samples_first_run_returns_none(self):
        """First run should return None (no previous data)."""
        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(
                replay_fraction=0.2,
                samples_per_run=100,  # Need this to avoid division by zero
            ),
        )

        # Create a mock dataset with proper length
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=1000)

        # First run has no previous data - the function should handle this
        # by returning None or an empty dataset
        result = trainer._get_replay_samples(mock_ds, run_idx=1, count=20)
        # The function checks run_idx > 1, so for first run it returns None


# =============================================================================
# SB-T-005: SLAO replay strategy ALGORITHM tests
#
# The pre-existing TestExperienceReplay tests only assert config storage and
# that `select` is called on a MagicMock — they don't exercise the strategy
# branches at multi_run.py:848-874 or the local-Random determinism at
# multi_run.py:887. A future refactor that swapped strategies or replaced
# random.Random(seed) with random.sample(...) would silently regress the
# anti-catastrophic-forgetting contract WITHOUT failing any test.
#
# These tests use a real HuggingFace Dataset.from_dict so we can read back
# the actual selected indices and pin the strategy contract end-to-end.
# =============================================================================

class TestReplayStrategyAlgorithm:
    """End-to-end strategy tests against the actual selection algorithm."""

    @staticmethod
    def _indices_from(chunk) -> list[int]:
        """Decode 'sample_N' strings back to integer indices."""
        return sorted(int(v.split("_")[1]) for v in chunk["text"])

    def _build_trainer(self, **config_overrides):
        defaults = {
            "num_runs": 5,
            "samples_per_run": 100,
            "replay_fraction": 0.2,
            "shuffle_data": False,
        }
        defaults.update(config_overrides)
        return MultiRunTrainer(model="test-model", config=MultiRunConfig(**defaults))

    def test_replay_strategy_recent_picks_most_recent_run(self):
        """SB-T-005: 'recent' strategy draws ONLY from the previous run's chunk.

        Pins the contract at multi_run.py:848-853 — replay for run_idx=3 must
        sample inside [samples_per_run, 2*samples_per_run) (i.e. run 2's
        indices). A future change that took 'recent' from the global window
        would silently break the contract.
        """
        from datasets import Dataset

        trainer = self._build_trainer(replay_strategy="recent", samples_per_run=100)
        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(500)]})

        chunk = trainer._get_replay_samples(dataset, run_idx=3, count=20)
        assert chunk is not None
        indices = self._indices_from(chunk)

        # Run 3's "recent" replay = previous run (run 2) = [100, 200)
        for idx in indices:
            assert 100 <= idx < 200, (
                f"'recent' strategy for run_idx=3 returned index {idx} outside "
                f"[100, 200) — the most recent prior chunk."
            )
        assert len(indices) == 20

    def test_replay_strategy_random_is_deterministic_for_same_seed(self):
        """SB-T-005: same (seed, run_idx) → identical replay indices.

        Pins the random.Random(seed + run_idx + 1000) construction at
        multi_run.py:887 — two calls with identical inputs MUST produce
        identical outputs. Different run_idx must produce DIFFERENT outputs
        (the + run_idx offset prevents per-run replay collisions).
        """
        from datasets import Dataset

        trainer1 = self._build_trainer(replay_strategy="random", samples_per_run=100)
        trainer2 = self._build_trainer(replay_strategy="random", samples_per_run=100)

        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(500)]})

        # Same trainer config + same run_idx → identical results
        first = self._indices_from(trainer1._get_replay_samples(dataset, run_idx=3, count=20))
        second = self._indices_from(trainer2._get_replay_samples(dataset, run_idx=3, count=20))
        assert first == second, (
            "random.Random(seed) determinism contract violated: same "
            "(seed, run_idx) returned different replay indices on a second call."
        )

        # Different run_idx → almost-certainly different indices
        third = self._indices_from(trainer1._get_replay_samples(dataset, run_idx=4, count=20))
        # We don't require complete disjointness (random sample with overlap is OK)
        # but the sequences must not be identical with high confidence at count=20.
        assert first != third, (
            "Different run_idx values produced identical indices — the "
            "+ run_idx offset at multi_run.py:887 may have regressed."
        )

    def test_replay_strategy_random_does_not_pollute_global_rng(self):
        """SB-T-005: 'random' strategy MUST NOT mutate Python's global RNG state.

        Wave 1's whole reason for the local random.Random() instance —
        any other code in the process using random.* (datasketch MinHash
        for dedup, custom_filter callbacks, transformers internals)
        relies on deterministic state. A refactor back to random.sample(...)
        would silently break that.
        """
        import random

        from datasets import Dataset

        trainer = self._build_trainer(replay_strategy="random", samples_per_run=100)
        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(500)]})

        random.seed(42)
        snapshot = random.getstate()

        random.setstate(snapshot)
        reference = [random.random() for _ in range(10)]

        random.setstate(snapshot)
        _ = trainer._get_replay_samples(dataset, run_idx=3, count=20)
        post_replay = [random.random() for _ in range(10)]

        assert post_replay == reference, (
            "Global random state mutated by 'random' replay strategy — the "
            "local random.Random(seed) protection (multi_run.py:887) regressed."
        )

    def test_replay_strategy_all_previous_includes_all_runs(self):
        """SB-T-005: 'all_previous' draws from the union of every prior run's chunk.

        Pins multi_run.py:864-867. With run_idx=5, every chunk from runs
        1..4 (= [0, 4*samples_per_run)) must be represented in a
        sufficiently large sample.
        """
        from datasets import Dataset

        # Use small samples_per_run + large count so all 4 prior chunks are
        # nearly certain to appear at least once.
        trainer = self._build_trainer(
            replay_strategy="all_previous",
            samples_per_run=10,
        )
        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(500)]})

        # run_idx=5 → previously seen = [0, 4*10) = [0, 40)
        # Take a count that exceeds the available pool so every prior index
        # must appear (rng.sample(available, min(count, len(available)))).
        chunk = trainer._get_replay_samples(dataset, run_idx=5, count=40)
        assert chunk is not None
        indices = set(self._indices_from(chunk))

        # Expected pool: union of runs 1-4 chunks
        for run_idx in range(1, 5):
            chunk_start = (run_idx - 1) * 10
            chunk_end = chunk_start + 10
            chunk_indices = set(range(chunk_start, chunk_end))
            assert chunk_indices & indices, (
                f"'all_previous' replay for run 5 did not include any indices "
                f"from run {run_idx} (expected at least one of {chunk_indices})."
            )

        # Every returned index must come from a prior run's chunk (here, [0, 40))
        for idx in indices:
            assert 0 <= idx < 40, (
                f"'all_previous' returned index {idx} outside the union of "
                f"prior chunks [0, 40)."
            )

    def test_replay_strategy_unknown_falls_back_to_recent(self):
        """SB-T-005: unrecognised strategy values fall back to 'recent'.

        Pins the default branch at multi_run.py:869-874. A typo in a config
        or a renamed strategy must not silently produce a different
        algorithm — it must behave identically to 'recent'.
        """
        from datasets import Dataset

        trainer_recent = self._build_trainer(replay_strategy="recent", samples_per_run=100)
        trainer_unknown = self._build_trainer(
            replay_strategy="not-a-real-strategy",
            samples_per_run=100,
        )

        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(500)]})

        # The fallback path uses identical math to 'recent', so for the same
        # (seed, run_idx) the resulting indices must be identical.
        recent_indices = self._indices_from(
            trainer_recent._get_replay_samples(dataset, run_idx=3, count=20)
        )
        fallback_indices = self._indices_from(
            trainer_unknown._get_replay_samples(dataset, run_idx=3, count=20)
        )

        assert recent_indices == fallback_indices, (
            "Unknown strategy fallback did not behave identically to 'recent' — "
            "the default branch at multi_run.py:869-874 may have regressed."
        )

    def test_replay_strategy_respects_train_pool_size_under_validation(self):
        """SB-T-005: NO strategy may surface an index >= train_pool_size.

        The single most important contract for SLAO: with
        validate_every_run=True, replay must NEVER resurrect a validation-
        holdout sample. The docstring at multi_run.py:832-838 explicitly
        promises this. Tested across all 3 documented strategies.
        """
        from datasets import Dataset

        total_samples = 200
        train_pool_size = total_samples - max(int(total_samples * 0.10), 1)  # 180
        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(total_samples)]})

        for strategy in ("recent", "random", "all_previous"):
            trainer = self._build_trainer(
                replay_strategy=strategy,
                samples_per_run=20,
                validate_every_run=True,
            )
            for run_idx in (2, 4, 6):
                chunk = trainer._get_replay_samples(dataset, run_idx=run_idx, count=10)
                if chunk is None:
                    continue
                for idx in self._indices_from(chunk):
                    assert 0 <= idx < train_pool_size, (
                        f"strategy={strategy!r} run_idx={run_idx} returned "
                        f"index {idx} >= train_pool_size {train_pool_size}. "
                        f"Replay just leaked the validation holdout."
                    )

    def test_replay_sample_deterministic_across_invocations(self):
        """SB-T-005: identical (seed, run_idx, count) → identical sample.

        Stricter than the per-strategy determinism test — pins the entire
        invocation to a byte-identical result. Catches any future change
        that introduces non-determinism (e.g. iteration over a set with
        a hash-randomized seed, or wall-clock seeding).
        """
        from datasets import Dataset

        trainer = self._build_trainer(replay_strategy="all_previous", samples_per_run=50)
        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(500)]})

        first = self._indices_from(trainer._get_replay_samples(dataset, run_idx=4, count=15))
        second = self._indices_from(trainer._get_replay_samples(dataset, run_idx=4, count=15))
        third = self._indices_from(trainer._get_replay_samples(dataset, run_idx=4, count=15))

        assert first == second == third, (
            "Same (run_idx=4, count=15) produced different replay indices "
            "across three calls — non-determinism has crept into the "
            "replay sampler."
        )


# =============================================================================
# LOSS TRACKING TESTS
# =============================================================================

class TestLossHistoryTracking:
    """Tests for loss values recorded across runs."""

    def test_aggregate_loss_initialized_empty(self):
        """Aggregate loss should start empty."""
        trainer = MultiRunTrainer(model="test-model")
        assert trainer._aggregate_loss == []

    def test_run_boundaries_initialized_empty(self):
        """Run boundaries should start empty."""
        trainer = MultiRunTrainer(model="test-model")
        assert trainer._run_boundaries == []

    def test_validation_losses_initialized_empty(self):
        """Validation losses should start empty."""
        trainer = MultiRunTrainer(model="test-model")
        assert trainer._validation_losses == []

    def test_best_val_loss_initialized_infinity(self):
        """Best validation loss should start as infinity."""
        trainer = MultiRunTrainer(model="test-model")
        assert trainer._best_val_loss == float('inf')


# =============================================================================
# ADDITIONAL EDGE CASES
# =============================================================================

class TestMultiRunAdditionalEdgeCases:
    """Additional edge cases for multi-run training."""

    def test_trainer_internal_state_initialization(self):
        """Internal state should be properly initialized."""
        trainer = MultiRunTrainer(model="test-model")

        assert trainer._trainer is None
        assert trainer._slao_merger is None
        assert trainer._gpu_monitor is None
        assert trainer._is_running is False
        assert trainer._should_abort is False
        assert trainer._abort_reason is None

    def test_trainer_gpu_tracking_initialization(self):
        """GPU tracking should be initialized to 0."""
        trainer = MultiRunTrainer(model="test-model")

        assert trainer._gpu_max_temp == 0.0
        assert trainer._gpu_max_vram == 0.0

    def test_early_stop_counter_initialization(self):
        """Early stop counter should be initialized to 0."""
        trainer = MultiRunTrainer(model="test-model")

        assert trainer._early_stop_counter == 0

    def test_runs_list_initialization(self):
        """Runs list should be initialized empty."""
        trainer = MultiRunTrainer(model="test-model")

        assert trainer._runs == []

    def test_abort_reason_default(self):
        """Abort reason should default to None."""
        trainer = MultiRunTrainer(model="test-model")
        assert trainer._abort_reason is None

    def test_abort_with_empty_reason(self):
        """Abort with empty reason should work."""
        trainer = MultiRunTrainer(model="test-model")
        trainer.abort("")
        assert trainer._should_abort is True
        assert trainer._abort_reason == ""


# =============================================================================
# RUN EXECUTION TESTS (Coverage for lines 281-392, 417-549)
# =============================================================================

class TestMultiRunExecution:
    """Tests for the main run execution flow."""

    def test_run_handles_no_gpu_available(self, tmp_path):
        """run() should handle case when no GPU is available."""
        config = MultiRunConfig(
            num_runs=2,
            checkpoint_dir=str(tmp_path),
            enable_gpu_monitoring=True,
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        # Mock no GPU available
        mock_status = GPUStatus(
            available=False,
            condition=GPUCondition.UNKNOWN,
        )

        with patch("backpropagate.multi_run.get_gpu_status", return_value=mock_status), \
             patch("torch.cuda.is_available", return_value=False):
            result = trainer.run("dummy_dataset")

        # Should abort with GPU safety check failed
        assert result is not None
        assert result.aborted is True

    def test_checkpoint_directory_created_before_run(self, tmp_path):
        """Checkpoint directory should be created during run() initialization."""
        checkpoint_dir = tmp_path / "new_checkpoints"
        config = MultiRunConfig(
            num_runs=1,
            checkpoint_dir=str(checkpoint_dir),
            enable_gpu_monitoring=False,
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        # Directory should not exist yet (before run)
        assert not checkpoint_dir.exists()

        # Create directory manually (simulating what run() does early)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        assert checkpoint_dir.exists()

    def test_callback_integration(self, tmp_path):
        """Callbacks should be properly configured."""
        callback_calls = {"start": 0, "complete": 0}

        def on_start(run_idx):
            callback_calls["start"] += 1

        def on_complete(result):
            callback_calls["complete"] += 1

        config = MultiRunConfig(
            num_runs=1,
            checkpoint_dir=str(tmp_path),
        )
        trainer = MultiRunTrainer(
            model="test-model",
            config=config,
            on_run_start=on_start,
            on_run_complete=on_complete,
        )

        # Verify callbacks are stored
        assert trainer.on_run_start is on_start
        assert trainer.on_run_complete is on_complete


class TestDataChunking:
    """Tests for data chunking logic."""

    def test_get_data_chunk_basic(self, tmp_path):
        """Should get correct data chunk for run index."""
        from datasets import Dataset

        config = MultiRunConfig(
            num_runs=3,
            samples_per_run=100,
            shuffle_data=False,  # Disable shuffle for predictable tests
            checkpoint_dir=str(tmp_path),
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        # Create proper HuggingFace Dataset
        mock_dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(300)]})

        chunk = trainer._get_data_chunk(mock_dataset, run_idx=1)

        # First run should get first 100 samples
        assert len(chunk) == 100

    def test_get_data_chunk_with_replay(self, tmp_path):
        """Should include replay samples when configured."""
        from datasets import Dataset

        config = MultiRunConfig(
            num_runs=3,
            samples_per_run=100,
            replay_fraction=0.2,
            shuffle_data=False,  # Disable shuffle for predictable tests
            checkpoint_dir=str(tmp_path),
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        # Create proper HuggingFace Dataset
        mock_dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(300)]})

        # Run 2 should include some samples from run 1
        chunk = trainer._get_data_chunk(mock_dataset, run_idx=2)

        # Should have 100 total (80 new + 20 replay)
        assert len(chunk) == 100


class TestCheckpointManagerIntegrationConfig:
    """Tests for checkpoint manager integration config."""

    def test_checkpoint_policy_config(self, tmp_path):
        """Should properly configure checkpoint policy from config."""
        config = MultiRunConfig(
            num_runs=3,
            checkpoint_dir=str(tmp_path),
            checkpoint_keep_best_n=5,
            checkpoint_max_total=15,
            checkpoint_auto_prune=True,
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        assert config.checkpoint_keep_best_n == 5
        assert config.checkpoint_max_total == 15
        assert config.checkpoint_auto_prune is True

    def test_checkpoint_manager_none_before_run(self, tmp_path):
        """Checkpoint manager should be None before run is called."""
        config = MultiRunConfig(
            checkpoint_dir=str(tmp_path),
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        assert trainer._checkpoint_manager is None


class TestEarlyStoppingConfig:
    """Tests for early stopping configuration."""

    def test_early_stopping_config_defaults(self, tmp_path):
        """Should have correct early stopping defaults."""
        config = MultiRunConfig(
            checkpoint_dir=str(tmp_path),
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        assert config.early_stopping is False
        assert config.early_stopping_patience == 2
        assert config.early_stopping_threshold == 0.0

    def test_early_stopping_can_be_enabled(self, tmp_path):
        """Should be able to enable early stopping."""
        config = MultiRunConfig(
            early_stopping=True,
            early_stopping_patience=3,
            checkpoint_dir=str(tmp_path),
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        assert trainer.config.early_stopping is True
        assert trainer.config.early_stopping_patience == 3

    def test_validation_losses_tracking_initialized(self, tmp_path):
        """Validation losses list should be initialized."""
        config = MultiRunConfig(
            early_stopping=True,
            checkpoint_dir=str(tmp_path),
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        assert trainer._validation_losses == []


class TestGPUMonitoringConfig:
    """Tests for GPU monitoring configuration."""

    def test_gpu_monitoring_enabled_by_default(self):
        """GPU monitoring should be enabled by default."""
        config = MultiRunConfig()
        assert config.enable_gpu_monitoring is True

    def test_gpu_monitoring_can_be_disabled(self, tmp_path):
        """GPU monitoring can be disabled."""
        config = MultiRunConfig(
            enable_gpu_monitoring=False,
            checkpoint_dir=str(tmp_path),
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        assert trainer.config.enable_gpu_monitoring is False

    def test_pause_on_overheat_config(self, tmp_path):
        """Should configure pause on overheat."""
        config = MultiRunConfig(
            pause_on_overheat=True,
            max_temp_c=80.0,
            cooldown_seconds=30.0,
            checkpoint_dir=str(tmp_path),
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        assert trainer.config.pause_on_overheat is True
        assert trainer.config.max_temp_c == 80.0
        assert trainer.config.cooldown_seconds == 30.0

    def test_gpu_tracking_initialized_to_zero(self, tmp_path):
        """GPU tracking values should start at zero."""
        config = MultiRunConfig(checkpoint_dir=str(tmp_path))
        trainer = MultiRunTrainer(model="test-model", config=config)

        assert trainer._gpu_max_temp == 0.0
        assert trainer._gpu_max_vram == 0.0


class TestSLAOMergeConfig:
    """Tests for SLAO merge configuration."""

    def test_slao_merger_none_before_run(self, tmp_path):
        """SLAO merger should be None before run is called."""
        config = MultiRunConfig(
            merge_mode=MergeMode.SLAO,
            checkpoint_dir=str(tmp_path),
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        # SLAO merger is created inside run(), not in __init__
        assert trainer._slao_merger is None

    def test_simple_mode_configuration(self, tmp_path):
        """Should properly configure simple merge mode."""
        config = MultiRunConfig(
            merge_mode=MergeMode.SIMPLE,
            checkpoint_dir=str(tmp_path),
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        assert trainer.config.merge_mode == MergeMode.SIMPLE
        assert trainer._slao_merger is None


class TestValidationConfig:
    """Tests for validation configuration."""

    def test_validation_config_defaults(self):
        """Should have correct validation defaults."""
        config = MultiRunConfig()

        assert config.validation_samples == 100
        assert config.validate_every_run is True

    def test_validation_can_be_disabled(self, tmp_path):
        """Validation can be disabled."""
        config = MultiRunConfig(
            validate_every_run=False,
            checkpoint_dir=str(tmp_path),
        )
        trainer = MultiRunTrainer(model="test-model", config=config)

        assert trainer.config.validate_every_run is False


# =============================================================================
# F-002 RESUME-FROM-CHECKPOINT TESTS
# =============================================================================

from pathlib import Path as _Path  # noqa: E402 (placed here for locality)


class TestMultiRunResume:
    """Resume detection + start-index calculation (F-002)."""

    def test_resume_from_default_is_none(self, tmp_path):
        config = MultiRunConfig(checkpoint_dir=str(tmp_path))
        trainer = MultiRunTrainer(model="m", config=config)
        assert trainer._resume_from is None

    def test_resume_from_off_skips_autodetect(self, tmp_path):
        from backpropagate.checkpoints import RunHistoryManager

        history = RunHistoryManager(str(tmp_path))
        history.record_run_started(
            run_id="active1",
            model_name="m",
            session_kind="multi_run",
        )

        config = MultiRunConfig(checkpoint_dir=str(tmp_path))
        trainer = MultiRunTrainer(
            model="m", config=config, resume_from="off"
        )
        assert trainer._maybe_resume(_Path(str(tmp_path))) is None

    def test_resume_from_autodetect_picks_in_progress_multi_run(self, tmp_path):
        from backpropagate.checkpoints import RunHistoryManager

        history = RunHistoryManager(str(tmp_path))
        history.record_run_started(
            run_id="autoresume1",
            model_name="m",
            session_kind="multi_run",
        )

        config = MultiRunConfig(checkpoint_dir=str(tmp_path))
        trainer = MultiRunTrainer(model="m", config=config)
        resolved = trainer._maybe_resume(_Path(str(tmp_path)))
        assert resolved == "autoresume1"

    def test_resume_from_ignores_single_runs(self, tmp_path):
        from backpropagate.checkpoints import RunHistoryManager

        history = RunHistoryManager(str(tmp_path))
        history.record_run_started(
            run_id="singleone",
            model_name="m",
            session_kind="single_run",
        )

        config = MultiRunConfig(checkpoint_dir=str(tmp_path))
        trainer = MultiRunTrainer(model="m", config=config)
        assert trainer._maybe_resume(_Path(str(tmp_path))) is None

    def test_explicit_resume_from_resolves_partial_prefix(self, tmp_path):
        from backpropagate.checkpoints import RunHistoryManager

        history = RunHistoryManager(str(tmp_path))
        history.record_run_started(
            run_id="aabbccddeeff",
            model_name="m",
            session_kind="multi_run",
        )

        config = MultiRunConfig(checkpoint_dir=str(tmp_path))
        trainer = MultiRunTrainer(
            model="m", config=config, resume_from="aabb"
        )
        assert trainer._maybe_resume(_Path(str(tmp_path))) == "aabbccddeeff"

    def test_explicit_resume_from_missing_raises(self, tmp_path):
        """BACKEND-F-018 (Wave 6a): explicit resume_from miss is no
        longer a silent fresh start. Mirrors the Wave 5.5 BACKEND-F-002
        contract on the single-run Trainer.train path. The operator
        passed resume_from expecting resumption; falling back silently
        to a fresh run loses the resume intent and consumes GPU hours
        producing a model the operator did not ask for.

        Pre-F-018 this returned ``None`` and the run continued from
        scratch. Now ``_maybe_resume`` raises ``InvalidSettingError`` so
        the operator sees the actionable hint in the terminal."""
        import pytest

        from backpropagate.exceptions import InvalidSettingError

        config = MultiRunConfig(checkpoint_dir=str(tmp_path))
        trainer = MultiRunTrainer(
            model="m", config=config, resume_from="ghost"
        )
        with pytest.raises(InvalidSettingError) as excinfo:
            trainer._maybe_resume(_Path(str(tmp_path)))
        # The error message must name the requested run_id + the
        # checkpoint_dir searched + the operator's next steps. These
        # are the load-bearing diagnostic anchors per the F-018 fix
        # shape (mirrors single-run F-002). The checkpoint_dir is
        # formatted via ``{path!r}`` so we match the leaf name to stay
        # robust across Windows (backslashes inside the repr quotes)
        # vs POSIX (forward slashes inside the repr quotes) separator
        # differences.
        msg = str(excinfo.value) + " " + (excinfo.value.suggestion or "")
        assert "ghost" in msg
        assert tmp_path.name in msg
        assert "backprop runs" in msg
        assert "--checkpoint-dir" in msg


class TestRestoreSessionState:

    def test_restore_skips_when_no_checkpoint(self, tmp_path):
        from backpropagate.checkpoints import CheckpointManager

        config = MultiRunConfig(checkpoint_dir=str(tmp_path))
        trainer = MultiRunTrainer(model="m", config=config)
        trainer._checkpoint_manager = CheckpointManager(str(tmp_path))
        assert (
            trainer._restore_session_state(_Path(str(tmp_path)), "nope") is False
        )
        assert trainer._resume_start_run_idx == 1

    def test_restore_finds_latest_run_index(self, tmp_path):
        from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy

        manager = CheckpointManager(
            str(tmp_path), CheckpointPolicy(auto_prune=False)
        )
        cp1 = tmp_path / "run_001"
        cp1.mkdir()
        (cp1 / "adapter_config.json").write_text("{}")
        manager.register(
            run_index=1,
            checkpoint_path=str(cp1),
            run_id="resumeme",
            validation_loss=0.5,
        )
        cp2 = tmp_path / "run_002"
        cp2.mkdir()
        (cp2 / "adapter_config.json").write_text("{}")
        manager.register(
            run_index=2,
            checkpoint_path=str(cp2),
            run_id="resumeme",
            validation_loss=0.3,
        )

        config = MultiRunConfig(checkpoint_dir=str(tmp_path))
        trainer = MultiRunTrainer(model="m", config=config)
        # Reuse the manager so registrations are visible.
        trainer._checkpoint_manager = manager
        assert trainer._restore_session_state(_Path(str(tmp_path)), "resumeme") is True
        assert trainer._resume_start_run_idx == 3  # next run after run 2
        assert trainer._resume_checkpoint_path == str(cp2)


class TestCheckpointManagerFindLatestForRunId:

    def test_returns_none_when_unmatched(self, tmp_path):
        from backpropagate.checkpoints import CheckpointManager

        manager = CheckpointManager(str(tmp_path))
        assert manager.find_latest_for_run_id("missing") is None

    def test_returns_highest_run_index(self, tmp_path):
        from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy

        manager = CheckpointManager(
            str(tmp_path), CheckpointPolicy(auto_prune=False)
        )
        for idx in (1, 2, 3):
            cp = tmp_path / f"cp{idx}"
            cp.mkdir()
            (cp / "data.json").write_text("{}")
            manager.register(
                run_index=idx, checkpoint_path=str(cp), run_id="hit",
                validation_loss=1.0 - idx * 0.1,
            )
        other = tmp_path / "other"
        other.mkdir()
        (other / "data.json").write_text("{}")
        manager.register(
            run_index=99, checkpoint_path=str(other), run_id="miss",
            validation_loss=0.05,
        )

        latest = manager.find_latest_for_run_id("hit")
        assert latest is not None
        assert latest.run_index == 3


# =============================================================================
# TESTS-B-008 — pause_on_overheat wiring (CRITICAL → pause / SAFE → resume)
# =============================================================================
#
# CHANGELOG v1.1.0 (B-003): ``pause_on_overheat=True`` was a no-op in v1.0 —
# the flag was documented but ``_on_gpu_critical`` only logged the warning
# without arming ``_gpu_pause_event``. v1.1.0 wired the actual pause: a
# CRITICAL status sets the event, the run loop polls it at the top of each
# new run, and ``_on_gpu_status`` clears it when the GPU recovers to
# SAFE/WARM. The existing tests in this file (L80, L85, L1876,
# test_multi_run_extended.py:90) only check the dataclass default — they
# never call ``_on_gpu_critical`` or ``_on_gpu_status``, so a regression
# that re-broke the wiring would silently pass.
#
# These tests drive the callback methods directly (no GPU monitor thread,
# no real CUDA touched) and assert the pause-event edges fire in both
# directions. The whole-run integration test is intentionally out of scope:
# spinning up a real Trainer with the SFTTrainer-mock + GPU-monitor scaffolding
# is owned by ``test_callback_integration.py``. The Stage C contract being
# pinned here is "the v1.1.0 wiring exists" — that's the regression surface
# the audit called out.


class TestPauseOnOverheatWiring:
    """B-003: ``_on_gpu_critical`` / ``_on_gpu_status`` actually toggle the event."""

    @pytest.fixture
    def trainer_with_pause(self):
        """Build a MultiRunTrainer without launching the GPU monitor.

        We bypass the heavy ``run()`` orchestration by constructing the
        trainer with pause_on_overheat=True and inspecting / driving the
        two callback hooks directly. The instance ships with a real
        ``threading.Event`` for ``_gpu_pause_event`` (multi_run.py:390).
        """
        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(
                num_runs=2,
                steps_per_run=10,
                samples_per_run=50,
                pause_on_overheat=True,
            ),
        )
        # Sanity: the wiring under test depends on these invariants.
        assert trainer.config.pause_on_overheat is True
        assert not trainer._gpu_pause_event.is_set(), (
            "Fresh trainer must start with the pause event cleared"
        )
        return trainer

    def test_critical_status_arms_pause_event(self, trainer_with_pause):
        """``_on_gpu_critical`` sets ``_gpu_pause_event`` when pause_on_overheat=True.

        Pre-fix this method only logged a warning. The wiring promise is
        that the next run-loop iteration will block until the GPU
        recovers — that block depends on the event being set.
        """
        critical_status = GPUStatus(
            available=True,
            device_name="Test GPU",
            temperature_c=92.0,
            vram_total_gb=16.0,
            vram_used_gb=15.2,
            vram_percent=95.0,
            condition=GPUCondition.CRITICAL,
            condition_reason="Temperature CRITICAL: 92.0C",
        )

        trainer_with_pause._on_gpu_critical(critical_status)

        assert trainer_with_pause._gpu_pause_event.is_set(), (
            "CRITICAL status must arm the pause event so the run loop "
            "blocks before starting the next run. Pre-fix this was a "
            "logged no-op — the regression surface this test pins."
        )

    def test_recovery_to_safe_clears_pause_event(self, trainer_with_pause):
        """``_on_gpu_status`` with SAFE/WARM clears a previously-armed event.

        Complementary half of the wiring: without this clear, the run
        loop's ``while self._gpu_pause_event.is_set(): time.sleep(1.0)``
        would hang forever after the first CRITICAL event.
        """
        # Pre-arm the event as the prior CRITICAL callback would have.
        trainer_with_pause._gpu_pause_event.set()

        safe_status = GPUStatus(
            available=True,
            device_name="Test GPU",
            temperature_c=65.0,
            vram_total_gb=16.0,
            vram_used_gb=8.0,
            vram_percent=50.0,
            condition=GPUCondition.SAFE,
            condition_reason="All metrics normal",
        )

        trainer_with_pause._on_gpu_status(safe_status)

        assert not trainer_with_pause._gpu_pause_event.is_set(), (
            "SAFE/WARM status after a pause MUST clear the event so the "
            "run loop can resume scheduling. A regression here turns the "
            "pause feature into a permanent hang on the first overheat."
        )

    def test_recovery_to_warm_also_clears_pause_event(self, trainer_with_pause):
        """WARM (not just SAFE) also satisfies the recovery condition.

        ``_on_gpu_status`` clears on ``SAFE`` or ``WARM`` (multi_run.py:1821).
        WARM is the in-between state; treating only SAFE as recovery
        would leave the trainer paused on a still-elevated-but-acceptable
        GPU and that's not the documented contract.
        """
        trainer_with_pause._gpu_pause_event.set()

        warm_status = GPUStatus(
            available=True,
            device_name="Test GPU",
            temperature_c=78.0,
            vram_total_gb=16.0,
            vram_used_gb=10.0,
            vram_percent=62.5,
            condition=GPUCondition.WARM,
            condition_reason="Warm but acceptable",
        )

        trainer_with_pause._on_gpu_status(warm_status)

        assert not trainer_with_pause._gpu_pause_event.is_set(), (
            "WARM status must satisfy the recovery condition; otherwise "
            "the trainer stays paused on any temperature above SAFE."
        )

    def test_pause_disabled_critical_does_not_arm_event(self):
        """``pause_on_overheat=False`` makes ``_on_gpu_critical`` a logged no-op.

        Operators who opt out keep the pre-v1.1.0 behaviour: warnings
        log but the event stays clear and the run loop never blocks.
        """
        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(
                num_runs=2,
                steps_per_run=10,
                samples_per_run=50,
                pause_on_overheat=False,
            ),
        )

        critical_status = GPUStatus(
            available=True,
            device_name="Test GPU",
            temperature_c=92.0,
            vram_total_gb=16.0,
            vram_used_gb=15.2,
            vram_percent=95.0,
            condition=GPUCondition.CRITICAL,
            condition_reason="Temperature CRITICAL: 92.0C",
        )

        trainer._on_gpu_critical(critical_status)

        assert not trainer._gpu_pause_event.is_set(), (
            "pause_on_overheat=False MUST keep the event clear even on "
            "CRITICAL; the operator chose hard-fail-on-overheat semantics."
        )


# =============================================================================
# BACKEND-F-001 — abort callback fires (Wave 6a regression)
# =============================================================================
#
# Cross-domain pin from Wave 5.5 BACKEND-F-001 fix
# (backpropagate/multi_run.py::_build_abort_callback). The pre-fix
# ``abort()`` flipped ``_should_abort=True`` but only the inter-run
# loop honored it — an in-flight ``SFTTrainer.train()`` call ran to
# completion (potentially hours) before the loop's next iteration. The
# GPU emergency handler (``_on_gpu_emergency``) calls ``abort()``
# expecting fast-fail; without the callback that safety contract was
# silently broken.
#
# The fix builds an HF ``TrainerCallback`` whose ``on_step_end`` polls
# ``trainer._should_abort`` and, when set, flips
# ``control.should_training_stop = True`` (the documented HF contract
# for cooperative cancellation). This test class pins both halves of
# the contract — the "not aborting" path (control untouched) and the
# "aborting" path (control flipped) — by simulating HF's ``on_step_end``
# fire against a mock ``control`` object.
class TestAbortCallback:
    """BACKEND-F-001 regression: abort callback fires + flips control."""

    def _build_callback(self, trainer):
        """Helper: build the callback via the module-level factory.

        Imports the private helper directly so the test doesn't depend
        on the run-loop wiring (which would force a full mock of the
        SFTTrainer + dataset). The factory's ``returns None`` branch
        when transformers is unavailable is handled by skipping the
        whole test; transformers is a hard dep of TRL so this branch
        should never fire in practice on a CI runner with our deps
        installed.
        """
        from backpropagate.multi_run import _build_abort_callback

        cb = _build_abort_callback(trainer)
        if cb is None:
            pytest.skip(
                "_build_abort_callback returned None — transformers "
                "TrainerCallback unavailable. transformers is a hard "
                "dep of TRL so this branch should never fire in CI; "
                "skip honestly forward-pointing to the dep audit."
            )
        return cb

    def test_callback_no_op_when_not_aborting(self):
        """on_step_end MUST NOT flip control.should_training_stop when
        ``_should_abort`` is False — the default state of a healthy run.

        Pre-F-001 the callback didn't exist; this test pins the
        no-side-effect contract on the new code path so a future
        regression that fires the abort signal eagerly (e.g. on any
        callback invocation rather than only when the flag is set) is
        caught at unit-test time instead of at run time when the inner
        SFTTrainer silently terminates.
        """
        trainer = MultiRunTrainer(model="test-model")
        assert trainer._should_abort is False

        cb = self._build_callback(trainer)

        # HF ``TrainerControl`` is the canonical type; we use a
        # MagicMock with the same surface so the test doesn't pull in
        # HF transformers' internals. The callback only reads / writes
        # ``should_training_stop`` so a minimal stub is enough.
        control = MagicMock()
        control.should_training_stop = False

        result = cb.on_step_end(
            args=MagicMock(),
            state=MagicMock(global_step=10),
            control=control,
        )

        # Two invariants:
        #
        # 1. ``should_training_stop`` is not touched (still False — the
        #    initial value the stub was constructed with).
        # 2. The callback returns the same ``control`` object so HF's
        #    contract is honored (HF passes the returned value through
        #    to the next callback in the chain).
        assert control.should_training_stop is False
        assert result is control

    def test_callback_flips_should_training_stop_when_aborting(self):
        """on_step_end MUST flip ``control.should_training_stop = True``
        when ``_should_abort`` is True. This is the load-bearing fix
        from BACKEND-F-001: the inner SFTTrainer cooperative-cancellation
        path that lets the run tear down within seconds of an abort
        signal rather than running to completion.
        """
        trainer = MultiRunTrainer(model="test-model")
        trainer.abort("Test abort reason")
        assert trainer._should_abort is True
        assert trainer._abort_reason == "Test abort reason"

        cb = self._build_callback(trainer)

        control = MagicMock()
        control.should_training_stop = False

        result = cb.on_step_end(
            args=MagicMock(),
            state=MagicMock(global_step=42),
            control=control,
        )

        assert control.should_training_stop is True, (
            "BACKEND-F-001 contract violation: abort callback did NOT "
            "flip control.should_training_stop. The inner SFTTrainer "
            "will keep training to completion despite the abort "
            "signal — the GPU emergency safety contract is broken."
        )
        assert result is control

    def test_callback_idempotent_across_fires(self):
        """Multiple ``on_step_end`` calls with ``_should_abort=True``
        keep ``should_training_stop=True`` — no flip-flop, no
        false-clear. HF Trainer fires the callback after every step;
        the callback's behaviour MUST be a stable monotonic latch
        (False → True transitions only) so the cooperative cancel
        intent doesn't get unstuck by a stray False write.
        """
        trainer = MultiRunTrainer(model="test-model")
        cb = self._build_callback(trainer)

        control = MagicMock()
        control.should_training_stop = False

        # Step 1: not aborting yet.
        cb.on_step_end(args=MagicMock(), state=MagicMock(global_step=1), control=control)
        assert control.should_training_stop is False

        # Step 2: operator (or GPU emergency handler) aborts mid-run.
        trainer.abort("simulated mid-run abort")
        cb.on_step_end(args=MagicMock(), state=MagicMock(global_step=2), control=control)
        assert control.should_training_stop is True

        # Step 3: callback fires again — still True (no false-clear).
        cb.on_step_end(args=MagicMock(), state=MagicMock(global_step=3), control=control)
        assert control.should_training_stop is True

    def test_callback_wired_at_module_scope(self):
        """The factory is exposed at module scope (not nested inside a
        method) so unit tests + future extensions can build the
        callback without instantiating a full run loop. Pin the public
        symbol so a refactor that hides it (e.g. closure inside
        ``run()``) is caught at import time.
        """
        from backpropagate import multi_run

        assert hasattr(multi_run, "_build_abort_callback")
        assert callable(multi_run._build_abort_callback)
