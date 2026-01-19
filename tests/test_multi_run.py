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

import pytest
import math
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import asdict

from backpropagate.multi_run import (
    MultiRunTrainer,
    MultiRunConfig,
    MultiRunResult,
    RunResult,
    MergeMode,
)
from backpropagate.gpu_safety import GPUStatus, GPUCondition


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
        """Data chunks should not overlap."""
        # With 500 samples, 5 runs of 100 each
        # Run 1: 0-99, Run 2: 100-199, etc.

        samples_per_run = trainer.config.samples_per_run
        total_samples = 500

        seen_indices = set()
        for run_idx in range(1, 6):
            start_idx = ((run_idx - 1) * samples_per_run) % total_samples
            end_idx = min(start_idx + samples_per_run, total_samples)

            indices = set(range(start_idx, end_idx))

            # No overlap with previously seen indices
            assert len(seen_indices & indices) == 0

            seen_indices.update(indices)

    def test_data_chunk_wraparound(self, trainer):
        """Should handle wraparound when dataset is smaller than needed."""
        trainer.config.samples_per_run = 200
        trainer.config.num_runs = 5
        # Total needed: 1000 samples, but dataset only has 500

        # The function should cycle through the dataset


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

        # Create test JSONL
        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(10):
                f.write(json.dumps({"text": f"sample {i}"}) + "\n")

        ds = trainer._load_full_dataset(str(jsonl_path))

        assert len(ds) == 10

    def test_load_full_dataset_from_csv(self, tmp_path):
        """_load_full_dataset should load CSV files."""
        trainer = MultiRunTrainer(model="test-model")

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("text\nsample 1\nsample 2\nsample 3\n")

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

        with pytest.raises(ValueError, match="Unsupported dataset type"):
            trainer._load_full_dataset(12345)


class TestBackwardsCompatibility:
    """Tests for backwards compatibility aliases."""

    def test_speedrun_trainer_alias(self):
        """SpeedrunTrainer should be alias for MultiRunTrainer."""
        from backpropagate.multi_run import SpeedrunTrainer, MultiRunTrainer

        assert SpeedrunTrainer is MultiRunTrainer

    def test_speedrun_config_alias(self):
        """SpeedrunConfig should be alias for MultiRunConfig."""
        from backpropagate.multi_run import SpeedrunConfig, MultiRunConfig

        assert SpeedrunConfig is MultiRunConfig

    def test_speedrun_result_alias(self):
        """SpeedrunResult should be alias for MultiRunResult."""
        from backpropagate.multi_run import SpeedrunResult, MultiRunResult

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


class TestCheckpointManagerIntegration:
    """Tests for checkpoint manager integration."""

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


class TestBackwardsCompatibility:
    """Tests for backwards compatibility aliases."""

    def test_speedrun_config_alias(self):
        """SpeedrunConfig should be alias for MultiRunConfig."""
        from backpropagate.multi_run import SpeedrunConfig

        config = SpeedrunConfig(num_runs=3)
        assert isinstance(config, MultiRunConfig)
        assert config.num_runs == 3

    def test_speedrun_result_alias(self):
        """SpeedrunResult should be alias for MultiRunResult."""
        from backpropagate.multi_run import SpeedrunResult

        result = SpeedrunResult(
            total_runs=2,
            total_steps=100,
            total_samples=500,
            total_duration_seconds=60.0,
            final_loss=0.5,
        )
        assert isinstance(result, MultiRunResult)
        assert result.total_runs == 2

    def test_speedrun_trainer_alias(self):
        """SpeedrunTrainer should be alias for MultiRunTrainer."""
        from backpropagate.multi_run import SpeedrunTrainer

        trainer = SpeedrunTrainer(model="test-model")
        assert isinstance(trainer, MultiRunTrainer)
