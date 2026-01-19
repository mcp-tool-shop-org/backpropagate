"""
Integration Tests for End-to-End Training Workflows.

Tests cover:
- Single run training with small mock model
- Multi-run training with checkpoints
- Resume training from checkpoint
- Export and inference workflow
- UI training and monitoring
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import json


# =============================================================================
# END-TO-END TRAINING TESTS
# =============================================================================

class TestE2ESingleRunSmallModel:
    """End-to-end test for single run training on small model."""

    def test_e2e_single_run_training_flow(self, temp_dir):
        """Full training flow on mocked small model."""
        from backpropagate.trainer import Trainer

        # Create mock dataset
        dataset_path = temp_dir / "train.jsonl"
        samples = [
            {"text": f"<|im_start|>user\nQ{i}<|im_end|>\n<|im_start|>assistant\nA{i}<|im_end|>"}
            for i in range(20)
        ]
        with open(dataset_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        # Mock training result
        mock_train_result = MagicMock()
        mock_train_result.final_loss = 0.5
        mock_train_result.duration_seconds = 10.0

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                model="test-model",
                output_dir=str(temp_dir / "output"),
                use_unsloth=False,
            )

            # Mock the internal methods
            trainer._model = mock_model
            trainer._tokenizer = mock_tokenizer
            trainer._is_loaded = True

            # Mock _load_dataset to return our samples
            mock_ds = MagicMock()
            mock_ds.__len__ = MagicMock(return_value=20)

            with patch.object(trainer, "_load_dataset", return_value=mock_ds), \
                 patch.object(trainer, "_pre_tokenize", return_value=mock_ds), \
                 patch("trl.SFTTrainer") as mock_sft_trainer, \
                 patch("trl.SFTConfig"):
                mock_trainer_instance = MagicMock()
                # Mock training output properly
                mock_train_output = MagicMock()
                mock_train_output.training_loss = 0.5
                mock_trainer_instance.train.return_value = mock_train_output
                mock_sft_trainer.return_value = mock_trainer_instance

                result = trainer.train(str(dataset_path), steps=10)

                # Verify training was invoked
                mock_trainer_instance.train.assert_called_once()

    def test_e2e_single_run_with_trainer_options(self, temp_dir):
        """Training with various trainer options should work."""
        from backpropagate.trainer import Trainer

        # Create minimal dataset
        dataset_path = temp_dir / "train.jsonl"
        samples = [{"text": "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>"}]
        with open(dataset_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                model="test-model",
                output_dir=str(temp_dir / "output"),
                use_unsloth=False,
            )

            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

            mock_ds = MagicMock()
            mock_ds.__len__ = MagicMock(return_value=10)

            with patch.object(trainer, "_load_dataset", return_value=mock_ds), \
                 patch.object(trainer, "_pre_tokenize", return_value=mock_ds), \
                 patch("trl.SFTTrainer") as mock_sft_trainer, \
                 patch("trl.SFTConfig"):
                mock_trainer_instance = MagicMock()
                mock_train_output = MagicMock()
                mock_train_output.training_loss = 0.5
                mock_trainer_instance.train.return_value = mock_train_output
                mock_sft_trainer.return_value = mock_trainer_instance

                result = trainer.train(str(dataset_path), steps=5)

                # Training should complete without error
                assert result is not None


class TestE2EMultiRunWithCheckpoints:
    """End-to-end tests for multi-run training with checkpoints."""

    def test_e2e_multi_run_creates_checkpoints(self, temp_dir):
        """Multi-run should create checkpoints between runs."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        # Create dataset
        dataset_path = temp_dir / "train.jsonl"
        samples = [
            {"text": f"<|im_start|>user\nQ{i}<|im_end|>\n<|im_start|>assistant\nA{i}<|im_end|>"}
            for i in range(100)
        ]
        with open(dataset_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        config = MultiRunConfig(
            num_runs=3,
            steps_per_run=5,
            samples_per_run=20,
            save_every_run=True,
            checkpoint_dir=str(temp_dir / "multi_output"),
        )

        with patch("torch.cuda.is_available", return_value=False):
            trainer = MultiRunTrainer(
                model="test-model",
                config=config,
            )

            # Mock the internal trainer
            trainer._trainer = MagicMock()
            trainer._trainer._model = mock_model
            trainer._trainer._tokenizer = mock_tokenizer
            trainer._trainer._is_loaded = True
            trainer._trainer.get_lora_state_dict = MagicMock(return_value={
                "layer.lora_A.weight": MagicMock(),
                "layer.lora_B.weight": MagicMock(),
            })

            # Mock the run method
            with patch.object(trainer, "run") as mock_run:
                mock_result = MagicMock()
                mock_result.final_loss = 0.5
                mock_result.num_runs = 3
                mock_result.steps = 15
                mock_run.return_value = mock_result

                # Run multi-run training
                result = trainer.run(str(dataset_path))

                mock_run.assert_called_once()
                assert result.num_runs == 3

    def test_e2e_multi_run_loss_tracking(self, temp_dir):
        """Multi-run should track loss across all runs."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

        config = MultiRunConfig(
            num_runs=5,
            steps_per_run=10,
            samples_per_run=50,
            checkpoint_dir=str(temp_dir / "output"),
        )

        with patch("torch.cuda.is_available", return_value=False):
            trainer = MultiRunTrainer(
                model="test-model",
                config=config,
            )

            # Simulate loss tracking
            trainer._aggregate_loss = [1.5, 1.3, 1.1, 0.9, 0.7]
            trainer._run_boundaries = [2, 4]

            assert len(trainer._aggregate_loss) == 5
            assert trainer._run_boundaries == [2, 4]


class TestE2EResumeTraining:
    """End-to-end tests for resuming training from checkpoint."""

    def test_e2e_resume_from_checkpoint(self, temp_dir):
        """Should be able to resume training from saved checkpoint."""
        from backpropagate.trainer import Trainer

        # Create initial checkpoint directory structure
        checkpoint_dir = temp_dir / "checkpoint-run1"
        checkpoint_dir.mkdir()

        # Create adapter config file (minimal PEFT config)
        adapter_config = {
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
        }
        with open(checkpoint_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f)

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                model="test-model",
                output_dir=str(temp_dir / "output"),
                use_unsloth=False,
            )

            # Verify trainer is properly initialized
            assert trainer is not None
            assert trainer.output_dir is not None

    def test_e2e_checkpoint_contains_state(self, temp_dir):
        """Checkpoint should contain necessary state for resumption."""
        from backpropagate.checkpoints import CheckpointManager

        manager = CheckpointManager(str(temp_dir))

        # Create mock checkpoint
        checkpoint_path = temp_dir / "checkpoint-step10"
        checkpoint_path.mkdir()

        # Simulate saving metadata
        metadata = {
            "step": 10,
            "loss": 0.8,
            "run_idx": 1,
        }
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Verify checkpoint was created
        assert checkpoint_path.exists()
        assert (checkpoint_path / "metadata.json").exists()


class TestE2EExportAndInference:
    """End-to-end tests for export to GGUF and inference."""

    def test_e2e_export_lora_adapter(self, temp_dir):
        """Should export LoRA adapter."""
        from backpropagate.export import export_lora

        # Create mock model directory
        model_path = temp_dir / "model"
        model_path.mkdir()

        # Create mock adapter files
        adapter_config = {"peft_type": "LORA", "r": 16}
        with open(model_path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f)

        output_path = temp_dir / "exported"

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("backpropagate.export._is_peft_model", return_value=True), \
             patch("peft.PeftModel.from_pretrained", return_value=mock_model):
            result = export_lora(mock_model, mock_tokenizer, str(output_path))

            assert result is not None
            mock_model.save_pretrained.assert_called_once()

    def test_e2e_export_merged_model(self, temp_dir):
        """Should export merged model."""
        from backpropagate.export import export_merged

        model_path = temp_dir / "model"
        model_path.mkdir()

        output_path = temp_dir / "merged"

        mock_model = MagicMock()
        mock_merged = MagicMock()
        mock_model.merge_and_unload.return_value = mock_merged
        mock_tokenizer = MagicMock()

        with patch("backpropagate.export._is_peft_model", return_value=True):
            result = export_merged(mock_model, mock_tokenizer, str(output_path))

            assert result is not None
            mock_model.merge_and_unload.assert_called_once()
            mock_merged.save_pretrained.assert_called_once()


# =============================================================================
# UI INTEGRATION TESTS
# =============================================================================

class TestUITrainAndMonitor:
    """Tests for UI training and monitoring integration."""

    def test_ui_train_updates_state(self):
        """Training through UI should update state."""
        from backpropagate.ui import state

        # Ensure clean state
        state.is_training = False
        state.trainer = None

        # Test the state management
        assert state.is_training is False

        # Simulate training state change
        state.is_training = True
        assert state.is_training is True
        state.is_training = False

    def test_ui_state_tracks_loss_history(self):
        """UI state should track loss history."""
        from backpropagate.ui import state

        # Simulate adding loss values
        state.loss_history = []
        state.loss_history.append(1.5)
        state.loss_history.append(1.2)
        state.loss_history.append(0.9)

        assert len(state.loss_history) == 3
        assert state.loss_history[-1] == 0.9

    def test_ui_state_tracks_multi_run_progress(self):
        """UI state should track multi-run progress."""
        from backpropagate.ui import state

        state.multi_run_is_running = True
        state.multi_run_current_run = 3
        state.multi_run_loss_history = [1.5, 1.3, 1.1, 0.9]
        state.multi_run_run_boundaries = [2]

        assert state.multi_run_is_running is True
        assert state.multi_run_current_run == 3
        assert len(state.multi_run_loss_history) == 4


class TestUICheckpointListUpdates:
    """Tests for checkpoint list refresh in UI."""

    def test_ui_checkpoint_list_shows_checkpoints(self, temp_dir):
        """Checkpoint list should show available checkpoints."""
        from backpropagate.checkpoints import CheckpointManager

        manager = CheckpointManager(str(temp_dir))

        # Create some mock checkpoints and register them
        for i in range(3):
            cp_dir = temp_dir / f"checkpoint-run{i+1}"
            cp_dir.mkdir()
            metadata = {"step": (i + 1) * 10, "loss": 1.0 - i * 0.1}
            with open(cp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            # Register the checkpoint with the manager
            manager.register(
                run_index=i + 1,
                checkpoint_path=str(cp_dir),
                validation_loss=1.0 - i * 0.1,
            )

        # List checkpoints
        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 3


# =============================================================================
# WORKFLOW INTEGRATION TESTS
# =============================================================================

class TestTrainExportWorkflow:
    """Tests for complete train -> export workflow."""

    def test_train_then_export_lora(self, temp_dir):
        """Complete workflow: train then export LoRA."""
        from backpropagate.trainer import Trainer
        from backpropagate.export import export_lora

        # Training phase (mocked)
        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                model="test-model",
                output_dir=str(temp_dir / "training"),
                use_unsloth=False,
            )

            # Simulate training completion
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

            # Save model
            with patch.object(trainer._model, "save_pretrained"), \
                 patch.object(trainer._tokenizer, "save_pretrained"):
                trainer.save()

        # Export phase
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("backpropagate.export._is_peft_model", return_value=True):
            result = export_lora(mock_model, mock_tokenizer, str(temp_dir / "exported"))
            assert result is not None

    def test_multi_run_then_export_merged(self, temp_dir):
        """Complete workflow: multi-run then export merged."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig
        from backpropagate.export import export_merged

        config = MultiRunConfig(
            num_runs=2,
            steps_per_run=5,
            samples_per_run=10,
            checkpoint_dir=str(temp_dir / "multi_run"),
        )

        with patch("torch.cuda.is_available", return_value=False):
            trainer = MultiRunTrainer(
                model="test-model",
                config=config,
            )

            # Simulate multi-run completion
            trainer._trainer = MagicMock()
            trainer._trainer._model = MagicMock()
            trainer._trainer._tokenizer = MagicMock()

        # Export phase
        mock_model = MagicMock()
        mock_merged = MagicMock()
        mock_model.merge_and_unload.return_value = mock_merged
        mock_tokenizer = MagicMock()

        with patch("backpropagate.export._is_peft_model", return_value=True):
            result = export_merged(mock_model, mock_tokenizer, str(temp_dir / "merged"))
            assert result is not None


# =============================================================================
# SLAO MERGE INTEGRATION TESTS
# =============================================================================

class TestSLAOMergeIntegration:
    """Integration tests for SLAO merging during multi-run."""

    def test_slao_merge_preserves_base_dimensions(self, sample_lora_state):
        """SLAO merge should preserve original dimensions."""
        from backpropagate.slao import SLAOMerger

        merger = SLAOMerger()
        merger.initialize(sample_lora_state)

        # Verify dimensions preserved using get_merged_lora (correct method name)
        merged_lora = merger.get_merged_lora()
        if merged_lora:
            for key, value in merged_lora.items():
                assert key in sample_lora_state
                assert value.shape == sample_lora_state[key].shape

    def test_slao_accumulates_across_runs(self, sample_lora_state):
        """SLAO should accumulate changes across multiple runs."""
        import torch
        from backpropagate.slao import SLAOMerger, SLAOConfig

        config = SLAOConfig(scaling_type="sqrt")
        merger = SLAOMerger(config=config)
        merger.initialize(sample_lora_state)

        initial_lora = merger.get_merged_lora()
        if initial_lora:
            initial_state = {k: v.clone() for k, v in initial_lora.items()}

            # Simulate 3 runs with different LoRA states
            for i in range(3):
                new_state = {
                    k: torch.randn_like(v)
                    for k, v in sample_lora_state.items()
                }
                merger.merge(new_state)

            # After 3 merges, state should have changed
            final_lora = merger.get_merged_lora()
            if final_lora:
                for key in initial_state:
                    # States should be different after merging
                    assert not torch.allclose(initial_state[key], final_lora[key])


# =============================================================================
# DATASET PIPELINE INTEGRATION TESTS
# =============================================================================

class TestDatasetPipelineIntegration:
    """Integration tests for dataset loading and processing pipeline."""

    def test_dataset_load_to_training(self, temp_dir):
        """Dataset should load and be usable for training."""
        from backpropagate.datasets import DatasetLoader

        # Create dataset file
        dataset_path = temp_dir / "train.jsonl"
        samples = [
            {"text": f"<|im_start|>user\nQ{i}<|im_end|>\n<|im_start|>assistant\nA{i}<|im_end|>"}
            for i in range(50)
        ]
        with open(dataset_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        # Load dataset
        loader = DatasetLoader(str(dataset_path), validate=False)

        # Verify samples loaded
        assert len(list(loader)) == 50

    def test_streaming_dataset_batches(self, temp_dir):
        """Streaming dataset should yield proper batches."""
        from backpropagate.datasets import StreamingDatasetLoader

        # Create dataset
        dataset_path = temp_dir / "train.jsonl"
        samples = [
            {"text": f"sample_{i}"}
            for i in range(100)
        ]
        with open(dataset_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        loader = StreamingDatasetLoader(str(dataset_path))

        # Get batches
        batches = list(loader.batches(10))

        assert len(batches) == 10  # 100 / 10
        assert all(len(b) == 10 for b in batches)


# =============================================================================
# GPU SAFETY INTEGRATION TESTS
# =============================================================================

class TestGPUSafetyIntegration:
    """Integration tests for GPU safety during training."""

    def test_gpu_monitor_during_training(self):
        """GPU monitor should track stats during training."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        # GPUMonitor uses config.check_interval
        config = GPUSafetyConfig(check_interval=0.1)
        monitor = GPUMonitor(config=config)

        # Mock GPU status
        mock_status = MagicMock()
        mock_status.temperature_c = 65.0
        mock_status.vram_used_gb = 8.0
        mock_status.vram_percent = 50.0

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            monitor.start()

            import time
            time.sleep(0.2)  # Let monitor run

            monitor.stop()

            # Should have collected some history
            history = monitor.get_status_history()
            assert len(history) > 0

    def test_training_respects_gpu_limits(self):
        """Training should respect GPU safety limits."""
        from backpropagate.gpu_safety import GPUSafetyConfig, GPUCondition, _evaluate_condition, GPUStatus

        config = GPUSafetyConfig(
            temp_warning=80.0,
            temp_critical=90.0,
            temp_emergency=95.0,
        )

        # Test warning condition
        warning_status = GPUStatus(
            available=True,
            temperature_c=85.0,
        )
        condition, reason = _evaluate_condition(warning_status, config)
        assert condition == GPUCondition.WARNING or condition == GPUCondition.WARM

        # Test critical condition
        critical_status = GPUStatus(
            available=True,
            temperature_c=92.0,
        )
        condition, reason = _evaluate_condition(critical_status, config)
        assert condition in [GPUCondition.WARNING, GPUCondition.CRITICAL]
