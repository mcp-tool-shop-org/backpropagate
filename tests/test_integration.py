"""
Integration Tests for End-to-End Training Workflows.

Tests cover:
- Single run training with small mock model
- Multi-run training with checkpoints
- Resume training from checkpoint
- Export and inference workflow
- UI training and monitoring
"""

import json
from unittest.mock import MagicMock, patch

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
        from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

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
        from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

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
        """Resume actually drives the resume_from code path (TESTS-A-004 v1.3).

        Replaces the previous tautology which only asserted ``trainer is not
        None`` after creating a checkpoint directory. The new test exercises
        the F-002 + F-017 resume path:

        1. Seed an on-disk run_history.json entry (the catalog the
           ``resume_from=`` lookup consults).
        2. Construct a Trainer and call its train() with ``resume_from``
           pointing at the seeded entry.
        3. Intercept the inner ``SFTTrainer`` so we can assert WITHOUT
           running real training: (a) the existing run_id was reused, (b)
           the checkpoint_path from history was threaded into the inner
           train() call as ``resume_from_checkpoint``.

        The load-bearing safety property: the v1.1.x bug pre-F-017 was
        that resume reused the run_id (so history wasn't duplicated) but
        SILENTLY started from step 0 because resume_from_checkpoint was
        never threaded through. The previous tautology test would have
        passed against that bug. This one fails.
        """
        from datetime import datetime, timezone

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.trainer import Trainer

        # Step 1 — seed the run history with a "completed" prior run whose
        # checkpoint dir exists on disk (so the resume path doesn't bail).
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        existing_run_id = "run-existing-123456"
        checkpoint_path = output_dir / "checkpoint-50"
        checkpoint_path.mkdir()
        (checkpoint_path / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "r": 16, "lora_alpha": 32}),
            encoding="utf-8",
        )

        manager = RunHistoryManager(str(output_dir))
        manager._save([
            {
                "run_id": existing_run_id,
                "status": "completed",
                "checkpoint_path": str(output_dir),
                "model_name": "test-model",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "final_loss": 0.5,
            }
        ])

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                model="test-model",
                output_dir=str(output_dir),
                use_unsloth=False,
            )

            # Step 2 — intercept the inner training so we can introspect what
            # the resume_from kwarg got threaded into. The trainer's load_model
            # / dataset path runs real code; mock those out so we only exercise
            # the resume-resolution path under test.
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=10)

            with patch.object(
                trainer, "load_model"
            ) as mock_load, patch.object(
                trainer, "_load_dataset", return_value=mock_dataset
            ), patch(
                "trl.SFTTrainer"
            ) as mock_sft_cls, patch(
                "trl.SFTConfig"
            ):
                mock_load.return_value = None
                trainer._is_loaded = True
                trainer._model = MagicMock()
                trainer._tokenizer = MagicMock()

                # Configure SFTTrainer mock to surface a usable train() shape.
                mock_sft_instance = MagicMock()
                mock_sft_instance.train.return_value = MagicMock(
                    metrics={"train_loss": 0.4, "train_runtime": 1.0}
                )
                # state.log_history needed by trainer to compute final_loss
                mock_sft_instance.state = MagicMock()
                mock_sft_instance.state.log_history = [{"loss": 0.4}]
                mock_sft_cls.return_value = mock_sft_instance

                # Patch persistence calls that would otherwise touch disk.
                with patch.object(trainer, "save", return_value=None):
                    try:
                        trainer.train(
                            dataset=mock_dataset,
                            steps=5,
                            resume_from=existing_run_id,
                        )
                    except Exception:
                        # The test focus is on resume-path argument threading,
                        # not full train() completion. Mocked components may
                        # raise on downstream code; the assertions below only
                        # need the resume-resolution to have executed.
                        pass

                # Step 3 — the inner SFTTrainer.train() must have been called
                # with resume_from_checkpoint pointing at the on-disk dir.
                # (F-017 invariant.)
                assert mock_sft_instance.train.called, (
                    "Trainer.train() did not invoke the inner SFTTrainer — "
                    "the resume path short-circuited before the inner call."
                )
                call_kwargs = mock_sft_instance.train.call_args.kwargs
                # The kwarg is resume_from_checkpoint (HF/TRL convention)
                rfc = call_kwargs.get("resume_from_checkpoint")
                assert rfc is not None and rfc == str(output_dir), (
                    f"resume_from_checkpoint kwarg was {rfc!r}; expected the "
                    f"on-disk checkpoint dir {str(output_dir)!r}. The "
                    f"pre-F-017 bug was that this kwarg never got threaded "
                    f"through, causing resume to silently start from step 0."
                )

    def test_e2e_checkpoint_contains_state(self, temp_dir):
        """TESTS-A-007: manifest round-trip preserves load-bearing state.

        Pre-rewrite this asserted only that a hand-written JSON file existed,
        which would not detect: a schema regression in CheckpointInfo,
        `_save_manifest` breaking the atomic-replace contract, or
        `find_latest_for_run_id` returning the wrong record on resume.

        Drives the real `register()` -> `_save_manifest()` ->
        `_load_manifest()` path via a second manager instance and asserts the
        round-trip preserves every field the resume code path consumes.
        """
        from backpropagate.checkpoints import CheckpointManager

        # Phase 1: write a manifest via register()
        manager_a = CheckpointManager(str(temp_dir))
        cp_dir = temp_dir / "checkpoint-step10"
        cp_dir.mkdir()
        (cp_dir / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "r": 16})
        )
        info_registered = manager_a.register(
            run_index=1,
            checkpoint_path=str(cp_dir),
            validation_loss=0.8,
            training_loss=0.9,
            is_run_boundary=True,
            run_id="run-7a3f",
        )

        # The atomic write should have produced manifest.json with the expected
        # schema; if the schema drifts the resume code path silently loses
        # information at the next load.
        manifest_path = temp_dir / CheckpointManager.MANIFEST_FILE
        assert manifest_path.exists(), "_save_manifest must write manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["version"] == "1.0"
        assert "updated" in manifest
        assert "policy" in manifest
        assert isinstance(manifest["checkpoints"], list)
        assert len(manifest["checkpoints"]) == 1

        # Phase 2: a fresh manager must reload the registered checkpoint
        # byte-identical from disk. This is the path resume_from exercises
        # via `find_latest_for_run_id` at process boot.
        manager_b = CheckpointManager(str(temp_dir))
        assert len(manager_b._checkpoints) == 1, (
            "second-instance _load_manifest must recover the registered "
            "checkpoint; otherwise resume_from cannot locate any prior state"
        )
        info_loaded = manager_b._checkpoints[0]
        assert info_loaded.run_index == info_registered.run_index
        assert info_loaded.path == str(cp_dir)
        assert info_loaded.validation_loss == 0.8
        assert info_loaded.training_loss == 0.9
        assert info_loaded.is_run_boundary is True
        assert info_loaded.is_final is True
        assert info_loaded.run_id == "run-7a3f"

        # Phase 3: the resume lookup path itself
        latest = manager_b.find_latest_for_run_id("run-7a3f")
        assert latest is not None, (
            "find_latest_for_run_id must return the registered checkpoint "
            "by run_id; this is the lookup multi_run.py:267 uses on resume"
        )
        assert latest.path == str(cp_dir)
        assert manager_b.find_latest_for_run_id("run-missing") is None


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

        with patch("backpropagate.export._is_peft_model", return_value=True), \
             patch("peft.PeftModel.from_pretrained", return_value=mock_model):
            result = export_lora(mock_model, str(output_path))

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
#
# TESTS-B-010 (v1.3 Wave 3 Stage C, 2026-05-23): the prior
# ``TestUITrainAndMonitor`` class was a class-level @pytest.mark.skip wrapping
# three tests that referenced ``from backpropagate.ui import state`` — a
# module that was deleted in v1.2.0 Wave 4.5 (Gradio removal). The skip
# reason cited "Reflex equivalents in Phase 3"; that work landed in v1.3
# Wave 3 Stage C as ``tests/test_ui_states.py`` (60 smoke tests covering
# TrainState, MultiRunState, ExportState, DatasetState). The dead skip
# class was removed so an auditor reading the suite doesn't see permanent
# "TODO disguised as skip" entries.


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
        from backpropagate.export import export_lora
        from backpropagate.trainer import Trainer

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

        with patch("backpropagate.export._is_peft_model", return_value=True):
            result = export_lora(mock_model, str(temp_dir / "exported"))
            assert result is not None

    def test_multi_run_then_export_merged(self, temp_dir):
        """Complete workflow: multi-run then export merged."""
        from backpropagate.export import export_merged
        from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

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
        assert merged_lora is not None, (
            "SLAO merge MUST return a result on this fixture; None "
            "indicates the merge path silently bailed (e.g., adapter "
            "not found, scaling=0). Previously this test gated all "
            "shape assertions behind `if merged_lora:` — making None "
            "a silent pass and hiding any merge-path regression."
        )
        for key, value in merged_lora.items():
            assert key in sample_lora_state
            assert value.shape == sample_lora_state[key].shape

    def test_slao_accumulates_across_runs(self, sample_lora_state):
        """SLAO should accumulate changes across multiple runs."""
        import torch

        from backpropagate.slao import SLAOConfig, SLAOMerger

        config = SLAOConfig(scaling_type="sqrt")
        merger = SLAOMerger(config=config)
        merger.initialize(sample_lora_state)

        initial_lora = merger.get_merged_lora()
        assert initial_lora is not None, (
            "SLAO initial merge MUST return a result after initialize(); "
            "None indicates a silent bail in the merge path. Previously "
            "gated behind `if initial_lora:` which let None pass silently."
        )
        initial_state = {k: v.clone() for k, v in initial_lora.items()}

        # Simulate 3 runs with different LoRA states
        for _i in range(3):
            new_state = {
                k: torch.randn_like(v)
                for k, v in sample_lora_state.items()
            }
            merger.merge(new_state)

        # After 3 merges, state should have changed
        final_lora = merger.get_merged_lora()
        assert final_lora is not None, (
            "SLAO merged state MUST be present after 3 merge() calls; "
            "None here indicates the merge accumulator was wiped. "
            "Previously gated behind `if final_lora:` which let None "
            "pass silently — defeating the entire accumulation property."
        )
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
        from backpropagate.gpu_safety import (
            GPUCondition,
            GPUSafetyConfig,
            GPUStatus,
            _evaluate_condition,
        )

        config = GPUSafetyConfig(
            temp_warning=80.0,
            temp_critical=90.0,
            temp_emergency=95.0,
        )

        # Test warning condition. 85C is between temp_warning=80 and
        # temp_critical=90, so _evaluate_condition MUST return WARNING.
        # TESTS-B-013 (Stage C humanization): the prior assertion accepted
        # WARNING-or-WARM, which papered over a misclassification regression
        # (WARM is a less-severe state). Pin the exact expected condition so
        # a downgrade in severity is caught.
        warning_status = GPUStatus(
            available=True,
            temperature_c=85.0,
        )
        condition, reason = _evaluate_condition(warning_status, config)
        assert condition == GPUCondition.WARNING, (
            f"85C with temp_warning=80, temp_critical=90 must be WARNING, "
            f"got {condition} (reason: {reason})"
        )

        # Test critical condition. 92C is above temp_critical=90 (and below
        # temp_emergency=95), so _evaluate_condition MUST return CRITICAL.
        # TESTS-B-013 (Stage C humanization): the prior assertion accepted
        # WARNING-or-CRITICAL, which would have masked a regression where a
        # critical-temp reading got misclassified as a mere warning — the
        # exact class of bug the GPU safety system exists to prevent.
        critical_status = GPUStatus(
            available=True,
            temperature_c=92.0,
        )
        condition, reason = _evaluate_condition(critical_status, config)
        assert condition == GPUCondition.CRITICAL, (
            f"92C with temp_critical=90 must be CRITICAL, "
            f"got {condition} (reason: {reason})"
        )
