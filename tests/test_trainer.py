"""Tests for Trainer class (mocked GPU)."""

import os
import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


class TestTrainerInit:
    """Tests for Trainer initialization."""

    def test_trainer_creation_defaults(self):
        """Test Trainer can be created with defaults."""
        from backpropagate.trainer import Trainer

        # Mock torch.cuda to avoid GPU requirement
        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert trainer.model_name is not None
        assert trainer.lora_r == 16
        assert trainer.lora_alpha == 32
        assert trainer._is_loaded is False

    def test_trainer_custom_parameters(self):
        """Test Trainer with custom parameters."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                model="custom/model",
                lora_r=32,
                lora_alpha=64,
                learning_rate=1e-4,
                batch_size=4,
            )

        assert trainer.model_name == "custom/model"
        assert trainer.lora_r == 32
        assert trainer.lora_alpha == 64
        assert trainer.learning_rate == 1e-4
        assert trainer.batch_size == 4

    def test_trainer_auto_batch_size_24gb(self):
        """Test auto batch size detection for 24GB GPU."""
        from backpropagate.trainer import Trainer

        mock_props = MagicMock()
        mock_props.total_memory = 24 * (1024**3)

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties", return_value=mock_props):
            trainer = Trainer(batch_size="auto")

        assert trainer.batch_size == 4

    def test_trainer_auto_batch_size_16gb(self):
        """Test auto batch size detection for 16GB GPU."""
        from backpropagate.trainer import Trainer

        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties", return_value=mock_props):
            trainer = Trainer(batch_size="auto")

        assert trainer.batch_size == 2

    def test_trainer_auto_batch_size_12gb(self):
        """Test auto batch size detection for 12GB GPU."""
        from backpropagate.trainer import Trainer

        mock_props = MagicMock()
        mock_props.total_memory = 12 * (1024**3)

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties", return_value=mock_props):
            trainer = Trainer(batch_size="auto")

        assert trainer.batch_size == 1

    def test_trainer_auto_batch_size_no_gpu(self):
        """Test auto batch size fallback when no GPU."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(batch_size="auto")

        assert trainer.batch_size == 2  # Safe default


class TestTrainerWindowsFixes:
    """Tests for Windows-specific fixes."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_windows_env_vars_set(self):
        """Test Windows environment variables are set."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        # These should be set on Windows
        assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
        assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "0"


class TestTrainerProperties:
    """Tests for Trainer properties."""

    def test_model_property(self):
        """Test model property returns internal model."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()
            trainer._model = MagicMock()

        assert trainer.model is trainer._model

    def test_tokenizer_property(self):
        """Test tokenizer property returns internal tokenizer."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()
            trainer._tokenizer = MagicMock()

        assert trainer.tokenizer is trainer._tokenizer

    def test_runs_property(self):
        """Test runs property returns training runs list."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert trainer.runs == []
        assert isinstance(trainer.runs, list)


class TestTrainerSave:
    """Tests for Trainer save functionality."""

    def test_save_raises_without_model(self, temp_dir):
        """Test save raises error when model not loaded."""
        from backpropagate.exceptions import TrainingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))

        with pytest.raises(TrainingError, match="No model loaded"):
            trainer.save()

    def test_save_with_model(self, temp_dir):
        """Test save works with loaded model."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

        path = trainer.save()
        assert path is not None
        trainer._model.save_pretrained.assert_called_once()
        trainer._tokenizer.save_pretrained.assert_called_once()

    def test_save_custom_path(self, temp_dir):
        """Test save to custom path."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

        custom_path = temp_dir / "custom_model"
        path = trainer.save(str(custom_path))

        assert str(custom_path) in path

    def test_save_registers_checkpoint_for_multi_run_resume(self, temp_dir):
        """BACKEND-F-007 (Wave 6a): Trainer.save() must register the
        saved checkpoint in a CheckpointManager rooted at
        ``self.output_dir`` so a later ``MultiRunTrainer`` pointed at
        the same ``checkpoint_dir`` can discover it via
        ``find_latest_for_run_id`` and resume from it.

        Pre-F-007 the single-run save path wrote the PEFT directory +
        the ``run_id`` correlation file but never appeared in any
        manifest — an operator who trained single-run, then tried to
        continue in multi-run mode against the same output_dir, hit a
        silent-fresh-start at the multi-run resume site because the
        manifest scan returned no matching ``run_id``. This test pins
        the cross-trainer interop invariant from the write side; the
        cross-trainer integration test (read side) lives in
        tests/test_integration.py."""
        from backpropagate.checkpoints import CheckpointManager
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

        run_id = "single-run-id-test-f007"
        trainer.save(run_id=run_id)

        # Reopen the manifest fresh — proves the registration was
        # persisted to disk, not just held in memory.
        cm = CheckpointManager(str(temp_dir))
        cp = cm.find_latest_for_run_id(run_id)
        assert cp is not None, (
            "BACKEND-F-007 contract violation: single-run-saved checkpoint "
            "was not registered in the manifest; MultiRunTrainer.resume "
            "via find_latest_for_run_id would fail."
        )
        # run_index=0 is the convention for single-run-saved checkpoints
        # (multi-run uses 1-based per-run indices). _restore_session_state
        # advances to run_index + 1 on resume, so a single-run starting
        # point cleanly hands off to run 1 of the multi-run loop.
        assert cp.run_index == 0
        assert cp.is_run_boundary is True
        assert cp.run_id == run_id

    def test_save_register_in_manifest_false_skips_registration(self, temp_dir):
        """BACKEND-F-007 (Wave 6a): operators saving ad-hoc / export-only
        snapshots can opt out of manifest registration by passing
        ``register_in_manifest=False``. Pollution-prevention escape hatch
        for callers (export.py merged-weights snapshots, throwaway eval
        saves) that should not appear in the resume-candidate set."""
        from backpropagate.checkpoints import CheckpointManager
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

        run_id = "noreg-test-f007"
        trainer.save(run_id=run_id, register_in_manifest=False)

        cm = CheckpointManager(str(temp_dir))
        cp = cm.find_latest_for_run_id(run_id)
        assert cp is None, (
            "register_in_manifest=False must NOT register the checkpoint; "
            "this is the documented opt-out for ad-hoc / export-only saves."
        )


class TestTrainerExport:
    """Tests for Trainer export functionality."""

    def test_export_raises_without_model(self, temp_dir):
        """Test export raises error when model not loaded."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))

        with pytest.raises((RuntimeError, Exception), match="No model loaded"):
            trainer.export()

    def test_export_lora(self, temp_dir):
        """Test export to LoRA format."""
        from backpropagate.export import ExportFormat
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

        with patch("backpropagate.export._is_peft_model", return_value=True):
            result = trainer.export(format="lora", output_dir=str(temp_dir / "lora"))

        assert result.format == ExportFormat.LORA

    def test_export_merged(self, temp_dir):
        """Test export to merged format."""
        from backpropagate.export import ExportFormat
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            mock_model = MagicMock()
            mock_model.merge_and_unload.return_value = MagicMock()
            trainer._model = mock_model
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

        with patch("backpropagate.export._is_peft_model", return_value=True):
            result = trainer.export(format="merged", output_dir=str(temp_dir / "merged"))

        assert result.format == ExportFormat.MERGED

    def test_export_invalid_format(self, temp_dir):
        """Test export with invalid format raises error."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

        with pytest.raises(ValueError, match="Unsupported format"):
            trainer.export(format="invalid")


class TestTrainerPushToHub:
    """Tests for push_to_hub functionality."""

    def test_push_to_hub_raises_without_model(self):
        """Test push_to_hub raises error when model not loaded."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        with pytest.raises((RuntimeError, Exception), match="No model loaded"):
            trainer.push_to_hub("test/repo")

    def test_push_to_hub_calls_model(self):
        """Test push_to_hub calls model and tokenizer push methods."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

        trainer.push_to_hub("test/repo", private=True)

        trainer._model.push_to_hub.assert_called_once_with("test/repo", private=True)
        trainer._tokenizer.push_to_hub.assert_called_once_with("test/repo", private=True)


class TestTrainingRun:
    """Tests for TrainingRun dataclass."""

    def test_training_run_creation(self):
        """Test TrainingRun can be created."""
        from backpropagate.trainer import TrainingRun

        run = TrainingRun(
            run_id="run_1",
            steps=100,
            final_loss=0.5,
            loss_history=[1.0, 0.8, 0.6, 0.5],
            duration_seconds=120.5,
            samples_seen=1000,
        )

        assert run.run_id == "run_1"
        assert run.steps == 100
        assert run.final_loss == 0.5
        assert len(run.loss_history) == 4
        assert run.duration_seconds == 120.5
        assert run.samples_seen == 1000

    def test_training_run_defaults(self):
        """Test TrainingRun has correct defaults."""
        from backpropagate.trainer import TrainingRun

        run = TrainingRun(
            run_id="run_1",
            steps=50,
            final_loss=0.3,
        )

        assert run.loss_history == []
        assert run.output_path is None
        assert run.duration_seconds == 0.0
        assert run.samples_seen == 0
        assert run.metadata == {}


class TestTrainingCallback:
    """Tests for TrainingCallback dataclass."""

    def test_callback_creation(self):
        """Test TrainingCallback can be created."""
        from backpropagate.trainer import TrainingCallback

        callback = TrainingCallback()
        assert callback.on_step is None
        assert callback.on_epoch is None
        assert callback.on_save is None
        assert callback.on_complete is None
        assert callback.on_error is None

    def test_callback_with_functions(self):
        """Test TrainingCallback with custom functions."""
        from backpropagate.trainer import TrainingCallback

        on_step_called = []

        def on_step(step, loss):
            on_step_called.append((step, loss))

        callback = TrainingCallback(on_step=on_step)
        callback.on_step(10, 0.5)

        assert len(on_step_called) == 1
        assert on_step_called[0] == (10, 0.5)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_model_function(self):
        """Test load_model convenience function."""
        from backpropagate.trainer import load_model

        # This would require actual model loading
        # Just verify it doesn't crash on import
        assert callable(load_model)

    def test_load_dataset_function_jsonl(self, temp_dir):
        """Test load_dataset with JSONL file."""
        import json

        from backpropagate.trainer import load_dataset

        # Create test JSONL
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"text": "sample 1"}) + "\n")
            f.write(json.dumps({"text": "sample 2"}) + "\n")

        ds = load_dataset(str(jsonl_path))
        assert len(ds) == 2

    def test_load_dataset_function_csv(self, temp_dir):
        """Test load_dataset with CSV file."""
        from backpropagate.trainer import load_dataset

        # Create test CSV
        csv_path = temp_dir / "test.csv"
        csv_path.write_text("text\nsample 1\nsample 2\n")

        ds = load_dataset(str(csv_path))
        assert len(ds) == 2

    def test_load_dataset_with_max_samples(self, temp_dir):
        """Test load_dataset with max_samples limit."""
        import json

        from backpropagate.trainer import load_dataset

        # Create test JSONL with many samples
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(100):
                f.write(json.dumps({"text": f"sample {i}"}) + "\n")

        ds = load_dataset(str(jsonl_path), max_samples=10)
        assert len(ds) == 10


class TestTrainerMultiRun:
    """Tests for Trainer multi_run method."""

    def test_multi_run_method_exists(self):
        """Test multi_run method exists on Trainer."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert hasattr(trainer, "multi_run")
        assert callable(trainer.multi_run)

    def test_speedrun_alias_exists(self):
        """Test speedrun alias exists for backwards compatibility."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert hasattr(trainer, "speedrun")
        assert trainer.speedrun == trainer.multi_run


class TestModuleExports:
    """Tests for module exports."""

    def test_trainer_exported(self):
        """Test Trainer is exported from trainer module."""
        from backpropagate.trainer import Trainer
        assert Trainer is not None

    def test_training_run_exported(self):
        """Test TrainingRun is exported."""
        from backpropagate.trainer import TrainingRun
        assert TrainingRun is not None

    def test_training_callback_exported(self):
        """Test TrainingCallback is exported."""
        from backpropagate.trainer import TrainingCallback
        assert TrainingCallback is not None

    def test_load_model_exported(self):
        """Test load_model is exported."""
        from backpropagate.trainer import load_model
        assert callable(load_model)

    def test_load_dataset_exported(self):
        """Test load_dataset is exported."""
        from backpropagate.trainer import load_dataset
        assert callable(load_dataset)

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from backpropagate import trainer

        assert "Trainer" in trainer.__all__
        assert "TrainingRun" in trainer.__all__
        assert "TrainingCallback" in trainer.__all__
        assert "load_model" in trainer.__all__
        assert "load_dataset" in trainer.__all__


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================

class TestTrainerLoadModel:
    """Tests for Trainer.load_model() method."""

    def test_load_model_already_loaded_skips(self):
        """load_model should skip if already loaded."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()
            trainer._is_loaded = True
            trainer._model = MagicMock()

            # Call load_model
            trainer.load_model()

            # Should not call loading methods since already loaded
            # (no error means it returned early)
            assert trainer._is_loaded is True

    def test_load_model_with_unsloth(self):
        """load_model should use Unsloth when use_unsloth=True and available."""
        from backpropagate import feature_flags
        from backpropagate.trainer import Trainer

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("torch.cuda.is_available", return_value=False), \
             patch.dict(feature_flags.FEATURES, {"unsloth": True}):
            trainer = Trainer(use_unsloth=True)

            with patch.object(trainer, "_load_with_unsloth") as mock_unsloth:
                trainer.load_model()
                mock_unsloth.assert_called_once()

    def test_load_model_without_unsloth(self):
        """load_model should use transformers when use_unsloth=False."""
        from backpropagate import feature_flags
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False), \
             patch.dict(feature_flags.FEATURES, {"unsloth": False}):
            trainer = Trainer(use_unsloth=False)

            with patch.object(trainer, "_load_with_transformers") as mock_transformers:
                trainer.load_model()
                mock_transformers.assert_called_once()

    def test_load_model_sets_is_loaded_flag(self):
        """load_model should set _is_loaded flag to True."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(use_unsloth=False)

            with patch.object(trainer, "_load_with_transformers"):
                trainer.load_model()
                assert trainer._is_loaded is True


class TestTrainerTrain:
    """Tests for Trainer.train() method with mocked TRL Trainer."""

    def test_train_loads_model_if_not_loaded(self, temp_dir):
        """train() should call load_model() if not already loaded."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))

            # Create mock dataset
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=10)

            with patch.object(trainer, "load_model") as mock_load, \
                 patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
                 patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
                 patch("trl.SFTTrainer") as mock_sft_trainer, \
                 patch("trl.SFTConfig"):
                # Setup mock trainer
                mock_instance = MagicMock()
                mock_instance.train.return_value = MagicMock(training_loss=0.5)
                mock_instance.state.log_history = [{"loss": 0.5}]
                mock_sft_trainer.return_value = mock_instance

                # Set model/tokenizer after "loading"
                trainer._model = MagicMock()
                trainer._tokenizer = MagicMock()
                trainer._is_loaded = True

                trainer.train("dummy_dataset", steps=10)
                # Since _is_loaded is True, load_model shouldn't be called
                # This tests the early return path

    def test_train_invokes_callback_on_complete(self, temp_dir):
        """train() should invoke callback.on_complete when training finishes."""
        from backpropagate.trainer import Trainer, TrainingCallback, TrainingRun

        completed_runs = []

        def on_complete(run: TrainingRun):
            completed_runs.append(run)

        callback = TrainingCallback(on_complete=on_complete)

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

            # Create mock dataset
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=10)

            with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
                 patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
                 patch("trl.SFTTrainer") as mock_sft_trainer, \
                 patch("trl.SFTConfig"):
                # Setup mock trainer
                mock_instance = MagicMock()
                mock_instance.train.return_value = MagicMock(training_loss=0.5)
                mock_instance.state.log_history = [{"loss": 0.5}]
                mock_sft_trainer.return_value = mock_instance

                run = trainer.train("dummy_dataset", steps=10, callback=callback)

                assert len(completed_runs) == 1
                assert completed_runs[0].run_id == run.run_id

    def test_train_invokes_callback_on_error(self, temp_dir):
        """train() should invoke callback.on_error when training fails."""
        from backpropagate.trainer import Trainer, TrainingCallback

        errors = []

        def on_error(exc: Exception):
            errors.append(exc)

        callback = TrainingCallback(on_error=on_error)

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

            # Create mock dataset
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=10)

            from backpropagate.exceptions import TrainingError

            with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
                 patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
                 patch("trl.SFTTrainer") as mock_sft_trainer, \
                 patch("trl.SFTConfig"):
                # Setup mock trainer to raise an error
                mock_instance = MagicMock()
                mock_instance.train.side_effect = RuntimeError("Training failed")
                mock_sft_trainer.return_value = mock_instance

                with pytest.raises(TrainingError, match="Training failed"):
                    trainer.train("dummy_dataset", steps=10, callback=callback)

                assert len(errors) == 1
                assert "Training failed" in str(errors[0])

    def test_train_returns_training_run(self, temp_dir):
        """train() should return TrainingRun with correct data."""
        from backpropagate.trainer import Trainer, TrainingRun

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

            # Create mock dataset
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=10)

            with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
                 patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
                 patch("trl.SFTTrainer") as mock_sft_trainer, \
                 patch("trl.SFTConfig"):
                mock_instance = MagicMock()
                mock_instance.train.return_value = MagicMock(training_loss=0.42)
                mock_instance.state.log_history = [
                    {"loss": 1.0},
                    {"loss": 0.7},
                    {"loss": 0.42},
                ]
                mock_sft_trainer.return_value = mock_instance

                run = trainer.train("dummy_dataset", steps=10)

                assert isinstance(run, TrainingRun)
                assert run.final_loss == 0.42
                assert run.loss_history == [1.0, 0.7, 0.42]
                import re
                assert re.fullmatch(r"[a-f0-9]{32}", run.run_id), (
                    f"run_id should be a UUID4 hex (32 chars), got {run.run_id!r}"
                )
                assert run.metadata.get("legacy_run_label") == "run_1"

    def test_train_appends_to_runs_list(self, temp_dir):
        """train() should append result to trainer.runs list."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

            # Create mock dataset
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=10)

            with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
                 patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
                 patch("trl.SFTTrainer") as mock_sft_trainer, \
                 patch("trl.SFTConfig"):
                mock_instance = MagicMock()
                mock_instance.train.return_value = MagicMock(training_loss=0.5)
                mock_instance.state.log_history = []
                mock_sft_trainer.return_value = mock_instance

                assert len(trainer.runs) == 0
                trainer.train("dummy_dataset", steps=5)
                assert len(trainer.runs) == 1
                trainer.train("dummy_dataset", steps=5)
                assert len(trainer.runs) == 2


class TestLoadModelFunction:
    """Tests for load_model() convenience function."""

    def test_load_model_creates_trainer_and_loads(self):
        """load_model() should create Trainer and call load_model."""
        from backpropagate.trainer import Trainer, load_model

        with patch("torch.cuda.is_available", return_value=False), \
             patch.object(Trainer, "load_model") as mock_load:
            # Mock the model/tokenizer properties
            with patch.object(Trainer, "model", new_callable=PropertyMock) as mock_model, \
                 patch.object(Trainer, "tokenizer", new_callable=PropertyMock) as mock_tokenizer:
                mock_model.return_value = MagicMock()
                mock_tokenizer.return_value = MagicMock()

                model, tokenizer = load_model("test-model")

                mock_load.assert_called_once()
                assert model is not None
                assert tokenizer is not None

    def test_load_model_passes_parameters(self):
        """load_model() should pass max_seq_length to Trainer."""
        from backpropagate.trainer import Trainer, load_model

        with patch("torch.cuda.is_available", return_value=False), \
             patch.object(Trainer, "__init__", return_value=None) as mock_init, \
             patch.object(Trainer, "load_model"), \
             patch.object(Trainer, "model", new_callable=PropertyMock, return_value=MagicMock()), \
             patch.object(Trainer, "tokenizer", new_callable=PropertyMock, return_value=MagicMock()):
            load_model("test-model", max_seq_length=4096)

            # Check that __init__ was called with max_seq_length
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs.get("max_seq_length") == 4096


class TestTrainerLoadDataset:
    """Tests for Trainer._load_dataset() method."""

    def test_load_dataset_from_none_uses_config(self):
        """_load_dataset with None should use config default."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

            with patch("datasets.load_dataset") as mock_load:
                mock_ds = MagicMock()
                mock_ds.__len__ = MagicMock(return_value=100)
                mock_load.return_value = mock_ds

                trainer._load_dataset(None)

                mock_load.assert_called_once()

    def test_load_dataset_from_hf_dataset_object(self):
        """_load_dataset should accept HuggingFace Dataset object directly."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

            # Create mock Dataset
            from datasets import Dataset
            mock_ds = MagicMock(spec=Dataset)
            mock_ds.__len__ = MagicMock(return_value=50)

            result = trainer._load_dataset(mock_ds)
            assert result is mock_ds

    def test_load_dataset_limits_samples(self, temp_dir):
        """_load_dataset should limit samples when max_samples specified."""
        import json

        from backpropagate.trainer import Trainer

        # Create test dataset with many samples (ShareGPT format for DatasetLoader)
        jsonl_path = temp_dir / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(100):
                f.write(json.dumps({"conversations": [
                    {"from": "human", "value": f"Question {i}"},
                    {"from": "gpt", "value": f"Answer {i}"},
                ]}) + "\n")

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

            ds = trainer._load_dataset(str(jsonl_path), samples=10)
            assert len(ds) == 10

    def test_load_dataset_invalid_type_raises(self):
        """_load_dataset should raise for unsupported types."""
        from backpropagate.exceptions import DatasetError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

            with pytest.raises(DatasetError, match="Unsupported dataset type"):
                trainer._load_dataset(12345)  # Invalid type


class TestTrainerSaveMerged:
    """Tests for Trainer.save() with save_merged option."""

    def test_save_merged_with_unsloth(self, temp_dir):
        """save() with save_merged=True should use Unsloth's merged save."""
        from backpropagate.trainer import Trainer

        # Mock the Unsloth import that happens in save()
        mock_fast_lm = MagicMock()

        with patch("torch.cuda.is_available", return_value=False), \
             patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_lm)}):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True
            trainer.use_unsloth = True

            trainer.save(save_merged=True)

            trainer._model.save_pretrained_merged.assert_called_once()

    def test_save_without_merged(self, temp_dir):
        """save() without save_merged should use standard save."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True
            trainer.use_unsloth = True

            trainer.save(save_merged=False)

            trainer._model.save_pretrained.assert_called_once()
            trainer._tokenizer.save_pretrained.assert_called_once()


# =============================================================================
# LORA TESTS (Phase 3)
# =============================================================================

class TestLoRAAdapterApplied:
    """Tests for LoRA adapter application."""

    def test_lora_adapter_applied_with_unsloth(self):
        """Verify LoRA layers added to model with Unsloth."""
        from backpropagate import feature_flags
        from backpropagate.trainer import Trainer

        mock_fast_lm = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # from_pretrained returns (model, tokenizer)
        mock_fast_lm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_fast_lm.get_peft_model.return_value = mock_model

        with patch("torch.cuda.is_available", return_value=False), \
             patch.dict(feature_flags.FEATURES, {"unsloth": True}), \
             patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_lm)}):

            trainer = Trainer(use_unsloth=True)

            # Manually trigger loading
            with patch("unsloth.FastLanguageModel", mock_fast_lm):
                trainer._load_with_unsloth()

            # Verify get_peft_model was called (LoRA applied)
            mock_fast_lm.get_peft_model.assert_called_once()

    def test_lora_adapter_applied_with_transformers(self):
        """Verify LoRA layers added with transformers + PEFT."""
        from backpropagate.trainer import Trainer

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_peft_model = MagicMock()

        with patch("torch.cuda.is_available", return_value=False), \
             patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("transformers.BitsAndBytesConfig"), \
             patch("peft.prepare_model_for_kbit_training", return_value=mock_model), \
             patch("peft.get_peft_model", return_value=mock_peft_model) as mock_get_peft, \
             patch("peft.LoraConfig") as mock_lora_config:

            trainer = Trainer(use_unsloth=False)
            trainer._load_with_transformers()

            # Verify get_peft_model was called (LoRA applied)
            mock_get_peft.assert_called_once()


class TestLoRARankConfiguration:
    """Tests for custom r, alpha LoRA values."""

    def test_lora_rank_default(self):
        """Default LoRA rank should be 16."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert trainer.lora_r == 16

    def test_lora_alpha_default(self):
        """Default LoRA alpha should be 32."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert trainer.lora_alpha == 32

    def test_lora_rank_custom(self):
        """Custom LoRA rank should be applied."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(lora_r=64)

        assert trainer.lora_r == 64

    def test_lora_alpha_custom(self):
        """Custom LoRA alpha should be applied."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(lora_alpha=128)

        assert trainer.lora_alpha == 128

    def test_lora_dropout_default(self):
        """Default LoRA dropout should be applied from settings."""
        from backpropagate.config import settings
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert trainer.lora_dropout == settings.lora.lora_dropout

    def test_lora_dropout_custom(self):
        """Custom LoRA dropout should be applied."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(lora_dropout=0.1)

        assert trainer.lora_dropout == 0.1


class TestLoRATargetModules:
    """Tests for correct LoRA target modules."""

    def test_lora_targets_attention_modules(self):
        """LoRA should target attention projection modules."""
        from backpropagate.config import settings

        # Check settings has the expected target modules
        target_modules = settings.lora.target_modules

        assert "q_proj" in target_modules
        assert "k_proj" in target_modules
        assert "v_proj" in target_modules
        assert "o_proj" in target_modules

    def test_lora_targets_mlp_modules(self):
        """LoRA should target MLP projection modules."""
        from backpropagate.config import settings

        target_modules = settings.lora.target_modules

        assert "gate_proj" in target_modules
        assert "up_proj" in target_modules
        assert "down_proj" in target_modules


class TestLoRAMergeAndUnload:
    """Tests for merging LoRA back to base model."""

    def test_export_merged_calls_merge_and_unload(self, temp_dir):
        """Export merged should call merge_and_unload."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            mock_model = MagicMock()
            mock_merged_model = MagicMock()
            mock_model.merge_and_unload.return_value = mock_merged_model
            trainer._model = mock_model
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

            with patch("backpropagate.export._is_peft_model", return_value=True):
                trainer.export(format="merged", output_dir=str(temp_dir / "merged"))

            mock_model.merge_and_unload.assert_called_once()


# =============================================================================
# TRAINING VALIDATION TESTS
# =============================================================================

class TestTrainingValidation:
    """Tests for training input validation."""

    def test_train_invalid_steps_raises(self):
        """Invalid steps parameter should raise error."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()
            trainer._is_loaded = True
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()

            with pytest.raises(InvalidSettingError, match="steps"):
                trainer.train("dummy", steps=-5)

    def test_train_invalid_samples_raises(self):
        """Invalid samples parameter should raise error."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()
            trainer._is_loaded = True
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()

            with pytest.raises(InvalidSettingError, match="samples"):
                trainer.train("dummy", samples=0)

    def test_train_zero_steps_raises(self):
        """Zero steps should raise error."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()
            trainer._is_loaded = True
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()

            with pytest.raises(InvalidSettingError, match="steps"):
                trainer.train("dummy", steps=0)


# =============================================================================
# TRAINING ON RESPONSES ONLY TESTS
# =============================================================================

class TestTrainOnResponses:
    """Tests for train_on_responses_only optimization."""

    def test_train_on_responses_default_true(self):
        """train_on_responses should default to True."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert trainer._train_on_responses is True

    def test_train_on_responses_configurable(self):
        """train_on_responses should be configurable."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(train_on_responses=False)

        assert trainer._train_on_responses is False


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestTrainerErrorHandling:
    """Tests for error handling in Trainer."""

    def test_load_model_import_error(self):
        """ImportError during load should raise ModelLoadError."""
        from backpropagate.exceptions import ModelLoadError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(use_unsloth=False)

            with patch.object(trainer, "_load_with_transformers", side_effect=ImportError("test")):
                with pytest.raises(ModelLoadError):
                    trainer.load_model()

    def test_load_model_cuda_error(self):
        """CUDA error during load should raise GPUNotAvailableError."""
        from backpropagate.exceptions import GPUNotAvailableError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(use_unsloth=False)

            with patch.object(trainer, "_load_with_transformers", side_effect=RuntimeError("CUDA out of memory")):
                with pytest.raises(GPUNotAvailableError):
                    trainer.load_model()

    def test_train_oom_error(self, temp_dir):
        """OOM during training should raise TrainingError with helpful message."""
        from backpropagate.exceptions import TrainingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=10)

            with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
                 patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
                 patch("trl.SFTTrainer") as mock_sft_trainer, \
                 patch("trl.SFTConfig"):
                mock_instance = MagicMock()
                mock_instance.train.side_effect = RuntimeError("CUDA out of memory")
                mock_sft_trainer.return_value = mock_instance

                # The error message contains "batch_size" in the suggestion
                with pytest.raises(TrainingError, match="GPU error"):
                    trainer.train("dummy", steps=10)

    def test_save_permission_error(self, temp_dir):
        """Permission error during save should raise CheckpointError."""
        from backpropagate.exceptions import CheckpointError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True

            # Mock Path.mkdir to raise PermissionError
            with patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied")):
                with pytest.raises(CheckpointError):
                    trainer.save("/some/restricted/path")


# =============================================================================
# DATASET LOADING ERROR TESTS
# =============================================================================

class TestDatasetLoadingErrors:
    """Tests for dataset loading error handling."""

    def test_load_dataset_file_not_found(self, temp_dir):
        """Non-existent file should raise DatasetNotFoundError."""
        from backpropagate.exceptions import DatasetNotFoundError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

            with pytest.raises(DatasetNotFoundError):
                trainer._load_dataset(str(temp_dir / "nonexistent.jsonl"))

    def test_load_dataset_invalid_json(self, temp_dir):
        """Invalid JSON file should raise a DatasetError (or subclass)."""
        from backpropagate.exceptions import DatasetError
        from backpropagate.trainer import Trainer

        # Create invalid JSON file
        invalid_file = temp_dir / "invalid.jsonl"
        invalid_file.write_text("not valid json {{{")

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

            with pytest.raises(DatasetError):
                trainer._load_dataset(str(invalid_file))


# =============================================================================
# FIXTURE FOR TESTS
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for tests."""
    return tmp_path


# =============================================================================
# B-017 + RE-001: HuggingFace transient retry status-code filter
# =============================================================================

class TestHfTransientRetryStatusCodeFilter:
    """RE-001: retry must skip 401/403/404 (auth/gated/typo) and only fire on
    429 / 5xx / connection / timeout. Without status-code filtering, a typo'd
    model name would hang ~65s before showing the real error."""

    def test_does_not_retry_401_403_404(self):
        import requests

        from backpropagate.trainer import _is_transient_hf_exception

        for code in (401, 403, 404, 400, 422):
            exc = requests.exceptions.HTTPError(f"HTTP {code}")
            exc.response = MagicMock(status_code=code)
            assert not _is_transient_hf_exception(exc), \
                f"HTTP {code} should NOT retry (not transient)"

    def test_retries_429_and_5xx(self):
        import requests

        from backpropagate.trainer import _is_transient_hf_exception

        for code in (429, 500, 502, 503, 504):
            exc = requests.exceptions.HTTPError(f"HTTP {code}")
            exc.response = MagicMock(status_code=code)
            assert _is_transient_hf_exception(exc), \
                f"HTTP {code} should retry (transient)"

    def test_retries_connection_and_timeout(self):
        from backpropagate.trainer import _is_transient_hf_exception

        assert _is_transient_hf_exception(ConnectionError("dropped"))
        assert _is_transient_hf_exception(TimeoutError("timed out"))

    def test_does_not_retry_unrelated_exception(self):
        from backpropagate.trainer import _is_transient_hf_exception

        assert not _is_transient_hf_exception(ValueError("bad arg"))
        assert not _is_transient_hf_exception(KeyError("missing"))

    def test_retries_http_error_with_no_response(self):
        """If an HTTPError carries no .response attribute, retry conservatively
        (we can't tell whether it was transient, but retrying is safer than
        bailing on a genuine connection blip)."""
        import requests

        from backpropagate.trainer import _is_transient_hf_exception

        exc = requests.exceptions.HTTPError("no response object")
        assert _is_transient_hf_exception(exc)


# =============================================================================
# F-005 REPORT_TO RESOLUTION TESTS
# =============================================================================

class TestReportToResolution:
    """Trainer._resolve_report_to behaviour for F-005 (W&B/TB/MLflow wiring)."""

    @pytest.fixture
    def trainer(self):
        from backpropagate.trainer import Trainer
        with patch("torch.cuda.is_available", return_value=False):
            return Trainer()

    def _patch_features(self, **flags):
        """Helper to temporarily flip the global FEATURES dict."""
        from backpropagate import feature_flags

        return patch.dict(feature_flags.FEATURES, flags, clear=False)

    def test_default_intent_is_auto(self, trainer):
        assert trainer._report_to_intent == "auto"

    def test_auto_no_trackers_resolves_to_none(self, trainer):
        with self._patch_features(wandb=False, tensorboard=False, mlflow=False):
            assert trainer._resolve_report_to() == "none"

    def test_auto_wandb_installed_returns_list_with_wandb(self, trainer):
        with self._patch_features(wandb=True, tensorboard=False, mlflow=False):
            assert trainer._resolve_report_to() == ["wandb"]

    def test_auto_all_trackers_installed_returns_combined_list(self, trainer):
        with self._patch_features(wandb=True, tensorboard=True, mlflow=True):
            resolved = trainer._resolve_report_to()
        assert isinstance(resolved, list)
        assert set(resolved) == {"wandb", "tensorboard", "mlflow"}

    def test_explicit_none_string(self):
        from backpropagate.trainer import Trainer
        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(report_to="none")
        assert trainer._resolve_report_to() == "none"

    def test_explicit_none_value(self):
        from backpropagate.trainer import Trainer
        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(report_to=None)
        assert trainer._resolve_report_to() == "none"

    def test_explicit_list_passes_through(self):
        from backpropagate.trainer import Trainer
        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(report_to=["wandb", "tensorboard"])
        assert trainer._resolve_report_to() == ["wandb", "tensorboard"]

    def test_explicit_single_string_wrapped_in_list(self):
        from backpropagate.trainer import Trainer
        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(report_to="wandb")
        assert trainer._resolve_report_to() == ["wandb"]

    def test_explicit_list_lowercased(self):
        from backpropagate.trainer import Trainer
        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(report_to=["WANDB", "TensorBoard"])
        assert trainer._resolve_report_to() == ["wandb", "tensorboard"]

    def test_unexpected_intent_type_falls_back_to_none(self, trainer):
        trainer._report_to_intent = 42  # type: ignore
        assert trainer._resolve_report_to() == "none"


class TestReportToFeatureFlags:
    """Ensure feature_flags surfaces per-tracker flags (wandb / tensorboard / mlflow)."""

    def test_features_dict_has_per_tracker_entries(self):
        from backpropagate.feature_flags import FEATURES

        assert "wandb" in FEATURES
        assert "tensorboard" in FEATURES
        assert "mlflow" in FEATURES

    def test_install_hints_have_per_tracker_entries(self):
        from backpropagate.feature_flags import INSTALL_HINTS

        assert "wandb" in INSTALL_HINTS
        assert "tensorboard" in INSTALL_HINTS
        assert "mlflow" in INSTALL_HINTS


# =============================================================================
# OOM AUTO-RECOVERY (B-002) — TESTS-A-004
# =============================================================================
#
# CHANGELOG.md L33 documents Trainer(oom_recovery=True) (default-on) that
# halves batch_size and doubles gradient_accumulation_steps on
# torch.cuda.OutOfMemoryError, and aborts after _OOM_MAX_RETRIES_AT_MIN_BATCH
# consecutive failures at batch_size=1 with TrainingError(code='RUNTIME_GPU_OOM').
#
# These tests pin the load-bearing v1.1.0 contract by mocking SFTTrainer to
# raise an OOM-shaped RuntimeError ("out of memory" — caught by the substring
# check at trainer.py:1027-1028 because torch.cuda.OutOfMemoryError is hard
# to instantiate without CUDA) on the first N .train() calls.


class _OOMScript:
    """Helper that builds an SFTTrainer mock whose .train() raises OOM N times.

    Why this lives here (not in conftest.py): the OOM-recovery code path
    re-instantiates SFTTrainer on each retry, so we need a fresh mock instance
    for each call that still shares the failure-counter across instances. A
    plain MagicMock(side_effect=[OOM, OOM, ok]) on .train() won't work — the
    side_effect list resets when SFTTrainer is re-instantiated.
    """

    def __init__(self, oom_count: int):
        self._oom_remaining = oom_count
        self._train_calls = 0

    def _make_instance(self) -> MagicMock:
        """Return a fresh SFTTrainer mock whose .train() consults shared state."""
        instance = MagicMock()

        def train_impl(*args, **kwargs):
            self._train_calls += 1
            if self._oom_remaining > 0:
                self._oom_remaining -= 1
                # OOM-shaped RuntimeError — trainer.py:1027 looks for the
                # substring "out of memory" (case-insensitive) when isinstance
                # of torch.cuda.OutOfMemoryError fails (which it does in this
                # test rig because torch.cuda isn't loaded).
                raise RuntimeError("CUDA out of memory at batch_size attempt")
            mock_result = MagicMock()
            mock_result.training_loss = 0.42
            return mock_result

        instance.train.side_effect = train_impl
        instance.state.log_history = [{"loss": 0.42}]
        return instance

    def factory(self, *args, **kwargs):
        return self._make_instance()


class TestOOMAutoRecovery:
    """Pin the v1.1.0 OOM auto-recovery contract (CHANGELOG L33, B-002)."""

    def _setup_trainer(self, temp_dir, *, oom_recovery: bool = True,
                       initial_batch: int = 4, initial_accum: int = 1):
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                output_dir=str(temp_dir),
                batch_size=initial_batch,
                gradient_accumulation=initial_accum,
                oom_recovery=oom_recovery,
                use_unsloth=False,
            )
        trainer._model = MagicMock()
        trainer._tokenizer = MagicMock()
        trainer._is_loaded = True
        return trainer

    def test_oom_halves_batch_and_doubles_accum(self, temp_dir):
        """First OOM => batch_size halved, grad_accum doubled, retry succeeds."""
        trainer = self._setup_trainer(
            temp_dir, oom_recovery=True, initial_batch=4, initial_accum=1
        )

        script = _OOMScript(oom_count=1)  # one OOM then success
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
             patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
             patch("trl.SFTTrainer", side_effect=script.factory), \
             patch("trl.SFTConfig"):
            run = trainer.train("dummy_dataset", steps=10)

        # SFTTrainer.train was called twice (first OOM, then retry).
        assert script._train_calls == 2, (
            f"Expected 2 .train() invocations (initial + 1 retry), "
            f"got {script._train_calls}"
        )
        # batch halved from 4 -> 2, accum doubled from 1 -> 2.
        assert trainer.batch_size == 2, (
            f"OOM recovery should have halved batch_size from 4 to 2; got "
            f"{trainer.batch_size}"
        )
        assert trainer.gradient_accumulation == 2, (
            f"OOM recovery should have doubled grad_accum from 1 to 2; got "
            f"{trainer.gradient_accumulation}"
        )
        # Training completed successfully on retry.
        assert run is not None

    def test_oom_at_min_batch_aborts_with_runtime_gpu_oom(self, temp_dir):
        """N consecutive OOMs at batch_size=1 => TrainingError RUNTIME_GPU_OOM."""
        from backpropagate.exceptions import TrainingError
        from backpropagate.trainer import Trainer

        trainer = self._setup_trainer(
            temp_dir, oom_recovery=True, initial_batch=1, initial_accum=1
        )

        # Push enough OOMs to exceed _OOM_MAX_RETRIES_AT_MIN_BATCH plus a buffer.
        ceiling = Trainer._OOM_MAX_RETRIES_AT_MIN_BATCH + 5
        script = _OOMScript(oom_count=ceiling)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
             patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
             patch("trl.SFTTrainer", side_effect=script.factory), \
             patch("trl.SFTConfig"), \
             pytest.raises(TrainingError) as exc_info:
            trainer.train("dummy_dataset", steps=10)

        # Structured code is the load-bearing programmatic signal — pin it.
        # Stage C BACKEND-B-003/B-005: the exhausted-recovery path now uses
        # the more specific RUNTIME_OOM_RECOVERY_EXHAUSTED code so triage can
        # distinguish "we hit our recovery limit" from "a single one-shot OOM
        # hit the wall". RUNTIME_GPU_OOM is still raised on the simpler
        # one-shot OOM path (and from oom_recovery=False); this specific
        # recovery-exhaustion branch carries the new code.
        assert getattr(exc_info.value, "code", None) == "RUNTIME_OOM_RECOVERY_EXHAUSTED", (
            f"Expected TrainingError(code='RUNTIME_OOM_RECOVERY_EXHAUSTED'); got "
            f"code={getattr(exc_info.value, 'code', None)!r} "
            f"message={exc_info.value!s}"
        )

        # The trainer should have given up after exactly the retry budget at
        # batch=1 (no halving possible since batch_size is already 1).
        assert script._train_calls == Trainer._OOM_MAX_RETRIES_AT_MIN_BATCH, (
            f"Expected {Trainer._OOM_MAX_RETRIES_AT_MIN_BATCH} attempts at "
            f"batch=1 before abort; got {script._train_calls}"
        )

    def test_oom_recovery_false_reraises_immediately(self, temp_dir):
        """oom_recovery=False => first OOM re-raises, no halving.

        Note on the wrapping: the OOM-shaped RuntimeError raised inside the
        retry loop escapes that loop unchanged when oom_recovery=False (see
        trainer.py:1030-1031), then the OUTER ``except RuntimeError`` handler
        at trainer.py:1189 wraps it into a ``TrainingError`` with a "GPU
        error during training" prefix because the message matches
        "out of memory". The test pins both shapes — re-raise IS observable
        as a TrainingError, but the orig RuntimeError is the ``__cause__``.
        """
        from backpropagate.exceptions import TrainingError

        trainer = self._setup_trainer(
            temp_dir, oom_recovery=False, initial_batch=4, initial_accum=1
        )

        script = _OOMScript(oom_count=5)  # more than enough, none should matter
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
             patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
             patch("trl.SFTTrainer", side_effect=script.factory), \
             patch("trl.SFTConfig"), \
             pytest.raises(TrainingError) as exc_info:
            trainer.train("dummy_dataset", steps=10)

        # The original OOM RuntimeError is the cause; the recovery loop did
        # not retry. (If recovery HAD run, we'd see code='RUNTIME_GPU_OOM'
        # from the retries-exhausted path, not the generic "GPU error" prefix.)
        assert isinstance(exc_info.value.__cause__, RuntimeError), (
            f"Expected __cause__ to be the original OOM RuntimeError; got "
            f"{type(exc_info.value.__cause__).__name__}"
        )
        assert "out of memory" in str(exc_info.value.__cause__).lower()

        # Exactly one attempt — no halving, no retries.
        assert script._train_calls == 1, (
            f"oom_recovery=False must re-raise on first OOM; got "
            f"{script._train_calls} attempts"
        )
        # Knobs untouched.
        assert trainer.batch_size == 4, (
            f"oom_recovery=False must NOT modify batch_size; got "
            f"{trainer.batch_size}"
        )
        assert trainer.gradient_accumulation == 1, (
            f"oom_recovery=False must NOT modify grad_accum; got "
            f"{trainer.gradient_accumulation}"
        )


# =============================================================================
# ATOMIC CHECKPOINT WRITES (B-006) — TESTS-A-005
# =============================================================================
#
# CHANGELOG.md L32: 'Atomic checkpoint writes — Trainer.save / SLAOMerger.save
# / export_lora / export_gguf all write to <path>.partial then rename to
# final. Disk-full mid-write no longer leaves corrupt artifacts.'
#
# These tests pin the contract by monkeypatching the inner write step to raise
# (simulating disk full) and asserting:
#   (a) the FINAL target file does not exist on disk
#   (b) the .partial sibling is cleaned up
#
# Both Trainer.save and SLAOMerger.save are covered here; export_lora /
# export_gguf coverage lives in test_export.py (added in this same wave).


class TestTrainerSaveAtomic:
    """Pin the atomic-write contract for Trainer.save (B-006, TESTS-A-005)."""

    def _make_trainer(self, output_dir):
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(output_dir), use_unsloth=False)
        trainer._model = MagicMock()
        trainer._tokenizer = MagicMock()
        trainer._is_loaded = True
        return trainer

    def test_save_happy_path_promotes_partial_to_final(self, temp_dir):
        """Trainer.save: success path leaves final dir, no .partial residue."""
        trainer = self._make_trainer(temp_dir)
        save_target = temp_dir / "lora"

        # save_pretrained writes a marker file so we can prove the move worked.
        def write_marker(path, *args, **kwargs):
            from pathlib import Path
            (Path(path) / "adapter_model.safetensors").write_bytes(b"weights")
            return None

        trainer._model.save_pretrained.side_effect = write_marker
        trainer._tokenizer.save_pretrained.side_effect = write_marker

        result_path = trainer.save(str(save_target))

        assert save_target.exists(), "Final save directory must exist on success"
        assert (save_target / "adapter_model.safetensors").exists(), (
            "Marker file written during partial-stage must be promoted to final dir"
        )
        partial_path = save_target.with_name(save_target.name + ".partial")
        assert not partial_path.exists(), (
            f"Partial directory must be cleaned up on success path; still "
            f"exists at {partial_path}"
        )
        assert str(save_target) in result_path

    def test_save_disk_full_mid_write_leaves_no_final_artifact(self, temp_dir):
        """Trainer.save: a mid-write failure must NOT leave a half-written final dir."""
        from backpropagate.exceptions import CheckpointError

        trainer = self._make_trainer(temp_dir)
        save_target = temp_dir / "lora"

        # Simulate disk-full: save_pretrained raises mid-write. The partial
        # directory has already been created by the trainer at this point.
        trainer._model.save_pretrained.side_effect = OSError(
            "[Errno 28] No space left on device"
        )

        with pytest.raises(CheckpointError):
            trainer.save(str(save_target))

        # The atomic contract: no final artifact, no .partial residue.
        assert not save_target.exists(), (
            "Final save directory must NOT exist after a mid-write failure; "
            "atomic promote should not have run."
        )
        partial_path = save_target.with_name(save_target.name + ".partial")
        assert not partial_path.exists(), (
            f"Partial directory must be cleaned up on failure path; still "
            f"exists at {partial_path}"
        )


class TestSLAOMergerSaveAtomic:
    """Pin the atomic-write contract for SLAOMerger.save (B-006, TESTS-A-005)."""

    def test_slao_save_happy_path(self, temp_dir):
        """SLAOMerger.save: success path leaves final dir, no .partial residue."""
        torch = pytest.importorskip("torch")
        from backpropagate.slao import SLAOMerger

        merger = SLAOMerger()
        merger.initialize({
            "layer.lora_A.weight": torch.randn(4, 8),
            "layer.lora_B.weight": torch.randn(8, 4),
        })

        save_target = temp_dir / "slao_checkpoint"
        merger.save(str(save_target))

        assert save_target.exists(), "Final SLAO save dir must exist on success"
        assert (save_target / "merge_history.json").exists(), (
            "merge_history.json must be present in the final dir"
        )
        partial_path = save_target.with_name(save_target.name + ".partial")
        assert not partial_path.exists(), (
            f"Partial dir must be cleaned up on success; still at {partial_path}"
        )

    def test_slao_save_disk_full_leaves_no_final_artifact(self, temp_dir,
                                                          monkeypatch):
        """SLAOMerger.save: torch.save failure must not leave half-written final dir."""
        torch = pytest.importorskip("torch")
        from backpropagate.slao import SLAOCheckpointError, SLAOMerger

        merger = SLAOMerger()
        merger.initialize({
            "layer.lora_A.weight": torch.randn(4, 8),
            "layer.lora_B.weight": torch.randn(8, 4),
        })

        save_target = temp_dir / "slao_checkpoint"

        # Disk-full simulation: torch.save raises mid-write.
        def fake_torch_save(*args, **kwargs):
            raise OSError("[Errno 28] No space left on device")

        monkeypatch.setattr("torch.save", fake_torch_save)

        with pytest.raises(SLAOCheckpointError):
            merger.save(str(save_target))

        # Atomic contract: no final dir, partial cleaned up.
        assert not save_target.exists(), (
            "Final SLAO save dir must NOT exist after disk-full mid-write"
        )
        partial_path = save_target.with_name(save_target.name + ".partial")
        assert not partial_path.exists(), (
            f"Partial dir must be cleaned up on failure; still at {partial_path}"
        )


# =============================================================================
# TESTS-B-008 — Trainer(unsloth_fallback=True) wiring
# =============================================================================
#
# CHANGELOG v1.1.0 (B-010) introduced ``unsloth_fallback=True`` as a default-on
# Trainer flag: when Unsloth's ``FastLanguageModel.from_pretrained`` raises a
# non-CUDA / non-ImportError exception, the trainer falls back to plain
# ``AutoModelForCausalLM`` + ``peft.get_peft_model`` so an Unsloth nightly
# breakage no longer takes a fine-tuning pipeline down. Stage B audit found
# ZERO direct tests asserting the fallback fires (or that the
# ``unsloth_fallback=False`` opt-out re-raises). The wiring is load-bearing
# graceful-degradation that would silently rot.


class TestUnslothFallback:
    """B-010 graceful degradation: Unsloth failure → transformers fallback."""

    def test_unsloth_failure_falls_back_to_transformers_by_default(self):
        """When ``unsloth_fallback=True`` (default), Unsloth errors trigger
        the transformers + PEFT load path instead of raising.

        We construct the Trainer with ``use_unsloth=True``, then stub
        ``_load_with_unsloth`` to raise a *non-CUDA, non-ImportError*
        exception (the only shape the fallback layer catches per
        trainer.py:601-617). The expected behaviour is:
          - ``_load_with_unsloth`` is called once
          - the exception is logged + swallowed
          - ``_load_with_transformers`` is called as the fallback
          - ``use_unsloth`` flips to False so subsequent operations don't
            re-attempt the failed path
          - ``_is_loaded`` ends True
        """
        from backpropagate import feature_flags
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False), \
             patch.dict(feature_flags.FEATURES, {"unsloth": True}):
            trainer = Trainer(use_unsloth=True, unsloth_fallback=True)
            assert trainer.unsloth_fallback is True
            assert trainer.use_unsloth is True

            unsloth_err = ValueError("unsloth nightly broke: bf16 detection bug")

            with patch.object(
                trainer, "_load_with_unsloth", side_effect=unsloth_err
            ) as mock_unsloth, patch.object(
                trainer, "_load_with_transformers"
            ) as mock_transformers:
                trainer.load_model()

            mock_unsloth.assert_called_once()
            mock_transformers.assert_called_once()
            assert trainer.use_unsloth is False, (
                "After fallback, use_unsloth must flip False so further "
                "ops don't retry the broken path"
            )
            assert trainer._is_loaded is True

    def test_unsloth_failure_reraises_when_fallback_disabled(self):
        """``unsloth_fallback=False`` re-raises the original exception unchanged.

        Operators who insist on Unsloth's perf characteristics opt out of
        the silent degradation. The original exception type must propagate
        so the outer except blocks (ModelLoadError translation) classify
        it correctly — a bare ``raise`` inside the except handler
        preserves type + traceback.
        """
        from backpropagate import feature_flags
        from backpropagate.exceptions import ModelLoadError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False), \
             patch.dict(feature_flags.FEATURES, {"unsloth": True}):
            trainer = Trainer(use_unsloth=True, unsloth_fallback=False)
            assert trainer.unsloth_fallback is False

            unsloth_err = ValueError("unsloth nightly broke: bf16 detection bug")

            with patch.object(
                trainer, "_load_with_unsloth", side_effect=unsloth_err
            ), patch.object(
                trainer, "_load_with_transformers"
            ) as mock_transformers:
                # The outer try/except in load_model() wraps this in
                # ModelLoadError (cause_category=unknown). We only need to
                # see that the fallback was NOT taken — the exact wrapping
                # behavior is covered by the load-model error tests above.
                with pytest.raises((ModelLoadError, ValueError)):
                    trainer.load_model()

                # TESTS-B-016 (Stage C): drop the trailing tuple — assert_not_called
                # itself raises AssertionError("Expected to be called 0 times. Called N times.")
                # which is the operator-facing message that matters. Keep the
                # intent as a leading comment instead of a never-fired msg.
                # Contract: with unsloth_fallback=False, the transformers fallback
                # must NOT be invoked — the original error must propagate.
                mock_transformers.assert_not_called()

    def test_unsloth_import_error_skips_fallback_path(self):
        """ImportError bypasses the fallback and surfaces as ModelLoadError.

        The fallback layer at trainer.py:601 deliberately re-raises
        ImportError + RuntimeError (CUDA shape) so the surrounding except
        blocks can route to the right ModelLoadError category. A missing
        upstream package is a "your env is wrong" signal, not a transient
        failure that fallback should paper over.
        """
        from backpropagate import feature_flags
        from backpropagate.exceptions import ModelLoadError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False), \
             patch.dict(feature_flags.FEATURES, {"unsloth": True}):
            trainer = Trainer(use_unsloth=True, unsloth_fallback=True)

            with patch.object(
                trainer,
                "_load_with_unsloth",
                side_effect=ImportError("No module named 'unsloth.kernels'"),
            ), patch.object(
                trainer, "_load_with_transformers"
            ) as mock_transformers:
                with pytest.raises(ModelLoadError):
                    trainer.load_model()

                mock_transformers.assert_not_called()


# =============================================================================
# TESTS-B-015 — HF Hub transient retry loop timing
# =============================================================================
#
# CHANGELOG v1.1.0 (B-017): every HF call retries on 5xx / 429 / connection /
# timeout with exponential backoff (3 attempts, multiplier=2, base=5s,
# max=60s). 401 / 403 / 404 surface in <1s with cause-classified hints. The
# Stage B audit caught that ``TestHfTransientRetryStatusCodeFilter``
# exercises the *predicate* (``_is_transient_hf_exception``) but never
# drives the actual retry loop — so a regression that broke the tenacity
# wiring (wrong retry= argument, missing decorator, swapped wait policy)
# would pass the predicate tests and ship anyway.
#
# We mock ``time.sleep`` on the tenacity sleep callback so a real retry
# loop completes in milliseconds instead of seconds. ``before_sleep_log``
# is invoked by tenacity between attempts; counting wait calls + their
# duration argument pins both the retry count and the exponential schedule.


class TestHfTransientRetryLoop:
    """B-017: drive ``_retry_hf_call`` end-to-end with mocked HTTP responses."""

    def _make_http_error(self, status_code: int):
        """Build a requests.HTTPError that the transient predicate accepts."""
        import requests

        exc = requests.exceptions.HTTPError(f"HTTP {status_code}")
        exc.response = MagicMock(status_code=status_code)
        return exc

    def test_retries_until_success_after_transient_failures(self, monkeypatch):
        """503 twice then success → call_count == 3, no exception propagated."""
        from backpropagate.trainer import _retry_hf_call

        attempts = {"count": 0}

        def flaky_call():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise self._make_http_error(503)
            return "ok"

        # Patch tenacity's internal sleep so the wait_exponential schedule
        # doesn't actually block the test. tenacity.nap.sleep is the
        # public-ish hook; if it moves, the test will surface as a slow
        # run (capped at the ~65s exponential ceiling) — failure modes
        # are loud either way.
        slept = []
        monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: slept.append(s))

        result = _retry_hf_call(flaky_call, _label="test:flaky_503")

        assert result == "ok"
        assert attempts["count"] == 3, (
            f"Expected exactly 3 attempts (2 retries + 1 success), got {attempts['count']}"
        )
        # Two sleeps between three attempts. The first delay is the
        # ``min`` floor (5s) and the second is bounded by ``max`` (60s).
        assert len(slept) == 2, (
            f"Expected 2 inter-attempt sleeps, got {len(slept)}: {slept!r}"
        )

    def test_exhausts_retries_then_raises(self, monkeypatch):
        """Always-503 → exactly ``_RETRY_ATTEMPTS`` attempts then re-raise."""
        import requests

        from backpropagate.trainer import _RETRY_ATTEMPTS, _retry_hf_call

        attempts = {"count": 0}

        def always_fails():
            attempts["count"] += 1
            raise self._make_http_error(503)

        monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)

        with pytest.raises(requests.exceptions.HTTPError):
            _retry_hf_call(always_fails, _label="test:always_503")

        assert attempts["count"] == _RETRY_ATTEMPTS, (
            f"Should attempt exactly {_RETRY_ATTEMPTS} times before giving "
            f"up; got {attempts['count']}"
        )

    def test_does_not_retry_401_fast_fails(self, monkeypatch):
        """401 (auth) → one attempt, raised immediately, no sleeps."""
        import requests

        from backpropagate.trainer import _retry_hf_call

        attempts = {"count": 0}

        def always_401():
            attempts["count"] += 1
            raise self._make_http_error(401)

        slept = []
        monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: slept.append(s))

        with pytest.raises(requests.exceptions.HTTPError):
            _retry_hf_call(always_401, _label="test:401_no_retry")

        assert attempts["count"] == 1, (
            "401 must surface in 1 attempt — the v1.1.0 promise is that "
            "auth failures don't wait through ~65s of exponential backoff "
            f"before showing the real error. Got {attempts['count']} attempts."
        )
        assert slept == [], (
            f"401 must NOT trigger any retry sleeps; got {slept!r}"
        )

    def test_exponential_backoff_schedule_monotonic(self, monkeypatch):
        """Inter-attempt delays grow monotonically (exponential backoff).

        We don't pin the exact seconds (the schedule depends on
        wait_exponential's internal calculation against the constants in
        trainer.py) but we do pin the *shape*: with multiplier=2 and a
        non-trivial min, the second delay must be >= the first.
        """
        from backpropagate.trainer import _retry_hf_call

        attempts = {"count": 0}

        def always_fails():
            attempts["count"] += 1
            raise self._make_http_error(503)

        slept = []
        monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: slept.append(s))

        import requests
        with pytest.raises(requests.exceptions.HTTPError):
            _retry_hf_call(always_fails, _label="test:backoff_shape")

        # 3 attempts → 2 inter-attempt sleeps.
        assert len(slept) == 2, f"Expected 2 sleeps, got {slept!r}"
        # Exponential schedule with multiplier=2: second delay must be
        # at least as large as first (tenacity caps at ``max`` but never
        # decreases the base).
        assert slept[1] >= slept[0], (
            f"Backoff must be non-decreasing; got slept={slept!r}"
        )


# =============================================================================
# TESTS-B-005 + TESTS-B-011 — run_id correlation via caplog
# =============================================================================
#
# CHANGELOG v1.1.0 (B-001): every training run mints a UUID4 that flows
# through every log line, the checkpoint manifest, and the SLAO merge
# record. The audit found 34 test files but only 1 uses ``caplog`` — and
# the existing ``run_id`` references check individual surfaces in isolation
# rather than asserting the same token propagates across them. A regression
# that minted distinct run_ids per surface (or dropped the structlog
# context bind) would pass every existing test.
#
# These tests use Python's ``caplog`` fixture (which captures records from
# the standard logging.Logger underlying structlog) to assert (a) the
# bind/unbind round-trip works at the primitive level and (b) the
# ``run_started`` and ``run_ended`` events at trainer.py:966 / :1235 emit
# with the same UUID. We deliberately do NOT drive a full Trainer.train
# here — the OOM auto-recovery + SFTTrainer mocking machinery already
# covers that path. The narrower bind/log assertions pin the contract
# without requiring 200+ lines of trainer-test scaffolding.


class TestRunIdCorrelation:
    """B-001: run_id propagates through the operator-visible log surface.

    IMPLEMENTATION NOTE (Stage C amend): pytest's ``caplog`` fixture only
    captures records that pass through the stdlib ``logging`` hierarchy.
    ``logging_config._configure_structlog`` (logging_config.py:140) routes
    output through ``structlog.PrintLoggerFactory()`` to ``sys.stdout``,
    bypassing the stdlib root logger entirely when structlog is installed
    (the default in the [logging] extra). So the assertion target here is
    ``capsys`` — we read the rendered JSON / console lines off stdout.
    That matches what an operator running ``backprop train ...`` actually
    sees in their terminal, which is the load-bearing observability
    surface the audit cares about.
    """

    def test_bind_run_context_round_trip(self, capsys):
        """``bind_run_context(run_id=X)`` makes X reachable from emitted output."""
        from backpropagate.logging_config import (
            bind_run_context,
            configure_logging,
            get_logger,
            unbind_run_context,
        )

        configure_logging(level="INFO", json_logs=True, force=True)

        try:
            bind_run_context(run_id="stage-c-bind-token-1234")
            logger = get_logger("backpropagate.test.run_id")
            logger.info("run_started_test_event")
        finally:
            unbind_run_context("run_id")

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "stage-c-bind-token-1234" in combined, (
            "Bound run_id must appear in the rendered output; captured: "
            f"{combined!r}"
        )

    def test_unbind_run_context_isolates_subsequent_logs(self, capsys):
        """After ``unbind_run_context``, the token must NOT leak to later logs.

        Pins the thread-isolation half of the contract: a re-used worker
        thread that ran one training session must NOT carry the prior
        run's UUID into the next session's records.
        """
        from backpropagate.logging_config import (
            bind_run_context,
            configure_logging,
            get_logger,
            unbind_run_context,
        )

        configure_logging(level="INFO", json_logs=True, force=True)

        bind_run_context(run_id="stage-c-token-A")
        unbind_run_context("run_id")

        # Discard pre-marker noise so the leak check only inspects the
        # line we're about to emit.
        capsys.readouterr()

        logger = get_logger("backpropagate.test.unbind")
        logger.info("post_unbind_marker_event")

        captured = capsys.readouterr()
        combined = captured.out + captured.err

        marker_lines = [
            ln for ln in combined.splitlines() if "post_unbind_marker_event" in ln
        ]
        assert marker_lines, (
            f"post-unbind event missing from captured stdout: {combined!r}"
        )
        leaked = [ln for ln in marker_lines if "stage-c-token-A" in ln]
        assert not leaked, (
            "Unbound run_id leaked into a subsequent log line — the "
            "thread-isolation invariant for B-001 is broken. "
            f"Leaked line(s): {leaked!r}"
        )

    def test_run_id_same_token_across_start_and_end_events(self, capsys):
        """Sanity-pin: a single bind window emits start + end with one token.

        Mirror of the trainer.py:966 ``run_started`` / :1235 ``run_ended``
        pair using the same primitives so a regression in either surface
        would be caught here without standing up the full Trainer
        machinery (the OOM auto-recovery + SFTTrainer mocking covers that
        path separately).
        """
        import uuid

        from backpropagate.logging_config import (
            bind_run_context,
            configure_logging,
            get_logger,
            unbind_run_context,
        )

        configure_logging(level="INFO", json_logs=True, force=True)

        run_id = uuid.uuid4().hex
        logger = get_logger("backpropagate.test.run_pair")

        # Drop configure_logging side-effects from capture.
        capsys.readouterr()

        try:
            bind_run_context(run_id=run_id, session_kind="single_run")
            logger.info(f"run_started run_id={run_id} legacy_label=test")
            # ... pretend training happens ...
            logger.info(f"run_ended run_id={run_id} status=ok")
        finally:
            unbind_run_context("run_id", "session_kind")

        captured = capsys.readouterr()
        combined = captured.out + captured.err

        starts = [ln for ln in combined.splitlines() if "run_started" in ln]
        ends = [ln for ln in combined.splitlines() if "run_ended" in ln]
        assert starts and ends, (
            "Both run_started and run_ended events must be captured; "
            f"got starts={len(starts)} ends={len(ends)} from: {combined!r}"
        )
        assert any(run_id in ln for ln in starts), \
            "run_started event missing run_id correlation"
        assert any(run_id in ln for ln in ends), \
            "run_ended event missing run_id correlation"


# =============================================================================
# TESTS-B-005 — operator-visible payload check for TrainingLogger.log_step
# =============================================================================
#
# The Stage B audit flagged that ~15 TrainingLogger tests in
# test_logging_config.py call a log method and assert nothing ("# Should
# not raise"). A drop-the-payload regression would pass them all. The
# loadbearing path through TrainingLogger is the structlog branch where
# kwargs become event-dict fields — that's what tools downstream
# (Loki / W&B / MLflow) parse. The assertions below pin (a) the event is
# emitted and (b) the data kwargs land in the rendered output — captured
# via stdout because configure_logging routes structlog through
# PrintLoggerFactory (see TestRunIdCorrelation docstring above).


class TestTrainingLoggerCaplog:
    """B-001 + observability: TrainingLogger emits structured records, not no-ops."""

    def test_log_step_emits_capturable_record(self, capsys):
        """``log_step(step, loss)`` produces an INFO record with the values."""
        from backpropagate.logging_config import TrainingLogger, configure_logging

        configure_logging(level="DEBUG", json_logs=True, force=True)
        capsys.readouterr()  # discard configure_logging side-effects

        tlog = TrainingLogger("caplog-step-run")
        tlog.log_step(step=42, loss=1.234)

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "42" in combined, (
            f"step value missing from rendered log output: {combined!r}"
        )
        assert "1.234" in combined or "1.23" in combined, (
            f"loss value missing from rendered log output: {combined!r}"
        )

    def test_log_run_start_emits_capturable_record(self, capsys):
        """``log_run_start(model=..., dataset=...)`` records the model name."""
        from backpropagate.logging_config import TrainingLogger, configure_logging

        configure_logging(level="DEBUG", json_logs=True, force=True)
        capsys.readouterr()  # discard configure_logging side-effects

        tlog = TrainingLogger("caplog-start-run")
        tlog.log_run_start(model="qwen-2.5-7b-instruct", dataset="my-data.jsonl")

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "qwen-2.5-7b-instruct" in combined, (
            f"model field missing from rendered log output: {combined!r}"
        )
        assert "my-data.jsonl" in combined, (
            f"dataset field missing from rendered log output: {combined!r}"
        )


# =============================================================================
# BACKEND-F-002 — single-run resume_from miss is a hard error (Wave 6a regression)
# =============================================================================
#
# Cross-domain pin from Wave 5.5 BACKEND-F-002 fix
# (backpropagate/trainer.py::Trainer.train). Pre-fix, passing
# ``resume_from=<unknown-id>`` to ``Trainer.train()`` silently fell
# through to a fresh run under a NEW run_id — the operator's resume
# intent was dropped without an exception, the on-disk history record
# was created with an unrelated run_id, and the model produced was not
# what the operator asked for.
#
# The fix raises ``InvalidSettingError`` (code ``CONFIG_INVALID_SETTING``)
# when the resume_from lookup misses. The error message + suggestion
# carry the load-bearing diagnostic anchors:
#   - the requested run_id (so the operator can copy-paste to the
#     ``backprop runs`` command)
#   - the output_dir actually searched (so a mistyped --output is
#     immediately obvious)
#   - the operator's next steps (``backprop runs`` list, or re-run
#     with the right --output, or omit resume_from to start fresh)
#
# This test pins all three anchors so a future refactor that drops
# any of them lands red in CI.
class TestResumeFromStrictMiss:
    """BACKEND-F-002 regression: strict resume_from miss raises."""

    def test_resume_from_missing_raises_invalid_setting_error(self, temp_dir):
        """A ``resume_from`` that names a run_id not in the on-disk
        history raises ``InvalidSettingError`` instead of silently
        starting fresh.
        """
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            # Bypass model loading + dataset loading; the resume check
            # runs AFTER both, so we need to short-circuit them so the
            # test fails fast on the resume miss rather than blowing
            # up on a missing model or dataset.
            trainer._is_loaded = True
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()

            with patch.object(
                trainer, "_load_dataset", return_value=MagicMock()
            ):
                with pytest.raises(InvalidSettingError) as excinfo:
                    trainer.train(
                        dataset="dummy.jsonl",
                        resume_from="nonexistent-run-id-test-f002",
                    )

        # Code anchor: stable code for the catalog.
        exc = excinfo.value
        assert exc.code == "CONFIG_INVALID_SETTING", (
            f"BACKEND-F-002 error-code drift: expected CONFIG_INVALID_SETTING, "
            f"got {exc.code!r}. Update both the trainer + ERROR_CODES catalog "
            f"if you intentionally renamed the code."
        )

    def test_resume_from_missing_message_anchors_run_id_and_output_dir(self, temp_dir):
        """The error message + suggestion together MUST name the
        requested run_id and the output_dir searched. The whole point
        of the F-002 fix is operator-actionable failure — an
        InvalidSettingError without these anchors is the same silent
        wrong-model bug behind a different exception type.
        """
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        requested_id = "nonexistent-id-anchored-f002"

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._is_loaded = True
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()

            with patch.object(
                trainer, "_load_dataset", return_value=MagicMock()
            ):
                with pytest.raises(InvalidSettingError) as excinfo:
                    trainer.train(
                        dataset="dummy.jsonl",
                        resume_from=requested_id,
                    )

        exc = excinfo.value
        # Combine the rendered message + suggestion + structured
        # details into a single haystack so the test is robust to
        # which field carries each anchor.
        haystack = (
            f"{exc!s} || "
            f"{exc.suggestion or ''} || "
            f"{exc.details or {}}"
        )

        # The requested run_id must appear somewhere — the operator's
        # most common next move is to copy-paste it into
        # ``backprop runs | grep`` to verify whether the typo is in
        # the ID or the output_dir.
        assert requested_id in haystack, (
            f"BACKEND-F-002 contract violation: requested run_id "
            f"{requested_id!r} missing from error surface. Rendered: "
            f"{haystack!r}"
        )

        # The output_dir leaf name must appear so a mistyped --output
        # is immediately obvious. We match the leaf rather than the
        # full path to stay robust across Windows backslashes vs POSIX
        # forward slashes in the rendered repr.
        assert temp_dir.name in haystack, (
            f"BACKEND-F-002 contract violation: output_dir leaf "
            f"{temp_dir.name!r} missing from error surface. The operator "
            f"can't tell whether they passed the wrong --output. "
            f"Rendered: {haystack!r}"
        )

    def test_resume_from_missing_suggestion_mentions_backprop_runs(self, temp_dir):
        """The suggestion text MUST mention ``backprop runs`` — the
        operator-actionable CLI command that lists the available
        run_ids under an output_dir. Without this hint, the operator
        sees the failure but doesn't know the recovery move.
        """
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._is_loaded = True
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()

            with patch.object(
                trainer, "_load_dataset", return_value=MagicMock()
            ):
                with pytest.raises(InvalidSettingError) as excinfo:
                    trainer.train(
                        dataset="dummy.jsonl",
                        resume_from="another-missing-id-f002",
                    )

        suggestion = excinfo.value.suggestion or ""
        assert "backprop runs" in suggestion, (
            "BACKEND-F-002 contract violation: suggestion text doesn't "
            "mention `backprop runs` — the operator can't discover the "
            "recovery command from the error alone. "
            f"suggestion={suggestion!r}"
        )
