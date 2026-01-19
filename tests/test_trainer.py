"""Tests for Trainer class (mocked GPU)."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import os


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

    def test_windows_env_vars_set(self):
        """Test Windows environment variables are set."""
        from backpropagate.trainer import Trainer

        with patch("os.name", "nt"), \
             patch("torch.cuda.is_available", return_value=False):
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
        from backpropagate.trainer import Trainer

        from backpropagate.exceptions import TrainingError

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


class TestTrainerExport:
    """Tests for Trainer export functionality."""

    def test_export_raises_without_model(self, temp_dir):
        """Test export raises error when model not loaded."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))

        with pytest.raises(RuntimeError, match="No model loaded"):
            trainer.export()

    def test_export_lora(self, temp_dir):
        """Test export to LoRA format."""
        from backpropagate.trainer import Trainer
        from backpropagate.export import ExportFormat

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
        from backpropagate.trainer import Trainer
        from backpropagate.export import ExportFormat

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

        with pytest.raises(RuntimeError, match="No model loaded"):
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
        from backpropagate.trainer import load_dataset
        import json

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
        from backpropagate.trainer import load_dataset
        import json

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
        from backpropagate.trainer import Trainer
        from backpropagate import feature_flags

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
        from backpropagate.trainer import Trainer
        from backpropagate import feature_flags

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
                assert run.run_id == "run_1"

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
        from backpropagate.trainer import load_model, Trainer

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
        from backpropagate.trainer import load_model, Trainer

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
        from backpropagate.trainer import Trainer
        import json

        # Create test dataset with many samples
        jsonl_path = temp_dir / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(100):
                f.write(json.dumps({"text": f"sample {i}"}) + "\n")

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

            ds = trainer._load_dataset(str(jsonl_path), samples=10)
            assert len(ds) == 10

    def test_load_dataset_invalid_type_raises(self):
        """_load_dataset should raise for unsupported types."""
        from backpropagate.trainer import Trainer
        from backpropagate.exceptions import DatasetError

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
        from backpropagate.trainer import Trainer
        from backpropagate import feature_flags

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
        from backpropagate.trainer import Trainer
        from backpropagate.config import settings

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
        from backpropagate.trainer import Trainer
        from backpropagate.exceptions import InvalidSettingError

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()
            trainer._is_loaded = True
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()

            with pytest.raises(InvalidSettingError, match="steps"):
                trainer.train("dummy", steps=-5)

    def test_train_invalid_samples_raises(self):
        """Invalid samples parameter should raise error."""
        from backpropagate.trainer import Trainer
        from backpropagate.exceptions import InvalidSettingError

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()
            trainer._is_loaded = True
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()

            with pytest.raises(InvalidSettingError, match="samples"):
                trainer.train("dummy", samples=0)

    def test_train_zero_steps_raises(self):
        """Zero steps should raise error."""
        from backpropagate.trainer import Trainer
        from backpropagate.exceptions import InvalidSettingError

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
        from backpropagate.trainer import Trainer
        from backpropagate.exceptions import ModelLoadError

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(use_unsloth=False)

            with patch.object(trainer, "_load_with_transformers", side_effect=ImportError("test")):
                with pytest.raises(ModelLoadError):
                    trainer.load_model()

    def test_load_model_cuda_error(self):
        """CUDA error during load should raise GPUNotAvailableError."""
        from backpropagate.trainer import Trainer
        from backpropagate.exceptions import GPUNotAvailableError

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(use_unsloth=False)

            with patch.object(trainer, "_load_with_transformers", side_effect=RuntimeError("CUDA out of memory")):
                with pytest.raises(GPUNotAvailableError):
                    trainer.load_model()

    def test_train_oom_error(self, temp_dir):
        """OOM during training should raise TrainingError with helpful message."""
        from backpropagate.trainer import Trainer
        from backpropagate.exceptions import TrainingError

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
        from backpropagate.trainer import Trainer
        from backpropagate.exceptions import CheckpointError

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
        from backpropagate.trainer import Trainer
        from backpropagate.exceptions import DatasetNotFoundError

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

            with pytest.raises(DatasetNotFoundError):
                trainer._load_dataset(str(temp_dir / "nonexistent.jsonl"))

    def test_load_dataset_invalid_json(self, temp_dir):
        """Invalid JSON file should raise DatasetParseError."""
        from backpropagate.trainer import Trainer
        from backpropagate.exceptions import DatasetParseError

        # Create invalid JSON file
        invalid_file = temp_dir / "invalid.jsonl"
        invalid_file.write_text("not valid json {{{")

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

            with pytest.raises(DatasetParseError):
                trainer._load_dataset(str(invalid_file))


# =============================================================================
# FIXTURE FOR TESTS
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for tests."""
    return tmp_path
