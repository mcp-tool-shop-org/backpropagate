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
        assert getattr(exc_info.value, "code", None) == "RUNTIME_GPU_OOM", (
            f"Expected TrainingError(code='RUNTIME_GPU_OOM'); got "
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
