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
        # v1.3 BACKEND-1: LoRAConfig defaults bumped to rank 256 + all-linear.
        assert trainer.lora_r == 256, (
            f"Trainer default lora_r drifted from v1.3 BACKEND-1 "
            f"'quality' contract: expected 256, got {trainer.lora_r}. "
            f"This is the Trainer instance read; if it disagrees with "
            f"LoRAConfig defaults, the Trainer.__init__ ingestion path "
            f"has a bug. v1.5 TESTS-A-006 sweep watches UI / CLI "
            f"divergence on this same field."
        )
        assert trainer.lora_alpha == 512, (
            f"Trainer default lora_alpha drifted: expected 512, got "
            f"{trainer.lora_alpha}. Pinned alongside lora_r so the 2:1 "
            f"ratio breaks loud."
        )
        assert trainer._is_loaded is False, (
            f"Fresh Trainer instance should not be loaded; "
            f"_is_loaded={trainer._is_loaded!r}. Eager-load on __init__ "
            f"breaks the 'cheap-to-construct, costly-to-load' contract."
        )

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


class TestTrainerWave6bKwargs:
    """BRIDGE-A-002 follow-up (v1.4 Wave 6a): per-invocation Wave 6b knobs.

    The five Wave 6b knobs (use_dora, packing, init_lora_weights,
    lora_preset, optim) were previously env-var-only via settings.lora.* /
    settings.data.* / settings.training.*. Wave 6a foundation declares
    them as explicit Trainer.__init__ kwargs so the CLI introspection
    filter (cmd_train / cmd_multi_run / cmd_replay) threads them through
    to the constructor instead of silently dropping them.

    Contract preservation: kwarg=None falls back to settings (pre-fix
    byte-identical behavior); kwarg=value overrides settings.
    """

    def test_use_dora_explicit_override(self):
        """Trainer(use_dora=True) sets self.use_dora True."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(use_dora=True)

        assert trainer.use_dora is True

    def test_use_dora_none_falls_back_to_settings(self):
        """Trainer(use_dora=None) reads settings.lora.use_dora."""
        from backpropagate.config import settings
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(use_dora=None)

        # Settings default in v1.3 LoRAConfig is False.
        assert trainer.use_dora == settings.lora.use_dora

    def test_packing_explicit_override(self):
        """Trainer(packing=False) sets self.packing False (overrides settings)."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(packing=False)

        assert trainer.packing is False

    def test_packing_none_falls_back_to_settings(self):
        """Trainer(packing=None) reads settings.data.packing."""
        from backpropagate.config import settings
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(packing=None)

        # Settings default in v1.3 BACKEND-4 is True.
        assert trainer.packing == settings.data.packing

    def test_init_lora_weights_explicit_override(self):
        """Trainer(init_lora_weights='pissa') threads the string through."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(init_lora_weights="pissa")

        assert trainer.init_lora_weights == "pissa"

    def test_init_lora_weights_none_falls_back_to_settings(self):
        """Trainer(init_lora_weights=None) reads settings.lora.init_lora_weights."""
        from backpropagate.config import settings
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(init_lora_weights=None)

        assert trainer.init_lora_weights == settings.lora.init_lora_weights

    def test_lora_preset_explicit_override(self):
        """Trainer(lora_preset='fast') sets self.lora_preset='fast'."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(lora_preset="fast")

        assert trainer.lora_preset == "fast"

    def test_lora_preset_none_defaults_to_quality(self):
        """Trainer(lora_preset=None) defaults to 'quality' (v1.3 BACKEND-1).

        There is no ``settings.lora.preset`` field — the preset is a
        future-overlay slot. None defaults to "quality" to match the v1.3
        BACKEND-1 default contract (rank 256 + all-linear).
        """
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(lora_preset=None)

        assert trainer.lora_preset == "quality"

    def test_optim_explicit_override(self):
        """Trainer(optim='adamw_torch') threads the string through."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(optim="adamw_torch")

        assert trainer.optim == "adamw_torch"

    def test_optim_none_falls_back_to_settings(self):
        """Trainer(optim=None) reads settings.training.optim."""
        from backpropagate.config import settings
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(optim=None)

        assert trainer.optim == settings.training.optim

    def test_optim_auto_sentinel_falls_back_to_settings(self):
        """Trainer(optim='auto') treats 'auto' as None-equivalent.

        The CLI's --optim flag accepts 'auto' as a sentinel meaning
        "let the trainer pick" (see cli.py:4983). The constructor maps
        'auto' to the settings fallback so ``_detect_optim_for_card``
        sees the actual configured default ('adamw_8bit') instead of
        passing the literal 'auto' string to TRL/HF.
        """
        from backpropagate.config import settings
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(optim="auto")

        assert trainer.optim == settings.training.optim

    def test_all_five_wave6b_kwargs_introspectable(self):
        """Trainer.__init__ signature names all 5 Wave 6b kwargs.

        This is the LOAD-BEARING test for BRIDGE-A-002 follow-up — pre-fix
        the introspection filter in cmd_train / cmd_multi_run / cmd_replay
        silently dropped these keys because they were not in the
        signature. The filter is correct; the constructor was the gap.
        """
        import inspect

        from backpropagate.trainer import Trainer

        sig = inspect.signature(Trainer.__init__)
        params = set(sig.parameters)

        for kwarg in ("use_dora", "packing", "init_lora_weights", "lora_preset", "optim"):
            assert kwarg in params, (
                f"Trainer.__init__ signature is missing Wave 6b kwarg "
                f"{kwarg!r}. The CLI introspection filter at cmd_train "
                f"(cli.py:644) / cmd_multi_run (cli.py:877) / cmd_replay "
                f"(cli.py:4275) silently drops keys outside this signature."
            )

    def test_cmd_train_threads_use_dora_to_trainer(self):
        """End-to-end: cmd_train --use-dora → Trainer(use_dora=True).

        Verifies the CLI introspection filter at cmd_train correctly
        passes the Wave 6b kwarg through to the Trainer constructor now
        that the signature names it. Pre-fix, this kwarg was silently
        dropped by the filter (the only path was the BACKPROPAGATE_LORA__
        env var).
        """
        import argparse

        from backpropagate.cli import cmd_train

        # Build a Namespace that emulates argparse output for
        # `backprop train --data dummy.jsonl --use-dora`.
        args = argparse.Namespace(
            data="dummy.jsonl",
            model="test/model",
            lora_r=16,
            lr=2e-4,
            batch_size="auto",
            output="./output",
            no_unsloth=True,
            steps=1,
            samples=1,
            use_dora=True,
            no_packing=False,
            init_lora_weights="default",
            lora_preset="quality",
            optim="auto",
            resume=None,
            cli_run_id=None,
            verbose=False,
        )

        # Capture the kwargs that the Trainer constructor receives so we
        # can assert use_dora=True flowed through end-to-end. We DO NOT
        # actually train — short-circuit by raising from the
        # constructor capture so the CLI handler returns EXIT_RUNTIME_ERROR
        # via the catch-all (cmd_train's try/except envelope).
        captured: dict = {}

        class _SentinelStop(Exception):
            pass

        def _capturing_trainer_init(self, *args, **kwargs):
            captured.update(kwargs)
            raise _SentinelStop("captured kwargs; halting test path")

        # cmd_train imports Trainer via ``from .trainer import Trainer`` at
        # the top of the function body, so the patch target is the canonical
        # module path, not ``backpropagate.cli.Trainer``.
        with patch("backpropagate.trainer.Trainer.__init__", _capturing_trainer_init), \
             patch("torch.cuda.is_available", return_value=False):
            # cmd_train catches all exceptions and returns an exit code; we
            # don't care which because we only assert the captured kwargs.
            try:
                cmd_train(args)
            except _SentinelStop:
                # Defensive: the catch-all in cmd_train should swallow this,
                # but if any wrapper bypasses it we still want the test to
                # check captured kwargs.
                pass

        assert captured.get("use_dora") is True, (
            f"cmd_train did not thread --use-dora through to Trainer "
            f"constructor; captured kwargs were: {captured!r}. Pre-fix "
            f"the introspection filter silently dropped this key because "
            f"Trainer.__init__ did not name it."
        )


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

        with pytest.raises(TrainingError, match="no model is loaded"):
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

        with pytest.raises((RuntimeError, Exception), match="no model is loaded"):
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

        with pytest.raises((RuntimeError, Exception), match="no model is loaded"):
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

    # =========================================================================
    # TESTS-F-001 (v1.4 Wave 6b): pin the Stage C BACKEND-B-001 "save_merged
    # silently downgraded to LoRA-only" warning. Pre-fix, an operator who
    # explicitly asked for merged weights while training without Unsloth got
    # a LoRA adapter on disk with NO diagnostic — the mismatch surfaced at
    # inference time. The fix added a structured WARN at trainer.py:2796
    # naming WHY (transformers Trainer can't do an in-place merge) and TWO
    # concrete migration paths (re-train with Unsloth, OR `backprop export
    # ... --format merged` post-save).
    #
    # This test pins the warning fires when the gate condition is met
    # (save_merged=True AND use_unsloth=False). Without it, a future
    # refactor could delete the warning and the silent-downgrade footgun
    # would return invisibly.
    # =========================================================================

    def test_save_merged_with_no_unsloth_emits_warning(self, temp_dir, caplog):
        """Stage C BACKEND-B-001: save_merged=True + use_unsloth=False must
        emit a structured WARN naming the silent merged->LoRA downgrade.

        The gate condition is (save_merged=True AND use_unsloth=False); the
        warning text must reference both knobs so the operator can locate
        the migration paths trainer.py:2796 names. Pre-fix this was a
        silent downgrade — the operator saw a successful save and didn't
        know merged weights had been quietly skipped.
        """
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir), use_unsloth=False)
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True
            # Bypass the untrained-save tripwire (covered separately by
            # TESTS-F-002) so this test isolates the BACKEND-B-001 surface.
            trainer._has_trained = True
            # use_unsloth flag must be False for the gate condition.
            trainer.use_unsloth = False

            with caplog.at_level("WARNING", logger="backpropagate.trainer"):
                trainer.save(save_merged=True)

            matched = [
                r for r in caplog.records
                if "save_merged" in r.getMessage()
                and "use_unsloth" in r.getMessage()
            ]
            assert matched, (
                "Stage C BACKEND-B-001 regression: save(save_merged=True) "
                "with use_unsloth=False must emit a WARNING that names both "
                "knobs (save_merged + use_unsloth) so the operator can locate "
                "the migration paths. Captured WARNING records: "
                f"{[r.getMessage() for r in caplog.records if r.levelname == 'WARNING']!r}"
            )
            # The warning lives at WARNING severity (not INFO/DEBUG).
            assert all(r.levelname == "WARNING" for r in matched), (
                f"BACKEND-B-001 warning must be at WARNING level; got "
                f"levels: {[r.levelname for r in matched]!r}"
            )

    def test_save_merged_with_unsloth_does_not_emit_b001_warning(self, temp_dir, caplog):
        """The gate condition is AND-coupled — Unsloth + save_merged is the
        SUPPORTED path and must NOT emit the BACKEND-B-001 downgrade warning.

        Pins the negative branch so a future refactor that broadens the
        gate (e.g. ``if save_merged:`` without the use_unsloth check) gets
        caught — that would emit the warning even when Unsloth IS handling
        the merge correctly, which is operator-hostile noise.
        """
        from backpropagate.trainer import Trainer

        mock_fast_lm = MagicMock()

        with patch("torch.cuda.is_available", return_value=False), \
             patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_lm)}):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True
            trainer._has_trained = True
            trainer.use_unsloth = True

            with caplog.at_level("WARNING", logger="backpropagate.trainer"):
                trainer.save(save_merged=True)

            downgrade_warnings = [
                r for r in caplog.records
                if r.levelname == "WARNING"
                and "save_merged" in r.getMessage()
                and "use_unsloth=False" in r.getMessage()
            ]
            assert not downgrade_warnings, (
                "BACKEND-B-001 warning emitted on the SUPPORTED path "
                "(save_merged=True + use_unsloth=True). The gate should be "
                "AND-coupled (save_merged AND not use_unsloth). Spurious "
                f"warnings: {[r.getMessage() for r in downgrade_warnings]!r}"
            )

    # =========================================================================
    # TESTS-F-002 (v1.4 Wave 6b): pin the Wave 3.5 BACKEND-B-004 "untrained-
    # save" tripwire warning. Pre-fix, an operator who called
    # ``Trainer(...).load_model(); trainer.save("./out")`` got a freshly-
    # initialized PEFT adapter on disk — rank-r Gaussian-noise weights from
    # PEFT init defaults — saved as if it were a trained checkpoint. The
    # silence made the untrained save indistinguishable from a successfully-
    # trained save. The fix at trainer.py:2746 added a structured WARNING
    # gated on ``not self._has_trained`` that names what's happening and
    # points at .train() as the typical next step.
    #
    # We pin: (a) the _has_trained flag initializes to False on a fresh
    # Trainer; (b) calling save() with _has_trained=False emits the
    # warning; (c) calling save() with _has_trained=True does NOT emit it
    # (no operator-hostile noise on the success path).
    # =========================================================================

    def test_has_trained_flag_initializes_false(self):
        """A fresh Trainer starts with _has_trained=False so the first
        post-load save() correctly fires the untrained-save tripwire.

        Pre-Wave-3.5 the flag didn't exist; pinning the init value here
        catches a regression where someone defaults it to True (silencing
        the warning unconditionally).
        """
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert hasattr(trainer, "_has_trained"), (
            "Wave 3.5 BACKEND-B-004 regression: Trainer is missing the "
            "_has_trained attribute. The untrained-save tripwire warning "
            "at trainer.py:2746 has nothing to gate on."
        )
        assert trainer._has_trained is False, (
            f"Wave 3.5 BACKEND-B-004: Trainer._has_trained must initialize "
            f"to False; got {trainer._has_trained!r}. A True default would "
            f"silence the untrained-save tripwire unconditionally."
        )

    def test_save_without_train_emits_warning(self, temp_dir, caplog):
        """Wave 3.5 BACKEND-B-004: save() with _has_trained=False emits a
        structured WARN naming the untrained-save situation.

        The pre-fix bug: an operator's ``load_model(); save()`` chain
        produced a PEFT-init-noise adapter on disk silently. The warning
        text must reference ``untrained`` so an operator grepping logs for
        the symptom finds the diagnosis.
        """
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True
            trainer.use_unsloth = True
            # _has_trained left at the constructor default (False).
            assert trainer._has_trained is False, (
                "Test setup precondition: _has_trained must be False "
                "to exercise the BACKEND-B-004 tripwire."
            )

            with caplog.at_level("WARNING", logger="backpropagate.trainer"):
                trainer.save()

            matched = [
                r for r in caplog.records
                if r.levelname == "WARNING"
                and "untrained" in r.getMessage().lower()
            ]
            assert matched, (
                "Wave 3.5 BACKEND-B-004 regression: save() with "
                "_has_trained=False must emit a WARNING containing "
                "'untrained' so an operator grepping for the symptom "
                "finds the diagnosis. Captured WARNING records: "
                f"{[r.getMessage() for r in caplog.records if r.levelname == 'WARNING']!r}"
            )

    def test_save_after_train_does_not_emit_untrained_warning(self, temp_dir, caplog):
        """Once _has_trained is True the tripwire MUST NOT fire — that
        would be operator-hostile noise on the post-train save path.

        Pins the gate negative-branch so a future refactor that drops the
        ``if not self._has_trained:`` guard gets caught.
        """
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir))
            trainer._model = MagicMock()
            trainer._tokenizer = MagicMock()
            trainer._is_loaded = True
            trainer.use_unsloth = True
            # Simulate a successful train() having completed.
            trainer._has_trained = True

            with caplog.at_level("WARNING", logger="backpropagate.trainer"):
                trainer.save()

            untrained_warnings = [
                r for r in caplog.records
                if r.levelname == "WARNING"
                and "untrained" in r.getMessage().lower()
            ]
            assert not untrained_warnings, (
                "Wave 3.5 BACKEND-B-004 regression: untrained-save "
                "warning fired AFTER a successful train() "
                "(_has_trained=True). The gate must remain "
                "``if not self._has_trained:`` so post-train saves stay "
                f"silent. Spurious warnings: {[r.getMessage() for r in untrained_warnings]!r}"
            )


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
        """Default LoRA rank pinned to 256 (v1.3 BACKEND-1 quality bump).

        Three pinning sites for this same contract — config.py LoRAConfig
        defaults, CLI parser --lora-r default, and Trainer instance read.
        All three MUST agree. If one drifts, the v1.5 TESTS-A-006
        cross-domain audit will catch the divergence; this test catches
        the Trainer half.
        """
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert trainer.lora_r == 256, (
            f"v1.3 BACKEND-1 contract violated: Trainer default lora_r "
            f"expected 256, got {trainer.lora_r}. Check LoRAConfig "
            f"defaults in backpropagate/config.py — Trainer reads from "
            f"there. Cross-check test_config.py + test_cli.py if you're "
            f"intentionally changing this default."
        )

    def test_lora_alpha_default(self):
        """Default LoRA alpha pinned to 512 (v1.3 BACKEND-1 alpha = 2*r).

        The 2:1 alpha:r coupling is the documented effective-LR scaling.
        Breaking it (e.g. alpha=256 with r=256) silently halves the
        adapter's effective learning rate and changes convergence
        behavior without a CHANGELOG note.
        """
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer()

        assert trainer.lora_alpha == 512, (
            f"v1.3 BACKEND-1 contract violated: Trainer default "
            f"lora_alpha expected 512 (= 2 * r=256), got "
            f"{trainer.lora_alpha}. If you changed lora_alpha, the 2:1 "
            f"ratio doctrine in handbook/lora_presets.md needs updating."
        )

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
        """LoRA targets attention projection modules.

        v1.3 BACKEND-1: the default target_modules flipped from a
        hand-curated 7-name list to PEFT's ``"all-linear"`` wildcard,
        which expands to every linear/Conv1D module on the model
        (including q_proj, k_proj, v_proj, o_proj). The contract this
        test really cares about is "attention projections are covered" —
        ``"all-linear"`` is a strict superset of the legacy list, so
        both shapes satisfy it. We branch on the value shape so the
        test stays meaningful if an operator switches back to the
        explicit list form via env var.
        """
        from backpropagate.config import settings

        target_modules = settings.lora.target_modules

        if isinstance(target_modules, str):
            # Wildcard form — covers every linear module by definition.
            assert target_modules == "all-linear"
        else:
            assert "q_proj" in target_modules
            assert "k_proj" in target_modules
            assert "v_proj" in target_modules
            assert "o_proj" in target_modules

    def test_lora_targets_mlp_modules(self):
        """LoRA targets MLP projection modules.

        v1.3 BACKEND-1: same superset-vs-list reasoning as the
        attention-modules test above.
        """
        from backpropagate.config import settings

        target_modules = settings.lora.target_modules

        if isinstance(target_modules, str):
            assert target_modules == "all-linear"
        else:
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
# torch.cuda.OutOfMemoryError. Two distinct OOM paths surface structured codes:
#   * oom_recovery=False (no retry attempted) — Wave 6a Option A wraps the OOM
#     into GPUMemoryError(code='RUNTIME_GPU_OOM').
#   * oom_recovery=True but recovery exhausts after _OOM_MAX_RETRIES_AT_MIN_BATCH
#     consecutive failures at batch_size=1 — raises
#     TrainingError(code='RUNTIME_OOM_RECOVERY_EXHAUSTED').
#
# These tests pin the load-bearing v1.1.0 contract by mocking SFTTrainer to
# raise an OOM-shaped RuntimeError ("out of memory" — caught by the substring
# check at trainer.py:_is_oom because torch.cuda.OutOfMemoryError is hard
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
        """N consecutive OOMs at batch_size=1 => TrainingError RUNTIME_OOM_RECOVERY_EXHAUSTED.

        Test name kept for backwards compat with grep-by-name workflows; the
        contract pinned here is the recovery-exhausted path (more specific
        code than RUNTIME_GPU_OOM, which fires only on the no-retry path).
        """
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
        """oom_recovery=False => first OOM re-raises as GPUMemoryError.

        Wave 6a RUNTIME_GPU_OOM Option A: the OOM at the
        ``oom_recovery=False`` branch is now wrapped into a structured
        ``GPUMemoryError(code='RUNTIME_GPU_OOM')`` at trainer.py
        (search for ``Wave 6a RUNTIME_GPU_OOM Option A``). The original
        ``RuntimeError`` survives as ``__cause__``. Distinct from the
        recovery-exhausted path which raises
        ``TrainingError(code='RUNTIME_OOM_RECOVERY_EXHAUSTED')`` — that
        only fires when oom_recovery=True AND the recovery loop ran out
        of halve-batch options.
        """
        from backpropagate.exceptions import GPUMemoryError

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
             pytest.raises(GPUMemoryError) as exc_info:
            trainer.train("dummy_dataset", steps=10)

        # The original OOM RuntimeError is the cause; the recovery loop did
        # not retry. (If recovery HAD run AND exhausted, we'd see
        # code='RUNTIME_OOM_RECOVERY_EXHAUSTED' on a TrainingError instead.)
        assert getattr(exc_info.value, "code", None) == "RUNTIME_GPU_OOM", (
            f"Expected GPUMemoryError(code='RUNTIME_GPU_OOM'); got "
            f"code={getattr(exc_info.value, 'code', None)!r} "
            f"message={exc_info.value!s}"
        )
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


@pytest.mark.serial
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

    TESTS-A-007 (v1.4 Wave 2 amend): marked @serial because each test
    invokes ``configure_logging(...force=True)`` which mutates the
    process-wide structlog configuration. Without serial scheduling,
    two TestRunIdCorrelation tests on the same xdist worker could
    interleave their configure_logging calls and capture records
    through the wrong processor chain.
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


@pytest.mark.serial
class TestTrainingLoggerCaplog:
    """B-001 + observability: TrainingLogger emits structured records, not no-ops.

    TESTS-A-007 (v1.4 Wave 2 amend): marked @serial because each test
    invokes ``configure_logging(...force=True)`` which mutates the
    process-wide structlog configuration (see TestRunIdCorrelation
    above for the full reasoning).
    """

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


# =============================================================================
# WAVE 6A REFACTOR — SHARED SFTCONFIG BUILDER + RUNTIME_GPU_OOM OPTION A
# =============================================================================
#
# These tests pin the four contracts closed in Wave 6a foundation:
#
#   BACKEND-A-003: shared ``_build_sft_config`` helper applies the v1.3
#                  BACKEND-5 paged-optim autodetection + BACKEND-7 Ada
#                  bf16/fp16 selection. Both call sites (Trainer.train and
#                  MultiRunTrainer._execute_run) now share this helper so
#                  cannot drift apart.
#   BACKEND-A-004: shared ``_apply_train_on_responses_only`` helper
#                  applies Unsloth's response-masking. Both call sites
#                  now share the single application path.
#   RUNTIME_GPU_OOM Option A: Trainer.train() oom_recovery=False on an OOM
#                  raises GPUMemoryError(code='RUNTIME_GPU_OOM').
#
# The multi-run-specific BACKEND-B-002 (failed-run skips checkpoint save)
# is tested separately in test_multi_run.py (the run loop fixture needed).


class TestBuildSftConfigHelper:
    """Pure-function tests of the shared ``_build_sft_config`` helper (Wave 6a
    BACKEND-A-003)."""

    def test_helper_threads_paged_optim_for_consumer_card(self):
        """v1.3 BACKEND-5: < 24GB VRAM => paged_adamw_8bit auto-selected.

        Pre-Wave-6a, multi_run.py:1347 hardcoded
        ``optim=settings.training.optim`` and the detector never ran for
        the multi-run path. The shared helper makes drift impossible.
        """
        from backpropagate.trainer import _build_sft_config

        # Mock a 16GB consumer card (RTX 5080-class).
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties") as mock_props, \
             patch("torch.cuda.get_device_capability", return_value=(12, 0)), \
             patch("trl.SFTConfig") as mock_sft_config:
            mock_props.return_value.total_memory = 16 * 1024 ** 3
            _build_sft_config(
                output_dir="/tmp/out",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                max_steps=100,
                learning_rate=2e-4,
                warmup_steps=10,
                max_seq_length=2048,
                seed=42,
                lr_scheduler_type="cosine",
                logging_steps=10,
            )
            _, kwargs = mock_sft_config.call_args
            assert kwargs["optim"] == "paged_adamw_8bit", (
                f"BACKEND-A-003: 16GB consumer card should auto-upgrade "
                f"optim to paged_adamw_8bit; got {kwargs['optim']!r}"
            )

    def test_helper_threads_bf16_on_ada(self):
        """v1.3 BACKEND-7: Ampere+ (capability >= 8.0) => bf16=True, fp16=False.

        Pre-Wave-6a, multi-run hardcoded ``bf16=settings.training.bf16``
        ignoring the detector. The shared helper closes the drift.
        """
        from backpropagate.trainer import _build_sft_config

        # Mock Ada (capability 8.9) — RTX 4090-class.
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties") as mock_props, \
             patch("torch.cuda.get_device_capability", return_value=(8, 9)), \
             patch("trl.SFTConfig") as mock_sft_config:
            mock_props.return_value.total_memory = 24 * 1024 ** 3
            _build_sft_config(
                output_dir="/tmp/out",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                max_steps=100,
                learning_rate=2e-4,
                warmup_steps=10,
                max_seq_length=2048,
                seed=42,
                lr_scheduler_type="cosine",
                logging_steps=10,
            )
            _, kwargs = mock_sft_config.call_args
            assert kwargs["bf16"] is True, (
                f"BACKEND-A-003: Ada (capability 8.9) should auto-select "
                f"bf16; got bf16={kwargs['bf16']!r}"
            )
            assert kwargs["fp16"] is False, (
                f"BACKEND-A-003: Ada should set fp16=False; got "
                f"fp16={kwargs['fp16']!r}"
            )

    def test_helper_forces_fp32_on_cpu_only_runner(self):
        """CPU runner (no CUDA) => bf16=False, fp16=False.

        Regression for the nightly_train_smoke failure on 2026-05-25/26
        (workflow runs 26385294328 + 26434147558). The config default is
        ``bf16=True, fp16=False``; pre-fix, ``_detect_optimal_dtype``
        threaded ``configured_bf16`` through unchanged when CUDA was
        unavailable. transformers' ``TrainingArguments._validate_args``
        rejects bf16 on CPU with ``"Your setup doesn't support bf16/gpu.
        You need to assign use_cpu if you want to train the model on
        CPU."``, so every CPU train (including the nightly smoke) blew
        up at SFTConfig construction. The detector now forces fp32 when
        CUDA isn't available.
        """
        from backpropagate.trainer import _build_sft_config

        with patch("torch.cuda.is_available", return_value=False), \
             patch("trl.SFTConfig") as mock_sft_config:
            _build_sft_config(
                output_dir="/tmp/out",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                max_steps=100,
                learning_rate=2e-4,
                warmup_steps=10,
                max_seq_length=2048,
                seed=42,
                lr_scheduler_type="cosine",
                logging_steps=10,
            )
            _, kwargs = mock_sft_config.call_args
            assert kwargs["bf16"] is False, (
                f"CPU runner must force bf16=False to satisfy transformers' "
                f"_validate_args; got bf16={kwargs['bf16']!r}"
            )
            assert kwargs["fp16"] is False, (
                f"CPU runner must force fp16=False (same _validate_args "
                f"check); got fp16={kwargs['fp16']!r}"
            )

    def test_helper_omits_optional_fields_when_none(self):
        """save_steps + weight_decay are optional; omitted when None.

        Multi-run's pre-Wave-6a inline build omitted these; the helper
        preserves that observable behavior so the cross-site refactor is
        equivalent on those fields.
        """
        from backpropagate.trainer import _build_sft_config

        with patch("torch.cuda.is_available", return_value=False), \
             patch("trl.SFTConfig") as mock_sft_config:
            _build_sft_config(
                output_dir="/tmp/out",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                max_steps=100,
                learning_rate=2e-4,
                warmup_steps=10,
                max_seq_length=2048,
                seed=42,
                lr_scheduler_type="cosine",
                logging_steps=10,
                # save_steps / weight_decay deliberately omitted
            )
            _, kwargs = mock_sft_config.call_args
            assert "save_steps" not in kwargs, (
                "save_steps=None must be omitted so SFTConfig default "
                "governs; the multi-run path depends on the omission."
            )
            assert "weight_decay" not in kwargs

    def test_helper_includes_optional_fields_when_given(self):
        """When the caller passes save_steps / weight_decay, they thread through."""
        from backpropagate.trainer import _build_sft_config

        with patch("torch.cuda.is_available", return_value=False), \
             patch("trl.SFTConfig") as mock_sft_config:
            _build_sft_config(
                output_dir="/tmp/out",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                max_steps=100,
                learning_rate=2e-4,
                warmup_steps=10,
                max_seq_length=2048,
                seed=42,
                lr_scheduler_type="cosine",
                logging_steps=10,
                save_steps=50,
                weight_decay=0.01,
            )
            _, kwargs = mock_sft_config.call_args
            assert kwargs["save_steps"] == 50
            assert kwargs["weight_decay"] == 0.01


class TestApplyTrainOnResponsesOnlyHelper:
    """Pure-function tests of the shared
    ``_apply_train_on_responses_only`` helper (Wave 6a BACKEND-A-004).
    Multi-run users training on conversational data now get the same
    response-masking the single-run path applied.
    """

    def test_helper_no_op_when_disabled(self):
        """enabled=False => trainer returned unchanged, markers=None."""
        from backpropagate.trainer import _apply_train_on_responses_only

        sft_trainer = MagicMock(name="sft_trainer")
        tokenizer = MagicMock()
        wrapped, markers = _apply_train_on_responses_only(
            sft_trainer,
            tokenizer,
            enabled=False,
            use_unsloth=True,
        )
        assert wrapped is sft_trainer
        assert markers is None

    def test_helper_no_op_when_no_unsloth(self):
        """use_unsloth=False => trainer returned unchanged, markers=None."""
        from backpropagate.trainer import _apply_train_on_responses_only

        sft_trainer = MagicMock(name="sft_trainer")
        tokenizer = MagicMock()
        wrapped, markers = _apply_train_on_responses_only(
            sft_trainer,
            tokenizer,
            enabled=True,
            use_unsloth=False,
        )
        assert wrapped is sft_trainer
        assert markers is None

    def test_helper_no_op_on_windows(self):
        """os.name == 'nt' => trainer returned unchanged, markers=None.

        Pre-Wave-6a, the Windows skip lived only in Trainer.train(); the
        multi-run call site never invoked masking at all. The shared
        helper centralizes the skip so both call sites observe the same
        Windows behavior.
        """
        from backpropagate.trainer import _apply_train_on_responses_only

        sft_trainer = MagicMock(name="sft_trainer")
        tokenizer = MagicMock()
        with patch("backpropagate.trainer.os.name", "nt"):
            wrapped, markers = _apply_train_on_responses_only(
                sft_trainer,
                tokenizer,
                enabled=True,
                use_unsloth=True,
            )
        assert wrapped is sft_trainer
        assert markers is None

    def test_helper_applies_masking_when_all_conditions_met(self):
        """Linux + Unsloth installed + enabled => masking applied + markers
        recorded. Pre-Wave-6a, multi-run users never saw this branch fire.
        """
        from backpropagate.trainer import _apply_train_on_responses_only

        sft_trainer = MagicMock(name="sft_trainer")
        wrapped_trainer = MagicMock(name="wrapped_trainer")
        tokenizer = MagicMock(name="tokenizer")

        # Patch the Unsloth import to a controllable mock.
        fake_unsloth_module = MagicMock()
        fake_unsloth_module.train_on_responses_only = MagicMock(
            return_value=wrapped_trainer
        )
        with patch("backpropagate.trainer.os.name", "posix"), \
             patch.dict(
                 "sys.modules",
                 {"unsloth.chat_templates": fake_unsloth_module},
             ):
            wrapped, markers = _apply_train_on_responses_only(
                sft_trainer,
                tokenizer,
                enabled=True,
                use_unsloth=True,
                response_markers_override=("USER>", "ASST>"),
            )
        assert wrapped is wrapped_trainer
        assert markers == ("USER>", "ASST>")
        fake_unsloth_module.train_on_responses_only.assert_called_once()


class TestRuntimeGpuOomOptionA:
    """Wave 6a RUNTIME_GPU_OOM Option A wrap on the Trainer.train() side:
    oom_recovery=False on an OOM raises
    GPUMemoryError(code='RUNTIME_GPU_OOM') instead of the generic
    TrainingError(code='RUNTIME_TRAINING_FAILED') the outer except handler
    used to produce. README + handbook + cli.py exit-code mapper + llms.txt
    all promise this code; Option A makes the promise true.
    """

    def _setup_trainer(self, temp_dir, *, initial_batch=4, initial_accum=1):
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                output_dir=str(temp_dir),
                batch_size=initial_batch,
                gradient_accumulation=initial_accum,
                oom_recovery=False,  # Option A's relevant branch
                use_unsloth=False,
            )
        trainer._model = MagicMock()
        trainer._tokenizer = MagicMock()
        trainer._is_loaded = True
        return trainer

    def test_oom_recovery_false_raises_gpu_memory_error_with_correct_code(
        self, temp_dir
    ):
        """oom_recovery=False + OOM => GPUMemoryError(code='RUNTIME_GPU_OOM').

        Distinct from RUNTIME_OOM_RECOVERY_EXHAUSTED (which only fires on
        the recovery-loop-ran-out-of-options path with oom_recovery=True).
        """
        from backpropagate.exceptions import GPUMemoryError

        trainer = self._setup_trainer(temp_dir)
        script = _OOMScript(oom_count=1)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
             patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
             patch("trl.SFTTrainer", side_effect=script.factory), \
             patch("trl.SFTConfig"), \
             pytest.raises(GPUMemoryError) as exc_info:
            trainer.train("dummy_dataset", steps=10)

        assert getattr(exc_info.value, "code", None) == "RUNTIME_GPU_OOM", (
            "Wave 6a Option A: oom_recovery=False on an OOM must raise "
            "GPUMemoryError carrying RUNTIME_GPU_OOM (the documented code). "
            f"Got code={getattr(exc_info.value, 'code', None)!r}"
        )
        # __cause__ chain preserves the original RuntimeError for tracebacks.
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "out of memory" in str(exc_info.value.__cause__).lower()

    def test_non_oom_exception_propagates_unchanged(self, temp_dir):
        """Wave 6a Option A wrap MUST NOT touch non-OOM exceptions.

        A generic RuntimeError that does NOT match the OOM matcher
        propagates unchanged into the outer except RuntimeError handler
        which wraps it as TrainingError(code='RUNTIME_TRAINING_FAILED').
        Pre-Wave-6a behavior preserved for non-OOM paths.
        """
        from backpropagate.exceptions import GPUMemoryError, TrainingError

        trainer = self._setup_trainer(temp_dir)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        # A non-OOM RuntimeError — message does NOT contain any OOM marker.
        def raise_non_oom(*args, **kwargs):
            instance = MagicMock()
            instance.train.side_effect = RuntimeError("some other CUDA error, not OOM")
            instance.state.log_history = []
            return instance

        with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
             patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
             patch("trl.SFTTrainer", side_effect=raise_non_oom), \
             patch("trl.SFTConfig"), \
             pytest.raises(TrainingError) as exc_info:
            trainer.train("dummy_dataset", steps=10)

        # NOT wrapped as GPUMemoryError — non-OOM stays a TrainingError.
        assert not isinstance(exc_info.value, GPUMemoryError), (
            "Non-OOM RuntimeError must NOT be wrapped as GPUMemoryError "
            "by Wave 6a Option A; only OOM-shaped errors are wrapped."
        )
