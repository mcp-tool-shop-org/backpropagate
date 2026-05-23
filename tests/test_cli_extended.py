"""
Extended CLI tests for comprehensive coverage.

Covers:
- Color support detection edge cases
- Error handling in all commands
- Exception types with suggestions
- Windows-specific behavior
- UI command
"""

import argparse
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# COLOR SUPPORT DETECTION TESTS
# =============================================================================


class TestSupportsColorExtended:
    """Extended tests for _supports_color function."""

    def test_no_color_env_disables_colors(self):
        """NO_COLOR environment variable should disable colors."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            # Need to reimport to get fresh evaluation
            from backpropagate import cli

            result = cli._supports_color()
            assert result is False

    def test_force_color_env_enables_colors(self):
        """FORCE_COLOR environment variable should enable colors."""
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            # Remove NO_COLOR if present
            env = os.environ.copy()
            env.pop("NO_COLOR", None)
            env["FORCE_COLOR"] = "1"
            with patch.dict(os.environ, env, clear=True):
                from backpropagate import cli
                result = cli._supports_color()
                assert result is True

    def test_non_tty_stream_no_colors(self):
        """Non-TTY streams should not get colors."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = False

        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("NO_COLOR", None)
            env.pop("FORCE_COLOR", None)
            with patch.dict(os.environ, env, clear=True):
                with patch.object(sys, 'stdout', mock_stdout):
                    from backpropagate import cli
                    result = cli._supports_color()
                    assert result is False

    def test_missing_isatty_attribute(self):
        """Streams without isatty should return False."""
        mock_stdout = MagicMock(spec=[])  # No isatty attribute

        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("NO_COLOR", None)
            env.pop("FORCE_COLOR", None)
            with patch.dict(os.environ, env, clear=True):
                with patch.object(sys, 'stdout', mock_stdout):
                    from backpropagate import cli
                    result = cli._supports_color()
                    assert result is False

    def test_windows_with_wt_session(self):
        """Windows Terminal (WT_SESSION) should enable colors."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = True

        with patch.dict(os.environ, {"WT_SESSION": "1"}, clear=False):
            with patch.object(sys, 'stdout', mock_stdout):
                with patch.object(os, 'name', 'nt'):
                    from backpropagate import cli
                    result = cli._supports_color()
                    assert result is True

    def test_windows_with_term(self):
        """Windows with TERM should enable colors."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = True

        with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False):
            with patch.object(sys, 'stdout', mock_stdout):
                with patch.object(os, 'name', 'nt'):
                    from backpropagate import cli
                    result = cli._supports_color()
                    assert result is True

    def test_windows_without_term_or_wt(self):
        """Windows without TERM or WT_SESSION should disable colors."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = True

        env = {"PATH": "/usr/bin"}  # No TERM, no WT_SESSION
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys, 'stdout', mock_stdout):
                with patch.object(os, 'name', 'nt'):
                    from backpropagate import cli
                    result = cli._supports_color()
                    # Windows without TERM or WT_SESSION should return False
                    assert result is False


# =============================================================================
# TRAIN COMMAND ERROR HANDLING TESTS
# =============================================================================


class TestCmdTrainErrorHandling:
    """Error handling tests for cmd_train."""

    def test_dataset_error_with_suggestion(self, capsys, tmp_path):
        """DatasetError with suggestion field displays it."""
        from backpropagate.cli import cmd_train
        from backpropagate.exceptions import DatasetError

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = DatasetError(
            message="Invalid format",
            suggestion="Try converting to JSONL format"
        )

        with patch("backpropagate.trainer.Trainer", return_value=mock_trainer), \
             patch("backpropagate.trainer.TrainingCallback"):
            args = argparse.Namespace(
                data="test_data.jsonl",
                model="test-model",
                steps=10,
                samples=None,
                batch_size="auto",
                lr=2e-4,
                lora_r=16,
                output=str(tmp_path),
                no_unsloth=True,
                verbose=False,
            )

            result = cmd_train(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Invalid format" in captured.err
            assert "Try converting to JSONL format" in captured.out

    def test_training_error_with_suggestion(self, capsys, tmp_path):
        """TrainingError with suggestion field displays it.

        Ship Gate B2: TrainingError is a runtime-level failure -> exit 2.
        """
        from backpropagate.cli import cmd_train
        from backpropagate.exceptions import TrainingError

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = TrainingError(
            message="Out of memory",
            suggestion="Try reducing batch size"
        )

        with patch("backpropagate.trainer.Trainer", return_value=mock_trainer), \
             patch("backpropagate.trainer.TrainingCallback"):
            args = argparse.Namespace(
                data="test_data.jsonl",
                model="test-model",
                steps=10,
                samples=None,
                batch_size="auto",
                lr=2e-4,
                lora_r=16,
                output=str(tmp_path),
                no_unsloth=True,
                verbose=False,
            )

            result = cmd_train(args)

            assert result == 2
            captured = capsys.readouterr()
            assert "Out of memory" in captured.err
            assert "reducing batch size" in captured.out

    def test_backpropagate_error_generic(self, capsys, tmp_path):
        """BackpropagateError base class handling.

        Ship Gate B2: generic structured BackpropagateError -> exit 2
        (runtime error) unless explicitly user-actionable.
        """
        from backpropagate.cli import cmd_train
        from backpropagate.exceptions import BackpropagateError

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = BackpropagateError(
            message="Generic error",
            suggestion="Check configuration"
        )

        with patch("backpropagate.trainer.Trainer", return_value=mock_trainer), \
             patch("backpropagate.trainer.TrainingCallback"):
            args = argparse.Namespace(
                data="test_data.jsonl",
                model="test-model",
                steps=10,
                samples=None,
                batch_size="auto",
                lr=2e-4,
                lora_r=16,
                output=str(tmp_path),
                no_unsloth=True,
                verbose=False,
            )

            result = cmd_train(args)

            assert result == 2
            captured = capsys.readouterr()
            assert "Generic error" in captured.err

    def test_dataset_error_verbose_traceback(self, capsys, tmp_path):
        """DatasetError with verbose shows traceback."""
        from backpropagate.cli import cmd_train
        from backpropagate.exceptions import DatasetError

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = DatasetError(message="Test error")

        with patch("backpropagate.trainer.Trainer", return_value=mock_trainer), \
             patch("backpropagate.trainer.TrainingCallback"):
            args = argparse.Namespace(
                data="test_data.jsonl",
                model="test-model",
                steps=10,
                samples=None,
                batch_size="auto",
                lr=2e-4,
                lora_r=16,
                output=str(tmp_path),
                no_unsloth=True,
                verbose=True,  # Verbose enabled
            )

            result = cmd_train(args)
            assert result == 1

    def test_training_error_verbose_traceback(self, capsys, tmp_path):
        """TrainingError with verbose shows traceback.

        Ship Gate B2: TrainingError -> exit 2 (runtime error).
        """
        from backpropagate.cli import cmd_train
        from backpropagate.exceptions import TrainingError

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = TrainingError(message="Training failed")

        with patch("backpropagate.trainer.Trainer", return_value=mock_trainer), \
             patch("backpropagate.trainer.TrainingCallback"):
            args = argparse.Namespace(
                data="test_data.jsonl",
                model="test-model",
                steps=10,
                samples=None,
                batch_size="auto",
                lr=2e-4,
                lora_r=16,
                output=str(tmp_path),
                no_unsloth=True,
                verbose=True,
            )

            result = cmd_train(args)
            assert result == 2


# =============================================================================
# MULTI-RUN COMMAND ERROR HANDLING TESTS
# =============================================================================


class TestCmdMultiRunErrorHandling:
    """Error handling tests for cmd_multi_run."""

    def test_backpropagate_error_with_suggestion(self, capsys, tmp_path):
        """BackpropagateError with suggestion in multi-run.

        Ship Gate B2: BackpropagateError -> exit 2 (runtime error).
        """
        from backpropagate.cli import cmd_multi_run
        from backpropagate.exceptions import BackpropagateError

        mock_trainer = MagicMock()
        mock_trainer.run.side_effect = BackpropagateError(
            message="Config invalid",
            suggestion="Check num_runs parameter"
        )

        with patch("backpropagate.multi_run.MultiRunTrainer", return_value=mock_trainer), \
             patch("backpropagate.multi_run.MultiRunConfig"), \
             patch("backpropagate.multi_run.MergeMode"):
            args = argparse.Namespace(
                data="test_data",
                model="test-model",
                runs=5,
                steps=100,
                samples=1000,
                merge_mode="slao",
                output=str(tmp_path),
                verbose=False,
            )

            result = cmd_multi_run(args)

            assert result == 2
            captured = capsys.readouterr()
            assert "Config invalid" in captured.err
            assert "num_runs" in captured.out

    def test_generic_exception_handling(self, capsys, tmp_path):
        """Generic exception in multi-run.

        Ship Gate B2: unexpected RuntimeError -> exit 2 (runtime error).
        """
        from backpropagate.cli import cmd_multi_run

        mock_trainer = MagicMock()
        mock_trainer.run.side_effect = RuntimeError("Unexpected error")

        with patch("backpropagate.multi_run.MultiRunTrainer", return_value=mock_trainer), \
             patch("backpropagate.multi_run.MultiRunConfig"), \
             patch("backpropagate.multi_run.MergeMode"):
            args = argparse.Namespace(
                data="test_data",
                model="test-model",
                runs=5,
                steps=100,
                samples=1000,
                merge_mode="slao",
                output=str(tmp_path),
                verbose=False,
            )

            result = cmd_multi_run(args)

            assert result == 2
            captured = capsys.readouterr()
            assert "Unexpected error" in captured.err

    def test_verbose_traceback(self, capsys, tmp_path):
        """Verbose mode shows traceback in multi-run."""
        from backpropagate.cli import cmd_multi_run

        mock_trainer = MagicMock()
        mock_trainer.run.side_effect = ValueError("Verbose test error")

        with patch("backpropagate.multi_run.MultiRunTrainer", return_value=mock_trainer), \
             patch("backpropagate.multi_run.MultiRunConfig"), \
             patch("backpropagate.multi_run.MergeMode"):
            args = argparse.Namespace(
                data="test_data",
                model="test-model",
                runs=5,
                steps=100,
                samples=1000,
                merge_mode="slao",
                output=str(tmp_path),
                verbose=True,
            )

            result = cmd_multi_run(args)
            assert result == 1

    def test_keyboard_interrupt(self, capsys, tmp_path):
        """KeyboardInterrupt in multi-run returns 130."""
        from backpropagate.cli import cmd_multi_run

        mock_trainer = MagicMock()
        mock_trainer.run.side_effect = KeyboardInterrupt()

        with patch("backpropagate.multi_run.MultiRunTrainer", return_value=mock_trainer), \
             patch("backpropagate.multi_run.MultiRunConfig"), \
             patch("backpropagate.multi_run.MergeMode"):
            args = argparse.Namespace(
                data="test_data",
                model="test-model",
                runs=5,
                steps=100,
                samples=1000,
                merge_mode="slao",
                output=str(tmp_path),
                verbose=False,
            )

            result = cmd_multi_run(args)

            assert result == 130
            captured = capsys.readouterr()
            assert "interrupted" in captured.out.lower()

    def test_on_run_complete_callback_invoked(self, capsys, tmp_path):
        """on_run_complete callback is invoked.

        Preserves the happy-path intent of this test: explicitly set
        failed_runs=0 so cmd_multi_run does not classify the result as
        partial success (exit 3). MagicMock auto-attributes are truthy,
        so leaving this unset triggers the `failed_runs > 0` branch.
        """
        from backpropagate.cli import cmd_multi_run

        mock_result = MagicMock()
        mock_result.total_runs = 3
        mock_result.final_loss = 0.25
        mock_result.total_duration_seconds = 180.0
        mock_result.final_checkpoint_path = str(tmp_path / "model")
        mock_result.failed_runs = 0

        mock_trainer = MagicMock()
        mock_trainer.run.return_value = mock_result

        # Track if callback was passed
        captured_callback = None

        def capture_trainer(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get('on_run_complete')
            return mock_trainer

        with patch("backpropagate.multi_run.MultiRunTrainer", side_effect=capture_trainer), \
             patch("backpropagate.multi_run.MultiRunConfig"), \
             patch("backpropagate.multi_run.MergeMode"):
            args = argparse.Namespace(
                data="test_data",
                model="test-model",
                runs=3,
                steps=100,
                samples=1000,
                merge_mode="slao",
                output=str(tmp_path),
                verbose=False,
            )

            result = cmd_multi_run(args)

            assert result == 0
            assert captured_callback is not None


# =============================================================================
# EXPORT COMMAND ERROR HANDLING TESTS
# =============================================================================


class TestCmdExportErrorHandling:
    """Error handling tests for cmd_export."""

    def test_path_traversal_blocked(self, capsys, tmp_path):
        """Paths with ../ are rejected."""
        from backpropagate.cli import cmd_export
        from backpropagate.security import PathTraversalError

        with patch("backpropagate.cli.safe_path") as mock_safe_path:
            mock_safe_path.side_effect = PathTraversalError("Path traversal detected")

            args = argparse.Namespace(
                model_path="../../../etc/passwd",
                format="lora",
                quantization="q4_k_m",
                output=None,
                ollama=False,
                ollama_name=None,
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Security error" in captured.err or "ERROR" in captured.err

    def test_invalid_format_rejected(self, capsys, tmp_path):
        """Unknown export formats raise error."""
        from backpropagate.cli import cmd_export

        # Create a real model directory
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}")

        args = argparse.Namespace(
            model_path=str(model_dir),
            format="invalid_format",
            quantization="q4_k_m",
            output=None,
            ollama=False,
            ollama_name=None,
            verbose=False,
        )

        result = cmd_export(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown format" in captured.err or "ERROR" in captured.err

    def test_export_error_with_suggestion(self, capsys, tmp_path):
        """ExportError with suggestion displays it.

        Ship Gate B2: ExportError -> exit 2 (runtime error).
        """
        from backpropagate.cli import cmd_export
        from backpropagate.exceptions import ExportError

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}")

        with patch("backpropagate.export.export_lora") as mock_export:
            mock_export.side_effect = ExportError(
                message="Export failed",
                suggestion="Check disk space"
            )

            args = argparse.Namespace(
                model_path=str(model_dir),
                format="lora",
                quantization="q4_k_m",
                output=None,
                ollama=False,
                ollama_name=None,
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 2
            captured = capsys.readouterr()
            assert "Export failed" in captured.err or "Export error" in captured.err

    def test_backpropagate_error_in_export(self, capsys, tmp_path):
        """BackpropagateError in export.

        Ship Gate B2: BackpropagateError -> exit 2 (runtime error).
        """
        from backpropagate.cli import cmd_export
        from backpropagate.exceptions import BackpropagateError

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}")

        with patch("backpropagate.export.export_lora") as mock_export:
            mock_export.side_effect = BackpropagateError(
                message="General error",
                suggestion="Try again"
            )

            args = argparse.Namespace(
                model_path=str(model_dir),
                format="lora",
                quantization="q4_k_m",
                output=None,
                ollama=False,
                ollama_name=None,
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 2

    def test_generic_exception_in_export(self, capsys, tmp_path):
        """Generic exception in export.

        Ship Gate B2: unexpected RuntimeError -> exit 2 (runtime error).
        """
        from backpropagate.cli import cmd_export

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}")

        with patch("backpropagate.export.export_lora") as mock_export:
            mock_export.side_effect = RuntimeError("Disk full")

            args = argparse.Namespace(
                model_path=str(model_dir),
                format="lora",
                quantization="q4_k_m",
                output=None,
                ollama=False,
                ollama_name=None,
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 2
            captured = capsys.readouterr()
            assert "Disk full" in captured.err

    def test_verbose_traceback_in_export(self, capsys, tmp_path):
        """Verbose mode shows traceback in export.

        Ship Gate B2: unexpected ValueError -> exit 2 (runtime error).
        """
        from backpropagate.cli import cmd_export

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}")

        with patch("backpropagate.export.export_lora") as mock_export:
            mock_export.side_effect = ValueError("Test error")

            args = argparse.Namespace(
                model_path=str(model_dir),
                format="lora",
                quantization="q4_k_m",
                output=None,
                ollama=False,
                ollama_name=None,
                verbose=True,
            )

            result = cmd_export(args)
            assert result == 2

    def test_ollama_registration_failure(self, capsys, tmp_path):
        """Ollama registration failure handled.

        Documented partial-success path: export succeeded, only the optional
        Ollama registration failed -> Ship Gate B2 exit code 3 (partial).
        """
        from backpropagate.cli import cmd_export

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}")

        mock_result = MagicMock()
        mock_result.path = tmp_path / "model.gguf"
        mock_result.size_mb = 4000.0
        mock_result.export_time_seconds = 120.0

        with patch("backpropagate.export.export_gguf", return_value=mock_result), \
             patch("backpropagate.export.register_with_ollama", return_value=False), \
             patch("backpropagate.trainer.load_model", return_value=(MagicMock(), MagicMock())):
            args = argparse.Namespace(
                model_path=str(model_dir),
                format="gguf",
                quantization="q4_k_m",
                output=None,
                ollama=True,
                ollama_name="test-model",
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 3
            captured = capsys.readouterr()
            assert "Failed to register" in captured.err or "Ollama" in captured.err

    def test_ollama_registration_success(self, capsys, tmp_path):
        """Ollama registration success."""
        from backpropagate.cli import cmd_export

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}")

        mock_result = MagicMock()
        mock_result.path = tmp_path / "model.gguf"
        mock_result.size_mb = 4000.0
        mock_result.export_time_seconds = 120.0

        with patch("backpropagate.export.export_gguf", return_value=mock_result), \
             patch("backpropagate.export.register_with_ollama", return_value=True), \
             patch("backpropagate.trainer.load_model", return_value=(MagicMock(), MagicMock())):
            args = argparse.Namespace(
                model_path=str(model_dir),
                format="gguf",
                quantization="q4_k_m",
                output=None,
                ollama=True,
                ollama_name="test-model",
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Registered with Ollama" in captured.out


# =============================================================================
# INFO COMMAND TESTS
# =============================================================================


class TestCmdInfoExtended:
    """Extended tests for cmd_info."""

    def test_no_gpu_shows_message(self, capsys):
        """Shows 'No GPU detected' when CUDA unavailable."""
        from backpropagate.cli import cmd_info

        with patch("backpropagate.feature_flags.get_gpu_info", return_value=None):
            args = argparse.Namespace(verbose=False)
            result = cmd_info(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "No GPU" in captured.out or "GPU" in captured.out

    def test_gpu_temperature_display(self, capsys):
        """GPU temperature displayed with color coding."""
        from backpropagate.cli import cmd_info

        mock_gpu_info = {
            "name": "RTX 5080",
            "vram_total_gb": 16.0,
            "vram_free_gb": 12.0,
        }

        mock_status = MagicMock()
        mock_status.temperature_c = 65.0

        with patch("backpropagate.feature_flags.get_gpu_info", return_value=mock_gpu_info), \
             patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            args = argparse.Namespace(verbose=False)
            result = cmd_info(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "RTX 5080" in captured.out

    def test_pynvml_import_error_handled(self, capsys):
        """Missing pynvml shows N/A for temperature."""
        from backpropagate.cli import cmd_info

        mock_gpu_info = {
            "name": "Test GPU",
            "vram_total_gb": 8.0,
            "vram_free_gb": 6.0,
        }

        with patch("backpropagate.feature_flags.get_gpu_info", return_value=mock_gpu_info), \
             patch("backpropagate.gpu_safety.get_gpu_status", side_effect=ImportError("pynvml")):
            args = argparse.Namespace(verbose=False)
            result = cmd_info(args)

            assert result == 0

    def test_temperature_read_exception(self, capsys):
        """Temperature read exception handled gracefully."""
        from backpropagate.cli import cmd_info

        mock_gpu_info = {
            "name": "Test GPU",
            "vram_total_gb": 8.0,
            "vram_free_gb": 6.0,
        }

        with patch("backpropagate.feature_flags.get_gpu_info", return_value=mock_gpu_info), \
             patch("backpropagate.gpu_safety.get_gpu_status", side_effect=RuntimeError("NVML error")):
            args = argparse.Namespace(verbose=False)
            result = cmd_info(args)

            assert result == 0


# =============================================================================
# UI COMMAND TESTS
# =============================================================================


class TestCmdUI:
    """Tests for cmd_ui command.

    v1.1.0 (2026-05-21): the Web UI migrated from Gradio (in-process launch)
    to Reflex (subprocess via ``reflex run``). The CLI-layer share+auth gate
    is preserved end-to-end — the tests below pin it through the new
    subprocess boundary instead of the old in-process ``launch()`` call.
    The mocking surface moved from ``backpropagate.ui.launch`` to
    ``backpropagate.cli.subprocess.run``.
    """

    @staticmethod
    def _mock_subprocess_result(returncode: int = 0) -> MagicMock:
        result = MagicMock()
        result.returncode = returncode
        return result

    def test_ui_import_error(self, capsys, monkeypatch):
        """Missing Reflex (no [ui] extra) shows helpful error."""
        import builtins

        from backpropagate import cli as cli_module

        args = argparse.Namespace(
            port=7860,
            share=False,
            auth=None,
            verbose=False,
        )

        # Simulate ImportError when cmd_ui tries to import reflex.
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "reflex":
                raise ImportError("No module named 'reflex'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = cli_module.cmd_ui(args)

        assert result == 1
        captured = capsys.readouterr()
        combined = (captured.err + captured.out).lower()
        assert "ui" in combined or "reflex" in combined or "install" in combined

    def test_auth_invalid_format(self, capsys):
        """Auth string (any shape) currently triggers refuse-to-start.

        FRONTEND-A-001 (v1.1.2 amend wave, 2026-05-22): the CLI now refuses
        to start when ``--auth`` is passed because the Reflex UI does not
        yet enforce the auth contract. The pre-fix code parsed the auth
        string and would have rejected "invalid_no_colon" with a friendlier
        "Invalid auth format" message; the new refuse-to-start runs BEFORE
        the format-validation path, so any ``--auth`` value — valid or not
        — raises ``BackpropagateError`` with code ``RUNTIME_UI_AUTH_NOT_ENFORCED``.

        When the Reflex auth middleware lands and ``ENFORCEMENT_AVAILABLE``
        flips to True, this test should be rewritten to exercise the
        ``validate_auth_shape`` failure path again.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth="invalid_no_colon",  # Missing colon
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            mock_run.assert_not_called()

    def test_auth_parsed_correctly(self, capsys):
        """Well-formed ``user:pass`` --auth currently refuses to start.

        FRONTEND-A-001: pre-fix this test verified that ``cmd_ui`` parsed
        a valid auth string into a tuple and exported BACKPROPAGATE_UI_AUTH
        for the subprocess. The refuse-to-start guard now fires before any
        of that runs. Reinstate the env-var assertion once the Reflex
        middleware enforces the contract.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth="testuser:testpass",
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            mock_run.assert_not_called()

    def test_auth_with_colon_in_password(self, capsys):
        """Colons-in-password auth currently refuses to start.

        FRONTEND-A-001: the shape of the credentials is irrelevant while
        the Reflex middleware is missing — every ``--auth`` invocation
        triggers RUNTIME_UI_AUTH_NOT_ENFORCED. Restore the original
        BACKPROPAGATE_UI_AUTH assertion once Phase 3 lands.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth="user:pass:with:colons",
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            mock_run.assert_not_called()

    def test_launch_success(self, capsys):
        """Successful UI launch via subprocess (no --auth, no --share).

        FRONTEND-A-001: pre-fix this test exercised the ``--share --auth
        user:password`` happy path. Both flags now trigger refuse-to-start,
        so this test now pins the only remaining happy path: a plain local
        UI launch with no share / no auth. The subprocess command-line
        construction assertion is preserved.
        """
        from backpropagate.cli import cmd_ui

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7862,
                share=False,
                auth=None,
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == 0
            mock_run.assert_called_once()
            # Check the constructed command line passed to reflex.
            cmd = mock_run.call_args.args[0]
            assert "reflex" in cmd
            assert "run" in cmd
            assert "--frontend-port" in cmd
            assert str(7862) in cmd

    # ------------------------------------------------------------------ #
    # FRONTEND-A-001 refuse-to-start contract (v1.1.2 amend wave).
    #
    # The three tests below pin the *new* contract directly: ``--auth`` and
    # ``--share`` each cause cmd_ui to raise BackpropagateError with the
    # structured ``RUNTIME_UI_AUTH_NOT_ENFORCED`` code BEFORE any subprocess
    # is spawned. The companion happy-path test verifies that the plain
    # local launch (no auth / no share) is still allowed through.
    # ------------------------------------------------------------------ #

    def test_cmd_ui_auth_refuses_to_start_with_runtime_code(self):
        """--auth alone (no --share) raises RUNTIME_UI_AUTH_NOT_ENFORCED."""
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth="user:pass",
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            assert "--auth" in str(excinfo.value)
            # subprocess must NOT have been launched.
            mock_run.assert_not_called()

    def test_cmd_ui_share_refuses_to_start(self):
        """--share alone (no --auth) raises RUNTIME_UI_AUTH_NOT_ENFORCED."""
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=True,
                auth=None,
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            assert "--share" in str(excinfo.value)
            mock_run.assert_not_called()

    def test_cmd_ui_no_auth_no_share_proceeds(self):
        """Without --auth or --share, cmd_ui proceeds and launches Reflex."""
        from backpropagate.cli import cmd_ui

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth=None,
                verbose=False,
            )

            # Should NOT raise; should reach the subprocess launch.
            result = cmd_ui(args)

            assert result == 0
            mock_run.assert_called_once()

    def test_launch_keyboard_interrupt(self, capsys):
        """KeyboardInterrupt during UI exits cleanly."""
        from backpropagate.cli import cmd_ui

        with patch("backpropagate.cli.subprocess.run", side_effect=KeyboardInterrupt()):
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth=None,
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "stopped" in captured.out.lower()

    def test_launch_exception(self, capsys):
        """Launch exception handled gracefully.

        Ship Gate B2: unexpected RuntimeError -> exit 2 (runtime error).
        """
        from backpropagate.cli import cmd_ui

        with patch("backpropagate.cli.subprocess.run", side_effect=RuntimeError("Port in use")):
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth=None,
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == 2
            captured = capsys.readouterr()
            assert "Port in use" in captured.err

    def test_launch_exception_verbose(self, capsys):
        """Launch exception with verbose shows traceback.

        Ship Gate B2: unexpected ValueError -> exit 2 (runtime error).
        """
        from backpropagate.cli import cmd_ui

        with patch("backpropagate.cli.subprocess.run", side_effect=ValueError("Test error")):
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth=None,
                verbose=True,
            )

            result = cmd_ui(args)

            assert result == 2

    # ------------------------------------------------------------------ #
    # SB-T-001: --share + --auth gate at the CLI layer.
    #
    # These tests pin the CLI-layer F-001 contract — the [INPUT_AUTH_REQUIRED]
    # structured prefix, the env-var opt-out, and the parsed-auth-string
    # hand-off — that has SURVIVED the Gradio→Reflex migration unchanged.
    # The mocking boundary moved from backpropagate.ui.launch to the
    # ``backpropagate.cli.subprocess.run`` subprocess call site.
    # ------------------------------------------------------------------ #

    def test_cmd_ui_share_without_auth_blocked_by_default(self, capsys, monkeypatch):
        """--share without --auth refuses to start with RUNTIME_UI_AUTH_NOT_ENFORCED.

        FRONTEND-A-001 (v1.1.2 amend wave): the SB-T-001 INPUT_AUTH_REQUIRED
        gate was upgraded to a hard refuse-to-start so both ``--share`` and
        ``--auth`` raise a structured BackpropagateError before any
        subprocess is spawned. The pre-fix exit-code-1 + stderr-prefix
        contract is preserved at the ``main()`` boundary (still surfaces a
        non-zero exit and a structured-prefix line) but the underlying
        ``cmd_ui`` raise is what tests now pin.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        monkeypatch.delenv("BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE", raising=False)

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=True,
                auth=None,
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            mock_run.assert_not_called()

    def test_cmd_ui_share_with_auth_allowed(self, capsys):
        """--share + --auth currently refuses to start.

        FRONTEND-A-001 (v1.1.2 amend wave): the pre-fix code parsed the
        credentials and launched the subprocess with BACKPROPAGATE_UI_AUTH
        exported. The Reflex UI never read that variable, so launching
        with --share + --auth advertised an auth contract the runtime did
        not deliver. The honest fix is to refuse-to-start until the Reflex
        middleware enforces the credentials end-to-end.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=True,
                auth="alice:secret123",
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            mock_run.assert_not_called()

    def test_cmd_ui_share_env_opt_out_no_longer_respected(self, capsys, monkeypatch):
        """BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false is now ignored.

        FRONTEND-A-001 (v1.1.2 amend wave): the env-var operator opt-out
        was deleted because the Reflex UI cannot enforce the auth contract
        even when the operator acknowledges the risk — the unauthenticated
        public URL would still serve every request without checking
        credentials. Setting the env-var to "false" no longer disables
        the gate; --share refuses to start regardless.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        monkeypatch.setenv("BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE", "false")

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=True,
                auth=None,
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            # Env opt-out used to flip this to exit-0; now it must still raise.
            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            mock_run.assert_not_called()

    def test_cmd_ui_share_env_opt_in_explicit_refuses(self, capsys, monkeypatch):
        """Env-var set to 'true' is the default behaviour — --share refuses.

        Parity with the env-unset case: --share triggers refuse-to-start
        regardless of the env-var value.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        monkeypatch.setenv("BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE", "true")

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=True,
                auth=None,
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            mock_run.assert_not_called()

    def test_cmd_ui_local_without_share_no_auth_required(self, capsys, monkeypatch):
        """Local UI (no --share) does NOT require --auth."""
        from backpropagate.cli import cmd_ui

        monkeypatch.delenv("BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE", raising=False)

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth=None,
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == 0
            mock_run.assert_called_once()


# =============================================================================
# TESTS-B-006 — ENFORCEMENT_AVAILABLE flipped path (post-middleware contract)
# =============================================================================
#
# The refuse-to-start tests above pin behavior while
# ``backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE`` is False (today's state
# — Reflex middleware hasn't landed yet). Stage B audit flagged that nothing
# pins the symmetric post-middleware contract: when the middleware DOES land
# and the flag flips True, ``--auth`` MUST flow through to subprocess launch
# instead of falsely refusing.
#
# Without these tests, a future commit that lands middleware + flips the flag
# has nothing telling it the refuse-to-start guard is now wrong. The guard
# becomes accidentally-permanent and operators stay locked out of auth even
# after the underlying contract is honoured. Pinning the True path now
# guarantees the flip will trip a failing test that someone has to look at.


class TestCmdUiEnforcementFlipped:
    """TESTS-B-006: post-middleware behaviour pinned in advance.

    Each test monkeypatches ``ENFORCEMENT_AVAILABLE = True`` on the
    ``backpropagate.ui_app.auth`` module and asserts ``cmd_ui`` no longer
    raises ``RUNTIME_UI_AUTH_NOT_ENFORCED`` for ``--auth``-bearing
    invocations. The ``--share`` gate is a separate guard with its own
    rationale (no Reflex tunnel) and stays refusing even when middleware
    enforces auth — pinned independently below.
    """

    @staticmethod
    def _mock_subprocess_result(returncode: int = 0) -> MagicMock:
        result = MagicMock()
        result.returncode = returncode
        return result

    def test_auth_proceeds_when_enforcement_available(self, monkeypatch):
        """--auth alone proceeds to subprocess launch once the flag flips.

        The middleware landing flips ``ENFORCEMENT_AVAILABLE = True``. From
        that point on, ``--auth user:pass`` must reach subprocess.run with
        the credentials handed to the Reflex child via
        ``BACKPROPAGATE_UI_AUTH``. The pre-fix test surface only covered
        the False path; this is the symmetric assertion.
        """
        from backpropagate.cli import cmd_ui

        # Flip the load-bearing flag on the source module — ``cmd_ui`` does
        # ``from .ui_app.auth import ENFORCEMENT_AVAILABLE`` lazily inside
        # the function body, so patching the *module attribute* (not a
        # snapshot inside cli.py) is what the import re-reads.
        monkeypatch.setattr(
            "backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE", True
        )

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth="alice:hunter2",
                verbose=False,
            )

            # Must NOT raise — the refuse-to-start guard is gated on
            # ENFORCEMENT_AVAILABLE being False.
            result = cmd_ui(args)

            assert result == 0
            mock_run.assert_called_once()

            # The credentials must reach the Reflex child via env-var.
            # subprocess.run is called as run(cmd, env=..., cwd=...).
            kwargs = mock_run.call_args.kwargs
            child_env = kwargs.get("env", {})
            assert child_env.get("BACKPROPAGATE_UI_AUTH") == "alice:hunter2", (
                "When ENFORCEMENT_AVAILABLE flips True, --auth must export "
                "BACKPROPAGATE_UI_AUTH to the Reflex subprocess so the "
                "middleware can read it. Got env={!r}".format(
                    {k: v for k, v in child_env.items()
                     if k.startswith("BACKPROPAGATE_")}
                )
            )

    def test_share_still_refuses_even_when_enforcement_available(self, monkeypatch):
        """--share is gated independently of ENFORCEMENT_AVAILABLE.

        FRONTEND-A-001 split into two guards: ``--auth`` blocks because
        middleware isn't wired; ``--share`` blocks because Reflex has no
        first-class tunnel equivalent. Flipping ENFORCEMENT_AVAILABLE only
        addresses the first gate. This test pins that ``--share`` keeps
        refusing post-middleware — operators must reach for SSH port-forwards
        or Cloudflare Tunnel either way.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        monkeypatch.setattr(
            "backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE", True
        )

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=True,
                auth=None,
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            # Same code as the no-middleware path — the gate's rationale
            # differs but the error envelope operators see is identical.
            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            mock_run.assert_not_called()

    def test_share_with_auth_still_refuses_when_enforcement_available(self, monkeypatch):
        """--share + --auth together still refuse, because --share is the blocker."""
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        monkeypatch.setattr(
            "backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE", True
        )

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                share=True,
                auth="alice:hunter2",
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            mock_run.assert_not_called()


# =============================================================================
# CONFIG COMMAND TESTS
# =============================================================================


class TestCmdConfigExtended:
    """Extended tests for cmd_config."""

    def test_config_set_not_implemented(self, capsys):
        """Config --set shows not implemented message."""
        from backpropagate.cli import cmd_config

        args = argparse.Namespace(
            show=False,
            set="key=value",
            reset=False,
            verbose=False,
        )

        result = cmd_config(args)

        assert result == 1
        captured = capsys.readouterr()
        msg = (captured.out + captured.err).lower()
        assert "planned" in msg or "environment" in msg or "not implemented" in msg

    def test_windows_config_shown_on_windows(self, capsys):
        """Windows settings shown on Windows platform."""
        from backpropagate.cli import cmd_config

        with patch.object(os, 'name', 'nt'):
            args = argparse.Namespace(
                show=False,
                set=None,
                reset=False,
                verbose=False,
            )

            result = cmd_config(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Windows" in captured.out

    def test_windows_config_hidden_on_posix(self, capsys):
        """Windows settings not shown on Linux/Mac."""
        from backpropagate.cli import cmd_config

        with patch.object(os, 'name', 'posix'):
            args = argparse.Namespace(
                show=False,
                set=None,
                reset=False,
                verbose=False,
            )

            result = cmd_config(args)

            assert result == 0
            # Windows section should not appear on non-Windows


# =============================================================================
# PROGRESS BAR EXTENDED TESTS
# =============================================================================


class TestProgressBarExtended:
    """Extended tests for ProgressBar."""

    def test_progress_bar_zero_total(self, capsys):
        """Progress bar handles zero total gracefully."""
        from backpropagate.cli import ProgressBar

        progress = ProgressBar(total=0, width=20)
        progress.update(0)  # Should not crash

    def test_progress_bar_with_suffix(self, capsys):
        """Progress bar displays suffix."""
        from backpropagate.cli import ProgressBar

        progress = ProgressBar(total=100, width=20, prefix="Test: ")
        progress.update(50, suffix="loss=0.5")

        captured = capsys.readouterr()
        assert "loss=0.5" in captured.out

    def test_progress_bar_completes_at_total(self, capsys):
        """Progress bar prints newline at completion."""
        from backpropagate.cli import ProgressBar

        progress = ProgressBar(total=100, width=20)
        progress.update(100)

        captured = capsys.readouterr()
        assert captured.out.endswith("\n")


# =============================================================================
# EXPORT FORMAT TESTS
# =============================================================================


class TestExportFormats:
    """Tests for different export formats."""

    def test_export_lora_format(self, capsys, tmp_path):
        """Export lora format."""
        from backpropagate.cli import cmd_export

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}")

        mock_result = MagicMock()
        mock_result.path = tmp_path / "lora_export"
        mock_result.size_mb = 100.0
        mock_result.export_time_seconds = 5.0

        with patch("backpropagate.export.export_lora", return_value=mock_result):
            args = argparse.Namespace(
                model_path=str(model_dir),
                format="lora",
                quantization="q4_k_m",
                output=str(tmp_path / "output"),
                ollama=False,
                ollama_name=None,
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 0

    def test_export_merged_format(self, capsys, tmp_path):
        """Export merged format."""
        from backpropagate.cli import cmd_export

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}")

        mock_result = MagicMock()
        mock_result.path = tmp_path / "merged_export"
        mock_result.size_mb = 8000.0
        mock_result.export_time_seconds = 60.0

        with patch("backpropagate.export.export_merged", return_value=mock_result), \
             patch("backpropagate.trainer.load_model", return_value=(MagicMock(), MagicMock())):
            args = argparse.Namespace(
                model_path=str(model_dir),
                format="merged",
                quantization="q4_k_m",
                output=str(tmp_path / "output"),
                ollama=False,
                ollama_name=None,
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 0

    def test_export_gguf_format(self, capsys, tmp_path):
        """Export gguf format."""
        from backpropagate.cli import cmd_export

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}")

        mock_result = MagicMock()
        mock_result.path = tmp_path / "model.gguf"
        mock_result.size_mb = 4000.0
        mock_result.export_time_seconds = 120.0

        with patch("backpropagate.export.export_gguf", return_value=mock_result), \
             patch("backpropagate.trainer.load_model", return_value=(MagicMock(), MagicMock())):
            args = argparse.Namespace(
                model_path=str(model_dir),
                format="gguf",
                quantization="q8_0",
                output=str(tmp_path / "output"),
                ollama=False,
                ollama_name=None,
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Export complete" in captured.out
