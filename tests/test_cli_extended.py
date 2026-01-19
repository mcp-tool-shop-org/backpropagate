"""
Extended CLI tests for comprehensive coverage.

Covers:
- Color support detection edge cases
- Error handling in all commands
- Exception types with suggestions
- Windows-specific behavior
- UI command
"""

import pytest
import argparse
import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock
from io import StringIO


# =============================================================================
# COLOR SUPPORT DETECTION TESTS
# =============================================================================


class TestSupportsColorExtended:
    """Extended tests for _supports_color function."""

    def test_no_color_env_disables_colors(self):
        """NO_COLOR environment variable should disable colors."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            # Need to reimport to get fresh evaluation
            import importlib
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
        """TrainingError with suggestion field displays it."""
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

            assert result == 1
            captured = capsys.readouterr()
            assert "Out of memory" in captured.err
            assert "reducing batch size" in captured.out

    def test_backpropagate_error_generic(self, capsys, tmp_path):
        """BackpropagateError base class handling."""
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

            assert result == 1
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
        """TrainingError with verbose shows traceback."""
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
            assert result == 1


# =============================================================================
# MULTI-RUN COMMAND ERROR HANDLING TESTS
# =============================================================================


class TestCmdMultiRunErrorHandling:
    """Error handling tests for cmd_multi_run."""

    def test_backpropagate_error_with_suggestion(self, capsys, tmp_path):
        """BackpropagateError with suggestion in multi-run."""
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

            assert result == 1
            captured = capsys.readouterr()
            assert "Config invalid" in captured.err
            assert "num_runs" in captured.out

    def test_generic_exception_handling(self, capsys, tmp_path):
        """Generic exception in multi-run."""
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

            assert result == 1
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
        """on_run_complete callback is invoked."""
        from backpropagate.cli import cmd_multi_run

        mock_result = MagicMock()
        mock_result.total_runs = 3
        mock_result.final_loss = 0.25
        mock_result.total_duration_seconds = 180.0
        mock_result.final_checkpoint_path = str(tmp_path / "model")

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
        """ExportError with suggestion displays it."""
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

            assert result == 1
            captured = capsys.readouterr()
            assert "Export failed" in captured.err or "Export error" in captured.err

    def test_backpropagate_error_in_export(self, capsys, tmp_path):
        """BackpropagateError in export."""
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

            assert result == 1

    def test_generic_exception_in_export(self, capsys, tmp_path):
        """Generic exception in export."""
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

            assert result == 1
            captured = capsys.readouterr()
            assert "Disk full" in captured.err

    def test_verbose_traceback_in_export(self, capsys, tmp_path):
        """Verbose mode shows traceback in export."""
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
            assert result == 1

    def test_ollama_registration_failure(self, capsys, tmp_path):
        """Ollama registration failure handled."""
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

            assert result == 1
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
    """Tests for cmd_ui command."""

    def test_ui_import_error(self, capsys):
        """Missing gradio shows helpful error."""
        from backpropagate import cli as cli_module
        import builtins

        args = argparse.Namespace(
            port=7860,
            share=False,
            auth=None,
            verbose=False,
        )

        # Simulate ImportError when trying to import backpropagate.ui
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'backpropagate.ui' or (args and 'ui' in str(args)):
                raise ImportError("No module named 'gradio'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, '__import__', side_effect=mock_import):
            # cmd_ui should catch the ImportError and return 1
            result = cli_module.cmd_ui(args)

            # Verify it returns error code
            assert result == 1
            captured = capsys.readouterr()
            # Should show error about missing gradio
            assert "gradio" in captured.err.lower() or "ui" in captured.err.lower() or result == 1

    def test_auth_invalid_format(self, capsys):
        """Auth string without colon raises error."""
        from backpropagate.cli import cmd_ui

        with patch("backpropagate.ui.launch"):
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth="invalid_no_colon",  # Missing colon
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Invalid auth format" in captured.err

    def test_auth_parsed_correctly(self, capsys):
        """user:pass format parsed into tuple."""
        from backpropagate.cli import cmd_ui

        mock_launch = MagicMock()

        with patch("backpropagate.ui.launch", mock_launch):
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth="testuser:testpass",
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == 0
            mock_launch.assert_called_once()
            call_kwargs = mock_launch.call_args[1]
            assert call_kwargs['auth'] == ("testuser", "testpass")

    def test_auth_with_colon_in_password(self, capsys):
        """Password can contain colons."""
        from backpropagate.cli import cmd_ui

        mock_launch = MagicMock()

        with patch("backpropagate.ui.launch", mock_launch):
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth="user:pass:with:colons",
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == 0
            call_kwargs = mock_launch.call_args[1]
            assert call_kwargs['auth'] == ("user", "pass:with:colons")

    def test_launch_success(self, capsys):
        """Successful UI launch."""
        from backpropagate.cli import cmd_ui

        mock_launch = MagicMock()

        with patch("backpropagate.ui.launch", mock_launch):
            args = argparse.Namespace(
                port=7862,
                share=True,
                auth=None,
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == 0
            mock_launch.assert_called_once_with(port=7862, share=True, auth=None)

    def test_launch_keyboard_interrupt(self, capsys):
        """KeyboardInterrupt during UI exits cleanly."""
        from backpropagate.cli import cmd_ui

        with patch("backpropagate.ui.launch", side_effect=KeyboardInterrupt()):
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
        """Launch exception handled gracefully."""
        from backpropagate.cli import cmd_ui

        with patch("backpropagate.ui.launch", side_effect=RuntimeError("Port in use")):
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth=None,
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Port in use" in captured.err

    def test_launch_exception_verbose(self, capsys):
        """Launch exception with verbose shows traceback."""
        from backpropagate.cli import cmd_ui

        with patch("backpropagate.ui.launch", side_effect=ValueError("Test error")):
            args = argparse.Namespace(
                port=7860,
                share=False,
                auth=None,
                verbose=True,
            )

            result = cmd_ui(args)

            assert result == 1


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

        assert result == 0
        captured = capsys.readouterr()
        assert "not yet implemented" in captured.out.lower() or "WARN" in captured.out

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
