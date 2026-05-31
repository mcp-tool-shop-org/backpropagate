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

    @pytest.fixture(autouse=True)
    def _free_ports(self):
        """CLIUI-B-004: neutralise the port pre-flight for the launch-path
        tests in this class.

        ``cmd_ui`` now bind-probes ``--port`` and ``--port + 1`` before handing
        off to the Reflex subprocess (so a busy port surfaces a structured
        EADDRINUSE error instead of a 30-60s-deferred traceback). The existing
        happy-path tests assert ``subprocess.run`` is reached; without this
        fixture they'd flake on any box that happens to have 7860/7862 bound.
        Force the probe to report "all free" by default; the dedicated
        port-in-use test overrides it with its own patch.
        """
        with patch("backpropagate.cli._find_port_in_use", return_value=None):
            yield

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
        """``--auth user`` without a colon is rejected by validate_auth_shape.

        Post-Wave-6 (ENFORCEMENT_AVAILABLE=True): --auth is no longer a
        blanket refuse-to-start. The shape validator still rejects malformed
        credentials at the CLI boundary, but the error path is now
        EXIT_USER_ERROR rather than a raised BackpropagateError, because
        cmd_ui catches BackpropagateError around validate_auth_shape and
        returns EXIT_USER_ERROR.
        """
        from backpropagate.cli import EXIT_USER_ERROR, cmd_ui

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=False,
                auth="invalid_no_colon",  # Missing colon
                verbose=False,
            )

            # cmd_ui catches the malformed-auth error and prints "Invalid
            # auth format" via the early ValueError branch (before
            # validate_auth_shape ever runs), then returns EXIT_USER_ERROR.
            result = cmd_ui(args)

            assert result == EXIT_USER_ERROR
            mock_run.assert_not_called()
            captured = capsys.readouterr()
            combined = (captured.err + captured.out).lower()
            assert "invalid auth format" in combined

    def test_auth_parsed_correctly(self, capsys):
        """Well-formed ``user:pass`` launches Reflex with the env var set.

        Post-Wave-6: ENFORCEMENT_AVAILABLE=True means --auth flows through
        validate_auth_shape, gets stored in BACKPROPAGATE_UI_AUTH on the
        subprocess env, and the launch returns EXIT_OK. This is the
        contract that was unreachable under the Wave 3.5 refuse-to-start.
        """
        from backpropagate.cli import EXIT_OK, cmd_ui

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=False,
                auth="testuser:testpass",
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == EXIT_OK
            mock_run.assert_called_once()
            call_env = mock_run.call_args.kwargs.get("env", {})
            assert call_env.get("BACKPROPAGATE_UI_AUTH") == "testuser:testpass"
            # Bind address is communicated to the middleware via env.
            assert call_env.get("BACKPROPAGATE_UI_HOST_BIND") == "127.0.0.1"

    def test_auth_with_colon_in_password(self, capsys):
        """Colons in the password are preserved (split on first colon only).

        Post-Wave-6: ``--auth user:pass:word`` parses as user="user" and
        password="pass:word"; the colon-in-password is preserved verbatim
        in BACKPROPAGATE_UI_AUTH on the subprocess env.
        """
        from backpropagate.cli import EXIT_OK, cmd_ui

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=False,
                auth="user:pass:with:colons",
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == EXIT_OK
            mock_run.assert_called_once()
            call_env = mock_run.call_args.kwargs.get("env", {})
            # Split on first colon: user="user", password="pass:with:colons"
            assert call_env.get("BACKPROPAGATE_UI_AUTH") == "user:pass:with:colons"

    def test_launch_success(self, capsys):
        """Successful UI launch via subprocess (no --auth, no --share).

        Post-Wave-6 happy path: a plain local UI launch with no share /
        no auth still proceeds (loopback bind is the default; no auth
        gate fires). The subprocess command-line construction assertion
        is preserved.
        """
        from backpropagate.cli import cmd_ui

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7862,
                host=None,
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
    # Post-Wave-6 contract (v1.2.0): with ENFORCEMENT_AVAILABLE=True the
    # auth middleware honors --auth, so --auth no longer triggers
    # refuse-to-start by itself. What remains gated:
    #   * --share without --auth — public URL without auth is the bug
    #     v1.2 closed; preserved as a hard error.
    #   * --host with a non-loopback bind without --auth — DNS-rebinding
    #     defense per DESIGN_BRIEF.
    # The companion happy-path test pins --auth user:pass launching the
    # Reflex subprocess with BACKPROPAGATE_UI_AUTH set on the env.
    # ------------------------------------------------------------------ #

    def test_cmd_ui_auth_proceeds_with_enforcement_available(self):
        """--auth user:pass launches the subprocess with the env var set.

        Replaces the deleted ``test_cmd_ui_auth_refuses_to_start_with_runtime_code``
        which pinned the Wave 3.5 refuse-to-start (no longer the contract).
        """
        from backpropagate.cli import EXIT_OK, cmd_ui

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=False,
                auth="user:pass",
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == EXIT_OK
            mock_run.assert_called_once()
            call_env = mock_run.call_args.kwargs.get("env", {})
            assert call_env.get("BACKPROPAGATE_UI_AUTH") == "user:pass"

    def test_cmd_ui_share_without_auth_still_refuses(self):
        """--share without --auth still hard-errors post-Wave-6.

        Pins the v1.2 contract: a public --share URL requires --auth so the
        middleware can enforce per-request auth on the tunnel.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=True,
                auth=None,
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            assert "--share" in str(excinfo.value)
            mock_run.assert_not_called()

    def test_cmd_ui_share_refuses_to_start(self):
        """Alias for the v1.2 --share + missing --auth refuse-to-start.

        Kept under the original name so any external CI or doc reference
        continues to resolve; behavior matches
        ``test_cmd_ui_share_without_auth_still_refuses``.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
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
                host=None,
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
        """--share + --auth proceeds; the middleware enforces per-request auth.

        Post-Wave-6 (ENFORCEMENT_AVAILABLE=True): the v1.2 contract is that
        a public --share URL requires --auth so the middleware can enforce
        per-request basic auth. With both flags supplied, the launch
        proceeds and BACKPROPAGATE_UI_AUTH is exported to the Reflex child.
        """
        from backpropagate.cli import EXIT_OK, cmd_ui

        with patch("backpropagate.cli.subprocess.run") as mock_run, patch(
            "backpropagate.cli._spawn_cloudflared_tunnel",
            return_value=(MagicMock(), "https://abc.trycloudflare.com"),
        ):
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=True,
                auth="alice:secret123",
                auth_file=None,
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == EXIT_OK
            mock_run.assert_called_once()
            call_env = mock_run.call_args.kwargs.get("env", {})
            assert call_env.get("BACKPROPAGATE_UI_AUTH") == "alice:secret123"
            # Wave 6a cloudflared wiring exports the parsed tunnel host
            # (was an empty-string placeholder pre-v1.3).
            assert call_env.get("BACKPROPAGATE_UI_SHARE_HOST") == "abc.trycloudflare.com"  # hostname only, not URL

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

    def test_cmd_ui_port_in_use_refuses_with_structured_error(self, capsys):
        """CLIUI-B-004: an occupied port → structured EADDRINUSE, not a traceback.

        Before the fix, a port already bound by a previous `backprop ui` (or any
        other dev server) surfaced only as a bare Reflex traceback + non-zero
        exit, 30-60s into the launch. The pre-flight now bind-probes --port and
        --port+1 and raises a structured ``RUNTIME_UI_PORT_IN_USE`` BEFORE any
        subprocess is spawned. This test forces the probe to report the
        frontend port busy and pins the raised code + that the subprocess is
        never launched.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        with patch("backpropagate.cli.subprocess.run") as mock_run, patch(
            "backpropagate.cli._find_port_in_use", return_value=7860
        ):
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=False,
                auth=None,
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_PORT_IN_USE"
            # The message must name the busy port and the --port remedy.
            assert "7860" in excinfo.value.message
            assert excinfo.value.suggestion and "--port" in excinfo.value.suggestion
            # Refuse-to-start: the Reflex subprocess is never spawned.
            mock_run.assert_not_called()

    def test_cmd_ui_port_in_use_skips_cloudflared_spawn(self, capsys):
        """CLIUI-B-004: a busy port fails BEFORE cloudflared is spawned.

        The pre-flight sits ahead of the cloudflared spawn so a busy port never
        leaks a public tunnel. With --share + --auth (which would otherwise
        spawn cloudflared) and a busy port, _spawn_cloudflared_tunnel must NOT
        be called.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        with patch("backpropagate.cli.subprocess.run") as mock_run, patch(
            "backpropagate.cli._spawn_cloudflared_tunnel"
        ) as mock_spawn, patch(
            "backpropagate.cli._find_port_in_use", return_value=7861
        ):
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=True,
                auth="alice:secret123",
                auth_file=None,
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_PORT_IN_USE"
            mock_spawn.assert_not_called()
            mock_run.assert_not_called()


class TestFindPortInUse:
    """CLIUI-B-004: unit tests for the _find_port_in_use bind-probe helper."""

    def test_free_ports_return_none(self):
        """All-free candidate ports → None (no false positive)."""
        import socket

        from backpropagate.cli import _find_port_in_use

        # Grab an ephemeral port, learn its number, then release it so the
        # probe sees it free.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        assert _find_port_in_use("127.0.0.1", [free_port]) is None

    def test_bound_port_is_detected(self):
        """A genuinely-bound port is reported back."""
        import socket

        from backpropagate.cli import _find_port_in_use

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            bound_port = sock.getsockname()[1]
            # Probe a free port first, then the bound one — confirms the loop
            # returns the FIRST occupied candidate (here the second slot).
            assert _find_port_in_use("127.0.0.1", [bound_port]) == bound_port
        finally:
            sock.close()

    def test_returns_first_busy_of_pair(self):
        """With [free, busy] the busy one is returned (frontend free, backend busy)."""
        import socket

        from backpropagate.cli import _find_port_in_use

        # Reserve a free-then-released port number for the frontend slot.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            busy.bind(("127.0.0.1", 0))
            busy.listen(1)
            busy_port = busy.getsockname()[1]
            assert _find_port_in_use("127.0.0.1", [free_port, busy_port]) == busy_port
        finally:
            busy.close()

    def test_unbindable_host_is_skipped_not_false_positive(self):
        """A host the kernel won't bind (EADDRNOTAVAIL) is skipped, not flagged.

        Probing an IP that isn't assigned to this machine raises
        EADDRNOTAVAIL, which is out of scope for a port pre-flight — the
        helper must return None (let Reflex surface its own error) rather
        than a false EADDRINUSE.
        """
        from backpropagate.cli import _find_port_in_use

        # 192.0.2.0/24 is TEST-NET-1 (RFC 5737) — guaranteed not local.
        assert _find_port_in_use("192.0.2.1", [54321]) is None


# =============================================================================
# TESTS-A-002 — Cloudflared subprocess shutdown (v1.4 Wave 2 amend)
# =============================================================================
#
# v1.3 Wave 6a wired cloudflared as a Quick-Tunnel subprocess spawned by
# cmd_ui when --share is passed. The happy-path tests at
# test_cmd_ui_share_with_auth_allowed cover env-var propagation, but
# nothing tests that the cloudflared subprocess gets cleaned up on the
# various exit paths. A regression that dropped the .terminate() call
# in cmd_ui's finally block would leak a public tunnel after every
# Ctrl+C — operator footprint that is impossible to spot from logs.
#
# These tests mock _spawn_cloudflared_tunnel to return a MagicMock
# whose .terminate() / .wait() / .poll() / .kill() are tracked, then
# drive cmd_ui through KeyboardInterrupt, normal exit (0), and
# RuntimeError paths. Each one MUST end with .terminate() called on
# the tunnel mock. The shape mirrors the SIGKILL recovery pattern in
# tests/test_e2e_chain.py::TestResumeAfterSigkill (subprocess-cleanup
# discipline).
# =============================================================================


class TestCloudflaredShutdown:
    """TESTS-A-002: --share spawns cloudflared; cleanup MUST run on every exit.

    The finally block in cmd_ui calls cloudflared_proc.terminate() +
    .wait(timeout=5) and falls back to .kill() if the wait times out.
    These tests pin that cleanup contract across the three relevant
    cmd_ui exit paths:

    1. KeyboardInterrupt (operator Ctrl+C)
    2. Normal subprocess.run return (reflex exited 0)
    3. RuntimeError from subprocess.run (port-in-use, exec failure, etc.)

    Each test injects a mock cloudflared subprocess whose .terminate()
    is the load-bearing assertion. We also pin a 4th case for the
    .wait() timeout path that escalates to .kill().
    """

    @pytest.fixture(autouse=True)
    def _free_ports(self):
        """CLIUI-B-004: neutralise the cmd_ui port pre-flight so these
        cleanup-contract tests don't flake on a box with 7860/7861 bound.
        These tests drive cmd_ui through to subprocess.run; the port probe
        is not their subject."""
        with patch("backpropagate.cli._find_port_in_use", return_value=None):
            yield

    @staticmethod
    def _mock_cloudflared_proc(wait_raises=None, poll_return=None):
        """Build a MagicMock that quacks like a subprocess.Popen for cloudflared.

        Defaults: .wait() returns 0 cleanly, .poll() returns None
        (running). Pass wait_raises=subprocess.TimeoutExpired to
        force the kill() fallback path.
        """
        proc = MagicMock()
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        if wait_raises is not None:
            proc.wait = MagicMock(side_effect=wait_raises)
        else:
            proc.wait = MagicMock(return_value=0)
        proc.poll = MagicMock(return_value=poll_return)
        return proc

    @staticmethod
    def _mock_subprocess_result(returncode: int = 0) -> MagicMock:
        """Mirror TestCmdUI._mock_subprocess_result for consistency."""
        result = MagicMock()
        result.returncode = returncode
        return result

    def test_cloudflared_terminated_on_normal_exit(self, capsys):
        """Reflex subprocess exits cleanly → cloudflared.terminate() called."""
        from backpropagate.cli import cmd_ui

        cloudflared_mock = self._mock_cloudflared_proc()

        with patch("backpropagate.cli.subprocess.run") as mock_run, patch(
            "backpropagate.cli._spawn_cloudflared_tunnel",
            return_value=(cloudflared_mock, "https://abc.trycloudflare.com"),
        ):
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=True,
                auth="alice:secret123",
                auth_file=None,
                verbose=False,
            )

            result = cmd_ui(args)

        # Normal exit should return 0.
        assert result == 0
        # The load-bearing assertion: cleanup ran.
        cloudflared_mock.terminate.assert_called_once()
        # And .wait(timeout=5) was used for the graceful join.
        cloudflared_mock.wait.assert_called()

    def test_cloudflared_terminated_on_keyboard_interrupt(self, capsys):
        """Ctrl+C during reflex run → cloudflared.terminate() still called.

        The KeyboardInterrupt branch in cmd_ui returns EXIT_OK (0) and
        prints "UI stopped"; the finally clause MUST run terminate so
        the public tunnel doesn't outlive the operator's terminal.
        """
        from backpropagate.cli import cmd_ui

        cloudflared_mock = self._mock_cloudflared_proc()

        with patch("backpropagate.cli.subprocess.run", side_effect=KeyboardInterrupt), patch(
            "backpropagate.cli._spawn_cloudflared_tunnel",
            return_value=(cloudflared_mock, "https://abc.trycloudflare.com"),
        ):
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=True,
                auth="alice:secret123",
                auth_file=None,
                verbose=False,
            )

            result = cmd_ui(args)

        # KeyboardInterrupt path returns EXIT_OK (0).
        assert result == 0
        cloudflared_mock.terminate.assert_called_once()

    def test_cloudflared_terminated_on_subprocess_runtime_error(self, capsys):
        """RuntimeError from subprocess.run → cloudflared.terminate() still called.

        Unhandled subprocess errors exit non-zero; the finally clause
        MUST still terminate the tunnel. Without this assertion, a
        regression that put cleanup inside a try-block-only would
        silently leak tunnels on every reflex-startup failure.
        """
        from backpropagate.cli import cmd_ui

        cloudflared_mock = self._mock_cloudflared_proc()

        with patch(
            "backpropagate.cli.subprocess.run",
            side_effect=RuntimeError("port in use"),
        ), patch(
            "backpropagate.cli._spawn_cloudflared_tunnel",
            return_value=(cloudflared_mock, "https://abc.trycloudflare.com"),
        ):
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=True,
                auth="alice:secret123",
                auth_file=None,
                verbose=False,
            )

            result = cmd_ui(args)

        # RuntimeError exits EXIT_RUNTIME_ERROR (2).
        assert result == 2
        cloudflared_mock.terminate.assert_called_once()

    def test_cloudflared_killed_when_terminate_wait_times_out(self, capsys):
        """terminate() + wait(timeout=5) timing out → escalates to .kill().

        The finally block uses the documented Popen graceful-shutdown
        pattern: terminate first, wait up to 5s, kill on timeout. If
        cloudflared ignores SIGTERM, the test pins the SIGKILL fallback.
        """
        import subprocess

        from backpropagate.cli import cmd_ui

        cloudflared_mock = self._mock_cloudflared_proc(
            wait_raises=subprocess.TimeoutExpired(cmd="cloudflared", timeout=5),
        )

        with patch("backpropagate.cli.subprocess.run") as mock_run, patch(
            "backpropagate.cli._spawn_cloudflared_tunnel",
            return_value=(cloudflared_mock, "https://abc.trycloudflare.com"),
        ):
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=True,
                auth="alice:secret123",
                auth_file=None,
                verbose=False,
            )

            cmd_ui(args)

        # terminate() always called first.
        cloudflared_mock.terminate.assert_called_once()
        # On TimeoutExpired, kill() is the escalation.
        cloudflared_mock.kill.assert_called_once()

    def test_cloudflared_cleanup_swallows_oserror(self, capsys):
        """If .terminate() raises OSError (already dead), cleanup must not crash.

        Real-world: the cloudflared process may have already exited by
        the time cmd_ui's finally clause runs. terminate() on a dead
        Popen raises ProcessLookupError on POSIX / OSError on Windows.
        The cleanup is wrapped in `except (OSError, ValueError)` so
        these don't bubble up as a confusing post-shutdown crash.
        """
        from backpropagate.cli import cmd_ui

        cloudflared_mock = self._mock_cloudflared_proc()
        cloudflared_mock.terminate.side_effect = OSError("already dead")

        with patch("backpropagate.cli.subprocess.run") as mock_run, patch(
            "backpropagate.cli._spawn_cloudflared_tunnel",
            return_value=(cloudflared_mock, "https://abc.trycloudflare.com"),
        ):
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=True,
                auth="alice:secret123",
                auth_file=None,
                verbose=False,
            )

            # Must not raise.
            result = cmd_ui(args)

        # Normal exit returns 0; the OSError is swallowed.
        assert result == 0
        cloudflared_mock.terminate.assert_called_once()

    def test_no_cloudflared_cleanup_when_share_false(self, capsys):
        """Without --share, no cloudflared is spawned → no cleanup call attempted.

        Negative control for the cleanup-path tests above: the finally
        block guards on `cloudflared_proc is not None` so the
        no-share path doesn't try to terminate a None.
        """
        from backpropagate.cli import cmd_ui

        # If --share is False, _spawn_cloudflared_tunnel should NOT be
        # called. We patch it anyway and assert call_count == 0 to pin
        # this contract (a regression that called the spawn helper
        # unconditionally would burn an extra subprocess on every
        # non-share launch).
        with patch("backpropagate.cli.subprocess.run") as mock_run, patch(
            "backpropagate.cli._spawn_cloudflared_tunnel",
        ) as mock_spawn:
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=False,
                auth=None,
                auth_file=None,
                verbose=False,
            )

            result = cmd_ui(args)

        assert result == 0
        mock_spawn.assert_not_called()


# =============================================================================
# CLIUI-B-008 — cloudflared deadline-aware read (Stage C proactive)
# =============================================================================
#
# The tunnel-URL scrape loop used a bare blocking ``proc.stdout.readline()``
# and only re-checked the deadline at the TOP of the loop. A cloudflared that
# was alive but silent (DNS wedged, captive portal swallowing the handshake)
# parked the call inside readline() forever — the deadline was never reached
# and ``backprop ui --share`` hung with no output until the operator Ctrl+C'd.
# The fix moves blocking reads onto a daemon reader thread and pulls lines via
# ``queue.get(timeout=remaining)`` so the deadline is authoritative regardless
# of whether cloudflared ever speaks. These tests pin that contract.


class TestCloudflaredDeadline:
    """CLIUI-B-008: the URL-scrape loop honors the deadline even on silence."""

    @staticmethod
    def _silent_proc(alive: bool = True):
        """Build a fake Popen whose stdout.readline() blocks until released.

        Models an alive-but-silent cloudflared: ``poll()`` returns None
        (running) and ``readline()`` parks on an Event that the test releases
        in teardown, so the reader thread would block indefinitely without the
        deadline-aware consumer. ``read()``/iteration return nothing.
        """
        import threading as _threading

        release = _threading.Event()

        class _BlockingStdout:
            def readline(self_inner):
                # Block until the test releases us (simulating no output).
                # Return "" (EOF) once released so the daemon reader exits
                # cleanly during teardown rather than leaking.
                release.wait(timeout=5.0)
                return ""

            def read(self_inner):
                return ""

            def __iter__(self_inner):
                return iter(())

        proc = MagicMock()
        proc.stdout = _BlockingStdout()
        proc.poll = MagicMock(return_value=None if alive else 0)
        proc.returncode = None if alive else 0
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        proc._release = release  # exposed so the test can unblock the reader
        return proc

    def test_silent_cloudflared_respects_deadline(self, capsys, monkeypatch):
        """An alive-but-silent cloudflared returns None at the deadline, not a hang.

        Pre-fix this would block forever in readline(). We set a 1s timeout
        via the env override and assert the call returns None within a few
        seconds (generous bound to avoid CI flake) — proving the deadline is
        enforced rather than gated behind a line that never arrives.
        """
        import time as _time

        from backpropagate import cli as cli_module

        # Tight deadline so the test is fast; the helper clamps <=0 to default
        # but 1 is valid.
        monkeypatch.setenv("BACKPROPAGATE_CLOUDFLARED_TIMEOUT", "1")

        proc = self._silent_proc(alive=True)

        # shutil.which must find a binary (else the spawn aborts before the
        # read loop); Popen returns our fake silent process.
        with patch("shutil.which", return_value="/usr/bin/cloudflared"), patch(
            "backpropagate.cli.subprocess.Popen", return_value=proc
        ):
            start = _time.monotonic()
            try:
                result = cli_module._spawn_cloudflared_tunnel(7860)
            finally:
                proc._release.set()  # let the daemon reader exit
            elapsed = _time.monotonic() - start

        # The contract: None (no URL surfaced) — NOT a hang.
        assert result is None
        # Bounded: must return near the 1s deadline, well under the 5s readline
        # cap. A pre-fix blocking readline would sit until proc._release (or
        # forever in production). Allow generous headroom for slow CI.
        assert elapsed < 4.5, f"deadline not enforced; took {elapsed:.1f}s"
        # And it terminated the silent subprocess on the timeout path.
        proc.terminate.assert_called()

    def test_url_parsed_from_first_line_returns_proc(self, capsys, monkeypatch):
        """When cloudflared emits the URL promptly, it's parsed and returned.

        Positive-path companion: confirms the queue-backed reader still finds
        the trycloudflare URL on a normal fast start (the common case) and
        returns the (proc, url) tuple.
        """
        from backpropagate import cli as cli_module

        url = "https://happy-fast-tunnel.trycloudflare.com"
        lines = iter([f"INF connection registered url={url}\n", ""])  # then EOF

        class _ChattyStdout:
            def readline(self_inner):
                return next(lines, "")

            def read(self_inner):
                return ""

            def __iter__(self_inner):
                return iter(())

        proc = MagicMock()
        proc.stdout = _ChattyStdout()
        proc.poll = MagicMock(return_value=None)
        proc.returncode = None
        proc.terminate = MagicMock()
        proc.kill = MagicMock()

        with patch("shutil.which", return_value="/usr/bin/cloudflared"), patch(
            "backpropagate.cli.subprocess.Popen", return_value=proc
        ):
            result = cli_module._spawn_cloudflared_tunnel(7860)

        assert result is not None
        returned_proc, returned_url = result
        assert returned_proc is proc
        assert returned_url == url
        # Happy path does NOT terminate — the caller owns the live tunnel.
        proc.terminate.assert_not_called()

    def test_dead_cloudflared_pre_url_returns_none(self, capsys, monkeypatch):
        """A cloudflared that exits before any URL → None + tail surfaced.

        Reader hits EOF immediately (process already dead). The consumer's
        None-sentinel branch fires, surfaces the tail, and returns None.
        """
        from backpropagate import cli as cli_module

        # readline returns the error tail then EOF; poll() reports dead.
        lines = iter(["ERR failed to connect to edge\n", ""])

        class _DeadStdout:
            def readline(self_inner):
                return next(lines, "")

            def read(self_inner):
                return ""

            def __iter__(self_inner):
                return iter(())

        proc = MagicMock()
        proc.stdout = _DeadStdout()
        proc.poll = MagicMock(return_value=1)  # already exited
        proc.returncode = 1
        proc.terminate = MagicMock()
        proc.kill = MagicMock()

        with patch("shutil.which", return_value="/usr/bin/cloudflared"), patch(
            "backpropagate.cli.subprocess.Popen", return_value=proc
        ):
            result = cli_module._spawn_cloudflared_tunnel(7860)

        assert result is None
        captured = capsys.readouterr()
        combined = captured.err + captured.out
        assert "exited before publishing" in combined


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

    @pytest.fixture(autouse=True)
    def _free_ports(self):
        """CLIUI-B-004: neutralise the cmd_ui port pre-flight so the
        enforcement-flip proceed-path tests don't flake on 7860/7861."""
        with patch("backpropagate.cli._find_port_in_use", return_value=None):
            yield

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

    def test_share_with_auth_proceeds_when_enforcement_available(self, monkeypatch):
        """--share + --auth proceeds once the middleware is wired.

        Post-Wave-6 (v1.2.0) inverted the Wave 3.5 contract: the v1.2
        rule is "--share REQUIRES --auth so the middleware can enforce
        per-request basic auth on the public tunnel". Both flags supplied
        means the gate is satisfied; cmd_ui launches the Reflex subprocess
        with BACKPROPAGATE_UI_AUTH and BACKPROPAGATE_UI_SHARE_HOST set.
        Renamed from ``test_share_with_auth_still_refuses_when_enforcement_available``
        because that contract no longer holds.
        """
        from backpropagate.cli import EXIT_OK, cmd_ui

        monkeypatch.setattr(
            "backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE", True
        )

        with patch("backpropagate.cli.subprocess.run") as mock_run, patch(
            "backpropagate.cli._spawn_cloudflared_tunnel",
            return_value=(MagicMock(), "https://abc.trycloudflare.com"),
        ):
            mock_run.return_value = self._mock_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=None,
                share=True,
                auth="alice:hunter2",
                auth_file=None,
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == EXIT_OK
            mock_run.assert_called_once()
            call_env = mock_run.call_args.kwargs.get("env", {})
            assert call_env.get("BACKPROPAGATE_UI_AUTH") == "alice:hunter2"
            assert call_env.get("BACKPROPAGATE_UI_SHARE_HOST") == "abc.trycloudflare.com"  # hostname only, not URL


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
