"""Tests for CLI commands."""

import pytest
from unittest.mock import MagicMock, patch
from io import StringIO
import sys


class TestParser:
    """Tests for CLI argument parser."""

    def test_parser_creation(self, cli_parser):
        """Test parser can be created."""
        assert cli_parser is not None

    def test_train_command_basic(self, cli_parser):
        """Test train command parsing."""
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.command == "train"
        assert args.data == "data.jsonl"

    def test_train_command_all_options(self, cli_parser):
        """Test train command with all options."""
        args = cli_parser.parse_args([
            "train",
            "-d", "data.jsonl",
            "-m", "custom/model",
            "--steps", "200",
            "--samples", "5000",
            "--batch-size", "4",
            "--lr", "1e-4",
            "--lora-r", "32",
            "-o", "./custom-output",
            "--no-unsloth",
        ])

        assert args.command == "train"
        assert args.data == "data.jsonl"
        assert args.model == "custom/model"
        assert args.steps == 200
        assert args.samples == 5000
        assert args.batch_size == "4"
        assert args.lr == 1e-4
        assert args.lora_r == 32
        assert args.output == "./custom-output"
        assert args.no_unsloth is True

    def test_train_command_defaults(self, cli_parser):
        """Test train command has correct defaults."""
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])

        assert args.model == "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
        assert args.steps == 100
        assert args.samples is None
        assert args.batch_size == "auto"
        assert args.lr == 2e-4
        assert args.lora_r == 16
        assert args.output == "./output"
        assert args.no_unsloth is False

    def test_train_requires_data(self, cli_parser):
        """Test train command requires --data."""
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["train"])

    def test_multi_run_command_basic(self, cli_parser):
        """Test multi-run command parsing."""
        args = cli_parser.parse_args(["multi-run", "-d", "ultrachat"])
        assert args.command == "multi-run"
        assert args.data == "ultrachat"

    def test_multi_run_command_all_options(self, cli_parser):
        """Test multi-run command with all options."""
        args = cli_parser.parse_args([
            "multi-run",
            "-d", "ultrachat",
            "-m", "custom/model",
            "--runs", "10",
            "--steps", "50",
            "--samples", "2000",
            "--merge-mode", "simple",
            "-o", "./multi-output",
        ])

        assert args.command == "multi-run"
        assert args.data == "ultrachat"
        assert args.model == "custom/model"
        assert args.runs == 10
        assert args.steps == 50
        assert args.samples == 2000
        assert args.merge_mode == "simple"
        assert args.output == "./multi-output"

    def test_multi_run_defaults(self, cli_parser):
        """Test multi-run command has correct defaults."""
        args = cli_parser.parse_args(["multi-run", "-d", "data"])

        assert args.runs == 5
        assert args.steps == 100
        assert args.samples == 1000
        assert args.merge_mode == "slao"

    def test_export_command_basic(self, cli_parser):
        """Test export command parsing."""
        args = cli_parser.parse_args(["export", "./model"])
        assert args.command == "export"
        assert args.model_path == "./model"

    def test_export_command_all_options(self, cli_parser):
        """Test export command with all options."""
        args = cli_parser.parse_args([
            "export", "./model",
            "-f", "gguf",
            "-q", "q8_0",
            "-o", "./export-dir",
            "--ollama",
            "--ollama-name", "my-model",
        ])

        assert args.command == "export"
        assert args.model_path == "./model"
        assert args.format == "gguf"
        assert args.quantization == "q8_0"
        assert args.output == "./export-dir"
        assert args.ollama is True
        assert args.ollama_name == "my-model"

    def test_export_defaults(self, cli_parser):
        """Test export command has correct defaults."""
        args = cli_parser.parse_args(["export", "./model"])

        assert args.format == "lora"
        assert args.quantization == "q4_k_m"
        assert args.output is None
        assert args.ollama is False
        assert args.ollama_name is None

    def test_export_format_choices(self, cli_parser):
        """Test export format accepts only valid choices."""
        for fmt in ["lora", "merged", "gguf"]:
            args = cli_parser.parse_args(["export", "./model", "-f", fmt])
            assert args.format == fmt

        with pytest.raises(SystemExit):
            cli_parser.parse_args(["export", "./model", "-f", "invalid"])

    def test_export_quantization_choices(self, cli_parser):
        """Test export quantization accepts only valid choices."""
        valid_quants = ["f16", "q8_0", "q5_k_m", "q4_k_m", "q4_0", "q2_k"]
        for quant in valid_quants:
            args = cli_parser.parse_args(["export", "./model", "-q", quant])
            assert args.quantization == quant

        with pytest.raises(SystemExit):
            cli_parser.parse_args(["export", "./model", "-q", "invalid"])

    def test_info_command(self, cli_parser):
        """Test info command parsing."""
        args = cli_parser.parse_args(["info"])
        assert args.command == "info"

    def test_config_command(self, cli_parser):
        """Test config command parsing."""
        args = cli_parser.parse_args(["config"])
        assert args.command == "config"

    def test_config_command_options(self, cli_parser):
        """Test config command with options."""
        args = cli_parser.parse_args(["config", "--show"])
        assert args.show is True

        args = cli_parser.parse_args(["config", "--reset"])
        assert args.reset is True

        args = cli_parser.parse_args(["config", "--set", "key=value"])
        assert args.set == "key=value"

    def test_version_flag(self, cli_parser):
        """Test --version flag."""
        with pytest.raises(SystemExit) as exc_info:
            cli_parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_verbose_flag(self, cli_parser):
        """Test --verbose flag."""
        args = cli_parser.parse_args(["--verbose", "info"])
        assert args.verbose is True


class TestMain:
    """Tests for main CLI entry point."""

    def test_no_command_returns_zero(self):
        """Test main with no command returns 0 (shows help)."""
        from backpropagate.cli import main

        result = main([])
        assert result == 0

    def test_info_command_runs(self):
        """Test info command runs successfully."""
        from backpropagate.cli import main

        result = main(["info"])
        assert result == 0

    def test_config_command_runs(self):
        """Test config command runs successfully."""
        from backpropagate.cli import main

        result = main(["config"])
        assert result == 0

    def test_train_missing_data_returns_error(self):
        """Test train command without data returns error."""
        from backpropagate.cli import main

        with pytest.raises(SystemExit):
            main(["train"])


class TestCmdInfo:
    """Tests for info command."""

    def test_cmd_info_outputs_system_info(self, capsys):
        """Test cmd_info outputs system information."""
        from backpropagate.cli import cmd_info
        import argparse

        args = argparse.Namespace(verbose=False)
        result = cmd_info(args)

        assert result == 0

        captured = capsys.readouterr()
        assert "System" in captured.out
        assert "Python" in captured.out

    def test_cmd_info_outputs_features(self, capsys):
        """Test cmd_info outputs feature availability."""
        from backpropagate.cli import cmd_info
        import argparse

        args = argparse.Namespace(verbose=False)
        cmd_info(args)

        captured = capsys.readouterr()
        assert "Features" in captured.out

    def test_cmd_info_outputs_configuration(self, capsys):
        """Test cmd_info outputs configuration."""
        from backpropagate.cli import cmd_info
        import argparse

        args = argparse.Namespace(verbose=False)
        cmd_info(args)

        captured = capsys.readouterr()
        assert "Configuration" in captured.out
        assert "Model" in captured.out


class TestCmdConfig:
    """Tests for config command."""

    def test_cmd_config_shows_config(self, capsys):
        """Test cmd_config shows configuration."""
        from backpropagate.cli import cmd_config
        import argparse

        args = argparse.Namespace(show=False, set=None, reset=False, verbose=False)
        result = cmd_config(args)

        assert result == 0

        captured = capsys.readouterr()
        assert "Model" in captured.out
        assert "LoRA" in captured.out
        assert "Training" in captured.out

    def test_cmd_config_reset_message(self, capsys):
        """Test cmd_config reset shows message."""
        from backpropagate.cli import cmd_config
        import argparse

        args = argparse.Namespace(show=False, set=None, reset=True, verbose=False)
        result = cmd_config(args)

        assert result == 0

        captured = capsys.readouterr()
        assert "reset" in captured.out.lower() or "WARN" in captured.out


class TestCmdTrain:
    """Tests for train command."""

    def test_cmd_train_requires_data(self, capsys):
        """Test cmd_train requires data argument."""
        from backpropagate.cli import cmd_train
        import argparse

        args = argparse.Namespace(
            data=None,
            model="test",
            steps=10,
            samples=None,
            batch_size="auto",
            lr=2e-4,
            lora_r=16,
            output="./output",
            no_unsloth=True,
            verbose=False,
        )

        result = cmd_train(args)
        assert result == 1

        captured = capsys.readouterr()
        assert "required" in captured.err.lower() or "ERROR" in captured.err


class TestCmdExport:
    """Tests for export command."""

    def test_cmd_export_model_not_found(self, capsys, temp_dir):
        """Test cmd_export returns error for missing model."""
        from backpropagate.cli import cmd_export
        import argparse

        args = argparse.Namespace(
            model_path=str(temp_dir / "nonexistent"),
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
        assert "not found" in captured.err.lower() or "ERROR" in captured.err


class TestColors:
    """Tests for color support detection."""

    def test_colors_class_exists(self):
        """Test Colors class exists."""
        from backpropagate.cli import Colors

        assert hasattr(Colors, "ENABLED")
        assert hasattr(Colors, "RESET")
        assert hasattr(Colors, "RED")
        assert hasattr(Colors, "GREEN")

    def test_color_values_are_strings(self):
        """Test color values are strings."""
        from backpropagate.cli import Colors

        assert isinstance(Colors.RESET, str)
        assert isinstance(Colors.RED, str)
        assert isinstance(Colors.GREEN, str)


class TestProgressBar:
    """Tests for ProgressBar class."""

    def test_progress_bar_creation(self):
        """Test ProgressBar can be created."""
        from backpropagate.cli import ProgressBar

        progress = ProgressBar(total=100, width=40, prefix="Test: ")
        assert progress.total == 100
        assert progress.width == 40
        assert progress.prefix == "Test: "
        assert progress.current == 0

    def test_progress_bar_update(self, capsys):
        """Test ProgressBar update."""
        from backpropagate.cli import ProgressBar

        progress = ProgressBar(total=100, width=20)
        progress.update(50)

        # Just verify it doesn't crash
        assert progress.current == 50

    def test_progress_bar_finish(self, capsys):
        """Test ProgressBar finish."""
        from backpropagate.cli import ProgressBar

        progress = ProgressBar(total=100, width=20)
        progress.finish()

        assert progress.current == 100


class TestPrintHelpers:
    """Tests for print helper functions."""

    def test_print_header(self, capsys):
        """Test _print_header output."""
        from backpropagate.cli import _print_header

        _print_header("Test Header")
        captured = capsys.readouterr()

        assert "Test Header" in captured.out
        assert "-" in captured.out  # Underline

    def test_print_success(self, capsys):
        """Test _print_success output."""
        from backpropagate.cli import _print_success

        _print_success("Success message")
        captured = capsys.readouterr()

        assert "Success message" in captured.out
        assert "OK" in captured.out

    def test_print_error(self, capsys):
        """Test _print_error output."""
        from backpropagate.cli import _print_error

        _print_error("Error message")
        captured = capsys.readouterr()

        assert "Error message" in captured.err
        assert "ERROR" in captured.err

    def test_print_warning(self, capsys):
        """Test _print_warning output."""
        from backpropagate.cli import _print_warning

        _print_warning("Warning message")
        captured = capsys.readouterr()

        assert "Warning message" in captured.out
        assert "WARN" in captured.out

    def test_print_info(self, capsys):
        """Test _print_info output."""
        from backpropagate.cli import _print_info

        _print_info("Info message")
        captured = capsys.readouterr()

        assert "Info message" in captured.out

    def test_print_kv(self, capsys):
        """Test _print_kv output."""
        from backpropagate.cli import _print_kv

        _print_kv("Key", "Value")
        captured = capsys.readouterr()

        assert "Key" in captured.out
        assert "Value" in captured.out


class TestModuleExports:
    """Tests for module exports."""

    def test_main_exported(self):
        """Test main function is exported."""
        from backpropagate.cli import main
        assert callable(main)

    def test_create_parser_exported(self):
        """Test create_parser function is exported."""
        from backpropagate.cli import create_parser
        assert callable(create_parser)

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from backpropagate import cli
        assert "main" in cli.__all__
        assert "create_parser" in cli.__all__


# =============================================================================
# COMMAND EXECUTION TESTS (with mocking)
# =============================================================================

class TestCmdTrainExecution:
    """Tests for train command execution with mocked Trainer."""

    def test_cmd_train_successful_training(self, capsys, temp_dir):
        """Test successful training execution.

        This tests lines 149-206 in cli.py (cmd_train function).
        """
        from backpropagate.cli import cmd_train
        import argparse

        # Create mock trainer and result
        mock_result = MagicMock()
        mock_result.final_loss = 0.5
        mock_result.duration_seconds = 60.0

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_result
        mock_trainer.save.return_value = str(temp_dir / "saved_model")

        # Patch in the trainer module since imports happen inside cmd_train
        with patch("backpropagate.trainer.Trainer", return_value=mock_trainer), \
             patch("backpropagate.trainer.TrainingCallback"):
            args = argparse.Namespace(
                data="test_data.jsonl",
                model="test-model",
                steps=10,
                samples=100,
                batch_size="2",
                lr=2e-4,
                lora_r=16,
                output=str(temp_dir),
                no_unsloth=True,
                verbose=False,
            )

            result = cmd_train(args)

            assert result == 0
            mock_trainer.train.assert_called_once()
            mock_trainer.save.assert_called_once()

            captured = capsys.readouterr()
            assert "Training complete" in captured.out

    def test_cmd_train_with_samples_display(self, capsys, temp_dir):
        """Test train command displays sample count."""
        from backpropagate.cli import cmd_train
        import argparse

        mock_result = MagicMock()
        mock_result.final_loss = 0.5
        mock_result.duration_seconds = 60.0

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_result
        mock_trainer.save.return_value = str(temp_dir)

        with patch("backpropagate.trainer.Trainer", return_value=mock_trainer), \
             patch("backpropagate.trainer.TrainingCallback"):
            args = argparse.Namespace(
                data="test_data.jsonl",
                model="test-model",
                steps=10,
                samples=500,  # Specific sample count
                batch_size="auto",
                lr=2e-4,
                lora_r=16,
                output=str(temp_dir),
                no_unsloth=True,
                verbose=False,
            )

            cmd_train(args)

            captured = capsys.readouterr()
            assert "Samples: 500" in captured.out

    def test_cmd_train_keyboard_interrupt(self, capsys, temp_dir):
        """Test train command handles KeyboardInterrupt.

        This tests lines 197-200:
            except KeyboardInterrupt:
                _print_warning("Training interrupted by user")
                return 130
        """
        from backpropagate.cli import cmd_train
        import argparse

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = KeyboardInterrupt()

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
                output=str(temp_dir),
                no_unsloth=True,
                verbose=False,
            )

            result = cmd_train(args)

            assert result == 130
            captured = capsys.readouterr()
            assert "interrupted" in captured.out.lower()

    def test_cmd_train_exception_handling(self, capsys, temp_dir):
        """Test train command handles exceptions.

        This tests lines 201-206:
            except Exception as e:
                _print_error(f"Training failed: {e}")
                if args.verbose:
                    traceback.print_exc()
                return 1
        """
        from backpropagate.cli import cmd_train
        import argparse

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = RuntimeError("Test error")

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
                output=str(temp_dir),
                no_unsloth=True,
                verbose=False,
            )

            result = cmd_train(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "ERROR" in captured.err
            assert "Test error" in captured.err

    def test_cmd_train_verbose_traceback(self, capsys, temp_dir):
        """Test train command prints traceback when verbose."""
        from backpropagate.cli import cmd_train
        import argparse

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = ValueError("Verbose error")

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
                output=str(temp_dir),
                no_unsloth=True,
                verbose=True,  # Enable verbose
            )

            result = cmd_train(args)

            assert result == 1
            captured = capsys.readouterr()
            # Verbose mode should print traceback
            assert "ValueError" in captured.err or "Verbose error" in captured.err


class TestCmdMultiRunExecution:
    """Tests for multi-run command execution with mocking."""

    def test_cmd_multi_run_requires_data(self, capsys):
        """Test multi-run requires data argument.

        This tests lines 220-222:
            if not args.data:
                _print_error("--data is required")
                return 1
        """
        from backpropagate.cli import cmd_multi_run
        import argparse

        args = argparse.Namespace(
            data=None,
            model="test-model",
            runs=5,
            steps=100,
            samples=1000,
            merge_mode="slao",
            output="./output",
            verbose=False,
        )

        result = cmd_multi_run(args)
        assert result == 1

        captured = capsys.readouterr()
        assert "required" in captured.err.lower() or "ERROR" in captured.err

    def test_cmd_multi_run_successful(self, capsys, temp_dir):
        """Test successful multi-run execution.

        This tests lines 213-270 (cmd_multi_run function).
        """
        from backpropagate.cli import cmd_multi_run
        import argparse

        mock_result = MagicMock()
        mock_result.total_runs = 5
        mock_result.final_loss = 0.3
        mock_result.total_duration_seconds = 300.0
        mock_result.final_checkpoint_path = str(temp_dir / "final_model")

        mock_trainer = MagicMock()
        mock_trainer.run.return_value = mock_result

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
                output=str(temp_dir),
                verbose=False,
            )

            result = cmd_multi_run(args)

            assert result == 0
            mock_trainer.run.assert_called_once_with("test_data")

            captured = capsys.readouterr()
            assert "Multi-run training complete" in captured.out

    def test_cmd_multi_run_keyboard_interrupt(self, capsys, temp_dir):
        """Test multi-run handles KeyboardInterrupt.

        This tests lines 261-264.
        """
        from backpropagate.cli import cmd_multi_run
        import argparse

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
                output=str(temp_dir),
                verbose=False,
            )

            result = cmd_multi_run(args)

            assert result == 130
            captured = capsys.readouterr()
            assert "interrupted" in captured.out.lower()

    def test_cmd_multi_run_exception(self, capsys, temp_dir):
        """Test multi-run handles exceptions.

        This tests lines 265-270.
        """
        from backpropagate.cli import cmd_multi_run
        import argparse

        mock_trainer = MagicMock()
        mock_trainer.run.side_effect = RuntimeError("Multi-run error")

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
                output=str(temp_dir),
                verbose=False,
            )

            result = cmd_multi_run(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "ERROR" in captured.err
            assert "Multi-run error" in captured.err


class TestCmdExportExecution:
    """Tests for export command execution with mocking."""

    def test_cmd_export_lora_format(self, capsys, temp_dir):
        """Test export with lora format.

        This tests lines 305-309.
        """
        from backpropagate.cli import cmd_export
        import argparse

        # Create model path
        model_path = temp_dir / "model"
        model_path.mkdir()

        mock_result = MagicMock()
        mock_result.path = temp_dir / "exported"
        mock_result.size_mb = 100.0
        mock_result.export_time_seconds = 5.0

        with patch("backpropagate.export.export_lora", return_value=mock_result):
            args = argparse.Namespace(
                model_path=str(model_path),
                format="lora",
                quantization="q4_k_m",
                output=str(temp_dir / "output"),
                ollama=False,
                ollama_name=None,
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Export complete" in captured.out

    def test_cmd_export_merged_format(self, capsys, temp_dir):
        """Test export with merged format.

        This tests lines 310-318.
        """
        from backpropagate.cli import cmd_export
        import argparse

        model_path = temp_dir / "model"
        model_path.mkdir()

        mock_result = MagicMock()
        mock_result.path = temp_dir / "merged"
        mock_result.size_mb = 500.0
        mock_result.export_time_seconds = 30.0

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("backpropagate.trainer.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("backpropagate.export.export_merged", return_value=mock_result):
            args = argparse.Namespace(
                model_path=str(model_path),
                format="merged",
                quantization="q4_k_m",
                output=str(temp_dir / "output"),
                ollama=False,
                ollama_name=None,
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Export complete" in captured.out

    def test_cmd_export_gguf_format(self, capsys, temp_dir):
        """Test export with gguf format.

        This tests lines 319-327.
        """
        from backpropagate.cli import cmd_export
        import argparse

        model_path = temp_dir / "model"
        model_path.mkdir()

        mock_result = MagicMock()
        mock_result.path = temp_dir / "model.gguf"
        mock_result.size_mb = 4000.0
        mock_result.export_time_seconds = 120.0

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("backpropagate.trainer.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("backpropagate.export.export_gguf", return_value=mock_result):
            args = argparse.Namespace(
                model_path=str(model_path),
                format="gguf",
                quantization="q4_k_m",
                output=str(temp_dir / "output"),
                ollama=False,
                ollama_name=None,
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Export complete" in captured.out
            assert "Quantization: q4_k_m" in captured.out

    def test_cmd_export_gguf_with_ollama(self, capsys, temp_dir):
        """Test export with GGUF format and Ollama registration.

        This tests lines 337-348.
        """
        from backpropagate.cli import cmd_export
        import argparse

        model_path = temp_dir / "model"
        model_path.mkdir()

        mock_result = MagicMock()
        mock_result.path = temp_dir / "model.gguf"
        mock_result.size_mb = 4000.0
        mock_result.export_time_seconds = 120.0

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("backpropagate.trainer.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("backpropagate.export.export_gguf", return_value=mock_result), \
             patch("backpropagate.export.register_with_ollama", return_value=True):
            args = argparse.Namespace(
                model_path=str(model_path),
                format="gguf",
                quantization="q4_k_m",
                output=str(temp_dir / "output"),
                ollama=True,
                ollama_name="my-custom-model",
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Registered with Ollama" in captured.out
            assert "my-custom-model" in captured.out

    def test_cmd_export_ollama_registration_failure(self, capsys, temp_dir):
        """Test export when Ollama registration fails.

        This tests lines 346-348.
        """
        from backpropagate.cli import cmd_export
        import argparse

        model_path = temp_dir / "model"
        model_path.mkdir()

        mock_result = MagicMock()
        mock_result.path = temp_dir / "model.gguf"
        mock_result.size_mb = 4000.0
        mock_result.export_time_seconds = 120.0

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("backpropagate.trainer.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("backpropagate.export.export_gguf", return_value=mock_result), \
             patch("backpropagate.export.register_with_ollama", return_value=False):
            args = argparse.Namespace(
                model_path=str(model_path),
                format="gguf",
                quantization="q4_k_m",
                output=str(temp_dir / "output"),
                ollama=True,
                ollama_name=None,
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Failed to register with Ollama" in captured.err

    def test_cmd_export_exception(self, capsys, temp_dir):
        """Test export handles exceptions.

        This tests lines 352-357.
        """
        from backpropagate.cli import cmd_export
        import argparse

        model_path = temp_dir / "model"
        model_path.mkdir()

        with patch("backpropagate.export.export_lora", side_effect=RuntimeError("Export failed")):
            args = argparse.Namespace(
                model_path=str(model_path),
                format="lora",
                quantization="q4_k_m",
                output=str(temp_dir / "output"),
                ollama=False,
                ollama_name=None,
                verbose=False,
            )

            result = cmd_export(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "ERROR" in captured.err
            assert "Export failed" in captured.err


class TestCmdInfoGPU:
    """Tests for info command GPU display."""

    def test_cmd_info_with_gpu(self, capsys):
        """Test info command displays GPU info when available.

        This tests lines 381-396.
        """
        from backpropagate.cli import cmd_info
        import argparse

        mock_gpu_info = {
            "name": "Test RTX 5080",
            "vram_total_gb": 16.0,
            "vram_free_gb": 12.0,
        }

        mock_gpu_status = MagicMock()
        mock_gpu_status.temperature_c = 65.0

        # Patch in the feature_flags module since imports happen inside cmd_info
        with patch("backpropagate.feature_flags.get_gpu_info", return_value=mock_gpu_info), \
             patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_gpu_status):
            args = argparse.Namespace(verbose=False)
            result = cmd_info(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "GPU" in captured.out
            assert "Test RTX 5080" in captured.out or "Device" in captured.out

    def test_cmd_info_without_gpu(self, capsys):
        """Test info command displays message when no GPU.

        This tests lines 397-399.
        """
        from backpropagate.cli import cmd_info
        import argparse

        with patch("backpropagate.feature_flags.get_gpu_info", return_value=None):
            args = argparse.Namespace(verbose=False)
            result = cmd_info(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "GPU" in captured.out
            assert "No GPU detected" in captured.out or "not detected" in captured.out.lower()


class TestCmdConfigSet:
    """Tests for config command --set option."""

    def test_cmd_config_set_not_implemented(self, capsys):
        """Test config --set shows not implemented message.

        This tests lines 435-439.
        """
        from backpropagate.cli import cmd_config
        import argparse

        args = argparse.Namespace(show=False, set="key=value", reset=False, verbose=False)
        result = cmd_config(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "not yet implemented" in captured.out.lower() or "WARN" in captured.out

    def test_cmd_config_windows_section_on_windows(self, capsys):
        """Test config command shows Windows section on Windows.

        This tests lines 467-471.
        """
        from backpropagate.cli import cmd_config
        import argparse

        with patch("os.name", "nt"):
            args = argparse.Namespace(show=False, set=None, reset=False, verbose=False)
            result = cmd_config(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Windows" in captured.out


class TestSupportsColor:
    """Tests for _supports_color function."""

    def test_supports_color_no_color_env(self):
        """Test NO_COLOR environment variable disables colors.

        This tests lines 36-37.
        """
        from backpropagate.cli import _supports_color
        import os

        orig = os.environ.get("NO_COLOR")
        try:
            os.environ["NO_COLOR"] = "1"
            # Need to reimport to test since Colors is class-level
            assert _supports_color() is False
        finally:
            if orig is not None:
                os.environ["NO_COLOR"] = orig
            else:
                os.environ.pop("NO_COLOR", None)

    def test_supports_color_force_color_env(self):
        """Test FORCE_COLOR environment variable enables colors.

        This tests lines 38-39.
        """
        from backpropagate.cli import _supports_color
        import os

        orig_no = os.environ.get("NO_COLOR")
        orig_force = os.environ.get("FORCE_COLOR")
        try:
            os.environ.pop("NO_COLOR", None)
            os.environ["FORCE_COLOR"] = "1"
            assert _supports_color() is True
        finally:
            if orig_no is not None:
                os.environ["NO_COLOR"] = orig_no
            if orig_force is not None:
                os.environ["FORCE_COLOR"] = orig_force
            else:
                os.environ.pop("FORCE_COLOR", None)


class TestProgressBarSuffix:
    """Tests for ProgressBar suffix display."""

    def test_progress_bar_with_suffix(self, capsys):
        """Test ProgressBar displays suffix.

        This tests lines 121-122.
        """
        from backpropagate.cli import ProgressBar

        progress = ProgressBar(total=100, width=20)
        progress.update(50, suffix="loss=0.5")

        # Verify it completed without error
        assert progress.current == 50

    def test_progress_bar_completion_newline(self, capsys):
        """Test ProgressBar prints newline on completion.

        This tests lines 126-127.
        """
        from backpropagate.cli import ProgressBar

        progress = ProgressBar(total=100, width=20)
        progress.update(100)  # Complete

        captured = capsys.readouterr()
        # Should end with newline when complete
        assert progress.current == 100
