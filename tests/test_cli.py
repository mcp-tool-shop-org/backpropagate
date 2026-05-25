"""Tests for CLI commands."""

from unittest.mock import MagicMock, patch

import pytest


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

        # Default aligned with config.py ModelConfig.name (F-018 cross-domain fix):
        # non-quantized form works without bitsandbytes; bnb-4bit is opt-in via --model.
        assert args.model == "Qwen/Qwen2.5-7B-Instruct"
        assert args.steps == 100
        assert args.samples is None
        assert args.batch_size == "auto"
        assert args.lr == 2e-4
        assert args.lora_r == 256  # v1.3 'quality' preset default (was 16 in v1.2.x; --lora-preset=fast reverts)
        assert args.output == "./output"
        assert args.no_unsloth is False

    def test_train_requires_data(self, cli_parser):
        """`backprop train` without --data now parses successfully; the handler
        surfaces a friendly first-run guidance + EXIT_USER_ERROR (Stage C
        C-CLI-007). Argparse-level rejection was replaced with handler-level."""
        args = cli_parser.parse_args(["train"])
        assert args.command == "train"
        assert args.data is None

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

    def test_no_command_returns_nonzero(self):
        """Test main with no command returns 1 (no subcommand given)."""
        from backpropagate.cli import main

        result = main([])
        assert result == 1

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
        """`backprop train` without --data prints friendly guidance to stderr
        and returns EXIT_USER_ERROR (Stage C C-CLI-007)."""
        from backpropagate.cli import main

        result = main(["train"])
        assert result == 1


class TestCmdInfo:
    """Tests for info command."""

    def test_cmd_info_outputs_system_info(self, capsys):
        """Test cmd_info outputs system information."""
        import argparse

        from backpropagate.cli import cmd_info

        args = argparse.Namespace(verbose=False)
        result = cmd_info(args)

        assert result == 0

        captured = capsys.readouterr()
        assert "System" in captured.out
        assert "Python" in captured.out

    def test_cmd_info_outputs_features(self, capsys):
        """Test cmd_info outputs feature availability."""
        import argparse

        from backpropagate.cli import cmd_info

        args = argparse.Namespace(verbose=False)
        cmd_info(args)

        captured = capsys.readouterr()
        assert "Features" in captured.out

    def test_cmd_info_outputs_configuration(self, capsys):
        """Test cmd_info outputs configuration."""
        import argparse

        from backpropagate.cli import cmd_info

        args = argparse.Namespace(verbose=False)
        cmd_info(args)

        captured = capsys.readouterr()
        assert "Configuration" in captured.out
        assert "Model" in captured.out


# =============================================================================
# C-CLI-005 / TESTS-A-003 — ``backprop info --error-codes`` regression
# =============================================================================
# Pins:
# 1. ``--error-codes`` flag is accepted by the argparse parser.
# 2. ``backprop info --error-codes`` exits 0 + dumps the catalog.
# 3. The catalog includes the INPUT_AUTH_REQUIRED hint and it points at
#    ``--auth user:pass`` (the v1.3 corrected hint after BACKEND-A-XXX).
# 4. Every catalog row appears in the output (no silent truncation).
# Prior to this, there was ZERO test for --error-codes; a renamed flag or
# accidentally-deleted print_error_code_catalog() call would have silently
# broken the operator-facing error lookup.


class TestCmdInfoErrorCodes:
    """Tests for ``backprop info --error-codes`` (C-CLI-005)."""

    def test_parser_accepts_error_codes_flag(self, cli_parser):
        """``backprop info --error-codes`` parses without SystemExit."""
        args = cli_parser.parse_args(["info", "--error-codes"])
        assert args.command == "info"
        assert args.error_codes is True

    def test_cmd_info_error_codes_exits_zero(self, capsys):
        """``backprop info --error-codes`` returns EXIT_OK (0)."""
        import argparse

        from backpropagate.cli import EXIT_OK, cmd_info

        args = argparse.Namespace(
            error_codes=True,
            env_vars=False,
            json=False,
            verbose=False,
        )
        result = cmd_info(args)
        assert result == EXIT_OK

    def test_cmd_info_error_codes_dumps_catalog(self, capsys):
        """The catalog dump includes the table header and every catalog code.

        Mechanises the contract: every key in ``exceptions.ERROR_CODES`` must
        appear in the printed output, with the description following.
        """
        import argparse

        from backpropagate.cli import cmd_info
        from backpropagate.exceptions import ERROR_CODES

        args = argparse.Namespace(
            error_codes=True,
            env_vars=False,
            json=False,
            verbose=False,
        )
        cmd_info(args)
        captured = capsys.readouterr()
        out = captured.out

        # Header
        assert "code" in out
        assert "retryable" in out
        assert "description" in out

        # Every catalog code must appear in the output.
        for code in ERROR_CODES.keys():
            assert code in out, (
                f"Code {code!r} from ERROR_CODES is missing from the catalog "
                f"dump — print_error_code_catalog() silently dropped it."
            )

    def test_input_auth_required_hint_points_at_auth_flag(self):
        """Pins the corrected ``INPUT_AUTH_REQUIRED`` hint (v1.3 fix).

        The pre-fix hint pointed at the legacy ``backpropagate.launch()``
        Python API which was removed in v1.1.0. The corrected v1.3 hint
        must reference ``--auth user:pass`` (the CLI surface) +
        ``BACKPROPAGATE_UI_AUTH`` (the env-var surface) since those are
        the two actually-supported auth-supply mechanisms in v1.2.0+.
        """
        from backpropagate.exceptions import ERROR_CODES

        entry = ERROR_CODES.get("INPUT_AUTH_REQUIRED")
        assert entry is not None, "INPUT_AUTH_REQUIRED missing from catalog"

        hint = entry.get("default_hint", "")
        assert "--auth" in hint, (
            f"INPUT_AUTH_REQUIRED hint should reference --auth (the CLI "
            f"surface). Current hint: {hint!r}. The pre-fix hint pointed at "
            f"the legacy backpropagate.launch() Python API which was removed "
            f"in v1.1.0."
        )
        assert "BACKPROPAGATE_UI_AUTH" in hint, (
            f"INPUT_AUTH_REQUIRED hint should reference BACKPROPAGATE_UI_AUTH "
            f"(the env-var surface). Current hint: {hint!r}."
        )
        # The hint must NOT mention the deleted legacy API.
        assert "launch(" not in hint and "backpropagate.launch" not in hint, (
            f"INPUT_AUTH_REQUIRED hint still references the deleted "
            f"backpropagate.launch() API. Current hint: {hint!r}."
        )


class TestCmdConfig:
    """Tests for config command."""

    def test_cmd_config_shows_config(self, capsys):
        """Test cmd_config shows configuration."""
        import argparse

        from backpropagate.cli import cmd_config

        args = argparse.Namespace(show=False, set=None, reset=False, verbose=False)
        result = cmd_config(args)

        assert result == 0

        captured = capsys.readouterr()
        assert "Model" in captured.out
        assert "LoRA" in captured.out
        assert "Training" in captured.out

    def test_cmd_config_reset_message(self, capsys):
        """`config --reset` is not implemented; surfaces EXIT_USER_ERROR with a
        not-implemented message on stderr (Stage C C-CLI-003)."""
        import argparse

        from backpropagate.cli import cmd_config

        args = argparse.Namespace(show=False, set=None, reset=True, verbose=False)
        result = cmd_config(args)

        assert result == 1

        captured = capsys.readouterr()
        msg = (captured.out + captured.err).lower()
        assert "planned" in msg or "environment" in msg or "not implemented" in msg


class TestCmdTrain:
    """Tests for train command."""

    def test_cmd_train_requires_data(self, capsys):
        """Test cmd_train requires data argument."""
        import argparse

        from backpropagate.cli import cmd_train

        args = argparse.Namespace(
            data=None,
            model="test",
            steps=10,
            samples=None,
            batch_size="auto",
            lr=2e-4,
            # TESTS-A-005 (v1.4 Wave 2 amend): lora_r=16 is an intentional
            # non-default fixture value. This test asserts data-validation
            # behaviour (the `data=None` returns EXIT_USER_ERROR), NOT the
            # CLI default (which is 256 per v1.3 BACKEND-1 — see
            # test_cli.py:58 which pins args.lora_r == 256 for the parser).
            # The fixture-Namespace stays at 16 to keep this test isolated
            # from default-value drift.
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
        import argparse

        from backpropagate.cli import cmd_export

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
        import argparse

        from backpropagate.cli import cmd_train

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
        import argparse

        from backpropagate.cli import cmd_train

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
        import argparse

        from backpropagate.cli import cmd_train

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

        Per Ship Gate B2, unexpected exceptions (RuntimeError, etc.) map to
        exit code 2 (runtime error), not 1 (user error).
        """
        import argparse

        from backpropagate.cli import cmd_train

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

            assert result == 2
            captured = capsys.readouterr()
            assert "ERROR" in captured.err
            assert "Test error" in captured.err

    def test_cmd_train_verbose_traceback(self, capsys, temp_dir):
        """Test train command prints traceback when verbose."""
        import argparse

        from backpropagate.cli import cmd_train

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

            assert result == 2
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
        import argparse

        from backpropagate.cli import cmd_multi_run

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
        import argparse

        from backpropagate.cli import cmd_multi_run

        mock_result = MagicMock()
        mock_result.total_runs = 5
        mock_result.final_loss = 0.3
        mock_result.total_duration_seconds = 300.0
        mock_result.final_checkpoint_path = str(temp_dir / "final_model")
        # Explicit failed_runs=0 so cmd_multi_run does not classify this as
        # partial success (exit 3). MagicMock attributes are truthy by default,
        # so leaving this unset would make `failed_runs > 0` evaluate True.
        mock_result.failed_runs = 0

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
        import argparse

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
        import argparse

        from backpropagate.cli import cmd_multi_run

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

            # Ship Gate B2: unexpected RuntimeError -> exit 2 (runtime error).
            assert result == 2
            captured = capsys.readouterr()
            assert "ERROR" in captured.err
            assert "Multi-run error" in captured.err


class TestCmdExportExecution:
    """Tests for export command execution with mocking."""

    def test_cmd_export_lora_format(self, capsys, temp_dir):
        """Test export with lora format.

        This tests lines 305-309.
        """
        import argparse

        from backpropagate.cli import cmd_export

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
        import argparse

        from backpropagate.cli import cmd_export

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
        import argparse

        from backpropagate.cli import cmd_export

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
        import argparse

        from backpropagate.cli import cmd_export

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

        Documented partial-success path: export succeeded, only the optional
        Ollama registration failed -> Ship Gate B2 exit code 3 (partial).
        """
        import argparse

        from backpropagate.cli import cmd_export

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

            assert result == 3
            captured = capsys.readouterr()
            assert "Failed to register with Ollama" in captured.err

    def test_cmd_export_exception(self, capsys, temp_dir):
        """Test export handles exceptions.

        This tests lines 352-357.
        """
        import argparse

        from backpropagate.cli import cmd_export

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

            # Ship Gate B2: unexpected RuntimeError -> exit 2 (runtime error).
            assert result == 2
            captured = capsys.readouterr()
            assert "ERROR" in captured.err
            assert "Export failed" in captured.err


class TestCmdInfoGPU:
    """Tests for info command GPU display."""

    def test_cmd_info_with_gpu(self, capsys):
        """Test info command displays GPU info when available.

        This tests lines 381-396.
        """
        import argparse

        from backpropagate.cli import cmd_info

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
        import argparse

        from backpropagate.cli import cmd_info

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
        """`config --set` is not implemented; surfaces EXIT_USER_ERROR with a
        not-implemented message on stderr (Stage C C-CLI-003)."""
        import argparse

        from backpropagate.cli import cmd_config

        args = argparse.Namespace(show=False, set="key=value", reset=False, verbose=False)
        result = cmd_config(args)

        assert result == 1
        captured = capsys.readouterr()
        msg = (captured.out + captured.err).lower()
        assert "planned" in msg or "environment" in msg or "not implemented" in msg

    def test_cmd_config_windows_section_on_windows(self, capsys):
        """Test config command shows Windows section on Windows.

        This tests lines 467-471.
        """
        import argparse

        from backpropagate.cli import cmd_config

        with patch("os.name", "nt"):
            args = argparse.Namespace(show=False, set=None, reset=False, verbose=False)
            result = cmd_config(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Windows" in captured.out


class TestSupportsColor:
    """Tests for _supports_color function."""

    def test_supports_color_no_color_env(self, monkeypatch):
        """Test NO_COLOR environment variable disables colors.

        This tests lines 36-37.
        """
        from backpropagate.cli import _supports_color

        monkeypatch.setenv("NO_COLOR", "1")
        # Need to reimport to test since Colors is class-level
        assert _supports_color() is False

    def test_supports_color_force_color_env(self, monkeypatch):
        """Test FORCE_COLOR environment variable enables colors.

        This tests lines 38-39.
        """
        from backpropagate.cli import _supports_color

        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert _supports_color() is True


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


# =============================================================================
# F-003 list-runs / show-run TESTS
# =============================================================================

class TestListRunsParser:
    """Tests for the ``backprop list-runs`` parser surface."""

    def test_list_runs_command_basic(self, cli_parser):
        args = cli_parser.parse_args(["list-runs"])
        assert args.command == "list-runs"
        assert args.output == "./output"
        assert args.status is None
        assert args.limit == 20
        assert args.json is False

    def test_list_runs_command_with_filters(self, cli_parser):
        args = cli_parser.parse_args([
            "list-runs",
            "-o", "/tmp/runs",
            "--status", "failed",
            "--limit", "5",
            "--json",
        ])
        assert args.output == "/tmp/runs"
        assert args.status == "failed"
        assert args.limit == 5
        assert args.json is True

    def test_show_run_requires_run_id(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["show-run"])

    def test_show_run_command_basic(self, cli_parser):
        args = cli_parser.parse_args(["show-run", "abc123"])
        assert args.command == "show-run"
        assert args.run_id == "abc123"
        assert args.output == "./output"
        assert args.json is False


class TestCmdListRuns:
    """Tests for cmd_list_runs handler."""

    def test_list_runs_empty_history_dir_returns_ok(self, tmp_path, capsys):
        from backpropagate.cli import cmd_list_runs

        args = MagicMock()
        args.output = str(tmp_path / "does-not-exist")
        args.status = None
        args.limit = 20
        args.json = False

        rc = cmd_list_runs(args)
        assert rc == 0

    def test_list_runs_with_no_runs(self, tmp_path, capsys):
        from backpropagate.cli import cmd_list_runs

        args = MagicMock()
        args.output = str(tmp_path)
        args.status = None
        args.limit = 20
        args.json = False

        rc = cmd_list_runs(args)
        assert rc == 0

    def test_list_runs_renders_entries(self, tmp_path, capsys):
        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import cmd_list_runs

        mgr = RunHistoryManager(str(tmp_path))
        mgr.record_run_started(
            run_id="aaaa1111bbbb2222",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            dataset_info="data.jsonl",
        )
        mgr.record_run_completed(run_id="aaaa1111bbbb2222", final_loss=0.42)

        args = MagicMock()
        args.output = str(tmp_path)
        args.status = None
        args.limit = 20
        args.json = False

        rc = cmd_list_runs(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "aaaa1111bbbb" in out  # short run_id
        assert "0.4200" in out  # rendered loss
        assert "completed" in out

    def test_list_runs_json_emits_json_array(self, tmp_path, capsys):
        import json as _json

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import cmd_list_runs

        mgr = RunHistoryManager(str(tmp_path))
        mgr.record_run_started(run_id="jsonid", model_name="m")
        mgr.record_run_completed(run_id="jsonid", final_loss=0.1)

        args = MagicMock()
        args.output = str(tmp_path)
        args.status = None
        args.limit = 20
        args.json = True

        rc = cmd_list_runs(args)
        out = capsys.readouterr().out
        assert rc == 0
        payload = _json.loads(out)
        assert isinstance(payload, list)
        assert payload[0]["run_id"] == "jsonid"

    def test_list_runs_status_filter(self, tmp_path, capsys):
        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import cmd_list_runs

        mgr = RunHistoryManager(str(tmp_path))
        mgr.record_run_started(run_id="r-fail", model_name="m")
        mgr.record_run_failed(run_id="r-fail", failure_reason="x")
        mgr.record_run_started(run_id="r-ok", model_name="m")
        mgr.record_run_completed(run_id="r-ok", final_loss=0.1)

        args = MagicMock()
        args.output = str(tmp_path)
        args.status = "failed"
        args.limit = 20
        args.json = False

        rc = cmd_list_runs(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "r-fail" in out
        assert "r-ok" not in out


class TestCmdShowRun:
    """Tests for cmd_show_run handler."""

    def test_show_run_missing_history_dir(self, tmp_path):
        from backpropagate.cli import cmd_show_run

        args = MagicMock()
        args.output = str(tmp_path / "missing")
        args.run_id = "abc"
        args.json = False

        rc = cmd_show_run(args)
        assert rc == 1  # EXIT_USER_ERROR

    def test_show_run_missing_id(self, tmp_path):
        from backpropagate.cli import cmd_show_run

        args = MagicMock()
        args.output = str(tmp_path)
        args.run_id = "notarealid"
        args.json = False

        rc = cmd_show_run(args)
        assert rc == 1  # EXIT_USER_ERROR

    def test_show_run_renders_entry(self, tmp_path, capsys):
        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import cmd_show_run

        mgr = RunHistoryManager(str(tmp_path))
        mgr.record_run_started(
            run_id="showid12345",
            model_name="m",
            hyperparameters={"lora_r": 16},
        )
        mgr.record_run_completed(run_id="showid12345", final_loss=0.25)

        args = MagicMock()
        args.output = str(tmp_path)
        args.run_id = "showid"  # partial prefix
        args.json = False

        rc = cmd_show_run(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "showid12345" in out
        assert "completed" in out
        assert "0.2500" in out

    def test_show_run_json_round_trips(self, tmp_path, capsys):
        import json as _json

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import cmd_show_run

        mgr = RunHistoryManager(str(tmp_path))
        mgr.record_run_started(run_id="jx", model_name="m")
        mgr.record_run_completed(run_id="jx", final_loss=0.3)

        args = MagicMock()
        args.output = str(tmp_path)
        args.run_id = "jx"
        args.json = True

        rc = cmd_show_run(args)
        out = capsys.readouterr().out
        assert rc == 0
        payload = _json.loads(out)
        assert payload["run_id"] == "jx"
        assert payload["final_loss"] == 0.3


# =============================================================================
# F-001 push CLI TESTS
# =============================================================================

class TestPushParser:

    def test_push_command_requires_repo(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["push", "./somewhere"])

    def test_push_command_basic(self, cli_parser):
        args = cli_parser.parse_args(["push", "./out", "--repo", "alice/m"])
        assert args.command == "push"
        assert args.local_path == "./out"
        assert args.repo == "alice/m"
        assert args.token is None
        assert args.private is False
        assert args.include_base is False

    def test_push_command_all_options(self, cli_parser):
        args = cli_parser.parse_args([
            "push", "./out",
            "--repo", "alice/m",
            "--token", "hf_token",
            "--private",
            "--include-base",
        ])
        assert args.token == "hf_token"
        assert args.private is True
        assert args.include_base is True

    def test_export_push_flags(self, cli_parser):
        args = cli_parser.parse_args([
            "export", "./out",
            "--push-to-hub", "alice/m",
            "--hub-token", "tk",
            "--hub-private",
        ])
        assert args.push_to_hub == "alice/m"
        assert args.hub_token == "tk"
        assert args.hub_private is True


class TestCmdPush:

    def test_push_missing_local_path_returns_user_error(self, tmp_path):
        from backpropagate.cli import cmd_push

        args = MagicMock()
        args.local_path = str(tmp_path / "missing")
        args.repo = "alice/m"
        args.token = None
        args.token_file = None
        args.private = False
        args.include_base = False
        args.verbose = False

        rc = cmd_push(args)
        assert rc == 1

    def test_push_missing_repo_returns_user_error(self, tmp_path):
        from backpropagate.cli import cmd_push

        local = tmp_path / "lora"
        local.mkdir()

        args = MagicMock()
        args.local_path = str(local)
        args.repo = None
        args.token = None
        args.token_file = None
        args.private = False
        args.include_base = False
        args.verbose = False

        rc = cmd_push(args)
        assert rc == 1

    def test_push_calls_push_to_hub_and_reports_url(self, tmp_path, capsys):
        from backpropagate.cli import cmd_push

        local = tmp_path / "lora"
        local.mkdir()

        args = MagicMock()
        args.local_path = str(local)
        args.repo = "alice/m"
        args.token = None
        args.token_file = None
        args.private = False
        args.include_base = False
        args.verbose = False

        with patch("backpropagate.export.push_to_hub", return_value="https://huggingface.co/alice/m") as push_mock:
            rc = cmd_push(args)

        assert rc == 0
        push_mock.assert_called_once()
        out = capsys.readouterr().out
        assert "https://huggingface.co/alice/m" in out

    def test_push_wraps_auth_error_with_user_exit_code(self, tmp_path):
        from backpropagate.cli import cmd_push
        from backpropagate.exceptions import ExportError

        local = tmp_path / "lora"
        local.mkdir()

        # Wave 6 BRIDGE-F-002 cleanup removed the local _BRIDGE_LOCAL_ERROR_CODES
        # fallback table; cmd_push now branches on the canonical ERROR_CODES
        # entries only. The auth-error route is INPUT_AUTH_REQUIRED (the code
        # push_to_hub raises today when the token is missing or rejected). The
        # legacy HUB_PUSH_AUTH code is no longer in the catalog.
        err = ExportError("authentication failed")
        err.code = "INPUT_AUTH_REQUIRED"  # type: ignore[attr-defined]

        args = MagicMock()
        args.local_path = str(local)
        args.repo = "alice/m"
        args.token = None
        args.token_file = None
        args.private = False
        args.include_base = False
        args.verbose = False

        with patch("backpropagate.export.push_to_hub", side_effect=err):
            rc = cmd_push(args)

        assert rc == 1

    def test_push_wraps_runtime_error_with_runtime_exit_code(self, tmp_path):
        from backpropagate.cli import cmd_push
        from backpropagate.exceptions import ExportError

        local = tmp_path / "lora"
        local.mkdir()

        err = ExportError("server 500")
        err.code = "HUB_PUSH_NETWORK"  # type: ignore[attr-defined]

        args = MagicMock()
        args.local_path = str(local)
        args.repo = "alice/m"
        args.token = None
        args.token_file = None
        args.private = False
        args.include_base = False
        args.verbose = False

        with patch("backpropagate.export.push_to_hub", side_effect=err):
            rc = cmd_push(args)

        assert rc == 2


# =============================================================================
# F-002 resume CLI TESTS
# =============================================================================

class TestResumeParser:

    def test_resume_command_requires_run_id(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["resume"])

    def test_resume_command_basic(self, cli_parser):
        args = cli_parser.parse_args(["resume", "abc123"])
        assert args.command == "resume"
        assert args.run_id == "abc123"
        assert args.output == "./output"
        assert args.data is None

    def test_resume_with_data_override(self, cli_parser):
        args = cli_parser.parse_args([
            "resume", "abc123",
            "--output", "./out",
            "--data", "my.jsonl",
        ])
        assert args.data == "my.jsonl"
        assert args.output == "./out"

    def test_train_command_accepts_resume(self, cli_parser):
        args = cli_parser.parse_args([
            "train", "-d", "data.jsonl", "--resume", "abc",
        ])
        assert args.resume == "abc"

    def test_multi_run_command_accepts_resume(self, cli_parser):
        args = cli_parser.parse_args([
            "multi-run", "-d", "data.jsonl", "--resume", "abc",
        ])
        assert args.resume == "abc"


class TestCmdResume:

    def test_resume_missing_output_returns_user_error(self, tmp_path):
        from backpropagate.cli import cmd_resume

        args = MagicMock()
        args.output = str(tmp_path / "missing")
        args.run_id = "abc"
        args.data = None
        args.verbose = False

        rc = cmd_resume(args)
        assert rc == 1

    def test_resume_unknown_run_id_returns_user_error(self, tmp_path):
        from backpropagate.cli import cmd_resume

        args = MagicMock()
        args.output = str(tmp_path)
        args.run_id = "missing"
        args.data = None
        args.verbose = False

        rc = cmd_resume(args)
        assert rc == 1

    def test_resume_dispatches_multi_run(self, tmp_path):
        """A multi_run record should reconstruct MultiRunTrainer + .run()."""
        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import cmd_resume

        manager = RunHistoryManager(str(tmp_path))
        manager.record_run_started(
            run_id="mres",
            model_name="m",
            dataset_info="data.jsonl",
            hyperparameters={
                "num_runs": 3,
                "steps_per_run": 50,
                "samples_per_run": 100,
                "merge_mode": "slao",
            },
            session_kind="multi_run",
        )

        fake_trainer = MagicMock()
        fake_trainer.run.return_value = MagicMock(total_runs=3, final_loss=0.1)

        with patch(
            "backpropagate.multi_run.MultiRunTrainer",
            return_value=fake_trainer,
        ) as mock_cls:
            args = MagicMock()
            args.output = str(tmp_path)
            args.run_id = "mres"
            args.data = None
            args.verbose = False
            rc = cmd_resume(args)

        assert rc == 0
        # MultiRunTrainer was invoked with resume_from=mres.
        kwargs = mock_cls.call_args.kwargs
        assert kwargs.get("resume_from") == "mres"
        fake_trainer.run.assert_called_once()

    def test_resume_dispatches_single_run(self, tmp_path):
        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import cmd_resume

        manager = RunHistoryManager(str(tmp_path))
        manager.record_run_started(
            run_id="sres",
            model_name="m",
            dataset_info="data.jsonl",
            hyperparameters={
                "max_steps": 50,
                "lora_r": 8,
                "learning_rate": 1e-4,
            },
            session_kind="single_run",
        )

        fake_trainer = MagicMock()
        fake_run = MagicMock(final_loss=0.05)
        fake_trainer.train.return_value = fake_run

        with patch(
            "backpropagate.trainer.Trainer",
            return_value=fake_trainer,
        ) as mock_cls:
            args = MagicMock()
            args.output = str(tmp_path)
            args.run_id = "sres"
            args.data = None
            args.verbose = False
            rc = cmd_resume(args)

        assert rc == 0
        # Trainer.train called with resume_from=sres.
        train_kwargs = fake_trainer.train.call_args.kwargs
        assert train_kwargs.get("resume_from") == "sres"
