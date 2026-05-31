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
        # v1.3 BACKEND-1 contract: 'quality' preset default. Was 16 in
        # v1.2.x; --lora-preset=fast reverts to rank 16. Pinning here so
        # the CLI parser default cannot silently re-diverge from config.py
        # (TESTS-A-006 v1.5 sweep tracks any UI/CLI lora_r divergence).
        assert args.lora_r == 256, (
            f"CLI parser default for --lora-r drifted from v1.3 BACKEND-1 "
            f"'quality' contract: expected 256, got {args.lora_r}. If you "
            f"are intentionally reverting the default, also update "
            f"backpropagate/config.py ModelConfig + handbook/lora_presets.md "
            f"and remove the LoraConfig 'quality' preset alias."
        )
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

    # CLI-A-001 (Wave A1): the root-level logging flags (--verbose/-v,
    # --log-level, --log-format, --log-file) must work BOTH before AND after
    # the subcommand. Pre-fix they lived only on the top-level parser, so
    # ``backprop train --verbose`` (flag after the subcommand) died with
    # ``error: unrecognized arguments: --verbose`` (exit 2) — even though
    # nearly every error handler advises "Run with --verbose for full
    # traceback". The parent-parser fix registers them on every subparser.
    def test_verbose_after_subcommand(self, cli_parser):
        """`backprop train --data x --verbose` (flag AFTER subcommand)."""
        args = cli_parser.parse_args(["train", "--data", "x", "--verbose"])
        assert args.verbose is True
        assert args.data == "x"

    def test_verbose_before_subcommand_still_works(self, cli_parser):
        """`backprop --verbose train --data x` (flag BEFORE subcommand) must
        not regress — the parent-parser fix must not clobber the top-level
        value back to the subparser default during the subparse."""
        args = cli_parser.parse_args(["--verbose", "train", "--data", "x"])
        assert args.verbose is True
        assert args.data == "x"

    def test_verbose_short_flag_after_subcommand(self, cli_parser):
        """The ``-v`` short form must also work after the subcommand."""
        args = cli_parser.parse_args(["train", "--data", "x", "-v"])
        assert args.verbose is True

    def test_verbose_default_suppressed_in_parser_backfilled_by_handler(self):
        """SUPPRESS contract: the four shared logging flags use
        default=argparse.SUPPRESS so a value set BEFORE the subcommand is not
        clobbered by the subparse. The flip side is that an omitted flag is
        ABSENT from the bare parser namespace; the backfill table provides the
        real defaults that main() applies before any handler runs."""
        from backpropagate.cli import _COMMON_FLAG_DEFAULTS, create_parser

        args = create_parser().parse_args(["info"])
        # Pre-backfill: absent (this is what lets before-subcommand placement win).
        assert not hasattr(args, "verbose")
        # The backfill table main() applies is the documented default surface.
        assert _COMMON_FLAG_DEFAULTS["verbose"] is False
        assert _COMMON_FLAG_DEFAULTS["log_level"] is None

    def test_log_level_after_subcommand(self, cli_parser):
        """`backprop export ./m --log-level DEBUG` (spot-check a second
        subcommand with a positional + the --log-level flag after it)."""
        args = cli_parser.parse_args(["export", "./m", "--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"
        assert args.model_path == "./m"

    def test_log_level_before_subcommand_not_clobbered(self, cli_parser):
        """`backprop --log-level DEBUG train` must survive the subparse."""
        args = cli_parser.parse_args(["--log-level", "DEBUG", "train", "--data", "x"])
        assert args.log_level == "DEBUG"

    def test_log_format_and_file_after_subcommand(self, cli_parser):
        """--log-format / --log-file must also be accepted after the subcommand."""
        args = cli_parser.parse_args(
            ["train", "--data", "x", "--log-format", "json", "--log-file", "run.log"]
        )
        assert args.log_format == "json"
        assert args.log_file == "run.log"

    @pytest.mark.parametrize(
        "argv",
        [
            ["train", "--data", "x", "--verbose"],
            ["multi-run", "--data", "x", "--verbose"],
            ["export", "./m", "--verbose"],
            ["info", "--verbose"],
            ["config", "--verbose"],
            ["list-runs", "--verbose"],
            ["runs", "--verbose"],
            ["show-run", "abc123", "--verbose"],
            ["diff-runs", "a", "b", "--verbose"],
            ["replay", "abc123", "--verbose"],
            ["export-runs", "--verbose"],
            ["resume", "abc123", "--verbose"],
            ["push", "./m", "--repo", "a/b", "--verbose"],
            ["validate", "x.jsonl", "--verbose"],
            ["estimate-vram", "--verbose"],
            ["ui", "--verbose"],
        ],
    )
    def test_verbose_accepted_after_every_subcommand(self, cli_parser, argv):
        """Family-of-call-sites probe: every subcommand must accept --verbose
        AFTER the subcommand verb, not just the top-level parser."""
        args = cli_parser.parse_args(argv)
        assert args.verbose is True


class TestHostStr:
    """CLI-A-007: `backprop ui --host` argparse type validator (`_host_str`).

    Pre-fix `--host` had no `type=`, so malformed values (option-shaped,
    whitespace, junk hostnames) slipped past argparse and only failed deep
    in the Reflex subprocess. Robustness/UX hardening — NOT a security
    control (the DNS-rebinding gate stays in cmd_ui).
    """

    @pytest.mark.parametrize(
        "value",
        ["127.0.0.1", "0.0.0.0", "192.168.1.50", "::1", "localhost",
         "my-host.local", "example.com", "[::1]"],
    )
    def test_valid_hosts_accepted(self, value):
        from backpropagate.cli import _host_str

        assert _host_str(value) == value

    @pytest.mark.parametrize(
        "value",
        ["-0.0.0.0", "--evil", "0.0.0.0 ", " 127.0.0.1", "a b",
         "host\twith\ttab", "", "bad_host!", "-.foo", "foo-.bar",
         "a" * 300],
    )
    def test_invalid_hosts_rejected(self, value):
        import argparse

        from backpropagate.cli import _host_str

        with pytest.raises(argparse.ArgumentTypeError):
            _host_str(value)

    def test_parser_accepts_valid_host(self, cli_parser):
        """End-to-end: a valid --host parses (with --auth for the LAN bind)."""
        args = cli_parser.parse_args(
            ["ui", "--host", "0.0.0.0", "--auth", "alice:s3cret"]
        )
        assert args.host == "0.0.0.0"

    def test_parser_rejects_option_shaped_host(self, cli_parser):
        """`--host -0.0.0.0` (option-shaped) is rejected at argparse, exit 2."""
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["ui", "--host", "-0.0.0.0"])


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

    def test_verbose_after_subcommand_reaches_handler(self):
        """CLI-A-001 end-to-end: `backprop info --verbose` (flag AFTER the
        subcommand) must run the handler cleanly (rc 0), not die in argparse
        with `unrecognized arguments: --verbose` (the pre-fix exit 2). Also
        proves main()'s SUPPRESS backfill leaves args.verbose readable so the
        handler's `if args.verbose:` reads don't AttributeError."""
        from backpropagate.cli import main

        assert main(["info", "--verbose"]) == 0
        # Sanity: omitting the flag (backfill path) still runs the handler.
        assert main(["info"]) == 0

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

    def test_main_catchall_redacts_unexpected_exception(self, capsys, monkeypatch):
        """CLI-A-003: the last-resort `except Exception` in main() redacts
        secrets in the printed message. A handler that lets an exception
        escape (bypassing its own redaction) must NOT leak a credential to
        stderr."""
        import backpropagate.cli as cli

        def _boom(_args):
            raise RuntimeError("download failed for api_key=EXAMPLE-NOT-A-REAL-KEY")

        # set_defaults(func=cmd_info) binds at parser-build time, so patch
        # the module attr BEFORE main() builds the parser.
        monkeypatch.setattr(cli, "cmd_info", _boom)
        monkeypatch.delenv("BACKPROPAGATE_DEBUG", raising=False)

        rc = cli.main(["info"])
        err = capsys.readouterr().err

        assert rc != 0
        assert "Unexpected error" in err
        assert "EXAMPLE-NOT-A-REAL-KEY" not in err, (
            f"CLI-A-003: catch-all leaked a credential. stderr: {err!r}"
        )
        assert "<REDACTED>" in err

    def test_main_catchall_redacts_backpropagate_error(self, capsys, monkeypatch):
        """CLI-A-003: the last-resort `except BackpropagateError` in main()
        redacts secrets too (the pre-try-raise path that bypasses the
        per-handler redaction)."""
        import backpropagate.cli as cli
        from backpropagate.exceptions import BackpropagateError

        def _boom(_args):
            raise BackpropagateError(
                "hub push failed: token=hunter2!@#secret_value"
            )

        monkeypatch.setattr(cli, "cmd_info", _boom)
        monkeypatch.delenv("BACKPROPAGATE_DEBUG", raising=False)

        cli.main(["info"])
        err = capsys.readouterr().err

        assert "hunter2" not in err, (
            f"CLI-A-003: BackpropagateError handler leaked a credential. "
            f"stderr: {err!r}"
        )
        assert "<REDACTED>" in err


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
# TESTS-F-005 (v1.4 Wave 6b): BRIDGE-B-013 (Stage C) schema_version
# pinning across every CLI ``--json`` emitter.
# =============================================================================
#
# Pre-Stage-C only ``runs --json`` carried a schema_version field — peers
# (info, info --env-vars, list-runs, show-run, diff-runs, export-runs,
# estimate-vram) emitted bare JSON dicts/arrays so a future shape change
# was indistinguishable from an additive field addition. The Stage C fix
# at backpropagate/cli.py:3610 introduced ``CLI_JSON_SCHEMA_VERSION = "1"``
# and threaded it onto every emitter as an additive field.
#
# Per the inherited doctrine ratchet
# [[grep-all-instances-when-fixing-pattern]], we pin schema_version on ALL
# documented --json emitters here (not just one) so a partial regression
# that drops the field on one surface gets caught.
#
# Emitters covered:
#   1. backprop info --json                      (top-level field)
#   2. backprop info --env-vars --json           (per-row injection)
#   3. backprop list-runs --json                 (per-row injection)
#   4. backprop runs --json                      (top-level field;
#                                                  RUNS_JSON_SCHEMA_VERSION
#                                                  twin)
#   5. backprop show-run --json                  (additive peer)
#   6. backprop diff-runs --format=json          (top-level field)
#   7. backprop export-runs (JSONL stdout)       (per-record field)
#   8. backprop estimate-vram --json             (top-level field)
#
# NOTE: `backprop info --error-codes` emits a plaintext catalog table
# (not JSON) — there is no `--error-codes --json` emitter to pin today.
# When that surface lands the matching test should be added here.
# =============================================================================


class TestCliJsonSchemaVersion:
    """Pin every documented ``--json`` emitter ships a schema_version field.

    The contract: ``CLI_JSON_SCHEMA_VERSION`` is the canonical name in
    ``backpropagate.cli``; every emitter must thread it into the output.
    The constant is "1" today; the bump policy lives in the docstring at
    cli.py:3595.
    """

    def test_cli_json_schema_version_constant_exists(self):
        """The shared constant is importable from cli.py."""
        from backpropagate.cli import CLI_JSON_SCHEMA_VERSION

        assert isinstance(CLI_JSON_SCHEMA_VERSION, str), (
            f"CLI_JSON_SCHEMA_VERSION must be a string; got "
            f"{type(CLI_JSON_SCHEMA_VERSION).__name__}."
        )
        assert CLI_JSON_SCHEMA_VERSION, (
            "CLI_JSON_SCHEMA_VERSION must be non-empty."
        )

    def test_info_json_carries_schema_version(self, capsys):
        """``backprop info --json`` has schema_version at top level."""
        import argparse
        import json as _json

        from backpropagate.cli import CLI_JSON_SCHEMA_VERSION, cmd_info

        args = argparse.Namespace(
            error_codes=False,
            env_vars=False,
            json=True,
            verbose=False,
        )
        rc = cmd_info(args)
        out = capsys.readouterr().out
        assert rc == 0
        payload = _json.loads(out)
        assert "schema_version" in payload, (
            "BRIDGE-B-013 regression: ``backprop info --json`` is "
            "missing the schema_version field. cli.py:1900 should "
            "set payload['schema_version'] = CLI_JSON_SCHEMA_VERSION."
        )
        assert payload["schema_version"] == CLI_JSON_SCHEMA_VERSION, (
            f"info --json schema_version drifted from the shared "
            f"constant: payload={payload['schema_version']!r} vs "
            f"constant={CLI_JSON_SCHEMA_VERSION!r}."
        )

    def test_info_env_vars_json_carries_schema_version_on_each_row(self, capsys):
        """``backprop info --env-vars --json`` injects schema_version
        on each row (the surface is a JSON array, not a dict envelope).
        """
        import argparse
        import json as _json

        from backpropagate.cli import CLI_JSON_SCHEMA_VERSION, cmd_info

        args = argparse.Namespace(
            error_codes=False,
            env_vars=True,
            json=True,
            verbose=False,
        )
        rc = cmd_info(args)
        out = capsys.readouterr().out
        assert rc == 0
        payload = _json.loads(out)
        assert isinstance(payload, list), (
            f"info --env-vars --json shape changed; expected list, "
            f"got {type(payload).__name__}."
        )
        assert payload, "info --env-vars --json produced empty list; cannot pin schema_version."
        for row in payload:
            assert "schema_version" in row, (
                "BRIDGE-B-013 regression: ``info --env-vars --json`` "
                "row is missing schema_version. Per-row injection "
                "lives at cli.py:1853."
            )
            assert row["schema_version"] == CLI_JSON_SCHEMA_VERSION

    def test_list_runs_json_carries_schema_version_per_row(self, tmp_path, capsys):
        """``backprop list-runs --json`` injects schema_version per row."""
        import json as _json

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import CLI_JSON_SCHEMA_VERSION, cmd_list_runs

        mgr = RunHistoryManager(str(tmp_path))
        mgr.record_run_started(run_id="lr-schema", model_name="m")
        mgr.record_run_completed(run_id="lr-schema", final_loss=0.42)

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
        assert payload, "list-runs --json produced empty list."
        for row in payload:
            assert "schema_version" in row, (
                "BRIDGE-B-013 regression: list-runs --json row is "
                "missing schema_version. Per-row injection lives at "
                "cli.py:3555."
            )
            assert row["schema_version"] == CLI_JSON_SCHEMA_VERSION

    def test_runs_json_carries_schema_version_top_level(self, tmp_path, capsys):
        """``backprop runs --json`` carries schema_version at the
        top level (BRIDGE-F-001 — predates the BRIDGE-B-013 sweep but
        uses the same contract name via ``RUNS_JSON_SCHEMA_VERSION``).
        """
        import json as _json

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import RUNS_JSON_SCHEMA_VERSION, cmd_runs

        mgr = RunHistoryManager(str(tmp_path))
        mgr.record_run_started(run_id="runs-schema", model_name="m")
        mgr.record_run_completed(run_id="runs-schema", final_loss=0.1)

        args = MagicMock()
        args.output = str(tmp_path)
        args.status = None
        args.limit = 20
        args.json = True

        rc = cmd_runs(args)
        out = capsys.readouterr().out
        assert rc == 0
        payload = _json.loads(out)
        assert payload.get("schema_version") == RUNS_JSON_SCHEMA_VERSION, (
            f"BRIDGE-F-001 regression: runs --json schema_version = "
            f"{payload.get('schema_version')!r}; expected "
            f"{RUNS_JSON_SCHEMA_VERSION!r}."
        )

    def test_show_run_json_carries_schema_version(self, tmp_path, capsys):
        """``backprop show-run --json`` has schema_version as an
        additive peer field on the run dict.
        """
        import json as _json

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import CLI_JSON_SCHEMA_VERSION, cmd_show_run

        mgr = RunHistoryManager(str(tmp_path))
        mgr.record_run_started(run_id="sr-schema", model_name="m")
        mgr.record_run_completed(run_id="sr-schema", final_loss=0.2)

        args = MagicMock()
        args.output = str(tmp_path)
        args.run_id = "sr-schema"
        args.json = True

        rc = cmd_show_run(args)
        out = capsys.readouterr().out
        assert rc == 0
        payload = _json.loads(out)
        assert payload.get("schema_version") == CLI_JSON_SCHEMA_VERSION, (
            "BRIDGE-B-013 regression: show-run --json schema_version "
            "missing or drifted. Additive injection lives at "
            f"cli.py:3800. payload={payload.get('schema_version')!r}."
        )

    def test_diff_runs_format_json_carries_schema_version(self, tmp_path, capsys):
        """``backprop diff-runs --format=json`` has schema_version at
        the top level alongside ``run_a`` / ``run_b`` / ``diff``.
        """
        import argparse
        import json as _json
        from datetime import datetime, timezone

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import CLI_JSON_SCHEMA_VERSION, cmd_diff_runs

        mgr = RunHistoryManager(str(tmp_path))
        mgr._save([
            {
                "run_id": "diff-a",
                "status": "completed",
                "model_name": "m-a",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "final_loss": 0.5,
            },
            {
                "run_id": "diff-b",
                "status": "completed",
                "model_name": "m-b",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "final_loss": 0.4,
            },
        ])

        args = argparse.Namespace(
            run_id_a="diff-a",
            run_id_b="diff-b",
            output=str(tmp_path),
            format="json",
        )
        rc = cmd_diff_runs(args)
        out = capsys.readouterr().out
        assert rc == 0
        payload = _json.loads(out)
        assert payload.get("schema_version") == CLI_JSON_SCHEMA_VERSION, (
            "BRIDGE-B-013 regression: diff-runs --format=json "
            "schema_version missing or drifted. Top-level injection "
            f"lives at cli.py:4006. payload={payload.get('schema_version')!r}."
        )

    def test_export_runs_jsonl_carries_schema_version_per_record(self, tmp_path, capsys):
        """``backprop export-runs`` emits one JSONL row per run, each
        with a schema_version field. Per-record injection (additive)
        keeps consumers that ignore unknown keys working byte-identically.
        """
        import argparse
        import json as _json
        from datetime import datetime, timezone

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import CLI_JSON_SCHEMA_VERSION, cmd_export_runs

        mgr = RunHistoryManager(str(tmp_path))
        mgr._save([
            {
                "run_id": "exp-a",
                "status": "completed",
                "model_name": "m",
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "run_id": "exp-b",
                "status": "completed",
                "model_name": "m",
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        ])

        args = argparse.Namespace(
            output=str(tmp_path),
            format="jsonl",
            to=None,
            status=None,
        )
        rc = cmd_export_runs(args)
        out = capsys.readouterr().out
        assert rc == 0
        lines = [ln for ln in out.splitlines() if ln.strip()]
        assert len(lines) == 2
        for ln in lines:
            record = _json.loads(ln)
            assert record.get("schema_version") == CLI_JSON_SCHEMA_VERSION, (
                "BRIDGE-B-013 regression: export-runs JSONL record "
                "missing schema_version. Per-record injection lives "
                "at cli.py:4508. "
                f"record_keys={sorted(record.keys())!r}."
            )

    def test_estimate_vram_json_carries_schema_version(self, capsys):
        """``backprop estimate-vram --json`` has schema_version at top
        level alongside model + vram_gb + tiers.
        """
        import argparse
        import json as _json

        from backpropagate.cli import CLI_JSON_SCHEMA_VERSION, cmd_estimate_vram

        args = argparse.Namespace(
            model="Qwen/Qwen2.5-7B-Instruct",
            vram_gb=16.0,  # bypass torch.cuda probe
            json=True,
        )
        rc = cmd_estimate_vram(args)
        out = capsys.readouterr().out
        assert rc == 0
        payload = _json.loads(out)
        assert payload.get("schema_version") == CLI_JSON_SCHEMA_VERSION, (
            "BRIDGE-B-013 regression: estimate-vram --json "
            "schema_version missing or drifted. Top-level injection "
            f"lives at cli.py:4740. payload={payload.get('schema_version')!r}."
        )


# =============================================================================
# TESTS-F-006 (v1.4 Wave 6b): BRIDGE-A-004 (Wave 2) _read_hub_token_file
# direct coverage.
# =============================================================================
#
# Wave 2 BRIDGE-A-004 added ``backprop push --hub-token-file <path>`` /
# ``backprop export --hub-token-file <path>`` plumbing to keep the HF
# token off the argv surface (where it leaks to ``ps aux`` + shell
# history). The helper at cli.py:1001 owns the error-path discipline:
# existence check, mode-0600 verification on POSIX (warn, don't refuse),
# content read with a clear error on empty/unreadable files. Returns the
# stripped token; raises UserInputError on any failure mode so the
# catch-all in cmd_push / cmd_export emits a friendly redacted message
# instead of a stack trace.
#
# Pre-Wave-6b TESTS-F-006 the helper had ZERO direct coverage — its 6
# error paths could regress silently. This file pins each one.
# =============================================================================


class TestReadHubTokenFile:
    """Direct tests for ``_read_hub_token_file`` (Wave 2 BRIDGE-A-004)."""

    def test_happy_path_returns_stripped_token(self, tmp_path):
        """File exists + has token ⇒ returns the stripped string."""
        import os

        from backpropagate.cli import _read_hub_token_file

        token_path = tmp_path / "hf_token"
        token_path.write_text("hf_abc123\n", encoding="utf-8")
        if os.name == "posix":
            os.chmod(token_path, 0o600)

        result = _read_hub_token_file(str(token_path), flag_name="--hub-token-file")
        assert result == "hf_abc123", (
            f"_read_hub_token_file must strip trailing whitespace; "
            f"got {result!r}."
        )

    def test_missing_path_raises_user_input_error(self, tmp_path):
        """Non-existent path ⇒ UserInputError with named-path hint."""
        from backpropagate.cli import _read_hub_token_file
        from backpropagate.exceptions import UserInputError

        missing_path = tmp_path / "does_not_exist"

        with pytest.raises(UserInputError) as exc_info:
            _read_hub_token_file(str(missing_path), flag_name="--hub-token-file")

        message = str(exc_info.value)
        # The error message must name the path so the operator sees
        # WHICH file is missing (vs a generic 'file not found').
        assert str(missing_path) in message, (
            f"Wave 2 BRIDGE-A-004 contract: missing-path error must "
            f"name the resolved path. Got message: {message!r}."
        )
        # Code is the validation-failure code so cli.py's catch-all
        # emits the right exit code (EXIT_USER_ERROR).
        assert getattr(exc_info.value, "code", None) == "INPUT_VALIDATION_FAILED", (
            f"_read_hub_token_file raised UserInputError without "
            f"code='INPUT_VALIDATION_FAILED'; got "
            f"{getattr(exc_info.value, 'code', None)!r}."
        )

    def test_empty_file_raises_user_input_error(self, tmp_path):
        """Empty file ⇒ UserInputError naming what was expected."""
        import os

        from backpropagate.cli import _read_hub_token_file
        from backpropagate.exceptions import UserInputError

        token_path = tmp_path / "empty_token"
        token_path.write_text("", encoding="utf-8")
        if os.name == "posix":
            os.chmod(token_path, 0o600)

        with pytest.raises(UserInputError) as exc_info:
            _read_hub_token_file(str(token_path), flag_name="--hub-token-file")

        message = str(exc_info.value)
        assert "empty" in message.lower(), (
            f"Empty-file error must use the word 'empty' so the "
            f"operator's regex can match. Got message: {message!r}."
        )

    def test_whitespace_only_file_treated_as_empty(self, tmp_path):
        """A file with ONLY whitespace strips to empty ⇒ same error."""
        import os

        from backpropagate.cli import _read_hub_token_file
        from backpropagate.exceptions import UserInputError

        token_path = tmp_path / "whitespace_token"
        token_path.write_text("   \n\t\n", encoding="utf-8")
        if os.name == "posix":
            os.chmod(token_path, 0o600)

        with pytest.raises(UserInputError):
            _read_hub_token_file(str(token_path), flag_name="--hub-token-file")

    def test_trailing_newline_stripped(self, tmp_path):
        """``\\n``-terminated file ⇒ token returned without the newline."""
        import os

        from backpropagate.cli import _read_hub_token_file

        token_path = tmp_path / "newline_token"
        token_path.write_text("hf_xyz789\n", encoding="utf-8")
        if os.name == "posix":
            os.chmod(token_path, 0o600)

        result = _read_hub_token_file(str(token_path), flag_name="--hub-token-file")
        assert result == "hf_xyz789", (
            f"Trailing newline must be stripped from the returned "
            f"token (the HF API would 401 on a token with embedded "
            f"newline). Got: {result!r}."
        )
        assert "\n" not in result, (
            f"Returned token must not contain newlines; got: {result!r}"
        )

    def test_world_readable_mode_emits_warning_but_returns_token(self, tmp_path, capsys):
        """POSIX only: mode 0644 ⇒ warning emits + token returned.

        The mode check is advisory (operators may have intentionally
        widened the mode). The helper warns via _print_warning but does
        NOT refuse to read.
        """
        import os

        if os.name != "posix":
            pytest.skip("File-mode check is POSIX-only")

        from backpropagate.cli import _read_hub_token_file

        token_path = tmp_path / "world_readable_token"
        token_path.write_text("hf_widemode\n", encoding="utf-8")
        os.chmod(token_path, 0o644)

        result = _read_hub_token_file(str(token_path), flag_name="--hub-token-file")

        # The token is still returned — the mode check is advisory.
        assert result == "hf_widemode"

        # Warning fires on stdout / stderr (depending on _print_warning's
        # destination). We assert the warning text references chmod or
        # the mode so the operator sees the consequence.
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "chmod" in combined.lower() or "mode" in combined.lower(), (
            "Wave 2 BRIDGE-A-004 contract: a world-readable token "
            "file must surface a warning that names 'chmod' or 'mode' "
            "so the operator sees the security consequence. Captured "
            f"output: stdout={captured.out!r}, stderr={captured.err!r}."
        )

    def test_user_input_error_message_carries_flag_name(self, tmp_path):
        """The ``flag_name`` kwarg threads into the error message so an
        operator sees which flag's path was bad (``--hub-token-file``
        vs a future spelling like ``--token-file``).
        """
        from backpropagate.cli import _read_hub_token_file
        from backpropagate.exceptions import UserInputError

        missing_path = tmp_path / "absent"

        with pytest.raises(UserInputError) as exc_info:
            _read_hub_token_file(str(missing_path), flag_name="--token-file")

        message = str(exc_info.value)
        assert "--token-file" in message, (
            f"flag_name kwarg must surface in the error message so "
            f"the operator sees which flag rejected the path; got "
            f"{message!r}."
        )


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


class TestDeprecationMarkerLifecycle:
    """CLIUI-B-007: the deprecation-removal version must stay ahead of shipped.

    ``backpropagate.__init__._REMOVED_IN_VERSION`` is the version named in the
    DeprecationWarning + ImportError raised when a caller touches a removed
    Gradio-era attribute (``launch`` / ``create_backpropagate_theme`` / …). It
    promises "this still works as ImportError until <version>, then becomes a
    hard AttributeError." That promise silently goes stale if the marker names
    a version that has already shipped — which is EXACTLY what happened at
    v1.4.0 (the marker still said ``v1.4`` per the CLI-A-005 note before it was
    bumped to ``v1.5``). This tripwire fails the moment ``_REMOVED_IN_VERSION``
    is <= ``__version__`` so the bump can't be forgotten across a release.
    """

    def test_removed_in_version_is_strictly_ahead_of_current(self):
        from packaging.version import Version

        import backpropagate
        from backpropagate import __version__

        removed_in = backpropagate._REMOVED_IN_VERSION
        # Markers are written with a leading "v" (e.g. "v1.5"); __version__ is
        # bare PEP 440 (e.g. "1.4.0"). Normalise before comparing.
        removed_v = Version(removed_in.lstrip("vV"))
        current_v = Version(__version__)

        assert removed_v > current_v, (
            f"_REMOVED_IN_VERSION={removed_in!r} is not strictly ahead of the "
            f"shipped __version__={__version__!r}. The deprecation shim "
            f"promises callers it stays ImportError until {removed_in}, then "
            f"becomes a hard AttributeError — but that version has already "
            f"shipped (or IS the current release), so the promise is stale. "
            f"Bump _REMOVED_IN_VERSION in backpropagate/__init__.py to the "
            f"next planned removal release, and flip __getattr__ to raise "
            f"AttributeError at the cut."
        )

    def test_removed_in_version_is_parseable_and_prefixed(self):
        """The marker stays in the documented ``vMAJOR.MINOR`` shape.

        Guards the comparison above: a malformed marker (e.g. a bare int or a
        typo) would make the Version() parse throw or silently mis-order. Pin
        the leading-``v`` convention the __init__ comments document.
        """
        from packaging.version import Version

        import backpropagate

        removed_in = backpropagate._REMOVED_IN_VERSION
        assert isinstance(removed_in, str)
        assert removed_in[:1] in ("v", "V"), (
            f"_REMOVED_IN_VERSION={removed_in!r} should keep the documented "
            f"leading 'v' (e.g. 'v1.5') for consistency with the warning text."
        )
        # Must parse cleanly as a PEP 440 version once the prefix is stripped.
        Version(removed_in.lstrip("vV"))


# ---------------------------------------------------------------------------
# Help-surface rendering guard (Phase 8 re-audit remediation)
# ---------------------------------------------------------------------------
#
# Two SHIPPED --help crashes slipped past every prior test because nothing
# actually *rendered* the help for any subcommand:
#
#   1. `backprop train --help` crashed on EVERY platform: the --fp8 flag's help
#      embedded a literal lone "%" ("~60% less base memory"). argparse runs
#      `help % action.__dict__` on every action while formatting → TypeError:
#      "must be real number, not dict".
#   2. `backprop multi-run --help` crashed on a default Windows console
#      (cp1252) because three v1.5 multi-run flag help strings embedded
#      non-cp1252 glyphs (→, ⇒). Rendering them to a cp1252 stdout raised
#      UnicodeEncodeError. (`backprop data --help` had the same shape in its
#      epilog: ">10% near-duplicates".)
#
# The class below renders format_help() for the top-level parser AND every
# subparser (discovered by walking the parser tree, so it auto-covers future
# subcommands + the nested ollama/data verbs), asserting it neither raises nor
# contains a byte that cp1252 cannot encode. One of these two assertions would
# have caught each crash; together they guard the whole help surface going
# forward. This is deliberately introspective rather than a hand-maintained
# command list precisely because the hand-maintained surface is what drifted.


def _walk_parsers(parser, prefix=""):
    """Yield (command_path, parser) for `parser` and every nested subparser.

    Recurses through argparse `_SubParsersAction` so nested groups (e.g.
    `ollama register`, `data report`) are covered, not just top-level verbs.
    """
    import argparse as _argparse

    label = prefix or "<top-level>"
    yield (label, parser)
    for action in parser._actions:
        if isinstance(action, _argparse._SubParsersAction):
            # `choices` maps subcommand name -> its ArgumentParser. Sort for
            # deterministic parametrization order / stable test IDs.
            for name, subparser in sorted(action.choices.items()):
                child_prefix = f"{prefix} {name}".strip()
                yield from _walk_parsers(subparser, child_prefix)


def _collect_help_cases():
    """Build the (command_path, parser) list once for parametrization."""
    from backpropagate.cli import create_parser

    return list(_walk_parsers(create_parser()))


_HELP_CASES = _collect_help_cases()
_HELP_IDS = [path for path, _ in _HELP_CASES]


class TestHelpSurfaceRenders:
    """Every parser's --help must render and be Windows-console (cp1252) safe."""

    def test_help_surface_is_nontrivial(self):
        """Sanity: the walk found the top-level parser + a healthy fan-out.

        If a refactor accidentally flattens the parser tree (or `create_parser`
        stops wiring subparsers), the parametrized guards below would silently
        shrink to ~1 case and stop protecting anything. Pin a floor so that
        regression is loud. The shipped surface has ~25 parsers (top-level +
        ~20 verbs + nested ollama/data verbs); 15 is a comfortable floor.
        """
        assert len(_HELP_CASES) >= 15, (
            f"Help-surface walk only found {len(_HELP_CASES)} parsers "
            f"({_HELP_IDS}); expected >=15. The subparser tree may have been "
            f"flattened or create_parser() stopped wiring subcommands — these "
            f"guards are only meaningful if they cover the real surface."
        )
        # The two commands whose --help shipped broken must be present.
        assert "train" in _HELP_IDS
        assert "multi-run" in _HELP_IDS

    @pytest.mark.parametrize("command_path,parser", _HELP_CASES, ids=_HELP_IDS)
    def test_format_help_does_not_raise(self, command_path, parser):
        """`format_help()` must not raise for any parser.

        Directly catches the argparse %-expansion crash: a lone "%" in any
        help/description/epilog makes argparse do `text % params` and blow up
        with TypeError. This is exactly what `train --help` did via --fp8.
        """
        try:
            rendered = parser.format_help()
        except Exception as exc:  # noqa: BLE001 - we want ANY failure surfaced
            pytest.fail(
                f"`{command_path} --help` (format_help) raised "
                f"{type(exc).__name__}: {exc}. A lone '%' in a help/"
                f"description/epilog string triggers argparse's "
                f"`text % params` expansion — escape it as '%%'."
            )
        assert isinstance(rendered, str) and rendered, (
            f"`{command_path} --help` rendered empty/None help."
        )

    @pytest.mark.parametrize("command_path,parser", _HELP_CASES, ids=_HELP_IDS)
    def test_help_is_cp1252_encodable(self, command_path, parser):
        """Rendered help must encode under cp1252 (default Windows console).

        Directly catches the Windows-console crash class: a glyph that cp1252
        cannot encode (→, ⇒, ✓, ✗, ≥, ≤, ×, …) makes Python's stdout writer
        raise UnicodeEncodeError when argparse prints --help on a default
        Windows terminal. This is what `multi-run --help` did via the v1.5
        merge-gate flags. Note: em-dash (—, U+2014) IS cp1252-safe and is
        intentionally allowed.
        """
        rendered = parser.format_help()
        try:
            rendered.encode("cp1252")
        except UnicodeEncodeError as exc:
            # Surface the exact offending character for a fast fix.
            bad = rendered[exc.start:exc.end]
            pytest.fail(
                f"`{command_path} --help` contains {bad!r} "
                f"(U+{ord(bad[0]):04X}) which cp1252 cannot encode — it would "
                f"crash `--help` on a default Windows console with "
                f"UnicodeEncodeError. Use an ASCII equivalent (e.g. '->' for "
                f"'→', '=>' for '⇒')."
            )

    @pytest.mark.parametrize(
        "subcommand",
        ["train", "multi-run", "data"],
        ids=["train", "multi-run", "data"],
    )
    def test_parse_args_help_exits_cleanly(self, cli_parser, subcommand):
        """End-to-end: `parse_args(['<sub>', '--help'])` exits 0, not crashes.

        Mirrors the exact code path a user hits (argparse prints help then
        SystemExit(0)). The three named regressions (`train`/`multi-run`/`data`)
        are pinned explicitly; the parametrized guards above cover the rest of
        the surface via format_help(). A %-expansion or other rendering error
        would surface here as a non-SystemExit exception or a nonzero code.
        """
        with pytest.raises(SystemExit) as exc_info:
            cli_parser.parse_args([subcommand, "--help"])
        # argparse exits 0 on a successful --help render.
        assert exc_info.value.code == 0, (
            f"`backprop {subcommand} --help` exited with code "
            f"{exc_info.value.code!r}; expected 0 (clean help render)."
        )
