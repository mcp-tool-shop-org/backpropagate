"""Tests for v1.4 Wave 6b CLI feature additions (BRIDGE-F-002 / F-007 / F-010 /
F-011 / F-014 / F-015 + ollama nested-subparser triad + cross-domain --mode flag).

Surfaces under test:

  Item 1 (Wave 5 Decision 4):
    `backprop ollama register|list|rm` nested subparser + the new
    :func:`backpropagate.export.remove_ollama_model` helper.

  Item 2 (BRIDGE-F-002):
    Root `--log-level` / `--log-format` / `--log-file` flags overwrite the
    corresponding BACKPROPAGATE_LOG_* env vars before configure_logging
    fires. CLI flag wins over env var; absence of flag preserves env value.

  Item 3 (BRIDGE-F-007):
    `backprop validate --json` and `backprop replay --json` emit
    schema_version-carrying payloads.

  Item 4 (BRIDGE-F-010):
    `backprop info --json` includes a `logging` block describing the
    active level / format / file.

  Item 5 (BRIDGE-F-011):
    Root `backprop --help` epilog enumerates all 18 subcommands grouped
    by workflow.

  Item 6 (BRIDGE-F-014):
    `backprop push --hub-revision <branch>` and `--hub-commit-message <msg>`
    thread through to ``push_to_hub(revision=..., commit_message=...)``.

  Item 7 (BRIDGE-F-015):
    `backprop info --subcommand-tiers` prints the SUBCOMMAND_TIERS
    registry (closes the [[no-banner-documenting-no-op]] tripwire).

  Cross-domain:
    `--mode=lora|full` on `train` / `multi-run` / `estimate-vram` /
    `replay --override mode=full`.

The tests use the existing ``cli_parser`` fixture (tests/conftest.py:479).
Subprocess-invoking handlers are exercised via direct function call with
patched dependencies so the suite stays hermetic.
"""

from __future__ import annotations

import json
import os
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Item 1: ollama nested subparser + remove_ollama_model helper
# =============================================================================


class TestOllamaSubparser:
    """`backprop ollama {register,list,rm}` nested-subparser parse contract."""

    def test_ollama_register_parses(self, cli_parser):
        args = cli_parser.parse_args(["ollama", "register", "./model.gguf"])
        assert args.command == "ollama"
        assert args.ollama_command == "register"
        assert args.path == "./model.gguf"
        assert args.name is None
        assert args.modelfile is None

    def test_ollama_register_with_name(self, cli_parser):
        args = cli_parser.parse_args([
            "ollama", "register", "./model.gguf", "--name", "my-finetune",
        ])
        assert args.ollama_command == "register"
        assert args.name == "my-finetune"

    def test_ollama_list_parses(self, cli_parser):
        args = cli_parser.parse_args(["ollama", "list"])
        assert args.command == "ollama"
        assert args.ollama_command == "list"

    def test_ollama_rm_parses(self, cli_parser):
        args = cli_parser.parse_args(["ollama", "rm", "my-finetune"])
        assert args.ollama_command == "rm"
        assert args.name == "my-finetune"

    def test_ollama_rm_requires_name(self, cli_parser):
        # argparse exits 2 on a missing positional
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["ollama", "rm"])


class TestRemoveOllamaModel:
    """Backend helper :func:`backpropagate.export.remove_ollama_model`."""

    def test_returns_false_when_ollama_missing(self):
        from backpropagate.export import remove_ollama_model
        with patch("backpropagate.export.shutil.which", return_value=None):
            assert remove_ollama_model("my-finetune") is False

    def test_invokes_ollama_rm(self):
        from backpropagate.export import remove_ollama_model

        completed = MagicMock(returncode=0, stdout="", stderr="")
        with patch("backpropagate.export.shutil.which", return_value="/usr/bin/ollama"), \
             patch("backpropagate.export.subprocess.run", return_value=completed) as run_mock:
            assert remove_ollama_model("my-finetune") is True
        # First positional is the argv list — verify the shape.
        run_mock.assert_called_once()
        argv = run_mock.call_args[0][0]
        assert argv == ["ollama", "rm", "my-finetune"]

    def test_rejects_leading_dash_name(self):
        """Same allowlist as the register side — leading dash is option injection."""
        from backpropagate.exceptions import ExportError
        from backpropagate.export import remove_ollama_model

        with pytest.raises(ExportError) as exc_info:
            remove_ollama_model("-rm-rf")
        # _validate_model_name raises ExportError with INPUT_VALIDATION_FAILED
        assert getattr(exc_info.value, "code", None) == "INPUT_VALIDATION_FAILED"

    def test_not_found_message_routes_to_targeted_hint(self):
        """Distinguish 'model not found' from generic daemon failures."""
        import subprocess

        from backpropagate.exceptions import OllamaRegistrationError
        from backpropagate.export import remove_ollama_model

        err = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ollama", "rm", "ghost"],
            output="",
            stderr="Error: model 'ghost' not found\n",
        )
        with patch("backpropagate.export.shutil.which", return_value="/usr/bin/ollama"), \
             patch("backpropagate.export.subprocess.run", side_effect=err):
            with pytest.raises(OllamaRegistrationError) as exc_info:
                remove_ollama_model("ghost")
        assert "not found" in (exc_info.value.suggestion or "").lower()
        assert "backprop ollama list" in (exc_info.value.suggestion or "")

    def test_remove_ollama_model_reexported(self):
        """The helper is reachable from the top-level package."""
        import backpropagate

        assert hasattr(backpropagate, "remove_ollama_model")
        assert callable(backpropagate.remove_ollama_model)


class TestCmdOllamaHandlers:
    """Smoke tests for the three cmd_ollama_* handlers (mocked subprocess)."""

    def _ns(self, **overrides):
        defaults = {
            "verbose": False,
            "command": "ollama",
        }
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_register_path_missing_returns_user_error(self, tmp_path):
        from backpropagate.cli import EXIT_USER_ERROR, cmd_ollama_register

        args = self._ns(
            ollama_command="register",
            path=str(tmp_path / "missing.gguf"),
            name=None,
            modelfile=None,
        )
        assert cmd_ollama_register(args) == EXIT_USER_ERROR

    def test_register_calls_helper_with_resolved_name(self, tmp_path):
        from backpropagate.cli import EXIT_OK, cmd_ollama_register

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake-gguf")

        with patch("backpropagate.export.register_with_ollama", return_value=True) as helper:
            args = self._ns(
                ollama_command="register",
                path=str(gguf),
                name="my-finetune",
                modelfile=None,
            )
            assert cmd_ollama_register(args) == EXIT_OK
        helper.assert_called_once()
        # Second positional arg is model_name
        assert helper.call_args[0][1] == "my-finetune"

    def test_register_derives_name_from_filename_stem(self, tmp_path):
        from backpropagate.cli import EXIT_OK, cmd_ollama_register

        gguf = tmp_path / "qwen-finetune.gguf"
        gguf.write_bytes(b"fake-gguf")

        with patch("backpropagate.export.register_with_ollama", return_value=True) as helper:
            args = self._ns(
                ollama_command="register",
                path=str(gguf),
                name=None,  # default — derive from stem
                modelfile=None,
            )
            assert cmd_ollama_register(args) == EXIT_OK
        assert helper.call_args[0][1] == "qwen-finetune"

    def test_list_prints_models(self, capsys):
        from backpropagate.cli import EXIT_OK, cmd_ollama_list

        with patch("backpropagate.export.list_ollama_models",
                   return_value=["model-a", "model-b"]):
            args = self._ns(ollama_command="list", verbose=False)
            assert cmd_ollama_list(args) == EXIT_OK
        out = capsys.readouterr().out
        assert "model-a" in out
        assert "model-b" in out

    def test_list_no_daemon_returns_ex_unavailable(self):
        from backpropagate.cli import EXIT_UNAVAILABLE, cmd_ollama_list

        with patch("backpropagate.export.list_ollama_models", return_value=[]), \
             patch("shutil.which", return_value=None):
            args = self._ns(ollama_command="list")
            assert cmd_ollama_list(args) == EXIT_UNAVAILABLE

    def test_rm_calls_helper_and_returns_ok(self):
        from backpropagate.cli import EXIT_OK, cmd_ollama_rm

        with patch("backpropagate.export.remove_ollama_model", return_value=True) as helper:
            args = self._ns(ollama_command="rm", name="my-finetune", verbose=False)
            assert cmd_ollama_rm(args) == EXIT_OK
        helper.assert_called_once_with("my-finetune")


# =============================================================================
# Item 2: --log-level / --log-format / --log-file root flags
# =============================================================================


class TestRootLoggingFlags:
    """BRIDGE-F-002 (v1.4): root flags overwrite BACKPROPAGATE_LOG_* env vars."""

    def test_log_level_flag_parses(self, cli_parser):
        args = cli_parser.parse_args(["--log-level", "DEBUG", "info"])
        assert args.log_level == "DEBUG"

    def test_log_format_flag_parses(self, cli_parser):
        args = cli_parser.parse_args(["--log-format", "json", "info"])
        assert args.log_format == "json"

    def test_log_file_flag_parses(self, cli_parser):
        args = cli_parser.parse_args(["--log-file", "/tmp/backprop.log", "info"])
        assert args.log_file == "/tmp/backprop.log"

    def test_log_level_choices(self, cli_parser):
        # Bogus value triggers SystemExit
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["--log-level", "TRACE", "info"])

    def test_log_format_choices(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["--log-format", "yaml", "info"])

    def test_main_overwrites_env_var_when_flag_set(self, monkeypatch, tmp_path):
        """``main(['--log-level', 'DEBUG', 'info', ...])`` sets BACKPROPAGATE_LOG_LEVEL."""
        from backpropagate.cli import main

        monkeypatch.setenv("BACKPROPAGATE_LOG_LEVEL", "WARNING")
        # Pre-condition: env var is WARNING.
        assert os.environ["BACKPROPAGATE_LOG_LEVEL"] == "WARNING"

        # Patch all the heavy lifting so main() doesn't actually try to
        # load a model. We only care that env var was overwritten by the
        # time the subcommand handler runs.
        observed = {}

        def fake_info(args):  # noqa: ANN001
            observed["log_level"] = os.environ.get("BACKPROPAGATE_LOG_LEVEL")
            return 0

        # Import BEFORE patching so `real_create` is the real function,
        # not the MagicMock that would replace it inside the patch context.
        from backpropagate.cli import create_parser as real_create

        with patch("backpropagate.cli.cmd_info", side_effect=fake_info):
            # Re-create parser so set_defaults binds our patched cmd_info
            with patch("backpropagate.cli.create_parser") as parser_mock:
                p = real_create()
                # Find info subparser's set_defaults and rebind
                for action in p._actions:
                    if hasattr(action, "choices") and "info" in (action.choices or {}):
                        action.choices["info"].set_defaults(func=fake_info)
                parser_mock.return_value = p
                main(["--log-level", "DEBUG", "info"])
        assert observed["log_level"] == "DEBUG"

    def test_log_format_flag_maps_to_json_true(self, monkeypatch):
        """`--log-format=json` sets BACKPROPAGATE_LOG_JSON=true."""
        from backpropagate.cli import main

        observed = {}

        def fake_info(args):  # noqa: ANN001
            observed["json"] = os.environ.get("BACKPROPAGATE_LOG_JSON")
            return 0

        from backpropagate.cli import create_parser as real_create
        with patch("backpropagate.cli.create_parser") as parser_mock:
            p = real_create()
            for action in p._actions:
                if hasattr(action, "choices") and "info" in (action.choices or {}):
                    action.choices["info"].set_defaults(func=fake_info)
            parser_mock.return_value = p
            main(["--log-format", "json", "info"])
        assert observed["json"] == "true"

    def test_log_format_flag_console_maps_to_false(self, monkeypatch):
        """`--log-format=console` sets BACKPROPAGATE_LOG_JSON=false."""
        from backpropagate.cli import main

        observed = {}

        def fake_info(args):  # noqa: ANN001
            observed["json"] = os.environ.get("BACKPROPAGATE_LOG_JSON")
            return 0

        from backpropagate.cli import create_parser as real_create
        with patch("backpropagate.cli.create_parser") as parser_mock:
            p = real_create()
            for action in p._actions:
                if hasattr(action, "choices") and "info" in (action.choices or {}):
                    action.choices["info"].set_defaults(func=fake_info)
            parser_mock.return_value = p
            main(["--log-format", "console", "info"])
        assert observed["json"] == "false"


# =============================================================================
# Item 3: validate / replay --json
# =============================================================================


class TestValidateJsonFlag:
    """BRIDGE-F-007: ``backprop validate --json`` parse + handler."""

    def test_validate_json_flag_parses(self, cli_parser):
        args = cli_parser.parse_args(["validate", "data.jsonl", "--json"])
        assert args.json is True

    def test_validate_json_missing_file_emits_payload(self, tmp_path, capsys):
        from backpropagate.cli import EXIT_USER_ERROR, cmd_validate

        args = Namespace(
            dataset=str(tmp_path / "missing.jsonl"),
            format="auto",
            max_errors=100,
            max_samples=None,
            json=True,
            verbose=False,
        )
        rc = cmd_validate(args)
        assert rc == EXIT_USER_ERROR
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["schema_version"] == "1"
        assert payload["is_valid"] is False
        assert payload["error"] == "dataset_not_found"

    def test_validate_json_clean_dataset(self, tmp_path, capsys):
        from backpropagate.cli import EXIT_OK, cmd_validate

        # ShareGPT-format dataset (5 rows)
        path = tmp_path / "data.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({
                    "conversations": [
                        {"from": "human", "value": f"q{i}"},
                        {"from": "gpt", "value": f"a{i}"},
                    ],
                })
                for i in range(5)
            ),
            encoding="utf-8",
        )

        args = Namespace(
            dataset=str(path),
            format="auto",
            max_errors=100,
            max_samples=None,
            json=True,
            verbose=False,
        )
        rc = cmd_validate(args)
        assert rc == EXIT_OK
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["schema_version"] == "1"
        assert payload["is_valid"] is True
        assert payload["total_rows"] == 5


class TestReplayJsonFlag:
    """BRIDGE-F-007: ``backprop replay --json`` parse contract."""

    def test_replay_json_flag_parses(self, cli_parser):
        args = cli_parser.parse_args(["replay", "abc123", "--json"])
        assert args.json is True


# =============================================================================
# Item 4: info --json logging block
# =============================================================================


class TestInfoJsonLogging:
    """BRIDGE-F-010: ``backprop info --json`` carries a ``logging`` block."""

    def test_logging_block_present(self, monkeypatch, capsys):
        """The info --json payload includes level / format / file / json_var_set."""
        from backpropagate.cli import cmd_info

        monkeypatch.setenv("BACKPROPAGATE_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("BACKPROPAGATE_LOG_JSON", "true")
        monkeypatch.delenv("BACKPROPAGATE_LOG_FILE", raising=False)

        args = Namespace(
            error_codes=False,
            env_vars=False,
            json=True,
            subcommand_tiers=False,
        )
        rc = cmd_info(args)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "logging" in payload
        log = payload["logging"]
        assert log["level"] == "DEBUG"
        assert log["format"] == "json"
        assert log["file"] is None
        assert log["json_var_set"] is True

    def test_logging_file_picked_up(self, monkeypatch, capsys, tmp_path):
        from backpropagate.cli import cmd_info

        log_path = tmp_path / "out.log"
        monkeypatch.setenv("BACKPROPAGATE_LOG_FILE", str(log_path))

        args = Namespace(
            error_codes=False,
            env_vars=False,
            json=True,
            subcommand_tiers=False,
        )
        cmd_info(args)
        payload = json.loads(capsys.readouterr().out)
        assert payload["logging"]["file"] == str(log_path)


# =============================================================================
# Item 5: root --help epilog enumerates 18 subcommands
# =============================================================================


class TestRootHelpEpilog:
    """BRIDGE-F-011: root --help epilog enumerates ALL subcommands grouped by workflow."""

    def test_epilog_lists_every_subcommand(self, cli_parser):
        epilog = cli_parser.epilog or ""
        # All subcommands the v1.4 CLI ships
        for name in [
            "train", "multi-run", "resume", "replay",
            "info", "config", "list-runs", "runs", "show-run", "diff-runs",
            "export-runs", "validate", "estimate-vram",
            # v1.5 T1.1 additions (data-quality moat + eval harness)
            "data report", "eval",
            "export", "push",
            "ollama register", "ollama list", "ollama rm",
            "ui",
        ]:
            assert name in epilog, f"Subcommand '{name}' missing from root --help epilog"

    def test_epilog_has_workflow_groups(self, cli_parser):
        epilog = cli_parser.epilog or ""
        for group in ["Training:", "Inspection:", "Export:", "Ollama", "UI:"]:
            assert group in epilog, f"Workflow group '{group}' missing from epilog"


# =============================================================================
# Item 6: push --hub-revision / --hub-commit-message
# =============================================================================


class TestPushHubFlags:
    """BRIDGE-F-014: --hub-revision and --hub-commit-message thread through to push_to_hub."""

    def test_hub_revision_flag_parses(self, cli_parser):
        args = cli_parser.parse_args([
            "push", ".", "--repo", "alice/foo", "--hub-revision", "dev",
        ])
        assert args.hub_revision == "dev"

    def test_hub_commit_message_flag_parses(self, cli_parser):
        args = cli_parser.parse_args([
            "push", ".", "--repo", "alice/foo",
            "--hub-commit-message", "Initial upload",
        ])
        assert args.hub_commit_message == "Initial upload"

    def test_push_handler_forwards_kwargs(self, tmp_path):
        """cmd_push passes revision / commit_message to push_to_hub."""
        from backpropagate.cli import cmd_push

        local = tmp_path / "model"
        local.mkdir()
        (local / "adapter_config.json").write_text("{}")

        args = Namespace(
            local_path=str(local),
            repo="alice/foo",
            token=None,
            token_file=None,
            private=False,
            include_base=False,
            hub_revision="dev",
            hub_commit_message="experiment",
            verbose=False,
        )

        with patch("backpropagate.export.push_to_hub", return_value="https://hf.co/alice/foo") as p:
            cmd_push(args)
        p.assert_called_once()
        kwargs = p.call_args.kwargs
        assert kwargs["revision"] == "dev"
        assert kwargs["commit_message"] == "experiment"

    def test_push_handler_omits_kwargs_when_unset(self, tmp_path):
        """Default invocation passes None for both knobs."""
        from backpropagate.cli import cmd_push

        local = tmp_path / "model"
        local.mkdir()
        (local / "adapter_config.json").write_text("{}")

        args = Namespace(
            local_path=str(local),
            repo="alice/foo",
            token=None,
            token_file=None,
            private=False,
            include_base=False,
            hub_revision=None,
            hub_commit_message=None,
            verbose=False,
        )

        with patch("backpropagate.export.push_to_hub", return_value="https://hf.co/alice/foo") as p:
            cmd_push(args)
        kwargs = p.call_args.kwargs
        assert kwargs["revision"] is None
        assert kwargs["commit_message"] is None


# =============================================================================
# Item 7: info --subcommand-tiers
# =============================================================================


class TestSubcommandTiers:
    """BRIDGE-F-015: ``backprop info --subcommand-tiers`` exposes SUBCOMMAND_TIERS."""

    def test_subcommand_tiers_flag_parses(self, cli_parser):
        args = cli_parser.parse_args(["info", "--subcommand-tiers"])
        assert args.subcommand_tiers is True

    def test_subcommand_tiers_human_output(self, capsys):
        from backpropagate.cli import EXIT_OK, cmd_info

        args = Namespace(
            error_codes=False,
            env_vars=False,
            json=False,
            subcommand_tiers=True,
        )
        rc = cmd_info(args)
        assert rc == EXIT_OK
        out = capsys.readouterr().out
        # Header text
        assert "SUBCOMMAND" in out
        assert "TIER" in out
        # Known stable + experimental + deprecated entries
        assert "train" in out
        assert "ollama" in out
        assert "list-runs" in out

    def test_subcommand_tiers_json_output(self, capsys):
        from backpropagate.cli import EXIT_OK, cmd_info

        args = Namespace(
            error_codes=False,
            env_vars=False,
            json=True,
            subcommand_tiers=True,
        )
        rc = cmd_info(args)
        assert rc == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["schema_version"] == "1"
        assert "subcommand_tiers" in payload
        tiers = payload["subcommand_tiers"]
        assert tiers["train"] == "stable"
        assert tiers["ollama"] == "experimental"
        assert tiers["list-runs"] == "deprecated-prefer-runs"


# =============================================================================
# Cross-domain: --mode flag wiring
# =============================================================================


class TestModeFlag:
    """Cross-domain: --mode=lora|full on train / multi-run / estimate-vram."""

    def test_train_mode_default(self, cli_parser):
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.mode == "lora"

    def test_train_mode_full(self, cli_parser):
        args = cli_parser.parse_args(["train", "-d", "data.jsonl", "--mode", "full"])
        assert args.mode == "full"

    def test_multi_run_mode_default(self, cli_parser):
        args = cli_parser.parse_args(["multi-run", "-d", "data.jsonl"])
        assert args.mode == "lora"

    def test_multi_run_mode_full(self, cli_parser):
        args = cli_parser.parse_args(["multi-run", "-d", "data.jsonl", "--mode", "full"])
        assert args.mode == "full"

    def test_mode_invalid_choice(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["train", "-d", "data.jsonl", "--mode", "qlora"])

    def test_estimate_vram_mode_flag(self, cli_parser):
        args = cli_parser.parse_args([
            "estimate-vram", "Qwen/Qwen2.5-7B-Instruct", "--mode", "full",
            "--batch-size", "1",
        ])
        assert args.mode == "full"
        assert args.batch_size == 1

    def test_estimate_vram_lora_r_flag(self, cli_parser):
        args = cli_parser.parse_args([
            "estimate-vram", "Qwen/Qwen2.5-7B-Instruct",
            "--lora-r", "16", "--batch-size", "2",
        ])
        assert args.lora_r == 16
        assert args.batch_size == 2

    def test_replay_allowed_override_includes_mode(self):
        """Cross-domain: replay --override mode=full is on the whitelist."""
        from backpropagate.cli import _REPLAY_ALLOWED_OVERRIDE_KEYS

        assert "mode" in _REPLAY_ALLOWED_OVERRIDE_KEYS
