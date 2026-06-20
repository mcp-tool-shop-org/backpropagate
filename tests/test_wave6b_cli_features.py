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


# =============================================================================
# v1.5 T1.2 (ORPO): --method / --orpo-beta flag wiring
# =============================================================================


class TestOrpoMethodFlag:
    """v1.5 T1.2 (ORPO Wave 1): ``--method`` + ``--orpo-beta`` on ``train``.

    The flags parse onto ``args`` and thread into the
    ``wave6b_candidate_kwargs`` introspection-filter dict. We pin BOTH ends:
    the parser surface, and that the keys reach the Trainer constructor when
    the installed Trainer advertises a catch-all signature (a future trainer
    wave adds the matching ``__init__`` kwargs; until then the filter drops
    them for a concrete signature — that's the forward-compatible contract).
    """

    def test_train_method_default_is_sft(self, cli_parser):
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.method == "sft"

    def test_train_method_orpo(self, cli_parser):
        args = cli_parser.parse_args(["train", "-d", "data.jsonl", "--method", "orpo"])
        assert args.method == "orpo"

    def test_train_orpo_beta_default(self, cli_parser):
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.orpo_beta == 0.1

    def test_train_orpo_beta_custom(self, cli_parser):
        args = cli_parser.parse_args([
            "train", "-d", "data.jsonl", "--orpo-beta", "0.2",
        ])
        assert args.orpo_beta == pytest.approx(0.2)

    def test_train_method_rejects_unknown_choice(self, cli_parser):
        """--method dpo (not implemented in v1.5) fails argparse choice check."""
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["train", "-d", "data.jsonl", "--method", "dpo"])

    def test_method_and_orpo_beta_thread_into_trainer_kwargs(self, tmp_path):
        """``--method orpo`` + ``--orpo-beta 0.2`` reach the Trainer constructor.

        Uses the same catch-all-MagicMock pattern as the replay tests: a
        MagicMock Trainer advertises ``(*args, **kwargs)``, so cmd_train's
        ``_trainer_sig_params`` filter sets the sentinel ``None`` and forwards
        EVERY wave6b_candidate_kwargs key unfiltered. This pins that the two
        new keys are present in the dict construction without asserting the
        real Trainer.__init__ accepts them yet (the trainer wave adds those).
        """
        from backpropagate.cli import EXIT_OK, cmd_train

        fake_trainer = MagicMock()
        # train()/save() return numeric-fielded results so cmd_train's
        # f"{result.final_loss:.4f}" / duration formatting doesn't blow up on
        # a bare MagicMock (mirrors test_replay_dispatches_single_run).
        fake_trainer.train.return_value = MagicMock(
            final_loss=0.42, duration_seconds=1.0, run_id="r-orpo"
        )
        fake_trainer.save.return_value = str(tmp_path / "out")

        args = Namespace(
            model="test-model",
            data="data.jsonl",
            steps=10,
            samples=None,
            batch_size="auto",
            lr=2e-4,
            lora_r=256,
            output=str(tmp_path),
            no_unsloth=True,
            use_dora=False,
            no_packing=False,
            init_lora_weights="default",
            lora_preset="quality",
            optim="auto",
            mode="lora",
            method="orpo",
            orpo_beta=0.2,
            resume=None,
            cli_run_id=None,
            verbose=False,
        )

        with patch(
            "backpropagate.trainer.Trainer", return_value=fake_trainer
        ) as mock_cls:
            rc = cmd_train(args)

        assert rc == EXIT_OK
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs.get("method") == "orpo", (
            f"--method orpo must thread into the Trainer kwargs; got "
            f"method={init_kwargs.get('method')!r}"
        )
        assert init_kwargs.get("orpo_beta") == pytest.approx(0.2), (
            f"--orpo-beta 0.2 must thread into the Trainer kwargs; got "
            f"orpo_beta={init_kwargs.get('orpo_beta')!r}"
        )

    def test_method_defaults_thread_into_trainer_kwargs(self, tmp_path):
        """Default invocation forwards method='sft' + orpo_beta=0.1.

        The forward-compatible default path must carry the SFT default so a
        later trainer wave's dispatch sees 'sft' (byte-identical v1.4 behavior)
        when the operator passes nothing.
        """
        from backpropagate.cli import EXIT_OK, cmd_train

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(
            final_loss=0.5, duration_seconds=1.0, run_id="r-sft"
        )
        fake_trainer.save.return_value = str(tmp_path / "out")

        args = Namespace(
            model="test-model",
            data="data.jsonl",
            steps=10,
            samples=None,
            batch_size="auto",
            lr=2e-4,
            lora_r=256,
            output=str(tmp_path),
            no_unsloth=True,
            use_dora=False,
            no_packing=False,
            init_lora_weights="default",
            lora_preset="quality",
            optim="auto",
            mode="lora",
            method="sft",
            orpo_beta=0.1,
            resume=None,
            cli_run_id=None,
            verbose=False,
        )

        with patch(
            "backpropagate.trainer.Trainer", return_value=fake_trainer
        ) as mock_cls:
            rc = cmd_train(args)

        assert rc == EXIT_OK
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs.get("method") == "sft"
        assert init_kwargs.get("orpo_beta") == pytest.approx(0.1)


# =============================================================================
# v1.5 T2.1 (Wave 6b GLUE): train --fp8 / --use-rslora
# =============================================================================


class TestTrainFp8RsLoraFlags:
    """``backprop train --fp8`` / ``--use-rslora`` parse + thread contract."""

    def test_fp8_flag_parses(self, cli_parser):
        args = cli_parser.parse_args(["train", "-d", "data.jsonl", "--fp8"])
        assert args.fp8 is True

    def test_use_rslora_flag_parses(self, cli_parser):
        args = cli_parser.parse_args(["train", "-d", "data.jsonl", "--use-rslora"])
        assert args.use_rslora is True

    def test_fp8_and_rslora_default_false(self, cli_parser):
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.fp8 is False
        assert args.use_rslora is False

    def test_fp8_and_rslora_thread_into_trainer_kwargs(self, tmp_path):
        """``--fp8`` + ``--use-rslora`` reach the Trainer constructor kwargs.

        Same catch-all-MagicMock pattern as the ORPO threading test: a
        MagicMock Trainer advertises ``(*args, **kwargs)`` so cmd_train's
        introspection filter forwards every wave6b_candidate_kwargs key
        unfiltered. Pins that ``fp8`` / ``use_rslora`` (named to match
        Trainer.__init__) land in the threaded dict.
        """
        from backpropagate.cli import EXIT_OK, cmd_train

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(
            final_loss=0.42, duration_seconds=1.0, run_id="r-fp8"
        )
        fake_trainer.save.return_value = str(tmp_path / "out")

        args = Namespace(
            model="test-model",
            data="data.jsonl",
            steps=10,
            samples=None,
            batch_size="auto",
            lr=2e-4,
            lora_r=256,
            output=str(tmp_path),
            no_unsloth=True,
            use_dora=False,
            no_packing=False,
            init_lora_weights="default",
            lora_preset="quality",
            optim="auto",
            mode="lora",
            method="sft",
            orpo_beta=0.1,
            fp8=True,
            use_rslora=True,
            resume=None,
            cli_run_id=None,
            verbose=False,
        )

        with patch(
            "backpropagate.trainer.Trainer", return_value=fake_trainer
        ) as mock_cls:
            rc = cmd_train(args)

        assert rc == EXIT_OK
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs.get("fp8") is True, (
            f"--fp8 must thread into the Trainer kwargs; got "
            f"fp8={init_kwargs.get('fp8')!r}"
        )
        assert init_kwargs.get("use_rslora") is True, (
            f"--use-rslora must thread into the Trainer kwargs; got "
            f"use_rslora={init_kwargs.get('use_rslora')!r}"
        )

    def test_fp8_rslora_defaults_thread_false(self, tmp_path):
        """Default invocation forwards fp8=False + use_rslora=False."""
        from backpropagate.cli import EXIT_OK, cmd_train

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(
            final_loss=0.5, duration_seconds=1.0, run_id="r-nofp8"
        )
        fake_trainer.save.return_value = str(tmp_path / "out")

        args = Namespace(
            model="test-model",
            data="data.jsonl",
            steps=10,
            samples=None,
            batch_size="auto",
            lr=2e-4,
            lora_r=256,
            output=str(tmp_path),
            no_unsloth=True,
            use_dora=False,
            no_packing=False,
            init_lora_weights="default",
            lora_preset="quality",
            optim="auto",
            mode="lora",
            method="sft",
            orpo_beta=0.1,
            fp8=False,
            use_rslora=False,
            resume=None,
            cli_run_id=None,
            verbose=False,
        )

        with patch(
            "backpropagate.trainer.Trainer", return_value=fake_trainer
        ) as mock_cls:
            rc = cmd_train(args)

        assert rc == EXIT_OK
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs.get("fp8") is False
        assert init_kwargs.get("use_rslora") is False


class TestTrainReasoningTraceFlag:
    """v1.5 T3.2: ``backprop train --reasoning-trace`` parse + thread contract.

    Pins BOTH ends of the wave6b introspection-filter wiring: the flag binds
    onto ``args.reasoning_trace`` at parse time, and ``cmd_train`` forwards a
    ``reasoning_trace`` kwarg (named to match ``Trainer.__init__``) so the
    introspection filter passes it through to the constructor.
    """

    def test_reasoning_trace_flag_parses(self, cli_parser):
        args = cli_parser.parse_args(
            ["train", "-d", "data.jsonl", "--reasoning-trace"]
        )
        assert args.reasoning_trace is True

    def test_reasoning_trace_default_false(self, cli_parser):
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.reasoning_trace is False

    def test_reasoning_trace_threads_into_trainer_kwargs(self, tmp_path):
        """``--reasoning-trace`` reaches the Trainer constructor kwargs.

        Same catch-all-MagicMock pattern as the fp8/rsLoRA threading test: a
        MagicMock Trainer advertises ``(*args, **kwargs)`` so cmd_train's
        introspection filter forwards every wave6b_candidate_kwargs key
        unfiltered. Pins that ``reasoning_trace`` lands in the threaded dict.
        """
        from backpropagate.cli import EXIT_OK, cmd_train

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(
            final_loss=0.33, duration_seconds=1.0, run_id="r-think"
        )
        fake_trainer.save.return_value = str(tmp_path / "out")

        args = Namespace(
            model="test-model",
            data="data.jsonl",
            steps=10,
            samples=None,
            batch_size="auto",
            lr=2e-4,
            lora_r=256,
            output=str(tmp_path),
            no_unsloth=True,
            use_dora=False,
            no_packing=False,
            init_lora_weights="default",
            lora_preset="quality",
            optim="auto",
            mode="lora",
            method="sft",
            orpo_beta=0.1,
            fp8=False,
            use_rslora=False,
            reasoning_trace=True,
            resume=None,
            cli_run_id=None,
            verbose=False,
        )

        with patch(
            "backpropagate.trainer.Trainer", return_value=fake_trainer
        ) as mock_cls:
            rc = cmd_train(args)

        assert rc == EXIT_OK
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs.get("reasoning_trace") is True, (
            f"--reasoning-trace must thread into the Trainer kwargs; got "
            f"reasoning_trace={init_kwargs.get('reasoning_trace')!r}"
        )

    def test_reasoning_trace_default_threads_false(self, tmp_path):
        """Default invocation forwards reasoning_trace=False."""
        from backpropagate.cli import EXIT_OK, cmd_train

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(
            final_loss=0.5, duration_seconds=1.0, run_id="r-nothink"
        )
        fake_trainer.save.return_value = str(tmp_path / "out")

        args = Namespace(
            model="test-model",
            data="data.jsonl",
            steps=10,
            samples=None,
            batch_size="auto",
            lr=2e-4,
            lora_r=256,
            output=str(tmp_path),
            no_unsloth=True,
            use_dora=False,
            no_packing=False,
            init_lora_weights="default",
            lora_preset="quality",
            optim="auto",
            mode="lora",
            method="sft",
            orpo_beta=0.1,
            fp8=False,
            use_rslora=False,
            reasoning_trace=False,
            resume=None,
            cli_run_id=None,
            verbose=False,
        )

        with patch(
            "backpropagate.trainer.Trainer", return_value=fake_trainer
        ) as mock_cls:
            rc = cmd_train(args)

        assert rc == EXIT_OK
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs.get("reasoning_trace") is False


# =============================================================================
# v1.5 T3.1 (Wave 6b GLUE): train --backend (MLX / Apple-Silicon selector)
# =============================================================================


class TestTrainBackendFlag:
    """v1.5 T3.1: ``backprop train --backend`` parse + thread contract.

    Pins BOTH ends of the wave6b introspection-filter wiring: the flag binds
    onto ``args.backend`` at parse time (with the {auto, cuda, mlx} set enforced
    by argparse ``choices``), and ``cmd_train`` forwards a ``backend`` kwarg
    (named to match ``Trainer.__init__``) so the introspection filter passes it
    through to the constructor. The cross-field "mlx forced on non-Apple" gate
    lives in the Trainer constructor, NOT here — these tests only assert the CLI
    surface + threading, so no Apple-Silicon host is required.
    """

    def test_backend_default_is_auto(self, cli_parser):
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.backend == "auto"

    def test_backend_mlx_parses(self, cli_parser):
        args = cli_parser.parse_args(
            ["train", "-d", "data.jsonl", "--backend", "mlx"]
        )
        assert args.backend == "mlx"

    def test_backend_cuda_parses(self, cli_parser):
        args = cli_parser.parse_args(
            ["train", "-d", "data.jsonl", "--backend", "cuda"]
        )
        assert args.backend == "cuda"

    def test_backend_rejects_unknown_choice(self, cli_parser):
        """An out-of-set value (e.g. ``zzz``) is rejected by argparse choices."""
        with pytest.raises(SystemExit):
            cli_parser.parse_args(
                ["train", "-d", "data.jsonl", "--backend", "zzz"]
            )

    def test_backend_threads_into_trainer_kwargs(self, tmp_path):
        """``--backend mlx`` reaches the Trainer constructor kwargs.

        Same catch-all-MagicMock pattern as the fp8/rsLoRA/reasoning-trace
        threading tests: a MagicMock Trainer advertises ``(*args, **kwargs)`` so
        cmd_train's introspection filter forwards every wave6b_candidate_kwargs
        key unfiltered. Pins that ``backend`` lands in the threaded dict.
        """
        from backpropagate.cli import EXIT_OK, cmd_train

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(
            final_loss=0.21, duration_seconds=1.0, run_id="r-mlx"
        )
        fake_trainer.save.return_value = str(tmp_path / "out")

        args = Namespace(
            model="test-model",
            data="data.jsonl",
            steps=10,
            samples=None,
            batch_size="auto",
            lr=2e-4,
            lora_r=256,
            output=str(tmp_path),
            no_unsloth=True,
            use_dora=False,
            no_packing=False,
            init_lora_weights="default",
            lora_preset="quality",
            optim="auto",
            mode="lora",
            method="sft",
            orpo_beta=0.1,
            fp8=False,
            use_rslora=False,
            reasoning_trace=False,
            backend="mlx",
            resume=None,
            cli_run_id=None,
            verbose=False,
        )

        with patch(
            "backpropagate.trainer.Trainer", return_value=fake_trainer
        ) as mock_cls:
            rc = cmd_train(args)

        assert rc == EXIT_OK
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs.get("backend") == "mlx", (
            f"--backend must thread into the Trainer kwargs; got "
            f"backend={init_kwargs.get('backend')!r}"
        )

    def test_backend_default_threads_auto(self, tmp_path):
        """Default invocation forwards backend='auto'."""
        from backpropagate.cli import EXIT_OK, cmd_train

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(
            final_loss=0.5, duration_seconds=1.0, run_id="r-auto"
        )
        fake_trainer.save.return_value = str(tmp_path / "out")

        args = Namespace(
            model="test-model",
            data="data.jsonl",
            steps=10,
            samples=None,
            batch_size="auto",
            lr=2e-4,
            lora_r=256,
            output=str(tmp_path),
            no_unsloth=True,
            use_dora=False,
            no_packing=False,
            init_lora_weights="default",
            lora_preset="quality",
            optim="auto",
            mode="lora",
            method="sft",
            orpo_beta=0.1,
            fp8=False,
            use_rslora=False,
            reasoning_trace=False,
            backend="auto",
            resume=None,
            cli_run_id=None,
            verbose=False,
        )

        with patch(
            "backpropagate.trainer.Trainer", return_value=fake_trainer
        ) as mock_cls:
            rc = cmd_train(args)

        assert rc == EXIT_OK
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs.get("backend") == "auto"


# =============================================================================
# v1.5 T2.2 (Wave 6b GLUE): multi-run merge framework flags
# =============================================================================


class TestMultiRunMergeFlags:
    """``backprop multi-run`` merge-strategy / drift-gate / eval-gate parse."""

    def test_merge_strategy_default_qiao_mahdavi(self, cli_parser):
        args = cli_parser.parse_args(["multi-run", "-d", "data.jsonl"])
        assert args.merge_strategy == "qiao_mahdavi"

    @pytest.mark.parametrize("strategy", ["qiao_mahdavi", "linear", "ties", "dare"])
    def test_merge_strategy_accepts_choices(self, cli_parser, strategy):
        args = cli_parser.parse_args(
            ["multi-run", "-d", "data.jsonl", "--merge-strategy", strategy]
        )
        assert args.merge_strategy == strategy

    def test_merge_strategy_rejects_unknown(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(
                ["multi-run", "-d", "data.jsonl", "--merge-strategy", "slerp"]
            )

    def test_ties_trim_rejects_out_of_unit_range(self, cli_parser):
        # _unit_float rejects > 1 (argparse exits 2).
        with pytest.raises(SystemExit):
            cli_parser.parse_args(
                ["multi-run", "-d", "data.jsonl", "--ties-trim", "5"]
            )

    def test_merge_and_gate_flags_parse(self, cli_parser):
        args = cli_parser.parse_args([
            "multi-run", "-d", "data.jsonl",
            "--merge-strategy", "ties",
            "--ties-trim", "0.3",
            "--dare-drop-rate", "0.4",
            "--dare-seed", "7",
            "--drift-gate",
            "--drift-threshold", "0.15",
            "--eval-gate",
            "--eval-max-regression", "0.05",
            "--eval-heldout", "held.jsonl",
        ])
        assert args.merge_strategy == "ties"
        assert args.ties_trim == pytest.approx(0.3)
        assert args.dare_drop_rate == pytest.approx(0.4)
        assert args.dare_seed == 7
        assert args.drift_gate is True
        assert args.drift_threshold == pytest.approx(0.15)
        assert args.eval_gate is True
        assert args.eval_max_regression == pytest.approx(0.05)
        assert args.eval_heldout == "held.jsonl"

    def test_merge_flags_thread_into_multirun_config(self, tmp_path):
        """The 9 merge knobs reach MultiRunConfig (flag→field mapping).

        cmd_multi_run threads them via the dataclasses.fields(MultiRunConfig)
        filter. A real MultiRunConfig is a dataclass with the T2.2 fields, so
        the keys route to wave6b_cfg_kwargs and land on the constructed
        config. We patch MultiRunTrainer (so no real training fires) and read
        the MultiRunConfig the handler built off the trainer's call kwargs.
        """
        from backpropagate.cli import EXIT_OK, cmd_multi_run

        fake_trainer = MagicMock()
        fake_trainer.run.return_value = MagicMock(
            total_runs=2,
            final_loss=0.3,
            total_duration_seconds=1.0,
            final_checkpoint_path=str(tmp_path / "out"),
            failed_runs=0,
        )

        args = Namespace(
            model="test-model",
            data="data.jsonl",
            runs=2,
            steps=10,
            samples=100,
            merge_mode="slao",
            output=str(tmp_path),
            use_dora=False,
            no_packing=False,
            init_lora_weights="default",
            lora_preset="quality",
            optim="auto",
            mode="lora",
            merge_strategy="ties",
            ties_trim=0.3,
            dare_drop_rate=0.4,
            dare_seed=7,
            drift_gate=True,
            drift_threshold=0.15,
            eval_gate=True,
            eval_max_regression=0.05,
            eval_heldout="held.jsonl",
            resume=None,
            cli_run_id=None,
            verbose=False,
        )

        with patch(
            "backpropagate.multi_run.MultiRunTrainer", return_value=fake_trainer
        ) as mock_trainer_cls:
            rc = cmd_multi_run(args)

        assert rc == EXIT_OK
        # The handler built a real MultiRunConfig and passed it as the
        # ``config=`` kwarg to MultiRunTrainer.
        cfg = mock_trainer_cls.call_args.kwargs["config"]
        assert cfg.merge_strategy == "ties"
        assert cfg.ties_trim_threshold == pytest.approx(0.3)
        assert cfg.dare_drop_rate == pytest.approx(0.4)
        assert cfg.dare_seed == 7
        assert cfg.drift_gate is True
        assert cfg.drift_threshold == pytest.approx(0.15)
        assert cfg.eval_gate is True
        assert cfg.eval_max_regression == pytest.approx(0.05)
        assert cfg.eval_heldout_path == "held.jsonl"

    def test_multirun_banner_names_strategy_and_gates(self, tmp_path, capsys):
        """The multi-run banner surfaces the chosen strategy + gate state."""
        from backpropagate.cli import EXIT_OK, cmd_multi_run

        fake_trainer = MagicMock()
        fake_trainer.run.return_value = MagicMock(
            total_runs=1,
            final_loss=0.3,
            total_duration_seconds=1.0,
            final_checkpoint_path=str(tmp_path / "out"),
            failed_runs=0,
        )

        args = Namespace(
            model="test-model",
            data="data.jsonl",
            runs=1,
            steps=10,
            samples=100,
            merge_mode="slao",
            output=str(tmp_path),
            use_dora=False,
            no_packing=False,
            init_lora_weights="default",
            lora_preset="quality",
            optim="auto",
            mode="lora",
            merge_strategy="dare",
            ties_trim=0.2,
            dare_drop_rate=0.5,
            dare_seed=None,
            drift_gate=True,
            drift_threshold=0.1,
            eval_gate=False,
            eval_max_regression=0.0,
            eval_heldout=None,
            resume=None,
            cli_run_id=None,
            verbose=False,
        )

        with patch(
            "backpropagate.multi_run.MultiRunTrainer", return_value=fake_trainer
        ):
            rc = cmd_multi_run(args)

        assert rc == EXIT_OK
        out = capsys.readouterr().out
        assert "Merge strategy: dare" in out
        assert "drift=on" in out
        assert "eval=off" in out


# =============================================================================
# v1.5 T2.3 (Wave 6b GLUE): export --format ollama-adapter + ollama shelf
# =============================================================================


class TestExportOllamaAdapter:
    """``backprop export --format ollama-adapter`` parse + handler contract."""

    def test_format_accepts_ollama_adapter(self, cli_parser):
        args = cli_parser.parse_args([
            "export", "./out/lora", "--format", "ollama-adapter",
            "--base-model", "llama3.2",
        ])
        assert args.format == "ollama-adapter"
        assert args.base_model == "llama3.2"

    def test_adapter_tag_parses(self, cli_parser):
        args = cli_parser.parse_args([
            "export", "./out/lora", "--format", "ollama-adapter",
            "--base-model", "llama3.2", "--adapter-tag", "taskA",
        ])
        assert args.adapter_tag == "taskA"

    def test_ollama_adapter_calls_export_ollama_adapter(self, tmp_path, capsys):
        """A mocked export_ollama_adapter is invoked with base_model + tag."""
        from backpropagate.cli import EXIT_OK, cmd_export
        from backpropagate.export import ExportFormat

        model_path = tmp_path / "lora"
        model_path.mkdir()

        mock_result = MagicMock()
        mock_result.path = model_path / "Modelfile"
        mock_result.size_mb = 0.01
        mock_result.export_time_seconds = 0.5
        # export_ollama_adapter records the derived <base>:<tag> name here.
        mock_result.quantization = "llama3.2:taskA"
        mock_result.format = ExportFormat.OLLAMA_ADAPTER

        with patch(
            "backpropagate.export.export_ollama_adapter", return_value=mock_result
        ) as mock_export:
            args = Namespace(
                model_path=str(model_path),
                format="ollama-adapter",
                quantization="q4_k_m",
                output=str(tmp_path / "output"),
                ollama=False,
                ollama_name=None,
                base_model="llama3.2",
                adapter_tag="taskA",
                no_model_card=False,
                push_to_hub=None,
                cli_run_id=None,
                verbose=False,
            )
            rc = cmd_export(args)

        assert rc == EXIT_OK
        mock_export.assert_called_once()
        call = mock_export.call_args
        # adapter_path is the first positional; base_model + tag are kw-only.
        assert call.kwargs["base_model"] == "llama3.2"
        assert call.kwargs["tag"] == "taskA"
        out = capsys.readouterr().out
        assert "Registered with Ollama: llama3.2:taskA" in out
        assert "ollama run llama3.2:taskA" in out

    def test_ollama_adapter_missing_base_model_is_user_error(self, tmp_path, capsys):
        """--format ollama-adapter without --base-model → EXIT_USER_ERROR."""
        from backpropagate.cli import EXIT_USER_ERROR, cmd_export

        model_path = tmp_path / "lora"
        model_path.mkdir()

        args = Namespace(
            model_path=str(model_path),
            format="ollama-adapter",
            quantization="q4_k_m",
            output=str(tmp_path / "output"),
            ollama=False,
            ollama_name=None,
            base_model=None,
            adapter_tag=None,
            no_model_card=False,
            push_to_hub=None,
            cli_run_id=None,
            verbose=False,
        )
        rc = cmd_export(args)

        assert rc == EXIT_USER_ERROR
        err = capsys.readouterr().err
        assert "--base-model" in err


class TestOllamaShelfVerb:
    """``backprop ollama shelf <base>`` parse + handler contract."""

    def test_shelf_parses(self, cli_parser):
        args = cli_parser.parse_args(["ollama", "shelf", "llama3.2"])
        assert args.command == "ollama"
        assert args.ollama_command == "shelf"
        assert args.base_model == "llama3.2"

    def test_shelf_requires_base_model(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["ollama", "shelf"])

    def test_shelf_lists_entries(self, capsys):
        """A mocked list_adapter_shelf prints each entry + a count."""
        from backpropagate.cli import EXIT_OK, cmd_ollama_shelf
        from backpropagate.export import AdapterShelfEntry

        entries = [
            AdapterShelfEntry(
                model_name="llama3.2:taskA", base="llama3.2", tag="taskA",
                size="2.0 GB", modified="2 hours ago",
            ),
            AdapterShelfEntry(
                model_name="llama3.2:taskB", base="llama3.2", tag="taskB",
            ),
        ]

        with patch(
            "backpropagate.export.list_adapter_shelf", return_value=entries
        ) as mock_shelf:
            args = Namespace(base_model="llama3.2", verbose=False)
            rc = cmd_ollama_shelf(args)

        assert rc == EXIT_OK
        mock_shelf.assert_called_once_with("llama3.2")
        out = capsys.readouterr().out
        assert "llama3.2:taskA" in out
        assert "llama3.2:taskB" in out
        assert "Listed 2 adapter(s)" in out

    def test_shelf_empty_when_ollama_missing(self, capsys):
        """Empty shelf + ollama CLI absent → EX_UNAVAILABLE."""
        from backpropagate.cli import EXIT_UNAVAILABLE, cmd_ollama_shelf

        # cmd_ollama_shelf does a local ``import shutil`` then ``shutil.which``;
        # patch the global shutil.which (no module-level cli.shutil to patch).
        with patch(
            "backpropagate.export.list_adapter_shelf", return_value=[]
        ), patch("shutil.which", return_value=None):
            args = Namespace(base_model="llama3.2", verbose=False)
            rc = cmd_ollama_shelf(args)

        assert rc == EXIT_UNAVAILABLE


# =============================================================================
# CLI-A-001 (v1.6): train --batch-size validator + replay --override batch_size
# coercion contract. Two siblings of the same defect: a non-numeric batch size
# slipping past the CLI surface to detonate as a bare int() ValueError deep in
# Trainer, masked by `except Exception` -> redacted EXIT_RUNTIME_ERROR (exit 2)
# instead of a flag-named EXIT_USER_ERROR (exit 1).
# =============================================================================


class TestTrainBatchSizeValidator:
    """``train --batch-size`` now has an ``_auto_or_positive_int`` validator."""

    def test_batch_size_auto_default_stays_string(self, cli_parser):
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.batch_size == "auto"

    def test_batch_size_auto_literal_accepted(self, cli_parser):
        args = cli_parser.parse_args(
            ["train", "-d", "data.jsonl", "--batch-size", "auto"]
        )
        assert args.batch_size == "auto"

    def test_batch_size_auto_case_insensitive(self, cli_parser):
        args = cli_parser.parse_args(
            ["train", "-d", "data.jsonl", "--batch-size", "AUTO"]
        )
        assert args.batch_size == "auto"

    def test_batch_size_numeric_parses_to_int(self, cli_parser):
        args = cli_parser.parse_args(
            ["train", "-d", "data.jsonl", "--batch-size", "8"]
        )
        assert args.batch_size == 8
        assert isinstance(args.batch_size, int)

    def test_batch_size_non_numeric_rejected_at_parse(self, cli_parser, capsys):
        """``--batch-size foo`` exits 2 at argparse with a flag-named message.

        Pre-fix this slipped through (no ``type=``) and the tautology at
        cmd_train forwarded the raw string to Trainer, which did a bare
        ``int(batch_size)`` -> ValueError -> redacted EXIT_RUNTIME_ERROR.
        """
        with pytest.raises(SystemExit) as exc:
            cli_parser.parse_args(["train", "-d", "data.jsonl", "--batch-size", "foo"])
        # argparse exits with code 2 on a type-validation failure.
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "batch-size" in err.lower()

    def test_batch_size_zero_rejected_at_parse(self, cli_parser):
        with pytest.raises(SystemExit) as exc:
            cli_parser.parse_args(["train", "-d", "data.jsonl", "--batch-size", "0"])
        assert exc.value.code == 2

    def test_auto_or_positive_int_helper_directly(self):
        """Unit-level contract for the new validator helper."""
        import argparse

        from backpropagate.cli import _auto_or_positive_int

        assert _auto_or_positive_int("auto") == "auto"
        assert _auto_or_positive_int("AUTO") == "auto"
        assert _auto_or_positive_int("4") == 4
        with pytest.raises(argparse.ArgumentTypeError):
            _auto_or_positive_int("foo")
        with pytest.raises(argparse.ArgumentTypeError):
            _auto_or_positive_int("0")


class TestReplayOverrideCoercion:
    """``replay --override batch_size=...`` honors the _coerce docstring."""

    def _record_single_run(self, tmp_path):
        from backpropagate.checkpoints import RunHistoryManager

        mgr = RunHistoryManager(str(tmp_path))
        mgr.record_run_started(
            run_id="ovr-run",
            model_name="m",
            dataset_info="data.jsonl",
            hyperparameters={
                "max_steps": 10,
                "lora_r": 8,
                "learning_rate": 1e-4,
            },
            session_kind="single_run",
        )
        mgr.record_run_completed(run_id="ovr-run", final_loss=0.1)
        return mgr

    def _replay_args(self, tmp_path, override_tokens):
        return Namespace(
            run_id="ovr-run",
            output=str(tmp_path),
            override=override_tokens,
            data=None,
            json=False,
            cli_run_id=None,
            verbose=False,
        )

    def test_bad_batch_size_override_is_user_error(self, tmp_path, capsys):
        """``--override batch_size=foo`` -> EXIT_USER_ERROR (1), NOT exit 2.

        Pre-fix ``_coerce`` returned the raw string "foo" (falling through its
        own docstring promise), which became batch_size and hit a bare int()
        ValueError deep in Trainer -> redacted EXIT_RUNTIME_ERROR.
        """
        from backpropagate.cli import EXIT_USER_ERROR, _parse_replay_override, cmd_replay

        self._record_single_run(tmp_path)
        args = self._replay_args(tmp_path, [_parse_replay_override("batch_size=foo")])

        rc = cmd_replay(args)

        assert rc == EXIT_USER_ERROR
        combined = (capsys.readouterr().out + capsys.readouterr().err).lower()
        # The error must name the offending override key (not a redacted
        # generic runtime failure).
        assert "batch_size" in combined or "batch-size" in combined

    def test_bad_lr_override_is_user_error(self, tmp_path, capsys):
        """A non-numeric value on another numeric key (lr) also user-errors."""
        from backpropagate.cli import EXIT_USER_ERROR, _parse_replay_override, cmd_replay

        self._record_single_run(tmp_path)
        args = self._replay_args(tmp_path, [_parse_replay_override("lr=fast")])

        rc = cmd_replay(args)
        assert rc == EXIT_USER_ERROR

    def test_batch_size_auto_override_accepted(self, tmp_path):
        """``--override batch_size=auto`` is the documented escape hatch."""
        from backpropagate.cli import EXIT_OK, _parse_replay_override, cmd_replay

        self._record_single_run(tmp_path)
        args = self._replay_args(
            tmp_path, [_parse_replay_override("batch_size=auto")]
        )

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(final_loss=0.05, run_id="r2")
        with patch("backpropagate.trainer.Trainer", return_value=fake_trainer) as cls:
            rc = cmd_replay(args)

        assert rc == EXIT_OK
        # "auto" reached the Trainer kwargs as the literal string, not coerced.
        assert cls.call_args.kwargs.get("batch_size") == "auto"

    def test_numeric_batch_size_override_coerces_to_int(self, tmp_path):
        """``--override batch_size=4`` still coerces to int 4."""
        from backpropagate.cli import EXIT_OK, _parse_replay_override, cmd_replay

        self._record_single_run(tmp_path)
        args = self._replay_args(tmp_path, [_parse_replay_override("batch_size=4")])

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(final_loss=0.05, run_id="r3")
        with patch("backpropagate.trainer.Trainer", return_value=fake_trainer) as cls:
            rc = cmd_replay(args)

        assert rc == EXIT_OK
        assert cls.call_args.kwargs.get("batch_size") == 4

    def test_string_valued_override_unaffected(self, tmp_path):
        """A genuinely string-valued key (optim) keeps best-effort behavior."""
        from backpropagate.cli import EXIT_OK, _parse_replay_override, cmd_replay

        self._record_single_run(tmp_path)
        args = self._replay_args(tmp_path, [_parse_replay_override("optim=adamw_8bit")])

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(final_loss=0.05, run_id="r4")
        with patch("backpropagate.trainer.Trainer", return_value=fake_trainer):
            rc = cmd_replay(args)

        # No InvalidSettingError for a string key — replay proceeds.
        assert rc == EXIT_OK
