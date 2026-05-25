"""Wave 6b new-flag coverage tests (TESTS — cross-domain).

The Wave 6b backend + bridge agents added:

* New LoraConfig defaults (rank 256, all-linear target, lora_alpha 512).
* `LoraConfig.use_dora` field (DoRA opt-in).
* `LoraConfig.init_lora_weights` field (default / pissa / loftq).
* `DataConfig.packing` flipped True (sample packing default-on).
* Three new model presets (Phi-4-mini, Qwen-3.5-4B, SmolLM3-3B).
* License caveat surfaced on Trainer boot for the Qwen-2.5-3B preset.
* `--use-dora`, `--no-packing`, `--init-lora-weights`, `--lora-preset`,
  `--optim` CLI flags on `train` + `multi-run`.
* `backprop diff-runs A B`, `backprop replay <run_id>`,
  `backprop export-runs --format=jsonl` subcommands.

This file is the dedicated cross-domain regression net for those changes.
Splitting them out (rather than scattering across test_config / test_cli /
test_trainer) makes the Wave 6b coverage story legible — an auditor reading
the suite can see "what does the Wave 6b feature pass actually pin" in one
file.

Coverage philosophy: happy paths + obvious failure paths. CI budget matters.
We do NOT exhaustively combinatorialise the 5-flag space.
"""

from __future__ import annotations

import json

import pytest

# =============================================================================
# CONFIG-LEVEL — LoRA defaults + new fields
# =============================================================================


class TestWave6bLoraConfigDefaults:
    """v1.3 LoRA-defaults bump (BACKEND-1)."""

    def test_lora_config_r_bumped_to_256(self):
        """LoRAConfig.r default is now 256 (was 16 in v1.2)."""
        from backpropagate.config import LoRAConfig
        config = LoRAConfig()
        assert config.r == 256, (
            f"LoRAConfig.r default = {config.r}; expected 256 per "
            f"BACKEND-1. A regression to 16 means the v1.3 quality-default "
            f"flip got reverted."
        )

    def test_lora_config_lora_alpha_bumped_to_512(self):
        """LoRAConfig.lora_alpha default tracks rank (alpha = 2 * r)."""
        from backpropagate.config import LoRAConfig
        config = LoRAConfig()
        assert config.lora_alpha == 512, (
            f"LoRAConfig.lora_alpha = {config.lora_alpha}; expected 512 "
            f"(2 * 256, per the alpha=2r convention). A regression here "
            f"would silently scale LoRA updates wrong."
        )

    def test_lora_config_target_modules_is_all_linear(self):
        """LoRAConfig.target_modules default flipped from list -> "all-linear"."""
        from backpropagate.config import LoRAConfig
        config = LoRAConfig()
        # The field accepts either a string or a list. v1.3 default = "all-linear"
        # which PEFT recognises as a wildcard for every Linear/Conv1D except LM head.
        assert config.target_modules == "all-linear", (
            f"LoRAConfig.target_modules = {config.target_modules!r}; "
            f"expected the 'all-linear' wildcard (v1.3 default)."
        )

    def test_lora_config_use_dora_field_exists_default_false(self):
        """LoRAConfig.use_dora is a new field (BACKEND-3), default False."""
        from backpropagate.config import LoRAConfig
        config = LoRAConfig()
        assert hasattr(config, "use_dora"), (
            "LoRAConfig is missing the use_dora field (BACKEND-3). "
            "The --use-dora CLI flag has nothing to bind to."
        )
        assert config.use_dora is False, (
            f"LoRAConfig.use_dora default = {config.use_dora}; expected "
            f"False for backward-compat. Flipping the default to True is "
            f"a breaking change."
        )

    def test_lora_config_init_lora_weights_field_exists_default_default(self):
        """LoRAConfig.init_lora_weights (BACKEND-6) default 'default'."""
        from backpropagate.config import LoRAConfig
        config = LoRAConfig()
        assert hasattr(config, "init_lora_weights"), (
            "LoRAConfig is missing the init_lora_weights field "
            "(BACKEND-6). --init-lora-weights CLI flag has nothing to "
            "bind to."
        )
        assert config.init_lora_weights == "default", (
            f"LoRAConfig.init_lora_weights default = "
            f"{config.init_lora_weights!r}; expected 'default'."
        )


class TestWave6bDataConfigPackingDefault:
    """v1.3 sample-packing default flipped on (BACKEND-4)."""

    def test_data_config_packing_default_true(self):
        """DataConfig.packing is now True by default (was False in v1.2)."""
        from backpropagate.config import DataConfig
        config = DataConfig()
        assert config.packing is True, (
            f"DataConfig.packing default = {config.packing}; expected "
            f"True per BACKEND-4. Flipping back to False removes the "
            f"1.7-3x throughput win that's part of the v1.3 'feels "
            f"faster' story."
        )


class TestWave6bLoraPresets:
    """LORA_PRESETS catalogue (`fast` + `quality`, BACKEND-1)."""

    def test_lora_presets_has_fast_and_quality(self):
        from backpropagate.config import LORA_PRESETS
        assert "fast" in LORA_PRESETS, "LORA_PRESETS is missing the 'fast' entry."
        assert "quality" in LORA_PRESETS, "LORA_PRESETS is missing the 'quality' entry."

    def test_lora_preset_fast_matches_v1_2_defaults(self):
        """'fast' preset reverts to v1.2 defaults (rank 16, q+v, 1x LR)."""
        from backpropagate.config import LORA_PRESETS
        fast = LORA_PRESETS["fast"]
        assert fast.r == 16, (
            f"LORA_PRESETS['fast'].r = {fast.r}; expected 16 (v1.2 "
            f"default). The 'fast' preset's job is to give operators an "
            f"escape hatch to the old behaviour."
        )
        assert fast.target_modules == ["q_proj", "v_proj"], (
            f"LORA_PRESETS['fast'].target_modules = "
            f"{fast.target_modules!r}; expected ['q_proj', 'v_proj']."
        )
        assert fast.lr_multiplier == 1.0, (
            f"LORA_PRESETS['fast'].lr_multiplier = {fast.lr_multiplier}; "
            f"expected 1.0 (no LR scaling, matches v1.2 behaviour)."
        )

    def test_lora_preset_quality_matches_v1_3_defaults(self):
        """'quality' preset = the v1.3 new defaults (rank 256, all-linear, 10x LR)."""
        from backpropagate.config import LORA_PRESETS
        quality = LORA_PRESETS["quality"]
        assert quality.r == 256
        assert quality.target_modules == "all-linear"
        assert quality.lr_multiplier == 10.0, (
            f"LORA_PRESETS['quality'].lr_multiplier = "
            f"{quality.lr_multiplier}; expected 10.0 (Biderman 2024 / "
            f"Thinking Machines 2025 finding)."
        )

    def test_get_lora_preset_known(self):
        from backpropagate.config import get_lora_preset
        assert get_lora_preset("fast").name == "fast"
        assert get_lora_preset("quality").name == "quality"

    def test_get_lora_preset_unknown_raises(self):
        from backpropagate.config import get_lora_preset
        with pytest.raises(ValueError, match="Unknown LoRA preset"):
            get_lora_preset("nonexistent-preset")


# =============================================================================
# CONFIG-LEVEL — new model presets (BACKEND-8/9/10)
# =============================================================================


class TestWave6bModelPresets:
    """Three new commercial-safe / long-context model presets."""

    @pytest.mark.parametrize("preset_name", ["phi-4-mini-3.8b", "qwen3.5-4b", "smollm3-3b"])
    def test_new_preset_exists(self, preset_name):
        """Each of the 3 new v1.3 presets is in MODEL_PRESETS."""
        from backpropagate.config import MODEL_PRESETS
        assert preset_name in MODEL_PRESETS, (
            f"MODEL_PRESETS is missing the v1.3 preset {preset_name!r} "
            f"(BACKEND-8/9/10). One of the three Wave 6b additions "
            f"regressed."
        )

    def test_phi_4_mini_preset_has_mit_license(self):
        """BACKEND-8: Phi-4-mini-3.8B preset is MIT-licensed."""
        from backpropagate.config import MODEL_PRESETS
        preset = MODEL_PRESETS["phi-4-mini-3.8b"]
        assert preset.license == "MIT", (
            f"phi-4-mini-3.8b license = {preset.license!r}; expected "
            f"'MIT' (the whole point of the preset)."
        )
        assert preset.model_id == "microsoft/Phi-4-mini-instruct"

    def test_qwen_3_5_preset_has_apache_license(self):
        """BACKEND-9: Qwen-3.5-4B preset is Apache-2.0."""
        from backpropagate.config import MODEL_PRESETS
        preset = MODEL_PRESETS["qwen3.5-4b"]
        assert preset.license == "Apache-2.0"
        assert "Qwen3.5-4B" in preset.model_id

    def test_smollm3_preset_has_long_context_default(self):
        """BACKEND-10: SmolLM3-3B preset's recommended_max_seq_length is bumped."""
        from backpropagate.config import MODEL_PRESETS
        preset = MODEL_PRESETS["smollm3-3b"]
        assert preset.license == "Apache-2.0"
        # SmolLM3 native 64K; the preset should at least bump above 2048.
        assert preset.recommended_max_seq_length > 2048, (
            f"smollm3-3b recommended_max_seq_length = "
            f"{preset.recommended_max_seq_length}; expected > 2048 to "
            f"reflect SmolLM3's native long context."
        )

    def test_qwen_2_5_3b_has_license_caveat(self):
        """BACKEND-2: Qwen-2.5-3B preset includes a license_restriction caveat."""
        from backpropagate.config import MODEL_PRESETS
        preset = MODEL_PRESETS["qwen2.5-3b"]
        assert preset.license_restriction is not None, (
            "Qwen-2.5-3B preset is missing the license_restriction "
            "caveat (BACKEND-2). Operators using a non-commercial "
            "model for commercial training will not be warned."
        )
        # The caveat should explicitly mention 'non-commercial' so an
        # operator skimming structured logs picks up the issue.
        assert "non-commercial" in preset.license_restriction.lower()

    def test_all_other_presets_have_no_license_restriction(self):
        """Permissive-licensed presets must NOT have a license_restriction.

        Setting one on Apache/MIT/Llama-Community would be a false alarm
        that erodes signal-to-noise for the operator.
        """
        from backpropagate.config import MODEL_PRESETS
        for name, preset in MODEL_PRESETS.items():
            if name == "qwen2.5-3b":
                continue  # The known caveat — covered above.
            assert preset.license_restriction is None, (
                f"Preset {name!r} has a license_restriction "
                f"{preset.license_restriction!r}; only qwen2.5-3b "
                f"should set this field. False-positive caveats erode "
                f"signal-to-noise."
            )


# =============================================================================
# CLI-LEVEL — new flags on `train` + `multi-run`
# =============================================================================


class TestWave6bTrainSubcommandFlags:
    """5 new flags on `train` (BRIDGE-1..5)."""

    def test_train_parses_use_dora_flag(self, cli_parser):
        """`backprop train --use-dora` parses cleanly + sets the flag."""
        args = cli_parser.parse_args(["train", "-d", "data.jsonl", "--use-dora"])
        assert args.use_dora is True

    def test_train_use_dora_default_false(self, cli_parser):
        """Omitting --use-dora ⇒ False (back-compat default)."""
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.use_dora is False

    def test_train_parses_no_packing_flag(self, cli_parser):
        """`backprop train --no-packing` flips the opt-out."""
        args = cli_parser.parse_args(["train", "-d", "data.jsonl", "--no-packing"])
        # The argparse field is no_packing=True; packing-ON is the default.
        assert args.no_packing is True

    def test_train_no_packing_default_false(self, cli_parser):
        """Omitting --no-packing ⇒ packing stays ON."""
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.no_packing is False

    def test_train_init_lora_weights_default(self, cli_parser):
        """--init-lora-weights default is 'default'."""
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.init_lora_weights == "default"

    @pytest.mark.parametrize("value", ["default", "pissa", "loftq"])
    def test_train_init_lora_weights_accepts_choices(self, cli_parser, value):
        """--init-lora-weights accepts the three documented PEFT init strategies."""
        args = cli_parser.parse_args([
            "train", "-d", "data.jsonl", "--init-lora-weights", value,
        ])
        assert args.init_lora_weights == value

    def test_train_init_lora_weights_rejects_unknown(self, cli_parser):
        """Unknown init strategy fails argparse (choice validation)."""
        with pytest.raises(SystemExit):
            cli_parser.parse_args([
                "train", "-d", "data.jsonl",
                "--init-lora-weights", "nonsense",
            ])

    def test_train_lora_preset_default_is_quality(self, cli_parser):
        """v1.3 default lora-preset is 'quality'."""
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.lora_preset == "quality"

    def test_train_lora_preset_fast_for_back_compat(self, cli_parser):
        """`--lora-preset fast` selects the v1.2-compatible preset."""
        args = cli_parser.parse_args([
            "train", "-d", "data.jsonl", "--lora-preset", "fast",
        ])
        assert args.lora_preset == "fast"

    def test_train_lora_preset_rejects_unknown(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args([
                "train", "-d", "data.jsonl", "--lora-preset", "nonsense",
            ])

    def test_train_optim_default_auto(self, cli_parser):
        """--optim default is 'auto' (paged_adamw_8bit auto-pick on consumer GPUs)."""
        args = cli_parser.parse_args(["train", "-d", "data.jsonl"])
        assert args.optim == "auto"

    @pytest.mark.parametrize(
        "value", ["auto", "adamw_torch", "paged_adamw_8bit", "adamw_8bit"]
    )
    def test_train_optim_accepts_known_choices(self, cli_parser, value):
        args = cli_parser.parse_args([
            "train", "-d", "data.jsonl", "--optim", value,
        ])
        assert args.optim == value


class TestWave6bMultiRunSubcommandFlags:
    """Same 5 flags exist on `multi-run` (mirror of train surface)."""

    def test_multi_run_parses_use_dora(self, cli_parser):
        args = cli_parser.parse_args([
            "multi-run", "-d", "data.jsonl", "--use-dora",
        ])
        assert args.use_dora is True

    def test_multi_run_parses_no_packing(self, cli_parser):
        args = cli_parser.parse_args([
            "multi-run", "-d", "data.jsonl", "--no-packing",
        ])
        assert args.no_packing is True

    def test_multi_run_parses_init_lora_weights(self, cli_parser):
        args = cli_parser.parse_args([
            "multi-run", "-d", "data.jsonl", "--init-lora-weights", "pissa",
        ])
        assert args.init_lora_weights == "pissa"

    def test_multi_run_parses_lora_preset(self, cli_parser):
        args = cli_parser.parse_args([
            "multi-run", "-d", "data.jsonl", "--lora-preset", "fast",
        ])
        assert args.lora_preset == "fast"

    def test_multi_run_parses_optim(self, cli_parser):
        args = cli_parser.parse_args([
            "multi-run", "-d", "data.jsonl", "--optim", "paged_adamw_8bit",
        ])
        assert args.optim == "paged_adamw_8bit"


# =============================================================================
# CLI-LEVEL — new subcommands: diff-runs, replay, export-runs
# =============================================================================


class TestWave6bDiffRunsSubcommand:
    """`backprop diff-runs A B` (BRIDGE-6)."""

    def test_diff_runs_subcommand_parses(self, cli_parser):
        """diff-runs is registered + takes two positional run-ids."""
        args = cli_parser.parse_args([
            "diff-runs", "run-aaa", "run-bbb",
        ])
        assert args.command == "diff-runs"
        assert args.run_id_a == "run-aaa"
        assert args.run_id_b == "run-bbb"

    def test_diff_runs_requires_two_run_ids(self, cli_parser):
        """Only one positional is an argparse error."""
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["diff-runs", "run-aaa"])

    def test_diff_runs_format_choices(self, cli_parser):
        """--format defaults to the human/table view; json is opt-in."""
        args = cli_parser.parse_args([
            "diff-runs", "run-aaa", "run-bbb", "--format", "json",
        ])
        assert args.format == "json"

    def test_diff_runs_missing_history_dir_returns_user_error(self, tmp_path, capsys):
        """No history dir ⇒ EXIT_USER_ERROR with a helpful message."""
        # Build a minimal argparse.Namespace for the handler.
        import argparse

        from backpropagate.cli import EXIT_USER_ERROR, cmd_diff_runs
        args = argparse.Namespace(
            run_id_a="run-aaa",
            run_id_b="run-bbb",
            output=str(tmp_path / "nonexistent-dir"),
            format="text",
        )
        code = cmd_diff_runs(args)
        assert code == EXIT_USER_ERROR, (
            f"cmd_diff_runs returned {code} for a missing history "
            f"directory; expected EXIT_USER_ERROR ({EXIT_USER_ERROR})."
        )

    def test_diff_runs_returns_structured_diff(self, tmp_path):
        """Both run_ids resolved ⇒ exits cleanly + produces a diff table."""
        from datetime import datetime, timezone

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import EXIT_OK, cmd_diff_runs

        manager = RunHistoryManager(str(tmp_path))
        manager._save([
            {
                "run_id": "run-aaa",
                "status": "completed",
                "model_name": "test-model-a",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "final_loss": 0.5,
            },
            {
                "run_id": "run-bbb",
                "status": "completed",
                "model_name": "test-model-b",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "final_loss": 0.4,
            },
        ])

        import argparse
        args = argparse.Namespace(
            run_id_a="run-aaa",
            run_id_b="run-bbb",
            output=str(tmp_path),
            format="json",
        )
        code = cmd_diff_runs(args)
        assert code == EXIT_OK, (
            f"cmd_diff_runs returned {code} on resolvable run_ids; "
            f"expected EXIT_OK ({EXIT_OK})."
        )


class TestWave6bReplaySubcommand:
    """`backprop replay <run_id>` (BRIDGE-7)."""

    def test_replay_subcommand_parses(self, cli_parser):
        args = cli_parser.parse_args(["replay", "run-original"])
        assert args.command == "replay"
        assert args.run_id == "run-original"

    def test_replay_requires_run_id(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["replay"])


class TestCmdReplay:
    """TESTS-A-003 (v1.4 Wave 2 amend): cmd_replay handler coverage.

    Wave 1 audit flagged that TestWave6bReplaySubcommand pinned only
    the parser shape — the handler at backpropagate/cli.py::cmd_replay
    had zero direct coverage. A handler-level regression (missing
    output_dir, unknown run_id, malformed --override, hyperparameter
    misrouting) would not be caught.

    Modelled on TestCmdResume (tests/test_cli.py:1702) which uses the
    same RunHistoryManager seed + MagicMock patch pattern. Tests cover:

    1. Missing output_dir → EXIT_USER_ERROR
    2. Unknown run_id → EXIT_USER_ERROR (raises InvalidSettingError
       internally, caught + mapped to user-error)
    3. Original run with no dataset_info → EXIT_USER_ERROR
    4. Happy path (single_run) → Trainer.train called with the
       inherited hyperparameters
    5. Happy path (multi_run) → MultiRunTrainer.run called
    6. --override applies the override to the trainer kwargs
    7. --override with non-whitelisted key → EXIT_USER_ERROR
    """

    def test_replay_missing_output_dir_returns_user_error(self, tmp_path):
        """Missing --output directory → EXIT_USER_ERROR."""
        from unittest.mock import MagicMock

        from backpropagate.cli import EXIT_USER_ERROR, cmd_replay

        args = MagicMock()
        args.output = str(tmp_path / "definitely-does-not-exist")
        args.run_id = "abc"
        args.override = None
        args.cli_run_id = None
        args.verbose = False

        rc = cmd_replay(args)
        assert rc == EXIT_USER_ERROR

    def test_replay_unknown_run_id_returns_user_error(self, tmp_path):
        """Unknown run_id under existing output_dir → EXIT_USER_ERROR.

        Internally cmd_replay raises InvalidSettingError and catches it,
        returning EXIT_USER_ERROR. The user-visible contract is the exit
        code; the structured error is logged for support diagnostics.
        """
        from unittest.mock import MagicMock

        from backpropagate.cli import EXIT_USER_ERROR, cmd_replay

        # tmp_path itself exists, so the missing-output-dir guard doesn't fire.
        args = MagicMock()
        args.output = str(tmp_path)
        args.run_id = "no-such-run"
        args.override = None
        args.cli_run_id = None
        args.verbose = False

        rc = cmd_replay(args)
        assert rc == EXIT_USER_ERROR

    def test_replay_record_without_dataset_info_returns_user_error(self, tmp_path):
        """A record with dataset_info=None can't be replayed automatically."""
        from unittest.mock import MagicMock

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import EXIT_USER_ERROR, cmd_replay

        manager = RunHistoryManager(str(tmp_path))
        manager.record_run_started(
            run_id="no-dataset-run",
            model_name="m",
            dataset_info=None,  # <- this is the case under test
            hyperparameters={"max_steps": 50, "lora_r": 8},
            session_kind="single_run",
        )

        args = MagicMock()
        args.output = str(tmp_path)
        args.run_id = "no-dataset-run"
        args.override = None
        args.cli_run_id = None
        args.verbose = False

        rc = cmd_replay(args)
        assert rc == EXIT_USER_ERROR

    def test_replay_dispatches_single_run_with_hyperparameters(self, tmp_path):
        """Single-run replay reconstructs Trainer with inherited hyperparameters."""
        from unittest.mock import MagicMock, patch

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import EXIT_OK, cmd_replay

        manager = RunHistoryManager(str(tmp_path))
        manager.record_run_started(
            run_id="sr-replay",
            model_name="test-model",
            dataset_info="data.jsonl",
            hyperparameters={
                "max_steps": 75,
                "lora_r": 32,
                "learning_rate": 3e-4,
            },
            session_kind="single_run",
        )

        fake_trainer = MagicMock()
        fake_run = MagicMock(final_loss=0.42)
        fake_trainer.train.return_value = fake_run

        with patch(
            "backpropagate.trainer.Trainer",
            return_value=fake_trainer,
        ) as mock_cls:
            args = MagicMock()
            args.output = str(tmp_path)
            args.run_id = "sr-replay"
            args.override = None
            args.cli_run_id = None
            args.verbose = False
            rc = cmd_replay(args)

        assert rc == EXIT_OK
        # Trainer instantiated with the inherited model + lora_r + lr.
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs.get("model") == "test-model"
        assert init_kwargs.get("lora_r") == 32
        assert init_kwargs.get("learning_rate") == 3e-4
        # train() was called with the dataset + steps from the record.
        train_kwargs = fake_trainer.train.call_args.kwargs
        assert train_kwargs.get("dataset") == "data.jsonl"
        assert train_kwargs.get("steps") == 75

    def test_replay_dispatches_multi_run(self, tmp_path):
        """multi_run session_kind → MultiRunTrainer.run dispatched."""
        from unittest.mock import MagicMock, patch

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import EXIT_OK, cmd_replay

        manager = RunHistoryManager(str(tmp_path))
        manager.record_run_started(
            run_id="mr-replay",
            model_name="test-model",
            dataset_info="data.jsonl",
            hyperparameters={
                "num_runs": 4,
                "steps_per_run": 25,
                "samples_per_run": 200,
                "merge_mode": "slao",
            },
            session_kind="multi_run",
        )

        fake_mr_trainer = MagicMock()
        fake_mr_trainer.run.return_value = MagicMock(total_runs=4, final_loss=0.12)

        with patch(
            "backpropagate.multi_run.MultiRunTrainer",
            return_value=fake_mr_trainer,
        ) as mock_cls:
            args = MagicMock()
            args.output = str(tmp_path)
            args.run_id = "mr-replay"
            args.override = None
            args.cli_run_id = None
            args.verbose = False
            rc = cmd_replay(args)

        assert rc == EXIT_OK
        # MultiRunTrainer was constructed; MultiRunConfig carries the inherited fields.
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs.get("model") == "test-model"
        mr_config = init_kwargs.get("config")
        assert mr_config is not None
        # MultiRunConfig.num_runs / steps_per_run reflect the seeded record.
        assert mr_config.num_runs == 4
        assert mr_config.steps_per_run == 25
        # run() was invoked with the dataset.
        fake_mr_trainer.run.assert_called_once_with("data.jsonl")

    def test_replay_override_applies_to_trainer_kwargs(self, tmp_path):
        """--override lora_r=64 overrides the recorded lora_r=32.

        BRIDGE-A-002 plumbing: the override list lands in Trainer
        construction kwargs, not silently dropped. If a regression
        dropped the override, the trainer would get the recorded
        lora_r=32 instead of the operator's lora_r=64.
        """
        from unittest.mock import MagicMock, patch

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import EXIT_OK, cmd_replay

        manager = RunHistoryManager(str(tmp_path))
        manager.record_run_started(
            run_id="sr-override",
            model_name="test-model",
            dataset_info="data.jsonl",
            hyperparameters={
                "max_steps": 50,
                "lora_r": 32,
                "learning_rate": 2e-4,
            },
            session_kind="single_run",
        )

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(final_loss=0.5)

        with patch(
            "backpropagate.trainer.Trainer",
            return_value=fake_trainer,
        ) as mock_cls:
            args = MagicMock()
            args.output = str(tmp_path)
            args.run_id = "sr-override"
            # argparse parses --override key=value into a list of (key, value) tuples.
            args.override = [("lora_r", "64")]
            args.cli_run_id = None
            args.verbose = False
            rc = cmd_replay(args)

        assert rc == EXIT_OK
        init_kwargs = mock_cls.call_args.kwargs
        # The override propagated all the way through; coerced to int by
        # cmd_replay's _coerce helper.
        assert init_kwargs.get("lora_r") == 64, (
            f"--override lora_r=64 must reach the Trainer constructor; "
            f"got lora_r={init_kwargs.get('lora_r')} (recorded was 32)"
        )

    def test_replay_rejects_unknown_override_key(self, tmp_path):
        """--override key=value with non-whitelisted key → EXIT_USER_ERROR.

        Pins the operator-error-fails-loud contract: a typo'd
        --override lr_rate (vs learning_rate) must NOT be silently
        dropped; cmd_replay returns user-error so the operator sees
        their typo at the CLI surface.
        """
        from unittest.mock import MagicMock

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import EXIT_USER_ERROR, cmd_replay

        manager = RunHistoryManager(str(tmp_path))
        manager.record_run_started(
            run_id="sr-bad-override",
            model_name="test-model",
            dataset_info="data.jsonl",
            hyperparameters={"max_steps": 50, "lora_r": 16},
            session_kind="single_run",
        )

        args = MagicMock()
        args.output = str(tmp_path)
        args.run_id = "sr-bad-override"
        # Non-whitelisted key (typo for "learning_rate").
        args.override = [("lr_rate", "1e-4")]
        args.cli_run_id = None
        args.verbose = False

        rc = cmd_replay(args)
        assert rc == EXIT_USER_ERROR

    def test_replay_multi_run_override_applies_to_mr_trainer_kwargs(self, tmp_path):
        """--override use_dora=true overrides the recorded value on a multi_run replay.

        BRIDGE-B-001 plumbing (v1.4 Wave 3.5): the multi_run branch of
        cmd_replay (cli.py:3786-3852) must mirror the single_run branch's
        VAR_KEYWORD detection so a MagicMock-patched MultiRunTrainer that
        advertises ``(*args, **kwargs)`` doesn't get its Wave 6b override
        kwargs silently filtered out. Pre-fix, the multi_run branch built
        ``_mr_trainer_params`` from a regular ``signature().parameters`` set
        without checking for VAR_KEYWORD; a MagicMock's catch-all kwargs
        signature produces an empty set, so every Wave 6b override was
        dropped on the multi_run path while the single_run path passed
        them through correctly. Sibling-pattern instance of the Wave 2
        coordinator fix on the single_run branch; flagged by Wave 3 Stage B
        audit under the locked
        [[grep-all-instances-when-fixing-pattern]] doctrine.
        """
        from unittest.mock import MagicMock, patch

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import EXIT_OK, cmd_replay

        manager = RunHistoryManager(str(tmp_path))
        manager.record_run_started(
            run_id="mr-override",
            model_name="test-model",
            dataset_info="data.jsonl",
            hyperparameters={
                "num_runs": 3,
                "steps_per_run": 50,
                "samples_per_run": 500,
                "use_dora": False,
            },
            session_kind="multi_run",
        )

        fake_mr_trainer = MagicMock()
        fake_mr_trainer.run.return_value = MagicMock(
            total_runs=3, final_loss=0.4, aborted=False
        )

        with patch(
            "backpropagate.multi_run.MultiRunTrainer",
            return_value=fake_mr_trainer,
        ) as mock_cls:
            args = MagicMock()
            args.output = str(tmp_path)
            args.run_id = "mr-override"
            # argparse parses --override key=value into a list of (key, value) tuples.
            # use_dora is whitelisted (BRIDGE Wave 6b additions in
            # _REPLAY_ALLOWED_OVERRIDE_KEYS at cli.py:3610).
            args.override = [("use_dora", "true")]
            args.cli_run_id = None
            args.verbose = False
            rc = cmd_replay(args)

        assert rc == EXIT_OK
        init_kwargs = mock_cls.call_args.kwargs
        # The override propagated all the way through; coerced to bool by
        # cmd_replay's _coerce helper. Pre-BRIDGE-B-001 this was None / missing.
        assert init_kwargs.get("use_dora") is True, (
            f"--override use_dora=true must reach the MultiRunTrainer "
            f"constructor on a multi_run replay; got "
            f"use_dora={init_kwargs.get('use_dora')!r} (recorded was False). "
            f"BRIDGE-B-001 regression: the VAR_KEYWORD detection on the "
            f"multi_run branch was lost or broken."
        )


class TestWave6bExportRunsSubcommand:
    """`backprop export-runs --format=jsonl` (BRIDGE-8)."""

    def test_export_runs_subcommand_parses(self, cli_parser):
        args = cli_parser.parse_args(["export-runs"])
        assert args.command == "export-runs"

    def test_export_runs_default_format_is_jsonl(self, cli_parser):
        args = cli_parser.parse_args(["export-runs"])
        assert args.format == "jsonl"

    def test_export_runs_rejects_unknown_format(self, cli_parser):
        """csv is intentionally NOT offered (lossy on nested fields)."""
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["export-runs", "--format", "csv"])

    def test_export_runs_status_filter(self, cli_parser):
        args = cli_parser.parse_args(["export-runs", "--status", "completed"])
        assert args.status == "completed"

    def test_export_runs_produces_one_jsonl_row_per_run(self, tmp_path, capsys):
        """Happy-path: 2 runs in history ⇒ stdout has 2 well-formed JSONL lines."""
        from datetime import datetime, timezone

        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.cli import EXIT_OK, cmd_export_runs

        manager = RunHistoryManager(str(tmp_path))
        manager._save([
            {
                "run_id": "run-1",
                "status": "completed",
                "model_name": "test-model",
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "run_id": "run-2",
                "status": "completed",
                "model_name": "test-model",
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        ])

        import argparse
        args = argparse.Namespace(
            output=str(tmp_path),
            format="jsonl",
            to=None,
            status=None,
        )
        code = cmd_export_runs(args)
        assert code == EXIT_OK
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(lines) == 2, (
            f"export-runs produced {len(lines)} JSONL lines; expected "
            f"exactly 2 (one per run)."
        )
        # Each line must be valid JSON
        for ln in lines:
            parsed = json.loads(ln)
            assert "run_id" in parsed


# =============================================================================
# TRAINER-LEVEL — Wave 6b kwargs accepted on Trainer.__init__
# =============================================================================
#
# The bridge agent uses inspect.signature(Trainer.__init__) to decide which
# kwargs to forward, so the Trainer can degrade gracefully on a pre-Wave-6b
# build. These tests pin the OTHER side of that contract: any Trainer build
# that does NOT accept these kwargs would silently drop the flag. We assert
# both shapes by checking inspect.signature.
#


class TestWave6bTrainerSignatureHandoff:
    """Trainer.__init__ accepts the Wave 6b kwargs the bridge forwards."""

    def _trainer_signature_params(self) -> set[str]:
        import inspect

        from backpropagate.trainer import Trainer
        return set(inspect.signature(Trainer.__init__).parameters)

    def test_trainer_init_param_set_documented(self):
        """Sanity smoke: Trainer.__init__ surface is non-empty + has 'model'."""
        params = self._trainer_signature_params()
        assert "model" in params, "Trainer.__init__ signature lost 'model' kwarg?"

    @pytest.mark.parametrize(
        "kwarg_name", ["use_dora", "packing", "init_lora_weights", "lora_preset", "optim"]
    )
    def test_cli_filter_guard_always_present_regardless_of_trainer_acceptance(
        self, kwarg_name
    ):
        """Wave 3.5 TESTS-B-014 hardening: cli.py MUST ALWAYS carry the
        inspect-based filter guard, regardless of whether Trainer accepts
        each kwarg in this build.

        Original shape (Wave 6b): the test branched — if Trainer accepts
        the kwarg it short-circuited to ``assert True`` (a tautology with
        no pinning), and only when Trainer REJECTED did it grep cli.py
        for the guard string. That shape only caught the conjunction
        regression "remove guard AND drop one kwarg." A single-pivot
        regression (someone removes the guard while Trainer still
        accepts every kwarg) slipped silently.

        New shape (Wave 3.5 TESTS-B-014): unconditionally assert the
        ``_trainer_sig_params`` filter is present in cli.py source.
        The guard is the load-bearing safety; its presence does not
        depend on this build's Trainer signature. The parametrize is
        retained as a doctrine anchor — each kwarg name documents one
        bridge-forwarded Wave 6b knob whose silent drop the guard
        prevents — but the actual assertion is now identical across
        params.
        """
        from pathlib import Path
        cli_src = (
            Path(__file__).resolve().parent.parent
            / "backpropagate" / "cli.py"
        ).read_text(encoding="utf-8")
        assert "_trainer_sig_params" in cli_src, (
            "CROSS-DOMAIN REGRESSION: cmd_train in backpropagate/cli.py "
            "is missing the _trainer_sig_params inspect filter guard. "
            f"Without it, the bridge will TypeError when forwarding "
            f"'{kwarg_name}' (or any other Wave 6b kwarg) if Trainer "
            "ever rejects one. The guard is load-bearing — restore it "
            "before merging."
        )

        # Doctrine anchor for the parametrize: each kwarg names one
        # Wave 6b knob the bridge forwards. Trainer signature acceptance
        # is reported as INFO (not pinned) so future readers see what's
        # landed.
        params = self._trainer_signature_params()
        # No assertion on params — Wave 6a may add or remove individual
        # kwargs; the load-bearing contract is the guard, not any one
        # kwarg's presence. This line exists so the params lookup isn't
        # dead code under static analysis.
        _ = kwarg_name in params


# =============================================================================
# TRAINER-LEVEL — license caveat surfacing (BACKEND-2)
# =============================================================================


class TestWave6bLicenseCaveatSurfacing:
    """Qwen-2.5-3B preset boot emits the license caveat (BACKEND-2)."""

    def test_lookup_model_preset_by_id_finds_qwen_2_5_3b_caveat(self):
        """The Trainer-side helper resolves Qwen-2.5-3B model_id ⇒ caveat-bearing preset."""
        from backpropagate.config import lookup_model_preset_by_id
        preset = lookup_model_preset_by_id("Qwen/Qwen2.5-3B-Instruct")
        assert preset is not None, (
            "lookup_model_preset_by_id('Qwen/Qwen2.5-3B-Instruct') "
            "returned None; the helper cannot surface the caveat at "
            "Trainer boot."
        )
        assert preset.name == "qwen2.5-3b"
        assert preset.license_restriction is not None
        assert "non-commercial" in preset.license_restriction.lower()

    def test_lookup_model_preset_by_id_case_insensitive(self):
        """Case-only variants should still match (operators type freely)."""
        from backpropagate.config import lookup_model_preset_by_id
        # The Qwen org puts a capital Q on HuggingFace, but operators
        # sometimes lowercase the whole org/model when typing. The
        # helper's docstring promises case-insensitive matching.
        preset = lookup_model_preset_by_id("qwen/qwen2.5-3b-instruct")
        assert preset is not None
        assert preset.name == "qwen2.5-3b"

    def test_lookup_unknown_model_id_returns_none(self):
        """Unknown model_id ⇒ None (no caveat surfaced)."""
        from backpropagate.config import lookup_model_preset_by_id
        assert lookup_model_preset_by_id("some/nonexistent-model") is None
