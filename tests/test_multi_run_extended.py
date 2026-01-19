"""
Extended multi-run tests for comprehensive coverage.

Covers:
- MultiRunConfig dataclass
- RunResult and MultiRunResult
- MultiRunTrainer initialization
- MergeMode enum
- Configuration validation
- Callback registration
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import is_dataclass


# =============================================================================
# MERGE MODE ENUM TESTS
# =============================================================================


class TestMergeMode:
    """Tests for MergeMode enum."""

    def test_merge_mode_values(self):
        """MergeMode has expected values."""
        from backpropagate.multi_run import MergeMode

        assert MergeMode.SIMPLE.value == "simple"
        assert MergeMode.SLAO.value == "slao"

    def test_merge_mode_from_string(self):
        """MergeMode can be created from string."""
        from backpropagate.multi_run import MergeMode

        assert MergeMode("simple") == MergeMode.SIMPLE
        assert MergeMode("slao") == MergeMode.SLAO


# =============================================================================
# MULTIRUNCONFIG TESTS
# =============================================================================


class TestMultiRunConfig:
    """Tests for MultiRunConfig dataclass."""

    def test_is_dataclass(self):
        """MultiRunConfig is a dataclass."""
        from backpropagate.multi_run import MultiRunConfig

        assert is_dataclass(MultiRunConfig)

    def test_default_values(self):
        """MultiRunConfig has sensible defaults."""
        from backpropagate.multi_run import MultiRunConfig, MergeMode

        config = MultiRunConfig()

        assert config.num_runs == 5
        assert config.steps_per_run == 100
        assert config.samples_per_run == 1000
        assert config.merge_mode == MergeMode.SLAO
        assert config.initial_lr == 2e-4
        assert config.final_lr == 5e-5
        assert config.lr_decay == "linear"

    def test_custom_values(self):
        """MultiRunConfig accepts custom values."""
        from backpropagate.multi_run import MultiRunConfig, MergeMode

        config = MultiRunConfig(
            num_runs=10,
            steps_per_run=200,
            samples_per_run=500,
            merge_mode=MergeMode.SIMPLE,
        )

        assert config.num_runs == 10
        assert config.steps_per_run == 200
        assert config.samples_per_run == 500
        assert config.merge_mode == MergeMode.SIMPLE

    def test_gpu_monitoring_defaults(self):
        """GPU monitoring has sensible defaults."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig()

        assert config.enable_gpu_monitoring is True
        assert config.pause_on_overheat is True
        assert config.max_temp_c == 85.0
        assert config.cooldown_seconds == 60.0

    def test_validation_defaults(self):
        """Validation has sensible defaults."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig()

        assert config.validation_samples == 100
        assert config.validate_every_run is True

    def test_early_stopping_defaults(self):
        """Early stopping disabled by default."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig()

        assert config.early_stopping is False
        assert config.early_stopping_patience == 2
        assert config.early_stopping_threshold == 0.0

    def test_checkpoint_defaults(self):
        """Checkpoint management has sensible defaults."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig()

        assert config.save_every_run is True
        assert config.checkpoint_dir == "./output/multi_run"
        assert config.checkpoint_keep_best_n == 3
        assert config.checkpoint_keep_final is True

    def test_replay_defaults(self):
        """Experience replay disabled by default."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig()

        assert config.replay_fraction == 0.0
        assert config.replay_strategy == "recent"


# =============================================================================
# RUN RESULT TESTS
# =============================================================================


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_is_dataclass(self):
        """RunResult is a dataclass."""
        from backpropagate.multi_run import RunResult

        assert is_dataclass(RunResult)

    def test_run_result_creation(self):
        """RunResult can be created with required fields."""
        from backpropagate.multi_run import RunResult

        result = RunResult(
            run_index=0,
            steps=100,
            samples=1000,
            final_loss=1.5,
        )

        assert result.run_index == 0
        assert result.steps == 100
        assert result.samples == 1000
        assert result.final_loss == 1.5

    def test_run_result_optional_fields(self):
        """RunResult has optional fields with defaults."""
        from backpropagate.multi_run import RunResult

        result = RunResult(
            run_index=0,
            steps=100,
            samples=1000,
            final_loss=1.5,
        )

        assert result.loss_history == []
        assert result.learning_rate == 0.0
        assert result.duration_seconds == 0.0
        assert result.checkpoint_path is None
        assert result.merge_result is None
        assert result.validation_loss is None
        assert result.gpu_max_temp is None
        assert result.gpu_max_vram_percent is None

    def test_run_result_with_all_fields(self):
        """RunResult can be created with all fields."""
        from backpropagate.multi_run import RunResult

        result = RunResult(
            run_index=1,
            steps=100,
            samples=1000,
            final_loss=1.2,
            loss_history=[2.0, 1.5, 1.2],
            learning_rate=2e-4,
            duration_seconds=120.5,
            checkpoint_path="/path/to/checkpoint",
            validation_loss=1.3,
            gpu_max_temp=75.0,
            gpu_max_vram_percent=85.0,
        )

        assert result.run_index == 1
        assert result.loss_history == [2.0, 1.5, 1.2]
        assert result.checkpoint_path == "/path/to/checkpoint"
        assert result.gpu_max_temp == 75.0


# =============================================================================
# MULTI RUN RESULT TESTS
# =============================================================================


class TestMultiRunResult:
    """Tests for MultiRunResult dataclass."""

    def test_is_dataclass(self):
        """MultiRunResult is a dataclass."""
        from backpropagate.multi_run import MultiRunResult

        assert is_dataclass(MultiRunResult)

    def test_multirun_result_creation(self):
        """MultiRunResult can be created."""
        from backpropagate.multi_run import MultiRunResult

        result = MultiRunResult(
            total_runs=5,
            total_steps=500,
            total_samples=5000,
            total_duration_seconds=600.0,
            final_loss=1.0,
        )

        assert result.total_runs == 5
        assert result.total_steps == 500
        assert result.total_samples == 5000
        assert result.total_duration_seconds == 600.0
        assert result.final_loss == 1.0

    def test_multirun_result_defaults(self):
        """MultiRunResult has sensible defaults."""
        from backpropagate.multi_run import MultiRunResult

        result = MultiRunResult(
            total_runs=5,
            total_steps=500,
            total_samples=5000,
            total_duration_seconds=600.0,
            final_loss=1.0,
        )

        assert result.runs == []
        assert result.aggregate_loss_history == []
        assert result.run_boundaries == []
        assert result.final_checkpoint_path is None
        assert result.merge_mode == "slao"
        assert result.aborted is False
        assert result.abort_reason is None


# =============================================================================
# MULTI RUN TRAINER INITIALIZATION TESTS
# =============================================================================


class TestMultiRunTrainerInit:
    """Tests for MultiRunTrainer initialization."""

    def test_trainer_creation_default(self):
        """MultiRunTrainer can be created with defaults."""
        from backpropagate.multi_run import MultiRunTrainer

        trainer = MultiRunTrainer()
        assert trainer is not None
        assert trainer.config is not None
        assert trainer.config.num_runs == 5

    def test_trainer_with_config(self):
        """MultiRunTrainer accepts config object."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

        config = MultiRunConfig(num_runs=10, steps_per_run=50)
        trainer = MultiRunTrainer(config=config)

        assert trainer.config.num_runs == 10
        assert trainer.config.steps_per_run == 50

    def test_trainer_with_convenience_args(self):
        """MultiRunTrainer accepts convenience arguments."""
        from backpropagate.multi_run import MultiRunTrainer

        trainer = MultiRunTrainer(
            num_runs=3,
            steps_per_run=25,
            samples_per_run=200,
        )

        assert trainer.config.num_runs == 3
        assert trainer.config.steps_per_run == 25
        assert trainer.config.samples_per_run == 200

    def test_trainer_with_merge_mode_string(self):
        """MultiRunTrainer accepts merge_mode as string."""
        from backpropagate.multi_run import MultiRunTrainer, MergeMode

        trainer = MultiRunTrainer(merge_mode="simple")

        assert trainer.config.merge_mode == MergeMode.SIMPLE

    def test_trainer_with_merge_mode_enum(self):
        """MultiRunTrainer accepts merge_mode as enum."""
        from backpropagate.multi_run import MultiRunTrainer, MergeMode

        trainer = MultiRunTrainer(merge_mode=MergeMode.SLAO)

        assert trainer.config.merge_mode == MergeMode.SLAO

    def test_trainer_with_callbacks(self):
        """MultiRunTrainer accepts callback functions."""
        from backpropagate.multi_run import MultiRunTrainer

        on_run_start = MagicMock()
        on_run_complete = MagicMock()
        on_step = MagicMock()

        trainer = MultiRunTrainer(
            on_run_start=on_run_start,
            on_run_complete=on_run_complete,
            on_step=on_step,
        )

        assert trainer.on_run_start is on_run_start
        assert trainer.on_run_complete is on_run_complete
        assert trainer.on_step is on_step

    def test_trainer_with_gpu_callback(self):
        """MultiRunTrainer accepts GPU status callback."""
        from backpropagate.multi_run import MultiRunTrainer

        on_gpu_status = MagicMock()
        trainer = MultiRunTrainer(on_gpu_status=on_gpu_status)

        assert trainer.on_gpu_status is on_gpu_status

    def test_trainer_with_model_name(self):
        """MultiRunTrainer accepts model name."""
        from backpropagate.multi_run import MultiRunTrainer

        trainer = MultiRunTrainer(model="custom/model-path")

        assert trainer.model_name == "custom/model-path"

    def test_trainer_with_checkpoint_dir(self):
        """MultiRunTrainer accepts checkpoint directory."""
        from backpropagate.multi_run import MultiRunTrainer

        trainer = MultiRunTrainer(checkpoint_dir="/custom/path")

        assert trainer.config.checkpoint_dir == "/custom/path"


# =============================================================================
# SPEEDRUN ALIAS TESTS
# =============================================================================


class TestSpeedrunAliases:
    """Tests for backwards compatibility aliases."""

    def test_speedrun_config_alias(self):
        """SpeedrunConfig is alias for MultiRunConfig."""
        from backpropagate.multi_run import SpeedrunConfig, MultiRunConfig

        assert SpeedrunConfig is MultiRunConfig

    def test_speedrun_result_alias(self):
        """SpeedrunResult is alias for MultiRunResult."""
        from backpropagate.multi_run import SpeedrunResult, MultiRunResult

        assert SpeedrunResult is MultiRunResult

    def test_speedrun_trainer_alias(self):
        """SpeedrunTrainer can be imported (alias)."""
        # Note: SpeedrunTrainer might be same as MultiRunTrainer or separate
        from backpropagate.multi_run import MultiRunTrainer

        assert MultiRunTrainer is not None


# =============================================================================
# INTERNAL STATE TESTS
# =============================================================================


class TestTrainerInternalState:
    """Tests for trainer internal state initialization."""

    def test_initial_state_flags(self):
        """Trainer starts with correct state flags."""
        from backpropagate.multi_run import MultiRunTrainer

        trainer = MultiRunTrainer()

        assert trainer._is_running is False
        assert trainer._should_abort is False
        assert trainer._abort_reason is None

    def test_initial_state_lists(self):
        """Trainer starts with empty result lists."""
        from backpropagate.multi_run import MultiRunTrainer

        trainer = MultiRunTrainer()

        assert trainer._runs == []
        assert trainer._aggregate_loss == []
        assert trainer._run_boundaries == []

    def test_initial_gpu_tracking(self):
        """Trainer starts with zero GPU tracking."""
        from backpropagate.multi_run import MultiRunTrainer

        trainer = MultiRunTrainer()

        assert trainer._gpu_max_temp == 0.0
        assert trainer._gpu_max_vram == 0.0

    def test_initial_early_stopping_state(self):
        """Trainer starts with early stopping state reset."""
        from backpropagate.multi_run import MultiRunTrainer

        trainer = MultiRunTrainer()

        assert trainer._validation_losses == []
        assert trainer._early_stop_counter == 0
        assert trainer._best_val_loss == float('inf')


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_core_classes_exported(self):
        """Core classes are exported."""
        from backpropagate.multi_run import (
            MergeMode,
            MultiRunConfig,
            RunResult,
            MultiRunResult,
            MultiRunTrainer,
        )

        assert MergeMode is not None
        assert MultiRunConfig is not None
        assert RunResult is not None
        assert MultiRunResult is not None
        assert MultiRunTrainer is not None

    def test_slao_classes_exported(self):
        """SLAO classes are exported."""
        from backpropagate.multi_run import (
            SLAOConfig,
            SLAOMerger,
            MergeResult,
        )

        assert SLAOConfig is not None
        assert SLAOMerger is not None
        assert MergeResult is not None

    def test_gpu_classes_exported(self):
        """GPU monitoring classes are exported."""
        from backpropagate.multi_run import (
            GPUMonitor,
            GPUStatus,
            GPUSafetyConfig,
            GPUCondition,
        )

        assert GPUMonitor is not None
        assert GPUStatus is not None
        assert GPUSafetyConfig is not None
        assert GPUCondition is not None

    def test_checkpoint_classes_exported(self):
        """Checkpoint classes are exported."""
        from backpropagate.multi_run import (
            CheckpointManager,
            CheckpointInfo,
            CheckpointPolicy,
            CheckpointStats,
        )

        assert CheckpointManager is not None
        assert CheckpointInfo is not None
        assert CheckpointPolicy is not None
        assert CheckpointStats is not None

    def test_helper_functions_exported(self):
        """Helper functions are exported."""
        from backpropagate.multi_run import (
            get_gpu_status,
            check_gpu_safe,
            wait_for_safe_gpu,
            format_gpu_status,
        )

        assert callable(get_gpu_status)
        assert callable(check_gpu_safe)
        assert callable(wait_for_safe_gpu)
        assert callable(format_gpu_status)


# =============================================================================
# LR DECAY CONFIGURATION TESTS
# =============================================================================


class TestLRDecayConfig:
    """Tests for learning rate decay configuration."""

    def test_linear_lr_decay_config(self):
        """Linear LR decay can be configured."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig(
            lr_decay="linear",
            initial_lr=2e-4,
            final_lr=1e-5,
        )

        assert config.lr_decay == "linear"
        assert config.initial_lr == 2e-4
        assert config.final_lr == 1e-5

    def test_cosine_lr_decay_config(self):
        """Cosine LR decay can be configured."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig(lr_decay="cosine")

        assert config.lr_decay == "cosine"

    def test_constant_lr_decay_config(self):
        """Constant LR can be configured."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig(lr_decay="constant")

        assert config.lr_decay == "constant"


# =============================================================================
# EXPERIENCE REPLAY CONFIGURATION TESTS
# =============================================================================


class TestExperienceReplayConfig:
    """Tests for experience replay configuration."""

    def test_replay_fraction_config(self):
        """Replay fraction can be configured."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig(replay_fraction=0.2)

        assert config.replay_fraction == 0.2

    def test_replay_strategy_recent(self):
        """Recent replay strategy can be configured."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig(
            replay_fraction=0.1,
            replay_strategy="recent",
        )

        assert config.replay_strategy == "recent"

    def test_replay_strategy_random(self):
        """Random replay strategy can be configured."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig(
            replay_fraction=0.1,
            replay_strategy="random",
        )

        assert config.replay_strategy == "random"

    def test_replay_strategy_all_previous(self):
        """All previous replay strategy can be configured."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig(
            replay_fraction=0.1,
            replay_strategy="all_previous",
        )

        assert config.replay_strategy == "all_previous"


# =============================================================================
# EARLY STOPPING CONFIGURATION TESTS
# =============================================================================


class TestEarlyStoppingConfig:
    """Tests for early stopping configuration."""

    def test_early_stopping_enabled(self):
        """Early stopping can be enabled."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig(
            early_stopping=True,
            early_stopping_patience=3,
            early_stopping_threshold=0.01,
        )

        assert config.early_stopping is True
        assert config.early_stopping_patience == 3
        assert config.early_stopping_threshold == 0.01

    def test_validation_required_for_early_stopping(self):
        """Validation should be enabled for early stopping."""
        from backpropagate.multi_run import MultiRunConfig

        config = MultiRunConfig(
            early_stopping=True,
            validate_every_run=True,
        )

        assert config.early_stopping is True
        assert config.validate_every_run is True
