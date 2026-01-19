"""
Backpropagate - Multi-Run Training
===================================

Multi-run training orchestrator using SLAO (Single LoRA via Asymmetric Merging)
for continual learning without catastrophic forgetting.

Multi-Run = Short training bursts on fresh data chunks with intelligent
LoRA merging between runs.

Features:
- Configurable runs/steps/samples per run
- Two modes: Simple continuation or SLAO merge
- GPU safety monitoring throughout
- Automatic checkpointing after each run
- Aggregate loss history with run boundaries
- Decaying learning rate across runs

Research basis:
- SLAO: https://arxiv.org/abs/2512.23017
- K-Merge: https://arxiv.org/abs/2510.13537
- Forgetting Scaling Laws: https://arxiv.org/abs/2401.05605

Usage:
    from backpropagate.multi_run import MultiRunTrainer

    runner = MultiRunTrainer(
        model="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        num_runs=5,
        steps_per_run=100,
        samples_per_run=1000,
    )

    results = runner.run()
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union, Tuple
from enum import Enum

from .config import settings
from .slao import SLAOMerger, SLAOConfig, MergeResult
from .checkpoints import CheckpointManager, CheckpointPolicy, CheckpointInfo, CheckpointStats
from .gpu_safety import (
    GPUMonitor,
    GPUSafetyConfig,
    GPUStatus,
    GPUCondition,
    check_gpu_safe,
    wait_for_safe_gpu,
    get_gpu_status,
    format_gpu_status,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MultiRunTrainer",
    "MultiRunConfig",
    "MultiRunResult",
    "RunResult",
    "MergeMode",
    # Backwards compatibility aliases
    "SpeedrunTrainer",
    "SpeedrunConfig",
    "SpeedrunResult",
]


class MergeMode(Enum):
    """LoRA merge mode between runs."""
    SIMPLE = "simple"   # Load previous, continue training
    SLAO = "slao"       # SLAO asymmetric merge with orthogonal init


@dataclass
class MultiRunConfig:
    """Configuration for multi-run training."""

    # Run configuration
    num_runs: int = 5
    steps_per_run: int = 100
    samples_per_run: int = 1000

    # Merge strategy
    merge_mode: MergeMode = MergeMode.SLAO

    # Learning rate strategy
    initial_lr: float = 2e-4
    final_lr: float = 5e-5
    lr_decay: str = "linear"  # "linear", "cosine", "constant"

    # Warmup (reset each run)
    warmup_steps_per_run: int = 10

    # Data handling
    shuffle_data: bool = True
    replay_fraction: float = 0.0  # Fraction of previous samples to replay (0.0-0.5)
    replay_strategy: str = "recent"  # "recent", "random", "all_previous"

    # Checkpointing
    save_every_run: bool = True
    checkpoint_dir: str = "./output/multi_run"

    # GPU safety
    enable_gpu_monitoring: bool = True
    pause_on_overheat: bool = True
    max_temp_c: float = 85.0  # Pause if exceeded
    cooldown_seconds: float = 60.0

    # Validation
    validation_samples: int = 100
    validate_every_run: bool = True

    # Phase 4.3: Early stopping per run
    early_stopping: bool = False
    early_stopping_patience: int = 2  # Stop if val loss increases for N consecutive runs
    early_stopping_threshold: float = 0.0  # Min improvement required (0.0 = any increase is bad)

    # Phase 5.3: Checkpoint management
    checkpoint_keep_best_n: int = 3  # Keep N best checkpoints by validation loss
    checkpoint_keep_final: bool = True  # Always keep the last checkpoint
    checkpoint_keep_run_boundaries: bool = False  # Keep first checkpoint of each run
    checkpoint_max_total: int = 10  # Hard limit (0 = unlimited)
    checkpoint_auto_prune: bool = True  # Automatically prune after each run (set False to keep all)


# Backwards compatibility alias
SpeedrunConfig = MultiRunConfig


@dataclass
class RunResult:
    """Result of a single training run."""
    run_index: int
    steps: int
    samples: int
    final_loss: float
    loss_history: List[float] = field(default_factory=list)
    learning_rate: float = 0.0
    duration_seconds: float = 0.0
    checkpoint_path: Optional[str] = None
    merge_result: Optional[MergeResult] = None
    validation_loss: Optional[float] = None
    gpu_max_temp: Optional[float] = None
    gpu_max_vram_percent: Optional[float] = None


@dataclass
class MultiRunResult:
    """Aggregate result of all multi-run training."""
    total_runs: int
    total_steps: int
    total_samples: int
    total_duration_seconds: float
    final_loss: float
    runs: List[RunResult] = field(default_factory=list)
    aggregate_loss_history: List[float] = field(default_factory=list)
    run_boundaries: List[int] = field(default_factory=list)  # Step indices where runs start
    final_checkpoint_path: Optional[str] = None
    merge_mode: str = "slao"
    aborted: bool = False
    abort_reason: Optional[str] = None
    # Phase 5.3: Checkpoint stats
    checkpoint_stats: Optional[CheckpointStats] = None


# Backwards compatibility alias
SpeedrunResult = MultiRunResult


class MultiRunTrainer:
    """
    Multi-run trainer for continual learning with SLAO.

    Orchestrates multiple short training runs with intelligent LoRA merging
    to maximize learning while preventing catastrophic forgetting.
    """

    def __init__(
        self,
        model: str = None,
        config: Optional[MultiRunConfig] = None,
        # Convenience overrides
        num_runs: int = None,
        steps_per_run: int = None,
        samples_per_run: int = None,
        merge_mode: Union[str, MergeMode] = None,
        checkpoint_dir: str = None,
        # Callbacks
        on_run_start: Optional[Callable[[int], None]] = None,
        on_run_complete: Optional[Callable[[RunResult], None]] = None,
        on_step: Optional[Callable[[int, int, float], None]] = None,
        on_gpu_status: Optional[Callable[[GPUStatus], None]] = None,
    ):
        """
        Initialize multi-run trainer.

        Args:
            model: Model name/path (HuggingFace or local)
            config: Full MultiRunConfig (or use convenience args)
            num_runs: Number of training runs
            steps_per_run: Steps per run
            samples_per_run: Fresh samples per run
            merge_mode: "simple" or "slao"
            checkpoint_dir: Where to save checkpoints
            on_run_start: Callback when run starts
            on_run_complete: Callback when run completes
            on_step: Callback on each step (run_idx, step, loss)
            on_gpu_status: Callback for GPU status updates
        """
        # Build config
        self.config = config or MultiRunConfig()

        if num_runs is not None:
            self.config.num_runs = num_runs
        if steps_per_run is not None:
            self.config.steps_per_run = steps_per_run
        if samples_per_run is not None:
            self.config.samples_per_run = samples_per_run
        if checkpoint_dir is not None:
            self.config.checkpoint_dir = checkpoint_dir
        if merge_mode is not None:
            if isinstance(merge_mode, str):
                self.config.merge_mode = MergeMode(merge_mode.lower())
            else:
                self.config.merge_mode = merge_mode

        self.model_name = model or settings.model.name

        # Callbacks
        self.on_run_start = on_run_start
        self.on_run_complete = on_run_complete
        self.on_step = on_step
        self.on_gpu_status = on_gpu_status

        # Internal state
        self._trainer = None
        self._slao_merger: Optional[SLAOMerger] = None
        self._gpu_monitor: Optional[GPUMonitor] = None
        self._is_running = False
        self._should_abort = False
        self._abort_reason: Optional[str] = None

        # Results
        self._runs: List[RunResult] = []
        self._aggregate_loss: List[float] = []
        self._run_boundaries: List[int] = []

        # GPU tracking
        self._gpu_max_temp = 0.0
        self._gpu_max_vram = 0.0

        # Phase 4.3: Early stopping tracking
        self._validation_losses: List[float] = []
        self._early_stop_counter = 0
        self._best_val_loss = float('inf')

        # Phase 5.3: Checkpoint manager
        self._checkpoint_manager: Optional[CheckpointManager] = None

        logger.info(f"SpeedrunTrainer initialized: {self.config.num_runs} runs x "
                    f"{self.config.steps_per_run} steps x {self.config.samples_per_run} samples")
        logger.info(f"Merge mode: {self.config.merge_mode.value}")

    def run(self, dataset: Union[str, Any] = None) -> SpeedrunResult:
        """
        Execute all speedrun training runs.

        Args:
            dataset: Dataset name/path or HuggingFace dataset object

        Returns:
            SpeedrunResult with all run results and aggregate metrics
        """
        from .trainer import Trainer

        start_time = time.time()
        self._is_running = True
        self._should_abort = False
        self._abort_reason = None

        # Setup checkpoint directory
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Phase 5.3: Initialize checkpoint manager
        checkpoint_policy = CheckpointPolicy(
            keep_best_n=self.config.checkpoint_keep_best_n,
            keep_final=self.config.checkpoint_keep_final,
            keep_run_boundaries=self.config.checkpoint_keep_run_boundaries,
            max_total=self.config.checkpoint_max_total,
            auto_prune=self.config.checkpoint_auto_prune,
        )
        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            policy=checkpoint_policy,
        )
        logger.info(f"Checkpoint manager initialized: keep_best={checkpoint_policy.keep_best_n}, "
                    f"max_total={checkpoint_policy.max_total}, auto_prune={checkpoint_policy.auto_prune}")

        # Initialize SLAO merger if using SLAO mode
        if self.config.merge_mode == MergeMode.SLAO:
            self._slao_merger = SLAOMerger(SLAOConfig(
                scaling_type="sqrt",
                use_orthogonal_init=True,
            ))

        # Pre-flight GPU check
        if not self._preflight_gpu_check():
            return self._create_abort_result("GPU safety check failed")

        # Start GPU monitoring
        if self.config.enable_gpu_monitoring:
            self._start_gpu_monitor()

        # Load dataset
        logger.info(f"Loading dataset...")
        full_dataset = self._load_full_dataset(dataset)
        total_samples = len(full_dataset)
        logger.info(f"Dataset loaded: {total_samples} total samples")

        # Calculate sample chunks
        samples_needed = self.config.num_runs * self.config.samples_per_run
        if samples_needed > total_samples:
            logger.warning(
                f"Requested {samples_needed} samples but only {total_samples} available. "
                f"Will cycle through dataset."
            )

        try:
            # Initialize trainer
            logger.info(f"Initializing trainer with {self.model_name}")
            self._trainer = Trainer(
                model=self.model_name,
                learning_rate=self.config.initial_lr,
            )
            self._trainer.load_model()

            # Execute runs
            for run_idx in range(1, self.config.num_runs + 1):
                if self._should_abort:
                    break

                # Phase 4.3: Use validation wrapper if validation or early stopping enabled
                if self.config.validate_every_run or self.config.early_stopping:
                    run_result, val_loss = self._execute_run_with_validation(
                        run_idx=run_idx,
                        full_dataset=full_dataset,
                        checkpoint_dir=checkpoint_dir,
                    )

                    # Phase 4.3: Check early stopping
                    if self.config.early_stopping and val_loss is not None:
                        if self._check_early_stopping(val_loss, run_idx):
                            self._abort_reason = "Early stopping triggered"
                            break
                else:
                    run_result = self._execute_run(
                        run_idx=run_idx,
                        full_dataset=full_dataset,
                        checkpoint_dir=checkpoint_dir,
                    )

                self._runs.append(run_result)

                # Callback
                if self.on_run_complete:
                    self.on_run_complete(run_result)

                # GPU cooldown check between runs
                if self.config.pause_on_overheat and not self._should_abort:
                    self._check_cooldown()

        except Exception as e:
            logger.error(f"Speedrun failed: {e}")
            self._abort_reason = str(e)
            raise

        finally:
            self._is_running = False
            if self._gpu_monitor:
                self._gpu_monitor.stop()

        # Create final result
        total_duration = time.time() - start_time
        return self._create_result(total_duration)

    def abort(self, reason: str = "User requested abort") -> None:
        """Request abort of current speedrun."""
        self._should_abort = True
        self._abort_reason = reason
        logger.warning(f"Speedrun abort requested: {reason}")

    def get_checkpoint_manager(self) -> Optional[CheckpointManager]:
        """Get the checkpoint manager for external access (e.g., UI)."""
        return self._checkpoint_manager

    def get_checkpoint_stats(self) -> Optional[CheckpointStats]:
        """Get current checkpoint statistics."""
        if self._checkpoint_manager:
            return self._checkpoint_manager.get_stats()
        return None

    def _execute_run(
        self,
        run_idx: int,
        full_dataset: Any,
        checkpoint_dir: Path,
    ) -> RunResult:
        """Execute a single training run."""
        import torch
        from trl import SFTTrainer, SFTConfig

        run_start = time.time()

        logger.info(f"\n{'='*60}")
        logger.info(f"SPEEDRUN {run_idx}/{self.config.num_runs}")
        logger.info(f"{'='*60}")

        # Callback
        if self.on_run_start:
            self.on_run_start(run_idx)

        # Record run boundary
        self._run_boundaries.append(len(self._aggregate_loss))

        # Get data chunk for this run
        chunk_dataset = self._get_data_chunk(full_dataset, run_idx)
        logger.info(f"Data chunk: {len(chunk_dataset)} samples")

        # Calculate learning rate for this run
        lr = self._get_learning_rate(run_idx)
        logger.info(f"Learning rate: {lr:.2e}")

        # Initialize/update LoRA weights
        if run_idx > 1:
            self._prepare_for_next_run(run_idx)

        # Pre-tokenize for Windows
        if os.name == "nt" and settings.windows.pre_tokenize:
            chunk_dataset = self._trainer._pre_tokenize(chunk_dataset)

        # Training arguments (TRL 0.24+ uses SFTConfig)
        training_args = SFTConfig(
            output_dir=str(checkpoint_dir / f"run_{run_idx:03d}"),
            per_device_train_batch_size=self._trainer.batch_size,
            gradient_accumulation_steps=self._trainer.gradient_accumulation,
            max_steps=self.config.steps_per_run,
            learning_rate=lr,
            warmup_steps=self.config.warmup_steps_per_run,
            optim=settings.training.optim,
            lr_scheduler_type="cosine",
            logging_steps=10,
            bf16=settings.training.bf16,
            fp16=settings.training.fp16,
            overwrite_output_dir=True,
            dataloader_num_workers=0 if os.name == "nt" else 4,
            report_to="none",
            seed=settings.training.seed + run_idx,  # Different seed each run
            # SFT-specific args (TRL 0.24+)
            max_length=self._trainer.max_seq_length,
            packing=settings.data.packing,
        )

        # Create trainer for this run (TRL 0.24+ uses processing_class)
        trainer = SFTTrainer(
            model=self._trainer._model,
            processing_class=self._trainer._tokenizer,
            train_dataset=chunk_dataset,
            args=training_args,
        )

        # Train
        logger.info(f"Training run {run_idx}...")
        result = trainer.train()

        # Extract loss history
        loss_history = []
        if hasattr(trainer, 'state') and trainer.state.log_history:
            loss_history = [
                log.get('loss', 0) for log in trainer.state.log_history
                if 'loss' in log
            ]

        # Add to aggregate
        self._aggregate_loss.extend(loss_history)

        final_loss = result.training_loss if hasattr(result, 'training_loss') else (
            loss_history[-1] if loss_history else 0.0
        )

        # Merge LoRA weights
        merge_result = None
        if self.config.merge_mode == MergeMode.SLAO and self._slao_merger:
            lora_state = self._get_lora_state_dict()
            merge_result = self._slao_merger.merge(lora_state, run_index=run_idx)

        # Save checkpoint
        checkpoint_path = None
        if self.config.save_every_run:
            checkpoint_path = str(checkpoint_dir / f"run_{run_idx:03d}" / "lora")
            self._trainer.save(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Also save SLAO merger state
            if self._slao_merger:
                self._slao_merger.save(str(checkpoint_dir / f"run_{run_idx:03d}" / "slao"))

            # Phase 5.3: Register with checkpoint manager for smart pruning
            if self._checkpoint_manager:
                is_first_run = (run_idx == 1)
                self._checkpoint_manager.register(
                    run_index=run_idx,
                    checkpoint_path=checkpoint_path,
                    validation_loss=None,  # Will be updated if validation is enabled
                    training_loss=final_loss,
                    is_run_boundary=is_first_run,
                )
                # Note: auto_prune happens inside register() if enabled

        duration = time.time() - run_start

        run_result = RunResult(
            run_index=run_idx,
            steps=self.config.steps_per_run,
            samples=len(chunk_dataset),
            final_loss=final_loss,
            loss_history=loss_history,
            learning_rate=lr,
            duration_seconds=duration,
            checkpoint_path=checkpoint_path,
            merge_result=merge_result,
            gpu_max_temp=self._gpu_max_temp,
            gpu_max_vram_percent=self._gpu_max_vram,
        )

        # Reset GPU tracking for next run
        self._gpu_max_temp = 0.0
        self._gpu_max_vram = 0.0

        logger.info(f"Run {run_idx} complete: loss={final_loss:.4f}, time={duration:.1f}s")

        return run_result

    def _execute_run_with_validation(
        self,
        run_idx: int,
        full_dataset: Any,
        checkpoint_dir: Path,
    ) -> Tuple[RunResult, Optional[float]]:
        """
        Execute a single training run with optional validation.

        Phase 4.3: Wraps _execute_run to add validation loss computation.
        Phase 5.3: Updates checkpoint manager with validation loss.

        Returns:
            Tuple of (RunResult, validation_loss or None)
        """
        run_result = self._execute_run(run_idx, full_dataset, checkpoint_dir)

        # Compute validation loss if enabled
        val_loss = None
        if self.config.validate_every_run:
            val_loss = self._compute_validation_loss(full_dataset, run_idx)
            run_result.validation_loss = val_loss

            # Phase 5.3: Update checkpoint with validation loss for smarter pruning
            if self._checkpoint_manager:
                for cp in self._checkpoint_manager.list_checkpoints():
                    if cp.run_index == run_idx and cp.validation_loss is None:
                        cp.validation_loss = val_loss
                        self._checkpoint_manager._save_manifest()
                        break

        return run_result, val_loss

    def _get_data_chunk(self, full_dataset: Any, run_idx: int) -> Any:
        """
        Get data chunk for a specific run with optional experience replay.

        Experience replay mixes in samples from previous runs to prevent
        catastrophic forgetting. This is Phase 2.4 of the SLAO improvements.

        Args:
            full_dataset: Complete training dataset
            run_idx: Current run index (1-based)

        Returns:
            Dataset chunk with optional replay samples mixed in
        """
        from datasets import concatenate_datasets

        total_samples = len(full_dataset)
        chunk_size = self.config.samples_per_run

        # Calculate how many new vs replay samples
        replay_fraction = min(self.config.replay_fraction, 0.5)  # Cap at 50%
        if run_idx == 1 or replay_fraction <= 0:
            # First run or no replay - just get new samples
            new_samples_count = chunk_size
            replay_count = 0
        else:
            replay_count = int(chunk_size * replay_fraction)
            new_samples_count = chunk_size - replay_count

        # Get new samples for this run
        start_idx = ((run_idx - 1) * self.config.samples_per_run) % total_samples
        end_idx = start_idx + new_samples_count

        # Handle wrap-around for new samples
        if end_idx > total_samples:
            new_indices = list(range(start_idx, total_samples)) + list(range(0, end_idx - total_samples))
        else:
            new_indices = list(range(start_idx, end_idx))

        new_chunk = full_dataset.select(new_indices)

        # Add replay samples if configured
        if replay_count > 0 and run_idx > 1:
            replay_chunk = self._get_replay_samples(full_dataset, run_idx, replay_count)
            if replay_chunk is not None and len(replay_chunk) > 0:
                chunk = concatenate_datasets([new_chunk, replay_chunk])
                logger.info(f"Data chunk: {len(new_chunk)} new + {len(replay_chunk)} replay = {len(chunk)} total")
            else:
                chunk = new_chunk
        else:
            chunk = new_chunk

        # Shuffle if configured (mixes new and replay samples)
        if self.config.shuffle_data:
            chunk = chunk.shuffle(seed=settings.training.seed + run_idx)

        return chunk

    def _get_replay_samples(self, full_dataset: Any, run_idx: int, count: int) -> Any:
        """
        Get replay samples from previous runs.

        Args:
            full_dataset: Complete training dataset
            run_idx: Current run index
            count: Number of replay samples to get

        Returns:
            Dataset of replay samples
        """
        import random

        total_samples = len(full_dataset)
        samples_per_run = self.config.samples_per_run

        # Calculate indices from previous runs
        if self.config.replay_strategy == "recent":
            # Get samples from the most recent previous run
            prev_run = run_idx - 1
            prev_start = ((prev_run - 1) * samples_per_run) % total_samples
            prev_end = min(prev_start + samples_per_run, total_samples)
            available_indices = list(range(prev_start, prev_end))

        elif self.config.replay_strategy == "random":
            # Random samples from all previous runs
            all_prev_indices = []
            for prev_run in range(1, run_idx):
                prev_start = ((prev_run - 1) * samples_per_run) % total_samples
                prev_end = min(prev_start + samples_per_run, total_samples)
                all_prev_indices.extend(range(prev_start, prev_end))
            available_indices = list(set(all_prev_indices))  # Remove duplicates

        elif self.config.replay_strategy == "all_previous":
            # Uniform sample from all data seen so far
            total_seen = min((run_idx - 1) * samples_per_run, total_samples)
            available_indices = list(range(total_seen))

        else:
            # Default to recent
            prev_run = run_idx - 1
            prev_start = ((prev_run - 1) * samples_per_run) % total_samples
            prev_end = min(prev_start + samples_per_run, total_samples)
            available_indices = list(range(prev_start, prev_end))

        # Sample from available indices
        if len(available_indices) == 0:
            return None

        random.seed(settings.training.seed + run_idx + 1000)  # Different seed for replay
        replay_indices = random.sample(available_indices, min(count, len(available_indices)))

        return full_dataset.select(replay_indices)

    def _get_learning_rate(self, run_idx: int) -> float:
        """Calculate learning rate for this run (with optional decay)."""
        if self.config.lr_decay == "constant":
            return self.config.initial_lr

        # Calculate progress (0 to 1)
        progress = (run_idx - 1) / max(self.config.num_runs - 1, 1)

        if self.config.lr_decay == "linear":
            # Linear interpolation
            return self.config.initial_lr + progress * (self.config.final_lr - self.config.initial_lr)

        elif self.config.lr_decay == "cosine":
            # Cosine annealing
            import math
            return self.config.final_lr + 0.5 * (self.config.initial_lr - self.config.final_lr) * (
                1 + math.cos(math.pi * progress)
            )

        return self.config.initial_lr

    def _prepare_for_next_run(self, run_idx: int) -> None:
        """Prepare model for next run (load weights if SLAO, or just continue)."""
        if self.config.merge_mode == MergeMode.SLAO and self._slao_merger:
            # Get initialization weights from SLAO merger
            init_weights = self._slao_merger.get_init_weights()
            if init_weights:
                self._load_lora_state_dict(init_weights)
                logger.info(f"Loaded SLAO-initialized weights for run {run_idx}")
        # For SIMPLE mode, just continue with current weights

    def _get_lora_state_dict(self) -> Dict[str, Any]:
        """Extract LoRA adapter state dict from model."""
        import torch

        model = self._trainer._model

        # Try to get PEFT adapter state
        if hasattr(model, 'get_adapter_state_dict'):
            return model.get_adapter_state_dict()

        # Fallback: extract lora parameters manually
        lora_state = {}
        for name, param in model.named_parameters():
            if 'lora_' in name.lower():
                lora_state[name] = param.data.clone()

        return lora_state

    def _load_lora_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load LoRA state dict into model."""
        import torch

        model = self._trainer._model

        # Try PEFT method first
        if hasattr(model, 'load_adapter_state_dict'):
            model.load_adapter_state_dict(state_dict)
            return

        # Fallback: manual loading
        model_state = model.state_dict()
        for name, param in state_dict.items():
            if name in model_state:
                model_state[name].copy_(param)

    def _load_full_dataset(self, dataset: Union[str, Any]) -> Any:
        """Load the full dataset for chunking."""
        from datasets import load_dataset, Dataset

        if dataset is None:
            dataset = settings.data.dataset_name

        if isinstance(dataset, str):
            if dataset.endswith('.jsonl') or dataset.endswith('.json'):
                ds = load_dataset('json', data_files=dataset, split='train')
            elif dataset.endswith('.csv'):
                ds = load_dataset('csv', data_files=dataset, split='train')
            else:
                ds = load_dataset(dataset, split=settings.data.dataset_split)
        elif isinstance(dataset, Dataset):
            ds = dataset
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

        return ds

    def _preflight_gpu_check(self) -> bool:
        """Pre-flight GPU safety check."""
        logger.info("Running pre-flight GPU check...")

        status = get_gpu_status()

        if not status.available:
            logger.error("No GPU available!")
            return False

        logger.info(f"GPU: {status.device_name}")
        logger.info(f"VRAM: {status.vram_total_gb:.1f} GB")

        if status.temperature_c:
            logger.info(f"Temperature: {status.temperature_c}C")

        if status.condition == GPUCondition.EMERGENCY:
            logger.error(f"GPU in EMERGENCY state: {status.condition_reason}")
            return False

        if status.condition == GPUCondition.CRITICAL:
            logger.warning(f"GPU in CRITICAL state: {status.condition_reason}")
            logger.info("Waiting for GPU to cool down...")
            return wait_for_safe_gpu(max_wait_seconds=self.config.cooldown_seconds)

        logger.info(f"GPU status: {status.condition.value}")
        return True

    def _start_gpu_monitor(self) -> None:
        """Start background GPU monitoring."""
        safety_config = GPUSafetyConfig(
            temp_critical=self.config.max_temp_c,
            check_interval=10.0,
        )

        self._gpu_monitor = GPUMonitor(
            config=safety_config,
            on_critical=self._on_gpu_critical,
            on_emergency=self._on_gpu_emergency,
            on_status=self._on_gpu_status,
        )

        self._gpu_monitor.start()

    def _on_gpu_status(self, status: GPUStatus) -> None:
        """Handle GPU status update."""
        # Track max values
        if status.temperature_c and status.temperature_c > self._gpu_max_temp:
            self._gpu_max_temp = status.temperature_c

        if status.vram_percent > self._gpu_max_vram:
            self._gpu_max_vram = status.vram_percent

        # Callback
        if self.on_gpu_status:
            self.on_gpu_status(status)

    def _on_gpu_critical(self, status: GPUStatus) -> None:
        """Handle critical GPU condition."""
        logger.warning(f"GPU CRITICAL: {status.condition_reason}")

        if self.config.pause_on_overheat:
            logger.info("Pausing for cooldown...")
            # Note: actual pause would need deeper integration with training loop

    def _on_gpu_emergency(self, status: GPUStatus) -> None:
        """Handle emergency GPU condition."""
        logger.error(f"GPU EMERGENCY: {status.condition_reason}")
        self.abort(f"GPU emergency: {status.condition_reason}")

    def _check_cooldown(self) -> None:
        """Check if cooldown is needed between runs."""
        status = get_gpu_status()

        if status.temperature_c and status.temperature_c > self.config.max_temp_c:
            logger.info(f"GPU at {status.temperature_c}C, cooling down...")
            wait_for_safe_gpu(
                max_wait_seconds=self.config.cooldown_seconds,
                check_interval=5.0,
            )

    def _compute_validation_loss(self, full_dataset: Any, run_idx: int) -> float:
        """
        Compute validation loss on held-out samples.

        Phase 4.3: Validation is used for early stopping decisions.

        Args:
            full_dataset: Complete training dataset
            run_idx: Current run index

        Returns:
            Validation loss (average)
        """
        import torch

        model = self._trainer._model
        tokenizer = self._trainer._tokenizer

        # Get validation samples (use samples from far in the future to avoid overlap)
        total_samples = len(full_dataset)
        val_start = (self.config.num_runs * self.config.samples_per_run) % total_samples
        val_end = min(val_start + self.config.validation_samples, total_samples)

        if val_end <= val_start:
            # Wrap around
            val_indices = list(range(val_start, total_samples)) + list(range(0, self.config.validation_samples - (total_samples - val_start)))
        else:
            val_indices = list(range(val_start, val_end))

        val_dataset = full_dataset.select(val_indices)

        # Compute loss on validation set
        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for sample in val_dataset:
                # Get text
                if 'text' in sample:
                    text = sample['text']
                elif 'messages' in sample:
                    text = tokenizer.apply_chat_template(sample['messages'], tokenize=False)
                elif 'conversations' in sample:
                    # ShareGPT format
                    text = '\n'.join([c.get('value', '') for c in sample['conversations']])
                else:
                    continue

                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self._trainer.max_seq_length,
                )

                # Move to device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Forward pass
                outputs = model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item()
                count += 1

                # Limit to prevent slow validation
                if count >= self.config.validation_samples:
                    break

        model.train()

        avg_loss = total_loss / max(count, 1)
        logger.info(f"Validation loss (run {run_idx}): {avg_loss:.4f}")
        return avg_loss

    def _check_early_stopping(self, val_loss: float, run_idx: int) -> bool:
        """
        Check if early stopping should be triggered.

        Phase 4.3: Stop training if validation loss increases for
        `early_stopping_patience` consecutive runs.

        Args:
            val_loss: Current validation loss
            run_idx: Current run index

        Returns:
            True if training should stop
        """
        self._validation_losses.append(val_loss)

        # Need at least 2 runs to compare
        if run_idx < 2:
            self._best_val_loss = min(self._best_val_loss, val_loss)
            return False

        # Check if loss improved
        improvement = self._best_val_loss - val_loss
        if improvement > self.config.early_stopping_threshold:
            # Improved - reset counter
            self._best_val_loss = val_loss
            self._early_stop_counter = 0
            logger.info(f"Validation improved by {improvement:.4f}, new best: {val_loss:.4f}")
            return False
        else:
            # No improvement - increment counter
            self._early_stop_counter += 1
            logger.info(f"No improvement ({self._early_stop_counter}/{self.config.early_stopping_patience})")

            if self._early_stop_counter >= self.config.early_stopping_patience:
                logger.warning(f"Early stopping triggered after {run_idx} runs")
                return True

        return False

    def _create_result(self, total_duration: float) -> SpeedrunResult:
        """Create final SpeedrunResult."""
        total_steps = sum(r.steps for r in self._runs)
        total_samples = sum(r.samples for r in self._runs)
        final_loss = self._runs[-1].final_loss if self._runs else 0.0

        # Final checkpoint path
        final_checkpoint = None
        if self._runs and self._runs[-1].checkpoint_path:
            final_checkpoint = self._runs[-1].checkpoint_path

        # Phase 5.3: Get checkpoint stats
        checkpoint_stats = None
        if self._checkpoint_manager:
            checkpoint_stats = self._checkpoint_manager.get_stats()
            logger.info(f"Checkpoint stats: {checkpoint_stats.summary()}")

        return SpeedrunResult(
            total_runs=len(self._runs),
            total_steps=total_steps,
            total_samples=total_samples,
            total_duration_seconds=total_duration,
            final_loss=final_loss,
            runs=self._runs,
            aggregate_loss_history=self._aggregate_loss,
            run_boundaries=self._run_boundaries,
            final_checkpoint_path=final_checkpoint,
            merge_mode=self.config.merge_mode.value,
            aborted=self._should_abort,
            abort_reason=self._abort_reason,
            checkpoint_stats=checkpoint_stats,
        )

    def _create_abort_result(self, reason: str) -> SpeedrunResult:
        """Create result for aborted run."""
        return SpeedrunResult(
            total_runs=0,
            total_steps=0,
            total_samples=0,
            total_duration_seconds=0.0,
            final_loss=0.0,
            aborted=True,
            abort_reason=reason,
        )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for speedrun training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SLAO Speedruns - Multi-run LLM fine-tuning"
    )

    parser.add_argument("--model", type=str, help="Model name/path")
    parser.add_argument("--dataset", type=str, help="Dataset name/path")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--steps", type=int, default=100, help="Steps per run")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per run")
    parser.add_argument("--mode", choices=["simple", "slao"], default="slao",
                        help="Merge mode")
    parser.add_argument("--output", type=str, default="./output/speedruns",
                        help="Checkpoint directory")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--lr-final", type=float, default=5e-5, help="Final learning rate")

    args = parser.parse_args()

    config = SpeedrunConfig(
        num_runs=args.runs,
        steps_per_run=args.steps,
        samples_per_run=args.samples,
        merge_mode=MergeMode(args.mode),
        initial_lr=args.lr,
        final_lr=args.lr_final,
        checkpoint_dir=args.output,
    )

    trainer = SpeedrunTrainer(
        model=args.model,
        config=config,
    )

    result = trainer.run(args.dataset)

    print(f"\n{'='*60}")
    print("SPEEDRUN COMPLETE")
    print(f"{'='*60}")
    print(f"Runs: {result.total_runs}")
    print(f"Total steps: {result.total_steps}")
    print(f"Total samples: {result.total_samples}")
    print(f"Final loss: {result.final_loss:.4f}")
    print(f"Duration: {result.total_duration_seconds/60:.1f} minutes")
    print(f"Checkpoint: {result.final_checkpoint_path}")


# Backwards compatibility alias
SpeedrunTrainer = MultiRunTrainer


if __name__ == "__main__":
    main()
