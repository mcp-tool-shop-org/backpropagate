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

import gc
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .checkpoints import (
    CheckpointInfo,
    CheckpointManager,
    CheckpointPolicy,
    CheckpointStats,
    RunHistoryManager,
)
from .config import settings
from .datasets import DatasetLoader
from .exceptions import (
    BackpropagateError,
    ConfigurationError,
    DatasetError,
    DatasetNotFoundError,
    TrainingError,
)
from .gpu_safety import (
    GPUCondition,
    GPUMonitor,
    GPUSafetyConfig,
    GPUStatus,
    check_gpu_safe,
    format_gpu_status,
    get_gpu_status,
    wait_for_safe_gpu,
)
from .logging_config import bind_run_context, unbind_run_context
from .slao import MergeResult, SLAOConfig, SLAOMerger

logger = logging.getLogger(__name__)

__all__ = [
    "MultiRunTrainer",
    "MultiRunConfig",
    "MultiRunResult",
    "RunResult",
    "MergeMode",
    # Re-exported from submodules
    "CheckpointInfo",
    "CheckpointManager",
    "CheckpointPolicy",
    "CheckpointStats",
    "TrainingError",
    "check_gpu_safe",
    "format_gpu_status",
    "get_gpu_status",
    "wait_for_safe_gpu",
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

    # F-015: SLAO merge advanced scaling (data contract; fields land in v1.1.0,
    # the SLAOMerger implementation backing them lands in v1.2). When either
    # flag is True we thread it through to SLAOConfig so the corresponding
    # use_adaptive_scaling / use_layer_scaling slot on the merger is
    # populated; merge_history.json then records the operator's intent even
    # if the underlying scaling logic is a no-op in the merge mechanic.
    adaptive_scaling: bool = False  # If True, SLAO merger scales adapter contribution by recent loss trend
    layer_scaling: bool = False     # If True, per-layer scaling factor learned from train/val curves


# Backwards compatibility alias
SpeedrunConfig = MultiRunConfig


@dataclass
class RunResult:
    """Result of a single training run."""
    run_index: int
    steps: int
    samples: int
    final_loss: float
    loss_history: list[float] = field(default_factory=list)
    learning_rate: float = 0.0
    duration_seconds: float = 0.0
    checkpoint_path: str | None = None
    merge_result: MergeResult | None = None
    validation_loss: float | None = None
    gpu_max_temp: float | None = None
    gpu_max_vram_percent: float | None = None
    failed: bool = False
    failure_reason: str | None = None
    # B-001: correlation token (UUID4-derived). Shared with the session-level
    # run_id so per-run records can be grouped back to the parent multi-run.
    run_id: str | None = None
    # B-002: number of OOM-driven retries the run survived (0 on the happy
    # path). When >0, batch_size / gradient_accumulation differ from config.
    oom_retries: int = 0


@dataclass
class MultiRunResult:
    """Aggregate result of all multi-run training."""
    total_runs: int
    total_steps: int
    total_samples: int
    total_duration_seconds: float
    final_loss: float
    runs: list[RunResult] = field(default_factory=list)
    aggregate_loss_history: list[float] = field(default_factory=list)
    run_boundaries: list[int] = field(default_factory=list)  # Step indices where runs start
    final_checkpoint_path: str | None = None
    merge_mode: str = "slao"
    aborted: bool = False
    abort_reason: str | None = None
    # Phase 5.3: Checkpoint stats
    checkpoint_stats: CheckpointStats | None = None
    # B-001: stable session-wide correlation token. Persist this and grep
    # logs/checkpoints/run_history.json/merge_history.json by it.
    run_id: str | None = None


# Backwards compatibility alias
SpeedrunResult = MultiRunResult


class MultiRunTrainer:
    """
    Multi-run trainer for continual learning with SLAO.

    Orchestrates multiple short training runs with intelligent LoRA merging
    to maximize learning while preventing catastrophic forgetting.
    """

    # B-002: OOM-recovery tuning constants. Exposed at module scope so future
    # contributors don't have to grep the body for magic numbers.
    _OOM_MAX_RETRIES_AT_MIN_BATCH = 3  # consecutive OOMs at batch=1 before abort

    def __init__(
        self,
        model: str | None = None,
        config: MultiRunConfig | None = None,
        # Convenience overrides
        num_runs: int | None = None,
        steps_per_run: int | None = None,
        samples_per_run: int | None = None,
        merge_mode: str | MergeMode | None = None,
        checkpoint_dir: str | None = None,
        # B-002: opt out of OOM recovery (default ON — backward-compatible
        # because preserves prior observable surface for non-OOM runs).
        oom_recovery: bool = True,
        # F-005: experiment-tracker wiring forwarded to the inner Trainer.
        report_to: str | list[str] | None = "auto",
        # F-002: resume-from-checkpoint hint.
        #   * None (default) → auto-detect any in-progress run in the same
        #     output dir and resume it; if no in-progress run is found,
        #     start a fresh session.
        #   * <run_id> str → resume that specific run.
        #   * "off" → never resume; always start a fresh session.
        resume_from: str | None = None,
        # Callbacks
        on_run_start: Callable[[int], None] | None = None,
        on_run_complete: Callable[[RunResult], None] | None = None,
        on_step: Callable[[int, int, float], None] | None = None,
        on_gpu_status: Callable[[GPUStatus], None] | None = None,
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
            oom_recovery: When True (default), a CUDA OOM in any single run
                triggers batch_size halving + gradient_accumulation doubling
                (preserving effective batch size) and the run retries. After
                3 consecutive OOMs at batch=1 the session aborts with a
                structured error (code=RUNTIME_GPU_OOM). Set False to make
                OOMs hard-fail the run.
            on_run_start: Callback when run starts
            on_run_complete: Callback when run completes
            on_step: Callback on each step (run_idx, step, loss)
            on_gpu_status: Callback for GPU status updates

        Production features (Stage B / Stage C, May 2026):
            - run_id correlation token: minted once per run() call and
              attached to every log line, RunResult, MultiRunResult, SLAO
              merge_history entry, and CheckpointManager manifest record
              produced by the session. Use it to grep across logs and
              artifacts when triaging a divergent run.
            - PEFT API invariant check (B-005): SLAO mode fails loud at
              startup if the installed PEFT version cannot expose
              .lora_A. / .lora_B. parameters, preventing a silent no-op
              merger across runs.
            - Train/validation overlap guard: when validation is active,
              the last 10% of the dataset is reserved and training never
              draws from it even on wrap-around (see _get_data_chunk).
            - Atomic checkpoint writes (B-006): per-run save() goes
              through <path>.partial then shutil.move() into place.
            - The underlying Trainer carries the ``unsloth_fallback``
              knob (default True) — see Trainer.__init__. MultiRunTrainer
              does not expose it directly but inherits the behaviour via
              its internal Trainer instance.
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
                try:
                    self.config.merge_mode = MergeMode(merge_mode.lower())
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid merge mode: '{merge_mode}'. Available: slao, simple"
                    ) from None
            else:
                self.config.merge_mode = merge_mode

        self.model_name = model or settings.model.name

        # B-002: OOM recovery flag (default True for graceful degradation).
        self.oom_recovery = oom_recovery

        # F-005: persist the operator's report_to intent; threaded into the
        # inner Trainer instance at run() time.
        self._report_to_intent = report_to

        # F-002: resume hint. Resolved at run() entry by inspecting the
        # on-disk run history.
        self._resume_from = resume_from
        # Populated by _maybe_resume() once we know which run we're picking
        # up — used by _execute_run to skip already-completed run indices
        # and to load the SLAO merger from the persisted state.
        self._resume_start_run_idx: int = 1
        self._resume_checkpoint_path: str | None = None
        self._resumed_from_run_id: str | None = None

        # Callbacks
        self.on_run_start = on_run_start
        self.on_run_complete = on_run_complete
        self.on_step = on_step
        self.on_gpu_status = on_gpu_status

        # Internal state
        self._trainer: Any = None
        self._slao_merger: SLAOMerger | None = None
        self._gpu_monitor: GPUMonitor | None = None
        self._is_running = False
        self._should_abort = False
        self._abort_reason: str | None = None

        # B-001: session-wide correlation token. Minted lazily at run() entry
        # so a re-used MultiRunTrainer instance gets a fresh ID per session.
        self._run_id: str | None = None

        # F-003: RunHistoryManager bound at run() entry once the checkpoint
        # directory is known. Set here so other methods can defensively
        # check ``self._run_history is not None`` without an AttributeError.
        self._run_history: RunHistoryManager | None = None

        # Results
        self._runs: list[RunResult] = []
        self._aggregate_loss: list[float] = []
        self._run_boundaries: list[int] = []

        # GPU tracking
        self._gpu_max_temp = 0.0
        self._gpu_max_vram = 0.0

        # GPU pause event: set when GPU is overheating, cleared when safe
        self._gpu_pause_event = threading.Event()

        # B-002: OOM-retry bookkeeping (lives across runs to detect a
        # cascading failure pattern; reset on each successful run).
        self._oom_consecutive_at_min_batch = 0

        # Phase 4.3: Early stopping tracking
        self._validation_losses: list[float] = []
        self._early_stop_counter = 0
        self._best_val_loss = float('inf')

        # Phase 5.3: Checkpoint manager
        self._checkpoint_manager: CheckpointManager | None = None

        # B-033: log labels reflect the canonical MultiRun naming. The
        # SpeedrunTrainer alias still works for callers, but the log surface
        # consistently says "MultiRun".
        logger.info(
            f"MultiRunTrainer initialized: {self.config.num_runs} runs x "
            f"{self.config.steps_per_run} steps x {self.config.samples_per_run} samples"
        )
        logger.info(f"Merge mode: {self.config.merge_mode.value}, oom_recovery={self.oom_recovery}")

    def _maybe_resume(self, checkpoint_dir: Path) -> str | None:
        """F-002: resolve the resume hint into a concrete run_id (or None).

        Returns the run_id of a previous session to resume, or ``None`` if
        we should start fresh.
        """
        if self._resume_from is not None and self._resume_from.lower() == "off":
            return None

        history = RunHistoryManager(str(checkpoint_dir))

        # Explicit run_id requested.
        if self._resume_from:
            record = history.get_run(self._resume_from)
            if record is None:
                logger.warning(
                    f"resume_from={self._resume_from!r} not found in run history; "
                    "starting fresh."
                )
                return None
            return str(record.get("run_id"))

        # Default: auto-detect an in-progress entry.
        in_progress = history.in_progress_runs()
        # Only auto-resume multi-run sessions; single-run resume is a different
        # workflow handled by the Trainer directly.
        in_progress = [r for r in in_progress if r.get("session_kind") == "multi_run"]
        if not in_progress:
            return None
        # Pick the most recent.
        in_progress.sort(
            key=lambda r: str(r.get("started_at") or r.get("timestamp") or ""),
            reverse=True,
        )
        candidate = in_progress[0]
        run_id = str(candidate.get("run_id"))
        logger.info(
            f"Auto-detected in-progress multi-run {run_id} — resuming. "
            "Pass resume_from='off' to start fresh."
        )
        return run_id

    def _restore_session_state(
        self,
        checkpoint_dir: Path,  # noqa: ARG002 — reserved for future use (verify checkpoint files on disk match the manifest)
        run_id: str,
    ) -> bool:
        """F-002: rehydrate session state from disk for a resumed run_id.

        Returns True when at least the checkpoint manager identified a
        prior checkpoint we can load from; False when nothing useful was
        found (the caller falls back to a fresh start).
        """
        assert self._checkpoint_manager is not None

        last_cp = self._checkpoint_manager.find_latest_for_run_id(run_id)
        if last_cp is None:
            logger.warning(
                f"No checkpoint found for run_id={run_id}; cannot resume — "
                "starting from run 1."
            )
            return False

        self._resume_checkpoint_path = last_cp.path
        # The next run we execute starts AFTER the last completed run.
        self._resume_start_run_idx = last_cp.run_index + 1
        self._resumed_from_run_id = run_id
        logger.info(
            f"Resuming run_id={run_id} from checkpoint {last_cp.path} "
            f"(run_index={last_cp.run_index}); next run = "
            f"{self._resume_start_run_idx}/{self.config.num_runs}."
        )

        # Restore SLAO merger state if a merger directory exists alongside
        # the checkpoint and we're in SLAO mode.
        if self._slao_merger is not None:
            slao_path = Path(last_cp.path).parent / "slao"
            if slao_path.exists():
                try:
                    # SLAOMerger.load is the symmetric counterpart of
                    # SLAOMerger.save (used in _execute_run after every
                    # successful merge).
                    self._slao_merger.load(str(slao_path))
                    logger.info(f"Restored SLAO merger state from {slao_path}")
                except Exception as exc:
                    logger.warning(
                        f"Failed to restore SLAO state from {slao_path}: {exc}. "
                        "Continuing without it — the first resumed run will "
                        "re-initialize the merger."
                    )

        return True

    def run(self, dataset: DatasetLoader | str | Any = None) -> SpeedrunResult:
        """
        Execute all multi-run training runs.

        Args:
            dataset: One of:
                - ``DatasetLoader`` instance (e.g. from a UI upload — passes
                  through to ``_load_full_dataset`` for validation + ChatML
                  conversion without a string round-trip).
                - String path to a local file (.jsonl/.json/.csv/.parquet/.txt/.md)
                  or a HuggingFace Hub dataset name.
                - Pre-loaded ``datasets.Dataset`` object.
                - ``None`` to use ``settings.data.dataset_name`` from config.

            The ``DatasetLoader`` branch was added for F-016 so the UI no
            longer has to stringify ``state.dataset_loader.source`` to round-
            trip a user-uploaded file through this entrypoint. The body
            dispatch lives in ``_load_full_dataset`` which already handles all
            three concrete types.

        Returns:
            MultiRunResult with all run results and aggregate metrics. The
            ``run_id`` field on the result is the same correlation token
            attached to every log line, checkpoint manifest entry, and SLAO
            merge_history.json entry produced by this session.
        """
        from .trainer import Trainer

        start_time = time.time()
        self._is_running = True
        self._should_abort = False
        self._abort_reason = None

        # Setup checkpoint directory first so the resume detector can read
        # the on-disk run history before we mint a new run_id.
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # F-002: probe for a resume candidate before minting a new run_id.
        resume_run_id = self._maybe_resume(checkpoint_dir)

        # B-001: mint a session-wide correlation token and bind it to the
        # structured-logger context so every log line emitted from this point
        # forward carries `run_id=<id>`. The unbind happens in the finally
        # block at the bottom of this method so the context doesn't leak into
        # callers that reuse the same thread.
        # F-002: when resuming, re-use the prior run_id so the on-disk history
        # entry / checkpoint manifest / SLAO merge_history all stay grouped
        # under the same correlation token.
        self._run_id = resume_run_id or uuid.uuid4().hex
        bind_run_context(run_id=self._run_id, session_kind="multi_run")
        if resume_run_id:
            logger.info(f"run_resumed run_id={self._run_id}")
        else:
            logger.info(f"run_started run_id={self._run_id}")

        # F-003: record the multi-run session start in the on-disk run history.
        # Errors here never kill the run — we never want history persistence
        # to gate training.
        run_history = RunHistoryManager(str(checkpoint_dir))
        dataset_info = (
            dataset if isinstance(dataset, str) else type(dataset).__name__
        )
        hyperparameters = {
            "num_runs": self.config.num_runs,
            "steps_per_run": self.config.steps_per_run,
            "samples_per_run": self.config.samples_per_run,
            "merge_mode": self.config.merge_mode.value,
            "initial_lr": self.config.initial_lr,
            "final_lr": self.config.final_lr,
            "lr_decay": self.config.lr_decay,
            "warmup_steps_per_run": self.config.warmup_steps_per_run,
            "shuffle_data": self.config.shuffle_data,
            "replay_fraction": self.config.replay_fraction,
            "replay_strategy": self.config.replay_strategy,
            "validation_samples": self.config.validation_samples,
            "validate_every_run": self.config.validate_every_run,
            "early_stopping": self.config.early_stopping,
            "early_stopping_patience": self.config.early_stopping_patience,
            "adaptive_scaling": self.config.adaptive_scaling,
            "layer_scaling": self.config.layer_scaling,
            "seed": settings.training.seed,
        }
        # F-004: deferred import to avoid circular dependency with trainer.py
        # (multi_run is imported by trainer.py at runtime via __getattr__).
        try:
            from .trainer import _compute_dataset_hash

            dataset_hash = _compute_dataset_hash(dataset)
        except Exception:
            dataset_hash = None
        try:
            if resume_run_id is None:
                run_history.record_run_started(
                    run_id=self._run_id,
                    model_name=self.model_name,
                    dataset_info=dataset_info,
                    hyperparameters=hyperparameters,
                    session_kind="multi_run",
                    checkpoint_path=str(checkpoint_dir),
                    dataset_hash=dataset_hash,
                )
            else:
                # F-002: keep the original run record but flip its status
                # back to "running" so list-runs --status running shows it
                # while the resume is live.
                run_history.update_run(self._run_id, status="running")
        except Exception as hist_err:
            logger.warning(
                f"RunHistoryManager.record_run_started failed: {hist_err}"
            )
        self._run_history = run_history

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

        # F-002: if we're resuming, rehydrate the start index + checkpoint
        # path from the persisted state. This MUST happen after the
        # checkpoint manager is initialised so the manifest has been loaded.
        if resume_run_id is not None:
            if not self._restore_session_state(checkpoint_dir, resume_run_id):
                # Couldn't rehydrate — fall back to a fresh session for this
                # run_id (the existing history record gets re-used; we just
                # don't skip any run indices).
                self._resume_start_run_idx = 1

        # Initialize SLAO merger if using SLAO mode.
        # F-015: thread MultiRunConfig.adaptive_scaling + layer_scaling through
        # to SLAOConfig.use_adaptive_scaling + use_layer_scaling so the
        # operator's intent reaches the merger. The SLAOMerger implementation
        # may still treat these as no-ops in v1.1.0, but the data contract is
        # now end-to-end and merge_history.json carries the flag values for
        # post-hoc audit.
        if self.config.merge_mode == MergeMode.SLAO:
            self._slao_merger = SLAOMerger(SLAOConfig(
                scaling_type="sqrt",
                use_orthogonal_init=True,
                use_adaptive_scaling=self.config.adaptive_scaling,
                use_layer_scaling=self.config.layer_scaling,
            ))

        # Pre-flight GPU check
        if not self._preflight_gpu_check():
            # B-001: emit run_ended and unbind context on this early-exit path
            # so the correlation token doesn't leak into the next caller.
            logger.info(f"run_ended run_id={self._run_id} status=aborted")
            # F-003: record GPU-safety-abort as a failed run.
            try:
                self._run_history.record_run_failed(
                    run_id=self._run_id,
                    failure_reason="GPU safety check failed",
                    duration_seconds=time.time() - start_time,
                    checkpoint_path=str(checkpoint_dir),
                )
            except Exception as hist_err:
                logger.warning(
                    f"RunHistoryManager.record_run_failed failed: {hist_err}"
                )
            unbind_run_context("run_id", "session_kind")
            return self._create_abort_result("GPU safety check failed")

        # Start GPU monitoring
        if self.config.enable_gpu_monitoring:
            self._start_gpu_monitor()

        # Load dataset
        logger.info("Loading dataset...")
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
                train_on_responses=True,  # FT-010: same quality optimization as single-run
                report_to=self._report_to_intent,  # F-005
            )
            self._trainer.load_model()

            # B-005: fail loud at startup if the installed PEFT version does
            # not expose a usable extraction path. Without this, a future PEFT
            # rename of get_adapter_state_dict / lora_A / lora_B substrings
            # would silently no-op the SLAO accumulator across runs.
            if self.config.merge_mode == MergeMode.SLAO:
                self._verify_peft_api()

            # Execute runs
            # F-002: when resuming, start from where the prior session left off.
            # _resume_start_run_idx defaults to 1 for fresh sessions.
            start_idx = max(1, self._resume_start_run_idx)
            if start_idx > 1:
                logger.info(
                    f"Resuming multi-run from run {start_idx}/{self.config.num_runs}"
                )
                # F-002: when resuming, hydrate the model with the last
                # checkpoint so the SLAO accumulator + LoRA weights pick up
                # where they left off.
                if self._resume_checkpoint_path:
                    try:
                        self._load_resume_checkpoint(self._resume_checkpoint_path)
                    except Exception as exc:
                        logger.warning(
                            f"Failed to load resume checkpoint "
                            f"{self._resume_checkpoint_path}: {exc}. "
                            "Continuing with the freshly loaded base model — "
                            "results may diverge from the original run."
                        )
            for run_idx in range(start_idx, self.config.num_runs + 1):
                if self._should_abort:
                    break

                # Check if GPU pause is active (B-020: overheat protection).
                # The monitor thread sets the event when overheating and clears it
                # when the GPU is safe again. We poll here so the main thread blocks.
                if self._gpu_pause_event.is_set():
                    logger.info("Waiting for GPU cooldown before starting run...")
                    while self._gpu_pause_event.is_set() and not self._should_abort:
                        time.sleep(1.0)

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
            logger.error(f"MultiRun failed: {e}")
            self._abort_reason = str(e)
            # B-001: emit run_ended with status before re-raising so the
            # operator sees both endpoints of the correlation window.
            logger.info(f"run_ended run_id={self._run_id} status=error")
            # F-003: persist the failure in run history before re-raising so
            # `backprop list-runs --status failed` surfaces this session.
            try:
                self._run_history.record_run_failed(
                    run_id=self._run_id,
                    failure_reason=f"{type(e).__name__}: {e}",
                    duration_seconds=time.time() - start_time,
                    checkpoint_path=str(checkpoint_dir),
                    extra={"merge_history": self._collect_merge_history()},
                )
            except Exception as hist_err:
                logger.warning(
                    f"RunHistoryManager.record_run_failed failed: {hist_err}"
                )
            raise

        finally:
            self._is_running = False
            if self._gpu_monitor:
                self._gpu_monitor.stop()
            # B-001: unbind the correlation token so subsequent unrelated work
            # on this thread doesn't accidentally pick up our run_id.
            unbind_run_context("run_id", "session_kind")

        # B-001: success path — emit run_ended explicitly. (The error path
        # already emitted run_ended inside the except above.)
        logger.info(f"run_ended run_id={self._run_id} status=ok")

        # Create final result
        total_duration = time.time() - start_time
        result = self._create_result(total_duration)

        # F-003: record successful completion of the multi-run session.
        try:
            self._run_history.record_run_completed(
                run_id=self._run_id,
                final_loss=result.final_loss,
                loss_history=list(self._aggregate_loss),
                steps=result.total_steps,
                duration_seconds=total_duration,
                gpu_max_temp=self._gpu_max_temp or None,
                checkpoint_path=result.final_checkpoint_path,
                merge_history=self._collect_merge_history(),
            )
        except Exception as hist_err:
            logger.warning(
                f"RunHistoryManager.record_run_completed failed: {hist_err}"
            )
        return result

    def abort(self, reason: str = "User requested abort") -> None:
        """Request abort of current multi-run session."""
        self._should_abort = True
        self._abort_reason = reason
        logger.warning(f"MultiRun abort requested: {reason}")

    def get_checkpoint_manager(self) -> CheckpointManager | None:
        """Get the checkpoint manager for external access (e.g., UI)."""
        return self._checkpoint_manager

    def get_checkpoint_stats(self) -> CheckpointStats | None:
        """Get current checkpoint statistics."""
        if self._checkpoint_manager:
            return self._checkpoint_manager.get_stats()
        return None

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        """B-002: identify CUDA OOMs across torch + RuntimeError variants.

        ``torch.cuda.OutOfMemoryError`` was added in torch 2.0. Older torch
        and some driver-level errors surface as a plain ``RuntimeError``
        whose message contains ``"out of memory"``. We accept both.
        """
        try:
            import torch
            if isinstance(exc, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
                return True
        except (ImportError, AttributeError):
            pass
        if isinstance(exc, RuntimeError):
            return "out of memory" in str(exc).lower()
        return False

    def _execute_run(
        self,
        run_idx: int,
        full_dataset: Any,
        checkpoint_dir: Path,
    ) -> RunResult:
        """Execute a single training run.

        B-002: the actual ``trainer.train()`` call is wrapped in an inner
        retry loop that halves ``per_device_train_batch_size`` and doubles
        ``gradient_accumulation_steps`` on CUDA OOM (preserving effective
        batch size), clears the CUDA allocator, and retries. After
        :attr:`_OOM_MAX_RETRIES_AT_MIN_BATCH` consecutive OOMs at
        ``batch_size==1`` the session is aborted with a structured error
        pointing the operator at smaller-model / shorter-seq remediations.
        """
        from trl import SFTConfig, SFTTrainer

        run_start = time.time()

        logger.info(f"\n{'='*60}")
        logger.info(f"MULTIRUN {run_idx}/{self.config.num_runs} run_id={self._run_id}")
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

        # B-019: reset GPU tracking BEFORE the run so the snapshot at the end
        # reflects ONLY this run's window (eliminates the GPU-monitor-races-
        # main-thread observation: the reset used to happen after the snapshot).
        self._gpu_max_temp = 0.0
        self._gpu_max_vram = 0.0

        # Train — wrapped so a single run crash (OOM, CUDA error) does not
        # kill the entire multi-run session. B-002 layers an inner OOM-retry
        # loop on top so the run can survive a one-shot VRAM spike instead
        # of compounding the failure across the remaining runs.
        logger.info(f"Training run {run_idx}...")
        run_failed = False
        failure_reason: str | None = None
        loss_history: list[float] = []
        final_loss = 0.0
        merge_result = None
        checkpoint_path: str | None = None
        oom_retries = 0
        trainer = None  # noqa: F841 — assigned inside the loop, referenced after
        result: Any = None

        # F-005: resolve report_to from the inner Trainer instance (carries
        # the operator's intent). MultiRun inherits the same wiring policy,
        # so a single multi-run session emits a single W&B project with a
        # per-run name like ``backprop-<run_id_prefix>-run-<run_idx>``.
        report_to_resolved = self._trainer._resolve_report_to()
        run_name = (
            f"backprop-{self._run_id[:12]}-run-{run_idx:03d}"
            if (self._run_id and report_to_resolved != "none")
            else None
        )

        # B-002: per-run OOM retry. We keep retrying with halved batch until
        # batch_size hits 1 (the floor). At the floor, _OOM_MAX_RETRIES_AT_MIN_BATCH
        # consecutive OOMs trigger session abort.
        while True:
            # Build training args inside the loop so a retry picks up the
            # halved batch / doubled accumulation.
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
                report_to=report_to_resolved,  # F-005
                run_name=run_name,
                seed=settings.training.seed + run_idx,  # Different seed each run
                # SFT-specific args (TRL 0.24+)
                max_length=self._trainer.max_seq_length,
                packing=settings.data.packing,
            )

            trainer = SFTTrainer(
                model=self._trainer._model,
                processing_class=self._trainer._tokenizer,
                train_dataset=chunk_dataset,
                args=training_args,
            )

            try:
                result = trainer.train()

                # Extract loss history
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

                # Merge LoRA weights (B-008: passes run_id through for the
                # divergence-trigger envelope)
                if self.config.merge_mode == MergeMode.SLAO and self._slao_merger:
                    lora_state = self._get_lora_state_dict()
                    merge_result = self._slao_merger.merge(
                        lora_state,
                        run_index=run_idx,
                        run_id=self._run_id,
                    )

                # B-002: success resets the consecutive-OOM-at-floor counter.
                self._oom_consecutive_at_min_batch = 0
                break  # Successful run — exit retry loop

            except (KeyboardInterrupt, SystemExit):
                # Never swallow these — let them propagate.
                raise
            except Exception as exc:
                if self.oom_recovery and self._is_oom_error(exc):
                    # B-002: OOM-specific recovery path.
                    import torch as _torch

                    current_batch = self._trainer.batch_size
                    current_accum = self._trainer.gradient_accumulation
                    effective = current_batch * current_accum
                    logger.warning(
                        f"OOM detected on run {run_idx} (run_id={self._run_id}) "
                        f"batch={current_batch} grad_accum={current_accum} "
                        f"effective_batch={effective}: {exc}"
                    )

                    # Free everything we can before retrying.
                    try:
                        del trainer
                    except UnboundLocalError:
                        pass
                    trainer = None
                    gc.collect()
                    try:
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                    except Exception:  # nosec B110 — best-effort GPU cache reclaim; failures are non-fatal
                        pass

                    if current_batch > 1:
                        # Halve batch, double accumulation to preserve
                        # effective batch size. Reset floor-counter since
                        # we're not at the floor yet.
                        new_batch = max(1, current_batch // 2)
                        new_accum = max(1, current_accum * 2)
                        logger.warning(
                            f"OOM recovery: halving batch_size {current_batch}->{new_batch}, "
                            f"doubling gradient_accumulation {current_accum}->{new_accum} "
                            f"(effective batch preserved at {new_batch * new_accum})"
                        )
                        self._trainer.batch_size = new_batch
                        self._trainer.gradient_accumulation = new_accum
                        self._oom_consecutive_at_min_batch = 0
                        oom_retries += 1
                        continue  # Retry with new args

                    # Already at min batch — count toward the abort threshold.
                    self._oom_consecutive_at_min_batch += 1
                    oom_retries += 1
                    logger.error(
                        f"OOM at batch=1 (run_id={self._run_id}) "
                        f"consecutive={self._oom_consecutive_at_min_batch}/"
                        f"{self._OOM_MAX_RETRIES_AT_MIN_BATCH}"
                    )
                    if self._oom_consecutive_at_min_batch >= self._OOM_MAX_RETRIES_AT_MIN_BATCH:
                        raise BackpropagateError(
                            f"Persistent CUDA OOM at batch_size=1 "
                            f"({self._oom_consecutive_at_min_batch} consecutive "
                            f"OOMs across runs); cannot recover automatically.",
                            code="RUNTIME_GPU_OOM",
                            details={
                                "run_index": run_idx,
                                "run_id": self._run_id,
                                "consecutive_oom_at_min_batch": self._oom_consecutive_at_min_batch,
                            },
                            suggestion=(
                                "Use a smaller model (3B → 1B), reduce "
                                "max_seq_length, enable gradient_checkpointing, "
                                "or apply 4-bit / 8-bit quantization. If you "
                                "want OOMs to hard-fail the run instead of "
                                "session-abort, construct the trainer with "
                                "oom_recovery=False."
                            ),
                            retryable=False,
                            cause=exc,
                        ) from exc

                    # Below the threshold — record the failure for this run
                    # but don't retry (we've already retried at min batch
                    # `consecutive` times across the session).
                    run_failed = True
                    failure_reason = f"{type(exc).__name__}: {exc}"
                    break

                # Non-OOM exception (or OOM with oom_recovery=False).
                run_failed = True
                failure_reason = f"{type(exc).__name__}: {exc}"
                logger.error(f"Run {run_idx} failed: {failure_reason}")

                # Salvage any partial loss history from the trainer state
                if (
                    trainer is not None
                    and hasattr(trainer, 'state')
                    and hasattr(trainer.state, 'log_history')
                    and trainer.state.log_history
                ):
                    loss_history = [
                        log.get('loss', 0) for log in trainer.state.log_history
                        if 'loss' in log
                    ]
                    self._aggregate_loss.extend(loss_history)
                    final_loss = loss_history[-1] if loss_history else 0.0
                break

        # Save checkpoint (even on failure — preserves partial work)
        if self.config.save_every_run:
            try:
                # B-006: atomic write. Trainer.save() now writes into
                # <path>.partial and shutil.move()s it into place on success;
                # the partial dir is removed if anything raises mid-write.
                checkpoint_path = str(checkpoint_dir / f"run_{run_idx:03d}" / "lora")
                self._trainer.save(
                    checkpoint_path,
                    run_id=self._run_id,
                )
                logger.info(f"Checkpoint saved: {checkpoint_path}")

                # Also save SLAO merger state (B-001: thread run_id through)
                if self._slao_merger:
                    self._slao_merger.save(
                        str(checkpoint_dir / f"run_{run_idx:03d}" / "slao"),
                        run_id=self._run_id,
                    )

                # Phase 5.3: Register with checkpoint manager for smart pruning
                # (B-001: pass run_id so manifest can carry the correlation token)
                if self._checkpoint_manager:
                    is_first_run = (run_idx == 1)
                    self._checkpoint_manager.register(
                        run_index=run_idx,
                        checkpoint_path=checkpoint_path,
                        validation_loss=None,  # Will be updated if validation is enabled
                        training_loss=final_loss,
                        is_run_boundary=is_first_run,
                        run_id=self._run_id,
                    )
                    # Note: auto_prune happens inside register() if enabled
            except Exception as save_err:
                logger.error(f"Failed to save checkpoint for run {run_idx}: {save_err}")
                checkpoint_path = None

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
            failed=run_failed,
            failure_reason=failure_reason,
            run_id=self._run_id,
            oom_retries=oom_retries,
        )

        # VRAM cleanup between runs (FT-003): release the per-run SFTTrainer
        # and reclaim GPU memory so subsequent runs don't OOM.
        import torch

        if trainer is not None:
            del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                f"VRAM cleanup after run {run_idx}: "
                f"{torch.cuda.memory_allocated() / 1e9:.1f}GB allocated"
            )

        if run_failed:
            logger.warning(f"Run {run_idx} FAILED: {failure_reason} (time={duration:.1f}s)")
        else:
            logger.info(f"Run {run_idx} complete: loss={final_loss:.4f}, time={duration:.1f}s")

        return run_result

    def _execute_run_with_validation(
        self,
        run_idx: int,
        full_dataset: Any,
        checkpoint_dir: Path,
    ) -> tuple[RunResult, float | None]:
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

    def _validation_active(self) -> bool:
        """True if validation will be computed (so a holdout split is needed)."""
        return bool(self.config.validate_every_run or self.config.early_stopping)

    def _get_validation_holdout(self, total_samples: int) -> tuple[int, int]:
        """
        Compute the validation holdout split for a dataset.

        Returns ``(holdout_size, val_start)`` where validation samples occupy
        ``[val_start, total_samples)``. When validation is disabled the
        holdout is zero and ``val_start == total_samples`` (no holdout).

        Keeping this in one place lets the training-data chunker and the
        validation-loss computer agree on the split boundary, which is the
        precondition for ``no_overlap`` between train and validation indices.
        """
        if not self._validation_active():
            return 0, total_samples
        holdout_size = max(int(total_samples * 0.10), 1)
        val_start = total_samples - holdout_size
        return holdout_size, val_start

    def _get_train_pool_size(self, total_samples: int) -> int:
        """
        Size of the indexable training pool ``[0, train_pool_size)``.

        Equals ``total_samples - holdout_size`` when validation is active,
        else ``total_samples``. Training chunks (and replay) must draw their
        indices from this range to avoid silently leaking validation samples
        into the train set when ``num_runs * samples_per_run`` exceeds the
        train pool and wrap-around kicks in.
        """
        _, val_start = self._get_validation_holdout(total_samples)
        return val_start

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

        Train/validation separation contract:
            When validation is active (``validate_every_run`` or
            ``early_stopping``), the last 10% of the dataset is reserved as
            a held-out validation split and training NEVER reads from it,
            even on wrap-around. Indices are drawn from the training pool
            ``[0, total_samples - holdout_size)`` and wrap inside that range.
            When validation is disabled, the full dataset is the training pool.

        Raises:
            ConfigurationError: if ``samples_per_run`` is larger than the
                training pool (a single chunk would not fit). The fix is to
                reduce ``samples_per_run`` or disable validation.
        """
        from datasets import concatenate_datasets

        total_samples = len(full_dataset)
        train_pool_size = self._get_train_pool_size(total_samples)
        chunk_size = self.config.samples_per_run

        # A chunk larger than the train pool cannot be drawn without spilling
        # into validation. Fail loud rather than silently produce a polluted
        # chunk. The strict-equality case (`>`) preserves the wrap-around
        # behaviour when chunk_size == train_pool_size — every run pulls the
        # same indices, but no validation overlap occurs.
        if train_pool_size <= 0:
            raise ConfigurationError(
                f"Training pool is empty (total_samples={total_samples}, "
                f"holdout_size={total_samples - train_pool_size}). Dataset is too small "
                f"to leave room for both training and validation.",
                details={
                    "total_samples": total_samples,
                    "train_pool_size": train_pool_size,
                },
                suggestion=(
                    "Use a larger dataset, or disable validation "
                    "(validate_every_run=False, early_stopping=False)."
                ),
            )
        if chunk_size > train_pool_size:
            if self._validation_active():
                raise ConfigurationError(
                    f"samples_per_run ({chunk_size}) exceeds the available training pool "
                    f"({train_pool_size} = total_samples {total_samples} minus 10% validation holdout). "
                    f"A training chunk cannot fit without overlapping the validation set.",
                    details={
                        "samples_per_run": chunk_size,
                        "total_samples": total_samples,
                        "train_pool_size": train_pool_size,
                        "holdout_size": total_samples - train_pool_size,
                    },
                    suggestion=(
                        # F-017: align CLI flag name with the real argparse
                        # surface in cli.py:1338 (--samples, NOT
                        # --samples-per-run). The Python-API knob is still
                        # called ``samples_per_run`` (see MultiRunConfig), so
                        # we name both surfaces explicitly to avoid leading
                        # the operator at the wrong layer.
                        "Reduce --samples (Python: samples_per_run) below "
                        f"{train_pool_size}, increase dataset size, "
                        "or disable validation (validate_every_run=False, early_stopping=False)."
                    ),
                )
            raise ConfigurationError(
                f"samples_per_run ({chunk_size}) exceeds total dataset size ({total_samples}).",
                details={"samples_per_run": chunk_size, "total_samples": total_samples},
                # F-017: see comment above — CLI flag is --samples in cli.py:1338.
                suggestion=(
                    f"Reduce --samples (Python: samples_per_run) below "
                    f"{total_samples} or use a larger dataset."
                ),
            )

        # Calculate how many new vs replay samples
        replay_fraction = min(self.config.replay_fraction, 0.5)  # Cap at 50%
        if run_idx == 1 or replay_fraction <= 0:
            # First run or no replay - just get new samples
            new_samples_count = chunk_size
            replay_count = 0
        else:
            replay_count = int(chunk_size * replay_fraction)
            new_samples_count = chunk_size - replay_count

        # Get new samples for this run. We index into the TRAINING POOL only
        # (`[0, train_pool_size)`) and wrap inside it, so wrap-around can never
        # reach into the validation holdout `[train_pool_size, total_samples)`.
        start_idx = ((run_idx - 1) * self.config.samples_per_run) % train_pool_size
        end_idx = start_idx + new_samples_count

        # Handle wrap-around for new samples (inside the train pool only).
        if end_idx > train_pool_size:
            new_indices = list(range(start_idx, train_pool_size)) + list(range(0, end_idx - train_pool_size))
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

        Note:
            Replay indices are drawn from the same training pool used by
            ``_get_data_chunk`` (i.e. ``[0, train_pool_size)``) so replay
            cannot resurrect a sample from the validation holdout, which would
            re-introduce the train/validation overlap that the chunker is
            careful to avoid.
        """
        import random

        total_samples = len(full_dataset)
        train_pool_size = self._get_train_pool_size(total_samples)
        samples_per_run = self.config.samples_per_run

        # Calculate indices from previous runs. All ranges are clamped to the
        # training pool so we never pull validation samples into replay.
        if self.config.replay_strategy == "recent":
            # Get samples from the most recent previous run
            prev_run = run_idx - 1
            prev_start = ((prev_run - 1) * samples_per_run) % train_pool_size
            prev_end = min(prev_start + samples_per_run, train_pool_size)
            available_indices = list(range(prev_start, prev_end))

        elif self.config.replay_strategy == "random":
            # Random samples from all previous runs
            all_prev_indices: list[int] = []
            for prev_run in range(1, run_idx):
                prev_start = ((prev_run - 1) * samples_per_run) % train_pool_size
                prev_end = min(prev_start + samples_per_run, train_pool_size)
                all_prev_indices.extend(range(prev_start, prev_end))
            available_indices = list(set(all_prev_indices))  # Remove duplicates

        elif self.config.replay_strategy == "all_previous":
            # Uniform sample from all data seen so far (within the train pool).
            total_seen = min((run_idx - 1) * samples_per_run, train_pool_size)
            available_indices = list(range(total_seen))

        else:
            # Default to recent
            prev_run = run_idx - 1
            prev_start = ((prev_run - 1) * samples_per_run) % train_pool_size
            prev_end = min(prev_start + samples_per_run, train_pool_size)
            available_indices = list(range(prev_start, prev_end))

        # Sample from available indices
        if len(available_indices) == 0:
            return None

        # Use a LOCAL Random instance so we don't mutate the global Python RNG.
        # The global mutation pattern (`random.seed(...)`) pollutes any other
        # code running in the same process that uses `random` (datasketch.MinHash
        # for dedup, user-supplied custom_filter callbacks, transformers internals,
        # etc.) and silently makes successive replay calls deterministically
        # identical within the same run. A local Random preserves reproducibility
        # without that footgun.
        # nosec B311 — random.Random is correct here: deterministic replay sampling, not crypto.
        # Per B-002 doctrine: use local random.Random instance, NOT the global random.seed(),
        # so we don't pollute the process-wide RNG state.
        rng = random.Random(settings.training.seed + run_idx + 1000)  # nosec B311 — non-crypto replay seed; see comment above
        replay_indices = rng.sample(available_indices, min(count, len(available_indices)))

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

    def _get_lora_state_dict(self) -> dict[str, Any]:
        """Extract LoRA adapter state dict from model.

        B-005: logs the extraction path (PEFT API vs manual) at DEBUG so
        operators can see which code path produced the SLAO inputs when
        triaging a divergent merge. The startup invariant
        (see :meth:`_verify_peft_api`) is the LOUD failure surface; this
        method's logging is the quiet breadcrumb.
        """

        model = self._trainer._model

        # Try to get PEFT adapter state via published API.
        if hasattr(model, 'get_adapter_state_dict'):
            logger.debug("LoRA extraction path=peft_get_adapter_state_dict")
            result: dict[str, Any] = model.get_adapter_state_dict()
            return result

        # Fallback: extract lora parameters manually (kept for forward-compat
        # with PEFT versions that may rename or drop the helper).
        logger.debug("LoRA extraction path=manual_named_parameters_scan")
        lora_state = {}
        for name, param in model.named_parameters():
            if 'lora_' in name.lower():
                lora_state[name] = param.data.clone()

        return lora_state

    def _verify_peft_api(self) -> None:
        """B-005: invariant check at session start.

        After the model is loaded, verify that at least one known PEFT
        extraction path produces a non-empty LoRA state dict whose keys
        match the expected ``.lora_A.`` / ``.lora_B.`` pattern. If not,
        raise :class:`BackpropagateError(code="PEFT_API_INCOMPATIBLE")`
        with a hint pointing at the supported PEFT version range from
        pyproject.toml.

        The pre-existing silent-fallback path (manual ``named_parameters``
        scan) means a future PEFT rename of ``lora_A`` / ``lora_B``
        substrings would produce an EMPTY state dict, which SLAOMerger
        would happily merge into a no-op accumulator — the model trains
        fine each run, but multi-run merging is silently dead. This
        check fails loud at startup before any training happens.
        """
        try:
            lora_state = self._get_lora_state_dict()
        except Exception as e:
            raise BackpropagateError(
                f"PEFT adapter state extraction raised an exception at startup: {e}",
                code="PEFT_API_INCOMPATIBLE",
                details={"reason": str(e)},
                suggestion=(
                    "Backpropagate requires peft>=0.7.0 (see pyproject.toml). "
                    "Reinstall: pip install 'peft>=0.7.0'. If you pinned a "
                    "specific PEFT version, verify it still exposes "
                    "get_adapter_state_dict / load_adapter_state_dict OR "
                    "uses '.lora_A.' / '.lora_B.' parameter names."
                ),
                retryable=False,
            ) from e

        if not lora_state:
            raise BackpropagateError(
                "PEFT API invariant check failed: no LoRA parameters extracted.",
                code="PEFT_API_INCOMPATIBLE",
                details={"extracted_param_count": 0},
                suggestion=(
                    "The installed PEFT version does not expose any "
                    "`.lora_A.` / `.lora_B.` parameters and lacks "
                    "get_adapter_state_dict. Required range: peft>=0.7.0 "
                    "(see pyproject.toml). Reinstall: pip install 'peft>=0.7.0'."
                ),
                retryable=False,
            )

        a_keys = sum(1 for k in lora_state if '.lora_A.' in k or 'lora_A' in k.lower())
        b_keys = sum(1 for k in lora_state if '.lora_B.' in k or 'lora_B' in k.lower())
        path = (
            "peft_get_adapter_state_dict"
            if hasattr(self._trainer._model, 'get_adapter_state_dict')
            else "manual_named_parameters_scan"
        )
        logger.debug(
            f"PEFT API invariant OK: path={path} "
            f"params={len(lora_state)} A_keys={a_keys} B_keys={b_keys}"
        )
        if a_keys == 0 or b_keys == 0:
            # Soft warn: extraction yielded SOMETHING but not the canonical
            # A/B split SLAO depends on. This isn't necessarily fatal (some
            # adapter modules have only one matrix per layer), but it's
            # worth surfacing.
            logger.warning(
                f"PEFT extraction yielded {len(lora_state)} params but A_keys={a_keys} "
                f"and B_keys={b_keys} — SLAO merge expects both. Verify PEFT "
                f"version compatibility (pyproject.toml requires peft>=0.7.0)."
            )

    def _load_lora_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load LoRA state dict into model."""

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

    def _load_full_dataset(self, dataset: DatasetLoader | str | Any) -> Any:
        """
        Load the full dataset for chunking.

        Routes local file paths through DatasetLoader for format detection,
        validation, and ChatML conversion (same behaviour as Trainer._load_dataset).
        HuggingFace Hub names and pre-loaded Dataset objects still work as before.

        Raises:
            DatasetNotFoundError: If a local file path doesn't exist
            DatasetError: For network errors, parse failures, or unsupported types
        """
        from datasets import Dataset, load_dataset

        if dataset is None:
            dataset = settings.data.dataset_name

        # File extensions handled by DatasetLoader
        _LOCAL_FILE_EXTENSIONS = (
            '.jsonl', '.json', '.csv', '.parquet', '.txt', '.md',
        )

        try:
            if isinstance(dataset, DatasetLoader):
                # Already a DatasetLoader — validate and convert
                validation = dataset.validation_result
                if validation and validation.warnings:
                    for w in validation.warnings[:5]:
                        logger.warning(f"Dataset validation warning: {w}")
                ds = dataset.to_hf_dataset()
            elif isinstance(dataset, str):
                if any(dataset.lower().endswith(ext) for ext in _LOCAL_FILE_EXTENSIONS):
                    # Local file — route through DatasetLoader for format
                    # detection, validation, and ChatML conversion
                    try:
                        loader = DatasetLoader(dataset, validate=True)
                    except FileNotFoundError as e:
                        raise DatasetNotFoundError(
                            dataset,
                            suggestion="Create the file or use a HuggingFace dataset name",
                        ) from e

                    logger.info(f"DatasetLoader detected format: {loader.detected_format.value}")

                    # Log validation warnings/errors
                    validation = loader.validation_result
                    if validation and validation.warnings:
                        for w in validation.warnings[:5]:
                            logger.warning(f"Dataset validation warning: {w}")
                    if validation and not validation.is_valid:
                        logger.warning(
                            f"Dataset has {validation.error_count} validation errors "
                            f"({validation.error_rate:.1%} error rate) — proceeding anyway"
                        )
                        for err in validation.errors[:5]:
                            logger.warning(f"  {err}")

                    ds = loader.to_hf_dataset()
                else:
                    # HuggingFace Hub dataset name (B-017: transient retry)
                    from .trainer import _retry_hf_call

                    ds = _retry_hf_call(
                        load_dataset,
                        dataset,
                        split=settings.data.dataset_split,
                        _label=f"multi_run_load_dataset:{dataset}",
                    )
            elif isinstance(dataset, Dataset):
                ds = dataset
            else:
                raise DatasetError(
                    f"Unsupported dataset type: {type(dataset).__name__}",
                    suggestion="Use a file path (JSONL, CSV), HuggingFace dataset name, or Dataset object",
                )
        except (DatasetNotFoundError, DatasetError):
            raise
        except FileNotFoundError as e:
            raise DatasetNotFoundError(
                str(e),
                suggestion="Check the file path and ensure the file exists",
            ) from e
        except Exception as e:
            raise DatasetError(
                f"Failed to load dataset: {e}",
                suggestion="Check dataset name/path and network connection",
            ) from e

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
        """Handle GPU status update.

        B-003: when the GPU returns to a SAFE / WARM condition AND
        ``pause_on_overheat`` is enabled, clear ``_gpu_pause_event`` so the
        run loop's polling wake-up at multi_run.py:383 can proceed. The
        complementary ``set()`` lives in :meth:`_on_gpu_critical` — together
        they implement the safety promise the docstring made and the
        original wiring forgot.
        """
        # Track max values
        if status.temperature_c and status.temperature_c > self._gpu_max_temp:
            self._gpu_max_temp = status.temperature_c

        if status.vram_percent > self._gpu_max_vram:
            self._gpu_max_vram = status.vram_percent

        # B-003: pause/resume edge — clear the pause event when the GPU
        # recovers so the run loop can resume scheduling new runs.
        if (
            self.config.pause_on_overheat
            and status.condition in (GPUCondition.SAFE, GPUCondition.WARM)
            and self._gpu_pause_event.is_set()
        ):
            logger.warning(
                f"GPU recovered ({status.condition.value}, "
                f"temp={status.temperature_c}C); resuming multi-run scheduling"
            )
            self._gpu_pause_event.clear()

        # Callback
        if self.on_gpu_status:
            self.on_gpu_status(status)

    def _on_gpu_critical(self, status: GPUStatus) -> None:
        """Handle critical GPU condition.

        B-003: when ``pause_on_overheat=True``, set ``_gpu_pause_event`` so
        the run loop at the top of :meth:`run` blocks before starting the
        NEXT run until the GPU recovers (cleared from :meth:`_on_gpu_status`
        on SAFE/WARM). Mid-training pause is still out of scope (would
        require deeper SFTTrainer callback integration), but the safety
        promise — "the next run gates on cooldown" — is now real.
        """
        logger.warning(f"GPU CRITICAL: {status.condition_reason}")

        if self.config.pause_on_overheat:
            # B-003: actually arm the pause event. Pre-fix this was a logged
            # no-op despite the run-loop polling for it.
            if not self._gpu_pause_event.is_set():
                logger.warning(
                    "pause_on_overheat armed: next run start will wait for "
                    "GPU recovery before launching"
                )
                self._gpu_pause_event.set()

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

        # Dedicated hold-out split: reserve the last 10% of the dataset for
        # validation. Training data (_get_data_chunk) draws indices from
        # [0, val_start) — wrap-around is constrained to the training pool by
        # _get_train_pool_size, so the train/validation split is preserved
        # regardless of how many runs cycle through the dataset.
        total_samples = len(full_dataset)
        holdout_size, val_start = self._get_validation_holdout(total_samples)
        val_count = min(self.config.validation_samples, holdout_size)
        val_indices = list(range(val_start, val_start + val_count))

        val_dataset = full_dataset.select(val_indices)

        # Compute loss on validation set
        model.eval()
        try:
            total_loss = 0.0
            count = 0
            skipped = 0

            with torch.no_grad():
                for sample in val_dataset:
                    try:
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
                    except Exception as e:
                        skipped += 1
                        logger.warning(f"Skipped validation sample (error: {e})")
        finally:
            # Ensure model is restored to training mode on ANY exit path
            # (normal completion, exception, KeyboardInterrupt, etc.)
            model.train()

        if skipped > 0:
            logger.warning(f"Skipped {skipped} validation samples due to errors")

        if count == 0:
            logger.warning("No validation samples were successfully evaluated")
            return float('inf')

        avg_loss = total_loss / count
        logger.info(f"Validation loss (run {run_idx}): {avg_loss:.4f} ({count} samples)")
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

    def _load_resume_checkpoint(self, checkpoint_path: str) -> None:
        """F-002: load LoRA adapter weights from a previous run's checkpoint.

        ``checkpoint_path`` points at a directory created by
        ``Trainer.save()`` — a PEFT adapter directory with
        ``adapter_config.json`` + ``adapter_model.safetensors``. We
        re-attach those weights onto the current trainer's model via
        PEFT's ``load_adapter`` (when available) or by feeding the
        state dict through :meth:`_load_lora_state_dict`.
        """
        from pathlib import Path as _P

        cp_path = _P(checkpoint_path)
        if not cp_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {cp_path}")

        model = self._trainer._model
        if hasattr(model, "load_adapter"):
            # PEFT >= 0.7 provides load_adapter on PeftModel instances.
            try:
                model.load_adapter(str(cp_path), adapter_name="default", is_trainable=True)
                logger.info(f"Resumed LoRA weights from {cp_path} via PeftModel.load_adapter")
                return
            except Exception as exc:
                logger.debug(f"load_adapter path failed: {exc}; falling back to state-dict load")

        # Fallback: read the state dict and apply via existing helper.
        # BACKEND-A-001 fix: dispatch the loader by file extension. The prior
        # code called `safetensors.load_file()` on a .bin path whenever only
        # the .bin existed, which raises (safetensors loader cannot parse
        # torch pickles). Now we resolve which file exists first, then pick
        # the loader. The `except ImportError` clause is scoped to the
        # safetensors import line ONLY — it must not function as a generic
        # fallback for unrelated errors inside the loader call.
        try:
            from safetensors.torch import load_file  # type: ignore

            safetensors_available = True
        except ImportError:
            safetensors_available = False
            load_file = None  # type: ignore[assignment]

        adapter_file = cp_path / "adapter_model.safetensors"
        if not adapter_file.exists():
            adapter_file = cp_path / "adapter_model.bin"
        if not adapter_file.exists():
            raise FileNotFoundError(
                f"No adapter weights at {adapter_file.parent}"
            )

        if adapter_file.suffix == ".safetensors":
            if not safetensors_available:
                # Caller saved a .safetensors checkpoint but the resuming
                # environment does not have safetensors installed. We cannot
                # silently fall back to torch.load — the formats are not
                # interchangeable. Surface a clear error.
                raise ImportError(
                    "safetensors is required to load "
                    f"{adapter_file.name}; install with: pip install safetensors"
                )
            state_dict = load_file(str(adapter_file))  # type: ignore[misc]
        else:
            # .bin checkpoint — use torch.load regardless of whether
            # safetensors is installed.
            import torch

            # B614 + Ship Gate A: torch.load is unsafe by default (arbitrary
            # code execution via crafted pickle). weights_only=True is the
            # safe path and is what the rest of the codebase uses; this resume
            # site missed it during F-002. Adapter files are user-trusted
            # (the user produced them in a prior training run) but defense in
            # depth costs nothing here.
            state_dict = torch.load(adapter_file, map_location="cpu", weights_only=True)

        self._load_lora_state_dict(state_dict)
        logger.info(f"Resumed LoRA weights from {cp_path} via fallback state-dict load")

    def _collect_merge_history(self) -> list[dict[str, Any]]:
        """F-003 / F-004: collect serialisable SLAO merge_result dicts.

        Each ``RunResult.merge_result`` is a :class:`MergeResult` (or
        ``None``). We collapse the populated ones into dicts so they
        survive the JSON round-trip into ``run_history.json`` and the
        model card.
        """
        out: list[dict[str, Any]] = []
        for run in self._runs:
            mr = getattr(run, "merge_result", None)
            if mr is None:
                continue
            entry: dict[str, Any] = {"run_index": run.run_index}
            # MergeResult is a dataclass — defensively turn it into a dict
            # via asdict() if available, else iterate its public attrs.
            try:
                from dataclasses import asdict, is_dataclass

                if is_dataclass(mr):
                    entry["result"] = asdict(mr)  # type: ignore[arg-type]
                else:
                    entry["result"] = {
                        k: v for k, v in vars(mr).items()
                        if not k.startswith("_")
                    }
            except Exception:
                entry["result"] = str(mr)
            out.append(entry)
        return out

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
            # B-001: correlation token surfaced on the aggregate result so
            # callers can persist it alongside their own bookkeeping.
            run_id=self._run_id,
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
            run_id=self._run_id,
        )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main() -> None:
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
