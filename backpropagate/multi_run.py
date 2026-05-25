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
    CheckpointError,
    ConfigurationError,
    DatasetError,
    DatasetNotFoundError,
    InvalidSettingError,
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


def _build_abort_callback(trainer: "MultiRunTrainer") -> Any:
    """BACKEND-F-001: bridge ``MultiRunTrainer._should_abort`` into HF's
    TrainerCallback so the abort signal interrupts the inner
    ``SFTTrainer.train()`` call mid-step instead of only between runs.

    Pre-F-001 ``abort()`` set ``_should_abort=True`` and the next iteration
    of the inter-run loop honored it — but the SFTTrainer.train() call
    that was already in flight ran to completion (potentially hours)
    before the loop checked again. The GPU emergency handler
    (``_on_gpu_emergency``) calls ``abort()`` expecting fast-fail; without
    this callback that safety contract is silently broken.

    Mechanism: HF Trainer fires :meth:`on_step_end` after every gradient
    step. The callback polls ``trainer._should_abort`` and, when set,
    flips ``control.should_training_stop = True`` (the documented HF
    contract for cooperative cancellation). The inner SFTTrainer then
    exits its training loop cleanly at the next step boundary —
    typically within seconds — and ``_execute_run`` returns normally so
    the run-loop's ``if self._should_abort: break`` check at the top of
    :meth:`run` fires and the session tears down cleanly.

    Returns ``None`` when transformers is unavailable; the caller should
    skip callback installation in that case (defensive — transformers is
    a hard dep of TRL so this branch should never fire in practice).
    """
    try:
        from transformers import TrainerCallback as _HFTrainerCallback
    except Exception as exc:
        logger.debug(
            f"_build_abort_callback: transformers TrainerCallback "
            f"unavailable ({exc!r}); mid-run abort will fall back to "
            f"between-run-only semantics."
        )
        return None

    class _AbortCallback(_HFTrainerCallback):  # type: ignore[misc, valid-type]
        """HF callback that interrupts SFTTrainer.train() when _should_abort fires."""

        def __init__(self, multi_run_trainer: "MultiRunTrainer") -> None:
            super().__init__()
            self._mrt = multi_run_trainer

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002
            if self._mrt._should_abort:
                # Cooperative cancellation — HF Trainer reads this flag
                # after every step and exits its training loop cleanly.
                control.should_training_stop = True
                logger.warning(
                    f"AbortCallback: should_training_stop=True at "
                    f"global_step={getattr(state, 'global_step', '?')} "
                    f"(reason={self._mrt._abort_reason!r}); inner "
                    f"SFTTrainer will exit at the next step boundary."
                )
            return control

    return _AbortCallback(trainer)


def _build_multi_run_step_callback(
    trainer: "MultiRunTrainer", run_idx: int
) -> Any:
    """v1.4 BACKEND-F-003 (Wave 6b features): bridge MultiRunTrainer.on_step
    into the HF TrainerCallback API.

    Pre-Wave-6b, the ``on_step`` callback exposed on
    ``MultiRunTrainer.__init__`` was a dead surface — declared, stored on
    ``self.on_step``, but never invoked anywhere in ``_execute_run`` or the
    inner SFTTrainer. Operators wiring a per-step progress bar / live loss
    chart / external logger via on_step got a silent no-op for every step.

    The fix mirrors the single-run F-003 bridge pattern in
    ``trainer.py:_build_trl_bridge_callback``: a private TrainerCallback
    subclass that polls HF's ``on_log`` for the latest loss entry and
    forwards ``(run_idx, step, loss)`` to the user's
    :class:`MultiRunTrainer.on_step` callable. Same callback isolation
    contract — a buggy user callback is logged at WARN and training
    continues (no exception propagates out of the bridge).

    Returns ``None`` when transformers is unavailable OR when no on_step
    callback was supplied; the caller skips this entry in the callbacks
    list in that case.
    """
    if trainer.on_step is None:
        return None
    try:
        from transformers import TrainerCallback as _HFTrainerCallback
    except Exception as exc:
        logger.debug(
            f"_build_multi_run_step_callback: transformers TrainerCallback "
            f"unavailable ({exc!r}); on_step will not fire for run {run_idx}."
        )
        return None

    class _StepCallback(_HFTrainerCallback):  # type: ignore[misc, valid-type]
        """HF callback that forwards (run_idx, step, loss) to on_step."""

        def __init__(self, multi_run_trainer: "MultiRunTrainer", run_index: int) -> None:
            super().__init__()
            self._mrt = multi_run_trainer
            self._run_index = run_index

        def on_log(  # noqa: D401
            self,
            args: Any,  # noqa: ANN401, ARG002
            state: Any,  # noqa: ANN401
            control: Any,  # noqa: ANN401, ARG002
            logs: dict | None = None,
            **_kwargs: Any,
        ) -> None:
            on_step = self._mrt.on_step
            if on_step is None:
                return
            # Prefer the explicit ``logs`` dict, fall back to
            # ``state.log_history`` tail (older transformers builds).
            loss_val: float | None = None
            if isinstance(logs, dict) and "loss" in logs:
                try:
                    loss_val = float(logs["loss"])
                except (TypeError, ValueError):
                    loss_val = None
            if loss_val is None:
                history = getattr(state, "log_history", None) or []
                for entry in reversed(history[-5:]):
                    if isinstance(entry, dict) and "loss" in entry:
                        try:
                            loss_val = float(entry["loss"])
                            break
                        except (TypeError, ValueError):
                            continue
            if loss_val is None:
                # No usable loss in this log batch (e.g. eval-only entry).
                return
            step = int(getattr(state, "global_step", 0) or 0)
            try:
                on_step(self._run_index, step, loss_val)
            except Exception as cb_error:  # noqa: BLE001 — user callback isolation
                logger.warning(
                    f"on_step callback raised error "
                    f"(run_index={self._run_index} step={step} "
                    f"loss={loss_val:.4f}): {cb_error}"
                )

    return _StepCallback(trainer, run_idx)


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

    # Stage C BACKEND-B-003: OOM-recovery ceiling on gradient_accumulation.
    # The recovery loop halves batch_size and doubles gradient_accumulation to
    # preserve the effective batch size across an OOM retry. Without a ceiling,
    # an operator who pinned a specific accum value (because their LR schedule
    # or token-budget assumes accum<=N) has it silently overridden. When set
    # (>0), the recovery aborts with RUNTIME_OOM_RECOVERY_EXHAUSTED instead of
    # exceeding the ceiling. Default 0 = "no ceiling, current behavior."
    max_grad_accumulation: int = 0

    # Stage C amend BACKEND-B-002: maximum wall-clock seconds the run loop
    # waits for a GPU cooldown event before aborting. The pause polling loop
    # at multi_run.py blocks while ``_gpu_pause_event`` is set. Without a
    # ceiling, a wedged GPU (thermal-runaway sensor stuck high, daemon
    # crash, driver hang) blocks the run forever and the operator's only
    # escape is SIGINT — which is invisible in monitoring dashboards.
    # Default 1800.0 = 30 minutes; set 0.0 to disable the timeout entirely
    # (the prior unbounded behavior, preserved for operators who explicitly
    # opt out).
    max_pause_seconds: float = 1800.0

    # v1.4 BRIDGE-A-002 follow-up (Wave 6a foundation): per-invocation
    # overrides for the five Wave 6b knobs (Trainer constructor mirrors —
    # ``use_dora`` / ``packing`` / ``init_lora_weights`` / ``lora_preset`` /
    # ``optim``). ``cmd_multi_run`` (cli.py ~820) + ``cmd_replay`` (cli.py
    # ~4280) assemble a ``wave6b_candidate_kwargs`` dict and filter via
    # ``dataclasses.fields(MultiRunConfig)`` — pre-fix these five keys were
    # silently dropped because the dataclass did not name them. Now declared
    # as explicit fields with ``None`` defaults; the per-invocation override
    # is forwarded to the inner ``Trainer`` constructor in
    # ``MultiRunTrainer.__init__`` so the SFTConfig assembled by
    # ``_build_sft_config`` reads the per-invocation values from
    # ``self._trainer.packing`` / ``self._trainer.optim`` etc. When the
    # field is ``None``, the inner Trainer falls back to the settings layer
    # (``settings.lora.use_dora`` / ``settings.data.packing`` / etc.),
    # preserving pre-Wave-6a env-var-driven behavior byte-identically.
    use_dora: bool | None = None
    packing: bool | None = None
    init_lora_weights: str | None = None  # "default" | "pissa" | "loftq"
    lora_preset: str | None = None  # "fast" | "quality"
    optim: str | None = None  # passthrough to TrainingArguments.optim

    # v1.4 BACKEND-F-008 (Wave 6b features): training mode forwarded to the
    # inner Trainer instance. ``"lora"`` (default) preserves pre-Wave-6b
    # multi-run behavior byte-identically; ``"full"`` enables full
    # fine-tuning for models <=3B parameters. mode='full' on a model >3B
    # raises RUNTIME_FULL_FT_MODEL_TOO_LARGE at MultiRunTrainer
    # construction time via the inner Trainer's construction-time gate
    # (and again at the inner Trainer's load_model() time as belt-and-
    # braces). The full-FT optimizer + gradient_checkpointing + LR
    # divisor logic lives in the shared ``_build_sft_config`` helper so
    # multi-run inherits the same contract end-to-end.
    mode: str = "lora"


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
                structured error (code=RUNTIME_OOM_RECOVERY_EXHAUSTED — the
                recovery loop ran out of options). When False, the first
                uncovered OOM is recorded as ``run_failed=True`` with the
                ``RUNTIME_GPU_OOM`` code prefixed onto the failure_reason
                (Wave 6a RUNTIME_GPU_OOM Option A multi-run symmetric: the
                code surface that operators grep for IS produced — the
                contract carrier is ``run_result.failure_reason``, not a
                raise site, because the multi-run contract is "record +
                continue to the next run," not "session-abort"). Failed
                runs (run_failed=True) also skip the checkpoint save +
                manifest register branch (Wave 6a BACKEND-B-002) so a
                future resume cannot latch onto post-failure model state.
                Set False to make OOMs fail the individual run and continue.
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
        # Build config.
        #
        # Stage C amend BACKEND-B-010: when BOTH ``config`` and the
        # convenience kwarg name the same field with different values,
        # emit a single WARN line per conflict so the operator sees the
        # silent precedence rule (convenience kwarg wins). Pre-fix, a
        # user supplying ``config=MyConfig(num_runs=5)`` AND
        # ``num_runs=10`` got num_runs=10 with no signal — a refactor
        # that flipped one of the surfaces would change behavior
        # invisibly. The fix is "warn, don't raise" because v1.4 is the
        # promotion target for hard-fail semantics (per the recommended
        # path in the audit). The warn includes the override direction
        # so the operator can flip the call site cleanly.
        explicit_config_supplied = config is not None
        self.config = config or MultiRunConfig()

        def _warn_override(field: str, config_value: Any, kwarg_value: Any) -> None:
            if explicit_config_supplied and config_value != kwarg_value:
                logger.warning(
                    f"MultiRunTrainer: convenience kwarg {field}={kwarg_value!r} "
                    f"overrides config.{field}={config_value!r}. Pick ONE "
                    f"source for this field — kwargs take precedence today "
                    f"but v1.4 may promote this to a hard error."
                )

        if num_runs is not None:
            _warn_override("num_runs", self.config.num_runs, num_runs)
            self.config.num_runs = num_runs
        if steps_per_run is not None:
            _warn_override("steps_per_run", self.config.steps_per_run, steps_per_run)
            self.config.steps_per_run = steps_per_run
        if samples_per_run is not None:
            _warn_override("samples_per_run", self.config.samples_per_run, samples_per_run)
            self.config.samples_per_run = samples_per_run
        if checkpoint_dir is not None:
            _warn_override("checkpoint_dir", self.config.checkpoint_dir, checkpoint_dir)
            self.config.checkpoint_dir = checkpoint_dir
        if merge_mode is not None:
            if isinstance(merge_mode, str):
                try:
                    _resolved_mode = MergeMode(merge_mode.lower())
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid merge mode: '{merge_mode}'. Available: slao, simple"
                    ) from None
                _warn_override("merge_mode", self.config.merge_mode, _resolved_mode)
                self.config.merge_mode = _resolved_mode
            else:
                _warn_override("merge_mode", self.config.merge_mode, merge_mode)
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
        # BACKEND-A-004: True once _restore_session_state successfully loaded
        # SLAO merger state from disk. The run-loop uses this to decide
        # whether a subsequent _load_resume_checkpoint failure is a coherent-
        # state corruption (SLAO restored but model weights diverged from it)
        # vs a no-op (SLAO never restored, so falling back to the freshly
        # loaded base model is safe). When True and the checkpoint load
        # fails, the run aborts with CheckpointError instead of silently
        # continuing in a mismatched state.
        self._resume_slao_state_restored: bool = False

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

        # Stage C amend BACKEND-B-015: session-wide OOM counter. NEVER
        # resets — accumulates across the entire MultiRunTrainer.run()
        # call so the final summary log can report "session: N OOMs
        # survived across M runs" without losing signal when a clean
        # run between OOM cascades resets ``_oom_consecutive_at_min_batch``.
        self._session_oom_count: int = 0

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
                # BACKEND-F-018 (v1.3): mirror the Wave 5.5 BACKEND-F-002
                # fix on the single-run path. The operator passed
                # resume_from expecting resumption; falling back silently
                # to a fresh multi-run under a NEW run_id drops the
                # resume intent and consumes GPU hours producing a model
                # the operator did not ask for. Pre-F-018 this site
                # logged a WARN and returned None — easy to miss in a
                # multi-hour session log, and the silent-divergence
                # symptom would not surface until export / eval time.
                #
                # The error message names the requested run_id, the
                # checkpoint_dir actually searched, and the operator's
                # next steps so the failure is actionable in the
                # terminal without grep'ing logs. INPUT_RESUME_NOT_FOUND
                # was added to the error-codes catalog in Wave 5.5
                # alongside the single-run fix.
                raise InvalidSettingError(
                    setting_name="resume_from",
                    value=self._resume_from,
                    expected=(
                        f"a multi-run run_id present in the on-disk run "
                        f"history under checkpoint_dir="
                        f"{str(checkpoint_dir)!r}"
                    ),
                    suggestion=(
                        f"resume_from={self._resume_from!r} was NOT found "
                        f"in the multi-run history at "
                        f"{str(checkpoint_dir)!r}. The resume lookup is "
                        f"scoped to the configured checkpoint_dir (not a "
                        f"global run_id index) and only matches sessions "
                        f"recorded with session_kind='multi_run'. Next "
                        f"steps:\n"
                        f"  1) Run `backprop runs` to list run_ids "
                        f"available under this checkpoint_dir.\n"
                        f"  2) If the multi-run was trained under a "
                        f"different checkpoint_dir, re-run with "
                        f"`--checkpoint-dir <that-dir>` so the lookup "
                        f"hits.\n"
                        f"  3) If the run_id belongs to a single-run "
                        f"session, use the single-run resume path "
                        f"(`Trainer.train(resume_from=...)`) instead — "
                        f"multi-run resume only picks up sessions saved "
                        f"by another MultiRunTrainer.\n"
                        f"  4) To start truly fresh under a NEW run_id, "
                        f"omit resume_from (or pass resume_from='off')."
                    ),
                )
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
                    # BACKEND-A-004: mark SLAO state as restored so the
                    # run-loop knows a subsequent checkpoint-load failure
                    # leaves the merger and model in inconsistent states.
                    self._resume_slao_state_restored = True
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

        # BACKEND-A-001: fail fast on the silent-disable trap where
        # ``early_stopping=True`` is combined with ``validate_every_run=False``.
        # The downstream code at the run-execute site (this file's run-loop)
        # only wraps the run with the validation harness when
        # ``validate_every_run or early_stopping`` is truthy, but the
        # ``early_stopping`` branch needs a ``val_loss`` to compare against —
        # which only the validation harness computes. The prior behaviour
        # reserved a 10% validation holdout, ran the validation pass, and then
        # quietly suppressed the early-stopping check because the loop only
        # called ``_check_early_stopping`` when ``val_loss is not None`` (it
        # would be ``None`` whenever ``validate_every_run`` was False because
        # ``_execute_run_with_validation`` keys the validation compute on
        # ``validate_every_run`` alone). Operators got the dataset-shrink cost
        # without the early-stopping benefit. Surface the inconsistency now,
        # before any model/data work happens.
        if self.config.early_stopping and not self.config.validate_every_run:
            raise InvalidSettingError(
                setting_name="early_stopping",
                value=True,
                expected="validate_every_run=True (early stopping requires a per-run validation loss)",
                suggestion=(
                    "Either enable validation (set validate_every_run=True) or "
                    "disable early stopping (set early_stopping=False)."
                ),
            )

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
            # v1.4 BRIDGE-A-002 follow-up (Wave 6a): forward the Wave 6b
            # overrides from MultiRunConfig fields to the inner Trainer
            # constructor. Each field defaults to ``None`` on the dataclass;
            # when None the Trainer constructor falls back to
            # ``settings.lora.*`` / ``settings.data.*`` / ``settings.training.*``
            # (env-var driven). When set, the per-invocation override flows
            # through to ``_build_sft_config`` via the inner Trainer's
            # instance attributes (``self._trainer.packing``,
            # ``self._trainer.optim``) so the multi-run path threads the
            # operator's flag value end-to-end (matches the cmd_multi_run
            # CLI introspection-filter contract).
            self._trainer = Trainer(
                model=self.model_name,
                learning_rate=self.config.initial_lr,
                train_on_responses=True,  # FT-010: same quality optimization as single-run
                report_to=self._report_to_intent,  # F-005
                use_dora=self.config.use_dora,
                packing=self.config.packing,
                init_lora_weights=self.config.init_lora_weights,
                lora_preset=self.config.lora_preset,
                optim=self.config.optim,
                # v1.4 BACKEND-F-008 (Wave 6b features): forward the multi-run
                # mode setting to the inner Trainer. mode='full' on a model
                # >3B raises RUNTIME_FULL_FT_MODEL_TOO_LARGE here (Trainer
                # construction); mode='lora' (default) preserves byte-
                # identical pre-Wave-6b multi-run behavior.
                mode=self.config.mode,
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
                #
                # BACKEND-A-004: this load is part of a two-step transaction
                # with the SLAO-merger restore that _restore_session_state
                # performed earlier. If we let the LoRA-weights load fail
                # silently after the SLAO state was already swapped in, the
                # merger's accumulated A/B matrices reference an adapter
                # geometry that the freshly-loaded base model does not
                # carry — every subsequent merge then operates on a
                # mismatched pair and the resumed run silently diverges
                # from the prior session. Two acceptable resolutions:
                #   (a) SLAO never restored ⇒ falling back to the fresh
                #       base model is coherent (just slower convergence on
                #       the first resumed run). Warn and continue.
                #   (b) SLAO restored ⇒ the merger has prior-run state but
                #       the model would not. Abort the resume with
                #       CheckpointError so the operator can re-run from a
                #       known-good checkpoint, rather than burn GPU hours
                #       producing silently corrupted output.
                if self._resume_checkpoint_path:
                    try:
                        self._load_resume_checkpoint(self._resume_checkpoint_path)
                    except Exception as exc:
                        if self._resume_slao_state_restored:
                            logger.error(
                                "Resume aborted: SLAO merger state was "
                                f"restored from a prior session but the LoRA "
                                f"checkpoint at {self._resume_checkpoint_path} "
                                f"failed to load ({exc}). The merger and "
                                "model are now in inconsistent states; "
                                "continuing would silently produce corrupted "
                                "merges across runs. Re-run with "
                                "resume_from='off' to start fresh, or "
                                "supply a checkpoint that pairs with the "
                                "persisted SLAO state."
                            )
                            raise CheckpointError(
                                operation="load",
                                path=str(self._resume_checkpoint_path),
                                reason=(
                                    f"resume aborted: SLAO merger state was "
                                    f"already restored but the paired LoRA "
                                    f"checkpoint load failed ({exc}); "
                                    "merger/model state would be inconsistent. "
                                    "Re-run with resume_from='off' to start "
                                    "fresh."
                                ),
                            ) from exc
                        logger.warning(
                            f"Failed to load resume checkpoint "
                            f"{self._resume_checkpoint_path}: {exc}. "
                            "SLAO state was not restored either, so the "
                            "merger will re-initialize on the first resumed "
                            "run; results may diverge from the original "
                            "session but the merger/model pair stays "
                            "coherent."
                        )
            for run_idx in range(start_idx, self.config.num_runs + 1):
                if self._should_abort:
                    break

                # Check if GPU pause is active (B-020: overheat protection).
                # The monitor thread sets the event when overheating and clears it
                # when the GPU is safe again. We poll here so the main thread blocks.
                #
                # Stage C amend BACKEND-B-002: bound the wait so a wedged GPU
                # (sensor failure, daemon crash, thermal runaway with no
                # recovery) doesn't block the run forever. Operator's escape
                # was previously SIGINT; now a humanized error explains the
                # condition and points at the lever (raise / disable the
                # ceiling, or fix cooling).
                if self._gpu_pause_event.is_set():
                    logger.info("Waiting for GPU cooldown before starting run...")
                    pause_started = time.time()
                    pause_ceiling = float(getattr(self.config, "max_pause_seconds", 0.0) or 0.0)
                    while self._gpu_pause_event.is_set() and not self._should_abort:
                        if pause_ceiling > 0.0 and (time.time() - pause_started) > pause_ceiling:
                            elapsed = time.time() - pause_started
                            logger.error(
                                f"GPU pause exceeded max_pause_seconds="
                                f"{pause_ceiling:.0f}s (elapsed={elapsed:.0f}s, "
                                f"run_id={self._run_id}, next run_index={run_idx}); "
                                f"the cooldown event never cleared — likely a "
                                f"wedged sensor, daemon crash, or thermal runaway "
                                f"with no recovery. Aborting cleanly."
                            )
                            raise BackpropagateError(
                                f"GPU cooldown wait exceeded max_pause_seconds="
                                f"{pause_ceiling:.0f}s and the pause event never "
                                f"cleared. The run loop will not proceed against "
                                f"a GPU the monitor cannot confirm is safe.",
                                code="RUNTIME_GPU_TEMPERATURE_CRITICAL",
                                details={
                                    "run_id": self._run_id,
                                    "next_run_index": run_idx,
                                    "elapsed_seconds": elapsed,
                                    "max_pause_seconds": pause_ceiling,
                                },
                                suggestion=(
                                    "Verify GPU cooling (nvidia-smi to confirm "
                                    "temperature has actually recovered); if the "
                                    "sensor is reporting bad data, restart the "
                                    "GPU driver / nvidia-persistenced. To disable "
                                    "the timeout entirely (prior unbounded "
                                    "behavior), set max_pause_seconds=0 in "
                                    "MultiRunConfig. To accept longer cooldowns, "
                                    "raise the value (default 1800s = 30 min)."
                                ),
                                retryable=False,
                            )
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

        # Stage C amend BACKEND-B-015: session-wide OOM summary. Surfaces
        # the cumulative count even when each individual cascade was
        # survived (so the per-cascade counter reset zero'd out the
        # signal). Useful for post-mortem "did this session hit VRAM
        # pressure?" diagnostics.
        if self._session_oom_count > 0:
            logger.info(
                f"session_oom_summary run_id={self._run_id} "
                f"total_oom_events={self._session_oom_count} "
                f"runs_completed={len(self._runs)} "
                f"(consider smaller model / shorter max_seq_length / "
                f"4-bit quantization if this number is high for next session)"
            )

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
        """Request abort of current multi-run session.

        BACKEND-F-001: as of v1.3 the abort signal is honored BOTH:

        - Mid-run, via the ``_AbortCallback`` wired into the inner
          ``SFTTrainer.train()`` call. HF Trainer fires ``on_step_end``
          after every gradient step; when ``_should_abort`` is set the
          callback flips ``control.should_training_stop = True`` and the
          inner training loop exits cleanly at the next step boundary
          (typically within seconds — bounded by ``logging_steps`` and
          dataloader-batch granularity).
        - Between runs, via the ``if self._should_abort: break`` check
          at the top of the per-run loop in :meth:`run`.

        This is the safety contract the GPU emergency handler
        (``_on_gpu_emergency``) relies on. Pre-F-001 the in-flight
        ``SFTTrainer.train()`` call ran to completion before the
        between-run check fired — potentially hours of wasted GPU after a
        thermal emergency. The mid-run interruption closes that gap.

        Operators calling ``abort()`` from a UI button, signal handler,
        or programmatic safety check can rely on the call resolving in
        seconds, not hours, regardless of whether a run is in flight.
        Note that mid-run abort relies on transformers being importable;
        on a malformed environment where the callback cannot install
        (logged at DEBUG) the behavior gracefully degrades to the prior
        between-run-only semantics.
        """
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

    # Stage C BACKEND-B-002: substrings PyTorch (2.0-2.6) and CUDA libraries
    # surface for OOM-adjacent failures. Matching these widens the recovery
    # net beyond strict ``torch.cuda.OutOfMemoryError`` so a CUBLAS alloc
    # failure or a CUDNN init failure (which is almost always a downstream
    # symptom of VRAM exhaustion) routes through the same halve-batch loop
    # instead of falling through as an opaque generic failure.
    _OOM_ADJACENT_SUBSTRINGS: tuple[str, ...] = (
        "out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_not_initialized",
        "cuda error: out of memory",
        "memory_allocator",
        "cuda out of memory",
    )

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

    @classmethod
    def _is_oom_adjacent(cls, exc: BaseException) -> tuple[bool, str | None]:
        """Stage C BACKEND-B-002: widened OOM detection.

        Returns ``(matched, substring_that_matched)``. Matches:
        - any ``torch.cuda.OutOfMemoryError`` (delegates to strict matcher)
        - any ``RuntimeError`` whose ``str(exc)`` contains a known OOM-adjacent
          marker (CUBLAS_STATUS_ALLOC_FAILED, CUDNN_STATUS_NOT_INITIALIZED,
          NCCL communication failures that follow VRAM exhaustion, etc.)

        This is the **observability + recovery** lens on top of ``_is_oom_error``.
        Callers route adjacent matches through the same halve-batch recovery
        path AND emit a structured WARN log so operators can see which
        signature triggered. If a future PyTorch release changes the wording,
        the catalog at :data:`_OOM_ADJACENT_SUBSTRINGS` is the single edit
        site — no scattered if/elif chains to chase.
        """
        if cls._is_oom_error(exc):
            return True, "out of memory"
        if isinstance(exc, RuntimeError):
            msg_lower = str(exc).lower()
            for marker in cls._OOM_ADJACENT_SUBSTRINGS:
                if marker in msg_lower:
                    return True, marker
        return False, None

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

        Wave 6a BACKEND-A-003 / A-004 / B-002 + RUNTIME_GPU_OOM Option A:
        the SFTConfig + train_on_responses_only application path delegates to
        the shared :func:`_build_sft_config` / :func:`_apply_train_on_responses_only`
        helpers in trainer.py so the multi-run path picks up the same v1.3
        paged-optim / Ada bf16/fp16 autodetection + Unsloth response-masking
        as the single-run path. Failed runs skip the checkpoint save +
        manifest register branch (B-002). The oom_recovery=False raise site
        wraps the OOM as ``GPUMemoryError(code='RUNTIME_GPU_OOM')`` so the
        documented contract holds.
        """
        from trl import SFTTrainer

        from .trainer import (
            _apply_train_on_responses_only,
            _build_sft_config,
        )

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

        # Pre-tokenize for Windows.
        #
        # Stage C BACKEND-B-009: log the OS-conditional decision once so a
        # post-mortem comparing a Windows multi-run vs a Linux multi-run
        # can confirm "both saw the same tokenization path" via one grep
        # instead of an artifact-diff. The loss curve shape on the same
        # dataset CAN differ between OSes here (pre-tokenized chunk vs
        # tokenize-on-the-fly inside SFTTrainer), and operators reproducing
        # a Windows-only loss anomaly need this breadcrumb to triage.
        if os.name == "nt" and settings.windows.pre_tokenize:
            logger.info(
                "Pre-tokenization: applied (os.name=nt, windows.pre_tokenize=True) "
                "for chunk of %d samples (run_idx=%d, run_id=%s)",
                len(chunk_dataset),
                run_idx,
                self._run_id,
            )
            chunk_dataset = self._trainer._pre_tokenize(chunk_dataset)
        else:
            logger.info(
                "Pre-tokenization: deferred to SFTTrainer "
                "(os.name=%s, windows.pre_tokenize=%s) for chunk of %d samples "
                "(run_idx=%d, run_id=%s)",
                os.name,
                settings.windows.pre_tokenize,
                len(chunk_dataset),
                run_idx,
                self._run_id,
            )

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
            # Wave 6a BACKEND-A-003: SFTConfig assembly is delegated to the
            # shared :func:`_build_sft_config` helper in trainer.py so the
            # multi-run path inherits the same v1.3 BACKEND-5 paged-optim
            # autodetection + BACKEND-7 Ada bf16/fp16 selection that
            # ``Trainer.train()`` applies. Pre-Wave-6a, this site built
            # SFTConfig inline with ``optim=settings.training.optim`` and
            # ``bf16=settings.training.bf16`` raw — bypassing both detectors.
            # Per-run + per-retry-only fields (learning_rate, warmup_steps,
            # seed offset, lr_scheduler_type="cosine", logging_steps=10) are
            # threaded explicitly so the multi-run-specific behavior is
            # preserved.
            #
            # Stage C BACKEND-B-003: per-retry seed offset. When an OOM
            # retry fires, ``oom_retries > 0`` and the seed shifts so the
            # per-batch shuffle order changes — an outlier-long sample that
            # triggered the OOM at position N no longer lands at position N
            # on the next attempt. The offset is deterministic given
            # (run_idx, oom_retries) so the retry is itself reproducible.
            training_args = _build_sft_config(
                output_dir=str(checkpoint_dir / f"run_{run_idx:03d}"),
                per_device_train_batch_size=self._trainer.batch_size,
                gradient_accumulation_steps=self._trainer.gradient_accumulation,
                max_steps=self.config.steps_per_run,
                learning_rate=lr,
                warmup_steps=self.config.warmup_steps_per_run,
                max_seq_length=self._trainer.max_seq_length,
                seed=settings.training.seed + run_idx + oom_retries * 1000,
                lr_scheduler_type="cosine",
                logging_steps=10,
                report_to=report_to_resolved,  # F-005
                run_name=run_name,
                # v1.4 BRIDGE-A-002 follow-up (Wave 6a): thread the
                # constructor-resolved per-invocation overrides from the
                # inner Trainer instance through to the helper, mirroring
                # the single-run train() call site. Pre-fix the multi-run
                # path read ``settings.data.packing`` / ``settings.training.optim``
                # directly, silently bypassing the per-invocation overrides
                # that the CLI introspection filter (cmd_multi_run /
                # cmd_replay) now passes through.
                packing=self._trainer.packing,
                optim=self._trainer.optim,
                # v1.4 BACKEND-F-008 (Wave 6b features): thread the
                # multi-run-configured mode (default ``"lora"``) through to
                # the helper. Single source of truth: the inner Trainer's
                # ``self.mode`` instance attribute (mirrors the
                # ``packing`` / ``optim`` threading pattern above).
                mode=self._trainer.mode,
            )

            # BACKEND-F-001: wire the abort callback into the inner
            # SFTTrainer so MultiRunTrainer.abort() interrupts mid-run
            # via control.should_training_stop, not just between runs.
            # Rebuilt per-attempt so the OOM retry path picks up a fresh
            # adapter bound to the new SFTTrainer instance.
            #
            # v1.4 BACKEND-F-003 (Wave 6b features): also wire the
            # per-step on_step callback bridge. Pre-Wave-6b the
            # ``on_step`` surface was a dead callback (declared on
            # MultiRunTrainer.__init__, stored on self.on_step, but never
            # invoked). The bridge polls HF on_log for the latest loss
            # entry and forwards (run_idx, step, loss) to the user
            # callable; returns None when no on_step is configured (so
            # callers without the callback see byte-identical behavior).
            _abort_cb = _build_abort_callback(self)
            _step_cb = _build_multi_run_step_callback(self, run_idx)
            _bridge_callbacks: list[Any] = [
                cb for cb in (_abort_cb, _step_cb) if cb is not None
            ]
            sft_callbacks = _bridge_callbacks if _bridge_callbacks else None

            trainer = SFTTrainer(
                model=self._trainer._model,
                processing_class=self._trainer._tokenizer,
                train_dataset=chunk_dataset,
                args=training_args,
                callbacks=sft_callbacks,
            )

            # Wave 6a BACKEND-A-004: apply Unsloth's train_on_responses_only
            # masking before training. Pre-Wave-6a, this site silently
            # skipped the masking step despite the docstring claim at
            # multi_run.py:893 (``train_on_responses=True``) — multi-run
            # users training on conversational data got full-conversation
            # loss leakage. The shared helper centralizes the
            # Windows-skip / Unsloth-missing / detection-failure paths so
            # Trainer.train() and this site stay byte-equivalent on
            # masking behavior.
            trainer, _resolved_markers = _apply_train_on_responses_only(
                trainer,
                self._trainer._tokenizer,
                enabled=self._trainer._train_on_responses,
                use_unsloth=self._trainer.use_unsloth,
                response_markers_override=self._trainer._response_markers_override,
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

                # Stage C amend BACKEND-B-005: SLAO merge moved OUTSIDE
                # this try block (below the loop break). The wrapping was
                # too wide pre-fix — a SLAO_MERGE_DIVERGED exception would
                # propagate through this except handler and a future
                # SLAO error message that happens to contain "memory" or
                # any other OOM-adjacent substring would be misclassified
                # as an OOM and trigger a pointless batch-halving retry.
                # Moving the merge below the break narrows the try-block
                # scope to just trainer.train() + loss extraction.
                break  # Successful run — exit retry loop

            except (KeyboardInterrupt, SystemExit):
                # Never swallow these — let them propagate.
                raise
            except Exception as exc:
                # Stage C BACKEND-B-002: widen OOM detection beyond the strict
                # ``torch.cuda.OutOfMemoryError`` matcher. CUBLAS_STATUS_ALLOC_FAILED
                # and CUDNN_STATUS_NOT_INITIALIZED are almost always downstream
                # symptoms of VRAM exhaustion — route them through the same
                # halve-batch recovery and emit a structured WARN so operators
                # can see which signature fired (and so we can grow the matcher
                # when PyTorch changes wording in a future release).
                adjacent_matched, adjacent_marker = self._is_oom_adjacent(exc)
                strict_oom = self._is_oom_error(exc)
                if adjacent_matched and not strict_oom:
                    logger.warning(
                        f"oom-adjacent error intercepted by recovery path: "
                        f"type={type(exc).__name__} marker={adjacent_marker!r} "
                        f"run_id={self._run_id} run_index={run_idx} "
                        f"code=RUNTIME_OOM_ADJACENT message={str(exc)[:240]!r}"
                    )
                if self.oom_recovery and adjacent_matched:
                    # B-002: OOM-specific recovery path.
                    import torch as _torch

                    current_batch = self._trainer.batch_size
                    current_accum = self._trainer.gradient_accumulation
                    effective = current_batch * current_accum
                    # Stage C humanization: structured event fields so
                    # operators can grep `event=oom_recovery_` across a
                    # session to correlate the start / adjust / exhaust
                    # phases. [[grep-all-instances]] — same shape as the
                    # parallel trainer.py:1914 site.
                    logger.warning(
                        "event=oom_recovery_started run_id=%s run_index=%d "
                        "attempt=%d batch_size=%d grad_accum=%d "
                        "effective_batch=%d exc=%s",
                        self._run_id,
                        run_idx,
                        oom_retries + 1,
                        current_batch,
                        current_accum,
                        effective,
                        exc,
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

                        # Stage C BACKEND-B-003: honor max_grad_accumulation
                        # ceiling. If the new combination would exceed the
                        # operator's pinned ceiling, abort cleanly instead of
                        # silently overriding (which can wreck an LR schedule
                        # that depends on token-budget assumptions).
                        ceiling = getattr(self.config, "max_grad_accumulation", 0) or 0
                        if ceiling > 0 and new_accum > ceiling:
                            logger.error(
                                f"OOM recovery: would set grad_accumulation to "
                                f"{new_accum} but max_grad_accumulation ceiling "
                                f"is {ceiling}. Aborting cleanly."
                            )
                            raise BackpropagateError(
                                f"OOM recovery exhausted: halving would exceed "
                                f"max_grad_accumulation={ceiling} (next step "
                                f"would be {new_accum}); preserving the operator's "
                                f"pinned ceiling.",
                                code="RUNTIME_OOM_RECOVERY_EXHAUSTED",
                                details={
                                    "run_index": run_idx,
                                    "run_id": self._run_id,
                                    "attempted_grad_accumulation": new_accum,
                                    "max_grad_accumulation": ceiling,
                                    "current_batch_size": current_batch,
                                    "current_grad_accumulation": current_accum,
                                    "oom_retries": oom_retries,
                                },
                                suggestion=(
                                    "Raise max_grad_accumulation in MultiRunConfig "
                                    "if your LR schedule tolerates a larger "
                                    "effective batch, OR use a smaller model / "
                                    "shorter max_seq_length / 4-bit quantization "
                                    "so the original batch_size fits in VRAM."
                                ),
                                retryable=False,
                                cause=exc,
                            ) from exc

                        # Stage C humanization: structured event with the
                        # effective-batch invariant called out explicitly.
                        # Parallel to trainer.py:1926. [[grep-all-instances]].
                        logger.warning(
                            "event=oom_recovery_adjust run_id=%s run_index=%d "
                            "attempt=%d batch_size=%d->%d grad_accum=%d->%d "
                            "effective_batch=%d (preserved). "
                            "Retrying with halved batch + doubled accumulation; "
                            "no operator action required.",
                            self._run_id,
                            run_idx,
                            oom_retries + 1,
                            current_batch, new_batch,
                            current_accum, new_accum,
                            new_batch * new_accum,
                        )
                        self._trainer.batch_size = new_batch
                        self._trainer.gradient_accumulation = new_accum
                        self._oom_consecutive_at_min_batch = 0
                        oom_retries += 1
                        # Stage C amend BACKEND-B-015: track session-wide
                        # OOM count for the post-mortem summary line.
                        self._session_oom_count += 1
                        continue  # Retry with new args

                    # Already at min batch — count toward the abort threshold.
                    self._oom_consecutive_at_min_batch += 1
                    oom_retries += 1
                    # Stage C amend BACKEND-B-015: track session-wide
                    # OOM count even when the retry budget is exhausted.
                    self._session_oom_count += 1
                    # Stage C humanization: keep the structured event
                    # field shape consistent. [[grep-all-instances]].
                    logger.error(
                        "event=oom_recovery_at_floor run_id=%s run_index=%d "
                        "attempt=%d consecutive_at_min_batch=%d/%d "
                        "session_oom_count=%d. "
                        "batch_size is already 1; we cannot halve further. "
                        "If %d more consecutive OOMs hit, recovery aborts "
                        "with RUNTIME_OOM_RECOVERY_EXHAUSTED.",
                        self._run_id,
                        run_idx,
                        oom_retries,
                        self._oom_consecutive_at_min_batch,
                        self._OOM_MAX_RETRIES_AT_MIN_BATCH,
                        self._session_oom_count,
                        max(0, self._OOM_MAX_RETRIES_AT_MIN_BATCH - self._oom_consecutive_at_min_batch),
                    )
                    if self._oom_consecutive_at_min_batch >= self._OOM_MAX_RETRIES_AT_MIN_BATCH:
                        # Stage C BACKEND-B-003: use RUNTIME_OOM_RECOVERY_EXHAUSTED
                        # to distinguish "we hit our recovery limit" from a single
                        # one-shot OOM that the strict matcher caught. The
                        # operator action is the same (smaller model / shorter
                        # seq) but the code carries the semantic signal that
                        # the recovery loop ran and lost.
                        raise BackpropagateError(
                            f"Persistent CUDA OOM at batch_size=1 "
                            f"({self._oom_consecutive_at_min_batch} consecutive "
                            f"OOMs across runs); cannot recover automatically.",
                            code="RUNTIME_OOM_RECOVERY_EXHAUSTED",
                            details={
                                "run_index": run_idx,
                                "run_id": self._run_id,
                                "consecutive_oom_at_min_batch": self._oom_consecutive_at_min_batch,
                                "oom_retries": oom_retries,
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
                #
                # Wave 6a RUNTIME_GPU_OOM Option A (multi-run symmetric): if
                # the failure was an OOM (strict or adjacent) AND the operator
                # opted out of recovery, prefix the failure_reason with the
                # structured ``RUNTIME_GPU_OOM`` code so post-mortem log greps
                # find the documented contract surface. We deliberately do
                # NOT raise here — the multi-run contract documents OOM-with-
                # oom_recovery=False as a per-run failure that records into
                # the RunResult and continues to the next run (see
                # MultiRunTrainer.__init__ docstring). The symmetric
                # raise-as-GPUMemoryError lives in Trainer.train() because
                # the single-run contract IS "raise on OOM"; this site's
                # contract is "record + continue." Both routes now name
                # RUNTIME_GPU_OOM in their structured output.
                run_failed = True
                is_oom_no_recovery = (
                    not self.oom_recovery
                    and (strict_oom or adjacent_matched)
                )
                if is_oom_no_recovery:
                    failure_reason = (
                        f"RUNTIME_GPU_OOM: {type(exc).__name__}: {exc}"
                    )
                else:
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

        # Stage C amend BACKEND-B-005: SLAO merge OUTSIDE the OOM-retry
        # try block. Pre-fix, a SLAO_MERGE_DIVERGED exception propagating
        # out of the inner try could be misclassified by the OOM matcher
        # (any future error message that happens to contain an OOM-
        # adjacent substring would trigger a batch-halving retry on a
        # divergence event — wrong recovery, wasted retries). Now the
        # merge runs after the loop, with bookend log lines for triage,
        # and its exceptions propagate cleanly out of _execute_run to
        # the caller. Skip the merge on failed runs — merging against a
        # broken/empty LoRA state would either produce nonsense or
        # propagate a misleading error.
        merge_result = None
        if not run_failed and self.config.merge_mode == MergeMode.SLAO and self._slao_merger:
            logger.info(
                f"merge_started run_id={self._run_id} run_index={run_idx} "
                f"merge_mode=slao"
            )
            try:
                lora_state = self._get_lora_state_dict()
                merge_result = self._slao_merger.merge(
                    lora_state,
                    run_index=run_idx,
                    run_id=self._run_id,
                )
                logger.info(
                    f"merge_complete run_id={self._run_id} run_index={run_idx} "
                    f"a_merged={merge_result.a_matrices_merged} "
                    f"b_merged={merge_result.b_matrices_merged}"
                )
                # Stage C amend BACKEND-B-015: only reset the OOM
                # consecutive-at-min-batch counter AFTER both train AND
                # merge have succeeded. Pre-fix the reset happened
                # inside the try block at the success path, so if the
                # merge then raised, the next OOM cascade started with
                # a clean counter and the cumulative "how many failures
                # have we accumulated this session?" signal was lost.
                self._oom_consecutive_at_min_batch = 0
            except Exception as merge_exc:
                logger.error(
                    f"merge_failed run_id={self._run_id} run_index={run_idx} "
                    f"exc={type(merge_exc).__name__}: {merge_exc}"
                )
                raise
        elif not run_failed:
            # No-merge run (simple mode or no merger). Reset the floor
            # counter at the equivalent success boundary so behavior
            # matches the SLAO path.
            self._oom_consecutive_at_min_batch = 0
        # NOTE on the run_failed path: we deliberately DO NOT reset the
        # floor counter here. _session_oom_count below records the
        # session-wide signal; the per-cascade counter is reset only on
        # a true success so cascading failures across runs accumulate
        # correctly through the abort threshold.

        # Save checkpoint.
        #
        # Wave 6a BACKEND-B-002: gate the save+register branch on
        # ``not run_failed`` in addition to ``save_every_run``. Pre-Wave-6a
        # the branch was gated on ``self.config.save_every_run`` alone, so a
        # failed run (OOM at floor / non-OOM exception) still wrote its
        # post-failure model state to disk AND registered it with the
        # CheckpointManager — a subsequent resume could then latch onto the
        # broken state. The docstring above (line 1683) used to say "even on
        # failure — preserves partial work" but the manifest registration
        # makes the resume-target semantics wrong: the post-failure adapter
        # is NOT a valid resume candidate (an OOM may have left the optimizer
        # state inconsistent with the model state; a non-OOM exception left
        # the inner SFTTrainer in an unspecified state). Skipping the save
        # on failure preserves the contract "manifest entries are resume-safe."
        # Operators who genuinely want to inspect a failed run's partial
        # state can still call ``self._trainer.save(...)`` directly from a
        # post-run callback — the multi-run loop just stops registering
        # broken states as resume candidates.
        if self.config.save_every_run and not run_failed:
            try:
                # B-006: atomic write. Trainer.save() now writes into
                # <path>.partial and shutil.move()s it into place on success;
                # the partial dir is removed if anything raises mid-write.
                checkpoint_path = str(checkpoint_dir / f"run_{run_idx:03d}" / "lora")
                # BACKEND-F-007 (Wave 6a): opt out of Trainer.save()'s
                # auto-manifest registration here — multi-run owns its own
                # CheckpointManager at ``checkpoint_dir`` (per-run indices,
                # validation_loss-aware pruning) which we register against
                # explicitly a few lines below. The Trainer-level manifest
                # write would pollute self._trainer.output_dir/manifest.json
                # with a series of run_index=0 entries that the multi-run
                # resume path never consults; opting out keeps the canonical
                # multi-run manifest at checkpoint_dir as the single source
                # of truth for resume-candidate lookup.
                self._trainer.save(
                    checkpoint_path,
                    run_id=self._run_id,
                    register_in_manifest=False,
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
                # Stage C BACKEND-B-012 humanization: the prior log was a
                # bare ERROR line. Operators watching `on_run_complete` saw
                # a successful train+merge with checkpoint_path=None and
                # had no way to distinguish "save was never attempted
                # (save_every_run=False)" from "save attempted but failed."
                # Surface run_id, the path we attempted, and the
                # operator-actionable next step in a single structured
                # line. We deliberately do NOT raise (save failures are
                # observability concerns, not training-aborts — the run
                # itself succeeded), but the line shape gives triage tools
                # something to grep on.
                logger.error(
                    "Failed to save checkpoint for run %d (run_id=%s, "
                    "attempted_path=%s): %s. The training run itself "
                    "completed and the in-memory model is intact; only "
                    "the on-disk checkpoint is missing. To recover the "
                    "current weights, call trainer.save(<path>) on the "
                    "MultiRunTrainer's underlying ._trainer once the "
                    "session completes; to retry per-run saves, verify "
                    "free disk space and write permission on "
                    "checkpoint_dir, then re-run from this run_index.",
                    run_idx,
                    self._run_id,
                    checkpoint_path,
                    save_err,
                )
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

        Stage C amend BACKEND-B-001: failed runs (``run_result.failed=True``,
        e.g. after exhausted OOM recovery) leave the model in an indeterminate
        state — its weights may be partially-restored from the pre-OOM
        snapshot, may be the freshly-OOMed copy, and the per-run SFTTrainer
        was del'd in the failure cleanup. Computing validation loss against
        that model yields a value that's either NaN/inf or simply
        non-comparable to the prior runs' val_losses; either way, threading
        it into ``_best_val_loss`` / early-stop comparisons poisons the rest
        of the session. The skip is explicit: ``validation_loss=None`` on
        failed runs, and a humanized log line explains why.

        Stage C amend BACKEND-B-001: NaN/inf guard on the val_loss itself.
        If a validation forward pass returns a non-finite loss (e.g. mixed-
        precision overflow on a recovered model), surface it but DO NOT let
        it overwrite ``_best_val_loss`` — the next comparison would always
        produce NaN and early-stop would never fire again.

        Returns:
            Tuple of (RunResult, validation_loss or None)
        """
        import math as _math

        run_result = self._execute_run(run_idx, full_dataset, checkpoint_dir)

        # Compute validation loss if enabled
        val_loss = None
        if self.config.validate_every_run:
            if run_result.failed:
                # Stage C amend BACKEND-B-001: skip validation for failed runs.
                # The model state is suspect; computing val_loss against it
                # would produce a number that can never be meaningfully
                # compared to subsequent runs' losses. Operators see this in
                # the run record as ``validation_loss=None`` and in the log
                # as the explanatory warning below.
                logger.warning(
                    f"Run {run_idx} failed ({run_result.failure_reason}); "
                    f"skipping validation loss computation — model state "
                    f"after failure is indeterminate and feeding val_loss "
                    f"into early-stop / checkpoint scoring would poison "
                    f"subsequent comparisons. validation_loss=None for "
                    f"run_id={self._run_id}."
                )
                run_result.validation_loss = None
                return run_result, None

            val_loss = self._compute_validation_loss(full_dataset, run_idx)

            # Stage C amend BACKEND-B-001: NaN/inf guard. A non-finite
            # val_loss (mixed-precision overflow, divergent gradient, etc.)
            # surfaces here. Do NOT propagate it into ``_best_val_loss``;
            # do NOT use it for checkpoint scoring. The run_result keeps the
            # observed value so the operator's post-mortem can see it.
            if val_loss is not None and not _math.isfinite(val_loss):
                logger.warning(
                    f"Run {run_idx} validation_loss is non-finite "
                    f"({val_loss!r}); recording on run_result but NOT "
                    f"threading into early-stop / checkpoint scoring to "
                    f"avoid poisoning subsequent comparisons. "
                    f"run_id={self._run_id}."
                )
                run_result.validation_loss = val_loss
                return run_result, None

            run_result.validation_loss = val_loss

            # Phase 5.3: Update checkpoint with validation loss for smarter pruning
            #
            # v1.4 BACKEND-F-004 (Wave 6b features): wrap the
            # validation-loss-update manifest save in the same cross-process
            # lock as the in-class CheckpointManager mutators (register /
            # prune / protect). Concurrent multi-run sessions otherwise
            # race on this write.
            if self._checkpoint_manager:
                for cp in self._checkpoint_manager.list_checkpoints():
                    if cp.run_index == run_idx and cp.validation_loss is None:
                        cp.validation_loss = val_loss
                        with self._checkpoint_manager._locked_manifest_write(
                            "validation_loss_update"
                        ):
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

        Stage C amend BACKEND-B-025: thread-safety contract for
        ``_gpu_max_temp`` and ``_gpu_max_vram``.

        These two attributes are mutated from the GPU monitor thread
        (this callback) and read from the main thread at the end of each
        run (``_execute_run`` populates RunResult). Under CPython's GIL,
        single-attribute float assignment IS atomic — neither read sees
        a torn float. The cross-attribute consistency is NOT atomic: a
        main-thread read of ``(_gpu_max_temp, _gpu_max_vram)`` may
        observe temp from update N and vram from update N+1. For these
        ADVISORY metrics (post-mortem max values, not a real-time
        decision input) the per-attribute atomicity is sufficient; we
        deliberately do NOT add a threading.Lock here because the
        contention cost on every poll outweighs the cross-attribute
        consistency win for a non-load-bearing pair. v1.4 may revisit
        this if a metrics dashboard needs strict pairing.
        """
        # Track max values (per-attribute atomic under GIL; cross-attribute
        # consistency is not guaranteed and deliberately not enforced —
        # see contract above).
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
        """Handle emergency GPU condition.

        BACKEND-F-001: as of v1.3 :meth:`abort` interrupts the inner
        ``SFTTrainer.train()`` mid-step via the ``_AbortCallback``
        bridge — so a thermal/VRAM emergency mid-run now actually
        halts training within seconds instead of waiting for the
        in-flight run to finish. This restores the safety contract
        the GPU emergency path documented but pre-F-001 silently
        could not deliver.
        """
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

        # Stage C amend BACKEND-B-017: defensive None-check on the model
        # before any .eval() / forward call. A model can legitimately be
        # None when a SLAO restore partially failed and the trainer is in
        # a half-built state; the prior code would AttributeError deep in
        # .eval(), producing a stack trace that pointed at "the validation
        # loop" instead of the real root cause (a broken model). Return
        # +inf so the early-stop / checkpoint-scoring math treats this
        # run as worst-case and the operator's session moves on.
        if model is None:
            logger.warning(
                f"_compute_validation_loss: model is None for run {run_idx} "
                f"(run_id={self._run_id}); the trainer state is incomplete "
                f"— likely a partially-failed SLAO restore. Returning +inf "
                f"so this run is treated as worst-case for early-stop / "
                f"checkpoint scoring; subsequent runs continue normally."
            )
            return float("inf")

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

        # Compute loss on validation set.
        #
        # Stage C amend BACKEND-B-017: guard .eval() against AttributeError
        # on a corrupt model object so we don't poison the model's training
        # mode for the next run. If .eval() fails the model state is
        # already broken; return +inf and let upstream early-stop logic
        # see this as a worst-case run.
        #
        # Wave 3.5 BACKEND-B-003: the eval() call MUST live inside
        # the try/finally pair whose finally restores model.train() — pre-
        # fix the eval() succeeded outside the try block, so if ANY
        # exception fired between the eval() success and the for-loop
        # entry (e.g. an exotic dataset iterator raising in __iter__,
        # `with torch.no_grad():` failing in a partially-corrupt CUDA
        # context, an asynchronous interrupt mid-statement), the model
        # stayed in eval mode forever. Subsequent training runs would
        # then silently train with disabled dropout / disabled BN-stat
        # updates, degrading quality without any error. The fix moves
        # eval() to the FIRST statement of the try block so the finally
        # at the bottom always fires on the success-of-eval branch.
        try:
            try:
                model.eval()
            except AttributeError as exc:
                logger.warning(
                    f"_compute_validation_loss: model.eval() raised "
                    f"AttributeError ({exc}); model object is corrupt. "
                    f"Returning +inf so this run is treated as worst-case."
                )
                return float("inf")

            total_loss = 0.0
            count = 0
            skipped = 0
            # Stage C BACKEND-B-012: track samples skipped because they lack a
            # recognized text field (silent-skip path; pre-fix this was
            # invisible). Operators tuning early_stopping_threshold need to
            # know if half their holdout is being silently dropped — otherwise
            # the early-stop signal is based on a shrinking sample of
            # compatible rows rather than the configured holdout.
            silent_skipped = 0

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
                            # Stage C BACKEND-B-012: count silent-skip alongside
                            # exception-skip so the operator sees the true
                            # validation sample count.
                            silent_skipped += 1
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
            # (normal completion, exception, KeyboardInterrupt, etc.).
            #
            # Stage C amend BACKEND-B-017: guard against AttributeError +
            # None — a corrupt model object from a partially-failed SLAO
            # restore shouldn't poison the next run's exception trace with
            # a misleading "validation loop" frame. Best-effort restore;
            # if it can't run we log and move on.
            try:
                if model is not None:
                    model.train()
            except AttributeError as restore_exc:
                logger.warning(
                    f"_compute_validation_loss: model.train() restore "
                    f"raised AttributeError ({restore_exc}); model object "
                    f"is corrupt and could not be returned to training "
                    f"mode. Next training step may behave unpredictably."
                )

        if skipped > 0:
            logger.warning(f"Skipped {skipped} validation samples due to errors")

        # Stage C BACKEND-B-012: warn when silent-skip is material (>10% of
        # the holdout). The silent-skip path is the common one when a dataset
        # schema drifts mid-session (e.g. a streaming dataset where later
        # samples don't carry text/messages/conversations). Without this
        # warning, the operator's early-stopping decisions are based on a
        # silently-shrinking sample.
        if silent_skipped > 0:
            total_attempted = count + skipped + silent_skipped
            pct = (silent_skipped / total_attempted * 100.0) if total_attempted else 0.0
            log_fn = logger.warning if pct > 10.0 else logger.info
            log_fn(
                f"_compute_validation_loss: silently skipped {silent_skipped} "
                f"samples ({pct:.1f}% of {total_attempted}) lacking "
                f"text/messages/conversations field (run {run_idx})"
            )

        if count == 0:
            logger.warning(
                f"No validation samples were successfully evaluated "
                f"(skipped={skipped} silent_skipped={silent_skipped})"
            )
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
