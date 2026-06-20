"""
Backpropagate - Configuration Management
=========================================

Modern configuration using pydantic-settings for type-safe environment variable parsing.
All settings can be overridden via environment variables with BACKPROPAGATE_ prefix.

Example:
    BACKPROPAGATE_TRAINING__LEARNING_RATE=2e-4
    BACKPROPAGATE_TRAINING__BATCH_SIZE=4
    BACKPROPAGATE_MODEL__NAME=unsloth/Qwen2.5-7B-Instruct-bnb-4bit

Features:
- Type-safe configuration with automatic validation
- Nested config via double underscore delimiter (__)
- .env file support
- Cached settings instance via @lru_cache
- Windows-safe defaults baked in
"""

import os
import threading
from dataclasses import dataclass as dc_dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any


def _safe_pkg_version() -> str:
    """Return the installed backpropagate version, or a sentinel fallback.

    BRIDGE-A-015 (Stage C amend): the prior ``_pkg_version("backpropagate")``
    call ran at class-body evaluation time, so loading the package from a
    source tree without ``pip install -e .`` (common in CI checkouts,
    container builds, dev rebases) raised ``PackageNotFoundError`` and
    crashed module import BEFORE pytest had a chance to collect tests.

    The fallback string ``"0.0.0+unknown"`` is the PEP 440 ``+local``
    suffix convention for "we know this is a dev build but cannot resolve
    the canonical version" — distinct from any real release so downstream
    parsers don't confuse it with a tagged ``0.0.0``.
    """
    try:
        return _pkg_version("backpropagate")
    except PackageNotFoundError:
        return "0.0.0+unknown"


# =============================================================================
# DATA-B-009 — DEPRECATED ENV-VAR SCAN
# =============================================================================
# Settings uses ``extra="ignore"`` so an unknown ``BACKPROPAGATE_*`` env var
# is silently dropped. That is the right default for forward-compat, but it
# means a RENAMED knob (the operator set the old name they remember) is
# applied to NOTHING with zero feedback — the run quietly uses the default.
# We keep a small map of names that were renamed across releases and emit a
# one-time WARN naming the replacement when the old name is still set in the
# environment. New renames append here; the map is intentionally tiny.
#
# Keys are FULL env-var names (the BACKPROPAGATE_ prefix included). Values are
# the current name to use instead (or ``None`` when the knob was removed
# outright with no replacement).
_DEPRECATED_ENV_VARS: dict[str, str | None] = {
    # BRIDGE-B-004 (v1.4): the env-var model was renamed
    # MultiRunConfig -> MultiRunSettings. Its env prefix was historically
    # written both ways in the wild; the canonical prefix is now
    # BACKPROPAGATE_MULTIRUN__ (no underscore between MULTI and RUN).
    "BACKPROPAGATE_MULTI_RUN__NUM_RUNS": "BACKPROPAGATE_MULTIRUN__NUM_RUNS",
    "BACKPROPAGATE_MULTI_RUN__STEPS_PER_RUN": "BACKPROPAGATE_MULTIRUN__STEPS_PER_RUN",
    "BACKPROPAGATE_MULTI_RUN__SAMPLES_PER_RUN": "BACKPROPAGATE_MULTIRUN__SAMPLES_PER_RUN",
}


def _warn_deprecated_env_vars() -> list[str]:
    """Scan ``os.environ`` for known-deprecated BACKPROPAGATE_* names.

    Emits one WARN per deprecated name found that names the replacement (or
    says the knob was removed). Returns the list of deprecated names that
    were present, so callers / tests can assert on the scan result without
    parsing logs. Pure read of the environment — never mutates it.
    """
    found: list[str] = []
    for old_name, new_name in _DEPRECATED_ENV_VARS.items():
        if old_name in os.environ:
            found.append(old_name)
            try:
                from .logging_config import get_logger as _get_logger

                _logger = _get_logger(__name__)
            except Exception:  # noqa: BLE001 — logging must never block config
                import logging as _logging

                _logger = _logging.getLogger(__name__)
            if new_name:
                _logger.warning(
                    "Deprecated environment variable %s is set but no longer "
                    "read; rename it to %s. The current value is being "
                    "ignored and the default applies.",
                    old_name,
                    new_name,
                )
            else:
                _logger.warning(
                    "Environment variable %s is set but has been removed and "
                    "is no longer read; remove it from your environment.",
                    old_name,
                )
    return found


__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "reload_settings",
    "get_output_dir",
    "get_cache_dir",
    # Sub-configs
    "ModelConfig",
    "TrainingConfig",
    "LoRAConfig",
    "DataConfig",
    "UIConfig",
    "WindowsConfig",
    "SecurityConfig",
    # BRIDGE-B-004 (Stage C): canonical name for the env-var-driven
    # MultiRunSettings pydantic-settings model. The legacy alias
    # ``MultiRunConfig`` is kept in this module's namespace ONLY for
    # back-compat with code that did ``from backpropagate.config import
    # MultiRunConfig``; new code should import the alias from
    # ``backpropagate.multi_run`` (the dataclass with merge_mode / lr_decay
    # / replay fields used by ``cmd_multi_run``).
    "MultiRunSettings",
    # Training presets (Phase 1.2)
    "TrainingPreset",
    # v1.4 rename (Wave 6a, Wave 5 Decision 3): the multi-run-loop preset
    # dict is canonically ``MULTI_RUN_PRESETS`` to disambiguate from the
    # v1.3-era ``LORA_PRESETS`` (LoRA-architecture shape; CLI
    # ``--lora-preset``). Both formerly shared the keys ``"fast"`` +
    # ``"quality"`` with semantically different values. The legacy
    # ``TRAINING_PRESETS`` name continues to resolve via module-level
    # ``__getattr__`` + ``DeprecationWarning`` (v1.4 → present) →
    # ``AttributeError`` (future release, v1.7 or later).
    "MULTI_RUN_PRESETS",
    "TRAINING_PRESETS",
    "MODEL_PRESETS",
    "ModelPreset",
    "get_preset",
    "get_model_preset",
    "lookup_model_preset_by_id",
    # LoRA presets (v1.3 BACKEND-1)
    "LoRAPreset",
    "LORA_PRESETS",
    "get_lora_preset",
    # LR scaling helpers (Phase 1.3)
    "get_recommended_lr",
    "get_recommended_warmup",
    # Constants
    "PYDANTIC_SETTINGS_AVAILABLE",
]

try:
    from pydantic import Field, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    PYDANTIC_SETTINGS_AVAILABLE = False


# =============================================================================
# WINDOWS-SAFE DEFAULTS (Based on RTX 5080 testing)
# =============================================================================

WINDOWS_DEFAULTS = {
    "dataloader_num_workers": 0,
    "tokenizers_parallelism": False,
    "xformers_disabled": True,  # SM 12.0+ (Blackwell/Ada)
    # Mirror WindowsConfig.cuda_launch_blocking (default False): blocking
    # launches slow training and are debug-only. Keep this dict in lockstep
    # with the dataclass default (CONFIG-A-003).
    "cuda_launch_blocking": False,
    "pre_tokenize": True,  # Avoid multiprocessing crashes
}


# =============================================================================
# WINDOWS-FIXES MUTATION TRACKING (BRIDGE-B-011 Stage C)
# =============================================================================
# Track the original env-var values that apply_windows_fixes overwrites so
# unapply_windows_fixes can restore (not just delete) the pre-mutation state
# for test isolation. _UNSET sentinel distinguishes "var was unset before"
# from "var was empty-string before".

class _Unset:
    """Sentinel for env vars that were not set prior to apply_windows_fixes."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<UNSET>"


_UNSET = _Unset()

# Mapping of env-var name -> (prior_value_or_UNSET, new_value). Populated by
# _apply_env_mutations; consumed by unapply_windows_fixes.
_WINDOWS_FIXES_APPLIED: dict[str, tuple[object, str]] = {}


def _apply_env_mutations(mutations: dict[str, str]) -> None:
    """Apply env-var mutations, recording the prior value of each key.

    Used by ``Settings.apply_windows_fixes`` (both BaseSettings and dataclass
    fallback branches) so the assignments are uniformly observable + reversible.

    Emits a structlog ``windows_fixes_applied`` event listing the mutated
    keys and new values. The event is best-effort — if structlog is not
    installed or logging is misconfigured, the mutations still take effect
    (the trainer's load-bearing fix path must not depend on observability).
    """
    for key, new_value in mutations.items():
        prior: object = os.environ.get(key, _UNSET)
        # Record only the FIRST prior value seen — if apply_windows_fixes is
        # called twice without unapply in between, the "real" original
        # (pre-our-changes) is the one from the first call, not the second.
        if key not in _WINDOWS_FIXES_APPLIED:
            _WINDOWS_FIXES_APPLIED[key] = (prior, new_value)
        else:
            # Update the new_value (in case the second call wrote a different
            # one) but keep the original prior.
            original_prior, _ = _WINDOWS_FIXES_APPLIED[key]
            _WINDOWS_FIXES_APPLIED[key] = (original_prior, new_value)
        os.environ[key] = new_value

    # Best-effort structlog event so JSON consumers can grep for the
    # mutation. Import lazily to avoid a circular dependency with
    # logging_config (which imports config indirectly via Settings reload).
    try:
        from .logging_config import get_logger as _get_logger
        _get_logger(__name__).info(
            "windows_fixes_applied",
            mutated=list(mutations.keys()),
            values=mutations,
        )
    except Exception:  # noqa: BLE001 — observability must not block the load-bearing path  # nosec B110
        pass


def unapply_windows_fixes() -> None:
    """Restore env vars mutated by :func:`Settings.apply_windows_fixes`.

    BRIDGE-B-011 (Stage C humanization): primarily a test-isolation helper.
    A test that exercises a non-Windows code path on a Windows CI runner
    can call this in teardown so the next test doesn't see leaked
    ``TOKENIZERS_PARALLELISM`` / ``XFORMERS_DISABLED`` env vars.

    Behavior:
    * If the var was unset before apply, it is popped.
    * If the var had a prior value, that value is restored exactly.
    * Vars not touched by apply_windows_fixes are left alone.

    Calling unapply multiple times is safe (the second call is a no-op).
    """
    while _WINDOWS_FIXES_APPLIED:
        key, (prior, _new_value) = _WINDOWS_FIXES_APPLIED.popitem()
        if isinstance(prior, _Unset):
            os.environ.pop(key, None)
        else:
            assert isinstance(prior, str)  # narrow from tuple[object, str]
            os.environ[key] = prior
    try:
        from .logging_config import get_logger as _get_logger
        _get_logger(__name__).info("windows_fixes_unapplied")
    except Exception:  # noqa: BLE001  # nosec B110
        pass


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

if PYDANTIC_SETTINGS_AVAILABLE:

    class ModelConfig(BaseSettings):
        """Model configuration."""
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_MODEL__",
            env_ignore_empty=True,
        )

        # Model identifier (HuggingFace path or local path)
        # Using official Qwen model - Unsloth handles 4-bit quantization
        name: str = "Qwen/Qwen2.5-7B-Instruct"
        # Whether to use 4-bit quantization
        load_in_4bit: bool = True
        # Maximum sequence length
        max_seq_length: int = 2048
        # Data type for training
        dtype: str | None = None  # Auto-detect (bf16 on Ampere+)
        # Trust remote code from HuggingFace
        trust_remote_code: bool = True

    class LoRAConfig(BaseSettings):
        """LoRA/QLoRA configuration.

        v1.3 BACKEND-1 default bump: rank 16 / q+v+gate target / 1× LR
        defaults left ~15-20% quality on the table per Biderman 2024 +
        Thinking Machines 2025 — LoRA at rank 256 + all-linear target +
        higher LR matches full fine-tuning quality on most post-training
        tasks at roughly 67% of the compute. Defaults are now the
        "quality" preset; operators who liked the old speed-tilted
        behavior can opt in via ``--lora-preset=fast`` (CLI) or
        ``LORA_PRESETS["fast"]`` (programmatic).

        ``target_modules`` accepts either a list of module names (legacy
        shape) or the PEFT wildcard string ``"all-linear"`` (PEFT >= 0.7;
        wires to every linear/Conv1D module except the LM head). The
        latter is the v1.3 default — wider adaptation surface = better
        quality at the cost of ~2-3× LoRA parameter count.

        ``use_dora``: enable DoRA (Liu et al., 2024 — Weight-Decomposed
        Low-Rank Adaptation). When True, PEFT decomposes weight updates
        into magnitude + direction so a small rank (e.g. 8) matches
        plain LoRA rank 32 quality at zero inference overhead. Default
        OFF for backward-compat; flip to True for better quality at
        +5-10% training time cost. Requires PEFT >= 0.10 (the field is
        ignored on older PEFT — trainer.py degrades gracefully).

        ``init_lora_weights``: PEFT initialization scheme for the LoRA
        weights. ``"default"`` (Microsoft reference impl, B=0 so the
        adapter is a no-op pre-training), ``"pissa"`` (PiSSA — Meng et
        al. 2024; initialize from top-r SVD of W, faster convergence on
        QLoRA), ``"loftq"`` (LoftQ — Li et al. 2023; coupled
        quantization-aware initialization that recovers some
        quantization error on QLoRA). Both pissa/loftq are free quality
        recovery for QLoRA runs.
        """
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_LORA__",
            env_ignore_empty=True,
        )

        # LoRA rank (dimension). v1.3 default bumped 16 -> 256 (quality preset).
        r: int = 256
        # LoRA alpha (scaling factor). Keep alpha = 2 * r convention.
        lora_alpha: int = 512
        # Dropout rate
        lora_dropout: float = 0.05
        # Target modules for LoRA. v1.3 default bumped from a hand-curated
        # 7-module list to the PEFT "all-linear" wildcard. Accepts a list
        # of names for explicit control (e.g. ["q_proj", "v_proj"] for the
        # legacy q+v shape).
        target_modules: str | list[str] = Field(default="all-linear")
        # Use gradient checkpointing (reduces VRAM by ~30%)
        use_gradient_checkpointing: str = "unsloth"  # "unsloth" or True/False
        # Random state for reproducibility
        random_state: int = 42
        # v1.3 BACKEND-3: DoRA opt-in. Default False for backward-compat.
        use_dora: bool = False
        # v1.3 BACKEND-6: PiSSA / LoftQ initialization. Literal-shaped string;
        # PEFT accepts {"default", "pissa", "loftq"} (we map "default" -> True
        # at the call site in trainer.py to honor the PEFT API).
        init_lora_weights: str = "default"
        # v1.5 T2.3 (rsLoRA, finding 19 — Kalajdzievski 2023, arXiv:2312.03732).
        # Rank-Stabilized LoRA: scale the adapter by alpha/sqrt(r) instead of the
        # standard alpha/r. Standard alpha/r over-throttles gradients at high
        # rank, so the rank-256 default may under-train the adapter; rsLoRA's
        # benefit GROWS with rank, at ZERO inference cost (it is a pure scaling
        # choice — the merged weights are identical-shaped, so the GGUF -> Ollama
        # export path is unaffected). Default OFF for backward-compat; flip to
        # True via this field, ``BACKPROPAGATE_LORA__USE_RSLORA``, or
        # ``Trainer(use_rslora=True)``. Threaded into PEFT's ``LoraConfig
        # (use_rslora=...)`` at the adapter-build call site in trainer.py.
        use_rslora: bool = False

    class TrainingConfig(BaseSettings):
        """Training hyperparameters."""
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_TRAINING__",
            env_ignore_empty=True,
        )

        # Batch size per device
        per_device_train_batch_size: int = 2
        # Gradient accumulation steps (effective batch = batch_size * grad_accum)
        gradient_accumulation_steps: int = 4
        # Number of training steps (0 = use num_train_epochs)
        max_steps: int = 100
        # Number of epochs (ignored if max_steps > 0)
        num_train_epochs: int = 1
        # Learning rate
        learning_rate: float = 2e-4
        # Weight decay
        weight_decay: float = 0.01
        # Warmup steps
        warmup_steps: int = 10
        # Warmup ratio (alternative to warmup_steps)
        warmup_ratio: float = 0.0
        # Optimizer
        optim: str = "adamw_8bit"
        # LR scheduler type
        lr_scheduler_type: str = "cosine"
        # Logging steps
        logging_steps: int = 10
        # Save steps
        save_steps: int = 100
        # Use bf16 (recommended for Ampere+)
        bf16: bool = True
        # Use fp16 (for older GPUs)
        fp16: bool = False
        # Seed for reproducibility
        seed: int = 42
        # Output directory
        output_dir: str = "./output"
        # Overwrite output directory
        overwrite_output_dir: bool = True
        # v1.5 T1.2 (ORPO): training objective selector. "sft" (default) =
        # supervised fine-tuning — byte-identical to v1.4 behavior. "orpo" =
        # reference-free monolithic preference optimization (Hong, Lee &
        # Thorne 2024, arXiv:2403.07691): standard SFT NLL loss + a per-step
        # odds-ratio penalty over (chosen, rejected) pairs, single stage, no
        # reference model — so the VRAM envelope matches SFT. A later trainer
        # wave dispatches on this; the default preserves the existing path.
        #
        # NB: deliberately typed ``str`` (not ``Literal["sft", "orpo"]``) so
        # the {"sft", "orpo"} constraint is enforced by the
        # ``_reject_invalid_method`` after-validator below, which raises a
        # structured ``InvalidSettingError`` (CONFIG_INVALID_SETTING) —
        # mirroring ``_reject_bf16_and_fp16``. A ``Literal`` field would make
        # pydantic's own type-check the gate, raising a generic
        # ``ValidationError`` BEFORE the after-validator runs, so the
        # contract's structured code/hint (the shape the trainer wave +
        # operators key on) would never surface. The valid set is documented
        # in the validator + the ``--method`` CLI choices + env-vars.md.
        method: str = "sft"
        # v1.5 T1.2 (ORPO): the odds-ratio weight (the ORPO "lambda" /
        # ``beta`` in TRL's ORPOConfig). Scales the relative-ratio loss term
        # added on top of the NLL loss. Default 0.1 (the paper's headline
        # setting). Ignored unless ``method == "orpo"``. Keep > 0 — a
        # non-positive weight degenerates ORPO back to plain SFT.
        orpo_beta: float = 0.1
        # v1.5 T2.1 (FP8 compute path): opt-in FP8 training via torchao's
        # float8 (Blackwell 5th-gen tensor cores; Hong-/Dettmers-class memory
        # win — ~1.4x throughput, up to 60% less model memory, and the adapter
        # still merges). Experimental in v1.5: default OFF so existing runs are
        # byte-identical. When True, the trainer converts the BASE projection
        # linears to Float8Linear AFTER the LoRA adapter is attached (the
        # adapter's rank-r sub-linears + lm_head + embeddings are excluded), and
        # degrades gracefully to bf16 (one WARN, no raise) on a non-CUDA /
        # pre-Hopper card or when torchao is absent. No validator: a bool cannot
        # be malformed (unlike ``method``, whose {sft, orpo} set needs guarding).
        # mode='full' + fp8 and method='orpo' + fp8 are rejected by the Trainer
        # constructor gate ladder, NOT here (they are cross-field combinations
        # the per-field config layer doesn't see).
        fp8: bool = False
        # v1.5 T3.1 (MLX / Apple-Silicon backend): the compute-backend selector.
        # "auto" (the default) resolves to "mlx" on an Apple-Silicon Mac with the
        # [mlx] extra installed, else "cuda" — so existing CUDA rigs are
        # byte-identical. "cuda" forces the canonical CUDA rail. "mlx" forces the
        # Apple-Silicon rail (mlx_lm.lora under the hood); on a non-Apple host
        # the Trainer constructor rejects a forced "mlx" with a structured
        # CONFIG_INVALID_SETTING (it is an unrunnable request on that hardware).
        #
        # NB: deliberately typed ``str`` (not ``Literal["auto","cuda","mlx"]``)
        # so the {auto, cuda, mlx} constraint is enforced by the
        # ``_reject_invalid_backend`` after-validator below, which raises a
        # structured ``InvalidSettingError`` (CONFIG_INVALID_SETTING) — mirroring
        # ``_reject_invalid_method``. A ``Literal`` field would make pydantic's
        # own type-check the gate, raising a generic ``ValidationError`` BEFORE
        # the after-validator runs, so the contract's structured code/hint would
        # never surface. Env: ``BACKPROPAGATE_TRAINING__BACKEND``.
        backend: str = "auto"

        @model_validator(mode="after")
        def _reject_invalid_backend(self) -> "TrainingConfig":
            """Reject a ``backend`` outside {"auto", "cuda", "mlx"} at construction.

            v1.5 T3.1: ``backend`` is the compute-rail selector and the field is
            typed ``str`` (not ``Literal``) precisely so this validator — not
            pydantic's type machinery — decides the valid set, raising a
            structured ``InvalidSettingError`` (``CONFIG_INVALID_SETTING``) for a
            bad value supplied either as a kwarg
            (``TrainingConfig(backend="rocm")``) or via env var
            (``BACKPROPAGATE_TRAINING__BACKEND=rocm``). Mirrors
            ``_reject_invalid_method``. This validator only gates the VALUE; the
            cross-field "backend='mlx' but this is not Apple Silicon" check lives
            in the Trainer constructor (the config layer can't see the host).
            A non-``ValueError`` raised from a pydantic ``after`` validator
            propagates as-is (NOT re-wrapped in ``ValidationError``), so the
            structured code/hint survive.
            """
            if self.backend not in ("auto", "cuda", "mlx"):
                from .exceptions import InvalidSettingError

                raise InvalidSettingError(
                    "backend",
                    self.backend,
                    "one of {'auto', 'cuda', 'mlx'}",
                    suggestion=(
                        "Set backend='auto' (the default — routes to MLX on an "
                        "Apple-Silicon Mac with the [mlx] extra, else CUDA), "
                        "backend='cuda' to force the CUDA rail, or backend='mlx' "
                        "to force the Apple-Silicon rail (macOS + arm64 only)."
                    ),
                )
            return self

        @model_validator(mode="after")
        def _reject_invalid_method(self) -> "TrainingConfig":
            """Reject a ``method`` outside {"sft", "orpo"} at construction.

            v1.5 T1.2: ``method`` is the ORPO/SFT objective selector and is the
            authoritative validation gate for the field (the field is typed
            ``str``, not ``Literal``, precisely so this validator — not
            pydantic's type machinery — decides the valid set). It raises a
            structured ``InvalidSettingError`` (``CONFIG_INVALID_SETTING``),
            mirroring ``_reject_bf16_and_fp16``, so a bad value supplied either
            as a kwarg (``TrainingConfig(method="dpo")``) or via env var
            (``BACKPROPAGATE_TRAINING__METHOD=dpo``) surfaces the SAME
            actionable code/hint the rest of the config-validation path emits.
            A non-``ValueError`` exception raised from a pydantic ``after``
            validator propagates as-is (it is NOT re-wrapped in
            ``ValidationError``), so the structured code/hint survive.
            """
            if self.method not in ("sft", "orpo"):
                from .exceptions import InvalidSettingError

                raise InvalidSettingError(
                    "method",
                    self.method,
                    "one of {'sft', 'orpo'}",
                    suggestion=(
                        "Set method='sft' for supervised fine-tuning (the "
                        "default) or method='orpo' for reference-free "
                        "preference tuning (needs a {chosen, rejected} "
                        "dataset). Other objectives (DPO/SimPO/KTO/PPO/GRPO) "
                        "are not implemented in v1.5 — use TRL / LLaMA-Factory "
                        "for those."
                    ),
                )
            return self

        @model_validator(mode="after")
        def _reject_invalid_orpo_beta(self) -> "TrainingConfig":
            """Reject a non-positive ``orpo_beta`` at construction.

            v1.5 T1.2: ``orpo_beta`` is the odds-ratio weight on ORPO's
            relative-ratio loss term. The field comment already warns "Keep
            > 0 — a non-positive weight degenerates ORPO back to plain SFT,"
            but nothing enforced it: ``orpo_beta=0.0`` silently zeroes the
            odds-ratio term (the run is SFT wearing an ORPO label) and a
            NEGATIVE beta trains TOWARD the rejected completion — both are
            silent correctness bugs that surface only as a mysteriously bad
            model. We gate it here, mirroring ``_reject_invalid_method``, so a
            bad value supplied as a kwarg (``TrainingConfig(orpo_beta=0)``) or
            via env var (``BACKPROPAGATE_TRAINING__ORPO_BETA=0``) raises a
            structured ``InvalidSettingError`` (``CONFIG_INVALID_SETTING``)
            with an actionable hint up front. The constraint is enforced
            unconditionally (not only when ``method == 'orpo'``) so the error
            is deterministic regardless of field-evaluation order; a stray
            non-positive ``orpo_beta`` left under ``method='sft'`` is still a
            misconfiguration worth surfacing. A non-``ValueError`` exception
            raised from a pydantic ``after`` validator propagates as-is (NOT
            re-wrapped in ``ValidationError``), so the structured code/hint
            survive.
            """
            if self.orpo_beta <= 0:
                from .exceptions import InvalidSettingError

                raise InvalidSettingError(
                    "orpo_beta",
                    self.orpo_beta,
                    "a positive number",
                    suggestion=(
                        "Set orpo_beta to a value > 0 (the ORPO paper's "
                        "headline setting is 0.1). A value of 0 zeroes the "
                        "odds-ratio term — the run silently degenerates to "
                        "plain SFT — and a negative value trains toward the "
                        "REJECTED completion."
                    ),
                )
            return self

        @model_validator(mode="after")
        def _reject_bf16_and_fp16(self) -> "TrainingConfig":
            """Reject bf16=True AND fp16=True at construction time.

            DATA-A-007: transformers' ``TrainingArguments`` raises
            ``ValueError("At most one of fp16 and bf16 can be True ...")``
            deep inside the trainer setup — long after the run has spun up
            the model + dataset. Catching the contradiction here keeps the
            "pydantic validates the config early" promise: the operator gets
            a structured ``CONFIG_INVALID_SETTING`` with an actionable hint
            at construction instead of an opaque framework crash mid-run.
            A non-``ValueError`` exception raised from a pydantic ``after``
            validator propagates as-is (it is NOT re-wrapped in
            ``ValidationError``), so the structured code/hint survive.
            """
            if self.bf16 and self.fp16:
                from .exceptions import InvalidSettingError

                raise InvalidSettingError(
                    "bf16/fp16",
                    {"bf16": self.bf16, "fp16": self.fp16},
                    "at most one of bf16 / fp16 enabled",
                    suggestion=(
                        "Set exactly one of bf16 / fp16 (or neither for "
                        "fp32). Use bf16 on Ampere+ / Blackwell GPUs and "
                        "fp16 on older cards; they are mutually exclusive "
                        "mixed-precision modes."
                    ),
                )
            return self

    class DataConfig(BaseSettings):
        """Dataset configuration."""
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_DATA__",
            env_ignore_empty=True,
        )

        # Dataset name or path
        dataset_name: str = "HuggingFaceH4/ultrachat_200k"
        # Dataset split
        dataset_split: str = "train_sft"
        # Number of samples (0 = all)
        max_samples: int = 1000
        # Text column name
        text_column: str = "text"
        # Chat template format (chatml, llama, alpaca, sharegpt)
        chat_format: str = "chatml"
        # Pre-tokenize dataset (Windows-safe)
        pre_tokenize: bool = True
        # Shuffle dataset
        shuffle: bool = True
        # Packing (combine short sequences). v1.3 BACKEND-4: flipped from
        # False to True. Sample packing is the single biggest wall-clock
        # lever for SFT runs (1.7-3× throughput per recent benchmarks,
        # attention-backend agnostic). Operators who hit edge cases
        # (boundary-token leakage, exotic chat templates) can opt out
        # via ``--no-packing`` (CLI) or
        # ``BACKPROPAGATE_DATA__PACKING=false`` (env).
        packing: bool = True
        # v1.5 T3.2 (reasoning-trace SFT / R1 distillation, finding 24).
        # When True the trainer keeps the ``<think>`` chain-of-thought in the
        # training target (the converters already preserve it — no special
        # tokens, no embedding resize, so the merge→GGUF→Ollama export stays
        # intact), applies trace-length filtering (drops empty / over-long
        # traces via datasets.filter_by_trace_length), and bumps the DEFAULT
        # max_seq_length to 8192 if the operator left it at the shipped 2048
        # (longer CoT needs the room; an explicit max_seq_length always wins).
        # Default False ⇒ byte-identical v1.4 SFT. SFT only — ignored under
        # method='orpo'. Env: ``BACKPROPAGATE_DATA__REASONING_TRACE``.
        reasoning_trace: bool = False
        # Minimum think-span token count to keep a sample (reasoning_trace
        # only). Samples whose ``<think>`` content tokenizes below this are
        # dropped as empty/degenerate traces. Env:
        # ``BACKPROPAGATE_DATA__MIN_TRACE_TOKENS``.
        min_trace_tokens: int = 8
        # Maximum think-span token count to keep a sample (reasoning_trace
        # only). Samples whose ``<think>`` content tokenizes above this are
        # dropped as over-long traces. Env:
        # ``BACKPROPAGATE_DATA__MAX_TRACE_TOKENS``.
        max_trace_tokens: int = 8192

    class UIConfig(BaseSettings):
        """Reflex (Radix UI) web interface configuration.

        Migrated from Gradio in v1.1.0. The host / port / share / auto_open
        knobs continue to apply — `cmd_ui` forwards them to the Reflex
        subprocess launch. ``host`` defaults to ``127.0.0.1`` (localhost
        only) for security; expose externally via SSH port-forwarding or a
        Cloudflare Tunnel rather than ``host=0.0.0.0``.
        """
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_UI__",
            env_ignore_empty=True,
        )

        port: int = 7862
        host: str = "127.0.0.1"  # Localhost only for security
        share: bool = False
        auto_open: bool = True

    class WindowsConfig(BaseSettings):
        """Windows-specific settings (auto-applied on Windows)."""
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_WINDOWS__",
            env_ignore_empty=True,
        )

        # Number of dataloader workers (0 for Windows)
        dataloader_num_workers: int = 0
        # Disable tokenizers parallelism
        tokenizers_parallelism: bool = False
        # Disable xformers (incompatible with SM 12.0+)
        xformers_disabled: bool = True
        # CUDA launch blocking for debugging (slows training, only enable for debugging)
        cuda_launch_blocking: bool = False  # Disabled by default for performance
        # Pre-tokenize to avoid multiprocessing issues
        pre_tokenize: bool = True

    class MultiRunSettings(BaseSettings):
        """Multi-run training settings driven by BACKPROPAGATE_MULTIRUN__* env vars.

        BRIDGE-B-004 (Stage C humanization): pre-fix this class was named
        ``MultiRunConfig`` and collided with
        :class:`backpropagate.multi_run.MultiRunConfig` (a dataclass with
        merge_mode + lr_decay + replay fields used by ``cmd_multi_run``).
        The collision left every ``BACKPROPAGATE_MULTIRUN__*`` env var dead
        because ``cmd_multi_run`` instantiates the multi_run.py class, never
        ``Settings().multi_run``. Renaming to ``MultiRunSettings`` makes the
        ownership distinct: this class is the operator-facing env-var
        knobs, multi_run.MultiRunConfig is the in-process call-site object.

        Environment variables (env_prefix=BACKPROPAGATE_MULTIRUN__):
            BACKPROPAGATE_MULTIRUN__NUM_RUNS=5
            BACKPROPAGATE_MULTIRUN__STEPS_PER_RUN=100
            BACKPROPAGATE_MULTIRUN__SAMPLES_PER_RUN=1000
            BACKPROPAGATE_MULTIRUN__CONTINUE_FROM_PREVIOUS=true
            BACKPROPAGATE_MULTIRUN__SAVE_INTERMEDIATE=true

        Validation failures (non-numeric NUM_RUNS, etc.) surface as
        pydantic ValidationError with the env var name in the location
        field — operators can grep the message for the var they set.
        """
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_MULTIRUN__",
            env_ignore_empty=True,
        )

        # Number of training runs
        num_runs: int = 5
        # Steps per run
        steps_per_run: int = 100
        # Samples per run (new samples each run)
        samples_per_run: int = 1000
        # Whether to continue from previous LoRA
        continue_from_previous: bool = True
        # Save intermediate checkpoints
        save_intermediate: bool = True

    class SecurityConfig(BaseSettings):
        """
        Security configuration for production deployments.

        Environment Variables:
            BACKPROPAGATE_SECURITY__REQUIRE_AUTH=true
            BACKPROPAGATE_SECURITY__ALLOWED_PATHS=/data,/models
            BACKPROPAGATE_SECURITY__SESSION_TIMEOUT_MINUTES=30
            BACKPROPAGATE_SECURITY__ENABLE_CSRF=true

        2026 Best Practices:
        - Require auth when share=True (public URLs)
        - Restrict file access to allowed directories
        - Use JWT with short expiry for sessions
        - Enable CSRF protection for state-changing requests
        """
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_SECURITY__",
            env_ignore_empty=True,
        )

        # Authentication
        require_auth: bool = False  # Set True in production
        auth_username: str | None = None
        auth_password: str | None = Field(default=None, json_schema_extra={"secret": True})  # nosec B105 — Pydantic metadata flag (the value "True" is a boolean, not a password)

        # Path restrictions
        allowed_paths: list[str] | None = None  # None = no restriction
        block_path_traversal: bool = True

        # Session management
        session_timeout_minutes: int = 30
        jwt_secret: str | None = Field(default=None, json_schema_extra={"secret": True})  # nosec B105 — Pydantic metadata flag (the value "True" is a boolean, not a password)
        jwt_algorithm: str = "HS256"

        # CSRF protection
        enable_csrf: bool = True
        csrf_token_expiry_minutes: int = 60

        # Rate limiting
        rate_limit_training: int = 3  # Max training starts per minute
        rate_limit_export: int = 5    # Max exports per minute

        # Logging
        audit_log_enabled: bool = True
        audit_log_file: str | None = None  # None = stdout only

        # Content Security Policy
        enable_csp: bool = True
        csp_report_only: bool = False  # Set False to enforce

        def get_auth_tuple(self) -> tuple | None:
            """Return ``(username, password)`` if both auth credentials are set, else ``None``.

            Originally fed Gradio's basic-auth tuple; in v1.1.0+ the Reflex
            UI consumes the same shape via ``ui_app/auth.py``. The format is
            preserved so existing BACKPROPAGATE_SECURITY__AUTH_USERNAME /
            AUTH_PASSWORD env vars stay valid across the migration.
            """
            if self.auth_username and self.auth_password:
                return (self.auth_username, self.auth_password)
            return None

        def validate_production_config(self) -> list[str]:
            """Check for security misconfigurations. Returns list of warnings."""
            warnings = []
            if not self.require_auth:
                warnings.append("SECURITY: require_auth is False - UI is unprotected")
            if not self.jwt_secret:
                warnings.append("SECURITY: jwt_secret not set - using random secret (sessions lost on restart)")
            if not self.enable_csrf:
                warnings.append("SECURITY: CSRF protection disabled")
            if self.session_timeout_minutes > 480:  # 8 hours
                warnings.append("SECURITY: session_timeout_minutes > 8 hours - consider shorter timeout")
            return warnings

    class Settings(BaseSettings):
        """
        Main settings container using pydantic-settings.

        All settings are loaded from environment variables with BACKPROPAGATE_ prefix.
        Nested settings use double underscore (__) as delimiter.

        Usage:
            from backpropagate.config import get_settings

            settings = get_settings()
            print(settings.model.name)
            print(settings.training.learning_rate)

        Environment Examples:
            BACKPROPAGATE_MODEL__NAME=unsloth/Llama-3.2-3B-Instruct-bnb-4bit
            BACKPROPAGATE_TRAINING__LEARNING_RATE=1e-4
            BACKPROPAGATE_LORA__R=32
        """
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_",
            env_file=".env",
            env_file_encoding="utf-8",
            env_ignore_empty=True,
            env_nested_delimiter="__",
            extra="ignore",
        )

        # Nested configs
        model: ModelConfig = Field(default_factory=ModelConfig)
        training: TrainingConfig = Field(default_factory=TrainingConfig)
        lora: LoRAConfig = Field(default_factory=LoRAConfig)
        data: DataConfig = Field(default_factory=DataConfig)
        ui: UIConfig = Field(default_factory=UIConfig)
        windows: WindowsConfig = Field(default_factory=WindowsConfig)
        # BRIDGE-B-004 (Stage C): renamed config.MultiRunConfig -> MultiRunSettings.
        multi_run: MultiRunSettings = Field(default_factory=MultiRunSettings)
        security: SecurityConfig = Field(default_factory=SecurityConfig)

        # Package info
        # BRIDGE-A-015 (Stage C amend): wrap in default_factory so the version
        # lookup is deferred until Settings() is actually instantiated, AND
        # routed through _safe_pkg_version which returns "0.0.0+unknown" when
        # the package metadata is missing. Pre-fix this was
        # ``version: str = _pkg_version("backpropagate")`` evaluated at class-
        # body time — a fresh CI checkout without `pip install -e .` would
        # raise PackageNotFoundError and crash module import before any test
        # collection.
        version: str = Field(default_factory=_safe_pkg_version)
        name: str = "backpropagate"

        def to_dict(self) -> dict:
            """Export settings as dictionary."""
            return {
                "version": self.version,
                "model": {
                    "name": self.model.name,
                    "load_in_4bit": self.model.load_in_4bit,
                    "max_seq_length": self.model.max_seq_length,
                },
                "training": {
                    "batch_size": self.training.per_device_train_batch_size,
                    "grad_accum": self.training.gradient_accumulation_steps,
                    "learning_rate": self.training.learning_rate,
                    "max_steps": self.training.max_steps,
                },
                "lora": {
                    "r": self.lora.r,
                    "alpha": self.lora.lora_alpha,
                    "dropout": self.lora.lora_dropout,
                },
                "data": {
                    "dataset": self.data.dataset_name,
                    "max_samples": self.data.max_samples,
                },
            }

        def apply_windows_fixes(self) -> None:
            """Apply Windows-specific environment variables.

            BRIDGE-B-011 (Stage C humanization): each mutation is now recorded
            in the module-level ``_WINDOWS_FIXES_APPLIED`` dict (key -> prior
            value or _UNSET sentinel) so :func:`unapply_windows_fixes` can
            restore the original environment for test isolation. A structlog
            event lists every mutated env var + new value so operators
            grepping JSON logs can correlate "the trainer disabled xformers
            on me" with the exact env-var write.
            """
            if os.name == "nt":  # Windows
                mutations: dict[str, str] = {
                    "TOKENIZERS_PARALLELISM": str(self.windows.tokenizers_parallelism).lower(),
                }
                if self.windows.xformers_disabled:
                    mutations["XFORMERS_DISABLED"] = "1"
                if self.windows.cuda_launch_blocking:
                    mutations["CUDA_LAUNCH_BLOCKING"] = "1"
                _apply_env_mutations(mutations)

else:
    # Fallback implementation using dataclasses
    from dataclasses import dataclass, field

    def _get_env(key: str, default: str | None = None) -> str | None:
        return os.environ.get(f"BACKPROPAGATE_{key}", default)

    def _get_env_int(key: str, default: int) -> int:
        val = _get_env(key)
        return int(val) if val else default

    def _get_env_float(key: str, default: float) -> float:
        val = _get_env(key)
        return float(val) if val else default

    def _get_env_bool(key: str, default: bool) -> bool:
        val = _get_env(key)
        return val.lower() in ("true", "1", "yes") if val else default

    @dataclass
    class ModelConfig:  # type: ignore[no-redef]
        name: str = "Qwen/Qwen2.5-7B-Instruct"  # Official model, Unsloth handles 4-bit
        load_in_4bit: bool = True
        max_seq_length: int = 2048
        dtype: str | None = None
        trust_remote_code: bool = True

    @dataclass
    class LoRAConfig:  # type: ignore[no-redef]
        # v1.3 BACKEND-1: defaults bumped to "quality" preset (rank 256 +
        # all-linear). See BaseSettings branch docstring above for the
        # research-backed rationale. Defaults match the BaseSettings
        # branch byte-identically so a missing pydantic-settings install
        # doesn't silently change training behavior.
        r: int = 256
        lora_alpha: int = 512
        lora_dropout: float = 0.05
        target_modules: str | list[str] = field(default="all-linear")
        use_gradient_checkpointing: str = "unsloth"
        random_state: int = 42
        use_dora: bool = False
        init_lora_weights: str = "default"
        # v1.5 T2.3 (rsLoRA): parity with the BaseSettings branch above —
        # byte-identical default so a pydantic-settings-less install scales the
        # adapter the same way. alpha/sqrt(r) vs alpha/r; zero inference cost.
        use_rslora: bool = False

    @dataclass
    class TrainingConfig:  # type: ignore[no-redef]
        per_device_train_batch_size: int = 2
        gradient_accumulation_steps: int = 4
        max_steps: int = 100
        num_train_epochs: int = 1
        learning_rate: float = 2e-4
        weight_decay: float = 0.01
        warmup_steps: int = 10
        warmup_ratio: float = 0.0
        optim: str = "adamw_8bit"
        lr_scheduler_type: str = "cosine"
        logging_steps: int = 10
        save_steps: int = 100
        bf16: bool = True
        fp16: bool = False
        seed: int = 42
        output_dir: str = "./output"
        overwrite_output_dir: bool = True
        # v1.5 T1.2 (ORPO): parity with the pydantic branch above. The
        # dataclass fallback can't express a Literal at the type level, so
        # the {"sft", "orpo"} constraint is enforced in __post_init__ below
        # to keep behavior byte-identical across the two installs.
        method: str = "sft"
        orpo_beta: float = 0.1
        # v1.5 T2.1 (FP8 compute path): parity with the pydantic branch above —
        # byte-identical default so a pydantic-settings-less install doesn't
        # silently change FP8 behavior. No validation needed (a bool can't be
        # malformed); the Trainer gate ladder enforces the cross-field rules.
        fp8: bool = False
        # v1.5 T3.1 (MLX / Apple-Silicon backend): parity with the pydantic
        # branch above. The dataclass fallback can't express a Literal at the
        # type level, so the {auto, cuda, mlx} constraint is enforced in
        # __post_init__ below to keep behavior byte-identical across installs.
        backend: str = "auto"

        def __post_init__(self) -> None:
            # DATA-A-007 (parity with the pydantic branch above): reject the
            # bf16+fp16 contradiction here too so a pydantic-settings-less
            # install validates identically and doesn't silently defer the
            # crash to transformers mid-run.
            if self.bf16 and self.fp16:
                from .exceptions import InvalidSettingError

                raise InvalidSettingError(
                    "bf16/fp16",
                    {"bf16": self.bf16, "fp16": self.fp16},
                    "at most one of bf16 / fp16 enabled",
                    suggestion=(
                        "Set exactly one of bf16 / fp16 (or neither for "
                        "fp32). They are mutually exclusive mixed-precision "
                        "modes."
                    ),
                )
            # v1.5 T1.2 (ORPO): reject an invalid method here too so the
            # dataclass-fallback install gives the same structured
            # CONFIG_INVALID_SETTING the pydantic _reject_invalid_method
            # validator raises.
            if self.method not in ("sft", "orpo"):
                from .exceptions import InvalidSettingError

                raise InvalidSettingError(
                    "method",
                    self.method,
                    "one of {'sft', 'orpo'}",
                    suggestion=(
                        "Set method='sft' for supervised fine-tuning (the "
                        "default) or method='orpo' for reference-free "
                        "preference tuning (needs a {chosen, rejected} "
                        "dataset). Other objectives (DPO/SimPO/KTO/PPO/GRPO) "
                        "are not implemented in v1.5 — use TRL / LLaMA-Factory "
                        "for those."
                    ),
                )
            # v1.5 T1.2 (ORPO): reject a non-positive orpo_beta here too so the
            # dataclass-fallback install gives the same structured
            # CONFIG_INVALID_SETTING the pydantic _reject_invalid_orpo_beta
            # validator raises. A 0 weight silently degenerates ORPO to SFT;
            # a negative weight trains toward the rejected completion.
            if self.orpo_beta <= 0:
                from .exceptions import InvalidSettingError

                raise InvalidSettingError(
                    "orpo_beta",
                    self.orpo_beta,
                    "a positive number",
                    suggestion=(
                        "Set orpo_beta to a value > 0 (the ORPO paper's "
                        "headline setting is 0.1). A value of 0 zeroes the "
                        "odds-ratio term — the run silently degenerates to "
                        "plain SFT — and a negative value trains toward the "
                        "REJECTED completion."
                    ),
                )
            # v1.5 T3.1 (MLX): reject an invalid backend here too so the
            # dataclass-fallback install gives the same structured
            # CONFIG_INVALID_SETTING the pydantic _reject_invalid_backend
            # validator raises.
            if self.backend not in ("auto", "cuda", "mlx"):
                from .exceptions import InvalidSettingError

                raise InvalidSettingError(
                    "backend",
                    self.backend,
                    "one of {'auto', 'cuda', 'mlx'}",
                    suggestion=(
                        "Set backend='auto' (the default — routes to MLX on an "
                        "Apple-Silicon Mac with the [mlx] extra, else CUDA), "
                        "backend='cuda' to force the CUDA rail, or backend='mlx' "
                        "to force the Apple-Silicon rail (macOS + arm64 only)."
                    ),
                )

    @dataclass
    class DataConfig:  # type: ignore[no-redef]
        dataset_name: str = "HuggingFaceH4/ultrachat_200k"
        dataset_split: str = "train_sft"
        max_samples: int = 1000
        text_column: str = "text"
        chat_format: str = "chatml"
        pre_tokenize: bool = True
        shuffle: bool = True
        # v1.3 BACKEND-4: packing default flipped True (see BaseSettings
        # branch docstring above for rationale).
        packing: bool = True
        # v1.5 T3.2 reasoning-trace SFT — see BaseSettings branch docstring
        # above for the full rationale (keep <think>, trace-length filter,
        # default max_seq_length bump; default False = byte-identical v1.4 SFT).
        reasoning_trace: bool = False
        min_trace_tokens: int = 8
        max_trace_tokens: int = 8192

    @dataclass
    class UIConfig:  # type: ignore[no-redef]
        port: int = 7862
        host: str = "127.0.0.1"
        share: bool = False
        auto_open: bool = True

    @dataclass
    class WindowsConfig:  # type: ignore[no-redef]
        dataloader_num_workers: int = 0
        tokenizers_parallelism: bool = False
        xformers_disabled: bool = True
        cuda_launch_blocking: bool = False
        pre_tokenize: bool = True

    @dataclass
    class MultiRunSettings:  # type: ignore[no-redef]
        """Dataclass fallback for MultiRunSettings — see BRIDGE-B-004 above."""

        num_runs: int = 5
        steps_per_run: int = 100
        samples_per_run: int = 1000
        continue_from_previous: bool = True
        save_intermediate: bool = True

    @dataclass
    class SecurityConfig:  # type: ignore[no-redef]
        """Security configuration (fallback without pydantic-settings)."""
        require_auth: bool = False
        auth_username: str | None = None
        auth_password: str | None = None
        allowed_paths: list[str] | None = None
        block_path_traversal: bool = True
        session_timeout_minutes: int = 30
        jwt_secret: str | None = None
        jwt_algorithm: str = "HS256"
        enable_csrf: bool = True
        csrf_token_expiry_minutes: int = 60
        rate_limit_training: int = 3
        rate_limit_export: int = 5
        audit_log_enabled: bool = True
        audit_log_file: str | None = None
        enable_csp: bool = True
        csp_report_only: bool = False

        def get_auth_tuple(self) -> tuple | None:
            if self.auth_username and self.auth_password:
                return (self.auth_username, self.auth_password)
            return None

        def validate_production_config(self) -> list[str]:
            warnings = []
            if not self.require_auth:
                warnings.append("SECURITY: require_auth is False")
            if not self.jwt_secret:
                warnings.append("SECURITY: jwt_secret not set")
            return warnings

    @dataclass
    class Settings:  # type: ignore[no-redef]
        model: ModelConfig = field(default_factory=ModelConfig)
        training: TrainingConfig = field(default_factory=TrainingConfig)
        lora: LoRAConfig = field(default_factory=LoRAConfig)
        data: DataConfig = field(default_factory=DataConfig)
        ui: UIConfig = field(default_factory=UIConfig)
        windows: WindowsConfig = field(default_factory=WindowsConfig)
        # BRIDGE-B-004 (Stage C): renamed config.MultiRunConfig -> MultiRunSettings.
        multi_run: MultiRunSettings = field(default_factory=MultiRunSettings)
        security: SecurityConfig = field(default_factory=SecurityConfig)
        version: str = "0.1.0"
        name: str = "backpropagate"

        def to_dict(self) -> dict:
            return {"version": self.version}

        def apply_windows_fixes(self) -> None:
            # BRIDGE-B-011 (Stage C): mirror the BaseSettings branch — route
            # through _apply_env_mutations so the unapply helper + structlog
            # event fire identically when pydantic-settings is absent.
            if os.name == "nt":
                mutations: dict[str, str] = {
                    "TOKENIZERS_PARALLELISM": str(self.windows.tokenizers_parallelism).lower(),
                }
                if self.windows.xformers_disabled:
                    mutations["XFORMERS_DISABLED"] = "1"
                if self.windows.cuda_launch_blocking:
                    mutations["CUDA_LAUNCH_BLOCKING"] = "1"
                _apply_env_mutations(mutations)


# =============================================================================
# CACHED SETTINGS INSTANCE
# =============================================================================

@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses @lru_cache to avoid re-reading environment/.env on every call.
    Call get_settings.cache_clear() to reload settings.
    """
    # DATA-B-009: warn once (per cache lifetime) about deprecated env-var
    # names still set in the environment that ``extra="ignore"`` would
    # otherwise drop silently.
    _warn_deprecated_env_vars()
    return Settings()


# Backwards-compatible singleton
settings = get_settings()


# BRIDGE-B-004 (Stage C): name-collision back-compat alias.
#
# Pre-fix, ``config.MultiRunConfig`` and ``multi_run.MultiRunConfig`` were
# distinct classes with overlapping but non-identical fields; the public
# surface ``from backpropagate import MultiRunConfig`` resolved to the
# multi_run.py dataclass (the one cmd_multi_run uses) but
# ``from backpropagate.config import MultiRunConfig`` returned the env-var
# Settings shape. Operators got the wrong shape depending on import path.
#
# Fix: the env-var class is now ``MultiRunSettings`` (load-bearing
# operator surface for BACKPROPAGATE_MULTIRUN__* vars). The canonical
# call-site object stays ``multi_run.MultiRunConfig``. This alias keeps the
# legacy ``from backpropagate.config import MultiRunConfig`` import
# resolving to the env-var class for one release; it can be removed in
# v1.4 once downstream code has migrated to ``MultiRunSettings``.
MultiRunConfig = MultiRunSettings


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_reload_lock = threading.Lock()


def reload_settings() -> Settings:
    """Reload all settings from environment variables.

    Note: Settings is not designed for concurrent modifications.  This lock
    serialises reload calls so that cache_clear + re-read is atomic, but
    callers should not rely on settings being safe for truly concurrent
    read-while-write scenarios across threads.
    """
    global settings
    with _reload_lock:
        get_settings.cache_clear()
        settings = get_settings()
    return settings


def get_output_dir() -> Path:
    """Get the output directory for trained models."""
    output_dir = Path(settings.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_cache_dir() -> Path:
    """Get the cache directory for this package."""
    cache_dir = Path.home() / ".cache" / "backpropagate"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_training_args() -> dict:
    """
    Get training arguments as a dict for TrainingArguments.

    Returns:
        Dict compatible with transformers.TrainingArguments
    """
    s = settings
    return {
        "per_device_train_batch_size": s.training.per_device_train_batch_size,
        "gradient_accumulation_steps": s.training.gradient_accumulation_steps,
        "max_steps": s.training.max_steps if s.training.max_steps > 0 else -1,
        "num_train_epochs": s.training.num_train_epochs,
        "learning_rate": s.training.learning_rate,
        "weight_decay": s.training.weight_decay,
        "warmup_steps": s.training.warmup_steps,
        "warmup_ratio": s.training.warmup_ratio,
        "optim": s.training.optim,
        "lr_scheduler_type": s.training.lr_scheduler_type,
        "logging_steps": s.training.logging_steps,
        "save_steps": s.training.save_steps,
        "bf16": s.training.bf16,
        "fp16": s.training.fp16,
        "seed": s.training.seed,
        "output_dir": s.training.output_dir,
        "overwrite_output_dir": s.training.overwrite_output_dir,
        "dataloader_num_workers": s.windows.dataloader_num_workers if os.name == "nt" else 4,
    }


# =============================================================================
# TRAINING PRESETS (Phase 1.2)
# =============================================================================
# Research shows LoRA works best with effective batch size 8-32
# See: https://arxiv.org/abs/2512.23017, Unsloth hyperparameters guide



@dc_dataclass
class TrainingPreset:
    """Training configuration preset for different use cases."""
    name: str
    description: str
    # LoRA
    lora_r: int
    lora_alpha: int
    # Batch
    batch_size: int
    gradient_accumulation: int
    # Learning rate
    learning_rate: float
    warmup_steps: int
    # Multi-run
    steps_per_run: int
    num_runs: int
    # Optional
    samples_per_run: int = 1000
    replay_fraction: float = 0.0
    validate_every_run: bool = False

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size (batch_size * gradient_accumulation)."""
        return self.batch_size * self.gradient_accumulation


# =============================================================================
# LORA PRESETS (v1.3 BACKEND-1)
# =============================================================================
# Decoupled from MULTI_RUN_PRESETS (which govern the training loop —
# steps/runs/batch; legacy name was TRAINING_PRESETS, see v1.4 rename)
# so an operator can opt into "fast" LoRA shape (rank 16 + q+v) without
# also pinning batch size or step count. The "fast" preset reproduces
# the pre-v1.3 LoRAConfig defaults so anyone who liked the speed-tilted
# behavior can opt in via ``--lora-preset=fast``.


@dc_dataclass
class LoRAPreset:
    """LoRA shape preset (rank + target_modules + LR multiplier).

    Used by ``--lora-preset {fast,quality}`` to swap an entire set of
    LoRA defaults at the CLI without forcing operators to pass each
    knob individually. ``lr_multiplier`` is applied on top of whichever
    base LR the operator supplies (CLI ``--lr`` or
    ``settings.training.learning_rate``); a multiplier of 1.0 means
    "use the base LR unchanged."
    """

    name: str
    description: str
    r: int
    lora_alpha: int
    target_modules: str | list[str]
    lr_multiplier: float


LORA_PRESETS: dict[str, LoRAPreset] = {
    "fast": LoRAPreset(
        name="fast",
        description=(
            "Pre-v1.3 defaults — rank 16 + q+v target modules + 1x LR. "
            "Materially faster per step + lower VRAM, but leaves ~15-20% "
            "post-training quality on the table vs. the quality preset. "
            "Pick when you're iterating on data formatting / pipeline "
            "wiring and want the fastest possible signal."
        ),
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lr_multiplier=1.0,
    ),
    "quality": LoRAPreset(
        name="quality",
        description=(
            "v1.3 default — rank 256 + all-linear target + 10x LR. "
            "Matches full fine-tuning quality on most post-training "
            "tasks per Biderman 2024 + Thinking Machines 2025 at ~67% "
            "of the compute. Slightly more VRAM (verify it fits your "
            "card before kicking off a long run)."
        ),
        r=256,
        lora_alpha=512,
        target_modules="all-linear",
        lr_multiplier=10.0,
    ),
}


def get_lora_preset(name: str) -> LoRAPreset:
    """Get a LoRA shape preset by name.

    Args:
        name: Preset name ("fast" or "quality").

    Returns:
        LoRAPreset.

    Raises:
        ValueError: If preset name is not recognized.
    """
    if name not in LORA_PRESETS:
        available = ", ".join(LORA_PRESETS.keys())
        raise ValueError(
            f"Unknown LoRA preset {name!r}. Available: {available}"
        )
    return LORA_PRESETS[name]


# =============================================================================
# MODEL PRESETS (v1.3 BACKEND-2 + 8 + 9 + 10)
# =============================================================================
# Operator-facing catalog of model_id presets the trainer knows about.
# Each preset carries: the HF model_id, recommended LoRA rank /
# max_seq_length / packing toggle, an "best for X" hint, and an
# OPTIONAL license_restriction string. When non-None, Trainer.__init__
# prints the caveat at boot so an operator using a non-commercial
# model for commercial training cannot miss it.


@dc_dataclass
class ModelPreset:
    """Model preset entry (model_id + recommendations + license metadata)."""

    name: str
    model_id: str
    description: str
    # License SPDX identifier (e.g. "Apache-2.0", "MIT", "Qwen-Research").
    license: str
    # Recommended LoRA rank for this model. Smaller models (< 4B) should
    # cap at 128 to avoid wasting parameters on layers that don't have
    # enough information capacity to use them.
    recommended_lora_r: int
    # Recommended max_seq_length. Defaults to 2048 (matches the
    # ModelConfig default) for legacy models; native-long-context
    # models (SmolLM3 64K, Qwen3.5 native long) can use larger values.
    recommended_max_seq_length: int
    # Recommended packing toggle. True for most models (1.7-3× throughput
    # per the v1.3 BACKEND-4 rationale); False only for models with
    # known boundary-token-leakage issues.
    recommended_packing: bool
    # "Best for X" hint surfaced to the operator. One short sentence.
    best_for: str
    # OPTIONAL license caveat. When non-None, Trainer.__init__ prints
    # this text at boot so an operator using e.g. Qwen2.5-3B (which is
    # released under the Qwen-Research non-commercial license) for
    # commercial training cannot silently violate the license. Default
    # None (no caveat needed for permissive licenses).
    license_restriction: str | None = None


# Research-backed catalog. New v1.3 presets land here (Phi-4-mini-3.8B,
# Qwen-3.5-4B, SmolLM3-3B) alongside the existing v1.2 presets that the
# MULTI_RUN_PRESETS table (legacy name: TRAINING_PRESETS) referred to in
# prose.
MODEL_PRESETS: dict[str, ModelPreset] = {
    "qwen2.5-7b": ModelPreset(
        name="qwen2.5-7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        description="Qwen2.5 7B Instruct — canonical 16GB target",
        license="Apache-2.0",
        recommended_lora_r=256,
        recommended_max_seq_length=2048,
        recommended_packing=True,
        best_for="General-purpose post-training on a 16GB card (RTX 5080 / 4080).",
    ),
    "qwen2.5-3b": ModelPreset(
        name="qwen2.5-3b",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        description="Qwen2.5 3B Instruct — small, fast iteration",
        # SPDX-ish marker; Qwen-Research is non-OSI but the catalog
        # field is descriptive, not normative.
        license="Qwen-Research",
        recommended_lora_r=128,
        recommended_max_seq_length=2048,
        recommended_packing=True,
        best_for="Fast iteration on workflow plumbing when you don't need 7B quality.",
        # v1.3 BACKEND-2: surface the Qwen-Research non-commercial
        # caveat at Trainer boot so a commercial-use operator cannot
        # miss it. Operators who need a commercial 3B should pick
        # phi-4-mini-3.8b (MIT) or qwen3.5-4b (Apache 2.0).
        license_restriction=(
            "WARNING: Qwen-Research license — non-commercial only. See "
            "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE "
            "for terms. For commercial use, pick phi-4-mini-3.8b (MIT) or "
            "qwen3.5-4b (Apache 2.0) — both new in v1.3."
        ),
    ),
    "llama-3.2-3b": ModelPreset(
        name="llama-3.2-3b",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        description="Llama 3.2 3B Instruct — Meta's small instruct model",
        license="Llama-3.2-Community",
        recommended_lora_r=128,
        recommended_max_seq_length=2048,
        recommended_packing=True,
        best_for="Mainstream small-model baseline; ecosystem-rich Meta tooling.",
    ),
    "llama-3.2-1b": ModelPreset(
        name="llama-3.2-1b",
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        description="Llama 3.2 1B Instruct — tightest VRAM footprint",
        license="Llama-3.2-Community",
        recommended_lora_r=64,
        recommended_max_seq_length=2048,
        recommended_packing=True,
        best_for="Edge / sub-8GB VRAM cards; pipeline smoke tests.",
    ),
    "mistral-7b": ModelPreset(
        name="mistral-7b",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        description="Mistral 7B Instruct v0.3 — Apache-licensed 7B alternative",
        license="Apache-2.0",
        recommended_lora_r=256,
        recommended_max_seq_length=2048,
        recommended_packing=True,
        best_for="Apache-2.0 7B when you want a non-Qwen baseline.",
    ),
    # ----- v1.3 BACKEND-8: Phi-4-mini-3.8B (MIT, commercial-safe) -----
    "phi-4-mini-3.8b": ModelPreset(
        name="phi-4-mini-3.8b",
        model_id="microsoft/Phi-4-mini-instruct",
        description="Phi-4-mini 3.8B Instruct — Microsoft's MIT-licensed small model",
        license="MIT",
        # 3.8B sits between Llama-3.2-3B and Qwen2.5-7B — recommend 128
        # so the LoRA adapter has room without bloating the parameter
        # count beyond what 3.8B can use.
        recommended_lora_r=128,
        recommended_max_seq_length=2048,
        recommended_packing=True,
        best_for=(
            "Commercial-safe 3.8B with strong reasoning. MMLU ~73, "
            "HumanEval ~74.4. Drop-in for qwen2.5-3b when you need an "
            "OSI license."
        ),
    ),
    # ----- v1.3 BACKEND-9: Qwen-3.5-4B (Apache 2.0) -----
    "qwen3.5-4b": ModelPreset(
        name="qwen3.5-4b",
        model_id="Qwen/Qwen3.5-4B-Instruct",
        description="Qwen 3.5 4B Instruct — Apache-2.0 4B with native long context",
        license="Apache-2.0",
        recommended_lora_r=128,
        # Qwen 3.5 supports native long context out of the box; bump
        # the suggestion from 2048 to 4096 so operators don't leave
        # context on the table for chat / RAG workloads. Operators can
        # still override down to 2048 for tighter VRAM.
        recommended_max_seq_length=4096,
        recommended_packing=True,
        best_for=(
            "Commercial-safe 4B with native long context and strong "
            "MMLU-Pro (~79.1). Best Apache-2.0 4B at v1.3 launch."
        ),
    ),
    # ----- v1.3 BACKEND-10: SmolLM3-3B (Apache 2.0) -----
    "smollm3-3b": ModelPreset(
        name="smollm3-3b",
        model_id="HuggingFaceTB/SmolLM3-3B",
        description="SmolLM3 3B — HuggingFace's Apache-2.0 small with 64K context",
        license="Apache-2.0",
        recommended_lora_r=128,
        # SmolLM3 ships native 64K context; lift recommendation to
        # 8192 (still well under the 64K ceiling) so operators
        # eyeballing the preset see the model's character.
        recommended_max_seq_length=8192,
        recommended_packing=True,
        best_for=(
            "Long-context 3B — beats Llama-3.2-3B + Qwen-2.5-3B at "
            "the same parameter count. Pick when you need 32K+ context "
            "in a small model."
        ),
    ),
}


def get_model_preset(name: str) -> ModelPreset:
    """Get a model preset by name.

    Args:
        name: Preset name (one of MODEL_PRESETS.keys()).

    Returns:
        ModelPreset.

    Raises:
        ValueError: If preset name is not recognized.
    """
    if name not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise ValueError(
            f"Unknown model preset {name!r}. Available: {available}"
        )
    return MODEL_PRESETS[name]


def lookup_model_preset_by_id(model_id: str) -> ModelPreset | None:
    """Return the ModelPreset whose ``model_id`` matches ``model_id``.

    Used by Trainer.__init__ to surface license caveats when the
    operator passes a HF model_id directly (instead of by preset name).
    Match is case-insensitive on the model_id to avoid silent misses
    on capitalization-only differences. Returns None when no preset
    knows about the model.
    """
    needle = model_id.strip().lower()
    for preset in MODEL_PRESETS.values():
        if preset.model_id.lower() == needle:
            return preset
    return None


# Research-backed presets based on SLAO paper, Unsloth docs, and Databricks
# guide. Canonical name as of v1.4 is ``MULTI_RUN_PRESETS`` — these govern
# the multi-run training LOOP (steps_per_run / num_runs / samples_per_run /
# replay_fraction). The architecture-shape preset ``LORA_PRESETS`` (rank,
# target_modules, lr_multiplier — surfaced via the CLI flag ``--lora-preset``)
# lives just above and is INDEPENDENT: pick one of each.
#
# The two namespaces formerly shared the keys ``"fast"`` and ``"quality"``
# with semantically different values, which surfaced as a Wave 5 audit
# finding (operator-trap class). The legacy ``TRAINING_PRESETS`` name
# continues to resolve via the module-level ``__getattr__`` shim at the
# bottom of this file + emits a ``DeprecationWarning``.
MULTI_RUN_PRESETS = {
    "fast-3b": TrainingPreset(
        name="fast-3b",
        description="Ultra-fast with Qwen2.5-3B (~6GB VRAM) for rapid iteration",
        lora_r=8,
        lora_alpha=16,
        batch_size=4,  # 3B model fits larger batches
        gradient_accumulation=2,  # effective=8
        learning_rate=5e-4,
        warmup_steps=5,
        steps_per_run=50,
        num_runs=3,
        samples_per_run=500,
    ),
    "fast": TrainingPreset(
        name="fast",
        description="Quick iterations for testing and debugging (7B model)",
        lora_r=8,
        lora_alpha=16,
        batch_size=2,
        gradient_accumulation=4,  # effective=8
        learning_rate=5e-4,
        warmup_steps=5,
        steps_per_run=50,
        num_runs=3,
        samples_per_run=500,
    ),
    "balanced": TrainingPreset(
        name="balanced",
        description="Default recommended preset for most use cases",
        lora_r=16,
        lora_alpha=32,
        batch_size=2,
        gradient_accumulation=8,  # effective=16
        learning_rate=2e-4,
        warmup_steps=10,
        steps_per_run=100,
        num_runs=5,
        samples_per_run=1000,
    ),
    "quality": TrainingPreset(
        name="quality",
        description="Maximum training effectiveness for final models",
        lora_r=32,
        lora_alpha=64,
        batch_size=4,
        gradient_accumulation=8,  # effective=32
        learning_rate=1e-4,
        warmup_steps=20,
        steps_per_run=200,
        num_runs=10,
        samples_per_run=2000,
        replay_fraction=0.1,
        validate_every_run=True,
    ),
}


def get_preset(name: str) -> TrainingPreset:
    """
    Get a multi-run training preset by name.

    The preset table was renamed from ``TRAINING_PRESETS`` to
    ``MULTI_RUN_PRESETS`` in v1.4 to disambiguate from the v1.3-era
    ``LORA_PRESETS`` (LoRA-shape preset). The legacy
    ``backpropagate.config.TRAINING_PRESETS`` name continues to resolve
    via module-level ``__getattr__`` + ``DeprecationWarning`` until v1.6.

    Args:
        name: Preset name ("fast", "balanced", or "quality")

    Returns:
        TrainingPreset configuration

    Raises:
        ValueError: If preset name is not recognized

    Example:
        >>> preset = get_preset("balanced")
        >>> trainer = Trainer(
        ...     lora_r=preset.lora_r,
        ...     learning_rate=preset.learning_rate,
        ... )
    """
    if name not in MULTI_RUN_PRESETS:
        available = ", ".join(MULTI_RUN_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return MULTI_RUN_PRESETS[name]


def get_recommended_lr(
    dataset_size: int, base_lr: float = 2e-4, method: str = "sft"
) -> float:
    """
    Get recommended learning rate based on dataset size (Phase 1.3).

    Research shows LoRA benefits from ~10× higher LR than full fine-tuning,
    but should be adjusted based on dataset size to prevent overfitting.

    Args:
        dataset_size: Number of training samples
        base_lr: Base learning rate (default: 2e-4). Ignored when
            ``method == "orpo"`` (the ORPO ladder is anchored on its own
            published settings, not scaled off the SFT base).
        method: Training objective (v1.5 T1.2). ``"sft"`` (default) returns
            the SFT ladder UNCHANGED from earlier releases. ``"orpo"``
            returns the ORPO ladder (small=2e-5, medium=1e-5, large=5e-6) —
            roughly an order of magnitude below the SFT LRs because ORPO's
            odds-ratio loss is sensitive to large steps (Hong, Lee & Thorne
            2024, arXiv:2403.07691, train Mistral-ORPO around 5e-6 / 8e-6).

    Returns:
        Recommended learning rate

    Reference:
        - Small datasets (<1K): Higher LR with more warmup to learn quickly
        - Medium datasets (1K-10K): Standard LR
        - Large datasets (>10K): Lower LR for stability

    Example:
        >>> lr = get_recommended_lr(500)  # Returns 5e-4
        >>> lr = get_recommended_lr(5000)  # Returns 2e-4
        >>> lr = get_recommended_lr(50000)  # Returns 1e-4
        >>> lr = get_recommended_lr(500, method="orpo")  # Returns 2e-5
    """
    if method == "orpo":
        # v1.5 T1.2: ORPO ladder. Fixed anchors (not base_lr-scaled) — ORPO's
        # odds-ratio penalty is unstable at the SFT LR magnitudes, so the
        # published runs sit ~10x lower. Same monotone shape (small corpora
        # tolerate a higher LR) as the SFT ladder.
        small_lr = 2e-5
        medium_lr = 1e-5
        large_lr = 5e-6
        # Keep the same strict-monotone invariant the SFT branch asserts so an
        # anchor regression on the ORPO ladder is caught at the source too.
        assert small_lr > medium_lr > large_lr, (
            "get_recommended_lr ORPO ladder must satisfy small > medium > "
            f"large (got {small_lr} / {medium_lr} / {large_lr})"
        )
        if dataset_size < 1000:
            return small_lr
        elif dataset_size < 10000:
            return medium_lr
        else:
            return large_lr

    # Scale all ranges proportionally relative to base_lr.
    # The default base_lr (2e-4) maps to: small=5e-4, medium=2e-4, large=1e-4.
    # A custom base_lr scales these proportionally (e.g. base_lr=4e-4 gives small=1e-3).
    scale = base_lr / 2e-4

    # DATA-A-008: the medium branch must apply ``scale`` like the other two
    # branches. ``2e-4 * scale`` is numerically identical to the prior bare
    # ``base_lr`` (since ``scale == base_lr / 2e-4``), but writing it
    # explicitly keeps all three anchors (5e-4 / 2e-4 / 1e-4) visibly on the
    # same ``* scale`` footing — a future edit to one anchor can no longer
    # silently desync the medium tier from small/large.
    small_lr = 5e-4 * scale
    medium_lr = 2e-4 * scale
    large_lr = 1e-4 * scale
    # The recommended ladder is strictly monotone (smaller corpora tolerate a
    # higher LR). This holds for any positive base_lr; assert it so a sign /
    # anchor regression is caught at the source instead of mid-training.
    assert small_lr > medium_lr > large_lr, (
        "get_recommended_lr ladder must satisfy small > medium > large "
        f"(got {small_lr} / {medium_lr} / {large_lr})"
    )

    if dataset_size < 1000:
        # Small dataset: higher LR, more aggressive learning
        return small_lr
    elif dataset_size < 10000:
        # Medium dataset: standard LoRA LR
        return medium_lr
    else:
        # Large dataset: lower LR for stability
        return large_lr


def get_recommended_warmup(dataset_size: int, num_steps: int) -> int:
    """
    Get recommended warmup steps based on dataset size (Phase 1.3).

    Small datasets need more warmup to prevent early instability.
    Large datasets can use standard warmup ratios.

    Args:
        dataset_size: Number of training samples
        num_steps: Total number of training steps

    Returns:
        Recommended warmup steps

    Example:
        >>> warmup = get_recommended_warmup(500, 100)  # Returns 15 (15%)
        >>> warmup = get_recommended_warmup(5000, 100)  # Returns 10 (10%)
        >>> warmup = get_recommended_warmup(50000, 100)  # Returns 5 (5%)
    """
    if dataset_size < 1000:
        # Small dataset: 15% warmup
        ratio = 0.15
    elif dataset_size < 10000:
        # Medium dataset: 10% warmup
        ratio = 0.10
    else:
        # Large dataset: 5% warmup
        ratio = 0.05

    return max(1, int(num_steps * ratio))


# =============================================================================
# LEGACY ALIAS SHIM (v1.4 rename — Wave 6a foundation, Wave 5 Decision 3)
# =============================================================================
#
# ``TRAINING_PRESETS`` was renamed to ``MULTI_RUN_PRESETS`` to disambiguate
# from ``LORA_PRESETS`` (LoRA-shape preset, surfaced via the CLI flag
# ``--lora-preset``). Both formerly shared the keys ``"fast"`` + ``"quality"``
# with semantically different values — a Wave 5 audit finding flagged the
# collision as an operator-trap class.
#
# Deprecation cycle (advisor 2026-05-25 Q4; removal version revised
# 2026-06-20 to track the actual ship schedule):
#   v1.4 → present → DeprecationWarning (silent by default; -W default shows it)
#   future release (v1.7 or later) → AttributeError (legacy name removed)

_LEGACY_CONFIG_ALIASES: dict[str, str] = {
    "TRAINING_PRESETS": "MULTI_RUN_PRESETS",
}


def __getattr__(name: str) -> Any:
    """Module-level legacy-alias resolver (PEP 562).

    Resolves the legacy ``TRAINING_PRESETS`` name to its v1.4 canonical
    ``MULTI_RUN_PRESETS`` form while emitting a ``DeprecationWarning`` at
    the import / attribute-access site. ``stacklevel=2`` points the
    warning at the caller, not this function.

    Anything not in ``_LEGACY_CONFIG_ALIASES`` raises ``AttributeError``
    per the PEP 562 contract so ``hasattr`` / ``getattr`` with a default
    still work naturally.
    """
    if name in _LEGACY_CONFIG_ALIASES:
        new_name = _LEGACY_CONFIG_ALIASES[name]
        import warnings as _warnings

        _warnings.warn(
            f"{name!r} is deprecated in v1.4; use {new_name!r} instead. "
            f"The legacy name will be removed in a future release "
            f"(v1.7 or later). "
            f"Note: this is the multi-run-loop preset table, NOT the "
            f"LoRA-shape preset 'LORA_PRESETS' (surfaced via --lora-preset).",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(
        f"module 'backpropagate.config' has no attribute {name!r}"
    )
