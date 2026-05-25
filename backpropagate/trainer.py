"""
Backpropagate - Core Trainer Module
====================================

Production-ready LLM fine-tuning with smart defaults and Windows support.

Usage:
    from backpropagate import Trainer

    # Simple usage
    trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    trainer.train("my_data.jsonl")
    trainer.save("./my-model")

    # With options
    trainer = Trainer(
        model="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        lora_r=32,
        learning_rate=1e-4,
    )
    trainer.train(dataset="my_data.jsonl", steps=200)
    trainer.export("gguf")  # Export to GGUF for Ollama

Features:
- Auto VRAM detection for batch size
- Windows-safe multiprocessing
- QLoRA with Unsloth optimization
- Multiple export formats (LoRA, merged, GGUF)
"""

from __future__ import annotations

import gc
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .export import ExportResult
    from .multi_run import MultiRunResult

from .checkpoints import RunHistoryManager
from .config import settings
from .datasets import DatasetLoader
from .exceptions import (
    BackpropagateError,
    DatasetError,
    DatasetNotFoundError,
    DatasetParseError,
    GPUNotAvailableError,
    InvalidSettingError,
    ModelLoadCauseCategory,
    ModelLoadError,
    TrainingAbortedError,
    TrainingError,
)
from .feature_flags import check_feature
from .gpu_safety import check_gpu_safe
from .logging_config import bind_run_context, unbind_run_context

logger = logging.getLogger(__name__)


# =============================================================================
# B-017 HUGGINGFACE HUB TRANSIENT RETRY
# =============================================================================
# HF Hub has ~99.5% uptime with periodic 503/timeout spikes that last 30-60s.
# A multi-hour training job that starts by loading a model + dataset will die
# in the first 30s if HF blips during the load. tenacity wraps the call with
# exponential backoff so a transient blip becomes a logged WARN instead of a
# session-killing exception.
#
# Wrapped surfaces: model from_pretrained, dataset_load, snapshot_download.

_RETRY_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 5
_RETRY_MAX_SECONDS = 60
_RETRY_MULTIPLIER = 2


def _hf_transient_exceptions() -> tuple[type[BaseException], ...]:
    """Return the exception classes tenacity should consider for HF calls.

    Note this returns the *candidate* set; status-code filtering happens in
    ``_is_transient_hf_exception`` so we don't over-retry 401/403/404.
    Imported lazily so the trainer module doesn't hard-require huggingface_hub
    or requests at import time.
    """
    excs: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import requests  # type: ignore

        excs.extend([
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
        ])
    except ImportError:
        pass
    try:
        from huggingface_hub.utils import HfHubHTTPError  # type: ignore

        excs.append(HfHubHTTPError)
    except ImportError:
        pass
    return tuple(excs)


def _is_transient_hf_exception(exc: BaseException) -> bool:
    """Decide whether ``exc`` is worth retrying as a transient HF failure.

    Connection / timeout errors → always retry. HTTPError-shaped exceptions
    (requests.HTTPError, HfHubHTTPError) → retry ONLY on status 429 / 5xx /
    unknown. Skips 4xx (401 auth, 403 gated repo, 404 typo'd model) so users
    see the real error in <1s instead of waiting ~65s for the backoff to
    exhaust.
    """
    transient_excs = _hf_transient_exceptions()
    if not isinstance(exc, transient_excs):
        return False
    # If the exception carries an HTTP response, inspect its status code.
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        # No status code attached → connection/timeout/other → retry
        return True
    return bool(status == 429 or status >= 500)


def _retry_hf_call(
    fn: Callable[..., Any],
    *args: Any,
    _label: str = "hf_call",
    **kwargs: Any,
) -> Any:
    """B-017: invoke ``fn(*args, **kwargs)`` with tenacity-based retry.

    Retries on transient HF Hub failures (5xx, 429, connection timeouts) up
    to ``_RETRY_ATTEMPTS`` attempts with exponential backoff. Each retry is
    logged at WARN with the URL hint, status (if available), and delay.
    Non-transient exceptions and the final failure propagate to the caller
    untouched.

    Args:
        fn: Callable to invoke (e.g. ``FastLanguageModel.from_pretrained``).
        *args / **kwargs: Forwarded to ``fn``.
        _label: Short string identifying the call site (used in retry logs).
    """
    from tenacity import (
        before_sleep_log,
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential,
    )

    transient_excs = _hf_transient_exceptions()

    @retry(
        stop=stop_after_attempt(_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=_RETRY_MULTIPLIER,
            min=_RETRY_BASE_SECONDS,
            max=_RETRY_MAX_SECONDS,
        ),
        retry=retry_if_exception(_is_transient_hf_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call() -> Any:
        return fn(*args, **kwargs)

    try:
        return _call()
    except transient_excs as exc:
        # Last attempt exhausted — log a structured failure line before
        # re-raising so triage doesn't have to scroll back through retries.
        logger.error(
            f"HF transient retry exhausted: label={_label} "
            f"err={type(exc).__name__}: {exc}"
        )
        raise


# =============================================================================
# F-019 ModelLoadError CAUSE CLASSIFICATION
# =============================================================================
# exceptions.py:67 defines ModelLoadCauseCategory =
#     Literal["auth", "not_found", "network", "version", "unknown"]
# and exceptions.py:285 carries a per-category remediation hint table. The
# raise sites below classify the underlying exception so the operator gets
# the right hint instead of the generic "check model name + network" line.
#
# We are defensive about huggingface_hub / requests availability: both
# imports are optional in the trainer's static dep set (the [unsloth] /
# [validation] extras pull them in, but a headless trainer.py import must
# survive without them). The classifier wraps each isinstance check in a
# try/except ImportError and degrades gracefully to "unknown" rather than
# raising at import-time when an upstream dep is missing.


def _classify_model_load_cause(exc: BaseException) -> ModelLoadCauseCategory:
    """Best-effort classification of a model-load exception.

    Returns one of the ``ModelLoadCauseCategory`` Literal values:
    ``"auth"`` | ``"not_found"`` | ``"network"`` | ``"version"`` | ``"unknown"``.

    Classification rules (in order — first match wins):
    * ``HfHubHTTPError`` with status 401 → ``"auth"``
    * ``HfHubHTTPError`` with status 403 → ``"auth"`` (gated repo)
    * ``HfHubHTTPError`` with status 404 → ``"not_found"``
    * ``requests.ConnectionError`` / ``Timeout`` / builtin
      ``ConnectionError`` / ``TimeoutError`` → ``"network"``
    * ``ImportError`` → ``"version"`` (transformers/peft/unsloth missing
      or version mismatch)
    * anything else → ``"unknown"``
    """
    # 1. HfHubHTTPError with HTTP status carries the most signal.
    try:
        from huggingface_hub.utils import HfHubHTTPError  # type: ignore

        if isinstance(exc, HfHubHTTPError):
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status == 401:
                return "auth"
            if status == 403:
                # Gated-repo failures (e.g. Llama 3 without acceptance)
                # surface as 403; from the user's POV the remediation is
                # still "fix your HF auth / request access" so we route
                # to the auth hint.
                return "auth"
            if status == 404:
                return "not_found"
            # 5xx and unknown statuses fall through to network/unknown.
            if status is not None and status >= 500:
                return "network"
    except ImportError:
        pass

    # 2. Generic requests-shaped network errors.
    try:
        import requests  # type: ignore

        if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
            return "network"
        # requests.HTTPError carries a response — try to peek at status.
        if isinstance(exc, requests.HTTPError):
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status == 401 or status == 403:
                return "auth"
            if status == 404:
                return "not_found"
            if status is not None and status >= 500:
                return "network"
    except ImportError:
        pass

    # 3. Builtin connection / timeout (raised by lower-level sockets).
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "network"

    # 4. ImportError — typically a transformers/peft/unsloth version
    # mismatch surfaced as a missing symbol on a deferred import. The
    # remediation is "upgrade / reinstall the package" so we route to
    # the version hint.
    if isinstance(exc, ImportError):
        return "version"

    # 5. Default fall-through. The generic hint already covers
    # "check the model name and your network" which is the right
    # advice when we genuinely don't know.
    return "unknown"


__all__ = [
    "Trainer",
    "TrainingRun",
    "TrainingCallback",
    "load_model",
    "load_dataset",
    "MultiRunTrainer",
    "SpeedrunTrainer",  # Backwards compatibility
]


def _compute_dataset_hash(dataset: Any) -> str | None:
    """Best-effort sha256-prefix of a dataset, if it's a local file.

    F-003 / F-004: the model card and run-history layer both want a
    stable identifier for the training data. When the dataset is a path
    to a local file we hash the bytes; for HF dataset names / in-memory
    Dataset objects / DatasetLoader instances we return ``None`` and let
    the model card mark the provenance as "remote dataset" rather than
    making up a hash.

    Returns the first 16 hex chars of sha256(file_bytes) on success,
    ``None`` if the dataset is not a hashable local artefact or the hash
    fails for any reason (best-effort — never raises).
    """
    import hashlib

    if not isinstance(dataset, (str, Path)):
        return None
    path = Path(dataset)
    if not path.exists() or not path.is_file():
        return None
    try:
        sha = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                sha.update(chunk)
        return sha.hexdigest()[:16]
    except (OSError, PermissionError) as exc:
        logger.debug(f"_compute_dataset_hash failed for {path}: {exc}")
        return None


# F-014: chat-template marker detection for ``train_on_responses_only``.
# Unsloth's masker needs literal substrings that uniquely tag the start of the
# user turn (``instruction_part``) and the start of the assistant turn
# (``response_part``). Hardcoding ChatML (`<|im_start|>user` / `<|im_start|>assistant`)
# silently no-ops on Llama 3 / Gemma / Llama 4, where loss then leaks back onto
# the user prompt — opposite of what the operator asked for.
#
# Detection strategy (family-name table primary, ChatML probe secondary —
# chosen over a probe-and-slice design because each chat template has its own
# quirks; a single slicer that handles all of them within 2 hours of work
# proved fragile in the F-014 implementation pass):
#
# 1. Match tokenizer.name_or_path / class name against a curated family
#    table verified against upstream chat templates.
# 2. If no family match, run a quick apply_chat_template render and check
#    for canonical ChatML tokens around the probe sentinels — catches small
#    fine-tunes that ship a ChatML template under an unbranded name.
# 3. Final fallback: ChatML markers + a WARN log.
#
# Operators always have an authoritative override via
# ``Trainer(response_markers=(instr, resp))`` that bypasses detection.

# Probe sentinels chosen to be (a) ASCII letters only so no tokenizer
# tokenizes them away and (b) long enough that they're unlikely to collide
# with any literal text inside a real chat template.
_CHAT_MARKER_PROBE_USER = "ZZZUSERPROBE"
_CHAT_MARKER_PROBE_ASST = "ZZZASSISTANTPROBE"

# Family table — PRIMARY detection path. Keys are lowercase substrings to
# match against tokenizer.name_or_path / class name. Order matters: more
# specific families before less specific (``llama-3`` before ``llama``).
# Marker pairs were verified against the upstream chat_template on
# Hugging Face as of 2026-05; new families should be added here.
_CHAT_MARKER_FAMILY_TABLE: tuple[tuple[str, tuple[str, str]], ...] = (
    # Llama 3 / 3.1 / 3.2 — header-id format.
    ("llama-3", ("<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>")),
    ("llama3", ("<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>")),
    # Gemma 1/2/3 — start-of-turn format; assistant turn labelled ``model``.
    ("gemma", ("<start_of_turn>user", "<start_of_turn>model")),
    # Qwen 2 / 2.5 / 3 — ChatML.
    ("qwen", ("<|im_start|>user", "<|im_start|>assistant")),
    # Mistral / Mixtral — ``[INST]`` instruction wrapping; see WARN at detect
    # time (Unsloth's matcher does not mask Mistral templates reliably).
    ("mistral", ("[INST]", "[/INST]")),
    # Phi 3 — ChatML-adjacent format used by Microsoft.
    ("phi-3", ("<|user|>", "<|assistant|>")),
    ("phi3", ("<|user|>", "<|assistant|>")),
    # Generic ChatML hint for fine-tunes that explicitly label themselves
    # as ChatML in the model name. Most ChatML tokenizers ship under a
    # vendor name (e.g. ``qwen``) and hit one of the entries above first.
    ("chatml", ("<|im_start|>user", "<|im_start|>assistant")),
)

# ChatML fallback when nothing else matches. Logged at WARN so the operator
# sees the situation and can supply an explicit override via the new kwarg.
_CHAT_MARKER_DEFAULT = ("<|im_start|>user", "<|im_start|>assistant")


def _detect_chat_markers(tokenizer: Any) -> tuple[str, str]:
    """Return ``(instruction_marker, response_marker)`` for the tokenizer.

    Strategy (family-name primary; ChatML-shape probe secondary):

        1. **Primary** — match the tokenizer's ``name_or_path`` / class name
           against :data:`_CHAT_MARKER_FAMILY_TABLE`. Marker pairs were
           verified against the upstream chat templates. Order in the table
           matters: more-specific keys first (``llama-3`` before ``llama``).
        2. **Secondary** — render a known-shaped dummy conversation through
           ``apply_chat_template`` and check for canonical ChatML tokens.
           This catches small fine-tunes shipping a ChatML template under an
           unbranded name.
        3. **Final fallback** — ChatML markers with a WARN log so the
           operator can supply ``Trainer(response_markers=(instr, resp))``
           if their model uses something else.

    Chosen over a probe-and-slice design because each chat template has
    quirks (Llama 3's lack of newlines, Gemma's ``<start_of_turn>`` non-pipe
    tokens, Mistral's bracket pairs) and a single slicer that handles all of
    them within the F-014 time budget proved fragile. The family table is
    safer + auditable; the probe handles the long tail of ChatML clones.

    The function never raises; failure paths log at WARN/DEBUG and return
    the best available default.
    """
    # --- 1. Family-name detection (primary path) ---
    name = ""
    for attr in ("name_or_path", "init_kwargs"):
        try:
            value = getattr(tokenizer, attr, None)
            if isinstance(value, str):
                name = value
                break
            if isinstance(value, dict):
                name = str(value.get("name_or_path", "")) or str(value.get("_name_or_path", ""))
                if name:
                    break
        except Exception:  # nosec B112 — best-effort tokenizer name probe; next candidate is the right behavior
            continue
    cls_name = type(tokenizer).__name__.lower()
    haystack = f"{name.lower()} {cls_name}"
    for needle, markers in _CHAT_MARKER_FAMILY_TABLE:
        if needle in haystack:
            logger.info(
                f"_detect_chat_markers: family-table match {needle!r} -> {markers}"
            )
            if needle == "mistral":
                logger.warning(
                    "_detect_chat_markers: Mistral [INST]/[/INST] template "
                    "may not mask responses reliably under Unsloth (no "
                    "explicit assistant-turn marker). Consider passing "
                    "Trainer(response_markers=(...)) explicitly or "
                    "train_on_responses=False for Mistral fine-tunes."
                )
            return markers

    # --- 2. ChatML-shape probe (secondary path) ---
    try:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": _CHAT_MARKER_PROBE_USER},
                {"role": "assistant", "content": _CHAT_MARKER_PROBE_ASST},
            ],
            tokenize=False,
        )
    except Exception as exc:
        logger.debug(
            f"_detect_chat_markers: apply_chat_template probe failed ({exc!r})"
        )
        rendered = None

    if rendered and isinstance(rendered, str):
        user_idx = rendered.find(_CHAT_MARKER_PROBE_USER)
        asst_idx = rendered.find(_CHAT_MARKER_PROBE_ASST)
        if (
            0 < user_idx < asst_idx
            and "<|im_start|>user" in rendered[:user_idx]
            and "<|im_start|>assistant" in rendered[user_idx:asst_idx]
        ):
            logger.info("_detect_chat_markers: probe matched ChatML-shaped template")
            return _CHAT_MARKER_DEFAULT

    # --- 3. Final ChatML fallback ---
    logger.warning(
        f"_detect_chat_markers: could not detect chat markers for tokenizer "
        f"{cls_name!r} (name={name!r}); falling back to ChatML "
        f"({_CHAT_MARKER_DEFAULT}). If your model uses a different chat "
        "template, train_on_responses_only will silently no-op. Pass "
        "Trainer(response_markers=(instr, resp)) to override."
    )
    return _CHAT_MARKER_DEFAULT


@dataclass
class TrainingRun:
    """Container for training run results."""
    run_id: str
    steps: int
    final_loss: float
    loss_history: list[float] = field(default_factory=list)
    output_path: str | None = None
    duration_seconds: float = 0.0
    samples_seen: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingCallback:
    """Callback hooks for training events.

    Wired into the underlying TRL/HF Trainer lifecycle via
    :func:`_build_trl_bridge_callback`:

    * ``on_step(step: int, loss: float)`` fires from HF ``on_log`` whenever a
      new loss value lands in ``state.log_history``. The pre-F-003 build
      defined this field but never invoked it.
    * ``on_epoch(epoch: int)`` fires from HF ``on_epoch_end``.
    * ``on_save(checkpoint_path: str)`` fires from HF ``on_save`` with the
      latest ``checkpoint-<step>`` directory.
    * ``on_complete(run: TrainingRun)`` fires from ``Trainer.train`` after the
      result is built. (Already wired pre-F-003.)
    * ``on_error(exc: Exception)`` fires from ``Trainer.train`` exception
      handlers. (Already wired pre-F-003.)

    Each hook is invoked inside its own try/except so a buggy user callback
    cannot poison training — the v1.1.0 callback isolation contract.
    """
    on_step: Callable[[int, float], None] | None = None
    on_epoch: Callable[[int], None] | None = None
    on_save: Callable[[str], None] | None = None
    on_complete: Callable[[TrainingRun], None] | None = None
    on_error: Callable[[Exception], None] | None = None


def _build_trl_bridge_callback(user_callback: TrainingCallback) -> Any:
    """F-003: bridge our :class:`TrainingCallback` onto HF's TrainerCallback API.

    Returns an instance of a private ``TrainerCallback`` subclass that forwards
    ``on_log`` / ``on_epoch_end`` / ``on_save`` events into the user's
    ``on_step`` / ``on_epoch`` / ``on_save`` hooks. Each forward is wrapped in
    a try/except that logs at WARN — a buggy user callback must never abort
    training (v1.1.0 contract).

    Returns ``None`` when transformers is unavailable; the caller should skip
    bridge installation in that case (defensive — transformers is a hard dep
    of TRL so this branch should never fire in practice).

    Built lazily so the trainer module doesn't hard-import transformers at top
    of file (matches the project's general lazy-import convention for heavy
    optional/runtime deps).
    """
    try:
        from transformers import TrainerCallback as _HFTrainerCallback
    except Exception as exc:
        logger.debug(
            f"_build_trl_bridge_callback: transformers TrainerCallback "
            f"unavailable ({exc!r}); on_step/on_epoch/on_save will not fire."
        )
        return None

    class _BackpropCallbackAdapter(_HFTrainerCallback):  # type: ignore[misc, valid-type]
        """Adapter wiring TrainingCallback.{on_step, on_epoch, on_save} into HF."""

        def __init__(self, cb: TrainingCallback) -> None:
            super().__init__()
            self._cb = cb
            # Track the last log entry index we forwarded so on_log can
            # synthesise on_step from log_history. ``on_log`` fires roughly
            # once per logging_steps; the latest entry holds the freshest loss.
            self._last_log_index: int = -1

        # ---- HF lifecycle hooks ----
        def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:  # noqa: D401, ANN401
            if self._cb.on_step is None:
                return
            # Prefer the explicit ``logs`` dict (current HF API); fall back to
            # the tail of ``state.log_history`` for older transformers builds
            # that still populate that path.
            loss_val: float | None = None
            if isinstance(logs, dict) and "loss" in logs:
                try:
                    loss_val = float(logs["loss"])
                except (TypeError, ValueError):
                    loss_val = None
            if loss_val is None:
                history = getattr(state, "log_history", None) or []
                # Walk back from the tail for the most recent entry with a loss.
                for entry in reversed(history[-5:]):
                    if isinstance(entry, dict) and "loss" in entry:
                        try:
                            loss_val = float(entry["loss"])
                            break
                        except (TypeError, ValueError):
                            continue
            if loss_val is None:
                # No usable loss in this log batch (e.g. an eval-only log
                # entry). Skip — on_step semantics require a loss value.
                return
            step = int(getattr(state, "global_step", 0) or 0)
            try:
                self._cb.on_step(step, loss_val)
            except Exception as cb_error:  # noqa: BLE001 — user callback isolation
                logger.warning(
                    f"on_step callback raised error (step={step} loss={loss_val:.4f}): {cb_error}"
                )

        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:  # noqa: ANN401
            if self._cb.on_epoch is None:
                return
            # ``state.epoch`` is a float (e.g. 1.0); cast to int for the
            # documented signature.
            try:
                epoch = int(getattr(state, "epoch", 0) or 0)
            except (TypeError, ValueError):
                epoch = 0
            try:
                self._cb.on_epoch(epoch)
            except Exception as cb_error:  # noqa: BLE001
                logger.warning(f"on_epoch callback raised error (epoch={epoch}): {cb_error}")

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:  # noqa: ANN401
            if self._cb.on_save is None:
                return
            # HF writes checkpoints under output_dir/checkpoint-<step>.
            checkpoint_path = ""
            try:
                output_dir = getattr(args, "output_dir", None)
                step = int(getattr(state, "global_step", 0) or 0)
                if output_dir:
                    checkpoint_path = str(Path(output_dir) / f"checkpoint-{step}")
            except Exception:
                checkpoint_path = ""
            try:
                self._cb.on_save(checkpoint_path)
            except Exception as cb_error:  # noqa: BLE001
                logger.warning(
                    f"on_save callback raised error (path={checkpoint_path!r}): {cb_error}"
                )

    return _BackpropCallbackAdapter(user_callback)


# =============================================================================
# SHARED SFTCONFIG BUILDER (Wave 6a BACKEND-A-003 / A-004 — multi-run refactor)
# =============================================================================
# Pre-Wave-6a, ``MultiRunTrainer._execute_run`` built its own ``SFTConfig`` +
# ``SFTTrainer`` inline (multi_run.py:1347), bypassing the v1.3 BACKEND-5 / 7
# autodetection that ``Trainer.train()`` applied: consumer-card paged-optim
# upgrade (``_detect_optim_for_card``) and Ada-class bf16/fp16 selection
# (``_detect_optimal_dtype``). It also bypassed Unsloth's
# ``train_on_responses_only`` masking despite the docstring claim. The Wave 6a
# refactor extracts the SFTConfig assembly + the train_on_responses_only
# application into shared module-level helpers so both call sites converge
# on the same defaults. Single source of truth = anti-drift.
#
# Why module-level (not a static method on Trainer): MultiRunTrainer needs to
# call this without instantiating a separate Trainer, and the static-method
# form would put a hidden coupling on Trainer's MRO. Plain functions stay
# easy to mock in tests too.


def _build_sft_config(
    output_dir: str,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    learning_rate: float,
    warmup_steps: int,
    max_seq_length: int,
    *,
    seed: int,
    lr_scheduler_type: str,
    logging_steps: int,
    save_steps: int | None = None,
    weight_decay: float | None = None,
    report_to: Any = None,
    run_name: str | None = None,
    packing: bool = False,
    # v1.4 BRIDGE-A-002 follow-up (Wave 6a): optional per-invocation
    # optim override. When None the helper falls back to
    # ``settings.training.optim`` (preserves the pre-Wave-6a env-var path
    # for callers who don't supply the override). Both call sites (single
    # run train() + multi-run _execute_run) now thread the Trainer's
    # ``self.optim`` through so the per-invocation kwarg flows
    # end-to-end.
    optim: str | None = None,
) -> Any:
    """Assemble an ``SFTConfig`` with the v1.3 quality contracts applied.

    The helper resolves two GPU-dependent fields automatically so call sites
    cannot drift apart:

    * ``optim`` — via :meth:`Trainer._detect_optim_for_card`. Operators on
      consumer cards (< 24GB VRAM) get ``paged_adamw_8bit`` automatically
      (v1.3 BACKEND-5); explicit operator overrides on ``settings.training.optim``
      are honored. The detector is a pure function of the configured value
      so the resolution is stable across OOM retries.
    * ``bf16`` / ``fp16`` — via :meth:`Trainer._detect_optimal_dtype`. Ada and
      Hopper cards prefer bf16 over fp16 (v1.3 BACKEND-7); pre-Ampere cards
      get fp16. Explicit operator override (``--fp16``) is honored.

    Everything else is plumbed through unchanged. ``save_steps`` and
    ``weight_decay`` are optional so multi-run can omit them — the SFTConfig
    constructor's defaults then govern.

    Parameters:
        output_dir: Where SFTTrainer writes checkpoints.
        per_device_train_batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation.
        max_steps: Maximum training steps.
        learning_rate: Optimizer learning rate.
        warmup_steps: Warmup steps (run-scoped in multi-run, session-scoped
            in single-run).
        max_seq_length: Maximum input sequence length.
        seed: RNG seed. Multi-run passes ``settings.training.seed +
            run_idx + oom_retries * 1000`` for per-retry shuffle reorder
            (Stage C BACKEND-B-003); single-run passes the unmodified seed.
        lr_scheduler_type: e.g. ``"cosine"``, ``"linear"``, ``"constant"``.
        logging_steps: Logging cadence.
        save_steps: Save cadence (optional). When None, the SFTConfig
            default governs.
        weight_decay: AdamW weight decay (optional). When None, the
            SFTConfig default governs.
        report_to: Pre-resolved report_to value (string, list, or ``"none"``).
        run_name: W&B / experiment-tracker run name.
        packing: Whether SFTTrainer should pack sequences.

    Returns:
        A configured ``SFTConfig`` instance.
    """
    from trl import SFTConfig

    # v1.4 BRIDGE-A-002 follow-up (Wave 6a): honor the per-invocation
    # ``optim`` override when supplied (forwarded by Trainer.train() /
    # MultiRunTrainer._execute_run via the Trainer instance's ``self.optim``,
    # which the constructor pre-resolved from operator kwarg OR
    # ``settings.training.optim``). When ``optim`` is None we read
    # ``settings.training.optim`` directly (pre-Wave-6a behavior preserved
    # for callers that don't thread the per-invocation override).
    _configured_optim = optim if optim is not None else settings.training.optim
    resolved_optim = Trainer._detect_optim_for_card(_configured_optim)
    resolved_bf16, resolved_fp16 = Trainer._detect_optimal_dtype(
        settings.training.bf16, settings.training.fp16
    )

    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "optim": resolved_optim,
        "lr_scheduler_type": lr_scheduler_type,
        "logging_steps": logging_steps,
        "bf16": resolved_bf16,
        "fp16": resolved_fp16,
        "seed": seed,
        "overwrite_output_dir": True,
        "dataloader_num_workers": 0 if os.name == "nt" else 4,
        "report_to": report_to,
        "run_name": run_name,
        # SFT-specific args (TRL 0.27+ moved these from SFTTrainer to SFTConfig)
        "max_length": max_seq_length,
        "packing": packing,
    }
    if save_steps is not None:
        kwargs["save_steps"] = save_steps
    if weight_decay is not None:
        kwargs["weight_decay"] = weight_decay
    return SFTConfig(**kwargs)


def _apply_train_on_responses_only(
    sft_trainer: Any,
    tokenizer: Any,
    *,
    enabled: bool,
    use_unsloth: bool,
    response_markers_override: tuple[str, str] | None = None,
) -> tuple[Any, tuple[str, str] | None]:
    """Apply Unsloth's ``train_on_responses_only`` masking to an SFTTrainer.

    Returns ``(wrapped_trainer, resolved_markers)``. When the masking does
    not apply (Windows / non-Unsloth path / disabled / Unsloth missing /
    detection failed), the original ``sft_trainer`` is returned unchanged
    and ``resolved_markers`` is ``None``.

    Centralizes the v1.4 Wave 6a BACKEND-A-004 fix: pre-refactor, only
    ``Trainer.train()`` applied this masking and ``MultiRunTrainer._execute_run``
    silently skipped it (so multi-run users training on conversational data
    got loss leakage onto the user prompt). Both call sites now share this
    single application path so a future change to the masking surface stays
    in lockstep.

    Parameters:
        sft_trainer: The freshly-constructed ``SFTTrainer`` to wrap.
        tokenizer: The tokenizer (for marker auto-detection when the operator
            did not supply ``response_markers_override``).
        enabled: Operator's ``train_on_responses`` intent (typically
            ``Trainer._train_on_responses``).
        use_unsloth: Whether the underlying model load went through Unsloth
            (the masker is an Unsloth utility; non-Unsloth runs cannot mask).
        response_markers_override: Explicit ``(instruction_part, response_part)``
            tuple, bypassing detection. None ⇒ auto-detect via
            :func:`_detect_chat_markers`.
    """
    if not (use_unsloth and enabled):
        return sft_trainer, None
    if os.name == "nt":
        logger.warning(
            "train_on_responses_only disabled on Windows (multiprocessing issues) "
            "- training will compute loss on full conversations including user prompts, "
            "which may reduce fine-tuning quality"
        )
        return sft_trainer, None
    try:
        from unsloth.chat_templates import train_on_responses_only
        # F-014: derive markers from the tokenizer's chat template so
        # Llama 3 / Gemma / Qwen / ChatML all work; explicit operator
        # override wins.
        if response_markers_override is not None:
            instruction_part, response_part = response_markers_override
            logger.info(
                f"train_on_responses_only: using operator override "
                f"instruction={instruction_part!r} response={response_part!r}"
            )
        else:
            instruction_part, response_part = _detect_chat_markers(tokenizer)
        wrapped = train_on_responses_only(
            sft_trainer,
            instruction_part=instruction_part,
            response_part=response_part,
            num_proc=1,  # Single process to avoid Windows issues
        )
        logger.info("Applied train_on_responses_only optimization")
        return wrapped, (instruction_part, response_part)
    except ImportError:
        logger.warning("train_on_responses_only not available in this Unsloth version")
        return sft_trainer, None
    except Exception as e:
        logger.warning(f"Failed to apply train_on_responses_only: {e}")
        return sft_trainer, None


class Trainer:
    """
    Headless LLM fine-tuning trainer with smart defaults.

    Args:
        model: Model name/path (HuggingFace or local)
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha (default: 32)
        learning_rate: Learning rate (default: 2e-4)
        batch_size: Batch size per device (default: "auto")
        output_dir: Output directory (default: "./output")
        oom_recovery: If True (default), retry on torch.cuda.OutOfMemoryError
            by halving batch_size and doubling gradient_accumulation_steps
            (preserves effective batch). Aborts with
            RUNTIME_OOM_RECOVERY_EXHAUSTED after 3 consecutive failures at
            batch=1 (the recovery loop ran out of options). When False, the
            first uncovered OOM surfaces as
            ``GPUMemoryError(code='RUNTIME_GPU_OOM')`` with the original
            ``torch.cuda.OutOfMemoryError`` (or OOM-adjacent
            ``RuntimeError``) chained as ``__cause__`` — Wave 6a Option A
            activated this raise site so the documented
            ``RUNTIME_GPU_OOM`` contract (README + handbook + cli.py
            exit-code mapper + llms.txt) actually fires from the Python
            codebase. Distinct from the recovery-exhausted path which
            still raises ``TrainingError(code=
            'RUNTIME_OOM_RECOVERY_EXHAUSTED')``. Set False to hard-fail
            on first OOM with the structured-error envelope.
        unsloth_fallback: If True (default), fall back to
            AutoModelForCausalLM + get_peft_model when
            unsloth.FastLanguageModel.from_pretrained fails. Set False to
            hard-fail.

    Production features (Stage B / Stage C, May 2026):
        - Stable error codes via the ERROR_CODES registry; every
          BackpropagateError carries ``code`` / ``message`` / ``hint`` /
          ``cause`` / ``retryable`` for machine-readable triage.
        - Atomic checkpoint writes (B-006): ``save()`` writes into
          ``<path>.partial`` then ``shutil.move()``s into place; crash-safe.
        - HuggingFace Hub transient retry (B-017): from_pretrained calls
          are wrapped with exponential backoff for 5xx / 429 / timeout.
          Auth (401/403) and not-found (404) skip retry.
        - run_id correlation token: every log line, checkpoint manifest,
          and SLAO merge_history entry carries the same UUID4-derived ID
          for cross-surface grep.
        - ModelLoadError carries a ``cause_category`` (auth / not_found /
          network / version / unknown) so the operator gets the right
          remediation hint instead of the generic "check model + network"
          line — see F-019 / the ``_classify_model_load_cause`` helper.

    Example:
        >>> trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
        >>> trainer.train("data.jsonl", steps=100)
        >>> trainer.save("./my-model")
    """

    # B-002: OOM-recovery tuning constants (Trainer mirror of MultiRunTrainer).
    _OOM_MAX_RETRIES_AT_MIN_BATCH = 3

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

    def __init__(
        self,
        model: str | None = None,
        lora_r: int | None = None,
        lora_alpha: int | None = None,
        lora_dropout: float | None = None,
        learning_rate: float | None = None,
        batch_size: int | str = "auto",
        gradient_accumulation: int | None = None,
        max_seq_length: int | None = None,
        output_dir: str | None = None,
        use_unsloth: bool = True,
        train_on_responses: bool = True,  # Phase 1.1: Only compute loss on assistant responses
        # B-002: opt out of OOM recovery (default ON for graceful degradation).
        oom_recovery: bool = True,
        # B-010: opt out of the Unsloth -> transformers fallback (default ON
        # so an Unsloth nightly that breaks loading doesn't take a pipeline
        # down; set False to make Unsloth failures hard-fail for operators
        # who insist on Unsloth's speed).
        unsloth_fallback: bool = True,
        # F-005: experiment-tracker wiring. Default "auto" detects installed
        # trackers via feature_flags (wandb / tensorboard / mlflow) and wires
        # them all. Pass "none" or None to disable. Pass a list of strings
        # (e.g. ["wandb"]) to force a specific tracker set; we DO NOT
        # validate that the tracker is installed in that branch — TRL will
        # raise a clean ImportError if it isn't.
        report_to: str | list[str] | None = "auto",
        # F-014: explicit override for the (instruction_marker, response_marker)
        # pair fed into Unsloth's train_on_responses_only. Default ``None`` ⇒
        # auto-detect via the tokenizer's chat template. Supply a 2-tuple to
        # short-circuit the probe — useful when the detection misfires on an
        # exotic model or when an operator wants to mask a custom turn label.
        response_markers: tuple[str, str] | None = None,
        # v1.4 BRIDGE-A-002 follow-up (Wave 6a foundation): the five Wave 6b
        # knobs were previously env-var-only via ``settings.lora.*`` /
        # ``settings.data.*`` / ``settings.training.*``. The CLI handlers
        # (cmd_train / cmd_multi_run / cmd_replay) all assembled a
        # ``wave6b_candidate_kwargs`` dict and filtered via
        # ``inspect.signature(Trainer.__init__)`` — which silently dropped
        # every key because the constructor signature didn't name them. Now
        # they are explicit constructor parameters with ``None`` defaults;
        # when set they override the settings layer for THIS Trainer
        # instance, when None the ``settings.lora.*`` / ``settings.data.*``
        # / ``settings.training.*`` paths remain the source of truth
        # (preserves v1.3 byte-identical behavior for callers who do not
        # pass these kwargs). Note: ``lora_preset`` is stored for
        # introspection / future overlay wiring — no consumer in trainer.py
        # currently reads it, but threading the value preserves operator
        # intent across the CLI introspection filter.
        use_dora: bool | None = None,
        packing: bool | None = None,
        init_lora_weights: str | None = None,  # "default" | "pissa" | "loftq"
        lora_preset: str | None = None,  # "fast" | "quality" — stored for future overlay wiring
        optim: str | None = None,  # passthrough to TrainingArguments.optim (via _detect_optim_for_card)
    ) -> None:
        # Use settings as defaults, override with provided values
        # NOTE: Use `is not None` checks instead of falsy `or` to allow
        # legitimate zero values (e.g. lora_dropout=0.0, learning_rate=0.0)
        self.model_name = model if model is not None else settings.model.name
        self.lora_r = lora_r if lora_r is not None else settings.lora.r
        self.lora_alpha = lora_alpha if lora_alpha is not None else settings.lora.lora_alpha
        self.lora_dropout = lora_dropout if lora_dropout is not None else settings.lora.lora_dropout
        self.learning_rate = learning_rate if learning_rate is not None else settings.training.learning_rate
        self.gradient_accumulation = gradient_accumulation if gradient_accumulation is not None else settings.training.gradient_accumulation_steps
        self.max_seq_length = max_seq_length if max_seq_length is not None else settings.model.max_seq_length
        self.output_dir = Path(output_dir if output_dir is not None else settings.training.output_dir)
        self.use_unsloth = use_unsloth and check_feature("unsloth")

        # Auto batch size
        if batch_size == "auto":
            self.batch_size: int = self._detect_batch_size()
        else:
            self.batch_size = int(batch_size)

        # Stage C BACKEND-B (constructor validation): catch obviously-broken
        # hyperparameter combinations at construction time with structured
        # errors. Pre-fix these would fail deep inside TRL with opaque error
        # messages (e.g. ``ValueError: batch_size must be positive`` thrown
        # from a SFTConfig field validator several frames deep). Surfacing
        # the problem at ``Trainer(...)`` lets the operator see WHICH knob
        # they got wrong before training spins up the model and tokenizer.
        if self.batch_size <= 0:
            raise InvalidSettingError(
                setting_name="batch_size",
                value=self.batch_size,
                expected="a positive integer (>= 1)",
                suggestion=(
                    "Pass batch_size=1 for tightest VRAM, or omit the "
                    "argument to let _detect_batch_size pick a value."
                ),
            )
        if self.gradient_accumulation <= 0:
            raise InvalidSettingError(
                setting_name="gradient_accumulation",
                value=self.gradient_accumulation,
                expected="a positive integer (>= 1)",
                suggestion="Use gradient_accumulation=1 for no accumulation.",
            )
        if self.learning_rate <= 0:
            raise InvalidSettingError(
                setting_name="learning_rate",
                value=self.learning_rate,
                expected="a positive float (> 0.0)",
                suggestion=(
                    "Try learning_rate=2e-4 (LoRA default) or 5e-5 "
                    "(full-finetune-style)."
                ),
            )
        if self.lora_r <= 0:
            raise InvalidSettingError(
                setting_name="lora_r",
                value=self.lora_r,
                expected="a positive integer (>= 1)",
                suggestion="LoRA rank 8/16/32 is typical; 16 is a good default.",
            )
        if self.lora_alpha <= 0:
            raise InvalidSettingError(
                setting_name="lora_alpha",
                value=self.lora_alpha,
                expected="a positive integer (>= 1)",
                suggestion="Pair lora_alpha with lora_r (e.g. r=16, alpha=32).",
            )
        if not (0.0 <= self.lora_dropout <= 1.0):
            raise InvalidSettingError(
                setting_name="lora_dropout",
                value=self.lora_dropout,
                expected="a float in [0.0, 1.0]",
                suggestion="LoRA dropout 0.0-0.1 is typical; default is 0.05.",
            )
        if self.max_seq_length <= 0:
            raise InvalidSettingError(
                setting_name="max_seq_length",
                value=self.max_seq_length,
                expected="a positive integer (>= 1)",
                suggestion=(
                    "Try max_seq_length=2048 for typical chat data; smaller "
                    "for tighter VRAM."
                ),
            )

        # Phase 1.1: Train on responses only
        self._train_on_responses = train_on_responses

        # F-014: explicit override for the markers Unsloth uses to mask user
        # tokens. None ⇒ auto-detect at train()-time so the tokenizer has
        # been loaded by then.
        self._response_markers_override = response_markers

        # B-002 / B-010: degradation knobs surfaced as instance attrs so the
        # multi-run trainer (and operator callers) can introspect.
        self.oom_recovery = oom_recovery
        self.unsloth_fallback = unsloth_fallback

        # v1.4 BRIDGE-A-002 follow-up (Wave 6a foundation): resolve the five
        # Wave 6b knobs from per-invocation kwarg (if not None) OR fall back
        # to the settings layer (env-var path). Same ``is not None`` pattern
        # as the load-bearing fields above so legitimate ``False`` /
        # ``"default"`` values are not mis-treated as "unset". The mapping
        # to settings paths is canonical:
        #   * ``use_dora`` → ``settings.lora.use_dora`` (default False)
        #   * ``packing`` → ``settings.data.packing`` (default True per
        #     v1.3 BACKEND-4)
        #   * ``init_lora_weights`` → ``settings.lora.init_lora_weights``
        #     (default "default")
        #   * ``optim`` → ``settings.training.optim`` (default
        #     "adamw_8bit")
        # ``lora_preset`` does NOT have a settings path today — there is no
        # ``settings.lora.preset`` field; the preset shape (rank /
        # target_modules / lr_mult) is applied via the
        # ``LORA_PRESETS["fast"|"quality"]`` overlay at the call site
        # (currently a future-wiring slot — see ``backpropagate/config.py``
        # ``LORA_PRESETS`` + ``get_lora_preset``). The kwarg is stored on
        # ``self.lora_preset`` so the CLI introspection filter passes it
        # through, preserving operator intent for the overlay-wiring slot
        # to consume in a follow-up. When None we default to "quality" to
        # match the v1.3 BACKEND-1 default contract documented in
        # ``LoRAConfig``.
        self.use_dora = use_dora if use_dora is not None else settings.lora.use_dora
        self.packing = packing if packing is not None else settings.data.packing
        self.init_lora_weights = (
            init_lora_weights
            if init_lora_weights is not None
            else settings.lora.init_lora_weights
        )
        self.lora_preset = lora_preset if lora_preset is not None else "quality"
        # ``optim="auto"`` is a CLI sentinel meaning "let the trainer pick"
        # (--optim accepts {auto, adamw_torch, paged_adamw_8bit, adamw_8bit}
        # with default "auto"). Treat it as equivalent to None so the
        # settings fallback fires and ``_detect_optim_for_card`` sees the
        # actual configured default ("adamw_8bit") for the consumer-card
        # paged-optim upgrade decision. Without this, "auto" would
        # passthrough verbatim and TRL/HF would reject it.
        if optim is not None and optim != "auto":
            self.optim = optim
        else:
            self.optim = settings.training.optim

        # F-005: store the operator's report_to intent; the resolver is
        # invoked lazily at train()-time so feature detection picks up
        # late-installed trackers (e.g. tracker installed after import).
        self._report_to_intent: str | list[str] | None = report_to

        # Internal state
        self._model: Any = None
        self._tokenizer: Any = None
        self._trainer: Any = None
        self._is_loaded = False
        self._training_runs: list[TrainingRun] = []

        # Wave 3.5 BACKEND-B-004: track whether train() actually completed at
        # least once. Pre-fix, an operator who called load_model() then save()
        # WITHOUT train() got a freshly-initialized LoRA adapter on disk —
        # rank-r Gaussian-noise weights per PEFT defaults (Biderman 2024 /
        # Thinking Machines 2025) — saved as if it were a trained checkpoint.
        # The save itself is technically valid (operators may want to verify
        # the base model loaded right), but the silence makes it indistinguishable
        # from a successfully-trained save. _has_trained gates a tripwire
        # warning in .save() that names the situation and points at .train()
        # as the typical next step. Set to True at the end of .train() after
        # the successful training run is appended to self._training_runs.
        self._has_trained: bool = False

        # Apply Windows fixes
        self._apply_windows_fixes()

        logger.info(f"Trainer initialized: {self.model_name}")
        logger.info(f"  LoRA: r={self.lora_r}, alpha={self.lora_alpha}")
        logger.info(f"  Batch: {self.batch_size}, LR: {self.learning_rate}")
        logger.info(
            f"  Degradation knobs: oom_recovery={self.oom_recovery}, "
            f"unsloth_fallback={self.unsloth_fallback}"
        )

        # v1.3 BACKEND-2: surface license caveats for known restrictively-
        # licensed presets at Trainer boot. ``lookup_model_preset_by_id``
        # matches by HF model_id (case-insensitive) so an operator who
        # passed the raw id (e.g. "Qwen/Qwen2.5-3B-Instruct") still gets
        # the caveat, not just operators who use ``get_model_preset``.
        # Caveat is logged at WARNING so it survives operators who
        # filter for >= WARN (CI / batch runners).
        try:
            from .config import lookup_model_preset_by_id

            _preset = lookup_model_preset_by_id(self.model_name)
            if _preset is not None and _preset.license_restriction:
                logger.warning(
                    "%s preset=%s license=%s",
                    _preset.license_restriction,
                    _preset.name,
                    _preset.license,
                )
        except Exception as _license_check_err:  # noqa: BLE001 — observability must not block boot  # nosec B110
            # License-caveat surface is opportunistic; never let a lookup
            # bug crash Trainer construction.
            logger.debug(
                f"license-restriction check skipped: {_license_check_err!r}"
            )

    def _resolve_report_to(self) -> str | list[str]:
        """F-005: resolve the user's report_to intent to a TRL-compatible value.

        Returns whatever TRL's :class:`SFTConfig` will accept on the
        ``report_to`` field:

        * ``"none"`` → no tracker (string accepted by SFTConfig).
        * ``["wandb"]`` / ``["tensorboard"]`` / etc — list of tracker names.

        Resolution rules (in order):

        1. ``self._report_to_intent`` is ``None`` or the string ``"none"`` →
           ``"none"`` (explicit opt-out).
        2. The intent is a list → return as-is (operator override; TRL
           will raise a clean error if a name is bogus).
        3. The intent is the string ``"auto"`` (the default) → detect
           installed trackers via feature_flags; return a list of the
           ones present, or ``"none"`` if none are.
        4. Otherwise the intent is a single tracker name as a string
           → wrap it in a list.
        """
        intent = self._report_to_intent
        if intent is None:
            return "none"
        if isinstance(intent, str):
            normalized = intent.strip().lower()
            if normalized in {"none", ""}:
                return "none"
            if normalized != "auto":
                # Single named tracker (e.g. "wandb").
                return [normalized]
            # Auto-resolve: pick up everything that's installed.
            trackers: list[str] = []
            if check_feature("wandb"):
                trackers.append("wandb")
            if check_feature("tensorboard"):
                trackers.append("tensorboard")
            if check_feature("mlflow"):
                trackers.append("mlflow")
            return trackers if trackers else "none"
        if isinstance(intent, list):
            # Operator-supplied list — pass through unchanged. Lower-case
            # names so trivial case mismatches don't make TRL choke.
            return [str(t).strip().lower() for t in intent if t]
        # Unknown type — fall back to "none" defensively.
        logger.warning(
            f"Trainer.report_to has unexpected type {type(intent).__name__}; "
            "falling back to 'none'."
        )
        return "none"

    def _apply_windows_fixes(self) -> None:
        """Apply Windows-specific environment variables."""
        if os.name == "nt":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            if settings.windows.xformers_disabled:
                os.environ["XFORMERS_DISABLED"] = "1"
            if settings.windows.cuda_launch_blocking:
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            logger.debug("Applied Windows-specific fixes")

    def _detect_batch_size(self) -> int:
        """Auto-detect optimal batch size based on available VRAM.

        Stage C amend BACKEND-B-009: log which branch fired so triage on
        slow training doesn't have to grep for "PyTorch not available" to
        disambiguate "I'm on a low-VRAM card" from "my CUDA query broke".
        Every return path now emits a single line naming the resolved
        tier + the reason for the fallback (if any).
        """
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 24:
                    logger.info(
                        f"_detect_batch_size: vram={vram_gb:.1f}GB tier=>=24 -> batch_size=4"
                    )
                    return 4
                elif vram_gb >= 16:
                    logger.info(
                        f"_detect_batch_size: vram={vram_gb:.1f}GB tier=16-24 -> batch_size=2"
                    )
                    return 2
                elif vram_gb >= 12:
                    logger.info(
                        f"_detect_batch_size: vram={vram_gb:.1f}GB tier=12-16 -> batch_size=1"
                    )
                    return 1
                else:
                    logger.info(
                        f"_detect_batch_size: vram={vram_gb:.1f}GB tier=<12 -> batch_size=1"
                    )
                    return 1
            else:
                logger.info(
                    "_detect_batch_size: fell back to default batch_size=2 "
                    "(reason=cuda_not_available — CPU/MPS run)"
                )
        except ImportError:
            logger.info(
                "_detect_batch_size: fell back to default batch_size=2 "
                "(reason=torch_not_installed)"
            )
        except RuntimeError as e:
            logger.info(
                f"_detect_batch_size: fell back to default batch_size=2 "
                f"(reason=cuda_query_failed: {e})"
            )
        except Exception as e:
            logger.warning(
                f"_detect_batch_size: fell back to default batch_size=2 "
                f"(reason=unexpected_error: {type(e).__name__}: {e})"
            )
        return 2  # Safe default

    # =========================================================================
    # v1.3 BACKEND-5 / BACKEND-7 — per-card optim + dtype resolution
    # =========================================================================
    # The pydantic-settings defaults ("adamw_8bit" for optim, bf16=True/
    # fp16=False) are conservative cross-card defaults. The runtime
    # resolvers below upgrade them when the card warrants it AND the
    # operator hasn't explicitly overridden them. The "explicit
    # override" detection compares the resolved settings value against
    # the documented default; any deviation means the operator either
    # set the env var or passed a CLI flag, and we leave their choice
    # alone.

    @staticmethod
    def _detect_optim_for_card(configured_optim: str) -> str:
        """v1.3 BACKEND-5: resolve the optimizer for the current GPU.

        Pre-fix: every card got ``adamw_8bit`` (the conservative
        default). Consumer cards (< 24GB VRAM) benefit materially from
        ``paged_adamw_8bit`` — the paged variant uses
        CPU<->GPU memory paging to reduce peak VRAM at a tiny
        wall-clock cost, which is the right tradeoff on a 16GB card.
        24GB+ cards have headroom and can stay on the non-paged variant.

        Detection rules (in order):

        1. If the operator passed anything other than the documented
           default ``adamw_8bit``, honor their choice unchanged. The
           explicit-override surface is the canonical knob for
           operators who pinned a specific optimizer for an LR
           schedule / fairness constraint / token budget that depends
           on it.
        2. If torch is missing or CUDA is unavailable, leave the
           default in place — no runtime data to act on.
        3. If the card has < 24GB VRAM, upgrade ``adamw_8bit`` →
           ``paged_adamw_8bit`` (consumer-card tier).
        4. Otherwise leave the default in place (datacenter-class card
           with VRAM to spare).

        Returns the resolved optimizer string. Pure function; the
        caller threads it into SFTConfig.optim.
        """
        # Rule 1: explicit operator override wins. The documented v1.3
        # default in config.py is "adamw_8bit"; anything else is the
        # operator's explicit pick.
        if configured_optim != "adamw_8bit":
            return configured_optim
        try:
            import torch
            if not torch.cuda.is_available():
                return configured_optim
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception as exc:  # noqa: BLE001
            # Defensive: never let a CUDA query failure crash the
            # optimizer resolver — fall back to the default.
            logger.debug(
                f"_detect_optim_for_card: CUDA query failed ({exc!r}); "
                f"leaving optim={configured_optim!r} unchanged."
            )
            return configured_optim
        if vram_gb < 24:
            logger.info(
                f"_detect_optim_for_card: vram={vram_gb:.1f}GB < 24 -> "
                f"upgrading optim adamw_8bit -> paged_adamw_8bit (reduces "
                f"peak VRAM via CPU<->GPU paging; override with --optim "
                f"adamw_torch / adamw_8bit / etc. to opt out)."
            )
            return "paged_adamw_8bit"
        logger.debug(
            f"_detect_optim_for_card: vram={vram_gb:.1f}GB >= 24 -> "
            f"keeping optim=adamw_8bit (datacenter tier; paged variant "
            f"would trade throughput for unneeded VRAM headroom)."
        )
        return configured_optim

    @staticmethod
    def _detect_optimal_dtype(configured_bf16: bool, configured_fp16: bool) -> tuple[bool, bool]:
        """v1.3 BACKEND-7: resolve (bf16, fp16) for the current GPU.

        bf16 has hardware support starting at Ampere (compute capability
        8.0) and is preferred over fp16 wherever it's available — same
        numerical range as fp32 (no LR-loss-scale dance) at fp16
        storage cost. Pre-fix the resolver hardcoded fp16=True for
        every non-Ampere card, including Ada (8.9) and Hopper (9.0)
        which both have bf16 hardware. The Ada miss matters because
        RTX 40-series cards (4090/4080/4070) are the most common
        consumer hardware for fine-tuning at v1.3 ship.

        Detection rules (in order):

        1. If the operator already set bf16=True OR fp16=True
           explicitly (config.py default is bf16=True, fp16=False),
           honor their choice. We treat the config default as
           "no explicit override" so the resolver can flip fp16 ->
           bf16 on Ada when the conservative default was carried.
           The operator's only way to *force* fp16 on Ada today is
           to pass ``--fp16`` AND ``--no-bf16`` (or set both env vars)
           — the resolver respects that by leaving the explicit choice
           alone (bf16=False, fp16=True ⇒ skip).
        2. If torch is missing or CUDA is unavailable, leave the
           defaults in place.
        3. If the card supports bf16 (compute capability >= 8.0, i.e.
           Ampere / Ada / Hopper / Blackwell): prefer (bf16=True,
           fp16=False).
        4. Otherwise prefer (bf16=False, fp16=True) — pre-Ampere
           cards.

        Returns ``(bf16, fp16)`` for SFTConfig.
        """
        # Rule 1: honor explicit fp16=True (operator forced fp16 over
        # the bf16 default — they know their card / their LR schedule).
        if configured_fp16:
            return (configured_bf16, configured_fp16)
        try:
            import torch
            if not torch.cuda.is_available():
                return (configured_bf16, configured_fp16)
            major, minor = torch.cuda.get_device_capability(0)
            capability = float(f"{major}.{minor}")
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                f"_detect_optimal_dtype: CUDA capability query failed "
                f"({exc!r}); leaving (bf16={configured_bf16}, "
                f"fp16={configured_fp16}) unchanged."
            )
            return (configured_bf16, configured_fp16)
        if capability >= 8.0:
            # Ampere / Ada / Hopper / Blackwell — bf16 hardware
            # available. Pre-Ada this means SM 8.0 (A100, 3090). Ada
            # is 8.9 (4090/4080/4070); Hopper is 9.0 (H100); Blackwell
            # is 12.x (B100, RTX 5080 / 5090).
            if not configured_bf16:
                logger.info(
                    f"_detect_optimal_dtype: capability={capability} "
                    f"(Ampere+) -> upgrading dtype to bf16 (was fp16). "
                    f"bf16 has fp32 numerical range without the loss-"
                    f"scale dance; opt out with --fp16."
                )
            return (True, False)
        # Pre-Ampere — bf16 hardware unavailable, must use fp16.
        if configured_bf16:
            logger.warning(
                f"_detect_optimal_dtype: capability={capability} "
                f"(pre-Ampere) does not support bf16; downgrading to "
                f"fp16. Operator-supplied bf16=True ignored on this card."
            )
        return (False, True)

    def _cleanup_vram(self) -> None:
        """
        Release unused GPU memory.

        Calls gc.collect() to free Python-side references, then
        torch.cuda.empty_cache() to return unreferenced GPU memory to the
        CUDA allocator. Safe to call even when no GPU is present.

        Intended for use between multi_run iterations or after training
        completes to reclaim VRAM before the next operation.
        """
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("VRAM cleanup: gc.collect() + torch.cuda.empty_cache()")
            else:
                logger.debug("VRAM cleanup: gc.collect() (no CUDA available)")
        except ImportError:
            logger.debug("VRAM cleanup: gc.collect() (torch not installed)")

    def load_model(self) -> None:
        """
        Load the model and tokenizer.

        B-010: when ``use_unsloth=True`` and ``unsloth_fallback=True`` (the
        defaults), an Unsloth load failure that is not a CUDA/network issue
        falls through to the plain transformers + PEFT path. The fallback
        produces a functionally equivalent (but slower) training setup so an
        Unsloth nightly that breaks loading no longer takes the entire
        pipeline down.

        Raises:
            ModelLoadError: If the model or tokenizer cannot be loaded
            GPUNotAvailableError: If CUDA is required but not available
        """
        if self._is_loaded:
            return

        logger.info(f"Loading model: {self.model_name}")

        try:
            if self.use_unsloth:
                try:
                    self._load_with_unsloth()
                except (ImportError, RuntimeError):
                    # Don't downgrade ImportError / RuntimeError — those are
                    # the "your env is wrong" / "CUDA is wrong" signals the
                    # surrounding except blocks rely on for accurate error
                    # routing.
                    raise
                except Exception as unsloth_err:
                    if not self.unsloth_fallback:
                        raise
                    # B-010: graceful degradation to plain transformers.
                    logger.warning(
                        f"Unsloth load failed ({type(unsloth_err).__name__}: "
                        f"{unsloth_err}); falling back to transformers + PEFT. "
                        "Set Trainer(unsloth_fallback=False) to disable."
                    )
                    self.use_unsloth = False
                    self._load_with_transformers()
            else:
                self._load_with_transformers()
        except ImportError as e:
            # F-019: ImportError = missing/incompatible upstream package.
            # We keep the explicit "pip install" suggestion (more specific
            # than the generic version-category hint) and pass
            # cause_category="version" so the cause tag still lands in
            # details for downstream log scraping.
            raise ModelLoadError(
                self.model_name,
                f"Missing required package: {e.name if hasattr(e, 'name') else str(e)}",
                suggestion="Install required packages: pip install unsloth transformers peft",
                cause_category="version",
            ) from e
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg or "gpu" in error_msg:
                raise GPUNotAvailableError(
                    suggestion="Ensure CUDA is installed and GPU is available"
                ) from e
            # F-019: classify the underlying failure so the per-category hint
            # table can fire. Dropping the explicit suggestion arg lets the
            # ModelLoadError ctor pick the right line from
            # exceptions._MODEL_LOAD_HINTS for auth / not_found / network /
            # version / unknown. RuntimeErrors that aren't CUDA-shaped are
            # rare here — usually a transformers internal — so most will
            # land in "unknown" with the generic line, which is correct.
            raise ModelLoadError(
                self.model_name,
                str(e),
                cause_category=_classify_model_load_cause(e),
            ) from e
        except Exception as e:
            # F-019: the catchall sees the largest population of
            # interesting cases — HfHubHTTPError 401/403/404, requests
            # connection errors, version-mismatch ImportErrors that slip
            # past the explicit ImportError branch via a deferred import.
            raise ModelLoadError(
                self.model_name,
                str(e),
                cause_category=_classify_model_load_cause(e),
            ) from e

        self._is_loaded = True
        logger.info("Model loaded successfully")

    def _load_with_unsloth(self) -> None:
        """Load model using Unsloth for 2x faster training.

        B-017: ``FastLanguageModel.from_pretrained`` is wrapped with the HF
        Hub transient-retry decorator so a 5xx / 429 / connection timeout
        from the Hub doesn't fail the load on the first blip.
        """
        from unsloth import FastLanguageModel

        try:
            self._model, self._tokenizer = _retry_hf_call(
                FastLanguageModel.from_pretrained,
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=True,
                trust_remote_code=settings.model.trust_remote_code,
                _label=f"unsloth_from_pretrained:{self.model_name}",
            )
        except Exception as e:
            # F-019: Unsloth's from_pretrained tunnels through huggingface_hub
            # for the actual weight download, so 401/403/404/connection
            # errors surface here too. Classify so the per-category hint
            # fires instead of the previous generic "check model and
            # network" line (which conflated auth and network failures).
            raise ModelLoadError(
                self.model_name,
                f"Unsloth model loading failed: {e}",
                cause_category=_classify_model_load_cause(e),
            ) from e

        # Apply LoRA. v1.3 BACKEND-3 / BACKEND-6: thread use_dora and
        # init_lora_weights through to the Unsloth call site. Built as a
        # kwargs dict so we can ``if-check`` each new field before
        # passing it — keeps the call backward-compatible with older
        # Unsloth releases whose get_peft_model signature doesn't accept
        # the v1.3 kwargs.
        lora_kwargs: dict[str, Any] = {
            "r": self.lora_r,
            "target_modules": settings.lora.target_modules,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": "none",
            "use_gradient_checkpointing": settings.lora.use_gradient_checkpointing,
            "random_state": settings.lora.random_state,
        }
        # v1.3 BACKEND-3: forward use_dora only when True so we don't
        # tickle the kwarg on an older Unsloth that may not accept it.
        # v1.4 BRIDGE-A-002 follow-up (Wave 6a): read from ``self.use_dora``
        # which the constructor resolved to either the operator's
        # per-invocation kwarg OR ``settings.lora.use_dora`` (env-var
        # default). Same shape for ``self.init_lora_weights``.
        if self.use_dora:
            lora_kwargs["use_dora"] = True
        # v1.3 BACKEND-6: forward init_lora_weights only when the
        # operator picked something other than the PEFT default. PEFT
        # accepts the string {"pissa", "loftq"} OR the bool True
        # (default-initialization). Map "default" -> True per the PEFT API.
        _init_w = self.init_lora_weights
        if _init_w and _init_w != "default":
            lora_kwargs["init_lora_weights"] = _init_w
        try:
            self._model = FastLanguageModel.get_peft_model(
                self._model,
                **lora_kwargs,
            )
        except Exception as e:
            # F-019: post-download LoRA application — usually a PEFT
            # version mismatch (peft renamed a kwarg) or an
            # unsupported-target-module config error. ImportError ⇒
            # "version", everything else ⇒ "unknown" via the classifier.
            raise ModelLoadError(
                self.model_name,
                f"Failed to apply LoRA: {e}",
                cause_category=_classify_model_load_cause(e),
            ) from e

    def _load_with_transformers(self) -> None:
        """Load model using standard transformers + PEFT.

        B-017: model + tokenizer ``from_pretrained`` calls are wrapped in
        the HF Hub transient-retry decorator.
        """
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load model
        self._model = _retry_hf_call(
            AutoModelForCausalLM.from_pretrained,
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=settings.model.trust_remote_code,
            _label=f"transformers_from_pretrained:{self.model_name}",
        )

        # Load tokenizer
        self._tokenizer = _retry_hf_call(
            AutoTokenizer.from_pretrained,
            self.model_name,
            trust_remote_code=settings.model.trust_remote_code,
            _label=f"tokenizer_from_pretrained:{self.model_name}",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Prepare for training
        self._model = prepare_model_for_kbit_training(self._model)

        # Apply LoRA. v1.3 BACKEND-3 / BACKEND-6: thread use_dora and
        # init_lora_weights through to PEFT. Built as a kwargs dict so
        # an older PEFT (pre-0.10 for DoRA, pre-0.7 for PiSSA/LoftQ)
        # that doesn't accept the field doesn't make us crash on this
        # call. The trainer logs a warning + carries on with vanilla
        # LoRA when the kwarg is rejected.
        lora_kwargs: dict[str, Any] = {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": settings.lora.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        # v1.4 BRIDGE-A-002 follow-up (Wave 6a): same instance-attr reads
        # as the Unsloth branch so the constructor's settings-fallback
        # resolution is honored uniformly across both load paths.
        if self.use_dora:
            lora_kwargs["use_dora"] = True
        _init_w = self.init_lora_weights
        if _init_w and _init_w != "default":
            # PEFT's init_lora_weights accepts {True, False, "gaussian",
            # "pissa", "loftq"}; the trainer's surface only exposes
            # "default"|"pissa"|"loftq" (str -> str passthrough).
            lora_kwargs["init_lora_weights"] = _init_w
        try:
            lora_config = LoraConfig(**lora_kwargs)
        except TypeError as exc:
            # Old PEFT rejected one of the v1.3 kwargs. Strip them and
            # retry with the legacy shape so the trainer doesn't hard-
            # fail on an Unsloth-pinned environment that ships an older
            # PEFT. WARN so the operator notices the silent downgrade.
            stripped: list[str] = []
            for k in ("use_dora", "init_lora_weights"):
                if k in lora_kwargs:
                    stripped.append(k)
                    lora_kwargs.pop(k)
            logger.warning(
                f"PEFT LoraConfig rejected v1.3 kwarg(s) {stripped!r} "
                f"({exc!r}); retrying with the legacy LoraConfig shape. "
                f"Upgrade PEFT >= 0.10 (DoRA) / >= 0.7 (PiSSA/LoftQ) to "
                f"enable these features."
            )
            lora_config = LoraConfig(**lora_kwargs)
        self._model = get_peft_model(self._model, lora_config)

    def train(
        self,
        dataset: str | Any = None,
        steps: int | None = None,
        samples: int | None = None,
        callback: TrainingCallback | None = None,
        resume_from: str | None = None,
    ) -> TrainingRun:
        """
        Train the model on a dataset.

        Args:
            dataset: Dataset path (JSONL, CSV) or HuggingFace dataset name
            steps: Number of training steps (overrides config)
            samples: Number of samples to use (overrides config)
            callback: Optional callback for training events
            resume_from: F-002 — when set, look up the run_id in the on-disk
                run history (scoped to ``self.output_dir``) and reuse its
                run_id + last checkpoint path. When ``None`` (default),
                start a fresh run. Multi-run resume lives on
                ``MultiRunTrainer``; the Trainer-level resume hook is a
                lighter "pick up the existing run_id and let the SFTTrainer
                resume from the latest HF checkpoint in output_dir" path.

                BACKEND-F-002 (v1.3): the lookup is output_dir-scoped (not
                a global run_id index). When the requested run_id is NOT
                found under the configured output_dir, this method raises
                ``InvalidSettingError`` with code
                ``INPUT_RESUME_NOT_FOUND``. Pre-F-002 the lookup miss fell
                through to a silent fresh-run under a NEW run_id —
                operators passing ``resume_from=<id-from-different-output>``
                lost their resume intent without an exception. Now the
                failure is loud and actionable: the error message names
                the requested run_id, the output_dir searched, and the
                operator's next steps (``backprop runs`` to list, or
                re-run with the correct ``--output``).

        Returns:
            TrainingRun with results

        Raises:
            InvalidSettingError: If steps or samples are invalid
            DatasetError: If dataset cannot be loaded
            TrainingError: If training fails
        """
        import time

        from trl import SFTTrainer

        # Validate inputs
        if steps is not None:
            if not isinstance(steps, int) or steps <= 0:
                raise InvalidSettingError(
                    "steps", steps, "positive integer",
                    suggestion="Use steps=100 or higher"
                )
        if samples is not None:
            if not isinstance(samples, int) or samples <= 0:
                raise InvalidSettingError(
                    "samples", samples, "positive integer",
                    suggestion="Use samples=1000 or higher"
                )

        # Load model if not loaded
        if not self._is_loaded:
            self.load_model()

        # Load dataset
        train_dataset = self._load_dataset(dataset, samples)

        # Pre-tokenize for Windows safety.
        #
        # Stage C BACKEND-B-009: log the OS-conditional decision once so a
        # post-mortem comparing a Windows train() run vs a Linux train()
        # run can confirm via one grep ("both took the pre_tokenize path"
        # or "both deferred"). The loss curve shape can differ between
        # paths because pre-tokenize chunks ahead of training while the
        # SFTTrainer-deferred path tokenizes on the fly; reproducibility
        # across OSes hinges on the operator knowing which one fired.
        # (run_id is minted below; this log fires before that point so we
        # don't thread it here — a follow-up log line at run_id mint time
        # carries the correlation token.)
        if os.name == "nt" and settings.windows.pre_tokenize:
            logger.info(
                "Pre-tokenization: applied (os.name=nt, windows.pre_tokenize=True) "
                "for dataset of %d samples",
                len(train_dataset),
            )
            train_dataset = self._pre_tokenize(train_dataset)
        else:
            logger.info(
                "Pre-tokenization: deferred to SFTTrainer "
                "(os.name=%s, windows.pre_tokenize=%s) for dataset of %d samples",
                os.name,
                settings.windows.pre_tokenize,
                len(train_dataset),
            )

        # F-005: resolve report_to once for this train() call. The auto-mode
        # picks up wandb / tensorboard / mlflow if their packages are installed
        # (which the [monitoring] extra installs by default), so a user who
        # ran ``pip install backpropagate[monitoring]`` AND ``wandb login``
        # gets W&B wiring for free without re-passing it on every call.
        # We pre-mint the run_id so the W&B run_name can correlate with our
        # internal correlation token (see B-001 below).
        report_to = self._resolve_report_to()
        # F-002: reuse the run_id when resuming so the on-disk history record
        # is updated in place rather than producing a duplicate row.
        # F-017: ALSO recover the checkpoint path so the inner SFTTrainer can
        # actually pick up where the prior run left off. Pre-F-017 the
        # resume path reused run_id + appended to history but never threaded
        # ``resume_from_checkpoint`` into ``self._trainer.train()`` — the
        # inner training started fresh from step 0, silently invalidating
        # the operator's "resume" intent.
        run_id_for_resume: str | None = None
        resume_checkpoint_path: str | None = None
        if resume_from:
            try:
                resume_manager = RunHistoryManager(str(self.output_dir))
                record = resume_manager.get_run(resume_from)
                if record is not None:
                    # Stage C amend BACKEND-B-008: hoist the
                    # checkpoint-path existence check BEFORE we mutate
                    # the prior run_history entry's status to "running".
                    # Pre-fix: the record was flipped to "running"
                    # downstream and THEN the SFTTrainer choked on a
                    # missing path, leaving the on-disk record in a
                    # broken in-progress state pointing at a checkpoint
                    # that no longer exists. The next resume auto-detect
                    # would latch onto this zombie entry forever.
                    record_cp = record.get("checkpoint_path")
                    if record_cp and not Path(str(record_cp)).exists():
                        logger.warning(
                            f"resume_from={resume_from!r}: prior "
                            f"checkpoint path {record_cp!r} no longer "
                            f"exists on disk. Marking the prior entry "
                            f"as failed (reason='resume_checkpoint_missing') "
                            f"and minting a FRESH run_id for this session "
                            f"rather than mutating the prior record. "
                            f"To recover the prior data, restore the "
                            f"checkpoint directory and re-run with "
                            f"resume_from={resume_from!r}."
                        )
                        try:
                            resume_manager.record_run_failed(
                                run_id=str(record.get("run_id")),
                                failure_reason="resume_checkpoint_missing",
                            )
                        except Exception as fail_err:
                            logger.warning(
                                f"Failed to mark stale resume entry as "
                                f"failed: {fail_err}"
                            )
                        # Fall through with run_id_for_resume=None — a
                        # fresh run_id will be minted below.
                    else:
                        run_id_for_resume = str(record.get("run_id"))
                        # The prior run's checkpoint_path is the directory we
                        # told HF to write to. HF's own resume detection scans
                        # that dir for ``checkpoint-<N>`` subfolders and picks
                        # the latest, so we hand it the directory (not a
                        # specific checkpoint-N path) — same convention TRL
                        # uses when you pass ``resume_from_checkpoint=True``.
                        if record_cp:
                            resume_checkpoint_path = str(record_cp)
                else:
                    # BACKEND-F-002: hard-error on lookup miss instead of
                    # silent fresh-start. The operator passed resume_from
                    # expecting resumption; falling back to a fresh run
                    # under a NEW run_id silently drops the resume intent
                    # and produces a model the operator did not ask for.
                    # The error message names the requested run_id, the
                    # output_dir actually searched, and the operator's
                    # next steps so the failure is actionable in the
                    # terminal without grep'ing logs.
                    raise InvalidSettingError(
                        setting_name="resume_from",
                        value=resume_from,
                        expected=(
                            f"a run_id present in the on-disk run history "
                            f"under output_dir={str(self.output_dir)!r}"
                        ),
                        suggestion=(
                            f"resume_from={resume_from!r} was NOT found in the "
                            f"run history at {str(self.output_dir)!r}. The "
                            f"resume lookup is scoped to the configured "
                            f"output_dir (not a global run_id index). Next "
                            f"steps:\n"
                            f"  1) Run `backprop runs` to list run_ids "
                            f"available under this output_dir.\n"
                            f"  2) If the run was trained under a different "
                            f"output_dir, re-run with `--output <that-dir>` "
                            f"so the lookup hits.\n"
                            f"  3) To start truly fresh under a NEW run_id, "
                            f"omit resume_from (or pass resume_from=None)."
                        ),
                    )
            except InvalidSettingError:
                # BACKEND-F-002: never swallow the hard resume-miss error.
                # The try/except below catches generic infra failures
                # (RunHistoryManager IO errors, malformed JSON, etc.) and
                # downgrades them to a WARN + fresh run. The F-002 strict
                # contract violation must propagate unchanged so the
                # operator sees the actionable hint.
                raise
            except Exception as exc:
                logger.warning(f"Resume lookup failed: {exc}")
        run_id = run_id_for_resume or uuid.uuid4().hex
        run_name = f"backprop-{run_id[:12]}" if report_to != "none" else None

        # v1.3 BACKEND-5 / BACKEND-7 + Wave 6a BACKEND-A-003: SFTConfig assembly
        # is delegated to the module-level :func:`_build_sft_config` helper so
        # the same defaults reach the MultiRunTrainer call site. The helper
        # owns the consumer-card paged-optim upgrade + Ada bf16/fp16 selection;
        # operator overrides on ``settings.training.*`` are honored.
        training_args = _build_sft_config(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation,
            max_steps=steps or settings.training.max_steps,
            learning_rate=self.learning_rate,
            warmup_steps=settings.training.warmup_steps,
            max_seq_length=self.max_seq_length,
            seed=settings.training.seed,
            lr_scheduler_type=settings.training.lr_scheduler_type,
            logging_steps=settings.training.logging_steps,
            save_steps=settings.training.save_steps,
            weight_decay=settings.training.weight_decay,
            report_to=report_to,  # F-005: dynamic — see _resolve_report_to.
            run_name=run_name,
            # v1.4 BRIDGE-A-002 follow-up (Wave 6a): thread the constructor-
            # resolved instance attributes (per-invocation kwarg OR settings
            # fallback) through to the helper. Pre-fix the helper read
            # ``settings.data.packing`` / ``settings.training.optim`` directly,
            # silently bypassing the per-invocation override path that the
            # CLI introspection filter now passes through.
            packing=self.packing,
            optim=self.optim,
        )

        # F-003: build the HF-TrainerCallback bridge ONCE for this train()
        # call. None when no user callback or when transformers is unavailable
        # (the second branch should never fire in practice — transformers is a
        # hard dep of TRL — but guard so a malformed env doesn't crash).
        _bridge_cb = (
            _build_trl_bridge_callback(callback) if callback is not None else None
        )
        sft_callbacks = [_bridge_cb] if _bridge_cb is not None else None

        # Create trainer (TRL 0.27+ uses processing_class instead of tokenizer)
        self._trainer = SFTTrainer(
            model=self._model,
            processing_class=self._tokenizer,
            train_dataset=train_dataset,
            args=training_args,
            callbacks=sft_callbacks,
        )

        # Apply train_on_responses_only if using Unsloth (Phase 1.1 optimization)
        # This focuses loss only on assistant responses, not user prompts.
        #
        # Wave 6a BACKEND-A-004: the application is delegated to the
        # module-level :func:`_apply_train_on_responses_only` helper so the
        # MultiRunTrainer call site picks up the same masking (pre-Wave-6a,
        # the multi-run path silently skipped this step despite the docstring
        # claim — loss leaked back onto the user prompt for multi-run users).
        #
        # F-014: track the resolved (instruction, response) marker pair so the
        # hyperparameters dict built further down can persist it into run
        # history (auditable post-mortem when a probe falls back to ChatML on
        # a non-ChatML tokenizer).
        self._trainer, resolved_response_markers = _apply_train_on_responses_only(
            self._trainer,
            self._tokenizer,
            enabled=self._train_on_responses,
            use_unsloth=self.use_unsloth,
            response_markers_override=self._response_markers_override,
        )

        # Train
        # B-001: ``run_id`` (the UUID4 correlation token) was minted above
        # so we could thread it into the SFTConfig.run_name for W&B (F-005).
        # The token is bound into the structured-logger context here,
        # embedded in TrainingRun.metadata, and unbound in the finally
        # block at the end of the method so it doesn't leak into the
        # caller's thread.
        legacy_run_label = f"run_{len(self._training_runs) + 1}"
        bind_run_context(run_id=run_id, session_kind="single_run")
        start_time = time.time()
        loss_history: list[float] = []
        status = "error"  # success path overwrites to "ok"
        run: TrainingRun | None = None

        # F-003: record the run start in the on-disk run history so
        # ``backprop list-runs`` / ``backprop show-run`` can surface it.
        run_history = RunHistoryManager(str(self.output_dir))
        dataset_info = dataset if isinstance(dataset, str) else type(dataset).__name__
        hyperparameters: dict[str, Any] = {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "max_seq_length": self.max_seq_length,
            "max_steps": steps or settings.training.max_steps,
            "max_samples": samples or settings.data.max_samples,
            "use_unsloth": self.use_unsloth,
            "seed": settings.training.seed,
        }
        # F-014: auditable per-run record of the chat markers actually used by
        # train_on_responses_only. Absent when the mode is disabled.
        if resolved_response_markers is not None:
            hyperparameters["response_markers"] = list(resolved_response_markers)
        try:
            if run_id_for_resume:
                # F-002 resume: flip status back to "running" without
                # clobbering the existing hyperparameters / dataset record.
                run_history.update_run(run_id, status="running")
            else:
                run_history.record_run_started(
                    run_id=run_id,
                    model_name=self.model_name,
                    dataset_info=dataset_info,
                    hyperparameters=hyperparameters,
                    session_kind="single_run",
                    checkpoint_path=str(self.output_dir / "lora"),
                    dataset_hash=_compute_dataset_hash(dataset),
                )
        except Exception as hist_err:
            # Never let history persistence kill a training session.
            logger.warning(f"RunHistoryManager.record_run_started failed: {hist_err}")

        logger.info(f"run_started run_id={run_id} legacy_label={legacy_run_label}")
        logger.info(f"  Steps: {steps or settings.training.max_steps}")
        logger.info(f"  Samples: {len(train_dataset)}")

        try:
            # B-002: OOM-recovery loop. We re-instantiate the SFTTrainer on
            # each retry so it picks up the halved batch / doubled accum.
            oom_consecutive_at_min = 0
            oom_retries = 0
            result: Any = None

            while True:
                try:
                    # Re-create SFTTrainer with current batch / accum on retry.
                    # (The first iteration uses the trainer built above.)
                    if oom_retries > 0:
                        # Wave 6a BACKEND-A-003: re-build via the shared helper
                        # so the OOM-retry path inherits the same paged/bf16
                        # upgrades as the first attempt. The detectors are
                        # pure functions of the configured value, so the
                        # resolution is stable across retries.
                        training_args = _build_sft_config(
                            output_dir=str(self.output_dir),
                            per_device_train_batch_size=self.batch_size,
                            gradient_accumulation_steps=self.gradient_accumulation,
                            max_steps=steps or settings.training.max_steps,
                            learning_rate=self.learning_rate,
                            warmup_steps=settings.training.warmup_steps,
                            max_seq_length=self.max_seq_length,
                            seed=settings.training.seed,
                            lr_scheduler_type=settings.training.lr_scheduler_type,
                            logging_steps=settings.training.logging_steps,
                            save_steps=settings.training.save_steps,
                            weight_decay=settings.training.weight_decay,
                            report_to=report_to,  # F-005: same resolution on retry.
                            run_name=run_name,
                            # v1.4 BRIDGE-A-002 follow-up (Wave 6a): same
                            # per-invocation threading as the first attempt
                            # so the OOM-retry path inherits the operator's
                            # ``packing`` / ``optim`` overrides instead of
                            # silently reverting to the settings layer.
                            packing=self.packing,
                            optim=self.optim,
                        )
                        # F-003: rebuild the bridge for each OOM retry so the
                        # adapter is bound to the fresh SFTTrainer instance.
                        # ``callback`` is captured from the enclosing train()
                        # call; if None, sft_callbacks stays None.
                        _bridge_cb_retry = (
                            _build_trl_bridge_callback(callback) if callback is not None else None
                        )
                        sft_callbacks_retry = (
                            [_bridge_cb_retry] if _bridge_cb_retry is not None else None
                        )
                        self._trainer = SFTTrainer(
                            model=self._model,
                            processing_class=self._tokenizer,
                            train_dataset=train_dataset,
                            args=training_args,
                            callbacks=sft_callbacks_retry,
                        )

                    # F-017: thread resume_from_checkpoint into the inner
                    # SFTTrainer.train so the operator's ``resume_from`` kwarg
                    # actually resumes step count + optimizer state + LR
                    # scheduler position. None ⇒ fresh start (TRL default).
                    # An explicit path that no longer exists on disk: WARN +
                    # fall back to fresh start so the resume kwarg never
                    # crashes a session.
                    _resume_arg: str | None = None
                    if resume_checkpoint_path:
                        if Path(resume_checkpoint_path).exists():
                            _resume_arg = resume_checkpoint_path
                            logger.info(
                                f"Resuming single-run training from {_resume_arg}"
                            )
                        else:
                            logger.warning(
                                f"resume_from checkpoint path {resume_checkpoint_path!r} "
                                "no longer exists on disk; starting fresh."
                            )
                    if _resume_arg is not None:
                        result = self._trainer.train(resume_from_checkpoint=_resume_arg)
                    else:
                        result = self._trainer.train()
                    break  # Success — exit retry loop

                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    # OOM detection (B-002): torch.cuda.OutOfMemoryError OR
                    # RuntimeError whose message contains "out of memory".
                    is_oom = False
                    try:
                        import torch as _torch

                        if isinstance(exc, _torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
                            is_oom = True
                    except (ImportError, AttributeError):
                        pass
                    if not is_oom and isinstance(exc, RuntimeError):
                        is_oom = "out of memory" in str(exc).lower()

                    # Stage C BACKEND-B-002: widen OOM detection to include
                    # CUDA library / driver errors that almost always indicate
                    # VRAM exhaustion (CUBLAS alloc fail, CUDNN init fail,
                    # NCCL post-OOM). When matched via the widened net, emit a
                    # structured WARN so operators see which signature fired —
                    # the matcher catalog lives in :data:`_OOM_ADJACENT_SUBSTRINGS`
                    # for a single-edit-site update when PyTorch wording shifts.
                    adjacent_marker: str | None = None
                    if not is_oom and isinstance(exc, RuntimeError):
                        msg_lower = str(exc).lower()
                        for marker in self._OOM_ADJACENT_SUBSTRINGS:
                            if marker in msg_lower:
                                is_oom = True
                                adjacent_marker = marker
                                logger.warning(
                                    f"oom-adjacent error intercepted by recovery "
                                    f"path: type={type(exc).__name__} "
                                    f"marker={marker!r} run_id={run_id} "
                                    f"code=RUNTIME_OOM_ADJACENT "
                                    f"message={str(exc)[:240]!r}"
                                )
                                break

                    if not self.oom_recovery and is_oom:
                        # Wave 6a RUNTIME_GPU_OOM Option A: wrap the OOM at
                        # the raise site so the documented contract holds
                        # end-to-end (README + handbook + cli.py exit-code
                        # mapper + llms.txt all promise this code; pre-Wave-6a
                        # the raise site re-raised the bare RuntimeError and
                        # the outer ``except RuntimeError`` handler around
                        # line 2440 wrapped it as
                        # ``TrainingError(code='RUNTIME_TRAINING_FAILED')``).
                        # Operators with oom_recovery=False now see a
                        # structured GPUMemoryError with the OOM as
                        # ``__cause__``. Distinct from
                        # RUNTIME_OOM_RECOVERY_EXHAUSTED (which fires on the
                        # path that DID retry and ran out of options).
                        from .exceptions import GPUMemoryError as _GPUMemErr

                        raise _GPUMemErr(
                            suggestion=(
                                "OOM hit with oom_recovery=False — the trainer "
                                "did not attempt batch_size halving. To survive "
                                "transient spikes automatically, reconstruct "
                                "the Trainer with oom_recovery=True (the "
                                "default). For a permanent fix, reduce "
                                "batch_size, enable gradient_checkpointing, "
                                "shorten max_seq_length, or apply 4-bit / 8-bit "
                                "quantization."
                            ),
                        ) from exc
                    if not is_oom:
                        raise

                    # OOM-specific recovery.
                    import torch as _torch

                    current_batch = self.batch_size
                    current_accum = self.gradient_accumulation
                    effective = current_batch * current_accum
                    # Stage C humanization: structured fields
                    # (event=oom_recovery_started, attempt=N, batch_size,
                    # grad_accum, effective_batch) so an operator grepping
                    # `event=oom_recovery_` can correlate the start /
                    # adjust / exhaust phases of one recovery episode.
                    # Recovery is GOOD news (the system handled it) so we
                    # name what's happening + what changes, not just that
                    # something broke.
                    logger.warning(
                        "event=oom_recovery_started run_id=%s attempt=%d "
                        "batch_size=%d grad_accum=%d effective_batch=%d "
                        "exc=%s",
                        run_id,
                        oom_retries + 1,
                        current_batch,
                        current_accum,
                        effective,
                        exc,
                    )

                    gc.collect()
                    try:
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                    except Exception:  # nosec B110 — best-effort CUDA cache reclaim; failures are non-fatal
                        pass

                    if current_batch > 1:
                        new_batch = max(1, current_batch // 2)
                        new_accum = max(1, current_accum * 2)
                        # Stage C humanization: name the recovery delta in
                        # operator-readable terms — the effective batch is
                        # PRESERVED, so the LR schedule and gradient
                        # statistics stay equivalent; only peak VRAM
                        # changes. Operators tuning batch_size deserve to
                        # see this invariant called out so they don't
                        # think the recovery secretly changed their
                        # effective hyperparameters.
                        logger.warning(
                            "event=oom_recovery_adjust run_id=%s attempt=%d "
                            "batch_size=%d->%d grad_accum=%d->%d "
                            "effective_batch=%d (preserved). "
                            "Retrying with halved batch + doubled accumulation; "
                            "no operator action required.",
                            run_id,
                            oom_retries + 1,
                            current_batch, new_batch,
                            current_accum, new_accum,
                            new_batch * new_accum,
                        )
                        self.batch_size = new_batch
                        self.gradient_accumulation = new_accum
                        oom_consecutive_at_min = 0
                        oom_retries += 1
                        continue

                    # Already at floor.
                    oom_consecutive_at_min += 1
                    oom_retries += 1
                    # Stage C humanization: keep the structured-event
                    # field shape consistent across the recovery family
                    # (started / adjust / floor / exhausted) so post-
                    # mortem grep correlates one episode end-to-end.
                    logger.error(
                        "event=oom_recovery_at_floor run_id=%s attempt=%d "
                        "consecutive_at_min_batch=%d/%d. "
                        "batch_size is already 1; we cannot halve further. "
                        "If %d more consecutive OOMs hit, recovery aborts "
                        "with RUNTIME_OOM_RECOVERY_EXHAUSTED.",
                        run_id,
                        oom_retries,
                        oom_consecutive_at_min,
                        self._OOM_MAX_RETRIES_AT_MIN_BATCH,
                        max(0, self._OOM_MAX_RETRIES_AT_MIN_BATCH - oom_consecutive_at_min),
                    )
                    if oom_consecutive_at_min >= self._OOM_MAX_RETRIES_AT_MIN_BATCH:
                        # Stage C BACKEND-B-005: pass code/details/cause via
                        # constructor kwargs (TrainingError already supports
                        # them) instead of assigning to instance attributes
                        # after the fact. The post-construction err.code= form
                        # circumvents the structured-error contract and is a
                        # template future contributors will copy from the most-
                        # debugged path in the codebase. Use the more specific
                        # RUNTIME_OOM_RECOVERY_EXHAUSTED code so triage can
                        # distinguish "the recovery loop ran and lost" from
                        # "a single one-shot OOM hit the wall".
                        raise TrainingError(
                            f"GPU error during training: persistent CUDA OOM "
                            f"at batch_size=1 ({oom_consecutive_at_min} "
                            f"consecutive attempts); cannot recover automatically.",
                            code="RUNTIME_OOM_RECOVERY_EXHAUSTED",
                            details={
                                "run_id": run_id,
                                "consecutive_oom_at_min_batch": oom_consecutive_at_min,
                                "oom_retries": oom_retries,
                                "adjacent_marker": adjacent_marker,
                            },
                            suggestion=(
                                "Use a smaller model (e.g. 7B -> 3B), reduce "
                                "max_seq_length, enable gradient_checkpointing, "
                                "or apply quantization. Set "
                                "Trainer(oom_recovery=False) to make OOMs "
                                "hard-fail on the first attempt."
                            ),
                            cause=exc,
                        ) from exc

                    # Below the threshold but at floor — retry once more
                    # with the same args (transient OOM may clear after
                    # empty_cache).
                    continue

            duration = time.time() - start_time

            # Validate training result
            if not hasattr(result, 'training_loss'):
                logger.warning("Training result missing 'training_loss' attribute - using 0.0")
            final_loss = getattr(result, 'training_loss', 0.0)

            # Extract loss history from logs
            loss_history = []
            if hasattr(self._trainer, 'state') and self._trainer.state.log_history:
                loss_history = [
                    log.get('loss', 0) for log in self._trainer.state.log_history
                    if 'loss' in log
                ]

            run = TrainingRun(
                run_id=run_id,
                steps=steps or settings.training.max_steps,
                final_loss=final_loss,
                loss_history=loss_history,
                duration_seconds=duration,
                samples_seen=len(train_dataset),
                output_path=str(self.output_dir),
                metadata={
                    "legacy_run_label": legacy_run_label,
                    "oom_retries": oom_retries,
                },
            )

            self._training_runs.append(run)

            # Wave 3.5 BACKEND-B-004: flip the tripwire flag now that an
            # actual SFTTrainer.train() invocation has completed and produced
            # a TrainingRun. Set here (after the run is appended) rather
            # than at train()-entry so a failed train() still trips the
            # save() warning — the failure may have left the adapter
            # weights in an indeterminate state, which is exactly the
            # situation operators benefit from being warned about.
            self._has_trained = True

            if callback and callback.on_complete:
                try:
                    callback.on_complete(run)
                except Exception as cb_error:
                    logger.warning(f"on_complete callback raised error: {cb_error}")

            logger.info(f"Training complete: loss={final_loss:.4f}, time={duration:.1f}s")
            status = "ok"

            # F-003: record successful completion in run history.
            try:
                run_history.record_run_completed(
                    run_id=run_id,
                    final_loss=final_loss,
                    loss_history=loss_history,
                    steps=run.steps,
                    duration_seconds=duration,
                    checkpoint_path=run.output_path,
                )
            except Exception as hist_err:
                logger.warning(
                    f"RunHistoryManager.record_run_completed failed: {hist_err}"
                )
            return run

        except KeyboardInterrupt:
            duration = time.time() - start_time
            # F-003: best-effort failure recording on interrupt.
            try:
                run_history.record_run_failed(
                    run_id=run_id,
                    failure_reason="KeyboardInterrupt: user interrupted training",
                    loss_history=loss_history,
                    duration_seconds=duration,
                )
            except Exception as hist_err:
                logger.warning(
                    f"RunHistoryManager.record_run_failed failed: {hist_err}"
                )
            raise TrainingAbortedError(
                reason="User interrupted training",
                steps_completed=getattr(self._trainer.state, 'global_step', 0) if self._trainer else 0,
            ) from None
        except BackpropagateError as exc:
            # F-003: record failure before re-raising.
            try:
                run_history.record_run_failed(
                    run_id=run_id,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    loss_history=loss_history,
                    duration_seconds=time.time() - start_time,
                )
            except Exception as hist_err:
                logger.warning(
                    f"RunHistoryManager.record_run_failed failed: {hist_err}"
                )
            # Already structured — propagate without re-wrapping (B-002 path).
            raise
        except RuntimeError as e:
            duration = time.time() - start_time
            try:
                run_history.record_run_failed(
                    run_id=run_id,
                    failure_reason=f"{type(e).__name__}: {e}",
                    loss_history=loss_history,
                    duration_seconds=duration,
                )
            except Exception as hist_err:
                logger.warning(
                    f"RunHistoryManager.record_run_failed failed: {hist_err}"
                )
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                # Stage C humanization: name the recovery levers in
                # priority order. Most operators hit this from "model
                # too big for VRAM" which is fixed by batch + gradient
                # checkpointing; mention quantization for the case where
                # those don't suffice.
                raise TrainingError(
                    f"GPU error during training: {e}",
                    suggestion=(
                        "Recovery options, cheapest first: "
                        "(1) reduce --batch-size (halve and re-run); "
                        "(2) enable gradient_checkpointing (saves "
                        "~30%% VRAM at ~20%% speed cost); "
                        "(3) lower max_seq_length if your dataset has "
                        "outlier-long samples; "
                        "(4) use a smaller model (7B -> 3B) or 4-bit "
                        "quantization. Set Trainer(oom_recovery=True, "
                        "default) to enable automatic batch-halving."
                    ),
                ) from e
            if callback and callback.on_error:
                callback.on_error(e)
            raise TrainingError(
                f"Training failed: {e}",
                suggestion=(
                    "Recovery checklist: "
                    "(1) run with --verbose for the full traceback so "
                    "you can see the exception below the wrapper; "
                    "(2) check package versions match the supported "
                    "matrix — `pip show trl transformers peft accelerate`; "
                    "(3) for tokenization errors, inspect the first 5 "
                    "rows of your dataset for malformed JSON / missing "
                    "fields; "
                    "(4) for OOM / CUDA errors, see the dedicated "
                    "RUNTIME_GPU_* codes via "
                    "`backprop info --error-codes`."
                ),
            ) from e
        except Exception as e:
            duration = time.time() - start_time
            try:
                run_history.record_run_failed(
                    run_id=run_id,
                    failure_reason=f"{type(e).__name__}: {e}",
                    loss_history=loss_history,
                    duration_seconds=duration,
                )
            except Exception as hist_err:
                logger.warning(
                    f"RunHistoryManager.record_run_failed failed: {hist_err}"
                )
            if callback and callback.on_error:
                callback.on_error(e)
            # Stage C humanization: never raise a bare TrainingError
            # without a suggestion — at the outer-except catch-all an
            # operator hits the WORST error message in the codebase (no
            # context, no remediation). Even a generic "run with
            # --verbose" is better than nothing.
            raise TrainingError(
                f"Training failed: {e}",
                suggestion=(
                    "An unexpected exception escaped the training loop. "
                    "Run with --verbose to see the full traceback "
                    "(the wrapped exception is the one with the real "
                    "diagnostic). If the failure is reproducible, file "
                    "a bug at "
                    "https://github.com/mcp-tool-shop-org/backpropagate/issues "
                    "with the traceback + your training command."
                ),
            ) from e
        finally:
            # B-001: emit run_ended with status and release the context-var
            # binding so the thread doesn't carry our run_id into the next
            # caller. We deliberately do this in `finally` so the log line
            # fires on both happy and error paths.
            logger.info(f"run_ended run_id={run_id} status={status}")
            unbind_run_context("run_id", "session_kind")

    def _load_dataset(
        self,
        dataset: str | Any,
        samples: int | None = None,
    ) -> Any:
        """
        Load dataset from various sources.

        Accepts:
            - None: uses default dataset from config (HuggingFace)
            - str with file extension (.jsonl, .json, .csv, .parquet, .txt, .md):
              routes through DatasetLoader for format detection, validation, and
              ChatML conversion, then returns a HuggingFace Dataset
            - str without file extension: treated as HuggingFace dataset name
            - datasets.Dataset: used directly
            - DatasetLoader: calls .to_hf_dataset() directly

        Raises:
            DatasetNotFoundError: If dataset file doesn't exist
            DatasetParseError: If dataset cannot be parsed
            DatasetError: For other dataset-related errors
        """
        from datasets import Dataset, load_dataset

        max_samples = samples or settings.data.max_samples

        # File extensions that DatasetLoader handles
        _LOCAL_FILE_EXTENSIONS = (
            '.jsonl', '.json', '.csv', '.parquet', '.txt', '.md',
        )

        try:
            if dataset is None:
                # Use default dataset from config (B-017: retry on transient HF Hub failures)
                ds = _retry_hf_call(
                    load_dataset,
                    settings.data.dataset_name,
                    split=settings.data.dataset_split,
                    _label=f"load_dataset:{settings.data.dataset_name}",
                )
            elif isinstance(dataset, DatasetLoader):
                # DatasetLoader passed directly — use its validated output
                validation = dataset.validation_result
                if validation.warnings:
                    for w in validation.warnings[:5]:
                        logger.warning(f"Dataset validation warning: {w}")
                if not validation.is_valid:
                    logger.warning(
                        f"Dataset has {validation.error_count} validation errors "
                        f"({validation.error_rate:.1%} error rate) — proceeding anyway"
                    )
                    for err in validation.errors[:5]:
                        logger.warning(f"  {err}")
                ds = dataset.to_hf_dataset()
            elif isinstance(dataset, str):
                if any(dataset.lower().endswith(ext) for ext in _LOCAL_FILE_EXTENSIONS):
                    # Local file — route through DatasetLoader for format detection,
                    # validation, and ChatML conversion
                    try:
                        loader = DatasetLoader(dataset, validate=True)
                    except FileNotFoundError as e:
                        raise DatasetNotFoundError(
                            dataset,
                            suggestion="Create the file or use a HuggingFace dataset name"
                        ) from e

                    # Log format detection
                    logger.info(f"DatasetLoader detected format: {loader.detected_format.value}")

                    # Log validation warnings/errors
                    validation = loader.validation_result
                    if validation.warnings:
                        for w in validation.warnings[:5]:
                            logger.warning(f"Dataset validation warning: {w}")
                    if not validation.is_valid:
                        logger.warning(
                            f"Dataset has {validation.error_count} validation errors "
                            f"({validation.error_rate:.1%} error rate) — proceeding anyway"
                        )
                        for err in validation.errors[:5]:
                            logger.warning(f"  {err}")

                    ds = loader.to_hf_dataset()
                else:
                    # No file extension — assume HuggingFace dataset name
                    # B-017: retry on transient HF Hub failures (5xx, 429, timeouts).
                    try:
                        ds = _retry_hf_call(
                            load_dataset,
                            dataset,
                            split=settings.data.dataset_split,
                            _label=f"load_dataset:{dataset}",
                        )
                    except Exception as e:
                        raise DatasetError(
                            f"Failed to load HuggingFace dataset '{dataset}': {e}",
                            suggestion="Check dataset name and network connection"
                        ) from e
            elif isinstance(dataset, Dataset):
                ds = dataset
            else:
                raise DatasetError(
                    f"Unsupported dataset type: {type(dataset).__name__}",
                    suggestion="Use a file path (JSONL, CSV, Parquet), HuggingFace dataset name, Dataset object, or DatasetLoader"
                )
        except (DatasetNotFoundError, DatasetParseError, DatasetError):
            raise
        except FileNotFoundError as e:
            raise DatasetNotFoundError(str(e)) from e
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {e}") from e

        # Limit samples
        if max_samples > 0 and len(ds) > max_samples:
            if settings.data.shuffle:
                ds = ds.shuffle(seed=settings.training.seed)
            ds = ds.select(range(max_samples))

        logger.info(f"Loaded {len(ds)} samples")
        return ds

    def _pre_tokenize(self, dataset: Any) -> Any:
        """Pre-tokenize dataset for Windows safety."""
        logger.info("Pre-tokenizing dataset (Windows-safe mode)")

        def tokenize_fn(examples: dict[str, Any]) -> Any:
            try:
                return self._tokenizer(
                    examples[settings.data.text_column],
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=False,
                )
            except KeyError:
                raise DatasetError(
                    f"Dataset missing required column '{settings.data.text_column}'",
                    suggestion=f"Available columns: {list(examples.keys()) if hasattr(examples, 'keys') else 'unknown'}"
                )

        try:
            tokenized = dataset.map(
                tokenize_fn,
                batched=True,
                num_proc=None,  # None = run in main process (avoids Windows multiprocessing issues)
                remove_columns=dataset.column_names,
                desc="Tokenizing",
            )
        except DatasetError:
            raise
        except Exception as e:
            raise DatasetError(f"Tokenization failed: {e}") from e

        return tokenized

    def save(
        self,
        path: str | None = None,
        save_merged: bool = False,
        run_id: str | None = None,
        register_in_manifest: bool = True,
    ) -> str:
        """
        Save the trained model atomically.

        B-006: writes flow into ``<path>.partial`` first and ``shutil.move()``
        promotes the directory to the final path on success. The partial
        directory is cleaned up on any failure path so the operator never
        sees a half-written PEFT directory (config.json present, weights
        missing — which raises a cryptic 'state_dict missing keys' on the
        next resume attempt).

        BACKEND-F-007 (Wave 6a): on success the saved checkpoint is also
        registered with a :class:`CheckpointManager` rooted at
        ``self.output_dir`` so a later ``MultiRunTrainer`` pointed at the
        same ``checkpoint_dir=<output_dir>`` can discover this checkpoint
        via :meth:`CheckpointManager.find_latest_for_run_id` and resume
        from it. Pre-F-007 the single-run save path wrote the PEFT
        directory + the ``run_id`` correlation file but never appeared in
        any manifest — an operator who trained single-run, then tried to
        continue in multi-run mode, hit the silent-fresh-start failure
        mode at the multi-run resume site because the manifest scan
        returned no matching ``run_id``. Cross-trainer interop is now a
        documented invariant: any checkpoint registered by a Trainer
        can be resumed by a MultiRunTrainer pointing at the same
        ``checkpoint_dir`` (specifically, the Trainer's ``output_dir``).
        Operators using a separate ``checkpoint_dir`` for multi-run
        (e.g. the default ``output_dir/multi_run``) should construct
        the resuming MultiRunTrainer with
        ``MultiRunConfig(checkpoint_dir=<single-run-trainer.output_dir>)``.

        Args:
            path: Output path (default: output_dir/lora)
            save_merged: Whether to save merged weights (larger but standalone)
            run_id: Optional correlation token (B-001) persisted into a
                ``run_id`` file inside the saved checkpoint dir so operators
                can grep by the same identifier across logs + manifests.
            register_in_manifest: BACKEND-F-007 (Wave 6a) — when True
                (default), the saved checkpoint is registered with a
                :class:`CheckpointManager` rooted at ``self.output_dir``
                so a later MultiRunTrainer pointed at the same
                ``checkpoint_dir`` can discover and resume from it. Pass
                False for ad-hoc / export-only saves where manifest
                registration would pollute the resume-candidate set
                (e.g. an operator saving a merged-weights snapshot for
                upload that should never be considered as a resume
                target). Registration failures are swallowed (logged
                at WARN); the PEFT save itself is the load-bearing
                contract.

        Returns:
            Path to saved model

        Raises:
            TrainingError: If no model loaded or save fails
            CheckpointError: If atomic promotion fails
        """
        import shutil

        from .exceptions import CheckpointError

        if not self._is_loaded:
            # Stage C humanization: name the typical operator workflow
            # that recovers from this state. Pre-fix the message was
            # correct but terse; an operator looking at the traceback for
            # the first time may not know which method to call when.
            raise TrainingError(
                "Cannot save: no model is loaded in this Trainer instance. "
                "Call trainer.load_model() to load the configured base model, "
                "OR call trainer.train(dataset=...) which loads-then-trains "
                "in one step. If you intended to save the merged weights of "
                "a previously-trained adapter, instantiate a fresh Trainer "
                "with the same model name and call load_model() before save().",
                code="RUNTIME_TRAINING_FAILED",
            )

        # Wave 3.5 BACKEND-B-004: tripwire warning for the load-then-save-
        # without-train workflow. Pre-fix, an operator who did
        # ``Trainer(...).load_model(); trainer.save("./out")`` got a
        # freshly-initialized LoRA adapter on disk — rank-r Gaussian-noise
        # weights from PEFT init defaults — saved as if it were a trained
        # checkpoint. The save itself is technically valid (the operator
        # may legitimately be verifying the base model loaded right), but
        # the silence makes the untrained save indistinguishable from a
        # successfully-trained save. We warn instead of blocking because
        # the "save the base model to verify it loaded right" workflow is
        # a real operator pattern; the warning is a tripwire that names
        # the situation and points at .train() as the typical next step.
        if not self._has_trained:
            logger.warning(
                "Saving a loaded but untrained model — Trainer.train() was "
                "never called. The on-disk adapter weights are init values, "
                "not learned weights. If this is intentional (e.g. verifying "
                "base-model load), ignore this warning. If you meant to train "
                "first, call trainer.train(dataset=...) before save()."
            )

        output_path = Path(path or self.output_dir / "lora")
        partial_path = output_path.with_name(output_path.name + ".partial")

        # Ensure parent exists; the atomic promote at the end places the
        # leaf directory.
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise CheckpointError(
                "save", str(output_path),
                f"Permission denied creating parent directory: {e}"
            ) from e
        except OSError as e:
            raise CheckpointError(
                "save", str(output_path),
                f"Failed to create parent directory: {e}"
            ) from e

        # Wipe any leftover .partial from a previous crash.
        if partial_path.exists():
            shutil.rmtree(partial_path, ignore_errors=True)

        try:
            partial_path.mkdir(parents=True, exist_ok=False)
        except OSError as e:
            raise CheckpointError(
                "save", str(partial_path),
                f"Failed to create partial directory: {e}"
            ) from e

        # Stage C BACKEND-B-001: humanize the silent merged→LoRA-only
        # downgrade. Pre-fix, an operator who explicitly asked for merged
        # weights (`save_merged=True`) while training without Unsloth got a
        # LoRA adapter on disk with no diagnosis — they'd discover the
        # mismatch at inference / deployment time. We log a structured
        # warning that names the WHY (transformers Trainer can't do an
        # in-place merge) and points at TWO concrete migration paths:
        # re-train with Unsloth (the default), or call `backprop export`
        # with `--format merged` after this save. Tone per the same shape
        # as the Wave 3.5 BACKEND-B-004 untrained-save warning:
        # actionable + terse + names the next step.
        if save_merged and not self.use_unsloth:
            logger.warning(
                "save_merged=True was requested but use_unsloth=False — "
                "the underlying transformers Trainer does not support an "
                "in-place merge. Saving the LoRA adapter instead. To get "
                "merged weights, either re-train with use_unsloth=True "
                "(the default), or run `backprop export %s --format merged` "
                "after this save.",
                output_path,
            )

        try:
            if save_merged and self.use_unsloth:
                self._model.save_pretrained_merged(
                    str(partial_path),
                    self._tokenizer,
                    save_method="merged_16bit",
                )
            else:
                self._model.save_pretrained(str(partial_path))
                self._tokenizer.save_pretrained(str(partial_path))

            # B-001: drop run_id into the checkpoint dir so an operator can
            # match the on-disk artefact back to a log session without
            # cross-referencing wall-clock timestamps.
            if run_id is not None:
                try:
                    (partial_path / "run_id").write_text(run_id, encoding="utf-8")
                except OSError as e:
                    logger.warning(f"Failed to write run_id file: {e}")

            # Atomic promote.
            if output_path.exists():
                shutil.rmtree(output_path)
            shutil.move(str(partial_path), str(output_path))
        except CheckpointError:
            raise
        except Exception as e:
            raise CheckpointError(
                "save", str(output_path),
                str(e)
            ) from e
        finally:
            # Belt-and-braces: leave no .partial behind on any path.
            if partial_path.exists():
                shutil.rmtree(partial_path, ignore_errors=True)

        logger.info(f"Model saved to: {output_path}")

        # BACKEND-F-007 (Wave 6a): register the saved checkpoint in a
        # CheckpointManager rooted at self.output_dir so a later
        # MultiRunTrainer pointed at the same checkpoint_dir can
        # discover this checkpoint via find_latest_for_run_id(run_id)
        # and resume from it. The registration writes a single
        # manifest.json entry next to the run_history.json that
        # Trainer.train() already maintains; both files share the same
        # output_dir so cross-trainer interop is a documented
        # invariant. Failures are swallowed (logged at WARN) — the
        # PEFT save is the load-bearing contract; the manifest entry
        # is the cross-trainer convenience that lights up multi-run
        # resume from a single-run checkpoint.
        if register_in_manifest:
            try:
                from .checkpoints import CheckpointManager, CheckpointPolicy

                # auto_prune=False so a single-run save into a manifest
                # never silently deletes prior multi-run entries — the
                # operator's pruning policy belongs to the MultiRunTrainer
                # that owns the multi-run lifecycle, not to ad-hoc
                # single-run saves.
                cm_policy = CheckpointPolicy(auto_prune=False)
                cm = CheckpointManager(
                    checkpoint_dir=str(self.output_dir),
                    policy=cm_policy,
                )
                # run_index=0 is the convention for single-run-saved
                # checkpoints (multi-run uses 1-based per-run indices).
                # is_run_boundary=True so retention policies that honor
                # boundary checkpoints preserve this even if a later
                # multi-run session prunes around it.
                cm.register(
                    run_index=0,
                    checkpoint_path=str(output_path),
                    training_loss=None,
                    validation_loss=None,
                    is_run_boundary=True,
                    run_id=run_id,
                )
                logger.debug(
                    f"Registered single-run checkpoint in manifest at "
                    f"{self.output_dir} (run_id={run_id!r}); a later "
                    f"MultiRunTrainer(checkpoint_dir={str(self.output_dir)!r}) "
                    f"can resume from this checkpoint."
                )
            except Exception as reg_err:
                logger.warning(
                    f"BACKEND-F-007: failed to register checkpoint in "
                    f"manifest at {self.output_dir}: {reg_err}. The PEFT "
                    f"directory at {output_path} is intact; only the "
                    f"cross-trainer resume index is missing. To re-build "
                    f"it manually, instantiate CheckpointManager("
                    f"checkpoint_dir={str(self.output_dir)!r}) and call "
                    f"register(run_index=0, checkpoint_path="
                    f"{str(output_path)!r}, run_id={run_id!r})."
                )

        return str(output_path)

    def export(
        self,
        format: str = "lora",
        output_dir: str | None = None,
        quantization: str = "q4_k_m",
        push_to_hub: bool = False,
        repo_id: str | None = None,
        **kwargs: Any,
    ) -> ExportResult:
        """
        Export the trained model.

        Args:
            format: Export format - "lora", "merged", or "gguf"
            output_dir: Output directory (default: self.output_dir/format)
            quantization: GGUF quantization type (q4_k_m, q5_k_m, q8_0, f16)
            push_to_hub: Whether to push to HuggingFace Hub (merged only)
            repo_id: Repository ID for Hub (required if push_to_hub=True)
            **kwargs: Additional arguments passed to export functions

        Returns:
            ExportResult with path, size, and timing info

        Example:
            >>> trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
            >>> trainer.train("data.jsonl", steps=100)
            >>> result = trainer.export("gguf", quantization="q4_k_m")
            >>> print(result.summary())
        """
        from .export import (
            export_gguf,
            export_lora,
            export_merged,
        )

        if not self._is_loaded:
            # Stage C humanization: same pattern as save() — name the
            # recovery workflow. [[grep-all-instances]] applied across
            # the 3 sites in trainer.py that share this guard.
            raise TrainingError(
                "Cannot export: no model is loaded in this Trainer instance. "
                "Call trainer.load_model() first to load the configured base "
                "model + any saved adapter; then trainer.export(format=...) "
                "writes the on-disk artifact. To export from an existing "
                "checkpoint without re-training, instantiate Trainer with "
                "the same model name + load_model() and the saved adapter "
                "is picked up automatically.",
                code="RUNTIME_EXPORT_FAILED",
            )

        output_path = Path(output_dir or self.output_dir / format)

        format_lower = format.lower()

        if format_lower == "lora":
            result = export_lora(
                model=self._model,
                output_dir=output_path,
                **kwargs,
            )
            logger.info(f"Exported LoRA adapter: {result.path}")

        elif format_lower == "merged":
            result = export_merged(
                model=self._model,
                tokenizer=self._tokenizer,
                output_dir=output_path,
                push_to_hub=push_to_hub,
                repo_id=repo_id,
                **kwargs,
            )
            logger.info(f"Exported merged model: {result.path}")

        elif format_lower == "gguf":
            result = export_gguf(
                model=self._model,
                tokenizer=self._tokenizer,
                output_dir=output_path,
                quantization=quantization,
                **kwargs,
            )
            logger.info(f"Exported GGUF ({quantization}): {result.path}")

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'lora', 'merged', or 'gguf'")

        return result

    def push_to_hub(self, repo_id: str, private: bool = True) -> None:
        """Push model to HuggingFace Hub."""
        if not self._is_loaded:
            # Stage C humanization: same pattern as save() / export() —
            # name the recovery workflow. [[grep-all-instances]] —
            # all 3 in-trainer.py "No model loaded" sites updated
            # together.
            raise TrainingError(
                "Cannot push to Hub: no model is loaded in this Trainer "
                "instance. Call trainer.load_model() to load the base "
                "model + any saved adapter, OR run trainer.train() to "
                "produce the in-memory adapter, before push_to_hub(). "
                "For a Hub-only push without re-training, use "
                f"`backprop export <adapter-dir> --format merged "
                f"--push-to-hub --repo-id {repo_id}` from the CLI.",
                code="RUNTIME_TRAINING_FAILED",
            )

        self._model.push_to_hub(repo_id, private=private)
        self._tokenizer.push_to_hub(repo_id, private=private)
        logger.info(f"Pushed to HuggingFace Hub: {repo_id}")

    @property
    def model(self) -> Any:
        """Access the underlying model."""
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Access the tokenizer."""
        return self._tokenizer

    @property
    def runs(self) -> list[TrainingRun]:
        """Get all training runs."""
        return self._training_runs

    def multi_run(
        self,
        dataset: str | Any = None,
        num_runs: int = 5,
        steps_per_run: int = 100,
        samples_per_run: int = 1000,
        merge_mode: str = "slao",
        checkpoint_dir: str | None = None,
        on_run_complete: Callable[..., Any] | None = None,
        resume_from: str | None = None,
    ) -> MultiRunResult:
        """
        Execute SLAO Multi-Run training (multiple short runs with LoRA merging).

        This is the recommended approach for fine-tuning as it:
        - Prevents catastrophic forgetting via SLAO merging
        - Exposes the model to diverse data across runs
        - Saves checkpoints after each run for rollback
        - Monitors GPU safety throughout

        Args:
            dataset: Dataset name/path or HuggingFace dataset
            num_runs: Number of training runs (default: 5)
            steps_per_run: Steps per run (default: 100)
            samples_per_run: Fresh samples per run (default: 1000)
            merge_mode: "slao" (recommended) or "simple"
            checkpoint_dir: Where to save checkpoints
            on_run_complete: Callback after each run

        Returns:
            MultiRunResult with aggregate metrics and run history

        Example:
            >>> trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
            >>> result = trainer.multi_run(
            ...     dataset="HuggingFaceH4/ultrachat_200k",
            ...     num_runs=5,
            ...     steps_per_run=100,
            ... )
            >>> print(f"Final loss: {result.final_loss}")
        """
        from .multi_run import MergeMode, MultiRunConfig, MultiRunTrainer

        # Pre-flight GPU check
        if not check_gpu_safe():
            raise GPUNotAvailableError("GPU safety check failed. Check temperature and VRAM.")

        config = MultiRunConfig(
            num_runs=num_runs,
            steps_per_run=steps_per_run,
            samples_per_run=samples_per_run,
            merge_mode=MergeMode(merge_mode.lower()),
            checkpoint_dir=checkpoint_dir or str(self.output_dir / "multi_run"),
            initial_lr=self.learning_rate,
        )

        multi_run_trainer = MultiRunTrainer(
            model=self.model_name,
            config=config,
            on_run_complete=on_run_complete,
            resume_from=resume_from,
        )

        return multi_run_trainer.run(dataset)

    # Backwards compatibility alias
    speedrun = multi_run


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_model(
    model_name: str | None = None,
    load_in_4bit: bool = True,
    max_seq_length: int = 2048,
) -> tuple[Any, Any]:
    """
    Load a model and tokenizer.

    Args:
        model_name: Model name/path
        load_in_4bit: Use 4-bit quantization
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (model, tokenizer)
    """
    trainer = Trainer(model=model_name, max_seq_length=max_seq_length)
    trainer.load_model()
    return trainer.model, trainer.tokenizer


def load_dataset(
    dataset: str | Any,
    max_samples: int | None = None,
    split: str | None = None,
) -> Any:
    """
    Load a dataset.

    Args:
        dataset: Dataset path or name
        max_samples: Maximum samples to load
        split: Dataset split

    Returns:
        HuggingFace Dataset
    """
    from datasets import load_dataset as hf_load_dataset

    if isinstance(dataset, str):
        if dataset.endswith('.jsonl') or dataset.endswith('.json'):
            ds = hf_load_dataset('json', data_files=dataset, split='train')
        elif dataset.endswith('.csv'):
            ds = hf_load_dataset('csv', data_files=dataset, split='train')
        else:
            ds = hf_load_dataset(dataset, split=split or "train")
    else:
        ds = dataset

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    return ds


# Import MultiRunTrainer for re-export (lazy to avoid circular imports)
def __getattr__(name: str) -> Any:
    if name == "MultiRunTrainer":
        from .multi_run import MultiRunTrainer
        return MultiRunTrainer
    if name == "SpeedrunTrainer":  # Backwards compatibility
        from .multi_run import SpeedrunTrainer
        return SpeedrunTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
