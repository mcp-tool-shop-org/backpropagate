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
    FullFinetuneModelTooLargeError,
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
from .mlx_backend import detect_apple_silicon, resolve_backend

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
    # v1.4 BACKEND-F-002 (Wave 6b features): VRAM pre-flight estimator
    "VRAMEstimate",
    "estimate_vram",
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
# v1.4 BACKEND-F-008 (Wave 6b features) — FULL FINE-TUNING MODE
# =============================================================================
# mode='full' supports models up to the 3B parameter ceiling on consumer
# 16GB cards (per the v1.3 16GB fine-tuning study-swarm: Biderman 2024 /
# Thinking Machines 2025). The trainer's mode='full' contract:
#
#   * `_build_sft_config(mode='full', ...)` returns an SFTConfig with no
#     peft_config (full FT), paged_adamw_8bit optimizer (memory ceiling),
#     gradient_checkpointing=True, and a 10x-lower learning rate by default.
#   * Trainer / MultiRunTrainer construction-time gate refuses models > 3B
#     with RUNTIME_FULL_FT_MODEL_TOO_LARGE.
#   * The gate probes the parameter count via (1) the preset table when a
#     known HF model_id is matched, then (2) the loaded model when available.
#     For unknown / unloaded models we accept construction and the gate
#     re-fires at load_model() time. Construction-time refusal is the
#     primary contract; load_model-time is the belt-and-braces second check.
#
# The mode parameter is a string Literal so SFTConfig assembly stays in one
# helper (`_build_sft_config(mode='lora'|'full', ...)`); a parallel
# `_build_full_ft_sft_config` would fork the code path and violate the
# v1.4 Wave 6b doctrine ("extend, don't fork").

# Documented production-feasible parameter count ceiling for mode='full'
# on a 16GB consumer GPU. Operators who genuinely have a 24GB+ card can
# still hit the gate (the threshold is conservative) — the escape hatch
# is to construct with mode='lora' OR pick a smaller preset. Future
# v1.5 may expose a `--full-ft-ceiling-billions` flag for 24GB+ operators.
_FULL_FT_PARAM_CEILING_BILLIONS: float = 3.0

# Default learning rate for mode='full'. Full fine-tuning literature
# (Biderman 2024 / Thinking Machines 2025) recommends ~10x lower LR than
# LoRA — typical LoRA default is 2e-4, so full FT default is 2e-5. The
# operator can override via Trainer(learning_rate=...).
_FULL_FT_DEFAULT_LR_DIVISOR: float = 10.0

# v1.5 T1.2 (ORPO): default learning rate for method='orpo'. ORPO's combined
# SFT + odds-ratio preference loss is sensitive to LR — the reference-free
# preference literature (Hong 2024 "ORPO: Monolithic Preference Optimization
# without Reference Model", arXiv:2403.07691) and TRL's own ORPO example
# scripts anchor on the 8e-6 / 5e-6 band, an order of magnitude below the
# 2e-4 LoRA SFT default. We lower the default to 8e-6 so a bare
# Trainer(method='orpo') lands in the documented-stable range without
# operator intervention; explicit ``learning_rate=`` wins. Mirrors the
# mode='full' divisor path but is a fixed anchor (ORPO's stable band does
# not scale with the LoRA default). full+orpo is blocked, so the two
# LR-default branches are mutually exclusive.
_ORPO_DEFAULT_LR: float = 8e-6


def _estimate_param_count_billions(model_id: str) -> float | None:
    """v1.4 BACKEND-F-008: estimate model parameter count for the mode='full' gate.

    Resolution order:
      1. Match against the preset table; preset entries with a "Bn"-style
         token in their name (e.g. "qwen2.5-7b", "phi-4-mini-3.8b") yield a
         parsed numeric value. The preset name is the canonical signal.
      2. Heuristic regex over the model_id string for "<N>B" / "<N>b"
         substrings (e.g. "Qwen/Qwen2.5-7B-Instruct-bnb-4bit" -> 7.0).
         Best-effort; the substring need not be a preset.
      3. Return None when no signal is available. The caller's gate then
         degrades to accept-construction-and-recheck-at-load.

    Returns the estimated parameter count in billions, or None.
    """
    import re

    if not model_id:
        return None

    # 1. Preset-table primary path. Preset names follow ``<family>-<N>b``
    #    convention (case-insensitive). lookup_model_preset_by_id matches
    #    HF model_id case-insensitively, so an operator who passed the raw
    #    HF id still hits the preset entry.
    try:
        from .config import MODEL_PRESETS, lookup_model_preset_by_id

        preset = lookup_model_preset_by_id(model_id)
        if preset is None:
            # Operator may have passed the preset name directly (e.g.
            # "qwen2.5-7b") rather than the HF model_id. Walk the table.
            for name, candidate in MODEL_PRESETS.items():
                if name.lower() == model_id.strip().lower():
                    preset = candidate
                    break
        if preset is not None:
            # Parse "Nb" from the preset name. The catalog convention is
            # ``<family>-<count>b`` so split on "-" and look for a numeric
            # suffix on each segment. "phi-4-mini-3.8b" → 3.8; "qwen2.5-7b"
            # → 7.0; "smollm3-3b" → 3.0; "llama-3.2-1b" → 1.0.
            for segment in reversed(preset.name.split("-")):
                segment_clean = segment.strip().lower().rstrip("b")
                try:
                    parsed = float(segment_clean)
                    # Sanity bound: model sizes range 0.1B to 200B+.
                    if 0.05 <= parsed <= 1000.0:
                        return parsed
                except ValueError:
                    continue
    except Exception as exc:  # noqa: BLE001 — preset lookup is best-effort
        logger.debug(f"_estimate_param_count_billions: preset path raised {exc!r}")

    # 2. Heuristic regex fallback. Matches "<N>B" or "<N>.<M>B" anywhere
    #    in the model_id; case-insensitive. Captures the LARGEST match
    #    (an id like "Qwen2.5-7B-Instruct-bnb-4bit" should yield 7.0,
    #    not the "4bit" trailing fragment which the regex also matches
    #    against the "4" but lacks the "B" suffix — the [Bb] anchor
    #    prevents that collision).
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb](?:[^a-zA-Z]|$)")
    matches = pattern.findall(model_id)
    if matches:
        try:
            # Pick the largest match — handles "phi-4-mini-3.8b" by
            # rejecting "4" (a Phi-version literal, not a param count)
            # in favor of "3.8" (the actual param count). Operators
            # whose model id has both a series number and a param
            # count benefit; pure-name models like "Mistral-7B" still
            # work since the only match is the right one.
            best = max(float(m) for m in matches)
            # Sanity bound: a model id mentioning "0.5B" is plausible
            # (e.g. Qwen 0.5B), but a "100B" would be an outlier.
            if 0.05 <= best <= 1000.0:
                return best
        except (TypeError, ValueError):
            pass

    return None


def _enforce_full_ft_param_ceiling(
    model_id: str,
    *,
    ceiling_billions: float = _FULL_FT_PARAM_CEILING_BILLIONS,
    loaded_model: Any = None,
) -> None:
    """v1.4 BACKEND-F-008: refuse mode='full' for models > 3B.

    Probe order:
      1. ``loaded_model.num_parameters()`` when a loaded model is supplied
         (authoritative). Falls back to summing ``parameters()`` lengths
         when ``num_parameters`` is unavailable on the model object.
      2. :func:`_estimate_param_count_billions` over the model id string.
      3. If neither produces a count, accept construction silently (the
         load-time recheck is the safety net).

    Raises :class:`FullFinetuneModelTooLargeError` when the count exceeds
    the ceiling. Pure function side-effect-free otherwise.
    """
    estimated_billions: float | None = None

    # 1. Authoritative: ask the loaded model directly.
    if loaded_model is not None:
        try:
            if hasattr(loaded_model, "num_parameters"):
                # transformers / PEFT models expose this. Returns raw count.
                count = int(loaded_model.num_parameters())
                estimated_billions = count / 1e9
            else:
                # Manual sum — works for any nn.Module-like object.
                total = 0
                for param in loaded_model.parameters():
                    try:
                        total += int(getattr(param, "numel", lambda: 0)())
                    except Exception:  # nosec B112 — best-effort per-param probe; one bad param shouldn't kill the count
                        continue
                if total > 0:
                    estimated_billions = total / 1e9
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                f"_enforce_full_ft_param_ceiling: loaded model probe raised "
                f"{exc!r}; falling back to model_id heuristics."
            )

    # 2. Fallback: probe the model_id string.
    if estimated_billions is None:
        estimated_billions = _estimate_param_count_billions(model_id)

    # 3. No signal — accept (the load-time recheck will catch oversized
    # models when they are actually loaded).
    if estimated_billions is None:
        logger.info(
            f"_enforce_full_ft_param_ceiling: could not estimate parameter "
            f"count for {model_id!r}; deferring the mode='full' check to "
            f"load_model() time when the model is actually instantiated."
        )
        return

    if estimated_billions > ceiling_billions:
        raise FullFinetuneModelTooLargeError(
            model_name=model_id,
            param_count_billions=estimated_billions,
            ceiling_billions=ceiling_billions,
        )

    logger.info(
        f"_enforce_full_ft_param_ceiling: mode='full' approved — "
        f"model={model_id!r} estimated_params={estimated_billions:.2f}B "
        f"<= ceiling={ceiling_billions:.0f}B."
    )


# =============================================================================
# v1.4 BACKEND-F-002 (Wave 6b features) — VRAM PRE-FLIGHT ESTIMATOR
# =============================================================================
# Today operators learn a config is too big AT FIRST OOM. The estimator
# returns a structured VRAM ceiling estimate before .train() fires so an
# operator can ask "will this config OOM?" at construction time.
#
# The math (matches industry-standard back-of-envelope; see
# Biderman 2024 / Hugging Face peft docs):
#
#   total_vram_gb =
#       model_weights_gb (params * bytes_per_param) +
#       lora_adapter_gb  (rank * (in_dim + out_dim) * num_layers * bytes_per_param * 2)  [LoRA only]
#       optimizer_state_gb (params * bytes_per_optim_state) +
#       activations_gb (batch * seq_len * hidden_dim * num_layers * bytes_per_param * 2) +
#       kv_cache_gb (batch * seq_len * num_kv_heads * head_dim * 2 layers worth) +
#       overhead_gb (15% margin for fragmentation + framework overhead)


@dataclass
class VRAMEstimate:
    """v1.4 BACKEND-F-002: structured VRAM estimate for a training config.

    Each field is a GB estimate for one VRAM consumer. ``total_gb`` is the
    headline number operators compare against their card's VRAM. The
    breakdown helps post-mortem ("which component blew the budget?").
    """

    total_gb: float
    model_weights_gb: float
    lora_adapter_gb: float
    optimizer_state_gb: float
    activations_gb: float
    kv_cache_gb: float
    overhead_gb: float
    # Reproducibility inputs — same shape on the way out as the way in.
    param_count_billions: float
    mode: str
    batch_size: int
    gradient_accumulation: int
    max_seq_length: int
    lora_r: int
    notes: list[str] = field(default_factory=list)

    def fits_on_card(self, vram_gb: float) -> bool:
        """Will this config fit on a card with ``vram_gb`` total VRAM?"""
        return self.total_gb <= vram_gb

    def summary(self) -> str:
        """Operator-readable one-line summary of the estimate."""
        return (
            f"VRAM estimate ({self.mode}, {self.param_count_billions:.1f}B "
            f"params, batch={self.batch_size}, seq={self.max_seq_length}): "
            f"total={self.total_gb:.1f}GB "
            f"(weights={self.model_weights_gb:.1f} + "
            f"lora={self.lora_adapter_gb:.1f} + "
            f"optim={self.optimizer_state_gb:.1f} + "
            f"activations={self.activations_gb:.1f} + "
            f"kv={self.kv_cache_gb:.1f} + "
            f"overhead={self.overhead_gb:.1f})"
        )


def estimate_vram(
    model: str,
    *,
    mode: str = "lora",
    lora_r: int = 16,
    lora_alpha: int | None = None,  # noqa: ARG001 — accepted for symmetry, alpha doesn't affect memory
    batch_size: int = 1,
    gradient_accumulation: int = 1,  # noqa: ARG001 — accepted for symmetry, accum doesn't affect peak VRAM
    max_seq_length: int = 2048,
    bytes_per_param: int = 2,  # bf16 / fp16 default; 4 for fp32, 1 for int8, 0.5 for nf4
    quantize_base: bool = True,  # nf4 base + bf16 adapter (the trainer default)
    hidden_dim: int = 4096,  # 7B-class default; operator can override
    num_layers: int = 32,  # 7B-class default; operator can override
    num_heads: int = 32,  # 7B-class default
    overhead_fraction: float = 0.15,
    param_count_billions: float | None = None,
) -> VRAMEstimate:
    """v1.4 BACKEND-F-002: pre-flight VRAM estimator.

    Returns a structured estimate before ``.train()`` so an operator can
    ask "will this config OOM?" instead of finding out at first OOM. The
    math is back-of-envelope (15% overhead margin); accuracy is within
    ~10-20% of empirical peak for well-known training configs.

    Args:
        model: Model identifier — preset name or HF id. Used to estimate
            parameter count via :func:`_estimate_param_count_billions`.
        mode: ``"lora"`` (default) or ``"full"``. Full FT skips the
            ``lora_adapter_gb`` line + uses higher optimizer state.
        lora_r: LoRA rank (default 16). Ignored when mode='full'.
        lora_alpha: Accepted for API symmetry; does not affect memory.
        batch_size: Per-device batch size.
        gradient_accumulation: Accepted; does not affect peak VRAM.
        max_seq_length: Maximum input sequence length.
        bytes_per_param: 2 for bf16/fp16 (default), 4 for fp32, 1 for
            int8, 0.5 for nf4.
        quantize_base: When True (default — matches the trainer's
            ``load_in_4bit=True``), the base model is in nf4 (0.5 bytes
            per param) and the LoRA adapter is in bf16 (2 bytes per
            param). When False, the base model uses ``bytes_per_param``.
        hidden_dim: Model hidden dim (default 4096 — 7B-class).
        num_layers: Model num_layers (default 32 — 7B-class).
        num_heads: Model num_heads (default 32 — 7B-class).
        overhead_fraction: Fragmentation + framework overhead (default 15%).
        param_count_billions: Optional explicit parameter count. When
            None, estimated via :func:`_estimate_param_count_billions`.

    Returns:
        :class:`VRAMEstimate` carrying the headline number + breakdown.
    """
    notes: list[str] = []

    if param_count_billions is None:
        param_count_billions = _estimate_param_count_billions(model)
    if param_count_billions is None:
        # Defensive default: 7B is the v1.3 canonical 16GB target. Surface
        # the assumption in notes so operators see the imputation.
        param_count_billions = 7.0
        notes.append(
            f"param_count_billions not provided and could not be estimated "
            f"from model={model!r}; assumed 7.0B (v1.3 canonical 16GB target)."
        )

    params = param_count_billions * 1e9
    bytes_to_gb = 1.0 / (1024 ** 3)

    # 1. Model weights. nf4 base when quantize_base=True (the trainer
    #    default with load_in_4bit=True); otherwise use bytes_per_param.
    if quantize_base:
        # nf4: 0.5 bytes per param. The LoRA adapter (if mode='lora')
        # still lives in bf16 — that's the lora_adapter_gb line.
        model_weights_gb = (params * 0.5) * bytes_to_gb
        notes.append("base model quantized to nf4 (0.5 bytes/param)")
    else:
        model_weights_gb = (params * bytes_per_param) * bytes_to_gb

    # 2. LoRA adapter. Per-layer cost = rank * (in_dim + out_dim) * 2 (A + B).
    #    Modern PEFT applies LoRA to ~7 modules per layer (q, k, v, o, gate,
    #    up, down for Llama/Qwen-style architectures). Approximate with a
    #    7-module-per-layer constant.
    if mode == "lora":
        lora_modules_per_layer = 7
        lora_adapter_gb = (
            lora_r
            * (hidden_dim + hidden_dim)  # in + out (typically same)
            * num_layers
            * lora_modules_per_layer
            * bytes_per_param  # adapters in bf16/fp16 even when base is nf4
        ) * bytes_to_gb
    else:
        lora_adapter_gb = 0.0

    # 3. Optimizer state. paged_adamw_8bit (the trainer default on consumer
    #    cards) stores 2 momentum buffers per trainable param at 1 byte each.
    #    Full FT trains the whole model; LoRA only trains the adapter (rank
    #    * (in + out) * num_layers * 7 modules).
    trainable_params: float
    if mode == "lora":
        trainable_params = lora_r * (hidden_dim + hidden_dim) * num_layers * 7
    else:
        trainable_params = params
    # paged 8-bit Adam: 2 buffers * 1 byte + gradient (bytes_per_param)
    optimizer_state_gb = (
        trainable_params * (2 * 1 + bytes_per_param)
    ) * bytes_to_gb

    # 4. Activations. With gradient checkpointing the activation memory
    #    scales as sqrt(num_layers) instead of linearly. Mode='full'
    #    enables gradient_checkpointing=True by default; mode='lora'
    #    inherits the setting from settings.lora.use_gradient_checkpointing.
    activation_layer_factor = (
        max(1.0, num_layers ** 0.5) if mode == "full"
        else float(num_layers)
    )
    activations_gb = (
        batch_size
        * max_seq_length
        * hidden_dim
        * activation_layer_factor
        * bytes_per_param
        * 2  # forward + backward
    ) * bytes_to_gb
    if mode == "full":
        notes.append(
            "mode='full' assumes gradient_checkpointing=True (sqrt(L) "
            "activation memory)"
        )

    # 5. KV cache. batch * seq_len * num_heads * head_dim * num_layers * 2 (k+v)
    #    bytes_per_param-sized. Training rarely keeps the full KV cache (it's
    #    primarily an inference cost) but transformers libraries allocate it
    #    transiently during forward; the constant approximates that share.
    head_dim = hidden_dim // max(1, num_heads)
    kv_cache_gb = (
        batch_size
        * max_seq_length
        * num_heads
        * head_dim
        * num_layers
        * 2  # k + v
        * bytes_per_param
        * 0.25  # Training amortization factor — full cache not retained
    ) * bytes_to_gb

    subtotal = (
        model_weights_gb
        + lora_adapter_gb
        + optimizer_state_gb
        + activations_gb
        + kv_cache_gb
    )
    overhead_gb = subtotal * overhead_fraction
    total_gb = subtotal + overhead_gb

    return VRAMEstimate(
        total_gb=total_gb,
        model_weights_gb=model_weights_gb,
        lora_adapter_gb=lora_adapter_gb,
        optimizer_state_gb=optimizer_state_gb,
        activations_gb=activations_gb,
        kv_cache_gb=kv_cache_gb,
        overhead_gb=overhead_gb,
        param_count_billions=param_count_billions,
        mode=mode,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        notes=notes,
    )


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
    # v1.4 BACKEND-F-008 (Wave 6b features): training mode. ``"lora"`` (the
    # default) preserves pre-Wave-6b behavior byte-identically — operators
    # who don't pass mode='full' see no change. ``"full"`` enables full
    # fine-tuning: gradient_checkpointing=True (sqrt(L) activation memory),
    # paged_adamw_8bit optimizer (memory ceiling), and a 10x-lower learning
    # rate by default. The mode parameter is INTERNAL to the helper — the
    # SFTConfig assembly stays in ONE place; mode='full' is an EXTENSION,
    # not a parallel implementation (per advisor Wave 6b lock Q2). When
    # the helper detects an irreconcilable contract (e.g. an SFTConfig
    # parameter that doesn't map cleanly to mode-dispatch), escalate to
    # Wave 6a.5 hotfix rather than fork the helper.
    mode: str = "lora",
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

    # v1.4 BACKEND-F-008 (Wave 6b features): mode dispatch. The helper stays
    # SINGLE; mode='full' adjusts optimizer + LR + gradient_checkpointing
    # in-place rather than forking the SFTConfig assembly. Operators on
    # mode='lora' (the default) see byte-identical pre-Wave-6b behavior.
    if mode not in {"lora", "full"}:
        raise ValueError(
            f"_build_sft_config: mode={mode!r} not recognized. "
            f"Expected 'lora' (default) or 'full'."
        )

    # v1.4 BRIDGE-A-002 follow-up (Wave 6a): honor the per-invocation
    # ``optim`` override when supplied (forwarded by Trainer.train() /
    # MultiRunTrainer._execute_run via the Trainer instance's ``self.optim``,
    # which the constructor pre-resolved from operator kwarg OR
    # ``settings.training.optim``). When ``optim`` is None we read
    # ``settings.training.optim`` directly (pre-Wave-6a behavior preserved
    # for callers that don't thread the per-invocation override).
    _configured_optim = optim if optim is not None else settings.training.optim
    # v1.4 BACKEND-F-008 (Wave 6b): mode='full' forces paged_adamw_8bit
    # because full FT of a 3B model on a 16GB consumer card cannot afford
    # the non-paged AdamW state. Operators who explicitly overrode to a
    # specific optimizer (anything other than the documented adamw_8bit
    # default) keep their choice — the detector at _detect_optim_for_card
    # honors explicit overrides. Mode='full' with optim='auto' / default
    # surfaces paged_adamw_8bit; mode='full' with optim='adamw_torch'
    # respects the operator's pin (they know their card / their LR
    # schedule; if it OOMs that's on them).
    if mode == "full" and _configured_optim == "adamw_8bit":
        # Force the consumer-tier paged variant for the full FT memory
        # ceiling. The detector below would upgrade adamw_8bit to
        # paged_adamw_8bit on <24GB cards already; for mode='full' we
        # short-circuit that path so the upgrade fires even on 24GB cards
        # (which the LoRA detector would have left alone).
        _configured_optim = "paged_adamw_8bit"
    resolved_optim = Trainer._detect_optim_for_card(_configured_optim)
    resolved_bf16, resolved_fp16 = Trainer._detect_optimal_dtype(
        settings.training.bf16, settings.training.fp16
    )

    # v1.4 BACKEND-F-008 (Wave 6b): mode='full' scales the learning rate
    # down by ~10x by default per the Biderman 2024 / Thinking Machines
    # 2025 full-FT-vs-LoRA quality math. The operator's explicit
    # `learning_rate=` kwarg is the authority — we only adjust when the
    # caller passed the v1.3 LoRA default. The decision happens in
    # Trainer.__init__ (mode='full' construction-time default-rewrite); the
    # helper below applies the value as-given by the caller. Documenting
    # the contract here so the LR field stays a one-edit-site invariant.

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
        # Note: `overwrite_output_dir` was removed in current trl (was a
        # transformers.TrainingArguments inherited kwarg pre-trl 0.20+).
        # SFTConfig in current trl raises TypeError if passed; the v1.3
        # nightly-train-smoke caught this against the real trl install.
        # The default behavior is preserved (output_dir overwrites on
        # collision); we just drop the explicit kwarg.
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

    # v1.4 BACKEND-F-008 (Wave 6b): full FT mandates gradient checkpointing.
    # SFTConfig inherits from TrainingArguments which accepts the
    # ``gradient_checkpointing`` kwarg. Forcing it for mode='full' is the
    # documented contract — full FT of a 3B model on a 16GB card requires
    # sqrt(L) activation memory; without checkpointing the activations line
    # alone would exceed 8GB on seq=2048. Operators who genuinely have
    # 24GB+ and want to skip checkpointing should pass mode='lora' with
    # explicit full-rank LoRA settings; mode='full' is the consumer-card
    # contract.
    if mode == "full":
        kwargs["gradient_checkpointing"] = True
        # gradient_checkpointing_kwargs is supported by recent transformers
        # (>=4.36). Pass use_reentrant=False which is the recommended
        # default for HF gradient checkpointing — avoids the silent bug
        # where reentrant=True breaks DDP gradients on some models.
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return SFTConfig(**kwargs)


# =============================================================================
# SHARED ORPOCONFIG BUILDER (v1.5 T1.2 — ORPO Wave 2)
# =============================================================================
# Mirror of :func:`_build_sft_config` for the reference-free ORPO objective.
# It REUSES the same two GPU-dependent detectors —
# :meth:`Trainer._detect_optim_for_card` and
# :meth:`Trainer._detect_optimal_dtype` — so card detection (consumer-card
# paged-optim upgrade + Ada/Hopper/Blackwell bf16 selection + the Stage-A
# CPU-runner downgrades) CANNOT diverge between the SFT and ORPO configs.
# Because those detectors downgrade bnb-8bit → adamw_torch on CPU and force
# fp32 on CPU, this config is CPU-constructible for free (the unit tests build
# it under ``torch.cuda.is_available() == False``).
#
# Contract differences vs SFTConfig:
#   * NO ``packing`` — ORPOConfig has no such field (it rejects unknown
#     kwargs); preference rows are paired, not packed.
#   * NO ``gradient_checkpointing`` forcing — ORPO is mode='lora'-only in v1.5
#     (full+orpo is blocked at construction), so there is no full-FT
#     activation-memory contract to enforce here; ORPOConfig's own default
#     governs.
#   * ``beta`` (the odds-ratio loss weight) and ``max_length`` (mapped from
#     the operator's ``max_seq_length``) are ORPO-specific.
#   * ``max_prompt_length`` / ``max_completion_length`` are optional ORPO
#     truncation knobs — added only when not None so ORPOConfig's defaults
#     govern otherwise.


def _build_orpo_config(
    output_dir: str,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    learning_rate: float,
    warmup_steps: int,
    max_seq_length: int,
    *,
    orpo_beta: float,
    seed: int,
    lr_scheduler_type: str,
    logging_steps: int,
    save_steps: int | None = None,
    weight_decay: float | None = None,
    max_prompt_length: int | None = None,
    max_completion_length: int | None = None,
    report_to: Any = None,
    run_name: str | None = None,
    optim: str | None = None,
) -> Any:
    """Assemble an ``ORPOConfig`` with the v1.3 quality contracts applied.

    Like :func:`_build_sft_config`, this resolves ``optim`` and ``bf16`` /
    ``fp16`` via the SAME detectors so the ORPO config cannot drift from the
    SFT config on card-specific decisions:

    * ``optim`` — via :meth:`Trainer._detect_optim_for_card` (consumer-card
      paged-optim upgrade; CPU → ``adamw_torch`` downgrade — the Stage-A
      CPU-runner fix, inherited here so ORPO is CPU-constructible).
    * ``bf16`` / ``fp16`` — via :meth:`Trainer._detect_optimal_dtype`
      (Ampere+ → bf16; CPU → fp32 so ORPOConfig does not reject bf16=True
      at construction).

    Parameters mirror :func:`_build_sft_config` with three ORPO-specific
    additions (``orpo_beta`` → ``beta``, plus the optional
    ``max_prompt_length`` / ``max_completion_length`` truncation knobs).
    ``max_seq_length`` maps to ORPOConfig's ``max_length``.

    Returns:
        A configured ``ORPOConfig`` instance.
    """
    from trl import ORPOConfig

    # Reuse the exact detectors the SFT builder uses so the two configs
    # converge on the same card-specific resolution (single source of truth).
    # Note: ORPO is mode='lora'-only in v1.5, so there is no mode='full'
    # adamw_8bit → paged short-circuit here — the consumer-card upgrade in
    # _detect_optim_for_card still fires for the adamw_8bit default on
    # <24GB cards, and the CPU downgrade fires when CUDA is absent.
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
        "dataloader_num_workers": 0 if os.name == "nt" else 4,
        "report_to": report_to,
        "run_name": run_name,
        # ORPO-specific: max_length is ORPOConfig's name for the combined
        # prompt+completion sequence cap; beta is the odds-ratio loss weight.
        "max_length": max_seq_length,
        "beta": orpo_beta,
    }
    if save_steps is not None:
        kwargs["save_steps"] = save_steps
    if weight_decay is not None:
        kwargs["weight_decay"] = weight_decay
    if max_prompt_length is not None:
        kwargs["max_prompt_length"] = max_prompt_length
    if max_completion_length is not None:
        kwargs["max_completion_length"] = max_completion_length

    # Deliberately DO NOT pass ``packing`` (ORPOConfig has no such field and
    # rejects it) or force ``gradient_checkpointing`` (no full-FT contract for
    # ORPO in v1.5; ORPOConfig's own default governs).
    return ORPOConfig(**kwargs)


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
        # v1.4 BACKEND-F-008 (Wave 6b features): training mode. ``"lora"`` (the
        # default) is the v1.3 contract — fine-tune a LoRA adapter on the
        # configured base model. ``"full"`` enables full fine-tuning for
        # models up to 3B parameters: no LoRA adapter, paged_adamw_8bit
        # optimizer (memory ceiling), gradient_checkpointing=True (sqrt(L)
        # activation memory), and a 10x-lower learning rate by default. Full
        # FT of a model >3B raises RUNTIME_FULL_FT_MODEL_TOO_LARGE at
        # construction time (the gate re-fires at load_model() time as a
        # belt-and-braces second check). Operators with 24GB+ datacenter
        # cards who want full FT of a 7B model should use a fork of the
        # trainer that lifts the ceiling — the 3B gate is the documented
        # consumer-tier contract and not a soft-warning.
        mode: str = "lora",
        # v1.5 T1.2 (ORPO Wave 2): training objective. ``"sft"`` (the default)
        # is the supervised fine-tuning path — byte-identical pre-v1.5
        # behavior for callers who do not pass this kwarg. ``"orpo"`` selects
        # the reference-free ORPO objective (Hong 2024, arXiv:2403.07691):
        # SFT + odds-ratio preference loss in one pass, NO separate reference
        # model. ORPO requires a preference dataset (``{chosen, rejected}`` ±
        # ``prompt``) and is supported with mode='lora' ONLY in v1.5 — a
        # method='orpo' + mode='full' combination raises InvalidSettingError
        # at construction. Defaults to None so the settings layer
        # (``settings.training.method``) governs when the kwarg is omitted;
        # config.py validates the value too, but a DIRECT Trainer(method=...)
        # call is re-validated here (defense in depth). The CLI threads
        # ``--method`` through this kwarg via the wave6b_candidate_kwargs
        # introspection filter — naming the param here is what makes the CLI
        # flag flow through.
        method: str | None = None,
        # v1.5 T1.2 (ORPO Wave 2): the ORPO odds-ratio loss weight (lambda in
        # the paper; ``beta`` in ORPOConfig). Ignored unless method='orpo'.
        # Defaults to None so ``settings.training.orpo_beta`` (default 0.1)
        # governs when omitted.
        orpo_beta: float | None = None,
        # v1.5 T2.1 (FP8 compute path): opt-in FP8 training via torchao float8
        # (Blackwell sm_90+). None ⇒ ``settings.training.fp8`` (default False)
        # governs. When True AND the card supports it (CUDA + sm>=9 + torchao
        # present), the BASE projection linears are converted to Float8Linear
        # AFTER the LoRA adapter is attached (adapter rank-r sub-linears,
        # lm_head, and embeddings excluded — converting the rank-r linears
        # crashes on backward because r is not divisible by 16). On an
        # unsupported card / missing torchao the trainer logs ONE WARN and
        # trains in bf16 (graceful degrade, no raise). mode='full'+fp8 and
        # method='orpo'+fp8 and explicit-4bit+fp8 are rejected by the gate
        # ladder below. Named EXACTLY ``fp8`` so the CLI's wave6b introspection
        # filter threads ``--fp8`` through (GLUE owns the flag).
        fp8: bool | None = None,
        # v1.5 T2.3 (rsLoRA, finding 19): rank-stabilized LoRA — alpha/sqrt(r)
        # scaling instead of alpha/r. None ⇒ ``settings.lora.use_rslora``
        # (default False). Threaded into PEFT's ``LoraConfig(use_rslora=...)``
        # at the adapter-build call site (both the unsloth + transformers
        # loaders). Zero inference cost; the merged adapter is unaffected.
        # Named EXACTLY ``use_rslora`` so the CLI introspection filter threads
        # ``--use-rslora`` through (GLUE owns the flag).
        use_rslora: bool | None = None,
        # v1.5 T2.1 (FP8 compute path): explicit 4-bit-quantization control.
        # Pre-v1.5 the trainer always loaded the base in 4-bit (nf4) — there
        # was no constructor knob, so 4-bit was *only ever the default*. This
        # kwarg makes an EXPLICIT 4-bit request detectable so the FP8 gate
        # ladder can refuse the stacked combination (FP8 and 4-bit are
        # alternatives, not stackable). None ⇒ the historical default-on
        # behavior (and FP8, when effective, silently flips it off with an INFO
        # log). load_in_4bit=True is an EXPLICIT request and FP8 + it raises;
        # load_in_4bit=False forces an unquantized base load.
        load_in_4bit: bool | None = None,
        # v1.5 T3.2 (reasoning-trace SFT / R1 distillation, finding 24): keep
        # the ``<think>`` chain-of-thought in the SFT target, drop empty /
        # over-long traces (datasets.filter_by_trace_length), and raise the
        # DEFAULT max_seq_length to 8192 when the operator left it at the
        # shipped 2048. ``<think>`` is treated as PLAIN TEXT — no special
        # tokens, no embedding resize — so the merge→GGUF→Ollama export stays
        # intact. SFT only (ignored under method='orpo'). None ⇒
        # ``settings.data.reasoning_trace`` (default False ⇒ byte-identical
        # v1.4 SFT). Named EXACTLY ``reasoning_trace`` so the CLI's wave6b
        # introspection filter threads ``--reasoning-trace`` through (GLUE owns
        # the flag).
        reasoning_trace: bool | None = None,
        # v1.5 T3.1 (MLX / Apple-Silicon backend): the compute-rail selector.
        # None ⇒ ``settings.training.backend`` (default "auto"). "auto" resolves
        # to "mlx" on an Apple-Silicon Mac with the [mlx] extra, else "cuda"
        # (so existing CUDA rigs are byte-identical). "cuda" forces the canonical
        # CUDA rail; "mlx" forces the Apple-Silicon rail. A FORCED "mlx" on a
        # non-Apple host raises InvalidSettingError (CONFIG_INVALID_SETTING) from
        # the guard below — it is an unrunnable request on that hardware. When
        # the effective backend is "mlx", train() short-circuits to
        # _train_with_mlx (which drives mlx_lm.lora via a subprocess seam) and
        # NEVER calls load_model(); orpo / fp8 / mode='full' are rejected for the
        # MLX rail (out of scope for v1.5). Named EXACTLY ``backend`` so the CLI
        # introspection filter threads ``--backend`` through (GLUE owns the flag).
        backend: str | None = None,
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

        # v1.4 BACKEND-F-008 (Wave 6b features): mode resolution. The kwarg is
        # already a string Literal candidate (``"lora"`` default | ``"full"``).
        # Validate the value before any expensive work so an operator typo
        # surfaces immediately rather than after the model has loaded. The
        # accepted set is the same as ``_build_sft_config``'s `mode` parameter.
        if mode not in {"lora", "full"}:
            raise InvalidSettingError(
                setting_name="mode",
                value=mode,
                expected="one of {'lora', 'full'}",
                suggestion=(
                    "Pass mode='lora' (the default) for LoRA fine-tuning, "
                    "OR mode='full' for full fine-tuning of a model <=3B "
                    "parameters. mode='full' on a model >3B raises "
                    "RUNTIME_FULL_FT_MODEL_TOO_LARGE at construction time."
                ),
            )
        self.mode = mode

        # v1.5 T1.2 (ORPO Wave 2): training-objective resolution. Kwarg-
        # authoritative (per-invocation ``method=`` wins), falling back to
        # ``settings.training.method`` (the env-var path), default "sft".
        # Validate the resolved value here even though config.py's pydantic
        # ``_reject_invalid_method`` validator already guards the settings
        # path — a DIRECT ``Trainer(method="dpo")`` call bypasses the settings
        # layer entirely, so the constructor must re-validate (defense in
        # depth). InvalidSettingError carries code CONFIG_INVALID_SETTING.
        self.method = method if method is not None else settings.training.method
        if self.method not in {"sft", "orpo"}:
            raise InvalidSettingError(
                setting_name="method",
                value=self.method,
                expected="one of {'sft', 'orpo'}",
                suggestion=(
                    "Pass method='sft' (the default) for supervised "
                    "fine-tuning, OR method='orpo' for reference-free ORPO "
                    "preference training (requires a {chosen, rejected} "
                    "dataset and mode='lora')."
                ),
            )
        # v1.5 T1.2 (ORPO Wave 2): the ORPO odds-ratio loss weight. Kwarg-
        # authoritative, falling back to ``settings.training.orpo_beta``
        # (default 0.1). Ignored unless method='orpo'.
        self.orpo_beta = (
            orpo_beta if orpo_beta is not None else settings.training.orpo_beta
        )
        # Re-audit #6 (trainer half): config.py validation only covers the
        # SETTINGS path; a DIRECT ``Trainer(orpo_beta=-1.0)`` / ``=0.0`` sets
        # ``self.orpo_beta`` here and would flow UNCLAMPED into
        # ``ORPOConfig(beta=...)`` — beta=0 degenerates ORPO to plain SFT (the
        # odds-ratio term vanishes) and a negative beta trains TOWARD the
        # rejected response. Defend the direct kwarg with the same
        # CONFIG_INVALID_SETTING the settings validator raises. Only fires for
        # method='orpo' (beta is inert for SFT, so a stray value there is
        # harmless and must not block a non-ORPO run).
        if self.method == "orpo" and self.orpo_beta <= 0:
            raise InvalidSettingError(
                "orpo_beta",
                self.orpo_beta,
                "a positive number",
                suggestion=(
                    "ORPO's odds-ratio weight beta must be > 0 (TRL/Hong 2024 "
                    "default 0.1). beta=0 collapses ORPO to plain SFT and a "
                    "negative beta trains toward the REJECTED response. Pass a "
                    "small positive value, e.g. orpo_beta=0.1."
                ),
            )

        # v1.5 T2.1 (FP8) / T2.3 (rsLoRA): resolve the two new feature knobs,
        # kwarg-authoritative with the settings layer as fallback (same
        # ``is not None`` discipline as the load-bearing fields above so a
        # legitimate ``False`` is not mistaken for "unset").
        self.fp8 = fp8 if fp8 is not None else settings.training.fp8
        self.use_rslora = (
            use_rslora if use_rslora is not None else settings.lora.use_rslora
        )
        # v1.5 T3.2 (reasoning-trace SFT): resolve the recipe flag + its
        # trace-length bounds (kwarg-authoritative, settings as fallback —
        # same ``is not None`` discipline). When on, _load_dataset filters the
        # materialized SFT dataset by ``<think>`` token length and the
        # max_seq_length auto-bump below widens the default window for CoT.
        self.reasoning_trace = (
            reasoning_trace
            if reasoning_trace is not None
            else settings.data.reasoning_trace
        )
        # Re-audit #7: reasoning-trace SFT is an SFT-ONLY recipe. The trace
        # filter in _load_dataset is gated on method=="sft", so on ORPO (or any
        # non-sft objective) reasoning_trace does NOTHING useful — yet leaving
        # it truthy would still fire the max_seq_length 2048->8192 bump (a real
        # memory change) and stamp ``reasoning_trace: True`` into run-history
        # (a lie). Neutralize it here: emit ONE WARN, force it False so the
        # bump is skipped AND hyperparameters record it honestly as inert.
        if self.reasoning_trace and self.method != "sft":
            logger.warning(
                "reasoning-trace SFT does not apply to method='%s'; ignoring "
                "reasoning_trace (it is an SFT-only recipe — the <think> "
                "trace-length filter runs only on the SFT data path). "
                "max_seq_length is left unchanged and run-history records "
                "reasoning_trace=False. Use method='sft' to train on reasoning "
                "traces.",
                self.method,
            )
            self.reasoning_trace = False
        self.min_trace_tokens = settings.data.min_trace_tokens
        self.max_trace_tokens = settings.data.max_trace_tokens
        # v1.5 T3.2: max_seq_length auto-bump. Reasoning traces (R1/QwQ CoT)
        # routinely exceed the shipped 2048 default; raise it to 8192 ONLY when
        # the operator left max_seq_length at the shipped default (kwarg None
        # AND settings still at 2048). An explicit max_seq_length — passed as a
        # kwarg OR set via BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH — always wins.
        # (Gated by self.reasoning_trace, which the non-sft guard above already
        # forced False, so an ORPO run never bumps max_seq.)
        if (
            self.reasoning_trace
            and max_seq_length is None
            and settings.model.max_seq_length == 2048
        ):
            logger.info(
                "reasoning_trace on: raised max_seq_length 2048->8192 for "
                "longer CoT; pass max_seq_length to override"
            )
            self.max_seq_length = 8192
        # v1.5 T2.1: remember whether the operator EXPLICITLY asked for 4-bit
        # (vs relying on the historical default-on behavior). The gate ladder
        # below refuses ``fp8 + explicit-4bit``; when 4-bit is only the default
        # an active FP8 path flips it off with an INFO log instead of raising.
        # ``self._load_in_4bit`` holds the resolved effective value the loaders
        # thread into BitsAndBytesConfig / FastLanguageModel.
        self._load_in_4bit_explicit = load_in_4bit is not None
        self._load_in_4bit = load_in_4bit if load_in_4bit is not None else True
        # v1.5 T2.1: FP8-effective state. Stays False unless load_model()
        # successfully converts ≥1 base linear to Float8Linear; the gate ladder
        # below may also force it False (unsupported card / missing lib) BEFORE
        # any load. Persisted into run-history hyperparameters.
        self._fp8_effective = False

        # v1.5 T1.2 (ORPO Wave 2): ORPO + mode='full' guard. ORPO is supported
        # with mode='lora' ONLY in v1.5 — the adapter is attached by
        # load_model()'s get_peft_model before train(), and the ORPO+full
        # combination (full-param weights + the odds-ratio objective +
        # gradient-checkpointing memory ceiling) is out of scope for this
        # release. Refuse the combination at construction so the operator
        # sees an actionable error before any model load. (Fires after BOTH
        # self.mode and self.method resolve.)
        if self.method == "orpo" and self.mode == "full":
            raise InvalidSettingError(
                setting_name="method+mode",
                value={"method": "orpo", "mode": "full"},
                expected="ORPO is supported with mode='lora' only in v1.5",
                suggestion="Use mode='lora' (default) with method='orpo'.",
            )

        # v1.4 BACKEND-F-008 (Wave 6b features): mode='full' gate. Refuse
        # construction for models whose parameter count exceeds the 3B
        # ceiling. The probe is best-effort at construction time (preset
        # table + heuristic regex over the model_id); load_model() re-fires
        # the check with the authoritative ``num_parameters()`` reading as
        # a belt-and-braces second gate. Operators passing an obscure HF id
        # we can't estimate get a deferred check; operators passing a
        # preset name or canonical HF id get the early refusal.
        if self.mode == "full":
            _enforce_full_ft_param_ceiling(self.model_name)
            # Per the Biderman 2024 / Thinking Machines 2025 quality math
            # (full FT needs ~10x lower LR than LoRA). Apply the divisor
            # ONLY when the operator did not explicitly override the
            # learning rate at construction time. Detection: the operator's
            # explicit ``learning_rate=`` value differs from
            # ``settings.training.learning_rate`` (the configured default).
            # When they pass the bare LoRA default we rewrite to ~10x lower;
            # explicit operator-supplied values win. This is a behavior
            # change documented in the mode='full' contract.
            if learning_rate is None:
                # Caller relied on the settings default — apply the full-FT
                # LR divisor so mode='full' lands at a sensible default
                # without operator intervention.
                self.learning_rate = (
                    settings.training.learning_rate / _FULL_FT_DEFAULT_LR_DIVISOR
                )
                logger.info(
                    f"mode='full': applied default learning_rate divisor "
                    f"({_FULL_FT_DEFAULT_LR_DIVISOR}x lower than LoRA default) "
                    f"-> learning_rate={self.learning_rate:.2e}. Pass "
                    f"learning_rate=<value> to override."
                )
            else:
                logger.info(
                    f"mode='full': honoring operator-supplied "
                    f"learning_rate={self.learning_rate:.2e} (no divisor "
                    f"applied; explicit override beats the full-FT default)."
                )

        # v1.5 T1.2 (ORPO Wave 2): ORPO LR default. Parallel to the mode='full'
        # LR-lowering block above and keyed on the SAME ``learning_rate is
        # None`` detection (operator relied on the settings default). ORPO's
        # combined SFT + odds-ratio loss is stable around 8e-6 (Hong 2024 /
        # TRL examples), ~25x below the 2e-4 LoRA SFT default, so a bare
        # Trainer(method='orpo') would otherwise train at a divergent LR.
        # Explicit ``learning_rate=`` wins. full+orpo is blocked above, so
        # this branch and the mode='full' branch are mutually exclusive (an
        # ``elif`` on self.mode != "full" would be equivalent — we gate on
        # self.method directly for readability).
        elif self.method == "orpo":
            if learning_rate is None:
                self.learning_rate = _ORPO_DEFAULT_LR
                logger.info(
                    f"method='orpo': applied default learning_rate "
                    f"-> {self.learning_rate:.2e} (ORPO's combined SFT + "
                    f"odds-ratio loss is stable in the ~8e-6 band, well below "
                    f"the LoRA SFT default). Pass learning_rate=<value> to "
                    f"override."
                )
            else:
                logger.info(
                    f"method='orpo': honoring operator-supplied "
                    f"learning_rate={self.learning_rate:.2e} (no ORPO default "
                    f"applied; explicit override wins)."
                )

        # v1.5 T2.1 (FP8 compute path): the gate ladder. Two distinct failure
        # philosophies, in priority order:
        #   * MISCONFIGURATION (a combination that can never work) → raise an
        #     InvalidSettingError (CONFIG_INVALID_SETTING) at construction so
        #     the operator fixes their call. These fire regardless of hardware.
        #   * ENVIRONMENT DEGRADE (fp8 asked for, hardware/library can't honor
        #     it) → log ONE WARN naming the reason + fix, set
        #     ``_fp8_effective=False``, and proceed in bf16. NEVER raise — this
        #     mirrors the unsloth→transformers fallback (a missing capability is
        #     not an operator error).
        # The ladder only runs when fp8 was requested; ``self._fp8_effective``
        # is already False from the resolution block above for the common path.
        if self.fp8:
            # (1) MISCONFIG: FP8 + full fine-tuning. FP8 is validated only for
            # the LoRA path in v1.5 (the adapter-excluding module filter assumes
            # an attached PEFT adapter; full-FT has no adapter to exclude and the
            # combined FP8 + full-param + checkpointing memory profile is
            # untested). Mirrors the ORPO+full guard above.
            if self.mode == "full":
                raise InvalidSettingError(
                    setting_name="fp8+mode",
                    value={"fp8": True, "mode": "full"},
                    expected="FP8 is supported with mode='lora' only in v1.5",
                    suggestion=(
                        "FP8 + full FT not supported in v1.5; use mode='lora'."
                    ),
                )
            # (2) MISCONFIG: FP8 + ORPO. FP8 was dogfood-validated with
            # method='sft' only in v1.5; the ORPO odds-ratio loss on FP8 base
            # linears is unverified.
            if self.method == "orpo":
                raise InvalidSettingError(
                    setting_name="fp8+method",
                    value={"fp8": True, "method": "orpo"},
                    expected="FP8 is supported with method='sft' only in v1.5",
                    suggestion=(
                        "FP8 validated with method='sft' only in v1.5."
                    ),
                )
            # (3) FP8 vs 4-bit. They are alternatives: FP8 keeps the base in
            # float8, 4-bit keeps it in nf4 — you cannot do both. If the
            # operator EXPLICITLY requested 4-bit (load_in_4bit=True), refuse
            # the stack. This is a MISCONFIG and fires regardless of hardware
            # (an explicit fp8+4bit request can never be honored). The DEFAULT
            # 4-bit flip is deliberately NOT done here — see (4)/(5) below: it
            # is deferred until we know FP8 is actually EFFECTIVE, so an
            # unsupported host that degrades to bf16 KEEPS the operator's
            # default 4-bit (the historical OOM-safe path) instead of silently
            # loading an unquantized ~2x-larger bf16 base.
            if self._load_in_4bit_explicit and self._load_in_4bit:
                raise InvalidSettingError(
                    setting_name="fp8+load_in_4bit",
                    value={"fp8": True, "load_in_4bit": True},
                    expected="FP8 and 4-bit are alternatives, not stackable",
                    suggestion=(
                        "FP8 and 4-bit are alternatives, not stackable. Drop "
                        "load_in_4bit=True (FP8 keeps the base in float8) or "
                        "drop fp8=True."
                    ),
                )
            # (4) ENVIRONMENT axis: is FP8 actually runnable here? CPU /
            # pre-Hopper / torchao-absent → degrade to bf16 with ONE WARN, no
            # raise. The capability + library probe is centralized in
            # ``_fp8_supported`` so load_model() and the tests share one truth.
            supported, reason = self._fp8_supported()
            if not supported:
                # DEGRADE. FP8 is NOT effective, so the default 4-bit is left
                # exactly as the operator had it: ON (the OOM-safe default) or
                # whatever they passed. Critically we do NOT strip the default
                # 4-bit here — doing so would load an unquantized bf16 base
                # (~2x the nf4 footprint) on a card that only fit via 4-bit.
                # The WARN must NOT claim "FP8 keeps the base in float8" (FP8
                # was disabled); it states the bf16 degrade and that 4-bit (if
                # any) is unchanged.
                still_4bit = self._load_in_4bit
                logger.warning(
                    "fp8=True requested but unavailable on this host: %s. "
                    "Falling back to bf16 (training proceeds, just without the "
                    "FP8 speed/memory win). This is an environment degrade, not "
                    "an error. Your base quantization is UNCHANGED: "
                    "load_in_4bit=%s (FP8 was NOT enabled, so it cannot have "
                    "replaced 4-bit).",
                    reason,
                    still_4bit,
                )
                self._fp8_effective = False
            else:
                # Provisionally effective — load_model()'s _apply_fp8_to_base
                # may still flip this False if the actual conversion raises
                # (try/except → bf16 fallback). The ONE experimental WARN fires
                # here so it is emitted exactly once per Trainer, at construction.
                self._fp8_effective = True
                logger.warning(
                    "fp8=True: FP8 training is EXPERIMENTAL in v1.5 (torchao "
                    "float8 path, validated on Blackwell sm_120). The base "
                    "projection linears will be converted to Float8Linear after "
                    "the LoRA adapter is attached; the adapter, lm_head, and "
                    "embeddings stay in bf16. Report anomalies (loss spikes, "
                    "export mismatches) so the path can graduate to stable."
                )
                # v1.5 T2.1 (FP8): FP8 vs default 4-bit. Now that FP8 is
                # EFFECTIVE (supported host + library), the default 4-bit is
                # superseded — FP8 keeps the base in float8, the two are
                # alternatives. Flip it off with an INFO log (the explicit
                # fp8+4bit case already raised above). This is deliberately
                # INSIDE the effective branch: on a degrade the operator's
                # default 4-bit is preserved so the base still loads quantized
                # (no surprise unquantized-bf16 OOM — re-audit finding #5).
                if self._load_in_4bit and not self._load_in_4bit_explicit:
                    self._load_in_4bit = False
                    logger.info(
                        "fp8=True: disabling the default 4-bit base quantization "
                        "(FP8 keeps the base in float8 — the two are "
                        "alternatives). Pass load_in_4bit=True to force 4-bit, "
                        "which conflicts with fp8 and will raise."
                    )
                # v1.5 T2.1 (FP8): packing is INCOMPATIBLE with FP8 and must be
                # off. TRL's ``packing=True`` enables padding-free training,
                # which flattens a batch into ONE variable-length sequence; the
                # token count is then whatever the rows happen to sum to (e.g.
                # 211). torchao's float8 matmul (``_scaled_mm``) hard-requires
                # the contracted dimension divisible by 16 and raises
                # "Expected trailing dimension of mat1 to be divisible by 16"
                # on the FIRST backward otherwise (dogfood-verified on sm_120,
                # SmolLM2-135M). With packing off, sequences pad to the fixed
                # ``max_seq_length`` so the token dim is batch x max_seq_length —
                # a multiple of 16 whenever max_seq_length is. Force it off here
                # (the resolved ``self.packing`` is what _build_sft_config reads;
                # the builder itself stays untouched) with an INFO breadcrumb so
                # an operator who set packing=True sees why it was overridden.
                if self.packing:
                    logger.info(
                        "fp8: disabling packing (was on). FP8's float8 matmul "
                        "requires the token dimension divisible by 16; packing's "
                        "padding-free variable-length sequences violate that and "
                        "crash on backward. Sequences now pad to max_seq_length="
                        f"{self.max_seq_length}."
                    )
                self.packing = False

        # v1.5 T3.1 (MLX / Apple-Silicon backend): backend resolution + gates.
        # Placed AFTER the FP8 ladder (and after mode/method resolve) so the
        # MLX unsupported-feature gates can see the resolved self.mode /
        # self.method / self.fp8. Kwarg-authoritative with the settings layer as
        # fallback (same ``is not None`` discipline as the fields above).
        self.backend = backend if backend is not None else settings.training.backend
        # Defense-in-depth value check (config.py validates the settings path; a
        # DIRECT Trainer(backend="rocm") bypasses it). Mirrors the method gate.
        if self.backend not in {"auto", "cuda", "mlx"}:
            raise InvalidSettingError(
                setting_name="backend",
                value=self.backend,
                expected="one of {'auto', 'cuda', 'mlx'}",
                suggestion=(
                    "Pass backend='auto' (the default — routes to MLX on an "
                    "Apple-Silicon Mac with the [mlx] extra, else CUDA), "
                    "backend='cuda' to force the CUDA rail, or backend='mlx' to "
                    "force the Apple-Silicon rail (macOS + arm64 only)."
                ),
            )
        # Resolve "auto" → concrete rail. resolve_backend("auto") consults
        # detect_apple_silicon(); "cuda"/"mlx" pass through unchanged.
        self._effective_backend = resolve_backend(self.backend)

        # Forced-mlx-on-non-Apple guard: a FORCED backend='mlx' on a host that
        # is not Apple Silicon (macOS + arm64 + the [mlx] extra) is unrunnable —
        # mlx-lm cannot exist here. Refuse it with a structured
        # CONFIG_INVALID_SETTING so the operator gets an actionable error before
        # any work. (backend='auto' never reaches here as "mlx" on a non-Apple
        # host — resolve_backend routed it to "cuda".)
        if self.backend == "mlx" and not detect_apple_silicon():
            raise InvalidSettingError(
                setting_name="backend",
                value="mlx",
                expected="Apple Silicon (macOS+arm64) with the [mlx] extra",
                suggestion=(
                    "Use backend='auto' (routes to CUDA here) or run on an "
                    "M-series Mac with pip install 'backpropagate[mlx]'."
                ),
            )

        # MLX unsupported-feature gates — only when the EFFECTIVE backend is the
        # MLX rail. These features are out of scope for the MLX backend in v1.5
        # (mlx_lm.lora is LoRA-SFT-only here). Co-located with the FP8 ladder so
        # all the cross-field training-shape refusals live together.
        if self._effective_backend == "mlx":
            if self.method == "orpo":
                raise InvalidSettingError(
                    setting_name="backend+method",
                    value={"backend": "mlx", "method": "orpo"},
                    expected="ORPO is not supported on the MLX backend in v1.5",
                    suggestion=(
                        "ORPO (reference-free preference tuning) runs on the "
                        "CUDA rail only in v1.5. Use method='sft' with "
                        "backend='mlx', or method='orpo' with a CUDA GPU."
                    ),
                )
            if self.fp8:
                raise InvalidSettingError(
                    setting_name="backend+fp8",
                    value={"backend": "mlx", "fp8": True},
                    expected="FP8 is a CUDA/Blackwell path; not supported on MLX",
                    suggestion=(
                        "FP8 (torchao float8) is a CUDA/Blackwell-only compute "
                        "path. Drop fp8=True on the MLX backend — Apple Silicon "
                        "uses its own unified-memory precision."
                    ),
                )
            if self.mode == "full":
                raise InvalidSettingError(
                    setting_name="backend+mode",
                    value={"backend": "mlx", "mode": "full"},
                    expected="mode='full' is not supported on the MLX backend in v1.5",
                    suggestion=(
                        "Full fine-tuning on the MLX rail is out of scope for "
                        "v1.5. Use mode='lora' (the default) with backend='mlx'."
                    ),
                )
            # use_rslora / use_dora are PEFT-specific knobs with no mlx_lm.lora
            # equivalent — WARN-and-ignore (do NOT raise) so an operator who set
            # them on a CUDA-tuned config can still run on MLX.
            if getattr(self, "use_rslora", False) or getattr(self, "use_dora", False):
                logger.warning(
                    "backend='mlx': use_rslora / use_dora are PEFT-specific and "
                    "have no mlx_lm.lora equivalent — ignored for this MLX run. "
                    "mlx_lm applies its own LoRA scaling (the alpha/r -> scale "
                    "mapping) on the last num_layers blocks."
                )

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
        # v1.5 T1.2 (ORPO Wave 2): surface the objective so a post-mortem grep
        # can confirm which path ran without reading the config.
        if self.method == "orpo":
            logger.info(f"  Method: orpo (beta={self.orpo_beta})")
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
    def _is_bnb_8bit_optim(optim: str) -> bool:
        """True when ``optim`` is a bitsandbytes 8-bit (CUDA-only) optimizer.

        bitsandbytes registers its quantized optimizers under names that
        either end in ``_8bit`` (``adamw_8bit``, ``adam_8bit``,
        ``lion_8bit``, ``lamb_8bit``, ...) or carry the ``paged_`` prefix
        (``paged_adamw_8bit``, ``paged_adamw_32bit``, ``paged_lion_32bit``,
        ...). Every one of them allocates CUDA tensors in its optimizer
        state and raises from ``bnb.functional.is_on_gpu`` if asked to
        ``.step()`` on CPU. This predicate is the single source of truth
        for "this optimizer cannot run without CUDA" so the CPU downgrade
        in :meth:`_detect_optim_for_card` covers the whole family rather
        than just the ``adamw_8bit`` default (TRAINER-A-001 / TRAINER-A-007).
        """
        if not optim:
            return False
        name = optim.lower()
        return name.endswith("_8bit") or name.startswith("paged_")

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

        1. If CUDA is unavailable (CPU-only runner / no GPU visible),
           downgrade ANY bitsandbytes 8-bit optimizer
           (``adamw_8bit`` / ``paged_adamw_8bit`` /
           ``paged_adamw_32bit`` / any ``*_8bit`` / ``paged_*`` bnb
           variant) → ``adamw_torch``. bitsandbytes 8-bit optimizers
           are CUDA-only — calling ``.step()`` on CPU tensors raises
           ``RuntimeError: All input tensors need to be on the same
           GPU`` from ``bnb.functional.is_on_gpu``. ``adamw_torch`` is
           the transformers-standard CPU-compatible AdamW. This rule
           runs BEFORE the explicit-override short-circuit (TRAINER-A-001 /
           TRAINER-A-007): a CPU runner physically cannot run a bnb
           optimizer, so even an explicit operator pin (or the
           ``mode='full'`` force to ``paged_adamw_8bit``) must be
           downgraded — otherwise the run crashes at ``.step()``.
        2. If the operator passed anything other than the documented
           default ``adamw_8bit`` (and we have CUDA), honor their choice
           unchanged. The explicit-override surface is the canonical
           knob for operators who pinned a specific optimizer for an LR
           schedule / fairness constraint / token budget that depends
           on it.
        3. If torch is missing or the CUDA capability query raises,
           leave the operator's config in place.
        4. If the card has < 24GB VRAM, upgrade ``adamw_8bit`` →
           ``paged_adamw_8bit`` (consumer-card tier).
        5. Otherwise leave the default in place (datacenter-class card
           with VRAM to spare).

        Returns the resolved optimizer string. Pure function; the
        caller threads it into SFTConfig.optim.
        """
        # Rule 1: a CPU-only runner cannot run ANY bitsandbytes 8-bit
        # optimizer — the downgrade must therefore run BEFORE the
        # explicit-override short-circuit below. This catches the
        # ``mode='full'`` force to ``paged_adamw_8bit`` (TRAINER-A-001)
        # and any explicit operator pin of a bnb 8-bit variant
        # (TRAINER-A-007), both of which previously slipped past the
        # ``!= "adamw_8bit"`` short-circuit and crashed at ``.step()``
        # with the bnb ``is_on_gpu`` RuntimeError. We only reach the
        # torch import / CUDA query for bnb 8-bit optimizers; non-bnb
        # optimizers (adamw_torch, sgd, adafactor, ...) skip straight to
        # the explicit-override return and never pay the import cost.
        if Trainer._is_bnb_8bit_optim(configured_optim):
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.info(
                        "_detect_optim_for_card: CUDA unavailable — "
                        "downgrading optim %r -> adamw_torch "
                        "(bitsandbytes 8-bit optimizers are CUDA-only).",
                        configured_optim,
                    )
                    return "adamw_torch"
            except Exception as exc:  # noqa: BLE001
                # Defensive: if torch is missing / the CUDA probe raises,
                # fall through to the legacy resolution path below rather
                # than crashing the optimizer resolver.
                logger.debug(
                    f"_detect_optim_for_card: CUDA availability probe failed "
                    f"({exc!r}); proceeding with legacy resolution for "
                    f"optim={configured_optim!r}."
                )

        # Rule 2: explicit operator override wins (CUDA present). The
        # documented v1.3 default in config.py is "adamw_8bit"; anything
        # else is the operator's explicit pick.
        if configured_optim != "adamw_8bit":
            return configured_optim
        try:
            import torch
            if not torch.cuda.is_available():
                logger.info(
                    "_detect_optim_for_card: CUDA unavailable — "
                    "downgrading optim adamw_8bit -> adamw_torch "
                    "(bitsandbytes 8-bit optimizers are CUDA-only)."
                )
                return "adamw_torch"
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

    def _fp8_supported(self) -> tuple[bool, str | None]:
        """v1.5 T2.1: is the FP8 (torchao float8) compute path runnable here?

        Three AND-ed requirements, each a distinct ENVIRONMENT axis (not an
        operator misconfiguration — those are handled by the constructor gate
        ladder). Returns ``(True, None)`` when all hold, else ``(False, reason)``
        where ``reason`` names the exact missing piece + the fix, so the
        constructor can emit ONE actionable degrade-to-bf16 WARN.

        1. **CUDA present.** FP8 tensor-core ops have no CPU kernel; on a
           CPU-only host FP8 is meaningless. (Also the most common test path —
           ``torch.cuda.is_available()`` patched False.)
        2. **torchao installed.** The float8 conversion lives in
           ``torchao.float8``; without the library there is nothing to convert
           with. Probed via ``feature_flags.check_feature("fp8")`` (find_spec —
           no import cost) so a late ``pip install`` is honored after
           ``refresh_features()``.
        3. **Compute capability >= 9.0 (Hopper / Blackwell).** FP8 needs 4th-gen
           (Hopper sm_90) or 5th-gen (Blackwell sm_120) tensor cores. Ada
           (sm_89) and earlier lack the FP8 matmul path — torchao would fall
           back to emulation or fail. The brief's verified contract targets the
           RTX 5090 (sm_120).

        torch import failures / capability-query errors are swallowed into a
        ``(False, reason)`` so a half-broken CUDA stack degrades rather than
        crashes (consistent with ``_detect_optimal_dtype``'s defensive query).
        """
        try:
            import torch
        except Exception as exc:  # noqa: BLE001 - torch genuinely optional at import edges
            return False, (
                f"PyTorch is not importable ({exc!r}); FP8 needs torch + CUDA. "
                "Install the training extras: pip install 'backpropagate[fp8]'."
            )
        if not torch.cuda.is_available():
            return False, (
                "CUDA is not available (CPU-only host); FP8 tensor-core ops have "
                "no CPU kernel. Run on a Hopper/Blackwell GPU, or drop fp8=True."
            )
        if not check_feature("fp8"):
            return False, (
                "torchao is not installed; the float8 conversion lives in "
                "torchao.float8. Install it: pip install 'backpropagate[fp8]' "
                "(or pip install torchao)."
            )
        try:
            major, _minor = torch.cuda.get_device_capability(0)
        except Exception as exc:  # noqa: BLE001 - defensive capability probe
            return False, (
                f"could not query CUDA compute capability ({exc!r}); cannot "
                "confirm FP8 (Hopper sm_90+) support. Drop fp8=True to be safe."
            )
        if major < 9:
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001 - name is best-effort for the message
                name = "this GPU"
            return False, (
                f"GPU compute capability is sm_{major}x ({name}); FP8 needs "
                "sm_90+ (Hopper) or sm_120 (Blackwell). Ada (sm_89) and earlier "
                "have no FP8 matmul path. Drop fp8=True; bf16 is the right mode "
                "for this card."
            )
        return True, None

    @staticmethod
    def _fp8_module_filter(module: Any, fqn: str) -> bool:
        """v1.5 T2.1: which modules torchao should convert to Float8Linear.

        THE LOAD-BEARING GOTCHA (dogfood-verified on Blackwell sm_120): a naive
        ``convert_to_float8_training(peft_model)`` crashes on the FIRST backward
        pass. torchao's default filter converts EVERY ``nn.Linear``, including
        the LoRA adapter's rank-``r`` sub-linears (``lora_A`` / ``lora_B``). FP8
        scaled-matmul requires inner dimensions divisible by 16; a rank like
        r=16 happens to pass but r=8 / r=24 / r=anything-not-%16 does not, and
        even when r%16==0 the adapter math is numerically fragile in float8.
        The fix is to convert ONLY the base projection linears and EXCLUDE:

        * any module whose FQN contains ``"lora_"`` (the adapter A/B linears);
        * the ``lm_head`` (output projection — float8 there hurts logit quality
          and the vocab dim is often not %16-friendly);
        * token / positional embeddings (``nn.Embedding`` — not a Linear, but
          some architectures wrap an ``embed``-named Linear; exclude by name).

        Returns True ⇒ convert this module. The predicate is intentionally a
        pure, static function of ``(module, fqn)`` so it can be unit-tested on a
        tiny mocked module tree without standing up torchao or a real model.
        """
        import torch.nn as nn

        # Only Linear layers are convertible at all.
        if not isinstance(module, nn.Linear):
            return False
        lowered = fqn.lower()
        # Exclude the LoRA adapter sub-linears (the load-bearing exclusion), the
        # LM head, and any embedding-named projection. Everything else — the
        # base q/k/v/o/gate/up/down projection linears — is converted.
        excluded_markers = ("lora_", "lm_head", "embed", "embedding")
        return not any(marker in lowered for marker in excluded_markers)

    def _apply_fp8_to_base(self) -> None:
        """v1.5 T2.1: convert the base projection linears to torchao Float8Linear.

        Called from :meth:`load_model` AFTER the LoRA adapter is attached (so
        the adapter's rank-``r`` sub-linears are present in the module tree and
        the :meth:`_fp8_module_filter` predicate can exclude them). No-op unless
        ``self._fp8_effective`` is True (the constructor gate ladder already
        confirmed CUDA + torchao + sm>=9 and emitted the experimental WARN).

        Failure policy mirrors the unsloth→transformers fallback: ANY exception
        during the CONVERSION (an architecture the filter mishandles, a
        torch-version skew, a torchao runtime quirk) is caught, logged at WARN,
        and the run continues in bf16 with ``self._fp8_effective`` flipped back
        to False — a partial/aborted FP8 conversion must never take training
        down.

        The ONE structured hard-failure escape hatch is the IMPORT: if
        ``_fp8_supported()`` reported torchao present (find_spec succeeded) but
        ``import torchao.float8`` then fails, the install is in a contradictory,
        unrecoverable state (half-installed torchao). That is NOT a graceful
        environment degrade — the gate already promised FP8 — so it re-raises as
        a structured ``TrainingError(code="RUNTIME_FP8_UNSUPPORTED")`` rather
        than silently dropping to bf16. (The GLUE agent registers that catalog
        row in exceptions.py this wave; referencing it by string before the row
        lands is safe — BackpropagateError WARNs on an unknown code, it does not
        crash.)

        torchao prints a "Skipping import of cpp extensions / upgrade to torch
        >= 2.11" banner on import — that is NOISE (it falls back to torch-native
        ``_scaled_mm``, which works on sm_120), not a fatal error.
        """
        if not self._fp8_effective:
            return
        # Import is the hard-failure axis (see docstring): a broken torchao that
        # passed the find_spec gate but can't actually import is a structured
        # RUNTIME_FP8_UNSUPPORTED, not a silent bf16 degrade.
        try:
            from torchao.float8 import (
                Float8LinearConfig,
                convert_to_float8_training,
            )
        except Exception as exc:  # noqa: BLE001 - broken-install detection
            self._fp8_effective = False
            raise TrainingError(
                "FP8 was reported available (torchao spec found) but "
                "'torchao.float8' could not be imported — the torchao install "
                f"is broken or incompatible ({type(exc).__name__}: {exc}).",
                suggestion=(
                    "Reinstall torchao matching your torch build: "
                    "pip install --force-reinstall 'backpropagate[fp8]'. Or run "
                    "without FP8 (drop fp8=True) to train in bf16."
                ),
                code="RUNTIME_FP8_UNSUPPORTED",
                cause=exc,
            ) from exc
        # Conversion is the graceful-degrade axis: ANY failure → bf16 + WARN.
        try:
            # Default rowwise-ish recipe is fine for LoRA fine-tuning; an
            # explicit config future-proofs against torchao default drift.
            fp8_config = Float8LinearConfig()
            convert_to_float8_training(
                self._model,
                config=fp8_config,
                module_filter_fn=self._fp8_module_filter,
            )
            # Count what we converted so the smoke can assert ≥1 base
            # Float8Linear AND zero lora_* Float8Linear, and so a post-mortem
            # log shows the conversion actually bit.
            try:
                from torchao.float8.float8_linear import Float8Linear

                converted = sum(
                    1
                    for _n, m in self._model.named_modules()
                    if isinstance(m, Float8Linear)
                )
            except Exception:  # noqa: BLE001 - counting is best-effort telemetry
                converted = -1
            if converted == 0:
                # Nothing matched the filter — the adapter is attached but no
                # base linear was converted. Treat as a soft degrade: FP8 is
                # not actually active, so be honest about it.
                self._fp8_effective = False
                logger.warning(
                    "fp8: convert_to_float8_training matched 0 base linears "
                    "(model architecture may not expose nn.Linear projections "
                    "the filter recognizes); proceeding in bf16."
                )
                return
            logger.info(
                "fp8: converted %s base projection linear(s) to Float8Linear "
                "(LoRA adapter, lm_head, embeddings excluded). FP8 compute "
                "path active.",
                converted if converted >= 0 else "an unknown number of",
            )
        except Exception as exc:  # noqa: BLE001 - broad by design: degrade, don't crash
            self._fp8_effective = False
            logger.warning(
                "fp8: conversion to Float8Linear failed (%s: %s); falling back "
                "to bf16. Training proceeds without the FP8 speed/memory win. "
                "Set fp8=False to silence this, or report the stack so the "
                "torchao float8 path can be hardened.",
                type(exc).__name__,
                exc,
            )

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
        2. If CUDA is unavailable (CPU-only runner / no GPU
           visible), force ``(False, False)`` — transformers'
           ``TrainingArguments._validate_args`` hard-rejects
           ``bf16=True`` or ``fp16=True`` on CPU with
           ``"Your setup doesn't support bf16/gpu. You need to
           assign use_cpu if you want to train the model on CPU."``
           The config defaults (``bf16=True, fp16=False``) would
           otherwise blow up every CPU train at SFTConfig
           construction time. fp32 is the only valid CPU choice.
        3. If torch is missing or the CUDA capability query raises,
           leave the operator's config in place (we can't tell
           what they have).
        4. If the card supports bf16 (compute capability >= 8.0, i.e.
           Ampere / Ada / Hopper / Blackwell): prefer (bf16=True,
           fp16=False).
        5. Otherwise prefer (bf16=False, fp16=True) — pre-Ampere
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
                if configured_bf16 or configured_fp16:
                    logger.info(
                        "_detect_optimal_dtype: CUDA unavailable — "
                        "forcing fp32 (bf16/fp16 are not supported on "
                        "CPU; transformers rejects them at SFTConfig "
                        "construction)."
                    )
                return (False, False)
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

    def estimate_vram(
        self,
        *,
        bytes_per_param: int = 2,
        quantize_base: bool = True,
        hidden_dim: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        overhead_fraction: float = 0.15,
        param_count_billions: float | None = None,
    ) -> VRAMEstimate:
        """v1.4 BACKEND-F-002 (Wave 6b features): VRAM pre-flight estimate.

        Operator-actionable wrapper around :func:`estimate_vram` that pulls
        the trainer's configured ``mode`` / ``lora_r`` / ``batch_size`` /
        ``gradient_accumulation`` / ``max_seq_length`` and returns a
        structured :class:`VRAMEstimate`. Call before ``.train()`` to ask
        "will this config OOM?" without paying the cost of a real attempt.

        Args:
            bytes_per_param: 2 for bf16/fp16 (default), 4 for fp32.
            quantize_base: True (default) — matches ``load_in_4bit=True``
                in the trainer's actual load path.
            hidden_dim: Model hidden dim — default 4096 (7B-class).
                Override for non-7B-class architectures.
            num_layers: Model num_layers — default 32 (7B-class).
            num_heads: Model num_heads — default 32 (7B-class).
            overhead_fraction: Fragmentation + framework overhead (15%).
            param_count_billions: Explicit param count if known; otherwise
                estimated from the model name.

        Returns:
            :class:`VRAMEstimate` carrying total_gb + per-component breakdown.

        Example:
            >>> trainer = Trainer("Qwen/Qwen2.5-7B-Instruct", lora_r=256)
            >>> est = trainer.estimate_vram()
            >>> if not est.fits_on_card(16):
            ...     print(f"Won't fit on 16GB: {est.summary()}")
        """
        return estimate_vram(
            model=self.model_name,
            mode=self.mode,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            batch_size=self.batch_size,
            gradient_accumulation=self.gradient_accumulation,
            max_seq_length=self.max_seq_length,
            bytes_per_param=bytes_per_param,
            quantize_base=quantize_base,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            overhead_fraction=overhead_fraction,
            param_count_billions=param_count_billions,
        )

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

        # v1.5 T2.1 (FP8): FP8 prefers the transformers backend. Unsloth may
        # inject its own quantized / fused linears that the nn.Linear module
        # filter does not recognize (so they would either be missed or
        # mis-converted), and the dogfood-verified contract is transformers +
        # PEFT + torchao float8. When FP8 is effective and the operator left
        # Unsloth on, force the transformers loader with an INFO log so the
        # behavior is visible rather than silent. (No-op when fp8 is inactive.)
        if self._fp8_effective and self.use_unsloth:
            logger.info(
                "fp8: forcing the transformers backend (use_unsloth disabled "
                "for this run). Unsloth injects fused/quantized linears the FP8 "
                "nn.Linear filter can miss; the verified FP8 path is "
                "transformers + PEFT + torchao float8."
            )
            self.use_unsloth = False

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

        # v1.4 BACKEND-F-008 (Wave 6b features): belt-and-braces re-fire of
        # the mode='full' parameter-count gate against the loaded model. The
        # construction-time gate accepted models whose param count couldn't
        # be estimated (preset table miss + heuristic regex miss). Now that
        # the model is actually loaded we can ask num_parameters() directly
        # — if the authoritative reading exceeds the ceiling, refuse before
        # any training happens. The check is no-op for mode='lora'.
        if self.mode == "full":
            _enforce_full_ft_param_ceiling(
                self.model_name, loaded_model=self._model
            )

        # v1.5 T2.1 (FP8): convert the base projection linears to Float8Linear
        # AFTER the LoRA adapter is attached (the loaders call get_peft_model
        # before returning), so the module filter can see — and exclude — the
        # adapter's rank-r sub-linears. No-op unless self._fp8_effective is True
        # (the gate ladder confirmed CUDA + torchao + sm>=9 at construction).
        # Any conversion failure inside degrades to bf16 with a WARN; a broken
        # torchao install raises RUNTIME_FP8_UNSUPPORTED.
        self._apply_fp8_to_base()

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
                # v1.5 T2.1 (FP8): was hardcoded True. Now reads the resolved
                # ``self._load_in_4bit`` — the gate ladder flips it False when
                # FP8 is effective (FP8 keeps the base in float8, not nf4).
                load_in_4bit=self._load_in_4bit,
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
        # v1.5 T2.3 (rsLoRA): forward use_rslora only when True so we don't
        # tickle the kwarg on an older Unsloth/PEFT that may not accept it
        # (same defensive shape as use_dora). alpha/sqrt(r) scaling.
        if self.use_rslora:
            lora_kwargs["use_rslora"] = True
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

        # v1.5 T2.1 (FP8): the base-load shape now depends on self._load_in_4bit
        # (was unconditionally nf4 4-bit). When FP8 is effective the gate ladder
        # flipped _load_in_4bit False, so we load an UNQUANTIZED bf16 base —
        # torchao's convert_to_float8_training operates on plain nn.Linear
        # layers, not bitsandbytes Linear4bit (which it can't convert). When
        # FP8 is inactive this is byte-identical to the pre-v1.5 nf4 path.
        from_pretrained_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": settings.model.trust_remote_code,
            "_label": f"transformers_from_pretrained:{self.model_name}",
        }
        if self._load_in_4bit:
            # Quantization config (the historical default path).
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            from_pretrained_kwargs["quantization_config"] = bnb_config
        else:
            # FP8 path: unquantized bf16 base so torchao can convert the
            # plain nn.Linear projections to Float8Linear post-LoRA-attach.
            from_pretrained_kwargs["torch_dtype"] = torch.bfloat16

        # Load model
        self._model = _retry_hf_call(
            AutoModelForCausalLM.from_pretrained,
            self.model_name,
            **from_pretrained_kwargs,
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

        # Prepare for training. prepare_model_for_kbit_training is a k-bit
        # (4/8-bit) helper; on the FP8 unquantized path there is no k-bit base
        # to prepare, but the call is still useful for gradient-checkpointing +
        # input-grad enablement and is safe on a non-quantized model, so it
        # runs in both paths (byte-identical to pre-v1.5 for the 4-bit path).
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
        # v1.5 T2.3 (rsLoRA): thread use_rslora into PEFT's LoraConfig (the
        # canonical wiring point). PEFT >= 0.7 accepts use_rslora; older PEFT is
        # handled by the strip-retry below. alpha/sqrt(r) scaling vs alpha/r;
        # zero inference cost, merge-safe. Forwarded only when True so the
        # legacy-shape default is byte-identical for callers who never set it.
        if self.use_rslora:
            lora_kwargs["use_rslora"] = True
        _init_w = self.init_lora_weights
        if _init_w and _init_w != "default":
            # PEFT's init_lora_weights accepts {True, False, "gaussian",
            # "pissa", "loftq"}; the trainer's surface only exposes
            # "default"|"pissa"|"loftq" (str -> str passthrough).
            lora_kwargs["init_lora_weights"] = _init_w
        try:
            lora_config = LoraConfig(**lora_kwargs)
        except TypeError as exc:
            # Old PEFT rejected one of the v1.3/v1.5 kwargs. Strip them and
            # retry with the legacy shape so the trainer doesn't hard-
            # fail on an Unsloth-pinned environment that ships an older
            # PEFT. WARN so the operator notices the silent downgrade.
            stripped: list[str] = []
            for k in ("use_dora", "use_rslora", "init_lora_weights"):
                if k in lora_kwargs:
                    stripped.append(k)
                    lora_kwargs.pop(k)
            logger.warning(
                f"PEFT LoraConfig rejected kwarg(s) {stripped!r} "
                f"({exc!r}); retrying with the legacy LoraConfig shape. "
                f"Upgrade PEFT >= 0.10 (DoRA) / >= 0.7 (rsLoRA / PiSSA / "
                f"LoftQ) to enable these features."
            )
            lora_config = LoraConfig(**lora_kwargs)
        self._model = get_peft_model(self._model, lora_config)

    def _build_trainer(
        self,
        training_args: Any,
        train_dataset: Any,
        callbacks: list[Any] | None,
    ) -> Any:
        """Construct the inner TRL trainer for the resolved objective.

        v1.5 T1.2 (ORPO Wave 2): single construction site for BOTH the
        first-attempt build AND the OOM-retry rebuild in :meth:`train`. Pre-
        ORPO those two sites each inlined ``SFTTrainer(...)`` and drifted apart
        once (BACKEND-A-003 — the retry path bypassed the shared SFTConfig
        helper). Routing both through this one helper makes construction-drift
        between the first attempt and the retry structurally impossible: when
        the objective grows a third construction wrinkle, there is exactly one
        place to change.

        For ``method == "orpo"`` builds an ``ORPOTrainer`` (NO reference model —
        ORPO is reference-free; NO ``peft_config`` — the LoRA adapter is
        already attached by :meth:`load_model`'s ``get_peft_model`` before
        train() runs, exactly as the SFT path relies on). Otherwise builds an
        ``SFTTrainer`` identically to the pre-ORPO path. Imports are
        method-conditional and local so the test mocks
        (``patch("trl.SFTTrainer")`` / ``patch("trl.ORPOTrainer")``) bind at
        call time.

        TRL 0.27+ uses ``processing_class`` (not ``tokenizer``) on both
        trainers.
        """
        if self.method == "orpo":
            from trl import ORPOTrainer

            # Cross-version shim (v1.5 T1.2): trl's ORPOTrainer (through 0.24)
            # sets ``model.warnings_issued["estimate_tokens"] = True`` to mute a
            # transformers FLOP-estimate warning. transformers 5.x REMOVED the
            # ``warnings_issued`` attribute from ``PreTrainedModel``, so that
            # write raises ``AttributeError`` inside the constructor — before a
            # single step — on the project's own target stack (trl 0.24 +
            # transformers 5.5, verified on LlamaForCausalLM under a PEFT
            # wrapper). Provide an inert dict when the attribute is absent so
            # ORPO works across the transformers 4.x/5.x boundary. Harmless on
            # 4.x (the attribute is already present via ``PreTrainedModel.
            # __init__`` and is left untouched); on 5.x it is an empty dict trl
            # writes to and nothing reads. Scoped to the ORPO path — SFT is
            # unaffected. Remove once trl's ORPOTrainer stops touching
            # ``warnings_issued`` (it is slated to move to trl.experimental).
            if not hasattr(self._model, "warnings_issued"):
                self._model.warnings_issued = {}

            return ORPOTrainer(
                model=self._model,
                processing_class=self._tokenizer,
                train_dataset=train_dataset,
                args=training_args,
                callbacks=callbacks,
            )
        from trl import SFTTrainer

        return SFTTrainer(
            model=self._model,
            processing_class=self._tokenizer,
            train_dataset=train_dataset,
            args=training_args,
            callbacks=callbacks,
        )

    def _build_training_args(
        self,
        *,
        steps: int | None,
        report_to: Any,
        run_name: str | None,
    ) -> Any:
        """Assemble the inner trainer's config for the resolved objective.

        v1.5 T1.2 (ORPO Wave 2): single config-assembly site for BOTH the
        first-attempt build AND the OOM-retry rebuild in :meth:`train`,
        mirroring :meth:`_build_trainer`. The OOM-retry path reads
        ``self.batch_size`` / ``self.gradient_accumulation`` AFTER the recovery
        loop has halved/doubled them in place, so this reads those instance
        attributes live (it is called fresh on each retry). Everything else is
        a pure function of constructor-resolved state + the per-call
        ``steps`` / ``report_to`` / ``run_name``.

        For ``method == "orpo"`` delegates to :func:`_build_orpo_config`
        (beta + max_length; NO packing / gradient checkpointing). Otherwise
        delegates to :func:`_build_sft_config` exactly as the pre-ORPO path.
        """
        if self.method == "orpo":
            return _build_orpo_config(
                output_dir=str(self.output_dir),
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation,
                max_steps=steps or settings.training.max_steps,
                learning_rate=self.learning_rate,
                warmup_steps=settings.training.warmup_steps,
                max_seq_length=self.max_seq_length,
                orpo_beta=self.orpo_beta,
                seed=settings.training.seed,
                lr_scheduler_type=settings.training.lr_scheduler_type,
                logging_steps=settings.training.logging_steps,
                save_steps=settings.training.save_steps,
                weight_decay=settings.training.weight_decay,
                report_to=report_to,
                run_name=run_name,
                optim=self.optim,
            )
        sft_config = _build_sft_config(
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
            # fallback) through to the helper.
            packing=self.packing,
            optim=self.optim,
            # v1.4 BACKEND-F-008 (Wave 6b features): thread the constructor-
            # resolved training mode (``"lora"`` default | ``"full"``).
            mode=self.mode,
        )
        # v1.5 T2.1 (FP8): layer the FP8 shape requirement ON TOP of the built
        # SFTConfig rather than threading fp8 into _build_sft_config (which
        # stays byte-identical — the dtype/precision logic there is untouched).
        # torchao's float8 matmul hard-requires the contracted TOKEN dimension
        # divisible by 16; with ordinary right-padding the token count is
        # batch x seq, and seq is dynamically padded to the longest row in the
        # batch (e.g. 280) — not a multiple of 16, which crashes _scaled_mm on
        # backward (dogfood-verified, sm_120). pad_to_multiple_of=16 rounds the
        # padded sequence length up to the next multiple of 16 so batch x seq is
        # always %16; padding_free=False guarantees a rectangular (not flattened
        # ragged) batch the multiple-of-16 padding can act on. Both are no-ops
        # for the non-FP8 path (we only touch the config when effective).
        if self._fp8_effective:
            _set = self._set_fp8_shape_constraints_on_sft_config
            _set(sft_config)
        return sft_config

    @staticmethod
    def _set_fp8_shape_constraints_on_sft_config(sft_config: Any) -> None:
        """v1.5 T2.1: set pad_to_multiple_of=16 + padding_free=False on an
        already-built SFTConfig so FP8's float8 matmul gets 16-aligned token
        dimensions.

        Factored out (and attribute-guarded) so a TRL version whose SFTConfig
        lacks either field degrades gracefully — we set what exists and skip
        what doesn't, logging at DEBUG. The fields exist on trl 0.24+ (the
        project's target stack); the guard is belt-and-braces for older/newer
        trl that renamed them.
        """
        if hasattr(sft_config, "pad_to_multiple_of"):
            sft_config.pad_to_multiple_of = 16
        else:  # pragma: no cover - version-dependent
            logger.debug(
                "fp8: SFTConfig has no pad_to_multiple_of field on this trl "
                "version; FP8 matmul may hit a non-16-divisible token dim."
            )
        if hasattr(sft_config, "padding_free"):
            sft_config.padding_free = False
        if hasattr(sft_config, "packing"):
            # Defensive: the constructor already forced self.packing False for
            # FP8, but re-assert on the config so a future code path that sets
            # packing elsewhere can't silently re-enable the ragged-sequence
            # shape FP8 can't handle.
            sft_config.packing = False

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

        # v1.5 T1.2 (ORPO Wave 2): the inner-trainer class (SFTTrainer or
        # ORPOTrainer) is imported inside :meth:`_build_trainer` so the
        # method-conditional import resolves at construction time (and so the
        # two test mocks ``patch("trl.SFTTrainer")`` / ``patch("trl.ORPOTrainer")``
        # both bind correctly). No module-level ``from trl import SFTTrainer``
        # here anymore — construction is centralized in _build_trainer.

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

        # v1.5 T3.1 (MLX / Apple-Silicon backend): route the WHOLE run to the
        # MLX rail when the effective backend is "mlx". This MUST happen BEFORE
        # ``if not self._is_loaded: self.load_model()`` — the CUDA load_model()
        # path probes torch.cuda and would raise GPUNotAvailableError on a Mac,
        # which is exactly the boundary this rail retires (V1_5_BRIEF finding
        # 21). _train_with_mlx drives mlx_lm.lora via a subprocess seam and does
        # NOT load a model into this process. The constructor already gated out
        # orpo / fp8 / mode='full' for the MLX rail, so this path is SFT-LoRA.
        if self._effective_backend == "mlx":
            return self._train_with_mlx(
                dataset, steps=steps, samples=samples, callback=callback
            )

        # Load model if not loaded
        if not self._is_loaded:
            self.load_model()

        # Load dataset.
        # v1.5 T1.2 (ORPO Wave 2): thread the objective so the loader returns
        # raw {chosen, rejected} preference rows for ORPO (via
        # to_preference_dataset) instead of collapsing to a single text
        # column, and so a mismatched dataset surfaces a structured
        # DatasetFormatError before the trainer is built.
        train_dataset = self._load_dataset(dataset, samples, method=self.method)

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
        # v1.5 T1.2 (ORPO Wave 2): the Windows pre-tokenize path tokenizes a
        # single ``text`` column ahead of training (see _pre_tokenize). ORPO
        # rows have NO text column — they carry raw {chosen, rejected, prompt}
        # and ORPOTrainer tokenizes its own paired rows with the chat template
        # at train time. Pre-tokenizing them would either KeyError on the
        # missing text column or destroy the pairing, so we gate this to
        # method='sft' only. ORPO on Windows defers tokenization to ORPOTrainer
        # (same as the non-Windows SFT path).
        if self.method == "sft" and os.name == "nt" and settings.windows.pre_tokenize:
            logger.info(
                "Pre-tokenization: applied (os.name=nt, windows.pre_tokenize=True) "
                "for dataset of %d samples",
                len(train_dataset),
            )
            train_dataset = self._pre_tokenize(train_dataset)
        else:
            logger.info(
                "Pre-tokenization: deferred to inner trainer "
                "(method=%s, os.name=%s, windows.pre_tokenize=%s) for dataset of %d samples",
                self.method,
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

        # v1.3 BACKEND-5 / BACKEND-7 + Wave 6a BACKEND-A-003: training-args
        # assembly is delegated to a module-level helper so the same defaults
        # reach the OOM-retry rebuild (and, for SFT, the MultiRunTrainer call
        # site). Both helpers own the consumer-card paged-optim upgrade + Ada
        # bf16/fp16 selection + CPU downgrades; operator overrides on
        # ``settings.training.*`` are honored.
        #
        # v1.5 T1.2 (ORPO Wave 2): pick the builder by objective. ORPO uses
        # _build_orpo_config (beta + max_length; NO packing / gradient
        # checkpointing); SFT uses _build_sft_config exactly as before. The
        # config-build is also re-run identically in the OOM-retry block below
        # — keep the two call shapes in lockstep.
        training_args = self._build_training_args(
            steps=steps, report_to=report_to, run_name=run_name
        )

        # F-003: build the HF-TrainerCallback bridge ONCE for this train()
        # call. None when no user callback or when transformers is unavailable
        # (the second branch should never fire in practice — transformers is a
        # hard dep of TRL — but guard so a malformed env doesn't crash).
        _bridge_cb = (
            _build_trl_bridge_callback(callback) if callback is not None else None
        )
        sft_callbacks = [_bridge_cb] if _bridge_cb is not None else None

        # v1.5 T1.2 (ORPO Wave 2): construct via the shared _build_trainer
        # helper (SFTTrainer or ORPOTrainer per self.method). This is the
        # FIRST of two construction sites; the OOM-retry rebuild below calls
        # the SAME helper so the first-attempt-vs-retry construction can never
        # drift (kills the BACKEND-A-003 drift-bug class for ORPO too).
        self._trainer = self._build_trainer(
            training_args, train_dataset, sft_callbacks
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
        # v1.5 T1.2 (ORPO Wave 2): train_on_responses_only is MEANINGLESS for
        # ORPO — it masks the user prompt in a single ``text`` column so loss
        # is computed only on the assistant turn, but ORPO trains on PAIRED
        # {chosen, rejected} rows (no single text column to mask). Gate it to
        # method='sft' so the ORPO trainer is never passed to the Unsloth
        # masker. resolved_response_markers stays None for ORPO.
        #
        # F-014: track the resolved (instruction, response) marker pair so the
        # hyperparameters dict built further down can persist it into run
        # history (auditable post-mortem when a probe falls back to ChatML on
        # a non-ChatML tokenizer).
        resolved_response_markers: tuple[str, str] | None = None
        if self.method == "sft":
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
            # v1.5 T1.2 (ORPO Wave 2): persist the training objective + the
            # ORPO loss weight so a run-history audit can tell which run was
            # ORPO vs SFT (and reproduce the beta) without re-reading the
            # config. orpo_beta is recorded for every run (harmless for SFT;
            # the audit field stays uniform across the run table).
            "method": self.method,
            "orpo_beta": self.orpo_beta,
            # v1.5 T2.1 (FP8): persist the EFFECTIVE FP8 state (not the
            # requested ``self.fp8``) so a run-history audit reflects what
            # actually ran — fp8=True that degraded to bf16 on an unsupported
            # card records ``"fp8": False`` honestly. v1.5 T2.3 (rsLoRA):
            # persist use_rslora for adapter-scaling provenance (which scaling
            # produced this adapter — alpha/r vs alpha/sqrt(r)).
            "fp8": self._fp8_effective,
            "use_rslora": self.use_rslora,
            # v1.5 T3.2: persist whether this run kept <think> CoT in the SFT
            # target + applied trace-length filtering (reasoning-trace recipe
            # provenance — distinguishes an R1-distillation run from plain SFT
            # in the run-history audit table).
            "reasoning_trace": self.reasoning_trace,
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

        # TRAINER-A-006: snapshot the operator-configured batch / accum BEFORE
        # the OOM-recovery loop. The loop halves self.batch_size (and doubles
        # gradient_accumulation) in place on each OOM; pre-fix those mutations
        # were permanent, so (a) a transient OOM silently shrank the Trainer's
        # configured batch for every SUBSEQUENT train() call on the same
        # instance, and (b) the run_history record (written from the original
        # values at run-start) mis-described what actually trained. We restore
        # the snapshot in the finally below (fixes the leak) and record the
        # EFFECTIVE batch that completed into run_history + TrainingRun.metadata
        # (fixes the record mismatch).
        _orig_batch_size = self.batch_size
        _orig_gradient_accumulation = self.gradient_accumulation

        try:
            # B-002: OOM-recovery loop. We re-instantiate the SFTTrainer on
            # each retry so it picks up the halved batch / doubled accum.
            oom_consecutive_at_min = 0
            oom_retries = 0
            result: Any = None

            while True:
                try:
                    # Re-create the inner trainer with current batch / accum on
                    # retry. (The first iteration uses the trainer built above.)
                    if oom_retries > 0:
                        # Wave 6a BACKEND-A-003 + v1.5 T1.2 (ORPO Wave 2):
                        # re-build via the SAME shared helpers the first attempt
                        # used (_build_training_args + _build_trainer) so the
                        # OOM-retry path inherits the identical paged/bf16
                        # upgrades AND the identical objective (SFT vs ORPO).
                        # The detectors are pure functions of the configured
                        # value, so the resolution is stable across retries;
                        # _build_training_args reads self.batch_size /
                        # self.gradient_accumulation live, picking up the
                        # recovery loop's halved/doubled values. Routing the
                        # rebuild through the same two helpers makes
                        # first-attempt-vs-retry drift structurally impossible —
                        # an ORPO retry rebuilds an ORPOTrainer, never an
                        # SFTTrainer (the BACKEND-A-003 drift-bug class, now
                        # also closed for ORPO).
                        training_args = self._build_training_args(
                            steps=steps, report_to=report_to, run_name=run_name
                        )
                        # F-003: rebuild the bridge for each OOM retry so the
                        # adapter is bound to the fresh trainer instance.
                        # ``callback`` is captured from the enclosing train()
                        # call; if None, sft_callbacks stays None.
                        _bridge_cb_retry = (
                            _build_trl_bridge_callback(callback) if callback is not None else None
                        )
                        sft_callbacks_retry = (
                            [_bridge_cb_retry] if _bridge_cb_retry is not None else None
                        )
                        self._trainer = self._build_trainer(
                            training_args, train_dataset, sft_callbacks_retry
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

                    # TRAINER-A-003: drop the reference to the dead SFTTrainer
                    # (and its bound model/optimizer graph that just OOM'd)
                    # BEFORE gc.collect() + empty_cache(). Pre-fix the cache
                    # reclaim fired while self._trainer still pinned the failed
                    # trainer, so gc couldn't collect it and empty_cache()
                    # couldn't return the VRAM it held — the halved-batch retry
                    # then started against an un-reclaimed allocator (worse for
                    # mode='full', where the optimizer state dominates VRAM).
                    # The retry path at the top of this loop unconditionally
                    # rebuilds self._trainer (oom_retries > 0 branch), so
                    # clearing it here is safe.
                    self._trainer = None
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

            # Stage C amend CORE-B-009: coerce loss values to finite floats
            # before persisting. HF's ``training_loss`` is normally a float,
            # but a degenerate run can surface None (no logged steps) or a
            # non-finite value (NaN/inf on divergence / fp16 overflow).
            # Persisting NaN/inf into ``final_loss`` / ``loss_history``
            # poisons downstream JSON (json.dump emits non-spec ``NaN``),
            # best-checkpoint comparisons (NaN compares false against
            # everything), and the run-history record. Coerce to a finite
            # float, defaulting non-finite/non-numeric to 0.0 with a warn.
            import math

            def _finite_loss(value: Any, *, context: str) -> float:
                try:
                    f = float(value)
                except (TypeError, ValueError):
                    logger.warning(
                        "Non-numeric %s (%r); recording 0.0 instead.",
                        context,
                        value,
                    )
                    return 0.0
                if not math.isfinite(f):
                    logger.warning(
                        "Non-finite %s (%r); recording 0.0 instead "
                        "(training may have diverged — check learning rate / "
                        "fp16 overflow).",
                        context,
                        f,
                    )
                    return 0.0
                return f

            final_loss = _finite_loss(
                getattr(result, 'training_loss', 0.0),
                context="final training_loss",
            )

            # Extract loss history from logs. Drop non-finite / non-numeric
            # entries rather than threading NaN through the persisted history.
            loss_history = []
            if hasattr(self._trainer, 'state') and self._trainer.state.log_history:
                for log in self._trainer.state.log_history:
                    if 'loss' not in log:
                        continue
                    try:
                        f = float(log.get('loss', 0))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(f):
                        loss_history.append(f)

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
                    # TRAINER-A-006: the batch / accum that ACTUALLY trained
                    # (post any OOM-recovery halving). When oom_retries == 0
                    # these equal the configured values; when recovery fired
                    # they reflect the survived-with settings, so the run
                    # record is honest about what produced these weights.
                    "effective_batch_size": self.batch_size,
                    "effective_gradient_accumulation": self.gradient_accumulation,
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

            # TRAINER-A-006: if OOM recovery halved the batch mid-run, patch the
            # persisted record with the effective hyperparameters that actually
            # produced these weights. The run-start record (record_run_started)
            # captured the CONFIGURED batch; without this patch run_history would
            # claim the run trained at the requested batch when it didn't.
            if oom_retries > 0 and (
                self.batch_size != _orig_batch_size
                or self.gradient_accumulation != _orig_gradient_accumulation
            ):
                try:
                    run_history.update_run(
                        run_id,
                        effective_batch_size=self.batch_size,
                        effective_gradient_accumulation=self.gradient_accumulation,
                        oom_retries=oom_retries,
                    )
                except Exception as hist_err:
                    logger.warning(
                        f"RunHistoryManager.update_run (effective batch) "
                        f"failed: {hist_err}"
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
            # TRAINER-A-006: restore the operator-configured batch / accum that
            # the OOM-recovery loop may have halved in place. Done in finally so
            # it fires on every exit path (success, recovery-exhausted, or any
            # other raise) — a transient OOM during ONE train() must not
            # silently re-configure the Trainer for the NEXT train() on the same
            # instance. The effective values that actually trained are already
            # captured in TrainingRun.metadata + the run_history record above,
            # so restoring here loses no information. No-op when recovery never
            # mutated them.
            self.batch_size = _orig_batch_size
            self.gradient_accumulation = _orig_gradient_accumulation
            # B-001: emit run_ended with status and release the context-var
            # binding so the thread doesn't carry our run_id into the next
            # caller. We deliberately do this in `finally` so the log line
            # fires on both happy and error paths.
            logger.info(f"run_ended run_id={run_id} status={status}")
            unbind_run_context("run_id", "session_kind")

    def _train_with_mlx(
        self,
        dataset: str | Any,
        *,
        steps: int | None,
        samples: int | None,
        callback: TrainingCallback | None,
    ) -> TrainingRun:
        """v1.5 T3.1: train on the MLX (Apple-Silicon) rail via ``mlx_lm.lora``.

        Structurally parallel to the CUDA ``train()`` bookkeeping so export /
        run-history / model-card stay uniform across rails: it mints + binds a
        run_id, records run-start/-completion in the on-disk run history (tagged
        ``backend="mlx"`` in the hyperparameters), wraps the
        :class:`~backpropagate.mlx_backend.MLXRunResult` into a
        :class:`TrainingRun`, appends it to ``self._training_runs``, sets
        ``self._has_trained = True``, and fires the ``on_complete`` callback.

        It does NOT call :meth:`load_model` — the MLX toolchain loads the model
        itself inside the ``mlx_lm.lora`` subprocess (unified memory; no CUDA
        probe), which is the whole point of retiring the "no macOS training"
        boundary. ``mlx_lm.lora`` performs its own dataset materialization from
        the prepared data dir, so there is no in-process ``_load_dataset`` /
        ``_pre_tokenize`` step on this rail.
        """
        import time

        from .mlx_backend import MLXBackend, prepare_mlx_data_dir

        # Mint + bind the run_id exactly like the CUDA path (single_run kind).
        run_id = uuid.uuid4().hex
        legacy_run_label = f"run_{len(self._training_runs) + 1}"
        bind_run_context(run_id=run_id, session_kind="single_run")
        start_time = time.time()
        status = "error"  # success path overwrites to "ok"

        # iters = steps (per-invocation) or the configured max_steps default.
        iters = steps or settings.training.max_steps
        max_samples = samples or settings.data.max_samples

        # The MLX adapter dir lives under output_dir/mlx_adapter; it is plain
        # safetensors and feeds the existing export_ollama_adapter path.
        adapter_dir = self.output_dir / "mlx_adapter"
        data_dir = self.output_dir / "mlx_data"

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
            "max_steps": iters,
            "max_samples": max_samples,
            "use_unsloth": False,  # never on the MLX rail
            "seed": settings.training.seed,
            "method": self.method,
            "orpo_beta": self.orpo_beta,
            "fp8": False,  # gated out for the MLX rail
            "use_rslora": self.use_rslora,
            "reasoning_trace": self.reasoning_trace,
            # v1.5 T3.1: tag the rail so a run-history audit can tell an MLX run
            # from a CUDA run without re-reading the config.
            "backend": "mlx",
        }
        try:
            run_history.record_run_started(
                run_id=run_id,
                model_name=self.model_name,
                dataset_info=dataset_info,
                hyperparameters=hyperparameters,
                session_kind="single_run",
                checkpoint_path=str(adapter_dir),
                dataset_hash=_compute_dataset_hash(dataset),
            )
        except Exception as hist_err:
            logger.warning(
                f"RunHistoryManager.record_run_started failed: {hist_err}"
            )

        logger.info(
            f"run_started run_id={run_id} legacy_label={legacy_run_label} "
            f"backend=mlx iters={iters}"
        )

        try:
            # Materialize the dataset into the mlx_lm.lora data-dir layout
            # (train.jsonl + optional valid.jsonl, one chat record per line).
            # v1.5 T3.2 / re-audit #10: thread reasoning_trace + its bounds so
            # the <think> trace-length filter runs on THIS rail too (the CUDA
            # rail filters in _load_dataset, which the MLX path skips). When
            # self.reasoning_trace is False (the common path, and always under
            # ORPO after the #7 neutralization) this is a no-op.
            prepare_mlx_data_dir(
                dataset,
                data_dir,
                seed=settings.training.seed,
                max_samples=max_samples,
                shuffle=settings.data.shuffle,
                reasoning_trace=self.reasoning_trace,
                min_trace_tokens=self.min_trace_tokens,
                max_trace_tokens=self.max_trace_tokens,
            )

            backend = MLXBackend(
                model=self.model_name,
                dataset_dir=data_dir,
                adapter_path=adapter_dir,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                learning_rate=self.learning_rate,
                iters=iters,
                batch_size=self.batch_size,
                max_seq_length=self.max_seq_length,
                seed=settings.training.seed,
            )
            result = backend.run()

            duration = time.time() - start_time
            # Re-audit LOW: a successful run whose loss could NOT be parsed from
            # mlx_lm.lora stdout must NOT be coerced to 0.0 — 0.0 reads as a
            # *perfect* run and silently corrupts run-history / model-card
            # provenance. The raw value (``None`` when unparseable) is the
            # honest signal: it flows verbatim to record_run_completed (which
            # takes ``float | None``). The ``TrainingRun.final_loss`` field is
            # typed ``float``, so use ``nan`` as the in-object sentinel for
            # "ran, loss unknown" — nan is never mistaken for a real (let alone
            # perfect) loss, and ``final_loss_parsed`` below records the truth.
            parsed_loss = result.final_loss  # None when stdout parse missed
            run_final_loss = parsed_loss if parsed_loss is not None else float("nan")
            run = TrainingRun(
                run_id=run_id,
                steps=result.iters,
                final_loss=run_final_loss,
                loss_history=[],
                duration_seconds=duration,
                samples_seen=max_samples,
                output_path=result.adapter_path,
                metadata={
                    "legacy_run_label": legacy_run_label,
                    "backend": "mlx",
                    # Preserve the honest "could not parse a loss" signal — the
                    # CUDA path always has a numeric final_loss; the MLX rail's
                    # best-effort stdout parse may not, so record the raw flag.
                    "final_loss_parsed": parsed_loss is not None,
                    "val_loss": result.val_loss,
                },
            )

            self._training_runs.append(run)
            # Mirror the CUDA path: flip the save()-tripwire flag now that a
            # real mlx_lm.lora run completed and produced a TrainingRun.
            self._has_trained = True

            if callback and callback.on_complete:
                try:
                    callback.on_complete(run)
                except Exception as cb_error:
                    logger.warning(
                        f"on_complete callback raised error: {cb_error}"
                    )

            loss_str = (
                f"{parsed_loss:.4f}" if parsed_loss is not None else "unparsed"
            )
            logger.info(
                f"Training complete (mlx): final_loss={loss_str}, "
                f"time={duration:.1f}s"
            )
            status = "ok"

            try:
                run_history.record_run_completed(
                    run_id=run_id,
                    # Honest provenance: the raw parsed value (None when the
                    # stdout loss parse missed), NOT a 0.0 that reads as perfect.
                    final_loss=parsed_loss,
                    loss_history=[],
                    steps=run.steps,
                    duration_seconds=duration,
                    checkpoint_path=run.output_path,
                )
            except Exception as hist_err:
                logger.warning(
                    f"RunHistoryManager.record_run_completed failed: {hist_err}"
                )

            return run
        except Exception as exc:
            # Record the failure in run history (best-effort) and fire on_error,
            # then re-raise — structurally parallel to the CUDA path. MLX
            # failures arrive as structured TrainingError / MLXUnavailableError
            # from MLXBackend.run(), or DatasetError from prepare_mlx_data_dir.
            try:
                run_history.record_run_failed(
                    run_id=run_id,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            except Exception as hist_err:
                logger.warning(
                    f"RunHistoryManager.record_run_failed failed: {hist_err}"
                )
            if callback and callback.on_error:
                try:
                    callback.on_error(exc)
                except Exception as cb_error:
                    logger.warning(f"on_error callback raised error: {cb_error}")
            raise
        finally:
            logger.info(f"run_ended run_id={run_id} status={status}")
            unbind_run_context("run_id", "session_kind")

    def _load_dataset(
        self,
        dataset: str | Any,
        samples: int | None = None,
        method: str = "sft",
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

        v1.5 T1.2 (ORPO Wave 2): when ``method == "orpo"`` the loader returns
        RAW preference pairs (``{chosen, rejected, [prompt]}``) instead of the
        collapsed single-text SFT shape:

        * DatasetLoader paths (a ``DatasetLoader`` passed directly OR a local
          file string) call :meth:`DatasetLoader.to_preference_dataset`, which
          preserves the pair columns and raises ``DatasetFormatError``
          (``INPUT_DATASET_FORMAT_UNSUPPORTED``) when no row carries both
          ``chosen`` and ``rejected``.
        * HF-name and in-memory ``Dataset`` paths cannot re-run format
          detection, so they assert ``chosen`` and ``rejected`` are present in
          ``column_names`` and raise ``DatasetFormatError`` otherwise.

        Symmetrically, when ``method == "sft"`` and a DatasetLoader detects a
        PREFERENCE-shaped file, a WARN is emitted — SFT training proceeds on
        the ``chosen`` response (the loader knows how to render preference
        rows for SFT: each row becomes prompt -> chosen, dropping
        ``rejected``), and the operator is told they can pass ``--method
        orpo`` to also learn from the ``rejected`` response.

        Raises:
            DatasetNotFoundError: If dataset file doesn't exist
            DatasetParseError: If dataset cannot be parsed
            DatasetFormatError: method='orpo' on a dataset with no
                chosen/rejected columns (code INPUT_DATASET_FORMAT_UNSUPPORTED)
            DatasetError: For other dataset-related errors
        """
        from datasets import Dataset, load_dataset

        from .datasets import DatasetFormat
        from .exceptions import DatasetFormatError

        is_orpo = method == "orpo"
        max_samples = samples or settings.data.max_samples

        # File extensions that DatasetLoader handles
        _LOCAL_FILE_EXTENSIONS = (
            '.jsonl', '.json', '.csv', '.parquet', '.txt', '.md',
        )

        # v1.5 T1.2 (ORPO Wave 2): the supported_formats list reused in every
        # ORPO DatasetFormatError raised from the non-loader paths below.
        _ORPO_SUPPORTED = ["{chosen, rejected}", "{prompt, chosen, rejected}"]

        def _emit_sft_on_preference_warning(detected: DatasetFormat) -> None:
            """WARN when an SFT run is pointed at a preference-shaped file."""
            if not is_orpo and detected == DatasetFormat.PREFERENCE:
                logger.warning(
                    "Dataset looks like preference pairs (detected format "
                    "'preference': rows carry {chosen, rejected}), but "
                    "method='sft'. Training SFT on the 'chosen' response "
                    "(each row renders as prompt -> chosen, dropping "
                    "'rejected'). Pass --method orpo to also learn from the "
                    "'rejected' response (reference-free ORPO preference "
                    "training)."
                )

        def _require_preference_columns(hf_ds: Any, source: str) -> None:
            """Assert an HF/in-memory Dataset carries chosen+rejected for ORPO."""
            cols = getattr(hf_ds, "column_names", []) or []
            if "chosen" not in cols or "rejected" not in cols:
                raise DatasetFormatError(
                    f"method='orpo' requires preference pairs but the {source} "
                    f"has no chosen/rejected columns (columns: {list(cols)}). "
                    f"ORPO needs each row to carry both 'chosen' and "
                    f"'rejected' (optionally 'prompt').",
                    detected_format=None,
                    supported_formats=_ORPO_SUPPORTED,
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
                if is_orpo:
                    _require_preference_columns(ds, "default HuggingFace dataset")
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
                if is_orpo:
                    # Preserves raw chosen/rejected/[prompt]; raises
                    # DatasetFormatError when no pair row exists.
                    ds = dataset.to_preference_dataset()
                else:
                    _emit_sft_on_preference_warning(dataset.detected_format)
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

                    if is_orpo:
                        # to_preference_dataset raises DatasetFormatError when
                        # the file is an SFT file with no chosen/rejected rows
                        # (the "operator pointed ORPO at an SFT dataset" case).
                        ds = loader.to_preference_dataset()
                    else:
                        _emit_sft_on_preference_warning(loader.detected_format)
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
                    if is_orpo:
                        _require_preference_columns(
                            ds, f"HuggingFace dataset '{dataset}'"
                        )
            elif isinstance(dataset, Dataset):
                ds = dataset
                if is_orpo:
                    _require_preference_columns(ds, "in-memory Dataset")
            else:
                raise DatasetError(
                    f"Unsupported dataset type: {type(dataset).__name__}",
                    suggestion="Use a file path (JSONL, CSV, Parquet), HuggingFace dataset name, Dataset object, or DatasetLoader"
                )
        except (DatasetNotFoundError, DatasetParseError, DatasetError):
            # DatasetFormatError is a DatasetError subclass — propagates here
            # unchanged (the ORPO format guard must not be downgraded to a
            # generic DatasetError by the catch-all below).
            raise
        except FileNotFoundError as e:
            raise DatasetNotFoundError(str(e)) from e
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {e}") from e

        # Limit samples. Method-agnostic: preference Datasets are ordinary
        # datasets.Dataset objects, so shuffle/select apply identically.
        if max_samples > 0 and len(ds) > max_samples:
            if settings.data.shuffle:
                ds = ds.shuffle(seed=settings.training.seed)
            ds = ds.select(range(max_samples))

        # v1.5 T3.2 (reasoning-trace SFT): drop empty / over-long <think>
        # traces from the materialized SFT dataset. Gated on method == "sft" —
        # reasoning-ORPO is out of scope (the ORPO path returns raw preference
        # pairs, not a single collapsed text column, so trace filtering would
        # not apply). load_model() runs before _load_dataset, so the tokenizer
        # is available for an exact (non-approximate) token count.
        if self.reasoning_trace and method == "sft":
            ds = self._filter_reasoning_traces(ds)

        logger.info(f"Loaded {len(ds)} samples")
        return ds

    def _filter_reasoning_traces(self, ds: Any) -> Any:
        """Drop empty / over-long ``<think>`` traces from an SFT dataset.

        v1.5 T3.2. Routes the materialized HF dataset's text rows through
        :func:`datasets.filter_by_trace_length` (CORE's PINNED contract), which
        keeps only rows whose ``<think>`` span tokenizes within
        ``[self.min_trace_tokens, self.max_trace_tokens]`` and (by default)
        carries a ``<think>`` span at all. ``<think>`` stays PLAIN TEXT — this
        function never touches the tokenizer's vocabulary (no
        ``add_special_tokens`` / ``resize_token_embeddings``); it only *counts*
        tokens via ``self._tokenizer.encode`` so the merge→GGUF→Ollama export
        pipeline is unaffected.

        Also runs CORE's empty-``<think>`` advisory: if the model's chat
        template itself injects an empty ``<think></think>`` (some R1/QwQ
        templates do), a dataset whose targets ALSO open with ``<think>`` would
        double the marker. The template is rendered once (the same
        ``apply_chat_template`` probe :func:`_detect_chat_markers` uses) to
        detect that, and the advisory warns when the combination would double.

        Best-effort: a missing tokenizer or a malformed row never aborts the
        run — the dataset is returned unfiltered with a WARN. The filter's own
        :meth:`TraceFilterStats.summary` is logged at INFO (mirrors the
        ``filter_by_quality`` INFO pattern).
        """
        from .datasets import (
            _extract_think_spans,
            filter_by_trace_length,
            warn_on_doubled_think,
        )

        text_column = settings.data.text_column
        column_names = list(getattr(ds, "column_names", []) or [])
        if text_column not in column_names:
            logger.warning(
                f"reasoning_trace: dataset has no '{text_column}' column "
                f"(columns: {column_names}); skipping trace-length filtering. "
                "Reasoning-trace SFT expects a single collapsed text column "
                "containing the <think>...</think> target."
            )
            return ds

        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            logger.warning(
                "reasoning_trace: tokenizer not loaded; skipping trace-length "
                "filtering (run load_model() before _load_dataset)."
            )
            return ds

        def _token_counter(text: str) -> int:
            # Plain length of the encoded ids — NO vocabulary mutation. Falls
            # back to a whitespace count if the tokenizer rejects the text so a
            # single odd row cannot abort the whole run.
            try:
                return len(tokenizer.encode(text))
            except Exception:  # nosec B110 — best-effort token count; whitespace fallback
                return len(text.split())

        # Empty-<think> advisory: render the chat template once (same probe as
        # _detect_chat_markers) to learn whether the template itself opens an
        # empty <think>. CORE's warn_on_doubled_think compares that against the
        # dataset targets and warns on the doubling case.
        template_injects_think = False
        try:
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": _CHAT_MARKER_PROBE_USER},
                    {"role": "assistant", "content": _CHAT_MARKER_PROBE_ASST},
                ],
                tokenize=False,
            )
            if isinstance(rendered, str):
                template_injects_think = bool(_extract_think_spans(rendered))
        except Exception as exc:
            logger.debug(
                f"reasoning_trace: chat-template <think> probe failed ({exc!r})"
            )

        # Materialize the text rows as dicts for CORE's filter. CORE's
        # filter_by_trace_length + warn_on_doubled_think read the literal
        # ``"text"`` key (the convert_to_chatml shape), so build the probe
        # dicts under ``"text"`` regardless of the configured text_column, then
        # rebuild the HF dataset under the ORIGINAL column name below so
        # downstream packing / pre-tokenization (which read
        # settings.data.text_column) are unaffected when text_column != "text".
        texts = [str(row) for row in ds[text_column]]
        samples = [{"text": t} for t in texts]

        # Advisory only — never mutates samples, never raises on the happy
        # path; guard anyway so a future CORE change can't abort a run.
        try:
            warn_on_doubled_think(
                samples, template_injects_think=template_injects_think
            )
        except Exception as exc:  # pragma: no cover - advisory is non-fatal
            logger.debug(f"reasoning_trace: doubled-<think> advisory skipped ({exc!r})")

        try:
            kept, stats = filter_by_trace_length(
                samples,
                min_trace_tokens=self.min_trace_tokens,
                max_trace_tokens=self.max_trace_tokens,
                require_think=True,
                token_counter=_token_counter,
            )
        except Exception as exc:
            logger.warning(
                f"reasoning_trace: trace-length filtering failed ({exc!r}); "
                "proceeding with the unfiltered dataset."
            )
            return ds

        logger.info(f"reasoning_trace: {stats.summary()}")

        if not kept:
            logger.warning(
                "reasoning_trace: trace-length filtering removed every sample "
                f"(min={self.min_trace_tokens}, max={self.max_trace_tokens} "
                "think tokens, require_think=True). Returning the unfiltered "
                "dataset so the run does not start on an empty set — relax the "
                "trace bounds or check that the targets carry <think> spans."
            )
            return ds

        from datasets import Dataset

        # Rebuild under the original column name (kept rows carry the "text"
        # key CORE filtered on). When text_column == "text" this is a straight
        # passthrough; otherwise remap so the column matches the rest of the
        # pipeline.
        return Dataset.from_list([{text_column: row["text"]} for row in kept])

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

        B-006 + TRAINER-A-004: writes flow into ``<path>.partial`` first; the
        partial directory is cleaned up on any failure path so the operator
        never sees a half-written PEFT directory (config.json present, weights
        missing — which raises a cryptic 'state_dict missing keys' on the next
        resume attempt). Promotion is crash-safe even when overwriting an
        existing checkpoint: the prior checkpoint is renamed aside to
        ``<path>.backup`` (a fast same-directory rename) BEFORE the new one is
        moved into place, and the backup is deleted only after the promote
        succeeds. A crash at any point therefore leaves a recoverable
        checkpoint — the new one at ``<path>``, or the prior one at
        ``<path>.backup`` (auto-recovered on the next ``save``). This closes
        the pre-fix window where ``rmtree(output); move(partial)`` could leave
        NOTHING on disk if the process died mid-promote (a window widened on
        cross-filesystem moves, where ``shutil.move`` degrades to copy+delete).

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
        # TRAINER-A-004: sibling holding the PRIOR checkpoint while the new one
        # is promoted into place, so a crash mid-promote leaves a recoverable
        # checkpoint (either the new one at output_path or the prior one here)
        # rather than nothing.
        backup_path = output_path.with_name(output_path.name + ".backup")

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
        # TRAINER-A-004: a stale .backup means a PRIOR save crashed after it
        # renamed the old checkpoint aside but before it could delete the
        # backup. If output_path is missing, that backup is the only surviving
        # copy — recover it rather than wiping it. Only wipe the backup when a
        # live output_path already exists (the prior promote actually finished).
        if backup_path.exists():
            if not output_path.exists():
                logger.warning(
                    "Recovering checkpoint from a previous crashed save: "
                    "promoting %s back to %s (the active checkpoint was missing).",
                    backup_path, output_path,
                )
                try:
                    shutil.move(str(backup_path), str(output_path))
                except OSError as e:
                    logger.warning(f"Failed to recover stale backup checkpoint: {e}")
            else:
                shutil.rmtree(backup_path, ignore_errors=True)

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

            # TRAINER-A-004: crash-safe promote that keeps a recoverable
            # checkpoint through the entire window. Pre-fix the sequence was
            # ``rmtree(output_path); move(partial -> output)`` — between those
            # two statements (a window WIDENED on cross-filesystem moves, where
            # ``shutil.move`` degrades to copy-then-delete) a crash left NO
            # checkpoint at all: the prior one deleted, the new one half-copied.
            # New sequence: rename the prior checkpoint ASIDE (fast same-dir
            # rename), move the new one in, then delete the backup. A crash
            # after the rename but before the move leaves the prior checkpoint
            # at ``backup_path`` (recovered on the next save by the stale-backup
            # handler above); a crash mid-move leaves the new (partial) at
            # output_path AND the prior intact at backup_path.
            had_prior = output_path.exists()
            if had_prior:
                # Clear any leftover backup target first (defensive — the
                # stale-backup handler above already dealt with the common
                # case, but a same-process double-save could re-create it).
                if backup_path.exists():
                    shutil.rmtree(backup_path, ignore_errors=True)
                os.rename(str(output_path), str(backup_path))
            try:
                shutil.move(str(partial_path), str(output_path))
            except Exception:
                # Promote failed — restore the prior checkpoint so the operator
                # is never left WORSE off than before the save attempt.
                if had_prior and not output_path.exists() and backup_path.exists():
                    try:
                        os.rename(str(backup_path), str(output_path))
                        logger.warning(
                            "Promote failed; restored the prior checkpoint at %s.",
                            output_path,
                        )
                    except OSError as restore_err:
                        logger.error(
                            "Promote failed AND prior-checkpoint restore failed: "
                            "%s. The prior checkpoint is at %s.",
                            restore_err, backup_path,
                        )
                raise
            # Promote succeeded — the prior checkpoint is now safe to delete.
            if had_prior and backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)
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

        # v1.5 T3.1 (MLX / Apple-Silicon backend): SLAO multi-run is
        # PEFT-tensor-specific (it loads, merges, and re-applies LoRA A/B
        # tensors via peft + the merge framework) and has no mlx_lm equivalent —
        # the MLX rail produces an mlx safetensors adapter through a subprocess,
        # not in-process PEFT tensors. Refuse the combination with a structured
        # CONFIG_INVALID_SETTING before the CUDA GPU probe below (which would
        # itself raise on a Mac). Single-run MLX (Trainer.train) IS supported.
        if self._effective_backend == "mlx":
            raise InvalidSettingError(
                setting_name="backend",
                value="mlx",
                expected="multi_run() is CUDA-only in v1.5",
                suggestion=(
                    "SLAO multi-run LoRA merging is PEFT-tensor-specific and "
                    "out of scope for the MLX backend in v1.5. Use the "
                    "single-run path (Trainer.train) with backend='mlx', or run "
                    "multi_run() on a CUDA GPU (backend='cuda'/'auto')."
                ),
            )

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
            # v1.4 BACKEND-F-008 (Wave 6b features): forward the outer
            # Trainer's mode to MultiRunConfig so ``Trainer(mode='full')
            # .multi_run(...)`` produces a coherent full-FT multi-run
            # session. Operator-level intent is preserved end-to-end:
            # outer Trainer mode → MultiRunConfig.mode → inner Trainer
            # mode → _build_sft_config(mode=...).
            mode=self.mode,
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
