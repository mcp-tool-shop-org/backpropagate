"""
Backpropagate — Reflex state classes
=====================================

The ``rx.State`` subclasses that drive the four UI surfaces (Train, Multi-Run,
Export, Dataset). Phase 1 stubs: the field definitions match the design digest
contract, and the event handlers transition state without wiring to the
training backend yet. Phase 3 of the migration plan replaces the stubs with
real ``backpropagate.Trainer`` / ``backpropagate.MultiRunTrainer`` integration.

The state classes are intentionally split (not one mega-class) so each
surface's WebSocket bundle stays small. Reflex coalesces the per-class state
into a single client connection automatically.

A shared ``_TrainConfigMixin`` carries the Model + Training-shape + LoRA +
Dataset config fields plus their validated setters; ``TrainState`` and
``MultiRunState`` both inherit, so config typed on one surface persists when
the operator switches to the other. The setters fan in for every numeric and
path-shaped input; numeric clamps land in the setter (not the form) so the
``Trainer`` hookup in Phase 3 sees validated state regardless of the input
route.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Literal

import reflex as rx

# Shared literal types — referenced across multiple State classes.
RunState = Literal["idle", "loading", "active", "paused", "done", "error"]
Theme = Literal["dark", "light"]
ActiveSurface = Literal["train", "multi-run", "export", "dataset"]
ExportFormat = Literal["lora", "merged", "gguf"]
Quantization = Literal["4-bit", "8-bit", "16-bit"]
MergeMode = Literal["slao", "weighted", "ties"]
GgufQuant = Literal["q2_K", "q3_K_M", "q4_K_M", "q5_K_M", "q6_K", "q8_0"]
DatasetFormatHint = Literal["auto", "sharegpt", "alpaca", "openai", "jsonl"]

# Constants for the setters' clamps. Centralised so an operator can read the
# bounds in one place — they also appear in the operator-facing error strings.
_STEPS_MIN, _STEPS_MAX = 1, 100_000
_LR_MIN, _LR_MAX = 1e-7, 1.0
_LORA_R_MIN, _LORA_R_MAX = 1, 256
_LORA_ALPHA_MIN, _LORA_ALPHA_MAX = 1, 512
_LORA_DROPOUT_MIN, _LORA_DROPOUT_MAX = 0.0, 1.0
_GPU_TEMP_MIN, _GPU_TEMP_MAX = 40, 110
_NUM_RUNS_MIN, _NUM_RUNS_MAX = 1, 100
_SAMPLES_PER_RUN_MIN, _SAMPLES_PER_RUN_MAX = 1, 1_000_000
_TOKENS_MIN, _TOKENS_MAX = 0, 1_000_000

# Comma-separated identifier list (LoRA target modules). The character set is
# strict on purpose — anything outside it cannot resolve to a real attention
# module name and the only reason to type it is mistake or injection probe.
_TARGET_MODULES_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_,\s]*$")

# W&B run name: same shape that wandb itself accepts.
_WANDB_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._\-]+$")


# ---------------------------------------------------------------------------
# Path validation helper — FRONTEND-A-002 fix
# ---------------------------------------------------------------------------
#
# All user-supplied path fields on the four state classes flow through this
# helper so the FB-003 + F-002 hardening in ``ui_security`` is exercised by
# the Reflex surface (it was previously dead code there). The helper resolves
# the value against ``get_ui_output_dir()`` as the allowed base; absolute
# escapes and ``..`` traversal raise ``PathTraversalError`` which the caller
# surfaces as the ``*_error`` companion field.
#
# Empty strings are allowed (they represent "not yet set"). The validator is
# deliberately lenient on relative paths that resolve INSIDE the allowed
# base — operators routinely drop in ``runs/run-x/adapter`` style paths.


def _validate_ui_path(value: str) -> tuple[str, str]:
    """Validate a user-supplied path against the UI output dir.

    Returns a ``(cleaned_value, error_message)`` tuple. On success the error
    is the empty string; on failure ``cleaned_value`` is the empty string and
    the error carries a short operator-facing message. Empty input is a
    pass-through (no error, no value) so the input field can be cleared.
    """
    if not value or not value.strip():
        return "", ""
    candidate = value.strip()
    try:
        from .security import safe_path
        from .ui_security import get_ui_output_dir

        base = get_ui_output_dir()
        resolved = safe_path(candidate, allowed_base=base, allow_relative=True)
        return str(resolved), ""
    except Exception as exc:  # noqa: BLE001 — surface as operator-facing string
        return "", f"Invalid path: {exc}"


def _coerce_int(value: object) -> int | None:
    """Best-effort ``int`` cast; returns ``None`` if value can't be parsed.

    The Reflex setter signature is ``str | int | float`` depending on the
    input widget, so a single helper keeps the per-field setter terse.
    """
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            try:
                return int(float(s))
            except ValueError:
                return None
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _clamp_int(name: str, raw: object, lo: int, hi: int) -> tuple[int | None, str]:
    """Parse + clamp an integer. Returns ``(value, error)``.

    On parse failure ``value`` is ``None`` and the caller should keep the
    previous state. Out-of-range values are clamped silently INSIDE the
    range — the error string carries the operator-facing nudge so the UI
    can surface it via the ``*_error`` companion.
    """
    n = _coerce_int(raw)
    if n is None:
        return None, f"{name} must be an integer (got {raw!r})"
    if n < lo:
        return lo, f"{name} clamped to minimum {lo} (was {n})"
    if n > hi:
        return hi, f"{name} clamped to maximum {hi} (was {n})"
    return n, ""


def _clamp_float(
    name: str, raw: object, lo: float, hi: float
) -> tuple[float | None, str]:
    f = _coerce_float(raw)
    if f is None:
        return None, f"{name} must be a number (got {raw!r})"
    if f < lo:
        return lo, f"{name} clamped to minimum {lo:g} (was {f:g})"
    if f > hi:
        return hi, f"{name} clamped to maximum {hi:g} (was {f:g})"
    return f, ""


class AppState(rx.State):
    """Top-level state: theme toggle, active surface, current run_id.

    Cross-surface state lives here so the header / left nav / footer can read
    it without coupling to a specific page's state class.
    """

    theme: Theme = "dark"
    active_surface: ActiveSurface = "train"
    run_id: str = ""

    @rx.event
    def toggle_theme(self) -> None:
        """Flip dark/light. The TOKENS_CSS stylesheet keys off
        ``[data-theme="light"]`` so this re-themes the whole tree."""
        self.theme = "light" if self.theme == "dark" else "dark"

    @rx.event
    def set_active_surface(self, surface: str) -> None:
        """Left-nav click handler. The ``surface`` arg is a free string from
        the click event; clamp to the known set."""
        if surface in ("train", "multi-run", "export", "dataset"):
            self.active_surface = surface  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Shared training-config setter helpers (FRONTEND-A-005)
# ---------------------------------------------------------------------------
#
# Both TrainState and MultiRunState carry the same Model / Training-shape /
# LoRA config fields. Reflex's State metaclass only auto-registers event
# handlers that are DIRECTLY declared on the rx.State subclass (mixin
# inheritance is invisible to the framework's event-trigger scan, per
# 2026-05-22 verification), so we cannot share via a plain Python mixin.
# The fields and setters are therefore duplicated by design — but the
# logic each setter calls into is centralised in the helpers below so the
# behaviour stays in lockstep.
#
# When the Phase 3 Trainer hookup lands, both TrainState and MultiRunState
# will read from the same config snapshot helper, so an operator typing on
# one surface still sees the value carried to the other (the Trainer
# accepts a single config struct, regardless of which surface fired it).


def _apply_model(value: str) -> tuple[str, str]:
    """Validation logic for the HF model id / local model path setter."""
    if value and ("/" in value or "\\" in value) and Path(value).is_absolute():
        cleaned, err = _validate_ui_path(value)
        return cleaned, err
    return value, ""


def _apply_batch_size(value: object) -> tuple[str | None, str]:
    """``'auto'`` OR a positive int.

    Returns ``(stored_str_or_none, error)``. On parse failure the caller
    keeps the previous value and surfaces the error string.
    """
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto", ""
    n = _coerce_int(value)
    if n is None:
        return None, (
            f"Batch size must be 'auto' or a positive integer (got {value!r})"
        )
    if n < 1:
        return "1", "Batch size clamped to minimum 1"
    if n > 4096:
        return "4096", "Batch size clamped to maximum 4096"
    return str(n), ""


def _apply_target_modules(value: str) -> tuple[str | None, str]:
    """Comma-separated identifier list with strict character set + 32-item cap.

    Returns ``(canonical_string_or_none, error)``. ``None`` means parse
    failure — caller keeps the previous value.
    """
    if not value or not value.strip():
        return "", ""
    cleaned = value.strip()
    if not _TARGET_MODULES_RE.match(cleaned):
        return None, (
            "Target modules: only letters, digits, underscore, comma, "
            "whitespace allowed"
        )
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        return "", "Target modules cannot be empty"
    if len(parts) > 32:
        return None, "At most 32 target modules per LoRA"
    return ", ".join(parts), ""


def _apply_wandb_run_name(value: str) -> tuple[str | None, str]:
    if not value or not value.strip():
        return "", ""
    cleaned = value.strip()
    if len(cleaned) > 128:
        return None, "Run name too long (max 128 chars)"
    if not _WANDB_RUN_NAME_RE.match(cleaned):
        return None, "Run name: alphanumerics, dot, underscore, dash only"
    return cleaned, ""


class TrainState(rx.State):
    """Train surface state: config + live run progress.

    Config fields are duplicated with MultiRunState by design (Reflex's
    metaclass requires direct declaration); validation routes through the
    module-level ``_apply_*`` helpers so the two classes share one
    behaviour spec.
    """

    # ---- Configuration form ------------------------------------------------
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    model_error: str = ""
    dataset_path: str = ""
    dataset_path_error: str = ""
    steps: int = 100
    steps_error: str = ""
    batch_size: str = "auto"
    batch_size_error: str = ""
    learning_rate: float = 2e-4
    learning_rate_error: str = ""
    lora_r: int = 16
    lora_r_error: str = ""
    lora_alpha: int = 32
    lora_alpha_error: str = ""
    lora_dropout: float = 0.05
    lora_dropout_error: str = ""
    target_modules: str = "q_proj, k_proj, v_proj, o_proj"
    target_modules_error: str = ""
    quantization: Quantization = "4-bit"

    # ---- Advanced flags ----------------------------------------------------
    gpu_temp_threshold: int = 85
    gpu_temp_threshold_error: str = ""
    wandb_run_name: str = ""
    wandb_run_name_error: str = ""
    gradient_checkpointing: bool = True
    flash_attention: bool = True

    # ---- Live run progress -------------------------------------------------
    run_state: RunState = "idle"
    current_step: int = 0
    current_loss: float = 0.0
    loss_history: list[float] = []
    gpu_temp: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0

    # Event log — each entry is a dict with keys: t (timestamp str),
    # level (one of info/ok/warn/err/tx/hf), msg (str).
    events: list[dict] = []

    # ---- Setters (FRONTEND-A-002 + FRONTEND-B-002) -------------------------

    @rx.event
    def set_model(self, value: str) -> None:
        self.model, self.model_error = _apply_model(value)

    @rx.event
    def set_dataset_path(self, value: str) -> None:
        self.dataset_path, self.dataset_path_error = _validate_ui_path(value)

    @rx.event
    def set_steps(self, value: str | int) -> None:
        n, err = _clamp_int("Steps", value, _STEPS_MIN, _STEPS_MAX)
        if n is not None:
            self.steps = n
        self.steps_error = err

    @rx.event
    def set_batch_size(self, value: str) -> None:
        new, err = _apply_batch_size(value)
        if new is not None:
            self.batch_size = new
        self.batch_size_error = err

    @rx.event
    def set_learning_rate(self, value: str | float) -> None:
        f, err = _clamp_float("Learning rate", value, _LR_MIN, _LR_MAX)
        if f is not None:
            self.learning_rate = f
        self.learning_rate_error = err

    @rx.event
    def set_lora_r(self, value: str | int) -> None:
        n, err = _clamp_int("LoRA rank", value, _LORA_R_MIN, _LORA_R_MAX)
        if n is not None:
            self.lora_r = n
        self.lora_r_error = err

    @rx.event
    def set_lora_alpha(self, value: str | int) -> None:
        n, err = _clamp_int("LoRA alpha", value, _LORA_ALPHA_MIN, _LORA_ALPHA_MAX)
        if n is not None:
            self.lora_alpha = n
        self.lora_alpha_error = err

    @rx.event
    def set_lora_dropout(self, value: str | float) -> None:
        f, err = _clamp_float(
            "Dropout", value, _LORA_DROPOUT_MIN, _LORA_DROPOUT_MAX
        )
        if f is not None:
            self.lora_dropout = f
        self.lora_dropout_error = err

    @rx.event
    def set_target_modules(self, value: str) -> None:
        new, err = _apply_target_modules(value)
        if new is not None:
            self.target_modules = new
        self.target_modules_error = err

    @rx.event
    def set_quantization(self, value: str) -> None:
        if value in ("4-bit", "8-bit", "16-bit"):
            self.quantization = value  # type: ignore[assignment]

    @rx.event
    def set_gpu_temp_threshold(self, value: str | int) -> None:
        n, err = _clamp_int(
            "GPU temp threshold", value, _GPU_TEMP_MIN, _GPU_TEMP_MAX
        )
        if n is not None:
            self.gpu_temp_threshold = n
        self.gpu_temp_threshold_error = err

    @rx.event
    def set_wandb_run_name(self, value: str) -> None:
        new, err = _apply_wandb_run_name(value)
        if new is not None:
            self.wandb_run_name = new
        self.wandb_run_name_error = err

    @rx.event
    def set_gradient_checkpointing(self, value: bool) -> None:
        self.gradient_checkpointing = bool(value)

    @rx.event
    def set_flash_attention(self, value: bool) -> None:
        self.flash_attention = bool(value)

    # ---- Event handlers (stubs; backend hookup in Phase 3) -----------------

    @rx.event
    def start_training(self) -> None:
        """Stub handler for "Start training" button.

        Transitions to ``loading`` and writes a placeholder event so the
        skeleton's side rail visibly responds to the click. Phase 3 will
        replace this with a real ``Trainer.train(...)`` call dispatched to
        a background task.
        """
        self.run_state = "loading"
        self.events = [
            {"t": "00:00:00", "level": "info", "msg": "[stub] training start clicked"}
        ]

    @rx.event
    def stop_training(self) -> None:
        """Stub handler for "Stop / pause" button."""
        if self.run_state in ("active", "loading", "paused"):
            self.run_state = "idle"
            self.events = [
                *self.events,
                {"t": "00:00:00", "level": "info", "msg": "[stub] training stopped"},
            ]


class MultiRunState(rx.State):
    """Multi-Run surface state: config + num_runs, samples_per_run,
    merge_mode, replay_fraction.

    Config fields duplicate TrainState's by design (see TrainState docstring);
    setter logic routes through the same module-level helpers.
    """

    # ---- Configuration form (mirrors TrainState) ---------------------------
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    model_error: str = ""
    dataset_path: str = ""
    dataset_path_error: str = ""
    steps: int = 100
    steps_error: str = ""
    batch_size: str = "auto"
    batch_size_error: str = ""
    learning_rate: float = 2e-4
    learning_rate_error: str = ""
    lora_r: int = 16
    lora_r_error: str = ""
    lora_alpha: int = 32
    lora_alpha_error: str = ""
    lora_dropout: float = 0.05
    lora_dropout_error: str = ""
    target_modules: str = "q_proj, k_proj, v_proj, o_proj"
    target_modules_error: str = ""
    quantization: Quantization = "4-bit"

    # ---- Multi-Run specific ------------------------------------------------
    num_runs: int = 3
    num_runs_error: str = ""
    samples_per_run: int = 500
    samples_per_run_error: str = ""
    merge_mode: MergeMode = "slao"
    replay_fraction: float = 0.0
    replay_fraction_error: str = ""

    # ---- Live state --------------------------------------------------------
    run_state: RunState = "idle"
    current_run_index: int = 0
    runs: list[dict] = []  # per-run summary (loss, step, status)
    events: list[dict] = []

    # ---- Setters (shared logic with TrainState via _apply_* helpers) -------

    @rx.event
    def set_model(self, value: str) -> None:
        self.model, self.model_error = _apply_model(value)

    @rx.event
    def set_dataset_path(self, value: str) -> None:
        self.dataset_path, self.dataset_path_error = _validate_ui_path(value)

    @rx.event
    def set_steps(self, value: str | int) -> None:
        n, err = _clamp_int("Steps", value, _STEPS_MIN, _STEPS_MAX)
        if n is not None:
            self.steps = n
        self.steps_error = err

    @rx.event
    def set_batch_size(self, value: str) -> None:
        new, err = _apply_batch_size(value)
        if new is not None:
            self.batch_size = new
        self.batch_size_error = err

    @rx.event
    def set_learning_rate(self, value: str | float) -> None:
        f, err = _clamp_float("Learning rate", value, _LR_MIN, _LR_MAX)
        if f is not None:
            self.learning_rate = f
        self.learning_rate_error = err

    @rx.event
    def set_lora_r(self, value: str | int) -> None:
        n, err = _clamp_int("LoRA rank", value, _LORA_R_MIN, _LORA_R_MAX)
        if n is not None:
            self.lora_r = n
        self.lora_r_error = err

    @rx.event
    def set_lora_alpha(self, value: str | int) -> None:
        n, err = _clamp_int("LoRA alpha", value, _LORA_ALPHA_MIN, _LORA_ALPHA_MAX)
        if n is not None:
            self.lora_alpha = n
        self.lora_alpha_error = err

    @rx.event
    def set_lora_dropout(self, value: str | float) -> None:
        f, err = _clamp_float(
            "Dropout", value, _LORA_DROPOUT_MIN, _LORA_DROPOUT_MAX
        )
        if f is not None:
            self.lora_dropout = f
        self.lora_dropout_error = err

    @rx.event
    def set_target_modules(self, value: str) -> None:
        new, err = _apply_target_modules(value)
        if new is not None:
            self.target_modules = new
        self.target_modules_error = err

    @rx.event
    def set_quantization(self, value: str) -> None:
        if value in ("4-bit", "8-bit", "16-bit"):
            self.quantization = value  # type: ignore[assignment]

    @rx.event
    def set_num_runs(self, value: str | int) -> None:
        n, err = _clamp_int("Num runs", value, _NUM_RUNS_MIN, _NUM_RUNS_MAX)
        if n is not None:
            self.num_runs = n
        self.num_runs_error = err

    @rx.event
    def set_samples_per_run(self, value: str | int) -> None:
        n, err = _clamp_int(
            "Samples per run", value, _SAMPLES_PER_RUN_MIN, _SAMPLES_PER_RUN_MAX
        )
        if n is not None:
            self.samples_per_run = n
        self.samples_per_run_error = err

    @rx.event
    def set_merge_mode(self, value: str) -> None:
        if value in ("slao", "weighted", "ties"):
            self.merge_mode = value  # type: ignore[assignment]

    @rx.event
    def set_replay_fraction(self, value: str | float) -> None:
        f, err = _clamp_float("Replay fraction", value, 0.0, 1.0)
        if f is not None:
            self.replay_fraction = f
        self.replay_fraction_error = err

    @rx.event
    def start_multi_run(self) -> None:
        """Stub handler for "Start multi-run" button."""
        self.run_state = "loading"
        self.events = [
            {"t": "00:00:00", "level": "info", "msg": "[stub] multi-run start clicked"}
        ]


class ExportState(rx.State):
    """Export surface state: source model, format, quantization, ollama config."""

    source_model_path: str = ""
    source_model_path_error: str = ""
    format: ExportFormat = "lora"
    gguf_quant: GgufQuant = "q4_K_M"
    ollama_register: bool = False
    ollama_name: str = ""
    ollama_name_error: str = ""

    # ---- Validated path / name setters (FRONTEND-A-002) --------------------

    @rx.event
    def set_source_model_path(self, value: str) -> None:
        """Validate and set the adapter / model path."""
        cleaned, err = _validate_ui_path(value)
        self.source_model_path = cleaned
        self.source_model_path_error = err

    @rx.event
    def set_format(self, value: str) -> None:
        if value in ("lora", "merged", "gguf"):
            self.format = value  # type: ignore[assignment]

    @rx.event
    def set_gguf_quant(self, value: str) -> None:
        if value in ("q2_K", "q3_K_M", "q4_K_M", "q5_K_M", "q6_K", "q8_0"):
            self.gguf_quant = value  # type: ignore[assignment]

    @rx.event
    def set_ollama_register(self, value: bool) -> None:
        self.ollama_register = bool(value)

    @rx.event
    def set_ollama_name(self, value: str) -> None:
        """Validate the Ollama model name — alphanumeric / dash / underscore /
        colon (for tag) only; reject anything that smells like a path."""

        if not value:
            self.ollama_name = ""
            self.ollama_name_error = ""
            return
        cleaned = value.strip()
        # Ollama names: lowercase alnum + . _ - : / (the slash is for registry,
        # but we forbid backslash + .. + leading slash + null).
        if (
            ".." in cleaned
            or "\x00" in cleaned
            or "\\" in cleaned
            or cleaned.startswith("/")
            or not re.match(r"^[A-Za-z0-9._:/-]+$", cleaned)
        ):
            self.ollama_name = ""
            self.ollama_name_error = "Invalid Ollama model name"
            return
        self.ollama_name = cleaned
        self.ollama_name_error = ""

    # Live state.
    export_state: RunState = "idle"
    output_path: str = ""
    events: list[dict] = []

    @rx.event
    def start_export(self) -> None:
        """Stub handler for "Export" button."""
        self.export_state = "loading"
        self.events = [
            {"t": "00:00:00", "level": "info", "msg": "[stub] export start clicked"}
        ]


class DatasetState(rx.State):
    """Dataset surface state: upload, format detect, preview, dedup config."""

    uploaded_path: str = ""
    upload_error: str = ""
    upload_count: int = 0  # per-session cap
    detected_format: str = ""
    preview_records: list[dict] = []

    # FRONTEND-B-013: backend-computed basename so the UI never has to
    # split the full path on the client. The full path (with home prefix)
    # is kept for the Trainer hookup; only the basename is rendered.
    @rx.var
    def uploaded_basename(self) -> str:
        if not self.uploaded_path:
            return ""
        return Path(self.uploaded_path).name

    # Format hint — operator can override the auto-detect when it guesses wrong.
    format_hint: DatasetFormatHint = "auto"

    # Dedup + filter knobs.
    dedup_enabled: bool = True
    drop_empty: bool = True
    apply_curriculum: bool = False
    min_tokens: int = 0
    min_tokens_error: str = ""
    max_tokens: int = 2048
    max_tokens_error: str = ""

    record_count: int = 0
    dedup_hits: int = 0

    # Per-session upload cap. Reflex state is per-WebSocket-connection so this
    # is effectively per-tab; an unauthenticated abuser can still open many
    # tabs (cf. FRONTEND-A-001), but the cap caps each session's foot-gun.
    _MAX_UPLOADS_PER_SESSION: int = 5

    @rx.event
    async def handle_upload(self, files: list) -> None:  # type: ignore[type-arg]
        """Validate and persist uploaded dataset files (FRONTEND-A-003).

        The handler is wired to ``rx.upload``'s ``on_drop``. Each file goes
        through ``FileValidator`` (extension allowlist + size cap + magic-byte
        sniff when enabled) and ``sanitize_filename`` before being written
        inside ``get_ui_output_dir() / 'uploads'``. Failures populate
        ``upload_error`` rather than raising — the UI binds to it via
        ``rx.cond``.
        """
        from .ui_security import (
            ALLOWED_DATASET_EXTENSIONS,
            DEFAULT_SECURITY_CONFIG,
            FileValidator,
            get_ui_output_dir,
            sanitize_filename,
        )

        # FRONTEND-B-006: defense-in-depth against a malicious WS client that
        # sends multiple files in one ``on_drop`` payload despite ``multiple=
        # False`` on the rx.upload widget. The entry check below is fast-fail;
        # the per-iteration check inside the loop is the real cap enforcement.
        if not isinstance(files, list) or len(files) == 0:
            self.upload_error = "No files received"
            return
        if len(files) > 1:
            # Server contract is multiple=False; a payload with >1 file is a
            # WebSocket-direct bypass attempt. Reject the whole drop rather
            # than partially processing it.
            self.upload_error = (
                "Only one file per upload (multiple-file drops are rejected)"
            )
            return
        if self.upload_count >= self._MAX_UPLOADS_PER_SESSION:
            self.upload_error = (
                f"Per-session upload cap reached "
                f"({self._MAX_UPLOADS_PER_SESSION} files). "
                "Restart the page to upload more."
            )
            return

        try:
            base = get_ui_output_dir()
        except Exception as exc:  # noqa: BLE001
            self.upload_error = f"Output dir unavailable: {exc}"
            return

        upload_dir = base / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        validator = FileValidator(
            allowed_extensions=ALLOWED_DATASET_EXTENSIONS,
            max_size_mb=DEFAULT_SECURITY_CONFIG.max_upload_size_mb,
        )
        max_bytes = DEFAULT_SECURITY_CONFIG.max_upload_size_mb * 1024 * 1024

        for f in files:
            # FRONTEND-B-006: per-iteration cap check — if the multiple-file
            # guard above is ever weakened (e.g. for a future drag-multiple
            # feature) this is the floor that still holds.
            if self.upload_count >= self._MAX_UPLOADS_PER_SESSION:
                self.upload_error = (
                    f"Per-session upload cap reached "
                    f"({self._MAX_UPLOADS_PER_SESSION} files)."
                )
                return
            filename = getattr(f, "filename", None) or "unnamed"
            try:
                data = await f.read()
            except Exception as exc:  # noqa: BLE001
                self.upload_error = f"Could not read {filename}: {exc}"
                return

            if len(data) > max_bytes:
                self.upload_error = (
                    f"Rejected {filename}: exceeds "
                    f"{DEFAULT_SECURITY_CONFIG.max_upload_size_mb} MB cap."
                )
                return

            # Stage to a temp file so FileValidator can run its file-on-disk
            # checks (extension + size + magic-bytes when enabled).
            safe_name = sanitize_filename(filename)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(safe_name).suffix
            ) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)

            try:
                # Adapt to FileValidator's "object with .name" contract.
                class _FObj:
                    name = str(tmp_path)

                is_valid, msg, _ = validator.validate(_FObj(), purpose="upload")
                if not is_valid:
                    self.upload_error = f"Rejected {filename}: {msg}"
                    return
            finally:
                # We persist data ourselves under the allowed base; drop temp.
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

            target = upload_dir / safe_name
            target.write_bytes(data)
            self.uploaded_path = str(target)
            self.upload_count += 1

        self.upload_error = ""

    @rx.event
    def set_format_hint(self, value: str) -> None:
        if value in ("auto", "sharegpt", "alpaca", "openai", "jsonl"):
            self.format_hint = value  # type: ignore[assignment]

    @rx.event
    def set_dedup_enabled(self, value: bool) -> None:
        self.dedup_enabled = bool(value)

    @rx.event
    def set_drop_empty(self, value: bool) -> None:
        self.drop_empty = bool(value)

    @rx.event
    def set_apply_curriculum(self, value: bool) -> None:
        self.apply_curriculum = bool(value)

    @rx.event
    def set_min_tokens(self, value: str | int) -> None:
        n, err = _clamp_int("Min tokens", value, _TOKENS_MIN, _TOKENS_MAX)
        if n is not None:
            self.min_tokens = n
            # Re-clamp max if min would exceed it.
            if self.max_tokens < n:
                self.max_tokens = n
        self.min_tokens_error = err

    @rx.event
    def set_max_tokens(self, value: str | int) -> None:
        n, err = _clamp_int("Max tokens", value, _TOKENS_MIN, _TOKENS_MAX)
        if n is not None:
            self.max_tokens = n
        self.max_tokens_error = err

    @rx.event
    def detect_format_stub(self) -> None:
        """Stub handler — placeholder for the upload->detect flow."""
        if self.uploaded_path:
            self.detected_format = "alpaca"


__all__ = [
    "RunState",
    "Theme",
    "ActiveSurface",
    "ExportFormat",
    "Quantization",
    "MergeMode",
    "GgufQuant",
    "DatasetFormatHint",
    "AppState",
    "TrainState",
    "MultiRunState",
    "ExportState",
    "DatasetState",
]
