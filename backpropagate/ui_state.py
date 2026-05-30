"""
Backpropagate - Reflex state classes
=====================================

The ``rx.State`` subclasses that drive the five UI surfaces (Train, Multi-Run,
Export, Dataset, Runs). ``TrainState`` / ``MultiRunState`` / ``ExportState``
remain pre-Phase-3 stubs for the backend-write side (Start training is still a
placeholder until ``Trainer`` integration lands); ``DatasetState`` is real
(file upload + size-cap streaming + validator) as of v1.2.0; ``RunsState`` is
real (RunHistoryManager-backed read of on-disk run JSON) as of v1.2.0
(FRONTEND-F-RUN-HISTORY-PAGE).

The state classes are intentionally split (not one mega-class) so each
surface's WebSocket bundle stays small. Reflex coalesces the per-class state
into a single client connection automatically.

Config fields shared by ``TrainState`` and ``MultiRunState`` (Model, Training
shape, LoRA, Dataset) are duplicated by design: Reflex's State metaclass only
auto-registers event handlers DIRECTLY declared on the ``rx.State`` subclass
(plain Python mixin inheritance is invisible to the framework's event-trigger
scan, per 2026-05-22 verification). The setters route through module-level
``_apply_*`` helpers so the duplicated fields share one validation contract.
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


# ---------------------------------------------------------------------------
# Run-id validation helper — UI-A-003 fix (Wave A1 HIGH)
# ---------------------------------------------------------------------------
#
# Run IDs originate from two user-controlled surfaces: the dynamic route
# param ``rid`` (``/runs/[rid]``) and the ``diff_other_run_id`` text input.
# Both flow into ``subprocess`` argv for the diff-runs / replay shell-outs.
# An option-shaped value (e.g. ``--to=/etc/passwd`` or ``-o``) is parsed by
# the downstream argparse-based CLI as a FLAG, not a positional run id —
# letting a remote operator (under the documented ``--share + --auth`` flow)
# smuggle arbitrary flags into the spawned process.
#
# Real run IDs are UUID-hex / wandb-style slugs: ``[A-Za-z0-9_-]``. We pin a
# strict allowlist (1-64 chars, no leading ``-``) at the trust boundary so a
# malformed value is rejected with a clean error before it can reach argv.
# The leading char is constrained to ``[A-Za-z0-9_]`` (NOT ``-``) so the
# value can never be parsed as a CLI flag even before the ``--`` separator.
# (The ``--`` end-of-options separator inserted in each shell-out is the
# defense-in-depth second layer; this regex is the first.)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$")


def _validate_run_id(value: str) -> tuple[str, str]:
    """Validate a user-supplied run id against the strict allowlist.

    Returns a ``(cleaned_value, error_message)`` tuple mirroring
    ``_validate_ui_path``: on success the error is empty; on failure the
    value is empty and the error carries a short operator-facing message.
    Empty input is a pass-through (no error, no value).

    The allowlist (``^[A-Za-z0-9_-]{1,64}$``) rejects any leading ``-`` (so
    the value can never be parsed as a CLI flag), path separators, dots
    (no ``..`` traversal), whitespace, and shell metacharacters.
    """
    if not value or not value.strip():
        return "", ""
    candidate = value.strip()
    if not _RUN_ID_RE.match(candidate):
        return "", (
            "Invalid run id — expected 1-64 chars of letters, digits, "
            "'_' or '-' (no leading '-', no path separators)."
        )
    return candidate, ""


# ---------------------------------------------------------------------------
# Action-string redaction — V2-a fix (Wave A2 verifier gap)
# ---------------------------------------------------------------------------
#
# RunDetailState's in-process action handlers (delete_run / export_run /
# replay, rewired in Wave A1 from broken subprocess shell-outs) build
# operator-facing strings that embed RAW absolute paths — the sandbox
# ``history_dir`` / ``out_path`` (both under ``~/.backpropagate/ui-outputs``,
# which contains the OS home dir + username) and the bare ``{exc}`` repr of an
# OSError (``[Errno 2] ... '/home/<user>/...'``). Those strings are assigned to
# ``action_error`` / ``action_result``, which are PUBLIC (client-serialized)
# Reflex vars — under the documented ``--share + --auth`` flow they ship to the
# remote browser, leaking the operator's home dir + username (the exact FB-011
# class ``sanitize_error_for_user`` exists to prevent). Route every
# path-bearing action string through this helper before assignment so the home
# prefix is replaced with ``<redacted-path>``.
def _redact_action(text: str) -> str:
    """Redact absolute filesystem paths from an operator-facing action string.

    Thin wrapper over ``ui_security._redact_paths`` (the same redactor
    ``sanitize_error_for_user`` uses) so the in-process action handlers can
    scrub ``history_dir`` / ``out_path`` / raw ``OSError`` reprs out of the
    client-serialized ``action_error`` / ``action_result`` vars. Falls back to
    returning the text unchanged if the import fails (the redactor module is
    framework-agnostic and always importable, but be defensive — a redaction
    failure must never crash the handler).
    """
    try:
        from .ui_security import _redact_paths

        return _redact_paths(text)
    except Exception:  # noqa: BLE001 — redaction is best-effort, never fatal
        return text


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
        """Flip dark/light (deprecated path — kept for back-compat).

        FRONTEND-F-001 (Wave 5.5): the load-bearing theme toggle now lives
        on Reflex's built-in ``rx.color_mode`` + ``rx.toggle_color_mode``
        plumbing — the header button binds to those directly so the DOM
        actually mutates (Radix theme provider writes ``class="light"`` /
        ``class="dark"`` on the html root, which fires the ``.light`` /
        ``.light-theme`` selector in ``ui_theme.TOKENS_CSS``).

        This server-side ``AppState.theme`` field + handler stay for
        backward compatibility — external automation or future surfaces
        may still introspect/mutate it — but they no longer drive the
        visible theme. The v1.2 bug was that ONLY this server-side
        toggle existed, so the icon swapped but the page stayed dark.
        """
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

    # ---- Computed Vars (FRONTEND-6 Wave 6b) --------------------------------

    @rx.var
    def loss_chart_data(self) -> list[dict]:
        """Shape ``loss_history`` for ``rx.recharts.line_chart`` consumption.

        Returns ``[{"step": i, "loss": v}, ...]`` — the dict shape recharts
        needs. Computed-Var so the chart re-renders without an explicit
        event handler when the trainer pushes a new step into the history.
        """
        return [{"step": i, "loss": v} for i, v in enumerate(self.loss_history)]

    @rx.var
    def run_complete(self) -> bool:
        """True when the run has reached a terminal state (``done`` or ``error``).

        FRONTEND-10 (post-run "next steps" panel) reads this Var to know when
        to surface the post-run affordances. ``error`` counts as complete for
        UI purposes — the operator still wants the "export what you have /
        start another / view checkpoints" affordances after a failure.
        """
        return self.run_state in ("done", "error")

    # ---- Recovery-banner Vars (FRONTEND-A-004, v1.4 Wave 2) -----------------
    #
    # The Train page surfaces the MOST RECENT ``ok`` / ``warn`` event as a
    # ``BpRecoveryBanner``. Pre-fix the component existed but no page rendered
    # it (the docstring at pages/train.py:6 even claimed "Recovery banners
    # (when applicable)" but the body never wired one). The component takes
    # plain Python strings for the variant (color + icon are looked up at
    # component-build time, not via Vars), so the three variants are exposed
    # as three separate Vars and the page renders three conditional banners.
    #
    # "Most recent" means walking the events list in reverse and returning
    # the first ok / warn entry; if none, return an empty string and the
    # page's ``rx.cond`` keeps the banner unmounted.

    @rx.var
    def latest_recovery_ok_msg(self) -> str:
        """Message of the most recent ``ok``-level event, or empty.

        Drives the Train page's "good news" recovery banner — e.g. trainer
        successfully resumed from an OOM bisect, GPU temp dropped below
        threshold, checkpoint was written after a near-miss.
        """
        for ev in reversed(self.events):
            if isinstance(ev, dict) and ev.get("level") == "ok":
                return str(ev.get("msg") or "")
        return ""

    @rx.var
    def latest_recovery_warn_msg(self) -> str:
        """Message of the most recent ``warn``-level event, or empty.

        Drives the Train page's "heads-up" recovery banner — e.g. GPU temp
        approaching threshold, dataset row skipped, batch auto-shrunk for
        VRAM. Distinct from ``err``: warn is a recovered-or-recoverable
        condition; err is a hard failure (rendered via the structured
        error callout instead).
        """
        for ev in reversed(self.events):
            if isinstance(ev, dict) and ev.get("level") == "warn":
                return str(ev.get("msg") or "")
        return ""

    @rx.var
    def latest_recovery_info_msg(self) -> str:
        """Message of the most recent ``info``-level event, or empty.

        Drives the Train page's neutral-tint banner — e.g. "trainer started",
        "dataset loaded", "exporting adapter". This one fires for routine
        lifecycle events; the page renders it at lower visual weight than
        ok / warn (or the page may opt to hide it entirely — see the
        train.py wiring rationale).
        """
        for ev in reversed(self.events):
            if isinstance(ev, dict) and ev.get("level") == "info":
                return str(ev.get("msg") or "")
        return ""

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

        Transitions to ``loading`` and APPENDS a placeholder event so the
        skeleton's side rail visibly responds to the click without erasing
        prior log lines (FRONTEND-B-LOW-EVENTS-APPEND, Stage C humanization
        - the previous shape ``self.events = [...]`` overwrote any prior
        events, which was confusing during repeated stop/start cycles).
        Phase 3 will replace this with a real ``Trainer.train(...)`` call
        dispatched to a background task.
        """
        self.run_state = "loading"
        self.events = [
            *self.events,
            {"t": "00:00:00", "level": "info", "msg": "[stub] training start clicked"},
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
    """Export surface state: source model, format, quantization, ollama config.

    Wave 6b (FRONTEND-11): adds push_to_hub fields — backend support already
    exists in ``backpropagate.export.push_to_hub``; this state surfaces the
    inputs to the UI form.
    """

    source_model_path: str = ""
    source_model_path_error: str = ""
    format: ExportFormat = "lora"
    gguf_quant: GgufQuant = "q4_K_M"
    ollama_register: bool = False
    ollama_name: str = ""
    ollama_name_error: str = ""

    # ---- HuggingFace Hub push (FRONTEND-11 Wave 6b) ------------------------
    hub_enabled: bool = False
    hub_repo_id: str = ""
    hub_repo_id_error: str = ""
    hub_private: bool = True
    hub_branch: str = "main"
    hub_branch_error: str = ""
    # UI-A-001 (Wave A1 CRITICAL): the raw HF token is held in a BACKEND-only
    # var (``_``-prefix → Reflex never serializes it into the client WS state
    # bundle). Pre-fix ``hub_token`` was a public (base_vars) var bound
    # two-way to a type=password input, so the live write-scoped credential
    # round-tripped to the browser on every keystroke; the password mask was
    # visual only. The input is now write-only: the setter populates the
    # backend var but the field does NOT bind ``value=`` back (drop the
    # controlled-input round-trip). ``hub_token_set`` is a public BOOL mirror
    # so the form can show a "token set" affordance without echoing the
    # secret. ``hub_token_error`` stays public — it's an operator-facing
    # validation string, never the credential.
    _hub_token: str = ""  # nosec B105 — backend-only secret store, not a credential literal
    hub_token_set: bool = False
    hub_token_error: str = ""
    hub_status: str = ""  # "" / "pushing" / "done" / "error"
    hub_message: str = ""  # operator-facing status / error message

    # FRONTEND-F-004 (v1.4 Wave 6b features): surface the two CLI flags that
    # Wave 2 BRIDGE-A-004 added but the UI form was missing — ``--include-base``
    # (push merged model with base weights, not just the LoRA adapter) and
    # ``--token-file`` (read the HF token from a mode-0600 file instead of
    # the inline ``--token`` argument). The token-file path is mutually
    # exclusive with ``hub_token``; ``push_to_hub`` enforces the precedence.
    hub_include_base: bool = False
    hub_token_file_path: str = ""
    hub_token_file_path_error: str = ""

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

    # ---- HuggingFace Hub push setters + handler (FRONTEND-11) --------------

    # HF repo id is ``<owner>/<repo>`` — allow [A-Za-z0-9_-./]. Strict char
    # set rejects injection probes; the 200-char cap is HF's documented
    # upper limit (see huggingface_hub.utils.validate_repo_id).
    _HF_REPO_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
    _HF_BRANCH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")

    @rx.event
    def set_hub_enabled(self, value: bool) -> None:
        self.hub_enabled = bool(value)

    @rx.event
    def set_hub_repo_id(self, value: str) -> None:
        if not value or not value.strip():
            self.hub_repo_id = ""
            self.hub_repo_id_error = ""
            return
        cleaned = value.strip()
        if len(cleaned) > 200:
            self.hub_repo_id_error = "Repo id too long (max 200 chars)"
            return
        if "/" not in cleaned:
            self.hub_repo_id_error = "Repo id must be <owner>/<repo>"
            return
        if not self._HF_REPO_RE.match(cleaned):
            self.hub_repo_id_error = "Repo id: alnum / dot / dash / underscore / slash only"
            return
        self.hub_repo_id = cleaned
        self.hub_repo_id_error = ""

    @rx.event
    def set_hub_private(self, value: bool) -> None:
        self.hub_private = bool(value)

    @rx.event
    def set_hub_branch(self, value: str) -> None:
        if not value or not value.strip():
            self.hub_branch = "main"
            self.hub_branch_error = ""
            return
        cleaned = value.strip()
        if len(cleaned) > 100:
            self.hub_branch_error = "Branch name too long (max 100 chars)"
            return
        if not self._HF_BRANCH_RE.match(cleaned):
            self.hub_branch_error = "Branch: alnum / dot / dash / underscore / slash only"
            return
        self.hub_branch = cleaned
        self.hub_branch_error = ""

    @rx.event
    def set_hub_include_base(self, value: bool) -> None:
        """Toggle ``include_base`` — whether to push merged base weights too.

        FRONTEND-F-004: mirrors the ``--include-base`` CLI flag. Default
        (``False``) pushes the LoRA adapter only when the source directory
        contains adapter files; ``True`` uploads every file in the
        directory (including the base model if it's there).
        """
        self.hub_include_base = bool(value)

    @rx.event
    def set_hub_token_file_path(self, value: str) -> None:
        """Validate the HF token-file path and stage it for the push.

        FRONTEND-F-004: mirrors the ``--token-file`` CLI flag. The file is
        not read here — only validated for shape (length / no traversal /
        no nulls). ``push_to_hub`` reads the file at push time via the
        existing ``_read_hub_token_file`` helper so the file-mode check
        + POSIX-warning + empty-file error stay in one place.

        Mutual exclusion with ``hub_token`` is enforced at push time
        (the inline token wins the field-clear when both are set; the
        push handler raises a structured error if both reach it).
        """
        if not value or not value.strip():
            self.hub_token_file_path = ""  # nosec B105 — path sentinel, not a password
            self.hub_token_file_path_error = ""  # nosec B105 — error-message sentinel, not a password
            return
        cleaned, err = _validate_ui_path(value)
        self.hub_token_file_path = cleaned
        self.hub_token_file_path_error = err

    @rx.event
    def set_hub_token(self, value: str) -> None:
        """Set the HF API token (UI-A-001: write-only into a backend var).

        The input is write-only — the form does NOT bind ``value=`` back to
        this field, so the raw secret never round-trips to the client. The
        value lives ONLY in the backend var ``self._hub_token`` (never in
        base_vars / the WS bundle), is never logged, never serialized to run
        history, never echoed in error messages. ``_hub_token`` clears after
        a successful push to limit exposure. ``hub_token_set`` is a public
        bool mirror so the form can render a "token set" affordance without
        exposing the credential.
        """
        cleaned = (value or "").strip()
        if not cleaned:
            self._hub_token = ""  # nosec B105 — backend-var clear, not a credential literal
            self.hub_token_set = False
            self.hub_token_error = ""  # nosec B105 — error-message clear, not a credential literal
            return
        # HF tokens are ``hf_<40 base62 chars>``; we don't pin the exact
        # prefix because operators may use org-scoped tokens with a
        # different prefix. Sanity-check the length is in the expected
        # 30-100 char range so we catch "I pasted my username by accident".
        if len(cleaned) < 20 or len(cleaned) > 200:
            self.hub_token_error = "Token doesn't look like an HF token (20-200 chars expected)"  # nosec B105 — operator-facing validation message, not a credential
            self.hub_token_set = False
            return
        self._hub_token = cleaned  # nosec B105 — backend-only secret store
        self.hub_token_set = True
        self.hub_token_error = ""  # nosec B105 — error-message clear, not a credential literal

    @rx.event
    def push_to_hub(self) -> None:
        """Push the trained adapter / merged model to a HuggingFace Hub repo.

        Delegates to ``backpropagate.export.push_to_hub`` (the established
        backend API). Failures surface via ``self.hub_message``; success
        clears the token so it doesn't sit in the state for the lifetime of
        the WS session.

        Pre-flight validation:
        - source_model_path must be set
        - hub_repo_id + hub_token must be set
        - all field-level errors must be clear

        The handler runs synchronously; the operator sees ``hub_status =
        "pushing"`` for the duration. v1.4 should move this to a background
        task with an SSE progress stream.
        """
        if self.source_model_path == "":
            self.hub_status = "error"
            self.hub_message = "Set a source adapter / model path before pushing."
            return
        if not self.hub_repo_id or self.hub_repo_id_error:
            self.hub_status = "error"
            self.hub_message = "Set a valid HuggingFace repo id (<owner>/<repo>)."
            return
        # FRONTEND-F-004: mutual-exclusion + at-least-one check on the two
        # token surfaces. Mirrors the CLI's `--token` vs `--token-file`
        # contract in cmd_push (cli.py ~3411).
        inline_token_set = bool(self._hub_token) and not self.hub_token_error
        token_file_set = (
            bool(self.hub_token_file_path) and not self.hub_token_file_path_error
        )
        if inline_token_set and token_file_set:
            self.hub_status = "error"
            self.hub_message = (
                "Token and Token-file are mutually exclusive — clear one "
                "before pushing. The token-file path is the safer floor "
                "(mode 0600, not visible to spawned children)."
            )
            return
        if not inline_token_set and not token_file_set:
            self.hub_status = "error"
            self.hub_message = (
                "Set a valid HuggingFace API token (inline) or a token-file "
                "path before pushing."
            )
            return
        if self.hub_branch_error:
            self.hub_status = "error"
            self.hub_message = "Fix the branch field error before pushing."
            return
        if self.hub_token_file_path_error:
            self.hub_status = "error"
            self.hub_message = "Fix the Token-file path error before pushing."
            return

        self.hub_status = "pushing"
        self.hub_message = (
            f"Pushing {self.source_model_path} to "
            f"{self.hub_repo_id}@{self.hub_branch or 'main'}…"
        )
        try:
            from .export import push_to_hub as _push

            # FRONTEND-F-004: resolve the token at push time. The
            # token-file path is read via the shared CLI helper so the
            # mode-0600 warning + empty-file error live in one spot.
            resolved_token: str
            if token_file_set:
                from .cli import _read_hub_token_file

                resolved_token = _read_hub_token_file(
                    self.hub_token_file_path,
                    flag_name="--token-file (UI)",
                )
            else:
                resolved_token = self._hub_token

            _push(
                local_path=self.source_model_path,
                repo_id=self.hub_repo_id,
                token=resolved_token,
                private=bool(self.hub_private),
                revision=(self.hub_branch or "main"),
                include_base=bool(self.hub_include_base),
            )
            self.hub_status = "done"
            self.hub_message = (
                f"Pushed to https://huggingface.co/{self.hub_repo_id} "
                f"on branch {self.hub_branch or 'main'}."
            )
            # Clear token after successful push (UI-A-001: backend var).
            self._hub_token = ""  # nosec B105 — token wipe after push, not a credential literal
            self.hub_token_set = False
            # FRONTEND-F-004: the token-file PATH itself is not a credential
            # — it's a reference to a file the operator manages outside
            # the UI session. Leaving it in state is intentional so a
            # second push (e.g. after a transient HF outage) doesn't
            # require re-typing the path.
        except Exception as exc:  # noqa: BLE001 — operator-facing string
            # Sanitize the error so HF token / operator paths don't leak.
            try:
                from .ui_security import sanitize_error_for_user

                message, suggestion = sanitize_error_for_user(
                    exc, operation="pushing to HuggingFace Hub"
                )
                self.hub_status = "error"
                self.hub_message = (
                    message + (f" Try: {suggestion}" if suggestion else "")
                )
            except Exception:  # noqa: BLE001
                # Last-resort fallback — preserve the operator-facing
                # message but trim to 200 chars so we don't spill a giant
                # traceback into the WS bundle. V2-a (sibling): redact absolute
                # paths from the raw exception repr — ``hub_message`` is a
                # public (client-serialized) var, so an HF exception embedding
                # the operator's home dir would otherwise ship to the browser.
                self.hub_status = "error"
                self.hub_message = _redact_action(
                    f"Push failed: {type(exc).__name__}: {str(exc)[:200]}"
                )

    @rx.event
    def clear_hub_status(self) -> None:
        """Dismiss the HF push status banner."""
        self.hub_status = ""
        self.hub_message = ""


class DatasetState(rx.State):
    """Dataset surface state: upload, format detect, preview, dedup config."""

    # UI-A-002 (Wave A2): the full upload path embeds the operator's home dir
    # + username and was previously a PUBLIC (client-serialized) Reflex var —
    # so the home prefix shipped in the WS bundle on every state delta even
    # though FRONTEND-B-013 only RENDERED the basename. The full path is now
    # held in a backend-only ('_'-prefixed) var that Reflex never serializes;
    # the client sees only the basename via the ``uploaded_basename`` computed
    # var. The full path stays available server-side for the Trainer hookup.
    _uploaded_path: str = ""
    upload_error: str = ""
    upload_count: int = 0  # per-session cap
    detected_format: str = ""
    preview_records: list[dict] = []

    # FRONTEND-B-013 / UI-A-002: backend-computed basename so the UI never has
    # to split the full path on the client AND the full path (with home
    # prefix) never enters the serialized state bundle. Only the basename is
    # rendered.
    @rx.var
    def uploaded_basename(self) -> str:
        if not self._uploaded_path:
            return ""
        return Path(self._uploaded_path).name

    @rx.var
    def has_upload(self) -> bool:
        """Client-safe truthiness for the 'Uploaded: …' chrome.

        The template can't test the backend-only ``_uploaded_path`` (it isn't
        serialized), so expose a boolean computed var for the ``rx.cond``.
        """
        return bool(self._uploaded_path)

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
    # FRONTEND-F-005 (v1.4 Wave 6b features): plumb the Stats-grid value
    # that dataset.py was hardcoding as '—'. The figure comes from
    # ``backpropagate.datasets.get_dataset_stats`` during the upload
    # handler — same computation the CLI emits, just thread it to the UI
    # state so the grid doesn't drift from the backend truth.
    avg_tokens: int = 0

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
            # FRONTEND-B-007 (Stage C humanization): route the raw OSError /
            # BackpropagateError through sanitize_error_for_user so the UI
            # banner cannot leak operator paths (FB-011 invariant).
            from .ui_security import sanitize_error_for_user

            message, suggestion = sanitize_error_for_user(
                exc, operation="resolving the UI output directory"
            )
            self.upload_error = (
                message + (f" Try: {suggestion}" if suggestion else "")
            )
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
            # FRONTEND-B-004 (Stage C humanization): stream-read in fixed-size
            # chunks with a running size counter so the cap is enforced
            # incrementally rather than after a full in-memory buffer. Peak
            # memory per upload is bounded at max_bytes + _CHUNK regardless of
            # what the client sends; a 10 GB rogue payload claiming to be a
            # .jsonl is rejected after the first chunk over the cap rather
            # than buffering the whole 10 GB.
            #
            # The fallback path covers reader objects that lack a chunked
            # async read (the rx.upload framing exposes ``await f.read()`` as
            # an all-bytes call; if the underlying reader is something else,
            # we still bail safely).
            _CHUNK = 1 << 20  # 1 MB per chunk
            chunks: list[bytes] = []
            size = 0
            try:
                # Probe for chunked-read support. ``read(n)`` returning bytes
                # is the asyncio.StreamReader contract; rx.upload's wrapper
                # supports it as of Reflex 0.9.x.
                while True:
                    chunk = await f.read(_CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        # Drop already-buffered chunks; we never persist a
                        # rejected upload.
                        chunks.clear()
                        self.upload_error = (
                            f"Rejected {filename}: exceeds "
                            f"{DEFAULT_SECURITY_CONFIG.max_upload_size_mb} MB "
                            "cap (aborted mid-stream). Trim the file or "
                            "increase BACKPROPAGATE_UI__MAX_UPLOAD_SIZE_MB."
                        )
                        return
                    chunks.append(chunk)
            except TypeError:
                # Reader doesn't honor a chunk-size argument (e.g. a
                # one-shot bytes object surfaced as a fake reader); fall
                # back to a single read and the post-buffer size check.
                try:
                    data = await f.read()
                except Exception as exc:  # noqa: BLE001
                    from .ui_security import sanitize_error_for_user

                    message, suggestion = sanitize_error_for_user(
                        exc, operation=f"reading upload '{filename}'"
                    )
                    self.upload_error = (
                        message
                        + (f" Try: {suggestion}" if suggestion else "")
                    )
                    return
                if len(data) > max_bytes:
                    self.upload_error = (
                        f"Rejected {filename}: exceeds "
                        f"{DEFAULT_SECURITY_CONFIG.max_upload_size_mb} MB cap. "
                        "Trim the file or increase "
                        "BACKPROPAGATE_UI__MAX_UPLOAD_SIZE_MB."
                    )
                    return
                chunks = [data]
            except Exception as exc:  # noqa: BLE001
                # FRONTEND-B-007 (Stage C humanization): never echo raw OSError
                # messages into the UI banner — operator paths leak via the
                # exception repr.
                from .ui_security import sanitize_error_for_user

                message, suggestion = sanitize_error_for_user(
                    exc, operation=f"reading upload '{filename}'"
                )
                self.upload_error = (
                    message + (f" Try: {suggestion}" if suggestion else "")
                )
                return

            data = b"".join(chunks)

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
            # UI-A-002: store the full path in the backend-only var; the
            # client receives only the basename via ``uploaded_basename``.
            self._uploaded_path = str(target)
            self.upload_count += 1

            # FRONTEND-F-005 (v1.4 Wave 6b features): compute dataset stats
            # via the canonical backend API + thread them into state so the
            # Stats grid (record_count / avg_tokens / dedup_hits) and the
            # Format badge bind to live values rather than hardcoded '—'
            # placeholders. The computation matches what
            # `backprop validate-dataset` emits, so UI + CLI stay in
            # lockstep. Failure here is non-fatal — the file is already
            # persisted; we just leave the Stats grid empty and surface a
            # one-line note inside the same upload_error channel.
            try:
                import json as _json

                from .datasets import (
                    DatasetFormat,
                    _detect_format_from_file,
                    get_dataset_stats,
                )

                samples: list[dict | str] = []
                suffix = target.suffix.lower()
                if suffix == ".jsonl":
                    with target.open(encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                try:
                                    samples.append(_json.loads(line))
                                except _json.JSONDecodeError:
                                    # Skip malformed lines in stats —
                                    # the strict validate-dataset
                                    # surface catches these.
                                    continue
                elif suffix == ".json":
                    with target.open(encoding="utf-8") as fh:
                        parsed = _json.load(fh)
                        if isinstance(parsed, list):
                            samples = parsed
                        else:
                            samples = [parsed]
                # Other extensions: leave samples empty → stats render 0s.

                detected = _detect_format_from_file(target)
                stats = get_dataset_stats(samples, format_type=detected)
                self.record_count = int(stats.total_samples)
                # avg_tokens_per_sample is float; round to int for the
                # 4-size big-number cell. Sub-token resolution isn't
                # operator-useful at this glance-level.
                self.avg_tokens = int(round(stats.avg_tokens_per_sample))
                if detected != DatasetFormat.UNKNOWN:
                    # Map enum to a short human badge — the Format group
                    # already binds ``detected_format`` and is None-safe.
                    self.detected_format = str(detected.value).capitalize()
            except Exception:  # noqa: BLE001  # nosec B110 — stats are advisory, never block upload
                # Leave the grid empty rather than raising. The upload
                # itself succeeded; a stats failure shouldn't surface as
                # an upload error.
                pass

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
        if self._uploaded_path:
            self.detected_format = "alpaca"


# ---------------------------------------------------------------------------
# RunsState — backs the /runs page (FRONTEND-F-RUN-HISTORY-PAGE, Wave 6)
# ---------------------------------------------------------------------------
#
# Loads the recent training-run history via the CLI's RunHistoryManager so the
# UI shows the same data ``backprop list-runs`` shows. The implementation
# imports RunHistoryManager directly rather than shelling out to the CLI —
# subprocess-shelling from a Reflex state handler would block the WS event
# loop and the CLI prints decorated text, not the clean dicts the UI needs.
#
# Drill-down to a per-run page is INTENTIONALLY OUT OF SCOPE for v1.2.0 (the
# user narrowed the brief). Each table row is a read-only summary; v1.3 adds
# a /runs/<id> route that mirrors ``backprop show-run``.


class RunsState(rx.State):
    """Run-history surface state — populates the /runs page table."""

    runs: list[dict] = []
    loading: bool = False
    error: str = ""
    status_filter: str = ""  # "" / running / completed / failed
    output_dir_override: str = ""
    last_loaded_at: str = ""

    # Hard cap on rows rendered at once. The CLI defaults to 50; the table can
    # comfortably render this many without pagination. v1.3 will add a
    # "Load more" affordance + per-status filter pills.
    _DEFAULT_LIMIT: int = 50

    @rx.event
    def load_runs(self) -> None:
        """Populate ``self.runs`` from the on-disk run history.

        The default output directory is ``~/.backpropagate/ui-outputs`` (the
        same directory the UI writes adapters/exports into). Operators who
        train from the CLI to a different ``--output`` directory can set the
        ``output_dir_override`` field on this state before calling
        ``load_runs``; the UI's settings surface will wire that in v1.3.
        """
        from datetime import datetime, timezone
        from pathlib import Path as _Path

        self.loading = True
        self.error = ""
        try:
            # Resolve the history directory. Use the override if set, otherwise
            # fall back to the UI's own output dir (the default training sink).
            if self.output_dir_override.strip():
                # UI-A-005 (Wave A2): ``output_dir_override`` is a PUBLIC
                # (client-serialized) Reflex var. ``set_output_dir_override``
                # validates via ``_validate_ui_path`` — but a malicious WS
                # client can write the public var directly, bypassing the
                # setter. Re-validate the forbidden-base guard HERE, at the
                # read sink, so a value pointing at a system / credential dir
                # (``/etc``, ``~/.ssh``, ``C:\\Windows`` …) is refused even if
                # it never passed through the setter. This mirrors the env-var
                # guard ``get_ui_output_dir()`` applies to its override.
                history_dir = _Path(self.output_dir_override).expanduser()
                try:
                    from .ui_security import _is_forbidden_output_base

                    if _is_forbidden_output_base(history_dir):
                        self.runs = []
                        self.error = _redact_action(
                            "Refusing to read run history from a system or "
                            f"credential directory: {history_dir.resolve()}. "
                            "Point the override at a non-system directory."
                        )
                        return
                except Exception:  # noqa: BLE001 — guard import/resolve must
                    # fail closed: if we can't validate, refuse the override
                    # and fall back to the sandbox default rather than reading
                    # an unvalidated operator-supplied path.
                    try:
                        from .ui_security import get_ui_output_dir

                        history_dir = get_ui_output_dir()
                    except Exception:
                        history_dir = _Path.home() / ".backpropagate" / "ui-outputs"
            else:
                try:
                    from .ui_security import get_ui_output_dir

                    history_dir = get_ui_output_dir()
                except Exception:
                    # Final fallback: the documented default.
                    history_dir = _Path.home() / ".backpropagate" / "ui-outputs"

            if not history_dir.exists():
                self.runs = []
                # UI-A-002 (sibling): ``error`` is a public RunsState var;
                # history_dir embeds the home dir + username. Redact.
                self.error = _redact_action(
                    f"No run history at {history_dir}. Train a model from the "
                    "UI or CLI; runs will appear here automatically."
                )
                return

            try:
                from .checkpoints import RunHistoryManager
            except ImportError as exc:
                self.error = _redact_action(f"checkpoints module unavailable: {exc}")
                self.runs = []
                return

            manager = RunHistoryManager(str(history_dir))
            status = self.status_filter.strip() or None
            try:
                rows = manager.list_runs(status=status, limit=self._DEFAULT_LIMIT)
            except ValueError as exc:
                # ValueError is operator-actionable (bad filter value); the
                # exception message itself is shaped for display.
                self.error = f"Invalid filter: {exc}"
                self.runs = []
                return
            except Exception as exc:  # noqa: BLE001 — surface as operator string
                # FRONTEND-B-007 (Stage C humanization): route through
                # sanitize_error_for_user so raw OSError / JSONDecodeError
                # messages (which embed filesystem paths) don't leak into the
                # UI banner. The full traceback is still logged server-side
                # via the caller's logger.exception (FB-011 invariant).
                from .ui_security import sanitize_error_for_user

                message, suggestion = sanitize_error_for_user(
                    exc, operation="loading run history"
                )
                self.error = (
                    message + (f" Try: {suggestion}" if suggestion else "")
                )
                self.runs = []
                return

            # Normalize to a small, JSON-serializable shape for Reflex's WS
            # bundle. The CLI emits dicts already; we just trim fields the
            # table doesn't use so the bundle stays small. We also pre-format
            # the run_id to its short 8-char form to avoid an f-string in
            # the template (Reflex template f-strings get awkward fast).
            trimmed: list[dict] = []
            for raw in rows:
                run_id = str(raw.get("run_id") or "")
                short_id = run_id[:8] if run_id else "-"
                started = str(raw.get("started_at") or "-")
                duration = raw.get("duration_seconds")
                if duration is None:
                    duration_str = "-"
                else:
                    try:
                        duration_str = f"{float(duration):.0f}s"
                    except (TypeError, ValueError):
                        duration_str = "-"
                final_loss = raw.get("final_loss")
                if final_loss is None:
                    final_loss_str = "-"
                else:
                    try:
                        final_loss_str = f"{float(final_loss):.4f}"
                    except (TypeError, ValueError):
                        final_loss_str = "-"
                trimmed.append({
                    "run_id": run_id,
                    "run_id_short": short_id,
                    "started_at": started,
                    "model": str(raw.get("model") or "-"),
                    "dataset": str(raw.get("dataset") or "-"),
                    "status": str(raw.get("status") or "-"),
                    "duration": duration_str,
                    "final_loss": final_loss_str,
                })
            self.runs = trimmed
            self.last_loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        finally:
            self.loading = False

    # Canonical status set. Mirrors the values RunHistoryManager.list_runs
    # accepts; the dropdown in pages/runs.py renders the same set. Update
    # both surfaces together when adding a new status.
    _STATUS_FILTER_VALUES: tuple[str, ...] = (
        "",
        "running",
        "completed",
        "failed",
        "interrupted",
    )

    @rx.event
    def set_status_filter(self, value: str) -> None:
        """Update the status filter and reload.

        Unknown values are silently discarded (the previous filter persists)
        but logged at WARNING so the silent-drop is observable in operator
        logs - FRONTEND-B-014 (Stage C humanization). A future status added
        in RunHistoryManager but missing from this list would otherwise
        present as 'filter does nothing' with no breadcrumb.

        ``load_runs`` is only invoked when the value is accepted - the
        previous code unconditionally reloaded even on rejection, which
        produced a confusing double-trigger of the table.
        """
        if value in self._STATUS_FILTER_VALUES:
            self.status_filter = value
            self.load_runs()
            return
        import logging

        logging.getLogger(__name__).warning(
            "RunsState: status filter received unknown value %r "
            "(keeping previous %r). Allowed values: %s",
            value,
            self.status_filter,
            ", ".join(repr(v) for v in self._STATUS_FILTER_VALUES),
        )

    @rx.event
    def set_output_dir_override(self, value: str) -> None:
        """Operator-supplied output directory (validated lightly)."""
        cleaned, err = _validate_ui_path(value)
        if err:
            self.error = err
            return
        self.output_dir_override = cleaned

    @rx.event
    def clear_error(self) -> None:
        """Dismiss the error banner."""
        self.error = ""


# ---------------------------------------------------------------------------
# AuthBadgeState - backs the footer auth-badge UI
# (FRONTEND-F-FOOTER-AUTH-BADGE, Stage C humanization).
# ---------------------------------------------------------------------------
#
# Reads ``ui_security.get_auth_badge_context()`` at first access and exposes
# the 6 fields the footer chip needs. The state is server-side only - the
# fields are populated from env vars that the CLI exports BEFORE spawning the
# Reflex subprocess, so the values are stable for the lifetime of the UI
# process. No event handlers mutate the fields.
#
# Critically: this class READS the auth posture; it does NOT participate in
# auth enforcement. The GHSA-f65r-h4g3-3h9h contracts (pre-accept WS cookie
# validation, 4-mode resolution, constant-time compares, Host/Origin
# allowlists) remain the exclusive responsibility of
# ``ui_app/auth.py::basic_auth_transformer``.


class AuthBadgeState(rx.State):
    """State backing the footer auth-mode badge.

    All 6 fields are populated at class-init from the env-var surface the
    middleware also consumes (``ui_security.get_auth_badge_context``). They
    are read-only from the operator's perspective: the badge reflects the
    posture the CLI established at launch.
    """

    mode_key: str = ""
    mode_color: str = "green"
    mode_text: str = ""
    hover_text: str = ""
    bind_host: str = ""
    bind_port: str = ""
    reachable_from: str = ""
    # FRONTEND-A-002 (v1.4 Wave 2): wire ``ctx.auth_user`` end-to-end.
    # Pre-fix the field was populated on the context dataclass but discarded
    # in ``refresh()``; the auth-mode label said "Basic" but the operator
    # could not see WHICH credential pair was active from the badge alone
    # (had to hover for the tooltip text). Surfacing the username as a
    # separate chip mirrors the bind-host chip pattern from Wave 5.5 and
    # closes the producer-without-consumer dead-state.
    auth_user: str = ""

    @rx.event
    def refresh(self) -> None:
        """Recompute the badge state from the current env.

        Mounted on the chrome's footer ``on_mount`` so a refreshed env var
        (rare - the CLI exports them once at launch) is reflected without a
        process restart. The cost is one ``os.environ`` snapshot per page
        load, which is dwarfed by every other request the Reflex backend
        handles.

        FRONTEND-B-006 (v1.4 Wave 4 Stage C humanization): the footer renders
        on EVERY route, so ``on_mount=AuthBadgeState.refresh`` previously
        fired on every page navigation. The env vars are stable for the
        process lifetime (the CLI exports them once at launch), so re-
        snapshotting on every nav is wasted server work + a small WS round-
        trip that flickers the chip on slow connections. We now early-exit
        when ``mode_key`` is already populated, making the badge refresh
        once-per-WS-session in steady state. The CLI is free to call
        ``refresh()`` explicitly if it rotates env vars mid-process; the
        early-exit only suppresses the per-nav repeat.
        """
        if self.mode_key:
            # Badge state was populated on the first footer mount of this
            # WS session; the env-var surface is stable so the second mount
            # would produce identical state. Skip the snapshot.
            return

        from .ui_security import get_auth_badge_context

        ctx = get_auth_badge_context()
        self.mode_key = ctx.mode_key
        self.mode_color = ctx.mode_color
        self.mode_text = ctx.mode_text
        self.hover_text = ctx.hover_text
        self.bind_host = ctx.bind_host
        self.bind_port = ctx.bind_port
        self.reachable_from = ctx.reachable_from
        # FRONTEND-A-002 (v1.4 Wave 2): mirror ``ctx.auth_user`` into state so
        # the footer badge can render it as a visible "@username" suffix on
        # the three Basic-auth modes (basic_local / basic_shared /
        # basic_network). The non-Basic modes (no_auth_local / token_local /
        # insecure) populate this with the empty string and the badge
        # component skips the chip via ``rx.cond``.
        self.auth_user = ctx.auth_user


# ---------------------------------------------------------------------------
# RunDetailState — backs /runs/[run_id] (Wave 6b drill-down)
# ---------------------------------------------------------------------------
#
# Wave 6 shipped the read-only run list; Wave 6b adds the drill-down per
# V1_3_BRIEF P1. The state mirrors what ``backprop show-run`` would emit
# (metadata header + hyperparameter dump + training metrics + checkpoint
# list + log tail) using the existing ``RunHistoryManager.get_run`` API
# (which supports partial-prefix matching for operator convenience).
#
# The four action buttons (Diff, Replay, Delete, Export) DO NOT modify
# state from inside this class — they shell out to the bridge subcommands
# via the action handlers, then re-load on completion. The bridge owns the
# CLI surfaces; this UI surface just renders the output.


class RunDetailState(rx.State):
    """Per-run drill-down state.

    The active run id is read from the dynamic route parameter inside
    ``load_run`` (Reflex 0.9's ``self.router.page.params.get("run_id")``
    pattern). We cannot name the state field ``run_id`` because Reflex
    refuses to bind a dynamic route arg that shadows an existing state
    var (DynamicRouteArgShadowsStateVarError); the state field is named
    ``current_run_id`` and the route arg writes to it on mount.

    The remaining fields are filled by ``load_run`` on mount.
    """

    current_run_id: str = ""
    loading: bool = False
    error: str = ""
    not_found: bool = False
    # FRONTEND-B-001 (v1.4 Wave 3.5): a successful ``delete_run`` shell-out
    # leaves the page showing data for a run that no longer exists on disk.
    # Pre-fix the handler set ``not_found = True`` to suppress the stale
    # body — but that latched the operator into the "Run not found" chrome
    # (designed for unknown-id navigations), which misframed a successful
    # operation as failure and stranded the success message in
    # ``action_result``. ``was_deleted`` is a separate, deletion-specific
    # surface so the template can render a "Run deleted." confirmation
    # with a "Back to runs list" affordance instead of the not-found
    # chrome. The template branches on ``was_deleted`` BEFORE
    # ``not_found`` so a successful delete always wins.
    was_deleted: bool = False

    # Header fields
    status: str = "-"
    model: str = "-"
    dataset: str = "-"
    started_at: str = "-"
    completed_at: str = "-"
    duration: str = "-"
    final_loss: str = "-"
    # UI-A-002 (Wave A2): the checkpoint path embeds the operator's home dir +
    # username. Held in a backend-only var; the client sees only the redacted
    # form via the ``checkpoint_path_display`` computed var (home prefix
    # replaced with ``<redacted-path>`` so the operator still sees the
    # run-relative tail but not their username).
    _checkpoint_path: str = "-"

    # Hyperparameter table — list of {key, value} dicts so Reflex's foreach
    # can render them as table rows without on-template f-strings.
    hyperparameters: list[dict] = []

    # Metrics — read from training_metrics.jsonl when present. v1.4 ships
    # with loss_history only. The multi-line metrics view (V1_4_BRIEF item
    # 10: lr + grad_norm + val_loss as additional series) was DEFERRED to
    # v1.5 per advisor lock 2026-05-25 (Wave 5 feature audit, decision 5)
    # because the upstream data pipeline is dead: trainer.py's log-history
    # extraction reads only ``log.get('loss')`` from HF Trainer.state.log_
    # history, dropping the ``grad_norm`` / ``learning_rate`` keys HF
    # populates per step; checkpoints.py's manifest schema has no parallel
    # ``grad_norm_history`` / ``lr_history`` / ``val_loss_history`` fields.
    # v1.5 will land the full cohesive slice in one wave — trainer
    # extraction + schema bump + RunDetailState fields + BpLossChart
    # multi-line + dual-axis y-scale + per-series toggle — rather than
    # ship the data plumbing as a banner-documenting-no-op intermediate
    # in v1.4 (see [[no-banner-documenting-no-op]]).
    loss_history: list[float] = []

    # Checkpoint list — {name, size_mb, timestamp} per entry.
    checkpoints: list[dict] = []

    # Log tail — last 200 lines of training.log when present (trimmed for
    # WS bundle size). UI-A-002: each line is run through ``_redact_paths``
    # before assignment in ``load_run`` so absolute paths a training log may
    # contain (the checkpoint dir, the HF cache, tempdirs) don't ship the
    # operator's home dir + username to the client.
    log_lines: list[str] = []

    # UI-A-002 (Wave A2): client-facing, redacted form of the checkpoint
    # path. The full path lives in the backend-only ``_checkpoint_path``;
    # this computed var replaces the home prefix with ``<redacted-path>`` so
    # the operator still sees the run-relative tail without leaking their
    # username into the WS bundle / screenshots.
    @rx.var
    def checkpoint_path_display(self) -> str:
        if not self._checkpoint_path or self._checkpoint_path == "-":
            return "-"
        return _redact_action(self._checkpoint_path)

    # Action panel — last shell-out result (for the operator-facing toast).
    action_result: str = ""
    action_error: str = ""
    # FRONTEND-B-014-EXTENDED (v1.4 Wave 4 Stage C humanization): action-in-
    # flight state for the diff / replay / delete / export shell-outs. The
    # handlers run synchronously and can block up to 30s (diff/replay/delete)
    # or 60s (export). Pre-fix the operator clicked the button and saw NO
    # feedback until the subprocess returned — a long, silent gap that read
    # as a frozen UI. The action panel now branches on this Var to render an
    # inline spinner + "Running …" copy so the operator knows the shell-out
    # is in flight. Set to a short human label (e.g. ``"diff-runs"``) at the
    # start of each handler and cleared at the end via try/finally.
    action_in_flight: str = ""

    # FRONTEND-A-001 (v1.4 Wave 2): comparison run id for the Diff button.
    # Pre-fix ``diff_against`` was a fully-implemented handler with no UI
    # control invoking it (the brief promised 4 action buttons; only 3
    # shipped). The Diff form on ``run_detail.py:_action_panel`` writes into
    # this field via ``set_diff_other_run_id``; the Compare button calls
    # ``diff_against`` with the current value. Form lives next to the
    # primary action row so the operator never leaves the page to compare.
    diff_other_run_id: str = ""

    @rx.event
    def set_diff_other_run_id(self, value: str) -> None:
        """Update the comparison-run-id text input (FRONTEND-A-001).

        UI-A-003 (Wave A1 HIGH): validate at the trust boundary. The value
        flows into ``subprocess`` argv (``diff_against``); an option-shaped
        run id (``--to=…``) would be parsed as a CLI flag downstream. Reject
        anything outside the strict ``[A-Za-z0-9_-]{1,64}`` allowlist and
        surface a clean error instead of storing it.
        """
        # Strip whitespace so a copy-pasted run id with trailing whitespace
        # doesn't trip RunHistoryManager's exact-id lookup downstream.
        cleaned, err = _validate_run_id(value)
        self.diff_other_run_id = cleaned
        if err:
            self.action_error = err
        elif self.action_error.startswith("Invalid run id"):
            # Clear a stale run-id validation error once the field is valid.
            self.action_error = ""

    @rx.event
    def diff_with_input(self) -> None:
        """Form submit handler — calls ``diff_against`` with the input value.

        Thin wrapper so the Compare button can be a plain ``on_click`` rather
        than needing to thread the input's local var through the closure.
        Mirrors the input-state-then-handler pattern Train/Multi-Run use for
        every config field.
        """
        # FRONTEND-B-002 (Stage C humanization): clear ``action_result`` on
        # both validation-failure branches. Pre-fix, a previous Replay /
        # Export success could sit next to a fresh Diff validation error,
        # rendering two contradictory action-panel messages at once. Mirrors
        # the OTHER-field-clear pattern already used in ``diff_against``.
        if not self.diff_other_run_id:
            self.action_error = "Enter a comparison run id."
            self.action_result = ""
            return
        if self.diff_other_run_id == self.current_run_id:
            self.action_error = (
                "Comparison run id must differ from the current run id."
            )
            self.action_result = ""
            return
        self.diff_against(self.diff_other_run_id)

    @rx.event
    def load_run(self) -> None:
        """Populate fields from the on-disk run history."""
        from pathlib import Path as _Path

        # Resolve run_id from the dynamic route parameter; fall back to
        # whatever the operator set programmatically into ``current_run_id``.
        # Route arg is named ``rid`` (not ``run_id``) to avoid shadowing the
        # existing ``AppState.run_id`` state var — Reflex 0.9 refuses to
        # bind a dynamic route arg that shadows any state var anywhere in
        # the state tree (DynamicRouteArgShadowsStateVarError).
        route_run_id = ""
        try:
            route_run_id = str(self.router.page.params.get("rid", "") or "")
        except Exception:  # noqa: BLE001 — defensive
            pass  # nosec B110 — defensive route-param read; missing param falls back to ""

        self.loading = True
        self.error = ""
        self.not_found = False
        # FRONTEND-B-001 (v1.4 Wave 3.5): clear the post-delete chrome
        # when (re)loading any run — navigating from a just-deleted run's
        # URL to a different run id must not carry the "Run deleted."
        # surface forward. This reset happens BEFORE the UI-A-003 route-param
        # validation early-return so a malformed URL still clears the chrome.
        self.was_deleted = False

        # UI-A-003 (Wave A1 HIGH): validate the route param at the trust
        # boundary before it reaches ``current_run_id`` (and thence the
        # diff/replay subprocess argv). An option-shaped ``rid`` (``--to=…``)
        # would be parsed as a CLI flag downstream. Reject it with a clean
        # error rather than assigning the malicious value.
        if route_run_id:
            cleaned_rid, rid_err = _validate_run_id(route_run_id)
            if rid_err:
                self.error = rid_err
                self.not_found = True
                self.loading = False
                return
            self.current_run_id = cleaned_rid

        try:
            try:
                from .ui_security import get_ui_output_dir
                history_dir = get_ui_output_dir()
            except Exception:
                history_dir = _Path.home() / ".backpropagate" / "ui-outputs"

            if not history_dir.exists():
                # V2-a (sibling): ``error`` is a public RunDetailState var;
                # history_dir embeds the home dir + username. Redact.
                self.error = _redact_action(f"No run history at {history_dir}.")
                return

            try:
                from .checkpoints import RunHistoryManager
            except ImportError as exc:
                self.error = _redact_action(f"checkpoints module unavailable: {exc}")
                return

            manager = RunHistoryManager(str(history_dir))
            entry = manager.get_run(self.current_run_id) if self.current_run_id else None
            if entry is None:
                self.not_found = True
                self.error = _redact_action(
                    f"Run '{self.current_run_id}' not found in {history_dir}."
                )
                return

            # Populate header fields.
            self.status = str(entry.get("status") or "-")
            self.model = str(entry.get("model_name") or "-")
            self.dataset = str(entry.get("dataset_info") or "-")
            self.started_at = str(entry.get("started_at") or entry.get("timestamp") or "-")
            self.completed_at = str(entry.get("completed_at") or "-")
            duration = entry.get("duration_seconds")
            if duration is None:
                self.duration = "-"
            else:
                try:
                    self.duration = f"{float(duration):.0f}s"
                except (TypeError, ValueError):
                    self.duration = "-"
            final_loss = entry.get("final_loss")
            if final_loss is None:
                self.final_loss = "-"
            else:
                try:
                    self.final_loss = f"{float(final_loss):.4f}"
                except (TypeError, ValueError):
                    self.final_loss = "-"
            # UI-A-002: full path into the backend-only var; the client reads
            # the redacted ``checkpoint_path_display`` computed var.
            self._checkpoint_path = str(entry.get("checkpoint_path") or "-")

            # Hyperparameter table — flatten the entry dict into {key, value}
            # rows, skipping the fields surfaced as headers above so the
            # operator only sees the per-run config knobs.
            header_keys = {
                "run_id", "status", "model_name", "dataset_info",
                "started_at", "completed_at", "timestamp",
                "duration_seconds", "final_loss", "checkpoint_path",
                "loss_history",
            }
            hp_rows: list[dict] = []
            for key in sorted(entry.keys()):
                if key in header_keys:
                    continue
                value = entry[key]
                # Coerce all values to short strings for table render.
                if isinstance(value, (dict, list)):
                    import json as _json
                    value_str = _json.dumps(value, default=str)[:200]
                else:
                    value_str = str(value)[:200]
                hp_rows.append({"key": key, "value": value_str})
            self.hyperparameters = hp_rows

            # Loss history — embedded in the run entry (preferred) or
            # read from training_metrics.jsonl alongside the checkpoint.
            loss_hist = entry.get("loss_history") or []
            if isinstance(loss_hist, list):
                try:
                    self.loss_history = [float(x) for x in loss_hist if isinstance(x, (int, float))]
                except (TypeError, ValueError):
                    self.loss_history = []
            else:
                self.loss_history = []

            # Checkpoint list — walk the checkpoint_path directory.
            self.checkpoints = []
            if entry.get("checkpoint_path"):
                cp_dir = _Path(str(entry["checkpoint_path"])).expanduser()
                if cp_dir.exists() and cp_dir.is_dir():
                    try:
                        for child in sorted(cp_dir.iterdir()):
                            if not child.is_dir():
                                continue
                            size_bytes = sum(
                                p.stat().st_size for p in child.rglob("*") if p.is_file()
                            )
                            self.checkpoints.append({
                                "name": child.name,
                                "size_mb": f"{size_bytes / (1024**2):.1f}",
                                "timestamp": str(
                                    __import__("datetime").datetime.fromtimestamp(
                                        child.stat().st_mtime
                                    ).isoformat(timespec="seconds")
                                ),
                            })
                    except OSError:
                        # Best-effort — the run page still renders without
                        # checkpoints if filesystem walk fails.
                        pass

            # Log tail — read last 200 lines from training.log if present
            # alongside the checkpoint. Capped at 200 lines (~16 KB) to
            # keep the WS bundle small.
            self.log_lines = []
            if entry.get("checkpoint_path"):
                log_path = _Path(str(entry["checkpoint_path"])).expanduser() / "training.log"
                if log_path.exists() and log_path.is_file():
                    try:
                        with open(log_path, encoding="utf-8", errors="replace") as f:
                            tail = f.readlines()[-200:]
                            # UI-A-002: a training log routinely embeds absolute
                            # paths (checkpoint dir, HF cache, tempdirs) that
                            # carry the operator's home dir + username. Redact
                            # each line before it enters the client-serialized
                            # ``log_lines`` var.
                            self.log_lines = [
                                _redact_action(line.rstrip("\n")) for line in tail
                            ]
                    except OSError:
                        pass
        finally:
            self.loading = False

    @rx.var
    def loss_chart_data(self) -> list[dict]:
        """Shape ``loss_history`` for ``rx.recharts.line_chart`` consumption.

        Returns ``[{"step": i, "loss": v}, ...]`` — the dict shape recharts
        needs. Computed-Var so the chart re-renders without an explicit
        event handler when ``load_run`` repopulates ``loss_history``.
        """
        return [{"step": i, "loss": v} for i, v in enumerate(self.loss_history)]

    @rx.event
    def diff_against(self, other_run_id: str) -> None:
        """Shell out to ``backprop diff-runs <self.current_run_id> <other_run_id>``.

        Bridge owns the subcommand (V1_3_BRIEF / Wave 6b BRIDGE-6); this
        UI just dispatches and surfaces the result. Failures surface via
        ``action_error``.
        """
        import shutil
        import subprocess

        if not self.current_run_id or not other_run_id:
            self.action_error = "Both run IDs are required for diff."
            return
        cmd = shutil.which("backprop") or shutil.which("backpropagate")
        if not cmd:
            self.action_error = "`backprop` CLI not found on PATH."
            return
        # FRONTEND-B-014-EXTENDED (Stage C humanization): mark the shell-out
        # as in flight so the action panel renders an inline spinner. Cleared
        # in the finally so a thrown exception still leaves the UI in a
        # consistent state.
        # UI-A-003 (Wave A1 HIGH): validate both run IDs at the boundary even
        # though set_diff_other_run_id / load_run already guard their inputs —
        # belt-and-suspenders so a future caller that sets current_run_id /
        # other_run_id by another path can't smuggle an option-shaped id into
        # argv. Reject rather than shelling out.
        _, err_cur = _validate_run_id(self.current_run_id)
        _, err_other = _validate_run_id(other_run_id)
        if err_cur or err_other:
            self.action_error = err_cur or err_other
            return
        self.action_in_flight = "diff-runs"
        try:
            # ``--`` ends option parsing so the run IDs are always treated as
            # positionals by the downstream argparse CLI, never as flags.
            result = subprocess.run(
                [cmd, "diff-runs", "--", self.current_run_id, other_run_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                self.action_result = result.stdout[:5000]
                self.action_error = ""
            else:
                self.action_error = (result.stderr or result.stdout)[:1000]
                self.action_result = ""
        except (subprocess.TimeoutExpired, OSError) as exc:
            # V2-a (sibling): the OSError / TimeoutExpired repr can embed the
            # resolved ``backprop`` binary path (an absolute path under the
            # operator's environment); redact before surfacing to the client.
            self.action_error = _redact_action(f"diff-runs failed: {exc}")
        finally:
            self.action_in_flight = ""

    @rx.event
    def replay(self) -> None:
        """Validate that the current run is replayable IN-PROCESS.

        Pre-fix this shelled out to ``backprop replay --dry-run -- <id>`` —
        but the ``replay`` subcommand has NO ``--dry-run`` flag (it accepts
        only ``run_id`` / ``--output`` / ``--override`` / ``--json``), so
        argparse rejected the token (``unrecognized arguments: --dry-run``)
        and the Replay button always failed. This is the same phantom-CLI-
        surface bug class as UI-A-004 (delete_run / export_run) — the sibling
        fix the UI-A-003 separator test flagged as out-of-scope at the time —
        and the remedy matches UI-A-004: do the check IN-PROCESS via
        ``RunHistoryManager``. No subprocess, no PATH dependency (the
        ``backprop`` binary need not be on the Reflex server's PATH), no
        phantom flag.

        The button is a *preflight*, not the heavy replay: it confirms the
        run exists and carries the one hard precondition ``cmd_replay``
        enforces before launching training — a recorded ``dataset_info``
        (cli.py ``cmd_replay`` returns EXIT_USER_ERROR when it is ``None``) —
        then directs the operator to run ``backprop replay <id>`` from the
        shell to start the actual (heavy) job.
        """
        if not self.current_run_id:
            self.action_error = "No run loaded."
            return
        # UI-A-003: validate even though the id no longer reaches argv — a
        # malformed id should surface a clean error rather than a confusing
        # "not found" deeper in the lookup.
        _, rid_err = _validate_run_id(self.current_run_id)
        if rid_err:
            self.action_error = rid_err
            return
        # FRONTEND-B-014-EXTENDED (Stage C humanization): see diff_against.
        self.action_in_flight = "replay"
        try:
            history_dir = self._resolve_history_dir()
            if not history_dir.exists():
                # V2-a: history_dir embeds the home dir + username; redact
                # before assigning to the client-serialized action_error.
                self.action_error = _redact_action(f"No run history at {history_dir}.")
                self.action_result = ""
                return
            from .checkpoints import RunHistoryManager

            manager = RunHistoryManager(str(history_dir))
            entry = manager.get_run(self.current_run_id)
            if entry is None:
                self.action_error = _redact_action(
                    f"Run '{self.current_run_id}' not found in {history_dir}."
                )
                self.action_result = ""
                return
            # Mirror cmd_replay's one hard precondition (cli.py cmd_replay):
            # a run with no recorded dataset_info cannot be replayed
            # automatically, so a dry-run reporting "OK" here would promise a
            # replay the real command rejects. Surface the same verdict.
            if entry.get("dataset_info") is None:
                self.action_error = (
                    f"Run '{self.current_run_id}' has no dataset_info "
                    "recorded — cannot replay automatically. Re-run manually "
                    "with `backprop train --data <dataset>` matching the "
                    "original configuration."
                )
                self.action_result = ""
                return
            model = entry.get("model_name") or "(default model)"
            session_kind = entry.get("session_kind") or "single_run"
            self.action_result = (
                f"Dry-run OK — run {self.current_run_id} is replayable "
                f"(session={session_kind}, model={model}, "
                f"dataset={entry.get('dataset_info')}). To actually replay: "
                f"`backprop replay {self.current_run_id}` from the shell."
            )
            self.action_error = ""
        except Exception as exc:  # noqa: BLE001 — operator-facing string
            # V2-a: a raw OSError repr embeds absolute paths (the home dir +
            # username); redact before surfacing to the client-serialized var.
            self.action_error = _redact_action(
                f"replay check failed: {type(exc).__name__}: {exc}"
            )
            self.action_result = ""
        finally:
            self.action_in_flight = ""

    def _resolve_history_dir(self):
        """Resolve the sandboxed run-history directory.

        Mirrors ``load_run``'s resolution: prefer the UI sandbox
        (``get_ui_output_dir()``), fall back to the legacy default. Kept as
        one helper so the in-process action handlers (UI-A-004) and
        ``load_run`` stay in lockstep.
        """
        from pathlib import Path as _Path

        try:
            from .ui_security import get_ui_output_dir

            return get_ui_output_dir()
        except Exception:  # noqa: BLE001 — defensive; fall back to legacy default
            return _Path.home() / ".backpropagate" / "ui-outputs"

    @rx.event
    def delete_run(self) -> None:
        """Delete the current run's history entry IN-PROCESS (UI-A-004).

        Pre-fix this shelled out to ``backprop delete-run <id> --yes`` — a
        subcommand that DOES NOT EXIST (argparse rejected it → the Delete
        button always failed). We use ``RunHistoryManager.delete_run``
        directly, exactly as ``load_run`` reads via ``get_run``. No
        subprocess, no PATH dependency, no phantom CLI surface.

        Operator confirmation is the responsibility of the UI button (a
        confirm-dialog wrap); this handler unconditionally executes.
        """
        if not self.current_run_id:
            self.action_error = "No run loaded."
            return
        # UI-A-003: the id flows nowhere dangerous now (in-process), but
        # validate anyway so a malformed id surfaces a clean error rather
        # than a confusing "not found".
        _, rid_err = _validate_run_id(self.current_run_id)
        if rid_err:
            self.action_error = rid_err
            return
        # FRONTEND-B-014-EXTENDED (Stage C humanization): see diff_against.
        self.action_in_flight = "delete-run"
        try:
            history_dir = self._resolve_history_dir()
            if not history_dir.exists():
                # V2-a: redact the home-dir-bearing history_dir before it
                # reaches the client-serialized action_error.
                self.action_error = _redact_action(f"No run history at {history_dir}.")
                self.action_result = ""
                return
            from .checkpoints import RunHistoryManager

            manager = RunHistoryManager(str(history_dir))
            deleted = manager.delete_run(self.current_run_id)
            if deleted:
                self.action_result = f"Run {self.current_run_id} deleted."
                self.action_error = ""
                # FRONTEND-B-001 (v1.4 Wave 3.5): a successful delete must
                # render the "Run deleted." chrome with a Back-to-runs
                # affordance, NOT the "Run not found" chrome. Setting
                # ``not_found = True`` here pre-fix latched the page into
                # the unknown-id surface and stranded ``action_result``
                # behind a template that never displays it. The template
                # in ``run_detail.py`` branches on ``was_deleted`` BEFORE
                # ``not_found`` so the deletion confirmation wins.
                self.was_deleted = True
            else:
                self.action_error = _redact_action(
                    f"Run '{self.current_run_id}' not found in {history_dir}."
                )
                self.action_result = ""
        except Exception as exc:  # noqa: BLE001 — operator-facing string
            # V2-a: redact absolute paths from the raw exception repr.
            self.action_error = _redact_action(
                f"delete failed: {type(exc).__name__}: {exc}"
            )
            self.action_result = ""
        finally:
            self.action_in_flight = ""

    @rx.event
    def export_run(self) -> None:
        """Export the current run as a single-record JSONL IN-PROCESS (UI-A-004).

        Pre-fix this shelled out to ``backprop export-runs --run-id <id>`` —
        but ``export-runs`` has NO ``--run-id`` flag (only ``--output/-o``,
        ``--format``, ``--to``, ``--status``), so argparse rejected it and
        the Export button always failed. We read the single run via
        ``RunHistoryManager.get_run`` and write one JSONL record to a
        SANDBOXED path under ``get_ui_output_dir()`` — no phantom flag, no
        filesystem-wide write surface.
        """
        import json
        from datetime import datetime, timezone

        if not self.current_run_id:
            self.action_error = "No run loaded."
            return
        _, rid_err = _validate_run_id(self.current_run_id)
        if rid_err:
            self.action_error = rid_err
            return
        # FRONTEND-B-014-EXTENDED (Stage C humanization): see diff_against.
        self.action_in_flight = "export-runs"
        try:
            history_dir = self._resolve_history_dir()
            if not history_dir.exists():
                # V2-a: redact the home-dir-bearing history_dir.
                self.action_error = _redact_action(f"No run history at {history_dir}.")
                self.action_result = ""
                return
            from .checkpoints import RunHistoryManager

            manager = RunHistoryManager(str(history_dir))
            entry = manager.get_run(self.current_run_id)
            if entry is None:
                self.action_error = _redact_action(
                    f"Run '{self.current_run_id}' not found in {history_dir}."
                )
                self.action_result = ""
                return

            # Write inside a sandboxed ``exports/`` subdir of the UI output
            # dir. The filename is derived from the (already-validated, so
            # filesystem-safe ``[A-Za-z0-9_-]``) run id plus a UTC stamp so
            # repeat exports don't clobber. No user-controlled path segment
            # escapes the sandbox.
            exports_dir = history_dir / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = exports_dir / f"run-{self.current_run_id}-{stamp}.jsonl"
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(entry, default=str))
                fh.write("\n")
            # V2-a: out_path is the sandbox absolute path (home dir +
            # username); redact before surfacing the success string to the
            # client. The basename is preserved so the operator can still
            # locate the file inside their UI output dir.
            self.action_result = _redact_action(
                f"Exported run {self.current_run_id} to {out_path}."
            )
            self.action_error = ""
        except Exception as exc:  # noqa: BLE001 — operator-facing string
            # V2-a: redact absolute paths from the raw exception repr.
            self.action_error = _redact_action(
                f"export failed: {type(exc).__name__}: {exc}"
            )
            self.action_result = ""
        finally:
            self.action_in_flight = ""

    @rx.event
    def clear_action_message(self) -> None:
        """Dismiss the action result / error banner."""
        self.action_result = ""
        self.action_error = ""


# ---------------------------------------------------------------------------
# ModelsState — backs /models (Wave 6b)
# ---------------------------------------------------------------------------
#
# Lists local Hugging Face cache contents + per-model disk usage + unused-
# model cleanup affordance. Pulls from ``~/.cache/huggingface/hub/``
# directly via filesystem APIs — no `huggingface_hub` dep required (avoids
# pulling the heavy optional dep into the [ui] extra path).
#
# v1.3 surfaces:
#   - Total cache size + per-model breakdown
#   - Last-modified timestamp (proxy for "last used")
#   - Delete-model affordance (operator confirms before the rm -rf)


class ModelsState(rx.State):
    """Models surface state — local HF cache inventory."""

    models: list[dict] = []
    total_size_mb: str = "0"
    # UI-A-002 (Wave A2): the HF cache dir is ``~/.cache/huggingface/hub`` —
    # it embeds the operator's home dir + username. Held in a backend-only
    # var; the client reads the redacted ``cache_dir_display`` computed var.
    _cache_dir: str = ""
    loading: bool = False
    error: str = ""
    last_loaded_at: str = ""

    @rx.var
    def cache_dir_display(self) -> str:
        """Client-facing, redacted form of the HF cache directory (UI-A-002)."""
        if not self._cache_dir:
            return ""
        return _redact_action(self._cache_dir)

    @rx.event
    def load_models(self) -> None:
        """Walk ``~/.cache/huggingface/hub/`` and populate ``self.models``.

        HF cache layout (per huggingface_hub docs):
            <cache>/models--<owner>--<model>/snapshots/<sha>/<files>
            <cache>/models--<owner>--<model>/refs/<rev>

        We surface one entry per ``models--*`` top-level dir; size is the
        recursive sum of all files inside.
        """
        from datetime import datetime, timezone
        from pathlib import Path as _Path

        self.loading = True
        self.error = ""
        try:
            cache_dir = _Path.home() / ".cache" / "huggingface" / "hub"
            # UI-A-002: full path into the backend-only var; the client reads
            # the redacted ``cache_dir_display`` computed var.
            self._cache_dir = str(cache_dir)
            if not cache_dir.exists():
                self.models = []
                self.total_size_mb = "0"
                # UI-A-002: ``error`` is a public var; redact the cache path.
                self.error = _redact_action(
                    f"No HF cache at {cache_dir}. Models download on first "
                    "use via `transformers.AutoModel.from_pretrained(...)`."
                )
                return

            model_rows: list[dict] = []
            total_bytes = 0
            try:
                for entry in sorted(cache_dir.iterdir()):
                    if not entry.is_dir():
                        continue
                    name = entry.name
                    if not name.startswith("models--"):
                        continue
                    # Unmangle: ``models--meta-llama--Llama-3.1-8B`` ->
                    # ``meta-llama/Llama-3.1-8B``
                    pretty = name[len("models--"):].replace("--", "/", 1)
                    try:
                        size_bytes = sum(
                            p.stat().st_size for p in entry.rglob("*") if p.is_file()
                        )
                    except OSError:
                        size_bytes = 0
                    total_bytes += size_bytes
                    mtime = 0.0
                    try:
                        mtime = entry.stat().st_mtime
                    except OSError:
                        pass
                    model_rows.append({
                        "name": pretty,
                        "dir_name": name,
                        "size_mb": f"{size_bytes / (1024**2):.1f}",
                        "size_bytes": size_bytes,
                        "last_modified": (
                            datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
                            if mtime else "-"
                        ),
                    })
            except OSError as exc:
                # UI-A-002: the OSError repr embeds the cache path.
                self.error = _redact_action(f"Cannot walk HF cache: {exc}")
                self.models = []
                self.total_size_mb = "0"
                return

            # Sort by size descending — heaviest cache offenders first.
            model_rows.sort(key=lambda r: r["size_bytes"], reverse=True)
            self.models = model_rows
            self.total_size_mb = f"{total_bytes / (1024**2):.1f}"
            self.last_loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        finally:
            self.loading = False

    @rx.event
    def delete_model(self, dir_name: str) -> None:
        """Delete one ``models--*`` directory from the HF cache.

        The operator confirms via a UI button click; the handler unconditionally
        proceeds. Failures surface via ``self.error``.

        FRONTEND-F-007 safety: the path is validated to live under
        ``~/.cache/huggingface/hub/`` so a malicious operator-controlled
        ``dir_name`` can't escape the cache via ``..`` traversal.

        UI-A-006 (Wave A2) hardening:
          - Use ``Path.is_relative_to`` for the confinement check instead of
            ``str.startswith``. A string-prefix test treats a SIBLING dir
            whose name shares the cache prefix (e.g. ``…/hub-evil``) as
            "inside" the cache; the path-component check does not.
          - Refuse the delete if ``target`` (the UNRESOLVED path) is a
            symlink. Pre-fix the handler resolved the symlink and then
            ``rmtree``'d the resolved target — a ``models--*`` symlink in the
            cache pointing at, say, ``~/important`` would pass the
            resolved-prefix check (target resolves outside, but the OLD
            ``startswith`` compared against the resolved cache root) OR, worse,
            delete the link's target outside the cache. Refusing symlinks
            outright removes the foot-gun; real HF cache entries are plain
            directories.
        """
        import shutil
        from pathlib import Path as _Path

        cache_dir = _Path.home() / ".cache" / "huggingface" / "hub"
        if not dir_name or not dir_name.startswith("models--") or "/" in dir_name or "\\" in dir_name or ".." in dir_name:
            self.error = f"Invalid model directory name: {dir_name!r}"
            return
        target = cache_dir / dir_name
        try:
            # UI-A-006: refuse symlinks BEFORE resolving so a malicious
            # ``models--*`` symlink can't redirect the rmtree outside the
            # cache. Check the unresolved path.
            if target.is_symlink():
                self.error = _redact_action(
                    f"Refusing to delete a symlinked cache entry: {target}"
                )
                return
            target_resolved = target.resolve()
            cache_resolved = cache_dir.resolve()
            # UI-A-006: path-component confinement (is_relative_to), not a
            # string-prefix test. ``is_relative_to`` returns False for a
            # sibling dir that merely shares the prefix string.
            if not target_resolved.is_relative_to(cache_resolved):
                self.error = _redact_action(
                    f"Refusing to delete outside HF cache: {target_resolved}"
                )
                return
            if not target_resolved.exists():
                self.error = _redact_action(f"Model directory not found: {target}")
                return
            # Operate on the confined, resolved path. ``rmtree`` does not
            # follow a top-level symlink (and we refused one above anyway).
            shutil.rmtree(target_resolved)
        except OSError as exc:
            self.error = _redact_action(f"Failed to delete {dir_name}: {exc}")
            return
        # Reload to reflect the deletion.
        self.load_models()


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
    "RunsState",
    "RunDetailState",
    "ModelsState",
    "AuthBadgeState",
]
