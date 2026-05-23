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
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import reflex as rx

# Shared literal types — referenced across multiple State classes.
RunState = Literal["idle", "loading", "active", "paused", "done", "error"]
Theme = Literal["dark", "light"]
ActiveSurface = Literal["train", "multi-run", "export", "dataset"]
ExportFormat = Literal["lora", "merged", "gguf"]


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


class AppState(rx.State):
    """Top-level state: theme toggle, active surface, current run_id.

    Cross-surface state lives here so the header / left nav / footer can read
    it without coupling to a specific page's state class.
    """

    theme: Theme = "dark"
    active_surface: ActiveSurface = "train"
    run_id: str = ""

    def toggle_theme(self) -> None:
        """Flip dark/light. The TOKENS_CSS stylesheet keys off
        ``[data-theme="light"]`` so this re-themes the whole tree."""
        self.theme = "light" if self.theme == "dark" else "dark"

    def set_active_surface(self, surface: str) -> None:
        """Left-nav click handler. The ``surface`` arg is a free string from
        the click event; clamp to the known set."""
        if surface in ("train", "multi-run", "export", "dataset"):
            self.active_surface = surface  # type: ignore[assignment]


class TrainState(rx.State):
    """Train surface state: model, dataset, hyperparams, run progress."""

    # ---- Configuration form ------------------------------------------------
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    model_error: str = ""
    dataset_path: str = ""
    dataset_path_error: str = ""
    steps: int = 100
    batch_size: str = "auto"
    learning_rate: float = 2e-4
    lora_r: int = 16
    quantization: str = "4-bit"

    # ---- Validated path setters (FRONTEND-A-002) ---------------------------

    @rx.event
    def set_model(self, value: str) -> None:
        """Validate and set the HF model id / local model path.

        Local paths are checked against ``get_ui_output_dir()``; HF repo ids
        (no path separators) pass through untouched since they're not paths.
        """
        if value and ("/" in value or "\\" in value) and Path(value).is_absolute():
            cleaned, err = _validate_ui_path(value)
            self.model = cleaned
            self.model_error = err
        else:
            self.model = value
            self.model_error = ""

    @rx.event
    def set_dataset_path(self, value: str) -> None:
        """Validate and set the dataset path; clears error on success."""
        cleaned, err = _validate_ui_path(value)
        self.dataset_path = cleaned
        self.dataset_path_error = err

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

    # ---- Event handlers (stubs; backend hookup in Phase 3) -----------------

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

    def stop_training(self) -> None:
        """Stub handler for "Stop / pause" button."""
        if self.run_state in ("active", "loading", "paused"):
            self.run_state = "idle"
            self.events = [
                *self.events,
                {"t": "00:00:00", "level": "info", "msg": "[stub] training stopped"},
            ]


class MultiRunState(rx.State):
    """Multi-Run surface state: same as Train + num_runs, samples_per_run,
    merge_mode."""

    # Mirrors TrainState's config block.
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    model_error: str = ""
    dataset_path: str = ""
    dataset_path_error: str = ""
    steps: int = 100
    batch_size: str = "auto"
    learning_rate: float = 2e-4
    lora_r: int = 16

    # Multi-Run specific.
    num_runs: int = 3
    samples_per_run: int = 500
    merge_mode: Literal["slao", "weighted", "ties"] = "slao"

    # ---- Validated path setters (FRONTEND-A-002) ---------------------------

    @rx.event
    def set_model(self, value: str) -> None:
        """Validate and set the HF model id / local model path."""
        if value and ("/" in value or "\\" in value) and Path(value).is_absolute():
            cleaned, err = _validate_ui_path(value)
            self.model = cleaned
            self.model_error = err
        else:
            self.model = value
            self.model_error = ""

    @rx.event
    def set_dataset_path(self, value: str) -> None:
        """Validate and set the dataset path; clears error on success."""
        cleaned, err = _validate_ui_path(value)
        self.dataset_path = cleaned
        self.dataset_path_error = err

    # Live state.
    run_state: RunState = "idle"
    current_run_index: int = 0
    runs: list[dict] = []  # per-run summary (loss, step, status)
    events: list[dict] = []

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
    gguf_quant: str = "q4_K_M"
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
    def set_ollama_name(self, value: str) -> None:
        """Validate the Ollama model name — alphanumeric / dash / underscore /
        colon (for tag) only; reject anything that smells like a path."""
        import re

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
    dedup_enabled: bool = True
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
    def detect_format_stub(self) -> None:
        """Stub handler — placeholder for the upload->detect flow."""
        if self.uploaded_path:
            self.detected_format = "alpaca"


__all__ = [
    "RunState",
    "Theme",
    "ActiveSurface",
    "ExportFormat",
    "AppState",
    "TrainState",
    "MultiRunState",
    "ExportState",
    "DatasetState",
]
