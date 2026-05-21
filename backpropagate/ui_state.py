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

from typing import Literal

import reflex as rx

# Shared literal types — referenced across multiple State classes.
RunState = Literal["idle", "loading", "active", "paused", "done", "error"]
Theme = Literal["dark", "light"]
ActiveSurface = Literal["train", "multi-run", "export", "dataset"]
ExportFormat = Literal["lora", "merged", "gguf"]


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
    dataset_path: str = ""
    steps: int = 100
    batch_size: str = "auto"
    learning_rate: float = 2e-4
    lora_r: int = 16
    quantization: str = "4-bit"

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
    dataset_path: str = ""
    steps: int = 100
    batch_size: str = "auto"
    learning_rate: float = 2e-4
    lora_r: int = 16

    # Multi-Run specific.
    num_runs: int = 3
    samples_per_run: int = 500
    merge_mode: Literal["slao", "weighted", "ties"] = "slao"

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
    format: ExportFormat = "lora"
    gguf_quant: str = "q4_K_M"
    ollama_register: bool = False
    ollama_name: str = ""

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
    detected_format: str = ""
    preview_records: list[dict] = []
    dedup_enabled: bool = True
    record_count: int = 0
    dedup_hits: int = 0

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
