"""Train page — ``/`` — the single-run surface (ui-v2 P3).

Every control reaches a real ``backprop train`` flag (``ui_jobs._training_
flags``), and the form opens on the CLI's own defaults:

- **Model**: a preset from ``config.MODEL_PRESETS`` (fills the model id and
  its recommended LoRA rank) or any model id.
- **Dataset**, with the format the chosen method needs.
- **Method**: SFT / ORPO / SimPO / KTO and each method's knobs.
- **Mode**: QLoRA / LoRA / Full fine-tune.
- **Training shape** and **LoRA** (Quality / Fast shapes, or your own).
- **Advanced**: the GPU temperature limit, run name, gradient checkpointing.
- **Estimated VRAM**: "Fits / Tight / Won't fit", the same numbers as
  ``backprop estimate-vram``.

A live run gets the shared progress card, the loss chart and Stop and save;
config fields lock while a run is active.
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import TrainState

from ..components.job_panel import (
    job_error_callout,
    job_loss_chart_card,
    job_next_steps_panel,
    job_progress_card,
    job_reattach_banner,
    job_refusal_callout,
)
from ..components.page import bp_page
from ..components.train_form import (
    advanced_card,
    dataset_card,
    lora_card,
    method_card,
    mode_card,
    start_from_card,
    training_shape_card,
    vram_estimate_card,
)


def _column(*children: rx.Component) -> rx.Component:
    return rx.flex(*children, direction="column", gap="var(--space-6)", width="100%")


def _start_stop_button() -> rx.Component:
    """Start / Stop-and-save toggle (ui-v2 P1).

    While a run is active the primary action becomes the cooperative
    "Stop and save checkpoint" — never an immediate kill. WCAG 2.5.3: the
    accessible name CONTAINS the visible text, exactly, so assistive tech
    (and the acceptance test's ``get_by_role``) match the rendered label.
    """
    return rx.cond(
        TrainState.run_state == "active",
        rx.button(
            "Stop and save checkpoint",
            variant="soft",
            color_scheme="red",
            size="3",
            on_click=TrainState.stop_training,
            disabled=TrainState.stop_requested,
            style={"border_radius": "var(--bp-r-pill)", "min_width": "200px"},
            aria_label="Stop and save checkpoint",
        ),
        rx.button(
            "Start training",
            variant="solid",
            color_scheme="teal",
            size="3",
            disabled=TrainState.form_disabled,
            style={"border_radius": "var(--bp-r-pill)", "min_width": "200px"},
            on_click=TrainState.start_training,
            aria_label="Start training",
        ),
    )


def _action_bar() -> rx.Component:
    """The estimate next to the primary action: you see whether it fits
    right where you press Start."""
    return rx.box(
        vram_estimate_card(action=_start_stop_button()),
        class_name="bp-action-bar",
        width="100%",
    )


def train_page() -> rx.Component:
    """The Train surface."""
    return bp_page(
        job_reattach_banner(),
        job_refusal_callout(),
        job_error_callout(),
        job_progress_card(),
        rx.grid(
            _column(start_from_card(TrainState), dataset_card(TrainState), method_card(TrainState)),
            _column(
                mode_card(TrainState, allow_full=True),
                training_shape_card(TrainState),
                lora_card(TrainState),
            ),
            columns=rx.breakpoints(initial="1", lg="2"),
            gap="var(--space-6)",
            width="100%",
            align="start",
        ),
        advanced_card(TrainState),
        job_loss_chart_card(),
        job_next_steps_panel(),
        _action_bar(),
        active="train",
        title="Single run",
        description=(
            "Fine-tune a model in one run. The form starts on the same defaults as "
            "backprop train. Training runs in its own process, and the panel on "
            "the right shows live progress."
        ),
        on_mount=[
            TrainState.refresh_gpu,
            TrainState.attach_active_job,
            TrainState.refresh_estimate,
        ],
    )
