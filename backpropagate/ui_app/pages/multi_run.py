"""Multi-Run page — ``/multi-run`` — SLAO sweep surface (ui-v2 P2/P3).

Starts a real ``backprop multi-run`` job through the JobManager. Every field
on the form reaches a CLI flag: the shared training cards (model preset,
dataset, method, QLoRA/LoRA mode, learning rate, batch, LoRA shape, the
advanced knobs) plus runs, steps per run, samples per run and the merge
choice (SLAO / simple average / TIES). A multi-run merges LoRA adapters, so
there is no full fine-tune mode here.

While the job runs the page shows the shared progress card ("run 2 of 3",
one bar for the whole session), the loss chart, and Stop and save; the rail
follows the same job (``TrainState`` follows every UI job).
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import MultiRunState, TrainState

from ..components.field import FIELD_STYLE as _FIELD_STYLE
from ..components.field import bp_field as _field
from ..components.group import Group
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
)


def _number_input(value, on_change, placeholder: str, aria_label: str) -> rx.Component:
    return rx.input(
        placeholder=placeholder,
        value=value.to_string(),
        on_change=on_change,
        type="number",
        size="2",
        class_name="bp-num",
        disabled=TrainState.form_disabled,
        style={**_FIELD_STYLE, "width": "100%"},
        aria_label=aria_label,
    )


def _sweep_shape_group() -> rx.Component:
    return Group(
        rx.grid(
            _field(
                "Runs",
                _number_input(
                    MultiRunState.num_runs,
                    MultiRunState.set_num_runs,
                    "3",
                    "Number of runs in the sweep",
                ),
                MultiRunState.num_runs_error,
                info="runs",
            ),
            _field(
                "Steps per run",
                _number_input(
                    MultiRunState.steps,
                    MultiRunState.set_steps,
                    "100",
                    "Training steps in each run",
                ),
                MultiRunState.steps_error,
                info="steps",
            ),
            _field(
                "Samples per run",
                _number_input(
                    MultiRunState.samples_per_run,
                    MultiRunState.set_samples_per_run,
                    "500",
                    "Training samples in each run",
                ),
                MultiRunState.samples_per_run_error,
                info="samples_per_run",
            ),
            columns="repeat(3, 1fr)",
            gap="var(--space-5)",
            width="100%",
        ),
        _field(
            "Merge mode",
            rx.select.root(
                rx.select.trigger(
                    placeholder="SLAO",
                    style={**_FIELD_STYLE, "width": "100%"},
                    aria_label="Merge mode: SLAO, simple average, or TIES",
                ),
                rx.select.content(
                    rx.select.item("SLAO (default)", value="slao"),
                    rx.select.item("Simple average", value="simple"),
                    rx.select.item("TIES", value="ties"),
                ),
                value=MultiRunState.merge_mode,
                on_change=MultiRunState.set_merge_mode,
                disabled=TrainState.form_disabled,
            ),
            info="merge_mode",
        ),
        rx.text(
            "Each run trains on a fresh slice of the dataset, and the runs' "
            "adapters are merged as the sweep goes.",
            size="1",
            style={"color": "var(--bp-muted)", "font_size": "13px"},
        ),
        title="Rounds",
    )


def _start_stop_button() -> rx.Component:
    """Start, or Stop and save while this page's multi-run is going."""
    return rx.cond(
        (TrainState.run_state == "active") & (TrainState.job_kind == "multi_run"),
        rx.button(
            "Stop and save checkpoint",
            variant="soft",
            color_scheme="red",
            size="3",
            on_click=TrainState.stop_training,
            disabled=TrainState.stop_requested,
            style={"border_radius": "var(--bp-r-pill)", "min_width": "220px"},
            aria_label="Stop and save checkpoint",
        ),
        rx.button(
            "Start multi-run",
            variant="solid",
            color_scheme="teal",
            size="3",
            on_click=MultiRunState.start_multi_run,
            disabled=TrainState.form_disabled,
            style={"border_radius": "var(--bp-r-pill)", "min_width": "220px"},
            aria_label="Start multi-run",
        ),
    )


def _column(*children: rx.Component) -> rx.Component:
    return rx.flex(*children, direction="column", gap="var(--space-6)", width="100%")


def multi_run_page() -> rx.Component:
    """The Multi-Run surface."""
    return bp_page(
        job_reattach_banner(),
        job_refusal_callout(),
        job_error_callout(),
        job_progress_card(),
        rx.grid(
            _column(
                start_from_card(MultiRunState),
                dataset_card(MultiRunState),
                method_card(MultiRunState),
            ),
            _column(
                _sweep_shape_group(),
                mode_card(MultiRunState, allow_full=False),
                training_shape_card(MultiRunState, with_steps=False),
                lora_card(MultiRunState, recommend=True),
            ),
            columns=rx.breakpoints(initial="1", lg="2"),
            gap="var(--space-6)",
            width="100%",
            align="start",
        ),
        advanced_card(MultiRunState),
        job_loss_chart_card(),
        job_next_steps_panel(),
        rx.flex(
            _start_stop_button(),
            gap="var(--space-3)",
            align="center",
            justify="end",
            class_name="bp-action-bar",
            width="100%",
        ),
        active="multi-run",
        title="Multi-run",
        description=(
            "Train in several short rounds and merge the result after each one, "
            "so the model keeps what it learned earlier while it learns more."
        ),
        info="page_multi_run",
        on_mount=[
            TrainState.refresh_gpu,
            TrainState.attach_active_job,
            MultiRunState.refresh_shape,
        ],
    )
