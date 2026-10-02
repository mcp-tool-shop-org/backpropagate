"""Multi-Run page — ``/multi-run`` — SLAO sweep surface (ui-v2 P2).

Starts a real ``backprop multi-run`` job through the JobManager. Every field
on the form reaches the CLI: model, dataset, runs, steps per run, samples
per run and the merge choice (SLAO / simple average / TIES). Settings the
multi-run CLI has no flag for (learning rate, batch size, LoRA shape) are
not shown, so nothing on the form is silently ignored.

While the job runs the page shows the shared progress card ("run 2 of 3",
one bar for the whole session), the loss chart, and Stop and save; the rail
follows the same job (``TrainState`` follows every UI job).
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import MultiRunState, TrainState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
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


def _model_group() -> rx.Component:
    return Group(
        _field(
            "HuggingFace model id",
            rx.input(
                placeholder="meta-llama/Llama-3.1-8B",
                default_value=MultiRunState.model,
                on_change=MultiRunState.set_model,
                size="2",
                disabled=TrainState.form_disabled,
                style={**_FIELD_STYLE, "width": "100%"},
                aria_label="HuggingFace model id",
            ),
            MultiRunState.model_error,
        ),
        _field(
            "Dataset path",
            rx.input(
                placeholder="path/to/dataset.jsonl",
                default_value=MultiRunState.dataset_path,
                on_change=MultiRunState.set_dataset_path,
                size="2",
                disabled=TrainState.form_disabled,
                style={**_FIELD_STYLE, "width": "100%"},
                aria_label="Path to training dataset (JSONL)",
            ),
            MultiRunState.dataset_path_error,
        ),
        rx.text(
            "Each run trains on a fresh slice of the dataset; the runs' adapters "
            "are merged as the sweep goes.",
            size="1",
            style={"color": "var(--bp-muted)"},
        ),
        title="Model and data",
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
        ),
        rx.text(
            "Learning rate and LoRA settings use the multi-run defaults.",
            size="1",
            style={"color": "var(--bp-muted)"},
        ),
        title="Sweep shape",
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


def multi_run_page() -> rx.Component:
    """The Multi-Run surface."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="multi-run"),
            rx.scroll_area(
                rx.flex(
                    rx.flex(
                        rx.heading(
                            "Multi-run",
                            size="7",
                            style={
                                "color": "var(--bp-text)",
                                "font_weight": "600",
                                "letter_spacing": "-0.02em",
                            },
                        ),
                        rx.text(
                            "SLAO sweep: train several short runs and merge their "
                            "LoRA adapters, which keeps earlier learning from being "
                            "overwritten.",
                            size="2",
                            style={"color": "var(--bp-muted)"},
                        ),
                        direction="column",
                        gap="var(--space-2)",
                        width="100%",
                    ),
                    job_reattach_banner(),
                    job_refusal_callout(),
                    job_error_callout(),
                    job_progress_card(),
                    rx.grid(
                        _model_group(),
                        _sweep_shape_group(),
                        columns=rx.breakpoints(initial="1", md="2"),
                        gap="var(--space-6)",
                        width="100%",
                        align="start",
                    ),
                    job_loss_chart_card(),
                    job_next_steps_panel(),
                    rx.flex(
                        _start_stop_button(),
                        gap="var(--space-3)",
                        margin_top="var(--space-2)",
                        align="center",
                        justify="end",
                    ),
                    direction="column",
                    gap="var(--space-6)",
                    padding="var(--space-7)",
                    max_width="1320px",
                    width="100%",
                    on_mount=[TrainState.refresh_gpu, TrainState.attach_active_job],
                ),
                flex_grow="1",
                style={"height": "100%"},
                type="auto",
                scrollbars="vertical",
            ),
            BpSideRail(),
            flex_grow="1",
            width="100%",
            style={"overflow": "hidden", "min_height": "0"},
        ),
        BpFooter(),
        direction="column",
        height="100vh",
        width="100%",
        style={"background": "var(--bp-bg)"},
    )
