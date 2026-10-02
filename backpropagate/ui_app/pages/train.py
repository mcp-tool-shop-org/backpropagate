"""Train page — ``/`` — the single-run surface (ui-v2 P1 redesign).

The Director's layout bar:

- Content lives in rounded cards (14px radius, soft shadow) on the page
  background — Model, Dataset, Training shape, LoRA tuning, Advanced.
- Two-column grid at desktop widths (no half-empty screen beside a crowd);
  one spacing scale (4/8/12/16/24/32/48) everywhere.
- A live run gets its own progress CARD: large "26 / 400" counter, rounded
  progress bar, heartbeat ("last step 2 s ago") and ETA range, loss + EMA.
  The loss chart (raw faint, smoothed bold) sits in its own card.
- Config fields lock while a run is active (they never re-target mid-run).
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import TrainState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.field import FIELD_STYLE, bp_err_text, bp_field, bp_label
from ..components.group import Group
from ..components.job_panel import (
    job_error_callout as _error_callout,
)
from ..components.job_panel import (
    job_loss_chart_card as _loss_chart_card,
)
from ..components.job_panel import (
    job_next_steps_panel as _next_steps_panel,
)
from ..components.job_panel import (
    job_progress_card as _run_progress_card,
)
from ..components.job_panel import (
    job_reattach_banner as _reattach_banner,
)
from ..components.job_panel import (
    job_refusal_callout as _refusal_callout,
)

# Field chrome is shared via components/field.py (ui-v2 P1 redesign pass 2);
# the private aliases keep this page's existing call sites + tests stable.
_FIELD_STYLE = FIELD_STYLE
_label = bp_label
_err_text = bp_err_text
_field = bp_field


def _model_group() -> rx.Component:
    return Group(
        rx.grid(
            _field(
                "HuggingFace model id",
                rx.input(
                    placeholder="meta-llama/Llama-3.1-8B",
                    default_value=TrainState.model,
                    on_change=TrainState.set_model,
                    size="2",
                    disabled=TrainState.form_disabled,
                    style={**_FIELD_STYLE, "width": "100%"},
                    aria_label="HuggingFace model id",
                ),
                TrainState.model_error,
            ),
            _field(
                "Quantization",
                rx.select.root(
                    rx.select.trigger(
                        placeholder="4-bit",
                        style={**_FIELD_STYLE, "width": "100%"},
                        aria_label="Quantization level — 4-bit, 8-bit, or 16-bit",
                    ),
                    rx.select.content(
                        rx.select.item("4-bit", value="4-bit"),
                        rx.select.item("8-bit", value="8-bit"),
                        rx.select.item("16-bit", value="16-bit"),
                    ),
                    value=TrainState.quantization,
                    on_change=TrainState.set_quantization,
                    disabled=TrainState.form_disabled,
                ),
            ),
            columns="2fr 1fr",
            gap="var(--space-5)",
            width="100%",
        ),
        title="Model",
    )


def _training_shape_group() -> rx.Component:
    return Group(
        rx.grid(
            _field(
                "Steps",
                rx.input(
                    placeholder="100",
                    value=TrainState.steps.to_string(),
                    on_change=TrainState.set_steps,
                    type="number",
                    size="2",
                    class_name="bp-num",
                    disabled=TrainState.form_disabled,
                    style=_FIELD_STYLE,
                    aria_label="Number of training steps",
                ),
                TrainState.steps_error,
            ),
            _field(
                "Batch size",
                rx.input(
                    placeholder="auto",
                    value=TrainState.batch_size,
                    on_change=TrainState.set_batch_size,
                    size="2",
                    class_name="bp-num",
                    disabled=TrainState.form_disabled,
                    style=_FIELD_STYLE,
                    aria_label="Batch size (number or auto)",
                ),
                TrainState.batch_size_error,
            ),
            _field(
                "Learning rate",
                rx.input(
                    placeholder="2e-4",
                    value=TrainState.learning_rate.to_string(),
                    on_change=TrainState.set_learning_rate,
                    size="2",
                    class_name="bp-num",
                    disabled=TrainState.form_disabled,
                    style=_FIELD_STYLE,
                    aria_label="Learning rate",
                ),
                TrainState.learning_rate_error,
            ),
            columns="repeat(3, 1fr)",
            gap="var(--space-5)",
            width="100%",
        ),
        title="Training shape",
    )


def _lora_group() -> rx.Component:
    return Group(
        rx.grid(
            _field(
                "LoRA rank",
                rx.input(
                    placeholder="16",
                    value=TrainState.lora_r.to_string(),
                    on_change=TrainState.set_lora_r,
                    type="number",
                    size="2",
                    class_name="bp-num",
                    disabled=TrainState.form_disabled,
                    style=_FIELD_STYLE,
                    aria_label="LoRA rank (r)",
                ),
                TrainState.lora_r_error,
            ),
            _field(
                "LoRA alpha",
                rx.input(
                    placeholder="32",
                    value=TrainState.lora_alpha.to_string(),
                    on_change=TrainState.set_lora_alpha,
                    size="2",
                    class_name="bp-num",
                    disabled=TrainState.form_disabled,
                    style=_FIELD_STYLE,
                    aria_label="LoRA alpha",
                ),
                TrainState.lora_alpha_error,
            ),
            _field(
                "Dropout",
                rx.input(
                    placeholder="0.05",
                    value=TrainState.lora_dropout.to_string(),
                    on_change=TrainState.set_lora_dropout,
                    size="2",
                    class_name="bp-num",
                    disabled=TrainState.form_disabled,
                    style=_FIELD_STYLE,
                    aria_label="LoRA dropout (0 to 1)",
                ),
                TrainState.lora_dropout_error,
            ),
            columns="repeat(3, 1fr)",
            gap="var(--space-5)",
            width="100%",
        ),
        _field(
            "Target modules (comma-separated)",
            rx.input(
                placeholder="q_proj, k_proj, v_proj, o_proj",
                value=TrainState.target_modules,
                on_change=TrainState.set_target_modules,
                size="2",
                disabled=TrainState.form_disabled,
                style={**_FIELD_STYLE, "width": "100%"},
                aria_label="LoRA target modules — comma-separated attention layer names",
            ),
            TrainState.target_modules_error,
        ),
        title="LoRA tuning",
    )


def _dataset_group() -> rx.Component:
    return Group(
        _field(
            "Dataset path",
            rx.input(
                placeholder="path/to/dataset.jsonl",
                default_value=TrainState.dataset_path,
                on_change=TrainState.set_dataset_path,
                size="2",
                disabled=TrainState.form_disabled,
                style={**_FIELD_STYLE, "width": "100%"},
                aria_label="Path to training dataset (JSONL)",
            ),
            TrainState.dataset_path_error,
        ),
        rx.text(
            "Format auto-detected from contents: Alpaca · ShareGPT · OpenAI · raw JSONL.",
            size="1",
            style={"color": "var(--bp-muted)"},
        ),
        title="Dataset",
    )


def _advanced_group() -> rx.Component:
    return Group(
        rx.grid(
            _field(
                "GPU temp threshold (°C)",
                rx.input(
                    placeholder="85",
                    value=TrainState.gpu_temp_threshold.to_string(),
                    on_change=TrainState.set_gpu_temp_threshold,
                    size="2",
                    class_name="bp-num",
                    disabled=TrainState.form_disabled,
                    style=_FIELD_STYLE,
                    aria_label=(
                        "GPU temperature threshold in Celsius "
                        "(pause training above this)"
                    ),
                ),
                TrainState.gpu_temp_threshold_error,
            ),
            _field(
                "W&B run name",
                rx.input(
                    placeholder="(optional)",
                    value=TrainState.wandb_run_name,
                    on_change=TrainState.set_wandb_run_name,
                    size="2",
                    disabled=TrainState.form_disabled,
                    style=_FIELD_STYLE,
                    aria_label="Weights and Biases run name (optional)",
                ),
                TrainState.wandb_run_name_error,
            ),
            columns="repeat(2, 1fr)",
            gap="var(--space-5)",
            width="100%",
        ),
        rx.flex(
            rx.checkbox(
                "Gradient checkpointing",
                checked=TrainState.gradient_checkpointing,
                on_change=TrainState.set_gradient_checkpointing,
                disabled=TrainState.form_disabled,
            ),
            rx.checkbox(
                "Flash attention",
                checked=TrainState.flash_attention,
                on_change=TrainState.set_flash_attention,
                disabled=TrainState.form_disabled,
            ),
            direction="row",
            gap="var(--space-5)",
        ),
        title="Advanced",
        collapsible=True,
        default_open=False,
    )


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
            style={"border_radius": "var(--bp-r-pill)", "min_width": "220px"},
            aria_label="Stop and save checkpoint",
        ),
        rx.button(
            "Start training",
            variant="solid",
            color_scheme="teal",
            size="3",
            disabled=TrainState.form_disabled,
            style={"border_radius": "var(--bp-r-pill)", "min_width": "220px"},
            on_click=TrainState.start_training,
            aria_label="Start training",
        ),
    )


def train_page() -> rx.Component:
    """The Train surface."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="train"),
            rx.scroll_area(
                rx.flex(
                    rx.flex(
                        rx.heading(
                            "Single run",
                            size="7",
                            style={
                                "color": "var(--bp-text)",
                                "font_weight": "600",
                                "letter_spacing": "-0.02em",
                            },
                        ),
                        rx.text(
                            "Configure a one-shot fine-tuning run. Sensible defaults "
                            "for Qwen 2.5 7B on a 32 GB GPU; smaller cards work at "
                            "lower batch sizes. Runs in a separate process — the "
                            "rail on the right shows live progress.",
                            size="2",
                            style={"color": "var(--bp-muted)"},
                        ),
                        direction="column",
                        gap="var(--space-2)",
                        width="100%",
                    ),
                    _reattach_banner(),
                    _refusal_callout(),
                    _error_callout(),
                    _run_progress_card(),
                    # Two-column form grid on wide screens; stacks at narrow.
                    rx.grid(
                        rx.flex(
                            _model_group(),
                            _dataset_group(),
                            direction="column",
                            gap="var(--space-6)",
                            width="100%",
                            align="start",
                        ),
                        rx.flex(
                            _training_shape_group(),
                            _lora_group(),
                            direction="column",
                            gap="var(--space-6)",
                            width="100%",
                            align="start",
                        ),
                        columns=rx.breakpoints(initial="1", md="2"),
                        gap="var(--space-6)",
                        width="100%",
                        align="start",
                    ),
                    _advanced_group(),
                    _loss_chart_card(),
                    _next_steps_panel(),
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
                    on_mount=[
                        TrainState.refresh_gpu,
                        TrainState.attach_active_job,
                    ],
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
