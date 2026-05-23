"""Train page — ``/`` — the single-run surface.

Component tree per design digest §3:

- Hero heading + subtitle
- Recovery banners (when applicable)
- Group "Model"            — HF model id + quantization
- Group "Training shape"   — steps + batch + learning rate
- Group "LoRA tuning"      — r + alpha + dropout + target modules
- Group "Dataset"          — path + format auto-detect
- Group "Advanced" (collapsed) — gpu safety + run name + flags
- Start / Stop button row
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import TrainState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.group import Group


def _label(text: str) -> rx.Component:
    """Tiny labelled control eyebrow."""
    return rx.text(
        text,
        size="1",
        style={
            "color": "var(--bp-text-2)",
            "font_size": "11px",
            "margin_bottom": "4px",
        },
    )


def _model_group() -> rx.Component:
    return Group(
        rx.grid(
            rx.flex(
                _label("HuggingFace model id"),
                rx.input(
                    placeholder="meta-llama/Llama-3.1-8B",
                    default_value=TrainState.model,
                    on_change=TrainState.set_model,
                    size="2",
                    style={"width": "100%"},
                ),
                rx.cond(
                    TrainState.model_error != "",
                    rx.text(
                        TrainState.model_error,
                        size="1",
                        style={"color": "var(--bp-peach)", "font_size": "11px"},
                    ),
                    rx.fragment(),
                ),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Quantization"),
                rx.select.root(
                    rx.select.trigger(placeholder="4-bit", style={"width": "100%"}),
                    rx.select.content(
                        rx.select.item("4-bit", value="4-bit"),
                        rx.select.item("8-bit", value="8-bit"),
                        rx.select.item("16-bit", value="16-bit"),
                    ),
                    value=TrainState.quantization,
                    on_change=TrainState.set_quantization,
                ),
                direction="column",
                width="100%",
            ),
            columns="2fr 1fr",
            gap="3",
            width="100%",
        ),
        title="Model",
    )


def _err_text(error_var) -> rx.Component:
    """Inline error label — peach text, 11px, only renders when non-empty."""
    return rx.cond(
        error_var != "",
        rx.text(
            error_var,
            size="1",
            style={"color": "var(--bp-peach)", "font_size": "11px"},
        ),
        rx.fragment(),
    )


def _training_shape_group() -> rx.Component:
    return Group(
        rx.grid(
            rx.flex(
                _label("Steps"),
                rx.input(
                    placeholder="100",
                    value=TrainState.steps.to_string(),
                    on_change=TrainState.set_steps,
                    type="number",
                    size="2",
                    class_name="bp-num",
                ),
                _err_text(TrainState.steps_error),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Batch size"),
                rx.input(
                    placeholder="auto",
                    value=TrainState.batch_size,
                    on_change=TrainState.set_batch_size,
                    size="2",
                    class_name="bp-num",
                ),
                _err_text(TrainState.batch_size_error),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Learning rate"),
                rx.input(
                    placeholder="2e-4",
                    value=TrainState.learning_rate.to_string(),
                    on_change=TrainState.set_learning_rate,
                    size="2",
                    class_name="bp-num",
                ),
                _err_text(TrainState.learning_rate_error),
                direction="column",
                width="100%",
            ),
            columns="repeat(3, 1fr)",
            gap="3",
            width="100%",
        ),
        title="Training shape",
    )


def _lora_group() -> rx.Component:
    return Group(
        rx.grid(
            rx.flex(
                _label("LoRA rank"),
                rx.input(
                    placeholder="16",
                    value=TrainState.lora_r.to_string(),
                    on_change=TrainState.set_lora_r,
                    type="number",
                    size="2",
                    class_name="bp-num",
                ),
                _err_text(TrainState.lora_r_error),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("LoRA alpha"),
                rx.input(
                    placeholder="32",
                    value=TrainState.lora_alpha.to_string(),
                    on_change=TrainState.set_lora_alpha,
                    size="2",
                    class_name="bp-num",
                ),
                _err_text(TrainState.lora_alpha_error),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Dropout"),
                rx.input(
                    placeholder="0.05",
                    value=TrainState.lora_dropout.to_string(),
                    on_change=TrainState.set_lora_dropout,
                    size="2",
                    class_name="bp-num",
                ),
                _err_text(TrainState.lora_dropout_error),
                direction="column",
                width="100%",
            ),
            columns="repeat(3, 1fr)",
            gap="3",
            width="100%",
        ),
        rx.flex(
            _label("Target modules (comma-separated)"),
            rx.input(
                placeholder="q_proj, k_proj, v_proj, o_proj",
                value=TrainState.target_modules,
                on_change=TrainState.set_target_modules,
                size="2",
                style={"width": "100%"},
            ),
            _err_text(TrainState.target_modules_error),
            direction="column",
            width="100%",
        ),
        title="LoRA tuning",
    )


def _dataset_group() -> rx.Component:
    return Group(
        rx.flex(
            _label("Dataset path"),
            rx.input(
                placeholder="path/to/dataset.jsonl",
                default_value=TrainState.dataset_path,
                on_change=TrainState.set_dataset_path,
                size="2",
                style={"width": "100%"},
            ),
            rx.cond(
                TrainState.dataset_path_error != "",
                rx.text(
                    TrainState.dataset_path_error,
                    size="1",
                    style={"color": "var(--bp-peach)", "font_size": "11px"},
                ),
                rx.fragment(),
            ),
            direction="column",
            width="100%",
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
            rx.flex(
                _label("GPU temp threshold (°C)"),
                rx.input(
                    placeholder="85",
                    value=TrainState.gpu_temp_threshold.to_string(),
                    on_change=TrainState.set_gpu_temp_threshold,
                    size="2",
                    class_name="bp-num",
                ),
                _err_text(TrainState.gpu_temp_threshold_error),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("W&B run name"),
                rx.input(
                    placeholder="(optional)",
                    value=TrainState.wandb_run_name,
                    on_change=TrainState.set_wandb_run_name,
                    size="2",
                ),
                _err_text(TrainState.wandb_run_name_error),
                direction="column",
                width="100%",
            ),
            columns="repeat(2, 1fr)",
            gap="3",
            width="100%",
        ),
        rx.flex(
            rx.checkbox(
                "Gradient checkpointing",
                checked=TrainState.gradient_checkpointing,
                on_change=TrainState.set_gradient_checkpointing,
            ),
            rx.checkbox(
                "Flash attention",
                checked=TrainState.flash_attention,
                on_change=TrainState.set_flash_attention,
            ),
            direction="row",
            gap="4",
        ),
        title="Advanced",
        collapsible=True,
        default_open=False,
    )


def train_page() -> rx.Component:
    """The Train surface."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="train"),
            rx.scroll_area(
                rx.flex(
                    rx.heading(
                        "Single run",
                        size="6",
                        style={"color": "var(--bp-text)", "font_weight": "500"},
                    ),
                    rx.text(
                        "Configure a one-shot fine-tuning run. Sensible defaults "
                        "for Qwen 2.5 7B on a 16 GB GPU; tweak any field below.",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    _model_group(),
                    _training_shape_group(),
                    _lora_group(),
                    _dataset_group(),
                    _advanced_group(),
                    rx.flex(
                        rx.button(
                            rx.cond(
                                TrainState.run_state == "loading",
                                rx.spinner(size="2"),
                                rx.fragment(),
                            ),
                            rx.cond(
                                TrainState.run_state == "loading",
                                rx.text("Starting…"),
                                rx.text("Start training"),
                            ),
                            variant="solid",
                            color_scheme="teal",
                            size="3",
                            # FRONTEND-B-003: disable while a run is in flight so
                            # double-clicks don't queue duplicate trainer jobs.
                            disabled=(TrainState.run_state == "loading")
                            | (TrainState.run_state == "active"),
                            on_click=TrainState.start_training,
                        ),
                        rx.button(
                            "Stop",
                            variant="soft",
                            color_scheme="gray",
                            size="3",
                            # Stop is meaningless when nothing is running.
                            disabled=(TrainState.run_state == "idle")
                            | (TrainState.run_state == "done")
                            | (TrainState.run_state == "error"),
                            on_click=TrainState.stop_training,
                        ),
                        gap="3",
                        margin_top="2",
                        align="center",
                    ),
                    direction="column",
                    gap="4",
                    padding="6",
                    max_width="780px",
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
