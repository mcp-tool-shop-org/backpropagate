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
                    size="2",
                    style={"width": "100%"},
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
                    default_value=TrainState.quantization,
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


def _training_shape_group() -> rx.Component:
    return Group(
        rx.grid(
            rx.flex(
                _label("Steps"),
                rx.input(
                    placeholder="100",
                    default_value="100",
                    type="number",
                    size="2",
                    class_name="bp-num",
                ),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Batch size"),
                rx.input(
                    placeholder="auto",
                    default_value=TrainState.batch_size,
                    size="2",
                    class_name="bp-num",
                ),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Learning rate"),
                rx.input(
                    placeholder="2e-4",
                    default_value="2e-4",
                    size="2",
                    class_name="bp-num",
                ),
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
                    default_value="16",
                    type="number",
                    size="2",
                    class_name="bp-num",
                ),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("LoRA alpha"),
                rx.input(
                    placeholder="32",
                    default_value="32",
                    size="2",
                    class_name="bp-num",
                ),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Dropout"),
                rx.input(
                    placeholder="0.05",
                    default_value="0.05",
                    size="2",
                    class_name="bp-num",
                ),
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
                default_value="q_proj, k_proj, v_proj, o_proj",
                size="2",
                style={"width": "100%"},
            ),
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
                size="2",
                style={"width": "100%"},
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
                    default_value="85",
                    size="2",
                    class_name="bp-num",
                ),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("W&B run name"),
                rx.input(
                    placeholder="(optional)",
                    size="2",
                ),
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
                default_checked=True,
            ),
            rx.checkbox(
                "Flash attention",
                default_checked=True,
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
                            "Start training",
                            variant="solid",
                            color_scheme="teal",
                            size="3",
                            on_click=TrainState.start_training,
                        ),
                        rx.button(
                            "Stop",
                            variant="soft",
                            color_scheme="gray",
                            size="3",
                            on_click=TrainState.stop_training,
                        ),
                        gap="3",
                        margin_top="2",
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
