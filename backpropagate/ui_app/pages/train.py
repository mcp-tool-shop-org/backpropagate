"""Train page — ``/`` — the single-run surface.

Phase 1: minimal skeleton mirroring the design digest's Train component tree.
Each Group renders with one placeholder field so the layout is visible; Phase 2
fills in the real form controls + wires them to ``TrainState``.
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import TrainState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.group import Group


def train_page() -> rx.Component:
    """The Train surface."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="train"),
            rx.scroll_area(
                rx.flex(
                    rx.heading("Single run", size="6"),
                    rx.text(
                        "Configure a one-shot fine-tuning run. Visual polish "
                        "(refined Ocean Mist palette, heartbeat indicators, "
                        "live sparkline) lands in Phase 2.",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    Group(
                        rx.text("[stub] HF model + quantization"),
                        rx.input(
                            placeholder="HuggingFace model id",
                            default_value=TrainState.model,
                        ),
                        title="Model",
                    ),
                    Group(
                        rx.text("[stub] training shape group — steps, batch, LR"),
                        rx.input(
                            placeholder="steps",
                            default_value="100",
                        ),
                        title="Training shape",
                    ),
                    Group(
                        rx.text("[stub] LoRA r, alpha, dropout, target modules"),
                        title="LoRA tuning",
                    ),
                    Group(
                        rx.text("[stub] dataset path, format auto-detect"),
                        title="Dataset",
                    ),
                    Group(
                        rx.text("[stub] gpu safety thresholds, wandb run name"),
                        title="Advanced",
                        collapsible=True,
                        default_open=False,
                    ),
                    rx.flex(
                        rx.button(
                            "Start training",
                            size="3",
                            on_click=TrainState.start_training,
                        ),
                        rx.button(
                            "Stop",
                            size="3",
                            variant="soft",
                            on_click=TrainState.stop_training,
                        ),
                        gap="3",
                        margin_top="4",
                    ),
                    direction="column",
                    gap="4",
                    padding="6",
                    max_width="720px",
                ),
                flex_grow="1",
                style={"height": "100%"},
            ),
            BpSideRail(),
            flex_grow="1",
            width="100%",
            style={"overflow": "hidden"},
        ),
        BpFooter(),
        direction="column",
        height="100vh",
        width="100%",
    )
