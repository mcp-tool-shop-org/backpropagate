"""Multi-Run page — ``/multi-run`` — SLAO sweep surface.

Phase 1: minimal skeleton. Phase 2 fills in the per-run table with inline
trajectories per the design digest.
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import MultiRunState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.group import Group


def multi_run_page() -> rx.Component:
    """The Multi-Run surface."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="multi-run"),
            rx.scroll_area(
                rx.flex(
                    rx.heading("Multi-run", size="6"),
                    rx.text(
                        "SLAO sweep — train multiple runs and merge the LoRA "
                        "adapters to defeat catastrophic forgetting.",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    Group(
                        rx.text("[stub] model + quantization (shared across runs)"),
                        title="Model",
                    ),
                    Group(
                        rx.text("[stub] num_runs, samples_per_run, merge_mode"),
                        title="Sweep shape",
                    ),
                    Group(
                        rx.text("[stub] per-run table with inline loss trajectories"),
                        title="Runs",
                    ),
                    Group(
                        rx.text("[stub] cross-run metrics, comparison chart"),
                        title="Cross-run analysis",
                    ),
                    rx.button(
                        "Start multi-run",
                        size="3",
                        on_click=MultiRunState.start_multi_run,
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
