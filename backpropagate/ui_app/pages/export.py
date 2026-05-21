"""Export page — ``/export`` — adapter / merged / GGUF export.

Phase 1: minimal skeleton. Phase 2 fills the 3-format radio + GGUF quant grid
+ Ollama register form per the design digest.
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import ExportState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.group import Group


def export_page() -> rx.Component:
    """The Export surface."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="export"),
            rx.scroll_area(
                rx.flex(
                    rx.heading("Export", size="6"),
                    rx.text(
                        "Convert a trained adapter into LoRA / merged / GGUF "
                        "and optionally register with Ollama.",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    Group(
                        rx.text("[stub] source adapter path"),
                        title="Source",
                    ),
                    Group(
                        rx.text("[stub] LoRA / merged / GGUF radio"),
                        title="Format",
                    ),
                    Group(
                        rx.text("[stub] GGUF quantization grid (q4_K_M default)"),
                        title="GGUF quantization",
                    ),
                    Group(
                        rx.text("[stub] ollama register checkbox + model name"),
                        title="Ollama",
                    ),
                    Group(
                        rx.text("[stub] output preview"),
                        title="Output",
                    ),
                    rx.button(
                        "Export",
                        size="3",
                        on_click=ExportState.start_export,
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
