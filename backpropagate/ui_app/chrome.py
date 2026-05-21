"""Shared shell pieces — header, left nav, side rail, footer.

Phase 1: minimal placeholder content. Phase 2 polishes per the design digest's
shell layout (56px header, 188px left rail, 296px side rail, 32px footer).
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import AppState, TrainState

from .components.event_log import BpEventLog
from .components.gpu_ring import BpGpuRing
from .components.sparkline import BpSparkline
from .components.status_pill import BpStatusPill
from .components.vram_bar import BpVramBar

# Surface → (label, href) for the left nav.
_NAV_ITEMS = (
    ("train", "Single run", "/"),
    ("multi-run", "Multi-run", "/multi-run"),
    ("export", "Export", "/export"),
    ("dataset", "Dataset", "/dataset"),
)


def BpHeader() -> rx.Component:
    """The 56px header strip — logo / version / run_id / theme / gh."""
    return rx.flex(
        rx.flex(
            rx.heading("backpropagate", size="4"),
            rx.text("v1.1.0", size="1", style={"color": "var(--bp-muted)"}),
            gap="3",
            align="center",
        ),
        rx.spacer(),
        rx.flex(
            rx.text("run_id:", size="1", style={"color": "var(--bp-muted)"}),
            rx.text(
                AppState.run_id,
                size="1",
                style={"font_family": "var(--bp-mono)"},
            ),
            rx.button(
                "Toggle theme",
                size="1",
                variant="soft",
                on_click=AppState.toggle_theme,
            ),
            rx.link(
                "GitHub",
                href="https://github.com/mcp-tool-shop-org/backpropagate",
                size="1",
                is_external=True,
            ),
            gap="3",
            align="center",
        ),
        padding="3",
        height="56px",
        align="center",
        width="100%",
        style={
            "background": "var(--bp-surface)",
            "border_bottom": "1px solid var(--bp-border)",
        },
    )


def _nav_item(key: str, label: str, href: str) -> rx.Component:  # noqa: ARG001 — key reserved for Phase 2 active-state styling
    return rx.link(
        rx.box(
            rx.text(label, size="2"),
            padding="2",
            width="100%",
            style={"border_radius": "var(--bp-r-2)"},
        ),
        href=href,
        underline="none",
        width="100%",
    )


def BpLeftNav(active: str = "train") -> rx.Component:  # noqa: ARG001 — active reserved for Phase 2 highlight binding
    """188px left rail — tabs styled as a vertical nav."""
    return rx.flex(
        *(_nav_item(*item) for item in _NAV_ITEMS),
        direction="column",
        gap="1",
        width="188px",
        padding="3",
        height="100%",
        style={
            "background": "var(--bp-surface)",
            "border_right": "1px solid var(--bp-border)",
        },
    )


def BpSideRail() -> rx.Component:
    """296px right-side rail — status pill, sparkline, GPU ring, VRAM bar, log."""
    return rx.flex(
        BpStatusPill(state="idle", label="Idle", detail="no run active"),
        rx.divider(),
        rx.text("Loss · last 80 steps", size="1", style={"color": "var(--bp-muted)"}),
        BpSparkline(data=[]),
        rx.divider(),
        rx.flex(
            BpGpuRing(temp_c=0),
            BpVramBar(used_gb=0.0, total_gb=16.0),
            direction="column",
            gap="3",
            width="100%",
        ),
        rx.divider(),
        rx.text("Events", size="1", style={"color": "var(--bp-muted)"}),
        BpEventLog(events=[]),
        direction="column",
        gap="3",
        width="296px",
        padding="3",
        height="100%",
        style={
            "background": "var(--bp-surface)",
            "border_left": "1px solid var(--bp-border)",
            "overflow_y": "auto",
        },
    )


def BpFooter() -> rx.Component:
    """32px footer strip — run_id mirror, version, status text."""
    return rx.flex(
        rx.text(
            "backpropagate · v1.1.0",
            size="1",
            style={"color": "var(--bp-muted-2)"},
        ),
        rx.spacer(),
        rx.text(
            TrainState.run_state,
            size="1",
            style={"font_family": "var(--bp-mono)", "color": "var(--bp-muted)"},
        ),
        padding_x="3",
        padding_y="1",
        height="32px",
        align="center",
        width="100%",
        style={
            "background": "var(--bp-surface)",
            "border_top": "1px solid var(--bp-border)",
        },
    )
