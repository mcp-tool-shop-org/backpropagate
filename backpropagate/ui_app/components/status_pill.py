"""``BpStatusPill`` — the "is this alive AND healthy?" indicator.

Phase 1: minimal — a coloured circle + label + detail text. No heartbeat
animation yet (Phase 2 layers that on per the design digest's 2.4s cadence).
"""

from __future__ import annotations

import reflex as rx

# State → dot colour map (CSS custom property names from ui_theme.TOKENS_CSS).
_STATE_COLOR = {
    "idle":    "var(--bp-muted-2)",
    "loading": "var(--bp-blue)",
    "active":  "var(--bp-teal)",
    "paused":  "var(--bp-amber)",
    "done":    "var(--bp-seafoam)",
    "error":   "var(--bp-peach)",
}


def BpStatusPill(state, label, detail: str = "") -> rx.Component:
    """A status pill with coloured dot + label + monospace detail line.

    Parameters
    ----------
    state:
        Either a string literal or a State.var resolving to one of the keys
        in ``_STATE_COLOR``. Reflex's reactive resolution handles both.
    label, detail:
        Strings or state.vars; the label sits beside the dot, the detail
        renders below in monospace, low emphasis.
    """
    # If we got a raw string, resolve the colour at build time. If we got a
    # state.var, defer to a CSS variable lookup (Reflex handles re-rendering).
    if isinstance(state, str):
        dot_color = _STATE_COLOR.get(state, "var(--bp-muted-2)")
    else:
        # Build a CSS class lookup that the runtime can switch via state. For
        # Phase 1 we just always use the teal dot when state is a var — the
        # animation/colour-by-state binding lands in Phase 2.
        dot_color = "var(--bp-teal)"

    return rx.flex(
        rx.box(
            width="10px",
            height="10px",
            border_radius="9999px",
            background=dot_color,
            margin_top="6px",
            flex_shrink="0",
        ),
        rx.flex(
            rx.text(label, size="2", weight="medium"),
            rx.text(
                detail,
                size="1",
                style={"font_family": "var(--bp-mono)", "color": "var(--bp-muted)"},
            ) if detail else rx.fragment(),
            direction="column",
            gap="1",
        ),
        gap="2",
        align="start",
        padding="2",
        style={"background": "var(--bp-surface-2)", "border_radius": "var(--bp-r-2)"},
    )
