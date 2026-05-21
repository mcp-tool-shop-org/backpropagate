"""``BpEventLog`` — last-N entries from the live event stream.

Phase 1: static list. Phase 2 wires it to ``TrainState.events`` via
``rx.foreach`` and adds the level→colour dot per the design digest.
"""

from __future__ import annotations

import reflex as rx

# Level → dot colour, matching the design digest event-log surface.
_LEVEL_COLOR = {
    "info":  "var(--bp-text-2)",
    "ok":    "var(--bp-seafoam)",
    "warn":  "var(--bp-amber)",
    "err":   "var(--bp-peach)",
    "tx":    "var(--bp-teal)",
    "hf":    "var(--bp-blue)",
}


def _row(entry: dict) -> rx.Component:
    """Render one log row: timestamp · coloured dot · message."""
    level = entry.get("level", "info")
    color = _LEVEL_COLOR.get(level, "var(--bp-text-2)")
    return rx.flex(
        rx.text(
            entry.get("t", ""),
            size="1",
            style={"font_family": "var(--bp-mono)", "color": "var(--bp-muted-2)"},
        ),
        rx.box(
            width="6px",
            height="6px",
            border_radius="9999px",
            background=color,
            margin_top="6px",
            flex_shrink="0",
        ),
        rx.text(
            entry.get("msg", ""),
            size="1",
            style={"font_family": "var(--bp-mono)", "color": color},
        ),
        gap="2",
        align="start",
    )


def BpEventLog(events: list[dict] | None = None, max_n: int = 6) -> rx.Component:
    """Render the last ``max_n`` events as a column of monospace rows.

    Phase 1 caveat: ``events`` is resolved at build time. For a state.var-backed
    list, Phase 2 swaps this for ``rx.foreach(state.events[-max_n:], _row)``.
    """
    entries = events or []
    tail = entries[-max_n:]
    if not tail:
        return rx.text(
            "No events yet",
            size="1",
            style={"color": "var(--bp-muted-2)", "font_style": "italic"},
        )
    return rx.vstack(
        *(_row(e) for e in tail),
        align="start",
        gap="1",
    )
