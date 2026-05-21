"""``BpErrorCallout`` — structured error display.

Reading order per design digest Section 4d: code · title · message · hint · action.
"""

from __future__ import annotations

import reflex as rx


def BpErrorCallout(
    code: str = "",
    message: str = "",
    hint: str = "",
    apply_action: rx.event.EventSpec | None = None,
) -> rx.Component:
    """Render a structured error callout.

    Phase 1: static fields. Phase 2 wires the "Apply hint" button + the
    ``<details>`` for stack trace per design digest 4d.
    """
    children: list[rx.Component] = []
    if code:
        children.append(
            rx.text(
                code,
                size="1",
                weight="bold",
                style={
                    "font_family": "var(--bp-mono)",
                    "color": "var(--bp-peach)",
                    "text_transform": "uppercase",
                },
            )
        )
    if message:
        children.append(rx.text(message, size="2"))
    if hint:
        children.append(
            rx.text(hint, size="1", style={"color": "var(--bp-muted)"})
        )
    if apply_action is not None:
        children.append(
            rx.button("Apply hint", size="1", on_click=apply_action)
        )
    return rx.callout.root(
        rx.flex(*children, direction="column", gap="2"),
        color_scheme="red",
        size="2",
    )
