"""``Group`` — section wrapper for form blocks.

Plain mode renders as a labelled box; collapsible mode wraps in an accordion.
"""

from __future__ import annotations

import reflex as rx


def Group(
    *children: rx.Component,
    title: str = "",
    collapsible: bool = False,
    default_open: bool = True,
) -> rx.Component:
    """A titled section. Pass children positionally."""
    body = rx.flex(*children, direction="column", gap="3", width="100%")
    if not collapsible:
        return rx.box(
            rx.heading(title, size="3", margin_bottom="3"),
            body,
            padding="4",
            style={
                "background": "var(--bp-surface)",
                "border": "1px solid var(--bp-border)",
                "border_radius": "var(--bp-r-3)",
            },
            width="100%",
        )
    return rx.accordion.root(
        rx.accordion.item(
            header=title,
            content=body,
            value=title,
        ),
        type="single",
        collapsible=True,
        default_value=title if default_open else None,
        width="100%",
    )
