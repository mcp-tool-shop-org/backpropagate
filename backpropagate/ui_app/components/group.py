"""``Group`` — section wrapper for form blocks.

Plain mode: ``rx.box`` with an eyebrow heading above the content.
Collapsible mode: ``rx.accordion.root`` single-item with the same eyebrow on
the trigger.

The eyebrow is the small-caps section label from the design digest's foundation
sheet — ``var(--bp-text-2)``, uppercase, ``letter-spacing: 0.06em``, 11px.
"""

from __future__ import annotations

import reflex as rx


def _eyebrow(title: str) -> rx.Component:
    """The shared small-caps section label."""
    return rx.text(
        title,
        size="1",
        weight="medium",
        style={
            "color": "var(--bp-text-2)",
            "text_transform": "uppercase",
            "letter_spacing": "0.06em",
            "font_size": "11px",
            "line_height": "1",
        },
    )


def Group(
    *children: rx.Component,
    title: str = "",
    collapsible: bool = False,
    default_open: bool = True,
) -> rx.Component:
    """A titled section. Pass children positionally.

    Parameters
    ----------
    title:
        Section eyebrow. Rendered in small-caps above the content (or as the
        accordion trigger label in collapsible mode).
    collapsible:
        When ``True``, wraps in an accordion so the user can fold the section.
    default_open:
        Initial accordion state. Ignored in plain mode. Per the design digest,
        ``"Model"`` / ``"Training shape"`` / ``"LoRA tuning"`` default open;
        ``"Advanced"`` defaults closed.
    """
    body = rx.flex(*children, direction="column", gap="3", width="100%")

    if not collapsible:
        return rx.box(
            _eyebrow(title),
            rx.box(body, margin_top="3"),
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
            header=_eyebrow(title),
            content=rx.box(body, padding_top="3"),
            value=title or "section",
        ),
        type="single",
        collapsible=True,
        default_value=(title or "section") if default_open else "",
        width="100%",
        style={
            "background": "var(--bp-surface)",
            "border": "1px solid var(--bp-border)",
            "border_radius": "var(--bp-r-3)",
            "padding": "12px 16px",
        },
    )
