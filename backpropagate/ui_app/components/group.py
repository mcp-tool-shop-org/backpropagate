"""``Group`` — section card for form blocks (ui-v2 P1 redesign).

The Director's bar: content lives in ROUNDED cards (14px) with a soft
shadow on a slightly different page background; section titles are 16px
semibold (not small-caps eyebrows); the single 4/8/12/16/24/32/48 spacing
scale governs padding and gaps.

Collapsible mode uses a native ``<details>/<summary>`` fold: JS-free,
keyboard-accessible, and it can't reintroduce the solid-teal Radix
accordion bar that read as a primary button in draft2.
"""

from __future__ import annotations

import reflex as rx


def _eyebrow(title: str) -> rx.Component:
    """Small-caps label — kept for collapsible triggers/secondary places."""
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


def _section_title(title: str) -> rx.Component:
    """The 16px semibold section title (ui-v2 P1 redesign type scale)."""
    return rx.text(
        title,
        size="3",
        weight="medium",
        style={
            "color": "var(--bp-text)",
            "font_size": "16px",
            "line_height": "1.3",
            "letter_spacing": "-0.01em",
        },
    )


def _card_style() -> dict[str, str]:
    """The shared card treatment: rounded, soft shadow, surface on page bg."""
    return {
        "background": "var(--bp-surface)",
        "border": "1px solid var(--bp-border)",
        "border_radius": "var(--bp-r-lg)",
        "box_shadow": "var(--bp-shadow-card)",
    }


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
    body = rx.flex(*children, direction="column", gap="var(--space-5)", width="100%")

    if not collapsible:
        return rx.box(
            _section_title(title),
            rx.box(body, margin_top="var(--space-4)"),
            padding="var(--space-5)",
            style=_card_style(),
            width="100%",
        )

    # ui-v2 P1 visual foundation: native <details>/<summary> disclosure.
    # The Radix accordion trigger rendered as a full-width SOLID TEAL bar
    # (draft2 + P1 screenshots) that read as a primary button, not a fold.
    # <details> is JS-free, keyboard-accessible, and styles cleanly with our
    # tokens: surface card, left-aligned eyebrow, chevron at the right.
    summary = rx.el.summary(
        rx.flex(
            _section_title(title),
            rx.spacer(),
            rx.html(
                "<svg width='14' height='14' viewBox='0 0 24 24' fill='none' "
                "stroke='currentColor' stroke-width='2' stroke-linecap='round'>"
                "<path d='m6 9 6 6 6-6'/></svg>",
                class_name="bp-accordion-chevron",
            ),
            align="center",
            width="100%",
        ),
        style={
            "list_style": "none",
            "cursor": "pointer",
            "user_select": "none",
            "color": "var(--bp-text)",
        },
    )
    if default_open:
        return rx.el.details(
            summary,
            rx.box(body, padding_top="var(--space-4)"),
            open=True,
            class_name="bp-accordion",
            width="100%",
            style={**_card_style(), "padding": "16px 20px"},
        )
    return rx.el.details(
        summary,
        rx.box(body, padding_top="var(--space-4)"),
        class_name="bp-accordion",
        width="100%",
        style={**_card_style(), "padding": "16px 20px"},
    )
