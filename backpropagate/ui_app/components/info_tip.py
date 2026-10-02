"""``info_tip`` — the small "i" next to a label that explains it.

The web UI is for people who are curious, not only for people who already
know what a LoRA rank is. Every area carries one of these: hover it, or
focus it with the keyboard, and a card says what the thing is, what changing
it does and where to start (the words live in ``help_text.TIPS``).

How it behaves:

* The trigger is a real ``<button>``, so it is in the Tab order and has a
  focus ring. Its accessible name is "About <title>".
* The card opens on hover and on keyboard focus (Radix HoverCard) and is
  positioned by Radix, so it flips away from the window edge.
* The same words are always in the page as a visually hidden description of
  the button (``aria-describedby``), so a screen reader reads the tip when
  the button is focused, whether or not the card is drawn.
* Clicking the button does nothing else: it never submits or toggles.
"""

from __future__ import annotations

import reflex as rx

from ..help_text import Tip, tip

_GLYPH = (
    "<svg width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
    "stroke-width='2' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'>"
    "<circle cx='12' cy='12' r='9.5'/><path d='M12 11v6'/>"
    "<circle cx='12' cy='7.6' r='0.6' fill='currentColor'/></svg>"
)


def _card(t: Tip) -> rx.Component:
    rows: list[rx.Component] = [
        rx.text(
            t.title,
            weight="medium",
            style={"color": "var(--bp-text)", "font_size": "14px", "line_height": "1.35"},
        ),
        *(
            rx.text(
                paragraph,
                style={"color": "var(--bp-text-2)", "font_size": "13px", "line_height": "1.55"},
            )
            for paragraph in t.body
        ),
    ]
    if t.start:
        rows.append(
            rx.box(
                rx.text(
                    "Good starting point",
                    style={
                        "color": "var(--bp-teal)",
                        "font_size": "11px",
                        "font_weight": "600",
                        "letter_spacing": "0.04em",
                        "text_transform": "uppercase",
                    },
                ),
                rx.text(
                    t.start,
                    style={"color": "var(--bp-text)", "font_size": "13px", "line_height": "1.5"},
                ),
                class_name="bp-tip-start",
            )
        )
    if t.url:
        rows.append(
            rx.link(
                "Read more in the handbook",
                href=t.url,
                is_external=True,
                style={"color": "var(--bp-teal)", "font_size": "12px"},
            )
        )
    return rx.flex(*rows, direction="column", gap="8px")


def info_tip(key: str, *, side: str = "bottom") -> rx.Component:
    """The "i" button for ``help_text.TIPS[key]``."""
    t = tip(key)
    desc_id = f"bp-tip-{key.replace('_', '-')}"
    return rx.fragment(
        rx.hover_card.root(
            rx.hover_card.trigger(
                rx.el.button(
                    rx.html(_GLYPH),
                    type="button",
                    class_name="bp-tip",
                    aria_label=f"About: {t.title}",
                    aria_describedby=desc_id,
                    custom_attrs={"data-tip": key},
                ),
            ),
            rx.hover_card.content(
                _card(t),
                side=side,
                align="start",
                size="1",
                class_name="bp-tip-card",
                style={"max_width": "340px"},
            ),
            open_delay=120,
            close_delay=140,
        ),
        rx.el.span(t.text, id=desc_id, class_name="bp-visually-hidden"),
    )


def with_tip(label: rx.Component, key: str | None) -> rx.Component:
    """``label`` followed by its tip, on one line (or just the label)."""
    if not key:
        return label
    return rx.flex(label, info_tip(key), align="center", gap="6px")


__all__ = ["info_tip", "with_tip"]
