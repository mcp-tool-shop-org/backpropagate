"""``BpErrorCallout`` — structured error display.

Reading order per design digest §4d: code · title · message · hint · action.

- Code: monospace, ``var(--bp-peach)``, uppercase, 0.04em letter-spacing.
  Stable identifier (e.g. ``BackpropError · E_VRAM_EXHAUSTED``).
- Title: 13px, ``var(--bp-text)``.
- Message: 12px, ``var(--bp-text)``.
- Hint: same row as message; embeds ``<code>`` for token-level identifiers
  (``batch_size``, ``gradient_checkpointing``).
- **Apply hint** button when ``apply_action`` is provided.
- Stack trace folded behind a ``<details>`` element — present but not loud.
"""

from __future__ import annotations

import reflex as rx


def BpErrorCallout(
    code: str = "",
    title: str = "",
    message: str = "",
    hint: str = "",
    apply_action=None,
    stack_trace: str | None = None,
) -> rx.Component:
    """Render a structured error callout.

    Parameters
    ----------
    code:
        Stable error identifier, e.g. ``BackpropError · E_VRAM_EXHAUSTED``.
        Rendered first; sets reading order.
    title:
        One-line title under the code. Defaults to empty so the message
        carries the weight when there's no separate title.
    message:
        The error body. 12px, ``var(--bp-text)``.
    hint:
        Operator-facing suggestion. May contain inline code references; for
        richer formatting pass a pre-built component as ``message`` instead.
    apply_action:
        Event handler for the "Apply hint" button. When provided, the button
        renders inline below the hint.
    stack_trace:
        When non-empty, renders behind a folded ``<details>`` element so the
        callout stays compact by default.
    """
    children: list[rx.Component] = []

    if code:
        children.append(
            rx.text(
                code,
                size="1",
                style={
                    "font_family": "var(--bp-mono)",
                    "color": "var(--bp-peach)",
                    "text_transform": "uppercase",
                    "letter_spacing": "0.04em",
                    "font_size": "11px",
                    "margin_bottom": "2px",
                },
            )
        )

    if title:
        children.append(
            rx.text(
                title,
                size="2",
                weight="medium",
                style={"color": "var(--bp-text)", "font_size": "13px"},
            )
        )

    if message:
        children.append(
            rx.text(
                message,
                size="2",
                style={"color": "var(--bp-text)", "font_size": "12px"},
            )
        )

    if hint:
        children.append(
            rx.text(
                hint,
                size="2",
                style={"color": "var(--bp-text-2)", "font_size": "12px"},
            )
        )

    if apply_action is not None:
        children.append(
            rx.button(
                "Apply hint",
                size="1",
                variant="soft",
                color_scheme="orange",
                on_click=apply_action,
                style={"align_self": "flex-start", "margin_top": "4px"},
            )
        )

    if stack_trace:
        # Escape the raw trace text so '<' / '>' / '&' don't render as HTML.
        # The surrounding ``<details>`` keeps the native disclosure semantics
        # (Radix has no equivalent primitive). The trace itself is internal
        # observability surface, never user-supplied, but defence in depth.
        escaped = (
            stack_trace.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        stack_html = (
            '<details style="margin-top: 8px;">'
            '<summary style="cursor: pointer; color: var(--bp-muted); '
            'font-size: 11px; font-family: var(--bp-mono);">'
            'Stack trace</summary>'
            '<pre style="margin: 6px 0 0; padding: 8px; '
            'background: var(--bp-surface-2); border-radius: 4px; '
            'font-family: var(--bp-mono); font-size: 11px; '
            'color: var(--bp-text-2); overflow-x: auto; '
            f'white-space: pre-wrap;">{escaped}</pre>'
            '</details>'
        )
        children.append(rx.html(stack_html))

    return rx.callout.root(
        rx.flex(
            *children,
            direction="column",
            gap="1",
            width="100%",
        ),
        color_scheme="red",
        variant="surface",
        size="2",
    )
