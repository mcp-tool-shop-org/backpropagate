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
    # FRONTEND-A-004 (v1.4 Wave 2): the runs / models / run-detail pages bind a
    # state ``message`` (e.g. ``RunsState.error``), so every optional field is
    # rendered with ``rx.cond`` rather than a Python ``if`` — ``if message:`` on
    # a Reflex ``Var`` raises VarTypeError at compile. ``rx.cond(field != "")``
    # handles literals and Vars alike; an empty field collapses to
    # ``rx.fragment()`` (no DOM), preserving the design-digest §4d reading
    # order: code · title · message · hint · action.
    children: list[rx.Component] = [
        rx.cond(
            code != "",
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
            ),
            rx.fragment(),
        ),
        rx.cond(
            title != "",
            rx.text(
                title,
                size="2",
                weight="medium",
                style={"color": "var(--bp-text)", "font_size": "13px"},
            ),
            rx.fragment(),
        ),
        rx.cond(
            message != "",
            rx.text(
                message,
                size="2",
                style={"color": "var(--bp-text)", "font_size": "12px"},
            ),
            rx.fragment(),
        ),
        rx.cond(
            hint != "",
            rx.text(
                hint,
                size="2",
                style={"color": "var(--bp-text-2)", "font_size": "12px"},
            ),
            rx.fragment(),
        ),
    ]

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

    # ``stack_trace`` defaults to ``None`` (identity check — Var-safe); when a
    # value is present (str or Var) render the folded disclosure reactively so
    # a state-bound trace can't trip a Python truthiness check.
    if stack_trace is not None:
        children.append(
            rx.cond(
                stack_trace != "",
                # FRONTEND-A-004: native Reflex primitives instead of
                # ``rx.html``. Using ``rx.el.details`` + ``rx.el.summary`` +
                # ``rx.el.pre`` preserves the native ``<details>`` disclosure
                # semantics while routing the trace text through Reflex's
                # normal escape pipeline. The previous ``rx.html`` chrome
                # required hand-rolled HTML escaping; this removes the
                # raw-HTML surface and the precedent that came with it.
                rx.el.details(
                    rx.el.summary(
                        "Stack trace",
                        style={
                            "cursor": "pointer",
                            "color": "var(--bp-muted)",
                            "font_size": "11px",
                            "font_family": "var(--bp-mono)",
                        },
                    ),
                    rx.el.pre(
                        stack_trace,
                        style={
                            "margin": "6px 0 0",
                            "padding": "8px",
                            "background": "var(--bp-surface-2)",
                            "border_radius": "4px",
                            "font_family": "var(--bp-mono)",
                            "font_size": "11px",
                            "color": "var(--bp-text-2)",
                            "overflow_x": "auto",
                            "white_space": "pre-wrap",
                        },
                    ),
                    style={"margin_top": "8px"},
                ),
                rx.fragment(),
            )
        )

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
