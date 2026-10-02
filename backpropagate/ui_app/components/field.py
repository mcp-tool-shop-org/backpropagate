"""Shared form-field chrome (ui-v2 P1 redesign pass 2).

One design language for every page's forms:

- controls sit on ``--bp-field-bg`` (one step deeper than the card they're
  in), ``--bp-r-md`` radius, 38px tall;
- labels are 13px ``--bp-text-2``, 6px above their control;
- validation errors render peach at 11px, only when non-empty.

Introduced for train.py in P1 pass 1; promoted here so multi-run, export and
dataset stop falling back to stock Radix chrome on the same screens.
"""

from __future__ import annotations

import reflex as rx

FIELD_STYLE: dict[str, str] = {
    "border_radius": "var(--bp-r-md)",
    "background": "var(--bp-field-bg)",
    "height": "38px",
}


def bp_label(text: str) -> rx.Component:
    """Field label — 13px, sits 6px above its control."""
    return rx.text(
        text,
        size="1",
        style={
            "color": "var(--bp-text-2)",
            "font_size": "13px",
            "margin_bottom": "2px",
        },
    )


def bp_err_text(error_var) -> rx.Component:
    """Inline error label — peach text, 11px, only renders when non-empty."""
    return rx.cond(
        error_var != "",
        rx.text(
            error_var,
            size="1",
            style={"color": "var(--bp-peach)", "font_size": "11px"},
        ),
        rx.fragment(),
    )


def bp_field(label: str, control: rx.Component, error_var=None) -> rx.Component:
    """A labelled form field (6px label gap, optional error slot)."""
    children = [bp_label(label), control]
    if error_var is not None:
        children.append(bp_err_text(error_var))
    return rx.flex(*children, direction="column", gap="1", width="100%")


__all__ = ["FIELD_STYLE", "bp_label", "bp_err_text", "bp_field"]
