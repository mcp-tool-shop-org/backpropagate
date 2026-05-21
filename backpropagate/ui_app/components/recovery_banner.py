"""``BpRecoveryBanner`` — inline banner for self-recovered events.

Per design digest §4e. Three variants — info / warn / ok — each with a tinted
background, a 35%-opacity border in the same hue, and an icon at the left.

Bolded lead-in tells the user what HAPPENED. Body explains WHY.
Never red for recovered events: recovery is good news, even if the original
event wasn't.

Inline placement (NOT floating). Persists for the run. Floating toasts are
reserved for "click to undo" actions.
"""

from __future__ import annotations

import reflex as rx

# Variant → (accent CSS var, icon URL). Info + warn use the info circle;
# the ok variant uses the checkmark.
_VARIANTS: dict[str, tuple[str, str]] = {
    "info": ("var(--bp-blue)",    "/icons/info.svg"),
    "warn": ("var(--bp-amber)",   "/icons/info.svg"),
    "ok":   ("var(--bp-seafoam)", "/icons/check.svg"),
}


def BpRecoveryBanner(
    variant: str = "info",
    lead: str = "",
    body: str = "",
) -> rx.Component:
    """Render an inline recovery banner.

    Parameters
    ----------
    variant:
        One of ``"info"``, ``"warn"``, ``"ok"``. Unknown variants fall back
        to info.
    lead:
        Bold lead-in. What happened. Required.
    body:
        Plain body. Why. Optional.
    """
    color, icon_url = _VARIANTS.get(variant, _VARIANTS["info"])

    return rx.box(
        rx.flex(
            rx.image(
                src=icon_url,
                width="16px",
                height="16px",
                style={
                    "color": color,
                    "filter": "none",
                    "flex_shrink": "0",
                    "margin_top": "2px",
                },
                alt="",
            ),
            rx.flex(
                rx.text(
                    lead,
                    weight="bold",
                    size="2",
                    style={"color": "var(--bp-text)", "display": "inline"},
                ),
                rx.text(
                    " " + body if body else "",
                    size="2",
                    style={"color": "var(--bp-text-2)", "display": "inline"},
                ),
                direction="row",
                wrap="wrap",
                gap="1",
            ),
            gap="2",
            align="start",
            width="100%",
        ),
        padding="3",
        width="100%",
        style={
            "background": f"color-mix(in srgb, {color} 8%, var(--bp-surface))",
            "border": f"1px solid color-mix(in srgb, {color} 35%, transparent)",
            "border_radius": "var(--bp-r-2)",
        },
        role="status",
        aria_live="polite",
    )
