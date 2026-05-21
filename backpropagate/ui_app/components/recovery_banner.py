"""``BpRecoveryBanner`` — inline banner for self-recovered events.

Three variants per design digest Section 4e: ``info``, ``warn``, ``ok``.
"""

from __future__ import annotations

import reflex as rx

# Variant → (color_scheme, icon hint)
_VARIANTS = {
    "info": "blue",
    "warn": "amber",
    "ok":   "green",
}


def BpRecoveryBanner(
    variant: str = "info",
    lead: str = "",
    body: str = "",
) -> rx.Component:
    """Render an inline recovery banner.

    Bold ``lead`` says what happened, plain ``body`` explains why. Never red:
    recovery is good news per the digest's rule.
    """
    color = _VARIANTS.get(variant, "blue")
    return rx.callout.root(
        rx.flex(
            rx.text(lead, weight="bold", size="2"),
            rx.text(body, size="2"),
            direction="column",
            gap="1",
        ),
        color_scheme=color,
        size="2",
    )
