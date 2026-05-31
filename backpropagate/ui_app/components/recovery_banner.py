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
    lead: str | rx.Var[str] = "",
    body: str | rx.Var[str] = "",
) -> rx.Component:
    """Render an inline recovery banner.

    Parameters
    ----------
    variant:
        One of ``"info"``, ``"warn"``, ``"ok"``. Unknown variants fall back
        to info. Static at composition time; not state-bound.
    lead:
        Bold lead-in. What happened. May be a literal ``str`` or a Reflex
        ``Var[str]`` bound to component state.
    body:
        Plain body. Why. Optional. May be a literal ``str`` or a Reflex
        ``Var[str]`` bound to component state (e.g.
        ``TrainState.latest_recovery_ok_msg``).
    """
    color, icon_url = _VARIANTS.get(variant, _VARIANTS["info"])

    # FRONTEND-B-005 (Stage C polish): SVG icons served as ``<img src=...>``
    # cannot be re-tinted via CSS ``color`` - the SVG fill is baked into the
    # file. The previous ``style={"color": color}`` was dead code (browsers
    # silently ignore it on ``<img>``). The variant accent is now carried
    # ONLY by the box border + background tint below, which IS the only
    # place the variant color visibly applies. The icon itself is whatever
    # color it ships with on disk - if a future polish pass wants accent-
    # tinted icons, swap to inline-SVG (rx.html with the SVG source) so the
    # ``currentColor`` fill picks up the parent's ``color``.
    return rx.box(
        rx.flex(
            rx.image(
                src=icon_url,
                width="16px",
                height="16px",
                style={
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
                # ``body`` may be a Reflex ``Var`` (e.g.
                # ``TrainState.latest_recovery_ok_msg``), so render it directly.
                # The old ``" " + body if body else ""`` branched on the Var's
                # truthiness in Python → VarTypeError at compile. Lead↔body
                # spacing is carried by the parent flex ``gap="1"`` below (no
                # manual leading space needed), and the sole caller
                # (``train._recovery_banners``) already guards each banner on
                # ``... != ""`` so ``body`` is non-empty whenever this renders.
                rx.text(
                    body,
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
        # FRONTEND-B-014-EXTENDED (Stage C accessibility): aria_atomic="true"
        # makes screen readers announce the entire banner (lead + body) as a
        # single coherent unit when its content changes — otherwise some AT
        # would only announce the changed run of text, stripping the
        # "Recovered." / "Heads-up." framing that gives the message context.
        aria_atomic="true",
    )
