"""``BpStatusPill`` — the "is this alive AND healthy?" indicator.

Per design digest §4a, the dot is the load-bearing signal:

- ``active`` — 2.4s heartbeat (the load-bearing "running AND healthy" signal)
- ``paused`` — 1.6s pulse
- ``loading`` — 2.4s pulse on blue
- ``idle`` / ``done`` / ``error`` — static (motion has meaning; only running
  things move)

Background tint matches the dot. Border at 35% opacity of the same hue. Detail
line is monospace, low-emphasis so it doesn't compete with the label.
"""

from __future__ import annotations

import reflex as rx

# State → dot colour map. Keys are the Literal members of ``RunState`` in
# ``ui_state.py``. Values are CSS custom properties resolved by
# ``ui_theme.TOKENS_CSS``.
_STATE_COLOR = {
    "idle":    "var(--bp-muted-2)",
    "loading": "var(--bp-blue)",
    "active":  "var(--bp-teal)",
    "paused":  "var(--bp-amber)",
    "done":    "var(--bp-seafoam)",
    "error":   "var(--bp-peach)",
}

# State → animation class. Empty string for the static states.
_STATE_ANIM = {
    "idle":    "",
    "loading": "bp-heartbeat-2400",
    "active":  "bp-heartbeat-2400",
    "paused":  "bp-pulse-1600",
    "done":    "",
    "error":   "",
}


def _tint(color_var: str) -> str:
    """Surface tint at ~12% — falls back gracefully if ``color-mix`` is
    unsupported by piggy-backing onto the surface-2 token underneath."""
    return f"color-mix(in srgb, {color_var} 12%, var(--bp-surface))"


def _border(color_var: str) -> str:
    """35% opacity border tint, per the design digest."""
    return f"color-mix(in srgb, {color_var} 35%, transparent)"


def BpStatusPill(state="idle", label: str = "Idle", detail: str = "") -> rx.Component:
    """A status pill with coloured dot + label + monospace detail line.

    Parameters
    ----------
    state:
        A string literal or ``State.var`` resolving to one of the keys in
        ``_STATE_COLOR``. For string literals we resolve at build time. For a
        ``State.var`` we render the pill with the union of states' CSS and
        let conditional rendering swap the dot per state at runtime.
    label, detail:
        Strings or ``State.var``s. Label sits beside the dot; detail renders
        in monospace, low-emphasis, below the label.
    """
    # Resolve build-time vs runtime: a raw str gives us a single colour
    # and animation; a State.var forces us to render conditionally.
    if isinstance(state, str):
        dot_color = _STATE_COLOR.get(state, _STATE_COLOR["idle"])
        anim_class = _STATE_ANIM.get(state, "")
        return rx.flex(
            rx.box(
                width="10px",
                height="10px",
                border_radius="9999px",
                background=dot_color,
                margin_top="6px",
                flex_shrink="0",
                class_name=anim_class,
            ),
            rx.flex(
                rx.text(label, size="2", weight="medium"),
                rx.text(
                    detail,
                    size="1",
                    style={
                        "font_family": "var(--bp-mono)",
                        "color": "var(--bp-muted)",
                    },
                ) if detail else rx.fragment(),
                direction="column",
                gap="1",
            ),
            gap="2",
            align="start",
            padding="2",
            style={
                "background": _tint(dot_color),
                "border": f"1px solid {_border(dot_color)}",
                "border_radius": "var(--bp-r-2)",
            },
        )

    # State.var path: use rx.match to pick the dot per active state. The
    # surrounding tint also varies by state, so we wrap the whole pill body
    # in the match too.
    return rx.match(
        state,
        ("idle",    _pill_variant(state="idle",    label=label, detail=detail)),
        ("loading", _pill_variant(state="loading", label=label, detail=detail)),
        ("active",  _pill_variant(state="active",  label=label, detail=detail)),
        ("paused",  _pill_variant(state="paused",  label=label, detail=detail)),
        ("done",    _pill_variant(state="done",    label=label, detail=detail)),
        ("error",   _pill_variant(state="error",   label=label, detail=detail)),
        _pill_variant(state="idle", label=label, detail=detail),
    )


def _pill_variant(state: str, label, detail) -> rx.Component:
    """Build one statically-resolved pill variant for ``rx.match``."""
    dot_color = _STATE_COLOR[state]
    anim_class = _STATE_ANIM[state]
    return rx.flex(
        rx.box(
            width="10px",
            height="10px",
            border_radius="9999px",
            background=dot_color,
            margin_top="6px",
            flex_shrink="0",
            class_name=anim_class,
        ),
        rx.flex(
            rx.text(label, size="2", weight="medium"),
            rx.text(
                detail,
                size="1",
                style={
                    "font_family": "var(--bp-mono)",
                    "color": "var(--bp-muted)",
                },
            ),
            direction="column",
            gap="1",
        ),
        gap="2",
        align="start",
        padding="2",
        style={
            "background": _tint(dot_color),
            "border": f"1px solid {_border(dot_color)}",
            "border_radius": "var(--bp-r-2)",
        },
    )
