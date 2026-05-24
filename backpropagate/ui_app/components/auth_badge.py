"""BpAuthBadge - the footer auth-mode chip.

Per V1_3_BRIEF P0 item 1 (FRONTEND-F-FOOTER-AUTH-BADGE) and Stage C
humanization. Shows the active auth posture as a colored badge with a
tooltip the operator can hover for the full context (bind host:port,
reachability, auth scheme + username if any).

Six states:

| State              | Color | Text                  |
|--------------------|-------|-----------------------|
| no_auth_local      | green | Local - no auth       |
| token_local        | green | Local - token         |
| basic_local        | green | Local - Basic         |
| basic_shared       | amber | Shared - Basic        |
| basic_network      | amber | Network - Basic       |
| insecure           | red   | INSECURE - no auth    |

Implementation notes:

- The badge READS ``AuthBadgeState`` (populated from
  ``ui_security.get_auth_badge_context()`` at server boot); it does NOT
  participate in the auth decision. All GHSA-f65r-h4g3-3h9h contracts
  remain in ``ui_app/auth.py``.
- ``aria_label`` mirrors the hover-text verbatim so vision-impaired
  operators get the same context a sighted operator gets from the tooltip
  (Stage C accessibility floor).
- Color is mapped to a Radix ``color_scheme`` (green / amber / red - the
  three Radix tints that read clearly across both dark and light themes
  per the design digest token doc).

FRONTEND-F-002 (Wave 5.5): the badge now READS ``bind_host`` /
``bind_port`` / ``reachable_from`` from ``AuthBadgeState`` and surfaces
them as a compact ``host:port`` chip immediately to the right of the
auth-mode label. Pre-fix the badge populated the 3 fields but never
rendered them — operators saw the auth mode but not the load-bearing
"WHERE is this UI reachable from" context. The tooltip continues to
carry the full multi-clause hover text (mode + bind + reach + auth).
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import AuthBadgeState

# Radix color_scheme values keyed by the badge state's ``mode_color`` field.
# Kept as a tuple of (color_var_key, radix_scheme) for clarity; we use a
# rx.match below since rx.Var values can't index a Python dict directly.
_GREEN_SCHEME = "green"
_AMBER_SCHEME = "amber"
_RED_SCHEME = "red"


def _color_scheme() -> rx.Var[str]:
    """Map AuthBadgeState.mode_color (Python string in state) to a Radix
    color_scheme. ``rx.match`` evaluates client-side so the badge re-tints
    without a round-trip if the env changes (it shouldn't, but cheap).
    """
    return rx.match(
        AuthBadgeState.mode_color,
        ("green", _GREEN_SCHEME),
        ("amber", _AMBER_SCHEME),
        ("red", _RED_SCHEME),
        _GREEN_SCHEME,  # default fallback
    )


def _emoji_for_mode() -> rx.Var[str]:
    """Lead glyph by mode_key. The emojis are paired with the mode_text in
    AuthBadgeState (e.g. mode_text='Local - Basic', emoji='LOCK') so the
    full chip label reads as one phrase.

    We render the emoji as a separate ``rx.text`` rather than embedding it
    in mode_text so the aria-label can carry the full English context
    without the glyph leaking into screen-reader output (the chip's
    aria-label comes from hover_text, which already names the auth scheme
    in plain words).
    """
    return rx.match(
        AuthBadgeState.mode_key,
        ("no_auth_local", "[OPEN]"),
        ("token_local", "[KEY]"),
        ("basic_local", "[LOCK]"),
        ("basic_shared", "[SHARED]"),
        ("basic_network", "[NETWORK]"),
        ("insecure", "[ALERT]"),
        "",
    )


def _bind_chip() -> rx.Component:
    """Compact ``host:port · reach`` chip rendered after the mode label.

    FRONTEND-F-002 (Wave 5.5): the load-bearing surface that exposes the
    three ``bind_host`` / ``bind_port`` / ``reachable_from`` fields that
    Stage C wired into ``AuthBadgeState`` but never rendered.

    The chip is rendered in muted text (var(--bp-muted-2)) so it visually
    subordinates to the auth-mode label but stays readable for the
    operator scanning the footer for "where is this UI listening".

    aria-hidden because the same context is already in the badge's
    ``aria_label`` (which mirrors hover_text) — preventing the screen
    reader from re-announcing the bind tuple in a separate audible
    chunk after the mode label.
    """
    return rx.flex(
        rx.text(
            "·",
            size="1",
            style={
                "color": "var(--bp-muted-2)",
                "font_size": "10px",
            },
            aria_hidden="true",
        ),
        rx.text(
            AuthBadgeState.bind_host + ":" + AuthBadgeState.bind_port,
            size="1",
            class_name="bp-num",
            style={
                "font_family": "var(--bp-mono)",
                "font_size": "10px",
                "color": "var(--bp-muted-2)",
                "letter_spacing": "0.01em",
            },
            aria_hidden="true",
        ),
        rx.text(
            "(" + AuthBadgeState.reachable_from + ")",
            size="1",
            style={
                "font_size": "10px",
                "color": "var(--bp-muted-2)",
                "font_style": "italic",
            },
            aria_hidden="true",
        ),
        gap="1",
        align="center",
    )


def BpAuthBadge() -> rx.Component:
    """Render the footer auth-mode badge.

    The badge always renders (even in NO_AUTH_LOCAL_ONLY mode) because the
    operator should always be able to glance at the footer and confirm the
    posture. Hiding it in any state would be the wrong UX choice - the
    point is at-a-glance reassurance / warning.

    The component returns a ``rx.tooltip`` wrapping a colored
    ``rx.badge``. The tooltip text and the badge's ``aria_label`` carry
    the same hover-text string so the chip is accessible to operators
    using either input modality.

    FRONTEND-F-002 (Wave 5.5): badge now renders a compact bind chip
    (``host:port (reach)``) right after the mode label so the three
    ``AuthBadgeState`` bind fields are actually surfaced. Pre-fix they
    were populated but never read.
    """
    return rx.tooltip(
        rx.badge(
            rx.flex(
                rx.text(
                    _emoji_for_mode(),
                    size="1",
                    style={
                        "font_family": "var(--bp-mono)",
                        "font_size": "10px",
                        "letter_spacing": "0.02em",
                    },
                ),
                rx.text(
                    AuthBadgeState.mode_text,
                    size="1",
                    style={"font_size": "11px"},
                ),
                _bind_chip(),
                gap="1",
                align="center",
            ),
            color_scheme=_color_scheme(),
            variant="soft",
            size="1",
            # aria-label mirrors the hover-text so screen readers get the
            # full context (Stage C accessibility floor). The bind tuple
            # is already inside hover_text (per ui_security.py) so the
            # visible bind chip stays aria-hidden to avoid re-announcing.
            aria_label=AuthBadgeState.hover_text,
            # Stable ID so external test harnesses can target the badge.
            id="bp-auth-badge",
        ),
        content=AuthBadgeState.hover_text,
    )
