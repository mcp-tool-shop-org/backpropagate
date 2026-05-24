"""Shared shell pieces — header, left nav, side rail, footer.

Per design digest §3 (Train surface component tree), §8 (Reflex implementation
map), and §4 (interaction patterns).

Width budget:

- ``BpHeader``  — 56px tall
- ``BpLeftNav`` — 188px wide
- main scroll area — flex-grow
- ``BpSideRail`` — 296px wide
- ``BpFooter``  — 32px tall
"""

from __future__ import annotations

import reflex as rx

from backpropagate import __version__
from backpropagate.ui_state import AppState, AuthBadgeState, TrainState

from .components.auth_badge import BpAuthBadge
from .components.event_log import BpEventLog
from .components.gpu_ring import BpGpuRing
from .components.sparkline import BpSparkline
from .components.status_pill import BpStatusPill
from .components.vram_bar import BpVramBar

# FRONTEND-A-015: shared dynamic version label so header + footer cannot drift.
_BRAND_VERSION = f"v{__version__}"

# Surface key -> (label, href, icon URL). The icons live under
# ``backpropagate/assets/icons/`` so Reflex serves them at ``/icons/<name>.svg``.
#
# Wave 6 added the ``runs`` entry (FRONTEND-F-RUN-HISTORY-PAGE) - closes the
# CLI/UI parity gap created when F-003 shipped ``backprop list-runs`` /
# ``backprop show-run``. Stage C (FRONTEND-B-008): the runs row now uses
# the dedicated ``records.svg`` glyph instead of reusing ``train.svg`` -
# distinct visual identity matters in a 5-item nav.
_NAV_ITEMS = (
    ("train",     "Single run", "/",          "/icons/train.svg"),
    ("multi-run", "Multi-run",  "/multi-run", "/icons/multi-run.svg"),
    ("export",    "Export",     "/export",    "/icons/export.svg"),
    ("dataset",   "Dataset",    "/dataset",   "/icons/dataset.svg"),
    ("runs",      "Runs",       "/runs",      "/icons/records.svg"),
)


# ---------------------------------------------------------------------------
# BpHeader — 56px logo / wordmark / version · run_id · theme · gh
# ---------------------------------------------------------------------------


def BpHeader() -> rx.Component:
    """The 56px header strip."""
    return rx.flex(
        # Left: logo + wordmark + version badge
        rx.flex(
            rx.image(
                src="/logo.png",
                width="32px",
                height="32px",
                style={"border_radius": "9999px"},
                alt="backpropagate logo",
            ),
            rx.text(
                "backpropagate",
                size="2",
                weight="medium",
                style={
                    "font_family": "var(--bp-sans)",
                    "color": "var(--bp-text)",
                    "font_size": "14px",
                },
            ),
            rx.badge(
                _BRAND_VERSION,
                variant="soft",
                color_scheme="teal",
                size="1",
            ),
            gap="3",
            align="center",
        ),
        rx.spacer(),
        # Right: run_id (when present) · theme toggle · gh link
        rx.flex(
            rx.cond(
                AppState.run_id != "",
                rx.flex(
                    rx.text(
                        "run_id",
                        size="1",
                        style={
                            "color": "var(--bp-muted)",
                            "text_transform": "uppercase",
                            "letter_spacing": "0.06em",
                            "font_size": "10px",
                        },
                    ),
                    rx.text(
                        AppState.run_id,
                        size="1",
                        class_name="bp-num",
                        style={
                            "font_family": "var(--bp-mono)",
                            "color": "var(--bp-peach)",
                            "font_size": "12px",
                        },
                    ),
                    gap="2",
                    align="center",
                ),
                rx.fragment(),
            ),
            # FRONTEND-F-001 (Wave 5.5): bind to Reflex's built-in color
            # mode so the toggle actually mutates the DOM (Radix theme
            # provider writes ``class="light"`` / ``class="dark"`` on the
            # html root, which fires the ``.light`` / ``.light-theme``
            # selector in TOKENS_CSS). The previous wiring flipped
            # ``AppState.theme`` server-side but never reached the DOM,
            # so the icon swapped but the page stayed dark.
            #
            # ``rx.color_mode`` is "system" until the operator overrides
            # it once; we show the sun icon (= will switch to light) when
            # NOT currently light, regardless of whether dark is the
            # resolved system pref or an explicit choice.
            rx.button(
                rx.cond(
                    rx.color_mode == "light",
                    rx.image(
                        src="/icons/moon.svg",
                        width="16px",
                        height="16px",
                        alt="switch to dark theme",
                    ),
                    rx.image(
                        src="/icons/sun.svg",
                        width="16px",
                        height="16px",
                        alt="switch to light theme",
                    ),
                ),
                size="1",
                variant="ghost",
                on_click=rx.toggle_color_mode,
                aria_label="Toggle theme",
            ),
            rx.link(
                rx.image(
                    src="/icons/github.svg",
                    width="16px",
                    height="16px",
                    alt="GitHub repository",
                ),
                href="https://github.com/mcp-tool-shop-org/backpropagate",
                is_external=True,
                style={"color": "var(--bp-muted)"},
            ),
            gap="3",
            align="center",
        ),
        padding_x="20px",
        height="56px",
        align="center",
        width="100%",
        style={
            "background": "var(--bp-surface)",
            "border_bottom": "1px solid var(--bp-border)",
            "flex_shrink": "0",
        },
    )


# ---------------------------------------------------------------------------
# BpLeftNav — 188px vertical nav (Radix tabs styled as left rail)
# ---------------------------------------------------------------------------


def _nav_link(key: str, label: str, href: str, icon_url: str, active_key: str) -> rx.Component:
    """One nav row. Active gets a teal inset bar + brighter text."""
    is_active = active_key == key
    return rx.link(
        rx.flex(
            rx.image(
                src=icon_url,
                width="18px",
                height="18px",
                style={
                    "color": "var(--bp-text)" if is_active else "var(--bp-muted)",
                    "flex_shrink": "0",
                },
                alt="",
            ),
            rx.text(
                label,
                size="2",
                weight="medium" if is_active else "regular",
                style={
                    "color": "var(--bp-text)" if is_active else "var(--bp-muted)",
                },
            ),
            gap="3",
            align="center",
            padding_x="3",
            padding_y="2",
            width="100%",
            style={
                "border_radius": "var(--bp-r-2)",
                "box_shadow": (
                    "inset 3px 0 0 var(--bp-teal)" if is_active else "none"
                ),
                "background": (
                    "var(--bp-surface-2)" if is_active else "transparent"
                ),
                "transition": "background 0.15s ease, color 0.15s ease",
            },
            class_name="bp-nav-row",
        ),
        href=href,
        underline="none",
        width="100%",
        aria_current="page" if is_active else None,
    )


def BpLeftNav(active: str = "train") -> rx.Component:
    """188px left rail - vertical nav.

    FRONTEND-B-011 (Stage C truth-in-advertising): the Settings link was
    removed in v1.3 because the /settings route is not yet registered in
    ``ui_app/app.py`` (Reflex returns a generic not-found page on click).
    The link returns when the Settings surface ships (currently scoped to
    a v1.4 follow-up); shipping a visible dead link is operator-hostile.
    """
    return rx.flex(
        rx.flex(
            *(_nav_link(key, label, href, icon, active) for key, label, href, icon in _NAV_ITEMS),
            direction="column",
            gap="1",
            width="100%",
        ),
        rx.spacer(),
        direction="column",
        width="188px",
        padding="3",
        height="100%",
        style={
            "background": "var(--bp-surface)",
            "border_right": "1px solid var(--bp-border)",
            "flex_shrink": "0",
            "overflow_y": "auto",
        },
    )


# ---------------------------------------------------------------------------
# BpSideRail — 296px live-run sidebar
# ---------------------------------------------------------------------------


def _rail_section(*children: rx.Component, first: bool = False) -> rx.Component:
    """One bordered section in the side rail."""
    style: dict[str, str] = {
        "padding": "16px 0",
        "width": "100%",
    }
    if not first:
        style["border_top"] = "1px solid var(--bp-border)"
    return rx.flex(*children, direction="column", gap="2", style=style)


def BpSideRail() -> rx.Component:
    """296px right-side rail — status / sparkline / GPU + VRAM / log."""
    return rx.flex(
        # Status pill bound to TrainState
        _rail_section(
            BpStatusPill(
                state=TrainState.run_state,
                label="Run state",
                detail=TrainState.run_state,
            ),
            first=True,
        ),
        # Sparkline section — caption + meta from state
        _rail_section(
            BpSparkline(
                data=[],
                w=264,
                h=48,
                caption="Loss · last 80 steps",
                meta="",
            ),
            rx.flex(
                rx.text(
                    TrainState.current_step,
                    size="4",
                    weight="medium",
                    class_name="bp-num bp-tick",
                    style={"color": "var(--bp-text)", "letter_spacing": "-0.02em"},
                ),
                rx.spacer(),
                rx.text(
                    TrainState.current_loss,
                    size="2",
                    class_name="bp-num",
                    style={"color": "var(--bp-teal)"},
                ),
                direction="row",
                align="baseline",
                width="100%",
            ),
        ),
        # GPU ring + VRAM bar
        _rail_section(
            rx.flex(
                BpGpuRing(temp_c=0, size=60),
                rx.flex(
                    BpVramBar(used_gb=0.0, total_gb=16.0),
                    direction="column",
                    flex_grow="1",
                    gap="2",
                ),
                direction="row",
                gap="3",
                align="center",
                width="100%",
            ),
        ),
        # Event log
        _rail_section(
            rx.text(
                "Events",
                size="1",
                style={
                    "color": "var(--bp-text-2)",
                    "text_transform": "uppercase",
                    "letter_spacing": "0.06em",
                    "font_size": "10px",
                },
            ),
            BpEventLog(events=TrainState.events, max_n=6, show_view_full=True),
        ),
        direction="column",
        gap="0",
        width="296px",
        padding_x="16px",
        height="100%",
        style={
            "background": "var(--bp-surface)",
            "border_left": "1px solid var(--bp-border)",
            "overflow_y": "auto",
            "flex_shrink": "0",
        },
    )


# ---------------------------------------------------------------------------
# BpFooter — 32px handbook · version · run_id · gh
# ---------------------------------------------------------------------------


def BpFooter() -> rx.Component:
    """32px footer strip.

    Stage C (FRONTEND-F-FOOTER-AUTH-BADGE): the auth-mode badge sits in the
    center, between handbook/version on the left and run_id/github on the
    right. The badge gives the operator at-a-glance reassurance (or red-flag
    warning) about the current auth posture without leaving the page.

    The ``on_mount`` populates AuthBadgeState from the env-var surface that
    ``ui_app/auth.py::basic_auth_transformer`` also consumes - the values
    are stable for the lifetime of the Reflex subprocess (the CLI exports
    them once at launch).
    """
    return rx.flex(
        # Left: handbook + version
        rx.flex(
            rx.link(
                "Handbook",
                href="https://mcp-tool-shop-org.github.io/backpropagate/",
                is_external=True,
                size="1",
                style={"color": "var(--bp-muted)"},
            ),
            rx.text(
                "·",
                size="1",
                style={"color": "var(--bp-muted-2)"},
            ),
            rx.text(
                _BRAND_VERSION,
                size="1",
                style={"color": "var(--bp-muted-2)"},
            ),
            gap="2",
            align="center",
        ),
        rx.spacer(),
        # Center: auth-mode badge (FRONTEND-F-FOOTER-AUTH-BADGE, Stage C)
        BpAuthBadge(),
        rx.spacer(),
        # Right: run_id mirror (when present) + gh
        rx.flex(
            rx.cond(
                AppState.run_id != "",
                rx.text(
                    AppState.run_id,
                    size="1",
                    class_name="bp-num",
                    style={
                        "font_family": "var(--bp-mono)",
                        "color": "var(--bp-muted)",
                        "font_size": "11px",
                    },
                ),
                rx.fragment(),
            ),
            rx.link(
                "GitHub",
                href="https://github.com/mcp-tool-shop-org/backpropagate",
                is_external=True,
                size="1",
                style={"color": "var(--bp-muted)"},
            ),
            gap="3",
            align="center",
        ),
        padding_x="20px",
        height="32px",
        align="center",
        width="100%",
        style={
            "background": "var(--bp-surface)",
            "border_top": "1px solid var(--bp-border)",
            "flex_shrink": "0",
        },
        on_mount=AuthBadgeState.refresh,
    )
