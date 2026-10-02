"""Shared shell pieces — header, left nav, side rail, footer.

ui-v2 P1 redesign (Director's bar): curves + breathing room.

- ``BpHeader``  — 60px tall, 20px control icons with 36px hit areas
- ``BpLeftNav`` — 208px wide, active page is a rounded PILL highlight
- main scroll area — flex-grow on the page background
- ``BpSideRail`` — 300px wide, sections are rounded cards (no grid lines)
- ``BpFooter``  — 32px tall
"""

from __future__ import annotations

import reflex as rx

from backpropagate import __version__
from backpropagate.ui_state import AppState, AuthBadgeState, TrainState

from .components.auth_badge import BpAuthBadge
from .components.event_log import BpEventLog
from .components.icon import bp_icon
from .components.info_tip import info_tip
from .components.loss_chart import BpLossChart
from .components.status_pill import BpStatusPill

# FRONTEND-A-015: shared dynamic version label so header + footer cannot drift.
_BRAND_VERSION = f"v{__version__}"

# Surface key -> (label, href, icon name). Icons are inlined by
# ``components/icon.py`` so their ``currentColor`` strokes follow the row's
# text color (as ``<img>`` they always rendered black — FRONTEND-B-005).
#
# Wave 6 added the ``runs`` entry (FRONTEND-F-RUN-HISTORY-PAGE) - closes the
# CLI/UI parity gap created when F-003 shipped ``backprop list-runs`` /
# ``backprop show-run``. Stage C (FRONTEND-B-008): the runs row now uses
# the dedicated ``records.svg`` glyph instead of reusing ``train.svg`` -
# distinct visual identity matters in a 5-item nav.
_NAV_ITEMS = (
    ("train",     "Single run", "/",          "train"),
    ("multi-run", "Multi-run",  "/multi-run", "multi-run"),
    ("export",    "Export",     "/export",    "export"),
    ("dataset",   "Dataset",    "/dataset",   "dataset"),
    ("runs",      "Runs",       "/runs",      "records"),
    # Wave 6b (FRONTEND-7): /models — local HF cache inventory + cleanup.
    # ``chip`` reads as "compute hardware / model artifact"; matches the
    # mental model that lives alongside Runs in the operator's workflow.
    ("models",    "Models",     "/models",    "chip"),
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
            gap="var(--space-3)",
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
                    gap="var(--space-2)",
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
            #
            # ui-v2 P1 redesign: 20px icon inside a 36x36 pill hit area.
            rx.button(
                rx.cond(
                    rx.color_mode == "light",
                    bp_icon("moon", 20),
                    bp_icon("sun", 20),
                ),
                size="2",
                variant="ghost",
                on_click=rx.toggle_color_mode,
                aria_label="Toggle theme",
                style={
                    "width": "36px",
                    "height": "36px",
                    "border_radius": "var(--bp-r-pill)",
                    "padding": "0",
                    "align_items": "center",
                    "justify_content": "center",
                    "display": "flex",
                    "color": "var(--bp-text-2)",
                },
            ),
            rx.link(
                rx.box(
                    bp_icon("github", 20),
                    width="36px",
                    height="36px",
                    border_radius="var(--bp-r-pill)",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    class_name="bp-nav-row",
                ),
                href="https://github.com/mcp-tool-shop-org/backpropagate",
                is_external=True,
                aria_label="GitHub repository",
                style={"color": "var(--bp-muted)"},
            ),
            gap="var(--space-2)",
            align="center",
        ),
        padding_x="24px",
        height="60px",
        align="center",
        width="100%",
        style={
            "background": "var(--bp-surface)",
            "border_bottom": "1px solid var(--bp-border)",
            "flex_shrink": "0",
        },
        # FRONTEND-B-014-EXTENDED (Stage C accessibility): landmark role so
        # screen reader users can jump to the header. Pairs with the
        # ``role="navigation"`` on BpLeftNav and ``role="contentinfo"`` on
        # BpFooter — Radix doesn't auto-emit semantic landmarks on flex
        # boxes, so we surface them explicitly.
        role="banner",
    )


# ---------------------------------------------------------------------------
# BpLeftNav — 188px vertical nav (Radix tabs styled as left rail)
# ---------------------------------------------------------------------------


def _nav_link(key: str, label: str, href: str, icon_name: str, active_key: str) -> rx.Component:
    """One nav row. Active page = rounded pill highlight (ui-v2 P1 redesign)."""
    is_active = active_key == key
    return rx.link(
        rx.flex(
            # Inlined SVG: currentColor inherits the row color below
            # (muted when inactive, full text color when active).
            bp_icon(icon_name, 20),
            rx.text(
                label,
                size="2",
                weight="medium" if is_active else "regular",
                style={
                    "color": "var(--bp-text)" if is_active else "var(--bp-muted)",
                },
            ),
            gap="var(--space-3)",
            align="center",
            width="100%",
            style={
                "padding": "9px 14px",
                "border_radius": "var(--bp-r-pill)",
                "color": "var(--bp-text)" if is_active else "var(--bp-muted)",
                # The pill: filled surface for the active page, transparent
                # otherwise. ``.bp-nav-row:hover`` (TOKENS_CSS) covers hover.
                "background": (
                    "var(--bp-surface-2)" if is_active else "transparent"
                ),
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
    The Settings surface did NOT land in v1.4 either — per the Wave 5
    feature audit (FRONTEND-F-015) the route is now a v1.5 candidate
    homing theme toggle + default-output-dir + default-quantization +
    default-lora-preset + BACKPROPAGATE_UI_* env-var inspector + auth-
    token rotation (FRONTEND-F-007). The link returns when /settings is
    registered in ``ui_app/app.py``; shipping a visible dead link is
    operator-hostile.
    """
    return rx.flex(
        rx.flex(
            *(_nav_link(key, label, href, icon, active) for key, label, href, icon in _NAV_ITEMS),
            direction="column",
            gap="var(--space-1)",
            width="100%",
        ),
        rx.spacer(),
        direction="column",
        width="208px",
        padding="var(--space-4)",
        height="100%",
        style={
            "background": "var(--bp-surface)",
            "border_right": "1px solid var(--bp-border)",
            "flex_shrink": "0",
            "overflow_y": "auto",
        },
        # FRONTEND-B-014-EXTENDED (Stage C accessibility): landmark role so
        # screen readers + keyboard users can jump straight to the primary
        # navigation. ``aria_label`` distinguishes it from any future
        # secondary nav (e.g. side rail). The individual nav items already
        # carry ``aria_current="page"`` on the active row.
        role="navigation",
        aria_label="Primary",
    )


# ---------------------------------------------------------------------------
# BpSideRail — 296px live-run sidebar
# ---------------------------------------------------------------------------


def _rail_section(*children: rx.Component, first: bool = False) -> rx.Component:  # noqa: ARG001 — kept for call-site stability
    """One card in the side rail (ui-v2 P1 redesign: rounded cards, generous
    gaps — no hard 1px divider lines between sections)."""
    return rx.flex(
        *children,
        direction="column",
        gap="var(--space-3)",
        padding="var(--space-4)",
        width="100%",
        style={
            "background": "var(--bp-surface)",
            "border": "1px solid var(--bp-border)",
            "border_radius": "var(--bp-r-lg)",
            "box_shadow": "var(--bp-shadow-card)",
        },
    )


def BpSideRail() -> rx.Component:
    """296px right-side rail — status / sparkline / GPU + VRAM / log."""
    return rx.flex(
        # Status pill bound to TrainState. The pill is its own tinted card:
        # wrapping it in a rail card drew a box inside a box.
        rx.box(
            BpStatusPill(
                state=TrainState.run_state,
                label="Run state",
                detail=TrainState.run_state,
            ),
            # The pill swaps its whole body per state, so its tip sits on the
            # wrapper: one button, top right, whatever the state.
            rx.box(
                info_tip("run_state", side="left"),
                style={"position": "absolute", "top": "14px", "right": "12px"},
            ),
            width="100%",
            style={"position": "relative"},
        ),
        # Loss chart section — FRONTEND-6 (Wave 6b): wire to TrainState.
        # loss_history (via loss_chart_data computed Var) so the side-rail
        # actually shows live training loss instead of the v1.2 literal
        # placeholder. When the list is empty (idle / first frame) show a
        # one-line hint — not an empty chart frame, which reads as broken.
        # Recharts re-renders only the chart, not the full page tree, so
        # per-step updates are cheap.
        _rail_section(
            rx.cond(
                TrainState.loss_history.length() == 0,
                rx.flex(
                    rx.flex(
                        rx.text(
                            "Loss · last 80 steps",
                            size="1",
                            style={
                                "color": "var(--bp-text-2)",
                                "text_transform": "uppercase",
                                "letter_spacing": "0.06em",
                                "font_size": "10px",
                            },
                        ),
                        info_tip("loss", side="left"),
                        align="center",
                        gap="4px",
                    ),
                    rx.text(
                        "Start a run to see the curve.",
                        size="1",
                        style={"color": "var(--bp-muted-2)"},
                    ),
                    direction="column",
                    gap="var(--space-1)",
                ),
                rx.flex(
                    rx.flex(
                        rx.text(
                            "Loss · live",
                            size="1",
                            style={
                                "color": "var(--bp-text-2)",
                                "text_transform": "uppercase",
                                "letter_spacing": "0.06em",
                                "font_size": "10px",
                            },
                        ),
                        info_tip("loss", side="left"),
                        align="center",
                        gap="4px",
                    ),
                    BpLossChart(
                        TrainState.loss_chart_data,
                        height=80,
                        label="loss",
                        ema_label="ema",
                    ),
                    direction="column",
                    gap="var(--space-1)",
                ),
            ),
            # Hidden until the first step: an idle "step 0" read as a stuck run.
            rx.cond(
                TrainState.loss_history.length() > 0,
                rx.flex(
                    rx.text(
                        "step " + TrainState.current_step.to_string(),
                        size="2",
                        weight="medium",
                        class_name="bp-num bp-tick",
                        style={"color": "var(--bp-text)", "letter_spacing": "-0.02em"},
                    ),
                    rx.spacer(),
                    rx.text(
                        # ui-v2 P1 fix: formatted (4 decimals), not the raw repr.
                        TrainState.loss_label,
                        size="2",
                        class_name="bp-num",
                        style={"color": "var(--bp-teal)"},
                    ),
                    direction="row",
                    align="baseline",
                    width="100%",
                ),
                rx.fragment(),
            ),
        ),
        # GPU ring + VRAM bar — ui-v2 P1: LIVE-bound to TrainState's live
        # readings (the v1.4 build passed literal zeros, which the draft2
        # screenshots showed as the fake "VRAM 0.0 / 16.0 GB" placeholder:
        # BpGpuRing/BpVramBar compute their geometry at build time so they can
        # never bind live — the rail uses computed-var-driven boxes instead).
        _rail_section(
            rx.flex(
                # Temp ring as a conic-gradient disc (live via gpu_fill_pct)
                rx.box(
                    rx.box(
                        rx.text(
                            TrainState.gpu_temp_label,
                            size="1",
                            weight="medium",
                            class_name="bp-num",
                            style={"color": "var(--bp-text)"},
                        ),
                        width="44px",
                        height="44px",
                        border_radius="50%",
                        background="var(--bp-surface)",
                        display="flex",
                        align_items="center",
                        justify_content="center",
                    ),
                    width="60px",
                    height="60px",
                    border_radius="50%",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    background=(
                        "conic-gradient(var(--bp-teal) 0%, var(--bp-teal) "
                        + TrainState.gpu_fill_pct
                        + ", var(--bp-surface-3) "
                        + TrainState.gpu_fill_pct
                        + ", var(--bp-surface-3) 100%)"
                    ),
                    flex_shrink="0",
                ),
                rx.flex(
                    rx.text(
                        TrainState.gpu_name,
                        size="1",
                        style={"color": "var(--bp-text-2)", "flex_grow": "1"},
                    ),
                    rx.box(
                        rx.box(
                            width=TrainState.vram_fill_pct,
                            height="100%",
                            background="var(--bp-teal)",
                            border_radius="var(--bp-r-pill)",
                            style={"transition": "width 0.4s ease-out"},
                        ),
                        width="100%",
                        height="8px",
                        background="var(--bp-surface-3)",
                        border_radius="var(--bp-r-pill)",
                        overflow="hidden",
                        role="progressbar",
                        aria_label="VRAM used",
                    ),
                    rx.flex(
                        rx.text(
                            TrainState.vram_label,
                            size="1",
                            class_name="bp-num",
                            style={"color": "var(--bp-text-2)"},
                        ),
                        info_tip("gpu", side="left"),
                        align="center",
                        gap="4px",
                    ),
                    direction="column",
                    flex_grow="1",
                    gap="var(--space-2)",
                ),
                direction="row",
                gap="var(--space-3)",
                align="center",
                width="100%",
            ),
        ),
        # Event log
        _rail_section(
            rx.flex(
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
                info_tip("events", side="left"),
                align="center",
                gap="4px",
            ),
            BpEventLog(events=TrainState.events, max_n=6, show_view_full=True),
        ),
        direction="column",
        gap="var(--space-3)",
        width="300px",
        padding="var(--space-4)",
        height="100%",
        style={
            "border_left": "1px solid var(--bp-border)",
            "overflow_y": "auto",
            "flex_shrink": "0",
        },
        # FRONTEND-B-014-EXTENDED (Stage C accessibility): landmark role so
        # screen reader users can jump to the live-run sidebar. "Complementary"
        # is the canonical HTML landmark for related-but-secondary content
        # alongside the main scroll area.
        role="complementary",
        aria_label="Live run status",
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
            gap="var(--space-2)",
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
            gap="var(--space-3)",
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
        # FRONTEND-B-014-EXTENDED (Stage C accessibility): landmark role so
        # screen reader users can jump to the footer (auth badge / handbook
        # link / version). Pairs with ``role="banner"`` on BpHeader and
        # ``role="navigation"`` on BpLeftNav.
        role="contentinfo",
    )
