"""The page shell every surface shares (ui-v2 P3 visual pass).

Header, left nav, a scrolling content column with one max width and one
gutter, the right rail, and the footer. Pages pass their title, a one-line
description and the body; the spacing scale and layout live here once, so
the pages cannot drift apart.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail

CONTENT_MAX_WIDTH = "1320px"


def page_title(title: str, description: Any = None) -> rx.Component:
    children = [
        rx.heading(
            title,
            size="7",
            as_="h1",
            style={
                "color": "var(--bp-text)",
                "font_weight": "600",
                "letter_spacing": "-0.02em",
            },
        )
    ]
    if description is not None:
        children.append(
            rx.text(
                description,
                size="2",
                style={"color": "var(--bp-muted)", "max_width": "72ch", "line_height": "1.55"},
            )
        )
    return rx.flex(*children, direction="column", gap="var(--space-2)", width="100%")


def bp_page(
    *body: rx.Component,
    active: str,
    title: str,
    description: Any = None,
    on_mount: Any = None,
    rail: bool = True,
) -> rx.Component:
    """One page: header, nav, the content column, the rail and the footer."""
    content_kwargs: dict[str, Any] = {}
    if on_mount is not None:
        content_kwargs["on_mount"] = on_mount
    row = [
        BpLeftNav(active=active),
        rx.scroll_area(
            rx.flex(
                page_title(title, description),
                *body,
                direction="column",
                gap="var(--space-6)",
                padding=rx.breakpoints(initial="var(--space-4)", md="var(--space-6)", xl="var(--space-7)"),
                max_width=CONTENT_MAX_WIDTH,
                width="100%",
                margin_x="auto",
                **content_kwargs,
            ),
            flex_grow="1",
            style={"height": "100%"},
            type="auto",
            scrollbars="vertical",
        ),
    ]
    if rail:
        row.append(BpSideRail())
    return rx.flex(
        BpHeader(),
        rx.flex(
            *row,
            flex_grow="1",
            width="100%",
            style={"overflow": "hidden", "min_height": "0"},
        ),
        BpFooter(),
        direction="column",
        height="100vh",
        width="100%",
        style={"background": "var(--bp-bg)"},
    )


__all__ = ["CONTENT_MAX_WIDTH", "bp_page", "page_title"]
