"""Dataset page — ``/dataset`` — drop zone + format auto-detect + preview.

Phase 1: minimal skeleton. Phase 2 wires ``rx.upload`` + the preview pane +
filter controls per the design digest.
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import (
    DatasetState,  # noqa: F401 — exported so phase-2 page wiring can reach it
)

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.group import Group

_ = DatasetState  # keep the import live until Phase 2 wires the upload handler


def dataset_page() -> rx.Component:
    """The Dataset surface."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="dataset"),
            rx.scroll_area(
                rx.flex(
                    rx.heading("Dataset", size="6"),
                    rx.text(
                        "Upload a dataset, auto-detect the format, preview "
                        "records, and configure dedup / filtering.",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    Group(
                        rx.text("[stub] upload drop zone (rx.upload)"),
                        title="Upload",
                    ),
                    Group(
                        rx.text("[stub] detected format indicator"),
                        title="Format",
                    ),
                    Group(
                        rx.text("[stub] preview pane — first 5 records"),
                        title="Preview",
                    ),
                    Group(
                        rx.text("[stub] record count, token stats, dedup hits"),
                        title="Stats",
                    ),
                    Group(
                        rx.text("[stub] dedup + filter controls"),
                        title="Filter / dedup",
                    ),
                    direction="column",
                    gap="4",
                    padding="6",
                    max_width="720px",
                ),
                flex_grow="1",
                style={"height": "100%"},
            ),
            BpSideRail(),
            flex_grow="1",
            width="100%",
            style={"overflow": "hidden"},
        ),
        BpFooter(),
        direction="column",
        height="100vh",
        width="100%",
    )
