"""Dataset page — ``/dataset`` — drop zone + format auto-detect + preview.

Component tree:

- Group "Upload"      — ``rx.upload`` drop zone
- Group "Format"      — detected format badge (Alpaca / ShareGPT / OpenAI / JSONL)
- Group "Preview"     — first 5 records in a table
- Group "Stats"       — record count · token stats · dedup hits
- Group "Filter / dedup" — toggles + thresholds
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import DatasetState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.group import Group


def _label(text: str) -> rx.Component:
    return rx.text(
        text,
        size="1",
        style={
            "color": "var(--bp-text-2)",
            "font_size": "11px",
            "margin_bottom": "4px",
        },
    )


def _upload_group() -> rx.Component:
    return Group(
        rx.upload(
            rx.flex(
                rx.image(
                    src="/icons/upload.svg",
                    width="32px",
                    height="32px",
                    style={"color": "var(--bp-muted)", "margin": "0 auto"},
                    alt="",
                ),
                rx.text(
                    "Drop a JSONL / Alpaca / ShareGPT file here, or click to browse.",
                    size="2",
                    style={"color": "var(--bp-text-2)", "text_align": "center"},
                ),
                rx.text(
                    "Format auto-detected from contents.",
                    size="1",
                    style={"color": "var(--bp-muted)", "text_align": "center"},
                ),
                direction="column",
                gap="2",
                align="center",
                justify="center",
                width="100%",
                padding="6",
            ),
            id="dataset_upload",
            multiple=False,
            accept={
                "application/json": [".json"],
                "application/jsonl": [".jsonl"],
                "text/plain": [".txt"],
            },
            style={
                "background": "var(--bp-surface-2)",
                "border": "1px dashed var(--bp-border-2)",
                "border_radius": "var(--bp-r-3)",
                "cursor": "pointer",
            },
        ),
        rx.cond(
            DatasetState.uploaded_path != "",
            rx.text(
                "Uploaded: " + DatasetState.uploaded_path,
                size="1",
                style={"font_family": "var(--bp-mono)", "color": "var(--bp-text-2)"},
            ),
            rx.fragment(),
        ),
        title="Upload",
    )


def _format_group() -> rx.Component:
    return Group(
        rx.flex(
            _label("Detected format"),
            rx.cond(
                DatasetState.detected_format != "",
                rx.badge(
                    DatasetState.detected_format,
                    variant="soft",
                    color_scheme="teal",
                    size="2",
                ),
                rx.text(
                    "(awaiting upload)",
                    size="2",
                    style={"color": "var(--bp-muted)", "font_style": "italic"},
                ),
            ),
            direction="column",
            gap="2",
        ),
        title="Format",
    )


def _preview_group() -> rx.Component:
    """Preview pane — first 5 records when uploaded."""
    return Group(
        rx.cond(
            DatasetState.preview_records.length() == 0,
            rx.text(
                "Upload a dataset to preview the first 5 records here.",
                size="1",
                style={"color": "var(--bp-muted)", "font_style": "italic"},
            ),
            rx.vstack(
                rx.foreach(
                    DatasetState.preview_records,
                    lambda rec, idx: rx.box(
                        rx.flex(
                            rx.text(
                                "#",
                                size="1",
                                style={
                                    "color": "var(--bp-muted-2)",
                                    "font_family": "var(--bp-mono)",
                                },
                            ),
                            rx.text(
                                idx + 1,
                                size="1",
                                class_name="bp-num",
                                style={"color": "var(--bp-muted-2)"},
                            ),
                            gap="1",
                            align="baseline",
                        ),
                        rx.text(
                            rec.to_string(),
                            size="1",
                            style={
                                "font_family": "var(--bp-mono)",
                                "color": "var(--bp-text-2)",
                                "white_space": "pre-wrap",
                                "word_break": "break-word",
                            },
                        ),
                        padding="3",
                        style={
                            "background": "var(--bp-surface-2)",
                            "border": "1px solid var(--bp-border)",
                            "border_radius": "var(--bp-r-2)",
                            "width": "100%",
                        },
                    ),
                ),
                align="stretch",
                gap="2",
                width="100%",
            ),
        ),
        title="Preview",
    )


def _stats_group() -> rx.Component:
    return Group(
        rx.grid(
            rx.flex(
                _label("Records"),
                rx.text(
                    DatasetState.record_count,
                    size="4",
                    weight="medium",
                    class_name="bp-num",
                    style={"color": "var(--bp-text)"},
                ),
                direction="column",
                gap="1",
            ),
            rx.flex(
                _label("Dedup hits"),
                rx.text(
                    DatasetState.dedup_hits,
                    size="4",
                    weight="medium",
                    class_name="bp-num",
                    style={"color": "var(--bp-amber)"},
                ),
                direction="column",
                gap="1",
            ),
            rx.flex(
                _label("Avg tokens"),
                rx.text(
                    "—",
                    size="4",
                    weight="medium",
                    class_name="bp-num",
                    style={"color": "var(--bp-muted)"},
                ),
                direction="column",
                gap="1",
            ),
            columns="repeat(3, 1fr)",
            gap="3",
            width="100%",
        ),
        title="Stats",
    )


def _filter_group() -> rx.Component:
    return Group(
        rx.flex(
            rx.checkbox(
                "Enable dedup",
                default_checked=DatasetState.dedup_enabled,
            ),
            rx.checkbox(
                "Drop empty records",
                default_checked=True,
            ),
            rx.checkbox(
                "Apply curriculum (sort by length)",
                default_checked=False,
            ),
            direction="column",
            gap="2",
            align="start",
        ),
        rx.grid(
            rx.flex(
                _label("Min tokens"),
                rx.input(
                    placeholder="0",
                    default_value="0",
                    type="number",
                    size="2",
                    class_name="bp-num",
                ),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Max tokens"),
                rx.input(
                    placeholder="2048",
                    default_value="2048",
                    type="number",
                    size="2",
                    class_name="bp-num",
                ),
                direction="column",
                width="100%",
            ),
            columns="repeat(2, 1fr)",
            gap="3",
            width="100%",
        ),
        title="Filter / dedup",
        collapsible=True,
        default_open=False,
    )


def dataset_page() -> rx.Component:
    """The Dataset surface."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="dataset"),
            rx.scroll_area(
                rx.flex(
                    rx.heading(
                        "Dataset",
                        size="6",
                        style={"color": "var(--bp-text)", "font_weight": "500"},
                    ),
                    rx.text(
                        "Upload a dataset, auto-detect the format, preview "
                        "records, and configure dedup / filtering.",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    _upload_group(),
                    _format_group(),
                    _preview_group(),
                    _stats_group(),
                    _filter_group(),
                    direction="column",
                    gap="4",
                    padding="6",
                    max_width="780px",
                ),
                flex_grow="1",
                style={"height": "100%"},
                type="auto",
                scrollbars="vertical",
            ),
            BpSideRail(),
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
