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
            # FRONTEND-B-014-EXTENDED (Stage C accessibility): aria_label
            # describes the drop affordance for screen readers — they'd
            # otherwise hear only "click to upload" from the underlying
            # button. Maintains parity with the form's other named controls.
            aria_label="Upload a training dataset (JSONL / Alpaca / ShareGPT / OpenAI)",
            accept={
                "application/json": [".json"],
                "application/jsonl": [".jsonl"],
                "text/plain": [".txt"],
            },
            # FRONTEND-A-003: wire on_drop into the validator so extension /
            # size / magic-bytes / sanitize-filename actually fire on the
            # Reflex surface instead of relying on default behavior.
            on_drop=DatasetState.handle_upload(  # type: ignore[operator]
                rx.upload_files(upload_id="dataset_upload")
            ),
            style={
                "background": "var(--bp-surface-2)",
                "border": "1px dashed var(--bp-border-2)",
                "border_radius": "var(--bp-r-3)",
                "cursor": "pointer",
            },
        ),
        # FRONTEND-B-014-EXTENDED (Stage C accessibility): wrap the inline
        # upload-error text in role=alert so the screen reader announces
        # validation failures (extension / size / magic-bytes / sanitize-
        # filename rejects) at the same time the operator sees them turn
        # peach. ``role="alert"`` implies aria_live=assertive which is
        # appropriate for an actionable validation error.
        rx.cond(
            DatasetState.upload_error != "",
            rx.box(
                rx.text(
                    DatasetState.upload_error,
                    size="1",
                    style={"color": "var(--bp-peach)", "font_size": "11px"},
                ),
                role="alert",
            ),
            rx.fragment(),
        ),
        # FRONTEND-B-013: render only the basename so the operator's home
        # directory doesn't appear in the UI / screenshots. Full path stays on
        # the backend for the Trainer hookup.
        rx.cond(
            DatasetState.uploaded_path != "",
            rx.text(
                "Uploaded: " + DatasetState.uploaded_basename,
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
        rx.flex(
            _label("Override (when auto-detect guesses wrong)"),
            rx.select.root(
                rx.select.trigger(
                    placeholder="auto",
                    style={"width": "100%"},
                    aria_label="Dataset format override — auto / ShareGPT / Alpaca / OpenAI / JSONL",
                ),
                rx.select.content(
                    rx.select.item("auto-detect", value="auto"),
                    rx.select.item("ShareGPT", value="sharegpt"),
                    rx.select.item("Alpaca", value="alpaca"),
                    rx.select.item("OpenAI", value="openai"),
                    rx.select.item("JSONL", value="jsonl"),
                ),
                value=DatasetState.format_hint,
                on_change=DatasetState.set_format_hint,
            ),
            direction="column",
            gap="1",
        ),
        title="Format",
    )


def _preview_group() -> rx.Component:
    """Preview pane — first 5 records when uploaded.

    FRONTEND-9 (Wave 6b): mirror the runs.py empty-state pattern — name a
    concrete next action rather than just "upload to see preview." Norman
    1988 affordance framing.
    """
    return Group(
        rx.cond(
            DatasetState.preview_records.length() == 0,  # type: ignore[attr-defined]
            rx.flex(
                rx.text(
                    "No dataset loaded yet.",
                    size="2",
                    style={"color": "var(--bp-text-2)"},
                ),
                rx.text(
                    "Drop a .jsonl / .json / .csv file into the upload zone "
                    "above. We'll auto-detect ShareGPT / Alpaca / OpenAI "
                    "shape and show the first 5 records here. From the shell: "
                    "examples are at examples/quickstart.jsonl in the repo.",
                    size="1",
                    style={"color": "var(--bp-muted)"},
                ),
                direction="column",
                gap="2",
                align="center",
                padding="4",
                width="100%",
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
                checked=DatasetState.dedup_enabled,
                on_change=DatasetState.set_dedup_enabled,
            ),
            rx.checkbox(
                "Drop empty records",
                checked=DatasetState.drop_empty,
                on_change=DatasetState.set_drop_empty,
            ),
            rx.checkbox(
                "Apply curriculum (sort by length)",
                checked=DatasetState.apply_curriculum,
                on_change=DatasetState.set_apply_curriculum,
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
                    value=DatasetState.min_tokens.to_string(),
                    on_change=DatasetState.set_min_tokens,
                    type="number",
                    size="2",
                    class_name="bp-num",
                    aria_label="Minimum tokens per record (filter)",
                ),
                rx.cond(
                    DatasetState.min_tokens_error != "",
                    rx.text(
                        DatasetState.min_tokens_error,
                        size="1",
                        style={"color": "var(--bp-peach)", "font_size": "11px"},
                    ),
                    rx.fragment(),
                ),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Max tokens"),
                rx.input(
                    placeholder="2048",
                    value=DatasetState.max_tokens.to_string(),
                    on_change=DatasetState.set_max_tokens,
                    type="number",
                    size="2",
                    class_name="bp-num",
                    aria_label="Maximum tokens per record (filter)",
                ),
                rx.cond(
                    DatasetState.max_tokens_error != "",
                    rx.text(
                        DatasetState.max_tokens_error,
                        size="1",
                        style={"color": "var(--bp-peach)", "font_size": "11px"},
                    ),
                    rx.fragment(),
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
