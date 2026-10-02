"""Dataset page (``/dataset``): look inside a file, clean it up, train on it.

Component tree:

- Group "Upload"      — ``rx.upload`` drop zone
- Group "Format"      — the layout the examples are in, and an override
- Group "Preview"     — the first five examples as the trainer reads them
- Group "Stats"       — examples, repeats, how long they are
- Group "Clean up"    — what to remove, what that leaves, "Save a cleaned copy"
- Group "Train on it" — put the file in the Single run or Multi-run form

Everything the page shows comes from ``backpropagate.dataset_prep``.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from backpropagate.ui_state import DatasetState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.field import FIELD_STYLE as _FIELD_STYLE
from ..components.field import bp_label as _label
from ..components.group import Group
from ..components.icon import bp_icon
from ..components.info_tip import info_tip, with_tip


def _upload_group() -> rx.Component:
    return Group(
        rx.upload(
            rx.flex(
                # ui-v2 P1 redesign pass 2: inlined (bp_icon) so the glyph
                # follows --bp-muted; as <img> its currentColor bake was
                # black on every theme (FRONTEND-B-005).
                bp_icon("upload", 32, color="var(--bp-muted)"),
                rx.text(
                    "Drop a .jsonl or .json dataset file here, or click to browse.",
                    size="2",
                    style={"color": "var(--bp-text-2)", "text_align": "center"},
                ),
                rx.text(
                    "The layout of the examples is recognised from the file.",
                    size="1",
                    style={"color": "var(--bp-muted)", "text_align": "center"},
                ),
                direction="column",
                gap="var(--space-2)",
                align="center",
                justify="center",
                width="100%",
                padding="var(--space-6)",
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
                "background": "var(--bp-field-bg)",
                "border": "1px dashed var(--bp-border-2)",
                "border_radius": "var(--bp-r-md)",
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
        # FRONTEND-B-013 / UI-A-002: render only the basename so the
        # operator's home directory doesn't appear in the UI / screenshots —
        # AND the full path is now a backend-only var that never enters the
        # serialized state bundle. ``has_upload`` is the client-safe boolean
        # for the cond (the backend-only ``_uploaded_path`` isn't serialized).
        rx.cond(
            DatasetState.has_upload,
            rx.text(
                "Uploaded: " + DatasetState.uploaded_basename,
                size="1",
                style={"font_family": "var(--bp-mono)", "color": "var(--bp-text-2)"},
            ),
            rx.fragment(),
        ),
        rx.cond(
            DatasetState.inspect_note != "",
            rx.text(
                DatasetState.inspect_note,
                size="1",
                id="bp-dataset-note",
                style={"color": "var(--bp-muted)", "font_size": "13px", "line_height": "1.5"},
            ),
            rx.fragment(),
        ),
        title="Upload",
    )


_NOTE_STYLE = {"color": "var(--bp-muted)", "font_size": "13px", "line_height": "1.5"}
_ERROR_STYLE = {"color": "var(--bp-peach)", "font_size": "13px", "line_height": "1.5"}


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
                    id="bp-dataset-format",
                ),
                rx.text(
                    rx.cond(DatasetState.has_upload, "(not recognised)", "(awaiting upload)"),
                    size="2",
                    style={"color": "var(--bp-muted)", "font_style": "italic"},
                ),
            ),
            direction="column",
            gap="var(--space-2)",
            align="start",
        ),
        rx.flex(
            _label("Read the file as"),
            rx.select.root(
                rx.select.trigger(
                    placeholder="auto",
                    style={**_FIELD_STYLE, "width": "100%"},
                    aria_label="Read the file as: auto-detect, ShareGPT, Alpaca, OpenAI or JSONL",
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
            gap="var(--space-1)",
        ),
        title="Format",
        info="dataset_format",
    )


def _preview_card(rec: Any) -> rx.Component:
    return rx.box(
        rx.text(
            "Example ",
            rec["number"],
            " · about ",
            rec["tokens"],
            " tokens",
            size="1",
            class_name="bp-num",
            style={"color": "var(--bp-muted-2)", "margin_bottom": "var(--space-2)"},
        ),
        rx.text(
            rec["text"],
            size="2",
            style={
                "color": "var(--bp-text-2)",
                "white_space": "pre-wrap",
                "word_break": "break-word",
                "line_height": "1.55",
            },
        ),
        padding="var(--space-3)",
        class_name="bp-dataset-example",
        style={
            "background": "var(--bp-surface-2)",
            "border": "1px solid var(--bp-border)",
            "border_radius": "var(--bp-r-md)",
            "width": "100%",
        },
    )


def _preview_group() -> rx.Component:
    """The first five examples, as the trainer reads them."""
    return Group(
        rx.cond(
            DatasetState.preview_records.length() == 0,  # type: ignore[attr-defined]
            rx.flex(
                rx.text(
                    rx.cond(
                        DatasetState.has_upload,
                        "No examples to show for this file.",
                        "No dataset loaded yet.",
                    ),
                    size="2",
                    style={"color": "var(--bp-text-2)"},
                ),
                rx.text(
                    "Drop a .jsonl or .json file into the upload zone above. The "
                    "first five examples appear here, laid out as a conversation. "
                    "To try the page, use examples/quickstart.jsonl from the repo.",
                    size="1",
                    style={"color": "var(--bp-muted)", "text_align": "center"},
                ),
                direction="column",
                gap="var(--space-2)",
                align="center",
                padding="var(--space-4)",
                width="100%",
            ),
            rx.vstack(
                rx.foreach(DatasetState.preview_records, _preview_card),
                align="stretch",
                gap="var(--space-2)",
                width="100%",
                id="bp-dataset-preview",
            ),
        ),
        title="Preview",
        info="dataset_preview",
    )


def _stat(label: str, value: Any, *, id: str, color: Any = "var(--bp-text)") -> rx.Component:
    """One big number. A dash stands in until a file has been read."""
    return rx.flex(
        _label(label),
        rx.text(
            rx.cond(DatasetState.record_count > 0, value, "—"),
            size="4",
            weight="medium",
            class_name="bp-num",
            id=id,
            style={"color": rx.cond(DatasetState.record_count > 0, color, "var(--bp-muted)")},
        ),
        direction="column",
        gap="var(--space-1)",
    )


def _stats_group() -> rx.Component:
    return Group(
        rx.grid(
            _stat("Examples", DatasetState.stat_text["examples"], id="bp-stat-examples"),
            _stat(
                "Repeats",
                DatasetState.stat_text["repeats"],
                id="bp-stat-repeats",
                color=rx.cond(DatasetState.dedup_hits > 0, "var(--bp-amber)", "var(--bp-text)"),
            ),
            _stat("Average tokens", DatasetState.stat_text["average"], id="bp-stat-average"),
            _stat("Shortest", DatasetState.stat_text["shortest"], id="bp-stat-shortest"),
            _stat("Longest", DatasetState.stat_text["longest"], id="bp-stat-longest"),
            columns=rx.breakpoints(initial="repeat(2, 1fr)", sm="repeat(5, 1fr)"),
            gap="var(--space-3)",
            width="100%",
        ),
        rx.cond(
            DatasetState.skipped_note != "",
            rx.text(
                DatasetState.skipped_note,
                size="1",
                id="bp-dataset-skipped",
                style=_NOTE_STYLE,
            ),
            rx.fragment(),
        ),
        title="Stats",
        info="dataset_stats",
    )


def _check(label: str, checked: Any, on_change: Any, tip: str | None = None) -> rx.Component:
    box = rx.checkbox(label, checked=checked, on_change=on_change)
    return rx.flex(box, info_tip(tip), align="center", gap="6px") if tip else box


def _token_limit(
    label: str, value: Any, on_change: Any, error: Any, *, aria: str, placeholder: str
) -> rx.Component:
    return rx.flex(
        _label(label),
        rx.input(
            placeholder=placeholder,
            value=value.to_string(),
            on_change=on_change,
            type="number",
            min="0",
            size="2",
            class_name="bp-num",
            style={**_FIELD_STYLE, "width": "100%"},
            aria_label=aria,
        ),
        rx.cond(error != "", rx.text(error, size="1", style=_ERROR_STYLE), rx.fragment()),
        direction="column",
        width="100%",
    )


def _cleanup_group() -> rx.Component:
    """What to remove, what that leaves, and the button that writes the copy."""
    return Group(
        rx.flex(
            _check(
                "Remove repeats",
                DatasetState.dedup_enabled,
                DatasetState.set_dedup_enabled,
            ),
            _check(
                "Remove empty examples",
                DatasetState.drop_empty,
                DatasetState.set_drop_empty,
            ),
            _check(
                "Order from short to long",
                DatasetState.apply_curriculum,
                DatasetState.set_apply_curriculum,
                tip="cleanup_order",
            ),
            direction="column",
            gap="var(--space-2)",
            align="start",
        ),
        rx.flex(
            with_tip(_label("Length limits, in tokens"), "cleanup_length"),
            rx.grid(
                _token_limit(
                    "Shortest",
                    DatasetState.min_tokens,
                    DatasetState.set_min_tokens,
                    DatasetState.min_tokens_error,
                    aria="Remove examples shorter than this many tokens",
                    placeholder="0",
                ),
                _token_limit(
                    "Longest (0 = no limit)",
                    DatasetState.max_tokens,
                    DatasetState.set_max_tokens,
                    DatasetState.max_tokens_error,
                    aria="Remove examples longer than this many tokens; 0 means no limit",
                    placeholder="0",
                ),
                columns="repeat(2, 1fr)",
                gap="var(--space-3)",
                width="100%",
            ),
            direction="column",
            gap="var(--space-2)",
        ),
        # What the settings do to the uploaded file, updated as they change.
        rx.box(
            rx.text(
                DatasetState.cleanup_summary,
                size="2",
                id="bp-cleanup-summary",
                style={"color": "var(--bp-text-2)", "line_height": "1.5"},
            ),
            aria_live="polite",
        ),
        rx.flex(
            rx.button(
                "Save a cleaned copy",
                variant="soft",
                color_scheme="teal",
                size="2",
                on_click=DatasetState.save_cleaned_copy,
                disabled=~DatasetState.can_save_copy,
                id="bp-save-copy",
                style={"border_radius": "var(--bp-r-pill)", "white_space": "nowrap"},
            ),
            rx.cond(
                DatasetState.prepared_name != "",
                rx.text(
                    "Saved ",
                    rx.text.span(
                        DatasetState.prepared_name,
                        style={"font_family": "var(--bp-mono)", "color": "var(--bp-text-2)"},
                    ),
                    ". Your uploaded file is unchanged.",
                    size="1",
                    id="bp-copy-saved",
                    style=_NOTE_STYLE,
                ),
                rx.text(
                    "The copy is a new file. Your uploaded file is never changed.",
                    size="1",
                    style=_NOTE_STYLE,
                ),
            ),
            align="center",
            gap="var(--space-3)",
            wrap="wrap",
        ),
        rx.cond(
            DatasetState.prepare_error != "",
            rx.box(
                rx.text(DatasetState.prepare_error, size="1", style=_ERROR_STYLE),
                role="alert",
            ),
            rx.fragment(),
        ),
        title="Clean up",
        info="dataset_cleanup",
    )


def _use_button(label: str, on_click: Any, id: str) -> rx.Component:
    return rx.button(
        label,
        variant="solid",
        color_scheme="teal",
        size="2",
        on_click=on_click,
        disabled=~DatasetState.has_upload,
        id=id,
        style={"border_radius": "var(--bp-r-pill)", "white_space": "nowrap"},
    )


def _use_group() -> rx.Component:
    """Hand the cleaned copy (or the upload) to a training form."""
    return Group(
        rx.cond(
            DatasetState.has_upload,
            rx.flex(
                rx.text(
                    DatasetState.training_file_name,
                    size="2",
                    id="bp-training-file",
                    style={"font_family": "var(--bp-mono)", "color": "var(--bp-text)"},
                ),
                rx.text(DatasetState.training_file_note, size="1", style=_NOTE_STYLE),
                rx.cond(
                    DatasetState.can_save_copy & (DatasetState.prepared_name == ""),
                    rx.text(
                        "The clean-up settings above only apply to a saved copy. "
                        "Save one first to train on the cleaned examples.",
                        size="1",
                        id="bp-copy-hint",
                        style=_NOTE_STYLE,
                    ),
                    rx.fragment(),
                ),
                direction="column",
                gap="var(--space-1)",
            ),
            rx.text("Upload a file first.", size="2", style={"color": "var(--bp-muted)"}),
        ),
        rx.flex(
            _use_button("Use in Single run", DatasetState.use_in_single_run, "bp-use-single"),
            _use_button("Use in Multi-run", DatasetState.use_in_multi_run, "bp-use-multi"),
            gap="var(--space-3)",
            wrap="wrap",
        ),
        title="Train on it",
        info="dataset_use",
    )


def dataset_page() -> rx.Component:
    """The Dataset surface."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="dataset"),
            rx.scroll_area(
                rx.flex(
                    rx.flex(
                        with_tip(
                            rx.heading(
                                "Dataset",
                                size="7",
                                style={
                                    "color": "var(--bp-text)",
                                    "font_weight": "600",
                                    "letter_spacing": "-0.02em",
                                },
                            ),
                            "page_dataset",
                        ),
                        rx.text(
                            "See what a dataset file contains before you train on it, "
                            "clean it up, and send it to a training form.",
                            size="2",
                            style={"color": "var(--bp-muted)"},
                        ),
                        direction="column",
                        gap="var(--space-2)",
                        width="100%",
                    ),
                    _upload_group(),
                    _format_group(),
                    _preview_group(),
                    _stats_group(),
                    _cleanup_group(),
                    _use_group(),
                    direction="column",
                    gap="var(--space-6)",
                    padding=rx.breakpoints(
                        initial="var(--space-4)", md="var(--space-6)", xl="var(--space-7)"
                    ),
                    max_width="1320px",
                    margin_x="auto",
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
