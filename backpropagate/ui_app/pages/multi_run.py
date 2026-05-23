"""Multi-Run page — ``/multi-run`` — SLAO sweep surface.

Mirrors the Train page's Model + Training shape Groups, then adds:

- Group "Sweep shape" — num_runs · samples_per_run · merge_mode
- Group "Runs" — per-run table (status / step / loss / inline trajectory)
- Group "Cross-run analysis" — aggregate stats (placeholder for now)
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import MultiRunState

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


def _model_group() -> rx.Component:
    return Group(
        rx.grid(
            rx.flex(
                _label("HuggingFace model id"),
                rx.input(
                    placeholder="meta-llama/Llama-3.1-8B",
                    default_value=MultiRunState.model,
                    on_change=MultiRunState.set_model,
                    size="2",
                ),
                rx.cond(
                    MultiRunState.model_error != "",
                    rx.text(
                        MultiRunState.model_error,
                        size="1",
                        style={"color": "var(--bp-peach)", "font_size": "11px"},
                    ),
                    rx.fragment(),
                ),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Quantization"),
                rx.select.root(
                    rx.select.trigger(placeholder="4-bit", style={"width": "100%"}),
                    rx.select.content(
                        rx.select.item("4-bit", value="4-bit"),
                        rx.select.item("8-bit", value="8-bit"),
                        rx.select.item("16-bit", value="16-bit"),
                    ),
                    default_value="4-bit",
                ),
                direction="column",
                width="100%",
            ),
            columns="2fr 1fr",
            gap="3",
            width="100%",
        ),
        title="Model",
    )


def _sweep_shape_group() -> rx.Component:
    return Group(
        rx.grid(
            rx.flex(
                _label("Num runs"),
                rx.input(
                    placeholder="3",
                    default_value="3",
                    type="number",
                    size="2",
                    class_name="bp-num",
                ),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Samples per run"),
                rx.input(
                    placeholder="500",
                    default_value="500",
                    type="number",
                    size="2",
                    class_name="bp-num",
                ),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Merge mode"),
                rx.select.root(
                    rx.select.trigger(placeholder="slao", style={"width": "100%"}),
                    rx.select.content(
                        rx.select.item("SLAO (default)", value="slao"),
                        rx.select.item("Weighted", value="weighted"),
                        rx.select.item("TIES", value="ties"),
                    ),
                    default_value="slao",
                ),
                direction="column",
                width="100%",
            ),
            columns="repeat(3, 1fr)",
            gap="3",
            width="100%",
        ),
        title="Sweep shape",
    )


def _runs_table() -> rx.Component:
    """The per-run summary table.

    Empty in Phase 2. MultiRunState's ``runs`` list will populate this when
    the backend hookup lands (Phase 3 of the migration plan).
    """
    headers = ["Run", "Status", "Step", "Loss", "Trajectory"]
    return Group(
        rx.box(
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        *(
                            rx.table.column_header_cell(
                                rx.text(
                                    h,
                                    size="1",
                                    style={
                                        "color": "var(--bp-text-2)",
                                        "text_transform": "uppercase",
                                        "letter_spacing": "0.06em",
                                        "font_size": "10px",
                                    },
                                ),
                            )
                            for h in headers
                        ),
                    ),
                ),
                rx.table.body(
                    rx.cond(
                        MultiRunState.runs.length() == 0,  # type: ignore[attr-defined]
                        rx.table.row(
                            rx.table.cell(
                                rx.text(
                                    "No runs yet — press Start multi-run to begin.",
                                    size="1",
                                    style={
                                        "color": "var(--bp-muted)",
                                        "font_style": "italic",
                                    },
                                ),
                                col_span=len(headers),
                                style={"text_align": "center", "padding": "16px"},
                            ),
                        ),
                        rx.foreach(
                            MultiRunState.runs,
                            lambda run, idx: rx.table.row(
                                rx.table.cell(
                                    rx.text(
                                        idx + 1,
                                        size="1",
                                        class_name="bp-num",
                                    ),
                                ),
                                rx.table.cell(
                                    rx.text(
                                        run["status"],
                                        size="1",
                                        style={"font_family": "var(--bp-mono)"},
                                    ),
                                ),
                                rx.table.cell(
                                    rx.text(
                                        run["step"],
                                        size="1",
                                        class_name="bp-num",
                                    ),
                                ),
                                rx.table.cell(
                                    rx.text(
                                        run["loss"],
                                        size="1",
                                        class_name="bp-num",
                                        style={"color": "var(--bp-teal)"},
                                    ),
                                ),
                                rx.table.cell(
                                    rx.text(
                                        "—",
                                        size="1",
                                        style={"color": "var(--bp-muted)"},
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                size="1",
                variant="surface",
                style={"width": "100%"},
            ),
        ),
        title="Runs",
    )


def _cross_run_group() -> rx.Component:
    return Group(
        rx.text(
            "Cross-run loss + per-adapter contribution will surface here once "
            "the sweep completes.",
            size="1",
            style={"color": "var(--bp-muted)"},
        ),
        title="Cross-run analysis",
    )


def multi_run_page() -> rx.Component:
    """The Multi-Run surface."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="multi-run"),
            rx.scroll_area(
                rx.flex(
                    rx.heading(
                        "Multi-run",
                        size="6",
                        style={"color": "var(--bp-text)", "font_weight": "500"},
                    ),
                    rx.text(
                        "SLAO sweep — train multiple runs and merge the LoRA "
                        "adapters to defeat catastrophic forgetting.",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    _model_group(),
                    _sweep_shape_group(),
                    _runs_table(),
                    _cross_run_group(),
                    rx.flex(
                        rx.button(
                            "Start multi-run",
                            variant="solid",
                            color_scheme="teal",
                            size="3",
                            on_click=MultiRunState.start_multi_run,
                        ),
                        gap="3",
                        margin_top="2",
                    ),
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
