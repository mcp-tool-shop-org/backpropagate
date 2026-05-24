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


def _err_text(error_var) -> rx.Component:
    """Inline error label — peach text, 11px, only renders when non-empty."""
    return rx.cond(
        error_var != "",
        rx.text(
            error_var,
            size="1",
            style={"color": "var(--bp-peach)", "font_size": "11px"},
        ),
        rx.fragment(),
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
                    aria_label="HuggingFace model id",
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
                    rx.select.trigger(
                        placeholder="4-bit",
                        style={"width": "100%"},
                        aria_label="Quantization level — 4-bit, 8-bit, or 16-bit",
                    ),
                    rx.select.content(
                        rx.select.item("4-bit", value="4-bit"),
                        rx.select.item("8-bit", value="8-bit"),
                        rx.select.item("16-bit", value="16-bit"),
                    ),
                    value=MultiRunState.quantization,
                    on_change=MultiRunState.set_quantization,
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
                    value=MultiRunState.num_runs.to_string(),
                    on_change=MultiRunState.set_num_runs,
                    type="number",
                    size="2",
                    class_name="bp-num",
                    aria_label="Number of independent training runs to launch",
                ),
                _err_text(MultiRunState.num_runs_error),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Samples per run"),
                rx.input(
                    placeholder="500",
                    value=MultiRunState.samples_per_run.to_string(),
                    on_change=MultiRunState.set_samples_per_run,
                    type="number",
                    size="2",
                    class_name="bp-num",
                    aria_label="Number of training samples per run",
                ),
                _err_text(MultiRunState.samples_per_run_error),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Merge mode"),
                rx.select.root(
                    rx.select.trigger(
                        placeholder="slao",
                        style={"width": "100%"},
                        aria_label="LoRA merge mode — SLAO, Weighted, or TIES",
                    ),
                    rx.select.content(
                        rx.select.item("SLAO (default)", value="slao"),
                        rx.select.item("Weighted", value="weighted"),
                        rx.select.item("TIES", value="ties"),
                    ),
                    value=MultiRunState.merge_mode,
                    on_change=MultiRunState.set_merge_mode,
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
                        # FRONTEND-9 (Wave 6b): mirror the runs.py empty-state
                        # pattern — name a concrete next action rather than
                        # just "no runs yet." Norman 1988 affordance framing.
                        rx.table.row(
                            rx.table.cell(
                                rx.flex(
                                    rx.text(
                                        "No runs in this sweep yet.",
                                        size="2",
                                        style={"color": "var(--bp-text-2)"},
                                    ),
                                    rx.text(
                                        "Set Num runs + Samples per run + Merge mode "
                                        "above, then click Start multi-run. From the "
                                        "shell: `backprop multi-run <model> <dataset> "
                                        "--num-runs N --samples-per-run M`.",
                                        size="1",
                                        style={"color": "var(--bp-muted)"},
                                    ),
                                    direction="column",
                                    gap="2",
                                    align="center",
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
                            rx.cond(
                                MultiRunState.run_state == "loading",
                                rx.spinner(size="2"),
                                rx.fragment(),
                            ),
                            rx.cond(
                                MultiRunState.run_state == "loading",
                                rx.text("Starting…"),
                                rx.text("Start multi-run"),
                            ),
                            variant="solid",
                            color_scheme="teal",
                            size="3",
                            # FRONTEND-B-003: disable while a sweep is in flight.
                            disabled=(MultiRunState.run_state == "loading")
                            | (MultiRunState.run_state == "active"),
                            on_click=MultiRunState.start_multi_run,
                        ),
                        gap="3",
                        margin_top="2",
                        align="center",
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
