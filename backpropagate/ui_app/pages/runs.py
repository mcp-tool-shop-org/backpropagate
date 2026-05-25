"""Runs page — ``/runs`` — read-only run-history table.

Closes the CLI/UI parity gap identified by FRONTEND-F-RUN-HISTORY-PAGE: the
CLI has ``backprop list-runs`` (cli.py:1845) backed by RunHistoryManager
(checkpoints.py:689), but the Reflex UI had no equivalent surface. Operators
training via the UI could not review prior runs without dropping to the
shell.

Wave 6 v1.2.0 scope (deliberately narrow):

- Data + list page only — NO per-run drill-down. Clicking a row does not
  navigate to ``/runs/<id>``; that route is v1.3 scope.
- No pagination — the table caps at 50 most-recent rows (RunsState default).
  v1.3 adds "Load more" + status-filter pills.
- Status filter exists in state (RunsState.status_filter) but the UI binding
  is a single ``select`` rather than the pill row planned for v1.3.

Component tree (matches the design digest §3 chrome contract):

- BpHeader / BpLeftNav / BpFooter (shared chrome)
- Hero heading + subtitle
- Status filter + refresh
- Error callout (rx.cond on RunsState.error)
- Loading row (rx.cond on RunsState.loading)
- Empty-state copy (rx.cond on len(runs) == 0)
- Run table (rx.foreach over RunsState.runs)
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import RunsState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.error_callout import BpErrorCallout


def _label(text: str) -> rx.Component:
    """Tiny eyebrow label — matches the train/export page style."""
    return rx.text(
        text,
        size="1",
        style={
            "color": "var(--bp-text-2)",
            "font_size": "11px",
            "margin_bottom": "4px",
        },
    )


def _filter_bar() -> rx.Component:
    """Status filter + refresh button.

    The filter is a select element bound to RunsState.set_status_filter, which
    re-triggers the load on change. The refresh button is for operators who
    want to pick up new runs from a CLI training that started after the page
    loaded; Reflex doesn't auto-poll the filesystem.
    """
    return rx.flex(
        rx.flex(
            _label("Status"),
            rx.select.root(
                rx.select.trigger(
                    placeholder="All statuses",
                    style={"width": "180px"},
                    aria_label="Filter run history by status",
                ),
                rx.select.content(
                    rx.select.item("All statuses", value=""),
                    rx.select.item("Running", value="running"),
                    rx.select.item("Completed", value="completed"),
                    rx.select.item("Failed", value="failed"),
                    rx.select.item("Interrupted", value="interrupted"),
                ),
                value=RunsState.status_filter,
                on_change=RunsState.set_status_filter,
            ),
            direction="column",
        ),
        rx.spacer(),
        rx.button(
            "Refresh",
            on_click=RunsState.load_runs,
            variant="soft",
            color_scheme="teal",
            size="2",
            disabled=RunsState.loading,
        ),
        direction="row",
        align="end",
        gap="3",
        width="100%",
    )


def _table_header() -> rx.Component:
    """Column headers — uppercase eyebrow style to match the rail labels."""
    eyebrow = {
        "color": "var(--bp-text-2)",
        "font_size": "10px",
        "text_transform": "uppercase",
        "letter_spacing": "0.06em",
        "font_weight": "500",
    }
    return rx.grid(
        rx.text("Run ID", style=eyebrow),
        rx.text("Started", style=eyebrow),
        rx.text("Model", style=eyebrow),
        rx.text("Dataset", style=eyebrow),
        rx.text("Status", style=eyebrow),
        rx.text("Duration", style=eyebrow),
        rx.text("Loss", style=eyebrow),
        # Column widths: short id / time / model / dataset / status / dur / loss
        columns="80px 1.2fr 1.4fr 1.4fr 90px 80px 80px",
        gap="3",
        width="100%",
        padding_x="3",
        padding_y="2",
        style={"border_bottom": "1px solid var(--bp-border)"},
    )


def _status_badge(status_var) -> rx.Component:
    """Colored badge for run status. Reflex doesn't have a switch primitive
    we can use on a Var without rx.match — keep it simple with rx.cond.
    """
    return rx.cond(
        status_var == "completed",
        rx.badge("completed", color_scheme="green", variant="soft", size="1"),
        rx.cond(
            status_var == "running",
            rx.badge("running", color_scheme="teal", variant="soft", size="1"),
            rx.cond(
                status_var == "failed",
                rx.badge("failed", color_scheme="red", variant="soft", size="1"),
                rx.cond(
                    status_var == "interrupted",
                    rx.badge("interrupted", color_scheme="amber", variant="soft", size="1"),
                    rx.badge(status_var, color_scheme="gray", variant="soft", size="1"),
                ),
            ),
        ),
    )


def _run_row(run: dict) -> rx.Component:
    """One table row. The run-id cell links to ``/runs/<rid>`` per FRONTEND-4
    (Wave 6b drill-down). Other cells are read-only (clicking the row body
    does NOT navigate; clicking the run-id link does).
    """
    return rx.grid(
        rx.link(
            rx.text(
                run["run_id_short"],
                class_name="bp-num",
                style={
                    "font_family": "var(--bp-mono)",
                    "color": "var(--bp-peach)",
                    "font_size": "12px",
                    "text_decoration": "underline",
                    "text_decoration_color": "var(--bp-peach)",
                    "text_underline_offset": "3px",
                },
            ),
            href="/runs/" + run["run_id"],
            aria_label="Open run detail page for this run id",
        ),
        rx.text(
            run["started_at"],
            size="1",
            style={"color": "var(--bp-muted)", "font_size": "11px"},
        ),
        rx.text(
            run["model"],
            size="2",
            style={
                "color": "var(--bp-text)",
                "overflow": "hidden",
                "text_overflow": "ellipsis",
                "white_space": "nowrap",
            },
        ),
        rx.text(
            run["dataset"],
            size="2",
            style={
                "color": "var(--bp-text)",
                "overflow": "hidden",
                "text_overflow": "ellipsis",
                "white_space": "nowrap",
            },
        ),
        _status_badge(run["status"]),
        rx.text(
            run["duration"],
            class_name="bp-num",
            style={
                "font_family": "var(--bp-mono)",
                "color": "var(--bp-text-2)",
                "font_size": "11px",
            },
        ),
        rx.text(
            run["final_loss"],
            class_name="bp-num",
            style={
                "font_family": "var(--bp-mono)",
                "color": "var(--bp-teal)",
                "font_size": "12px",
            },
        ),
        columns="80px 1.2fr 1.4fr 1.4fr 90px 80px 80px",
        gap="3",
        width="100%",
        padding_x="3",
        padding_y="2",
        style={
            "border_bottom": "1px solid var(--bp-border)",
            "transition": "background 0.15s ease",
        },
        class_name="bp-runs-row",
    )


def _empty_state() -> rx.Component:
    """Empty-state message - only renders when ``runs`` is empty AND no error.

    FRONTEND-B-006 (Stage C humanization): the empty-state copy now names a
    concrete next action (`backprop train ...`) instead of just "runs will
    appear here." Norman 1988 affordance framing - tell the operator what to
    do, not just what the screen is for.
    """
    return rx.flex(
        rx.text(
            "No training runs recorded yet.",
            size="2",
            style={"color": "var(--bp-text-2)"},
        ),
        rx.text(
            "Train a model to see it here. From the UI: open the Single run "
            "or Multi-run tab and click Start. From the shell: run "
            "`backprop train <model> <dataset.jsonl>` and refresh this page.",
            size="1",
            style={"color": "var(--bp-muted)"},
        ),
        direction="column",
        gap="2",
        padding="6",
        align="center",
        width="100%",
    )


def _error_callout() -> rx.Component:
    """Inline error banner — consolidated via ``BpErrorCallout``.

    FRONTEND-A-004 (v1.4 Wave 2): pre-fix this rolled its own
    ``rx.flex`` chrome with a peach border + Dismiss button — the
    canonical ``BpErrorCallout`` (components/error_callout.py) existed
    but no page imported it. Now uses the canonical component so the
    error surface shares ARIA semantics + design-digest-conformant
    styling across pages.

    ``BpErrorCallout`` already renders ``rx.callout.root`` with
    ``color_scheme="red"`` (which carries the accessible role + the
    correct contrast across light/dark themes), so the inline chrome
    here collapses to a single Component + a Dismiss button below.
    The Dismiss button stays in this wrapper (not the component) so
    other consumers of ``BpErrorCallout`` don't auto-acquire a
    state-bound Dismiss they don't have a handler for.
    """
    return rx.flex(
        BpErrorCallout(
            code="UI · RUNS",
            title="Could not load run history",
            message=RunsState.error,
        ),
        rx.button(
            "Dismiss",
            on_click=RunsState.clear_error,
            variant="ghost",
            size="1",
            style={"align_self": "flex-end"},
        ),
        direction="column",
        gap="2",
        width="100%",
    )


def runs_page() -> rx.Component:
    """The ``/runs`` route.

    Uses ``on_mount=RunsState.load_runs`` so the table populates the first
    time the operator navigates here; subsequent visits re-fetch (cheap —
    RunHistoryManager just lists JSON files).
    """
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="runs"),
            rx.scroll_area(
                rx.flex(
                    rx.text(
                        "Run history",
                        size="6",
                        style={"color": "var(--bp-text)", "font_weight": "500"},
                    ),
                    rx.text(
                        "Recent training runs from this output directory. "
                        "Mirrors `backprop list-runs`. Refresh after a CLI "
                        "training to pick up new entries.",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    _filter_bar(),
                    rx.cond(
                        RunsState.error != "",
                        _error_callout(),
                        rx.fragment(),
                    ),
                    rx.cond(
                        RunsState.loading,
                        rx.flex(
                            rx.spinner(size="2"),
                            rx.text(
                                "Loading runs…",
                                size="2",
                                style={"color": "var(--bp-muted)"},
                            ),
                            direction="row",
                            gap="2",
                            align="center",
                            padding="4",
                        ),
                        rx.fragment(),
                    ),
                    # FRONTEND-B-006 (Stage C humanization): mutex between
                    # error / empty-state / table. When an error is present
                    # we render ONLY the callout above (no empty-state table
                    # chrome below it). When there is no error and no rows,
                    # we render the empty-state with its concrete next-action
                    # copy. When there are rows, we render the table.
                    rx.cond(
                        RunsState.error != "",
                        rx.fragment(),
                        rx.cond(
                            RunsState.runs.length() == 0,
                            rx.cond(
                                ~RunsState.loading,
                                _empty_state(),
                                rx.fragment(),
                            ),
                            rx.flex(
                                _table_header(),
                                rx.foreach(RunsState.runs, _run_row),
                                direction="column",
                                width="100%",
                                style={
                                    "background": "var(--bp-surface)",
                                    "border": "1px solid var(--bp-border)",
                                    "border_radius": "var(--bp-r-2)",
                                    "overflow": "hidden",
                                },
                            ),
                        ),
                    ),
                    rx.cond(
                        RunsState.last_loaded_at != "",
                        rx.text(
                            f"Loaded {RunsState.runs.length()} run(s) at "
                            + RunsState.last_loaded_at,
                            size="1",
                            style={"color": "var(--bp-muted-2)", "font_size": "10px"},
                        ),
                        rx.fragment(),
                    ),
                    direction="column",
                    gap="4",
                    padding="6",
                    max_width="980px",
                    width="100%",
                    on_mount=RunsState.load_runs,
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
