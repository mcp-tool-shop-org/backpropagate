"""Run-detail page — ``/runs/[rid]`` — per-run drill-down.

V1_3_BRIEF P1 / Wave 6b FRONTEND-4. Closes the read-only-summary-only gap
that Wave 6 deferred (``runs.py`` notes "Drill-down to a per-run page is
INTENTIONALLY OUT OF SCOPE for v1.2.0; v1.3 adds a /runs/<id> route that
mirrors ``backprop show-run``.").

Surfaces from V1_3_BRIEF:

- Run metadata header (run_id, status, model, dataset, started_at,
  completed_at, duration, outcome, checkpoint_path)
- Hyperparameter table (config dump)
- Training metrics chart (loss — from ``training_metrics.jsonl`` for the run)
- Checkpoint list (numbered checkpoint dirs with size + timestamp)
- Logs viewer (read-only; tail of ``training.log`` for the run)
- Action buttons: "Diff", "Replay", "Delete run", "Export this run".
  Only Diff shells out — to ``backprop diff-runs`` with a ``--``
  end-of-options separator and run-id allowlist validation (UI-A-003).
  Replay / Delete / Export run IN-PROCESS via ``RunHistoryManager``
  (UI-A-004): there is no ``delete-run`` subcommand, ``export-runs`` has no
  ``--run-id`` flag, and ``replay`` has no ``--dry-run`` flag, so the prior
  shell-outs for those three were broken phantom-CLI surfaces. Replay now
  validates replayability in-process; Export writes a single-run JSONL to a
  sandboxed path; Delete removes the run via ``RunHistoryManager.delete_run``.

Routing: ``/runs/[rid]`` is a Reflex dynamic route; the parameter is exposed
via ``self.router.page.params.get("rid")`` inside the state's ``load_run``
event (see ``ui_state.RunDetailState``). The route arg is named ``rid``
(not ``run_id``) so it doesn't shadow a state var.
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import RunDetailState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.error_callout import BpErrorCallout
from ..components.loss_chart import BpLossChart


def _label(text: str) -> rx.Component:
    return rx.text(
        text,
        size="1",
        style={
            "color": "var(--bp-text-2)",
            "font_size": "11px",
            "text_transform": "uppercase",
            "letter_spacing": "0.06em",
            "margin_bottom": "4px",
        },
    )


def _value(text_var) -> rx.Component:
    return rx.text(
        text_var,
        size="2",
        style={
            "color": "var(--bp-text)",
            "font_family": "var(--bp-mono)",
            "font_size": "12px",
            "word_break": "break-all",
        },
    )


def _metadata_header() -> rx.Component:
    """Run metadata — id, status, model, dataset, times, outcome, path."""
    return rx.flex(
        rx.flex(
            rx.heading(
                "Run " + RunDetailState.current_run_id,
                size="5",
                style={"color": "var(--bp-text)", "font_weight": "500"},
            ),
            rx.cond(
                RunDetailState.status == "completed",
                rx.badge("completed", color_scheme="green", variant="soft", size="2"),
                rx.cond(
                    RunDetailState.status == "running",
                    rx.badge("running", color_scheme="teal", variant="soft", size="2"),
                    rx.cond(
                        RunDetailState.status == "failed",
                        rx.badge("failed", color_scheme="red", variant="soft", size="2"),
                        rx.cond(
                            # VIS-UI-003: parity with the runs-list badge
                            # (runs.py ``_status_badge``) which gives an
                            # interrupted run an amber badge.
                            RunDetailState.status == "interrupted",
                            rx.badge("interrupted", color_scheme="amber", variant="soft", size="2"),
                            rx.badge(RunDetailState.status, color_scheme="gray", variant="soft", size="2"),
                        ),
                    ),
                ),
            ),
            direction="row",
            align="center",
            gap="3",
        ),
        rx.grid(
            rx.flex(_label("Model"), _value(RunDetailState.model), direction="column"),
            rx.flex(_label("Dataset"), _value(RunDetailState.dataset), direction="column"),
            rx.flex(_label("Started"), _value(RunDetailState.started_at), direction="column"),
            rx.flex(_label("Completed"), _value(RunDetailState.completed_at), direction="column"),
            rx.flex(_label("Duration"), _value(RunDetailState.duration), direction="column"),
            rx.flex(_label("Final loss"), _value(RunDetailState.final_loss), direction="column"),
            rx.flex(
                _label("Checkpoint path"),
                # UI-A-002: render the redacted form (home prefix stripped) so
                # the operator's username never appears in the UI / screenshots
                # and the full path never ships in the WS state bundle.
                _value(RunDetailState.checkpoint_path_display),
                direction="column",
                style={"grid_column": "1 / span 2"},
            ),
            columns="1fr 1fr",
            gap="3",
            width="100%",
        ),
        direction="column",
        gap="3",
        padding="4",
        style={
            "background": "var(--bp-surface)",
            "border": "1px solid var(--bp-border)",
            "border_radius": "var(--bp-r-2)",
        },
    )


def _hyperparam_table() -> rx.Component:
    """Config dump as a 2-column key/value table."""
    eyebrow = {
        "color": "var(--bp-text-2)",
        "font_size": "10px",
        "text_transform": "uppercase",
        "letter_spacing": "0.06em",
        "font_weight": "500",
    }
    return rx.flex(
        rx.text(
            "Hyperparameters",
            size="3",
            style={"color": "var(--bp-text)", "font_weight": "500"},
        ),
        rx.cond(
            RunDetailState.hyperparameters.length() == 0,
            rx.text(
                "No additional hyperparameter fields recorded for this run.",
                size="1",
                style={"color": "var(--bp-muted)"},
            ),
            rx.flex(
                rx.grid(
                    rx.text("Key", style=eyebrow),
                    rx.text("Value", style=eyebrow),
                    columns="200px 1fr",
                    gap="3",
                    width="100%",
                    padding_x="3",
                    padding_y="2",
                    style={"border_bottom": "1px solid var(--bp-border)"},
                ),
                rx.foreach(
                    RunDetailState.hyperparameters,
                    lambda row: rx.grid(
                        rx.text(
                            row["key"],
                            size="2",
                            class_name="bp-num",
                            style={
                                "font_family": "var(--bp-mono)",
                                "color": "var(--bp-peach)",
                                "font_size": "12px",
                            },
                        ),
                        rx.text(
                            row["value"],
                            size="2",
                            style={
                                "font_family": "var(--bp-mono)",
                                "color": "var(--bp-text)",
                                "font_size": "12px",
                                "word_break": "break-all",
                            },
                        ),
                        columns="200px 1fr",
                        gap="3",
                        width="100%",
                        padding_x="3",
                        padding_y="2",
                        style={"border_bottom": "1px solid var(--bp-border)"},
                    ),
                ),
                direction="column",
                width="100%",
            ),
        ),
        direction="column",
        gap="2",
    )


def _metrics_chart() -> rx.Component:
    """Loss-curve sparkline backed by RunDetailState.loss_history.

    BpSparkline accepts a state-backed list; the SVG re-renders server-side
    when the state changes (same pattern the side-rail uses).
    """
    return rx.flex(
        rx.text(
            "Training loss",
            size="3",
            style={"color": "var(--bp-text)", "font_weight": "500"},
        ),
        rx.cond(
            RunDetailState.loss_history.length() == 0,
            rx.text(
                "No loss history recorded for this run. The training script "
                "writes `loss_history` into the run entry when the metrics "
                "callback is enabled (`callbacks=[\"history\"]`).",
                size="1",
                style={"color": "var(--bp-muted)"},
            ),
            BpLossChart(
                RunDetailState.loss_chart_data,
                height=200,
                label="loss",
            ),
        ),
        direction="column",
        gap="2",
        padding="4",
        style={
            "background": "var(--bp-surface)",
            "border": "1px solid var(--bp-border)",
            "border_radius": "var(--bp-r-2)",
        },
    )


def _checkpoint_list() -> rx.Component:
    """List of numbered checkpoint dirs with size + timestamp."""
    eyebrow = {
        "color": "var(--bp-text-2)",
        "font_size": "10px",
        "text_transform": "uppercase",
        "letter_spacing": "0.06em",
        "font_weight": "500",
    }
    return rx.flex(
        rx.text(
            "Checkpoints",
            size="3",
            style={"color": "var(--bp-text)", "font_weight": "500"},
        ),
        rx.cond(
            RunDetailState.checkpoints.length() == 0,
            rx.text(
                "No checkpoints found at the recorded path. The directory "
                "may have been moved, pruned, or never written (some run "
                "modes only emit a final adapter without intermediate "
                "snapshots).",
                size="1",
                style={"color": "var(--bp-muted)"},
            ),
            rx.flex(
                rx.grid(
                    rx.text("Name", style=eyebrow),
                    rx.text("Size", style=eyebrow),
                    rx.text("Modified", style=eyebrow),
                    columns="2fr 100px 200px",
                    gap="3",
                    width="100%",
                    padding_x="3",
                    padding_y="2",
                    style={"border_bottom": "1px solid var(--bp-border)"},
                ),
                rx.foreach(
                    RunDetailState.checkpoints,
                    lambda cp: rx.grid(
                        rx.text(
                            cp["name"],
                            size="2",
                            class_name="bp-num",
                            style={
                                "font_family": "var(--bp-mono)",
                                "color": "var(--bp-peach)",
                                "font_size": "12px",
                            },
                        ),
                        rx.text(
                            # f-string, not ``+``: ``cp["size_mb"]`` is an
                            # untyped foreach-item Var; ``Var + str`` raises
                            # TypeError at compile.
                            f"{cp['size_mb']} MB",
                            size="2",
                            class_name="bp-num",
                            style={
                                "font_family": "var(--bp-mono)",
                                "color": "var(--bp-text-2)",
                                "font_size": "11px",
                            },
                        ),
                        rx.text(
                            cp["timestamp"],
                            size="1",
                            style={"color": "var(--bp-muted)", "font_size": "11px"},
                        ),
                        columns="2fr 100px 200px",
                        gap="3",
                        width="100%",
                        padding_x="3",
                        padding_y="2",
                        style={"border_bottom": "1px solid var(--bp-border)"},
                    ),
                ),
                direction="column",
                width="100%",
            ),
        ),
        direction="column",
        gap="2",
    )


def _log_viewer() -> rx.Component:
    """Read-only tail of training.log."""
    return rx.flex(
        rx.text(
            "Log tail (last 200 lines)",
            size="3",
            style={"color": "var(--bp-text)", "font_weight": "500"},
        ),
        rx.cond(
            RunDetailState.log_lines.length() == 0,
            rx.text(
                "No training.log found alongside the checkpoint. "
                "Configure `BACKPROPAGATE_LOG_FILE` to capture per-run logs.",
                size="1",
                style={"color": "var(--bp-muted)"},
            ),
            rx.scroll_area(
                rx.flex(
                    rx.foreach(
                        RunDetailState.log_lines,
                        lambda line: rx.text(
                            line,
                            size="1",
                            style={
                                "font_family": "var(--bp-mono)",
                                "color": "var(--bp-text)",
                                "font_size": "11px",
                                "white_space": "pre",
                            },
                        ),
                    ),
                    direction="column",
                    gap="0",
                ),
                type="auto",
                scrollbars="vertical",
                style={
                    "height": "300px",
                    "background": "var(--bp-bg)",
                    "border": "1px solid var(--bp-border)",
                    "border_radius": "var(--bp-r-2)",
                    "padding": "8px",
                },
            ),
        ),
        direction="column",
        gap="2",
    )


def _action_panel() -> rx.Component:
    """Diff / Replay / Delete / Export action row — all 4 V1_3_BRIEF actions.

    Each button shells out to the bridge subcommand via the state's handler:

    - **Diff** (FRONTEND-A-001, v1.4 Wave 2): inline form with a text input
      for the comparison run id + Compare button. Wires
      ``RunDetailState.diff_against`` end-to-end — pre-v1.4, the handler
      existed but no UI control invoked it (closed the producer-without-
      consumer dead-state). The result / error surfaces via the same
      ``action_result`` / ``action_error`` panes the other actions use.
    - **Replay** — validates replayability IN-PROCESS via
      ``RunHistoryManager`` and surfaces the resolved hyperparams so the
      operator sees them before re-launching (no ``backprop replay``
      shell-out — that subcommand has no ``--dry-run`` flag).
    - **Export this run** — writes a single-run JSONL dump IN-PROCESS via
      ``RunHistoryManager`` to a sandboxed path (no ``export-runs``
      shell-out — there is no ``--run-id`` flag).
    - **Delete run** — removes the run IN-PROCESS via
      ``RunHistoryManager.delete_run`` (no ``delete-run`` subcommand exists).

    The diff form is rendered above the action row so the operator's eye
    lands on it first when arriving at the panel after a multi-run pass.
    """
    return rx.flex(
        rx.text(
            "Actions",
            size="3",
            style={"color": "var(--bp-text)", "font_weight": "500"},
        ),
        # FRONTEND-A-001 (v1.4 Wave 2): Diff form — text input + Compare
        # button. Wired to ``RunDetailState.set_diff_other_run_id`` for the
        # input and ``diff_with_input`` for the submit. The form sits above
        # the action row so it doesn't crowd the other 3 actions (Replay /
        # Delete / Export, all in-process via RunHistoryManager). Diff is the
        # only shell-out (``backprop diff-runs``).
        rx.flex(
            rx.text(
                "Diff against another run",
                size="1",
                style={
                    "color": "var(--bp-text-2)",
                    "font_size": "11px",
                    "text_transform": "uppercase",
                    "letter_spacing": "0.06em",
                },
            ),
            rx.flex(
                rx.input(
                    placeholder="other run id (full or prefix)",
                    value=RunDetailState.diff_other_run_id,
                    on_change=RunDetailState.set_diff_other_run_id,
                    size="2",
                    style={"flex_grow": "1", "min_width": "0"},
                    aria_label="Comparison run id for diff (full or prefix)",
                ),
                rx.button(
                    "Compare",
                    on_click=RunDetailState.diff_with_input,
                    variant="soft",
                    color_scheme="teal",
                    size="2",
                    # FRONTEND-B-014-EXTENDED (Stage C humanization): match
                    # the action-row disabled-while-in-flight pattern so the
                    # operator can't queue a Compare on top of a running
                    # diff / replay / export / delete.
                    disabled=RunDetailState.action_in_flight != "",
                    aria_label="Diff this run against the comparison run id",
                ),
                direction="row",
                gap="2",
                width="100%",
                align="center",
            ),
            direction="column",
            gap="1",
            width="100%",
        ),
        rx.flex(
            rx.button(
                "Replay (dry-run)",
                on_click=RunDetailState.replay,
                variant="soft",
                color_scheme="teal",
                size="2",
                # FRONTEND-B-014-EXTENDED (Stage C humanization): disable
                # the action buttons while ANY shell-out is in flight so a
                # second click can't queue an overlapping subprocess.
                disabled=RunDetailState.action_in_flight != "",
                aria_label="Dry-run a replay of this run via the CLI",
            ),
            rx.button(
                "Export this run",
                on_click=RunDetailState.export_run,
                variant="soft",
                color_scheme="teal",
                size="2",
                disabled=RunDetailState.action_in_flight != "",
                aria_label="Export this run to a JSONL file via the CLI",
            ),
            rx.button(
                "Delete run",
                on_click=RunDetailState.delete_run,
                variant="soft",
                color_scheme="red",
                size="2",
                disabled=RunDetailState.action_in_flight != "",
                aria_label="Delete this run from history via the CLI",
            ),
            direction="row",
            gap="2",
            wrap="wrap",
        ),
        # FRONTEND-B-014-EXTENDED (Stage C humanization): inline "Running
        # <subcommand>…" status row while a shell-out is in flight. The
        # diff / replay / delete / export handlers each block synchronously
        # for up to 30-60s; pre-fix the operator clicked the button and saw
        # NO feedback until the subprocess returned (the panel read as a
        # frozen UI). aria_live=polite announces the running state to screen
        # readers; the spinner gives sighted operators the visible cue.
        rx.cond(
            RunDetailState.action_in_flight != "",
            rx.box(
                rx.flex(
                    rx.spinner(size="2"),
                    rx.text(
                        "Running ",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    rx.text(
                        RunDetailState.action_in_flight,
                        size="2",
                        style={
                            "color": "var(--bp-peach)",
                            "font_family": "var(--bp-mono)",
                        },
                    ),
                    rx.text(
                        "… (this can take up to a minute)",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    direction="row",
                    gap="2",
                    align="center",
                    padding="3",
                    style={
                        "background": "var(--bp-surface-2)",
                        "border": "1px dashed var(--bp-border)",
                        "border_radius": "var(--bp-r-2)",
                    },
                ),
                role="status",
                aria_live="polite",
            ),
            rx.fragment(),
        ),
        rx.cond(
            RunDetailState.action_result != "",
            rx.flex(
                rx.text(
                    RunDetailState.action_result,
                    size="1",
                    style={
                        "color": "var(--bp-text)",
                        "font_family": "var(--bp-mono)",
                        "font_size": "11px",
                        "white_space": "pre-wrap",
                    },
                ),
                rx.button("Dismiss", on_click=RunDetailState.clear_action_message, variant="ghost", size="1"),
                direction="column",
                gap="2",
                padding="3",
                style={
                    "background": "var(--bp-surface-2)",
                    "border": "1px solid var(--bp-teal)",
                    "border_radius": "var(--bp-r-2)",
                },
            ),
            rx.fragment(),
        ),
        rx.cond(
            RunDetailState.action_error != "",
            # FRONTEND-A-004 (v1.4 Wave 2): consolidated via ``BpErrorCallout``.
            # Pre-fix this rolled its own ``rx.flex`` chrome with a peach
            # border; now uses the canonical component (same as runs /
            # models) so action failures share ARIA semantics + design-
            # digest-conformant styling. The Dismiss button stays outside
            # the component (the callout itself is content-only — wrapping
            # state-bound dismiss would couple the component to a state
            # surface it doesn't own).
            rx.flex(
                BpErrorCallout(
                    code="UI · RUN-DETAIL",
                    title="Action failed",
                    message=RunDetailState.action_error,
                    # HUX-02: remedy slot wired to the dedicated suggestion
                    # var so it renders on its own dimmed line (design-digest
                    # §4d) rather than as a run-on. Empty today → fragment.
                    hint=RunDetailState.action_error_suggestion,
                ),
                rx.button(
                    "Dismiss",
                    on_click=RunDetailState.clear_action_message,
                    variant="ghost",
                    size="1",
                    style={"align_self": "flex-end"},
                ),
                direction="column",
                gap="2",
                width="100%",
            ),
            rx.fragment(),
        ),
        direction="column",
        gap="2",
    )


def _not_found() -> rx.Component:
    return rx.flex(
        rx.text(
            "Run not found.",
            size="4",
            style={"color": "var(--bp-text)", "font_weight": "500"},
        ),
        rx.text(
            RunDetailState.error,
            size="2",
            style={"color": "var(--bp-muted)"},
        ),
        rx.link(
            "← Back to run history",
            href="/runs",
            style={"color": "var(--bp-teal)"},
        ),
        direction="column",
        gap="3",
        padding="6",
        align="center",
    )


def _was_deleted() -> rx.Component:
    """FRONTEND-B-001 (v1.4 Wave 3.5): post-successful-delete chrome.

    Renders when ``RunDetailState.was_deleted`` is true — distinct from
    ``_not_found()`` (unknown-id navigation surface). The pre-fix code
    set ``not_found=True`` after a successful delete which latched the
    operator into the not-found chrome and stranded the success message
    in ``action_result`` behind a template that never displays it. This
    chrome shows the deletion confirmation + an explicit "Back to runs
    list" button so the next action is unambiguous.
    """
    return rx.flex(
        rx.text(
            "Run deleted.",
            size="4",
            style={"color": "var(--bp-text)", "font_weight": "500"},
        ),
        rx.text(
            RunDetailState.action_result,
            size="2",
            style={"color": "var(--bp-muted)"},
        ),
        rx.button(
            rx.icon("arrow-left", size=16),
            "Back to runs list",
            on_click=lambda: rx.redirect("/runs"),
            variant="soft",
            style={"margin_top": "8px"},
        ),
        direction="column",
        gap="3",
        padding="6",
        align="center",
    )


def run_detail_page() -> rx.Component:
    """The ``/runs/[rid]`` route."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="runs"),
            rx.scroll_area(
                rx.flex(
                    rx.cond(
                        RunDetailState.loading,
                        # FRONTEND-B-014 (Stage C accessibility): wrap loading
                        # row in role=status / aria_live so AT users hear the
                        # state on page mount. Pre-fix the spinner was visually
                        # present but silent for screen readers.
                        rx.box(
                            rx.flex(
                                rx.spinner(size="2"),
                                rx.text(
                                    "Loading run details…",
                                    size="2",
                                    style={"color": "var(--bp-muted)"},
                                ),
                                direction="row",
                                gap="2",
                                align="center",
                                padding="4",
                            ),
                            role="status",
                            aria_live="polite",
                        ),
                        rx.fragment(),
                    ),
                    # FRONTEND-B-001 (v1.4 Wave 3.5): branch on
                    # ``was_deleted`` FIRST so a successful delete renders
                    # the "Run deleted." confirmation chrome, NOT the
                    # "Run not found" surface. The two states are
                    # distinct user-facing situations (the run was just
                    # removed by THIS operator vs. an unknown id arrived
                    # via navigation) and must not collapse into one
                    # template.
                    rx.cond(
                        RunDetailState.was_deleted,
                        _was_deleted(),
                        rx.cond(
                            RunDetailState.not_found,
                            _not_found(),
                            rx.flex(
                                _metadata_header(),
                                _metrics_chart(),
                                _hyperparam_table(),
                                _checkpoint_list(),
                                _log_viewer(),
                                _action_panel(),
                                direction="column",
                                gap="4",
                                width="100%",
                            ),
                        ),
                    ),
                    direction="column",
                    gap="4",
                    padding="6",
                    max_width="980px",
                    width="100%",
                    on_mount=RunDetailState.load_run,
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
