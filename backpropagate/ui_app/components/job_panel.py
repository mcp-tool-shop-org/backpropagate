"""Live-job panels shared by the Train, Multi-run and Export pages (ui-v2 P2).

One UI job runs at a time and ``TrainState`` follows it whatever its kind, so
the pages share these pieces: the reattach banner, the start-refusal and
failure callouts (with the log tail), the progress card, the loss chart and
the post-job panel. Moved out of ``pages/train.py``, which keeps its private
aliases.
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import TrainState

from .field import bp_label as _label
from .group import _card_style
from .loss_chart import BpLossChart
from .recovery_banner import BpRecoveryBanner


def _reattach_banner() -> rx.Component:
    """Shown ONLY when this page adopted a run it didn't start (fix-round).

    Pre-fix the page surfaced the latest ok/warn event as "Recovered." /
    "Heads-up." banners — which fired after every NORMAL finish too, leaving
    the stopped state with both banners up. Recovery framing belongs to
    reattach, so the banner keys off ``reattach_notice``.
    """
    return rx.cond(
        TrainState.reattach_notice != "",
        BpRecoveryBanner(
            variant="ok",
            lead="Reattached.",
            body=TrainState.reattach_notice,
        ),
        rx.fragment(),
    )


def _refusal_callout() -> rx.Component:
    """Start-refusal banner — validation + one-job refusals land on screen."""
    return rx.cond(
        TrainState.job_refusal != "",
        rx.flex(
            rx.text(
                TrainState.job_refusal,
                size="2",
                style={"color": "var(--bp-text)", "flex_grow": "1"},
            ),
            rx.button(
                "Dismiss",
                variant="ghost",
                color_scheme="gray",
                size="1",
                on_click=TrainState.dismiss_refusal,
                style={"border_radius": "var(--bp-r-pill)"},
            ),
            padding="var(--space-4)",
            gap="var(--space-3)",
            align="center",
            width="100%",
            style={
                "background": "var(--bp-surface)",
                "border": "1px solid var(--bp-peach)",
                "border_radius": "var(--bp-r-lg)",
                "box_shadow": "var(--bp-shadow-card)",
            },
            role="status",
            aria_live="polite",
        ),
        rx.fragment(),
    )


def _error_callout() -> rx.Component:
    """Structured failure callout — code first, then message/hint."""
    from .error_callout import BpErrorCallout

    return rx.cond(
        TrainState.run_state == "error",
        rx.flex(
            rx.cond(
                TrainState.job_error_code != "",
                BpErrorCallout(
                    code=TrainState.job_error_code,
                    title="Run failed",
                    message=TrainState.job_error_message,
                    hint=TrainState.job_error_hint,
                ),
                rx.fragment(),
            ),
            # Requirement 10: the last log lines, so a failure is explainable
            # without opening a terminal.
            rx.cond(
                TrainState.job_log_tail.length() > 0,
                rx.box(
                    rx.text(
                        "Last lines of the log",
                        size="1",
                        weight="medium",
                        style={"color": "var(--bp-text-2)", "font_size": "12px"},
                    ),
                    rx.box(
                        rx.foreach(
                            TrainState.job_log_tail,
                            lambda line: rx.text(
                                line,
                                style={
                                    "font_family": "var(--bp-mono)",
                                    "font_size": "12px",
                                    "color": "var(--bp-text)",
                                    "white_space": "pre-wrap",
                                    "overflow_wrap": "anywhere",
                                },
                            ),
                        ),
                        margin_top="var(--space-2)",
                        padding="var(--space-3)",
                        style={
                            "background": "var(--bp-surface-2)",
                            "border_radius": "var(--bp-r-2)",
                            "max_height": "260px",
                            "overflow_y": "auto",
                        },
                    ),
                    padding="var(--space-4)",
                    width="100%",
                    style=_card_style(),
                ),
                rx.fragment(),
            ),
            direction="column",
            gap="var(--space-3)",
            width="100%",
        ),
        rx.fragment(),
    )


def _chip(text, *, tone: str = "teal") -> rx.Component:
    """Small pill chip (progress card)."""
    colors = {
        "teal": ("var(--bp-teal)",),
        "amber": ("var(--bp-amber)",),
        "muted": ("var(--bp-muted)",),
    }
    color = colors.get(tone, colors["teal"])[0]
    return rx.text(
        text,
        size="1",
        weight="medium",
        style={
            "color": color,
            "font_size": "11px",
            "letter_spacing": "0.06em",
            "text_transform": "uppercase",
            "background": f"color-mix(in srgb, {color} 12%, transparent)",
            "border": f"1px solid color-mix(in srgb, {color} 30%, transparent)",
            "border_radius": "var(--bp-r-pill)",
            "padding": "3px 10px",
        },
    )


def _run_progress_card() -> rx.Component:
    """The live run's own card (Design bar: large counter, rounded bar,
    heartbeat, ETA, loss numbers).

    "TRAINING" text stays in the DOM — the P1 acceptance test waits on it.
    """
    return rx.cond(
        TrainState.run_state == "active",
        rx.box(
            # row 1: state chip, phase, run id | stalled, heartbeat, ETA
            rx.flex(
                _chip(TrainState.job_chip_label),
                rx.cond(
                    TrainState.job_run_label != "",
                    rx.text(
                        TrainState.job_run_label,
                        size="1",
                        weight="medium",
                        style={"color": "var(--bp-text)", "font_size": "12px"},
                    ),
                    rx.fragment(),
                ),
                rx.text(
                    TrainState.job_phase,
                    size="1",
                    style={"color": "var(--bp-text-2)", "font_size": "12px"},
                ),
                rx.text(
                    "run " + TrainState.job_id,
                    size="1",
                    class_name="bp-num",
                    style={"color": "var(--bp-muted-2)", "font_size": "11px"},
                ),
                rx.spacer(),
                rx.cond(
                    TrainState.job_stalled,
                    _chip("no progress for 2 min", tone="amber"),
                    rx.fragment(),
                ),
                rx.text(
                    TrainState.heartbeat_label,
                    size="1",
                    class_name="bp-num",
                    style={"color": "var(--bp-muted)", "font_size": "12px"},
                ),
                rx.text(
                    TrainState.eta_label,
                    size="1",
                    class_name="bp-num",
                    style={"color": "var(--bp-muted)", "font_size": "12px"},
                ),
                gap="var(--space-3)",
                align="center",
                width="100%",
                wrap="wrap",
            ),
            # row 2: big step counter + loss numbers; an export has phases,
            # not steps, so it shows the phase in the counter's place.
            rx.cond(
                TrainState.job_has_steps,
                rx.flex(
                    rx.flex(
                        rx.text(
                            TrainState.current_step.to_string(),
                            class_name="bp-num bp-tick",
                            style={
                                "color": "var(--bp-text)",
                                "font_size": "34px",
                                "font_weight": "600",
                                "letter_spacing": "-0.02em",
                                "line_height": "1",
                            },
                        ),
                        rx.text(
                            " / " + TrainState.job_total_steps.to_string(),
                            class_name="bp-num",
                            style={
                                "color": "var(--bp-muted)",
                                "font_size": "18px",
                                "line_height": "1",
                            },
                        ),
                        direction="row",
                        align="baseline",
                        gap="var(--space-1)",
                    ),
                    rx.spacer(),
                    rx.flex(
                        rx.flex(
                            _label("loss"),
                            rx.text(
                                TrainState.loss_label,
                                class_name="bp-num",
                                style={
                                    "color": "var(--bp-text)",
                                    "font_size": "18px",
                                    "line_height": "1.1",
                                },
                            ),
                            direction="column",
                            align="end",
                            gap="var(--space-1)",
                        ),
                        rx.flex(
                            _label("smoothed"),
                            rx.text(
                                TrainState.ema_loss_label,
                                class_name="bp-num",
                                style={
                                    "color": "var(--bp-teal)",
                                    "font_size": "18px",
                                    "font_weight": "600",
                                    "line_height": "1.1",
                                },
                            ),
                            direction="column",
                            align="end",
                            gap="var(--space-1)",
                        ),
                        direction="row",
                        gap="var(--space-5)",
                        align="center",
                    ),
                    direction="row",
                    align="center",
                    width="100%",
                ),
                rx.text(
                    TrainState.job_phase,
                    class_name="bp-num",
                    style={
                        "color": "var(--bp-text)",
                        "font_size": "28px",
                        "font_weight": "600",
                        "letter_spacing": "-0.02em",
                        "text_transform": "capitalize",
                        "margin_top": "var(--space-2)",
                    },
                ),
            ),
            # row 3: the rounded progress bar (10px pill track + teal fill)
            rx.box(
                rx.cond(
                    TrainState.job_has_steps,
                    rx.box(
                        width=TrainState.step_progress_pct,
                        height="100%",
                        background="var(--bp-teal)",
                        border_radius="var(--bp-r-pill)",
                        style={"transition": "width 0.4s ease-out"},
                    ),
                    # No step count to measure: a full, pulsing bar says
                    # "working" without inventing a percentage.
                    rx.box(
                        width="100%",
                        height="100%",
                        background="var(--bp-teal)",
                        border_radius="var(--bp-r-pill)",
                        class_name="bp-pulse-1600",
                    ),
                ),
                width="100%",
                height="10px",
                margin_top="var(--space-4)",
                background="var(--bp-surface-3)",
                border_radius="var(--bp-r-pill)",
                overflow="hidden",
                role="progressbar",
                aria_label="Job progress",
            ),
            padding="var(--space-5)",
            width="100%",
            style={
                **_card_style(),
                "border": "1px solid color-mix(in srgb, var(--bp-teal) 30%, var(--bp-border))",
            },
            role="status",
            aria_live="polite",
        ),
        rx.fragment(),
    )


def _loss_chart_card() -> rx.Component:
    """Loss in its own card — raw (faint) + debiased EMA (bold)."""
    return rx.cond(
        TrainState.loss_history.length() > 0,
        rx.box(
            rx.flex(
                rx.text(
                    "Training loss",
                    size="3",
                    weight="medium",
                    style={"color": "var(--bp-text)", "font_size": "16px"},
                ),
                rx.spacer(),
                rx.text(
                    "raw faint · smoothed bold",
                    size="1",
                    style={"color": "var(--bp-muted)", "font_size": "11px"},
                ),
                align="baseline",
                width="100%",
            ),
            rx.box(
                BpLossChart(
                    TrainState.loss_chart_data,
                    height=170,
                    label="loss",
                    ema_label="ema",
                ),
                margin_top="var(--space-3)",
                width="100%",
            ),
            padding="var(--space-5)",
            width="100%",
            style=_card_style(),
        ),
        rx.fragment(),
    )


def _next_steps_panel() -> rx.Component:
    """Post-run affordances — export / view checkpoints / start another."""
    return rx.cond(
        TrainState.run_complete,
        rx.box(
            rx.text(
                TrainState.done_title,
                size="3",
                weight="medium",
                style={"color": "var(--bp-text)", "font_size": "16px"},
            ),
            rx.cond(
                TrainState.job_kind != "export",
                rx.flex(
                    rx.link(
                        rx.button(
                            "Export to GGUF",
                            variant="soft",
                            color_scheme="teal",
                            size="2",
                            style={"border_radius": "var(--bp-r-pill)"},
                            aria_label="Convert this adapter to GGUF for Ollama or llama.cpp",
                        ),
                        href="/export",
                    ),
                    rx.link(
                        rx.button(
                            "Push to HF Hub",
                            variant="soft",
                            color_scheme="teal",
                            size="2",
                            style={"border_radius": "var(--bp-r-pill)"},
                            aria_label="Push the trained adapter to a HuggingFace repo",
                        ),
                        href="/export",
                    ),
                    rx.link(
                        rx.button(
                            "Register with Ollama",
                            variant="soft",
                            color_scheme="teal",
                            size="2",
                            style={"border_radius": "var(--bp-r-pill)"},
                            aria_label="Register the model with the local Ollama daemon",
                        ),
                        href="/export",
                    ),
                    rx.link(
                        rx.button(
                            "View checkpoints",
                            variant="soft",
                            color_scheme="gray",
                            size="2",
                            style={"border_radius": "var(--bp-r-pill)"},
                            aria_label="Browse this run's checkpoint files in the runs page",
                        ),
                        href="/runs",
                    ),
                    rx.cond(
                        TrainState.job_kind == "sft",
                        rx.button(
                            "Start another run",
                            variant="ghost",
                            color_scheme="teal",
                            size="2",
                            on_click=TrainState.start_training,
                            style={"border_radius": "var(--bp-r-pill)"},
                            aria_label="Start another run",
                        ),
                        rx.fragment(),
                    ),
                    direction="row",
                    gap="var(--space-3)",
                    wrap="wrap",
                    margin_top="var(--space-2)",
                ),
                rx.fragment(),
            ),
            rx.cond(
                TrainState.job_output_path != "",
                rx.text(
                    "Saved to: " + TrainState.job_output_path,
                    size="1",
                    class_name="bp-num",
                    style={
                        "color": "var(--bp-muted)",
                        "font_size": "11px",
                        "margin_top": "8px",
                        "overflow_wrap": "anywhere",
                    },
                ),
                rx.fragment(),
            ),
            padding="var(--space-5)",
            width="100%",
            style={
                **_card_style(),
                "border": "1px solid color-mix(in srgb, var(--bp-seafoam) 35%, var(--bp-border))",
            },
            role="region",
            aria_label="Run complete — next steps",
        ),
        rx.fragment(),
    )


# Public names for the pages.
job_reattach_banner = _reattach_banner
job_refusal_callout = _refusal_callout
job_error_callout = _error_callout
job_progress_card = _run_progress_card
job_loss_chart_card = _loss_chart_card
job_next_steps_panel = _next_steps_panel
job_chip = _chip
