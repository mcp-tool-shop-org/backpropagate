"""Train page — ``/`` — the single-run surface.

Component tree per design digest §3:

- Hero heading + subtitle
- Recovery banners (when applicable)
- Group "Model"            — HF model id + quantization
- Group "Training shape"   — steps + batch + learning rate
- Group "LoRA tuning"      — r + alpha + dropout + target modules
- Group "Dataset"          — path + format auto-detect
- Group "Advanced" (collapsed) — gpu safety + run name + flags
- Start / Stop button row
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import TrainState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.group import Group
from ..components.recovery_banner import BpRecoveryBanner


def _label(text: str) -> rx.Component:
    """Tiny labelled control eyebrow."""
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
                    default_value=TrainState.model,
                    on_change=TrainState.set_model,
                    size="2",
                    style={"width": "100%"},
                    aria_label="HuggingFace model id",
                ),
                rx.cond(
                    TrainState.model_error != "",
                    rx.text(
                        TrainState.model_error,
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
                    value=TrainState.quantization,
                    on_change=TrainState.set_quantization,
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


def _training_shape_group() -> rx.Component:
    return Group(
        rx.grid(
            rx.flex(
                _label("Steps"),
                rx.input(
                    placeholder="100",
                    value=TrainState.steps.to_string(),
                    on_change=TrainState.set_steps,
                    type="number",
                    size="2",
                    class_name="bp-num",
                    aria_label="Number of training steps",
                ),
                _err_text(TrainState.steps_error),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Batch size"),
                rx.input(
                    placeholder="auto",
                    value=TrainState.batch_size,
                    on_change=TrainState.set_batch_size,
                    size="2",
                    class_name="bp-num",
                    aria_label="Batch size (number or auto)",
                ),
                _err_text(TrainState.batch_size_error),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Learning rate"),
                rx.input(
                    placeholder="2e-4",
                    value=TrainState.learning_rate.to_string(),
                    on_change=TrainState.set_learning_rate,
                    size="2",
                    class_name="bp-num",
                    aria_label="Learning rate",
                ),
                _err_text(TrainState.learning_rate_error),
                direction="column",
                width="100%",
            ),
            columns="repeat(3, 1fr)",
            gap="3",
            width="100%",
        ),
        title="Training shape",
    )


def _lora_group() -> rx.Component:
    return Group(
        rx.grid(
            rx.flex(
                _label("LoRA rank"),
                rx.input(
                    placeholder="16",
                    value=TrainState.lora_r.to_string(),
                    on_change=TrainState.set_lora_r,
                    type="number",
                    size="2",
                    class_name="bp-num",
                    aria_label="LoRA rank (r)",
                ),
                _err_text(TrainState.lora_r_error),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("LoRA alpha"),
                rx.input(
                    placeholder="32",
                    value=TrainState.lora_alpha.to_string(),
                    on_change=TrainState.set_lora_alpha,
                    size="2",
                    class_name="bp-num",
                    aria_label="LoRA alpha",
                ),
                _err_text(TrainState.lora_alpha_error),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("Dropout"),
                rx.input(
                    placeholder="0.05",
                    value=TrainState.lora_dropout.to_string(),
                    on_change=TrainState.set_lora_dropout,
                    size="2",
                    class_name="bp-num",
                    aria_label="LoRA dropout (0 to 1)",
                ),
                _err_text(TrainState.lora_dropout_error),
                direction="column",
                width="100%",
            ),
            columns="repeat(3, 1fr)",
            gap="3",
            width="100%",
        ),
        rx.flex(
            _label("Target modules (comma-separated)"),
            rx.input(
                placeholder="q_proj, k_proj, v_proj, o_proj",
                value=TrainState.target_modules,
                on_change=TrainState.set_target_modules,
                size="2",
                style={"width": "100%"},
                aria_label="LoRA target modules — comma-separated attention layer names",
            ),
            _err_text(TrainState.target_modules_error),
            direction="column",
            width="100%",
        ),
        title="LoRA tuning",
    )


def _dataset_group() -> rx.Component:
    return Group(
        rx.flex(
            _label("Dataset path"),
            rx.input(
                placeholder="path/to/dataset.jsonl",
                default_value=TrainState.dataset_path,
                on_change=TrainState.set_dataset_path,
                size="2",
                style={"width": "100%"},
                aria_label="Path to training dataset (JSONL)",
            ),
            rx.cond(
                TrainState.dataset_path_error != "",
                rx.text(
                    TrainState.dataset_path_error,
                    size="1",
                    style={"color": "var(--bp-peach)", "font_size": "11px"},
                ),
                rx.fragment(),
            ),
            direction="column",
            width="100%",
        ),
        rx.text(
            "Format auto-detected from contents: Alpaca · ShareGPT · OpenAI · raw JSONL.",
            size="1",
            style={"color": "var(--bp-muted)"},
        ),
        title="Dataset",
    )


def _recovery_banners() -> rx.Component:
    """Most-recent ``ok`` / ``warn`` event banners (FRONTEND-A-004, v1.4 Wave 2).

    The Train page surfaces the latest recovery-class event as an inline
    ``BpRecoveryBanner``. Pre-v1.4 the component existed but no page
    rendered it — the docstring at the top of this module promised
    "Recovery banners (when applicable)" but the body never wired one.

    Rendering rules:

    - ``ok``-level event present → render the green "ok" banner. Operator
      sees the good-news recovery (e.g. trainer resumed from an OOM
      bisect, GPU temp dropped below threshold, checkpoint written after
      a near-miss).
    - ``warn``-level event present → render the amber "warn" banner.
      Operator sees the heads-up condition (e.g. GPU temp approaching
      threshold, dataset row skipped, batch auto-shrunk for VRAM).
    - Both can render simultaneously — they convey different signals and
      the operator benefits from seeing both.
    - ``info``-level events are NOT surfaced as banners (would flood the
      page during a normal run). They remain in the side-rail event log
      via ``BpEventLog`` for operators who want the full timeline.
    - ``err``-level events are NOT surfaced here either — hard failures
      route through the structured error callout (``BpErrorCallout``,
      consumed by the per-page inline error chrome in v1.4 Wave 2).

    Per design digest §4e: recovery is good news, even if the original
    event wasn't — never render in red.

    FRONTEND-B-008 (Stage C polish): the outer ``rx.flex`` only renders
    when at least one banner has content. Pre-fix the column container
    rendered on every Train page mount even when both messages were
    empty (no DOM children, but a 100%-width styled flex with gap
    still emitted) — a minor layout-cost noise the Stage B audit
    surfaced. Now the entire outer flex collapses to a fragment when
    both messages are empty.
    """
    return rx.cond(
        (TrainState.latest_recovery_ok_msg != "")
        | (TrainState.latest_recovery_warn_msg != ""),
        rx.flex(
            rx.cond(
                TrainState.latest_recovery_ok_msg != "",
                BpRecoveryBanner(
                    variant="ok",
                    lead="Recovered.",
                    # body is the message; the lead is the canonical "this is
                    # a recovery" tag so screen readers get the framing first.
                    body=TrainState.latest_recovery_ok_msg,
                ),
                rx.fragment(),
            ),
            rx.cond(
                TrainState.latest_recovery_warn_msg != "",
                BpRecoveryBanner(
                    variant="warn",
                    lead="Heads-up.",
                    body=TrainState.latest_recovery_warn_msg,
                ),
                rx.fragment(),
            ),
            direction="column",
            gap="2",
            width="100%",
        ),
        rx.fragment(),
    )


def _next_steps_panel() -> rx.Component:
    """Post-run affordances — FRONTEND-10 (Wave 6b).

    Surfaces after a run reaches a terminal state (``done`` or ``error``).
    Hidden in idle / loading / active to avoid cognitive noise during a
    live run. The links route to the relevant surface — clicking "Export
    to GGUF" navigates to /export with the format pre-selected (the
    pre-select is a state hand-off; the link sets the route, the export
    page picks up TrainState's last run_id on mount in a follow-up wave).

    Per Wave 5 audit FRONTEND-F-010: after a run completes, surface
    affordances: "Export to GGUF", "Push to HF Hub", "Register with
    Ollama", "View checkpoints", "Start another run".
    """
    return rx.cond(
        TrainState.run_complete,
        rx.flex(
            rx.text(
                "Run complete · what next?",
                size="2",
                style={
                    "color": "var(--bp-text-2)",
                    "text_transform": "uppercase",
                    "letter_spacing": "0.06em",
                    "font_size": "11px",
                },
            ),
            rx.flex(
                rx.link(
                    rx.button(
                        "Export to GGUF",
                        variant="soft",
                        color_scheme="teal",
                        size="2",
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
                        aria_label="Browse this run's checkpoint files in the runs page",
                    ),
                    href="/runs",
                ),
                rx.button(
                    "Start another run",
                    variant="ghost",
                    color_scheme="teal",
                    size="2",
                    on_click=TrainState.start_training,
                    aria_label="Reset and start another training run with the same config",
                ),
                direction="row",
                gap="2",
                wrap="wrap",
            ),
            direction="column",
            gap="3",
            padding="4",
            style={
                "background": "var(--bp-surface-2)",
                "border": "1px solid var(--bp-teal)",
                "border_radius": "var(--bp-r-2)",
            },
            # FRONTEND-B-014-EXTENDED (Stage C accessibility): tag the post-run
            # affordances as a labeled region so screen readers announce the
            # context when the operator tabs into the cluster after the run
            # completes.
            role="region",
            aria_label="Run complete — next steps",
        ),
        rx.fragment(),
    )


def _advanced_group() -> rx.Component:
    return Group(
        rx.grid(
            rx.flex(
                _label("GPU temp threshold (°C)"),
                rx.input(
                    placeholder="85",
                    value=TrainState.gpu_temp_threshold.to_string(),
                    on_change=TrainState.set_gpu_temp_threshold,
                    size="2",
                    class_name="bp-num",
                    aria_label="GPU temperature threshold in Celsius (pause training above this)",
                ),
                _err_text(TrainState.gpu_temp_threshold_error),
                direction="column",
                width="100%",
            ),
            rx.flex(
                _label("W&B run name"),
                rx.input(
                    placeholder="(optional)",
                    value=TrainState.wandb_run_name,
                    on_change=TrainState.set_wandb_run_name,
                    size="2",
                    aria_label="Weights and Biases run name (optional)",
                ),
                _err_text(TrainState.wandb_run_name_error),
                direction="column",
                width="100%",
            ),
            columns="repeat(2, 1fr)",
            gap="3",
            width="100%",
        ),
        rx.flex(
            rx.checkbox(
                "Gradient checkpointing",
                checked=TrainState.gradient_checkpointing,
                on_change=TrainState.set_gradient_checkpointing,
            ),
            rx.checkbox(
                "Flash attention",
                checked=TrainState.flash_attention,
                on_change=TrainState.set_flash_attention,
            ),
            direction="row",
            gap="4",
        ),
        title="Advanced",
        collapsible=True,
        default_open=False,
    )


def train_page() -> rx.Component:
    """The Train surface."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="train"),
            rx.scroll_area(
                rx.flex(
                    rx.heading(
                        "Single run",
                        size="6",
                        style={"color": "var(--bp-text)", "font_weight": "500"},
                    ),
                    rx.text(
                        "Configure a one-shot fine-tuning run. Sensible defaults "
                        "for Qwen 2.5 7B on a 16 GB GPU; tweak any field below.",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    # FRONTEND-A-004 (v1.4 Wave 2): wire BpRecoveryBanner so
                    # the most-recent ok / warn event surfaces inline above
                    # the form. Closes the docstring promise at the top of
                    # this module ("Recovery banners (when applicable)") and
                    # gets the canonical recovery surface that design digest
                    # §4e specified onto the Train page.
                    _recovery_banners(),
                    _model_group(),
                    _training_shape_group(),
                    _lora_group(),
                    _dataset_group(),
                    _advanced_group(),
                    _next_steps_panel(),
                    rx.flex(
                        rx.button(
                            rx.cond(
                                TrainState.run_state == "loading",
                                rx.spinner(size="2"),
                                rx.fragment(),
                            ),
                            rx.cond(
                                TrainState.run_state == "loading",
                                rx.text("Starting…"),
                                rx.text("Start training"),
                            ),
                            variant="solid",
                            color_scheme="teal",
                            size="3",
                            # FRONTEND-B-003: disable while a run is in flight so
                            # double-clicks don't queue duplicate trainer jobs.
                            disabled=(TrainState.run_state == "loading")
                            | (TrainState.run_state == "active"),
                            on_click=TrainState.start_training,
                        ),
                        rx.button(
                            "Stop",
                            variant="soft",
                            color_scheme="gray",
                            size="3",
                            # Stop is meaningless when nothing is running.
                            disabled=(TrainState.run_state == "idle")
                            | (TrainState.run_state == "done")
                            | (TrainState.run_state == "error"),
                            on_click=TrainState.stop_training,
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
