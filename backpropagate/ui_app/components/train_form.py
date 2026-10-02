"""Shared training-form cards (ui-v2 P3 CLI parity).

The Single run and Multi-run pages carry the same training knobs, and every
one of them reaches a real ``backprop train`` / ``multi-run`` flag (see
``ui_jobs._training_flags``). The cards here are parameterised by the state
class (``TrainState`` or ``MultiRunState``: same field and setter names) so
the two pages stay in lockstep.

- **Start from**: a model preset (``config.MODEL_PRESETS``) or a custom id;
  the preset fills the model and its recommended LoRA rank.
- **Method**: SFT / ORPO / SimPO / KTO with each method's own knobs.
- **Mode**: QLoRA (4-bit base, the CLI default) / LoRA (16-bit base,
  ``--no-4bit``) / Full fine-tune (``--mode full``, Single run only).
- **LoRA**: Quality / Fast shape picks plus rank, alpha, dropout, targets.
- **Advanced**: GPU temperature limit, run name, gradient checkpointing.
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import TrainState, model_preset_options

from .field import FIELD_STYLE, bp_field
from .group import Group

_LOCKED = TrainState.form_disabled


def _muted(text, **style) -> rx.Component:
    return rx.text(
        text,
        size="1",
        style={"color": "var(--bp-muted)", "font_size": "13px", "line_height": "1.5", **style},
    )


def choice_card(label: str, value: str, description: str, on_pick=None) -> rx.Component:
    """One radio option as a selectable card (label + one-line description).

    The radio keeps its own short accessible name ("LoRA"); the description
    is tied to it with ``aria-describedby``. Clicking anywhere on the card
    picks it (``on_pick``); the checked card takes the teal edge and the
    focused radio's card shows the focus ring (``.bp-choice`` in ui_theme).
    """
    desc_id = f"bp-choice-{value}-desc"
    kwargs = {"on_click": on_pick} if on_pick is not None else {}
    return rx.box(
        rx.flex(
            rx.radio.item(label, value=value, aria_describedby=desc_id),
            rx.text(
                description,
                id=desc_id,
                size="1",
                style={"color": "var(--bp-muted)", "font_size": "12px", "line_height": "1.45"},
            ),
            direction="column",
            gap="var(--space-1)",
        ),
        class_name="bp-choice",
        **kwargs,
    )


def _number(value, on_change, placeholder: str, aria_label: str, *, as_int: bool = True):
    return rx.input(
        placeholder=placeholder,
        value=value.to_string(),
        on_change=on_change,
        type="number" if as_int else "text",
        size="2",
        class_name="bp-num",
        disabled=_LOCKED,
        style={**FIELD_STYLE, "width": "100%"},
        aria_label=aria_label,
    )


# ---- Start from (preset + model) --------------------------------------------------


def start_from_card(S) -> rx.Component:
    options = model_preset_options()
    return Group(
        bp_field(
            "Preset",
            rx.select.root(
                rx.select.trigger(
                    placeholder="Custom model",
                    style={**FIELD_STYLE, "width": "100%"},
                    aria_label="Model preset",
                ),
                rx.select.content(
                    *(rx.select.item(o["label"], value=o["key"]) for o in options),
                    rx.select.separator(),
                    rx.select.item("Custom model", value="custom"),
                ),
                value=S.preset,
                on_change=S.set_preset,
                disabled=_LOCKED,
            ),
        ),
        bp_field(
            "HuggingFace model id",
            rx.input(
                placeholder="org/model-name or a local folder",
                value=S.model,
                on_change=S.set_model,
                size="2",
                disabled=_LOCKED,
                style={**FIELD_STYLE, "width": "100%"},
                aria_label="HuggingFace model id",
            ),
            S.model_error,
        ),
        _muted(S.preset_note),
        title="Model",
    )


# ---- Method -------------------------------------------------------------------------

_METHODS = (
    ("SFT", "sft", "Supervised fine-tuning on examples."),
    ("ORPO", "orpo", "Preference pairs, no reference model."),
    ("SimPO", "simpo", "Preference pairs, length-normalised reward."),
    ("KTO", "kto", "Thumbs-up / thumbs-down feedback."),
)

# (field, label, aria) per method knob.
_METHOD_FIELDS = {
    "orpo": (("orpo_beta", "Beta (odds-ratio weight)", "ORPO beta"),),
    "simpo": (
        ("simpo_beta", "Beta (reward scale)", "SimPO beta"),
        ("simpo_gamma", "Gamma (target margin)", "SimPO gamma"),
    ),
    "kto": (
        ("kto_beta", "Beta", "KTO beta"),
        ("kto_desirable_weight", "Desirable weight", "KTO desirable weight"),
        ("kto_undesirable_weight", "Undesirable weight", "KTO undesirable weight"),
    ),
}


def _param_handler(S, key: str):
    return lambda value: S.set_method_param(key, value)


def _method_knobs(S, method: str) -> rx.Component:
    fields = _METHOD_FIELDS[method]
    return rx.grid(
        *(
            bp_field(
                label,
                rx.input(
                    value=getattr(S, key).to_string(),
                    on_change=_param_handler(S, key),
                    size="2",
                    class_name="bp-num",
                    disabled=_LOCKED,
                    style={**FIELD_STYLE, "width": "100%"},
                    aria_label=aria,
                ),
            )
            for key, label, aria in fields
        ),
        columns="repeat(auto-fit, minmax(140px, 1fr))",
        gap="var(--space-4)",
        width="100%",
    )


def method_card(S) -> rx.Component:
    return Group(
        rx.radio.root(
            rx.grid(
                *(choice_card(label, value, desc, S.set_method(value)) for label, value, desc in _METHODS),
                columns="repeat(auto-fit, minmax(190px, 1fr))",
                gap="var(--space-3)",
                width="100%",
            ),
            value=S.method,
            on_change=S.set_method,
            disabled=_LOCKED,
            aria_label="Training method",
        ),
        rx.match(
            S.method,
            ("orpo", _method_knobs(S, "orpo")),
            ("simpo", _method_knobs(S, "simpo")),
            ("kto", _method_knobs(S, "kto")),
            rx.fragment(),
        ),
        rx.cond(
            S.method_param_error != "",
            rx.text(S.method_param_error, size="1", style={"color": "var(--bp-peach)"}),
            rx.fragment(),
        ),
        title="Method",
    )


# ---- Mode -----------------------------------------------------------------------------


def mode_card(S, *, allow_full: bool) -> rx.Component:
    cards = [
        choice_card(
            "QLoRA", "qlora", "LoRA on a 4-bit base. Least memory; the default.",
            S.set_train_mode("qlora"),
        ),
        choice_card(
            "LoRA", "lora", "LoRA on a 16-bit base. More memory, no quantization.",
            S.set_train_mode("lora"),
        ),
    ]
    if allow_full:
        cards.append(
            choice_card(
                "Full fine-tune", "full", "Trains every weight. SFT only; small models.",
                S.set_train_mode("full"),
            )
        )
    return Group(
        rx.radio.root(
            rx.grid(
                *cards,
                columns="repeat(auto-fit, minmax(150px, 1fr))",
                gap="var(--space-3)",
                width="100%",
            ),
            value=S.train_mode,
            on_change=S.set_train_mode,
            disabled=_LOCKED,
            aria_label="Training mode",
        ),
        rx.cond(
            S.method != "sft",
            _muted("Full fine-tuning supports SFT only; preference methods train a LoRA adapter.")
            if allow_full
            else rx.fragment(),
            rx.fragment(),
        ),
        title="Mode",
    )


# ---- Dataset ----------------------------------------------------------------------------


def dataset_card(S, extra: rx.Component | None = None) -> rx.Component:
    return Group(
        bp_field(
            "Dataset path",
            rx.input(
                placeholder="path/to/dataset.jsonl",
                value=S.dataset_path,
                on_change=S.set_dataset_path,
                size="2",
                disabled=_LOCKED,
                style={**FIELD_STYLE, "width": "100%"},
                aria_label="Path to training dataset (JSONL)",
            ),
            S.dataset_path_error,
        ),
        _muted(S.method_data_hint),
        *([extra] if extra is not None else []),
        title="Dataset",
    )


# ---- Training shape -----------------------------------------------------------------------


def training_shape_card(S, *, steps_label: str = "Steps", with_steps: bool = True) -> rx.Component:
    cells = []
    if with_steps:
        cells.append(
            bp_field(
                steps_label,
                _number(S.steps, S.set_steps, "100", "Number of training steps"),
                S.steps_error,
            )
        )
    cells += [
        bp_field(
            "Batch size",
            rx.input(
                placeholder="auto",
                value=S.batch_size,
                on_change=S.set_batch_size,
                size="2",
                class_name="bp-num",
                disabled=_LOCKED,
                style={**FIELD_STYLE, "width": "100%"},
                aria_label="Batch size (number or auto)",
            ),
            S.batch_size_error,
        ),
        bp_field(
            "Learning rate",
            rx.input(
                placeholder="2e-4",
                value=S.learning_rate.to_string(),
                on_change=S.set_learning_rate,
                size="2",
                class_name="bp-num",
                disabled=_LOCKED,
                style={**FIELD_STYLE, "width": "100%"},
                aria_label="Learning rate",
            ),
            S.learning_rate_error,
        ),
    ]
    return Group(
        rx.grid(
            *cells,
            columns="repeat(auto-fit, minmax(90px, 1fr))",
            gap="var(--space-4)",
            width="100%",
        ),
        title="Training shape",
    )


# ---- LoRA -----------------------------------------------------------------------------------


def _shape_pill(S, label: str, value: str) -> rx.Component:
    selected = S.lora_shape == value
    return rx.button(
        label,
        size="2",
        variant=rx.cond(selected, "soft", "outline"),
        color_scheme=rx.cond(selected, "teal", "gray"),
        on_click=S.apply_lora_shape(value),
        disabled=_LOCKED,
        aria_pressed=rx.cond(selected, "true", "false"),
        style={"border_radius": "var(--bp-r-pill)"},
    )


def lora_card(S) -> rx.Component:
    body = rx.flex(
        rx.flex(
            _shape_pill(S, "Quality", "quality"),
            _shape_pill(S, "Fast", "fast"),
            rx.cond(
                S.lora_shape == "custom",
                rx.badge("Custom", variant="surface", color_scheme="gray", radius="full"),
                rx.fragment(),
            ),
            gap="var(--space-2)",
            align="center",
            wrap="wrap",
        ),
        _muted(
            rx.match(
                S.lora_shape,
                ("quality", "Rank 256 on every linear layer: close to full fine-tuning quality."),
                ("fast", "Rank 16 on the attention q and v projections: fastest, least memory."),
                "Your own rank, alpha and target modules.",
            )
        ),
        rx.grid(
            bp_field(
                "Rank",
                _number(S.lora_r, S.set_lora_r, "256", "LoRA rank (r)"),
                S.lora_r_error,
            ),
            bp_field(
                "Alpha",
                _number(S.lora_alpha, S.set_lora_alpha, "512", "LoRA alpha"),
                S.lora_alpha_error,
            ),
            bp_field(
                "Dropout",
                _number(
                    S.lora_dropout, S.set_lora_dropout, "0.05", "LoRA dropout (0 to 1)", as_int=False
                ),
                S.lora_dropout_error,
            ),
            columns="repeat(auto-fit, minmax(90px, 1fr))",
            gap="var(--space-4)",
            width="100%",
        ),
        bp_field(
            "Target modules",
            rx.input(
                placeholder="all-linear, or q_proj, v_proj",
                value=S.target_modules,
                on_change=S.set_target_modules,
                size="2",
                disabled=_LOCKED,
                style={**FIELD_STYLE, "width": "100%"},
                aria_label="LoRA target modules: all-linear or comma-separated layer names",
            ),
            S.target_modules_error,
        ),
        direction="column",
        gap="var(--space-4)",
        width="100%",
    )
    return Group(
        rx.cond(
            S.is_full_ft,
            _muted("Full fine-tuning trains every weight, so there is no LoRA adapter to shape."),
            body,
        ),
        title="LoRA",
    )


# ---- Advanced ---------------------------------------------------------------------------------


def advanced_card(S) -> rx.Component:
    return Group(
        rx.grid(
            bp_field(
                "GPU temperature limit (°C)",
                _number(
                    S.gpu_temp_threshold,
                    S.set_gpu_temp_threshold,
                    "90",
                    "GPU temperature limit in Celsius (stop and save above this)",
                ),
                S.gpu_temp_threshold_error,
            ),
            bp_field(
                "Run name",
                rx.input(
                    placeholder="(optional)",
                    value=S.wandb_run_name,
                    on_change=S.set_wandb_run_name,
                    size="2",
                    disabled=_LOCKED,
                    style={**FIELD_STYLE, "width": "100%"},
                    aria_label="Run name for experiment trackers (optional)",
                ),
                S.wandb_run_name_error,
            ),
            columns="repeat(auto-fit, minmax(200px, 1fr))",
            gap="var(--space-4)",
            width="100%",
        ),
        _muted(
            "If the GPU stays above the limit, the run saves a checkpoint and stops, "
            "just like pressing Stop. The run name labels the run in W&B, TensorBoard or MLflow."
        ),
        rx.checkbox(
            "Gradient checkpointing (less memory, slightly slower)",
            checked=S.gradient_checkpointing,
            on_change=S.set_gradient_checkpointing,
            disabled=_LOCKED,
        ),
        title="Advanced",
        collapsible=True,
        default_open=False,
    )


# ---- Inline VRAM estimate (Single run) -----------------------------------------------------------

_VERDICT_COLOR = {
    "fits": "var(--bp-seafoam)",
    "tight": "var(--bp-amber)",
    "wont_fit": "var(--bp-peach)",
    "unknown": "var(--bp-muted)",
}


def vram_estimate_card() -> rx.Component:
    """"Fits / Tight / Won't fit" with the estimate on a bar against the card.

    The numbers are ``backprop estimate-vram``'s (ui_jobs.vram_verdict).
    """
    S = TrainState
    color = rx.match(
        S.vram_est_verdict,
        ("fits", _VERDICT_COLOR["fits"]),
        ("tight", _VERDICT_COLOR["tight"]),
        ("wont_fit", _VERDICT_COLOR["wont_fit"]),
        _VERDICT_COLOR["unknown"],
    )
    return rx.box(
        rx.flex(
            rx.flex(
                rx.text(
                    "Estimated VRAM",
                    style={"color": "var(--bp-text-2)", "font_size": "13px"},
                ),
                rx.text(
                    S.vram_est_label,
                    class_name="bp-num",
                    style={"color": "var(--bp-text)", "font_size": "15px", "font_weight": "600"},
                ),
                direction="column",
                gap="2px",
            ),
            rx.spacer(),
            rx.box(
                S.vram_est_title,
                aria_label=S.vram_est_title,
                style={
                    "color": color,
                    "border": "1px solid currentColor",
                    "border_radius": "var(--bp-r-pill)",
                    "padding": "4px 12px",
                    "font_size": "13px",
                    "font_weight": "600",
                },
                id="bp-vram-verdict",
            ),
            align="center",
            width="100%",
            gap="var(--space-4)",
        ),
        rx.box(
            rx.box(
                style={
                    "width": S.vram_est_pct,
                    "height": "100%",
                    "background": color,
                    "border_radius": "var(--bp-r-pill)",
                    "transition": "width 0.25s ease",
                },
            ),
            style={
                "height": "8px",
                "background": "var(--bp-field-bg)",
                "border_radius": "var(--bp-r-pill)",
                "overflow": "hidden",
                "margin": "12px 0 10px",
            },
            role="presentation",
        ),
        _muted(S.vram_est_detail),
        padding="var(--space-5)",
        width="100%",
        style={
            "background": "var(--bp-surface)",
            "border": "1px solid var(--bp-border)",
            "border_radius": "var(--bp-r-lg)",
            "box_shadow": "var(--bp-shadow-card)",
        },
    )


__all__ = [
    "advanced_card",
    "choice_card",
    "dataset_card",
    "lora_card",
    "method_card",
    "mode_card",
    "start_from_card",
    "training_shape_card",
    "vram_estimate_card",
]
