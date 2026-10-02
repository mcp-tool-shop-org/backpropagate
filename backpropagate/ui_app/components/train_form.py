"""Shared training-form cards.

The Single run and Multi-run pages carry the same training settings, and
every one of them reaches a real ``backprop train`` / ``multi-run`` flag (see
``ui_jobs._training_flags``). The cards here are parameterised by the state
class (``TrainState`` or ``MultiRunState``: same field and setter names) so
the two pages stay in lockstep.

The form is written for someone who is curious and may be new to this: every
section and field has an "i" tip (``info_tip`` / ``help_text.TIPS``) that says
what it is, what changing it does and where to start. Nothing an expert
expects is hidden; the explanation is added, the control is not removed.

- **Model**: a preset (``config.MODEL_PRESETS``) or any model id.
- **Method**: SFT / ORPO / SimPO / KTO with each method's own settings.
- **Mode**: QLoRA (4-bit base, the CLI default) / LoRA (16-bit base,
  ``--no-4bit``) / Full fine-tune (``--mode full``, Single run only).
- **LoRA**: Quality / Balanced / Fast shapes. On Single run each shows what
  it needs for the chosen model, and the form keeps the largest one that
  fits the GPU until the user picks their own.
- **Advanced**: GPU temperature limit, run name, gradient checkpointing.
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import TrainState, model_preset_options

from .field import FIELD_STYLE, bp_field
from .group import Group
from .info_tip import info_tip

_LOCKED = TrainState.form_disabled


def _muted(text, id: str | None = None, **style) -> rx.Component:  # noqa: A002 - the HTML id
    extra = {"id": id} if id else {}
    return rx.text(
        text,
        size="1",
        style={"color": "var(--bp-muted)", "font_size": "13px", "line_height": "1.5", **style},
        **extra,
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


# ---- Model ------------------------------------------------------------------------


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
        info="model",
    )


# ---- Method -------------------------------------------------------------------------

_METHODS = (
    ("SFT", "sft", "Learn from examples of good answers. The standard choice."),
    ("ORPO", "orpo", "Learn to prefer a better answer over a worse one."),
    ("SimPO", "simpo", "Preferences too, fair to short and long answers alike."),
    ("KTO", "kto", "Learn from single answers marked good or bad."),
)

# (field, label, aria) per method setting.
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
                info="method_knobs" if index == 0 else None,
            )
            for index, (key, label, aria) in enumerate(fields)
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
        info="method",
    )


# ---- Mode -----------------------------------------------------------------------------


def mode_card(S, *, allow_full: bool) -> rx.Component:
    cards = [
        choice_card(
            "QLoRA", "qlora", "A small adapter on a compressed model. Least memory; the usual choice.",
            S.set_train_mode("qlora"),
        ),
        choice_card(
            "LoRA", "lora", "The same adapter on the uncompressed model. About three times the memory.",
            S.set_train_mode("lora"),
        ),
    ]
    if allow_full:
        cards.append(
            choice_card(
                "Full fine-tune", "full", "Changes every weight. Most memory; for small models.",
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
            _muted("Full fine-tuning works with SFT only; the preference methods train an adapter.")
            if allow_full
            else rx.fragment(),
            rx.fragment(),
        ),
        title="Mode",
        info="mode",
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
        rx.text(
            "Not sure what is in a file? ",
            rx.link("Open it on the Dataset page", href="/dataset", style={"color": "var(--bp-teal)"}),
            " to see the first examples and the detected format.",
            size="1",
            style={"color": "var(--bp-muted)", "font_size": "13px", "line_height": "1.5"},
        ),
        *([extra] if extra is not None else []),
        title="Dataset",
        info="dataset",
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
                info="steps",
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
            info="batch_size",
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
            info="learning_rate",
        ),
    ]
    return Group(
        rx.grid(
            *cells,
            columns="repeat(auto-fit, minmax(120px, 1fr))",
            gap="var(--space-4)",
            width="100%",
        ),
        title="Training",
    )


# ---- LoRA -----------------------------------------------------------------------------------

#: (name, label, what it is). The order is largest to smallest.
_SHAPES = (
    ("quality", "Quality", "Rank 256 · every layer"),
    ("balanced", "Balanced", "Rank 64 · every layer"),
    ("fast", "Fast", "Rank 16 · two layers per block"),
)


def _shape_card(S, name: str, label: str, what: str, *, recommend: bool) -> rx.Component:
    """One LoRA shape as a pressable card.

    A toggle button (``aria-pressed``), named by its label so "Quality",
    "Balanced" and "Fast" are what assistive tech and tests find. On Single
    run it also shows what the shape needs for the chosen model and marks
    the one that fits this GPU.
    """
    selected = S.lora_shape == name
    lines: list[rx.Component] = [
        rx.text(label, weight="medium", style={"font_size": "14px", "color": "var(--bp-text)"}),
        rx.text(what, style={"font_size": "12px", "color": "var(--bp-muted)", "line_height": "1.4"}),
    ]
    if recommend:
        gb = getattr(S, f"lora_gb_{name}")
        lines.append(
            rx.cond(
                gb != "",
                rx.text(
                    "about " + gb,
                    class_name="bp-num",
                    style={"font_size": "12px", "color": "var(--bp-text-2)"},
                ),
                rx.fragment(),
            )
        )
        # Last line, so the three cards keep the same first lines.
        lines.append(
            rx.cond(
                S.lora_recommended == name,
                rx.el.span("Recommended", class_name="bp-shape-badge"),
                rx.fragment(),
            )
        )
    return rx.el.button(
        rx.flex(*lines, direction="column", gap="4px", align="start"),
        type="button",
        class_name="bp-shape",
        on_click=S.apply_lora_shape(name),
        disabled=_LOCKED,
        aria_pressed=rx.cond(selected, "true", "false"),
        aria_label=label,
    )


def lora_card(S, *, recommend: bool = False) -> rx.Component:
    """The LoRA card. ``recommend`` (Single run) adds each shape's memory
    need for the chosen model, the "Recommended" mark and the reason."""
    caption = (
        _muted(S.lora_caption, id="bp-lora-caption")
        if recommend
        else _muted(
            rx.match(
                S.lora_shape,
                ("quality", "Rank 256 on every layer: the largest adapter, close to full fine-tuning."),
                ("balanced", "Rank 64 on every layer: a quarter of the size."),
                ("fast", "Rank 16 on two attention layers per block: small and quick."),
                "Your own rank, alpha and target modules.",
            )
        )
    )
    follow = (
        rx.cond(
            S.lora_can_follow_gpu,
            rx.button(
                "Use the recommended shape",
                variant="ghost",
                color_scheme="teal",
                size="1",
                on_click=S.follow_gpu_shape,
                disabled=_LOCKED,
            ),
            rx.fragment(),
        )
        if recommend
        else rx.fragment()
    )
    body = rx.flex(
        rx.grid(
            *(_shape_card(S, name, label, what, recommend=recommend) for name, label, what in _SHAPES),
            columns="repeat(auto-fit, minmax(150px, 1fr))",
            gap="var(--space-3)",
            width="100%",
            role="group",
            aria_label="LoRA shape",
        ),
        rx.flex(caption, follow, direction="column", gap="6px", align="start"),
        rx.grid(
            bp_field(
                "Rank",
                _number(S.lora_r, S.set_lora_r, "256", "LoRA rank (r)"),
                S.lora_r_error,
                info="rank",
            ),
            bp_field(
                "Alpha",
                _number(S.lora_alpha, S.set_lora_alpha, "512", "LoRA alpha"),
                S.lora_alpha_error,
                info="alpha",
            ),
            bp_field(
                "Dropout",
                _number(
                    S.lora_dropout, S.set_lora_dropout, "0.05", "LoRA dropout (0 to 1)", as_int=False
                ),
                S.lora_dropout_error,
                info="dropout",
            ),
            columns="repeat(auto-fit, minmax(120px, 1fr))",
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
            info="target_modules",
        ),
        direction="column",
        gap="var(--space-4)",
        width="100%",
    )
    return Group(
        rx.cond(
            S.is_full_ft,
            _muted("Full fine-tuning changes every weight, so there is no adapter to shape."),
            body,
        ),
        title="LoRA adapter",
        info="lora",
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
                info="gpu_temp",
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
                info="run_name",
            ),
            columns="repeat(auto-fit, minmax(200px, 1fr))",
            gap="var(--space-4)",
            width="100%",
        ),
        rx.flex(
            rx.checkbox(
                "Gradient checkpointing (less memory, slightly slower)",
                checked=S.gradient_checkpointing,
                on_change=S.set_gradient_checkpointing,
                disabled=_LOCKED,
            ),
            info_tip("gradient_checkpointing"),
            align="center",
            gap="6px",
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


def vram_estimate_card(action: rx.Component | None = None) -> rx.Component:
    """"Fits / Tight / Won't fit" with the estimate on a bar against the card.

    One compact docked row: the number, the bar, the verdict and (when given)
    the page's primary ``action`` button, with one line of detail under it.
    When the setup is Tight or Won't fit, a one-click fix sits next to the
    verdict. The numbers are ``backprop estimate-vram``'s
    (ui_jobs.vram_verdict).
    """
    S = TrainState
    color = rx.match(
        S.vram_est_verdict,
        ("fits", _VERDICT_COLOR["fits"]),
        ("tight", _VERDICT_COLOR["tight"]),
        ("wont_fit", _VERDICT_COLOR["wont_fit"]),
        _VERDICT_COLOR["unknown"],
    )
    row = [
        rx.flex(
            rx.flex(
                rx.text(
                    S.vram_est_heading,
                    style={"color": "var(--bp-text-2)", "font_size": "12px", "white_space": "nowrap"},
                ),
                info_tip("vram", side="top"),
                align="center",
                gap="4px",
            ),
            rx.text(
                S.vram_est_label,
                id="bp-vram-estimate",
                class_name="bp-num",
                style={
                    "color": "var(--bp-text)",
                    "font_size": "15px",
                    "font_weight": "600",
                    "white_space": "nowrap",
                },
            ),
            direction="column",
            gap="2px",
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
                "flex": "1 1 60px",
                "min_width": "70px",
                "background": "var(--bp-field-bg)",
                "border_radius": "var(--bp-r-pill)",
                "overflow": "hidden",
            },
            role="presentation",
        ),
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
                "white_space": "nowrap",
            },
            id="bp-vram-verdict",
        ),
        rx.cond(
            S.vram_fix_label != "",
            rx.button(
                S.vram_fix_label,
                variant="soft",
                color_scheme="teal",
                size="2",
                on_click=S.apply_vram_fix,
                disabled=S.form_disabled,
                id="bp-vram-fix",
                style={"border_radius": "var(--bp-r-pill)", "white_space": "nowrap"},
            ),
            rx.fragment(),
        ),
        rx.flex(
            rx.button(
                rx.cond(S.vram_est_measured, "Measure again", "Measure on this GPU"),
                variant="outline",
                color_scheme="gray",
                size="2",
                on_click=S.start_calibration,
                disabled=S.form_disabled,
                style={"border_radius": "var(--bp-r-pill)", "white_space": "nowrap"},
            ),
            info_tip("measure", side="top"),
            align="center",
            gap="4px",
        ),
    ]
    if action is not None:
        row.append(action)
    return rx.box(
        rx.flex(*row, align="center", gap="var(--space-4)", wrap="wrap", width="100%"),
        rx.text(
            S.vram_est_detail,
            size="1",
            style={
                "color": "var(--bp-muted)",
                "font_size": "12px",
                "line_height": "1.45",
                "margin_top": "8px",
            },
        ),
        padding="14px 20px",
        width="100%",
        style={
            "background": "var(--bp-surface)",
            "border": "1px solid var(--bp-border-2)",
            "border_radius": "var(--bp-r-lg)",
            "box_shadow": "var(--bp-shadow-pop)",
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
