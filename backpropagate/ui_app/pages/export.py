"""Export page — ``/export`` — adapter / merged / GGUF export.

Component tree:

- Group "Source"            — adapter or merged model path
- Group "Format"            — 3-way radio: LoRA · merged · GGUF
- Group "GGUF quantization" — grid of quant choices (q4_K_M default)
- Group "Ollama"            — register checkbox + model name
- Group "Output"            — destination + preview text
- Export button
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import ExportState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.group import Group

# GGUF quant grid — q4_K_M is the recommended default (per the design canvas).
_GGUF_QUANTS = [
    ("q2_K",   "smallest · lowest quality"),
    ("q3_K_M", "small · low quality"),
    ("q4_K_M", "default · recommended"),
    ("q5_K_M", "larger · higher quality"),
    ("q6_K",   "near-original quality"),
    ("q8_0",   "lossless 8-bit"),
]


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


def _source_group() -> rx.Component:
    return Group(
        rx.flex(
            _label("Adapter or model path"),
            rx.input(
                placeholder="runs/run-2026-05-21/adapter",
                default_value=ExportState.source_model_path,
                on_change=ExportState.set_source_model_path,
                size="2",
                style={"width": "100%"},
                aria_label="Source adapter or merged-model path",
            ),
            rx.cond(
                ExportState.source_model_path_error != "",
                rx.text(
                    ExportState.source_model_path_error,
                    size="1",
                    style={"color": "var(--bp-peach)", "font_size": "11px"},
                ),
                rx.fragment(),
            ),
            direction="column",
            width="100%",
        ),
        title="Source",
    )


def _format_group() -> rx.Component:
    return Group(
        rx.radio.root(
            rx.flex(
                rx.flex(
                    rx.radio.item("LoRA", value="lora"),
                    rx.text(
                        "Just the adapter weights (small · portable).",
                        size="1",
                        style={"color": "var(--bp-muted)"},
                    ),
                    direction="column",
                    gap="1",
                ),
                rx.flex(
                    rx.radio.item("Merged", value="merged"),
                    rx.text(
                        "Adapter merged into base — full HF model.",
                        size="1",
                        style={"color": "var(--bp-muted)"},
                    ),
                    direction="column",
                    gap="1",
                ),
                rx.flex(
                    rx.radio.item("GGUF", value="gguf"),
                    rx.text(
                        "Quantized GGUF — Ollama / llama.cpp ready.",
                        size="1",
                        style={"color": "var(--bp-muted)"},
                    ),
                    direction="column",
                    gap="1",
                ),
                direction="row",
                gap="4",
                wrap="wrap",
            ),
            value=ExportState.format,
            on_change=ExportState.set_format,
        ),
        title="Format",
    )


def _quant_grid() -> rx.Component:
    """The GGUF quant grid. ``q4_K_M`` flagged as the default."""
    return Group(
        rx.radio.root(
            rx.grid(
                *(
                    rx.flex(
                        rx.radio.item(quant, value=quant),
                        rx.text(
                            note,
                            size="1",
                            style={"color": "var(--bp-muted)", "font_size": "11px"},
                        ),
                        direction="column",
                        gap="1",
                        padding="3",
                        style={
                            "background": "var(--bp-surface-2)",
                            "border": "1px solid var(--bp-border)",
                            "border_radius": "var(--bp-r-2)",
                        },
                    )
                    for quant, note in _GGUF_QUANTS
                ),
                columns="repeat(3, 1fr)",
                gap="3",
                width="100%",
            ),
            value=ExportState.gguf_quant,
            on_change=ExportState.set_gguf_quant,
        ),
        title="GGUF quantization",
    )


def _ollama_group() -> rx.Component:
    return Group(
        rx.flex(
            rx.checkbox(
                "Register with Ollama",
                checked=ExportState.ollama_register,
                on_change=ExportState.set_ollama_register,
            ),
            direction="row",
            gap="2",
            align="center",
        ),
        rx.flex(
            _label("Ollama model name"),
            rx.input(
                placeholder="my-finetuned-model",
                default_value=ExportState.ollama_name,
                on_change=ExportState.set_ollama_name,
                size="2",
                style={"width": "100%"},
                aria_label="Name to register the model under in Ollama",
            ),
            rx.cond(
                ExportState.ollama_name_error != "",
                rx.text(
                    ExportState.ollama_name_error,
                    size="1",
                    style={"color": "var(--bp-peach)", "font_size": "11px"},
                ),
                rx.fragment(),
            ),
            direction="column",
            width="100%",
        ),
        title="Ollama",
    )


def _hub_group() -> rx.Component:
    """HuggingFace Hub push form — FRONTEND-11 (Wave 6b).

    Surfaces the existing ``backpropagate.export.push_to_hub`` backend API
    to the UI. Hidden behind an "Enable push to HF Hub" checkbox so the
    fields don't add visual noise for the common operator who's only
    exporting locally.

    Token is a password field — write-once, cleared on success, never
    logged. Repo id is validated (<owner>/<repo>, alnum + . _ - / only).
    """
    return Group(
        rx.flex(
            rx.checkbox(
                "Enable push to HuggingFace Hub",
                checked=ExportState.hub_enabled,
                on_change=ExportState.set_hub_enabled,
            ),
            direction="row",
            gap="2",
            align="center",
        ),
        rx.cond(
            ExportState.hub_enabled,
            rx.flex(
                rx.flex(
                    _label("Repo id (<owner>/<repo>)"),
                    rx.input(
                        placeholder="my-org/my-finetuned-model",
                        value=ExportState.hub_repo_id,
                        on_change=ExportState.set_hub_repo_id,
                        size="2",
                        style={"width": "100%"},
                        aria_label="HuggingFace repo id in <owner>/<repo> form",
                    ),
                    rx.cond(
                        ExportState.hub_repo_id_error != "",
                        rx.text(
                            ExportState.hub_repo_id_error,
                            size="1",
                            style={"color": "var(--bp-peach)", "font_size": "11px"},
                        ),
                        rx.fragment(),
                    ),
                    direction="column",
                    width="100%",
                ),
                rx.grid(
                    rx.flex(
                        _label("Branch"),
                        rx.input(
                            placeholder="main",
                            value=ExportState.hub_branch,
                            on_change=ExportState.set_hub_branch,
                            size="2",
                            aria_label="Branch / revision to push to (defaults to main)",
                        ),
                        rx.cond(
                            ExportState.hub_branch_error != "",
                            rx.text(
                                ExportState.hub_branch_error,
                                size="1",
                                style={"color": "var(--bp-peach)", "font_size": "11px"},
                            ),
                            rx.fragment(),
                        ),
                        direction="column",
                        width="100%",
                    ),
                    rx.flex(
                        _label("Visibility"),
                        rx.flex(
                            rx.checkbox(
                                "Private repo",
                                checked=ExportState.hub_private,
                                on_change=ExportState.set_hub_private,
                            ),
                            direction="row",
                            align="center",
                            gap="2",
                            style={"padding_top": "6px"},
                        ),
                        direction="column",
                        width="100%",
                    ),
                    columns="1fr 1fr",
                    gap="3",
                    width="100%",
                ),
                rx.flex(
                    _label("HuggingFace API token (write-once)"),
                    rx.input(
                        placeholder="hf_…",
                        value=ExportState.hub_token,
                        on_change=ExportState.set_hub_token,
                        size="2",
                        type="password",
                        style={"width": "100%"},
                        aria_label="HuggingFace API token (write scope) — cleared on successful push",
                    ),
                    rx.cond(
                        ExportState.hub_token_error != "",  # nosec B105 — empty-string comparison for UI cond, not a credential
                        rx.text(
                            ExportState.hub_token_error,
                            size="1",
                            style={"color": "var(--bp-peach)", "font_size": "11px"},
                        ),
                        rx.fragment(),
                    ),
                    direction="column",
                    width="100%",
                ),
                rx.flex(
                    rx.button(
                        rx.cond(
                            ExportState.hub_status == "pushing",
                            rx.spinner(size="2"),
                            rx.fragment(),
                        ),
                        rx.cond(
                            ExportState.hub_status == "pushing",
                            rx.text("Pushing…"),
                            rx.text("Push to HF Hub"),
                        ),
                        on_click=ExportState.push_to_hub,
                        variant="solid",
                        color_scheme="teal",
                        size="2",
                        disabled=(ExportState.hub_status == "pushing"),
                        aria_label="Push the source adapter / model to the HuggingFace Hub repo",
                    ),
                    direction="row",
                    gap="2",
                    align="center",
                ),
                rx.cond(
                    ExportState.hub_message != "",
                    rx.flex(
                        rx.text(
                            ExportState.hub_message,
                            size="1",
                            style={
                                "color": rx.cond(
                                    ExportState.hub_status == "error",
                                    "var(--bp-peach)",
                                    "var(--bp-teal)",
                                ),
                                "font_family": "var(--bp-mono)",
                                "font_size": "11px",
                                "flex_grow": "1",
                            },
                        ),
                        rx.button(
                            "Dismiss",
                            on_click=ExportState.clear_hub_status,
                            variant="ghost",
                            size="1",
                        ),
                        direction="row",
                        align="center",
                        gap="2",
                        padding="3",
                        style={
                            "background": "var(--bp-surface-2)",
                            "border": rx.cond(
                                ExportState.hub_status == "error",
                                "1px solid var(--bp-peach)",
                                "1px solid var(--bp-teal)",
                            ),
                            "border_radius": "var(--bp-r-2)",
                        },
                    ),
                    rx.fragment(),
                ),
                direction="column",
                gap="3",
                width="100%",
            ),
            rx.fragment(),
        ),
        title="HuggingFace Hub",
    )


def _output_group() -> rx.Component:
    """Output path + (FRONTEND-9 Wave 6b) empty-state guidance.

    When no source path is set we surface a concrete next-action hint with
    the same shape as runs.py's empty-state — Norman 1988 affordance
    framing rather than just showing a placeholder string.
    """
    return Group(
        rx.flex(
            _label("Output path"),
            rx.text(
                rx.cond(
                    ExportState.output_path != "",
                    ExportState.output_path,
                    "(will be written to ./exports/<format>/<timestamp>)",
                ),
                size="2",
                style={"font_family": "var(--bp-mono)", "color": "var(--bp-text-2)"},
            ),
            rx.cond(
                ExportState.source_model_path == "",
                rx.flex(
                    rx.text(
                        "No source loaded yet.",
                        size="2",
                        style={"color": "var(--bp-text-2)"},
                    ),
                    rx.text(
                        "Paste an adapter path (e.g. ~/.backpropagate/ui-outputs/"
                        "runs/run-abc12345/adapter) into the Source field above, "
                        "pick a format, then Export. Open the Runs tab to find "
                        "a recent adapter path. From the shell: "
                        "`backprop export <adapter> --format gguf`.",
                        size="1",
                        style={"color": "var(--bp-muted)"},
                    ),
                    direction="column",
                    gap="2",
                    padding_y="3",
                ),
                rx.fragment(),
            ),
            direction="column",
            gap="1",
            width="100%",
        ),
        title="Output",
    )


def export_page() -> rx.Component:
    """The Export surface."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="export"),
            rx.scroll_area(
                rx.flex(
                    rx.heading(
                        "Export",
                        size="6",
                        style={"color": "var(--bp-text)", "font_weight": "500"},
                    ),
                    rx.text(
                        "Convert a trained adapter into LoRA / merged / GGUF "
                        "and optionally register with Ollama.",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    _source_group(),
                    _format_group(),
                    _quant_grid(),
                    _ollama_group(),
                    _hub_group(),
                    _output_group(),
                    rx.flex(
                        rx.button(
                            rx.cond(
                                ExportState.export_state == "loading",
                                rx.spinner(size="2"),
                                rx.fragment(),
                            ),
                            rx.cond(
                                ExportState.export_state == "loading",
                                rx.text("Exporting…"),
                                rx.text("Export"),
                            ),
                            variant="solid",
                            color_scheme="teal",
                            size="3",
                            # FRONTEND-B-003: disable while an export is in flight.
                            disabled=(ExportState.export_state == "loading")
                            | (ExportState.export_state == "active"),
                            on_click=ExportState.start_export,
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
