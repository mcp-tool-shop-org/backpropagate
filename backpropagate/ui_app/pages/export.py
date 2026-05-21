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
                size="2",
                style={"width": "100%"},
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
            default_value=ExportState.format,
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
            default_value=ExportState.gguf_quant,
        ),
        title="GGUF quantization",
    )


def _ollama_group() -> rx.Component:
    return Group(
        rx.flex(
            rx.checkbox(
                "Register with Ollama",
                default_checked=ExportState.ollama_register,
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
                size="2",
                style={"width": "100%"},
            ),
            direction="column",
            width="100%",
        ),
        title="Ollama",
    )


def _output_group() -> rx.Component:
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
                    _output_group(),
                    rx.flex(
                        rx.button(
                            "Export",
                            variant="solid",
                            color_scheme="teal",
                            size="3",
                            on_click=ExportState.start_export,
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
