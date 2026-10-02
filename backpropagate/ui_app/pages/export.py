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
from ..components.field import FIELD_STYLE as _FIELD_STYLE
from ..components.field import bp_label as _label
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


def _source_group() -> rx.Component:
    return Group(
        rx.flex(
            _label("Adapter or model path"),
            rx.input(
                placeholder="runs/run-2026-05-21/adapter",
                default_value=ExportState.source_model_path,
                on_change=ExportState.set_source_model_path,
                size="2",
                style={**_FIELD_STYLE, "width": "100%"},
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


def _choice_card(radio_label: str, value: str, description: str) -> rx.Component:
    """One radio option as a bordered choice card: label on top, muted
    description on the next line. (ui-v2 P1 redesign pass 2 — the previous
    wrap=wrap flex row rendered all descriptions concatenated on one line.)"""
    return rx.flex(
        rx.radio.item(radio_label, value=value),
        rx.text(
            description,
            size="1",
            style={"color": "var(--bp-muted)", "font_size": "11px"},
        ),
        direction="column",
        gap="var(--space-1)",
        padding="var(--space-3)",
        style={
            "background": "var(--bp-surface-2)",
            "border": "1px solid var(--bp-border)",
            "border_radius": "var(--bp-r-md)",
        },
    )


def _format_group() -> rx.Component:
    return Group(
        rx.radio.root(
            rx.grid(
                _choice_card("LoRA", "lora", "Just the adapter weights (small · portable)."),
                _choice_card("Merged", "merged", "Adapter merged into base — full HF model."),
                _choice_card("GGUF", "gguf", "Quantized GGUF — Ollama / llama.cpp ready."),
                columns="repeat(3, 1fr)",
                gap="var(--space-3)",
                width="100%",
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
                *(_choice_card(quant, quant, note) for quant, note in _GGUF_QUANTS),
                columns="repeat(3, 1fr)",
                gap="var(--space-3)",
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
            gap="var(--space-2)",
            align="center",
        ),
        rx.flex(
            _label("Ollama model name"),
            rx.input(
                placeholder="my-finetuned-model",
                default_value=ExportState.ollama_name,
                on_change=ExportState.set_ollama_name,
                size="2",
                style={**_FIELD_STYLE, "width": "100%"},
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
            gap="var(--space-2)",
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
                        style={**_FIELD_STYLE, "width": "100%"},
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
                            style={**_FIELD_STYLE, "width": "100%"},
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
                            gap="var(--space-2)",
                            style={"padding_top": "6px"},
                        ),
                        direction="column",
                        width="100%",
                    ),
                    columns="1fr 1fr",
                    gap="var(--space-3)",
                    width="100%",
                ),
                rx.flex(
                    _label("HuggingFace API token (write-once)"),
                    # UI-A-001 (Wave A1 CRITICAL): write-only input. We do
                    # NOT bind ``value=`` back to a state var — the raw token
                    # lives only in the backend-only ExportState._hub_token
                    # and never round-trips to the client. ``on_change``
                    # pushes each keystroke into the backend setter; the
                    # field renders empty on every server-driven re-render
                    # (uncontrolled), which is the intended write-once UX for
                    # a secret. ``hub_token_set`` (public bool) drives the
                    # "token captured" affordance below without echoing it.
                    rx.input(
                        placeholder="hf_…",
                        on_change=ExportState.set_hub_token,
                        size="2",
                        type="password",
                        style={**_FIELD_STYLE, "width": "100%"},
                        aria_label="HuggingFace API token (write scope) — cleared on successful push",
                    ),
                    rx.cond(
                        ExportState.hub_token_set,
                        rx.text(
                            "Token captured (held server-side only).",
                            size="1",
                            style={"color": "var(--bp-teal)", "font_size": "11px"},
                        ),
                        rx.fragment(),
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
                # FRONTEND-F-004 (v1.4 Wave 6b features): surface the two CLI
                # flags Wave 2 BRIDGE-A-004 added but the UI form was missing
                # — ``--token-file`` (mode-0600 path; mutually exclusive with
                # the inline token above) and ``--include-base`` (push merged
                # base weights, not just LoRA adapter). The CLI's mutual-
                # exclusion + safety calibration is mirrored in
                # ExportState.push_to_hub.
                rx.flex(
                    _label("Token-file path (mutually exclusive with token above)"),
                    rx.input(
                        placeholder="~/.config/backpropagate/hf-token",
                        value=ExportState.hub_token_file_path,
                        on_change=ExportState.set_hub_token_file_path,
                        size="2",
                        style={**_FIELD_STYLE, "width": "100%"},
                        aria_label=(
                            "Path to a file containing the HF token (mode-0600 "
                            "recommended). Mutually exclusive with the token "
                            "input field above."
                        ),
                    ),
                    rx.text(
                        "Safer than the inline token field — the path is "
                        "validated here; the file is read at push time so "
                        "the credential never enters the WS state. Mode "
                        "0600 (rw owner-only) is recommended; widened modes "
                        "trigger a stderr warning, not a hard error.",
                        size="1",
                        style={
                            "color": "var(--bp-muted)",
                            "font_size": "11px",
                        },
                    ),
                    rx.cond(
                        ExportState.hub_token_file_path_error != "",  # nosec B105 — empty-string sentinel, not a password
                        rx.text(
                            ExportState.hub_token_file_path_error,
                            size="1",
                            style={
                                "color": "var(--bp-peach)",
                                "font_size": "11px",
                            },
                        ),
                        rx.fragment(),
                    ),
                    direction="column",
                    width="100%",
                ),
                rx.flex(
                    rx.checkbox(
                        "Include base model weights (push merged model)",
                        checked=ExportState.hub_include_base,
                        on_change=ExportState.set_hub_include_base,
                    ),
                    rx.text(
                        "Default (off) uploads only the LoRA adapter files. "
                        "Turn on to push every file in the source directory "
                        "— useful when the source is a merged model export "
                        "and you want the base weights uploaded too. The "
                        "upload size grows by the size of the base model.",
                        size="1",
                        style={
                            "color": "var(--bp-muted)",
                            "font_size": "11px",
                        },
                    ),
                    direction="column",
                    gap="var(--space-2)",
                    align="start",
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
                    gap="var(--space-2)",
                    align="center",
                ),
                rx.cond(
                    ExportState.hub_message != "",
                    # FRONTEND-B-014-EXTENDED (Stage C accessibility): wrap
                    # the hub-status row in role=status / aria_live=polite so
                    # screen readers announce push outcomes (pushing /
                    # success / failure). aria_atomic ensures the entire
                    # message is read as one unit, not just the changed run
                    # of text.
                    rx.box(
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
                            gap="var(--space-2)",
                            padding="var(--space-3)",
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
                        role="status",
                        aria_live="polite",
                        aria_atomic="true",
                    ),
                    rx.fragment(),
                ),
                direction="column",
                gap="var(--space-3)",
                width="100%",
            ),
            rx.fragment(),
        ),
        title="HuggingFace Hub",
    )


def _cli_notice() -> rx.Component:
    """Inline "use the CLI" notice — CLIUI-B-001 (Stage C UI honesty floor).

    Surfaces ``ExportState.cli_notice`` (set on the "coming soon" Export
    click) as a neutral, NON-error callout pointing at `backprop export`.
    Renders nothing until the notice is set.
    """
    return rx.cond(
        ExportState.cli_notice != "",
        rx.box(
            rx.flex(
                rx.text(
                    ExportState.cli_notice,
                    size="1",
                    style={
                        "color": "var(--bp-text-2)",
                        "font_size": "12px",
                        "flex_grow": "1",
                    },
                ),
                direction="row",
                align="center",
                gap="var(--space-2)",
                padding="var(--space-3)",
                style={
                    "background": "var(--bp-surface-2)",
                    "border": "1px solid var(--bp-border)",
                    "border_radius": "var(--bp-r-2)",
                },
            ),
            role="status",
            aria_live="polite",
            aria_atomic="true",
            margin_top="var(--space-2)",
        ),
        rx.fragment(),
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
                    gap="var(--space-2)",
                    padding_y="var(--space-3)",
                ),
                rx.fragment(),
            ),
            direction="column",
            gap="var(--space-1)",
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
                    rx.flex(
                        rx.heading(
                            "Export",
                            size="7",
                            style={
                                "color": "var(--bp-text)",
                                "font_weight": "600",
                                "letter_spacing": "-0.02em",
                            },
                        ),
                        rx.text(
                            "Convert a trained adapter into LoRA / merged / GGUF "
                            "and optionally register with Ollama.",
                            size="2",
                            style={"color": "var(--bp-muted)"},
                        ),
                        direction="column",
                        gap="var(--space-2)",
                        width="100%",
                    ),
                    _source_group(),
                    _format_group(),
                    _quant_grid(),
                    _ollama_group(),
                    _hub_group(),
                    _output_group(),
                    # CLIUI-B-001 (Stage C UI honesty floor): the LOCAL
                    # export-to-disk path is not wired from the UI yet, so the
                    # Export button is marked "coming soon" and clicking it
                    # surfaces an inline notice pointing at `backprop export`.
                    # The HuggingFace Hub push above (_hub_group) is a SEPARATE,
                    # fully-wired handler and is unaffected.
                    # ui-v2: badge points at the shell command instead of
                    # promising a date — web export lands in P2.
                    rx.flex(
                        rx.button(
                            rx.text("Export"),
                            rx.badge(
                                "CLI only in 1.8.2",
                                color_scheme="gray",
                                variant="soft",
                                size="1",
                            ),
                            variant="solid",
                            color_scheme="teal",
                            size="3",
                            on_click=ExportState.start_export,
                            style={"border_radius": "var(--bp-r-pill)", "min_width": "220px"},
                            aria_label=(
                                "Export — web-UI export ships in a future "
                                "release; use the backprop export shell command "
                                "for now"
                            ),
                        ),
                        gap="var(--space-3)",
                        margin_top="var(--space-2)",
                        align="center",
                        justify="end",
                    ),
                    _cli_notice(),
                    direction="column",
                    gap="var(--space-6)",
                    padding="var(--space-7)",
                    max_width="1320px",
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
