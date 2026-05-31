"""Models page — ``/models`` — local Hugging Face cache inventory.

V1_3_BRIEF / Wave 6b FRONTEND-7. Shows:

- Total cache size on disk
- Per-model dir + size + last-modified
- Per-model cleanup affordance ("Delete model")

Reads ``~/.cache/huggingface/hub/`` directly via filesystem APIs (no
``huggingface_hub`` dep — that lives in optional extras and we want this
page to render even when only the ``[ui]`` extra is installed).

Layout mirrors the runs page: hero heading, refresh action, error callout,
empty-state, table.

Cleanup affordance opens a confirm dialog (Reflex's ``rx.alert_dialog``)
before the rm -rf. The state's ``delete_model`` handler validates the
target path lives under the cache root before deleting.
"""

from __future__ import annotations

import reflex as rx

from backpropagate.ui_state import ModelsState

from ..chrome import BpFooter, BpHeader, BpLeftNav, BpSideRail
from ..components.error_callout import BpErrorCallout


def _eyebrow_style() -> dict:
    return {
        "color": "var(--bp-text-2)",
        "font_size": "10px",
        "text_transform": "uppercase",
        "letter_spacing": "0.06em",
        "font_weight": "500",
    }


def _filter_bar() -> rx.Component:
    return rx.flex(
        rx.flex(
            rx.text(
                "Cache directory",
                size="1",
                style={
                    "color": "var(--bp-text-2)",
                    "font_size": "11px",
                    "margin_bottom": "4px",
                },
            ),
            rx.text(
                # UI-A-002: render the redacted cache dir (home prefix
                # stripped) so the operator's username never appears in the UI
                # / screenshots and the full path never ships in the WS bundle.
                ModelsState.cache_dir_display,
                size="2",
                class_name="bp-num",
                style={
                    "font_family": "var(--bp-mono)",
                    "color": "var(--bp-text)",
                    "font_size": "12px",
                },
            ),
            direction="column",
        ),
        rx.spacer(),
        rx.flex(
            rx.text(
                "Total",
                size="1",
                style={
                    "color": "var(--bp-text-2)",
                    "font_size": "11px",
                    "margin_bottom": "4px",
                },
            ),
            rx.text(
                ModelsState.total_size_mb + " MB",
                size="3",
                class_name="bp-num",
                style={
                    "font_family": "var(--bp-mono)",
                    "color": "var(--bp-teal)",
                },
            ),
            direction="column",
            align="end",
        ),
        rx.button(
            "Refresh",
            on_click=ModelsState.load_models,
            variant="soft",
            color_scheme="teal",
            size="2",
            disabled=ModelsState.loading,
            aria_label="Refresh the HuggingFace model cache listing",
        ),
        direction="row",
        align="end",
        gap="3",
        width="100%",
    )


def _table_header() -> rx.Component:
    return rx.grid(
        rx.text("Model", style=_eyebrow_style()),
        rx.text("Size", style=_eyebrow_style()),
        rx.text("Last modified", style=_eyebrow_style()),
        rx.text("", style=_eyebrow_style()),  # action column
        columns="1.8fr 100px 200px 100px",
        gap="3",
        width="100%",
        padding_x="3",
        padding_y="2",
        style={"border_bottom": "1px solid var(--bp-border)"},
    )


def _model_row(row) -> rx.Component:
    """One table row with a delete-model affordance.

    The delete button is wrapped in an alert-dialog so a stray click can't
    nuke a 14 GB Llama checkout. The dialog's confirm action calls
    ``ModelsState.delete_model(row["dir_name"])`` — the state validates the
    path lives under the cache root before deleting.

    CLIUI-B-005 (Stage C): the confirm button is disabled while THIS row's
    delete is in flight (``ModelsState.deleting_dir == row["dir_name"]``) so a
    double-click can't re-enter ``delete_model`` and surface a spurious
    "not found" once the first rmtree completes. The state handler carries the
    same guard, so correctness doesn't depend on the UI debounce alone.
    """
    return rx.grid(
        rx.text(
            row["name"],
            size="2",
            style={
                "color": "var(--bp-text)",
                "font_family": "var(--bp-mono)",
                "font_size": "12px",
                "word_break": "break-all",
            },
        ),
        rx.text(
            # f-string, not ``+``: ``row["size_mb"]`` is an untyped foreach-item
            # Var; ``Var + str`` raises TypeError at compile.
            f"{row['size_mb']} MB",
            size="2",
            class_name="bp-num",
            style={
                "font_family": "var(--bp-mono)",
                "color": "var(--bp-text-2)",
                "font_size": "11px",
            },
        ),
        rx.text(
            row["last_modified"],
            size="1",
            style={"color": "var(--bp-muted)", "font_size": "11px"},
        ),
        rx.alert_dialog.root(
            rx.alert_dialog.trigger(
                rx.button(
                    "Delete",
                    variant="ghost",
                    color_scheme="red",
                    size="1",
                    aria_label="Delete this cached model from disk",
                ),
            ),
            rx.alert_dialog.content(
                rx.alert_dialog.title("Delete cached model?"),
                rx.alert_dialog.description(
                    "This permanently removes the model's snapshot directory "
                    "from disk. It will be re-downloaded on the next "
                    "`AutoModel.from_pretrained(...)` call.",
                ),
                rx.flex(
                    rx.alert_dialog.cancel(
                        rx.button("Cancel", variant="soft", color_scheme="gray", size="2"),
                    ),
                    rx.alert_dialog.action(
                        rx.button(
                            "Delete",
                            variant="solid",
                            color_scheme="red",
                            size="2",
                            # CLIUI-B-005: disable while THIS row's delete is in
                            # flight so a double-click can't re-enter the handler.
                            disabled=ModelsState.deleting_dir == row["dir_name"],
                            on_click=lambda: ModelsState.delete_model(row["dir_name"]),
                        ),
                    ),
                    gap="3",
                    justify="end",
                    margin_top="3",
                ),
            ),
        ),
        columns="1.8fr 100px 200px 100px",
        gap="3",
        width="100%",
        padding_x="3",
        padding_y="2",
        align="center",
        style={
            "border_bottom": "1px solid var(--bp-border)",
        },
    )


def _empty_state() -> rx.Component:
    return rx.flex(
        rx.text(
            "No cached models found.",
            size="2",
            style={"color": "var(--bp-text-2)"},
        ),
        rx.text(
            "Models download on first use. Try `backprop train <model> "
            "<dataset.jsonl>` from the shell, or open the Single run tab "
            "and start a training — the first run pulls the model into the "
            "cache and it will appear here.",
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

    FRONTEND-A-004 (v1.4 Wave 2): pre-fix this rolled its own ``rx.flex``
    chrome (mirrored from runs.py). Now uses the canonical
    ``BpErrorCallout`` so the error surface shares ARIA semantics and
    design-digest-conformant styling with runs / run-detail.
    """
    return BpErrorCallout(
        code="UI · MODELS",
        title="Could not load model cache",
        message=ModelsState.error,
    )


def models_page() -> rx.Component:
    """The ``/models`` route."""
    return rx.flex(
        BpHeader(),
        rx.flex(
            BpLeftNav(active="models"),
            rx.scroll_area(
                rx.flex(
                    rx.heading(
                        "Local models",
                        size="6",
                        style={"color": "var(--bp-text)", "font_weight": "500"},
                    ),
                    rx.text(
                        "Hugging Face cache inventory. Each entry is a "
                        "snapshot directory under "
                        "~/.cache/huggingface/hub/; delete to free disk.",
                        size="2",
                        style={"color": "var(--bp-muted)"},
                    ),
                    _filter_bar(),
                    rx.cond(
                        ModelsState.error != "",
                        _error_callout(),
                        rx.fragment(),
                    ),
                    rx.cond(
                        ModelsState.loading,
                        # FRONTEND-B-014 (Stage C accessibility): wrap the
                        # spinner in role=status / aria_live=polite so screen
                        # readers announce the loading state. Mirrors the
                        # runs.py and run_detail.py fixes.
                        rx.box(
                            rx.flex(
                                rx.spinner(size="2"),
                                rx.text(
                                    "Loading model cache…",
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
                    rx.cond(
                        ModelsState.error != "",
                        rx.fragment(),
                        rx.cond(
                            ModelsState.models.length() == 0,
                            rx.cond(
                                ~ModelsState.loading,
                                _empty_state(),
                                rx.fragment(),
                            ),
                            rx.flex(
                                _table_header(),
                                rx.foreach(ModelsState.models, _model_row),
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
                        ModelsState.last_loaded_at != "",
                        rx.text(
                            "Loaded at " + ModelsState.last_loaded_at,
                            size="1",
                            style={
                                "color": "var(--bp-muted-2)",
                                "font_size": "10px",
                            },
                        ),
                        rx.fragment(),
                    ),
                    direction="column",
                    gap="4",
                    padding="6",
                    max_width="980px",
                    width="100%",
                    on_mount=ModelsState.load_models,
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
