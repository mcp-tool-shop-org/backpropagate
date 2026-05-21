"""``BpEventLog`` — last-N entries from the live event stream.

Per design digest §4c, this is NOT a textbox. Each row is:

- ``t`` — timestamp, monospace, ``var(--bp-muted-2)``
- a 6px coloured dot keyed to the level
- ``msg`` — message, monospace, level-coloured

Levels and their accent colours:

- ``info`` → ``text-2``
- ``ok``   → ``seafoam``
- ``warn`` → ``amber``
- ``err``  → ``peach``
- ``tx``   → ``teal`` (training events / checkpoint saves)
- ``hf``   → ``blue`` (HF Hub download events — own channel)

Recovery events (OOM auto-recover, retry success) ARE shown — they earn the
user's trust. The "View full log" button opens a dialog with the entire buffer.
"""

from __future__ import annotations

import reflex as rx

# Level → dot colour. Matches the design digest event-log surface exactly.
_LEVEL_COLOR = {
    "info":  "var(--bp-text-2)",
    "ok":    "var(--bp-seafoam)",
    "warn":  "var(--bp-amber)",
    "err":   "var(--bp-peach)",
    "tx":    "var(--bp-teal)",
    "hf":    "var(--bp-blue)",
}


def _row_static(entry: dict) -> rx.Component:
    """Render one log row from a build-time dict."""
    level = entry.get("level", "info")
    color = _LEVEL_COLOR.get(level, "var(--bp-text-2)")
    return rx.flex(
        rx.text(
            entry.get("t", ""),
            size="1",
            style={
                "font_family": "var(--bp-mono)",
                "color": "var(--bp-muted-2)",
                "min_width": "60px",
            },
        ),
        rx.box(
            width="6px",
            height="6px",
            border_radius="9999px",
            background=color,
            margin_top="6px",
            flex_shrink="0",
        ),
        rx.text(
            entry.get("msg", ""),
            size="1",
            style={"font_family": "var(--bp-mono)", "color": color},
        ),
        gap="2",
        align="start",
        width="100%",
    )


def _row_var(entry) -> rx.Component:
    """Render one log row from a ``State.var`` (foreach iteratee).

    ``entry`` is a ``Var[dict]`` here; subscript access (``entry["msg"]``)
    becomes a reactive expression. The colour-by-level dot uses ``rx.match``
    so the runtime can swap dots without re-rendering the whole row.
    """
    return rx.flex(
        rx.text(
            entry["t"],
            size="1",
            style={
                "font_family": "var(--bp-mono)",
                "color": "var(--bp-muted-2)",
                "min_width": "60px",
            },
        ),
        rx.match(
            entry["level"],
            ("info", _dot("var(--bp-text-2)")),
            ("ok",   _dot("var(--bp-seafoam)")),
            ("warn", _dot("var(--bp-amber)")),
            ("err",  _dot("var(--bp-peach)")),
            ("tx",   _dot("var(--bp-teal)")),
            ("hf",   _dot("var(--bp-blue)")),
            _dot("var(--bp-text-2)"),
        ),
        rx.match(
            entry["level"],
            ("info", _msg(entry["msg"], "var(--bp-text-2)")),
            ("ok",   _msg(entry["msg"], "var(--bp-seafoam)")),
            ("warn", _msg(entry["msg"], "var(--bp-amber)")),
            ("err",  _msg(entry["msg"], "var(--bp-peach)")),
            ("tx",   _msg(entry["msg"], "var(--bp-teal)")),
            ("hf",   _msg(entry["msg"], "var(--bp-blue)")),
            _msg(entry["msg"], "var(--bp-text-2)"),
        ),
        gap="2",
        align="start",
        width="100%",
    )


def _dot(color: str) -> rx.Component:
    return rx.box(
        width="6px",
        height="6px",
        border_radius="9999px",
        background=color,
        margin_top="6px",
        flex_shrink="0",
    )


def _msg(text, color: str) -> rx.Component:
    return rx.text(
        text,
        size="1",
        style={"font_family": "var(--bp-mono)", "color": color},
    )


def _empty() -> rx.Component:
    return rx.text(
        "No events yet",
        size="1",
        style={"color": "var(--bp-muted-2)", "font_style": "italic"},
    )


def BpEventLog(
    events=None,
    max_n: int = 6,
    show_view_full: bool = True,
) -> rx.Component:
    """Render the last ``max_n`` events with a "View full log" affordance.

    Parameters
    ----------
    events:
        Either a literal ``list[dict]`` (build-time) or a ``State.var`` that
        resolves to one. The reactive path uses ``rx.foreach``.
    max_n:
        Cap the in-rail render at this count. Six is the design digest default.
    show_view_full:
        When ``True``, appends a "View full log" button that opens a dialog
        with the full event buffer. Set ``False`` for places (e.g. an
        embedded panel) where the dialog would compete with surrounding chrome.
    """
    # Build-time literal path — keep it simple.
    if isinstance(events, list):
        tail = events[-max_n:]
        if not tail:
            return _empty()
        rows = rx.vstack(
            *(_row_static(e) for e in tail),
            align="start",
            gap="1",
            width="100%",
        )
        return rows

    # State.var path. ``events`` is None or a Var here. Render reactively.
    if events is None:
        return _empty()

    rows = rx.cond(
        events.length() == 0,
        _empty(),
        rx.vstack(
            rx.foreach(events[-max_n:], _row_var),
            align="start",
            gap="1",
            width="100%",
        ),
    )

    if not show_view_full:
        return rows

    return rx.flex(
        rows,
        rx.dialog.root(
            rx.dialog.trigger(
                rx.button(
                    "View full log",
                    size="1",
                    variant="ghost",
                    style={"color": "var(--bp-muted)"},
                ),
            ),
            rx.dialog.content(
                rx.dialog.title("Event log"),
                rx.dialog.description(
                    "Full event buffer for this run.",
                    style={"color": "var(--bp-muted)"},
                ),
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(events, _row_var),
                        align="start",
                        gap="1",
                        width="100%",
                    ),
                    type="auto",
                    scrollbars="vertical",
                    style={"height": "60vh"},
                ),
                rx.flex(
                    rx.dialog.close(
                        rx.button("Close", variant="soft"),
                    ),
                    justify="end",
                    margin_top="3",
                ),
                style={"max_width": "640px"},
            ),
        ),
        direction="column",
        gap="2",
        width="100%",
    )
