"""``BpLossChart`` — state-backed loss curve.

Wave 6b (FRONTEND-6 / FRONTEND-4) sibling to ``BpSparkline``.

``BpSparkline`` accepts a Python-resolved ``list[float]`` and builds the SVG
markup server-side. That's fine for the always-mounted side-rail (which got
a literal ``data=[]`` placeholder pre-v1.3 because the loss history wasn't
plumbed yet), but it cannot react to ``TrainState.loss_history`` changes
without a re-render of the whole page tree.

``BpLossChart`` solves that by delegating to Reflex's built-in recharts
binding (``rx.recharts.line_chart``). The chart consumes a state Var
directly — when the state's ``loss_history`` list changes, only the chart
re-renders, not the full page tree.

Shape mapping: ``rx.recharts.line_chart`` expects a list of dicts with
named keys (e.g. ``[{"step": 0, "loss": 0.5}, ...]``); the helper
``loss_chart_data_from`` builds that shape from a plain ``list[float]`` Var
via ``.foreach`` + index arithmetic — server-side at render time so the
client gets a single chunk of dict-shaped data per re-render.

Caveat: recharts re-renders are cheaper than a full SVG rebuild but not
free. Long-running training runs with thousands of steps should sample
(every 10th / every 100th point) before storing into state; the chart
shows up to ~2000 points smoothly on a 5080-class machine.
"""

from __future__ import annotations

import reflex as rx


def BpLossChart(
    chart_data,
    *,
    height: int = 200,
    color: str = "var(--bp-teal)",
    label: str = "loss",
) -> rx.Component:
    """Reactive loss-curve chart.

    Args:
        chart_data: Pre-shaped recharts data — a Var (or literal) pointing at
            a ``list[dict[str, Any]]`` where each dict has at minimum keys
            ``step`` (the x-axis tick) and ``label`` (the y value). State
            classes typically expose this via an ``@rx.var`` computed property
            that maps the raw ``list[float]`` to the dict shape.
        height: Chart height in pixels. Defaults to 200.
        color: Stroke color CSS string (defaults to the brand teal token).
        label: dataKey label shown in the tooltip (must match the dict key).
    """
    return rx.recharts.line_chart(
        rx.recharts.line(
            data_key=label,
            type_="monotone",
            stroke=color,
            stroke_width=2,
            dot=False,
        ),
        rx.recharts.x_axis(
            data_key="step",
            tick_size=4,
            tick_line=False,
            stroke="var(--bp-text-2)",
        ),
        rx.recharts.y_axis(
            tick_size=4,
            tick_line=False,
            stroke="var(--bp-text-2)",
            width=44,
        ),
        rx.recharts.cartesian_grid(
            stroke="var(--bp-border)",
            stroke_dasharray="3 3",
            vertical=False,
        ),
        rx.recharts.graphing_tooltip(
            content_style={
                "background": "var(--bp-surface-2)",
                "border": "1px solid var(--bp-border)",
                "borderRadius": "var(--bp-r-2)",
                "color": "var(--bp-text)",
            },
        ),
        data=chart_data,
        width="100%",
        height=height,
    )


__all__ = ["BpLossChart"]
