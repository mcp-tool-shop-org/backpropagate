"""``BpSparkline`` — inline SVG mini-chart for a numeric series.

Per design digest §4b:

- 1.5px polyline in ``var(--bp-teal)``
- Subtle 8% area fill below the line — presence at small sizes
- End dot: 2px solid + 4.5px halo — catches the eye without screaming
- Optional ``caption`` eyebrow ("Loss · last 80 steps") + right-aligned ``meta``
  ("min 0.398")
- ``viewBox`` + ``preserveAspectRatio`` for crispness at any size

Implementation note (FRONTEND-B-012 fix): the SVG body is built with native
Reflex SVG primitives (``rx.el.svg`` + children) rather than a hand-rolled
HTML string fed to ``rx.html``. That keeps every attribute on the Reflex
diffing path (so a 1px movement of the end-dot ships ~60 bytes instead of
the full 2KB body) AND removes the raw-HTML surface that an f-string
interpolation could otherwise turn into an XSS vector if someone later
embeds an operator-controlled ``caption`` into an aria-label.
"""

from __future__ import annotations

import reflex as rx


def _build_polyline_points(
    data: list[float], w: int, h: int, pad: float = 2.0
) -> tuple[str, str, tuple[float, float] | None]:
    """Compute the polyline + closed-area paths and the end-dot coords.

    Returns
    -------
    (line_points, area_path, end_xy)
        ``line_points``: space-separated "x,y" pairs for ``<polyline points>``.
        ``area_path``: SVG ``<path d>`` that traces the same shape and closes
        to the baseline (for the 8% area fill).
        ``end_xy``: ``(x, y)`` of the last point, or ``None`` if no data.
    """
    if not data:
        return "", "", None
    if len(data) == 1:
        y = h - pad
        x = w / 2
        line = f"{pad},{y:.1f} {w - pad:.1f},{y:.1f}"
        area = (
            f"M{pad},{y:.1f} L{w - pad:.1f},{y:.1f} "
            f"L{w - pad:.1f},{h - pad:.1f} L{pad},{h - pad:.1f} Z"
        )
        return line, area, (x, y)

    lo, hi = min(data), max(data)
    span = (hi - lo) or 1.0
    plot_w = w - 2 * pad
    plot_h = h - 2 * pad
    step = plot_w / (len(data) - 1)

    pts: list[tuple[float, float]] = [
        (pad + i * step, pad + plot_h - ((v - lo) / span) * plot_h)
        for i, v in enumerate(data)
    ]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    # Area path: trace the line, drop to baseline at the right edge, walk back
    # to the left baseline, close. This is the 8% fill region.
    baseline = h - pad
    area_segments = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
    area_segments.extend(f"L{x:.1f},{y:.1f}" for x, y in pts[1:])
    area_segments.append(f"L{pts[-1][0]:.1f},{baseline:.1f}")
    area_segments.append(f"L{pts[0][0]:.1f},{baseline:.1f}")
    area_segments.append("Z")
    area = " ".join(area_segments)
    return line, area, pts[-1]


def BpSparkline(
    data: list[float] | None = None,
    w: int = 220,
    h: int = 48,
    caption: str = "",
    meta: str = "",
) -> rx.Component:
    """Return a sparkline card with an inline SVG chart.

    Phase 2 caveat: ``data`` is still resolved at build time. State.var-backed
    rendering would require a reactive SVG builder; the cheaper path for now is
    to recompute the markup on the server when ``data`` changes. That mirrors
    how the side rail re-renders when ``TrainState.loss_history`` updates.
    """
    series = data or []
    line_points, area_path, end_xy = _build_polyline_points(series, w, h)

    # FRONTEND-B-012: native Reflex SVG primitives instead of ``rx.html``.
    # ``rx.el.svg`` accepts arbitrary SVG children; each attribute on a child
    # is a normal Reflex prop so the diffing path is the same as any other
    # component. The view_box / preserveAspectRatio props keep the chart
    # crisp at any container width.
    svg_children: list[rx.Component] = []
    if area_path:
        svg_children.append(
            rx.el.svg.path(
                d=area_path,
                fill="var(--bp-teal)",
                fill_opacity="0.08",
                stroke="none",
            )
        )
    if line_points:
        svg_children.append(
            rx.el.svg.polyline(
                fill="none",
                stroke="var(--bp-teal)",
                stroke_width="1.5",
                stroke_linejoin="round",
                stroke_linecap="round",
                points=line_points,
            )
        )
    if end_xy is not None:
        ex, ey = end_xy
        # Halo first (drawn under), then solid dot on top.
        svg_children.append(
            rx.el.svg.circle(
                cx=f"{ex:.1f}",
                cy=f"{ey:.1f}",
                r="4.5",
                fill="var(--bp-teal)",
                fill_opacity="0.25",
            )
        )
        svg_children.append(
            rx.el.svg.circle(
                cx=f"{ex:.1f}",
                cy=f"{ey:.1f}",
                r="2",
                fill="var(--bp-teal)",
                stroke="none",
            )
        )

    svg = rx.el.svg(
        *svg_children,
        view_box=f"0 0 {w} {h}",
        width="100%",
        height=f"{h}",
        preserveAspectRatio="none",
        xmlns="http://www.w3.org/2000/svg",
        role="img",
        # Aria-label uses len() which is an int — safe to interpolate. We
        # deliberately do NOT thread ``caption`` into the aria-label here; if a
        # future caller wants that, route it through the React-side text node
        # rather than concatenating into this attribute.
        aria_label=f"loss sparkline · last {len(series)} steps",
    )

    header_row = rx.flex(
        rx.text(
            caption,
            size="1",
            style={
                "color": "var(--bp-text-2)",
                "text_transform": "uppercase",
                "letter_spacing": "0.06em",
                "font_size": "10px",
            },
        ) if caption else rx.fragment(),
        rx.spacer(),
        rx.text(
            meta,
            size="1",
            class_name="bp-num",
            style={"color": "var(--bp-muted)", "font_size": "11px"},
        ) if meta else rx.fragment(),
        direction="row",
        align="baseline",
        width="100%",
    ) if (caption or meta) else rx.fragment()

    return rx.flex(
        header_row,
        svg,
        direction="column",
        gap="1",
        width="100%",
    )
