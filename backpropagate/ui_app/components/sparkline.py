"""``BpSparkline`` — inline SVG mini-chart for a numeric series.

Phase 1: static SVG polyline. The end-dot halo + 1.6s counter dim from the
design digest interaction pattern lands in Phase 2.
"""

from __future__ import annotations

import reflex as rx


def _build_polyline_points(data: list[float], w: int, h: int) -> str:
    """Compute the SVG ``points`` attribute for a list of values."""
    if not data:
        return ""
    if len(data) == 1:
        return f"0,{h / 2:.1f} {w},{h / 2:.1f}"
    lo, hi = min(data), max(data)
    span = (hi - lo) or 1.0
    step = w / (len(data) - 1)
    return " ".join(
        f"{i * step:.1f},{h - ((v - lo) / span) * h:.1f}"
        for i, v in enumerate(data)
    )


def BpSparkline(
    data: list[float] | None = None,
    w: int = 220,
    h: int = 48,
) -> rx.Component:
    """Return an inline-SVG sparkline.

    Phase 1 caveat: ``data`` is resolved at build time. For a state.var-backed
    series the rendering will fall back to a placeholder line until Phase 2
    swaps in a reactive ``rx.html`` builder via ``rx.foreach``.
    """
    series = data or []
    points = _build_polyline_points(series, w, h)
    svg = (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'xmlns="http://www.w3.org/2000/svg" aria-label="loss sparkline">'
        f'<polyline fill="none" stroke="var(--bp-teal)" stroke-width="1.5" '
        f'points="{points}" />'
        f'</svg>'
    )
    return rx.html(svg)
