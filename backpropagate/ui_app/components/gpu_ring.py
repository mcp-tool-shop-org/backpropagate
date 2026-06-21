"""``BpGpuRing`` — circular GPU-temperature gauge.

Color thresholds (per design digest §4 sidebar):

- ``< 70°C`` → ``var(--bp-seafoam)`` (cool, fine)
- ``70 – 85°C`` → ``var(--bp-amber)`` (warm, watch)
- ``> 85°C`` → ``var(--bp-peach)`` (hot, throttling soon)

Center text shows the integer temperature in tabular-num so it doesn't
jiggle as digits change.

Implementation note (FRONTEND-B-012 fix): the SVG body is built with native
Reflex SVG primitives (``rx.el.svg`` + children) rather than a hand-rolled
HTML string fed to ``rx.html``. Same reasoning as ``BpSparkline``: keeps the
diffing path narrow AND removes the raw-HTML surface that could otherwise
turn into an XSS vector if a future caller threaded a caption / aria-label
through an f-string.
"""

from __future__ import annotations

import math

import reflex as rx


def _temp_color(temp_c: float) -> str:
    """Map temperature to one of the three semantic accents."""
    if temp_c < 70:
        return "var(--bp-seafoam)"
    if temp_c < 85:
        return "var(--bp-amber)"
    return "var(--bp-peach)"


def BpGpuRing(temp_c: float = 0.0, max_c: int = 95, size: int = 60) -> rx.Component:
    """A small SVG ring showing the GPU's temperature as a fill arc.

    Parameters
    ----------
    temp_c:
        Current GPU temperature in °C. Build-time scalar; for live binding the
        parent re-renders when ``TrainState.gpu_temp`` updates.
    max_c:
        Right-edge of the arc. Anything past this clamps to a full circle.
    size:
        Square pixel dimension of the SVG. Default 60px per the design canvas.
    """
    r = (size - 12) / 2  # leave 6px padding on each side for the stroke
    cx = cy = size / 2
    circumference = 2 * math.pi * r
    fill_ratio = max(0.0, min(1.0, temp_c / max_c))
    dash = circumference * fill_ratio
    gap = circumference - dash
    color = _temp_color(temp_c)

    # FRONTEND-B-012: native Reflex SVG primitives instead of ``rx.html``.
    svg = rx.el.svg(
        # Background track. VIS-UI-005: ``--bp-surface-3`` reads as a slightly
        # more present outline than the very faint ``--bp-border``, so the
        # unfilled portion of the ring stays visible.
        rx.el.svg.circle(
            cx=f"{cx}",
            cy=f"{cy}",
            r=f"{r}",
            fill="none",
            stroke="var(--bp-surface-3)",
            stroke_width="4",
        ),
        # Fill arc — rotates -90deg so 0% starts at 12 o'clock.
        rx.el.svg.circle(
            cx=f"{cx}",
            cy=f"{cy}",
            r=f"{r}",
            fill="none",
            stroke=color,
            stroke_width="4",
            stroke_linecap="round",
            stroke_dasharray=f"{dash:.1f} {gap:.1f}",
            transform=f"rotate(-90 {cx} {cy})",
        ),
        view_box=f"0 0 {size} {size}",
        width=f"{size}",
        height=f"{size}",
        xmlns="http://www.w3.org/2000/svg",
        role="img",
        aria_label=f"gpu temperature {temp_c:.0f} degrees celsius",
    )

    return rx.box(
        svg,
        rx.flex(
            rx.text(
                f"{temp_c:.0f}",
                size="3",
                weight="medium",
                class_name="bp-num",
                style={"color": color, "line_height": "1"},
            ),
            rx.text(
                "°C",
                size="1",
                style={"color": "var(--bp-muted)", "line_height": "1"},
            ),
            direction="row",
            gap="0",
            align="baseline",
            justify="center",
            style={
                "position": "absolute",
                "top": "50%",
                "left": "50%",
                "transform": "translate(-50%, -50%)",
                "pointer_events": "none",
            },
        ),
        position="relative",
        width=f"{size}px",
        height=f"{size}px",
    )
