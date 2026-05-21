"""``BpVramBar`` — VRAM utilization progress bar.

Color thresholds:

- ``< 80%`` → ``var(--bp-blue)`` (normal)
- ``80 – 95%`` → ``var(--bp-amber)`` (high, watch)
- ``> 95%`` → ``var(--bp-peach)`` (about to OOM)

The label below the bar uses tabular numerals so the GB digits don't jitter
as memory ticks up and down.
"""

from __future__ import annotations

import reflex as rx


def _vram_color(ratio: float) -> str:
    """Map fill ratio (0–1) to a semantic accent."""
    if ratio < 0.80:
        return "var(--bp-blue)"
    if ratio < 0.95:
        return "var(--bp-amber)"
    return "var(--bp-peach)"


def BpVramBar(used_gb: float = 0.0, total_gb: float = 16.0) -> rx.Component:
    """A horizontal progress bar showing VRAM used / total."""
    ratio = (used_gb / total_gb) if total_gb else 0.0
    ratio = max(0.0, min(1.0, ratio))
    pct = ratio * 100
    color = _vram_color(ratio)

    return rx.flex(
        # Bar
        rx.box(
            rx.box(
                width=f"{pct:.1f}%",
                height="100%",
                background=color,
                border_radius="var(--bp-r-1)",
                style={"transition": "width 0.4s ease-out, background 0.2s linear"},
            ),
            width="100%",
            height="6px",
            background="var(--bp-surface-3)",
            border_radius="var(--bp-r-1)",
            overflow="hidden",
            role="progressbar",
            aria_valuemin="0",
            aria_valuemax="100",
            aria_valuenow=f"{pct:.0f}",
            aria_label=f"VRAM {used_gb:.1f} of {total_gb:.1f} gigabytes used",
        ),
        # Label row: "VRAM" eyebrow + tabular GB
        rx.flex(
            rx.text(
                "VRAM",
                size="1",
                style={
                    "color": "var(--bp-text-2)",
                    "text_transform": "uppercase",
                    "letter_spacing": "0.06em",
                    "font_size": "10px",
                },
            ),
            rx.spacer(),
            rx.text(
                f"{used_gb:.1f} / {total_gb:.1f} GB",
                size="1",
                class_name="bp-num",
                style={"color": "var(--bp-muted)", "font_size": "11px"},
            ),
            direction="row",
            align="baseline",
            width="100%",
        ),
        direction="column",
        gap="1",
        width="100%",
    )
