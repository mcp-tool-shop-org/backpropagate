"""``BpVramBar`` — VRAM utilization progress bar."""

from __future__ import annotations

import reflex as rx


def BpVramBar(used_gb: float = 0.0, total_gb: float = 16.0) -> rx.Component:
    """A horizontal progress bar showing VRAM used / total."""
    ratio = (used_gb / total_gb) if total_gb else 0.0
    pct = max(0.0, min(1.0, ratio)) * 100
    return rx.flex(
        rx.text(
            f"VRAM {used_gb:.1f} / {total_gb:.1f} GB",
            size="1",
            style={"color": "var(--bp-muted)"},
        ),
        rx.box(
            rx.box(
                width=f"{pct:.1f}%",
                height="100%",
                background="var(--bp-teal)",
                border_radius="var(--bp-r-1)",
            ),
            width="100%",
            height="6px",
            background="var(--bp-surface-3)",
            border_radius="var(--bp-r-1)",
            overflow="hidden",
        ),
        direction="column",
        gap="1",
        width="100%",
    )
