"""``BpGpuRing`` — circular GPU-temperature gauge.

Phase 1: static stroke-dasharray ring. Phase 2 layers the heartbeat
synchronization + colour-by-temp from the design digest.
"""

from __future__ import annotations

import reflex as rx


def BpGpuRing(temp_c: float = 0.0, max_c: int = 95) -> rx.Component:
    """A small SVG ring showing the GPU's temperature as a fill arc.

    Phase 1 caveat: ``temp_c`` is resolved at build time. For state.var-backed
    temperature, Phase 2 rebuilds this via ``rx.html`` indirection.
    """
    r = 18
    circumference = 2 * 3.14159265 * r
    fill_ratio = min(1.0, max(0.0, temp_c / max_c))
    dash = circumference * fill_ratio
    gap = circumference - dash
    svg = (
        '<svg viewBox="0 0 48 48" width="48" height="48" '
        'xmlns="http://www.w3.org/2000/svg" aria-label="gpu temperature ring">'
        f'<circle cx="24" cy="24" r="{r}" fill="none" '
        f'stroke="var(--bp-border)" stroke-width="3" />'
        f'<circle cx="24" cy="24" r="{r}" fill="none" '
        f'stroke="var(--bp-teal)" stroke-width="3" '
        f'stroke-dasharray="{dash:.1f} {gap:.1f}" '
        f'stroke-linecap="round" transform="rotate(-90 24 24)" />'
        '</svg>'
    )
    return rx.flex(
        rx.html(svg),
        rx.text(f"{temp_c:.0f}°C", size="1", style={"color": "var(--bp-muted)"}),
        direction="column",
        align="center",
        gap="1",
    )
