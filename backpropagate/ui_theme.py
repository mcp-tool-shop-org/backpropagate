"""
Backpropagate — Reflex UI theme tokens
=======================================

The Ocean Mist palette (refined v1.1.0) plus the Radix theme dict used by
``backpropagate/ui_app/app.py``. Sourced verbatim from the Stage D Claude Design
digest at
``E:/AI/dogfood-labs/swarms/swarm-1779335775-02be/stage-d/claude-design-out/prompt-1-reflex-ui/design-digest.md``.

The split:

- ``RADIX_THEME`` — passed to ``rx.theme(**RADIX_THEME)`` at app construction.
  Drives Radix's component primitives (accent / gray palettes, radius, panel
  background).
- ``THEME_TOKENS`` — CSS custom properties for dark mode (default). These
  override Radix's stock surfaces so the UI breathes deeper than slate.
- ``LIGHT_TOKENS`` — light-mode parity tokens. Activated when Reflex's
  next-themes provider writes ``class="light"`` onto the ``<html>`` root
  (Reflex configures next-themes with ``attribute: "class"`` — see
  ``reflex_base/compiler/templates.py``). We also keep the legacy
  ``[data-theme="light"]`` selector as a fallback in case future Reflex
  versions switch back to a data-attribute strategy.
- ``STYLESHEETS`` — external CSS hrefs (Geist + Geist Mono from Google Fonts).
- ``TOKENS_CSS`` — a pre-assembled stylesheet string that lays the THEME_TOKENS
  under ``:root`` and LIGHT_TOKENS under ``.light, .light-theme, [data-theme="light"]``.
  Inject via ``rx.html(f"<style>{TOKENS_CSS}</style>")`` at the app root.

FRONTEND-F-001 (Wave 5.5): the LIGHT_TOKENS selector was previously
``[data-theme="light"]`` only, which never fired because nothing in the
codebase set that attribute on the document root. Reflex's color-mode
provider writes ``class="dark"`` / ``class="light"`` on ``<html>`` (via
next-themes ``attribute="class"``), so we now key off both the Radix
``.dark-theme`` / ``.light-theme`` shape AND the bare ``.dark`` / ``.light``
classes plus the legacy data-attribute. The toggle in ``BpHeader`` now
binds to ``rx.color_mode`` + ``rx.toggle_color_mode`` instead of the
server-side ``AppState.toggle_theme`` so the DOM actually mutates AND
``prefers-color-scheme`` is honored on first load (next-themes defaults to
``system``).

This module is pure data — no Reflex import — so it can be read from anywhere
(tests, headless contexts, etc.) without pulling in the UI dependency.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Radix theme
# ---------------------------------------------------------------------------

# Drop into ``rx.theme(**RADIX_THEME)``. The named accent + gray palettes wire
# Radix's component primitives; the THEME_TOKENS hex set below overrides
# surfaces so the UI breathes deeper than Radix's stock slate.
#
# FRONTEND-F-001 (Wave 5.5): ``appearance`` is INTENTIONALLY OMITTED here.
# ``ui_app/app.py`` passes ``appearance=rx.color_mode`` at the call site so
# the Radix theme re-tints whenever Reflex's next-themes provider flips
# (operator click on the header toggle OR ``prefers-color-scheme`` change on
# first load). Hard-coding ``"dark"`` here would override that binding and
# strand the toggle button — the v1.2 bug FRONTEND-F-001 was caught for.
RADIX_THEME: dict[str, object] = {
    "accent_color": "teal",        # Ocean Mist primary
    "gray_color": "slate",         # cool neutrals; matches bg #0F1316
    "radius": "medium",            # corresponds to our --bp-r-2 / r-3 ladder
    "scaling": "100%",
    "panel_background": "solid",
    "has_background": True,
}


# ---------------------------------------------------------------------------
# 2. THEME_TOKENS (dark mode, default)
# ---------------------------------------------------------------------------

THEME_TOKENS: dict[str, str] = {
    # surfaces
    "--bp-bg":        "#0F1316",   # refined: deeper than original #141618 for AA contrast on 14px body
    "--bp-surface":   "#1A1F25",
    "--bp-surface-2": "#232830",
    "--bp-surface-3": "#2D3540",
    "--bp-border":    "#2F3744",
    "--bp-border-2":  "#3A4554",
    # text
    "--bp-text":      "#ECF1F5",
    "--bp-text-2":    "#C7D1D9",
    "--bp-muted":     "#8DA0AD",   # refined: lifted from #78909C for AA at 14px
    "--bp-muted-2":   "#6B7C88",
    # accents
    "--bp-teal":      "#7EC8C8",   # primary
    "--bp-blue":      "#A8C5E2",   # secondary (HF download events)
    "--bp-seafoam":   "#98D4BB",   # success / OK
    "--bp-amber":     "#E8C28B",   # warn (recovered, not erroring)
    "--bp-peach":     "#E8A88B",   # error code identifiers (BackpropError · E_*)
    # type
    "--bp-sans": '"Geist", ui-sans-serif, system-ui, -apple-system, sans-serif',
    "--bp-mono": '"Geist Mono", ui-monospace, "SF Mono", Menlo, monospace',
    # radii
    "--bp-r-1": "4px",
    "--bp-r-2": "6px",
    "--bp-r-3": "8px",
    "--bp-r-4": "12px",
    # focus ring — WCAG 2.4.7, do not remove
    "--bp-focus": "0 0 0 2px var(--bp-bg), 0 0 0 4px var(--bp-teal)",
}


# ---------------------------------------------------------------------------
# 2a. LIGHT_TOKENS (light mode parity)
# ---------------------------------------------------------------------------

LIGHT_TOKENS: dict[str, str] = {
    "--bp-bg":        "#F4F6F8",
    "--bp-surface":   "#FFFFFF",
    "--bp-surface-2": "#F0F3F6",
    "--bp-surface-3": "#E4EAEF",
    "--bp-border":    "#D8DFE5",
    "--bp-border-2":  "#C5CFD7",
    "--bp-text":      "#131820",
    "--bp-text-2":    "#2E3A47",
    "--bp-muted":     "#5A6B78",
    "--bp-muted-2":   "#8A99A6",
    "--bp-teal":      "#2B8A8A",
    "--bp-blue":      "#4F7CAE",
    "--bp-seafoam":   "#3FA37A",
    "--bp-amber":     "#B07A2C",
    "--bp-peach":     "#B85A38",
}


# ---------------------------------------------------------------------------
# 2b. External stylesheets (Geist + Geist Mono from Google Fonts)
# ---------------------------------------------------------------------------

STYLESHEETS: list[str] = [
    "https://fonts.googleapis.com/css2?"
    "family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap",
]


# ---------------------------------------------------------------------------
# 2c. TOKENS_CSS — pre-assembled stylesheet for inline injection
# ---------------------------------------------------------------------------


def _emit_block(selector: str, tokens: dict[str, str]) -> str:
    body = "\n".join(f"  {k}: {v};" for k, v in tokens.items())
    return f"{selector} {{\n{body}\n}}"


TOKENS_CSS: str = "\n\n".join([
    _emit_block(":root", THEME_TOKENS),
    # FRONTEND-F-001 (Wave 5.5): match every selector that Reflex /
    # next-themes / Radix Themes may emit when the operator flips to light:
    # - ``.light`` / ``.light-theme``: classes next-themes + Radix put on
    #   the document root when ``attribute="class"``.
    # - ``[data-theme="light"]``: legacy fallback in case Reflex switches
    #   back to a data-attribute strategy (cheap defensive measure).
    _emit_block(
        '.light, .light-theme, [data-theme="light"]',
        LIGHT_TOKENS,
    ),
    # Body baseline — picks up the dark surface + Geist body type.
    """body {
  background: var(--bp-bg);
  color: var(--bp-text);
  font-family: var(--bp-sans);
  font-size: 14px;
  line-height: 1.5;
}

code, pre, .mono {
  font-family: var(--bp-mono);
}

/* WCAG 2.4.7 — preserve focus rings for keyboard users.
   Tailored after the Stage C theme.py contract; this stays so accessibility
   audits keep passing across the framework migration. */
:focus-visible {
  outline: none;
  box-shadow: var(--bp-focus);
  border-radius: var(--bp-r-2);
}

@media (forced-colors: active) {
  :focus-visible {
    outline: 3px solid CanvasText;
    box-shadow: none;
  }
}

@media (prefers-contrast: more) {
  :focus-visible {
    outline: 3px solid var(--bp-text);
    box-shadow: none;
  }
}

/* ─────────────────────────────────────────────────────────────────────
   Animation keyframes — the load-bearing "alive AND healthy" signals.

   - bp-heartbeat / .bp-heartbeat-2400: 2.4s active-state pulse on the
     status dot. Slow on purpose; faster reads as nervous.
   - .bp-pulse-1600: 1.6s pulse, used for the paused state.
   - bp-tick / .bp-tick: dim 100% → 45% for 80ms every 1.6s on the live
     step counter. Imperceptible unless watched — the "data is still
     arriving" signal. (Per design digest §4b.)
   - .bp-num: tabular numerals for live-metrics (loss, step count, etc.).

   Respect prefers-reduced-motion: silence the animations rather than
   removing them, so the layout stays stable for vestibular-sensitive
   users.
   ───────────────────────────────────────────────────────────────────── */
@keyframes bp-heartbeat {
  0%, 100% { opacity: 1.0; }
  50%      { opacity: 0.55; }
}

.bp-heartbeat-2400 {
  animation: bp-heartbeat 2.4s cubic-bezier(0.4, 0.0, 0.6, 1.0) infinite;
}

.bp-pulse-1600 {
  animation: bp-heartbeat 1.6s ease-in-out infinite;
}

@keyframes bp-tick {
  0%, 95%, 100% { opacity: 1.0; }
  97.5%         { opacity: 0.45; }
}

.bp-tick {
  animation: bp-tick 1.6s linear infinite;
}

.bp-num {
  font-variant-numeric: tabular-nums;
}

@media (prefers-reduced-motion: reduce) {
  .bp-heartbeat-2400,
  .bp-pulse-1600,
  .bp-tick {
    animation: none;
  }
}""",
])


__all__ = [
    "RADIX_THEME",
    "THEME_TOKENS",
    "LIGHT_TOKENS",
    "STYLESHEETS",
    "TOKENS_CSS",
]
