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
- ``STYLESHEETS`` — external CSS hrefs (none since ui-v2 P1; fonts are
  vendored woff2 under ``assets/fonts/`` and inlined via ``FONTS_CSS``).
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
# ``ui_app/app.py`` passes ``appearance="inherit"`` at the call site so
# the Radix theme re-tints whenever Reflex's next-themes provider flips
# (operator click on the header toggle OR ``prefers-color-scheme`` change on
# first load). Hard-coding ``"dark"`` here would override that and
# strand the toggle button — the v1.2 bug FRONTEND-F-001 was caught for.
# (Not ``appearance=rx.color_mode``: on Reflex 0.9.3 / 0.9.5 that compiles to
# ``defaultColorMode = rawColorMode``, a ReferenceError at page load.)
RADIX_THEME: dict[str, object] = {
    "accent_color": "teal",        # Ocean Mist primary
    "gray_color": "slate",         # cool neutrals; matches bg #0F1316
    # ui-v2 P1 redesign: Radix's "large" radius (~8-10px on inputs/selects)
    # matches the Director's 8-10px field-radius bar; cards/buttons set
    # explicit radii via the --bp-r-* tokens.
    "radius": "large",
    "scaling": "100%",
    "panel_background": "solid",
    "has_background": True,
}


# ---------------------------------------------------------------------------
# 2. THEME_TOKENS (dark mode, default)
# ---------------------------------------------------------------------------

THEME_TOKENS: dict[str, str] = {
    # surfaces
    # ui-v2 P1 redesign pass 2/3: page bg dark AND card face lifted.
    # Pass 2 (#0C0F13 on #1A1F25) measured only ~4 perceptual units of
    # separation at 1px borders — invisible on wide shots. Pass 3 lifts the
    # card face to #20262E (≈2x the step) with a brighter hairline.
    "--bp-bg":        "#0C0F13",
    "--bp-surface":   "#20262E",
    "--bp-surface-2": "#2A313B",
    "--bp-surface-3": "#343D49",
    "--bp-border":    "#424D5C",
    "--bp-border-2":  "#515D6E",
    # text
    "--bp-text":      "#ECF1F5",
    "--bp-text-2":    "#C7D1D9",
    "--bp-muted":     "#8DA0AD",   # refined: lifted from #78909C for AA at 14px
    "--bp-muted-2":   "#8B9BA7",   # ui-v2 P3: AA (4.6:1) on surface-2; was 3.5:1
    # accents
    "--bp-teal":      "#7EC8C8",   # primary
    "--bp-blue":      "#A8C5E2",   # secondary (HF download events)
    "--bp-seafoam":   "#98D4BB",   # success / OK
    "--bp-amber":     "#E8C28B",   # warn (recovered, not erroring)
    "--bp-peach":     "#E8A88B",   # error code identifiers (BackpropError · E_*)
    # type
    "--bp-sans": '"Geist", ui-sans-serif, system-ui, -apple-system, sans-serif',
    "--bp-mono": '"Geist Mono", ui-monospace, "SF Mono", Menlo, monospace',
    # legacy radii ladder (kept for v1.x call sites)
    "--bp-r-1": "4px",
    "--bp-r-2": "6px",
    "--bp-r-3": "8px",
    "--bp-r-4": "12px",
    # ui-v2 P1 redesign: the Director's curve + spacing bar.
    # Radii: inputs/selects 10px, cards 14px, buttons pill.
    "--bp-r-sm": "6px",
    "--bp-r-md": "10px",
    "--bp-r-lg": "14px",
    "--bp-r-pill": "999px",
    # One spacing scale used everywhere: 4/8/12/16/24/32/48.
    "--bp-space-1": "4px",
    "--bp-space-2": "8px",
    "--bp-space-3": "12px",
    "--bp-space-4": "16px",
    "--bp-space-5": "24px",
    "--bp-space-6": "32px",
    "--bp-space-7": "48px",
    # Field (input) surface sits one step deeper than the card it lives in,
    # so fields read as inset wells rather than bordered boxes.
    "--bp-field-bg": "#12171D",
    # Soft elevation: cards float over the page background without a hard
    # 1px-grid look.
    "--bp-shadow-card": "0 1px 2px rgba(0, 0, 0, 0.45), 0 16px 36px rgba(0, 0, 0, 0.50)",
    "--bp-shadow-pop": "0 2px 6px rgba(0, 0, 0, 0.48), 0 20px 56px rgba(0, 0, 0, 0.55)",
    # focus ring — WCAG 2.4.7, do not remove
    "--bp-focus": "0 0 0 2px var(--bp-bg), 0 0 0 4px var(--bp-teal)",
}


# ---------------------------------------------------------------------------
# 2a. LIGHT_TOKENS (light mode parity)
# ---------------------------------------------------------------------------

LIGHT_TOKENS: dict[str, str] = {
    # ui-v2 P1 redesign pass 2/3: page bg one step deeper/bluer so white
    # cards read as panels; hairline border + shadow carry the edge (the
    # 0.06-alpha shadow from pass 2 was invisible at 1920px).
    "--bp-bg":        "#E9EEF2",
    "--bp-surface":   "#FFFFFF",
    "--bp-surface-2": "#F0F3F6",
    "--bp-surface-3": "#E4EAEF",
    "--bp-border":    "#C3CDD6",
    "--bp-border-2":  "#ADBAC6",
    "--bp-text":      "#131820",
    "--bp-text-2":    "#2E3A47",
    "--bp-muted":     "#5A6B78",
    # ui-v2 P3 accessibility pass: every text-bearing token is AA (>= 4.5:1)
    # on the light card faces; the previous values measured 2.9-4.1:1.
    "--bp-muted-2":   "#5C6A76",
    "--bp-teal":      "#1F7373",
    "--bp-blue":      "#3F6A99",
    "--bp-seafoam":   "#2B7F5E",
    "--bp-amber":     "#8F6020",
    "--bp-peach":     "#A04B2C",
    # ui-v2 P1 redesign additions (parity with THEME_TOKENS)
    "--bp-field-bg":  "#FBFDFE",
    "--bp-shadow-card": "0 1px 2px rgba(23, 31, 41, 0.08), 0 12px 32px rgba(23, 31, 41, 0.12)",
    "--bp-shadow-pop":  "0 2px 6px rgba(23, 31, 41, 0.10), 0 20px 48px rgba(23, 31, 41, 0.16)",
}


# ---------------------------------------------------------------------------
# 2b. Self-hosted fonts (ui-v2 P1)
#
# Draft2 audit: Geist never actually loaded — the Google Fonts stylesheet
# tripped the CSP (headless Chrome logs showed the block) AND it cannot work
# offline, which the Microsoft Store build requires. The variable woff2
# builds are vendored under ``backpropagate/assets/fonts/`` (OFL 1.1, Geist
# v1.7.2 — see assets/fonts/NOTICE.md for hashes + upstream URL) and served
# same-origin at ``/fonts/<name>`` by Reflex's assets handler.

FONTS_CSS: str = """@font-face {
  font-family: 'Geist';
  font-weight: 100 900;
  font-display: swap;
  src: url('/fonts/geist-var.woff2') format('woff2');
}
@font-face {
  font-family: 'Geist Mono';
  font-weight: 100 900;
  font-display: swap;
  src: url('/fonts/geist-mono.woff2') format('woff2');
}"""

#: External stylesheet hrefs. Empty on purpose — every style the app needs
#: is inlined via TOKENS_CSS so first paint has zero third-party surface
#: (CSP ``style-src 'self' 'unsafe-inline'``; ``font-src 'self'``).
STYLESHEETS: list[str] = []


# ---------------------------------------------------------------------------
# 2c. TOKENS_CSS — pre-assembled stylesheet for inline injection
# ---------------------------------------------------------------------------


def _emit_block(selector: str, tokens: dict[str, str]) -> str:
    body = "\n".join(f"  {k}: {v};" for k, v in tokens.items())
    return f"{selector} {{\n{body}\n}}"


TOKENS_CSS: str = "\n\n".join([
    FONTS_CSS,
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

/* Radix Themes reads these custom props for its component font stacks
   (ui-v2 P1): route them at our self-hosted Geist pair so buttons, badges,
   headings and code blocks stop falling back to the Radix default stack. */
.radix-themes {
  --default-font-family: var(--bp-sans);
  --heading-font-family: var(--bp-sans);
  --code-font-family: var(--bp-mono);
  --strong-font-family: var(--bp-sans);
  --em-font-family: var(--bp-sans);
  --quote-font-family: var(--bp-sans);
}

/* Native <details> accordion (ui-v2 P1): hide the UA triangle, rotate our
   chevron, keep the fold keyboard-accessible. */
.bp-accordion > summary {
  list-style: none;
}
.bp-accordion > summary::-webkit-details-marker {
  display: none;
}
.bp-accordion .bp-accordion-chevron {
  display: inline-flex;
  transition: transform 0.2s ease;
  color: var(--bp-muted-2);
}
.bp-accordion[open] .bp-accordion-chevron {
  transform: rotate(180deg);
}
.bp-accordion > summary:hover .bp-accordion-chevron {
  color: var(--bp-text-2);
}

code, pre, .mono {
  font-family: var(--bp-mono);
}

/* Inline SVG icons (ui-v2 P1 redesign pass 2): the icon pack uses
   stroke/fill=currentColor, which only inherits when the SVG is inlined
   (served via <img> it always resolves to black). See components/icon.py. */
.bp-icon {
  display: inline-flex;
  flex-shrink: 0;
  line-height: 0;
}
.bp-icon svg {
  width: 100%;
  height: 100%;
  display: block;
}

/* ─────────────────────────────────────────────────────────────────────
   ui-v2 P1 redesign — motion discipline. All interactive surfaces get a
   short (≤180ms) ease transition on hover/focus/state; nothing bounces,
   nothing longer than 0.25s. Radix Buttons + TextFields get the shared
   base so every page inherits it without per-site style noise.
   ───────────────────────────────────────────────────────────────────── */
.rt-Button {
  transition: background-color 0.15s ease, box-shadow 0.15s ease,
    color 0.15s ease, opacity 0.15s ease;
  cursor: pointer;
}
.rt-Button:disabled {
  cursor: not-allowed;
}
.rt-TextFieldInput,
.rt-SelectTrigger {
  transition: border-color 0.15s ease, box-shadow 0.15s ease,
    background-color 0.15s ease;
}
.bp-nav-row {
  border-radius: var(--bp-r-pill);
  transition: background-color 0.15s ease, color 0.15s ease;
}
.bp-nav-row:hover {
  background: var(--bp-surface-2);
}

/* ui-v2 P3 accessibility pass (WCAG AA text contrast):
   - solid teal buttons: dark text on teal-9 (6.3:1; white was 3.1:1);
   - light mode: Radix teal-11 (accent text, soft badges/buttons) darkened
     to 5.4:1 on teal-3 (was 4.1:1), green-11 likewise;
   - select placeholders use the muted token (Radix's alpha gray was 4.4:1).
   Radix sets --accent-11 per [data-accent-color]; overriding it there (one
   class more specific) reaches badges, soft buttons and the auth chip. */
.rt-Button.rt-variant-solid[data-accent-color="teal"] {
  color: #04201d;
}
:is(.light, .light-theme, [data-theme="light"]) [data-accent-color="teal"],
:is(.light, .light-theme, [data-theme="light"])[data-accent-color="teal"] {
  --accent-11: #00705f;
  --accent-a11: #00705f;
}
:is(.light, .light-theme, [data-theme="light"]) [data-accent-color="green"],
:is(.light, .light-theme, [data-theme="light"])[data-accent-color="green"] {
  --accent-11: #0b5d33;
  --accent-a11: #0b5d33;
}
.rt-SelectTrigger[data-placeholder] .rt-SelectTriggerInner,
.rt-SelectTrigger[data-placeholder] {
  color: var(--bp-muted);
}

/* The primary action stays in view (ui-v2 P3): the estimate and Start
   button dock at the bottom of the scrolling column; the form scrolls
   underneath a short fade. */
.bp-action-bar {
  position: sticky;
  bottom: 0;
  z-index: 5;
  padding: 20px 0 12px;
  background: linear-gradient(to bottom, transparent, var(--bp-bg) 20px);
}

/* Choice cards (ui-v2 P3): a radio option as a selectable card. The whole
   card is the label, so clicking anywhere picks it; the checked card takes
   the teal edge, the focused radio's card shows the focus ring. */
.bp-choice {
  display: block;
  padding: 12px 14px;
  background: var(--bp-surface-2);
  border: 1px solid var(--bp-border);
  border-radius: var(--bp-r-md);
  cursor: pointer;
  transition: border-color 0.15s ease, background-color 0.15s ease,
    box-shadow 0.15s ease;
}
.bp-choice:hover {
  border-color: var(--bp-border-2);
}
.bp-choice:has([data-state="checked"]) {
  border-color: var(--bp-teal);
  background: color-mix(in srgb, var(--bp-teal) 10%, var(--bp-surface-2));
}
.bp-choice:has(:focus-visible) {
  box-shadow: var(--bp-focus);
}
.bp-choice:has([data-disabled]) {
  cursor: not-allowed;
  opacity: 0.6;
}

/* LoRA shape cards (components/train_form.py): three pressable cards. The
   pressed one takes the teal edge, like a checked choice card. */
.bp-shape {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  width: 100%;
  height: 100%;
  padding: 12px 14px;
  text-align: left;
  font: inherit;
  color: inherit;
  background: var(--bp-surface-2);
  border: 1px solid var(--bp-border);
  border-radius: var(--bp-r-md);
  cursor: pointer;
  transition: border-color 0.15s ease, background-color 0.15s ease,
    box-shadow 0.15s ease;
}
.bp-shape:hover {
  border-color: var(--bp-border-2);
}
.bp-shape[aria-pressed="true"] {
  border-color: var(--bp-teal);
  background: color-mix(in srgb, var(--bp-teal) 10%, var(--bp-surface-2));
}
.bp-shape:focus-visible {
  border-radius: var(--bp-r-md);
}
.bp-shape:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}
.bp-shape-badge {
  margin-top: 4px;
  padding: 1px 8px;
  border-radius: var(--bp-r-pill);
  font-size: 11px;
  font-weight: 600;
  color: var(--bp-teal);
  background: color-mix(in srgb, var(--bp-teal) 16%, transparent);
  white-space: nowrap;
}

/* Info tips: the small "i" beside a label (components/info_tip.py). A real
   button with a 24px target (WCAG 2.5.8); muted until hovered or focused. */
.bp-tip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  margin: -4px 0;
  padding: 0;
  border: 0;
  border-radius: 50%;
  background: transparent;
  color: var(--bp-muted);
  cursor: help;
  flex: none;
  transition: color 0.15s ease, background-color 0.15s ease;
}
.bp-tip:hover,
.bp-tip:focus-visible,
.bp-tip[data-state="open"] {
  color: var(--bp-teal);
  background: color-mix(in srgb, var(--bp-teal) 14%, transparent);
}
.bp-tip:focus-visible {
  border-radius: 50%;
}
.bp-tip-card {
  background: var(--bp-surface-2) !important;
  border: 1px solid var(--bp-border-2);
  border-radius: var(--bp-r-md) !important;
  box-shadow: var(--bp-shadow-pop) !important;
  padding: 14px 16px !important;
}
.bp-tip-start {
  margin-top: 2px;
  padding: 8px 10px;
  border-left: 2px solid var(--bp-teal);
  background: color-mix(in srgb, var(--bp-teal) 8%, transparent);
  border-radius: 0 var(--bp-r-2) var(--bp-r-2) 0;
}
/* Text that is in the page for screen readers and not drawn. */
.bp-visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
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
