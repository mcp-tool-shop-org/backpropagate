"""``bp_icon`` — inline-SVG icon helper (ui-v2 P1 redesign).

Why this exists: every icon in ``backpropagate/assets/icons/`` is authored
with ``stroke``/``fill`` = ``currentColor`` so the glyph inherits the text
color around it. That only works when the SVG is INLINED into the document.
The old pattern served them via ``rx.image(src="/icons/<name>.svg")`` — an
``<img>`` — where ``currentColor`` has no parent context and resolves to the
SVG default (black). Every icon therefore rendered near-invisible in dark
mode and could never pick up muted/accent tints in either theme
(FRONTEND-B-005 documented this as a dead ``color`` style; the note
predicted this fix: "swap to inline-SVG … so currentColor picks up the
parent's color").

The files are static assets we ship, read once per name at build time and
memoized. Names are validated against a strict allow-pattern so a dynamic
call site can never traverse out of the icon directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import reflex as rx

_ICON_DIR = Path(__file__).resolve().parents[2] / "assets" / "icons"
_NAME_RE = re.compile(r"[a-z0-9-]+")
_cache: dict[str, str] = {}


def _load_svg(name: str) -> str:
    """Read ``assets/icons/<name>.svg`` once and memoize the source."""
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"invalid icon name: {name!r}")
    if name not in _cache:
        _cache[name] = (_ICON_DIR / f"{name}.svg").read_text(encoding="utf-8").strip()
    return _cache[name]


def bp_icon(name: str, size: int = 20, color: str = "inherit", label: str = "") -> rx.Component:
    """Inline one icon.

    Parameters
    ----------
    name:
        File stem under ``assets/icons/`` (e.g. ``"train"``).
    size:
        Square edge in px. Director's bar: 20px for nav and header controls.
    color:
        CSS color applied to the wrapper; the SVG's ``currentColor``
        strokes/fills inherit it. ``"inherit"`` (default) follows the
        surrounding text color.
    label:
        When non-empty the glyph is meaningful rather than decorative:
        the wrapper gets ``role="img"`` + ``aria-label``. Empty (default)
        marks it ``aria-hidden``.
    """
    svg = _load_svg(name)
    a11y = (
        f'role="img" aria-label="{label}"'
        if label
        else 'aria-hidden="true"'
    )
    return rx.html(
        f'<span class="bp-icon" data-icon="{name}" {a11y} '
        f'style="width:{size}px;height:{size}px;color:{color}">{svg}</span>'
    )


__all__ = ["bp_icon"]
