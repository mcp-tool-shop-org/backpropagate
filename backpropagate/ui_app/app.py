"""Reflex app entry point — wires theme + tokens + pages.

This module is imported by:

- ``reflex run`` (via the ``rxconfig.py`` ``app_name`` setting), which loads
  ``ui_app.app:app`` and serves it on a backend port + frontend port.
- Unit tests that smoke-import the app to verify the page tree builds cleanly
  (``python -c "from backpropagate.ui_app.app import app"``).
"""

from __future__ import annotations

import reflex as rx

# Absolute imports so this module works whether Reflex loads it as
# ``ui_app.app`` (CLI subprocess from inside backpropagate/) or as
# ``backpropagate.ui_app.app`` (Python smoke tests).
from backpropagate.ui_theme import RADIX_THEME, STYLESHEETS, TOKENS_CSS

from .pages.dataset import dataset_page
from .pages.export import export_page
from .pages.multi_run import multi_run_page
from .pages.train import train_page


def _with_tokens(page: rx.Component) -> rx.Component:
    """Wrap a page in a fragment that injects the THEME_TOKENS stylesheet.

    Reflex doesn't have a "root layout" concept; injecting the tokens via
    ``rx.el.style`` (which renders as a literal ``<style>`` tag) on every page
    is the cleanest way to ensure the CSS custom properties are present
    regardless of which route is loaded directly.
    """
    return rx.fragment(
        rx.el.style(TOKENS_CSS),
        page,
    )


app = rx.App(
    theme=rx.theme(**RADIX_THEME),
    stylesheets=STYLESHEETS,
)


# Register the 4 surfaces. ``add_page`` is preferred over ``@rx.page`` here so
# the page registry stays in one place and the imports remain explicit.
app.add_page(lambda: _with_tokens(train_page()), route="/", title="backpropagate · train")
app.add_page(lambda: _with_tokens(multi_run_page()), route="/multi-run", title="backpropagate · multi-run")
app.add_page(lambda: _with_tokens(export_page()), route="/export", title="backpropagate · export")
app.add_page(lambda: _with_tokens(dataset_page()), route="/dataset", title="backpropagate · dataset")
