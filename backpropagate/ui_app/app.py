"""Reflex app entry point — wires theme + tokens + pages.

This module is imported by:

- ``reflex run`` (via the ``rxconfig.py`` ``app_name`` setting), which loads
  ``ui_app.app:app`` and serves it on a backend port + frontend port.
- Unit tests that smoke-import the app to verify the page tree builds cleanly
  (``python -c "from backpropagate.ui_app.app import app"``).
"""

from __future__ import annotations

import os

import reflex as rx

# Absolute imports so this module works whether Reflex loads it as
# ``ui_app.app`` (CLI subprocess from inside backpropagate/) or as
# ``backpropagate.ui_app.app`` (Python smoke tests).
from backpropagate.ui_theme import RADIX_THEME, STYLESHEETS, TOKENS_CSS

from .auth import ENFORCEMENT_AVAILABLE
from .pages.dataset import dataset_page
from .pages.export import export_page
from .pages.multi_run import multi_run_page
from .pages.train import train_page

# FRONTEND-B-001 / GHSA-pending defense-in-depth:
# cli.py:cmd_ui refuses --auth/--share when ENFORCEMENT_AVAILABLE is False, but
# that check is bypassed if a user runs ``python -m reflex run`` or ``reflex
# run`` directly from the package directory. Fire the same refuse-to-start
# check here at module import time so the Reflex app cannot bind a port and
# serve an unauthenticated UI when the operator believes auth is wired up.
#
# The check requires BOTH: ENFORCEMENT_AVAILABLE=False (middleware not landed)
# AND BACKPROPAGATE_UI_AUTH set in the env (operator believes auth is active).
# When ENFORCEMENT_AVAILABLE flips to True (Wave 6), this guard becomes inert
# without any further code change.
if not ENFORCEMENT_AVAILABLE and os.environ.get("BACKPROPAGATE_UI_AUTH"):
    raise RuntimeError(
        "FRONTEND-B-001 / GHSA-pending: The Reflex web UI does not yet "
        "enforce BACKPROPAGATE_UI_AUTH. Refusing to start to prevent the "
        "v1.1.0 false-promise (the CLI announced 'Auth: enabled' but the "
        "runtime ignored the env var). "
        "Use SSH port-forwarding for remote access until middleware lands: "
        "ssh -L 7860:localhost:7860 <host>"
    )


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
