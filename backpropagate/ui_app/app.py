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

# FRONTEND-B-003 (Stage C humanization): catch a broad ``Exception`` (not
# just ``ImportError``) so a syntax error or transitive import crash in
# auth.py produces a humanized refuse-to-start rather than a confusing
# ``NameError`` further down. If ``basic_auth_transformer`` is missing we
# cannot wire ``rx.App(api_transformer=...)``; failing fast with the same
# canonical message that the env-var guard prints is the correct shape.
try:
    from .auth import ENFORCEMENT_AVAILABLE, basic_auth_transformer
except Exception as _exc:  # noqa: BLE001
    raise RuntimeError(
        "FRONTEND-B-001 / FRONTEND-B-003 / GHSA-f65r-h4g3-3h9h: the auth "
        "middleware module (backpropagate.ui_app.auth) failed to import - "
        "refusing to start the Reflex UI because it would bind a port and "
        "serve traffic without the load-bearing middleware. Reinstall the "
        "[ui] extra with `pip install --force-reinstall 'backpropagate[ui]'` "
        "to restore the module, then re-run `backprop ui`. Underlying import "
        f"error: {type(_exc).__name__}: {_exc}"
    ) from _exc

from .pages.dataset import dataset_page
from .pages.export import export_page
from .pages.multi_run import multi_run_page
from .pages.runs import runs_page
from .pages.train import train_page

# FRONTEND-B-001 / GHSA-f65r-h4g3-3h9h defense-in-depth (layer 3 of 4):
# cli.py:cmd_ui refuses --auth/--share when ENFORCEMENT_AVAILABLE is False, but
# that check is bypassed if a user runs ``python -m reflex run`` or ``reflex
# run`` directly from the package directory. Fire the same refuse-to-start
# check here at module import time so the Reflex app cannot bind a port and
# serve an unauthenticated UI when the operator believes auth is wired up.
#
# The check requires BOTH: ENFORCEMENT_AVAILABLE=False (the [ui] extra is
# degraded or the auth module failed to import) AND BACKPROPAGATE_UI_AUTH set
# in the env (operator believes auth is active). In a healthy v1.2.0 install,
# ENFORCEMENT_AVAILABLE is True and this guard is inert — the real FastAPI
# middleware wired below via rx.App(api_transformer=basic_auth_transformer)
# enforces the credential on every HTTP route and the /_event WS upgrade.
if not ENFORCEMENT_AVAILABLE and os.environ.get("BACKPROPAGATE_UI_AUTH"):
    raise RuntimeError(
        "FRONTEND-B-001 / GHSA-f65r-h4g3-3h9h: BACKPROPAGATE_UI_AUTH is set, "
        "but backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE is False — the "
        "[ui] extra is degraded or the auth middleware module failed to "
        "import. Refusing to start to prevent the v1.1.0 false-promise (CLI "
        "announces 'Auth: enabled' while the runtime ignores the credential). "
        "Reinstall the [ui] extra (`pip install -U 'backpropagate[ui]'`) to "
        "restore the middleware, or unset BACKPROPAGATE_UI_AUTH and use SSH "
        "port-forwarding for remote access: ssh -L 7860:localhost:7860 <host>"
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


# Wave 6: real auth middleware via Reflex's ``api_transformer`` hook (the
# documented Reflex >=0.8 surface — App.api was removed in 0.8.0). The
# transformer wraps the WHOLE ASGI app, so HTTP routes AND the /_event
# WebSocket upgrade go through the same gate. See ``ui_app/auth.py`` for
# the full mode matrix + cookie shape + pre-accept WS validation.
#
# Order matters: the auth middleware is the outermost wrapper, so it sees
# the raw scope before Reflex (or any other future api_transformer entry)
# touches it. v1.3 will chain a request-logging + rate-limit middleware in
# the same slot.
app = rx.App(
    theme=rx.theme(**RADIX_THEME),
    stylesheets=STYLESHEETS,
    api_transformer=basic_auth_transformer,
)


# Register the 5 surfaces. ``add_page`` is preferred over ``@rx.page`` here so
# the page registry stays in one place and the imports remain explicit. The
# ``/runs`` surface is the Wave 6 v1.2.0 add — it closes the CLI/UI parity
# gap (``backprop list-runs`` had no UI equivalent until now).
app.add_page(lambda: _with_tokens(train_page()), route="/", title="backpropagate · train")
app.add_page(lambda: _with_tokens(multi_run_page()), route="/multi-run", title="backpropagate · multi-run")
app.add_page(lambda: _with_tokens(export_page()), route="/export", title="backpropagate · export")
app.add_page(lambda: _with_tokens(dataset_page()), route="/dataset", title="backpropagate · dataset")
app.add_page(lambda: _with_tokens(runs_page()), route="/runs", title="backpropagate · runs")
