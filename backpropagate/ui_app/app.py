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

# Wave 6b (v1.3) middleware siblings — wrap the same ASGI chain as auth but
# do not participate in the GHSA-f65r-h4g3-3h9h auth contracts. Wired below
# via ``rx.App(api_transformer=(...))`` as a Sequence; Reflex 0.9.x applies
# the sequence in order (first entry is the innermost wrap, last entry is
# the outermost / network-facing wrap). See ``middleware/__init__.py`` for
# the chain-ordering rationale.
from .middleware import healthz_middleware, rate_limit_middleware, request_logging_middleware
from .pages.dataset import dataset_page
from .pages.export import export_page
from .pages.models import models_page
from .pages.multi_run import multi_run_page
from .pages.run_detail import run_detail_page
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
# Wave 6b (v1.3): chain healthz + rate-limit + auth + request-logging via
# the ``Sequence`` form of ``api_transformer``. Reflex 0.9.x iterates the
# sequence in order and applies each entry as ``asgi_app = entry(asgi_app)``
# (verified against ``reflex==0.9.2.post1`` via inspect.getsource(rx.App)).
# So the FIRST entry is the innermost wrap (closest to Reflex), and the
# LAST entry is the outermost (network-facing). The desired runtime order
# when a request arrives is:
#
#     healthz_middleware               (outermost — /healthz early-exit;
#                                       orchestrator probe never touches auth
#                                       or rate-limit, returns JSON directly)
#       → rate_limit_middleware        (fast-fail 429 before auth so brute-
#                                       force can't exhaust HMAC budget)
#         → basic_auth_transformer     (UNCHANGED — GHSA contracts intact)
#           → request_logging_mw       (innermost — log line carries the
#                                       resolved auth state)
#             → Reflex
#
# Which in api_transformer sequence-order is:
#
#     (request_logging, basic_auth, rate_limit, healthz)
#       innermost first ──────────────────────────► outermost last
#
# FRONTEND-F-001 (Wave 5.5): bind ``appearance`` to ``rx.color_mode`` so
# the Radix theme re-tints whenever the operator toggles theme OR the
# ``prefers-color-scheme`` media query flips. Reflex's color-mode provider
# writes ``class="light"`` / ``class="dark"`` on ``<html>`` (next-themes
# with attribute="class"), which fires the ``.light, .light-theme,
# [data-theme="light"]`` selector in TOKENS_CSS. The v1.2 bug was that
# ``RADIX_THEME`` hard-coded ``appearance="dark"`` which stranded the
# header toggle button — it flipped server-side state but never wrote
# the DOM mutation needed to swap the active CSS variable set.
#
# The toggle button in ``BpHeader`` reads/writes via ``rx.color_mode`` +
# ``rx.toggle_color_mode`` (the documented Reflex 0.9 surface); see
# ``ui_app/chrome.py``.
app = rx.App(
    theme=rx.theme(appearance=rx.color_mode, **RADIX_THEME),
    stylesheets=STYLESHEETS,
    api_transformer=(
        request_logging_middleware,
        basic_auth_transformer,
        rate_limit_middleware,
        healthz_middleware,
    ),
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
# Wave 6b (v1.3): drill-down + models surface.
# Dynamic route: ``/runs/[run_id]`` populates ``RunDetailState`` on mount and
# renders the per-run page (mirrors ``backprop show-run``). Models surface
# lists local HF cache contents + disk usage + per-model cleanup affordance.
app.add_page(lambda: _with_tokens(run_detail_page()), route="/runs/[rid]", title="backpropagate · run detail")
app.add_page(lambda: _with_tokens(models_page()), route="/models", title="backpropagate · models")
