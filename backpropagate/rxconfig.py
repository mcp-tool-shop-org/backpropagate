"""Reflex configuration — read by ``reflex run`` / ``reflex export``.

The CLI in ``backpropagate/cli.py``'s ``cmd_ui`` runs ``reflex run`` from
inside the ``backpropagate/`` package directory (the directory containing
this file), so Reflex picks up ``ui_app/`` as the app package and loads
``ui_app.app`` for the ``app = rx.App(...)`` instance.

Reflex's ``app_name`` must match ``^[a-zA-Z][a-zA-Z0-9_]*$`` (no dots), so
we use the flat name ``ui_app`` — cwd-based resolution finds the ``ui_app``
subdirectory directly inside the cwd Reflex is invoked from.
"""

from __future__ import annotations

import os

import reflex as rx

# FRONTEND-B-001 / GHSA-pending defense-in-depth (belt-and-suspenders):
# rxconfig.py is the first module Reflex imports when ``reflex run`` is
# invoked from the package directory. Some invocations may bypass app.py's
# import chain (e.g., ``reflex export`` style probes), so we fire the same
# refuse-to-start check here too. Importing from ui_app.auth is safe — that
# module has no Reflex dependencies and only exposes the ENFORCEMENT_AVAILABLE
# flag. See ``backpropagate/ui_app/app.py`` for the same guard with full
# rationale.
try:
    from ui_app.auth import ENFORCEMENT_AVAILABLE
except ImportError:  # pragma: no cover — package-mode import fallback
    from backpropagate.ui_app.auth import ENFORCEMENT_AVAILABLE

if not ENFORCEMENT_AVAILABLE and os.environ.get("BACKPROPAGATE_UI_AUTH"):
    raise RuntimeError(
        "FRONTEND-B-001 / GHSA-pending: The Reflex web UI does not yet "
        "enforce BACKPROPAGATE_UI_AUTH. Refusing to start to prevent the "
        "v1.1.0 false-promise (the CLI announced 'Auth: enabled' but the "
        "runtime ignored the env var). "
        "Use SSH port-forwarding for remote access until middleware lands: "
        "ssh -L 7860:localhost:7860 <host>"
    )

config = rx.Config(
    app_name="ui_app",
    # Override Reflex's default ``<app_name>/<app_name>.py`` convention so the
    # app definition lives in ``ui_app/app.py`` (more readable / standard).
    app_module_import="ui_app.app",
    db_url=None,
    backend_only=False,
)
