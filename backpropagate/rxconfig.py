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

import reflex as rx

config = rx.Config(
    app_name="ui_app",
    # Override Reflex's default ``<app_name>/<app_name>.py`` convention so the
    # app definition lives in ``ui_app/app.py`` (more readable / standard).
    app_module_import="ui_app.app",
    db_url=None,
    backend_only=False,
)
