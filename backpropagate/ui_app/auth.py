"""Auth enforcement for the Reflex web UI.

This module is the integration point for future HTTP/WS authentication
middleware. Until that middleware lands (tracked as a Wave 6 feature in the
dogfood swarm), `ENFORCEMENT_AVAILABLE` is False.

When False, `cli.py:cmd_ui` refuses to start with `--auth` or `--share` and
returns `RUNTIME_UI_AUTH_NOT_ENFORCED` (see `backpropagate.exceptions`).
This is the patched behavior for the v1.1.0 / v1.1.1 false-promise issue
(GHSA-pending) where the CLI announced "Auth: enabled" but the runtime
ignored `BACKPROPAGATE_UI_AUTH` entirely.

When the middleware lands, flip `ENFORCEMENT_AVAILABLE` to True and add
the FastAPI middleware registration to `ui_app/app.py`.
"""

ENFORCEMENT_AVAILABLE: bool = False
"""True when the Reflex UI actually enforces ``BACKPROPAGATE_UI_AUTH``.

See module docstring for the rollout plan."""
