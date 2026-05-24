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

# FRONTEND-B-001 / GHSA-f65r-h4g3-3h9h defense-in-depth (layer 4 of 4):
# rxconfig.py is the first module Reflex imports when ``reflex run`` is
# invoked from the package directory. Some invocations may bypass app.py's
# import chain (e.g., ``reflex export`` style probes), so we fire the same
# refuse-to-start check here too. Importing from ui_app.auth is safe - that
# module has no Reflex dependencies and only exposes the ENFORCEMENT_AVAILABLE
# flag. See ``backpropagate/ui_app/app.py`` for the same guard with full
# rationale.
#
# FRONTEND-B-003 (Stage C humanization): catch a broad ``Exception`` (not
# just ``ImportError``) so that a syntax error in auth.py or a transitive
# import crash (e.g. a downgraded Reflex with a broken import path) does
# NOT fall through to a confusing ``NameError`` further down. If both the
# package-relative AND the absolute import fail, we surface a humanized
# refuse-to-start with the underlying cause AND the install hint - the
# operator should never see a bare ``NameError: name 'ENFORCEMENT_AVAILABLE'
# is not defined`` traceback for a load-bearing security check.
ENFORCEMENT_AVAILABLE: bool
try:
    from ui_app.auth import ENFORCEMENT_AVAILABLE  # type: ignore[no-redef]
except Exception as _exc:  # noqa: BLE001 — broken module is broken module
    try:
        from backpropagate.ui_app.auth import ENFORCEMENT_AVAILABLE  # type: ignore[no-redef]
    except Exception as _exc2:  # noqa: BLE001
        if os.environ.get("BACKPROPAGATE_UI_AUTH"):
            raise RuntimeError(
                "FRONTEND-B-001 / FRONTEND-B-003 / GHSA-f65r-h4g3-3h9h: "
                "BACKPROPAGATE_UI_AUTH is set, but the auth middleware module "
                "failed to import. Refusing to start to prevent the v1.1.0 "
                "false-promise (CLI announces 'Auth: enabled' while the "
                "runtime ignores the credential). Reinstall the [ui] extra "
                "(`pip install --force-reinstall 'backpropagate[ui]'`) to "
                "restore the middleware, or unset BACKPROPAGATE_UI_AUTH and "
                "use SSH port-forwarding for remote access: "
                f"`ssh -L 7860:localhost:7860 <host>`. Underlying import "
                f"error: {type(_exc2).__name__}: {_exc2}"
            ) from _exc2
        # Operator did NOT set BACKPROPAGATE_UI_AUTH, so the no-auth-local-
        # only path is legitimate; the auth module being unimportable is
        # still surprising (Reflex would have crashed elsewhere), but the
        # safe behavior here is to treat enforcement as unavailable and let
        # the no-auth path proceed (the CLI's refuse-to-start rails are the
        # gate that ensures we only reach this code for the loopback case).
        ENFORCEMENT_AVAILABLE = False

if not ENFORCEMENT_AVAILABLE and os.environ.get("BACKPROPAGATE_UI_AUTH"):
    raise RuntimeError(
        "FRONTEND-B-001 / GHSA-f65r-h4g3-3h9h: BACKPROPAGATE_UI_AUTH is set, "
        "but backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE is False - the "
        "[ui] extra is degraded or the auth middleware module failed to "
        "import. Refusing to start to prevent the v1.1.0 false-promise (CLI "
        "announces 'Auth: enabled' while the runtime ignores the credential). "
        "Reinstall the [ui] extra (`pip install -U 'backpropagate[ui]'`) to "
        "restore the middleware, or unset BACKPROPAGATE_UI_AUTH and use SSH "
        "port-forwarding for remote access: ssh -L 7860:localhost:7860 <host>"
    )

# FRONTEND-F-CORS-ORIGINS-LOCK (Wave 6, FRONTEND-B-011 closed):
# Reflex's default cors_allowed_origins is ['*'], which lets any browser /
# any origin open the backend WebSocket. The Wave 6 auth middleware
# (ui_app/auth.py::basic_auth_transformer) adds Origin allowlist validation
# on WS upgrade as the load-bearing CSWSH defense (per Schneider 2013 /
# CWE-1385); cors_allowed_origins is the FastAPI-CORSMiddleware layer that
# runs at a different level (HTTP response Access-Control-Allow-Origin) and
# is the FIRST gate browsers consult for cross-origin XHR/fetch. Lock both
# layers as defense-in-depth.
#
# The default allowlist covers loopback at the two Reflex ports (frontend +
# backend). When the auth middleware lands and runs in --share / --host
# modes, the operator-supplied tunnel/LAN host is added DYNAMICALLY by the
# middleware on a per-request basis (Host-header allowlist there); CORS is
# the static fallback for the common loopback case.
_DEFAULT_CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:7860",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:7860",
]

# Honor the operator port if they override BACKPROPAGATE_UI_PORT.
_ui_port = os.environ.get("BACKPROPAGATE_UI_PORT", "").strip()
if _ui_port and _ui_port.isdigit():
    for host in ("localhost", "127.0.0.1"):
        candidate = f"http://{host}:{_ui_port}"
        if candidate not in _DEFAULT_CORS_ALLOWED_ORIGINS:
            _DEFAULT_CORS_ALLOWED_ORIGINS.append(candidate)

# Operator escape hatch (env var, comma-separated) — additive, never replaces
# the loopback defaults. Documented in handbook/env-vars.md.
_extra_cors = os.environ.get("BACKPROPAGATE_UI_CORS_EXTRA_ORIGINS", "").strip()
if _extra_cors:
    for raw in _extra_cors.split(","):
        origin = raw.strip()
        if origin and origin not in _DEFAULT_CORS_ALLOWED_ORIGINS:
            _DEFAULT_CORS_ALLOWED_ORIGINS.append(origin)


config = rx.Config(
    app_name="ui_app",
    # Override Reflex's default ``<app_name>/<app_name>.py`` convention so the
    # app definition lives in ``ui_app/app.py`` (more readable / standard).
    app_module_import="ui_app.app",
    db_url=None,
    backend_only=False,
    cors_allowed_origins=_DEFAULT_CORS_ALLOWED_ORIGINS,
)
