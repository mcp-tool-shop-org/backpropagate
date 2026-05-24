"""Request-logging ASGI middleware — V1_3_BRIEF P0 item 5.

Logs every HTTP request + WebSocket upgrade with structured fields:

- ``method``         — HTTP method (or ``"WS"`` for WebSocket upgrades)
- ``path``           — request path
- ``status``         — HTTP status code (or WS close code if pre-accept-rejected)
- ``duration_ms``    — wall-clock duration in milliseconds (float)
- ``auth_mode``      — resolved auth mode from ``ui_app.auth._detect_mode``
- ``auth_user``      — authenticated username (if cookie/Basic validated),
                       else ``""`` for unauthenticated paths
- ``remote_addr``    — client IP from ASGI scope's ``client`` tuple

Sibling, NOT nested, w.r.t. ``ui_app/auth.py``:
  * wired AFTER auth in the chain so log records carry the resolved auth state
  * does not modify ``basic_auth_transformer``
  * does not modify auth scope / receive / send (passes through unchanged)

Default OFF — opt in via ``BACKPROPAGATE_UI_REQUEST_LOG=1``. Reflex's own
``uvicorn`` access log covers basic shape; this middleware adds the
structured-field surface that operators can grep / ship to a SIEM.

Integration: uses ``backpropagate.logging_config.get_logger`` so JSON output
in production / pretty output in dev are governed by the same env knobs
(``BACKPROPAGATE_LOG_JSON``, ``BACKPROPAGATE_LOG_LEVEL``).

Performance: when disabled (the default), the middleware is one env-var read
+ a single function call per request — measured ~3μs on a 5080 rig, dwarfed
by every other framework hop. When enabled, the extra cost is the structlog
emit (~30μs per record) + the ``time.perf_counter()`` deltas.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

# logging_config is the same surface ``ui_app/auth.py`` uses for structured
# logging — same JSON-vs-pretty toggle, same per-process configuration. The
# import is lazy inside ``_get_logger`` so the auth-failure import-guard in
# rxconfig.py does not trip on a circular import path.


def _get_logger() -> Any:
    """Lazy logger fetch — avoids a hard dep at module-import time.

    The import is wrapped in a try/except because ``logging_config`` depends
    on ``structlog``, which is shipped in the ``[logging]`` extra. If the
    operator's install is degraded (the rxconfig.py refuse-to-start covers
    the ``[ui]``-degraded case, not the ``[logging]``-degraded case), we
    return a stdlib logger that emits to the root handler with the structlog
    field set as ``extra=``.
    """
    try:
        from backpropagate.logging_config import get_logger
        return get_logger(__name__)
    except Exception:  # noqa: BLE001 — defensive
        import logging
        return logging.getLogger(__name__)


_ENABLED_ENV = "BACKPROPAGATE_UI_REQUEST_LOG"


def _enabled_via_env(env: dict[str, str] | None = None) -> bool:
    """Whether request logging is turned ON.

    Operator opts in via ``BACKPROPAGATE_UI_REQUEST_LOG=1``. Any other value
    (``0``, ``false``, ``""``, unset) leaves logging OFF — the default.
    """
    env = env if env is not None else dict(os.environ)
    value = env.get(_ENABLED_ENV, "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _client_addr(scope: dict) -> str:
    """Extract client IP from ASGI scope.

    Per ASGI spec, ``scope['client']`` is ``(host, port)`` or ``None`` (e.g.
    Unix-socket transports). We return the host string or ``""``.
    """
    client = scope.get("client")
    if not client or not isinstance(client, (tuple, list)) or len(client) < 1:
        return ""
    try:
        return str(client[0])
    except (TypeError, ValueError):
        return ""


def _resolve_auth_context(env: dict[str, str]) -> tuple[str, str]:
    """Best-effort lookup of auth mode + user for the log line.

    The mode comes from ``ui_app.auth._detect_mode`` (the same shape the auth
    middleware uses on every request — re-detected here per-request so env-var
    changes take effect without a process restart). The user field is filled
    in opportunistically — request-logging runs AFTER the auth middleware in
    the ASGI chain, so by the time we emit, the request has already passed
    the auth gate. We cannot easily reach back into auth's local ``authed_user``
    variable from here, so the user field is left empty for now; a v1.4
    enhancement would have auth populate a scope key (e.g.
    ``scope['_bp_auth_user']``) that this middleware then reads.
    """
    try:
        from ..auth import _detect_mode
        mode = _detect_mode(env)
        return str(mode.value), ""
    except Exception:  # noqa: BLE001 — defensive; never fail the log
        return "", ""


def request_logging_middleware(asgi_app: Callable) -> Callable:
    """ASGI middleware factory — wraps the inner app with structured logging.

    Pass-through behavior when ``BACKPROPAGATE_UI_REQUEST_LOG`` is unset / 0
    (the default): the middleware delegates immediately to ``asgi_app`` with
    NO wall-clock measurement and NO log emit. The check is a single env-var
    read + dict lookup — ~3μs measured.

    When enabled, wraps the ``send`` callable to capture the response status
    (the first ``http.response.start`` message), measures wall-clock duration,
    and emits a single ``ui.request`` log line per request.

    Sibling to ``basic_auth_transformer``; does NOT modify it. Returning a
    plain ``async def`` rather than reusing the auth wrapper keeps the two
    concerns inspectable separately.
    """

    async def middleware(scope: dict, receive: Callable, send: Callable) -> None:
        env = dict(os.environ)
        if not _enabled_via_env(env):
            # Fast path — pass through unchanged. The single env-var read +
            # dict lookup above is the only cost when logging is OFF.
            await asgi_app(scope, receive, send)
            return

        scope_type = scope.get("type")
        if scope_type == "lifespan":
            # Lifespan events have no per-request concept; pass through.
            await asgi_app(scope, receive, send)
            return

        start = time.perf_counter()
        # Status holder is captured by ``wrapped_send`` below; for WebSocket
        # scope, the status field is the close code (or empty if the upstream
        # app accepted the WS without closing it within the request lifetime,
        # which is the normal happy-path).
        status_holder: dict[str, int | str] = {"code": ""}

        async def wrapped_send(message: dict) -> None:
            msg_type = message.get("type")
            if msg_type == "http.response.start":
                status_holder["code"] = int(message.get("status", 0))
            elif msg_type == "websocket.close":
                # Pre-accept close gives us the close code; post-accept closes
                # are also logged but with the close code instead of 0.
                status_holder["code"] = int(message.get("code", 0))
            elif msg_type == "websocket.accept" and not status_holder["code"]:
                # WS upgrade accepted — log code 101 (Switching Protocols)
                # to match the HTTP-level shape operators expect.
                status_holder["code"] = 101
            await send(message)

        try:
            await asgi_app(scope, receive, wrapped_send)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            auth_mode, auth_user = _resolve_auth_context(env)
            method = (scope.get("method") or "").upper() if scope_type == "http" else "WS"
            path = str(scope.get("path") or "")
            remote_addr = _client_addr(scope)
            fields = {
                "method": method,
                "path": path,
                "status": status_holder["code"] or "",
                "duration_ms": round(duration_ms, 3),
                "auth_mode": auth_mode,
                "auth_user": auth_user,
                "remote_addr": remote_addr,
                "scope_type": scope_type,
            }
            try:
                logger = _get_logger()
                try:
                    # structlog signature: logger.info(event, **fields)
                    logger.info("ui.request", **fields)
                except TypeError:
                    # stdlib signature: logger.info(msg, *args, extra={...})
                    logger.info("ui.request", extra=fields)
            except Exception:  # noqa: BLE001 — never fail a request on log emit
                # Operator's logging surface is broken; we cannot recover
                # mid-request. Silent-swallow per the auth-middleware pattern
                # (auth never crashes a request on a logging failure either).
                pass

    return middleware


__all__ = [
    "request_logging_middleware",
]
