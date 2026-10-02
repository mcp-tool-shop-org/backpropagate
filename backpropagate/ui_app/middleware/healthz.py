"""``/healthz`` ASGI middleware — FRONTEND-5 (Wave 6b).

Lightweight orchestrator probe endpoint. Returns JSON
``{"status": "ok"}`` on GET ``/healthz``.

Why a middleware and not a Reflex page: Reflex pages render React components
and are reached through the Next.js SPA shell. An orchestrator (Kubernetes
liveness probe, AWS ELB health check, cloudflared tunnel health check)
wants a plain HTTP route with a JSON body and no HTML overhead. Wrapping
the ASGI app with an early-exit on the ``/healthz`` path is the cleanest
way to add that route without coupling to Reflex's page-tree internals.

Wired in the ASGI chain as the OUTERMOST wrap (even outside rate-limit) so
the probe is unaffected by per-IP caps and doesn't go through the auth
gate. The probe is intentionally unauthenticated — orchestrators that need
to know "is the process alive" should not have to carry credentials. The
body is exactly one key. A foreign Host is refused with 421 before that
body is built. Because this wrap sits outside the security-headers
middleware, the response stamps ``X-Content-Type-Options`` and
``X-Frame-Options`` itself.

Pre-existing ``/ping`` Reflex route remains the framework-internal health
check (it's hardcoded into Reflex's SPA). ``/healthz`` is the orchestrator-
facing canonical name (matches the Kubernetes / Knative convention).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

_HEALTH_BODY = b'{"status": "ok"}'


def _health_payload() -> bytes:
    """JSON body for ``/healthz``: exactly ``{"status": "ok"}``."""
    return _HEALTH_BODY


def _health_security_headers() -> list[tuple[bytes, bytes]]:
    """nosniff and frame headers for a response this middleware sends itself.

    Prefer the shared hardened set. If that builder fails, still stamp the
    two headers a probe response has to carry.
    """
    pairs: list[tuple[bytes, bytes]] = []
    try:
        from ..auth import _hardened_header_pairs

        pairs = list(_hardened_header_pairs())
    except Exception as exc:  # noqa: BLE001 — a probe must still answer
        logger.debug("healthz: security headers unavailable: %s", type(exc).__name__)
    present = {name.lower() for name, _ in pairs}
    if b"x-content-type-options" not in present:
        pairs.append((b"x-content-type-options", b"nosniff"))
    if b"x-frame-options" not in present:
        pairs.append((b"x-frame-options", b"SAMEORIGIN"))
    return pairs


def healthz_middleware(asgi_app: Callable) -> Callable:
    """ASGI middleware factory — early-exit handler for ``/healthz``.

    On GET ``/healthz`` with an allowed Host, returns 200 with JSON. A
    foreign Host returns 421. Every other path passes through unchanged.
    Method != GET/HEAD returns 405 with an empty body.

    Wired as the OUTERMOST wrap so the probe is unaffected by rate-limit
    or the credential check. The Host allowlist still applies.
    """

    async def middleware(scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/healthz":
            await asgi_app(scope, receive, send)
            return

        from ..auth import (
            _build_421_response,
            _header_value,
            _host_allowlist,
            _host_matches_allowlist,
        )

        host_header = _header_value(scope.get("headers") or [], b"host")
        if not _host_matches_allowlist(host_header, _host_allowlist()):
            body, headers = _build_421_response(host_header)
            await send({
                "type": "http.response.start",
                "status": 421,
                "headers": headers,
            })
            await send({"type": "http.response.body", "body": body})
            return

        method = (scope.get("method") or "GET").upper()
        if method not in ("GET", "HEAD"):
            await send({
                "type": "http.response.start",
                "status": 405,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"allow", b"GET, HEAD"),
                    (b"content-length", b"0"),
                    *_health_security_headers(),
                ],
            })
            await send({"type": "http.response.body", "body": b""})
            return

        body = _health_payload()
        headers = [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"cache-control", b"no-store"),
            *_health_security_headers(),
        ]
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": headers,
        })
        # HEAD shape: same headers, empty body.
        if method == "HEAD":
            await send({"type": "http.response.body", "body": b""})
        else:
            await send({"type": "http.response.body", "body": body})

    return middleware


__all__ = ["healthz_middleware"]
