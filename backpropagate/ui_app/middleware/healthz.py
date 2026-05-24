"""``/healthz`` ASGI middleware — FRONTEND-5 (Wave 6b).

Lightweight orchestrator probe endpoint. Returns JSON
``{"status": "ok", "auth_mode": "...", "version": "..."}`` on GET ``/healthz``.

Why a middleware and not a Reflex page: Reflex pages render React components
and are reached through the Next.js SPA shell. An orchestrator (Kubernetes
liveness probe, AWS ELB health check, cloudflared ``--tunnel-token`` health
check) wants a plain HTTP route with a JSON body and no HTML overhead.
Wrapping the ASGI app with an early-exit on the ``/healthz`` path is the
cleanest way to add that route without coupling to Reflex's page-tree
internals.

Wired in the ASGI chain as the OUTERMOST wrap (even outside rate-limit) so
the probe is unaffected by per-IP caps and doesn't bother going through the
auth gate. The probe is intentionally unauthenticated — orchestrators that
need to know "is the process alive" should not have to carry credentials,
and the response carries no operator-sensitive data (auth_mode is the
configured posture, not a session token; version is public).

Pre-existing ``/ping`` Reflex route remains the framework-internal health
check (it's hardcoded into Reflex's SPA). ``/healthz`` is the orchestrator-
facing canonical name (matches the Kubernetes / Knative convention).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable


def _health_payload() -> bytes:
    """Build the JSON body for the /healthz response.

    Fields:
    - ``status``    — always ``"ok"`` when the process is reachable enough
                      to execute this middleware. A more detailed status
                      (e.g. "draining", "degraded") would require the
                      orchestrator to wire in per-component health, which
                      v1.3 deliberately defers.
    - ``auth_mode`` — resolved auth mode from ``ui_app.auth._detect_mode``.
                      No credentials are returned; this is just the posture
                      (``no_auth_local_only``, ``token_auto``, ...) so the
                      orchestrator can verify the process is configured the
                      way it expects.
    - ``version``   — the backpropagate package version from
                      ``backpropagate.__version__``. Useful for rolling-
                      deploy probes that need to confirm the new image is
                      actually serving traffic.
    """
    try:
        from ..auth import _detect_mode
        mode = _detect_mode(dict(os.environ))
        auth_mode = str(mode.value)
    except Exception:  # noqa: BLE001 — never fail the probe
        auth_mode = "unknown"

    try:
        from backpropagate import __version__
        version = str(__version__)
    except Exception:  # noqa: BLE001
        version = "unknown"

    payload = {
        "status": "ok",
        "auth_mode": auth_mode,
        "version": version,
    }
    return json.dumps(payload).encode("utf-8")


def healthz_middleware(asgi_app: Callable) -> Callable:
    """ASGI middleware factory — early-exit handler for ``/healthz``.

    On GET ``/healthz`` (any host, any auth), returns 200 with JSON. Every
    other path passes through unchanged. Method != GET returns 405 with an
    empty body.

    Wired as the OUTERMOST wrap so the probe is unaffected by rate-limit,
    auth, or request-logging. The probe response time is dominated by the
    JSON dump + env-var read (~10μs).
    """

    async def middleware(scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/healthz":
            await asgi_app(scope, receive, send)
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
                ],
            })
            await send({"type": "http.response.body", "body": b""})
            return

        body = _health_payload()
        headers = [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"cache-control", b"no-store"),
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
