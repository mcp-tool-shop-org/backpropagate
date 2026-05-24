"""Rate-limit ASGI middleware — V1_3_BRIEF P0 item 6.

Per-IP sliding-window rate limit. Wired BEFORE ``basic_auth_transformer`` in
the ASGI chain so brute-force attempts can't exhaust the HMAC budget — the
429 fires before any auth work begins.

Defaults (per remote IP):

- HTTP requests: 100 req/min
- WebSocket upgrades: 10 upgrades/min  (lower because each accepted WS holds
  a backend connection until cookie expires; cheap to flood without a cap)

Override via env vars:

- ``BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN``  — integer; 0 disables
- ``BACKPROPAGATE_UI_RATE_LIMIT_WS_PER_MIN``    — integer; 0 disables

Implementation: stdlib-only sliding window over ``collections.deque`` of
event timestamps per IP. Old entries are pruned lazily on each check so the
memory footprint is bounded by ``unique_ips * max(http_cap, ws_cap)``. For
a typical loopback-only operator this is single-digit IPs * 100 ints = a
few KB. The data structure is process-local (single-process Reflex);
multi-process deployments behind a load balancer would need a Redis-backed
limiter (out of scope for v1.3 — operators deploying that way are expected
to terminate rate-limiting at the LB).

Thread safety: ``threading.Lock`` protects the per-IP state dict. ASGI is
single-event-loop in CPython, so the lock contention is essentially zero;
the lock exists so that if a future deployment introduces a thread pool the
counter doesn't race.

Sibling, NOT nested, w.r.t. ``ui_app/auth.py``: this middleware fires BEFORE
auth in the chain and does not modify ``basic_auth_transformer``. The two
remain inspectable / testable / version-bumpable independently.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from http import HTTPStatus
from typing import Any

logger = logging.getLogger(__name__)

# Defaults from V1_3_BRIEF — 100 HTTP/min, 10 WS/min per IP. The window is
# rolling 60s; clamp the deque growth to avoid pathological memory use under
# a flood from a single IP (we drop the oldest entries past the cap because
# they're already older than the window).
_DEFAULT_HTTP_PER_MIN = 100
_DEFAULT_WS_PER_MIN = 10
_WINDOW_SECONDS = 60.0

_HTTP_ENV = "BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN"
_WS_ENV = "BACKPROPAGATE_UI_RATE_LIMIT_WS_PER_MIN"

# The WS close code for rate-limit rejections. RFC 6455 reserves 4000-4999
# for application use; 4429 reads as "HTTP 429 equivalent over WS" and is
# unused by Reflex and by ``ui_app/auth.py``'s 4401/4403/4404 codes.
_WS_CLOSE_CODE_RATE_LIMIT = 4429


def _resolve_cap(env_var: str, default: int, env: dict[str, str]) -> int:
    """Read an integer cap from env; fall back to default on parse error.

    Negative values and non-integers are treated as the default. Value ``0``
    is honored as "disable rate-limiting for this kind" — the middleware
    skips the bookkeeping path entirely in that case.
    """
    raw = env.get(env_var, "").strip()
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        logger.warning(
            "rate_limit: %s=%r is not an integer; falling back to default %d",
            env_var, raw, default,
        )
        return default
    if n < 0:
        return default
    return n


class _SlidingWindow:
    """Per-IP sliding-window event log.

    Each call to ``record_and_check(now, cap)`` appends ``now`` to the per-IP
    deque, prunes entries older than ``_WINDOW_SECONDS``, and returns ``True``
    if the IP is now over the cap. ``cap = 0`` disables the check (fast path).
    """

    __slots__ = ("_events", "_lock")

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def record_and_check(self, ip: str, now: float, cap: int) -> bool:
        """True if the IP is OVER the cap after this event.

        ``cap=0`` returns False immediately (rate-limiting disabled). Empty
        IP returns False (we never reject a request because we couldn't
        identify the client — that's an ASGI-transport oddity, not abuse).
        """
        if cap <= 0 or not ip:
            return False
        cutoff = now - _WINDOW_SECONDS
        with self._lock:
            q = self._events.get(ip)
            if q is None:
                q = deque()
                self._events[ip] = q
            # Prune entries older than the window. ``popleft`` is O(1).
            while q and q[0] < cutoff:
                q.popleft()
            # Append the current event AFTER pruning so the cap check sees
            # the entry that triggered it.
            q.append(now)
            # Trim from the left if the deque grew past 2*cap (pathological
            # flood from a single IP — we don't need to keep more than the
            # cap-sized window of recent events to make the decision). This
            # bounds the per-IP memory.
            while len(q) > max(cap * 2, 16):
                q.popleft()
            return len(q) > cap

    def prune_idle_ips(self, now: float) -> None:
        """Drop IPs with no events inside the window.

        Called opportunistically on a low-frequency interval (every 100th
        event) so the dict doesn't grow without bound on long-running
        processes. The pruning walks every entry; with single-digit unique
        IPs (typical operator load) it's microseconds.
        """
        cutoff = now - _WINDOW_SECONDS
        with self._lock:
            stale = [ip for ip, q in self._events.items() if not q or q[-1] < cutoff]
            for ip in stale:
                del self._events[ip]


# Module-level singletons. Re-initialised on hot-reload in development; in
# production they live for the lifetime of the Reflex worker process.
_HTTP_WINDOW = _SlidingWindow()
_WS_WINDOW = _SlidingWindow()

# Counter used to throttle the prune-idle-ips sweep (every 100 events).
_PRUNE_INTERVAL = 100
_event_count: int = 0
_event_count_lock = threading.Lock()


def _maybe_prune(now: float) -> None:
    """Trigger an idle-IP sweep every Nth event."""
    global _event_count
    with _event_count_lock:
        _event_count += 1
        should_prune = _event_count % _PRUNE_INTERVAL == 0
    if should_prune:
        _HTTP_WINDOW.prune_idle_ips(now)
        _WS_WINDOW.prune_idle_ips(now)


def _client_addr(scope: dict) -> str:
    """Extract client IP from ASGI scope (mirror of request_logging helper).

    Duplicated to avoid a cross-module import dependency on a sibling
    middleware; both helpers are 6 lines and the duplication is intentional.
    """
    client = scope.get("client")
    if not client or not isinstance(client, (tuple, list)) or len(client) < 1:
        return ""
    try:
        return str(client[0])
    except (TypeError, ValueError):
        return ""


def _build_429_response() -> tuple[bytes, list[tuple[bytes, bytes]]]:
    """Build the 429 Too Many Requests body + headers.

    The body is operator-facing (informational). The ``Retry-After`` header
    is fixed at the rolling-window size (60s) — the actual time-to-availability
    depends on which event will fall out of the window first, but 60s is the
    safe upper bound that always works.
    """
    body = (
        b"429 Too Many Requests: rate limit exceeded for your IP. "
        b"Retry after the rolling 60-second window expires. "
        b"This limit protects the UI's auth surface from brute-force probes; "
        b"if your legitimate traffic exceeds it, set "
        b"BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN to raise it."
    )
    headers = [
        (b"content-type", b"text/plain; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"retry-after", str(int(_WINDOW_SECONDS)).encode("ascii")),
        (b"cache-control", b"no-store"),
    ]
    return body, headers


def rate_limit_middleware(asgi_app: Callable) -> Callable:
    """ASGI middleware factory — per-IP rate-limit gate.

    The middleware fires FIRST in the chain (before auth) so brute-force
    attempts never reach the HMAC verifier. On rate-limit-exceeded:

    - HTTP: respond 429 with ``Retry-After: 60`` + operator-facing body.
    - WebSocket: send ``websocket.close`` with code 4429 BEFORE accept.

    Pass-through paths (matches ``auth.py::_PASSTHROUGH_PATHS``):

    - ``/ping`` (orchestration health check — never rate-limited)
    - ``/_next/`` (Reflex SPA static assets — heavy but trusted)
    - ``/favicon`` (browser auto-fetched, never adversarial)

    When the caps are set to 0 via env, the middleware degenerates to a pass-
    through with a single integer-comparison cost per request.
    """

    async def middleware(scope: dict, receive: Callable, send: Callable) -> None:
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await asgi_app(scope, receive, send)
            return

        env = dict(os.environ)
        http_cap = _resolve_cap(_HTTP_ENV, _DEFAULT_HTTP_PER_MIN, env)
        ws_cap = _resolve_cap(_WS_ENV, _DEFAULT_WS_PER_MIN, env)

        # Both caps disabled? Skip entirely.
        if http_cap == 0 and ws_cap == 0:
            await asgi_app(scope, receive, send)
            return

        # Pass-through paths — never rate-limited. These mirror auth.py's
        # _PASSTHROUGH_PATHS so orchestration / SPA assets are unaffected.
        path = str(scope.get("path") or "")
        if path == "/ping" or path.startswith("/_next/") or path.startswith("/favicon"):
            await asgi_app(scope, receive, send)
            return

        ip = _client_addr(scope)
        now = time.monotonic()

        if scope_type == "http":
            over = _HTTP_WINDOW.record_and_check(ip, now, http_cap)
            if over:
                logger.warning(
                    "rate_limit: HTTP cap exceeded for %s (cap=%d/min, path=%s)",
                    ip or "<unknown>", http_cap, path,
                )
                body, h429 = _build_429_response()
                await send({
                    "type": "http.response.start",
                    "status": int(HTTPStatus.TOO_MANY_REQUESTS),
                    "headers": h429,
                })
                await send({"type": "http.response.body", "body": body})
                _maybe_prune(now)
                return
        elif scope_type == "websocket":
            over = _WS_WINDOW.record_and_check(ip, now, ws_cap)
            if over:
                logger.warning(
                    "rate_limit: WS cap exceeded for %s (cap=%d/min, path=%s)",
                    ip or "<unknown>", ws_cap, path,
                )
                # Pre-accept close — same shape as ui_app/auth.py's
                # 4401/4403/4404 rejections. Mirrors the DESIGN_BRIEF
                # anti-pattern doctrine: "Validation AFTER websocket.accept()
                # — DoS via thousands of accepted-then-rejected connections."
                await send({
                    "type": "websocket.close",
                    "code": _WS_CLOSE_CODE_RATE_LIMIT,
                    "reason": "rate_limit_exceeded",
                })
                _maybe_prune(now)
                return

        _maybe_prune(now)
        await asgi_app(scope, receive, send)

    return middleware


def _reset_for_tests() -> None:
    """Clear all per-IP state — for unit-test isolation only.

    Not exported via ``__all__``. Tests reach in via the module name when
    they need to start from a clean slate between assertions.
    """
    global _event_count
    with _event_count_lock:
        _event_count = 0
    with _HTTP_WINDOW._lock, _WS_WINDOW._lock:
        _HTTP_WINDOW._events.clear()
        _WS_WINDOW._events.clear()


__all__ = [
    "rate_limit_middleware",
    "_WS_CLOSE_CODE_RATE_LIMIT",
]


def _structured_log_hook(record: dict[str, Any]) -> None:  # pragma: no cover
    """Reserved hook for future structlog integration.

    Today rate-limit decisions emit via stdlib logger.warning so they integrate
    with whatever the operator has configured at the Python logging root.
    The request-logging middleware (separate file) handles per-request
    structured logging; rate-limit emits ONLY on rejection (sparse).
    """
    _ = record
