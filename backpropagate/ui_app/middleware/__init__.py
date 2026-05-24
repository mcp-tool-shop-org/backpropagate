"""ASGI middleware chain for the Reflex web UI.

Wave 6b (v1.3) introduces two sibling middlewares to ``ui_app/auth.py``:

- ``rate_limit_middleware``  — per-IP sliding-window rate limit (fast-fail
  ``429`` BEFORE auth so brute-force attempts can't exhaust HMAC budget).
- ``request_logging_middleware`` — structured request log (method, path,
  status, duration_ms, auth_mode, auth_user, remote_addr) AFTER auth so the
  log carries the resolved auth state.

These wrap ``ui_app/auth.py::basic_auth_transformer`` (the GHSA-f65r-h4g3-3h9h
load-bearing gate, UNCHANGED in Wave 6b). The composed chain wired in
``ui_app/app.py`` via ``rx.App(api_transformer=(...))`` is:

    rate_limit_middleware
      → basic_auth_transformer            (UNCHANGED — GHSA contracts intact)
        → request_logging_middleware
          → Reflex

Reflex 0.9.x ``api_transformer`` accepts a ``Sequence[Callable[[ASGIApp],
ASGIApp]]`` — verified via ``inspect.signature(rx.App.__init__)`` against
``reflex==0.9.2.post1``. The order of the sequence is applied OUTER → INNER,
matching the natural HTTP-request reading order (rate-limit fires first on
ingress; request-logging fires last so it sees the resolved auth context).

Default-off knobs:

- ``BACKPROPAGATE_UI_REQUEST_LOG=1``         — turn request logging ON
  (default OFF; Reflex's own logging covers the basics).
- ``BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN`` — override HTTP cap
  (default 100 req/min per IP).
- ``BACKPROPAGATE_UI_RATE_LIMIT_WS_PER_MIN``   — override WS-upgrade cap
  (default 10 upgrades/min per IP).

Set the HTTP / WS knobs to ``0`` to fully DISABLE rate-limiting (smoke
tests, local-only operator use). Any positive integer is the per-minute
budget per remote IP.
"""

from __future__ import annotations

from .healthz import healthz_middleware
from .rate_limit import rate_limit_middleware
from .request_logging import request_logging_middleware

__all__ = [
    "healthz_middleware",
    "rate_limit_middleware",
    "request_logging_middleware",
]
