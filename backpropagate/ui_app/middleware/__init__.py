"""ASGI middleware chain for the Reflex web UI.

Wave 6b (v1.3) introduces three sibling middlewares to ``ui_app/auth.py``:

- ``healthz_middleware``     — early-exit for the orchestrator probe (returns
  JSON outside the auth/rate-limit/CSP-hardened envelope so probes don't
  need credentials and don't get a script-bearing response).
- ``rate_limit_middleware``  — per-IP sliding-window rate limit (fast-fail
  ``429`` BEFORE auth so brute-force attempts can't exhaust HMAC budget).
- ``request_logging_middleware`` — structured request log (method, path,
  status, duration_ms, auth_mode, auth_user, remote_addr) AFTER auth so the
  log carries the resolved auth state.

v1.4 Wave 2 (FRONTEND-A-003) adds a FIFTH layer:

- ``security_headers_middleware`` — stamps the OWASP security-header set
  (Content-Security-Policy, X-Content-Type-Options, X-Frame-Options,
  Referrer-Policy, Permissions-Policy, X-XSS-Protection) on every HTTP
  response, AFTER auth (so even 401 / 403 responses ship hardened) and
  INSIDE request_logging (so the log line reflects the bytes that
  actually went on the wire). Continuation of the v1.2.0
  GHSA-f65r-h4g3-3h9h narrative — covers the browser-side defense-in-
  depth that the auth gate's credential-on-the-wire half doesn't.

These wrap ``ui_app/auth.py::basic_auth_transformer`` (the GHSA-f65r-h4g3-3h9h
load-bearing gate, UNCHANGED across Wave 6b + v1.4 Wave 2). The composed
chain wired in ``ui_app/app.py`` via ``rx.App(api_transformer=(...))`` is:

    healthz_middleware                   (outermost — /healthz early-exit)
      → rate_limit_middleware            (fast-fail 429 before auth)
        → basic_auth_transformer         (UNCHANGED — GHSA contracts intact)
          → request_logging_middleware   (structured log w/ resolved auth state)
            → security_headers_middleware  (NEW — stamps OWASP headers on
                                            responses leaving the server)
              → Reflex

Reflex 0.9.x ``api_transformer`` accepts a ``Sequence[Callable[[ASGIApp],
ASGIApp]]`` — verified via ``inspect.signature(rx.App.__init__)`` against
``reflex==0.9.2.post1``. The sequence is applied INNERMOST FIRST (the
first entry is the closest wrap around Reflex; the last entry is the
outermost / network-facing wrap). In sequence-order the 5-layer wiring
reads:

    (security_headers, request_logging, basic_auth, rate_limit, healthz)
      innermost first ──────────────────────────────────────► outermost last

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

``security_headers_middleware`` has no env knobs in v1.4 (the CSP is
tuned for Reflex compat in ``ui_security.DEFAULT_REFLEX_CSP`` and goes
straight to enforce mode). A future wave may add a
``BACKPROPAGATE_UI_CSP_REPORT_ONLY=1`` flip for deployment shakeouts.
"""

from __future__ import annotations

from .healthz import healthz_middleware
from .rate_limit import rate_limit_middleware
from .request_logging import request_logging_middleware
from .security_headers import security_headers_middleware

__all__ = [
    "healthz_middleware",
    "rate_limit_middleware",
    "request_logging_middleware",
    "security_headers_middleware",
]
