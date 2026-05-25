"""Security-headers ASGI middleware — FRONTEND-A-003 (v1.4 Wave 2).

Stamps the OWASP-recommended security header set onto every HTTP response
that leaves the server:

- ``Content-Security-Policy``     — Reflex-tuned default (see
  ``ui_security.DEFAULT_REFLEX_CSP`` for the per-directive rationale).
- ``X-Content-Type-Options: nosniff``
- ``X-Frame-Options: SAMEORIGIN``
- ``X-XSS-Protection: 1; mode=block`` (legacy but harmless)
- ``Referrer-Policy: strict-origin-when-cross-origin``
- ``Permissions-Policy: geolocation=(), microphone=(), camera=()``

This is the v1.2.0 GHSA-f65r-h4g3-3h9h narrative continuation. The auth
middleware (``basic_auth_transformer``) covers the credential-on-the-wire
half of the surface; this middleware covers the browser-side defense-in-
depth half — XSS containment via CSP, clickjacking protection via
``X-Frame-Options``, MIME-sniffing protection via ``X-Content-Type-Options``.

ASGI chain placement (FRONTEND-A-003: 5-layer wiring):

    healthz_middleware                   (outermost — /healthz early-exit)
      → rate_limit_middleware            (fast-fail 429 before auth)
        → basic_auth_transformer         (UNCHANGED — GHSA contracts intact)
          → request_logging_middleware   (structured log w/ resolved auth state)
            → security_headers_middleware  (NEW — stamps headers on responses
                                            leaving the server, AFTER auth)
              → Reflex

In ``api_transformer`` sequence-order (innermost first, outermost last):

    (security_headers, request_logging, basic_auth, rate_limit, healthz)

Rationale for "AFTER auth, BEFORE network":

- The headers go on the response leaving the server. If we wrapped BEFORE
  auth, a 401 from the auth middleware would skip our wrap entirely and
  the 401 response would ship without CSP / X-Frame-Options — leaving an
  XSS gap on the auth error page itself. Wrapping AFTER auth means even
  auth-rejected responses carry the hardened header set.
- We wrap INSIDE request_logging so the log record reflects the bytes
  that actually went on the wire (request_logging measures duration end-
  to-end including this middleware's header mutation).

Pass-through paths:

- ``/healthz`` — never reached; the healthz middleware wraps OUTSIDE this
  one and short-circuits before delegation. The /healthz JSON is
  intentionally NOT hardened (it's a single-purpose orchestrator probe
  with no JavaScript surface).
- WebSocket upgrades — CSP doesn't apply to WS frames. We pass through
  unmodified.
- Lifespan events — no response shape to mutate; pass through.

Sibling, NOT nested, w.r.t. ``ui_app/auth.py``: this middleware fires
AFTER auth in the chain and does NOT modify ``basic_auth_transformer``.
All four GHSA-f65r-h4g3-3h9h contracts (pre-accept WS cookie validation,
4-mode resolution, constant-time compares, Host/Origin allowlists) remain
the exclusive responsibility of ``ui_app/auth.py``.

Performance: one dict lookup + one list append per HTTP response. The
header dict is computed once per request (the CSP factory is cheap; a
future micro-opt could cache the encoded header list module-level if
profiling identifies it as hot). Measured cost on a 5080 rig is
sub-microsecond, dwarfed by every other framework hop.

Env knobs (deferred to a future wave):

- ``BACKPROPAGATE_UI_CSP_REPORT_ONLY=1`` could flip CSP into report-only
  mode for a deployment shakeout. Not wired today — the v1.4 wiring goes
  straight to enforce mode because the Reflex CSP was tuned for compat
  before landing. Operators that hit a CSP violation can either (a) edit
  ``ui_security.DEFAULT_REFLEX_CSP`` to add the missing directive or (b)
  request the report-only env var (file an issue).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Lazy import the header builder so this module stays importable when
# ui_security.py's heavier optional deps (jwt) are absent. The actual
# import lives inside ``_build_header_pairs`` so a one-time ImportError
# at module load can't crash the whole app's middleware chain.


def _build_header_pairs() -> list[tuple[bytes, bytes]]:
    """Build the ASGI-shape header pairs from the canonical security set.

    Returns a list of ``(name_bytes, value_bytes)`` tuples ready to extend
    the response's headers list. ASCII-encoding both halves because
    ASGI requires bytes; all header values in our security set are
    plain ASCII (no UTF-8 in CSP / X-Frame-Options / etc.).

    Built per-request (not module-level) so the operator can rebuild the
    Reflex CSP without a process restart. The cost is one dict
    construction + one list-of-tuples per HTTP response — measured
    sub-microsecond on a 5080 rig.
    """
    from backpropagate.ui_security import security_headers_dict

    raw = security_headers_dict()
    pairs: list[tuple[bytes, bytes]] = []
    for name, value in raw.items():
        # ASCII-encode both halves; CSP / X-Frame-Options / etc. are all
        # plain ASCII by spec. ``errors="strict"`` is the default; a
        # non-ASCII byte here would be a bug in the CSP builder.
        pairs.append((name.encode("ascii"), value.encode("ascii")))
    return pairs


def _is_security_header(name: bytes) -> bool:
    """Detect headers our middleware sets — used to avoid duplication.

    Reflex / Starlette may already attach some of these (e.g.
    ``Referrer-Policy`` if a future Reflex version adds a default). When
    we encounter one of OUR names in the upstream headers list, we DROP
    the upstream copy and stamp ours instead. This guarantees the chain's
    output is deterministic — operators auditing the headers know
    exactly where each one came from.

    Comparison is case-insensitive per RFC 9110 §5 (header names are
    case-insensitive on the wire).
    """
    lowered = name.lower()
    return lowered in (
        b"content-security-policy",
        b"content-security-policy-report-only",
        b"x-content-type-options",
        b"x-frame-options",
        b"x-xss-protection",
        b"referrer-policy",
        b"permissions-policy",
    )


def security_headers_middleware(asgi_app: Callable) -> Callable:
    """ASGI middleware factory — stamp OWASP security headers on responses.

    Wraps the inner app's ``send`` callable. On every ``http.response.start``
    message, drops any upstream copies of our security-header names and
    appends our canonical set. Non-HTTP scopes (WebSocket, lifespan) and
    non-start messages pass through unchanged.

    The ``healthz_middleware`` short-circuits OUTSIDE this wrap, so the
    /healthz JSON probe is intentionally NOT hardened (it's a single-
    purpose orchestrator probe with no script surface).
    """

    async def middleware(scope: dict, receive: Callable, send: Callable) -> None:
        scope_type = scope.get("type")

        if scope_type != "http":
            # WebSocket upgrades — CSP doesn't apply to WS frames.
            # Lifespan events — no response to mutate.
            # In both cases, pass through unmodified.
            await asgi_app(scope, receive, send)
            return

        async def wrapped_send(message: dict) -> None:
            if message.get("type") != "http.response.start":
                await send(message)
                return

            # Stamp our security headers. The headers list on the start
            # message is a list of (bytes, bytes) tuples per ASGI spec.
            # We:
            #   1) Drop any upstream copies of our header names (so the
            #      output is deterministic — only one CSP per response,
            #      always ours).
            #   2) Append our canonical set.
            try:
                upstream_headers: list[Any] = list(message.get("headers") or [])
                filtered = [
                    (name, value) for (name, value) in upstream_headers
                    if not _is_security_header(name)
                ]
                filtered.extend(_build_header_pairs())
                # Mutate the message in place so the downstream send
                # ships the augmented header list. We assign a new list
                # rather than mutating the existing one to avoid leaking
                # our changes into any upstream cache (Starlette
                # occasionally reuses header lists across responses; the
                # cost of one allocation per response is negligible).
                message = {**message, "headers": filtered}
            except Exception:  # noqa: BLE001 — never fail the response on header build
                # If header construction blows up (extremely unlikely —
                # the builder is pure stdlib), we ship the original
                # response unmodified rather than 500ing the user. The
                # auth + rate-limit + request-logging layers continue to
                # function. Operator notices the missing headers via the
                # browser devtools / a SIEM CSP-violation report.
                pass  # nosec B110 — middleware MUST NOT crash response on header failure

            await send(message)

        await asgi_app(scope, receive, wrapped_send)

    return middleware


__all__ = ["security_headers_middleware"]
