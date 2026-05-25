"""Tests for the security-headers ASGI middleware (FRONTEND-A-003 + TESTS-A-001).

FRONTEND-A-003 (v1.4 Wave 2) shipped ``security_headers_middleware`` at
``backpropagate/ui_app/middleware/security_headers.py``. The middleware
stamps the OWASP-recommended security header set onto every HTTP response
that leaves the server, wired AFTER auth (so auth-rejected 401 responses
also carry the hardened headers) but BEFORE the network.

This file mirrors the v1.3 ``test_rate_limit_middleware.py`` /
``test_request_logging_middleware.py`` shape:

- Direct ASGI invocation (no Reflex import needed — middleware is
  framework-agnostic).
- Stub send-recorder captures the headers list on http.response.start.
- Coverage per the FRONTEND-A-003 follow-up note in WAVE_6A_TODO.md:

  (a) headers appear on a vanilla 200 response
  (b) headers appear on a 401 auth-rejected response (proves AFTER-auth wiring)
  (c) _is_security_header drops upstream duplicates (deterministic chain output)
  (d) WS upgrade passes through unmodified (CSP doesn't apply to WS)
  (e) lifespan messages pass through unmodified (no response to mutate)
  (f) the canonical header set is present (CSP / X-Content-Type-Options /
      X-Frame-Options / X-XSS-Protection / Referrer-Policy / Permissions-Policy)
  (g) defensive: a broken ui_security.security_headers_dict() does NOT crash
      the response — the upstream response ships unmodified.

Cross-references:

- ``backpropagate/ui_app/middleware/security_headers.py`` — the SUT.
- ``backpropagate/ui_security.py::security_headers_dict`` — the canonical
  header-set builder (the v1.4 Reflex CSP + 5 sibling headers).
"""

from __future__ import annotations

import pytest

from backpropagate.ui_app.middleware.security_headers import (
    _build_header_pairs,
    _is_security_header,
    security_headers_middleware,
)

# =============================================================================
# HELPERS
# =============================================================================


async def _stub_http_200_app(scope, receive, send) -> None:
    """Minimal ASGI app — sends a 200 OK back via the wrapped send."""
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [(b"content-type", b"text/plain")],
    })
    await send({"type": "http.response.body", "body": b"ok"})


async def _stub_http_401_app(scope, receive, send) -> None:
    """Minimal ASGI app — sends a 401 (proves AFTER-auth wiring)."""
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [(b"www-authenticate", b'Basic realm="backpropagate"')],
    })
    await send({"type": "http.response.body", "body": b"unauthorized"})


async def _stub_ws_app(scope, receive, send) -> None:
    """Minimal ASGI app for WebSocket scopes — accept + close."""
    message = await receive()
    if message["type"] == "websocket.connect":
        await send({"type": "websocket.accept"})
        await send({"type": "websocket.close", "code": 1000})


async def _empty_receive():
    return {"type": "websocket.connect"}


def _make_http_scope(path: str = "/", method: str = "GET") -> dict:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }


def _make_ws_scope(path: str = "/_event") -> dict:
    return {
        "type": "websocket",
        "path": path,
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "ws",
    }


class _Recorder:
    """Captures all ASGI send() messages."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, message: dict) -> None:
        self.messages.append(message)

    @property
    def headers(self) -> dict[bytes, bytes]:
        """Return the header dict from http.response.start (case-folded names).

        ASGI / RFC 9110 §5 header names are case-insensitive on the wire.
        We fold to lowercase here so tests can assert on canonical keys
        regardless of whether the upstream / middleware emitted Title-Case
        (e.g. ``Content-Security-Policy``) or lowercase
        (``content-security-policy``).
        """
        for m in self.messages:
            if m.get("type") == "http.response.start":
                return {name.lower(): value for name, value in m.get("headers", [])}
        return {}

    @property
    def status(self) -> int | None:
        for m in self.messages:
            if m.get("type") == "http.response.start":
                return int(m.get("status", 0))
        return None


# =============================================================================
# (a) HEADERS ON VANILLA 200
# =============================================================================


@pytest.mark.asyncio
async def test_security_headers_present_on_200_response():
    """A vanilla 200 response carries the security-header set."""
    middleware = security_headers_middleware(_stub_http_200_app)
    recorder = _Recorder()
    await middleware(_make_http_scope(), _empty_receive, recorder.send)

    assert recorder.status == 200
    # Each canonical header must be present (case-insensitive — but
    # the middleware emits lowercase ASCII per ASGI convention).
    headers = recorder.headers
    assert b"content-security-policy" in headers, (
        f"CSP header missing from response; got headers={list(headers.keys())}"
    )
    assert b"x-content-type-options" in headers
    assert b"x-frame-options" in headers
    assert b"referrer-policy" in headers
    assert b"permissions-policy" in headers
    # The upstream content-type header should still be present (we only
    # filter our OWN names, not unrelated headers).
    assert b"content-type" in headers


# =============================================================================
# (b) HEADERS ON 401 (AFTER-AUTH WIRING)
# =============================================================================


@pytest.mark.asyncio
async def test_security_headers_present_on_401_response():
    """A 401 from upstream auth still carries security headers.

    The FRONTEND-A-003 rationale: if we wrapped BEFORE auth, the 401
    would skip our wrap entirely and ship without CSP / X-Frame-Options
    — leaving an XSS gap on the auth error page itself. This test pins
    that the headers stamp on EVERY HTTP response, regardless of status.
    """
    middleware = security_headers_middleware(_stub_http_401_app)
    recorder = _Recorder()
    await middleware(_make_http_scope(), _empty_receive, recorder.send)

    assert recorder.status == 401
    headers = recorder.headers
    assert b"content-security-policy" in headers, (
        "401 response must carry CSP — see security_headers.py FRONTEND-A-003 rationale"
    )
    assert b"x-frame-options" in headers
    # The upstream www-authenticate header should still be present.
    assert b"www-authenticate" in headers


# =============================================================================
# (c) DROPS UPSTREAM DUPLICATES — DETERMINISTIC CHAIN OUTPUT
# =============================================================================


@pytest.mark.asyncio
async def test_upstream_duplicate_csp_is_dropped():
    """If upstream attaches its own CSP, we drop it and stamp ours.

    Reflex / Starlette may set CSP independently. The middleware
    guarantees deterministic chain output: one CSP per response, and
    it's always the one our middleware emits.
    """
    async def _app_with_upstream_csp(scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/html"),
                (b"content-security-policy", b"default-src 'self'"),  # upstream
            ],
        })
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = security_headers_middleware(_app_with_upstream_csp)
    recorder = _Recorder()
    await middleware(_make_http_scope(), _empty_receive, recorder.send)

    # Count CSP occurrences — there should be EXACTLY one (ours).
    headers_list = []
    for m in recorder.messages:
        if m.get("type") == "http.response.start":
            headers_list = m.get("headers", [])
            break

    csp_count = sum(
        1 for name, _v in headers_list
        if name.lower() == b"content-security-policy"
    )
    assert csp_count == 1, (
        f"Expected exactly one CSP header (the chain's deterministic shape); "
        f"got {csp_count}. Headers: {headers_list}"
    )

    # The CSP value should NOT be the upstream literal — it should be ours.
    csp_value = next(
        v for name, v in headers_list if name.lower() == b"content-security-policy"
    )
    assert csp_value != b"default-src 'self'", (
        f"Upstream CSP was not dropped; got CSP={csp_value!r}"
    )


@pytest.mark.asyncio
async def test_upstream_duplicate_x_frame_options_dropped():
    """Upstream X-Frame-Options is also filtered before our stamp."""
    async def _app_with_upstream_xfo(scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"x-frame-options", b"DENY"),  # upstream
            ],
        })
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = security_headers_middleware(_app_with_upstream_xfo)
    recorder = _Recorder()
    await middleware(_make_http_scope(), _empty_receive, recorder.send)

    headers_list = []
    for m in recorder.messages:
        if m.get("type") == "http.response.start":
            headers_list = m.get("headers", [])
            break

    xfo_count = sum(
        1 for name, _v in headers_list
        if name.lower() == b"x-frame-options"
    )
    assert xfo_count == 1, (
        f"Expected exactly one X-Frame-Options header; got {xfo_count}"
    )


# =============================================================================
# (d) WEBSOCKET PASS-THROUGH (CSP DOESN'T APPLY TO WS FRAMES)
# =============================================================================


@pytest.mark.asyncio
async def test_websocket_upgrade_passes_through_unmodified():
    """WS upgrades skip header stamping (CSP doesn't apply to WS frames).

    The middleware short-circuits non-HTTP scopes; the only assertion
    we can make about WS is that the upstream messages flow through
    unchanged (the accept message is not mutated, no extra messages
    are injected).
    """
    middleware = security_headers_middleware(_stub_ws_app)
    recorder = _Recorder()
    await middleware(_make_ws_scope(), _empty_receive, recorder.send)

    # The stub app sent accept + close; we should see those unchanged.
    types = [m.get("type") for m in recorder.messages]
    assert "websocket.accept" in types
    assert "websocket.close" in types
    # No HTTP messages were injected.
    assert "http.response.start" not in types
    assert "http.response.body" not in types


# =============================================================================
# (e) LIFESPAN PASS-THROUGH
# =============================================================================


@pytest.mark.asyncio
async def test_lifespan_scope_passes_through_unmodified():
    """Lifespan events have no response to mutate; bypass the wrap."""
    invoked = []

    async def _lifespan_app(scope, receive, send):
        invoked.append(scope.get("type"))

    middleware = security_headers_middleware(_lifespan_app)
    await middleware({"type": "lifespan"}, _empty_receive, _Recorder().send)
    assert invoked == ["lifespan"]


# =============================================================================
# (f) CANONICAL HEADER SET
# =============================================================================


def test_build_header_pairs_returns_ascii_bytes_tuples():
    """``_build_header_pairs`` returns the ASGI-shape (bytes, bytes) tuples."""
    pairs = _build_header_pairs()
    assert isinstance(pairs, list)
    assert len(pairs) > 0
    for entry in pairs:
        assert isinstance(entry, tuple)
        assert len(entry) == 2
        name, value = entry
        assert isinstance(name, bytes), f"Header name must be bytes; got {type(name).__name__}"
        assert isinstance(value, bytes), f"Header value must be bytes; got {type(value).__name__}"
        # ASCII round-trip (the encoder must succeed).
        name.decode("ascii")
        value.decode("ascii")


def test_build_header_pairs_includes_owasp_set():
    """The built header pairs include the v1.4 OWASP-recommended set."""
    pairs = _build_header_pairs()
    names = {name.lower() for name, _v in pairs}

    required = {
        b"content-security-policy",
        b"x-content-type-options",
        b"x-frame-options",
        b"referrer-policy",
        b"permissions-policy",
    }
    missing = required - names
    assert not missing, (
        f"Built header set missing OWASP-recommended headers: {missing}"
    )


# =============================================================================
# _is_security_header — CASE-INSENSITIVE MATCH
# =============================================================================


def test_is_security_header_matches_case_insensitive():
    """``_is_security_header`` is case-insensitive (RFC 9110 §5)."""
    # Lowercase
    assert _is_security_header(b"content-security-policy") is True
    # Title case
    assert _is_security_header(b"Content-Security-Policy") is True
    # Upper case
    assert _is_security_header(b"X-FRAME-OPTIONS") is True
    # Report-Only variant
    assert _is_security_header(b"content-security-policy-report-only") is True


def test_is_security_header_rejects_unrelated_names():
    """Non-security headers return False."""
    assert _is_security_header(b"content-type") is False
    assert _is_security_header(b"www-authenticate") is False
    assert _is_security_header(b"cache-control") is False


# =============================================================================
# (g) DEFENSIVE — BROKEN BUILDER MUST NOT CRASH RESPONSE
# =============================================================================


@pytest.mark.asyncio
async def test_broken_header_builder_does_not_crash_response(monkeypatch):
    """If header construction raises, ship the upstream response unmodified.

    The middleware wraps header construction in try/except so a bug in
    ``ui_security.security_headers_dict`` cannot 5xx every response. The
    operator notices the missing headers via browser devtools / CSP
    violation reports rather than seeing the whole UI go dark.
    """
    def _broken_builder():
        raise RuntimeError("ui_security builder broken — fault-injected")

    monkeypatch.setattr(
        "backpropagate.ui_app.middleware.security_headers._build_header_pairs",
        _broken_builder,
    )

    middleware = security_headers_middleware(_stub_http_200_app)
    recorder = _Recorder()

    # Must not raise.
    await middleware(_make_http_scope(), _empty_receive, recorder.send)

    # The upstream 200 + content-type still made it through.
    assert recorder.status == 200
    headers = recorder.headers
    assert b"content-type" in headers
    # Our security headers may or may not be present (depending on
    # whether the exception fired before or after the in-place dict
    # update); the load-bearing assertion is that the response shipped.
