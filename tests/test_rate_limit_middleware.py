"""Tests for the rate-limit ASGI middleware (TESTS-A-001, v1.4 Wave 2 amend).

V1_3_BRIEF P0 item 6 shipped ``rate_limit_middleware`` at
``backpropagate/ui_app/middleware/rate_limit.py`` wired BEFORE
``basic_auth_transformer`` in the ASGI chain so brute-force probes
return 429 without consuming the HMAC budget. Wave 1 audit
(TESTS-A-001) found ZERO direct tests for the middleware — every
property documented in the module docstring was a regression surface
with no net.

Required coverage (per the Wave 1 finding's recommendation):

- Per-IP bucket isolation: two IPs share the limit independently
- Window expiry: requests after the rolling 60s window reset
- 429 response shape: status code + Retry-After header + content-type
- HTTP vs WebSocket separate limits: HTTP cap doesn't drain WS budget
- Disabled-via-env (cap=0): middleware degenerates to pass-through
- Pass-through paths (/ping, /_next/, /favicon): never rate-limited
- WebSocket 4429 close pre-accept: the middleware closes BEFORE
  the upstream app receives a websocket.connect (DESIGN_BRIEF
  anti-pattern: validation after websocket.accept() is DoS-vulnerable)
- lifespan scope: passes through unconditionally (per ASGI spec)

The tests drive the middleware directly with the ``stub_asgi_http_app``
helper (no Reflex import needed — the middleware is framework-agnostic).
Between assertions the per-IP state is reset via the
``_reset_for_tests`` hook in the module so each test starts from a
clean slate.

Cross-references:

- ``tests/test_auth_middleware.py`` — sibling middleware (auth), wired
  AFTER rate-limit in the chain. Same harness shape.
- ``backpropagate/ui_app/middleware/__init__.py`` — registers the
  middleware factory used by ``rxconfig.py``.
"""

from __future__ import annotations

import pytest

# httpx is a hard dep for ASGI testing; skip the whole module if missing.
httpx = pytest.importorskip(  # noqa: F841 — only checking importability
    "httpx",
    reason="httpx>=0.27 is required for rate-limit middleware tests. "
           "Install via the [dev] extra or `pip install httpx`.",
)

# The middleware is shipped with the package (not gated on [ui]). If it
# cannot be imported here, that is a regression and tests should fail
# loudly at collection rather than silently skip.
from backpropagate.ui_app.middleware.rate_limit import (  # noqa: E402
    _WS_CLOSE_CODE_RATE_LIMIT,
    rate_limit_middleware,
)

# =============================================================================
# HELPERS — minimal ASGI test scaffolding (no httpx client needed)
# =============================================================================


async def _stub_http_app(scope: dict, receive, send) -> None:
    """Minimal ASGI app — 200 OK for any HTTP, accept+close for WS.

    Used as the "inner" app under rate_limit_middleware. If the
    middleware passes through, the test sees a 200 (HTTP) or
    websocket.accept (WS). If the middleware rejects, the test sees
    a 429 (HTTP) or 4429 close (WS) BEFORE this stub runs.

    Mirrors ``tests/helpers/asgi.py::stub_asgi_http_app`` shape but is
    duplicated here to keep the rate-limit test file standalone
    (no cross-module import surprises if the helpers move).
    """
    if scope["type"] == "http":
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        })
        await send({"type": "http.response.body", "body": b"upstream-ok"})
    elif scope["type"] == "websocket":
        message = await receive()
        if message["type"] == "websocket.connect":
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.close", "code": 1000})


def _make_http_scope(path: str = "/", client_ip: str = "127.0.0.1") -> dict:
    """Build an ASGI HTTP scope dict for direct middleware invocation.

    The ``client`` tuple is the per-IP identity the middleware extracts
    via its ``_client_addr`` helper. Default port 12345 is arbitrary;
    only the host (first tuple element) is consumed.
    """
    return {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "client": (client_ip, 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }


def _make_ws_scope(path: str = "/_event", client_ip: str = "127.0.0.1") -> dict:
    """Build an ASGI WebSocket scope dict for direct middleware invocation."""
    return {
        "type": "websocket",
        "path": path,
        "headers": [],
        "client": (client_ip, 12345),
        "server": ("testserver", 80),
        "scheme": "ws",
    }


class _AsgiRecorder:
    """Captures the ASGI send() messages so tests can assert on them.

    Provides ``status`` (HTTP), ``headers`` (HTTP), ``body`` (HTTP),
    and ``ws_close_code`` (WS) attributes that the middleware-under-
    test populates via its single send() call on the reject path.
    """

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, message: dict) -> None:
        self.messages.append(message)

    @property
    def status(self) -> int | None:
        for m in self.messages:
            if m.get("type") == "http.response.start":
                return int(m.get("status", 0))
        return None

    @property
    def headers(self) -> dict[bytes, bytes]:
        for m in self.messages:
            if m.get("type") == "http.response.start":
                return dict(m.get("headers", []))
        return {}

    @property
    def body(self) -> bytes:
        chunks = [m.get("body", b"") for m in self.messages
                  if m.get("type") == "http.response.body"]
        return b"".join(chunks)

    @property
    def ws_close_code(self) -> int | None:
        for m in self.messages:
            if m.get("type") == "websocket.close":
                return int(m.get("code", 0))
        return None


async def _empty_receive() -> dict:  # pragma: no cover — never invoked on the reject path
    """ASGI receive callable that returns a websocket.connect message.

    Used for WS scopes where the middleware doesn't actually call
    receive() on the reject path (it closes pre-accept). Provided so
    the pass-through path can drive the stub app's `await receive()`.
    """
    return {"type": "websocket.connect"}


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    """Wipe the per-IP rate-limit state between tests.

    The middleware uses module-level _SlidingWindow singletons; without
    this reset, test order matters (a test that just exceeded the cap
    leaks state into the next test's first request). The
    ``_reset_for_tests`` hook is exposed by the module specifically
    for this purpose.
    """
    from backpropagate.ui_app.middleware import rate_limit as _rl
    _rl._reset_for_tests()
    yield
    _rl._reset_for_tests()


# =============================================================================
# DISABLED-VIA-ENV / PASS-THROUGH SMOKE TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_middleware_passthrough_when_both_caps_zero(monkeypatch):
    """Setting both HTTP+WS caps to 0 degenerates to pass-through.

    The module docstring guarantees ``cap = 0 ⇒ rate-limiting disabled``.
    A regression that started rejecting at cap=0 would break operators
    who explicitly opted out.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN", "0")
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_WS_PER_MIN", "0")

    middleware = rate_limit_middleware(_stub_http_app)
    recorder = _AsgiRecorder()

    # Drive 1000 requests from the same IP — should never trigger 429.
    for _ in range(1000):
        await middleware(_make_http_scope(), _empty_receive, recorder.send)

    # Every request should have reached the stub (200 OK), never 429.
    statuses = [m.get("status") for m in recorder.messages if m.get("type") == "http.response.start"]
    assert all(s == 200 for s in statuses), (
        f"With caps=0 the middleware must pass through; got "
        f"statuses={statuses[:10]}... (showing first 10)"
    )


@pytest.mark.asyncio
async def test_ping_path_passthrough_under_load(monkeypatch):
    """/ping is never rate-limited (orchestration health-check path).

    Mirrors auth.py's _PASSTHROUGH_PATHS. A regression that started
    rate-limiting /ping would break orchestrators that poll it
    frequently (k8s liveness probes, monitoring scripts).
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN", "5")

    middleware = rate_limit_middleware(_stub_http_app)
    recorder = _AsgiRecorder()

    # 50 /ping requests from the same IP — well over the 5/min cap.
    for _ in range(50):
        await middleware(_make_http_scope(path="/ping"), _empty_receive, recorder.send)

    statuses = [m.get("status") for m in recorder.messages if m.get("type") == "http.response.start"]
    assert all(s == 200 for s in statuses), (
        f"/ping must never be rate-limited; got statuses={statuses[:10]}..."
    )


@pytest.mark.asyncio
async def test_next_static_path_passthrough_under_load(monkeypatch):
    """``/_next/`` (Reflex SPA static assets) is never rate-limited.

    Mirrors auth.py's _PASSTHROUGH_PATHS. The SPA loads many JS/CSS
    chunks from /_next/static/... on cold start; rate-limiting these
    would break the UI in any browser that exceeds the HTTP cap during
    initial page load.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN", "3")

    middleware = rate_limit_middleware(_stub_http_app)
    recorder = _AsgiRecorder()

    for _ in range(20):
        await middleware(_make_http_scope(path="/_next/static/chunk.js"), _empty_receive, recorder.send)

    statuses = [m.get("status") for m in recorder.messages if m.get("type") == "http.response.start"]
    assert all(s == 200 for s in statuses), (
        f"/_next/ paths must never be rate-limited; got statuses={statuses[:10]}..."
    )


@pytest.mark.asyncio
async def test_favicon_path_passthrough(monkeypatch):
    """``/favicon`` paths are never rate-limited.

    Browsers auto-fetch /favicon.ico without operator action;
    rate-limiting it would create noise in the 429 logs.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN", "1")

    middleware = rate_limit_middleware(_stub_http_app)
    recorder = _AsgiRecorder()

    for _ in range(10):
        await middleware(_make_http_scope(path="/favicon.ico"), _empty_receive, recorder.send)

    statuses = [m.get("status") for m in recorder.messages if m.get("type") == "http.response.start"]
    assert all(s == 200 for s in statuses)


@pytest.mark.asyncio
async def test_lifespan_scope_passthrough(monkeypatch):
    """ASGI lifespan events bypass rate-limit entirely (per docstring).

    Lifespan messages drive server startup/shutdown; rate-limiting them
    would break the worker boot sequence.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN", "0")

    invoked = []

    async def _lifespan_app(scope, receive, send):
        invoked.append(scope.get("type"))

    middleware = rate_limit_middleware(_lifespan_app)
    await middleware(
        {"type": "lifespan"},
        _empty_receive,
        _AsgiRecorder().send,
    )
    assert invoked == ["lifespan"], (
        "Lifespan scope must reach the wrapped app unchanged."
    )


# =============================================================================
# HTTP RATE-LIMIT ENFORCEMENT
# =============================================================================


@pytest.mark.asyncio
async def test_http_cap_exceeded_returns_429(monkeypatch):
    """At cap+1 requests from the same IP, the (cap+1)th gets 429.

    The cap is inclusive: cap=N means N requests pass, the (N+1)th
    triggers 429. The middleware records each event before checking
    the count (so the cap-th request is the boundary).
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN", "3")

    middleware = rate_limit_middleware(_stub_http_app)

    # First 3 requests should pass (200).
    for i in range(3):
        recorder = _AsgiRecorder()
        await middleware(_make_http_scope(), _empty_receive, recorder.send)
        assert recorder.status == 200, (
            f"Request {i + 1} under cap should pass; got status={recorder.status}"
        )

    # 4th request should be 429.
    recorder = _AsgiRecorder()
    await middleware(_make_http_scope(), _empty_receive, recorder.send)
    assert recorder.status == 429, (
        f"Request 4 over cap=3 should be 429; got status={recorder.status}"
    )


@pytest.mark.asyncio
async def test_429_response_includes_retry_after_header(monkeypatch):
    """The 429 response includes a Retry-After header set to the window size.

    Per the module docstring, Retry-After is fixed at 60s (the rolling
    window upper bound). RFC 7231 lets it be either delta-seconds or
    HTTP-date; the middleware uses delta-seconds.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN", "1")

    middleware = rate_limit_middleware(_stub_http_app)

    # Burn the budget.
    await middleware(_make_http_scope(), _empty_receive, _AsgiRecorder().send)

    # Trigger the 429.
    recorder = _AsgiRecorder()
    await middleware(_make_http_scope(), _empty_receive, recorder.send)

    assert recorder.status == 429
    assert recorder.headers.get(b"retry-after") == b"60", (
        f"Retry-After must be set to the 60s window upper bound; "
        f"got headers={recorder.headers}"
    )
    assert recorder.headers.get(b"content-type", b"").startswith(b"text/plain"), (
        f"429 body should be plain text; got content-type="
        f"{recorder.headers.get(b'content-type')!r}"
    )
    # The body should mention the env var so the operator knows how to raise the cap.
    assert b"BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN" in recorder.body, (
        f"429 body must point to the env var for raising the cap; "
        f"got body={recorder.body!r}"
    )


@pytest.mark.asyncio
async def test_per_ip_buckets_isolated(monkeypatch):
    """Two IPs share the rate-limit cap independently.

    If IP A burns its budget, IP B's first request still succeeds.
    Without per-IP isolation, one noisy client could DoS every other
    operator behind the same UI process.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN", "2")

    middleware = rate_limit_middleware(_stub_http_app)

    # Burn IP A's full budget + push it over.
    for _ in range(3):
        await middleware(_make_http_scope(client_ip="10.0.0.1"), _empty_receive, _AsgiRecorder().send)

    # IP A's 4th request should be 429.
    recorder_a = _AsgiRecorder()
    await middleware(_make_http_scope(client_ip="10.0.0.1"), _empty_receive, recorder_a.send)
    assert recorder_a.status == 429, "IP A should be over the cap"

    # IP B's 1st request should still pass (200) — buckets are isolated.
    recorder_b = _AsgiRecorder()
    await middleware(_make_http_scope(client_ip="10.0.0.2"), _empty_receive, recorder_b.send)
    assert recorder_b.status == 200, (
        f"IP B must not inherit IP A's rate-limit state; got status={recorder_b.status}"
    )


@pytest.mark.asyncio
async def test_window_expiry_resets_bucket(monkeypatch):
    """Events older than the rolling window expire and free budget.

    The implementation prunes deque entries with timestamps older than
    ``_WINDOW_SECONDS`` (60s). Rather than wait 60s in a test, we
    drive ``time.monotonic`` directly via the _SlidingWindow internals
    — record events with synthesised old timestamps, then verify that
    a fresh event at the current monotonic time finds an empty bucket.
    """
    from backpropagate.ui_app.middleware import rate_limit as _rl

    cap = 5
    window = _rl._SlidingWindow()
    # Inject old events at synthesised timestamps that are
    # WAY outside the rolling window.
    import time
    now = time.monotonic()
    very_old = now - 1_000.0  # 1000s ago, well outside the 60s window

    # Pre-load the bucket with old events.
    for _ in range(cap + 2):
        # This is over the cap but the pruning step on the next call
        # should drop all of them because they're older than the window.
        window.record_and_check("10.0.0.5", very_old, cap)

    # Record one fresh event at the current monotonic time.
    over_after_prune = window.record_and_check("10.0.0.5", now, cap)
    assert over_after_prune is False, (
        "After all events expire from the rolling window, a fresh event "
        "should find an essentially-empty bucket and NOT be over the cap."
    )


@pytest.mark.asyncio
async def test_empty_client_ip_never_rate_limited(monkeypatch):
    """ASGI scope with no client tuple gets a pass.

    Per the module docstring: "we never reject a request because we
    couldn't identify the client — that's an ASGI-transport oddity,
    not abuse." Unix-socket transports and some test harnesses set
    scope['client']=None.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN", "1")

    middleware = rate_limit_middleware(_stub_http_app)

    # Drive many requests with no client identity.
    for _ in range(50):
        scope = _make_http_scope()
        scope["client"] = None  # ASGI transport without per-conn IP
        recorder = _AsgiRecorder()
        await middleware(scope, _empty_receive, recorder.send)
        assert recorder.status == 200, (
            f"Empty-client-IP requests must pass-through; got status={recorder.status}"
        )


# =============================================================================
# WEBSOCKET RATE-LIMIT ENFORCEMENT
# =============================================================================


@pytest.mark.asyncio
async def test_ws_cap_separate_from_http_cap(monkeypatch):
    """Burning the HTTP cap does NOT consume WS budget (or vice versa).

    HTTP and WebSocket use separate _SlidingWindow singletons; an
    operator who's hitting the UI hard via HTTP polling should still
    be able to upgrade to WebSocket for the live-train view.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN", "1")
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_WS_PER_MIN", "5")

    middleware = rate_limit_middleware(_stub_http_app)
    ip = "10.0.0.7"

    # Burn the HTTP budget.
    await middleware(_make_http_scope(client_ip=ip), _empty_receive, _AsgiRecorder().send)
    rec = _AsgiRecorder()
    await middleware(_make_http_scope(client_ip=ip), _empty_receive, rec.send)
    assert rec.status == 429, "HTTP cap should be over"

    # WS upgrade from the same IP should still succeed (separate bucket).
    ws_rec = _AsgiRecorder()
    await middleware(_make_ws_scope(client_ip=ip), _empty_receive, ws_rec.send)
    # The stub accepts the WS, so we should see websocket.accept,
    # NOT a 4429 pre-accept close.
    assert ws_rec.ws_close_code != _WS_CLOSE_CODE_RATE_LIMIT, (
        f"WS budget should be independent of HTTP budget; "
        f"got ws_close_code={ws_rec.ws_close_code}"
    )


@pytest.mark.asyncio
async def test_ws_cap_exceeded_closes_pre_accept_with_4429(monkeypatch):
    """At cap+1 WebSocket upgrades, the (cap+1)th gets close-code 4429.

    DESIGN_BRIEF anti-pattern: validation AFTER websocket.accept() is
    DoS-vulnerable. The middleware MUST close pre-accept on the
    reject path. The 4429 close code reads as "HTTP 429 equivalent
    over WS" and is in the RFC 6455 4000-4999 application range.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_RATE_LIMIT_WS_PER_MIN", "2")

    middleware = rate_limit_middleware(_stub_http_app)
    ip = "10.0.0.8"

    # First 2 WS upgrades from this IP should pass (cap=2).
    for _ in range(2):
        await middleware(_make_ws_scope(client_ip=ip), _empty_receive, _AsgiRecorder().send)

    # 3rd WS upgrade should be closed with 4429 BEFORE any accept message.
    recorder = _AsgiRecorder()
    await middleware(_make_ws_scope(client_ip=ip), _empty_receive, recorder.send)

    assert recorder.ws_close_code == _WS_CLOSE_CODE_RATE_LIMIT, (
        f"WS over cap=2 should close with code {_WS_CLOSE_CODE_RATE_LIMIT}; "
        f"got code={recorder.ws_close_code}, messages={recorder.messages}"
    )
    # Most-load-bearing check: the ONLY message sent should be
    # websocket.close — no websocket.accept first.
    types = [m.get("type") for m in recorder.messages]
    assert "websocket.accept" not in types, (
        f"Rate-limit reject must close PRE-accept (DESIGN_BRIEF anti-pattern: "
        f"validation after accept). Got message types={types}"
    )


# =============================================================================
# ENV-VAR PARSING
# =============================================================================


def test_resolve_cap_handles_non_integer(monkeypatch):
    """A non-integer env var falls back to the default cap.

    Operators sometimes set ``BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN=high``
    via copy-paste error. The middleware must not crash — fall back
    to the documented default.
    """
    from backpropagate.ui_app.middleware.rate_limit import _resolve_cap

    # Garbage value → default (100).
    cap = _resolve_cap(
        "BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN",
        default=100,
        env={"BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN": "twelve"},
    )
    assert cap == 100


def test_resolve_cap_negative_value_uses_default():
    """A negative env-var value is treated as default (not as "disabled").

    The disabled-via-env contract is ``cap=0`` specifically — negatives
    are operator typos / sign-confusion, not intent.
    """
    from backpropagate.ui_app.middleware.rate_limit import _resolve_cap

    cap = _resolve_cap(
        "BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN",
        default=100,
        env={"BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN": "-5"},
    )
    assert cap == 100


def test_resolve_cap_zero_disables():
    """A literal ``0`` env value is honored as "disable for this kind"."""
    from backpropagate.ui_app.middleware.rate_limit import _resolve_cap

    cap = _resolve_cap(
        "BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN",
        default=100,
        env={"BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN": "0"},
    )
    assert cap == 0


def test_resolve_cap_empty_value_uses_default():
    """Empty / unset env values fall back to the documented default."""
    from backpropagate.ui_app.middleware.rate_limit import _resolve_cap

    cap = _resolve_cap(
        "BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN",
        default=42,
        env={},
    )
    assert cap == 42

    cap_empty = _resolve_cap(
        "BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN",
        default=42,
        env={"BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN": ""},
    )
    assert cap_empty == 42


# =============================================================================
# SLIDING-WINDOW INTERNALS
# =============================================================================


def test_sliding_window_record_and_check_below_cap():
    """``record_and_check`` returns False when the bucket is under the cap."""
    import time

    from backpropagate.ui_app.middleware.rate_limit import _SlidingWindow

    w = _SlidingWindow()
    now = time.monotonic()
    for _ in range(4):
        over = w.record_and_check("10.0.0.10", now, cap=5)
        assert over is False


def test_sliding_window_record_and_check_at_cap():
    """``record_and_check`` returns True ONLY when strictly over the cap."""
    import time

    from backpropagate.ui_app.middleware.rate_limit import _SlidingWindow

    w = _SlidingWindow()
    now = time.monotonic()
    # First 5 calls should all return False (under or at cap=5).
    for _ in range(5):
        over = w.record_and_check("10.0.0.11", now, cap=5)
        assert over is False, "At-cap should not be over-cap"

    # The 6th call pushes the bucket over.
    assert w.record_and_check("10.0.0.11", now, cap=5) is True


def test_sliding_window_cap_zero_disables_check():
    """``cap=0`` short-circuits — never returns over-cap, never records."""
    import time

    from backpropagate.ui_app.middleware.rate_limit import _SlidingWindow

    w = _SlidingWindow()
    now = time.monotonic()
    for _ in range(1000):
        over = w.record_and_check("10.0.0.12", now, cap=0)
        assert over is False
    # No state should have been recorded.
    assert "10.0.0.12" not in w._events


def test_sliding_window_empty_ip_returns_false():
    """Empty IP string returns False without recording state."""
    import time

    from backpropagate.ui_app.middleware.rate_limit import _SlidingWindow

    w = _SlidingWindow()
    now = time.monotonic()
    assert w.record_and_check("", now, cap=5) is False
    assert "" not in w._events


def test_sliding_window_prune_idle_ips_removes_stale_entries():
    """``prune_idle_ips`` drops IPs whose latest event is outside the window."""
    import time

    from backpropagate.ui_app.middleware.rate_limit import _SlidingWindow

    w = _SlidingWindow()
    now = time.monotonic()

    # IP A: recent event (should be kept).
    w.record_and_check("10.0.0.20", now, cap=10)
    # IP B: old event (should be pruned).
    w.record_and_check("10.0.0.21", now - 1_000.0, cap=10)

    assert "10.0.0.20" in w._events
    assert "10.0.0.21" in w._events

    w.prune_idle_ips(now)

    assert "10.0.0.20" in w._events
    assert "10.0.0.21" not in w._events, (
        "Pruning must drop IPs whose latest event is older than the window."
    )


# =============================================================================
# CLIENT-ADDR EXTRACTION
# =============================================================================


def test_client_addr_extracts_host_from_tuple():
    """``_client_addr`` returns the host string from a (host, port) tuple."""
    from backpropagate.ui_app.middleware.rate_limit import _client_addr

    assert _client_addr({"client": ("10.20.30.40", 12345)}) == "10.20.30.40"


def test_client_addr_empty_for_none_client():
    """``_client_addr`` returns empty string when scope['client'] is None."""
    from backpropagate.ui_app.middleware.rate_limit import _client_addr

    assert _client_addr({"client": None}) == ""
    assert _client_addr({}) == ""


def test_client_addr_empty_for_malformed_tuple():
    """``_client_addr`` returns empty string for malformed client values."""
    from backpropagate.ui_app.middleware.rate_limit import _client_addr

    assert _client_addr({"client": ()}) == ""
    assert _client_addr({"client": "not-a-tuple"}) == ""


# =============================================================================
# 429 RESPONSE BUILDER
# =============================================================================


def test_build_429_response_returns_well_formed_envelope():
    """The 429 body + headers tuple is well-formed for ASGI send.

    Headers must be a list of (bytes, bytes) tuples per ASGI spec.
    Body is bytes. Status code is checked at the use site, not here.
    """
    from backpropagate.ui_app.middleware.rate_limit import _build_429_response

    body, headers = _build_429_response()

    assert isinstance(body, bytes)
    assert isinstance(headers, list)
    assert all(isinstance(h, tuple) and len(h) == 2 for h in headers)
    assert all(isinstance(k, bytes) and isinstance(v, bytes) for k, v in headers)

    header_map = dict(headers)
    assert header_map.get(b"retry-after") == b"60"
    assert header_map.get(b"content-type", b"").startswith(b"text/plain")
    assert header_map.get(b"cache-control") == b"no-store"
    # content-length must match the body bytes length.
    assert int(header_map[b"content-length"]) == len(body)
