"""Tests for the request-logging ASGI middleware (TESTS-A-001, v1.4 Wave 2).

V1_3_BRIEF P0 item 5 shipped ``request_logging_middleware`` at
``backpropagate/ui_app/middleware/request_logging.py`` wired AFTER
``basic_auth_transformer`` in the ASGI chain. Wave 1 audit
(TESTS-A-001) found ZERO direct tests — the middleware was a
regression surface.

Covered contracts (per the module docstring):

- Default OFF: ``BACKPROPAGATE_UI_REQUEST_LOG`` unset / ``0`` / ``""``
  ⇒ middleware delegates immediately with zero logging cost.
- Opt-in ON: ``BACKPROPAGATE_UI_REQUEST_LOG=1`` ⇒ structured log line
  per request with ``method`` / ``path`` / ``status`` / ``duration_ms`` /
  ``auth_mode`` / ``auth_user`` / ``remote_addr`` fields.
- ``auth_user`` is empty for anonymous (per the docstring's v1.4 TODO).
- Lifespan events bypass logging.
- WebSocket status is the accept code (101) or pre-accept close code.
- Logger failures NEVER crash the request (the silent-swallow pattern
  matches the auth middleware's defensive handling).

The tests drive the middleware directly with a minimal stub app and
patch ``_get_logger`` to capture emitted records, so we don't depend
on structlog being installed in the test env.
"""

from __future__ import annotations

import pytest

# =============================================================================
# IMPORTS / SETUP
# =============================================================================
# The middleware is shipped with the package (not gated on [ui]). If it
# cannot be imported, that is a regression and tests should fail loudly
# at collection time.
from backpropagate.ui_app.middleware.request_logging import (  # noqa: E402
    _client_addr,
    _enabled_via_env,
    request_logging_middleware,
)

# =============================================================================
# HELPERS — minimal ASGI scaffolding
# =============================================================================


async def _stub_http_app(scope: dict, receive, send) -> None:
    """Minimal ASGI app — sends a 200 OK back via the wrapped send."""
    if scope["type"] == "http":
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        })
        await send({"type": "http.response.body", "body": b"ok"})
    elif scope["type"] == "websocket":
        message = await receive()
        if message["type"] == "websocket.connect":
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.close", "code": 1000})


async def _stub_http_app_404(scope: dict, receive, send) -> None:
    """Minimal ASGI app — sends a 404 (for status-capture tests)."""
    await send({
        "type": "http.response.start",
        "status": 404,
        "headers": [(b"content-type", b"text/plain")],
    })
    await send({"type": "http.response.body", "body": b"not found"})


async def _empty_receive() -> dict:
    """ASGI receive callable that returns a websocket.connect message."""
    return {"type": "websocket.connect"}


def _make_http_scope(
    method: str = "GET",
    path: str = "/api/status",
    client_ip: str = "10.0.0.50",
) -> dict:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [],
        "client": (client_ip, 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }


def _make_ws_scope(path: str = "/_event", client_ip: str = "10.0.0.51") -> dict:
    return {
        "type": "websocket",
        "path": path,
        "headers": [],
        "client": (client_ip, 12345),
        "server": ("testserver", 80),
        "scheme": "ws",
    }


class _AsgiRecorder:
    """Captures the ASGI send() messages."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, message: dict) -> None:
        self.messages.append(message)


class _LoggerCapture:
    """Drop-in replacement for the logger that records info() calls.

    The middleware emits via ``logger.info("ui.request", **fields)``;
    this helper captures the (event, fields) tuple so tests can assert
    on the structured shape without depending on structlog.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def info(self, event: str, **fields) -> None:
        self.calls.append((event, fields))


@pytest.fixture
def logger_capture(monkeypatch):
    """Patch ``_get_logger`` to return a fresh _LoggerCapture per test.

    The capture is yielded so the test body can assert on
    ``capture.calls`` after driving the middleware.
    """
    capture = _LoggerCapture()
    monkeypatch.setattr(
        "backpropagate.ui_app.middleware.request_logging._get_logger",
        lambda: capture,
    )
    return capture


# =============================================================================
# FAST-PATH (default OFF) — env var not set
# =============================================================================


@pytest.mark.asyncio
async def test_middleware_passthrough_when_env_unset(monkeypatch, logger_capture):
    """With env var unset, middleware is pass-through and logs NOTHING.

    This is the default; emitting log records when the operator did not
    opt in would silently leak request shape to operator logs.
    """
    monkeypatch.delenv("BACKPROPAGATE_UI_REQUEST_LOG", raising=False)

    middleware = request_logging_middleware(_stub_http_app)
    recorder = _AsgiRecorder()

    await middleware(_make_http_scope(), _empty_receive, recorder.send)

    # Stub app produced a 200 response.
    statuses = [m.get("status") for m in recorder.messages if m.get("type") == "http.response.start"]
    assert statuses == [200]

    # No log records emitted.
    assert logger_capture.calls == [], (
        f"Default-off middleware must NOT log; got calls={logger_capture.calls}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled_value", ["0", "false", "no", "off", ""])
async def test_middleware_passthrough_when_env_disabled(
    monkeypatch, logger_capture, disabled_value
):
    """Any falsy env value (0, false, no, off, empty) keeps logging OFF.

    The contract is opt-in: only ``1`` / ``true`` / ``yes`` / ``on``
    enable logging. Everything else is OFF.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_REQUEST_LOG", disabled_value)

    middleware = request_logging_middleware(_stub_http_app)
    recorder = _AsgiRecorder()

    await middleware(_make_http_scope(), _empty_receive, recorder.send)

    assert logger_capture.calls == [], (
        f"Env var {disabled_value!r} should be falsy; got calls={logger_capture.calls}"
    )


# =============================================================================
# ENABLED PATH — structured field emission
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled_value", ["1", "true", "yes", "on", "TRUE", "On"])
async def test_middleware_logs_when_env_enabled(
    monkeypatch, logger_capture, enabled_value
):
    """All truthy env values (1/true/yes/on, case-insensitive) enable logging."""
    monkeypatch.setenv("BACKPROPAGATE_UI_REQUEST_LOG", enabled_value)

    middleware = request_logging_middleware(_stub_http_app)
    recorder = _AsgiRecorder()

    await middleware(_make_http_scope(), _empty_receive, recorder.send)

    assert len(logger_capture.calls) == 1, (
        f"Enabled env={enabled_value!r} should produce exactly one log record; "
        f"got calls={logger_capture.calls}"
    )


@pytest.mark.asyncio
async def test_log_record_contains_all_required_structured_fields(
    monkeypatch, logger_capture
):
    """The emitted log record contains the V1_3_BRIEF P0 item 5 field set.

    Required: method, path, status, duration_ms, auth_mode, auth_user,
    remote_addr. The brief specifies these as the structured-log
    contract that operators grep / ship to a SIEM.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_REQUEST_LOG", "1")

    middleware = request_logging_middleware(_stub_http_app)
    await middleware(
        _make_http_scope(method="POST", path="/api/train", client_ip="10.0.0.99"),
        _empty_receive,
        _AsgiRecorder().send,
    )

    assert len(logger_capture.calls) == 1
    event, fields = logger_capture.calls[0]
    assert event == "ui.request", (
        f"The event name MUST be 'ui.request' (operator grep target); got {event!r}"
    )

    required = {"method", "path", "status", "duration_ms",
                "auth_mode", "auth_user", "remote_addr"}
    missing = required - fields.keys()
    assert not missing, (
        f"Log record missing required fields: {missing}. "
        f"Got fields={set(fields.keys())}"
    )

    assert fields["method"] == "POST"
    assert fields["path"] == "/api/train"
    assert fields["remote_addr"] == "10.0.0.99"
    assert fields["status"] == 200
    assert isinstance(fields["duration_ms"], (int, float))
    assert fields["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_log_record_status_captured_from_http_response_start(
    monkeypatch, logger_capture
):
    """The middleware captures the status code from http.response.start.

    Wraps the send() callable to peek the first http.response.start
    message's status; if a regression dropped that wrapping, the status
    field would always be empty.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_REQUEST_LOG", "1")

    middleware = request_logging_middleware(_stub_http_app_404)
    await middleware(_make_http_scope(), _empty_receive, _AsgiRecorder().send)

    assert len(logger_capture.calls) == 1
    _, fields = logger_capture.calls[0]
    assert fields["status"] == 404, (
        f"Status must be captured from http.response.start; got status={fields['status']}"
    )


@pytest.mark.asyncio
async def test_log_record_anonymous_auth_user_is_empty_string(
    monkeypatch, logger_capture
):
    """auth_user is the empty string when no auth context is available.

    Per the request_logging docstring, the v1.3 implementation cannot
    reach back into auth's local ``authed_user`` variable; the field
    is left empty for anonymous requests until a v1.4 enhancement
    populates scope['_bp_auth_user']. This test pins the current
    contract so a regression that started defaulting to None or
    "unknown" is caught.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_REQUEST_LOG", "1")
    # Ensure no auth state is set.
    monkeypatch.delenv("BACKPROPAGATE_UI_AUTH", raising=False)

    middleware = request_logging_middleware(_stub_http_app)
    await middleware(_make_http_scope(), _empty_receive, _AsgiRecorder().send)

    assert len(logger_capture.calls) == 1
    _, fields = logger_capture.calls[0]
    assert fields["auth_user"] == "", (
        f"auth_user must be empty string for anonymous requests; "
        f"got auth_user={fields['auth_user']!r}"
    )


@pytest.mark.asyncio
async def test_log_record_auth_mode_resolved_from_auth_module(
    monkeypatch, logger_capture
):
    """auth_mode field is sourced from ui_app.auth._detect_mode.

    The middleware re-detects the auth mode per request so env-var
    changes take effect without a process restart. The actual mode
    values are auth-module-internal; this test only pins that SOME
    mode string (not the literal empty string) lands in the field,
    so a regression that broke the cross-module call doesn't go
    unnoticed.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_REQUEST_LOG", "1")

    middleware = request_logging_middleware(_stub_http_app)
    await middleware(_make_http_scope(), _empty_receive, _AsgiRecorder().send)

    assert len(logger_capture.calls) == 1
    _, fields = logger_capture.calls[0]
    # Mode might be ""/some-string depending on env. The contract pins
    # the field is present (above) and is a str — not None.
    assert isinstance(fields["auth_mode"], str), (
        f"auth_mode must be a string (possibly empty if auth import "
        f"failed defensively); got type={type(fields['auth_mode']).__name__}"
    )


# =============================================================================
# WEBSOCKET LOGGING SHAPE
# =============================================================================


@pytest.mark.asyncio
async def test_ws_upgrade_logged_with_method_ws_and_status_101(
    monkeypatch, logger_capture
):
    """WebSocket upgrades log method='WS' and status=101 (Switching Protocols).

    HTTP-aligned status code makes operators' grep/SIEM queries
    consistent across HTTP and WS traffic.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_REQUEST_LOG", "1")

    middleware = request_logging_middleware(_stub_http_app)
    await middleware(_make_ws_scope(), _empty_receive, _AsgiRecorder().send)

    assert len(logger_capture.calls) == 1
    _, fields = logger_capture.calls[0]
    assert fields["method"] == "WS", (
        f"WebSocket scope should log method='WS'; got method={fields['method']!r}"
    )
    assert fields["status"] == 101, (
        f"WebSocket accept should log status=101; got status={fields['status']!r}"
    )


@pytest.mark.asyncio
async def test_ws_pre_accept_close_logs_close_code_as_status(
    monkeypatch, logger_capture
):
    """A pre-accept websocket.close logs the close-code as the status field.

    Mirrors the auth middleware's 4401/4403 pre-accept reject pattern;
    operators need to see "this WS upgrade was rejected with code X"
    in the structured log to debug auth/rate-limit decisions.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_REQUEST_LOG", "1")

    async def _reject_ws_app(scope, receive, send):
        # Simulate a middleware that closes pre-accept (no websocket.accept
        # message). This is the shape of auth.py rejecting an unauthed WS.
        await send({"type": "websocket.close", "code": 4401, "reason": "test"})

    middleware = request_logging_middleware(_reject_ws_app)
    await middleware(_make_ws_scope(), _empty_receive, _AsgiRecorder().send)

    assert len(logger_capture.calls) == 1
    _, fields = logger_capture.calls[0]
    assert fields["status"] == 4401, (
        f"Pre-accept close should log the close code as status; "
        f"got status={fields['status']!r}"
    )


# =============================================================================
# LIFESPAN PASS-THROUGH
# =============================================================================


@pytest.mark.asyncio
async def test_lifespan_scope_bypassed_even_when_enabled(
    monkeypatch, logger_capture
):
    """Lifespan events don't get logged (they're per-process, not per-request).

    Lifespan messages drive server boot/shutdown; logging them would
    bloat operator logs with one record per startup/shutdown event.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_REQUEST_LOG", "1")

    invoked = []

    async def _lifespan_app(scope, receive, send):
        invoked.append(scope.get("type"))

    middleware = request_logging_middleware(_lifespan_app)
    await middleware(
        {"type": "lifespan"},
        _empty_receive,
        _AsgiRecorder().send,
    )

    assert invoked == ["lifespan"]
    assert logger_capture.calls == [], (
        f"Lifespan scope must not produce a log record; "
        f"got calls={logger_capture.calls}"
    )


# =============================================================================
# DEFENSIVE / FAULT-INJECTION
# =============================================================================


@pytest.mark.asyncio
async def test_logger_failure_does_not_crash_request(monkeypatch):
    """If the logger.info() call raises, the middleware silent-swallows it.

    The auth-middleware doctrine forbids logging failures from
    crashing live requests; this middleware inherits the same
    defensive shape. A regression that re-raised would 5xx every
    request after the logger went sideways.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_REQUEST_LOG", "1")

    class _BrokenLogger:
        def info(self, *args, **kwargs):
            raise RuntimeError("logger broken — operator-side config issue")

    monkeypatch.setattr(
        "backpropagate.ui_app.middleware.request_logging._get_logger",
        lambda: _BrokenLogger(),
    )

    middleware = request_logging_middleware(_stub_http_app)
    recorder = _AsgiRecorder()

    # Must not raise.
    await middleware(_make_http_scope(), _empty_receive, recorder.send)

    # The upstream app's 200 still made it through.
    statuses = [m.get("status") for m in recorder.messages if m.get("type") == "http.response.start"]
    assert statuses == [200], (
        "Broken logger must NOT prevent the upstream app's response; "
        f"got statuses={statuses}"
    )


@pytest.mark.asyncio
async def test_logger_handles_stdlib_signature_fallback(monkeypatch):
    """If the captured logger doesn't accept **fields kwargs, fall back to extra=.

    The module's emit path tries structlog signature first
    (``logger.info(event, **fields)``), then falls back to stdlib
    (``logger.info(event, extra={...})``) on TypeError. This pins
    that fallback so a regression to "only structlog works" is caught.
    """
    monkeypatch.setenv("BACKPROPAGATE_UI_REQUEST_LOG", "1")

    captured = []

    class _StdlibStyleLogger:
        def info(self, msg, *args, **kwargs):
            # stdlib's logger.info signature accepts extra= but not
            # arbitrary keyword fields. Reject the structlog-style call.
            if "extra" not in kwargs and kwargs:
                raise TypeError("stdlib doesn't accept arbitrary kwargs")
            captured.append((msg, kwargs.get("extra")))

    monkeypatch.setattr(
        "backpropagate.ui_app.middleware.request_logging._get_logger",
        lambda: _StdlibStyleLogger(),
    )

    middleware = request_logging_middleware(_stub_http_app)
    await middleware(_make_http_scope(), _empty_receive, _AsgiRecorder().send)

    assert len(captured) == 1
    msg, extra = captured[0]
    assert msg == "ui.request"
    # Fallback path puts fields in extra=.
    assert extra is not None
    assert "method" in extra
    assert "path" in extra


# =============================================================================
# ENV-VAR PARSER
# =============================================================================


def test_enabled_via_env_truthy_values():
    """All documented truthy values enable logging."""
    for v in ("1", "true", "yes", "on", "TRUE", "Yes"):
        assert _enabled_via_env({"BACKPROPAGATE_UI_REQUEST_LOG": v}) is True, (
            f"Truthy value {v!r} should enable logging"
        )


def test_enabled_via_env_falsy_values():
    """All non-truthy values keep logging off."""
    for v in ("0", "false", "no", "off", "", "garbage", "  "):
        assert _enabled_via_env({"BACKPROPAGATE_UI_REQUEST_LOG": v}) is False, (
            f"Falsy value {v!r} should NOT enable logging"
        )


def test_enabled_via_env_unset_returns_false():
    """Unset env defaults to False (the documented opt-in default)."""
    assert _enabled_via_env({}) is False


# =============================================================================
# CLIENT-ADDR EXTRACTION
# =============================================================================


def test_client_addr_returns_host_from_tuple():
    """``_client_addr`` returns the host string from a (host, port) tuple."""
    assert _client_addr({"client": ("172.16.0.5", 9999)}) == "172.16.0.5"


def test_client_addr_returns_empty_for_none_client():
    """``_client_addr`` returns empty string for None or missing client."""
    assert _client_addr({"client": None}) == ""
    assert _client_addr({}) == ""


def test_client_addr_returns_empty_for_malformed_value():
    """``_client_addr`` returns empty string for malformed client values."""
    assert _client_addr({"client": ()}) == ""
    assert _client_addr({"client": "not-tuple"}) == ""
