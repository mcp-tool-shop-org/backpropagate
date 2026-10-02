"""Coverage-raising tests for the small ASGI middlewares (healthz, request
logging, rate limit).

Each middleware is driven directly with a hand-rolled ASGI scope/receive/send
triple, so nothing binds a port. Mocked: the logger in the request-logging
tests (to capture the emitted record), ``ui_app.auth._detect_mode`` where a
test needs it to explode, and ``time.monotonic`` is not mocked at all (the
rate limiter is exercised with explicit ``now`` values instead).
"""

from __future__ import annotations

import json

import pytest

from backpropagate.ui_app.middleware import healthz as healthz_mod
from backpropagate.ui_app.middleware import rate_limit as rl
from backpropagate.ui_app.middleware import request_logging as rlog


async def _recording_app(scope, receive, send):
    """Inner app that records that it was reached and answers 200."""
    _recording_app.calls.append(scope.get("type"))
    if scope["type"] == "http":
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"inner"})


_recording_app.calls = []


async def _rejecting_app(scope, receive, send):
    """Inner app that rejects like the auth layer (401 / pre-accept 4401)."""
    _recording_app.calls.append(scope.get("type"))
    if scope["type"] == "http":
        await send({"type": "http.response.start", "status": 401, "headers": []})
        await send({"type": "http.response.body", "body": b"no"})
    elif scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 4401})


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    _recording_app.calls.clear()
    rl._reset_for_tests()
    for var in ("BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN",
                "BACKPROPAGATE_UI_RATE_LIMIT_WS_PER_MIN",
                "BACKPROPAGATE_UI_RATE_LIMIT_UPLOAD_PER_MIN",
                "BACKPROPAGATE_UI_REQUEST_LOG"):
        monkeypatch.delenv(var, raising=False)
    yield
    rl._reset_for_tests()


async def _call(mw, scope, messages=None):
    sent: list[dict] = []
    inbox = list(messages or [])

    async def receive():
        return inbox.pop(0) if inbox else {"type": "websocket.disconnect"}

    async def send(message):
        sent.append(message)

    await mw(scope, receive, send)
    return sent


def _http(path="/", method="GET", client=("10.0.0.1", 5555), host="localhost"):
    scope = {"type": "http", "path": path, "method": method, "client": client}
    if host is not None:
        scope["headers"] = [(b"host", host.encode())]
    return scope


# =============================================================================
# healthz
# =============================================================================


class TestHealthz:
    async def test_get_returns_status_only(self, monkeypatch):
        monkeypatch.delenv("BACKPROPAGATE_UI_AUTH", raising=False)
        mw = healthz_mod.healthz_middleware(_recording_app)
        sent = await _call(mw, _http("/healthz"))
        start, body = sent
        assert start["status"] == 200
        headers = {k.lower(): v for k, v in start["headers"]}
        assert headers[b"content-type"].startswith(b"application/json")
        assert headers[b"cache-control"] == b"no-store"
        assert headers[b"x-content-type-options"] == b"nosniff"
        assert headers[b"x-frame-options"] == b"SAMEORIGIN"
        assert int(headers[b"content-length"]) == len(body["body"])
        assert json.loads(body["body"]) == {"status": "ok"}
        assert _recording_app.calls == []  # never reached the inner app

    async def test_probe_is_unauthenticated_and_reports_configured_mode(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "alice:s3cret")
        mw = healthz_mod.healthz_middleware(_recording_app)
        sent = await _call(mw, _http("/healthz"))
        body = sent[1]["body"]
        assert sent[0]["status"] == 200
        assert b"s3cret" not in body and b"alice" not in body  # no credential echo

    async def test_head_returns_headers_and_empty_body(self):
        mw = healthz_mod.healthz_middleware(_recording_app)
        sent = await _call(mw, _http("/healthz", method="HEAD"))
        assert sent[0]["status"] == 200
        assert sent[1]["body"] == b""
        assert int(dict(sent[0]["headers"])[b"content-length"]) > 0

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    async def test_non_get_is_405_with_allow_header(self, method):
        mw = healthz_mod.healthz_middleware(_recording_app)
        sent = await _call(mw, _http("/healthz", method=method))
        assert sent[0]["status"] == 405
        headers = {k.lower(): v for k, v in sent[0]["headers"]}
        assert headers[b"allow"] == b"GET, HEAD"
        assert headers[b"x-content-type-options"] == b"nosniff"
        assert headers[b"x-frame-options"] == b"SAMEORIGIN"
        assert sent[1]["body"] == b""
        assert _recording_app.calls == []

    async def test_missing_method_defaults_to_get(self):
        mw = healthz_mod.healthz_middleware(_recording_app)
        scope = {"type": "http", "path": "/healthz", "headers": [(b"host", b"localhost")]}
        sent = await _call(mw, scope)
        assert sent[0]["status"] == 200

    @pytest.mark.parametrize("path", ["/", "/healthz/", "/healthzz", "/runs"])
    async def test_other_paths_pass_through(self, path):
        mw = healthz_mod.healthz_middleware(_recording_app)
        sent = await _call(mw, _http(path))
        assert sent[0]["status"] == 200 and sent[1]["body"] == b"inner"
        assert _recording_app.calls == ["http"]

    async def test_websocket_scope_on_healthz_path_passes_through(self):
        mw = healthz_mod.healthz_middleware(_recording_app)
        await _call(mw, {"type": "websocket", "path": "/healthz"})
        assert _recording_app.calls == ["websocket"]

    async def test_foreign_host_is_421(self):
        mw = healthz_mod.healthz_middleware(_recording_app)
        sent = await _call(mw, _http("/healthz", host="evil.com"))
        assert sent[0]["status"] == 421
        assert _recording_app.calls == []

    def test_header_builder_failure_still_stamps_nosniff(self, monkeypatch):
        """Mocked: the shared header builder raises."""
        import backpropagate.ui_app.auth as auth_mod

        def boom():
            raise RuntimeError("no headers")

        monkeypatch.setattr(auth_mod, "_hardened_header_pairs", boom)
        headers = dict(healthz_mod._health_security_headers())
        assert headers[b"x-content-type-options"] == b"nosniff"
        assert headers[b"x-frame-options"] == b"SAMEORIGIN"
        assert json.loads(healthz_mod._health_payload()) == {"status": "ok"}


# =============================================================================
# request_logging
# =============================================================================


class _Capture:
    def __init__(self, *, reject_kwargs=False, explode=False):
        self.records: list[tuple[str, dict]] = []
        self.reject_kwargs = reject_kwargs
        self.explode = explode

    def info(self, event, **fields):
        if self.explode:
            raise RuntimeError("sink down")
        if self.reject_kwargs and "extra" not in fields:
            raise TypeError("stdlib logger takes extra=")
        self.records.append((event, fields))


class TestRequestLoggingHelpers:
    @pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
    def test_truthy_env_values_enable(self, value):
        assert rlog._enabled_via_env({rlog._ENABLED_ENV: value}) is True

    @pytest.mark.parametrize("value", ["", "0", "false", "off", "nope"])
    def test_other_values_leave_logging_off(self, value):
        assert rlog._enabled_via_env({rlog._ENABLED_ENV: value}) is False

    def test_env_defaults_to_process_environment(self, monkeypatch):
        monkeypatch.setenv(rlog._ENABLED_ENV, "1")
        assert rlog._enabled_via_env() is True

    @pytest.mark.parametrize(
        ("client", "expected"),
        [(None, ""), ((), ""), ("1.2.3.4", ""), (("1.2.3.4", 80), "1.2.3.4"),
         (["::1", 80], "::1")],
    )
    def test_client_addr_forms(self, client, expected):
        assert rlog._client_addr({"client": client}) == expected

    def test_client_addr_survives_unstringifiable_host(self):
        class Bad:
            def __str__(self):
                raise TypeError("no str")

        assert rlog._client_addr({"client": (Bad(), 1)}) == ""

    def test_get_logger_returns_a_logger_with_info(self):
        assert callable(rlog._get_logger().info)

    def test_get_logger_falls_back_to_stdlib_when_logging_config_breaks(self, monkeypatch):
        """Mocked: ``backpropagate.logging_config.get_logger`` raises."""
        import logging

        import backpropagate.logging_config as lc

        def boom(_name=None):
            raise RuntimeError("structlog missing")

        monkeypatch.setattr(lc, "get_logger", boom)
        logger = rlog._get_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == rlog.__name__

    def test_resolve_auth_context_reports_mode_and_blank_user(self):
        mode, user = rlog._resolve_auth_context({})
        assert mode == "no_auth_local_only" and user == ""

    def test_resolve_auth_context_swallows_detection_failure(self, monkeypatch):
        """Mocked: ``_detect_mode`` raises."""
        import backpropagate.ui_app.auth as auth_mod

        def boom(_env):
            raise RuntimeError("x")

        monkeypatch.setattr(auth_mod, "_detect_mode", boom)
        assert rlog._resolve_auth_context({}) == ("", "")


class TestRequestLoggingMiddleware:
    async def test_disabled_by_default_delegates_without_logging(self, monkeypatch):
        cap = _Capture()
        monkeypatch.setattr(rlog, "_get_logger", lambda: cap)
        sent = await _call(rlog.request_logging_middleware(_recording_app), _http())
        assert sent[0]["status"] == 200
        assert cap.records == []

    async def test_enabled_logs_one_structured_record(self, monkeypatch):
        monkeypatch.setenv(rlog._ENABLED_ENV, "1")
        cap = _Capture()
        monkeypatch.setattr(rlog, "_get_logger", lambda: cap)
        await _call(rlog.request_logging_middleware(_recording_app), _http("/runs", "post"))
        assert len(cap.records) == 1
        event, f = cap.records[0]
        assert event == "ui.request"
        assert f["method"] == "POST" and f["path"] == "/runs" and f["status"] == 200
        assert f["remote_addr"] == "10.0.0.1" and f["scope_type"] == "http"
        assert f["auth_user"] == "" and f["duration_ms"] >= 0

    async def test_lifespan_is_not_logged(self, monkeypatch):
        monkeypatch.setenv(rlog._ENABLED_ENV, "1")
        cap = _Capture()
        monkeypatch.setattr(rlog, "_get_logger", lambda: cap)
        await _call(rlog.request_logging_middleware(_recording_app), {"type": "lifespan"})
        assert _recording_app.calls == ["lifespan"] and cap.records == []

    async def test_websocket_accept_logs_101_and_ws_method(self, monkeypatch):
        monkeypatch.setenv(rlog._ENABLED_ENV, "1")
        cap = _Capture()
        monkeypatch.setattr(rlog, "_get_logger", lambda: cap)

        async def ws_app(scope, receive, send):
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.close", "code": 1000})

        await _call(rlog.request_logging_middleware(ws_app),
                    {"type": "websocket", "path": "/_event", "client": None})
        f = cap.records[0][1]
        assert f["method"] == "WS" and f["status"] == 101 and f["remote_addr"] == ""

    async def test_websocket_pre_accept_close_logs_close_code(self, monkeypatch):
        monkeypatch.setenv(rlog._ENABLED_ENV, "1")
        cap = _Capture()
        monkeypatch.setattr(rlog, "_get_logger", lambda: cap)

        async def reject_app(scope, receive, send):
            await send({"type": "websocket.close", "code": 4401})

        await _call(rlog.request_logging_middleware(reject_app),
                    {"type": "websocket", "path": "/_event"})
        assert cap.records[0][1]["status"] == 4401

    async def test_logs_even_when_inner_app_raises(self, monkeypatch):
        monkeypatch.setenv(rlog._ENABLED_ENV, "1")
        cap = _Capture()
        monkeypatch.setattr(rlog, "_get_logger", lambda: cap)

        async def crash(scope, receive, send):
            raise ValueError("inner crashed")

        with pytest.raises(ValueError, match="inner crashed"):
            await _call(rlog.request_logging_middleware(crash), _http("/boom"))
        f = cap.records[0][1]
        assert f["path"] == "/boom" and f["status"] == ""

    async def test_falls_back_to_extra_kwarg_for_stdlib_loggers(self, monkeypatch):
        monkeypatch.setenv(rlog._ENABLED_ENV, "1")
        cap = _Capture(reject_kwargs=True)

        class StdlibLike:
            def info(self, event, **fields):
                if "extra" not in fields:
                    raise TypeError("unexpected kwargs")
                cap.records.append((event, fields["extra"]))

        monkeypatch.setattr(rlog, "_get_logger", lambda: StdlibLike())
        await _call(rlog.request_logging_middleware(_recording_app), _http("/x"))
        assert cap.records[0][1]["path"] == "/x"

    async def test_logger_failure_never_breaks_the_request(self, monkeypatch):
        monkeypatch.setenv(rlog._ENABLED_ENV, "1")
        monkeypatch.setattr(rlog, "_get_logger", lambda: _Capture(explode=True))
        sent = await _call(rlog.request_logging_middleware(_recording_app), _http())
        assert sent[0]["status"] == 200  # response was delivered despite sink failure


# =============================================================================
# rate_limit
# =============================================================================


class TestRateLimitHelpers:
    def test_resolve_cap_defaults_and_overrides(self):
        env = "X"
        assert rl._resolve_cap(env, 7, {}) == 7
        assert rl._resolve_cap(env, 7, {env: "  "}) == 7
        assert rl._resolve_cap(env, 7, {env: "42"}) == 42
        assert rl._resolve_cap(env, 7, {env: "0"}) == 0  # 0 disables

    def test_resolve_cap_negative_falls_back_to_default(self):
        assert rl._resolve_cap("X", 7, {"X": "-3"}) == 7

    def test_resolve_cap_non_integer_warns_and_falls_back(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger=rl.logger.name):
            assert rl._resolve_cap("X", 7, {"X": "lots"}) == 7
        assert any("not an integer" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize(
        ("client", "expected"),
        [(None, ""), ((), ""), ("1.2.3.4", ""), (("1.2.3.4", 1), "1.2.3.4")],
    )
    def test_client_addr(self, client, expected):
        assert rl._client_addr({"client": client}) == expected

    def test_client_addr_unstringifiable_host(self):
        class Bad:
            def __str__(self):
                raise ValueError("no")

        assert rl._client_addr({"client": (Bad(), 1)}) == ""

    def test_429_still_stamps_nosniff_when_the_builder_raises(self, monkeypatch):
        """Mocked: the shared header builder raises."""
        import backpropagate.ui_app.auth as auth_mod

        def boom():
            raise RuntimeError("no headers")

        monkeypatch.setattr(auth_mod, "_hardened_header_pairs", boom)
        _body, headers = rl._build_429_response()
        listed = dict(headers)
        assert listed[b"x-content-type-options"] == b"nosniff"
        assert listed[b"x-frame-options"] == b"SAMEORIGIN"

    def test_429_response_shape(self):
        body, headers = rl._build_429_response()
        h = {k.lower(): v for k, v in headers}
        assert h[b"retry-after"] == b"60" and h[b"cache-control"] == b"no-store"
        assert h[b"x-content-type-options"] == b"nosniff"
        assert h[b"x-frame-options"] == b"SAMEORIGIN"
        assert int(h[b"content-length"]) == len(body)
        assert body.startswith(b"429 Too Many Requests")


class TestSlidingWindow:
    def test_cap_zero_or_blank_ip_never_rejects(self):
        w = rl._SlidingWindow()
        assert w.record_and_check("1.1.1.1", 0.0, 0) is False
        assert w.record_and_check("", 0.0, 5) is False
        assert w._events == {}

    def test_over_cap_only_after_cap_events_inside_window(self):
        w = rl._SlidingWindow()
        results = [w.record_and_check("a", float(i), 3) for i in range(5)]
        assert results == [False, False, False, True, True]

    def test_old_events_age_out_of_the_window(self):
        w = rl._SlidingWindow()
        for i in range(3):
            w.record_and_check("a", float(i), 3)
        # 100s later the earlier events are outside the 60s window
        assert w.record_and_check("a", 100.0, 3) is False
        assert len(w._events["a"]) == 1

    def test_flood_memory_is_bounded_per_ip(self):
        w = rl._SlidingWindow()
        for i in range(500):
            w.record_and_check("flood", 1.0 + i * 1e-6, 2)
        assert len(w._events["flood"]) <= 16

    def test_ips_are_independent(self):
        w = rl._SlidingWindow()
        for i in range(5):
            w.record_and_check("a", float(i), 3)
        assert w.record_and_check("b", 5.0, 3) is False

    def test_prune_idle_ips_drops_only_stale_entries(self):
        w = rl._SlidingWindow()
        w.record_and_check("old", 0.0, 5)
        w.record_and_check("fresh", 95.0, 5)
        w._events["empty"] = type(w._events["old"])()
        w.prune_idle_ips(100.0)
        assert set(w._events) == {"fresh"}


class TestRateLimitMiddleware:
    async def test_lifespan_passes_through(self):
        await _call(rl.rate_limit_middleware(_recording_app), {"type": "lifespan"})
        assert _recording_app.calls == ["lifespan"]

    async def test_both_caps_zero_disables_everything(self, monkeypatch):
        monkeypatch.setenv(rl._HTTP_ENV, "0")
        monkeypatch.setenv(rl._WS_ENV, "0")
        monkeypatch.setenv(rl._UPLOAD_ENV, "0")
        mw = rl.rate_limit_middleware(_recording_app)
        for _ in range(300):
            sent = await _call(mw, _http())
            assert sent[0]["status"] == 200

    @pytest.mark.parametrize("path", ["/ping", "/_next/static/x.js", "/favicon.ico"])
    async def test_exempt_paths_are_never_limited(self, monkeypatch, path):
        monkeypatch.setenv(rl._HTTP_ENV, "1")
        mw = rl.rate_limit_middleware(_recording_app)
        for _ in range(5):
            sent = await _call(mw, _http(path))
            assert sent[0]["status"] == 200

    async def test_http_over_cap_gets_429_and_inner_app_not_reached(self, monkeypatch):
        monkeypatch.setenv(rl._HTTP_ENV, "2")
        mw = rl.rate_limit_middleware(_rejecting_app)
        statuses = [(await _call(mw, _http("/runs")))[0]["status"] for _ in range(4)]
        assert statuses == [401, 401, 429, 429]
        assert _recording_app.calls == ["http", "http"]
        sent = await _call(mw, _http("/runs"))
        assert dict(sent[0]["headers"])[b"retry-after"] == b"60"

    async def test_limit_is_per_ip(self, monkeypatch):
        monkeypatch.setenv(rl._HTTP_ENV, "1")
        mw = rl.rate_limit_middleware(_rejecting_app)
        ok = rl.rate_limit_middleware(_recording_app)
        assert (await _call(mw, _http(client=("1.1.1.1", 1))))[0]["status"] == 401
        assert (await _call(mw, _http(client=("1.1.1.1", 1))))[0]["status"] == 429
        assert (await _call(ok, _http(client=("2.2.2.2", 1))))[0]["status"] == 200

    async def test_unknown_client_is_never_rejected(self, monkeypatch):
        monkeypatch.setenv(rl._HTTP_ENV, "1")
        mw = rl.rate_limit_middleware(_recording_app)
        for _ in range(3):
            sent = await _call(mw, {"type": "http", "path": "/", "client": None})
            assert sent[0]["status"] == 200

    async def test_ws_over_cap_closes_pre_accept_with_4429(self, monkeypatch):
        monkeypatch.setenv(rl._WS_ENV, "1")
        mw = rl.rate_limit_middleware(_rejecting_app)
        scope = {"type": "websocket", "path": "/_event", "client": ("3.3.3.3", 9)}
        first = await _call(mw, scope)
        assert first == [{"type": "websocket.close", "code": 4401}]  # auth's rejection
        second = await _call(mw, scope)
        assert second == [{"type": "websocket.close", "code": rl._WS_CLOSE_CODE_RATE_LIMIT,
                           "reason": "rate_limit_exceeded"}]
        assert rl._WS_CLOSE_CODE_RATE_LIMIT == 4429
        assert _recording_app.calls == ["websocket"]

    async def test_http_and_ws_budgets_are_separate(self, monkeypatch):
        monkeypatch.setenv(rl._HTTP_ENV, "1")
        monkeypatch.setenv(rl._WS_ENV, "1")
        reject = rl.rate_limit_middleware(_rejecting_app)
        mw = rl.rate_limit_middleware(_recording_app)
        await _call(reject, _http())
        assert (await _call(mw, _http()))[0]["status"] == 429
        ws = {"type": "websocket", "path": "/_event", "client": ("10.0.0.1", 1)}
        assert await _call(mw, ws) == []  # first ws upgrade still allowed

    async def test_prune_sweep_runs_every_hundredth_event(self, monkeypatch):
        """The idle-IP sweep fires on the 100th event and clears aged-out IPs."""
        monkeypatch.setenv(rl._HTTP_ENV, "1000")
        # Seed a stale IP far in the past (monotonic timestamps are >> 60 apart).
        from collections import deque

        rl._HTTP_WINDOW._events["stale"] = deque([-1000.0])
        mw = rl.rate_limit_middleware(_recording_app)
        for _ in range(rl._PRUNE_INTERVAL):
            await _call(mw, _http())
        assert "stale" not in rl._HTTP_WINDOW._events

    async def test_unknown_scope_type_is_passed_through_unlimited(self, monkeypatch):
        monkeypatch.setenv(rl._HTTP_ENV, "1")
        monkeypatch.setenv(rl._WS_ENV, "1")
        mw = rl.rate_limit_middleware(_recording_app)
        for _ in range(3):
            await _call(mw, {"type": "custom", "path": "/", "client": ("9.9.9.9", 1)})
        assert _recording_app.calls == ["custom"] * 3

    async def test_upload_posts_have_their_own_cap(self, monkeypatch):
        monkeypatch.setenv(rl._UPLOAD_ENV, "2")
        monkeypatch.setenv(rl._HTTP_ENV, "100")
        mw = rl.rate_limit_middleware(_recording_app)
        statuses = [
            (await _call(mw, _http("/_upload", method="POST")))[0]["status"]
            for _ in range(4)
        ]
        assert statuses == [200, 200, 429, 429]
        assert (await _call(mw, _http("/runs")))[0]["status"] == 200
        blocked = await _call(mw, _http("/_upload", method="POST"))
        listed = {k.lower(): v for k, v in blocked[0]["headers"]}
        assert listed[b"x-content-type-options"] == b"nosniff"

    async def test_upload_cap_zero_does_not_limit_posts(self, monkeypatch):
        monkeypatch.setenv(rl._UPLOAD_ENV, "0")
        mw = rl.rate_limit_middleware(_recording_app)
        for _ in range(5):
            sent = await _call(mw, _http("/_upload", method="POST"))
            assert sent[0]["status"] == 200

    async def test_get_upload_is_not_on_the_upload_cap(self, monkeypatch):
        monkeypatch.setenv(rl._UPLOAD_ENV, "1")
        mw = rl.rate_limit_middleware(_recording_app)
        for _ in range(3):
            sent = await _call(mw, _http("/_upload"))
            assert sent[0]["status"] == 200

    async def test_accepted_event_frames_are_not_capped(self, monkeypatch):
        monkeypatch.setenv(rl._WS_ENV, "1")

        async def _accepting(scope, receive, send):
            _recording_app.calls.append(scope.get("type"))
            await send({"type": "websocket.accept"})
            for _ in range(20):
                await send({"type": "websocket.send", "text": "frame"})

        mw = rl.rate_limit_middleware(_accepting)
        scope = {"type": "websocket", "path": "/_event", "client": ("8.8.8.8", 1)}
        first = await _call(mw, scope)
        second = await _call(mw, scope)
        assert first[0]["type"] == "websocket.accept"
        assert second[0]["type"] == "websocket.accept"

    async def test_429_path_also_counts_toward_prune_interval(self, monkeypatch):
        monkeypatch.setenv(rl._HTTP_ENV, "1")
        mw = rl.rate_limit_middleware(_recording_app)
        await _call(mw, _http())
        before = rl._event_count
        await _call(mw, _http())  # 429 path
        assert rl._event_count == before + 1
