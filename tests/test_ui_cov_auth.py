"""Adversarial / branch tests for ``backpropagate.ui_app.auth`` (GHSA-f65r-h4g3-3h9h gate).

The middleware and every pure helper are driven directly (no server, no
sockets). Real collaborators throughout; nothing is mocked except
``ui_security.security_headers_dict`` in the single test that proves the
rejection builders survive a header-builder failure.

Complements ``test_auth_middleware.py`` / ``test_auth_middleware_fuzz.py``:
those cover the headline contracts, this file pins the remaining helper
edge cases (cookie parsing, host/origin matching, mode detection precedence,
TOKEN_AUTO redirect, WS pre-accept rejections).
"""

from __future__ import annotations

import base64
import time
from urllib.parse import quote

import pytest

from backpropagate.ui_app import auth

_AUTH_ENV = (
    "BACKPROPAGATE_UI_AUTH", "BACKPROPAGATE_UI_SHARE_HOST", "BACKPROPAGATE_UI_HOST_BIND",
    "BACKPROPAGATE_UI_LAUNCH_TOKEN",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _AUTH_ENV:
        monkeypatch.delenv(var, raising=False)


def _basic(user, pw):
    return "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()


# =============================================================================
# Mode detection / secret derivation
# =============================================================================


class TestDetectMode:
    @pytest.mark.parametrize(
        ("env", "mode"),
        [
            ({}, auth.AuthMode.NO_AUTH_LOCAL_ONLY),
            ({"BACKPROPAGATE_UI_AUTH": "u:p"}, auth.AuthMode.EXPLICIT_CREDS),
            ({"BACKPROPAGATE_UI_AUTH": "u:p", "BACKPROPAGATE_UI_SHARE_HOST": "x.trycloudflare.com"},
             auth.AuthMode.PRODUCTION),
            ({"BACKPROPAGATE_UI_AUTH": "u:p", "BACKPROPAGATE_UI_HOST_BIND": "192.168.1.5"},
             auth.AuthMode.PRODUCTION),
            ({"BACKPROPAGATE_UI_AUTH": "u:p", "BACKPROPAGATE_UI_HOST_BIND": "LocalHost"},
             auth.AuthMode.EXPLICIT_CREDS),
            ({"BACKPROPAGATE_UI_AUTH": "u:p", "BACKPROPAGATE_UI_HOST_BIND": "::1"},
             auth.AuthMode.EXPLICIT_CREDS),
            ({"BACKPROPAGATE_UI_LAUNCH_TOKEN": "tok"}, auth.AuthMode.TOKEN_AUTO),
            # explicit creds win over a launch token
            ({"BACKPROPAGATE_UI_AUTH": "u:p", "BACKPROPAGATE_UI_LAUNCH_TOKEN": "tok"},
             auth.AuthMode.EXPLICIT_CREDS),
            # whitespace-only values are "unset"
            ({"BACKPROPAGATE_UI_AUTH": "   "}, auth.AuthMode.NO_AUTH_LOCAL_ONLY),
            # a share host without creds does NOT upgrade to production
            ({"BACKPROPAGATE_UI_SHARE_HOST": "x.example"}, auth.AuthMode.NO_AUTH_LOCAL_ONLY),
        ],
    )
    def test_precedence(self, env, mode):
        assert auth._detect_mode(env) is mode

    def test_defaults_to_process_env(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        assert auth._detect_mode() is auth.AuthMode.EXPLICIT_CREDS


class TestDeriveSecret:
    def test_auth_creds_use_random_process_key_not_a_password_hash(self):
        import hashlib

        key = auth._derive_secret({"BACKPROPAGATE_UI_AUTH": " u:p "})
        assert key == auth._PROCESS_LOCAL_SECRET
        assert key != hashlib.sha256(b"u:p").digest()

    def test_launch_token_used_verbatim(self):
        assert auth._derive_secret({"BACKPROPAGATE_UI_LAUNCH_TOKEN": "abc"}) == b"abc"

    def test_creds_beat_token(self):
        env = {"BACKPROPAGATE_UI_AUTH": "u:p", "BACKPROPAGATE_UI_LAUNCH_TOKEN": "abc"}
        assert auth._derive_secret(env) != b"abc"

    def test_fallback_is_stable_non_empty_process_secret(self):
        s = auth._derive_secret({})
        assert s == auth._derive_secret({}) and len(s) == 32

    def test_defaults_to_process_env(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_LAUNCH_TOKEN", "from-env")
        assert auth._derive_secret() == b"from-env"


# =============================================================================
# Basic auth verification
# =============================================================================


class TestVerifyBasicAuth:
    ENV = {"BACKPROPAGATE_UI_AUTH": "alice:s3:cret"}

    def test_valid_credentials_return_username(self):
        assert auth._verify_basic_auth(_basic("alice", "s3:cret"), self.ENV) == "alice"

    def test_password_may_contain_colons(self):
        assert auth._verify_basic_auth(_basic("alice", "s3:cret"), self.ENV) == "alice"

    @pytest.mark.parametrize(
        "header",
        [
            "",                                  # missing
            "Basic",                             # no payload
            "Bearer abc",                        # wrong scheme
            "Basic !!!not-base64!!!",            # invalid base64
            "Basic " + base64.b64encode(b"\xff\xfe:x").decode(),  # not UTF-8
            "Basic " + base64.b64encode(b"nocolon").decode(),     # no ':'
            _basic("alice", "wrong"),
            _basic("mallory", "s3:cret"),
            _basic("", ""),
            _basic("alice", "s3:cret" + "x"),
        ],
    )
    def test_malformed_or_wrong_credentials_are_refused(self, header):
        assert auth._verify_basic_auth(header, self.ENV) is None

    def test_scheme_is_case_insensitive_and_whitespace_tolerant(self):
        hdr = "  bAsIc   " + base64.b64encode(b"alice:s3:cret").decode() + "  "
        assert auth._verify_basic_auth(hdr, self.ENV) == "alice"

    def test_unset_expected_creds_fail_closed(self):
        assert auth._verify_basic_auth(_basic("a", "b"), {}) is None

    def test_malformed_expected_creds_without_colon_fail_closed(self):
        assert auth._verify_basic_auth(_basic("a", "b"), {"BACKPROPAGATE_UI_AUTH": "nocolon"}) is None

    def test_defaults_to_process_env(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "bob:pw")
        assert auth._verify_basic_auth(_basic("bob", "pw")) == "bob"


# =============================================================================
# Session cookie sign / validate
# =============================================================================


class TestCookies:
    SECRET = b"k" * 32

    def test_roundtrip(self):
        c = auth._sign_cookie("alice", self.SECRET)
        assert auth._validate_cookie(c, self.SECRET) == "alice"

    def test_username_with_colons_roundtrips(self):
        c = auth._sign_cookie("a:b:c", self.SECRET)
        assert auth._validate_cookie(c, self.SECRET) == "a:b:c"

    def test_expired_cookie_is_refused(self):
        c = auth._sign_cookie("alice", self.SECRET, now=1_000.0)
        later = 1_000.0 + auth._COOKIE_TTL_SECONDS + 1
        assert auth._validate_cookie(c, self.SECRET, now=later) is None
        assert auth._validate_cookie(c, self.SECRET, now=1_001.0) == "alice"

    def test_signature_is_bound_to_secret_user_and_expiry(self):
        c = auth._sign_cookie("alice", self.SECRET)
        user, exp, sig = c.rsplit(":", 2)
        assert auth._validate_cookie(c, b"other-secret") is None
        assert auth._validate_cookie(f"mallory:{exp}:{sig}", self.SECRET) is None
        assert auth._validate_cookie(f"{user}:{int(exp) + 9999}:{sig}", self.SECRET) is None

    @pytest.mark.parametrize(
        "value", ["", "nocolons", "only:one", "a:notanint:sig", "alice:99999999999:", "::"],
    )
    def test_malformed_cookies_are_refused(self, value):
        assert auth._validate_cookie(value, self.SECRET) is None

    def test_now_defaults_to_wall_clock(self):
        c = auth._sign_cookie("alice", self.SECRET)
        _u, exp, _s = c.rsplit(":", 2)
        assert abs(int(exp) - (time.time() + auth._COOKIE_TTL_SECONDS)) < 5
        assert auth._validate_cookie(c, self.SECRET) == "alice"

    def test_set_cookie_header_flags(self):
        plain = auth._set_cookie_header("v", secure=False).decode()
        sec = auth._set_cookie_header("v", secure=True).decode()
        for flag in ("HttpOnly", "SameSite=Lax", "Path=/", f"Max-Age={auth._COOKIE_TTL_SECONDS}"):
            assert flag in plain
        assert "Secure" not in plain.replace("SameSite", "") and sec.endswith("Secure")

    def test_parse_cookies(self):
        parsed = auth._parse_cookies('a=1; b="two"; junk; =novalue; a=3;  c = x y ')
        assert parsed == {"a": "3", "b": "two", "c": "x y"}
        assert auth._parse_cookies("") == {}

    def test_parse_cookies_lone_quote_is_kept_verbatim(self):
        assert auth._parse_cookies('a="') == {"a": '"'}


# =============================================================================
# Host / Origin allowlists
# =============================================================================


class TestAllowlists:
    def test_host_allowlist_defaults_are_loopback_only(self):
        assert auth._host_allowlist({}) == {"localhost", "127.0.0.1", "::1"}

    def test_host_allowlist_adds_share_and_bind_but_never_wildcards(self):
        env = {"BACKPROPAGATE_UI_SHARE_HOST": "Tunnel.Example", "BACKPROPAGATE_UI_HOST_BIND": "10.0.0.7"}
        assert {"tunnel.example", "10.0.0.7"} <= auth._host_allowlist(env)
        for wildcard in ("0.0.0.0", "::"):
            assert wildcard not in auth._host_allowlist({"BACKPROPAGATE_UI_HOST_BIND": wildcard})

    def test_host_allowlist_defaults_to_process_env(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_SHARE_HOST", "t.example")
        assert "t.example" in auth._host_allowlist()

    def test_origin_allowlist_defaults(self):
        assert auth._origin_allowlist({}) == {
            "http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1",
        }

    def test_origin_allowlist_adds_share_and_lan_host_both_schemes(self):
        env = {"BACKPROPAGATE_UI_SHARE_HOST": "t.example", "BACKPROPAGATE_UI_HOST_BIND": "10.0.0.7"}
        allowed = auth._origin_allowlist(env)
        assert {"http://t.example", "https://t.example", "http://10.0.0.7", "https://10.0.0.7"} <= allowed

    @pytest.mark.parametrize("bind", ["0.0.0.0", "::", "localhost", "127.0.0.1", "::1"])
    def test_origin_allowlist_ignores_wildcard_and_loopback_bind(self, bind):
        assert auth._origin_allowlist({"BACKPROPAGATE_UI_HOST_BIND": bind}) == auth._origin_allowlist({})

    def test_origin_allowlist_defaults_to_process_env(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_SHARE_HOST", "t.example")
        assert "https://t.example" in auth._origin_allowlist()

    @pytest.mark.parametrize(
        ("host", "ok"),
        [
            ("localhost", True), ("LOCALHOST:7860", True), ("127.0.0.1:3000", True),
            ("[::1]:7860", True), ("[::1]", True),
            ("", False), ("evil.com", False), ("localhost.evil.com", False),
            ("evil.com:7860", False), ("127.0.0.1.evil.com", False),
            ("[::2]:80", False), ("[::1", False),
        ],
    )
    def test_host_matching_rebinding_cases(self, host, ok):
        assert auth._host_matches_allowlist(host, auth._host_allowlist({})) is ok

    @pytest.mark.parametrize(
        ("origin", "ok"),
        [
            ("", True),  # fail-open on missing Origin (HTTP); WS gate rejects separately
            ("http://localhost:3000", True), ("https://127.0.0.1", True),
            ("http://LOCALHOST", True),
            ("null", False), ("http://evil.com", False), ("http://localhost.evil.com", False),
            ("ftp://localhost", False),  # scheme is NOT in allowlist
            ("localhost", False), ("//localhost", False), ("http://", False),
            ("http://[::1", False),
        ],
    )
    def test_origin_matching_cswsh_cases(self, origin, ok):
        allow = auth._origin_allowlist({})
        assert auth._origin_matches_allowlist(origin, allow) is ok

    def test_origin_matching_non_string_is_refused(self):
        assert auth._origin_matches_allowlist(object(), auth._origin_allowlist({})) is False  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("host", "loopback"),
        [("", True), ("localhost:80", True), ("127.0.0.1", True), ("[::1]:80", True),
         ("example.com", False), ("10.0.0.5:7860", False), ("[fe80::1]:1", False)],
    )
    def test_is_loopback_host(self, host, loopback):
        assert auth._is_loopback_host(host) is loopback

    def test_header_value_lookup(self):
        headers = [(b"Host", b"a"), (b"cookie", b"c=1")]
        assert auth._header_value(headers, b"HOST") == "a"
        assert auth._header_value(headers, b"missing") == ""

    def test_header_value_bad_header_type_is_empty(self):
        class NoDecode:
            pass

        assert auth._header_value([(b"x", NoDecode())], b"x") == ""  # type: ignore[list-item]

    @pytest.mark.parametrize(
        ("path", "bypass"),
        [("/ping", True), ("/_next/static/a.js", True), ("/favicon.ico", True),
         ("", False), ("/", False),
         ("/runs", False), ("/_event", False)],
    )
    def test_passthrough_paths(self, path, bypass):
        assert auth._is_passthrough_http(path) is bypass


# =============================================================================
# Rejection builders
# =============================================================================


class TestRejectionBuilders:
    @pytest.mark.parametrize("hdr", [b"content-security-policy", b"x-frame-options",
                                      b"x-content-type-options"])
    def test_every_rejection_carries_hardened_headers(self, hdr):
        for _body, headers in (auth._build_401_response(), auth._build_421_response("x"),
                               auth._build_403_response("why")):
            assert hdr in {k.lower() for k, _ in headers}

    def test_401_challenges_and_is_not_cacheable(self):
        body, headers = auth._build_401_response(realm="r", hint="custom hint")
        h = dict(headers)
        assert body == b"custom hint"
        assert h[b"www-authenticate"] == b'Basic realm="r"'
        assert h[b"cache-control"] == b"no-store"
        assert int(h[b"content-length"]) == len(body)

    def test_401_default_hint(self):
        body, _ = auth._build_401_response()
        assert b"Authentication required" in body

    def test_421_and_403_bodies_name_the_defence(self):
        body421, _ = auth._build_421_response("evil.com")
        body403, _ = auth._build_403_response("bad origin")
        assert b"evil.com" in body421 and b"DNS-rebinding" in body421
        assert b"bad origin" in body403 and b"CWE-1385" in body403

    def test_rejection_survives_a_header_builder_failure(self, monkeypatch):
        """Mocked: ``ui_security.security_headers_dict`` raises."""
        import backpropagate.ui_security as sec

        def boom():
            raise RuntimeError("no headers for you")

        monkeypatch.setattr(sec, "security_headers_dict", boom)
        assert auth._hardened_header_pairs() == []
        body, headers = auth._build_401_response()
        assert body and (b"www-authenticate", b'Basic realm="backpropagate"') in headers


# =============================================================================
# Middleware: HTTP branch
# =============================================================================


async def _inner(scope, receive, send):
    _inner.calls.append(scope["type"])
    if scope["type"] == "http":
        await send({"type": "http.response.start", "status": 200, "headers": [(b"x-inner", b"1")]})
        await send({"type": "http.response.body", "body": b"inner"})


_inner.calls = []


async def _drive(scope):
    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(msg):
        sent.append(msg)

    await auth.basic_auth_transformer(_inner)(scope, receive, send)
    return sent


def _http(path="/", method="GET", host="localhost:7860", extra=None, query=b""):
    headers = [(b"host", host.encode())] if host is not None else []
    headers += extra or []
    return {"type": "http", "path": path, "method": method, "headers": headers,
            "query_string": query}


def _status(sent):
    return sent[0]["status"]


@pytest.fixture(autouse=True)
def _reset_inner():
    _inner.calls.clear()


class TestHttpMiddleware:
    async def test_lifespan_passes_through(self):
        await _drive({"type": "lifespan"})
        assert _inner.calls == ["lifespan"]

    async def test_unknown_scope_type_passes_through(self):
        await _drive({"type": "mystery"})
        assert _inner.calls == ["mystery"]

    async def test_no_auth_mode_allows_loopback_without_credentials(self):
        sent = await _drive(_http())
        assert _status(sent) == 200 and _inner.calls == ["http"]

    @pytest.mark.parametrize("host", ["evil.com", "localhost.evil.com:7860", None])
    async def test_dns_rebinding_host_is_421_in_every_mode(self, monkeypatch, host):
        for env in ({}, {"BACKPROPAGATE_UI_AUTH": "u:p"}):
            for k, v in env.items():
                monkeypatch.setenv(k, v)
            sent = await _drive(_http(host=host))
            assert _status(sent) == 421
            assert _inner.calls == []

    async def test_passthrough_paths_skip_host_and_auth_checks(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        sent = await _drive(_http("/ping", host="evil.com"))
        assert _status(sent) == 200 and _inner.calls == ["http"]

    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
    async def test_state_changing_method_with_foreign_origin_is_403(self, method):
        sent = await _drive(_http(method=method, extra=[(b"origin", b"http://evil.com")]))
        assert _status(sent) == 403 and _inner.calls == []
        assert b"CSWSH" in sent[1]["body"]

    async def test_get_with_foreign_origin_is_not_origin_checked(self):
        sent = await _drive(_http(extra=[(b"origin", b"http://evil.com")]))
        assert _status(sent) == 200

    async def test_post_with_loopback_origin_is_allowed(self):
        sent = await _drive(_http(method="POST", extra=[(b"origin", b"http://localhost:3000")]))
        assert _status(sent) == 200

    async def test_missing_credentials_get_401_challenge(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        sent = await _drive(_http())
        assert _status(sent) == 401
        assert (b"www-authenticate", b'Basic realm="backpropagate"') in sent[0]["headers"]
        assert _inner.calls == []

    async def test_wrong_credentials_get_401(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        sent = await _drive(_http(extra=[(b"authorization", _basic("u", "nope").encode())]))
        assert _status(sent) == 401

    async def test_good_basic_credentials_pass_and_set_hardened_session_cookie(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        sent = await _drive(_http(extra=[(b"authorization", _basic("u", "p").encode())]))
        assert _status(sent) == 200 and _inner.calls == ["http"]
        cookie = {k: v for k, v in sent[0]["headers"] if k == b"set-cookie"}[b"set-cookie"].decode()
        assert cookie.startswith("backprop_sess=u:") and "HttpOnly" in cookie
        assert "Secure" not in cookie.split("SameSite")[1]  # loopback: no Secure flag

    async def test_valid_session_cookie_skips_basic_check(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        secret = auth._derive_secret({"BACKPROPAGATE_UI_AUTH": "u:p"})
        cookie = auth._sign_cookie("u", secret)
        sent = await _drive(_http(extra=[(b"cookie", f"backprop_sess={cookie}".encode())]))
        assert _status(sent) == 200
        assert not any(k == b"set-cookie" for k, _ in sent[0]["headers"])

    async def test_cookie_signed_with_another_secret_is_refused(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        forged = auth._sign_cookie("u", b"attacker-secret")
        sent = await _drive(_http(extra=[(b"cookie", f"backprop_sess={forged}".encode())]))
        assert _status(sent) == 401

    async def test_production_mode_over_share_host_sets_secure_cookie(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        monkeypatch.setenv("BACKPROPAGATE_UI_SHARE_HOST", "t.example")
        sent = await _drive(_http(host="t.example",
                                  extra=[(b"authorization", _basic("u", "p").encode())]))
        assert _status(sent) == 200
        cookie = {k: v for k, v in sent[0]["headers"] if k == b"set-cookie"}[b"set-cookie"]
        assert cookie.endswith(b"Secure")

    async def test_production_mode_still_rejects_the_unlisted_host(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        monkeypatch.setenv("BACKPROPAGATE_UI_SHARE_HOST", "t.example")
        sent = await _drive(_http(host="other.example",
                                  extra=[(b"authorization", _basic("u", "p").encode())]))
        assert _status(sent) == 421


class TestTokenAutoMode:
    TOKEN = "launch-token-123"

    @pytest.fixture(autouse=True)
    def _token_env(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_LAUNCH_TOKEN", self.TOKEN)

    async def test_valid_token_redirects_to_clean_url_and_sets_cookie(self):
        sent = await _drive(_http("/runs", query=f"token={self.TOKEN}".encode()))
        assert _status(sent) == 302
        h = dict(sent[0]["headers"])
        assert h[b"location"] == b"/runs" and b"token" not in h[b"location"]
        assert h[b"cache-control"] == b"no-store"
        assert h[b"set-cookie"].startswith(b"backprop_sess=default-user:")
        assert _inner.calls == []

    async def test_redirect_cookie_is_secure_off_loopback(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_HOST_BIND", "10.0.0.9")
        sent = await _drive(_http("/", host="10.0.0.9:7860", query=f"token={self.TOKEN}".encode()))
        assert _status(sent) == 302
        assert dict(sent[0]["headers"])[b"set-cookie"].endswith(b"Secure")

    async def test_redirect_cookie_then_authenticates_the_next_request(self):
        sent = await _drive(_http(query=f"token={self.TOKEN}".encode()))
        cookie = dict(sent[0]["headers"])[b"set-cookie"].split(b";")[0]
        sent2 = await _drive(_http(extra=[(b"cookie", cookie)]))
        assert _status(sent2) == 200

    @pytest.mark.parametrize("query", [b"", b"token=", b"token=wrong", b"other=" , b"token=launch-token-12",
                                       b"token=" + b"x" * 5000])
    async def test_missing_or_wrong_token_is_401(self, query):
        sent = await _drive(_http(query=query))
        assert _status(sent) == 401

    async def test_string_query_string_is_accepted(self):
        scope = _http(query=f"token={self.TOKEN}")
        scope["query_string"] = f"token={quote(self.TOKEN)}"  # str, not bytes
        assert _status(await _drive(scope)) == 302

    async def test_basic_header_does_not_authenticate_in_token_mode(self):
        sent = await _drive(_http(extra=[(b"authorization", _basic("a", "b").encode())]))
        assert _status(sent) == 401


# =============================================================================
# Middleware: WebSocket branch (all rejections are PRE-accept)
# =============================================================================


def _ws(host="localhost:7860", origin="http://localhost:3000", cookie=None, path="/_event"):
    headers = []
    if host is not None:
        headers.append((b"host", host.encode()))
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    if cookie:
        headers.append((b"cookie", cookie.encode()))
    return {"type": "websocket", "path": path, "headers": headers}


class TestWebSocketMiddleware:
    def _assert_pre_accept_close(self, sent, code):
        assert [m["type"] for m in sent] == ["websocket.close"]  # never an accept
        assert sent[0]["code"] == code
        assert _inner.calls == []

    async def test_bad_host_closes_4404(self):
        self._assert_pre_accept_close(await _drive(_ws(host="evil.com")), 4404)

    async def test_bad_origin_closes_4403(self):
        sent = await _drive(_ws(origin="http://evil.com"))
        self._assert_pre_accept_close(sent, 4403)
        assert sent[0]["reason"] == "origin_not_allowed"

    async def test_no_auth_mode_accepts_loopback_even_without_origin(self):
        await _drive(_ws(origin=None))
        assert _inner.calls == ["websocket"]

    async def test_authed_mode_requires_a_present_origin(self, monkeypatch):
        """UI-A-008: a hand-rolled CSWSH client omitting Origin is refused."""
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        secret = auth._derive_secret({"BACKPROPAGATE_UI_AUTH": "u:p"})
        cookie = f"backprop_sess={auth._sign_cookie('u', secret)}"
        sent = await _drive(_ws(origin=None, cookie=cookie))
        self._assert_pre_accept_close(sent, 4403)
        assert sent[0]["reason"] == "origin_required"

    async def test_authed_mode_without_cookie_closes_4401(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        sent = await _drive(_ws())
        self._assert_pre_accept_close(sent, 4401)
        assert sent[0]["reason"] == "auth_required"

    async def test_authed_mode_with_forged_cookie_closes_4401(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        forged = auth._sign_cookie("u", b"wrong")
        self._assert_pre_accept_close(
            await _drive(_ws(cookie=f"backprop_sess={forged}")), 4401)

    async def test_authed_mode_with_expired_cookie_closes_4401(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        secret = auth._derive_secret({"BACKPROPAGATE_UI_AUTH": "u:p"})
        old = auth._sign_cookie("u", secret, now=time.time() - auth._COOKIE_TTL_SECONDS - 60)
        self._assert_pre_accept_close(await _drive(_ws(cookie=f"backprop_sess={old}")), 4401)

    async def test_valid_cookie_and_origin_reaches_the_app(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        secret = auth._derive_secret({"BACKPROPAGATE_UI_AUTH": "u:p"})
        cookie = f"backprop_sess={auth._sign_cookie('u', secret)}"
        await _drive(_ws(cookie=cookie))
        assert _inner.calls == ["websocket"]

    async def test_token_mode_ws_needs_the_cookie_not_the_token(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI_LAUNCH_TOKEN", "tok")
        self._assert_pre_accept_close(await _drive(_ws()), 4401)


def test_close_codes_are_distinct_application_codes():
    codes = {auth._WS_CLOSE_CODE_AUTH_FAILED, auth._WS_CLOSE_CODE_ORIGIN_FAILED,
             auth._WS_CLOSE_CODE_HOST_FAILED}
    assert codes == {4401, 4403, 4404}
