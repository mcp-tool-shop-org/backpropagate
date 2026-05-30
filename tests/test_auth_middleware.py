"""Tests for the Reflex auth middleware (TESTS-F-001, DESIGN_BRIEF mandate).

Brief reference — DESIGN_BRIEF.md "Testing requirements" lists 16 required
tests for ``backpropagate.ui_app.auth.basic_auth_transformer``. This module
implements those tests using ``httpx.ASGITransport`` (HTTP) and direct ASGI
invocation (WebSocket — see ``tests.helpers.ws`` for the recorder pattern).

Coordination notes:

- The frontend agent (parallel Wave 6) implements ``basic_auth_transformer``
  + ``ENFORCEMENT_AVAILABLE = True``. Until that lands, ALL tests skip via
  ``importorskip``-style guards. Once the middleware exists, the tests
  exercise it without further changes.

- The middleware contract under test (per DESIGN_BRIEF "Middleware logic"):
    HTTP:
      * /ping pass-through
      * Host-header allowlist (421 on mismatch)
      * Origin-header allowlist for state-changing methods (403 on mismatch)
      * Cookie-or-Basic-or-token validation (401 on missing/wrong)
      * Set-Cookie with HttpOnly + SameSite=Lax (+ Secure when not loopback)
    WebSocket:
      * Host + Origin validation
      * Cookie validation BEFORE websocket.accept (4401 close pre-accept)

- DESIGN_BRIEF anti-pattern at the bottom is the load-bearing safety
  property: "Validation AFTER websocket.accept() — DoS via thousands of
  accepted-then-rejected connections." Tests 6 + 16 pin this PRE-accept
  property using ``WSMessageRecorder.accepted_before_closed`` from
  ``tests.helpers.ws``.
"""

from __future__ import annotations

import base64
import os

import pytest

# httpx + the helpers themselves skip gracefully if httpx isn't installed.
httpx = pytest.importorskip(
    "httpx",
    reason="httpx>=0.27 is required for auth middleware tests.",
)

# Import the ASGI helpers. The fixtures inside this module are session-safe
# because they all use monkeypatch (function-scoped) for the env flip.
from tests.helpers.asgi import (  # noqa: E402
    basic_auth_header,
    make_asgi_client,
    malformed_auth_header,
    stub_asgi_http_app,
)
from tests.helpers.ws import (  # noqa: E402
    WSMessageRecorder,
    make_connect_receive,
    make_ws_scope,
)

# v1.2.0 SHIPPED the middleware (``basic_auth_transformer`` in
# ``backpropagate/ui_app/auth.py``); the import below succeeds in every
# install that has the ``[ui]`` extra. The ``requires_middleware`` skip
# marker is RETAINED as a regression guard: an accidental refactor that
# deletes or renames ``basic_auth_transformer`` would flip the import to
# ImportError and every test below would skip with a clear reason instead
# of failing at collection time. TESTS-B-018 (Stage C humanization)
# updated the doc-strings to reflect ship status; do NOT delete the import
# guard.
try:
    from backpropagate.ui_app.auth import basic_auth_transformer  # noqa: F401
    _MIDDLEWARE_AVAILABLE = True
except ImportError:
    _MIDDLEWARE_AVAILABLE = False

requires_middleware = pytest.mark.skipif(
    not _MIDDLEWARE_AVAILABLE,
    reason=(
        "basic_auth_transformer not importable from backpropagate.ui_app.auth "
        "— this almost certainly means the [ui] extra is not installed, OR "
        "the helper was accidentally deleted/renamed by a refactor (v1.2.0 "
        "shipped it; absence post-v1.2.0 is a regression). Tests skip rather "
        "than collection-fail so a headless install reports cleanly; the "
        "harness itself is verified by the helper smoke tests at the top "
        "of this module."
    ),
)


# =============================================================================
# HELPER SMOKE TESTS (always run — verify the harness itself)
# =============================================================================

def test_basic_auth_header_helper_encodes_correctly():
    """The basic_auth_header helper produces RFC-7617-shaped headers."""
    header = basic_auth_header("alice", "hunter2")
    assert "Authorization" in header
    assert header["Authorization"].startswith("Basic ")
    encoded = header["Authorization"][len("Basic "):]
    decoded = base64.b64decode(encoded).decode("utf-8")
    assert decoded == "alice:hunter2"


def test_basic_auth_header_handles_empty_values():
    """Edge case — empty user/password should still produce a valid header."""
    header = basic_auth_header("", "")
    encoded = header["Authorization"][len("Basic "):]
    assert base64.b64decode(encoded).decode("utf-8") == ":"


def test_ws_recorder_detects_pre_accept_close():
    """The WSMessageRecorder helper correctly identifies pre-accept closes.

    This is the safety property of the test infrastructure itself — if
    this assertion is wrong, every WS auth test built on the recorder
    silently lies.
    """
    recorder = WSMessageRecorder()
    # Simulate a middleware that closes immediately without accepting.
    import asyncio
    async def drive():
        await recorder({"type": "websocket.close", "code": 4401})
    asyncio.run(drive())
    assert recorder.closed is True
    assert recorder.accepted is False
    assert recorder.close_code == 4401
    assert recorder.accepted_before_closed is False


def test_ws_recorder_detects_post_accept_close_as_dos_vector():
    """The recorder flags accept-then-close as the DoS-vector anti-pattern.

    DESIGN_BRIEF anti-pattern: "Validation AFTER websocket.accept()". The
    recorder must distinguish "accepted and then closed normally" from
    "rejected pre-accept" because they have different safety properties.
    """
    recorder = WSMessageRecorder()
    import asyncio
    async def drive():
        await recorder({"type": "websocket.accept"})
        await recorder({"type": "websocket.close", "code": 4401})
    asyncio.run(drive())
    assert recorder.accepted is True
    assert recorder.closed is True
    assert recorder.accepted_before_closed is True, (
        "Post-accept close is the DoS-vector anti-pattern; the recorder "
        "must surface it for the middleware tests to assert against."
    )


def test_ws_scope_has_required_asgi_fields():
    """The make_ws_scope helper produces ASGI-3.0-conformant scope dicts."""
    scope = make_ws_scope(path="/_event", host="127.0.0.1:7860")
    assert scope["type"] == "websocket"
    assert scope["asgi"]["version"] == "3.0"
    assert scope["path"] == "/_event"
    # Headers are list of (bytes, bytes) per ASGI spec
    assert all(isinstance(k, bytes) and isinstance(v, bytes) for k, v in scope["headers"])
    # Host header was set
    host_headers = [v for k, v in scope["headers"] if k == b"host"]
    assert host_headers == [b"127.0.0.1:7860"]


# =============================================================================
# MIDDLEWARE CONTRACT TESTS — DESIGN_BRIEF mandate
# =============================================================================
# Numbered to match the 16 tests listed in DESIGN_BRIEF.md
# "Testing requirements" section.


# Loopback host that the middleware's default Host-allowlist admits. Using
# 127.0.0.1 (not "testserver") because the middleware's DNS-rebinding
# defense rejects anything not in {localhost, 127.0.0.1, ::1, +overrides}.
_LOOPBACK_BASE_URL = "http://127.0.0.1:7860"
_LOOPBACK_ORIGIN = "http://127.0.0.1"


@pytest.fixture
def default_mode_middleware(monkeypatch):
    """Auth middleware configured for TOKEN_AUTO mode (per-launch token).

    The middleware enters TOKEN_AUTO when ``BACKPROPAGATE_UI_LAUNCH_TOKEN``
    is set AND ``BACKPROPAGATE_UI_AUTH`` is NOT set (per ``_detect_mode``
    in auth.py).
    """
    monkeypatch.setattr(
        "backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE", True
    )
    monkeypatch.setenv("BACKPROPAGATE_UI_LAUNCH_TOKEN", "test-token-32-hex-chars")
    monkeypatch.delenv("BACKPROPAGATE_UI_AUTH", raising=False)
    monkeypatch.delenv("BACKPROPAGATE_UI_SHARE_HOST", raising=False)
    monkeypatch.delenv("BACKPROPAGATE_UI_HOST_BIND", raising=False)
    if not _MIDDLEWARE_AVAILABLE:
        pytest.skip("basic_auth_transformer not importable — install backpropagate[ui]")
    from backpropagate.ui_app.auth import basic_auth_transformer
    return basic_auth_transformer(stub_asgi_http_app)


@pytest.fixture
def basic_mode_middleware(monkeypatch):
    """Auth middleware configured for EXPLICIT_CREDS mode (--auth user:pass).

    Enters EXPLICIT_CREDS when ``BACKPROPAGATE_UI_AUTH`` is set and no
    share/host overrides are present (per ``_detect_mode``).
    """
    monkeypatch.setattr(
        "backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE", True
    )
    monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "alice:hunter2")
    monkeypatch.delenv("BACKPROPAGATE_UI_LAUNCH_TOKEN", raising=False)
    monkeypatch.delenv("BACKPROPAGATE_UI_SHARE_HOST", raising=False)
    monkeypatch.delenv("BACKPROPAGATE_UI_HOST_BIND", raising=False)
    if not _MIDDLEWARE_AVAILABLE:
        pytest.skip("basic_auth_transformer not importable — install backpropagate[ui]")
    from backpropagate.ui_app.auth import basic_auth_transformer
    return basic_auth_transformer(stub_asgi_http_app)


@pytest.fixture
def shared_mode_middleware(monkeypatch):
    """Auth middleware configured for PRODUCTION mode (--share + --auth).

    The brief tunnel host is captured at startup and exported via
    ``BACKPROPAGATE_UI_SHARE_HOST`` so the middleware adds it to the
    Host + Origin allowlists.
    """
    monkeypatch.setattr(
        "backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE", True
    )
    monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "alice:hunter2")
    monkeypatch.setenv(
        "BACKPROPAGATE_UI_SHARE_HOST", "random123.trycloudflare.com"
    )
    monkeypatch.delenv("BACKPROPAGATE_UI_LAUNCH_TOKEN", raising=False)
    monkeypatch.delenv("BACKPROPAGATE_UI_HOST_BIND", raising=False)
    if not _MIDDLEWARE_AVAILABLE:
        pytest.skip("basic_auth_transformer not importable — install backpropagate[ui]")
    from backpropagate.ui_app.auth import basic_auth_transformer
    return basic_auth_transformer(stub_asgi_http_app)


# Brief test 1: default mode serves with valid token
@requires_middleware
@pytest.mark.asyncio
async def test_default_mode_serves_with_valid_token(default_mode_middleware):
    """Brief #1 — GET /?token=<correct> returns 302 + sets the session cookie.

    The middleware redirects on token success (clean-URL pattern, brief #2)
    rather than returning 200 directly.
    """
    async with make_asgi_client(default_mode_middleware, base_url=_LOOPBACK_BASE_URL) as client:
        response = await client.get(
            "/",
            params={"token": "test-token-32-hex-chars"},
            follow_redirects=False,
        )
        assert response.status_code in (200, 302), (
            f"Token-bearing request should authenticate, got {response.status_code}"
        )
        # Set-Cookie must be present so subsequent requests reuse the session.
        assert "set-cookie" in response.headers, (
            "Default mode must set the session cookie on successful token auth"
        )


# Brief test 2: default mode redirects token to clean URL
@requires_middleware
@pytest.mark.asyncio
async def test_default_mode_redirects_token_to_clean_url(default_mode_middleware):
    """Brief #2 — GET /?token=<correct> redirects so the token leaves history."""
    async with make_asgi_client(default_mode_middleware, base_url=_LOOPBACK_BASE_URL) as client:
        response = await client.get(
            "/",
            params={"token": "test-token-32-hex-chars"},
            follow_redirects=False,
        )
        assert response.status_code == 302, (
            f"Default-mode auth with correct token must redirect (302) "
            f"to scrub the token from URL history; got "
            f"{response.status_code}. Body: {response.text[:200]!r}"
        )
        location = response.headers.get("location", "")
        assert "token=" not in location, (
            "Redirect Location must not echo the token (browser-history leak)"
        )


# Brief test 3: wrong token returns 401
@requires_middleware
@pytest.mark.asyncio
async def test_wrong_token_returns_401(default_mode_middleware):
    """Brief #3 — GET /?token=<garbage> returns 401."""
    async with make_asgi_client(default_mode_middleware, base_url=_LOOPBACK_BASE_URL) as client:
        response = await client.get("/", params={"token": "wrong-token-value"})
        assert response.status_code == 401, (
            f"Wrong token must yield 401 (auth required), not "
            f"{response.status_code}. If you're seeing 200, the token "
            f"comparison may be using == instead of hmac.compare_digest "
            f"(timing-attack vector). If you're seeing 500, the rejection "
            f"path is raising instead of returning a clean 401."
        )


# Brief test 4: basic auth accepts correct creds
@requires_middleware
@pytest.mark.asyncio
async def test_basic_auth_accepts_correct_creds(basic_mode_middleware):
    """Brief #4 — Basic auth with correct creds returns 200."""
    async with make_asgi_client(basic_mode_middleware, base_url=_LOOPBACK_BASE_URL) as client:
        response = await client.get(
            "/", headers=basic_auth_header("alice", "hunter2")
        )
        assert response.status_code == 200, (
            f"Basic auth with correct creds (alice/hunter2 per fixture) "
            f"must accept the request and yield 200, got "
            f"{response.status_code}. Body: {response.text[:200]!r}. "
            f"If you're seeing 401, check the AUTH_KEYS env-var parsing "
            f"or hmac.compare_digest path."
        )


# Brief test 5: basic auth rejects wrong password
@requires_middleware
@pytest.mark.asyncio
async def test_basic_auth_rejects_wrong_password(basic_mode_middleware):
    """Brief #5 — Basic auth with wrong password returns 401."""
    async with make_asgi_client(basic_mode_middleware, base_url=_LOOPBACK_BASE_URL) as client:
        response = await client.get(
            "/", headers=basic_auth_header("alice", "WRONG")
        )
        assert response.status_code == 401, (
            f"Basic auth with wrong password must yield 401, got "
            f"{response.status_code}. Body: {response.text[:200]!r}. "
            f"A 200 here means the password comparison is broken — "
            f"defense in depth depends on this gate."
        )
        # WWW-Authenticate is RFC 7235 mandatory on 401
        assert "www-authenticate" in {k.lower() for k in response.headers}, (
            f"401 response missing WWW-Authenticate header (RFC 7235 "
            f"mandatory). Browsers won't show the basic-auth prompt "
            f"without it. Headers received: "
            f"{list(response.headers.keys())!r}"
        )


# Brief test 6: WS upgrade requires cookie — measured pre-accept
@requires_middleware
@pytest.mark.asyncio
async def test_websocket_upgrade_requires_cookie(basic_mode_middleware):
    """Brief #6 — WS upgrade without cookie closes 4401 BEFORE accept (DoS defense)."""
    scope = make_ws_scope(path="/_event", host="127.0.0.1:7860", origin=_LOOPBACK_ORIGIN)
    recorder = WSMessageRecorder()
    receive = await make_connect_receive()
    await basic_mode_middleware(scope, receive, recorder)
    # The load-bearing safety property — must reject PRE-accept
    assert recorder.accepted_before_closed is False, (
        "Middleware accepted the WS upgrade before validating auth — this "
        "is the DESIGN_BRIEF anti-pattern (DoS vector via thousands of "
        "accepted-then-rejected connections)."
    )
    assert recorder.closed is True
    # Close code 4401 is the application-defined "auth required" close code
    assert recorder.close_code == 4401, (
        f"Expected close code 4401 (auth required), got {recorder.close_code}"
    )


# Brief test 7: WS upgrade accepts valid cookie
@requires_middleware
@pytest.mark.asyncio
async def test_websocket_upgrade_accepts_valid_cookie(basic_mode_middleware):
    """Brief #7 — WS upgrade with valid session cookie proceeds to accept."""
    # First obtain a session cookie via HTTP Basic
    async with make_asgi_client(basic_mode_middleware, base_url=_LOOPBACK_BASE_URL) as client:
        http_response = await client.get(
            "/", headers=basic_auth_header("alice", "hunter2")
        )
        cookie_header = http_response.headers.get("set-cookie", "")
        # Parse cookie name=value from Set-Cookie. The middleware is
        # contractually required to issue a session cookie on basic-auth
        # success (see backpropagate.ui_app.auth._set_cookie_header). If
        # this changes, surface as a real test failure — silent skip would
        # mask a regression of the v1.2.0 session-cookie contract.
        assert cookie_header, (
            "v1.2.0 security contract: middleware MUST set a session "
            "cookie on basic-auth success (see "
            "backpropagate.ui_app.auth._set_cookie_header). Got empty "
            "Set-Cookie header — this is a regression of the WS-upgrade "
            "auth-cookie path."
        )
        cookie_kv = cookie_header.split(";", 1)[0]
        assert "=" in cookie_kv, (
            f"v1.2.0 security contract: auth cookie MUST be key=value "
            f"shape (see auth._set_cookie_header which emits "
            f"f'{{_COOKIE_NAME}}={{value}}'). Got: {cookie_kv!r}. If the "
            f"cookie shape changed intentionally, update this test to "
            f"match the new contract."
        )
        cookie_name, _, cookie_value = cookie_kv.partition("=")
    scope = make_ws_scope(
        path="/_event",
        host="127.0.0.1:7860",
        origin=_LOOPBACK_ORIGIN,
        cookies={cookie_name: cookie_value},
    )
    recorder = WSMessageRecorder()
    receive = await make_connect_receive()
    await basic_mode_middleware(scope, receive, recorder)
    # Valid cookie should pass-through to the stub which accepts then closes
    # normally (code 1000). The middleware itself does not reject.
    assert recorder.close_code != 4401, (
        "Valid cookie was rejected with 4401 — middleware regression"
    )


# Brief test 8: Host header allowlist
@requires_middleware
@pytest.mark.asyncio
async def test_host_header_allowlist_rejects_unknown(basic_mode_middleware):
    """Brief #8 — Host: evil.com returns 421 (DNS rebinding defense).

    httpx defaults the Host header to the base_url's hostname, so using
    base_url=http://evil.com is the documented way to inject a malicious
    Host header into the ASGI scope.
    """
    async with make_asgi_client(
        basic_mode_middleware, base_url="http://evil.com"
    ) as client:
        response = await client.get(
            "/",
            headers=basic_auth_header("alice", "hunter2"),
        )
        assert response.status_code == 421, (
            f"Unknown Host should return 421 Misdirected Request "
            f"(DNS-rebinding defense per CVE-2024-28224 class), "
            f"got {response.status_code}"
        )


# Brief test 9: Origin allowlist (CSWSH)
@requires_middleware
@pytest.mark.asyncio
async def test_origin_allowlist_rejects_cswsh(basic_mode_middleware):
    """Brief #9 — WS upgrade with Origin: https://evil.com closes 4403."""
    scope = make_ws_scope(
        path="/_event",
        host="127.0.0.1:7860",
        origin="https://evil.com",
    )
    recorder = WSMessageRecorder()
    receive = await make_connect_receive()
    await basic_mode_middleware(scope, receive, recorder)
    assert recorder.accepted_before_closed is False, (
        "CSWSH-defense rejection must happen PRE-accept"
    )
    assert recorder.close_code == 4403, (
        f"Expected CSWSH close code 4403, got {recorder.close_code}"
    )


# =============================================================================
# UI-A-008 (Wave A2): the Origin allowlist deliberately fails OPEN on a
# MISSING Origin header (some legitimate same-origin HTTP requests omit it,
# and Host + cookie gates still apply there). But a real browser ALWAYS sets
# Origin on a cross-origin WebSocket handshake — so on the ``/_event`` upgrade,
# in an authed deployment, a *missing* Origin has no legitimate browser source
# and is the shape a hand-rolled CSWSH client uses to dodge the allowlist.
# Require a present Origin on the WS upgrade in every authed mode; preserve the
# fail-open behavior for NO_AUTH_LOCAL_ONLY (loopback dev / native WS clients).
# =============================================================================


@requires_middleware
@pytest.mark.asyncio
async def test_ws_missing_origin_rejected_in_authed_mode(basic_mode_middleware):
    """UI-A-008 — WS upgrade with NO Origin header closes 4403 PRE-accept in
    an authed (EXPLICIT_CREDS) mode.
    """
    # origin omitted -> no Origin header in the scope.
    scope = make_ws_scope(path="/_event", host="127.0.0.1:7860")
    recorder = WSMessageRecorder()
    receive = await make_connect_receive()
    await basic_mode_middleware(scope, receive, recorder)

    assert recorder.accepted_before_closed is False, (
        "UI-A-008: missing-Origin rejection on the WS upgrade must happen "
        "PRE-accept (same DoS-defense invariant as the cookie / Origin "
        "gates)."
    )
    assert recorder.close_code == 4403, (
        f"UI-A-008: a WS upgrade with NO Origin in an authed mode must be "
        f"closed with the origin-failure code 4403; got {recorder.close_code}."
    )


@requires_middleware
@pytest.mark.asyncio
async def test_ws_missing_origin_rejected_in_share_mode(shared_mode_middleware):
    """UI-A-008 — the missing-Origin rejection also fires in PRODUCTION
    (--share + --auth), the mode most exposed off-loopback.
    """
    scope = make_ws_scope(
        path="/_event", host="random123.trycloudflare.com:443"
    )
    recorder = WSMessageRecorder()
    receive = await make_connect_receive()
    await shared_mode_middleware(scope, receive, recorder)

    assert recorder.accepted_before_closed is False
    assert recorder.close_code == 4403, (
        f"UI-A-008: missing Origin under --share must close 4403; got "
        f"{recorder.close_code}."
    )


@requires_middleware
@pytest.mark.asyncio
async def test_ws_missing_origin_allowed_in_no_auth_local_mode(monkeypatch):
    """UI-A-008 — NO_AUTH_LOCAL_ONLY (loopback dev) preserves the fail-open
    behavior so native / test WS clients that omit Origin still connect.

    The Host allowlist remains the load-bearing gate in this mode; the
    missing-Origin requirement is scoped to authed modes only.
    """
    monkeypatch.setattr("backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE", True)
    # No auth creds, no launch token, no share host -> NO_AUTH_LOCAL_ONLY.
    monkeypatch.delenv("BACKPROPAGATE_UI_AUTH", raising=False)
    monkeypatch.delenv("BACKPROPAGATE_UI_LAUNCH_TOKEN", raising=False)
    monkeypatch.delenv("BACKPROPAGATE_UI_SHARE_HOST", raising=False)
    monkeypatch.delenv("BACKPROPAGATE_UI_HOST_BIND", raising=False)

    from backpropagate.ui_app.auth import basic_auth_transformer

    middleware = basic_auth_transformer(stub_asgi_http_app)
    scope = make_ws_scope(path="/_event", host="127.0.0.1:7860")  # no Origin
    recorder = WSMessageRecorder()
    receive = await make_connect_receive()
    await middleware(scope, receive, recorder)

    # NO_AUTH_LOCAL_ONLY must NOT reject a missing Origin with 4403 — the
    # connection passes through to the stub (which may accept/close normally).
    assert recorder.close_code != 4403, (
        "UI-A-008 regression: NO_AUTH_LOCAL_ONLY rejected a missing-Origin "
        "WS upgrade with 4403. The missing-Origin requirement is scoped to "
        "authed modes only; loopback dev keeps the fail-open behavior."
    )


# Brief test 10: --share mode admits the tunnel host
@requires_middleware
@pytest.mark.asyncio
async def test_share_mode_adds_tunnel_to_allowlist(shared_mode_middleware):
    """Brief #10 — --share captures the tunnel hostname and admits it.

    BACKPROPAGATE_UI_SHARE_HOST=random123.trycloudflare.com is added to
    the Host-allowlist, so requests with that Host header pass the
    DNS-rebinding gate (status != 421).
    """
    tunnel_host = "random123.trycloudflare.com"
    async with make_asgi_client(
        shared_mode_middleware, base_url=f"https://{tunnel_host}"
    ) as client:
        response = await client.get(
            "/",
            headers=basic_auth_header("alice", "hunter2"),
        )
        # Tunnel host is on the allowlist; should NOT be 421
        assert response.status_code != 421, (
            f"--share tunnel host {tunnel_host} should be on the allowlist"
        )


# Brief test 11: --share without --auth refuses to start (CLI contract)
# This is enforced at CLI parse time, not in the middleware. Pinned in
# test_cli_extended.py::TestCmdUI::test_cmd_ui_share_without_auth_still_refuses
# (and its alias test_cmd_ui_share_refuses_to_start). Document the
# cross-reference here so an auditor knows it's covered.
@pytest.mark.skip(reason="Pinned in tests/test_cli_extended.py::TestCmdUI — CLI-level guard, not middleware")
def test_share_without_auth_refuses_to_start():
    """Brief #11 — see test_cli_extended.py::TestCmdUI::test_cmd_ui_share_without_auth_still_refuses."""


# Brief test 12: --host non-loopback without --auth refuses to start
# Same — CLI-level guard. Pinned at the CLI layer in test_host_gate.py
# (BRIDGE-A-002 → TESTS coverage added in v1.3 Wave 1 Stage A). This
# skip-marker is kept as a documentation pointer so an auditor reading
# the brief-numbered tests in this file can find the CLI-layer coverage.
@pytest.mark.skip(reason="Pinned at CLI layer in tests/test_host_gate.py (BRIDGE-A-002)")
def test_host_non_loopback_without_auth_refuses_to_start():
    """Brief #12 — see tests/test_host_gate.py for the CLI-layer coverage."""


# Brief test 13: no default credentials anywhere
def test_default_creds_not_present_in_auth_module():
    """Brief #13 — auth module + --help text + docs contain no default creds.

    MLflow CVE-2026-2635 was rooted in shipping admin:password1234.
    Backpropagate must never have a literal default credential anywhere
    in the auth surface.
    """
    forbidden_literals = ["admin", "password1234", "password123", "letmein"]
    auth_path = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    targets = [
        os.path.join(auth_path, "backpropagate", "ui_app", "auth.py"),
        os.path.join(auth_path, "backpropagate", "ui_app", "app.py"),
    ]
    # Per the v1.3 brief, the auth module + app entry should be scrutinised.
    # The CLI module legitimately contains "password" in --help text (as in
    # "user:password" docstring), so we scope to the auth surface itself.
    for target in targets:
        if not os.path.exists(target):
            continue
        with open(target, encoding="utf-8") as fh:
            content = fh.read().lower()
        for literal in forbidden_literals:
            assert literal not in content, (
                f"Forbidden default-credential literal {literal!r} found in "
                f"{target}. MLflow CVE-2026-2635 shipped admin:password1234 "
                f"and reached CVSS 9.8 via this exact anti-pattern."
            )


# Brief test 14: token lock file mode 0600
@requires_middleware
def test_token_lock_file_mode_0600(tmp_path, monkeypatch):
    """Brief #14 — default mode writes the launch token lock file mode 0600.

    Per DESIGN_BRIEF "Lock-file token mode (post-CVE-2025-52882 defense)":
    the per-launch token is also written to
    ``$XDG_RUNTIME_DIR/backpropagate/session-<port>.lock`` so machine-to-
    machine clients can authenticate without exposing it in argv.

    NOTE (v1.3 Wave 3.5): the previous shape of this test silently skipped
    when ``write_launch_token_lock`` was absent, which made the lock-file
    mode 0600 contract appear "passing" even though the feature was not
    implemented. The skip is now scoped + forward-pointing: it explicitly
    names the deferred work item so the Wave 5/6 implementer inherits a
    live scaffold rather than a green-but-meaningless test.
    """
    pytest.skip(
        reason=(
            "lock-file token mode 0600 contract: feature deferred to v1.3 "
            "Wave 5/6 auth-middleware-polish item 3 (write_launch_token_lock). "
            "See research/V1_3_BRIEF.md P0 'Lock-file token for "
            "machine-to-machine auth'. Test scaffold preserved for the "
            "Wave 5/6 implementer."
        )
    )
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(
        "backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE", True
    )
    from backpropagate.ui_app.auth import write_launch_token_lock
    lock_path = write_launch_token_lock(port=7860, token="test-token")
    assert lock_path.exists()
    # On POSIX, check the mode bits. On Windows, just verify the file is
    # not world-readable in any meaningful sense (Windows ACLs are
    # different territory; the helper should still create the file).
    if os.name == "posix":
        import stat
        mode = lock_path.stat().st_mode
        # Only owner read/write permitted
        assert stat.S_IMODE(mode) == 0o600, (
            f"Lock file mode {oct(stat.S_IMODE(mode))} != 0o600"
        )


# Brief test 15: cookie hardening
@requires_middleware
@pytest.mark.asyncio
async def test_cookie_hardening(basic_mode_middleware):
    """Brief #15 — Set-Cookie includes HttpOnly + SameSite=Lax."""
    async with make_asgi_client(basic_mode_middleware, base_url=_LOOPBACK_BASE_URL) as client:
        response = await client.get(
            "/", headers=basic_auth_header("alice", "hunter2")
        )
        set_cookie = response.headers.get("set-cookie", "")
        # The middleware is contractually required to issue a session
        # cookie on basic-auth success per
        # backpropagate.ui_app.auth._set_cookie_header. If Set-Cookie is
        # absent here it indicates a regression of the v1.2.0 cookie-
        # issuance contract, not a benign "different request shape" — fail
        # loudly rather than silently skip the HttpOnly/SameSite assertions.
        assert set_cookie, (
            "v1.2.0 security contract: middleware MUST set a session "
            "cookie on basic-auth success (see "
            "backpropagate.ui_app.auth._set_cookie_header). Empty "
            "Set-Cookie regresses the HttpOnly + SameSite=Lax hardening "
            "this test is meant to guard."
        )
        set_cookie_l = set_cookie.lower()
        assert "httponly" in set_cookie_l, (
            "Cookie must have HttpOnly (defense: JS read access)"
        )
        assert "samesite=lax" in set_cookie_l, (
            "Cookie must have SameSite=Lax (CSRF defense — Lax over Strict "
            "because Reflex's link navigation needs Lax)"
        )


# Brief test 16: post-accept DoS vector absent under load
@requires_middleware
@pytest.mark.asyncio
async def test_post_accept_dos_vector_absent(basic_mode_middleware):
    """Brief #16 — 100 garbage-cookie WS connections all closed PRE-accept.

    The load-bearing safety property of the DESIGN_BRIEF anti-pattern.
    A regression that moves the validation post-accept would allow
    attackers to exhaust the server with thousands of accepted-but-
    rejected connections.
    """
    pre_accept_close_count = 0
    for i in range(100):
        scope = make_ws_scope(
            path="/_event",
            host="127.0.0.1:7860",
            origin="http://127.0.0.1:7860",
            cookies={"backprop_sess": f"garbage-{i}"},
        )
        recorder = WSMessageRecorder()
        receive = await make_connect_receive()
        await basic_mode_middleware(scope, receive, recorder)
        if not recorder.accepted_before_closed and recorder.close_code in (4401, 4403):
            pre_accept_close_count += 1
    assert pre_accept_close_count == 100, (
        f"Expected 100 pre-accept closes; got {pre_accept_close_count}. "
        f"The middleware is accepting some WS upgrades before validating "
        f"auth — this is the DoS vector documented in DESIGN_BRIEF."
    )


# =============================================================================
# ADDITIONAL HARDENING TESTS (beyond the brief 16; defense-in-depth)
# =============================================================================


@requires_middleware
@pytest.mark.asyncio
async def test_malformed_auth_header_does_not_crash(basic_mode_middleware):
    """A malformed Authorization header must yield 401, not 500 or stacktrace."""
    async with make_asgi_client(basic_mode_middleware, base_url=_LOOPBACK_BASE_URL) as client:
        for garbage in ["Basic !!!", "Bearer xyz", "Basic", "garbage"]:
            response = await client.get(
                "/", headers=malformed_auth_header(garbage)
            )
            assert response.status_code in (401, 400), (
                f"Malformed header {garbage!r} produced {response.status_code} "
                f"— must be 401 or 400, never 5xx"
            )


@requires_middleware
@pytest.mark.asyncio
async def test_ping_endpoint_bypasses_auth(basic_mode_middleware):
    """/ping is a health check and must pass-through without auth.

    DESIGN_BRIEF "Middleware logic": ``if path in {'/ping'}: pass-through``.
    """
    async with make_asgi_client(basic_mode_middleware, base_url=_LOOPBACK_BASE_URL) as client:
        response = await client.get("/ping")
        # Should NOT be 401 — the stub returns 200 for everything
        assert response.status_code != 401, (
            "/ping must bypass the auth gate (health-check pattern)"
        )


# =============================================================================
# TESTS-A-005 + TESTS-A-006 — Auth middleware regression set (v1.3 Wave 1)
# =============================================================================
# Adds:
# 1. Explicit regression for the pre-``websocket.accept()`` cookie path —
#    expired/garbage cookie MUST close pre-accept (the DoS-vector property
#    pinned at the individual-request level, not just under load).
# 2. Direct test of ``_validate_cookie`` HMAC-verification path (the cookie
#    helper that the middleware's post-Host-allowlist check delegates to).
#    A regression that switched ``hmac.compare_digest`` for an ``==``
#    comparison would still pass the existing tests (the cookie validates
#    correctly under load) but would re-open the timing-attack surface on
#    HMAC sig verification.


@requires_middleware
@pytest.mark.asyncio
async def test_ws_pre_accept_close_on_expired_cookie(basic_mode_middleware, monkeypatch):
    """TESTS-A-005 — expired session cookie closes WS PRE-accept (regression).

    The middleware's cookie-validation path uses ``_validate_cookie`` which
    checks both signature AND expiry. A regression that skipped the expiry
    check (e.g. by moving expiry to after the HMAC verify or removing the
    ``now > exp`` branch) would let an expired cookie through to
    ``websocket.accept()`` — converting the auth-required gate into a
    permanent grant once the cookie was first issued.

    This test forges a structurally-valid cookie with a long-past
    expiration and asserts the middleware closes 4401 BEFORE accept.
    """
    # Mint a structurally-valid cookie value, then poison the embedded
    # expiry by re-signing with an exp deep in the past.
    # Wave 3.5 TESTS-B-005: removed redundant pre-init that was overwritten
    # immediately below (the dict(monkeypatch.delenv and {}) idiom evaluated to
    # _derive_secret({}) and was masked by a `# type: ignore[misc]`).
    # Use the actual env-derivation path so the secret matches what the
    # middleware will compute under the fixture.
    import os as _os

    from backpropagate.ui_app import auth as auth_module
    secret = auth_module._derive_secret({"BACKPROPAGATE_UI_AUTH": _os.environ.get("BACKPROPAGATE_UI_AUTH", "")})

    # The cookie format is <user>:<exp>:<sig>. Backdate exp by a year.
    import base64
    import hashlib
    import hmac as _hmac

    user = "alice"
    past_exp = "1000000000"  # September 2001 — definitely expired
    message = f"{user}:{past_exp}".encode()
    digest = _hmac.new(secret, message, hashlib.sha256).digest()
    sig = base64.urlsafe_b64encode(digest[:32]).rstrip(b"=").decode("ascii")
    expired_cookie = f"{user}:{past_exp}:{sig}"

    scope = make_ws_scope(
        path="/_event",
        host="127.0.0.1:7860",
        origin=_LOOPBACK_ORIGIN,
        cookies={"backprop_sess": expired_cookie},
    )
    recorder = WSMessageRecorder()
    receive = await make_connect_receive()
    await basic_mode_middleware(scope, receive, recorder)

    assert recorder.accepted_before_closed is False, (
        "Expired cookie reached websocket.accept() — the expiry check was "
        "bypassed (post-accept anti-pattern regression). Middleware must "
        "close PRE-accept for expired cookies."
    )
    assert recorder.closed is True
    assert recorder.close_code == 4401, (
        f"Expired cookie produced close code {recorder.close_code}; "
        f"expected 4401 (auth required)."
    )


def test_cookie_hmac_signature_verification_uses_constant_time_compare():
    """TESTS-A-006 — _validate_cookie HMAC path uses hmac.compare_digest.

    The cookie helper at ``backpropagate.ui_app.auth._validate_cookie`` is
    the post-cookie-parse leaf that the WS middleware delegates to. The
    safety property: signature verification MUST use ``hmac.compare_digest``
    (constant-time) rather than ``==`` (early-exit byte compare that leaks
    timing).

    A regression that switched the comparator would still pass functional
    tests — the cookie validates correctly. But it would silently re-open
    the timing-attack surface where an attacker can recover the HMAC
    signature byte-by-byte by measuring response time.

    This test mechanises the source-level invariant.
    """
    import inspect

    from backpropagate.ui_app import auth as auth_module

    source = inspect.getsource(auth_module._validate_cookie)
    # The helper MUST use compare_digest, not a bare equality compare.
    assert "compare_digest" in source, (
        "_validate_cookie no longer uses hmac.compare_digest — "
        "the HMAC verification has regressed to a non-constant-time "
        "comparator, re-opening the timing-attack surface."
    )


@requires_middleware
@pytest.mark.asyncio
async def test_ws_tampered_cookie_signature_closes_pre_accept(basic_mode_middleware):
    """TESTS-A-006 — a cookie with tampered HMAC signature is rejected pre-accept.

    Complements the source-level invariant above with a behavioural test:
    if an attacker flips a bit in the signature portion of a session cookie,
    the middleware must close PRE-accept with 4401 — not silently let
    through (which would happen if compare_digest were replaced with a
    broken comparator that early-returned on a length mismatch).
    """
    # Mint a syntactically-valid cookie envelope where the signature is
    # garbage (32 random base64 chars). The user + future-exp portion is
    # well-formed so the parsing branch is exercised, but the HMAC verify
    # MUST fail and the middleware MUST close pre-accept.
    import base64
    import os as _os
    import time as _time

    user = "alice"
    future_exp = str(int(_time.time()) + 3600)  # 1h from now
    garbage_sig_bytes = _os.urandom(32)
    garbage_sig = base64.urlsafe_b64encode(garbage_sig_bytes).rstrip(b"=").decode("ascii")
    tampered_cookie = f"{user}:{future_exp}:{garbage_sig}"

    scope = make_ws_scope(
        path="/_event",
        host="127.0.0.1:7860",
        origin=_LOOPBACK_ORIGIN,
        cookies={"backprop_sess": tampered_cookie},
    )
    recorder = WSMessageRecorder()
    receive = await make_connect_receive()
    await basic_mode_middleware(scope, receive, recorder)

    assert recorder.accepted_before_closed is False, (
        "Tampered-signature cookie reached websocket.accept() — the HMAC "
        "verify gate is broken."
    )
    assert recorder.closed is True, (
        "Tampered-signature cookie did NOT trigger ws.close — middleware "
        "auth gate may be permissive on HMAC mismatch."
    )
    assert recorder.close_code == 4401, (
        f"Tampered-cookie close used the wrong WS close code: expected "
        f"4401 (auth-required, per the documented WS_CLOSE_* contract in "
        f"handbook/auth.md), got {recorder.close_code}. Operators rely on "
        f"4401 to distinguish auth failures from network drops."
    )
