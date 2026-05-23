"""Auth middleware for the Reflex web UI.

Implements the v1.2.0 DESIGN_BRIEF auth contract (Wave 6, Option B MVP):

- Three operator-facing modes:
    1. ``no_auth_local_only``  — no flags; loopback bind; no enforcement
       (matches the v1.1.x "naked localhost" path — preserved for back-compat
       so smoke-importing the app at module load doesn't 401 on first hit).
    2. ``token_auto``          — no ``--auth``; loopback bind; per-launch
       random token printed via the CLI startup banner (v1.3 polish; the
       middleware accepts ``?token=<hex>`` on the URL today and sets the
       session cookie on success).
    3. ``explicit_creds``      — ``--auth user:pass``; loopback bind; HTTP
       Basic on first hit, HMAC-signed session cookie thereafter.
    4. ``production``          — ``--share`` OR ``--host <non-loopback>``;
       requires ``--auth``; tunnel/LAN host added to the Host-header allowlist.

- Single ASGI middleware installed via ``rx.App(api_transformer=...)`` that
  gates BOTH HTTP routes AND the ``/_event`` WebSocket upgrade (the documented
  Reflex >=0.8 hook — see ``research/reflex-auth-middleware.md``).

- Cookie session: HMAC(``<user>:<exp>``) signed with SHA-256(``BACKPROPAGATE_UI_AUTH``)
  for explicit_creds mode, or the launch-token bytes for token_auto mode.
  ``HttpOnly`` + ``SameSite=Lax`` + ``Secure`` when non-loopback + 12h expiry.

- WS auth: cookie validated BEFORE ``websocket.accept()``; close code 4401 on
  failure (load-bearing — post-accept validation is a documented DoS vector
  per Peter Braden + Hexshift + dev.to consensus; see
  ``research/websocket-auth-failure-modes.md``).

- Defense-in-depth: Host-header allowlist (DNS-rebinding defense; CVE-2024-28224
  class), Origin allowlist on WS upgrade + state-changing HTTP methods (CSWSH
  defense; CWE-1385).

This MVP defers to v1.3:
- Footer auth-badge UI (FRONTEND-F-FOOTER-AUTH-BADGE)
- Jupyter-pattern startup banner (FRONTEND-F-STARTUP-BANNER-JUPYTER-PATTERN)
- Lock-file token at ``$XDG_RUNTIME_DIR/backpropagate/session-<port>.lock``
  (FRONTEND-F-LOCK-FILE-TOKEN)
- ``--auth-file <path>`` secure-variant flag (FRONTEND-F-AUTH-FILE-FLAG)
- Request-logging middleware (FRONTEND-F-MIDDLEWARE-REQUEST-LOGGING)
- Rate-limit middleware (FRONTEND-F-MIDDLEWARE-RATE-LIMIT)
"""

from __future__ import annotations

import base64
import enum
import hashlib
import hmac
import logging
import os
import secrets
import time
from collections.abc import Callable
from http import HTTPStatus
from urllib.parse import parse_qs, urlsplit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level flag — gates the Wave 3.5 belt-and-suspenders refuse-to-start
# checks in ``ui_app/app.py`` and ``rxconfig.py``. Flipped from False (v1.1.x)
# to True in Wave 6 of the v1.2.0 dogfood swarm. With the middleware below
# wired into ``rx.App(api_transformer=...)``, the CLI's --auth path now has a
# real enforcement layer — the belt-and-suspenders refuse-to-start becomes
# inert (the ``not ENFORCEMENT_AVAILABLE`` condition is False) but is left in
# place as a regression guard.
# ---------------------------------------------------------------------------
ENFORCEMENT_AVAILABLE: bool = True
"""True when the Reflex UI actually enforces ``BACKPROPAGATE_UI_AUTH``.

Wave-6 flip checklist (so the next maintainer doesn't half-flip):

1. ``basic_auth_transformer`` is wired in ``ui_app/app.py`` via
   ``rx.App(api_transformer=basic_auth_transformer)``.
2. The CLI's ``cli.py:cmd_ui`` refuse-to-start path for ``--auth`` is INVERTED
   (``--auth`` is now allowed; ``--share`` without ``--auth`` and ``--host
   <non-loopback>`` without ``--auth`` are the surviving hard errors).
3. The Wave-3.5 belt-and-suspenders refuse-to-start in ``ui_app/app.py`` +
   ``rxconfig.py`` is left in place but becomes inert (``not
   ENFORCEMENT_AVAILABLE`` is False).
4. ``cors_allowed_origins`` in ``rxconfig.py`` is locked to the loopback
   allowlist (FRONTEND-F-CORS-ORIGINS-LOCK).
5. CHANGELOG.md gets a v1.2.0 entry documenting the flip + middleware shape;
   SECURITY.md gets the DESIGN_BRIEF threat-model paragraph.
6. GHSA follow-up note: "v1.2.0 introduces real authentication enforcement;
   v1.1.x mitigations (refuse-to-start) remain valid as defense-in-depth on
   misconfiguration."
"""


# ---------------------------------------------------------------------------
# Configuration knobs (read from env once at first middleware invocation; the
# CLI exports them in ``cli.py:cmd_ui`` before launching the Reflex subprocess)
# ---------------------------------------------------------------------------

# Env vars consumed by this module:
#
# - ``BACKPROPAGATE_UI_AUTH``       — explicit_creds mode "user:pass"
# - ``BACKPROPAGATE_UI_PORT``       — port the operator passed to ``backprop ui``
#                                     (used to populate the Host-header
#                                     allowlist with the right loopback:port
#                                     combinations)
# - ``BACKPROPAGATE_UI_AUTH_MODE``  — v1.3 hand-off: ``token`` / ``basic`` /
#                                     ``shared`` / ``network`` / ``insecure``
#                                     (Wave 6 only consumes "basic" for the
#                                     explicit_creds branch; the rest are
#                                     forward-compatible for the v1.3 badge)
# - ``BACKPROPAGATE_UI_SHARE_HOST`` — v1.3 hand-off: extra Host-header entry
#                                     for ``--share`` mode (e.g.
#                                     ``random123.trycloudflare.com``)
# - ``BACKPROPAGATE_UI_HOST_BIND``  — v1.3 hand-off: bind address from
#                                     ``--host <addr>`` (used to populate the
#                                     Host-header allowlist with the LAN IP)
# - ``BACKPROPAGATE_UI_LAUNCH_TOKEN`` — token_auto mode launch token (the CLI
#                                     generates this if ``--auth`` is absent
#                                     and exports it). v1.3 polish item.

_COOKIE_NAME = "backprop_sess"
_COOKIE_TTL_SECONDS = 12 * 60 * 60  # 12 hours; matches DESIGN_BRIEF
_COOKIE_REALM = "backpropagate"
_WS_CLOSE_CODE_AUTH_FAILED = 4401  # Application-level "auth failed"
_WS_CLOSE_CODE_ORIGIN_FAILED = 4403  # Application-level "forbidden origin"
_LAUNCH_TOKEN_ENV = "BACKPROPAGATE_UI_LAUNCH_TOKEN"


class AuthMode(enum.Enum):
    """Operator-facing modes from the DESIGN_BRIEF matrix."""

    NO_AUTH_LOCAL_ONLY = "no_auth_local_only"
    TOKEN_AUTO = "token_auto"
    EXPLICIT_CREDS = "explicit_creds"
    PRODUCTION = "production"


def _detect_mode(env: dict[str, str] | None = None) -> AuthMode:
    """Pick mode from environment + bind address hints.

    Order (most-specific first):

    1. ``BACKPROPAGATE_UI_AUTH`` set + ``BACKPROPAGATE_UI_SHARE_HOST`` set →
       PRODUCTION (--share + --auth).
    2. ``BACKPROPAGATE_UI_AUTH`` set + ``BACKPROPAGATE_UI_HOST_BIND`` set
       (non-loopback) → PRODUCTION (--host + --auth).
    3. ``BACKPROPAGATE_UI_AUTH`` set → EXPLICIT_CREDS (loopback bind).
    4. ``BACKPROPAGATE_UI_LAUNCH_TOKEN`` set → TOKEN_AUTO (v1.3 polish; the
       CLI doesn't generate this in Wave 6 MVP but the middleware honors it
       if present).
    5. Otherwise → NO_AUTH_LOCAL_ONLY (back-compat; smoke-import / dev runs).
    """
    env = env if env is not None else dict(os.environ)
    auth_creds = env.get("BACKPROPAGATE_UI_AUTH", "").strip()
    share_host = env.get("BACKPROPAGATE_UI_SHARE_HOST", "").strip()
    host_bind = env.get("BACKPROPAGATE_UI_HOST_BIND", "").strip().lower()
    launch_token = env.get(_LAUNCH_TOKEN_ENV, "").strip()

    if auth_creds:
        if share_host:
            return AuthMode.PRODUCTION
        if host_bind and host_bind not in ("", "localhost", "127.0.0.1", "::1"):
            return AuthMode.PRODUCTION
        return AuthMode.EXPLICIT_CREDS

    if launch_token:
        return AuthMode.TOKEN_AUTO

    return AuthMode.NO_AUTH_LOCAL_ONLY


def _derive_secret(env: dict[str, str] | None = None) -> bytes:
    """Cookie HMAC secret derivation per DESIGN_BRIEF.

    - ``--auth`` set: SHA-256(BACKPROPAGATE_UI_AUTH) — operator-supplied key
      material; no separate ``BACKPROP_AUTH_SECRET`` env var needed.
    - ``BACKPROPAGATE_UI_LAUNCH_TOKEN`` set: token bytes used directly.
    - Otherwise: stable per-process random bytes (NO_AUTH_LOCAL_ONLY mode
      doesn't validate cookies, but the secret needs to be non-empty so the
      hmac.new() call doesn't raise).
    """
    env = env if env is not None else dict(os.environ)
    auth_creds = env.get("BACKPROPAGATE_UI_AUTH", "").strip()
    launch_token = env.get(_LAUNCH_TOKEN_ENV, "").strip()
    if auth_creds:
        return hashlib.sha256(auth_creds.encode("utf-8")).digest()
    if launch_token:
        return launch_token.encode("utf-8")
    # Stable per-process secret. Calling code in NO_AUTH_LOCAL_ONLY mode
    # short-circuits before ever validating a cookie, so this just keeps the
    # hmac primitives happy.
    return _PROCESS_LOCAL_SECRET


_PROCESS_LOCAL_SECRET = secrets.token_bytes(32)


def _verify_basic_auth(authorization_header: str, env: dict[str, str] | None = None) -> str | None:
    """Constant-time check of HTTP Basic against ``BACKPROPAGATE_UI_AUTH``.

    Returns the authenticated username on success, ``None`` on failure. We
    use ``hmac.compare_digest`` for both halves so timing doesn't leak the
    correct username when the password is wrong (and vice versa).
    """
    env = env if env is not None else dict(os.environ)
    expected = env.get("BACKPROPAGATE_UI_AUTH", "").strip()
    if not expected:
        return None

    # ``Authorization: Basic <b64(user:pass)>`` — strip casing/whitespace
    # robustly because some proxies normalize the scheme.
    if not authorization_header:
        return None
    parts = authorization_header.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "basic":
        return None

    try:
        decoded = base64.b64decode(parts[1], validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None

    if ":" not in decoded:
        return None
    user, password = decoded.split(":", 1)
    if ":" not in expected:
        # Malformed BACKPROPAGATE_UI_AUTH; the CLI validates shape upstream so
        # this shouldn't happen, but fail-closed if it does.
        return None
    exp_user, exp_pass = expected.split(":", 1)

    # Constant-time compare both halves. Different lengths are short-circuited
    # by hmac.compare_digest itself (still constant-time per Python docs).
    user_ok = hmac.compare_digest(user.encode("utf-8"), exp_user.encode("utf-8"))
    pass_ok = hmac.compare_digest(password.encode("utf-8"), exp_pass.encode("utf-8"))
    if user_ok and pass_ok:
        return user
    return None


def _sign_cookie(user: str, secret: bytes, now: float | None = None) -> str:
    """HMAC-signed session cookie payload ``<user>:<exp>:<sig>``.

    ``exp`` is unix seconds (int) — 12h from ``now``. ``sig`` is base64-url
    of ``HMAC-SHA256(secret, "<user>:<exp>")`` truncated to 32 bytes.

    The wire format is deliberately ``:``-separated (not JSON) so an attacker
    crafting a forged value cannot smuggle a structured payload past the
    HMAC verification — the parsing on the receive side is "split on ':' from
    the right twice; everything else is the user value." Usernames with ``:``
    in them are accepted (the split walks from the right).
    """
    if now is None:
        now = time.time()
    exp = int(now) + _COOKIE_TTL_SECONDS
    message = f"{user}:{exp}".encode()
    digest = hmac.new(secret, message, hashlib.sha256).digest()
    sig = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"{user}:{exp}:{sig}"


def _validate_cookie(
    cookie_value: str, secret: bytes, now: float | None = None
) -> str | None:
    """Verify cookie HMAC + expiry; return the username or ``None``.

    Constant-time signature compare via ``hmac.compare_digest``.
    """
    if not cookie_value:
        return None
    if now is None:
        now = time.time()

    # Split from the right because usernames may contain ``:``.
    try:
        rest, sig = cookie_value.rsplit(":", 1)
        user, exp_str = rest.rsplit(":", 1)
    except ValueError:
        return None

    try:
        exp = int(exp_str)
    except ValueError:
        return None

    if exp < now:
        return None

    message = f"{user}:{exp}".encode()
    expected_sig = hmac.new(secret, message, hashlib.sha256).digest()
    expected_b64 = base64.urlsafe_b64encode(expected_sig).rstrip(b"=").decode("ascii")
    if not hmac.compare_digest(expected_b64.encode("ascii"), sig.encode("ascii")):
        return None
    return user


def _host_allowlist(env: dict[str, str] | None = None) -> set[str]:
    """Compute the Host-header allowlist for the current mode.

    Defaults: ``localhost`` and ``127.0.0.1`` at any port. Production mode
    adds the operator-supplied ``BACKPROPAGATE_UI_SHARE_HOST`` (--share
    tunnel) and ``BACKPROPAGATE_UI_HOST_BIND`` (--host LAN IP).

    Host header may include ``:port`` — we strip it before compare. Bare IPv6
    addresses are wrapped in ``[]`` per RFC 3986; we strip the brackets too.
    """
    env = env if env is not None else dict(os.environ)
    allowed = {"localhost", "127.0.0.1", "::1"}
    share_host = env.get("BACKPROPAGATE_UI_SHARE_HOST", "").strip()
    if share_host:
        allowed.add(share_host.lower())
    host_bind = env.get("BACKPROPAGATE_UI_HOST_BIND", "").strip().lower()
    if host_bind and host_bind not in ("0.0.0.0", "::"):
        # 0.0.0.0 / :: are wildcards, never legitimate Host header values.
        allowed.add(host_bind)
    return allowed


def _origin_allowlist(env: dict[str, str] | None = None) -> set[str]:
    """Compute the Origin allowlist for the current mode.

    The Origin allowlist contains scheme+host combinations (no port) — the
    same defaults as rxconfig.py's cors_allowed_origins, plus the operator's
    --share / --host overrides. Origin format per RFC 6454 §6.1 is
    ``scheme://host[:port]``; we compare on scheme+host with port-tolerant
    matching (we trust the cors_allowed_origins layer for strict port pinning).
    """
    env = env if env is not None else dict(os.environ)
    allowed = {
        "http://localhost",
        "http://127.0.0.1",
        "https://localhost",
        "https://127.0.0.1",
    }
    share_host = env.get("BACKPROPAGATE_UI_SHARE_HOST", "").strip()
    if share_host:
        # Tunnel hosts are virtually always HTTPS (cloudflared/ngrok), but
        # accept both schemes — operators occasionally test over plain HTTP.
        allowed.add(f"http://{share_host}")
        allowed.add(f"https://{share_host}")
    host_bind = env.get("BACKPROPAGATE_UI_HOST_BIND", "").strip().lower()
    if host_bind and host_bind not in ("0.0.0.0", "::", "", "localhost", "127.0.0.1", "::1"):
        allowed.add(f"http://{host_bind}")
        allowed.add(f"https://{host_bind}")
    return allowed


def _host_matches_allowlist(host_header: str, allowlist: set[str]) -> bool:
    """Compare a ``Host`` header (which may include ``:port``) against the set."""
    if not host_header:
        return False
    bare = host_header.strip().lower()
    # Strip port.
    if bare.startswith("["):
        # IPv6: [::1]:7860 -> ::1
        end = bare.find("]")
        if end > 0:
            bare = bare[1:end]
    else:
        if ":" in bare:
            bare = bare.split(":", 1)[0]
    return bare in {a.lower() for a in allowlist}


def _origin_matches_allowlist(origin_header: str, allowlist: set[str]) -> bool:
    """Compare an ``Origin`` header against the set (scheme+host, port-tolerant)."""
    if not origin_header:
        # Some same-origin requests legitimately omit Origin (older browsers,
        # native fetches); we allow that — the Host check above is the load-
        # bearing gate. CSWSH attacks always carry an Origin header because
        # the browser sets it for cross-origin requests.
        return True
    try:
        parts = urlsplit(origin_header.strip())
    except (ValueError, AttributeError):
        return False
    if not parts.scheme or not parts.hostname:
        return False
    needle = f"{parts.scheme}://{parts.hostname}".lower()
    return needle in {a.lower() for a in allowlist}


def _parse_cookies(cookie_header: str) -> dict[str, str]:
    """Minimal RFC 6265 cookie-header parser (no SimpleCookie dependency).

    Returns a dict of ``name -> value``. Quoted values are unwrapped. Last
    occurrence wins on duplicate names.
    """
    result: dict[str, str] = {}
    if not cookie_header:
        return result
    for pair in cookie_header.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, _, value = pair.partition("=")
        name = name.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        if name:
            result[name] = value
    return result


def _set_cookie_header(value: str, secure: bool) -> bytes:
    """Build the Set-Cookie header bytes for the session cookie."""
    parts = [
        f"{_COOKIE_NAME}={value}",
        "Path=/",
        "HttpOnly",
        "SameSite=Lax",
        f"Max-Age={_COOKIE_TTL_SECONDS}",
    ]
    if secure:
        parts.append("Secure")
    return "; ".join(parts).encode("ascii")


def _is_loopback_host(host_header: str) -> bool:
    """True if Host header is a loopback address (drives Secure cookie flag)."""
    if not host_header:
        return True  # Conservative: don't add Secure for missing Host
    bare = host_header.strip().lower()
    if bare.startswith("["):
        end = bare.find("]")
        if end > 0:
            bare = bare[1:end]
    else:
        if ":" in bare:
            bare = bare.split(":", 1)[0]
    return bare in ("localhost", "127.0.0.1", "::1")


def _header_value(headers: list[tuple[bytes, bytes]], name: bytes) -> str:
    """Find an ASGI scope header (lowercase name)."""
    name_l = name.lower()
    for k, v in headers:
        if k.lower() == name_l:
            try:
                return v.decode("latin-1")
            except (UnicodeDecodeError, AttributeError):
                return ""
    return ""


def _build_401_response(realm: str = _COOKIE_REALM, hint: str | None = None) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    """Build the 401 body + headers (HTTP-Basic challenge).

    Returns ``(body, headers)``. The body is the operator-facing message from
    DESIGN_BRIEF §Operator UX → Error messages.
    """
    default_hint = (
        "Authentication required. If you launched without --auth, paste the "
        "URL from the startup banner (includes ?token=...). If you launched "
        "with --auth, supply username and password."
    )
    body = (hint or default_hint).encode("utf-8")
    headers = [
        (b"content-type", b"text/plain; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"www-authenticate", f'Basic realm="{realm}"'.encode("ascii")),
        (b"cache-control", b"no-store"),
    ]
    return body, headers


def _build_421_response(host: str) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    """Build the 421 Misdirected Request body for Host-header mismatch."""
    body = (
        f"421 Misdirected Request: Host header '{host}' is not in the "
        "allowlist for this backpropagate UI instance (DNS-rebinding defense)."
    ).encode()
    return body, [
        (b"content-type", b"text/plain; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]


def _build_403_response(reason: str) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    """Build the 403 body for Origin-header mismatch (CSWSH defense)."""
    body = (
        f"403 Forbidden: {reason} (CSWSH defense, CWE-1385)."
    ).encode()
    return body, [
        (b"content-type", b"text/plain; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]


# Reflex's reserved/passthrough paths that should NOT require auth even with
# enforcement enabled. ``/ping`` is the orchestration health-check; the upload
# and event WebSocket are reflex-internal but still go through the WS auth path
# above for ``/_event``. The asset/JS/CSS paths under ``/_next`` are SPA static
# delivery; we let them through (the gating happens on the API/WS layer).
_PASSTHROUGH_PATHS = (
    "/ping",
    "/_next/",  # Next.js static assets (Reflex bundles its SPA via Next)
    "/favicon",  # /favicon.ico and friends
)


def _is_passthrough_http(path: str) -> bool:
    """Whether an HTTP request path bypasses the auth gate."""
    if not path:
        return False
    return any(path == prefix or path.startswith(prefix) for prefix in _PASSTHROUGH_PATHS)


def basic_auth_transformer(asgi_app: Callable) -> Callable:
    """ASGI middleware factory — wraps the Reflex app with the auth gate.

    Handles three ASGI scope types:

    - ``http``: HTTP Basic check, Host-header allowlist, Origin allowlist on
      state-changing methods. On success, sets the HMAC-signed session cookie
      so subsequent requests skip the Basic check.
    - ``websocket``: Host + Origin validation, cookie HMAC validation —
      ALL BEFORE the upstream Reflex app's ``websocket.accept()``. On failure,
      sends ``websocket.close`` with code 4401 (auth) or 4403 (origin).
    - ``lifespan``: pass through unchanged.

    Pass-through paths (``/ping`` health check, ``/_next/`` SPA assets,
    ``/favicon``) bypass the auth gate so orchestration and static-asset
    delivery work without credentials.

    No-auth-local-only mode (no ``BACKPROPAGATE_UI_AUTH``,
    no ``BACKPROPAGATE_UI_LAUNCH_TOKEN``) is the v1.1.x back-compat path: the
    middleware applies the Host-header allowlist (loopback-only) but does NOT
    require credentials. The CLI's refuse-to-start rails are the gate that
    keeps this mode loopback-bound.
    """

    async def middleware(scope: dict, receive: Callable, send: Callable) -> None:
        scope_type = scope.get("type")

        if scope_type == "lifespan":
            # Pass through unchanged — lifespan is server startup/shutdown,
            # no per-request auth concept.
            await asgi_app(scope, receive, send)
            return

        # Re-detect the mode on every request so env-var changes take effect
        # without a process restart (cheap; just dict lookups). For very hot
        # paths the JIT cache hides this; for the auth path we're already
        # doing HMAC work that dwarfs the env reads.
        env = dict(os.environ)
        mode = _detect_mode(env)
        headers = scope.get("headers") or []
        host_header = _header_value(headers, b"host")
        origin_header = _header_value(headers, b"origin")
        host_allow = _host_allowlist(env)
        origin_allow = _origin_allowlist(env)

        # ---- HTTP branch -------------------------------------------------
        if scope_type == "http":
            path = scope.get("path", "")
            method = (scope.get("method") or "GET").upper()

            # Pass-through for orchestration + SPA static assets.
            if _is_passthrough_http(path):
                await asgi_app(scope, receive, send)
                return

            # Host-header allowlist (DNS-rebinding defense). Fires in EVERY
            # mode, including no_auth_local_only — that's the load-bearing
            # localhost-is-not-a-boundary defense per CVE-2024-28224.
            if not _host_matches_allowlist(host_header, host_allow):
                logger.warning(
                    "auth: rejected request with disallowed Host header",
                    extra={"host": host_header, "path": path, "method": method},
                )
                body, h421 = _build_421_response(host_header)
                await send({
                    "type": "http.response.start",
                    "status": int(HTTPStatus.MISDIRECTED_REQUEST),
                    "headers": h421,
                })
                await send({"type": "http.response.body", "body": body})
                return

            # Origin allowlist on state-changing methods only. GET/HEAD/OPTIONS
            # are read-only / preflight — the CORS layer + cookie SameSite=Lax
            # handle them; CSWSH defense requires Origin pinning on the
            # mutation surface.
            if method in ("POST", "PUT", "PATCH", "DELETE"):
                if not _origin_matches_allowlist(origin_header, origin_allow):
                    logger.warning(
                        "auth: rejected state-changing HTTP request with disallowed Origin",
                        extra={"origin": origin_header, "method": method, "path": path},
                    )
                    body, h403 = _build_403_response(
                        f"Origin '{origin_header}' is not in the allowlist"
                    )
                    await send({
                        "type": "http.response.start",
                        "status": int(HTTPStatus.FORBIDDEN),
                        "headers": h403,
                    })
                    await send({"type": "http.response.body", "body": body})
                    return

            # In no_auth_local_only mode, we're done — let Reflex handle it.
            if mode == AuthMode.NO_AUTH_LOCAL_ONLY:
                await asgi_app(scope, receive, send)
                return

            # Both authenticated modes (EXPLICIT_CREDS / TOKEN_AUTO /
            # PRODUCTION) check the session cookie first, then fall back to
            # the per-mode credential check.
            secret = _derive_secret(env)
            cookie_header = _header_value(headers, b"cookie")
            cookies = _parse_cookies(cookie_header)
            cookie_value = cookies.get(_COOKIE_NAME, "")
            user = _validate_cookie(cookie_value, secret)
            if user:
                # Valid session — pass through.
                await asgi_app(scope, receive, send)
                return

            authed_user: str | None = None

            if mode in (AuthMode.EXPLICIT_CREDS, AuthMode.PRODUCTION):
                auth_header = _header_value(headers, b"authorization")
                authed_user = _verify_basic_auth(auth_header, env)

            elif mode == AuthMode.TOKEN_AUTO:
                # ``?token=<hex>`` on the URL — accept on match against the
                # launch token, then set the cookie + 302 to clean URL so the
                # token doesn't sit in browser history.
                query_string = scope.get("query_string", b"")
                if isinstance(query_string, bytes):
                    query_string = query_string.decode("latin-1")
                qs = parse_qs(query_string)
                supplied = (qs.get("token") or [""])[0]
                expected = env.get(_LAUNCH_TOKEN_ENV, "").strip()
                if supplied and expected and hmac.compare_digest(
                    supplied.encode("utf-8"), expected.encode("utf-8")
                ):
                    # 302 redirect to the same path without ``?token=``.
                    is_loopback = _is_loopback_host(host_header)
                    cookie = _sign_cookie("default-user", secret)
                    set_cookie = _set_cookie_header(cookie, secure=not is_loopback)
                    redirect_headers = [
                        (b"location", path.encode("latin-1")),
                        (b"set-cookie", set_cookie),
                        (b"cache-control", b"no-store"),
                        (b"content-length", b"0"),
                    ]
                    await send({
                        "type": "http.response.start",
                        "status": int(HTTPStatus.FOUND),
                        "headers": redirect_headers,
                    })
                    await send({"type": "http.response.body", "body": b""})
                    return

            if authed_user is None:
                body, h401 = _build_401_response()
                await send({
                    "type": "http.response.start",
                    "status": int(HTTPStatus.UNAUTHORIZED),
                    "headers": h401,
                })
                await send({"type": "http.response.body", "body": body})
                return

            # Authed — set the session cookie via response-message rewriting,
            # then pass through to Reflex.
            is_loopback = _is_loopback_host(host_header)
            cookie = _sign_cookie(authed_user, secret)
            set_cookie = _set_cookie_header(cookie, secure=not is_loopback)

            async def send_with_cookie(message: dict) -> None:
                if message.get("type") == "http.response.start":
                    msg_headers = list(message.get("headers") or [])
                    msg_headers.append((b"set-cookie", set_cookie))
                    message = {**message, "headers": msg_headers}
                await send(message)

            await asgi_app(scope, receive, send_with_cookie)
            return

        # ---- WebSocket branch -------------------------------------------
        if scope_type == "websocket":
            # Host-header allowlist (DNS-rebinding on WS upgrade — close BEFORE
            # accept, never after).
            if not _host_matches_allowlist(host_header, host_allow):
                logger.warning(
                    "auth: WS rejected — disallowed Host",
                    extra={"host": host_header, "path": scope.get("path", "")},
                )
                await send({
                    "type": "websocket.close",
                    "code": _WS_CLOSE_CODE_ORIGIN_FAILED,
                    "reason": "host_header_not_allowed",
                })
                return

            # Origin allowlist (CSWSH defense — close BEFORE accept).
            if not _origin_matches_allowlist(origin_header, origin_allow):
                logger.warning(
                    "auth: WS rejected — disallowed Origin",
                    extra={"origin": origin_header, "path": scope.get("path", "")},
                )
                await send({
                    "type": "websocket.close",
                    "code": _WS_CLOSE_CODE_ORIGIN_FAILED,
                    "reason": "origin_not_allowed",
                })
                return

            # NO_AUTH_LOCAL_ONLY mode skips cookie validation — the Host check
            # above is the only gate. This preserves dev-mode behavior.
            if mode == AuthMode.NO_AUTH_LOCAL_ONLY:
                await asgi_app(scope, receive, send)
                return

            # Cookie validation BEFORE websocket.accept(). This is the load-
            # bearing pre-accept check per the brief.
            secret = _derive_secret(env)
            cookie_header = _header_value(headers, b"cookie")
            cookies = _parse_cookies(cookie_header)
            cookie_value = cookies.get(_COOKIE_NAME, "")
            user = _validate_cookie(cookie_value, secret)
            if not user:
                logger.warning(
                    "auth: WS rejected — invalid/missing session cookie",
                    extra={"path": scope.get("path", "")},
                )
                await send({
                    "type": "websocket.close",
                    "code": _WS_CLOSE_CODE_AUTH_FAILED,
                    "reason": "auth_required",
                })
                return

            # Valid session cookie — let Reflex accept the connection.
            await asgi_app(scope, receive, send)
            return

        # Unknown scope type — pass through (defensive; shouldn't happen).
        await asgi_app(scope, receive, send)

    return middleware


__all__ = [
    "AuthMode",
    "ENFORCEMENT_AVAILABLE",
    "basic_auth_transformer",
]
