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

- Credentials: the CLI never passes the plaintext password across the process
  boundary. It hands the subprocess ``BACKPROPAGATE_UI_AUTH_USER`` plus
  ``BACKPROPAGATE_UI_AUTH_VERIFIER`` (a salted scrypt verifier,
  ``scrypt$n$r$p$salt$hash``) and Basic-auth attempts are checked against it
  with a constant-time compare. A developer who sets ``BACKPROPAGATE_UI_AUTH=
  user:pass`` themselves when running Reflex directly is still accepted: the
  verifier is derived in memory at first use and never persisted.

- Cookie session: HMAC(``<user>:<exp>``) signed with a random per-process key
  (``secrets.token_bytes(32)``) for explicit_creds mode, never derived from
  the password, or the launch-token bytes for token_auto mode. Sessions
  therefore do not survive a UI restart. ``HttpOnly`` + ``SameSite=Lax`` +
  ``Secure`` when non-loopback + 12h expiry.

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
import functools
import hashlib
import hmac
import logging
import os
import secrets
import time
from collections.abc import Callable
from http import HTTPStatus
from urllib.parse import parse_qs, urlsplit

from backpropagate.ui_security import (
    _ui_listen_port,
    hash_password,
    is_valid_verifier,
    verify_password,
)

from .access_log import install_access_query_filter

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
# - ``BACKPROPAGATE_UI_AUTH_USER``  — explicit_creds mode username (not secret;
#                                     set by the CLI alongside the verifier)
# - ``BACKPROPAGATE_UI_AUTH_VERIFIER`` — explicit_creds mode salted scrypt
#                                     verifier ``scrypt$n$r$p$salt$hash`` (what
#                                     the CLI passes; never the plaintext)
# - ``BACKPROPAGATE_UI_AUTH``       — explicit_creds mode "user:pass" for people
#                                     who run Reflex directly. Accepted, turned
#                                     into an in-memory verifier at first use,
#                                     never written anywhere. The verifier wins
#                                     if both are set.
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
# - ``BACKPROPAGATE_UI_LAUNCH_TOKEN`` — token_auto mode launch token (``backprop ui``
#                                     generates one per launch when ``--auth``
#                                     is absent and exports it).

_COOKIE_NAME = "backprop_sess"
_COOKIE_TTL_SECONDS = 12 * 60 * 60  # 12 hours; matches DESIGN_BRIEF
_COOKIE_REALM = "backpropagate"
_WS_CLOSE_CODE_AUTH_FAILED = 4401  # Application-level "auth failed"
_WS_CLOSE_CODE_ORIGIN_FAILED = 4403  # Application-level "forbidden origin"
# FRONTEND-B-005 (Stage C observability): distinct close code for Host-header
# mismatch (DNS-rebinding defense) so operators / clients can differentiate it
# from Origin mismatch (CSWSH defense) in close-frame logs. RFC 6455 reserves
# 4000-4999 for application use; 4404 is unused by Reflex and reads as
# "Host not allowlisted" (echoes HTTP 421 Misdirected Request semantics).
_WS_CLOSE_CODE_HOST_FAILED = 4404  # Application-level "forbidden host"
_LAUNCH_TOKEN_ENV = "BACKPROPAGATE_UI_LAUNCH_TOKEN"  # nosec B105 — env var NAME, not a credential value
_AUTH_ENV = "BACKPROPAGATE_UI_AUTH"
_AUTH_USER_ENV = "BACKPROPAGATE_UI_AUTH_USER"
_AUTH_VERIFIER_ENV = "BACKPROPAGATE_UI_AUTH_VERIFIER"


class AuthMode(enum.Enum):
    """Operator-facing modes from the DESIGN_BRIEF matrix."""

    NO_AUTH_LOCAL_ONLY = "no_auth_local_only"
    TOKEN_AUTO = "token_auto"  # nosec B105 — enum identifier, not a credential
    EXPLICIT_CREDS = "explicit_creds"
    PRODUCTION = "production"


def _detect_mode(env: dict[str, str] | None = None) -> AuthMode:
    """Pick mode from environment + bind address hints.

    Order (most-specific first):

    "Credentials set" below means ``BACKPROPAGATE_UI_AUTH_VERIFIER`` (what the
    CLI passes) or the plaintext ``BACKPROPAGATE_UI_AUTH`` (direct-Reflex use).

    1. Credentials set + ``BACKPROPAGATE_UI_SHARE_HOST`` set →
       PRODUCTION (--share + --auth).
    2. Credentials set + ``BACKPROPAGATE_UI_HOST_BIND`` set
       (non-loopback) → PRODUCTION (--host + --auth).
    3. Credentials set → EXPLICIT_CREDS (loopback bind).
    4. ``BACKPROPAGATE_UI_LAUNCH_TOKEN`` set → TOKEN_AUTO (what a plain
       ``backprop ui`` launch gets).
    5. Otherwise → NO_AUTH_LOCAL_ONLY (only reachable by running Reflex
       directly with no env; ``backprop ui`` never lands here).
    """
    env = env if env is not None else dict(os.environ)
    auth_creds = (
        env.get(_AUTH_VERIFIER_ENV, "").strip() or env.get(_AUTH_ENV, "").strip()
    )
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
    """Cookie HMAC secret derivation.

    - Credentials set (``--auth``): a random per-process key. It is NOT derived
      from the password, so a stolen cookie or a leaked key reveals nothing
      about the credential, and nothing about the credential can be used to
      forge a cookie. Sessions do not survive a UI restart (new process, new
      key); users sign in again.
    - ``BACKPROPAGATE_UI_LAUNCH_TOKEN`` set: token bytes used directly (the
      token is itself a 256-bit random value, not a human-chosen password).
    - Otherwise: the same per-process random bytes (NO_AUTH_LOCAL_ONLY mode
      doesn't validate cookies, but the secret needs to be non-empty so the
      hmac.new() call doesn't raise).
    """
    env = env if env is not None else dict(os.environ)
    auth_creds = (
        env.get(_AUTH_VERIFIER_ENV, "").strip() or env.get(_AUTH_ENV, "").strip()
    )
    launch_token = env.get(_LAUNCH_TOKEN_ENV, "").strip()
    if auth_creds:
        return _PROCESS_LOCAL_SECRET
    if launch_token:
        return launch_token.encode("utf-8")
    # Stable per-process secret. Calling code in NO_AUTH_LOCAL_ONLY mode
    # short-circuits before ever validating a cookie, so this just keeps the
    # hmac primitives happy.
    return _PROCESS_LOCAL_SECRET


_PROCESS_LOCAL_SECRET = secrets.token_bytes(32)


@functools.lru_cache(maxsize=8)
def _verifier_for_plaintext(raw: str) -> tuple[str, str] | None:
    """Derive an in-memory ``(user, verifier)`` from a plaintext ``user:pass``.

    Only used when somebody sets ``BACKPROPAGATE_UI_AUTH`` directly (running
    Reflex without the CLI). Cached so the ~50 ms scrypt derivation runs once
    per distinct value, not once per request. The cache lives in process
    memory only; nothing is written to disk and the salt is fresh per process.
    """
    if ":" not in raw:
        return None
    user, password = raw.split(":", 1)
    return user, hash_password(password)


def _resolve_credential(env: dict[str, str]) -> tuple[str, str] | None:
    """Return ``(username, verifier)`` for explicit-creds mode, else ``None``.

    ``BACKPROPAGATE_UI_AUTH_VERIFIER`` (+ ``BACKPROPAGATE_UI_AUTH_USER``) wins
    over the plaintext ``BACKPROPAGATE_UI_AUTH``. A malformed verifier
    resolves to ``None`` so the caller fails closed (mode stays
    EXPLICIT_CREDS, nothing verifies).
    """
    verifier = env.get(_AUTH_VERIFIER_ENV, "").strip()
    if verifier:
        user = env.get(_AUTH_USER_ENV, "").strip()
        if not user or not is_valid_verifier(verifier):
            return None
        return user, verifier
    plaintext = env.get(_AUTH_ENV, "").strip()
    if plaintext:
        return _verifier_for_plaintext(plaintext)
    return None


def _verify_basic_auth(authorization_header: str, env: dict[str, str] | None = None) -> str | None:
    """Constant-time check of HTTP Basic against the configured credential.

    Returns the authenticated username on success, ``None`` on failure. The
    username is compared with ``hmac.compare_digest`` and the password is
    verified against the salted scrypt verifier (``verify_password``, also
    constant-time). The password check runs even when the username is wrong,
    so timing doesn't reveal which half failed.
    """
    env = env if env is not None else dict(os.environ)
    credential = _resolve_credential(env)
    if credential is None:
        return None
    exp_user, verifier = credential

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

    user_ok = hmac.compare_digest(user.encode("utf-8"), exp_user.encode("utf-8"))
    pass_ok = verify_password(password, verifier)
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


def _origin_host(hostname: str) -> str:
    """Lowercase a host and bracket IPv6 so the allowlist has one spelling."""
    host = hostname.strip().lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if ":" in host:
        return f"[{host}]"
    return host


def _origin_identity(scheme: str, hostname: str, port: int | None = None) -> str:
    """scheme://host:port. An omitted port is the scheme default (80 or 443)."""
    scheme_l = scheme.lower()
    if port is None:
        port = 443 if scheme_l == "https" else 80
    return f"{scheme_l}://{_origin_host(hostname)}:{port}"


def _origin_allowlist(env: dict[str, str] | None = None) -> set[str]:
    """Compute the Origin allowlist for the current mode.

    Cookies are not scoped to a port, so the allowlist is scheme + host +
    port. Loopback origins and a non-wildcard ``--host`` bind use the UI
    listen port (``BACKPROPAGATE_UI_PORT``, else 7862) on both schemes. A
    ``--share`` tunnel origin uses the scheme default port: the browser
    omits the port when the tunnel serves plain 80 or 443.
    """
    env = env if env is not None else dict(os.environ)
    port = _ui_listen_port(env)
    allowed: set[str] = set()
    for host in ("localhost", "127.0.0.1", "::1"):
        allowed.add(_origin_identity("http", host, port))
        allowed.add(_origin_identity("https", host, port))
    share_host = env.get("BACKPROPAGATE_UI_SHARE_HOST", "").strip()
    if share_host:
        # Tunnel hosts are virtually always HTTPS (cloudflared/ngrok), but
        # accept both schemes — operators occasionally test over plain HTTP.
        # The browser's Origin uses the default port, not the UI listen port.
        allowed.add(_origin_identity("http", share_host, 80))
        allowed.add(_origin_identity("https", share_host, 443))
    host_bind = env.get("BACKPROPAGATE_UI_HOST_BIND", "").strip().lower()
    if host_bind and host_bind not in ("0.0.0.0", "::", "", "localhost", "127.0.0.1", "::1"):
        allowed.add(_origin_identity("http", host_bind, port))
        allowed.add(_origin_identity("https", host_bind, port))
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
    """Compare an Origin header on scheme, host, and port.

    A missing Origin still matches. HTTP state-changing methods reject that
    case themselves, and the WebSocket upgrade rejects it in authenticated
    modes with a distinct reason. A present Origin that names another port
    does not match: cookies are sent to every port on the host.
    """
    if not isinstance(origin_header, str):
        return False
    if not origin_header:
        return True
    try:
        parts = urlsplit(origin_header.strip())
        # ``.port`` raises ValueError when the number is outside 0-65535.
        port = parts.port
    except (ValueError, AttributeError):
        return False
    if not parts.scheme or not parts.hostname:
        return False
    needle = _origin_identity(parts.scheme, parts.hostname, port)
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


def _hardened_header_pairs() -> list[tuple[bytes, bytes]]:
    """Build the OWASP security-header set as ASGI ``(bytes, bytes)`` pairs.

    UI-A-002: the auth-rejection paths (``_build_401`` / ``_build_421`` /
    ``_build_403``) emit responses by calling ``send()`` directly and
    ``return``-ing — they never delegate to the inner app, so the
    ``security_headers_middleware`` wrapped-send (which sits INSIDE this
    auth middleware in the ASGI chain) never runs on those responses. The
    ``security_headers.py`` docstring asserts "even auth-rejected responses
    carry the hardened header set", so we stamp the same set directly onto
    the rejection builders here.

    Reuses ``ui_security.security_headers_dict`` — the single source of truth
    for the CSP + X-Content-Type-Options + X-Frame-Options + X-XSS-Protection
    + Referrer-Policy + Permissions-Policy set — so the auth-error pages and
    the normal-response middleware never drift apart. Best-effort: if the
    builder import/raise (extremely unlikely — pure stdlib), the rejection
    response still ships without the extra headers rather than 500ing.
    """
    try:
        from backpropagate.ui_security import security_headers_dict

        raw = security_headers_dict()
        return [
            (name.encode("ascii"), value.encode("ascii"))
            for name, value in raw.items()
        ]
    except Exception:  # noqa: BLE001 — header build must never break a rejection
        return []


def _build_401_response(realm: str = _COOKIE_REALM, hint: str | None = None) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    """Build the 401 body + headers (HTTP-Basic challenge).

    Returns ``(body, headers)``. The body is the operator-facing message from
    DESIGN_BRIEF §Operator UX → Error messages. UI-A-002: the hardened OWASP
    security-header set is stamped here so the 401 auth-error page carries
    CSP / X-Frame-Options / X-Content-Type-Options even though it bypasses
    the security-headers middleware wrapped-send.
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
    headers.extend(_hardened_header_pairs())
    return body, headers


def _build_421_response(host: str) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    """Build the 421 Misdirected Request body for Host-header mismatch.

    UI-A-002: hardened OWASP security headers are stamped here too — the 421
    rejection bypasses the security-headers middleware wrapped-send.
    """
    body = (
        f"421 Misdirected Request: Host header '{host}' is not in the "
        "allowlist for this backpropagate UI instance (DNS-rebinding defense)."
    ).encode()
    headers = [
        (b"content-type", b"text/plain; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    headers.extend(_hardened_header_pairs())
    return body, headers


def _build_403_response(reason: str) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    """Build the 403 body for Origin-header mismatch (CSWSH defense).

    UI-A-002: hardened OWASP security headers are stamped here too — the 403
    rejection bypasses the security-headers middleware wrapped-send.
    """
    body = (
        f"403 Forbidden: {reason} (CSWSH defense, CWE-1385)."
    ).encode()
    headers = [
        (b"content-type", b"text/plain; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    headers.extend(_hardened_header_pairs())
    return body, headers


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
    """Whether an HTTP request path bypasses the credential gate.

    The Host allowlist still applies. ``/_next/`` is not a bypass when a
    share host is configured; the caller keeps that path on the cookie gate.
    """
    if not path:
        return False
    return any(path == prefix or path.startswith(prefix) for prefix in _PASSTHROUGH_PATHS)


def _query_has_token(scope: dict) -> bool:
    """True when the request query carries a ``token`` key.

    The value is not returned. A second visit that already holds the session
    cookie still has to leave the address bar, including when the value is
    blank or wrong.
    """
    raw = scope.get("query_string", b"")
    if isinstance(raw, bytes):
        text = raw.decode("latin-1", "replace")
    elif isinstance(raw, str):
        text = raw
    else:
        return False
    if not text:
        return False
    return "token" in parse_qs(text, keep_blank_values=True)


def _clean_redirect_headers(
    path: str, set_cookie: bytes | None = None,
) -> list[tuple[bytes, bytes]]:
    """302 to ``path`` with no query string, plus the hardened header set."""
    headers: list[tuple[bytes, bytes]] = [
        (b"location", path.encode("latin-1")),
        (b"cache-control", b"no-store"),
        (b"content-length", b"0"),
    ]
    if set_cookie is not None:
        headers.insert(1, (b"set-cookie", set_cookie))
    headers.extend(_hardened_header_pairs())
    return headers


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

    Pass-through paths (``/ping``, ``/favicon``, and ``/_next/`` when no
    share host is set) skip the credential check. They still require an
    allowed Host. ``/_next/`` requires the session cookie when
    ``BACKPROPAGATE_UI_SHARE_HOST`` is set.

    No-auth-local-only mode (no ``BACKPROPAGATE_UI_AUTH``,
    no ``BACKPROPAGATE_UI_LAUNCH_TOKEN``) is the v1.1.x back-compat path: the
    middleware applies the Host-header allowlist (loopback-only) but does NOT
    require credentials. The CLI's refuse-to-start rails are the gate that
    keeps this mode loopback-bound.
    """

    async def middleware(scope: dict, receive: Callable, send: Callable) -> None:
        # The server may configure logging before or after this app is
        # imported. Re-installing is idempotent and keeps the query string
        # out of the access line for the life of the process.
        install_access_query_filter()
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

            # Host-header allowlist (DNS-rebinding defense). Fires in EVERY
            # mode, including no_auth_local_only, and on the paths that
            # otherwise skip the credential check. That's the load-bearing
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

            # Credential bypass for orchestration and static assets. /_next/
            # stays on the cookie gate when a share host is configured, so a
            # tunnel cannot fetch the compiled frontend anonymously.
            share_host = env.get("BACKPROPAGATE_UI_SHARE_HOST", "").strip()
            if _is_passthrough_http(path) and not (
                share_host and path.startswith("/_next/")
            ):
                await asgi_app(scope, receive, send)
                return

            # Origin allowlist on state-changing methods only. GET/HEAD/OPTIONS
            # are read-only / preflight — the CORS layer + cookie SameSite=Lax
            # handle them; CSWSH defense requires Origin pinning on the
            # mutation surface. A missing Origin is rejected here too: the
            # matcher itself stays open so the WebSocket upgrade can report
            # "origin required" as its own reason.
            if method in ("POST", "PUT", "PATCH", "DELETE"):
                if not origin_header or not _origin_matches_allowlist(
                    origin_header, origin_allow
                ):
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
                # The launch URL keeps working in a second browser. Once this
                # browser already holds the cookie, drop ``?token=`` from the
                # address bar instead of rendering the page under that URL.
                if _query_has_token(scope):
                    await send({
                        "type": "http.response.start",
                        "status": int(HTTPStatus.FOUND),
                        "headers": _clean_redirect_headers(path),
                    })
                    await send({"type": "http.response.body", "body": b""})
                    return
                # Valid session — pass through. Log per-request validation at
                # DEBUG (FRONTEND-B-001 / advisor D3): per-session INFO + per-
                # request DEBUG is the Jupyter-style audit-trail floor on the
                # CVSS 9.8 surface. Silent by default; observable via
                # ``LOG_LEVEL=DEBUG`` on a busy UI without spam.
                logger.debug(
                    "auth: request validated",
                    extra={"auth_user": user, "path": path, "method": method},
                )
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
                    # FRONTEND-B-001 / advisor D3: log first-successful-auth at
                    # INFO with {user, mode, host}. Matches Jupyter's "you
                    # connected" pattern — one INFO line per session = manageable
                    # in CI / production logs. Cookie value MUST NOT appear here
                    # (HMAC payload contains the username which is already logged).
                    logger.info(
                        "auth: session opened",
                        extra={
                            "auth_user": "default-user",
                            "auth_mode": mode.value,
                            "remote_host": host_header,
                        },
                    )
                    await send({
                        "type": "http.response.start",
                        "status": int(HTTPStatus.FOUND),
                        "headers": _clean_redirect_headers(path, set_cookie),
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
            # FRONTEND-B-001 / advisor D3: log first-successful-auth at INFO
            # with {user, mode, host}. Matches Jupyter's "you connected"
            # pattern — one INFO line per session = manageable in CI / production
            # logs. ``authed_user`` is the verified username (NOT the password
            # or cookie value), safe to log.
            logger.info(
                "auth: session opened",
                extra={
                    "auth_user": authed_user,
                    "auth_mode": mode.value,
                    "remote_host": host_header,
                },
            )

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
            # Host-header allowlist (DNS-rebinding on WS upgrade - close BEFORE
            # accept, never after). FRONTEND-B-005 (Stage C observability): use
            # the dedicated Host-mismatch close code so post-mortem analysis
            # can distinguish Host mismatch (DNS-rebinding defense) from
            # Origin mismatch (CSWSH defense). Both paths still close BEFORE
            # websocket.accept() - the load-bearing pre-accept invariant is
            # preserved.
            if not _host_matches_allowlist(host_header, host_allow):
                logger.warning(
                    "auth: WS rejected - disallowed Host",
                    extra={"host": host_header, "path": scope.get("path", "")},
                )
                await send({
                    "type": "websocket.close",
                    "code": _WS_CLOSE_CODE_HOST_FAILED,
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

            # UI-A-008 (Wave A2): require a PRESENT Origin on the WS upgrade in
            # every authed mode. ``_origin_matches_allowlist`` deliberately
            # fails OPEN on a missing Origin (some legitimate same-origin HTTP
            # requests omit it, and the Host + cookie gates still apply there).
            # But a real browser ALWAYS sets Origin on a cross-origin WebSocket
            # handshake — so on the ``/_event`` upgrade, in an authed
            # deployment (especially under --share where the UI is reachable
            # off-loopback), a *missing* Origin has no legitimate browser
            # source and is the shape a hand-rolled CSWSH client uses to dodge
            # the allowlist. Reject it. NO_AUTH_LOCAL_ONLY (loopback dev) keeps
            # the fail-open behavior so native/test WS clients still connect.
            if mode != AuthMode.NO_AUTH_LOCAL_ONLY and not origin_header:
                logger.warning(
                    "auth: WS rejected — missing Origin on authed upgrade",
                    extra={"path": scope.get("path", ""), "mode": mode.value},
                )
                await send({
                    "type": "websocket.close",
                    "code": _WS_CLOSE_CODE_ORIGIN_FAILED,
                    "reason": "origin_required",
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
    # FRONTEND-B-005: distinct close codes for Host vs Origin failures.
    # Exported so post-mortem analysis scripts can name them rather than
    # decoding raw 44xx integers.
    "_WS_CLOSE_CODE_AUTH_FAILED",
    "_WS_CLOSE_CODE_ORIGIN_FAILED",
    "_WS_CLOSE_CODE_HOST_FAILED",
]
