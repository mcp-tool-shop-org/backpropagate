"""WebSocket-upgrade test helpers for the auth middleware (TESTS-F-001).

The DESIGN_BRIEF requirement #5 — "Validate BEFORE ``websocket.accept()``"
— is the load-bearing safety property of the middleware. Post-accept
validation is a documented DoS vector (attackers exhaust the server with
thousands of accepted-then-rejected connections). These helpers make it
easy to assert "the middleware sent ``websocket.close`` with code 4401
WITHOUT first sending ``websocket.accept``".

The pattern is direct ASGI invocation: build a ``scope`` with
``type="websocket"``, drive the middleware with a hand-rolled
``receive`` / ``send`` pair that records every message sent, then assert
on the recorded sequence.

This avoids needing a real WebSocket client (websockets / httpx_ws) for
the pre-accept tests — the contract is at the ASGI protocol level, not
the browser/WebSocket level. Integration tests that DO need a real
client can use websockets-asgi (gated behind importorskip).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WSMessageRecorder:
    """Records every ASGI message the middleware sends.

    Used as the ``send`` callable when driving a websocket scope through
    the middleware directly. After the middleware finishes, inspect
    ``messages`` to assert the close code, accept-vs-close ordering, etc.

    Usage:

        recorder = WSMessageRecorder()
        async def receive():
            return {"type": "websocket.connect"}
        await middleware({"type": "websocket", ...}, receive, recorder)
        assert recorder.accepted is False, "Middleware accepted before validating!"
        assert recorder.close_code == 4401
    """
    messages: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    @property
    def accepted(self) -> bool:
        """True if the middleware ever sent ``websocket.accept``."""
        return any(m.get("type") == "websocket.accept" for m in self.messages)

    @property
    def closed(self) -> bool:
        """True if the middleware sent ``websocket.close``."""
        return any(m.get("type") == "websocket.close" for m in self.messages)

    @property
    def close_code(self) -> int | None:
        """The code from the first ``websocket.close`` message, or None."""
        for m in self.messages:
            if m.get("type") == "websocket.close":
                return m.get("code")
        return None

    @property
    def accepted_before_closed(self) -> bool:
        """True if the middleware accepted BEFORE closing.

        This is the DoS-vector property the brief warns about (DESIGN_BRIEF
        requirement #5 — anti-pattern at the bottom of the brief). When
        True, the middleware is broken — auth must be validated PRE-accept.
        """
        first_accept_idx = None
        first_close_idx = None
        for i, m in enumerate(self.messages):
            if m.get("type") == "websocket.accept" and first_accept_idx is None:
                first_accept_idx = i
            if m.get("type") == "websocket.close" and first_close_idx is None:
                first_close_idx = i
        if first_accept_idx is None or first_close_idx is None:
            return False
        return first_accept_idx < first_close_idx


def make_ws_scope(
    *,
    path: str = "/_event",
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    host: str = "127.0.0.1:7860",
    origin: str | None = None,
) -> dict[str, Any]:
    """Build an ASGI ``scope`` dict for a WebSocket upgrade request.

    Args:
        path: Request path. Defaults to ``/_event`` (Reflex's WS endpoint).
        headers: Extra HTTP headers. Cookies and Origin handled separately.
        cookies: Cookie name -> value pairs. Encoded into a ``Cookie:``
            header per RFC 6265.
        host: ``Host:`` header value. The DNS-rebinding defense
            (DESIGN_BRIEF Host-header allowlist) reads this.
        origin: ``Origin:`` header value. The CSWSH defense
            (DESIGN_BRIEF Origin allowlist) reads this on WS upgrade.

    Returns:
        A dict shaped per the ASGI 3.0 WebSocket-scope spec. Headers are
        encoded as a list of ``(bytes, bytes)`` tuples per the spec.
    """
    raw_headers: list[tuple[bytes, bytes]] = [
        (b"host", host.encode("ascii")),
        (b"upgrade", b"websocket"),
        (b"connection", b"Upgrade"),
        (b"sec-websocket-version", b"13"),
        (b"sec-websocket-key", b"dGhlIHNhbXBsZSBub25jZQ=="),  # RFC 6455 example
    ]
    if origin is not None:
        raw_headers.append((b"origin", origin.encode("ascii")))
    if cookies:
        cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
        raw_headers.append((b"cookie", cookie_str.encode("ascii")))
    if headers:
        for k, v in headers.items():
            raw_headers.append((k.encode("ascii"), v.encode("ascii")))

    return {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws",
        "server": ("127.0.0.1", 7860),
        "client": ("127.0.0.1", 54321),
        "root_path": "",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
        "subprotocols": [],
    }


async def make_connect_receive():
    """Build a ``receive`` callable that yields a single ``websocket.connect``.

    The middleware reads this to learn about the upgrade attempt. After
    the middleware closes the connection (or accepts and forwards),
    further receives would block forever; tests typically only need the
    first connect message.
    """
    sent = [False]

    async def receive() -> dict[str, Any]:
        if not sent[0]:
            sent[0] = True
            return {"type": "websocket.connect"}
        # Subsequent receives — return disconnect to unblock the middleware.
        return {"type": "websocket.disconnect", "code": 1000}

    return receive
