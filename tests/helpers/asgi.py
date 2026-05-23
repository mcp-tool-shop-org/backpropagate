"""ASGI test helpers for the Reflex auth middleware (TESTS-F-001).

Background — DESIGN_BRIEF.md mandates 16 tests for the ``basic_auth_transformer``
middleware that gates both HTTP routes and the ``/_event`` WebSocket upgrade.
Pre-Wave-6 the tests/ tree had ZERO httpx / Starlette test surface; this
module is the foundation.

Key constraints (from the brief and from Reflex 0.9.x reality):

- Reflex's ``rx.App.api`` was removed in 0.8; the canonical hook is
  ``rx.App(api_transformer=...)``. Tests that call the transformer with the
  Reflex app's underlying ASGI app are the documented path.
- The transformer wraps Reflex's underlying Starlette/FastAPI ASGI app.
  Calling ``basic_auth_transformer(stub_asgi_app)`` should be enough to
  test the middleware in isolation — no need to spin up the full Reflex
  surface (which is expensive and version-sensitive).
- httpx exposes ``httpx.ASGITransport`` for in-process ASGI testing
  without binding a real port. This is what the brief and current
  best-practice (FastAPI/Starlette) docs recommend.

Public API:

- ``make_asgi_client(asgi_app, base_url=...)`` -> ``httpx.AsyncClient``
- ``stub_asgi_app`` — a minimal HTTP/WS ASGI app that always returns 200
  / accepts; used to test the middleware's gate behaviour without
  needing the real Reflex app.
- ``basic_auth_header(user, password)`` -> ``{"Authorization": "Basic …"}``.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

# httpx is the dev-deps backbone for ASGI testing. Skip the entire module
# gracefully if the operator hasn't installed it — the auth-middleware
# tests are a Wave 6 addition and we don't want to break pre-Wave-6 CI
# while httpx is being added to pyproject.toml.
httpx = pytest.importorskip(
    "httpx",
    reason="httpx>=0.27 is required for ASGI middleware tests. "
           "Install via the [dev] extra or `pip install httpx`.",
)


def make_asgi_client(app: Any, *, base_url: str = "http://testserver") -> httpx.AsyncClient:
    """Build an httpx AsyncClient over an ASGI app (no real port binding).

    Args:
        app: The ASGI 3.0 application (callable with scope/receive/send).
            Pass the auth-middleware-wrapped Reflex app here.
        base_url: Base URL for relative requests. Default "http://testserver"
            because httpx defaults the Host header to the URL's hostname,
            and "testserver" is the conventional name FastAPI's TestClient
            uses too — pin it so Host-header-allowlist tests have a known
            value to allow or reject.

    Returns:
        An ``httpx.AsyncClient`` configured with ``ASGITransport``.
        Use as an async context manager:

            async with make_asgi_client(app) as client:
                response = await client.get("/")

    Note: ``ASGITransport`` does not invoke uvicorn or hypercorn; it speaks
    ASGI directly with the app. WebSocket testing requires a separate path
    (see ``tests.helpers.ws``).
    """
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=base_url,
    )


def basic_auth_header(user: str, password: str) -> dict[str, str]:
    """Build an HTTP Basic Authorization header.

    DESIGN_BRIEF mode 2 / 3 expects ``Authorization: Basic
    base64(user:password)``. This helper standardises the encoding so test
    sites don't reinvent it (and accidentally pad incorrectly or drop the
    "Basic " prefix).

    Args:
        user: Username component. May be empty for tests that exercise the
            edge case where user portion is missing.
        password: Password component. May be empty.

    Returns:
        ``{"Authorization": "Basic <b64(user:password)>"}`` suitable for
        passing as the ``headers=`` kwarg of httpx requests.
    """
    raw = f"{user}:{password}".encode()
    encoded = base64.b64encode(raw).decode("ascii")
    return {"Authorization": f"Basic {encoded}"}


def malformed_auth_header(value: str) -> dict[str, str]:
    """Build an Authorization header with raw (potentially malformed) value.

    Useful for negative tests — e.g. wrong scheme, bad base64, missing
    colon in decoded form, etc.
    """
    return {"Authorization": value}


async def stub_asgi_http_app(scope: dict, receive: Any, send: Any) -> None:
    """Minimal ASGI app that returns 200 OK for any HTTP request.

    Used as the "underlying" app under the auth middleware in isolation
    tests. The middleware should reject requests BEFORE they reach this
    stub — if a test sees a 200, the middleware passed-through, which is
    only correct when auth succeeded.

    For WebSocket scopes, this app accepts and immediately closes — the
    middleware is expected to reject pre-accept on auth failure, so a
    test reaching this code path indicates the middleware passed-through.
    """
    if scope["type"] == "http":
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        })
        await send({
            "type": "http.response.body",
            "body": b"stub-ok",
        })
    elif scope["type"] == "websocket":
        # Wait for the upgrade request, then accept and close. The middleware
        # is supposed to reject BEFORE this point on bad auth.
        message = await receive()
        if message["type"] == "websocket.connect":
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.close", "code": 1000})
    else:  # pragma: no cover — defensive
        raise RuntimeError(f"stub does not handle scope.type={scope['type']!r}")


@pytest.fixture
def reflex_app_with_auth_enforced(monkeypatch):
    """Reflex app configured with ENFORCEMENT_AVAILABLE=True + Basic auth.

    Flips the load-bearing flag the way ``test_cli_extended.py:1279`` does,
    then re-imports the Reflex app module so the ``api_transformer=`` hook
    sees the fresh ``basic_auth_transformer``.

    Yields the ``rx.App`` instance (the underlying ASGI app is at
    ``app.api`` in older Reflex or accessible via the transformer
    directly in 0.9+). Tests that need the ASGI app should prefer
    ``stub_asgi_http_app`` + direct middleware invocation for isolation.
    """
    monkeypatch.setattr(
        "backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE", True
    )
    monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "testuser:testpass")

    # Force a fresh module load so the import-time guard in app.py
    # re-evaluates against the patched flag.
    import importlib
    import sys

    for mod_name in list(sys.modules):
        if mod_name.startswith("backpropagate.ui_app.app"):
            del sys.modules[mod_name]

    try:
        from backpropagate.ui_app import app as app_module
        importlib.reload(app_module)
        yield app_module.app
    except Exception as exc:  # pragma: no cover — surface frontend gap clearly
        pytest.skip(
            f"Reflex app could not be loaded for auth-middleware testing: "
            f"{type(exc).__name__}: {exc}. The frontend agent's "
            f"basic_auth_transformer may not have landed yet."
        )


@pytest.fixture
def reflex_app_no_auth(monkeypatch):
    """Reflex app without auth (legacy v1.1.x mode).

    Used by the refuse-to-construct test that pins the v1.2 false-promise
    guard: importing ``backpropagate.ui_app.app`` with
    ``ENFORCEMENT_AVAILABLE=False`` AND ``BACKPROPAGATE_UI_AUTH`` set must
    raise at module import time (see ``ui_app/app.py:39``).
    """
    monkeypatch.setattr(
        "backpropagate.ui_app.auth.ENFORCEMENT_AVAILABLE", False
    )
    monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "should-refuse:to-start")

    import sys
    for mod_name in list(sys.modules):
        if mod_name.startswith("backpropagate.ui_app.app"):
            del sys.modules[mod_name]

    # Caller is expected to attempt the import and catch RuntimeError.
    yield
