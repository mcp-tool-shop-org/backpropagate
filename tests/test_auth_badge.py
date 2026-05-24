"""Tests for the footer auth-badge UI (FRONTEND-F-FOOTER-AUTH-BADGE).

Smoke-tests the introspection helper in ``ui_security.get_auth_badge_context``
and the component factory in ``ui_app/components/auth_badge.py``. The badge
is a Stage C humanization layer that READS the auth posture (env vars); it
does NOT participate in the auth decision. The middleware's GHSA-f65r-h4g3-3h9h
contracts are tested separately in ``test_auth_middleware.py``.
"""

from __future__ import annotations

import pytest

from backpropagate.ui_security import get_auth_badge_context

# ---------------------------------------------------------------------------
# get_auth_badge_context - the 6 states from V1_3_BRIEF P0 item 1
# ---------------------------------------------------------------------------


def test_badge_state_no_auth_local() -> None:
    """No env vars set -> green 'Local - no auth' badge."""
    ctx = get_auth_badge_context(env={})
    assert ctx.mode_key == "no_auth_local"
    assert ctx.mode_color == "green"
    assert ctx.mode_text == "Local - no auth"
    assert ctx.bind_host == "localhost"
    assert ctx.bind_port == "7860"
    assert ctx.reachable_from == "loopback-only"
    assert "loopback-only" in ctx.hover_text
    assert "no authentication" in ctx.hover_text


def test_badge_state_token_local() -> None:
    """LAUNCH_TOKEN set -> green 'Local - token' badge."""
    ctx = get_auth_badge_context(env={
        "BACKPROPAGATE_UI_LAUNCH_TOKEN": "abc123hex",
    })
    assert ctx.mode_key == "token_local"
    assert ctx.mode_color == "green"
    assert ctx.mode_text == "Local - token"
    assert "URL-token authentication" in ctx.hover_text


def test_badge_state_basic_local() -> None:
    """BACKPROPAGATE_UI_AUTH set, no share/host -> green 'Local - Basic'."""
    ctx = get_auth_badge_context(env={
        "BACKPROPAGATE_UI_AUTH": "alice:secret123",
    })
    assert ctx.mode_key == "basic_local"
    assert ctx.mode_color == "green"
    assert ctx.mode_text == "Local - Basic"
    # The username is exposed; the password is NEVER in the hover text.
    assert "alice" in ctx.hover_text
    assert "secret123" not in ctx.hover_text
    assert ctx.auth_user == "alice"


def test_badge_state_basic_shared_amber() -> None:
    """SHARE_HOST + BASIC AUTH -> amber 'Shared - Basic'."""
    ctx = get_auth_badge_context(env={
        "BACKPROPAGATE_UI_AUTH": "alice:secret123",
        "BACKPROPAGATE_UI_SHARE_HOST": "abc.trycloudflare.com",
    })
    assert ctx.mode_key == "basic_shared"
    assert ctx.mode_color == "amber"
    assert ctx.mode_text == "Shared - Basic"
    assert "public network" in ctx.hover_text
    assert "abc.trycloudflare.com" in ctx.hover_text


def test_badge_state_basic_network_amber() -> None:
    """HOST_BIND on a LAN IP + BASIC AUTH -> amber 'Network - Basic'."""
    ctx = get_auth_badge_context(env={
        "BACKPROPAGATE_UI_AUTH": "alice:secret123",
        "BACKPROPAGATE_UI_HOST_BIND": "192.168.1.50",
    })
    assert ctx.mode_key == "basic_network"
    assert ctx.mode_color == "amber"
    assert ctx.mode_text == "Network - Basic"
    assert "LAN" in ctx.hover_text


def test_badge_state_insecure_red() -> None:
    """SHARE_HOST WITHOUT auth -> red 'INSECURE - no auth'.

    The CLI's refuse-to-start rails should prevent reaching this state, but
    if the operator bypassed via direct ``reflex run`` the badge screams red.
    """
    ctx = get_auth_badge_context(env={
        "BACKPROPAGATE_UI_SHARE_HOST": "abc.trycloudflare.com",
    })
    assert ctx.mode_key == "insecure"
    assert ctx.mode_color == "red"
    assert ctx.mode_text == "INSECURE - no auth"
    assert "NO AUTH" in ctx.hover_text


def test_badge_state_insecure_lan_bind_no_auth() -> None:
    """LAN bind WITHOUT auth -> red 'INSECURE - no auth'."""
    ctx = get_auth_badge_context(env={
        "BACKPROPAGATE_UI_HOST_BIND": "0.0.0.0",
    })
    assert ctx.mode_key == "insecure"
    assert ctx.mode_color == "red"


def test_badge_state_loopback_bind_is_local() -> None:
    """HOST_BIND of 127.0.0.1 (explicit loopback) still resolves as local."""
    for bind in ("localhost", "127.0.0.1", "::1"):
        ctx = get_auth_badge_context(env={
            "BACKPROPAGATE_UI_AUTH": "alice:secret",
            "BACKPROPAGATE_UI_HOST_BIND": bind,
        })
        assert ctx.mode_key == "basic_local", (
            f"explicit loopback {bind!r} should resolve as local; got {ctx.mode_key}"
        )


def test_badge_state_password_never_in_hover_text() -> None:
    """Defense-in-depth: hover text NEVER contains the password half of AUTH.

    The auth introspection helper splits on ':' and uses only the username.
    A password with embedded ':' should still be fully redacted.
    """
    for creds in (
        "alice:hunter2",
        "bob:pa:ss:word",  # password with colons
        "carol:",  # empty password (malformed; should still not leak)
    ):
        ctx = get_auth_badge_context(env={"BACKPROPAGATE_UI_AUTH": creds})
        username, _, password = creds.partition(":")
        if password:
            assert password not in ctx.hover_text, (
                f"password {password!r} leaked into hover_text for creds {creds!r}"
            )
            assert password not in ctx.mode_text
        # Username SHOULD appear (operator needs to know who's logged in).
        assert username in ctx.hover_text


def test_badge_state_respects_port_env_var() -> None:
    """BACKPROPAGATE_UI_PORT propagates into hover_text and bind_port."""
    ctx = get_auth_badge_context(env={"BACKPROPAGATE_UI_PORT": "9000"})
    assert ctx.bind_port == "9000"
    assert "9000" in ctx.hover_text


def test_badge_state_is_dataclass() -> None:
    """Sanity check: AuthBadgeContext is a dataclass with stable fields."""
    ctx = get_auth_badge_context(env={})
    for field_name in (
        "mode_key",
        "mode_color",
        "mode_text",
        "hover_text",
        "bind_host",
        "bind_port",
        "reachable_from",
        "auth_user",
    ):
        assert hasattr(ctx, field_name), f"missing field {field_name}"


# ---------------------------------------------------------------------------
# Component smoke test - the badge factory returns a non-None Reflex element
# ---------------------------------------------------------------------------


def test_auth_badge_component_constructs() -> None:
    """``BpAuthBadge()`` returns a Reflex component without raising."""
    # Lazy import: the component depends on Reflex; the introspection helper
    # tests above don't need it.
    pytest.importorskip("reflex")
    from backpropagate.ui_app.components.auth_badge import BpAuthBadge

    component = BpAuthBadge()
    assert component is not None
    # The component should expose a render method (Reflex components inherit
    # from a Component base class that has _render or similar). We just check
    # truthiness here to avoid coupling to Reflex internals.
    assert bool(component)


def test_chrome_footer_constructs_with_badge() -> None:
    """``BpFooter()`` constructs with the auth badge wired in."""
    pytest.importorskip("reflex")
    from backpropagate.ui_app.chrome import BpFooter

    footer = BpFooter()
    assert footer is not None
