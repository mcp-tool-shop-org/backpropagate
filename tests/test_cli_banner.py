"""Tests for the BRIDGE-B Jupyter-pattern UI startup banner.

Cross-domain handoff from bridge agent (v1.3 Wave 3 Stage C): the
``_print_ui_startup_banner`` helper in ``backpropagate/cli.py`` was added
to surface a Lee & See 2004 trust-calibration banner on UI startup. The
bridge agent could not write tests itself (tests/ domain belongs to the
tests agent), so this file covers the 4 auth-mode branches + the
``BACKPROPAGATE_UI_QUIET=1`` suppression escape hatch.

Banner contract (from the function docstring):

* Line 1: ``[backprop] http://<host>:<port>/  <- UI listening`` (or
  ``/?token=...`` when a token-query is present)
* Line 2: ``[backprop] auth: <mode_label> -- <consequence>``
* Line 3: ``[backprop] open the URL to start; stop with Ctrl+C``

Mode mapping (precedence order — share+no-auth > basic > token > loopback
> non-loopback-no-auth):

* ``no_auth_local`` — auth=None, share=False, loopback host → ``none (loopback-only)``
* ``token_auto`` — auth=None, share=False, token_query="xyz" → ``token (auto-generated)``
* ``basic`` — auth=("alice", "secret") → ``basic (user 'alice')``
* ``insecure (share)`` — auth=None, share=True → ``DISABLED (--share without --auth)``
* ``insecure (host)`` — auth=None, share=False, non-loopback host → ``DISABLED (non-loopback bind...)``

All tests assert against ``capsys.readouterr().err`` since the banner is
documented as writing to stderr.
"""

from __future__ import annotations

import pytest

# =============================================================================
# Quiet-mode suppression
# =============================================================================


class TestBannerQuietMode:
    """The BACKPROPAGATE_UI_QUIET=1 escape hatch suppresses the banner entirely."""

    def test_quiet_mode_suppresses_banner_completely(self, capsys, monkeypatch):
        """When BACKPROPAGATE_UI_QUIET=1, the banner prints nothing."""
        from backpropagate.cli import _print_ui_startup_banner

        monkeypatch.setenv("BACKPROPAGATE_UI_QUIET", "1")
        _print_ui_startup_banner(
            bound_host="127.0.0.1",
            port=7860,
            auth=None,
            share=False,
            token_query=None,
        )

        captured = capsys.readouterr()
        # Both stdout and stderr should be empty under quiet mode.
        assert captured.out == "", (
            f"Quiet mode leaked stdout: {captured.out!r}"
        )
        assert captured.err == "", (
            f"Quiet mode leaked stderr: {captured.err!r}"
        )

    def test_quiet_mode_inactive_when_not_set(self, capsys, monkeypatch):
        """Banner fires normally when the env var is unset."""
        from backpropagate.cli import _print_ui_startup_banner

        monkeypatch.delenv("BACKPROPAGATE_UI_QUIET", raising=False)
        _print_ui_startup_banner(
            bound_host="127.0.0.1",
            port=7860,
            auth=None,
            share=False,
            token_query=None,
        )

        captured = capsys.readouterr()
        # Banner went to stderr.
        assert "[backprop]" in captured.err

    def test_quiet_mode_inactive_when_set_to_zero(self, capsys, monkeypatch):
        """BACKPROPAGATE_UI_QUIET=0 does NOT suppress (only '1' suppresses)."""
        from backpropagate.cli import _print_ui_startup_banner

        monkeypatch.setenv("BACKPROPAGATE_UI_QUIET", "0")
        _print_ui_startup_banner(
            bound_host="127.0.0.1",
            port=7860,
            auth=None,
            share=False,
            token_query=None,
        )

        captured = capsys.readouterr()
        assert "[backprop]" in captured.err


# =============================================================================
# Per-mode banner content
# =============================================================================


class TestBannerNoAuthLocal:
    """Loopback host + no auth + no share = ``none (loopback-only)`` mode."""

    @pytest.fixture(autouse=True)
    def _ensure_not_quiet(self, monkeypatch):
        monkeypatch.delenv("BACKPROPAGATE_UI_QUIET", raising=False)

    def test_loopback_127_no_auth_produces_no_auth_local_banner(self, capsys):
        """127.0.0.1 + auth=None + share=False fires no_auth_local mode."""
        from backpropagate.cli import _print_ui_startup_banner

        _print_ui_startup_banner(
            bound_host="127.0.0.1",
            port=7860,
            auth=None,
            share=False,
            token_query=None,
        )
        err = capsys.readouterr().err

        assert "http://127.0.0.1:7860/" in err
        assert "UI listening" in err
        assert "auth: none (loopback-only)" in err
        assert "any local process can access" in err
        assert "Ctrl+C" in err

    def test_localhost_no_auth_produces_no_auth_local_banner(self, capsys):
        """'localhost' is also treated as loopback."""
        from backpropagate.cli import _print_ui_startup_banner

        _print_ui_startup_banner(
            bound_host="localhost",
            port=7861,
            auth=None,
            share=False,
            token_query=None,
        )
        err = capsys.readouterr().err

        assert "http://localhost:7861/" in err
        assert "auth: none (loopback-only)" in err

    def test_three_line_shape(self, capsys):
        """The banner is exactly 3 lines, all prefixed with [backprop]."""
        from backpropagate.cli import _print_ui_startup_banner

        _print_ui_startup_banner(
            bound_host="127.0.0.1",
            port=7860,
            auth=None,
            share=False,
            token_query=None,
        )
        err = capsys.readouterr().err

        lines = [line for line in err.strip().splitlines() if line.strip()]
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}: {lines}"
        for line in lines:
            assert line.startswith("[backprop]"), (
                f"Each banner line must start with [backprop], got {line!r}"
            )


class TestBannerTokenAuto:
    """No-auth + token_query = ``token (auto-generated)`` mode."""

    @pytest.fixture(autouse=True)
    def _ensure_not_quiet(self, monkeypatch):
        monkeypatch.delenv("BACKPROPAGATE_UI_QUIET", raising=False)

    def test_token_query_present_produces_token_auto_banner(self, capsys):
        """auth=None + token_query='deadbeef' fires token mode + URL has ?token=."""
        from backpropagate.cli import _print_ui_startup_banner

        _print_ui_startup_banner(
            bound_host="127.0.0.1",
            port=7860,
            auth=None,
            share=False,
            token_query="deadbeef",
        )
        err = capsys.readouterr().err

        # URL must contain the token query.
        assert "?token=deadbeef" in err
        # Mode label.
        assert "auth: token (auto-generated)" in err
        # Consequence prose.
        assert "share the URL above to grant access" in err
        assert "URL contains a secret" in err


class TestBannerBasic:
    """auth=(user, password) = ``basic (user '<u>')`` mode."""

    @pytest.fixture(autouse=True)
    def _ensure_not_quiet(self, monkeypatch):
        monkeypatch.delenv("BACKPROPAGATE_UI_QUIET", raising=False)

    def test_basic_auth_produces_basic_banner(self, capsys):
        """auth=('alice', 'secret') fires basic mode + URL does NOT leak password."""
        from backpropagate.cli import _print_ui_startup_banner

        _print_ui_startup_banner(
            bound_host="127.0.0.1",
            port=7860,
            auth=("alice", "secret-password"),
            share=False,
            token_query=None,
        )
        err = capsys.readouterr().err

        # Username surfaces in the mode label.
        assert "auth: basic" in err
        assert "alice" in err
        # Password MUST NOT appear anywhere in the banner output.
        assert "secret-password" not in err, (
            "Password leaked into startup banner — security regression!"
        )
        # Consequence prose.
        assert "password protects the UI" in err
        assert "rotate the password" in err

    def test_basic_auth_url_has_no_query_string(self, capsys):
        """Basic-mode URL has no ?token query."""
        from backpropagate.cli import _print_ui_startup_banner

        _print_ui_startup_banner(
            bound_host="127.0.0.1",
            port=7860,
            auth=("alice", "secret-password"),
            share=False,
            token_query=None,
        )
        err = capsys.readouterr().err

        assert "?token=" not in err


class TestBannerInsecureModes:
    """The two DISABLED branches — share-without-auth + non-loopback-without-auth.

    These should be unreachable under the v1.2.0 refuse-to-start gates, but
    the banner branches exist to make the function total over the 4 modes.
    """

    @pytest.fixture(autouse=True)
    def _ensure_not_quiet(self, monkeypatch):
        monkeypatch.delenv("BACKPROPAGATE_UI_QUIET", raising=False)

    def test_share_without_auth_produces_disabled_banner(self, capsys):
        """share=True + auth=None — DISABLED label + PUBLIC consequence."""
        from backpropagate.cli import _print_ui_startup_banner

        _print_ui_startup_banner(
            bound_host="127.0.0.1",
            port=7860,
            auth=None,
            share=True,
            token_query=None,
        )
        err = capsys.readouterr().err

        assert "auth: DISABLED" in err
        assert "--share without --auth" in err
        assert "PUBLIC" in err
        assert "anyone with the URL has full access" in err

    def test_non_loopback_without_auth_produces_disabled_banner(self, capsys):
        """Non-loopback host + auth=None — DISABLED label + network-reachable consequence."""
        from backpropagate.cli import _print_ui_startup_banner

        _print_ui_startup_banner(
            bound_host="192.168.1.10",
            port=7860,
            auth=None,
            share=False,
            token_query=None,
        )
        err = capsys.readouterr().err

        assert "auth: DISABLED" in err
        assert "non-loopback bind" in err
        assert "reachable from the network" in err
        assert "NO authentication" in err
