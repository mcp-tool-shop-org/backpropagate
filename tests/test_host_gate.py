"""Tests for the ``backprop ui --host <non-loopback>`` refuse-to-start gate.

BRIDGE-A-002 → TESTS (v1.3 swarm Wave 1): the v1.2.0 CLI added a DNS-
rebinding defense — passing ``--host 0.0.0.0`` (or any non-loopback bind)
without ``--auth`` raises ``BackpropagateError(RUNTIME_UI_AUTH_NOT_ENFORCED)``
BEFORE the Reflex subprocess is launched. This is the same threat model as
the v1.2 ``--share`` gate; the two should have symmetric coverage.

Before this file, the gate had ZERO regression tests. The previous skip-
TODO test at ``tests/test_auth_middleware.py::test_host_non_loopback_without_auth_refuses_to_start``
was a placeholder pointing at "CLI tests" that didn't exist for this gate.

This file pins:
1. ``--host 0.0.0.0`` (no --auth) raises BackpropagateError +
   code=RUNTIME_UI_AUTH_NOT_ENFORCED + does NOT launch subprocess.
2. ``--host 192.168.1.10`` (LAN bind, no --auth) — same.
3. ``--host 0.0.0.0 --auth user:pass`` — happy path; subprocess launches.
4. ``--host 127.0.0.1`` (explicit loopback, no --auth) — proceeds (loopback
   is on the allowlist; the gate only fires for non-loopback binds).
5. ``--host localhost`` (loopback alias, no --auth) — proceeds.
6. ``--host ::1`` (IPv6 loopback, no --auth) — proceeds.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest


def _make_subprocess_result(returncode: int = 0):
    """A subprocess.CompletedProcess-shaped mock for cmd_ui's check."""
    result = MagicMock()
    result.returncode = returncode
    return result


class TestHostNonLoopbackRefuseToStart:
    """Pin the ``--host <non-loopback>`` refuse-to-start gate (v1.2 contract)."""

    def test_host_0_0_0_0_without_auth_refuses(self):
        """``backprop ui --host 0.0.0.0`` (no --auth) hard-errors.

        BRIDGE-A-002 negative path. The bind-all-interfaces case is the
        canonical DNS-rebinding / LAN-discovery exposure; the v1.2 gate
        requires --auth so the middleware can enforce per-request auth.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = _make_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host="0.0.0.0",
                share=False,
                auth=None,
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED", (
                f"Expected RUNTIME_UI_AUTH_NOT_ENFORCED code; got "
                f"{excinfo.value.code!r}. The error code must be stable "
                f"so support docs / ERROR_CODES catalog stays accurate."
            )
            assert "--host" in str(excinfo.value), (
                "Error message must mention --host so the operator knows "
                "which flag triggered the gate."
            )
            # TESTS-B-016 (Stage C): drop the trailing tuple — assert_not_called
            # itself raises AssertionError("Expected to be called 0 times. Called N times.")
            # which is the operator-facing message that matters. The prior
            # statement-form ``call(), ("message")`` looked like an assert
            # message but was actually a tuple-expression discarded at runtime.
            mock_run.assert_not_called()

    def test_host_lan_ip_without_auth_refuses(self):
        """``backprop ui --host 192.168.1.10`` (LAN IP, no --auth) hard-errors.

        Specific-LAN-IP binds are just as exposed as 0.0.0.0 binds for the
        DNS-rebinding threat — the gate must reject these too, not just
        the wildcard.
        """
        from backpropagate.cli import cmd_ui
        from backpropagate.exceptions import BackpropagateError

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = _make_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host="192.168.1.10",
                share=False,
                auth=None,
                verbose=False,
            )

            with pytest.raises(BackpropagateError) as excinfo:
                cmd_ui(args)

            assert excinfo.value.code == "RUNTIME_UI_AUTH_NOT_ENFORCED"
            mock_run.assert_not_called()

    def test_host_0_0_0_0_with_auth_proceeds(self):
        """``backprop ui --host 0.0.0.0 --auth user:pass`` launches subprocess.

        Happy path — --auth satisfies the gate's invariant (middleware can
        enforce per-request auth on the non-loopback bind). The subprocess
        launches with BACKPROPAGATE_UI_HOST_BIND=0.0.0.0 in its env so the
        middleware's Host-allowlist can be configured for the wildcard.
        """
        from backpropagate.cli import EXIT_OK, cmd_ui

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = _make_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host="0.0.0.0",
                share=False,
                auth="alice:hunter2",
                verbose=False,
            )

            result = cmd_ui(args)

            assert result == EXIT_OK
            mock_run.assert_called_once()
            # Env propagation: BACKPROPAGATE_UI_HOST_BIND must carry the
            # operator-supplied host so the middleware's allowlist can
            # learn it.
            call_env = mock_run.call_args.kwargs.get("env", {})
            assert call_env.get("BACKPROPAGATE_UI_HOST_BIND") == "0.0.0.0", (
                "BACKPROPAGATE_UI_HOST_BIND must be exported to the Reflex "
                "subprocess so the auth middleware knows which Host header "
                "is legitimate (DNS-rebinding allowlist)."
            )
            assert call_env.get("BACKPROPAGATE_UI_AUTH") == "alice:hunter2", (
                "BACKPROPAGATE_UI_AUTH must propagate to the subprocess so "
                "the middleware can enforce per-request auth."
            )

    @pytest.mark.parametrize(
        "loopback_host",
        ["127.0.0.1", "localhost", "::1"],
    )
    def test_host_loopback_without_auth_proceeds(self, loopback_host):
        """Loopback binds are on the allowlist — gate does NOT fire.

        The CLI's ``_LOOPBACK_BINDS`` tuple is the source of truth for
        which hosts skip the gate. This test pins the contract for each
        documented loopback alias so a regression that drops one (e.g.
        IPv6 loopback) is caught immediately.
        """
        from backpropagate.cli import EXIT_OK, cmd_ui

        with patch("backpropagate.cli.subprocess.run") as mock_run:
            mock_run.return_value = _make_subprocess_result(0)
            args = argparse.Namespace(
                port=7860,
                host=loopback_host,
                share=False,
                auth=None,
                verbose=False,
            )

            # Must NOT raise — loopback bind doesn't need --auth.
            result = cmd_ui(args)
            assert result == EXIT_OK
            mock_run.assert_called_once()
