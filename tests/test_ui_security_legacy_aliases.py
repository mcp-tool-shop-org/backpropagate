"""
Legacy-alias tests for the v1.4 ``ui_security`` symbol rename
=============================================================

Wave 6a foundation (V1_4_BRIEF item 7) renamed the v1.0-era Gradio-prefixed
public names in ``backpropagate.ui_security`` to framework-agnostic
canonical forms while preserving the legacy names as deprecation aliases:

| Legacy (v1.0 Gradio era) | Canonical (v1.4+) |
|---|---|
| ``safe_gradio_handler`` | ``safe_ui_handler`` |
| ``raise_gradio_error`` | ``raise_ui_error`` |
| ``raise_gradio_warning`` | ``raise_ui_warning`` |
| ``raise_gradio_info`` | ``raise_ui_info`` |
| ``RequestContext.from_gradio_request`` | ``RequestContext.from_request`` |

The contract:

1. Both the legacy AND canonical names import + resolve.
2. Accessing the LEGACY name from ``backpropagate.ui_security`` emits a
   ``DeprecationWarning`` whose message pins the deprecation cycle text
   (so future changes are deliberate, not silent drifts).
3. The legacy name resolves to the SAME callable as the canonical name —
   they are identity-equal (``is`` comparison), not merely
   behavior-equivalent.
4. Accessing an attribute that's NOT in the legacy table raises
   ``AttributeError`` per the PEP 562 ``__getattr__`` contract.

The deprecation cycle is locked at advisor 2026-05-25 Q4:

* v1.4 — ``DeprecationWarning``
* v1.5 — ``UserWarning``
* v1.6 — ``AttributeError``

The warning-message regex below pins each phase of that cycle so a future
silent shortening of the message (e.g. dropping the v1.5/v1.6 phases)
fails the test rather than the audit catching it post-hoc.
"""

from __future__ import annotations

import re
import warnings
from unittest.mock import MagicMock

import pytest

# Regex that all legacy-alias DeprecationWarnings should match. Pins three
# load-bearing substrings: the v1.4 deprecation phase, the v1.5 escalation
# phase, and the v1.6 removal phase. If a future commit shortens the
# message and drops any of these phases, this test fails — exactly the
# "future changes are deliberate" property the brief asked for.
_DEPRECATION_MESSAGE_REGEX = re.compile(
    r"deprecated in v1\.4.*v1\.5 escalates.*v1\.6 removes",
    re.DOTALL,
)


class TestUiSecurityLegacyAliases:
    """Module-level ``__getattr__`` legacy-alias shim contract."""

    def test_safe_gradio_handler_emits_deprecation_warning(self):
        """``safe_gradio_handler`` emits ``DeprecationWarning`` on access."""
        import backpropagate.ui_security as ui_security

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            handler = ui_security.safe_gradio_handler

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1, (
            "Accessing legacy 'safe_gradio_handler' must emit DeprecationWarning"
        )
        msg = str(deprecation_warnings[0].message)
        assert "safe_gradio_handler" in msg, (
            "Warning must name the legacy symbol"
        )
        assert "safe_ui_handler" in msg, (
            "Warning must name the canonical replacement"
        )
        assert _DEPRECATION_MESSAGE_REGEX.search(msg), (
            f"Warning message must pin the v1.4 → v1.5 → v1.6 cycle. Got: {msg!r}"
        )
        # Identity-equal with the canonical: same callable, not a wrapper.
        assert handler is ui_security.safe_ui_handler, (
            "Legacy alias must resolve to the IDENTITY-SAME callable as the "
            "canonical name (not a wrapper that re-emits the warning per-call)."
        )

    def test_raise_gradio_error_emits_deprecation_warning(self):
        """``raise_gradio_error`` emits ``DeprecationWarning`` on access."""
        import backpropagate.ui_security as ui_security

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = ui_security.raise_gradio_error

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1
        msg = str(deprecation_warnings[0].message)
        assert "raise_gradio_error" in msg
        assert "raise_ui_error" in msg
        assert _DEPRECATION_MESSAGE_REGEX.search(msg), (
            f"Warning must pin the deprecation cycle. Got: {msg!r}"
        )
        assert legacy is ui_security.raise_ui_error

    def test_raise_gradio_warning_emits_deprecation_warning(self):
        """``raise_gradio_warning`` emits ``DeprecationWarning`` on access."""
        import backpropagate.ui_security as ui_security

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = ui_security.raise_gradio_warning

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1
        msg = str(deprecation_warnings[0].message)
        assert "raise_gradio_warning" in msg
        assert "raise_ui_warning" in msg
        assert _DEPRECATION_MESSAGE_REGEX.search(msg)
        assert legacy is ui_security.raise_ui_warning

    def test_raise_gradio_info_emits_deprecation_warning(self):
        """``raise_gradio_info`` emits ``DeprecationWarning`` on access."""
        import backpropagate.ui_security as ui_security

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = ui_security.raise_gradio_info

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1
        msg = str(deprecation_warnings[0].message)
        assert "raise_gradio_info" in msg
        assert "raise_ui_info" in msg
        assert _DEPRECATION_MESSAGE_REGEX.search(msg)
        assert legacy is ui_security.raise_ui_info

    def test_canonical_names_resolve_without_warning(self):
        """Canonical v1.4+ names import + resolve without any warning.

        The migration nudge fires on LEGACY name access only; brand-new
        callers using the canonical names see no noise.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from backpropagate.ui_security import (  # noqa: F401
                raise_ui_error,
                raise_ui_info,
                raise_ui_warning,
                safe_ui_handler,
            )

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0, (
            f"Canonical names must NOT emit DeprecationWarning. "
            f"Got: {[str(w.message) for w in deprecation_warnings]!r}"
        )

    def test_unknown_attribute_raises_attribute_error(self):
        """``__getattr__`` raises ``AttributeError`` for non-legacy names.

        Required by the PEP 562 contract so ``hasattr`` / ``getattr`` with
        a default still work naturally on the module.
        """
        import backpropagate.ui_security as ui_security

        with pytest.raises(AttributeError) as exc_info:
            ui_security.this_symbol_does_not_exist  # noqa: B018
        assert "ui_security" in str(exc_info.value)
        assert "this_symbol_does_not_exist" in str(exc_info.value)


class TestRequestContextLegacyClassmethod:
    """``RequestContext.from_gradio_request`` is a per-class shim, not
    module-level — module ``__getattr__`` cannot intercept class attribute
    lookups. The deprecation lives on the classmethod itself.
    """

    def test_from_gradio_request_emits_deprecation_warning(self):
        """Calling ``RequestContext.from_gradio_request`` warns + delegates."""
        from backpropagate.ui_security import RequestContext

        request = MagicMock()
        request.client = {"host": "192.168.1.42"}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctx = RequestContext.from_gradio_request(
                request, operation="train_legacy"
            )

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1, (
            "RequestContext.from_gradio_request must emit DeprecationWarning"
        )
        msg = str(deprecation_warnings[0].message)
        assert "from_gradio_request" in msg
        assert "from_request" in msg
        assert _DEPRECATION_MESSAGE_REGEX.search(msg), (
            f"Warning must pin the deprecation cycle. Got: {msg!r}"
        )

        # And the delegation: it should produce a normal RequestContext.
        assert ctx.client_ip == "192.168.1.42"
        assert ctx.operation == "train_legacy"
        assert len(ctx.request_id) == 8

    def test_from_request_canonical_does_not_warn(self):
        """``RequestContext.from_request`` produces no DeprecationWarning."""
        from backpropagate.ui_security import RequestContext

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctx = RequestContext.from_request(operation="train_canonical")

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0, (
            f"Canonical from_request must NOT emit DeprecationWarning. "
            f"Got: {[str(w.message) for w in deprecation_warnings]!r}"
        )
        # Sanity — the canonical form actually returns something usable.
        assert ctx.operation == "train_canonical"
        assert ctx.client_ip == "unknown"  # No request → default sentinel

    def test_legacy_and_canonical_produce_equivalent_contexts(self):
        """Both classmethods produce contexts with the same shape.

        We can't compare ``request_id`` (uuid) or ``timestamp`` (time.time),
        but everything else should be byte-identical for the same
        ``(request, operation)`` input.
        """
        from backpropagate.ui_security import RequestContext

        request = MagicMock()
        request.client = {"host": "127.0.0.1"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence the legacy warning
            legacy_ctx = RequestContext.from_gradio_request(
                request, operation="upload"
            )
            canonical_ctx = RequestContext.from_request(
                request, operation="upload"
            )

        # Identical except for the uuid request_id + monotonic timestamp.
        assert legacy_ctx.client_ip == canonical_ctx.client_ip
        assert legacy_ctx.operation == canonical_ctx.operation
        assert legacy_ctx.user_id == canonical_ctx.user_id
