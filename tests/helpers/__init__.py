"""Test helper utilities — shared scaffolding for the backpropagate test suite.

This subpackage is the canonical home for shared test-only fixtures and
utilities. It replaces the legacy ``tests/test_helpers.py`` module, which
was a helpers module masquerading as a test file (zero ``def test_``
functions, but pytest collected it on every run because of the ``test_*``
naming convention).

Modules:

- ``callbacks`` — spies, trackers, collectors for verifying callback
  invocations across sync + threaded code paths. Used by the
  ``callback_spy`` / ``callback_tracker`` / ``async_callback_collector``
  fixtures in ``tests/conftest.py``.
- ``asgi`` — httpx + ASGI helpers for testing the Reflex ``api_transformer``
  auth middleware (Wave 6 enabler; degrades gracefully via
  ``pytest.importorskip`` when ``httpx`` is not installed).
- ``ws`` — WebSocket-upgrade helpers for the same auth-middleware surface
  (validates auth PRE-``websocket.accept()`` per DESIGN_BRIEF requirement 5).

Public API: the symbols re-exported below match the legacy
``tests.test_helpers`` shape so the conftest.py diff stays one-character
(``tests.test_helpers`` -> ``tests.helpers``).
"""

from tests.helpers.callbacks import (
    AsyncCallbackCollector,
    CallbackInvocation,
    CallbackSpy,
    CallbackTracker,
    assert_callback_sequence,
    wait_for_callback,
)

__all__ = [
    "AsyncCallbackCollector",
    "CallbackInvocation",
    "CallbackSpy",
    "CallbackTracker",
    "assert_callback_sequence",
    "wait_for_callback",
]
