"""Regression tests for four defects found while writing ``test_ui_cov_security.py``.

Each test failed on ``main`` before the matching fix in ``ui_security.py``:

1. ``SecureSessionHandler`` keyed its CSRF store (and session registry) on
   ``access_token[:32]``. Every HS256 JWT begins with the same 36-character
   header (``eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9``), so that "session id" was a
   constant: one global CSRF token slot shared by every user. Anyone able to
   log in could validate a forged cross-site request for ANOTHER user's session
   with their own CSRF token (CSRF protection bypass). ``logout`` also only
   edited a dict ``validate_request`` never read, so a logged-out access token
   kept working.
2. ``CSRFProtection.validate_token`` called ``hmac.compare_digest`` on ``str``
   operands; a non-ASCII (or non-string) token raised ``TypeError`` instead of
   returning ``(False, "Invalid CSRF token")``.
3. ``FileValidator.validate`` accepted any object with a ``.name`` attribute, and
   a ``pathlib.Path`` has one - the *basename* - so passing a real ``Path`` made
   the size cap and magic-byte checks look for the file relative to the cwd,
   miss it, and silently pass.
4. ``validate_numeric_input`` let ``nan`` / ``inf`` through every min/max guard
   (NaN compares false to everything).

Nothing is mocked.
"""

from __future__ import annotations

import pytest

from backpropagate import ui_security as sec
from backpropagate.exceptions import UserInputError

pytest.importorskip("jwt", reason="PyJWT required for session tests")

_SECRET = "regression-secret-0123456789abcdef012"


def _handler():
    return sec.SecureSessionHandler(sec.JWTConfig(secret=_SECRET))


class TestSessionHandlerCsrfIsolation:
    def test_hs256_tokens_share_a_32_char_prefix(self):
        """The root cause, pinned: a prefix is not a usable session identifier."""
        h = _handler()
        a, b = h.login("alice")["access_token"], h.login("bob")["access_token"]
        assert a[:32] == b[:32]

    def test_another_users_csrf_token_does_not_validate_my_request(self):
        h = _handler()
        alice = h.login("alice")
        bob = h.login("bob")
        valid, user, msg = h.validate_request(alice["access_token"], bob["csrf_token"])
        assert (valid, user) == (False, None)
        assert msg == "Invalid CSRF token"

    def test_a_later_login_does_not_invalidate_an_earlier_users_csrf_token(self):
        h = _handler()
        alice = h.login("alice")
        h.login("bob")
        assert h.validate_request(alice["access_token"], alice["csrf_token"]) == (
            True, "alice", "Request valid")

    def test_each_session_gets_its_own_csrf_slot(self):
        h = _handler()
        for name in ("a", "b", "c"):
            h.login(name)
        assert len(h.csrf._tokens) == 3
        assert len(h._active_sessions) == 3

    def test_refreshed_session_validates_and_keeps_old_session_independent(self):
        h = _handler()
        alice = h.login("alice")
        ok, new, _ = h.refresh_session(alice["refresh_token"])
        assert ok
        assert h.validate_request(new["access_token"], new["csrf_token"])[0] is True
        # the original access token's CSRF token is unaffected by the refresh
        assert h.validate_request(alice["access_token"], alice["csrf_token"])[0] is True
        # and the two sessions' CSRF tokens are not interchangeable
        assert h.validate_request(new["access_token"], alice["csrf_token"])[0] is False


class TestLogoutRevokes:
    def test_logged_out_access_token_is_refused(self):
        h = _handler()
        tokens = h.login("alice")
        assert h.validate_request(tokens["access_token"], "", require_csrf=False)[0] is True
        h.logout(tokens["access_token"])
        valid, user, msg = h.validate_request(tokens["access_token"], tokens["csrf_token"])
        assert (valid, user) == (False, None)
        assert "session" in msg.lower()

    def test_logout_of_one_session_leaves_others_valid(self):
        h = _handler()
        alice, bob = h.login("alice"), h.login("bob")
        h.logout(alice["access_token"])
        assert h.validate_request(bob["access_token"], bob["csrf_token"])[0] is True

    def test_valid_signature_but_never_logged_in_is_refused(self):
        h = _handler()
        h.login("alice")
        stray = h.jwt.create_token("mallory")  # correctly signed, no session registered
        assert h.validate_request(stray, "x", require_csrf=False)[0] is False


class TestCsrfCompareIsTotal:
    @pytest.mark.parametrize("token", ["éclair", "‮", "tok\x00en", "", "x" * 10_000])
    def test_odd_string_tokens_are_invalid_not_exceptions(self, token):
        c = sec.CSRFProtection()
        c.generate_token("s")
        assert c.validate_token("s", token) == (False, "Invalid CSRF token")

    @pytest.mark.parametrize("token", [None, 123, b"bytes", ["a"]])
    def test_non_string_tokens_are_invalid_not_exceptions(self, token):
        c = sec.CSRFProtection()
        c.generate_token("s")
        assert c.validate_token("s", token) == (False, "Invalid CSRF token")  # type: ignore[arg-type]

    def test_good_token_still_validates(self):
        c = sec.CSRFProtection()
        assert c.validate_token("s", c.generate_token("s"))[0] is True


class TestFileValidatorAcceptsPathObjects:
    def _cfg(self, **kw):
        return sec.SecurityConfig(**kw)

    def test_path_object_gets_the_size_cap(self, tmp_path):
        f = tmp_path / "big.jsonl"
        f.write_bytes(b"{" + b"x" * (2 * 1024 * 1024))
        ok, msg, path = sec.FileValidator(max_size_mb=1, config=self._cfg()).validate(f)
        assert ok is False and path is None and "too large" in msg

    def test_path_object_gets_the_magic_byte_check(self, tmp_path):
        f = tmp_path / "spoof.jsonl"
        f.write_bytes(b"<!DOCTYPE html><html>")
        ok, msg, _ = sec.FileValidator(config=self._cfg(validate_file_magic=True)).validate(f)
        assert ok is False and "magic-bytes check failed" in msg

    def test_clean_path_object_returns_the_full_path(self, tmp_path):
        f = tmp_path / "ok.jsonl"
        f.write_text('{"a": 1}\n', encoding="utf-8")
        ok, msg, path = sec.FileValidator(config=self._cfg(validate_file_magic=True)).validate(f)
        assert (ok, msg, path) == (True, "", f)

    def test_str_path_is_treated_as_a_path(self, tmp_path):
        f = tmp_path / "big.jsonl"
        f.write_bytes(b"{" + b"x" * (2 * 1024 * 1024))
        ok, msg, _ = sec.FileValidator(max_size_mb=1, config=self._cfg()).validate(str(f))
        assert ok is False and "too large" in msg

    def test_name_attribute_objects_keep_working(self, tmp_path):
        from types import SimpleNamespace

        f = tmp_path / "ok.jsonl"
        f.write_text("{}", encoding="utf-8")
        ok, _msg, path = sec.FileValidator(config=self._cfg()).validate(SimpleNamespace(name=str(f)))
        assert ok is True and path == f


class TestNumericValidationRejectsNonFinite:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "nan", "inf", "-Infinity"])
    def test_non_finite_is_rejected_even_without_bounds(self, bad):
        with pytest.raises(UserInputError, match="finite"):
            sec.validate_numeric_input(bad, "lr")

    def test_non_finite_is_rejected_when_bounds_would_have_missed_it(self):
        with pytest.raises(UserInputError, match="finite"):
            sec.validate_numeric_input(float("nan"), "lr", min_value=0, max_value=1)

    def test_finite_values_unchanged(self):
        assert sec.validate_numeric_input("0.5", "lr", 0, 1) == 0.5
        assert sec.validate_numeric_input(1e300, "x") == 1e300
