"""Adversarial / branch tests for ``backpropagate.ui_security``.

The module is the UI's path sandbox, upload validator, rate/concurrency
limiter, session + JWT + CSRF primitives and CSP builder. These tests drive
the pieces the existing suites leave open with hostile inputs (traversal,
symlinks, forbidden bases, spoofed uploads, malformed credentials) and assert
the structured error codes / return tuples.

Mocked: only real boundaries - ``Path.home`` / env vars (to point the sandbox at
``tmp_path``), ``Path.resolve`` / ``Path.is_symlink`` failure injection for the
fail-closed branches, and ``backpropagate.gpu_safety.get_gpu_status`` for the
health check (no GPU is touched).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from backpropagate import ui_security as sec
from backpropagate.exceptions import BackpropagateError, UserInputError


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point ``Path.home()`` at an empty directory under tmp_path."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    return home.resolve()


@pytest.fixture
def fresh_session_manager():
    sec.SessionManager._instance = None
    yield sec.SessionManager()
    sec.SessionManager._instance = None


# =============================================================================
# Client IP extraction
# =============================================================================


class TestExtractClientIp:
    @pytest.mark.parametrize(
        ("request_obj", "expected"),
        [
            (None, "unknown"),
            (SimpleNamespace(), "unknown"),                                   # no .client
            (SimpleNamespace(client=None), "unknown"),
            (SimpleNamespace(client=SimpleNamespace(host="1.2.3.4", port=1)), "1.2.3.4"),
            (SimpleNamespace(client={"host": "5.6.7.8"}), "5.6.7.8"),
            (SimpleNamespace(client={"host": ""}), "unknown"),
            (SimpleNamespace(client={"host": 42}), "unknown"),
            (SimpleNamespace(client={}), "unknown"),
            (SimpleNamespace(client="9.9.9.9"), "9.9.9.9"),
            (SimpleNamespace(client=""), "unknown"),
            (SimpleNamespace(client=12345), "unknown"),
            (SimpleNamespace(client=SimpleNamespace(host=None)), "unknown"),
            (SimpleNamespace(client=SimpleNamespace(host="")), "unknown"),
        ],
    )
    def test_shapes(self, request_obj, expected):
        assert sec._extract_client_ip(request_obj) == expected

    def test_x_forwarded_for_is_never_trusted(self):
        req = SimpleNamespace(client=SimpleNamespace(host="10.0.0.1"),
                              headers={"x-forwarded-for": "6.6.6.6"})
        assert sec._extract_client_ip(req) == "10.0.0.1"


# =============================================================================
# Output sandbox
# =============================================================================


class TestForbiddenOutputBases:
    def test_denylist_covers_system_trees_and_home_credential_dirs(self, fake_home):
        bases = {str(p) for p in sec._forbidden_output_bases()}
        for sub in (".ssh", ".aws", ".kube", ".docker", ".gnupg", ".config"):
            assert str((fake_home / sub).resolve()) in bases
        assert any(b.endswith("etc") for b in bases)

    def test_appdata_crypto_is_denylisted_when_set(self, fake_home, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
        bases = {str(p) for p in sec._forbidden_output_bases()}
        assert str((tmp_path / "AppData" / "Microsoft" / "Crypto").resolve()) in bases

    def test_unresolvable_entries_are_skipped_not_fatal(self, fake_home, monkeypatch):
        """Mocked: ``Path.resolve`` raises for the Windows-style roots."""
        real = Path.resolve

        def flaky(self, *a, **k):
            if "Windows" in str(self) or "ProgramData" in str(self):
                raise OSError("no such drive")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", flaky)
        bases = {str(p) for p in sec._forbidden_output_bases()}
        assert bases and not any("ProgramData" in b for b in bases)

    def test_duplicate_resolved_entries_are_deduped(self, fake_home):
        bases = [str(p) for p in sec._forbidden_output_bases()]
        assert len(bases) == len(set(bases))


class TestIsForbiddenOutputBase:
    @pytest.mark.parametrize("sub", [".ssh", ".aws", ".kube", ".docker", ".gnupg", ".config"])
    def test_credential_dirs_under_home_are_forbidden(self, fake_home, sub):
        assert sec._is_forbidden_output_base(fake_home / sub) is True
        assert sec._is_forbidden_output_base(fake_home / sub / "deep" / "er") is True

    def test_ordinary_subdir_of_home_is_allowed(self, fake_home):
        assert sec._is_forbidden_output_base(fake_home / "work" / "out") is False

    def test_home_itself_is_not_special_cased_as_allowed_by_prefix(self, fake_home):
        # home == candidate falls through to the denylist loop; home is not on it.
        assert sec._is_forbidden_output_base(fake_home) is False

    def test_system_roots_are_forbidden(self, tmp_path, fake_home):
        for entry in sec._forbidden_output_bases():
            assert sec._is_forbidden_output_base(entry) is True
            assert sec._is_forbidden_output_base(entry / "child") is True

    def test_dotdot_traversal_into_credentials_is_normalised(self, fake_home):
        sneaky = fake_home / "work" / ".." / ".ssh" / "keys"
        assert sec._is_forbidden_output_base(sneaky) is True

    def test_unresolvable_candidate_fails_closed(self, fake_home, monkeypatch):
        """Mocked: ``Path.resolve`` raises ``OSError`` for the candidate."""
        target = fake_home / "x"
        real = Path.resolve

        def boom(self, *a, **k):
            if self == target:
                raise OSError("cannot resolve")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", boom)
        assert sec._is_forbidden_output_base(target) is True

    def test_runtime_error_on_resolve_fails_closed(self, fake_home, monkeypatch):
        """Mocked: symlink-loop style ``RuntimeError`` from ``Path.resolve``."""
        target = fake_home / "loop"
        real = Path.resolve

        def boom(self, *a, **k):
            if self == target:
                raise RuntimeError("Symlink loop")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", boom)
        assert sec._is_forbidden_output_base(target) is True

    def test_home_unresolvable_falls_back_to_denylist_only(self, fake_home, tmp_path, monkeypatch):
        """Mocked: resolving ``Path.home()`` raises; the denylist still rejects /etc-style roots."""
        real = Path.resolve
        home = Path.home()

        def boom(self, *a, **k):
            if self == home:
                raise OSError("home gone")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", boom)
        assert sec._is_forbidden_output_base(tmp_path / "anything") is False
        for entry in sec._forbidden_output_bases():
            assert sec._is_forbidden_output_base(entry) is True
            break

    def test_symlink_below_home_is_forbidden_and_logged(self, fake_home, tmp_path, monkeypatch):
        target = tmp_path / "outside"
        target.mkdir()
        link = fake_home / "innocent"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this host")
        events = []
        monkeypatch.setattr(sec, "log_security_event", lambda name, **kw: events.append((name, kw)))
        assert sec._is_forbidden_output_base(link) is True
        assert sec._is_forbidden_output_base(link / "sub") is True
        assert events and events[0][0] == "ui_output_dir_symlink_below_home"
        assert events[0][1]["symlink_component"].endswith("innocent")

    def test_symlink_stat_failure_fails_closed(self, fake_home, monkeypatch):
        """Mocked: ``Path.is_symlink`` raises ``OSError`` for a component below home."""
        sub = fake_home / "work"
        sub.mkdir()
        real = Path.is_symlink

        def boom(self):
            if self == sub:
                raise OSError("stat denied")
            return real(self)

        monkeypatch.setattr(Path, "is_symlink", boom)
        assert sec._is_forbidden_output_base(sub) is True

    def test_appdata_crypto_inside_home_is_forbidden(self, fake_home, monkeypatch):
        appdata = fake_home / "AppData" / "Roaming"
        monkeypatch.setenv("APPDATA", str(appdata))
        assert sec._is_forbidden_output_base(appdata / "Microsoft" / "Crypto" / "Keys") is True
        assert sec._is_forbidden_output_base(appdata / "Other") is False

    def test_appdata_resolution_error_is_ignored(self, fake_home, monkeypatch):
        """Mocked: resolving the APPDATA crypto root raises."""
        monkeypatch.setenv("APPDATA", str(fake_home / "AppData"))
        real = Path.resolve

        def boom(self, *a, **k):
            if self.name == "Crypto":
                raise OSError("no")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", boom)
        assert sec._is_forbidden_output_base(fake_home / "work") is False


class TestGetUiOutputDir:
    def test_default_is_created_under_home(self, fake_home):
        out = sec.get_ui_output_dir()
        assert out == (fake_home / ".backpropagate" / "ui-outputs").resolve()
        assert out.is_dir()

    def test_env_override_is_honoured_and_created(self, fake_home, tmp_path, monkeypatch):
        target = tmp_path / "custom" / "out"
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(target))
        assert sec.get_ui_output_dir() == target.resolve()
        assert target.is_dir()

    @pytest.mark.parametrize("sub", [".ssh", ".aws", ".config"])
    def test_override_into_credential_dir_is_refused_with_code(self, fake_home, monkeypatch, sub):
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(fake_home / sub))
        with pytest.raises(BackpropagateError) as exc:
            sec.get_ui_output_dir()
        err = exc.value
        assert err.code == "UI_OUTPUT_DIR_FORBIDDEN"
        assert err.details["source"] == "env"
        assert "forbidden" in err.message
        assert not (fake_home / sub).exists()  # nothing was mkdir'd into it

    def test_default_pointing_into_forbidden_base_reports_default_source(
        self, fake_home, monkeypatch
    ):
        """Mocked: the forbidden-base check is forced True for the default path."""
        monkeypatch.setattr(sec, "_is_forbidden_output_base", lambda p: True)
        with pytest.raises(BackpropagateError) as exc:
            sec.get_ui_output_dir()
        assert exc.value.details["source"] == "default"


# =============================================================================
# SecurityConfig / env loading
# =============================================================================


class TestLoadConfigFromEnv:
    def test_int_bool_and_unknown_overrides(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_SECURITY__TRAINING_RATE_LIMIT", "9")
        monkeypatch.setenv("BACKPROPAGATE_SECURITY__CSRF_ENABLED", "off")
        monkeypatch.setenv("BACKPROPAGATE_SECURITY__LOG_FORMAT_JSON", "YES")
        monkeypatch.setenv("BACKPROPAGATE_SECURITY__NOT_A_FIELD", "x")
        monkeypatch.setenv("OTHER_PREFIX__TRAINING_RATE_LIMIT", "99")
        cfg = sec.load_config_from_env(sec.SecurityConfig())
        assert cfg.training_rate_limit == 9
        assert cfg.csrf_enabled is False and cfg.log_format_json is True
        assert not hasattr(cfg, "not_a_field")

    def test_invalid_int_keeps_default_and_warns(self, monkeypatch, caplog):
        monkeypatch.setenv("BACKPROPAGATE_SECURITY__MAX_UPLOAD_SIZE_MB", "lots")
        with caplog.at_level(logging.WARNING, logger=sec.logger.name):
            cfg = sec.load_config_from_env()
        assert cfg.max_upload_size_mb == 500
        assert any("Invalid int value" in r.getMessage() for r in caplog.records)


# =============================================================================
# Rate limiter internals
# =============================================================================


class TestEnhancedRateLimiterInternals:
    def test_cleanup_drops_idle_clients_after_interval(self):
        rl = sec.EnhancedRateLimiter(max_requests=5, window_seconds=10)
        rl._requests = {"old": [time.time() - 100], "mixed": [time.time() - 100, time.time()]}
        rl._last_cleanup = time.time() - 1000
        rl._cleanup_old_entries()
        assert "old" not in rl._requests and len(rl._requests["mixed"]) == 1

    def test_cleanup_is_throttled_within_interval(self):
        rl = sec.EnhancedRateLimiter(window_seconds=10)
        rl._requests = {"old": [time.time() - 100]}
        rl._cleanup_old_entries()  # _last_cleanup is "now" from __init__
        assert "old" in rl._requests

    def test_client_cap_cleans_expired_then_evicts_oldest(self):
        rl = sec.EnhancedRateLimiter(window_seconds=60)
        rl._max_clients = 3
        now = time.time()
        rl._requests = {f"c{i}": [now - i] for i in range(6)}
        rl._requests["expired"] = [now - 1000]
        rl._enforce_client_cap()
        assert len(rl._requests) == 3
        assert "expired" not in rl._requests
        # the most recently active clients survive
        assert set(rl._requests) == {"c0", "c1", "c2"}

    def test_client_cap_noop_under_limit(self):
        rl = sec.EnhancedRateLimiter()
        rl._requests = {"a": [time.time()]}
        rl._enforce_client_cap()
        assert set(rl._requests) == {"a"}

    def test_burst_allowance_extends_the_limit(self):
        rl = sec.EnhancedRateLimiter(max_requests=1, burst_allowance=2, window_seconds=60)
        assert [rl.is_allowed() for _ in range(4)] == [True, True, True, False]

    def test_require_raises_with_wait_and_operation(self):
        rl = sec.EnhancedRateLimiter(max_requests=1, window_seconds=60, operation_name="export")
        rl.require()
        with pytest.raises(sec.RateLimitExceeded) as exc:
            rl.require()
        assert exc.value.operation == "export" and 0 < exc.value.wait_seconds <= 60
        assert "export" in str(exc.value)

    def test_each_ip_has_its_own_bucket(self):
        rl = sec.EnhancedRateLimiter(max_requests=1)
        a = SimpleNamespace(client=SimpleNamespace(host="1.1.1.1"))
        b = SimpleNamespace(client=SimpleNamespace(host="2.2.2.2"))
        assert rl.is_allowed(a) and rl.is_allowed(b)
        assert not rl.is_allowed(a)


# =============================================================================
# File validation / filename sanitising
# =============================================================================


def _upload(path):
    """The UI's upload adapter shape: an object whose ``.name`` is the full path."""
    return SimpleNamespace(name=str(path))


class TestFileValidatorAdversarial:
    def _validator(self, **cfg):
        return sec.FileValidator(config=sec.SecurityConfig(**cfg))

    def test_none_is_rejected(self):
        assert self._validator().validate(None) == (False, "No file provided", None)

    def test_object_with_broken_name_is_rejected(self):
        class Bad:
            @property
            def name(self):
                raise RuntimeError("no name")

        ok, msg, path = self._validator().validate(Bad())
        assert (ok, path) == (False, None) and msg.startswith("Invalid file object")

    @pytest.mark.parametrize("name", ["evil.html", "x.SVG", "run.exe", "a.PY", "p.php", "j.jar"])
    def test_dangerous_extensions_are_blocked_and_logged(self, name, monkeypatch):
        events = []
        monkeypatch.setattr(sec, "log_security_event", lambda n, **kw: events.append(n))
        ok, msg, path = self._validator().validate(name)
        assert ok is False and path is None and "not allowed for security reasons" in msg
        assert events == ["dangerous_file_blocked"]

    def test_double_extension_uses_final_suffix(self):
        ok, msg, _ = self._validator().validate("data.jsonl.html")
        assert ok is False and "'.html'" in msg

    def test_unsupported_extension_lists_allowed_ones(self):
        ok, msg, _ = self._validator().validate("notes.docx")
        assert ok is False and ".jsonl" in msg and "not supported" in msg

    def test_oversized_file_is_rejected_with_sizes(self, tmp_path):
        f = tmp_path / "big.jsonl"
        f.write_bytes(b"{" + b"x" * (2 * 1024 * 1024))
        v = sec.FileValidator(max_size_mb=1, config=sec.SecurityConfig())
        ok, msg, path = v.validate(_upload(f))
        assert ok is False and path is None
        assert "too large" in msg and "Maximum: 1MB" in msg

    def test_nonexistent_but_allowed_file_passes_size_stage(self, tmp_path):
        ok, _msg, path = self._validator().validate(_upload(tmp_path / "missing.jsonl"))
        assert ok is True and path == tmp_path / "missing.jsonl"

    def test_magic_check_rejects_spoofed_extension_when_enabled(self, tmp_path, monkeypatch):
        events = []
        monkeypatch.setattr(sec, "log_security_event", lambda n, **kw: events.append((n, kw)))
        f = tmp_path / "pretend.jsonl"
        f.write_bytes(b"<!DOCTYPE html><html><script>alert(1)</script>")
        ok, msg, path = self._validator(validate_file_magic=True).validate(_upload(f))
        assert ok is False and path is None
        assert "magic-bytes check failed" in msg and "HTML/script" in msg
        assert [n for n, _ in events] == ["suspicious_file_content", "file_magic_rejected"]

    def test_magic_check_accepts_genuine_file(self, tmp_path):
        f = tmp_path / "ok.jsonl"
        f.write_text('{"a": 1}\n', encoding="utf-8")
        ok, _msg, path = self._validator(validate_file_magic=True).validate(_upload(f))
        assert ok is True and path == f

    def test_magic_check_on_by_default_rejects_a_spoof(self, tmp_path):
        f = tmp_path / "pretend.jsonl"
        f.write_bytes(b"<html>")
        ok, msg, path = self._validator().validate(_upload(f))
        assert ok is False and path is None and "magic-bytes" in msg

    def test_magic_check_can_be_turned_off(self, tmp_path):
        f = tmp_path / "pretend.jsonl"
        f.write_bytes(b"<html>")
        ok, _msg, path = self._validator(validate_file_magic=False).validate(_upload(f))
        assert ok is True and path == f

    @pytest.mark.parametrize("name", ["data.jsonl", ".hidden.jsonl"])
    def test_filename_sanitised_event_is_emitted_when_name_changes(self, name, monkeypatch):
        events = []
        monkeypatch.setattr(sec, "log_security_event", lambda n, **kw: events.append((n, kw)))
        ok, _msg, _path = self._validator().validate(SimpleNamespace(name=name))
        assert ok is True
        assert events[0][0] == "filename_sanitized"
        assert events[0][1]["original"] == name
        assert events[0][1]["sanitized"] == sec.sanitize_filename(name) != name

    def test_clean_filename_emits_no_sanitise_event(self, monkeypatch):
        events = []
        monkeypatch.setattr(sec, "log_security_event", lambda n, **kw: events.append(n))
        assert self._validator().validate(SimpleNamespace(name="clean.jsonl"))[0] is True
        assert events == []


class TestSanitizeFilenameAdversarial:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("../../etc/passwd", "_.._etc_passwd"),   # leading dots stripped, separators flattened
            ("..\\..\\windows\\system32", "_.._windows_system32"),
            ("a\x00b.txt", "ab.txt"),
            ("  ...hidden  ", "hidden"),
            ("", "unnamed_file"),
            ("...", "unnamed_file"),
            ("tab\tname\x7f.txt", "tabname.txt"),
        ],
    )
    def test_cases(self, raw, expected):
        assert sec.sanitize_filename(raw) == expected

    def test_overlong_name_is_truncated_but_keeps_extension(self):
        out = sec.sanitize_filename("a" * 400 + ".jsonl")
        assert len(out) == 255 and out.endswith(".jsonl")

    @pytest.mark.parametrize("raw", ["../../x", "a/b\\c", "\x00../x"])
    def test_result_never_contains_separators_or_nul(self, raw):
        out = sec.sanitize_filename(raw)
        assert "/" not in out and "\\" not in out and "\x00" not in out


# =============================================================================
# Numeric / string validation
# =============================================================================


class TestValidateNumericInput:
    def test_valid_values_and_bounds(self):
        assert sec.validate_numeric_input("3.5", "lr", 0, 10) == 3.5
        assert sec.validate_numeric_input(10, "n", 0, 10) == 10.0
        assert sec.validate_numeric_input(0, "n", 0, 10) == 0.0

    def test_none_handling(self):
        assert sec.validate_numeric_input(None, "n", allow_none=True) is None
        with pytest.raises(UserInputError, match="n is required"):
            sec.validate_numeric_input(None, "n")

    @pytest.mark.parametrize("bad", ["abc", "", [], {}, object()])
    def test_non_numeric_is_rejected(self, bad):
        with pytest.raises(UserInputError, match="must be a number"):
            sec.validate_numeric_input(bad, "n")

    def test_below_min_and_above_max(self):
        with pytest.raises(UserInputError, match="at least"):
            sec.validate_numeric_input(-1, "n", min_value=0)
        with pytest.raises(UserInputError, match="at most"):
            sec.validate_numeric_input(11, "n", max_value=10)

    def test_error_carries_structured_code(self):
        with pytest.raises(UserInputError) as exc:
            sec.validate_numeric_input("x", "n")
        assert exc.value.code == "INPUT_VALIDATION_FAILED"


class TestValidateStringInput:
    def test_passes_through_and_strips_nul(self):
        assert sec.validate_string_input("ab\x00c", "s") == "abc"

    def test_none_handling(self):
        assert sec.validate_string_input(None, "s", allow_none=True) is None
        with pytest.raises(UserInputError, match="s is required"):
            sec.validate_string_input(None, "s")

    def test_length_and_emptiness(self):
        with pytest.raises(UserInputError, match="too long"):
            sec.validate_string_input("x" * 11, "s", max_length=10)
        with pytest.raises(UserInputError, match="cannot be empty"):
            sec.validate_string_input("   ", "s")
        assert sec.validate_string_input("", "s", allow_empty=True) == ""
        with pytest.raises(UserInputError, match="too short"):
            sec.validate_string_input("ab", "s", min_length=3)

    def test_nul_only_is_empty(self):
        with pytest.raises(UserInputError, match="cannot be empty"):
            sec.validate_string_input("\x00\x00", "s")

    def test_pattern_enforced(self):
        assert sec.validate_string_input("abc-1", "id", pattern=r"^[a-z0-9-]+$") == "abc-1"
        with pytest.raises(UserInputError, match="invalid format"):
            sec.validate_string_input("abc/../1", "id", pattern=r"^[a-z0-9-]+$")

    def test_non_string_is_coerced(self):
        assert sec.validate_string_input(123, "n") == "123"


class TestValidateAndLogRequest:
    def test_disabled_logging_emits_nothing(self, monkeypatch, caplog):
        monkeypatch.setattr(sec.DEFAULT_SECURITY_CONFIG, "log_all_requests", False)
        with caplog.at_level(logging.INFO, logger="backpropagate.security.ui"):
            sec.validate_and_log_request("train", prompt="x")
        assert caplog.records == []

    def test_enabled_logging_truncates_and_types_params(self, monkeypatch, caplog):
        monkeypatch.setattr(sec.DEFAULT_SECURITY_CONFIG, "log_all_requests", True)
        req = SimpleNamespace(client=SimpleNamespace(host="7.7.7.7"))
        with caplog.at_level(logging.INFO, logger="backpropagate.security.ui"):
            sec.validate_and_log_request(
                "train", req, long="x" * 300, n=3, flag=True, obj=[1], secret_dict={"k": "v"},
            )
        rec = caplog.records[-1]
        assert rec.operation == "train" and rec.client_id == "7.7.7.7"
        assert rec.params["long"] == "x" * 100 + "..."
        assert rec.params["n"] == 3 and rec.params["flag"] is True
        assert rec.params["obj"] == "list" and rec.params["secret_dict"] == "dict"  # contents never logged


# =============================================================================
# Security event logging, CSRF origin check
# =============================================================================


class TestSecurityEventLogging:
    def test_log_security_event_respects_config_flag(self, monkeypatch, caplog):
        monkeypatch.setattr(sec.DEFAULT_SECURITY_CONFIG, "log_security_events", False)
        with caplog.at_level(logging.DEBUG, logger="backpropagate.security.events"):
            sec.log_security_event("x_event")
        assert caplog.records == []

    def test_log_security_event_emits_structured_record(self, monkeypatch, caplog):
        monkeypatch.setattr(sec.DEFAULT_SECURITY_CONFIG, "log_security_events", True)
        with caplog.at_level(logging.DEBUG, logger="backpropagate.security.events"):
            sec.log_security_event("x_event", reason="because")
        rec = caplog.records[-1]
        assert rec.event_type == "x_event" and rec.reason == "because"
        assert isinstance(rec.timestamp, float)

    def test_security_logger_is_a_singleton_and_maps_severity(self, caplog):
        a, b = sec.SecurityLogger(), sec.SecurityLogger()
        assert a is b
        with caplog.at_level(logging.DEBUG, logger="backpropagate.security.events"):
            a.log("sev_event", severity="warning")
            a.log("bogus_sev", severity="NOPE")
        levels = {r.event_type: r.levelno for r in caplog.records if hasattr(r, "event_type")}
        assert levels["sev_event"] == logging.WARNING and levels["bogus_sev"] == logging.INFO


class TestCheckCsrfProtection:
    def _req(self, **headers):
        return SimpleNamespace(headers=headers)

    def test_disabled_config_allows_everything(self):
        cfg = sec.SecurityConfig(csrf_enabled=False)
        assert sec.check_csrf_protection(self._req(origin="http://evil.com"), cfg) is True

    def test_no_request_is_allowed(self):
        assert sec.check_csrf_protection(None, sec.SecurityConfig()) is True

    def test_missing_headers_are_allowed(self):
        assert sec.check_csrf_protection(self._req(), sec.SecurityConfig()) is True

    @pytest.mark.parametrize(
        "origin",
        ["http://localhost:3000", "https://127.0.0.1", "http://[::1]:7860"],
    )
    def test_loopback_origins_pass(self, origin):
        assert sec.check_csrf_protection(self._req(origin=origin), sec.SecurityConfig()) is True

    @pytest.mark.parametrize(
        "origin",
        ["http://evil.com", "http://localhost.evil.com", "http://evil.com/localhost",
         "null", "http://127.0.0.1.evil.com", "http://user@evil.com", "javascript:alert(1)"],
    )
    def test_foreign_origins_fail_and_are_logged(self, origin, monkeypatch):
        events = []
        monkeypatch.setattr(sec, "log_security_event", lambda n, **kw: events.append((n, kw)))
        assert sec.check_csrf_protection(self._req(origin=origin), sec.SecurityConfig()) is False
        assert events[0][0] == "csrf_check_failed"

    def test_foreign_referer_fails_even_with_good_origin(self):
        req = self._req(origin="http://localhost", referer="http://evil.com/page")
        assert sec.check_csrf_protection(req, sec.SecurityConfig()) is False

    def test_localhost_only_off_skips_origin_policing(self):
        cfg = sec.SecurityConfig(csrf_localhost_only=False)
        assert sec.check_csrf_protection(self._req(origin="http://evil.com"), cfg) is True

    def test_unparseable_url_counts_as_foreign(self):
        # An unterminated IPv6 literal makes urlparse raise ValueError.
        req = self._req(origin="http://[::1")
        assert sec.check_csrf_protection(req, sec.SecurityConfig()) is False

    def test_defaults_to_global_config(self):
        assert sec.check_csrf_protection(self._req(origin="http://evil.com")) is False


# =============================================================================
# Health
# =============================================================================


class TestHealth:
    def test_health_status_dict_without_gpu(self):
        hs = sec.HealthStatus(status="healthy", version="1", uptime_seconds=1.5)
        d = hs.to_dict()
        assert "gpu" not in d and d["status"] == "healthy" and d["timestamp"]

    def test_health_status_dict_with_gpu(self):
        hs = sec.HealthStatus(status="healthy", version="1", uptime_seconds=1.5, gpu_available=True,
                              gpu_name="X", gpu_temperature_c=50.0, timestamp="t")
        assert hs.to_dict()["gpu"]["name"] == "X" and hs.timestamp == "t"

    def _patch_gpu(self, monkeypatch, status=None, exc=None):
        import backpropagate.gpu_safety as gs

        def fake():
            if exc:
                raise exc
            return status

        monkeypatch.setattr(gs, "get_gpu_status", fake)

    def test_gpu_available_populates_fields(self, monkeypatch):
        """Mocked: ``gpu_safety.get_gpu_status`` (no GPU touched)."""
        st = SimpleNamespace(available=True, device_name="RTX", vram_used_gb=4.0,
                             vram_total_gb=32.0, temperature_c=60.0)
        self._patch_gpu(monkeypatch, st)
        hs = sec.get_health_status(include_gpu=True)
        assert hs.status == "healthy" and hs.gpu_name == "RTX" and hs.gpu_memory_total_gb == 32.0

    def test_hot_gpu_marks_degraded(self, monkeypatch):
        st = SimpleNamespace(available=True, device_name="RTX", vram_used_gb=4.0,
                             vram_total_gb=32.0, temperature_c=91.0)
        self._patch_gpu(monkeypatch, st)
        assert sec.get_health_status(include_gpu=True).status == "degraded"

    def test_no_gpu_stays_healthy(self, monkeypatch):
        self._patch_gpu(monkeypatch, SimpleNamespace(available=False))
        hs = sec.get_health_status(include_gpu=True)
        assert hs.status == "healthy" and hs.gpu_available is False

    def test_gpu_probe_failure_degrades_instead_of_raising(self, monkeypatch, caplog):
        self._patch_gpu(monkeypatch, exc=RuntimeError("nvml down"))
        with caplog.at_level(logging.ERROR, logger=sec.logger.name):
            hs = sec.get_health_status(include_gpu=True)
        assert hs.status == "degraded"
        assert any("GPU health check failed" in r.getMessage() for r in caplog.records)

    def test_include_gpu_false_never_probes(self, monkeypatch):
        self._patch_gpu(monkeypatch, exc=AssertionError("must not be called"))
        assert sec.get_health_status(include_gpu=False).status == "healthy"

    def test_config_flag_controls_probe_when_override_absent(self, monkeypatch):
        self._patch_gpu(monkeypatch, exc=RuntimeError("probe ran"))
        cfg = sec.SecurityConfig(health_check_include_gpu=True)
        assert sec.get_health_status(cfg).status == "degraded"
        cfg = sec.SecurityConfig(health_check_include_gpu=False)
        assert sec.get_health_status(cfg).status == "healthy"


# =============================================================================
# Auth badge context
# =============================================================================


class TestAuthBadgeContext:
    @pytest.mark.parametrize(
        ("env", "key", "color", "reach"),
        [
            ({}, "no_auth_local", "green", "loopback-only"),
            ({"BACKPROPAGATE_UI_LAUNCH_TOKEN": "t"}, "token_local", "green", "loopback-only"),
            ({"BACKPROPAGATE_UI_AUTH": "u:p"}, "basic_local", "green", "loopback-only"),
            ({"BACKPROPAGATE_UI_AUTH": "u:p", "BACKPROPAGATE_UI_HOST_BIND": "10.0.0.2"},
             "basic_network", "amber", "LAN"),
            ({"BACKPROPAGATE_UI_AUTH": "u:p", "BACKPROPAGATE_UI_SHARE_HOST": "x.example"},
             "basic_shared", "amber", "public network"),
            ({"BACKPROPAGATE_UI_SHARE_HOST": "x.example"}, "insecure", "red", "public network"),
            ({"BACKPROPAGATE_UI_HOST_BIND": "0.0.0.0"}, "insecure", "red",
             "any local interface (LAN)"),
        ],
    )
    def test_posture_matrix(self, env, key, color, reach):
        ctx = sec.get_auth_badge_context(env)
        assert (ctx.mode_key, ctx.mode_color, ctx.reachable_from) == (key, color, reach)

    def test_password_never_appears_in_the_context(self):
        ctx = sec.get_auth_badge_context({"BACKPROPAGATE_UI_AUTH": "alice:topsecret"})
        assert ctx.auth_user == "alice"
        blob = json.dumps(ctx.__dict__)
        assert "topsecret" not in blob

    def test_creds_without_colon_have_empty_user(self):
        assert sec.get_auth_badge_context({"BACKPROPAGATE_UI_AUTH": "nocolon"}).auth_user == ""

    def test_port_override_and_defaults(self):
        assert sec.get_auth_badge_context({}).bind_port == "7860"
        assert sec.get_auth_badge_context({"BACKPROPAGATE_UI_PORT": "9000"}).bind_port == "9000"
        assert "9000" in sec.get_auth_badge_context({"BACKPROPAGATE_UI_PORT": "9000"}).hover_text

    def test_defaults_to_process_env(self, monkeypatch):
        for k in ("BACKPROPAGATE_UI_AUTH", "BACKPROPAGATE_UI_SHARE_HOST",
                  "BACKPROPAGATE_UI_HOST_BIND", "BACKPROPAGATE_UI_LAUNCH_TOKEN"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("BACKPROPAGATE_UI_LAUNCH_TOKEN", "t")
        assert sec.get_auth_badge_context().mode_key == "token_local"

    @pytest.mark.parametrize(
        ("bind", "share", "label"),
        [("", "x", "public network"), ("", "", "loopback-only"), ("localhost", "", "loopback-only"),
         ("::", "", "any local interface (LAN)"), ("192.168.0.4", "", "LAN")],
    )
    def test_classify_reach(self, bind, share, label):
        assert sec._classify_reach(bind, share) == label


# =============================================================================
# Request context, request ids, JSON logging
# =============================================================================


class TestRequestContextAndIds:
    def test_request_id_reuses_inbound_correlation_headers_truncated(self):
        req = SimpleNamespace(headers={"x-request-id": "abcdefghijkl"})
        assert sec.get_request_id(req) == "abcdefgh"
        req2 = SimpleNamespace(headers={"x-correlation-id": "zz"})
        assert sec.get_request_id(req2) == "zz"

    def test_request_id_generated_when_absent(self):
        a, b = sec.get_request_id(), sec.get_request_id(SimpleNamespace(headers={}))
        assert len(a) == 8 and len(b) == 8 and a != b

    def test_context_from_request_and_log_dict(self):
        req = SimpleNamespace(client=SimpleNamespace(host="3.3.3.3"))
        ctx = sec.RequestContext.from_request(req, operation="train")
        d = ctx.to_log_dict()
        assert d["client_ip"] == "3.3.3.3" and d["operation"] == "train" and len(d["request_id"]) == 8


class TestJsonLogging:
    def _record(self, exc_info=None, **extra):
        rec = logging.LogRecord("n", logging.WARNING, __file__, 7, "m %s", ("x",), exc_info)
        for k, v in extra.items():
            setattr(rec, k, v)
        return rec

    def test_formatter_includes_known_extras_and_exception(self):
        try:
            raise ValueError("kaboom")
        except ValueError:
            import sys

            rec = self._record(sys.exc_info(), request_id="r1", operation="op", unknown_field="no")
        out = json.loads(sec.JSONSecurityFormatter().format(rec))
        assert out["message"] == "m x" and out["request_id"] == "r1" and out["operation"] == "op"
        assert "unknown_field" not in out  # only whitelisted fields are emitted
        assert "ValueError: kaboom" in out["exception"]

    def test_configure_json_logging_replaces_handlers(self):
        names = ["cov.sec.a", "cov.sec.b"]
        for n in names:
            logging.getLogger(n).addHandler(logging.NullHandler())
        sec.configure_json_logging(names, level=logging.DEBUG)
        for n in names:
            lg = logging.getLogger(n)
            assert len(lg.handlers) == 1
            assert isinstance(lg.handlers[0].formatter, sec.JSONSecurityFormatter)
            assert lg.level == logging.DEBUG
            lg.handlers.clear()

    def test_configure_json_logging_defaults_to_security_loggers(self):
        before = {n: list(logging.getLogger(n).handlers) for n in
                  ("backpropagate.security.ui", "backpropagate.security.events")}
        try:
            sec.configure_json_logging()
            for n in before:
                assert any(isinstance(h.formatter, sec.JSONSecurityFormatter)
                           for h in logging.getLogger(n).handlers)
        finally:
            for n, handlers in before.items():
                logging.getLogger(n).handlers[:] = handlers


# =============================================================================
# Sessions / concurrency
# =============================================================================


class TestSessionManagerAdversarial:
    def test_singleton(self, fresh_session_manager):
        assert sec.SessionManager() is fresh_session_manager

    def test_per_ip_cap_enforced_and_other_ips_unaffected(self, fresh_session_manager):
        cfg = sec.SecurityConfig(max_sessions_per_ip=2)
        for _ in range(2):
            assert fresh_session_manager.create_session("1.1.1.1", config=cfg)[0] is True
        ok, sid, msg = fresh_session_manager.create_session("1.1.1.1", config=cfg)
        assert ok is False and sid is None and "Maximum 2 sessions per IP" in msg
        assert fresh_session_manager.create_session("2.2.2.2", config=cfg)[0] is True

    def test_validate_unknown_session(self, fresh_session_manager):
        assert fresh_session_manager.validate_session("nope") == (False, "Session not found")

    def test_expired_session_is_removed_on_validate(self, fresh_session_manager):
        cfg = sec.SecurityConfig(session_timeout_minutes=1)
        _ok, sid, _ = fresh_session_manager.create_session("1.1.1.1", config=cfg)
        fresh_session_manager._sessions[sid].last_activity = time.time() - 3600
        assert fresh_session_manager.validate_session(sid, cfg) == (False, "Session expired")
        assert fresh_session_manager.get_active_count() == 0
        assert "1.1.1.1" not in fresh_session_manager._sessions_by_ip

    def test_validate_refreshes_activity(self, fresh_session_manager):
        cfg = sec.SecurityConfig(session_timeout_minutes=1)
        _ok, sid, _ = fresh_session_manager.create_session("1.1.1.1", config=cfg)
        fresh_session_manager._sessions[sid].last_activity = time.time() - 30
        assert fresh_session_manager.validate_session(sid, cfg) == (True, "Session valid")
        assert time.time() - fresh_session_manager._sessions[sid].last_activity < 5

    def test_create_cleans_expired_sessions_freeing_ip_slots(self, fresh_session_manager):
        cfg = sec.SecurityConfig(max_sessions_per_ip=1, session_timeout_minutes=1)
        _ok, sid, _ = fresh_session_manager.create_session("1.1.1.1", config=cfg)
        fresh_session_manager._sessions[sid].last_activity = time.time() - 3600
        assert fresh_session_manager.create_session("1.1.1.1", config=cfg)[0] is True
        assert fresh_session_manager.get_active_count() == 1

    def test_end_session(self, fresh_session_manager):
        _ok, sid, _ = fresh_session_manager.create_session("1.1.1.1")
        assert fresh_session_manager.end_session(sid) is True
        assert fresh_session_manager.end_session(sid) is False
        assert fresh_session_manager.end_session("never-existed") is False

    def test_remove_tolerates_stale_ip_index(self, fresh_session_manager):
        _ok, sid, _ = fresh_session_manager.create_session("1.1.1.1")
        fresh_session_manager._sessions_by_ip["1.1.1.1"].remove(sid)  # index drifted
        assert fresh_session_manager.end_session(sid) is True


class TestConcurrencyLimiterAdversarial:
    def test_acquire_release_cycle_and_totals(self):
        cl = sec.ConcurrencyLimiter(max_concurrent=1, operation_name="training")
        a = SimpleNamespace(client=SimpleNamespace(host="1.1.1.1"))
        b = SimpleNamespace(client=SimpleNamespace(host="2.2.2.2"))
        assert cl.acquire(a) == (True, "Acquired")
        ok, msg = cl.acquire(a)
        assert ok is False and msg == "Maximum 1 concurrent training(s)"
        assert cl.acquire(b)[0] is True
        assert cl.get_active_count(a) == 1 and cl.get_total_active() == 2
        cl.release(a)
        assert cl.get_active_count(a) == 0 and cl.get_total_active() == 1
        cl.release(a)  # double release must not go negative
        assert cl.get_active_count(a) == 0
        assert cl.acquire(a)[0] is True


class TestRateLimitInfoHeaders:
    def test_headers_with_and_without_retry_after(self):
        info = sec.RateLimitInfo(limit=5, remaining=2, reset_timestamp=1700000000.9)
        assert info.to_headers() == {
            "X-RateLimit-Limit": "5", "X-RateLimit-Remaining": "2",
            "X-RateLimit-Reset": "1700000000",
        }
        info = sec.RateLimitInfo(limit=5, remaining=0, reset_timestamp=1.0, retry_after=12.7)
        assert info.to_headers()["Retry-After"] == "12"


# =============================================================================
# Magic bytes
# =============================================================================


class TestValidateFileMagicAdversarial:
    @pytest.mark.parametrize(
        ("content", "ext", "ok", "needle"),
        [
            (b"MZ\x90\x00", ".csv", False, "binary executable"),
            (b"\x7fELF\x02", ".txt", False, "binary executable"),
            (b"PK\x03\x04rest", ".jsonl", False, "binary executable"),
            (b"\xca\xfe\xba\xbe", ".json", False, "binary executable"),
            (b"#!/bin/sh\nrm -rf /", ".csv", False, "HTML/script"),
            (b"<?php system($_GET[0]);", ".txt", False, "HTML/script"),
            (b"<SCRIPT>", ".json", False, "HTML/script"),
            (b"plain,csv,data\n1,2,3\n", ".csv", True, "No signature check"),
            (b'{"a":1}', ".jsonl", True, "Signature valid"),
            (b'"plain text"\n', ".jsonl", True, "Signature valid"),
            (b"[1,2]", ".json", True, "Signature valid"),
            (b"PAR1xxxx", ".parquet", True, "Signature valid"),
            (b"GGUF\x03", ".gguf", True, "Signature valid"),
            (b"not json at all", ".json", False, "does not match expected .json signature"),
            (b"XXXX", ".parquet", False, "does not match expected .parquet signature"),
            (b"", ".json", False, "does not match"),
        ],
    )
    def test_matrix(self, tmp_path, content, ext, ok, needle):
        f = tmp_path / f"sample{ext}"
        f.write_bytes(content)
        got_ok, msg = sec.validate_file_magic(f)
        assert got_ok is ok and needle in msg

    def test_missing_file(self, tmp_path):
        assert sec.validate_file_magic(tmp_path / "gone.json") == (False, "File does not exist")

    def test_expected_extension_overrides_suffix(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"GGUF\x03")
        assert sec.validate_file_magic(f, expected_extension=".gguf") == (True, "Signature valid")

    def test_unreadable_file_is_reported_not_raised(self, tmp_path, monkeypatch):
        """Mocked: ``open`` raises ``PermissionError`` (real FS boundary)."""
        f = tmp_path / "x.json"
        f.write_bytes(b"{}")
        import builtins

        real_open = builtins.open

        def deny(path, *a, **k):
            if str(path) == str(f):
                raise PermissionError("denied")
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", deny)
        ok, msg = sec.validate_file_magic(f)
        assert ok is False and msg.startswith("Failed to read file")


# =============================================================================
# JWT / CSRF primitives
# =============================================================================

jwt = pytest.importorskip("jwt", reason="PyJWT required for JWT tests")
_SECRET = "unit-test-secret-0123456789abcdef0123"


def _jwt_mgr(**cfg):
    return sec.JWTManager(sec.JWTConfig(secret=_SECRET, **cfg))


class TestJWTAdversarial:
    def test_random_secret_generated_when_unset(self, caplog):
        with caplog.at_level(logging.WARNING, logger=sec.logger.name):
            m = sec.JWTManager(sec.JWTConfig())
        assert len(m.config.secret) >= 32
        assert any("JWT secret not configured" in r.getMessage() for r in caplog.records)

    def test_roundtrip_and_claims(self):
        m = _jwt_mgr()
        tok = m.create_token("alice", {"role": "admin"})
        ok, payload, msg = m.verify_token(tok)
        assert ok and msg == "Token valid"
        assert payload["sub"] == "alice" and payload["role"] == "admin" and payload["type"] == "access"

    def test_refresh_token_cannot_be_used_as_access_token(self):
        m = _jwt_mgr()
        ok, payload, msg = m.verify_token(m.create_token("a", is_refresh=True))
        assert (ok, payload) == (False, None) and "expected access" in msg

    def test_access_token_cannot_be_used_as_refresh_token(self):
        m = _jwt_mgr()
        ok, new, msg = m.refresh_access_token(m.create_token("a"))
        assert ok is False and new is None and "expected refresh" in msg

    def test_token_signed_with_other_secret_is_rejected(self):
        other = sec.JWTManager(sec.JWTConfig(secret="different-secret-0123456789abcdef012"))
        ok, payload, msg = _jwt_mgr().verify_token(other.create_token("mallory"))
        assert (ok, payload) == (False, None) and msg.startswith("Invalid token")

    def test_alg_none_token_is_rejected(self):
        import base64

        def b64(d):
            return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

        now = int(time.time())
        forged = (b64({"alg": "none", "typ": "JWT"}) + "." + b64({
            "sub": "root", "iss": "backpropagate", "aud": "backpropagate-ui",
            "iat": now, "exp": now + 999, "type": "access"}) + ".")
        ok, payload, _msg = _jwt_mgr().verify_token(forged)
        assert (ok, payload) == (False, None)

    @pytest.mark.parametrize("junk", ["", "a.b.c", "not-a-jwt", "....", "x" * 500])
    def test_garbage_tokens_are_rejected_not_raised(self, junk):
        assert _jwt_mgr().verify_token(junk)[0] is False

    def test_expired_token(self):
        m = _jwt_mgr(expiry_minutes=-5)
        ok, _p, msg = m.verify_token(m.create_token("a"))
        assert ok is False and msg == "Token expired"

    def test_wrong_audience_and_issuer_rejected(self):
        tok = _jwt_mgr(audience="someone-else").create_token("a")
        assert _jwt_mgr().verify_token(tok)[0] is False
        tok = _jwt_mgr(issuer="evil").create_token("a")
        assert _jwt_mgr().verify_token(tok)[0] is False

    def test_refresh_without_subject_is_rejected(self):
        m = _jwt_mgr()
        tok = m.create_token("", is_refresh=True)  # empty subject
        ok, new, msg = m.refresh_access_token(tok)
        assert ok is False and new is None and "no subject" in msg

    def test_refresh_success_issues_access_token(self):
        m = _jwt_mgr()
        ok, new, _ = m.refresh_access_token(m.create_token("bob", is_refresh=True))
        assert ok and m.verify_token(new)[1]["sub"] == "bob"

    def test_missing_pyjwt_is_reported(self, monkeypatch):
        """Mocked: ``JWT_AVAILABLE`` forced False (PyJWT 'not installed')."""
        monkeypatch.setattr(sec, "JWT_AVAILABLE", False)
        m = _jwt_mgr()
        with pytest.raises(RuntimeError, match="PyJWT not installed"):
            m.create_token("a")
        assert m.verify_token("x") == (False, None, "PyJWT not installed")


class TestCSRFProtectionAdversarial:
    def test_token_is_single_use_by_default(self):
        c = sec.CSRFProtection()
        t = c.generate_token("s1")
        assert c.validate_token("s1", t) == (True, "CSRF token valid")
        assert c.validate_token("s1", t) == (False, "No CSRF token found for session")

    def test_consume_false_allows_reuse(self):
        c = sec.CSRFProtection()
        t = c.generate_token("s1")
        assert c.validate_token("s1", t, consume=False)[0] is True
        assert c.validate_token("s1", t, consume=False)[0] is True

    def test_wrong_token_and_other_sessions_token(self):
        c = sec.CSRFProtection()
        t1, _t2 = c.generate_token("s1"), c.generate_token("s2")
        assert c.validate_token("s1", "wrong") == (False, "Invalid CSRF token")
        assert c.validate_token("s2", t1)[0] is False  # token bound to its session
        assert c.validate_token("s1", t1)[0] is True   # failed guesses don't burn the token

    def test_unknown_and_empty_session_ids(self):
        c = sec.CSRFProtection()
        assert c.validate_token("ghost", "t")[0] is False
        assert c.validate_token("", "t")[0] is False

    def test_expired_token_is_swept_before_validation(self):
        c = sec.CSRFProtection(expiry_minutes=1)
        t = c.generate_token("s1")
        c._tokens["s1"].created_at = time.time() - 3600
        assert c.validate_token("s1", t) == (False, "No CSRF token found for session")

    def test_token_expiring_between_sweep_and_check_is_refused(self, monkeypatch):
        """Mocked: the sweep is disabled to reach the explicit expiry check (the
        window a token can expire in between the sweep and the age comparison)."""
        c = sec.CSRFProtection(expiry_minutes=1)
        t = c.generate_token("s2")
        c._tokens["s2"].created_at = time.time() - 3600
        monkeypatch.setattr(c, "_cleanup_expired", lambda: None)
        assert c.validate_token("s2", t) == (False, "CSRF token expired")
        assert "s2" not in c._tokens  # evicted so it cannot be retried

    def test_token_cap_evicts_oldest(self):
        c = sec.CSRFProtection(max_tokens=3)
        base = time.time() - 100
        for i in range(6):
            c.generate_token(f"s{i}")
            c._tokens[f"s{i}"].created_at = base + i  # recent, strictly ordered
        assert len(c._tokens) <= 4  # cap enforced before each insert
        assert "s5" in c._tokens and "s0" not in c._tokens

    def test_cap_noop_when_under_limit(self):
        c = sec.CSRFProtection(max_tokens=10)
        c.generate_token("a")
        c._enforce_token_cap()
        assert set(c._tokens) == {"a"}


class TestSecureSessionHandlerAdversarial:
    def _handler(self):
        return sec.SecureSessionHandler(sec.JWTConfig(secret=_SECRET))

    def test_login_registers_session_and_logout_forgets_it(self):
        h = self._handler()
        tokens = h.login("alice")
        assert "alice" in h._active_sessions.values()
        h.logout(tokens["access_token"])
        assert "alice" not in h._active_sessions.values()
        h.logout(tokens["access_token"])  # idempotent on unknown session
        h.logout("garbage")

    def test_full_flow_login_validate_refresh_validate(self):
        h = self._handler()
        tokens = h.login("alice")
        assert h.validate_request(tokens["access_token"], tokens["csrf_token"]) == (
            True, "alice", "Request valid")
        ok, new, msg = h.refresh_session(tokens["refresh_token"])
        assert ok and msg == "Session refreshed" and set(new) == {"access_token", "csrf_token"}
        assert h.validate_request(new["access_token"], new["csrf_token"])[1] == "alice"

    def test_csrf_can_be_skipped_but_jwt_cannot(self):
        h = self._handler()
        tokens = h.login("alice")
        assert h.validate_request(tokens["access_token"], "", require_csrf=False)[0] is True
        assert h.validate_request("junk", "", require_csrf=False)[0] is False

    def test_wrong_csrf_token_is_refused(self):
        h = self._handler()
        tokens = h.login("alice")
        valid, user, msg = h.validate_request(tokens["access_token"], "bad")
        assert (valid, user, msg) == (False, None, "Invalid CSRF token")

    def test_request_with_forged_jwt_is_refused(self):
        h = self._handler()
        tokens = h.login("alice")
        forged = sec.JWTManager(sec.JWTConfig(secret="other-secret-0123456789abcdef0123456")).create_token("alice")
        valid, user, msg = h.validate_request(forged, tokens["csrf_token"])
        assert (valid, user) == (False, None) and msg.startswith("Invalid token")

    def test_refresh_token_is_not_accepted_for_requests(self):
        h = self._handler()
        tokens = h.login("alice")
        valid, user, _ = h.validate_request(tokens["refresh_token"], tokens["csrf_token"])
        assert (valid, user) == (False, None)

    def test_refresh_with_access_token_is_refused(self):
        h = self._handler()
        tokens = h.login("alice")
        ok, new, _ = h.refresh_session(tokens["access_token"])
        assert ok is False and new is None

    def test_global_handler_is_a_singleton(self, monkeypatch):
        monkeypatch.setattr(sec, "_secure_session_handler", None)
        a = sec.get_secure_session_handler()
        assert sec.get_secure_session_handler() is a


# =============================================================================
# CSP / security headers
# =============================================================================


class TestCSP:
    def test_default_policy_is_restrictive_where_it_matters(self):
        policy = sec.ContentSecurityPolicy().build_policy()
        assert "default-src 'self'" in policy and "object-src 'none'" in policy
        assert "frame-ancestors 'self'" in policy and "base-uri 'self'" in policy
        assert "form-action 'self'" in policy and "'unsafe-eval'" not in policy

    def test_nonce_lifecycle(self):
        csp = sec.ContentSecurityPolicy()
        assert csp.get_nonce() is None
        assert "nonce-" not in csp.build_policy(include_nonce=True)  # nothing generated yet
        n = csp.generate_nonce()
        assert csp.get_nonce() == n and len(n) >= 22
        assert f"'nonce-{n}'" in csp.build_policy(include_nonce=True)
        assert f"'nonce-{n}'" not in csp.build_policy(include_nonce=False)
        assert csp.generate_nonce() != n  # fresh each time

    def test_report_only_and_report_uri(self):
        cfg = sec.CSPConfig(report_only=True, report_uri="/csp-report")
        name, value = sec.ContentSecurityPolicy(cfg).get_header()
        assert name == "Content-Security-Policy-Report-Only" and "report-uri /csp-report" in value

    def test_enforcing_header_name(self):
        assert sec.ContentSecurityPolicy().get_header()[0] == "Content-Security-Policy"

    def test_empty_directive_lists_are_omitted(self):
        cfg = sec.CSPConfig(default_src=[], script_src=[], style_src=[], img_src=[], font_src=[],
                            connect_src=[], frame_ancestors=[], base_uri=[], form_action=[],
                            object_src=[])
        assert sec.ContentSecurityPolicy(cfg).build_policy() == ""

    def test_all_security_headers(self):
        h = sec.ContentSecurityPolicy().get_all_security_headers()
        assert h["X-Content-Type-Options"] == "nosniff" and h["X-Frame-Options"] == "SAMEORIGIN"
        assert "Referrer-Policy" in h and "Permissions-Policy" in h and "X-XSS-Protection" in h

    def test_reflex_csp_is_a_copy_not_the_shared_constant(self):
        csp = sec.get_reflex_csp()
        csp.config.script_src.append("https://evil.example")
        assert "https://evil.example" not in sec.DEFAULT_REFLEX_CSP.script_src
        assert "https://evil.example" not in sec.get_reflex_csp().build_policy()

    def test_reflex_csp_report_only_flag(self):
        assert sec.get_reflex_csp(report_only=True).get_header()[0].endswith("Report-Only")
        assert sec.get_reflex_csp().get_header()[0] == "Content-Security-Policy"

    def test_apply_headers_to_headers_mapping(self):
        resp = SimpleNamespace(headers={})
        sec.apply_security_headers(resp)
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert "Content-Security-Policy" in resp.headers

    def test_apply_headers_to_set_header_style_response(self):
        calls = {}

        class Resp:
            def set_header(self, k, v):
                calls[k] = v

        sec.apply_security_headers(Resp(), sec.ContentSecurityPolicy())
        assert calls["X-Frame-Options"] == "SAMEORIGIN"

    def test_apply_headers_to_object_with_neither_is_a_noop(self):
        sec.apply_security_headers(object())  # must not raise

    def test_headers_dict_matches_policy_and_is_ascii(self):
        d = sec.security_headers_dict()
        assert all(k.isascii() and v.isascii() for k, v in d.items())
        custom = sec.security_headers_dict(sec.ContentSecurityPolicy(sec.CSPConfig(report_only=True)))
        assert "Content-Security-Policy-Report-Only" in custom


# =============================================================================
# Markdown fence, error sanitising, auth shape
# =============================================================================


class TestSafeMarkdownFence:
    @pytest.mark.parametrize(
        ("content", "fence"),
        [("plain", "```"), ("has ``` inside", "````"), ("a ````` b", "``````"),
         ("`one` and ``two``", "```"), ("", "```"), (None, "```")],
    )
    def test_fence_is_longer_than_any_inner_run(self, content, fence):
        out = sec.safe_markdown_fence(content, "txt")
        assert out.startswith(fence + "txt\n") and out.endswith("\n" + fence)

    def test_injection_cannot_close_the_fence_early(self):
        payload = "```\n# injected heading\n```"
        out = sec.safe_markdown_fence(payload)
        opening = out.split("\n", 1)[0]
        assert len(opening) > 3
        assert out.count(opening) == 2  # only the outer fences match the opener length


class TestSanitizeErrorForUser:
    def test_backprop_error_is_surfaced_with_code_and_redacted_paths(self):
        err = BackpropagateError(
            message="cannot open /home/alice/secret/model.bin and C:\\Users\\bob\\x.gguf",
            code="DEP_THING", suggestion="check \\\\fileserver\\share\\cfg and /tmp/abc",
        )
        msg, hint = sec.sanitize_error_for_user(err, "export")
        assert msg.startswith("DEP_THING: ")
        assert "alice" not in msg and "bob" not in msg and "<redacted-path>" in msg
        assert "fileserver" not in hint and "/tmp/abc" not in hint

    def test_long_messages_are_truncated_with_ellipsis(self):
        err = BackpropagateError(message="m" * 500, code="X", suggestion="s" * 500)
        msg, hint = sec.sanitize_error_for_user(err, "op", max_length=50)
        assert msg.endswith("…") and len(msg) == len("X: ") + 50
        assert hint.endswith("…") and len(hint) == 50

    def test_missing_suggestion_is_none(self):
        assert sec.sanitize_error_for_user(BackpropagateError(message="m", code="C"), "op")[1] is None

    def test_unknown_code_falls_back_to_generic_code(self):
        err = BackpropagateError(message="m")
        err.code = ""
        assert sec.sanitize_error_for_user(err, "op")[0].startswith("BACKPROPAGATE_ERROR: ")

    def test_foreign_exception_is_opaque(self):
        msg, hint = sec.sanitize_error_for_user(
            RuntimeError("secret /home/alice/key and token=abc"), "training")
        assert hint is None
        assert "alice" not in msg and "token" not in msg
        assert msg == ("Internal error during training (RUNTIME_UI). "
                       "Check the server logs for full details.")

    @pytest.mark.parametrize(
        ("raw", "needle_gone"),
        [("/home/alice/x", "alice"), ("/Users/bob/x", "bob"), ("/root/.ssh/id", ".ssh"),
         ("C:\\Users\\carol\\x", "carol"), ("\\\\srv\\share\\f", "srv"), ("/tmp/work/f", "work"),
         ("C:\\Windows\\Temp\\z", "Temp\\z"), ("~/.cache/huggingface/hub/m", "huggingface")],
    )
    def test_path_redaction_patterns(self, raw, needle_gone):
        out = sec._redact_paths(f"failed at {raw} now")
        assert needle_gone not in out and "<redacted-path>" in out

    def test_redact_empty_is_passthrough(self):
        assert sec._redact_paths("") == "" and sec._redact_paths("no paths here") == "no paths here"


class TestValidateAuthShape:
    def _code(self, auth):
        with pytest.raises(BackpropagateError) as exc:
            sec.validate_auth_shape(auth)
        assert exc.value.code == "INPUT_AUTH_INVALID_SHAPE"
        assert "Accepted shapes" in exc.value.suggestion
        return exc.value

    @pytest.mark.parametrize(
        "ok", [None, ("u", "p"), [("u", "p")], [("u", "p"), ("v", "q")], lambda u, p: True],
    )
    def test_accepted_shapes(self, ok):
        sec.validate_auth_shape(ok)

    @pytest.mark.parametrize(
        "bad",
        [("", "p"), ("u", ""), ("u",), ("u", "p", "x"), (1, 2), (None, None), ("u", b"p")],
    )
    def test_bad_tuples(self, bad):
        assert self._code(bad).details["shape"] == "tuple"

    def test_empty_list_would_silently_disable_auth_so_it_is_refused(self):
        err = self._code([])
        assert "silently disable auth" in err.message and err.details["length"] == 0

    def test_list_with_bad_element_reports_index(self):
        err = self._code([("u", "p"), ["u", "p"]])  # second element is a list, not a tuple
        assert err.details["bad_index"] == 1 and "auth[1]" in err.message

    @pytest.mark.parametrize("bad", ["user:pass", 5, {"u": "p"}, b"u:p", {("u", "p")}, 3.5])
    def test_other_types(self, bad):
        assert self._code(bad).details["shape"] == type(bad).__name__

    def test_private_alias_is_the_same_function(self):
        assert sec._validate_auth_shape is sec.validate_auth_shape


def test_module_all_names_exist():
    missing = [n for n in sec.__all__ if not hasattr(sec, n)]
    assert missing == []


def test_env_snapshot_is_restored_between_tests():
    # guard against a leaked override from the fixtures above
    assert os.environ.get("BACKPROPAGATE_UI__OUTPUT_DIR") in (None, "")
