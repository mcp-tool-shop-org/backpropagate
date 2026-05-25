"""
Tests for UI security features.
"""

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestRateLimiter:
    """Tests for RateLimiter class (using EnhancedRateLimiter from ui_security)."""

    def test_rate_limiter_allows_within_limit(self):
        """RateLimiter should allow requests within limit."""
        from backpropagate.ui_security import EnhancedRateLimiter

        limiter = EnhancedRateLimiter(max_requests=3, window_seconds=60)

        # Verify exact count behavior - must allow exactly 3
        result1 = limiter.is_allowed()
        result2 = limiter.is_allowed()
        result3 = limiter.is_allowed()
        assert result1 is True, "First request must be allowed"
        assert result2 is True, "Second request must be allowed"
        assert result3 is True, "Third request must be allowed"

    def test_rate_limiter_blocks_over_limit(self):
        """RateLimiter should block requests over limit."""
        from backpropagate.ui_security import EnhancedRateLimiter

        limiter = EnhancedRateLimiter(max_requests=2, window_seconds=60)

        # First two must succeed
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        # Third must fail - testing boundary
        result = limiter.is_allowed()
        assert result is False, "Request over limit must be blocked"

    def test_rate_limiter_time_until_allowed(self):
        """RateLimiter should report time until allowed."""
        from backpropagate.ui_security import EnhancedRateLimiter

        limiter = EnhancedRateLimiter(max_requests=1, window_seconds=60)

        allowed1, wait1 = limiter.check()  # Use the one allowed request
        assert allowed1 is True, "First check must be allowed"
        assert wait1 == 0.0, "Wait time for allowed request must be 0"

        allowed2, wait_time = limiter.check()  # Get wait time
        assert allowed2 is False, "Second check must be blocked"
        assert wait_time > 0, "Wait time must be positive"
        assert wait_time <= 60, "Wait time must not exceed window"

    def test_rate_limiter_window_expiry(self):
        """RateLimiter should allow requests after window expires."""
        from backpropagate.ui_security import EnhancedRateLimiter

        limiter = EnhancedRateLimiter(max_requests=1, window_seconds=0.1)  # Very short window

        assert limiter.is_allowed() is True, "First request must be allowed"
        assert limiter.is_allowed() is False, "Second immediate request must be blocked"

        time.sleep(0.15)  # Wait for window to expire

        assert limiter.is_allowed() is True, "Request after window expiry must be allowed"

    def test_rate_limiter_exact_boundary(self):
        """Test exact boundary conditions for rate limiting."""
        from backpropagate.ui_security import EnhancedRateLimiter

        # Test with limit of 1 - strictest case
        limiter = EnhancedRateLimiter(max_requests=1, window_seconds=60)
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is False
        assert limiter.is_allowed() is False  # Still blocked


# =============================================================================
# v1.2 cleanup: TestValidatePathInput / TestSanitizeModelName /
# TestSanitizeTextInput / TestGenerateAuthToken / TestLaunchSecurity /
# TestUISecurityExports were removed alongside backpropagate.ui_gradio_legacy
# (the source module they tested). The Reflex UI (canonical from v1.1.0) uses
# ui_app/auth.py + ui_security.py for the equivalent surfaces; coverage there
# lives in tests/test_ui_app_*.py and the EnhancedRateLimiter / FileValidator
# classes below.
# =============================================================================


# =============================================================================
# ENHANCED SECURITY MODULE TESTS (Production-hardened)
# =============================================================================

class TestEnhancedRateLimiter:
    """Tests for EnhancedRateLimiter with IP tracking."""

    def test_allows_initial_requests(self):
        """Should allow requests under the limit."""
        from backpropagate.ui_security import EnhancedRateLimiter

        limiter = EnhancedRateLimiter(max_requests=3, window_seconds=60)
        assert limiter.is_allowed()
        assert limiter.is_allowed()
        assert limiter.is_allowed()

    def test_blocks_after_limit(self):
        """Should block requests after limit is reached."""
        from backpropagate.ui_security import EnhancedRateLimiter

        limiter = EnhancedRateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed()
        assert limiter.is_allowed()
        assert not limiter.is_allowed()

    def test_returns_wait_time(self):
        """Should return wait time when blocked."""
        from backpropagate.ui_security import EnhancedRateLimiter

        limiter = EnhancedRateLimiter(max_requests=1, window_seconds=10)

        allowed, wait = limiter.check()
        assert allowed
        assert wait == 0.0

        allowed, wait = limiter.check()
        assert not allowed
        assert 0 < wait <= 10

    def test_require_raises_exception(self):
        """require() should raise exception when rate limited."""
        from backpropagate.ui_security import EnhancedRateLimiter, RateLimitExceeded

        limiter = EnhancedRateLimiter(max_requests=1, window_seconds=60)
        limiter.require()  # First request OK

        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.require()

        assert "Rate limit exceeded" in str(exc_info.value)

    def test_per_ip_tracking(self):
        """Should track requests per IP."""
        from backpropagate.ui_security import EnhancedRateLimiter

        limiter = EnhancedRateLimiter(max_requests=2, window_seconds=60)

        # Create mock requests with different IPs
        request1 = MagicMock()
        request1.client = {"host": "192.168.1.1"}

        request2 = MagicMock()
        request2.client = {"host": "192.168.1.2"}

        # Both IPs should have their own limits
        assert limiter.is_allowed(request1)
        assert limiter.is_allowed(request1)
        assert not limiter.is_allowed(request1)  # IP1 exhausted

        # IP2 should still have requests available
        assert limiter.is_allowed(request2)
        assert limiter.is_allowed(request2)

    def test_burst_allowance(self):
        """Should respect burst allowance."""
        from backpropagate.ui_security import EnhancedRateLimiter

        limiter = EnhancedRateLimiter(
            max_requests=2,
            window_seconds=60,
            burst_allowance=1,
        )

        # Should allow max_requests + burst_allowance
        assert limiter.is_allowed()
        assert limiter.is_allowed()
        assert limiter.is_allowed()  # Burst
        assert not limiter.is_allowed()

    def test_rate_limiter_buckets_per_ip_not_per_port(self):
        """Regression: same IP, different source ports must share a rate-limit bucket.

        Previously ``_get_client_id`` used ``str(request.client)`` which
        included the source port, giving every TCP connection its own
        bucket — effectively bypassing rate limiting. The new
        ``_extract_client_ip`` reads ``.host`` only.
        """
        from collections import namedtuple

        from backpropagate.ui_security import EnhancedRateLimiter

        Address = namedtuple("Address", ["host", "port"])
        limiter = EnhancedRateLimiter(max_requests=1, window_seconds=60)

        request1 = MagicMock()
        request1.client = Address(host="1.2.3.4", port=54321)
        assert limiter.is_allowed(request1)

        request2 = MagicMock()
        request2.client = Address(host="1.2.3.4", port=54322)  # same IP, different port
        assert not limiter.is_allowed(request2), \
            "Same IP must share bucket regardless of source port"


class TestSecurityConfig:
    """Tests for SecurityConfig dataclass."""

    def test_default_config_exists(self):
        """DEFAULT_SECURITY_CONFIG should be available."""
        from backpropagate.ui_security import DEFAULT_SECURITY_CONFIG, SecurityConfig

        assert DEFAULT_SECURITY_CONFIG is not None
        assert isinstance(DEFAULT_SECURITY_CONFIG, SecurityConfig)

    def test_default_rate_limits(self):
        """Default rate limits should be reasonable."""
        from backpropagate.ui_security import SecurityConfig

        config = SecurityConfig()
        # Assert exact values to catch mutations
        assert config.training_rate_limit == 3, "training_rate_limit must be 3"
        assert config.training_rate_window == 60, "training_rate_window must be 60"
        assert config.export_rate_limit == 5, "export_rate_limit must be 5"
        assert config.upload_rate_limit == 10, "upload_rate_limit must be 10"
        assert config.export_rate_window == 60, "export_rate_window must be 60"
        assert config.upload_rate_window == 60, "upload_rate_window must be 60"

    def test_blocked_extensions_include_dangerous(self):
        """Blocked extensions should include common dangerous types."""
        from backpropagate.ui_security import SecurityConfig

        config = SecurityConfig()
        dangerous = {".exe", ".bat", ".sh", ".py", ".html", ".svg"}
        assert dangerous.issubset(config.blocked_extensions)
        # Also verify specific dangerous extensions
        assert ".htm" in config.blocked_extensions
        assert ".js" in config.blocked_extensions
        assert ".php" in config.blocked_extensions

    def test_allowed_dataset_extensions(self):
        """Allowed dataset extensions should include common formats."""
        from backpropagate.ui_security import SecurityConfig

        config = SecurityConfig()
        assert ".jsonl" in config.allowed_dataset_extensions
        assert ".json" in config.allowed_dataset_extensions
        assert ".csv" in config.allowed_dataset_extensions
        assert ".txt" in config.allowed_dataset_extensions
        assert ".parquet" in config.allowed_dataset_extensions
        # Ensure dangerous extensions are NOT allowed
        assert ".exe" not in config.allowed_dataset_extensions
        assert ".html" not in config.allowed_dataset_extensions

    def test_config_field_types(self):
        """Config fields should have correct types."""
        from backpropagate.ui_security import SecurityConfig

        config = SecurityConfig()
        assert isinstance(config.training_rate_limit, int)
        assert isinstance(config.csrf_enabled, bool)
        assert isinstance(config.blocked_extensions, set)
        assert isinstance(config.max_upload_size_mb, int)
        assert config.max_upload_size_mb == 500


class TestFileValidator:
    """Tests for FileValidator (CVE-2024-47872 mitigation)."""

    def test_rejects_none_file(self):
        """Should reject None file."""
        from backpropagate.ui_security import FileValidator

        validator = FileValidator()
        valid, error, path = validator.validate(None)

        assert valid is False, "None file must be rejected"
        assert "No file" in error, "Error must mention 'No file'"
        assert path is None, "Path must be None for rejected file"

    def test_rejects_dangerous_extensions(self):
        """Should reject dangerous file extensions (CVE-2024-47872)."""
        from backpropagate.ui_security import FileValidator

        validator = FileValidator()

        # Create proper mocks with name attribute set correctly
        dangerous_names = ["test.html", "test.svg", "test.js", "test.exe", "test.py"]

        for fname in dangerous_names:
            file_obj = MagicMock()
            file_obj.name = fname  # Set name attribute properly

            valid, error, path = validator.validate(file_obj)
            assert valid is False, f"Should reject {fname}"
            assert "not allowed" in error.lower() or "not supported" in error.lower()
            assert path is None, f"Path must be None for rejected file {fname}"

    def test_accepts_safe_extensions(self):
        """Should accept safe dataset extensions."""
        from backpropagate.ui_security import FileValidator

        validator = FileValidator()

        # Create mock file with .jsonl extension
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'stat') as mock_stat:
                mock_stat.return_value.st_size = 1024

                file_obj = MagicMock()
                file_obj.name = "/tmp/test.jsonl"

                valid, error, path = validator.validate(file_obj)
                assert valid is True, "Valid file must be accepted"
                assert error == "", "Error must be empty for valid file"
                assert path is not None, "Path must be returned for valid file"

    def test_rejects_file_too_large(self):
        """Should reject files that exceed size limit.

        Verifies the size-check upload boundary (DoS-relevant). A file whose
        size exceeds FileValidator.max_size_bytes must be rejected with
        (valid=False, size-related error, path=None).

        Note: FileValidator.__init__ wires self.max_size_bytes from the
        `max_size_mb` constructor argument (NOT from
        config.max_upload_size_mb). The test passes max_size_mb=1 directly
        so the 2MB fixture exceeds the limit.
        """
        from backpropagate.ui_security import FileValidator, SecurityConfig

        config = SecurityConfig(max_upload_size_mb=1)  # config retained for parity
        validator = FileValidator(config=config, max_size_mb=1)  # 1MB limit

        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'stat') as mock_stat:
                # 2MB file - exceeds 1MB limit
                mock_stat.return_value.st_size = 2 * 1024 * 1024

                file_obj = MagicMock()
                file_obj.name = "/tmp/large.jsonl"

                valid, error, path = validator.validate(file_obj)

                assert valid is False, (
                    f"Expected rejection of oversize file, got valid=True "
                    f"(error={error!r}, path={path!r})"
                )
                err_lower = (error or "").lower()
                assert error and any(
                    needle in err_lower
                    for needle in ("size", "large", "limit", "maximum")
                ), f"Expected size-related error (size/large/limit/maximum), got: {error!r}"
                assert path is None, f"Expected None path for rejected file, got: {path!r}"


class TestDangerousExtensions:
    """Tests for dangerous file extension constants."""

    def test_html_blocked(self):
        """HTML files should be blocked (XSS vector)."""
        from backpropagate.ui_security import DANGEROUS_EXTENSIONS

        assert ".html" in DANGEROUS_EXTENSIONS
        assert ".htm" in DANGEROUS_EXTENSIONS
        assert ".xhtml" in DANGEROUS_EXTENSIONS

    def test_svg_blocked(self):
        """SVG files should be blocked (can contain JavaScript)."""
        from backpropagate.ui_security import DANGEROUS_EXTENSIONS

        assert ".svg" in DANGEROUS_EXTENSIONS
        assert ".svgz" in DANGEROUS_EXTENSIONS

    def test_scripts_blocked(self):
        """Script files should be blocked."""
        from backpropagate.ui_security import DANGEROUS_EXTENSIONS

        script_exts = {".js", ".py", ".sh", ".bat", ".ps1", ".php", ".rb"}
        assert script_exts.issubset(DANGEROUS_EXTENSIONS)


class TestSanitizeFilenameEnhanced:
    """Tests for enhanced filename sanitization."""

    def test_removes_path_separators(self):
        """Should remove path separators to prevent traversal."""
        from backpropagate.ui_security import sanitize_filename

        result = sanitize_filename("../../../etc/passwd")
        assert "/" not in result, "Forward slashes must be removed"
        assert "\\" not in result, "Backslashes must be removed"
        # Note: The implementation replaces / with _ so .. becomes _..
        # The key is path traversal is neutralized

    def test_removes_null_bytes(self):
        """Should remove null bytes (security vulnerability)."""
        from backpropagate.ui_security import sanitize_filename

        result = sanitize_filename("test\x00.txt")
        assert "\x00" not in result, "Null bytes must be removed"
        assert "test" in result, "Non-null content must be preserved"

    def test_limits_length(self):
        """Should limit filename length."""
        from backpropagate.ui_security import sanitize_filename

        long_name = "a" * 500 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 255, "Filename must be truncated to 255 chars"
        assert len(result) > 0, "Result must not be empty"

    def test_preserves_extension_on_truncate(self):
        """Should preserve extension when truncating."""
        from backpropagate.ui_security import sanitize_filename

        long_name = "a" * 500 + ".jsonl"
        result = sanitize_filename(long_name)
        assert result.endswith(".jsonl"), "Extension must be preserved"
        assert len(result) <= 255, "Must still respect length limit"

    def test_sanitize_empty_filename(self):
        """Should handle empty filename."""
        from backpropagate.ui_security import sanitize_filename

        result = sanitize_filename("")
        assert result == "" or result is not None  # Empty or default

    def test_sanitize_normal_filename(self):
        """Should preserve normal safe filenames."""
        from backpropagate.ui_security import sanitize_filename

        result = sanitize_filename("my_dataset.jsonl")
        assert result == "my_dataset.jsonl", "Safe filename must be unchanged"

    def test_handles_special_characters(self):
        """Should handle special characters in some way."""
        from backpropagate.ui_security import sanitize_filename

        # The implementation may replace, remove, or escape special chars
        # Main requirement: result should be a valid filename
        result = sanitize_filename("file<>:\"|?*.txt")
        # Result should be non-empty and contain the extension
        assert len(result) > 0, "Result must not be empty"
        assert ".txt" in result, "Extension should be preserved"


@pytest.mark.serial
class TestSecurityLogger:
    """Tests for SecurityLogger.

    TESTS-A-007 (v1.4 Wave 2 amend): marked @serial because the class
    mutates the process-wide ``SecurityLogger._instance`` singleton
    via the autouse fixture. The fixture clears the singleton within
    the class, but xdist could schedule a TestSecurityLogger test
    adjacent to an unrelated test that ALSO imports SecurityLogger
    (e.g. for a one-off integration assertion) and the unrelated test
    would see a None instance mid-run. Serial scheduling avoids that
    cross-class race.
    """

    @pytest.fixture(autouse=True)
    def _reset_security_logger(self):
        """Reset SecurityLogger singleton before AND after each test so
        previous-test instances cannot leak into ordering-sensitive runs.
        """
        from backpropagate.ui_security import SecurityLogger
        SecurityLogger._instance = None
        yield
        SecurityLogger._instance = None

    def test_singleton_pattern(self):
        """SecurityLogger should be a singleton."""
        from backpropagate.ui_security import SecurityLogger

        logger1 = SecurityLogger()
        logger2 = SecurityLogger()
        assert logger1 is logger2

    def test_log_function_exists(self):
        """log_security_event function should exist and be callable."""
        from backpropagate.ui_security import log_security_event

        # Should not raise
        log_security_event("test_event", test_param="value")


class TestRateLimitExceeded:
    """Tests for RateLimitExceeded exception."""

    def test_contains_wait_time(self):
        """Exception should contain wait time."""
        from backpropagate.ui_security import RateLimitExceeded

        exc = RateLimitExceeded(30.0, "training")
        assert exc.wait_seconds == 30.0, "wait_seconds must be exactly 30.0"
        assert exc.operation == "training", "operation must be 'training'"

    def test_message_includes_wait_time(self):
        """Exception message should include wait time."""
        from backpropagate.ui_security import RateLimitExceeded

        exc = RateLimitExceeded(45.0, "export")
        msg = str(exc)
        assert "45" in msg, "Message must include wait time"
        assert "export" in msg, "Message must include operation"

    def test_exception_is_exception(self):
        """RateLimitExceeded should be a proper Exception."""
        from backpropagate.ui_security import RateLimitExceeded

        exc = RateLimitExceeded(10.0, "test")
        assert isinstance(exc, Exception)
        # Verify it can be raised and caught
        with pytest.raises(RateLimitExceeded):
            raise exc

    def test_different_operations(self):
        """Should handle different operation names."""
        from backpropagate.ui_security import RateLimitExceeded

        for op in ["training", "export", "upload", "custom_op"]:
            exc = RateLimitExceeded(1.0, op)
            assert exc.operation == op
            assert op in str(exc)


# =============================================================================
# NEW FEATURE TESTS (Production-hardened additions)
# =============================================================================

class TestHealthCheck:
    """Tests for health check functionality."""

    def test_get_health_status_returns_healthy(self):
        """Health status should return healthy by default."""
        from backpropagate.ui_security import HealthStatus, get_health_status

        status = get_health_status(include_gpu=False)

        assert isinstance(status, HealthStatus), "Must return HealthStatus instance"
        assert status.status == "healthy", "Default status must be 'healthy'"
        assert status.uptime_seconds > 0, "Uptime must be positive"
        assert status.uptime_seconds < 86400 * 365, "Uptime must be reasonable"

    def test_health_status_to_dict(self):
        """Health status should convert to dict for JSON."""
        from backpropagate.ui_security import HealthStatus

        status = HealthStatus(
            status="healthy",
            version="1.0.0",
            uptime_seconds=100.0,
        )

        result = status.to_dict()

        assert result["status"] == "healthy", "status must match"
        assert result["version"] == "1.0.0", "version must match"
        assert result["uptime_seconds"] == 100.0, "uptime_seconds must match"
        assert "timestamp" in result, "timestamp must be present"
        assert isinstance(result["timestamp"], str), "timestamp must be string"

    def test_health_status_includes_gpu_when_available(self):
        """Health status should include GPU info when requested."""
        from backpropagate.ui_security import get_health_status

        status = get_health_status(include_gpu=True)

        # GPU may or may not be available, but check structure
        result = status.to_dict()
        assert "status" in result
        assert status.status in ("healthy", "degraded", "unhealthy")

    def test_health_status_with_gpu_info(self):
        """Health status should include GPU dict when GPU available."""
        from backpropagate.ui_security import HealthStatus

        status = HealthStatus(
            status="healthy",
            version="1.0.0",
            uptime_seconds=100.0,
            gpu_available=True,
            gpu_name="RTX 5080",
            gpu_memory_used_gb=4.0,
            gpu_memory_total_gb=16.0,
        )

        result = status.to_dict()
        assert "gpu" in result, "GPU info must be included when available"
        assert result["gpu"]["available"] is True
        assert result["gpu"]["name"] == "RTX 5080"
        assert result["gpu"]["memory_used_gb"] == 4.0
        assert result["gpu"]["memory_total_gb"] == 16.0

    def test_health_status_without_gpu(self):
        """Health status should not include GPU dict when unavailable."""
        from backpropagate.ui_security import HealthStatus

        status = HealthStatus(
            status="healthy",
            version="1.0.0",
            uptime_seconds=100.0,
            gpu_available=False,
        )

        result = status.to_dict()
        assert "gpu" not in result, "GPU info must not be included when unavailable"


class TestRequestContext:
    """Tests for request context and tracing."""

    def test_request_context_creation(self):
        """RequestContext should be creatable from Gradio request.

        Default-IP sentinel changed from "anonymous" to "unknown" so the
        per-IP rate limiter fails CLOSED to a single shared bucket on
        unknown request shapes instead of appearing per-client.
        """
        from backpropagate.ui_security import RequestContext

        ctx = RequestContext.from_gradio_request(operation="test_op")

        assert len(ctx.request_id) == 8, "request_id must be 8 chars"
        assert ctx.client_ip == "unknown", "Default IP must be 'unknown'"
        assert ctx.operation == "test_op", "operation must match"
        assert ctx.timestamp > 0, "timestamp must be positive"
        assert ctx.user_id is None, "user_id must default to None"

    def test_request_context_with_mock_request(self):
        """RequestContext should extract IP from request."""
        from backpropagate.ui_security import RequestContext

        request = MagicMock()
        request.client = {"host": "192.168.1.100"}

        ctx = RequestContext.from_gradio_request(request, operation="train")

        assert ctx.client_ip == "192.168.1.100", "IP must be extracted from request"
        assert ctx.operation == "train", "operation must match"

    def test_get_request_id_generates_unique_ids(self):
        """get_request_id should generate unique IDs."""
        from backpropagate.ui_security import get_request_id

        ids = [get_request_id() for _ in range(10)]

        assert len(set(ids)) == 10, "All IDs must be unique"
        assert all(len(id_) == 8 for id_ in ids), "All IDs must be 8 chars"
        # Verify they're alphanumeric (hex-like)
        for id_ in ids:
            assert id_.replace("-", "").isalnum(), f"ID {id_} must be alphanumeric"

    def test_request_context_to_log_dict(self):
        """RequestContext should convert to dict for logging."""
        from backpropagate.ui_security import RequestContext

        ctx = RequestContext(
            request_id="abc12345",
            client_ip="127.0.0.1",
            timestamp=1000.0,
            operation="export",
        )

        log_dict = ctx.to_log_dict()

        assert log_dict["request_id"] == "abc12345", "request_id must match"
        assert log_dict["client_ip"] == "127.0.0.1", "client_ip must match"
        assert log_dict["operation"] == "export", "operation must match"
        assert log_dict["timestamp"] == 1000.0, "timestamp must match"
        assert "user_id" in log_dict, "user_id key must be present"

    def test_request_context_with_user_id(self):
        """RequestContext should support user_id."""
        from backpropagate.ui_security import RequestContext

        ctx = RequestContext(
            request_id="test1234",
            client_ip="10.0.0.1",
            timestamp=500.0,
            operation="upload",
            user_id="user123",
        )

        assert ctx.user_id == "user123"
        log_dict = ctx.to_log_dict()
        assert log_dict["user_id"] == "user123"


class TestEnvConfigOverride:
    """Tests for environment variable configuration.

    TESTS-B-003 (Stage C amend wave): converted from the raw
    ``os.environ[X] = ...; del os.environ[X]`` pattern to
    ``monkeypatch.setenv`` so cleanup happens on assertion failure too.
    The leak surface here is unusually load-bearing because
    ``BACKPROPAGATE_SECURITY__*`` env vars are read by ``SecurityConfig()``
    at construction time — a leaked value silently mutates every
    downstream test's SecurityConfig defaults.
    """

    def test_load_config_from_env_int_values(self, monkeypatch):
        """Should load integer values from environment."""
        from backpropagate.ui_security import SecurityConfig, load_config_from_env

        monkeypatch.setenv("BACKPROPAGATE_SECURITY__TRAINING_RATE_LIMIT", "10")

        config = load_config_from_env(SecurityConfig())
        assert config.training_rate_limit == 10

    def test_load_config_from_env_bool_values(self, monkeypatch):
        """Should load boolean values from environment."""
        from backpropagate.ui_security import SecurityConfig, load_config_from_env

        monkeypatch.setenv("BACKPROPAGATE_SECURITY__LOG_FORMAT_JSON", "true")

        config = load_config_from_env(SecurityConfig())
        assert config.log_format_json is True

    def test_load_config_ignores_invalid_env_vars(self, monkeypatch):
        """Should ignore environment variables with invalid values."""
        from backpropagate.ui_security import SecurityConfig, load_config_from_env

        monkeypatch.setenv(
            "BACKPROPAGATE_SECURITY__TRAINING_RATE_LIMIT", "not_a_number"
        )

        config = load_config_from_env(SecurityConfig())
        # Should use default, not crash
        assert config.training_rate_limit == 3


@pytest.mark.serial
class TestSessionManager:
    """Tests for session management.

    TESTS-A-007 (v1.4 Wave 2 amend): marked @serial — same shape as
    TestSecurityLogger above (singleton ``SessionManager._instance``
    is mutated via the autouse fixture; serial scheduling protects
    cross-class adjacency under xdist).
    """

    @pytest.fixture(autouse=True)
    def _reset_session_manager(self):
        """Reset SessionManager singleton before AND after each test.

        Previously each test did ``SessionManager._instance = None`` inline
        without a post-reset, so the post-test instance leaked into whichever
        test ran next (a real risk under ``--lf`` / random ordering / xdist).
        """
        from backpropagate.ui_security import SessionManager
        SessionManager._instance = None
        yield
        SessionManager._instance = None

    def test_create_session(self):
        """Should create a session successfully."""
        from backpropagate.ui_security import SecurityConfig, SessionManager

        manager = SessionManager()
        config = SecurityConfig(max_sessions_per_ip=5)

        success, session_id, msg = manager.create_session("192.168.1.1", config=config)

        assert success is True, "Session creation must succeed"
        assert session_id is not None, "session_id must not be None"
        assert len(session_id) == 36, "session_id must be UUID format (36 chars)"
        assert "-" in session_id, "session_id must contain UUID dashes"
        assert msg == "Session created", "Success message must match"

        # Cleanup
        manager.end_session(session_id)

    def test_session_limit_per_ip(self):
        """Should enforce session limit per IP."""
        from backpropagate.ui_security import SecurityConfig, SessionManager

        manager = SessionManager()
        config = SecurityConfig(max_sessions_per_ip=2)

        # Create 2 sessions (should succeed)
        success1, sid1, msg1 = manager.create_session("10.0.0.1", config=config)
        success2, sid2, msg2 = manager.create_session("10.0.0.1", config=config)

        assert success1 is True, "First session must succeed"
        assert success2 is True, "Second session must succeed"
        assert sid1 != sid2, "Session IDs must be unique"

        # Third should fail
        success3, sid3, msg3 = manager.create_session("10.0.0.1", config=config)
        assert success3 is False, "Third session must fail"
        assert sid3 is None, "Failed session must return None ID"
        assert "Maximum" in msg3, "Error must mention maximum"
        assert "2" in msg3, "Error must mention the limit"

        # Cleanup
        manager.end_session(sid1)
        manager.end_session(sid2)

    def test_session_validation_and_timeout(self):
        """Should validate and timeout sessions."""
        from backpropagate.ui_security import SecurityConfig, SessionManager

        manager = SessionManager()
        config = SecurityConfig(session_timeout_minutes=1)

        success, session_id, _ = manager.create_session("127.0.0.1", config=config)
        assert success is True

        # Should validate successfully
        valid, msg = manager.validate_session(session_id, config=config)
        assert valid is True, "Valid session must pass validation"
        assert "valid" in msg.lower(), "Success message must mention valid"

        # Invalid session should fail
        valid2, msg2 = manager.validate_session("nonexistent", config=config)
        assert valid2 is False, "Nonexistent session must fail"
        assert "not found" in msg2, "Error must mention not found"

        # Cleanup
        manager.end_session(session_id)

    def test_end_session(self):
        """Should properly end sessions."""
        from backpropagate.ui_security import SecurityConfig, SessionManager

        manager = SessionManager()
        config = SecurityConfig(max_sessions_per_ip=5)

        _, session_id, _ = manager.create_session("1.2.3.4", config=config)

        # End the session
        result = manager.end_session(session_id)
        assert result is True, "Ending existing session must return True"

        # Try to end again - should return False
        result2 = manager.end_session(session_id)
        assert result2 is False, "Ending nonexistent session must return False"

        # Validate should fail now
        valid, _ = manager.validate_session(session_id, config=config)
        assert valid is False, "Ended session must not validate"

    def test_get_active_count(self):
        """Should track active session count."""
        from backpropagate.ui_security import SecurityConfig, SessionManager

        manager = SessionManager()
        config = SecurityConfig(max_sessions_per_ip=10)

        initial_count = manager.get_active_count()

        _, sid1, _ = manager.create_session("5.5.5.5", config=config)
        assert manager.get_active_count() == initial_count + 1

        _, sid2, _ = manager.create_session("5.5.5.6", config=config)
        assert manager.get_active_count() == initial_count + 2

        manager.end_session(sid1)
        assert manager.get_active_count() == initial_count + 1

        manager.end_session(sid2)


class TestConcurrencyLimiter:
    """Tests for concurrency limiting."""

    def test_acquire_and_release(self):
        """Should acquire and release concurrency slots."""
        from backpropagate.ui_security import ConcurrencyLimiter

        limiter = ConcurrencyLimiter(max_concurrent=2, operation_name="training")

        # Initial state
        assert limiter.get_total_active() == 0, "Initial count must be 0"

        # Acquire first slot
        success1, msg1 = limiter.acquire()
        assert success1 is True, "First acquire must succeed"
        assert limiter.get_total_active() == 1, "Count must be 1 after first acquire"
        assert msg1 == "Acquired", "Success message must be 'Acquired'"

        # Acquire second slot
        success2, msg2 = limiter.acquire()
        assert success2 is True, "Second acquire must succeed"
        assert limiter.get_total_active() == 2, "Count must be 2 after second acquire"

        # Third should fail - at max
        success3, msg3 = limiter.acquire()
        assert success3 is False, "Third acquire must fail"
        assert "Maximum" in msg3, "Error must mention Maximum"
        assert "2" in msg3, "Error must mention the limit"
        assert "training" in msg3, "Error must mention operation name"
        assert limiter.get_total_active() == 2, "Count must stay at 2"

        # Release one
        limiter.release()
        assert limiter.get_total_active() == 1, "Count must be 1 after release"

        # Now can acquire again
        success4, _ = limiter.acquire()
        assert success4 is True, "Acquire after release must succeed"
        assert limiter.get_total_active() == 2

    def test_concurrency_per_ip(self):
        """Should track concurrency per IP."""
        from backpropagate.ui_security import ConcurrencyLimiter

        limiter = ConcurrencyLimiter(max_concurrent=1, operation_name="export")

        request1 = MagicMock()
        request1.client = {"host": "1.1.1.1"}

        request2 = MagicMock()
        request2.client = {"host": "2.2.2.2"}

        # IP1 acquires
        success1, _ = limiter.acquire(request1)
        assert success1 is True, "IP1 first acquire must succeed"
        assert limiter.get_active_count(request1) == 1

        # IP1 can't acquire again - at per-IP limit
        success2, msg2 = limiter.acquire(request1)
        assert success2 is False, "IP1 second acquire must fail"
        assert "Maximum" in msg2

        # IP2 can still acquire - separate limit
        success3, _ = limiter.acquire(request2)
        assert success3 is True, "IP2 acquire must succeed"
        assert limiter.get_active_count(request2) == 1

        # Total should be 2
        assert limiter.get_total_active() == 2, "Total must be 2"

    def test_release_reduces_count(self):
        """Release should properly decrement counter."""
        from backpropagate.ui_security import ConcurrencyLimiter

        limiter = ConcurrencyLimiter(max_concurrent=3, operation_name="test")

        # Acquire 3
        limiter.acquire()
        limiter.acquire()
        limiter.acquire()
        assert limiter.get_total_active() == 3

        # Release all
        limiter.release()
        assert limiter.get_total_active() == 2
        limiter.release()
        assert limiter.get_total_active() == 1
        limiter.release()
        assert limiter.get_total_active() == 0

        # Extra release shouldn't go negative
        limiter.release()
        assert limiter.get_total_active() == 0


class TestRateLimitInfo:
    """Tests for rate limit response headers."""

    def test_to_headers(self):
        """Should generate proper rate limit headers."""
        from backpropagate.ui_security import RateLimitInfo

        info = RateLimitInfo(
            limit=100,
            remaining=95,
            reset_timestamp=1700000000.0,
            retry_after=30.0,
        )

        headers = info.to_headers()

        # Check exact header names (RFC 6585 compliant)
        assert headers["X-RateLimit-Limit"] == "100", "Limit header must be '100'"
        assert headers["X-RateLimit-Remaining"] == "95", "Remaining header must be '95'"
        assert headers["X-RateLimit-Reset"] == "1700000000", "Reset must be integer string"
        assert headers["Retry-After"] == "30", "Retry-After must be '30'"
        # Verify all values are strings
        for key, value in headers.items():
            assert isinstance(value, str), f"Header {key} must be string"

    def test_to_headers_without_retry_after(self):
        """Should omit Retry-After when not set."""
        from backpropagate.ui_security import RateLimitInfo

        info = RateLimitInfo(
            limit=50,
            remaining=49,
            reset_timestamp=1700000000.0,
        )

        headers = info.to_headers()

        assert "X-RateLimit-Limit" in headers
        assert headers["X-RateLimit-Limit"] == "50"
        assert headers["X-RateLimit-Remaining"] == "49"
        assert "Retry-After" not in headers, "Retry-After must be omitted when None"

    def test_to_headers_with_zero_remaining(self):
        """Should handle zero remaining correctly."""
        from backpropagate.ui_security import RateLimitInfo

        info = RateLimitInfo(
            limit=10,
            remaining=0,
            reset_timestamp=1700000000.0,
            retry_after=60.0,
        )

        headers = info.to_headers()

        assert headers["X-RateLimit-Remaining"] == "0", "Zero remaining must be '0'"
        assert headers["Retry-After"] == "60"


class TestFileMagicValidation:
    """Tests for file magic byte validation."""

    def test_validate_json_file(self, tmp_path):
        """Should validate JSON file by magic bytes."""
        from backpropagate.ui_security import validate_file_magic

        # Create a valid JSON file starting with {
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value"}')

        valid, msg = validate_file_magic(json_file)
        assert valid is True, "Valid JSON file must pass"
        assert "valid" in msg.lower() or "passed" in msg.lower(), "Message must indicate success"

    def test_validate_json_array(self, tmp_path):
        """Should validate JSON array file (starts with [)."""
        from backpropagate.ui_security import validate_file_magic

        json_file = tmp_path / "array.json"
        json_file.write_text('[1, 2, 3]')

        valid, msg = validate_file_magic(json_file)
        assert valid is True, "JSON array must pass"

    def test_validate_jsonl_file(self, tmp_path):
        """Should validate JSONL file by magic bytes."""
        from backpropagate.ui_security import validate_file_magic

        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"line": 1}\n{"line": 2}')

        valid, msg = validate_file_magic(jsonl_file)
        assert valid is True, "Valid JSONL file must pass"

    def test_reject_html_masquerading_as_json(self, tmp_path):
        """Should reject HTML file with .json extension."""
        from backpropagate.ui_security import validate_file_magic

        fake_json = tmp_path / "malicious.json"
        fake_json.write_text('<!DOCTYPE html><html><script>alert("xss")</script></html>')

        valid, msg = validate_file_magic(fake_json)
        assert valid is False, "HTML masquerading as JSON must be rejected"
        assert "HTML/script" in msg, "Error must mention HTML/script"

    def test_reject_php_masquerading_as_json(self, tmp_path):
        """Should reject PHP file with .json extension."""
        from backpropagate.ui_security import validate_file_magic

        fake_json = tmp_path / "malicious.json"
        fake_json.write_text('<?php echo "pwned"; ?>')

        valid, msg = validate_file_magic(fake_json)
        assert valid is False, "PHP masquerading as JSON must be rejected"

    def test_reject_shell_script_masquerading_as_json(self, tmp_path):
        """Should reject shell script with .json extension."""
        from backpropagate.ui_security import validate_file_magic

        fake_json = tmp_path / "malicious.json"
        fake_json.write_text('#!/bin/bash\nrm -rf /')

        valid, msg = validate_file_magic(fake_json)
        assert valid is False, "Shell script masquerading as JSON must be rejected"

    def test_no_signature_check_for_csv(self, tmp_path):
        """Should pass CSV files (no standard signature)."""
        from backpropagate.ui_security import validate_file_magic

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\nval1,val2")

        valid, msg = validate_file_magic(csv_file)
        assert valid is True, "CSV must pass (no signature)"
        assert "No signature check" in msg, "Message must indicate no check available"

    def test_nonexistent_file(self, tmp_path):
        """Should reject nonexistent files."""
        from backpropagate.ui_security import validate_file_magic

        fake_path = tmp_path / "nonexistent.json"

        valid, msg = validate_file_magic(fake_path)
        assert valid is False, "Nonexistent file must fail"
        assert "not exist" in msg.lower() or "does not exist" in msg.lower()


class TestJSONSecurityFormatter:
    """Tests for JSON log formatting."""

    def test_formats_as_json(self):
        """Should format log records as JSON."""
        import json

        from backpropagate.ui_security import JSONSecurityFormatter

        formatter = JSONSecurityFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["message"] == "Test message", "message must match"
        assert parsed["level"] == "INFO", "level must be INFO"
        assert parsed["logger"] == "test.logger", "logger must match"
        assert "timestamp" in parsed, "timestamp must be present"
        # Line number may be stored as 'line' or 'lineno'
        assert parsed.get("line") == 10 or parsed.get("lineno") == 10, "line number must match"

    def test_includes_extra_fields(self):
        """Should include extra fields in JSON output."""
        import json

        from backpropagate.ui_security import JSONSecurityFormatter

        formatter = JSONSecurityFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.request_id = "abc123"
        record.client_ip = "10.0.0.1"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["request_id"] == "abc123", "request_id must be included"
        assert parsed["client_ip"] == "10.0.0.1", "client_ip must be included"

    def test_handles_different_log_levels(self):
        """Should format different log levels correctly."""
        import json

        from backpropagate.ui_security import JSONSecurityFormatter

        formatter = JSONSecurityFormatter()

        for level, name in [(logging.DEBUG, "DEBUG"), (logging.WARNING, "WARNING"),
                            (logging.ERROR, "ERROR"), (logging.CRITICAL, "CRITICAL")]:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )

            result = formatter.format(record)
            parsed = json.loads(result)
            assert parsed["level"] == name, f"Level must be {name}"

    def test_result_is_single_line(self):
        """JSON output should be single line for log aggregation."""
        import json

        from backpropagate.ui_security import JSONSecurityFormatter

        formatter = JSONSecurityFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Multi\nline\nmessage",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        # JSON should be single line (no pretty printing)
        assert "\n" not in result.strip(), "JSON output must be single line"
        # But the message content can have newlines
        parsed = json.loads(result)
        assert "\n" in parsed["message"]


class TestSecurityConfigNewFields:
    """Tests for new SecurityConfig fields."""

    def test_session_timeout_default(self):
        """Should have default session timeout."""
        from backpropagate.ui_security import SecurityConfig

        config = SecurityConfig()
        assert config.session_timeout_minutes == 60

    def test_concurrent_limits_defaults(self):
        """Should have default concurrency limits."""
        from backpropagate.ui_security import SecurityConfig

        config = SecurityConfig()
        assert config.max_concurrent_trainings == 1
        assert config.max_concurrent_exports == 2

    def test_health_check_enabled_by_default(self):
        """Should have health check enabled by default."""
        from backpropagate.ui_security import SecurityConfig

        config = SecurityConfig()
        assert config.health_check_enabled is True
        assert config.health_check_include_gpu is True

    def test_validate_file_magic_disabled_by_default(self):
        """File magic validation should be disabled by default."""
        from backpropagate.ui_security import SecurityConfig

        config = SecurityConfig()
        assert config.validate_file_magic is False


# =============================================================================
# SB-T-002: get_ui_output_dir + BACKPROPAGATE_UI__OUTPUT_DIR env override
#
# Wave 1's F-002 fix introduced get_ui_output_dir() as the single allowed-base
# directory for every UI-initiated filesystem write (LoRA save, GGUF export,
# converted datasets, Modelfiles, Ollama registration). The function had ZERO
# direct test coverage before this class. A future refactor that dropped the
# env-var read, skipped the mkdir, or removed the system-path denylist would
# have silently degraded the security boundary with no CI signal.
# =============================================================================

class TestGetUiOutputDir:
    """Tests for get_ui_output_dir() resolution + env override + FB-003 denylist."""

    def test_default_output_dir_under_home(self, monkeypatch):
        """Default base resolves to ~/.backpropagate/ui-outputs under home.

        Pins the documented default. A change that points the default at, e.g.,
        a system temp dir would break the contract that UI outputs live under
        the operator's home (where they expect to find them).
        """
        from backpropagate.ui_security import get_ui_output_dir

        monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)

        result = get_ui_output_dir()

        expected = (Path.home() / ".backpropagate" / "ui-outputs").resolve()
        assert result == expected
        assert result.is_absolute()

    def test_env_override_respected(self, tmp_path, monkeypatch):
        """BACKPROPAGATE_UI__OUTPUT_DIR override is honored.

        Pins the operator-override path: setting the env var redirects every
        UI-initiated write into the supplied directory.
        """
        from backpropagate.ui_security import get_ui_output_dir

        target = tmp_path / "operator-override"
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(target))

        result = get_ui_output_dir()

        assert result == target.resolve()

    def test_creates_dir_on_first_call(self, tmp_path, monkeypatch):
        """Directory is created (with parents) on first call.

        Pins the mkdir-on-first-call contract — UI sinks pass the result to
        downstream APIs expecting a writable directory, so eliding the mkdir
        would surface as silent FileNotFoundError at the first write.
        """
        from backpropagate.ui_security import get_ui_output_dir

        # Use a nested-but-nonexistent target so we also catch parents=True
        target = tmp_path / "level1" / "level2" / "ui-outputs"
        assert not target.exists()
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(target))

        result = get_ui_output_dir()

        assert result.exists()
        assert result.is_dir()
        assert result == target.resolve()

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        """'~' in env override expands to the user's home.

        We point HOME / USERPROFILE at a tmp dir so the test doesn't actually
        create files under the real user home (which would survive teardown).
        Path.expanduser() reads those env vars on Posix / Windows respectively.
        """
        import sys

        from backpropagate.ui_security import get_ui_output_dir

        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()

        # expanduser reads HOME on Posix, USERPROFILE on Windows
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("USERPROFILE", str(fake_home))
        if sys.platform.startswith("win"):
            # On Windows, expanduser also consults HOMEDRIVE+HOMEPATH
            monkeypatch.setenv("HOMEDRIVE", str(fake_home.drive) if fake_home.drive else "")
            monkeypatch.setenv("HOMEPATH", str(fake_home)[len(fake_home.drive):] if fake_home.drive else str(fake_home))

        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", "~/custom-backprop-test")

        result = get_ui_output_dir()

        # Tilde must expand to (the patched) home, NOT the real user home
        assert result == (fake_home / "custom-backprop-test").resolve(), (
            f"Expected tilde to expand under fake home {fake_home}, "
            f"got {result}"
        )
        assert result.exists()

    def test_relative_path_absolutized(self, tmp_path, monkeypatch):
        """Relative env-override values get resolved to absolute paths."""
        from backpropagate.ui_security import get_ui_output_dir

        # Chdir to tmp_path so 'rel/foo' resolves predictably under it
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", "rel/foo")

        result = get_ui_output_dir()

        assert result.is_absolute()
        assert result == (tmp_path / "rel" / "foo").resolve()

    def test_idempotent_across_calls(self, tmp_path, monkeypatch):
        """Repeated calls return identical paths and don't fail on existing dir."""
        from backpropagate.ui_security import get_ui_output_dir

        target = tmp_path / "ui-outputs-idem"
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(target))

        first = get_ui_output_dir()
        second = get_ui_output_dir()

        assert first == second
        assert first.exists()

    def test_rejects_forbidden_base(self, monkeypatch):
        """FB-003: env override pointing at /etc raises UI_OUTPUT_DIR_FORBIDDEN.

        Guards the post-Wave-3 frontend hardening (FB-003 denylist). If the
        frontend amend hasn't landed yet, the test skips with a clear reason
        so it doesn't block this wave's merge.
        """
        # /etc is a Unix system root and won't exist on Windows; we still try
        # the call because the denylist is lexical (does not require the path
        # to exist on this host). The denylist on Windows specifically denies
        # C:\Windows / C:\Program Files / C:\ProgramData / C:\.
        import sys

        from backpropagate.exceptions import BackpropagateError
        from backpropagate.ui_security import get_ui_output_dir
        if sys.platform.startswith("win"):
            forbidden = "C:\\Windows"
        else:
            forbidden = "/etc"

        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", forbidden)

        # FB-003 contract: denylisted bases MUST raise BackpropagateError with
        # the structured code='UI_OUTPUT_DIR_FORBIDDEN'. Anything looser (plain
        # ValueError, OSError, etc.) is a regression — the user-facing error
        # surface depends on the code for programmatic handling.
        with pytest.raises(BackpropagateError) as exc_info:
            get_ui_output_dir()
        assert exc_info.value.code == "UI_OUTPUT_DIR_FORBIDDEN", (
            f"Expected BackpropagateError(code='UI_OUTPUT_DIR_FORBIDDEN') "
            f"for forbidden base {forbidden!r}, got code={exc_info.value.code!r}"
        )
