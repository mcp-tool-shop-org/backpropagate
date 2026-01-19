"""
Tests for advanced security features: JWT, CSRF, CSP.

Tests the security modules added based on 2026 best practices:
- JWT session management
- CSRF protection
- Content Security Policy headers
- SecureSessionHandler combined authentication
"""

import time
import pytest
from unittest.mock import patch, MagicMock

from backpropagate.ui_security import (
    # JWT
    JWTConfig,
    JWTManager,
    JWT_AVAILABLE,
    # CSRF
    CSRFToken,
    CSRFProtection,
    # Combined Session
    SecureSessionHandler,
    get_secure_session_handler,
    # CSP
    CSPConfig,
    ContentSecurityPolicy,
    DEFAULT_GRADIO_CSP,
    get_gradio_csp,
    apply_security_headers,
)


# =============================================================================
# JWT TESTS
# =============================================================================

class TestJWTConfig:
    """Tests for JWTConfig dataclass."""

    def test_default_config(self):
        """JWTConfig has sensible defaults."""
        config = JWTConfig()
        assert config.algorithm == "HS256"
        assert config.expiry_minutes == 30
        assert config.issuer == "backpropagate"
        assert config.audience == "backpropagate-ui"
        assert config.refresh_enabled is True
        assert config.refresh_expiry_minutes == 1440  # 24 hours

    def test_custom_config(self):
        """JWTConfig accepts custom values."""
        config = JWTConfig(
            secret="my-secret",
            algorithm="HS512",
            expiry_minutes=60,
            issuer="my-app",
        )
        assert config.secret == "my-secret"
        assert config.algorithm == "HS512"
        assert config.expiry_minutes == 60
        assert config.issuer == "my-app"


@pytest.mark.skipif(not JWT_AVAILABLE, reason="PyJWT not installed")
class TestJWTManager:
    """Tests for JWTManager class."""

    def test_create_manager_with_config(self):
        """JWTManager can be created with config."""
        config = JWTConfig(secret="test-secret-123456789012345678901234")
        manager = JWTManager(config)
        assert manager.config.secret == "test-secret-123456789012345678901234"

    def test_create_manager_generates_secret(self):
        """JWTManager generates random secret if not provided."""
        manager = JWTManager()
        assert manager.config.secret != ""
        assert len(manager.config.secret) > 20

    def test_create_access_token(self):
        """JWTManager creates access tokens."""
        config = JWTConfig(secret="test-secret-123456789012345678901234")
        manager = JWTManager(config)

        token = manager.create_token("user123")

        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self):
        """JWTManager creates refresh tokens."""
        config = JWTConfig(secret="test-secret-123456789012345678901234")
        manager = JWTManager(config)

        token = manager.create_token("user123", is_refresh=True)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_valid_token(self):
        """JWTManager verifies valid tokens."""
        config = JWTConfig(secret="test-secret-123456789012345678901234")
        manager = JWTManager(config)

        token = manager.create_token("user123")
        valid, payload, msg = manager.verify_token(token)

        assert valid is True
        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["type"] == "access"
        assert msg == "Token valid"

    def test_verify_refresh_token(self):
        """JWTManager verifies refresh tokens."""
        config = JWTConfig(secret="test-secret-123456789012345678901234")
        manager = JWTManager(config)

        token = manager.create_token("user123", is_refresh=True)
        valid, payload, msg = manager.verify_token(token, expected_type="refresh")

        assert valid is True
        assert payload is not None
        assert payload["type"] == "refresh"

    def test_verify_wrong_token_type(self):
        """JWTManager rejects wrong token type."""
        config = JWTConfig(secret="test-secret-123456789012345678901234")
        manager = JWTManager(config)

        access_token = manager.create_token("user123")
        valid, payload, msg = manager.verify_token(access_token, expected_type="refresh")

        assert valid is False
        assert "Invalid token type" in msg

    def test_verify_invalid_token(self):
        """JWTManager rejects invalid tokens."""
        config = JWTConfig(secret="test-secret-123456789012345678901234")
        manager = JWTManager(config)

        valid, payload, msg = manager.verify_token("invalid.token.here")

        assert valid is False
        assert payload is None
        assert "Invalid token" in msg

    def test_verify_expired_token(self):
        """JWTManager rejects expired tokens."""
        config = JWTConfig(
            secret="test-secret-123456789012345678901234",
            expiry_minutes=0,  # Immediate expiry
        )
        manager = JWTManager(config)

        token = manager.create_token("user123")
        time.sleep(0.1)  # Wait for expiry

        valid, payload, msg = manager.verify_token(token)

        assert valid is False
        assert "expired" in msg.lower()

    def test_refresh_access_token(self):
        """JWTManager refreshes access token from refresh token."""
        config = JWTConfig(secret="test-secret-123456789012345678901234")
        manager = JWTManager(config)

        refresh_token = manager.create_token("user123", is_refresh=True)
        success, new_token, msg = manager.refresh_access_token(refresh_token)

        assert success is True
        assert new_token is not None
        assert msg == "Token refreshed"

        # Verify new token is valid
        valid, payload, _ = manager.verify_token(new_token)
        assert valid is True
        assert payload["sub"] == "user123"

    def test_refresh_with_access_token_fails(self):
        """JWTManager rejects access token for refresh."""
        config = JWTConfig(secret="test-secret-123456789012345678901234")
        manager = JWTManager(config)

        access_token = manager.create_token("user123")
        success, new_token, msg = manager.refresh_access_token(access_token)

        assert success is False
        assert new_token is None

    def test_create_token_with_additional_claims(self):
        """JWTManager includes additional claims."""
        config = JWTConfig(secret="test-secret-123456789012345678901234")
        manager = JWTManager(config)

        token = manager.create_token(
            "user123",
            additional_claims={"role": "admin", "org": "acme"}
        )
        valid, payload, _ = manager.verify_token(token)

        assert valid is True
        assert payload["role"] == "admin"
        assert payload["org"] == "acme"


class TestJWTManagerWithoutPyJWT:
    """Tests for JWTManager when PyJWT is not available."""

    def test_create_token_raises_without_jwt(self):
        """create_token raises RuntimeError without PyJWT."""
        config = JWTConfig(secret="test-secret")
        manager = JWTManager(config)

        with patch("backpropagate.ui_security.JWT_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="PyJWT not installed"):
                manager.create_token("user123")

    def test_verify_token_returns_false_without_jwt(self):
        """verify_token returns False without PyJWT."""
        config = JWTConfig(secret="test-secret")
        manager = JWTManager(config)

        with patch("backpropagate.ui_security.JWT_AVAILABLE", False):
            valid, payload, msg = manager.verify_token("some-token")
            assert valid is False
            assert payload is None
            assert "not installed" in msg.lower()


# =============================================================================
# CSRF TESTS
# =============================================================================

class TestCSRFToken:
    """Tests for CSRFToken dataclass."""

    def test_csrf_token_creation(self):
        """CSRFToken stores token data."""
        token = CSRFToken(
            token="abc123",
            created_at=time.time(),
            expiry_minutes=30,
        )
        assert token.token == "abc123"
        assert token.expiry_minutes == 30

    def test_csrf_token_default_expiry(self):
        """CSRFToken has default expiry of 60 minutes."""
        token = CSRFToken(token="abc", created_at=time.time())
        assert token.expiry_minutes == 60


class TestCSRFProtection:
    """Tests for CSRFProtection class."""

    def test_create_csrf_protection(self):
        """CSRFProtection can be created."""
        csrf = CSRFProtection()
        assert csrf.expiry_minutes == 60

    def test_create_with_custom_expiry(self):
        """CSRFProtection accepts custom expiry."""
        csrf = CSRFProtection(expiry_minutes=30)
        assert csrf.expiry_minutes == 30

    def test_generate_token(self):
        """CSRFProtection generates tokens."""
        csrf = CSRFProtection()
        token = csrf.generate_token("session123")

        assert isinstance(token, str)
        assert len(token) > 20

    def test_generate_unique_tokens(self):
        """CSRFProtection generates unique tokens."""
        csrf = CSRFProtection()
        token1 = csrf.generate_token("session1")
        token2 = csrf.generate_token("session2")

        assert token1 != token2

    def test_validate_valid_token(self):
        """CSRFProtection validates correct token."""
        csrf = CSRFProtection()
        session_id = "session123"
        token = csrf.generate_token(session_id)

        valid, msg = csrf.validate_token(session_id, token)

        assert valid is True
        assert "valid" in msg.lower()

    def test_validate_invalid_token(self):
        """CSRFProtection rejects wrong token."""
        csrf = CSRFProtection()
        session_id = "session123"
        csrf.generate_token(session_id)

        valid, msg = csrf.validate_token(session_id, "wrong-token")

        assert valid is False
        assert "Invalid" in msg

    def test_validate_missing_session(self):
        """CSRFProtection rejects unknown session."""
        csrf = CSRFProtection()

        valid, msg = csrf.validate_token("unknown-session", "any-token")

        assert valid is False
        assert "No CSRF token found" in msg

    def test_validate_expired_token(self):
        """CSRFProtection rejects expired token."""
        csrf = CSRFProtection(expiry_minutes=0)  # Immediate expiry
        session_id = "session123"
        token = csrf.generate_token(session_id)

        time.sleep(0.1)  # Wait for expiry

        valid, msg = csrf.validate_token(session_id, token)

        assert valid is False
        # Token could be expired or cleaned up (both are valid outcomes)
        assert "expired" in msg.lower() or "no csrf token" in msg.lower()

    def test_validate_consumes_token(self):
        """CSRFProtection consumes token by default."""
        csrf = CSRFProtection()
        session_id = "session123"
        token = csrf.generate_token(session_id)

        # First validation should succeed
        valid1, _ = csrf.validate_token(session_id, token, consume=True)
        assert valid1 is True

        # Second validation should fail (token consumed)
        valid2, msg = csrf.validate_token(session_id, token)
        assert valid2 is False
        assert "No CSRF token found" in msg

    def test_validate_without_consuming(self):
        """CSRFProtection can validate without consuming."""
        csrf = CSRFProtection()
        session_id = "session123"
        token = csrf.generate_token(session_id)

        # First validation (no consume)
        valid1, _ = csrf.validate_token(session_id, token, consume=False)
        assert valid1 is True

        # Second validation should still succeed
        valid2, _ = csrf.validate_token(session_id, token, consume=False)
        assert valid2 is True

    def test_cleanup_expired_tokens(self):
        """CSRFProtection cleans up expired tokens."""
        csrf = CSRFProtection(expiry_minutes=0)

        # Generate some tokens
        csrf.generate_token("session1")
        csrf.generate_token("session2")

        time.sleep(0.1)

        # Generate new token (triggers cleanup)
        csrf.generate_token("session3")

        # Old sessions should be cleaned up
        assert len(csrf._tokens) == 1
        assert "session3" in csrf._tokens


# =============================================================================
# SECURE SESSION HANDLER TESTS
# =============================================================================

@pytest.mark.skipif(not JWT_AVAILABLE, reason="PyJWT not installed")
class TestSecureSessionHandler:
    """Tests for SecureSessionHandler class."""

    def test_create_handler(self):
        """SecureSessionHandler can be created."""
        handler = SecureSessionHandler()
        assert handler.jwt is not None
        assert handler.csrf is not None

    def test_create_handler_with_config(self):
        """SecureSessionHandler accepts JWT config."""
        jwt_config = JWTConfig(
            secret="test-secret-123456789012345678901234",
            expiry_minutes=60,
        )
        handler = SecureSessionHandler(jwt_config)
        assert handler.jwt.config.expiry_minutes == 60

    def test_login_returns_tokens(self):
        """login returns access, refresh, and CSRF tokens."""
        jwt_config = JWTConfig(secret="test-secret-123456789012345678901234")
        handler = SecureSessionHandler(jwt_config)

        result = handler.login("user123")

        assert "access_token" in result
        assert "refresh_token" in result
        assert "csrf_token" in result
        assert len(result["access_token"]) > 0
        assert len(result["refresh_token"]) > 0
        assert len(result["csrf_token"]) > 0

    def test_validate_request_success(self):
        """validate_request succeeds with valid tokens."""
        jwt_config = JWTConfig(secret="test-secret-123456789012345678901234")
        handler = SecureSessionHandler(jwt_config)

        tokens = handler.login("user123")
        valid, user_id, msg = handler.validate_request(
            tokens["access_token"],
            tokens["csrf_token"],
        )

        assert valid is True
        assert user_id == "user123"
        assert "valid" in msg.lower()

    def test_validate_request_invalid_jwt(self):
        """validate_request fails with invalid JWT."""
        jwt_config = JWTConfig(secret="test-secret-123456789012345678901234")
        handler = SecureSessionHandler(jwt_config)

        tokens = handler.login("user123")
        valid, user_id, msg = handler.validate_request(
            "invalid-jwt-token",
            tokens["csrf_token"],
        )

        assert valid is False
        assert user_id is None

    def test_validate_request_invalid_csrf(self):
        """validate_request fails with invalid CSRF."""
        jwt_config = JWTConfig(secret="test-secret-123456789012345678901234")
        handler = SecureSessionHandler(jwt_config)

        tokens = handler.login("user123")
        valid, user_id, msg = handler.validate_request(
            tokens["access_token"],
            "invalid-csrf-token",
        )

        assert valid is False
        assert user_id is None

    def test_validate_request_without_csrf(self):
        """validate_request can skip CSRF check."""
        jwt_config = JWTConfig(secret="test-secret-123456789012345678901234")
        handler = SecureSessionHandler(jwt_config)

        tokens = handler.login("user123")
        valid, user_id, msg = handler.validate_request(
            tokens["access_token"],
            "any-csrf-token",  # Would fail if checked
            require_csrf=False,
        )

        assert valid is True
        assert user_id == "user123"

    def test_logout(self):
        """logout ends the session."""
        jwt_config = JWTConfig(secret="test-secret-123456789012345678901234")
        handler = SecureSessionHandler(jwt_config)

        tokens = handler.login("user123")
        handler.logout(tokens["access_token"])

        # Session should be removed
        assert tokens["access_token"][:32] not in handler._active_sessions

    def test_refresh_session(self):
        """refresh_session creates new tokens."""
        jwt_config = JWTConfig(secret="test-secret-123456789012345678901234")
        handler = SecureSessionHandler(jwt_config)

        tokens = handler.login("user123")
        success, new_tokens, msg = handler.refresh_session(tokens["refresh_token"])

        assert success is True
        assert new_tokens is not None
        assert "access_token" in new_tokens
        assert "csrf_token" in new_tokens
        assert msg == "Session refreshed"

    def test_refresh_session_with_invalid_token(self):
        """refresh_session fails with invalid token."""
        jwt_config = JWTConfig(secret="test-secret-123456789012345678901234")
        handler = SecureSessionHandler(jwt_config)

        success, new_tokens, msg = handler.refresh_session("invalid-refresh-token")

        assert success is False
        assert new_tokens is None


class TestGetSecureSessionHandler:
    """Tests for get_secure_session_handler function."""

    def test_returns_singleton(self):
        """get_secure_session_handler returns same instance."""
        # Reset the global
        import backpropagate.ui_security as ui_sec
        ui_sec._secure_session_handler = None

        handler1 = get_secure_session_handler()
        handler2 = get_secure_session_handler()

        assert handler1 is handler2

        # Clean up
        ui_sec._secure_session_handler = None


# =============================================================================
# CSP TESTS
# =============================================================================

class TestCSPConfig:
    """Tests for CSPConfig dataclass."""

    def test_default_config(self):
        """CSPConfig has sensible defaults."""
        config = CSPConfig()

        assert "'self'" in config.default_src
        assert "'self'" in config.script_src
        assert "'self'" in config.style_src
        assert "'none'" in config.object_src
        assert config.report_only is False

    def test_custom_config(self):
        """CSPConfig accepts custom values."""
        config = CSPConfig(
            script_src=["'self'", "https://cdn.example.com"],
            report_only=True,
            report_uri="/csp-report",
        )

        assert "https://cdn.example.com" in config.script_src
        assert config.report_only is True
        assert config.report_uri == "/csp-report"


class TestContentSecurityPolicy:
    """Tests for ContentSecurityPolicy class."""

    def test_create_csp(self):
        """ContentSecurityPolicy can be created."""
        csp = ContentSecurityPolicy()
        assert csp.config is not None

    def test_create_csp_with_config(self):
        """ContentSecurityPolicy accepts custom config."""
        config = CSPConfig(script_src=["'self'", "'unsafe-inline'"])
        csp = ContentSecurityPolicy(config)
        assert "'unsafe-inline'" in csp.config.script_src

    def test_generate_nonce(self):
        """CSP generates valid nonce."""
        csp = ContentSecurityPolicy()
        nonce = csp.generate_nonce()

        assert isinstance(nonce, str)
        assert len(nonce) > 10
        assert csp.get_nonce() == nonce

    def test_generate_unique_nonces(self):
        """CSP generates unique nonces."""
        csp = ContentSecurityPolicy()
        nonce1 = csp.generate_nonce()
        nonce2 = csp.generate_nonce()

        assert nonce1 != nonce2

    def test_build_policy(self):
        """CSP builds policy string."""
        csp = ContentSecurityPolicy()
        policy = csp.build_policy()

        assert "default-src" in policy
        assert "script-src" in policy
        assert "style-src" in policy
        assert "'self'" in policy

    def test_build_policy_with_nonce(self):
        """CSP includes nonce in policy."""
        csp = ContentSecurityPolicy()
        nonce = csp.generate_nonce()
        policy = csp.build_policy(include_nonce=True)

        assert f"'nonce-{nonce}'" in policy

    def test_get_header_enforce(self):
        """CSP returns enforce header."""
        csp = ContentSecurityPolicy()
        name, value = csp.get_header()

        assert name == "Content-Security-Policy"
        assert len(value) > 0

    def test_get_header_report_only(self):
        """CSP returns report-only header when configured."""
        config = CSPConfig(report_only=True)
        csp = ContentSecurityPolicy(config)
        name, value = csp.get_header()

        assert name == "Content-Security-Policy-Report-Only"

    def test_get_all_security_headers(self):
        """CSP returns all security headers."""
        csp = ContentSecurityPolicy()
        headers = csp.get_all_security_headers()

        # Check CSP is included
        assert any("Content-Security-Policy" in k for k in headers)

        # Check other security headers
        assert "X-Content-Type-Options" in headers
        assert headers["X-Content-Type-Options"] == "nosniff"

        assert "X-Frame-Options" in headers
        assert headers["X-Frame-Options"] == "SAMEORIGIN"

        assert "X-XSS-Protection" in headers
        assert "1; mode=block" in headers["X-XSS-Protection"]

        assert "Referrer-Policy" in headers
        assert "Permissions-Policy" in headers

    def test_policy_includes_all_directives(self):
        """CSP policy includes all configured directives."""
        config = CSPConfig(
            default_src=["'self'"],
            script_src=["'self'", "https://scripts.com"],
            style_src=["'self'", "'unsafe-inline'"],
            img_src=["'self'", "data:"],
            font_src=["'self'"],
            connect_src=["'self'", "ws:"],
            frame_ancestors=["'none'"],
            base_uri=["'self'"],
            form_action=["'self'"],
            object_src=["'none'"],
            report_uri="/report",
        )
        csp = ContentSecurityPolicy(config)
        policy = csp.build_policy()

        assert "default-src 'self'" in policy
        assert "script-src 'self' https://scripts.com" in policy
        assert "style-src 'self' 'unsafe-inline'" in policy
        assert "img-src 'self' data:" in policy
        assert "font-src 'self'" in policy
        assert "connect-src 'self' ws:" in policy
        assert "frame-ancestors 'none'" in policy
        assert "base-uri 'self'" in policy
        assert "form-action 'self'" in policy
        assert "object-src 'none'" in policy
        assert "report-uri /report" in policy


class TestDefaultGradioCSP:
    """Tests for DEFAULT_GRADIO_CSP."""

    def test_gradio_csp_allows_eval(self):
        """Gradio CSP allows unsafe-eval for dynamic components."""
        assert "'unsafe-eval'" in DEFAULT_GRADIO_CSP.script_src

    def test_gradio_csp_allows_inline_styles(self):
        """Gradio CSP allows inline styles."""
        assert "'unsafe-inline'" in DEFAULT_GRADIO_CSP.style_src

    def test_gradio_csp_allows_websocket(self):
        """Gradio CSP allows WebSocket connections."""
        assert "ws:" in DEFAULT_GRADIO_CSP.connect_src
        assert "wss:" in DEFAULT_GRADIO_CSP.connect_src

    def test_gradio_csp_allows_data_images(self):
        """Gradio CSP allows data: images."""
        assert "data:" in DEFAULT_GRADIO_CSP.img_src

    def test_gradio_csp_blocks_objects(self):
        """Gradio CSP blocks object/embed."""
        assert "'none'" in DEFAULT_GRADIO_CSP.object_src


class TestGetGradioCSP:
    """Tests for get_gradio_csp function."""

    def test_returns_csp(self):
        """get_gradio_csp returns ContentSecurityPolicy."""
        csp = get_gradio_csp()
        assert isinstance(csp, ContentSecurityPolicy)

    def test_report_only_mode(self):
        """get_gradio_csp supports report-only mode."""
        csp = get_gradio_csp(report_only=True)
        name, _ = csp.get_header()
        assert name == "Content-Security-Policy-Report-Only"

    def test_enforce_mode(self):
        """get_gradio_csp defaults to enforce mode."""
        csp = get_gradio_csp(report_only=False)
        name, _ = csp.get_header()
        assert name == "Content-Security-Policy"


class TestApplySecurityHeaders:
    """Tests for apply_security_headers function."""

    def test_apply_to_headers_dict(self):
        """apply_security_headers works with headers dict."""
        response = MagicMock()
        response.headers = {}

        apply_security_headers(response)

        assert "Content-Security-Policy" in response.headers
        assert "X-Content-Type-Options" in response.headers

    def test_apply_with_custom_csp(self):
        """apply_security_headers accepts custom CSP."""
        response = MagicMock()
        response.headers = {}

        config = CSPConfig(report_only=True)
        csp = ContentSecurityPolicy(config)

        apply_security_headers(response, csp)

        assert "Content-Security-Policy-Report-Only" in response.headers

    def test_apply_to_set_header_method(self):
        """apply_security_headers works with set_header method."""
        response = MagicMock(spec=[])  # Empty spec, no headers attribute
        response.set_header = MagicMock()

        apply_security_headers(response)

        # Should call set_header
        assert response.set_header.called


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.skipif(not JWT_AVAILABLE, reason="PyJWT not installed")
class TestSecurityIntegration:
    """Integration tests for security components."""

    def test_full_login_flow(self):
        """Complete login, validate, refresh, logout flow."""
        jwt_config = JWTConfig(secret="test-secret-123456789012345678901234")
        handler = SecureSessionHandler(jwt_config)

        # 1. Login
        tokens = handler.login("user123", additional_claims={"role": "admin"})
        assert tokens is not None

        # 2. Validate request
        valid, user_id, _ = handler.validate_request(
            tokens["access_token"],
            tokens["csrf_token"],
        )
        assert valid is True
        assert user_id == "user123"

        # 3. Refresh session
        success, new_tokens, _ = handler.refresh_session(tokens["refresh_token"])
        assert success is True
        assert new_tokens is not None

        # 4. Validate with new tokens
        valid2, user_id2, _ = handler.validate_request(
            new_tokens["access_token"],
            new_tokens["csrf_token"],
        )
        assert valid2 is True
        assert user_id2 == "user123"

        # 5. Logout
        handler.logout(new_tokens["access_token"])

    def test_csp_with_nonce_workflow(self):
        """CSP with nonce generation workflow."""
        csp = ContentSecurityPolicy()

        # Generate nonce for this request
        nonce = csp.generate_nonce()

        # Build policy with nonce
        policy = csp.build_policy(include_nonce=True)

        # Verify nonce is in policy
        assert f"'nonce-{nonce}'" in policy

        # Get header
        name, value = csp.get_header(include_nonce=True)
        assert name == "Content-Security-Policy"
        assert f"'nonce-{nonce}'" in value
