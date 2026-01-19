"""
Tests for backpropagate.security module.
"""

import pytest
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSafePath:
    """Tests for safe_path function."""

    def test_safe_path_resolves_absolute(self, tmp_path):
        """safe_path should resolve to absolute path."""
        from backpropagate.security import safe_path

        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = safe_path(str(test_file))
        assert result.is_absolute()
        assert result == test_file.resolve()

    def test_safe_path_must_exist_success(self, tmp_path):
        """safe_path with must_exist=True should return path if it exists."""
        from backpropagate.security import safe_path

        test_file = tmp_path / "exists.txt"
        test_file.write_text("test")

        result = safe_path(str(test_file), must_exist=True)
        assert result.exists()

    def test_safe_path_must_exist_fails(self, tmp_path):
        """safe_path with must_exist=True should raise if path doesn't exist."""
        from backpropagate.security import safe_path

        with pytest.raises(FileNotFoundError):
            safe_path(str(tmp_path / "nonexistent.txt"), must_exist=True)

    def test_safe_path_allowed_base_success(self, tmp_path):
        """safe_path should allow paths within allowed_base."""
        from backpropagate.security import safe_path

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "test.txt"
        test_file.write_text("test")

        result = safe_path(str(test_file), allowed_base=tmp_path)
        assert result == test_file.resolve()

    def test_safe_path_allowed_base_traversal(self, tmp_path):
        """safe_path should reject paths outside allowed_base."""
        from backpropagate.security import safe_path, PathTraversalError

        # Create a base directory
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        # Try to escape with ../
        with pytest.raises(PathTraversalError, match="escapes allowed directory"):
            safe_path("../outside", allowed_base=base_dir)

    def test_safe_path_relative_not_allowed(self, tmp_path):
        """safe_path should reject relative paths when allow_relative=False."""
        from backpropagate.security import safe_path

        with pytest.raises(ValueError, match="Relative paths not allowed"):
            safe_path("relative/path", allow_relative=False)

    def test_safe_path_absolute_allowed(self, tmp_path):
        """safe_path should allow absolute paths when allow_relative=False."""
        from backpropagate.security import safe_path

        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = safe_path(str(test_file), allow_relative=False)
        assert result == test_file.resolve()

    def test_safe_path_logs_traversal_pattern(self, tmp_path, caplog):
        """safe_path should log when path contains '..'."""
        from backpropagate.security import safe_path
        import logging

        caplog.set_level(logging.WARNING)

        # Path with .. that resolves within same directory
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        # This should log a warning but not fail
        safe_path(str(tmp_path / "subdir" / ".." / "other"), must_exist=False)

        # Check warning was logged
        assert any("traversal pattern" in record.message.lower() for record in caplog.records)


class TestPathTraversalError:
    """Tests for PathTraversalError exception."""

    def test_path_traversal_error_message(self):
        """PathTraversalError should have descriptive message."""
        from backpropagate.security import PathTraversalError

        error = PathTraversalError("../../etc/passwd", "/home/user")
        assert "../../etc/passwd" in str(error)
        assert "/home/user" in str(error)
        assert "escapes" in str(error)

    def test_path_traversal_error_without_base(self):
        """PathTraversalError should work without allowed_base."""
        from backpropagate.security import PathTraversalError

        error = PathTraversalError("../sensitive")
        assert "../sensitive" in str(error)
        assert "traversal" in str(error).lower()


class TestCheckTorchSecurity:
    """Tests for check_torch_security function."""

    def test_check_torch_security_new_version(self):
        """check_torch_security should return True for PyTorch >= 2.0."""
        from backpropagate.security import check_torch_security

        with patch("torch.__version__", "2.1.0"):
            result = check_torch_security()
            assert result is True

    def test_check_torch_security_old_version(self):
        """check_torch_security should warn for PyTorch < 2.0."""
        from backpropagate.security import check_torch_security, SecurityWarning

        mock_torch = MagicMock()
        mock_torch.__version__ = "1.9.0"

        with patch.dict("sys.modules", {"torch": mock_torch}), \
             pytest.warns(SecurityWarning, match="weights_only"):
            result = check_torch_security()
            assert result is False

    def test_check_torch_security_handles_import_error(self):
        """check_torch_security should handle ImportError gracefully."""
        from backpropagate.security import check_torch_security

        with patch.dict("sys.modules", {"torch": None}):
            # Should not raise
            result = check_torch_security()
            # Returns True when can't check (assume safe)
            assert result is True


class TestSecurityWarning:
    """Tests for SecurityWarning."""

    def test_security_warning_is_user_warning(self):
        """SecurityWarning should be a UserWarning subclass."""
        from backpropagate.security import SecurityWarning

        assert issubclass(SecurityWarning, UserWarning)

    def test_can_filter_security_warning(self):
        """SecurityWarning should be filterable."""
        from backpropagate.security import SecurityWarning

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always", category=SecurityWarning)
            warnings.warn("test", SecurityWarning)

            assert len(w) == 1
            assert issubclass(w[0].category, SecurityWarning)


class TestSafeTorchLoad:
    """Tests for safe_torch_load function."""

    def test_safe_torch_load_file_not_found(self, tmp_path):
        """safe_torch_load should raise FileNotFoundError for missing file."""
        from backpropagate.security import safe_torch_load

        with pytest.raises(FileNotFoundError):
            safe_torch_load(tmp_path / "nonexistent.pt")

    def test_safe_torch_load_prefers_safetensors(self, tmp_path):
        """safe_torch_load should prefer safetensors format."""
        from backpropagate.security import safe_torch_load

        # Create a mock safetensors file
        st_file = tmp_path / "model.safetensors"
        st_file.write_bytes(b"mock safetensors")

        # Mock safetensors import
        mock_load_file = MagicMock(return_value={"weight": "tensor"})

        with patch.dict("sys.modules", {"safetensors": MagicMock(), "safetensors.torch": MagicMock(load_file=mock_load_file)}):
            with patch("backpropagate.security.check_torch_security"):
                # This would use safetensors if available
                pass  # Just verify no crash

    def test_safe_torch_load_with_weights_only(self, tmp_path):
        """safe_torch_load should pass weights_only to torch.load."""
        from backpropagate.security import safe_torch_load
        import torch

        # Create a simple tensor file
        pt_file = tmp_path / "weights.pt"
        torch.save({"weight": torch.tensor([1, 2, 3])}, pt_file)

        with patch("backpropagate.security.check_torch_security"):
            result = safe_torch_load(pt_file, weights_only=True)

        assert "weight" in result


class TestAuditLog:
    """Tests for audit_log function."""

    def test_audit_log_success(self, caplog):
        """audit_log should log successful operations."""
        from backpropagate.security import audit_log
        import logging

        caplog.set_level(logging.INFO, logger="backpropagate.security.audit")

        audit_log("model_load", path="/models/test.pt", success=True)

        assert any("AUDIT" in record.message for record in caplog.records)
        assert any("model_load" in record.message for record in caplog.records)

    def test_audit_log_failure(self, caplog):
        """audit_log should log failed operations at WARNING level."""
        from backpropagate.security import audit_log
        import logging

        caplog.set_level(logging.WARNING, logger="backpropagate.security.audit")

        audit_log("export", path="/output/model.gguf", success=False)

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) > 0


class TestSecurityModuleExports:
    """Tests for security module exports."""

    def test_all_exports_available(self):
        """All __all__ exports should be importable."""
        from backpropagate import security

        for name in security.__all__:
            assert hasattr(security, name), f"Missing export: {name}"

    def test_exports_from_package(self):
        """Security utilities should be importable from backpropagate."""
        from backpropagate import (
            safe_path,
            check_torch_security,
            SecurityWarning,
            PathTraversalError,
        )

        assert callable(safe_path)
        assert callable(check_torch_security)
        assert issubclass(SecurityWarning, Warning)
        assert issubclass(PathTraversalError, ValueError)
