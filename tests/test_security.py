"""
Tests for backpropagate.security module.
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest


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
        from backpropagate.security import PathTraversalError, safe_path

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
        """safe_path should log AND raise when an absolute path contains '..'.

        Under the new stricter default behavior (no ``allowed_base``), an
        absolute path containing ``..`` is rejected outright via
        :class:`PathTraversalError` AND a warning is logged. The legacy
        warn-only branch only fires for relative ``..`` paths that normalize
        back inside the current working directory.
        """
        import logging

        from backpropagate.security import PathTraversalError, safe_path

        caplog.set_level(logging.WARNING)

        # Path with .. that resolves within same directory
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        # Absolute path with ".." -> now raises AND warns.
        with pytest.raises(PathTraversalError):
            safe_path(str(tmp_path / "subdir" / ".." / "other"), must_exist=False)

        # Check warning was logged
        assert any("traversal pattern" in record.message.lower() for record in caplog.records)

    # ------------------------------------------------------------------ #
    # SB-T-004: complete the raise-vs-warn matrix for the Wave 1
    # safe_path harden (security.py:129-155). Without these, a future
    # refactor that drops the `if allowed_base is None` gate (or the
    # warn-only legacy branch for relative-cwd-bound ".." normalisation)
    # would not fail any test. The matrix exhaustively covers:
    #   (a) absolute  + ".." + no base                -> RAISE
    #   (b) relative  + ".." + outside cwd + no base  -> RAISE
    #   (c) relative  + ".." + inside  cwd + no base  -> WARN-ONLY (legacy)
    #   (d) absolute  + clean (no "..") + no base     -> unchanged
    #   (e) absolute  + outside allowed_base          -> RAISE
    #   (f) absolute  + inside  allowed_base          -> OK
    #   (g) absolute  + ".." + inside allowed_base    -> OK (strict gate is
    #                                                       allowed_base=None only)
    # ------------------------------------------------------------------ #

    def test_safe_path_relative_traversal_outside_cwd_raises(self, tmp_path, monkeypatch):
        """SB-T-004 case (b): relative `..` resolving OUTSIDE cwd raises."""
        from backpropagate.security import PathTraversalError, safe_path

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(subdir)

        # `../../escape` from tmp_path/subdir resolves to tmp_path.parent — outside cwd
        with pytest.raises(PathTraversalError):
            safe_path("../../escape", must_exist=False)

    def test_safe_path_relative_with_dotdot_inside_cwd_warns_only(
        self, tmp_path, caplog, monkeypatch
    ):
        """SB-T-004 case (c): relative `..` that normalizes back inside cwd warns, no raise.

        This is the legacy preserved branch — security.py:154 keeps
        warn-only behaviour so simple-normalization callers (e.g.
        ``safe_path('subdir/../other')`` for path cleanup) are unaffected.
        """
        import logging

        from backpropagate.security import safe_path

        # cwd is tmp_path; "subdir/../other" normalizes to tmp_path/other — still inside cwd
        monkeypatch.chdir(tmp_path)
        caplog.set_level(logging.WARNING)

        # Must not raise
        result = safe_path("subdir/../other", must_exist=False)

        # Resolves to a path under cwd
        assert result == (tmp_path / "other").resolve()
        # And the warn-only log fires
        assert any(
            "traversal pattern" in record.message.lower() for record in caplog.records
        ), (
            "Legacy warn-only path expected to log 'traversal pattern' "
            f"(security.py:154). caplog records: {[r.message for r in caplog.records]}"
        )

    def test_safe_path_absolute_clean_no_dotdot_returns_unchanged(self, tmp_path):
        """SB-T-004 case (d): clean absolute path with no '..' is unchanged."""
        from backpropagate.security import safe_path

        result = safe_path(str(tmp_path), must_exist=False)
        assert result == tmp_path.resolve()

    def test_safe_path_absolute_with_dotdot_raises(self, tmp_path):
        """SB-T-004 case (a): absolute path with `..` and no allowed_base raises."""
        from backpropagate.security import PathTraversalError, safe_path

        # Absolute path with ".." raises regardless of where it resolves
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        with pytest.raises(PathTraversalError):
            safe_path(str(subdir / ".." / "etc"), must_exist=False)

    def test_safe_path_with_allowed_base_outside_raises(self, tmp_path):
        """SB-T-004 case (e): path outside allowed_base raises PathTraversalError."""
        from backpropagate.security import PathTraversalError, safe_path

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        with pytest.raises(PathTraversalError):
            safe_path(str(outside), allowed_base=base_dir)

    def test_safe_path_with_allowed_base_inside_succeeds(self, tmp_path):
        """SB-T-004 case (f): path inside allowed_base returns the resolved path."""
        from backpropagate.security import safe_path

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        inside = base_dir / "child" / "leaf"

        result = safe_path(str(inside), allowed_base=base_dir, must_exist=False)
        assert result == inside.resolve()

    def test_safe_path_absolute_dotdot_with_allowed_base_inside_succeeds(self, tmp_path):
        """SB-T-004 case (g): absolute `..` path INSIDE allowed_base succeeds.

        The strict `..` check is gated by `if allowed_base is None`
        (security.py:137). With an allowed_base, the relative_to() check
        governs — and a `..` path that resolves back inside the base
        (e.g. /base/child/../sibling) must NOT trip the strict check.
        Pins the gate so a refactor that drops the `is None` clause
        doesn't break callers using path cleanup inside a known scope.
        """
        from backpropagate.security import safe_path

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        (base_dir / "child").mkdir()
        (base_dir / "sibling").mkdir()

        # /base/child/../sibling resolves to /base/sibling — inside allowed_base
        result = safe_path(
            str(base_dir / "child" / ".." / "sibling"),
            allowed_base=base_dir,
            must_exist=False,
        )
        assert result == (base_dir / "sibling").resolve()

    def test_safe_path_relative_dotdot_with_allowed_base_uses_base_check(self, tmp_path):
        """SB-T-004: with allowed_base, relative `..` is governed by relative_to() not the strict check.

        Verifies the gating: the strict `..` check at security.py:137 is
        ONLY active when allowed_base is None. When allowed_base IS set,
        the relative_to() check fires (and rejects only if the resolved
        path escapes the base).
        """
        from backpropagate.security import PathTraversalError, safe_path

        base_dir = tmp_path / "base"
        base_dir.mkdir()

        # Relative path that escapes via "..", with an allowed_base, must raise.
        # The point is: it raises via the relative_to() check, NOT the strict
        # `..` check (which is gated off when allowed_base is set).
        with pytest.raises(PathTraversalError, match="escapes"):
            safe_path("../outside", allowed_base=base_dir)


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
        from backpropagate.security import SecurityWarning, check_torch_security

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
            warnings.warn("test", SecurityWarning, stacklevel=2)

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
        """safe_torch_load should prefer safetensors format over torch.load.

        Verifies the security-critical preference path: when a .safetensors
        file is passed, safetensors.torch.load_file is invoked and torch.load
        is NOT called (avoiding pickle deserialization of untrusted weights).
        """
        from backpropagate.security import safe_torch_load

        # Create a fake safetensors file (contents irrelevant — load_file is mocked)
        st_file = tmp_path / "model.safetensors"
        st_file.write_bytes(b"mock safetensors")

        mock_load_file = MagicMock(return_value={"weight": "tensor"})

        # Build a fake safetensors.torch module with load_file
        fake_safetensors_torch = MagicMock()
        fake_safetensors_torch.load_file = mock_load_file

        with patch.dict(
            "sys.modules",
            {
                "safetensors": MagicMock(),
                "safetensors.torch": fake_safetensors_torch,
            },
        ), patch("backpropagate.security.check_torch_security"), patch(
            "torch.load"
        ) as mock_torch_load:
            result = safe_torch_load(st_file)

        # safetensors path was taken: load_file called exactly once with the
        # safetensors path; torch.load NOT called.
        assert mock_load_file.call_count == 1, (
            f"Expected safetensors.torch.load_file to be called once, "
            f"got {mock_load_file.call_count}"
        )
        # load_file is called with str(path) — see security.py:208
        called_arg = mock_load_file.call_args[0][0]
        assert str(st_file) == called_arg, (
            f"safetensors.torch.load_file called with {called_arg!r}, "
            f"expected {str(st_file)!r}"
        )
        assert not mock_torch_load.called, (
            "torch.load must NOT be invoked when a .safetensors file is "
            "available — this is the CVE-mitigation contract"
        )
        assert result == {"weight": "tensor"}

    def test_safe_torch_load_falls_back_to_torch_load_with_weights_only(self, tmp_path):
        """safe_torch_load should call torch.load with weights_only=True on non-safetensors paths."""
        from backpropagate.security import safe_torch_load

        # A .pt/.bin file should fall through to torch.load
        pt_file = tmp_path / "model.pt"
        pt_file.write_bytes(b"mock pickle")

        mock_state = {"weight": "tensor"}

        with patch("backpropagate.security.check_torch_security"), patch(
            "torch.load", return_value=mock_state
        ) as mock_torch_load:
            result = safe_torch_load(pt_file)

        assert mock_torch_load.call_count == 1, (
            f"Expected torch.load to be called once, got {mock_torch_load.call_count}"
        )
        # Verify weights_only=True was resolved as the kwarg passed to torch.load
        call_kwargs = mock_torch_load.call_args.kwargs
        assert call_kwargs.get("weights_only") is True, (
            f"torch.load must be called with weights_only=True (got kwargs={call_kwargs})"
        )
        assert result == mock_state

    def test_safe_torch_load_with_weights_only(self, tmp_path):
        """safe_torch_load should pass weights_only to torch.load."""
        import torch

        from backpropagate.security import safe_torch_load

        # Create a simple tensor file
        pt_file = tmp_path / "weights.pt"
        torch.save({"weight": torch.tensor([1, 2, 3])}, pt_file)

        with patch("backpropagate.security.check_torch_security"):
            result = safe_torch_load(pt_file, weights_only=True)

        assert "weight" in result

    def test_safe_torch_load_warns_on_weights_only_false(self, tmp_path):
        """CLI-A-006: weights_only=False must emit a SecurityWarning.

        The wrapper's whole value is the weights_only=True safety floor.
        Flipping it off turns the load into unrestricted pickle
        deserialization (RCE surface), so the "safe_" name must not be
        silently subvertible — we warn loudly while still honoring the
        request (a trusted legacy checkpoint may genuinely need it).
        """
        from backpropagate.security import SecurityWarning, safe_torch_load

        pt_file = tmp_path / "model.pt"
        pt_file.write_bytes(b"mock pickle")

        mock_state = {"weight": "tensor"}

        with patch("backpropagate.security.check_torch_security"), patch(
            "torch.load", return_value=mock_state
        ) as mock_torch_load, pytest.warns(SecurityWarning, match="weights_only"):
            result = safe_torch_load(pt_file, weights_only=False)

        # Request is still honored (warning, not refusal).
        assert result == mock_state
        call_kwargs = mock_torch_load.call_args.kwargs
        assert call_kwargs.get("weights_only") is False, (
            f"weights_only=False must still be forwarded (got {call_kwargs})"
        )

    def test_safe_torch_load_no_warning_on_default(self, tmp_path):
        """CLI-A-006: the safe default (weights_only=True) emits NO warning."""
        import warnings

        from backpropagate.security import SecurityWarning, safe_torch_load

        pt_file = tmp_path / "model.pt"
        pt_file.write_bytes(b"mock pickle")

        with patch("backpropagate.security.check_torch_security"), patch(
            "torch.load", return_value={"weight": "t"}
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("error", SecurityWarning)
                # Must not raise (no SecurityWarning on the safe path).
                safe_torch_load(pt_file)


class TestAuditLog:
    """Tests for audit_log function."""

    def test_audit_log_success(self, caplog):
        """audit_log should log successful operations."""
        import logging

        from backpropagate.security import audit_log

        caplog.set_level(logging.INFO, logger="backpropagate.security.audit")

        audit_log("model_load", path="/models/test.pt", success=True)

        assert any("AUDIT" in record.message for record in caplog.records)
        assert any("model_load" in record.message for record in caplog.records)

    def test_audit_log_failure(self, caplog):
        """audit_log should log failed operations at WARNING level."""
        import logging

        from backpropagate.security import audit_log

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
            PathTraversalError,
            SecurityWarning,
            check_torch_security,
            safe_path,
        )

        assert callable(safe_path)
        assert callable(check_torch_security)
        assert issubclass(SecurityWarning, Warning)
        assert issubclass(PathTraversalError, ValueError)
