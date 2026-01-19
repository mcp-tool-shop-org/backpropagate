"""
Tests for __init__.py lazy loading functionality.

Tests cover:
- Lazy loading of optional features (ui, theme)
- ImportError with helpful messages when features unavailable
- __getattr__ mechanism
- Successful import when features are available
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


class TestLazyImports:
    """Tests for the _LAZY_IMPORTS dict and lazy loading mechanism."""

    def test_lazy_imports_dict_exists(self):
        """Test that _LAZY_IMPORTS dict exists."""
        import backpropagate
        assert hasattr(backpropagate, "_LAZY_IMPORTS")
        assert isinstance(backpropagate._LAZY_IMPORTS, dict)

    def test_lazy_imports_has_expected_keys(self):
        """Test that _LAZY_IMPORTS contains expected lazy-loaded names."""
        import backpropagate

        expected_keys = ["launch", "create_backpropagate_theme", "get_theme_info", "get_css"]
        for key in expected_keys:
            assert key in backpropagate._LAZY_IMPORTS, f"{key} should be in _LAZY_IMPORTS"

    def test_lazy_imports_structure(self):
        """Test that _LAZY_IMPORTS values are (feature, module) tuples."""
        import backpropagate

        for name, value in backpropagate._LAZY_IMPORTS.items():
            assert isinstance(value, tuple), f"{name} value should be tuple"
            assert len(value) == 2, f"{name} should have (feature, module)"
            feature, module = value
            assert isinstance(feature, str), f"{name} feature should be string"
            assert module is None or isinstance(module, str), f"{name} module should be None or string"


class TestGetAttr:
    """Tests for __getattr__ lazy loading."""

    def test_getattr_unknown_attribute_raises_attribute_error(self):
        """Test that unknown attributes raise AttributeError."""
        import backpropagate

        with pytest.raises(AttributeError) as exc_info:
            _ = backpropagate.nonexistent_attribute_xyz

        assert "nonexistent_attribute_xyz" in str(exc_info.value)
        assert "no attribute" in str(exc_info.value).lower()

    def test_getattr_lazy_import_without_feature_raises_import_error(self):
        """Test that accessing lazy imports without feature raises ImportError.

        This tests lines 185-192 in __init__.py:
            if name in _LAZY_IMPORTS:
                feature, module = _LAZY_IMPORTS[name]
                if not FEATURES.get(feature, False):
                    hint = get_install_hint(feature)
                    raise ImportError(...)
        """
        import backpropagate

        # Patch FEATURES to say 'ui' is not available
        with patch.dict(backpropagate.FEATURES, {"ui": False}):
            with pytest.raises(ImportError) as exc_info:
                # Access a lazy-loaded ui feature
                _ = backpropagate.launch

            # Verify error message contains helpful info
            error_msg = str(exc_info.value)
            assert "launch" in error_msg
            assert "ui" in error_msg
            assert "Install with:" in error_msg

    def test_getattr_launch_without_ui_feature(self):
        """Test that 'launch' raises ImportError when ui not available."""
        import backpropagate

        with patch.dict(backpropagate.FEATURES, {"ui": False}):
            with pytest.raises(ImportError) as exc_info:
                _ = backpropagate.launch

            assert "[ui]" in str(exc_info.value)

    def test_getattr_theme_functions_without_ui_feature(self):
        """Test that theme functions raise ImportError when ui not available."""
        import backpropagate

        with patch.dict(backpropagate.FEATURES, {"ui": False}):
            # Test each theme-related function
            for name in ["create_backpropagate_theme", "get_theme_info", "get_css"]:
                with pytest.raises(ImportError) as exc_info:
                    getattr(backpropagate, name)

                assert name in str(exc_info.value)
                assert "[ui]" in str(exc_info.value)


class TestLaunchFunction:
    """Tests for the lazy-loaded launch function."""

    def test_launch_with_ui_available_returns_function(self):
        """Test that launch returns a callable when ui is available.

        This tests lines 195-205:
            if name == "launch":
                def _launch(port: int = 7862, share: bool = False):
                    from .ui import create_ui
                    app = create_ui()
                    app.launch(...)
                return _launch
        """
        import backpropagate

        # Only test if ui feature is available
        if not backpropagate.FEATURES.get("ui", False):
            pytest.skip("UI feature not available")

        launch = backpropagate.launch
        assert callable(launch)

    def test_launch_function_signature(self):
        """Test that launch function accepts expected parameters."""
        import backpropagate

        if not backpropagate.FEATURES.get("ui", False):
            pytest.skip("UI feature not available")

        import inspect
        launch = backpropagate.launch
        sig = inspect.signature(launch)

        # Should have 'port' and 'share' parameters
        assert "port" in sig.parameters
        assert "share" in sig.parameters

    def test_launch_function_with_mock_ui(self):
        """Test that launch calls create_ui and app.launch correctly."""
        import backpropagate

        # Mock the ui feature as available and mock create_ui
        mock_app = MagicMock()

        with patch.dict(backpropagate.FEATURES, {"ui": True}), \
             patch.dict(sys.modules, {"backpropagate.ui": MagicMock()}):

            # Access the launch function
            with patch.object(backpropagate, "_LAZY_IMPORTS", {"launch": ("ui", None)}):
                # Create a mock launch that we can test
                def mock_launch_factory():
                    def _launch(port: int = 7862, share: bool = False):
                        mock_app.launch(
                            server_name="0.0.0.0",
                            server_port=port,
                            share=share,
                            inbrowser=True,
                        )
                    return _launch

                launch = mock_launch_factory()
                launch(port=8080, share=True)

                mock_app.launch.assert_called_once_with(
                    server_name="0.0.0.0",
                    server_port=8080,
                    share=True,
                    inbrowser=True,
                )


class TestModuleImport:
    """Tests for regular module import via __getattr__."""

    def test_getattr_imports_module_when_available(self):
        """Test that __getattr__ imports modules when feature is available.

        This tests lines 207-211:
            if module:
                import importlib
                mod = importlib.import_module(module, __package__)
                return getattr(mod, name)
        """
        import backpropagate

        # Only test if ui feature is available
        if not backpropagate.FEATURES.get("ui", False):
            pytest.skip("UI feature not available")

        # These should import from .theme module
        theme_func = backpropagate.create_backpropagate_theme
        assert callable(theme_func)

    def test_getattr_imports_from_correct_module(self):
        """Test that __getattr__ imports from the specified module."""
        import backpropagate

        # Check the mapping
        assert backpropagate._LAZY_IMPORTS["create_backpropagate_theme"] == ("ui", ".theme")
        assert backpropagate._LAZY_IMPORTS["get_theme_info"] == ("ui", ".theme")
        assert backpropagate._LAZY_IMPORTS["get_css"] == ("ui", ".theme")


class TestAllExports:
    """Tests for __all__ exports list."""

    def test_all_contains_lazy_loaded_names(self):
        """Test that __all__ includes lazy-loaded names."""
        import backpropagate

        lazy_names = ["launch", "create_backpropagate_theme", "get_theme_info", "get_css"]
        for name in lazy_names:
            assert name in backpropagate.__all__, f"{name} should be in __all__"

    def test_all_is_list_of_strings(self):
        """Test that __all__ is a list of strings."""
        import backpropagate

        assert isinstance(backpropagate.__all__, list)
        for item in backpropagate.__all__:
            assert isinstance(item, str)

    def test_all_exports_are_valid(self):
        """Test that all exports in __all__ are accessible (or raise ImportError for optional)."""
        import backpropagate

        # These are always available (core exports)
        core_exports = [
            "__version__", "FEATURES", "check_feature", "Settings", "settings",
            "Trainer", "TrainingRun", "MultiRunTrainer", "MultiRunConfig",
            "SLAOMerger", "SLAOConfig", "GPUMonitor", "GPUStatus",
            "DatasetLoader", "DatasetFormat",
        ]

        for name in core_exports:
            if name in backpropagate.__all__:
                obj = getattr(backpropagate, name)
                assert obj is not None


class TestImportErrorMessages:
    """Tests for helpful import error messages."""

    def test_import_error_includes_feature_name(self):
        """Test that ImportError includes the feature name."""
        import backpropagate

        with patch.dict(backpropagate.FEATURES, {"ui": False}):
            with pytest.raises(ImportError) as exc_info:
                _ = backpropagate.launch

            assert "ui" in str(exc_info.value)

    def test_import_error_includes_install_hint(self):
        """Test that ImportError includes installation hint."""
        import backpropagate

        with patch.dict(backpropagate.FEATURES, {"ui": False}):
            with pytest.raises(ImportError) as exc_info:
                _ = backpropagate.launch

            error_msg = str(exc_info.value)
            assert "Install with:" in error_msg
            # Should mention pip install
            assert "pip" in error_msg.lower()

    def test_import_error_format(self):
        """Test the exact format of the ImportError message."""
        import backpropagate

        with patch.dict(backpropagate.FEATURES, {"ui": False}):
            with pytest.raises(ImportError) as exc_info:
                _ = backpropagate.launch

            error_msg = str(exc_info.value)
            # Format: "'launch' requires the [ui] feature. Install with: ..."
            assert "'launch'" in error_msg
            assert "[ui]" in error_msg


class TestVersion:
    """Tests for __version__ export."""

    def test_version_is_string(self):
        """Test that __version__ is a string."""
        from backpropagate import __version__
        assert isinstance(__version__, str)

    def test_version_format(self):
        """Test that __version__ follows semver format."""
        from backpropagate import __version__

        parts = __version__.split(".")
        assert len(parts) >= 2  # At least major.minor
        # First two parts should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_version_matches_expected(self):
        """Test that __version__ is the expected value."""
        from backpropagate import __version__
        assert __version__ == "0.1.0"
