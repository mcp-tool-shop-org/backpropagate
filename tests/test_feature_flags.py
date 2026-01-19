"""
Tests for feature flags module.

Tests cover:
- Feature detection for each optional dependency
- get_gpu_info() with/without CUDA
- get_system_info() returns expected keys
- require_feature decorator
- FeatureNotAvailable exception
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


# =============================================================================
# FEATURE DETECTION TESTS
# =============================================================================

class TestFeaturesDict:
    """Tests for FEATURES dict detection."""

    def test_features_dict_exists(self):
        """FEATURES dict should exist and contain expected keys."""
        from backpropagate.feature_flags import FEATURES

        assert isinstance(FEATURES, dict)
        expected_keys = [
            "unsloth", "ui", "validation", "export",
            "monitoring", "observability", "flash_attention", "triton"
        ]
        for key in expected_keys:
            assert key in FEATURES

    def test_features_values_are_bool(self):
        """All FEATURES values should be booleans."""
        from backpropagate.feature_flags import FEATURES

        for key, value in FEATURES.items():
            assert isinstance(value, bool), f"{key} should be bool, got {type(value)}"

    def test_check_feature_returns_bool(self):
        """check_feature should return boolean for any feature."""
        from backpropagate.feature_flags import check_feature

        assert isinstance(check_feature("unsloth"), bool)
        assert isinstance(check_feature("ui"), bool)
        assert isinstance(check_feature("unknown_feature"), bool)

    def test_check_feature_unknown_returns_false(self):
        """check_feature should return False for unknown features."""
        from backpropagate.feature_flags import check_feature

        assert check_feature("nonexistent_feature") is False
        assert check_feature("") is False


class TestInstallHints:
    """Tests for installation hints."""

    def test_get_install_hint_known_feature(self):
        """get_install_hint should return hint for known features."""
        from backpropagate.feature_flags import get_install_hint

        hint = get_install_hint("unsloth")
        assert "pip install" in hint
        assert "unsloth" in hint

    def test_get_install_hint_unknown_feature(self):
        """get_install_hint should return default hint for unknown features."""
        from backpropagate.feature_flags import get_install_hint

        hint = get_install_hint("unknown_feature")
        assert "pip install" in hint
        assert "unknown_feature" in hint


class TestListFeatures:
    """Tests for feature listing functions."""

    def test_list_available_features_returns_dict(self):
        """list_available_features should return dict."""
        from backpropagate.feature_flags import list_available_features

        result = list_available_features()
        assert isinstance(result, dict)

    def test_list_available_features_has_descriptions(self):
        """list_available_features values should be descriptions."""
        from backpropagate.feature_flags import list_available_features

        result = list_available_features()
        for feature, desc in result.items():
            assert isinstance(desc, str)

    def test_list_missing_features_returns_dict(self):
        """list_missing_features should return dict."""
        from backpropagate.feature_flags import list_missing_features

        result = list_missing_features()
        assert isinstance(result, dict)

    def test_list_missing_features_has_hints(self):
        """list_missing_features values should be install hints."""
        from backpropagate.feature_flags import list_missing_features

        result = list_missing_features()
        for feature, hint in result.items():
            assert isinstance(hint, str)
            # Should contain pip install command
            assert "pip" in hint or hint == ""


class TestRequireFeature:
    """Tests for require_feature decorator."""

    def test_require_feature_with_available_feature(self):
        """require_feature should allow execution when feature is available."""
        from backpropagate.feature_flags import require_feature, FEATURES

        # Find an available feature
        available_feature = None
        for name, available in FEATURES.items():
            if available:
                available_feature = name
                break

        if available_feature is None:
            pytest.skip("No features available to test")

        @require_feature(available_feature)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_require_feature_with_missing_feature(self):
        """require_feature should raise ImportError when feature is missing."""
        from backpropagate.feature_flags import require_feature, FEATURES

        # Find a missing feature
        missing_feature = None
        for name, available in FEATURES.items():
            if not available:
                missing_feature = name
                break

        if missing_feature is None:
            pytest.skip("All features are available")

        @require_feature(missing_feature)
        def test_func():
            return "success"

        with pytest.raises(ImportError) as exc_info:
            test_func()

        assert missing_feature in str(exc_info.value)
        assert "Install with:" in str(exc_info.value)

    def test_require_feature_preserves_function_name(self):
        """require_feature should preserve function metadata."""
        from backpropagate.feature_flags import require_feature

        @require_feature("ui")
        def my_special_function():
            """My docstring."""
            pass

        assert my_special_function.__name__ == "my_special_function"
        assert my_special_function.__doc__ == "My docstring."


class TestFeatureNotAvailable:
    """Tests for FeatureNotAvailable exception."""

    def test_feature_not_available_creation(self):
        """FeatureNotAvailable should be creatable with feature name."""
        from backpropagate.feature_flags import FeatureNotAvailable

        exc = FeatureNotAvailable("unsloth")
        assert exc.feature == "unsloth"
        assert isinstance(exc.install_hint, str)

    def test_feature_not_available_message(self):
        """FeatureNotAvailable should have informative message."""
        from backpropagate.feature_flags import FeatureNotAvailable

        exc = FeatureNotAvailable("unsloth")
        assert "unsloth" in str(exc)
        assert "Install with:" in str(exc)

    def test_feature_not_available_custom_message(self):
        """FeatureNotAvailable should accept custom message."""
        from backpropagate.feature_flags import FeatureNotAvailable

        exc = FeatureNotAvailable("unsloth", "Custom message")
        assert str(exc) == "Custom message"

    def test_feature_not_available_is_import_error(self):
        """FeatureNotAvailable should be subclass of ImportError."""
        from backpropagate.feature_flags import FeatureNotAvailable

        assert issubclass(FeatureNotAvailable, ImportError)

        exc = FeatureNotAvailable("test")
        assert isinstance(exc, ImportError)


class TestEnsureFeature:
    """Tests for ensure_feature function."""

    def test_ensure_feature_raises_for_missing(self):
        """ensure_feature should raise FeatureNotAvailable for missing feature."""
        from backpropagate.feature_flags import ensure_feature, FeatureNotAvailable, FEATURES

        # Find a missing feature
        missing_feature = None
        for name, available in FEATURES.items():
            if not available:
                missing_feature = name
                break

        if missing_feature is None:
            pytest.skip("All features are available")

        with pytest.raises(FeatureNotAvailable) as exc_info:
            ensure_feature(missing_feature)

        assert exc_info.value.feature == missing_feature

    def test_ensure_feature_passes_for_available(self):
        """ensure_feature should not raise for available feature."""
        from backpropagate.feature_flags import ensure_feature, FEATURES

        # Find an available feature
        available_feature = None
        for name, available in FEATURES.items():
            if available:
                available_feature = name
                break

        if available_feature is None:
            pytest.skip("No features available to test")

        # Should not raise
        ensure_feature(available_feature)


# =============================================================================
# GPU INFO TESTS
# =============================================================================

class TestGetGpuInfo:
    """Tests for get_gpu_info function."""

    def test_get_gpu_info_returns_dict(self):
        """get_gpu_info should return a dict."""
        from backpropagate.feature_flags import get_gpu_info

        result = get_gpu_info()
        assert isinstance(result, dict)

    def test_get_gpu_info_has_available_key(self):
        """get_gpu_info should always have 'available' key."""
        from backpropagate.feature_flags import get_gpu_info

        result = get_gpu_info()
        assert "available" in result
        assert isinstance(result["available"], bool)

    def test_get_gpu_info_without_cuda(self):
        """get_gpu_info should return available=False when CUDA not available."""
        with patch.dict(sys.modules, {"torch": None}):
            # Force reimport
            from backpropagate import feature_flags
            result = feature_flags.get_gpu_info()
            assert result["available"] is False

    def test_get_gpu_info_with_mock_cuda(self):
        """get_gpu_info should return GPU details when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 4 * (1024**3)
        mock_torch.cuda.memory_reserved.return_value = 6 * (1024**3)
        mock_torch.cuda.get_device_capability.return_value = (8, 0)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from backpropagate import feature_flags
            # Call directly to use mocked module
            import importlib
            importlib.reload(feature_flags)
            result = feature_flags.get_gpu_info()

            assert result["available"] is True
            assert result["device_name"] == "Test GPU"
            assert result["device_count"] == 1

    def test_get_gpu_info_handles_exception(self):
        """get_gpu_info should return available=False on exception."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("Test error")

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from backpropagate import feature_flags
            result = feature_flags.get_gpu_info()
            assert result["available"] is False


# =============================================================================
# SYSTEM INFO TESTS
# =============================================================================

class TestGetSystemInfo:
    """Tests for get_system_info function."""

    def test_get_system_info_returns_dict(self):
        """get_system_info should return a dict."""
        from backpropagate.feature_flags import get_system_info

        result = get_system_info()
        assert isinstance(result, dict)

    def test_get_system_info_has_expected_keys(self):
        """get_system_info should have expected keys."""
        from backpropagate.feature_flags import get_system_info

        result = get_system_info()

        # Required keys
        assert "python_version" in result
        assert "platform" in result
        assert "processor" in result
        assert "features" in result
        assert "gpu" in result

    def test_get_system_info_python_version(self):
        """get_system_info python_version should be string."""
        from backpropagate.feature_flags import get_system_info

        result = get_system_info()
        assert isinstance(result["python_version"], str)
        assert "." in result["python_version"]  # e.g. "3.10.0"

    def test_get_system_info_platform(self):
        """get_system_info platform should be string."""
        from backpropagate.feature_flags import get_system_info

        result = get_system_info()
        assert isinstance(result["platform"], str)

    def test_get_system_info_features_is_dict(self):
        """get_system_info features should be dict of booleans."""
        from backpropagate.feature_flags import get_system_info

        result = get_system_info()
        assert isinstance(result["features"], dict)
        for key, value in result["features"].items():
            assert isinstance(value, bool)

    def test_get_system_info_gpu_is_dict(self):
        """get_system_info gpu should be dict."""
        from backpropagate.feature_flags import get_system_info

        result = get_system_info()
        assert isinstance(result["gpu"], dict)
        assert "available" in result["gpu"]

    def test_get_system_info_memory_when_monitoring(self):
        """get_system_info should include memory when monitoring available."""
        from backpropagate.feature_flags import get_system_info, FEATURES

        result = get_system_info()

        if FEATURES.get("monitoring"):
            assert "memory" in result
            assert "total" in result["memory"]
            assert "available" in result["memory"]
            assert "percent" in result["memory"]


# =============================================================================
# FEATURE DETECTION INTERNALS
# =============================================================================

class TestFeatureDetection:
    """Tests for _detect_features internals."""

    def test_detect_features_unsloth_import_error(self):
        """_detect_features should handle unsloth ImportError."""
        from backpropagate.feature_flags import _detect_features, FEATURES

        # Just verify it doesn't crash
        _detect_features()
        assert "unsloth" in FEATURES

    def test_detect_features_runtime_error(self):
        """_detect_features should handle RuntimeError (Python 3.14+)."""
        # RuntimeError is caught for torch.compile issues
        from backpropagate.feature_flags import _detect_features, FEATURES

        # Just verify it runs without crashing
        _detect_features()
        assert isinstance(FEATURES["unsloth"], bool)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """__all__ should contain expected exports."""
        from backpropagate import feature_flags

        expected = [
            "FEATURES",
            "check_feature",
            "require_feature",
            "get_install_hint",
            "list_available_features",
            "list_missing_features",
            "FeatureNotAvailable",
        ]

        for name in expected:
            assert name in feature_flags.__all__

    def test_imports_from_package(self):
        """Should be importable from backpropagate package."""
        from backpropagate import (
            FEATURES,
            check_feature,
            require_feature,
            get_install_hint,
            list_available_features,
            list_missing_features,
            FeatureNotAvailable,
            get_gpu_info,
            get_system_info,
        )

        assert FEATURES is not None
        assert callable(check_feature)
        assert callable(require_feature)
        assert callable(get_install_hint)
        assert callable(list_available_features)
        assert callable(list_missing_features)
        assert issubclass(FeatureNotAvailable, Exception)
        assert callable(get_gpu_info)
        assert callable(get_system_info)


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================

class TestFeatureDescriptions:
    """Tests for feature descriptions."""

    def test_feature_descriptions_dict_exists(self):
        """FEATURE_DESCRIPTIONS should exist and be a dict."""
        from backpropagate.feature_flags import FEATURE_DESCRIPTIONS

        assert isinstance(FEATURE_DESCRIPTIONS, dict)

    def test_feature_descriptions_all_features_have_description(self):
        """All features should have descriptions."""
        from backpropagate.feature_flags import FEATURES, FEATURE_DESCRIPTIONS

        for feature in FEATURES:
            assert feature in FEATURE_DESCRIPTIONS
            assert isinstance(FEATURE_DESCRIPTIONS[feature], str)
            assert len(FEATURE_DESCRIPTIONS[feature]) > 0

    def test_install_hints_all_features_have_hints(self):
        """All features should have install hints."""
        from backpropagate.feature_flags import FEATURES, INSTALL_HINTS

        for feature in FEATURES:
            assert feature in INSTALL_HINTS
            assert "pip install" in INSTALL_HINTS[feature]


class TestGetSystemInfoWithMonitoring:
    """Tests for get_system_info when monitoring is available."""

    def test_get_system_info_memory_with_mock_psutil(self):
        """get_system_info should include memory info when psutil available.

        This tests lines 339-348.
        """
        from backpropagate.feature_flags import get_system_info, FEATURES

        # Mock psutil memory info
        mock_mem = MagicMock()
        mock_mem.total = 32 * (1024**3)  # 32 GB
        mock_mem.available = 16 * (1024**3)  # 16 GB
        mock_mem.percent = 50.0

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem

        # Patch FEATURES and psutil
        with patch.dict(FEATURES, {"monitoring": True}), \
             patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_system_info()

            assert "memory" in result
            assert result["memory"]["total"] == 32 * (1024**3)
            assert result["memory"]["available"] == 16 * (1024**3)
            assert result["memory"]["percent"] == 50.0

    def test_get_system_info_memory_handles_psutil_exception(self):
        """get_system_info should handle psutil exception gracefully.

        This tests lines 347-348.
        """
        from backpropagate.feature_flags import get_system_info, FEATURES

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.side_effect = RuntimeError("psutil error")

        with patch.dict(FEATURES, {"monitoring": True}), \
             patch.dict(sys.modules, {"psutil": mock_psutil}):
            # Should not raise, just skip memory info
            result = get_system_info()
            # Memory key may or may not be present depending on exception handling
            assert isinstance(result, dict)


class TestGetGpuInfoEdgeCases:
    """Additional edge case tests for get_gpu_info."""

    def test_get_gpu_info_cuda_not_available(self):
        """get_gpu_info should return available=False when CUDA not available.

        This tests lines 303-316.
        """
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from backpropagate import feature_flags
            result = feature_flags.get_gpu_info()
            assert result["available"] is False

    def test_get_gpu_info_full_details(self):
        """get_gpu_info should return all details when CUDA available.

        This tests lines 304-313.
        """
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 5080"

        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)
        mock_torch.cuda.get_device_properties.return_value = mock_props

        mock_torch.cuda.memory_allocated.return_value = 4 * (1024**3)
        mock_torch.cuda.memory_reserved.return_value = 6 * (1024**3)
        mock_torch.cuda.get_device_capability.return_value = (9, 0)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from backpropagate import feature_flags
            result = feature_flags.get_gpu_info()

            assert result["available"] is True
            assert result["device_count"] == 2
            assert result["current_device"] == 0
            assert result["device_name"] == "NVIDIA RTX 5080"
            assert result["memory_total"] == 16 * (1024**3)
            assert result["memory_allocated"] == 4 * (1024**3)
            assert result["memory_reserved"] == 6 * (1024**3)
            assert result["compute_capability"] == (9, 0)


class TestRequireFeatureEdgeCases:
    """Additional edge case tests for require_feature."""

    def test_require_feature_with_args_and_kwargs(self):
        """require_feature should pass args and kwargs correctly."""
        from backpropagate.feature_flags import require_feature, FEATURES

        # Find an available feature
        available_feature = None
        for name, available in FEATURES.items():
            if available:
                available_feature = name
                break

        if available_feature is None:
            pytest.skip("No features available to test")

        @require_feature(available_feature)
        def test_func(a, b, c=None):
            return (a, b, c)

        result = test_func(1, 2, c="test")
        assert result == (1, 2, "test")

    def test_require_feature_error_message_format(self):
        """require_feature should have properly formatted error message."""
        from backpropagate.feature_flags import require_feature, FEATURES

        # Find a missing feature
        missing_feature = None
        for name, available in FEATURES.items():
            if not available:
                missing_feature = name
                break

        if missing_feature is None:
            pytest.skip("All features are available")

        @require_feature(missing_feature)
        def test_func():
            pass

        with pytest.raises(ImportError) as exc_info:
            test_func()

        error_msg = str(exc_info.value)
        # Check format: "Feature 'X' is required but not installed. Install with: ..."
        assert f"'{missing_feature}'" in error_msg
        assert "required" in error_msg.lower()
        assert "not installed" in error_msg.lower()
        assert "Install with:" in error_msg
