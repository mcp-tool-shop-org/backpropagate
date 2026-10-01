"""Coverage tests for ``backpropagate.feature_flags``.

Detection matrix, ``_has_module`` edge cases, the public helpers, the
``check_unsloth_runtime`` probe, and the GPU / system info reporters.

Mocked (and nothing else): ``feature_flags._has_module`` (which optional
packages "exist"), ``importlib.util.find_spec`` for the lookup-failure test, a
fake ``unsloth`` module / import hook (importing real Unsloth is GPU-only),
``torch.cuda`` (GPU boundary: reported values are asserted, no device is
touched) and ``psutil`` absence.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import logging
import sys
import types

import pytest

from backpropagate import feature_flags as ff

LOGGER = "backpropagate.feature_flags"


@pytest.fixture
def pristine_features(monkeypatch):
    snapshot = dict(ff.FEATURES)
    monkeypatch.delenv("BACKPROPAGATE_DEFER_FEATURE_DETECTION", raising=False)
    monkeypatch.delenv("BACKPROPAGATE_DEFER_FEATURE_QUIET", raising=False)
    yield
    ff.FEATURES.clear()
    ff.FEATURES.update(snapshot)


def _detect(monkeypatch, *installed: str) -> dict[str, bool]:
    monkeypatch.setattr(ff, "_has_module", lambda name: name in installed)
    for key in ff.FEATURES:
        ff.FEATURES[key] = False
    ff._detect_features()
    return dict(ff.FEATURES)


# =============================================================================
# _has_module
# =============================================================================


class TestHasModule:
    def test_real_lookups(self):
        assert ff._has_module("json") is True
        assert ff._has_module("definitely_not_a_real_module_xyz") is False

    def test_half_installed_package_counts_as_missing(self, monkeypatch):
        """A module whose ``__spec__`` is None makes ``find_spec`` raise ValueError."""
        broken = types.ModuleType("half_installed_pkg")
        broken.__spec__ = None
        monkeypatch.setitem(sys.modules, "half_installed_pkg", broken)
        assert ff._has_module("half_installed_pkg") is False

    @pytest.mark.parametrize("exc", [ImportError("x"), ModuleNotFoundError("y"), ValueError("z")])
    def test_pathological_finder_errors_are_treated_as_absent(self, monkeypatch, exc):
        def boom(name):
            raise exc

        monkeypatch.setattr(importlib.util, "find_spec", boom)
        assert ff._has_module("anything") is False


# =============================================================================
# _detect_features matrix
# =============================================================================


class TestDetectFeatures:
    def test_nothing_installed(self, pristine_features, monkeypatch):
        result = _detect(monkeypatch)
        assert not any(result.values())

    def test_everything_installed(self, pristine_features, monkeypatch):
        result = _detect(
            monkeypatch, "unsloth", "reflex", "pydantic", "pydantic_settings", "llama_cpp", "psutil",
            "wandb", "flash_attn", "triton", "torchao", "mlx_lm", "tensorboard", "mlflow",
        )
        assert all(result.values()), [k for k, v in result.items() if not v]

    def test_validation_needs_both_pydantic_packages(self, pristine_features, monkeypatch):
        assert _detect(monkeypatch, "pydantic")["validation"] is False
        assert _detect(monkeypatch, "pydantic_settings")["validation"] is False
        assert _detect(monkeypatch, "pydantic", "pydantic_settings")["validation"] is True

    def test_monitoring_is_the_and_of_psutil_and_wandb(self, pristine_features, monkeypatch):
        only_psutil = _detect(monkeypatch, "psutil")
        assert only_psutil["psutil"] is True and only_psutil["monitoring"] is False and only_psutil["wandb"] is False
        only_wandb = _detect(monkeypatch, "wandb")
        assert only_wandb["wandb"] is True and only_wandb["psutil"] is False and only_wandb["monitoring"] is False
        both = _detect(monkeypatch, "psutil", "wandb")
        assert both["monitoring"] is True

    @pytest.mark.parametrize("pkg", ["tensorboard", "tensorboardX"])
    def test_tensorboard_accepts_either_distribution(self, pristine_features, monkeypatch, pkg):
        assert _detect(monkeypatch, pkg)["tensorboard"] is True

    @pytest.mark.parametrize(
        ("module", "flag"),
        [
            ("unsloth", "unsloth"), ("reflex", "ui"), ("llama_cpp", "export"),
            ("flash_attn", "flash_attention"), ("triton", "triton"), ("torchao", "fp8"),
            ("mlx_lm", "mlx"), ("mlflow", "mlflow"),
        ],
    )
    def test_single_module_flags(self, pristine_features, monkeypatch, module, flag):
        result = _detect(monkeypatch, module)
        assert result[flag] is True
        assert sum(result.values()) == 1

    def test_detection_logs_a_summary_at_debug(self, pristine_features, monkeypatch, caplog):
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            _detect(monkeypatch, "triton", "reflex")
        assert any("Feature detection complete: 2 available" in r.getMessage() for r in caplog.records)


# =============================================================================
# Public helpers
# =============================================================================


class TestPublicHelpers:
    def test_check_feature_and_unknown_names(self, pristine_features):
        ff.FEATURES["ui"] = True
        ff.FEATURES["triton"] = False
        assert ff.check_feature("ui") is True
        assert ff.check_feature("triton") is False
        assert ff.check_feature("not-a-feature") is False

    def test_install_hints(self):
        assert "backpropagate[unsloth]" in ff.get_install_hint("unsloth")
        assert ff.get_install_hint("custom") == "pip install backpropagate[custom]"
        assert "psutil" in ff.INSTALL_HINTS and "psutil" in ff.FEATURE_DESCRIPTIONS

    def test_available_and_missing_listings_partition_the_features(self, pristine_features):
        for key in ff.FEATURES:
            ff.FEATURES[key] = False
        ff.FEATURES["ui"] = True
        ff.FEATURES["triton"] = True
        available = ff.list_available_features()
        missing = ff.list_missing_features()
        assert set(available) == {"ui", "triton"}
        assert available["ui"] == ff.FEATURE_DESCRIPTIONS["ui"]
        assert set(available) | set(missing) == set(ff.FEATURES)
        assert not set(available) & set(missing)
        assert missing["unsloth"] == ff.INSTALL_HINTS["unsloth"]

    def test_every_feature_has_a_hint_and_description(self):
        for name in ff.FEATURES:
            assert ff.INSTALL_HINTS.get(name), name
            assert ff.FEATURE_DESCRIPTIONS.get(name), name

    def test_refresh_resets_stale_flags_first(self, pristine_features, monkeypatch):
        monkeypatch.setattr(ff, "_has_module", lambda name: name == "reflex")
        ff.FEATURES["triton"] = True  # stale: package was uninstalled meanwhile
        result = ff.refresh_features()
        assert result["triton"] is False and result["ui"] is True


class TestRequireAndEnsure:
    def test_require_feature_passes_arguments_and_preserves_metadata(self, pristine_features):
        ff.FEATURES["ui"] = True

        @ff.require_feature("ui")
        def launch(a, b=2):
            """doc kept"""
            return a + b

        assert launch(1, b=5) == 6
        assert launch.__name__ == "launch" and launch.__doc__ == "doc kept"

    def test_require_feature_error_names_feature_description_and_hint(self, pristine_features):
        ff.FEATURES["fp8"] = False

        @ff.require_feature("fp8")
        def go():
            return "ran"

        with pytest.raises(ImportError) as exc:
            go()
        msg = str(exc.value)
        assert "Feature 'fp8' (FP8 compute path" in msg
        assert "is required but not installed" in msg
        assert "pip install 'backpropagate[fp8]'" in msg

    def test_unknown_feature_gets_the_generic_hint_without_description(self, pristine_features):
        @ff.require_feature("quantum")
        def go():
            return "ran"

        with pytest.raises(ImportError) as exc:
            go()
        assert "Feature 'quantum' is required" in str(exc.value)
        assert "pip install backpropagate[quantum]" in str(exc.value)
        assert "()" not in str(exc.value)

    def test_ensure_feature(self, pristine_features):
        ff.FEATURES["ui"] = True
        ff.ensure_feature("ui")
        ff.FEATURES["ui"] = False
        with pytest.raises(ff.FeatureNotAvailable) as exc:
            ff.ensure_feature("ui")
        assert isinstance(exc.value, ImportError)
        assert exc.value.feature == "ui"
        assert exc.value.install_hint == ff.INSTALL_HINTS["ui"]
        assert "Install with:" in str(exc.value)

    def test_feature_not_available_custom_message(self):
        err = ff.FeatureNotAvailable("mlx", "custom words")
        assert str(err) == "custom words" and err.feature == "mlx"


# =============================================================================
# check_unsloth_runtime
# =============================================================================


class _RaiseOnImport(importlib.abc.MetaPathFinder):
    """Meta-path hook making ``import unsloth`` raise RuntimeError (py3.14-style failure)."""

    def find_spec(self, name, path=None, target=None):
        if name == "unsloth":
            raise RuntimeError("torch.compile is not supported on this Python")
        return None


class TestCheckUnslothRuntime:
    def test_not_installed(self, pristine_features):
        ff.FEATURES["unsloth"] = False
        assert ff.check_unsloth_runtime() == (False, "unsloth package not installed")

    def test_importable_unsloth_is_ok(self, pristine_features, monkeypatch):
        ff.FEATURES["unsloth"] = True
        monkeypatch.setitem(sys.modules, "unsloth", types.ModuleType("unsloth"))
        assert ff.check_unsloth_runtime() == (True, None)

    def test_runtime_error_is_reported_as_python_incompatibility(self, pristine_features, monkeypatch):
        ff.FEATURES["unsloth"] = True
        monkeypatch.delitem(sys.modules, "unsloth", raising=False)
        hook = _RaiseOnImport()
        monkeypatch.setattr(sys, "meta_path", [hook, *sys.meta_path])
        ok, message = ff.check_unsloth_runtime()
        assert ok is False
        assert message == "Python 3.14+ incompatibility: torch.compile is not supported on this Python"


# =============================================================================
# get_gpu_info / get_system_info
# =============================================================================


class TestGpuInfo:
    def test_no_cuda(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert ff.get_gpu_info() == {"available": False}

    def test_reports_device_properties(self, monkeypatch):
        """Mocks: ``torch.cuda`` (GPU boundary) with fixed, asserted values."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: f"Fake GPU {i}")
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(total_memory=32 * 2**30))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i: 111)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda i: 222)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (12, 0))
        assert ff.get_gpu_info() == {
            "available": True,
            "device_count": 2,
            "current_device": 1,
            "device_name": "Fake GPU 0",
            "memory_total": 32 * 2**30,
            "memory_allocated": 111,
            "memory_reserved": 222,
            "compute_capability": (12, 0),
        }

    def test_driver_errors_degrade_to_unavailable(self, monkeypatch):
        import torch

        def boom():
            raise RuntimeError("driver mismatch")

        monkeypatch.setattr(torch.cuda, "is_available", boom)
        assert ff.get_gpu_info() == {"available": False}

    def test_torch_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert ff.get_gpu_info() == {"available": False}


class TestSystemInfo:
    def test_shape_and_memory_when_psutil_flag_is_set(self, pristine_features, monkeypatch):
        pytest.importorskip("psutil")
        monkeypatch.setattr(ff, "get_gpu_info", lambda: {"available": False})
        ff.FEATURES["psutil"] = True
        info = ff.get_system_info()
        assert info["python_version"] == sys.version
        assert info["gpu"] == {"available": False}
        assert info["features"] == dict(ff.FEATURES)
        assert info["memory"]["total"] > 0 and 0 <= info["memory"]["percent"] <= 100
        assert info["memory"]["available"] <= info["memory"]["total"]

    def test_monitoring_umbrella_flag_also_enables_memory(self, pristine_features, monkeypatch):
        pytest.importorskip("psutil")
        monkeypatch.setattr(ff, "get_gpu_info", lambda: {"available": False})
        ff.FEATURES["psutil"] = False
        ff.FEATURES["monitoring"] = True
        assert "memory" in ff.get_system_info()

    def test_no_memory_section_without_the_flags(self, pristine_features, monkeypatch):
        monkeypatch.setattr(ff, "get_gpu_info", lambda: {"available": False})
        ff.FEATURES["psutil"] = False
        ff.FEATURES["monitoring"] = False
        assert "memory" not in ff.get_system_info()

    def test_flag_set_but_psutil_import_fails(self, pristine_features, monkeypatch):
        monkeypatch.setattr(ff, "get_gpu_info", lambda: {"available": False})
        ff.FEATURES["psutil"] = True
        monkeypatch.setitem(sys.modules, "psutil", None)
        info = ff.get_system_info()
        assert "memory" not in info and info["features"]["psutil"] is True
