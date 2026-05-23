"""
Extended UI tests for comprehensive coverage of the LEGACY Gradio UI module.

NOTE on the v1.1.0 transition: the Gradio UI module was deprecated in v1.1.0
and the canonical Web UI is now Reflex (``backpropagate.ui_app``). The legacy
module ``backpropagate.ui_gradio_legacy`` is preserved only as reference for
the framework-agnostic helpers; many of the original symbols this file
exercised have moved or been removed.

Stage A health pass (audit finding TESTS-A-001):
    The previous version of this file wrapped every test body in
    ``except AttributeError: pass`` — which converted ANY missing symbol into
    a silent green. After the v1.1.0 deprecation that wiped large swathes of
    the legacy surface, the entire file degraded to zero real coverage while
    still reporting ~70 passing tests.

The fix is two-layered:
    1. ``pytest.importorskip`` at module top — if gradio is not installed,
       the whole file SKIPS (not silently passes).
    2. Each test guards on ``hasattr(module, 'symbol')`` and calls
       ``pytest.skip(reason='symbol removed in v1.1.0')`` when the legacy
       symbol is gone. SKIP is visible in pytest output; the bare ``pass``
       was not.

Covers (when the legacy symbols are present):
- Theme and styling
- Training interface
- Dataset interface
- GPU monitoring display
- Export interface
- Callback system
- Security features
- Error handling
"""

from importlib import util as _importlib_util
from unittest.mock import MagicMock, patch

import pytest

# Skip the entire module when the legacy gradio UI module isn't even importable.
# (The module gracefully degrades when gradio itself is missing — see
# ``ui_gradio_legacy``'s _MissingGradio shim — but if the module file isn't
# present at all, every test below would just AttributeError.)
ui = pytest.importorskip(
    "backpropagate.ui_gradio_legacy",
    reason="gradio is required for legacy UI tests (install backpropagate[ui])",
)

# Tests below that DO need real gradio (because they ``patch("gradio.X")``)
# must be skipped when the gradio dependency itself is not installed. The
# legacy module imports cleanly without gradio thanks to the _MissingGradio
# shim, but ``mock.patch`` cannot resolve ``gradio.Blocks.launch`` etc. on
# such a system. The marker below is applied at the class level for tests
# that need the real gradio package.
#
# v1.1.0 migrated the [ui] extra from Gradio to Reflex; the legacy
# ``backpropagate.launch`` entrypoint still imports gradio internally, so
# these tests only run when gradio is installed as a separate dependency.
_GRADIO_AVAILABLE = _importlib_util.find_spec("gradio") is not None
_requires_gradio = pytest.mark.skipif(
    not _GRADIO_AVAILABLE,
    reason=(
        "gradio is not installed (v1.1.0 migrated [ui] extra to Reflex). "
        "Install gradio explicitly to exercise the legacy launch path."
    ),
)


def _require(symbol_name: str):
    """Resolve a legacy-UI symbol or skip the test with a clear reason.

    Replaces the old ``except AttributeError: pass`` anti-pattern with a
    visible SKIP that names the missing symbol. When ``ui_gradio_legacy`` is
    finally removed in v1.2, every dependent test will skip with a clear
    explanation rather than silently passing.
    """
    obj = getattr(ui, symbol_name, None)
    if obj is None:
        pytest.skip(
            f"backpropagate.ui_gradio_legacy.{symbol_name} not present "
            f"(removed in the v1.1.0 Reflex migration or never existed)"
        )
    return obj


# =============================================================================
# UI AVAILABILITY TESTS
# =============================================================================


class TestUIAvailability:
    """Tests for UI module availability."""

    def test_ui_module_imports(self):
        """UI module can be imported."""
        # If we got here, importorskip already proved this. Pin the docstring.
        assert ui is not None
        assert ui.__name__ == "backpropagate.ui_gradio_legacy"

    def test_launch_function_exists(self):
        """Launch function exists."""
        launch = _require("launch")
        assert callable(launch)

    def test_create_ui_function_exists(self):
        """create_ui function exists."""
        create_ui = _require("create_ui")
        assert callable(create_ui)


# =============================================================================
# THEME TESTS
# =============================================================================


class TestUITheme:
    """Tests for UI theme and styling."""

    def test_theme_creation(self):
        """Custom theme can be created."""
        create_theme = _require("create_theme")
        theme = create_theme()
        assert theme is not None

    def test_css_available(self):
        """Custom CSS is available."""
        custom_css = _require("CUSTOM_CSS")
        assert isinstance(custom_css, str)
        assert len(custom_css) > 0


# =============================================================================
# SECURITY TESTS
# =============================================================================


class TestUISecurity:
    """Tests for UI security features."""

    def test_rate_limiter_creation(self):
        """Rate limiter can be created."""
        rate_limiter_cls = _require("RateLimiter")
        limiter = rate_limiter_cls(max_calls=10, window_seconds=60)
        assert limiter is not None

    def test_rate_limiter_allows_calls(self):
        """Rate limiter allows calls within limit."""
        rate_limiter_cls = _require("RateLimiter")
        limiter = rate_limiter_cls(max_calls=5, window_seconds=60)
        for _ in range(5):
            assert limiter.check()

    def test_rate_limiter_blocks_over_limit(self):
        """Rate limiter blocks calls over limit."""
        rate_limiter_cls = _require("RateLimiter")
        limiter = rate_limiter_cls(max_calls=2, window_seconds=60)

        assert limiter.check()
        assert limiter.check()
        assert not limiter.check()  # Should be blocked

    def test_input_sanitization(self):
        """Input sanitization works."""
        sanitize_input = _require("sanitize_input")
        result = sanitize_input("<script>alert('xss')</script>")
        assert "<script>" not in result

    def test_path_validation(self):
        """Path validation prevents traversal."""
        validate_path = _require("validate_path")

        # Valid path should work
        assert validate_path("/home/user/data.jsonl")

        # Path traversal should be blocked
        assert not validate_path("../../../etc/passwd")


# =============================================================================
# TRAINING INTERFACE TESTS
# =============================================================================


class TestTrainingInterface:
    """Tests for training tab UI components."""

    def test_training_tab_creation(self):
        """Training tab can be created."""
        create_training_tab = _require("create_training_tab")

        with patch("gradio.Tab"):
            tab = create_training_tab()
            assert tab is not None

    def test_model_dropdown_options(self):
        """Model dropdown has options."""
        get_model_options = _require("get_model_options")
        options = get_model_options()
        assert isinstance(options, list)
        assert len(options) > 0

    def test_training_validation(self):
        """Training parameters are validated."""
        validate_training_params = _require("validate_training_params")

        # Valid params
        errors = validate_training_params(
            model="test-model",
            data_path="data.jsonl",
            steps=100,
            batch_size=2,
        )
        assert len(errors) == 0

        # Invalid params
        errors = validate_training_params(
            model="",
            data_path="",
            steps=-1,
            batch_size=0,
        )
        assert len(errors) > 0

    def test_start_training_handler(self):
        """Start training handler runs without raising."""
        handle_start_training = _require("handle_start_training")

        with patch("backpropagate.trainer.Trainer") as mock_trainer:
            mock_instance = MagicMock()
            mock_trainer.return_value = mock_instance

            # Just verify the handler is callable with the documented signature.
            # The handler may return a string status, a tuple, or None — the
            # invariant is "doesn't raise on a well-formed call".
            handle_start_training(
                model="test-model",
                data_path="data.jsonl",
                steps=10,
            )

    def test_stop_training_handler(self):
        """Stop training handler runs without raising."""
        handle_stop_training = _require("handle_stop_training")
        # Invariant: stop is always callable without args and never raises.
        handle_stop_training()


# =============================================================================
# DATASET INTERFACE TESTS
# =============================================================================


class TestDatasetInterface:
    """Tests for dataset tab UI components."""

    def test_dataset_tab_creation(self):
        """Dataset tab can be created."""
        create_dataset_tab = _require("create_dataset_tab")

        with patch("gradio.Tab"):
            tab = create_dataset_tab()
            assert tab is not None

    def test_dataset_preview(self, tmp_path):
        """Dataset preview shows samples."""
        import json

        preview_dataset = _require("preview_dataset")
        data_path = tmp_path / "test.jsonl"
        with open(data_path, "w") as f:
            f.write(json.dumps({"text": "Sample 1"}) + "\n")
            f.write(json.dumps({"text": "Sample 2"}) + "\n")

        preview = preview_dataset(str(data_path), num_samples=2)
        assert preview is not None

    def test_dataset_validation_display(self):
        """Validation results displayed correctly."""
        format_validation_results = _require("format_validation_results")

        results = {
            "is_valid": True,
            "num_samples": 100,
            "format": "chatml",
            "warnings": [],
            "errors": [],
        }

        formatted = format_validation_results(results)
        assert isinstance(formatted, str)

    def test_format_detection_display(self):
        """Detected format shown correctly."""
        format_detection_result = _require("format_detection_result")
        result = format_detection_result("sharegpt")
        assert "sharegpt" in result.lower()


# =============================================================================
# GPU MONITORING DISPLAY TESTS
# =============================================================================


class TestGPUMonitoringDisplay:
    """Tests for GPU monitoring dashboard."""

    def test_gpu_status_display(self):
        """GPU status displayed correctly."""
        from backpropagate.gpu_safety import GPUCondition, GPUStatus

        format_gpu_status = _require("format_gpu_status")

        # Use real GPUStatus object instead of MagicMock so the formatter
        # sees the documented attribute set.
        status = GPUStatus(
            temperature_c=65.0,
            vram_used_gb=8.0,
            vram_total_gb=16.0,
            vram_percent=50.0,
            power_watts=150.0,
            utilization_percent=75.0,
            condition=GPUCondition.SAFE,
        )

        display = format_gpu_status(status)
        assert display is not None
        assert isinstance(display, str)

    def test_temperature_color_coding(self):
        """Temperature displayed with color coding."""
        get_temperature_color = _require("get_temperature_color")

        color_safe = get_temperature_color(50)
        color_critical = get_temperature_color(95)

        # Different colors for different temps — the formatter MUST
        # distinguish safe from critical visually.
        assert color_safe != color_critical, (
            f"Safe ({color_safe!r}) and critical ({color_critical!r}) "
            f"temperatures must render with different colors."
        )

    def test_vram_bar_display(self):
        """VRAM usage shown with progress bar."""
        format_vram_display = _require("format_vram_display")

        display = format_vram_display(
            used_gb=8.0,
            total_gb=16.0,
            percent=50.0,
        )
        assert "8" in display or "50" in display

    def test_gpu_history_graph(self):
        """Temperature history can be graphed."""
        create_temperature_plot = _require("create_temperature_plot")
        history = [60, 62, 65, 63, 61, 64]
        plot = create_temperature_plot(history)
        assert plot is not None


# =============================================================================
# EXPORT INTERFACE TESTS
# =============================================================================


class TestExportInterface:
    """Tests for export tab UI components."""

    def test_export_tab_creation(self):
        """Export tab can be created."""
        create_export_tab = _require("create_export_tab")

        with patch("gradio.Tab"):
            tab = create_export_tab()
            assert tab is not None

    def test_format_options(self):
        """Export format options available."""
        export_formats = _require("EXPORT_FORMATS")
        assert "lora" in export_formats
        assert "merged" in export_formats
        assert "gguf" in export_formats

    def test_quantization_options(self):
        """Quantization options available for GGUF."""
        quant_options = _require("QUANTIZATION_OPTIONS")
        assert "q4_k_m" in quant_options
        assert "q8_0" in quant_options

    def test_export_handler(self, tmp_path):
        """Export handler runs without raising."""
        handle_export = _require("handle_export")

        with patch("backpropagate.export.export_lora") as mock_export:
            mock_result = MagicMock()
            mock_result.path = tmp_path / "export"
            mock_result.size_mb = 100.0
            mock_export.return_value = mock_result

            # Invariant: handler is callable end-to-end with the documented
            # signature. The exact return shape (status string, tuple, etc.)
            # is UI-implementation-specific — pin only "does not raise".
            handle_export(
                model_path=str(tmp_path / "model"),
                format="lora",
                output_dir=str(tmp_path / "output"),
            )


# =============================================================================
# CALLBACK SYSTEM TESTS
# =============================================================================


class TestUICallbacks:
    """Tests for UI callback system."""

    def test_progress_callback(self):
        """Progress callback updates UI."""
        create_progress_callback = _require("create_progress_callback")

        progress_updates = []

        def on_update(step, total, loss):
            progress_updates.append((step, total, loss))

        callback = create_progress_callback(on_update)
        callback(10, 100, 0.5)

        assert len(progress_updates) == 1
        assert progress_updates[0] == (10, 100, 0.5)

    def test_error_callback(self):
        """Error callback shows message."""
        create_error_callback = _require("create_error_callback")

        errors = []

        def on_error(msg):
            errors.append(msg)

        callback = create_error_callback(on_error)
        callback("Test error")

        assert len(errors) == 1
        assert errors[0] == "Test error"

    def test_completion_callback(self):
        """Completion callback shows success."""
        create_completion_callback = _require("create_completion_callback")

        completions = []

        def on_complete(result):
            completions.append(result)

        callback = create_completion_callback(on_complete)
        callback({"loss": 0.5, "steps": 100})

        assert len(completions) == 1


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestUIErrorHandling:
    """Tests for UI error handling."""

    def test_training_error_display(self):
        """Training errors displayed correctly."""
        from backpropagate.exceptions import TrainingError

        format_error_message = _require("format_error_message")

        error = TrainingError(
            message="Out of memory",
            suggestion="Reduce batch size",
        )

        message = format_error_message(error)
        assert "Out of memory" in message

    def test_dataset_error_display(self):
        """Dataset errors displayed correctly."""
        from backpropagate.exceptions import DatasetError

        format_error_message = _require("format_error_message")

        error = DatasetError(
            message="Invalid format",
            suggestion="Use JSONL format",
        )

        message = format_error_message(error)
        assert "Invalid format" in message

    def test_generic_error_display(self):
        """Generic errors displayed correctly."""
        format_error_message = _require("format_error_message")
        error = RuntimeError("Something went wrong")
        message = format_error_message(error)
        assert "Something went wrong" in message


# =============================================================================
# LAUNCH TESTS
# =============================================================================


@_requires_gradio
class TestUILaunch:
    """Tests for UI launch functionality.

    Every test in this class calls ``patch("gradio.Blocks.launch")``, which
    fails with ModuleNotFoundError when gradio isn't installed (the [ui]
    extra now installs Reflex, not Gradio, in v1.1.0). The class-level
    skipif keeps the legacy entrypoint coverable when an operator installs
    gradio manually while skipping cleanly on a stock Reflex-only install.
    """

    def test_launch_default_settings(self):
        """Launch with default settings."""
        launch = _require("launch")

        with patch("gradio.Blocks.launch") as mock_launch:
            launch()
            mock_launch.assert_called_once()

    def test_launch_custom_port(self):
        """Launch with custom port."""
        launch = _require("launch")

        with patch("gradio.Blocks.launch") as mock_launch:
            launch(port=7890)
            mock_launch.assert_called_once()

    def test_launch_with_share(self):
        """Launch with share enabled requires auth."""
        launch = _require("launch")

        with patch("gradio.Blocks.launch") as mock_launch:
            launch(share=True, auth=("admin", "password"))
            mock_launch.assert_called_once()

    def test_launch_with_auth(self):
        """Launch with authentication."""
        launch = _require("launch")

        with patch("gradio.Blocks.launch") as mock_launch:
            launch(auth=("user", "pass"))
            mock_launch.assert_called_once()


# =============================================================================
# STATE MANAGEMENT TESTS
# =============================================================================


class TestUIState:
    """Tests for UI state management."""

    def test_training_state_initial(self):
        """Initial training state is correct."""
        training_state_cls = _require("TrainingState")
        state = training_state_cls()
        assert not state.is_training
        assert state.current_step == 0

    def test_training_state_update(self):
        """Training state updates correctly."""
        training_state_cls = _require("TrainingState")

        state = training_state_cls()
        state.start_training()
        assert state.is_training

        state.update_progress(50, 0.5)
        assert state.current_step == 50

        state.stop_training()
        assert not state.is_training


# =============================================================================
# COMPONENT TESTS
# =============================================================================


class TestUIComponents:
    """Tests for individual UI components."""

    def test_model_selector_component(self):
        """Model selector component works."""
        create_model_selector = _require("create_model_selector")

        with patch("gradio.Dropdown"):
            selector = create_model_selector()
            assert selector is not None

    def test_file_browser_component(self):
        """File browser component works."""
        create_file_browser = _require("create_file_browser")

        with patch("gradio.File"):
            browser = create_file_browser()
            assert browser is not None

    def test_progress_component(self):
        """Progress component works."""
        create_progress_display = _require("create_progress_display")

        with patch("gradio.Progress"):
            progress = create_progress_display()
            assert progress is not None


# =============================================================================
# ACCESSIBILITY TESTS
# =============================================================================


class TestUIAccessibility:
    """Tests for UI accessibility features."""

    @_requires_gradio
    def test_components_have_labels(self):
        """UI components have accessibility labels.

        Smoke-level: ensures create_ui runs to completion under a mocked
        Blocks. The full a11y contract (every component carries a label that
        non-empty) is enforced elsewhere or — once Reflex migration completes
        — in the ui_app/ test surface.
        """
        create_ui = _require("create_ui")

        with patch("gradio.Blocks"):
            result = create_ui()
            # The factory must produce SOMETHING (a Blocks instance, a tuple,
            # etc.). The previous version asserted nothing here, which is what
            # the audit flagged as TESTS-A-001.
            assert result is not None, (
                "create_ui returned None under a mocked Blocks; the factory "
                "must always produce a UI object."
            )
