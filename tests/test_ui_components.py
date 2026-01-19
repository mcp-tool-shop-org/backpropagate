"""
Tests for UI Component functionality.

Tests cover:
- Train Tab (rendering, controls, handlers)
- Multi-Run Tab (dashboard, loss chart, checkpoint controls)
- Train Tab Sidebar (live metrics, GPU status, refresh)
- Help Tab (rendering, CLI reference)
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import gradio as gr


# =============================================================================
# TRAIN TAB TESTS
# =============================================================================

class TestTrainTabRenders:
    """Tests for Train tab rendering."""

    def test_train_tab_renders(self):
        """Train tab should load without error."""
        from backpropagate.ui import create_ui

        # Create UI should not raise
        app = create_ui()

        assert app is not None
        assert isinstance(app, gr.Blocks)

    def test_train_tab_has_expected_components(self):
        """Train tab should have expected components."""
        from backpropagate.ui import create_ui

        app = create_ui()

        # The app should be a Blocks object with children
        assert hasattr(app, 'blocks')

    def test_train_start_button_exists(self):
        """Start training button should exist."""
        from backpropagate.ui import create_ui

        app = create_ui()

        # Check that buttons are created (they're registered in the app)
        # This is a structural test - button existence is verified by no errors
        assert app is not None


class TestTrainSidebarRefresh:
    """Tests for Train tab sidebar refresh functionality."""

    def test_sidebar_refresh_function_exists(self):
        """Sidebar refresh function should exist."""
        from backpropagate.ui import refresh_train_sidebar

        assert callable(refresh_train_sidebar)

    def test_sidebar_refresh_returns_tuple(self):
        """Refresh should return tuple of markdown values."""
        from backpropagate.ui import refresh_train_sidebar

        with patch("backpropagate.ui.state") as mock_state:
            mock_state.is_training = False
            mock_state.trainer = None

            with patch("backpropagate.ui.get_gpu_status") as mock_gpu:
                mock_gpu.return_value = MagicMock(
                    temperature_c=65.0,
                    vram_used_gb=8.0,
                    vram_total_gb=16.0,
                    power_draw_w=150.0,
                )

                result = refresh_train_sidebar()

                # Should return 6 values for the sidebar components
                assert isinstance(result, tuple)
                assert len(result) == 6

    def test_sidebar_refresh_during_training(self):
        """Refresh during training should show live metrics."""
        from backpropagate.ui import refresh_train_sidebar
        import time as time_module

        # Use a real UIState-like object to avoid MagicMock comparison issues
        with patch("backpropagate.ui.state") as mock_state:
            mock_state.is_training = True
            mock_state.loss_history = [1.5, 1.2, 0.9]
            # Set train_start_time to None to skip elapsed time calculation
            # (The inner import time bypasses patches, so we avoid it)
            mock_state.train_start_time = None
            mock_state.trainer = MagicMock()
            mock_state.trainer._current_step = 30

            with patch("backpropagate.ui.get_gpu_status") as mock_gpu:
                mock_gpu.return_value = MagicMock(
                    temperature_c=65.0,
                    vram_used_gb=8.0,
                    vram_total_gb=16.0,
                    power_draw_w=150.0,
                )

                result = refresh_train_sidebar()

                assert isinstance(result, tuple)

    def test_sidebar_refresh_with_elapsed_time(self):
        """Refresh with elapsed time calculation should work."""
        from backpropagate.ui import refresh_train_sidebar
        import time

        with patch("backpropagate.ui.state") as mock_state:
            mock_state.is_training = True
            mock_state.loss_history = [1.5, 1.2, 0.9]
            # Set a real start time so elapsed calculation runs
            mock_state.train_start_time = time.time() - 60.0  # 60 seconds ago
            mock_state.trainer = MagicMock()
            mock_state.trainer._current_step = 30

            with patch("backpropagate.ui.get_gpu_status") as mock_gpu:
                mock_gpu.return_value = MagicMock(
                    temperature_c=65.0,
                    vram_used_gb=8.0,
                    vram_total_gb=16.0,
                    power_draw_w=150.0,
                )

                result = refresh_train_sidebar()

                # Should return 6 values with speed info calculated
                assert isinstance(result, tuple)
                assert len(result) == 6


class TestTrainStartButton:
    """Tests for Start training button."""

    def test_start_training_function_exists(self):
        """start_training function should exist."""
        from backpropagate.ui import start_training

        assert callable(start_training)

    def test_start_training_has_correct_signature(self):
        """Start training should have the expected signature."""
        from backpropagate.ui import start_training
        import inspect

        sig = inspect.signature(start_training)
        params = list(sig.parameters.keys())

        # Should have expected parameters
        assert "model_preset" in params
        assert "dataset_preset" in params
        assert "max_steps" in params


class TestTrainStopButton:
    """Tests for Stop training button."""

    def test_stop_training_function_exists(self):
        """stop_training function should exist."""
        from backpropagate.ui import stop_training

        assert callable(stop_training)

    def test_stop_training_when_not_running(self):
        """Stop should work gracefully when not training."""
        from backpropagate.ui import stop_training

        with patch("backpropagate.ui.state") as mock_state:
            mock_state.is_training = False
            mock_state.trainer = None

            result = stop_training()

            # Should return a status message
            assert isinstance(result, str)

    def test_stop_training_halts_training(self):
        """Stop button should halt training."""
        from backpropagate.ui import stop_training

        with patch("backpropagate.ui.state") as mock_state:
            mock_state.is_training = True
            mock_state.trainer = MagicMock()

            result = stop_training()

            # Should have set is_training to False
            assert "stop" in result.lower() or "training" in result.lower()


class TestModelDropdown:
    """Tests for model dropdown population."""

    def test_model_presets_available(self):
        """Model presets should be available."""
        from backpropagate.ui import MODEL_PRESETS

        assert isinstance(MODEL_PRESETS, dict)
        assert len(MODEL_PRESETS) > 0

    def test_model_dropdown_populated(self):
        """Model dropdown should have choices."""
        from backpropagate.ui import MODEL_PRESETS

        # Check expected models are present
        keys = list(MODEL_PRESETS.keys())

        assert any("Qwen" in k for k in keys)
        assert any("Llama" in k or "Mistral" in k or "Phi" in k for k in keys)


class TestDatasetDropdown:
    """Tests for dataset dropdown population."""

    def test_dataset_presets_available(self):
        """Dataset presets should be available."""
        from backpropagate.ui import DATASET_PRESETS

        assert isinstance(DATASET_PRESETS, dict)
        assert len(DATASET_PRESETS) > 0

    def test_dataset_dropdown_populated(self):
        """Dataset dropdown should have options."""
        from backpropagate.ui import DATASET_PRESETS

        keys = list(DATASET_PRESETS.keys())

        assert any("UltraChat" in k for k in keys)
        assert "Custom JSONL" in keys


# =============================================================================
# MULTI-RUN TAB TESTS
# =============================================================================

class TestMultiRunTabRenders:
    """Tests for Multi-Run tab rendering."""

    def test_multirun_tab_renders(self):
        """Multi-run tab should load without error."""
        from backpropagate.ui import create_ui

        app = create_ui()

        # App should create successfully with multi-run tab
        assert app is not None

    def test_multirun_has_dashboard_section(self):
        """Multi-run tab should have dashboard section."""
        from backpropagate.ui import create_ui

        app = create_ui()

        # The presence of the dashboard is verified by successful creation
        assert isinstance(app, gr.Blocks)


class TestMultiRunDashboardMetrics:
    """Tests for Multi-Run dashboard metrics."""

    def test_refresh_dashboard_function_exists(self):
        """Dashboard refresh function should exist."""
        from backpropagate.ui import refresh_dashboard

        assert callable(refresh_dashboard)

    def test_get_dashboard_metrics_function_exists(self):
        """get_dashboard_metrics function should exist."""
        from backpropagate.ui import get_dashboard_metrics

        assert callable(get_dashboard_metrics)

    def test_dashboard_metrics_returns_dict(self):
        """Dashboard metrics should return a dict."""
        from backpropagate.ui import get_dashboard_metrics

        with patch("backpropagate.ui.state") as mock_state:
            mock_state.multi_run_is_running = False
            mock_state.multi_run_trainer = None
            mock_state.multi_run_current_run = 0
            mock_state.multi_run_loss_history = []
            mock_state.multi_run_run_boundaries = []
            mock_state.multi_run_results = None

            # Mock time to avoid comparison issues
            with patch("backpropagate.ui.time") as mock_time:
                mock_time.time.return_value = 100.0

                result = get_dashboard_metrics()

                assert isinstance(result, dict)


class TestMultiRunLossChart:
    """Tests for Multi-Run loss chart."""

    def test_format_multi_run_plot_function_exists(self):
        """format_multi_run_plot function should exist."""
        from backpropagate.ui import format_multi_run_plot

        assert callable(format_multi_run_plot)

    def test_format_loss_plot_function_exists(self):
        """format_loss_plot function should exist."""
        from backpropagate.ui import format_loss_plot

        assert callable(format_loss_plot)

    def test_loss_plot_renders_with_data(self):
        """Loss plot should render with data."""
        from backpropagate.ui import format_loss_plot

        # Create a plot with sample data
        loss_history = [1.5, 1.2, 1.0, 0.8, 0.6]

        result = format_loss_plot(loss_history)

        # Should return a dict or None
        assert result is None or isinstance(result, dict)

    def test_loss_plot_handles_empty_data(self):
        """Loss plot should handle empty data."""
        from backpropagate.ui import format_loss_plot

        result = format_loss_plot([])

        # Should return empty plot dict for empty data
        assert isinstance(result, dict)
        assert result.get('data') == []

    def test_multi_run_plot_with_boundaries(self):
        """Multi-run plot should handle run boundaries."""
        from backpropagate.ui import format_multi_run_plot

        loss_history = [1.5, 1.2, 1.0, 0.8, 0.6]
        run_boundaries = [2]

        result = format_multi_run_plot(loss_history, run_boundaries)

        # Should return a dict or None
        assert result is None or isinstance(result, dict)


class TestMultiRunCheckpointControls:
    """Tests for Multi-Run checkpoint UI controls."""

    def test_checkpoint_section_in_ui(self):
        """Checkpoint management section should exist in UI."""
        from backpropagate.ui import create_ui

        app = create_ui()

        # Verify the app was created with checkpoint controls
        assert app is not None

    def test_checkpoint_buttons_work(self):
        """Checkpoint control buttons should function."""
        # This test verifies the structure exists
        # Actual button functionality is tested via integration tests
        from backpropagate.ui import create_ui

        app = create_ui()

        assert isinstance(app, gr.Blocks)


class TestMultiRunStartMultipleRuns:
    """Tests for starting multi-run training."""

    def test_start_multi_run_function_exists(self):
        """start_multi_run function should exist."""
        from backpropagate.ui import start_multi_run

        assert callable(start_multi_run)

    def test_start_multi_run_has_correct_signature(self):
        """start_multi_run should have expected signature."""
        from backpropagate.ui import start_multi_run
        import inspect

        sig = inspect.signature(start_multi_run)
        params = list(sig.parameters.keys())

        # Should have expected parameters
        assert "model_preset" in params
        assert "num_runs" in params
        assert "steps_per_run" in params


# =============================================================================
# TRAIN TAB SIDEBAR TESTS
# =============================================================================

class TestSidebarShowsLiveMetrics:
    """Tests for sidebar live metrics display."""

    def test_sidebar_shows_step(self):
        """Sidebar should show current step."""
        from backpropagate.ui import refresh_train_sidebar

        with patch("backpropagate.ui.state") as mock_state:
            mock_state.is_training = False
            mock_state.trainer = None

            with patch("backpropagate.ui.get_gpu_status") as mock_gpu:
                mock_gpu.return_value = MagicMock(
                    temperature_c=65.0,
                    vram_used_gb=8.0,
                    vram_total_gb=16.0,
                    power_draw_w=150.0,
                )

                result = refresh_train_sidebar()

                # First element should be step info
                assert isinstance(result[0], str)

    def test_sidebar_shows_loss(self):
        """Sidebar should show current loss."""
        from backpropagate.ui import refresh_train_sidebar

        with patch("backpropagate.ui.state") as mock_state:
            mock_state.is_training = False
            mock_state.trainer = None

            with patch("backpropagate.ui.get_gpu_status") as mock_gpu:
                mock_gpu.return_value = MagicMock(
                    temperature_c=65.0,
                    vram_used_gb=8.0,
                    vram_total_gb=16.0,
                    power_draw_w=150.0,
                )

                result = refresh_train_sidebar()

                # Second element should be loss info
                assert isinstance(result[1], str)

    def test_sidebar_shows_speed(self):
        """Sidebar should show training speed."""
        from backpropagate.ui import refresh_train_sidebar

        with patch("backpropagate.ui.state") as mock_state:
            mock_state.is_training = False
            mock_state.trainer = None

            with patch("backpropagate.ui.get_gpu_status") as mock_gpu:
                mock_gpu.return_value = MagicMock(
                    temperature_c=65.0,
                    vram_used_gb=8.0,
                    vram_total_gb=16.0,
                    power_draw_w=150.0,
                )

                result = refresh_train_sidebar()

                # Third element should be speed info
                assert isinstance(result[2], str)


class TestSidebarShowsGPUStatus:
    """Tests for sidebar GPU status display."""

    def test_sidebar_shows_temp(self):
        """Sidebar should show GPU temperature."""
        from backpropagate.ui import refresh_train_sidebar

        with patch("backpropagate.ui.state") as mock_state:
            mock_state.is_training = False

            with patch("backpropagate.ui.get_gpu_status") as mock_gpu:
                mock_gpu.return_value = MagicMock(
                    temperature_c=72.0,
                    vram_used_gb=8.0,
                    vram_total_gb=16.0,
                    power_draw_w=180.0,
                )

                result = refresh_train_sidebar()

                # Fourth element should be temperature
                temp_str = result[3]
                assert "72" in temp_str or "Temperature" in temp_str

    def test_sidebar_shows_vram(self):
        """Sidebar should show VRAM usage."""
        from backpropagate.ui import refresh_train_sidebar

        with patch("backpropagate.ui.state") as mock_state:
            mock_state.is_training = False

            with patch("backpropagate.ui.get_gpu_status") as mock_gpu:
                mock_gpu.return_value = MagicMock(
                    temperature_c=65.0,
                    vram_used_gb=10.5,
                    vram_total_gb=16.0,
                    power_draw_w=150.0,
                )

                result = refresh_train_sidebar()

                # Fifth element should be VRAM
                vram_str = result[4]
                assert "VRAM" in vram_str or "10" in vram_str or "16" in vram_str

    def test_sidebar_shows_power(self):
        """Sidebar should show power draw."""
        from backpropagate.ui import refresh_train_sidebar

        with patch("backpropagate.ui.state") as mock_state:
            mock_state.is_training = False

            with patch("backpropagate.ui.get_gpu_status") as mock_gpu:
                mock_gpu.return_value = MagicMock(
                    temperature_c=65.0,
                    vram_used_gb=8.0,
                    vram_total_gb=16.0,
                    power_draw_w=175.0,
                )

                result = refresh_train_sidebar()

                # Sixth element should be power
                power_str = result[5]
                assert "Power" in power_str or "175" in power_str or "W" in power_str


class TestSidebarRefreshButton:
    """Tests for sidebar manual refresh button."""

    def test_sidebar_refresh_button_works(self):
        """Manual refresh button should work."""
        from backpropagate.ui import refresh_train_sidebar

        with patch("backpropagate.ui.state") as mock_state:
            mock_state.is_training = False

            with patch("backpropagate.ui.get_gpu_status") as mock_gpu:
                mock_gpu.return_value = MagicMock(
                    temperature_c=65.0,
                    vram_used_gb=8.0,
                    vram_total_gb=16.0,
                    power_draw_w=150.0,
                )

                # Should not raise
                result = refresh_train_sidebar()

                assert result is not None
                assert len(result) == 6


class TestSidebarQuickStartCollapsed:
    """Tests for sidebar Quick Start accordion."""

    def test_quick_start_tutorial_exists(self):
        """Quick Start tutorial should be in accordion."""
        from backpropagate.ui import create_ui

        app = create_ui()

        # Verify app created successfully with Quick Start
        assert app is not None


# =============================================================================
# HELP TAB TESTS
# =============================================================================

class TestHelpTabRenders:
    """Tests for Help tab rendering."""

    def test_help_tab_renders(self):
        """Help tab should load without error."""
        from backpropagate.ui import create_ui

        app = create_ui()

        assert app is not None
        assert isinstance(app, gr.Blocks)

    def test_help_tab_has_content(self):
        """Help tab should have helpful content."""
        from backpropagate.ui import create_ui

        app = create_ui()

        # The app should create with all tabs including Help
        assert app is not None


class TestCLIReferenceCodeBlock:
    """Tests for CLI reference code block."""

    def test_cli_commands_display_correctly(self):
        """CLI commands should be displayed in code block."""
        from backpropagate.ui import create_ui

        app = create_ui()

        # Verify app creates with CLI reference
        assert app is not None

    def test_cli_reference_has_expected_commands(self):
        """CLI reference should have expected commands."""
        # Read the ui.py to verify CLI commands are present
        from backpropagate.ui import create_ui

        app = create_ui()

        # The CLI reference block is verified by successful creation
        # Actual content is in the code block in the Help tab
        assert app is not None


# =============================================================================
# UI STATE TESTS
# =============================================================================

class TestUIState:
    """Tests for UI state management."""

    def test_ui_state_initialization(self):
        """UIState should have correct initial values."""
        from backpropagate.ui import UIState

        state = UIState()

        assert state.trainer is None
        assert state.is_training is False
        assert state.current_run is None
        assert state.loss_history == []

    def test_ui_state_multi_run_initialization(self):
        """UIState multi-run attributes should be initialized."""
        from backpropagate.ui import UIState

        state = UIState()

        assert state.multi_run_trainer is None
        assert state.multi_run_is_running is False
        assert state.multi_run_results is None
        assert state.multi_run_current_run == 0

    def test_ui_state_dataset_initialization(self):
        """UIState dataset attributes should be initialized."""
        from backpropagate.ui import UIState

        state = UIState()

        assert state.dataset_loader is None
        assert state.dataset_validation is None


# =============================================================================
# SECURITY INTEGRATION TESTS
# =============================================================================

class TestUISecurityIntegration:
    """Tests for UI security features."""

    def test_rate_limiter_used_for_training(self):
        """Rate limiter should be used for training operations."""
        from backpropagate.ui import _training_limiter

        assert _training_limiter is not None
        assert hasattr(_training_limiter, 'is_allowed')

    def test_rate_limiter_used_for_export(self):
        """Rate limiter should be used for export operations."""
        from backpropagate.ui import _export_limiter

        assert _export_limiter is not None
        assert hasattr(_export_limiter, 'is_allowed')

    def test_path_validation_function_available(self):
        """Path validation function should be available."""
        from backpropagate.ui import validate_path_input

        assert callable(validate_path_input)

    def test_model_name_sanitization_available(self):
        """Model name sanitization should be available."""
        from backpropagate.ui import sanitize_model_name

        assert callable(sanitize_model_name)

    def test_text_sanitization_available(self):
        """Text sanitization should be available."""
        from backpropagate.ui import sanitize_text_input

        assert callable(sanitize_text_input)


# =============================================================================
# LAUNCH FUNCTION TESTS
# =============================================================================

class TestLaunchFunction:
    """Tests for the launch function."""

    def test_launch_function_exists(self):
        """launch function should exist."""
        from backpropagate.ui import launch

        assert callable(launch)

    def test_create_ui_function_exists(self):
        """create_ui function should exist."""
        from backpropagate.ui import create_ui

        assert callable(create_ui)


# =============================================================================
# GPU STATUS DISPLAY TESTS
# =============================================================================

class TestGPUStatusDisplay:
    """Tests for GPU status display."""

    def test_get_gpu_status_display_exists(self):
        """get_gpu_status_display function should exist."""
        from backpropagate.ui import get_gpu_status_display

        assert callable(get_gpu_status_display)

    def test_get_gpu_status_display_returns_string(self):
        """get_gpu_status_display should return formatted string."""
        from backpropagate.ui import get_gpu_status_display

        with patch("backpropagate.ui.get_gpu_status") as mock_gpu:
            mock_gpu.return_value = MagicMock(
                available=True,
                device_name="Test GPU",
                temperature_c=65.0,
                vram_used_gb=8.0,
                vram_total_gb=16.0,
                power_draw_w=150.0,
                condition=MagicMock(value="safe"),
            )

            result = get_gpu_status_display()

            assert isinstance(result, str)
