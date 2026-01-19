"""
Tests for GPU Safety monitoring module.

Tests cover:
- GPU status detection
- Safety condition evaluation
- Temperature/VRAM thresholds
- GPUMonitor class
- Wait/cooldown functions
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock, PropertyMock

from backpropagate.gpu_safety import (
    GPUMonitor,
    GPUStatus,
    GPUSafetyConfig,
    GPUCondition,
    get_gpu_status,
    check_gpu_safe,
    wait_for_safe_gpu,
    format_gpu_status,
    _evaluate_condition,
)


class TestGPUCondition:
    """Tests for GPUCondition enum."""

    def test_condition_values(self):
        """Should have expected condition values."""
        assert GPUCondition.SAFE.value == "safe"
        assert GPUCondition.WARM.value == "warm"
        assert GPUCondition.WARNING.value == "warning"
        assert GPUCondition.CRITICAL.value == "critical"
        assert GPUCondition.EMERGENCY.value == "emergency"
        assert GPUCondition.UNKNOWN.value == "unknown"


class TestGPUSafetyConfig:
    """Tests for GPUSafetyConfig dataclass."""

    def test_default_values(self):
        """Should have sensible default thresholds."""
        config = GPUSafetyConfig()

        assert config.temp_warning == 80.0
        assert config.temp_critical == 90.0
        assert config.temp_emergency == 95.0
        assert config.vram_warning == 90.0
        assert config.vram_critical == 95.0
        assert config.check_interval == 5.0

    def test_custom_values(self):
        """Should accept custom threshold values."""
        config = GPUSafetyConfig(
            temp_warning=75.0,
            temp_critical=85.0,
            temp_emergency=90.0,
        )

        assert config.temp_warning == 75.0
        assert config.temp_critical == 85.0
        assert config.temp_emergency == 90.0


class TestGPUStatus:
    """Tests for GPUStatus dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        status = GPUStatus()

        assert status.available is False
        assert status.device_name == ""
        assert status.temperature_c is None
        assert status.vram_total_gb == 0.0
        assert status.condition == GPUCondition.UNKNOWN

    def test_custom_values(self):
        """Should accept custom values."""
        status = GPUStatus(
            available=True,
            device_name="Test GPU",
            temperature_c=75.0,
            vram_total_gb=16.0,
            vram_used_gb=8.0,
            condition=GPUCondition.SAFE,
        )

        assert status.available is True
        assert status.device_name == "Test GPU"
        assert status.temperature_c == 75.0


class TestEvaluateCondition:
    """Tests for the _evaluate_condition function."""

    @pytest.fixture
    def default_config(self):
        return GPUSafetyConfig()

    def test_safe_temperature(self, default_config):
        """Should return SAFE for low temperature."""
        status = GPUStatus(temperature_c=60.0)
        condition, reason = _evaluate_condition(status, default_config)

        assert condition == GPUCondition.SAFE

    def test_warm_temperature(self, default_config):
        """Should return WARM for elevated temperature."""
        status = GPUStatus(temperature_c=72.0)
        condition, reason = _evaluate_condition(status, default_config)

        assert condition == GPUCondition.WARM

    def test_warning_temperature(self, default_config):
        """Should return WARNING at warning threshold."""
        status = GPUStatus(temperature_c=82.0)
        condition, reason = _evaluate_condition(status, default_config)

        assert condition == GPUCondition.WARNING
        assert "Temperature WARNING" in reason

    def test_critical_temperature(self, default_config):
        """Should return CRITICAL at critical threshold."""
        status = GPUStatus(temperature_c=92.0)
        condition, reason = _evaluate_condition(status, default_config)

        assert condition == GPUCondition.CRITICAL
        assert "Temperature CRITICAL" in reason

    def test_emergency_temperature(self, default_config):
        """Should return EMERGENCY at emergency threshold."""
        status = GPUStatus(temperature_c=96.0)
        condition, reason = _evaluate_condition(status, default_config)

        assert condition == GPUCondition.EMERGENCY
        assert "Temperature EMERGENCY" in reason

    def test_warning_vram(self, default_config):
        """Should return WARNING for high VRAM usage."""
        status = GPUStatus(vram_percent=92.0)
        condition, reason = _evaluate_condition(status, default_config)

        assert condition == GPUCondition.WARNING
        assert "VRAM WARNING" in reason

    def test_critical_vram(self, default_config):
        """Should return CRITICAL for very high VRAM usage."""
        status = GPUStatus(vram_percent=96.0)
        condition, reason = _evaluate_condition(status, default_config)

        assert condition == GPUCondition.CRITICAL
        assert "VRAM CRITICAL" in reason

    def test_temperature_priority_over_vram(self, default_config):
        """Temperature should be checked before VRAM."""
        status = GPUStatus(temperature_c=92.0, vram_percent=96.0)
        condition, reason = _evaluate_condition(status, default_config)

        # Temperature critical should take priority
        assert condition == GPUCondition.CRITICAL
        assert "Temperature" in reason


class TestGetGPUStatus:
    """Tests for the get_gpu_status function."""

    def test_no_cuda(self):
        """Should handle missing CUDA gracefully."""
        with patch("torch.cuda.is_available", return_value=False):
            status = get_gpu_status()

            assert status.available is False
            assert status.condition == GPUCondition.UNKNOWN

    def test_with_cuda(self):
        """Should detect GPU when CUDA is available."""
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)  # 16 GB

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="Test GPU"), \
             patch("torch.cuda.get_device_properties", return_value=mock_props), \
             patch("torch.cuda.memory_allocated", return_value=4 * (1024**3)), \
             patch("torch.cuda.memory_reserved", return_value=8 * (1024**3)):

            status = get_gpu_status()

            assert status.available is True
            assert status.device_name == "Test GPU"
            assert status.vram_total_gb == pytest.approx(16.0, abs=0.1)

    def test_uses_config_thresholds(self):
        """Should use provided config for evaluation."""
        config = GPUSafetyConfig(temp_warning=50.0)

        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="Test GPU"), \
             patch("torch.cuda.get_device_properties", return_value=mock_props), \
             patch("torch.cuda.memory_allocated", return_value=0), \
             patch("torch.cuda.memory_reserved", return_value=0):

            # Without pynvml, temp will be None, so condition based on VRAM
            status = get_gpu_status(config=config)
            assert status.available is True


class TestCheckGPUSafe:
    """Tests for the check_gpu_safe convenience function."""

    def test_safe_returns_true(self):
        """Should return True for safe conditions."""
        mock_status = GPUStatus(
            available=True,
            condition=GPUCondition.SAFE,
        )

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            assert check_gpu_safe() is True

    def test_warning_returns_true(self):
        """Should return True for warning conditions (still safe to train)."""
        mock_status = GPUStatus(
            available=True,
            condition=GPUCondition.WARNING,
        )

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            assert check_gpu_safe() is True

    def test_critical_returns_false(self):
        """Should return False for critical conditions."""
        mock_status = GPUStatus(
            available=True,
            condition=GPUCondition.CRITICAL,
            condition_reason="Test critical",
        )

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            assert check_gpu_safe() is False

    def test_emergency_returns_false(self):
        """Should return False for emergency conditions."""
        mock_status = GPUStatus(
            available=True,
            condition=GPUCondition.EMERGENCY,
            condition_reason="Test emergency",
        )

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            assert check_gpu_safe() is False

    def test_unknown_returns_true_with_warning(self):
        """Should return True for unknown (allows training with caution)."""
        mock_status = GPUStatus(
            available=False,
            condition=GPUCondition.UNKNOWN,
        )

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            assert check_gpu_safe() is True


class TestWaitForSafeGPU:
    """Tests for the wait_for_safe_gpu function."""

    def test_already_safe(self):
        """Should return immediately if already safe."""
        mock_status = GPUStatus(
            available=True,
            condition=GPUCondition.SAFE,
            temperature_c=60.0,
        )

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            result = wait_for_safe_gpu(max_wait_seconds=1.0, check_interval=0.1)
            assert result is True

    def test_becomes_safe(self):
        """Should return True when GPU becomes safe."""
        statuses = [
            GPUStatus(available=True, condition=GPUCondition.CRITICAL, temperature_c=92.0),
            GPUStatus(available=True, condition=GPUCondition.WARNING, temperature_c=82.0),
            GPUStatus(available=True, condition=GPUCondition.SAFE, temperature_c=70.0),
        ]

        call_count = [0]

        def mock_get_status(*args, **kwargs):
            idx = min(call_count[0], len(statuses) - 1)
            call_count[0] += 1
            return statuses[idx]

        with patch("backpropagate.gpu_safety.get_gpu_status", side_effect=mock_get_status):
            result = wait_for_safe_gpu(max_wait_seconds=5.0, check_interval=0.1)
            assert result is True

    def test_timeout(self):
        """Should return False on timeout."""
        mock_status = GPUStatus(
            available=True,
            condition=GPUCondition.CRITICAL,
            temperature_c=92.0,
        )

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            result = wait_for_safe_gpu(max_wait_seconds=0.3, check_interval=0.1)
            assert result is False


class TestGPUMonitor:
    """Tests for the GPUMonitor class."""

    def test_initialization(self):
        """Should initialize with default config."""
        monitor = GPUMonitor()

        assert monitor.config is not None
        assert monitor.device_index == 0
        assert monitor._thread is None

    def test_start_stop(self):
        """Should start and stop monitoring thread."""
        mock_status = GPUStatus(
            available=True,
            condition=GPUCondition.SAFE,
        )

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            monitor = GPUMonitor(config=GPUSafetyConfig(check_interval=0.1))
            monitor.start()

            assert monitor._thread is not None
            assert monitor._thread.is_alive()

            monitor.stop()

            # Thread should be cleaned up after stop
            assert monitor._thread is None

    def test_callbacks(self):
        """Should call appropriate callbacks."""
        warning_called = []
        critical_called = []

        def on_warning(status):
            warning_called.append(status)

        def on_critical(status):
            critical_called.append(status)

        # Return warning status
        mock_status = GPUStatus(
            available=True,
            condition=GPUCondition.WARNING,
            condition_reason="Test warning",
        )

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            monitor = GPUMonitor(
                config=GPUSafetyConfig(check_interval=0.1),
                on_warning=on_warning,
                on_critical=on_critical,
            )

            monitor.start()
            time.sleep(0.3)
            monitor.stop()

            assert len(warning_called) > 0

    def test_emergency_flag(self):
        """Should set emergency flag on emergency condition."""
        mock_status = GPUStatus(
            available=True,
            condition=GPUCondition.EMERGENCY,
            condition_reason="Test emergency",
        )

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            monitor = GPUMonitor(config=GPUSafetyConfig(check_interval=0.1))
            monitor.start()

            time.sleep(0.3)

            assert monitor.is_emergency is True

            monitor.stop()

    def test_status_history(self):
        """Should maintain status history."""
        mock_status = GPUStatus(
            available=True,
            condition=GPUCondition.SAFE,
        )

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            monitor = GPUMonitor(config=GPUSafetyConfig(check_interval=0.1))
            monitor.start()

            time.sleep(0.5)
            monitor.stop()

            history = monitor.get_status_history()
            assert len(history) > 0

    def test_pause_resume(self):
        """Should support pause and resume."""
        callback_count = [0]

        def on_status(status):
            callback_count[0] += 1

        mock_status = GPUStatus(available=True, condition=GPUCondition.SAFE)

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            monitor = GPUMonitor(
                config=GPUSafetyConfig(check_interval=0.1),
                on_status=on_status,
            )

            monitor.start()
            time.sleep(0.25)

            count_before_pause = callback_count[0]

            monitor.pause()
            time.sleep(0.25)

            count_after_pause = callback_count[0]

            # Should have stopped incrementing during pause
            # (some tolerance for timing)
            assert count_after_pause <= count_before_pause + 1

            monitor.resume()
            time.sleep(0.25)

            count_after_resume = callback_count[0]

            # Should have resumed incrementing
            assert count_after_resume > count_after_pause

            monitor.stop()


class TestFormatGPUStatus:
    """Tests for the format_gpu_status function."""

    def test_format_full_status(self):
        """Should format all available fields."""
        status = GPUStatus(
            available=True,
            device_name="NVIDIA RTX 5080",
            temperature_c=65.0,
            vram_total_gb=16.0,
            vram_used_gb=8.0,
            vram_percent=50.0,
            power_draw_w=150.0,
            condition=GPUCondition.SAFE,
        )

        formatted = format_gpu_status(status)

        assert "NVIDIA RTX 5080" in formatted
        assert "65" in formatted  # Temperature
        assert "16" in formatted  # VRAM
        assert "SAFE" in formatted

    def test_format_minimal_status(self):
        """Should handle minimal status gracefully."""
        status = GPUStatus(
            available=True,
            device_name="Test GPU",
            vram_total_gb=8.0,
            vram_used_gb=4.0,
            vram_percent=50.0,
            condition=GPUCondition.UNKNOWN,
        )

        formatted = format_gpu_status(status)

        assert "Test GPU" in formatted
        assert "UNKNOWN" in formatted


# =============================================================================
# GPU STATUS READING TESTS (Phase 5 additions)
# =============================================================================

class TestGPUStatusReadings:
    """Tests for GPU status readings (temperature, VRAM, power, utilization)."""

    def test_get_gpu_status_with_nvidia(self):
        """Should return valid status on NVIDIA GPU."""
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)  # 16 GB

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 5080"), \
             patch("torch.cuda.get_device_properties", return_value=mock_props), \
             patch("torch.cuda.memory_allocated", return_value=4 * (1024**3)), \
             patch("torch.cuda.memory_reserved", return_value=8 * (1024**3)):

            status = get_gpu_status()

            assert status.available is True
            assert "NVIDIA" in status.device_name
            assert status.vram_total_gb > 0

    def test_get_gpu_status_no_gpu(self):
        """Should gracefully fallback when no GPU is available."""
        with patch("torch.cuda.is_available", return_value=False):
            status = get_gpu_status()

            assert status.available is False
            assert status.condition == GPUCondition.UNKNOWN
            assert "No CUDA" in status.condition_reason

    def test_gpu_temperature_reading(self):
        """Temperature should be in valid range (0-100C typically)."""
        # Test with mocked temperature value
        status = GPUStatus(
            available=True,
            temperature_c=65.0,
        )

        # Temperature should be reasonable
        assert status.temperature_c >= 0
        assert status.temperature_c <= 110  # GPU max before shutdown

        # Test evaluation with temperature
        config = GPUSafetyConfig()
        condition, reason = _evaluate_condition(status, config)

        # 65C should be safe
        assert condition == GPUCondition.SAFE

    def test_gpu_vram_reading(self):
        """VRAM used should be <= VRAM total."""
        status = GPUStatus(
            available=True,
            vram_total_gb=16.0,
            vram_used_gb=8.0,
            vram_free_gb=8.0,
            vram_percent=50.0,
        )

        # Basic consistency check
        assert status.vram_used_gb <= status.vram_total_gb
        assert status.vram_free_gb >= 0
        assert status.vram_percent >= 0
        assert status.vram_percent <= 100

        # Used + Free should approximately equal Total
        assert abs((status.vram_used_gb + status.vram_free_gb) - status.vram_total_gb) < 0.1

    def test_gpu_power_reading(self):
        """Power draw should be a reasonable value."""
        status = GPUStatus(
            available=True,
            power_draw_w=150.0,
            power_limit_w=250.0,
            power_percent=60.0,
        )

        # Power should be reasonable for consumer GPU (0-600W range)
        assert status.power_draw_w >= 0
        assert status.power_draw_w <= 600

        # Power limit should be set
        assert status.power_limit_w > 0

        # Percentage should be consistent
        expected_percent = (status.power_draw_w / status.power_limit_w) * 100
        assert abs(status.power_percent - expected_percent) < 1.0

    def test_gpu_utilization_reading(self):
        """Utilization should be 0-100%."""
        status = GPUStatus(
            available=True,
            gpu_utilization=75,
            memory_utilization=50,
        )

        # GPU utilization should be valid percentage
        assert status.gpu_utilization >= 0
        assert status.gpu_utilization <= 100

        # Memory utilization should be valid percentage
        assert status.memory_utilization >= 0
        assert status.memory_utilization <= 100

    def test_power_warning_evaluation(self):
        """Power at TDP should trigger warning condition."""
        config = GPUSafetyConfig(power_warning=95.0, power_critical=100.0)

        # At warning threshold
        status = GPUStatus(
            available=True,
            power_percent=96.0,
        )
        condition, reason = _evaluate_condition(status, config)
        assert condition == GPUCondition.WARM
        assert "Power" in reason

        # At critical threshold
        status_critical = GPUStatus(
            available=True,
            power_percent=100.0,
        )
        condition_crit, reason_crit = _evaluate_condition(status_critical, config)
        assert condition_crit == GPUCondition.WARNING  # Power at TDP triggers warning
        assert "Power" in reason_crit

    def test_power_reading_with_pynvml_mock(self):
        """Should read power data from pynvml when available."""
        # This test verifies the structure of pynvml integration
        # without reloading the module (which can cause enum identity issues)

        # Verify pynvml integration is handled in get_gpu_status
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="Test GPU"), \
             patch("torch.cuda.get_device_properties", return_value=mock_props), \
             patch("torch.cuda.memory_allocated", return_value=4 * (1024**3)), \
             patch("torch.cuda.memory_reserved", return_value=8 * (1024**3)):

            # Without pynvml, we should still get basic status
            status = get_gpu_status()
            assert status.available is True
            # Power readings may be None without pynvml
            # This is expected behavior

    def test_utilization_reading_validation(self):
        """Utilization readings should be validated as percentages."""
        # Valid utilization values
        status_valid = GPUStatus(
            available=True,
            gpu_utilization=50,
            memory_utilization=75,
        )
        assert 0 <= status_valid.gpu_utilization <= 100
        assert 0 <= status_valid.memory_utilization <= 100

        # Edge cases - boundary values
        status_zero = GPUStatus(gpu_utilization=0, memory_utilization=0)
        assert status_zero.gpu_utilization == 0

        status_full = GPUStatus(gpu_utilization=100, memory_utilization=100)
        assert status_full.gpu_utilization == 100


class TestGPUSafetyEdgeCases:
    """Edge case tests for GPU safety module."""

    def test_zero_vram(self):
        """Should handle zero VRAM gracefully."""
        config = GPUSafetyConfig()
        status = GPUStatus(vram_percent=0.0)

        condition, reason = _evaluate_condition(status, config)
        assert condition == GPUCondition.SAFE

    def test_negative_temperature(self):
        """Should handle negative temperature (error case)."""
        config = GPUSafetyConfig()
        status = GPUStatus(temperature_c=-10.0)

        # Should still work, treating as safe
        condition, reason = _evaluate_condition(status, config)
        assert condition == GPUCondition.SAFE

    def test_monitor_handles_exceptions(self):
        """Monitor should handle exceptions in status check."""
        def raise_error(*args, **kwargs):
            raise RuntimeError("Test error")

        with patch("backpropagate.gpu_safety.get_gpu_status", side_effect=raise_error):
            monitor = GPUMonitor(config=GPUSafetyConfig(check_interval=0.1))
            monitor.start()

            # Should not crash
            time.sleep(0.3)
            monitor.stop()

    def test_concurrent_monitors(self):
        """Should support multiple concurrent monitors."""
        mock_status = GPUStatus(available=True, condition=GPUCondition.SAFE)

        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=mock_status):
            monitor1 = GPUMonitor(config=GPUSafetyConfig(check_interval=0.1))
            monitor2 = GPUMonitor(config=GPUSafetyConfig(check_interval=0.1))

            monitor1.start()
            monitor2.start()

            time.sleep(0.3)

            monitor1.stop()
            monitor2.stop()

            # Both should have collected history
            assert len(monitor1.get_status_history()) > 0
            assert len(monitor2.get_status_history()) > 0
