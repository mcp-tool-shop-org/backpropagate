"""
Backpropagate - GPU Safety Module
==================================

Comprehensive GPU monitoring and safety checks to prevent hardware damage
during intensive training operations.

Features:
- Temperature monitoring with configurable thresholds
- VRAM usage tracking and alerts
- Automatic throttling/pause when limits exceeded
- Power draw monitoring (if supported)
- Graceful shutdown on critical conditions

Safety Thresholds (NVIDIA GPUs):
- Warning: 80C (throttling begins)
- Critical: 90C (pause training)
- Emergency: 95C (abort immediately)

Usage:
    from backpropagate.gpu_safety import GPUMonitor, check_gpu_safe

    # Quick check
    if not check_gpu_safe():
        print("GPU conditions unsafe for training!")

    # Continuous monitoring during training
    monitor = GPUMonitor(check_interval=5.0)
    monitor.start()

    try:
        train_model(...)
    finally:
        monitor.stop()
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from enum import Enum

from .exceptions import GPUError, GPUMonitoringError, GPUTemperatureError

logger = logging.getLogger(__name__)

__all__ = [
    "GPUMonitor",
    "GPUStatus",
    "GPUSafetyConfig",
    "GPUCondition",
    "check_gpu_safe",
    "get_gpu_status",
    "wait_for_safe_gpu",
]


class GPUCondition(Enum):
    """GPU safety condition levels."""
    SAFE = "safe"           # All clear, full speed ahead
    WARM = "warm"           # Elevated temps, monitor closely
    WARNING = "warning"     # Approaching limits, consider throttling
    CRITICAL = "critical"   # Limits exceeded, pause recommended
    EMERGENCY = "emergency" # Dangerous conditions, abort immediately
    UNKNOWN = "unknown"     # Cannot determine (no GPU/no pynvml)


@dataclass
class GPUSafetyConfig:
    """Configuration for GPU safety thresholds."""

    # Temperature thresholds (Celsius)
    temp_warning: float = 80.0      # Begin monitoring closely
    temp_critical: float = 90.0     # Pause training
    temp_emergency: float = 95.0    # Abort immediately

    # VRAM thresholds (percentage of total)
    vram_warning: float = 90.0      # 90% VRAM usage warning
    vram_critical: float = 95.0     # 95% VRAM usage critical

    # Power thresholds (percentage of TDP)
    power_warning: float = 95.0     # Near TDP limit
    power_critical: float = 100.0   # At or over TDP

    # Monitoring settings
    check_interval: float = 5.0     # Seconds between checks
    cooldown_time: float = 30.0     # Seconds to wait when critical
    max_cooldown_attempts: int = 6  # Max cooldown cycles before abort

    # Behavior
    pause_on_critical: bool = True
    abort_on_emergency: bool = True
    log_warnings: bool = True


@dataclass
class GPUStatus:
    """Current GPU status snapshot."""

    # Basic info
    available: bool = False
    device_name: str = ""
    device_index: int = 0

    # Temperature
    temperature_c: Optional[float] = None
    temperature_max_c: Optional[float] = None

    # Memory
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    vram_free_gb: float = 0.0
    vram_percent: float = 0.0

    # Power
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    power_percent: Optional[float] = None

    # Utilization
    gpu_utilization: Optional[int] = None
    memory_utilization: Optional[int] = None

    # Computed condition
    condition: GPUCondition = GPUCondition.UNKNOWN
    condition_reason: str = ""

    # Timestamp
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_gpu_status(device_index: int = 0, config: Optional[GPUSafetyConfig] = None) -> GPUStatus:
    """
    Get current GPU status with safety evaluation.

    Args:
        device_index: GPU device index (default 0)
        config: Safety config for threshold evaluation

    Returns:
        GPUStatus with current readings and safety condition
    """
    config = config or GPUSafetyConfig()
    status = GPUStatus(device_index=device_index)

    # Try PyTorch first for basic info
    try:
        import torch

        if not torch.cuda.is_available():
            status.condition = GPUCondition.UNKNOWN
            status.condition_reason = "No CUDA GPU available"
            return status

        status.available = True
        status.device_name = torch.cuda.get_device_name(device_index)

        # Memory from PyTorch
        props = torch.cuda.get_device_properties(device_index)
        status.vram_total_gb = props.total_memory / (1024**3)

        allocated = torch.cuda.memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)
        status.vram_used_gb = reserved / (1024**3)
        status.vram_free_gb = status.vram_total_gb - status.vram_used_gb
        status.vram_percent = (status.vram_used_gb / status.vram_total_gb) * 100

    except Exception as e:
        logger.debug(f"PyTorch GPU query failed: {e}")
        status.condition = GPUCondition.UNKNOWN
        status.condition_reason = f"PyTorch error: {e}"
        return status

    # Try pynvml for detailed metrics (temperature, power)
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        # Temperature
        try:
            status.temperature_c = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError as e:
            logger.debug(f"pynvml temperature query failed: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error querying GPU temperature: {e}")

        # Max temperature threshold
        try:
            status.temperature_max_c = pynvml.nvmlDeviceGetTemperatureThreshold(
                handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN
            )
        except pynvml.NVMLError as e:
            logger.debug(f"pynvml temperature threshold query failed: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error querying temperature threshold: {e}")

        # Power
        try:
            status.power_draw_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            status.power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            if status.power_limit_w > 0:
                status.power_percent = (status.power_draw_w / status.power_limit_w) * 100
        except pynvml.NVMLError as e:
            logger.debug(f"pynvml power query failed: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error querying power: {e}")

        # Utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            status.gpu_utilization = util.gpu
            status.memory_utilization = util.memory
        except pynvml.NVMLError as e:
            logger.debug(f"pynvml utilization query failed: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error querying utilization: {e}")

        # Always shutdown pynvml
        try:
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.debug(f"pynvml shutdown error: {e}")

    except ImportError:
        logger.debug("pynvml not installed - temperature/power monitoring unavailable")
    except Exception as e:
        logger.debug(f"pynvml query failed: {e}")

    # Evaluate safety condition
    status.condition, status.condition_reason = _evaluate_condition(status, config)
    status.timestamp = time.time()

    return status


def _evaluate_condition(status: GPUStatus, config: GPUSafetyConfig) -> tuple:
    """Evaluate GPU condition based on status and thresholds."""

    # Check temperature (most critical)
    if status.temperature_c is not None:
        if status.temperature_c >= config.temp_emergency:
            return GPUCondition.EMERGENCY, f"Temperature EMERGENCY: {status.temperature_c}C >= {config.temp_emergency}C"
        if status.temperature_c >= config.temp_critical:
            return GPUCondition.CRITICAL, f"Temperature CRITICAL: {status.temperature_c}C >= {config.temp_critical}C"
        if status.temperature_c >= config.temp_warning:
            return GPUCondition.WARNING, f"Temperature WARNING: {status.temperature_c}C >= {config.temp_warning}C"

    # Check VRAM
    if status.vram_percent > 0:
        if status.vram_percent >= config.vram_critical:
            return GPUCondition.CRITICAL, f"VRAM CRITICAL: {status.vram_percent:.1f}% >= {config.vram_critical}%"
        if status.vram_percent >= config.vram_warning:
            return GPUCondition.WARNING, f"VRAM WARNING: {status.vram_percent:.1f}% >= {config.vram_warning}%"

    # Check power
    if status.power_percent is not None:
        if status.power_percent >= config.power_critical:
            return GPUCondition.WARNING, f"Power at TDP: {status.power_percent:.1f}%"
        if status.power_percent >= config.power_warning:
            return GPUCondition.WARM, f"Power high: {status.power_percent:.1f}%"

    # Check for warm conditions
    if status.temperature_c is not None and status.temperature_c >= 70:
        return GPUCondition.WARM, f"Temperature elevated: {status.temperature_c}C"

    return GPUCondition.SAFE, "All metrics within safe limits"


def check_gpu_safe(
    device_index: int = 0,
    config: Optional[GPUSafetyConfig] = None,
) -> bool:
    """
    Quick check if GPU is safe for training.

    Args:
        device_index: GPU device index
        config: Safety configuration

    Returns:
        True if GPU is safe (SAFE, WARM, or WARNING), False if CRITICAL/EMERGENCY
    """
    status = get_gpu_status(device_index, config)

    if status.condition in (GPUCondition.SAFE, GPUCondition.WARM, GPUCondition.WARNING):
        return True

    if status.condition == GPUCondition.UNKNOWN:
        # No GPU or can't determine - allow training but log
        logger.warning("GPU status unknown - proceeding with caution")
        return True

    logger.error(f"GPU unsafe: {status.condition_reason}")
    return False


def wait_for_safe_gpu(
    device_index: int = 0,
    config: Optional[GPUSafetyConfig] = None,
    max_wait_seconds: float = 300.0,
    check_interval: float = 10.0,
) -> bool:
    """
    Wait for GPU to reach safe conditions.

    Useful after a critical condition is detected to allow cooldown.

    Args:
        device_index: GPU device index
        config: Safety configuration
        max_wait_seconds: Maximum time to wait
        check_interval: Seconds between checks

    Returns:
        True if GPU became safe, False if timeout
    """
    config = config or GPUSafetyConfig()
    start_time = time.time()

    logger.info("Waiting for GPU to reach safe temperature...")

    while (time.time() - start_time) < max_wait_seconds:
        status = get_gpu_status(device_index, config)

        if status.condition in (GPUCondition.SAFE, GPUCondition.WARM):
            logger.info(f"GPU safe: {status.temperature_c}C")
            return True

        elapsed = time.time() - start_time
        remaining = max_wait_seconds - elapsed

        if status.temperature_c:
            logger.info(
                f"GPU cooling: {status.temperature_c}C "
                f"(waiting up to {remaining:.0f}s more)"
            )
        else:
            logger.info(f"Waiting for safe GPU ({remaining:.0f}s remaining)")

        time.sleep(check_interval)

    logger.error(f"GPU did not reach safe temperature within {max_wait_seconds}s")
    return False


# =============================================================================
# GPU MONITOR CLASS
# =============================================================================

class GPUMonitor:
    """
    Continuous GPU monitoring with callbacks for safety events.

    Runs in a background thread and can pause/abort training when
    dangerous conditions are detected.

    Usage:
        monitor = GPUMonitor(
            on_warning=lambda s: print(f"Warning: {s.temperature_c}C"),
            on_critical=lambda s: pause_training(),
            on_emergency=lambda s: abort_training(),
        )

        monitor.start()
        try:
            train_model(...)
        finally:
            monitor.stop()
    """

    def __init__(
        self,
        config: Optional[GPUSafetyConfig] = None,
        device_index: int = 0,
        on_warning: Optional[Callable[[GPUStatus], None]] = None,
        on_critical: Optional[Callable[[GPUStatus], None]] = None,
        on_emergency: Optional[Callable[[GPUStatus], None]] = None,
        on_status: Optional[Callable[[GPUStatus], None]] = None,
    ):
        """
        Initialize GPU monitor.

        Args:
            config: Safety configuration
            device_index: GPU device to monitor
            on_warning: Callback when WARNING condition detected
            on_critical: Callback when CRITICAL condition detected
            on_emergency: Callback when EMERGENCY condition detected
            on_status: Callback on every status check (for logging/display)
        """
        self.config = config or GPUSafetyConfig()
        self.device_index = device_index

        self.on_warning = on_warning
        self.on_critical = on_critical
        self.on_emergency = on_emergency
        self.on_status = on_status

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        self._status_history: List[GPUStatus] = []
        self._max_history: int = 100

        self._emergency_triggered = False
        self._critical_count = 0

    def start(self) -> None:
        """Start the monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("GPU monitor already running")
            return

        self._stop_event.clear()
        self._emergency_triggered = False
        self._critical_count = 0

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        logger.info(
            f"GPU monitor started: device={self.device_index}, "
            f"interval={self.config.check_interval}s"
        )

    def stop(self) -> None:
        """Stop the monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        logger.info("GPU monitor stopped")

    def pause(self) -> None:
        """Pause monitoring (still runs but skips callbacks)."""
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume monitoring after pause."""
        self._pause_event.set()

    def get_latest_status(self) -> Optional[GPUStatus]:
        """Get most recent GPU status."""
        if self._status_history:
            return self._status_history[-1]
        return None

    def get_status_history(self) -> List[GPUStatus]:
        """Get status history."""
        return list(self._status_history)

    @property
    def is_emergency(self) -> bool:
        """Check if emergency condition was triggered."""
        return self._emergency_triggered

    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in background thread)."""
        while not self._stop_event.is_set():
            try:
                status = get_gpu_status(self.device_index, self.config)

                # Store in history
                self._status_history.append(status)
                if len(self._status_history) > self._max_history:
                    self._status_history.pop(0)

                # Always call on_status if provided
                if self.on_status and self._pause_event.is_set():
                    try:
                        self.on_status(status)
                    except Exception as e:
                        logger.warning(f"on_status callback raised exception: {type(e).__name__}: {e}")

                # Handle conditions
                if self._pause_event.is_set():
                    self._handle_condition(status)

            except Exception as e:
                logger.error(f"GPU monitor error: {e}")

            # Wait for next check (interruptible)
            self._stop_event.wait(self.config.check_interval)

    def _handle_condition(self, status: GPUStatus) -> None:
        """Handle GPU condition with appropriate callbacks."""

        if status.condition == GPUCondition.EMERGENCY:
            self._emergency_triggered = True
            logger.critical(f"GPU EMERGENCY: {status.condition_reason}")

            if self.on_emergency:
                try:
                    self.on_emergency(status)
                except Exception as e:
                    logger.warning(f"on_emergency callback raised exception: {type(e).__name__}: {e}")

        elif status.condition == GPUCondition.CRITICAL:
            self._critical_count += 1
            logger.error(f"GPU CRITICAL: {status.condition_reason}")

            if self.config.log_warnings:
                logger.warning(
                    f"Critical condition #{self._critical_count}: "
                    f"Consider pausing training"
                )

            if self.on_critical:
                try:
                    self.on_critical(status)
                except Exception as e:
                    logger.warning(f"on_critical callback raised exception: {type(e).__name__}: {e}")

        elif status.condition == GPUCondition.WARNING:
            if self.config.log_warnings:
                logger.warning(f"GPU WARNING: {status.condition_reason}")

            if self.on_warning:
                try:
                    self.on_warning(status)
                except Exception as e:
                    logger.warning(f"on_warning callback raised exception: {type(e).__name__}: {e}")

        else:
            # Safe or warm - reset critical counter
            if self._critical_count > 0:
                logger.info("GPU returned to safe conditions")
                self._critical_count = 0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def format_gpu_status(status: GPUStatus) -> str:
    """Format GPU status for display."""
    lines = [f"GPU: {status.device_name}"]

    if status.temperature_c is not None:
        lines.append(f"Temp: {status.temperature_c}C")

    lines.append(
        f"VRAM: {status.vram_used_gb:.1f}/{status.vram_total_gb:.1f} GB "
        f"({status.vram_percent:.1f}%)"
    )

    if status.power_draw_w is not None:
        lines.append(f"Power: {status.power_draw_w:.0f}W")

    lines.append(f"Status: {status.condition.value.upper()}")

    return " | ".join(lines)


def install_pynvml_hint() -> str:
    """Get installation hint for pynvml."""
    return (
        "For full GPU monitoring (temperature, power), install pynvml:\n"
        "  pip install pynvml\n"
        "Or on Windows: pip install nvidia-ml-py"
    )
