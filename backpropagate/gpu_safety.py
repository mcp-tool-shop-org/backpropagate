"""
Backpropagate - GPU Safety Module
==================================

GPU monitoring and safety checks to prevent hardware damage during intensive
training operations. The VRAM metric here is per-process (the current
PyTorch CUDA context) — it does NOT see allocations made by other processes
sharing the same GPU. For system-wide VRAM accounting use ``nvidia-smi`` or
the future system-vram helper. Temperature, power, and utilization come from
pynvml and ARE system-wide.

Features:
- Temperature monitoring with configurable thresholds (system-wide via pynvml)
- VRAM usage tracking and alerts (per-process; see caveat above)
- Automatic throttling/pause when limits exceeded
- Power draw monitoring (if supported, system-wide via pynvml)
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

import atexit
import collections
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# TRAINER-A-009: degrees Celsius below the card's hardware shutdown threshold
# (NVML NVML_TEMPERATURE_THRESHOLD_SHUTDOWN) at which we escalate to EMERGENCY,
# independent of the statically-configured temp_emergency. A card already this
# close to its own protection cutoff is in danger regardless of our default.
_HW_SHUTDOWN_MARGIN_C = 3.0

# =============================================================================
# PYNVML INIT-ONCE PATTERN (BR-009)
# =============================================================================
# nvmlInit()/nvmlShutdown() on every get_gpu_status() call causes driver-level
# errors under rapid polling. Instead, init once and register atexit shutdown.

_nvml_initialized = False
_nvml_init_lock = threading.Lock()
_nvml_unavailable_logged = False
# Stage C amend BACKEND-B-014: cache runtime-init failures so a permanently-
# broken NVML environment (driver mismatch, broken WSL setup, sensor failure)
# doesn't spam N log lines per get_gpu_status() call inside the monitoring
# loop. The flag is reset by :func:`reset_nvml_state` for operators who
# fix their drivers mid-session.
_nvml_runtime_failed = False
_nvml_runtime_failure_logged = False


def reset_nvml_state() -> None:
    """Stage C amend BACKEND-B-014: clear the runtime-failure cache.

    After fixing a broken driver / NVML setup mid-session, call this to
    re-arm :func:`_ensure_nvml_initialized` so it'll attempt nvmlInit()
    again on the next call. Without this escape hatch, the monitoring
    loop would refuse to retry until the process is restarted.

    Safe to call from any thread (acquires the same lock the init path
    uses). No-op when no failure has been cached.
    """
    global _nvml_runtime_failed, _nvml_runtime_failure_logged
    with _nvml_init_lock:
        if _nvml_runtime_failed:
            logger.info(
                "reset_nvml_state: clearing cached NVML runtime-failure flag; "
                "next get_gpu_status() will retry nvmlInit()."
            )
        _nvml_runtime_failed = False
        _nvml_runtime_failure_logged = False


def _ensure_nvml_initialized() -> bool:
    """
    Initialize pynvml once. Thread-safe via double-checked locking.

    Returns True if pynvml is ready, False if unavailable or init failed.

    Stage C amend BACKEND-B-014: caches runtime-failure state in
    :data:`_nvml_runtime_failed` so a broken NVML env (driver mismatch,
    WSL bug, sensor failure) returns False fast on subsequent calls
    instead of re-attempting nvmlInit() forever. The first failure is
    logged at WARN with the explanatory message + recovery hint; later
    calls short-circuit silently. Operators who fix the env mid-session
    can call :func:`reset_nvml_state` to re-arm the init path.
    """
    global _nvml_initialized, _nvml_runtime_failed, _nvml_runtime_failure_logged

    if _nvml_initialized:
        return True
    if _nvml_runtime_failed:
        # Cached failure — short-circuit to avoid log spam.
        return False

    with _nvml_init_lock:
        # Double-check after acquiring lock
        if _nvml_initialized:
            return True
        if _nvml_runtime_failed:
            return False

        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml_initialized = True
            atexit.register(_shutdown_nvml)
            logger.debug("pynvml initialized (init-once pattern)")
            return True
        except ImportError:
            global _nvml_unavailable_logged
            if not _nvml_unavailable_logged:
                logger.info("pynvml not installed - temperature/power monitoring unavailable. Install with: pip install pynvml")
                _nvml_unavailable_logged = True
            return False
        except Exception as e:
            # Stage C amend BACKEND-B-014: cache the runtime failure so
            # subsequent calls return fast. Log loud the FIRST time, then
            # stay quiet to avoid swamping the monitoring loop output.
            _nvml_runtime_failed = True
            if not _nvml_runtime_failure_logged:
                logger.warning(
                    f"pynvml runtime init failed: {e}. "
                    f"Temperature/power monitoring will be unavailable for "
                    f"the rest of this process. Common causes: driver / "
                    f"library version mismatch, WSL without GPU passthrough, "
                    f"or NVIDIA driver hung. Call "
                    f"backpropagate.gpu_safety.reset_nvml_state() to retry "
                    f"after fixing the env."
                )
                _nvml_runtime_failure_logged = True
            return False


def _shutdown_nvml() -> None:
    """Atexit handler to cleanly shut down pynvml."""
    global _nvml_initialized

    if not _nvml_initialized:
        return

    try:
        import pynvml

        pynvml.nvmlShutdown()
        _nvml_initialized = False
        logger.debug("pynvml shut down via atexit")
    except Exception as e:
        logger.debug(f"pynvml shutdown error: {e}")

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
    temperature_c: float | None = None
    temperature_max_c: float | None = None

    # Memory
    # NOTE on scope: vram_used_gb / vram_free_gb / vram_percent are
    # PER-PROCESS — they reflect what the current PyTorch CUDA context has
    # reserved via torch.cuda.memory_reserved(), not what the GPU as a whole
    # has allocated. Other processes' allocations are invisible to these
    # fields. For system-wide VRAM, query nvidia-smi or the (future)
    # system-vram helper. vram_total_gb IS the system GPU total (from
    # cudaDeviceProp.total_memory) so percent math against it is "what
    # share of total GPU VRAM has THIS process reserved", not "how full is
    # the GPU overall".
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    vram_free_gb: float = 0.0
    vram_percent: float = 0.0

    # Power
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    power_percent: float | None = None

    # Utilization
    gpu_utilization: int | None = None
    memory_utilization: int | None = None

    # Computed condition
    condition: GPUCondition = GPUCondition.UNKNOWN
    condition_reason: str = ""

    # Timestamp
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_gpu_status(device_index: int = 0, config: GPUSafetyConfig | None = None) -> GPUStatus:
    """
    Get current GPU status with safety evaluation.

    Args:
        device_index: GPU device index (default 0)
        config: Safety config for threshold evaluation

    Returns:
        GPUStatus with current readings and safety condition. The
        ``vram_*`` fields are PER-PROCESS only (sourced from
        ``torch.cuda.memory_reserved``) — they do not see allocations
        from other processes sharing the GPU. Temperature, power, and
        utilization come from pynvml and ARE system-wide. For
        system-wide VRAM, use ``nvidia-smi`` or the future system-vram
        helper.
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
    # Uses init-once pattern — nvmlInit() called once, nvmlShutdown() via atexit
    if _ensure_nvml_initialized():
        try:
            import pynvml

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
        # TRAINER-A-009: the card's OWN hardware shutdown threshold
        # (status.temperature_max_c, from
        # nvmlDeviceGetTemperatureThreshold(SHUTDOWN)) was queried in
        # get_gpu_status but never consulted — a dead safety signal. A
        # statically-configured temp_emergency (default 95C) can sit ABOVE a
        # specific card's real shutdown point (some mobile / older GPUs shut
        # down at 90-92C), so a card already in its hardware-protection zone
        # would only register CRITICAL under the static thresholds. Factor the
        # real threshold in as an additional EMERGENCY floor: if we are within
        # HW_SHUTDOWN_MARGIN_C of (or past) the card's shutdown temp, that is
        # an emergency regardless of the configured value. Skipped when the
        # threshold is unknown (None), so software-only / mocked paths keep the
        # pure static-threshold behavior.
        if (
            status.temperature_max_c is not None
            and status.temperature_max_c > 0
            and status.temperature_c >= status.temperature_max_c - _HW_SHUTDOWN_MARGIN_C
        ):
            return (
                GPUCondition.EMERGENCY,
                f"Temperature EMERGENCY: {status.temperature_c}C is within "
                f"{_HW_SHUTDOWN_MARGIN_C}C of the card's hardware shutdown "
                f"threshold ({status.temperature_max_c}C)",
            )
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
    config: GPUSafetyConfig | None = None,
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
        logger.warning(
            "GPU safety status unknown (no CUDA GPU detected or monitoring unavailable). "
            "Training will proceed without temperature safety checks. "
            "For GPU monitoring: pip install pynvml"
        )
        return True

    logger.error(f"GPU unsafe: {status.condition_reason}")
    return False


def wait_for_safe_gpu(
    device_index: int = 0,
    config: GPUSafetyConfig | None = None,
    max_wait_seconds: float = 300.0,
    check_interval: float = 10.0,
    abort_event: threading.Event | None = None,
) -> bool:
    """
    Wait for GPU to reach safe conditions.

    Useful after a critical condition is detected to allow cooldown.

    .. warning::
        This function blocks the calling thread for up to ``max_wait_seconds``.
        Call it from a background thread, never from a UI event handler or an
        async event loop directly. Pass ``abort_event`` to make the wait
        interruptible (see below).

    Args:
        device_index: GPU device index
        config: Safety configuration
        max_wait_seconds: Maximum time to wait
        check_interval: Seconds between checks
        abort_event: Stage C BACKEND-B-014 humanization — optional
            ``threading.Event``. When set externally (e.g. from
            ``MultiRunTrainer.abort()``), the wait returns ``False``
            promptly without waiting out the remaining
            ``check_interval``. Pre-fix the wait was uninterruptible:
            an operator clicking "Abort" during a cooldown wait saw no
            effect until the cooldown finished. When ``None`` (the
            default), behavior is byte-identical to pre-fix
            (``time.sleep`` poll).

    Returns:
        True if GPU became safe, False if timeout OR abort_event was set
    """
    config = config or GPUSafetyConfig()
    start_time = time.time()

    logger.info(
        "Waiting for GPU to reach safe temperature "
        "(max_wait=%.0fs, check_interval=%.0fs, device=%d)...",
        max_wait_seconds,
        check_interval,
        device_index,
    )

    while (time.time() - start_time) < max_wait_seconds:
        # Stage C BACKEND-B-014 humanization: honor the abort signal at
        # the top of the loop so an operator-issued abort during a
        # multi-minute cooldown returns promptly with a clear log line.
        if abort_event is not None and abort_event.is_set():
            elapsed = time.time() - start_time
            logger.info(
                "GPU cooldown wait aborted by external signal after "
                "%.0fs (max_wait_seconds=%.0f).",
                elapsed,
                max_wait_seconds,
            )
            return False

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

        # Stage C BACKEND-B-014 humanization: when an abort_event is
        # provided, use Event.wait(check_interval) so the abort cuts the
        # current sleep short. The return value of Event.wait() is True
        # iff the event was set during the wait; we re-check at the top
        # of the loop for consistency with the no-abort_event branch.
        if abort_event is not None:
            if abort_event.wait(check_interval):
                continue  # let the top-of-loop check handle the return
        else:
            time.sleep(check_interval)

    # Stage C BACKEND-B-014 humanization: name the operator's options in
    # priority order — disk-cheap fixes first (close apps, raise interval),
    # then config knobs, then hardware.
    logger.error(
        "GPU did not reach safe temperature within %.0fs. "
        "Next steps (in order of cost): "
        "(1) close other GPU applications and re-try; "
        "(2) raise --cooldown-timeout or MultiRunConfig.max_pause_seconds "
        "if the GPU is in fact cooling but slower than expected; "
        "(3) reduce --batch-size to lower steady-state heat; "
        "(4) verify case airflow / fan curves with `nvidia-smi -q -d "
        "TEMPERATURE` over a stable workload.",
        max_wait_seconds,
    )
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
        config: GPUSafetyConfig | None = None,
        device_index: int = 0,
        on_warning: Callable[[GPUStatus], None] | None = None,
        on_critical: Callable[[GPUStatus], None] | None = None,
        on_emergency: Callable[[GPUStatus], None] | None = None,
        on_status: Callable[[GPUStatus], None] | None = None,
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

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        self._status_history: collections.deque[GPUStatus] = collections.deque(maxlen=100)

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

    def get_latest_status(self) -> GPUStatus | None:
        """Get most recent GPU status."""
        if self._status_history:
            return self._status_history[-1]
        return None

    def get_status_history(self) -> list[GPUStatus]:
        """Get status history."""
        return list(self._status_history)

    @property
    def is_emergency(self) -> bool:
        """Check if emergency condition was triggered."""
        return self._emergency_triggered

    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in background thread).

        Stage C amend CORE-B-008: a persistently-failing ``get_gpu_status``
        (driver crash, nvidia-smi gone, sensor wedged) previously logged a
        full ``GPU monitor error`` line on EVERY iteration — once per
        ``check_interval`` for the whole session, flooding logs and burying
        real signal. We now count consecutive failures: log the first at
        error level, escalate ONCE at the threshold to make clear that
        thermal safety is effectively disabled (no fresh readings ->
        pause/emergency callbacks can never fire), then fall back to debug
        for the steady-state spam. Any successful poll resets the counter and
        logs a one-line recovery so the operator sees monitoring came back.
        """
        consecutive_failures = 0
        # After this many back-to-back failures, escalate once: the monitor
        # has produced no usable reading for ~threshold * check_interval and
        # cannot enforce any thermal limit.
        _FAILURE_ESCALATE_AT = 3
        while not self._stop_event.is_set():
            try:
                status = get_gpu_status(self.device_index, self.config)

                if consecutive_failures > 0:
                    logger.warning(
                        "GPU monitor recovered after %d consecutive failed "
                        "poll(s); thermal safety re-armed.",
                        consecutive_failures,
                    )
                    consecutive_failures = 0

                # Store in history (deque with maxlen handles eviction)
                self._status_history.append(status)

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
                consecutive_failures += 1
                if consecutive_failures == 1:
                    logger.error(f"GPU monitor error: {e}")
                elif consecutive_failures == _FAILURE_ESCALATE_AT:
                    logger.error(
                        "GPU monitor has failed %d consecutive polls "
                        "(latest: %s). Thermal safety is effectively "
                        "DISABLED — no fresh GPU readings means overheat "
                        "auto-pause / emergency-abort cannot trigger. "
                        "Verify the driver / nvidia-smi / pynvml install. "
                        "Suppressing further per-poll error logs until a "
                        "reading succeeds.",
                        consecutive_failures,
                        e,
                    )
                else:
                    # Steady-state spam suppression: keep a debug breadcrumb
                    # but stop flooding error logs every check_interval.
                    logger.debug(
                        f"GPU monitor still failing "
                        f"(consecutive={consecutive_failures}): {e}"
                    )

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
            # Stage C humanization: name what CRITICAL means in operator
            # terms — temperature crossed the configured critical
            # threshold, training auto-pauses if pause_on_overheat=True,
            # the operator's next step is to verify cooling and
            # potentially lower batch_size for next session.
            logger.error(
                "GPU CRITICAL: %s. Training will auto-pause if "
                "MultiRunConfig.pause_on_overheat=True (default). To "
                "recover: improve case airflow / fan curves, lower "
                "--batch-size for next session, or raise GPUSafetyConfig "
                "critical_temp_c if the threshold is overly conservative "
                "for your card.",
                status.condition_reason,
            )

            if self.config.log_warnings:
                logger.warning(
                    "Critical GPU condition #%d this session: "
                    "consider pausing training if not already auto-paused.",
                    self._critical_count,
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
