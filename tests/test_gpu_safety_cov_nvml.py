"""Coverage tests for ``backpropagate.gpu_safety``: NVML lifecycle, the real
``get_gpu_status`` read path, the cooldown wait, and the monitor's callback
isolation.

Mock boundary: the hardware. ``pynvml`` (not installed on the CI/dev boxes) is
replaced by an in-process fake module injected through ``sys.modules`` and
``torch.cuda`` entry points are monkeypatched. Everything above that boundary
(``backpropagate.gpu_safety``) runs for real.
"""

from __future__ import annotations

import logging
import sys
import threading
import types

import pytest

from backpropagate import gpu_safety
from backpropagate.gpu_safety import (
    GPUCondition,
    GPUMonitor,
    GPUSafetyConfig,
    GPUStatus,
    check_gpu_safe,
    format_gpu_status,
    get_gpu_status,
    install_pynvml_hint,
    reset_nvml_state,
    wait_for_safe_gpu,
)


class _NVMLError(Exception):
    """Stand-in for ``pynvml.NVMLError``."""


def make_fake_pynvml(
    *,
    temp=70,
    temp_max=100,
    power_mw=200_000,
    limit_mw=400_000,
    util=(55, 33),
    init_exc=None,
    fail=(),
    fail_with=_NVMLError,
):
    """Build a fake ``pynvml`` module.

    ``fail`` names which queries raise ``fail_with`` (temp, temp_max, power,
    util, handle).
    """
    m = types.ModuleType("pynvml")
    m.NVMLError = _NVMLError
    m.NVML_TEMPERATURE_GPU = 0
    m.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN = 1
    m.init_calls = 0
    m.shutdown_calls = 0

    def nvmlInit():
        m.init_calls += 1
        if init_exc is not None:
            raise init_exc

    def nvmlShutdown():
        m.shutdown_calls += 1

    def _maybe(name):
        if name in fail:
            raise fail_with(f"{name} failed")

    def nvmlDeviceGetHandleByIndex(i):
        _maybe("handle")
        return ("handle", i)

    def nvmlDeviceGetTemperature(h, kind):
        _maybe("temp")
        return temp

    def nvmlDeviceGetTemperatureThreshold(h, kind):
        _maybe("temp_max")
        return temp_max

    def nvmlDeviceGetPowerUsage(h):
        _maybe("power")
        return power_mw

    def nvmlDeviceGetPowerManagementLimit(h):
        return limit_mw

    def nvmlDeviceGetUtilizationRates(h):
        _maybe("util")
        return types.SimpleNamespace(gpu=util[0], memory=util[1])

    m.nvmlInit = nvmlInit
    m.nvmlShutdown = nvmlShutdown
    m.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
    m.nvmlDeviceGetTemperature = nvmlDeviceGetTemperature
    m.nvmlDeviceGetTemperatureThreshold = nvmlDeviceGetTemperatureThreshold
    m.nvmlDeviceGetPowerUsage = nvmlDeviceGetPowerUsage
    m.nvmlDeviceGetPowerManagementLimit = nvmlDeviceGetPowerManagementLimit
    m.nvmlDeviceGetUtilizationRates = nvmlDeviceGetUtilizationRates
    return m


@pytest.fixture(autouse=True)
def _isolate_nvml_state(monkeypatch):
    """Reset the module-level NVML caches and keep real atexit untouched."""
    monkeypatch.setattr(gpu_safety, "_nvml_initialized", False)
    monkeypatch.setattr(gpu_safety, "_nvml_runtime_failed", False)
    monkeypatch.setattr(gpu_safety, "_nvml_runtime_failure_logged", False)
    monkeypatch.setattr(gpu_safety, "_nvml_unavailable_logged", False)
    registered = []
    monkeypatch.setattr(gpu_safety.atexit, "register", lambda fn, *a, **k: registered.append(fn))
    return registered


@pytest.fixture
def fake_cuda(monkeypatch):
    """Pretend one 16 GiB CUDA device exists (torch.cuda mocked)."""
    import torch

    props = types.SimpleNamespace(total_memory=16 * 1024**3)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "Fake GPU")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i=0: props)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i=0: 1 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda i=0: 4 * 1024**3)


@pytest.fixture
def install_nvml(monkeypatch):
    def _install(module):
        monkeypatch.setitem(sys.modules, "pynvml", module)
        return module

    return _install


# ---------------------------------------------------------------------------
# NVML lifecycle
# ---------------------------------------------------------------------------

class TestResetNvmlState:
    def test_clears_cached_failure_and_logs(self, monkeypatch, caplog):
        monkeypatch.setattr(gpu_safety, "_nvml_runtime_failed", True)
        monkeypatch.setattr(gpu_safety, "_nvml_runtime_failure_logged", True)
        with caplog.at_level(logging.INFO, logger="backpropagate.gpu_safety"):
            reset_nvml_state()
        assert gpu_safety._nvml_runtime_failed is False
        assert gpu_safety._nvml_runtime_failure_logged is False
        assert "clearing cached NVML runtime-failure flag" in caplog.text

    def test_noop_and_silent_when_no_failure_cached(self, caplog):
        with caplog.at_level(logging.INFO, logger="backpropagate.gpu_safety"):
            reset_nvml_state()
        assert gpu_safety._nvml_runtime_failed is False
        assert "clearing cached" not in caplog.text


class TestEnsureNvmlInitialized:
    def test_success_initializes_once_and_registers_atexit(
        self, install_nvml, _isolate_nvml_state
    ):
        fake = install_nvml(make_fake_pynvml())
        assert gpu_safety._ensure_nvml_initialized() is True
        assert gpu_safety._nvml_initialized is True
        assert _isolate_nvml_state == [gpu_safety._shutdown_nvml]
        # Init-once: the second call short-circuits.
        assert gpu_safety._ensure_nvml_initialized() is True
        assert fake.init_calls == 1

    def test_missing_pynvml_returns_false_and_logs_hint_once(
        self, monkeypatch, caplog
    ):
        monkeypatch.setitem(sys.modules, "pynvml", None)  # import -> ImportError
        with caplog.at_level(logging.INFO, logger="backpropagate.gpu_safety"):
            assert gpu_safety._ensure_nvml_initialized() is False
            assert gpu_safety._ensure_nvml_initialized() is False
        assert caplog.text.count("pynvml not installed") == 1
        assert gpu_safety._nvml_unavailable_logged is True
        # ImportError is not a "runtime failure": nothing is cached.
        assert gpu_safety._nvml_runtime_failed is False

    def test_runtime_failure_is_cached_and_logged_once(
        self, install_nvml, caplog
    ):
        fake = install_nvml(make_fake_pynvml(init_exc=RuntimeError("driver mismatch")))
        with caplog.at_level(logging.WARNING, logger="backpropagate.gpu_safety"):
            assert gpu_safety._ensure_nvml_initialized() is False
            assert gpu_safety._ensure_nvml_initialized() is False
        assert fake.init_calls == 1  # cached: nvmlInit not retried
        assert caplog.text.count("pynvml runtime init failed: driver mismatch") == 1
        assert "reset_nvml_state()" in caplog.text
        assert gpu_safety._nvml_runtime_failed is True

    def test_reset_rearms_init_after_failure(self, install_nvml):
        install_nvml(make_fake_pynvml(init_exc=RuntimeError("boom")))
        assert gpu_safety._ensure_nvml_initialized() is False
        good = install_nvml(make_fake_pynvml())
        reset_nvml_state()
        assert gpu_safety._ensure_nvml_initialized() is True
        assert good.init_calls == 1

    def test_runtime_failure_not_relogged_when_flag_already_set(
        self, install_nvml, monkeypatch, caplog
    ):
        # failure flag cleared but "logged" flag still set -> stays quiet.
        install_nvml(make_fake_pynvml(init_exc=RuntimeError("again")))
        monkeypatch.setattr(gpu_safety, "_nvml_runtime_failure_logged", True)
        with caplog.at_level(logging.WARNING, logger="backpropagate.gpu_safety"):
            assert gpu_safety._ensure_nvml_initialized() is False
        assert "pynvml runtime init failed" not in caplog.text
        assert gpu_safety._nvml_runtime_failed is True

    def test_double_check_inside_lock_sees_concurrent_init(self, monkeypatch):
        """A thread that loses the lock race returns True without re-init."""

        class FlipLock:
            def __enter__(self_inner):
                gpu_safety._nvml_initialized = True  # another thread won
                return self_inner

            def __exit__(self_inner, *a):
                return False

        monkeypatch.setattr(gpu_safety, "_nvml_init_lock", FlipLock())
        assert gpu_safety._ensure_nvml_initialized() is True

    def test_double_check_inside_lock_sees_concurrent_failure(self, monkeypatch):
        class FlipLock:
            def __enter__(self_inner):
                gpu_safety._nvml_runtime_failed = True  # another thread failed
                return self_inner

            def __exit__(self_inner, *a):
                return False

        monkeypatch.setattr(gpu_safety, "_nvml_init_lock", FlipLock())
        assert gpu_safety._ensure_nvml_initialized() is False


class TestShutdownNvml:
    def test_noop_when_not_initialized(self, install_nvml):
        fake = install_nvml(make_fake_pynvml())
        gpu_safety._shutdown_nvml()
        assert fake.shutdown_calls == 0

    def test_shutdown_clears_flag(self, install_nvml, monkeypatch):
        fake = install_nvml(make_fake_pynvml())
        monkeypatch.setattr(gpu_safety, "_nvml_initialized", True)
        gpu_safety._shutdown_nvml()
        assert fake.shutdown_calls == 1
        assert gpu_safety._nvml_initialized is False

    def test_shutdown_error_is_swallowed_and_flag_kept(
        self, install_nvml, monkeypatch, caplog
    ):
        fake = install_nvml(make_fake_pynvml())

        def boom():
            raise RuntimeError("nvml gone")

        fake.nvmlShutdown = boom
        monkeypatch.setattr(gpu_safety, "_nvml_initialized", True)
        with caplog.at_level(logging.DEBUG, logger="backpropagate.gpu_safety"):
            gpu_safety._shutdown_nvml()  # must not raise
        assert gpu_safety._nvml_initialized is True
        assert "pynvml shutdown error: nvml gone" in caplog.text


# ---------------------------------------------------------------------------
# get_gpu_status read path
# ---------------------------------------------------------------------------

class TestGetGpuStatusReadPath:
    def test_torch_query_failure_reports_unknown(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def boom(i=0):
            raise RuntimeError("cuda driver error")

        monkeypatch.setattr(torch.cuda, "get_device_name", boom)
        status = get_gpu_status()
        assert status.available is True  # set before the failing call
        assert status.condition == GPUCondition.UNKNOWN
        assert status.condition_reason == "PyTorch error: cuda driver error"

    def test_full_reading_with_nvml(self, fake_cuda, install_nvml):
        install_nvml(make_fake_pynvml(temp=65, temp_max=100, power_mw=150_000,
                                      limit_mw=300_000, util=(80, 40)))
        before = gpu_safety.time.time()
        s = get_gpu_status(0)
        assert s.available and s.device_name == "Fake GPU"
        assert s.vram_total_gb == pytest.approx(16.0)
        assert s.vram_used_gb == pytest.approx(4.0)
        assert s.vram_free_gb == pytest.approx(12.0)
        assert s.vram_percent == pytest.approx(25.0)
        assert s.temperature_c == 65
        assert s.temperature_max_c == 100
        assert s.power_draw_w == pytest.approx(150.0)
        assert s.power_limit_w == pytest.approx(300.0)
        assert s.power_percent == pytest.approx(50.0)
        assert (s.gpu_utilization, s.memory_utilization) == (80, 40)
        assert s.condition == GPUCondition.SAFE
        assert s.timestamp >= before

    def test_hot_card_escalates_through_real_read(self, fake_cuda, install_nvml):
        install_nvml(make_fake_pynvml(temp=96))
        s = get_gpu_status(0)
        assert s.condition == GPUCondition.EMERGENCY
        assert "96C" in s.condition_reason

    def test_zero_power_limit_leaves_percent_unset(self, fake_cuda, install_nvml):
        install_nvml(make_fake_pynvml(power_mw=100_000, limit_mw=0))
        s = get_gpu_status(0)
        assert s.power_draw_w == pytest.approx(100.0)
        assert s.power_limit_w == 0
        assert s.power_percent is None

    @pytest.mark.parametrize("query", ["temp", "temp_max", "power", "util"])
    @pytest.mark.parametrize("exc", [_NVMLError, ValueError])
    def test_individual_query_failures_degrade_only_that_field(
        self, fake_cuda, install_nvml, caplog, query, exc
    ):
        install_nvml(make_fake_pynvml(fail=(query,), fail_with=exc))
        with caplog.at_level(logging.DEBUG, logger="backpropagate.gpu_safety"):
            s = get_gpu_status(0)
        assert s.available is True
        none_fields = {
            "temp": ["temperature_c"],
            "temp_max": ["temperature_max_c"],
            "power": ["power_draw_w", "power_limit_w", "power_percent"],
            "util": ["gpu_utilization", "memory_utilization"],
        }
        for name in none_fields[query]:
            assert getattr(s, name) is None, name
        # The other metrics are still read.
        if query != "temp":
            assert s.temperature_c == 70
        if query != "util":
            assert s.gpu_utilization == 55
        assert f"{query} failed" in caplog.text
        # A final condition is still produced from what was read.
        assert s.condition in (GPUCondition.SAFE, GPUCondition.WARM)

    def test_handle_failure_is_swallowed(self, fake_cuda, install_nvml, caplog):
        install_nvml(make_fake_pynvml(fail=("handle",)))
        with caplog.at_level(logging.DEBUG, logger="backpropagate.gpu_safety"):
            s = get_gpu_status(0)
        assert "pynvml query failed: handle failed" in caplog.text
        assert s.temperature_c is None
        assert s.condition == GPUCondition.SAFE  # VRAM-only evaluation

    def test_nvml_unavailable_still_reports_vram(self, fake_cuda, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        s = get_gpu_status(0)
        assert s.available is True
        assert s.temperature_c is None and s.power_draw_w is None
        assert s.vram_percent == pytest.approx(25.0)


class TestCheckGpuSafeWithRealRead:
    def test_critical_temperature_is_unsafe(self, fake_cuda, install_nvml, caplog):
        install_nvml(make_fake_pynvml(temp=91))
        with caplog.at_level(logging.ERROR, logger="backpropagate.gpu_safety"):
            assert check_gpu_safe(0) is False
        assert "GPU unsafe" in caplog.text

    def test_unknown_allows_with_warning(self, monkeypatch, caplog):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with caplog.at_level(logging.WARNING, logger="backpropagate.gpu_safety"):
            assert check_gpu_safe(0) is True
        assert "safety status unknown" in caplog.text


# ---------------------------------------------------------------------------
# _evaluate_condition leftovers
# ---------------------------------------------------------------------------

class TestEvaluateConditionPower:
    def _status(self, **kw):
        return GPUStatus(available=True, vram_percent=10.0, **kw)

    def test_power_at_tdp_is_warning(self):
        cond, reason = gpu_safety._evaluate_condition(
            self._status(power_percent=101.0), GPUSafetyConfig())
        assert cond == GPUCondition.WARNING
        assert reason.startswith("Power at TDP")

    def test_power_high_is_warm(self):
        cond, reason = gpu_safety._evaluate_condition(
            self._status(power_percent=96.0), GPUSafetyConfig())
        assert cond == GPUCondition.WARM
        assert reason.startswith("Power high")

    def test_power_below_thresholds_is_safe(self):
        cond, _ = gpu_safety._evaluate_condition(
            self._status(power_percent=50.0), GPUSafetyConfig())
        assert cond == GPUCondition.SAFE

    def test_elevated_temperature_is_warm(self):
        cond, reason = gpu_safety._evaluate_condition(
            self._status(temperature_c=72.0), GPUSafetyConfig())
        assert cond == GPUCondition.WARM
        assert "72.0C" in reason

    def test_zero_hw_threshold_is_ignored(self):
        cond, _ = gpu_safety._evaluate_condition(
            self._status(temperature_c=60.0, temperature_max_c=0), GPUSafetyConfig())
        assert cond == GPUCondition.SAFE


# ---------------------------------------------------------------------------
# wait_for_safe_gpu
# ---------------------------------------------------------------------------

def _status(cond, temp=None):
    return GPUStatus(available=True, condition=cond, temperature_c=temp)


class TestWaitForSafeGpu:
    def test_abort_event_already_set_returns_false_immediately(
        self, monkeypatch, caplog
    ):
        calls = []
        monkeypatch.setattr(gpu_safety, "get_gpu_status",
                            lambda *a, **k: calls.append(1) or _status(GPUCondition.SAFE))
        ev = threading.Event()
        ev.set()
        with caplog.at_level(logging.INFO, logger="backpropagate.gpu_safety"):
            assert wait_for_safe_gpu(max_wait_seconds=30, check_interval=0.01,
                                     abort_event=ev) is False
        assert calls == []  # aborted before any read
        assert "aborted by external signal" in caplog.text

    def test_abort_during_wait_cuts_sleep_short(self, monkeypatch):
        reads = []

        def fake(*a, **k):
            reads.append(1)
            return _status(GPUCondition.CRITICAL, temp=92)

        monkeypatch.setattr(gpu_safety, "get_gpu_status", fake)
        ev = threading.Event()
        threading.Timer(0.05, ev.set).start()
        t0 = gpu_safety.time.time()
        assert wait_for_safe_gpu(max_wait_seconds=30, check_interval=10.0,
                                 abort_event=ev) is False
        assert gpu_safety.time.time() - t0 < 5.0  # not the 10 s interval
        assert reads == [1]

    def test_becomes_safe_with_abort_event_attached(self, monkeypatch):
        seq = iter([_status(GPUCondition.CRITICAL, 92), _status(GPUCondition.WARM, 70)])
        monkeypatch.setattr(gpu_safety, "get_gpu_status", lambda *a, **k: next(seq))
        ev = threading.Event()
        assert wait_for_safe_gpu(max_wait_seconds=5, check_interval=0.01,
                                 abort_event=ev) is True

    def test_timeout_logs_next_steps_and_logs_no_temperature_branch(
        self, monkeypatch, caplog
    ):
        monkeypatch.setattr(gpu_safety, "get_gpu_status",
                            lambda *a, **k: _status(GPUCondition.CRITICAL, temp=None))
        with caplog.at_level(logging.INFO, logger="backpropagate.gpu_safety"):
            assert wait_for_safe_gpu(max_wait_seconds=0.05, check_interval=0.01) is False
        assert "Waiting for safe GPU" in caplog.text
        assert "did not reach safe temperature" in caplog.text

    def test_cooling_message_includes_temperature(self, monkeypatch, caplog):
        monkeypatch.setattr(gpu_safety, "get_gpu_status",
                            lambda *a, **k: _status(GPUCondition.CRITICAL, temp=93))
        with caplog.at_level(logging.INFO, logger="backpropagate.gpu_safety"):
            assert wait_for_safe_gpu(max_wait_seconds=0.05, check_interval=0.01) is False
        assert "GPU cooling: 93C" in caplog.text


# ---------------------------------------------------------------------------
# GPUMonitor
# ---------------------------------------------------------------------------

class TestMonitorLifecycle:
    def test_start_twice_warns_and_keeps_single_thread(self, monkeypatch, caplog):
        monkeypatch.setattr(gpu_safety, "get_gpu_status",
                            lambda *a, **k: _status(GPUCondition.SAFE))
        m = GPUMonitor(config=GPUSafetyConfig(check_interval=0.01))
        m.start()
        first = m._thread
        try:
            with caplog.at_level(logging.WARNING, logger="backpropagate.gpu_safety"):
                m.start()
            assert "already running" in caplog.text
            assert m._thread is first
        finally:
            m.stop()
        assert m._thread is None

    def test_latest_status_none_then_populated(self):
        m = GPUMonitor()
        assert m.get_latest_status() is None
        s = _status(GPUCondition.SAFE)
        m._status_history.append(s)
        assert m.get_latest_status() is s
        assert m.get_status_history() == [s]
        assert m.is_emergency is False


class TestMonitorLoop:
    def _run_once(self, m):
        """Drive exactly one loop iteration synchronously."""
        m._stop_event.wait = lambda timeout=None: m._stop_event.set()
        m._monitor_loop()

    def test_on_status_exception_is_isolated(self, monkeypatch, caplog):
        monkeypatch.setattr(gpu_safety, "get_gpu_status",
                            lambda *a, **k: _status(GPUCondition.SAFE))

        def bad(status):
            raise ValueError("display crashed")

        m = GPUMonitor(on_status=bad)
        with caplog.at_level(logging.WARNING, logger="backpropagate.gpu_safety"):
            self._run_once(m)
        assert "on_status callback raised exception: ValueError: display crashed" in caplog.text
        assert len(m.get_status_history()) == 1

    def test_paused_monitor_records_history_but_skips_callbacks(self, monkeypatch):
        monkeypatch.setattr(gpu_safety, "get_gpu_status",
                            lambda *a, **k: _status(GPUCondition.EMERGENCY))
        seen = []
        m = GPUMonitor(on_status=seen.append, on_emergency=seen.append)
        m.pause()
        self._run_once(m)
        assert seen == []
        assert m.is_emergency is False
        assert len(m.get_status_history()) == 1

    def test_failure_log_levels_and_recovery(self, monkeypatch, caplog):
        n = {"i": 0}
        plan = [RuntimeError("f1"), RuntimeError("f2"), RuntimeError("f3"),
                RuntimeError("f4"), _status(GPUCondition.SAFE)]

        def fake(*a, **k):
            item = plan[n["i"]]
            n["i"] += 1
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(gpu_safety, "get_gpu_status", fake)
        m = GPUMonitor()
        waits = {"n": 0}

        def fake_wait(timeout=None):
            waits["n"] += 1
            if waits["n"] >= len(plan):
                m._stop_event.set()

        m._stop_event.wait = fake_wait
        with caplog.at_level(logging.DEBUG, logger="backpropagate.gpu_safety"):
            m._monitor_loop()
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("GPU monitor error: f1" in r.getMessage() for r in errors)
        assert any("Thermal safety is effectively" in r.getMessage() for r in errors)
        assert len(errors) == 2  # first + escalation only
        assert "GPU monitor still failing (consecutive=2): f2" in caplog.text
        assert "GPU monitor still failing (consecutive=4): f4" in caplog.text
        assert "recovered after 4 consecutive failed poll(s)" in caplog.text


class TestHandleCondition:
    def _s(self, cond, reason="why"):
        return GPUStatus(available=True, condition=cond, condition_reason=reason)

    def test_emergency_sets_flag_and_isolates_callback_error(self, caplog):
        def bad(s):
            raise RuntimeError("abort handler broke")

        m = GPUMonitor(on_emergency=bad)
        with caplog.at_level(logging.WARNING, logger="backpropagate.gpu_safety"):
            m._handle_condition(self._s(GPUCondition.EMERGENCY, "too hot"))
        assert m.is_emergency is True
        assert "GPU EMERGENCY: too hot" in caplog.text
        assert "on_emergency callback raised exception: RuntimeError: abort handler broke" in caplog.text

    def test_critical_counts_and_isolates_callback_error(self, caplog):
        def bad(s):
            raise RuntimeError("pause handler broke")

        m = GPUMonitor(on_critical=bad)
        with caplog.at_level(logging.WARNING, logger="backpropagate.gpu_safety"):
            m._handle_condition(self._s(GPUCondition.CRITICAL))
            m._handle_condition(self._s(GPUCondition.CRITICAL))
        assert m._critical_count == 2
        assert "Critical GPU condition #2 this session" in caplog.text
        assert "on_critical callback raised exception" in caplog.text

    def test_critical_without_log_warnings_skips_session_note(self, caplog):
        m = GPUMonitor(config=GPUSafetyConfig(log_warnings=False))
        with caplog.at_level(logging.WARNING, logger="backpropagate.gpu_safety"):
            m._handle_condition(self._s(GPUCondition.CRITICAL))
        assert "Critical GPU condition #" not in caplog.text
        assert m._critical_count == 1

    def test_warning_callback_invoked_and_isolated(self, caplog):
        got = []
        m = GPUMonitor(on_warning=got.append)
        m._handle_condition(self._s(GPUCondition.WARNING, "warm-ish"))
        assert [s.condition_reason for s in got] == ["warm-ish"]

        def bad(s):
            raise KeyError("x")

        m2 = GPUMonitor(on_warning=bad)
        with caplog.at_level(logging.WARNING, logger="backpropagate.gpu_safety"):
            m2._handle_condition(self._s(GPUCondition.WARNING))
        assert "on_warning callback raised exception: KeyError" in caplog.text

    def test_warning_log_suppressed_when_log_warnings_false(self, caplog):
        m = GPUMonitor(config=GPUSafetyConfig(log_warnings=False))
        with caplog.at_level(logging.WARNING, logger="backpropagate.gpu_safety"):
            m._handle_condition(self._s(GPUCondition.WARNING))
        assert "GPU WARNING" not in caplog.text

    def test_return_to_safe_resets_critical_counter(self, caplog):
        m = GPUMonitor()
        m._critical_count = 3
        with caplog.at_level(logging.INFO, logger="backpropagate.gpu_safety"):
            m._handle_condition(self._s(GPUCondition.SAFE))
        assert m._critical_count == 0
        assert "GPU returned to safe conditions" in caplog.text

    def test_safe_with_zero_counter_is_silent(self, caplog):
        m = GPUMonitor()
        with caplog.at_level(logging.INFO, logger="backpropagate.gpu_safety"):
            m._handle_condition(self._s(GPUCondition.WARM))
        assert "returned to safe" not in caplog.text


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_format_includes_power_when_known(self):
        s = GPUStatus(available=True, device_name="X", temperature_c=60,
                      vram_total_gb=16, vram_used_gb=4, vram_percent=25,
                      power_draw_w=212.6, condition=GPUCondition.SAFE)
        out = format_gpu_status(s)
        assert out == "GPU: X | Temp: 60C | VRAM: 4.0/16.0 GB (25.0%) | Power: 213W | Status: SAFE"

    def test_install_hint_names_both_packages(self):
        hint = install_pynvml_hint()
        assert "pip install pynvml" in hint
        assert "nvidia-ml-py" in hint


class TestMonitorPauseResumeStop:
    def test_stop_without_start_is_safe(self, caplog):
        m = GPUMonitor()
        with caplog.at_level(logging.INFO, logger="backpropagate.gpu_safety"):
            m.stop()
        assert m._stop_event.is_set()
        assert m._thread is None
        assert "GPU monitor stopped" in caplog.text

    def test_resume_rearms_callbacks_after_pause(self, monkeypatch):
        monkeypatch.setattr(gpu_safety, "get_gpu_status",
                            lambda *a, **k: _status(GPUCondition.SAFE))
        seen = []
        m = GPUMonitor(on_status=seen.append)
        m.pause()
        assert not m._pause_event.is_set()
        m.resume()
        assert m._pause_event.is_set()
        m._stop_event.wait = lambda timeout=None: m._stop_event.set()
        m._monitor_loop()
        assert len(seen) == 1

    def test_emergency_without_callback_still_flags(self):
        m = GPUMonitor()  # no on_emergency registered
        m._handle_condition(GPUStatus(condition=GPUCondition.EMERGENCY,
                                      condition_reason="melting"))
        assert m.is_emergency is True
