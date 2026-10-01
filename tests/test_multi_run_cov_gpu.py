"""GPU-safety plumbing of ``MultiRunTrainer`` (pre-flight, monitor callbacks, cooldown).

Mocked boundaries (there is no GPU on CI): ``get_gpu_status`` (the NVML/CUDA
reader), ``wait_for_safe_gpu`` (a polling sleep loop) and ``GPUMonitor`` (a
background thread). Everything the trainer does with the readings is real.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from backpropagate import multi_run
from backpropagate.gpu_safety import GPUCondition, GPUStatus
from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

MR_LOGGER = "backpropagate.multi_run"


def _status(condition=GPUCondition.SAFE, *, available=True, temp=60.0, vram_percent=10.0,
            reason="r"):
    return GPUStatus(
        available=available, device_name="Fake GPU", vram_total_gb=24.0,
        vram_percent=vram_percent, temperature_c=temp, condition=condition,
        condition_reason=reason,
    )


def _trainer(**cfg):
    return MultiRunTrainer(model="m", config=MultiRunConfig(**cfg))


class TestPreflight:
    def _with(self, monkeypatch, status):
        monkeypatch.setattr(multi_run, "get_gpu_status", lambda *a, **k: status)

    def test_no_gpu_fails_the_check(self, monkeypatch, caplog):
        self._with(monkeypatch, _status(available=False, condition=GPUCondition.UNKNOWN))
        with caplog.at_level(logging.ERROR, logger=MR_LOGGER):
            assert _trainer()._preflight_gpu_check() is False
        assert any("No GPU available" in r.getMessage() for r in caplog.records)

    def test_emergency_condition_fails_the_check(self, monkeypatch, caplog):
        self._with(monkeypatch, _status(GPUCondition.EMERGENCY, reason="99C"))
        with caplog.at_level(logging.ERROR, logger=MR_LOGGER):
            assert _trainer()._preflight_gpu_check() is False
        assert any("GPU in EMERGENCY state: 99C" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("recovered", [True, False])
    def test_critical_condition_waits_for_cooldown_and_returns_its_verdict(
        self, monkeypatch, recovered
    ):
        self._with(monkeypatch, _status(GPUCondition.CRITICAL, temp=92.0))
        waits = []
        monkeypatch.setattr(
            multi_run, "wait_for_safe_gpu", lambda **kw: waits.append(kw) or recovered
        )

        assert _trainer(cooldown_seconds=45.0)._preflight_gpu_check() is recovered
        assert waits == [{"max_wait_seconds": 45.0}]

    @pytest.mark.parametrize("temp", [64.0, None])
    def test_healthy_gpu_passes_with_or_without_a_temperature_reading(self, monkeypatch, temp):
        self._with(monkeypatch, _status(GPUCondition.SAFE, temp=temp))
        assert _trainer()._preflight_gpu_check() is True


class TestMonitorWiring:
    def test_monitor_is_configured_from_the_trainer_and_started(self, monkeypatch):
        created = []

        class _Monitor:
            def __init__(self, config, on_critical, on_emergency, on_status):
                self.config, self.callbacks = config, (on_critical, on_emergency, on_status)
                self.started = False
                created.append(self)

            def start(self):
                self.started = True

        monkeypatch.setattr(multi_run, "GPUMonitor", _Monitor)
        mrt = _trainer(max_temp_c=77.0)

        mrt._start_gpu_monitor()

        monitor = created[0]
        assert mrt._gpu_monitor is monitor and monitor.started is True
        assert monitor.config.temp_critical == 77.0 and monitor.config.check_interval == 10.0
        assert monitor.callbacks == (mrt._on_gpu_critical, mrt._on_gpu_emergency, mrt._on_gpu_status)


class TestStatusCallbacks:
    def test_peaks_only_ratchet_upwards(self):
        mrt = _trainer()
        mrt._on_gpu_status(_status(temp=70.0, vram_percent=40.0))
        mrt._on_gpu_status(_status(temp=65.0, vram_percent=55.0))
        mrt._on_gpu_status(_status(temp=None, vram_percent=20.0))

        assert mrt._gpu_max_temp == 70.0
        assert mrt._gpu_max_vram == 55.0

    def test_status_is_forwarded_to_the_user_callback(self):
        seen = []
        mrt = MultiRunTrainer(model="m", config=MultiRunConfig(), on_gpu_status=seen.append)
        status = _status()
        mrt._on_gpu_status(status)
        assert seen == [status]

    @pytest.mark.parametrize("condition", [GPUCondition.SAFE, GPUCondition.WARM])
    def test_recovery_clears_the_pause_event(self, condition, caplog):
        mrt = _trainer(pause_on_overheat=True)
        mrt._gpu_pause_event.set()
        with caplog.at_level(logging.WARNING, logger=MR_LOGGER):
            mrt._on_gpu_status(_status(condition))
        assert not mrt._gpu_pause_event.is_set()
        assert any("GPU recovered" in r.getMessage() for r in caplog.records)

    def test_still_hot_or_pause_disabled_keeps_the_event(self):
        mrt = _trainer(pause_on_overheat=True)
        mrt._gpu_pause_event.set()
        mrt._on_gpu_status(_status(GPUCondition.CRITICAL))
        assert mrt._gpu_pause_event.is_set()

        off = _trainer(pause_on_overheat=False)
        off._gpu_pause_event.set()
        off._on_gpu_status(_status(GPUCondition.SAFE))
        assert off._gpu_pause_event.is_set()

    def test_critical_arms_the_pause_once(self, caplog):
        mrt = _trainer(pause_on_overheat=True)
        with caplog.at_level(logging.WARNING, logger=MR_LOGGER):
            mrt._on_gpu_critical(_status(GPUCondition.CRITICAL, reason="hot"))
            mrt._on_gpu_critical(_status(GPUCondition.CRITICAL, reason="hot"))

        assert mrt._gpu_pause_event.is_set()
        armed = [r for r in caplog.records if "pause_on_overheat armed" in r.getMessage()]
        assert len(armed) == 1  # the second reading does not re-arm / re-log

    def test_critical_does_nothing_when_pause_is_disabled(self):
        mrt = _trainer(pause_on_overheat=False)
        mrt._on_gpu_critical(_status(GPUCondition.CRITICAL))
        assert not mrt._gpu_pause_event.is_set()

    def test_emergency_aborts_the_session_with_the_reason(self):
        mrt = _trainer()
        mrt._on_gpu_emergency(_status(GPUCondition.EMERGENCY, reason="thermal runaway"))
        assert mrt._should_abort is True
        assert mrt._abort_reason == "GPU emergency: thermal runaway"


class TestCooldown:
    def _run(self, monkeypatch, status, **cfg):
        waits = []
        monkeypatch.setattr(multi_run, "get_gpu_status", lambda *a, **k: status)
        monkeypatch.setattr(
            multi_run, "wait_for_safe_gpu", lambda **kw: waits.append(kw) or True
        )
        _trainer(**cfg)._check_cooldown()
        return waits

    def test_too_hot_waits_with_the_configured_budget(self, monkeypatch):
        waits = self._run(monkeypatch, _status(temp=91.0), max_temp_c=85.0, cooldown_seconds=30.0)
        assert waits == [{"max_wait_seconds": 30.0, "check_interval": 5.0}]

    def test_cool_enough_does_not_wait(self, monkeypatch):
        assert self._run(monkeypatch, _status(temp=60.0), max_temp_c=85.0) == []

    def test_unavailable_temperature_skips_the_gate_with_a_breadcrumb(self, monkeypatch, caplog):
        with caplog.at_level(logging.DEBUG, logger=MR_LOGGER):
            waits = self._run(monkeypatch, _status(temp=None))
        assert waits == []
        assert any("Cooldown gate skipped" in r.getMessage() for r in caplog.records)
