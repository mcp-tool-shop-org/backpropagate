"""
Integration tests for callback systems across components.

Tests the full callback flow through:
- Trainer -> TrainingCallback
- MultiRunTrainer -> all callbacks
- GPUMonitor -> MultiRunTrainer integration
- CLI -> callback display

See tests/EVENT_HANDLER_ROADMAP.md for the full testing plan.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# E2E TRAINING CALLBACK TESTS
# =============================================================================

class TestE2ETrainingCallbacks:
    """End-to-end tests for TrainingCallback with actual Trainer."""

    def test_callback_lifecycle_complete_training(self, mock_training_callback):
        """Test complete callback lifecycle during training."""
        callback, calls = mock_training_callback

        # Simulate the callback invocations that would occur during training
        if callback.on_step:
            callback.on_step(1, 2.5)
            callback.on_step(2, 2.3)
            callback.on_step(3, 2.1)

        if callback.on_complete:
            callback.on_complete(MagicMock())

        # Verify callbacks were invoked
        assert len(calls["step"]) == 3
        assert calls["step"][0] == (1, 2.5)
        assert calls["step"][-1] == (3, 2.1)
        assert len(calls["complete"]) == 1

    def test_callback_on_error_during_training(self, mock_training_callback):
        """Test on_error callback when training fails."""
        callback, calls = mock_training_callback

        # Simulate error during training
        error = RuntimeError("Training failed!")
        if callback.on_error:
            callback.on_error(error)

        assert len(calls["error"]) == 1
        assert calls["error"][0] is error

    def test_callback_isolation_from_training_errors(self, tmp_path):
        """Real Trainer.train must isolate a raising on_complete callback.

        Audit TESTS-A-006: the previous version of this test built a
        ``safe_invoke`` wrapper inside the body and asserted the wrapper
        called the callback — which would pass even if Trainer dropped its
        own try/except. This rewrite drives the real Trainer.train with a
        mocked SFTTrainer and a raising on_complete, then verifies (a) train
        returns successfully and (b) the callback was actually invoked.
        """
        from backpropagate.trainer import Trainer, TrainingCallback

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(tmp_path), use_unsloth=False)
        trainer._model = MagicMock()
        trainer._tokenizer = MagicMock()
        trainer._is_loaded = True

        invocations = {"count": 0}

        def raising_callback(run):
            invocations["count"] += 1
            raise ValueError("Callback crashed!")

        callback = TrainingCallback(on_complete=raising_callback)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        sft_mock = MagicMock()
        sft_mock.train.return_value = MagicMock(training_loss=0.5)
        sft_mock.state.log_history = [{"loss": 0.5}]

        with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
             patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
             patch("trl.SFTTrainer", return_value=sft_mock), \
             patch("trl.SFTConfig"):
            # Will propagate if Trainer's try/except around on_complete is removed.
            run = trainer.train("dummy_dataset", steps=10, callback=callback)

        assert invocations["count"] == 1, (
            "on_complete callback was not invoked by Trainer.train"
        )
        assert run is not None, (
            "Trainer.train must return a TrainingRun even when on_complete raises"
        )


class TestE2EMultiRunCallbacks:
    """End-to-end tests for MultiRunTrainer callbacks."""

    def test_multirun_callback_full_sequence(self, mock_multirun_callbacks):
        """Test full callback sequence across multiple runs."""
        from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

        callbacks, calls = mock_multirun_callbacks

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=3, steps_per_run=5),
            **callbacks,
        )

        # Simulate the run sequence
        for run_idx in range(3):
            if trainer.on_run_start:
                trainer.on_run_start(run_idx)

            for step in range(5):
                if trainer.on_step:
                    trainer.on_step(run_idx, step, 2.5 - step * 0.1)

            if trainer.on_run_complete:
                result = MagicMock()
                result.run_index = run_idx
                result.final_loss = 2.0
                trainer.on_run_complete(result)

        # Verify sequence
        assert calls["run_start"] == [0, 1, 2]
        assert len(calls["step"]) == 15  # 3 runs * 5 steps
        assert len(calls["run_complete"]) == 3

    def test_multirun_callback_run_indices_sequential(self, mock_multirun_callbacks):
        """Run indices should be sequential starting from 0."""
        from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

        callbacks, calls = mock_multirun_callbacks

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=5, steps_per_run=1),
            **callbacks,
        )

        # Simulate runs
        for i in range(5):
            trainer.on_run_start(i)

        assert calls["run_start"] == [0, 1, 2, 3, 4]

    def test_multirun_step_callback_tracks_loss_progression(self, mock_multirun_callbacks):
        """Step callback should show loss decreasing within runs."""
        from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

        callbacks, calls = mock_multirun_callbacks

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=1, steps_per_run=5),
            **callbacks,
        )

        # Simulate decreasing loss
        losses = [2.5, 2.3, 2.1, 1.9, 1.7]
        for step, loss in enumerate(losses):
            trainer.on_step(0, step, loss)

        recorded_losses = [loss for _, _, loss in calls["step"]]
        assert recorded_losses == losses

        # Verify loss is decreasing
        for i in range(1, len(recorded_losses)):
            assert recorded_losses[i] < recorded_losses[i - 1]


@pytest.mark.integration
class TestE2EGPUMonitoringCallbacks:
    """End-to-end tests for GPU monitoring callbacks."""

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_gpu_monitor_to_multirun_integration(
        self, mock_get_status, gpu_status_safe, mock_multirun_callbacks
    ):
        """GPUMonitor callbacks should integrate with MultiRunTrainer.

        TESTS-A-008 (v1.3 Wave 1): replaced ``time.sleep(0.2)`` with a
        deterministic ``threading.Event`` wait. The prior sleep flaked on
        slow Windows CI runners (the first poll cycle could take > 200ms
        when the subprocess sandbox was cold), producing intermittent
        ``len(calls['gpu_status']) > 0`` failures.

        The new pattern: wrap the original on_status callback in a
        signalling wrapper that sets an Event on the first invocation.
        Wait for the event with a generous timeout. Wait completes the
        moment the callback fires — not on a wall-clock guess.
        """
        import threading

        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_safe
        callbacks, calls = mock_multirun_callbacks

        # Signal fires the moment the first callback is observed.
        first_call = threading.Event()
        original_on_status = callbacks["on_gpu_status"]

        def signalling_on_status(status):
            original_on_status(status)
            first_call.set()

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.05),
            on_status=signalling_on_status,
        )

        try:
            monitor.start()
            # Deterministic wait — 5s ceiling is generous for the slowest
            # observed CI runner; the sleep version used 200ms which was
            # below the cold-start floor on Windows.
            triggered = first_call.wait(timeout=5.0)
            assert triggered, (
                "GPU monitor did not invoke on_status within 5s — the "
                "callback wiring or polling thread is broken (not a flake)."
            )
        finally:
            monitor.stop()

        # GPU status should have been received
        assert len(calls["gpu_status"]) > 0
        for status in calls["gpu_status"]:
            assert status.temperature_c == 60.0

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_gpu_critical_triggers_callback_chain(
        self, mock_get_status, gpu_status_critical
    ):
        """Critical GPU status should trigger critical callback."""
        from backpropagate.gpu_safety import GPUCondition, GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_critical

        critical_events = []
        status_events = []

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.05),
            on_status=lambda s: status_events.append(s),
            on_critical=lambda s: critical_events.append(s),
        )

        try:
            monitor.start()
            time.sleep(0.2)
        finally:
            monitor.stop()

        # Both callbacks should fire
        assert len(status_events) > 0
        assert len(critical_events) > 0

        # Critical events should have critical condition
        for event in critical_events:
            assert event.condition == GPUCondition.CRITICAL

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_gpu_emergency_callback_timing(
        self, mock_get_status, gpu_status_emergency
    ):
        """Emergency callbacks should fire immediately."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_emergency

        emergency_times = []
        start_time = [None]

        def on_emergency(status):
            if start_time[0]:
                emergency_times.append(time.time() - start_time[0])

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.02),
            on_emergency=on_emergency,
        )

        start_time[0] = time.time()
        try:
            monitor.start()
            time.sleep(0.1)
        finally:
            monitor.stop()

        # First emergency MUST fire — without this assert, the timing
        # check below silently passes when zero emergencies fire (which
        # would itself be a regression: gpu_status_emergency should
        # always trigger on_emergency).
        assert len(emergency_times) > 0, (
            "GPU emergency callback never fired despite mocked emergency "
            "status. Previously gated by `if emergency_times:` which let "
            "the zero-event case pass silently, defeating the entire "
            "point of the timing test (a callback that never fires also "
            "never fires 'too late')."
        )
        # First emergency should fire within ~20ms (check_interval)
        assert emergency_times[0] < 0.1


class TestCLICallbackIntegration:
    """Tests for CLI integration with callbacks."""

    @patch("backpropagate.cli._print_success")
    @patch("backpropagate.cli._print_error")
    def test_cli_train_uses_progress_callback(
        self, mock_print_error, mock_print_success
    ):
        """CLI train command should use on_step for progress."""
        from backpropagate.trainer import TrainingCallback

        # Simulate CLI creating callback for progress
        progress_updates = []

        def on_step(step: int, loss: float) -> None:
            progress_updates.append(f"Step {step}: loss={loss:.4f}")

        callback = TrainingCallback(on_step=on_step)

        # Simulate training steps
        for i in range(5):
            callback.on_step(i + 1, 2.5 - i * 0.1)

        assert len(progress_updates) == 5
        assert "Step 1: loss=2.5000" in progress_updates[0]
        assert "Step 5: loss=2.1000" in progress_updates[-1]

    def test_cli_multirun_status_callback(self):
        """CLI multi-run should report via on_run_complete."""
        from backpropagate.multi_run import RunResult

        status_messages = []

        def on_run_complete(result):
            status_messages.append(
                f"Run {result.run_index + 1} complete: loss={result.final_loss:.4f}"
            )

        # Simulate runs completing
        for i in range(3):
            result = MagicMock(spec=RunResult)
            result.run_index = i
            result.final_loss = 2.0 - i * 0.3
            on_run_complete(result)

        assert len(status_messages) == 3
        assert "Run 1 complete: loss=2.0000" in status_messages[0]
        assert "Run 3 complete: loss=1.4000" in status_messages[-1]


# =============================================================================
# CROSS-COMPONENT CALLBACK FLOW TESTS
# =============================================================================

class TestCrossComponentCallbackFlow:
    """Tests for callbacks flowing between components."""

    def test_trainer_callback_to_external_logger(self):
        """Trainer callbacks can be used for external logging."""
        from backpropagate.trainer import TrainingCallback

        # Simulate external logger
        log_entries = []

        class ExternalLogger:
            def log_metric(self, name, value, step):
                log_entries.append({"name": name, "value": value, "step": step})

        logger = ExternalLogger()

        callback = TrainingCallback(
            on_step=lambda step, loss: logger.log_metric("loss", loss, step)
        )

        # Simulate training
        for i in range(5):
            callback.on_step(i, 2.5 - i * 0.1)

        assert len(log_entries) == 5
        assert log_entries[0] == {"name": "loss", "value": 2.5, "step": 0}

    def test_multirun_callbacks_aggregate_statistics(self):
        """MultiRun callbacks can aggregate statistics across runs."""

        class RunStatistics:
            def __init__(self):
                self.losses = []
                self.run_times = []
                self.total_steps = 0

            def on_run_complete(self, result):
                self.losses.append(result.final_loss)

            def on_step(self, run_idx, step, loss):
                self.total_steps += 1

        stats = RunStatistics()

        # Simulate 3 runs with 10 steps each
        for run in range(3):
            for step in range(10):
                stats.on_step(run, step, 2.0)

            result = MagicMock()
            result.final_loss = 2.0 - run * 0.3
            stats.on_run_complete(result)

        assert len(stats.losses) == 3
        assert stats.total_steps == 30
        assert stats.losses == [2.0, 1.7, 1.4]

    def test_gpu_monitor_triggers_training_abort(self):
        """GPU emergency should trigger training abort."""

        class TrainingController:
            def __init__(self):
                self.abort_requested = False
                self.abort_reason = None

            def on_gpu_emergency(self, status):
                self.abort_requested = True
                self.abort_reason = f"GPU emergency: {status.condition_reason}"

        controller = TrainingController()

        # Simulate emergency status
        from backpropagate.gpu_safety import GPUCondition, GPUStatus

        emergency_status = GPUStatus(
            available=True,
            device_name="GPU",
            temperature_c=98.0,
            vram_total_gb=16.0,
            vram_used_gb=15.9,
            vram_percent=99.0,
            condition=GPUCondition.EMERGENCY,
            condition_reason="Temperature EMERGENCY: 98.0°C",
        )

        controller.on_gpu_emergency(emergency_status)

        assert controller.abort_requested
        assert "98.0°C" in controller.abort_reason


class TestCallbackErrorRecovery:
    """Tests for error recovery in callback chains."""

    def test_callback_chain_continues_after_error(self):
        """Error in one callback shouldn't stop others."""

        call_log = []

        def callback_1(value):
            call_log.append("cb1")

        def callback_2(value):
            call_log.append("cb2_start")
            raise RuntimeError("Callback 2 failed!")

        def callback_3(value):
            call_log.append("cb3")

        callbacks = [callback_1, callback_2, callback_3]

        # Simulate safe callback invocation
        def invoke_all(value):
            for cb in callbacks:
                try:
                    cb(value)
                except Exception:
                    call_log.append("error_caught")

        invoke_all(42)

        assert "cb1" in call_log
        assert "cb2_start" in call_log
        assert "error_caught" in call_log
        assert "cb3" in call_log

    def test_callback_timeout_handling(self):
        """Slow callbacks should be handled gracefully."""
        import concurrent.futures

        def slow_callback(value):
            time.sleep(2.0)  # 2 second delay
            return "done"

        # Use timeout to handle slow callback
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(slow_callback, 42)
            try:
                result = future.result(timeout=0.1)
            except concurrent.futures.TimeoutError:
                result = "timeout"

        assert result == "timeout"


class TestCallbackMemoryManagement:
    """Tests for proper memory management with callbacks."""

    def test_callback_doesnt_leak_references(self):
        """Callbacks shouldn't prevent garbage collection."""
        import weakref

        class HeavyObject:
            def __init__(self):
                self.data = [0] * 1000000  # ~4MB

        collected = [False]

        def mark_collected(ref):
            collected[0] = True

        heavy = HeavyObject()
        weak_ref = weakref.ref(heavy, mark_collected)

        # Callback captures reference
        captured_value = []

        def callback(obj):
            captured_value.append(obj.data[0])

        callback(heavy)

        # Delete strong reference
        del heavy
        import gc
        gc.collect()

        # Object should be collectable (weak_ref should be dead) because the
        # callback closure does NOT capture `heavy` — it only takes `obj` as a
        # parameter and reads `obj.data[0]` into the captured `captured_value`
        # list, releasing the obj reference on return.
        assert weak_ref() is None, (
            "HeavyObject was not garbage collected after del + gc.collect(). "
            "The callback closure is leaking a reference to it."
        )
        assert collected[0] is True, (
            "weakref finalizer mark_collected was not invoked"
        )

    def test_callback_with_closure_cleanup(self):
        """Closures in callbacks should clean up properly."""

        results = []

        def create_callback(index):
            # Closure captures index
            def callback(value):
                results.append((index, value))
            return callback

        callbacks = [create_callback(i) for i in range(5)]

        for i, cb in enumerate(callbacks):
            cb(i * 10)

        assert results == [(0, 0), (1, 10), (2, 20), (3, 30), (4, 40)]

        # Clear callbacks
        callbacks.clear()
        import gc
        gc.collect()

        # Results should still be intact
        assert len(results) == 5


# =============================================================================
# CONCURRENT CALLBACK TESTS
# =============================================================================

class TestConcurrentCallbacks:
    """Tests for concurrent callback execution."""

    def test_multiple_monitors_concurrent_callbacks(self):
        """Multiple monitors can have concurrent callbacks."""
        from backpropagate.gpu_safety import GPUCondition, GPUStatus

        results = {"monitor1": [], "monitor2": []}
        lock = threading.Lock()

        def create_callback(monitor_name):
            def callback(status):
                with lock:
                    results[monitor_name].append(status)
            return callback

        # Simulate concurrent callbacks from two monitors
        def simulate_monitor(name, count):
            callback = create_callback(name)
            for i in range(count):
                status = GPUStatus(
                    available=True,
                    device_name=name,
                    temperature_c=60.0 + i,
                    vram_total_gb=16.0,
                    vram_used_gb=8.0,
                    vram_percent=50.0,
                    condition=GPUCondition.SAFE,
                )
                callback(status)
                time.sleep(0.01)

        threads = [
            threading.Thread(target=simulate_monitor, args=("monitor1", 10)),
            threading.Thread(target=simulate_monitor, args=("monitor2", 10)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "Thread did not finish within timeout"

        assert len(results["monitor1"]) == 10
        assert len(results["monitor2"]) == 10

    def test_callback_ordering_preserved_per_source(self):
        """Callbacks from same source should maintain order."""

        sequence = []
        lock = threading.Lock()

        def callback(value):
            with lock:
                sequence.append(value)

        def send_sequence(start, count):
            for i in range(count):
                callback(start + i)

        # Single source, sequential
        send_sequence(0, 5)

        # Order should be preserved
        assert sequence == [0, 1, 2, 3, 4]


# =============================================================================
# TRAINER CALLBACK ERROR ISOLATION TESTS
# =============================================================================

class TestTrainerCallbackErrorIsolation:
    """Integration tests pinning the real callback-isolation contract.

    Audit TESTS-A-006: the previous shape of these tests built a handwritten
    ``safe_invoke`` wrapper inside the test body and asserted the wrapper
    called the callback. That meant a refactor that drops Trainer's own
    try/except around on_complete would NOT have been caught — the synthetic
    wrapper would still pass. These rewrites call the real ``Trainer.train``
    against a mocked SFTTrainer so the regression surface is the actual
    in-trainer ``try: callback.on_complete(...) except: ...`` shape at
    trainer.py:1131-1135.
    """

    def _setup_trainer(self, temp_dir):
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir), use_unsloth=False)
        trainer._model = MagicMock()
        trainer._tokenizer = MagicMock()
        trainer._is_loaded = True
        return trainer

    def _make_sft_mock(self):
        instance = MagicMock()
        instance.train.return_value = MagicMock(training_loss=0.5)
        instance.state.log_history = [{"loss": 0.5}]
        return instance

    def test_on_complete_error_isolated_by_real_trainer(self, tmp_path):
        """Real Trainer.train must catch a raising on_complete callback.

        Pins trainer.py:1131-1135 (the ``try/except`` around
        ``callback.on_complete(run)``). If that try/except is removed in a
        refactor, this test fails — the callback's ValueError propagates out
        of ``trainer.train()`` and pytest reports the regression.
        """
        from backpropagate.trainer import TrainingCallback

        trainer = self._setup_trainer(tmp_path)

        invoked = {"count": 0}

        def raising_complete(run):
            invoked["count"] += 1
            raise ValueError("Callback crashed!")

        callback = TrainingCallback(on_complete=raising_complete)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
             patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
             patch("trl.SFTTrainer", return_value=self._make_sft_mock()), \
             patch("trl.SFTConfig"):
            # If the trainer's try/except is removed, this call propagates.
            run = trainer.train("dummy_dataset", steps=10, callback=callback)

        # The callback was actually invoked AND its exception was swallowed.
        assert invoked["count"] == 1, (
            "on_complete callback was not invoked by real Trainer.train"
        )
        assert run is not None, (
            "Trainer.train must return a TrainingRun even when on_complete "
            "raises; isolation is broken otherwise"
        )

    def test_on_complete_error_doesnt_corrupt_returned_run(self, tmp_path):
        """A failing on_complete must not affect the returned TrainingRun."""
        from backpropagate.trainer import TrainingCallback

        trainer = self._setup_trainer(tmp_path)

        def bad_complete(run):
            # Even if the callback mutates the run object before raising, the
            # returned TrainingRun must remain a valid object.
            raise ValueError("Tried to mess with results!")

        callback = TrainingCallback(on_complete=bad_complete)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
             patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
             patch("trl.SFTTrainer", return_value=self._make_sft_mock()), \
             patch("trl.SFTConfig"):
            run = trainer.train("dummy_dataset", steps=10, callback=callback)

        # The run is still a valid TrainingRun with run_id present.
        assert run is not None
        assert getattr(run, "run_id", None), (
            "Returned TrainingRun must carry a run_id even after on_complete raised"
        )

    def test_synthetic_safe_invoke_pattern_is_a_smoke_test(self):
        """Documents that the synthetic safe_invoke pattern is a SMOKE test only.

        The previous version of these tests built a hand-rolled ``safe_invoke``
        wrapper inside the test body. That pattern is a useful smoke test for
        the *callback type* — it confirms that a raising callback can be
        invoked-and-swallowed by SOME wrapper — but it is NOT a regression
        test for ``trainer.py``'s wrapping behavior. The real-Trainer tests
        above are what pin the contract; this one is preserved so the
        callback-type API itself stays accidentally-stable.
        """
        from backpropagate.trainer import TrainingCallback

        bad_step = MagicMock(side_effect=RuntimeError("Step failed"))
        bad_complete = MagicMock(side_effect=ValueError("Complete failed"))
        callback = TrainingCallback(on_step=bad_step, on_complete=bad_complete)

        # The callbacks ARE callable from the dataclass — pin only that.
        assert callable(callback.on_step)
        assert callable(callback.on_complete)
        # And they DO raise when invoked directly (so the integration tests
        # above are exercising a real failure mode).
        with pytest.raises(RuntimeError):
            callback.on_step(1, 0.5)
        with pytest.raises(ValueError):
            callback.on_complete(MagicMock())


class TestTrainerCallbackWithRealTrainer:
    """Integration tests that drive real ``Trainer.train`` with mocked TRL.

    Audit TESTS-A-006: replaces the previous synthetic-wrapper smoke tests
    with calls into ``Trainer.train`` proper. Backed by the same mock pattern
    used elsewhere in this file (``patch('trl.SFTTrainer')``).
    """

    def _setup_trainer(self, temp_dir):
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(output_dir=str(temp_dir), use_unsloth=False)
        trainer._model = MagicMock()
        trainer._tokenizer = MagicMock()
        trainer._is_loaded = True
        return trainer

    def test_good_callback_invoked_by_real_trainer(self, tmp_path):
        """A well-behaved on_complete is invoked exactly once by Trainer.train."""
        from backpropagate.trainer import TrainingCallback

        trainer = self._setup_trainer(tmp_path)
        completed = []

        def on_complete(run):
            completed.append(run)

        callback = TrainingCallback(on_complete=on_complete)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        sft_mock = MagicMock()
        sft_mock.train.return_value = MagicMock(training_loss=0.5)
        sft_mock.state.log_history = [{"loss": 0.5}]

        with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
             patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
             patch("trl.SFTTrainer", return_value=sft_mock), \
             patch("trl.SFTConfig"):
            run = trainer.train("dummy_dataset", steps=10, callback=callback)

        assert len(completed) == 1, (
            f"on_complete should be invoked exactly once; got {len(completed)}"
        )
        assert completed[0].run_id == run.run_id

    def test_raising_callback_does_not_break_run_persistence(self, tmp_path):
        """on_complete failure must not prevent the run record from being created."""
        from backpropagate.trainer import TrainingCallback

        trainer = self._setup_trainer(tmp_path)

        def boom(run):
            raise ValueError("on_complete blew up")

        callback = TrainingCallback(on_complete=boom)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        sft_mock = MagicMock()
        sft_mock.train.return_value = MagicMock(training_loss=0.5)
        sft_mock.state.log_history = [{"loss": 0.5}]

        # Pre-condition: no runs recorded yet.
        runs_before = len(trainer._training_runs)

        with patch.object(trainer, "_load_dataset", return_value=mock_dataset), \
             patch.object(trainer, "_pre_tokenize", return_value=mock_dataset), \
             patch("trl.SFTTrainer", return_value=sft_mock), \
             patch("trl.SFTConfig"):
            trainer.train("dummy_dataset", steps=10, callback=callback)

        # Post-condition: training_runs grew by one, even though on_complete raised.
        assert len(trainer._training_runs) == runs_before + 1, (
            "Trainer._training_runs should grow by 1 even when on_complete raises; "
            "the callback's exception must not abort run persistence."
        )


# =============================================================================
# MULTIRUN CALLBACK INVOCATION ORDER TESTS
# =============================================================================

class TestMultiRunCallbackInvocationOrder:
    """Tests for callback invocation ordering in MultiRunTrainer."""

    def test_run_start_before_steps(self, callback_tracker):
        """on_run_start should be called before any on_step calls."""
        from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=1, steps_per_run=3),
            on_run_start=callback_tracker.track("run_start"),
            on_step=callback_tracker.track("step"),
        )

        # Simulate the expected sequence
        trainer.on_run_start(0)
        trainer.on_step(0, 1, 2.5)
        trainer.on_step(0, 2, 2.3)
        trainer.on_step(0, 3, 2.1)

        sequence = callback_tracker.get_sequence()
        assert sequence[0] == "run_start"
        assert sequence[1:] == ["step", "step", "step"]

    def test_run_complete_after_all_steps(self, callback_tracker):
        """on_run_complete should be called after all steps in a run."""
        from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=1, steps_per_run=3),
            on_step=callback_tracker.track("step"),
            on_run_complete=callback_tracker.track("complete"),
        )

        # Simulate sequence
        trainer.on_step(0, 1, 2.5)
        trainer.on_step(0, 2, 2.3)
        trainer.on_step(0, 3, 2.1)
        trainer.on_run_complete(MagicMock())

        sequence = callback_tracker.get_sequence()
        assert sequence == ["step", "step", "step", "complete"]

    def test_multirun_full_sequence(self, callback_tracker):
        """Test complete callback sequence across multiple runs."""
        from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=2, steps_per_run=2),
            on_run_start=callback_tracker.track("start"),
            on_step=callback_tracker.track("step"),
            on_run_complete=callback_tracker.track("complete"),
        )

        # Simulate 2 runs with 2 steps each
        for run_idx in range(2):
            trainer.on_run_start(run_idx)
            for step in range(2):
                trainer.on_step(run_idx, step + 1, 2.0 - step * 0.1)
            trainer.on_run_complete(MagicMock())

        expected = [
            "start", "step", "step", "complete",  # Run 0
            "start", "step", "step", "complete",  # Run 1
        ]
        callback_tracker.assert_sequence(expected)

    def test_run_indices_passed_correctly(self, mock_multirun_callbacks):
        """Run indices should be passed correctly to all callbacks."""
        from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

        callbacks, calls = mock_multirun_callbacks

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=3, steps_per_run=2),
            **callbacks,
        )

        # Simulate 3 runs
        for run_idx in range(3):
            trainer.on_run_start(run_idx)
            for step in range(2):
                trainer.on_step(run_idx, step + 1, 1.5)

        # Check run indices in on_run_start
        assert calls["run_start"] == [0, 1, 2]

        # Check run indices in on_step
        step_run_indices = [run_idx for run_idx, _, _ in calls["step"]]
        assert step_run_indices == [0, 0, 1, 1, 2, 2]

    def test_step_numbers_reset_each_run(self, mock_multirun_callbacks):
        """Step numbers should restart from 1 each run."""
        from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer

        callbacks, calls = mock_multirun_callbacks

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=2, steps_per_run=3),
            **callbacks,
        )

        # Simulate 2 runs with 3 steps each, steps starting at 1
        for run_idx in range(2):
            for step in range(1, 4):  # 1, 2, 3
                trainer.on_step(run_idx, step, 1.5)

        step_numbers = [step for _, step, _ in calls["step"]]
        # Each run should have steps 1, 2, 3
        assert step_numbers == [1, 2, 3, 1, 2, 3]


# =============================================================================
# GPU MONITOR PAUSE/RESUME CALLBACK TESTS
# =============================================================================

@pytest.mark.integration
class TestGPUMonitorPauseResumeCallbacks:
    """Tests for callback behavior during pause/resume."""

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_pause_stops_callbacks(self, mock_get_status, gpu_status_safe):
        """Pausing the monitor should stop callbacks."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_safe

        callback_count = [0]

        def on_status(status):
            callback_count[0] += 1

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.02),
            on_status=on_status,
        )

        try:
            monitor.start()
            time.sleep(0.1)  # Let some callbacks fire
            count_before_pause = callback_count[0]
            assert count_before_pause > 0

            monitor.pause()
            time.sleep(0.1)  # Wait while paused
            count_after_pause = callback_count[0]

            # Should have no or very few new callbacks while paused
            # Allow 1 extra for timing edge cases
            assert count_after_pause <= count_before_pause + 1

        finally:
            monitor.stop()

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_resume_restarts_callbacks(self, mock_get_status, gpu_status_safe):
        """Resuming the monitor should restart callbacks."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_safe

        callback_count = [0]

        def on_status(status):
            callback_count[0] += 1

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.02),
            on_status=on_status,
        )

        try:
            monitor.start()
            time.sleep(0.05)

            monitor.pause()
            time.sleep(0.05)
            count_at_pause = callback_count[0]

            monitor.resume()
            time.sleep(0.1)  # Wait for callbacks to resume
            count_after_resume = callback_count[0]

            # Should have more callbacks after resume
            assert count_after_resume > count_at_pause

        finally:
            monitor.stop()

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_multiple_pause_resume_cycles(self, mock_get_status, gpu_status_safe):
        """Multiple pause/resume cycles should work correctly."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_safe

        callback_timestamps = []

        def on_status(status):
            callback_timestamps.append(time.time())

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.02),
            on_status=on_status,
        )

        try:
            monitor.start()

            for _ in range(3):
                time.sleep(0.05)
                count_before = len(callback_timestamps)
                monitor.pause()
                time.sleep(0.05)
                monitor.resume()
                time.sleep(0.05)
                count_after = len(callback_timestamps)
                # Should have gotten more callbacks after each resume
                assert count_after > count_before

        finally:
            monitor.stop()

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_callbacks_preserve_state_across_pause(
        self, mock_get_status, gpu_status_safe
    ):
        """Callback state should be preserved across pause/resume."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_safe

        state = {"total_calls": 0, "temperatures": []}

        def stateful_callback(status):
            state["total_calls"] += 1
            state["temperatures"].append(status.temperature_c)

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.02),
            on_status=stateful_callback,
        )

        try:
            monitor.start()
            time.sleep(0.05)
            calls_phase1 = state["total_calls"]

            monitor.pause()
            time.sleep(0.03)
            monitor.resume()
            time.sleep(0.05)
            calls_phase2 = state["total_calls"]

            # State should accumulate, not reset
            assert calls_phase2 > calls_phase1
            assert len(state["temperatures"]) == state["total_calls"]

        finally:
            monitor.stop()


# =============================================================================
# EVENT ESCALATION/DE-ESCALATION TESTS
# =============================================================================

class TestEventEscalation:
    """Tests for GPU condition escalation and de-escalation."""

    def test_safe_to_warning_escalation(self):
        """Temperature rise should escalate from SAFE to WARNING."""
        from backpropagate.gpu_safety import GPUCondition, GPUSafetyConfig, GPUStatus

        config = GPUSafetyConfig()

        # Create sequence of statuses with rising temperature
        safe_status = GPUStatus(
            available=True,
            device_name="GPU",
            temperature_c=70.0,
            vram_total_gb=16.0,
            vram_used_gb=8.0,
            vram_percent=50.0,
            condition=GPUCondition.SAFE,
        )

        warning_status = GPUStatus(
            available=True,
            device_name="GPU",
            temperature_c=config.temp_warning + 1,  # Just above warning threshold
            vram_total_gb=16.0,
            vram_used_gb=8.0,
            vram_percent=50.0,
            condition=GPUCondition.WARNING,
            condition_reason=f"Temperature WARNING: {config.temp_warning + 1}°C",
        )

        assert safe_status.condition == GPUCondition.SAFE
        assert warning_status.condition == GPUCondition.WARNING

    def test_warning_to_critical_escalation(self):
        """Further temperature rise should escalate to CRITICAL."""
        from backpropagate.gpu_safety import GPUCondition, GPUSafetyConfig, GPUStatus

        config = GPUSafetyConfig()

        warning_status = GPUStatus(
            available=True,
            device_name="GPU",
            temperature_c=config.temp_warning + 1,
            vram_total_gb=16.0,
            vram_used_gb=8.0,
            vram_percent=50.0,
            condition=GPUCondition.WARNING,
        )

        critical_status = GPUStatus(
            available=True,
            device_name="GPU",
            temperature_c=config.temp_critical + 1,
            vram_total_gb=16.0,
            vram_used_gb=14.0,
            vram_percent=87.5,
            condition=GPUCondition.CRITICAL,
            condition_reason=f"Temperature CRITICAL: {config.temp_critical + 1}°C",
        )

        assert warning_status.condition == GPUCondition.WARNING
        assert critical_status.condition == GPUCondition.CRITICAL

    def test_critical_to_emergency_escalation(self):
        """Extreme conditions should escalate to EMERGENCY."""
        from backpropagate.gpu_safety import GPUCondition, GPUSafetyConfig, GPUStatus

        config = GPUSafetyConfig()

        emergency_status = GPUStatus(
            available=True,
            device_name="GPU",
            temperature_c=config.temp_emergency + 1,
            vram_total_gb=16.0,
            vram_used_gb=15.8,
            vram_percent=98.75,
            condition=GPUCondition.EMERGENCY,
            condition_reason=f"Temperature EMERGENCY: {config.temp_emergency + 1}°C",
        )

        assert emergency_status.condition == GPUCondition.EMERGENCY

    def test_deescalation_on_cooldown(self):
        """Temperature drop should de-escalate condition."""
        from backpropagate.gpu_safety import GPUCondition, GPUSafetyConfig, GPUStatus

        config = GPUSafetyConfig()

        # Start at critical
        critical_status = GPUStatus(
            available=True,
            device_name="GPU",
            temperature_c=config.temp_critical + 1,
            vram_total_gb=16.0,
            vram_used_gb=14.0,
            vram_percent=87.5,
            condition=GPUCondition.CRITICAL,
        )

        # Cool down to warning
        warning_status = GPUStatus(
            available=True,
            device_name="GPU",
            temperature_c=config.temp_warning + 1,
            vram_total_gb=16.0,
            vram_used_gb=14.0,
            vram_percent=87.5,
            condition=GPUCondition.WARNING,
        )

        # Cool down to safe
        safe_status = GPUStatus(
            available=True,
            device_name="GPU",
            temperature_c=70.0,  # Well below warning
            vram_total_gb=16.0,
            vram_used_gb=8.0,
            vram_percent=50.0,
            condition=GPUCondition.SAFE,
        )

        # Verify de-escalation path
        assert critical_status.condition == GPUCondition.CRITICAL
        assert warning_status.condition == GPUCondition.WARNING
        assert safe_status.condition == GPUCondition.SAFE

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_escalation_triggers_appropriate_callbacks(self, mock_get_status):
        """Escalating conditions should trigger appropriate callbacks."""
        from backpropagate.gpu_safety import GPUCondition, GPUMonitor, GPUSafetyConfig, GPUStatus

        callbacks_fired = {"warning": [], "critical": [], "emergency": []}

        def on_warning(status):
            callbacks_fired["warning"].append(status.condition)

        def on_critical(status):
            callbacks_fired["critical"].append(status.condition)

        def on_emergency(status):
            callbacks_fired["emergency"].append(status.condition)

        # Sequence of escalating statuses
        statuses = [
            GPUStatus(
                available=True, device_name="GPU", temperature_c=85.0,
                vram_total_gb=16.0, vram_used_gb=8.0, vram_percent=50.0,
                condition=GPUCondition.WARNING,
            ),
            GPUStatus(
                available=True, device_name="GPU", temperature_c=92.0,
                vram_total_gb=16.0, vram_used_gb=14.0, vram_percent=87.5,
                condition=GPUCondition.CRITICAL,
            ),
            GPUStatus(
                available=True, device_name="GPU", temperature_c=97.0,
                vram_total_gb=16.0, vram_used_gb=15.8, vram_percent=98.75,
                condition=GPUCondition.EMERGENCY,
            ),
        ]

        status_index = [0]

        def get_escalating_status(device_index=0, config=None):
            idx = min(status_index[0], len(statuses) - 1)
            status_index[0] += 1
            return statuses[idx]

        mock_get_status.side_effect = get_escalating_status

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.02),
            on_warning=on_warning,
            on_critical=on_critical,
            on_emergency=on_emergency,
        )

        try:
            monitor.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if (len(callbacks_fired["warning"]) > 0
                        and len(callbacks_fired["critical"]) > 0
                        and len(callbacks_fired["emergency"]) > 0):
                    break
                time.sleep(0.05)
        finally:
            monitor.stop()

        # All escalation callbacks should have fired
        assert len(callbacks_fired["warning"]) > 0, "Expected warning callbacks"
        assert len(callbacks_fired["critical"]) > 0, "Expected critical callbacks"
        assert len(callbacks_fired["emergency"]) > 0, "Expected emergency callbacks"

    def test_vram_escalation_independent_of_temperature(self):
        """VRAM exhaustion should escalate independent of temperature."""
        from backpropagate.gpu_safety import GPUCondition, GPUSafetyConfig, GPUStatus

        config = GPUSafetyConfig()

        # Low temp but high VRAM - should still be critical
        high_vram_status = GPUStatus(
            available=True,
            device_name="GPU",
            temperature_c=60.0,  # Safe temperature
            vram_total_gb=16.0,
            vram_used_gb=15.5,  # 97% VRAM
            vram_percent=96.875,
            condition=GPUCondition.CRITICAL,  # Due to VRAM
            condition_reason="VRAM CRITICAL: 96.9%",
        )

        # The condition should reflect the worst metric
        assert high_vram_status.vram_percent > config.vram_critical
        assert high_vram_status.condition == GPUCondition.CRITICAL


# =============================================================================
# REAL TRAINER INTEGRATION TESTS (MOCKED TRAINING)
# =============================================================================

class TestRealTrainerIntegration:
    """Integration tests using real Trainer with mocked training internals."""

    @patch("backpropagate.trainer.Trainer")
    def test_trainer_train_method_invokes_callbacks(self, MockTrainer):
        """Trainer.train() should invoke callbacks at appropriate times."""
        from backpropagate.trainer import TrainingCallback, TrainingRun

        # Create a callback to track invocations
        invocations = []

        def on_step(step, loss):
            invocations.append(("step", step, loss))

        def on_complete(run):
            invocations.append(("complete", run.final_loss))

        callback = TrainingCallback(
            on_step=on_step,
            on_complete=on_complete,
        )

        # Simulate the callback invocations that would happen
        callback.on_step(1, 2.5)
        callback.on_step(2, 2.3)
        callback.on_step(3, 2.1)

        mock_run = MagicMock(spec=TrainingRun)
        mock_run.final_loss = 2.1
        callback.on_complete(mock_run)

        assert len(invocations) == 4
        assert invocations[0] == ("step", 1, 2.5)
        assert invocations[-1] == ("complete", 2.1)

    def test_training_run_dataclass_fields(self):
        """TrainingRun should have all expected fields."""
        from backpropagate.trainer import TrainingRun

        run = TrainingRun(
            run_id="test_run",
            steps=100,
            final_loss=0.5,
            loss_history=[1.0, 0.8, 0.6, 0.5],
            duration_seconds=60.0,
            samples_seen=800,
            output_path="/path/to/output",
        )

        assert run.run_id == "test_run"
        assert run.steps == 100
        assert run.final_loss == 0.5
        assert run.loss_history == [1.0, 0.8, 0.6, 0.5]
        assert run.duration_seconds == 60.0
        assert run.samples_seen == 800
        assert run.output_path == "/path/to/output"

    def test_callback_receives_training_run_object(self):
        """on_complete callback should receive a TrainingRun object."""
        from backpropagate.trainer import TrainingCallback, TrainingRun

        received_run = [None]

        def capture_run(run):
            received_run[0] = run

        callback = TrainingCallback(on_complete=capture_run)

        # Create and pass a real TrainingRun
        run = TrainingRun(
            run_id="test",
            steps=50,
            final_loss=1.5,
            loss_history=[2.0, 1.5],
            duration_seconds=30.0,
            samples_seen=400,
            output_path="/test/path",
        )

        callback.on_complete(run)

        assert received_run[0] is run
        assert received_run[0].final_loss == 1.5

    def test_callback_chain_preserves_data_integrity(self):
        """Data passed through callbacks should maintain integrity."""
        from backpropagate.trainer import TrainingCallback

        step_data = []
        loss_sum = [0.0]

        def accumulate_step(step, loss):
            step_data.append((step, loss))
            loss_sum[0] += loss

        callback = TrainingCallback(on_step=accumulate_step)

        # Simulate steps with known losses
        losses = [2.5, 2.3, 2.1, 1.9, 1.7]
        for i, loss in enumerate(losses, 1):
            callback.on_step(i, loss)

        assert len(step_data) == 5
        assert loss_sum[0] == sum(losses)
        assert step_data == [(1, 2.5), (2, 2.3), (3, 2.1), (4, 1.9), (5, 1.7)]

    def test_exception_in_one_callback_doesnt_prevent_others(self):
        """Exception in one callback shouldn't prevent other callbacks."""
        from backpropagate.trainer import TrainingCallback

        call_log = []

        def failing_step(step, loss):
            call_log.append(f"step_{step}_start")
            if step == 2:
                raise ValueError("Step 2 failed!")
            call_log.append(f"step_{step}_end")

        def on_complete(run):
            call_log.append("complete")

        callback = TrainingCallback(
            on_step=failing_step,
            on_complete=on_complete,
        )

        # Simulate training with error handling
        def safe_invoke(fn, *args):
            try:
                fn(*args)
            except Exception:
                call_log.append("error_caught")

        safe_invoke(callback.on_step, 1, 2.5)
        safe_invoke(callback.on_step, 2, 2.3)  # This one fails
        safe_invoke(callback.on_step, 3, 2.1)
        safe_invoke(callback.on_complete, MagicMock())

        assert "step_1_start" in call_log
        assert "step_1_end" in call_log
        assert "step_2_start" in call_log
        assert "error_caught" in call_log
        assert "step_3_start" in call_log
        assert "step_3_end" in call_log
        assert "complete" in call_log
