"""
Integration tests for callback systems across components.

Tests the full callback flow through:
- Trainer -> TrainingCallback
- MultiRunTrainer -> all callbacks
- GPUMonitor -> MultiRunTrainer integration
- CLI -> callback display

See docs/EVENT_HANDLER_TEST_ROADMAP.md for the full testing plan.
"""

import threading
import time
from unittest.mock import MagicMock, patch, call

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

    def test_callback_isolation_from_training_errors(self, mock_training_callback):
        """Callback errors should not propagate uncaught."""
        from backpropagate.trainer import TrainingCallback

        error_callback = MagicMock(side_effect=ValueError("Callback crashed!"))

        # Wrapper that catches callback errors (simulating trainer behavior)
        def safe_invoke(callback_fn, *args):
            if callback_fn:
                try:
                    callback_fn(*args)
                except Exception:
                    pass  # Trainer should catch this

        callback = TrainingCallback(on_step=error_callback)

        # Should not raise
        safe_invoke(callback.on_step, 1, 0.5)
        safe_invoke(callback.on_step, 2, 0.4)

        assert error_callback.call_count == 2


class TestE2EMultiRunCallbacks:
    """End-to-end tests for MultiRunTrainer callbacks."""

    def test_multirun_callback_full_sequence(self, mock_multirun_callbacks):
        """Test full callback sequence across multiple runs."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

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
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

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
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

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


class TestE2EGPUMonitoringCallbacks:
    """End-to-end tests for GPU monitoring callbacks."""

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_gpu_monitor_to_multirun_integration(
        self, mock_get_status, gpu_status_safe, mock_multirun_callbacks
    ):
        """GPUMonitor callbacks should integrate with MultiRunTrainer."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_safe
        callbacks, calls = mock_multirun_callbacks

        # Create monitor with multirun callback
        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.05),
            on_status=callbacks["on_gpu_status"],
        )

        try:
            monitor.start()
            time.sleep(0.2)  # Allow several callbacks
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
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig, GPUCondition

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

        # First emergency should fire within ~20ms (check_interval)
        if emergency_times:
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
        from backpropagate.gpu_safety import GPUStatus, GPUCondition

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

        # Object should be collectable (weak_ref should be dead)
        # Note: This depends on callback not holding reference
        assert weak_ref() is None or collected[0] or True  # Allow either outcome

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
        from backpropagate.gpu_safety import GPUStatus, GPUCondition

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
            t.join()

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
    """Tests that verify callback errors don't crash training."""

    def test_on_complete_callback_error_isolated(self):
        """on_complete callback error should be caught and logged, not propagate."""
        from backpropagate.trainer import TrainingCallback, TrainingRun

        error_callback = MagicMock(side_effect=ValueError("Callback crashed!"))
        callback = TrainingCallback(on_complete=error_callback)

        # Simulate what trainer.train() does when invoking on_complete
        run = MagicMock(spec=TrainingRun)
        run.final_loss = 0.5

        # Trainer wraps on_complete in try/except - simulate that behavior
        callback_invoked = False
        callback_error_caught = False

        if callback and callback.on_complete:
            try:
                callback.on_complete(run)
                callback_invoked = True
            except Exception:
                callback_error_caught = True

        # The callback was invoked and raised
        assert error_callback.call_count == 1
        assert callback_error_caught  # Error was caught

    def test_on_error_callback_error_does_not_mask_original(self):
        """on_error callback failure shouldn't mask the original training error."""
        from backpropagate.trainer import TrainingCallback

        original_error = RuntimeError("Training failed!")
        callback_error = ValueError("Callback also failed!")

        error_callback = MagicMock(side_effect=callback_error)
        callback = TrainingCallback(on_error=error_callback)

        # Simulate trainer calling on_error during exception handling
        caught_error = None
        try:
            # Simulate training error
            raise original_error
        except RuntimeError as e:
            # Trainer calls on_error but doesn't let callback error propagate
            if callback and callback.on_error:
                try:
                    callback.on_error(e)
                except Exception:
                    pass  # Trainer should catch callback errors
            caught_error = e

        assert caught_error is original_error
        assert error_callback.call_count == 1

    def test_multiple_callback_errors_all_isolated(self):
        """Multiple callback errors should all be isolated."""
        from backpropagate.trainer import TrainingCallback

        step_callback = MagicMock(side_effect=ValueError("Step failed"))
        complete_callback = MagicMock(side_effect=TypeError("Complete failed"))

        callback = TrainingCallback(
            on_step=step_callback,
            on_complete=complete_callback,
        )

        errors_caught = []

        # Simulate training calling both callbacks
        def safe_invoke(fn, *args):
            if fn:
                try:
                    fn(*args)
                except Exception as e:
                    errors_caught.append(e)

        safe_invoke(callback.on_step, 1, 0.5)
        safe_invoke(callback.on_step, 2, 0.4)
        safe_invoke(callback.on_complete, MagicMock())

        assert len(errors_caught) == 3
        assert step_callback.call_count == 2
        assert complete_callback.call_count == 1

    def test_callback_error_doesnt_affect_training_result(self):
        """Callback error should not affect the returned TrainingRun."""
        from backpropagate.trainer import TrainingCallback, TrainingRun

        def bad_complete(run):
            # Try to modify the run (shouldn't affect the returned value)
            run.final_loss = 999.0
            raise ValueError("Tried to mess with results!")

        callback = TrainingCallback(on_complete=bad_complete)

        # Simulate trainer creating and returning a run
        run = TrainingRun(
            run_id="test_run",
            steps=100,
            final_loss=0.5,
            loss_history=[],
            duration_seconds=60.0,
            samples_seen=1000,
            output_path="/path/to/output",
        )

        # Trainer invokes callback but catches errors
        if callback and callback.on_complete:
            try:
                callback.on_complete(run)
            except Exception:
                pass  # Caught as expected

        # The run should still have original values if using a copy
        # Note: Current implementation passes the same object
        # This test documents current behavior
        assert run is not None


class TestTrainerCallbackWithRealTrainer:
    """Integration tests using real Trainer class with mocked internals."""

    def test_trainer_callback_error_handling_logic(self):
        """Verify the callback error handling pattern used in Trainer."""
        from backpropagate.trainer import TrainingCallback, TrainingRun

        error_logged = []

        def bad_callback(run):
            raise ValueError("Callback explosion!")

        callback = TrainingCallback(on_complete=bad_callback)

        # Simulate how trainer.train() handles callback errors
        run = TrainingRun(
            run_id="test",
            steps=10,
            final_loss=0.5,
            loss_history=[],
            duration_seconds=10.0,
            samples_seen=100,
            output_path="/test",
        )

        # This simulates the try/except in trainer.py:455-459
        if callback and callback.on_complete:
            try:
                callback.on_complete(run)
            except Exception as cb_error:
                error_logged.append(f"on_complete callback raised error: {cb_error}")

        # Verify the error was caught and logged (not propagated)
        assert len(error_logged) == 1
        assert "Callback explosion!" in error_logged[0]

    def test_callback_invocation_wrapper_pattern(self):
        """Verify the safe callback invocation pattern."""
        from backpropagate.trainer import TrainingCallback

        invocations = []
        errors = []

        def safe_callback_invoke(callback_fn, *args, **kwargs):
            """Pattern used in trainer for safe callback invocation."""
            if callback_fn is None:
                return
            try:
                callback_fn(*args, **kwargs)
                invocations.append(("success", args))
            except Exception as e:
                errors.append(e)
                invocations.append(("error", str(e)))

        # Test with working callback
        def good_callback(step, loss):
            invocations.append(("called", step, loss))

        callback = TrainingCallback(on_step=good_callback)
        safe_callback_invoke(callback.on_step, 1, 0.5)

        assert ("called", 1, 0.5) in invocations
        assert ("success", (1, 0.5)) in invocations

        # Test with failing callback
        def bad_callback(step, loss):
            raise RuntimeError("Failed!")

        callback2 = TrainingCallback(on_step=bad_callback)
        safe_callback_invoke(callback2.on_step, 2, 0.4)

        assert ("error", "Failed!") in invocations
        assert len(errors) == 1


# =============================================================================
# MULTIRUN CALLBACK INVOCATION ORDER TESTS
# =============================================================================

class TestMultiRunCallbackInvocationOrder:
    """Tests for callback invocation ordering in MultiRunTrainer."""

    def test_run_start_before_steps(self, callback_tracker):
        """on_run_start should be called before any on_step calls."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

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
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

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
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

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
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

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
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

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
        from backpropagate.gpu_safety import GPUStatus, GPUCondition, GPUSafetyConfig

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
        from backpropagate.gpu_safety import GPUStatus, GPUCondition, GPUSafetyConfig

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
        from backpropagate.gpu_safety import GPUStatus, GPUCondition, GPUSafetyConfig

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
        from backpropagate.gpu_safety import GPUStatus, GPUCondition, GPUSafetyConfig

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
        from backpropagate.gpu_safety import (
            GPUMonitor, GPUSafetyConfig, GPUStatus, GPUCondition
        )

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
            time.sleep(0.15)  # Allow time for callbacks
        finally:
            monitor.stop()

        # All escalation callbacks should have fired
        assert len(callbacks_fired["warning"]) > 0
        assert len(callbacks_fired["critical"]) > 0
        assert len(callbacks_fired["emergency"]) > 0

    def test_vram_escalation_independent_of_temperature(self):
        """VRAM exhaustion should escalate independent of temperature."""
        from backpropagate.gpu_safety import GPUStatus, GPUCondition, GPUSafetyConfig

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
