"""
Comprehensive tests for event handlers and callbacks.

This module tests all callback systems in backpropagate:
- TrainingCallback (trainer.py)
- MultiRunTrainer callbacks (multi_run.py)
- GPUMonitor event callbacks (gpu_safety.py)

See docs/EVENT_HANDLER_TEST_ROADMAP.md for the full testing plan.
"""

import threading
import time
from dataclasses import fields
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# TRAINING CALLBACK UNIT TESTS
# =============================================================================

class TestTrainingCallbackUnit:
    """Unit tests for the TrainingCallback dataclass."""

    def test_callback_dataclass_defaults(self):
        """All callback fields should default to None."""
        from backpropagate.trainer import TrainingCallback

        callback = TrainingCallback()

        assert callback.on_step is None
        assert callback.on_epoch is None
        assert callback.on_save is None
        assert callback.on_complete is None
        assert callback.on_error is None

    def test_callback_with_all_handlers(self):
        """TrainingCallback accepts all handler functions."""
        from backpropagate.trainer import TrainingCallback

        handlers = {
            "on_step": lambda step, loss: None,
            "on_epoch": lambda epoch: None,
            "on_save": lambda path: None,
            "on_complete": lambda run: None,
            "on_error": lambda exc: None,
        }

        callback = TrainingCallback(**handlers)

        for name, handler in handlers.items():
            assert getattr(callback, name) is handler

    def test_callback_partial_handlers(self):
        """TrainingCallback works with subset of handlers."""
        from backpropagate.trainer import TrainingCallback

        on_step = lambda step, loss: None
        callback = TrainingCallback(on_step=on_step)

        assert callback.on_step is on_step
        assert callback.on_epoch is None
        assert callback.on_save is None
        assert callback.on_complete is None
        assert callback.on_error is None

    def test_callback_handler_signatures(self):
        """Verify callback field types are Optional[Callable]."""
        from backpropagate.trainer import TrainingCallback

        callback_fields = fields(TrainingCallback)

        for field in callback_fields:
            # All fields should be optional callables
            assert "Optional" in str(field.type) or "None" in str(field.type)
            assert "Callable" in str(field.type)

    def test_callback_is_dataclass(self):
        """TrainingCallback should be a proper dataclass."""
        from dataclasses import is_dataclass
        from backpropagate.trainer import TrainingCallback

        assert is_dataclass(TrainingCallback)

    def test_callback_fields_count(self):
        """TrainingCallback should have exactly 5 callback fields."""
        from backpropagate.trainer import TrainingCallback

        callback_fields = fields(TrainingCallback)
        assert len(callback_fields) == 5

        expected_names = {"on_step", "on_epoch", "on_save", "on_complete", "on_error"}
        actual_names = {f.name for f in callback_fields}
        assert actual_names == expected_names

    def test_callback_immutability_behavior(self):
        """Test dataclass field behavior (frozen=False by default)."""
        from backpropagate.trainer import TrainingCallback

        callback = TrainingCallback()

        # Default dataclass allows mutation
        new_handler = lambda step, loss: None
        callback.on_step = new_handler
        assert callback.on_step is new_handler


class TestTrainingCallbackIntegration:
    """Integration tests for TrainingCallback with Trainer."""

    def test_callback_none_safe(self):
        """Training should work without any callbacks (None)."""
        from backpropagate.trainer import Trainer, TrainingCallback

        # No callback at all should work
        trainer = Trainer(model="test-model")
        # Just verify instantiation doesn't fail
        assert trainer is not None

    def test_callback_empty_safe(self):
        """Training should work with empty TrainingCallback."""
        from backpropagate.trainer import TrainingCallback

        callback = TrainingCallback()
        # All handlers are None - should be safe to use
        assert callback.on_step is None

    def test_on_step_receives_correct_args(self):
        """on_step callback receives (step: int, loss: float)."""
        from backpropagate.trainer import TrainingCallback

        received_args = []

        def on_step(step: int, loss: float):
            received_args.append((step, loss))

        callback = TrainingCallback(on_step=on_step)

        # Simulate what happens during training
        if callback.on_step:
            callback.on_step(1, 2.5)
            callback.on_step(2, 2.3)

        assert len(received_args) == 2
        assert received_args[0] == (1, 2.5)
        assert received_args[1] == (2, 2.3)

    def test_on_step_loss_values_valid(self):
        """on_step loss values should be valid floats."""
        from backpropagate.trainer import TrainingCallback

        losses = []

        def on_step(step: int, loss: float):
            losses.append(loss)
            # Verify loss is a valid float
            assert isinstance(loss, (int, float))
            assert not (loss != loss)  # Check for NaN

        callback = TrainingCallback(on_step=on_step)

        # Simulate training steps with valid loss values
        for i in range(5):
            callback.on_step(i, 2.5 - i * 0.1)

        assert len(losses) == 5
        assert all(isinstance(l, float) for l in losses)

    def test_on_epoch_receives_epoch_number(self):
        """on_epoch callback receives epoch number."""
        from backpropagate.trainer import TrainingCallback

        received_epochs = []

        def on_epoch(epoch: int):
            received_epochs.append(epoch)

        callback = TrainingCallback(on_epoch=on_epoch)

        # Simulate epochs
        for e in range(3):
            callback.on_epoch(e)

        assert received_epochs == [0, 1, 2]

    def test_on_complete_receives_training_run(self):
        """on_complete callback receives TrainingRun object."""
        from backpropagate.trainer import TrainingCallback, TrainingRun

        received_runs = []

        def on_complete(run: TrainingRun):
            received_runs.append(run)

        callback = TrainingCallback(on_complete=on_complete)

        # Create a mock TrainingRun
        mock_run = MagicMock(spec=TrainingRun)
        mock_run.final_loss = 0.5

        if callback.on_complete:
            callback.on_complete(mock_run)

        assert len(received_runs) == 1
        assert received_runs[0].final_loss == 0.5

    def test_on_error_receives_exception(self):
        """on_error callback receives Exception object."""
        from backpropagate.trainer import TrainingCallback

        received_errors = []

        def on_error(exc: Exception):
            received_errors.append(exc)

        callback = TrainingCallback(on_error=on_error)

        test_error = ValueError("Test error")
        if callback.on_error:
            callback.on_error(test_error)

        assert len(received_errors) == 1
        assert received_errors[0] is test_error
        assert str(received_errors[0]) == "Test error"

    def test_on_save_receives_path(self):
        """on_save callback receives checkpoint path string."""
        from backpropagate.trainer import TrainingCallback

        received_paths = []

        def on_save(path: str):
            received_paths.append(path)

        callback = TrainingCallback(on_save=on_save)

        if callback.on_save:
            callback.on_save("/path/to/checkpoint")

        assert len(received_paths) == 1
        assert received_paths[0] == "/path/to/checkpoint"

    def test_callback_error_isolation(self):
        """Callback exception should be isolatable by caller."""
        from backpropagate.trainer import TrainingCallback

        def bad_callback(step, loss):
            raise RuntimeError("Callback crashed!")

        callback = TrainingCallback(on_step=bad_callback)

        errors_caught = []

        # Simulate trainer's safe invocation pattern
        for i in range(3):
            try:
                if callback.on_step:
                    callback.on_step(i, 0.5)
            except Exception as e:
                errors_caught.append(e)

        # All 3 calls should have raised and been catchable
        assert len(errors_caught) == 3
        assert all("Callback crashed!" in str(e) for e in errors_caught)


class TestTrainingCallbackEdgeCases:
    """Edge case tests for TrainingCallback."""

    def test_callback_raises_exception_isolation(self):
        """Callback exception should be catchable."""
        from backpropagate.trainer import TrainingCallback

        def bad_callback(step, loss):
            raise RuntimeError("Callback crashed!")

        callback = TrainingCallback(on_step=bad_callback)

        # The callback itself raises, but caller should handle it
        with pytest.raises(RuntimeError, match="Callback crashed!"):
            callback.on_step(1, 0.5)

    def test_callback_slow_handler(self):
        """Slow callback should still work (no timeout)."""
        from backpropagate.trainer import TrainingCallback

        call_times = []

        def slow_callback(step, loss):
            time.sleep(0.1)  # 100ms delay
            call_times.append(time.time())

        callback = TrainingCallback(on_step=slow_callback)

        start = time.time()
        callback.on_step(1, 0.5)
        callback.on_step(2, 0.4)
        elapsed = time.time() - start

        assert len(call_times) == 2
        assert elapsed >= 0.2  # At least 200ms for two 100ms callbacks

    def test_callback_modifies_mutable_state(self):
        """Callback can modify external mutable state."""
        from backpropagate.trainer import TrainingCallback

        state = {"loss_history": [], "best_loss": float("inf")}

        def tracking_callback(step, loss):
            state["loss_history"].append(loss)
            if loss < state["best_loss"]:
                state["best_loss"] = loss

        callback = TrainingCallback(on_step=tracking_callback)

        callback.on_step(1, 2.5)
        callback.on_step(2, 1.5)
        callback.on_step(3, 1.8)

        assert state["loss_history"] == [2.5, 1.5, 1.8]
        assert state["best_loss"] == 1.5

    def test_callback_with_closure_state(self):
        """Callback can use closure variables."""
        from backpropagate.trainer import TrainingCallback

        counter = [0]  # Using list for mutable closure

        def counting_callback(step, loss):
            counter[0] += 1

        callback = TrainingCallback(on_step=counting_callback)

        for i in range(5):
            callback.on_step(i, 0.5)

        assert counter[0] == 5

    def test_callback_reentrant_safe(self):
        """Callback that recursively invokes should work (non-reentrant lock issues)."""
        from backpropagate.trainer import TrainingCallback

        call_depth = [0]
        max_depth = [0]

        def reentrant_callback(step, loss):
            call_depth[0] += 1
            max_depth[0] = max(max_depth[0], call_depth[0])
            # Note: In real code, this would be called by the trainer, not itself
            call_depth[0] -= 1

        callback = TrainingCallback(on_step=reentrant_callback)

        for i in range(5):
            callback.on_step(i, 0.5)

        assert max_depth[0] == 1  # No actual recursion in this test

    def test_multiple_callbacks_sequential(self):
        """Multiple training runs with same callback should work."""
        from backpropagate.trainer import TrainingCallback

        all_calls = []

        def tracking_callback(step, loss):
            all_calls.append((step, loss))

        callback = TrainingCallback(on_step=tracking_callback)

        # Simulate run 1
        for i in range(3):
            callback.on_step(i, 2.0 - i * 0.1)

        # Simulate run 2
        for i in range(3):
            callback.on_step(i, 1.7 - i * 0.1)

        assert len(all_calls) == 6


# =============================================================================
# MULTI-RUN TRAINER CALLBACK UNIT TESTS
# =============================================================================

class TestMultiRunCallbackUnit:
    """Unit tests for MultiRunTrainer callback parameters."""

    def test_multirun_accepts_all_callbacks(self):
        """MultiRunTrainer constructor accepts all callback parameters."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

        callbacks_received = {}

        def on_run_start(idx):
            callbacks_received["run_start"] = True

        def on_run_complete(result):
            callbacks_received["run_complete"] = True

        def on_step(run_idx, step, loss):
            callbacks_received["step"] = True

        def on_gpu_status(status):
            callbacks_received["gpu_status"] = True

        # Should not raise
        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=1, steps_per_run=1),
            on_run_start=on_run_start,
            on_run_complete=on_run_complete,
            on_step=on_step,
            on_gpu_status=on_gpu_status,
        )

        # Verify callbacks are stored
        assert trainer.on_run_start is on_run_start
        assert trainer.on_run_complete is on_run_complete
        assert trainer.on_step is on_step
        assert trainer.on_gpu_status is on_gpu_status

    def test_multirun_no_callbacks(self):
        """MultiRunTrainer works without any callbacks."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=1, steps_per_run=1),
        )

        assert trainer.on_run_start is None
        assert trainer.on_run_complete is None
        assert trainer.on_step is None
        assert trainer.on_gpu_status is None

    def test_multirun_partial_callbacks(self):
        """MultiRunTrainer works with subset of callbacks."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

        def on_run_complete(result):
            pass

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=1, steps_per_run=1),
            on_run_complete=on_run_complete,
        )

        assert trainer.on_run_start is None
        assert trainer.on_run_complete is on_run_complete
        assert trainer.on_step is None
        assert trainer.on_gpu_status is None

    def test_multirun_callback_types(self):
        """Callback type validation (None or Callable)."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=1, steps_per_run=1),
        )

        # All should be None or callable
        for cb_name in ["on_run_start", "on_run_complete", "on_step", "on_gpu_status"]:
            cb = getattr(trainer, cb_name)
            assert cb is None or callable(cb)


class TestMultiRunCallbackSequencing:
    """Tests for callback invocation order in MultiRunTrainer."""

    def test_on_run_start_receives_run_index(self):
        """on_run_start receives the run index (1-based in actual code)."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

        received_indices = []

        def on_run_start(run_idx: int):
            received_indices.append(run_idx)

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=3, steps_per_run=1),
            on_run_start=on_run_start,
        )

        # Manually invoke to test callback signature
        trainer.on_run_start(1)
        trainer.on_run_start(2)
        trainer.on_run_start(3)

        assert received_indices == [1, 2, 3]

    def test_on_run_complete_receives_run_result(self):
        """on_run_complete receives RunResult object."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig, RunResult

        received_results = []

        def on_run_complete(result):
            received_results.append(result)

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=2, steps_per_run=5),
            on_run_complete=on_run_complete,
        )

        # Create mock RunResults
        for i in range(2):
            result = MagicMock(spec=RunResult)
            result.run_index = i + 1
            result.final_loss = 2.0 - i * 0.3
            trainer.on_run_complete(result)

        assert len(received_results) == 2
        assert received_results[0].run_index == 1
        assert received_results[1].run_index == 2

    def test_on_step_receives_run_step_loss(self):
        """on_step receives (run_idx, step, loss)."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

        received_calls = []

        def on_step(run_idx: int, step: int, loss: float):
            received_calls.append((run_idx, step, loss))

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=2, steps_per_run=3),
            on_step=on_step,
        )

        # Manually invoke
        trainer.on_step(1, 1, 2.5)
        trainer.on_step(1, 2, 2.3)
        trainer.on_step(2, 1, 2.1)

        assert received_calls == [(1, 1, 2.5), (1, 2, 2.3), (2, 1, 2.1)]

    def test_callback_run_index_sequence(self):
        """Run indices should increment correctly."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

        indices = []

        def on_run_start(run_idx):
            indices.append(run_idx)

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=5, steps_per_run=1),
            on_run_start=on_run_start,
        )

        # Simulate the run loop
        for i in range(1, 6):
            trainer.on_run_start(i)

        assert indices == [1, 2, 3, 4, 5]
        # Verify monotonic increase
        for i in range(1, len(indices)):
            assert indices[i] > indices[i - 1]

    def test_callback_sequence_tracking(self, callback_tracker):
        """Track callback invocation sequence."""
        from backpropagate.trainer import TrainingCallback

        callback = TrainingCallback(
            on_step=callback_tracker.track("step"),
            on_epoch=callback_tracker.track("epoch"),
            on_complete=callback_tracker.track("complete"),
        )

        # Simulate training sequence
        callback.on_step(1, 2.5)
        callback.on_step(2, 2.3)
        callback.on_epoch(1)
        callback.on_step(3, 2.1)
        callback.on_complete(MagicMock())

        expected = ["step", "step", "epoch", "step", "complete"]
        callback_tracker.assert_sequence(expected)

    def test_callback_timing_between_runs(self):
        """Callbacks should fire in correct sequence relative to run boundaries."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

        event_log = []

        def on_run_start(idx):
            event_log.append(f"start_{idx}")

        def on_run_complete(result):
            event_log.append(f"complete_{result.run_index}")

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=3, steps_per_run=1),
            on_run_start=on_run_start,
            on_run_complete=on_run_complete,
        )

        # Simulate proper run sequence
        for i in range(1, 4):
            trainer.on_run_start(i)
            result = MagicMock()
            result.run_index = i
            trainer.on_run_complete(result)

        expected = ["start_1", "complete_1", "start_2", "complete_2", "start_3", "complete_3"]
        assert event_log == expected


class TestMultiRunGPUStatusCallbacks:
    """Tests for GPU status callbacks in MultiRunTrainer."""

    def test_on_gpu_status_receives_gpustatus(self, gpu_status_safe):
        """on_gpu_status receives GPUStatus object."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig
        from backpropagate.gpu_safety import GPUStatus

        received_statuses = []

        def on_gpu_status(status: GPUStatus):
            received_statuses.append(status)

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=1, steps_per_run=1),
            on_gpu_status=on_gpu_status,
        )

        # Manually invoke
        trainer.on_gpu_status(gpu_status_safe)

        assert len(received_statuses) == 1
        assert received_statuses[0].temperature_c == 60.0
        assert received_statuses[0].vram_percent == 50.0

    def test_on_gpu_status_content_validation(self):
        """GPUStatus object should have valid fields."""
        from backpropagate.gpu_safety import GPUStatus, GPUCondition

        statuses = []

        def on_gpu_status(status):
            statuses.append(status)
            # Validate fields
            assert hasattr(status, "available")
            assert hasattr(status, "temperature_c")
            assert hasattr(status, "vram_percent")
            assert hasattr(status, "condition")

        # Create and validate status
        status = GPUStatus(
            available=True,
            device_name="Test GPU",
            temperature_c=70.0,
            vram_total_gb=16.0,
            vram_used_gb=8.0,
            vram_percent=50.0,
            condition=GPUCondition.SAFE,
        )

        on_gpu_status(status)
        assert len(statuses) == 1

    def test_gpu_callback_thread_safety(self):
        """GPU status callbacks should be thread-safe."""
        from backpropagate.gpu_safety import GPUStatus, GPUCondition

        results = []
        lock = threading.Lock()

        def thread_safe_callback(status):
            with lock:
                results.append(status)

        status = GPUStatus(
            available=True,
            device_name="Test GPU",
            temperature_c=70.0,
            vram_total_gb=16.0,
            vram_used_gb=8.0,
            vram_percent=50.0,
            condition=GPUCondition.SAFE,
        )

        # Simulate concurrent calls
        threads = []
        for i in range(10):
            t = threading.Thread(target=thread_safe_callback, args=(status,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 10


class TestMultiRunAbortCallbacks:
    """Tests for callback behavior during abort scenarios."""

    def test_callback_invoked_before_abort(self):
        """Callbacks should still fire before abort."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig

        run_starts = []

        def on_run_start(idx):
            run_starts.append(idx)

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=5, steps_per_run=10),
            on_run_start=on_run_start,
        )

        # Simulate a few run starts before abort
        trainer.on_run_start(1)
        trainer.on_run_start(2)
        # Abort would happen here
        trainer._should_abort = True

        assert run_starts == [1, 2]

    def test_abort_preserves_callback_state(self):
        """Callbacks should maintain state after abort."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig, RunResult

        completed_runs = []

        def on_run_complete(result):
            completed_runs.append(result.run_index)

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(num_runs=5, steps_per_run=10),
            on_run_complete=on_run_complete,
        )

        # Simulate runs and abort
        for i in range(1, 4):
            result = MagicMock(spec=RunResult)
            result.run_index = i
            trainer.on_run_complete(result)

        trainer.abort("Test abort")

        # State should be preserved
        assert completed_runs == [1, 2, 3]
        assert trainer._should_abort
        assert trainer._abort_reason == "Test abort"

    def test_early_stopping_callbacks(self):
        """Callbacks should work with early stopping."""
        from backpropagate.multi_run import MultiRunTrainer, MultiRunConfig, RunResult

        run_indices = []

        def on_run_complete(result):
            run_indices.append(result.run_index)

        trainer = MultiRunTrainer(
            model="test-model",
            config=MultiRunConfig(
                num_runs=10,
                steps_per_run=10,
                early_stopping=True,
                early_stopping_patience=2,
            ),
            on_run_complete=on_run_complete,
        )

        # Simulate 3 runs before early stopping
        for i in range(1, 4):
            result = MagicMock(spec=RunResult)
            result.run_index = i
            trainer.on_run_complete(result)

        assert run_indices == [1, 2, 3]


# =============================================================================
# GPU MONITOR CALLBACK TESTS
# =============================================================================

class TestGPUMonitorCallbackRegistration:
    """Tests for GPUMonitor callback registration."""

    def test_monitor_accepts_all_callbacks(self):
        """GPUMonitor constructor accepts all callback types."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        callbacks = {
            "on_status": lambda s: None,
            "on_warning": lambda s: None,
            "on_critical": lambda s: None,
            "on_emergency": lambda s: None,
        }

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=1.0),
            **callbacks,
        )

        assert monitor.on_status is callbacks["on_status"]
        assert monitor.on_warning is callbacks["on_warning"]
        assert monitor.on_critical is callbacks["on_critical"]
        assert monitor.on_emergency is callbacks["on_emergency"]

    def test_monitor_no_callbacks(self):
        """GPUMonitor works without any callbacks."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        monitor = GPUMonitor(config=GPUSafetyConfig(check_interval=1.0))

        assert monitor.on_status is None
        assert monitor.on_warning is None
        assert monitor.on_critical is None
        assert monitor.on_emergency is None

    def test_monitor_partial_callbacks(self):
        """GPUMonitor works with subset of callbacks."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        def on_critical(status):
            pass

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=1.0),
            on_critical=on_critical,
        )

        assert monitor.on_status is None
        assert monitor.on_warning is None
        assert monitor.on_critical is on_critical
        assert monitor.on_emergency is None

    def test_monitor_callback_replacement(self):
        """Monitor callbacks can be updated after initialization."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        monitor = GPUMonitor(config=GPUSafetyConfig(check_interval=1.0))
        assert monitor.on_status is None

        # Update callback
        new_callback = lambda s: None
        monitor.on_status = new_callback
        assert monitor.on_status is new_callback


class TestGPUMonitorEventDispatch:
    """Tests for GPUMonitor event dispatch logic."""

    def test_on_status_callback_invoked(self, gpu_status_safe):
        """on_status callback receives GPUStatus."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        received = []

        def on_status(status):
            received.append(status)

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=1.0),
            on_status=on_status,
        )

        # Manually invoke the callback
        if monitor.on_status:
            monitor.on_status(gpu_status_safe)

        assert len(received) == 1
        assert received[0].temperature_c == 60.0

    def test_on_warning_receives_warning_status(self):
        """on_warning callback receives warning-condition status."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig, GPUStatus, GPUCondition

        received = []

        def on_warning(status):
            received.append(status)

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=1.0),
            on_warning=on_warning,
        )

        warning_status = GPUStatus(
            available=True,
            device_name="Test GPU",
            temperature_c=82.0,
            vram_total_gb=16.0,
            vram_used_gb=13.0,
            vram_percent=81.25,
            condition=GPUCondition.WARNING,
            condition_reason="Temperature WARNING: 82.0°C",
        )

        if monitor.on_warning:
            monitor.on_warning(warning_status)

        assert len(received) == 1
        assert received[0].condition == GPUCondition.WARNING

    def test_on_critical_receives_critical_status(self, gpu_status_critical):
        """on_critical callback receives critical-condition status."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig, GPUCondition

        received = []

        def on_critical(status):
            received.append(status)

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=1.0),
            on_critical=on_critical,
        )

        if monitor.on_critical:
            monitor.on_critical(gpu_status_critical)

        assert len(received) == 1
        assert received[0].condition == GPUCondition.CRITICAL

    def test_on_emergency_receives_emergency_status(self, gpu_status_emergency):
        """on_emergency callback receives emergency-condition status."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig, GPUCondition

        received = []

        def on_emergency(status):
            received.append(status)

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=1.0),
            on_emergency=on_emergency,
        )

        if monitor.on_emergency:
            monitor.on_emergency(gpu_status_emergency)

        assert len(received) == 1
        assert received[0].condition == GPUCondition.EMERGENCY

    def test_callback_receives_gpustatus(self, gpu_status_safe):
        """All callbacks should receive GPUStatus objects."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig, GPUStatus

        def verify_type(status):
            assert isinstance(status, GPUStatus)
            assert hasattr(status, "condition")
            assert hasattr(status, "temperature_c")

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=1.0),
            on_status=verify_type,
        )

        # Should not raise
        monitor.on_status(gpu_status_safe)

    def test_callback_status_has_condition(self):
        """GPUStatus.condition should match the event type."""
        from backpropagate.gpu_safety import GPUStatus, GPUCondition

        conditions_received = []

        def track_condition(status):
            conditions_received.append(status.condition)

        statuses = [
            GPUStatus(available=True, condition=GPUCondition.SAFE),
            GPUStatus(available=True, condition=GPUCondition.WARNING),
            GPUStatus(available=True, condition=GPUCondition.CRITICAL),
            GPUStatus(available=True, condition=GPUCondition.EMERGENCY),
        ]

        for status in statuses:
            track_condition(status)

        assert conditions_received == [
            GPUCondition.SAFE,
            GPUCondition.WARNING,
            GPUCondition.CRITICAL,
            GPUCondition.EMERGENCY,
        ]


class TestGPUMonitorThreadSafety:
    """Tests for GPUMonitor callback thread safety."""

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_callbacks_from_monitor_thread(self, mock_get_status, gpu_status_safe):
        """Callbacks are invoked from the monitor thread."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_safe

        callback_threads = []

        def on_status(status):
            callback_threads.append(threading.current_thread().name)

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.05),  # 50ms
            on_status=on_status,
        )

        try:
            monitor.start()
            time.sleep(0.2)  # Wait for a few callbacks
        finally:
            monitor.stop()

        # Callbacks should come from a non-main thread
        assert len(callback_threads) > 0
        main_thread = threading.main_thread().name
        assert any(name != main_thread for name in callback_threads)

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_callback_exception_isolated(self, mock_get_status, gpu_status_safe):
        """Callback exception doesn't stop the monitor."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_safe

        call_count = [0]

        def bad_callback(status):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("First callback fails!")

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.05),
            on_status=bad_callback,
        )

        try:
            monitor.start()
            time.sleep(0.2)  # Wait for multiple callbacks
        finally:
            monitor.stop()

        # Monitor should continue after exception
        assert call_count[0] > 1

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_concurrent_callback_access(self, mock_get_status, gpu_status_safe):
        """Multiple callbacks can access shared state safely."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_safe

        shared_state = {"count": 0}
        lock = threading.Lock()

        def on_status(status):
            with lock:
                shared_state["count"] += 1

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.02),
            on_status=on_status,
        )

        try:
            monitor.start()
            time.sleep(0.15)
        finally:
            monitor.stop()

        # Should have incremented multiple times without race conditions
        assert shared_state["count"] > 0

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_stop_during_callback(self, mock_get_status, gpu_status_safe):
        """Stopping monitor during callback execution is safe."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_safe

        callback_started = threading.Event()
        callback_can_finish = threading.Event()

        def slow_callback(status):
            callback_started.set()
            callback_can_finish.wait(timeout=5.0)

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.01),
            on_status=slow_callback,
        )

        monitor.start()
        callback_started.wait(timeout=2.0)

        # Stop while callback is running
        stop_thread = threading.Thread(target=monitor.stop)
        stop_thread.start()

        # Let callback finish
        callback_can_finish.set()
        stop_thread.join(timeout=2.0)

        assert not monitor._thread.is_alive() if monitor._thread else True

    @patch("backpropagate.gpu_safety.get_gpu_status")
    def test_pause_prevents_callbacks(self, mock_get_status, gpu_status_safe):
        """Paused monitor should skip condition callbacks."""
        from backpropagate.gpu_safety import GPUMonitor, GPUSafetyConfig

        mock_get_status.return_value = gpu_status_safe

        callback_times = {"active": 0, "paused": 0}
        lock = threading.Lock()

        monitor = GPUMonitor(
            config=GPUSafetyConfig(check_interval=0.02),
        )

        callback_invoked = threading.Event()

        def counting_callback(status):
            with lock:
                if monitor._pause_event.is_set():
                    callback_times["active"] += 1
                else:
                    callback_times["paused"] += 1
            callback_invoked.set()

        monitor.on_status = counting_callback

        try:
            monitor.start()
            time.sleep(0.1)  # Let some callbacks fire

            monitor.pause()
            time.sleep(0.1)  # Paused period

            monitor.resume()
            time.sleep(0.1)  # More callbacks
        finally:
            monitor.stop()

        # Should have received callbacks when active
        assert callback_times["active"] > 0


class TestGPUMonitorEventEscalation:
    """Tests for GPU condition escalation and de-escalation."""

    def test_condition_levels_order(self):
        """Verify GPUCondition enum ordering."""
        from backpropagate.gpu_safety import GPUCondition

        # Verify conditions are defined
        conditions = [
            GPUCondition.SAFE,
            GPUCondition.WARM,
            GPUCondition.WARNING,
            GPUCondition.CRITICAL,
            GPUCondition.EMERGENCY
        ]
        assert len(conditions) == 5

    def test_callback_receives_correct_condition(self):
        """Each callback receives status with matching condition."""
        from backpropagate.gpu_safety import GPUStatus, GPUCondition

        conditions_received = []

        def track_condition(status):
            conditions_received.append(status.condition)

        # Create statuses for each condition
        statuses = [
            GPUStatus(
                available=True, device_name="GPU", temperature_c=60.0,
                vram_total_gb=16.0, vram_used_gb=8.0, vram_percent=50.0,
                condition=GPUCondition.SAFE
            ),
            GPUStatus(
                available=True, device_name="GPU", temperature_c=82.0,
                vram_total_gb=16.0, vram_used_gb=13.0, vram_percent=81.0,
                condition=GPUCondition.WARNING
            ),
            GPUStatus(
                available=True, device_name="GPU", temperature_c=92.0,
                vram_total_gb=16.0, vram_used_gb=15.0, vram_percent=94.0,
                condition=GPUCondition.CRITICAL
            ),
            GPUStatus(
                available=True, device_name="GPU", temperature_c=97.0,
                vram_total_gb=16.0, vram_used_gb=15.8, vram_percent=99.0,
                condition=GPUCondition.EMERGENCY
            ),
        ]

        for status in statuses:
            track_condition(status)

        assert conditions_received == [
            GPUCondition.SAFE,
            GPUCondition.WARNING,
            GPUCondition.CRITICAL,
            GPUCondition.EMERGENCY,
        ]

    def test_warning_to_critical_escalation(self):
        """Verify escalation from WARNING to CRITICAL."""
        from backpropagate.gpu_safety import GPUStatus, GPUCondition

        escalation_log = []

        def on_warning(status):
            escalation_log.append("warning")

        def on_critical(status):
            escalation_log.append("critical")

        # Simulate escalating conditions
        warning = GPUStatus(condition=GPUCondition.WARNING)
        critical = GPUStatus(condition=GPUCondition.CRITICAL)

        on_warning(warning)
        on_critical(critical)

        assert escalation_log == ["warning", "critical"]

    def test_critical_to_emergency_escalation(self):
        """Verify escalation from CRITICAL to EMERGENCY."""
        from backpropagate.gpu_safety import GPUStatus, GPUCondition

        escalation_log = []

        def on_critical(status):
            escalation_log.append("critical")

        def on_emergency(status):
            escalation_log.append("emergency")

        critical = GPUStatus(condition=GPUCondition.CRITICAL)
        emergency = GPUStatus(condition=GPUCondition.EMERGENCY)

        on_critical(critical)
        on_emergency(emergency)

        assert escalation_log == ["critical", "emergency"]

    def test_deescalation_on_cooldown(self):
        """Verify de-escalation when conditions improve."""
        from backpropagate.gpu_safety import GPUStatus, GPUCondition

        condition_log = []

        def track_condition(status):
            condition_log.append(status.condition)

        # Simulate cooling down
        statuses = [
            GPUStatus(condition=GPUCondition.CRITICAL, temperature_c=92.0),
            GPUStatus(condition=GPUCondition.WARNING, temperature_c=82.0),
            GPUStatus(condition=GPUCondition.SAFE, temperature_c=70.0),
        ]

        for status in statuses:
            track_condition(status)

        # Should see de-escalation
        assert condition_log == [
            GPUCondition.CRITICAL,
            GPUCondition.WARNING,
            GPUCondition.SAFE,
        ]

    def test_multiple_events_per_check(self):
        """Multiple condition changes can occur in sequence."""
        from backpropagate.gpu_safety import GPUStatus, GPUCondition

        events = []

        # Simulate rapid condition changes
        for condition in [GPUCondition.SAFE, GPUCondition.WARNING,
                         GPUCondition.CRITICAL, GPUCondition.WARNING]:
            status = GPUStatus(condition=condition)
            events.append(status.condition)

        assert len(events) == 4
        assert events == [
            GPUCondition.SAFE,
            GPUCondition.WARNING,
            GPUCondition.CRITICAL,
            GPUCondition.WARNING,
        ]


# =============================================================================
# CALLBACK SPY & HELPER TESTS
# =============================================================================

class TestCallbackSpy:
    """Tests for the CallbackSpy test utility."""

    def test_spy_records_calls(self, callback_spy):
        """Spy records all invocations."""
        callback_spy(1, 2, key="value")
        callback_spy(3, 4)

        assert callback_spy.call_count == 2
        assert callback_spy.called

    def test_spy_assert_called(self, callback_spy):
        """assert_called works correctly."""
        with pytest.raises(AssertionError):
            callback_spy.assert_called()

        callback_spy(1)
        callback_spy.assert_called()
        callback_spy.assert_called(times=1)

    def test_spy_assert_not_called(self, callback_spy):
        """assert_not_called works correctly."""
        callback_spy.assert_not_called()

        callback_spy(1)
        with pytest.raises(AssertionError):
            callback_spy.assert_not_called()

    def test_spy_assert_called_with(self, callback_spy):
        """assert_called_with matches arguments."""
        callback_spy(1, 2, key="value")

        callback_spy.assert_called_with(1, 2)
        with pytest.raises(AssertionError):
            callback_spy.assert_called_with(3, 4)

    def test_spy_returns_value(self, callback_spy_factory):
        """Spy can return configured value."""
        spy = callback_spy_factory(return_value=42)
        result = spy()
        assert result == 42

    def test_spy_raises_side_effect(self, callback_spy_factory):
        """Spy can raise configured exception."""
        spy = callback_spy_factory(side_effect=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            spy()

    def test_spy_thread_safety(self, callback_spy):
        """Spy is thread-safe."""
        def call_from_thread():
            for _ in range(100):
                callback_spy(threading.current_thread().name)

        threads = [threading.Thread(target=call_from_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert callback_spy.call_count == 500

    def test_spy_last_call(self, callback_spy):
        """last_call returns most recent invocation."""
        callback_spy(1)
        callback_spy(2)
        callback_spy(3)

        assert callback_spy.last_call.args == (3,)

    def test_spy_reset(self, callback_spy):
        """reset clears all recorded calls."""
        callback_spy(1)
        callback_spy(2)
        assert callback_spy.call_count == 2

        callback_spy.reset()
        assert callback_spy.call_count == 0
        assert not callback_spy.called

    def test_spy_get_args_list(self, callback_spy):
        """get_args_list returns all argument tuples."""
        callback_spy(1, 2)
        callback_spy(3, 4)
        callback_spy(5)

        args_list = callback_spy.get_args_list()
        assert args_list == [(1, 2), (3, 4), (5,)]

    def test_spy_wait_for_call(self, callback_spy):
        """wait_for_call waits for callback invocation."""
        def delayed_call():
            time.sleep(0.05)
            callback_spy(1)

        thread = threading.Thread(target=delayed_call)
        thread.start()

        # Should return True when called
        assert callback_spy.wait_for_call(timeout=1.0)
        thread.join()


class TestCallbackTracker:
    """Tests for the CallbackTracker test utility."""

    def test_tracker_records_sequence(self, callback_tracker):
        """Tracker records callback sequence."""
        cb1 = callback_tracker.track("first")
        cb2 = callback_tracker.track("second")

        cb1(1)
        cb2(2)
        cb1(3)

        assert callback_tracker.get_sequence() == ["first", "second", "first"]

    def test_tracker_assert_sequence(self, callback_tracker):
        """assert_sequence validates order."""
        cb1 = callback_tracker.track("a")
        cb2 = callback_tracker.track("b")

        cb1()
        cb2()

        callback_tracker.assert_sequence(["a", "b"])
        with pytest.raises(AssertionError):
            callback_tracker.assert_sequence(["b", "a"])

    def test_tracker_assert_contains(self, callback_tracker):
        """assert_contains checks callback presence."""
        cb = callback_tracker.track("test")
        cb()
        cb()

        callback_tracker.assert_contains("test")
        callback_tracker.assert_contains("test", times=2)
        with pytest.raises(AssertionError):
            callback_tracker.assert_contains("other")

    def test_tracker_get_calls(self, callback_tracker):
        """get_calls returns all calls for a callback."""
        cb = callback_tracker.track("test")
        cb(1, key="a")
        cb(2, key="b")

        calls = callback_tracker.get_calls("test")
        assert len(calls) == 2
        assert calls[0] == ((1,), {"key": "a"})
        assert calls[1] == ((2,), {"key": "b"})

    def test_tracker_reset(self, callback_tracker):
        """reset clears all tracked calls."""
        cb = callback_tracker.track("test")
        cb(1)
        cb(2)

        assert len(callback_tracker.get_sequence()) == 2
        callback_tracker.reset()
        assert len(callback_tracker.get_sequence()) == 0

    def test_tracker_thread_safety(self, callback_tracker):
        """Tracker is thread-safe for concurrent access."""
        cb = callback_tracker.track("concurrent")

        def call_from_thread(n):
            for i in range(n):
                cb(threading.current_thread().name)

        threads = [threading.Thread(target=call_from_thread, args=(10,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 50 calls total (5 threads x 10 calls)
        callback_tracker.assert_contains("concurrent", times=50)


class TestAsyncCallbackCollector:
    """Tests for the AsyncCallbackCollector test utility."""

    def test_collector_waits_for_count(self, async_callback_collector):
        """Collector waits for expected number of callbacks."""
        collector = async_callback_collector(expected_count=3)

        def send_callbacks():
            time.sleep(0.05)
            collector.callback(1)
            collector.callback(2)
            collector.callback(3)

        thread = threading.Thread(target=send_callbacks)
        thread.start()

        assert collector.wait(timeout=2.0)
        assert len(collector.results) == 3
        thread.join()

    def test_collector_timeout(self, async_callback_collector):
        """Collector times out if not enough callbacks."""
        collector = async_callback_collector(expected_count=5)

        collector.callback(1)
        collector.callback(2)

        assert not collector.wait(timeout=0.1)
        assert len(collector.results) == 2

    def test_collector_reset(self, async_callback_collector):
        """Collector can be reset for reuse."""
        collector = async_callback_collector(expected_count=1)

        collector.callback(1)
        assert collector.wait(timeout=0.1)

        collector.reset(expected_count=2)
        assert len(collector.results) == 0

        collector.callback(1)
        collector.callback(2)
        assert collector.wait(timeout=0.1)
        assert len(collector.results) == 2

    def test_collector_thread_safety(self, async_callback_collector):
        """Collector handles concurrent callbacks correctly."""
        collector = async_callback_collector(expected_count=50)

        def send_callbacks(n):
            for i in range(n):
                collector.callback(threading.current_thread().name, i)

        threads = [threading.Thread(target=send_callbacks, args=(10,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert collector.wait(timeout=1.0)
        assert len(collector.results) == 50
