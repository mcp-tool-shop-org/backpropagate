"""
Test helper utilities for callback and event handler testing.

Provides:
- CallbackSpy: Records all callback invocations with metadata
- wait_for_callback: Wait for async/threaded callbacks
- assert_callback_sequence: Verify callback ordering
"""

import time
import threading
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CallbackInvocation:
    """Record of a single callback invocation."""
    args: tuple
    kwargs: dict
    timestamp: float
    thread_name: str
    thread_id: int


class CallbackSpy:
    """
    Spy object that records all callback invocations with metadata.

    Thread-safe and can be used to verify callback behavior in both
    synchronous and asynchronous contexts.

    Usage:
        spy = CallbackSpy()
        trainer = Trainer(callback=TrainingCallback(on_step=spy))
        trainer.train(...)
        spy.assert_called(times=10)
        spy.assert_called_with(step=5, loss=0.5)
    """

    def __init__(self, return_value: Any = None, side_effect: Optional[Exception] = None):
        self.calls: List[CallbackInvocation] = []
        self._lock = threading.Lock()
        self._return_value = return_value
        self._side_effect = side_effect
        self._event = threading.Event()

    def __call__(self, *args, **kwargs) -> Any:
        """Record the invocation and optionally raise/return."""
        with self._lock:
            invocation = CallbackInvocation(
                args=args,
                kwargs=kwargs,
                timestamp=time.time(),
                thread_name=threading.current_thread().name,
                thread_id=threading.current_thread().ident,
            )
            self.calls.append(invocation)
            self._event.set()

        if self._side_effect is not None:
            raise self._side_effect

        return self._return_value

    @property
    def call_count(self) -> int:
        """Number of times the callback was invoked."""
        with self._lock:
            return len(self.calls)

    @property
    def called(self) -> bool:
        """Whether the callback was invoked at least once."""
        return self.call_count > 0

    @property
    def last_call(self) -> Optional[CallbackInvocation]:
        """Get the most recent invocation, or None if never called."""
        with self._lock:
            return self.calls[-1] if self.calls else None

    def assert_called(self, times: Optional[int] = None) -> None:
        """Assert the callback was called (optionally a specific number of times)."""
        if times is not None:
            assert self.call_count == times, (
                f"Expected {times} calls, got {self.call_count}"
            )
        else:
            assert self.called, "Callback was never called"

    def assert_not_called(self) -> None:
        """Assert the callback was never called."""
        assert not self.called, f"Callback was called {self.call_count} times"

    def assert_called_with(self, *args, **kwargs) -> None:
        """Assert the callback was called with specific arguments."""
        with self._lock:
            for call in self.calls:
                if call.args == args:
                    # Check kwargs match (allow subset matching)
                    if all(call.kwargs.get(k) == v for k, v in kwargs.items()):
                        return
            raise AssertionError(
                f"No call with args={args}, kwargs={kwargs}. "
                f"Actual calls: {[(c.args, c.kwargs) for c in self.calls]}"
            )

    def assert_called_from_thread(self, thread_name: str) -> None:
        """Assert at least one call came from a specific thread."""
        with self._lock:
            for call in self.calls:
                if call.thread_name == thread_name:
                    return
            thread_names = [c.thread_name for c in self.calls]
            raise AssertionError(
                f"No calls from thread '{thread_name}'. "
                f"Calls came from: {set(thread_names)}"
            )

    def wait_for_call(self, timeout: float = 5.0) -> bool:
        """Wait for the callback to be called at least once."""
        return self._event.wait(timeout)

    def reset(self) -> None:
        """Clear all recorded invocations."""
        with self._lock:
            self.calls.clear()
            self._event.clear()

    def get_args_list(self) -> List[tuple]:
        """Get list of all args from all invocations."""
        with self._lock:
            return [c.args for c in self.calls]

    def get_timestamps(self) -> List[float]:
        """Get list of all timestamps from all invocations."""
        with self._lock:
            return [c.timestamp for c in self.calls]


def wait_for_callback(
    calls: Dict[str, List],
    key: str,
    count: int = 1,
    timeout: float = 5.0,
) -> List[Any]:
    """
    Wait for a callback to be called a certain number of times.

    Args:
        calls: Dict tracking callback invocations (key -> list of calls)
        key: The callback key to wait for
        count: Minimum number of calls to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        List of callback arguments

    Raises:
        TimeoutError: If callback not called enough times within timeout
    """
    start = time.time()
    while len(calls.get(key, [])) < count:
        if time.time() - start > timeout:
            actual = len(calls.get(key, []))
            raise TimeoutError(
                f"Callback '{key}' called {actual} times, expected {count} "
                f"within {timeout}s"
            )
        time.sleep(0.01)
    return calls[key]


def assert_callback_sequence(
    call_log: List[Tuple[str, Any]],
    expected_sequence: List[str],
) -> None:
    """
    Assert callbacks were called in expected order.

    Args:
        call_log: List of (callback_name, args) tuples in order received
        expected_sequence: List of callback names in expected order

    Raises:
        AssertionError: If sequence doesn't match
    """
    actual_sequence = [name for name, _ in call_log]

    # Check all expected callbacks are present in order
    expected_idx = 0
    for name in actual_sequence:
        if expected_idx < len(expected_sequence) and name == expected_sequence[expected_idx]:
            expected_idx += 1

    if expected_idx != len(expected_sequence):
        raise AssertionError(
            f"Callback sequence mismatch.\n"
            f"Expected: {expected_sequence}\n"
            f"Actual: {actual_sequence}\n"
            f"Missing: {expected_sequence[expected_idx:]}"
        )


@dataclass
class CallbackTracker:
    """
    Tracks multiple callbacks and their invocation order.

    Usage:
        tracker = CallbackTracker()
        callback = TrainingCallback(
            on_step=tracker.track("step"),
            on_complete=tracker.track("complete"),
        )
        trainer.train(callback=callback)
        tracker.assert_sequence(["step", "step", "step", "complete"])
    """
    calls: List[Tuple[str, tuple, dict]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def track(self, name: str):
        """Create a tracking callback for the given name."""
        def callback(*args, **kwargs):
            with self._lock:
                self.calls.append((name, args, kwargs))
        return callback

    def get_sequence(self) -> List[str]:
        """Get the sequence of callback names in order."""
        with self._lock:
            return [name for name, _, _ in self.calls]

    def get_calls(self, name: str) -> List[Tuple[tuple, dict]]:
        """Get all calls for a specific callback name."""
        with self._lock:
            return [(args, kwargs) for n, args, kwargs in self.calls if n == name]

    def assert_sequence(self, expected: List[str]) -> None:
        """Assert callbacks were called in expected sequence."""
        actual = self.get_sequence()
        assert actual == expected, f"Expected {expected}, got {actual}"

    def assert_contains(self, name: str, times: Optional[int] = None) -> None:
        """Assert a callback was called (optionally a specific number of times)."""
        calls = self.get_calls(name)
        if times is not None:
            assert len(calls) == times, f"Expected {name} called {times} times, got {len(calls)}"
        else:
            assert len(calls) > 0, f"Callback {name} was never called"

    def reset(self) -> None:
        """Clear all recorded calls."""
        with self._lock:
            self.calls.clear()


class AsyncCallbackCollector:
    """
    Collects callbacks from potentially multiple threads.

    Thread-safe collector that waits for a specific number of callbacks
    before proceeding. Useful for testing async/threaded callback systems.

    Usage:
        collector = AsyncCallbackCollector(expected_count=5)
        monitor = GPUMonitor(on_status=collector.callback)
        monitor.start()
        collector.wait(timeout=10.0)
        assert len(collector.results) == 5
    """

    def __init__(self, expected_count: int = 1):
        self.results: List[Any] = []
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._expected_count = expected_count

    def callback(self, *args, **kwargs) -> None:
        """Callback function to use with the system under test."""
        with self._lock:
            self.results.append({"args": args, "kwargs": kwargs})
            if len(self.results) >= self._expected_count:
                self._event.set()

    def wait(self, timeout: float = 5.0) -> bool:
        """Wait for expected number of callbacks."""
        return self._event.wait(timeout)

    def reset(self, expected_count: Optional[int] = None) -> None:
        """Reset the collector for reuse."""
        with self._lock:
            self.results.clear()
            self._event.clear()
            if expected_count is not None:
                self._expected_count = expected_count
