# Event Handler & WebSocket Test Roadmap

## Executive Summary

This roadmap outlines the comprehensive testing strategy for event handlers and callbacks in the backpropagate codebase. The project uses a callback-based architecture for the training core (`trainer.py` / `multi_run.py` / `gpu_safety.py`) AND ships an ASGI WebSocket transport in the Reflex Web UI from v1.1.0 onward — see Section 9 for the WS auth middleware that gates that transport.

**Current Coverage Analysis:**

The Jan-2026 baseline percentages cited here (TrainingCallback ~40%,
MultiRunTrainer callbacks ~30%, GPUMonitor callbacks ~50%, event lifecycle
~20%) are stale — `test_event_handlers.py` has grown substantially post
v1.3 Wave 1 (TESTS-A-008 threading.Event fix, TESTS-A-009 JWT expiry
patch). For the live numbers, see the per-test-file coverage report
emitted by `pytest --cov=backpropagate` in CI (artifacts uploaded by
`.github/workflows/ci.yml`).

**Target:** 85% coverage for all event systems.

---

## 1. TrainingCallback System

### Location: `backpropagate/trainer.py:78-85`

```python
@dataclass
class TrainingCallback:
    on_step: Optional[Callable[[int, float], None]] = None
    on_epoch: Optional[Callable[[int], None]] = None
    on_save: Optional[Callable[[str], None]] = None
    on_complete: Optional[Callable[[TrainingRun], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
```

### Test Categories

#### 1.1 Unit Tests (Priority: HIGH)

| Test | Description | Status |
|------|-------------|--------|
| `test_callback_dataclass_defaults` | All fields default to None | Exists |
| `test_callback_with_all_handlers` | Create with all handlers set | Missing |
| `test_callback_partial_handlers` | Create with subset of handlers | Partial |
| `test_callback_handler_signatures` | Verify handler type hints | Missing |
| `test_callback_immutability` | Dataclass field behavior | Missing |

#### 1.2 Integration Tests (Priority: HIGH)

| Test | Description | Status |
|------|-------------|--------|
| `test_on_step_called_each_step` | on_step receives (step, loss) | Missing |
| `test_on_step_loss_values_valid` | Loss values are valid floats | Missing |
| `test_on_epoch_called_at_epoch_end` | on_epoch receives epoch number | Missing |
| `test_on_save_called_with_path` | on_save receives checkpoint path | Missing |
| `test_on_complete_receives_run` | on_complete gets TrainingRun object | Exists |
| `test_on_error_receives_exception` | on_error gets exception on failure | Exists |
| `test_callback_error_isolation` | Callback exception doesn't crash training | Missing |
| `test_callback_none_safe` | None callbacks are safely skipped | Missing |

#### 1.3 Edge Cases (Priority: MEDIUM)

| Test | Description | Status |
|------|-------------|--------|
| `test_callback_raises_exception` | Handler that throws | Missing |
| `test_callback_slow_handler` | Handler that blocks | Missing |
| `test_callback_modifies_state` | Handler that modifies trainer state | Missing |
| `test_callback_reentrant` | Handler calls trainer method | Missing |
| `test_multiple_callbacks_sequential` | Multiple training runs with callbacks | Missing |

---

## 2. MultiRunTrainer Callbacks

### Location: `backpropagate/multi_run.py:196-199, 237-240`

```python
on_run_start: Optional[Callable[[int], None]] = None
on_run_complete: Optional[Callable[[RunResult], None]] = None
on_step: Optional[Callable[[int, int, float], None]] = None
on_gpu_status: Optional[Callable[[GPUStatus], None]] = None
```

### Test Categories

#### 2.1 Unit Tests (Priority: HIGH)

| Test | Description | Status |
|------|-------------|--------|
| `test_multirun_accepts_callbacks` | Constructor stores callbacks | Exists |
| `test_multirun_callback_types` | Type validation for callbacks | Missing |
| `test_multirun_no_callbacks` | Works without any callbacks | Exists |
| `test_multirun_partial_callbacks` | Works with subset of callbacks | Missing |

#### 2.2 Event Sequencing Tests (Priority: HIGH)

| Test | Description | Status |
|------|-------------|--------|
| `test_on_run_start_order` | Called before each run starts | Missing |
| `test_on_run_complete_order` | Called after each run completes | Missing |
| `test_callback_run_index_sequence` | run_idx increments correctly | Missing |
| `test_callback_timing_between_runs` | Callbacks fire in correct sequence | Missing |
| `test_on_step_per_run` | on_step includes (run_idx, step, loss) | Missing |

#### 2.3 GPU Status Callbacks (Priority: MEDIUM)

| Test | Description | Status |
|------|-------------|--------|
| `test_on_gpu_status_periodic` | GPU status callbacks fire periodically | Missing |
| `test_on_gpu_status_content` | GPUStatus object has valid fields | Missing |
| `test_on_gpu_status_during_pause` | Callbacks during training pause | Missing |
| `test_gpu_callback_thread_safety` | Callbacks from monitor thread | Missing |

#### 2.4 Abort & Error Callbacks (Priority: HIGH)

| Test | Description | Status |
|------|-------------|--------|
| `test_on_run_complete_on_abort` | Final callback on early abort | Missing |
| `test_abort_preserves_callback_state` | Callbacks still work after abort | Missing |
| `test_callback_during_validation` | Callbacks during validation runs | Missing |
| `test_early_stopping_callbacks` | Callbacks when early stopping triggers | Missing |

---

## 3. GPUMonitor Event System

### Location: `backpropagate/gpu_safety.py:380-406`

```python
on_warning: Optional[Callable[[GPUStatus], None]] = None
on_critical: Optional[Callable[[GPUStatus], None]] = None
on_emergency: Optional[Callable[[GPUStatus], None]] = None
on_status: Optional[Callable[[GPUStatus], None]] = None
```

### Test Categories

#### 3.1 Callback Registration (Priority: HIGH)

| Test | Description | Status |
|------|-------------|--------|
| `test_monitor_accepts_callbacks` | Constructor stores callbacks | Exists |
| `test_monitor_no_callbacks` | Monitor works without callbacks | Exists |
| `test_monitor_callback_replacement` | Can update callbacks after init | Missing |

#### 3.2 Event Dispatch Tests (Priority: HIGH)

| Test | Description | Status |
|------|-------------|--------|
| `test_on_status_fires_periodically` | Status callback fires on interval | Partial |
| `test_on_warning_at_threshold` | Warning fires at warning temp | Missing |
| `test_on_critical_at_threshold` | Critical fires at critical temp | Missing |
| `test_on_emergency_at_threshold` | Emergency fires at emergency temp | Missing |
| `test_callback_receives_gpustatus` | All callbacks get GPUStatus object | Missing |
| `test_callback_status_has_condition` | GPUStatus.condition matches event | Missing |

#### 3.3 Thread Safety Tests (Priority: HIGH)

| Test | Description | Status |
|------|-------------|--------|
| `test_callbacks_from_monitor_thread` | Callbacks run in background thread | Partial |
| `test_callback_exception_isolated` | Exception doesn't stop monitoring | Missing |
| `test_concurrent_callback_access` | Multiple callbacks access shared state | Exists |
| `test_stop_during_callback` | Stop called while callback running | Missing |
| `test_pause_prevents_callbacks` | Paused monitor skips callbacks | Missing |

#### 3.4 Event Escalation Tests (Priority: MEDIUM)

| Test | Description | Status |
|------|-------------|--------|
| `test_warning_to_critical_escalation` | Temp rise triggers escalation | Missing |
| `test_critical_to_emergency_escalation` | Further rise triggers emergency | Missing |
| `test_deescalation_on_cooldown` | Temp drop changes condition | Missing |
| `test_multiple_events_per_check` | Multiple conditions in one check | Missing |

---

## 4. Integration & E2E Tests

### 4.1 Full Training Flow (Priority: HIGH)

| Test | Description | Status |
|------|-------------|--------|
| `test_e2e_training_with_all_callbacks` | Complete flow with all handlers | Missing |
| `test_e2e_multirun_callback_sequence` | Full multi-run with callbacks | Missing |
| `test_e2e_gpu_monitoring_integration` | Training + GPU monitoring callbacks | Missing |
| `test_e2e_abort_callback_cleanup` | Callbacks during abort cleanup | Missing |

### 4.2 CLI Integration (Priority: MEDIUM)

| Test | Description | Status |
|------|-------------|--------|
| `test_cli_train_progress_callback` | CLI progress bar uses on_step | Missing |
| `test_cli_multirun_status_callback` | CLI status from on_run_complete | Missing |
| `test_cli_verbose_mode_callbacks` | Verbose output uses callbacks | Missing |

<!--
### 4.3 UI Integration — DELETED (TESTS-B-014, v1.3 Wave 3 Stage C)

This section listed three Gradio-era UI integration tests. Gradio was
deleted in v1.2.0 Wave 4.5 (`ui_gradio_legacy.py` / `theme_gradio_legacy.py`
were removed; the canonical UI is now Reflex). The Reflex equivalents are
tracked in `tests/COVERAGE_ROADMAP.md` section 5.1 and the v1.3 Wave 3
Stage C tests at `tests/test_ui_states.py` cover the four Reflex state
classes (TrainState, MultiRunState, ExportState, DatasetState) with smoke
tests for defaults + setters + stub event handlers.
-->

---

## 5. Mock & Test Utilities Needed

### 5.1 New Test Fixtures

```python
# conftest.py additions

@pytest.fixture
def mock_training_callback():
    """Create a mock TrainingCallback that tracks calls."""
    calls = {"step": [], "epoch": [], "save": [], "complete": [], "error": []}

    def on_step(step, loss):
        calls["step"].append((step, loss))

    def on_epoch(epoch):
        calls["epoch"].append(epoch)

    def on_save(path):
        calls["save"].append(path)

    def on_complete(run):
        calls["complete"].append(run)

    def on_error(exc):
        calls["error"].append(exc)

    callback = TrainingCallback(
        on_step=on_step,
        on_epoch=on_epoch,
        on_save=on_save,
        on_complete=on_complete,
        on_error=on_error,
    )
    return callback, calls


@pytest.fixture
def mock_multirun_callbacks():
    """Create mock callbacks for MultiRunTrainer."""
    calls = {"run_start": [], "run_complete": [], "step": [], "gpu_status": []}

    def on_run_start(run_idx):
        calls["run_start"].append(run_idx)

    def on_run_complete(result):
        calls["run_complete"].append(result)

    def on_step(run_idx, step, loss):
        calls["step"].append((run_idx, step, loss))

    def on_gpu_status(status):
        calls["gpu_status"].append(status)

    return {
        "on_run_start": on_run_start,
        "on_run_complete": on_run_complete,
        "on_step": on_step,
        "on_gpu_status": on_gpu_status,
    }, calls


@pytest.fixture
def mock_gpu_monitor_callbacks():
    """Create mock callbacks for GPUMonitor."""
    calls = {"status": [], "warning": [], "critical": [], "emergency": []}
    events = threading.Event()

    def on_status(status):
        calls["status"].append(status)
        events.set()

    def on_warning(status):
        calls["warning"].append(status)

    def on_critical(status):
        calls["critical"].append(status)

    def on_emergency(status):
        calls["emergency"].append(status)

    return {
        "on_status": on_status,
        "on_warning": on_warning,
        "on_critical": on_critical,
        "on_emergency": on_emergency,
    }, calls, events
```

### 5.2 Helper Functions

```python
# test_helpers.py

def wait_for_callback(calls, key, count=1, timeout=5.0):
    """Wait for a callback to be called a certain number of times."""
    import time
    start = time.time()
    while len(calls[key]) < count:
        if time.time() - start > timeout:
            raise TimeoutError(f"Callback '{key}' not called {count} times within {timeout}s")
        time.sleep(0.01)
    return calls[key]


def assert_callback_sequence(calls, expected_sequence):
    """Assert callbacks were called in expected order."""
    actual = []
    for key, items in calls.items():
        for i, item in enumerate(items):
            actual.append((key, i))

    # Verify sequence matches
    for expected_key, expected_idx in expected_sequence:
        assert (expected_key, expected_idx) in actual


class CallbackSpy:
    """Spy object that records all callback invocations."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self.lock:
            self.calls.append({
                "args": args,
                "kwargs": kwargs,
                "timestamp": time.time(),
                "thread": threading.current_thread().name,
            })

    def assert_called(self, times=None):
        if times is not None:
            assert len(self.calls) == times
        else:
            assert len(self.calls) > 0

    def assert_called_with(self, *args, **kwargs):
        for call in self.calls:
            if call["args"] == args and call["kwargs"] == kwargs:
                return
        raise AssertionError(f"No call with args={args}, kwargs={kwargs}")
```

---

## 6. Implementation Plan

### Phase 1: Foundation (Week 1)
- [ ] Create test fixtures in `conftest.py`
- [ ] Add helper functions to `test_helpers.py`
- [ ] Create `tests/test_event_handlers.py` file structure

### Phase 2: TrainingCallback Tests (Week 1-2)
- [ ] Unit tests for TrainingCallback dataclass
- [ ] Integration tests for callback invocation
- [ ] Edge case tests for error handling

### Phase 3: MultiRunTrainer Callback Tests (Week 2)
- [ ] Event sequencing tests
- [ ] GPU status callback tests
- [ ] Abort and error handling tests

### Phase 4: GPUMonitor Event Tests (Week 2-3)
- [ ] Thread safety tests
- [ ] Event escalation tests
- [ ] Condition threshold tests

### Phase 5: Integration Tests (Week 3)
- [ ] E2E training flow tests
- [ ] CLI integration tests
- [ ] Cross-component callback flow tests

### Phase 6: Documentation & Cleanup (Week 3)
- [ ] Document callback contracts
- [ ] Update test coverage report
- [ ] Create example usage in docstrings

---

## 7. Test File Structure

```
tests/
├── test_event_handlers.py          # NEW: Centralized event handler tests
│   ├── TestTrainingCallbackUnit
│   ├── TestTrainingCallbackIntegration
│   ├── TestTrainingCallbackEdgeCases
│   ├── TestMultiRunCallbackUnit
│   ├── TestMultiRunCallbackSequencing
│   ├── TestMultiRunGPUCallbacks
│   ├── TestMultiRunAbortCallbacks
│   ├── TestGPUMonitorCallbackRegistration
│   ├── TestGPUMonitorEventDispatch
│   ├── TestGPUMonitorThreadSafety
│   └── TestGPUMonitorEscalation
│
├── test_callback_integration.py    # NEW: Cross-component callback tests
│   ├── TestE2ETrainingCallbacks
│   ├── TestE2EMultiRunCallbacks
│   ├── TestE2EGPUMonitoringCallbacks
│   └── TestCLICallbackIntegration
│
├── conftest.py                     # UPDATE: Add callback fixtures
└── test_helpers.py                 # NEW: Callback testing utilities
```

---

## 8. Metrics & Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| TrainingCallback coverage | 40% | 90% |
| MultiRunTrainer callback coverage | 30% | 85% |
| GPUMonitor callback coverage | 50% | 90% |
| Event integration test count | 5 | 25 |
| Thread safety tests | 2 | 10 |

### Definition of Done

- [ ] All callback types have unit tests
- [ ] All callback invocation points have integration tests
- [ ] Thread safety verified for background callbacks
- [ ] Error isolation verified (callback exceptions don't crash training)
- [ ] Event sequencing verified for multi-run scenarios
- [ ] Coverage > 85% for all event handler code

---

## 9. WebSocket Surface — Reflex Web UI ASGI Middleware (v1.2.0+ — SHIPPED)

**Status correction (v1.3 Wave 1, 2026-05-23):** the prior version of this
section said "The codebase currently has NO WebSocket implementation." That
was true through v1.0.x but no longer holds — the v1.2.0 release introduced
a real ASGI WebSocket transport wired through
`backpropagate.ui_app.auth.basic_auth_transformer`, the auth middleware
that gates every Reflex `/_event` WS upgrade.

### What ships today

The Reflex Web UI (`backpropagate/ui_app/`) uses ASGI WebSockets for its
state-sync transport (Reflex's standard `/_event` path). The
`basic_auth_transformer` middleware wraps the Reflex ASGI app and:

1. **HTTP gate**: validates Host header (DNS-rebinding defense),
   Origin header (CSWSH defense), and either session-cookie /
   per-launch-token / Basic-auth credentials. Sets HMAC-signed
   session cookie on success.
2. **WebSocket gate**: validates Host + Origin BEFORE
   `websocket.accept()` (mandatory pre-accept close to prevent the
   DoS-vector anti-pattern), then validates the session cookie via
   `_validate_cookie` (constant-time HMAC compare). Rejected upgrades
   close with code 4401 (auth) or 4403 (origin) — never accepted then
   closed.

### Source of truth

- `backpropagate/ui_app/auth.py` — middleware implementation (~750 LOC).
- `DESIGN_BRIEF.md` "Middleware logic" + "Testing requirements" — the
  16 brief-numbered tests this middleware was designed against.
- `backpropagate/exceptions.py::ERROR_CODES` — `INPUT_AUTH_REQUIRED`,
  `INPUT_AUTH_INVALID_SHAPE`, `RUNTIME_UI_AUTH_NOT_ENFORCED`,
  `UI_OUTPUT_DIR_FORBIDDEN` are the auth-surface codes.

### Test coverage (where it lives now)

| Surface | Test file | Coverage |
|---|---|---|
| Brief tests 1-16 (the canonical DESIGN_BRIEF surface) | `tests/test_auth_middleware.py` | Complete (10 active + 2 cross-referenced + 4 environmental skips). |
| Helper-recorder smoke tests | `tests/test_auth_middleware.py` | Complete (pre/post-accept detection invariants). |
| Hardening additions (malformed headers, /ping bypass, cookie hardening) | `tests/test_auth_middleware.py` | Complete. |
| Pre-accept cookie-expiry close + tampered-signature close (v1.3 Wave 1) | `tests/test_auth_middleware.py` | New — TESTS-A-005 + TESTS-A-006. |
| HMAC `compare_digest` source-level invariant (v1.3 Wave 1) | `tests/test_auth_middleware.py` | New — TESTS-A-006 source inspection. |
| CLI `--share` refuse-to-start | `tests/test_cli_extended.py::TestCmdUI::test_cmd_ui_share_without_auth_still_refuses` | Complete. |
| CLI `--host <non-loopback>` refuse-to-start (v1.3 Wave 1) | `tests/test_host_gate.py` | New — BRIDGE-A-002. |
| ASGI helper infrastructure | `tests/helpers/asgi.py` + `tests/helpers/ws.py` | Complete; smoke-tested by the helper tests at the top of `test_auth_middleware.py`. |

### Open follow-ups (post-v1.3 Wave 1)

1. **Lock-file token mode on Windows** — the brief covers POSIX
   `0o600` mode bits; the Windows-ACL equivalent (DENY EVERYONE except
   current user) is not yet test-asserted.
2. **CSRF token TTL** — `CSRFProtection.validate_token` has a TTL
   field but no expiry-race regression test yet (the JWT analogue
   was fixed in v1.3 Wave 1 — see TESTS-A-009).
3. **Multi-client broadcast** — Reflex's transport doesn't broadcast
   today; if a future feature adds a fan-out path, the brief should
   grow tests for ordering / backpressure.

### Historical (deleted scaffolding)

The previous "potential WebSocket architecture" diagram + "if
implemented" test categories described a hypothetical future. That
future arrived in v1.2.0 — the actual coverage is enumerated above.
The diagram is dropped to avoid drift between the imagined and the
shipped architectures.

---

## Appendix A: Existing Test Coverage Map

### trainer.py Callback Tests (test_trainer.py)

```
Line 319-345: TestTrainingCallback
  - test_callback_creation: PASS
  - test_callback_with_handler: PASS

Line 563-570: test_train_invokes_callback_on_complete
  - Tests on_complete callback

Line 599-606: test_train_invokes_callback_on_error
  - Tests on_error callback
```

### gpu_safety.py Monitor Tests (test_gpu_safety.py)

```
Line 319-430: TestGPUMonitor
  - test_monitor_default_config: PASS
  - test_monitor_start_stop: PASS (threaded)
  - test_monitor_with_callback: PARTIAL
  - test_monitor_pause_resume: PASS
  - test_monitor_get_status: PASS
  - test_monitor_callback_exception: MISSING
```

### multi_run.py Callback Tests (test_multi_run.py)

```
Line 784-850: TestMultiRunTrainerGPUMonitoring
  - test_gpu_monitoring_config: PASS
  - test_gpu_monitoring_disabled: PASS

Line 1312-1348: TestCallbackIntegration
  - test_callback_integration: PASS (basic)
  - test_callback_invocation_order: MISSING
```
