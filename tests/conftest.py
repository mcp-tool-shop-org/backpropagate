"""Pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# PYTEST-XDIST PARALLEL MODE (TESTS-F-003, v1.3 Wave 6a)
# =============================================================================
#
# pytest-xdist is in dev deps but `pytest tests/` ran serial (~10 min for
# the full 1987-test suite). CI gates that block PR merge wear that
# latency on every push.
#
# TESTS-F-003 introduces an env-var-gated convention so:
#
#   * Local devs default to serial mode — easier ``--pdb`` / breakpoint
#     debugging, less stdout interleaving, fewer surprises for
#     contributors who haven't read this file.
#   * CI sets ``BACKPROPAGATE_PYTEST_PARALLEL=1`` to opt into parallel
#     execution.
#
# The actual ``-n auto --dist worksteal`` injection happens in
# ``.github/workflows/ci.yml`` (test step reads the env var and
# conditionally appends the flags). We do NOT attempt to mutate
# ``sys.argv`` from this conftest because rootdir / subdir conftests
# are loaded AFTER pytest has parsed argv (xdist's
# ``pytest_configure`` runs against the already-parsed
# ``numprocesses`` option). The only way to inject via a hook is via
# a pytest plugin registered through entry-points, which would
# require turning ``tests/`` into a published package — overkill for
# a CI optimization. The env var is the contract; ci.yml is the
# enforcement.
#
# Worker semantics (interpreted by ci.yml):
#
#   * ``BACKPROPAGATE_PYTEST_PARALLEL=1`` / ``=auto`` → ``-n auto``
#     (one worker per CPU; matches the most common CI runner config).
#   * ``BACKPROPAGATE_PYTEST_PARALLEL=N`` (integer) → ``-n N``.
#   * unset / ``0`` / ``off`` / ``false`` / ``no`` → no injection,
#     serial mode (default).
#
# Xdist-safety audit (recorded here so future contributors don't have
# to re-derive it):
#
#   * No test in the suite owns global process-level resources (no
#     fixed ports, no file-system singletons outside ``tmp_path``, no
#     module-level state that survives across worker processes).
#   * Tests that mutate ``os.environ`` go through ``monkeypatch.setenv``
#     which is per-test; the few that touch ``os.environ`` directly
#     (test_ui_security, test_windows_compat, test_config*) clean up
#     in finally blocks.
#   * ``threading`` use is intra-process (xdist parallelizes across
#     PROCESSES, not threads).
#   * The Hypothesis profile loader above runs once per worker process
#     at import time — idempotent, safe.
#
# v1.4 Wave 2 (TESTS-A-007): the @pytest.mark.serial convention is now
# applied to tests that mutate process-global state:
#
#   * ``structlog.configure(force=True)`` — overwrites the process-wide
#     structlog configuration; concurrent tests on the same xdist worker
#     can see records routed through the wrong processor chain.
#   * Singleton ``._instance`` reset — e.g. ``SecurityLogger._instance``,
#     ``SessionManager._instance``. The class-level autouse fixtures
#     already protect within a class, but cross-class adjacency on the
#     same worker is still a race surface in parallel mode.
#   * ``configure_logging(...force=True)`` from
#     ``backpropagate.logging_config`` — same shape as the structlog
#     case (forces a reconfiguration of the print-logger factory).
#
# Apply via ``@pytest.mark.serial`` on the class or test function. The
# marker is registered in ``pyproject.toml`` ``[tool.pytest.ini_options]``
# ``markers`` (declared so ``--strict-markers`` doesn't reject it).
#
# Enforcement: this conftest registers a ``pytest_collection_modifyitems``
# hook (below) that uses xdist's ``xdist_group`` to bin all
# ``serial``-marked tests into a single group; xdist then schedules
# every test in the group on the SAME worker (serialising them
# w.r.t. each other) while still parallelising the rest of the suite.
#
# The registration is best-effort — if xdist is not installed (serial
# mode), the marker still works as a documentation hint but provides
# no scheduling guarantee (because there is no scheduler).
_PARALLEL_ENV = "BACKPROPAGATE_PYTEST_PARALLEL"


def _resolve_parallel_workers() -> str | None:
    """Return the xdist worker count string for the env var, or None.

    Used internally by tests/regression tests that introspect the
    parallel-mode contract. ``None`` means serial mode (the default).
    """
    raw = os.environ.get(_PARALLEL_ENV, "").strip().lower()
    if not raw or raw in {"0", "off", "false", "no"}:
        return None
    try:
        import xdist  # noqa: F401 — only checking importability
    except ImportError:
        return None
    if raw in {"1", "auto"}:
        return "auto"
    try:
        return str(int(raw))
    except ValueError:
        return "auto"

# =============================================================================
# HYPOTHESIS PROFILE (TESTS-B-010)
# =============================================================================
#
# Property-based tests in this suite intermittently bumped against the default
# 200 ms Hypothesis deadline on slow Windows runners and inside CI containers
# where the first import of torch / structlog can push a single example past
# the limit even though the assertion itself is correctness-only (not perf).
# Wave 1 registered a "no_deadline" profile at the top of
# ``test_hypothesis_slao.py`` but only loaded it there — Hypothesis-using
# tests in ``test_fuzz_checkpoints.py`` kept hitting the same flake.
#
# Loading the profile from conftest applies it to the whole test session so
# every Hypothesis test inherits the deadline-free + too_slow-suppressed
# settings. Loading happens at import time (before any test or fixture runs)
# because Hypothesis caches the active profile per-process. The
# ``test_hypothesis_slao.py`` module still re-registers + loads the profile
# defensively to support running that file directly without conftest.
try:
    from hypothesis import HealthCheck
    from hypothesis import settings as _hyp_settings

    try:
        _hyp_settings.register_profile(
            "no_deadline",
            deadline=None,
            suppress_health_check=[HealthCheck.too_slow],
        )
    except Exception:  # pragma: no cover — profile already registered
        pass
    _hyp_settings.load_profile("no_deadline")
except ImportError:  # pragma: no cover — hypothesis is a test-only dep
    pass


# =============================================================================
# CUDA/GPU FIXTURES
# =============================================================================

@pytest.fixture
def mock_torch_cuda():
    """Mock torch.cuda for testing without GPU."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_cuda_available():
    """Mock torch.cuda as available with basic GPU properties."""
    mock_props = MagicMock()
    mock_props.total_memory = 16 * (1024**3)  # 16 GB

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="Test GPU"), \
         patch("torch.cuda.get_device_properties", return_value=mock_props), \
         patch("torch.cuda.memory_allocated", return_value=4 * (1024**3)), \
         patch("torch.cuda.memory_reserved", return_value=8 * (1024**3)):
        yield


# =============================================================================
# SETTINGS FIXTURES
# =============================================================================

@pytest.fixture
def mock_settings():
    """Provide test settings."""
    from backpropagate.config import Settings
    return Settings()


# =============================================================================
# DATASET FIXTURES
# =============================================================================

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return [
        {"text": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>"},
        {"text": "<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\nI'm good!<|im_end|>"},
    ]


@pytest.fixture
def large_sample_dataset():
    """Create a larger sample dataset for multi-run testing."""
    return [
        {"text": f"<|im_start|>user\nQuestion {i}<|im_end|>\n<|im_start|>assistant\nAnswer {i}<|im_end|>"}
        for i in range(100)
    ]


# =============================================================================
# TRAINER FIXTURES
# =============================================================================

@pytest.fixture
def mock_trainer():
    """Create a mock trainer for testing."""
    trainer = MagicMock()
    trainer.model_name = "test-model"
    trainer.lora_r = 16
    trainer.batch_size = 2
    trainer._is_loaded = False
    trainer._training_runs = []
    return trainer


@pytest.fixture
def mock_trainer_factory():
    """Factory fixture to create mock trainers with custom settings."""
    def _create_trainer(**kwargs):
        trainer = MagicMock()
        trainer.model_name = kwargs.get("model_name", "test-model")
        trainer.lora_r = kwargs.get("lora_r", 16)
        trainer.batch_size = kwargs.get("batch_size", 2)
        trainer._is_loaded = kwargs.get("is_loaded", False)
        trainer._training_runs = kwargs.get("training_runs", [])
        trainer.get_lora_state_dict = MagicMock(return_value={
            "layer.lora_A.weight": MagicMock(),
            "layer.lora_B.weight": MagicMock(),
        })
        return trainer
    return _create_trainer


# =============================================================================
# LORA STATE FIXTURES
# =============================================================================

@pytest.fixture
def sample_lora_state():
    """Create a sample LoRA state dict for testing."""
    torch = pytest.importorskip("torch")
    return {
        "layer1.lora_A.weight": torch.randn(16, 128),
        "layer1.lora_B.weight": torch.randn(256, 16),
        "layer2.lora_A.weight": torch.randn(16, 128),
        "layer2.lora_B.weight": torch.randn(256, 16),
    }


@pytest.fixture
def sample_lora_pair():
    """Create a pair of LoRA state dicts for merge testing."""
    torch = pytest.importorskip("torch")
    base = {
        "layer.lora_A.weight": torch.randn(16, 128),
        "layer.lora_B.weight": torch.randn(256, 16),
    }
    new = {
        "layer.lora_A.weight": torch.randn(16, 128),
        "layer.lora_B.weight": torch.randn(256, 16),
    }
    return base, new


# =============================================================================
# GPU SAFETY FIXTURES
# =============================================================================

@pytest.fixture
def gpu_safety_config():
    """Create a default GPU safety config."""
    from backpropagate.gpu_safety import GPUSafetyConfig
    return GPUSafetyConfig()


@pytest.fixture
def gpu_status_safe():
    """Create a safe GPU status for testing."""
    from backpropagate.gpu_safety import GPUCondition, GPUStatus
    return GPUStatus(
        available=True,
        device_name="Test GPU",
        temperature_c=60.0,
        vram_total_gb=16.0,
        vram_used_gb=8.0,
        vram_percent=50.0,
        condition=GPUCondition.SAFE,
    )


@pytest.fixture
def gpu_status_critical():
    """Create a critical GPU status for testing."""
    from backpropagate.gpu_safety import GPUCondition, GPUStatus
    return GPUStatus(
        available=True,
        device_name="Test GPU",
        temperature_c=92.0,
        vram_total_gb=16.0,
        vram_used_gb=15.2,
        vram_percent=95.0,
        condition=GPUCondition.CRITICAL,
        condition_reason="Temperature CRITICAL: 92.0°C",
    )


@pytest.fixture
def gpu_status_emergency():
    """Create an emergency GPU status for testing."""
    from backpropagate.gpu_safety import GPUCondition, GPUStatus
    return GPUStatus(
        available=True,
        device_name="Test GPU",
        temperature_c=96.0,
        vram_total_gb=16.0,
        vram_used_gb=15.8,
        vram_percent=98.75,
        condition=GPUCondition.EMERGENCY,
        condition_reason="Temperature EMERGENCY: 96.0°C",
    )


# =============================================================================
# SLAO FIXTURES
# =============================================================================

@pytest.fixture
def slao_config():
    """Create a default SLAO config."""
    from backpropagate.slao import SLAOConfig
    return SLAOConfig()


@pytest.fixture
def slao_merger():
    """Create an initialized SLAO merger."""
    torch = pytest.importorskip("torch")
    from backpropagate.slao import SLAOMerger

    merger = SLAOMerger()
    merger.initialize({
        "layer.lora_A.weight": torch.randn(16, 128),
        "layer.lora_B.weight": torch.randn(256, 16),
    })
    return merger


# =============================================================================
# MULTI-RUN FIXTURES
# =============================================================================

@pytest.fixture
def multi_run_config():
    """Create a default multi-run config."""
    from backpropagate.multi_run import MultiRunConfig
    return MultiRunConfig()


@pytest.fixture
def multi_run_config_fast():
    """Create a fast multi-run config for quick tests."""
    from backpropagate.multi_run import MergeMode, MultiRunConfig
    return MultiRunConfig(
        num_runs=2,
        steps_per_run=10,
        samples_per_run=50,
        merge_mode=MergeMode.SLAO,
        save_every_run=False,
    )


# Backwards compatibility aliases
@pytest.fixture
def speedrun_config(multi_run_config):
    """Backwards compatibility alias for multi_run_config."""
    return multi_run_config


@pytest.fixture
def speedrun_config_fast(multi_run_config_fast):
    """Backwards compatibility alias for multi_run_config_fast."""
    return multi_run_config_fast


# =============================================================================
# EXPORT FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_gguf_path(temp_dir):
    """Create a mock GGUF file for testing."""
    gguf_path = temp_dir / "model.gguf"
    gguf_path.write_bytes(b"GGUF mock content")
    return gguf_path


@pytest.fixture
def mock_peft_model():
    """Create a mock PeftModel for export testing."""
    model = MagicMock()
    model.save_pretrained = MagicMock()
    model.merge_and_unload = MagicMock(return_value=MagicMock())
    model.save_pretrained_gguf = MagicMock()
    model.save_pretrained_merged = MagicMock()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for export testing."""
    tokenizer = MagicMock()
    tokenizer.save_pretrained = MagicMock()
    tokenizer.push_to_hub = MagicMock()
    return tokenizer


# =============================================================================
# DATASET FORMAT FIXTURES
# =============================================================================

@pytest.fixture
def sample_sharegpt_data():
    """Sample ShareGPT format data."""
    return [
        {"conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there!"},
        ]},
        {"conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "4"},
        ]},
    ]


@pytest.fixture
def sample_alpaca_data():
    """Sample Alpaca format data."""
    return [
        {"instruction": "Say hello", "input": "", "output": "Hello!"},
        {"instruction": "Add numbers", "input": "2+2", "output": "4"},
    ]


@pytest.fixture
def sample_openai_data():
    """Sample OpenAI format data."""
    return [
        {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]},
    ]


@pytest.fixture
def sample_jsonl_file(temp_dir, sample_sharegpt_data):
    """Create a temporary JSONL file."""
    import json
    path = temp_dir / "data.jsonl"
    with open(path, "w") as f:
        for item in sample_sharegpt_data:
            f.write(json.dumps(item) + "\n")
    return path


# =============================================================================
# CLI FIXTURES
# =============================================================================

@pytest.fixture
def cli_parser():
    """Create CLI parser for testing."""
    from backpropagate.cli import create_parser
    return create_parser()


# =============================================================================
# INTEGRATION TEST FIXTURES
# =============================================================================

@pytest.fixture
def tiny_model():
    """A very small model for fast tests (mocked)."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.hidden_size = 256
    model.config.num_hidden_layers = 2
    model.parameters = MagicMock(return_value=[MagicMock()])
    return model


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Temporary directory for checkpoints."""
    checkpoint_path = tmp_path / "checkpoints"
    checkpoint_path.mkdir()
    return checkpoint_path


@pytest.fixture
def mock_gpu_status():
    """Mock GPU status for non-GPU testing."""
    from backpropagate.gpu_safety import GPUCondition, GPUStatus
    return GPUStatus(
        available=True,
        device_name="Mock GPU",
        temperature_c=60.0,
        vram_total_gb=16.0,
        vram_used_gb=4.0,
        vram_free_gb=12.0,
        vram_percent=25.0,
        power_draw_w=150.0,
        power_limit_w=300.0,
        power_percent=50.0,
        gpu_utilization=50,
        memory_utilization=25,
        condition=GPUCondition.SAFE,
        condition_reason="All metrics normal",
    )


@pytest.fixture
def training_dataset_file(tmp_path):
    """Create a temporary training dataset file."""
    import json

    dataset_path = tmp_path / "training_data.jsonl"
    samples = [
        {"text": f"<|im_start|>user\nQuestion {i}<|im_end|>\n<|im_start|>assistant\nAnswer {i}<|im_end|>"}
        for i in range(100)
    ]

    with open(dataset_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return dataset_path


@pytest.fixture
def mock_training_result():
    """Create a mock training result."""
    result = MagicMock()
    result.final_loss = 0.5
    result.duration_seconds = 60.0
    result.steps_completed = 100
    result.samples_seen = 800
    return result


# =============================================================================
# CALLBACK & EVENT HANDLER FIXTURES
# =============================================================================

@pytest.fixture
def mock_training_callback():
    """
    Create a mock TrainingCallback that tracks all invocations.

    Returns:
        tuple: (TrainingCallback, dict of call lists)

    Usage:
        callback, calls = mock_training_callback
        trainer.train(callback=callback)
        assert len(calls["step"]) == expected_steps
    """
    from backpropagate.trainer import TrainingCallback

    calls = {
        "step": [],
        "epoch": [],
        "save": [],
        "complete": [],
        "error": [],
    }

    def on_step(step: int, loss: float) -> None:
        calls["step"].append((step, loss))

    def on_epoch(epoch: int) -> None:
        calls["epoch"].append(epoch)

    def on_save(path: str) -> None:
        calls["save"].append(path)

    def on_complete(run) -> None:
        calls["complete"].append(run)

    def on_error(exc: Exception) -> None:
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
    """
    Create mock callbacks for MultiRunTrainer.

    Returns:
        tuple: (dict of callbacks, dict of call lists)

    Usage:
        callbacks, calls = mock_multirun_callbacks
        trainer = MultiRunTrainer(model="...", **callbacks)
        trainer.run()
        assert len(calls["run_complete"]) == num_runs
    """
    import threading

    calls = {
        "run_start": [],
        "run_complete": [],
        "step": [],
        "gpu_status": [],
    }
    _lock = threading.Lock()

    def on_run_start(run_idx: int) -> None:
        with _lock:
            calls["run_start"].append(run_idx)

    def on_run_complete(result) -> None:
        with _lock:
            calls["run_complete"].append(result)

    def on_step(run_idx: int, step: int, loss: float) -> None:
        with _lock:
            calls["step"].append((run_idx, step, loss))

    def on_gpu_status(status) -> None:
        with _lock:
            calls["gpu_status"].append(status)

    callbacks = {
        "on_run_start": on_run_start,
        "on_run_complete": on_run_complete,
        "on_step": on_step,
        "on_gpu_status": on_gpu_status,
    }
    return callbacks, calls


@pytest.fixture
def mock_gpu_monitor_callbacks():
    """
    Create mock callbacks for GPUMonitor.

    Returns:
        tuple: (dict of callbacks, dict of call lists, threading.Event)

    The Event is set when on_status is called, useful for waiting
    in tests.

    Usage:
        callbacks, calls, event = mock_gpu_monitor_callbacks
        monitor = GPUMonitor(**callbacks)
        monitor.start()
        event.wait(timeout=5.0)
        assert len(calls["status"]) > 0
    """
    import threading

    calls = {
        "status": [],
        "warning": [],
        "critical": [],
        "emergency": [],
    }
    _lock = threading.Lock()
    event = threading.Event()

    def on_status(status) -> None:
        with _lock:
            calls["status"].append(status)
            event.set()

    def on_warning(status) -> None:
        with _lock:
            calls["warning"].append(status)

    def on_critical(status) -> None:
        with _lock:
            calls["critical"].append(status)

    def on_emergency(status) -> None:
        with _lock:
            calls["emergency"].append(status)

    callbacks = {
        "on_status": on_status,
        "on_warning": on_warning,
        "on_critical": on_critical,
        "on_emergency": on_emergency,
    }
    return callbacks, calls, event


@pytest.fixture
def callback_spy():
    """
    Create a CallbackSpy instance for detailed invocation tracking.

    Returns:
        CallbackSpy: A spy that can be used as any callback

    Usage:
        spy = callback_spy
        trainer.train(callback=TrainingCallback(on_step=spy))
        spy.assert_called(times=10)
    """
    from tests.helpers import CallbackSpy
    return CallbackSpy()


@pytest.fixture
def callback_spy_factory():
    """
    Factory to create multiple CallbackSpy instances.

    Returns:
        Callable: Factory function that creates new spies

    Usage:
        step_spy = callback_spy_factory()
        error_spy = callback_spy_factory()
        callback = TrainingCallback(on_step=step_spy, on_error=error_spy)
    """
    from tests.helpers import CallbackSpy

    def _create_spy(return_value=None, side_effect=None):
        return CallbackSpy(return_value=return_value, side_effect=side_effect)

    return _create_spy


@pytest.fixture
def callback_tracker():
    """
    Create a CallbackTracker for sequence verification.

    Returns:
        CallbackTracker: Tracks multiple callbacks and their order

    Usage:
        tracker = callback_tracker
        callback = TrainingCallback(
            on_step=tracker.track("step"),
            on_complete=tracker.track("complete"),
        )
        trainer.train(callback=callback)
        tracker.assert_sequence(["step", "step", "complete"])
    """
    from tests.helpers import CallbackTracker
    return CallbackTracker()


@pytest.fixture
def async_callback_collector():
    """
    Factory to create AsyncCallbackCollector for threaded callback testing.

    Returns:
        Callable: Factory that creates collectors with expected count

    Usage:
        collector = async_callback_collector(expected_count=5)
        monitor = GPUMonitor(on_status=collector.callback)
        monitor.start()
        collector.wait(timeout=10.0)
    """
    from tests.helpers import AsyncCallbackCollector

    def _create_collector(expected_count: int = 1):
        return AsyncCallbackCollector(expected_count=expected_count)

    return _create_collector


# =============================================================================
# SERIAL-MARKER ENFORCEMENT (TESTS-A-007, v1.4 Wave 2)
# =============================================================================
#
# Tests that mutate process-global state (structlog config, singleton
# instances, env vars without monkeypatch) cannot safely run concurrently
# with other tests on the same xdist worker. The @pytest.mark.serial
# marker is the operator-facing convention; this hook gives it real
# scheduling power by pinning every serial-marked test to the same
# xdist_group, which xdist then runs sequentially on a single worker.
#
# Side-effects:
#   * In serial mode (xdist absent / no -n): marker is documentation only.
#   * In parallel mode (-n auto): all serial-marked tests run on a
#     single worker; the rest of the suite continues to parallelise.
#   * Tests already marked with their own xdist_group keep their group
#     (we only add xdist_group if the test has serial but NO existing
#     xdist_group marker).
#
# A best-effort warning fires when more than ~50 tests carry the serial
# marker — that's a smell suggesting parallel mode would benefit from
# breaking the serial bucket into multiple xdist_groups (e.g.
# "shared-structlog" + "shared-singleton-X"). The warning is emitted to
# stderr at collection time, not via the pytest API, to keep the hook
# side-effect-free for unrelated runs.


def pytest_collection_modifyitems(config, items):
    """Pin every ``@pytest.mark.serial`` test to a single xdist_group.

    Without this hook, the ``serial`` marker is documentation only:
    xdist's scheduler doesn't know that two serial-marked tests should
    not run concurrently across workers. By assigning them all to the
    same xdist_group, xdist serialises them on a single worker.
    """
    serial_count = 0
    for item in items:
        if "serial" not in item.keywords:
            continue
        serial_count += 1
        # Skip if the test already has an xdist_group marker (it explicitly
        # opted into a more specific group; respect the author's choice).
        if any(
            m.name == "xdist_group" for m in item.iter_markers()
        ):
            continue
        # Apply the catch-all serial group.
        item.add_marker(pytest.mark.xdist_group(name="serial"))

    if serial_count > 50:  # pragma: no cover — collection-time heuristic
        import sys
        print(
            f"[conftest] WARNING: {serial_count} tests are marked @serial. "
            f"Consider splitting into smaller xdist_groups (e.g. by which "
            f"global resource they share) so the serial bucket doesn't "
            f"become a long pole in parallel mode.",
            file=sys.stderr,
        )
