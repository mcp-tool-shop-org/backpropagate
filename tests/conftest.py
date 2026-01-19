"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import MagicMock, patch


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
    from backpropagate.gpu_safety import GPUStatus, GPUCondition
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
    from backpropagate.gpu_safety import GPUStatus, GPUCondition
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
    from backpropagate.gpu_safety import GPUStatus, GPUCondition
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
    from backpropagate.multi_run import MultiRunConfig, MergeMode
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
    from backpropagate.gpu_safety import GPUStatus, GPUCondition
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
    from tests.test_helpers import CallbackSpy
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
    from tests.test_helpers import CallbackSpy

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
    from tests.test_helpers import CallbackTracker
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
    from tests.test_helpers import AsyncCallbackCollector

    def _create_collector(expected_count: int = 1):
        return AsyncCallbackCollector(expected_count=expected_count)

    return _create_collector
