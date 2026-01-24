<div align="center">

# Backpropagate

**Headless LLM Fine-Tuning** - Making fine-tuning accessible without the complexity

[![PyPI version](https://img.shields.io/pypi/v/backpropagate?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/backpropagate/)
[![Downloads](https://img.shields.io/pypi/dm/backpropagate?color=green&logo=pypi&logoColor=white)](https://pypi.org/project/backpropagate/)
[![CI](https://github.com/mcp-tool-shop/backpropagate/actions/workflows/ci.yml/badge.svg)](https://github.com/mcp-tool-shop/backpropagate/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mcp-tool-shop/backpropagate/graph/badge.svg)](https://codecov.io/gh/mcp-tool-shop/backpropagate)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/mcp-tool-shop/backpropagate?style=social)](https://github.com/mcp-tool-shop/backpropagate)

*Train LLMs in 3 lines of code. Export to Ollama in one more.*

[Installation](#installation) • [Quick Start](#quick-start) • [Multi-Run Training](#multi-run-training-slao) • [Export to Ollama](#export--ollama-integration) • [Contributing](#contributing)

</div>

---

## Why Backpropagate?

| Problem | Solution |
|---------|----------|
| Fine-tuning is complex | 3 lines: load, train, save |
| Windows is a nightmare | First-class Windows support |
| VRAM management is hard | Auto batch sizing, GPU monitoring |
| Model export is confusing | One-click GGUF + Ollama registration |
| Long runs cause forgetting | Multi-run SLAO training |

<!--
## Demo

<p align="center">
  <img src="docs/assets/demo.gif" alt="Backpropagate Demo" width="600">
</p>
-->

## Quick Start

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

## Philosophy

- **For Users**: Upload data, pick a model, click train
- **For Developers**: Clean Python API with smart defaults
- **For Everyone**: Windows-safe, VRAM-aware, production-ready

## Installation

### Modular Installation (v0.1.0+)

Install only what you need:

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

### Available Extras

| Extra | Description | Dependencies |
|-------|-------------|--------------|
| `unsloth` | 2x faster training, 50% less VRAM | unsloth |
| `ui` | Gradio web interface | gradio>=5.6.0 |
| `validation` | Pydantic config validation | pydantic, pydantic-settings |
| `export` | GGUF export for Ollama | llama-cpp-python |
| `monitoring` | WandB + system monitoring | wandb, psutil |

### Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- PyTorch 2.0+

## Usage

### Use as Library

```python
from backpropagate import Trainer

# Dead simple
trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")

# Export to GGUF for Ollama
trainer.export("gguf", quantization="q4_k_m")
```

### With Options

```python
from backpropagate import Trainer

trainer = Trainer(
    model="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    lora_r=32,
    lora_alpha=64,
    learning_rate=1e-4,
    batch_size="auto",  # Auto-detects based on VRAM
)

run = trainer.train(
    dataset="HuggingFaceH4/ultrachat_200k",
    steps=200,
    samples=2000,
)

print(f"Final loss: {run.final_loss:.4f}")
print(f"Duration: {run.duration_seconds:.1f}s")
```

### Launch the Web UI

```bash
# CLI
backpropagate --ui

# Or from Python
from backpropagate import launch
launch(port=7862)
```

## Multi-Run Training (SLAO)

Multiple short runs with LoRA merging prevents catastrophic forgetting and improves results:

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")

# Run 5 training runs, each on fresh data
result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
    merge_mode="slao",  # Smart LoRA merging
)

print(f"Final loss: {result.final_loss:.4f}")
print(f"Total time: {result.total_time_seconds:.1f}s")
```

Or use the dedicated trainer:

```python
from backpropagate import MultiRunTrainer, MultiRunConfig

config = MultiRunConfig(
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
)

trainer = MultiRunTrainer(
    model="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    config=config,
)

result = trainer.run("my_data.jsonl")
```

## CLI Usage

```bash
# Show system info and features
backprop info

# Show current configuration
backprop config

# Train a model
backprop train \
    --data my_data.jsonl \
    --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
    --steps 100 \
    --samples 1000

# Multi-run training (recommended for best results)
backprop multi-run \
    --data HuggingFaceH4/ultrachat_200k \
    --runs 5 \
    --steps 100 \
    --samples 1000

# Export to GGUF for Ollama
backprop export ./output/lora \
    --format gguf \
    --quantization q4_k_m \
    --ollama \
    --ollama-name my-model

# Launch UI
backpropagate --ui --port 7862
```

## Feature Flags

Check which features are installed:

```python
from backpropagate import FEATURES, list_available_features

print(FEATURES)
# {'unsloth': True, 'ui': True, 'validation': False, ...}

for name, desc in list_available_features().items():
    print(f"{name}: {desc}")
```

## Configuration

All settings can be overridden via environment variables:

```bash
# Model settings
BACKPROPAGATE_MODEL__NAME=unsloth/Llama-3.2-3B-Instruct-bnb-4bit
BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH=4096

# Training settings
BACKPROPAGATE_TRAINING__LEARNING_RATE=1e-4
BACKPROPAGATE_TRAINING__MAX_STEPS=200
BACKPROPAGATE_TRAINING__BATCH_SIZE=4

# LoRA settings
BACKPROPAGATE_LORA__R=32
BACKPROPAGATE_LORA__ALPHA=64
```

Or use a `.env` file in your project root.

## Dataset Formats

### JSONL (Recommended)

```json
{"text": "<|im_start|>user\nWhat is Python?<|im_end|>\n<|im_start|>assistant\nPython is a programming language.<|im_end|>"}
```

### HuggingFace Datasets

Any dataset with a `text` column works:

```python
trainer.train(dataset="HuggingFaceH4/ultrachat_200k", samples=1000)
```

## Export & Ollama Integration

Export trained models to various formats:

```python
from backpropagate import (
    export_lora,
    export_merged,
    export_gguf,
    create_modelfile,
    register_with_ollama,
)

# Export to GGUF for Ollama/llama.cpp
result = export_gguf(
    model,
    tokenizer,
    output_dir="./gguf",
    quantization="q4_k_m",  # f16, q8_0, q5_k_m, q4_k_m, q4_0, q2_k
)

print(result.summary())

# Register with Ollama
register_with_ollama("./gguf/model-q4_k_m.gguf", "my-model")
# Now run: ollama run my-model
```

## GPU Safety Monitoring

Monitor GPU health during training:

```python
from backpropagate import check_gpu_safe, get_gpu_status, GPUMonitor

# Quick safety check
if check_gpu_safe():
    print("GPU is ready for training")

# Get detailed status
status = get_gpu_status()
print(f"GPU: {status.device_name}")
print(f"Temperature: {status.temperature_c}C")
print(f"VRAM: {status.vram_used_gb:.1f}/{status.vram_total_gb:.1f} GB")
print(f"Condition: {status.condition}")  # SAFE, WARNING, CRITICAL

# Continuous monitoring during training
with GPUMonitor(check_interval=30) as monitor:
    trainer.train(dataset, steps=1000)
```

## Windows Support

Backpropagate is designed to work on Windows out of the box:

- Pre-tokenization to avoid multiprocessing crashes
- Automatic xformers disable for RTX 40/50 series
- Safe dataloader settings
- Tested on RTX 5080 (16GB VRAM)

Windows fixes are applied automatically when `os.name == "nt"`.

## Model Presets

| Preset | VRAM | Speed | Quality |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12GB | Medium | Best |
| Qwen 2.5 3B | ~8GB | Fast | Good |
| Llama 3.2 3B | ~8GB | Fast | Good |
| Llama 3.2 1B | ~6GB | Fastest | Basic |
| Mistral 7B | ~12GB | Medium | Good |

## Architecture

```
backpropagate/
├── __init__.py          # Package exports, lazy loading
├── __main__.py          # CLI entry point
├── cli.py               # Command-line interface
├── trainer.py           # Core Trainer class
├── multi_run.py         # Multi-run SLAO training
├── slao.py              # SLAO LoRA merging algorithm
├── datasets.py          # Dataset loading & filtering
├── export.py            # GGUF/Ollama export
├── config.py            # Pydantic settings
├── feature_flags.py     # Optional dependency detection
├── gpu_safety.py        # GPU monitoring & safety
├── theme.py             # Ocean Mist Gradio theme
└── ui.py                # Gradio interface
```

## API Reference

### Trainer

```python
class Trainer:
    def __init__(
        self,
        model: str = None,           # Model name/path
        lora_r: int = 16,            # LoRA rank
        lora_alpha: int = 32,        # LoRA alpha
        learning_rate: float = 2e-4, # Learning rate
        batch_size: int | str = "auto",  # Batch size or "auto"
        output_dir: str = "./output",    # Output directory
    )

    def train(
        self,
        dataset: str | Dataset,  # Dataset path or HF name
        steps: int = 100,        # Training steps
        samples: int = 1000,     # Max samples
    ) -> TrainingRun

    def save(self, path: str = None) -> str
    def export(self, format: str, quantization: str = "q4_k_m") -> str
```

### TrainingRun

```python
@dataclass
class TrainingRun:
    run_id: str
    steps: int
    final_loss: float
    loss_history: List[float]
    duration_seconds: float
    samples_seen: int
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/mcp-tool-shop/backpropagate
cd backpropagate
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy backpropagate

# Linting
ruff check backpropagate
```

## Related Projects

Part of the **Compass Suite** for AI-powered development:

- [Tool Compass](https://github.com/mcp-tool-shop/tool-compass) - Semantic MCP tool discovery
- [File Compass](https://github.com/mcp-tool-shop/file-compass) - Semantic file search
- [Integradio](https://github.com/mcp-tool-shop/integradio) - Vector-embedded Gradio components
- [Comfy Headless](https://github.com/mcp-tool-shop/comfy-headless) - ComfyUI without the complexity

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for the amazing training optimizations
- [HuggingFace](https://huggingface.co/) for transformers, datasets, and PEFT
- [Gradio](https://gradio.app/) for the beautiful UI framework

---

<div align="center">

**[Documentation](https://github.com/mcp-tool-shop/backpropagate#readme)** • **[Issues](https://github.com/mcp-tool-shop/backpropagate/issues)** • **[Discussions](https://github.com/mcp-tool-shop/backpropagate/discussions)**

</div>
