# Backpropagate

**Headless LLM Fine-Tuning** - Making fine-tuning accessible without the complexity.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/mikeyfrilot/backpropagate/actions/workflows/ci.yml/badge.svg)](https://github.com/mikeyfrilot/backpropagate/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mikeyfrilot/backpropagate/graph/badge.svg)](https://codecov.io/gh/mikeyfrilot/backpropagate)
[![PyPI version](https://badge.fury.io/py/backpropagate.svg)](https://badge.fury.io/py/backpropagate)

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

## Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- PyTorch 2.0+

## Quick Start

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

## Feature Flags

Check which features are installed:

```python
from backpropagate import FEATURES, list_available_features

print(FEATURES)
# {'unsloth': True, 'ui': True, 'validation': False, ...}

for name, desc in list_available_features().items():
    print(f"{name}: {desc}")
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
{"text": "<|im_start|>user\nExplain ML<|im_end|>\n<|im_start|>assistant\nML is...<|im_end|>"}
```

### CSV

```csv
text
"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>"
```

### HuggingFace Datasets

Any dataset with a `text` column works:

```python
trainer.train(dataset="HuggingFaceH4/ultrachat_200k", samples=1000)
```

## Advanced Features

### Multi-Run Training (SLAO)

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

### Dataset Loading & Filtering

Load, validate, and filter datasets with quality controls:

```python
from backpropagate import DatasetLoader, detect_format

# Auto-detect format and load
loader = DatasetLoader("my_data.jsonl")
print(f"Format: {loader.detected_format}")
print(f"Samples: {len(loader)}")
print(f"Valid: {loader.is_valid}")

# Preview samples
for sample in loader.preview(3):
    print(sample)

# Convert to ChatML format
chatml_data = loader.to_chatml()

# Filter by quality
filtered = loader.filter(
    min_tokens=50,
    max_tokens=2048,
    min_turns=2,
    require_assistant=True,
)

# Remove duplicates
deduped = loader.deduplicate(method="exact")  # or "minhash"
```

### Perplexity-Based Filtering

Filter samples by perplexity score to remove outliers (requires model inference):

```python
from backpropagate import DatasetLoader, PerplexityFilter, filter_by_perplexity

# Option 1: Use DatasetLoader method
loader = DatasetLoader("my_data.jsonl")
filtered_loader, stats = loader.filter_perplexity(
    model_name="gpt2",       # Model for scoring (gpt2, gpt2-medium, etc.)
    min_percentile=5,        # Remove bottom 5% (too simple/repetitive)
    max_percentile=95,       # Remove top 5% (noisy/unusual)
)
print(stats.summary())

# Option 2: Use standalone function
samples = [{"text": "sample 1"}, {"text": "sample 2"}]
filtered, stats = filter_by_perplexity(
    samples,
    model_name="gpt2",
    min_percentile=5,
    max_percentile=95,
)

# Option 3: Use PerplexityFilter class for more control
pf = PerplexityFilter(model_name="gpt2", device="cuda", batch_size=16)
scores = pf.score(samples)  # Get raw scores
filtered = pf.filter_by_threshold(samples, scores, min_perplexity=10, max_perplexity=500)
```

Perplexity measures how "surprised" a language model is by text:
- **Low perplexity**: Very predictable (may be too simple or repetitive)
- **Medium perplexity**: Natural, typical language
- **High perplexity**: Unusual (may be noisy or low-quality)

### Export & Ollama Integration

Export trained models to various formats:

```python
from backpropagate import (
    export_lora,
    export_merged,
    export_gguf,
    create_modelfile,
    register_with_ollama,
)

# Export LoRA adapter
result = export_lora(model, output_dir="./lora")

# Export merged model (base + adapter)
result = export_merged(model, tokenizer, output_dir="./merged")

# Export to GGUF for Ollama/llama.cpp
result = export_gguf(
    model,
    tokenizer,
    output_dir="./gguf",
    quantization="q4_k_m",  # f16, q8_0, q5_k_m, q4_k_m, q4_0, q2_k
)

print(result.summary())
# Export Complete
#   Format: gguf
#   Path: ./gguf/model-q4_k_m.gguf
#   Size: 4096.0 MB
#   Quantization: q4_k_m
#   Time: 120.5s

# Create Ollama Modelfile
create_modelfile(
    "./gguf/model-q4_k_m.gguf",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
)

# Register with Ollama
register_with_ollama("./gguf/model-q4_k_m.gguf", "my-model")
# Now run: ollama run my-model
```

### GPU Safety Monitoring

Monitor GPU health during training:

```python
from backpropagate import (
    check_gpu_safe,
    get_gpu_status,
    wait_for_safe_gpu,
    GPUMonitor,
)

# Quick safety check
if check_gpu_safe():
    print("GPU is ready for training")

# Get detailed status
status = get_gpu_status()
print(f"GPU: {status.device_name}")
print(f"Temperature: {status.temperature_c}C")
print(f"VRAM: {status.vram_used_gb:.1f}/{status.vram_total_gb:.1f} GB")
print(f"Condition: {status.condition}")  # SAFE, WARNING, CRITICAL, EMERGENCY

# Wait for GPU to cool down
wait_for_safe_gpu(max_wait=300)  # Wait up to 5 minutes

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

### Key Design Principles

1. **Modular by default** - Install only what you need
2. **Smart defaults** - Works out of the box
3. **Windows-first** - No multiprocessing nightmares
4. **Fail gracefully** - Helpful error messages
5. **Type-safe** - Full type hints

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
git clone https://github.com/mikeyfrilot/backpropagate
cd backpropagate
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy backpropagate

# Linting
ruff check backpropagate
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for the amazing training optimizations
- [HuggingFace](https://huggingface.co/) for transformers, datasets, and PEFT
- [Gradio](https://gradio.app/) for the beautiful UI framework
- Built with the same love as [Comfy Headless](https://github.com/mikeyfrilot/comfy-headless) and [Tool Compass](https://github.com/mikeyfrilot/tool-compass)
