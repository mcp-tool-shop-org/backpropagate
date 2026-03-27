---
title: Getting Started
description: Install Backpropagate and train your first model.
sidebar:
  order: 1
---

## Prerequisites

- Python 3.10+
- CUDA GPU (8GB+ VRAM)
- PyTorch 2.0+

## Installation

```bash
pip install backpropagate[standard]    # Recommended: unsloth + ui
```

Other install options:

| Extra | What you get |
|-------|-------------|
| `backpropagate` | Core API only — minimal footprint |
| `[unsloth]` | 2x faster training, 50% less VRAM |
| `[ui]` | Gradio web interface |
| `[validation]` | Pydantic config validation |
| `[export]` | GGUF export for Ollama |
| `[monitoring]` | WandB + system monitoring |
| `[observability]` | OpenTelemetry tracing |
| `[logging]` | Structured logging via structlog |
| `[security]` | JWT auth + token generation |
| `[standard]` | unsloth + ui (recommended) |
| `[production]` | unsloth + ui + validation + logging + security |
| `[full]` | Everything |

## Train in 3 lines

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

## CLI usage

```bash
backprop train --data my_data.jsonl --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
```

## Verify your setup

```bash
backprop info
```

This prints your Python version, PyTorch version, CUDA status, GPU name and VRAM, which optional features are installed, and current configuration defaults. Run it before your first training to confirm everything is working.

## Web UI

```bash
backprop ui --port 7862
```
