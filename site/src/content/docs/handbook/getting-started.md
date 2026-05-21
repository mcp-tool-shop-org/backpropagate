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

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("examples/quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

The repo ships a small `examples/quickstart.jsonl` (5 ShareGPT examples) so the snippet above runs end-to-end on a clean install. `Qwen/Qwen2.5-7B-Instruct` is the canonical default — what `Trainer()` resolves with no model argument. For your own training, see the [dataset format docs](/backpropagate/handbook/training/#dataset-formats).

## CLI usage

```bash
backprop train --data my_data.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
```

See [CLI reference](/backpropagate/handbook/cli-reference/) for every flag.

## Verify your setup

```bash
backprop info
```

This prints your Python version, PyTorch version, CUDA status, GPU name and VRAM, which optional features are installed, and current configuration defaults. Run it before your first training to confirm everything is working.

## Web UI

```bash
backprop ui --port 7862
```

If you want a public-internet URL via Gradio's `--share`, you must also pass `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` without `--auth` exits with code `1` and `[INPUT_AUTH_REQUIRED]`. Without auth, anyone on the internet could drive your training pipeline. To opt out for an internal-only dev environment, set `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false` (loud warning prints at startup).

## What success looks like

A successful first training run prints something like:

```
run_started run_id=8f3a2c1d-9e4b-4c5a-...
Trainer initialized: Qwen/Qwen2.5-7B-Instruct
  LoRA: r=16, alpha=32
  Batch: 2, LR: 0.0002
  Degradation knobs: oom_recovery=True, unsloth_fallback=True
Training: [####################] 100% loss=0.42  steps=10
Saved to ./output/lora
run_ended run_id=8f3a2c1d-... duration_seconds=43.2
```

After the run completes:

```
output/
├── lora/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer.json
└── checkpoints/
    └── manifest.json
```

To verify the adapter loaded cleanly, run `backprop info` — it lists which optional features are detected and which model the resolved config points at. If you see the `Saved to ...` line and the directory tree above exists, your first run worked.

## Troubleshooting

If something failed during installation or first run, see the [troubleshooting page](/backpropagate/handbook/troubleshooting/) — it's a symptoms-first reverse index keyed by what you actually saw in stderr.
