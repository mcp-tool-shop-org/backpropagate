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
| `[ui]` | Reflex (Radix UI) web interface |
| `[validation]` | Pydantic config validation |
| `[export]` | GGUF export for Ollama |
| `[monitoring]` | WandB + system monitoring |
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

For remote access, use SSH port-forwarding rather than `--share`:

```bash
ssh -L 7860:localhost:7860 you@gpu-host
# Then on your laptop: http://localhost:7860
```

`backprop ui --share` and `backprop ui --auth USER:PASS` both exit `1` with `[RUNTIME_UI_AUTH_NOT_ENFORCED]` — the v1.1+ Reflex UI ships ahead of its auth middleware, so the runtime refuses to start either flag rather than expose an unauthenticated public URL. The refuse-to-start contract is enforced one layer deeper as well, so `python -m reflex run` from the package directory also refuses. See [the troubleshooting page](/backpropagate/handbook/troubleshooting/#what-does-runtime_ui_auth_not_enforced-mean) for the long version and the tracking GHSA.

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

## Upgrading from v1.1.x

If you are coming from v1.1.0 / v1.1.1, see the [migration page](/backpropagate/handbook/migrations/) for the breaking changes (notably the `--share` refuse-to-start contract and the removed `[observability]` extra) and the behavioural fixes that may surface bugs in your callback code.

## Exposing the UI to the network

For SSH port-forwarding, host/origin allowlists, the v1.1.x GHSA advisory, and the four-layer auth defense — see the [security page](/backpropagate/handbook/security/).
