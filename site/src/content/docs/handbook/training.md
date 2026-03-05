---
title: Training
description: Basic training, multi-run SLAO, and model presets.
sidebar:
  order: 2
---

## Basic training

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
```

Smart defaults automatically configure learning rate, batch size, gradient accumulation, and LoRA rank based on your hardware and dataset size.

## Multi-run SLAO training

SLAO (Smart Loss-Aware Ordering) prevents catastrophic forgetting during extended fine-tuning:

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")

result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
    merge_mode="slao",
)
```

CLI equivalent:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
```

## VRAM-aware batch sizing

Backpropagate monitors GPU memory and automatically adjusts batch size and gradient checkpointing to keep training stable. Built-in warnings fire before OOM. Works from 8GB up to multi-GPU setups.

## Model presets

| Preset | VRAM | Speed | Quality |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12GB | Medium | Best |
| Qwen 2.5 3B | ~8GB | Fast | Good |
| Llama 3.2 3B | ~8GB | Fast | Good |
| Llama 3.2 1B | ~6GB | Fastest | Basic |
| Mistral 7B | ~12GB | Medium | Good |

## Training methods

- **LoRA** — Low-rank adaptation for efficient fine-tuning
- **QLoRA** — 4-bit quantized LoRA for minimal VRAM usage
- **Unsloth** — 2x faster training with 50% less VRAM when installed
