---
title: Full fine-tuning (mode="full")
description: When to use full fine-tuning over LoRA, the 4B parameter ceiling, and the quality math from Biderman 2024 / Thinking Machines 2025.
sidebar:
  order: 2.5
---

`mode="full"` updates every weight of the base model during training, the same way HuggingFace's `transformers.Trainer` does on a 24GB+ datacenter card. It ships in v1.4 with a hard 4B parameter ceiling: the genuine ~3B presets fit a 16GB consumer card, the 3.8–4B class needs a 24GB+ card, and operators with a 7B+ model still reach for `transformers.Trainer` directly. This page covers when to use it, when NOT to use it, and how the trainer enforces the ceiling.

## TL;DR

- **Default is `mode="lora"`** (low-rank adapter; tracks ~70% of compute and matches full fine-tuning on most post-training tasks per [Biderman 2024](https://arxiv.org/abs/2405.09673) and [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)).
- **`mode="full"` only for ≤4B parameter models.** The genuine ~3B presets (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B, Llama-3.2-1B) fit a 16GB card; the 3.8–4B class (Phi-4-mini-3.8B, Qwen-3.5-4B) is admitted but wants a 24GB+ card. Models >4B exit with `RUNTIME_FULL_FT_MODEL_TOO_LARGE` — see [error codes](/backpropagate/handbook/error-codes/#runtime_full_ft_model_too_large).
- **The escape hatches when the gate fires** are `mode="lora"` (the default — fits 7B+ on 16GB) OR a preset under the ceiling: on a 16GB card pick a genuine ~3B (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B) or Llama-3.2-1B.

## When to use `mode="full"`

The honest, narrow case: you have measured a quality gap between LoRA and full fine-tuning on YOUR task, on YOUR data, and you've decided to spend the extra compute to close it. Most operators most of the time should stay with LoRA. The Biderman 2024 + Thinking Machines 2025 data make this clear:

- **Instruction following / RLHF-style preference learning:** LoRA at correct rank (256+) matches full FT.
- **Persona / style transfer:** LoRA matches.
- **Domain adaptation (medical / legal / financial):** LoRA matches at rank 256 + all-linear target modules.
- **Code generation (very long-tail token distribution):** small but consistent gap in favor of full FT.
- **Heavy reasoning post-training (RLHF on math / chain-of-thought):** full FT pulls ahead measurably.

If you're in one of the last two categories AND you've benchmarked the gap on your specific data, `mode="full"` is the tool. Otherwise stick with LoRA — the quality math is overwhelmingly in its favor at 67% of the compute.

## When to STAY with LoRA

- You haven't measured a gap on your specific task.
- Your model is > 4B parameters (the trainer will refuse `mode="full"`).
- You're prototyping (LoRA's faster iteration loop dominates the trade-off).
- You're doing single-task instruction tuning, persona transfer, or domain adaptation — the three cases the literature has settled in LoRA's favor.

## Python API

```python
from backpropagate import Trainer

# Default: LoRA, rank 256, all-linear target modules (v1.3 quality preset).
trainer = Trainer("phi-4-mini-3.8b")
trainer.train("my_data.jsonl", steps=100)

# Full fine-tuning. The 4B ceiling fires at __init__ for known presets.
trainer = Trainer("phi-4-mini-3.8b", mode="full")
trainer.train("my_data.jsonl", steps=100)

# This raises FullFinetuneModelTooLargeError (RUNTIME_FULL_FT_MODEL_TOO_LARGE):
trainer = Trainer("Qwen/Qwen2.5-7B-Instruct", mode="full")
```

## CLI

```bash
# Default (LoRA on the canonical 7B):
backprop train --data my_data.jsonl --steps 100

# Full fine-tuning on a sub-3B preset:
backprop train --model phi-4-mini-3.8b --mode=full --data my_data.jsonl --steps 100

# Multi-run full fine-tuning:
backprop multi-run --model smollm3-3b --mode=full --runs 5 --steps 100 --data my_data.jsonl

# The trainer refuses 7B with mode=full (exits 2 with RUNTIME_FULL_FT_MODEL_TOO_LARGE):
backprop train --model Qwen/Qwen2.5-7B-Instruct --mode=full --data my_data.jsonl
```

## What changes when `mode="full"` is set

The trainer applies four mode-specific settings inside `_build_sft_config`:

1. **No PEFT config** — full FT bypasses LoRA / adapter machinery entirely. `transformers.Trainer` updates every weight via standard backprop.
2. **`gradient_checkpointing=True` by default** — activation memory scales as sqrt(L) instead of linearly in layer count. Without this, the activation memory alone would blow a 16GB budget at a 3B model.
3. **`paged_adamw_8bit` optimizer** — the same paged 8-bit Adam the consumer-card LoRA path uses, but applied to every parameter (not just the adapter). 2 momentum buffers × 1 byte each + the gradient in bf16 keeps total optimizer state under the ceiling.
4. **Learning rate divided by 10** — full FT literature (Biderman 2024 / Thinking Machines 2025) consistently recommends ~10× lower LR than LoRA. Default LoRA LR is `2e-4`; default full FT LR becomes `2e-5`. Override via `Trainer(learning_rate=...)` to set explicitly.

## The 4B parameter ceiling

`mode="full"` for models > 4B raises `FullFinetuneModelTooLargeError` (code `RUNTIME_FULL_FT_MODEL_TOO_LARGE`). The ceiling is 4B (not 3B) on purpose: the marketed "3B" models have a true `num_parameters()` of 3.08–3.24B, so a 3.0B ceiling rejected every one of them at load time. The ceiling bounds the parameter COUNT only — it does not promise a 16GB fit (a genuine ~3B fits 16GB; the 3.8–4B class needs 24GB+). The ceiling is enforced in two places:

1. **`Trainer.__init__`** — preset-table lookup. If the model name resolves to a known preset (e.g. `qwen2.5-7b`), the parameter count is read from the preset and the gate fires before the model is loaded.
2. **`Trainer.load_model()`** — second check, this time using `model.num_parameters()` directly. Catches the case where the preset table doesn't know about the model id (custom HF model, fine-tune of a fine-tune, etc.); the actual loaded parameter count is the authoritative signal.

This is intentional belt-and-braces. An unknown 7B fine-tune that slips past the preset-table check still gets caught at load time. The trainer never starts training a >4B model in `mode="full"`.

### What if I genuinely have a 24GB card?

The 4B ceiling bounds the parameter count; the VRAM reality is tiered. The genuine ~3B presets fit a 16GB consumer card (RTX 4080 / 5080 / 4070 Ti Super). The 3.8–4B class (Phi-4-mini-3.8B, Qwen-3.5-4B) is admitted by the ceiling but wants a 24GB card (RTX 4090 / 3090 / A5000) — weights + gradients alone approach 16GB before the optimizer and activations. Operators on 24GB cards can also full-FT a 7B with more aggressive memory engineering, and 40GB+ datacenter cards (A100, H100) handle 13B+ — but for those v1.4 does NOT expose a `--full-ft-ceiling-billions` flag (v1.5 may), so the path is HuggingFace `transformers.Trainer` directly. Backpropagate's full-FT value-add tops out at 4B.

## Quality math: should I switch from LoRA to full FT?

The empirical answer for most operators is **no**. The two load-bearing references:

- **[Biderman, Dawkins, Marshall, Cooper, Bach, Sourbut, Falconer, Webson, & Hewitt (2024). "LoRA Learns Less and Forgets Less."](https://arxiv.org/abs/2405.09673)** Authoritative head-to-head LoRA vs full FT comparison across instruction tuning, math reasoning, and code. Headline: at correct rank (256+) + target modules (all-linear) + 10× LR scale, LoRA matches full FT on instruction tuning + math, and trails by ~5% on code. Compute cost: LoRA ~67% of full FT.
- **[Thinking Machines (2025). "LoRA Without Regret."](https://thinkingmachines.ai/blog/lora/)** Replication + extension of Biderman 2024 with detailed ablations. Headline: the "LoRA loses to full FT" stories in earlier literature were measuring rank 8-16 LoRA against full FT — a setup the v1.3 quality preset already left behind. At rank 256 + all-linear, LoRA matches or wins on most post-training benchmarks.

The v1.3 quality preset (`--lora-preset=quality`, the v1.3 default) is the bar the literature is comparing to. If you haven't measured a gap between that preset and full FT on your data, the literature says you won't find one.

## See also

- [Error codes → `RUNTIME_FULL_FT_MODEL_TOO_LARGE`](/backpropagate/handbook/error-codes/#runtime_full_ft_model_too_large)
- [CLI reference → `backprop train --mode`](/backpropagate/handbook/cli-reference/#backprop-train)
- [Training → Trainer parameters](/backpropagate/handbook/training/#trainer-parameters) — every Trainer kwarg, including `mode`.
- [README → What backpropagate is NOT for](/backpropagate/) — the anti-pitch section names the 7B+ full FT exclusion.
