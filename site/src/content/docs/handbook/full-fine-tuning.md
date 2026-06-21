---
title: Full fine-tuning (mode="full")
description: Card-aware full fine-tuning on one GPU — the 4-addend VRAM ceiling, the FSDP2 CPU-offload path for 7B-class models, and the quality math from Biderman 2024 / Thinking Machines 2025.
sidebar:
  order: 2.5
---

`mode="full"` updates every weight of the base model during training — no adapter, full-precision (bf16) weights. Backpropagate sizes it to the card you actually have: the parameter ceiling is **derived from your detected VRAM**, and `--full-ft-offload` extends it to **7B-class** by spilling params + optimizer state into host RAM via FSDP2. This page covers when to use it, the card-aware ceiling, the offload path, and the LoRA-vs-full quality math.

## TL;DR

- **Default is `mode="lora"`** (low-rank adapter; ~67% of the compute and matches full fine-tuning on most post-training tasks per [Biderman 2024](https://arxiv.org/abs/2405.09673) and [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)).
- **The full-FT ceiling is card-aware.** Derived from the 4-addend training-memory arithmetic against your detected VRAM: **16 GB → 4B, 24 GB → 5B, 32 GB → 6B** pure-GPU. A model past the ceiling exits with `RUNTIME_FULL_FT_MODEL_TOO_LARGE` — see [error codes](/backpropagate/handbook/error-codes/#runtime_full_ft_model_too_large).
- **`--full-ft-offload` lifts the ceiling to 7B-class** (≈8B on a 32 GB card) via FSDP2 `fully_shard` + `CPUOffloadPolicy`, spilling params + optimizer to ~64 GB host RAM. Slower (PCIe/CPU-bandwidth-bound); needs an NCCL backend (Linux/WSL2 — not Windows-native).
- **`--full-ft-ceiling-billions B`** overrides the derived ceiling explicitly when you know your card's headroom.

## When to use `mode="full"`

The honest, narrow case: you have measured a quality gap between LoRA and full fine-tuning on YOUR task, on YOUR data, and you've decided to spend the extra compute to close it. Most operators most of the time should stay with LoRA. The Biderman 2024 + Thinking Machines 2025 data make this clear:

- **Instruction following / RLHF-style preference learning:** LoRA at correct rank (256+) matches full FT.
- **Persona / style transfer:** LoRA matches.
- **Domain adaptation (medical / legal / financial):** LoRA matches at rank 256 + all-linear target modules.
- **Code generation (very long-tail token distribution):** small but consistent gap in favor of full FT.
- **Heavy reasoning post-training (RLHF on math / chain-of-thought):** full FT pulls ahead measurably.

If you're in one of the last two categories AND you've benchmarked the gap on your specific data, `mode="full"` is the tool. Otherwise stick with LoRA — the quality math is overwhelmingly in its favor at 67% of the compute, and QLoRA fits 7B–34B on a 32 GB card.

## When to STAY with LoRA / QLoRA

- You haven't measured a gap on your specific task.
- Your model is past the full-FT ceiling even with offload (≈13B+ on a 32 GB card) — QLoRA handles 14B–34B there instead.
- You're prototyping (LoRA's faster iteration loop dominates the trade-off).
- You're doing single-task instruction tuning, persona transfer, or domain adaptation — the three cases the literature has settled in LoRA's favor.

## Python API

```python
from backpropagate import Trainer

# Default: LoRA, rank 256, all-linear target modules (v1.3 quality preset).
trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)

# Pure-GPU full fine-tuning of a model within the card-aware ceiling
# (e.g. a genuine ~3B on 16 GB, up to ~6B on a 32 GB card).
trainer = Trainer("smollm3-3b", mode="full")
trainer.train("my_data.jsonl", steps=100)

# 7B-class full fine-tuning on a 32 GB card via FSDP2 CPU-offload
# (spills params + optimizer to host RAM; Linux/WSL2, ~64 GB host RAM).
trainer = Trainer("Qwen/Qwen2.5-7B-Instruct", mode="full", full_ft_offload=True)
trainer.train("my_data.jsonl", steps=100)

# Override the ceiling explicitly (you know your card's headroom):
trainer = Trainer("Qwen/Qwen2.5-7B-Instruct", mode="full", full_ft_ceiling_billions=8.0)
```

## CLI

```bash
# Default (LoRA on the canonical 7B):
backprop train --data my_data.jsonl --steps 100

# Pure-GPU full fine-tuning within the card-aware ceiling:
backprop train --model smollm3-3b --mode full --data my_data.jsonl --steps 100

# 7B-class full fine-tuning via FSDP2 CPU-offload (Linux/WSL2):
backprop train --model Qwen/Qwen2.5-7B-Instruct --mode full --full-ft-offload \
  --data my_data.jsonl --steps 100

# Override the ceiling explicitly:
backprop train --model Qwen/Qwen2.5-7B-Instruct --mode full --full-ft-ceiling-billions 8 \
  --data my_data.jsonl --steps 100
```

## What changes when `mode="full"` is set

The trainer applies these mode-specific settings inside `_build_sft_config`:

1. **No PEFT config** — full FT bypasses LoRA / adapter machinery entirely; every weight updates via standard backprop on full-precision (bf16) weights (no 4-bit base).
2. **`gradient_checkpointing=True` by default** — activation memory scales as sqrt(L) instead of linearly in layer count. (When `--full-ft-offload` is on, checkpointing moves into the FSDP config as `activation_checkpointing` to avoid a redundant all-gather.)
3. **`paged_adamw_8bit` optimizer** — the paged 8-bit Adam the consumer-card path uses, applied to every parameter. 2 momentum buffers × 1 byte + the gradient in bf16 keeps the optimizer addend small.
4. **Learning rate divided by 10** — full FT literature recommends ~10× lower LR than LoRA. Default LoRA LR `2e-4` → default full FT LR `2e-5`. Override via `learning_rate=`.

## The card-aware ceiling (the 4-addend arithmetic)

Training VRAM is the sum of four addends — **model weights + gradients + optimizer state + activations**. For full fine-tuning every parameter is trainable, so weights (bf16, 2 B/param) + gradients (2 B/param) + paged 8-bit AdamW optimizer (~2 B/param) dominate at ≈6 B/param GPU-resident. Against detected VRAM that yields the pure-GPU ceiling:

| Card | Pure-GPU full-FT ceiling | With `--full-ft-offload` |
|---|---|---|
| 16 GB | 4B | 4B (offload needs NCCL/host RAM) |
| 24 GB | 5B | 7B |
| 32 GB | **6B** | **8B (7B-class)** |
| 48 GB+ | 10B | 16B |

The ceiling bounds the parameter **count** — it does not promise a fit for every admitted model under every sequence length. It is enforced in two places:

1. **`Trainer.__init__`** — preset-table / model-id lookup; the gate fires before the model is loaded.
2. **`Trainer.load_model()`** — second check via `model.num_parameters()` (authoritative). Catches custom model ids the preset table doesn't know.

A model past the (offload-aware) ceiling exits `2` with `RUNTIME_FULL_FT_MODEL_TOO_LARGE`. The error is **contrastive**: if your model clears the offload ceiling but not the pure-GPU one, it names `--full-ft-offload`; if it's past even the offload ceiling, it names LoRA / QLoRA.

## FSDP2 CPU-offload (`--full-ft-offload`)

When a true full fine-tune of a 7B-class model won't fit pure-GPU on a 32 GB card, `--full-ft-offload` configures FSDP2 (`fully_shard` + `CPUOffloadPolicy` + activation checkpointing + bf16) to spill params + optimizer state into host RAM. This is the studio's **measured** recipe — a 7B-class full fine-tune fits a 32 GB card backed by ~64 GB host RAM.

It is a **documented escape hatch, not the default**:

- **Slower.** The run is PCIe/CPU-bandwidth-bound, not compute-bound. Budget for a noticeably slower run than a model that fits pure-GPU.
- **Needs host RAM.** ~64 GB is the measured target; the trainer warns if it detects much less.
- **Needs an NCCL backend → Linux / WSL2.** FSDP CUDA collectives require NCCL, which is not available on Windows-native (gloo can't carry CUDA collectives). On a host without NCCL the path fails fast with `DEP_FSDP_UNAVAILABLE` naming the WSL2 / Linux requirement — it never silently runs without offload and OOMs. The trainer initializes a single-process group automatically, so a bare `backprop train` works (no `accelerate launch` / `torchrun` needed) on a capable host.

If FSDP2 CPU-offload still OOMs host RAM, the next step is DeepSpeed ZeRO-Infinity NVMe offload (outside Backpropagate's single-command scope) — or drop to QLoRA, which fits 7B–34B on a 32 GB card natively.

## Quality math: should I switch from LoRA to full FT?

The empirical answer for most operators is **no**. The two load-bearing references:

- **[Biderman et al. (2024). "LoRA Learns Less and Forgets Less."](https://arxiv.org/abs/2405.09673)** Authoritative head-to-head LoRA vs full FT across instruction tuning, math reasoning, and code. Headline: at correct rank (256+) + all-linear targets + 10× LR scale, LoRA matches full FT on instruction tuning + math and trails by ~5% on code, at ~67% of the compute.
- **[Thinking Machines (2025). "LoRA Without Regret."](https://thinkingmachines.ai/blog/lora/)** Replication + ablations. Headline: the "LoRA loses to full FT" stories measured rank 8–16 LoRA — a setup the v1.3 quality preset already left behind. At rank 256 + all-linear, LoRA matches or wins on most post-training benchmarks.

The v1.3 quality preset (`--lora-preset=quality`, the default) is the bar the literature compares to. If you haven't measured a gap between that preset and full FT on your data, the literature says you won't find one — and QLoRA up to 34B on one card is the leaner path.

## See also

- [Error codes → `RUNTIME_FULL_FT_MODEL_TOO_LARGE`](/backpropagate/handbook/error-codes/#runtime_full_ft_model_too_large) · [`DEP_FSDP_UNAVAILABLE`](/backpropagate/handbook/error-codes/#dep_fsdp_unavailable)
- [CLI reference → `backprop train`](/backpropagate/handbook/cli-reference/#backprop-train) — `--mode`, `--full-ft-offload`, `--full-ft-ceiling-billions`.
- [What you can fine-tune on one GPU](/backpropagate/) — the full envelope (QLoRA 7B–34B + full-FT tiers).
