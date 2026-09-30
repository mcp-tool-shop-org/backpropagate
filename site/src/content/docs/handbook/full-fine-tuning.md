---
title: Full fine-tuning (mode="full")
description: Full fine-tuning on one GPU — the card-aware ceiling, the measured FSDP2 CPU-offload path for 7B-class models, and what the LoRA-vs-full evidence actually says.
sidebar:
  order: 2.5
---

`mode="full"` updates every weight of the base model — no adapter, bf16 weights. There are two ways to run it:

- **On the GPU** (the default for `mode="full"`): model, gradients and optimizer state all live in VRAM. Fast. Capped by your card.
- **With `--full-ft-offload`**: weights and gradients live in host RAM and stream to the GPU. Fits a 7B-class model on a 32 GB card, at a real cost in speed. Linux or WSL2 only.

This page covers when full fine-tuning is worth it, both paths, and the measured numbers behind them. Every figure marked *measured* comes from runs on an RTX 5090 (32 GB) on 2026-09-30; the receipts are in the repository under [`docs/receipts/2026-09-30-offload/`](https://github.com/mcp-tool-shop-org/backpropagate/tree/main/docs/receipts/2026-09-30-offload).

## TL;DR

- **Default is `mode="lora"`.** For most instruction, persona and style work on modest datasets, LoRA applied to every layer is the better use of one card. See [the evidence](#lora-or-full-fine-tuning-the-evidence) — it is more mixed than "LoRA always matches".
- **On the GPU, the ceiling is card-aware:** 16 GB → 4B, 24 GB → 5B, 32 GB → 6B. Those caps come from memory arithmetic; they have been *measured* only up to 3B (12.6 GB at 3B, batch 4, 512 tokens).
- **`--full-ft-offload` trains a 7.6B model on a 32 GB card** (*measured*: 5.3 GiB VRAM, 30.8 GiB host RAM, 14.7 s/step). The run is checked up front and refused, with the numbers, if the machine cannot hold it.
- **`--full-ft-ceiling-billions B`** overrides the ceiling (and turns a failed offload fit check into a warning) when you know better.

## When to use `mode="full"`

Full fine-tuning learns changes that a low-rank adapter cannot represent. [Biderman et al. 2024](https://arxiv.org/abs/2405.09673) found that full fine-tuning learns weight changes of 10–100× higher rank than typical LoRA settings, and that in standard low-rank settings LoRA substantially underperforms full fine-tuning on **code and math**. The same paper found LoRA **forgets less** of what the base model could already do.

[Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/) found LoRA matching full fine-tuning when two conditions hold: LoRA is applied to **every layer** (especially the MLP layers), and the dataset is **small enough for the adapter's capacity**. Past that capacity, LoRA underperforms.

So reach for `mode="full"` when:

- the task is code or math, or the dataset is large relative to an adapter's capacity, **and**
- you have measured a gap between LoRA (the default quality preset: rank 256, all linear layers) and full fine-tuning on your own data.

Otherwise stay with LoRA / QLoRA. It is faster, it forgets less, and QLoRA reaches 32B on one 32 GB card.

## Python API

```python
from backpropagate import Trainer

# Default: LoRA, rank 256, all linear layers.
trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)

# Full fine-tuning on the GPU, within the card-aware ceiling.
trainer = Trainer("smollm3-3b", mode="full")
trainer.train("my_data.jsonl", steps=100)

# 7B-class full fine-tuning with CPU offload (Linux / WSL2).
trainer = Trainer("Qwen/Qwen2.5-7B-Instruct", mode="full", full_ft_offload=True)
trainer.train("my_data.jsonl", steps=100)

# Override the ceiling (and the offload fit check) explicitly.
trainer = Trainer("Qwen/Qwen2.5-7B-Instruct", mode="full", full_ft_ceiling_billions=8.0)
```

## CLI

```bash
# Full fine-tuning on the GPU:
backprop train --model smollm3-3b --mode full --data my_data.jsonl --steps 100

# 7B-class full fine-tuning with CPU offload (Linux / WSL2):
backprop train --model Qwen/Qwen2.5-7B-Instruct --mode full --full-ft-offload \
  --data my_data.jsonl --steps 100

# Override the ceiling:
backprop train --model Qwen/Qwen2.5-7B-Instruct --mode full --full-ft-ceiling-billions 8 \
  --data my_data.jsonl --steps 100
```

## Full fine-tuning on the GPU

With `mode="full"` and no offload, the trainer:

1. Skips the adapter entirely. Every weight trains, in bf16; there is no 4-bit base.
2. Turns on gradient checkpointing, trading recomputation for activation memory.
3. Uses `paged_adamw_8bit`, so the optimizer state costs about 2 bytes per parameter.
4. Divides the learning rate by 10 (LoRA default `2e-4` → full fine-tuning default `2e-5`). Override with `learning_rate=`.

Weights (2 B/param) + gradients (2 B/param) + 8-bit optimizer state (~2 B/param) come to about 6 bytes per parameter on the card, plus activations. Against detected VRAM that gives the ceiling:

| Card | Ceiling on the GPU | Measured |
|---|---|---|
| 16 GB | 4B | — |
| 24 GB | 5B | — |
| 32 GB | 6B | 1.5B: 6.7 GB. 3B: 12.6 GB (batch 4, 512 tokens, about 0.63 s/step including startup) |
| 48 GB+ | 10B | — |

The ceiling bounds the parameter **count**. It does not promise a fit at every sequence length. It is checked when the `Trainer` is created (from the preset table or model id) and again after loading (from the actual parameter count). A model over the ceiling exits `2` with `RUNTIME_FULL_FT_MODEL_TOO_LARGE`; the error names `--full-ft-offload` when offload would fit it, and LoRA / QLoRA when it would not.

## Full fine-tuning with CPU offload (`--full-ft-offload`)

The offload engine keeps each weight and its gradient in host RAM in bf16, and streams one transformer layer at a time to the GPU. It shards the model with PyTorch's FSDP2 directly and runs its own training loop.

### Measured on an RTX 5090

| Model | Host RAM peak | VRAM (allocated / reserved) | Speed |
|---|---|---|---|
| Qwen2.5-7B (7.6B) | 30.8 GiB training, 32.2 GiB with save + reload | 5.3 / 14.7 GiB | 14.7 s/step |
| Qwen3-4B | 25.1 GiB with save + reload | — | 4.3 s/step |
| SmolLM3-3B | — | 4.3 GiB at batch 4 | 5.1 s/step |
| Qwen2.5-1.5B | — | — | 1.8 s/step |

All at 512 tokens. At 2048 tokens the 7.6B step was still 14.7 s: the run is limited by moving weights over PCIe, not by compute.

With VRAM capped on the same card, a 3B model trained under a 6 GiB cap, and 4B and 7.6B models under an 8 GiB cap. Those are emulated caps, not runs on real 8 GB hardware.

### What it costs

- **Speed.** About 8× slower than training on the GPU at 3B (5.1 against 0.63 s/step). Use it only when the model does not fit without it.
- **Optimizer: Adafactor, not AdamW.** It keeps no momentum and factors the second moment, so its state is a few MB even at 7B. The published evidence for Adafactor on LLM fine-tuning is thinner than for AdamW.
- **No fp32 copy of the weights.** Weights stay in bf16, and each update is written back with **stochastic rounding**. Round-to-nearest drops most small updates at a full fine-tuning learning rate: in our runs only about 17% of each intended update survived it, and the run stopped learning. Stochastic rounding keeps all of it on average. The engine measures the surviving fraction on every step and records it with the run.
- **Quality.** On one 3B run (Dolly-15k, 400 training examples, 150 held-out, 150 steps, one seed), held-out loss went from 2.45 to 1.93 with offload and to 1.84 with ordinary full fine-tuning on the GPU: about 85% of the improvement. The two paths use different optimizers, so this compares recipes, not just precision. One seed is not a benchmark.
- **Scope.** Plain supervised fine-tuning over the whole sequence. No packing, no response-only masking, no intermediate checkpoints (it saves at the end), no resume. `method="sft"` only.
- **Linux or WSL2 only.** FSDP2 needs NCCL, which Windows-native PyTorch does not have. On Windows-native it stops with `DEP_FSDP_UNAVAILABLE` before loading the model.

### The fit check

Before loading any weights, the trainer works out what the run needs and compares it with what the machine has:

- **Host RAM:** about **3.73 GiB per billion parameters + 10.1 GiB**, which covers training plus the save and reload at the end. It is compared with the RAM available when the run starts. Under WSL2 that is the VM's memory cap, not the machine's.
- **VRAM:** the largest layer and embedding in bf16, plus about 1.4 MiB per token of batch × sequence length, plus a margin.

If either does not fit, the run stops with `RUNTIME_FULL_FT_MODEL_TOO_LARGE`, showing required against available for both and the ways out: a smaller model, more RAM (or a higher WSL2 memory cap), a shorter sequence, or LoRA.

Worked examples from the formula:

| Model | Host RAM needed |
|---|---|
| 1.5B | ~16 GiB |
| 4B | ~25 GiB |
| 7.6B | ~39 GiB |
| 13B | ~59 GiB |

The constants come from four runs between 1.5B and 7.6B. Above 7.6B the formula is an extrapolation; nothing larger has been run. Under WSL2's default memory cap, raise it in `%UserProfile%\.wslconfig` (`[wsl2]` → `memory=`) and restart WSL with `wsl --shutdown`. A 28 GB cap holds about 4.8B.

### Environment variables

- `BACKPROPAGATE_OFFLOAD_PIN` — how host memory is page-locked (`register`, the default; `pinned`; `none`). `pinned` was 3–5× faster in our runs but rounds every block up to a power of two, which pushed a 7B run past 60 GiB. See [environment variables](/backpropagate/handbook/env-vars/).
- `BACKPROPAGATE_OFFLOAD_ROUNDING` — diagnostic only; leave it at `stochastic`.

### Not yet tested

Long runs, gradient accumulation above 1, a physical 64 GB machine (the test machine had more RAM, with a 60 GiB limit enforced by the test), and anything above 7.6B.

## LoRA or full fine-tuning: the evidence

- **[Biderman et al. 2024, "LoRA Learns Less and Forgets Less"](https://arxiv.org/abs/2405.09673).** In standard low-rank settings, LoRA substantially underperforms full fine-tuning on programming and mathematics. It forgets less of the base model's abilities outside the target domain, more than weight decay or dropout do. Full fine-tuning learns perturbations of 10–100× higher rank than typical LoRA.
- **[Thinking Machines 2025, "LoRA Without Regret"](https://thinkingmachines.ai/blog/lora/).** LoRA matches full fine-tuning when it is applied to all layers (especially MLP) and the dataset fits within its capacity; past that capacity it underperforms. Attention-only LoRA underperforms clearly. LoRA takes a little over two-thirds of the compute of full fine-tuning per pass, and its best learning rate is about 10× full fine-tuning's.

Backpropagate's default LoRA preset (rank 256, all linear layers) follows the conditions in the second paper. Whether it closes the gap on your task is something to measure — `backprop eval` compares two runs on held-out data.

## See also

- [Error codes → `RUNTIME_FULL_FT_MODEL_TOO_LARGE`](/backpropagate/handbook/error-codes/#runtime_full_ft_model_too_large) · [`DEP_FSDP_UNAVAILABLE`](/backpropagate/handbook/error-codes/#dep_fsdp_unavailable)
- [CLI reference → `backprop train`](/backpropagate/handbook/cli-reference/#backprop-train) — `--mode`, `--full-ft-offload`, `--full-ft-ceiling-billions`.
- [Estimate VRAM](/backpropagate/handbook/estimate-vram/)
