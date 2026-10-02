---
title: VRAM estimator
description: Estimate or measure the GPU memory a training config needs — backprop estimate-vram, Trainer.estimate_vram() and the web UI's Fits / Tight / Won't fit.
sidebar:
  order: 6.7
---

The VRAM estimator answers "will this config run out of memory on my card?" before the trainer downloads the model and runs to the first out-of-memory error. Three surfaces share one calculation:

- **`backprop estimate-vram`** (CLI): the batch-size tier table for your card, and a per-config estimate when you pass `--lora-r` / `--batch-size` / `--mode`.
- **`Trainer.estimate_vram(...)`** (Python): a structured `VRAMEstimate` with the total and a breakdown.
- **The web UI**: "Fits", "Tight" or "Won't fit" next to **Start training**, for the settings on the form.

Since 1.8.2 you can also **measure** instead of estimate: `backprop estimate-vram <model> --calibrate`, or **Measure on this GPU** in the UI. See [Measure on your own GPU](#measure-on-your-own-gpu).

## How accurate it is

The formula is fitted to peak memory measured on real training runs (RTX 5090, torch 2.10, transformers 5.5, Unsloth 2026.5, Windows). Estimate against measurement, before the 8% allocator margin the estimate adds on top:

| Run (QLoRA unless noted) | Measured peak | Formula |
|---|---|---|
| Llama 3.2 1B, rank 16, batch 4 x 2,048 tokens | 9.59 GiB | 9.69 (+1%) |
| Llama 3.2 1B, rank 16, batch 8 x 2,048 | 18.09 GiB | 18.23 (+1%) |
| Llama 3.2 1B, rank 256 all-linear, batch 4 x 2,048 | 11.32 GiB | 11.40 (+1%) |
| Llama 3.2 3B, rank 16, batch 4 x 2,048 | 9.07 GiB | 9.44 (+4%) |
| Qwen2.5 7B, rank 16, batch 2 x 2,048 | 10.69 GiB | 10.31 (-4%) |
| Qwen2.5 7B, rank 256 all-linear, batch 2 x 2,048 | 16.78 GiB | 16.46 (-2%) |
| Qwen2.5 14B, rank 32, batch 1 x 4,096 | 25.0 GiB | 23.5 (-6%) |
| Mistral-Small 24B, rank 32, batch 1 x 4,096 | 26.5 GiB | 27.8 (+5%) |
| Qwen2.5 32B, rank 32, batch 1 x 2,048 | 28.8 GiB | 28.7 (0%) |
| SmolLM3 3B, full fine-tune, batch 4 x 512 | 22.0 GiB (system-wide) | 22.0 |

The three large QLoRA rows (14B, 24B, 32B) were measured before the formula was written and were not used to fit it. The full fine-tune row sets the full fine-tune constant, so it matches by construction. Before 1.8.2 the estimator was 3 to 10 times low on small configs and 25-45% low on the large ones.

**It is still an estimate.** A different GPU, driver, PyTorch build or attention kernel shifts the numbers. Measure when it matters.

## What the estimate counts

| Part | Size |
|---|---|
| Weights, 16-bit | 2 bytes per parameter |
| Weights, 4-bit | the embedding table stays 16-bit; the rest costs about 0.7 bytes per parameter (Unsloth's dynamic 4-bit builds keep some layers in 16-bit) |
| LoRA adapter | 4 bytes per trainable parameter, plus about 6.3 while training (gradients and optimizer state). The count follows your rank and target modules. |
| Full fine-tune | 2 bytes per parameter of weights, plus about 5.3 for gradients and the paged 8-bit optimizer |
| A floor every run pays | one temporary full-precision copy of the embedding table: 0.98 GiB on Llama 3.2 1B, 2.03 GiB on Qwen2.5 7B |
| Each row in the batch | attention scores, `16 x heads x tokens^2` bytes, plus activations. This is the term that grows fastest: doubling the row length quadruples it. |
| Margin | 8% for allocator slack |

Two things follow from the row term:

- **On small models, batch size and row length cost far more than the model itself.** Llama 3.2 1B loads in about 1 GiB and needs 18 GiB at batch 8 with 2,048-token rows.
- **The `tokens^2` term applies when attention runs through PyTorch's built-in kernel**, which is every Windows install (flash-attention has no Windows build, and xFormers is disabled on RTX 40/50). With flash-attention or xFormers the estimate drops that term.

For full fine-tuning the total is system-wide. PyTorch's own counters (`torch.cuda.max_memory_allocated`) read about 40% lower, because the 8-bit optimizer keeps its state in memory they do not count.

## Measure on your own GPU

```bash
backprop estimate-vram Qwen/Qwen2.5-7B-Instruct --calibrate
```

This loads the model and runs up to three very short real training probes (a minute or two for a small model, longer for 7B), then stores what that model costs on your card, driver and library versions. Later estimates for that model on that GPU come from the measurement: the CLI shows `Source: measured on this GPU`, `--json` carries `"source": "measured"`, and the UI bar reads "VRAM · measured on this GPU".

On the RTX 5090, after calibrating Llama 3.2 1B and Qwen2.5 7B, predictions were within 0.4% of eight real runs the probes never ran (batch 8, 4,096-token rows, rank 256).

**It is built not to hurt the machine:**

- The probe process caps its own GPU memory below what is free. On Windows a run that needs slightly more VRAM than the card has does not fail: the driver spills into system memory and the whole desktop stutters. Capped, it fails cleanly instead.
- Each probe runs only if it is predicted to fit.
- If no informative probe fits (a big model on a small card), the load size is still measured and the per-row cost stays the formula's.

Use `--mode full` or `--no-4bit` to measure those modes; each is stored separately. One measurement covers every LoRA rank and target-module choice, because the adapter is an exact parameter count.

Measurements live in `~/.backpropagate/vram-calibration.json` (`BACKPROPAGATE_VRAM_CALIBRATION` moves it), one entry per GPU, model, mode and library versions. A new card or a PyTorch upgrade starts clean. `--no-calibration` ignores a stored measurement, and so does `--vram-gb`, since a measurement belongs to the GPU it was made on. If nothing can be measured the command exits `2` with `RUNTIME_VRAM_CALIBRATION_FAILED`.

## CLI: `backprop estimate-vram`

```bash
# The batch-size tier for the local GPU
backprop estimate-vram

# A per-config estimate
backprop estimate-vram Qwen/Qwen2.5-7B-Instruct --lora-r 256 --batch-size 2

# A smaller adapter on a 16-bit base
backprop estimate-vram Qwen/Qwen2.5-7B-Instruct --lora-r 16 --target-modules q_proj,v_proj --batch-size 2 --no-4bit

# Simulate a card you do not have (uses the formula, never a measurement)
backprop estimate-vram Qwen/Qwen2.5-7B-Instruct --lora-r 256 --batch-size 1 --vram-gb 16

# Machine-readable
backprop estimate-vram --vram-gb 16 --json | jq .recommended_batch_size

# 32 GB card, 7B full fine-tuning via FSDP2 CPU-offload: see host_ram_gb
backprop estimate-vram Qwen/Qwen2.5-7B-Instruct --vram-gb 32 --mode full --full-ft-offload --json
```

When the model's `config.json` is in your Hugging Face cache or a local folder, the estimate uses the model's real shape (it is never downloaded for this). Otherwise it uses a typical shape for the size in the model's name.

See [CLI reference → `backprop estimate-vram`](/backpropagate/handbook/cli-reference/#backprop-estimate-vram-v13) for every flag.

## Python API: `Trainer.estimate_vram()`

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")

estimate = trainer.estimate_vram(
    mode="lora",
    lora_r=256,
    batch_size=2,
    max_seq_length=2048,
)
print(estimate.summary())
# VRAM estimate (lora, 7.6B params, batch=2, seq=2048): total=17.8GB
# (weights=6.3 + lora=2.4 + optim=3.8 + activations=4.0 + kv=0.0 + overhead=1.3)

print(f"Fits on a 16 GB card: {estimate.fits_on_card(16.0)}")
# Fits on a 16 GB card: False
```

### `VRAMEstimate` fields

| Field | Description |
|-------|-------------|
| `total_gb` | The headline number. Compare it with your card's VRAM. |
| `source` | `"measured"` when this model was calibrated on this GPU, otherwise `"estimate"`. |
| `model_weights_gb` | The loaded base model (the measured load size when calibrated). |
| `lora_adapter_gb` | LoRA adapter weights, fp32 (0 when `mode="full"`). |
| `optimizer_state_gb` | Gradients and optimizer state for the trainable parameters. |
| `activations_gb` | The floor or the batch's rows, whichever is larger. |
| `kv_cache_gb` | Always 0; kept so the breakdown's shape does not change. |
| `overhead_gb` | The allocator margin (8% by default). |
| `param_count_billions` | The model size used. |
| `mode` / `batch_size` / `gradient_accumulation` / `max_seq_length` / `lora_r` | The inputs. |
| `notes: list[str]` | What the estimate assumed, or where the measurement came from. |

```python
estimate.fits_on_card(vram_gb)  # bool: total_gb <= vram_gb
estimate.summary()              # one-line summary
```

## Sample estimates

All with full 2,048-token rows and PyTorch's built-in attention (Windows). Shorter data uses less.

| Model | Config | Estimated total | 16 GB card | 24 GB card |
|-------|--------|-----------------|------------|------------|
| Llama 3.2 1B | QLoRA rank 64, batch 4 | 10.9 GB | fits | fits |
| Llama 3.2 3B | QLoRA rank 128, batch 2 | 8.5 GB | fits | fits |
| Qwen2.5 7B | QLoRA rank 16 on q and v, batch 2 | 11.1 GB | fits | fits |
| Qwen2.5 7B | QLoRA rank 256 all-linear, batch 1 | 15.7 GB | tight | fits |
| Qwen2.5 7B | QLoRA rank 256 all-linear, batch 2 | 17.8 GB | does not fit | fits |
| Qwen2.5 7B | QLoRA rank 64 all-linear, batch 4 | 17.1 GB | does not fit | fits |
| SmolLM3 3B | full fine-tune, batch 2 | 25.0 GB | does not fit | does not fit |
| Phi-4-mini 3.8B | full fine-tune, batch 1 | 30.6 GB | does not fit | does not fit |

**On a 16 GB card with the default settings** (Qwen2.5 7B, rank 256 on every linear layer), the automatic batch size is 2, which needs about 17.8 GB at full-length rows. The trainer's out-of-memory recovery then halves the batch to 1 and continues. To avoid that wasted attempt, pass `--batch-size 1`, or use `--lora-preset fast` (rank 16), or shorten `--max-seq-length`.

On a **32 GB** card (RTX 5090), **measured** peaks, batch 1 at each preset's full context window:

| Model | Config | GPU (allocated / reserved) | Host RAM |
|-------|--------|-----------|--------------------|
| Qwen2.5 14B | QLoRA rank 32, 4,096 tokens | 25.0 / 28.1 GiB | — |
| Mistral-Small 24B | QLoRA rank 32, 4,096 tokens | 26.5 / 29.6 GiB | — |
| Qwen2.5 32B | QLoRA rank 32, 2,048 tokens | 28.8 / 30.7 GiB (just fits) | — |
| Qwen2.5 7B | `mode="full"` on the GPU (7.6B > 6B ceiling) | refused → use `--full-ft-offload` | — |
| Qwen2.5 7B | `mode="full" --full-ft-offload`, 512 tokens | 5.3 / 14.7 GiB | 30.8 GiB training, 32.2 GiB with save |

For the offload path, `estimate_vram(offload=True)` uses the same measured constants as the trainer's fit check and reports `host_ram_gb` (about 39 GB for 7.6B, which includes the save). See [full fine-tuning](/backpropagate/handbook/full-fine-tuning/#the-fit-check).

The automatic batch size is 6 at 32 GB and 8 at 48 GB. It comes from a table by card size and does not look at the model, so it is too high for 14B and larger models at long rows: set `--batch-size` yourself for those, guided by the estimate.

## Limitations

- **Fitted on one stack.** One GPU family, one PyTorch and transformers version, Windows. Other architectures (non-standard attention, very large vocabularies) and other kernels can differ. `--calibrate` removes this uncertainty for a given model and machine.
- **Full fine-tuning is fitted on two runs.** Treat it as a guide and measure.
- **The preference methods** (ORPO, SimPO, KTO) process two sequences per example; the estimate does not model that yet.
- **Saving and merging** at the end of a run, and evaluation passes, are not modelled.
- **Other programs share the card.** The estimate is what training needs; your desktop, browser and other GPU programs come on top. The UI compares against the whole card, so leave headroom.

If a config that was estimated to fit still runs out of memory, the [`oom_recovery=True`](/backpropagate/handbook/training/#graceful-degradation-knobs) default halves the batch and retries up to 3 times before raising `RUNTIME_GPU_OOM`.

## See also

- [CLI reference → `backprop estimate-vram`](/backpropagate/handbook/cli-reference/#backprop-estimate-vram-v13)
- [Training → VRAM-aware batch sizing](/backpropagate/handbook/training/#vram-aware-batch-sizing)
- [Error codes → `RUNTIME_GPU_OOM`](/backpropagate/handbook/error-codes/#runtime_): the runtime contract when a config that was estimated to fit still runs out of memory.
- [Troubleshooting (CUDA)](/backpropagate/handbook/troubleshooting-cuda/): out-of-memory diagnosis.
