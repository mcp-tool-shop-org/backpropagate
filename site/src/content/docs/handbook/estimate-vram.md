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

The formula is fitted to peak memory measured on real training runs (RTX 5090, torch 2.10, transformers 5.5, Unsloth 2026.5, Windows, 24 QLoRA and LoRA runs on 2026-10-02). The estimate includes its 8% allocator margin:

| Run (QLoRA, 2,048-token rows) | Measured peak | Estimate |
|---|---|---|
| Llama 3.2 1B, rank 16, batch 4 | 5.35 GiB | 5.97 (+12%) |
| Llama 3.2 1B, rank 16, batch 8 | 9.86 GiB | 10.70 (+9%) |
| Llama 3.2 1B, rank 256 all-linear, batch 4 | 7.35 GiB | 8.20 (+12%) |
| Llama 3.2 3B, rank 16, batch 4 | 6.69 GiB | 7.82 (+17%) |
| Qwen2.5 7B, rank 16, batch 2 | 9.31 GiB | 9.80 (+5%) |
| Qwen2.5 7B, rank 16, batch 4 | 11.91 GiB | 12.75 (+7%) |
| Qwen2.5 7B, rank 256 all-linear, batch 2 | 16.61 GiB | 17.79 (+7%) |
| Qwen2.5 3B, rank 16, batch 2 | 3.60 GiB | 5.38 (+49%) |
| SmolLM3 3B, rank 16, batch 2 | 3.02 GiB | 4.92 (+63%) |
| SmolLM2 135M, rank 16, batch 8 | 2.28 GiB | 3.67 (+61%) |
| SmolLM3 3B, full fine-tune, batch 4 x 512 tokens | 22.0 GiB (system-wide) | 23.8 (+8%) |

The estimate never read below a measured peak on the 24 runs. It reads 3 to 19% above on Llama 3.2 1B and 3B and Qwen2.5 7B, and 50 to 76% above on Qwen2.5 3B, SmolLM3 3B and SmolLM2 135M. On those three, Unsloth's compiled loss needs about half the memory for the output logits, and the formula prices the full figure so that it does not read low. [Measuring](#measure-on-your-own-gpu) gives the real number for the model in hand. The full fine-tune row sets the full fine-tune constant, so it matches by construction.

"Measured peak" is PyTorch's peak allocation. On a card with room to spare PyTorch also keeps freed blocks cached, so Task Manager or `nvidia-smi` can show 15 to 30% more than that while a run is going (12.6 GiB for the 9.86 GiB run above). That cache is given back when memory gets short.

**It is still an estimate.** A different GPU, driver, PyTorch build or attention kernel shifts the numbers. Measure when it matters.

## What the estimate counts

| Part | Size |
|---|---|
| Weights, 16-bit | 2 bytes per parameter |
| Weights, 4-bit | the embedding table stays 16-bit; the rest costs about 0.7 bytes per parameter (Unsloth's dynamic 4-bit builds keep some layers in 16-bit) |
| LoRA adapter | 4 bytes per trainable parameter, plus about 8.4 while training (gradients, optimizer state, adapter activations). The count follows your rank and target modules. |
| Full fine-tune | 2 bytes per parameter of weights, plus about 5.3 for gradients and the paged 8-bit optimizer |
| A floor every run pays | one temporary full-precision copy of the embedding table: 0.98 GiB on Llama 3.2 1B, 2.03 GiB on Qwen2.5 7B |
| Each token in the batch (rows x row length) | the loss's full-precision logits, `4 x vocabulary` bytes, plus `30 x hidden size` bytes of activations: about 0.56 MB per token on Llama 3.2 1B, 0.70 MB on Qwen2.5 7B |
| Gradient checkpointing off | every layer's activations are kept too: `layers x (20 x hidden size + 6 x MLP size)` bytes per token. On Llama 3.2 1B that is 3.5 times the per-token cost. |
| Margin | 8% for allocator slack |

Three things follow:

- **Memory grows in a straight line with batch size and with row length.** Twice the rows, or rows twice as long, cost twice the per-token part. There is no term that grows with the square of the row length.
- **On small models, the batch costs more than the model.** Llama 3.2 1B loads in about 1 GiB and needs about 10 GiB at batch 8 with 2,048-token rows.
- **A large adapter is a real cost on a 7B model.** Rank 256 on every linear layer of Qwen2.5 7B is 646 million trainable parameters: 2.4 GiB to hold and about 5 GiB more to train.

For full fine-tuning the total is system-wide. PyTorch's own counters (`torch.cuda.max_memory_allocated`) read about 40% lower, because the 8-bit optimizer keeps its state in memory they do not count.

### Why there is no `tokens^2` term

Without flash-attention or xFormers (every Windows install: flash-attention has no Windows build, and xFormers is disabled on RTX 40/50), attention runs through PyTorch's built-in kernel. Unsloth asked that kernel to handle grouped-query attention itself, and PyTorch's Windows builds can only do that by building the full `heads x tokens x tokens` score matrix in full precision: 2 GiB for a single 2,048-token row at 32 heads. Since 1.8.2 the trainer has Unsloth expand the key and value heads first, which lets PyTorch use its memory-efficient kernel: 0.03 GiB for the same row, same loss. Packed samples stay whole and cannot see each other.

## Measure on your own GPU

```bash
backprop estimate-vram Qwen/Qwen2.5-7B-Instruct --calibrate
```

This loads the model and runs up to three very short real training probes (a minute or two for a small model, longer for 7B), then stores what that model costs on your card, driver and library versions. Later estimates for that model on that GPU come from the measurement: the CLI shows `Source: measured on this GPU`, `--json` carries `"source": "measured"`, and the UI bar reads "VRAM · measured on this GPU".

On the RTX 5090, after calibrating Llama 3.2 1B and Qwen2.5 7B, the measured cost predicted eleven real runs the probes never ran (batch 1 to 8, 1,024 to 4,096-token rows, rank 256): ten within 0.6% of the real peak and one, batch 8, 3% under it, before the 6% margin a measured estimate adds.

The per-token cost that is stored is the **highest** any probe showed. Unsloth compiles its loss, and the compiled version needs less memory, but the first steps of a run can happen before it takes effect. The first probe behaves like those first steps, so it is the one a fresh run peaks like. The command prints how far the probes differed.

**It is built not to hurt the machine:**

- The probe process caps its own GPU memory below what is free, less a headroom of 1.5 GiB or 8% of the card, whichever is larger. On Windows a run that needs slightly more VRAM than the card has does not fail: the driver spills into system memory and the whole desktop stutters. Capped, it fails cleanly instead. With less than 1.5 GiB left after the headroom, nothing is run.
- The probes always train with the non-paged `adamw_8bit`. A paged optimizer keeps its state in memory the cap does not cover, and on Windows that memory can stall the desktop.
- Each probe runs only if it is predicted to fit, including the gradients and optimizer state of a full fine-tune.
- The probe process cannot outlive the command that started it: closing the terminal, Ctrl+C or a crash ends it too.
- If no informative probe fits (a big model on a small card), the load size is still measured and the per-row cost stays the formula's.

Use `--mode full` or `--no-4bit` to measure those modes; each is stored separately. One measurement covers every LoRA rank and target-module choice, because the adapter is an exact parameter count. For a full fine-tune the measurement stores the gradients and optimizer state as one fixed cost and prices the rows with the formula.

**When a measurement is not used.** The estimate falls back to the formula, and says so, for:

- rows more than twice as long as the longest row the probes ran (2,048 tokens today, so beyond 4,096);
- the preference methods (ORPO, SimPO, KTO), which process two sequences per example;
- gradient checkpointing off, and `--full-ft-offload`.

Measurements live in `~/.backpropagate/vram-calibration.json` (`BACKPROPAGATE_VRAM_CALIBRATION` moves it), one entry per GPU, model, mode and library versions. A new card or a PyTorch upgrade starts clean. A store that cannot be read is moved aside to `vram-calibration.json.unreadable`, never overwritten. `--no-calibration` ignores a stored measurement, and so does `--vram-gb`, since a measurement belongs to the GPU it was made on. If nothing can be measured the command exits `2` with `RUNTIME_VRAM_CALIBRATION_FAILED`.

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
# (weights=6.3 + lora=2.4 + optim=5.1 + activations=2.7 + kv=0.0 + overhead=1.3)

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

All with full 2,048-token rows. Shorter data uses less.

| Model | Config | Estimated total | 16 GB card | 24 GB card |
|-------|--------|-----------------|------------|------------|
| Llama 3.2 1B | QLoRA rank 64, batch 4 | 6.5 GB | fits | fits |
| Llama 3.2 3B | QLoRA rank 128, batch 2 | 7.7 GB | fits | fits |
| Qwen2.5 7B | QLoRA rank 16 on q and v, batch 2 | 9.8 GB | fits | fits |
| Qwen2.5 7B | QLoRA rank 64 all-linear, batch 2 | 11.7 GB | fits | fits |
| Qwen2.5 7B | QLoRA rank 64 all-linear, batch 4 | 14.7 GB | tight | fits |
| Qwen2.5 7B | QLoRA rank 256 all-linear, batch 1 | 17.0 GB | does not fit | fits |
| Qwen2.5 7B | QLoRA rank 256 all-linear, batch 4 | 20.7 GB | does not fit | tight |
| SmolLM3 3B | full fine-tune, batch 2 | 24.9 GB | does not fit | does not fit |
| Phi-4-mini 3.8B | full fine-tune, batch 1 | 30.6 GB | does not fit | does not fit |

**On a 16 GB card, rank 256 does not fit a 7B model**: on Qwen2.5 7B it needs about 17 GB even at batch 1, and halving the batch cannot fix that. So the default LoRA shape follows the GPU. With nothing set, the trainer uses the largest of three presets that fits within 85% of the memory that is free, and logs the choice when it is not `quality`:

| Preset | Shape | Qwen2.5 7B at batch 1 | Chosen for a 7B model when free memory is |
|---|---|---|---|
| `quality` | rank 256, every linear layer | 17.0 GB | 20 GB or more (24 GB and 32 GB cards) |
| `balanced` | rank 64, every linear layer | 11.0 GB | 13 to 20 GB (16 GB cards) |
| `fast` | rank 16, `q_proj` and `v_proj` | 9.0 GB | 10.6 to 13 GB (12 GB cards) |

Below about 10.6 GB free, no shape is estimated to fit a 7B model: the trainer uses `fast`, warns, and a smaller model or a shorter `--max-seq-length` is the fix.

A 3B model fits `quality` on a 16 GB card, so it keeps it. `--lora-preset quality` (or any explicit `--lora-r`) is always used as given.

**The automatic batch size is checked the same way.** It starts from the table by card size and is lowered to the largest batch whose estimate is within 90% of free memory. It is never raised above the table's value. With rank 64 on a 16 GB card a 7B model starts at batch 2 (11.7 GB).

On a **32 GB** card (RTX 5090), **measured** peaks, batch 1 at each preset's full context window (QLoRA rows re-measured 2026-10-03 on torch 2.8.0, transformers 5.18, trl 1.14; receipts in `docs/receipts/2026-10-03-presets/`):

| Model | Config | GPU (allocated / reserved) | Host RAM |
|-------|--------|-----------|--------------------|
| Qwen2.5 14B | QLoRA rank 32, 4,096 tokens | 18.7 / 20.0 GiB | — |
| Mistral-Small 24B | QLoRA rank 32, 4,096 tokens | 22.8 / 24.2 GiB | — |
| Qwen2.5 32B | QLoRA rank 32, 2,048 tokens | 26.0 / 27.2 GiB | — |
| Qwen2.5 7B | `mode="full"` on the GPU (7.6B > 6B ceiling) | refused → use `--full-ft-offload` | — |
| Qwen2.5 7B | `mode="full" --full-ft-offload`, 512 tokens | 5.3 / 14.7 GiB | 30.8 GiB training, 32.2 GiB with save |

Before the attention change described above, the same three runs peaked at 25.0, 26.5 and 28.8 GiB (2026-09-30), with the 32B just fitting. Most of that was the score matrix. The formula's estimates for the 14B and 24B rows, 17.3 GB and 23.9 GB, land within 8% of the new measurements.

For the offload path, `estimate_vram(offload=True)` uses the same measured constants as the trainer's fit check and reports `host_ram_gb` (about 39 GB for 7.6B, which includes the save). See [full fine-tuning](/backpropagate/handbook/full-fine-tuning/#the-fit-check).

The automatic batch size starts at 6 on a 32 GB card and 8 at 48 GB, and is lowered when the estimate for the model in hand does not fit (see above).

## Limitations

- **Fitted on one stack.** One GPU family, one PyTorch and transformers version, Windows. Other architectures (non-standard attention, very large vocabularies) and other kernels can differ. `--calibrate` removes this uncertainty for a given model and machine.
- **It reads high on some models** (about 50 to 75% on three of the six measured), because the loss's memory depends on whether Unsloth's compiled version is in effect. It is built not to read low.
- **Gradient checkpointing off** is fitted on three models and is never taken from a measurement.
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
