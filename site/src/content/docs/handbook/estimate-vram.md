---
title: VRAM estimator
description: Pre-flight VRAM estimation for training configs — Trainer.estimate_vram() and the backprop estimate-vram CLI.
sidebar:
  order: 6.7
---

The VRAM estimator answers the operator question "will this config OOM on my card?" BEFORE the trainer pulls the model, loads the dataset, and runs to the first OOM. Two surfaces share the same back-of-envelope math:

- **`Trainer.estimate_vram(...)`** (v1.4 Python API) — returns a structured `VRAMEstimate` with the headline `total_gb` and a per-consumer breakdown.
- **`backprop estimate-vram`** (v1.3 CLI) — prints a tier table mapping VRAM ranges to recommended batch sizes for the auto-detection path.

Both surfaces are accurate within ~10-20% of empirical peak for well-known training configs. The estimator is a planning tool, not a guarantee — see [the limitations](#limitations) below.

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
# VRAM estimate (lora, 7.0B params, batch=2, seq=2048): total=13.4GB
# (weights=3.3 + lora=0.5 + optim=0.0 + activations=4.1 + kv=3.7 + overhead=1.7)

print(f"Fits on 16GB card: {estimate.fits_on_card(16.0)}")
# Fits on 16GB card: True
```

### `VRAMEstimate` fields

The returned dataclass carries the headline number, per-consumer breakdown, and the inputs that produced it:

| Field | Description |
|-------|-------------|
| `total_gb` | Headline number. Compare against your card's VRAM. |
| `model_weights_gb` | Base model in nf4 (0.5 bytes/param when `quantize_base=True`, default). |
| `lora_adapter_gb` | LoRA adapter weights in bf16 (0 when `mode="full"`). |
| `optimizer_state_gb` | paged 8-bit Adam state for the trainable parameters. |
| `activations_gb` | Forward + backward activations. `mode="full"` assumes `gradient_checkpointing=True` (sqrt(L) scaling). |
| `kv_cache_gb` | Transient KV cache during forward pass (training amortization factor 0.25). |
| `overhead_gb` | 15% margin for fragmentation + framework overhead. |
| `param_count_billions` | Estimated model size used in the math. |
| `mode` / `batch_size` / `gradient_accumulation` / `max_seq_length` / `lora_r` | Reproducibility inputs. |
| `notes: list[str]` | Diagnostic notes (e.g. "base model quantized to nf4", or "param_count_billions not provided and could not be estimated"). |

### Helper methods

```python
estimate.fits_on_card(vram_gb)  # bool — does total_gb <= vram_gb?
estimate.summary()              # str — operator-readable one-line summary
```

## CLI: `backprop estimate-vram`

The CLI surfaces the tier heuristic that `Trainer._detect_batch_size` uses when `--batch-size auto` is set. It's lighter-weight than the Python API (no Trainer construction, no torch / transformers import) — useful for sizing infra spend or previewing the auto-batch decision before kicking off training.

```bash
# Uses the local GPU's VRAM
backprop estimate-vram

# Override the model (model is used only for the printed header)
backprop estimate-vram Qwen/Qwen2.5-7B-Instruct

# Simulate a card you don't currently have
backprop estimate-vram --vram-gb 24

# Machine-readable output for CI consumers
backprop estimate-vram --vram-gb 16 --json | jq .recommended_batch_size

# 32 GB card, 7B full fine-tuning via FSDP2 CPU-offload — see host_ram_gb
backprop estimate-vram Qwen/Qwen2.5-7B-Instruct --vram-gb 32 --mode full --full-ft-offload --json
```

See [CLI reference → `backprop estimate-vram`](/backpropagate/handbook/cli-reference/#backprop-estimate-vram-v13) for the full flag table.

## Sample estimates

A few canonical configurations on 16GB consumer cards (RTX 4080 / 5080 / 4070 Ti Super):

| Model | Config | Estimated total | Fits 16GB? |
|-------|--------|-----------------|------------|
| Qwen 2.5 7B | LoRA r=256, batch=2, seq=2048 | ~13.4 GB | Yes |
| Qwen 2.5 7B | LoRA r=64, batch=4, seq=2048 | ~14.1 GB | Yes (tight) |
| Phi-4-mini-3.8B | `mode="full"`, batch=1, seq=2048 | ~12.8 GB | Yes |
| SmolLM3-3B | `mode="full"`, batch=2, seq=2048 | ~11.2 GB | Yes |
| Qwen 2.5 7B | `mode="full"` (raises before estimate) | n/a | Refused — `RUNTIME_FULL_FT_MODEL_TOO_LARGE` |

For the last row, the trainer's mode='full' gate fires at `Trainer.__init__` before the estimator gets a chance — see [full fine-tuning](/backpropagate/handbook/full-fine-tuning/).

On a **32 GB** card (RTX 5090) the envelope opens up — and the offload path reports `host_ram_gb`:

| Model | Config | GPU total | Host RAM (offload) |
|-------|--------|-----------|--------------------|
| Qwen 2.5 14B | QLoRA r=32, batch=2, seq=4096 | ~8.5 GB | — |
| Qwen 2.5 32B | QLoRA r=32, seq=2048 | ~26 GB (just fits) | — |
| Qwen 2.5 7B | `mode="full"` (pure-GPU, 7B > 6B ceiling) | refused → use `--full-ft-offload` | — |
| Qwen 2.5 7B | `mode="full" --full-ft-offload` | ~1 GB working set + activations | ~39 GB |

The recommended auto-batch is **6 at 32 GB** and **8 at 48 GB** (`backprop estimate-vram --vram-gb 32`).

## Limitations

The estimator is a planning tool, not a guarantee against OOM. Caveats:

- **Activation memory is the most variable component.** Real-world activation peaks depend on the model architecture, the dataset's sequence-length distribution, attention backend (Flash Attention 2 / 3 vs xFormers vs SDPA), and the gradient-checkpointing schedule. The 15% overhead margin absorbs typical variance; pathological inputs (very long sequences, exotic architectures) can exceed it.
- **The KV cache amortization factor (0.25) is a heuristic.** Training rarely keeps the full KV cache (it's primarily an inference cost), but transformers libraries allocate it transiently during forward. The constant approximates that share.
- **Pretrained-model defaults (hidden_dim, num_layers, num_heads) are 7B-class.** Pass explicit values to the Python API for non-Llama-architecture models.
- **The estimator does not account for OS-level GPU usage.** Your desktop compositor, browser, or background ML processes share the same VRAM pool. Leave headroom.

If the estimator says a config fits and it OOMs in practice, the [`oom_recovery=True`](/backpropagate/handbook/training/#graceful-degradation-knobs) default will halve the batch and retry up to 3 times before raising `RUNTIME_GPU_OOM` — that's the runtime safety net beneath the planning surface.

## See also

- [CLI reference → `backprop estimate-vram`](/backpropagate/handbook/cli-reference/#backprop-estimate-vram-v13)
- [Training → VRAM-aware batch sizing](/backpropagate/handbook/training/#vram-aware-batch-sizing)
- [Error codes → `RUNTIME_GPU_OOM`](/backpropagate/handbook/error-codes/#runtime_) — the runtime-level OOM contract when an estimate-passing config still hits OOM.
- [Troubleshooting (CUDA)](/backpropagate/handbook/troubleshooting-cuda/) — deep-dive on OOM diagnosis when an estimate-passing config still hits OOM.
