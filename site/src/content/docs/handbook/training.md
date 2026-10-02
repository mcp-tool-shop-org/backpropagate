---
title: Training
description: Basic training, multi-run SLAO, and model presets.
sidebar:
  order: 2
---

This page is the canonical reference for the training surface — every `Trainer(...)` parameter, every `multi_run` knob, every callback hook, every preset, every dataset format Backpropagate auto-detects. If you're just getting started, head to [Getting Started](/backpropagate/handbook/getting-started/) for the 3-line API and an end-to-end first run. If you have a specific operator goal in mind (resume after OOM, push to Hub, multi-GPU, diff two runs), see [Recipes](/backpropagate/handbook/recipes/) for paste-and-run snippets.

## Basic training

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
```

`Qwen/Qwen2.5-7B-Instruct` is the canonical default — what `Trainer()` resolves when called with no model argument. Older examples used the pre-quantized `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`; both work. Smart defaults automatically configure learning rate, batch size, gradient accumulation, and LoRA rank based on your hardware and dataset size.

## Multi-run SLAO training

SLAO — Single LoRA Continual Learning via Asymmetric Merging ([arXiv:2512.23017](https://arxiv.org/abs/2512.23017)) — prevents catastrophic forgetting during extended fine-tuning by merging LoRA adapters between runs using orthogonal initialization (QR-decomposition on A matrices), asymmetric A/B handling, and time-aware scaling `λ(i) = 1/√i`:

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")

result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
    merge_mode="slao",
)
```

CLI equivalent (the CLI flag is `--samples`; the matching Python-API field is `samples_per_run`):

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Trainer parameters

The `Trainer` constructor accepts optional overrides for all key hyperparameters. Anything you omit falls back to smart defaults (or environment-variable overrides via `BACKPROPAGATE_` prefix).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model name or local path |
| `lora_r` | `None` (the preset's rank) | LoRA rank (must be > 0). `None` takes it from `lora_preset`: 256, 64 or 16. |
| `lora_alpha` | `None` (the preset's alpha) | LoRA scaling factor. `None` takes it from `lora_preset`: 512, 128 or 32 (alpha = 2 x rank). |
| `lora_dropout` | 0.05 | LoRA dropout rate |
| `learning_rate` | 2e-4 | Learning rate |
| `batch_size` | `"auto"` | Per-device batch size (auto-detects from VRAM) |
| `gradient_accumulation` | 4 | Gradient accumulation steps |
| `max_seq_length` | 2048 | Maximum sequence length |
| `output_dir` | `./output` | Where to save checkpoints and exports |
| `use_unsloth` | `True` | Use Unsloth for 2x faster training (if installed) |
| `train_on_responses` | `True` (auto-disabled on Windows) | Compute loss only on assistant responses. See note below. |
| `oom_recovery` | `True` | B-002 graceful OOM handling: catch `torch.cuda.OutOfMemoryError`, halve batch size, clear cache, retry. Up to 3 retries at the minimum batch size. Set `False` to make OOM a hard failure. |
| `unsloth_fallback` | `True` | B-010 graceful degradation: if `use_unsloth=True` but Unsloth's loader raises (e.g. a broken nightly), fall back to plain `transformers` + `peft`. Set `False` to make Unsloth failures hard-fail. |
| `use_dora` | `False` | v1.3 BACKEND-1 — enable DoRA (Weight-Decomposed Low-Rank Adaptation). Rank-8 DoRA matches rank-32 LoRA quality (+2.8% on LLaMA-7B); merges to zero inference overhead. Requires `peft>=0.10`. |
| `packing` | `True` | v1.3 BACKEND-1 — sample packing (combine short sequences into single batches). Default ON gives 1.7-3× throughput on variable-length conversational datasets. Set `False` if you hit packing-incompatible behavior. |
| `init_lora_weights` | `"default"` | v1.3 BACKEND-1 — one of `"default"` / `"pissa"` / `"loftq"`. PiSSA + LoftQ recover quality lost during QLoRA quantization at zero runtime cost. |
| `optim` | `None` (auto) | v1.3 BACKEND-1 — optimizer string. `None` auto-picks `"paged_adamw_8bit"` on consumer GPUs (<24GB VRAM, per [arXiv:2509.12229](https://arxiv.org/abs/2509.12229) RTX 4060 study), `"adamw_torch_fused"` otherwise. Override with `"adamw_torch"` / `"paged_adamw_8bit"` / `"adamw_8bit"` etc. |
| `lora_preset` | `None` (auto) | One of `"auto"`, `"quality"`, `"balanced"`, `"fast"`. The LoRA shape. `quality` = rank 256, alpha 512, every linear layer. `balanced` = rank 64, alpha 128, every linear layer (a 7B model on a 16 GB card: about 11 GB at batch 1). `fast` = rank 16, alpha 32, `q_proj` and `v_proj` only. `auto` uses the largest of the three whose estimate fits within 85% of the GPU memory that is free, and says so in the log when it is not `quality`. Giving `lora_r`, `lora_alpha` or `target_modules` turns the automatic choice off. Per [Biderman 2024](https://arxiv.org/abs/2405.09673), rank 256 on every linear layer matches full fine-tuning on most post-training tasks. |
| `mode` | `"lora"` | **v1.4** — one of `"lora"` (the default — low-rank adapter) or `"full"` (full fine-tuning — every weight updated). `"full"` is supported only for models up to 4B parameters (genuine ~3B presets fit a 16GB card; the 3.8–4B class needs 24GB+); models >4B raise `FullFinetuneModelTooLargeError` (`RUNTIME_FULL_FT_MODEL_TOO_LARGE`). See [full fine-tuning](/backpropagate/handbook/full-fine-tuning/) for the LoRA-vs-full quality math + the recovery decision tree. |

### `train_on_responses` on Windows

The default is `True`, but Backpropagate automatically disables it on Windows because the underlying Unsloth helper uses multiprocessing in a way that hangs / crashes on Windows. Effectively:

- Linux: loss is computed only on assistant turns.
- Windows: loss is computed on the full conversation (user + assistant). Same model trains, but loss numbers and (slightly) the final quality differ from a Linux run on the same dataset.

If you need parity, run training in WSL or on a Linux host. There is no per-Trainer override for the Windows auto-disable — it is keyed off `os.name == "nt"`.

### Graceful degradation knobs

`oom_recovery` and `unsloth_fallback` are the two big "things will keep working when the environment misbehaves" knobs. Both default `True`. Operators triaging "why did my batch size silently halve?" or "why is the trainer using transformers when I asked for Unsloth?" should look at the startup log line `Degradation knobs: oom_recovery=... unsloth_fallback=...`, then either keep the defaults (preferred for production) or set the knob to `False` to force hard failures while you fix the underlying issue.

## Training callbacks

Hook into training events with `TrainingCallback`. All hooks are optional:

| Hook | Signature | When it fires |
|------|-----------|---------------|
| `on_step` | `(step: int, loss: float)` | After each training step |
| `on_epoch` | `(epoch: int)` | After each epoch |
| `on_save` | `(path: str)` | After a checkpoint save |
| `on_complete` | `(run: TrainingRun)` | When training finishes successfully |
| `on_error` | `(err: Exception)` | When training fails |

```python
from backpropagate import Trainer, TrainingCallback

callback = TrainingCallback(
    on_step=lambda step, loss: print(f"Step {step}: loss={loss:.4f}"),
    on_complete=lambda run: print(f"Done! Final loss: {run.final_loss:.4f}"),
    on_error=lambda err: print(f"Error: {err}"),
)

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("data.jsonl", steps=100, callback=callback)
```

## TrainingRun result

The `train()` method returns a `TrainingRun` dataclass with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Unique identifier for this run |
| `steps` | `int` | Number of training steps completed |
| `final_loss` | `float` | Loss at the end of training |
| `loss_history` | `list[float]` | Per-step loss values |
| `output_path` | `str` | Where the model was saved |
| `duration_seconds` | `float` | Wall-clock training time |
| `samples_seen` | `int` | Number of dataset samples processed |

## VRAM-aware batch sizing

When `batch_size="auto"` (the default), Backpropagate starts from a batch size for the size of your card (8 at 48 GB and up, 6 at 32 GB, 4 at 24 GB, 2 at 16 GB, 1 below that) and lowers it until the estimated memory for your model and settings fits in 90% of the memory free on the GPU. It never raises it. Gradient accumulation keeps the effective batch size up when the batch is small.

## Model presets

| Preset | GPU memory (fast / balanced / quality adapter) | Speed | Quality |
|--------|------|-------|---------|
| Qwen 2.5 7B | 9 / 11 / 17 GB | Medium | Best |
| Qwen 2.5 3B | 4 / 6 / 10 GB | Fast | Good |
| Llama 3.2 3B | 4 / 6 / 9 GB | Fast | Good |
| Llama 3.2 1B | 2 / 3 / 5 GB | Fastest | Basic |
| Mistral 7B | 6 / 8 / 14 GB | Medium | Good |

## Training methods

- **LoRA** -- Low-rank adaptation that trains small adapter matrices instead of the full model
- **QLoRA** -- 4-bit quantized LoRA for minimal VRAM usage (default when loading with `load_in_4bit=True`)
- **Unsloth** -- 2x faster training with 50% less VRAM when the `[unsloth]` extra is installed

## Dataset formats

Backpropagate accepts JSONL, CSV, or any HuggingFace dataset name. It auto-detects and converts between these chat formats:

- **ShareGPT**: `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`
- **Alpaca**: `{"instruction": "...", "input": "...", "output": "..."}`
- **OpenAI**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- **ChatML**: `{"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}`
- **Raw text**: Plain text, one document per line

You can also pass a HuggingFace `Dataset` object directly to `trainer.train()`.

## Advanced dataset features

The `backpropagate.datasets` module provides tools beyond basic loading:

- **Quality filtering** -- Remove low-quality samples by length, language, or custom criteria
- **Deduplication** -- Exact-match and MinHash near-duplicate removal
- **Perplexity filtering** -- Score samples by perplexity and filter outliers
- **Curriculum learning** -- Order samples by difficulty for progressive training
