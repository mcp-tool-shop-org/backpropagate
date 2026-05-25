---
title: Training
description: Basic training, multi-run SLAO, and model presets.
sidebar:
  order: 2
---

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
| `lora_r` | 256 | LoRA rank (must be > 0). v1.3 quality preset; pass --lora-preset=fast for v1.2.x rank-16. |
| `lora_alpha` | 512 | LoRA scaling factor (alpha = 2 * r convention; v1.3 quality preset. Pass `--lora-preset=fast` for the v1.2.x alpha=32). |
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

When `batch_size="auto"` (the default), Backpropagate queries your GPU VRAM and picks a safe batch size: 4 for 24GB+ cards, 2 for 16GB+, and 1 for smaller GPUs. Combined with gradient accumulation, this keeps effective batch size high without OOM.

## Model presets

| Preset | VRAM | Speed | Quality |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12GB | Medium | Best |
| Qwen 2.5 3B | ~8GB | Fast | Good |
| Llama 3.2 3B | ~8GB | Fast | Good |
| Llama 3.2 1B | ~6GB | Fastest | Basic |
| Mistral 7B | ~12GB | Medium | Good |

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
