---
title: Export
description: GGUF export, Ollama registration, and quantization options.
sidebar:
  order: 3
---

## Export to GGUF

```python
result = trainer.export("gguf", quantization="q4_k_m")
```

To register the exported model with Ollama:

```python
from backpropagate import register_with_ollama

register_with_ollama(result.path, "my-finetuned-model")
```

Then use it locally:

```bash
ollama run my-finetuned-model
```

## CLI export

```bash
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
```

## Quantization options

| Quantization | Size | Quality | Use case |
|-------------|------|---------|----------|
| `q2_k` | Smallest | Lower | Embedded, constrained environments |
| `q4_0` | Small | Fair | Fast inference, lower quality |
| `q4_k_m` | Small | Good | General use (recommended) |
| `q5_k_m` | Medium | Better | Balance of size and quality |
| `q8_0` | Large | High | When quality matters more than size |
| `f16` | Largest | Highest | Maximum quality, no compression |

## Export formats

Backpropagate supports three export formats via `trainer.export(format=...)`:

| Format | Description | Use case |
|--------|-------------|----------|
| `lora` | LoRA adapter only (default) | Smallest output, requires base model at inference |
| `merged` | Base model + adapter merged | Standalone model, larger but self-contained |
| `gguf` | Quantized GGUF file | For Ollama, llama.cpp, and LM Studio |

## What GGUF export does

1. Merges LoRA weights back into the base model
2. Converts to GGUF format (via Unsloth if available, otherwise llama.cpp)
3. Applies the chosen quantization level
4. Optionally creates an Ollama Modelfile and registers the model

## Custom Modelfile

Use `create_modelfile()` to build an Ollama Modelfile with a custom system prompt, temperature, or context length before registering:

```python
from backpropagate import create_modelfile, register_with_ollama

modelfile_path = create_modelfile(
    "output/gguf/model-q4_k_m.gguf",
    system_prompt="You are a helpful coding assistant.",
    temperature=0.5,
    context_length=8192,
)
```

If you only need the default Modelfile, `register_with_ollama()` creates one automatically.

## Listing Ollama models

After registering one or more fine-tuned models, list them from Python:

```python
from backpropagate import list_ollama_models

for model in list_ollama_models():
    print(model)
```

This calls `ollama list` under the hood and returns the model names.

## Model cards (v1.1.0)

Every export now writes a `model_card.md` alongside the artifact. The card follows the [Hugging Face model card schema](https://huggingface.co/docs/hub/model-cards), so when you push to the Hub it's picked up as the repo's landing page automatically.

The card includes:

- Frontmatter (`base_model`, `library_name: backpropagate`, `tags`)
- Property table (run_id, dataset, sha256, steps, final loss, LoRA rank/alpha, seed, duration, GPU, library version)
- Loss curve (unicode sparkline)
- Trust signals (Stage B/C/D + Ship Gate)
- Reproduce-this-run command

Disable card emission with `backprop export ... --no-model-card`.

## Hub push (v1.1.0)

Backpropagate ships first-class Hugging Face Hub push from the CLI:

```bash
# adapter-only push (default — smaller, faster, more useful for LoRA finetunes)
backprop push ./output/lora --repo alice/qwen-finetune

# private repo
backprop push ./output/lora --repo alice/qwen-finetune --private

# include the base model
backprop push ./output/merged --repo alice/qwen-finetune --include-base

# one-shot export + push
backprop export ./output/lora --format lora --push-to-hub alice/qwen-finetune
```

Token resolution order: `--token` flag → `HF_TOKEN` env var → `HUGGING_FACE_HUB_TOKEN` env var → `~/.cache/huggingface/token` (from `huggingface-cli login`).

The `model_card.md` next to the local export is mirrored as `README.md` inside the upload so the HF UI renders it as the repo's model card. Errors carry structured codes (`HUB_PUSH_AUTH` / `HUB_PUSH_NOT_FOUND` / `HUB_PUSH_NETWORK` / `HUB_PUSH_VERSION` / `HUB_PUSH_UNKNOWN`) for programmatic triage.
