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
