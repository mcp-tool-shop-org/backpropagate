---
title: Export
description: GGUF export, Ollama registration, and quantization options.
sidebar:
  order: 3
---

## Export to GGUF

```python
trainer.export(
    format="gguf",
    quantization="q4_k_m",
    register_ollama=True,
    model_name="my-finetuned-model",
)
```

After export, use it locally:

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
| `q4_k_m` | Small | Good | General use (recommended) |
| `q8_0` | Medium | High | When quality matters more than size |
| `f16` | Largest | Highest | Maximum quality, no compression |

## What export does

1. Merges LoRA weights back into the base model
2. Converts to GGUF format using llama-cpp-python
3. Applies the chosen quantization level
4. Optionally creates an Ollama Modelfile and registers the model
