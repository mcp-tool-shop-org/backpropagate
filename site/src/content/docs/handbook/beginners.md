---
title: Beginners Guide
description: New to LLM fine-tuning? Start here.
sidebar:
  order: 99
---

This guide walks you through your first fine-tuning run with Backpropagate, from zero to a working Ollama model. No prior experience with LoRA, GGUF, or training pipelines is required.

## 1. What is fine-tuning?

Large language models (LLMs) like Qwen and Llama are trained on broad internet text. Fine-tuning teaches an existing model new behavior using your own data -- customer support logs, code examples, domain-specific Q&A, or any conversational dataset. Instead of retraining billions of parameters from scratch, Backpropagate uses **LoRA** (Low-Rank Adaptation) to train a small set of adapter weights that modify the base model's behavior. This is fast, uses far less GPU memory, and produces results you can export and run locally with Ollama.

## 2. Prerequisites

Before you start, make sure you have:

- **Python 3.10 or newer** -- check with `python --version`
- **A CUDA GPU with 8GB+ VRAM** -- NVIDIA RTX 3060 or better. Check with `nvidia-smi`
- **PyTorch 2.0+** with CUDA support -- install from [pytorch.org](https://pytorch.org/)
- **Ollama** (optional) -- for running your exported model locally. Install from [ollama.com](https://ollama.com)

If you are on Windows, Backpropagate handles the common PyTorch/CUDA pitfalls automatically (multiprocessing crashes, xformers incompatibilities, dataloader issues).

## 3. Installation

Install Backpropagate with the recommended extras:

```bash
pip install backpropagate[standard]
```

This gives you the core library plus Unsloth (2x faster training) and the Gradio web UI. If you only want the Python API with no extras:

```bash
pip install backpropagate
```

Verify the install:

```bash
backprop info
```

This prints your Python version, GPU details, VRAM, and which optional features are available.

## 4. Prepare your dataset

Backpropagate accepts JSONL files with conversation data. The simplest format is OpenAI-style messages:

```json
{"messages": [{"role": "user", "content": "What is LoRA?"}, {"role": "assistant", "content": "LoRA stands for Low-Rank Adaptation..."}]}
{"messages": [{"role": "user", "content": "How do I export to GGUF?"}, {"role": "assistant", "content": "Use trainer.export('gguf')..."}]}
```

Save this as `my_data.jsonl`. Each line is one conversation. Aim for at least 100 examples for a meaningful fine-tune, though 500+ is better.

Backpropagate also auto-detects ShareGPT, Alpaca, and ChatML formats, so use whatever you have.

## 5. Train your first model

Three lines of Python:

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
```

What happens behind the scenes:

1. The model downloads from HuggingFace (first run only, cached afterward)
2. Backpropagate detects your GPU VRAM and picks a safe batch size
3. LoRA adapters are applied to the model's attention layers
4. Training runs for 100 steps with cosine learning rate scheduling
5. The trained adapter is saved to `./my-model`

You can also train from the command line:

```bash
backprop train --data my_data.jsonl --steps 100
```

Or use the web UI:

```bash
backprop ui
```

## 6. Export and run with Ollama

Once training is done, export to GGUF and register with Ollama:

```python
# Export to GGUF (quantized for fast local inference)
result = trainer.export("gguf", quantization="q4_k_m")

# Register with Ollama
from backpropagate import register_with_ollama
register_with_ollama(result.path, "my-finetuned-model")
```

Now run it:

```bash
ollama run my-finetuned-model
```

The `q4_k_m` quantization gives a good balance between file size and quality. For higher quality at larger file size, use `q8_0`. For the smallest file, use `q2_k`.

CLI equivalent for export:

```bash
backprop export ./my-model --format gguf --quantization q4_k_m --ollama --ollama-name my-finetuned-model
```

## 7. Next steps

Once you have a working fine-tune, here are ways to improve:

- **More data** -- Fine-tuning quality scales with dataset size and diversity. 1,000+ high-quality examples produce noticeably better results than 100.
- **Multi-run SLAO training** -- Prevents catastrophic forgetting during longer training by merging LoRA adapters between runs. Use `trainer.multi_run()` instead of `trainer.train()` for extended fine-tuning.
- **Training presets** -- Use `get_preset("balanced")` or `get_preset("quality")` from `backpropagate.config` for research-backed hyperparameter combinations.
- **Dataset quality tools** -- The `backpropagate.datasets` module offers deduplication, perplexity filtering, and curriculum learning to improve your training data before training.
- **GPU monitoring** -- For long training runs, `GPUMonitor` watches temperature and VRAM, pausing training before your hardware hits dangerous levels.
- **Experiment tracking** -- Install the `[monitoring]` extra to log training runs to Weights & Biases.

For detailed coverage of each topic, see the [Training](/backpropagate/handbook/training/), [Export](/backpropagate/handbook/export/), and [Reference](/backpropagate/handbook/reference/) pages.
