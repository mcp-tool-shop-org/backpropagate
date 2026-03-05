---
title: Reference
description: Architecture, CLI, Windows support, and development.
sidebar:
  order: 4
---

## Architecture

```
backpropagate/
├── trainer.py           # Core Trainer class
├── multi_run.py         # Multi-run SLAO training
├── slao.py              # SLAO LoRA merging algorithm
├── datasets.py          # Dataset loading & filtering
├── export.py            # GGUF/Ollama export
├── config.py            # Pydantic settings
├── gpu_safety.py        # GPU monitoring & safety
└── ui.py                # Gradio interface
```

## CLI commands

```bash
backprop train --data <file> --model <model> --steps <n>
backprop multi-run --data <file> --runs <n> --steps <n>
backprop export <path> --format gguf --quantization <q> [--ollama] [--ollama-name <name>]
backpropagate --ui --port <port>
```

## Windows support

Backpropagate is designed to work on Windows out of the box:

- Pre-tokenization to avoid multiprocessing crashes
- Automatic xformers disable for RTX 40/50 series
- Safe dataloader settings
- Tested on RTX 5080 (16GB VRAM)

## Privacy

All training happens locally on your GPU. No network requests except to download models from HuggingFace (which you initiate). No telemetry, no cloud dependency.

## Headless by design

Built for CI/CD pipelines, automated workflows, and programmatic execution. Full Python API with structured logging. Callbacks for progress tracking and early stopping. No UI required.
