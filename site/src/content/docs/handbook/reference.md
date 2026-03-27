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
├── datasets.py          # Dataset loading, filtering & curriculum
├── export.py            # GGUF/Ollama export
├── config.py            # Pydantic settings + training presets
├── gpu_safety.py        # GPU monitoring & safety
├── cli.py               # CLI entry point (backprop command)
├── checkpoints.py       # Checkpoint management
├── exceptions.py        # Structured error hierarchy
├── feature_flags.py     # Optional feature detection
├── security.py          # Path traversal & torch security
├── logging_config.py    # Structured logging setup
├── theme.py             # Gradio theme customization
├── ui.py                # Gradio interface
└── ui_security.py       # Rate limiting, CSRF, file validation
```

## CLI commands

```bash
backprop train --data <file> --model <model> --steps <n>
backprop multi-run --data <file> --runs <n> --steps <n>
backprop export <path> --format gguf --quantization <q> [--ollama] [--ollama-name <name>]
backprop ui --port <port> [--share] [--auth user:pass]
backprop info
backprop config
```

## Windows support

Backpropagate is designed to work on Windows out of the box:

- Pre-tokenization to avoid multiprocessing crashes
- Automatic xformers disable for RTX 40/50 series
- Safe dataloader settings
- Tested on RTX 5080 (16GB VRAM)

## Privacy

All training happens locally on your GPU. No network requests except to download models from HuggingFace (which you initiate). No telemetry, no cloud dependency.

## Configuration

All settings can be overridden via environment variables with the `BACKPROPAGATE_` prefix. Nested settings use double underscores as delimiters.

```bash
BACKPROPAGATE_MODEL__NAME=unsloth/Qwen2.5-7B-Instruct-bnb-4bit
BACKPROPAGATE_TRAINING__LEARNING_RATE=2e-4
BACKPROPAGATE_LORA__R=32
```

Backpropagate also reads from a `.env` file if present. Install the `[validation]` extra for full Pydantic-powered config with type checking.

## Training presets

Built-in presets for common scenarios:

| Preset | LoRA r | Eff. Batch | LR | Runs | Use case |
|--------|--------|-----------|-----|------|----------|
| `fast-3b` | 8 | 8 | 5e-4 | 3 | Rapid iteration with 3B models |
| `fast` | 8 | 8 | 5e-4 | 3 | Quick testing with 7B models |
| `balanced` | 16 | 16 | 2e-4 | 5 | Recommended default |
| `quality` | 32 | 32 | 1e-4 | 10 | Maximum training effectiveness |

```python
from backpropagate.config import get_preset

preset = get_preset("balanced")
trainer = Trainer(lora_r=preset.lora_r, learning_rate=preset.learning_rate)
```

## Dataset formats

Backpropagate auto-detects and converts between common dataset formats:

- **ShareGPT**: `{"conversations": [{"from": "human/gpt", "value": "..."}]}`
- **Alpaca**: `{"instruction": "...", "input": "...", "output": "..."}`
- **OpenAI**: `{"messages": [{"role": "user/assistant", "content": "..."}]}`
- **ChatML**: `{"text": "<|im_start|>user\n...<|im_end|>\n..."}`
- **Raw text**: Plain text files

## GPU safety

The `gpu_safety` module monitors your GPU during training and intervenes when conditions become unsafe:

| Level | Temperature | Action |
|-------|-------------|--------|
| Safe | Below 80C | Full speed |
| Warm | Near 80C | Elevated temps, monitoring closely |
| Warning | 80C | Approaching limits, throttling recommended |
| Critical | 90C | Training paused |
| Emergency | 95C | Training aborted |

VRAM thresholds are also monitored: a warning fires at 90% usage, and a critical alert at 95%.

Use `check_gpu_safe()` for a one-shot check, or `GPUMonitor` for continuous monitoring during long runs.

## Checkpoint management

The `checkpoints` module manages disk space during multi-run training:

```python
from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy

policy = CheckpointPolicy(keep_best_n=3, keep_final=True)
manager = CheckpointManager(checkpoint_dir, policy)

# After each run
manager.register(run_idx, checkpoint_path, val_loss=0.5)
manager.prune()  # Removes low-value checkpoints automatically
```

The manager keeps the best N checkpoints by validation loss, always preserves the final checkpoint, and optionally retains run-boundary checkpoints. A manifest file tracks metadata for each saved checkpoint.

## Headless by design

Built for CI/CD pipelines, automated workflows, and programmatic execution. Full Python API with structured logging. Callbacks for progress tracking and early stopping. No UI required.

## Error handling

Backpropagate uses a structured exception hierarchy rooted at `BackpropagateError`. Every error includes a human-readable `message` and an optional `suggestion` field with a recommended fix. The CLI maps exceptions to standard exit codes:

| Exit code | Meaning | Exception types |
|-----------|---------|-----------------|
| 0 | Success | -- |
| 1 | User error | `ConfigurationError`, `DatasetError` |
| 2 | Runtime error | `TrainingError`, `GPUError`, `ExportError` |
| 3 | Partial failure | `BatchOperationError` |

Exception hierarchy:

```
BackpropagateError
├── ConfigurationError
│   └── InvalidSettingError
├── DatasetError
│   ├── DatasetNotFoundError
│   ├── DatasetParseError
│   ├── DatasetValidationError
│   └── DatasetFormatError
├── TrainingError
│   ├── ModelLoadError
│   ├── TrainingAbortedError
│   └── CheckpointError
├── ExportError
│   ├── LoRAExportError
│   ├── GGUFExportError
│   ├── MergeExportError
│   └── OllamaRegistrationError
├── GPUError
│   ├── GPUNotAvailableError
│   ├── GPUMemoryError
│   ├── GPUTemperatureError
│   └── GPUMonitoringError
└── SLAOError
    ├── SLAOMergeError
    └── SLAOCheckpointError
```
