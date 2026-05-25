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
├── ui_app/              # Reflex (Radix UI) interface — v1.1.0+
│   ├── app.py           #   App entry + Reflex Page wiring
│   ├── auth.py          #   ENFORCEMENT_AVAILABLE flag + refuse-to-start guard
│   ├── chrome.py        #   Header / left nav / side rail / footer
│   ├── pages/           #   Train / Multi-Run / Export / Dataset surfaces
│   └── components/      #   Sparkline, GPU ring, event log, callouts
├── rxconfig.py          # Reflex Config (app_name, ports, refuse-to-start guard)
├── ui_state.py          # rx.State subclasses driving the four surfaces
└── ui_security.py       # Path-sandbox + file validation + denylist for UI writes
```

The UI is opt-in via the `[ui]` extra (`pip install backpropagate[ui]`) and pulls in Reflex per the `[ui]` extra pin in `pyproject.toml` (single source of truth — run `pip show reflex` after install for the resolved version). The v1.0 Gradio implementation (`ui_gradio_legacy.py` + `theme_gradio_legacy.py`) was preserved through v1.1.x as reference and removed in v1.2.0.

## CLI commands

```bash
backprop train --data <file> --model <model> --steps <n>
backprop multi-run --data <file> --runs <n> --steps <n> --samples <n>
backprop export <path> --format gguf --quantization <q> [--ollama] [--ollama-name <name>]
backprop ui --port <port> [--auth user:pass] [--share] [--host 0.0.0.0]
backprop info
backprop config
```

See [CLI reference](/backpropagate/handbook/cli-reference/) for every flag, every default, and the full exit-code contract.

### `--share` / `--host` require `--auth` post-v1.2.0

The v1.2.0 FastAPI auth middleware (`backpropagate/ui_app/auth.py::basic_auth_transformer`, wired in `ui_app/app.py` via `rx.App(api_transformer=...)`) enforces credentials on every HTTP route and the `/_event` WebSocket upgrade. `--auth user:pass` flows through `validate_auth_shape` and into the Reflex subprocess via `BACKPROPAGATE_UI_AUTH`. What refuses to start:

- `backprop ui --share` without `--auth` → exits `1` with `[RUNTIME_UI_AUTH_NOT_ENFORCED]` (a public URL with no credentials is the v1.1.x bug closed by [GHSA-f65r-h4g3-3h9h](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/GHSA-f65r-h4g3-3h9h)).
- `backprop ui --host <non-loopback>` without `--auth` → same code (DNS-rebinding defense).
- `backprop ui --auth user:pass` when `ENFORCEMENT_AVAILABLE=False` (degraded `[ui]` extra install) → same code.

The refuse-to-start contract is enforced one layer deeper too — `python -m reflex run` from the package directory refuses unless the legitimate `backprop ui` bridge sets its bypass env var. For remote access without a public URL, SSH port-forwarding (`ssh -L 7860:localhost:7860 <host>`) stays the lower-friction option. Full rationale + the four-layer defense chain in [the security page](/backpropagate/handbook/security/#four-layer-defense-in-depth) and the project [SECURITY.md](https://github.com/mcp-tool-shop-org/backpropagate/blob/main/SECURITY.md).

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
BACKPROPAGATE_MODEL__NAME=Qwen/Qwen2.5-7B-Instruct
BACKPROPAGATE_TRAINING__LEARNING_RATE=2e-4
BACKPROPAGATE_LORA__R=32
```

Backpropagate also reads from a `.env` file if present. Install the `[validation]` extra for full Pydantic-powered config with type checking.

See [Environment variables](/backpropagate/handbook/env-vars/) for the complete catalog — every knob grouped by family (logging, security, UI sandbox, model, LoRA, training, data, Windows, multi-run).

## Training presets

Built-in presets for common scenarios (these are `TRAINING_PRESETS` — multi-run loop hyperparameters; they predate the v1.3 `LORA_PRESETS` namespace which governs LoRA shape only and is referenced via `--lora-preset`):

| Preset | LoRA r | Eff. Batch | LR | Runs | Use case |
|--------|--------|-----------|-----|------|----------|
| `fast-3b` | 8 | 8 | 5e-4 | 3 | Rapid iteration with 3B models |
| `fast` | 8 | 8 | 5e-4 | 3 | Quick testing with 7B models |
| `balanced` | 16 | 16 | 2e-4 | 5 | Recommended default |
| `quality` | 32 | 32 | 1e-4 | 10 | Maximum training effectiveness |

> Note: `TRAINING_PRESETS["quality"]` (rank 32, above) is distinct from `LORA_PRESETS["quality"]` (rank 256 + all-linear + 10× LR, the v1.3 LoRA-shape default). The names collide for historical reasons; the v1.3 `--lora-preset=quality` / `--lora-preset=fast` flag controls LoRA shape only. See [CLI reference](/backpropagate/handbook/cli-reference/) for the LoRA preset table.

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

Backpropagate uses a structured exception hierarchy rooted at `BackpropagateError`. Every error carries a human-readable `message`, an optional `suggestion`, and a stable machine-readable `code` (Ship Gate B1) drawn from prefixes `INPUT_*` / `CONFIG_*` / `DEP_*` / `RUNTIME_*` / `STATE_*` / `PARTIAL_*`. Codes are stable across class renames — quote them in bug reports. See the [error code catalog](/backpropagate/handbook/error-codes/) for the full list and recommended fix per code.

The structured envelope is also exposed programmatically:

```python
try:
    trainer.train("data.jsonl", steps=100)
except BackpropagateError as e:
    envelope = e.to_dict()
    # {"type": "...", "code": "RUNTIME_GPU_OOM", "message": "...",
    #  "retryable": False, "suggestion": "...", "details": {...}, "cause": "..."}
```

The CLI maps exceptions to standard exit codes:

| Exit code | Meaning | Exception types |
|-----------|---------|-----------------|
| 0 | Success | -- |
| 1 | User error | `UserInputError`, `ConfigurationError`, `DatasetError` |
| 2 | Runtime error | `TrainingError`, `GPUError`, `ExportError`, `SLAOError` |
| 3 | Partial failure | `BatchOperationError`, `PartialSuccess` |

### Stderr redaction (B-006)

In non-verbose mode, unhandled-exception stderr is automatically redacted before printing: Bearer tokens, `sk-*` (OpenAI), `hf_*` (HuggingFace), AWS access keys (`AKIA*`), and `password=` / `token=` / `api_key=` key-value pairs are scrubbed. This means stderr is **safe to paste** into a public bug report. For the full unredacted trace, re-run with `--verbose`, but review the output for any secrets first.

## Reporting bugs

When training fails, the first thing in your bug report should be the **`run_id`** — Backpropagate prints `run_started run_id=<uuid>` at startup and binds the same id to:

- Every structured log line emitted during the run
- Every checkpoint manifest under `output_dir/`
- Every SLAO merge snapshot
- The `TrainingRun.run_id` / `RunResult.run_id` field in the return value

Quoting the run_id in an issue lets a maintainer correlate every log line, checkpoint, and merge for that exact run.

A good bug report includes:

1. The `run_id`
2. The structured error code (e.g. `[RUNTIME_GPU_OOM]`)
3. Output of `backprop info` (Python / PyTorch / CUDA / GPU / extras)
4. Redacted stderr (re-run with `--verbose` for the full trace; review before posting)
5. A minimal reproduction

For security issues, do not open a public issue — see [SECURITY.md](https://github.com/mcp-tool-shop-org/backpropagate/blob/main/SECURITY.md).

### Exception hierarchy

```
BackpropagateError
├── UserInputError
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
├── SLAOError
│   ├── SLAOMergeError
│   └── SLAOCheckpointError
├── BatchOperationError
└── PartialSuccess
```
