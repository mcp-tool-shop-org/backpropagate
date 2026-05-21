---
title: Environment variables
description: Every BACKPROPAGATE_* knob — what it does, what the default is, and which surface it affects.
sidebar:
  order: 7
---

Every Backpropagate setting can be overridden via an environment variable. The convention is `BACKPROPAGATE_<GROUP>__<FIELD>` — note the double underscore between the group and the field name (Pydantic's `env_nested_delimiter`).

Two ways to set them: export in your shell, or put them in a `.env` file in the working directory (loaded automatically when the `[validation]` extra is installed).

## Logging

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. Sets root logger level. |
| `BACKPROPAGATE_LOG_JSON` | auto-detect | `true` forces JSON logs, `false` forces console. Auto-detects from TTY by default (JSON when piped, console when interactive). |
| `BACKPROPAGATE_LOG_FILE` | unset | Path to a file to mirror logs into. If unset, logs go to stderr only. |

## CLI / runtime opt-outs

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_DEFER_FEATURE_DETECTION` | unset | When set (any truthy value), feature detection at startup is skipped — every optional dep flag stays `False` until you call `refresh_features()` explicitly. Use for the absolute-fastest CLI startup; pays the cost on first real use. |

## UI sandbox (FB-003)

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Single allowed-base directory for **all** UI-initiated filesystem writes (saved adapters, GGUF exports, converted datasets, Modelfiles). Every UI sink passes this as `allowed_base` to `safe_path` so user-supplied paths cannot escape the sandbox. **Denylist-validated** — resolves against a small set of forbidden bases (`/etc`, `/var`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.); if the override resolves into one of those, startup fails with `UI_OUTPUT_DIR_FORBIDDEN`. Pick a non-system directory. |
| `BACKPROPAGATE_UI__PORT` | `7862` | Port the Gradio app binds. |
| `BACKPROPAGATE_UI__HOST` | `127.0.0.1` | Bind host. Localhost-only by default for security. |
| `BACKPROPAGATE_UI__SHARE` | `false` | Whether to enable Gradio's public-URL feature. Equivalent to the CLI `--share` flag. |
| `BACKPROPAGATE_UI__AUTO_OPEN` | `true` | Open the browser automatically when the UI starts. |

## Security (F-001 / FB-003)

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE` | `true` | When `true` (the default), `backprop ui --share` refuses to launch without `--auth user:password`. Set to `false` to explicitly opt out — the CLI will print a loud warning every time it launches share without auth. |
| `BACKPROPAGATE_SECURITY__REQUIRE_AUTH` | `false` | Top-level "this is a production deployment" toggle; when `true`, the UI requires auth for every endpoint. |
| `BACKPROPAGATE_SECURITY__AUTH_USERNAME` | unset | Auth username for the UI (used by `SecurityConfig.get_auth_tuple()`). |
| `BACKPROPAGATE_SECURITY__AUTH_PASSWORD` | unset | Auth password for the UI. Stored as a secret in the structured config. |
| `BACKPROPAGATE_SECURITY__ALLOWED_PATHS` | unset (no restriction) | Comma-separated list of directories the UI is allowed to read/write. |
| `BACKPROPAGATE_SECURITY__BLOCK_PATH_TRAVERSAL` | `true` | Refuse paths containing `..` segments. |
| `BACKPROPAGATE_SECURITY__SESSION_TIMEOUT_MINUTES` | `30` | UI session lifetime. |
| `BACKPROPAGATE_SECURITY__JWT_SECRET` | unset (random) | JWT signing secret. If unset, a random one is generated per process and sessions are lost on restart. |
| `BACKPROPAGATE_SECURITY__JWT_ALGORITHM` | `HS256` | JWT signing algorithm. |
| `BACKPROPAGATE_SECURITY__ENABLE_CSRF` | `true` | CSRF protection for state-changing requests. |
| `BACKPROPAGATE_SECURITY__CSRF_TOKEN_EXPIRY_MINUTES` | `60` | CSRF token lifetime. |
| `BACKPROPAGATE_SECURITY__RATE_LIMIT_TRAINING` | `3` | Max training-start requests per minute. |
| `BACKPROPAGATE_SECURITY__RATE_LIMIT_EXPORT` | `5` | Max export requests per minute. |
| `BACKPROPAGATE_SECURITY__AUDIT_LOG_ENABLED` | `true` | Emit security-audit log lines (rate-limit hits, auth failures, path-traversal blocks). |
| `BACKPROPAGATE_SECURITY__AUDIT_LOG_FILE` | unset (stdout) | Path for audit log. |
| `BACKPROPAGATE_SECURITY__ENABLE_CSP` | `true` | Emit a Content-Security-Policy response header. |
| `BACKPROPAGATE_SECURITY__CSP_REPORT_ONLY` | `false` | When `true`, CSP runs in report-only mode (logs violations, does not block). |

## Model

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model id or local path. This is the **canonical default** when `Trainer()` is called with no model argument. |
| `BACKPROPAGATE_MODEL__LOAD_IN_4BIT` | `true` | 4-bit quantization at load time (saves ~50% VRAM). |
| `BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH` | `2048` | Maximum sequence length. |
| `BACKPROPAGATE_MODEL__DTYPE` | unset (auto) | Force `bf16` / `fp16` / `fp32`. Auto-detects bf16 on Ampere+. |
| `BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE` | `true` | Whether to trust custom modeling code from HF Hub. |

## LoRA

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_LORA__R` | `16` | LoRA rank. |
| `BACKPROPAGATE_LORA__LORA_ALPHA` | `32` | LoRA scaling factor. |
| `BACKPROPAGATE_LORA__LORA_DROPOUT` | `0.05` | LoRA dropout rate. |
| `BACKPROPAGATE_LORA__USE_GRADIENT_CHECKPOINTING` | `unsloth` | `unsloth` / `true` / `false`. |
| `BACKPROPAGATE_LORA__RANDOM_STATE` | `42` | RNG seed for LoRA init reproducibility. |

## Training

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_TRAINING__PER_DEVICE_TRAIN_BATCH_SIZE` | `2` | Per-device batch size. |
| `BACKPROPAGATE_TRAINING__GRADIENT_ACCUMULATION_STEPS` | `4` | Effective batch size = batch × grad-accum. |
| `BACKPROPAGATE_TRAINING__MAX_STEPS` | `100` | Hard cap on training steps. |
| `BACKPROPAGATE_TRAINING__NUM_TRAIN_EPOCHS` | `1` | Number of epochs (ignored if `MAX_STEPS > 0`). |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Learning rate. |
| `BACKPROPAGATE_TRAINING__WEIGHT_DECAY` | `0.01` | Weight decay. |
| `BACKPROPAGATE_TRAINING__WARMUP_STEPS` | `10` | Number of warmup steps. |
| `BACKPROPAGATE_TRAINING__WARMUP_RATIO` | `0.0` | Warmup ratio (alternative to warmup steps). |
| `BACKPROPAGATE_TRAINING__OPTIM` | `adamw_8bit` | Optimizer name. |
| `BACKPROPAGATE_TRAINING__LR_SCHEDULER_TYPE` | `cosine` | LR schedule. |
| `BACKPROPAGATE_TRAINING__LOGGING_STEPS` | `10` | Log every N steps. |
| `BACKPROPAGATE_TRAINING__SAVE_STEPS` | `100` | Save a checkpoint every N steps. |
| `BACKPROPAGATE_TRAINING__BF16` | `true` | Use bf16 (Ampere+ recommended). |
| `BACKPROPAGATE_TRAINING__FP16` | `false` | Use fp16 (older GPUs). |
| `BACKPROPAGATE_TRAINING__SEED` | `42` | RNG seed. |
| `BACKPROPAGATE_TRAINING__OUTPUT_DIR` | `./output` | Where to write checkpoints and exports. |
| `BACKPROPAGATE_TRAINING__OVERWRITE_OUTPUT_DIR` | `true` | Whether to overwrite an existing output dir. |

## Data

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_DATA__DATASET_NAME` | `HuggingFaceH4/ultrachat_200k` | Default HF dataset when none is passed. |
| `BACKPROPAGATE_DATA__DATASET_SPLIT` | `train_sft` | Which split to load. |
| `BACKPROPAGATE_DATA__MAX_SAMPLES` | `1000` | Cap dataset to N samples (`0` = all). |
| `BACKPROPAGATE_DATA__TEXT_COLUMN` | `text` | Column name for raw-text datasets. |
| `BACKPROPAGATE_DATA__CHAT_FORMAT` | `chatml` | Chat template (`chatml` / `llama` / `alpaca` / `sharegpt`). |
| `BACKPROPAGATE_DATA__PRE_TOKENIZE` | `true` | Pre-tokenize before training (Windows-safe). |
| `BACKPROPAGATE_DATA__SHUFFLE` | `true` | Shuffle the dataset. |
| `BACKPROPAGATE_DATA__PACKING` | `false` | Combine short sequences. |

## Windows

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_WINDOWS__DATALOADER_NUM_WORKERS` | `0` | Workers for the DataLoader (0 is the safe default on Windows). |
| `BACKPROPAGATE_WINDOWS__TOKENIZERS_PARALLELISM` | `false` | Forwarded to HF tokenizers. |
| `BACKPROPAGATE_WINDOWS__XFORMERS_DISABLED` | `true` | Disable xformers (incompatible with SM 12.0+). |
| `BACKPROPAGATE_WINDOWS__CUDA_LAUNCH_BLOCKING` | `false` | Set `CUDA_LAUNCH_BLOCKING=1`; useful for debugging, slows training. |
| `BACKPROPAGATE_WINDOWS__PRE_TOKENIZE` | `true` | Pre-tokenize to avoid multiprocessing issues. |

## Multi-run

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_MULTIRUN__NUM_RUNS` | `5` | Number of training runs in a multi-run campaign. |
| `BACKPROPAGATE_MULTIRUN__STEPS_PER_RUN` | `100` | Steps per run. |
| `BACKPROPAGATE_MULTIRUN__SAMPLES_PER_RUN` | `1000` | Samples per run. |
| `BACKPROPAGATE_MULTIRUN__CONTINUE_FROM_PREVIOUS` | `true` | Resume from the previous LoRA each run. |
| `BACKPROPAGATE_MULTIRUN__SAVE_INTERMEDIATE` | `true` | Save intermediate checkpoints. |

## See also

- [Error codes](/backpropagate/handbook/error-codes/) — what each `BackpropagateError.code` means, including the env-var-related ones (`UI_OUTPUT_DIR_FORBIDDEN`, `INPUT_AUTH_REQUIRED`).
- [Troubleshooting](/backpropagate/handbook/troubleshooting/) — symptoms-first reverse index.
