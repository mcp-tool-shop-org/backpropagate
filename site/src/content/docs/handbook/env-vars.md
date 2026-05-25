---
title: Environment variables
description: Every BACKPROPAGATE_* knob — what it does, what the default is, and which surface it affects.
sidebar:
  order: 7
---

Every Backpropagate setting can be overridden via an environment variable. Two conventions coexist:

- **Pydantic-nested-settings shape (`BACKPROPAGATE_<GROUP>__<FIELD>`, double underscore between group and field)** — used for every knob inside the structured `Settings` model (logging, model, lora, training, data, windows, multirun, security, plus the UI sandbox `__OUTPUT_DIR`). These are bound via Pydantic's `env_nested_delimiter`.
- **Raw `os.environ.get` shape (`BACKPROPAGATE_<NAME>` or `BACKPROPAGATE_UI_<NAME>`, single underscores throughout)** — used for runtime knobs that bypass Pydantic (UI subprocess launch config, debug-traceback toggle, GGUF convert-script discovery, structured-logger config).

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
| `BACKPROPAGATE_DEFER_FEATURE_QUIET` | unset | Suppresses the one-shot import-time WARN that fires when `BACKPROPAGATE_DEFER_FEATURE_DETECTION` is active. v1.4 Stage C added the WARN so operators aren't surprised by the deferred-detection auto-refresh path; set this var to `1` if you've internalized the trade-off and want quiet startup. |
| `BACKPROPAGATE_QUIET_TOKEN_HINT` | unset | Suppresses the one-shot stderr note that fires when `cmd_push` falls back to `HF_TOKEN` env var (because no `--token` / `--token-file` was passed). v1.4 Stage C added the note to calibrate the "env var vs file" safety trade-off; set this var to `1` to silence after you've calibrated. |
| `BACKPROPAGATE_DEBUG` | unset | When set to any truthy value, the top-level CLI exception net prints the full Python traceback in addition to the one-line error message. Off by default to keep operator-facing failures short; flip to `1` when filing a bug report. |

## UI sandbox (FB-003)

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Single allowed-base directory for **all** UI-initiated filesystem writes (saved adapters, GGUF exports, converted datasets, Modelfiles). Every UI sink passes this as `allowed_base` to `safe_path` so user-supplied paths cannot escape the sandbox. **Denylist-validated** — resolves against a denylist of system + credential trees (`/etc`, `/usr`, `/sys`, `/dev`, `/boot`, `/bin`, `/sbin`, `/var/run`, `/var/lib`, `/root`, `~/.ssh`, `~/.aws`, `~/.kube`, `~/.docker`, `~/.gnupg`, `~/.config`, plus the Windows system roots and per-user credential dirs); if the override resolves into a denied path, startup fails with `UI_OUTPUT_DIR_FORBIDDEN`. Bare `/var` is intentionally NOT denied because macOS's per-user temp tree lives at `/var/folders/<hash>/T/...`; only `/var/run` and `/var/lib` are denied individually. Pick a non-system directory. |
| `BACKPROPAGATE_UI_PORT` | `7862` | Port the Reflex frontend binds. Read directly by `cmd_ui` (cli.py), `rxconfig.py` (CORS allowlist build-up), and the auth-badge resolver (ui_security.py). Reflex also binds a backend WebSocket on `port + 1` by default. |
| `BACKPROPAGATE_UI_HOST_BIND` | `127.0.0.1` | Bind host for the Reflex backend. Read by `cmd_ui` and the auth-badge resolver. Non-loopback overrides (e.g. `0.0.0.0`) require `--auth user:pass` post-v1.2.0; otherwise the runtime refuses to start with `[RUNTIME_UI_AUTH_NOT_ENFORCED]`. SSH port-forwarding remains the lowest-friction remote-access pattern. |
| `BACKPROPAGATE_UI_SHARE_HOST` | unset | Tunnel host announced to the Host-header allowlist when `--share` is active (cli.py, ui_security.py). v1.3 transitional: populated by the cloudflared tunnel implementation once Wave 6 lands. See [the CLI reference](/backpropagate/handbook/cli-reference/#share--host-require-auth-v120-contract). |
| `BACKPROPAGATE_UI_AUTH` | unset | When set, the Reflex auth middleware enforces HTTP Basic credentials on every HTTP route and the `/_event` WebSocket upgrade. Same format as `--auth user:pass`. The CLI strips an ambient value when `--auth` was not also passed, to close the v1.1.x ambient-env-bypass path. |
| `BACKPROPAGATE_UI_LAUNCH_TOKEN` | unset | Per-launch random token honored by the auth middleware (token_auto mode). The Wave-6 MVP does not have the CLI generate one; the middleware honors it if an operator sets it explicitly. v1.3 polish item to wire CLI auto-generation. |
| `BACKPROPAGATE_UI_CORS_EXTRA_ORIGINS` | unset | Comma-separated extra CORS origins, additive to the loopback defaults. Used by operators who serve the UI behind a reverse proxy at a non-loopback origin. |
| `BACKPROPAGATE_UI_QUIET` | unset | When set to `1`, suppresses the 3-line UI startup banner (URL / auth mode / Ctrl+C hint) on stderr. Use for CI / headless launches that don't want banner noise. Only the exact value `1` suppresses; `0` / `false` / unset leave the banner on. |
| `BACKPROPAGATE_UI_REQUEST_LOG` | unset | When set to `1` / `true` / `yes` / `on`, enables the v1.3 request-logging ASGI middleware (`ui_app/middleware/request_logging.py`). Emits one `ui.request` structlog line per HTTP request / WebSocket upgrade with fields `method`, `path`, `status`, `duration_ms`, `auth_mode`, `auth_user`, `remote_addr`, `scope_type`. Default OFF — Reflex's own uvicorn access log covers the basics; opt in when you need structured per-request observability or want to ship records to a SIEM. |
| `BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN` | `100` | v1.3 per-IP HTTP rate-limit cap (rolling 60s window). The rate-limit middleware (`ui_app/middleware/rate_limit.py`) wraps the auth middleware so brute-force attempts can't exhaust the HMAC budget. Set to `0` to fully disable HTTP rate-limiting (smoke tests, local-only operator use); any positive integer is the per-minute budget. Pass-through paths (`/ping`, `/_next/`, `/favicon`) are never rate-limited. |
| `BACKPROPAGATE_UI_RATE_LIMIT_WS_PER_MIN` | `10` | v1.3 per-IP WebSocket-upgrade rate-limit cap (rolling 60s window). Lower than the HTTP cap because each accepted WS holds a backend connection until the cookie expires. Set to `0` to fully disable WS rate-limiting. WS rejections close pre-accept with application code `4429` (mirrors the auth middleware's 4401 / 4403 / 4404 pattern). |

## Security (F-001 / FB-003)

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE` | `true` | **No-op under the v1.1+ Reflex UI.** Held for forward-compat with the Gradio era. Post-v1.2.0 the `--share`-requires-`--auth` contract is enforced by `cli.py:cmd_ui`, not by this flag, and the gate this variable used to relax never fires. The opt-out the variable used to provide is intentionally unavailable — see [the CLI reference](/backpropagate/handbook/cli-reference/#share--host-require-auth-v120-contract). |
| `BACKPROPAGATE_SECURITY__REQUIRE_AUTH` | `false` | Top-level "this is a production deployment" toggle from the Gradio era. Held for forward-compat; the post-v1.2.0 Reflex auth middleware is controlled by `--auth user:pass` and `BACKPROPAGATE_UI_AUTH`, not by this flag. |
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

## Export / GGUF

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_LLAMA_CPP_PATH` | unset | Operator escape hatch for non-standard llama.cpp install locations used by `backprop export --format gguf`. Accepts either the path to `convert_hf_to_gguf.py` directly or the llama.cpp directory containing it. Searched FIRST, before `shutil.which` / `~/llama.cpp` / `/usr/local/bin`. |

## cloudflared tunnel (v1.3)

| Variable | Default | What it does |
|----------|---------|--------------|
| `BACKPROPAGATE_CLOUDFLARED_TIMEOUT` | `30` (seconds) | Maximum seconds `backprop ui --share` will wait for `cloudflared` to announce the `https://*.trycloudflare.com` URL on stderr. The default 30s is comfortable on a healthy uplink; operators on slow uplinks can extend the budget. On timeout the CLI emits a clear error pointing at SSH port-forwarding as the fallback. |

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
| `BACKPROPAGATE_LORA__R` | `256` | LoRA rank. v1.3 default (was 16 in v1.2.x — pass `--lora-preset=fast` on the CLI for the v1.2.x rank-16 footprint). |
| `BACKPROPAGATE_LORA__LORA_ALPHA` | `512` | LoRA scaling factor (alpha = 2 * r convention; v1.3 default. Was 32 in v1.2.x — pass `--lora-preset=fast` on the CLI for the v1.2.x alpha=32 footprint). |
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
| `BACKPROPAGATE_DATA__PACKING` | `true` | Combine short sequences via TRL sample packing. v1.3 default flipped from `false` to `true` per BACKEND-4 — 1.7-3× wall-clock throughput on SFT runs. Set to `false` to opt out (boundary-token leakage with exotic chat templates is the documented edge case). |

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
