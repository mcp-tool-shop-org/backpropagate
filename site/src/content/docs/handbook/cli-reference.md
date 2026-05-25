---
title: CLI reference
description: Every backprop subcommand and flag.
sidebar:
  order: 8
---

The `backprop` CLI exposes the subcommands below. Run `backprop <subcommand> --help` for the canonical flag list from argparse; this page mirrors that with descriptions and defaults. Run `backprop --help` for the complete list (additional subcommands `push` / `resume` / `list-runs` / `show-run` / `runs` are documented in their own pages and via `--help`).

## Exit codes (Ship Gate B2)

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | User error (bad argument, missing dependency for the chosen subcommand, `--share` without `--auth`) |
| `2` | Runtime error (training failure, GPU failure, export failure) |
| `3` | Partial success (operation ran to completion, some units failed) |

## `backprop train`

Fine-tune an LLM on a dataset.

```bash
backprop train --data my_data.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | `Qwen/Qwen2.5-7B-Instruct` | Model name (HF id or local path). |
| `--data`, `-d` | **required** | Dataset path (JSONL/CSV) or HuggingFace dataset name. |
| `--steps` | `100` | Number of training steps (must be > 0). |
| `--samples` | unset | Maximum samples to use from the dataset (must be > 0). |
| `--batch-size` | `auto` | Per-device batch size. `auto` queries GPU VRAM. |
| `--lr` | `2e-4` | Learning rate (must be > 0). |
| `--lora-r` | `256` | LoRA rank (must be > 0). v1.3 default; pass `--lora-preset=fast` for the v1.2.x rank-16 footprint. |
| `--output`, `-o` | `./output` | Output directory. |
| `--no-unsloth` | off | Disable Unsloth even if available. |
| `--lora-preset` | `quality` | One of `quality` / `fast`. `quality` = rank 256 + all-linear + 10× LR (v1.3 default, matches full fine-tuning per Biderman 2024). `fast` = rank 16 + q+v + 1× LR (v1.2.x footprint). |
| `--use-dora` | off | Enable DoRA (Weight-Decomposed Low-Rank Adaptation). Rank-8 DoRA ≈ rank-32 LoRA quality, zero inference overhead. Requires `peft>=0.10`. |
| `--no-packing` | (packing ON by default) | Disable sample packing. Default ON gives 1.7-3× throughput; disable only if you hit packing-incompatible behavior. |
| `--init-lora-weights` | `default` | One of `default` / `pissa` / `loftq`. PiSSA + LoftQ recover quality lost during QLoRA quantization at zero runtime cost. |
| `--optim` | auto | Optimizer string. `auto` picks `paged_adamw_8bit` on consumer GPUs (<24GB VRAM), `adamw_torch_fused` otherwise. Override with `adamw_torch` / `paged_adamw_8bit` / `adamw_8bit` etc. |

The CLI default aligns with `config.py`'s `ModelConfig.name` as of v1.3 (F-018). Pass `--model unsloth/Qwen2.5-7B-Instruct-bnb-4bit` to opt into the pre-quantized variant, but only with the `[unsloth]` extra installed.

## `backprop multi-run`

Train with multiple short runs and SLAO merging between them.

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | `Qwen/Qwen2.5-7B-Instruct` | Model name (HF id or local path). Same as `backprop train`. |
| `--data`, `-d` | **required** | Dataset path or HF name. |
| `--runs` | `5` | Number of training runs. |
| `--steps` | `100` | Steps per run. |
| `--samples` | `1000` | Samples per run. The matching Python-API knob is `samples_per_run`. |
| `--merge-mode` | `slao` | One of `slao` / `simple`. |
| `--output`, `-o` | `./output` | Output directory. |
| `--lora-preset` | `quality` | Same as `backprop train`. |
| `--use-dora` | off | Same as `backprop train`. |
| `--no-packing` | (packing ON by default) | Same as `backprop train`. |
| `--init-lora-weights` | `default` | Same as `backprop train`. |
| `--optim` | auto | Same as `backprop train`. |

## `backprop diff-runs` (v1.3, experimental)

Side-by-side comparison of two completed runs from the on-disk run history. Useful for "did this config tweak actually move the loss" workflows.

```bash
backprop diff-runs <run_id_a> <run_id_b> [--output ./output] [--format {table,json}]
```

| Flag | Default | Description |
|------|---------|-------------|
| `run_id_a` (positional) | **required** | First run_id (or unambiguous prefix). |
| `run_id_b` (positional) | **required** | Second run_id (or unambiguous prefix). |
| `--output`, `-o` | `./output` | Output directory containing `run_history.json`. |
| `--format` | `table` | One of `table` (colorized side-by-side) / `json` (machine-readable). |

Lookup miss raises `InvalidSettingError` naming the run_id + searched dir + next-step suggestions.

## `backprop replay` (v1.3, experimental)

Re-execute a recorded training run with the same model + dataset + hyperparameters. The replay gets a fresh `run_id` (no clobbering of the original) so it can be diffed via `backprop diff-runs`.

```bash
backprop replay <run_id> [--output ./output] [--override KEY=VALUE]
```

| Flag | Default | Description |
|------|---------|-------------|
| `run_id` (positional) | **required** | The run_id to replay (or any unambiguous prefix). |
| `--output`, `-o` | `./output` | Output directory. |
| `--override` | unset | Repeatable `KEY=VALUE`. Override a single hyperparameter (whitelist: `seed`, `learning_rate`, `lr`, `batch_size`, `gradient_accumulation`, `max_steps`, `steps`, `samples`, `lora_r`, `lora_alpha`, `lora_dropout`, `use_dora`, `packing`, `init_lora_weights`, `lora_preset`, `optim`). Unknown keys fail loudly. |

Inherits seed / learning_rate / batch_size / gradient_accumulation / max_steps / lora_r from the recorded entry. Supports both single-run and multi-run session_kinds.

## `backprop export-runs` (v1.3, experimental)

Bulk dump of every run history entry as JSONL (one record per line). Useful for offline analytics, pipeline integration with W&B / MLflow, or attaching a corpus to a support ticket.

```bash
backprop export-runs > runs.jsonl
backprop export-runs --to runs.jsonl --status completed
backprop export-runs | jq '.run_id'
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `./output` | Output directory containing `run_history.json`. |
| `--format` | `jsonl` | Today only `jsonl` supported (CSV intentionally not offered — loses nested loss_history shape). |
| `--to` | unset (stdout) | Write to `PATH` instead of stdout. Parent directory created if missing. |
| `--status` | unset (all) | Filter by status before export — one of `running` / `completed` / `failed`. |

Writes to stdout by default so `backprop export-runs | jq ...` works without intermediate files. Banner goes to stderr to keep stdout clean.

## `backprop export`

Export a trained model to LoRA / merged / GGUF.

```bash
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
```

| Flag | Default | Description |
|------|---------|-------------|
| `model_path` (positional) | **required** | Path to the trained LoRA adapter directory. |
| `--format`, `-f` | `lora` | One of `lora` / `merged` / `gguf`. |
| `--quantization`, `-q` | `q4_k_m` | One of `f16` / `q8_0` / `q5_k_m` / `q4_k_m` / `q4_0` / `q2_k`. |
| `--output`, `-o` | unset | Output directory (defaults to `<model_path>/<format>`). |
| `--ollama` | off | After GGUF export, register with Ollama. |
| `--ollama-name` | unset | Name to use when registering with Ollama. |
| `--hub-token` | unset | HuggingFace Hub token for `--push` flow. Visible to `ps aux` + shell history — prefer `--hub-token-file` or `HF_TOKEN` env var on shared hosts. |
| `--hub-token-file` | unset | Path to a file containing the HF Hub token (mode-0600 recommended). Mutually exclusive with `--hub-token`. Mirrors v1.3 `--auth-file` pattern for keeping credentials off argv. |

## `backprop push`

Push a trained adapter or merged model to the HuggingFace Hub.

```bash
backprop push ./output/lora --repo alice/qwen-finetune
backprop push ./output/lora --repo alice/qwen-finetune --token-file ~/.hf-token
```

| Flag | Default | Description |
|------|---------|-------------|
| `local_path` (positional) | **required** | Path to the trained LoRA adapter or merged model directory. |
| `--repo` | **required** | Target repo (`user/name` or `org/name`). |
| `--token` | unset | HuggingFace Hub token. Visible to `ps aux` + shell history — prefer `--token-file` or `HF_TOKEN` env var on shared hosts. |
| `--token-file` | unset | Path to a file containing the HF Hub token (mode-0600 recommended). Mutually exclusive with `--token`. |
| `--private` | off | Create the repo as private. |
| `--include-base` | off | Push merged model including the base weights (LoRA-only push by default). |
| `--verbose`, `-v` | off | Verbose logging during upload. |

## `backprop info`

Print Python / PyTorch / CUDA / GPU details and which optional features are available.

```bash
backprop info
```

No flags. Useful first command after install — confirms the env is wired up before you launch a long training.

## `backprop config`

View or modify configuration.

```bash
backprop config --show
backprop config --set MODEL__NAME=Qwen/Qwen2.5-7B-Instruct
backprop config --reset
```

| Flag | Default | Description |
|------|---------|-------------|
| `--show` | on | Show the current resolved configuration. |
| `--set KEY=VALUE` | unset | Set a configuration value. |
| `--reset` | off | Reset configuration to defaults. |

## `backprop ui`

Launch the Reflex (Radix UI) web interface.

```bash
backprop ui --port 7862
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port`, `-p` | `7862` | Port to bind (1..65535). |
| `--host` | `127.0.0.1` | Bind host. Non-loopback values (e.g. `0.0.0.0`, LAN IP) require `--auth user:pass` post-v1.2.0; otherwise the runtime exits `1` with `[RUNTIME_UI_AUTH_NOT_ENFORCED]`. **v1.3:** the value is now actually threaded to the Reflex backend via `--backend-host` (v1.1.0 → v1.2.x silently stayed loopback-only). |
| `--share` | off | Publish via a public tunnel. Requires `--auth user:pass` post-v1.2.0; otherwise the runtime exits `1` with `[RUNTIME_UI_AUTH_NOT_ENFORCED]`. The v1.2.0 FastAPI middleware enforces the credential on every request and the `/_event` WebSocket upgrade. **v1.3:** implemented as a real `cloudflared` tunnel (v1.1.0 → v1.2.x was a silent no-op). Requires `cloudflared` on `PATH` — install from <https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/>. The announced `https://*.trycloudflare.com` URL is added to the auth middleware's Host + Origin allowlist via `BACKPROPAGATE_UI_SHARE_HOST`. The CLI waits up to `BACKPROPAGATE_CLOUDFLARED_TIMEOUT` seconds (default `30`) for the URL to appear. |
| `--auth USER:PASS` | unset | Enable HTTP Basic auth on the Reflex UI. Required when `--share` or non-loopback `--host` is passed. Validated by `validate_auth_shape` — malformed values (missing colon, empty user or pass, etc.) exit `1` with `INPUT_AUTH_INVALID_SHAPE`. The credential flows into the Reflex subprocess via `BACKPROPAGATE_UI_AUTH`. Inline `--auth` lands in shell history — see `--auth-file` for the shell-history-safe alternative. |
| `--auth-file PATH` | unset | **v1.3+** — read `user:pass` from `PATH` instead of taking `--auth` on the command line. Same shape validation as `--auth`. Mutually exclusive with `--auth` (passing both exits `1` with `INPUT_AUTH_INVALID_SHAPE`). On POSIX, the file mode is checked: a mode wider than `0600` (group / other readable) emits a warning at startup. Create with `printf 'user:pass' > path && chmod 600 path`. Satisfies the `--share` / non-loopback `--host` gate. Recommended for repeat invocations and shell-history-sensitive deployments. See [recipes → --auth-file](/backpropagate/handbook/recipes/#use---auth-file-for-shell-history-safe-auth). |

### `--share` / `--host` require `--auth` (v1.2.0 contract)

The v1.2.0 FastAPI auth middleware (`backpropagate/ui_app/auth.py::basic_auth_transformer`, wired in `ui_app/app.py` via `rx.App(api_transformer=...)`) enforces credentials on every HTTP route and the `/_event` WebSocket upgrade. `--auth user:pass` flows through `validate_auth_shape` and into the Reflex subprocess via `BACKPROPAGATE_UI_AUTH`. The CLI rails refuse to start in these cases:

- `backprop ui --share` without `--auth` → exits `1` with `[RUNTIME_UI_AUTH_NOT_ENFORCED]`. A public URL with no credentials is the v1.1.x bug closed by [GHSA-f65r-h4g3-3h9h](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/GHSA-f65r-h4g3-3h9h) (CVSS 9.8, 2026-05-23).
- `backprop ui --host <non-loopback>` without `--auth` → same code (DNS-rebinding defense per CVE-2024-28224 / CVE-2025-49596 lineage).
- `backprop ui --auth user:pass` while the `[ui]` extra is degraded (`ENFORCEMENT_AVAILABLE=False`) → same code.

The refuse-to-start contract is enforced one layer deeper inside `ui_app/app.py` and `rxconfig.py`, so `python -m reflex run` from the package directory also refuses unless the legitimate `backprop ui` bridge has set its bypass env var.

**For remote access without a public URL, SSH port-forwarding stays the lower-friction option:**

```bash
ssh -L 7860:localhost:7860 you@gpu-host
# Then on your laptop, visit http://localhost:7860
```

SSH already handles auth, encryption, and audit — the runtime stays bound to `127.0.0.1` on the remote box and only your forwarded tunnel can reach it.

Filesystem writes are sandboxed to `BACKPROPAGATE_UI__OUTPUT_DIR` — see [Environment variables → UI sandbox](/backpropagate/handbook/env-vars/#ui-sandbox-fb-003) for the denylist (refuses `/etc`, `/var/run`, `~/.ssh`, etc. with `UI_OUTPUT_DIR_FORBIDDEN`). Full chain in [the security page → Four-layer defense in depth](/backpropagate/handbook/security/#four-layer-defense-in-depth).

## `backprop validate` (v1.3)

Pre-flight a dataset's format + content **before** kicking off a multi-hour training run. Thin wrapper over `backpropagate.datasets.validate_dataset`.

```bash
backprop validate my_data.jsonl
backprop validate my_data.jsonl --format sharegpt --max-errors 50
```

| Flag | Default | Description |
|------|---------|-------------|
| `dataset` (positional) | **required** | Path to the JSONL dataset to validate. |
| `--format` | `auto` | Format hint: `auto` / `sharegpt` / `alpaca` / `openai` / `raw`. Default auto-detects from the first row. |
| `--max-errors` | `100` | Maximum errors to collect before stopping (saves time on a thoroughly-broken dataset). |
| `--max-samples` | unset (all) | Maximum samples to validate (caps reads at `max_samples * 2` so a 100 GB file doesn't OOM during validation). |

Exit codes: `0` on clean validation, `1` on input problem (missing file / unreadable / bad encoding), `65` (`EX_DATAERR`) on detected validation errors.

## `backprop estimate-vram` (v1.3)

Print a small table of recommended batch sizes given the currently-visible GPU VRAM (or an override) and a model name. The math mirrors `Trainer._detect_batch_size`'s tier heuristic — the same one that fires automatically when `--batch-size auto` is used.

```bash
backprop estimate-vram                                       # uses the local GPU
backprop estimate-vram Qwen/Qwen2.5-7B-Instruct
backprop estimate-vram --vram-gb 24                          # simulate a 24 GB card
backprop estimate-vram --json                                # machine-readable
```

| Flag | Default | Description |
|------|---------|-------------|
| `model` (positional) | `Qwen/Qwen2.5-7B-Instruct` | Model name (used only for the printed header — VRAM tiers are model-agnostic). |
| `--vram-gb` | unset (auto-detect) | Override the detected VRAM (in GB) so you can simulate the table for a card you don't currently have. Default: query the primary CUDA device. Range: `(0, 512]`. |
| `--json` | off | Emit the table as JSON for CI / scripting consumers. |

Useful before starting a long training run on a card you haven't profiled, or while sizing infra spend.

## See also

- [Error codes](/backpropagate/handbook/error-codes/) — what each structured code means.
- [Environment variables](/backpropagate/handbook/env-vars/) — every `BACKPROPAGATE_*` override.
- [Troubleshooting](/backpropagate/handbook/troubleshooting/) — symptoms-first index.
