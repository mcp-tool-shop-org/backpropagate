---
title: CLI reference
description: Every backprop subcommand and flag.
sidebar:
  order: 8
---

The `backprop` CLI exposes six subcommands. Run `backprop <subcommand> --help` for the canonical flag list from argparse; this page mirrors that with descriptions and defaults.

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
| `--model`, `-m` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` * | Model name (HF id or local path). |
| `--data`, `-d` | **required** | Dataset path (JSONL/CSV) or HuggingFace dataset name. |
| `--steps` | `100` | Number of training steps (must be > 0). |
| `--samples` | unset | Maximum samples to use from the dataset (must be > 0). |
| `--batch-size` | `auto` | Per-device batch size. `auto` queries GPU VRAM. |
| `--lr` | `2e-4` | Learning rate (must be > 0). |
| `--lora-r` | `16` | LoRA rank (must be > 0). |
| `--output`, `-o` | `./output` | Output directory. |
| `--no-unsloth` | off | Disable Unsloth even if available. |

\* CLI defaults to the pre-quantized `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`. The Python `Trainer()` (no args) resolves to the official `Qwen/Qwen2.5-7B-Instruct` from `config.py` `ModelConfig.name`. Both work; the CLI default is a faster first-run because the weights are already 4-bit. A future patch may align the CLI default with the Python default — coordinate via the issue tracker.

## `backprop multi-run`

Train with multiple short runs and SLAO merging between them.

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` * | Model name. (Same CLI/Python-default divergence as `backprop train` — see note above.) |
| `--data`, `-d` | **required** | Dataset path or HF name. |
| `--runs` | `5` | Number of training runs. |
| `--steps` | `100` | Steps per run. |
| `--samples` | `1000` | Samples per run. (Note: the existing error message text suggests `--samples-per-run` — the actual flag is `--samples`.) |
| `--merge-mode` | `slao` | One of `slao` / `simple`. |
| `--output`, `-o` | `./output` | Output directory. |

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
| `--host` | `127.0.0.1` | Bind host. Non-loopback values (e.g. `0.0.0.0`, LAN IP) require `--auth user:pass` post-v1.2.0; otherwise the runtime exits `1` with `[RUNTIME_UI_AUTH_NOT_ENFORCED]`. |
| `--share` | off | Publish via a public tunnel. Requires `--auth user:pass` post-v1.2.0; otherwise the runtime exits `1` with `[RUNTIME_UI_AUTH_NOT_ENFORCED]`. The v1.2.0 FastAPI middleware enforces the credential on every request and the `/_event` WebSocket upgrade. |
| `--auth USER:PASS` | unset | Enable HTTP Basic auth on the Reflex UI. Required when `--share` or non-loopback `--host` is passed. Validated by `validate_auth_shape` — malformed values (missing colon, empty user or pass, etc.) exit `1` with `INPUT_AUTH_INVALID_SHAPE`. The credential flows into the Reflex subprocess via `BACKPROPAGATE_UI_AUTH`. |

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

## See also

- [Error codes](/backpropagate/handbook/error-codes/) — what each structured code means.
- [Environment variables](/backpropagate/handbook/env-vars/) — every `BACKPROPAGATE_*` override.
- [Troubleshooting](/backpropagate/handbook/troubleshooting/) — symptoms-first index.
