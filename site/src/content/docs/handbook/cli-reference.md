---
title: CLI reference
description: Every backprop subcommand and flag.
sidebar:
  order: 8
---

The `backprop` CLI exposes the subcommands below — every flag, every default, and the full exit-code contract. Run `backprop --help` for the complete subcommand list and `backprop <subcommand> --help` for the canonical argparse output of a specific subcommand. A few subcommands (`resume`, `list-runs`, `show-run`, `runs`, `info`, `config`) get mentioned here in passing and are fully covered by `--help` rather than getting a dedicated table.

## Root-parser flags (apply to every subcommand)

These flags are accepted on the root `backprop` parser, BEFORE the subcommand name. They take precedence over the matching `BACKPROPAGATE_LOG_*` env vars when both are set.

| Flag | Default | Description |
|------|---------|-------------|
| `--verbose`, `-v` | off | Show verbose output including unredacted stack traces. Same as `BACKPROPAGATE_DEBUG=1`. |
| `--version`, `-V` | n/a | Print version string and exit. |
| `--log-level` | `INFO` | **v1.4** — one of `DEBUG` / `INFO` / `WARNING` / `ERROR`. Overrides `BACKPROPAGATE_LOG_LEVEL` for this invocation. CLI flag wins when both are set. |
| `--log-format` | `console` | **v1.4** — one of `json` / `console`. `json` emits one JSON object per log record (for log aggregators / `jq` consumers); `console` emits the human-readable structlog rendering. Overrides `BACKPROPAGATE_LOG_JSON`. |
| `--log-file` | unset (stderr only) | **v1.4** — write logs to `PATH` in addition to stderr. Parent directory created if missing. Overrides `BACKPROPAGATE_LOG_FILE`. |

Example: `backprop --log-level=DEBUG --log-format=json train --data my.jsonl --steps 100 2> session.log` emits structured JSON log records to stderr for the whole training run, captured via shell redirection.

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
| `--mode` | `lora` | **v1.4** — one of `lora` / `full`. `lora` (the default) trains a low-rank adapter; `full` updates every weight of the base model. `full` is supported only for models up to 3B parameters on consumer 16GB cards — picking `--mode=full` with a >3B model exits `2` with `[RUNTIME_FULL_FT_MODEL_TOO_LARGE]` naming `--mode=lora` + the three sub-3B presets (Phi-4-mini-3.8B / Qwen-3.5-4B / SmolLM3-3B) as the recovery options. See [full fine-tuning →](/backpropagate/handbook/full-fine-tuning/) for the LoRA-vs-full quality math and the Biderman 2024 / Thinking Machines 2025 citations. |
| `--method` | `sft` | **v1.5** — one of `sft` / `orpo`. `sft` (the default) is supervised fine-tuning. `orpo` is reference-free preference tuning (Hong, Lee & Thorne 2024) — needs a `{chosen, rejected}` (or `{prompt, chosen, rejected}`) dataset, runs single-stage with no reference model, and so stays in the same VRAM envelope as SFT. ORPO supports `--mode lora` only in v1.5. |
| `--orpo-beta` | `0.1` | **v1.5** — ORPO odds-ratio weight (the `lambda` / `beta` in TRL's `ORPOConfig`). Keep > 0. Ignored unless `--method orpo`. |
| `--fp8` | off | **v1.5 (experimental)** — FP8 compute path on Blackwell/Hopper (sm_90+) via torchao. Base weights in float8 (~1.4× throughput, ~60% less base memory); the LoRA adapter stays bf16 and the merge still works. `--mode lora` + `--method sft` only in v1.5; falls back to bf16 with a warning if unsupported (a broken torchao install raises `RUNTIME_FP8_UNSUPPORTED`). Needs `pip install 'backpropagate[fp8]'`. Sets `BACKPROPAGATE_TRAINING__FP8`. |
| `--use-rslora` | off | **v1.5** — rank-stabilized LoRA scaling (`alpha/sqrt(r)` instead of `alpha/r`). Zero inference cost, still mergeable; the benefit grows with rank (relevant at the rank-256 default). Sets `BACKPROPAGATE_LORA__USE_RSLORA`. |
| `--reasoning-trace` | off | **v1.5 T3.2** — reasoning-trace SFT (R1/QwQ distillation). Keeps the `<think>` chain-of-thought in the SFT target, drops empty / over-long traces (trace-length filtering), and raises the default `max_seq_length` to `8192` (set `BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH` to override). `<think>` is plain text — no special tokens, no embedding resize — so the merged GGUF still exports to Ollama. SFT only (`--method sft`). Sets `BACKPROPAGATE_DATA__REASONING_TRACE`; tune the band with `BACKPROPAGATE_DATA__MIN_TRACE_TOKENS` / `BACKPROPAGATE_DATA__MAX_TRACE_TOKENS`. |
| `--backend` | `auto` | **v1.5 T3.1 (experimental — Apple-Silicon rail BUILT-BUT-UNVERIFIED)** — one of `auto` / `cuda` / `mlx`. The compute rail. `auto` (the default) routes to CUDA on an NVIDIA host and to the MLX rail on an Apple-Silicon Mac with the `[mlx]` extra installed — so existing CUDA rigs are byte-identical. `cuda` forces the CUDA rail; `mlx` forces the Apple-Silicon (`mlx_lm.lora`) rail. **MLX is LoRA SFT only in v1.5** — `--method orpo`, `--mode full`, `--fp8`, and `multi-run` are not supported on the MLX rail and are rejected at construction with `CONFIG_INVALID_SETTING`. Forcing `--backend mlx` on a non-Apple host is unrunnable (`mlx-lm` is macOS + arm64 only) and errors with `CONFIG_INVALID_SETTING`; if the resolved rail is `mlx` but `mlx_lm` is missing the run raises `DEP_MLX_UNAVAILABLE` — install `pip install 'backpropagate[mlx]'`. Sets `BACKPROPAGATE_TRAINING__BACKEND`. The MLX rail is built + unit-tested (mocked) but not yet dogfood-verified on real Apple Silicon as of v1.5 — see the [README MLX note](https://github.com/mcp-tool-shop-org/backpropagate#apple-silicon-mlx--experimental-v15). |

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
| `--mode` | `lora` | **v1.4** — same `lora` / `full` semantics as `backprop train`. The mode applies to every run in the multi-run loop. Same 3B parameter ceiling + `RUNTIME_FULL_FT_MODEL_TOO_LARGE` gate. |
| `--merge-strategy` | `qiao_mahdavi` | **v1.5** — per-tensor LoRA merge rule. `qiao_mahdavi` (default) = the v1.4 SLAO merge (behavior-preserving). `linear` = plain weighted average. `ties` = trim + elect-sign + disjoint-merge (uses `--ties-trim`). `dare` = Bernoulli-drop + rescale (uses `--dare-drop-rate`). All stay mergeable with zero inference cost. Applies only with `--merge-mode slao`. |
| `--ties-trim` | `0.2` | **v1.5** — TIES trim quantile in `[0, 1]` (fraction of lowest-magnitude delta params zeroed before the sign election; `0.2` = bottom 20%). Only used with `--merge-strategy ties`. |
| `--dare-drop-rate` | `0.5` | **v1.5** — DARE Bernoulli drop probability in `[0, 1)` (fraction of delta params randomly dropped; survivors rescaled by `1/(1-p)`). Only used with `--merge-strategy dare`. |
| `--dare-seed` | unset | **v1.5** — seed for DARE's local RNG (deterministic drop mask). Default derives from the run. Only used with `--merge-strategy dare`. |
| `--drift-gate` | off | **v1.5** — enable the drift gate: skip a merge whose LoRA-B cosine similarity to the running accumulator falls below `--drift-threshold`, keeping the run as a sibling branch instead of corrupting the merge. |
| `--drift-threshold` | `0.0` | **v1.5** — cosine-similarity floor for the drift gate (range `[-1, 1]`). `similarity < threshold ⇒ branch`. Only consulted when `--drift-gate` is set. |
| `--eval-gate` | off | **v1.5** — enable the eval gate: after each candidate merge, evaluate held-out loss and reject (restore the pre-merge accumulator) if the merge regressed loss past `--eval-max-regression`. Reuses the v1.5 eval seam. |
| `--eval-max-regression` | `0.0` | **v1.5** — tolerated held-out-loss increase for the eval gate (`0.0` = zero-tolerance). A merge whose after-loss exceeds before-loss by more than this is rejected (`RUNTIME_EVAL_GATE_REGRESSED`). Only consulted when `--eval-gate` is set. |
| `--eval-heldout` | unset | **v1.5** — held-out JSONL set for the eval gate. Default reuses the run's reserved last-10% holdout. Only consulted when `--eval-gate` is set. |

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
| `--override` | unset | Repeatable `KEY=VALUE`. Override a single hyperparameter (whitelist: `seed`, `learning_rate`, `lr`, `batch_size`, `gradient_accumulation`, `max_steps`, `steps`, `samples`, `lora_r`, `lora_alpha`, `lora_dropout`, `use_dora`, `packing`, `init_lora_weights`, `lora_preset`, `optim`, `mode`). Unknown keys fail loudly. |
| `--json` | off | **v1.4** — emit a JSON payload describing the replay (new `run_id`, the resolved config, each override applied) under a `schema_version` field. Useful for CI flows that need to chain on a previously-recorded run. |

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
| `--format`, `-f` | `lora` | One of `lora` / `merged` / `gguf` / `ollama-adapter`. **v1.5** `ollama-adapter` registers the LoRA adapter with Ollama UNMERGED on top of `--base-model` (a `FROM`+`ADAPTER` Modelfile + `ollama create`) — no model load, no merge, no quantization. Lighter than the merged-GGUF path and the building block for the adapter shelf (swap adapters on one base by tag). Requires `--base-model`. |
| `--quantization`, `-q` | `q4_k_m` | One of `f16` / `q8_0` / `q5_k_m` / `q4_k_m` / `q4_0` / `q2_k`. |
| `--output`, `-o` | unset | Output directory (defaults to `<model_path>/<format>`). |
| `--ollama` | off | After GGUF export, register with Ollama. |
| `--ollama-name` | unset | Name to use when registering with Ollama. |
| `--base-model` | unset | **v1.5** — base model for `--format ollama-adapter` (an Ollama model name like `llama3.2` or a base-GGUF path). **Required** with `ollama-adapter`; missing it errors with `INPUT_VALIDATION_FAILED` (exit 1). MUST match the adapter's origin base. Ignored by other formats. |
| `--adapter-tag` | unset | **v1.5** — adapter tag for the derived `<base>:<tag>` Ollama model name (`--format ollama-adapter`). Default: a sanitised adapter-dir basename. Lets you shelf sibling adapters on one base (`llama3.2:taskA` / `llama3.2:taskB`). Ignored by other formats. |
| `--hub-token` | unset | HuggingFace Hub token for `--push` flow. Visible to `ps aux` + shell history — prefer `--hub-token-file` or `HF_TOKEN` env var on shared hosts. |
| `--hub-token-file` | unset | Path to a file containing the HF Hub token (mode-0600 recommended). Mutually exclusive with `--hub-token`. Mirrors v1.3 `--auth-file` pattern for keeping credentials off argv. |

## `backprop ollama` (v1.4)

Register / list / unregister local Ollama models without re-exporting. v1.3.x's only path to Ollama was the one-shot `backprop export --ollama --ollama-name <name>` form — operators who had already exported a GGUF (or wanted to undo a registration) had to drop into the upstream `ollama` CLI directly.

```bash
# Register an already-exported GGUF (or LoRA adapter) with the local daemon
backprop ollama register ./output/lora.gguf --name my-finetune

# List currently-registered models (mirrors `ollama list`)
backprop ollama list

# Unregister a model (mirrors `ollama rm`)
backprop ollama rm my-finetune

# List the adapter shelf for a base (v1.5) — the <base>:<tag> variants
# produced by `backprop export --format ollama-adapter`
backprop ollama shelf llama3.2
```

| Verb | Positional / Flag | Description |
|------|-------------------|-------------|
| `backprop ollama register` | `path` (positional), `--name`, `--modelfile` | Register an already-exported GGUF (or a directory containing one) with the local daemon. |
| `backprop ollama list` | — | List currently-registered models (mirrors `ollama list`). |
| `backprop ollama rm` | `name` (positional) | Unregister a model (mirrors `ollama rm`). |
| `backprop ollama shelf` | `base_model` (positional) | **v1.5** — list the adapter variants registered against a base model: every Ollama `<base>:<tag>` model (produced by `backprop export --format ollama-adapter`) that shares the given base. The bare base itself is excluded; the shelf is the adapter VARIANTS you hot-swap by tag. Best-effort over `ollama list`. |

### Adapter shelf (v1.5)

The "adapter shelf" is the set of Ollama models that share one base and differ only by adapter tag — `llama3.2:taskA`, `llama3.2:taskB`, … — each produced by `backprop export --format ollama-adapter --base-model llama3.2 --adapter-tag taskA`. Because the adapter is applied UNMERGED (a `FROM`+`ADAPTER` Modelfile), the base weights are loaded once and only the small rank-`r` adapter differs per tag, so `ollama run llama3.2:taskA` / `ollama run llama3.2:taskB` swap behaviour without re-quantizing a full model each time. `backprop ollama shelf <base>` lists what's currently on a base's shelf.

### Architectural deviation note

`backprop ollama` is the one nested subparser in the otherwise-flat backpropagate CLI. The deviation is intentional and operator-facing: operators who already know `ollama create` / `ollama list` / `ollama rm` get the same grammar one prefix deeper, which matches their muscle memory rather than re-cutting the same operations as flat backpropagate subcommands. The existing `backprop export --ollama --ollama-name <name>` one-shot path stays untouched as the "I just trained, register in one command" surface; the new triad is for the "I already exported earlier, just register" case.

For maintainers: the nesting is the documented exception. Future Ollama-adjacent subcommands (e.g. `backprop ollama pull`, `backprop ollama push-to-registry`) extend the same nested group; non-Ollama subcommands stay flat under the root parser.

### `backprop ollama register`

| Flag | Default | Description |
|------|---------|-------------|
| `path` (positional) | **required** | Path to an already-exported GGUF file OR a LoRA adapter directory. When given a LoRA dir, the subcommand exports to GGUF q4_k_m as a side-effect (same code path as `backprop export --ollama`). |
| `--name` | derived from path basename | Name to register the model under in Ollama. Quoted in `ollama run <name>` afterwards. |
| `--modelfile` | unset | Path to a custom Ollama Modelfile. Default builds a minimal Modelfile from the GGUF path; pass `--modelfile <path>` to use your own template (system prompt, parameters, etc.). |

Exit codes: `0` on successful registration, `1` on user error (path doesn't exist, name collision with `--no-overwrite`), `2` on daemon failure (`DEP_OLLAMA_REGISTRATION_FAILED`).

### `backprop ollama list`

No flags. Enumerates the currently-registered model names by calling the Ollama daemon HTTP API at `localhost:11434`. Exit `0` on success, `2` on daemon unreachable.

### `backprop ollama rm <name>`

Unregister a model. Exit `0` on success, `1` if the model name doesn't exist in the daemon, `2` on daemon unreachable.

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
| `--hub-revision` | unset (default branch) | **v1.4** — push to a named branch / revision in the target repo. Plumbed through to `HfApi.upload_folder(revision=...)`. Useful for per-experiment branches or pushing to a `dev` branch for review-before-promote. |
| `--hub-commit-message` | `"Upload model"` | **v1.4** — override the default upload commit subject. Useful for tying a CI-pushed model to the workflow run that produced it (e.g. `--hub-commit-message "ci: $GITHUB_RUN_ID"`). |
| `--verbose`, `-v` | off | Verbose logging during upload. |

## `backprop info`

Print Python / PyTorch / CUDA / GPU details and which optional features are available.

```bash
backprop info
backprop info --json | jq .logging
backprop info --subcommand-tiers
```

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | Emit the full info payload as JSON under a `schema_version` field. v1.4 adds a `logging: {level, format, file, has_handler}` block surfacing the active log config (the new root-parser `--log-*` flags feed this). |
| `--subcommand-tiers` | off | **v1.4** — print the `SUBCOMMAND_TIERS` registry (which subcommands are `stable` / `experimental` / `deprecated-prefer-X`). Pair with `--json` to consume the table programmatically. |
| `--env-vars` | off | Print every `BACKPROPAGATE_*` env var the current process is reading. |
| `--error-codes` | off | Print the full `ERROR_CODES` catalog. |

Useful first command after install — confirms the env is wired up before you launch a long training. The `--json` surface makes `backprop info` a one-shot diagnostic for support tickets: `backprop info --json > info.json` captures Python / Torch / CUDA / GPU / logging / feature flags in a structured shape that's safe to attach.

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
| `--json` | off | **v1.4** — emit a JSON payload (`dataset`, `format_detected`, `total_samples`, `errors[]`) under a `schema_version` field instead of the human-readable banner. Useful for CI consumers and `jq` pipelines. |

Exit codes: `0` on clean validation, `1` on input problem (missing file / unreadable / bad encoding), `65` (`EX_DATAERR`) on detected validation errors.

## `backprop data report` (v1.5)

Dataset-quality report — duplicate clusters, length/format outliers, optional train/test contamination against a held-out split, format-validity, and a token-length distribution. Reads the JSONL file directly (no HuggingFace download, no torch), so it is safe to run as a pre-flight in CI. The advisory report becomes a **gate** when you pass any `--fail-*` flag (or `--strict`).

`data` is the second nested subparser in the otherwise-flat CLI (after `ollama`); the dataset-quality surface is grouped under a noun so future verbs (`data dedupe`, `data split`) extend the same group rather than sprinkling flat verbs at the root. `backprop validate` stays flat (it pre-dates this grouping); the new quality *report* lives under `data report` as the first of a family.

```bash
# Advisory report (human summary)
backprop data report my_data.jsonl

# Contamination check against a held-out split
backprop data report my_data.jsonl --against heldout.jsonl

# CI gate: fail if >10% near-duplicates or any contamination
backprop data report my_data.jsonl --fail-on-dups 0.1 --fail-on-contamination 0.0 --json
```

| Flag | Default | Description |
|------|---------|-------------|
| `dataset` (positional) | **required** | Path to the JSONL dataset to analyze. |
| `--format` | `auto` | Format hint: `auto` / `sharegpt` / `alpaca` / `openai` / `raw`. `auto` detects from the first row. |
| `--against` | unset | Path to a held-out JSONL split to check the main dataset for train/test contamination against. |
| `--max-samples` | unset (all) | Maximum samples to analyze. Caps reads on huge files. |
| `--dup-threshold` | `0.9` | Similarity threshold in `[0, 1]` above which two samples count as near-duplicates. Validated by `_unit_float`. |
| `--fail-on-dups` | unset | Gate: fail (exit `65`) if the near-duplicate **rate** exceeds this fraction in `[0, 1]` (e.g. `0.1` for 10%). Omit to stay advisory. |
| `--fail-on-contamination` | unset | Gate: fail (exit `65`) if the contamination rate against `--against` exceeds this fraction in `[0, 1]`. Omit to stay advisory. |
| `--max-outlier-rate` | unset | Gate: fail (exit `65`) if the length/format-outlier rate exceeds this fraction in `[0, 1]`. Omit to stay advisory. |
| `--strict` | off | Promote a WARN verdict to FAIL (exit `65`). For strict CI gates. |
| `--json` | off | Emit the report as JSON (`schema_version` + the `DataQualityReport` fields, including `verdict` and `failed_thresholds`) instead of the human summary. |

Exit codes: `0` advisory run or all gates passed / clean; `1` on bad input (dataset missing / is a directory / not UTF-8 / unreadable, or `--against` was passed but its file is missing); `65` (`EX_DATAERR`) when a `--fail-*` / `--strict` gate trips **or** the dataset has zero parseable rows. A tripped gate stamps `INPUT_DATASET_REPORT_THRESHOLD` into the structured log so the failure is greppable.

## `backprop eval` (v1.5)

Lightweight post-train eval harness — held-out loss + perplexity + N sample generations against a fixed prompt set, with an optional before/after diff (`--vs`) and an **eval-gate** (`--gate-against`) that backstops continual-merge / SLAO campaigns. Flat top-level subcommand, modeled on `diff-runs`. Imports the (torch-heavy) eval engine lazily, after the cheap run-resolution checks, so a typo'd run_id fails fast without loading a model.

```bash
# Evaluate one run
backprop eval <run_id> --output ./output

# Before/after diff between two runs
backprop eval <run_id> --vs <baseline_run_id>

# Gate: reject if <run_id> regressed the held-out metric vs the baseline
backprop eval <run_id> --gate-against <baseline_run_id> --max-regression 0.0
```

| Flag | Default | Description |
|------|---------|-------------|
| `run_id` (positional) | **required** | The run_id to evaluate (or any unambiguous prefix). |
| `--vs` | unset | Second run_id to evaluate and diff against (before/after comparison). If both `--vs` and `--gate-against` are passed, `--gate-against` wins. |
| `--gate-against` | unset | Baseline run_id to gate against. Evaluates both runs and runs the eval-gate; exits `65` if the evaluated run regressed beyond `--max-regression`. |
| `--output`, `-o` | `./output` | Output directory containing `run_history.json`. |
| `--heldout` | unset | Path to a held-out JSONL split for the held-out-loss metric. Default: the harness re-splits the run's recorded dataset (with a loud WARN that overlap is possible). |
| `--prompts` | unset | Path to a fixed prompt set (one prompt per line, or JSONL `{"prompt": ...}`). Default: the harness's built-in prompt set. |
| `--num-samples`, `-n` | `5` | Number of sample generations to produce. |
| `--max-new-tokens` | `128` | Max new tokens per sample generation. |
| `--max-regression` | `0.0` | Maximum tolerated regression for `--gate-against` (`0.0` = any regression rejects). Raise to allow a small regression. |
| `--seed` | `0` | Random seed for the re-split + generation determinism. |
| `--json` | off | Emit the eval outcome (single / diff / gate) as JSON under a `schema_version` field for CI consumers. |

Exit codes: `0` the eval ran (and, with `--gate-against`, the gate ACCEPTED); `1` run-not-found (eval target / `--vs` / `--gate-against`) or `--heldout` / `--prompts` could not be resolved / read; `65` (`EX_DATAERR`) when `--gate-against` tripped — the run regressed beyond `--max-regression` (stamps `RUNTIME_EVAL_GATE_REGRESSED`). An eval that crashes inside the engine surfaces via the catch-all: `RUNTIME_EVAL_FAILED` → `2`, CUDA OOM → `137`, Hub failure → `69`.

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
