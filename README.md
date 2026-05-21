<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/backpropagate/readme.png" alt="Backpropagate" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/backpropagate/"><img src="https://img.shields.io/pypi/v/backpropagate" alt="PyPI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/backpropagate"><img src="https://img.shields.io/codecov/c/github/mcp-tool-shop-org/backpropagate" alt="Coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**Headless LLM fine-tuning in 3 lines. Smart defaults, VRAM-aware batch sizing, multi-run SLAO, and one-click GGUF export for Ollama.**

*SLAO is Single LoRA Continual Learning via Asymmetric Merging — the merge-between-runs technique that prevents catastrophic forgetting in extended fine-tuning campaigns ([paper](https://arxiv.org/abs/2512.23017)).*

*Train LLMs in 3 lines of code. Export to Ollama in one more.*

## Quick Start

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("examples/quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

The repo ships a small `examples/quickstart.jsonl` (5 ShareGPT-format examples) so the snippet above runs end-to-end on a clean install. For your own training, see [Dataset Format](#dataset-format) below.

### No-code path: Web UI

Prefer a UI to a Python REPL? Install the same extra and run:

```bash
pip install backpropagate[standard]
backprop ui --port 7862
```

The Reflex (Radix UI) interface lets you point at a JSONL file, pick a model, train, and export — no Python required. The UI is local-first; for public-internet exposure see [Web UI](#web-ui) below for the `--share` + `--auth` security contract and supported tunnel options (Cloudflare Tunnel, ngrok).

## Dataset Format

Your JSONL training file should have one example per line. The simplest format is ShareGPT chat:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca (`instruction`/`output`), OpenAI chat (`messages`), and raw text formats are also supported. See `examples/quickstart.jsonl` for a copyable starting point.

## Why Backpropagate?

| Problem | Solution |
|---------|----------|
| Fine-tuning is complex | 3 lines: load, train, save |
| Windows is a nightmare | First-class Windows support |
| VRAM management is hard | Auto batch sizing, GPU monitoring |
| Model export is confusing | One-click GGUF + Ollama registration |
| Long runs cause forgetting | Multi-run SLAO training |

## Key Features

- **Headless by Design**: Built for CI/CD pipelines, automated workflows, and programmatic execution.
- **Smart Defaults**: Automatically configures optimal hyperparameters based on your hardware and dataset.
- **Multi-Run SLAO Training**: Advanced training strategies to prevent catastrophic forgetting during long runs.
- **First-Class Windows Support**: Tested and optimized for Windows environments, avoiding common PyTorch/CUDA pitfalls.
- **Seamless Export**: One-click export to GGUF format and automatic registration with Ollama.
- **Modular Architecture**: Install only the dependencies you need (e.g., `[unsloth]`, `[ui]`, `[export]`).

## Installation

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Reflex (Radix UI) web interface
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Extra | Description | Dependencies |
|-------|-------------|--------------|
| `unsloth` | 2x faster training, 50% less VRAM | unsloth |
| `ui` | Reflex (Radix UI) web interface | reflex>=0.9.2, fastapi>=0.115 |
| `validation` | Pydantic config validation | pydantic, pydantic-settings |
| `export` | GGUF export for Ollama | llama-cpp-python |
| `monitoring` | WandB + system monitoring (auto-wired into trainer in v1.1.0) | wandb, psutil |
| `observability` | OpenTelemetry tracing | opentelemetry-api, opentelemetry-sdk |
| `logging` | Structured logging | structlog |
| `security` | JWT auth + token generation | PyJWT, cryptography |
| `production` | unsloth + ui + validation + logging + security | (bundle) |

**Requirements:** Python 3.10+ · CUDA GPU (8GB+ VRAM) · PyTorch 2.0+

### Platform prerequisites

Backpropagate handles the runtime quirks (multiprocessing, xformers on RTX 40/50, dataloader workers on Windows). It does **not** handle the install-time platform pain — fix those first:

- **CUDA toolkit version.** PyTorch is published per-CUDA — picking the wrong wheel silently installs CPU-only torch. Use the picker at <https://pytorch.org/get-started/locally/> for the exact `pip install torch ...` command for your driver. Run `nvidia-smi` to see your driver / CUDA version.
- **Windows.** Visual Studio Build Tools (C++) and CMake are required for the `[export]` extra (`llama-cpp-python` builds from source). `bitsandbytes` wheel is published for Windows natively now (>= 0.43); older guides mentioning `bitsandbytes-windows` are stale.
- **macOS.** GPU training is **not supported** — no CUDA. You can install Backpropagate to run *inference* on an exported GGUF via Ollama, but `trainer.train()` raises `DEP_GPU_NOT_AVAILABLE`. Use a CUDA machine for training.
- **Linux.** Most distros work out of the box. If you're using the PyPI binary release, note that the Linux build uses CPU-only torch (to stay under GitHub's 2 GB release-asset cap); install with the matching CUDA wheel from pytorch.org first.

For the long-form install troubleshooting, see [the troubleshooting handbook page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/).

## Configuration

All settings can be overridden with environment variables using the `BACKPROPAGATE_` prefix (e.g., `BACKPROPAGATE_LOG_LEVEL=debug`). A `.env` file in the project root is loaded automatically when the `[validation]` extra is installed.

Common knobs (see [the full env-vars reference](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/) for everything):

| Variable | Default | Notes |
|----------|---------|-------|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Force JSON (`true`) or console (`false`) logs |
| `BACKPROPAGATE_LOG_FILE` | unset | Path to mirror logs into |
| `BACKPROPAGATE_DEFER_FEATURE_DETECTION` | unset | Skip optional-dep detection at startup for the fastest CLI cold start |
| `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE` | `true` | When `true`, refuses `backprop ui --share` without `--auth` |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sandbox base for all UI filesystem writes; denylist-validated |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Default model |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Learning rate |
| `BACKPROPAGATE_LORA__R` | `16` | LoRA rank |

Nested keys use double underscore as the delimiter (Pydantic `env_nested_delimiter` convention).

## Usage

### Basic Training

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

`Qwen/Qwen2.5-7B-Instruct` is the canonical default — the value `Trainer()` resolves when called with no model argument (see [`config.py`](backpropagate/config.py) `ModelConfig.name`). Older examples pinned the pre-quantized `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`; we switched the default to the official Qwen weights for better reliability ([CHANGELOG v0.1.3](CHANGELOG.md)). Either model works.

### Multi-Run SLAO Training

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")

result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
    merge_mode="slao",  # Single LoRA Continual Learning via Asymmetric Merging
)
```

SLAO (Single LoRA Continual Learning via Asymmetric Merging) implements the [Merge before Forget](https://arxiv.org/abs/2512.23017) paper: orthogonal A-matrix init via QR decomposition, asymmetric A/B handling, and time-aware `λ(i) = 1/√i` scaling. The CLI flag is `--samples` (the underlying field is `samples_per_run`).

### Export to Ollama

```python
# Export to GGUF
result = trainer.export("gguf", quantization="q4_k_m")

# Register with Ollama separately
from backpropagate import register_with_ollama
register_with_ollama(result.path, "my-finetuned-model")
# ollama run my-finetuned-model
```

### CLI

```bash
backprop train --data my_data.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backprop ui --port 7862
backprop info
backprop list-runs                              # v1.1.0: query past training runs
backprop show-run <run-id>                      # v1.1.0: detail view
backprop resume <run-id>                        # v1.1.0: resume a crashed multi-run
backprop push ./output/lora --repo me/my-model  # v1.1.0: push adapter to HF Hub
```

See the [CLI reference](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) for every subcommand and flag, or run `backprop <subcommand> --help`.

### Resume from checkpoint (v1.1.0)

A 5-run multi-run that crashes at run 4 is now recoverable. Every multi-run session writes its run_id into both `run_history.json` and the on-disk checkpoint manifest, so picking up where you left off is one command:

```bash
backprop resume <run-id>                       # picks up the in-progress session
backprop multi-run --data ... --resume <run-id> # explicit form
backprop train --data ... --resume <run-id>    # single-run resume (continues run_id)
```

The default behavior of `backprop multi-run` (no `--resume`) auto-detects an in-progress entry for the same output directory and continues it. Pass `resume_from="off"` (Python API) or omit `--resume` and start in a fresh output dir to force a clean session.

When a multi-run resumes, the latest checkpoint for that run_id is loaded into the model, the SLAO merger state is restored from `slao/` next to the checkpoint, and the run loop continues from `last_completed_run + 1`. The history entry's `status` flips back to `running` so `backprop list-runs --status running` shows the live session.

### Experiment tracking (v1.1.0)

`Trainer` auto-detects installed experiment trackers (`wandb`, `tensorboard`, `mlflow`) and wires them into the underlying `transformers.TrainingArguments`. The default `report_to="auto"` picks up whatever's importable:

```bash
pip install backpropagate[monitoring]  # installs wandb + psutil
wandb login                            # one-time
backprop train --data my_data.jsonl    # W&B run gets the same run_id prefix as the on-disk history
```

Override with `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])`, or `Trainer(report_to="none")` to opt out explicitly. For MLflow add `pip install mlflow`; for TensorBoard add `pip install tensorboard`. The W&B run name is `backprop-<run_id_prefix>` so an operator can grep across W&B, our logs, and `run_history.json` by the same identifier.

### Training history

Every `backprop train` and `backprop multi-run` invocation records a row in `<output>/run_history.json` with the run_id, model, dataset, hyperparameters, status, final loss, loss history, and (for multi-run) the SLAO merge timeline. List recent runs:

```bash
backprop list-runs                         # most recent 20 runs, all statuses
backprop list-runs --status failed         # filter
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial run_id ok)
```

Run history survives across processes — the `Runs` tab in the web UI is a separate, in-memory view; the on-disk history is the source of truth for `list-runs` / `show-run` / `resume`.

### Web UI

Launch the Reflex interface locally:

```bash
backprop ui --port 7862
```

To expose a public-internet URL, you must pair `--share` with `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` without `--auth` exits with code `1` and the structured error `[INPUT_AUTH_REQUIRED]`. The rationale: `--share` publishes a `*.gradio.live` URL that anyone on the internet can hit, and without auth that means anyone can drive your training pipeline.

To explicitly opt out (e.g. an internal dev environment), set the env var `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false`. A loud warning will print on every launch — and there's a 5-second grace period before the unauth'd UI binds, so you can `Ctrl-C` if it looks wrong.

Filesystem writes from the UI are sandboxed to a single directory:

- Default: `~/.backpropagate/ui-outputs`
- Override: `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- The override is **denylist-validated** — system / credential paths (`/etc`, `/var`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) are refused with `[UI_OUTPUT_DIR_FORBIDDEN]`.

## Windows Support

Backpropagate is designed to work on Windows out of the box:

- Pre-tokenization to avoid multiprocessing crashes
- Automatic xformers disable for RTX 40/50 series
- Safe dataloader settings
- Tested on RTX 5080 (16GB VRAM)

## Model Presets

| Preset | VRAM | Speed | Quality |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12GB | Medium | Best |
| Qwen 2.5 3B | ~8GB | Fast | Good |
| Llama 3.2 3B | ~8GB | Fast | Good |
| Llama 3.2 1B | ~6GB | Fastest | Basic |
| Mistral 7B | ~12GB | Medium | Good |

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
├── ui_theme.py          # Radix theme tokens + CSS (Reflex era)
├── ui_state.py          # rx.State subclasses
├── ui_app/              # Reflex web interface (Radix UI)
│   ├── app.py           #   rx.App entry point
│   ├── chrome.py        #   Header / LeftNav / SideRail / Footer
│   ├── pages/           #   Train / Multi-Run / Export / Dataset
│   └── components/      #   Bp* primitives (status pill, sparkline, event log…)
├── ui_security.py       # Rate limiting, CSRF, file validation (framework-agnostic)
├── ui_gradio_legacy.py  # DEPRECATED — preserved as v1.0 reference; removed in v1.2
└── theme_gradio_legacy.py  # DEPRECATED — same
```

## Troubleshooting

A short index of the most common first-run failures. The full reverse index lives at [the troubleshooting handbook page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/); every code below is documented at [error codes](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/).

| Symptom | Code | Fix |
|---------|------|-----|
| GPU runs out of memory mid-training | `RUNTIME_GPU_OOM` | OOM auto-recovery (B-002) halves batch size up to 3 times automatically. To opt out: `Trainer(oom_recovery=False)`. To force smaller: `--batch-size 1`. |
| HF Hub returns 401 / "model not found" | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` and re-try. For typos, copy the exact id from <https://huggingface.co/models>. |
| Bad model name typo | `INPUT_VALIDATION_FAILED` or `DEP_MODEL_LOAD_FAILED` | Verify the `org/name` identifier at <https://huggingface.co/models>. |
| `register_with_ollama` connection refused | `DEP_OLLAMA_REGISTRATION_FAILED` | Start the daemon: `ollama serve`. Install from <https://ollama.com>. Retryable. |
| Disk full during checkpoint save | `STATE_CHECKPOINT_INVALID` | Atomic writes leave a `.partial` directory on crash — safe to delete. Previous good checkpoint is intact. |
| Training paused / aborted on GPU overheat | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | B-003 monitor pauses on NVML temp threshold; resumes automatically as the GPU cools. Improve airflow or lower sustained load. |
| `backprop ui --share` rejected | `INPUT_AUTH_REQUIRED` | Pass `--auth user:password`, or set `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false` to opt out (loud warning). |
| Multi-run "validation overlap" | `CONFIG_INVALID` (Stage A backend B-001) | Lower `--samples` below the training-pool size, increase dataset, or disable validation. |
| GGUF export failed on first try | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; on Windows you also need Visual C++ Build Tools + CMake. |

## Reporting bugs

When something fails, Backpropagate prints a `run_started run_id=<uuid>` line at startup and binds the same id to checkpoint manifests, SLAO merge history, and structured log lines. Include the `run_id` in any bug report — it lets a maintainer correlate every log line, every checkpoint, and every merge for that exact run.

A good bug report includes:

1. **`run_id`** — the uuid printed at startup (also available as `TrainingRun.run_id` and `RunResult.run_id`).
2. **The error code** — the `[CODE_NAME]: message` line in stderr is what to grep for; see [error codes](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) for the catalog.
3. **The redacted command line.** Stderr in non-verbose mode is automatically redacted (Bearer tokens, `sk-*`, `hf_*`, AWS keys, `password=`/`token=`/`api_key=` pairs are scrubbed) — safe to paste. For the full unredacted traceback, re-run with `--verbose`, but review before posting.
4. **Python / PyTorch versions, GPU model, OS.** `backprop info` prints all of this in one go.

## Privacy

All training happens locally on your GPU. Backpropagate makes no network requests except to download models from HuggingFace (which you initiate). No telemetry, no cloud dependency.

## Scorecard

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | 6/8 | SECURITY.md, trust model, no secrets/telemetry, safe_path(). MCP items skipped |
| B. Error Handling | 5/7 | Structured exception shape (`code`/`message`/`hint`/`cause`/`retryable`) via ERROR_CODES registry; CLI exit codes 0/1/2/3; no raw stack traces without `--verbose`; `run_id` correlation; redacted stderr; `--share`+`--auth` gating. MCP/desktop/vscode skipped. |
| C. Operator Docs | 4/7 | README, CHANGELOG, LICENSE, --help. Logging/MCP/complex skipped |
| D. Shipping Hygiene | 6/9 | verify.sh, version=tag, 5 scanners in CI, dependabot, python_requires, clean build |
| E. Identity | 4/4 | Logo, translations, landing page, metadata |
| **Total** | **25/31** | 14 items skipped with justification · `shipcheck audit` passes 100% · Audit date: 2026-05-21 (B-row re-graded after Stage B + Stage A CLI exit-code work) |

Design history and what each line item maps to: see [ROADMAP.md](ROADMAP.md) — all Week 1–4 items are shipped in v1.1.0.

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
