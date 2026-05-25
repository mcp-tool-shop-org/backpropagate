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
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/backpropagate"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/backpropagate/badge" alt="OpenSSF Scorecard"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

# Train an adapter. Ship it to Ollama. Move on.

Backpropagate is a Python library for fine-tuning large language models on a single GPU. Three lines of code train a 7B model on a 16GB card. One more command exports it to Ollama so you can `ollama run` your finetune. Works first-class on Windows.

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")
```

```bash
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
ollama run my-model
```

That's it. There's no YAML config file. There's no `accelerate launch` ceremony. There's no separate "now convert it to GGUF" tutorial. If you have a CUDA GPU and a JSONL file with your training data, you're three lines away from a working finetune.

## Install

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

If you want the optional features, swap the install for one of these:

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

Prefer Docker? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` works too. Images ship for both `linux/amd64` and `linux/arm64`, so Apple Silicon and ARM Linux operators get a native image. A canonical `compose.yaml` for "UI in a container" lives at the repo root — `docker compose up` brings the web UI up on `http://localhost:7860` with a persistent `~/.backpropagate` volume mount.

## Where Backpropagate sits in the space

There are several good libraries for fine-tuning LLMs. They're each great at different things:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — if you like YAML configs and want a community of recipes to copy from
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — if you want a web GUI and built-in support for DPO/PPO/RLHF
- **[Unsloth](https://github.com/unslothai/unsloth)** — if you need the fastest possible training and you're on a supported model family
- **[torchtune](https://github.com/pytorch/torchtune)** — if you want Meta's first-party PyTorch-native recipes you can edit

Backpropagate is the missing option: **a 3-line Python API for solo operators on a single consumer GPU who want to train an adapter and ship it.** No YAML, no GUI, no DPO/PPO, no multi-node. Just the loop everyone actually needs and the export step that gets in the way.

If you tried one of the libraries above and bounced off the config-file ceremony, or hit a model-family gap, or wanted Windows-first defaults — Backpropagate is for you.

## What you can fine-tune on a 16GB consumer GPU

Here's the practical envelope on a 16GB card (RTX 4080 / 5080 / 4070 Ti Super):

| Model | Method | Status |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | Comfortable. Full sequence length, room to spare. |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | Standard. ~7-8 GB. Backpropagate's default presets. |
| Llama-3 13B | QLoRA + sample packing | Tight but works. Use shorter sequences. |
| Mixtral 8x7B (47B total parameters) | AQLM 2-bit + LoRA | Experimental in v1.4. The largest model you can touch on a 16GB card. |

For models 3B and smaller, full fine-tuning (not just LoRA) is feasible on 16GB and is planned as a `mode="full"` option for v1.4. For 7B+ models, full fine-tuning needs a 24GB+ GPU — consider an A100 cloud rental, or stick with LoRA, which recent research shows matches full fine-tuning quality on most post-training tasks anyway (see [the anti-pitch section](#what-backpropagate-is-not-for) for citations).

## What Backpropagate is NOT for

Honest scope helps everyone. Backpropagate doesn't do these things, and trying to make it do them would be a worse experience than reaching for the right tool:

- **Full-parameter fine-tuning of 7B+ models** — Backpropagate uses LoRA / QLoRA, which trains a small adapter rather than updating every weight. For models 7B and larger, full fine-tuning needs 24GB+ of GPU memory and doesn't fit on a 16GB consumer card. For models 3B and smaller, full fine-tuning IS feasible on 16GB; a `mode="full"` option is planned for v1.4. The bigger picture: recent research ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) shows that LoRA at correct configuration matches full fine-tuning quality on most post-training tasks (instruction-following, domain adaptation, persona/style) at 67% of the compute — so for the work most operators actually want, you don't lose anything by sticking with LoRA. If you genuinely need full fine-tuning of a 7B+ model, use HuggingFace `transformers.Trainer` directly on a 24GB+ card.
- **DPO / PPO / GRPO / preference tuning** — Backpropagate does single-stage supervised fine-tuning only. For preference learning, use TRL directly or LLaMA-Factory.
- **Multi-node training** — single GPU on one machine only. Multi-GPU on one machine works (via `accelerate launch`) but isn't officially supported.
- **macOS training** — Apple Silicon doesn't have CUDA, so training has to run on a Linux or Windows box with an NVIDIA GPU. You can still run the trained model on a Mac via Ollama.
- **Anything outside the tested model families** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Other models often work but aren't pinned in CI.

If you need any of those things, reach for one of the libraries listed above. They're better at them.

## What Backpropagate gives you

Four things, in one install:

**1. A real 3-line API that runs without a config file.**
The snippet at the top of this README runs end-to-end. No `accelerate config`, no YAML, no Hydra overrides. Just `Trainer(model).train(data)` and you have a finetune.

**2. Windows that actually works.**
Most ML libraries treat Windows like an afterthought. Backpropagate is tested first-class on Windows + RTX 5080. The library handles the runtime quirks for you — it knows how to pre-tokenize your data so Windows multiprocessing doesn't crash, it automatically disables xformers on RTX 40/50 cards where it would break, and it picks dataloader settings that don't blow up. You don't have to know any of this. It just runs.

**3. Built for unattended runs.**
Training takes hours. You don't want to babysit it. Backpropagate is designed to be left running:

- If you run out of GPU memory, it automatically halves the batch size and retries — up to three times. No hand-tuning.
- If your GPU gets too hot, it pauses until things cool down and then continues.
- Every checkpoint is written atomically — if your laptop crashes mid-save, the previous good checkpoint is still intact.
- Every training run gets a unique ID that's stamped onto every log line, every checkpoint, and every Weights & Biases entry. If something goes wrong, one ID lets a maintainer correlate everything.
- Errors come with stable codes (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, etc.) so you can grep your logs and the [troubleshooting guide](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) for the fix. CUDA-specific failures have a dedicated [CUDA troubleshooting page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

**4. One command from trained adapter to `ollama run`.**
Lots of libraries train a model. Few of them get out of your way when you want to actually use it. Backpropagate exports to GGUF (the format Ollama uses) and registers an Ollama model in one command. You go from "training done" to "I can chat with my finetune" in about 30 seconds.

## Quick Start

The repo ships a tiny example dataset so the snippet from the top of this README runs on a clean install:

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

This trains a Qwen 2.5 7B adapter on 5 short ShareGPT-format conversations, then exports the result to GGUF. For your own data, format your JSONL one example per line:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca (`instruction` / `output`), OpenAI chat (`messages`), and raw text formats also work — Backpropagate auto-detects the format.

For more end-to-end workflows (fine-tune-and-push-to-HF-Hub, resume after OOM, multi-run SLAO across a long campaign, etc.) see the [handbook recipes page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Web UI (optional)

If you'd rather click than type Python, install the UI extra and launch:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

A local web interface opens at `http://localhost:7862` where you can point at a dataset, pick a model, train, and export. The UI is local-only by default. To expose it to other devices, see [Web UI](#web-ui) below for the `--share` + `--auth` security contract.

## Multi-run training

If you want to fine-tune incrementally across multiple datasets — say you get new training data every week and want to add it without forgetting what you learned before — Backpropagate's `multi_run` mode is for you:

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")

result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
)
```

This runs five training passes, merging the adapter between runs in a way that preserves earlier knowledge while incorporating new examples. The technique is based on recent continual-learning research — see [References](#references) at the bottom of this README.

The CLI version:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Resume from checkpoint

A 5-run training that crashes at run 4 is recoverable. Every multi-run session writes its run ID into the on-disk history and checkpoint manifest, so picking up where you left off is one command:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

The default behavior of `backprop multi-run` (no `--resume`) auto-detects an in-progress entry in the same output directory and continues it. To force a clean start, point at a fresh output directory.

## Training history

Every `backprop train` and `backprop multi-run` invocation records a row in `<output>/run_history.json` — model used, dataset, hyperparameters, status, final loss, loss history. You can list and inspect past runs:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Experiment tracking

Backpropagate auto-detects installed experiment trackers (Weights & Biases, TensorBoard, MLflow) and wires them in. If `wandb` is installed and you're logged in, every run automatically logs to W&B with a run name that matches the on-disk run ID — so you can grep across W&B, your logs, and `run_history.json` using one identifier.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Override with `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])`, or `Trainer(report_to="none")` to opt out.

## Web UI

The Reflex web interface is opt-in — install with `pipx install "backpropagate[ui]"` and launch:

```bash
backprop ui --port 7862
```

The UI runs locally on `http://localhost:7862`. To expose it to other devices (other people on your network, a public URL, etc.) you must pair `--share` (or `--host`) with `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` without `--auth` exits with an error. The reason: `--share` publishes a URL anyone on the internet can reach, and without authentication that means anyone can drive your training pipeline and read your HuggingFace token. There is no opt-out for this — if you don't want to set credentials, use SSH port-forwarding instead:

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

See [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) for the full threat model.

Filesystem writes from the UI are sandboxed to a single directory:

- Default: `~/.backpropagate/ui-outputs`
- Override: set `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- The override is denylist-validated — system or credential paths (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) are refused

## Platform notes

**Requirements:** Python 3.10+ · CUDA GPU (8GB+ VRAM) · PyTorch 2.0+

Python 3.10 reaches upstream end-of-life in October 2026, and Backpropagate plans to drop 3.10 in v1.4. For new installs, prefer Python 3.11 or 3.12 — 3.11 is the most-tested floor.

Backpropagate handles the runtime quirks of training on different platforms, but it can't fix install-time problems. The two most common are:

- **Wrong CUDA wheel.** PyTorch is published one binary per CUDA version. If you pick the wrong one, you silently get CPU-only PyTorch and training is impossibly slow. Use the wheel picker at <https://pytorch.org/get-started/locally/> for your driver. Run `nvidia-smi` to see your driver / CUDA version.
- **Windows + GGUF export.** The `[export]` extra builds `llama-cpp-python` from source, which needs Visual Studio Build Tools (C++ component) and CMake.

**macOS:** GPU training is not supported (no CUDA). You can run the trained adapter on a Mac via Ollama, but `trainer.train()` raises `DEP_GPU_NOT_AVAILABLE`. Use a CUDA Linux or Windows machine for the training itself.

See the [troubleshooting handbook page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) for the long-form install fix-it guide, and the dedicated [CUDA troubleshooting page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) for driver / VRAM / xformers / bf16-vs-fp16 issues.

## CLI

Every Python API has a CLI mirror:

```bash
backprop train --data my_data.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backprop ui --port 7862
backprop info                          # environment + version snapshot
backprop list-runs                     # past training runs
backprop show-run <run-id>             # detail view
backprop resume <run-id>               # resume a crashed run
backprop push ./output/lora --repo me/my-model    # push adapter to HuggingFace Hub
backprop diff-runs <run-a> <run-b>     # diff two runs side by side
backprop replay <run-id>               # re-run with same config / dataset
backprop export-runs --format jsonl    # bulk export run history
```

Full reference at [the CLI handbook page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), or `backprop <subcommand> --help`.

## Configuration

Every setting can be overridden with an environment variable using the `BACKPROPAGATE_` prefix:

| Variable | Default | Notes |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Force JSON or console logs |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Default model |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Learning rate |
| `BACKPROPAGATE_LORA__R` | `256` | LoRA rank (v1.3 default; pass `--lora-preset=fast` for the v1.2.x default of 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | UI filesystem sandbox |

Nested keys use double underscore (`MODEL__NAME`, not `MODEL_NAME`). The full reference is at [the env-vars handbook page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Model presets

| Preset | VRAM | License | Notes |
|---|---|---|---|
| Qwen-3.5-4B | ~8GB | Apache 2.0 | Recommended default for sub-5B. Best quality at this size. |
| Phi-4-mini-3.8B | ~8GB | MIT | Strong on reasoning / math / code. Strict license-clean. |
| SmolLM3-3B | ~6GB | Apache 2.0 | Fully open recipe. Native 64K context. |
| Qwen 2.5 7B | ~12GB | Apache 2.0 | Existing default. Best quality of the legacy 7B presets. |
| Qwen 2.5 3B | ~8GB | Qwen-Research | ⚠ research license — see Qwen license terms before commercial use. |
| Llama 3.2 3B | ~8GB | Llama Community | Solid alternative to Qwen 3B with permissive caveats. |
| Llama 3.2 1B | ~6GB | Llama Community | For quick experiments on small cards. |
| Mistral 7B | ~12GB | Apache 2.0 | Comparable to Qwen 7B, different chat template. |

Other models often work, but only these eight are pinned in CI. Pass `--lora-preset=quality` (default) for rank-256 / all-linear targets per Biderman 2024 + Thinking Machines 2025, or `--lora-preset=fast` for the legacy rank-16 / q+v target if you need the v1.2.x footprint.

## Troubleshooting

A short index of the most common first-run failures. The full reverse index is at [the troubleshooting handbook page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). For driver / VRAM / mixed-precision deep-dive see the [CUDA troubleshooting page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Symptom | Error code | Fix |
|---|---|---|
| GPU runs out of memory mid-training | `RUNTIME_GPU_OOM` | Automatic — Backpropagate halves the batch size and retries up to 3 times. To opt out: `Trainer(oom_recovery=False)`. To force smaller: `--batch-size 1`. |
| HuggingFace returns 401 / "model not found" | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` and retry. For typos, copy the exact ID from <https://huggingface.co/models>. |
| `register_with_ollama` connection refused | `DEP_OLLAMA_REGISTRATION_FAILED` | Start the daemon: `ollama serve`. Install from <https://ollama.com>. Retryable. |
| Disk full during checkpoint save | `STATE_CHECKPOINT_INVALID` | Atomic writes leave a `.partial` directory on crash — safe to delete. The previous good checkpoint is intact. |
| Training paused on GPU overheat | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automatic — Backpropagate pauses on the temperature threshold and resumes as the GPU cools. Improve airflow if it keeps happening. |
| `backprop ui --share` rejected | `INPUT_AUTH_REQUIRED` | Pass `--auth user:password`, or use SSH port-forwarding instead (see [Web UI](#web-ui)). |
| GGUF export failed on first try | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; on Windows you also need Visual C++ Build Tools + CMake. |

## Reporting bugs

When something fails, Backpropagate prints a line at startup like `run_started run_id=<uuid>` and binds the same ID to every log line, every checkpoint, and every Weights & Biases entry. **Include the `run_id` in any bug report** — it lets a maintainer correlate everything for that exact run.

A good bug report includes:

1. **The `run_id`** — the UUID printed at startup.
2. **The error code** — the `[CODE_NAME]: message` line in stderr. See [error codes](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) for the catalog.
3. **The redacted command line.** Stderr is automatically redacted (Bearer tokens, `sk-*`, `hf_*`, AWS keys, `password=` / `token=` pairs are scrubbed) — safe to paste. For the full unredacted traceback, re-run with `--verbose`, but review before posting.
4. **Python / PyTorch versions, GPU model, OS.** `backprop info` prints all of this in one go.

Questions, ideas, or "is this expected" threads belong in [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Security issues should be reported privately via the [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) form — see [SECURITY.md](SECURITY.md) for the policy.

## Privacy

All training happens locally on your GPU. Backpropagate makes no network requests except to download models from HuggingFace (which you initiate). No telemetry, no cloud dependency.

## References

Backpropagate's defaults and multi-run training mode are built on recent research. If you're interested in the underlying techniques:

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — the foundational paper introducing LoRA, which is how Backpropagate trains adapters efficiently.
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — empirical evidence that LoRA at rank 256 with all-linear targets matches full fine-tuning quality on most post-training tasks at 67% of the compute. Drives Backpropagate's v1.3 default LoRA configuration.
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — the practical follow-up identifying the 10× learning-rate-vs-full-FT correction needed at high LoRA rank.
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — the original characterization of why neural networks "forget" earlier training when you fine-tune on new data (EWC — Elastic Weight Consolidation).
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, an earlier approach to using LoRA for continual learning by constraining new adapters to orthogonal subspaces.
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — a foundational technique for merging multiple fine-tuned models without interference.
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — the specific algorithm Backpropagate's multi-run merger implements. A December 2025 preprint; Backpropagate is the paper's first known downstream adopter.

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
