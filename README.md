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

# Fine-tune a 32B QLoRA — or a 7B end to end — on one GPU. Ship it to Ollama.

Backpropagate fine-tunes large language models on a **single** GPU, sized for the card you actually have. Three lines of Python QLoRA a 7B–34B model on one 32 GB consumer card (RTX 5090); one flag — `--full-ft-offload` — full-fine-tunes a 7B-class model by spilling the optimizer state to host RAM. One more command exports to Ollama, then `ollama run` your finetune. Scales cleanly down to 16 GB. First-class on Windows.

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
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — if you want DPO/PPO/RLHF and a web GUI
- **[Unsloth](https://github.com/unslothai/unsloth)** — if you need the fastest possible training and you're on a supported model family
- **[torchtune](https://github.com/pytorch/torchtune)** — if you want Meta's first-party PyTorch-native recipes you can edit

Backpropagate is the missing option: **a 3-line Python API for solo operators on a single consumer GPU who want to train an adapter and ship it.** No YAML, no GUI, no online RL (PPO/GRPO), no multi-node. Just the loop everyone actually needs and the export step that gets in the way.

If you tried one of the libraries above and bounced off the config-file ceremony, or hit a model-family gap, or wanted Windows-first defaults — Backpropagate is for you.

## What you can fine-tune on one GPU

Backpropagate sizes the run to your card. Here's the practical envelope on a **32 GB** consumer GPU (RTX 5090) with 64 GB host RAM — the rig it's tuned on:

| Model size | Method | Status on a 32 GB card |
|---|---|---|
| 7B (Qwen 2.5 7B / Llama-3.1-8B / Mistral 7B) | QLoRA | Comfortable — ~7–8 GB. Full sequence length, lots of headroom. |
| **14B** (Qwen2.5-14B) | QLoRA | **The daily-driver sweet spot — ~8.5 GB** measured. rank/alpha 32, paged 8-bit AdamW, 4096 ctx. |
| 24B (Mistral-Small-24B) | QLoRA | ~18 GB. Fits with headroom at 4096 ctx. |
| **32B** (Qwen2.5-32B) | QLoRA | **Just fits — ~26 GB** at `max_len 2048` + paged 8-bit AdamW. Top of the envelope. |
| ≤6B | `mode="full"` (true full fine-tuning) | Pure-GPU full FT — bf16 weights, no adapter. The card-aware ceiling is 6B on 32 GB. |
| **7B-class** (Qwen 2.5 7B / Llama-3.1-8B / Mistral 7B) | `mode="full" --full-ft-offload` | **Full fine-tuning via FSDP2 CPU-offload** — spills params + optimizer to 64 GB host RAM. Slower (bandwidth-bound); Linux/WSL2. |

Two things most single-GPU libraries send you elsewhere for — **24–34B QLoRA** and **single-card 7B-class full fine-tuning** — Backpropagate does on one consumer card, then exports the result straight to Ollama.

**The full-FT ceiling is card-aware.** It's derived from the 4-addend training-memory arithmetic (weights + gradients + optimizer + activations) against your *detected* VRAM: **16 GB → 4B, 24 GB → 5B, 32 GB → 6B** pure-GPU. `--full-ft-offload` lifts it to **7B-class** by spilling params + optimizer state into host RAM via FSDP2 `fully_shard` + `CPUOffloadPolicy` (slower, PCIe/CPU-bandwidth-bound; needs ~64 GB host RAM and an NCCL backend, i.e. Linux/WSL2). Override the ceiling explicitly with `--full-ft-ceiling-billions`. A model past even the offload ceiling exits with `RUNTIME_FULL_FT_MODEL_TOO_LARGE`, naming the recovery (`--full-ft-offload`, or LoRA/QLoRA). See [the full fine-tuning handbook page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/) for the VRAM math + the Biderman 2024 / Thinking Machines 2025 quality comparison.

### Scales down to 16 GB

The 16 GB envelope (RTX 4080 / 5080 / 4070 Ti Super) is still first-class: 7B QLoRA at ~7–8 GB, and true full fine-tuning of a genuine ~3B (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B/1B) inside 16 GB via `mode="full"` (bf16 weights + gradient checkpointing + paged 8-bit AdamW). The same code picks the batch size and full-FT ceiling that fit whatever card it detects — no flags to change between rigs.

2-bit quantization (AQLM / QuIP#) stays **out of scope** — a 2-bit base can't be cleanly merged back into full-precision weights, which breaks the mergeable-adapter → GGUF → Ollama export contract (the whole point of the pipeline). The headroom levers Backpropagate ships instead — QLoRA, `mode="full"`, `--full-ft-offload`, and the FP8 compute path (`--fp8`, Blackwell/Hopper) — all stay mergeable and exportable.

## What Backpropagate is NOT for

If your use case is below, you'll have a better time with a different library — Backpropagate is not the right pick and trying to make it work would cost more than just reaching for the right tool. Reading this section before you start saves the install-and-bounce cycle:

- **Full-parameter fine-tuning past the offload ceiling (≈13B+)** — Backpropagate full-fine-tunes up to **~6B pure-GPU and ~7B-class via `--full-ft-offload`** on a 32 GB card (see [the envelope](#what-you-can-fine-tune-on-one-gpu)). A *true full* fine-tune of a 13B+ model is past that — it wants multi-GPU FSDP or a bigger card (reach for `transformers.Trainer` across multiple GPUs, or rent an A100/H100). Before spending that compute, though: recent research ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) shows LoRA at correct configuration matches full fine-tuning quality on most post-training tasks (instruction-following, domain adaptation, persona/style) at ~67% of the compute — so QLoRA up to 34B, which Backpropagate does on one card, loses nothing for the work most operators actually want.
- **Online RL — PPO / GRPO / RLVR** — Backpropagate does single-stage SFT plus reference-free preference tuning (ORPO in v1.5; SimPO + KTO in v1.6). What it does *not* do is online reinforcement learning — PPO, GRPO, or RLVR — which needs a reward model or a generation-and-scoring loop on top of the training step. For those, use TRL directly or LLaMA-Factory. (Reference-free preference tuning fits the single-stage envelope because there's no separate reference model to hold in memory; see the ORPO note under [Quick Start](#quick-start).)
- **Multi-node training** — single GPU on one machine only. Multi-GPU on one machine works (via `accelerate launch`) but isn't officially supported.
- **macOS training on the CUDA rail** — Apple Silicon doesn't have CUDA, so the CUDA path runs on a Linux or Windows box with an NVIDIA GPU. You can still run the trained model on a Mac via Ollama. An **experimental, unverified-preview** MLX rail (`--backend mlx`) trains a LoRA adapter natively on Apple Silicon — see [Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview). It is LoRA-SFT-only and **not dogfood-verified on real silicon** (no support), so for anything beyond a LoRA SFT (ORPO, full fine-tune, FP8, multi-run) you want the CUDA rail.
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

### Preference tuning (ORPO, SimPO, KTO)

Train on preferences instead of plain demonstrations. ORPO is reference-free and single-stage — it folds the preference signal into the SFT step, so there's no separate reward or reference model and the 3-line shape is unchanged. Pass `--method orpo` (CLI) or `method="orpo"` (Python) and feed it a dataset of `{prompt, chosen, rejected}` (or just `{chosen, rejected}`) rows:

```jsonl
{"prompt": "What is Python?", "chosen": "A high-level programming language known for readability.", "rejected": "idk look it up"}
{"prompt": "Explain recursion.", "chosen": "A function that calls itself with a smaller input until a base case.", "rejected": "when something repeats"}
```

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct", method="orpo")
trainer.train("preferences.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")
```

```bash
backprop train --data preferences.jsonl --method orpo --steps 100
```

The default learning rate auto-lowers to `8e-6` for ORPO (the loss is sharper than plain SFT); tune `--orpo-beta` (default `0.1`) to weight the odds-ratio penalty. ORPO is `mode="lora"` only.

**New in v1.6 — SimPO and KTO.** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) is reference-free with a length-normalized reward and takes the same paired `{prompt, chosen, rejected}` data as ORPO (`--simpo-beta`, `--simpo-gamma`). `--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) takes **unpaired** `{prompt, completion, label}` data — per-example thumbs-up/down — for the large class of feedback that isn't curated A/B pairs; it auto-balances the desirable/undesirable loss weights from your label counts. Both are `mode="lora"` only and stay in the single-GPU SFT envelope (no separate reference model). See the [preference-tuning handbook](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) for which to use. For online RL (PPO/GRPO) see [What Backpropagate is NOT for](#what-backpropagate-is-not-for).

### Reasoning-trace SFT (R1 distillation)

Distill a reasoning model the easy way. Pass `--reasoning-trace` (CLI) or `Trainer(..., reasoning_trace=True)` (Python) and feed it traces that keep a `<think>...</think>` chain-of-thought inside the assistant turn — the pure-SFT half of [DeepSeek-R1](https://arxiv.org/abs/2501.12948) distillation, no RL required. Backpropagate keeps `<think>` in the training target, drops empty / over-long traces (trace-length filtering), and raises the default `max_seq_length` to 8192 for the longer CoT. Critically, `<think>` stays **plain text** — no special tokens, no embedding resize — so the merged GGUF still exports to Ollama like any other fine-tune. SFT only. See the [reasoning-trace recipe](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) for the dataset shape and the tunable token band.

### Apple Silicon (MLX) — unverified preview

> ⚠️ **Unverified preview — not part of the supported feature set.** The MLX rail is built and unit-tested but has **not** been dogfood-verified on real Apple Silicon (`mlx-lm` is Apple-only and can't run on the NVIDIA rigs Backpropagate is developed on). Treat everything below as experimental, use at your own risk, and [report anomalies](#reporting-bugs) if you run it on an M-series Mac.

**One API, two rails.** CUDA is the canonical, verified backend; MLX is a second rail that trains on an M-series Mac via Apple's [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) toolchain (unified memory, no CUDA). The 3-line shape picks the rail by hardware — `backend='auto'` (the default) routes to CUDA on NVIDIA and to MLX on Apple Silicon, so existing CUDA rigs are byte-identical:

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

The MLX rail is **LoRA SFT only** — no ORPO, no FP8, no `mode='full'`, no multi-run (each is rejected with `CONFIG_INVALID_SETTING`; use `backend='cuda'`/`'auto'` on an NVIDIA box for those). The resulting adapter is plain safetensors and exports to Ollama through the same path as the CUDA rail.

> Forcing `--backend mlx` on a non-Apple host errors with `CONFIG_INVALID_SETTING`; a missing `mlx_lm` toolchain on a Mac raises `DEP_MLX_UNAVAILABLE`.

For more end-to-end workflows (fine-tune-and-push-to-HF-Hub, resume after OOM, multi-run SLAO across a long campaign, etc.) see the [handbook recipes page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Web UI (optional)

If you'd rather click than type Python, install the UI extra and launch:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

A local web interface opens at `http://localhost:7862` for browsing datasets, validating formats, and assembling a training config visually. Training itself runs via `backprop train` (UI-driven training is on the roadmap — the Start button currently surfaces that note). The UI is local-only by default. To expose it to other devices, see [Web UI](#web-ui) below for the `--share` + `--auth` security contract.

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

The UI runs locally on `http://localhost:7862`. Today it covers the **browse / validate / configure** half of the workflow — point it at a dataset, check the auto-detected format and stats, pick a model, and assemble a run config. **Launching the run is done from the CLI** (`backprop train` / `backprop multi-run`); the in-UI Start button surfaces a note pointing there. UI-driven training is a planned follow-up — until then the UI is the on-ramp and the CLI is the trigger.

To expose it to other devices (other people on your network, a public URL, etc.) you must pair `--share` (or `--host`) with `--auth`:

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

Python 3.10 is supported through at least v1.6; it reaches upstream end-of-life in October 2026 and is scheduled for removal in the first release after that. For new installs, prefer Python 3.11 or 3.12 — 3.11 is the most-tested floor.

Backpropagate handles the runtime quirks of training on different platforms, but it can't fix install-time problems. The two most common are:

- **Wrong CUDA wheel.** PyTorch is published one binary per CUDA version. If you pick the wrong one, you silently get CPU-only PyTorch and training is impossibly slow. Use the wheel picker at <https://pytorch.org/get-started/locally/> for your driver. Run `nvidia-smi` to see your driver / CUDA version.
- **Windows + GGUF export.** The `[export]` extra builds `llama-cpp-python` from source, which needs Visual Studio Build Tools (C++ component) and CMake.

**macOS:** the CUDA rail is not supported (no CUDA) — a CUDA-routed `trainer.train()` raises `DEP_GPU_NOT_AVAILABLE`, and you can run the trained adapter on a Mac via Ollama. An **experimental, unverified-preview** MLX rail (`--backend mlx`, `pip install 'backpropagate[mlx]'`) trains a LoRA adapter natively on Apple Silicon via `mlx_lm.lora` — LoRA SFT only, and **not dogfood-verified on real silicon** (see [Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview)). For the CUDA path, or for ORPO / full fine-tune / FP8 / multi-run, use a CUDA Linux or Windows machine.

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
| Llama-3.1-8B | ~7-8GB (QLoRA) | Llama-3.1-Community | 8B QLoRA, 128K native context (the >700M-MAU clause needs a separate Meta license). |
| **Qwen2.5-14B** | ~8.5GB (QLoRA) | Apache 2.0 | **The 32 GB daily-driver sweet spot** — rank/alpha 32, paged 8-bit AdamW, 4096 ctx. |
| Mistral-Small-24B | ~18GB (QLoRA) | Apache 2.0 | 24B QLoRA on a 32 GB card with 4096-ctx headroom. |
| **Qwen2.5-32B** | ~26GB (QLoRA) | Apache 2.0 | **Top of the 32 GB envelope** — just fits at `max_len 2048` + paged 8-bit AdamW. |

Other models often work; the rows above are the curated presets — the 14B–32B tier is QLoRA-tuned for a 32 GB card (the measured envelope). Pass `--lora-preset=quality` (default) for rank-256 / all-linear targets per Biderman 2024 + Thinking Machines 2025, or `--lora-preset=fast` for the legacy rank-16 / q+v target if you need the v1.2.x footprint.

## Troubleshooting

A short index of the most common first-run failures. The full reverse index is at [the troubleshooting handbook page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). For driver / VRAM / mixed-precision deep-dive see the [CUDA troubleshooting page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Symptom | Error code | Fix |
|---|---|---|
| GPU runs out of memory mid-training | `RUNTIME_GPU_OOM` | Automatic — Backpropagate halves the batch size and retries up to 3 times. To opt out: `Trainer(oom_recovery=False)`. To force smaller: `--batch-size 1`. |
| HuggingFace returns 401 / "model not found" | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` and retry. For typos, copy the exact ID from <https://huggingface.co/models>. |
| `register_with_ollama` connection refused | `DEP_OLLAMA_REGISTRATION_FAILED` | Start the daemon: `ollama serve`. Install from <https://ollama.com>. Retryable. |
| Disk full during checkpoint save | `STATE_CHECKPOINT_INVALID` | Atomic writes leave a `.partial` directory on crash — safe to delete. The previous good checkpoint is intact. |
| Training paused on GPU overheat | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automatic — Backpropagate pauses on the temperature threshold and resumes as the GPU cools. Improve airflow if it keeps happening. |
| `backprop ui --share` rejected | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Pass `--auth user:password`, or use SSH port-forwarding instead (see [Web UI](#web-ui)). |
| GGUF export failed on first try | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; on Windows you also need Visual C++ Build Tools + CMake. |

## Reporting bugs

When something fails, Backpropagate prints a line at startup like `run_started run_id=<uuid>` and binds the same ID to every log line, every checkpoint, and every Weights & Biases entry. **Include the `run_id` in any bug report** — it lets a maintainer correlate everything for that exact run.

A good bug report includes:

1. **The `run_id`** — the UUID printed at startup. One UUID lets a maintainer correlate every log line, every checkpoint, and every Weights & Biases entry for that exact run.
2. **The error code** — the `[CODE_NAME]: message` line in stderr. See [error codes](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) for the catalog of stable codes.
3. **The redacted traceback.** Stderr is automatically redacted in non-verbose mode (Bearer tokens, `sk-*`, `hf_*`, AWS keys, `password=` / `token=` / `api_key=` pairs are scrubbed) — safe to paste. For the full unredacted traceback, re-run with `BACKPROPAGATE_DEBUG=1` (or `--verbose`); review before posting.
4. **The `backprop info` output.** One command prints Python / PyTorch / CUDA / GPU model / VRAM / OS / installed extras — everything the maintainer needs to bisect a platform-specific regression.

The [bug report template](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) prompts for each of these explicitly so triage moves fast. Questions, ideas, or "is this expected?" threads belong in [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Security issues should be reported privately via the [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) form — see [SECURITY.md](SECURITY.md) for the policy and response timelines.

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
