---
title: Recipes
description: Common operator workflows — paste-and-run snippets for the things most operators want to do on day one.
sidebar:
  order: 4
---

A library of short, paste-and-run snippets keyed by what you actually want to do. Each recipe assumes you have already installed Backpropagate (see [Getting Started](/backpropagate/handbook/getting-started/)) and have a CUDA GPU available.

If you are looking for symptoms-first triage, head to [troubleshooting](/backpropagate/handbook/troubleshooting/) instead.

## Fine-tune a Llama 3 model on a custom JSONL dataset

Llama 3 chat models (`meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`) are gated on Hugging Face — accept the license on the model page and run `huggingface-cli login` first.

```python
from backpropagate import Trainer

trainer = Trainer("meta-llama/Llama-3.2-3B-Instruct")
trainer.train("my_data.jsonl", steps=200)
trainer.save("./output/llama3-finetuned")
```

CLI equivalent:

```bash
backprop train \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --data my_data.jsonl \
  --steps 200 \
  --output ./output/llama3-finetuned
```

Your JSONL can be ShareGPT, Alpaca, OpenAI-chat, or ChatML — the auto-detector picks the right template (see [Training → Dataset formats](/backpropagate/handbook/training/#dataset-formats)). v1.2.0 fixed the tokenizer-aware `train_on_responses_only` masker so Llama 3 chat templates mask correctly (the v1.1.x bug silently trained on user prompts as well). If you were getting bad fine-tunes on Llama 3, re-run on v1.2.0+ — see [migrations → behavioural fixes](/backpropagate/handbook/migrations/#behavioural-fixes).

## Reasoning-trace SFT (R1 distillation)

**New in v1.5 (T3.2).** Distill a reasoning model the easy way: pure SFT on traces that interleave a `<think>...</think>` chain-of-thought with the final answer (the half of [DeepSeek-R1](https://arxiv.org/abs/2501.12948) distillation that needs no RL). Your dataset rows carry the thinking block *inside* the assistant turn:

```json
{"messages": [
  {"role": "user", "content": "What is 17 * 24?"},
  {"role": "assistant", "content": "<think>17 * 24 = 17 * 20 + 17 * 4 = 340 + 68 = 408.</think>408"}
]}
```

Turn on the recipe with one flag (Python):

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct", reasoning_trace=True)
trainer.train("reasoning_traces.jsonl", steps=200)
trainer.save("./output/qwen-reasoner")
```

CLI equivalent:

```bash
backprop train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data reasoning_traces.jsonl \
  --steps 200 \
  --reasoning-trace \
  --output ./output/qwen-reasoner
```

What `--reasoning-trace` does:

- **Keeps `<think>` in the training target.** The chat-template converters already preserve `<think>` blocks verbatim — nothing is stripped. Crucially, **`<think>` stays plain text**: Backpropagate does **not** add special tokens or resize the embedding matrix for it. That keeps the merge → GGUF → Ollama export path intact — a reasoning fine-tune ships to `ollama run` exactly like any other (see [Export to Ollama](#export-a-trained-adapter-to-ollama-one-command)).
- **Trace-length filtering.** Rows whose summed `<think>` token count falls outside `[8, 8192]` tokens are dropped — empty / degenerate traces and runaway ones both hurt distillation. Tune the band with `BACKPROPAGATE_DATA__MIN_TRACE_TOKENS` / `BACKPROPAGATE_DATA__MAX_TRACE_TOKENS` (the tokenizer's own `encode` does the counting, so the cutoffs are exact for your model). Rows with no `<think>` span at all are dropped too.
- **Raises the default `max_seq_length` to 8192.** Reasoning traces routinely exceed the shipped 2048-token window; the bump only fires when you left `max_seq_length` at the default. An explicit value — kwarg `max_seq_length=...` or `BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH` — always wins.

The recipe is **SFT only** — it is ignored under `--method orpo`. If your model's chat template injects its own empty `<think>` opener AND your data already opens with `<think>`, you'll get a one-line advisory warning about the doubled tag (strip the leading `<think>` from your data, or use a template that doesn't inject one).

## Export a trained adapter to Ollama (one command)

The fastest path from a trained LoRA to `ollama run`:

```python
from backpropagate import register_with_ollama

result = trainer.export("gguf", quantization="q4_k_m")
register_with_ollama(result.path, "my-finetuned-model")
```

Then:

```bash
ollama run my-finetuned-model
```

CLI equivalent in a single line:

```bash
backprop export ./output/lora --format gguf --quantization q4_k_m \
  --ollama --ollama-name my-finetuned-model
```

This merges the LoRA back into the base model, converts to GGUF at the chosen quantization, writes an Ollama Modelfile next to the `.gguf`, and registers the model with the local Ollama daemon. If Ollama is not running you'll see `DEP_OLLAMA_REGISTRATION_FAILED` — start it with `ollama serve` and retry (see [troubleshooting → Ollama not running](/backpropagate/handbook/troubleshooting/#ollama-not-running--connection-refused-on-register)).

## Resume an interrupted multi-run

Multi-run training writes per-run state to `output_dir/run_history.json`. If your training crashed at run 3 of 5, resume from where it stopped:

```python
from backpropagate import MultiRunTrainer

trainer = MultiRunTrainer("Qwen/Qwen2.5-7B-Instruct")
trainer.resume(run_id="<run_id_from_the_crashed_log_line>")
```

CLI equivalent:

```bash
backprop resume --run-id <run_id>
```

The resume path restores optimizer state, scheduler state, and step counter from the most recent checkpoint inside the run's `output_dir/checkpoint-<N>/`. v1.2.0 fixed the single-run resume path that silently restarted from step 0 in v1.1.x (BACKEND-F-017 — see [migrations](/backpropagate/handbook/migrations/#trainertrainresume_fromrun_id-now-actually-resumes)).

**Strict-miss contract (v1.3):** if `resume_from=<run_id>` refers to a run that no longer exists on disk (history record deleted, checkpoint directory wiped), the trainer raises `INPUT_RESUME_NOT_FOUND` rather than silently falling back to a fresh start. To resume by `run_id` you need the on-disk state to still be there; if you want a fresh start, omit `resume_from` or pass `resume_from=None`.

## Diff two runs with different learning rates

If you ran the same training with two different `--lr` values and want to compare:

```bash
backprop list-runs
backprop show-run <run_id_a>
backprop show-run <run_id_b>
```

For programmatic consumption (v1.2.0+):

```bash
backprop runs --limit 10        # JSON enumerator
backprop runs --status completed --limit 5
```

A dedicated `backprop diff-runs <run_id_a> <run_id_b>` subcommand that prints a side-by-side comparison of hyperparameters + final-loss + loss-curves is on the v1.3 roadmap (FRONTEND/BACKEND Wave 6b). Until it ships, the JSON output of `backprop runs` plus `jq` or a short Python script will get you the comparison you need:

```bash
backprop runs --limit 50 \
  | jq '.runs | map(select(.run_id == "<run_id_a>" or .run_id == "<run_id_b>"))'
```

## Add a custom callback for logging

`TrainingCallback` exposes five hooks: `on_step`, `on_epoch`, `on_save`, `on_complete`, `on_error`. v1.2.0 fixed the bug that left `on_step` / `on_epoch` / `on_save` as silent no-ops in v1.1.x — if you wrote a callback against v1.1.x and never saw the hooks fire, expect to see them now (see [migrations](/backpropagate/handbook/migrations/#trainingcallbackon_step--on_epoch--on_save-now-actually-fire)).

```python
from backpropagate import Trainer, TrainingCallback

def log_step(step: int, loss: float) -> None:
    if step % 10 == 0:
        print(f"step={step:5d} loss={loss:.4f}")

callback = TrainingCallback(
    on_step=log_step,
    on_complete=lambda run: print(f"done — final loss {run.final_loss:.4f}, run_id={run.run_id}"),
    on_error=lambda err: print(f"failed: {err}"),
)

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100, callback=callback)
```

Each hook is isolated — an exception in your `on_step` does **not** kill the training loop. The exception is caught, logged with the run_id, and training continues.

## Push to a private Hugging Face Hub repo

```bash
backprop push ./output/lora --repo your-org/qwen-finetune --private
```

The `--private` flag makes the repo private at creation time. The token resolution order is `--token` flag → `HF_TOKEN` env var → `HUGGING_FACE_HUB_TOKEN` env var → `~/.cache/huggingface/token` (from `huggingface-cli login`). Use `huggingface-cli login` to cache a token from <https://huggingface.co/settings/tokens> — make sure the token has **write** scope.

One-shot export + push:

```bash
backprop export ./output/lora --format lora --push-to-hub your-org/qwen-finetune
```

The `model_card.md` written next to the local export is mirrored as `README.md` inside the upload, so the HF UI renders it as the repo's model card. See [export → Hub push](/backpropagate/handbook/export/#hub-push-v110) for the full Hub-push surface.

## Run the Reflex UI with `--share` over a real cloudflared tunnel

**Prerequisite:** install `cloudflared` (Cloudflare's tunnel client) — see <https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/>. The v1.3 `--share` implementation shells out to `cloudflared tunnel --url http://127.0.0.1:<port>` and parses the announced `https://*.trycloudflare.com` URL out of the daemon's stderr. No account, no zone, no DNS setup required — `cloudflared` provisions an ephemeral quick-tunnel that lives for the duration of the `backprop ui` process.

```bash
backprop ui --share --auth alice:super-secret-password
```

You'll see the announced URL in the startup banner (and the same URL is added to the `Host` / `Origin` allowlist). The v1.2.0 FastAPI middleware enforces HTTP Basic auth on every request and the `/_event` WebSocket upgrade, so anyone who hits the URL is challenged for the credentials.

**Required:** `--share` without `--auth` exits `1` with `[RUNTIME_UI_AUTH_NOT_ENFORCED]` (closes the v1.1.x foot-gun published as [GHSA-f65r-h4g3-3h9h](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/GHSA-f65r-h4g3-3h9h)). For the full contract see [security → four-layer defense in depth](/backpropagate/handbook/security/#four-layer-defense-in-depth).

**If you don't want a public URL:** SSH port-forwarding stays the lower-friction option for "I just want to reach my remote training box from my laptop" — see [security → SSH port-forwarding recipe](/backpropagate/handbook/security/#ssh-port-forwarding-recipe).

## Fine-tune on multi-GPU

Multi-GPU training is **not officially supported** in v1.3 — the library targets the single-GPU operator (16 GB VRAM workstation as the canonical target). If you want to try it anyway, the recommended setup is HuggingFace's `accelerate` library:

```bash
pip install accelerate
accelerate config       # answer the prompts; pick "multi-GPU"
accelerate launch -m backpropagate.cli train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data my_data.jsonl \
  --steps 200
```

`accelerate launch` wraps the training entry point with multi-process initialisation (NCCL, distributed sampler, gradient sync). The Unsloth backend may not work cleanly under `accelerate` — start with `--no-unsloth` if you hit `unsloth` import errors. Expect rough edges; the GPU-monitoring (`gpu_safety.py`) is per-process and may report misleading temperatures across multiple GPUs. Multi-GPU NCCL failures emit `RUNTIME_*` errors and are not in the v1.3 retryable-error matrix.

For deeper multi-GPU support (FSDP, deepspeed) consider running training under `transformers.Trainer` directly and re-using only `backpropagate.export` for the GGUF + Ollama step.

## Custom dataset format / data collator

If your dataset doesn't fit ShareGPT / Alpaca / OpenAI-chat / ChatML / raw-text, the cleanest path is to pre-process to one of those formats with a 10-line script. Example: convert a CSV of `(prompt, completion)` pairs to OpenAI-chat JSONL:

```python
import csv, json

with open("pairs.csv") as fp_in, open("converted.jsonl", "w") as fp_out:
    for row in csv.DictReader(fp_in):
        record = {"messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]}
        fp_out.write(json.dumps(record) + "\n")
```

Then point Backpropagate at `converted.jsonl` — it auto-detects the OpenAI-chat shape.

For a **truly custom collator** (e.g. structured multi-turn with extra fields), load the dataset yourself as a HuggingFace `Dataset` and pass it directly:

```python
from datasets import load_dataset
from backpropagate import Trainer

ds = load_dataset("json", data_files="my_weird_format.jsonl", split="train")
ds = ds.map(my_custom_preprocessing, batched=True)

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train(ds, steps=100)
```

The auto-format detector is skipped when you pass a pre-built `Dataset`; you're responsible for shaping it into the columns Backpropagate's collator expects (the default expects either a `text` column with full ChatML or a `messages` column in OpenAI-chat shape). See [training → dataset formats](/backpropagate/handbook/training/#dataset-formats).

## Use `--auth-file` for shell-history-safe auth

Passing `--auth user:pass` on the command line works, but the credential lands in your shell history file and is briefly visible in `ps aux`. The v1.3 `--auth-file <path>` flag reads the credential from a file instead — same `user:pass` shape, one line, no trailing newline noise:

```bash
echo -n "alice:super-secret-password" > ~/.config/backpropagate/auth
chmod 600 ~/.config/backpropagate/auth
backprop ui --share --auth-file ~/.config/backpropagate/auth
```

The CLI reads the file, validates the shape with the same `validate_auth_shape` used for `--auth`, and threads the credential into the Reflex subprocess via `BACKPROPAGATE_UI_AUTH`. The file is never logged; the credential is redacted from any error output. `--auth` and `--auth-file` are mutually exclusive — passing both exits `1` with `INPUT_AUTH_INVALID_SHAPE`.

`--auth-file` satisfies the same `--share` / `--host <non-loopback>` requirement that `--auth` does — passing it means the four-layer defense is satisfied. See [security → auth middleware](/backpropagate/handbook/security/#auth-middleware-v120) for the full mode matrix.

## See also

- [Training](/backpropagate/handbook/training/) — basic training, SLAO multi-run, callbacks, dataset formats.
- [Export](/backpropagate/handbook/export/) — GGUF / Ollama / Hub push paths.
- [CLI reference](/backpropagate/handbook/cli-reference/) — every flag.
- [Troubleshooting](/backpropagate/handbook/troubleshooting/) — symptoms-first index.
- [Troubleshooting CUDA](/backpropagate/handbook/troubleshooting-cuda/) — GPU-specific failure modes.
