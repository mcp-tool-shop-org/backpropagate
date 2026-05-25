---
title: Troubleshooting
description: Common first-run failures and how to recover.
sidebar:
  order: 6
---

A symptoms-first reverse index. If you already know the error code, jump to [Error codes](/backpropagate/handbook/error-codes/) instead.

## "CUDA out of memory" / GPU OOM

**Symptom:** Training crashes mid-step with `RUNTIME_GPU_OOM` or PyTorch's raw `torch.cuda.OutOfMemoryError`.

**What Backpropagate does for you:** The OOM-recovery path (B-002, enabled by default) catches OOM errors during training, halves the per-device batch size, clears the CUDA cache, and resumes. It will retry up to 3 times at the minimum batch size before giving up.

**What to try next:**

1. Drop `--batch-size` manually: `backprop train --batch-size 1 ...`.
2. Reduce `--max-seq-length` (most VRAM is spent on attention over sequence length).
3. Pick a smaller model preset (Qwen 2.5 3B fits in ~8 GB).
4. If you want OOM to be a hard failure (e.g. you're benchmarking), pass `Trainer(oom_recovery=False)` in the Python API.

## "HF Hub 401" / gated model / model not found

**Symptom:** `DEP_MODEL_LOAD_FAILED` with a 401 or 404 from `huggingface_hub`.

**Why:** Some models (Llama 3, some Qwen variants) require accepting a license on the model page first; others are private to your account.

**Fix:**

```bash
huggingface-cli login   # paste a token from https://huggingface.co/settings/tokens
```

Or export `HF_TOKEN` in the environment. Then re-run. For typos in the model name, head to <https://huggingface.co/models?search=qwen> (or whichever family you want) and copy the exact `org/name` identifier.

## "Ollama not running" / connection refused on register

**Symptom:** `DEP_OLLAMA_REGISTRATION_FAILED`, often with `connection refused localhost:11434`.

**Fix:**

```bash
ollama serve     # start the daemon (foreground)
```

Or install Ollama from <https://ollama.com/download>. The default endpoint is `localhost:11434`; this is retryable, so the CLI will back off and retry a few times before giving up.

## "Disk full" mid-checkpoint

**Symptom:** Training crashes during a save with `STATE_CHECKPOINT_INVALID` and a disk-full IOError underneath.

**What Backpropagate does for you:** Checkpoints write atomically (write to `<path>.partial`, fsync, rename to `<path>`). If a write is interrupted, you get a stale `.partial` directory but the previous good checkpoint is intact.

**Fix:**

1. Free space (delete old checkpoints in `./output/`).
2. Remove any leftover `.partial` directories (they are safe to delete — they represent a half-written write that did not complete).
3. Resume training from the previous good checkpoint.

## "GPU temperature critical" pause

**Symptom:** Training pauses with `RUNTIME_GPU_TEMPERATURE_CRITICAL` and the message *Waiting for GPU to cool...*.

**Why:** The GPU monitor (B-003) watches NVML temperature and pauses training when the threshold is exceeded. This is retryable — training resumes automatically once the temperature drops back to safe.

**Fix:** Improve case airflow; lower sustained load via smaller batch size or gradient accumulation; raise the threshold in config if you intentionally run hot. See [GPU safety](/backpropagate/handbook/reference/#gpu-safety) for the thresholds.

## "samples_per_run exceeds training pool"

**Symptom:** `CONFIG_INVALID` from multi-run with the suggestion *Reduce --samples (Python: samples_per_run) below N*.

**Why:** Multi-run reserves 10% of the dataset as a validation holdout; the remaining 90% is the training pool. If `samples_per_run` exceeds the training pool, no chunk can be cut without overlapping the validation set.

**Fix:** Lower `--samples` on the CLI (or `samples_per_run=` in the Python API) below the reported training pool size, OR pass a larger dataset, OR disable validation with `validate_every_run=False, early_stopping=False`.

## "Multi-run training reported success but the model didn't actually learn" (v1.3.x only)

**Symptom:** A multi-run finishes with `status=completed`, no exceptions raised, but the final adapter performs the same as the base model on your eval. Loss curves for run 2+ look flat compared to run 1.

**Why (v1.3.x bug):** `MultiRunTrainer._compute_validation_loss` left the model stuck in `eval()` mode after a `CUDA out of memory` (or any exception escaping the validation loop). The next training pass silently produced no gradient updates — operators saw "training completed" but the model didn't learn anything new.

**Fix:** Upgrade to v1.4 (`pip install -U backpropagate`). The validation body is now wrapped in `try ... finally: model.train()`, so the train-mode invariant is restored even on exception. If you have a v1.3.x multi-run that may have hit this, re-run on v1.4 from a clean checkpoint — the on-disk adapter for run 2+ is effectively untrained. See [BACKEND-B-003 in the v1.4 CHANGELOG](https://github.com/mcp-tool-shop-org/backpropagate/blob/main/CHANGELOG.md#unreleased) for the full bug write-up.

## "I called `.save()` and the adapter looks tiny / random"

**Symptom:** `trainer.save("./out")` wrote a `.safetensors` file but the resulting LoRA performs like the base model. Or you copy-pasted an example that called `trainer.load_model()` then `.save()` without ever calling `.train()`.

**Why:** `.save()` writes whatever LoRA weights are currently in memory — including the initial (random / zero) weights if `.train()` never ran.

**Fix (v1.4):** Look in your stderr for a WARN line like `Trainer.save() called before train() — writing init-weight LoRA adapter to <path>`. v1.4 emits this cue at every `save()` call where `_has_trained=False`. The save still completes (no exception), so existing tooling that pre-creates output dirs still works — but the warning tells you what happened. If you see it, call `trainer.train(...)` before `trainer.save(...)`. See BACKEND-B-004 in the v1.4 CHANGELOG.

## "Unsloth import failed, falling back to transformers"

**Symptom:** Warning at startup: *Unsloth load failed (...); falling back to transformers + PEFT. Set Trainer(unsloth_fallback=False) to disable.*

**Why:** B-010 graceful degradation. An Unsloth nightly broke `from_pretrained`; rather than fail the whole pipeline, Backpropagate loads the model with plain `transformers` + `peft`. Training still works, just ~2x slower.

**Fix (if you insist on Unsloth):** pin a known-good Unsloth version, then run `Trainer(unsloth_fallback=False, ...)` to force a hard failure on Unsloth issues so they surface immediately.

## "llama-cpp-python build failed" on first GGUF export

**Symptom:** `trainer.export("gguf", ...)` raises `RUNTIME_GGUF_EXPORT_FAILED` with a build error from `pip install`.

**Why:** GGUF export uses `llama.cpp` via `llama-cpp-python`, which is a C++ extension. The wheel is not always available for your platform and may build from source.

**Fix:**

1. Install the export extra: `pip install backpropagate[export]`.
2. On Windows: install Visual Studio Build Tools (C++) and CMake. Then re-run pip.
3. On Linux: install `build-essential` and `cmake`.

## "Multi-run validation overlap"

**Symptom:** Same as *samples_per_run exceeds training pool* above. Stage A's backend B-001 fix raises a clean `ConfigurationError` instead of silently overlapping train/val.

## "What does `[RUNTIME_UI_AUTH_NOT_ENFORCED]` mean?"

You passed `--share` (without `--auth`), `--host <non-loopback>` (without `--auth`), or `--auth` while the `[ui]` extra was degraded. The v1.2.0 runtime refuses to start in each of those cases because a public URL or non-loopback bind without credentials is the v1.1.x bug — published in [GHSA-f65r-h4g3-3h9h](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/GHSA-f65r-h4g3-3h9h) (CVSS 9.8, 2026-05-23) — and the refuse-to-start contract makes sure it cannot silently come back.

**Fix (recommended for the public-URL case): pass `--auth user:pass`.** The v1.2.0 FastAPI auth middleware enforces credentials on every HTTP route and the `/_event` WebSocket upgrade plus the Host/Origin allowlist (DNS-rebinding + CSWSH defense).

```bash
backprop ui --share --auth user:pass
# OR
backprop ui --host 0.0.0.0 --auth user:pass
```

**Fix (recommended when you don't actually need a public URL): SSH port-forwarding.**

```bash
ssh -L 7860:localhost:7860 you@gpu-host
# Then on your laptop: http://localhost:7860
```

SSH already handles auth, encryption, and audit. The UI stays bound to `127.0.0.1` on the remote box; only your forwarded tunnel can reach it. No middleware required.

**The old `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false` flag from the Gradio era stays a no-op.** The refuse-to-start contract is enforced at both the CLI layer (`cli.py:cmd_ui`) and the app layer (`ui_app/app.py` + `rxconfig.py`), so `python -m reflex run` from the package directory also refuses unless the legitimate `backprop ui` bridge has set its bypass env var. Full chain in [the security page → Four-layer defense in depth](/backpropagate/handbook/security/#four-layer-defense-in-depth).

## "What does the `Run ID:` line at startup mean?"

Every training run prints `run_started run_id=<uuid>` in the structured log envelope (and binds the same id to checkpoint manifests and SLAO merge history). When something fails, quoting that `run_id` in a bug report lets a maintainer correlate every log line, checkpoint, and merge across the entire run. See [Reporting bugs](/backpropagate/handbook/reference/#reporting-bugs).

## Windows-specific install pain

`pip install backpropagate[standard]` handles the runtime quirks, not the install quirks. The most common Windows install issues are upstream:

- **CUDA toolkit version mismatch.** Pick the wheel that matches your installed driver — visit <https://pytorch.org/get-started/locally/> for the exact `pip install torch ...` command for your CUDA version.
- **bitsandbytes Windows wheel.** Use `pip install bitsandbytes` >= 0.43; older versions had a separate `bitsandbytes-windows` fork that is no longer needed.
- **xformers.** Auto-disabled by Backpropagate for RTX 40/50 series (no action needed).
- **llama-cpp-python.** Needs Visual Studio Build Tools + CMake to build from source. See the GGUF section above.

## macOS

GPU training is not supported on macOS (no CUDA). You can install Backpropagate to run *inference* against an already-exported GGUF model via Ollama, but `trainer.train()` will raise `DEP_GPU_NOT_AVAILABLE`. For training, use a CUDA-capable machine.

## See also

- [Error codes](/backpropagate/handbook/error-codes/) — full catalog of structured codes.
- [Environment variables](/backpropagate/handbook/env-vars/) — every knob.
- [Reporting bugs](/backpropagate/handbook/reference/#reporting-bugs) — what to include in an issue.
