---
title: Troubleshooting (CUDA)
description: GPU-specific failure modes — OOM, CUBLAS, NCCL, mixed-precision pitfalls, driver mismatch.
sidebar:
  order: 6.5
---

The depth section for CUDA failure modes. If you don't know which class of failure you're seeing, head to [troubleshooting](/backpropagate/handbook/troubleshooting/) first — it's the symptoms-first reverse index. This page assumes you already know the failure is GPU-side and want the longer explanation.

## OOM (Out of Memory)

**Symptom:** `RUNTIME_GPU_OOM`, or PyTorch's raw `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate <N> MiB...`

**What Backpropagate does for you:** the OOM-recovery path (BACKEND-B-002, enabled by default) catches OOM errors during training, halves the per-device batch size, clears the CUDA cache via `torch.cuda.empty_cache()`, runs `gc.collect()`, and resumes. It retries up to 3 times at the minimum batch size before giving up — at which point it raises `RUNTIME_OOM_RECOVERY_EXHAUSTED`.

**OOM-adjacent recovery (BACKEND-B-003):** some allocations fail with errors that aren't strictly `OutOfMemoryError` but are caused by VRAM pressure (CUDNN status `CUDNN_STATUS_NOT_INITIALIZED`, `CUDA error: an illegal memory access was encountered`). v1.2.0's OOM-adjacent recovery widens the catch to these adjacent failure modes and applies the same halve-batch-and-retry recovery. The error code is `RUNTIME_OOM_ADJACENT` when this path fires.

**What to try next** (in order of effort):

1. Drop `--batch-size` manually: `backprop train --batch-size 1 ...`.
2. Reduce `--max-seq-length` (most VRAM is spent on attention over sequence length — halving the sequence length saves more memory than halving the batch size in most cases).
3. Pick a smaller model preset (Qwen 2.5 3B fits in ~8 GB; Llama 3.2 1B fits in ~6 GB).
4. Enable gradient checkpointing: `BACKPROPAGATE_LORA__USE_GRADIENT_CHECKPOINTING=unsloth` (default) is already on; set to `true` to force the HF backend if Unsloth's checkpointer is the problem.
5. If you have multiple GPUs and OOM is killing the run, see [the multi-GPU recipe](/backpropagate/handbook/recipes/#fine-tune-on-multi-gpu) — multi-GPU is unofficial, but `accelerate launch` can spread the optimizer across devices.
6. If you want OOM to be a hard failure (e.g. you're benchmarking memory ceilings and don't want the recovery to mask the threshold), pass `Trainer(oom_recovery=False)` in the Python API or `BACKPROPAGATE_TRAINING__OOM_RECOVERY=false`.

**What does NOT save you from OOM:** turning off `bf16` and turning on `fp32` — that **doubles** the memory footprint. Lower-precision dtypes save VRAM; `fp32` is the most expensive.

## CUBLAS errors

**Symptom:** `CUBLAS_STATUS_NOT_INITIALIZED`, `CUBLAS_STATUS_EXECUTION_FAILED`, `CUBLAS_STATUS_ALLOC_FAILED`, often inside a `RuntimeError: CUDA error: ...` wrapped by PyTorch.

**Most common cause:** stealth OOM. CUBLAS allocates workspace lazily; when VRAM is tight, the first matmul of a step fails with `CUBLAS_STATUS_ALLOC_FAILED` instead of the cleaner `OutOfMemoryError`. The OOM-adjacent recovery path (v1.2.0+) catches the most common variants and retries with a smaller batch — see [OOM-adjacent recovery](#oom-out-of-memory) above.

**Second most common cause:** sticky CUDA state across runs in the same process (a previous error left the CUDA context in a bad state, and subsequent CUBLAS calls inherit it). The fix is to restart the Python process — this is also why `Trainer.train()` always isolates the CUDA context inside its own scope.

**Third (rare) cause:** CUDA + cuBLAS version mismatch with the installed PyTorch. Run `backprop info` — it prints the PyTorch CUDA version. Match it against your driver's reported CUDA capability (`nvidia-smi`'s top-right corner). PyTorch built against CUDA 12.4 will not run on a driver capped at CUDA 11.x. The fix is to reinstall PyTorch from <https://pytorch.org/get-started/locally/> with the right `--index-url` for your driver — see [driver / CUDA version mismatch](#driver--cuda-version-mismatch) below.

**Debugging tip:** set `CUDA_LAUNCH_BLOCKING=1` (or `BACKPROPAGATE_WINDOWS__CUDA_LAUNCH_BLOCKING=true` on Windows) — kernel launches become synchronous and the stack trace points at the line that actually failed instead of an arbitrary later op. This slows training meaningfully; turn it back off after you've located the failure.

## NCCL errors (multi-GPU)

**Symptom:** `NCCL_ERROR_*`, `nccl unhandled cuda error`, `nccl reported `<N>` ranks but only `<M>` are responding`.

Multi-GPU is **not officially supported** in v1.3. If you're seeing NCCL errors, you launched training under `accelerate launch` or `torchrun` (see [the multi-GPU recipe](/backpropagate/handbook/recipes/#fine-tune-on-multi-gpu)). Single-GPU operators can ignore this entire section.

If you are in a multi-GPU setup:

- **`nccl reported N ranks but only M are responding`:** one of your workers died (often OOM on a single rank). Check each rank's log — NCCL's error message comes from the *coordinator* and tells you the count, not which one died.
- **`Connection refused` on rendezvous:** the `MASTER_ADDR` / `MASTER_PORT` env vars are not reachable across nodes, or your firewall is blocking the rendezvous port. Single-node multi-GPU should not need any network setup; the failure is usually a stale process holding the port from a previous run.
- **`unhandled cuda error`:** restart all ranks. NCCL can wedge in ways that are not recoverable in-process.

The library's GPU-monitoring (`gpu_safety.py`) is **per-process** — under multi-GPU you'll get a separate VRAM / temperature reading per rank, and a temperature spike on one GPU will only pause the rank pinned to that GPU. The library is not testing this surface in v1.3.

## "xformers disabled on RTX 40/50" rationale

**Symptom:** startup warning *"xformers disabled — RTX 40/50 series (SM 12.0+) is incompatible with the bundled xformers wheel; falling back to native PyTorch SDPA."*

**Why:** the xformers attention kernel wheels for SM 12.0+ (RTX 40 / 50 series cards) have been flaky since the architecture launched — the bundled wheel either fails to load or runs slower than PyTorch's native scaled-dot-product attention (SDPA) for the shapes Backpropagate uses. The library auto-disables xformers on these cards via `BACKPROPAGATE_WINDOWS__XFORMERS_DISABLED=true` (the env var is named `WINDOWS__` for historical reasons but applies to all platforms — the auto-detect keys off GPU compute capability, not OS).

**This is fine.** PyTorch SDPA on Ampere+ is fast (often faster than xformers for the small-batch / long-sequence shapes typical of LoRA fine-tuning) and is what the upstream HF training stack ships with. The warning is informational — there is no action to take.

**If you really want to override** (you have an exotic xformers build, or you're benchmarking): pass `BACKPROPAGATE_WINDOWS__XFORMERS_DISABLED=false`. Expect import errors, segfaults, or silently slower training.

## Mixed-precision pitfalls (bf16 vs fp16)

Backpropagate defaults to `bf16=True, fp16=False` (the default for Ampere+ GPUs — RTX 30 series and newer). Mixed-precision saves ~50% memory and runs ~2x faster than fp32 on supported hardware. Each format has different failure modes.

### bf16 (preferred on Ampere+)

**Why bf16:** wider dynamic range (8-bit exponent, same as fp32) eliminates the underflow class of failures that plagued fp16. Training loss is numerically stable through the whole range LoRA fine-tuning uses.

**The catch:** bf16 has only 7 bits of mantissa (vs fp16's 10). For the rare op where you actually need precision (e.g. small accumulators), bf16 loses ~3 bits of precision. In practice this never matters for LoRA fine-tuning — the gradients and activations live well within bf16's dynamic range. If you see "loss is `nan`" with bf16 on a recent NVIDIA GPU, suspect a tokenizer / data bug, not the dtype.

### fp16 (older GPUs — pre-Ampere)

**Why fp16:** the only mixed-precision path on Turing / Volta (RTX 20-series, V100). Same memory savings as bf16 (16 bits).

**The catch:** fp16's narrow dynamic range (5-bit exponent) causes gradient underflow when small gradients round to zero. The training stack inserts a gradient scaler (`torch.cuda.amp.GradScaler`) to multiply gradients before backward, keeping them above the underflow threshold. When the scaler over-scales and you hit `inf`, the scaler halves itself for the next step.

**Symptoms of fp16 trouble:**

- Loss `nan` early in training → gradient scaler hasn't found a stable scale yet. Wait 10-20 steps; if it doesn't recover, switch to `bf16` if your GPU supports it.
- Loss `inf` mid-training → activations exceeded fp16's max (~65,504). Lower the learning rate or check that your dataset doesn't have outlier-length samples.
- Loss curves looking "noisier" than the equivalent bf16 run → expected. fp16 + scaler trades a little noise for memory savings.

### When to override

Backpropagate auto-picks `bf16` on Ampere+ via the dtype probe in `_detect_features` (and `BACKPROPAGATE_MODEL__DTYPE` is left unset). You only need to override if:

- You're on Turing / Volta and need `fp16` (the auto-detect already picks this — no action).
- You're debugging a `nan` loss issue and want to bisect the dtype: pass `BACKPROPAGATE_TRAINING__BF16=false BACKPROPAGATE_TRAINING__FP16=true` (or vice versa).
- You're running on CPU only for testing (`fp32` — but training on CPU is multi-hour-per-step territory and not recommended).

## Multi-GPU not officially supported

**The short version:** Backpropagate v1.3 targets single-GPU operators on a 16 GB VRAM workstation. Multi-GPU works in many cases via `accelerate launch` (see [the recipe](/backpropagate/handbook/recipes/#fine-tune-on-multi-gpu)), but the library does not test the multi-GPU surface, the GPU-monitoring is per-process, and NCCL failure modes are not in the v1.3 retryable-error matrix.

**What "not officially supported" means:**

- Bug reports tagged "multi-GPU" are triaged but not prioritised.
- The CI matrix does not include a multi-GPU cell.
- The Unsloth backend may not work cleanly across ranks — start with `--no-unsloth` if you hit `unsloth` import errors under `accelerate launch`.
- Memory accounting is per-process; if a single rank OOMs, the OOM-recovery on that rank halves the batch size only on that rank — you'll get rank-divergent state.

**If you have multiple GPUs and want to fine-tune big models:** consider running training under HuggingFace's `transformers.Trainer` directly with FSDP or DeepSpeed, then re-use only `backpropagate.export` for the GGUF + Ollama step.

## Driver / CUDA version mismatch

**Symptom:** at startup, you see `DEP_GPU_NOT_AVAILABLE` despite `nvidia-smi` showing your GPU; or `torch.cuda.is_available()` returns `False`; or CUBLAS errors firing on the first matmul.

**Cause:** the PyTorch wheel you installed is built against a CUDA version your driver does not support. The compatibility table:

| Driver CUDA capability (top-right of `nvidia-smi`) | Compatible PyTorch CUDA builds |
|----------------------------------------------------|--------------------------------|
| 12.x | CUDA 11.8, 12.1, 12.4, 12.6 (any) |
| 11.8 | CUDA 11.8 (only) |
| 11.7 | CUDA 11.7 or 11.8 |
| older | upgrade your driver |

**Fix:** uninstall the wrong PyTorch wheel and reinstall the right one. Visit <https://pytorch.org/get-started/locally/> and pick the `pip install torch ...` command for your CUDA version. On most modern setups:

```bash
pip uninstall torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu124
# (or cu121 / cu118 — match your driver)
```

Then verify:

```bash
backprop info
```

`backprop info` prints the resolved Python / PyTorch / CUDA / GPU details and the optional-features matrix — confirm the CUDA line matches your driver before re-running training.

**On Windows:** the `bitsandbytes` wheel must also match — use `pip install bitsandbytes >= 0.43`; older versions had a separate `bitsandbytes-windows` fork that is no longer needed. See [troubleshooting → Windows-specific install pain](/backpropagate/handbook/troubleshooting/#windows-specific-install-pain) for the full Windows checklist.

## See also

- [Troubleshooting](/backpropagate/handbook/troubleshooting/) — symptoms-first index (start here if you don't know it's a CUDA problem yet).
- [Error codes](/backpropagate/handbook/error-codes/) — every `RUNTIME_GPU_*` / `DEP_GPU_*` code and what triggers it.
- [Environment variables](/backpropagate/handbook/env-vars/) — every `BACKPROPAGATE_*` knob, including the Windows / CUDA-related ones.
- [Recipes → multi-GPU](/backpropagate/handbook/recipes/#fine-tune-on-multi-gpu) — the unofficial `accelerate launch` path.
- [Reference → GPU safety](/backpropagate/handbook/reference/#gpu-safety) — the temperature / VRAM / utilization monitor thresholds.
