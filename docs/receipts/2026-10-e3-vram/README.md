# E3: `estimate_vram` recalibration and auto-batch (receipts index)

Status: **pre-registered, no pod run yet.** This file was committed before any
measurement. The first section (the gate and the design) is frozen: later commits
may add results and findings below it, not change it.

Experiment E3 of `docs/handoff-2026-09-30-full-ft-experiments.md` (section 4, plan of
record in 4a): the QLoRA presets' `estimate_vram` under-reads real GPU memory by
25-45% (handoff 3.2), and `_detect_batch_size` resolves batch 6 on a 32 GB card
whatever the model, which the 14B-32B presets very likely cannot run at their
window. This directory holds the pre-registration, the CPU-only results and, after
the pod run, the receipts.

## 1. Pre-registration (frozen)

### 1.1 The gate

Handoff text: *estimator within +-15% of NVML on every measured point; no preset
OOMs at its default batch.* Made precise, computed by `scripts/e3_refit.py`
(which prints PASS or FAIL and exits 0 / 1; 2 means not enough data):

* **Measured point**: one (preset, batch, Unsloth on/off) with a receipt whose
  `status` is `ok`. Its measured figure is `peak_gib_for_fit`: the NVML device peak
  sampled at 100 Hz for the whole process, or `torch max reserved + CUDA context +
  paged optimizer state` when that is larger (a sampler can miss a transient).
  PyTorch's own counters alone are never used: they miss bitsandbytes' paged state
  (handoff 3.2).
* **The estimator under test** is `backpropagate.trainer.estimate_vram` with the
  *proposed* `VRAMCoefficients`, fed the model's real hidden size / layers / heads /
  vocab and its text-only parameter count, all read from its config.
* **G1**: at every measured point, `|estimate - measured| / measured <= 0.15`.
* **G2**: no preset OOMs at its default batch. The default batch is the one the
  proposed estimator picks for that preset and Unsloth arm: the largest of
  1, 2, 4, 6 whose predicted total is at most 95% of the card's memory (batch 1 if
  none is). That point must have been measured and must be `ok`; an `oom`, a
  `skipped_monotone_oom` or an unmeasured point is a G2 failure. (What today's
  `_detect_batch_size` would pick, 6 on a 32 GB card, is reported next to it and is
  not gated: changing auto-batch follows the refit, in a later PR.)
* **PASS = G1 and G2.**

Fixed now, not tuned toward: tolerance 15%, safety margin +5% (every fitted scale
and the fixed term are multiplied by 1.05, so the proposal leans toward
over-predicting), headroom 95%, 8 steps per point.

Reported but not gated: the error per measured point before (as shipped, with the
7B-class default dimensions the CLI and `Trainer.estimate_vram()` use; and as
shipped with the real dimensions) and after (without and with the margin); a
leave-one-model-out error per preset (how the fit generalises); the fit's rank and
condition number.

If the gate FAILS the receipts are still the deliverable. The lead then decides
between adding addends (the table is extensible) and a measured per-preset
batch table for auto-batch. The thresholds do not move.

### 1.2 What is measured and how

* **Presets**: every entry of `MODEL_PRESETS` (12), measured small to large:
  llama-3.2-1b, llama-3.2-3b, qwen2.5-3b, smollm3-3b, phi-4-mini-3.8b, qwen3.5-4b,
  mistral-7b, qwen2.5-7b, llama-3.1-8b, qwen2.5-14b, mistral-small-24b,
  qwen2.5-32b. Each at the preset's own `recommended_lora_r` (alpha = rank) and
  `recommended_max_seq_length`, the library defaults otherwise (paged 8-bit AdamW,
  gradient checkpointing as the trainer sets it, bf16).
* **Batches** 1, 2, 4, 6 (1 is the floor `oom_recovery` halves down to; 6 is what
  `_detect_batch_size` picks on a 32 GB card). Gradient accumulation 1.
* **Unsloth** off and on, each point in a fresh process (an OOM, a CUDA error or a
  leaked paged buffer must not touch the next point). The Unsloth extra is
  installed for the whole run; "off" is `use_unsloth=False` in that environment.
  `unsloth_fallback=False`: a failing Unsloth load is the result, never a silent
  fall back to the plain path.
* **96 points**, one JSON receipt each (`runs/e3_<preset>_b<batch>_<plain|unsloth>.json`).
* **Every step is at the preset's full window**: packing off, rows longer than the
  window, so each step input is exactly `(batch, window)` tokens (checked per point:
  `shape_ok`, `step_input_shapes`).
* **`oom_recovery=False`**: an out-of-memory is recorded as the result for the
  point (`status: oom`, with the phase it happened in), never silently halved.
  `oom_retries` is recorded and must be 0 or absent.
* **8 steps per point.** The paged optimizer state is allocated lazily at the first
  optimizer step, so activations and optimizer state first coexist in step 2, and
  the caching allocator settles by step 3-4. Eight steps leave at least four
  steady-state steps for the median seconds per step and a plateau check
  (`plateau_ok`: the running NVML peak after the last step is within 1% of the peak
  after step 4).
* **No `PYTORCH_CUDA_ALLOC_CONF`**: the sweep measures what a user's process gets
  with library defaults (stage d used `expandable_segments`, which lowers reserved
  memory and would flatter the estimator).
* Recorded per point: NVML peak (and per-step running peak), torch max allocated /
  reserved (and per step), paged optimizer state size, CUDA context, seconds per
  step, load seconds, what `estimate_vram` predicts for the point (`predicted_default`
  and `predicted_arch`), what `_detect_batch_size` would choose
  (`auto_batch_choice`), whether `oom_recovery` fired, the model class and the
  number of vision tensors actually loaded, versions, git SHA, image, seed.

### 1.3 Budget, disk, drop order

* Pod C, `RUNPOD_CONTAINER_DISK_GB=200`; HF cache on the container disk. A
  preset's weights are deleted after its points (the largest pair held at once, with
  the next preset prefetched, is about 112 GB); the sweep's share of pod C's $2.00
  cap is **$1.20** (the rest is E1 validation).
* The guard drops points in this order (first dropped first): highest tier number
  first; within a tier Unsloth-on before Unsloth-off, then later position first.
  Tiers: **0** 14B/24B/32B at batch 1 and 6 (the latent auto-batch question,
  protected longest); **1** 7B-13B at batch 1 and 6; **2** under 7B at batch 1 and
  6; **3** 14B+ at batch 2 and 4; **4** 7B-13B at batch 2 and 4; **5** under 7B at
  batch 2 and 4. It is applied once before the run (a plan) and again at run time
  from observed timings, and `E3_STOP_AFTER_EPOCH` is a hard wall.
* A partial sweep can still PASS G1, but G2 needs the proposed default batch of
  every preset measured; `e3_refit.py` prints the exact top-up commands for any
  point it lacks.

## 2. Findings and results

None yet. (CPU-only findings and the pod plan are added below in later commits.)
