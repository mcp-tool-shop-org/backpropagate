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

## 2. Amendments (before any pod measurement)

* **A1 (2026-09-30).** The sweep passes `train_on_responses=False` to every
  `Trainer`. The library default is `True`, which only matters with Unsloth and
  only changes which labels are masked, not memory; leaving it on would let an
  Unsloth response-marker detection failure on an exotic chat template void a whole
  arm. The receipt's `trainer_kwargs` records it. Everything else in 1.2 stands.

## 3. What was built (all of it CPU-tested; nothing has run on a GPU)

| Piece | Where |
|---|---|
| Table-driven `estimate_vram` (`vram_addends` x `VRAMCoefficients`), **identical outputs, no constant changed** | `backpropagate/trainer.py`, proof in `tests/test_vram_estimate_table.py` |
| Sweep driver (planner, per-point processes, NVML sampler, budget guard, prefetch / purge, receipts, summary, CPU dry run) | `scripts/pod_e3_sweep.py`, `scripts/pod_e3_sweep.sh`, `scripts/e3_lib.py` |
| Refit tool (proposes coefficients, computes the gate) | `scripts/e3_refit.py` |
| Meta-device parameter / vision-tower check | `scripts/e3_meta_device_check.py` |
| Architecture table (config only) | `arch/<preset>.json` (10 of 12; the two gated Llama-3.2 repos need the pod's token) |

**Table equality.** `tests/test_vram_estimate_table.py` embeds the pre-change
`estimate_vram` verbatim (from `origin/main` at `35ca73e`; the only edits are the
function name and one relative import) and asserts `==` on the whole `VRAMEstimate`,
floats and notes included, over 3,324 cases: all 12 presets x batch 1/2/4/6/8 x seq
512/2048/4096/8192 x `lora`/`full` x quantized or not x three architectures, plus
bytes-per-param, LoRA rank, overhead fraction, offload, vocab size and unknown model
ids. The shipped `VRAMCoefficients` are the identity table (every scale 1.0, the new
`embedding` and `logits` addends and the fixed term 0.0, the Unsloth factors 1.0).

**Why the table has an `embedding` and a `logits` addend (scale 0.0 today).** The
estimator has no vocabulary term. bitsandbytes leaves the embedding (and an untied
head) in bf16, which the 0.5 byte/param weights line cannot see (152,064 x 5,120 x 2
bytes is 1.5 GiB per copy on the 14B-32B Qwens), and the fp32 logits of a batch of
full windows are batch x window x vocab x 4 bytes (about 7 GiB at batch 6 x 2,048 x
152k). Both are plausible gaps; whether they matter is for the sweep to say, and a
zero-valued addend costs nothing to ship.

## 4. Baseline before the sweep (existing receipts, no pod)

`estimate_vram` as shipped, called the way `Trainer.estimate_vram()` and the CLI call
it (7B-class default dimensions, name-derived parameter count), and with the real
dimensions, against the measurements already in `docs/receipts/`:

| Point | Measured (GiB) | As shipped | Error | Real dims | Error |
|---|---|---|---|---|---|
| qwen2.5-3b, r256, b4, seq 512 | 8.28 (NVML, stage d) | 6.06 | -27% | 4.16 | -50% |
| qwen2.5-7b, r256, b4, seq 512 | 13.84 (NVML, stage d) | 8.20 | -41% | 7.49 | -46% |
| qwen2.5-14b, r32, b1, seq 4096 | 28.10 (torch reserved, preset smoke) | 10.75 | -62% | 14.01 | -50% |
| mistral-small-24b, r32, b1, seq 4096 | 29.58 (torch reserved) | 16.10 | -46% | 17.71 | -40% |
| qwen2.5-32b, r32, b1, seq 2048 | 30.71 (torch reserved) | 18.95 | -38% | 22.08 | -28% |

The estimator under-reads by 27-62% at every anchor (the handoff said 25-45%), and
feeding it the true dimensions does not fix it (it gets worse for the small models):
what is missing is structure (embedding, logits, a fixed term), not just inputs. The
gate is +-15%; this is the distance to cover.

## 5. CPU-only findings

### 5.1 `Qwen/Qwen3.5-4B` (the `qwen3.5-4b` preset): does it load the vision tower?

**No, on the transformers path.** Measured on the meta device from `config.json`
alone (`scripts/e3_meta_device_check.py`, transformers 5.5.0; JSON in
`cpu/qwen3.5-4b_meta.json`):

| | Parameters | Tensors |
|---|---|---|
| Text-only model the library loads (`AutoModelForCausalLM` -> `Qwen3_5ForCausalLM` on the text config; tied embeddings) | **4,205,751,296** (4.206 B) | 426 |
| Composite `Qwen3_5ForConditionalGeneration` (language model + vision tower) | 4,539,265,536 | 723 |
| of which the vision tower | 333,514,240 | 297 |
| Hub `safetensors.total` for the repo | 4,659,865,088 | 738 |
| Model card ("Number of Parameters") | "4B" | |

The Hub total exceeds the composite by 120,599,552 parameters: the checkpoint also
holds a 15-tensor multi-token-prediction head (`mtp.*`) that no transformers class
instantiates. So of the 738 tensors in the repo: 426 language model (loaded), 297
vision (not instantiated), 15 MTP (not instantiated). Neither the vision tower nor
the MTP head is loaded by `Trainer._load_with_transformers`, which calls
`AutoModelForCausalLM.from_pretrained`; they are still downloaded (same shards, about
0.9 GB). The preset description's claim "the trainer loads it text-only" holds, and
the 8.4 GiB QLoRA peak measured on 2026-09-30 is for the text-only model. The
estimator's name-derived 4.0 B is 5% under the real 4.206 B text-only count.

Architecture note: 32 layers, 24 of them linear-attention (gated delta net) and 8
full attention, vocab 248,320 (the largest in the preset set, so the embedding and
logits terms matter most here).

**Unsloth path: not measured, and a reason to expect a difference.** Unsloth
2026.5.8 (installed here) has no `qwen3_5` entry anywhere in its package, so the
model would take its generic loader, where `unsloth/models/loader.py` sets
`is_vlm = ... hasattr(model_config, "vision_config")` and picks
`AutoModelForVision2Seq`, which instantiates the vision tower (+333.5 M parameters).
Every Unsloth-on point records `model_class`, `vision_tensors_loaded` and
`vision_params_loaded`, so the sweep settles it.

### 5.2 `meta-llama/Llama-3.1-8B-Instruct` (the `llama-3.1-8b` preset) smoke

Gated repo; it needs `HF_TOKEN` on the pod, in the environment only. It is not run
locally. It is the existing opt-in smoke `tests/test_qlora_presets_smoke.py[llama-3.1-8b]`
(2 real QLoRA steps at the preset's rank 16 and window 4096, adapter saved, rank
checked), wrapped as the `llama` stage of the sweep script (command in 6). The
sweep also measures this preset at batch 1/2/4/6. The Llama-3.2 1B / 3B presets are
gated too: their configs returned 403 with this rig's token, so their architecture
table rows come from the pod (`arch` stage); the account behind the pod's token must
have accepted the `meta-llama/Llama-3.2-*` licences, or those 16 points are skipped
with a `skipped_no_access` receipt (never silently).

### 5.3 Architectures the refit will use (config only; `arch/<preset>.json`)

| Preset | Window | LoRA r | Name-derived B | Text-only params (B) | Hidden x layers | Heads / KV | Vocab | Tied | Loaded class |
|---|---|---|---|---|---|---|---|---|---|
| llama-3.2-1b | 2048 | 64 | 1 | from the pod (gated) | | | | | |
| llama-3.2-3b | 2048 | 128 | 3 | from the pod (gated) | | | | | |
| qwen2.5-3b | 2048 | 128 | 3 | 3.086 | 2048 x 36 | 16 / 2 | 151936 | yes | Qwen2ForCausalLM |
| smollm3-3b | 8192 | 128 | 3 | 3.075 | 2048 x 36 | 16 / 4 | 128256 | yes | SmolLM3ForCausalLM |
| phi-4-mini-3.8b | 2048 | 128 | 3.8 | 3.836 | 3072 x 32 | 24 / 8 | 200064 | yes | Phi3ForCausalLM |
| qwen3.5-4b | 4096 | 128 | 4 | 4.206 | 2560 x 32 | 16 / 4 | 248320 | yes | Qwen3_5ForCausalLM |
| mistral-7b | 2048 | 256 | 7 | 7.248 | 4096 x 32 | 32 / 8 | 32768 | no | MistralForCausalLM |
| qwen2.5-7b | 2048 | 256 | 7 | 7.616 | 3584 x 28 | 28 / 4 | 152064 | no | Qwen2ForCausalLM |
| llama-3.1-8b | 4096 | 16 | 8 | 8.030 | 4096 x 32 | 32 / 8 | 128256 | no | LlamaForCausalLM |
| qwen2.5-14b | 4096 | 32 | 14 | 14.770 | 5120 x 48 | 40 / 8 | 152064 | no | Qwen2ForCausalLM |
| mistral-small-24b | 4096 | 32 | 24 | 23.572 | 5120 x 40 | 32 / 8 | 131072 | no | MistralForCausalLM |
| qwen2.5-32b | 2048 | 32 | 32 | 32.764 | 5120 x 64 | 40 / 8 | 152064 | no | Qwen2ForCausalLM |

## 6. The pod plan (estimates; nothing has run)

**Cost.** Estimates come from `scripts/pod_e3_sweep.py plan` (a cost model fitted to
stage d: 0.41 s/step at 2,048 tokens for 3B and 0.51 s/step for 7B, load 14.8 s and
16.2 s from a warm cache, then compute-bound by parameter count beyond that; the
14B+ load and step times are extrapolations, so treat them as +-40%), at $0.90/h:

| Plan | Points | Pod time | Cost |
|---|---|---|---|
| Full grid (12 presets x 4 batches x 2 arms) | 96 | ~2.6 h | ~$2.33 |
| Default cap $1.20 (the sweep's share of pod C's $2.00; the guard keeps tiers 0 and 1 whole and 18 of tier 2) | 42 | ~1.3 h | ~$1.20 |
| `UNSLOTH=off` (the whole plain arm, all tiers) | 48 | ~1.3 h | ~$1.2 |
| Cap $2.00 | 70 | ~2.2 h | ~$1.98 |

All times include 10 minutes for boot, install and preflight. The handoff budgeted
E3 at about 1 h and $1; the full grid is more than twice that, which is why the plan
is guarded. Note the trade: at $1.20 with both arms the guard drops all of the batch
2 and 4 points (tiers 3-5), which G2 can need (the proposed default batch for a 14B+
preset is likely 1 or 2). `e3_refit.py` prints the exact top-up commands for any
point G2 lacks. The alternative at the same price is `UNSLOTH=off` (all 48 plain
points, tiers 0-5), then a top-up for the Unsloth arm. The first point dropped is
`e3_qwen3.5-4b_b4_unsloth`; tier 0 goes last.

**Disk.** `RUNPOD_CONTAINER_DISK_GB=200`, HF cache at `/root/hf-e3` (container disk;
the sweep halts if it is on a network mount or under 85 GB free). bf16 repo sizes
(2 x Hub parameter count): 1B 2.5 GB, 3B x3 about 6.2, 3.8B 7.7, 4B 9.3 (includes the
vision tower and MTP head), 7B x2 about 15, 8B 16, 14B 29.5, 24B 47, 32B 65.5; about
226 GB over the whole run, but each preset is purged after its points. The largest
pair held at once (24B + 32B, with the next preset prefetched during the current
measurement) is about 112 GB, which leaves roughly 60 GB for the image, venv and
checkpoints (the sweep saves none). Without prefetch the peak is 65.5 GB.

**Exact commands** (lead). Agents make no RunPod calls; these are yours:

```bash
# 1. pod C: 200 GB container disk, dead-man armed BEFORE anything else, then verify CUDA
RUNPOD_CONTAINER_DISK_GB=200 scripts/runpod/pod.sh create-retry "NVIDIA GeForce RTX 5090" 64 bp-e3 45
powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath python -ArgumentList 'scripts/runpod/deadman.py','<POD_ID>','2.3','deadman-e3.log'"
scripts/runpod/pod.sh verify <POD_ID>

# 2. put the script on the pod (the install stage clones the branch itself)
scp -i ~/.ssh/runpod_rustline -P <PORT> scripts/pod_e3_sweep.sh root@<IP>:/root/

# 3. on the pod: ssh in INTERACTIVELY (the token must never be on a command line)
tmux new -s e3
export HISTFILE=/dev/null
read -rs HF_TOKEN; export HF_TOKEN            # paste, Enter: no echo
BRANCH=feat/e3-vram-sweep nohup bash /root/pod_e3_sweep.sh > /dev/null 2>&1 &
tail -f /root/e3/pod.log                      # ends with "RESULT: E3 gate PASS|FAIL|INSUFFICIENT DATA"

# 4. optional, same shell, before deleting the pod: the Llama-3.1-8B preset smoke
STAGES="llama" bash /root/pod_e3_sweep.sh

# 5. top-ups the refit lists for G2 (example), then re-run the verdict
FORCE_STAGE=sweep STAGES="sweep refit" PRESETS=qwen2.5-14b BATCHES=2 UNSLOTH=off EXTRA_ARGS=--no-guard \
  bash /root/pod_e3_sweep.sh

# 6. receipts back (they contain no token; the script's leak stage checks), then delete the pod
scp -r -i ~/.ssh/runpod_rustline -P <PORT> root@<IP>:/root/e3/{runs,arch,plan.json,summary.json,refit.json,pod.log} \
  docs/receipts/2026-10-e3-vram/
scripts/runpod/pod.sh delete <POD_ID>
```

Useful knobs: `E3_BUDGET_USD` (default 1.20), `E3_STOP_AFTER_EPOCH` (hard wall),
`UNSLOTH=off|on|both`, `STEPS`, `PRESETS`, `BATCHES`. The refit can also be re-run
locally: `python scripts/e3_refit.py --receipts docs/receipts/2026-10-e3-vram --out refit.json`.

**Smoke it for free first** (CPU, no network, no weights; about a minute for 4
points, 5 minutes for 36):

```bash
python scripts/pod_e3_sweep.py run --out /tmp/e3dry --synthetic --presets llama-3.2-1b,qwen2.5-32b \
  --batches 1,2 --unsloth off --steps 3
python scripts/e3_refit.py --receipts /tmp/e3dry --allow-dry-run     # plumbing only: RSS stand-in peaks
```

Use a different `--out` for a dry run than for the real one.

## 7. Files

| Path | Contents |
|---|---|
| `README.md` | this file: the frozen pre-registration (1), amendments, findings, plan |
| `arch/<preset>.json` | architecture + text-only parameter count from config (meta device) |
| `cpu/qwen3.5-4b_meta.json` | the Qwen3.5-4B meta-device / model-card comparison |
| `runs/e3_*.json` | (after the pod run) one receipt per point |
| `summary.json`, `plan.json`, `refit.json`, `pod.log` | (after the pod run) |

## 8. Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | Every point records git SHA, image, versions, seed, preset, every Trainer argument; the grid is a pure function of its inputs (`e3_lib.build_points`). |
| ANDON_AUTHORITY | 2 | The orchestrator halts before spending on a dead GPU, a busy GPU, a network-mounted or too-small HF cache, or a missing Unsloth extra, and mid-run if a child leaves GPU memory held; the budget guard and `E3_STOP_AFTER_EPOCH` stop the run. |
| NAMED_COMPENSATORS | 2 | Pod create/delete are the lead's (`pod.sh delete`, dead-man timer). HF-cache purges are re-downloadable. Nothing is published or pushed by the scripts. |
| DECOMPOSE_BY_SECRETS | 2 | Planning (`e3_lib`), measuring (`pod_e3_sweep`), fitting (`e3_refit`) share only the receipt schema. |
| UNCERTAINTY_GATED_HUMANS | 2 | The refit proposes; the lead applies, and decides the batch-table fallback if the gate fails. Spend above the cap needs the Director. |
| EXTERNAL_VERIFIER | n/a | No specialized claims: a numerical rule on measured data. |
