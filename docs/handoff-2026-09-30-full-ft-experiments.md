# Handoff: full fine-tuning experiments (after v1.7.2, 2026-09-30)

> **Read this first in any session that continues the GPU experiments.**
> It records what was measured, where the evidence lives, what is still open,
> and how to run the next experiments without repeating the day's mistakes.
> Everything below was true at the end of 2026-09-30; re-check live state
> (`gh`, PyPI, the receipts) before acting on any of it.

## 1. Where things stand

- **v1.7.2 is released** (tag `v1.7.2` → `ea48ee5`): PyPI, npm, GitHub
  Release and GHCR all at 1.7.2; the post-publish smoke passed on Ubuntu and
  Windows, Python 3.10–3.13. The plan it executed:
  [`docs/production-quality-plan-2026-09-30.md`](production-quality-plan-2026-09-30.md).
- **Engine B (block-coordinate AdamW) is on `main` as experimental**
  (`--full-ft-engine block`, PR #237), for 1.8.0. It failed its
  pre-registered gate (section 3.3); the Director chose to keep it as an
  opt-in experiment. Docs say so plainly.
- Design and evidence:
  [`docs/full-ft-engines-design-2026-09-30.md`](full-ft-engines-design-2026-09-30.md).
- External review (Kimi K3, run by hand by the Director) and the lead's
  assessment: [`docs/consult/`](consult/).
- No pods are running. RunPod spend on 2026-09-30: about **$5.60**
  (budget given: $10 for experiments).

## 2. The three engines for full fine-tuning

| | Standard (on the GPU) | Engine A: `--full-ft-offload` | Engine B: `--full-ft-engine block` |
|---|---|---|---|
| What trains | every weight, every step | every weight, every step | one block for K steps, then the next |
| Memory | weights + grads + paged 8-bit AdamW on GPU | bf16 weights + grads in host RAM; Adafactor state on GPU | whole model bf16 on GPU; active block fp32 + AdamW |
| 7.6B on a 32 GB card | does not fit | 5.3 GiB VRAM, 30.8 GiB host, **14.7 s/step** | 30.0 GiB VRAM (NVML), **0.21 s/step** |
| Platform | any | Linux / WSL2 (NCCL) | any |
| Training loop | TRL SFTTrainer | own plain SFT loop (no packing, masking, checkpoints, resume, accum > 1 untested) | TRL SFTTrainer (everything works) |
| Status | shipped | shipped in 1.7.2 | experimental, on `main` for 1.8.0 |

Code: `backpropagate/offload_engine.py` (A), `backpropagate/block_engine.py`
(B), both wired in `backpropagate/trainer.py`.

## 3. What was measured (all on an RTX 5090, 32 GB)

Receipts are committed; every number below has a JSON source.

| Folder | Contents |
|---|---|
| `docs/receipts/2026-09-30-offload/` | Engine A: 7B final receipt, size ladder 1.5B–7.6B, VRAM-cap runs, 3B quality vs standard, QLoRA preset smokes |
| `docs/receipts/2026-09-30-block-engine/` | Engine B first run: Dolly, 150 steps, 1.5B / 3B / 7B, rounding × K |
| `docs/receipts/2026-09-30-gsm8k/` | GSM8K bake-off: per-item outputs, paired statistics, NVML peaks, loss traces |

### 3.1 Engine A (offload)

- 7.6B: 30.8 GiB host RSS training, 32.2 with save + reload; 5.3 / 14.7 GiB
  VRAM allocated / reserved; 14.7 s/step at 512 **and** 2048 tokens
  (transfer-bound). Loss 0.341 → 0.0067 on an overfit set; save → reload →
  generate works.
- Host-RAM fit check: 3.73 GiB per billion params + 10.1 GiB (conservative:
  38.5 predicted vs 32.2 measured at 7.6B). Under a 28 GB WSL2 cap the limit
  is about 4.8B.
- Precision: bf16 round-to-nearest keeps 17% of each intended update (run
  stops learning); stochastic rounding keeps ~100%. Kahan matched fp32 on one
  360M run (contradicts Collage, arXiv:2405.03637 — unresolved).
- 3B on Dolly (1 seed): held-out 2.45 → 1.93 vs 1.84 for standard full FT.
  The GSM8K Adafactor control (standard full FT with Adafactor: 0.534 vs
  0.539 with AdamW) shows **the optimizer is not the cause**, so the gap is
  offload numerics or the dataset. Unresolved.

### 3.2 VRAM truth

- **PyTorch's memory counters miss bitsandbytes' paged optimizer state**
  (CUDA managed memory, `bitsandbytes.functional.get_paged`). Standard full FT
  at 3B: 13.4 GiB torch-reserved vs **22.0 GiB NVML** (7.5 GiB paged state).
  Always sample NVML (`pynvml.nvmlDeviceGetMemoryInfo`) alongside torch.
- QLoRA presets at full window, batch 1: 14B 25.0 / 24B 26.5 / 32B 28.8 GiB.
- `estimate_vram` under-reads 14B–32B QLoRA by 25–45%; for 3B full FT it was
  close to NVML (20.1 vs 22.0).

### 3.3 GSM8K bake-off (the engine B gate)

1000 steps, batch 4, 512 tokens, all 7,473 train rows (0.535 epoch),
held-out answer loss on 250 test questions (primary), strict accuracy
(secondary, McNemar + paired bootstrap).

| Arm | Held-out loss | Accuracy | s/step | NVML peak |
|---|---|---|---|---|
| 7B QLoRA r=256 | **0.513** | 0.696 | 0.51 | 13.8 GiB |
| 7B engine B K=5 (2 seeds) | 0.565 | 0.734 | 0.21 | 30.0 GiB |
| 7B GaLore layerwise 8-bit | 0.589 | 0.760 | 0.52 (1.27 mean) | 23.3 GiB |
| 3B standard full FT (3 seeds) | 0.539 | 0.643 | 0.30 | 22.0 GiB |
| 3B engine B K=50 (3 seeds) | 0.546 | 0.663 | 0.14 | 15.3 GiB |
| 3B engine B K=5 (3 seeds) | 0.550 | 0.653 | 0.14 | 15.4 GiB |
| 3B QLoRA r=256 (2 seeds) | **0.522** | 0.578 | 0.41 | 8.3 GiB |

- Engine B vs QLoRA at 7B: loss +0.052 (95% CI 0.044–0.060); accuracy
  +0.038, not significant (McNemar p 0.16 / 0.43). **Gate: drop** (rule:
  ship only if ≥0.05 nats better or accuracy McNemar p<0.05). Kept as
  experimental by Director decision.
- **Caveat that limits every conclusion above:** fine-tuning on GSM8K made
  all models *worse* at the math (untrained lenient accuracy 0.868 at 7B,
  0.796 at 3B; every trained arm ≤ 0.768 / 0.663). The accuracy gains are
  mostly format learning. This test cannot show where full fine-tuning should
  beat LoRA (Biderman et al. 2024, arXiv:2405.09673: code and math, larger
  data). That question is **still open**.
- A first attempt was invalid: the library's default `max_samples=1000`
  silently capped the data (4 epochs over 1000 rows). That default is fixed
  in 1.7.2 (#241). Its results are kept separately as `x1000_*` and must not
  be mixed in.

## 4. Open experiments, in priority order

Each has a pre-registered decision rule. Write the rule into the PR or the
receipt README **before** the run; do not tune toward it.

### E1. Engine A speed (highest value, mostly CPU work first)

- **Why:** 14.7 s/step at 7.6B. Per step it moves about 91 GB across PCIe:
  forward and backward re-gathers, gradients down, then the optimizer
  re-streams parameters and gradients up and writes them back. For tensors
  over one chunk (2^26 elements: every 7B MLP weight, the embeddings) the
  gradient is streamed **three** times (`OffloadAdafactor.step`, passes 1–3).
- **Plan:** (1) instrument each leg with CUDA events for one step, plus a
  `torch.profiler` trace; (2) fuse the optimizer update into the backward
  pass (step a layer while its params and grads are on the GPU; write bf16
  back once); (3) one pinned host arena with per-layer views instead of
  `cudaHostRegister` per storage; (4) prefetch layer i+1 on a copy stream.
- **Expected:** ~3–4.5 s/step after (1)–(3), ~2–2.5 with (4) (Kimi K3's
  estimate; unmeasured).
- **Gate:** ship if s/step at 7.6B drops ≥3× with identical loss trace
  (same seed, ±1e-3 over 20 steps) and host RSS no higher.
- **Cost:** CPU development, then ~30 min of pod time.

### E2. Pure-GPU full-FT ceilings (correctness of shipped code)

- **Why:** `_FULL_FT_VRAM_CEILING_TIERS` (16→4B, 24→5B, 32→6B) is arithmetic.
  At 3B the real peak is 22.0 GiB NVML on a 32 GB card, but 7.5 GiB of it is
  paged state that can spill to host RAM on a smaller card.
- **Plan:** real 16 GB and 24 GB cards on RunPod (e.g. RTX 4080 / A4000 16 GB,
  RTX 4090 24 GB): 1.5B / 3B / 4B at batch 1 and 4, 512 and 2048 tokens;
  record NVML peak, s/step, and whether paging made it slow (compare with the
  same run on the 5090). `set_per_process_memory_fraction` does **not** cap
  managed memory, so emulating a small card on the 5090 is not valid here.
- **Gate:** set each tier to the largest size that trains at ≤1.5× the 5090's
  per-token speed without OOM at batch 1, 2048 tokens.
- **Cost:** ~1.5 h across two pods, ~$1.50.

### E3. `estimate_vram` recalibration and auto-batch

- **Why:** QLoRA under-read 25–45%; `_detect_batch_size` resolves batch 6 on
  32 GB, which the 14B–32B presets very likely cannot run at their window.
- **Plan:** NVML peaks for each preset at batch 1/2/4/6 and the default
  window, with and without Unsloth (Unsloth now actually engages, #230).
  Refit the estimator's addends against those points; make auto-batch
  model-size aware.
- **Gate:** estimator within ±15% of NVML on every measured point; no preset
  OOMs at its default batch.
- **Cost:** ~1 h, ~$1.

### E4. Where full fine-tuning should beat QLoRA (decides engine B's future)

- **Why:** the GSM8K test could not show it (section 3.3 caveat).
- **Plan:** a code task (e.g. a permissively licensed Python instruction set,
  evaluated with unit-test pass rate) or a larger dataset with 2+ epochs, at
  3B where standard full FT fits: standard full FT vs QLoRA r=256 vs engine B
  vs GaLore layerwise. Then 7B (engine B, GaLore, QLoRA). 3 seeds at 3B,
  paired statistics on per-item outputs, NVML, loss traces, the base model's
  score first (it must sit strictly between floor and ceiling).
- **Gate:** if standard full FT does not beat QLoRA at 3B, stop — no engine
  for full FT at 7B has a reason to exist on this task. If it does, engine B
  ships out of experimental only if it beats QLoRA at 7B (same rule as 3.3).
- **Cost:** ~2 h, ~$2.

### E5. Engine A long-run numerics

- **Why:** short runs cannot show stochastic-rounding noise floors, the
  momentum-free Adafactor wobble, or bf16 gradient accumulation loss.
  The 3B gap to standard full FT (section 3.1) is unexplained.
- **Plan:** engine A on the GSM8K harness at 3B, 2000 steps, 2 seeds; log
  per-tensor stagnation-pressure fraction (share of |Δ_fp32| < ulp/2),
  stochastic-flip fraction, and an fp32 shadow copy of one small tensor; a
  gradient-accumulation 1 vs 4 comparison.
- **Gate:** within the seed spread of standard full FT on held-out loss, and
  shadow-tensor drift below 1% of the tensor's RMS; otherwise document the gap
  and consider Kahan (+2 B/param).
- **Cost:** ~1.5 h, ~$1.50. Best run **after** E1 (5× cheaper per step).

### Smaller items

- Does `Qwen/Qwen3.5-4B` (the fixed `qwen3.5-4b` preset; Hub tag
  image-text-to-text) load its vision tower? Compare the loaded parameter
  count with the model card.
- Llama-3.1-8B preset smoke: gated repo; needs an HF token on the pod.
- Engine B: K default (50 vs 5 only differed by 0.0035 nats at 1000 steps);
  `--block-freeze-embeddings` quality cost.

## 5. How to run an experiment (the procedure that worked)

1. **Nothing heavy runs on the Director's machine.** He runs several
   sessions; local training ties it up. GPU work goes to RunPod. Local is for
   CPU tests and tiny (135M) Windows-specific checks only. README
   translations (TranslateGemma 27B) are the exception and run locally.
2. **Write the decision rule first.** Thresholds go in the PR / receipt
   README before the pod exists.
3. **Build the script and the CPU tests locally** in a worktree; an agent may
   do this. Agents make **no** RunPod API calls; the lead creates and deletes
   pods.
4. **Create the pod:** `scripts/runpod/pod.sh create-retry` (RTX 5090,
   ≥64 GB RAM; 5090 capacity is often zero — the loop retries). Immediately
   arm `scripts/runpod/deadman.py` (budget + margin). Then
   `scripts/runpod/pod.sh verify <id>` — check `torch.cuda.is_available()`
   before handing the pod to anyone.
5. **On the pod:** keep the HF cache on the container disk (`/workspace` is
   often a MooseFS network mount); run in tmux; log and write receipts to
   disk; scp receipts back as each stage finishes; never put a process
   pattern inside an ssh command that pkills it.
6. **Every receipt includes:** git SHA, image tag, torch / transformers / trl
   / peft / bitsandbytes versions, GPU, seeds, batch / seq / lr / optimizer /
   K per arm, NVML peak next to torch allocated / reserved, per-item outputs
   for paired tests, loss traces.
7. **Delete the pod** as soon as it is idle (`pod.sh delete <id>`), after
   confirming the receipts are on disk.
8. **Commit receipts** under `docs/receipts/<date>-<name>/` with an index
   README. Public docs (README, handbook, CHANGELOG) are written by the lead,
   never by an agent.

## 6. Repo gotchas met on 2026-09-30

- `main` requires PRs to be up to date (strict). Merge PRs **one at a time**;
  two in parallel knock each other out of date.
- The `atlas/` map is regenerated, not hand-merged: `npx --yes
  @dogfood-lab/atlas@1.24.0 map` (match the version in `ci.yml`), then
  `... check`.
- Bandit forbids `try/except/pass`; for a non-crypto RNG use
  `# nosec B311 — <reason>` inline.
- New CLI flags / env vars / error codes need rows in
  `cli-reference.md` / `env-vars.md` / `error-codes.md` or `drift-check` fails.
- Local-only test failures on the rig (fail identically on untouched main):
  `test_export.py::TestStageCHfTokenCap` (HF token in the environment) and two
  `TestBuildSftConfigHelper` tests when run in some orders.
- The rig's `.venv` is an editable install of the **main checkout**, so a CLI
  subprocess in a test run from a worktree runs main's code, not the worktree's.
- Files edited with Python text mode on Windows become CRLF; match a file's
  line endings when patching (read bytes, normalise, write back).
- Use `uv` ≥ 0.12 for `uv lock` (`uvx --from uv@latest uv lock`); 0.11
  rewrites unrelated marker lines.

## Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | Each experiment pins git SHA, image tag, library versions, seeds and per-arm settings in its receipt (section 5.6). |
| ANDON_AUTHORITY | 2 | A failed verify (no CUDA) or a failed smoke stops the run; decision rules are pre-registered and not tuned toward. |
| NAMED_COMPENSATORS | 2 | See the table below. |
| DECOMPOSE_BY_SECRETS | 2 | Scripts and code are built locally by agents; pods are created and deleted only by the lead; public docs only by the lead. |
| UNCERTAINTY_GATED_HUMANS | 2 | The Director decides when a gate result is mixed or a product default changes; spend above the stated budget needs his word. |
| EXTERNAL_VERIFIER | n/a | No specialized claims by default. When a result drives a product decision, an external review (as with Kimi K3 on 2026-09-30) is optional and at the Director's call. |

| Irreversible action | Undo | State after | Owner |
|---|---|---|---|
| Create a RunPod pod (billing starts) | `scripts/runpod/pod.sh delete <id>`; dead-man timer deletes at the deadline regardless | no pod, billing stopped | lead |
| Merge an experiment PR to `main` | `git revert -m 1 <merge sha>` via a PR | `main` as before | lead |
| Publish a release | `pypi` yank / `npm deprecate`, then ship the next patch | version hidden / warned | Director |
