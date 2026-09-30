# Low-memory full fine-tuning engines — design (2026-09-30)

> **Status:** design for the Director's decision. Nothing here is built.
> **Asked for:** "ship FSDP2 now, build block-swap next" and "start the
> block-swap design in parallel" (Director, 2026-09-30).
> **What changed during the research:** the evidence points to a simpler base
> engine than block swap. Block swap becomes the extension for small cards.

## Decision-ready summary

| | |
|---|---|
| **Recommendation** | Build a **block-coordinate AdamW engine** first (one transformer block trains at a time, the rest stay frozen on the GPU). Add **host-to-GPU block swap** on top for cards where the 16-bit model does not fit in VRAM. Keep the FSDP2 engine as the "every weight, every step" option. |
| **Why** | It is the best-evidenced design for quality (findings 1–3), it keeps real AdamW, it needs no host offload on a 24–32 GB card for a 7B–8B model, and it runs on native Windows. It removes the three caveats of the FSDP2 engine: Linux only, Adafactor, 14.6 s/step. |
| **Actionable** | Findings 1, 2, 3, 6, 7, 9, 10, 11, 12 each change a design choice below. |
| **Filler** | Zeroth-order methods (MeZO) and LISA/HiFT do not change the design; they are listed once under "considered". |
| **Not proven by anyone** | Block-coordinate descent combined with block swap. No paper or trainer does it. It needs our own A/B runs (see the test plan). |
| **Cost to find out** | One RunPod session, about 3 hours on a 5090 (about $3), for the phase-1 quality and memory evidence. |

## The three engines

| Engine | What trains each step | Host RAM | VRAM at 7.6B | Platform | Optimizer | State |
|---|---|---|---|---|---|---|
| **A. FSDP2 CPU offload** | every weight | ~4 B/param + 3 GiB (30.9 GiB measured) | 5.3 GiB allocated | Linux / WSL2 | Adafactor, bf16 + stochastic rounding | proven on a pod 2026-09-30; ships in 1.7.2 |
| **B. Block-coordinate AdamW** | one block (of ~D) for K steps, then the next | none beyond normal | ~`2M + 16M/D` GB = ~20 GB (projected from finding 1, not measured by us) | Windows and Linux | AdamW | this design, phase 1 |
| **C. B + host-to-GPU block swap** | same as B | ~2 B/param | set by how many blocks stay resident | Windows and Linux | AdamW | this design, phase 2 |

`M` = parameters in billions, `D` = number of blocks.

## Research grounding

Four research agents ran in parallel on 2026-09-30 (Sonnet 5.5, web retrieval
only; each could cite only what it fetched in that session). No separate
verifier pass was run (Standing Rule 3); the numbers this design depends on
(finding 1) were re-fetched by the lead from the arXiv page and the BAdam
repository. The phase-1 pod run is the real check.

1. **Training one block at a time with Adam matches full Adam on the benchmarks
   tested, in LoRA-class memory.** Luo, Yu, Li 2024, *BAdam* (NeurIPS;
   arXiv:2404.02827, https://arxiv.org/abs/2404.02827). Memory is
   `2M + 16M/D` GB with the model held in 16-bit: 23.5 GB for Llama 3-8B on
   one RTX 3090, against 144 GB+ for Adam. Math average 44.4 vs Adam 44.1,
   LoRA 43.3, LOMO 31.6. Block order and K (10–200 steps per block) barely
   matter. Evidence is supervised fine-tuning only, from one group, and the
   Adam baseline overfit at lr 1e-5. *Implication: engine B is the base.*
2. **SGD-style fused updates do not reach AdamW; the second moment is what
   closes the gap.** Lv et al. 2023, *AdaLomo* (arXiv:2310.10195): on LLaMA-7B,
   AdaLomo 30.8 vs AdamW 29.1 vs LOMO 24.0. Zhang et al. 2024 (arXiv:2402.16788)
   give the mechanism: transformer Hessian blocks are heterogeneous and one
   learning rate cannot fit them. *Implication: no engine ships plain fused
   SGD; an adaptive per-parameter rate is required.*
3. **LOMO's fused update forbids global gradient-norm clipping unless backward
   runs twice.** Lv et al. 2023, *LOMO* (arXiv:2306.09782): 14.6 GB for
   LLaMA-7B; per-tensor clipping is the cheap substitute. *Implication: engine
   B does not fuse. It runs an ordinary backward over the active block, so
   global clipping and gradient accumulation keep working.*
4. **Full fine-tuning learns updates LoRA cannot.** Biderman et al. 2024
   (arXiv:2405.09673, abstract only): LoRA substantially underperforms full
   fine-tuning on code and math. *Implication: the feature is worth building;
   it is the reason to offer full FT at all.*
5. **Round-to-nearest cancels small updates in 16-bit weights; stochastic
   rounding or Kahan summation fixes it.** Zamirai et al. 2021
   (arXiv:2010.06192): plain 16-bit loses 7% on BERT-Base MNLI; stochastic
   rounding matches fp32 on 5 of 7 tasks, Kahan on 7 of 7 at 2x weight memory.
   Our own pod run agrees: 3.0% of parameters changed and loss 0.96 with
   round-to-nearest, 0.07 with stochastic rounding (360M, one seed).
6. **bf16 AdamW with stochastic rounding works at LLM scale.** Ozkara et al.
   2025 (arXiv:2502.20566): beats bf16/fp32 mixed precision on GPT-2 and
   GPT-Neo up to 6.7B. It works best at 2–4x the default learning rate and
   struggles as step sizes decay toward zero. *Implication: engine B keeps the
   active block's master weights in fp32 while it trains and writes back to
   bf16 once per block visit, with stochastic rounding. K accumulated steps
   make the update large relative to a bf16 ulp, which sidesteps the
   per-step rounding problem.*
7. **Kahan-compensated bf16 did not match fp32 in one LLM study.** Yu et al.
   2024, *Collage* (ICML; arXiv:2405.03637). This conflicts with our single
   run where Kahan matched fp32 exactly. *Implication: no precision claim
   ships on one seed. The test plan requires 3 seeds and 2 sizes.*
8. **Adafactor's evidence for LLM fine-tuning is thin.** Shazeer & Stern 2018
   (arXiv:1804.04235) validated it on machine translation. AdaLomo's appendix
   reports Adafactor 30.0 vs AdamW 29.1 on LLaMA-7B, one data point.
   *Implication: engine A's Adafactor default is documented as a trade-off,
   and engines B and C use AdamW, where the evidence is.*
9. **8-bit Adam matches 32-bit at a quarter of the state.** Dettmers et al.
   2021 (arXiv:2110.02861): GLUE 88.7 vs 88.6. *Implication: a fallback for
   engine B on small cards. The default stays fp32 AdamW on the active block,
   because one block's state is small.*
10. **The working block-swap mechanism is an in-place exchange driven by
    hooks.** kohya-ss/sd-scripts `library/custom_offloading_utils.py`
    (https://github.com/kohya-ss/sd-scripts/pull/1779): weights are copied into
    existing CUDA storage (no free, no realloc), on one worker thread with a
    private stream, prefetching ahead of compute in forward order and in
    reverse order for backward. *Implication: engine C copies this shape.*
11. **A host-to-GPU-only swap exists but only for frozen weights.**
    kohya-ss/musubi-tuner `docs/block_swap.md`: `--block_swap_h2d_only` keeps a
    permanent CPU master and never copies back; it "cannot be used for full
    fine-tuning". Measured on Qwen-Image LoRA: about 11 s/sample against 14
    in exchange mode. *Implication: under block-coordinate descent every
    block except the active one IS frozen, so engine C can use the H2D-only
    path for all of them. Only the active block is written back, once per
    visit. This composition is ours; nobody has published it.*
12. **Windows limits page-locked memory and can silently spill VRAM into system
    RAM.** NVIDIA staff on the developer forum put the pinned limit at about
    half of RAM
    (https://forums.developer.nvidia.com/t/change-limit-of-50-for-cudahostalloc-pinned-memory-on-windows-10-11/228235).
    PyTorch's pinned allocator rounds every allocation up to a power of two
    (`aten/src/ATen/core/CachingHostAllocator.h`; pytorch/pytorch#95823).
    The driver's sysmem fallback slows training about 3x with no error, and no
    API detects it. *Implication: engine C keeps host masters pageable and
    uses a small pool of registered staging buffers (the musubi
    `_StagedCopier` pattern, and OneTrainer's `cudaHostRegister` arenas). It
    caps VRAM with `set_per_process_memory_fraction` and documents the
    "Prefer No Sysmem Fallback" driver setting.*
13. **Pinned transfers are about 4x faster than pageable on one measured
    system.** NVIDIA forum thread 355919: about 24 GB/s pinned against 6 GB/s
    pageable on an RTX 4090D, PCIe 4.0. *Implication: engine A's 14.6 s/step
    at 7B is far above a transfer floor of roughly 1.3–1.9 s (inferred, not
    measured on a 5090). There is an optimisation pass worth doing on A.*

**Considered and set aside:** MeZO (Malladi et al. 2023, arXiv:2305.17333) —
forward-only, 20–100x more steps and large accuracy deficits. LISA
(arXiv:2403.17919) and HiFT (arXiv:2401.15207) — close cousins of BAdam; LISA
needs the whole model resident and trails full FT by 6 GSM8K points at 70B.
ChunkFT (arXiv:2605.21177) — abstract only, unverified.

## Engine B — block-coordinate AdamW

1. Load the model on the GPU in bf16. Partition the trainable parameters into
   blocks: each transformer layer is one block; embeddings and the head are
   blocks of their own. *(Finding 1.)*
2. Pick the active block. Upcast its weights to fp32 on the GPU; every other
   block has `requires_grad=False`.
3. Train K steps (default 50, the low end of BAdam's 50–100) with AdamW on the
   active block. Backward runs through the active block and everything above
   it and stops there, because nothing below it needs a gradient.
   *(Findings 1, 3.)*
4. Write the block back to bf16 with stochastic rounding, drop its optimizer
   state, move to the next block. *(Findings 5, 6.)*
5. Because it is an optimizer wrapper and not a custom loop, it runs inside
   the existing TRL `SFTTrainer` path: packing, response-only masking,
   checkpoints, resume, gradient accumulation and global clipping all keep
   working. Engine A lost all of those.

Scope: `method="sft"` only. ORPO, SimPO and KTO are gated off until someone
tests them *(finding 1: evidence is SFT only)*.

Licence note: BAdam's reference code is Apache-2.0 (checked on the repository
page). kohya-ss sd-scripts and musubi-tuner are believed Apache-2.0 and
OneTrainer AGPL-3.0; **those three are from memory and must be checked before
any code is copied.** The plan is to implement from the papers and
descriptions, not to vendor code.

## Engine C — engine B plus host-to-GPU block swap

For cards where `2M` GB does not fit (7B on 12 GB, 3–4B on 8 GB):

1. Keep host masters for all blocks in pageable RAM in bf16 (2 B/param).
2. Keep `N − s` blocks resident on the GPU in fixed slots. The active block is
   always resident.
3. Frozen blocks are copied host-to-GPU only, ahead of compute, into existing
   CUDA storage; they are never copied back. *(Findings 10, 11.)*
4. Only the active block is written back to host, once per visit.
5. Blocks below the active block are needed for forward only; blocks above it
   for forward and backward. Prefetch order follows the kohya hooks.
6. Staging: a small ring of registered buffers, not per-tensor `pin_memory()`.
   *(Finding 12.)*

Gradient checkpointing stays on for C. OneTrainer hard-requires it for
offload and kohya's examples use it.

## What the documentation must not claim

- That block-coordinate training is the same as updating every weight every
  step. It is not; say "one block at a time".
- That it beats full AdamW. The published wins are against weakly tuned
  baselines *(finding 1)*.
- That it works for preference tuning or continued pretraining.
- Any precision or quality number measured on one seed *(finding 7)*.
- Windows speed parity with Linux. One kohya issue reports Flux fine-tuning
  25% slower on Windows with no swap, cause unknown (sd-scripts#2218).

## Test plan (all GPU work on RunPod, never on the rig)

**Measurement validity first.** Every quality comparison uses held-out loss on
real instruct data with a disjoint split of at least 100 examples; the
untrained baseline must sit strictly between floor and ceiling.

| Run | Proves | Size |
|---|---|---|
| B vs the library's standard full AdamW, 3 seeds | the quality cost of block-coordinate training | 1.5B and 3B (both fit pure-GPU) |
| B vs engine A vs QLoRA, 3 seeds | which to recommend at 7B | 7.6B |
| B under 24 GB and 16 GB VRAM caps | the memory formula on our code | 7.6B, 3B |
| Write-back rounding: nearest vs stochastic, K = 10 / 50 / 200 | whether finding 6's argument holds | 360M, 3B |
| Stochastic-rounding unit tests: unbiased mean, no second `copy_` overwriting the result (OneTrainer#994) | the rounding code is right | CPU |
| C vs B: identical loss trajectory on a model that fits both ways | swap changes memory, not maths | 1.5B |
| C under 12 GB and 8 GB caps; s/step vs B | the small-card claim | 7.6B, 3B |
| Native Windows smoke, tiny model | it runs where the library's users are | 135M on the rig |

Build order: B (phase 1) → its evidence → C (phase 2) → optional transfer
optimisation of A *(finding 13)*.

## Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | Each pod run records git SHA, resolved package versions, seed and model revision in its receipt. |
| ANDON_AUTHORITY | 2 | A failed fit check or a quality gap beyond the seed spread stops the phase; nothing is documented as supported without its receipt. |
| NAMED_COMPENSATORS | 2 | Paid pod: `DELETE /v1/pods/{id}` by the lead, plus a dead-man timer (owner: lead). Code: `git revert` of the engine PR (owner: lead). No publish happens in this design. |
| DECOMPOSE_BY_SECRETS | 2 | Engine B (optimizer wrapper) and engine C (transfer layer) are separate modules and separate PRs; C must leave B's loss trajectory unchanged. |
| UNCERTAINTY_GATED_HUMANS | 2 | The Director decides the build order here; he is asked again only if B's quality gap is larger than the published one. |
| EXTERNAL_VERIFIER | n/a | No separate verifier pass (Standing Rule 3). The load-bearing numbers were re-fetched from the primary source; the pod runs are the check. |
