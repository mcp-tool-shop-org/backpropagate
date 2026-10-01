# Consult response — Kimi K3 (run by hand by the Director, 2026-09-30)

Source: Kimi K3, given `consult-brief-full-ft-engines-2026-09-30.md`
verbatim. Recorded here as the provenance for the decisions it changed
(Standing Rule 1: record where every verdict came from). The full response
text is held by the Director; the lead's assessment of each point is below.

## Accepted and acted on

- Engine B: keep through a GSM8K gate against QLoRA r=256 at 7B, with
  paired statistics (McNemar on shared items, paired bootstrap, Wilson CIs),
  held-out answer loss as the primary metric, 3 seeds at 3B, a second seed
  for engine B at 7B, train-loss traces, a length check before fixing the
  sequence length, a learning-rate probe for engine B (5e-5), per-block
  visit logging. Ship / drop / experimental criteria adopted.
- Add GaLore (Zhao et al. 2024, arXiv:2403.03507) as the next candidate in
  this niche. transformers exposes it as an optimizer option, so it is a
  cheap arm.
- Standard full fine-tuning VRAM peaks (1.5B 6.7 GiB, 3B 12.6 GiB) are
  undercounts: they used PyTorch allocator counters, and the paged 8-bit
  AdamW state is allocated by bitsandbytes as CUDA managed memory
  (`bitsandbytes.functional.get_paged` -> `cget_managed_ptr`), which those
  counters do not see. Verified in the installed bitsandbytes source. All
  VRAM figures are to be re-measured with system-wide NVML sampling.
- The offload engine's speed: the optimizer re-streams parameters and
  gradients after backward. The lead's own reading of the code adds that
  for tensors larger than one chunk (2^26 elements: every 7B MLP weight and
  the embeddings) the gradient is streamed three times. Fusing the update
  into backward, one pinned arena, prefetch on a copy stream, and per-leg
  CUDA-event timing are planned for 1.8.0.
- The host-RAM fit is conservative at 7.6B (38.5 GiB predicted, 32.2
  measured); publish per-point values and do not extrapolate.
- The offload speed table mixed batch sizes; configs must be stated per row.
- Update retention under stochastic rounding is ~1 by construction: an
  acceptance test of the write-back path, not an ongoing health metric.
  Add stagnation-pressure fraction and an fp32 shadow tensor for long runs.
- Gradient accumulation above 1 in the offload engine accumulates in bf16;
  keep it documented as untested until tested.

## Checked and not accepted

- "Stochastic rounding by adding random low bits and truncating is biased
  for negative values": not for this implementation. IEEE-754 is
  sign-magnitude, so the bit-level add-and-mask rounds the magnitude and is
  symmetric in sign. Measured over 10^6 draws at +/-1.0010 and +/-0.0123:
  bias within Monte Carlo noise (|bias| <= 3e-6 against an ulp of 7.8e-3).
  The unit test is still worth keeping.
- "Every bitsandbytes arm is flattered": only paged optimizers use managed
  memory. QLoRA runs on 24 GB+ cards use non-paged `adamw_8bit`, whose state
  is ordinary PyTorch tensors, and the 4-bit base weights are PyTorch
  tensors, so the QLoRA peaks are likely correct. NVML re-measurement will
  confirm.
- "Engine B 3B memory K=5 vs K=50 is unexplained": at K=50 150 steps visit
  3 blocks and likely never the embedding/head block; K=5 visits ~30,
  including it. The embedding/head block in fp32 with Adam state sets the
  peak. To be confirmed by the per-block visit log.
- An engine A + 8-bit AdamW control arm: bitsandbytes optimizers cannot step
  CPU-offloaded DTensors. The same question (optimizer recipe vs offload
  numerics) is answered more cheaply by running standard full fine-tuning
  with Adafactor at 3B.

## Round 2 (same day)

Kimi conceded the stochastic-rounding point (the hazard belongs to the
arithmetic-space variant, not the bit-level add-and-mask) and narrowed the
bitsandbytes point to paged optimizer state, matching the lead's check.

Accepted and acted on:
- Re-derive the pure-GPU full fine-tuning ceiling constants in `trainer.py`
  (`_FULL_FT_VRAM_CEILING_TIERS`, whose comment says "measured") from the
  NVML numbers, not only the docs.
- NVML sampling on the GSM8K run itself, so it is the re-measurement source.
- Per-item outputs ({qid, gold, generated, parsed, correct}) for the paired
  tests; run manifest with library versions.
- The Adafactor control must match the offload engine's recipe exactly.
- Exposure: 1000 steps x batch 4 is about 0.53 epoch; a conditional 2000-step
  rerun of any tied pair is pre-authorized in the script.
- Two loose ends for the per-block log: engine B's identical peak at 1.5B
  across K, and the reserved-vs-allocated gap.

Not accepted as proposed:
- GaLore non-layerwise (`galore_adamw_8bit`) as the first choice: at 7.6B it
  keeps full bf16 gradients on the GPU (~15 GB on top of ~15 GB of weights),
  so it very likely does not fit 32 GB. The layerwise mode is what the
  paper's 7B-on-24 GB result depends on; it runs first, non-layerwise once as
  a fallback.
