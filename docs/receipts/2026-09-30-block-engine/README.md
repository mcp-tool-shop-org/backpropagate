# Engine B evidence run 1: receipts (2026-09-30)

These are the raw JSON receipts from the first RunPod evidence run for the block-coordinate AdamW engine (PR #237; design #232). The hardware was one RTX 5090 (32 GiB) with torch 2.8.0+cu128. pip resolved transformers 5.18.0, trl 1.14.1, peft 0.21.1, accelerate 1.15.0 and bitsandbytes 0.50.2; see `env.json` and the `versions` field of each run. The script that produced them is `scripts/pod_block_engine.sh` together with its driver `scripts/pod_block_engine.py`. The pod logs are not included. Only the JSON receipts are.

## Files

| File | What it is |
|---|---|
| `env.json`, `prep.json`, `receipt.json` | Pod environment, the dolly-15k split (400 train / 150 disjoint held-out), and the final per-stage verdicts |
| `stage_a.json` | Quality A/B: Engine B with K=50 vs the standard full fine-tune (paged 8-bit AdamW), Qwen2.5-1.5B-Instruct and SmolLM3-3B, 3 seeds, 150 steps |
| `stage_a2.json` | Same as stage a with Engine B at K=5 |
| `stage_b.json` | Qwen2.5-7B-Instruct: Engine B uncapped with save → reload → generate, fit under 24 GiB and 16 GiB caps, and QLoRA |
| `stage_c.json` | Write-back rounding: nearest vs stochastic at K ∈ {10, 50, 200}, SmolLM2-360M and SmolLM3-3B, 1 seed |
| `runs/*.json` | One receipt per run: seed, versions, git SHA, per-step losses, s/step, peak VRAM (whole run and per block visit), fit projection, and the held-out loss before and after |
| `runs/p_*.json` | Preset check: `Qwen/Qwen3.5-4B-Instruct` does not exist on the Hub, while `Qwen/Qwen3.5-4B` loads and trains through the text-only loader |
| `runs/b_block_k5_random*.json` | An extra 7B run with K=5 and random order. The `_ad76bcf` copy is the rerun after the write-back memory fix. It is bitwise-identical in loss and 3.2 GiB lower at the embedding switch |

## Git SHAs in the receipts

The branch was later rebased onto `main`. The SHAs recorded in the receipts map to the rebased commits as follows. The engine code is identical across all of them, except that `28a7fc8` adds the write-back memory fix.

| In receipts | Rebased |
|---|---|
| `fbdc343` | `cf9e8df` (engine) |
| `498ec1a` | `d7bd84f` |
| `209458d` | `e6c3466` (settings fix) |
| `ddc666a` | `5d7f703` (no trainer checkpoints) |
| `ad76bcf` | `28a7fc8` (write-back fix) |

## Caveats recorded during the run

- **The first stage-b attempt failed at the final checkpoint save.** Its 7B runs, `b_block_uncapped` and `b_block_cap24_frozen`, filled the 40 GB container disk. Both were rerun after the driver stopped writing trainer checkpoints. The receipts here come from the reruns, and `stage_b.json` reflects them (PASS).
- **The `BACKPROPAGATE_*` env vars were ignored.** pydantic-settings was not installed, so the settings layer never read them. The driver was changed to set the seed on the settings object before stage a, and every run records `settings_seed`.
- **Held-out loss after 150 dolly steps separates the arms only weakly.** In stage c, training one 3B layer alone (K=200 gives a single visit) reached 2.024, better than four blocks at K=50.

## Default K: kept at 50

With K=5, Engine B beats K=50 at both sizes, by more than the seed spread:

| Model | K=5 | K=50 |
|---|---|---|
| 1.5B | 1.912 ± 0.004 | 1.994 ± 0.002 |
| 3B | 2.015 ± 0.011 | 2.047 ± 0.043 |

The default stays at 50 for now. At 150 steps, K=50 visits only 3–4 of the ~30 blocks, so this comparison mostly measures coverage, not the block length. BAdam recommends K ≥ 50 for enough decrease per sub-problem, and for mixed-precision rounding. Stage c showed no consistent K effect. Stage d of the next run (GSM8K, 1000 steps, where K=50 covers ~20 blocks) compares K=5 and K=50 directly. The default follows that result.
