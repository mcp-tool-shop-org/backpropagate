# Engine B evidence run 2: GSM8K receipts (2026-09-30)

These receipts come from the stage d run for PR #237, the block-coordinate engine against standard full fine-tuning, QLoRA and GaLore.

## Setup

- **Hardware and software.** One RTX 5090 (32 GiB); the 5090-emulation cap stayed off. torch 2.8.0+cu128, transformers 5.18.0, trl 1.14.1, peft 0.21.1, bitsandbytes 0.50.2, galore-torch 1.0. The driver ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Code.** Driver commit `da33049` on `feat/block-coordinate-engine`. The script is `scripts/pod_block_engine.sh` (stage `d`), with `scripts/pod_block_engine.py` and `scripts/pod_gsm8k_summary.py`.
- **Data.** openai/gsm8k (config `main`, MIT). Training used all 7,473 train rows, shuffled with seed 0. Evaluation used a fixed sample of 250 test questions (`prep_gsm8k.json` lists their indices). Every arm trained 1000 steps at batch 4 (0.535 epoch), unpacked, at sequence length 512 (0% of gold `####` lines truncated; see `runs/lengths_*.json`).
- **Loss and data order.** Loss is computed over the full sequence, identically in every arm. `mask_report` in `stage_d.json` confirms the same first batch, and the same loss mask, per seed across every arm.
- **Metrics.** The primary metric is held-out loss on the 250 answers, counting answer tokens only. The secondary metric is strict accuracy: the number after `####` in greedy decoding (400 new tokens max); a parse failure counts as wrong.
- **Paired statistics.** Stage d reports McNemar exact tests and paired bootstrap confidence intervals (10k resamples) computed from the per-item files in `runs/items/*.jsonl`.
- **3B model choice.** Qwen2.5-3B-Instruct was picked over SmolLM3-3B on base strict accuracy: 0.596 vs 0.124 (`d_pick.json`).

## Files

| Path | Contents |
|---|---|
| `stage_d.json` | Per-arm results (accuracy with Wilson CIs, held-out answer loss, s/step, NVML peak, torch peaks, paged optimizer state) and the paired statistics per pair. Recomputed with the summary at `da33049`, which reports whole-run torch peaks |
| `runs/d_*.json` | One receipt per run: manifest, config, per-step train loss, per-block visit log, NVML phase peaks, and the optimizer class with its arguments |
| `runs/items/*.jsonl` | Per-item records: `{qid, gold, generated, parsed, correct, answer_loss_sum, answer_tokens}` |
| `runs/base_gsm8k_*.json` | Untrained baselines |
| `stage_x1000.json`, `runs/x1000_*.json` | The first stage-d attempt, kept as a separate result. The library's `max_samples=1000` default silently capped the train set, so 1000 steps meant **4 epochs over 1000 rows**. Do not mix these with `d_*` |
| `d2_triggers.json` | Pairs whose pooled loss CI spans zero (none triggered) |

## Not run

- 3B QLoRA seed 2 and Adafactor seed 1: the budget guard skipped them, following the planned drop order.
- Stage e: stopped before it began, because the run was past the budget.
