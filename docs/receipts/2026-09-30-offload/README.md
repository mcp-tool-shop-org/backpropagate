# Receipts: full_ft_offload + QLoRA presets (2026-09-30)

This is evidence, not documentation. The constants in `backpropagate/offload_engine.py`, and the checks in `tests/test_offload_fit.py`, cite these files by name.

**Where these ran:**
- Hardware: a RunPod RTX 5090 (31.36 GiB) with a 1 TB host. A 60 GiB RSS ceiling was enforced by an in-process watchdog.
- Software: torch 2.8.0+cu128, transformers 5.17.0, trl 0.27.2, accelerate 1.15.0, peft 0.21.1.

**Run settings, unless the row says otherwise:**
- 20 steps, seq 512, batch 1.
- `BACKPROPAGATE_OFFLOAD_PIN=register`.

**Fields:**
- `peak_rss_train_gb`: peak through training.
- `peak_rss_total_gb`: the same peak, plus save → reload → generate.
- VRAM figures are the torch allocator peaks.

The SHA column is the branch commit that the run's checkout was on (taken from the pod's `ladder.log` and `pod.log`).

| File | Run | SHA |
|---|---|---|
| `q7b_final_receipt.json` | Qwen2.5-7B-Instruct, `scripts/pod_offload_7b.sh`, **PASS** (the shipped receipt) | fc79bb9 |
| `q7b_first_receipt_d291aa2.json` | First 7B receipt. It **failed** its original bar ("≥ 50 % params changed", set before any data) and is superseded | d291aa2 |
| `q7b_seq2048.json` | 7.6B, seq 2048, 4 steps, no save (throughput + VRAM slope) | d291aa2 |
| `q7b_cap12.json`, `q7b_cap8.json` | 7.6B under a 12 / 8 GiB VRAM cap (`set_per_process_memory_fraction`), 3 steps | d291aa2 |
| `q15b_offload_reg.json` | Qwen2.5-1.5B-Instruct, register mode (host-RAM slope anchor) | d291aa2 |
| `smollm3_offload_reg.json` | SmolLM3-3B, register mode (host-RAM slope anchor) | d291aa2 |
| `qwen3_4b_offload_reg.json` | Qwen3-4B-Instruct-2507, register mode. Its peak with save sets the host-RAM fixed term | d291aa2 |
| `smollm3_cap8.json`, `smollm3_cap6.json`, `qwen3_4b_cap8.json` | 3B / 4B under 8 / 6 GiB VRAM caps, 5 steps | d291aa2 |
| `smollm3_pinned.json` | SmolLM3-3B, `BACKPROPAGATE_OFFLOAD_PIN=pinned` (speed/RAM comparison) | d291aa2 |
| `q15b_puregpu.json`, `smollm3_puregpu.json` | Pure-GPU full FT (SFTTrainer path), 10 steps, for comparison | d291aa2 |
| `q15b_offload.json`, `smollm3_offload.json` | Pinned mode, before the register fix (1.8x host RAM) | 58ac40c |
| `q15b_nearest.json` | 1.5B with `BACKPROPAGATE_OFFLOAD_ROUNDING=nearest`: update retention 0.171 | fc79bb9 |
| `q15b_sr_retention.json` | 1.5B, stochastic rounding: update retention 1.0001 | fc79bb9 |
| `quality.jsonl` | SmolLM3-3B held-out loss on dolly-15k (400 train / 150 held-out, seed 0, 150 steps, batch 4): base, offload engine, pure-GPU | fc79bb9 |
| `preset_qwen2.5-14b.log` | `tests/test_qlora_presets_smoke.py[qwen2.5-14b]`: PASS, 24.96 / 28.10 GiB | fc79bb9 |
| `preset_qwen2.5-32b.log` | `...[qwen2.5-32b]`: PASS, 28.79 / 30.71 GiB | fc79bb9 |
| `preset_mistral-small-24b.log` | `...[mistral-small-24b]`: PASS, 26.49 / 29.58 GiB | fc79bb9 |

`llama-3.1-8b` has no receipt. The repo is gated and the pod had no HF token.
