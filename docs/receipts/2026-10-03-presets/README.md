# Receipts: the 32 GB-envelope QLoRA presets (2026-10-03)

Evidence for the three QLoRA rows in the README's "What you can fine-tune on one GPU" table, the preset descriptions in `backpropagate/config.py`, and the measured column of the VRAM handbook page. It replaces the preset rows of `docs/receipts/2026-09-30-offload/`.

**Where this ran:**
- Hardware: a RunPod RTX 5090 (31.36 GiB), 186 GiB host RAM. Secure cloud, image `runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404`.
- Software: torch 2.8.0+cu128, transformers 5.18.0, trl 1.14.1, peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2, datasets 5.0.1 (`env.json`, `constraints.txt`, and the install lines in each log).
- Code: `main` at `d06f51de4e91104555c54cc8dcc08b725503ca1d`.
- Method: `tests/test_qlora_presets_smoke.py`, one preset per pytest run, serially, with `BACKPROPAGATE_RUN_PRESET_SMOKE=1`. The 14B case trains 2 QLoRA steps and saves the adapter; the 24B and 32B cases run a 1-step no-OOM probe at the preset's shipped `recommended_max_seq_length`, with `oom_recovery=False` so an OOM fails instead of retrying smaller. Each case passes `recommended_lora_r` (32, alpha 32) and the preset's packing setting, as an operator following the preset would. VRAM figures are the torch allocator's peak allocated / peak reserved, read by the test.

| Preset | Case | Context | Peak allocated | Peak reserved | Log |
|---|---|---|---|---|---|
| `qwen2.5-14b` | train, 2 steps, packing on | 4096 | 18.73 GiB | 19.97 GiB | `preset_qwen2.5-14b.log` |
| `mistral-small-24b` | probe, 1 step | 4096 | 22.82 GiB | 24.15 GiB | `preset_mistral-small-24b.log` |
| `qwen2.5-32b` | probe, 1 step | 2048 | 25.98 GiB | 27.23 GiB | `preset_qwen2.5-32b.log` |

The `PRESET_SMOKE_RECEIPT` line in each log is the record; `summary.txt` collects them.

**Against 2026-09-30** (25.0 / 28.1, 26.5 / 29.6 and 28.8 / 30.7 GiB on the same card, same rank, same contexts): every row is 3 to 6 GiB lower. Two things changed between the runs: the memory-efficient attention path without flash-attention or xFormers (#292), which is aimed at exactly this, and the library stack (trl 1.14 against 0.27, transformers 5.18 against 5.17, bitsandbytes 0.50.2). The two were not separated in this session.

Not re-measured: `llama-3.1-8b` (gated repository; the pod had no Hugging Face token), and the full fine-tuning rows, which stay with `docs/receipts/2026-09-30-offload/`.
