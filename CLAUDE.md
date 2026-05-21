# backpropagate

## What This Does

Headless LLM fine-tuning library with smart defaults, Windows support, and one-click GGUF export to Ollama. Train a 7B model with 3 lines of Python; ship to Ollama with one more.

Status: **stable / production** (Development Status :: 5 — Production/Stable in pyproject; v1.0.5 on PyPI; full Ship Gate passing).

## Architecture

- `trainer.py` — core `Trainer` class (load, train, save, export)
- `multi_run.py` + `slao.py` — multi-run SLAO LoRA-merge training (anti-catastrophic-forgetting)
- `datasets.py` — JSONL/ShareGPT/Alpaca/OpenAI format auto-detect + filtering + dedupe + curriculum
- `export.py` — LoRA/merged/GGUF export, Ollama Modelfile + registration
- `config.py` — Pydantic settings, presets (Qwen 2.5 7B/3B, Llama 3.2 3B/1B, Mistral 7B)
- `gpu_safety.py` — temp/VRAM/utilization monitoring, auto-pause
- `cli.py` + `__main__.py` — `backprop` / `backpropagate` entry points
- `ui.py` + `theme.py` + `ui_security.py` — Gradio UI (optional, requires `[ui]` extra)
- `security.py` — path traversal + safe torch loading
- `checkpoints.py` — checkpoint policies + cleanup
- `exceptions.py` — structured exception hierarchy (Ship Gate B1)
- `feature_flags.py` — lazy optional-dep detection

## Key Notes

- Headless-first; UI is opt-in via `pip install backpropagate[ui]`
- Modular extras: `[unsloth]`, `[ui]`, `[validation]`, `[export]`, `[monitoring]`, `[observability]`, `[logging]`, `[security]`; bundles: `[standard]`, `[full]`, `[production]`
- First-class Windows support (pre-tokenization, xformers auto-disable on RTX 40/50, safe dataloader)
- Tested on RTX 5080 (16GB VRAM)
- 1805 tests in `tests/`, 50% coverage floor (`fail_under = 50` in coverage config)
- All Ship Gate hard gates (A–D) checked 2026-02-27, scorecard 23/31 (14 SKIP with reasons), `shipcheck audit` passes 100%
