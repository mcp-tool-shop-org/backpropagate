# backpropagate

## What This Does

Headless LLM fine-tuning library with smart defaults, Windows support, and one-click GGUF export to Ollama. Train a 7B model with 3 lines of Python; ship to Ollama with one more.

Status: **stable / production** (Development Status :: 5 — Production/Stable in pyproject; v1.5.0 shipped on PyPI + npm + GitHub; v1.6.0 in preparation; full Ship Gate passing).

## Architecture

- `trainer.py` — core `Trainer` class (load, train, save, export)
- `multi_run.py` + `slao.py` — multi-run SLAO LoRA-merge training (anti-catastrophic-forgetting)
- `datasets.py` — JSONL/ShareGPT/Alpaca/OpenAI format auto-detect + filtering + dedupe + curriculum
- `export.py` — LoRA/merged/GGUF export, Ollama Modelfile + registration
- `config.py` — Pydantic settings, presets (Qwen 2.5 7B/3B, Llama 3.2 3B/1B, Mistral 7B)
- `gpu_safety.py` — temp/VRAM/utilization monitoring, auto-pause
- `cli.py` + `__main__.py` — `backprop` / `backpropagate` entry points
- `ui_app/` + `rxconfig.py` — Reflex (Radix UI) web UI shipped in v1.1.0 (canonical; optional, requires `[ui]` extra). The v1.0 Gradio implementation (`ui_gradio_legacy.py` + `theme_gradio_legacy.py`) was preserved through v1.1.x as reference and removed in v1.2.0.
- `ui_security.py` — shared UI auth + path-sandbox helpers
- `security.py` — path traversal + safe torch loading
- `checkpoints.py` — checkpoint policies + cleanup
- `exceptions.py` — structured exception hierarchy (Ship Gate B1)
- `feature_flags.py` — lazy optional-dep detection

## Key Notes

- Headless-first; UI is opt-in via `pip install backpropagate[ui]`
- Modular extras: `[unsloth]`, `[ui]`, `[validation]`, `[export]`, `[monitoring]`, `[logging]`, `[security]`; bundles: `[standard]`, `[full]`, `[production]`
- First-class Windows support (pre-tokenization, xformers auto-disable on RTX 40/50, safe dataloader)
- Tested on RTX 5080 (16GB VRAM)
- ~3139 tests in tests/ (3139 passed + 8 skipped, 32 gpu/slow/integration deselected), 50% coverage floor (single source of truth: `[tool.coverage.report].fail_under = 50` in pyproject.toml; ci.yml reads it via tomllib so the two surfaces stay in lockstep)
- Python 3.10 → 3.13 supported in CI; 3.10 is supported through at least v1.6 and reaches upstream EOL Oct 2026, scheduled for removal in the first release after that. Prefer 3.11 / 3.12 for new installs (3.11 is the most-tested floor — UI + Windows + macOS smoke cells all run on 3.11)
- All Ship Gate hard gates (A–D) checked 2026-02-27, scorecard 23/31 (14 SKIP with reasons), `shipcheck audit` passes 100%

## User-facing docs surface

- Canonical handbook lives under `site/src/content/docs/handbook/` (Astro/Starlight). The Jekyll tree at `docs/` is legacy — `docs/index.md` now redirects to the handbook.
- Stage B contracts documented (added 2026-05-21 by docs swarm agent):
  - `handbook/error-codes.md` — full catalog of stable codes (INPUT_/CONFIG_/DEP_/RUNTIME_/STATE_/PARTIAL_)
  - `handbook/troubleshooting.md` — symptoms-first reverse index
  - `handbook/env-vars.md` — every `BACKPROPAGATE_*` knob
  - `handbook/cli-reference.md` — every subcommand + flag
  - README "Troubleshooting" + "Reporting bugs" + "Web UI" + "Platform prerequisites" subsections cover the load-bearing user-facing contracts (run_id correlation, --share+--auth gating, redacted stderr, sandbox env var).
- `examples/quickstart.jsonl` is a 5-line ShareGPT-format starter dataset referenced by the README Quick Start.
