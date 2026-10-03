# backpropagate

## What This Does

Headless LLM fine-tuning library with smart defaults, Windows support, and one-click GGUF export to Ollama. Train a 7B model with 3 lines of Python; ship to Ollama with one more.

Status: **stable / production** (Development Status :: 5 — Production/Stable in pyproject). **v1.8.1** shipped 2026-10-01 on PyPI + npm + GitHub (`backprop ui` starts again in one-port production mode with token / scrypt-verifier auth; `pass_rate` code execution is opt-in and out-of-process; `docker compose up` runs the UI; Reflex telemetry off). Store (MSIX) packaging is planned for 1.8.2: `docs/handoff-2026-10-01-msix-store.md`. Experiment plan: `docs/handoff-2026-09-30-full-ft-experiments.md` §4a.

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
- Modular extras: `[unsloth]`, `[ui]`, `[validation]`, `[export]`, `[monitoring]`, `[logging]`, `[security]`, `[fp8]`, `[mlx]` (Apple-only, unverified preview, kept out of `[full]`); bundles: `[standard]` (= unsloth + ui, the README install), `[full]`, `[full-no-export]`, `[production]`
- First-class Windows support (pre-tokenization, xformers auto-disable on RTX 40/50, safe dataloader)
- Dev rig: **RTX 5090 (32 GB) + 64 GB RAM**, Windows 11. The repo is positioned 32 GB-first since v1.7 (scales down to 16 GB). FSDP2 offload (`--full-ft-offload`) needs NCCL → run it under WSL2, not Windows-native.
- Real-GPU smokes (`tests/test_*_smoke.py`, integration-marked) run by hand on the rig, never in CI; CI's weekly train smoke is CPU-only. Mocked-green unit tests have repeatedly hidden real training-path bugs — every new training path needs one non-mocked smoke.
- 8276 tests in tests/ (pinned 2026-10-02; `pytest --collect-only`), 90% coverage floor (single source of truth: `[tool.coverage.report].fail_under = 90` in pyproject.toml; ci.yml reads it via tomllib so the two surfaces stay in lockstep)
- Python 3.10 → 3.13 supported in CI; 3.10 is supported through at least v1.6 and reaches upstream EOL Oct 2026, scheduled for removal in the first release after that. Prefer 3.11 / 3.12 for new installs (3.11 is the most-tested floor — the UI and Windows cells run on 3.11; macOS cells were dropped in 1.7.1). Plan: 1.8.0 keeps 3.10; the first release after its Oct 2026 EOL drops it
- Ship Gate hard gates (A–D) last checked 2026-02-27 (scorecard 23/31, 14 SKIP with reasons) — stale; re-run `shipcheck audit` after 1.8.0 ships

## User-facing docs surface

- Canonical handbook lives under `site/src/content/docs/handbook/` (Astro/Starlight). The Jekyll tree at `docs/` is legacy — `docs/index.md` now redirects to the handbook.
- Stage B contracts documented (added 2026-05-21 by docs swarm agent):
  - `handbook/error-codes.md` — full catalog of stable codes (INPUT_/CONFIG_/DEP_/RUNTIME_/STATE_/PARTIAL_)
  - `handbook/troubleshooting.md` — symptoms-first reverse index
  - `handbook/env-vars.md` — every `BACKPROPAGATE_*` knob
  - `handbook/cli-reference.md` — every subcommand + flag
  - README "Troubleshooting" + "Reporting bugs" + "Web UI" + "Platform prerequisites" subsections cover the load-bearing user-facing contracts (run_id correlation, --share+--auth gating, redacted stderr, sandbox env var).
- `examples/quickstart.jsonl` is a 5-line ShareGPT-format starter dataset referenced by the README Quick Start.
