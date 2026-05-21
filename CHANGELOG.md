# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-05-21

A minor release that takes the project from "polished v1" to "real v1" via a 10-wave dogfood swarm. Bug + security pass, proactive health pass, UX humanization, full UI redesign (Gradio → Reflex), 5 P0 features.

### Added

- **Reflex web UI** — the optional `[ui]` extra now installs Reflex (Radix UI) instead of Gradio. Pure-Python implementation, WebSocket-driven live state, refined Ocean Mist palette, full dark + light mode, WCAG 2.4.7 focus indicators, 30 SVG icons, heartbeat / sparkline / event-log / structured-error / recovery-banner patterns
- **Hugging Face Hub push** — `backprop push <local> --repo <owner/name>` + `backprop export --push-to-hub <repo>` for one-shot export+push. Adapter-only by default; `--include-base` for the full merged model. Token resolution from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` / HF CLI cache. `model_card.md` is mirrored to the repo's `README.md` so HF picks it up as the model card
- **Resume from checkpoint** — `backprop resume <run_id>` (and `backprop train --resume <run_id>` / `backprop multi-run --resume`) reconstructs a crashed or interrupted run from RunHistoryManager + the atomic checkpoint manifest. A 5-run multi-run that crashes at run 4 is now recoverable
- **Run history** — `RunHistoryManager` is now actually wired into Trainer + MultiRunTrainer. New `backprop list-runs` (with `--json`, `--status`, `--limit` filters + aligned columns) and `backprop show-run <run_id>` (partial-prefix matching) subcommands surface the history
- **Model card generation** — every export emits a `model_card.md` following the HF model-card schema, with full provenance (run_id, base model, dataset hash, seed, training duration, ASCII loss sparkline, Ship Gate trust signals). Opt out via `--no-model-card`
- **Experiment tracking auto-wired** — `[monitoring]` extra (W&B, TensorBoard) now actually integrates. `report_to` defaults to `"auto"` (detect what's installed); the run shows up with name `backprop-<run_id_short>` for cross-system correlation
- **Atomic checkpoint writes** — Trainer.save / SLAOMerger.save / export_lora / export_gguf all write to `<path>.partial` then rename to final. Disk-full mid-write no longer leaves corrupt artifacts
- **OOM auto-recovery** — `Trainer(oom_recovery=True)` (default-on) halves batch_size + doubles gradient_accumulation_steps on `torch.cuda.OutOfMemoryError`, preserving effective batch. Aborts after 3 consecutive failures at batch=1
- **HF Hub transient retry** — every `from_pretrained` / `load_dataset` / `snapshot_download` retries on 5xx / 429 / connection errors with exponential backoff. 401 / 403 / 404 surface in < 1s with cause-classified hints
- **GPU pause-on-overheat** — `Trainer(pause_on_overheat=True)` now actually pauses training (the wiring was a no-op in v1.0)
- **Unsloth fallback** — `Trainer(unsloth_fallback=True)` (default-on) falls back to AutoModelForCausalLM + peft on Unsloth failures
- **run_id correlation** — every training run mints a UUID4 that flows through every log line + checkpoint manifest + SLAO merge record
- **Stable error codes** — `BackpropagateError.code` is now an explicit Ship Gate registry-prefixed identifier on every subclass. 28-entry `ERROR_CODES` catalog visible via `backprop info --error-codes`. `cause_category` enum on ModelLoadError surfaces cause-specific remediation hints
- **CLI exit codes** — proper 0 / 1 user-error / 2 runtime-error / 3 partial-success / 130 SIGINT contract
- **Stage C humanization** — structured errors with actionable hints, progress feedback on long ops, bare `backprop` prints help, `backprop info --json` for support attachments, friendly first-run messages
- **CI hardening** — every third-party GitHub Action SHA-pinned. PyPI publish via OIDC trusted publishing (Sigstore provenance). Docker image digest-pinned + HEALTHCHECK. Multi-OS test matrix (Linux + Windows + macOS + Python 3.13). pip-audit + Trivy + Bandit + Semgrep + TruffleHog all gate on findings
- **Documentation** — new handbook pages: `error-codes.md`, `troubleshooting.md`, `env-vars.md`, `cli-reference.md`. README Troubleshooting + Reporting bugs + Web UI subsections. `examples/quickstart.jsonl` so the "3 lines" Quick Start runs on a clean install

### Changed

- **Default model** — `Trainer()` (and `backprop train` / `multi-run` CLI defaults) now use `Qwen/Qwen2.5-7B-Instruct` instead of `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`. The non-quantized form works without bitsandbytes; users who want the bnb-4bit speedup install `[unsloth]` and pass `--model unsloth/...` explicitly
- **safe_path stricter** — absolute path + `..` segment + no `allowed_base` argument now raises `PathTraversalError` instead of warn-only-and-pass-through
- **Multi-run validation-overlap fix** — `_get_data_chunk` and `_get_replay_samples` now hard-cap at the train/validation boundary. Silent contamination is impossible; `ConfigurationError` surfaces a clear "reduce samples or increase dataset" hint
- **Random state isolation** — multi-run replay sampling uses a local `random.Random(seed)` instead of mutating the global Python RNG
- **SLAO NaN/inf detection** — `SLAOMerger.merge` raises `SLAO_MERGE_DIVERGED` with run_index + run_id + offending layer on non-finite weights
- **Rate limiter Address handling** — `_extract_client_ip` now correctly reads `.host` from Starlette's `Address` namedtuple (was including `:port`, giving every TCP connection its own bucket)
- **UI output dir denylist** — `BACKPROPAGATE_UI__OUTPUT_DIR` is validated against a denylist (`/etc`, `~/.ssh`, etc.) on first use
- **`--share` + `--auth` gating** — `backprop ui --share` now requires `--auth user:pass` (or explicit env-var opt-out with 5-second grace period + loud warning)
- **Scorecard re-audited** — B (Error Handling) row 3/7* → 5/7. Total 23/31 → 25/31

### Removed

- **Gradio web UI** — moved to `backpropagate/ui_gradio_legacy.py` with a DEPRECATED docstring. Preserved for v1.1 reference; will be removed in v1.2. `backpropagate.launch` / `create_backpropagate_theme` / `get_theme_info` / `get_css` now raise `ImportError` with the migration message

### Tests

1654 → 1766 (+112): regression tests for every Stage A/B contract that landed and every P0 feature that shipped. Coverage threshold holds at 50%.

## [1.0.5] - 2026-04-15

### Fixed

- Release binaries workflow: v1.0.4 tag was cut before Linux exclusion fix landed, causing >2GB upload failure

## [1.0.4] - 2026-04-14

### Fixed

- Linux binary build: replace CUDA torch (~870MB) with CPU-only torch (~200MB) after install to keep binary under 2GB GitHub release limit
- Strip step SIGPIPE crash: `du | head -5` with `set -eo pipefail` caused false build failure

## [1.0.3] - 2026-04-14

### Added

- `release-binaries.yml` workflow for standalone PyInstaller binaries on Windows + Linux

### Fixed

- PyInstaller build pipeline iteration: hidden-import handling for torch/transformers (recursion limit), `--collect-data` removed to stay under 4GB onefile cap, Linux binary size reduction via strip + module exclusion (lead-up fixes; the final size cut that actually landed under 2GB shipped in v1.0.4)
- Full-install CUDA-torch override on Linux (uses CPU torch index instead)
- `pywin32-ctypes` dependency for Windows PyInstaller builds
- Forced uninstall of CUDA packages before PyInstaller to avoid CUDA torch contamination

## [1.0.2] - 2026-03-25

### Fixed

- CLI `--version` was hardcoded to 0.1.0 — now reads from package metadata dynamically
- `__init__.py` docstring referenced v0.1.0 — updated to v1.0.1
- SECURITY.md supported versions updated from 0.x.x to 1.0.x

### Added

- 2 new version regression tests in test_cli.py

## [1.0.1] - 2026-02-27

### Added
- Ship Gate audit — all hard gates pass (23/31 checked, 14 skipped, 100%)
- verify.sh — single-command verification script (Ship Gate D1)
- Proper CLI exit codes: 1 user error, 2 runtime error, 3 partial success (Ship Gate B2)
- SHIP_GATE.md (the scorecard itself is rendered inline in README and on the landing page; no standalone SCORECARD.md file)

### Changed
- Scorecard in README and landing page reflects actual `shipcheck audit` results

## [1.0.0] - 2026-02-27

### Changed
- **v1.0.0 stable release** — production-ready
- Development status upgraded from Alpha to Production/Stable

## [0.1.7] - 2026-02-27

### Added
- Codecov badge in README
- Quality scorecard in README and landing page (46/50)
- Privacy section in README

### Changed
- Logo URL updated to brand repo (centralized)
- Landing page footer standardized to MCP Tool Shop link
- Landing page scorecard section added
- Updated translations (7 languages)

## [0.1.4] - 2026-02-22

### Fixed
- **Bandit Security Scan** - Fixed false positive security scan issues that caused CI to fail

## [0.1.3] - 2026-02-22

### Added
- **Qwen2.5-3B model preset** - Smaller model for faster iteration and testing on 16GB VRAM
- **Official Qwen model fallback** - When pre-quantized models have corrupted cache, fall back to official models with `load_in_4bit=True`
- **Local dataset path helper** - `DatasetLoader.from_local()` for easy loading of local JSONL/JSON files

### Changed
- **CUDA_LAUNCH_BLOCKING now optional** - Disabled by default to improve training speed (was slowing down RTX 5080)
- **Default model updated** - Changed default from `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` to `Qwen/Qwen2.5-7B-Instruct` for better reliability
- **Documentation** - Beefed up README with more badges, features, and GitHub Pages integration

### Fixed
- **BitsAndBytes JSON decode error** - Added fallback handling when pre-quantized model cache is corrupted

---

## [0.1.0] - 2026-01-19

### Added

#### Core Features
- **Trainer class** - Simple API for LLM fine-tuning with smart defaults
- **Multi-run training (SLAO)** - Multiple short runs with LoRA merging to prevent catastrophic forgetting
- **QLoRA support** - 4-bit quantization for training 7B models on 16GB VRAM
- **Windows support** - Pre-tokenization, safe multiprocessing, xformers auto-disable

#### Dataset Handling
- **DatasetLoader** - Auto-detect format (JSONL, CSV, HuggingFace)
- **Quality filtering** - Filter by token count, turn count, assistant presence
- **Perplexity filtering** - Remove outliers using GPT-2 perplexity scores
- **Deduplication** - Exact and MinHash-based duplicate removal
- **Curriculum learning** - Order samples by difficulty for progressive training

#### Export & Deployment
- **LoRA export** - Save adapter weights
- **Merged export** - Full model with adapter merged
- **GGUF export** - Quantized models for Ollama/llama.cpp (q4_k_m, q8_0, etc.)
- **Ollama integration** - Auto-generate Modelfile and register models

#### Safety & Monitoring
- **GPU monitoring** - Temperature, VRAM, utilization tracking
- **Safety thresholds** - Configurable limits with auto-pause
- **Checkpoint management** - Automatic saving with configurable policies

#### Security
- **Path traversal protection** - Safe file operations
- **Secure model loading** - `weights_only=True` for torch.load
- **Input validation** - Sanitized paths and parameters
- **Gradio CVE fix** - Requires gradio>=5.6.0

#### Developer Experience
- **Modular installation** - Install only what you need (`[unsloth]`, `[ui]`, `[full]`)
- **Feature flags** - Runtime detection of optional dependencies
- **Lazy imports** - Fast startup, helpful error messages
- **Type hints** - Full type coverage
- **Pre-commit hooks** - Ruff, mypy, bandit

### Technical Details
- Python 3.10+ required
- PyTorch 2.0+ with CUDA support
- Tested on RTX 5080 (16GB VRAM) with Windows 11

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.1.0 | 2026-05-21 | Reflex UI, HF Hub push, resume-from-checkpoint, run history, model cards, W&B wiring (10-wave dogfood swarm) |
| 1.0.5 | 2026-04-15 | Release-binaries workflow re-cut after v1.0.4 Linux exclusion fix |
| 1.0.4 | 2026-04-14 | Linux binary <2GB (CPU torch swap), strip SIGPIPE fix |
| 1.0.3 | 2026-04-14 | Standalone PyInstaller binary workflow (Windows + Linux) |
| 1.0.2 | 2026-03-25 | CLI version fix, regression tests |
| 1.0.1 | 2026-02-27 | Ship Gate audit, verify.sh, proper exit codes |
| 1.0.0 | 2026-02-27 | Stable release - production-ready |
| 0.1.7 | 2026-02-27 | Codecov, quality scorecard, privacy section |
| 0.1.4 | 2026-02-22 | Bandit false positive fix |
| 0.1.3 | 2026-02-22 | Qwen2.5-3B preset, local dataset helper |
| 0.1.0 | 2026-01-19 | Initial release - SLAO, QLoRA, Windows support |

---

[Unreleased]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.0.5...v1.1.0
[1.0.5]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v0.1.7...v1.0.0
[0.1.7]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v0.1.4...v0.1.7
[0.1.4]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v0.1.0...v0.1.3
[0.1.0]: https://github.com/mcp-tool-shop-org/backpropagate/releases/tag/v0.1.0
