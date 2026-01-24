# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Qwen2.5-3B model preset** - Smaller model for faster iteration and testing on 16GB VRAM
- **Official Qwen model fallback** - When pre-quantized models have corrupted cache, fall back to official models with `load_in_4bit=True`
- **Local dataset path helper** - `DatasetLoader.from_local()` for easy loading of local JSONL/JSON files

### Changed
- **CUDA_LAUNCH_BLOCKING now optional** - Disabled by default to improve training speed (was slowing down RTX 5080)
- **Default model updated** - Changed default from `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` to `Qwen/Qwen2.5-7B-Instruct` for better reliability

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
| 0.1.0 | 2026-01-19 | Initial release - SLAO, QLoRA, Windows support |

---

[Unreleased]: https://github.com/mcp-tool-shop/backpropagate/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mcp-tool-shop/backpropagate/releases/tag/v0.1.0
