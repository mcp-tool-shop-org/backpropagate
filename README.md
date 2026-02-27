<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/backpropagate/readme.png" alt="Backpropagate" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/backpropagate/"><img src="https://img.shields.io/pypi/v/backpropagate" alt="PyPI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/backpropagate"><img src="https://img.shields.io/codecov/c/github/mcp-tool-shop-org/backpropagate" alt="Coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**Headless LLM fine-tuning in 3 lines. Smart defaults, VRAM-aware batch sizing, multi-run SLAO, and one-click GGUF export for Ollama.**

*Train LLMs in 3 lines of code. Export to Ollama in one more.*

## Quick Start

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

## Why Backpropagate?

| Problem | Solution |
|---------|----------|
| Fine-tuning is complex | 3 lines: load, train, save |
| Windows is a nightmare | First-class Windows support |
| VRAM management is hard | Auto batch sizing, GPU monitoring |
| Model export is confusing | One-click GGUF + Ollama registration |
| Long runs cause forgetting | Multi-run SLAO training |

## Key Features

- **Headless by Design**: Built for CI/CD pipelines, automated workflows, and programmatic execution.
- **Smart Defaults**: Automatically configures optimal hyperparameters based on your hardware and dataset.
- **Multi-Run SLAO Training**: Advanced training strategies to prevent catastrophic forgetting during long runs.
- **First-Class Windows Support**: Tested and optimized for Windows environments, avoiding common PyTorch/CUDA pitfalls.
- **Seamless Export**: One-click export to GGUF format and automatic registration with Ollama.
- **Modular Architecture**: Install only the dependencies you need (e.g., `[unsloth]`, `[ui]`, `[export]`).

## Installation

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Extra | Description | Dependencies |
|-------|-------------|--------------|
| `unsloth` | 2x faster training, 50% less VRAM | unsloth |
| `ui` | Gradio web interface | gradio>=5.6.0 |
| `validation` | Pydantic config validation | pydantic, pydantic-settings |
| `export` | GGUF export for Ollama | llama-cpp-python |
| `monitoring` | WandB + system monitoring | wandb, psutil |

**Requirements:** Python 3.10+ · CUDA GPU (8GB+ VRAM) · PyTorch 2.0+

## Usage

### Basic Training

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

### Multi-Run SLAO Training

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")

result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
    merge_mode="slao",  # Smart LoRA merging
)
```

### Export to Ollama

```python
trainer.export(
    format="gguf",
    quantization="q4_k_m",
    register_ollama=True,
    model_name="my-finetuned-model",
)
# ollama run my-finetuned-model
```

### CLI

```bash
backprop train --data my_data.jsonl --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backpropagate --ui --port 7862
```

## Windows Support

Backpropagate is designed to work on Windows out of the box:

- Pre-tokenization to avoid multiprocessing crashes
- Automatic xformers disable for RTX 40/50 series
- Safe dataloader settings
- Tested on RTX 5080 (16GB VRAM)

## Model Presets

| Preset | VRAM | Speed | Quality |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12GB | Medium | Best |
| Qwen 2.5 3B | ~8GB | Fast | Good |
| Llama 3.2 3B | ~8GB | Fast | Good |
| Llama 3.2 1B | ~6GB | Fastest | Basic |
| Mistral 7B | ~12GB | Medium | Good |

## Architecture

```
backpropagate/
├── trainer.py           # Core Trainer class
├── multi_run.py         # Multi-run SLAO training
├── slao.py              # SLAO LoRA merging algorithm
├── datasets.py          # Dataset loading & filtering
├── export.py            # GGUF/Ollama export
├── config.py            # Pydantic settings
├── gpu_safety.py        # GPU monitoring & safety
└── ui.py                # Gradio interface
```

## Privacy

All training happens locally on your GPU. Backpropagate makes no network requests except to download models from HuggingFace (which you initiate). No telemetry, no cloud dependency.

## Scorecard

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | 6/8 | SECURITY.md, trust model, no secrets/telemetry, safe_path(). MCP items skipped |
| B. Error Handling | 3/7 | Structured exceptions + exit codes + no raw stacks. MCP/desktop/vscode skipped |
| C. Operator Docs | 4/7 | README, CHANGELOG, LICENSE, --help. Logging/MCP/complex skipped |
| D. Shipping Hygiene | 6/9 | verify.sh, version=tag, 5 scanners in CI, dependabot, python_requires, clean build |
| E. Identity | 4/4 | Logo, translations, landing page, metadata |
| **Total** | **23/31** | 14 items skipped with justification · `shipcheck audit` passes 100% |

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
