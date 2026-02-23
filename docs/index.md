# Backpropagate

**Headless LLM Fine-Tuning** -- Making fine-tuning accessible without the complexity.

Part of [MCP Tool Shop](https://mcp-tool-shop.github.io/).

---

## Features

- **3-Line Training** -- Load a model, train on your data, save. That simple.
- **Smart Defaults** -- Auto-configures hyperparameters based on your hardware and dataset.
- **Multi-Run SLAO** -- Prevents catastrophic forgetting during long training runs.
- **First-Class Windows Support** -- Tested on Windows with RTX 40/50 series GPUs.
- **One-Click Export** -- GGUF export and automatic Ollama registration.
- **Modular Extras** -- Install only the dependencies you need.

## Install

```bash
pip install backpropagate[standard]
```

## Quick Start

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")
```

## Links

- [PyPI](https://pypi.org/project/backpropagate/)
- [Source](https://github.com/mcp-tool-shop-org/backpropagate)
- [Issues](https://github.com/mcp-tool-shop-org/backpropagate/issues)
- [Changelog](https://github.com/mcp-tool-shop-org/backpropagate/blob/main/CHANGELOG.md)
- [MCP Tool Shop](https://mcp-tool-shop.github.io/)
