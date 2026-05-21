# Backpropagate

> **Documentation has moved.** The canonical documentation surface is the Astro/Starlight handbook at <https://mcp-tool-shop-org.github.io/backpropagate/>. This Jekyll-rendered folder is preserved for legacy links — start at the [handbook landing page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/) for the latest content.

**Headless LLM Fine-Tuning** -- Making fine-tuning accessible without the complexity.

Part of [MCP Tool Shop](https://mcp-tool-shop.github.io/).

---

## Features

- **3-Line Training** -- Load a model, train on your data, save. That simple.
- **Smart Defaults** -- Auto-configures hyperparameters based on your hardware and dataset.
- **Multi-Run SLAO** -- Single LoRA Continual Learning via Asymmetric Merging ([arXiv:2512.23017](https://arxiv.org/abs/2512.23017)) prevents catastrophic forgetting during long training runs.
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

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("examples/quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")
```

## Links

- [Handbook (canonical docs)](https://mcp-tool-shop-org.github.io/backpropagate/handbook/)
- [PyPI](https://pypi.org/project/backpropagate/)
- [Source](https://github.com/mcp-tool-shop-org/backpropagate)
- [Issues](https://github.com/mcp-tool-shop-org/backpropagate/issues)
- [Changelog](https://github.com/mcp-tool-shop-org/backpropagate/blob/main/CHANGELOG.md)
- [MCP Tool Shop](https://mcp-tool-shop.github.io/)
