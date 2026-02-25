<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="assets/logo.png" alt="Backpropagate" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/backpropagate/"><img src="https://img.shields.io/pypi/v/backpropagate" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**仅需 3 行代码即可进行无头 LLM 的微调。 智能默认设置、感知显存的批处理大小、多轮 SLAO 训练，以及一键导出为 Ollama 兼容的 GGUF 格式。**

*只需 3 行代码即可训练 LLM。 额外一行代码即可导出到 Ollama。*

## 快速入门

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

## 为什么需要反向传播？

| 问题 | 解决方案 |
| --------- | ---------- |
| 微调过程复杂 | 3 行代码：加载、训练、保存 |
| Windows 系统是个噩梦 | 提供一流的 Windows 支持 |
| 显存管理很困难 | 自动批处理大小调整、GPU 监控 |
| 模型导出很麻烦 | 一键导出为 GGUF 格式，并自动注册到 Ollama |
| 长时间运行会导致遗忘 | 多轮 SLAO 训练 |

## 主要特性

- **无头设计**: 专为 CI/CD 流水线、自动化工作流程和程序化执行而设计。
- **智能默认设置**: 根据您的硬件和数据集自动配置最佳超参数。
- **多轮 SLAO 训练**: 采用高级训练策略，以防止在长时间运行过程中出现灾难性遗忘。
- **一流的 Windows 支持**: 针对 Windows 环境进行了测试和优化，避免常见的 PyTorch/CUDA 问题。
- **无缝导出**: 一键导出为 GGUF 格式，并自动注册到 Ollama。
- **模块化架构**: 仅安装您需要的依赖项（例如，`[unsloth]`、`[ui]`、`[export]`）。

## 安装

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Extra | 描述 | 依赖项 |
| ------- | ------------- | -------------- |
| `unsloth` | 训练速度提升 2 倍，显存占用减少 50% | unsloth |
| `ui` | Gradio Web 界面 | gradio>=5.6.0 |
| `validation` | Pydantic 配置验证 | pydantic, pydantic-settings |
| `export` | GGUF 导出，用于 Ollama | llama-cpp-python |
| `monitoring` | WandB + 系统监控 | wandb, psutil |

**要求：** Python 3.10+ · CUDA GPU (8GB+ 显存) · PyTorch 2.0+

## 使用方法

### 基本训练

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

### 多轮 SLAO 训练

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

### 导出到 Ollama

```python
trainer.export(
    format="gguf",
    quantization="q4_k_m",
    register_ollama=True,
    model_name="my-finetuned-model",
)
# ollama run my-finetuned-model
```

### 命令行界面

```bash
backprop train --data my_data.jsonl --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backpropagate --ui --port 7862
```

## Windows 支持

Backpropagate 旨在在 Windows 系统上开箱即用：

- 预处理分词，以避免多进程崩溃
- 自动禁用 RTX 40/50 系列的 xformers
- 安全的数据加载器设置
- 已在 RTX 5080 (16GB 显存) 上进行测试

## 模型预设

| 预设 | VRAM | Speed | 质量 |
| -------- | ------ | ------- | --------- |
| Qwen 2.5 7B | ~12GB | 中等 | Best |
| Qwen 2.5 3B | ~8GB | Fast | Good |
| Llama 3.2 3B | ~8GB | Fast | Good |
| Llama 3.2 1B | ~6GB | 最快 | Basic |
| Mistral 7B | ~12GB | 中等 | Good |

## 架构

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

## 相关项目

是 [**MCP Tool Shop**](https://mcp-tool-shop.github.io/) 的一部分：

- [Tool Compass](https://github.com/mcp-tool-shop-org/tool-compass) — 语义化的 MCP 工具发现
- [File Compass](https://github.com/mcp-tool-shop-org/file-compass) — 语义化的文件搜索
- [Comfy Headless](https://github.com/mcp-tool-shop-org/comfy-headless) — 简化版的 ComfyUI

## 许可证

MIT — 详情请参阅 [LICENSE](LICENSE)。
