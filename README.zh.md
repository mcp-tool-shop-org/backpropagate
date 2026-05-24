<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**使用 3 行代码进行无头 LLM 微调。 智能默认设置、感知 VRAM 的批处理大小、多轮 SLAO 训练，以及一键导出为 Ollama 兼容的 GGUF 格式。**

*SLAO 是一种通过非对称合并实现的单 LoRA 持续学习方法，它是一种在长时间微调过程中防止灾难性遗忘的技术（[论文](https://arxiv.org/abs/2512.23017)）。*

*使用 3 行代码训练 LLM。 只需要再一行代码即可导出到 Ollama。*

## 快速开始

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("examples/quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

该仓库包含一个名为 `examples/quickstart.jsonl` 的小型文件（包含 5 个 ShareGPT 格式的示例），以便在全新安装的环境中直接运行上述代码。 对于您自己的训练，请参阅下面的 [数据集格式](#dataset-format)。

### 无代码方式：Web UI

是否更喜欢使用 UI 而不是 Python REPL？ 安装相同的扩展，然后运行：

```bash
pip install backpropagate[standard]
backprop ui --port 7862
```

Reflex (Radix UI) 界面允许您选择一个 JSONL 文件、选择一个模型、进行训练并导出，无需使用 Python。 该 UI 采用本地优先的设计；要将其暴露到公共互联网，请参阅下面的 [Web UI](#web-ui)，了解 `--share` + `--auth` 安全协议以及支持的隧道选项（Cloudflare Tunnel、ngrok）。

## 数据集格式

您的 JSONL 训练文件应每行包含一个示例。 最简单的格式是 ShareGPT 对话：

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

也支持 Alpaca (`instruction`/`output`)、OpenAI 对话 (`messages`) 以及原始文本格式。 请参阅 `examples/quickstart.jsonl`，其中包含可复制的起始示例。

## 为什么需要反向传播？

| 问题 | 解决方案 |
|---------|----------|
| 微调很复杂 | 3 行代码：加载、训练、保存 |
| Windows 系统很麻烦 | 一流的 Windows 支持 |
| VRAM 管理很困难 | 自动批处理大小、GPU 监控 |
| 模型导出很复杂 | 一键导出为 GGUF 格式，并自动注册到 Ollama |
| 长时间运行会导致遗忘 | 多轮 SLAO 训练 |

## 主要特性

- **无头设计**: 专为 CI/CD 流水线、自动化工作流程和程序化执行而设计。
- **智能默认设置**: 根据您的硬件和数据集自动配置最佳超参数。
- **多轮 SLAO 训练**: 采用高级训练策略，以防止长时间运行期间的灾难性遗忘。
- **一流的 Windows 支持**: 经过测试和优化，适用于 Windows 环境，避免常见的 PyTorch/CUDA 问题。
- **无缝导出**: 一键导出为 GGUF 格式，并自动注册到 Ollama。
- **模块化架构**: 只安装您需要的依赖项（例如，`[unsloth]`、`[ui]`、`[export]`）。

## 安装

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Reflex (Radix UI) web interface
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| 扩展 | 描述 | 依赖项 |
|-------|-------------|--------------|
| `unsloth` | 训练速度提升 2 倍，VRAM 减少 50% | unsloth |
| `ui` | Reflex (Radix UI) Web 界面 | reflex>=0.9.2, fastapi>=0.115 |
| `validation` | Pydantic 配置验证 | pydantic, pydantic-settings |
| `export` | 用于 Ollama 的 GGUF 导出 | llama-cpp-python |
| `monitoring` | WandB + 系统监控（在 v1.1.0 中自动集成到训练器中） | wandb, psutil |
| `logging` | 结构化日志记录 | structlog |
| `security` | JWT 身份验证 + 令牌生成 | PyJWT, cryptography |
| `production` | unsloth + ui + validation + logging + security | (捆绑) |

**要求：** Python 3.10+ · CUDA GPU (8GB+ VRAM) · PyTorch 2.0+

### 平台先决条件

Backpropagate 能够处理运行时的一些问题（多进程、RTX 40/50 上的 xformers、Windows 上的数据加载器）。它**不能**解决安装时遇到的平台相关问题，请先解决这些问题：

- **CUDA 工具包版本。** PyTorch 的发布与 CUDA 版本相关，选择错误的 wheel 可能会静默地安装仅支持 CPU 的 PyTorch 版本。请使用 <https://pytorch.org/get-started/locally/> 上的选择器，获取适用于您驱动程序的精确 `pip install torch ...` 命令。运行 `nvidia-smi` 以查看您的驱动程序/CUDA 版本。
- **Windows。** 需要 Visual Studio Build Tools (C++) 和 CMake，用于 `[export]` 扩展（从源代码构建 `llama-cpp-python`）。现在已发布适用于 Windows 的 `bitsandbytes` wheel（>= 0.43）；较旧的指南中提到的 `bitsandbytes-windows` 已经过时。
- **macOS。** **不支持** GPU 训练，因为它不支持 CUDA。您可以安装 Backpropagate，通过 Ollama 运行导出的 GGUF 文件进行*推理*，但 `trainer.train()` 会引发 `DEP_GPU_NOT_AVAILABLE` 错误。请使用支持 CUDA 的机器进行训练。
- **Linux。** 大多数发行版可以直接使用。如果您使用的是 PyPI 的二进制发布版本，请注意，Linux 构建版本仅支持 CPU 版本的 PyTorch（以保持在 GitHub 的 2 GB 发布文件大小限制内）；请先安装来自 pytorch.org 的匹配的 CUDA wheel。

有关更详细的安装故障排除信息，请参阅 [故障排除手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)。

## 配置

所有设置都可以通过使用 `BACKPROPAGATE_` 前缀的环境变量进行覆盖（例如，`BACKPROPAGATE_LOG_LEVEL=debug`）。当安装了 `[validation]` 扩展时，项目根目录下的 `.env` 文件将被自动加载。

常用配置项（请参阅 [完整的环境变量参考](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/)，了解所有选项）：

| 变量 | 默认值 | 说明 |
|----------|---------|-------|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | 自动 | 强制使用 JSON (`true`) 或控制台 (`false`) 日志 |
| `BACKPROPAGATE_LOG_FILE` | 未设置 | 用于将日志写入的路径 |
| `BACKPROPAGATE_DEFER_FEATURE_DETECTION` | 未设置 | 在启动时跳过可选依赖项检测，以实现最快的 CLI 启动速度 |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | 所有 UI 文件系统写入的基础目录；已进行白名单验证 |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | 默认模型 |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | 学习率 |
| `BACKPROPAGATE_LORA__R` | `16` | LoRA 秩 |

嵌套键使用双下划线作为分隔符（Pydantic 的 `env_nested_delimiter` 约定）。

## 用法

### 基本训练

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

`Qwen/Qwen2.5-7B-Instruct` 是默认的、标准的配置。当 `Trainer()` 函数在没有指定模型参数的情况下被调用时，该值会被解析（参见 `config.py` 文件的 `ModelConfig.name`）。 较早的示例使用了预量化的 `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` 模型；我们已将默认配置更改为官方的 Qwen 模型权重，以提高可靠性（[CHANGELOG v1.1.0](CHANGELOG.md#110---2026-05-21)）。 两种模型都可以使用。

### 多轮 SLAO 训练

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")

result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
    merge_mode="slao",  # Single LoRA Continual Learning via Asymmetric Merging
)
```

SLAO（通过非对称合并的单 LoRA 持续学习）实现了 [Merge before Forget](https://arxiv.org/abs/2512.23017) 论文：通过 QR 分解进行正交 A 矩阵初始化，非对称 A/B 处理，以及基于时间的 `λ(i) = 1/√i` 缩放。CLI 标志是 `--samples`（底层字段是 `samples_per_run`）。

### 导出到 Ollama

```python
# Export to GGUF
result = trainer.export("gguf", quantization="q4_k_m")

# Register with Ollama separately
from backpropagate import register_with_ollama
register_with_ollama(result.path, "my-finetuned-model")
# ollama run my-finetuned-model
```

### CLI

```bash
backprop train --data my_data.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backprop ui --port 7862
backprop info
backprop list-runs                              # v1.1.0: query past training runs
backprop show-run <run-id>                      # v1.1.0: detail view
backprop resume <run-id>                        # v1.1.0: resume a crashed multi-run
backprop push ./output/lora --repo me/my-model  # v1.1.0: push adapter to HF Hub
```

请参阅 [CLI 参考](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/)，了解每个子命令和标志，或运行 `backprop <子命令> --help`。

### 从检查点恢复（v1.1.0）

一个原本会在第4次运行时崩溃的5次连续运行流程现在可以恢复。 每次连续运行都会将运行ID写入`run_history.json`文件以及磁盘上的检查点清单，因此，要从中断处恢复，只需执行一个命令即可。

```bash
backprop resume <run-id>                       # picks up the in-progress session
backprop multi-run --data ... --resume <run-id> # explicit form
backprop train --data ... --resume <run-id>    # single-run resume (continues run_id)
```

`backprop multi-run` 命令的默认行为是自动检测同一输出目录中是否已存在的正在进行的任务，并继续执行。 如果要强制创建一个全新的会话，可以使用 `resume_from="off"` 参数（Python API），或者省略 `--resume` 参数，并指定一个新的输出目录。

当一个多轮训练恢复时，模型会加载该`run_id`对应的最新检查点，从检查点旁边的`slao/`目录恢复SLAO的合并状态，然后从`last_completed_run + 1`处继续训练循环。历史记录条目的`status`状态会恢复为`running`，因此运行`backprop list-runs --status running`命令可以显示正在进行的会话。

### 实验跟踪功能 (版本 1.1.0)

`Trainer` 会自动检测已安装的实验跟踪器（`wandb`、`tensorboard`、`mlflow`），并将它们与底层的 `transformers.TrainingArguments` 进行连接。默认情况下，`report_to="auto"` 会自动选择可导入的跟踪器。

```bash
pip install backpropagate[monitoring]  # installs wandb + psutil
wandb login                            # one-time
backprop train --data my_data.jsonl    # W&B run gets the same run_id prefix as the on-disk history
```

可以使用 `Trainer(report_to=["wandb"])`、`Trainer(report_to=["tensorboard"])` 或 `Trainer(report_to="none")` 来显式地禁用某些功能。要使用 MLflow，请安装 `pip install mlflow`；要使用 TensorBoard，请安装 `pip install tensorboard`。W&B 运行的名称格式为 `backprop-<run_id_prefix>`，这样操作人员可以使用相同的标识符在 W&B、我们的日志以及 `run_history.json` 文件中进行搜索。

### 培训经历

每次 `backprop train` 和 `backprop multi-run` 的执行都会在 `<output>/run_history.json` 文件中记录一行，包含运行ID、模型、数据集、超参数、状态、最终损失、损失历史，以及（对于多轮运行）SLAO 合并的时间线。列出最近的运行记录：

```bash
backprop list-runs                         # most recent 20 runs, all statuses
backprop list-runs --status failed         # filter
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial run_id ok)
```

运行历史信息在不同进程中得以保留。Web界面的“运行”标签显示的是一个独立的、内存中的视图；而存储在磁盘上的历史记录才是 `list-runs`（列出运行记录）、`show-run`（显示运行记录）和 `resume`（恢复运行）命令的权威数据来源。

### 网页用户界面

启动本地的 Reflex 界面。

```bash
backprop ui --port 7862
```

要公开一个可以通过公共互联网访问的URL，您必须同时使用 `--share` 和 `--auth` 选项。

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` 命令，如果未指定 `--auth` 参数，将以错误代码 `1` 退出，并显示结构化错误信息 `[RUNTIME_UI_AUTH_NOT_ENFORCED]`。 原因是：`--share` 会发布一个公开的 URL，任何人都可以访问。 如果没有身份验证，这意味着任何人都可以控制您的训练流水线。 如果您不想设置凭据，请使用 SSH 端口转发：`ssh -L 7860:localhost:7860 <host>`，然后在本地打开 `http://localhost:7860`。 请参阅 [handbook/security.md](site/src/content/docs/handbook/security.md) 以获取完整的安全风险评估。

用户界面中的文件写入操作会被限制在一个单独的目录中，以提高安全性。

- 默认值：`~/.backpropagate/ui-outputs`
- 可重写：`BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- 重写设置会经过**黑名单验证**—— 系统路径和凭据路径（如`/etc`、`/var`、`~/.ssh`、`~/.aws`、`C:\Windows\System32`等）会被拒绝，并显示错误信息`[UI_OUTPUT_DIR_FORBIDDEN]`。

## Windows 支持服务

Backpropagate 软件设计时就考虑了在 Windows 操作系统上的兼容性，可以直接运行。

- 为了避免多进程崩溃，进行了预处理。
- 自动禁用 xformers，适用于 RTX 40/50 系列显卡。
- 采用安全的 DataLoader 设置。
- 已在 RTX 5080 (16GB 显存) 上进行测试。

## 模型预设

| 预设。 | 显存 (xiǎn cún) | 速度。 | 质量。 |
|--------|------|-------|---------|
| Qwen 2.5 7B。 | 约12GB。 | 中等。 | 祝好。 |
| Qwen 2.5 3B 模型。 | 约8GB。 | 快速。 | 好的。 |
| Llama 3.2，30亿参数版本。 | 约8GB。 | 快速。 | 好的。 |
| Llama 3.2 1B | 约6GB。 | 最快的。 | 基础。 |
| Mistral 7B | 约12GB。 | 中等。 | 好的。 |

## 架构

```
backpropagate/
├── trainer.py           # Core Trainer class
├── multi_run.py         # Multi-run SLAO training
├── slao.py              # SLAO LoRA merging algorithm
├── datasets.py          # Dataset loading, filtering & curriculum
├── export.py            # GGUF/Ollama export
├── config.py            # Pydantic settings + training presets
├── gpu_safety.py        # GPU monitoring & safety
├── cli.py               # CLI entry point (backprop command)
├── checkpoints.py       # Checkpoint management
├── exceptions.py        # Structured error hierarchy
├── feature_flags.py     # Optional feature detection
├── security.py          # Path traversal & torch security
├── logging_config.py    # Structured logging setup
├── ui_theme.py          # Radix theme tokens + CSS (Reflex era)
├── ui_state.py          # rx.State subclasses
├── ui_app/              # Reflex web interface (Radix UI)
│   ├── app.py           #   rx.App entry point
│   ├── chrome.py        #   Header / LeftNav / SideRail / Footer
│   ├── pages/           #   Train / Multi-Run / Export / Dataset
│   └── components/      #   Bp* primitives (status pill, sparkline, event log…)
└── ui_security.py       # Rate limiting, CSRF, file validation (framework-agnostic)
```

v1.0 的 Gradio 实现（`ui_gradio_legacy.py` + `theme_gradio_legacy.py`）在 v1.1.x 版本中被保留，仅作为参考，并在 v1.2.0 版本中被移除。

## 故障排除

常见首次运行失败的简要索引。完整的反向索引位于[故障排除手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)；以下每个代码都记录在[错误代码](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/)中。

| 症状 | 代码 | 解决方法 |
|---------|------|-----|
| 训练过程中 GPU 内存耗尽 | `RUNTIME_GPU_OOM` | OOM 自动恢复 (B-002) 会自动将批次大小减半，最多 3 次。要禁用此功能，请使用：`Trainer(oom_recovery=False)`。要强制使用更小的批次大小，请使用：`--batch-size 1`。 |
| HF Hub 返回 401 错误 / "未找到模型" | `DEP_MODEL_LOAD_FAILED` | 使用 `huggingface-cli login` 重新登录。如果出现拼写错误，请从 <https://huggingface.co/models> 复制确切的 ID。 |
| 模型名称拼写错误 | `INPUT_VALIDATION_FAILED` 或 `DEP_MODEL_LOAD_FAILED` | 验证 <https://huggingface.co/models> 上的 `org/name` 标识符。 |
| `register_with_ollama` 连接被拒绝 | `DEP_OLLAMA_REGISTRATION_FAILED` | 启动守护进程：`ollama serve`。 从 <https://ollama.com> 安装。 可重试。 |
| 保存检查点时磁盘已满 | `STATE_CHECKPOINT_INVALID` | 在崩溃时，原子写入会在目录中留下一个 `.partial` 目录，可以安全删除。之前的有效检查点仍然存在。 |
| GPU 过热导致训练暂停/中止 | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | B-003 监视器在 NVML 温度阈值处暂停，GPU 冷却后会自动恢复。 改善散热或降低持续负载。 |
| `backprop ui --share` 被拒绝 | `INPUT_AUTH_REQUIRED` | 使用 `--auth user:password`。自 v1.2.0（GHSA-f65r-h4g3-3h9h）起，`--share` 在没有 `--auth` 的情况下是硬性错误，不可关闭。如果无法公开凭据，请使用 SSH 端口转发。 |
| 多次运行出现 "验证重叠" | `CONFIG_INVALID` (Stage A backend B-001) | 降低 `--samples` 的值，使其低于训练池大小，增加数据集，或禁用验证。 |
| 首次尝试 GGUF 导出失败 | `RUNTIME_GGUF_EXPORT_FAILED` | 使用 `pip install backpropagate[export]`。 在 Windows 上，还需要 Visual C++ Build Tools + CMake。 |

## 报告错误

当出现错误时，Backpropagate 在启动时会打印一条 `run_started run_id=<uuid>` 行，并将相同的 ID 绑定到检查点清单、SLAO 合并历史记录和结构化日志行。 在任何错误报告中，请包含 `run_id`，它允许维护人员关联该次运行中的每一行日志、每个检查点和每个合并。

一个好的错误报告应包含：

1. **`run_id`** — 启动时打印的 UUID（也可作为 `TrainingRun.run_id` 和 `RunResult.run_id`）。
2. **错误代码** — `stderr` 中出现的 `[CODE_NAME]: message` 行，可以使用 `grep` 命令查找；请参阅 [错误代码](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) 了解详细信息。
3. **已删除的命令行。** 在非详细模式下，`stderr` 会自动删除（Bearer 令牌、`sk-*`、`hf_*`、AWS 密钥、`password=`/`token=`/`api_key=` 对会被删除），可以安全粘贴。 要获取完整的未删除的堆栈跟踪，请使用 `--verbose` 重新运行，但在发布之前请仔细检查。
4. **Python / PyTorch 版本、GPU 型号、操作系统。** `backprop info` 可以一次性打印所有这些信息。

## 隐私

所有训练都在您的 GPU 上本地进行。 Backpropagate 仅向 HuggingFace 下载模型时才会进行网络请求（您需要主动发起）。 没有遥测，没有云依赖。

## 评分卡

| 类别 | 评分 | 说明 |
|----------|-------|-------|
| A. 安全性 | 6/8 | SECURITY.md，信任模型，无秘密/遥测，safe_path()。 MCP 项目已跳过。 |
| B. 错误处理 | 5/7 | 结构化异常信息（`code`/`message`/`hint`/`cause`/`retryable`），通过 `ERROR_CODES` 注册表提供；CLI 退出码：0/1/2/3；不显示原始堆栈跟踪，除非使用 `--verbose` 参数；`run_id` 关联；已屏蔽的标准错误输出；`--share` + `--auth` 权限控制。MCP/桌面/VS Code 已跳过。 |
| C. 运维文档 | 4/7 | README、CHANGELOG、LICENSE、--help。日志记录/MCP/复杂功能已跳过。 |
| D. 发布流程 | 6/9 | `verify.sh`、版本号=标签、CI 中包含 5 个扫描器、dependabot、`python_requires`、干净的构建。 |
| E. 身份标识 | 4/4 | Logo、翻译、着陆页、元数据。 |
| **Total** | **25/31** | 跳过了 14 个项目，并已给出理由；`shipcheck audit` 检查通过 100%；审计日期：2026-05-21（B 行在 B 阶段和 A 阶段 CLI 退出码工作完成后重新评估）。 |

设计历史以及每个条目对应的具体内容：请参阅 [ROADMAP.md](ROADMAP.md)——所有第 1-4 周的项目都已在 v1.1.0 版本中发布。

## 许可证

MIT — 详情请参阅 [LICENSE](LICENSE)。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
