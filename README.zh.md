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
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/backpropagate"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/backpropagate/badge" alt="OpenSSF Scorecard"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

# 训练一个适配器。将其部署到 Ollama。然后继续

Backpropagate 是一个 Python 库，用于在单个 GPU 上对大型语言模型进行微调。只需三行代码即可在一个 16GB 显卡上训练一个 7B 模型。再添加一条命令，即可将其导出到 Ollama，以便你可以使用 `ollama run` 命令运行你的微调模型。它在 Windows 上也能很好地运行。

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")
```

```bash
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
ollama run my-model
```

就是这样。没有 YAML 配置文件。没有 `accelerate launch` 命令。没有单独的“现在将其转换为 GGUF”教程。如果你有一个 CUDA GPU 和一个包含训练数据的 JSONL 文件，那么你只需三行代码就可以完成一个可用的微调。

## 安装

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

如果你需要可选功能，请将安装命令替换为以下命令之一：

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

更喜欢 Docker 吗？`docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` 也可以。镜像同时支持 `linux/amd64` 和 `linux/arm64`，因此 Apple Silicon 和 ARM Linux 用户可以获得原生镜像。一个标准的 `compose.yaml` 文件，用于“在容器中运行 UI”，位于仓库的根目录中——`docker compose up` 命令将在 `http://localhost:7860` 上启动 Web UI，并挂载一个持久的 `~/.backpropagate` 卷。

## Backpropagate 在这个领域中的定位

有几个优秀的库可以用于微调 LLM。它们各自擅长不同的方面：

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**——如果你喜欢 YAML 配置文件，并且想要一个可以从中复制配方的社区。
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**——如果你想要 DPO/PPO/RLHF 和一个 Web GUI。
- **[Unsloth](https://github.com/unslothai/unsloth)**——如果你需要最快的训练速度，并且使用的是受支持的模型系列。
- **[torchtune](https://github.com/pytorch/torchtune)**——如果你想要 Meta 提供的、可以编辑的 PyTorch 原生配方。

Backpropagate 是缺失的选项：**一个 3 行 Python API，适用于在单个消费级 GPU 上进行操作的个人用户，他们想要训练一个适配器并将其部署。** 没有 YAML，没有 GUI，没有在线 RL（PPO/GRPO），也没有多节点。只有每个人真正需要的循环和阻碍流程的导出步骤。

如果你尝试了上述库中的一个，但因为配置文件的复杂性而放弃，或者遇到了模型系列的问题，或者想要默认支持 Windows——那么 Backpropagate 适合你。

## 你可以在 16GB 消费级 GPU 上进行哪些微调

以下是在 16GB 显卡（RTX 4080 / 5080 / 4070 Ti Super）上的实际范围：

| 模型 | 方法 | 状态 |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | 运行良好。完整的序列长度，还有剩余空间。 |
| SmolLM3-3B / Qwen2.5-3B / Llama-3.2-3B / Llama-3.2-1B | `mode="full"`（完全微调） | v1.4——在 `backprop train` 命令或 `Trainer(..., mode="full")` 中传递 `--mode=full`。加载全精度（bf16）权重——没有 4 位，也没有适配器；梯度检查点 + 分页 8 位 Adam 保持在 16GB 以内。 |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | 标准。~7-8 GB。Backpropagate 的默认预设。 |
| Llama-3 13B | QLoRA + 样本打包 | 比较紧张，但可以运行。使用较短的序列。 |
| Mixtral 8x7B（总共 470 亿个参数） | — | 超出范围——2 位（AQLM / QuIP#）破坏了可合并的适配器 + GGUF 导出协议，因此在 [v1.5 路线图简报](docs/V1_5_BRIEF.md) 中已停止使用。在 16GB 显卡上，使用 ≤8B 的基础模型。 |

`mode="full"` 允许使用最多 **40 亿个参数**的模型。上述“完全微调”行中的四个预设都是真实的 ~3B（实际参数数量为 3.08–3.24B），并且可以适应 16GB 显卡。3.8–4B 类别（Phi-4-mini-3.8B、Qwen-3.5-4B）也符合上限，但需要 **24GB+** 显卡进行完全微调——仅权重和梯度就接近 16GB，然后是优化器和激活——因此，在 16GB 显卡上，对这些模型使用 `mode="lora"`（它们位于 LoRA 行中）。参数 >4B 的模型会显示 `RUNTIME_FULL_FT_MODEL_TOO_LARGE` 错误。

2 位量化（AQLM / QuIP#）**超出范围**。它最初计划用于 v1.4，然后在 [v1.5 路线图简报](docs/V1_5_BRIEF.md) 中停止使用：2 位基础模型无法干净地合并回全精度权重，这破坏了 Backpropagate 的可合并适配器 → GGUF → Ollama 导出协议（这是整个流水线的目的）。Backpropagate 提供的替代方案是 v1.5 **FP8 计算路径**（`--fp8`，Blackwell/Hopper）和 `mode="full"`，用于 ≤4B 的模型——两者都保持可合并和可导出。

对于 3B 及更小模型，在 16GB 上进行完全微调（不仅仅是 LoRA）是可行的，并且现在在 v1.4 中作为 `mode="full"` 提供。传递 `Trainer(..., mode="full")` 或 `backprop train --mode=full --model phi-4-mini-3.8b` 以启用它。一个硬性限制会拒绝参数 > 4B 模型的模式，并显示 `RUNTIME_FULL_FT_MODEL_TOO_LARGE` 错误，并将 LoRA 和小于 4B 的预设作为恢复选项。有关配置数学 + Biderman 2024 / Thinking Machines 2025 质量比较，请参阅[完整的微调手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/)。对于 7B+ 模型，完全微调需要 24GB+ GPU——可以考虑租用 A100 云服务器，或者坚持使用 LoRA，最近的研究表明，在大多数后训练任务中，LoRA 的质量与完全微调相当（请参阅[反驳部分](#backpropagate-not-for) 中的引用）。

## Backpropagate 不适用于以下情况

如果你的用例如下，你最好使用不同的库——Backpropagate 不是正确的选择，并且尝试使其工作会花费更多的时间。在开始之前阅读本部分可以节省安装和放弃的周期：

- **对 7B+ 模型进行全参数微调**——Backpropagate 使用 LoRA/QLoRA，它训练一个小型适配器，而不是更新每个权重。对于 7B 及更大的模型，全参数微调需要 24GB+ 的 GPU 内存，并且无法在 16GB 的消费级显卡上运行。对于 3B 及更小的模型，在 16GB 显卡上进行全参数微调是可行的，并在 v1.4 版本中以 `mode="full"` 的形式提供（在 CLI 中传递 `Trainer(..., mode="full")` 或 `--mode=full`；一个硬性限制会为大于 4B 的模型触发 `RUNTIME_FULL_FT_MODEL_TOO_LARGE`，并将 LoRA 以及小于 4B 的预设设置为备选方案）。更广泛地说：最近的研究（[Biderman 2024](https://arxiv.org/abs/2405.09673)，[Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/））表明，在正确的配置下，LoRA 在大多数训练后任务（指令遵循、领域适应、个性/风格）上的质量与全参数微调相当，并且计算量仅为后者的 67%——因此，对于大多数用户实际需要的工作，坚持使用 LoRA 没有任何损失。`mode="full"` 适用于您已经衡量了质量差距并决定花费更多计算资源的情况。如果您确实需要对 7B+ 模型进行全参数微调，请直接在 24GB+ 显卡上使用 HuggingFace `transformers.Trainer`。
- **在线强化学习——PPO/GRPO/RLVR**——Backpropagate 执行单阶段 SFT 以及无参考偏好调整（ORPO 包含在 v1.5 中；SimPO/KTO 计划中）。它不执行在线强化学习——PPO、GRPO 或 RLVR——这需要一个奖励模型或在训练步骤之上进行生成和评分循环。对于这些，请直接使用 TRL 或 LLaMA-Factory。（无参考偏好调整符合单阶段的范围，因为没有单独的参考模型需要保存在内存中；请参阅 [快速入门](#quick-start) 下的 ORPO 注释。）
- **多节点训练**——仅支持单个机器上的单个 GPU。单个机器上的多 GPU 也可以工作（通过 `accelerate launch`），但未正式支持。
- **在 CUDA 环境下进行 macOS 训练**——Apple Silicon 没有 CUDA，因此 CUDA 路径必须在配备 NVIDIA GPU 的 Linux 或 Windows 机器上运行。您仍然可以通过 Ollama 在 Mac 上运行训练好的模型。**v1.5 版本的新功能：**一个实验性的 MLX 路径（`--backend mlx`）可在 Apple Silicon 上本地训练 LoRA 适配器——请参阅 [Apple Silicon (MLX)](#apple-silicon-mlx--experimental-v15)。它仅支持 LoRA-SFT，并且已经在实际的硬件上构建和验证（但尚未完全测试），因此对于 LoRA SFT 之外的任何内容（ORPO、全参数微调、FP8、多轮运行），您仍然需要使用 CUDA 路径。
- **超出测试模型范围的任何内容**——Qwen 2.5 / 3.5（7B / 4B）、Phi-4-mini-3.8B、SmolLM3-3B、Llama 3.2（3B / 1B）、Mistral 7B。其他模型通常可以工作，但未在 CI 中进行固定。

如果您需要上述任何功能，请使用上面列出的库之一。它们在这方面表现更好。

## Backpropagate 提供的功能

四个功能，在一个安装包中：

**1. 真正的 3 行 API，无需配置文件即可运行。**
本文档顶部的代码片段可以端到端地运行。无需 `accelerate config`、YAML 或 Hydra 覆盖。只需 `Trainer(model).train(data)`，您就可以进行微调。

**2. 真正支持 Windows。**
大多数机器学习库都将 Windows 视为次要考虑。Backpropagate 在 Windows + RTX 5080 上进行了第一类测试。该库会处理运行时中的一些问题——它知道如何预先标记您的数据，以便 Windows 多进程不会崩溃，它会自动禁用 RTX 40/50 显卡上的 xformers，因为这会导致错误，并且它会选择不会导致错误的 dataloader 设置。您不必了解所有这些。它只是可以运行。

**3. 专为无人值守运行而设计。**
训练需要几个小时。您不想一直监控它。Backpropagate 旨在让其运行：

- 如果您耗尽了 GPU 内存，它会自动将批处理大小减半并重试——最多三次。无需手动调整。
- 如果您的 GPU 过热，它会暂停，直到温度降下来，然后继续。
- 每个检查点都会以原子方式写入——如果您的笔记本电脑在保存过程中崩溃，则之前的良好检查点仍然完好无损。
- 每次训练运行都会获得一个唯一的 ID，该 ID 会被标记到每条日志行、每个检查点以及每个 Weights & Biases 条目上。如果出现问题，一个 ID 可以让维护者关联所有内容。
- 错误会带有稳定的代码（`RUNTIME_GPU_OOM`、`DEP_OLLAMA_REGISTRATION_FAILED` 等），因此您可以搜索日志和 [故障排除指南](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) 以查找解决方法。特定于 CUDA 的错误具有专门的 [CUDA 故障排除页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)。

**4. 从训练好的适配器到 `ollama run`，只需一个命令。**
许多库都会训练一个模型。很少有库会在您想要实际使用它时让您摆脱麻烦。Backpropagate 导出到 GGUF（Ollama 使用的格式），并使用一个命令注册一个 Ollama 模型。从“训练完成”到“我可以与我的微调模型进行聊天”，大约需要 30 秒。

## 快速入门

该存储库包含一个小型示例数据集，因此本文档顶部的代码片段可以在全新安装的环境中运行：

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

这会在 5 个简短的 ShareGPT 格式对话上训练一个 Qwen 2.5 7B 适配器，然后将结果导出到 GGUF。对于您自己的数据，请将 JSONL 格式化为每行一个示例：

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca（`instruction` / `output`）、OpenAI 聊天（`messages`）和原始文本格式也可以工作——Backpropagate 会自动检测格式。

### 偏好调整（ORPO、SimPO、KTO）

v1.5 版本的新功能：使用偏好而不是纯演示进行训练。ORPO 是无参考且单阶段的——它将偏好信号折叠到 SFT 步骤中，因此没有单独的奖励或参考模型，并且 3 行的结构保持不变。通过 `--method orpo`（CLI）或 `method="orpo"`（Python）传递，并提供一个包含 `{prompt, chosen, rejected}`（或仅 `{chosen, rejected}`）行的数据集：

```jsonl
{"prompt": "What is Python?", "chosen": "A high-level programming language known for readability.", "rejected": "idk look it up"}
{"prompt": "Explain recursion.", "chosen": "A function that calls itself with a smaller input until a base case.", "rejected": "when something repeats"}
```

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct", method="orpo")
trainer.train("preferences.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")
```

```bash
backprop train --data preferences.jsonl --method orpo --steps 100
```

对于 ORPO，默认学习率会自动降低到 `8e-6`（损失函数比普通的 SFT 更陡峭）；通过调整 `--orpo-beta`（默认为 `0.1`）来设置赔率比惩罚的权重。ORPO 仅适用于 `mode="lora"`。

**v1.6 版本的新功能 — SimPO 和 KTO。** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) 不需要参考数据，使用长度归一化的奖励，并采用与 ORPO 相同的配对 `{prompt, chosen, rejected}` 数据（`--simpo-beta`、`--simpo-gamma`）。`--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) 使用**未配对的** `{prompt, completion, label}` 数据——每个示例的“好评/差评”——用于处理大量非人工筛选的 A/B 对反馈；它会自动平衡来自标签计数的理想/不理想损失权重。两者都仅适用于 `mode="lora"`，并且仍然在单个 GPU 的 SFT 范围内（没有单独的参考模型）。请参阅[偏好调整手册](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) 以了解应该使用哪一个。对于在线 RL（PPO/GRPO），请参见[Backpropagate 并非用于](#what-backpropagate-is-not-for)。

### 基于推理轨迹的 SFT（R1 蒸馏）

v1.5 中的新功能：以简单的方式蒸馏一个推理模型。传递 `--reasoning-trace`（命令行界面）或 `Trainer(..., reasoning_trace=True)`（Python），并提供包含 `<think>...</think>` 链式思维的轨迹，这些轨迹位于助手回复中——这是 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 蒸馏的纯 SFT 部分，无需 RL。反向传播会在训练目标中保留 `<think>`，删除空/过长的轨迹（轨迹长度过滤），并将默认 `max_seq_length` 提高到 8192，以适应更长的 CoT。重要的是，`<think>` 保持为**纯文本**——没有特殊的令牌，没有嵌入调整——因此合并后的 GGUF 仍然可以导出到 Ollama，就像任何其他微调模型一样。仅 SFT。请参阅 [reasoning-trace 配方](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation)，了解数据集的形状和可调整的令牌范围。

### Apple Silicon (MLX)——实验性，v1.5

v1.5 中的新功能：**一个 API，两种框架。** CUDA 仍然是经过验证的、标准的后端；MLX 是第二个框架，它通过 Apple 的 [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) 工具链在 M 系列 Mac 上进行训练（统一内存，无需 CUDA）。相同的 3 行代码可以根据硬件选择框架——`backend='auto'`（默认值）在 NVIDIA 上路由到 CUDA，在 Apple Silicon 上路由到 MLX，因此现有的 CUDA 设置在字节级别上是相同的：

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

在 v1.5 中，MLX 框架**仅支持 LoRA SFT**——不支持 ORPO、不支持 FP8、不支持 `mode='full'`，目前也不支持在 MLX 上进行多轮训练（每轮都会出现 `CONFIG_INVALID_SETTING` 错误；如果需要这些功能，请在 NVIDIA 机器上使用 `backend='cuda'`/`'auto'`）。生成的适配器是纯 safetensors 格式，并通过与 CUDA 框架相同的路径导出到 Ollama。

> ⚠️ **当前状态：**v1.5 版本中包含的 MLX 框架已经过构建和单元测试（模拟），但**尚未在真实的 Apple Silicon 上进行实际验证**——`mlx-lm` 仅适用于 Apple 设备，无法在此处编写代码时运行在 NVIDIA 系统上。请将其视为实验性功能——与 v1.5 中 FP8 路径的初始状态相同（FP8 在 v1.6 中已升级到经过实际验证的状态；MLX 仍需要在真实的硬件上进行测试），并且请在它在一系列 M 型 Mac 上运行时[报告异常](#reporting-bugs)。如果在非 Apple 主机上强制使用 `--backend mlx`，则会出错并显示 `CONFIG_INVALID_SETTING`；如果 Mac 上缺少 `mlx_lm` 工具链，则会显示 `DEP_MLX_UNAVAILABLE`。

有关更多端到端的流程（微调并推送到 HF Hub、在 OOM 之后恢复、在长时间的训练过程中进行多轮 SLAO 等），请参阅 [手册配方页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/)。

### Web UI（可选）

如果您更喜欢点击而不是输入 Python 代码，请安装 UI 扩展并启动：

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

一个本地 Web 界面将在 `http://localhost:7862` 上打开，用于浏览数据集、验证格式和以图形方式组装训练配置。训练本身通过 `backprop train` 运行（UI 驱动的训练计划在未来推出——“开始”按钮当前显示该说明）。默认情况下，UI 仅在本地运行。要将其暴露给其他设备，请参阅下面的 [Web UI](#web-ui)，了解 `--share` + `--auth` 安全协议。

## 多轮训练

如果您想在多个数据集上进行增量微调——例如，您每周获得新的训练数据，并且想添加它，而不会忘记之前学到的内容——Backpropagate 的 `multi_run` 模式适合您：

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")

result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
)
```

这将运行五个训练周期，并在周期之间合并适配器，从而在合并新示例的同时保留早期知识。该技术基于最近的持续学习研究——请参阅本 README 文档底部的 [参考文献](#references)。

命令行版本：

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## 从检查点恢复

如果在第 4 轮训练中崩溃，则可以恢复 5 轮训练。每个多轮会话都会将其运行 ID 写入磁盘上的历史记录和检查点清单中，因此您可以执行以下命令来从中断的地方继续：

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

`backprop multi-run` 的默认行为（没有 `--resume`）会自动检测相同输出目录中的进行中条目并继续。要强制从头开始，请指向一个新的输出目录。

## 训练历史记录

每次 `backprop train` 和 `backprop multi-run` 调用都会在 `<output>/run_history.json` 中记录一行——使用的模型、数据集、超参数、状态、最终损失、损失历史记录。您可以列出并检查过去的运行：

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## 实验跟踪

Backpropagate 会自动检测已安装的实验跟踪器（Weights & Biases、TensorBoard、MLflow）并将其连接起来。如果安装了 `wandb` 并且您已登录，则每次运行都会自动记录到 W&B，并且运行名称与磁盘上的运行 ID 匹配——因此您可以使用一个标识符在 W&B、日志和 `run_history.json` 中进行搜索。

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

使用 `Trainer(report_to=["wandb"])`、`Trainer(report_to=["tensorboard"])` 或 `Trainer(report_to="none")` 进行覆盖，以选择退出。

## Web UI

Reflex Web 界面是可选的——使用 `pipx install "backpropagate[ui]"` 安装并启动：

```bash
backprop ui --port 7862
```

UI 在 `http://localhost:7862` 上本地运行。今天，它涵盖了工作流程的**浏览/验证/配置**部分——将其指向一个数据集，检查自动检测到的格式和统计信息，选择一个模型，并组装一个运行配置。**运行的启动是通过 CLI 完成的**（`backprop train` / `backprop multi-run`）；UI 上的“开始”按钮会显示一个说明，指向该位置。UI 驱动的训练是计划中的后续步骤——在此之前，UI 是启动平台，CLI 是触发器。

为了使其能够被其他设备访问（例如，您网络中的其他设备、公共 URL 等），您必须将 `--share`（或 `--host`）与 `--auth` 结合使用：

```bash
backprop ui --share --auth alice:hunter2
```

在不使用 `--auth` 的情况下，`backprop ui --share` 会报错并退出。原因是：`--share` 会发布一个互联网上的任何人都可以访问的 URL，如果没有身份验证，这意味着任何人都可以驱动您的训练流水线并读取您的 HuggingFace 令牌。对于此项，没有可选的禁用设置——如果您不想设置凭据，请改用 SSH 端口转发：

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

有关完整的威胁模型，请参阅 [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/)。

UI 中的文件系统写入操作被限制在一个目录中：

- 默认：`~/.backpropagate/ui-outputs`
- 覆盖：设置 `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- 覆盖设置会进行白名单验证——系统或凭据路径（`/etc`、`~/.ssh`、`~/.aws`、`C:\Windows\System32` 等）将被拒绝。

## 平台说明

**要求：** Python 3.10+ · CUDA GPU（8GB+ VRAM）· PyTorch 2.0+

Python 3.10 至少支持到 v1.6 版本；它将在 2026 年 10 月达到上游生命周期结束，并且计划在之后发布的第一个版本中删除。对于新安装，建议使用 Python 3.11 或 3.12——3.11 是经过最多测试的版本。

Backpropagate 处理不同平台上训练时出现的一些运行时问题，但它无法修复安装时出现的问题。最常见的问题有两个：

- **错误的 CUDA wheel。** PyTorch 为每个 CUDA 版本发布一个二进制文件。如果您选择了错误的版本，您将默默地获得仅使用 CPU 的 PyTorch，并且训练速度将非常慢。使用 <https://pytorch.org/get-started/locally/> 上的 wheel 选择器来选择适合您驱动程序的版本。运行 `nvidia-smi` 以查看您的驱动程序/CUDA 版本。
- **Windows + GGUF 导出。** `[export]` 附加组件会从源代码构建 `llama-cpp-python`，这需要 Visual Studio Build Tools（C++ 组件）和 CMake。

**macOS：** 不支持 CUDA 轨道（没有 CUDA）——使用 CUDA 的 `trainer.train()` 会引发 `DEP_GPU_NOT_AVAILABLE` 错误，您可以通过 Ollama 在 Mac 上运行训练后的适配器。**v1.5 中的新功能：** 一个实验性的 MLX 轨道（`--backend mlx`，`pip install 'backpropagate[mlx]'`）使用 `mlx_lm.lora` 在 Apple Silicon 上本地训练 LoRA 适配器——仅支持 LoRA SFT，并且已构建和单元测试，但尚未在实际硬件上进行验证（请参阅 [Apple Silicon (MLX)](#apple-silicon-mlx--experimental-v15)）。对于 CUDA 路径，或者对于 ORPO / 完全微调 / FP8 / 多次运行，请使用 CUDA Linux 或 Windows 机器。

请参阅 [故障排除手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)，了解完整的安装修复指南，并参阅专门的 [CUDA 故障排除页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)，了解驱动程序/VRAM/xformers/bf16 与 fp16 相关的问题。

## CLI

每个 Python API 都有一个 CLI 镜像：

```bash
backprop train --data my_data.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backprop ui --port 7862
backprop info                          # environment + version snapshot
backprop list-runs                     # past training runs
backprop show-run <run-id>             # detail view
backprop resume <run-id>               # resume a crashed run
backprop push ./output/lora --repo me/my-model    # push adapter to HuggingFace Hub
backprop diff-runs <run-a> <run-b>     # diff two runs side by side
backprop replay <run-id>               # re-run with same config / dataset
backprop export-runs --format jsonl    # bulk export run history
```

完整的参考资料请参见 [CLI 手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/)，或者使用 `backprop <子命令> --help`。

## 配置

可以使用带有 `BACKPROPAGATE_` 前缀的环境变量来覆盖每个设置：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | 自动 | 强制使用 JSON 或控制台日志 |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | 默认模型 |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | 学习率 |
| `BACKPROPAGATE_LORA__R` | `256` | LoRA 秩（v1.3 默认值；如果需要 v1.2.x 的默认值 16，请传递 `--lora-preset=fast`） |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | UI 文件系统沙盒 |

嵌套键使用双下划线（`MODEL__NAME`，而不是 `MODEL_NAME`）。完整的参考资料请参见 [环境变量手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/)。

## 模型预设

| 预设 | VRAM | 许可证 | 说明 |
|---|---|---|---|
| Qwen-3.5-4B | ~8GB | Apache 2.0 | 推荐用于小于 5B 的模型。在该尺寸下，质量最佳。 |
| Phi-4-mini-3.8B | ~8GB | MIT | 在推理/数学/代码方面表现出色。许可证限制严格。 |
| SmolLM3-3B | ~6GB | Apache 2.0 | 完全开放的配方。原生 64K 上下文。 |
| Qwen 2.5 7B | ~12GB | Apache 2.0 | 现有的默认值。这是旧版 7B 预设中质量最好的。 |
| Qwen 2.5 3B | ~8GB | Qwen-Research | ⚠ 研究许可证——在商业用途之前，请查看 Qwen 许可证条款。 |
| Llama 3.2 3B | ~8GB | Llama Community | 是 Qwen 3B 的一个不错的替代方案，但有一些宽松的限制。 |
| Llama 3.2 1B | ~6GB | Llama Community | 用于在小型设备上进行快速实验。 |
| Mistral 7B | ~12GB | Apache 2.0 | 与 Qwen 7B 相当，但使用了不同的聊天模板。 |

其他模型通常也可以工作，但只有这八个模型在 CI 中被固定。传递 `--lora-preset=quality`（默认值），以获得 Biderman 2024 + Thinking Machines 2025 中定义的 256 秩/所有线性目标，或者传递 `--lora-preset=fast`，以获得旧版 16 秩/q+v 目标（如果您需要 v1.2.x 的占用空间）。

## 故障排除

这是最常见首次运行失败的简短索引。完整的反向索引请参见 [故障排除手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)。有关驱动程序/VRAM/混合精度深入分析，请参见 [CUDA 故障排除页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)。

| 症状 | 错误代码 | 解决方法 |
|---|---|---|
| GPU 在训练过程中耗尽内存 | `RUNTIME_GPU_OOM` | 自动模式——反向传播会将批次大小减半，并重试最多 3 次。要禁用此功能：`Trainer(oom_recovery=False)`。要强制使用较小的值：`--batch-size 1`。 |
| HuggingFace 返回 401 /“未找到模型”。 | `DEP_MODEL_LOAD_FAILED` | 运行 `huggingface-cli login` 并重试。如果存在拼写错误，请从 <https://huggingface.co/models> 复制确切的 ID。 |
| `register_with_ollama` 连接被拒绝。 | `DEP_OLLAMA_REGISTRATION_FAILED` | 启动守护进程：`ollama serve`。从 <https://ollama.com> 安装。可重试。 |
| 在保存检查点时，磁盘空间不足。 | `STATE_CHECKPOINT_INVALID` | 原子写入会在崩溃时留下一个 `.partial` 目录——可以安全地删除。之前的有效检查点完好无损。 |
| 训练因 GPU 过热而暂停。 | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | 自动模式——反向传播会在达到温度阈值时暂停，并在 GPU 冷却后恢复。如果问题持续发生，请改善散热。 |
| `backprop ui --share` 被拒绝。 | `RUNTIME_UI_AUTH_NOT_ENFORCED` | 传递 `--auth user:password`，或者使用 SSH 端口转发（请参阅 [Web UI](#web-ui)）。 |
| 首次尝试 GGUF 导出失败。 | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`；在 Windows 上，还需要 Visual C++ 构建工具 + CMake。 |

## 报告错误

当出现故障时，Backpropagate 会在启动时打印一行，例如 `run_started run_id=<uuid>`，并将相同的 ID 绑定到每条日志行、每个检查点以及每个 Weights & Biases 条目。**在任何错误报告中包含 `run_id`**——这可以让维护者关联同一运行的所有内容。

一份好的错误报告包括：

1. **`run_id`**——启动时打印的 UUID。一个 UUID 允许维护者关联同一运行的每条日志行、每个检查点以及每个 Weights & Biases 条目。
2. **错误代码**——stderr 中的 `[CODE_NAME]: message` 行。请参阅 [错误代码](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/)，以获取稳定代码的目录。
3. **已编辑的堆栈跟踪。** 在非详细模式下，stderr 会自动进行编辑（删除 Bearer 令牌、`sk-*`、`hf_*`、AWS 密钥、`password=` / `token=` / `api_key=` 对）。可以安全地粘贴。对于完整的未编辑堆栈跟踪，请使用 `BACKPROPAGATE_DEBUG=1`（或 `--verbose`）重新运行；在发布之前进行审核。
4. **`backprop info` 输出。** 一个命令会打印 Python / PyTorch / CUDA / GPU 模型 / VRAM / OS / 已安装的附加组件——维护者需要的所有内容，以隔离特定平台的回归。

[错误报告模板](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) 明确提示了这些内容，以便快速进行分类。问题、想法或“这是预期的吗？”的讨论应该在 [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions) 中进行。安全问题应通过 [GitHub 安全咨询](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) 表单私下报告——请参阅 [SECURITY.md](SECURITY.md)，了解策略和响应时间。

## 隐私

所有训练都在您的 GPU 上本地进行。Backpropagate 不会进行任何网络请求，除非是从 HuggingFace 下载模型（您会主动发起）。没有遥测，没有云依赖。

## 参考文献

Backpropagate 的默认设置和多运行训练模式是基于最近的研究。如果您对底层技术感兴趣：

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)——介绍 LoRA 的基础论文，Backpropagate 通过这种方式高效地训练适配器。
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)——经验证据表明，在秩为 256 且所有线性目标的情况下，LoRA 在大多数后训练任务中与完全微调的质量相匹配，并且计算量减少了 67%。这驱动了 Backpropagate v1.3 的默认 LoRA 配置。
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora)——实际的后续研究，确定了在高 LoRA 秩下所需的 10 倍学习率与完全微调的校正。
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)——最初的表征，解释了为什么神经网络在对新数据进行微调时会“忘记”之前的训练（EWC——弹性权重巩固）。
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152)——O-LoRA，一种较早的方法，通过将新的适配器限制在正交子空间中，用于语言模型的持续学习。
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708)——一种用于合并多个微调模型而不会产生干扰的基础技术。
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017)——Backpropagate 的多运行合并器实现的具体算法。这是一篇 2025 年 12 月的预印本；Backpropagate 是该论文已知的第一个下游采用者。

## 许可证

MIT——请参阅 [LICENSE](LICENSE)。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
