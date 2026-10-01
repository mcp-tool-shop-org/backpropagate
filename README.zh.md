<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/backpropagate/readme.png" alt="Backpropagate" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/backpropagate/"><img src="https://img.shields.io/pypi/v/backpropagate" alt="PyPI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/backpropagate"><img src="https://img.shields.io/codecov/c/github/mcp-tool-shop-org/backpropagate/main" alt="Coverage"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/backpropagate"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/backpropagate/badge" alt="OpenSSF Scorecard"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

# 在一个 GPU 上微调一个 32B QLoRA 模型，或者微调一个 7B 的端到端模型。然后将模型部署到 Ollama 上

在一个 **单个** GPU 上对大型语言模型进行反向传播微调，GPU 的大小要与你实际拥有的显卡相匹配。只需三行 Python 代码，即可在一个 32GB 的消费级显卡（RTX 5090）上微调一个 7B-32B 模型。使用一个标志 `--full-ft-offload`，可以对一个 7B 级别的模型进行完全微调，同时将其权重和梯度保存在主机 RAM 中（Linux 或 WSL2；速度较慢，具体结果见下文）。再添加一条命令，即可导出到 Ollama，然后 `ollama run` 你的微调模型。可以扩展到 16GB。在 Windows 上表现出色。

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")
```

```bash
backprop export ./output --format gguf --quantization q4_k_m --ollama --ollama-name my-model
ollama run my-model
```

就是这样。没有 YAML 配置文件。没有 `accelerate launch` 繁琐的设置。没有单独的“现在将其转换为 GGUF 格式”教程。如果你有一块 CUDA GPU 和一个包含训练数据的 JSONL 文件，那么你只需三行代码就可以得到一个可用的微调模型。

## 安装

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

如果你需要可选功能，请将安装替换为以下其中之一：

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

喜欢使用 Docker 吗？`docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` 也可以。为 `linux/amd64` 和 `linux/arm64` 提供了镜像，因此 Apple Silicon 和 ARM Linux 用户可以获得原生镜像。一个标准的 `compose.yaml`，用于“在容器中运行 UI”，位于仓库的根目录中——`docker compose up` 在 `http://localhost:7860` 上启动 Web UI，并使用持久的 `~/.backpropagate` 卷挂载。

## Backpropagate 在这个领域中的定位

有几个优秀的库可以用于微调 LLM。它们各自擅长不同的方面：

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**——如果你喜欢 YAML 配置文件，并且想要一个可以从中复制配方的社区。
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**——如果你想要 DPO/PPO/RLHF 和一个 Web GUI。
- **[Unsloth](https://github.com/unslothai/unsloth)**——如果你需要最快的训练速度，并且使用的是受支持的模型系列。
- **[torchtune](https://github.com/pytorch/torchtune)**——如果你想要 Meta 提供的、基于 PyTorch 的原生配方，并且可以对其进行编辑。

Backpropagate 是缺失的选项：**一个 3 行 Python API，适用于在单个消费级 GPU 上运行的独立用户，他们想要训练一个适配器并将其部署。** 没有 YAML，没有 GUI，没有在线 RL（PPO/GRPO），没有多节点。只有每个人真正需要的循环和阻碍部署的导出步骤。

如果你尝试了上述库中的一个，但因为配置文件的繁琐设置、或者遇到了模型系列限制、或者想要默认支持 Windows 的功能——那么 Backpropagate 适合你。

## 你可以在一个 GPU 上微调的内容

Backpropagate 会根据你的显卡调整运行参数。以下是在 2026-09-30 上使用 32GB RTX 5090 显卡进行测量的结果（凭证：[`docs/receipts/2026-09-30-offload/`](docs/receipts/2026-09-30-offload/))。QLoRA 的峰值是在预设的完整上下文窗口下，批处理大小为 1，这是该预设的最坏情况；较短的示例使用的资源更少。

| 模型 | 方法 | 在 32GB 显卡上测得 |
|---|---|---|
| **14B** (Qwen2.5-14B) | QLoRA | **25.0 GiB** 峰值，上下文为 4096（保留 28.1 GiB）。 |
| 24B (Mistral-Small-24B) | QLoRA | 26.5 GiB 峰值，上下文为 4096（保留 29.6 GiB）。 |
| **32B** (Qwen2.5-32B) | QLoRA | **刚好可以运行：** 28.8 GiB 峰值，上下文为 2048（保留 30.7 GiB，大约有 0.65 GiB 的可用空间）。 |
| 3B | `mode="full"`（真正的完全微调，在 GPU 上） | **22.0 GiB** 峰值（系统范围），批处理大小为 4，上下文为 512，速度为 0.30 s/步。其中 7.5 GiB 是分页的优化器状态，可以在较小的显卡上溢出到主机 RAM 中（未经测试）。 |
| **7B 级别** (Qwen2.5-7B，7.6B 参数) | `mode="full" --full-ft-offload` | **训练：** 5.3 GiB VRAM，**30.8 GiB 主机 RAM**（保存时为 32.2 GiB），**14.7 s/步**。仅适用于 Linux 或 WSL2。 |

在本次测试中未重新测量：7B QLoRA、Llama-3.1-8B（私有仓库，测试机器上没有令牌），以及高于 3B 的纯 GPU 完全微调。文档中其他地方的数字是估计值。

Backpropagate 可以实现大多数单 GPU 库无法实现的两件事：**24-32B QLoRA** 和 **单卡 7B 级别完全微调**，并且可以在单个消费级显卡上完成，然后将结果直接导出到 Ollama。

**完全微调有两种方法。** 如果不进行卸载，模型、其梯度和优化器状态都将位于 GPU 上。该库通过检测到的 VRAM 限制模型大小（**16 GB → 4B，24 GB → 5B，32 GB → 6B**）；这些限制来自内存计算，并且仅测量到 3B。使用 `--full-ft-ceiling-billions` 覆盖。

`--full-ft-offload` 将权重和梯度保存在主机 RAM 中，并将其流式传输到 GPU（FSDP2 CPU 卸载）。测得的成本如下：

- **主机 RAM：** 适配检查要求大约 3.7 GiB/十亿参数 + 10 GiB，这是一种保守的估计（在 7.6B 时为 39 GiB，与测得的 32.2 GiB 相比）。如果机器无法容纳，则会在一开始拒绝运行。7.6B 模型无法在 28 GB WSL2 内存限制下运行；大约 5B 是实际限制。
- **速度：** 在 7.6B 时为 14.7 s/步（批处理大小为 1），在 3B 时为 5.1 s/步（批处理大小为 4），而 3B 在 GPU 上的批处理大小为 4 时约为 0.63 s/步。仅在模型无法在没有它的情况下运行时才使用它。计划推出更快的版本。
- **优化器：** Adafactor，而不是 AdamW。权重保持为 bf16，并且每次更新都会使用随机舍入写回；没有 fp32 副本。
- **质量：** 在一次 3B 运行（150 步，保留损失，一个种子）中，它达到了普通完全微调的约 85% 的改进效果（2.45 → 1.93，而普通完全微调为 2.45 → 1.84）。一个种子不是基准。
- **范围：** 纯监督微调。没有打包，没有仅响应掩码，没有中间检查点，没有恢复。仅适用于 Linux 或 WSL2（FSDP2 需要 NCCL）；在 Windows 本机上，它会在 `DEP_FSDP_UNAVAILABLE` 时停止。
- **尚未测试：** 长期运行、高于 1 的梯度累积以及物理 64 GB 机器（测试机器有更多的 RAM，但测试时强制限制为 60 GiB）。

一个不适合的模型会以 `RUNTIME_FULL_FT_MODEL_TOO_LARGE` 退出，并命名退出方式。请参阅[完整的微调手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/)。

### 可扩展至 16 GB

16 GB 规格（RTX 4080 / 5080 / 4070 Ti Super）仍然是顶级配置：7B QLoRA，以及通过 `mode="full"` 对约 3B 模型（SmolLM3-3B、Qwen2.5-3B、Llama-3.2-3B/1B）进行真正的完全微调（在 32 GB 显卡上，3B 模型占用 22.0 GiB，其中 7.5 GiB 是可以溢出到主机 RAM 的分页优化器状态；是否可以在 16 GB 显卡上正常运行尚未测试）。使用 `--full-ft-offload`，GPU 占用的内存要少得多：在测试显卡上，VRAM 被限制，3B 模型在 6 GiB 限制下进行训练，4B 和 7.6B 模型在 8 GiB 限制下进行训练。这些是在 32 GB 显卡上模拟的限制，而不是在真实的 8 GB 硬件上运行。相同的代码会选择适合检测到的显卡的批大小和上限。

2 位量化（AQLM / QuIP#）不在本次范围之内——2 位基础模型无法干净地合并回全精度权重，这会破坏可合并的适配器 → GGUF → Ollama 导出流程（这是整个流水线的目的）。Backpropagate 提供的替代方案——QLoRA、`mode="full"`、`--full-ft-offload` 以及 FP8 计算路径（`--fp8`，Blackwell/Hopper）——都保持可合并和可导出。

## Backpropagate 不适用于以下情况

如果您的使用场景符合以下条件，您最好使用其他库——Backpropagate 并不是最佳选择，并且尝试使其工作可能会比直接使用正确的工具花费更多的时间。在开始之前阅读本部分，可以避免安装和重试的循环。

- **对 13B+ 模型进行完全参数微调**——Backpropagate 在 32 GB GPU 上最多可以对约 6B 模型进行完全微调，并使用 `--full-ft-offload` 对 7B 级别的模型进行微调（请参阅[规格](#what-you-can-fine-tune-on-one-gpu)）。对 13B+ 模型进行完全微调需要多 GPU FSDP 或更大的显卡。在投入计算资源之前，请权衡利弊。 [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/) 报告称，当 LoRA 应用于每一层，并且数据集适合适配器的容量时，LoRA 可以与完全微调相匹配，每次迭代的计算量约为三分之二。 [Biderman et al. 2024](https://arxiv.org/abs/2405.09673) 发现，在标准的低秩设置中，LoRA 在代码和数学方面表现明显低于完全微调，但遗忘的程度更小。对于指令遵循、角色和风格在适度数据集上的工作，通常使用最多 32B 的 QLoRA 效果更好。
- **在线 RL——PPO / GRPO / RLVR**——Backpropagate 执行单阶段 SFT 以及无参考偏好调整（v1.5 中的 ORPO；v1.6 中的 SimPO + KTO）。它不执行在线强化学习——PPO、GRPO 或 RLVR——这需要一个奖励模型或一个在训练步骤之上进行生成和评分的循环。对于这些，请直接使用 TRL 或 LLaMA-Factory。（无参考偏好调整适合单阶段流程，因为不需要在内存中保留单独的参考模型；请参阅[快速入门](#quick-start) 下的 ORPO 注释。）
- **多节点训练**——仅支持单个 GPU 在一台机器上。一台机器上的多 GPU 也可以工作（通过 `accelerate launch`），但未正式支持。
- **在 CUDA 轨道上进行 macOS 训练**——Apple Silicon 没有 CUDA，因此 CUDA 路径在配备 NVIDIA GPU 的 Linux 或 Windows 机器上运行。您仍然可以通过 Ollama 在 Mac 上运行训练后的模型。一个**实验性的、未经验证的预览** MLX 轨道（`--backend mlx`）可以在 Apple Silicon 上本地训练 LoRA 适配器——请参阅[Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview)。它仅支持 LoRA-SFT，并且**未在实际硬件上进行过狗试验证证**（无支持），因此对于 LoRA SFT（ORPO、完全微调、FP8、多轮运行）之外的任何操作，您都应该使用 CUDA 轨道。
- **超出已测试的模型系列**——Qwen 2.5 / 3.5（7B / 4B）、Phi-4-mini-3.8B、SmolLM3-3B、Llama 3.2（3B / 1B）、Mistral 7B。其他模型通常可以工作，但未在 CI 中进行固定。

如果您需要以上任何功能，请使用上面列出的库之一。它们在这方面表现更好。

## Backpropagate 提供的功能

在一个安装中提供四项功能：

**1. 真正的 3 行 API，无需配置文件即可运行。**
本 README 顶部的代码片段可以端到端运行。没有 `accelerate config`，没有 YAML，没有 Hydra 覆盖。只需 `Trainer(model).train(data)`，您就可以进行微调。

**2. 真正适用于 Windows 的解决方案。**
大多数 ML 库都将 Windows 视为事后才考虑的问题。Backpropagate 在 Windows + RTX 5080 上进行了首次测试。该库会为您处理运行时问题——它知道如何预先标记数据，以便 Windows 多进程不会崩溃，它会自动禁用 RTX 40/50 显卡上的 xformers，因为这会导致故障，并且它会选择不会导致错误的 dataloader 设置。您不必了解所有这些。它只是可以运行。

**3. 专为无人值守运行而设计。**
训练需要数小时。您不想一直监控它。Backpropagate 旨在让其运行：

- 如果您耗尽了 GPU 内存，它会自动将批大小减半并重试——最多三次。无需手动调整。
- 如果您的 GPU 过热，它会暂停，直到温度降下来，然后继续。
- 每个检查点都会以原子方式写入——如果您的笔记本电脑在保存过程中崩溃，则之前的良好检查点仍然完好无损。
- 每次训练运行都会获得一个唯一的 ID，该 ID 会被标记到每条日志行、每个检查点以及每个 Weights & Biases 条目上。如果出现问题，一个 ID 可以让维护者关联所有内容。
- 错误带有稳定的代码（`RUNTIME_GPU_OOM`、`DEP_OLLAMA_REGISTRATION_FAILED` 等），因此您可以搜索日志和[故障排除指南](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)，以查找解决方法。特定于 CUDA 的故障有一个专门的[CUDA 故障排除页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)。

**4. 通过训练好的适配器，只需一条命令即可完成 `ollama run`。**
许多库都会训练一个模型。但很少有库在使用模型时不会碍事。Backpropagate 可以将模型导出为 GGUF 格式（Ollama 使用的格式），并通过一条命令注册一个 Ollama 模型。从“训练完成”到“我可以与我的微调模型进行对话”，大约只需 30 秒。

## 快速入门

从命令行开始，使用包含 5 组对话的示例数据集：

```bash
pipx install "backpropagate[standard]"
curl -LO https://raw.githubusercontent.com/mcp-tool-shop-org/backpropagate/main/examples/quickstart.jsonl

backprop train --data quickstart.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 10
backprop generate ./output "What is Python?"      # did it learn anything?
backprop export ./output --format gguf --quantization q4_k_m --ollama --ollama-name my-first-finetune
ollama run my-first-finetune
```

`backprop train` 将适配器写入 `./output`（使用 `--output` 进行更改）。在 Python 中，执行相同的操作：

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")
```

使用虚拟环境，并使用 `pip install "backpropagate[standard]"` 运行 Python API；`pipx` 将 `backprop` 命令安装在其自己的环境中，因此 `import backpropagate` 无法找到它。

**GGUF 导出需要什么。** 导出会将您的适配器合并到基础模型中，并使用 llama.cpp 的转换脚本进行转换。您需要以下任一条件：

- 一个 llama.cpp **源代码**（`git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp`）以及在同一环境中安装的 `pip install sentencepiece protobuf`，或者
- Unsloth，它已经构建了自己的 llama.cpp。

使用 `--ollama`，`q4_k_m` 量化由 `ollama create` 完成，因此无需进行编译。Backpropagate 不会允许 Unsloth 安装系统软件包来为您构建 llama.cpp；如果您需要，请设置 `BACKPROPAGATE_UNSLOTH_AUTO_INSTALL=1`。详情请参见：[export](https://mcp-tool-shop-org.github.io/backpropagate/handbook/export/)。

对于您自己的数据，请将 JSONL 格式化为每行一个示例：

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca（`instruction` / `output`）、OpenAI 对话（`messages`）和纯文本格式也可以使用——Backpropagate 会自动检测格式。

### 循环：检查数据、训练、评估、导出

```bash
backprop data report my_data.jsonl                     # duplicates, length outliers, format problems
backprop data split my_data.jsonl --heldout-ratio 0.1  # a held-out set the model never trains on
backprop train --data my_data.train.jsonl --steps 200 --output ./run-a
backprop eval <run-id> --heldout my_data.heldout.jsonl # held-out loss + sample generations
backprop eval <run-b> --vs <run-a>                     # did the change help?
backprop export ./run-a --format gguf --ollama --ollama-name my-model
```

评估是默认不使用人工评估：保留的损失加上确定性的任务指标（`normalized_exact_match`、`token_f1`、`contains`、`regex`、`pass_rate`）。要使用 LLM 进行评估，请自行运行它，并将其应用于 `backprop generate` 的输出。请参见 [recipes](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/)。

### 偏好调整（ORPO、SimPO、KTO）

使用偏好而不是简单的演示进行训练。ORPO 不需要参考，并且是单阶段的——它将偏好信号合并到 SFT 步骤中，因此没有单独的奖励或参考模型，并且 3 行的形状保持不变。传递 `--method orpo`（CLI）或 `method="orpo"`（Python），并提供一个包含 `{prompt, chosen, rejected}`（或仅 `{chosen, rejected}`）行的数据集：

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

默认学习率会自动降低到 `8e-6`，以用于 ORPO（损失比简单的 SFT 更陡峭）；调整 `--orpo-beta`（默认 `0.1`），以调整优势比惩罚的权重。ORPO 仅适用于 `mode="lora"`。

**v1.6 中的新功能——SimPO 和 KTO。** `--method simpo`（[Meng et al. 2024](https://arxiv.org/abs/2405.14734)）不需要参考，并使用长度归一化的奖励，它使用与 ORPO 相同的配对 `{prompt, chosen, rejected}` 数据（`--simpo-beta`、`--simpo-gamma`）。`--method kto`（[Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)）使用 **非配对** `{prompt, completion, label}` 数据——每个示例的“赞成/反对”——用于大量不是策划的 A/B 对的反馈；它会自动平衡来自标签计数的理想/非理想损失权重。两者都仅适用于 `mode="lora"`，并且保持在单个 GPU SFT 范围内（没有单独的参考模型）。请参见 [preference-tuning handbook](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/)，以了解应该使用哪一个。对于在线 RL（PPO/GRPO），请参见 [What Backpropagate is NOT for](#what-backpropagate-is-not-for)。

### 推理轨迹 SFT（R1 蒸馏）

以简单的方式蒸馏一个推理模型。传递 `--reasoning-trace`（CLI）或 `Trainer(..., reasoning_trace=True)`（Python），并提供包含 `<think>...</think>` 链式思维的轨迹，这些轨迹保留在助手回复中——这是 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 蒸馏的纯 SFT 部分，不需要 RL。Backpropagate 会将 `<think>` 保留在训练目标中，删除空/过长的轨迹（轨迹长度过滤），并将默认 `max_seq_length` 提高到 8192，以适应更长的 CoT。重要的是，`<think>` 仍然是 **纯文本**——没有特殊的令牌，没有嵌入调整——因此合并的 GGUF 仍然可以导出到 Ollama，就像任何其他微调一样。仅 SFT。请参见 [reasoning-trace recipe](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation)，了解数据集的形状和可调整的令牌范围。

### Apple Silicon (MLX)——未经验证的预览版

> ⚠️ **未经验证的预览版——不属于受支持的功能集。** MLX 框架已经构建并经过单元测试，但尚未在真实的 Apple Silicon 上进行“实际测试”（`mlx-lm` 仅适用于 Apple，无法在 Backpropagate 开发的 NVIDIA 机器上运行）。请将以下所有内容视为实验性的，并自行承担风险，如果您在 M 系列 Mac 上运行它，请 [报告异常](#reporting-bugs)。

**一个 API，两个框架。** CUDA 是经过验证的规范后端；MLX 是第二个框架，它通过 Apple 的 [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) 工具链在 M 系列 Mac 上进行训练（统一内存，没有 CUDA）。3 行的形状会根据硬件选择框架——`backend='auto'`（默认）会将 NVIDIA 上的请求路由到 CUDA，并将 Apple Silicon 上的请求路由到 MLX，因此现有的 CUDA 机器是字节级的：

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

MLX 框架 **仅支持 LoRA SFT**——没有 ORPO，没有 FP8，没有 `mode='full'`，没有多轮运行（每个请求都会被拒绝，并显示 `CONFIG_INVALID_SETTING`；如果需要，请在 NVIDIA 机器上使用 `backend='cuda'`/`'auto'`）。生成的适配器是纯 safetensors，并且通过与 CUDA 框架相同的路径导出到 Ollama。

> 在非 Apple 主机上强制使用 `--backend mlx` 会导致错误 `CONFIG_INVALID_SETTING`；在 Mac 上缺少 `mlx_lm` 工具链会引发 `DEP_MLX_UNAVAILABLE`。

有关更多端到端的流程（微调并推送到 HF Hub、在 OOM 之后恢复、在长时间的活动中进行多轮 SLAO 等），请参见 [handbook recipes page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/)。

### Web UI（可选）

如果您更喜欢点击而不是键入 Python，请安装 UI 扩展并启动：

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

将打开一个本地 Web 界面，网址为 `http://localhost:7862`，用于浏览数据集、验证格式以及以可视化的方式组装训练配置。训练本身通过 `backprop train` 运行（UI 驱动的训练也在计划中——“开始”按钮当前会显示该提示）。默认情况下，UI 仅在本地运行。要将其暴露给其他设备，请参阅下面的 [Web UI](#web-ui)，了解 `--share` + `--auth` 安全协议。

## 多轮训练

如果您想在多个数据集上进行增量微调——例如，您每周获得新的训练数据，并且希望在不忘记之前所学内容的情况下添加它——Backpropagate 的 `multi_run` 模式非常适合您：

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

它运行五次训练，并在每次训练之间合并适配器，从而在整合新示例的同时保留早期知识。该技术基于最近的持续学习研究——请参阅本 README 底部 [参考文献](#references)。

CLI 版本：

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## 从检查点恢复

在第四次运行中崩溃的 5 轮训练是可以恢复的。每次多轮会话都会将其运行 ID 写入磁盘上的历史记录和检查点清单中，因此从上次中断的地方继续只需一条命令：

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

`backprop multi-run`（没有 `--resume`）的默认行为会自动检测同一输出目录中的正在进行的条目并继续执行。要强制从头开始，请指向一个新的输出目录。

## 训练历史记录

每次 `backprop train` 和 `backprop multi-run` 调用都会在 `<output>/run_history.json` 中记录一行——使用的模型、数据集、超参数、状态、最终损失、损失历史记录。您可以列出并检查过去的运行：

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## 实验跟踪

Backpropagate 会自动检测已安装的实验跟踪器（Weights & Biases、TensorBoard、MLflow）并将其连接起来。如果安装了 `wandb` 并且您已登录，则每次运行都会自动记录到 W&B，并且运行名称与磁盘上的运行 ID 匹配——因此您可以使用一个标识符在 W&B、您的日志和 `run_history.json` 中进行搜索。

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

使用 `Trainer(report_to=["wandb"])`、`Trainer(report_to=["tensorboard"])` 或 `Trainer(report_to="none")` 进行覆盖以选择退出。

## Web UI

Reflex Web 界面是可选的——使用 `pipx install "backpropagate[ui]"` 安装并启动：

```bash
backprop ui --port 7862
```

UI 在 `http://localhost:7862` 上本地运行。今天，它涵盖了工作流程的 **浏览/验证/配置** 部分——将其指向一个数据集，检查自动检测到的格式和统计信息，选择一个模型，然后组装一个运行配置。**运行的启动是从 CLI 完成的**（`backprop train` / `backprop multi-run`）；UI 中的“开始”按钮会显示一个指向该位置的提示。UI 驱动的训练是一个计划中的后续步骤——在此之前，UI 是入口，CLI 是触发器。

要将其暴露给其他设备（您网络中的其他人、公共 URL 等），您必须将 `--share`（或 `--host`）与 `--auth` 配对：

```bash
backprop ui --share --auth alice:hunter2
```

没有 `--auth` 的 `backprop ui --share` 会出错并退出。原因是：`--share` 发布一个互联网上的任何人都可以访问的 URL，如果没有身份验证，这意味着任何人都可以驱动您的训练流水线并读取您的 HuggingFace 令牌。没有为此提供选择退出的选项——如果您不想设置凭据，请改用 SSH 端口转发：

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

有关完整的威胁模型，请参阅 [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/)。

UI 中的文件系统写入操作被限制在一个目录中：

- 默认值：`~/.backpropagate/ui-outputs`
- 覆盖：设置 `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- 覆盖内容经过白名单验证——系统或凭据路径（`/etc`、`~/.ssh`、`~/.aws`、`C:\Windows\System32` 等）将被拒绝。

## 平台说明

**要求：** Python 3.10+ · CUDA GPU（8GB+ VRAM）· PyTorch 2.0+

Python 3.10 至少支持到 v1.6；它将于 2026 年 10 月达到上游生命周期结束，并且计划在之后的第一个版本中删除。对于新安装，请首选 Python 3.11 或 3.12——3.11 是经过最多测试的最低版本。

Backpropagate 处理在不同平台上进行训练时的运行时问题，但无法修复安装时出现的问题。最常见的是：

- **错误的 CUDA wheel。** PyTorch 为每个 CUDA 版本发布一个二进制文件。如果您选择了错误的 wheel，您将默默地获得仅 CPU 的 PyTorch，并且训练速度会慢得令人难以置信。使用 <https://pytorch.org/get-started/locally/> 上的 wheel 选择器来选择您的驱动程序。运行 `nvidia-smi` 以查看您的驱动程序/CUDA 版本。
- **Windows + GGUF 导出。** `[export]` 附加构建会从源代码构建 `llama-cpp-python`，这需要 Visual Studio Build Tools（C++ 组件）和 CMake。

**macOS：** 不支持 CUDA 轨道（没有 CUDA）——CUDA 路由的 `trainer.train()` 会引发 `DEP_GPU_NOT_AVAILABLE`，并且您可以通过 Ollama 在 Mac 上运行训练后的适配器。一个**实验性的、未经验证的预览** MLX 轨道（`--backend mlx`、`pip install 'backpropagate[mlx]'`）通过 `mlx_lm.lora` 在 Apple Silicon 上本地训练 LoRA 适配器——仅限 LoRA SFT，并且**未在实际的硬件上进行测试**（请参阅 [Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview)）。对于 CUDA 路径，或者对于 ORPO / 完整微调 / FP8 / 多轮训练，请使用 CUDA Linux 或 Windows 机器。

有关完整的安装故障排除指南，请参阅 [troubleshooting handbook page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)，并参阅专门的 [CUDA troubleshooting page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)，了解驱动程序/VRAM/xformers/bf16 与 fp16 相关的问题。

## CLI

每个 Python API 都有一个 CLI 镜像：

```bash
backprop train --data my_data.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output --format gguf --quantization q4_k_m --ollama --ollama-name my-model
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

完整的参考资料请参见 [CLI handbook page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/)，或 `backprop <subcommand> --help`。

## 配置

可以使用 `BACKPROPAGATE_` 前缀覆盖每个设置中的环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | 自动 | 强制使用 JSON 或控制台日志 |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | 默认模型 |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | 学习率 |
| `BACKPROPAGATE_LORA__R` | `256` | LoRA 秩（v1.3 默认；传递 `--lora-preset=fast` 以使用 v1.2.x 的默认值 16） |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | UI 文件系统沙盒 |

嵌套键使用双下划线（`MODEL__NAME`，而不是 `MODEL_NAME`）。完整的参考资料请参见[环境变量手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/)。

## 模型预设

| 预设 | 显存 (VRAM) | 许可证 | 说明 |
|---|---|---|---|
| Qwen-3.5-4B | ~8GB | Apache 2.0 | 推荐用于小于 5B 的模型。在该尺寸下，效果最佳。 |
| Phi-4-mini-3.8B | ~8GB | MIT | 在推理/数学/代码方面表现出色。许可证限制严格。 |
| SmolLM3-3B | ~6GB | Apache 2.0 | 完全开放的配方。原生 64K 上下文。 |
| Qwen 2.5 7B | ~12GB | Apache 2.0 | 现有默认值。旧版 7B 预设中效果最佳。 |
| Qwen 2.5 3B | ~8GB | Qwen-Research | ⚠ 研究许可证——在商业用途之前，请查看 Qwen 许可证条款。 |
| Llama 3.2 3B | ~8GB | Llama Community | 相对于 Qwen 3B 而言，是一个不错的替代方案，但有一些限制。 |
| Llama 3.2 1B | ~6GB | Llama Community | 用于在小型显卡上进行快速实验。 |
| Mistral 7B | ~12GB | Apache 2.0 | 与 Qwen 7B 相当，但使用了不同的聊天模板。 |
| Llama-3.1-8B | ~7-8GB（QLoRA） | Llama-3.1-Community | 8B QLoRA，128K 原生上下文（>700M-MAU 条款需要单独的 Meta 许可证）。 |
| **Qwen2.5-14B** | 在 4096 上下文中，峰值显存占用为 25 GiB（QLoRA）。 | Apache 2.0 | **适用于 32GB 显卡的日常模型。** 秩/alpha 32，8 位 AdamW。仅 4 位权重大约为 8.5 GB；完整的 4096 个令牌窗口需要剩余的显存。 |
| Mistral-Small-24B | 在 4096 上下文中，峰值显存占用为 26.5 GiB（QLoRA）。 | Apache 2.0 | 24B QLoRA，适用于 32GB 显卡。仅 4 位权重大约为 18 GB。 |
| **Qwen2.5-32B** | 在 2048 上下文中，峰值显存占用为 28.8 GiB（QLoRA）。 | Apache 2.0 | **32GB 显卡的上限。** 仅在 `max_len 2048` 中，使用 8 位 AdamW 即可运行。 |

其他模型通常也可以工作；上述行是精选的预设——14B-32B 级别针对 32GB 显卡进行了 QLoRA 调整（测量的范围）。传递 `--lora-preset=quality`（默认值）以使用 Biderman 2024 + Thinking Machines 2025 的秩-256 / 所有线性目标，或者传递 `--lora-preset=fast` 以使用旧版秩-16 / q+v 目标（如果您需要 v1.2.x 的显存占用）。

## 故障排除

这是最常见的首次运行失败的简短索引。完整的反向索引请参见[故障排除手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)。有关驱动程序/显存/混合精度深入分析，请参见[CUDA 故障排除页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)。

| 症状 | 错误代码 | 解决方法 |
|---|---|---|
| GPU 在训练过程中耗尽内存 | `RUNTIME_GPU_OOM` | 自动——Backpropagate 会将批大小减半，并最多重试 3 次。要选择退出：`Trainer(oom_recovery=False)`。要强制使用更小的值：`--batch-size 1`。 |
| HuggingFace 返回 401 / “模型未找到” | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` 并重试。对于拼写错误，请从 <https://huggingface.co/models> 复制确切的 ID。 |
| `register_with_ollama` 连接被拒绝 | `DEP_OLLAMA_REGISTRATION_FAILED` | 启动守护进程：`ollama serve`。从 <https://ollama.com> 安装。可重试。 |
| 在检查点保存期间，磁盘已满 | `STATE_CHECKPOINT_INVALID` | 原子写入会在崩溃时留下一个 `.partial` 目录——可以安全地删除。之前的良好检查点完好无损。 |
| 训练因 GPU 过热而暂停 | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | 自动——Backpropagate 在达到温度阈值时暂停，并在 GPU 冷却后恢复。如果持续发生，请改善散热。 |
| `backprop ui --share` 被拒绝 | `RUNTIME_UI_AUTH_NOT_ENFORCED` | 传递 `--auth user:password`，或者使用 SSH 端口转发（请参见[Web UI](#web-ui)）。 |
| 首次尝试 GGUF 导出失败 | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`；在 Windows 上，还需要 Visual C++ 构建工具 + CMake。 |

## 报告错误

当出现故障时，Backpropagate 会在启动时打印一行，例如 `run_started run_id=<uuid>`，并将相同的 ID 绑定到每条日志行、每个检查点以及每个 Weights & Biases 条目。**在任何错误报告中包含 `run_id`**——这可以让维护者关联同一运行的所有内容。

一份好的错误报告包括：

1. **`run_id`**——启动时打印的 UUID。一个 UUID 允许维护者关联同一运行的每条日志行、每个检查点以及每个 Weights & Biases 条目。
2. **错误代码**——stderr 中的 `[CODE_NAME]: message` 行。请参见[错误代码](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/)，以获取稳定代码的目录。
3. **已编辑的堆栈跟踪。** 在非详细模式下，stderr 会自动进行编辑（Bearer 令牌、`sk-*`、`hf_*`、AWS 密钥、`password=` / `token=` / `api_key=` 对将被删除——可以安全地粘贴）。对于完整的未编辑堆栈跟踪，请使用 `BACKPROPAGATE_DEBUG=1`（或 `--verbose`）重新运行；在发布之前进行审核。
4. **`backprop info` 输出。** 一个命令会打印 Python / PyTorch / CUDA / GPU 模型 / 显存 / 操作系统 / 已安装的附加组件——维护者需要的所有内容，以对特定于平台的回归进行分析。

[错误报告模板](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) 明确提示了所有这些内容，以便快速进行分类。问题、想法或“这是预期的吗？”的讨论应该在[GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions) 中进行。安全问题应通过[GitHub 安全咨询](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) 表单私下报告——请参见[SECURITY.md](SECURITY.md)，了解策略和响应时间表。

## 隐私

所有训练都在您的 GPU 上本地进行。Backpropagate 不会发起任何网络请求，除了从 HuggingFace 下载模型（您需要手动启动）。不收集遥测数据，不依赖云服务。

## 参考文献

Backpropagate 的默认设置和多轮训练模式都是基于最新的研究成果。如果您对底层技术感兴趣：

- **Hu 等人，2021。**《LoRA：大型语言模型的低秩适配》。[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)——这是介绍 LoRA 的基础论文，Backpropagate 通过 LoRA 高效地训练适配器。
- **Biderman 等人，2024。**《LoRA 学习更少，遗忘更少》。[arXiv:2405.09673](https://arxiv.org/abs/2405.09673)——实证证据表明，在秩为 256 且所有目标均为线性的情况下，LoRA 在大多数后训练任务上的性能与完全微调的质量相当，且计算量仅为后者的 67%。这推动了 Backpropagate v1.3 默认 LoRA 配置的采用。
- **Thinking Machines，2025。**《无需遗憾的 LoRA》。[thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/)——这是一篇实践性文章，它确定了在高 LoRA 秩下所需的 10 倍学习率与完全微调的修正。
- **Kirkpatrick 等人，2017。**《克服神经网络中的灾难性遗忘》。[arXiv:1612.00796](https://arxiv.org/abs/1612.00796)——这是对神经网络在对新数据进行微调时“遗忘”早期训练的原因的最初描述（EWC——弹性权重合并）。
- **Wang 等人，2023。**《用于语言模型持续学习的正交子空间学习》。[arXiv:2310.14152](https://arxiv.org/abs/2310.14152)——O-LoRA，这是一种较早的方法，它通过将新的适配器限制在正交子空间中，从而使用 LoRA 进行持续学习。
- **Yadav 等人，2023。**《模型合并时解决干扰：TIES-合并》。[arXiv:2306.01708](https://arxiv.org/abs/2306.01708)——这是一种用于合并多个微调模型而不会产生干扰的基础技术。
- **Qiao 和 Mahdavi，2025。**《合并后再遗忘：通过持续合并实现单 LoRA 持续学习》。[arXiv:2512.23017](https://arxiv.org/abs/2512.23017)——这是 Backpropagate 的多轮合并器所实现的特定算法。这是一篇 2025 年 12 月的预印本；Backpropagate 是该论文已知的第一个下游采用者。

## 许可证

MIT——请参阅 [LICENSE](LICENSE)。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
