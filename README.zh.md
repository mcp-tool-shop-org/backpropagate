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

# 训练一个适配器。将其发送到 Ollama。完成

Backpropagate 是一个 Python 库，用于在单个 GPU 上微调大型语言模型。只需三行代码，即可在 16GB 的显卡上训练一个 7B 模型。再执行一个命令，即可将其导出到 Ollama，然后您就可以使用 `ollama run` 命令进行微调。它在 Windows 上的兼容性良好。

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

就这样。没有 YAML 配置文件。没有 `accelerate launch` 命令。也没有单独的“如何将其转换为 GGUF 格式”的教程。如果您有一个 CUDA GPU 并且有一个包含训练数据的 JSONL 文件，您只需三行代码就可以完成微调。

## 安装

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

如果您想要可选功能，请将安装替换为以下选项之一：

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

更喜欢 Docker？`docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` 也可以。镜像支持 `linux/amd64` 和 `linux/arm64`，因此 Apple Silicon 和 ARM Linux 用户可以获得原生镜像。一个标准的 `compose.yaml` 文件，用于“在容器中运行 UI”，位于仓库根目录，使用 `docker compose up` 命令可以启动 Web UI，地址为 `http://localhost:7860`，并且会挂载一个持久化的 `~/.backpropagate` 卷。

## Backpropagate 的定位

有很多优秀的库可用于微调大型语言模型。每个库在不同的方面都非常出色：

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — 如果您喜欢 YAML 配置文件，并且想要参考一些配方。
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — 如果您想要一个 Web GUI，并且内置了对 DPO/PPO/RLHF 的支持。
- **[Unsloth](https://github.com/unslothai/unsloth)** — 如果您需要尽可能快的训练速度，并且使用的是受支持的模型系列。
- **[torchtune](https://github.com/pytorch/torchtune)** — 如果您想要 Meta 提供的原生 PyTorch 配方，可以进行编辑。

Backpropagate 是一个缺失的选项：**一个用于单个消费级 GPU 的独立操作员的 3 行 Python API，用于训练适配器并将其部署。** 没有 YAML，没有 GUI，没有 DPO/PPO，没有多节点。 只有每个人真正需要的循环，以及一个会妨碍您的导出步骤。

如果您尝试了上述任何一个库，但因为配置文件而感到困扰，或者遇到了模型系列的限制，或者想要 Windows 优先的默认设置，那么 Backpropagate 就是为您准备的。

## 您可以在 16GB 的消费级 GPU 上微调的内容

以下是在 16GB 显卡（RTX 4080 / 5080 / 4070 Ti Super）上的实际限制：

| 模型 | 方法 | 状态 |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | 良好。完整的序列长度，还有剩余空间。 |
| Phi-4-mini-3.8B / Qwen-3.5-4B / SmolLM3-3B (参数上限≤3B) | `mode="full"` (全量微调) | v1.4 — 在 `backprop train` 命令中使用 `--mode=full` 参数，或者在 `Trainer` 类中设置 `mode="full"`。梯度检查点 + Paged 8-bit Adam 算法可以使激活内存保持在 sqrt(L) 的水平。 |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | 标准。约 7-8 GB。Backpropagate 的默认设置。 |
| Llama-3 13B | QLoRA + 采样压缩 | 勉强可用。使用较短的序列。 |
| Mixtral 8x7B (总参数 47B) | AQLM 2-bit + LoRA | 计划在 v1.5 版本中实现 — 详情请参考 V1_5_BRIEF（发布后查看）。 |

AQLM 2-bit 量化 (`quant_method="aqlm"`) 是一种实验性功能，适用于 16GB 内存上的 Mixtral-8x7B 模型。该功能最初计划在 v1.4 版本中实现，但现在计划在 v1.5 版本中实现。`aqlm` 库已经比较成熟；在 v1.4 版本的开发中，我们优先支持 ≤3B 模型的全量微调 (`mode="full"`)，而不是添加新的量化后端。请在 V1_5_BRIEF 发布后查看 v1.5 的具体实现计划。

对于 3B 及以下的模型，全量微调（而不仅仅是 LoRA）可以在 16GB 内存上实现，并且现在已在 v1.4 版本中以 `mode="full"` 的形式提供。使用 `Trainer(..., mode="full")` 或 `backprop train --mode=full --model phi-4-mini-3.8b` 命令启用此功能。对于大于 3B 的模型，系统会强制阻止使用此模式，并会显示 `RUNTIME_FULL_FT_MODEL_TOO_LARGE` 错误，同时建议使用 LoRA 以及其他适用于小于 3B 的预设模型作为替代方案。请参考 [完整的微调指南页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/)，了解配置细节以及 Biderman 2024 / Thinking Machines 2025 的质量比较。对于 7B 及以上的模型，全量微调需要 24GB 或更大的 GPU 内存 — 考虑租用 A100 云服务器，或者继续使用 LoRA，因为最近的研究表明，在大多数后续训练任务中，LoRA 的质量可以与全量微调相媲美（请参阅 [“不适用场景”部分](#what-backpropagate-is-not-for) 中的引用）。

## Backpropagate 不适合的场景

如果您的使用场景符合以下情况，您可能更适合使用其他库 — Backpropagate 并非最佳选择，尝试使其工作可能会花费更多，不如直接选择合适的工具。在开始之前阅读此部分可以避免安装和卸载的循环：

- **7B+ 模型的全参数微调** — Backpropagate 使用 LoRA / QLoRA，它训练一个小的适配器，而不是更新每个权重。对于 7B 及更大的模型，全量微调需要 24GB 或更多的 GPU 内存，无法在 16GB 的消费级显卡上运行。对于 3B 及以下的模型，全量微调可以在 16GB 内存上运行，并且已在 v1.4 版本中以 `mode="full"` 的形式提供（使用 `Trainer(..., mode="full")` 或 `--mode=full` 命令；对于大于 3B 的模型，系统会强制阻止此模式，并会显示 `RUNTIME_FULL_FT_MODEL_TOO_LARGE` 错误，同时建议使用 LoRA 以及其他适用于小于 3B 的预设模型作为替代方案）。总的来说：最近的研究 ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) 表明，在正确的配置下，LoRA 可以在大多数后续训练任务（指令遵循、领域适应、角色/风格）中达到与全量微调相当的质量，并且只需要 67% 的计算资源 — 因此，对于大多数用户来说，坚持使用 LoRA 并不会带来任何损失。`mode="full"` 适用于那些已经测量到质量差距，并且愿意为此付出额外计算成本的情况。如果确实需要对 7B+ 的模型进行全量微调，请直接使用 HuggingFace `transformers.Trainer`，并在 24GB 或更大的显卡上运行。
- **DPO / PPO / GRPO / 偏好学习** — Backpropagate 仅支持单阶段的监督微调。对于偏好学习，请直接使用 TRL 或 LLaMA-Factory。
- **多节点训练** — 仅支持单个 GPU，且只能在一台机器上运行。支持在一台机器上使用多个 GPU（通过 `accelerate launch`），但未正式支持。
- **macOS 训练** — Apple Silicon 芯片不支持 CUDA，因此训练需要在运行 Linux 或 Windows 的机器上，并使用 NVIDIA GPU。您仍然可以在 Mac 上通过 Ollama 运行训练好的模型。
- **任何不在测试模型系列中的模型** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B。其他模型通常可以工作，但未在 CI 测试中进行验证。

如果您需要这些功能，请使用上面列出的库。它们在这方面做得更好。

## Backpropagate 提供的功能：

四个功能，一次安装：

**1. 一个真正的 3 行 API，无需配置文件即可运行。**
README 顶部的代码片段可以实现端到端的运行。无需 `accelerate config`，无需 YAML，无需 Hydra 覆盖。只需 `Trainer(model).train(data)`，即可完成微调。

**2. 真正适用于 Windows 的解决方案。**
大多数机器学习库对 Windows 的支持往往不够完善。Backpropagate 在 Windows + RTX 5080 上经过了全面测试。该库处理了 Windows 运行时的各种问题——它知道如何预处理您的数据，以防止 Windows 多进程崩溃；它会自动禁用 RTX 40/50 显卡上的 xformers，因为这会导致问题；它还会选择不导致崩溃的数据加载器设置。您无需了解这些细节，它只需正常运行即可。

**3. 专为无人值守运行而设计。**
训练需要几个小时。您不想一直盯着它。Backpropagate 旨在可以长时间运行：

- 如果 GPU 内存不足，它会自动将批次大小减半并重试——最多重试三次。无需手动调整。
- 如果 GPU 过热，它会暂停直到温度降下来，然后继续。
- 每个检查点都会以原子方式写入——如果您的笔记本电脑在保存过程中崩溃，之前的有效检查点仍然可用。
- 每次训练都会生成一个唯一的 ID，该 ID 会附加到每条日志行、每个检查点和每个 Weights & Biases 条目上。如果出现问题，一个 ID 可以让维护人员将所有内容关联起来。
- 错误会显示稳定的代码（例如 `RUNTIME_GPU_OOM`、`DEP_OLLAMA_REGISTRATION_FAILED` 等），以便您可以搜索日志文件，并参考[故障排除指南](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) 查找解决方案。CUDA 相关的错误有专门的[CUDA 故障排除页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)。

**4. 一条命令，从训练好的适配器到 `ollama run`。**
许多库可以训练一个模型。但很少有库会在您想要实际使用它时，让您轻松操作。Backpropagate 可以将模型导出为 GGUF 格式（Ollama 使用的格式），并使用一条命令注册 Ollama 模型。您可以从“训练完成”到“我可以与我的微调模型进行对话”，只需大约 30 秒。

## 快速开始

这个仓库包含一个微小的示例数据集，因此 README 文件的开头部分的代码可以在干净的环境中运行。

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

这会使用 5 个短的 ShareGPT 格式的对话来训练一个 Qwen 2.5 7B 的适配器，然后将结果导出为 GGUF 格式。对于您自己的数据，请将 JSONL 文件格式化为每行一个示例。

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca（`instruction` / `output`）、OpenAI 聊天（`messages`）以及原始文本格式也适用——Backpropagate 会自动检测格式。

有关更多端到端的工作流程（例如，微调并推送到 Hugging Face Hub，在出现 OOM 错误后恢复，在较长的训练周期中进行多次迭代训练等），请参阅[手册配方页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/)。

### Web UI（可选）

如果您更喜欢点击而不是输入 Python 代码，请安装 UI 扩展，然后启动：

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

一个本地 Web 界面将在 `http://localhost:7862` 上打开，您可以在其中选择数据集、选择模型、进行训练和导出。默认情况下，UI 仅在本地可用。要将其暴露给其他设备，请参阅下面的“[Web UI](#web-ui)”部分，了解 `--share` + `--auth` 安全机制。

## 多次迭代训练

如果您希望在多个数据集上逐步进行微调——例如，您每周都会获得新的训练数据，并且希望在不忘记之前学到的知识的情况下将其添加到模型中——那么 Backpropagate 的 `multi_run` 模式非常适合您：

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

这会进行五次训练迭代，并在每次迭代之间合并适配器，从而在保留先前知识的同时，整合新的示例。该技术基于最近的持续学习研究，请参阅此 README 文件的底部部分的“[参考文献](#references)”。

命令行版本：

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## 从检查点恢复

如果在第四次迭代时崩溃的五次迭代训练可以恢复。 每次多迭代会话都会将其迭代 ID 写入磁盘上的历史记录和检查点清单，因此您可以从上次中断的地方继续，只需一个命令即可：

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

`backprop multi-run` 的默认行为（不带 `--resume`）会自动检测到同一输出目录中的正在进行的会话，并继续执行。 要强制从头开始，请指定一个全新的输出目录。

## 培训经历

每次 `backprop train` 和 `backprop multi-run` 的执行都会在 `<output>/run_history.json` 中记录一行，包括使用的模型、数据集、超参数、状态、最终损失和损失历史记录。 您可以列出和检查过去的训练过程：

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## 实验跟踪

Backpropagate 会自动检测已安装的实验跟踪器（Weights & Biases、TensorBoard、MLflow），并将其集成。 如果安装了 `wandb` 并且您已登录，则每次运行都会自动记录到 W&B，并且运行名称与磁盘上的运行 ID 匹配，这样您可以使用一个标识符在 W&B、您的日志和 `run_history.json` 文件中进行搜索。

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

您可以使用 `Trainer(report_to=["wandb"])`、`Trainer(report_to=["tensorboard"])` 或 `Trainer(report_to="none")` 来禁用此功能。

## 网页用户界面

Reflex Web 界面是可选的，请使用 `pipx install "backpropagate[ui]"` 安装，然后启动：

```bash
backprop ui --port 7862
```

UI 在本地运行在 `http://localhost:7862` 上。 要将其暴露给其他设备（例如，您网络中的其他用户或公共 URL），您必须将 `--share`（或 `--host`）与 `--auth` 结合使用：

```bash
backprop ui --share --auth alice:hunter2
```

如果 `backprop ui --share` 不带 `--auth`，则会显示错误。 原因：`--share` 会发布一个 URL，任何互联网用户都可以访问，而没有身份验证，这意味着任何人都可能控制您的流水线并读取您的 Hugging Face 令牌。 没有禁用此功能的选项——如果您不想设置凭据，请使用 SSH 端口转发：

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

请参阅 [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/)，了解完整的威胁模型。

用户界面中的文件写入操作会被限制在一个单独的目录中，以提高安全性。

- 默认值：`~/.backpropagate/ui-outputs`
- 覆盖：设置 `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- 覆盖设置会进行白名单验证，系统路径或凭据路径（如 `/etc`、`~/.ssh`、`~/.aws`、`C:\Windows\System32` 等）将被拒绝。

## 平台说明

**要求：** Python 3.10+ · CUDA GPU (8GB+ VRAM) · PyTorch 2.0+

Python 3.10将于2026年10月结束官方支持，Backpropagate计划在v1.4版本中移除对Python 3.10的支持。对于新安装，建议使用Python 3.11或3.12，其中3.11是经过最广泛测试的版本。

Backpropagate可以处理不同平台上的训练过程中的一些问题，但它无法解决安装过程中的问题。最常见的问题有两种：

- **错误的CUDA驱动程序。** PyTorch为每个CUDA版本发布一个二进制文件。如果选择了错误的驱动程序，您将只能使用CPU版本的PyTorch，并且训练速度会非常慢。请使用<https://pytorch.org/get-started/locally/> 上的驱动程序选择器，选择适合您驱动程序的版本。运行 `nvidia-smi` 命令可以查看您的驱动程序/CUDA版本。
- **Windows + GGUF导出。** `[export]` 扩展会从源代码构建 `llama-cpp-python`，这需要Visual Studio Build Tools（C++组件）和CMake。

**macOS：** 不支持GPU训练（没有CUDA）。您可以在Mac上通过Ollama运行训练好的适配器，但 `trainer.train()` 会抛出 `DEP_GPU_NOT_AVAILABLE` 错误。对于实际的训练，请使用支持CUDA的Linux或Windows机器。

请参阅[故障排除手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)，获取详细的安装问题解决方案，以及专门的[CUDA故障排除页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)，用于解决驱动程序/显存/xformers/bf16与fp16相关的问题。

## CLI

每个Python API都有一个命令行界面（CLI）的对应版本：

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

完整的参考信息请参见[命令行界面手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/)，或者使用 `backprop <子命令> --help` 命令。

## 配置

所有设置都可以通过使用 `BACKPROPAGATE_` 前缀的环境变量进行覆盖：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | 自动 | 强制JSON或控制台日志输出 |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | 默认模型 |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | 学习率 |
| `BACKPROPAGATE_LORA__R` | `256` | LoRA秩（v1.3的默认值；使用 `--lora-preset=fast` 参数可以设置成v1.2.x的默认值16） |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | UI文件系统沙箱 |

嵌套键使用双下划线（`MODEL__NAME`，而不是 `MODEL_NAME`）。完整的参考信息请参见[环境变量手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/)。

## 模型预设

| 预设。 | 显存 (xiǎn cún) | 许可证 | 说明 |
|---|---|---|---|
| Qwen-3.5-4B | 约8GB。 | Apache 2.0 | 适用于小于5B的模型，在该尺寸下质量最佳。 |
| Phi-4-mini-3.8B | 约8GB。 | MIT | 在推理/数学/代码方面表现出色。许可证清洁。 |
| SmolLM3-3B | 约6GB。 | Apache 2.0 | 完全开放的配置。原生64K上下文。 |
| Qwen 2.5 7B。 | 约12GB。 | Apache 2.0 | 现有默认值。在旧版7B预设中质量最佳。 |
| Qwen 2.5 3B 模型。 | 约8GB。 | Qwen-Research | ⚠ 研究许可证 — 在商业用途前请查看Qwen的许可证条款。 |
| Llama 3.2，30亿参数版本。 | 约8GB。 | Llama Community | 是Qwen 3B的良好替代方案，具有宽松的限制条件。 |
| Llama 3.2 1B | 约6GB。 | Llama Community | 适用于小型设备上的快速实验。 |
| Mistral 7B | 约12GB。 | Apache 2.0 | 与Qwen 7B相当，但使用了不同的对话模板。 |

其他模型通常也适用，但只有这八个模型在CI中被固定。使用 `--lora-preset=quality`（默认值）可以设置成秩为256/全线性目标，参考Biderman 2024 + Thinking Machines 2025，或者使用 `--lora-preset=fast` 设置成秩为16/q+v目标，以获得与v1.2.x相同的内存占用。

## 故障排除

这是最常见的首次运行失败的简要索引。完整的反向索引请参见[故障排除手册页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)。有关驱动程序/显存/混合精度深度分析，请参阅[CUDA故障排除页面](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)。

| 症状 | 错误代码 | 解决方法 |
|---|---|---|
| 训练过程中 GPU 内存耗尽 | `RUNTIME_GPU_OOM` | 自动模式：当出现内存不足错误时，Backpropagate 会减小批次大小，并最多重试 3 次。要禁用此功能，请使用 `Trainer(oom_recovery=False)`。要强制使用更小的批次大小，请使用 `--batch-size 1`。 |
| HuggingFace 返回 401 错误 / "未找到模型" | `DEP_MODEL_LOAD_FAILED` | 使用 `huggingface-cli login` 重新登录。如果出现拼写错误，请从 <https://huggingface.co/models> 复制确切的 ID。 |
| `register_with_ollama` 连接被拒绝 | `DEP_OLLAMA_REGISTRATION_FAILED` | 启动守护进程：`ollama serve`。 从 <https://ollama.com> 安装。 可重试。 |
| 保存检查点时磁盘已满 | `STATE_CHECKPOINT_INVALID` | 在崩溃时，原子写入会在目录中留下一个 `.partial` 目录，可以安全删除。之前的有效检查点仍然存在。 |
| 训练因 GPU 过热而暂停。 | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | 自动模式：当 GPU 温度超过阈值时，Backpropagate 会暂停，并在 GPU 冷却后恢复。如果此问题频繁发生，请改善散热。 |
| `backprop ui --share` 被拒绝 | `INPUT_AUTH_REQUIRED` | 使用 `--auth user:password`，或者使用 SSH 端口转发（请参阅 [Web UI](#web-ui)）。 |
| 首次尝试 GGUF 导出失败 | `RUNTIME_GGUF_EXPORT_FAILED` | 使用 `pip install backpropagate[export]`。 在 Windows 上，还需要 Visual C++ Build Tools + CMake。 |

## 报告错误

当出现错误时，Backpropagate 在启动时会打印一行，例如 `run_started run_id=<uuid>`，并将相同的 ID 绑定到每一行日志、每个检查点以及每个 Weights & Biases 条目。**在任何错误报告中，请包含 `run_id`** — 它可以帮助维护者将所有内容与该特定运行关联起来。

一个好的错误报告应包含：

1. **`run_id`**：启动时打印的 UUID。一个 UUID 允许维护人员将每一行日志、每一个检查点以及 Weights & Biases 中的每一条记录与该特定运行关联起来。
2. **错误代码**：`stderr` 中出现的 `[代码名称]: 消息` 格式的行。请参考[错误代码目录](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/)，了解稳定代码的详细信息。
3. **已屏蔽的堆栈跟踪信息。** 在非详细模式下，`stderr` 会自动进行屏蔽（Bearer 令牌、`sk-*`、`hf_*`、AWS 密钥以及 `password=` / `token=` / `api_key=` 组合会被删除），因此可以安全地复制粘贴。要查看完整的、未屏蔽的堆栈跟踪信息，请重新运行程序，并设置 `BACKPROPAGATE_DEBUG=1`（或使用 `--verbose` 参数）；在发布之前，请仔细审查。
4. **`backprop info` 的输出结果。** 该命令会打印 Python / PyTorch / CUDA / GPU 模型 / VRAM / 操作系统 / 已安装的扩展模块等信息——维护人员需要的所有信息，以便分析特定平台的回归问题。

[错误报告模板](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) 会明确提示您提供以上所有信息，因此问题可以快速得到处理。任何问题、想法或“这是预期的吗？”的讨论，都应该在[GitHub 讨论区](https://github.com/mcp-tool-shop-org/backpropagate/discussions)中进行。安全问题应通过[GitHub 安全建议](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new)表单私下报告，请参阅[SECURITY.md](SECURITY.md) 了解相关政策和响应时间。

## 隐私

所有训练都在您的 GPU 上本地进行。 Backpropagate 仅向 HuggingFace 下载模型时才会进行网络请求（您需要主动发起）。 没有遥测，没有云依赖。

## 参考资料

Backpropagate 的默认设置和多轮训练模式是基于最近的研究。如果您对底层技术感兴趣：

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — 介绍了 LoRA 的基础论文，Backpropagate 使用 LoRA 来高效地训练适配器。
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — 提供了经验证据，表明在大多数后续训练任务中，秩为 256 且针对所有线性目标的 LoRA 质量与完整微调相当，但计算量仅为 67%。这驱动了 Backpropagate v1.3 的默认 LoRA 配置。
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — 这是一个实践性的后续，它识别了在较高 LoRA 秩下，与完整微调相比，所需的 10 倍学习率修正。
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — 原始的对神经网络“遗忘”之前训练内容的描述（EWC — Elastic Weight Consolidation）。
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA，一种较早的方法，通过将新的适配器限制在正交子空间，使用 LoRA 进行持续学习。
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — 一种用于合并多个微调模型的、无需干扰的基础技术。
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — Backpropagate 的多轮合并算法的具体实现。这是一篇 2025 年 12 月的预印本；Backpropagate 是该论文的第一个已知的下游应用。

## 许可证

MIT — 参见 [LICENSE](LICENSE)。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
