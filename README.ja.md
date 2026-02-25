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

**わずか3行のコードで、LLMをファインチューニング。スマートなデフォルト設定、VRAMを考慮したバッチサイズ、マルチランによるSLAO、そしてOllamaへのワンクリックGGUFエクスポート。**

*わずか3行のコードでLLMをトレーニングし、さらに1行でOllamaにエクスポートできます。*

## クイックスタート

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

## なぜバックプロパゲーションが必要なのでしょうか？

| 問題点 | 解決策 |
| --------- | ---------- |
| ファインチューニングは複雑です。 | 3行：ロード、トレーニング、保存 |
| Windowsは悪夢でした。 | 完全なWindowsサポート |
| VRAM管理は困難です。 | 自動バッチサイズ調整、GPU監視 |
| モデルのエクスポートは混乱を招きます。 | ワンクリックでGGUF形式にエクスポートし、Ollamaに自動登録します。 |
| 長時間のトレーニングは忘れ去りを引き起こします。 | マルチランによるSLAOトレーニング |

## 主な機能

- **設計上、ヘッドレス**: CI/CDパイプライン、自動化されたワークフロー、およびプログラムによる実行向けに設計されています。
- **スマートなデフォルト**: ハードウェアとデータセットに基づいて、最適なハイパーパラメータを自動的に設定します。
- **マルチランSLAOトレーニング**: 長時間のトレーニング中に発生する可能性のある破滅的な忘れを防ぐための高度なトレーニング戦略。
- **完全なWindowsサポート**: Windows環境でテストおよび最適化されており、一般的なPyTorch/CUDAの問題を回避します。
- **シームレスなエクスポート**: ワンクリックでGGUF形式にエクスポートし、Ollamaに自動登録します。
- **モジュール式のアーキテクチャ**: 必要な依存関係のみをインストールします（例：`[unsloth]`、`[ui]`、`[export]`）。

## インストール

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Extra | 説明 | 依存関係 |
| ------- | ------------- | -------------- |
| `unsloth` | トレーニング速度が2倍、VRAM使用量が50%削減 | unsloth |
| `ui` | Gradioウェブインターフェース | gradio>=5.6.0 |
| `validation` | Pydanticによる設定検証 | pydantic, pydantic-settings |
| `export` | Ollama用GGUFエクスポート | llama-cpp-python |
| `monitoring` | WandB + システム監視 | wandb, psutil |

**要件:** Python 3.10以上、CUDA GPU (8GB以上のVRAM)、PyTorch 2.0以上

## 使い方

### 基本的なトレーニング

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

### マルチランSLAOトレーニング

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

### Ollamaへのエクスポート

```python
trainer.export(
    format="gguf",
    quantization="q4_k_m",
    register_ollama=True,
    model_name="my-finetuned-model",
)
# ollama run my-finetuned-model
```

### コマンドラインインターフェース (CLI)

```bash
backprop train --data my_data.jsonl --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backpropagate --ui --port 7862
```

## Windowsサポート

Backpropagateは、Windows環境で問題なく動作するように設計されています。

- マルチプロセッシングのクラッシュを避けるための事前トークン化
- RTX 40/50シリーズ向けにxformersを自動的に無効化
- 安全なデータローダー設定
- RTX 5080 (16GB VRAM)でテスト済み

## モデルプリセット

| プリセット | VRAM | Speed | 品質 |
| -------- | ------ | ------- | --------- |
| Qwen 2.5 7B | ~12GB | 中 | Best |
| Qwen 2.5 3B | ~8GB | Fast | Good |
| Llama 3.2 3B | ~8GB | Fast | Good |
| Llama 3.2 1B | ~6GB | 最も速い | Basic |
| Mistral 7B | ~12GB | 中 | Good |

## アーキテクチャ

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

## 関連プロジェクト

[**MCP Tool Shop**](https://mcp-tool-shop.github.io/)の一部です:

- [Tool Compass](https://github.com/mcp-tool-shop-org/tool-compass) — セマンティックなMCPツール検索
- [File Compass](https://github.com/mcp-tool-shop-org/file-compass) — セマンティックなファイル検索
- [Comfy Headless](https://github.com/mcp-tool-shop-org/comfy-headless) — 複雑さを排除したComfyUI

## ライセンス

MIT — 詳細については[LICENSE](LICENSE)を参照してください。
