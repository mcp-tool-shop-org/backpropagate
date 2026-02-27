<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**ヘッドレスLLMのファインチューニングをわずか3行で実現。最適なデフォルト設定、VRAMを考慮したバッチサイズ、マルチランSLAO、そしてOllamaへのワンクリックGGUFエクスポート機能。**

*わずか3行のコードでLLMをトレーニング。さらに1行でOllamaにエクスポートできます。*

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

## なぜバックプロパゲーションを行うのか？

| 問題点 | 解決策 |
|---------|----------|
| ファインチューニングは複雑 | 3行：ロード、トレーニング、保存 |
| Windowsは悪夢 | 完全対応のWindowsサポート |
| VRAM管理は難しい | 自動バッチサイズ調整、GPU監視 |
| モデルのエクスポートは混乱を招く | ワンクリックでGGUF形式にエクスポートし、Ollamaに自動登録 |
| 長時間のトレーニングは忘却を引き起こす | マルチランSLAOトレーニング |

## 主な機能

- **設計上ヘッドレス**: CI/CDパイプライン、自動化されたワークフロー、およびプログラムによる実行向けに最適化されています。
- **スマートなデフォルト**: ハードウェアとデータセットに基づいて、最適なハイパーパラメータを自動的に設定します。
- **マルチランSLAOトレーニング**: 長時間のトレーニング中に発生する可能性のある破滅的な忘却を防ぐための高度なトレーニング戦略。
- **完全対応のWindowsサポート**: Windows環境で動作確認済みであり、一般的なPyTorch/CUDAの問題を回避します。
- **シームレスなエクスポート**: ワンクリックでGGUF形式にエクスポートし、Ollamaへの自動登録を行います。
- **モジュール式のアーキテクチャ**: 必要な依存関係のみをインストールできます (例: `[unsloth]`, `[ui]`, `[export]`)。

## インストール

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| その他 | 説明 | 依存関係 |
|-------|-------------|--------------|
| `unsloth` | トレーニング速度が2倍、VRAM使用量が50%削減 | unsloth |
| `ui` | Gradioウェブインターフェース | gradio>=5.6.0 |
| `validation` | Pydanticによる設定検証 | pydantic, pydantic-settings |
| `export` | Ollama用GGUFエクスポート | llama-cpp-python |
| `monitoring` | WandB + システム監視 | wandb, psutil |

**必要条件**: Python 3.10以上、CUDA対応GPU (8GB以上のVRAM)、PyTorch 2.0以上

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

### コマンドラインインターフェース

```bash
backprop train --data my_data.jsonl --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backpropagate --ui --port 7862
```

## Windowsサポート

Backpropagateは、Windows上で問題なく動作するように設計されています。

- マルチプロセッシングによるクラッシュを回避するための事前トークン化
- RTX 40/50シリーズ向けにxformersを自動的に無効化
- 安全なデータローダー設定
- RTX 5080 (16GB VRAM)で動作確認済み

## モデルプリセット

| プリセット | VRAM | 速度 | 品質 |
|--------|------|-------|---------|
| Qwen 2.5 7B | 約12GB | 中 | 最高 |
| Qwen 2.5 3B | 約8GB | 高速 | 良好 |
| Llama 3.2 3B | 約8GB | 高速 | 良好 |
| Llama 3.2 1B | 約6GB | 最速 | 基本 |
| Mistral 7B | 約12GB | 中 | 良好 |

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

## プライバシー

トレーニングはすべてローカルのGPU上で行われます。Backpropagateは、HuggingFaceからモデルをダウンロードするためにネットワークリクエストを行う以外は、一切リクエストを行いません。テレメトリーも、クラウドへの依存もありません。

## スコアカード

| カテゴリ | スコア | 備考 |
|----------|-------|-------|
| A. セキュリティ | 10/10 | SECURITY.md、CIにおけるBandit+Semgrep+Trivy+TruffleHog、パストラバーサル保護 |
| B. エラー処理 | 8/10 | 構造化されたエラー、GPUの安全マージン、チェックポイント復元 |
| C. ドキュメント | 9/10 | README、CHANGELOG、モジュール式のインストールガイド、CLIヘルプ |
| D. ソフトウェア品質 | 9/10 | CI + テスト (33ファイル)、PyPI公開、Codecovカバレッジ |
| E. 識別情報 | 10/10 | ロゴ、翻訳、ランディングページ、PyPIへの登録情報 |
| **Total** | **46/50** | |

## ライセンス

MITライセンス — 詳細については、[LICENSE](LICENSE) を参照してください。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
