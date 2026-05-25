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
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/backpropagate"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/backpropagate/badge" alt="OpenSSF Scorecard"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

# アダプターをトレーニングします。それをOllamaに送信します。完了です

Backpropagateは、単一のGPUで大規模言語モデルをファインチューニングするためのPythonライブラリです。わずか3行のコードで、16GBのカードで7Bモデルをトレーニングできます。さらに1つのコマンドで、ファインチューニングしたモデルをOllamaにエクスポートし、`ollama run`コマンドで実行できます。Windowsでも問題なく動作します。

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

これだけで完了です。YAMLの設定ファイルは不要です。`accelerate launch`コマンドも不要です。また、GGUF形式への変換に関するチュートリアルもありません。CUDA対応のGPUと、トレーニングデータを含むJSONLファイルがあれば、わずか3行のコードでファインチューニングを開始できます。

## インストール

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

オプション機能が必要な場合は、以下のいずれかでインストール方法を変更してください。

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

Dockerを使用したい場合は、`docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest`でも動作します。`linux/amd64`と`linux/arm64`の両方のイメージが提供されており、Apple SiliconやARM Linux環境でもネイティブイメージを使用できます。UIをコンテナで実行するための標準的な`compose.yaml`ファイルは、リポジトリのルートにあります。`docker compose up`コマンドを実行すると、Web UIが`http://localhost:7860`で起動し、永続的な`~/.backpropagate`ボリュームがマウントされます。

## Backpropagateの立ち位置

大規模言語モデルのファインチューニングには、いくつかの優れたライブラリがあります。それぞれが異なる強みを持っています。

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**：YAML設定を使用し、他のユーザーのレシピを参考にしたい場合に最適です。
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**：Web GUIを使用し、DPO/PPO/RLHFなどの機能を利用したい場合に最適です。
- **[Unsloth](https://github.com/unslothai/unsloth)**：可能な限り高速なトレーニングが必要で、対応するモデルファミリーを使用する場合に最適です。
- **[torchtune](https://github.com/pytorch/torchtune)**：Metaが提供する、PyTorchネイティブのレシピを編集したい場合に最適です。

Backpropagateは、これらのライブラリとは異なるアプローチを提供します。**単一のコンシューマーGPUで動作するユーザー向けに、アダプターをトレーニングし、配布するための、3行のPython APIです。** YAML設定、GUI、DPO/PPO、マルチノード構成は不要です。必要な機能と、邪魔になるエクスポート手順のみを提供します。

もし、上記のライブラリを試してみて、設定ファイルの煩雑さに困ったり、対応するモデルファミリーがないことに気づいたり、Windows環境での利用を優先したい場合は、Backpropagateが適しています。

## 16GBのコンシューマーGPUでファインチューニングできるモデル

16GBのカード（RTX 4080 / 5080 / 4070 Ti Super）で動作するモデルの目安です。

| モデル | 方法 | 状態 |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | 快適。最大シーケンス長で動作し、余裕があります。 |
| Phi-4-mini-3.8B / Qwen-3.5-4B / SmolLM3-3B (パラメータ数が30億以下) | `mode="full"` (フルファインチューニング) | v1.4 — `backprop train` コマンドで `--mode=full` オプションを付与するか、`Trainer(..., mode="full")` を指定します。勾配チェックポイントとPaged 8-bit Adamを使用することで、活性化メモリをsqrt(L)に抑えます。 |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | 標準的。約7〜8GBのメモリを使用。Backpropagateのデフォルト設定。 |
| Llama-3 13B | QLoRA + サンプルパッキング | ギリギリだが動作する。短いシーケンスを使用してください。 |
| Mixtral 8x7B (合計470億パラメータ) | AQLM 2-bit + LoRA | v1.5では以下の機能が予定されています。詳細は、公開されたV1_5_BRIEFをご確認ください。 |

AQLM 2-bit量子化 (`quant_method="aqlm"`) は、Mixtral-8x7Bを16GBのメモリで動作させるための実験的なオプションとしてv1.4で検討されていましたが、現在はv1.5での実装が予定されています。`aqlm` ライブラリは成熟していますが、v1.4では、フルファインチューニングのサポート（パラメータ数が30億以下のモデルに対して `mode="full"` を設定）を優先し、新しい量子化バックエンドの追加は見送られました。v1.5の実装計画については、公開されたV1_5_BRIEFをご確認ください。

30億パラメータ以下のモデルでは、フルファインチューニング（LoRAだけでなく）が16GBのメモリでも可能であり、v1.4では `mode="full"` として提供されています。`Trainer(..., mode="full")` を指定するか、`backprop train --mode=full --model phi-4-mini-3.8b` コマンドを実行することで有効にできます。30億パラメータを超えるモデルに対しては、`RUNTIME_FULL_FT_MODEL_TOO_LARGE` エラーが発生し、フルファインチューニングは利用できません。この場合、LoRAや30億パラメータ以下のモデルのプリセットを使用できます。設定の詳細や、Biderman 2024 / Thinking Machines 2025による品質比較については、[フルファインチューニングに関する詳細なドキュメント](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/) を参照してください。70億パラメータ以上のモデルでは、フルファインチューニングには24GB以上のGPUメモリが必要です。A100クラウドの利用を検討するか、LoRAを使用することをお勧めします。最近の研究では、LoRAが多くの後処理タスクにおいて、フルファインチューニングと同等の品質を達成できることが示されています（詳細は、[この機能ではないものに関するセクション](#what-backpropagate-is-not-for) を参照）。

## Backpropagateが適さないケース

もし、以下の用途で使用する場合は、別のライブラリを使用することをお勧めします。Backpropagateは適切な選択肢ではなく、無理に使うと、適切なツールを使用するよりも手間と時間がかかります。このセクションを読んでから始めることで、インストールと設定の繰り返しを避けることができます。

- **70億パラメータ以上のモデルのフルパラメータファインチューニング** — BackpropagateはLoRA/QLoRAを使用しており、すべてのパラメータを更新するのではなく、小さなアダプターを学習します。70億パラメータ以上のモデルでは、フルファインチューニングには24GB以上のGPUメモリが必要であり、16GBの一般的なGPUでは動作しません。30億パラメータ以下のモデルでは、フルファインチューニングは16GBのメモリでも可能であり、v1.4では `mode="full"` として提供されています（`Trainer(..., mode="full")` を指定するか、`--mode=full` オプションをCLIで指定します。30億パラメータを超えるモデルに対しては、`RUNTIME_FULL_FT_MODEL_TOO_LARGE` エラーが発生し、LoRAや30億パラメータ以下のモデルのプリセットを使用できます）。補足として、最近の研究（[Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)）によると、適切な設定のLoRAは、ほとんどの後処理タスク（指示に従う、ドメイン適応、ペルソナ/スタイル）において、フルファインチューニングと同等の品質を67%の計算量で達成できます。したがって、ほとんどのユーザーが求めるタスクでは、LoRAを使用しても何も失うことはありません。`mode="full"` は、品質の差を測定し、追加の計算リソースを使用する必要がある場合に利用できます。70億パラメータ以上のモデルをフルファインチューニングする必要がある場合は、HuggingFaceの `transformers.Trainer` を直接、24GB以上のGPUで実行してください。
- **DPO / PPO / GRPO / 嗜好学習** — Backpropagateは、シングルステージの教師ありファインチューニングのみをサポートしています。嗜好学習の場合は、TRLまたはLLaMA-Factoryを直接使用してください。
- **マルチノードトレーニング** — シングルGPUのみをサポートしています。シングルマシン上でのマルチGPU構成は可能ですが（`accelerate launch` を使用）、公式にはサポートされていません。
- **macOSでのトレーニング** — Apple SiliconにはCUDAがないため、トレーニングはLinuxまたはWindowsのNVIDIA GPU搭載マシンで行う必要があります。トレーニング済みのモデルは、Ollamaを使用してMac上で実行できます。
- **サポートされていないモデルファミリー** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B。他のモデルでも動作する場合がありますが、CI環境でのテストは行われていません。

上記のような機能が必要な場合は、上記に記載されているライブラリをご利用ください。それらは、それらの機能に特化しています。

## Backpropagateが提供するもの

インストール時に利用できる、以下の4つの機能：

**1. 設定ファイルが不要な、シンプルな3行のAPI**
このREADMEの冒頭にあるコードは、エンドツーエンドで実行できます。`accelerate config`、YAMLファイル、Hydraの設定などは不要です。`Trainer(model).train(data)`と記述するだけで、ファインチューニングが完了します。

**2. 実際に動作するWindowsサポート**
多くの機械学習ライブラリでは、Windowsは後回しに扱われています。Backpropagateは、Windows + RTX 5080環境で、最初からテストされています。ライブラリは、Windows特有の動作を自動的に処理します。例えば、Windowsのマルチプロセスでクラッシュしないようにデータを事前にトークン化したり、RTX 40/50シリーズのカードでxformersが動作しない場合に自動的に無効にしたり、データローダーの設定を自動的に調整したりします。これらのことをユーザーが知る必要はありません。すべて自動的に処理されます。

**3. 無人実行に対応**
トレーニングには時間がかかります。ユーザーが常に監視する必要はありません。Backpropagateは、バックグラウンドで実行されるように設計されています。

- GPUメモリが不足した場合、バッチサイズを自動的に半分にし、最大3回まで再試行します。手動での調整は不要です。
- GPUの温度が上昇した場合、温度が下がるまで一時停止し、その後、再開します。
- すべてのチェックポイントは、アトミックに書き込まれます。つまり、ラップトップが保存中にクラッシュした場合でも、以前の正常なチェックポイントは保持されます。
- すべてのトレーニング実行には、一意のIDが割り当てられ、ログの各行、チェックポイント、およびWeights & Biasesのエントリに記録されます。問題が発生した場合、このIDを使用することで、管理者がすべての情報を関連付けることができます。
- エラーには、安定したコード（`RUNTIME_GPU_OOM`、`DEP_OLLAMA_REGISTRATION_FAILED`など）が付属しています。これにより、ログを検索したり、[トラブルシューティングガイド](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)を参照して、解決策を見つけることができます。CUDA関連のエラーについては、専用の[CUDAトラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)があります。

**4. 学習済みのアダプターから`ollama run`まで、1つのコマンド**
多くのライブラリがモデルを学習しますが、実際に使用したい場合に、ユーザーの邪魔をしないものも多くありません。Backpropagateは、Ollamaで使用される形式（GGUF）にエクスポートし、Ollamaモデルを登録する機能を、1つのコマンドで提供します。トレーニングが完了してから、ファインチューニングしたモデルでチャットを開始するまで、わずか30秒で完了します。

## クイックスタート

このリポジトリには、小さなサンプルデータセットが含まれており、このREADMEの冒頭にあるコードが、クリーンな環境で実行できるように設計されています。

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

これは、5つの短いShareGPT形式の会話データセットでQwen 2.5 7Bのアダプターを学習し、その結果をGGUF形式でエクスポートするものです。ご自身のデータを使用する場合は、JSONLファイルを1行1つの例で記述してください。

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca（`instruction` / `output`）、OpenAIのチャット（`messages`）、および生のテキスト形式も使用できます。Backpropagateは、形式を自動的に検出します。

より包括的なワークフロー（ファインチューニングとHugging Face Hubへのアップロード、OOMエラーからの再開、長期的なキャンペーンにおける複数回の学習など）については、[handbook recipes page](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/)を参照してください。

### Web UI（オプション）

Pythonコードを直接入力する代わりに、Web UIを使用したい場合は、UIの追加機能をインストールして起動します。

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

ローカルのWebインターフェースが`http://localhost:7862`で起動します。ここで、データセットを指定し、モデルを選択し、学習を実行し、エクスポートすることができます。UIはデフォルトではローカルでのみ利用可能です。他のデバイスからアクセスできるようにするには、[Web UI](#web-ui)を参照して、`--share`と`--auth`のセキュリティ設定を確認してください。

## 複数回の学習

複数のデータセットで段階的にファインチューニングを行いたい場合（たとえば、毎週新しい学習データを受け取り、以前に学習した内容を忘れることなく追加したい場合）は、Backpropagateの`multi_run`モードが適しています。

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

これは、5回の学習を繰り返し実行し、各回の実行でアダプターをマージすることで、以前の知識を維持しながら新しい例を取り入れる方法です。この手法は、最近の研究に基づいています。詳細は、このREADMEの末尾にある[References](#references)を参照してください。

CLI（コマンドラインインターフェース）版：

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## チェックポイントからの再開

4回目の実行でクラッシュした5回の学習も、再開可能です。すべての`multi_run`セッションは、実行IDをオンディスクの履歴ファイルとチェックポイントマニフェストに書き込みます。そのため、中断した場所から再開するには、1つのコマンドで済みます。

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

`backprop multi-run`のデフォルト設定（`--resume`オプションなし）では、同じ出力ディレクトリ内に実行中のセッションが存在する場合、それを自動的に検出し、再開します。クリーンな状態から開始するには、新しい出力ディレクトリを指定してください。

## トレーニング履歴

`backprop train`および`backprop multi-run`の実行ごとに、`<output>/run_history.json`ファイルに1行が記録されます。記録される内容は、使用したモデル、データセット、ハイパーパラメータ、ステータス、最終的な損失、損失履歴などです。過去の実行を一覧表示し、確認することができます。

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## 実験の追跡

Backpropagateは、インストールされている実験追跡ツール（Weights & Biases、TensorBoard、MLflow）を自動的に検出し、連携させます。`wandb`がインストールされており、ログインしている場合は、各実行が自動的にW&Bに記録されます。記録される実行名は、オンディスクの実行IDと一致します。これにより、W&B、ログ、および`run_history.json`ファイルを、1つの識別子で検索することができます。

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

`Trainer(report_to=["wandb"])`、`Trainer(report_to=["tensorboard"])`、または`Trainer(report_to="none")`を使用して、この機能を無効にすることができます。

## Web UI

ReflexのWebインターフェースは、オプションで利用可能です。`pipx install "backpropagate[ui]"`でインストールし、起動します。

```bash
backprop ui --port 7862
```

UIは、ローカルで`http://localhost:7862`で実行されます。他のデバイス（ネットワーク上の他のユーザー、パブリックURLなど）からアクセスできるようにするには、`--share`（または`--host`）と`--auth`を組み合わせて使用する必要があります。

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share`を`--auth`なしで実行すると、エラーが発生します。その理由は、`--share`はインターネット上の誰でもアクセスできるURLを公開しますが、認証なしでは、誰でも学習パイプラインを操作したり、Hugging Faceのトークンを読み取ったりできる可能性があるためです。この機能を無効にすることはできません。認証情報を設定したくない場合は、SSHポートフォワードを使用してください。

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

完全な脅威モデルについては、[handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/)を参照してください。

UI からのファイルシステムへの書き込みは、単一のディレクトリにサンドボックス化されています。

- デフォルト: `~/.backpropagate/ui-outputs`
- 上書き: `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own` を設定
- 上書きは、許可リスト方式で検証されます。システムや認証情報に関連するパス (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32` など) は許可されません。

## プラットフォームに関する注意点

**要件:** Python 3.10以上、CUDA対応GPU（8GB以上のVRAM）、PyTorch 2.0以上

Python 3.10 は、2026年10月にサポート終了となります。Backpropagate は、v1.4 で Python 3.10 のサポートを終了する予定です。新規インストールの場合は、Python 3.11 または 3.12 を推奨します。3.11 が最もテストされているバージョンです。

Backpropagate は、異なるプラットフォームでのトレーニング時の動作に関する問題を処理しますが、インストール時の問題を解決することはできません。最も一般的な問題は以下の2つです。

- **誤った CUDA wheel の選択:** PyTorch は、CUDA のバージョンごとに異なるバイナリが提供されています。誤ったものを選択すると、CPU のみを使用する PyTorch がインストールされ、トレーニング速度が著しく低下します。お使いのドライバに合った wheel を、<https://pytorch.org/get-started/locally/> で選択してください。`nvidia-smi` コマンドを実行して、ドライバと CUDA のバージョンを確認してください。
- **Windows + GGUF エクスポート:** `[export]` オプションは、ソースコードから `llama-cpp-python` をビルドしますが、これには Visual Studio Build Tools (C++ コンポーネント) と CMake が必要です。

**macOS:** GPU トレーニングはサポートされていません (CUDA 非対応)。トレーニング済みのアダプターは、Ollama を使用して Mac で実行できますが、`trainer.train()` を実行すると `DEP_GPU_NOT_AVAILABLE` エラーが発生します。トレーニング自体は、CUDA 対応の Linux または Windows マシンで行ってください。

詳細なインストールに関するトラブルシューティングガイドは、[トラブルシューティングハンドブックのページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) を参照してください。また、ドライバ / VRAM / xformers / bf16-vs-fp16 に関する問題については、専用の [CUDA トラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) を参照してください。

## CLI

すべての Python API には、対応する CLI (コマンドラインインターフェース) が用意されています。

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

完全なリファレンスは、[CLI ハンドブックのページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) を参照するか、`backprop <サブコマンド> --help` を実行してください。

## 設定

すべての設定は、`BACKPROPAGATE_` というプレフィックスが付いた環境変数を使用して上書きできます。

| 変数 | デフォルト値 | 備考 |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | JSON またはコンソールログの出力を強制 |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | デフォルトモデル |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | 学習率 |
| `BACKPROPAGATE_LORA__R` | `256` | LoRA のランク (v1.3 のデフォルト値。v1.2.x のデフォルト値である 16 を使用するには、`--lora-preset=fast` を指定) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | UI ファイルシステムサンドボックス |

ネストされたキーには、二重アンダースコアを使用します (`MODEL__NAME`、`MODEL_NAME` ではありません)。完全なリファレンスは、[環境変数ハンドブックのページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/) を参照してください。

## モデルプリセット

| プリセット | VRAM | ライセンス | 備考 |
|---|---|---|---|
| Qwen-3.5-4B | ~8GB | Apache 2.0 | 5B 以下のモデルに推奨されるデフォルト値。このサイズで最高の品質を提供します。 |
| Phi-4-mini-3.8B | ~8GB | MIT | 推論 / 数学 / コードにおいて高い性能を発揮します。ライセンスの制約が緩やかです。 |
| SmolLM3-3B | ~6GB | Apache 2.0 | 完全にオープンソースのレシピ。ネイティブな 64K コンテキストに対応。 |
| Qwen 2.5 7B | ~12GB | Apache 2.0 | 既存のデフォルト値。レガシーの 7B プリセットの中で最高の品質を提供します。 |
| Qwen 2.5 3B | ~8GB | Qwen-Research | ⚠ 研究ライセンス — 商用利用の際は、Qwen のライセンス条項を確認してください。 |
| Llama 3.2 3B | ~8GB | Llama Community | Qwen 3B の優れた代替案であり、許可条件があります。 |
| Llama 3.2 1B | ~6GB | Llama Community | 小規模な環境での迅速な実験に適しています。 |
| Mistral 7B | ~12GB | Apache 2.0 | Qwen 7B と同等ですが、チャットテンプレートが異なります。 |

他のモデルも動作する可能性がありますが、CI (継続的インテグレーション) で使用されるのはこれらの 8 つのモデルのみです。ランク 256 / all-linear ターゲット (Biderman 2024 + Thinking Machines 2025 の推奨) を使用するには、`--lora-preset=quality` (デフォルト) を指定します。v1.2.x のフットプリントが必要な場合は、`--lora-preset=fast` を指定して、レガシーのランク 16 / q+v ターゲットを使用します。

## トラブルシューティング

初回実行時に発生する可能性のある一般的なエラーの簡単な一覧です。完全な逆引き一覧は、[トラブルシューティングハンドブックのページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) を参照してください。ドライバ / VRAM / 混合精度に関する詳細については、[CUDA トラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) を参照してください。

| 症状 | エラーコード | 対処法 |
|---|---|---|
| GPUメモリ不足によるトレーニング中断 | `RUNTIME_GPU_OOM` | 自動モードでは、メモリ不足が発生した場合、バッチサイズを半分にし、最大3回まで再試行します。この機能を無効にするには、`Trainer(oom_recovery=False)` と指定してください。バッチサイズを強制的に小さくするには、`--batch-size 1` オプションを使用してください。 |
| HuggingFaceから401エラー（「モデルが見つかりません」）が返ってきました。 | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` を実行し、再度試してください。入力ミスがある場合は、<https://huggingface.co/models> から正確な ID をコピーしてください。 |
| `register_with_ollama` への接続拒否 | `DEP_OLLAMA_REGISTRATION_FAILED` | デーモンを起動します: `ollama serve`。 <https://ollama.com> からインストールしてください。再試行可能です。 |
| チェックポイント保存中にディスク容量不足 | `STATE_CHECKPOINT_INVALID` | クラッシュ時に、`.partial` というディレクトリが作成されますが、削除しても問題ありません。以前の正常なチェックポイントは保持されています。 |
| GPUの過熱により、トレーニングを一時中断しました。 | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | 自動モードでは、温度が設定された閾値を超えると一時的に動作が停止し、GPUが冷却されると再開されます。この現象が頻繁に発生する場合は、冷却効率を改善するために、エアフローを見直してください。 |
| `backprop ui --share` が拒否される | `INPUT_AUTH_REQUIRED` | `--auth user:password` オプションを指定するか、代わりに SSH のポートフォワード機能を使用してください（[Web UI](#web-ui) を参照）。 |
| GGUF エクスポートが初回試行で失敗 | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]` を実行してください。Windows の場合は、Visual C++ Build Tools + CMake も必要です。 |

## バグの報告

Backpropagateがエラーを検出した場合、起動時に`run_started run_id=<UUID>`のようなメッセージを表示し、同じIDをすべてのログ、チェックポイント、およびWeights & Biasesのエントリに紐付けます。**バグ報告を行う際は、必ず`run_id`を含めてください。** これにより、開発者が特定の実行に関する情報をすべて関連付けることができます。

良いバグレポートには、以下の情報が含まれている必要があります。

1. **`run_id`**: 起動時に表示されるUUID。このUUIDを使用することで、管理者は特定の実行に関するすべてのログ行、チェックポイント、およびWeights & Biasesのエントリを関連付けることができます。
2. **エラーコード**: `stderr`に出力される`[コード名]: メッセージ`という形式の文字列。安定したコードの一覧は、[エラーコード](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/)を参照してください。
3. **追跡情報のマスキング**: `stderr`は、詳細表示モードでない場合、自動的にマスキングされます（Bearerトークン、`sk-*`、`hf_*`、AWSキー、`password=` / `token=` / `api_key=`のペアなどが削除されます）。貼り付けても安全です。完全な追跡情報を確認するには、`BACKPROPAGATE_DEBUG=1`（または`--verbose`）を指定して再度実行し、投稿前に内容を確認してください。
4. **`backprop info`の出力**: このコマンドは、Python、PyTorch、CUDA、GPUモデル、VRAM、OS、インストールされている追加機能など、管理者がプラットフォーム固有の問題を特定するために必要なすべての情報を表示します。

[バグレポートテンプレート](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml)では、これらの情報をすべて明示的に入力するように促しており、そのため問題の切り分けが迅速に進みます。質問、アイデア、または「これは想定される動作ですか？」といった議論は、[GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions)で行ってください。セキュリティに関する問題は、[GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new)のフォームを通じて、非公開で報告してください。ポリシーと対応のタイムラインについては、[SECURITY.md](SECURITY.md)を参照してください。

## プライバシー

トレーニングはすべて、ローカルの GPU 上で行われます。Backpropagate は、HuggingFace からモデルをダウンロードするために必要なネットワークリクエストを除き、他のネットワークリクエストは行いません。テレメトリーやクラウドへの依存はありません。

## 参考文献

Backpropagateのデフォルト設定や複数回のトレーニングを行う機能は、最新の研究に基づいて設計されています。もし、その基盤となる技術にご興味があれば：

- **Hu et al. 2021.** *LoRA: 低ランク適応による大規模言語モデルの効率化.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — LoRAを紹介する基礎論文。Backpropagateは、この技術を用いてアダプターを効率的に学習します。
- **Biderman et al. 2024.** *LoRAはより少なく学習し、より少なく忘却する.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — 実験的な証拠として、ランク256のLoRAで、すべての線形ターゲットを使用した場合、ほとんどの追加学習タスクにおいて、フルファインチューニングと同等の品質を、計算量の67%で達成できることが示されています。これは、Backpropagateのバージョン1.3のデフォルトLoRA設定の基盤となっています。
- **Thinking Machines 2025.** *後悔のないLoRA.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — 高いLoRAランクを使用する場合に必要な、学習率とフルファインチューニングとの関係を明らかにする実践的な解説。
- **Kirkpatrick et al. 2017.** *ニューラルネットワークにおける破滅的な忘却の克服.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — ニューラルネットワークが、新しいデータでファインチューニングを行う際に、以前の学習内容を「忘れてしまう」理由を最初に説明した論文（EWC：Elastic Weight Consolidation）。
- **Wang et al. 2023.** *言語モデルの継続学習のための直交部分空間学習.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRAは、新しいアダプターを直交部分空間に制限することで、LoRAを継続学習に利用する初期のアプローチです。
- **Yadav et al. 2023.** *TIES-マージ：モデルのマージ時の干渉の解消.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — 複数のファインチューニング済みモデルを、干渉なくマージするための基本的な技術です。
- **Qiao & Mahdavi 2025.** *忘却する前にマージ：継続的なマージによる単一のLoRA継続学習.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — Backpropagateが実装している、特定のアルゴリズムです。2025年12月に発表されたプレプリントであり、Backpropagateがこの論文を最初に実用化した事例として知られています。

## ライセンス

MITライセンスについては、[LICENSE](LICENSE) をご参照ください。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
