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

**わずか3行で大規模言語モデル（LLM）のファインチューニングが可能。最適なデフォルト設定、VRAMを考慮したバッチサイズ調整、複数回の学習によるSLAO、そしてOllamaへのワンクリックGGUFエクスポート機能。**

*SLAOは、非対称なマージによる継続学習（Single LoRA Continual Learning via Asymmetric Merging）であり、長期間のファインチューニング中に発生する破滅的な忘却を防ぐための手法です。（[論文](https://arxiv.org/abs/2512.23017)）*

*わずか3行のコードでLLMを学習させ、さらに1行でOllamaにエクスポートできます。*

## クイックスタート

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("examples/quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

このリポジトリには、`examples/quickstart.jsonl`という小さなファイル（ShareGPT形式の5つの例）が含まれており、これにより、上記のコードがクリーンな環境でエンドツーエンドで実行できます。独自の学習を行う場合は、以下の「データセット形式」を参照してください。

### コード不要：Web UI

Python REPLではなく、UIを使用したい場合は、同じ追加機能をインストールして実行してください。

```bash
pip install backpropagate[standard]
backprop ui --port 7862
```

Reflex (Radix UI) インターフェースを使用すると、JSONLファイルを選択し、モデルを選択し、学習を実行し、エクスポートすることができます。Pythonの知識は不要です。このUIはローカル環境での利用を前提としており、インターネット経由で公開する場合は、`--share` + `--auth`によるセキュリティ設定と、サポートされているトンネルオプション（Cloudflare Tunnel、ngrok）を参照してください（[Web UI](#web-ui)）。

## データセット形式

JSONL形式の学習ファイルは、各行に1つの例が記述されている必要があります。最もシンプルな形式は、ShareGPTのチャット形式です。

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca (`instruction`/`output`)、OpenAIのチャット (`messages`)、および生のテキスト形式もサポートされています。`examples/quickstart.jsonl`には、コピーして利用できるサンプルが含まれています。

## なぜバックプロパゲーションを行うのか？

| 問題点 | 解決策 |
|---------|----------|
| ファインチューニングは複雑 | 3行で完了：ロード、学習、保存 |
| Windowsは使い物にならない | Windowsへの完全な対応 |
| VRAM管理は難しい | 自動バッチサイズ調整、GPU監視 |
| モデルのエクスポートは分かりにくい | ワンクリックでGGUF形式にエクスポートし、Ollamaに自動登録 |
| 長時間の学習は忘却を引き起こす | 複数回の学習によるSLAO |

## 主な機能

- **設計上はヘッドレス**: CI/CDパイプライン、自動化されたワークフロー、およびプログラムによる実行向けに最適化されています。
- **スマートなデフォルト設定**: ハードウェアとデータセットに基づいて、最適なハイパーパラメータを自動的に設定します。
- **複数回の学習によるSLAO**: 長時間の学習中に発生する破滅的な忘却を防ぐための高度な学習戦略。
- **Windowsへの完全な対応**: Windows環境での動作をテストおよび最適化し、一般的なPyTorch/CUDAの問題を回避します。
- **シームレスなエクスポート**: ワンクリックでGGUF形式にエクスポートし、Ollamaへの自動登録を行います。
- **モジュール式のアーキテクチャ**: 必要な依存関係のみをインストールできます（例：`[unsloth]`、`[ui]`、`[export]`）。

## インストール

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Reflex (Radix UI) web interface
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| 追加機能 | 説明 | 依存関係 |
|-------|-------------|--------------|
| `unsloth` | 学習速度が2倍、VRAM使用量が50%削減 | unsloth |
| `ui` | Reflex (Radix UI) Webインターフェース | reflex>=0.9.2, fastapi>=0.115 |
| `validation` | Pydanticによる設定検証 | pydantic, pydantic-settings |
| `export` | Ollama用GGUFエクスポート | llama-cpp-python |
| `monitoring` | WandB + システム監視（v1.1.0でトレーナーに自動統合） | wandb, psutil |
| `observability` | OpenTelemetryによるトレース | opentelemetry-api, opentelemetry-sdk |
| `logging` | 構造化されたロギング | structlog |
| `security` | JWT認証 + トークン生成 | PyJWT, cryptography |
| `production` | unsloth + ui + 検証 + ロギング + セキュリティ | （バンドル） |

**必要条件:** Python 3.10以上、CUDA対応GPU（8GB以上のVRAM）、PyTorch 2.0以上

### プラットフォームの前提条件

Backpropagateは、実行時の問題（マルチプロセッシング、RTX 40/50でのxformersの使用、Windows上のデータローダー）を処理します。ただし、インストール時のプラットフォーム固有の問題は処理しません。それらの問題は事前に解決してください。

- **CUDAツールキットのバージョン。** PyTorchはCUDAバージョンごとにリリースされます。誤ったホイールを選択すると、CPUのみのPyTorchがインストールされます。正しい`pip install torch ...`コマンドについては、<https://pytorch.org/get-started/locally/>にあるツールを使用してください。`nvidia-smi`を実行して、ドライバー/CUDAのバージョンを確認してください。
- **Windows。** `[export]`オプションを使用する場合（`llama-cpp-python`をソースからビルドする場合）、Visual Studio Build Tools (C++)とCMakeが必要です。`bitsandbytes`のホイールは、Windowsネイティブで利用可能になりました（>= 0.43）。古いドキュメントで`bitsandbytes-windows`について言及しているものは、最新ではありません。
- **macOS。** GPUによる学習は**サポートされていません**。CUDAは利用できません。エクスポートされたGGUFファイルをOllama経由で実行して*推論*を行うことはできますが、`trainer.train()`を実行すると`DEP_GPU_NOT_AVAILABLE`エラーが発生します。学習にはCUDA対応の環境を使用してください。
- **Linux。** ほとんどのディストリビューションで問題なく動作します。PyPIのバイナリリリースを使用している場合は、LinuxのビルドではCPUのみのPyTorchが使用されていることに注意してください（GitHubの2GBのリリースアセットの制限を超えるため）。最初にpytorch.orgから対応するCUDAホイールをインストールしてください。

詳細なインストールに関するトラブルシューティングについては、[トラブルシューティングガイド](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)を参照してください。

## 設定

すべての設定は、`BACKPROPAGATE_`というプレフィックスを持つ環境変数を使用して上書きできます（例：`BACKPROPAGATE_LOG_LEVEL=debug`）。プロジェクトのルートディレクトリにある`.env`ファイルは、`[validation]`オプションがインストールされている場合に自動的に読み込まれます。

一般的な設定項目（詳細は[環境変数リファレンス](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/)を参照）：

| 変数 | デフォルト値 | 備考 |
|----------|---------|-------|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | 自動 | JSON形式のログを出力するか（`true`）、コンソールに出力するか（`false`） |
| `BACKPROPAGATE_LOG_FILE` | 未設定 | ログを保存するパス |
| `BACKPROPAGATE_DEFER_FEATURE_DETECTION` | 未設定 | 起動時のオプション依存関係の検出をスキップし、CLIの起動を高速化 |
| `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE` | `true` | `true`の場合、`--auth`オプションなしで`backprop ui --share`コマンドを実行できない |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | すべてのUIファイルシステムの書き込みのサンドボックスベース。許可リストで検証済み |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | デフォルトモデル |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | 学習率 |
| `BACKPROPAGATE_LORA__R` | `16` | LoRAのランク |

ネストされたキーでは、区切り文字としてダブルアンダースコアを使用します（Pydanticの`env_nested_delimiter`の規約）。

## 使い方

### 基本的な学習

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

`Qwen/Qwen2.5-7B-Instruct`がデフォルトです。`Trainer()`を呼び出す際にモデルの引数を指定しない場合、この値が使用されます（`config.py`の`ModelConfig.name`を参照）。以前の例では、事前量子化された`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`が使用されていましたが、より安定性の高い公式のQwenの重みを使用するように変更されました（[CHANGELOG v0.1.3](CHANGELOG.md)）。どちらのモデルでも動作します。

### マルチランSLAO学習

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

SLAO（Single LoRA Continual Learning via Asymmetric Merging）は、[Merge before Forget](https://arxiv.org/abs/2512.23017)という論文を実装しています。QR分解による直交A行列の初期化、非対称なA/Bの処理、および時間依存の`λ(i) = 1/√i`によるスケーリングが行われます。CLIフラグは`--samples`（対応するフィールドは`samples_per_run`）です。

### Ollamaへのエクスポート

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

すべてのサブコマンドとフラグについては、[CLIリファレンス](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/)を参照するか、`backprop <サブコマンド> --help`を実行してください。

### チェックポイントからの再開（v1.1.0）

5回の実行を含むマルチ実行で、4回目の実行でクラッシュが発生した場合でも、実行を再開できます。すべてのマルチ実行セッションは、`run_history.json`ファイルとディスク上のチェックポイントマニフェストの両方に`run_id`を書き込みます。そのため、中断した場所から再開するには、1つのコマンドで済みます。

```bash
backprop resume <run-id>                       # picks up the in-progress session
backprop multi-run --data ... --resume <run-id> # explicit form
backprop train --data ... --resume <run-id>    # single-run resume (continues run_id)
```

`backprop multi-run`のデフォルトの動作（`--resume`オプションなし）では、同じ出力ディレクトリ内の実行中のエントリを自動的に検出し、それを継続します。`resume_from="off"`（Python API）を指定するか、`--resume`オプションを省略して、新しい出力ディレクトリで開始することで、クリーンなセッションを強制できます。

マルチ実行が再開されると、その`run_id`に対応する最新のチェックポイントがモデルにロードされ、`slao/`ディレクトリにあるSLAOマージの状態が復元され、実行ループは`last_completed_run + 1`から継続されます。履歴エントリの`status`が`running`に戻るため、`backprop list-runs --status running`コマンドで実行中のセッションが表示されます。

### 実験の追跡（v1.1.0）

`Trainer`は、インストールされている実験追跡ツール（`wandb`、`tensorboard`、`mlflow`）を自動的に検出し、それらを基盤となる`transformers.TrainingArguments`に統合します。デフォルトでは、`report_to="auto"`が設定されており、インポート可能なものが使用されます。

```bash
pip install backpropagate[monitoring]  # installs wandb + psutil
wandb login                            # one-time
backprop train --data my_data.jsonl    # W&B run gets the same run_id prefix as the on-disk history
```

`Trainer(report_to=["wandb"])`、`Trainer(report_to=["tensorboard"])`、または`Trainer(report_to="none")`を使用して、明示的に無効化できます。MLflowを使用するには、`pip install mlflow`を実行します。TensorBoardを使用するには、`pip install tensorboard`を実行します。W&Bの実行名は`backprop-<run_id_prefix>`なので、オペレーターはW&B、ログ、および`run_history.json`を同じ識別子で検索できます。

### トレーニング履歴

`backprop train`および`backprop multi-run`の実行ごとに、`<output>/run_history.json`ファイルに、`run_id`、モデル、データセット、ハイパーパラメータ、ステータス、最終的な損失、損失履歴、および（マルチ実行の場合）SLAOマージのタイムラインが記録されます。最近の実行の一覧を表示するには、次のコマンドを使用します。

```bash
backprop list-runs                         # most recent 20 runs, all statuses
backprop list-runs --status failed         # filter
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial run_id ok)
```

実行履歴は、プロセスをまたいで保持されます。Web UIの「Runs」タブは、メモリ内のビューであり、オンディスクの履歴が`list-runs` / `show-run` / `resume`の真実のソースです。

### Web UI

ローカルでReflexインターフェースを起動します。

```bash
backprop ui --port 7862
```

パブリックインターネットURLを公開するには、`--share`オプションを`--auth`オプションと組み合わせて使用する必要があります。

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share`コマンドを`--auth`オプションなしで実行すると、コード`1`で終了し、構造化されたエラーメッセージ`[INPUT_AUTH_REQUIRED]`が表示されます。これは、`--share`オプションが`*.gradio.live`というURLを公開し、そのURLはインターネット上の誰でもアクセスできるため、認証なしでは、誰でもトレーニングパイプラインを制御できる可能性があるためです。

明示的に無効化するには（例：内部開発環境）、環境変数`BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false`を設定します。この場合、起動時に大きな警告が表示されます。また、認証されていないUIが起動するまでに5秒の猶予があるので、問題がある場合は`Ctrl-C`で中断できます。

UIからのファイルシステムへの書き込みは、単一のディレクトリにサンドボックス化されています。

- デフォルト: `~/.backpropagate/ui-outputs`
- 上書き: `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- 上書きは、許可リストで検証されます。システム/認証パス（`/etc`、`/var`、`~/.ssh`、`~/.aws`、`C:\Windows\System32`など）は、`[UI_OUTPUT_DIR_FORBIDDEN]`というエラーで拒否されます。

## Windowsサポート

Backpropagateは、Windows上で動作するように設計されています。

- マルチプロセッシングのクラッシュを回避するための事前トークン化
- RTX 40/50シリーズに対するxformersの自動無効化
- 安全なデータローダー設定
- RTX 5080 (16GB VRAM)でテスト済み

## モデルプリセット

| プリセット | VRAM | 速度 | 品質 |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12GB | 中 | 最高 |
| Qwen 2.5 3B | ~8GB | 高速 | 良 |
| Llama 3.2 3B | ~8GB | 高速 | 良 |
| Llama 3.2 1B | ~6GB | 最速 | 基本 |
| Mistral 7B | ~12GB | 中 | 良 |

## アーキテクチャ

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
├── ui_security.py       # Rate limiting, CSRF, file validation (framework-agnostic)
├── ui_gradio_legacy.py  # DEPRECATED — preserved as v1.0 reference; removed in v1.2
└── theme_gradio_legacy.py  # DEPRECATED — same
```

## トラブルシューティング

よくある初期エラーの簡単な索引。詳細な逆索引は、[トラブルシューティングハンドブックのページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) にあります。以下のすべてのコードは、[エラーコード](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) で説明されています。

| 症状 | コード | 解決策 |
|---------|------|-----|
| トレーニング中にGPUのメモリが不足する | `RUNTIME_GPU_OOM` | OOM（メモリ不足）自動復旧（B-002）が、バッチサイズを最大3回まで自動的に半分にします。無効にするには、`Trainer(oom_recovery=False)` を指定します。バッチサイズを強制的に小さくするには、`--batch-size 1` を指定します。 |
| HF Hubが401エラー/「モデルが見つかりません」と表示される | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` を実行し、再試行してください。入力ミスがある場合は、<https://huggingface.co/models> から正確なIDをコピーしてください。 |
| モデル名の入力ミス | `INPUT_VALIDATION_FAILED` または `DEP_MODEL_LOAD_FAILED` | <https://huggingface.co/models> での `org/name` の識別子を確認してください。 |
| `register_with_ollama` への接続が拒否される | `DEP_OLLAMA_REGISTRATION_FAILED` | デーモンを起動します: `ollama serve`。 <https://ollama.com> からインストールしてください。再試行可能です。 |
| チェックポイントの保存中にディスクがいっぱいになる | `STATE_CHECKPOINT_INVALID` | クラッシュが発生すると、`.partial` ディレクトリが残ります。削除しても安全です。以前の正常なチェックポイントはそのままです。 |
| GPUの過熱によりトレーニングが一時停止/中断される | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | B-003: NVMLの温度閾値を超えるとモニタが一時停止し、GPUが冷却されると自動的に再開されます。エアフローを改善するか、持続的な負荷を下げてください。 |
| `backprop ui --share` が拒否される | `INPUT_AUTH_REQUIRED` | `--auth user:password` を指定するか、`BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false` を設定して、認証を無効にします（警告が表示されます）。 |
| 複数回の実行における「検証の重複」 | `CONFIG_INVALID` (Stage A backend B-001) | `--samples` の値を、トレーニングプールのサイズよりも小さくするか、データセットを増やしたり、検証を無効にしてください。 |
| GGUFのエクスポートが最初に失敗する | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]` を実行します。Windowsでは、Visual C++ Build Tools + CMake も必要です。 |

## バグの報告

問題が発生した場合、Backpropagateは起動時に `run_started run_id=<uuid>` という行を出力し、同じIDをチェックポイントのマニフェスト、SLAOのマージ履歴、および構造化されたログ行に関連付けます。バグ報告には、`run_id` を必ず含めてください。これにより、開発者が、その実行に関するすべてのログ行、すべてのチェックポイント、およびすべてのマージを関連付けることができます。

良いバグレポートには、以下の情報が含まれている必要があります。

1. **`run_id`** — 起動時に表示されるUUID（`TrainingRun.run_id` および `RunResult.run_id` でも利用可能）。
2. **エラーコード** — `stderr` に表示される `[CODE_NAME]: message` の行を検索してください。詳細については、[エラーコード](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) を参照してください。
3. **コマンドライン（個人情報削除済み）。** 詳細表示モードでない場合、`stderr` は自動的に個人情報が削除されます（Bearerトークン、`sk-*`、`hf_*`、AWSキー、`password=`/`token=`/`api_key=` のペアは削除されます）。貼り付けても安全です。完全な未加工のトレースバックを表示するには、`--verbose` オプションで再実行しますが、投稿する前に内容を確認してください。
4. **Python / PyTorchのバージョン、GPUモデル、OS。** `backprop info` コマンドで、これらすべてを一度に表示できます。

## プライバシー

トレーニングはすべて、ローカルのGPUで行われます。Backpropagateは、HuggingFaceからモデルをダウンロードするためにネットワークリクエストを行う場合を除き、ネットワークリクエストは行いません。テレメトリーやクラウドへの依存はありません。

## スコアカード

| カテゴリ | スコア | 備考 |
|----------|-------|-------|
| A. セキュリティ | 6/8 | SECURITY.md、信頼できるモデル、秘密情報/テレメトリーなし、safe_path()。MCP関連項目はスキップされます。 |
| B. エラー処理 | 5/7 | 構造化された例外情報（`code`/`message`/`hint`/`cause`/`retryable`）は、ERROR_CODES レジストリを通じて提供されます。CLIの終了コードは0/1/2/3です。`--verbose`オプションなしでは、生のスタックトレースは表示されません。`run_id`による関連付け、内容が一部伏せられた標準エラー出力、`--share`と`--auth`の組み合わせによる制限。MCP、デスクトップ版、VS Codeは対象外です。 |
| C. オペレーター向けドキュメント | 4/7 | README、CHANGELOG、LICENSE、`--help`。ロギング、MCP、複雑な機能は対象外。 |
| D. リリース時の品質管理 | 6/9 | `verify.sh`、バージョンはタグ、CI環境での5つのスキャナー、dependabot、`python_requires`、クリーンなビルド。 |
| E. アイデンティティ | 4/4 | ロゴ、翻訳、ランディングページ、メタデータ。 |
| **Total** | **25/31** | 14項目は理由によりスキップされました。`shipcheck audit`は100%合格。監査日: 2026年5月21日（B行は、ステージBとステージAのCLI終了コード関連の作業後に再評価）。 |

設計の経緯と、各項目が何に対応しているかについては、[ROADMAP.md](ROADMAP.md)を参照してください。v1.1.0では、Week 1～4のすべての項目がリリースされます。

## ライセンス

MIT — 詳細については、[LICENSE](LICENSE)を参照してください。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
