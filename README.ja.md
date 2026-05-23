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

**わずか3行のコードでLLMをファインチューニング。最適なデフォルト設定、VRAMを考慮したバッチサイズ、複数回の学習によるSLAO、そしてOllamaへのワンクリックGGUFエクスポート機能。**

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

このリポジトリには、`examples/quickstart.jsonl`という小さなファイル（ShareGPT形式の5つの例）が含まれており、これにより、上記のコードスニペットをクリーンな環境でエンドツーエンドで実行できます。ご自身の学習を行う場合は、以下の「データセット形式」を参照してください。

### コード不要：Web UI

Python REPLではなく、UIを使用したいですか？同じ追加機能をインストールして実行してください。

```bash
pip install backpropagate[standard]
backprop ui --port 7862
```

Reflex (Radix UI)インターフェースを使用すると、JSONLファイルを選択し、モデルを選択し、学習を実行し、エクスポートすることができます。Pythonの知識は不要です。このUIはローカル環境での使用を前提としており、インターネット経由での公開については、以下の「Web UI」を参照してください。セキュリティ契約（`--share` + `--auth`）と、サポートされているトンネルオプション（Cloudflare Tunnel、ngrok）について説明しています。

## データセット形式

JSONL形式の学習ファイルは、各行に1つの例が記述されている必要があります。最もシンプルな形式は、ShareGPTのチャット形式です。

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca（`instruction`/`output`）、OpenAIのチャット（`messages`）、および生のテキスト形式もサポートされています。`examples/quickstart.jsonl`には、コピーしてすぐに使えるサンプルが含まれています。

## なぜバックプロパゲーションを行うのか？

| 問題点 | 解決策 |
|---------|----------|
| ファインチューニングは複雑 | 3行で完了：ロード、学習、保存 |
| Windowsは使い物にならない | Windowsへの完全なサポート |
| VRAM管理は難しい | 自動バッチサイズ調整、GPU監視 |
| モデルのエクスポートは混乱を招く | ワンクリックでGGUF形式にエクスポートし、Ollamaに自動登録 |
| 長時間の学習は忘却を引き起こす | 複数回の学習によるSLAO |

## 主な機能

- **設計上、ヘッドレス**: CI/CDパイプライン、自動化されたワークフロー、およびプログラムによる実行向けに設計されています。
- **最適なデフォルト設定**: ハードウェアとデータセットに基づいて、最適なハイパーパラメータを自動的に設定します。
- **複数回の学習によるSLAO**: 長時間の学習中に発生する破滅的な忘却を防ぐための高度な学習戦略。
- **Windowsへの完全なサポート**: Windows環境で動作するようにテストおよび最適化されており、一般的なPyTorch/CUDAの問題を回避します。
- **シームレスなエクスポート**: ワンクリックでGGUF形式にエクスポートし、Ollamaに自動登録します。
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
| `ui` | Reflex (Radix UI)のWebインターフェース | reflex>=0.9.2, fastapi>=0.115 |
| `validation` | Pydanticによる設定検証 | pydantic, pydantic-settings |
| `export` | Ollama用のGGUFエクスポート | llama-cpp-python |
| `monitoring` | WandB + システム監視（v1.1.0でトレーナーに自動的に組み込み） | wandb, psutil |
| `logging` | 構造化されたロギング | structlog |
| `security` | JWT認証 + トークン生成 | PyJWT, cryptography |
| `production` | unsloth + ui + 検証 + ロギング + セキュリティ | （バンドル） |

**要件:** Python 3.10以上、CUDA対応GPU（8GB以上のVRAM）、PyTorch 2.0以上

### プラットフォームの前提条件

Backpropagateは、実行時の問題を解決します（マルチプロセッシング、RTX 40/50でのxformers、Windowsでのデータローダー）。ただし、インストール時のプラットフォームの問題は解決しません。それらの問題は事前に解決してください。

- **CUDA ツールキットのバージョン。** PyTorch は CUDA に合わせてリリースされます。誤った wheel を選択すると、CPU のみで動作する torch がインストールされてしまいます。お使いのドライバに合った正確な `pip install torch ...` コマンドについては、<https://pytorch.org/get-started/locally/> の選択ツールをご利用ください。`nvidia-smi` コマンドを実行して、ドライバまたは CUDA のバージョンを確認してください。
- **Windows。** `[export]` オプションを使用する場合（`llama-cpp-python` はソースコードからビルドされます）、Visual Studio Build Tools (C++) と CMake が必要です。`bitsandbytes` wheel は、Windows ネイティブで利用できるようになりました（>= 0.43）。古いドキュメントで `bitsandbytes-windows` が言及されている場合は、最新の情報ではありません。
- **macOS。** GPU を使用した学習は**サポートされていません**。CUDA は利用できません。エクスポートされた GGUF モデルを Ollama を介して *推論* 実行するには、Backpropagate をインストールできますが、`trainer.train()` を実行すると `DEP_GPU_NOT_AVAILABLE` エラーが発生します。学習には CUDA 対応の環境を使用してください。
- **Linux。** ほとんどのディストリビューションで問題なく動作します。PyPI からのバイナリリリースを使用している場合は、Linux のビルドでは CPU のみで動作する torch が使用されていることに注意してください（GitHub のリリースアセットの 2GB 制限を超えるため）。最初に、pytorch.org から対応する CUDA wheel をインストールしてください。

詳細なインストールに関するトラブルシューティングについては、[トラブルシューティングガイド](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) を参照してください。

## 設定

すべての設定は、`BACKPROPAGATE_` というプレフィックスが付いた環境変数を使用して上書きできます（例：`BACKPROPAGATE_LOG_LEVEL=debug`）。プロジェクトのルートディレクトリにある `.env` ファイルは、`[validation]` オプションがインストールされている場合に自動的に読み込まれます。

一般的な設定項目（詳細については、[環境変数リファレンス](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/) を参照してください）。

| 変数 | デフォルト値 | 備考 |
|----------|---------|-------|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | JSON ログを強制 (`true`) するか、コンソールログ (`false`) を使用するか |
| `BACKPROPAGATE_LOG_FILE` | 未設定 | ログを保存するパス |
| `BACKPROPAGATE_DEFER_FEATURE_DETECTION` | 未設定 | 起動時のオプション依存関係の検出をスキップし、CLI の起動を高速化 |
| `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE` | `true` | `true` の場合、`--auth` オプションなしでの `backprop ui --share` を拒否 |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | すべての UI ファイルシステムの書き込みのサンドボックスベース。許可リストで検証済み |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | デフォルトモデル |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | 学習率 |
| `BACKPROPAGATE_LORA__R` | `16` | LoRA のランク |

ネストされたキーでは、区切り文字として二重アンダースコアを使用します（Pydantic の `env_nested_delimiter` 規則）。

## 使い方

### 基本的な学習

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

`Qwen/Qwen2.5-7B-Instruct` がデフォルトです。`Trainer()` を呼び出す際にモデル引数を指定しない場合、この値が適用されます（`config.py` の `ModelConfig.name` を参照）。以前の例では、事前量子化された `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` が使用されていましたが、より安定性の高い公式の Qwen モデルをデフォルトに変更しました（[CHANGELOG v1.1.0](CHANGELOG.md#110---2026-05-21)）。どちらのモデルでも動作します。

### マルチラン SLAO 学習

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

SLAO (Single LoRA Continual Learning via Asymmetric Merging) は、[Merge before Forget](https://arxiv.org/abs/2512.23017) という論文を実装しています。QR 分解による直交 A 行列の初期化、非対称な A/B 行列の処理、および時間依存の `λ(i) = 1/√i` スケーリングが含まれます。CLI フラグは `--samples` で、対応するフィールドは `samples_per_run` です。

### Ollama へのエクスポート

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

すべてのサブコマンドとフラグについては、[CLI リファレンス](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) を参照するか、`backprop <サブコマンド> --help` を実行してください。

### チェックポイントからの再開 (v1.1.0)

4 回の実行でクラッシュした 5 回のマルチランが、現在では復旧可能です。すべてのマルチランセッションでは、`run_id` が `run_history.json` とオンディスクのチェックポイントマニフェストの両方に書き込まれるため、中断した場所から再開するには、次のコマンドを実行します。

```bash
backprop resume <run-id>                       # picks up the in-progress session
backprop multi-run --data ... --resume <run-id> # explicit form
backprop train --data ... --resume <run-id>    # single-run resume (continues run_id)
```

`backprop multi-run` のデフォルト動作（`--resume` オプションなし）では、同じ出力ディレクトリで実行中のジョブを自動的に検出し、それを継続します。クリーンなセッションを開始するには、`resume_from="off"`（Python API）を指定するか、`--resume` オプションを省略して、新しい出力ディレクトリで実行してください。

マルチラン再開時には、その `run_id` に対応する最新のチェックポイントがモデルにロードされ、チェックポイントの隣にある `slao/` ディレクトリから SLAO のマージ状態が復元され、実行ループが `last_completed_run + 1` から継続されます。履歴エントリの `status` が `running` に戻るため、`backprop list-runs --status running` コマンドで実行中のセッションが表示されます。

### 実験の追跡機能（v1.1.0）

`Trainer` は、インストールされている実験追跡ツール (`wandb`, `tensorboard`, `mlflow`) を自動的に検出し、それらを基盤となる `transformers.TrainingArguments` に統合します。デフォルトでは `report_to="auto"` が設定されており、インポート可能なものがすべて利用されます。

```bash
pip install backpropagate[monitoring]  # installs wandb + psutil
wandb login                            # one-time
backprop train --data my_data.jsonl    # W&B run gets the same run_id prefix as the on-disk history
```

明示的に無効にするには、`Trainer(report_to=["wandb"])`、`Trainer(report_to=["tensorboard"])`、または `Trainer(report_to="none")` を使用します。MLflow を使用するには `pip install mlflow` を、TensorBoard を使用するには `pip install tensorboard` を実行してください。W&B の実行名は `backprop-<run_id_prefix>` なので、オペレーターは同じ識別子で W&B、ログ、および `run_history.json` を検索できます。

### トレーニング履歴

`backprop train` および `backprop multi-run` の実行ごとに、`<output>/run_history.json` に、実行 ID、モデル、データセット、ハイパーパラメータ、ステータス、最終的な損失、損失履歴、および（マルチランの場合）SLAO のマージのタイムラインが記録されます。最近の実行の一覧を表示するには、以下のコマンドを使用します。

```bash
backprop list-runs                         # most recent 20 runs, all statuses
backprop list-runs --status failed         # filter
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial run_id ok)
```

実行履歴はプロセスを越えて保持されます。Web UI の「Runs」タブは、メモリ内のビューであり、オンディスクの履歴が `list-runs` / `show-run` / `resume` のための信頼できる情報源です。

### Web UI

ローカルで Reflex インターフェースを起動します。

```bash
backprop ui --port 7862
```

パブリックインターネット経由でアクセスできるようにするには、`--share` オプションを `--auth` オプションと組み合わせて使用する必要があります。

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` コマンドを `--auth` オプションなしで実行すると、エラーコード `1` が発生し、構造化されたエラーメッセージ `[RUNTIME_UI_AUTH_NOT_ENFORCED]` が表示されます。これは、`--share` オプションが公開URLを生成し、認証なしで利用可能にすると、インターネット上の誰でもトレーニングパイプラインを操作できてしまうためです。この動作を回避するオプションはありません。認証情報を設定したくない場合は、SSHのポートフォワード機能を使用してください。具体的には、`ssh -L 7860:localhost:7860 <ホスト>` コマンドを実行し、その後、`http://localhost:7860` をローカルで開きます。詳細なセキュリティに関する情報は、[handbook/security.md](site/src/content/docs/handbook/security.md) を参照してください。

UI からのファイルシステムへの書き込みは、単一のディレクトリにサンドボックス化されています。

- デフォルト: `~/.backpropagate/ui-outputs`
- 上書き: `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- 上書きは、許可リストで検証されます。システムパスや認証情報パス（`/etc`, `/var`, `~/.ssh`, `~/.aws`, `C:\Windows\System32` など）は、`[UI_OUTPUT_DIR_FORBIDDEN]` というエラーで拒否されます。

## Windows サポート

Backpropagate は、Windows での動作を前提として設計されています。

- マルチプロセッシングのクラッシュを回避するための事前トークン化
- RTX 40/50 シリーズ向け xformers の自動無効化
- 安全なデータローダー設定
- RTX 5080 (16GB VRAM) での動作確認済み

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
└── ui_security.py       # Rate limiting, CSRF, file validation (framework-agnostic)
```

v1.0 の Gradio 実装 (`ui_gradio_legacy.py` + `theme_gradio_legacy.py`) は、v1.1.x まで参照として保持され、v1.2.0 で削除されました。

## トラブルシューティング

よく発生する初期エラーの簡単な一覧です。詳細な逆引き索引は、[トラブルシューティングハンドブックのページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) にあります。以下に示すすべてのコードは、[エラーコード](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) に詳しく説明されています。

| 症状 | コード | 対処法 |
|---------|------|-----|
| GPUメモリ不足によるトレーニング中断 | `RUNTIME_GPU_OOM` | OOM自動復旧 (B-002) が、バッチサイズを最大3回まで自動的に半分にします。無効にするには、`Trainer(oom_recovery=False)` を指定します。バッチサイズを強制的に小さくするには、`--batch-size 1` を指定します。 |
| HF Hub から 401 エラー / "モデルが見つかりません" というエラー | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` を実行し、再度試してください。入力ミスがある場合は、<https://huggingface.co/models> から正確な ID をコピーしてください。 |
| モデル名の入力ミス | `INPUT_VALIDATION_FAILED` または `DEP_MODEL_LOAD_FAILED` | <https://huggingface.co/models> での `org/name` の識別子を確認してください。 |
| `register_with_ollama` への接続拒否 | `DEP_OLLAMA_REGISTRATION_FAILED` | デーモンを起動します: `ollama serve`。 <https://ollama.com> からインストールしてください。再試行可能です。 |
| チェックポイント保存中にディスク容量不足 | `STATE_CHECKPOINT_INVALID` | クラッシュ時に、`.partial` というディレクトリが作成されますが、削除しても問題ありません。以前の正常なチェックポイントは保持されています。 |
| GPUの過熱によるトレーニングの一時停止/中断 | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | B-003: NVMLの温度閾値を超えるとモニタが一時停止し、GPUが冷却されると自動的に再開されます。エアフローを改善するか、GPUへの負荷を軽減してください。 |
| `backprop ui --share` が拒否される | `INPUT_AUTH_REQUIRED` | `--auth user:password` を指定するか、`BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false` を設定して、認証を無効にします（警告が表示されます）。 |
| 複数回の実行における "validation overlap" (検証データの重複) | `CONFIG_INVALID` (Stage A backend B-001) | `--samples` の値を、トレーニングプールサイズよりも小さくするか、データセットを増やすか、検証を無効にしてください。 |
| GGUF エクスポートが初回試行で失敗 | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]` を実行してください。Windows の場合は、Visual C++ Build Tools + CMake も必要です。 |

## バグの報告

エラーが発生した場合、Backpropagate は起動時に `run_started run_id=<uuid>` という行を出力し、同じ ID をチェックポイントのマニフェスト、SLAOのマージ履歴、および構造化されたログ行に関連付けます。バグ報告には、`run_id` を必ず含めてください。これにより、開発者が、その特定の実行に関するすべてのログ行、すべてのチェックポイント、およびすべてのマージを関連付けることができます。

良いバグレポートには、以下の情報が含まれている必要があります。

1. **`run_id`** — 起動時に表示される UUID (また、`TrainingRun.run_id` および `RunResult.run_id` でも利用可能です)。
2. **エラーコード** — `stderr` に表示される `[CODE_NAME]: message` の行を検索してください。詳細については、[エラーコード](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) を参照してください。
3. **コマンドライン (一部情報を削除したもの)。** 詳細表示モードでない場合、`stderr` は自動的に情報が削除されます (Bearerトークン、`sk-*`、`hf_*`、AWSキー、`password=`/`token=`/`api_key=` のペアは削除されます)。貼り付けても安全です。完全なトレースバックを表示するには、`--verbose` オプションで再実行しますが、投稿する前に内容を確認してください。
4. **Python / PyTorch のバージョン、GPU のモデル、OS。** `backprop info` コマンドで、これらすべてを表示できます。

## プライバシー

トレーニングはすべて、ローカルの GPU 上で行われます。Backpropagate は、HuggingFace からモデルをダウンロードするために必要なネットワークリクエストを除き、他のネットワークリクエストは行いません。テレメトリーやクラウドへの依存はありません。

## スコアカード

| カテゴリ | スコア | 備考 |
|----------|-------|-------|
| A. セキュリティ | 6/8 | SECURITY.md、信頼できるモデル、秘密情報/テレメトリーなし、safe_path()。MCP の項目はスキップされます。 |
| B. エラー処理 | 5/7 | 構造化された例外情報（`code`/`message`/`hint`/`cause`/`retryable`）は、ERROR_CODES レジストリを通じて提供されます。CLIの終了コードは0/1/2/3です。`--verbose`オプションなしでは、生のスタックトレースは表示されません。`run_id`による関連付け、内容が一部隠蔽された標準エラー出力、`--share`と`--auth`の組み合わせによる制限。MCP、デスクトップ版、Visual Studio Codeは対象外です。 |
| C. オペレーター向けドキュメント | 4/7 | README、CHANGELOG、LICENSE、`--help`。ロギング、MCP、複雑な機能は対象外。 |
| D. リリース時の品質管理 | 6/9 | `verify.sh`、バージョン=タグ、CI環境での5つのスキャナー、dependabot、`python_requires`、クリーンなビルド。 |
| E. アイデンティティ | 4/4 | ロゴ、翻訳、ランディングページ、メタデータ。 |
| **Total** | **25/31** | 14項目は理由によりスキップされました。`shipcheck audit`は100%合格。監査日: 2026年5月21日（B行は、ステージBとステージAのCLI終了コードの作業後に再評価）。 |

設計の経緯と、各項目が何に対応しているかについては、[ROADMAP.md](ROADMAP.md)を参照してください。v1.1.0では、Week 1～4のすべての項目がリリースされます。

## ライセンス

MIT — 詳細については、[LICENSE](LICENSE)を参照してください。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
