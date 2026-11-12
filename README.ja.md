<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

# 320億パラメータのQLoRAモデル、または70億パラメータのエンドツーエンドモデルを、1つのGPUで微調整します。そして、Ollamaにデプロイします

Backpropagateは、実際に使用しているGPUに合わせてサイズ調整された、**単一の**GPUで大規模言語モデルの微調整を行います。わずか3行のPythonコードQLoRAで、7B～32Bのモデルを1つの32GBのコンシューマー向けGPU（RTX 5090）で実行できます。1つのフラグ`--full-ft-offload`を使用すると、モデルの重みと勾配をホストRAM（LinuxまたはWSL2）に保持することで、7Bクラスのモデルを完全に微調整できます（速度は遅く、後述します）。さらに1つのコマンドでOllamaにエクスポートし、その後`ollama run`で微調整したモデルを使用します。16GBまでスケールダウンできます。Windows環境でも最適に動作します。Pythonの代わりにブラウザを使用したいですか？`backprop ui`を使用すれば、コードなしでこれらすべてを実行できます（[ツアーはこちら](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/)）。

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

以上です。YAML設定ファイルはありません。`accelerate launch`の複雑な手順もありません。「GGUFに変換する」という別のチュートリアルもありません。CUDA GPUと、トレーニングデータを含むJSONLファイルがあれば、すぐに動作する微調整モデルを3行のコードで作成できます。

## インストール

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

オプションの機能が必要な場合は、次のいずれかのインストールに置き換えてください。

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

Docker をお使いですか？ `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` でも動作します。 `linux/amd64` と `linux/arm64` の両方のイメージが提供されるため、Apple Silicon および ARM Linux 環境でもネイティブなイメージを使用できます。 「コンテナ内の UI」の標準的な `compose.yaml` は、リポジトリのルートにあります。 `user:password` をその隣の `ui-auth.txt` に配置し、`docker compose up` を実行して、`http://127.0.0.1:7860` にログインします（初回起動時にフロントエンドがビルドされ、1 ～ 2 分かかります）。実行履歴は `~/.backpropagate` に保存されます。

## Backpropagateがどのような位置にあるか

LLMの微調整には、いくつかの優れたライブラリがあります。それぞれ異なる点で優れています。

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**：YAML設定を好み、コピーできるレシピのコミュニティが欲しい場合に。
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**：DPO/PPO/RLHFとWeb GUIが必要な場合に。
- **[Unsloth](https://github.com/unslothai/unsloth)**：可能な限り最速のトレーニングが必要で、サポートされているモデルファミリーを使用している場合に。
- **[torchtune](https://github.com/pytorch/torchtune)**：Metaの公式のPyTorchネイティブのレシピを編集したい場合に。

Backpropagateは、不足しているオプションです。**1つのコンシューマーGPUでアダプターをトレーニングし、その結果をエクスポートするための、3行のPython APIです。** YAMLはなく、オンラインRL（PPO/GRPO）も、マルチノードもありません。コードを書きたくない場合は、同じループを実行するためのブラウザUIもあります。必要なループと、邪魔になるエクスポートステップだけです。

上記のライブラリのいずれかを試して、設定ファイルの複雑さに戸惑ったり、モデルファミリーの制限に遭遇したり、Windowsを優先するデフォルト設定が必要になった場合は、Backpropagateが最適です。

## 1つのGPUで微調整できるモデル

Backpropagateは、実行をカードのサイズに合わせて調整します。以下は、32GBのRTX 5090で測定された数値です。2026-10-03のQLoRA行（証拠：[`docs/receipts/2026-10-03-presets/`](docs/receipts/2026-10-03-presets/)）、2026-09-30の完全な微調整行（証拠：[`docs/receipts/2026-09-30-offload/`](docs/receipts/2026-09-30-offload/)）。QLoRAのピークは、設定の最大のコンテキストウィンドウでバッチ1の場合であり、その設定では最悪の場合です。短い例では、より少ないリソースを使用します。

| モデル | 方法 | 32GBのカードで測定 |
|---|---|---|
| **14B** (Qwen2.5-14B) | QLoRA | **18.7 GiB**、4096コンテキスト時のピーク（20.0 GiB予約済み）。 |
| 24B (Mistral-Small-24B) | QLoRA | 22.8 GiB、4096コンテキスト時のピーク（24.2 GiB予約済み）。 |
| **32B** (Qwen2.5-32B) | QLoRA | **適合:** 26.0 GiB、2048コンテキスト時のピーク（27.2 GiB予約済み、約4 GiBの余裕あり）。 |
| 3B | `mode="full"`（GPU上での完全な微調整） | **22.0 GiB**、ピーク（システム全体）、バッチ4、512コンテキストで0.30秒/ステップ。そのうち7.5 GiBはページングされたオプティマイザーの状態であり、より小さいカードではホストRAMにスピルする可能性があります（未テスト）。 |
| **7Bクラス** (Qwen2.5-7B、7.6Bパラメータ) | `mode="full" --full-ft-offload` | **トレーニング:** 5.3 GiB VRAM、**30.8 GiBホストRAM**（32.2 GiB保存時）、**14.7秒/ステップ**。LinuxまたはWSL2のみ。 |

このセッションでは再測定していません：7B QLoRA、Llama-3.1-8B（ゲートされたリポジトリ、テストマシンにトークンなし）、および3Bを超える純粋なGPUによる完全な微調整。それらの数値は、ドキュメントの他の場所に記載されており、概算です。

ほとんどのシングルGPUライブラリが、**24〜32B QLoRA**と**単一カードの7Bクラスの完全な微調整**のために、他の場所に誘導するのに対し、Backpropagateは1つのコンシューマーカードでこれらを行い、その結果をOllamaに直接エクスポートします。

**完全な微調整には2つの方法があります。** オフロードを使用しない場合、モデル、その勾配、およびオプティマイザーの状態はすべてGPUに配置されます。ライブラリは、検出されたVRAMによってモデルサイズを制限します（**16GB → 4B、24GB → 5B、32GB → 6B**）。これらの制限は、メモリ計算から得られ、3Bまでしか測定されていません。`--full-ft-ceiling-billions`でオーバーライドします。

`--full-ft-offload`は、重みと勾配をホストRAMに保持し、GPUにストリーミングします（FSDP2 CPUオフロード）。測定されたコストは次のとおりです。

- **ホストRAM:** 適合性チェックでは、10億パラメータあたり約3.7 GiBと、それに加えて10 GiBが必要とされます。これは控えめな設定です（7.6Bの場合、測定値は32.2 GiBですが、ここでは39 GiBを使用）。もしマシンがそれを処理できない場合、実行は事前に拒否されます。7.6Bモデルは、28GB WSL2のメモリ制限を超えるため、約5Bが現実的な上限となります。
- **速度:** 7.6Bの場合、1ステップあたり14.7秒（バッチサイズ1）、3Bの場合、1ステップあたり5.1秒（バッチサイズ4）。GPU上でバッチサイズ4の3Bモデルを使用した場合、1ステップあたり約0.63秒です。モデルがGPUなしでは実行できない場合にのみ使用してください。より高速なバージョンを計画中です。
- **オプティマイザー:** Adafactorを使用し、AdamWは使用しません。重みはbf16に保持され、各更新は確率的丸めを使用して書き戻されます。したがって、fp32のコピーは発生しません。
- **品質:** 3Bモデルを1回実行（150ステップ、ホールドアウト損失、1つのシード）した結果、通常の完全なファインチューニングで得られた改善の約85%に達しました（2.45 → 1.93、対照的に通常の完全なファインチューニングでは2.45 → 1.84）。1つのシードだけでは、ベンチマークとして適切ではありません。
- **範囲:** 単純な教師ありファインチューニング。パッキング、応答のみのマスキング、中間チェックポイント、再開は行いません。LinuxまたはWSL2のみ（FSDP2にはNCCLが必要です）。Windowsネイティブ環境では、`DEP_FSDP_UNAVAILABLE`が発生して停止します。
- **まだテストされていません:** 長時間の実行、1を超える勾配累積、および物理的な64GBマシン（テストマシンはより多くのRAMを備えており、テストでは60 GiBの制限が適用されました）。

モデルが適合しない場合、`RUNTIME_FULL_FT_MODEL_TOO_LARGE`で終了し、解決策を示します。[完全な微調整ハンドブックのページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/)を参照してください。

### 16GBにスケールダウン

16GBの範囲（RTX 4080 / 5080 / 4070 Ti Super）は、依然として最高レベルです：7B QLoRA（アダプターのサイズは、16GBカードに適合するように選択されます。256のランクには約17GBが必要です）、そして約3Bモデルの真の完全なファインチューニング（SmolLM3-3B、Qwen2.5-3B、Llama-3.2-3B/1B）を`mode="full"`経由で行います（32GBカードで3Bの場合、22.0 GiBが測定され、そのうち7.5 GiBはホストRAMにスピルできるページングされたオプティマイザーの状態です。16GBカードで許容範囲内で動作するかどうかはテストされていません）。`--full-ft-offload`を使用すると、GPUははるかに少ないメモリを保持します。テストカードでVRAMが制限されている場合、6 GiBの制限下でトレーニングされた3Bモデル、および8 GiBの制限下でトレーニングされた4Bおよび7.6Bモデルです。これらは32GBカードでのエミュレートされた制限であり、実際の8GBハードウェアでの実行ではありません。同じコードが、検出されたカードに適合するようにバッチサイズと上限を選択します。

2ビット量子化（AQLM / QuIP#）は**対象外**です。2ビットのベースを、完全精度ウェイトにクリーンにマージすることはできません。これは、マージ可能なアダプター→GGUF→Ollamaエクスポートの契約を破ります（パイプライン全体の目的です）。代わりに、Backpropagateが提供するヘッドルームレバーは、QLoRA、`mode="full"`、`--full-ft-offload`、およびFP8の計算パス（`--fp8`、Blackwell/Hopper）です。これらはすべてマージ可能で、エクスポート可能です。

## Backpropagateが適さない用途

もし、以下の用途であれば、別のライブラリを使用した方が良い結果が得られます。Backpropagateは適切な選択肢ではなく、無理に動かそうとすると、適切なツールを使用するよりもコストがかかります。使い始める前にこのセクションを読んで、インストールと再試行のサイクルを回避してください。

- **130億以上のパラメータを持つモデルに対する完全なパラメータによるファインチューニング** — 32GBのGPUと70億程度のモデルで、最大約60億のパラメータまでを対象とした完全なファインチューニングをバックプロパゲーションで行います（`--full-ft-offload`、詳細については[この資料](#1つのgpuで微調整できるモデル)を参照）。130億以上のパラメータを持つモデルに対して完全なファインチューニングを行うには、複数のGPUを使用したFSDPまたはより大きなGPUが必要です。その計算リソースを使用する前に、両方の可能性を検討してください。[Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)の報告によると、すべてのレイヤーに適用し、データセットがアダプターの容量に収まる場合、完全なファインチューニングは、1回の処理あたりの計算量を約3分の2に抑えながら、LoRAに匹敵する結果をもたらします。[Biderman et al. 2024](https://arxiv.org/abs/2405.09673)の研究では、標準的な低ランク設定（LoRA）において、コードや数学の問題に対して完全なファインチューニングよりも大幅に性能が劣るものの、より少ない情報を忘れることがわかりました。指示に従う、特定のペルソナやスタイルを学習させる場合など、比較的小さなデータセットを使用する場合は、通常、320億までのモデルを1つのGPUで実行する方が効率的です（QLoRA）。
- **オンライン強化学習 — PPO / GRPO / RLVR** — バックプロパゲーションは、単一段階のSFTと参照を必要としない嗜好調整を行います（v1.5のORPO、v1.6のSimPO + KTO）。これによって行われないのは、報酬モデルまたは生成と評価のループを学習ステップに追加する必要がある、オンライン強化学習、つまりPPO、GRPO、またはRLVRです。それらの場合は、TRLを直接使用するか、LLaMA-Factoryを使用してください。（参照を必要としない嗜好調整は、単一段階の範囲に適合します。なぜなら、メモリに保持する必要がある別の参照モデルがないからです。詳細は、[クイックスタート](#クイックスタート)のORPOに関する注記を参照してください。）
- **複数ノードでのトレーニング** — 1つのマシンにGPUを1つだけ使用。1つのマシンに複数のGPUを使用することも可能ですが（`accelerate launch`を使用）、公式にはサポートされていません。
- **CUDA環境でのmacOSトレーニング** — Apple SiliconにはCUDAが搭載されていないため、CUDA環境でのトレーニングは、NVIDIA GPUを搭載したLinuxまたはWindowsマシンで行います。トレーニング済みのモデルは、Ollamaを通じてMacでも実行できます。**実験的で、検証されていないプレビュー版**のMLX環境（`--backend mlx`）は、Apple Silicon上でネイティブにLoRAアダプターをトレーニングします。詳細は[Apple Silicon (MLX)](#apple-silicon-mlx--検証されていないプレビュー)をご覧ください。これはLoRA-SFT専用であり、**実際のハードウェアでの検証は行われていません**（サポート対象外です）。したがって、単純なLoRA SFT（ORPO、完全なファインチューニング、FP8、複数回の実行）以上の処理を行う場合は、CUDA環境を使用することをお勧めします。
- **テスト対象のモデルファミリー以外のモデル** — Qwen 2.5 / 3.5（7B / 4B）、Phi-4-mini-3.8B、SmolLM3-3B、Llama 3.2（3B / 1B）、Mistral 7B。他のモデルでも動作する場合もありますが、CIで固定されていません。

これらのいずれかの機能が必要な場合は、上記のライブラリのいずれかを使用してください。それらのライブラリの方が適しています。

## Backpropagateが提供するもの

1つのインストールで、次の4つの機能を提供します。

**1. 実際の3行のAPIで、設定ファイルなしで実行できます。**
このREADMEの先頭にあるスニペットは、最初から最後まで実行されます。`accelerate config`、YAML、Hydraオーバーライドは不要です。単に`Trainer(model).train(data)`を実行するだけで、ファインチューニングが完了します。

**2. 実際に動作するWindowsサポート。**
ほとんどのMLライブラリは、Windowsを後回しにします。Backpropagateは、RTX 50シリーズカードを搭載したWindows 11で開発およびテストされています。このライブラリは、ランタイムの癖を処理します。Windowsのマルチプロセッシングがクラッシュしないように、データを事前にトークン化する方法を知っており、RTX 40/50カードで動作が停止するxformersを自動的に無効にし、動作が停止しないデータローダー設定を選択します。これらのことを知る必要はありません。単に動作します。

**3. 無人での実行用に構築されています。**
トレーニングには数時間かかります。見守る必要はありません。Backpropagateは、放置して実行できるように設計されています。

- GPUメモリが不足した場合、バッチサイズを自動的に半分にし、最大3回まで再試行します。手動での調整は不要です。
- GPUが過熱した場合、冷却されるまで一時停止し、その後続行します。
- すべてのチェックポイントはアトミックに書き込まれます。ラップトップが保存中にクラッシュした場合でも、以前の良好なチェックポイントはそのまま残ります。
- すべてのトレーニング実行には、一意のIDが割り当てられ、すべてのログ行、すべてのチェックポイント、およびすべてのWeights & Biasesエントリにスタンプが付けられます。問題が発生した場合、1つのIDを使用すると、メンテナーがすべてを関連付けることができます。
- エラーには安定したコード（`RUNTIME_GPU_OOM`、`DEP_OLLAMA_REGISTRATION_FAILED`など）が付属しているため、ログと[トラブルシューティングガイド](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)をgrepして修正を見つけることができます。CUDA固有の障害には、専用の[CUDAトラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)があります。

**4. トレーニングされたアダプターから`ollama run`への1つのコマンド。**
多くのライブラリがモデルをトレーニングします。しかし、実際に使用したいときに、邪魔にならないようにするライブラリはほとんどありません。Backpropagateは、GGUF（Ollamaが使用する形式）にエクスポートし、1つのコマンドでOllamaモデルを登録します。トレーニングが完了してから、「自分のファインチューニングとチャットできる」状態になるまで、約30秒かかります。

## クイックスタート

コマンドラインから、5つの会話を含むサンプルデータセットを使用します。

```bash
pipx install "backpropagate[standard]"
curl -LO https://raw.githubusercontent.com/mcp-tool-shop-org/backpropagate/main/examples/quickstart.jsonl

backprop train --data quickstart.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 10
backprop generate ./output "What is Python?"      # did it learn anything?
backprop export ./output --format gguf --quantization q4_k_m --ollama --ollama-name my-first-finetune
ollama run my-first-finetune
```

`backprop train` は `./output` にアダプターを書き込みます（`--output` で変更します）。Python では同じことは次のようになります。

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")
```

Python API には、`pip install "backpropagate[standard]"` を使用した仮想環境を使用してください。`pipx` は、独自の環境に `backprop` コマンドをインストールするため、`import backpropagate` はそれを検出できません。

**GGUF エクスポートに必要なもの。** エクスポートは、アダプターをベースモデルにマージし、llama.cpp のコンバータースクリプトで変換します。次のいずれかが必要です。

- llama.cpp の **ソースコードのチェックアウト** (`git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp`) と、同じ環境に `pip install sentencepiece protobuf` をインストールするか、
- 独自の llama.cpp がすでにビルドされている Unsloth を使用します。

`--ollama` を使用すると、`q4_k_m` の量子化は `ollama create` によって行われるため、コンパイルする必要はありません。Backpropagate は、llama.cpp をビルドするためにシステムパッケージをインストールすることを許可しません。許可する場合は、`BACKPROPAGATE_UNSLOTH_AUTO_INSTALL=1` を設定してください。詳細: [エクスポート](https://mcp-tool-shop-org.github.io/backpropagate/handbook/export/)。

独自のデータの場合、JSONL を 1 行に 1 つの例としてフォーマットします。

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca (`instruction` / `output`)、OpenAI チャット (`messages`)、および生のテキスト形式も使用できます。Backpropagate は形式を自動的に検出します。

### ループ: データの確認、トレーニング、評価、エクスポート

```bash
backprop data report my_data.jsonl                     # duplicates, length outliers, format problems
backprop data split my_data.jsonl --heldout-ratio 0.1  # a held-out set the model never trains on
backprop train --data my_data.train.jsonl --steps 200 --output ./run-a
backprop eval <run-id> --heldout my_data.heldout.jsonl # held-out loss + sample generations
backprop eval <run-b> --vs <run-a>                     # did the change help?
backprop export ./run-a --format gguf --ollama --ollama-name my-model
```

評価は、設計上、判断なしで行われます。保留された損失と決定的なタスクメトリック (`normalized_exact_match`、`token_f1`、`contains`、`regex`、`pass_rate`) を使用します。LLM ジャッジを使用するには、それを `backprop generate` の出力に対して実行してください。詳細: [レシピ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/)。

### プリファレンスチューニング (ORPO、SimPO、KTO)

単純なデモンストレーションではなく、プリファレンスでトレーニングします。ORPO は参照不要で、単一ステージです。プリファレンス信号を SFT ステップに組み込むため、個別の報酬モデルや参照モデルは必要なく、3 行の形状は変更されません。`--method orpo` (CLI) または `method="orpo"` (Python) を渡し、`{prompt, chosen, rejected}` (または `{chosen, rejected}` のみ) の行を含むデータセットを渡します。

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

デフォルトの学習率は、ORPO に対して自動的に `8e-6` に低下します (損失は単純な SFT よりも鋭くなります)。`--orpo-beta` (デフォルトは `0.1`) を調整して、オッズ比ペナルティの重みを設定します。ORPO は `mode="lora"` のみです。

**v1.6 の新機能 — SimPO と KTO。** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) は参照不要で、長さで正規化された報酬を使用し、ORPO と同じペアの `{prompt, chosen, rejected}` データを使用します (`--simpo-beta`、`--simpo-gamma`)。`--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) は、**ペアでない** `{prompt, completion, label}` データを使用します。つまり、キュレーションされた A/B ペアではない、各例に対する肯定/否定の評価です。望ましい/望ましくない損失の重みをラベルの数から自動的に調整します。どちらも `mode="lora"` のみであり、単一 GPU SFT の範囲内に留まります (個別の参照モデルはありません)。使用するものを選択するには、[プリファレンスチューニングハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) を参照してください。オンライン RL (PPO/GRPO) については、[Backpropagate が適さないもの](#backpropagateが適さない用途) を参照してください。

### 推論トレース SFT (R1 蒸留)

推論モデルを簡単に蒸留します。`--reasoning-trace` (CLI) または `Trainer(..., reasoning_trace=True)` (Python) を渡し、アシスタントの応答内に `<think>...</think>` の思考の連鎖を保持するトレースを渡します。これは、[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 蒸留の純粋な SFT の半分であり、RL は必要ありません。Backpropagate は、`<think>` をトレーニングターゲットに保持し、空の/長すぎるトレースを削除 (トレース長のフィルタリング)、および長い CoT に対してデフォルトの `max_seq_length` を 8192 に引き上げます。重要なのは、`<think>` は **プレーンテキスト** のままであることです。特別なトークンはなく、埋め込みのサイズを変更する必要もありません。そのため、マージされた GGUF は、他のファインチューンと同様に Ollama にエクスポートできます。SFT のみです。データセットの形状と調整可能なトークンバンドについては、[推論トレースレシピ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) を参照してください。

### Apple Silicon (MLX) — 検証されていないプレビュー

> ⚠️ **検証されていないプレビュー — サポートされている機能セットの一部ではありません。** MLX レールは構築され、ユニットテストされていますが、実際の Apple Silicon 上で **ドッグフード検証されていません** (`mlx-lm` は Apple 専用であり、Backpropagate が開発されている NVIDIA リグでは実行できません)。以下すべてを実験的なものとして扱い、ご自身の責任で使用し、M シリーズの Mac で実行した場合は [異常を報告](#バグの報告) してください。

**1 つの API、2 つのレール。** CUDA は、検証済みの標準的なバックエンドです。MLX は、Apple の [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) ツールチェーン (統合メモリ、CUDA なし) を介して M シリーズの Mac でトレーニングする 2 番目のレールです。3 行の形状は、ハードウェアによってレールを選択します。`backend='auto'` (デフォルト) は、NVIDIA 上の CUDA に、および Apple Silicon 上の MLX にルーティングするため、既存の CUDA リグはバイト単位で同一です。

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

MLX レールは **LoRA SFT のみ** です。ORPO、FP8、`mode='full'`、または複数回の実行はありません (それぞれ `CONFIG_INVALID_SETTING` で拒否されます。それらの場合は、NVIDIA ボックスで `backend='cuda'`/`'auto'` を使用してください)。結果として得られるアダプターは、プレーンな safetensors であり、CUDA レールと同じパスを通じて Ollama にエクスポートされます。SFT のみです。

> Apple 以外のホストで `--backend mlx` を強制すると、`CONFIG_INVALID_SETTING` でエラーが発生します。Mac で `mlx_lm` ツールチェーンが見つからない場合、`DEP_MLX_UNAVAILABLE` が発生します。

よりエンドツーエンドのワークフロー (ファインチューンして HF Hub にプッシュ、OOM 後に再開、長期間のキャンペーンにわたる複数回の SLAO など) については、[ハンドブックのレシピページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/) を参照してください。

### Web UI (オプション)

Python を入力する代わりにクリックしたい場合は、UI エクストラをインストールして起動します。

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

表示される URL を開きます。`http://127.0.0.1:7862/?token=...` (各起動で新しいトークンが作成されます。最初の起動ではフロントエンドが構築され、1 分または 2 分かかる場合があります)。これは、トレーニングのためのローカル Web インターフェイスです。実行、複数回のスイープ、またはエクスポートを開始し、リアルタイムで監視します (ステップ数、損失、残り時間、GPU 温度とメモリ)。停止すると、保存されたチェックポイントが作成されます。各ジョブは、独自のプロセスで、一度に 1 つずつ実行されます。ページをリロードすると、実行中のジョブが再開されます。データセットページには、ファイルの内容が表示され、クリーンなコピーが保存されます (繰り返しと空の例が削除され)、トレーニングフォームに渡されます。過去の実行と、Hugging Face キャッシュ内のモデルにはそれぞれ専用のページがあり、すべての設定には説明が記載された "i" があります。[Web UI ツアー](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/) を参照してください。UI はデフォルトではローカル専用です。他のデバイスからアクセスできるようにするには、[Web UI](#web-ui) の `--share` + `--auth` セキュリティ契約を参照してください。

## 複数回のトレーニング

複数のデータセットにわたって段階的にファインチューニングしたい場合 (たとえば、毎週新しいトレーニングデータを入手し、以前に学習したことを忘れることなく追加したい場合)、Backpropagate の `multi_run` モードが最適です。

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

これにより、5回のトレーニングパスが実行され、各パスの間にアダプターをマージすることで、以前の知識を保持しながら新しい例を組み込むことができます。この手法は、最近の継続学習研究に基づいています。詳細は、このREADMEの末尾にある[参考文献](#参考文献)を参照してください。

CLIバージョン：

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## チェックポイントから再開

5回のトレーニングのうち、4回目の実行でクラッシュが発生した場合でも、再開できます。複数の実行セッションでは、各セッションの実行IDがディスク上の履歴とチェックポイントマニフェストに書き込まれるため、中断したところから再開するには、次のコマンドを実行するだけです。

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

`backprop multi-run`（`--resume`なし）のデフォルトの動作では、同じ出力ディレクトリにある進行中のエントリを自動的に検出し、続行します。クリーンな状態から強制的に開始するには、新しい出力ディレクトリを指定します。

## トレーニング履歴

各`backprop train`および`backprop multi-run`の呼び出しは、`<output>/run_history.json`に1行のレコードを記録します。レコードには、使用されたモデル、データセット、ハイパーパラメータ、ステータス、最終的な損失、および損失履歴が含まれます。過去の実行を一覧表示して確認できます。

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## 実験の追跡

Backpropagateは、インストールされている実験追跡ツール（Weights & Biases、TensorBoard、MLflow）を自動的に検出し、それらを連携させます。`wandb`がインストールされており、ログインしている場合、各実行は自動的にW&Bにログを記録し、ディスク上の実行IDと一致する実行名が付けられます。これにより、W&B、ログ、および`run_history.json`全体で、1つの識別子を使用して検索できます。

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

`Trainer(report_to=["wandb"])`、`Trainer(report_to=["tensorboard"])`、または`Trainer(report_to="none")`を使用して、この機能を無効にすることができます。

## Web UI

ReflexのWebインターフェースは、オプションで有効にできます。`pipx install "backpropagate[ui]"`でインストールし、起動します。

```bash
backprop ui --port 7862
```

UIはローカルで実行されます。UIが出力するURLを開いてください（例：`http://127.0.0.1:7862/?token=...`）。`--auth`がない場合、起動するたびに新しいトークンが生成され、UIはそのトークンがないリクエストを拒否します。UIからは、データセットの確認とクリーニング、トレーニング（単一の実行または複数の実行）、実行のライブ監視、保存されたチェックポイントを使用した実行の停止、および結果のエクスポートを行うことができます。各ジョブは、独自のプロセスで、一度に1つずつ実行されます。UIを閉じると、ジョブも停止します。[Web UIツアー](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/)では、スクリーンショット付きで各ページを説明します。

他のデバイス（ネットワーク上の他のユーザー、パブリックURLなど）からアクセスできるようにするには、`--share`（または`--host`）を`--auth`とペアにする必要があります。

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share`は、`--auth`がない場合、エラーで終了します。その理由は、`--share`はインターネット上の誰でもアクセスできるURLを公開するため、認証がない場合、誰でもトレーニングパイプラインを制御し、あなたのHuggingFaceトークンを読み取ることができるからです。この機能を無効にするオプションはありません。資格情報を設定したくない場合は、代わりにSSHポートフォワーディングを使用してください。

```bash
# On the client:
ssh -L 7862:localhost:7862 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open the URL the server printed (http://127.0.0.1:7862/?token=...) locally
```

完全な脅威モデルについては、[handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/)を参照してください。

UIからのファイルシステムへの書き込みは、単一のディレクトリにサンドボックス化されます。

- デフォルト：`~/.backpropagate/ui-outputs`
- オーバーライド：`BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`を設定
- オーバーライドは、許可リストで検証されます。システムまたは資格情報パス（`/etc`、`~/.ssh`、`~/.aws`、`C:\Windows\System32`など）は拒否されます。

## プラットフォームに関する注意

**要件：** Python 3.11+ · CUDAを搭載したNVIDIA GPU · PyTorch 2.0以上。8GBのカードで、1Bから3Bのプリセットをトレーニングできます。16GBのカードで7Bを、32GBのカードで32Bまで、QLoRAを使用してトレーニングできます。

Python 3.10はサポートされなくなりました（EOLは2026年10月1日）。現在の最小バージョンは3.11です（v1.9.0時点）。Python 3.11は、最もテストされた最小バージョンです。

Backpropagateは、異なるプラットフォームでのトレーニングにおけるランタイムの癖に対処しますが、インストール時の問題を修正することはできません。最も一般的な問題は次の2つです。

- **CUDAホイールの間違い。** PyTorchは、CUDAバージョンごとに1つのバイナリとして公開されます。間違ったものを選択すると、CPUのみのPyTorchがサイレントに実行され、トレーニングは非常に遅くなります。ドライバーに適したホイールを<https://pytorch.org/get-started/locally/>で選択してください。`nvidia-smi`を実行して、ドライバー/CUDAバージョンを確認してください。
- **Windows + GGUFエクスポート。** `[export]`の追加ビルドは、ソースから`llama-cpp-python`をビルドします。これには、Visual Studio Build Tools（C++コンポーネント）とCMakeが必要です。

**macOS：** CUDAレールはサポートされていません（CUDAなし）。CUDAを使用するように設定された`trainer.train()`は、`DEP_GPU_NOT_AVAILABLE`を発生させます。トレーニングされたアダプターは、Ollamaを介してMacで実行できます。**実験的で、検証されていない**MLXレール（`--backend mlx`、`pip install 'backpropagate[mlx]'`）は、Apple Silicon上で`mlx_lm.lora`を介して、ネイティブにLoRAアダプターをトレーニングします。これは、LoRA SFTのみであり、**実際のシリコンでドッグフード検証されていません**（[Apple Silicon (MLX)](#apple-silicon-mlx--検証されていないプレビュー)を参照）。CUDAパスを使用する場合、またはORPO / 完全なファインチューニング / FP8 / 複数回の実行を行う場合は、CUDA LinuxまたはWindowsマシンを使用してください。

完全なインストール手順とトラブルシューティングガイドについては、[トラブルシューティングハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)を参照し、ドライバー/VRAM/xformers/bf16-vs-fp16の問題については、専用の[CUDAトラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)を参照してください。

## CLI

すべてのPython APIには、対応するCLIミラーがあります。

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

完全なリファレンスは、[CLIハンドブックページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/)または`backprop <subcommand> --help`にあります。

## 設定

すべての設定は、`BACKPROPAGATE_`プレフィックスを使用して環境変数でオーバーライドできます。

| 変数 | デフォルト | 備考 |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | JSONまたはコンソールログを強制 |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | デフォルトモデル |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | 学習率 |
| `BACKPROPAGATE_LORA__R` | `256` | LoRAランク。設定すると、自動アダプターサイズ選択が無効になります（[モデルプリセット](#モデルプリセット)の`--lora-preset`を参照）。 |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | UIファイルシステムサンドボックス |

ネストされたキーには、ダブルアンダースコア（`MODEL__NAME`、`MODEL_NAME`ではない）を使用します。完全なリファレンスは、[環境変数ハンドブックページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/)にあります。

## モデルプリセット

| プリセット | GPUメモリ | ライセンス | 備考 |
|---|---|---|---|
| Qwen-3.5-4B | 6 / 7 / 11 GB | Apache 2.0 | sub-5Bの推奨されるデフォルト。このサイズで最高の品質が得られます。 |
| Phi-4-mini-3.8B | 6 / 7 / 12 GB | MIT | 推論/数学/コードに強い。厳密なライセンスで提供。 |
| SmolLM3-3B | 4 / 5 / 10 GB | Apache 2.0 | 完全にオープンなレシピ。ネイティブの64Kコンテキスト。 |
| Qwen 2.5 7B | 9 / 11 / 17 GB | Apache 2.0 | 既存のデフォルト。従来の7Bプリセットの中で最高の品質。 |
| Qwen 2.5 3B | 4 / 6 / 10 GB | Qwen-Research | ⚠ 研究ライセンス — 商業利用の前に、Qwenライセンス条項を確認してください。 |
| Llama 3.2 3B | 4 / 6 / 9 GB | Llama コミュニティ | Qwen 3B の優れた代替モデルであり、制限事項も明確に記載されています。 |
| Llama 3.2 1B | 2 / 3 / 5 GB | Llama コミュニティ | 小規模な GPU での簡単な実験用。 |
| Mistral 7B | 6 / 8 / 14 GB | Apache 2.0 | Qwen 7B と比較可能ですが、チャットテンプレートが異なります。 |
| Llama-3.1-8B | 9 / 11 / 18 GB | Llama-3.1-Community | 8B QLoRA、128K のネイティブコンテキスト（700M MAU を超える場合は、別途 Meta のライセンスが必要です）。 |
| **Qwen2.5-14B** | 18.7 GiB、4096 ctx でのピーク値（QLoRA） | Apache 2.0 | **毎日使用する 32 GB モデル。** rank/alpha 32、8 ビット AdamW。4 ビットの重みだけで約 8.5 GB です。4096 トークンの完全なウィンドウには、残りの容量が必要です。 |
| Mistral-Small-24B | 22.8 GiB、4096 ctx でのピーク値（QLoRA） | Apache 2.0 | 24B QLoRA、32 GB の GPU で使用。4 ビットの重みだけで約 18 GB です。 |
| **Qwen2.5-32B** | 26.0 GiB、2048 ctx でのピーク値（QLoRA） | Apache 2.0 | **32 GB の GPU で使用できる最大のモデル。** 8 ビット AdamW で `max_len 2048` に収まります。 |

他のモデルでも動作することがよくあります。上記の行は、厳選されたプリセットです。14B ～ 32B の範囲は、32GB のカード用に QLoRA で調整されています（測定された範囲）。8B までのプリセットの場合、3 つの数値は、バッチサイズ 1 で 2,048 トークンの例を使用した場合の、`fast`、`balanced`、および `quality` アダプターのサイズの QLoRA 推定値です。これらはやや大きめに設定されており、短い例ではより少ないサイズが使用されます。アダプターのサイズは、カードに合わせて選択されます。`--lora-preset auto`（デフォルト）は、GPU で利用可能なメモリに収まる、`quality`（Biderman 2024 および Thinking Machines 2025 に基づく、すべての線形レイヤーでランク 256）、`balanced`（すべての線形レイヤーでランク 64）、および `fast`（ブロックごとに 2 つのレイヤーでランク 16）のうち、最も大きいものを選択します。特定のものを指定して強制することもできます。`backprop estimate-vram` は、任意のモデルと設定の推定値を表示します。

## トラブルシューティング

最も一般的な初回実行時のエラーの簡単な一覧です。完全な逆引きインデックスは、[トラブルシューティングハンドブックのページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)にあります。ドライバー/VRAM/混合精度に関する詳細な情報については、[CUDA トラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)を参照してください。

| 症状 | エラーコード | 修正方法 |
|---|---|---|
| GPU のメモリがトレーニング中に不足 | `RUNTIME_GPU_OOM` | 自動 — Backpropagate は、バッチサイズを半分にして、最大 3 回まで再試行します。これを無効にするには、`Trainer(oom_recovery=False)` を使用します。より小さいサイズに強制するには、`--batch-size 1` を使用します。 |
| HuggingFace が 401 / "モデルが見つかりません" を返します | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` を実行して再試行します。タイプミスの場合、<https://huggingface.co/models> から正確な ID をコピーします。 |
| `register_with_ollama` 接続が拒否されました | `DEP_OLLAMA_REGISTRATION_FAILED` | デーモンを起動します: `ollama serve`。 <https://ollama.com> からインストールします。再試行可能です。 |
| チェックポイントの保存中にディスクがいっぱい | `STATE_CHECKPOINT_INVALID` | アトミック書き込みにより、クラッシュ時に `.partial` ディレクトリが残ります。削除しても安全です。以前の正常なチェックポイントはそのままです。 |
| GPU の過熱によりトレーニングが一時停止 | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | 自動 — Backpropagate は、温度の閾値に達すると一時停止し、GPU が冷却されると再開します。頻繁に発生する場合は、エアフローを改善してください。 |
| `backprop ui --share` が拒否されました | `RUNTIME_UI_AUTH_NOT_ENFORCED` | `--auth user:password` を渡すか、代わりに SSH ポートフォワーディングを使用します（[Web UI](#web-ui) を参照）。 |
| GGUF エクスポートが初回で失敗 | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]` を実行します。Windows の場合は、Visual C++ Build Tools + CMake も必要です。 |

## バグの報告

何らかの理由でエラーが発生した場合、Backpropagate は起動時に `run_started run_id=<uuid>` のような行を出力し、同じ ID をすべてのログ行、すべてのチェックポイント、およびすべての Weights & Biases エントリにバインドします。**バグ報告には、必ず `run_id` を含めてください**。これにより、開発者は、その特定の実行に関連するすべての情報を関連付けることができます。

優れたバグ報告には、次のものが含まれます。

1. **`run_id`** — 起動時に出力される UUID。1 つの UUID を使用すると、開発者は、その特定の実行に関連するすべてのログ行、すべてのチェックポイント、およびすべての Weights & Biases エントリを関連付けることができます。
2. **エラーコード** — stderr の `[CODE_NAME]: message` 行。安定したコードのカタログについては、[エラーコード](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) を参照してください。
3. **編集されたトレースバック。** 非詳細モードでは、stderr が自動的に編集されます（Bearer トークン、`sk-*`、`hf_*`、AWS キー、`password=` / `token=` / `api_key=` ペアが削除されます）。安全に貼り付けることができます。完全な編集されていないトレースバックを取得するには、`BACKPROPAGATE_DEBUG=1`（または `--verbose`）で再実行し、投稿する前に確認してください。
4. **`backprop info` の出力。** 1 つのコマンドで、Python / PyTorch / CUDA / GPU モデル / VRAM / OS / インストールされた追加機能が出力されます。これは、開発者がプラットフォーム固有の回帰を特定するために必要なすべての情報です。

[バグ報告テンプレート](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) では、これらすべてを明示的に確認するように促すため、トリアージが迅速に進みます。質問、アイデア、または「これは想定どおりですか？」というスレッドは、[GitHub ディスカッション](https://github.com/mcp-tool-shop-org/backpropagate/discussions) に投稿してください。セキュリティの問題は、[GitHub セキュリティアドバイザリ](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) フォームを通じて非公開で報告してください。ポリシーと対応のタイムラインについては、[SECURITY.md](SECURITY.md) を参照してください。

## プライバシー

すべてのトレーニングは、GPU 上でローカルに実行されます。Backpropagate は、HuggingFace からモデルをダウンロードする場合を除き、ネットワークリクエストを行いません（これはユーザーが開始します）。テレメトリはなく、クラウドへの依存もありません。

## 参考文献

Backpropagate のデフォルト設定と複数回のトレーニングモードは、最近の研究に基づいています。関連する技術に興味がある場合は、以下を参照してください。

- **Hu ら、2021 年。** *LoRA: 大規模言語モデルの低ランク適応。* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — LoRA を導入した基礎論文。これは、Backpropagate がアダプターを効率的にトレーニングする方法です。
- **Biderman ら、2024 年。** *LoRA はより少なく学習し、より少なく忘れる。* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — 実証的な証拠。ランク 256 ですべての線形ターゲットを使用した LoRA が、ほとんどのポストトレーニングタスクで、計算量の 67% で完全なファインチューニングの品質に匹敵することを示しています。Backpropagate の v1.3 デフォルト LoRA 構成を決定します。
- **Thinking Machines、2025 年。** *LoRA 後悔することなく。* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — 実用的なフォローアップ。高い LoRA ランクで必要な 10 倍の学習率と完全な FT の補正を特定しています。
- **Kirkpatrick ら、2017 年。** *ニューラルネットワークにおける破滅的な忘却の克服。* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — ニューラルネットワークが、新しいデータでファインチューニングを行うと、以前のトレーニングを「忘れてしまう」理由を最初に説明した論文（EWC — 弾性重み集約）。
- **Wang ら、2023 年。** *言語モデルの継続学習のための直交部分空間学習。* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA。これは、新しいアダプターを直交部分空間に制約することで、継続学習に LoRA を使用する、より早いアプローチです。
- **Yadav ら、2023 年。** *モデルをマージする際の干渉を解決する TIES-マージ。* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — 複数のファインチューニングされたモデルを干渉なしでマージするための基礎的な手法。
- **Qiao & Mahdavi、2025 年。** *忘れる前にマージする：継続的なマージによる単一の LoRA 継続学習。* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — Backpropagate の複数回の実行マージが実装する特定のアルゴリズム。2025 年 12 月のプレプリント。Backpropagate は、この論文の最初の既知のダウンストリーム採用者です。

## ライセンス

MIT — [LICENSE](LICENSE)を参照。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
