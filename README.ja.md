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

# 32B QLoRA、または7Bの完全なモデルを、1つのGPUで微調整します。Ollamaに送信します

**1つの**GPUで、実際に使用しているカードに合わせてサイズ調整された大規模言語モデルの微調整をバックプロパゲーションします。3行のPythonコードで、32GBのコンシューマーカード（RTX 5090）で7B～32Bのモデルを微調整します。1つのフラグ、`--full-ft-offload`で、7Bクラスのモデルの重みと勾配をホストRAM（LinuxまたはWSL2）に保持することで、完全な微調整を行います（速度は遅く、後述）。さらに1つのコマンドでOllamaにエクスポートし、次に`ollama run`で微調整したモデルを使用します。16GBまでスケールダウンします。Windowsでも優れたパフォーマンスを発揮します。Pythonの代わりにブラウザを使用したいですか？`backprop ui`を使用すると、コードなしでこれらすべてを実行できます（[ツアーはこちら](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/)）。

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

以上です。YAML設定ファイルはありません。`accelerate launch`のような複雑な設定もありません。「GGUFに変換する」という別のチュートリアルもありません。CUDA GPUと、トレーニングデータを含むJSONLファイルがあれば、すぐに動作する微調整モデルが3行のコードで作成できます。

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

Prefer Docker? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` works too. Images ship for both `linux/amd64` and `linux/arm64`, so Apple Silicon and ARM Linux operators get a native image. A canonical `compose.yaml` for "UI in a container" lives at the repo root: put `user:password` in a `ui-auth.txt` next to it, run `docker compose up`, and sign in at `http://127.0.0.1:7860` (the first start builds the frontend, which takes a minute or two). Run history persists in `~/.backpropagate`.

## Backpropagateが提供するもの

LLMの微調整を行うための優れたライブラリがいくつかあります。それぞれ異なる点で優れています。

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — YAML設定を好み、コピーできるレシピのコミュニティが欲しい場合に
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — DPO/PPO/RLHFとWeb GUIが欲しい場合に
- **[Unsloth](https://github.com/unslothai/unsloth)** — 可能な限り最速のトレーニングが必要で、サポートされているモデルファミリーを使用している場合に
- **[torchtune](https://github.com/pytorch/torchtune)** — Metaの公式のPyTorchネイティブのレシピを編集したい場合に

Backpropagateは、不足しているオプションです。**1つのコンシューマーGPUでアダプターをトレーニングし、その結果を送信するための、3行のPython APIです。** YAMLはなく、オンラインRL（PPO/GRPO）も、マルチノードもありません。コードを書きたくない場合は、同じループを実行するためのブラウザUIもあります。誰もが必要とし、邪魔になるエクスポートステップだけです。

上記のライブラリのいずれかを試して、設定ファイルの複雑さにうんざりしたり、モデルファミリーの制限に遭遇したり、Windowsを優先するデフォルト設定が必要になった場合は、Backpropagateが最適です。

## 1つのGPUで微調整できるもの

Backpropagateは、実行をカードのサイズに合わせて調整します。以下は、32GBのRTX 5090で測定された数値です。QLoRAの行は2026年10月3日（証拠：[`docs/receipts/2026-10-03-presets/`](docs/receipts/2026-10-03-presets/)）、完全な微調整の行は2026年9月30日（証拠：[`docs/receipts/2026-09-30-offload/`](docs/receipts/2026-09-30-offload/))です。QLoRAのピークは、設定の最大のコンテキストウィンドウでバッチ1の場合であり、これはその設定の最悪のケースです。短い例では、より少ないリソースを使用します。

| モデル | 方法 | 32GBのカードで測定 |
|---|---|---|
| **14B** (Qwen2.5-14B) | QLoRA | **18.7 GiB**（4096コンテキストの場合のピーク、20.0 GiB予約）。 |
| 24B (Mistral-Small-24B) | QLoRA | 22.8 GiB（4096コンテキストの場合のピーク、24.2 GiB予約）。 |
| **32B** (Qwen2.5-32B) | QLoRA | **適合:** 26.0 GiB（2048コンテキストの場合のピーク、27.2 GiB予約、約4 GiBの余裕あり）。 |
| 3B | `mode="full"`（GPU上での完全な微調整） | **22.0 GiB**（システム全体でのピーク）、バッチ4、512コンテキストで0.30秒/ステップ。そのうち7.5 GiBはページングされたオプティマイザーの状態であり、より小さいカードではホストRAMにスピルする可能性があります（未テスト）。 |
| **7Bクラス** (Qwen2.5-7B、7.6Bパラメータ) | `mode="full" --full-ft-offload` | **トレーニング:** 5.3 GiB VRAM、**30.8 GiBホストRAM**（32.2 GiB、保存時）、**14.7秒/ステップ**。LinuxまたはWSL2のみ。 |

このセッションでは再測定していません：7B QLoRA、Llama-3.1-8B（ゲートされたリポジトリ、テストマシンにトークンなし）、および3Bを超える純粋なGPUによる完全な微調整。それらの数値は、ドキュメントの他の場所に記載されている推定値です。

ほとんどのシングルGPUライブラリが、**24〜32B QLoRA**と**単一カードの7Bクラスの完全な微調整**のために、他の場所に誘導するのに対し、Backpropagateは、1つのコンシューマーカードでこれらを行い、その結果をOllamaに直接エクスポートします。

**完全な微調整には2つの方法があります。** オフロードを使用しない場合、モデル、その勾配、およびオプティマイザーの状態はすべてGPUに配置されます。ライブラリは、検出されたVRAMによってモデルサイズを制限します（**16 GB → 4B、24 GB → 5B、32 GB → 6B**）。これらの制限は、メモリ計算から得られ、3Bまでしか測定されていません。`--full-ft-ceiling-billions`でオーバーライドします。

`--full-ft-offload`は、重みと勾配をホストRAMに保持し、GPUにストリーミングします（FSDP2 CPUオフロード）。測定されたコスト：

- **ホストRAM:** 適合性チェックでは、10億パラメータあたり約3.7GiBに加え、さらに10GiBが必要とされます。これは控えめな設定です（7.6Bモデルの場合、測定値の32.2GiBに対して39GiB）。マシンがそれを処理できない場合、実行は事前に拒否されます。7.6Bモデルは、28GBのWSL2メモリ制限下では動作しません。実用的な上限は約5Bです。
- **速度:** 7.6Bの場合、1ステップあたり14.7秒（バッチ1）、3Bの場合、1ステップあたり5.1秒（バッチ4）。GPU上でバッチ4の3Bモデルの場合、約0.63秒/ステップです。モデルがこれなしでは収まらない場合にのみ使用してください。より高速なバージョンが計画されています。
- **オプティマイザ:** Adafactor、AdamWではありません。重みはbf16のまま、各更新は確率的丸めを使用して書き戻されます。fp32コピーはありません。
- **品質:** 3Bモデルで1回の実行（150ステップ、保留された損失、1つのシード）を行ったところ、通常の完全なファインチューニングで得られた改善の約85％に達しました（2.45 → 1.93、通常の完全なファインチューニングでは2.45 → 1.84）。1つのシードはベンチマークではありません。
- **範囲:** 単純な教師ありファインチューニング。パッキング、レスポンスのみのマスキング、中間チェックポイント、再開はありません。LinuxまたはWSL2のみ（FSDP2にはNCCLが必要です）。Windowsネイティブでは、`DEP_FSDP_UNAVAILABLE`で停止します。
- **まだテストされていません:** 長時間の実行、1を超える勾配の累積、および物理的な64GBマシン（テストマシンはより多くのRAMを持っており、テストによって60GiBの制限が適用されました）。

モデルが収まらない場合、`RUNTIME_FULL_FT_MODEL_TOO_LARGE`で終了し、その方法を示します。[完全なファインチューニングハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/)を参照してください。

### 16GBにスケールダウン

16GBの範囲（RTX 4080 / 5080 / 4070 Ti Super）は、依然として最優先です。7B QLoRA（アダプターのサイズは、16GBカードに収まるように調整されます。ランク256の場合は約17GBが必要）、および約3Bモデル（SmolLM3-3B、Qwen2.5-3B、Llama-3.2-3B/1B）の真の完全なファインチューニングを、`mode="full"`を使用して行います（32GBカードで3Bの場合、測定値は22.0GiBで、そのうち7.5GiBはホストRAMにスピルできるページングされたオプティマイザの状態です。16GBカードで許容範囲で実行できるかどうかはテストされていません）。`--full-ft-offload`を使用すると、GPUはさらに少なくなります。テストカードでVRAMが制限されている場合、6GiBの制限下でトレーニングされた3Bモデル、および8GiBの制限下でトレーニングされた4Bおよび7.6Bモデルです。これらはすべて、32GBカード上でエミュレートされた制限であり、実際の8GBハードウェアでの実行ではありません。同じコードが、検出されたカードに収まるバッチサイズと上限を選択します。

2ビット量子化（AQLM / QuIP#）は**範囲外**です。2ビットのベースモデルを、完全精度重みにクリーンにマージすることはできません。これにより、マージ可能なアダプター→GGUF→Ollamaエクスポートの契約が破綻します（パイプラインの目的のすべてです）。Backpropagateが提供する代替手段は、QLoRA、`mode="full"`、`--full-ft-offload`、およびFP8計算パス（`--fp8`、Blackwell/Hopper）であり、これらはすべてマージ可能でエクスポート可能です。

## Backpropagateが適さないもの

ユースケースが以下の場合、別のライブラリを使用する方が良いでしょう。Backpropagateは適切な選択肢ではなく、使用しようとすると、適切なツールを使用するよりも多くのコストがかかります。開始する前にこのセクションを読むことで、インストールと再試行のサイクルを回避できます。

- **13B+モデルの完全パラメータファインチューニング** — Backpropagateは、32GB GPUで約6Bまで、および`--full-ft-offload`を使用して7Bクラスのモデルまで、完全なファインチューニングを行います（[エンベロープ](#what-you-can-fine-tune-on-one-gpu)を参照）。13B+モデルの完全なファインチューニングには、マルチGPU FSDPまたはより大きなカードが必要です。その計算リソースを使用する前に、両方の証拠を検討してください。[Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)は、すべてのレイヤーに適用され、データセットがアダプターの容量に収まる場合、LoRAは1回のパスあたりの計算量の約3分の2で、完全なファインチューニングに匹敵すると報告しています。[Biderman et al. 2024](https://arxiv.org/abs/2405.09673)は、標準的な低ランク設定では、LoRAはコードと数学において完全なファインチューニングよりも大幅にパフォーマンスが劣り、忘れが少ないことを発見しました。指示に従う、ペルソナ、およびスタイルに関する作業では、適度なデータセットでQLoRAを使用する方が、通常は1つのカードをより有効に活用できます。
- **オンラインRL — PPO / GRPO / RLVR** — Backpropagateは、単一段階のSFTと参照なしの優先度チューニング（v1.5ではORPO、v1.6ではSimPO + KTO）を行います。行わないことは、オンライン強化学習（PPO、GRPO、またはRLVR）です。これには、報酬モデルまたはトレーニングステップの上に構築された生成とスコアリングループが必要です。それらの場合は、TRLまたはLLaMA-Factoryを直接使用してください。（参照なしの優先度チューニングは、単一段階の範囲に適合します。なぜなら、メモリに保持する必要のある個別の参照モデルがないからです。ORPOに関する注釈は、[クイックスタート](#quick-start)を参照してください。）
- **マルチノードトレーニング** — 1つのマシン上の単一のGPUのみ。1つのマシン上のマルチGPUは機能しますが、公式にはサポートされていません（`accelerate launch`を使用）。
- **CUDAレール上のmacOSトレーニング** — Apple SiliconにはCUDAがないため、CUDAパスはLinuxまたはWindowsボックス上のNVIDIA GPUで実行されます。トレーニングされたモデルは、Ollamaを介してMacで実行できます。**実験的で、検証されていないプレビュー**のMLXレール（`--backend mlx`）は、Apple Silicon上でLoRAアダプターをネイティブにトレーニングします。これはLoRA-SFTのみであり、**実際のシリコンでドッグフード検証されていません**（サポートはありません）。したがって、LoRA SFT（ORPO、完全なファインチューニング、FP8、複数回の実行）以外のものについては、CUDAレールを使用する必要があります。
- **テストされたモデルファミリー外のもの** — Qwen 2.5 / 3.5（7B / 4B）、Phi-4-mini-3.8B、SmolLM3-3B、Llama 3.2（3B / 1B）、Mistral 7B。他のモデルも機能することがありますが、CIで固定されていません。

これらのいずれかが必要な場合は、上記のライブラリを使用してください。それらのライブラリの方が適しています。

## Backpropagateが提供するもの

1つのインストールで、次の4つの機能を提供します。

**1. 3行の実際のAPIで、設定ファイルなしで実行できます。**
このREADMEの先頭にあるスニペットは、最初から最後まで実行されます。`accelerate config`、YAML、Hydraオーバーライドはありません。`Trainer(model).train(data)`を使用するだけで、ファインチューニングができます。

**2. 実際に動作する Windows 版。**
ほとんどの機械学習ライブラリは、Windows を後回しにして扱っています。Backpropagate は、RTX 50 シリーズのカードを搭載した Windows 11 で開発およびテストされています。このライブラリは、実行時の問題を自動的に処理します。Windows のマルチプロセッシングがクラッシュしないように、データを事前にトークン化する方法を把握し、RTX 40/50 カードで問題が発生する可能性がある xformers を自動的に無効にし、問題が発生しないデータローダー設定を選択します。これらのことを知っておく必要はありません。単に実行するだけです。

**3. 無人での実行用に設計。**
トレーニングには数時間かかります。常に監視する必要はありません。Backpropagate は、実行したままにしておくように設計されています。

- GPU メモリが不足した場合、バッチサイズを自動的に半分にし、最大 3 回まで再試行します。手動での調整は不要です。
- GPU が過熱した場合、温度が下がるまで一時停止し、その後再開します。
- すべてのチェックポイントはアトミックに書き込まれます。ラップトップが保存中にクラッシュした場合でも、以前の正常なチェックポイントはそのまま残ります。
- すべてのトレーニング実行には、一意の ID が割り当てられ、すべてのログ行、すべてのチェックポイント、およびすべての Weights & Biases エントリに記録されます。問題が発生した場合、1 つの ID で、すべての情報を関連付けることができます。
- エラーには、安定したコード (`RUNTIME_GPU_OOM`、`DEP_OLLAMA_REGISTRATION_FAILED` など) が付随するため、ログを検索して、[トラブルシューティングガイド](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) で修正方法を確認できます。CUDA に固有の障害については、[CUDA トラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) を参照してください。

**4. トレーニング済みのアダプターから `ollama run` へのワンコマンド。**
多くのライブラリがモデルをトレーニングします。しかし、実際に使用したいときに、その邪魔にならないものはほとんどありません。Backpropagate は、GGUF (Ollama が使用する形式) にエクスポートし、1 つのコマンドで Ollama モデルを登録します。トレーニングが完了してから、「自分のファインチューンモデルとチャットできる」状態になるまで、約 30 秒です。

## クイックスタート

コマンドラインから、5 つの会話例を含むデータセットを使用します。

```bash
pipx install "backpropagate[standard]"
curl -LO https://raw.githubusercontent.com/mcp-tool-shop-org/backpropagate/main/examples/quickstart.jsonl

backprop train --data quickstart.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 10
backprop generate ./output "What is Python?"      # did it learn anything?
backprop export ./output --format gguf --quantization q4_k_m --ollama --ollama-name my-first-finetune
ollama run my-first-finetune
```

`backprop train` は、アダプターを `./output` に書き込みます (`--output` で変更します)。Python では、同じことを次のように行います。

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")
```

Python API には、`pip install "backpropagate[standard]"` を使用した仮想環境を使用してください。`pipx` は、独自の環境に `backprop` コマンドをインストールするため、`import backpropagate` はそれを検出できません。

**GGUF エクスポートに必要なもの。** エクスポートでは、アダプターをベースモデルにマージし、llama.cpp のコンバータースクリプトを使用して変換します。次のいずれかが必要です。

- llama.cpp の **ソースコードのチェックアウト** (`git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp`) と、同じ環境に `pip install sentencepiece protobuf` がインストールされているか、
- 独自の llama.cpp がすでにビルドされている Unsloth。

`--ollama` を使用すると、`q4_k_m` 量子化は `ollama create` によって実行されるため、コンパイルする必要はありません。Backpropagate は、llama.cpp をビルドするために、Unsloth がシステムパッケージをインストールすることを許可しません。許可する場合は、`BACKPROPAGATE_UNSLOTH_AUTO_INSTALL=1` を設定してください。詳細: [エクスポート](https://mcp-tool-shop-org.github.io/backpropagate/handbook/export/)。

独自のデータの場合、JSONL 形式で、1 行に 1 つの例を記述します。

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca (`instruction` / `output`)、OpenAI チャット (`messages`)、および生のテキスト形式も使用できます。Backpropagate は、形式を自動的に検出します。

### ループ: データの確認、トレーニング、評価、エクスポート

```bash
backprop data report my_data.jsonl                     # duplicates, length outliers, format problems
backprop data split my_data.jsonl --heldout-ratio 0.1  # a held-out set the model never trains on
backprop train --data my_data.train.jsonl --steps 200 --output ./run-a
backprop eval <run-id> --heldout my_data.heldout.jsonl # held-out loss + sample generations
backprop eval <run-b> --vs <run-a>                     # did the change help?
backprop export ./run-a --format gguf --ollama --ollama-name my-model
```

評価は、設計上、判断を必要としません。保留された損失と、決定的なタスクメトリック (`normalized_exact_match`、`token_f1`、`contains`、`regex`、`pass_rate`) を使用します。LLM ジャッジを使用する場合は、それを `backprop generate` の出力に対して自分で実行してください。[レシピ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/) を参照してください。

### 優先度チューニング (ORPO、SimPO、KTO)

単純なデモンストレーションではなく、優先度に基づいてトレーニングします。ORPO は参照を必要とせず、1 段階で実行されます。優先度のシグナルを SFT ステップに組み込むため、個別の報酬モデルや参照モデルは必要なく、3 行の形状は変更されません。`--method orpo` (CLI) または `method="orpo"` (Python) を渡し、`{prompt, chosen, rejected}` (または `{chosen, rejected}` のみ) 行のデータセットを渡します。

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

**v1.6 での新機能 — SimPO と KTO。** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) は、長さで正規化された報酬を使用し、参照を必要とせず、ORPO と同じペアの `{prompt, chosen, rejected}` データを使用します (`--simpo-beta`、`--simpo-gamma`)。`--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) は、**ペアでない** `{prompt, completion, label}` データを使用します。つまり、キュレーションされた A/B ペアではない、例ごとの肯定/否定のフィードバックを使用します。望ましい/望ましくない損失の重みを、ラベルの数から自動的に調整します。どちらも `mode="lora"` のみであり、単一の GPU SFT の範囲内に収まります (個別の参照モデルはありません)。使用するものを選択するには、[優先度チューニングハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) を参照してください。オンライン RL (PPO/GRPO) については、[Backpropagate が適さない理由](#what-backpropagate-is-not-for) を参照してください。

### 推論トレース SFT (R1 蒸留)

推論モデルを簡単に蒸留します。`--reasoning-trace` (CLI) または `Trainer(..., reasoning_trace=True)` (Python) を渡し、アシスタントのターン内に `<think>...</think>` の連鎖思考を保持するトレースを渡します。これは、[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 蒸留の純粋な SFT の半分であり、RL は必要ありません。Backpropagate は、`<think>` をトレーニングターゲットに保持し、空の/長すぎるトレースを削除 (トレース長のフィルタリング) し、デフォルトの `max_seq_length` を 8192 に上げて、より長い CoT に対応します。重要な点として、`<think>` は **プレーンテキスト** のままです。特別なトークンはなく、埋め込みのサイズを変更する必要もありません。そのため、マージされた GGUF は、他のファインチューンモデルと同様に、Ollama にエクスポートできます。SFT のみです。[推論トレースレシピ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) を参照して、データセットの形状と調整可能なトークンバンドを確認してください。

### Apple Silicon (MLX) — 検証されていないプレビュー

> ⚠️ **検証されていないプレビュー版であり、サポート対象の機能セットには含まれません。** MLXフレームワークは構築され、ユニットテストも行われていますが、実際のApple Siliconデバイス（`mlx-lm`はApple専用であり、NVIDIAの環境では動作しません。BackpropagateはNVIDIAの環境で開発されています）での実機検証は**行われていません**。以下に示す内容はすべて実験的なものとして扱い、自己責任で使用し、MシリーズのMacで実行した際に異常が発生した場合は、[不具合を報告してください](#reporting-bugs)。

**1つのAPI、2つのフレームワーク。** CUDAは、標準的で検証済みのバックエンドです。MLXは、Appleの[`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm)ツールチェーン（統合メモリ、CUDA不要）を介して、MシリーズのMacでトレーニングを行う2番目のフレームワークです。3行のコードで、ハードウェアに応じてフレームワークを選択します。具体的には、`backend='auto'`（デフォルト）は、NVIDIAではCUDAに、Apple SiliconではMLXにルーティングされるため、既存のCUDA環境は、同じ結果を出力します。

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

MLXレールは**LoRA SFTのみ**に対応しており、ORPO、FP8、`mode='full'`、複数回の実行はサポートしていません（それぞれが`CONFIG_INVALID_SETTING`で拒否されます。それらの機能を使用する場合は、NVIDIA環境で`backend='cuda'`/`'auto'`を使用してください）。生成されるアダプターは、単純なsafetensors形式であり、CUDAレールと同じパスを通じてOllamaにエクスポートされます。

> Apple 製ではないホストに `--backend mlx` を強制的に適用すると、エラー `CONFIG_INVALID_SETTING` が発生します。また、Mac に必要な `mlx_lm` ツールチェーンがインストールされていない場合、エラー `DEP_MLX_UNAVAILABLE` が発生します。

より包括的なワークフロー（ファインチューニングとHF Hubへのプッシュ、OOM（メモリ不足）発生後の再開、長期間にわたるキャンペーンにおける複数回のSLAO実行など）については、[ハンドブックのレシピページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/)をご覧ください。

### Webユーザーインターフェース（オプション）

Pythonのコードをタイプするよりも、クリック操作の方がお好みであれば、UI拡張機能をインストールして、以下のコマンドを実行してください。

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

表示されたURLを開きます（`http://127.0.0.1:7862/?token=...`。起動するたびに新しいトークンが生成されます。最初の起動時にフロントエンドがビルドされ、1～2分かかることがあります）。これは、トレーニング用のローカルWebインターフェースです。実行を開始したり、複数の実行をまとめて行ったり、エクスポートしたり、リアルタイムで状況を確認したり（ステップ、損失、残り時間、GPU温度、メモリの使用量）、保存されたチェックポイントで停止したりできます。各ジョブは個別のプロセスで、一度に1つずつ実行され、ページをリロードすると、実行中のジョブを再開できます。データセットのページには、ファイルの内容が表示され、クリーニングされたコピー（重複や空のサンプルが削除されたもの）が保存され、トレーニングフォームに渡されます。過去の実行や、Hugging Faceキャッシュに保存されているモデルは、それぞれ別のページに表示され、すべての設定には、その内容を説明する「i」アイコンが付いています。\[Web UIツアー](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/)では、各ページが紹介されています。デフォルトでは、UIはローカルでのみ利用可能です。他のデバイスからアクセスできるようにするには、以下に示す\[Web UI](#web-ui)の`--share` + `--auth`セキュリティに関する記述を参照してください。

## 複数回のトレーニング

複数のデータセットにわたって段階的に微調整を行いたい場合（たとえば、毎週新しい学習データを入手し、以前に学習した内容を忘れることなく、それを追加したい場合）は、Backpropagateの`multi_run`モードが最適です。

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

この手法では、5回の学習サイクルを実行し、各サイクルの間にアダプターを統合することで、以前の知識を維持しつつ、新しい事例を取り入れます。この技術は、近年の継続学習に関する研究に基づいています。詳細については、このREADMEの末尾にある「参考文献」を参照してください。

CLI版：

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## チェックポイントから再開

5回の反復で構成されるトレーニングで、4回目の反復でエラーが発生した場合でも、再開は可能です。各複数回の反復セッションでは、その反復のIDがディスク上の履歴およびチェックポイントマニフェストに記録されるため、中断したところから再開するには、1つのコマンドを実行するだけです。

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

`backprop multi-run`（`--resume`なし）のデフォルトの動作では、同じ出力ディレクトリにある進行中のエントリを自動的に検出し、それを続行します。完全に新しい状態から開始するには、新しい出力ディレクトリを指定してください。

## トレーニング履歴

すべての`backprop train`および`backprop multi-run`の実行は、`<output>/run_history.json`に1行の記録として保存されます。記録には、使用したモデル、データセット、ハイパーパラメータ、ステータス、最終的な損失、および損失の履歴が含まれます。過去の実行を一覧表示して確認することができます。

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## 実験の進捗状況の追跡

Backpropagateは、インストールされている実験追跡ツール（Weights & Biases、TensorBoard、MLflow）を自動的に検出し、それらを連携させます。もし`wandb`がインストールされており、ログインしている場合、すべての実行結果は自動的にW&Bに記録され、記録される実行結果の名前は、ディスク上の実行IDと一致します。これにより、W&B、ログ、および`run_history.json`全体で、単一の識別子を使用して検索を行うことができます。

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

機能を無効にするには、`Trainer(report_to=["wandb"])`、`Trainer(report_to=["tensorboard"])`、または`Trainer(report_to="none")`を指定して上書きしてください。

## ウェブユーザーインターフェース

The Reflex web interface is opt-in — install with `pipx install "backpropagate[ui]"` and launch:

```bash
backprop ui --port 7862
```

UIはローカルで実行されます。表示されるURL（`http://127.0.0.1:7862/?token=...`）を開いてください。`--auth`がない場合、起動するたびに新しいトークンが生成され、そのトークンがないリクエストはUIによって拒否されます。UIからは、データセットの確認やクリーニング、モデルのトレーニング（単一の実行または複数回の実行）、実行状況のリアルタイム監視、保存されたチェックポイントを使用した実行の中止、結果のエクスポートなどを行うことができます。各ジョブは個別のプロセスで、一度に1つずつ実行され、UIを閉じるとジョブが停止します。\[Web UIツアー](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/)では、スクリーンショット付きで各ページを順番に説明します。

他のデバイス（ネットワーク上の他のユーザー、公開URLなど）からアクセスできるようにするには、`--share`（または`--host`）と`--auth`をペアリングする必要があります。

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` に `--auth` が含まれていない場合、エラーが発生して終了します。その理由は、`--share` がインターネット上の誰でもアクセスできる URL を公開するため、認証を行わないと、誰でもトレーニングのパイプラインを操作し、HuggingFace のトークンを読み取ることができるようになるからです。この設定を無効にするオプションはありません。認証情報を設定したくない場合は、代わりに SSH ポートフォワーディングを使用してください。

```bash
# On the client:
ssh -L 7862:localhost:7862 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open the URL the server printed (http://127.0.0.1:7862/?token=...) locally
```

完全な脅威モデルについては、[handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) を参照してください。

UIからのファイルシステムへの書き込みは、単一のディレクトリに限定されます。

- デフォルト：`~/.backpropagate/ui-outputs`
- 上書き：`BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own` を設定
- 上書き設定は、許可リストによる検証が行われます。システムまたは認証情報のパス（`/etc`、`~/.ssh`、`~/.aws`、`C:\Windows\System32`など）は拒否されます。

## プラットフォームに関する注記

**必要なもの:** Python 3.10 以降、CUDA 対応の NVIDIA GPU、PyTorch 2.0 以降。8GB の GPU であれば、10 億から 30 億パラメータのモデルを学習できます。16GB の GPU であれば、70 億パラメータのモデルを、32GB の GPU であれば、QLoRA を使用して最大 320 億パラメータのモデルを学習できます。

Python 3.10 は、少なくとも v1.6 までサポートされます。2026 年 10 月にサポートが終了し、その後最初のリリースで削除される予定です。新しいインストールの場合、Python 3.11 または 3.12 を推奨します。3.11 は最もテストされたバージョンです。

Backpropagate は、さまざまなプラットフォームでのトレーニングにおける実行時の問題を処理しますが、インストール時の問題を修正することはできません。最も一般的な問題は次の 2 つです。

- **誤った CUDA ホイール。** PyTorch は、CUDA バージョンごとに 1 つのバイナリとして公開されます。誤ったものを選択すると、CPU のみを使用する PyTorch がサイレントにインストールされ、トレーニングは非常に遅くなります。ドライバーに合わせて、<https://pytorch.org/get-started/locally/> のホイール ピッカーを使用してください。`nvidia-smi` を実行して、ドライバー/CUDA バージョンを確認します。
- **Windows + GGUF エクスポート。** `[export]` は、ソースから `llama-cpp-python` を追加でビルドします。これには、Visual Studio Build Tools (C++ コンポーネント) と CMake が必要です。

**macOS:** CUDA はサポートされていません (CUDA がない)。CUDA を使用する `trainer.train()` を実行すると、`DEP_GPU_NOT_AVAILABLE` が発生し、トレーニング済みのアダプターを Ollama を介して Mac で実行できます。**実験的で検証されていない** MLX レール (`--backend mlx`、`pip install 'backpropagate[mlx]'`) は、Apple Silicon 上で `mlx_lm.lora` を介して LoRA アダプターをネイティブにトレーニングします。LoRA SFT のみで、**実際のハードウェアで検証されていません** (「[Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview)」を参照)。CUDA パス、または ORPO / 完全なファインチューニング / FP8 / 複数回の実行の場合は、CUDA Linux または Windows マシンを使用してください。

詳細なインストール手順とトラブルシューティングについては、[トラブルシューティングハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) を参照してください。ドライバー/VRAM/xformers/bf16 と fp16 の問題については、[CUDA トラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) を参照してください。

## CLI

すべての Python API には、CLI の対応するものがあります。

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

完全なリファレンスは、[CLI ハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/)、または `backprop <subcommand> --help` にあります。

## 設定

すべての設定は、`BACKPROPAGATE_` プレフィックスを使用して環境変数でオーバーライドできます。

| 変数 | デフォルト | 備考 |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | JSON またはコンソールログを強制します |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | デフォルトモデル |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | 学習率 |
| `BACKPROPAGATE_LORA__R` | `256` | LoRA ランク。設定すると、自動アダプターサイズ選択が無効になります ([モデルプリセット](#model-presets) の `--lora-preset` を参照)。 |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | UI ファイルシステムサンドボックス |

ネストされたキーには、二重アンダースコア (`MODEL__NAME`、`MODEL_NAME` ではない) を使用します。完全なリファレンスは、[環境変数ハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/) にあります。

## モデルプリセット

| プリセット | GPU メモリ | ライセンス | 備考 |
|---|---|---|---|
| Qwen-3.5-4B | 6 / 7 / 11 GB | Apache 2.0 | 5B 未満の場合の推奨されるデフォルト。このサイズで最高の品質。 |
| Phi-4-mini-3.8B | 6 / 7 / 12 GB | MIT | 推論/数学/コードに強い。厳格なライセンスでクリーン。 |
| SmolLM3-3B | 4 / 5 / 10 GB | Apache 2.0 | 完全にオープンなレシピ。ネイティブの 64K コンテキスト。 |
| Qwen 2.5 7B | 9 / 11 / 17 GB | Apache 2.0 | 既存のデフォルト。従来の 7B プリセットの中で最高の品質。 |
| Qwen 2.5 3B | 4 / 6 / 10 GB | Qwen-Research | ⚠ 研究ライセンス — 商業利用の前に、Qwen ライセンス条項を確認してください。 |
| Llama 3.2 3B | 4 / 6 / 9 GB | Llama Community | Qwen 3B の優れた代替手段で、許可的な条件があります。 |
| Llama 3.2 1B | 2 / 3 / 5 GB | Llama Community | 小さなカードで迅速な実験を行うためのもの。 |
| Mistral 7B | 6 / 8 / 14 GB | Apache 2.0 | Qwen 7B と比較可能で、異なるチャットテンプレートを使用します。 |
| Llama-3.1-8B | 9 / 11 / 18 GB | Llama-3.1-Community | 8B QLoRA、128K ネイティブコンテキスト (700M-MAU を超える場合は、個別の Meta ライセンスが必要です)。 |
| **Qwen2.5-14B** | 4096 ctx でのピークは 18.7 GiB (QLoRA) | Apache 2.0 | **32 GB の日常的な使用に適したモデル。** ランク/アルファ 32、8 ビット AdamW。4 ビットの重みだけで約 8.5 GB です。完全な 4096 トークンのウィンドウには、残りの容量が必要です。 |
| Mistral-Small-24B | 4096 ctx でのピークは 22.8 GiB (QLoRA) | Apache 2.0 | 32 GB のカードで 24B QLoRA を実行します。4 ビットの重みだけで約 18 GB です。 |
| **Qwen2.5-32B** | 2048 ctx でのピークは 26.0 GiB (QLoRA) | Apache 2.0 | **32 GB の上限。** 8 ビット AdamW で `max_len 2048` に適合します。 |

Other models often work; the rows above are the curated presets — the 14B–32B tier is QLoRA-tuned for a 32 GB card (the measured envelope). For the presets up to 8B, the three figures are QLoRA estimates for the `fast`, `balanced` and `quality` adapter sizes at batch 1 with 2,048-token examples; they err on the high side, and shorter examples use less. The adapter size is chosen for your card: `--lora-preset auto` (the default) takes the largest of `quality` (rank 256 on every linear layer, per Biderman 2024 and Thinking Machines 2025), `balanced` (rank 64 on every linear layer) and `fast` (rank 16 on two layers per block) that fits the memory free on your GPU. Name one to force it. `backprop estimate-vram` prints the estimate for any model and settings.

## トラブルシューティング

初回実行時に発生する可能性のある最も一般的な問題の簡単なインデックス。完全な逆インデックスは、[トラブルシューティングハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) にあります。ドライバー/VRAM/混合精度に関する詳細なトラブルシューティングについては、[CUDA トラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) を参照してください。

| 症状 | エラーコード | 修正 |
|---|---|---|
| GPUのメモリがトレーニング中に不足しました。 | `RUNTIME_GPU_OOM` | 自動 — バックプロパゲーションにより、バッチサイズが半分になり、最大3回再試行します。無効にするには：`Trainer(oom_recovery=False)`。より小さい値に強制するには：`--batch-size 1`。 |
| HuggingFaceから401 / "モデルが見つかりません"というエラーが返されました。 | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login`で再試行してください。タイプミスの場合、<https://huggingface.co/models>から正確なIDをコピーしてください。 |
| `register_with_ollama`接続が拒否されました。 | `DEP_OLLAMA_REGISTRATION_FAILED` | デーモンを開始します：`ollama serve`。 <https://ollama.com>からインストールしてください。再試行可能です。 |
| チェックポイント保存中にディスクがいっぱいになりました。 | `STATE_CHECKPOINT_INVALID` | アトミック書き込みにより、クラッシュ時に`.partial`ディレクトリが残ります。削除しても安全です。以前の正常なチェックポイントはそのままです。 |
| GPUの過熱によりトレーニングが一時停止しました。 | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | 自動 — バックプロパゲーションは、温度の閾値で一時停止し、GPUが冷却されると再開します。頻繁に発生する場合は、エアフローを改善してください。 |
| `backprop ui --share`拒否されました。 | `RUNTIME_UI_AUTH_NOT_ENFORCED` | `--auth user:password`を渡すか、代わりにSSHポートフォワーディングを使用してください（[Web UI](#web-ui)を参照）。 |
| GGUFエクスポートが最初の試行で失敗しました。 | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`。Windowsでは、Visual C++ビルドツールとCMakeも必要です。 |

## バグの報告

何らかの理由で失敗した場合、Backpropagateは起動時に次のような行を出力し、同じIDをすべてのログ行、すべてのチェックポイント、およびすべてのWeights & Biasesのエントリにバインドします：`run_started run_id=<uuid>`。**バグ報告には`run_id`を含めてください**。これにより、担当者はその特定の実行に関連するすべての情報を関連付けることができます。

優れたバグ報告には、次のものが含まれます。

1. **`run_id`** — 起動時に出力されるUUID。1つのUUIDにより、担当者はその特定の実行に関連するすべてのログ行、すべてのチェックポイント、およびすべてのWeights & Biasesのエントリを関連付けることができます。
2. **エラーコード** — stderrの`[CODE_NAME]: message`行。安定したコードのカタログについては、[エラーコード](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/)を参照してください。
3. **編集されたトレースバック。** 詳細モードでない場合、stderrは自動的に編集されます（Bearerトークン、`sk-*`、`hf_*`、AWSキー、`password=` / `token=` / `api_key=`ペアが削除されます）。貼り付けても安全です。完全な編集されていないトレースバックについては、`BACKPROPAGATE_DEBUG=1`（または`--verbose`）で再実行し、投稿する前に確認してください。
4. **`backprop info`出力。** 1つのコマンドで、Python / PyTorch / CUDA / GPUモデル / VRAM / OS / インストールされた追加機能が出力されます。これは、担当者がプラットフォーム固有の回帰を特定するために必要なすべての情報です。

[バグ報告テンプレート](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml)には、これらすべてが明示的に記載されているため、トリアージが迅速に進みます。質問、アイデア、または「これは想定通りですか？」というスレッドは、[GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions)に投稿してください。セキュリティの問題は、[GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new)フォームを通じて非公開で報告してください。ポリシーと対応のタイムラインについては、[SECURITY.md](SECURITY.md)を参照してください。

## プライバシー

すべてのトレーニングは、ローカルのGPU上で行われます。Backpropagateは、HuggingFaceからモデルをダウンロードする場合を除き、ネットワークリクエストを行いません（これはユーザーが開始します）。テレメトリはなく、クラウドへの依存もありません。

## 参考文献

Backpropagateのデフォルト設定と複数回のトレーニングモードは、最近の研究に基づいています。関連する技術に興味がある場合は、次の資料を参照してください。

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — LoRAを紹介する基礎論文。Backpropagateは、この技術を使用してアダプターを効率的にトレーニングします。
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — ランク256で、すべての線形ターゲットを使用したLoRAが、ほとんどのポストトレーニングタスクで、計算量の67%で完全なファインチューニングの品質に匹敵するという実証的な証拠。Backpropagateのv1.3のデフォルトLoRA構成を決定します。
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — 高いLoRAランクで必要な10倍の学習率と完全なFTの補正を特定する、実践的なフォローアップ。
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — ニューラルネットワークが新しいデータでファインチューニングすると、以前のトレーニングを「忘れてしまう」理由を最初に説明した論文（EWC — Elastic Weight Consolidation）。
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA。これは、新しいアダプターを直交部分空間に制約することにより、継続学習のためにLoRAを使用する、より早いアプローチです。
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — 複数のファインチューニングされたモデルを干渉なしでマージするための基礎的な技術。
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — Backpropagateの複数回の実行マージャーが実装する特定のアルゴリズム。2025年12月のプレプリント。Backpropagateは、この論文の最初の既知のダウンストリームアダプターです。

## ライセンス

MIT — [LICENSE](LICENSE)を参照してください。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
