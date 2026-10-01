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

# 32B QLoRA、または7Bの完全なモデルを、1つのGPUで微調整します。Ollamaにデプロイします

大規模言語モデルの微調整を、**1つ**のGPUで実行します。GPUの実際のスペックに合わせてサイズを調整します。3行のPythonコードで、7B～32Bのモデルを、32GBのコンシューマー向けGPU（RTX 5090）で微調整します。1つのフラグ、`--full-ft-offload`を使用すると、7Bクラスのモデルの重みと勾配をホストRAM（LinuxまたはWSL2）に保持することで、完全な微調整を行います（速度は遅く、後述します）。さらに1つのコマンドでOllamaにエクスポートし、次に`ollama run`で微調整を行います。16GBまでスケールダウンできます。Windowsでは最高のパフォーマンスを発揮します。

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

これだけです。YAML設定ファイルはありません。`accelerate launch`のような複雑な設定もありません。「GGUF形式に変換する」ための別のチュートリアルもありません。CUDA GPUと、トレーニングデータを含むJSONLファイルがあれば、すぐに動作する微調整モデルを3行のコードで作成できます。

## インストール

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

オプションの機能が必要な場合は、次のいずれかのインストール方法に切り替えてください。

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

Docker をお使いの場合は、`docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` を実行することもできます。`linux/amd64` と `linux/arm64` の両方のイメージが提供されるため、Apple Silicon や ARM Linux 環境でもネイティブなイメージを使用できます。「コンテナ内で UI を実行する」ための標準的な `compose.yaml` ファイルは、リポジトリのルートにあります。このファイルと同じ場所に `ui-auth.txt` ファイルを作成し、`user:password` を記述して、`docker compose up` を実行すると、`http://127.0.0.1:7860` にアクセスしてログインできます（初回起動時にはフロントエンドがビルドされ、数分かかります）。実行履歴は `~/.backpropagate` に保存されます。

## Backpropagateがどのような位置にあるか

LLMの微調整には、いくつかの優れたライブラリがあります。それぞれ異なる点で優れています。

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — YAML設定を好み、コピーできるレシピのコミュニティが必要な場合
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — DPO/PPO/RLHFとWeb GUIが必要な場合
- **[Unsloth](https://github.com/unslothai/unsloth)** — 可能な限り最速のトレーニングが必要で、サポートされているモデルファミリーを使用している場合
- **[torchtune](https://github.com/pytorch/torchtune)** — Metaの公式のPyTorchネイティブのレシピを編集したい場合

Backpropagateは、不足しているオプションです。**1つのコンシューマー向けGPUで動作し、アダプターをトレーニングしてデプロイしたいユーザー向けの、3行のPython APIです。** YAMLもGUIも、オンラインRL（PPO/GRPO）も、マルチノードもありません。必要なループと、それが邪魔になるエクスポートステップだけです。

上記のライブラリのいずれかを試して、設定ファイルの複雑さにうんざりしたり、モデルファミリーの制限に遭遇したり、Windowsを優先するデフォルトが必要になった場合は、Backpropagateが最適です。

## 1つのGPUで微調整できるもの

Backpropagateは、実行をGPUのスペックに合わせて調整します。以下は、2026年9月30日に32GBのRTX 5090で測定された数値です（証拠：[`docs/receipts/2026-09-30-offload/`](docs/receipts/2026-09-30-offload/))。QLoRAのピークは、設定の最大のコンテキストウィンドウでバッチサイズ1の場合であり、これはその設定で最悪の場合です。短い例では、より少ないリソースを使用します。

| モデル | 方法 | 32GBのカードで測定 |
|---|---|---|
| **14B**（Qwen2.5-14B） | QLoRA | **25.0 GiB**（コンテキストサイズ4096の場合、ピーク。28.1 GiB予約済み）。 |
| 24B（Mistral-Small-24B） | QLoRA | 26.5 GiB（コンテキストサイズ4096の場合、ピーク。29.6 GiB予約済み）。 |
| **32B**（Qwen2.5-32B） | QLoRA | **ギリギリ収まる:** 28.8 GiB（コンテキストサイズ2048の場合、ピーク。30.7 GiB予約済み、約0.65 GiBの余裕あり）。 |
| 3B | `mode="full"`（GPU上での完全な微調整） | **22.0 GiB**（システム全体でピーク）、バッチサイズ4、コンテキストサイズ512で0.30秒/ステップ。そのうち7.5 GiBはページングされたオプティマイザーの状態であり、より小さなカードではホストRAMにスピルする可能性があります（未テスト）。 |
| **7Bクラス**（Qwen2.5-7B、7.6Bパラメータ） | `mode="full" --full-ft-offload` | **トレーニング:** 5.3 GiB VRAM、**30.8 GiBホストRAM**（32.2 GiB、保存時）、**14.7秒/ステップ**。LinuxまたはWSL2のみ。 |

このセッションでは再測定していません：7B QLoRA、Llama-3.1-8B（ゲートされたリポジトリ、テストマシンにトークンなし）、および3Bを超える純粋なGPUでの完全な微調整。それらの数値は、ドキュメントの他の場所に記載されており、概算です。

ほとんどのシングルGPUライブラリが、**24〜32B QLoRA**と**単一カードの7Bクラスの完全な微調整**のために、他の場所に誘導するのに対し、Backpropagateは、1つのコンシューマー向けカードでこれらを行い、結果をOllamaに直接エクスポートします。

**完全な微調整には2つの方法があります。** オフロードを使用しない場合、モデル、その勾配、およびオプティマイザーの状態はすべてGPUに配置されます。ライブラリは、検出されたVRAMによってモデルサイズを制限します（**16 GB → 4B、24 GB → 5B、32 GB → 6B**）。これらの制限は、メモリ計算から得られ、3Bまでしか測定されていません。`--full-ft-ceiling-billions`でオーバーライドします。

`--full-ft-offload`は、重みと勾配をホストRAMに保持し、GPUにストリーミングします（FSDP2 CPUオフロード）。測定されたコストは次のとおりです。

- **ホストRAM:** フィットチェックでは、10億パラメータあたり約3.7 GiBと10 GiBが必要であり、これは控えめな値です（7.6Bの場合、測定値は32.2 GiBで、39 GiB）。マシンがそれを保持できない場合、実行は事前に拒否され、数値が表示されます。7.6Bモデルは、28GBのWSL2メモリ制限の下では収まりません。約5Bが、そこで実用的な制限となります。
- **速度:** 7.6Bの場合、14.7秒/ステップ（バッチ1）、3Bの場合、5.1秒/ステップ（バッチ4）。GPUで3Bをバッチ4で実行した場合、約0.63秒/ステップです。モデルがオフロードなしでは収まらない場合にのみ使用してください。より高速なバージョンが計画されています。
- **オプティマイザー:** Adafactor、AdamWではありません。重みはbf16のままになり、各更新は確率的丸めを使用して書き戻されます。fp32コピーはありません。
- **品質:** 1つの3Bの実行（150ステップ、保留された損失、1つのシード）では、通常の完全な微調整の改善の約85％に達しました（2.45 → 1.93、通常の完全な微調整は2.45 → 1.84）。1つのシードはベンチマークではありません。
- **範囲:** 単純な教師あり微調整。パッキング、応答のみのマスキング、中間チェックポイント、再開はありません。LinuxまたはWSL2のみ（FSDP2にはNCCLが必要です）。Windowsネイティブでは、`DEP_FSDP_UNAVAILABLE`で停止します。
- **まだテストされていません:** 長時間の実行、1を超える勾配累積、および物理的な64GBマシン（テストマシンはより多くのRAMを持っていましたが、テストでは60GiBの制限が適用されました）。

モデルが適合しない場合、コード `RUNTIME_FULL_FT_MODEL_TOO_LARGE` で終了し、終了方法を指定します。詳細については、[完全なファインチューニングハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/) を参照してください。

### 16GBにスケールダウン

16GBの範囲（RTX 4080 / 5080 / 4070 Ti Super）は、依然として最上位の性能を発揮します。7B QLoRA、および約3Bモデル（SmolLM3-3B、Qwen2.5-3B、Llama-3.2-3B/1B）の真の完全なファインチューニングを、コード `mode="full"` を使用して実行できます（32GBカードで3Bの場合、22.0GiBを計測。そのうち7.5GiBはホストRAMにスピルするページングされたオプティマイザーの状態です。16GBカードで許容範囲内で動作するかどうかはテストされていません）。コード `--full-ft-offload` を使用すると、GPUが保持するデータ量は大幅に少なくなります。テストカードでVRAMが制限されている場合、6GiBの制限下でトレーニングされた3Bモデル、および8GiBの制限下でトレーニングされた4Bおよび7.6Bモデルが使用されます。これらは32GBカードでのエミュレートされた制限であり、実際の8GBハードウェアでの実行ではありません。同じコードが、検出されたカードに適合するバッチサイズと上限を選択します。

2ビット量子化（AQLM / QuIP#）は、**対象外**です。2ビットのベースモデルを、完全精度ウェイトにクリーンにマージすることはできません。これにより、マージ可能なアダプター → GGUF → Ollamaのエクスポートという一連の流れが中断されます（このパイプラインの目的はこれです）。代わりに、Backpropagateには、QLoRA、コード `mode="full"`、コード `--full-ft-offload`、およびFP8計算パス（コード `--fp8`、Blackwell/Hopper）などの機能が搭載されており、これらはすべてマージ可能でエクスポート可能です。

## Backpropagateが適さない用途

以下の用途に該当する場合、別のライブラリを使用する方が良い結果が得られます。Backpropagateは適切な選択肢ではなく、無理に動作させようとすると、適切なツールを使用するよりも多くの労力が必要になります。使い始める前にこのセクションを読むことで、インストールと再試行のサイクルを回避できます。

- **13B以上のモデルの完全パラメータによるファインチューニング** — Backpropagateは、32GB GPUで約6Bまで、コード `--full-ft-offload` を使用して7Bクラスのモデルまで、完全なファインチューニングを実行できます（[この範囲](#what-you-can-fine-tune-on-one-gpu) を参照）。13B以上のモデルの完全なファインチューニングには、マルチGPU FSDPまたはより大きなカードが必要です。計算リソースを投入する前に、両方の可能性を検討してください。[Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)は、LoRAがすべてのレイヤーに適用され、データセットがアダプターの容量に収まる場合、1回のパスあたりの計算量が約3分の2で、完全なファインチューニングに匹敵すると報告しています。[Biderman et al. 2024](https://arxiv.org/abs/2405.09673)は、標準的な低ランク設定では、LoRAはコードと数学の問題において、完全なファインチューニングよりも大幅にパフォーマンスが劣り、一方で忘却が少ないことを発見しました。指示に従う、ペルソナ、およびスタイルに関するタスクで、適度なデータセットを使用する場合、32BまでのQLoRAは、通常、1つのカードを使用する場合に最適な選択肢です。
- **オンラインRL — PPO / GRPO / RLVR** — Backpropagateは、単一段階のSFTと参照なしのプリファレンスチューニング（v1.5ではORPO、v1.6ではSimPO + KTO）を実行します。ただし、オンライン強化学習（PPO、GRPO、またはRLVR）は実行しません。これには、報酬モデルまたはトレーニングステップの上に構築された生成とスコアリングのループが必要です。これらの場合は、TRLまたはLLaMA-Factoryを直接使用してください。（参照なしのプリファレンスチューニングは、単一段階の範囲に適合します。なぜなら、メモリに保持する必要のある個別の参照モデルがないからです。詳細については、[クイックスタート](#quick-start)のORPOの注記を参照してください。）
- **マルチノードトレーニング** — 1つのマシン上の単一のGPUのみ。1つのマシン上のマルチGPUは機能しますが、公式にはサポートされていません（コード `accelerate launch` を使用）。
- **CUDA環境でのmacOSトレーニング** — Apple SiliconにはCUDAがないため、CUDAパスはLinuxまたはWindowsのボックス上でNVIDIA GPUを使用して実行されます。トレーニングされたモデルは、Ollamaを介してMac上で引き続き実行できます。**実験的で検証されていないプレビュー版**のMLX環境（コード `--backend mlx`）は、Apple Silicon上でLoRAアダプターをネイティブにトレーニングします。詳細については、[Apple Silicon（MLX）](#apple-silicon-mlx--unverified-preview) を参照してください。これはLoRA-SFT専用であり、実際のシリコンで**検証されていません**（サポートなし）。したがって、LoRA SFT（ORPO、完全なファインチューニング、FP8、複数回の実行）以外のものについては、CUDA環境を使用することをお勧めします。
- **テスト済みのモデルファミリー以外のもの** — Qwen 2.5 / 3.5（7B / 4B）、Phi-4-mini-3.8B、SmolLM3-3B、Llama 3.2（3B / 1B）、Mistral 7B。他のモデルも多くの場合機能しますが、CIで固定されていません。

これらの機能が必要な場合は、上記のライブラリのいずれかを使用してください。それらのライブラリの方が適しています。

## Backpropagateが提供するもの

1つのインストールで、次の4つの機能を提供します。

**1. 実際の3行のAPIで、設定ファイルなしで実行できます。**
このREADMEの冒頭にあるスニペットは、最初から最後まで実行されます。コード `accelerate config`、YAML、Hydraのオーバーライドは必要ありません。コード `Trainer(model).train(data)` を使用するだけで、ファインチューニングが完了します。

**2. 実際に動作するWindowsサポート。**
ほとんどのMLライブラリは、Windowsを後回しにします。Backpropagateは、Windows + RTX 5080で最初にテストされます。このライブラリは、ランタイムの癖を処理します。Windowsのマルチプロセッシングがクラッシュしないように、データを事前にトークン化する方法を認識しています。RTX 40/50カードで動作が停止する可能性があるため、xformersを自動的に無効にし、データローダーの設定を選択することで、動作が停止しないようにします。これらのことを知る必要はありません。単に実行するだけです。

**3. 無人での実行用に構築されています。**
トレーニングには数時間かかります。監視する必要はありません。Backpropagateは、放置して実行できるように設計されています。

- GPUメモリが不足した場合、バッチサイズを自動的に半分にし、最大3回まで再試行します。手動で調整する必要はありません。
- GPUが過熱した場合、冷却されるまで一時停止し、その後続行します。
- すべてのチェックポイントはアトミックに書き込まれます。ラップトップが保存中にクラッシュした場合でも、以前の良好なチェックポイントはそのまま残ります。
- すべてのトレーニング実行には、一意のIDが割り当てられ、すべてのログ行、すべてのチェックポイント、およびすべてのWeights & Biasesのエントリにスタンプが付けられます。問題が発生した場合、1つのIDを使用すると、メンテナーはすべてを関連付けることができます。
- エラーには安定したコード（コード `RUNTIME_GPU_OOM`、コード `DEP_OLLAMA_REGISTRATION_FAILED`など）が付属しているため、ログをgrepして、[トラブルシューティングガイド](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)で修正方法を見つけることができます。CUDA固有の障害には、専用の[CUDAトラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)があります。

**4. トレーニング済みのアダプターから、1つのコマンドで `ollama run` を実行します。**
多くのライブラリがモデルをトレーニングします。しかし、実際にそれを使用したいときに、それらのライブラリが邪魔にならないものはほとんどありません。Backpropagate は、GGUF（Ollama が使用する形式）へのエクスポートと、1 つのコマンドで Ollama モデルの登録を行います。わずか 30 秒で、「トレーニング完了」から「ファインチューンモデルとチャットできる」状態に移行できます。

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

`backprop train` はアダプターを `./output` に書き込みます（必要に応じて `--output` で変更します）。Python では、同じ処理を次のように記述します。

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")
```

Python API には、`pip install "backpropagate[standard]"` を使用した仮想環境を使用してください。`pipx` は `backprop` コマンドを独自の環境にインストールするため、`import backpropagate` はそれを検出できません。

**GGUF エクスポートに必要なもの。** エクスポートは、アダプターをベースモデルにマージし、llama.cpp のコンバータースクリプトを使用して変換します。次のいずれかが必要です。

- llama.cpp の **ソースコード** (`git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp`) と、同じ環境に `pip install sentencepiece protobuf` をインストールするか、
- 独自の llama.cpp がすでにビルドされている Unsloth を使用します。

`--ollama` を使用すると、`q4_k_m` 量子化は `ollama create` によって実行されるため、コンパイルする必要はありません。Backpropagate は、llama.cpp をビルドするために Unsloth がシステムパッケージをインストールすることを許可しません。許可する場合は、`BACKPROPAGATE_UNSLOTH_AUTO_INSTALL=1` を設定してください。詳細: [エクスポート](https://mcp-tool-shop-org.github.io/backpropagate/handbook/export/)。

独自のデータの場合、JSONL 形式で、1 行に 1 つの例を記述します。

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

評価は、設計上、判断を必要としません。保留された損失と、決定的なタスクメトリック (`normalized_exact_match`、`token_f1`、`contains`、`regex`、`pass_rate`) を使用します。LLM ジャッジを使用する場合は、それを `backprop generate` の出力に適用してください。詳細: [レシピ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/)。

### 嗜好性チューニング (ORPO、SimPO、KTO)

単純なデモンストレーションではなく、嗜好性に基づいてトレーニングします。ORPO は参照を必要とせず、1 段階で実行されます。嗜好性シグナルを SFT ステップに組み込むため、個別の報酬モデルや参照モデルは必要なく、3 行の形状は変更されません。`--method orpo` (CLI) または `method="orpo"` (Python) を渡し、`{prompt, chosen, rejected}` (または `{chosen, rejected}` のみ) 行のデータセットを渡します。

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

デフォルトの学習率は、ORPO に対して自動的に `8e-6` に低下します (損失は単純な SFT よりも鋭くなります)。`--orpo-beta` (デフォルトは `0.1`) を調整して、オッズ比ペナルティの重みを調整します。ORPO は `mode="lora"` のみです。

**v1.6 での新機能 — SimPO と KTO。** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) は、長さで正規化された報酬を使用し、参照を必要としません。ORPO と同じペアの `{prompt, chosen, rejected}` データを入力します (`--simpo-beta`、`--simpo-gamma`)。`--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) は、**ペアでない** `{prompt, completion, label}` データを入力します。つまり、キュレーションされた A/B ペアではない、例ごとの肯定/否定のフィードバックです。望ましい/望ましくない損失の重みをラベルの数から自動的に調整します。どちらも `mode="lora"` のみであり、単一の GPU SFT の範囲内に収まります (個別の参照モデルはありません)。使用するものを選択するには、[嗜好性チューニングハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) を参照してください。オンライン RL (PPO/GRPO) については、[Backpropagate が適さないもの](#what-backpropagate-is-not-for) を参照してください。

### 推論トレース SFT (R1 蒸留)

推論モデルを簡単に蒸留します。`--reasoning-trace` (CLI) または `Trainer(..., reasoning_trace=True)` (Python) を渡し、アシスタントの応答内に `<think>...</think>` の連鎖思考を保持するトレースを入力します。これは、[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 蒸留の純粋な SFT の半分であり、RL は必要ありません。Backpropagate は、`<think>` をトレーニングターゲットに保持し、空の/長すぎるトレースを削除します (トレース長のフィルタリング)、および、より長い CoT に対してデフォルトの `max_seq_length` を 8192 に引き上げます。重要な点として、`<think>` は **プレーンテキスト** のままです。特別なトークンや、埋め込みのリサイズは行われません。そのため、マージされた GGUF は、他のファインチューンと同様に、Ollama にエクスポートできます。SFT のみです。データセットの形状と調整可能なトークンバンドについては、[推論トレースレシピ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) を参照してください。

### Apple Silicon (MLX) — 検証されていないプレビュー

> ⚠️ **検証されていないプレビュー — サポートされている機能セットの一部ではありません。** MLX レールは構築され、ユニットテストされていますが、実際の Apple Silicon 上でドッグフード検証は **行われていません** (`mlx-lm` は Apple 専用であり、Backpropagate が開発されている NVIDIA リグでは実行できません)。以下はすべて実験的なものとして扱い、ご自身の責任で使用し、M シリーズの Mac で実行した場合は、[バグの報告](#reporting-bugs) を行ってください。

**1 つの API、2 つのレール。** CUDA は、検証済みの標準的なバックエンドです。MLX は、M シリーズの Mac で Apple の [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) ツールチェーンを使用してトレーニングする 2 番目のレールです (統合メモリ、CUDA は不要)。3 行の形状は、ハードウェアに基づいてレールを選択します。`backend='auto'` (デフォルト) は、NVIDIA 上では CUDA に、Apple Silicon 上では MLX にルーティングするため、既存の CUDA リグはバイト単位で同一です。

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

MLX レールは、**LoRA SFT のみ** です。ORPO、FP8、`mode='full'`、複数回の実行はできません (それぞれ `CONFIG_INVALID_SETTING` で拒否されます。それらの機能を使用する場合は、NVIDIA ボックスで `backend='cuda'`/`'auto'` を使用してください)。結果として得られるアダプターは、プレーンな safetensors であり、CUDA レールと同じパスを通じて Ollama にエクスポートされます。

> `--backend mlx` を Apple 以外のホストで強制すると、`CONFIG_INVALID_SETTING` エラーが発生します。Mac で `mlx_lm` ツールチェーンが見つからない場合、`DEP_MLX_UNAVAILABLE` が発生します。

より包括的なワークフロー (ファインチューンして HF Hub にプッシュ、OOM 後に再開、長期間のキャンペーンにわたる SLAO の複数回の実行など) については、[ハンドブックのレシピページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/) を参照してください。

### Web UI (オプション)

Python ではなく、クリックして操作したい場合は、UI エクストラをインストールして起動します。

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

URL を開き、`http://127.0.0.1:7862/?token=...` を実行します（起動するたびに新しいトークンが生成されます。最初の起動ではフロントエンドがビルドされ、1 ～ 2 分かかる場合があります）。これは、データセットを参照したり、形式を検証したり、トレーニング構成を視覚的に組み立てたりするためのローカル Web インターフェイスです。トレーニング自体は `backprop train` を介して実行されます（UI を介したトレーニングはロードマップにあります。現在の「開始」ボタンにはそのメモが表示されます）。デフォルトでは、UI はローカルでのみ実行されます。他のデバイスからアクセスできるようにするには、以下にある [Web UI](#web-ui) を参照して、`--share` + `--auth` のセキュリティ要件を確認してください。

## 複数回のトレーニング

複数のデータセットにわたって段階的にファインチューニングを行いたい場合（たとえば、毎週新しいトレーニングデータを入手し、以前に学習したことを忘れることなく追加したい場合）、Backpropagate の `multi_run` モードを使用してください。

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

これは、5 回のトレーニングパスを実行し、各パスの間にアダプターをマージすることで、以前の知識を保持しながら新しい例を組み込みます。この手法は、最近の継続学習の研究に基づいています。詳細については、この README の下にある [参考文献](#references) を参照してください。

CLI バージョン：

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## チェックポイントからの再開

4 回目の実行でクラッシュした 5 回のトレーニングは、再開できます。複数回のトレーニングセッションでは、各実行の実行 ID がディスク上の履歴とチェックポイントマニフェストに書き込まれるため、中断したところから再開するには、次のコマンドを実行するだけです。

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

デフォルトの動作（`backprop multi-run`、`--resume` なし）では、同じ出力ディレクトリにある進行中のエントリを自動的に検出し、続行します。クリーンな開始を強制するには、新しい出力ディレクトリを指定します。

## トレーニング履歴

各 `backprop train` および `backprop multi-run` の実行では、`<output>/run_history.json` に行が記録されます。記録される内容は、使用されたモデル、データセット、ハイパーパラメータ、ステータス、最終的な損失、および損失履歴です。過去の実行を一覧表示して確認できます。

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## 実験の追跡

Backpropagate は、インストールされている実験追跡ツール（Weights & Biases、TensorBoard、MLflow）を自動的に検出し、それらを連携させます。`wandb` がインストールされており、ログインしている場合、各実行は自動的に W&B にログを記録し、実行名はディスク上の実行 ID と一致します。これにより、W&B、ログ、および `run_history.json` を 1 つの識別子を使用して検索できます。

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

`Trainer(report_to=["wandb"])`、`Trainer(report_to=["tensorboard"])`、または `Trainer(report_to="none")` を使用してオーバーライドし、連携を無効にすることができます。

## Web UI

Reflex Web インターフェイスは、オプションで有効にできます。`pipx install "backpropagate[ui]"` を使用してインストールし、起動します。

```bash
backprop ui --port 7862
```

UI はローカルで実行されます。URL を開き、`http://127.0.0.1:7862/?token=...` を実行します。`--auth` がない場合、起動するたびに新しいトークンが生成され、UI はそのトークンがないリクエストを拒否します。現在、UI はワークフローの **参照 / 検証 / 構成** の部分をカバーしています。データセットを指定し、自動検出された形式と統計を確認し、モデルを選択し、実行構成を組み立てます。**実行の起動は CLI から行われます**（`backprop train` / `backprop multi-run`）。UI 内の「開始」ボタンには、その旨を示すメモが表示されます。UI を介したトレーニングは、今後の計画です。それまでは、UI はオンランプであり、CLI はトリガーとなります。

他のデバイス（ネットワーク上の他のユーザー、パブリック URL など）からアクセスできるようにするには、`--share`（または `--host`）を `--auth` と組み合わせる必要があります。

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` で `--auth` がない場合、エラーが発生して終了します。その理由は、`--share` がインターネット上の誰でもアクセスできる URL を公開するため、認証がない場合、誰でもトレーニングパイプラインを制御し、Hugging Face トークンを読み取ることができるためです。これを無効にするオプションはありません。資格情報を設定したくない場合は、代わりに SSH ポートフォワーディングを使用してください。

```bash
# On the client:
ssh -L 7862:localhost:7862 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open the URL the server printed (http://127.0.0.1:7862/?token=...) locally
```

完全な脅威モデルについては、[handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) を参照してください。

UI からのファイルシステムへの書き込みは、単一のディレクトリにサンドボックス化されます。

- デフォルト：`~/.backpropagate/ui-outputs`
- オーバーライド：`BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own` を設定
- オーバーライドは、許可リストで検証されます。システムまたは資格情報パス（`/etc`、`~/.ssh`、`~/.aws`、`C:\Windows\System32` など）は拒否されます。

## プラットフォームに関する注意点

**要件：** Python 3.10+ · CUDA GPU（8GB+ VRAM）· PyTorch 2.0+

Python 3.10 は、少なくとも v1.6 までサポートされています。2026 年 10 月にアップストリームのサポートが終了し、その後の最初のリリースで削除される予定です。新しいインストールでは、Python 3.11 または 3.12 を使用することをお勧めします。3.11 は最もテストされたバージョンです。

Backpropagate は、さまざまなプラットフォームでのトレーニングにおけるランタイムの癖に対処しますが、インストール時の問題を修正することはできません。最も一般的な問題は次の 2 つです。

- **間違った CUDA ホイール。** PyTorch は、CUDA バージョンごとに 1 つのバイナリとして公開されます。間違ったものを選択すると、サイレントに CPU のみを使用する PyTorch がインストールされ、トレーニングは非常に遅くなります。ドライバーに適したものを選択するには、<https://pytorch.org/get-started/locally/> のホイールピッカーを使用してください。`nvidia-smi` を実行して、ドライバー/CUDA バージョンを確認してください。
- **Windows + GGUF エクスポート。** `[export]` の追加ビルドでは、ソースから `llama-cpp-python` をビルドします。これには、Visual Studio Build Tools（C++ コンポーネント）と CMake が必要です。

**macOS：** CUDA レールはサポートされていません（CUDA がありません）。CUDA を使用するように設定された `trainer.train()` は `DEP_GPU_NOT_AVAILABLE` を発生させ、トレーニングされたアダプターは Ollama を介して Mac で実行できます。**実験的で、検証されていないプレビュー版**の MLX レール（`--backend mlx`、`pip install 'backpropagate[mlx]'`）は、Apple Silicon 上で `mlx_lm.lora` を介して LoRA アダプターをネイティブにトレーニングします。LoRA SFT のみで、**実際のシリコンでドッグフードテストは行われていません**（[Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview) を参照）。CUDA パスを使用する場合、または ORPO / 完全なファインチューニング / FP8 / 複数回の実行を行う場合は、CUDA Linux または Windows マシンを使用してください。

詳細なインストール手順とトラブルシューティングガイドについては、[トラブルシューティングハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) を参照し、ドライバー/VRAM/xformers/bf16 対 fp16 の問題については、[CUDA トラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) を参照してください。

## CLI

すべての Python API には、CLI の対応する機能があります。

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

完全なリファレンスは、[CLI ハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) または `backprop <subcommand> --help` にあります。

## 構成

すべての設定は、`BACKPROPAGATE_` プレフィックスを使用して環境変数でオーバーライドできます。

| 変数 | デフォルト | 注釈 |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | 自動 | JSONまたはコンソールログを強制的に出力 |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | デフォルトモデル |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | 学習率 |
| `BACKPROPAGATE_LORA__R` | `256` | LoRAランク（v1.3のデフォルト。v1.2.xのデフォルト値16にするには、`--lora-preset=fast`を指定） |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | UIファイルシステムサンドボックス |

ネストされたキーは、二重アンダースコアを使用します（`MODEL__NAME`、`MODEL_NAME`ではありません）。完全なリファレンスは、[env-varsハンドブックページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/)にあります。

## モデルプリセット

| プリセット | VRAM | ライセンス | 注釈 |
|---|---|---|---|
| Qwen-3.5-4B | 約8GB | Apache 2.0 | 5B以下のモデルに対する推奨されるデフォルト。このサイズで最高の品質。 |
| Phi-4-mini-3.8B | 約8GB | MIT | 推論/数学/コードに強い。厳格なライセンスで提供。 |
| SmolLM3-3B | 約6GB | Apache 2.0 | 完全にオープンなレシピ。ネイティブ64Kコンテキスト。 |
| Qwen 2.5 7B | 約12GB | Apache 2.0 | 既存のデフォルト。従来の7Bプリセットの中で最高の品質。 |
| Qwen 2.5 3B | 約8GB | Qwen-Research | ⚠ 研究ライセンス — 商業利用の前に、Qwenライセンス条項を確認してください。 |
| Llama 3.2 3B | 約8GB | Llama Community | Qwen 3Bの優れた代替手段で、制限も緩やか。 |
| Llama 3.2 1B | 約6GB | Llama Community | 小規模なカードでの迅速な実験用。 |
| Mistral 7B | 約12GB | Apache 2.0 | Qwen 7Bと同等で、チャットテンプレートは異なる。 |
| Llama-3.1-8B | 約7〜8GB（QLoRA） | Llama-3.1-Community | 8B QLoRA、128Kネイティブコンテキスト（>700M-MAU条項には、別途Metaライセンスが必要）。 |
| **Qwen2.5-14B** | 4096コンテキストでピーク時25GiB（QLoRA） | Apache 2.0 | **32GBの環境で毎日使用できるモデル。**ランク/アルファ32、8ビットAdamW。4ビットの重みだけで約8.5GB。完全な4096トークンウィンドウには、残りの容量が必要。 |
| Mistral-Small-24B | 4096コンテキストでピーク時26.5GiB（QLoRA） | Apache 2.0 | 32GBのカードで24B QLoRA。4ビットの重みだけで約18GB。 |
| **Qwen2.5-32B** | 2048コンテキストでピーク時28.8GiB（QLoRA） | Apache 2.0 | **32GBの環境で最大限に活用できるモデル。**8ビットAdamWで、`max_len 2048`に収まる。 |

他のモデルも多くの場合機能します。上記の行は、厳選されたプリセットです。14B〜32Bの範囲は、32GBのカード用にQLoRAで調整されています（測定された範囲）。ランク256 / Biderman 2024 + Thinking Machines 2025のすべての線形ターゲットにするには、`--lora-preset=quality`（デフォルト）を指定するか、v1.2.xのフットプリントが必要な場合は、従来のランク16 / q+vターゲットにするには、`--lora-preset=fast`を指定します。

## トラブルシューティング

最も一般的な初回実行時のエラーの簡単なインデックス。完全な逆インデックスは、[トラブルシューティングハンドブックページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)にあります。ドライバー/VRAM/混合精度に関する詳細については、[CUDAトラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)を参照してください。

| 症状 | エラーコード | 修正 |
|---|---|---|
| GPUのメモリがトレーニング中に不足 | `RUNTIME_GPU_OOM` | 自動 — Backpropagateは、バッチサイズを半分にし、最大3回再試行します。無効にするには、`Trainer(oom_recovery=False)`を指定します。より小さいサイズに強制するには、`--batch-size 1`を指定します。 |
| HuggingFaceが401 /「モデルが見つかりません」を返す | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login`を指定して再試行します。タイプミスの場合、<https://huggingface.co/models>から正確なIDをコピーします。 |
| `register_with_ollama`接続拒否 | `DEP_OLLAMA_REGISTRATION_FAILED` | デーモンを開始します：`ollama serve`。 <https://ollama.com>からインストールします。再試行可能です。 |
| チェックポイント保存中にディスクがいっぱい | `STATE_CHECKPOINT_INVALID` | アトミック書き込みにより、クラッシュ時に`.partial`ディレクトリが残ります。削除しても安全です。前の正常なチェックポイントはそのままです。 |
| GPUの過熱によりトレーニングが一時停止 | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | 自動 — Backpropagateは、温度のしきい値で一時停止し、GPUが冷却されると再開します。頻繁に発生する場合は、エアフローを改善します。 |
| `backprop ui --share`拒否 | `RUNTIME_UI_AUTH_NOT_ENFORCED` | `--auth user:password`を指定するか、代わりにSSHポートフォワーディングを使用します（[Web UI](#web-ui)を参照）。 |
| GGUFエクスポートが初回で失敗 | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`。Windowsでは、Visual C++ Build Tools + CMakeも必要です。 |

## バグの報告

何らかの処理が失敗した場合、Backpropagateは起動時に`run_started run_id=<uuid>`のような行を出力し、同じIDをすべてのログ行、すべてのチェックポイント、およびすべてのWeights & Biasesのエントリーに紐付けます。**バグ報告には必ず`run_id`を含めてください**。これにより、開発者は特定の実行に関するすべての情報を関連付けて確認できます。

優れたバグ報告には、次のものが含まれます。

1. **`run_id`** — 起動時に出力されるUUID。1つのUUIDにより、メンテナンス担当者は、その特定の実行に関連するすべてのログ行、すべてのチェックポイント、およびすべてのWeights & Biasesエントリを関連付けることができます。
2. **エラーコード** — stderrの`[CODE_NAME]: message`行。安定したコードのカタログについては、[エラーコード](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/)を参照してください。
3. **編集されたトレースバック。** stderrは、非詳細モードでは自動的に編集されます（Bearerトークン、`sk-*`、`hf_*`、AWSキー、`password=` / `token=` / `api_key=`のペアが削除されます）。貼り付けても安全です。完全な編集されていないトレースバックについては、`BACKPROPAGATE_DEBUG=1`（または`--verbose`）で再実行し、投稿する前に確認してください。
4. **`backprop info`出力。** 1つのコマンドは、Python / PyTorch / CUDA / GPUモデル / VRAM / OS / インストールされた追加機能を出力します。メンテナンス担当者がプラットフォーム固有の回帰を特定するために必要なすべての情報が含まれています。

[バグ報告テンプレート](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml)では、これらの項目について具体的に記述するように促すため、問題の分類と対応が迅速に進みます。質問、アイデア、または「これは想定された動作ですか？」といった内容は、[GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions)で行ってください。セキュリティに関する問題は、[GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new)フォームを通じて非公開で報告してください。ポリシーと対応期間については、[SECURITY.md](SECURITY.md)を参照してください。

## プライバシー

すべてのトレーニングは、ローカルのGPU上で行われます。Backpropagateは、Hugging Faceからモデルをダウンロードする場合を除き、ネットワークへのリクエストを行いません（ダウンロードはユーザーが開始します）。テレメトリーは行わず、クラウドへの依存もありません。

## 参考文献

Backpropagateのデフォルト設定と複数回のトレーニングモードは、最近の研究に基づいています。関連する技術にご興味がある場合は、以下の資料をご覧ください。

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — LoRAを紹介する基礎論文。Backpropagateは、この技術を用いてアダプターを効率的にトレーニングします。
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — ランク256で、すべての線形ターゲットを使用したLoRAが、ほとんどのポストトレーニングタスクにおいて、計算量の67%で完全なファインチューニングと同等の品質を達成するという実証的な証拠。Backpropagateのv1.3のデフォルトLoRA構成を決定づけるものです。
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — 高いLoRAランクで必要な学習率と完全なファインチューニングとの間の10倍の補正を特定した、実践的なフォローアップ論文。
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — ニューラルネットワークが新しいデータでファインチューニングを行うと、以前のトレーニング内容を「忘れてしまう」理由を最初に説明した論文（EWC — Elastic Weight Consolidation）。
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA。新しいアダプターを直交部分空間に制限することで、LoRAを継続学習に利用する、以前のアプローチ。
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — 複数のファインチューニングされたモデルを干渉なしでマージするための基礎的な技術。
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — Backpropagateの複数回のトレーニングマージャーが実装する具体的なアルゴリズム。2025年12月のプレプリントであり、Backpropagateは、この論文を最初に採用したダウンストリームアプリケーションです。

## ライセンス

MIT — [LICENSE](LICENSE)を参照。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
