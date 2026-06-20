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

# アダプターをトレーニングします。Ollamaにデプロイします。次に進みます

Backpropagateは、単一のGPUで大規模言語モデルを微調整するためのPythonライブラリです。3行のコードで、16GBのカード上で7Bモデルをトレーニングできます。さらに1つのコマンドで、微調整したモデルをOllamaにエクスポートし、`ollama run`コマンドで実行できるようにします。Windowsで最適に動作します。

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

これだけです。YAML設定ファイルはありません。`accelerate launch`コマンドも必要ありません。「変換してGGUF形式にする」という別のチュートリアルもありません。CUDA GPUと、トレーニングデータを含むJSONLファイルがあれば、すぐに微調整を開始できます。

## インストール

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

オプションの機能が必要な場合は、以下のいずれかのインストール方法に切り替えてください。

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

Dockerを使用しますか？`docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest`も使用できます。`linux/amd64`と`linux/arm64`の両方のイメージが提供されるため、Apple SiliconおよびARM Linuxユーザーはネイティブイメージを使用できます。コンテナ内でUIを実行するための標準的な`compose.yaml`ファイルは、リポジトリのルートにあります。`docker compose up`コマンドを実行すると、Web UIが`http://localhost:7860`で起動し、`~/.backpropagate`ボリュームが永続的にマウントされます。

## Backpropagateがどのような位置にあるか

LLMの微調整には、いくつかの優れたライブラリがあります。それぞれ異なる点で優れています。

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — YAML設定を好み、既存のレシピを参考にしたい場合に最適です。
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — DPO/PPO/RLHFを使用し、Web GUIが必要な場合に最適です。
- **[Unsloth](https://github.com/unslothai/unsloth)** — 可能な限り高速なトレーニングが必要で、サポートされているモデルファミリーを使用する場合に最適です。
- **[torchtune](https://github.com/pytorch/torchtune)** — Metaが提供する、PyTorchネイティブのレシピを編集したい場合に最適です。

Backpropagateは、不足している選択肢です。**単一のコンシューマーGPUでアダプターをトレーニングし、それをデプロイしたい個人のオペレーター向けの、3行のPython APIです。** YAMLもGUIも、オンラインRL（PPO/GRPO）も、マルチノードもありません。必要なループと、その邪魔になるエクスポートステップだけです。

上記のいずれかのライブラリを試して、設定ファイルの操作に苦労したり、モデルファミリーの制限に遭遇したり、Windowsを優先するデフォルト設定が必要になった場合は、Backpropagateが最適です。

## 16GBのコンシューマーGPUで微調整できること

16GBのカード（RTX 4080 / 5080 / 4070 Ti Super）で実際に使用できる範囲は次のとおりです。

| モデル | 方法 | 状態 |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | 快適。完全なシーケンス長で、余裕があります。 |
| SmolLM3-3B / Qwen2.5-3B / Llama-3.2-3B / Llama-3.2-1B | `mode="full"`（完全な微調整） | v1.4 — `backprop train`コマンドまたは`Trainer(..., mode="full")`で`--mode=full`を指定します。完全な精度（bf16）の重みをロードします。4ビット、アダプターは使用しません。勾配チェックポイントとページ化された8ビットAdamにより、フットプリントを16GB以内に収めることができます。 |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | 標準。約7〜8GB。Backpropagateのデフォルトプリセット。 |
| Llama-3 13B | QLoRA + サンプルパッキング | ぎりぎりですが、動作します。短いシーケンスを使用してください。 |
| Mixtral 8x7B（合計470億パラメータ） | — | 範囲外 — 2ビット（AQLM / QuIP#）は、マージ可能なアダプターとGGUFエクスポートの契約を破るため、[v1.5の概要](docs/V1_5_BRIEF.md)で廃止されました。16GBのカードでは、≤8Bのベースモデルを使用してください。 |

`mode="full"`は、最大**40億パラメータ**のモデルをサポートします。上記の完全な微調整行にある4つのプリセットは、実際には約30億（実際のパラメータ数は3.08〜3.24億）であり、16GBのカードに適合します。3.8〜40億のクラス（Phi-4-mini-3.8B、Qwen-3.5-4B）も上限に達しますが、完全な微調整には**24GB以上の**カードが必要です。重みと勾配だけでも16GBに近づき、オプティマイザーと活性化も考慮する必要があります。そのため、16GBのカードでは、これらのモデルに対して`mode="lora"`を使用してください（LoRA行にあります）。40億を超えるモデルは、`RUNTIME_FULL_FT_MODEL_TOO_LARGE`というエラーで終了します。

2ビット量子化（AQLM / QuIP#）は、**範囲外**です。[v1.4で検討された後、[v1.5の概要](docs/V1_5_BRIEF.md)で廃止されました。2ビットのベースモデルを、完全な精度の重みにクリーンにマージすることはできません。これにより、Backpropagateのマージ可能なアダプター→GGUF→Ollamaエクスポートの契約（パイプラインの目的）が破られます。代わりに、Backpropagateが提供するヘッドルームは、v1.5の**FP8コンピューティングパス**（`--fp8`、Blackwell / Hopper）と、≤40億のモデルに対する`mode="full"`です。どちらもマージ可能で、エクスポート可能です。

30億以下のモデルの場合、16GBで完全な微調整（LoRAだけでなく）が可能になり、v1.4で`mode="full"`として提供されます。`Trainer(..., mode="full")`または`backprop train --mode=full --model phi-4-mini-3.8b`を指定して有効にします。40億を超えるモデルの場合、`RUNTIME_FULL_FT_MODEL_TOO_LARGE`というエラーが発生し、LoRAと40億以下のプリセットが代替手段として提案されます。構成の計算と、Biderman 2024 / Thinking Machines 2025による品質比較については、[完全な微調整のハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/)をご覧ください。70億以上のモデルの場合、完全な微調整には24GB以上のGPUが必要です。A100クラウドレンタルを検討するか、最新の研究では、ほとんどのポストトレーニングタスクで完全な微調整の品質に匹敵することが示されているため、LoRAを使用してください（[アンチピッチセクション](#what-backpropagate-is-not-for)に参考文献を参照）。

## Backpropagateが適さない場合

以下のユースケースに該当する場合は、別のライブラリを使用する方が良い結果が得られます。Backpropagateは適切な選択肢ではなく、無理に使用しようとすると、適切なツールを選択するよりも多くの労力がかかります。インストールを開始する前に、このセクションを読んでください。

- **7B以上のモデルに対するフルパラメータのファインチューニング** — BackpropagateはLoRA / QLoRAを使用し、すべての重みを更新するのではなく、小さなアダプターをトレーニングします。7B以上のモデルの場合、フルファインチューニングには24GB以上のGPUメモリが必要であり、16GBのコンシューマーカードでは実行できません。3B以下のモデルの場合、16GBでフルファインチューニングが可能であり、v1.4で`mode="full"`として提供されます（CLIで`Trainer(..., mode="full")`または`--mode=full`を渡します。4Bを超えるモデルの場合、ハードゲートが`RUNTIME_FULL_FT_MODEL_TOO_LARGE`を発生させ、LoRAと4B未満のプリセットをリカバリとして指定します）。より大きな視点として、最近の研究（[Biderman 2024](https://arxiv.org/abs/2405.09673)、[Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/））は、適切な設定のLoRAが、ほとんどのポストトレーニングタスク（指示への従順、ドメインへの適応、ペルソナ/スタイル）において、フルファインチューニングの品質に匹敵し、計算コストは67%で済むことを示しています。したがって、ほとんどのオペレーターが実際に求める作業においては、LoRAを使用し続けることで何も失うことはありません。`mode="full"`は、品質の差を測定し、追加の計算コストを費やすことを決定した場合に使用します。7B以上のモデルのフルファインチューニングを本当に必要とする場合は、HuggingFaceの`transformers.Trainer`を24GB以上のカードで直接使用してください。
- **オンラインRL — PPO / GRPO / RLVR** — Backpropagateは、単一ステージのSFTと参照なしの優先度調整（ORPOはv1.5で提供され、SimPO / KTOは計画中）を実行します。実行しないのは、PPO、GRPO、またはRLVRなどのオンライン強化学習です。これには、報酬モデルまたはトレーニングステップの上に生成とスコアリングのループが必要です。これらの場合は、TRLまたはLLaMA-Factoryを直接使用してください。（参照なしの優先度調整は、単一ステージの範囲に適合します。なぜなら、メモリに保持する必要のある個別の参照モデルがないからです。詳細は、[クイックスタート](#quick-start)のORPOの注記を参照してください。）
- **マルチノードトレーニング** — 単一のGPUを1つのマシンでのみ使用します。1つのマシンでのマルチGPUは機能しますが、公式にはサポートされていません（`accelerate launch`経由）。
- **CUDAレール上のmacOSトレーニング** — Apple SiliconにはCUDAがないため、CUDAパスは、NVIDIA GPUを搭載したLinuxまたはWindowsマシンで実行する必要があります。トレーニングされたモデルは、Ollamaを介してMacで引き続き実行できます。**v1.5の新機能：** 実験的なMLXレール（`--backend mlx`）は、Apple Silicon上でLoRAアダプターをネイティブにトレーニングします。詳細は、[Apple Silicon（MLX）](#apple-silicon-mlx--experimental-v15)を参照してください。これはLoRA-SFTのみであり、実際のシリコン上で構築および検証されていますが、まだ完全に検証されていません。したがって、LoRA SFT（ORPO、フルファインチューニング、FP8、マルチラン）以外のものについては、引き続きCUDAレールを使用することをお勧めします。
- **テストされたモデルファミリー以外のもの** — Qwen 2.5 / 3.5（7B / 4B）、Phi-4-mini-3.8B、SmolLM3-3B、Llama 3.2（3B / 1B）、Mistral 7B。他のモデルも多くの場合機能しますが、CIでは固定されていません。

これらの機能が必要な場合は、上記のライブラリのいずれかを使用してください。それらの機能により優れています。

## Backpropagateが提供するもの

1つのインストールで4つの機能：

**1. 3行の実際のAPIで、設定ファイルなしで実行できます。**
このREADMEの冒頭にあるスニペットは、最初から最後まで実行されます。`accelerate config`、YAML、Hydraオーバーライドは不要です。`Trainer(model).train(data)`を実行するだけで、ファインチューニングが完了します。

**2. 実際に動作するWindows。**
ほとんどのMLライブラリは、Windowsを後回しにします。Backpropagateは、Windows + RTX 5080で最初からテストされています。このライブラリは、ランタイムの癖を処理します。データの前処理方法を認識しているため、Windowsのマルチプロセッシングがクラッシュすることはありません。また、RTX 40/50カードで動作しなくなる場合は、xformersを自動的に無効にし、クラッシュしないデータローダーの設定を選択します。これらのことを知る必要はありません。単に実行するだけです。

**3. 無人実行用に構築。**
トレーニングには数時間かかります。監視する必要はありません。Backpropagateは、放置して実行できるように設計されています。

- GPUメモリが不足した場合、バッチサイズを自動的に半分にして、最大3回再試行します。手動で調整する必要はありません。
- GPUが過熱した場合、冷却されるまで一時停止し、その後再開します。
- すべてのチェックポイントはアトミックに書き込まれます。ラップトップが保存中にクラッシュした場合でも、以前の良好なチェックポイントはそのまま残ります。
- すべてのトレーニング実行には、一意のIDが割り当てられ、すべてのログ行、すべてのチェックポイント、およびすべてのWeights & Biasesエントリにスタンプが付けられます。問題が発生した場合、1つのIDを使用すると、メンテナーはすべてを関連付けることができます。
- エラーには、安定したコード（`RUNTIME_GPU_OOM`、`DEP_OLLAMA_REGISTRATION_FAILED`など）が付属しているため、ログを検索して、[トラブルシューティングガイド](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)で修正方法を見つけることができます。CUDA固有の障害には、専用の[CUDAトラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)があります。

**4. トレーニングされたアダプターから`ollama run`までのワンコマンド。**
多くのライブラリがモデルをトレーニングします。しかし、実際に使用したいときに、邪魔にならないようにするライブラリはほとんどありません。Backpropagateは、GGUF（Ollamaが使用する形式）にエクスポートし、1つのコマンドでOllamaモデルを登録します。トレーニングが完了してから、「チャットでファインチューニングモデルを使用できる」状態になるまで約30秒です。

## クイックスタート

このリポジトリには、小さなサンプルデータセットが含まれており、このREADMEの冒頭にあるスニペットは、クリーンなインストールで実行されます。

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

独自のデータの場合、JSONLファイルを1行に1つの例としてフォーマットします。

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca（`instruction` / `output`）、OpenAIチャット（`messages`）、および生のテキスト形式も機能します。Backpropagateは、形式を自動的に検出します。

### 嗜好性チューニング（ORPO、SimPO、KTO）

v1.5の新機能：プレーンなデモンストレーションではなく、優先度に基づいてトレーニングします。ORPOは参照なしで、単一ステージです。優先度のシグナルをSFTステップに組み込むため、個別の報酬モデルや参照モデルはなく、3行の構造は変わりません。`--method orpo`（CLI）または`method="orpo"`（Python）を渡し、`{prompt, chosen, rejected}`（または`{chosen, rejected}`のみ）の行のデータセットを渡します。

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

デフォルトの学習率は、ORPOに対して自動的に`8e-6`まで低下します（損失は単純なSFTよりも急峻です）。オッズ比ペナルティの重みを調整するために、`--orpo-beta`（デフォルトは`0.1`）を調整してください。ORPOは`mode="lora"`でのみ使用できます。

**v1.6の新機能 — SimPOとKTO。** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734))は、長さで正規化された報酬を使用し、参照を必要とせず、ORPOと同じペアの`{prompt, chosen, rejected}`データを受け取ります（`--simpo-beta`、`--simpo-gamma`）。`--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306))は、**ペアになっていない** `{prompt, completion, label}`データを受け取ります。これは、キュレーションされたA/Bペアではない大規模なフィードバックのクラスに対して、サンプルごとの肯定/否定の評価を行います。また、ラベル数に基づいて、望ましい/望ましくない損失の重みを自動的に調整します。どちらも`mode="lora"`でのみ使用でき、単一GPU SFT環境に留まります（個別の参照モデルは使用しません）。どの方法を使用するかについては、[嗜好性チューニングハンドブック](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/)を参照してください。オンラインRL（PPO/GRPO）については、[Backpropagateが適さない理由](#what-backpropagate-is-not-for)をご覧ください。

### 推論トレースSFT（R1蒸留）

v1.5の新機能：推論モデルを簡単に蒸留します。`--reasoning-trace`（CLI）または `Trainer(..., reasoning_trace=True)`（Python）を渡し、アシスタントの応答内に `<think>...</think>` の連鎖的な思考を保持するトレースを入力します。これは、[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 蒸留の純粋なSFT部分であり、RLは必要ありません。バックプロパゲーションは `<think>` をトレーニングターゲットに保持し、空の/長すぎるトレースを削除（トレース長フィルタリング）、およびより長いCoTのためにデフォルトの `max_seq_length` を8192に引き上げます。重要な点として、`<think>` は **プレーンテキスト** のままです。特別なトークンや、埋め込みのリサイズは行われません。そのため、マージされたGGUFは、他のファインチューンと同様にOllamaにエクスポートできます。SFTのみです。データセットの形状と調整可能なトークンバンドについては、[reasoning-trace recipe](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) を参照してください。

### Apple Silicon（MLX）—実験的、v1.5

v1.5の新機能：**1つのAPI、2つのレール。** CUDAは、引き続き標準で検証済みのバックエンドです。MLXは、Appleの [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) ツールチェーンを介して、MシリーズMacでトレーニングを行う2番目のレールです（統合メモリ、CUDAは不要）。同じ3行のコードで、ハードウェアによってレールを選択します。`backend='auto'`（デフォルト）は、NVIDIAではCUDAに、Apple SiliconではMLXにルーティングするため、既存のCUDA環境はバイト単位で同一です。

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

v1.5では、MLXレールは **LoRA SFTのみ** です。ORPO、FP8、`mode='full'`、MLXでのマルチランはまだサポートされていません（それぞれ `CONFIG_INVALID_SETTING` で拒否されます。それらの機能を使用する場合は、NVIDIA環境で `backend='cuda'` / `'auto'` を使用してください）。結果として得られるアダプターは、プレーンなsafetensorsであり、CUDAレールと同じパスを通じてOllamaにエクスポートされます。

> ⚠️ **現状:** v1.5でMLXレールは**構築され、ユニットテストも完了（モックを使用）**していますが、**まだ実際のApple Siliconでの実証検証は行われていません** — `mlx-lm`はApple専用であり、このコードが作成されたNVIDIA環境では実行できません。実験的なものとして扱ってください。これは、v1.5でFP8パスが採用されたときと同じ考え方です（FP8はv1.6でBlackwell上で実証検証に合格しました。MLXはまだ実際のシリコンでの検証が必要です）。MシリーズのMacで実行したら、[不具合を報告してください](#reporting-bugs)。Apple以外のホストで`--backend mlx`を強制すると、`CONFIG_INVALID_SETTING`エラーが発生します。Macに`mlx_lm`ツールチェーンがない場合、`DEP_MLX_UNAVAILABLE`エラーが発生します。

よりエンドツーエンドのワークフロー（ファインチューンとHF Hubへのプッシュ、OOM後の再開、長期間のキャンペーンにおけるマルチランSLAOなど）については、[ハンドブックのレシピページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/) を参照してください。

### Web UI（オプション）

Pythonのコードを入力する代わりにクリックしたい場合は、UIエクストラをインストールして起動します。

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

ローカルのWebインターフェイスが `http://localhost:7862` で開き、データセットを閲覧したり、形式を検証したり、トレーニング構成を視覚的に組み立てたりできます。トレーニング自体は `backprop train` を介して実行されます（UIによるトレーニングはロードマップにあります。現在の「開始」ボタンは、そのメモを表示します）。UIはデフォルトではローカルでのみ動作します。他のデバイスからアクセスできるようにするには、[Web UI](#web-ui) の `--share` + `--auth` セキュリティ契約を参照してください。

## マルチラントレーニング

複数のデータセットにわたって段階的にファインチューンしたい場合（たとえば、毎週新しいトレーニングデータを入手し、以前に学習したことを忘れないように追加したい場合）、Backpropagateの `multi_run` モードを使用します。

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

これは、5回のトレーニングパスを実行し、各パスの間にアダプターをマージすることで、以前の知識を保持しながら新しい例を組み込みます。この手法は、最近の継続学習の研究に基づいています。詳細については、このREADMEの末尾にある[参考文献](#references) を参照してください。

CLIバージョン：

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## チェックポイントからの再開

5回のトレーニングが4回目の実行でクラッシュした場合でも、再開できます。各マルチランセッションは、実行IDをディスク上の履歴とチェックポイントマニフェストに書き込みます。そのため、中断したところから再開するには、1つのコマンドを実行するだけです。

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

`backprop multi-run` のデフォルトの動作（`--resume` なし）は、同じ出力ディレクトリに進行中のエントリを自動的に検出し、続行します。クリーンな開始を強制するには、新しい出力ディレクトリを指定します。

## トレーニング履歴

すべての `backprop train` および `backprop multi-run` 呼び出しは、`<output>/run_history.json` に行を記録します。これには、使用されたモデル、データセット、ハイパーパラメータ、ステータス、最終的な損失、損失履歴が含まれます。過去の実行をリスト表示および検査できます。

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## 実験追跡

Backpropagateは、インストールされている実験追跡ツール（Weights & Biases、TensorBoard、MLflow）を自動的に検出し、それらを連携させます。`wandb` がインストールされていてログインしている場合、すべての実行は、ディスク上の実行IDと一致する実行名でW&Bに自動的にログ記録されます。これにより、W&B、ログ、および `run_history.json` を1つの識別子を使用して検索できます。

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

`Trainer(report_to=["wandb"])`、`Trainer(report_to=["tensorboard"])`、または `Trainer(report_to="none")` を使用してオーバーライドし、追跡を無効にすることができます。

## Web UI

Reflex Webインターフェイスは、オプションです。`pipx install "backpropagate[ui]"` でインストールし、起動します。

```bash
backprop ui --port 7862
```

UIはローカルの `http://localhost:7862` で実行されます。現在、ワークフローの **参照/検証/構成** の部分をカバーしています。データセットを指し示し、自動検出された形式と統計をチェックし、モデルを選択し、実行構成を組み立てます。**実行の開始はCLIから行われます** (`backprop train` / `backprop multi-run`)。UIの「開始」ボタンには、そのメモが表示されます。UIによるトレーニングは、今後の計画です。それまでは、UIはオンランプであり、CLIはトリガーです。

他のデバイス（ネットワーク上の他のユーザー、パブリックURLなど）に公開するには、`--share`（または`--host`）を`--auth`と組み合わせて使用する必要があります。

```bash
backprop ui --share --auth alice:hunter2
```

`--auth`なしで`backprop ui --share`を実行すると、エラーが発生します。理由は、`--share`はインターネット上の誰でもアクセスできるURLを公開し、認証がない場合、誰でもトレーニングのパイプラインを制御し、Hugging Faceのトークンを読み取ることができるためです。この設定を無効にするオプションはありません。認証情報を設定したくない場合は、代わりにSSHポートフォワーディングを使用してください。

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

完全な脅威モデルについては、[handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/)を参照してください。

UIからのファイルシステムへの書き込みは、単一のディレクトリにサンドボックス化されます。

- デフォルト：`~/.backpropagate/ui-outputs`
- 上書き：`BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`を設定します。
- 上書きは、許可リストで検証されます。システムまたは認証情報のパス（`/etc`、`~/.ssh`、`~/.aws`、`C:\Windows\System32`など）は拒否されます。

## プラットフォームに関する注意点

**要件：** Python 3.10+、CUDA GPU（8GB以上のVRAM）、PyTorch 2.0+

Python 3.10は少なくともv1.6までサポートされます。2026年10月にアップストリームのサポートが終了し、その後最初のリリースで削除される予定です。新しいインストールでは、Python 3.11または3.12を推奨します。3.11は最もテストされたバージョンです。

Backpropagateは、さまざまなプラットフォームでのトレーニングにおける実行時の癖に対処しますが、インストール時の問題を修正することはできません。最も一般的な問題は次の2つです。

- **誤ったCUDAホイール。** PyTorchは、CUDAバージョンごとに1つのバイナリとして公開されます。誤ったものを選択すると、CPUのみのPyTorchがサイレントにインストールされ、トレーニングは非常に遅くなります。ドライバーに適したホイールを<https://pytorch.org/get-started/locally/>で選択してください。`nvidia-smi`を実行して、ドライバー/CUDAバージョンを確認してください。
- **Windows + GGUFエクスポート。** `[export]`エクストラは、`llama-cpp-python`をソースからビルドします。これには、Visual Studio Build Tools（C++コンポーネント）とCMakeが必要です。

**macOS：** CUDAレールはサポートされていません（CUDAがありません）。CUDAを使用するように設定された`trainer.train()`は、`DEP_GPU_NOT_AVAILABLE`を発生させます。トレーニングされたアダプターは、Ollamaを介してMacで実行できます。**v1.5の新機能：** 実験的なMLXレール（`--backend mlx`、`pip install 'backpropagate[mlx]'`）は、`mlx_lm.lora`を介してApple Silicon上でLoRAアダプターをネイティブにトレーニングします。LoRA SFTのみで、ビルドおよびユニットテストは完了していますが、実際のハードウェアでの検証はまだ行われていません（[Apple Silicon (MLX)](#apple-silicon-mlx--experimental-v15)を参照）。CUDAパスを使用するか、ORPO / 完全なファインチューニング / FP8 / 複数回の実行を行う場合は、CUDA LinuxまたはWindowsマシンを使用してください。

詳細については、[トラブルシューティングハンドブックページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)にある詳細なインストール修正ガイドと、[CUDAトラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)にあるドライバー/VRAM/xformers/bf16対fp16の問題に関するページを参照してください。

## CLI

すべてのPython APIには、対応するCLIミラーがあります。

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

完全なリファレンスは、[CLIハンドブックページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/)または`backprop <サブコマンド> --help`で確認できます。

## 設定

すべての設定は、`BACKPROPAGATE_`プレフィックスを使用して環境変数で上書きできます。

| 変数 | デフォルト値 | 備考 |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | 自動 | JSONまたはコンソールログを強制的に出力します。 |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | デフォルトモデル |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | 学習率 |
| `BACKPROPAGATE_LORA__R` | `256` | LoRAランク（v1.3のデフォルト。v1.2.xのデフォルトの16にするには、`--lora-preset=fast`を渡します） |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | UIファイルシステムサンドボックス |

ネストされたキーには、二重アンダースコアを使用します（`MODEL__NAME`、`MODEL_NAME`ではありません）。完全なリファレンスは、[環境変数ハンドブックページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/)にあります。

## モデルプリセット

| プリセット | VRAM | ライセンス | 備考 |
|---|---|---|---|
| Qwen-3.5-4B | 約8GB | Apache 2.0 | 5GB以下のモデルに推奨されるデフォルト値。このサイズで最高の品質。 |
| Phi-4-mini-3.8B | 約8GB | MIT | 推論/数学/コードに優れています。ライセンスの制限が厳しい。 |
| SmolLM3-3B | 約6GB | Apache 2.0 | 完全にオープンなレシピ。ネイティブで64Kのコンテキストをサポート。 |
| Qwen 2.5 7B | 約12GB | Apache 2.0 | 既存のデフォルト値。従来の7Bプリセットの中で最高の品質。 |
| Qwen 2.5 3B | 約8GB | Qwen-Research | ⚠ 研究ライセンス — 商業利用前にQwenライセンス条項を確認してください。 |
| Llama 3.2 3B | 約8GB | Llama Community | Qwen 3Bの優れた代替手段で、許可に関する制限があります。 |
| Llama 3.2 1B | 約6GB | Llama Community | 小さなカードで迅速な実験を行うためのものです。 |
| Mistral 7B | 約12GB | Apache 2.0 | Qwen 7Bと同等で、異なるチャットテンプレートを使用します。 |

他のモデルも動作する場合がありますが、これらの8つだけがCIで固定されています。`--lora-preset=quality`（デフォルト）を渡すと、Biderman 2024 + Thinking Machines 2025のランク256 / すべての線形ターゲットが使用されます。v1.2.xのフットプリントが必要な場合は、`--lora-preset=fast`を渡すと、従来のランク16 / q+vターゲットが使用されます。

## トラブルシューティング

最も一般的な初回実行時のエラーの簡単なインデックスです。完全な逆インデックスは、[トラブルシューティングハンドブックページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/)にあります。ドライバー/VRAM/混合精度に関する詳細については、[CUDAトラブルシューティングページ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/)を参照してください。

| 症状 | エラーコード | 修正方法 |
|---|---|---|
| GPUのメモリがトレーニング中に不足する | `RUNTIME_GPU_OOM` | 自動 — バックプロパゲーションにより、バッチサイズが半分になり、最大3回再試行されます。無効にするには、`Trainer(oom_recovery=False)` を使用します。より小さいサイズに強制するには、`--batch-size 1` を使用します。 |
| HuggingFace から 401 / "モデルが見つかりません" というエラーが返されます。 | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` を実行し、再試行します。タイプミスがある場合は、<https://huggingface.co/models> から正確な ID をコピーしてください。 |
| `register_with_ollama` 接続が拒否されました。 | `DEP_OLLAMA_REGISTRATION_FAILED` | デーモンを開始します: `ollama serve`。 <https://ollama.com> からインストールします。再試行可能です。 |
| チェックポイント保存中にディスクがいっぱいになりました。 | `STATE_CHECKPOINT_INVALID` | アトミック書き込みにより、クラッシュ時に `.partial` ディレクトリが残ります。削除しても安全です。以前の正常なチェックポイントはそのままです。 |
| GPU の過熱によりトレーニングが一時停止しました。 | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | 自動 — バックプロパゲーションは、温度のしきい値で一時停止し、GPU が冷却されると再開します。頻繁に発生する場合は、エアフローを改善してください。 |
| `backprop ui --share` が拒否されました。 | `RUNTIME_UI_AUTH_NOT_ENFORCED` | `--auth user:password` を渡すか、代わりに SSH ポートフォワーディングを使用します (「[Web UI](#web-ui)」を参照)。 |
| GGUF エクスポートが最初の試行で失敗しました。 | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`。Windows の場合は、Visual C++ ビルドツールと CMake も必要です。 |

## バグの報告

何らかの理由で処理が失敗した場合、Backpropagate は起動時に `run_started run_id=<uuid>` のような行を出力し、同じ ID をすべてのログ行、すべてのチェックポイント、およびすべての Weights & Biases エントリに関連付けます。**バグ報告には `run_id` を含めてください**。これにより、担当者がその特定の実行に関連するすべての情報を関連付けることができます。

適切なバグ報告には、次のものが含まれます。

1. **`run_id`** — 起動時に出力される UUID。1 つの UUID により、担当者はその特定の実行に関連するすべてのログ行、すべてのチェックポイント、およびすべての Weights & Biases エントリを関連付けることができます。
2. **エラーコード** — `stderr` の `[CODE_NAME]: message` 行。安定したコードのカタログについては、[エラーコード](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) を参照してください。
3. **編集されたトレースバック**。非詳細モードでは、`stderr` が自動的に編集されます (Bearer トークン、`sk-*`、`hf_*`、AWS キー、`password=` / `token=` / `api_key=` ペアが削除されます)。貼り付けても安全です。完全な編集されていないトレースバックを取得するには、`BACKPROPAGATE_DEBUG=1` (または `--verbose`) で再実行し、投稿する前に確認してください。
4. **`backprop info` の出力**。1 つのコマンドで、Python / PyTorch / CUDA / GPU モデル / VRAM / OS / インストールされた追加機能が出力されます。これは、担当者がプラットフォーム固有の回帰を特定するために必要なすべての情報です。

[バグ報告テンプレート](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) には、これらすべてが明示的に記載されているため、トリアージが迅速に進みます。質問、アイデア、または「これは想定どおりですか？」というスレッドは、[GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions) に投稿してください。セキュリティの問題は、[GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) フォームを通じて非公開で報告してください。ポリシーと対応のタイムラインについては、[SECURITY.md](SECURITY.md) を参照してください。

## プライバシー

すべてのトレーニングは、ローカルの GPU 上で実行されます。Backpropagate は、HuggingFace からモデルをダウンロードする場合を除き、ネットワーク要求を行いません (これはユーザーが開始します)。テレメトリはなく、クラウドへの依存もありません。

## 参考文献

Backpropagate のデフォルト設定と複数回のトレーニングモードは、最近の研究に基づいています。関連する基盤技術に興味がある場合は、次の資料を参照してください。

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — Backpropagate がアダプターを効率的にトレーニングする方法である LoRA を導入した基礎論文。
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — ランク 256 ですべての線形ターゲットを使用した LoRA が、ほとんどのポストトレーニングタスクで 67% の計算量で完全なファインチューニングの品質に匹敵するという実証的な証拠。Backpropagate の v1.3 デフォルト LoRA 構成を推進します。
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — 高い LoRA ランクで必要な 10 倍の学習率と完全な FT の補正を特定した、実践的なフォローアップ。
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — ニューラルネットワークが新しいデータでファインチューニングすると、以前のトレーニングを「忘れてしまう」理由を最初に説明した論文 (EWC — 弾性重み集約)。
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA。これは、新しいアダプターを直交部分空間に制約することにより、LoRA を継続学習に使用する、より早いアプローチです。
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — 複数のファインチューニングされたモデルを干渉なしでマージするための基礎的な手法。
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — Backpropagate の複数回の実行マージャーが実装する特定のアルゴリズム。2025 年 12 月のプレプリント。Backpropagate は、この論文の最初の既知のダウンストリーム採用者です。

## ライセンス

MIT — [LICENSE](LICENSE) を参照してください。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
