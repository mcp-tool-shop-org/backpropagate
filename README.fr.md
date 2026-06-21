<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

# Ajustez finement un modèle QLoRA de 32 milliards de paramètres ou un modèle complet de 7 milliards de paramètres sur une seule carte graphique (GPU). Déployez-le sur Ollama

Effectuez une rétropropagation pour ajuster finement les grands modèles de langage sur une **seule** carte graphique, dimensionnée en fonction de la carte dont vous disposez réellement. Trois lignes de code Python pour un modèle QLoRA de 7 à 34 milliards de paramètres sur une seule carte grand public de 32 Go (RTX 5090) ; un seul indicateur — `--full-ft-offload` — permet d’effectuer un ajustement fin complet d’un modèle de classe 7B en déplaçant l’état de l’optimiseur vers la mémoire vive de l’hôte. Une commande supplémentaire exporte les données vers Ollama, puis `ollama run` lance votre modèle affiné. L’opération est facilement adaptable pour fonctionner avec une carte de 16 Go. Performances optimales sous Windows.

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

C'est tout. Il n'y a pas de fichier de configuration YAML. Il n'y a pas de procédure d'initialisation avec `accelerate launch`. Il n'y a pas de tutoriel séparé intitulé « Maintenant, convertissez-le au format GGUF ». Si vous avez une GPU CUDA et un fichier JSONL contenant vos données d'entraînement, il vous suffit de trois lignes pour obtenir un modèle affiné fonctionnel.

## Installation

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

Si vous souhaitez les fonctionnalités optionnelles, remplacez l'installation par l'une des suivantes :

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

Préférez-vous Docker ? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` fonctionne également. Des images sont disponibles pour `linux/amd64` et `linux/arm64`, de sorte que les utilisateurs d'Apple Silicon et d'ARM Linux bénéficient d'une image native. Un fichier `compose.yaml` standard pour « Interface utilisateur dans un conteneur » se trouve à la racine du dépôt ; `docker compose up` lance l'interface utilisateur web sur `http://localhost:7860` avec un montage de volume persistant `~/.backpropagate`.

## La place de Backpropagate dans l'écosystème

Il existe plusieurs bonnes bibliothèques pour affiner les LLM. Elles sont toutes excellentes dans différents domaines :

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** : si vous aimez les configurations YAML et que vous souhaitez disposer d'une communauté de recettes à partir desquelles vous pouvez vous inspirer.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** : si vous souhaitez utiliser DPO/PPO/RLHF et une interface graphique web.
- **[Unsloth](https://github.com/unslothai/unsloth)** : si vous avez besoin de l'entraînement le plus rapide possible et que vous utilisez une famille de modèles prise en charge.
- **[torchtune](https://github.com/pytorch/torchtune)** : si vous souhaitez utiliser les recettes PyTorch natives de Meta que vous pouvez modifier.

Backpropagate est l'option manquante : une API Python en 3 lignes pour les utilisateurs individuels disposant d'une seule GPU grand public qui souhaitent entraîner un adaptateur et l'envoyer. Pas de YAML, pas d'interface graphique, pas d'apprentissage par renforcement en ligne (PPO/GRPO), pas de multi-nœuds. Juste la boucle dont tout le monde a réellement besoin et l'étape d'exportation qui pose problème.

Si vous avez essayé l'une des bibliothèques ci-dessus et que vous avez été rebuté par la complexité de la configuration, ou que vous avez rencontré un problème de compatibilité avec une famille de modèles, ou que vous souhaitiez des paramètres par défaut optimisés pour Windows, Backpropagate est fait pour vous.

## Ce que vous pouvez ajuster finement sur une seule carte graphique

La rétropropagation adapte l’exécution à votre carte. Voici les limites pratiques sur une carte graphique grand public de **32 Go** (RTX 5090) avec 64 Go de mémoire vive : c’est la configuration sur laquelle elle est optimisée :

| Taille du modèle | Méthode | État sur une carte de 32 Go |
|---|---|---|
| 7B (Qwen 2.5 7B / Llama-3.1-8B / Mistral 7B) | QLoRA | Facile — environ 7 à 8 Go. Longueur de séquence complète, beaucoup de marge. |
| **14B** (Qwen2.5-14B) | QLoRA | **La configuration idéale pour une utilisation quotidienne — environ 8,5 Go**, mesuré. Rang/alpha 32, paged 8-bit AdamW, 4096 ctx. |
| 24B (Mistral-Small-24B) | QLoRA | Environ 18 Go. S’adapte avec une marge à 4096 ctx. |
| **32B** (Qwen2.5-32B) | QLoRA | **S’adapte tout juste — environ 26 Go** avec `max_len 2048` + paged 8-bit AdamW. Limite supérieure. |
| ≤6B | `mode="full"` (affinement complet) | Ajustement fin complet sur GPU uniquement — poids bf16, pas d’adaptateur. La limite tenant compte de la carte est de 6 milliards de paramètres sur une carte de 32 Go. |
| **Modèle de classe 7B** (Qwen 2.5 7B / Llama-3.1-8B / Mistral 7B) | `mode="full" --full-ft-offload` | **Ajustement fin complet via FSDP2 avec déchargement sur le CPU** — les paramètres et l’optimiseur sont transférés vers la mémoire vive de l’hôte (64 Go). Plus lent (limité par la bande passante) ; Linux/WSL2. |

Deux opérations pour lesquelles la plupart des bibliothèques à carte graphique unique vous renvoient vers d’autres solutions — **QLoRA de 24 à 34 milliards de paramètres** et **ajustement fin complet sur une seule carte de modèle de classe 7B** — Backpropagate effectue ces opérations sur une seule carte grand public, puis exporte le résultat directement vers Ollama.

**La limite pour l’ajustement fin complet tient compte de la carte.** Elle est dérivée du calcul de la mémoire d’entraînement à quatre termes (poids + gradients + optimiseur + activations) par rapport à votre VRAM *détectée* : **16 Go → 4 milliards, 24 Go → 5 milliards, 32 Go → 6 milliards** sur GPU uniquement. `--full-ft-offload` augmente cette limite à **modèle de classe 7B** en déplaçant les paramètres et l’état de l’optimiseur vers la mémoire vive via FSDP2 `fully_shard` + `CPUOffloadPolicy` (plus lent, limité par la bande passante PCIe/CPU ; nécessite environ 64 Go de mémoire vive et une implémentation NCCL, c’est-à-dire Linux/WSL2). Définissez explicitement la limite avec `--full-ft-ceiling-billions`. Un modèle dépassant même la limite du déchargement se termine par `RUNTIME_FULL_FT_MODEL_TOO_LARGE`, en indiquant la solution ( `--full-ft-offload` ou LoRA/QLoRA). Consultez [la page complète sur l’ajustement fin](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/) pour les calculs de la VRAM et la comparaison de qualité Biderman 2024 / Thinking Machines 2025.

### S’adapte à une carte de 16 Go

La limite d’une carte de 16 Go (RTX 4080 / 5080 / 4070 Ti Super) reste optimale : QLoRA de 7B à environ 7–8 Go, et véritable ajustement fin complet d’un modèle authentique d’environ 3 milliards de paramètres (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B/1B) dans les 16 Go via `mode="full"` (poids bf16 + contrôle des gradients + paged 8-bit AdamW). Le même code sélectionne la taille du lot et la limite de l’ajustement fin complet qui s’adaptent à la carte détectée, sans indicateurs à modifier entre les configurations.

La quantification sur 2 bits (AQLM / QuIP#) est **hors de portée** — une base quantifiée sur 2 bits ne peut pas être fusionnée proprement avec des poids en pleine précision, ce qui rompt le contrat d’adaptateur fusionnable → GGUF → Ollama (le but principal du processus). Les leviers que Backpropagate propose à la place — QLoRA, `mode="full"`, `--full-ft-offload` et le chemin de calcul FP8 (`--fp8`, Blackwell/Hopper) — restent fusionnables et exportables.

## Ce que Backpropagate NE permet PAS

Si votre cas d'utilisation est l'un des suivants, vous obtiendrez de meilleurs résultats avec une autre bibliothèque : Backpropagate n'est pas le bon choix et essayer de le faire fonctionner coûterait plus cher que de simplement utiliser l'outil approprié. La lecture de cette section avant de commencer vous évitera de devoir installer et abandonner :

- **Affinement complet des paramètres au-delà du seuil de déchargement (≈13B+)** — Effectuer une rétropropagation d’un affinement complet jusqu’à **~6 Go de GPU pur et ~7 Go via `--full-ft-offload`** sur une carte de 32 Go (voir [l’enveloppe](#what-you-can-fine-tune-on-one-gpu)). Un affinement *véritablement complet* d’un modèle de 13B+ dépasse cette limite : il nécessite un FSDP multi-GPU ou une carte plus grande (utiliser `transformers.Trainer` sur plusieurs GPU, ou louer une A100/H100). Avant de dépenser ces ressources de calcul, cependant : des recherches récentes ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) montrent que LoRA, avec une configuration correcte, offre une qualité d’affinement comparable à celle de l’affinement complet pour la plupart des tâches post-entraînement (suivi des instructions, adaptation au domaine, personnalité/style) en utilisant environ 67 % des ressources de calcul. Ainsi, QLoRA jusqu’à 34B, ce que Backpropagate effectue sur une seule carte, ne présente aucun inconvénient pour les tâches que la plupart des utilisateurs souhaitent effectuer.
- **Apprentissage par renforcement en ligne — PPO / GRPO / RLVR** — Backpropagate effectue un affinement SFT en une seule étape, ainsi qu’un ajustement de préférence sans référence (ORPO dans v1.5 ; SimPO + KTO dans v1.6). Ce que cela ne fait *pas*, c’est l’apprentissage par renforcement en ligne — PPO, GRPO ou RLVR —, qui nécessite un modèle de récompense ou une boucle de génération et d’évaluation en plus de l’étape d’entraînement. Pour ces cas, utilisez TRL directement ou LLaMA-Factory. (L’ajustement de préférence sans référence correspond à l’enveloppe d’une seule étape car il n’y a pas de modèle de référence distinct à conserver en mémoire ; voir la note sur ORPO dans [Quick Start](#quick-start).)
- **Entraînement multi-nœuds** — un seul GPU sur une seule machine. L’utilisation de plusieurs GPU sur une seule machine est possible (via `accelerate launch`), mais n’est pas officiellement prise en charge.
- **Entraînement macOS avec CUDA** — Apple Silicon ne dispose pas de CUDA, donc le chemin CUDA s’exécute sur une machine Linux ou Windows dotée d’un GPU NVIDIA. Vous pouvez toujours exécuter le modèle entraîné sur un Mac via Ollama. Un rail MLX **expérimental et non vérifié** (`--backend mlx`) entraîne un adaptateur LoRA de manière native sur Apple Silicon — voir [Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview). Il n’effectue qu’un affinement SFT avec LoRA et **n’a pas été vérifié en conditions réelles** (pas de prise en charge), donc pour tout ce qui dépasse un affinement SFT avec LoRA (ORPO, affinement complet, FP8, exécution multiple), vous devez utiliser le rail CUDA.
- **Tout modèle autre que ceux testés** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. D’autres modèles fonctionnent souvent, mais ne sont pas inclus dans les tests CI.

Si vous avez besoin de l’une de ces fonctionnalités, utilisez l’une des bibliothèques répertoriées ci-dessus. Elles sont plus performantes dans ce domaine.

## Ce que Backpropagate vous offre

Quatre éléments, dans une seule installation :

**1. Une véritable API en 3 lignes qui fonctionne sans fichier de configuration.**
L’extrait en haut de ce fichier README s’exécute de bout en bout. Pas de `accelerate config`, pas de YAML, pas de substitutions Hydra. Il suffit de `Trainer(model).train(data)` et vous avez un modèle affiné.

**2. Windows, qui fonctionne réellement.**
La plupart des bibliothèques d’apprentissage automatique considèrent Windows comme une réflexion après coup. Backpropagate est testé en priorité sur Windows + RTX 5080. La bibliothèque gère les particularités de l’exécution pour vous : elle sait comment pré-tokeniser vos données afin que le traitement parallèle de Windows ne plante pas, elle désactive automatiquement xformers sur les cartes RTX 40/50 où cela causerait des problèmes, et elle sélectionne les paramètres du chargeur de données qui ne provoquent pas d’erreurs. Vous n’avez pas besoin d’en connaître les détails. Il fonctionne simplement.

**3. Conçu pour les exécutions sans surveillance.**
L’entraînement prend des heures. Vous ne voulez pas le surveiller constamment. Backpropagate est conçu pour être laissé en fonctionnement :

- Si vous manquez de mémoire GPU, il réduit automatiquement de moitié la taille du lot et réessaie — jusqu’à trois fois. Pas de réglage manuel.
- Si votre GPU devient trop chaud, il fait une pause jusqu’à ce que les choses se refroidissent, puis reprend.
- Chaque point de contrôle est écrit de manière atomique — si votre ordinateur portable plante au milieu de l’enregistrement, le point de contrôle précédent et valide reste intact.
- Chaque exécution d’entraînement reçoit un ID unique qui est estampillé sur chaque ligne de journal, chaque point de contrôle et chaque entrée Weights & Biases. Si quelque chose ne va pas, un seul ID permet à un mainteneur de corréler tous les éléments.
- Les erreurs sont accompagnées de codes stables (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, etc.) afin que vous puissiez rechercher dans vos journaux et dans le [guide de dépannage](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) pour trouver la solution. Les erreurs spécifiques à CUDA ont une page de dépannage [CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) dédiée.

**4. Une seule commande, de l’adaptateur entraîné à `ollama run`.**
De nombreuses bibliothèques entraînent un modèle. Peu d’entre elles vous facilitent la tâche lorsque vous souhaitez réellement l’utiliser. Backpropagate exporte au format GGUF (le format utilisé par Ollama) et enregistre un modèle Ollama en une seule commande. Vous passez de « entraînement terminé » à « je peux discuter avec mon modèle affiné » en environ 30 secondes.

## Démarrage rapide

Le dépôt contient un petit ensemble de données d’exemple afin que l’extrait en haut de ce fichier README fonctionne sur une installation propre :

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

Cela entraîne un adaptateur Qwen 2.5 7B sur 5 courtes conversations au format ShareGPT, puis exporte le résultat au format GGUF. Pour vos propres données, formatez votre fichier JSONL avec un exemple par ligne :

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Les formats Alpaca (`instruction` / `output`), OpenAI chat (`messages`) et texte brut fonctionnent également — Backpropagate détecte automatiquement le format.

### Ajustement de préférence (ORPO, SimPO, KTO)

Nouveau dans la version 1.5 : entraînez-vous sur des préférences au lieu de simples démonstrations. ORPO est sans référence et en une seule étape — il intègre le signal de préférence dans l’étape d’affinage supervisé, il n’y a donc pas de modèle de récompense ou de référence distinct et la forme en 3 lignes reste la même. Passez `--method orpo` (CLI) ou `method="orpo"` (Python) et fournissez un ensemble de données de lignes `{prompt, chosen, rejected}` (ou simplement `{chosen, rejected}` :

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

Le taux d’apprentissage par défaut diminue automatiquement à `8e-6` pour ORPO (la perte est plus marquée que pour un simple SFT) ; ajustez `--orpo-beta` (par défaut `0.1`) pour pondérer la pénalité du rapport de cotes. ORPO utilise uniquement le mode `"lora"`.

**Nouveauté dans v1.6 — SimPO et KTO.** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) est sans référence, avec une récompense normalisée en fonction de la longueur, et utilise les mêmes données appariées `{prompt, chosen, rejected}` que ORPO (`--simpo-beta`, `--simpo-gamma`). `--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) utilise des données **non appariées** `{prompt, completion, label}` — évaluations positives/négatives par exemple — pour la vaste classe de commentaires qui ne sont pas des paires A/B organisées ; il équilibre automatiquement les pondérations de perte souhaitables/indésirables en fonction du nombre d’étiquettes. Les deux utilisent uniquement le mode `"lora"` et restent dans l’enveloppe SFT avec un seul GPU (pas de modèle de référence distinct). Consultez le [manuel sur l’ajustement de préférence](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) pour savoir lequel utiliser. Pour l’apprentissage par renforcement en ligne (PPO/GRPO), consultez [Ce que Backpropagate ne permet PAS](#what-backpropagate-is-not-for).

### SFT avec suivi du raisonnement (distillation R1)

Nouveau dans la version 1.5 : distillez un modèle de raisonnement de manière simple. Utilisez `--reasoning-trace` (CLI) ou `Trainer(..., reasoning_trace=True)` (Python) et fournissez-lui des données qui conservent une chaîne de pensée `<think>...</think>` dans les interactions de l’assistant — la moitié SFT pure de la distillation de [DeepSeek-R1](https://arxiv.org/abs/2501.12948), sans apprentissage par renforcement requis. La rétropropagation conserve `<think>` dans l’objectif d’entraînement, supprime les données vides ou trop longues (filtrage de la longueur des données) et augmente la valeur par défaut de `max_seq_length` à 8192 pour la chaîne de pensée plus longue. Il est essentiel que `<think>` reste du **texte brut** — aucun jeton spécial, aucun redimensionnement de l’intégration — de sorte que le GGUF fusionné puisse toujours être exporté vers Ollama comme tout autre modèle affiné. Uniquement SFT. Consultez la [recette reasoning-trace](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) pour la forme de l’ensemble de données et les jetons ajustables.

### Apple Silicon (MLX) — aperçu non vérifié

> ⚠️ **Aperçu non vérifié — ne fait pas partie de l’ensemble des fonctionnalités prises en charge.** Le rail MLX est construit et testé avec des tests unitaires, mais n’a **pas** été vérifié en conditions réelles sur Apple Silicon (mlx-lm est spécifique à Apple et ne peut pas s’exécuter sur les machines NVIDIA sur lesquelles Backpropagate est développé). Considérez tout ce qui suit comme expérimental, utilisez-le à vos propres risques et [signalez les anomalies](#reporting-bugs) si vous l’exécutez sur un Mac de la série M.

Nouveau dans la version 1.5 : **une API, deux options.** CUDA reste le backend canonique et vérifié ; MLX est une deuxième option qui entraîne sur un Mac de la série M via la chaîne d’outils [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) d’Apple (mémoire unifiée, pas de CUDA). La même structure en 3 lignes sélectionne l’option en fonction du matériel — `backend='auto'` (par défaut) redirige vers CUDA sur NVIDIA et vers MLX sur Apple Silicon, de sorte que les configurations CUDA existantes sont identiques au niveau des octets.

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

Dans la version 1.5, l’option MLX est **uniquement LoRA SFT** — pas d’ORPO, pas de FP8, pas de `mode='full'`, pas d’exécution multiple sur MLX pour le moment (chacune est rejetée avec `CONFIG_INVALID_SETTING` ; utilisez `backend='cuda'`/`'auto'` sur une machine NVIDIA pour ces options). L’adaptateur résultant est un simple fichier safetensors et est exporté vers Ollama via le même chemin que l’option CUDA.

> Forcer `--backend mlx` sur un hôte non Apple génère une erreur `CONFIG_INVALID_SETTING` ; l’absence d’une chaîne d’outils `mlx_lm` sur un Mac déclenche `DEP_MLX_UNAVAILABLE`.

Pour des flux de travail de bout en bout plus complets (affiner et pousser vers HF-Hub, reprendre après une erreur de mémoire, SLAO multi-exécution sur une longue campagne, etc.), consultez la [page des recettes du manuel](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interface utilisateur Web (facultative)

Si vous préférez cliquer plutôt que taper en Python, installez l’extra de l’interface utilisateur et lancez :

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Une interface Web locale s’ouvre à l’adresse `http://localhost:7862` pour parcourir les ensembles de données, valider les formats et assembler une configuration d’entraînement visuellement. L’entraînement lui-même s’exécute via `backprop train` (l’entraînement piloté par l’interface utilisateur est prévu ; le bouton Démarrer affiche actuellement cette note). L’interface utilisateur est locale par défaut. Pour la rendre accessible à d’autres appareils, consultez la section [Interface utilisateur Web](#web-ui) ci-dessous pour le contrat de sécurité `--share` + `--auth`.

## Entraînement multi-exécution

Si vous souhaitez affiner de manière incrémentale sur plusieurs ensembles de données — par exemple, si vous recevez de nouvelles données d’entraînement chaque semaine et que vous souhaitez les ajouter sans oublier ce que vous avez appris auparavant — le mode `multi_run` de Backpropagate est fait pour vous :

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

Cela exécute cinq passes d’entraînement, en fusionnant l’adaptateur entre les exécutions de manière à préserver les connaissances antérieures tout en intégrant de nouveaux exemples. La technique est basée sur des recherches récentes sur l’apprentissage continu — voir [Références](#references) en bas de ce fichier README.

La version CLI :

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Reprise à partir d’un point de contrôle

Un entraînement de 5 exécutions qui plante à la 4e exécution peut être récupéré. Chaque session multi-exécution écrit son ID d’exécution dans l’historique et le manifeste des points de contrôle sur disque, de sorte que la reprise là où vous vous êtes arrêté ne nécessite qu’une seule commande :

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

Le comportement par défaut de `backprop multi-run` (sans `--resume`) détecte automatiquement une entrée en cours dans le même répertoire de sortie et la poursuit. Pour forcer un nouveau démarrage, pointez vers un nouveau répertoire de sortie.

## Historique de l’entraînement

Chaque invocation de `backprop train` et `backprop multi-run` enregistre une ligne dans `<output>/run_history.json` — modèle utilisé, ensemble de données, hyperparamètres, statut, perte finale, historique des pertes. Vous pouvez lister et examiner les exécutions passées :

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Suivi des expériences

Backpropagate détecte automatiquement les outils de suivi des expériences installés (Weights & Biases, TensorBoard, MLflow) et les configure. Si `wandb` est installé et que vous êtes connecté, chaque exécution enregistre automatiquement les données dans W&B avec un nom d’exécution qui correspond à l’ID d’exécution sur disque — de sorte que vous pouvez effectuer une recherche dans W&B, vos journaux et `run_history.json` en utilisant un seul identifiant.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Remplacez par `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` ou `Trainer(report_to="none")` pour vous désinscrire.

## Interface utilisateur Web

L’interface Web Reflex est une option — installez-la avec `pipx install "backpropagate[ui]"` et lancez :

```bash
backprop ui --port 7862
```

L’interface utilisateur s’exécute localement à l’adresse `http://localhost:7862`. Aujourd’hui, elle couvre la moitié du flux de travail qui consiste à **parcourir / valider / configurer** — pointez-la vers un ensemble de données, vérifiez le format et les statistiques détectés automatiquement, choisissez un modèle et assemblez une configuration d’exécution. **Le lancement de l’exécution se fait à partir de la ligne de commande** (`backprop train` / `backprop multi-run`) ; le bouton Démarrer dans l’interface utilisateur affiche une note qui y fait référence. L’entraînement piloté par l’interface utilisateur est une prochaine étape prévue — jusqu’alors, l’interface utilisateur est le point d’entrée et la ligne de commande est le déclencheur.

Pour exposer l’application à d’autres appareils (d’autres personnes sur votre réseau, une URL publique, etc.), vous devez associer l’option `--share` (ou `--host`) à l’option `--auth` :

```bash
backprop ui --share --auth alice:hunter2
```

L’exécution de `backprop ui --share` sans l’option `--auth` entraîne une erreur. La raison : l’option `--share` publie une URL accessible à toute personne sur Internet, et sans authentification, cela signifie que n’importe qui peut exécuter votre pipeline d’entraînement et lire votre jeton Hugging Face. Il n’est pas possible de désactiver cette fonctionnalité. Si vous ne souhaitez pas définir d’informations d’identification, utilisez plutôt le transfert de port SSH :

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

Consultez la page [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) pour obtenir le modèle complet des menaces.

Les écritures dans le système de fichiers à partir de l’interface utilisateur sont isolées dans un seul répertoire :

- Par défaut : `~/.backpropagate/ui-outputs`
- Remplacement : définissez `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- Le remplacement est validé à l’aide d’une liste de blocage : les chemins système ou d’identification (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) sont refusés.

## Notes sur la plateforme

**Configuration requise :** Python 3.10+ ; GPU CUDA (8 Go ou plus de VRAM) ; PyTorch 2.0+

Python 3.10 est pris en charge jusqu’à au moins la version 1.6 ; il atteindra sa fin de vie en octobre 2026 et sera supprimé dans la première version après cette date. Pour les nouvelles installations, préférez Python 3.11 ou 3.12 — 3.11 est la version la plus testée.

Backpropagate gère les particularités de l’exécution de l’entraînement sur différentes plateformes, mais il ne peut pas résoudre les problèmes survenant lors de l’installation. Les deux problèmes les plus courants sont les suivants :

- **Mauvaise version de CUDA.** PyTorch est publié avec un seul fichier binaire par version de CUDA. Si vous choisissez la mauvaise version, vous obtiendrez silencieusement PyTorch en mode CPU uniquement et l’entraînement sera extrêmement lent. Utilisez le sélecteur de fichiers à l’adresse <https://pytorch.org/get-started/locally/> pour votre pilote. Exécutez `nvidia-smi` pour afficher votre version de pilote/CUDA.
- **Windows + exportation GGUF.** L’option `[export]` crée `llama-cpp-python` à partir du code source, ce qui nécessite les outils de création de Visual Studio (composant C++) et CMake.

**macOS :** la prise en charge de CUDA n’est pas disponible (pas de CUDA) ; l’exécution de `trainer.train()` avec une configuration CUDA génère l’erreur `DEP_GPU_NOT_AVAILABLE`, et vous pouvez exécuter l’adaptateur entraîné sur un Mac via Ollama. **Nouveau dans la version 1.5 :** une configuration MLX expérimentale (`--backend mlx`, `pip install 'backpropagate[mlx]'`) entraîne un adaptateur LoRA de manière native sur Apple Silicon via `mlx_lm.lora` ; uniquement pour LoRA SFT, et créé et testé, mais pas encore vérifié sur du matériel réel (voir [Apple Silicon (MLX)](#apple-silicon-mlx--experimental-v15)). Pour le chemin CUDA, ou pour ORPO / affinage complet / FP8 / exécution multiple, utilisez une machine Linux ou Windows avec CUDA.

Consultez la page [du manuel de résolution des problèmes](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) pour obtenir un guide complet de résolution des problèmes d’installation, et la page [dédiée à la résolution des problèmes de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) pour les problèmes de pilote / VRAM / xformers / bf16 par rapport à fp16.

## CLI

Chaque API Python a un équivalent CLI :

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

Référence complète à la page [du manuel CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), ou `backprop <sous-commande> --help`.

## Configuration

Chaque paramètre peut être remplacé par une variable d’environnement en utilisant le préfixe `BACKPROPAGATE_` :

| Variable | Valeur par défaut | Notes |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Force l’utilisation de journaux JSON ou de la console |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modèle par défaut |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Taux d’apprentissage |
| `BACKPROPAGATE_LORA__R` | `256` | Rang LoRA (valeur par défaut de la version 1.3 ; utilisez l’option `--lora-preset=fast` pour la valeur par défaut de la version 1.2.x, qui est de 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Isolation du système de fichiers de l’interface utilisateur |

Les clés imbriquées utilisent un double soulignement (`MODEL__NAME`, et non `MODEL_NAME`). La référence complète se trouve à la page [du manuel des variables d’environnement](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Prédéfinis de modèle

| Prédéfinis | VRAM | Licence | Notes |
|---|---|---|---|
| Qwen-3.5-4B | ~8 Go | Apache 2.0 | Valeur par défaut recommandée pour les modèles de moins de 5 milliards de paramètres. Meilleure qualité pour cette taille. |
| Phi-4-mini-3.8B | ~8 Go | MIT | Bon en raisonnement / mathématiques / code. Licence strictement propre. |
| SmolLM3-3B | ~6 Go | Apache 2.0 | Recette entièrement ouverte. Contexte natif de 64 Ko. |
| Qwen 2.5 7B | ~12 Go | Apache 2.0 | Valeur par défaut existante. Meilleure qualité des prédéfinis de 7 milliards de paramètres. |
| Qwen 2.5 3B | ~8 Go | Qwen-Research | ⚠ Licence de recherche : consultez les conditions de licence de Qwen avant une utilisation commerciale. |
| Llama 3.2 3B | ~8 Go | Llama Community | Alternative solide à Qwen 3B avec des réserves permissives. |
| Llama 3.2 1B | ~6 Go | Llama Community | Pour des expériences rapides sur des cartes de petite taille. |
| Mistral 7B | ~12 Go | Apache 2.0 | Comparable à Qwen 7B, modèle de chat différent. |
| Llama-3.1-8B | ~7 à 8 Go (QLoRA) | Llama-3.1-Community | 8B QLoRA, contexte natif de 128 k (la clause des >700 millions d’utilisateurs actifs mensuels nécessite une licence Meta distincte). |
| **Qwen2.5-14B** | ~8,5 Go (QLoRA) | Apache 2.0 | **Le point idéal pour un usage quotidien avec 32 Go** — rang/alpha 32, AdamW à 8 bits paginé, 4096 ctx. |
| Mistral-Small-24B | ~18 Go (QLoRA) | Apache 2.0 | 24B QLoRA sur une carte de 32 Go avec une marge de 4096 ctx. |
| **Qwen2.5-32B** | ~26 Go (QLoRA) | Apache 2.0 | **Au sommet de l’enveloppe des 32 Go** — s’adapte tout juste à `max_len 2048` + AdamW à 8 bits paginé. |

D’autres modèles fonctionnent souvent ; les lignes ci-dessus sont les préréglages organisés — la couche de 14B à 32B est affinée avec QLoRA pour une carte de 32 Go (l’enveloppe mesurée). Utilisez `--lora-preset=quality` (par défaut) pour les cibles de rang 256 / toutes linéaires, comme indiqué dans Biderman 2024 + Thinking Machines 2025, ou `--lora-preset=fast` pour la cible héritée de rang 16 / q+v si vous avez besoin de l’empreinte de v1.2.x.

## Résolution des problèmes

Un bref index des échecs les plus courants lors de la première exécution. L’index inverse complet se trouve à la page [du manuel de résolution des problèmes](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Pour une analyse approfondie du pilote / VRAM / précision mixte, consultez la page [de résolution des problèmes de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Symptôme | Code d’erreur | Solution |
|---|---|---|
| La mémoire du GPU est épuisée au milieu de l’entraînement. | `RUNTIME_GPU_OOM` | Automatique — La rétropropagation divise par deux la taille du lot et réessaie jusqu'à 3 fois. Pour désactiver : `Trainer(oom_recovery=False)`. Pour forcer une taille plus petite : `--batch-size 1`. |
| HuggingFace renvoie 401 / « modèle introuvable ». | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` et réessayer. En cas de fautes de frappe, copiez l’ID exact depuis <https://huggingface.co/models>. |
| `register_with_ollama` : connexion refusée. | `DEP_OLLAMA_REGISTRATION_FAILED` | Démarrez le démon : `ollama serve`. Installez à partir de <https://ollama.com>. Peut être réessayé. |
| Espace disque insuffisant lors de la sauvegarde du point de contrôle. | `STATE_CHECKPOINT_INVALID` | Les écritures atomiques laissent un répertoire `.partial` en cas de plantage — il est sûr de le supprimer. Le point de contrôle précédent et valide est intact. |
| Entraînement interrompu en raison d’une surchauffe du GPU. | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automatique — La rétropropagation s’interrompt lorsque la température atteint le seuil et reprend lorsque le GPU refroidit. Améliorez le flux d’air si cela se reproduit. |
| `backprop ui --share` rejeté. | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Passez `--auth user:password`, ou utilisez le transfert de port SSH à la place (voir [Interface utilisateur Web](#interface-utilisateur-web)). |
| L’exportation GGUF a échoué lors de la première tentative. | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]` ; sous Windows, vous avez également besoin des outils de création Visual C++ et de CMake. |

## Signaler des bogues

Lorsqu’une opération échoue, Backpropagate affiche une ligne au démarrage, comme `run_started run_id=<uuid>`, et associe le même ID à chaque ligne de journal, à chaque point de contrôle et à chaque entrée Weights & Biases. **Incluez le `run_id` dans tout signalement de bogue** — cela permet à la personne chargée de la maintenance de corréler tous les éléments de cette exécution spécifique.

Un bon signalement de bogue comprend :

1. **Le `run_id`** — l’UUID affiché au démarrage. Un seul UUID permet à la personne chargée de la maintenance de corréler chaque ligne de journal, chaque point de contrôle et chaque entrée Weights & Biases pour cette exécution spécifique.
2. **Le code d’erreur** — la ligne `[CODE_NAME]: message` dans stderr. Consultez [les codes d’erreur](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) pour obtenir le catalogue des codes stables.
3. **La trace d’exécution expurgée.** Stderr est automatiquement expurgé en mode non verbeux (les jetons Bearer, `sk-*`, `hf_*`, les clés AWS, les paires `password=` / `token=` / `api_key=` sont supprimés) — il est sûr de la copier-coller. Pour la trace d’exécution complète et non expurgée, réexécutez avec `BACKPROPAGATE_DEBUG=1` (ou `--verbose`) ; examinez-la avant de la publier.
4. **La sortie de `backprop info`.** Une seule commande affiche Python / PyTorch / CUDA / modèle GPU / VRAM / OS / extras installés — tout ce dont la personne chargée de la maintenance a besoin pour identifier une régression spécifique à une plateforme.

Le [modèle de signalement de bogue](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) demande explicitement chacun de ces éléments afin d’accélérer le processus de triage. Les questions, les idées ou les messages du type « est-ce normal ? » doivent être publiés dans [les discussions GitHub](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Les problèmes de sécurité doivent être signalés en privé via le [formulaire de signalement de sécurité GitHub](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) — consultez [SECURITY.md](SECURITY.md) pour connaître la politique et les délais de réponse.

## Confidentialité

Tout l’entraînement se déroule localement sur votre GPU. Backpropagate n’effectue aucune requête réseau, sauf pour télécharger des modèles depuis HuggingFace (ce que vous initiez). Pas de télémétrie, pas de dépendance au cloud.

## Références

Les valeurs par défaut de Backpropagate et le mode d’entraînement multi-exécution sont basés sur des recherches récentes. Si vous êtes intéressé par les techniques sous-jacentes :

- **Hu et al. 2021.** *LoRA : adaptation de faible rang des grands modèles de langage.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — l’article fondateur qui présente LoRA, qui est la méthode utilisée par Backpropagate pour entraîner efficacement les adaptateurs.
- **Biderman et al. 2024.** *LoRA apprend moins et oublie moins.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — preuves empiriques que LoRA avec un rang de 256 et des cibles entièrement linéaires correspond à la qualité de l’ajustement complet sur la plupart des tâches post-entraînement, pour 67 % de la puissance de calcul. Cela détermine la configuration LoRA par défaut de Backpropagate v1.3.
- **Thinking Machines 2025.** *LoRA sans regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — le suivi pratique qui identifie la correction de 10 fois du taux d’apprentissage par rapport à l’ajustement complet nécessaire à un rang LoRA élevé.
- **Kirkpatrick et al. 2017.** *Surmonter l’oubli catastrophique dans les réseaux neuronaux.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la caractérisation originale de la raison pour laquelle les réseaux neuronaux « oublient » l’entraînement antérieur lorsque vous effectuez un ajustement sur de nouvelles données (EWC — consolidation du poids élastique).
- **Wang et al. 2023.** *Apprentissage de sous-espace orthogonal pour l’apprentissage continu de modèles de langage.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, une approche antérieure de l’utilisation de LoRA pour l’apprentissage continu en contraignant les nouveaux adaptateurs à des sous-espaces orthogonaux.
- **Yadav et al. 2023.** *TIES-Merging : résolution des interférences lors de la fusion de modèles.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — une technique fondamentale pour fusionner plusieurs modèles ajustés sans interférence.
- **Qiao & Mahdavi 2025.** *Fusionner avant d’oublier : un apprentissage continu LoRA unique via une fusion continue.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — l’algorithme spécifique que le fusionneur multi-exécution de Backpropagate met en œuvre. Un prépublication de décembre 2025 ; Backpropagate est le premier utilisateur connu de cet article.

## Licence

MIT — voir [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
