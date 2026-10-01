<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

# Ajustez finement un modèle QLoRA de 32 milliards de paramètres, ou un modèle de 7 milliards de paramètres, sur une seule carte graphique. Déployez-le sur Ollama

Effectuez une rétropropagation pour ajuster finement de grands modèles de langage sur une **seule** carte graphique, dimensionnée en fonction de la carte dont vous disposez réellement. Trois lignes de code Python pour ajuster finement un modèle QLoRA de 7 à 32 milliards de paramètres sur une seule carte graphique grand public de 32 Go (RTX 5090). Un seul paramètre, `--full-ft-offload`, permet d’effectuer un ajustement complet d’un modèle de 7 milliards de paramètres en conservant ses poids et ses gradients dans la RAM de l’hôte (Linux ou WSL2 ; c’est lent, et les résultats sont présentés ci-dessous). Une commande supplémentaire permet d’exporter vers Ollama, puis `ollama run` votre modèle ajusté. Il peut être réduit à 16 Go. Fonctionne parfaitement sur Windows.

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

C’est tout. Il n’y a pas de fichier de configuration YAML. Il n’y a pas de « cérémonie » `accelerate launch`. Il n’y a pas de tutoriel distinct intitulé « convertissez-le maintenant au format GGUF ». Si vous avez une carte graphique CUDA et un fichier JSONL contenant vos données d’entraînement, vous n’avez plus que trois lignes à écrire pour obtenir un modèle ajusté fonctionnel.

## Installation

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

Si vous souhaitez les fonctionnalités optionnelles, remplacez l’installation par l’une des suivantes :

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

Préférez-vous Docker ? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` fonctionne également. Des images sont disponibles pour `linux/amd64` et `linux/arm64`, de sorte que les utilisateurs d’Apple Silicon et d’ARM Linux bénéficient d’une image native. Une version standard `compose.yaml` pour « interface utilisateur dans un conteneur » est disponible à la racine du dépôt : placez `user:password` dans un `ui-auth.txt` à côté, exécutez `docker compose up` et connectez-vous à `http://127.0.0.1:7860` (le premier démarrage crée l’interface, ce qui prend une minute ou deux). L’historique des exécutions est conservé dans `~/.backpropagate`.

## La place de Backpropagate dans l’écosystème

Il existe plusieurs bonnes bibliothèques pour l’ajustement fin des LLM. Elles sont toutes excellentes pour différentes tâches :

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** : si vous aimez les configurations YAML et que vous souhaitez disposer d’une communauté de recettes à partir desquelles vous pouvez vous inspirer.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** : si vous souhaitez utiliser DPO/PPO/RLHF et une interface graphique Web.
- **[Unsloth](https://github.com/unslothai/unsloth)** : si vous avez besoin de l’entraînement le plus rapide possible et que vous utilisez une famille de modèles prise en charge.
- **[torchtune](https://github.com/pytorch/torchtune)** : si vous souhaitez utiliser les recettes PyTorch natives de Meta que vous pouvez modifier.

Backpropagate est l’option manquante : une API Python en 3 lignes pour les utilisateurs individuels disposant d’une seule carte graphique grand public qui souhaitent entraîner un adaptateur et le déployer. Pas de YAML, pas d’interface graphique, pas d’apprentissage par renforcement en ligne (PPO/GRPO), pas de multi-nœuds. Juste la boucle dont tout le monde a réellement besoin et l’étape d’exportation qui pose problème.

Si vous avez essayé l’une des bibliothèques ci-dessus et que vous avez été rebuté par la « cérémonie » des fichiers de configuration, ou que vous avez rencontré un problème de famille de modèles, ou que vous souhaitiez des paramètres par défaut axés sur Windows, Backpropagate est fait pour vous.

## Ce que vous pouvez ajuster finement sur une seule carte graphique

Backpropagate dimensionne l’exécution en fonction de votre carte. Ce sont des chiffres **mesurés** du 30 septembre 2026 sur une carte RTX 5090 de 32 Go (preuves : [`docs/receipts/2026-09-30-offload/`](docs/receipts/2026-09-30-offload/)). Les pics de QLoRA se produisent à la taille de fenêtre de contexte complète du préréglage avec un lot de 1, ce qui est le pire des cas pour ce préréglage ; les exemples plus courts utilisent moins de ressources.

| Modèle | Méthode | Mesuré sur une carte de 32 Go |
|---|---|---|
| **14B** (Qwen2.5-14B) | QLoRA | **25,0 Go** au maximum avec une fenêtre de contexte de 4 096 (28,1 Go réservés). |
| 24B (Mistral-Small-24B) | QLoRA | 26,5 Go au maximum avec une fenêtre de contexte de 4 096 (29,6 Go réservés). |
| **32B** (Qwen2.5-32B) | QLoRA | **S’adapte tout juste :** 28,8 Go au maximum avec une fenêtre de contexte de 2 048 (30,7 Go réservés, environ 0,65 Go d’espace libre). |
| 3B | `mode="full"` (ajustement complet réel, sur la carte graphique) | **22,0 Go** au maximum (au niveau du système), 0,30 s/étape avec un lot de 4, une fenêtre de contexte de 512. 7,5 Go de ce total correspondent à l’état de l’optimiseur paginé, qui peut être déversé dans la RAM de l’hôte sur une carte moins puissante (non testé). |
| **7B** (Qwen2.5-7B, 7,6 milliards de paramètres) | `mode="full" --full-ft-offload` | **Entraîne :** 5,3 Go de VRAM, **30,8 Go de RAM de l’hôte** (32,2 Go lors de la sauvegarde), **14,7 s/étape**. Uniquement pour Linux ou WSL2. |

Non mesuré à nouveau lors de cette session : QLoRA 7B, Llama-3.1-8B (dépôt fermé, pas de jeton sur la machine de test) et ajustement complet uniquement sur GPU au-dessus de 3 milliards de paramètres. Les chiffres pour ces éléments, qui figurent ailleurs dans la documentation, sont des estimations.

Deux choses pour lesquelles la plupart des bibliothèques pour une seule carte graphique vous dirigent vers d’autres solutions, **QLoRA 24 à 32 milliards de paramètres** et **ajustement complet de modèles de 7 milliards de paramètres sur une seule carte**, Backpropagate le fait sur une seule carte graphique grand public, puis exporte le résultat directement vers Ollama.

**L’ajustement complet comporte deux approches.** Sans délestage, le modèle, ses gradients et l’état de l’optimiseur se trouvent tous sur la carte graphique. La bibliothèque limite la taille du modèle en fonction de la VRAM détectée (**16 Go → 4 milliards de paramètres, 24 Go → 5 milliards de paramètres, 32 Go → 6 milliards de paramètres**); ces limites sont basées sur des calculs de mémoire et ne sont mesurées que jusqu’à 3 milliards de paramètres. Remplacez-les avec `--full-ft-ceiling-billions`.

`--full-ft-offload` conserve les poids et les gradients dans la RAM de l’hôte et les transmet en continu à la carte graphique (délestage CPU FSDP2). Voici ce que cela coûte, mesuré :

- **RAM de l’hôte :** la vérification de l’adéquation demande environ 3,7 Go par milliard de paramètres, plus 10 Go, ce qui est une valeur conservatrice (39 Go pour 7,6 milliards de paramètres, contre les 32,2 Go mesurés). L’exécution est refusée dès le départ, avec les chiffres, si la machine ne peut pas le prendre en charge. Un modèle de 7,6 milliards de paramètres ne peut pas être utilisé avec une limite de mémoire WSL2 de 28 Go ; environ 5 milliards de paramètres constituent la limite pratique.
- **Vitesse :** 14,7 s/étape pour 7,6 milliards de paramètres (lot de 1) et 5,1 s/étape pour 3 milliards de paramètres (lot de 4), contre environ 0,63 s/étape pour 3 milliards de paramètres sur la carte graphique avec un lot de 4. N’utilisez-le que lorsque le modèle ne peut pas être utilisé sans cela. Une version plus rapide est prévue.
- **Optimiseur :** Adafactor, et non AdamW. Les poids restent en bf16 et chaque mise à jour est réécrite avec un arrondi stochastique ; il n’y a pas de copie fp32.
- **Qualité :** sur une seule exécution de 3 milliards de paramètres (150 étapes, perte sur un ensemble de données distinct, une seule graine), il a atteint environ 85 % de l’amélioration obtenue par l’ajustement complet ordinaire (2,45 → 1,93 contre 2,45 → 1,84). Une seule graine ne constitue pas une référence.
- **Portée :** simple ajustement supervisé. Pas d’empaquetage, pas de masquage uniquement des réponses, pas de points de contrôle intermédiaires, pas de reprise. Uniquement pour Linux ou WSL2 (FSDP2 nécessite NCCL) ; sur Windows natif, il s’arrête avec `DEP_FSDP_UNAVAILABLE`.
- **Pas encore testé :** longues exécutions, accumulation de gradients supérieure à 1 et une machine physique de 64 Go (la machine de test avait plus de RAM, avec une limite de 60 Go imposée par le test).

Un modèle qui ne convient pas se termine avec `RUNTIME_FULL_FT_MODEL_TOO_LARGE` et indique la voie à suivre. Voir [la page complète du guide d’optimisation](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/).

### Réduction à 16 Go

L’espace de 16 Go (RTX 4080 / 5080 / 4070 Ti Super) reste de première qualité : 7 B QLoRA, et véritable optimisation complète d’un modèle d’environ 3 B (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B/1B) via `mode="full"`, (22,0 Go mesurés sur une carte de 32 Go à 3 B, dont 7,5 Go d’état d’optimiseur paginé qui peut déborder vers la RAM de l’hôte ; on n’a pas vérifié si cela fonctionne de manière acceptable sur une carte de 16 Go). Avec `--full-ft-offload`, la GPU conserve beaucoup moins : avec la VRAM limitée sur la carte de test, un modèle de 3 B est entraîné avec une limite de 6 Go et des modèles de 4 B et 7,6 B avec une limite de 8 Go. Ce sont des limites simulées sur une carte de 32 Go, et non des exécutions sur du matériel réel de 8 Go. Le même code sélectionne la taille du lot et le plafond qui conviennent à la carte qu’il détecte.

La quantification à 2 bits (AQLM / QuIP#) est **hors de portée** : une base à 2 bits ne peut pas être fusionnée proprement avec des poids en pleine précision, ce qui rompt le contrat d’exportation adaptateur fusionnable → GGUF → Ollama (le but principal du pipeline). Backpropagate propose plutôt des leviers d’optimisation : QLoRA, `mode="full"`, `--full-ft-offload` et le chemin de calcul FP8 (`--fp8`, Blackwell/Hopper), qui restent tous fusionnables et exportables.

## Ce que Backpropagate NE permet PAS

Si votre cas d’utilisation est celui qui suit, vous obtiendrez de meilleurs résultats avec une bibliothèque différente : Backpropagate n’est pas le bon choix et essayer de le faire fonctionner coûterait plus cher que de simplement utiliser l’outil approprié. La lecture de cette section avant de commencer vous évitera de devoir installer, puis de changer de bibliothèque :

- **Optimisation complète des paramètres des modèles de 13 B+** : Backpropagate effectue une optimisation complète jusqu’à environ 6 B sur une GPU de 32 Go et un modèle de classe 7 B avec `--full-ft-offload` (voir [l’espace disponible](#what-you-can-fine-tune-on-one-gpu)). Une optimisation complète d’un modèle de 13 B+ nécessite une FSDP multi-GPU ou une carte plus grande. Avant de dépenser autant de ressources de calcul, évaluez les arguments des deux côtés. [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/) indique que LoRA correspond à une optimisation complète lorsqu’elle est appliquée à chaque couche et que l’ensemble de données correspond à la capacité de l’adaptateur, soit environ les deux tiers des ressources de calcul par passage. [Biderman et al. 2024](https://arxiv.org/abs/2405.09673) ont constaté que, dans les contextes de faible rang standard, LoRA est nettement moins performant que l’optimisation complète pour le code et les mathématiques, tout en oubliant moins d’informations. Pour le suivi des instructions, la personnalisation et le travail sur des ensembles de données modestes, QLoRA jusqu’à 32 B est généralement la meilleure utilisation d’une seule carte.
- **Apprentissage par renforcement en ligne : PPO / GRPO / RLVR** : Backpropagate effectue une optimisation SFT en une seule étape, ainsi qu’une optimisation des préférences sans référence (ORPO dans la version 1.5 ; SimPO + KTO dans la version 1.6). Ce qu’il ne fait PAS, c’est l’apprentissage par renforcement en ligne : PPO, GRPO ou RLVR, qui nécessite un modèle de récompense ou une boucle de génération et de notation en plus de l’étape d’entraînement. Pour cela, utilisez TRL directement ou LLaMA-Factory. (L’optimisation des préférences sans référence correspond à l’espace d’une seule étape, car il n’y a pas de modèle de référence distinct à conserver en mémoire ; voir la note sur ORPO dans [Démarrage rapide](#quick-start).)
- **Entraînement multi-nœuds** : une seule GPU sur une seule machine. Le multi-GPU sur une seule machine fonctionne (via `accelerate launch`), mais n’est pas pris en charge officiellement.
- **Entraînement macOS sur le rail CUDA** : Apple Silicon n’a pas de CUDA, le chemin CUDA s’exécute donc sur une machine Linux ou Windows avec une GPU NVIDIA. Vous pouvez toujours exécuter le modèle entraîné sur un Mac via Ollama. Un rail MLX **expérimental et non vérifié** (`--backend mlx`) entraîne un adaptateur LoRA en mode natif sur Apple Silicon ; voir [Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview). Il ne prend en charge que LoRA-SFT et n’a pas été vérifié en conditions réelles (pas de prise en charge), donc pour tout ce qui dépasse un SFT LoRA (ORPO, optimisation complète, FP8, exécution multiple), vous devez utiliser le rail CUDA.
- **Tout ce qui se situe en dehors des familles de modèles testées** : Qwen 2.5 / 3.5 (7 B / 4 B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3 B / 1 B), Mistral 7 B. D’autres modèles fonctionnent souvent, mais ne sont pas pris en compte dans les tests CI.

Si vous avez besoin de l’une de ces choses, utilisez l’une des bibliothèques répertoriées ci-dessus. Elles sont plus performantes pour cela.

## Ce que Backpropagate vous offre

Quatre éléments, dans une seule installation :

**1. Une véritable API en 3 lignes qui fonctionne sans fichier de configuration.**
L’extrait en haut de ce fichier README s’exécute de bout en bout. Pas de `accelerate config`, pas de YAML, pas de remplacements Hydra. Il suffit de `Trainer(model).train(data)` et vous avez une optimisation.

**2. Windows qui fonctionne réellement.**
La plupart des bibliothèques ML considèrent Windows comme une réflexion après coup. Backpropagate est testé en priorité sur Windows + RTX 5080. La bibliothèque gère les particularités de l’exécution pour vous : elle sait comment pré-tokeniser vos données afin que le multiprocessing Windows ne plante pas, elle désactive automatiquement xformers sur les cartes RTX 40/50 où cela causerait des problèmes, et elle sélectionne les paramètres du chargeur de données qui ne provoquent pas de plantage. Vous n’avez pas besoin d’en connaître les détails. Il fonctionne simplement.

**3. Conçu pour les exécutions sans surveillance.**
L’entraînement prend des heures. Vous ne voulez pas le surveiller. Backpropagate est conçu pour fonctionner en continu :

- Si vous manquez de mémoire GPU, il réduit automatiquement de moitié la taille du lot et réessaie, jusqu’à trois fois. Pas de réglage manuel.
- Si votre GPU devient trop chaud, il fait une pause jusqu’à ce que les choses se refroidissent, puis il reprend.
- Chaque point de contrôle est écrit de manière atomique : si votre ordinateur portable plante au milieu de l’enregistrement, le point de contrôle précédent et valide reste intact.
- Chaque exécution d’entraînement reçoit un ID unique qui est estampillé sur chaque ligne de journal, chaque point de contrôle et chaque entrée Weights & Biases. Si quelque chose ne va pas, un seul ID permet à un mainteneur de corréler tous les éléments.
- Les erreurs sont accompagnées de codes stables (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, etc.) afin que vous puissiez rechercher dans vos journaux et dans le [guide de dépannage](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) pour trouver la solution. Les erreurs spécifiques à CUDA ont une [page de dépannage CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) dédiée.

**4. Une seule commande pour appliquer l’adaptateur entraîné à `ollama run`.**
De nombreuses bibliothèques entraînent un modèle. Peu d’entre elles facilitent son utilisation lorsque vous souhaitez réellement l’utiliser. Backpropagate exporte vers GGUF (le format utilisé par Ollama) et enregistre un modèle Ollama en une seule commande. Vous passez de « l’entraînement est terminé » à « je peux discuter avec mon modèle affiné » en environ 30 secondes.

## Démarrage rapide

À partir de la ligne de commande, avec un ensemble de données d’exemple de 5 conversations :

```bash
pipx install "backpropagate[standard]"
curl -LO https://raw.githubusercontent.com/mcp-tool-shop-org/backpropagate/main/examples/quickstart.jsonl

backprop train --data quickstart.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 10
backprop generate ./output "What is Python?"      # did it learn anything?
backprop export ./output --format gguf --quantization q4_k_m --ollama --ollama-name my-first-finetune
ollama run my-first-finetune
```

`backprop train` écrit l’adaptateur dans `./output` (modifiez-le avec `--output`). En Python, la même chose s’écrit :

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")
```

Utilisez un environnement virtuel avec `pip install "backpropagate[standard]"` pour l’API Python ; `pipx` installe la commande `backprop` dans son propre environnement, de sorte que `import backpropagate` ne la trouvera pas.

**Ce dont l’exportation GGUF a besoin.** L’exportation fusionne votre adaptateur dans le modèle de base et le convertit à l’aide du script de conversion de llama.cpp. Vous avez besoin soit :

- d’une copie de la source de llama.cpp (`git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp`) plus `pip install sentencepiece protobuf` dans le même environnement, ou
- d’Unsloth avec son propre llama.cpp déjà installé.

Avec `--ollama`, la quantification `q4_k_m` est effectuée par `ollama create`, il n’est donc rien à compiler. Backpropagate ne permet jamais à Unsloth d’installer des paquets système pour construire llama.cpp pour vous ; définissez `BACKPROPAGATE_UNSLOTH_AUTO_INSTALL=1` si vous le souhaitez. Détails : [export](https://mcp-tool-shop-org.github.io/backpropagate/handbook/export/).

Pour vos propres données, formatez votre fichier JSONL avec un exemple par ligne :

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca (`instruction` / `output`), OpenAI chat (`messages`) et les formats de texte brut fonctionnent également : Backpropagate détecte automatiquement le format.

### La boucle : vérifier les données, entraîner, évaluer, exporter

```bash
backprop data report my_data.jsonl                     # duplicates, length outliers, format problems
backprop data split my_data.jsonl --heldout-ratio 0.1  # a held-out set the model never trains on
backprop train --data my_data.train.jsonl --steps 200 --output ./run-a
backprop eval <run-id> --heldout my_data.heldout.jsonl # held-out loss + sample generations
backprop eval <run-b> --vs <run-a>                     # did the change help?
backprop export ./run-a --format gguf --ollama --ollama-name my-model
```

L’évaluation est conçue pour être sans jugement : perte des données non utilisées plus métriques de tâche déterministes (`normalized_exact_match`, `token_f1`, `contains`, `regex`, `pass_rate`). Pour utiliser un juge LLM, exécutez-le sur `backprop generate` et analysez les résultats vous-même. Voir [recettes](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Affinement par préférence (ORPO, SimPO, KTO)

Entraînez-vous sur des préférences au lieu de simples démonstrations. ORPO est sans référence et en une seule étape : il intègre le signal de préférence dans l’étape SFT, il n’y a donc pas de modèle de récompense ou de référence distinct et la forme en 3 lignes reste inchangée. Passez `--method orpo` (CLI) ou `method="orpo"` (Python) et fournissez-lui un ensemble de données de `{prompt, chosen, rejected}` (ou simplement `{chosen, rejected}`) lignes :

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

Le taux d’apprentissage par défaut diminue automatiquement à `8e-6` pour ORPO (la perte est plus marquée que pour le simple SFT) ; ajustez `--orpo-beta` (par défaut `0.1`) pour pondérer la pénalité du rapport de cotes. ORPO est `mode="lora"` uniquement.

**Nouveauté dans la version 1.6 : SimPO et KTO.** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) est sans référence avec une récompense normalisée en fonction de la longueur et prend les mêmes données appariées `{prompt, chosen, rejected}` que ORPO (`--simpo-beta`, `--simpo-gamma`). `--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) prend des données **non appariées** `{prompt, completion, label}` : des évaluations positives/négatives par exemple — pour la vaste classe de commentaires qui ne sont pas des paires A/B organisées ; il équilibre automatiquement les pondérations de perte souhaitables/indésirables à partir de vos décomptes d’étiquettes. Les deux sont `mode="lora"` uniquement et restent dans l’enveloppe SFT à GPU unique (pas de modèle de référence distinct). Voir le [guide d’affinement par préférence](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) pour savoir lequel utiliser. Pour l’apprentissage par renforcement en ligne (PPO/GRPO), voir [ce que Backpropagate n’est PAS conçu pour faire](#what-backpropagate-is-not-for).

### Affinement SFT basé sur la trace de raisonnement (distillation R1)

Distillez un modèle de raisonnement de manière simple. Passez `--reasoning-trace` (CLI) ou `Trainer(..., reasoning_trace=True)` (Python) et fournissez-lui des traces qui conservent une chaîne de pensée `<think>...</think>` dans la réponse de l’assistant : la moitié SFT pure de la distillation [DeepSeek-R1](https://arxiv.org/abs/2501.12948), aucun apprentissage par renforcement requis. Backpropagate conserve `<think>` dans la cible d’entraînement, supprime les traces vides ou trop longues (filtrage de la longueur de la trace) et augmente la valeur par défaut de `max_seq_length` à 8192 pour la chaîne de pensée plus longue. Plus important encore, `<think>` reste du **texte brut** : pas de jetons spéciaux, pas de redimensionnement d’intégration — de sorte que le GGUF fusionné s’exporte toujours vers Ollama comme tout autre modèle affiné. SFT uniquement. Voir la [recette de la trace de raisonnement](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) pour la forme de l’ensemble de données et les jetons réglables.

### Apple Silicon (MLX) : aperçu non vérifié

> ⚠️ **Aperçu non vérifié : ne fait pas partie de l’ensemble des fonctionnalités prises en charge.** Le rail MLX est construit et testé avec des tests unitaires, mais il n’a **pas** été vérifié en conditions réelles sur du matériel Apple Silicon (`mlx-lm` est spécifique à Apple et ne peut pas s’exécuter sur les machines NVIDIA sur lesquelles Backpropagate est développé). Considérez tout ce qui suit comme expérimental, utilisez-le à vos propres risques et [signalez les anomalies](#reporting-bugs) si vous l’exécutez sur un Mac de la série M.

**Une API, deux rails.** CUDA est le backend canonique et vérifié ; MLX est un deuxième rail qui s’entraîne sur un Mac de la série M via la chaîne d’outils [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) d’Apple (mémoire unifiée, pas de CUDA). La forme en 3 lignes sélectionne le rail en fonction du matériel : `backend='auto'` (par défaut) achemine vers CUDA sur NVIDIA et vers MLX sur Apple Silicon, de sorte que les installations CUDA existantes sont identiques au niveau des octets :

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

Le rail MLX est **SFT LoRA uniquement** : pas d’ORPO, pas de FP8, pas de `mode='full'`, pas d’exécution multiple (chacun est rejeté avec `CONFIG_INVALID_SETTING` ; utilisez `backend='cuda'`/`'auto'` sur une machine NVIDIA pour ces options). L’adaptateur résultant est un simple fichier safetensors et s’exporte vers Ollama via le même chemin que le rail CUDA.

> Forcer `--backend mlx` sur un hôte non Apple génère une erreur avec `CONFIG_INVALID_SETTING` ; l’absence de la chaîne d’outils `mlx_lm` sur un Mac génère `DEP_MLX_UNAVAILABLE`.

Pour des flux de travail de bout en bout plus complets (affiner et publier sur HF Hub, reprendre après une erreur de mémoire insuffisante, SLAO multi-exécution sur une longue campagne, etc.), voir la [page des recettes du guide](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interface utilisateur Web (facultatif)

Si vous préférez cliquer plutôt que taper en Python, installez l’extra de l’interface utilisateur et lancez :

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Ouvrez l’URL qu’il affiche, `http://127.0.0.1:7862/?token=...` (chaque lancement crée un nouveau jeton ; le premier démarrage configure l’interface et peut prendre une ou deux minutes). Il s’agit d’une interface web locale pour parcourir les ensembles de données, valider les formats et assembler visuellement une configuration d’entraînement. L’entraînement lui-même s’exécute via `backprop train` (l’entraînement piloté par l’interface utilisateur est prévu — le bouton Démarrer affiche actuellement cette note). Par défaut, l’interface utilisateur est uniquement locale. Pour la rendre accessible à d’autres appareils, consultez la section [Interface utilisateur web](#interface-utilisateur-web) ci-dessous pour le contrat de sécurité `--share` + `--auth`.

## Entraînement multi-exécution

Si vous souhaitez affiner progressivement sur plusieurs ensembles de données — par exemple, si vous recevez de nouvelles données d’entraînement chaque semaine et que vous souhaitez les ajouter sans oublier ce que vous avez appris auparavant — le mode `multi_run` de Backpropagate est fait pour vous :

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

Cela exécute cinq passes d’entraînement, en fusionnant l’adaptateur entre les exécutions de manière à préserver les connaissances antérieures tout en intégrant de nouveaux exemples. La technique est basée sur des recherches récentes sur l’apprentissage continu — voir [Références](#références) en bas de ce fichier README.

La version CLI :

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Reprise à partir d’un point de contrôle

Un entraînement en cinq étapes qui plante à l’étape 4 peut être récupéré. Chaque session multi-exécution enregistre son ID d’exécution dans l’historique et le manifeste des points de contrôle stockés sur le disque, de sorte que la reprise là où vous vous êtes arrêté se fait en une seule commande :

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

Le comportement par défaut de `backprop multi-run` (sans `--resume`) détecte automatiquement une entrée en cours dans le même répertoire de sortie et la poursuit. Pour forcer un nouveau démarrage, pointez vers un nouveau répertoire de sortie.

## Historique de l’entraînement

Chaque invocation de `backprop train` et `backprop multi-run` enregistre une ligne dans `<output>/run_history.json` : modèle utilisé, ensemble de données, hyperparamètres, statut, perte finale, historique des pertes. Vous pouvez lister et examiner les exécutions passées :

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Suivi des expériences

Backpropagate détecte automatiquement les outils de suivi des expériences installés (Weights & Biases, TensorBoard, MLflow) et les configure. Si `wandb` est installé et que vous êtes connecté, chaque exécution enregistre automatiquement les données dans W&B avec un nom d’exécution correspondant à l’ID d’exécution stocké sur le disque — vous pouvez donc effectuer une recherche dans W&B, vos journaux et `run_history.json` en utilisant un seul identifiant.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Remplacez par `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` ou `Trainer(report_to="none")` pour vous désinscrire.

## Interface utilisateur web

L’interface web Reflex est activée par défaut — installez-la avec `pipx install "backpropagate[ui]"` et lancez-la :

```bash
backprop ui --port 7862
```

L’interface utilisateur s’exécute localement : ouvrez l’URL qu’elle affiche, `http://127.0.0.1:7862/?token=...`. Sans `--auth`, chaque lancement génère un nouveau jeton et l’interface utilisateur refuse les requêtes qui n’en contiennent pas. Aujourd’hui, elle couvre la moitié du flux de travail, à savoir la **navigation / validation / configuration** — pointez-la vers un ensemble de données, vérifiez le format et les statistiques détectés automatiquement, choisissez un modèle et assemblez une configuration d’exécution. **Le lancement de l’exécution se fait à partir de la ligne de commande** (`backprop train` / `backprop multi-run`) ; le bouton Démarrer dans l’interface utilisateur affiche une note pointant vers cette option. L’entraînement piloté par l’interface utilisateur est une fonctionnalité ultérieure prévue — jusqu’alors, l’interface utilisateur est le point d’entrée et la ligne de commande est le déclencheur.

Pour la rendre accessible à d’autres appareils (d’autres personnes sur votre réseau, une URL publique, etc.), vous devez associer `--share` (ou `--host`) à `--auth` :

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` sans `--auth` se termine par une erreur. La raison : `--share` publie une URL accessible à toute personne sur Internet, et sans authentification, cela signifie que n’importe qui peut exécuter votre pipeline d’entraînement et lire votre jeton HuggingFace. Il n’est pas possible de se désinscrire de cette option — si vous ne souhaitez pas définir d’informations d’identification, utilisez plutôt le transfert de port SSH :

```bash
# On the client:
ssh -L 7862:localhost:7862 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open the URL the server printed (http://127.0.0.1:7862/?token=...) locally
```

Consultez la page [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) pour obtenir le modèle de menace complet.

Les écritures sur le système de fichiers à partir de l’interface utilisateur sont isolées dans un seul répertoire :

- Par défaut : `~/.backpropagate/ui-outputs`
- Remplacement : définissez `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- Le remplacement est validé par une liste de blocage — les chemins d’accès au système ou aux informations d’identification (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) sont refusés.

## Notes sur la plateforme

**Configuration requise :** Python 3.10+ · GPU CUDA (8 Go+ de VRAM) · PyTorch 2.0+

Python 3.10 est pris en charge jusqu’à la version 1.6 au moins ; il atteindra la fin de sa durée de vie en octobre 2026 et sa suppression est prévue dans la première version après cette date. Pour les nouvelles installations, préférez Python 3.11 ou 3.12 — 3.11 est la version la plus testée.

Backpropagate gère les particularités d’exécution de l’entraînement sur différentes plateformes, mais il ne peut pas corriger les problèmes d’installation. Les deux problèmes les plus courants sont :

- **Mauvaise version de CUDA.** PyTorch est publié avec un seul binaire par version de CUDA. Si vous choisissez la mauvaise, vous obtiendrez silencieusement PyTorch en mode CPU uniquement et l’entraînement sera impossible. Utilisez le sélecteur de versions disponible à l’adresse <https://pytorch.org/get-started/locally/> pour votre pilote. Exécutez `nvidia-smi` pour afficher votre version de pilote / CUDA.
- **Windows + exportation GGUF.** L’option `[export]` supplémentaire crée `llama-cpp-python` à partir du code source, ce qui nécessite les outils de création Visual Studio (composant C++) et CMake.

**macOS :** la prise en charge de CUDA n’est pas disponible (pas de CUDA) — une invocation de `trainer.train()` avec CUDA génère `DEP_GPU_NOT_AVAILABLE`, et vous pouvez exécuter l’adaptateur entraîné sur un Mac via Ollama. Une option **expérimentale et non vérifiée** MLX (`--backend mlx`, `pip install 'backpropagate[mlx]'`) entraîne un adaptateur LoRA en mode natif sur Apple Silicon via `mlx_lm.lora` — uniquement LoRA SFT, et **non vérifiée sur du matériel réel** (voir [Apple Silicon (MLX)](#apple-silicon-mlx--non-vérifié)). Pour le chemin CUDA, ou pour ORPO / affinage complet / FP8 / multi-exécution, utilisez une machine Linux ou Windows avec CUDA.

Consultez la [page du guide de dépannage](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) pour obtenir le guide complet de résolution des problèmes d’installation, et la [page de dépannage CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) pour les problèmes de pilote / VRAM / xformers / bf16 par rapport à fp16.

## CLI

Chaque API Python a un équivalent en ligne de commande :

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

Référence complète à la page [du guide de la ligne de commande](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), ou `backprop <subcommand> --help`.

## Configuration

Chaque paramètre peut être remplacé par une variable d’environnement en utilisant le préfixe `BACKPROPAGATE_` :

| Variable | Valeur par défaut | Notes |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Forcer l’affichage des journaux JSON ou de la console |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modèle par défaut |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Taux d’apprentissage |
| `BACKPROPAGATE_LORA__R` | `256` | Rang LoRA (par défaut pour la version 1.3 ; indiquer `--lora-preset=fast` pour la valeur par défaut de la version 1.2.x, qui est 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Bac à sable du système de fichiers de l’interface utilisateur |

Les clés imbriquées utilisent un double soulignement (`MODEL__NAME`, et non `MODEL_NAME`). La référence complète se trouve dans [la page du manuel des variables d’environnement](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Préréglages de modèle

| Préréglage | VRAM | Licence | Notes |
|---|---|---|---|
| Qwen-3.5-4B | ~8 Go | Apache 2.0 | Valeur par défaut recommandée pour les modèles de moins de 5 Go. Meilleure qualité pour cette taille. |
| Phi-4-mini-3.8B | ~8 Go | MIT | Excellent en matière de raisonnement, de mathématiques et de code. Licence stricte et claire. |
| SmolLM3-3B | ~6 Go | Apache 2.0 | Recette entièrement ouverte. Contexte natif de 64 Ko. |
| Qwen 2.5 7B | ~12 Go | Apache 2.0 | Valeur par défaut existante. Meilleure qualité des préréglages 7 B existants. |
| Qwen 2.5 3B | ~8 Go | Qwen-Research | ⚠ Licence de recherche — consulter les conditions de licence de Qwen avant toute utilisation commerciale. |
| Llama 3.2 3B | ~8 Go | Llama Community | Alternative solide à Qwen 3B avec des réserves permissives. |
| Llama 3.2 1B | ~6 Go | Llama Community | Pour des expériences rapides sur des petites cartes. |
| Mistral 7B | ~12 Go | Apache 2.0 | Comparable à Qwen 7B, modèle de chat différent. |
| Llama-3.1-8B | ~7-8 Go (QLoRA) | Llama-3.1-Community | 8 B QLoRA, contexte natif de 128 Ko (la clause des > 700 millions d’utilisateurs actifs mensuels nécessite une licence Meta distincte). |
| **Qwen2.5-14B** | 25 GiB au maximum à 4 096 ctx (QLoRA) | Apache 2.0 | **Le modèle principal de 32 Go.** Rang/alpha 32, AdamW 8 bits. Les poids 4 bits seuls représentent environ 8,5 Go ; une fenêtre complète de 4 096 jetons nécessite le reste. |
| Mistral-Small-24B | 26,5 GiB au maximum à 4 096 ctx (QLoRA) | Apache 2.0 | 24 B QLoRA sur une carte de 32 Go. Les poids 4 bits seuls représentent environ 18 Go. |
| **Qwen2.5-32B** | 28,8 GiB au maximum à 2 048 ctx (QLoRA) | Apache 2.0 | **Au sommet de l’enveloppe de 32 Go.** S’adapte tout juste à `max_len 2048` avec AdamW 8 bits. |

D’autres modèles fonctionnent souvent ; les lignes ci-dessus sont les préréglages sélectionnés — la catégorie 14 B à 32 B est optimisée avec QLoRA pour une carte de 32 Go (l’enveloppe mesurée). Indiquer `--lora-preset=quality` (par défaut) pour les cibles de rang-256 / entièrement linéaire selon Biderman 2024 + Thinking Machines 2025, ou `--lora-preset=fast` pour la valeur de rang-16 / cible q+v héritée si vous avez besoin de l’empreinte de la version 1.2.x.

## Résolution des problèmes

Un bref index des échecs les plus courants lors de la première exécution. L’index inverse complet se trouve dans [la page du manuel de résolution des problèmes](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Pour une analyse approfondie du pilote / VRAM / précision mixte, consulter [la page de résolution des problèmes CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Symptôme | Code d’erreur | Solution |
|---|---|---|
| La mémoire de la GPU est épuisée au milieu de l’entraînement | `RUNTIME_GPU_OOM` | Automatique — Backpropagate réduit de moitié la taille du lot et réessaie jusqu’à 3 fois. Pour refuser : `Trainer(oom_recovery=False)`. Pour forcer une taille plus petite : `--batch-size 1`. |
| HuggingFace renvoie 401 / « modèle introuvable » | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` et réessayer. Pour les fautes de frappe, copier l’ID exact depuis <https://huggingface.co/models>. |
| `register_with_ollama` connexion refusée | `DEP_OLLAMA_REGISTRATION_FAILED` | Démarrer le démon : `ollama serve`. Installer depuis <https://ollama.com>. Peut être réessayé. |
| L’espace disque est insuffisant lors de la sauvegarde du point de contrôle | `STATE_CHECKPOINT_INVALID` | Les écritures atomiques laissent un répertoire `.partial` en cas de plantage — il est sûr de le supprimer. Le point de contrôle précédent est intact. |
| L’entraînement est interrompu en raison d’une surchauffe de la GPU | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automatique — Backpropagate fait une pause lorsque le seuil de température est atteint et reprend lorsque la GPU se refroidit. Améliorer le flux d’air si cela se reproduit. |
| `backprop ui --share` rejeté | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Indiquer `--auth user:password`, ou utiliser le transfert de port SSH à la place (voir [l’interface utilisateur Web](#web-ui)). |
| L’exportation GGUF a échoué lors de la première tentative | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]` ; sous Windows, vous avez également besoin des outils de création Visual C++ et de CMake. |

## Signaler les bogues

Lorsqu’un échec se produit, Backpropagate affiche une ligne au démarrage, comme `run_started run_id=<uuid>`, et associe le même ID à chaque ligne de journal, à chaque point de contrôle et à chaque entrée de Weights & Biases. **Inclure le `run_id` dans tout signalement de bogue** — cela permet à un responsable de corréler tous les éléments de cette exécution spécifique.

Un bon signalement de bogue comprend :

1. **Le `run_id`** — l’UUID affiché au démarrage. Un seul UUID permet à un responsable de corréler chaque ligne de journal, chaque point de contrôle et chaque entrée de Weights & Biases pour cette exécution spécifique.
2. **Le code d’erreur** — la ligne `[CODE_NAME]: message` dans stderr. Voir [les codes d’erreur](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) pour le catalogue des codes stables.
3. **La trace d’exécution éditée.** Stderr est automatiquement éditée en mode non verbeux (les jetons Bearer, `sk-*`, `hf_*`, les clés AWS, les paires `password=` / `token=` / `api_key=` sont supprimées) — il est sûr de la coller. Pour la trace d’exécution complète non éditée, relancer avec `BACKPROPAGATE_DEBUG=1` (ou `--verbose`) ; examiner avant de la publier.
4. **La sortie `backprop info`.** Une commande affiche Python / PyTorch / CUDA / modèle GPU / VRAM / OS / extras installés — tout ce dont le responsable a besoin pour identifier une régression spécifique à une plateforme.

Le [modèle de rapport de bug](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) contient des instructions précises pour chacun de ces éléments, ce qui accélère le processus de triage. Les questions, les idées ou les discussions du type « est-ce normal ? » doivent être publiées dans les [discussions GitHub](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Les problèmes de sécurité doivent être signalés en privé via le formulaire [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) — consultez le fichier [SECURITY.md](SECURITY.md) pour connaître la politique et les délais de réponse.

## Confidentialité

Tout l’apprentissage se fait localement sur votre GPU. Backpropagate n’effectue aucune requête réseau, sauf pour télécharger les modèles depuis Hugging Face (ce que vous initiez). Pas de télémétrie, pas de dépendance au cloud.

## Références

Les paramètres par défaut de Backpropagate et le mode d’entraînement multi-exécution sont basés sur des recherches récentes. Si vous êtes intéressé par les techniques sous-jacentes :

- **Hu et al. 2021.** *LoRA : Adaptation de faible rang de grands modèles de langage.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — l’article fondateur qui présente LoRA, qui est la méthode utilisée par Backpropagate pour entraîner efficacement les adaptateurs.
- **Biderman et al. 2024.** *LoRA apprend moins et oublie moins.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — preuves empiriques que LoRA avec un rang de 256 et des cibles entièrement linéaires correspond à la qualité d’un affinage complet sur la plupart des tâches post-entraînement, pour 67 % de la puissance de calcul. Cela détermine la configuration par défaut de LoRA v1.3 de Backpropagate.
- **Thinking Machines 2025.** *LoRA sans regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — le suivi pratique qui identifie la correction de 10 fois du taux d’apprentissage par rapport à l’affinage complet nécessaire à un rang LoRA élevé.
- **Kirkpatrick et al. 2017.** *Surmonter l’oubli catastrophique dans les réseaux neuronaux.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la caractérisation originale de la raison pour laquelle les réseaux neuronaux « oublient » les apprentissages antérieurs lorsque vous effectuez un affinage sur de nouvelles données (EWC — Consolidation du poids élastique).
- **Wang et al. 2023.** *Apprentissage de l’espace orthogonal pour l’apprentissage continu de modèles de langage.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, une approche antérieure de l’utilisation de LoRA pour l’apprentissage continu en contraignant les nouveaux adaptateurs à des sous-espaces orthogonaux.
- **Yadav et al. 2023.** *TIES-Merging : résolution des interférences lors de la fusion de modèles.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — une technique fondamentale pour fusionner plusieurs modèles affinés sans interférence.
- **Qiao & Mahdavi 2025.** *Fusionner avant d’oublier : un apprentissage continu unique de LoRA via une fusion continue.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — l’algorithme spécifique que le fusionneur multi-exécution de Backpropagate met en œuvre. Un prépublication de décembre 2025 ; Backpropagate est le premier utilisateur connu de cet article.

## Licence

MIT — voir [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
