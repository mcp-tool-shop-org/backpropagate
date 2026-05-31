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

# Entraînez un adaptateur. Envoyez-le à Ollama. Passez à autre chose

Backpropagate est une bibliothèque Python pour affiner les grands modèles de langage sur une seule GPU. Trois lignes de code suffisent pour entraîner un modèle de 7 milliards de paramètres sur une carte de 16 Go. Une commande supplémentaire l'exporte vers Ollama afin que vous puissiez exécuter votre modèle affiné avec `ollama run`. Fonctionne parfaitement sous Windows.

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

## Ce que vous pouvez affiner sur une GPU grand public de 16 Go

Voici les limites pratiques sur une carte de 16 Go (RTX 4080 / 5080 / 4070 Ti Super) :

| Modèle | Méthode | État |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | Facile. Longueur de séquence complète, beaucoup de marge. |
| SmolLM3-3B / Qwen2.5-3B / Llama-3.2-3B / Llama-3.2-1B | `mode="full"` (affinement complet) | v1.4 : passez `--mode=full` dans `backprop train` ou `Trainer(..., mode="full")`. Charge les poids en pleine précision (bf16) ; pas de 4 bits, pas d'adaptateur ; le contrôle de gradient et l'Adam paginé à 8 bits permettent de maintenir l'empreinte dans les 16 Go. |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | Standard. Environ 7 à 8 Go. Paramètres par défaut de Backpropagate. |
| Llama-3 13B | QLoRA + échantillonnage par lots | Difficile, mais fonctionne. Utilisez des séquences plus courtes. |
| Mixtral 8x7B (47 milliards de paramètres au total) | — | Hors de portée : la quantification à 2 bits (AQLM / QuIP#) rompt le contrat d'adaptateur fusionnable + exportation GGUF, elle a donc été abandonnée dans le [bref aperçu de la trajectoire v1.5](docs/V1_5_BRIEF.md). Sur une carte de 16 Go, utilisez une base ≤8 milliards de paramètres. |

`mode="full"` prend en charge les modèles jusqu'à **4 milliards de paramètres**. Les quatre préréglages de la ligne d'affinement complet ci-dessus sont authentiques (nombre réel de paramètres de 3,08 à 3,24 milliards) et s'adaptent à une carte de 16 Go. La classe de 3,8 à 4 milliards de paramètres (Phi-4-mini-3.8B, Qwen-3.5-4B) est également acceptée, mais nécessite une carte de **24 Go ou plus** pour un affinement complet : les poids et les gradients seuls atteignent presque 16 Go avant l'optimiseur et les activations, de sorte qu'une carte de 16 Go doit utiliser `mode="lora"` pour ces modèles (ils se trouvent dans la ligne LoRA). Les modèles >4 milliards de paramètres quittent le programme avec `RUNTIME_FULL_FT_MODEL_TOO_LARGE`.

La quantification à 2 bits (AQLM / QuIP#) est **hors de portée**. Elle a été envisagée pour la version 1.4, puis abandonnée dans le [bref aperçu de la trajectoire v1.5](docs/V1_5_BRIEF.md) : une base à 2 bits ne peut pas être proprement réintégrée dans des poids en pleine précision, ce qui rompt le contrat d'adaptateur fusionnable → GGUF → Ollama (le but de la chaîne de traitement). Les leviers de marge que Backpropagate propose sont le **chemin de calcul FP8** (v1.5, `--fp8`, Blackwell/Hopper) et `mode="full"` pour les modèles ≤4 milliards de paramètres : les deux restent fusionnables et exportables.

Pour les modèles de 3 milliards de paramètres et moins, l'affinement complet (et non seulement LoRA) est possible sur 16 Go et est désormais disponible dans la version 1.4 sous la forme `mode="full"`. Passez `Trainer(..., mode="full")` ou `backprop train --mode=full --model phi-4-mini-3.8b` pour l'activer. Une barrière stricte refuse le mode pour les modèles > 4 milliards de paramètres avec `RUNTIME_FULL_FT_MODEL_TOO_LARGE`, en indiquant LoRA + les préréglages inférieurs à 4 milliards de paramètres comme options de secours. Consultez [la page du guide complet sur l'affinement](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/) pour les calculs de configuration et la comparaison de la qualité de Biderman 2024 / Thinking Machines 2025. Pour les modèles de 7 milliards de paramètres et plus, l'affinement complet nécessite une GPU de 24 Go ou plus : envisagez de louer une A100 dans le cloud, ou utilisez LoRA, ce qui, selon les recherches récentes, correspond à la qualité de l'affinement complet pour la plupart des tâches post-entraînement (voir [la section anti-argumentaire](#what-backpropagate-is-not-for) pour les références).

## Ce que Backpropagate NE permet PAS

Si votre cas d'utilisation est l'un des suivants, vous obtiendrez de meilleurs résultats avec une autre bibliothèque : Backpropagate n'est pas le bon choix et essayer de le faire fonctionner coûterait plus cher que de simplement utiliser l'outil approprié. La lecture de cette section avant de commencer vous évitera de devoir installer et abandonner :

- **Affinage complet des paramètres des modèles 7B+** — Backpropagate utilise LoRA/QLoRA, ce qui entraîne un petit adaptateur plutôt que de mettre à jour tous les poids. Pour les modèles 7B et plus, l’affinage complet nécessite 24 Go ou plus de mémoire GPU et ne peut pas être effectué sur une carte grand public de 16 Go. Pour les modèles 3B et moins, l’affinage complet est possible avec 16 Go et est disponible dans la version 1.4 sous la forme `mode="full"` (passez `Trainer(..., mode="full")` ou `--mode=full` dans l’interface en ligne de commande ; une restriction stricte déclenche `RUNTIME_FULL_FT_MODEL_TOO_LARGE` pour les modèles > 4B et nomme LoRA + les préréglages inférieurs à 4B comme solutions de repli). Pour résumer : des recherches récentes ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) montrent que LoRA, avec une configuration appropriée, correspond à la qualité de l’affinage complet pour la plupart des tâches post-entraînement (suivi des instructions, adaptation au domaine, personnalité/style) avec 67 % de la puissance de calcul. Ainsi, pour le travail que la plupart des utilisateurs souhaitent effectuer, vous ne perdez rien en vous en tenant à LoRA. `mode="full"` existe pour les cas où vous avez mesuré une différence de qualité et décidé d’investir davantage de puissance de calcul. Si vous avez réellement besoin d’un affinage complet d’un modèle 7B+, utilisez directement HuggingFace `transformers.Trainer` sur une carte de 24 Go ou plus.
- **Apprentissage par renforcement en ligne — PPO / GRPO / RLVR** — Backpropagate effectue un affinage supervisé en une seule étape, plus un affinage des préférences sans référence (ORPO est disponible dans la version 1.5 ; SimPO/KTO sont prévus). Ce que Backpropagate ne fait *pas*, c’est l’apprentissage par renforcement en ligne — PPO, GRPO ou RLVR — qui nécessite un modèle de récompense ou une boucle de génération et d’évaluation en plus de l’étape d’entraînement. Pour ces derniers, utilisez TRL ou LLaMA-Factory directement. (L’affinage des préférences sans référence s’intègre dans l’enveloppe d’une seule étape, car il n’y a pas de modèle de référence distinct à conserver en mémoire ; voir la note sur ORPO dans [Démarrage rapide](#quick-start).)
- **Entraînement multi-nœuds** — une seule GPU sur une seule machine. L’utilisation de plusieurs GPU sur une seule machine est possible (via `accelerate launch`), mais n’est pas officiellement prise en charge.
- **Entraînement macOS sur le système CUDA** — Apple Silicon n’a pas CUDA, donc le chemin CUDA doit s’exécuter sur une machine Linux ou Windows avec une GPU NVIDIA. Vous pouvez toujours exécuter le modèle entraîné sur un Mac via Ollama. **Nouveau dans la version 1.5 :** un système MLX expérimental (`--backend mlx`) entraîne un adaptateur LoRA de manière native sur Apple Silicon — voir [Apple Silicon (MLX)](#apple-silicon-mlx--experimental-v15). Il est limité à LoRA-SFT et a été mis en œuvre, mais pas encore validé sur du matériel réel. Par conséquent, pour tout ce qui dépasse un LoRA SFT (ORPO, affinage complet, FP8, exécution multiple), vous devriez toujours utiliser le système CUDA.
- **Tout ce qui se trouve en dehors des familles de modèles testées** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. D’autres modèles fonctionnent souvent, mais ne sont pas pris en charge dans les tests d’intégration continue.

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

### Affinage des préférences (ORPO)

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

Le taux d’apprentissage par défaut est automatiquement réduit à `8e-6` pour ORPO (la perte est plus marquée que pour SFT simple) ; ajustez `--orpo-beta` (par défaut `0.1`) pour pondérer la pénalité du rapport de cotes. Dans la version 1.5, ORPO est uniquement en mode `mode="lora"`. SimPO et KTO sont prévus comme prochaines étapes ; pour l’apprentissage par renforcement en ligne (PPO/GRPO), consultez [Ce que la rétropropagation n’est PAS](#what-backpropagate-is-not-for).

### SFT avec suivi du raisonnement (distillation R1)

Nouveau dans la version 1.5 : distillez un modèle de raisonnement de manière simple. Utilisez `--reasoning-trace` (CLI) ou `Trainer(..., reasoning_trace=True)` (Python) et fournissez-lui des données qui conservent une chaîne de pensée `<think>...</think>` dans les interactions de l’assistant — la moitié SFT pure de la distillation de [DeepSeek-R1](https://arxiv.org/abs/2501.12948), sans apprentissage par renforcement requis. La rétropropagation conserve `<think>` dans l’objectif d’entraînement, supprime les données vides ou trop longues (filtrage de la longueur des données) et augmente la valeur par défaut de `max_seq_length` à 8192 pour la chaîne de pensée plus longue. Il est essentiel que `<think>` reste du **texte brut** — aucun jeton spécial, aucun redimensionnement de l’intégration — de sorte que le GGUF fusionné puisse toujours être exporté vers Ollama comme tout autre modèle affiné. Uniquement SFT. Consultez la [recette reasoning-trace](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) pour la forme de l’ensemble de données et les jetons ajustables.

### Apple Silicon (MLX) — expérimental, version 1.5

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

> ⚠️ **État honnête :** l’option MLX est incluse dans la version 1.5, **construite et testée avec des simulations**, mais **pas encore vérifiée sur du matériel Apple Silicon réel** — `mlx-lm` est uniquement pour Apple et n’a pas pu être exécutée sur la machine NVIDIA sur laquelle ce code a été écrit. Considérez-la comme expérimentale (même approche que l’option FP8) et veuillez [signaler les anomalies](#reporting-bugs) une fois qu’elle sera exécutée sur un Mac de la série M. Forcer `--backend mlx` sur un hôte non Apple génère une erreur `CONFIG_INVALID_SETTING` ; l’absence de la chaîne d’outils `mlx_lm` sur un Mac génère une erreur `DEP_MLX_UNAVAILABLE`.

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

La prise en charge de Python 3.10 sera interrompue en octobre 2026 et sa suppression est prévue dans la version 1.5. Pour les nouvelles installations, privilégiez Python 3.11 ou 3.12 ; la version 3.11 est la version la plus testée.

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

D’autres modèles fonctionnent souvent, mais seuls ces huit sont pris en charge dans les tests CI. Utilisez l’option `--lora-preset=quality` (valeur par défaut) pour les cibles de rang 256 / entièrement linéaires selon Biderman 2024 + Thinking Machines 2025, ou l’option `--lora-preset=fast` pour la cible héritée de rang 16 / q+v si vous avez besoin de l’empreinte de la version 1.2.x.

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
