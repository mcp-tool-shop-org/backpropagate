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

Backpropagate est une bibliothèque Python pour l'ajustement fin de grands modèles de langage sur un seul GPU. Trois lignes de code suffisent pour entraîner un modèle de 7 milliards de paramètres sur une carte de 16 Go. Une seule commande permet de l'exporter vers Ollama, ce qui vous permet d'utiliser la commande `ollama run` pour exécuter votre modèle ajusté. Il fonctionne parfaitement sous Windows.

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

C'est tout. Il n'y a pas de fichier de configuration YAML. Il n'y a pas de "cérémonie" `accelerate launch`. Il n'y a pas de tutoriel séparé pour "convertir ensuite en GGUF". Si vous avez un GPU CUDA et un fichier JSONL contenant vos données d'entraînement, vous n'êtes qu'à trois lignes d'un ajustement fin fonctionnel.

## Installation

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

Si vous souhaitez les fonctionnalités optionnelles, remplacez l'installation par l'une de ces options :

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

Préférez Docker ? La commande `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` fonctionne également. Des images sont disponibles pour `linux/amd64` et `linux/arm64`, ce qui permet aux utilisateurs d'Apple Silicon et d'ARM Linux d'utiliser une image native. Un fichier `compose.yaml` standard pour "l'interface utilisateur dans un conteneur" se trouve à la racine du dépôt. La commande `docker compose up` lance l'interface utilisateur web sur `http://localhost:7860` avec un volume persistant `~/.backpropagate`.

## La place de Backpropagate

Il existe plusieurs bonnes bibliothèques pour l'ajustement fin des LLM. Chacune est excellente pour des choses différentes :

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — si vous aimez les configurations YAML et que vous souhaitez avoir une communauté de recettes à suivre.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — si vous voulez une interface utilisateur web et une prise en charge intégrée de DPO/PPO/RLHF.
- **[Unsloth](https://github.com/unslothai/unsloth)** — si vous avez besoin de la formation la plus rapide possible et que vous utilisez un modèle pris en charge.
- **[torchtune](https://github.com/pytorch/torchtune)** — si vous voulez les recettes PyTorch natives de Meta que vous pouvez modifier.

Backpropagate est l'option manquante : **une API Python en 3 lignes pour les utilisateurs individuels sur un seul GPU grand public, qui souhaitent entraîner un adaptateur et le distribuer.** Pas de YAML, pas d'interface utilisateur graphique, pas de DPO/PPO, pas de configuration multi-nœuds. Juste la boucle dont tout le monde a réellement besoin et l'étape d'exportation qui est un obstacle.

Si vous avez essayé l'une des bibliothèques ci-dessus et que vous avez été rebuté par la "cérémonie" des fichiers de configuration, ou si vous avez rencontré un problème de compatibilité avec un modèle, ou si vous vouliez des paramètres par défaut adaptés à Windows, Backpropagate est fait pour vous.

## Ce que vous pouvez ajuster finement sur un GPU grand public de 16 Go

Voici les limites pratiques sur une carte de 16 Go (RTX 4080 / 5080 / 4070 Ti Super) :

| Modèle | Méthode | Statut |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | Confortable. Longueur de séquence complète, avec de la marge. |
| Phi-4-mini-3.8B / Qwen-3.5-4B / SmolLM3-3B (limite de 3 milliards de paramètres) | `mode="full"` (ajustement fin complet) | v1.4 — Utilisez l'option `--mode=full` avec `backprop train` ou `Trainer(..., mode="full")`. Le checkpointing des gradients et l'optimiseur Adam en 8 bits avec pagination maintiennent la mémoire d'activation à sqrt(L). |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | Standard. Environ 7-8 Go. Les paramètres par défaut de Backpropagate. |
| Llama-3 13B | QLoRA + échantillonnage | Juste limite, mais fonctionne. Utilisez des séquences plus courtes. |
| Mixtral 8x7B (47 milliards de paramètres au total) | AQLM 2 bits + LoRA | Prévu pour v1.5 — Consultez le document V1_5_BRIEF lorsqu'il sera publié. |

La quantification AQLM 2 bits (`quant_method="aqlm"`, option expérimentale pour Mixtral-8x7B sur 16 Go) était prévue pour v1.4 et est maintenant prévue pour v1.5. La bibliothèque `aqlm` est mature ; la feuille de route de v1.4 a donné la priorité au support de l'ajustement fin complet pour les modèles ≤ 3 milliards de paramètres (`mode="full"`) plutôt qu'à l'ajout d'un nouveau backend de quantification. Consultez le document V1_5_BRIEF lorsqu'il sera publié pour connaître le plan de mise en œuvre de v1.5.

Pour les modèles de 3 milliards de paramètres et moins, l'ajustement fin complet (et non pas seulement LoRA) est possible sur 16 Go et est désormais disponible dans v1.4 avec l'option `mode="full"`. Utilisez `Trainer(..., mode="full")` ou `backprop train --mode=full --model phi-4-mini-3.8b` pour l'activer. Un mécanisme de protection empêche l'utilisation de ce mode pour les modèles > 3 milliards de paramètres, en proposant LoRA et les configurations pré-établies pour les modèles < 3 milliards de paramètres comme solutions de contournement. Consultez [la page complète sur l'ajustement fin](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/) pour les détails de configuration et la comparaison de qualité entre Biderman 2024 et Thinking Machines 2025. Pour les modèles de 7 milliards de paramètres et plus, l'ajustement fin complet nécessite une GPU de 24 Go ou plus ; envisagez une location de GPU A100 dans le cloud, ou utilisez LoRA, qui, selon des recherches récentes, offre une qualité équivalente à l'ajustement fin complet pour la plupart des tâches après l'entraînement (voir [la section "Ce que Backpropagate n'est pas"](#what-backpropagate-is-not) pour les références).

## Ce que Backpropagate N'EST PAS

Si votre cas d'utilisation correspond à l'un des éléments suivants, vous obtiendrez de meilleurs résultats avec une autre bibliothèque. Backpropagate n'est pas le choix idéal, et essayer de le faire fonctionner coûterait plus cher que d'utiliser l'outil approprié. Lire cette section avant de commencer vous évitera de devoir installer et désinstaller à plusieurs reprises :

- **Ajustement fin complet des paramètres des modèles de 7 milliards de paramètres et plus** — Backpropagate utilise LoRA / QLoRA, qui entraîne un petit adaptateur au lieu de mettre à jour chaque poids. Pour les modèles de 7 milliards de paramètres et plus, l'ajustement fin complet nécessite 24 Go de mémoire GPU et ne peut pas être exécuté sur une carte grand public de 16 Go. Pour les modèles de 3 milliards de paramètres et moins, l'ajustement fin complet est possible sur 16 Go et est disponible dans v1.4 avec l'option `mode="full"` (utilisez `Trainer(..., mode="full")` ou `--mode=full` dans l'interface de ligne de commande ; un mécanisme de protection empêche l'utilisation de ce mode pour les modèles > 3 milliards de paramètres et propose LoRA et les configurations pré-établies pour les modèles < 3 milliards de paramètres comme solutions de contournement). Pour résumer : des recherches récentes ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) montrent que LoRA, avec une configuration appropriée, offre une qualité équivalente à l'ajustement fin complet pour la plupart des tâches après l'entraînement (suivi d'instructions, adaptation à un domaine spécifique, personnalisation/style) avec seulement 67 % de la puissance de calcul. Ainsi, pour la plupart des tâches que les utilisateurs souhaitent effectuer, vous ne perdez rien à utiliser LoRA. L'option `mode="full"` est disponible pour les cas où vous avez constaté un écart de qualité et que vous avez décidé de consacrer une puissance de calcul supplémentaire. Si vous avez réellement besoin d'un ajustement fin complet d'un modèle de 7 milliards de paramètres ou plus, utilisez directement le module `transformers.Trainer` de HuggingFace sur une carte de 24 Go ou plus.
- **DPO / PPO / GRPO / ajustement des préférences** — Backpropagate ne prend en charge que l'ajustement fin supervisé en une seule étape. Pour l'apprentissage par préférences, utilisez directement TRL ou LLaMA-Factory.
- **Entraînement multi-nœuds** — Fonctionne uniquement avec une seule GPU sur une seule machine. L'utilisation de plusieurs GPU sur une seule machine est possible (via `accelerate launch`) mais n'est pas officiellement prise en charge.
- **Entraînement sur macOS** — Apple Silicon ne dispose pas de CUDA, l'entraînement doit donc être effectué sur une machine Linux ou Windows avec une GPU NVIDIA. Vous pouvez toujours exécuter le modèle entraîné sur un Mac via Ollama.
- **Tout ce qui se trouve en dehors des familles de modèles testées** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. D'autres modèles peuvent fonctionner, mais ne sont pas inclus dans les tests automatisés.

Si vous avez besoin de ces fonctionnalités, utilisez l'une des bibliothèques mentionnées ci-dessus. Elles sont plus adaptées à cela.

## Ce que Backpropagate vous offre :

Quatre éléments, dans une seule installation :

**1. Une API simple en 3 lignes qui fonctionne sans fichier de configuration.**
Le code en haut de ce fichier README est exécutable de bout en bout. Pas de `accelerate config`, pas de YAML, pas de surcharge Hydra. Il suffit de `Trainer(model).train(data)` et vous avez un modèle affiné.

**2. Une compatibilité Windows qui fonctionne réellement.**
La plupart des bibliothèques de machine learning traitent Windows comme une option secondaire. Backpropagate est testé de manière approfondie sur Windows + RTX 5080. La bibliothèque gère les particularités de l'environnement d'exécution : elle sait comment prétraiter vos données pour éviter les plantages liés au multiprocessing de Windows, elle désactive automatiquement xformers sur les cartes RTX 40/50 où cela entraînerait des problèmes, et elle sélectionne les paramètres du chargeur de données qui évitent les erreurs. Vous n'avez pas besoin de connaître ces détails. Tout simplement, cela fonctionne.

**3. Conçu pour les exécutions automatisées.**
L'entraînement prend des heures. Vous ne voulez pas devoir le surveiller en permanence. Backpropagate est conçu pour fonctionner en arrière-plan :

- Si vous manquez de mémoire GPU, il réduit automatiquement la taille du lot et retente, jusqu'à trois fois. Pas de réglages manuels nécessaires.
- Si votre GPU devient trop chaud, il met en pause jusqu'à ce que la température redescende, puis reprend.
- Chaque point de contrôle est enregistré de manière atomique : si votre ordinateur portable plante pendant la sauvegarde, le dernier point de contrôle valide est toujours conservé.
- Chaque exécution d'entraînement reçoit un identifiant unique qui est ajouté à chaque ligne de journal, à chaque point de contrôle et à chaque entrée Weights & Biases. Si quelque chose ne va pas, cet identifiant permet à un développeur de corréler toutes les informations.
- Les erreurs sont accompagnées de codes stables (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, etc.), ce qui vous permet de rechercher dans vos journaux et dans le [guide de dépannage](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) pour trouver la solution. Les erreurs spécifiques à CUDA ont une [page de dépannage dédiée](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

**4. Une seule commande pour passer d'un adaptateur entraîné à `ollama run`.**
De nombreuses bibliothèques entraînent un modèle. Peu d'entre elles vous facilitent la tâche lorsque vous souhaitez réellement l'utiliser. Backpropagate exporte vers GGUF (le format utilisé par Ollama) et enregistre un modèle Ollama en une seule commande. Vous passez de "entraînement terminé" à "je peux discuter avec mon modèle affiné" en environ 30 secondes.

## Démarrage rapide

Le dépôt contient un petit ensemble de données d'exemple afin que le code du début de ce fichier README puisse être exécuté sur une installation propre :

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

Cela entraîne un adaptateur Qwen 2.5 7B sur 5 courtes conversations au format ShareGPT, puis exporte le résultat au format GGUF. Pour vos propres données, formatez votre fichier JSONL avec un exemple par ligne :

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Les formats Alpaca (`instruction` / `output`), OpenAI chat (`messages`) et texte brut fonctionnent également. Backpropagate détecte automatiquement le format.

Pour des flux de travail plus complets (affinage et publication sur le Hub Hugging Face, reprise après une erreur de mémoire, exécution multiple de SLAO sur une longue période, etc.), consultez la [page des recettes du manuel](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interface utilisateur web (facultatif)

Si vous préférez cliquer plutôt que taper du code Python, installez le module d'interface utilisateur et lancez-le :

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Une interface web locale s'ouvre à l'adresse `http://localhost:7862`, où vous pouvez spécifier un ensemble de données, choisir un modèle, effectuer un entraînement et exporter le résultat. L'interface utilisateur est par défaut accessible uniquement localement. Pour la rendre accessible à d'autres appareils, consultez la section [Interface utilisateur web](#web-ui) ci-dessous pour connaître le contrat de sécurité `--share` + `--auth`.

## Entraînement en plusieurs étapes

Si vous souhaitez effectuer un affinage incrémental sur plusieurs ensembles de données (par exemple, si vous recevez de nouvelles données d'entraînement chaque semaine et que vous souhaitez les ajouter sans oublier ce que vous avez appris auparavant), le mode `multi_run` de Backpropagate est fait pour vous :

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

Cela effectue cinq passes d'entraînement, en fusionnant l'adaptateur entre chaque passe de manière à préserver les connaissances antérieures tout en intégrant de nouveaux exemples. Cette technique est basée sur des recherches récentes en matière d'apprentissage continu. Consultez la section [Références](#references) en bas de ce fichier README.

La version en ligne de commande :

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Reprise à partir d'un point de contrôle

Une session d'entraînement en cinq étapes qui se bloque à la quatrième étape peut être reprise. Chaque session d'entraînement en plusieurs étapes enregistre son ID de session dans l'historique et le manifeste du point de contrôle, ce qui vous permet de reprendre là où vous vous étiez arrêté en une seule commande :

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

Par défaut, `backprop multi-run` (sans `--resume`) détecte automatiquement une session en cours dans le même répertoire de sortie et la reprend. Pour forcer un démarrage propre, spécifiez un nouveau répertoire de sortie.

## Historique de l'entraînement

Chaque invocation de `backprop train` et `backprop multi-run` enregistre une ligne dans `<output>/run_history.json`, contenant des informations sur le modèle utilisé, l'ensemble de données, les hyperparamètres, l'état, la perte finale et l'historique des pertes. Vous pouvez afficher et examiner les sessions précédentes :

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Suivi des expériences

Backpropagate détecte automatiquement les outils de suivi d'expériences installés (Weights & Biases, TensorBoard, MLflow) et les configure. Si `wandb` est installé et que vous êtes connecté, chaque session enregistre automatiquement les données sur W&B avec un nom de session correspondant à l'ID de session enregistré sur le disque. Vous pouvez ainsi effectuer une recherche sur W&B, vos journaux et `run_history.json` en utilisant un seul identifiant.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Pour ne pas utiliser ces outils, utilisez `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` ou `Trainer(report_to="none")`.

## Interface web

L'interface web Reflex est facultative. Pour l'installer, utilisez `pipx install "backpropagate[ui]"` et lancez-la :

```bash
backprop ui --port 7862
```

L'interface utilisateur s'exécute localement à l'adresse `http://localhost:7862`. Pour la rendre accessible à d'autres appareils (autres personnes sur votre réseau, une URL publique, etc.), vous devez utiliser les options `--share` (ou `--host`) en combinaison avec `--auth` :

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` sans `--auth` génère une erreur. La raison est que `--share` publie une URL que toute personne sur Internet peut atteindre, et sans authentification, cela signifie que toute personne peut contrôler votre pipeline d'entraînement et lire votre jeton HuggingFace. Il n'y a pas d'option pour désactiver cette fonctionnalité. Si vous ne souhaitez pas définir de crédentielles, utilisez plutôt le transfert de port SSH :

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

Consultez [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) pour connaître le modèle de menace complet.

Les opérations d'écriture sur le système de fichiers via l'interface utilisateur sont limitées à un seul répertoire :

- Par défaut : `~/.backpropagate/ui-outputs`
- Pour modifier : définissez `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- La modification est validée par une liste de contrôle — les chemins système ou d'informations d'identification (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) sont refusés.

## Notes sur la plateforme

**Prérequis :** Python 3.10+ · GPU CUDA (8 Go+ de VRAM) · PyTorch 2.0+

Python 3.10 atteindra la fin de son cycle de vie en octobre 2026, et Backpropagate prévoit de supprimer le support de Python 3.10 dans la version 1.4. Pour les nouvelles installations, privilégiez Python 3.11 ou 3.12 — Python 3.11 est la version la plus testée.

Backpropagate gère les particularités de l'exécution sur différentes plateformes, mais il ne peut pas résoudre les problèmes d'installation. Les deux problèmes les plus courants sont :

- **Mauvais fichier "wheel" CUDA.** PyTorch est publié avec une version binaire par version de CUDA. Si vous choisissez la mauvaise version, vous obtenez silencieusement une version de PyTorch qui n'utilise que le processeur, et l'entraînement est extrêmement lent. Utilisez le sélecteur de "wheel" à l'adresse <https://pytorch.org/get-started/locally/> pour votre pilote. Exécutez `nvidia-smi` pour voir votre version de pilote / CUDA.
- **Windows + exportation GGUF.** L'option `[export]` construit `llama-cpp-python` à partir du code source, ce qui nécessite les outils de construction Visual Studio (composant C++) et CMake.

**macOS :** L'entraînement sur GPU n'est pas pris en charge (pas de CUDA). Vous pouvez exécuter l'adaptateur entraîné sur un Mac via Ollama, mais `trainer.train()` génère une erreur `DEP_GPU_NOT_AVAILABLE`. Utilisez une machine Linux ou Windows avec CUDA pour l'entraînement lui-même.

Consultez la [page du manuel de dépannage](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) pour obtenir un guide complet de résolution des problèmes d'installation, et la [page de dépannage CUDA dédiée](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) pour les problèmes liés aux pilotes / VRAM / xformers / bf16-vs-fp16.

## Interface en ligne de commande

Chaque API Python a un équivalent en ligne de commande (CLI) :

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

Référence complète disponible sur la [page de référence du CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), ou `backprop <sous-commande> --help`.

## Configuration

Chaque paramètre peut être modifié via une variable d'environnement en utilisant le préfixe `BACKPROPAGATE_` :

| Variable | Valeur par défaut | Notes |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Forcer les journaux JSON ou console |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modèle par défaut |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Taux d'apprentissage |
| `BACKPROPAGATE_LORA__R` | `256` | Rang LoRA (valeur par défaut de la version 1.3 ; utilisez `--lora-preset=fast` pour la valeur par défaut de la version 1.2.x, qui est 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Espace de stockage du système de fichiers de l'interface utilisateur |

Les clés imbriquées utilisent un double tiret bas (`MODEL__NAME`, et non `MODEL_NAME`). La référence complète est disponible sur la [page des variables d'environnement du manuel](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Modèles préconfigurés

| Modèle | VRAM | Licence | Notes |
|---|---|---|---|
| Qwen-3.5-4B | ~8 Go | Apache 2.0 | Valeur par défaut recommandée pour les modèles de moins de 5 milliards de paramètres. Meilleure qualité pour cette taille. |
| Phi-4-mini-3.8B | ~8 Go | MIT | Excellent pour le raisonnement, les mathématiques et la programmation. Licence très permissive. |
| SmolLM3-3B | ~6 Go | Apache 2.0 | Recette entièrement ouverte. Contexte natif de 64 Ko. |
| Qwen 2.5 7B | ~12 Go | Apache 2.0 | Valeur par défaut existante. Meilleure qualité parmi les anciens modèles 7B. |
| Qwen 2.5 3B | ~8 Go | Qwen-Research | ⚠ Licence de recherche — consultez les conditions de licence de Qwen avant toute utilisation commerciale. |
| Llama 3.2 3B | ~8 Go | Llama Community | Alternative intéressante à Qwen 3B avec des conditions d'utilisation permissives. |
| Llama 3.2 1B | ~6 Go | Llama Community | Idéal pour les premières expérimentations sur de petites cartes. |
| Mistral 7B | ~12 Go | Apache 2.0 | Comparable à Qwen 7B, mais avec un modèle de conversation différent. |

D'autres modèles peuvent fonctionner, mais seuls ces huit sont intégrés dans les tests automatisés (CI). Utilisez `--lora-preset=quality` (par défaut) pour obtenir un rang de 256 / des cibles "all-linear" selon Biderman 2024 + Thinking Machines 2025, ou `--lora-preset=fast` pour obtenir l'empreinte de la version 1.2.x avec un rang de 16 / des cibles q+v si vous avez besoin de cette empreinte.

## Dépannage

Un bref index des erreurs les plus courantes lors de la première exécution. L'index inversé complet est disponible sur la [page du manuel de dépannage](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Pour une analyse approfondie des pilotes / VRAM / précision mixte, consultez la [page de dépannage CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Symptôme | Code d'erreur | Solution |
|---|---|---|
| La mémoire de la GPU est épuisée pendant l'entraînement. | `RUNTIME_GPU_OOM` | Automatic — Backpropagate réduit la taille du lot de moitié et tente jusqu'à 3 fois. Pour désactiver cette fonctionnalité : `Trainer(oom_recovery=False)`. Pour forcer une taille plus petite : `--batch-size 1`. |
| HuggingFace renvoie 401 / "modèle introuvable" | `DEP_MODEL_LOAD_FAILED` | Utilisez `huggingface-cli login` et réessayez. En cas de faute de frappe, copiez l'identifiant exact depuis <https://huggingface.co/models>. |
| `register_with_ollama` : connexion refusée. | `DEP_OLLAMA_REGISTRATION_FAILED` | Démarrez le démon : `ollama serve`. Installez depuis <https://ollama.com>. Opération pouvant être répétée. |
| Le disque est plein lors de la sauvegarde du point de contrôle. | `STATE_CHECKPOINT_INVALID` | Les écritures atomiques laissent un répertoire `.partial` en cas de plantage ; il est sûr de le supprimer. Le point de contrôle précédent est intact. |
| L'entraînement est interrompu en raison de la surchauffe du GPU. | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automatic — Backpropagate met l'entraînement en pause lorsque la température atteint un seuil et le reprend lorsque le GPU refroidit. Améliorez la circulation de l'air si cela se produit fréquemment. |
| `backprop ui --share` est refusé. | `INPUT_AUTH_REQUIRED` | Utilisez l'option `--auth user:password`, ou utilisez plutôt le transfert de port SSH (voir [Interface Web](#web-ui)). |
| L'export GGUF a échoué lors de la première tentative. | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; sous Windows, vous avez également besoin des outils de construction Visual C++ et de CMake. |

## Signaler des bogues

Lorsqu'une erreur se produit, Backpropagate affiche une ligne au démarrage, comme `run_started run_id=<uuid>`, et associe cet ID à chaque ligne de journal, à chaque point de contrôle et à chaque entrée Weights & Biases. **Incluez l'ID de la session (`run_id`) dans tout rapport de bug** ; cela permet à un développeur de corréler tous les éléments pour cette exécution spécifique.

Un bon rapport de bogue comprend :

1. **L'identifiant de session (`run_id`)** : L'UUID affiché au démarrage. Un UUID permet à un responsable de corréler chaque ligne de journal, chaque point de contrôle et chaque entrée Weights & Biases pour cette session spécifique.
2. **Le code d'erreur** : La ligne `[NOM_DU_CODE]: message` dans la sortie d'erreur standard (stderr). Consultez [la liste des codes d'erreur](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) pour connaître les codes stables.
3. **La trace de la pile masquée.** La sortie d'erreur standard est automatiquement masquée en mode non verbeux (les jetons d'authentification, les clés commençant par `sk-*`, `hf_*`, les clés AWS, les paires `password=` / `token=` / `api_key=` sont supprimées) ; vous pouvez la copier sans risque. Pour afficher la trace de la pile complète et non masquée, relancez le programme avec `BACKPROPAGATE_DEBUG=1` (ou `--verbose`) ; examinez-la attentivement avant de la publier.
4. **La sortie de la commande `backprop info`**. Une seule commande affiche les informations sur Python, PyTorch, CUDA, le modèle GPU, la VRAM, le système d'exploitation et les modules complémentaires installés : toutes les informations dont un responsable a besoin pour identifier une régression spécifique à une plateforme.

Le [modèle de rapport de bug](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) vous demande explicitement de fournir ces informations, ce qui accélère le processus de triage. Les questions, les idées ou les discussions sur le type de comportement attendu doivent être posées dans [les discussions GitHub](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Les problèmes de sécurité doivent être signalés de manière privée via le formulaire [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) ; consultez le fichier [SECURITY.md](SECURITY.md) pour connaître la politique et les délais de réponse.

## Confidentialité

Toute l'exécution se fait localement sur votre GPU. Backpropagate ne fait aucune requête réseau, sauf pour télécharger les modèles depuis HuggingFace (ce que vous initiez). Pas de télémétrie, pas de dépendance au cloud.

## Références

Les paramètres par défaut de Backpropagate et son mode d'entraînement multi-sessions sont basés sur des recherches récentes. Si vous souhaitez en savoir plus sur les techniques utilisées :

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — l'article fondateur qui présente LoRA, la technique utilisée par Backpropagate pour entraîner efficacement les adaptateurs.
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — des preuves empiriques que LoRA avec un rang de 256 et des cibles linéaires atteint une qualité équivalente à un réglage fin complet pour la plupart des tâches après l'entraînement, tout en utilisant 67 % moins de ressources de calcul. Cela influence la configuration LoRA par défaut de Backpropagate (v1.3).
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — une suite pratique qui identifie la correction du taux d'apprentissage 10 fois par rapport au réglage fin complet nécessaire pour les rangs LoRA élevés.
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la caractérisation originale de la raison pour laquelle les réseaux neuronaux "oublient" les entraînements précédents lors du réglage fin sur de nouvelles données (EWC — Elastic Weight Consolidation).
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, une approche antérieure utilisant LoRA pour l'apprentissage continu en contraignant les nouveaux adaptateurs à des sous-espaces orthogonaux.
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — une technique fondamentale pour fusionner plusieurs modèles réglés finement sans interférence.
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — l'algorithme spécifique implémenté par le module de fusion multi-sessions de Backpropagate. Il s'agit d'une prépublication de décembre 2025 ; Backpropagate est le premier utilisateur connu de cet article.

## Licence

MIT — voir [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
