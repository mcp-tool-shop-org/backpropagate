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
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**Affinage de LLM sans interface graphique en 3 lignes. Paramètres par défaut intelligents, gestion de la taille des lots en fonction de la VRAM, apprentissage continu SLAO sur plusieurs exécutions, et exportation GGUF en un clic pour Ollama.**

*SLAO est l'apprentissage continu LoRA via une fusion asymétrique, une technique de fusion entre les exécutions qui empêche l'oubli catastrophique lors de sessions d'affinage prolongées ([article](https://arxiv.org/abs/2512.23017)).*

*Entraînez des LLM en 3 lignes de code. Exportez-les vers Ollama avec une ligne supplémentaire.*

## Démarrage rapide

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("examples/quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

Le dépôt contient un petit fichier `examples/quickstart.jsonl` (5 exemples au format ShareGPT) afin que le code ci-dessus puisse être exécuté de bout en bout sur une installation propre. Pour votre propre entraînement, consultez la section [Format des données](#dataset-format) ci-dessous.

### Option sans code : Interface web

Préférez-vous une interface graphique plutôt qu'un REPL Python ? Installez le module correspondant et exécutez la commande suivante :

```bash
pip install backpropagate[standard]
backprop ui --port 7862
```

L'interface Reflex (Radix UI) vous permet de spécifier un fichier JSONL, de choisir un modèle, de l'entraîner et de l'exporter, sans nécessiter de code Python. L'interface est conçue pour fonctionner localement ; pour une exposition sur Internet, consultez la section [Interface web](#web-ui) ci-dessous pour connaître le contrat de sécurité `--share` + `--auth` et les options de tunnel prises en charge (Cloudflare Tunnel, ngrok).

## Format des données

Votre fichier d'entraînement au format JSONL doit contenir un exemple par ligne. Le format le plus simple est le chat ShareGPT :

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Les formats Alpaca (`instruction`/`output`), OpenAI chat (`messages`) et texte brut sont également pris en charge. Consultez le fichier `examples/quickstart.jsonl` pour un point de départ que vous pouvez copier.

## Pourquoi propager le gradient ?

| Problème | Solution |
|---------|----------|
| L'affinage est complexe | 3 lignes : chargement, entraînement, sauvegarde |
| Windows est un cauchemar | Prise en charge complète de Windows |
| La gestion de la VRAM est difficile | Ajustement automatique de la taille des lots, surveillance du GPU |
| L'exportation des modèles est déroutante | Exportation GGUF en un clic + enregistrement automatique avec Ollama |
| Les longues exécutions entraînent l'oubli | Entraînement SLAO sur plusieurs exécutions |

## Fonctionnalités clés

- **Conçu sans interface graphique :** Conçu pour les pipelines CI/CD, les flux de travail automatisés et l'exécution programmatique.
- **Paramètres par défaut intelligents :** Configure automatiquement les hyperparamètres optimaux en fonction de votre matériel et de votre ensemble de données.
- **Entraînement SLAO sur plusieurs exécutions :** Stratégies d'entraînement avancées pour éviter l'oubli catastrophique lors de longues exécutions.
- **Prise en charge complète de Windows :** Testé et optimisé pour les environnements Windows, évitant les pièges courants de PyTorch/CUDA.
- **Exportation transparente :** Exportation en un clic au format GGUF et enregistrement automatique avec Ollama.
- **Architecture modulaire :** Installez uniquement les dépendances dont vous avez besoin (par exemple, `[unsloth]`, `[ui]`, `[export]`).

## Installation

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Reflex (Radix UI) web interface
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Modules complémentaires | Description | Dépendances |
|-------|-------------|--------------|
| `unsloth` | Entraînement 2 fois plus rapide, 50 % de VRAM en moins | unsloth |
| `ui` | Interface web Reflex (Radix UI) | reflex>=0.9.2, fastapi>=0.115 |
| `validation` | Validation de configuration Pydantic | pydantic, pydantic-settings |
| `export` | Exportation GGUF pour Ollama | llama-cpp-python |
| `monitoring` | WandB + surveillance du système (intégré au trainer depuis la version 1.1.0) | wandb, psutil |
| `logging` | Journalisation structurée | structlog |
| `security` | Authentification JWT + génération de jetons | PyJWT, cryptography |
| `production` | unsloth + ui + validation + journalisation + sécurité | (ensemble) |

**Prérequis :** Python 3.10+ · GPU CUDA (8 Go+ de VRAM) · PyTorch 2.0+

### Prérequis de la plateforme

Backpropagate gère les particularités liées à l'exécution (multiprocessing, xformers sur RTX 40/50, workers du dataloader sur Windows). Il **ne** gère pas les problèmes liés à l'installation propres à chaque plateforme ; corrigez d'abord ces problèmes :

- **Version du kit de développement CUDA.** PyTorch est publié en fonction de la version de CUDA ; choisir la mauvaise version installe silencieusement une version de torch qui n'utilise que le CPU. Utilisez le sélecteur à l'adresse <https://pytorch.org/get-started/locally/> pour obtenir la commande `pip install torch ...` exacte correspondant à votre pilote. Exécutez `nvidia-smi` pour voir la version de votre pilote/CUDA.
- **Windows.** Visual Studio Build Tools (C++) et CMake sont requis pour l'extension `[export]` (construction de `llama-cpp-python` à partir du code source). La version de `bitsandbytes` est maintenant disponible nativement pour Windows (>= 0.43) ; les anciens guides mentionnant `bitsandbytes-windows` sont obsolètes.
- **macOS.** L'entraînement sur GPU **n'est pas pris en charge** ; il n'y a pas de CUDA. Vous pouvez installer Backpropagate pour exécuter l' *inférence* sur un fichier GGUF exporté via Ollama, mais `trainer.train()` génère une erreur `DEP_GPU_NOT_AVAILABLE`. Utilisez une machine avec CUDA pour l'entraînement.
- **Linux.** La plupart des distributions fonctionnent sans problème. Si vous utilisez la version binaire PyPI, notez que la version Linux utilise torch uniquement pour le CPU (afin de respecter la limite de 2 Go des fichiers joints sur GitHub) ; installez d'abord la version CUDA correspondante à partir de pytorch.org.

Pour le dépannage de l'installation, consultez [la page du guide de dépannage](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/).

## Configuration

Tous les paramètres peuvent être remplacés à l'aide de variables d'environnement, en utilisant le préfixe `BACKPROPAGATE_` (par exemple, `BACKPROPAGATE_LOG_LEVEL=debug`). Un fichier `.env` à la racine du projet est chargé automatiquement lorsque l'extension `[validation]` est installée.

Paramètres courants (voir [la référence complète des variables d'environnement](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/) pour tout) :

| Variable | Valeur par défaut | Notes |
|----------|---------|-------|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Force les journaux en JSON (`true`) ou sur la console (`false`) |
| `BACKPROPAGATE_LOG_FILE` | non défini | Chemin vers le répertoire où les journaux sont enregistrés |
| `BACKPROPAGATE_DEFER_FEATURE_DETECTION` | non défini | Ignore la détection des dépendances optionnelles au démarrage pour un démarrage plus rapide de l'interface en ligne de commande |
| `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE` | `true` | Lorsque `true`, refuse `backprop ui --share` sans l'option `--auth` |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Répertoire de base pour toutes les opérations d'écriture sur le système de fichiers de l'interface utilisateur ; liste de contrôle d'accès validée |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modèle par défaut |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Taux d'apprentissage |
| `BACKPROPAGATE_LORA__R` | `16` | Rang LoRA |

Les clés imbriquées utilisent un double tiret-bas comme délimiteur (convention Pydantic `env_nested_delimiter`).

## Utilisation

### Entraînement de base

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

`Qwen/Qwen2.5-7B-Instruct` est la valeur par défaut standard. Lorsque la fonction `Trainer()` est appelée sans argument de modèle, c'est cette valeur qui est utilisée (voir [`config.py`](backpropagate/config.py) `ModelConfig.name`). Les exemples précédents utilisaient la version quantifiée `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`; nous avons modifié la valeur par défaut pour utiliser les poids officiels de Qwen afin d'améliorer la fiabilité ([CHANGELOG v1.1.0](CHANGELOG.md#110---2026-05-21)). Les deux modèles fonctionnent.

### Entraînement SLAO multi-exécution

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

SLAO (Single LoRA Continual Learning via Asymmetric Merging) implémente l'article [Merge before Forget](https://arxiv.org/abs/2512.23017) : initialisation orthogonale de la matrice A via la décomposition QR, gestion asymétrique de A/B et mise à l'échelle temporelle `λ(i) = 1/√i`. Le paramètre de l'interface en ligne de commande est `--samples` (le champ sous-jacent est `samples_per_run`).

### Exportation vers Ollama

```python
# Export to GGUF
result = trainer.export("gguf", quantization="q4_k_m")

# Register with Ollama separately
from backpropagate import register_with_ollama
register_with_ollama(result.path, "my-finetuned-model")
# ollama run my-finetuned-model
```

### Interface en ligne de commande

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

Consultez [la référence de l'interface en ligne de commande](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) pour chaque sous-commande et chaque paramètre, ou exécutez `backprop <sous-commande> --help`.

### Reprise à partir d'un point de contrôle (v1.1.0)

Un cycle d'entraînement interrompu à la quatrième étape peut désormais être repris. Chaque session d'entraînement enregistre son identifiant (`run_id`) à la fois dans le fichier `run_history.json` et dans le fichier de manifeste des points de contrôle sur le disque, ce qui permet de reprendre l'entraînement en une seule commande.

```bash
backprop resume <run-id>                       # picks up the in-progress session
backprop multi-run --data ... --resume <run-id> # explicit form
backprop train --data ... --resume <run-id>    # single-run resume (continues run_id)
```

Par défaut, la commande `backprop multi-run` (sans l'option `--resume`) détecte automatiquement une session en cours pour le même répertoire de sortie et la reprend. Pour forcer une nouvelle session, utilisez `resume_from="off"` (API Python) ou omettez l'option `--resume` et démarrez dans un nouveau répertoire de sortie.

Lorsqu'une session d'entraînement est reprise, le dernier point de contrôle associé à cet `run_id` est chargé dans le modèle, l'état de fusion SLAO est restauré à partir du répertoire `slao/` situé à côté du point de contrôle, et le cycle d'entraînement reprend à partir de `last_completed_run + 1`. L'état de la ligne d'historique passe à `running`, ce qui permet à la commande `backprop list-runs --status running` d'afficher la session en cours.

### Suivi des expériences (v1.1.0)

Le module `Trainer` détecte automatiquement les outils de suivi d'expériences installés (`wandb`, `tensorboard`, `mlflow`) et les intègre aux paramètres d'entraînement de `transformers`. La valeur par défaut `report_to="auto"` utilise tout ce qui est importable.

```bash
pip install backpropagate[monitoring]  # installs wandb + psutil
wandb login                            # one-time
backprop train --data my_data.jsonl    # W&B run gets the same run_id prefix as the on-disk history
```

Pour désactiver explicitement le suivi, utilisez `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` ou `Trainer(report_to="none")`. Pour utiliser MLflow, installez le package `mlflow` avec `pip install mlflow`; pour TensorBoard, installez `pip install tensorboard`. Le nom de l'exécution W&B est `backprop-<run_id_prefix>`, ce qui permet aux utilisateurs de rechercher des informations à l'aide de ce même identifiant dans W&B, les journaux et le fichier `run_history.json`.

### Historique de l'entraînement

Chaque invocation de `backprop train` et `backprop multi-run` enregistre une ligne dans le fichier `<output>/run_history.json` contenant l'identifiant de l'exécution, le modèle, le jeu de données, les hyperparamètres, l'état, la perte finale, l'historique des pertes et, pour les exécutions multiples, le calendrier de fusion SLAO. Pour afficher les exécutions récentes, utilisez la commande appropriée.

```bash
backprop list-runs                         # most recent 20 runs, all statuses
backprop list-runs --status failed         # filter
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial run_id ok)
```

L'historique des exécutions est conservé entre les processus. L'onglet "Runs" de l'interface web est une vue en mémoire distincte ; l'historique enregistré sur le disque est la source de vérité pour les commandes `list-runs` / `show-run` / `resume`.

### Interface web

Lancez l'interface Reflex localement :

```bash
backprop ui --port 7862
```

Pour rendre l'URL accessible via Internet, vous devez combiner les options `--share` et `--auth` :

```bash
backprop ui --share --auth alice:hunter2
```

La commande `backprop ui --share` sans l'option `--auth` renvoie un code d'erreur `1` et un message d'erreur structuré `[INPUT_AUTH_REQUIRED]`. La raison est que l'option `--share` publie une URL `*.gradio.live` que toute personne sur Internet peut consulter, et sans authentification, cela signifie que toute personne peut contrôler votre pipeline d'entraînement.

Pour désactiver explicitement cette fonctionnalité (par exemple, dans un environnement de développement interne), définissez la variable d'environnement `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false`. Un avertissement important s'affiche à chaque lancement, et il existe une période de grâce de 5 secondes avant que l'interface utilisateur non authentifiée ne s'active, ce qui vous permet d'utiliser `Ctrl-C` si quelque chose ne vous semble pas correct.

Les opérations d'écriture sur le système de fichiers via l'interface utilisateur sont limitées à un seul répertoire :

- Par défaut : `~/.backpropagate/ui-outputs`
- Pour modifier : `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- Cette modification est **validée par une liste de contrôle** : les chemins système et d'informations d'identification (`/etc`, `/var`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) sont refusés avec le message `[UI_OUTPUT_DIR_FORBIDDEN]`.

## Prise en charge de Windows

Backpropagate est conçu pour fonctionner sur Windows sans configuration particulière :

- Pré-tokenisation pour éviter les plantages liés au multiprocessing.
- Désactivation automatique de xformers pour les séries RTX 40/50.
- Paramètres de dataloader sécurisés.
- Testé sur RTX 5080 (16 Go de VRAM).

## Modèles préconfigurés

| Modèle | VRAM | Vitesse | Qualité |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12 Go | Moyenne | Meilleure |
| Qwen 2.5 3B | ~8 Go | Rapide | Bonne |
| Llama 3.2 3B | ~8 Go | Rapide | Bonne |
| Llama 3.2 1B | ~6 Go | Très rapide | Basique |
| Mistral 7B | ~12 Go | Moyenne | Bonne |

## Architecture

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

L'implémentation Gradio de la version 1.0 (`ui_gradio_legacy.py` + `theme_gradio_legacy.py`) a été conservée jusqu'à la version 1.1.x à titre de référence et a été supprimée dans la version 1.2.0.

## Dépannage

Un bref index des erreurs les plus courantes rencontrées au démarrage. L'index complet se trouve sur la page du manuel de dépannage : [https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/); chaque code ci-dessous est documenté dans la section [codes d'erreur](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/).

| Symptôme | Code | Solution |
|---------|------|-----|
| La mémoire de la GPU est épuisée pendant l'entraînement. | `RUNTIME_GPU_OOM` | La fonctionnalité de récupération automatique en cas de manque de mémoire (OOM) (B-002) réduit automatiquement la taille du lot jusqu'à 3 fois. Pour la désactiver : `Trainer(oom_recovery=False)`. Pour forcer une taille plus petite : `--batch-size 1`. |
| Le hub Hugging Face renvoie un code 401 / "modèle introuvable". | `DEP_MODEL_LOAD_FAILED` | Utilisez `huggingface-cli login` et réessayez. En cas de faute de frappe, copiez l'identifiant exact depuis <https://huggingface.co/models>. |
| Faute de frappe dans le nom du modèle. | `INPUT_VALIDATION_FAILED` ou `DEP_MODEL_LOAD_FAILED`. | Vérifiez l'identifiant `org/name` sur <https://huggingface.co/models>. |
| `register_with_ollama` : connexion refusée. | `DEP_OLLAMA_REGISTRATION_FAILED` | Démarrez le démon : `ollama serve`. Installez depuis <https://ollama.com>. Opération pouvant être répétée. |
| Le disque est plein lors de la sauvegarde du point de contrôle. | `STATE_CHECKPOINT_INVALID` | Les écritures atomiques laissent un répertoire `.partial` en cas de plantage ; il est sûr de le supprimer. Le point de contrôle précédent est intact. |
| L'entraînement est interrompu / arrêté en raison de la surchauffe de la GPU. | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | B-003 : le moniteur met l'entraînement en pause lorsque le seuil de température NVML est atteint ; il reprend automatiquement lorsque la GPU refroidit. Améliorez la circulation de l'air ou réduisez la charge. |
| `backprop ui --share` est refusé. | `INPUT_AUTH_REQUIRED` | Passez `--auth user:password` ou définissez `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false` pour le désactiver (avec un avertissement). |
| "Chevauchement" des exécutions multiples lors de la validation. | `CONFIG_INVALID` (Étape A, backend B-001). | Réduisez `--samples` en dessous de la taille du pool d'entraînement, augmentez la taille du jeu de données ou désactivez la validation. |
| L'export GGUF a échoué lors de la première tentative. | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; sous Windows, vous avez également besoin des outils de construction Visual C++ et de CMake. |

## Signaler des bogues

Lorsqu'une erreur se produit, Backpropagate affiche une ligne `run_started run_id=<uuid>` au démarrage et associe cet ID aux manifestes des points de contrôle, à l'historique des fusions SLAO et aux lignes de journal structurées. Incluez l'`run_id` dans tout rapport de bogue ; cela permet à un mainteneur de corréler chaque ligne de journal, chaque point de contrôle et chaque fusion pour cette exécution spécifique.

Un bon rapport de bogue comprend :

1. **`run_id`** — l'UUID affiché au démarrage (également disponible sous `TrainingRun.run_id` et `RunResult.run_id`).
2. **Le code d'erreur** — la ligne `[CODE_NAME]: message` dans stderr ; consultez [codes d'erreur](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) pour le catalogue.
3. **La ligne de commande masquée.** La sortie stderr en mode non verbeux est automatiquement masquée (les jetons Bearer, `sk-*`, `hf_*`, les clés AWS, les paires `password=/token=/api_key=` sont supprimées) ; il est donc sûr de la copier. Pour obtenir la trace complète et non masquée, relancez avec `--verbose`, mais examinez-la avant de la publier.
4. **Versions de Python / PyTorch, modèle de GPU, système d'exploitation.** `backprop info` affiche tout cela en une seule fois.

## Confidentialité

Toute l'exécution se fait localement sur votre GPU. Backpropagate ne fait aucune requête réseau, sauf pour télécharger les modèles depuis HuggingFace (ce que vous initiez). Pas de télémétrie, pas de dépendance au cloud.

## Tableau de bord

| Catégorie | Score | Notes |
|----------|-------|-------|
| A. Sécurité | 6/8 | SECURITY.md, modèle de confiance, pas de secrets/télémétrie, safe_path(). Les éléments MCP sont ignorés. |
| B. Gestion des erreurs | 5/7 | Structure des exceptions (code/message/indice/cause/pouvant être retentée) via le registre ERROR_CODES ; codes de sortie de la CLI : 0/1/2/3 ; pas de traces de pile brutes sans l'option `--verbose` ; corrélation par `run_id` ; stderr masqué ; blocage via `--share` + `--auth`. MCP/bureau/vscode ignorés. |
| C. Documentation pour les opérateurs | 4/7 | README, CHANGELOG, LICENSE, --help. Journalisation/MCP/éléments complexes ignorés. |
| D. Hygiène de la livraison | 6/9 | verify.sh, version=tag, 5 scanners dans l'intégration continue, dependabot, python_requires, build propre. |
| E. Identité | 4/4 | Logo, traductions, page d'accueil, métadonnées. |
| **Total** | **25/31** | 14 éléments ignorés avec justification. `shipcheck audit` réussit à 100 %. Date de l'audit : 2026-05-21 (la ligne B a été réévaluée après les travaux de la phase B et les codes de sortie de la CLI de la phase A). |

Historique de la conception et correspondance de chaque élément : voir [ROADMAP.md](ROADMAP.md) — tous les éléments des semaines 1 à 4 sont livrés dans la version 1.1.0.

## Licence

MIT — voir [LICENSE](LICENSE) pour plus de détails.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
