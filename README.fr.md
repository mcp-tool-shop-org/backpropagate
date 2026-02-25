<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="assets/logo.png" alt="Backpropagate" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/backpropagate/"><img src="https://img.shields.io/pypi/v/backpropagate" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**Affinage de LLM sans interface graphique en 3 lignes. Paramètres par défaut intelligents, gestion de la VRAM pour la taille des lots, entraînement SLAO en plusieurs étapes et exportation GGUF en un clic pour Ollama.**

*Entraînez des LLM en 3 lignes de code. Exportez-les vers Ollama en une ligne supplémentaire.*

## Démarrage rapide

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

## Pourquoi la rétropropagation ?

| Problème | Solution |
| --------- | ---------- |
| L'affinage est complexe | 3 lignes : chargement, entraînement, sauvegarde |
| Windows est un cauchemar | Prise en charge complète de Windows |
| La gestion de la VRAM est difficile | Taille automatique des lots, surveillance du GPU |
| L'exportation des modèles est déroutante | Exportation GGUF en un clic + enregistrement automatique avec Ollama |
| Les longues sessions d'entraînement entraînent l'oubli | Entraînement SLAO en plusieurs étapes |

## Fonctionnalités clés

- **Sans interface graphique par conception :** Conçu pour les pipelines CI/CD, les flux de travail automatisés et l'exécution programmatique.
- **Paramètres par défaut intelligents :** Configure automatiquement les hyperparamètres optimaux en fonction de votre matériel et de votre ensemble de données.
- **Entraînement SLAO en plusieurs étapes :** Stratégies d'entraînement avancées pour éviter l'oubli catastrophique lors de longues sessions d'entraînement.
- **Prise en charge complète de Windows :** Testé et optimisé pour les environnements Windows, évitant les problèmes courants de PyTorch/CUDA.
- **Exportation transparente :** Exportation en un clic au format GGUF et enregistrement automatique avec Ollama.
- **Architecture modulaire :** Installez uniquement les dépendances dont vous avez besoin (par exemple, `[unsloth]`, `[ui]`, `[export]`).

## Installation

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Extra | Description | Dépendances |
| ------- | ------------- | -------------- |
| `unsloth` | Entraînement 2 fois plus rapide, 50 % de VRAM en moins | unsloth |
| `ui` | Interface web Gradio | gradio>=5.6.0 |
| `validation` | Validation de configuration Pydantic | pydantic, pydantic-settings |
| `export` | Exportation GGUF pour Ollama | llama-cpp-python |
| `monitoring` | WandB + surveillance du système | wandb, psutil |

**Prérequis :** Python 3.10+ · GPU CUDA (8 Go+ de VRAM) · PyTorch 2.0+

## Utilisation

### Entraînement de base

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

### Entraînement SLAO en plusieurs étapes

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")

result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
    merge_mode="slao",  # Smart LoRA merging
)
```

### Exportation vers Ollama

```python
trainer.export(
    format="gguf",
    quantization="q4_k_m",
    register_ollama=True,
    model_name="my-finetuned-model",
)
# ollama run my-finetuned-model
```

### Ligne de commande

```bash
backprop train --data my_data.jsonl --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backpropagate --ui --port 7862
```

## Prise en charge de Windows

Backpropagate est conçu pour fonctionner sur Windows sans configuration particulière :

- Pré-tokenisation pour éviter les plantages liés au multiprocessing
- Désactivation automatique de xformers pour les séries RTX 40/50
- Paramètres de dataloader sécurisés
- Testé sur RTX 5080 (16 Go de VRAM)

## Modèles prédéfinis

| Modèle | VRAM | Speed | Qualité |
| -------- | ------ | ------- | --------- |
| Qwen 2.5 7B | ~12GB | Moyenne | Best |
| Qwen 2.5 3B | ~8GB | Fast | Good |
| Llama 3.2 3B | ~8GB | Fast | Good |
| Llama 3.2 1B | ~6GB | Le plus rapide | Basic |
| Mistral 7B | ~12GB | Moyenne | Good |

## Architecture

```
backpropagate/
├── trainer.py           # Core Trainer class
├── multi_run.py         # Multi-run SLAO training
├── slao.py              # SLAO LoRA merging algorithm
├── datasets.py          # Dataset loading & filtering
├── export.py            # GGUF/Ollama export
├── config.py            # Pydantic settings
├── gpu_safety.py        # GPU monitoring & safety
└── ui.py                # Gradio interface
```

## Projets connexes

Fait partie de [**MCP Tool Shop**](https://mcp-tool-shop.github.io/):

- [Tool Compass](https://github.com/mcp-tool-shop-org/tool-compass) — Découverte sémantique d'outils MCP
- [File Compass](https://github.com/mcp-tool-shop-org/file-compass) — Recherche sémantique de fichiers
- [Comfy Headless](https://github.com/mcp-tool-shop-org/comfy-headless) — ComfyUI sans la complexité

## Licence

MIT — voir [LICENSE](LICENSE) pour plus de détails.
