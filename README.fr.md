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

**Affinage de LLM sans interface graphique en 3 lignes. Paramètres par défaut intelligents, gestion de la taille des lots en fonction de la VRAM, entraînement SLAO multi-exécution et exportation GGUF en un clic pour Ollama.**

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
|---------|----------|
| L'affinage est complexe | 3 lignes : chargement, entraînement, sauvegarde |
| Windows est un cauchemar | Prise en charge complète de Windows |
| La gestion de la VRAM est difficile | Ajustement automatique de la taille des lots, surveillance du GPU |
| L'exportation de modèles est déroutante | Exportation GGUF en un clic + enregistrement automatique avec Ollama |
| Les longues exécutions entraînent l'oubli | Entraînement SLAO multi-exécution |

## Fonctionnalités clés

- **Sans interface graphique par conception :** Conçu pour les pipelines CI/CD, les flux de travail automatisés et l'exécution programmatique.
- **Paramètres par défaut intelligents :** Configure automatiquement les hyperparamètres optimaux en fonction de votre matériel et de votre ensemble de données.
- **Entraînement SLAO multi-exécution :** Stratégies d'entraînement avancées pour éviter l'oubli catastrophique lors de longues exécutions.
- **Prise en charge complète de Windows :** Testé et optimisé pour les environnements Windows, évitant les pièges courants de PyTorch/CUDA.
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

| Complémentaire | Description | Dépendances |
|-------|-------------|--------------|
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

### Entraînement SLAO multi-exécution

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

## Modèles préconfigurés

| Modèle | VRAM | Vitesse | Qualité |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12 Go | Moyenne | Meilleure |
| Qwen 2.5 3B | ~8 Go | Rapide | Bon |
| Llama 3.2 3B | ~8 Go | Rapide | Bon |
| Llama 3.2 1B | ~6 Go | Le plus rapide | Basique |
| Mistral 7B | ~12 Go | Moyenne | Bon |

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

## Confidentialité

Toutes les opérations d'entraînement se déroulent localement sur votre GPU. Backpropagate ne fait aucune requête réseau, sauf pour télécharger les modèles depuis HuggingFace (ce que vous initiez). Aucune télémétrie, aucune dépendance au cloud.

## Tableau de bord

| Catégorie | Score | Notes |
|----------|-------|-------|
| A. Sécurité | 10/10 | SECURITY.md, Bandit+Semgrep+Trivy+TruffleHog dans CI, protection contre les parcours de chemin |
| B. Gestion des erreurs | 8/10 | Erreurs structurées, seuils de sécurité du GPU, récupération des points de contrôle |
| C. Documentation pour les opérateurs | 9/10 | README, CHANGELOG, guide d'installation modulaire, aide de la ligne de commande |
| D. Hygiène de déploiement | 9/10 | CI + tests (33 fichiers), publié sur PyPI, couverture Codecov |
| E. Identité | 10/10 | Logo, traductions, page d'accueil, fiche PyPI. |
| **Total** | **46/50** | |

## Licence

MIT — voir [LICENSE](LICENSE) pour plus de détails.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
