<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Fine-tuning di LLM senza interfaccia grafica in 3 righe di codice. Impostazioni predefinite intelligenti, gestione della VRAM per l'ottimizzazione del batch, addestramento SLAO multi-run e esportazione GGUF con un solo clic per Ollama.**

*Addestra modelli linguistici di grandi dimensioni con 3 righe di codice. Esportali su Ollama con una riga in più.*

## Guida Rapida

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

## Perché la retropropagazione?

| Problema | Soluzione |
|---------|----------|
| Il fine-tuning è complesso | 3 righe: caricamento, addestramento, salvataggio |
| Windows è un incubo | Supporto completo per Windows |
| La gestione della VRAM è difficile | Dimensionamento automatico del batch, monitoraggio della GPU |
| L'esportazione dei modelli è complicata | Esportazione GGUF con un clic + registrazione automatica con Ollama |
| Addestramenti prolungati causano dimenticanza | Addestramento SLAO multi-run |

## Caratteristiche Principali

- **Senza interfaccia grafica per design**: Progettato per pipeline CI/CD, flussi di lavoro automatizzati ed esecuzione programmatica.
- **Impostazioni predefinite intelligenti**: Configura automaticamente gli iperparametri ottimali in base all'hardware e al dataset.
- **Addestramento SLAO multi-run**: Strategie di addestramento avanzate per prevenire la perdita di informazioni durante addestramenti prolungati.
- **Supporto completo per Windows**: Testato e ottimizzato per ambienti Windows, evitando i problemi comuni di PyTorch/CUDA.
- **Esportazione semplificata**: Esportazione con un clic in formato GGUF e registrazione automatica con Ollama.
- **Architettura modulare**: Installa solo le dipendenze necessarie (ad esempio, `[unsloth]`, `[ui]`, `[export]`).

## Installazione

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Extra | Descrizione | Dipendenze |
|-------|-------------|--------------|
| `unsloth` | Addestramento 2 volte più veloce, 50% in meno di VRAM | unsloth |
| `ui` | Interfaccia web Gradio | gradio>=5.6.0 |
| `validation` | Validazione della configurazione Pydantic | pydantic, pydantic-settings |
| `export` | Esportazione GGUF per Ollama | llama-cpp-python |
| `monitoring` | WandB + monitoraggio del sistema | wandb, psutil |

**Requisiti:** Python 3.10+, GPU CUDA (8GB+ di VRAM), PyTorch 2.0+

## Utilizzo

### Addestramento di base

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

### Addestramento SLAO multi-run

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

### Esportazione su Ollama

```python
trainer.export(
    format="gguf",
    quantization="q4_k_m",
    register_ollama=True,
    model_name="my-finetuned-model",
)
# ollama run my-finetuned-model
```

### Interfaccia a riga di comando (CLI)

```bash
backprop train --data my_data.jsonl --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backpropagate --ui --port 7862
```

## Supporto per Windows

Backpropagate è progettato per funzionare su Windows senza problemi:

- Pre-tokenizzazione per evitare crash dovuti al multiprocessing
- Disattivazione automatica di xformers per serie RTX 40/50
- Impostazioni sicure del dataloader
- Testato su RTX 5080 (16GB di VRAM)

## Modelli predefiniti

| Modello | VRAM | Velocità | Qualità |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12GB | Media | Ottima |
| Qwen 2.5 3B | ~8GB | Veloce | Buona |
| Llama 3.2 3B | ~8GB | Veloce | Buona |
| Llama 3.2 1B | ~6GB | Velocissima | Base |
| Mistral 7B | ~12GB | Media | Buona |

## Architettura

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

## Privacy

Tutto l'addestramento avviene localmente sulla tua GPU. Backpropagate non effettua richieste di rete, ad eccezione del download dei modelli da HuggingFace (che viene avviato da te). Nessuna telemetria, nessuna dipendenza dal cloud.

## Tabella di valutazione

| Categoria | Punteggio | Note |
|----------|-------|-------|
| A. Sicurezza | 10/10 | SECURITY.md, Bandit+Semgrep+Trivy+TruffleHog in CI, protezione contro l'accesso non autorizzato |
| B. Gestione degli errori | 8/10 | Errori strutturati, limiti di sicurezza della GPU, ripristino dei checkpoint |
| C. Documentazione per gli operatori | 9/10 | README, CHANGELOG, guida all'installazione modulare, aiuto CLI |
| D. Igiene durante la distribuzione | 9/10 | CI + test (33 file), pubblicato su PyPI, copertura Codecov |
| E. Identità | 10/10 | Logo, traduzioni, pagina di destinazione, elenco su PyPI. |
| **Total** | **46/50** | |

## Licenza

MIT — vedere [LICENSE](LICENSE) per i dettagli.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
