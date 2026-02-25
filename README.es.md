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

**Ajuste fino de LLM sin interfaz gráfica en 3 líneas. Configuraciones predeterminadas inteligentes, ajuste de tamaño de lote que considera la VRAM, entrenamiento SLAO con múltiples ejecuciones y exportación a GGUF con un solo clic para Ollama.**

*Entrene LLM con 3 líneas de código. Exporte a Ollama con una línea más.*

## Cómo empezar

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

## ¿Por qué usar la retropropagación?

| Problema | Solución |
| --------- | ---------- |
| El ajuste fino es complejo | 3 líneas: cargar, entrenar, guardar |
| Windows es un problema | Soporte completo para Windows |
| La gestión de la VRAM es difícil | Ajuste automático del tamaño del lote, monitoreo de la GPU |
| La exportación de modelos es confusa | Exportación a GGUF con un solo clic + registro automático en Ollama |
| Las ejecuciones largas provocan olvido | Entrenamiento SLAO con múltiples ejecuciones |

## Características principales

- **Diseñado para funcionar sin interfaz gráfica:** Ideal para pipelines de CI/CD, flujos de trabajo automatizados y ejecución programática.
- **Configuraciones predeterminadas inteligentes:** Configura automáticamente los hiperparámetros óptimos según su hardware y conjunto de datos.
- **Entrenamiento SLAO con múltiples ejecuciones:** Estrategias de entrenamiento avanzadas para evitar el olvido catastrófico durante las ejecuciones largas.
- **Soporte completo para Windows:** Probado y optimizado para entornos Windows, evitando problemas comunes de PyTorch/CUDA.
- **Exportación sencilla:** Exportación a formato GGUF con un solo clic y registro automático en Ollama.
- **Arquitectura modular:** Instale solo las dependencias que necesita (por ejemplo, `[unsloth]`, `[ui]`, `[export]`).

## Instalación

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Extra | Descripción | Dependencias |
| ------- | ------------- | -------------- |
| `unsloth` | Entrenamiento 2 veces más rápido, 50% menos de VRAM | unsloth |
| `ui` | Interfaz web de Gradio | gradio>=5.6.0 |
| `validation` | Validación de configuración de Pydantic | pydantic, pydantic-settings |
| `export` | Exportación a GGUF para Ollama | llama-cpp-python |
| `monitoring` | WandB + monitoreo del sistema | wandb, psutil |

**Requisitos:** Python 3.10+, GPU con CUDA (8GB+ de VRAM), PyTorch 2.0+

## Uso

### Entrenamiento básico

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

### Entrenamiento SLAO con múltiples ejecuciones

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

### Exportación a Ollama

```python
trainer.export(
    format="gguf",
    quantization="q4_k_m",
    register_ollama=True,
    model_name="my-finetuned-model",
)
# ollama run my-finetuned-model
```

### Interfaz de línea de comandos (CLI)

```bash
backprop train --data my_data.jsonl --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backpropagate --ui --port 7862
```

## Soporte para Windows

Backpropagate está diseñado para funcionar en Windows de forma predeterminada:

- Pre-tokenización para evitar fallos de multiproceso
- Desactivación automática de xformers para series RTX 40/50
- Configuraciones de cargador de datos seguras
- Probado en RTX 5080 (16GB de VRAM)

## Modelos preconfigurados

| Preconfiguración | VRAM | Speed | Calidad |
| -------- | ------ | ------- | --------- |
| Qwen 2.5 7B | ~12GB | Media | Best |
| Qwen 2.5 3B | ~8GB | Fast | Good |
| Llama 3.2 3B | ~8GB | Fast | Good |
| Llama 3.2 1B | ~6GB | La más rápida | Basic |
| Mistral 7B | ~12GB | Media | Good |

## Arquitectura

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

## Proyectos relacionados

Parte de [**MCP Tool Shop**](https://mcp-tool-shop.github.io/):

- [Tool Compass](https://github.com/mcp-tool-shop-org/tool-compass) — Descubrimiento semántico de herramientas MCP
- [File Compass](https://github.com/mcp-tool-shop-org/file-compass) — Búsqueda semántica de archivos
- [Comfy Headless](https://github.com/mcp-tool-shop-org/comfy-headless) — ComfyUI sin la complejidad

## Licencia

MIT — consulte [LICENSE](LICENSE) para obtener más detalles.
