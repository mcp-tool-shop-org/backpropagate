<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Ajuste fino de LLM sin interfaz gráfica en 3 líneas. Configuraciones predeterminadas inteligentes, ajuste de tamaño de lote consciente de la VRAM, entrenamiento SLAO en múltiples ejecuciones y exportación a GGUF con un solo clic para Ollama.**

*Entrena LLM con 3 líneas de código. Exporta a Ollama con una línea más.*

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

## ¿Por qué retropropagación?

| Problema | Solución |
|---------|----------|
| El ajuste fino es complejo | 3 líneas: cargar, entrenar, guardar |
| Windows es una pesadilla | Soporte completo para Windows |
| La gestión de la VRAM es difícil | Ajuste automático del tamaño del lote, monitoreo de la GPU |
| La exportación de modelos es confusa | Exportación a GGUF con un solo clic + registro automático en Ollama |
| Las ejecuciones largas causan olvido | Entrenamiento SLAO en múltiples ejecuciones |

## Características principales

- **Diseñado para funcionar sin interfaz gráfica:** Ideal para pipelines de CI/CD, flujos de trabajo automatizados y ejecución programática.
- **Configuraciones predeterminadas inteligentes:** Configura automáticamente los hiperparámetros óptimos según tu hardware y conjunto de datos.
- **Entrenamiento SLAO en múltiples ejecuciones:** Estrategias de entrenamiento avanzadas para evitar el olvido catastrófico durante las ejecuciones largas.
- **Soporte completo para Windows:** Probado y optimizado para entornos Windows, evitando problemas comunes de PyTorch/CUDA.
- **Exportación sencilla:** Exportación a formato GGUF con un solo clic y registro automático en Ollama.
- **Arquitectura modular:** Instala solo las dependencias que necesitas (por ejemplo, `[unsloth]`, `[ui]`, `[export]`).

## Instalación

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Información adicional | Descripción | Dependencias |
|-------|-------------|--------------|
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

### Entrenamiento SLAO en múltiples ejecuciones

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

Backpropagate está diseñado para funcionar en Windows de forma nativa:

- Pre-tokenización para evitar fallos de multiproceso
- Desactivación automática de xformers para series RTX 40/50
- Configuraciones de cargador de datos seguras
- Probado en RTX 5080 (16GB de VRAM)

## Configuraciones predefinidas

| Configuración | VRAM | Velocidad | Calidad |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12GB | Media | Óptima |
| Qwen 2.5 3B | ~8GB | Rápida | Buena |
| Llama 3.2 3B | ~8GB | Rápida | Buena |
| Llama 3.2 1B | ~6GB | La más rápida | Básica |
| Mistral 7B | ~12GB | Media | Buena |

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

## Privacidad

Todo el entrenamiento se realiza localmente en tu GPU. Backpropagate no realiza solicitudes de red excepto para descargar modelos de HuggingFace (que tú inicias). No hay telemetría, ni dependencia de la nube.

## Informe de rendimiento

| Categoría | Puntuación | Notas |
|----------|-------|-------|
| A. Seguridad | 10/10 | SECURITY.md, Bandit+Semgrep+Trivy+TruffleHog en CI, protección contra recorrido de rutas |
| B. Manejo de errores | 8/10 | Errores estructurados, umbrales de seguridad de la GPU, recuperación de puntos de control |
| C. Documentación para operadores | 9/10 | README, CHANGELOG, guía de instalación modular, ayuda de la interfaz de línea de comandos |
| D. Higiene de envío | 9/10 | CI + pruebas (33 archivos), publicado en PyPI, cobertura de Codecov |
| E. Identidad | 10/10 | Logotipo, traducciones, página de inicio, listado en PyPI. |
| **Total** | **46/50** | |

## Licencia

MIT: Consulte el archivo [LICENSE](LICENSE) para obtener más detalles.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
