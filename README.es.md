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

**Ajuste fino de LLM sin interfaz gráfica en 3 líneas. Configuraciones predeterminadas inteligentes, ajuste automático del tamaño de los lotes según la VRAM, entrenamiento SLAO en múltiples ejecuciones y exportación a GGUF con un solo clic para Ollama.**

*SLAO es Single LoRA Continual Learning via Asymmetric Merging, una técnica de combinación entre ejecuciones que evita el olvido catastrófico durante las campañas de ajuste fino prolongadas ([artículo](https://arxiv.org/abs/2512.23017)).*

*Entrena LLM con 3 líneas de código. Exporte a Ollama con una línea más.*

## Inicio rápido

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("examples/quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

El repositorio incluye un pequeño archivo `examples/quickstart.jsonl` (5 ejemplos en formato ShareGPT) para que el fragmento de código anterior se ejecute de extremo a extremo en una instalación limpia. Para su propio entrenamiento, consulte el formato de conjunto de datos [Dataset Format](#dataset-format) a continuación.

### Opción sin código: Interfaz web

¿Prefiere una interfaz gráfica en lugar de una terminal de Python? Instale el paquete correspondiente y ejecute:

```bash
pip install backpropagate[standard]
backprop ui --port 7862
```

La interfaz de Reflex (Radix UI) le permite seleccionar un archivo JSONL, elegir un modelo, entrenar y exportar, sin necesidad de Python. La interfaz es local; para acceder desde Internet, consulte la sección [Web UI](#web-ui) a continuación, donde se explica el contrato de seguridad `--share` + `--auth` y las opciones de túnel admitidas (Cloudflare Tunnel, ngrok).

## Formato del conjunto de datos

Su archivo de entrenamiento en formato JSONL debe tener un ejemplo por línea. El formato más sencillo es el chat de ShareGPT:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

También se admiten formatos Alpaca (`instruction`/`output`), OpenAI chat (`messages`) y texto sin formato. Consulte `examples/quickstart.jsonl` para obtener un punto de partida que se puede copiar.

## ¿Por qué propagar hacia atrás?

| Problema | Solución |
|---------|----------|
| El ajuste fino es complejo | 3 líneas: cargar, entrenar, guardar |
| Windows es un problema | Soporte completo para Windows |
| La gestión de la VRAM es difícil | Ajuste automático del tamaño de los lotes, monitoreo de la GPU |
| La exportación de modelos es confusa | Exportación a GGUF con un solo clic + registro automático en Ollama |
| Las ejecuciones prolongadas causan olvido | Entrenamiento SLAO en múltiples ejecuciones |

## Características principales

- **Diseñado para funcionar sin interfaz gráfica:** Ideal para pipelines de CI/CD, flujos de trabajo automatizados y ejecución programática.
- **Configuraciones predeterminadas inteligentes:** Configura automáticamente los hiperparámetros óptimos según su hardware y conjunto de datos.
- **Entrenamiento SLAO en múltiples ejecuciones:** Estrategias de entrenamiento avanzadas para evitar el olvido catastrófico durante las ejecuciones prolongadas.
- **Soporte completo para Windows:** Probado y optimizado para entornos Windows, evitando problemas comunes de PyTorch/CUDA.
- **Exportación sencilla:** Exportación con un solo clic al formato GGUF y registro automático en Ollama.
- **Arquitectura modular:** Instale solo las dependencias que necesita (por ejemplo, `[unsloth]`, `[ui]`, `[export]`).

## Instalación

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Reflex (Radix UI) web interface
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Paquetes adicionales | Descripción | Dependencias |
|-------|-------------|--------------|
| `unsloth` | Entrenamiento 2 veces más rápido, 50% menos de VRAM | unsloth |
| `ui` | Interfaz web Reflex (Radix UI) | reflex>=0.9.2, fastapi>=0.115 |
| `validation` | Validación de configuración de Pydantic | pydantic, pydantic-settings |
| `export` | Exportación a GGUF para Ollama | llama-cpp-python |
| `monitoring` | WandB + monitoreo del sistema (integrado automáticamente en el entrenador en v1.1.0) | wandb, psutil |
| `observability` | Trazado de OpenTelemetry | opentelemetry-api, opentelemetry-sdk |
| `logging` | Registro estructurado | structlog |
| `security` | Autenticación JWT + generación de tokens | PyJWT, cryptography |
| `production` | unsloth + ui + validación + registro + seguridad | (paquete) |

**Requisitos:** Python 3.10+ · GPU con CUDA (8GB+ de VRAM) · PyTorch 2.0+

### Requisitos de la plataforma

Backpropagate se encarga de las peculiaridades de la ejecución (multiproceso, xformers en RTX 40/50, trabajadores del cargador de datos en Windows). **No** se encarga de los problemas de instalación relacionados con la plataforma; resuélvalos primero:

- **Versión del kit de herramientas CUDA.** PyTorch se publica por versión de CUDA; elegir la versión incorrecta instala silenciosamente solo la versión de torch para CPU. Utilice el selector en <https://pytorch.org/get-started/locally/> para obtener el comando exacto `pip install torch ...` para su controlador. Ejecute `nvidia-smi` para ver la versión de su controlador/CUDA.
- **Windows.** Visual Studio Build Tools (C++) y CMake son necesarios para el extra `[export]` (las compilaciones de `llama-cpp-python` se realizan desde el código fuente). El paquete `bitsandbytes` se publica ahora de forma nativa para Windows (>= 0.43); las guías anteriores que mencionan `bitsandbytes-windows` están desactualizadas.
- **macOS.** El entrenamiento con GPU **no está soportado**; no hay CUDA. Puede instalar Backpropagate para ejecutar la *inferencia* en un modelo GGUF exportado a través de Ollama, pero `trainer.train()` genera un error `DEP_GPU_NOT_AVAILABLE`. Utilice una máquina con CUDA para el entrenamiento.
- **Linux.** La mayoría de las distribuciones funcionan sin problemas. Si está utilizando la versión binaria de PyPI, tenga en cuenta que la compilación de Linux utiliza solo la versión de torch para CPU (para mantenerse dentro del límite de 2 GB de archivos adjuntos de lanzamiento de GitHub); instale primero el paquete CUDA correspondiente de pytorch.org.

Para la resolución de problemas de instalación más detallada, consulte [la página del manual de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/).

## Configuración

Todas las configuraciones se pueden sobrescribir con variables de entorno utilizando el prefijo `BACKPROPAGATE_` (por ejemplo, `BACKPROPAGATE_LOG_LEVEL=debug`). Un archivo `.env` en la raíz del proyecto se carga automáticamente cuando se instala el extra `[validation]`.

Parámetros comunes (consulte [la referencia completa de las variables de entorno](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/) para obtener información completa):

| Variable | Valor predeterminado | Notas |
|----------|---------|-------|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Fuerza registros en formato JSON (`true`) o en la consola (`false`) |
| `BACKPROPAGATE_LOG_FILE` | no definido | Ruta para guardar los registros |
| `BACKPROPAGATE_DEFER_FEATURE_DETECTION` | no definido | Omite la detección de dependencias opcionales al inicio para un inicio más rápido de la interfaz de línea de comandos. |
| `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE` | `true` | Cuando es `true`, rechaza `backprop ui --share` sin `--auth` |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Directorio base para todas las escrituras del sistema de archivos de la interfaz de usuario; validado por denylist. |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modelo predeterminado |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Tasa de aprendizaje |
| `BACKPROPAGATE_LORA__R` | `16` | Rango de LoRA |

Las claves anidadas utilizan el doble guion bajo como delimitador (convención de Pydantic `env_nested_delimiter`).

## Uso

### Entrenamiento básico

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

`Qwen/Qwen2.5-7B-Instruct` es el valor predeterminado canónico; el valor `Trainer()` se resuelve cuando se llama sin un argumento de modelo (consulte [`config.py`](backpropagate/config.py) `ModelConfig.name`). Los ejemplos anteriores utilizaban `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` pre-cuantizado; hemos cambiado el valor predeterminado a los pesos oficiales de Qwen para una mejor fiabilidad ([REGISTRO DE CAMBIOS v0.1.3](CHANGELOG.md)). Cualquier modelo funciona.

### Entrenamiento multi-ejecución SLAO

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

SLAO (Aprendizaje continuo de LoRA único mediante fusión asimétrica) implementa el artículo [Merge before Forget](https://arxiv.org/abs/2512.23017): inicialización ortogonal de la matriz A mediante descomposición QR, manejo asimétrico de A/B y escalado dependiente del tiempo `λ(i) = 1/√i`. La opción de la interfaz de línea de comandos es `--samples` (el campo subyacente es `samples_per_run`).

### Exportación a Ollama

```python
# Export to GGUF
result = trainer.export("gguf", quantization="q4_k_m")

# Register with Ollama separately
from backpropagate import register_with_ollama
register_with_ollama(result.path, "my-finetuned-model")
# ollama run my-finetuned-model
```

### Interfaz de línea de comandos

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

Consulte [la referencia de la interfaz de línea de comandos](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) para cada subcomando y opción, o ejecute `backprop <subcomando> --help`.

### Reanudación desde el punto de control (v1.1.0)

Una ejecución multi-ejecución que falla en la ejecución 4 ahora es recuperable. Cada sesión de ejecución multi-ejecución escribe su `run_id` tanto en `run_history.json` como en el manifiesto de puntos de control en el disco, por lo que para reanudar desde donde lo dejó, solo necesita un comando:

```bash
backprop resume <run-id>                       # picks up the in-progress session
backprop multi-run --data ... --resume <run-id> # explicit form
backprop train --data ... --resume <run-id>    # single-run resume (continues run_id)
```

El comportamiento predeterminado de `backprop multi-run` (sin `--resume`) detecta automáticamente una entrada en curso para el mismo directorio de salida y la continúa. Para forzar una sesión limpia, pase `resume_from="off"` (API de Python) o omita `--resume` y comience en un directorio de salida nuevo.

Cuando se reanuda una ejecución multi-ejecución, el último punto de control para ese `run_id` se carga en el modelo, el estado de fusión SLAO se restaura desde el directorio `slao/` junto al punto de control, y el bucle de ejecución continúa desde `last_completed_run + 1`. El estado de la entrada de historial cambia de nuevo a `running`, por lo que `backprop list-runs --status running` muestra la sesión activa.

### Seguimiento de experimentos (v1.1.0)

`Trainer` detecta automáticamente los rastreadores de experimentos instalados (`wandb`, `tensorboard`, `mlflow`) y los integra en los `transformers.TrainingArguments` subyacentes. El valor predeterminado `report_to="auto"` selecciona lo que sea importable:

```bash
pip install backpropagate[monitoring]  # installs wandb + psutil
wandb login                            # one-time
backprop train --data my_data.jsonl    # W&B run gets the same run_id prefix as the on-disk history
```

Para optar explícitamente por no usarlo, use `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` o `Trainer(report_to="none")`. Para MLflow, instale `pip install mlflow`; para TensorBoard, instale `pip install tensorboard`. El nombre de la ejecución de W&B es `backprop-<run_id_prefix>`, lo que permite a un operador buscar en W&B, nuestros registros y `run_history.json` utilizando el mismo identificador.

### Historial de entrenamiento

Cada invocación de `backprop train` y `backprop multi-run` registra una fila en `<output>/run_history.json` con el `run_id`, el modelo, el conjunto de datos, los hiperparámetros, el estado, la pérdida final, el historial de pérdidas y, para las ejecuciones multi-ejecución, la línea de tiempo de fusión SLAO. Para ver las ejecuciones recientes:

```bash
backprop list-runs                         # most recent 20 runs, all statuses
backprop list-runs --status failed         # filter
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial run_id ok)
```

El historial de ejecuciones persiste entre procesos; la pestaña `Runs` en la interfaz web es una vista en memoria; el historial en el disco es la fuente de verdad para `list-runs` / `show-run` / `resume`.

### Interfaz web

Inicie la interfaz de Reflex localmente:

```bash
backprop ui --port 7862
```

Para exponer una URL accesible desde Internet, debe combinar `--share` con `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` sin `--auth` termina con el código `1` y el error estructurado `[INPUT_AUTH_REQUIRED]`. La razón: `--share` publica una URL `*.gradio.live` que cualquier persona en Internet puede acceder, y sin autenticación, eso significa que cualquier persona puede controlar su canal de entrenamiento.

Para optar explícitamente por no usarlo (por ejemplo, en un entorno de desarrollo interno), establezca la variable de entorno `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false`. Se imprimirá una advertencia audible en cada inicio, y hay un período de gracia de 5 segundos antes de que la interfaz de usuario sin autenticación se active, por lo que puede presionar `Ctrl-C` si parece incorrecta.

Las escrituras en el sistema de archivos desde la interfaz de usuario están aisladas en un solo directorio:

- Predeterminado: `~/.backpropagate/ui-outputs`
- Para sobrescribir: `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- La sobrescritura está **validada mediante una lista de denegación**; las rutas del sistema/credenciales (`/etc`, `/var`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) se rechazan con `[UI_OUTPUT_DIR_FORBIDDEN]`.

## Soporte para Windows

Backpropagate está diseñado para funcionar en Windows de forma predeterminada:

- Pre-tokenización para evitar fallos de multiprocesamiento
- Desactivación automática de xformers para series RTX 40/50
- Configuración segura del cargador de datos
- Probado en RTX 5080 (16GB de VRAM)

## Preajustes de modelos

| Preajuste | VRAM | Velocidad | Calidad |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12GB | Medio | Mejor |
| Qwen 2.5 3B | ~8GB | Rápido | Bueno |
| Llama 3.2 3B | ~8GB | Rápido | Bueno |
| Llama 3.2 1B | ~6GB | Más rápido | Básico |
| Mistral 7B | ~12GB | Medio | Bueno |

## Arquitectura

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
├── ui_security.py       # Rate limiting, CSRF, file validation (framework-agnostic)
├── ui_gradio_legacy.py  # DEPRECATED — preserved as v1.0 reference; removed in v1.2
└── theme_gradio_legacy.py  # DEPRECATED — same
```

## Resolución de problemas

Un índice breve de los fallos más comunes al inicio. El índice inverso completo se encuentra en [la página del manual de resolución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/); cada código que se muestra a continuación está documentado en [códigos de error](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/).

| Síntoma | Código | Solución |
|---------|------|-----|
| La GPU se queda sin memoria durante el entrenamiento. | `RUNTIME_GPU_OOM` | La recuperación automática de errores de memoria (B-002) reduce automáticamente el tamaño del lote hasta 3 veces. Para desactivar esta función: `Trainer(oom_recovery=False)`. Para forzar un tamaño de lote más pequeño: `--batch-size 1`. |
| El centro de descargas de Hugging Face devuelve un error 401 / "modelo no encontrado". | `DEP_MODEL_LOAD_FAILED` | Ejecute `huggingface-cli login` e inténtelo de nuevo. Para corregir errores tipográficos, copie el ID exacto de <https://huggingface.co/models>. |
| Error tipográfico en el nombre del modelo. | `INPUT_VALIDATION_FAILED` o `DEP_MODEL_LOAD_FAILED`. | Verifique el identificador `org/name` en <https://huggingface.co/models>. |
| `register_with_ollama` rechaza la conexión. | `DEP_OLLAMA_REGISTRATION_FAILED` | Inicie el demonio: `ollama serve`. Instálelo desde <https://ollama.com>. Se puede volver a intentar. |
| El disco se llena durante el guardado del punto de control. | `STATE_CHECKPOINT_INVALID` | Las escrituras atómicas dejan un directorio `.partial` en caso de fallo; es seguro eliminarlo. El punto de control anterior y válido está intacto. |
| El entrenamiento se pausa/interrumpe debido al sobrecalentamiento de la GPU. | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | B-003: el monitor se pausa debido al umbral de temperatura de NVML; se reanuda automáticamente a medida que la GPU se enfría. Mejore el flujo de aire o reduzca la carga sostenida. |
| `backprop ui --share` es rechazado. | `INPUT_AUTH_REQUIRED` | Pase `--auth user:password` o configure `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false` para desactivar esta función (con una advertencia). |
| "Superposición de validación" en múltiples ejecuciones. | `CONFIG_INVALID` (Etapa A, backend B-001). | Reduzca `--samples` por debajo del tamaño del conjunto de entrenamiento, aumente el conjunto de datos o desactive la validación. |
| La exportación a GGUF falló en el primer intento. | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; en Windows, también necesita Visual C++ Build Tools + CMake. |

## Informar de errores

Cuando algo falla, Backpropagate imprime una línea `run_started run_id=<uuid>` al inicio y asocia el mismo ID a los manifiestos de puntos de control, la historia de combinación de SLAO y las líneas de registro estructuradas. Incluya el `run_id` en cualquier informe de error; esto permite a un mantenedor correlacionar cada línea de registro, cada punto de control y cada combinación para esa ejecución específica.

Un buen informe de error incluye:

1. **`run_id`** — el UUID impreso al inicio (también disponible como `TrainingRun.run_id` y `RunResult.run_id`).
2. **El código de error** — la línea `[CODE_NAME]: message` en stderr es lo que debe buscar; consulte [códigos de error](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) para ver el catálogo.
3. **La línea de comandos sin información confidencial.** La salida de error en modo no detallado se borra automáticamente (los tokens Bearer, `sk-*`, `hf_*`, las claves de AWS, los pares `password=/token=/api_key=` se eliminan): es seguro pegarla. Para obtener el rastreo completo y sin información confidencial, vuelva a ejecutar con `--verbose`, pero revise antes de publicarlo.
4. **Versiones de Python / PyTorch, modelo de GPU, sistema operativo.** `backprop info` imprime todo esto de una vez.

## Privacidad

Todo el entrenamiento se realiza localmente en su GPU. Backpropagate no realiza solicitudes de red excepto para descargar modelos de HuggingFace (lo cual usted inicia). No hay telemetría, ni dependencia de la nube.

## Tabla de rendimiento

| Categoría | Puntuación | Notas |
|----------|-------|-------|
| A. Seguridad | 6/8 | SECURITY.md, modelo de confianza, sin secretos/telemetría, safe_path(). Se omiten los elementos de MCP. |
| B. Manejo de errores | 5/7 | Estructura de excepciones (`código`/`mensaje`/`indicación`/`causa`/`reintentable`) a través del registro ERROR_CODES; códigos de salida de la CLI: 0/1/2/3; no se muestran rastreos de pila sin `--verbose`; correlación con `run_id`; salida de error estándar (stderr) censurada; bloqueo mediante `--share` + `--auth`. MCP/escritorio/VS Code omitidos. |
| C. Documentación para operadores | 4/7 | README, CHANGELOG, LICENCIA, --help. Registro/MCP/elementos complejos omitidos. |
| D. Higiene del proceso de entrega | 6/9 | verify.sh, versión=etiqueta, 5 analizadores en CI, dependabot, python_requires, compilación limpia. |
| E. Identidad | 4/4 | Logotipo, traducciones, página de inicio, metadatos. |
| **Total** | **25/31** | 14 elementos omitidos con justificación. `shipcheck audit` pasa el 100%. Fecha de auditoría: 2026-05-21 (la fila B fue reevaluada después de la etapa B y el trabajo de códigos de salida de la CLI). |

Historial de diseño y correspondencia de cada elemento: consulte [ROADMAP.md](ROADMAP.md). Todos los elementos de las semanas 1 a 4 se incluyen en la versión 1.1.0.

## Licencia

MIT: Consulte [LICENSE](LICENSE) para obtener más detalles.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
