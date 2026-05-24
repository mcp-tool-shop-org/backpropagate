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
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/backpropagate"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/backpropagate/badge" alt="OpenSSF Scorecard"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

# Entrena un adaptador. Envíalo a Ollama. ¡Listo!

Backpropagate es una biblioteca de Python para el ajuste fino de modelos de lenguaje grandes en una sola GPU. Tres líneas de código entrenan un modelo de 7B en una tarjeta de 16GB. Un comando más lo exporta a Ollama para que puedas ejecutar tu ajuste fino con `ollama run`. Funciona perfectamente en Windows.

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")
```

```bash
backprop ollama register ./output/lora --name my-model
ollama run my-model
```

Eso es todo. No hay un archivo de configuración YAML. No hay una "ceremonia" de `accelerate launch`. No hay un tutorial separado para "convertirlo a GGUF". Si tienes una GPU con CUDA y un archivo JSONL con tus datos de entrenamiento, estás a solo tres líneas de tener un ajuste fino funcional.

## Instalación

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

Si quieres las funciones opcionales, reemplaza la instalación con una de estas:

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

¿Prefieres Docker? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` también funciona. Las imágenes están disponibles para `linux/amd64` y `linux/arm64`, por lo que los usuarios de Apple Silicon y ARM Linux obtienen una imagen nativa. Un archivo `compose.yaml` canónico para "UI en un contenedor" se encuentra en la raíz del repositorio; `docker compose up` inicia la interfaz web en `http://localhost:7860` con un volumen persistente `~/.backpropagate` montado.

## Dónde encaja Backpropagate

Existen varias bibliotecas excelentes para el ajuste fino de LLM. Cada una es excelente para diferentes cosas:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**: si te gustan las configuraciones YAML y quieres una comunidad de recetas para copiar.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**: si quieres una GUI web y soporte integrado para DPO/PPO/RLHF.
- **[Unsloth](https://github.com/unslothai/unsloth)**: si necesitas el entrenamiento más rápido posible y estás utilizando una familia de modelos compatible.
- **[torchtune](https://github.com/pytorch/torchtune)**: si quieres las recetas nativas de PyTorch de Meta que puedes editar.

Backpropagate es la opción que falta: **una API de Python de 3 líneas para usuarios individuales en una sola GPU de consumo que desean entrenar un adaptador y enviarlo.** Sin YAML, sin GUI, sin DPO/PPO, sin nodos múltiples. Solo el bucle que realmente necesita todo el mundo y el paso de exportación que dificulta las cosas.

Si has probado alguna de las bibliotecas anteriores y te has sentido frustrado por la "ceremonia" del archivo de configuración, o has encontrado una limitación en la familia de modelos, o querías configuraciones predeterminadas para Windows: Backpropagate es para ti.

## Lo que puedes ajustar en una GPU de consumo de 16GB

Aquí tienes un rango práctico en una tarjeta de 16GB (RTX 4080 / 5080 / 4070 Ti Super):

| Modelo | Método | Estado |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | Cómodo. Longitud de secuencia completa, con espacio de sobra. |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | Estándar. ~7-8 GB. Configuraciones predeterminadas de Backpropagate. |
| Llama-3 13B | QLoRA + empaquetado de muestras | Apretado pero funciona. Usa secuencias más cortas. |
| Mixtral 8x7B (47B de parámetros totales) | AQLM de 2 bits + LoRA | Experimental en v1.4. El modelo más grande que puedes usar en una tarjeta de 16GB. |

Para modelos de 3B y menos, el ajuste fino completo (no solo LoRA) es factible en 16GB y está planeado como una opción `mode="full"` para v1.4. Para modelos de 7B o más, el ajuste fino completo requiere una GPU de 24GB o más: considera alquilar una instancia de A100 en la nube, o quédate con LoRA, que la investigación reciente muestra que coincide con la calidad del ajuste fino completo en la mayoría de las tareas posteriores al entrenamiento (consulta la sección "lo que Backpropagate no es" para obtener citas).

## Lo que Backpropagate NO es

La honestidad ayuda a todos. Backpropagate no hace estas cosas, e intentar que lo haga sería una experiencia peor que buscar la herramienta adecuada:

- **Ajuste fino completo de parámetros para modelos de 7B+** — Backpropagate utiliza LoRA/QLoRA, que entrena un adaptador pequeño en lugar de actualizar cada peso. Para modelos de 7B y superiores, el ajuste fino completo requiere 24 GB+ de memoria de GPU y no cabe en una tarjeta de consumo de 16 GB. Para modelos de 3B y menos, el ajuste fino completo SÍ es posible con 16 GB; se planea una opción `mode="full"` para la versión 1.4. En resumen: investigaciones recientes ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) muestran que LoRA, con la configuración correcta, iguala la calidad del ajuste fino completo en la mayoría de las tareas posteriores al entrenamiento (seguimiento de instrucciones, adaptación de dominio, personalidad/estilo) con el 67% de la capacidad de cómputo; por lo tanto, para el trabajo que la mayoría de los usuarios realmente desean, no se pierde nada al usar LoRA. Si realmente necesita el ajuste fino completo de un modelo de 7B+, utilice `transformers.Trainer` de HuggingFace directamente en una tarjeta de 24 GB+.
- **DPO / PPO / GRPO / ajuste de preferencias** — Backpropagate solo realiza un ajuste fino supervisado en una sola etapa. Para el aprendizaje por preferencias, utilice TRL directamente o LLaMA-Factory.
- **Entrenamiento en múltiples nodos** — solo una GPU en una sola máquina. El uso de múltiples GPU en una sola máquina funciona (a través de `accelerate launch`), pero no está oficialmente soportado.
- **Entrenamiento en macOS** — Apple Silicon no tiene CUDA, por lo que el entrenamiento debe ejecutarse en una máquina con Linux o Windows con una GPU NVIDIA. Aún puede ejecutar el modelo entrenado en una Mac a través de Ollama.
- **Cualquier cosa fuera de las familias de modelos probadas** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Otros modelos a menudo funcionan, pero no están incluidos en las pruebas automatizadas.

Si necesita alguna de estas funciones, utilice una de las bibliotecas mencionadas anteriormente. Son mejores en eso.

## Lo que Backpropagate le ofrece:

Cuatro cosas, en una sola instalación:

**1. Una API real de 3 líneas que funciona sin un archivo de configuración.**
El fragmento de código que se encuentra al principio de este archivo README se ejecuta de principio a fin. No requiere `accelerate config`, ni YAML, ni anulaciones de Hydra. Simplemente `Trainer(model).train(data)` y tendrá un modelo ajustado.

**2. Funcionalidad que realmente funciona en Windows.**
La mayoría de las bibliotecas de aprendizaje automático tratan a Windows como una opción secundaria. Backpropagate se prueba exhaustivamente en Windows + RTX 5080. La biblioteca se encarga de las peculiaridades del entorno de ejecución: sabe cómo pre-tokenizar sus datos para que el procesamiento multiproceso de Windows no se bloquee, desactiva automáticamente xformers en las tarjetas RTX 40/50 donde podría fallar, y selecciona la configuración del cargador de datos que no causa problemas. No tiene que saber nada de esto. Simplemente funciona.

**3. Diseñado para ejecuciones no supervisadas.**
El entrenamiento lleva horas. No quiere tener que vigilarlo constantemente. Backpropagate está diseñado para que se pueda dejar funcionando:

- Si se queda sin memoria de GPU, reduce automáticamente el tamaño del lote e intenta de nuevo, hasta tres veces. No requiere ajuste manual.
- Si la GPU se calienta demasiado, se pausa hasta que se enfría y luego continúa.
- Cada punto de control se guarda de forma atómica: si su computadora portátil se bloquea durante el guardado, el punto de control anterior y válido sigue intacto.
- Cada ejecución de entrenamiento recibe un ID único que se imprime en cada línea de registro, cada punto de control y cada entrada de Weights & Biases. Si algo sale mal, un solo ID permite a un desarrollador correlacionar todo.
- Los errores vienen con códigos estables (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, etc.) para que pueda buscar en sus registros y en la [guía de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para encontrar la solución. Los fallos específicos de CUDA tienen una [página de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) dedicada.

**4. Un solo comando para pasar del adaptador entrenado a `ollama run`.**
Muchas bibliotecas entrenan un modelo. Pocas de ellas le permiten continuar una vez que realmente quiere usarlo. Backpropagate exporta a GGUF (el formato que utiliza Ollama) y registra un modelo de Ollama con un solo comando. Pasa de "entrenamiento completado" a "puedo chatear con mi modelo ajustado" en aproximadamente 30 segundos.

## Cómo empezar

El repositorio incluye un conjunto de datos de ejemplo pequeño para que el fragmento de código que se muestra al principio de este archivo README se ejecute correctamente en una instalación limpia:

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

Esto entrena un adaptador Qwen 2.5 de 7B con 5 conversaciones cortas en formato ShareGPT, y luego exporta el resultado a GGUF. Para sus propios datos, formatee su archivo JSONL con un ejemplo por línea:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

También funcionan los formatos de Alpaca (`instruction` / `output`), OpenAI chat (`messages`) y texto sin formato; Backpropagate detecta automáticamente el formato.

Para flujos de trabajo más completos (ajuste fino y publicación en Hugging Face Hub, reanudación después de errores de memoria, ejecuciones múltiples de SLAO durante una campaña larga, etc.), consulte la [página de recetas del manual](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interfaz web (opcional)

Si prefiere hacer clic en lugar de escribir código Python, instale el paquete de la interfaz de usuario y ejecútelo:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Se abre una interfaz web local en `http://localhost:7862`, donde puede seleccionar un conjunto de datos, elegir un modelo, entrenar y exportar. La interfaz de usuario solo funciona localmente de forma predeterminada. Para permitir el acceso desde otros dispositivos, consulte la sección [Interfaz web](#web-ui) a continuación para obtener información sobre el contrato de seguridad `--share` + `--auth`.

## Entrenamiento con múltiples ejecuciones

Si desea realizar un ajuste fino incremental en varios conjuntos de datos, por ejemplo, si recibe nuevos datos de entrenamiento cada semana y desea agregarlos sin olvidar lo que aprendió antes, el modo `multi_run` de Backpropagate es para usted:

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

Esto realiza cinco pasadas de entrenamiento, fusionando el adaptador entre cada ejecución de una manera que preserva el conocimiento previo al tiempo que incorpora nuevos ejemplos. Esta técnica se basa en investigaciones recientes sobre aprendizaje continuo; consulte la sección [Referencias](#references) al final de este archivo README.

La versión de la línea de comandos:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Reanudar desde un punto de control

Una ejecución de entrenamiento de 5 pasadas que se interrumpe en la pasada 4 se puede recuperar. Cada sesión de entrenamiento con múltiples ejecuciones escribe su ID de ejecución en el historial y el manifiesto del punto de control almacenados en el disco, por lo que para reanudar desde donde lo dejó, solo necesita un comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

El comportamiento predeterminado de `backprop multi-run` (sin `--resume`) detecta automáticamente una ejecución en curso en el mismo directorio de salida y la continúa. Para forzar un inicio limpio, especifique un directorio de salida nuevo.

## Historial de entrenamiento

Cada invocación de `backprop train` y `backprop multi-run` registra una fila en `<output>/run_history.json`, que incluye el modelo utilizado, el conjunto de datos, los hiperparámetros, el estado, la pérdida final y el historial de pérdidas. Puede enumerar e inspeccionar ejecuciones anteriores:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Seguimiento de experimentos

Backpropagate detecta automáticamente los sistemas de seguimiento de experimentos instalados (Weights & Biases, TensorBoard, MLflow) y los integra. Si `wandb` está instalado y ha iniciado sesión, cada ejecución se registra automáticamente en W&B con un nombre de ejecución que coincide con el ID de ejecución almacenado en el disco, lo que le permite buscar en W&B, sus registros y `run_history.json` utilizando un único identificador.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Puede anular este comportamiento con `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` o `Trainer(report_to="none")` para desactivarlo.

## Interfaz web

La interfaz web de Reflex es opcional; instálela con `pipx install "backpropagate[ui]"` y ejecútela:

```bash
backprop ui --port 7862
```

La interfaz de usuario se ejecuta localmente en `http://localhost:7862`. Para permitir el acceso desde otros dispositivos (otras personas en su red, una URL pública, etc.), debe combinar `--share` (o `--host`) con `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` sin `--auth` genera un error. La razón es que `--share` publica una URL a la que cualquier persona en Internet puede acceder, y sin autenticación, eso significa que cualquier persona puede controlar su canal de entrenamiento y leer su token de Hugging Face. No hay forma de desactivar esta función; si no desea establecer credenciales, utilice el reenvío de puertos SSH en su lugar:

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

Consulte [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) para obtener información detallada sobre el modelo de amenazas.

Las escrituras en el sistema de archivos desde la interfaz de usuario están restringidas a un solo directorio:

- Predeterminado: `~/.backpropagate/ui-outputs`
- Para sobrescribir: establezca `BACKPROPAGATE_UI__OUTPUT_DIR=/ruta/propia`
- La sobrescritura se valida mediante una lista de denegación; se rechazan las rutas del sistema o de credenciales (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.).

## Notas sobre la plataforma

**Requisitos:** Python 3.10+ · GPU con CUDA (8GB+ de VRAM) · PyTorch 2.0+

Python 3.10 llegará al final de su vida útil en octubre de 2026, y Backpropagate planea eliminar el soporte para 3.10 en la versión 1.4. Para nuevas instalaciones, prefiera Python 3.11 o 3.12; 3.11 es la versión más probada.

Backpropagate gestiona las peculiaridades del entorno de ejecución al entrenar en diferentes plataformas, pero no puede solucionar problemas de instalación. Los dos problemas más comunes son:

- **Controlador CUDA incorrecto.** PyTorch se publica con un binario por cada versión de CUDA. Si elige el incorrecto, obtendrá una versión de PyTorch solo para CPU, y el entrenamiento será extremadamente lento. Utilice el selector de controladores en <https://pytorch.org/get-started/locally/> para su controlador. Ejecute `nvidia-smi` para ver la versión de su controlador/CUDA.
- **Windows + exportación GGUF.** La opción `[export]` compila `llama-cpp-python` desde el código fuente, lo que requiere Visual Studio Build Tools (componente C++) y CMake.

**macOS:** El entrenamiento con GPU no está soportado (no hay CUDA). Puede ejecutar el adaptador entrenado en un Mac a través de Ollama, pero `trainer.train()` genera un error `DEP_GPU_NOT_AVAILABLE`. Utilice una máquina con Linux o Windows con CUDA para el entrenamiento.

Consulte la [página del manual de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para obtener una guía completa sobre la solución de problemas de instalación, y la [página de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) dedicada para problemas de controlador/VRAM/xformers/bf16-vs-fp16.

## Interfaz de línea de comandos

Cada API de Python tiene un equivalente de línea de comandos (CLI):

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

Consulte la referencia completa en [la página de referencia del CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), o `backprop <subcomando> --help`.

## Configuración

Cada configuración se puede sobrescribir con una variable de entorno utilizando el prefijo `BACKPROPAGATE_`:

| Variable | Valor predeterminado | Notas |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | automático | Forzar registros JSON o de consola |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modelo predeterminado |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Tasa de aprendizaje |
| `BACKPROPAGATE_LORA__R` | `256` | Rango LoRA (valor predeterminado de la versión 1.3; use `--lora-preset=fast` para el valor predeterminado de la versión 1.2.x, que es 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sistema de archivos de la interfaz de usuario (sandbox) |

Las claves anidadas utilizan doble guion bajo (`MODEL__NAME`, no `MODEL_NAME`). La referencia completa se encuentra en [la página del manual de variables de entorno](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Preajustes de modelos

| Preajuste | VRAM | Licencia | Notas |
|---|---|---|---|
| Qwen-3.5-4B | ~8GB | Apache 2.0 | Valor predeterminado recomendado para modelos de menos de 5B. La mejor calidad para este tamaño. |
| Phi-4-mini-3.8B | ~8GB | MIT | Excelente para razonamiento, matemáticas y código. Licencia limpia. |
| SmolLM3-3B | ~6GB | Apache 2.0 | Receta completamente abierta. Contexto nativo de 64K. |
| Qwen 2.5 7B | ~12GB | Apache 2.0 | Valor predeterminado existente. La mejor calidad de los modelos heredados de 7B. |
| Qwen 2.5 3B | ~8GB | Qwen-Research | ⚠ licencia de investigación; consulte los términos de la licencia de Qwen antes de usarlo comercialmente. |
| Llama 3.2 3B | ~8GB | Llama Community | Una alternativa sólida a Qwen 3B con algunas restricciones. |
| Llama 3.2 1B | ~6GB | Llama Community | Ideal para experimentos rápidos en tarjetas pequeñas. |
| Mistral 7B | ~12GB | Apache 2.0 | Comparable a Qwen 7B, pero con una plantilla de chat diferente. |

Otros modelos pueden funcionar, pero solo estos ocho están incluidos en las pruebas automatizadas (CI). Utilice `--lora-preset=quality` (predeterminado) para obtener un rango de 256 / objetivos de línea recta, según Biderman 2024 + Thinking Machines 2025, o `--lora-preset=fast` para obtener el rango de 16 / objetivo q+v de la versión 1.2.x si necesita la huella de la versión 1.2.x.

## Resolución de problemas

Un índice breve de los fallos más comunes al ejecutarlo por primera vez. El índice inverso completo se encuentra en [la página del manual de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Para obtener información detallada sobre el controlador/VRAM/precisión mixta, consulte la [página de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Síntoma | Código de error | Solución |
|---|---|---|
| La GPU se queda sin memoria durante el entrenamiento. | `RUNTIME_GPU_OOM` | Automatic — Backpropagate reduce la mitad del tamaño del lote e intenta hasta 3 veces. Para desactivar esta función: `Trainer(oom_recovery=False)`. Para forzar un tamaño de lote más pequeño: `--batch-size 1`. |
| HuggingFace devuelve 401 / "modelo no encontrado" | `DEP_MODEL_LOAD_FAILED` | Ejecute `huggingface-cli login` e inténtelo de nuevo. Si hay errores tipográficos, copie el ID exacto de <https://huggingface.co/models>. |
| `register_with_ollama` rechaza la conexión. | `DEP_OLLAMA_REGISTRATION_FAILED` | Inicie el demonio: `ollama serve`. Instálelo desde <https://ollama.com>. Se puede volver a intentar. |
| El disco se llena durante el guardado del punto de control. | `STATE_CHECKPOINT_INVALID` | Las escrituras atómicas dejan un directorio `.partial` en caso de fallo; es seguro eliminarlo. El punto de control anterior y válido está intacto. |
| Entrenamiento pausado debido a sobrecalentamiento de la GPU. | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automatic — Backpropagate pausa cuando se alcanza el umbral de temperatura y reanuda cuando la GPU se enfría. Mejore el flujo de aire si esto ocurre con frecuencia. |
| `backprop ui --share` es rechazado. | `INPUT_AUTH_REQUIRED` | Utilice `--auth user:password` o, en su lugar, use el reenvío de puertos SSH (consulte [Interfaz de usuario web](#web-ui)). |
| La exportación a GGUF falló en el primer intento. | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; en Windows, también necesita Visual C++ Build Tools + CMake. |

## Informar de errores

Cuando algo falla, Backpropagate imprime una línea al inicio, como `run_started run_id=<uuid>`, y asocia el mismo ID a cada línea de registro, a cada punto de control y a cada entrada de Weights & Biases. **Incluya el `run_id` en cualquier informe de error**; esto permite al responsable del mantenimiento correlacionar todo para esa ejecución específica.

Un buen informe de error incluye:

1. **El `run_id`**: el UUID que se imprime al inicio.
2. **El código de error**: la línea `[CODE_NAME]: message` en stderr. Consulte [códigos de error](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) para ver el catálogo.
3. **La línea de comandos con información sensible eliminada**. El stderr se elimina automáticamente (los tokens Bearer, `sk-*`, `hf_*`, las claves de AWS, los pares `password=` / `token=` se eliminan), por lo que es seguro pegarlo. Para obtener el rastreo completo y sin eliminar información, vuelva a ejecutar con `--verbose`, pero revise antes de publicarlo.
4. **Versiones de Python / PyTorch, modelo de GPU, sistema operativo**. `backprop info` imprime todo esto de una vez.

Las preguntas, ideas o discusiones sobre si algo es "esperado" deben realizarse en [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Los problemas de seguridad deben informarse de forma privada a través del formulario [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new); consulte [SECURITY.md](SECURITY.md) para ver la política.

## Privacidad

Todo el entrenamiento se realiza localmente en su GPU. Backpropagate no realiza solicitudes de red excepto para descargar modelos de HuggingFace (lo cual usted inicia). No hay telemetría, ni dependencia de la nube.

## Referencias

Los valores predeterminados de Backpropagate y el modo de entrenamiento con múltiples ejecuciones se basan en investigaciones recientes. Si está interesado en las técnicas subyacentes:

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — el artículo fundamental que introduce LoRA, que es la forma en que Backpropagate entrena adaptadores de manera eficiente.
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidencia empírica de que LoRA con un rango de 256 y objetivos totalmente lineales coincide con la calidad del ajuste fino completo en la mayoría de las tareas posteriores al entrenamiento, utilizando el 67% de la capacidad de cómputo. Define la configuración predeterminada de LoRA de la versión 1.3 de Backpropagate.
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — el seguimiento práctico que identifica la corrección de 10 veces la tasa de aprendizaje con respecto al ajuste fino completo necesaria para rangos LoRA altos.
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la caracterización original de por qué las redes neuronales "olvidan" el entrenamiento anterior cuando se ajustan a nuevos datos (EWC — Elastic Weight Consolidation).
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, un enfoque anterior para usar LoRA para el aprendizaje continuo, restringiendo los nuevos adaptadores a subespacios ortogonales.
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — una técnica fundamental para fusionar múltiples modelos ajustados sin interferencia.
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — el algoritmo específico que implementa el fusionador de múltiples ejecuciones de Backpropagate. Un preprint de diciembre de 2025; Backpropagate es el primer adoptador conocido en un entorno de producción.

## Licencia

MIT — consulte [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
