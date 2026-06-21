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

# Ajusta un modelo QLoRA de 32B o un modelo completo de 7B en una sola GPU. Luego, intégralo en Ollama

Realiza el ajuste fino de modelos de lenguaje grandes en una **única** GPU, dimensionada según la tarjeta que realmente tienes. Tres líneas de código Python para ajustar un modelo QLoRA de 7B a 34B en una sola tarjeta de consumo de 32 GB (RTX 5090); una opción: `--full-ft-offload` — realiza el ajuste fino completo de un modelo de la clase 7B descargando el estado del optimizador a la RAM del host. Un comando más exporta a Ollama y, luego, `ollama run` ejecuta tu modelo ajustado. Se adapta fácilmente hasta los 16 GB. Funciona perfectamente en Windows.

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

Eso es todo. No hay un archivo de configuración YAML. No hay una "ceremonia" de `accelerate launch`. No hay un tutorial separado de "ahora conviértelo a GGUF". Si tienes una GPU CUDA y un archivo JSONL con tus datos de entrenamiento, estás a solo tres líneas de distancia de un modelo ajustado funcional.

## Instala

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

Si deseas las funciones opcionales, reemplaza la instalación por una de estas:

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

¿Prefieres Docker? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` también funciona. Se proporcionan imágenes para `linux/amd64` y `linux/arm64`, por lo que los usuarios de Apple Silicon y ARM Linux obtienen una imagen nativa. Un archivo `compose.yaml` canónico para "UI en un contenedor" se encuentra en la raíz del repositorio; `docker compose up` activa la interfaz de usuario web en `http://localhost:7860` con un volumen persistente `~/.backpropagate`.

## Dónde se ubica Backpropagate

Existen varias bibliotecas buenas para el ajuste fino de LLM. Cada una es excelente para diferentes cosas:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**: si te gustan las configuraciones YAML y deseas una comunidad de recetas de las que puedas copiar.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**: si deseas DPO/PPO/RLHF y una GUI web.
- **[Unsloth](https://github.com/unslothai/unsloth)**: si necesitas el entrenamiento más rápido posible y utilizas una familia de modelos compatible.
- **[torchtune](https://github.com/pytorch/torchtune)**: si deseas las recetas nativas de PyTorch de Meta que puedes editar.

Backpropagate es la opción que faltaba: **una API de Python de 3 líneas para usuarios individuales en una sola GPU de consumo que desean entrenar un adaptador y enviarlo.** Sin YAML, sin GUI, sin RL en línea (PPO/GRPO), sin multi-nodo. Solo el ciclo que realmente necesita todo el mundo y el paso de exportación que dificulta las cosas.

Si probaste una de las bibliotecas anteriores y te frustraste con la "ceremonia" del archivo de configuración, o te encontraste con una limitación de la familia de modelos, o querías opciones predeterminadas para Windows, Backpropagate es para ti.

## Qué puedes ajustar en una sola GPU

Backpropagate dimensiona la ejecución según tu tarjeta. Aquí tienes el rango práctico en una GPU de consumo de **32 GB** (RTX 5090) con 64 GB de RAM del host: este es el equipo en el que se realiza el ajuste:

| Tamaño del modelo | Método | Estado en una tarjeta de 32 GB |
|---|---|---|
| 7B (Qwen 2.5 7B / Llama-3.1-8B / Mistral 7B) | QLoRA | Cómodo: ~7–8 GB. Longitud completa de la secuencia, mucho margen. |
| **14B** (Qwen2.5-14B) | QLoRA | **El punto óptimo para el uso diario: ~8,5 GB**, según lo medido. rank/alpha 32, paged 8-bit AdamW, 4096 ctx. |
| 24B (Mistral-Small-24B) | QLoRA | ~18 GB. Encaja con margen en 4096 ctx. |
| **32B** (Qwen2.5-32B) | QLoRA | **Apenas encaja: ~26 GB** con `max_len 2048` + paged 8-bit AdamW. Límite máximo. |
| ≤6B | `mode="full"` (ajuste fino completo) | Ajuste fino completo solo en GPU: pesos bf16, sin adaptador. El límite consciente de la tarjeta es de 6B en 32 GB. |
| **Clase 7B** (Qwen 2.5 7B / Llama-3.1-8B / Mistral 7B) | `mode="full" --full-ft-offload` | **Ajuste fino completo mediante la descarga a la CPU FSDP2:** descarga los parámetros y el optimizador a 64 GB de RAM del host. Más lento (limitado por el ancho de banda); Linux/WSL2. |

Dos cosas para las que la mayoría de las bibliotecas de una sola GPU te dirigen a otro lugar: **QLoRA de 24–34B** y **ajuste fino completo en una sola tarjeta de la clase 7B**. Backpropagate lo hace en una sola tarjeta de consumo y, luego, exporta el resultado directamente a Ollama.

**El límite del ajuste fino completo es consciente de la tarjeta.** Se deriva de la aritmética de memoria de entrenamiento de 4 sumandos (pesos + gradientes + optimizador + activaciones) en relación con tu VRAM *detectada*: **16 GB → 4B, 24 GB → 5B, 32 GB → 6B** solo en GPU. `--full-ft-offload` lo amplía a la **clase 7B** descargando los parámetros y el estado del optimizador a la RAM del host mediante FSDP2 `fully_shard` + `CPUOffloadPolicy` (más lento, limitado por el ancho de banda PCIe/CPU; requiere ~64 GB de RAM del host y un backend NCCL, es decir, Linux/WSL2). Anula explícitamente el límite con `--full-ft-ceiling-billions`. Un modelo que supere incluso el límite de descarga finalizará con `RUNTIME_FULL_FT_MODEL_TOO_LARGE`, indicando la solución (`--full-ft-offload` o LoRA/QLoRA). Consulta [la página completa del manual de ajuste fino](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/) para ver las matemáticas de VRAM y la comparación de calidad de Biderman 2024 / Thinking Machines 2025.

### Se adapta a 16 GB

El rango de 16 GB (RTX 4080 / 5080 / 4070 Ti Super) sigue siendo excelente: QLoRA de 7B en ~7–8 GB y un verdadero ajuste fino completo de un modelo genuino de ~3B (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B/1B) dentro de 16 GB mediante `mode="full"` (pesos bf16 + checkpointing de gradiente + paged 8-bit AdamW). El mismo código selecciona el tamaño del lote y el límite de ajuste fino completo que se ajustan a la tarjeta que detecta; no hay opciones para cambiar entre equipos.

La cuantización de 2 bits (AQLM / QuIP#) queda **fuera del alcance**: una base de 2 bits no se puede fusionar limpiamente con los pesos de precisión completa, lo que interrumpe el contrato de adaptador fusionable → GGUF → Ollama (el objetivo principal de la canalización). En su lugar, Backpropagate ofrece las opciones: QLoRA, `mode="full"`, `--full-ft-offload` y la ruta de cálculo FP8 (`--fp8`, Blackwell/Hopper); todas ellas siguen siendo fusionables y exportables.

## Para lo que NO sirve Backpropagate

Si tu caso de uso es el siguiente, te irá mejor con una biblioteca diferente; Backpropagate no es la opción correcta e intentar que funcione costaría más que simplemente utilizar la herramienta adecuada. Leer esta sección antes de comenzar te ahorrará el ciclo de instalación y abandono:

- **Ajuste fino con todos los parámetros más allá del límite de descarga (≈13B+)** — Propague hacia atrás ajustes finos completos hasta **~6B en GPU pura y ~7B mediante `--full-ft-offload`** en una tarjeta de 32 GB (vea [el rango](#qué-puede-ajustar-en-una-sola-GPU)). Un ajuste fino *verdaderamente completo* de un modelo de 13B+ va más allá de eso; requiere FSDP multi-GPU o una tarjeta más grande (utilice `transformers.Trainer` en varias GPU, o alquile una A100/H100). Sin embargo, antes de invertir esos recursos computacionales: investigaciones recientes ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) muestran que LoRA, con la configuración correcta, iguala la calidad del ajuste fino completo en la mayoría de las tareas posteriores al entrenamiento (seguimiento de instrucciones, adaptación de dominio, personalidad/estilo) con aproximadamente el 67% de los recursos computacionales; por lo tanto, QLoRA hasta 34B, que Backpropagate realiza en una sola tarjeta, no pierde nada para el trabajo que la mayoría de los usuarios realmente desean.
- **RL en línea: PPO / GRPO / RLVR** — Backpropagate realiza un ajuste fino SFT de una sola etapa más un ajuste de preferencias sin referencia (ORPO en v1.5; SimPO + KTO en v1.6). Lo que *no* hace es el aprendizaje por refuerzo en línea: PPO, GRPO o RLVR, lo cual requiere un modelo de recompensa o un ciclo de generación y puntuación además del paso de entrenamiento. Para estos casos, utilice TRL directamente o LLaMA-Factory. (El ajuste de preferencias sin referencia se ajusta al rango de una sola etapa porque no hay un modelo de referencia separado que mantener en la memoria; vea la nota sobre ORPO en [Inicio rápido](#inicio-rápido)).
- **Entrenamiento multi-nodo** — solo GPU única en una máquina. El uso de varias GPU en una misma máquina funciona (mediante `accelerate launch`), pero no está oficialmente soportado.
- **Entrenamiento en macOS con CUDA** — Apple Silicon no tiene CUDA, por lo que la ruta de CUDA se ejecuta en un equipo Linux o Windows con una GPU NVIDIA. Aún puede ejecutar el modelo entrenado en un Mac a través de Ollama. Una ruta MLX **experimental y sin verificar** (`--backend mlx`) entrena un adaptador LoRA de forma nativa en Apple Silicon; vea [Apple Silicon (MLX)](#apple-silicon-mlx--vista-previa-sin-verificar). Solo es para LoRA-SFT y **no ha sido verificado en hardware real** (sin soporte), por lo que, para cualquier cosa más allá de un LoRA SFT (ORPO, ajuste fino completo, FP8, ejecución múltiple), utilice la ruta CUDA.
- **Cualquier modelo fuera de las familias probadas** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Otros modelos a menudo funcionan, pero no están incluidos en las pruebas de CI.

Si necesita alguna de estas cosas, utilice una de las bibliotecas enumeradas anteriormente. Son mejores para ello.

## Lo que ofrece Backpropagate

Cuatro cosas, en una sola instalación:

**1. Una API real de 3 líneas que se ejecuta sin un archivo de configuración.**
El fragmento de código que aparece al principio de este archivo README se ejecuta de principio a fin. No hay `accelerate config`, ni YAML, ni reemplazos de Hydra. Simplemente `Trainer(model).train(data)` y ya tiene un ajuste fino.

**2. Windows que realmente funciona.**
La mayoría de las bibliotecas de aprendizaje automático tratan a Windows como algo secundario. Backpropagate se prueba de forma nativa en Windows + RTX 5080. La biblioteca gestiona las peculiaridades del tiempo de ejecución por usted: sabe cómo pre-tokenizar sus datos para que el procesamiento en paralelo de Windows no falle, desactiva automáticamente xformers en las tarjetas RTX 40/50 donde esto provocaría un error y elige la configuración del cargador de datos que no causa problemas. No tiene que saber nada de esto. Simplemente funciona.

**3. Diseñado para ejecuciones sin supervisión.**
El entrenamiento lleva horas. No quiere tener que vigilarlo. Backpropagate está diseñado para que se pueda dejar funcionando:

- Si se queda sin memoria de GPU, reduce automáticamente a la mitad el tamaño del lote y lo vuelve a intentar, hasta tres veces. No requiere ajustes manuales.
- Si su GPU se calienta demasiado, se detiene hasta que las cosas se enfríen y luego continúa.
- Cada punto de control se escribe de forma atómica: si su portátil se bloquea en medio del guardado, el punto de control anterior y válido sigue intacto.
- Cada ejecución de entrenamiento obtiene un ID único que se estampa en cada línea del registro, en cada punto de control y en cada entrada de Weights & Biases. Si algo sale mal, un solo ID permite a un mantenedor correlacionar todo.
- Los errores vienen con códigos estables (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, etc.) para que pueda buscar en sus registros y en la [guía de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para encontrar la solución. Los fallos específicos de CUDA tienen una [página de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) dedicada.

**4. Un solo comando, desde el adaptador entrenado hasta `ollama run`.**
Muchas bibliotecas entrenan un modelo. Pocas le facilitan el uso real. Backpropagate exporta a GGUF (el formato que utiliza Ollama) y registra un modelo de Ollama en un solo comando. Pasa de "entrenamiento completado" a "puedo chatear con mi modelo ajustado" en unos 30 segundos.

## Guía de inicio rápido

El repositorio incluye un pequeño conjunto de datos de ejemplo para que el fragmento de código que aparece al principio de este archivo README se ejecute en una instalación limpia:

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

Esto entrena un adaptador Qwen 2.5 de 7B con 5 conversaciones cortas en formato ShareGPT y, a continuación, exporta el resultado a GGUF. Para sus propios datos, formatee su JSONL con un ejemplo por línea:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Los formatos Alpaca (`instruction` / `output`), OpenAI chat (`messages`) y texto sin formato también funcionan: Backpropagate detecta automáticamente el formato.

### Ajuste de preferencias (ORPO, SimPO, KTO)

Novedad en la v1.5: entrene con preferencias en lugar de demostraciones simples. ORPO no requiere referencia y es de una sola etapa: integra la señal de preferencia en el paso de ajuste fino supervisado, por lo que no hay un modelo de recompensa o de referencia separado y la forma de 3 líneas se mantiene. Pase `--method orpo` (CLI) o `method="orpo"` (Python) y proporcione un conjunto de datos de filas `{prompt, chosen, rejected}` (o simplemente `{chosen, rejected}`):

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

La tasa de aprendizaje predeterminada se reduce automáticamente a `8e-6` para ORPO (la pérdida es más pronunciada que en el SFT simple); ajuste `--orpo-beta` (valor predeterminado: `0.1`) para ponderar la penalización de la razón de probabilidades. ORPO solo funciona con `mode="lora"`.

**Novedad en la versión 1.6: SimPO y KTO.** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) no requiere referencia, utiliza una recompensa normalizada por longitud y toma los mismos datos emparejados `{prompt, chosen, rejected}` que ORPO (`--simpo-beta`, `--simpo-gamma`). `--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) utiliza datos **no emparejados** `{prompt, completion, label}` — valoraciones positivas/negativas por ejemplo — para la gran clase de comentarios que no son pares A/B seleccionados; equilibra automáticamente los pesos deseables/indeseables de la pérdida a partir del recuento de etiquetas. Ambos solo funcionan con `mode="lora"` y se mantienen dentro del ámbito SFT de una sola GPU (sin un modelo de referencia separado). Consulte el [manual de ajuste de preferencias](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) para saber cuál utilizar. Para RL en línea (PPO/GRPO), consulte [Qué NO es Backpropagate](#what-backpropagate-is-not-for).

### SFT de trazado de razonamiento (destilación R1)

Nuevo en v1.5: destile un modelo de razonamiento de forma sencilla. Pase `--reasoning-trace` (CLI) o `Trainer(..., reasoning_trace=True)` (Python) y proporcione trazas que mantengan una cadena de pensamiento `<think>...</think>` dentro del turno del asistente; la mitad de SFT puro de la destilación de [DeepSeek-R1](https://arxiv.org/abs/2501.12948), no se requiere RL. La retropropagación mantiene `<think>` en el objetivo de entrenamiento, elimina las trazas vacías o demasiado largas (filtrado de la longitud de la traza) y aumenta el valor predeterminado de `max_seq_length` a 8192 para la cadena de pensamiento más larga. Lo más importante es que `<think>` permanece como **texto sin formato**; no hay tokens especiales, no se cambia el tamaño del embedding; por lo tanto, el GGUF combinado aún se exporta a Ollama como cualquier otro ajuste fino. Solo SFT. Consulte la [receta de trazado de razonamiento](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) para conocer el formato del conjunto de datos y el rango de tokens ajustable.

### Apple Silicon (MLX) — vista previa sin verificar

> ⚠️ **Vista previa sin verificar: no forma parte del conjunto de funciones compatibles.** La ruta MLX se construye y se prueba a nivel de unidad, pero **no** ha sido verificada en hardware Apple Silicon real (`mlx-lm` solo es para Apple y no puede ejecutarse en los equipos NVIDIA en los que se desarrolla Backpropagate). Considere todo lo siguiente como experimental, úselo bajo su propio riesgo y [informe sobre cualquier anomalía](#informar-sobre-errores) si lo ejecuta en un Mac de la serie M.

Nuevo en v1.5: **una API, dos vías**. CUDA sigue siendo el backend canónico y verificado; MLX es una segunda vía que se entrena en un Mac de la serie M a través del conjunto de herramientas [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) de Apple (memoria unificada, sin CUDA). La misma estructura de 3 líneas selecciona la vía según el hardware: `backend='auto'` (el valor predeterminado) dirige el flujo a CUDA en NVIDIA y a MLX en Apple Silicon, por lo que los sistemas CUDA existentes son idénticos a nivel de bytes.

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

En v1.5, la vía MLX es **solo LoRA SFT**; no hay ORPO, no hay FP8, no hay `mode='full'`, no hay ejecución múltiple en MLX todavía (cada una se rechaza con `CONFIG_INVALID_SETTING`; use `backend='cuda'`/`'auto'` en un sistema NVIDIA para esas opciones). El adaptador resultante es un archivo safetensors simple y se exporta a Ollama a través de la misma ruta que la vía CUDA.

> Forzar `--backend mlx` en un host que no sea Apple genera el error `CONFIG_INVALID_SETTING`; la falta de una cadena de herramientas `mlx_lm` en un Mac genera `DEP_MLX_UNAVAILABLE`.

Para obtener más flujos de trabajo de principio a fin (ajuste fino y carga en HF Hub, reanudación después de que se agota la memoria, SLAO de ejecución múltiple en una campaña larga, etc.), consulte la [página de recetas del manual](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interfaz de usuario web (opcional)

Si prefiere hacer clic en lugar de escribir en Python, instale el paquete adicional de la interfaz de usuario y ejecute:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Se abre una interfaz web local en `http://localhost:7862` para explorar conjuntos de datos, validar formatos y ensamblar una configuración de entrenamiento visualmente. El entrenamiento en sí se ejecuta a través de `backprop train` (el entrenamiento impulsado por la interfaz de usuario está en la hoja de ruta; el botón Iniciar actualmente muestra esa nota). La interfaz de usuario es solo local de forma predeterminada. Para exponerla a otros dispositivos, consulte [Interfaz de usuario web](#web-ui) a continuación para conocer el contrato de seguridad `--share` + `--auth`.

## Entrenamiento de ejecución múltiple

Si desea realizar un ajuste fino de forma incremental en varios conjuntos de datos (por ejemplo, si obtiene nuevos datos de entrenamiento cada semana y desea agregarlos sin olvidar lo que aprendió antes), el modo `multi_run` de Backpropagate es para usted:

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

Esto ejecuta cinco pases de entrenamiento, fusionando el adaptador entre ejecuciones de una manera que preserva el conocimiento anterior al tiempo que incorpora nuevos ejemplos. La técnica se basa en investigaciones recientes sobre el aprendizaje continuo; consulte [Referencias](#references) al final de este archivo README.

La versión de la línea de comandos:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Reanudar desde un punto de control

Un entrenamiento de 5 ejecuciones que falla en la ejecución 4 se puede recuperar. Cada sesión de ejecución múltiple escribe su ID de ejecución en el historial y el manifiesto de puntos de control en el disco, por lo que reanudar donde lo dejó es un solo comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

El comportamiento predeterminado de `backprop multi-run` (sin `--resume`) detecta automáticamente una entrada en curso en el mismo directorio de salida y la continúa. Para forzar un inicio limpio, apunte a un directorio de salida nuevo.

## Historial de entrenamiento

Cada invocación de `backprop train` y `backprop multi-run` registra una fila en `<output>/run_history.json`: modelo utilizado, conjunto de datos, hiperparámetros, estado, pérdida final, historial de pérdidas. Puede enumerar e inspeccionar ejecuciones anteriores:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Seguimiento de experimentos

Backpropagate detecta automáticamente los rastreadores de experimentos instalados (Weights & Biases, TensorBoard, MLflow) y los conecta. Si `wandb` está instalado y ha iniciado sesión, cada ejecución registra automáticamente en W&B con un nombre de ejecución que coincide con el ID de ejecución en el disco, por lo que puede buscar en W&B, sus registros y `run_history.json` utilizando un identificador.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Anule la configuración con `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` o `Trainer(report_to="none")` para optar por no participar.

## Interfaz de usuario web

La interfaz web de Reflex es opcional; instálela con `pipx install "backpropagate[ui]"` y ejecute:

```bash
backprop ui --port 7862
```

La interfaz de usuario se ejecuta localmente en `http://localhost:7862`. Hoy cubre la mitad de **explorar / validar / configurar** del flujo de trabajo: apunte a un conjunto de datos, verifique el formato y las estadísticas detectados automáticamente, elija un modelo y cree una configuración de ejecución. **El lanzamiento de la ejecución se realiza desde la línea de comandos** (`backprop train` / `backprop multi-run`); el botón Iniciar en la interfaz de usuario muestra una nota que indica esto. El entrenamiento impulsado por la interfaz de usuario es un desarrollo futuro; hasta entonces, la interfaz de usuario es el punto de entrada y la línea de comandos es el disparador.

Para exponerlo a otros dispositivos (otras personas en su red, una URL pública, etc.), debe combinar `--share` (o `--host`) con `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` sin `--auth` finaliza con un error. La razón: `--share` publica una URL a la que puede acceder cualquier persona en Internet, y sin autenticación, esto significa que cualquiera puede ejecutar su proceso de entrenamiento y leer su token de HuggingFace. No hay opción para desactivar esto; si no desea configurar credenciales, utilice en su lugar el reenvío de puertos SSH:

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

Consulte [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) para obtener el modelo completo de amenazas.

Las operaciones de escritura en el sistema de archivos desde la interfaz de usuario se limitan a un único directorio:

- Predeterminado: `~/.backpropagate/ui-outputs`
- Anulación: establezca `BACKPROPAGATE_UI__OUTPUT_DIR=/ruta/que/desea`
- La anulación se valida mediante una lista de denegación: se rechazan las rutas del sistema o de credenciales (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.)

## Notas de la plataforma

**Requisitos:** Python 3.10+ · GPU CUDA (8 GB+ de VRAM) · PyTorch 2.0+

Python 3.10 es compatible hasta al menos la versión 1.6; su soporte oficial finaliza en octubre de 2026 y está programado para ser eliminado en la primera versión posterior a esa fecha. Para nuevas instalaciones, prefiera Python 3.11 o 3.12; 3.11 es la versión más probada.

Backpropagate gestiona las peculiaridades del tiempo de ejecución al entrenar en diferentes plataformas, pero no puede solucionar los problemas que surgen durante la instalación. Los dos más comunes son:

- **Paquete CUDA incorrecto.** PyTorch se publica con un binario por cada versión de CUDA. Si elige el incorrecto, obtendrá silenciosamente PyTorch solo para CPU y el entrenamiento será increíblemente lento. Utilice el selector de paquetes en <https://pytorch.org/get-started/locally/> para su controlador. Ejecute `nvidia-smi` para ver su versión de controlador/CUDA.
- **Windows + exportación GGUF.** La opción `[export]` construye `llama-cpp-python` a partir del código fuente, lo que requiere las herramientas de compilación de Visual Studio (componente C++) y CMake.

**macOS:** la compatibilidad con CUDA no está habilitada (no hay CUDA); un `trainer.train()` con CUDA genera `DEP_GPU_NOT_AVAILABLE`, y puede ejecutar el adaptador entrenado en un Mac a través de Ollama. **Nuevo en la versión 1.5:** una compatibilidad experimental con MLX (`--backend mlx`, `pip install 'backpropagate[mlx]'`) entrena un adaptador LoRA de forma nativa en Apple Silicon a través de `mlx_lm.lora`; solo SFT LoRA, y compilado y probado, pero aún no verificado en hardware real (consulte [Apple Silicon (MLX)](#apple-silicon-mlx--experimental-v15)). Para la ruta CUDA o para ORPO/ajuste completo/FP8/ejecuciones múltiples, utilice una máquina Linux o Windows con CUDA.

Consulte la [página de solución de problemas del manual](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para obtener una guía completa de solución de problemas de instalación, y la [página dedicada de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) para obtener información sobre problemas de controladores/VRAM/xformers/bf16 frente a fp16.

## CLI

Cada API de Python tiene un equivalente en la CLI:

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

Referencia completa en [la página del manual de la CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), o `backprop <subcomando> --help`.

## Configuración

Se puede anular cada configuración con una variable de entorno utilizando el prefijo `BACKPROPAGATE_`:

| Variable | Predeterminado | Notas |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | automático | Forzar registros en JSON o en la consola |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modelo predeterminado |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Tasa de aprendizaje |
| `BACKPROPAGATE_LORA__R` | `256` | Rango LoRA (predeterminado en v1.3; pase `--lora-preset=fast` para el valor predeterminado de v1.2.x de 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sandbox del sistema de archivos de la interfaz de usuario |

Las claves anidadas utilizan doble guion bajo (`MODEL__NAME`, no `MODEL_NAME`). La referencia completa está en [la página del manual de las variables de entorno](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Presets de modelo

| Preset | VRAM | Licencia | Notas |
|---|---|---|---|
| Qwen-3.5-4B | ~8 GB | Apache 2.0 | Predeterminado recomendado para modelos de menos de 5B. La mejor calidad para este tamaño. |
| Phi-4-mini-3.8B | ~8 GB | MIT | Fuerte en razonamiento/matemáticas/código. Licencia estrictamente limpia. |
| SmolLM3-3B | ~6 GB | Apache 2.0 | Receta totalmente abierta. Contexto nativo de 64K. |
| Qwen 2.5 7B | ~12 GB | Apache 2.0 | Predeterminado existente. La mejor calidad de los presets de 7B heredados. |
| Qwen 2.5 3B | ~8 GB | Qwen-Research | ⚠ licencia de investigación: consulte los términos de la licencia de Qwen antes de su uso comercial. |
| Llama 3.2 3B | ~8 GB | Llama Community | Una alternativa sólida a Qwen 3B con advertencias permisivas. |
| Llama 3.2 1B | ~6 GB | Llama Community | Para experimentos rápidos en tarjetas pequeñas. |
| Mistral 7B | ~12 GB | Apache 2.0 | Comparable a Qwen 7B, plantilla de chat diferente. |
| Llama-3.1-8B | ~7-8 GB (QLoRA) | Llama-3.1-Community | 8B QLoRA, 128K de contexto nativo (la cláusula de >700M de usuarios activos mensuales requiere una licencia Meta separada). |
| **Qwen2.5-14B** | ~8.5 GB (QLoRA) | Apache 2.0 | **El punto óptimo para el uso diario con 32 GB** — rango/alfa 32, AdamW de 8 bits paginado, 4096 ctx. |
| Mistral-Small-24B | ~18 GB (QLoRA) | Apache 2.0 | 24B QLoRA en una tarjeta de 32 GB con margen de 4096 ctx. |
| **Qwen2.5-32B** | ~26 GB (QLoRA) | Apache 2.0 | **Límite superior del rango de 32 GB** — apenas cabe con `max_len 2048` + AdamW de 8 bits paginado. |

Otros modelos a menudo funcionan; las filas anteriores son los ajustes preestablecidos seleccionados: el nivel de 14B–32B está ajustado con QLoRA para una tarjeta de 32 GB (el rango medido). Utilice `--lora-preset=quality` (predeterminado) para objetivos de rango 256 / todos los lineales según Biderman 2024 + Thinking Machines 2025, o `--lora-preset=fast` para el objetivo de rango 16 / q+v heredado si necesita la huella de v1.2.x.

## Solución de problemas

Un breve índice de los fallos más comunes en la primera ejecución. El índice inverso completo está en [la página del manual de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Para obtener información detallada sobre el controlador/VRAM/precisión mixta, consulte la [página de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Síntoma | Código de error | Solución |
|---|---|---|
| La GPU se queda sin memoria a mitad del entrenamiento | `RUNTIME_GPU_OOM` | Automático: la retropropagación reduce a la mitad el tamaño del lote y reintenta hasta 3 veces. Para desactivar: `Trainer(oom_recovery=False)`. Para forzar un tamaño menor: `--batch-size 1`. |
| HuggingFace devuelve 401 / "modelo no encontrado". | `DEP_MODEL_LOAD_FAILED` | Ejecute `huggingface-cli login` y vuelva a intentarlo. Para errores tipográficos, copie el ID exacto de <https://huggingface.co/models>. |
| `register_with_ollama` conexión rechazada. | `DEP_OLLAMA_REGISTRATION_FAILED` | Inicie el demonio: `ollama serve`. Instale desde <https://ollama.com>. Se puede reintentar. |
| Disco lleno durante el guardado del punto de control. | `STATE_CHECKPOINT_INVALID` | Las escrituras atómicas dejan un directorio `.partial` en caso de fallo; es seguro eliminarlo. El punto de control anterior y correcto está intacto. |
| Entrenamiento pausado debido al sobrecalentamiento de la GPU. | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automático: la retropropagación se pausa al alcanzar el umbral de temperatura y se reanuda a medida que la GPU se enfría. Mejore el flujo de aire si esto sigue ocurriendo. |
| `backprop ui --share` rechazado. | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Pase `--auth user:password` o utilice el reenvío de puertos SSH en su lugar (consulte [Interfaz de usuario web](#web-ui)). |
| La exportación a GGUF falló en el primer intento. | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; en Windows, también necesita las herramientas de compilación de Visual C++ y CMake. |

## Informar sobre errores

Cuando algo falla, Backpropagate imprime una línea al inicio, como `run_started run_id=<uuid>`, y vincula el mismo ID a cada línea del registro, a cada punto de control y a cada entrada de Weights & Biases. **Incluya el `run_id` en cualquier informe de errores**, ya que esto permite al responsable correlacionar todo para esa ejecución específica.

Un buen informe de errores incluye:

1. **El `run_id`**: el UUID impreso al inicio. Un UUID permite al responsable correlacionar cada línea del registro, cada punto de control y cada entrada de Weights & Biases para esa ejecución específica.
2. **El código de error**: la línea `[CODE_NAME]: message` en stderr. Consulte [códigos de error](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) para obtener el catálogo de códigos estables.
3. **El rastreo de pila redactado**. Stderr se redacta automáticamente en el modo no detallado (los tokens de Bearer, `sk-*`, `hf_*`, las claves de AWS y los pares `password=` / `token=` / `api_key=` se eliminan; es seguro pegarlos). Para obtener el rastreo de pila completo y no redactado, vuelva a ejecutar con `BACKPROPAGATE_DEBUG=1` (o `--verbose`); revíselo antes de publicarlo.
4. **La salida de `backprop info`**. Un comando imprime Python / PyTorch / CUDA / modelo de GPU / VRAM / SO / extras instalados: todo lo que el responsable necesita para analizar una regresión específica de la plataforma.

La [plantilla de informe de errores](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) solicita explícitamente cada uno de estos elementos para que la evaluación inicial se realice rápidamente. Las preguntas, ideas o consultas sobre si algo es "esperado" deben dirigirse a [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Los problemas de seguridad deben informarse de forma privada a través del formulario [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new); consulte [SECURITY.md](SECURITY.md) para conocer la política y los plazos de respuesta.

## Privacidad

Todo el entrenamiento se realiza localmente en su GPU. Backpropagate no realiza ninguna solicitud de red, excepto para descargar modelos de HuggingFace (lo que usted inicia). No hay telemetría ni dependencia de la nube.

## Referencias

Los valores predeterminados de Backpropagate y el modo de entrenamiento multi-ejecución se basan en investigaciones recientes. Si está interesado en las técnicas subyacentes:

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — el documento fundamental que presenta LoRA, que es la forma en que Backpropagate entrena los adaptadores de manera eficiente.
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidencia empírica de que LoRA con rango 256 y objetivos totalmente lineales coincide con la calidad del ajuste fino completo en la mayoría de las tareas posteriores al entrenamiento con el 67% de la capacidad de cómputo. Esto impulsa la configuración predeterminada de LoRA v1.3 de Backpropagate.
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora) — la continuación práctica que identifica la corrección de 10× en la tasa de aprendizaje frente al ajuste fino completo necesaria con un rango LoRA alto.
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la caracterización original de por qué las redes neuronales "olvidan" el entrenamiento anterior cuando se realiza un ajuste fino con nuevos datos (EWC: consolidación elástica del peso).
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, un enfoque anterior para utilizar LoRA para el aprendizaje continuo restringiendo los nuevos adaptadores a subespacios ortogonales.
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — una técnica fundamental para fusionar varios modelos ajustados sin interferencias.
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — el algoritmo específico que el fusionador multi-ejecución de Backpropagate implementa. Un preprint de diciembre de 2025; Backpropagate es el primer usuario conocido de este documento.

## Licencia

MIT — consulte [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
