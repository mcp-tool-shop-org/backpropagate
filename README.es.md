<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/backpropagate/readme.png" alt="Backpropagate" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/backpropagate/"><img src="https://img.shields.io/pypi/v/backpropagate" alt="PyPI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/backpropagate"><img src="https://img.shields.io/codecov/c/github/mcp-tool-shop-org/backpropagate/main" alt="Coverage"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/backpropagate"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/backpropagate/badge" alt="OpenSSF Scorecard"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

# Ajusta un modelo QLoRA de 32B o un modelo completo de 7B en una sola GPU. Luego, intégralo en Ollama

Realiza el ajuste fino de modelos de lenguaje grandes en una **única** GPU, dimensionada según la tarjeta que realmente tienes. Tres líneas de código Python para ajustar un modelo QLoRA de 7B a 32B en una tarjeta de consumo de 32 GB (RTX 5090). Con una sola opción, `--full-ft-offload`, se realiza un ajuste fino completo de un modelo de 7B manteniendo sus pesos y gradientes en la RAM del host (Linux o WSL2; es más lento y se mide a continuación). Un comando adicional exporta a Ollama y, a continuación, `ollama run` tu ajuste fino. Se reduce a 16 GB. Funciona perfectamente en Windows. ¿Prefieres un navegador a Python? `backprop ui` hace todo sin necesidad de código ([consulta el tutorial](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/)).

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")
```

```bash
backprop export ./output --format gguf --quantization q4_k_m --ollama --ollama-name my-model
ollama run my-model
```

Eso es todo. No hay un archivo de configuración YAML. No hay una ceremonia de `accelerate launch`. No hay un tutorial separado que diga "ahora conviértelo a GGUF". Si tienes una GPU CUDA y un archivo JSONL con tus datos de entrenamiento, solo necesitas tres líneas de código para obtener un ajuste fino funcional.

## Instalación

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

¿Prefieres Docker? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` también funciona. Hay imágenes disponibles tanto para `linux/amd64` como para `linux/arm64`, por lo que los usuarios de Apple Silicon y ARM Linux obtienen una imagen nativa. Un ejemplo canónico de `compose.yaml` para "UI en un contenedor" se encuentra en la raíz del repositorio: coloca `user:password` en un `ui-auth.txt` junto a él, ejecuta `docker compose up` e inicia sesión en `http://127.0.0.1:7860` (la primera ejecución construye el frontend, lo que tarda un minuto o dos). El historial de ejecución se guarda en `~/.backpropagate`.

## Dónde se ubica Backpropagate

Existen varias bibliotecas excelentes para el ajuste fino de LLM. Cada una de ellas destaca en diferentes aspectos:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**: si te gustan las configuraciones YAML y deseas una comunidad de recetas de las que puedas copiar.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**: si deseas DPO/PPO/RLHF y una GUI web.
- **[Unsloth](https://github.com/unslothai/unsloth)**: si necesitas el entrenamiento más rápido posible y utilizas una familia de modelos compatible.
- **[torchtune](https://github.com/pytorch/torchtune)**: si deseas las recetas nativas de PyTorch de Meta que puedes editar.

Backpropagate es la opción que faltaba: **una API de Python de 3 líneas para usuarios individuales en una sola GPU de consumo que desean entrenar un adaptador e integrarlo.** Sin YAML, sin RL en línea (PPO/GRPO), sin multi-nodo. Existe una interfaz de usuario de navegador para el mismo proceso si prefieres no escribir código. Solo el proceso que realmente necesita todo el mundo y el paso de exportación que dificulta las cosas.

Si has probado alguna de las bibliotecas anteriores y te has topado con la complejidad de la configuración, o has encontrado una limitación en la familia de modelos, o has preferido una configuración predeterminada para Windows, Backpropagate es para ti.

## Lo que puedes ajustar en una sola GPU

Backpropagate dimensiona la ejecución según tu tarjeta. Estos son números **medidos** en una RTX 5090 de 32 GB: las filas de QLoRA del 2026-10-03 (resultados: [`docs/receipts/2026-10-03-presets/`](docs/receipts/2026-10-03-presets/)), las filas de ajuste fino completo del 2026-09-30 (resultados: [`docs/receipts/2026-09-30-offload/`](docs/receipts/2026-09-30-offload/)). Los picos de QLoRA se producen en la ventana de contexto completa del preajuste con un lote de 1, que es el peor de los casos para ese preajuste; los ejemplos más cortos utilizan menos recursos.

| Modelo | Método | Medido en una tarjeta de 32 GB |
|---|---|---|
| **14B** (Qwen2.5-14B) | QLoRA | **18,7 GiB** de pico a 4096 de contexto (20,0 GiB reservados). |
| 24B (Mistral-Small-24B) | QLoRA | 22,8 GiB de pico a 4096 de contexto (24,2 GiB reservados). |
| **32B** (Qwen2.5-32B) | QLoRA | **Se ajusta:** 26,0 GiB de pico a 2048 de contexto (27,2 GiB reservados, aproximadamente 4 GiB de espacio libre). |
| 3B | `mode="full"` (ajuste fino completo real, en la GPU) | **22,0 GiB** de pico (a nivel del sistema), 0,30 s/paso con un lote de 4, 512 de contexto. 7,5 GiB de eso son el estado del optimizador paginado, que puede descargarse en la RAM del host en una tarjeta más pequeña (no probado). |
| **7B (clase)** (Qwen2.5-7B, 7,6B de parámetros) | `mode="full" --full-ft-offload` | **Entrena:** 5,3 GiB de VRAM, **30,8 GiB de RAM del host** (32,2 GiB al guardar), **14,7 s/paso**. Solo para Linux o WSL2. |

No se ha vuelto a medir en esa sesión: QLoRA de 7B, Llama-3.1-8B (repositorio con acceso restringido, sin token en la máquina de prueba) y ajuste fino completo en la GPU por encima de 3B. Las cifras de estos se encuentran en otros lugares de la documentación y son estimaciones.

Dos cosas para las que la mayoría de las bibliotecas de una sola GPU te dirigen a otro lugar: **QLoRA de 24 a 32B** y **ajuste fino completo de 7B en una sola tarjeta**, Backpropagate lo hace en una sola tarjeta de consumo y, a continuación, exporta el resultado directamente a Ollama.

**El ajuste fino completo tiene dos rutas.** Sin descarga, el modelo, sus gradientes y el estado del optimizador se encuentran todos en la GPU. La biblioteca limita el tamaño del modelo mediante la detección de la VRAM (**16 GB → 4B, 24 GB → 5B, 32 GB → 6B**); estos límites se obtienen mediante cálculos de memoria y solo se miden hasta 3B. Anula con `--full-ft-ceiling-billions`.

`--full-ft-offload` mantiene los pesos y los gradientes en la RAM del host y los transmite a la GPU (descarga de CPU FSDP2). Lo que cuesta, medido:

- **RAM del host:** la verificación de compatibilidad requiere aproximadamente 3,7 GiB por cada mil millones de parámetros, más 10 GiB, lo cual es un valor conservador (39 GiB con 7,6 mil millones de parámetros, frente a los 32,2 GiB medidos). Se rechaza la ejecución de inmediato si la máquina no puede soportarlo. Un modelo de 7,6 mil millones de parámetros no cabe en un límite de memoria de 28 GB de WSL2; aproximadamente 5 mil millones es el límite práctico.
- **Velocidad:** 14,7 s/paso con 7,6 mil millones de parámetros (lote 1) y 5,1 s/paso con 3 mil millones de parámetros (lote 4), frente a aproximadamente 0,63 s/paso para 3 mil millones de parámetros en la GPU con lote 4. Úselo solo cuando el modelo no quepa sin él. Se planea una versión más rápida.
- **Optimizador:** Adafactor, no AdamW. Los pesos permanecen en bf16 y cada actualización se vuelve a escribir con redondeo estocástico; no hay una copia fp32.
- **Calidad:** en una ejecución de 3 mil millones de parámetros (150 pasos, pérdida retenida, una semilla), alcanzó aproximadamente el 85% de la mejora que obtuvo el ajuste fino completo ordinario (2,45 → 1,93 frente a 2,45 → 1,84). Una sola semilla no es un punto de referencia.
- **Alcance:** ajuste fino supervisado simple. Sin empaquetamiento, sin enmascaramiento solo de respuesta, sin puntos de control intermedios, sin reanudación. Solo Linux o WSL2 (FSDP2 requiere NCCL); en Windows nativo, se detiene con `DEP_FSDP_UNAVAILABLE`.
- **Aún no probado:** ejecuciones largas, acumulación de gradiente superior a 1 y una máquina física de 64 GB (la máquina de prueba tenía más RAM, con un límite de 60 GiB impuesto por la prueba).

Un modelo que no cabe sale con `RUNTIME_FULL_FT_MODEL_TOO_LARGE` y especifica la solución. Consulte [la página completa del manual de ajuste fino](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/).

### Se reduce a 16 GB

El rango de 16 GB (RTX 4080 / 5080 / 4070 Ti Super) sigue siendo de primera clase: 7B QLoRA (el tamaño del adaptador se elige para que quepa: rango 64 en una tarjeta de 16 GB, donde un rango de 256 requiere aproximadamente 17 GB) y un verdadero ajuste fino completo de un modelo de ~3 mil millones de parámetros (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B/1B) a través de `mode="full"` (22,0 GiB medidos en una tarjeta de 32 GB con 3 mil millones de parámetros, de los cuales 7,5 GiB son el estado del optimizador paginado que puede desbordarse a la RAM del host; no se ha probado si eso se ejecuta de manera aceptable en una tarjeta de 16 GB). Con `--full-ft-offload`, la GPU contiene mucho menos: con la VRAM limitada en la tarjeta de prueba, un modelo de 3 mil millones de parámetros entrenado bajo un límite de 6 GiB y modelos de 4 mil millones y 7,6 mil millones bajo un límite de 8 GiB. Esas son limitaciones simuladas en una tarjeta de 32 GB, no ejecuciones en hardware real de 8 GB. El mismo código elige el tamaño del lote y el límite que se ajustan a la tarjeta que detecta.

La cuantificación de 2 bits (AQLM / QuIP#) queda **fuera del alcance**: una base de 2 bits no se puede combinar limpiamente con pesos de precisión completa, lo que interrumpe el contrato de exportación de adaptador combinable → GGUF → Ollama (el objetivo principal de la canalización). En cambio, Backpropagate ofrece las opciones de margen: QLoRA, `mode="full"`, `--full-ft-offload` y la ruta de cálculo FP8 (`--fp8`, Blackwell/Hopper), todas las cuales siguen siendo combinables y exportables.

## Para qué NO sirve Backpropagate

Si su caso de uso está por debajo, obtendrá mejores resultados con una biblioteca diferente: Backpropagate no es la opción correcta y tratar de hacerlo funcionar costaría más que simplemente utilizar la herramienta adecuada. Leer esta sección antes de comenzar evita el ciclo de instalación y prueba fallida:

- **Ajuste fino de parámetros completos de modelos de 13 mil millones de parámetros o más:** Backpropagate realiza un ajuste fino completo de hasta aproximadamente 6 mil millones de parámetros en una GPU de 32 GB y un modelo de 7 mil millones de parámetros con `--full-ft-offload` (consulte [el rango](#qué-puede-ajustar-en-una-sola-GPU)). Un ajuste fino completo de un modelo de 13 mil millones de parámetros requiere FSDP multi-GPU o una tarjeta más grande. Antes de invertir en esa capacidad de cómputo, evalúe la evidencia en ambos sentidos. [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/) informa que LoRA coincide con el ajuste fino completo cuando se aplica a cada capa y el conjunto de datos se ajusta a la capacidad del adaptador, con aproximadamente dos tercios de la capacidad de cómputo por pasada. [Biderman et al. 2024](https://arxiv.org/abs/2405.09673) descubrió que, en entornos de bajo rango estándar, LoRA tiene un rendimiento significativamente inferior al del ajuste fino completo en código y matemáticas, al tiempo que olvida menos. Para el seguimiento de instrucciones, el trabajo de personalidad y estilo en conjuntos de datos modestos, QLoRA de hasta 32 mil millones suele ser el mejor uso de una sola tarjeta.
- **RL en línea: PPO / GRPO / RLVR:** Backpropagate realiza un ajuste fino SFT de una sola etapa más un ajuste de preferencia sin referencia (ORPO en v1.5; SimPO + KTO en v1.6). Lo que no hace es el aprendizaje por refuerzo en línea: PPO, GRPO o RLVR, que requiere un modelo de recompensa o un bucle de generación y puntuación además del paso de entrenamiento. Para esos casos, utilice TRL directamente o LLaMA-Factory. (El ajuste de preferencia sin referencia se ajusta al rango de una sola etapa porque no hay un modelo de referencia separado que mantener en la memoria; consulte la nota de ORPO en [Inicio rápido](#inicio-rápido)).
- **Entrenamiento multi-nodo:** solo una GPU en una sola máquina. Multi-GPU en una sola máquina funciona (a través de `accelerate launch`) pero no está oficialmente admitido.
- **Entrenamiento en macOS en el entorno CUDA:** Apple Silicon no tiene CUDA, por lo que la ruta CUDA se ejecuta en una máquina Linux o Windows con una GPU NVIDIA. Aún puede ejecutar el modelo entrenado en una Mac a través de Ollama. Una ruta MLX **experimental y no verificada** (`--backend mlx`) entrena un adaptador LoRA de forma nativa en Apple Silicon; consulte [Apple Silicon (MLX)](#apple-silicon-mlx--vista-previa-no-verificada). Es solo LoRA-SFT y **no está verificado en silicona real** (sin soporte), por lo que para cualquier cosa más allá de un LoRA SFT (ORPO, ajuste fino completo, FP8, ejecución múltiple), desea la ruta CUDA.
- **Cualquier cosa fuera de las familias de modelos probadas:** Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Otros modelos a menudo funcionan, pero no están fijados en CI.

Si necesita alguna de esas cosas, utilice una de las bibliotecas enumeradas anteriormente. Son mejores para ello.

## Lo que le ofrece Backpropagate

Cuatro cosas, en una sola instalación:

**1. Una API real de 3 líneas que se ejecuta sin un archivo de configuración.**
El fragmento en la parte superior de este README se ejecuta de principio a fin. No `accelerate config`, no YAML, no anulación de Hydra. Solo `Trainer(model).train(data)` y ya tiene un ajuste fino.

**2. Windows que realmente funciona.**
La mayoría de las bibliotecas de aprendizaje automático tratan a Windows como algo secundario. Backpropagate se desarrolla y se prueba en Windows 11 con tarjetas de la serie RTX 50. La biblioteca gestiona las peculiaridades del entorno de ejecución por ti: sabe cómo pre-tokenizar tus datos para que el procesamiento en paralelo de Windows no falle, desactiva automáticamente xformers en las tarjetas RTX 40/50 donde esto causaría problemas y elige la configuración del cargador de datos que no generará errores. No tienes que saber nada de esto. Simplemente funciona.

**3. Diseñado para ejecuciones sin supervisión.**
El entrenamiento lleva horas. No quieres tener que estar pendiente de él. Backpropagate está diseñado para que se pueda dejar funcionando:

- Si se agota la memoria de la GPU, reduce automáticamente a la mitad el tamaño del lote y lo vuelve a intentar, hasta tres veces. No requiere ajustes manuales.
- Si la GPU se calienta demasiado, se detiene hasta que se enfríe y luego continúa.
- Cada punto de control se escribe de forma atómica: si tu portátil falla en medio del guardado, el punto de control anterior y válido seguirá intacto.
- Cada ejecución de entrenamiento recibe un ID único que se incluye en cada línea del registro, en cada punto de control y en cada entrada de Weights & Biases. Si algo sale mal, un solo ID permite a un mantenedor correlacionar todo.
- Los errores vienen con códigos estables (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, etc.) para que puedas buscar en tus registros y en la [guía de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para encontrar la solución. Los fallos específicos de CUDA tienen una [página de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) dedicada.

**4. Con un solo comando, desde el adaptador entrenado hasta `ollama run`.**
Muchas bibliotecas entrenan un modelo. Pocas se apartan cuando quieres usarlo realmente. Backpropagate exporta a GGUF (el formato que usa Ollama) y registra un modelo de Ollama con un solo comando. Pasas de "entrenamiento completado" a "puedo chatear con mi modelo ajustado" en unos 30 segundos.

## Guía de inicio rápido

Desde la línea de comandos, con un conjunto de datos de ejemplo de 5 conversaciones:

```bash
pipx install "backpropagate[standard]"
curl -LO https://raw.githubusercontent.com/mcp-tool-shop-org/backpropagate/main/examples/quickstart.jsonl

backprop train --data quickstart.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 10
backprop generate ./output "What is Python?"      # did it learn anything?
backprop export ./output --format gguf --quantization q4_k_m --ollama --ollama-name my-first-finetune
ollama run my-first-finetune
```

`backprop train` escribe el adaptador en `./output` (cámbialo con `--output`). En Python, lo mismo se hace así:

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")
```

Utiliza un entorno virtual con `pip install "backpropagate[standard]"` para la API de Python; `pipx` instala el comando `backprop` en su propio entorno, por lo que `import backpropagate` no lo encontrará.

**Qué necesita la exportación a GGUF.** La exportación fusiona tu adaptador con el modelo base y lo convierte con el script de conversión de llama.cpp. Necesitas:

- una copia del código fuente de llama.cpp (`git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp`) más `pip install sentencepiece protobuf` en el mismo entorno, o
- Unsloth con su propio llama.cpp ya compilado.

Con `--ollama`, la cuantización `q4_k_m` se realiza mediante `ollama create`, por lo que no es necesario compilar nada. Backpropagate nunca permite que Unsloth instale paquetes del sistema para compilar llama.cpp por ti; establece `BACKPROPAGATE_UNSLOTH_AUTO_INSTALL=1` si quieres que lo haga. Detalles: [export](https://mcp-tool-shop-org.github.io/backpropagate/handbook/export/).

Para tus propios datos, formatea tu archivo JSONL con un ejemplo por línea:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Los formatos Alpaca (`instruction` / `output`), OpenAI chat (`messages`) y texto sin formato también funcionan: Backpropagate detecta automáticamente el formato.

### El ciclo: comprueba los datos, entrena, evalúa, exporta

```bash
backprop data report my_data.jsonl                     # duplicates, length outliers, format problems
backprop data split my_data.jsonl --heldout-ratio 0.1  # a held-out set the model never trains on
backprop train --data my_data.train.jsonl --steps 200 --output ./run-a
backprop eval <run-id> --heldout my_data.heldout.jsonl # held-out loss + sample generations
backprop eval <run-b> --vs <run-a>                     # did the change help?
backprop export ./run-a --format gguf --ollama --ollama-name my-model
```

La evaluación está diseñada para que no requiera un juez: pérdida retenida más métricas de tareas deterministas (`normalized_exact_match`, `token_f1`, `contains`, `regex`, `pass_rate`). Para utilizar un juez LLM, ejecútalo tú mismo sobre la salida de `backprop generate`. Consulta las [recetas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Ajuste de preferencias (ORPO, SimPO, KTO)

Entrena con preferencias en lugar de demostraciones simples. ORPO no requiere referencias y es de una sola etapa: integra la señal de preferencia en el paso de ajuste fino (SFT), por lo que no hay un modelo de recompensa o de referencia separado y la forma de 3 líneas no cambia. Pasa `--method orpo` (CLI) o `method="orpo"` (Python) y proporciónale un conjunto de datos de `{prompt, chosen, rejected}` (o simplemente `{chosen, rejected}`) filas:

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

The default learning rate auto-lowers to `8e-6` for ORPO (the loss is sharper than plain SFT); tune `--orpo-beta` (default `0.1`) to weight the odds-ratio penalty. ORPO is `mode="lora"` only.

**New in v1.6 — SimPO and KTO.** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) is reference-free with a length-normalized reward and takes the same paired `{prompt, chosen, rejected}` data as ORPO (`--simpo-beta`, `--simpo-gamma`). `--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) takes **unpaired** `{prompt, completion, label}` data — per-example thumbs-up/down — for the large class of feedback that isn't curated A/B pairs; it auto-balances the desirable/undesirable loss weights from your label counts. Both are `mode="lora"` only and stay in the single-GPU SFT envelope (no separate reference model). See the [preference-tuning handbook](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) for which to use. For online RL (PPO/GRPO) see [What Backpropagate is NOT for](#what-backpropagate-is-not-for).

### Ajuste fino SFT de razonamiento-rastreo (destilación R1)

Destila un modelo de razonamiento de forma sencilla. Pasa `--reasoning-trace` (CLI) o `Trainer(..., reasoning_trace=True)` (Python) y proporciónale rastros que mantengan una cadena de pensamiento `<think>...</think>` dentro del turno del asistente: la mitad de la destilación SFT pura de [DeepSeek-R1](https://arxiv.org/abs/2501.12948), no se requiere aprendizaje por refuerzo. Backpropagate mantiene `<think>` en el objetivo de entrenamiento, elimina los rastros vacíos u excesivamente largos (filtrado de la longitud del rastro) y aumenta el valor predeterminado de `max_seq_length` a 8192 para el CoT más largo. Lo más importante es que `<think>` sigue siendo **texto sin formato**: no hay tokens especiales, no se cambia el tamaño del embedding, por lo que el GGUF fusionado sigue exportándose a Ollama como cualquier otro ajuste fino. Solo SFT. Consulta la [receta de razonamiento-rastreo](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) para conocer la forma del conjunto de datos y los tokens ajustables.

### Apple Silicon (MLX): vista previa no verificada

> ⚠️ **Versión preliminar no verificada: no forma parte del conjunto de funciones compatibles.** El entorno MLX está construido y se han realizado pruebas unitarias, pero **no** se ha verificado su funcionamiento en dispositivos Apple Silicon reales. `mlx-lm` solo es compatible con Apple y no se puede ejecutar en los equipos NVIDIA en los que se desarrolla Backpropagate. Considere todo lo que se muestra a continuación como experimental, úselo bajo su propio riesgo y [informe de cualquier anomalía](#reporting-bugs) si lo ejecuta en un Mac de la serie M.

**Una API, dos entornos.** CUDA es el entorno principal y verificado; MLX es un segundo entorno que se utiliza para entrenar en un Mac de la serie M a través del conjunto de herramientas [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) de Apple (memoria unificada, sin CUDA). La configuración de 3 líneas selecciona el entorno según el hardware: `backend='auto'` (el valor predeterminado) dirige el flujo a CUDA en NVIDIA y a MLX en Apple Silicon, por lo que los equipos CUDA existentes son idénticos a nivel de bytes.

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

El entorno MLX es **solo LoRA SFT**: no hay ORPO, no hay FP8, no hay `mode='full'`, no hay ejecución múltiple (cada uno se rechaza con `CONFIG_INVALID_SETTING`; utilice `backend='cuda'`/`'auto'` en un equipo NVIDIA para estas opciones). El adaptador resultante es un archivo safetensors simple y se exporta a Ollama a través de la misma ruta que el entorno CUDA.

> Forzar el uso de `--backend mlx` en un host que no sea de Apple genera un error con `CONFIG_INVALID_SETTING`; la falta del conjunto de herramientas `mlx_lm` en un Mac genera `DEP_MLX_UNAVAILABLE`.

Para obtener más flujos de trabajo completos (ajuste fino y carga en HF Hub, reanudación después de que se agote la memoria, SLAO de ejecución múltiple en una campaña larga, etc.), consulte la [página de recetas del manual](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interfaz de usuario web (opcional)

Si prefiere hacer clic en lugar de escribir en Python, instale el paquete adicional de la interfaz de usuario y ejecute:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Abra la URL que muestra, `http://127.0.0.1:7862/?token=...` (cada ejecución genera un token nuevo; la primera ejecución crea la interfaz y puede tardar uno o dos minutos). Es una interfaz web local para el entrenamiento: inicie una ejecución, una serie de ejecuciones múltiples o una exportación, observe el progreso en tiempo real (pasos, pérdida, tiempo restante, temperatura y memoria de la GPU) y deténgala con un punto de control guardado. Cada trabajo se ejecuta en su propio proceso, uno a la vez, y al volver a cargar la página se reanuda el trabajo que se estaba ejecutando. La página de Conjunto de datos muestra el contenido de un archivo, guarda una copia limpia (se eliminan las repeticiones y los ejemplos vacíos) y la pasa al formulario de entrenamiento. Las ejecuciones anteriores y los modelos en su caché de Hugging Face tienen sus propias páginas, y cada configuración tiene un icono "i" que la explica. El [recorrido por la interfaz de usuario web](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/) muestra cada página. De forma predeterminada, la interfaz de usuario es solo local. Para que sea accesible desde otros dispositivos, consulte la sección [Interfaz de usuario web](#web-ui) a continuación para conocer el contrato de seguridad `--share` + `--auth`.

## Entrenamiento con múltiples ejecuciones

Si desea realizar un ajuste fino de forma incremental en varios conjuntos de datos (por ejemplo, si recibe nuevos datos de entrenamiento cada semana y desea agregarlos sin olvidar lo que aprendió antes), el modo `multi_run` de Backpropagate es para usted:

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

Esto ejecuta cinco pases de entrenamiento, fusionando el adaptador entre ejecuciones de una manera que preserva el conocimiento anterior al tiempo que incorpora nuevos ejemplos. La técnica se basa en investigaciones recientes sobre el aprendizaje continuo; consulte la sección [Referencias](#references) al final de este archivo README.

La versión de la CLI:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Reanudar desde un punto de control

Una ejecución de entrenamiento de 5 iteraciones que falla en la iteración 4 se puede recuperar. Cada sesión de ejecución múltiple escribe el ID de la ejecución en el historial y el manifiesto de puntos de control en el disco, por lo que reanudar donde lo dejó es un solo comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

El comportamiento predeterminado de `backprop multi-run` (sin `--resume`) detecta automáticamente una entrada en curso en el mismo directorio de salida y la continúa. Para forzar un inicio limpio, apunte a un directorio de salida nuevo.

## Historial de entrenamiento

Cada invocación de `backprop train` y `backprop multi-run` registra una fila en `<output>/run_history.json`: modelo utilizado, conjunto de datos, hiperparámetros, estado, pérdida final, historial de pérdidas. Puede listar e inspeccionar las ejecuciones anteriores:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Seguimiento de experimentos

Backpropagate detecta automáticamente los rastreadores de experimentos instalados (Weights & Biases, TensorBoard, MLflow) y los integra. Si `wandb` está instalado y ha iniciado sesión, cada ejecución registra automáticamente los datos en W&B con un nombre de ejecución que coincide con el ID de ejecución en el disco, por lo que puede buscar en W&B, sus registros y `run_history.json` utilizando un único identificador.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Anule la configuración con `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` o `Trainer(report_to="none")` para desactivar esta función.

## Interfaz de usuario web

La interfaz web de Reflex es opcional: instálela con `pipx install "backpropagate[ui]"` y ejecute:

```bash
backprop ui --port 7862
```

La interfaz de usuario se ejecuta localmente: abra la URL que muestra, `http://127.0.0.1:7862/?token=...`. Sin `--auth`, cada ejecución genera un token nuevo y la interfaz de usuario rechaza las solicitudes que no lo incluyan. Desde la interfaz, puede ver y limpiar un conjunto de datos, entrenar (una sola ejecución o una ejecución múltiple), observar el progreso en tiempo real, detenerlo con un punto de control guardado y exportar el resultado. Cada trabajo se ejecuta en su propio proceso, uno a la vez, y cerrar la interfaz de usuario lo detiene. El [recorrido por la interfaz de usuario web](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/) muestra cada página con capturas de pantalla.

Para que sea accesible desde otros dispositivos (otras personas en su red, una URL pública, etc.), debe combinar `--share` (o `--host`) con `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` sin `--auth` finaliza con un error. La razón: `--share` publica una URL a la que puede acceder cualquier persona en Internet y, sin autenticación, esto significa que cualquier persona puede controlar su canalización de entrenamiento y leer su token de Hugging Face. No hay opción para desactivar esta función: si no desea configurar credenciales, utilice el reenvío de puertos SSH en su lugar:

```bash
# On the client:
ssh -L 7862:localhost:7862 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open the URL the server printed (http://127.0.0.1:7862/?token=...) locally
```

Consulte [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) para obtener el modelo de amenazas completo.

Las escrituras en el sistema de archivos desde la interfaz de usuario se limitan a un solo directorio:

- Predeterminado: `~/.backpropagate/ui-outputs`
- Anular: establecer `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- La anulación se valida mediante una lista de denegación: las rutas del sistema o de las credenciales (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) se rechazan.

## Notas sobre la plataforma

**Requisitos:** Python 3.10+ · GPU NVIDIA con CUDA · PyTorch 2.0+. Una tarjeta de 8 GB entrena los modelos preestablecidos de 1B a 3B, una tarjeta de 16 GB un modelo de 7B y una tarjeta de 32 GB hasta 32B con QLoRA.

Python 3.10 es compatible con la versión 1.6 como mínimo; su soporte oficial finaliza en octubre de 2026 y está programado para ser eliminado en la primera versión posterior. Para nuevas instalaciones, se recomienda Python 3.11 o 3.12; 3.11 es la versión más probada.

Backpropagate gestiona las peculiaridades del tiempo de ejecución al entrenar en diferentes plataformas, pero no puede solucionar los problemas que surgen durante la instalación. Los dos más comunes son:

- **Paquete CUDA incorrecto.** PyTorch se publica con un archivo binario por cada versión de CUDA. Si elige el incorrecto, obtendrá silenciosamente PyTorch solo para CPU y el entrenamiento será increíblemente lento. Utilice el selector de paquetes en <https://pytorch.org/get-started/locally/> para su controlador. Ejecute `nvidia-smi` para ver la versión de su controlador/CUDA.
- **Windows + exportación GGUF.** El comando `[export]` crea archivos adicionales `llama-cpp-python` a partir del código fuente, lo que requiere las herramientas de compilación de Visual Studio (componente C++) y CMake.

**macOS:** la compatibilidad con CUDA no está habilitada (no hay CUDA); un comando `trainer.train()` que requiera CUDA generará `DEP_GPU_NOT_AVAILABLE`, y puede ejecutar el adaptador entrenado en un Mac a través de Ollama. Un sistema MLX **experimental y no verificado** (`--backend mlx`, `pip install 'backpropagate[mlx]'`) entrena un adaptador LoRA de forma nativa en Apple Silicon a través de `mlx_lm.lora`; solo LoRA SFT y **no verificado en hardware real** (consulte [Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview)). Para la ruta CUDA o para ORPO / ajuste completo / FP8 / ejecución múltiple, utilice una máquina Linux o Windows con CUDA.

Consulte la [página de la guía de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para obtener una guía completa de solución de problemas de instalación, y la [página dedicada de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) para problemas relacionados con el controlador / VRAM / xformers / bf16 frente a fp16.

## CLI

Cada API de Python tiene un equivalente en la CLI:

```bash
backprop train --data my_data.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output --format gguf --quantization q4_k_m --ollama --ollama-name my-model
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

Referencia completa en [la página de la guía de la CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), o `backprop <subcommand> --help`.

## Configuración

Se puede anular cada configuración con una variable de entorno utilizando el prefijo `BACKPROPAGATE_`:

| Variable | Valor predeterminado | Notas |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Forzar registros en formato JSON o en la consola |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modelo predeterminado |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Tasa de aprendizaje |
| `BACKPROPAGATE_LORA__R` | `256` | Rango de LoRA. Establecerlo desactiva la selección automática del tamaño del adaptador (consulte `--lora-preset` en [Modelos predefinidos](#model-presets)). |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sandbox del sistema de archivos de la interfaz de usuario |

Las claves anidadas utilizan doble guion bajo (`MODEL__NAME`, no `MODEL_NAME`). La referencia completa está en [la página de la guía de las variables de entorno](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Modelos predefinidos

| Predefinición | Memoria de la GPU | Licencia | Notas |
|---|---|---|---|
| Qwen-3.5-4B | 6 / 7 / 11 GB | Apache 2.0 | Valor predeterminado recomendado para modelos de menos de 5B. La mejor calidad para este tamaño. |
| Phi-4-mini-3.8B | 6 / 7 / 12 GB | MIT | Destaca en razonamiento / matemáticas / código. Licencia estrictamente limpia. |
| SmolLM3-3B | 4 / 5 / 10 GB | Apache 2.0 | Receta totalmente abierta. Contexto nativo de 64K. |
| Qwen 2.5 7B | 9 / 11 / 17 GB | Apache 2.0 | Valor predeterminado existente. La mejor calidad de las predefiniciones de 7B. |
| Qwen 2.5 3B | 4 / 6 / 10 GB | Qwen-Research | ⚠ licencia de investigación: consulte los términos de la licencia de Qwen antes de su uso comercial. |
| Llama 3.2 3B | 4 / 6 / 9 GB | Llama Community | Una alternativa sólida a Qwen 3B con algunas limitaciones permisivas. |
| Llama 3.2 1B | 2 / 3 / 5 GB | Llama Community | Para experimentos rápidos en tarjetas pequeñas. |
| Mistral 7B | 6 / 8 / 14 GB | Apache 2.0 | Comparable a Qwen 7B, con una plantilla de chat diferente. |
| Llama-3.1-8B | 9 / 11 / 18 GB | Llama-3.1-Community | 8B QLoRA, contexto nativo de 128K (la cláusula de >700M de usuarios activos mensuales requiere una licencia de Meta independiente). |
| **Qwen2.5-14B** | 18.7 GiB de uso máximo con 4096 de contexto (QLoRA) | Apache 2.0 | **La opción para un uso diario en una tarjeta de 32 GB.** Rango/alfa 32, AdamW de 8 bits. Los pesos de 4 bits por sí solos ocupan unos 8.5 GB; una ventana completa de 4096 tokens necesita el resto. |
| Mistral-Small-24B | 22.8 GiB de uso máximo con 4096 de contexto (QLoRA) | Apache 2.0 | 24B QLoRA en una tarjeta de 32 GB. Los pesos de 4 bits por sí solos ocupan unos 18 GB. |
| **Qwen2.5-32B** | 26.0 GiB de uso máximo con 2048 de contexto (QLoRA) | Apache 2.0 | **La opción más potente para una tarjeta de 32 GB.** Se ajusta en `max_len 2048` con AdamW de 8 bits. |

Otros modelos suelen funcionar; las filas anteriores son las predefiniciones seleccionadas; el rango de 14B a 32B está ajustado con QLoRA para una tarjeta de 32 GB (el rango medido). Para las predefiniciones de hasta 8B, las tres cifras son estimaciones de QLoRA para los tamaños de adaptador `fast`, `balanced` y `quality` con un lote de 1 y ejemplos de 2048 tokens; se inclinan hacia el lado alto, y los ejemplos más cortos utilizan menos. El tamaño del adaptador se elige para su tarjeta: `--lora-preset auto` (el valor predeterminado) toma el mayor de `quality` (rango 256 en cada capa lineal, según Biderman 2024 y Thinking Machines 2025), `balanced` (rango 64 en cada capa lineal) y `fast` (rango 16 en dos capas por bloque) que quepa en la memoria disponible en su GPU. Especifique uno para forzarlo. `backprop estimate-vram` imprime la estimación para cualquier modelo y configuración.

## Solución de problemas

Un índice breve de los fallos más comunes que ocurren durante la primera ejecución. El índice completo está en [la página de la guía de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Para obtener información detallada sobre el controlador / VRAM / precisión mixta, consulte la [página de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Síntoma | Código de error | Solución |
|---|---|---|
| La GPU se queda sin memoria durante el entrenamiento. | `RUNTIME_GPU_OOM` | Automático: Backpropagate reduce a la mitad el tamaño del lote y vuelve a intentarlo hasta 3 veces. Para desactivar: `Trainer(oom_recovery=False)`. Para forzar un tamaño menor: `--batch-size 1`. |
| HuggingFace devuelve 401 / "modelo no encontrado". | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` y vuelve a intentarlo. Para errores tipográficos, copie el ID exacto de <https://huggingface.co/models>. |
| `register_with_ollama`, conexión rechazada. | `DEP_OLLAMA_REGISTRATION_FAILED` | Inicie el demonio: `ollama serve`. Instale desde <https://ollama.com>. Se puede reintentar. |
| El disco está lleno durante el guardado del punto de control. | `STATE_CHECKPOINT_INVALID` | Las escrituras atómicas dejan un directorio `.partial` en caso de fallo; es seguro eliminarlo. El punto de control anterior y correcto está intacto. |
| El entrenamiento se pausa debido al sobrecalentamiento de la GPU. | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automático: Backpropagate se pausa al alcanzar el umbral de temperatura y se reanuda a medida que la GPU se enfría. Mejore el flujo de aire si esto sigue ocurriendo. |
| `backprop ui --share` rechazado. | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Pase `--auth user:password` o utilice el reenvío de puertos SSH en su lugar (consulte [Interfaz de usuario web](#web-ui)). |
| La exportación de GGUF falló en el primer intento. | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; en Windows también necesita las herramientas de compilación de Visual C++ y CMake. |

## Informar sobre errores

Cuando algo falla, Backpropagate imprime una línea al inicio, como `run_started run_id=<uuid>`, y vincula el mismo ID a cada línea del registro, a cada punto de control y a cada entrada de Weights & Biases. **Incluya el `run_id` en cualquier informe de errores**, ya que esto permite al responsable del mantenimiento correlacionar todo para esa ejecución específica.

Un buen informe de errores incluye:

1. **El `run_id`**: el UUID que se imprime al inicio. Un UUID permite al responsable del mantenimiento correlacionar cada línea del registro, cada punto de control y cada entrada de Weights & Biases para esa ejecución específica.
2. **El código de error**: la línea `[CODE_NAME]: message` en stderr. Consulte [códigos de error](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) para obtener el catálogo de códigos estables.
3. **El rastreo de pila redactado**. Stderr se redacta automáticamente en el modo no detallado (los tokens de Bearer, `sk-*`, `hf_*`, las claves de AWS, los pares `password=` / `token=` / `api_key=` se eliminan; es seguro pegarlo). Para obtener el rastreo de pila completo y no redactado, vuelva a ejecutarlo con `BACKPROPAGATE_DEBUG=1` (o `--verbose`); revíselo antes de publicarlo.
4. **La salida de `backprop info`**. Un comando imprime el modelo de Python / PyTorch / CUDA / GPU / VRAM / SO / extras instalados; todo lo que el responsable del mantenimiento necesita para identificar una regresión específica de la plataforma.

La [plantilla de informe de errores](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) solicita explícitamente cada uno de estos elementos para que la evaluación inicial sea rápida. Las preguntas, ideas o consultas sobre si algo es "esperado" deben realizarse en [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Los problemas de seguridad deben informarse de forma privada a través del formulario [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new); consulte [SECURITY.md](SECURITY.md) para conocer la política y los plazos de respuesta.

## Privacidad

Todo el entrenamiento se realiza localmente en su GPU. Backpropagate no realiza ninguna solicitud de red, excepto para descargar modelos de HuggingFace (lo que usted inicia). No hay telemetría ni dependencia de la nube.

## Referencias

Los valores predeterminados de Backpropagate y el modo de entrenamiento de múltiples ejecuciones se basan en investigaciones recientes. Si está interesado en las técnicas subyacentes:

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — el documento fundamental que presenta LoRA, que es la forma en que Backpropagate entrena los adaptadores de manera eficiente.
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidencia empírica de que LoRA con un rango de 256 y objetivos totalmente lineales coincide con la calidad del ajuste fino completo en la mayoría de las tareas posteriores al entrenamiento, con un 67% del poder de cómputo. Esto impulsa la configuración predeterminada de LoRA v1.3 de Backpropagate.
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora) — la continuación práctica que identifica la corrección de 10× en la tasa de aprendizaje frente al ajuste fino completo necesaria a un rango LoRA alto.
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la caracterización original de por qué las redes neuronales "olvidan" el entrenamiento anterior cuando se realiza un ajuste fino con nuevos datos (EWC: consolidación elástica del peso).
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, un enfoque anterior para utilizar LoRA para el aprendizaje continuo restringiendo los nuevos adaptadores a subespacios ortogonales.
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — una técnica fundamental para fusionar varios modelos ajustados sin interferencias.
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — el algoritmo específico que el fusionador de múltiples ejecuciones de Backpropagate implementa. Un preprint de diciembre de 2025; Backpropagate es el primer usuario conocido de este documento.

## Licencia

MIT — consulte [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
