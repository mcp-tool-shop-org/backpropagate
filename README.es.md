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

Realiza el ajuste fino de modelos de lenguaje grandes mediante retropropagación en una **única** GPU, dimensionada según la tarjeta que realmente tienes. Tres líneas de código Python para ajustar un modelo QLoRA de 7B a 32B en una tarjeta de consumo de 32 GB (RTX 5090). Una sola opción, `--full-ft-offload`, realiza un ajuste fino completo de un modelo de la clase 7B manteniendo sus pesos y gradientes en la RAM del host (Linux o WSL2; es más lento y se mide a continuación). Un comando más exporta a Ollama y, a continuación, `ollama run` realiza el ajuste fino. Se reduce a 16 GB. Funciona perfectamente en Windows.

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

Eso es todo. No hay ningún archivo de configuración YAML. No hay ninguna "ceremonia" de `accelerate launch`. No hay ningún tutorial independiente sobre "cómo convertirlo a GGUF". Si tienes una GPU CUDA y un archivo JSONL con tus datos de entrenamiento, estarás a solo tres líneas de distancia de obtener un ajuste fino funcional.

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

¿Prefieres Docker? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` también funciona. Hay imágenes disponibles tanto para `linux/amd64` como para `linux/arm64`, por lo que los usuarios de Apple Silicon y ARM Linux obtienen una imagen nativa. Un ejemplo canónico de `compose.yaml` para "UI en un contenedor" se encuentra en la raíz del repositorio; `docker compose up` inicia la interfaz de usuario web en `http://localhost:7860` con un volumen persistente `~/.backpropagate`.

## Dónde se ubica Backpropagate en el panorama

Existen varias bibliotecas excelentes para el ajuste fino de LLM. Cada una de ellas destaca en diferentes aspectos:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**: si te gustan las configuraciones YAML y deseas una comunidad de recetas de las que puedas copiar.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**: si deseas DPO/PPO/RLHF y una GUI web.
- **[Unsloth](https://github.com/unslothai/unsloth)**: si necesitas el entrenamiento más rápido posible y utilizas una familia de modelos compatible.
- **[torchtune](https://github.com/pytorch/torchtune)**: si deseas las recetas nativas de PyTorch de Meta que puedes editar.

Backpropagate es la opción que faltaba: **una API de Python de 3 líneas para usuarios individuales en una sola GPU de consumo que desean entrenar un adaptador e integrarlo.** Sin YAML, sin GUI, sin RL en línea (PPO/GRPO), sin multi-nodo. Solo el ciclo que realmente necesita todo el mundo y el paso de exportación que dificulta el proceso.

Si has probado alguna de las bibliotecas anteriores y te has topado con la "ceremonia" de los archivos de configuración, o has encontrado una limitación en la familia de modelos, o has deseado tener opciones predeterminadas para Windows, Backpropagate es para ti.

## Qué puedes ajustar en una sola GPU

Backpropagate dimensiona la ejecución según tu tarjeta. Estos son números **medidos** del 30 de septiembre de 2026 en una RTX 5090 de 32 GB (comprobantes: [`docs/receipts/2026-09-30-offload/`](docs/receipts/2026-09-30-offload/)). Los picos de QLoRA se producen en la ventana de contexto completa del ajuste preestablecido con un lote de 1, que es el peor de los casos para ese ajuste preestablecido; los ejemplos más cortos utilizan menos recursos.

| Modelo | Método | Medido en una tarjeta de 32 GB |
|---|---|---|
| **14B** (Qwen2.5-14B) | QLoRA | **25,0 GiB** de pico a 4096 de contexto (28,1 GiB reservados). |
| 24B (Mistral-Small-24B) | QLoRA | 26,5 GiB de pico a 4096 de contexto (29,6 GiB reservados). |
| **32B** (Qwen2.5-32B) | QLoRA | **Apenas cabe:** 28,8 GiB de pico a 2048 de contexto (30,7 GiB reservados, aproximadamente 0,65 GiB de espacio libre). |
| 3B | `mode="full"` (ajuste fino completo real, en la GPU) | **22,0 GiB** de pico (a nivel del sistema), 0,30 s/paso con un lote de 4, 512 de contexto. 7,5 GiB de eso son el estado del optimizador paginado, que puede descargarse en la RAM del host en una tarjeta más pequeña (no probado). |
| **Clase 7B** (Qwen2.5-7B, 7,6B de parámetros) | `mode="full" --full-ft-offload` | **Entrena:** 5,3 GiB de VRAM, **30,8 GiB de RAM del host** (32,2 GiB al guardar), **14,7 s/paso**. Solo para Linux o WSL2. |

No se ha vuelto a medir en esa sesión: QLoRA de 7B, Llama-3.1-8B (repositorio con acceso restringido, sin token en la máquina de prueba) y ajuste fino completo en la GPU por encima de 3B. Las cifras de estos se encuentran en otros lugares de la documentación y son estimaciones.

Dos cosas para las que la mayoría de las bibliotecas de una sola GPU te dirigen a otro lugar: **QLoRA de 24 a 32B** y **ajuste fino completo de la clase 7B en una sola tarjeta**, Backpropagate lo hace en una sola tarjeta de consumo y, a continuación, exporta el resultado directamente a Ollama.

**El ajuste fino completo tiene dos rutas.** Sin descarga, el modelo, sus gradientes y el estado del optimizador se encuentran todos en la GPU. La biblioteca limita el tamaño del modelo mediante la detección de la VRAM (**16 GB → 4B, 24 GB → 5B, 32 GB → 6B**); estos límites provienen de cálculos de memoria y solo se miden hasta 3B. Anula con `--full-ft-ceiling-billions`.

`--full-ft-offload` mantiene los pesos y los gradientes en la RAM del host y los transmite a la GPU (descarga de CPU FSDP2). Lo que cuesta, medido:

- **RAM del host:** la comprobación de ajuste solicita aproximadamente 3,7 GiB por cada mil millones de parámetros más 10 GiB, lo cual es conservador (39 GiB a 7,6B frente a los 32,2 GiB medidos). La ejecución se rechaza de antemano, con los números, si la máquina no puede contenerlo. Un modelo de 7,6B no cabe en un límite de memoria de 28 GB de WSL2; aproximadamente 5B es el límite práctico.
- **Velocidad:** 14,7 s/paso a 7,6B (lote de 1) y 5,1 s/paso a 3B (lote de 4), frente a aproximadamente 0,63 s/paso para 3B en la GPU con un lote de 4. Úsalo solo cuando el modelo no quepa sin él. Se planea una versión más rápida.
- **Optimizador:** Adafactor, no AdamW. Los pesos permanecen en bf16 y cada actualización se vuelve a escribir con un redondeo estocástico; no hay una copia fp32.
- **Calidad:** en una ejecución de 3B (150 pasos, pérdida de datos no utilizados, una semilla), alcanzó aproximadamente el 85% de la mejora que obtuvo el ajuste fino completo normal (2,45 → 1,93 frente a 2,45 → 1,84). Una sola semilla no es un punto de referencia.
- **Ámbito:** ajuste fino supervisado simple. Sin empaquetamiento, sin enmascaramiento solo de respuestas, sin puntos de control intermedios, sin reanudación. Solo para Linux o WSL2 (FSDP2 necesita NCCL); en Windows nativo, se detiene con `DEP_FSDP_UNAVAILABLE`.
- **Aún no probado:** ejecuciones largas, acumulación de gradientes por encima de 1 y una máquina física de 64 GB (la máquina de prueba tenía más RAM, con un límite de 60 GiB aplicado en la prueba).

Un modelo que no se ajusta correctamente sale con `RUNTIME_FULL_FT_MODEL_TOO_LARGE` y indica la forma de salir. Consulte [la página completa del manual de ajuste fino](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/).

### Se reduce a 16 GB

El rango de 16 GB (RTX 4080 / 5080 / 4070 Ti Super) sigue siendo de primera clase: 7B QLoRA y un ajuste fino completo de un modelo de ~3B (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B/1B) a través de `mode="full"` (22,0 GiB medidos en una tarjeta de 32 GB a 3B, de los cuales 7,5 GiB son el estado del optimizador paginado que puede desbordarse a la RAM del host; no se ha probado si esto funciona de manera aceptable en una tarjeta de 16 GB). Con `--full-ft-offload`, la GPU ocupa mucho menos espacio: con la VRAM limitada en la tarjeta de prueba, un modelo de 3B entrenado con un límite de 6 GiB y modelos de 4B y 7,6B con un límite de 8 GiB. Estos son límites simulados en una tarjeta de 32 GB, no ejecuciones en hardware real de 8 GB. El mismo código elige el tamaño del lote y el límite que se ajustan a la tarjeta que detecta.

La cuantización de 2 bits (AQLM / QuIP#) queda **fuera del alcance**: una base de 2 bits no se puede combinar limpiamente con pesos de precisión completa, lo que interrumpe el contrato de exportación de adaptador combinable → GGUF → Ollama (que es el objetivo principal de la canalización). En su lugar, Backpropagate ofrece las opciones que permiten ampliar las capacidades: QLoRA, `mode="full"`, `--full-ft-offload` y la ruta de cálculo FP8 (`--fp8`, Blackwell/Hopper), todas las cuales siguen siendo combinables y exportables.

## Para qué NO sirve Backpropagate

Si su caso de uso se encuentra entre los siguientes, obtendrá mejores resultados con una biblioteca diferente: Backpropagate no es la opción correcta y tratar de hacerlo funcionar costaría más que simplemente utilizar la herramienta adecuada. Leer esta sección antes de comenzar evita el ciclo de instalación y prueba:

- **Ajuste fino de parámetros completos de modelos de 13B+** — Backpropagate realiza un ajuste fino completo de hasta aproximadamente 6B en una GPU de 32 GB y un modelo de la clase 7B con `--full-ft-offload` (consulte [el rango](#what-you-can-fine-tune-on-one-gpu)). Un ajuste fino completo de un modelo de 13B+ requiere FSDP multi-GPU o una tarjeta más grande. Antes de invertir en esa capacidad de cómputo, evalúe la evidencia en ambos sentidos. [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/) informa que LoRA coincide con el ajuste fino completo cuando se aplica a cada capa y el conjunto de datos se ajusta a la capacidad del adaptador, en aproximadamente dos tercios de la capacidad de cómputo por pasada. [Biderman et al. 2024](https://arxiv.org/abs/2405.09673) descubrió que, en entornos de bajo rango estándar, LoRA tiene un rendimiento significativamente inferior al del ajuste fino completo en código y matemáticas, al tiempo que olvida menos. Para el seguimiento de instrucciones, el trabajo de personalidad y estilo en conjuntos de datos modestos, QLoRA de hasta 32B suele ser la mejor opción para una sola tarjeta.
- **RL en línea: PPO / GRPO / RLVR** — Backpropagate realiza un ajuste fino SFT de una sola etapa más un ajuste de preferencias sin referencia (ORPO en v1.5; SimPO + KTO en v1.6). Lo que *no* hace es el aprendizaje por refuerzo en línea: PPO, GRPO o RLVR, que requiere un modelo de recompensa o un bucle de generación y puntuación además del paso de entrenamiento. Para estos, utilice TRL directamente o LLaMA-Factory. (El ajuste de preferencias sin referencia se ajusta al rango de una sola etapa porque no hay un modelo de referencia separado que mantener en la memoria; consulte la nota de ORPO en [Inicio rápido](#quick-start).)
- **Entrenamiento multi-nodo** — solo una GPU en una máquina. El multi-GPU en una máquina funciona (a través de `accelerate launch`) pero no está oficialmente admitido.
- **Entrenamiento en macOS en el entorno CUDA** — Apple Silicon no tiene CUDA, por lo que la ruta CUDA se ejecuta en una máquina Linux o Windows con una GPU NVIDIA. Aún puede ejecutar el modelo entrenado en un Mac a través de Ollama. Un entorno MLX **experimental y no verificado** (`--backend mlx`) entrena un adaptador LoRA de forma nativa en Apple Silicon; consulte [Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview). Solo es LoRA-SFT y **no está verificado en silicona real** (sin soporte), por lo que, para cualquier cosa que vaya más allá de un SFT LoRA (ORPO, ajuste fino completo, FP8, ejecución múltiple), desea el entorno CUDA.
- **Cualquier cosa fuera de las familias de modelos probadas** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Otros modelos a menudo funcionan, pero no están fijados en CI.

Si necesita alguna de estas cosas, utilice una de las bibliotecas enumeradas anteriormente. Son mejores para ello.

## Lo que le ofrece Backpropagate

Cuatro cosas, en una sola instalación:

**1. Una API real de 3 líneas que se ejecuta sin un archivo de configuración.**
El fragmento que se encuentra en la parte superior de este archivo README se ejecuta de principio a fin. No hay `accelerate config`, ni YAML, ni reemplazos de Hydra. Solo `Trainer(model).train(data)` y ya tiene un ajuste fino.

**2. Windows que realmente funciona.**
La mayoría de las bibliotecas de ML tratan a Windows como algo secundario. Backpropagate se prueba de forma nativa en Windows + RTX 5080. La biblioteca gestiona las peculiaridades del tiempo de ejecución por usted: sabe cómo pre-tokenizar sus datos para que el procesamiento multi-hilo de Windows no falle, desactiva automáticamente xformers en las tarjetas RTX 40/50 donde provocaría un error y elige la configuración del cargador de datos que no causa problemas. No tiene que saber nada de esto. Simplemente funciona.

**3. Diseñado para ejecuciones sin supervisión.**
El entrenamiento lleva horas. No quiere tener que vigilarlo. Backpropagate está diseñado para que se pueda dejar funcionando:

- Si se queda sin memoria de la GPU, reduce automáticamente a la mitad el tamaño del lote y lo vuelve a intentar, hasta tres veces. No requiere ajustes manuales.
- Si su GPU se calienta demasiado, se detiene hasta que las cosas se enfrían y luego continúa.
- Cada punto de control se escribe de forma atómica: si su computadora portátil falla a la mitad del guardado, el punto de control anterior y válido permanece intacto.
- Cada ejecución de entrenamiento obtiene un ID único que se estampa en cada línea del registro, en cada punto de control y en cada entrada de Weights & Biases. Si algo sale mal, un ID permite a un mantenedor correlacionar todo.
- Los errores vienen con códigos estables (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, etc.) para que pueda buscar en sus registros y en la [guía de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para encontrar la solución. Los fallos específicos de CUDA tienen una [página de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) dedicada.

**4. Un solo comando del adaptador entrenado a `ollama run`.**
Muchas bibliotecas entrenan un modelo. Pocas se apartan cuando realmente quieres usarlo. Backpropagate exporta a GGUF (el formato que usa Ollama) y registra un modelo de Ollama con un solo comando. Pasas de "entrenamiento completado" a "puedo conversar con mi modelo ajustado" en unos 30 segundos.

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

`backprop train` escribe el adaptador en `./output` (cámbialo con `--output`). En Python, lo mismo es:

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")
```

Utiliza un entorno virtual con `pip install "backpropagate[standard]"` para la API de Python; `pipx` instala el comando `backprop` en su propio entorno, por lo que `import backpropagate` no lo encontrará.

**Qué necesita la exportación a GGUF.** La exportación fusiona tu adaptador con el modelo base y lo convierte con el script de conversión de llama.cpp. Necesitas:

- una copia de la fuente de llama.cpp (`git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp`) más `pip install sentencepiece protobuf` en el mismo entorno, o
- Unsloth con su propio llama.cpp ya compilado.

Con `--ollama`, la cuantización `q4_k_m` se realiza mediante `ollama create`, por lo que no es necesario compilar nada. Backpropagate nunca permite que Unsloth instale paquetes del sistema para compilar llama.cpp por ti; establece `BACKPROPAGATE_UNSLOTH_AUTO_INSTALL=1` si quieres que lo haga. Detalles: [export](https://mcp-tool-shop-org.github.io/backpropagate/handbook/export/).

Para tus propios datos, formatea tu archivo JSONL con un ejemplo por línea:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca (`instruction` / `output`), OpenAI chat (`messages`) y formatos de texto sin formato también funcionan; Backpropagate detecta automáticamente el formato.

### El ciclo: verifica los datos, entrena, evalúa, exporta

```bash
backprop data report my_data.jsonl                     # duplicates, length outliers, format problems
backprop data split my_data.jsonl --heldout-ratio 0.1  # a held-out set the model never trains on
backprop train --data my_data.train.jsonl --steps 200 --output ./run-a
backprop eval <run-id> --heldout my_data.heldout.jsonl # held-out loss + sample generations
backprop eval <run-b> --vs <run-a>                     # did the change help?
backprop export ./run-a --format gguf --ollama --ollama-name my-model
```

La evaluación se realiza sin juicio: pérdida retenida más métricas de tareas deterministas (`normalized_exact_match`, `token_f1`, `contains`, `regex`, `pass_rate`). Para utilizar un juez LLM, ejecútalo sobre la salida de `backprop generate`. Consulta [recetas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Ajuste de preferencias (ORPO, SimPO, KTO)

Entrena con preferencias en lugar de demostraciones simples. ORPO no requiere referencias y es de una sola etapa; integra la señal de preferencia en el paso SFT, por lo que no hay un modelo de recompensa o referencia separado y la forma de 3 líneas no cambia. Pasa `--method orpo` (CLI) o `method="orpo"` (Python) y proporciónale un conjunto de datos de `{prompt, chosen, rejected}` (o solo `{chosen, rejected}`) filas:

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

La tasa de aprendizaje predeterminada se reduce automáticamente a `8e-6` para ORPO (la pérdida es más pronunciada que en el SFT simple); ajusta `--orpo-beta` (predeterminado `0.1`) para ponderar la penalización de la razón de probabilidades. ORPO es solo `mode="lora"`.

**Novedad en la v1.6: SimPO y KTO.** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) no requiere referencias y utiliza una recompensa normalizada por longitud, y toma los mismos datos emparejados `{prompt, chosen, rejected}` que ORPO (`--simpo-beta`, `--simpo-gamma`). `--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) toma datos **no emparejados** `{prompt, completion, label}`: calificaciones positivas/negativas por ejemplo, para la gran clase de comentarios que no son pares A/B seleccionados; equilibra automáticamente los pesos de pérdida deseables/indeseables a partir de los recuentos de etiquetas. Ambos son solo `mode="lora"` y permanecen dentro del ámbito SFT de una sola GPU (sin un modelo de referencia separado). Consulta el [manual de ajuste de preferencias](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) para saber cuál utilizar. Para RL en línea (PPO/GRPO), consulta [para qué NO sirve Backpropagate](#what-backpropagate-is-not-for).

### SFT de razonamiento-rastreo (destilación R1)

Destila un modelo de razonamiento de forma sencilla. Pasa `--reasoning-trace` (CLI) o `Trainer(..., reasoning_trace=True)` (Python) y proporciónale rastros que mantengan una cadena de pensamiento `<think>...</think>` dentro del turno del asistente; la mitad SFT pura de la destilación de [DeepSeek-R1](https://arxiv.org/abs/2501.12948), no se requiere RL. Backpropagate mantiene `<think>` en el objetivo de entrenamiento, elimina los rastros vacíos o demasiado largos (filtrado de la longitud del rastro) y aumenta el valor predeterminado de `max_seq_length` a 8192 para el CoT más largo. Lo más importante es que `<think>` sigue siendo **texto sin formato**: sin tokens especiales, sin cambio de tamaño del incrustamiento, por lo que el GGUF fusionado sigue exportándose a Ollama como cualquier otro modelo ajustado. Solo SFT. Consulta la [receta de razonamiento-rastreo](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) para conocer la forma del conjunto de datos y el rango de tokens ajustable.

### Apple Silicon (MLX): vista previa no verificada

> ⚠️ **Vista previa no verificada: no forma parte del conjunto de funciones admitidas.** El "rail" MLX está construido y se han realizado pruebas unitarias, pero **no** se ha verificado en Apple Silicon real (`mlx-lm` solo funciona en Apple y no se puede ejecutar en los equipos NVIDIA en los que se desarrolla Backpropagate). Considera todo lo que se indica a continuación como experimental, úsalo bajo tu propio riesgo y [informa de las anomalías](#reporting-bugs) si lo ejecutas en un Mac de la serie M.

**Una API, dos "rails".** CUDA es el backend canónico y verificado; MLX es un segundo "rail" que se entrena en un Mac de la serie M a través del conjunto de herramientas [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) de Apple (memoria unificada, sin CUDA). La forma de 3 líneas selecciona el "rail" según el hardware: `backend='auto'` (el valor predeterminado) se dirige a CUDA en NVIDIA y a MLX en Apple Silicon, por lo que los equipos CUDA existentes son idénticos a nivel de bytes:

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

El "rail" MLX es **solo SFT LoRA**: no ORPO, no FP8, no `mode='full'`, no ejecución múltiple (cada uno se rechaza con `CONFIG_INVALID_SETTING`; utiliza `backend='cuda'`/`'auto'` en un equipo NVIDIA para ello). El adaptador resultante es un archivo safetensors simple y se exporta a Ollama a través del mismo camino que el "rail" CUDA.

> Forzar `--backend mlx` en un host que no sea Apple genera un error con `CONFIG_INVALID_SETTING`; la falta del conjunto de herramientas `mlx_lm` en un Mac genera `DEP_MLX_UNAVAILABLE`.

Para obtener más flujos de trabajo de extremo a extremo (ajuste fino y carga en HF Hub, reanudación después de que se agota la memoria, SLAO de ejecución múltiple en una campaña larga, etc.), consulta la [página de recetas del manual](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interfaz de usuario web (opcional)

Si prefieres hacer clic en lugar de escribir en Python, instala el complemento de la interfaz de usuario y ejecútalo:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Se abre una interfaz web local en `http://localhost:7862` para explorar conjuntos de datos, validar formatos y ensamblar una configuración de entrenamiento visualmente. El entrenamiento en sí se ejecuta a través de `backprop train` (el entrenamiento basado en la interfaz de usuario está en la hoja de ruta; el botón "Iniciar" muestra actualmente esa nota). De forma predeterminada, la interfaz de usuario es solo local. Para exponerla a otros dispositivos, consulte la sección [Interfaz de usuario web](#web-ui) a continuación para conocer el contrato de seguridad `--share` + `--auth`.

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

Esto ejecuta cinco ciclos de entrenamiento, fusionando el adaptador entre las ejecuciones de manera que se preserve el conocimiento previo al tiempo que se incorporan nuevos ejemplos. La técnica se basa en investigaciones recientes sobre el aprendizaje continuo; consulte la sección [Referencias](#references) al final de este archivo README.

La versión de la CLI:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Reanudar desde un punto de control

Un entrenamiento de 5 ejecuciones que falla en la ejecución 4 se puede recuperar. Cada sesión con múltiples ejecuciones escribe su ID de ejecución en el historial y el manifiesto del punto de control en el disco, por lo que reanudar donde lo dejó es un solo comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

El comportamiento predeterminado de `backprop multi-run` (sin `--resume`) detecta automáticamente una entrada en curso en el mismo directorio de salida y la continúa. Para forzar un inicio limpio, apunte a un directorio de salida nuevo.

## Historial de entrenamiento

Cada invocación de `backprop train` y `backprop multi-run` registra una fila en `<output>/run_history.json`: modelo utilizado, conjunto de datos, hiperparámetros, estado, pérdida final, historial de pérdidas. Puede enumerar e inspeccionar las ejecuciones anteriores:

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

Anule la configuración con `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` o `Trainer(report_to="none")` para optar por no participar.

## Interfaz de usuario web

La interfaz web de Reflex es opcional: instálela con `pipx install "backpropagate[ui]"` y ejecútela:

```bash
backprop ui --port 7862
```

La interfaz de usuario se ejecuta localmente en `http://localhost:7862`. Hoy en día, cubre la mitad del flujo de trabajo que implica **explorar / validar / configurar**: apunte a un conjunto de datos, verifique el formato y las estadísticas detectados automáticamente, elija un modelo y cree una configuración de ejecución. **El lanzamiento de la ejecución se realiza desde la CLI** (`backprop train` / `backprop multi-run`); el botón "Iniciar" en la interfaz de usuario muestra una nota que indica dónde hacerlo. El entrenamiento basado en la interfaz de usuario es una función de seguimiento planificada; hasta entonces, la interfaz de usuario es el punto de entrada y la CLI es el disparador.

Para exponerla a otros dispositivos (otras personas en su red, una URL pública, etc.), debe emparejar `--share` (o `--host`) con `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` sin `--auth` finaliza con un error. La razón: `--share` publica una URL a la que puede acceder cualquier persona en Internet y, sin autenticación, esto significa que cualquier persona puede controlar su canalización de entrenamiento y leer su token de HuggingFace. No hay opción para desactivar esto; si no desea establecer credenciales, utilice el reenvío de puertos SSH en su lugar:

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

Consulte [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) para obtener el modelo de amenazas completo.

Las escrituras en el sistema de archivos desde la interfaz de usuario se limitan a un solo directorio:

- Predeterminado: `~/.backpropagate/ui-outputs`
- Anular: establezca `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- La anulación se valida mediante una lista de denegación: las rutas del sistema o de las credenciales (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) se rechazan.

## Notas de la plataforma

**Requisitos:** Python 3.10+ · GPU CUDA (8 GB+ de VRAM) · PyTorch 2.0+

Python 3.10 es compatible con la versión 1.6 como mínimo; su soporte oficial finaliza en octubre de 2026 y está programado para su eliminación en la primera versión posterior a esa fecha. Para las nuevas instalaciones, prefiera Python 3.11 o 3.12; 3.11 es la versión más probada.

Backpropagate gestiona las peculiaridades del tiempo de ejecución del entrenamiento en diferentes plataformas, pero no puede solucionar los problemas de instalación. Los dos más comunes son:

- **Rueda CUDA incorrecta.** PyTorch se publica con un binario por versión de CUDA. Si elige la incorrecta, obtendrá silenciosamente PyTorch solo para CPU y el entrenamiento será increíblemente lento. Utilice el selector de ruedas en <https://pytorch.org/get-started/locally/> para su controlador. Ejecute `nvidia-smi` para ver su versión de controlador / CUDA.
- **Windows + exportación GGUF.** El comando `[export]` compila `llama-cpp-python` desde el código fuente, lo que requiere Visual Studio Build Tools (componente C++) y CMake.

**macOS:** el soporte para CUDA no está habilitado (no hay CUDA); un comando `trainer.train()` con CUDA genera `DEP_GPU_NOT_AVAILABLE`, y puede ejecutar el adaptador entrenado en un Mac a través de Ollama. Un canal MLX **experimental y no verificado** (`--backend mlx`, `pip install 'backpropagate[mlx]'`) entrena un adaptador LoRA de forma nativa en Apple Silicon a través de `mlx_lm.lora`; solo SFT LoRA y **no verificado en hardware real** (consulte [Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview)). Para la ruta CUDA o para ORPO / ajuste fino completo / FP8 / múltiples ejecuciones, utilice una máquina Linux o Windows con CUDA.

Consulte la [página de solución de problemas del manual](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para obtener la guía completa de solución de problemas de instalación y la página dedicada de [solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) para los problemas de controlador / VRAM / xformers / bf16 vs. fp16.

## CLI

Cada API de Python tiene un espejo de CLI:

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

Referencia completa en [la página de referencia de la CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), o `backprop <subcommand> --help`.

## Configuración

Se puede anular cada configuración con una variable de entorno utilizando el prefijo `BACKPROPAGATE_`:

| Variable | Predeterminado | Notas |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | automático | Forzar registros JSON o de consola |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modelo predeterminado |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Tasa de aprendizaje |
| `BACKPROPAGATE_LORA__R` | `256` | Rango de LoRA (valor predeterminado de v1.3; pase `--lora-preset=fast` para el valor predeterminado de v1.2.x de 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Entorno de pruebas del sistema de archivos de la interfaz de usuario |

Las claves anidadas utilizan doble guion bajo (`MODEL__NAME`, no `MODEL_NAME`). La referencia completa está en [la página del manual de las variables de entorno](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Ajustes preestablecidos del modelo

| Ajuste preestablecido | VRAM | Licencia | Notas |
|---|---|---|---|
| Qwen-3.5-4B | ~8 GB | Apache 2.0 | Valor predeterminado recomendado para modelos de menos de 5B. La mejor calidad con este tamaño. |
| Phi-4-mini-3.8B | ~8 GB | MIT | Destaca en razonamiento, matemáticas y código. Licencia estricta y limpia. |
| SmolLM3-3B | ~6 GB | Apache 2.0 | Receta completamente abierta. Contexto nativo de 64K. |
| Qwen 2.5 7B | ~12 GB | Apache 2.0 | Valor predeterminado existente. La mejor calidad de los ajustes preestablecidos de 7B anteriores. |
| Qwen 2.5 3B | ~8 GB | Qwen-Research | ⚠ licencia de investigación: consulte los términos de la licencia de Qwen antes de su uso comercial. |
| Llama 3.2 3B | ~8 GB | Llama Community | Alternativa sólida a Qwen 3B con algunas limitaciones permisivas. |
| Llama 3.2 1B | ~6 GB | Llama Community | Para experimentos rápidos en tarjetas pequeñas. |
| Mistral 7B | ~12 GB | Apache 2.0 | Comparable a Qwen 7B, plantilla de chat diferente. |
| Llama-3.1-8B | ~7-8 GB (QLoRA) | Llama-3.1-Community | 8B QLoRA, contexto nativo de 128K (la cláusula de >700M-MAU requiere una licencia Meta separada). |
| **Qwen2.5-14B** | 25 GiB de pico a 4096 ctx (QLoRA) | Apache 2.0 | **El modelo principal para uso diario con 32 GB.** rango/alfa 32, AdamW de 8 bits. Los pesos de 4 bits por sí solos son de aproximadamente 8,5 GB; una ventana completa de 4096 tokens necesita el resto. |
| Mistral-Small-24B | 26,5 GiB de pico a 4096 ctx (QLoRA) | Apache 2.0 | 24B QLoRA en una tarjeta de 32 GB. Los pesos de 4 bits por sí solos son de aproximadamente 18 GB. |
| **Qwen2.5-32B** | 28,8 GiB de pico a 2048 ctx (QLoRA) | Apache 2.0 | **El mejor modelo para 32 GB.** Apenas cabe en `max_len 2048` con AdamW de 8 bits. |

Otros modelos suelen funcionar; las filas anteriores son los ajustes preestablecidos seleccionados; el rango de 14B a 32B está ajustado con QLoRA para una tarjeta de 32 GB (el rango medido). Pase `--lora-preset=quality` (valor predeterminado) para los objetivos de rango-256 / totalmente lineal por Biderman 2024 + Thinking Machines 2025, o `--lora-preset=fast` para el objetivo de rango-16 / q+v anterior si necesita la huella de v1.2.x.

## Solución de problemas

Un índice breve de los fallos más comunes que ocurren al ejecutarlo por primera vez. El índice inverso completo está en [la página del manual de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Para obtener información detallada sobre el controlador, la VRAM y la precisión mixta, consulte [la página de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Síntoma | Código de error | Solución |
|---|---|---|
| La GPU se queda sin memoria a mitad del entrenamiento | `RUNTIME_GPU_OOM` | Automático: Backpropagate reduce a la mitad el tamaño del lote y lo vuelve a intentar hasta 3 veces. Para desactivar: `Trainer(oom_recovery=False)`. Para forzar un tamaño menor: `--batch-size 1`. |
| HuggingFace devuelve 401 / "modelo no encontrado" | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` y vuelva a intentarlo. Para errores tipográficos, copie el ID exacto de <https://huggingface.co/models>. |
| `register_with_ollama` conexión rechazada | `DEP_OLLAMA_REGISTRATION_FAILED` | Inicie el demonio: `ollama serve`. Instale desde <https://ollama.com>. Se puede volver a intentar. |
| El disco se llena durante el guardado del punto de control | `STATE_CHECKPOINT_INVALID` | Las escrituras atómicas dejan un directorio `.partial` en caso de fallo; es seguro eliminarlo. El punto de control anterior y correcto está intacto. |
| El entrenamiento se pausa debido al sobrecalentamiento de la GPU | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automático: Backpropagate se pausa al alcanzar el umbral de temperatura y se reanuda a medida que la GPU se enfría. Mejore el flujo de aire si esto sigue ocurriendo. |
| `backprop ui --share` rechazado | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Pase `--auth user:password` o utilice el reenvío de puertos SSH en su lugar (consulte [la interfaz de usuario web](#web-ui)). |
| La exportación de GGUF falló en el primer intento | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; en Windows, también necesita las herramientas de compilación de Visual C++ y CMake. |

## Informar sobre errores

Cuando algo falla, Backpropagate imprime una línea al inicio, como `run_started run_id=<uuid>`, y vincula el mismo ID a cada línea del registro, a cada punto de control y a cada entrada de Weights & Biases. **Incluya el `run_id` en cualquier informe de error**, ya que esto permite a un mantenedor correlacionar todo para esa ejecución específica.

Un buen informe de errores incluye:

1. **El `run_id`**: el UUID que se imprime al inicio. Un UUID permite a un mantenedor correlacionar cada línea del registro, cada punto de control y cada entrada de Weights & Biases para esa ejecución específica.
2. **El código de error**: la línea `[CODE_NAME]: message` en stderr. Consulte [los códigos de error](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) para obtener el catálogo de códigos estables.
3. **El rastreo de pila redactado**. Stderr se redacta automáticamente en el modo no detallado (los tokens de Bearer, `sk-*`, `hf_*`, las claves de AWS, los pares `password=` / `token=` / `api_key=` se eliminan; es seguro pegarlo. Para obtener el rastreo de pila completo y no redactado, vuelva a ejecutarlo con `BACKPROPAGATE_DEBUG=1` (o `--verbose`); revíselo antes de publicarlo.
4. **La salida de `backprop info`**. Un comando imprime Python, PyTorch, CUDA, el modelo de GPU, la VRAM, el SO y los extras instalados: todo lo que el mantenedor necesita para identificar una regresión específica de la plataforma.

La [plantilla de informe de errores](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) solicita explícitamente cada uno de estos elementos para que la clasificación sea rápida. Las preguntas, las ideas o los hilos sobre si algo es "esperado" deben publicarse en [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Los problemas de seguridad deben informarse de forma privada a través del formulario [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new); consulte [SECURITY.md](SECURITY.md) para obtener información sobre la política y los plazos de respuesta.

## Privacidad

Todo el entrenamiento se realiza localmente en su GPU. Backpropagate no realiza ninguna solicitud a la red, excepto para descargar modelos de HuggingFace (lo que usted inicia). No hay telemetría, ni dependencia de la nube.

## Referencias

Los valores predeterminados de Backpropagate y el modo de entrenamiento con múltiples ejecuciones se basan en investigaciones recientes. Si está interesado en las técnicas subyacentes:

- **Hu et al. 2021.** *LoRA: Adaptación de bajo rango de modelos de lenguaje grandes.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — el documento fundamental que introduce LoRA, que es la forma en que Backpropagate entrena los adaptadores de manera eficiente.
- **Biderman et al. 2024.** *LoRA aprende menos y olvida menos.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidencia empírica de que LoRA con rango 256 y objetivos totalmente lineales iguala la calidad del ajuste fino completo en la mayoría de las tareas posteriores al entrenamiento, utilizando el 67% de la capacidad de cálculo. Impulsa la configuración predeterminada de LoRA v1.3 de Backpropagate.
- **Thinking Machines 2025.** *LoRA sin arrepentimientos.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — la continuación práctica que identifica la corrección de 10 veces la tasa de aprendizaje frente al ajuste fino completo necesaria a un rango LoRA alto.
- **Kirkpatrick et al. 2017.** *Superando el olvido catastrófico en las redes neuronales.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la caracterización original de por qué las redes neuronales "olvidan" el entrenamiento anterior cuando se realiza un ajuste fino con nuevos datos (EWC — Consolidación de pesos elásticos).
- **Wang et al. 2023.** *Aprendizaje de subespacio ortogonal para el aprendizaje continuo de modelos de lenguaje.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, un enfoque anterior para utilizar LoRA para el aprendizaje continuo mediante la restricción de los nuevos adaptadores a subespacios ortogonales.
- **Yadav et al. 2023.** *TIES-Merging: Resolviendo la interferencia al fusionar modelos.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — una técnica fundamental para fusionar múltiples modelos ajustados sin interferencias.
- **Qiao & Mahdavi 2025.** *Fusionar antes de olvidar: un único aprendizaje continuo de LoRA a través de la fusión continua.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — el algoritmo específico que implementa el fusionador de múltiples ejecuciones de Backpropagate. Un preimpreso de diciembre de 2025; Backpropagate es el primer usuario conocido de este documento.

## Licencia

MIT — consulte [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
