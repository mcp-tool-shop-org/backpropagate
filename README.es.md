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

# Ajuste un modelo de 32B QLoRA o un modelo completo de 7B en una sola GPU. Luego, envíelo a Ollama

Backpropagate permite ajustar modelos de lenguaje grandes en una **única** GPU, dimensionada según la tarjeta que realmente tenga. Con solo tres líneas de Python QLoRA, puede ajustar un modelo de 7B a 32B en una sola tarjeta de consumo de 32 GB (RTX 5090). Con una sola opción, `--full-ft-offload`, puede realizar un ajuste completo de un modelo de la clase 7B manteniendo sus pesos y gradientes en la memoria RAM del host (Linux o WSL2; es más lento y se mide a continuación). Un comando más exporta a Ollama y, a continuación, `ollama run` aplica su ajuste. Se puede escalar hasta 16 GB. Funciona perfectamente en Windows. ¿Prefiere un navegador a Python? `backprop ui` hace todo esto sin necesidad de código ([haga el recorrido](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/)).

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

Eso es todo. No hay ningún archivo de configuración YAML. No hay ningún proceso `accelerate launch` complicado. No hay ningún tutorial aparte que explique cómo convertirlo a GGUF. Si tienes una GPU CUDA y un archivo JSONL con tus datos de entrenamiento, solo necesitas tres líneas de código para obtener un ajuste fino funcional.

## Instalar

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

Si desea las funciones opcionales, sustituya la instalación actual por una de las siguientes:

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

¿Prefiere usar Docker? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` también funciona. Las imágenes se proporcionan tanto para `linux/amd64` como para `linux/arm64`, por lo que los usuarios de Apple Silicon y ARM Linux obtienen una imagen nativa. Un ejemplo estándar de `compose.yaml` para «interfaz de usuario en un contenedor» se encuentra en el directorio raíz del repositorio: coloque `user:password` en un `ui-auth.txt` junto a él, ejecute `docker compose up` e inicie sesión en `http://127.0.0.1:7860` (la primera vez que se ejecuta, se construye la interfaz, lo que tarda uno o dos minutos). El historial de ejecuciones se guarda en `~/.backpropagate`.

## ¿Cuál es la posición de Backpropagate en este contexto?

Existen varias bibliotecas excelentes para ajustar modelos lingüísticos grandes (LLM). Cada una de ellas destaca en diferentes aspectos:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**: si te gustan las configuraciones YAML y quieres tener acceso a una comunidad de recetas para usarlas como referencia.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**: si necesitas DPO/PPO/RLHF y una interfaz gráfica web.
- **[Unsloth](https://github.com/unslothai/unsloth)**: si necesitas el entrenamiento más rápido posible y utilizas una de las familias de modelos compatibles.
- **[torchtune](https://github.com/pytorch/torchtune)**: si quieres utilizar las recetas nativas de PyTorch, desarrolladas por Meta, y poder editarlas.

«Backpropagate» es la opción que faltaba: una API de Python de 3 líneas para desarrolladores individuales que utilizan una única GPU y desean entrenar un adaptador y ponerlo en funcionamiento. Sin archivos YAML, sin aprendizaje por refuerzo en línea (PPO/GRPO) ni configuración multi-nodo. Existe una interfaz de usuario en el navegador para el mismo proceso, en caso de que prefiera no escribir código. Solo el proceso que realmente necesita y el paso de exportación que dificulta las cosas.

Si probaste alguna de las bibliotecas mencionadas anteriormente y te resultó complicado configurar los archivos, o encontraste alguna limitación en cuanto a los modelos que admite, o preferías que tuviera opciones predeterminadas para Windows, entonces Backpropagate es la solución ideal para ti.

## Qué aspectos se pueden ajustar con precisión en una GPU

Backpropagate ajusta el tamaño de la ejecución a tu tarjeta. Estos son valores **medidos** en una RTX 5090 de 32 GB: las QLoRA filas correspondientes al 3 de octubre de 2026 (comprobantes: [`docs/receipts/2026-10-03-presets/`](docs/receipts/2026-10-03-presets/)), y las filas completas de ajuste fino correspondientes al 30 de septiembre de 2026 (comprobantes: [`docs/receipts/2026-09-30-offload/`](docs/receipts/2026-09-30-offload/)). QLoRA picos se alcanzan con la ventana de contexto completa del ajuste preestablecido y un lote de 1, que es el peor de los casos para ese ajuste preestablecido; los ejemplos más cortos utilizan menos recursos.

| Modelo. | Método | Medido con una tarjeta de 32 GB. |
|---|---|---|
| **14B** (Qwen2.5-14B) | QLoRA | **18,7 GiB** como máximo, con un contexto de 4096 (20,0 GiB reservados). |
| 24B (Mistral-Small-24B) | QLoRA | 22,8 GiB es el valor máximo con un contexto de 4096 (24,2 GiB reservados). |
| **32B** (Qwen2.5-32B) | QLoRA | **Capacidad:** 26,0 GiB, con un máximo de 2048 en el contexto (27,2 GiB reservados, aproximadamente 4 GiB disponibles). |
| 3B | `mode="full"` (ajuste fino completo y real, en la GPU) | **22,0 GiB** de máximo (a nivel de todo el sistema), 0,30 s/paso en el lote 4, 512 de contexto. 7,5 GiB de esto corresponde al estado del optimizador almacenado en la memoria paginada, que puede desbordarse y utilizar la memoria RAM del host en una tarjeta con menos capacidad (no probado). |
| **Modelo de la clase 7B** (Qwen2.5-7B, 7,6 mil millones de parámetros) | `mode="full" --full-ft-offload` | **Entrenamiento:** 5,3 GiB de VRAM, **30,8 GiB de RAM del host** (32,2 GiB durante el guardado), **14,7 s/paso**. Solo Linux o WSL2. |

No se volvieron a medir en esa sesión: 7B QLoRA, Llama-3.1-8B (repositorio con acceso restringido, sin token en la máquina de prueba) y ajuste completo con GPU para modelos de más de 3B. Los datos que se presentan en otras secciones de la documentación son estimaciones.

Dos de las funciones por las que la mayoría de las bibliotecas de GPU individuales te redirigen a otros lugares, **24–32B QLoRA** y el **ajuste fino completo de modelos de 7B en una sola tarjeta**, son realizadas por Backpropagate en una sola tarjeta de consumo, y luego el resultado se exporta directamente a Ollama.

**El ajuste fino completo tiene dos opciones.** Sin descargar datos, el modelo, sus gradientes y el estado del optimizador se almacenan todos en la GPU. La biblioteca limita el tamaño del modelo según la VRAM detectada (**16 GB → 4B, 24 GB → 5B, 32 GB → 6B**); estos límites se basan en cálculos de memoria y se miden solo hasta 3B. Se pueden anular con `--full-ft-ceiling-billions`.

`--full-ft-offload` almacena los pesos y los gradientes en la memoria RAM del host y los transmite a la GPU (descarga de la CPU FSDP2). Lo que implica, según las mediciones:

- **RAM del host:** la comprobación de ajuste requiere aproximadamente 3,7 GiB por cada mil millones de parámetros, más 10 GiB, lo cual es un valor conservador (39 GiB para 7,6 mil millones, frente a los 32,2 GiB medidos). La ejecución se rechaza de inmediato si la máquina no puede soportarlo. Un modelo de 7,6 mil millones no cabe en una memoria de 28 GB WSL2; el límite práctico es de aproximadamente 5 mil millones.
- **Velocidad:** 14,7 s/paso a 7,6 mil millones (lote 1) y 5,1 s/paso a 3 mil millones (lote 4), frente a aproximadamente 0,63 s/paso para 3 mil millones en la GPU con el lote 4. Úselo solo cuando el modelo no quepa sin él. Se planea una versión más rápida.
- **Optimizador:** Adafactor, no AdamW. Los pesos permanecen en bf16 y cada actualización se vuelve a escribir con redondeo estocástico; no hay una copia fp32.
- **Calidad:** en una ejecución de 3 mil millones (150 pasos, pérdida con datos de validación, una semilla), alcanzó aproximadamente el 85 % de la mejora que se obtuvo con el ajuste fino completo ordinario (2,45 → 1,93 frente a 2,45 → 1,84). Una sola semilla no es una referencia.
- **Alcance:** ajuste fino supervisado simple. Sin empaquetamiento, sin enmascaramiento solo de respuestas, sin puntos de control intermedios, sin reanudación. Solo Linux o WSL2 (FSDP2 requiere NCCL); en Windows nativo, se detiene con `DEP_FSDP_UNAVAILABLE`.
- **Aún no probado:** ejecuciones largas, acumulación de gradientes superior a 1 y una máquina física de 64 GB (la máquina de prueba tenía más RAM, con un límite de 60 GiB impuesto por la prueba).

Un modelo que no se ajusta correctamente finaliza su ejecución con el código `RUNTIME_FULL_FT_MODEL_TOO_LARGE` e indica la razón del fallo. Consulte [la página completa de la guía de ajuste fino](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/).

### Se reduce a 16 GB

El rango de 16 GB (RTX 4080 / 5080 / 4070 Ti Super) sigue siendo de primera clase: 7B QLoRA (el tamaño del adaptador se elige para que se ajuste: rango 64 en una tarjeta de 16 GB, donde un rango de 256 necesita aproximadamente 17 GB), y un ajuste fino completo de un modelo de ~3B (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B/1B) a través de `mode="full"`, (22.0 GiB medido en una tarjeta de 32 GB a 3B, de los cuales 7.5 GiB son el estado del optimizador que se puede transferir a la RAM del host; no se ha probado si esto funciona de manera aceptable en una tarjeta de 16 GB). Con `--full-ft-offload`, la GPU ocupa mucho menos espacio: con la VRAM limitada en la tarjeta de prueba, un modelo de 3B entrenado con un límite de 6 GiB y modelos de 4B y 7.6B con un límite de 8 GiB. Estos son límites simulados en una tarjeta de 32 GB, no ejecuciones en hardware real de 8 GB. El mismo código elige el tamaño del lote y el límite que se ajustan a la tarjeta que detecta.

La cuantificación de 2 bits (AQLM / QuIP#) queda **fuera del alcance**; una base de 2 bits no se puede combinar limpiamente con pesos de precisión completa, lo que interrumpe el contrato de exportación de adaptador combinable → GGUF → Ollama (que es el objetivo principal de la canalización). En cambio, Backpropagate ofrece las opciones de ajuste — QLoRA, `mode="full"`, `--full-ft-offload` y la ruta de cálculo FP8 (`--fp8`, Blackwell/Hopper—, todas las cuales siguen siendo combinables y exportables.

## Para qué NO sirve Backpropagate

Si su caso de uso es uno de los siguientes, obtendrá mejores resultados con una biblioteca diferente; Backpropagate no es la opción correcta y tratar de hacerlo funcionar costaría más que simplemente utilizar la herramienta adecuada. Leer esta sección antes de comenzar evita el ciclo de instalación y prueba:

- **Ajuste fino de parámetros completos de modelos de 13B+** — Backpropagate realiza un ajuste fino completo de hasta aproximadamente 6B en una GPU de 32 GB y un modelo de la clase 7B con `--full-ft-offload` (consulte [el rango](#qué-aspectos-se-pueden-ajustar-con-precisión-en-una-gpu)). Un ajuste fino completo de un modelo de 13B+ requiere FSDP multi-GPU o una tarjeta más grande. Antes de invertir en esa capacidad de cómputo, evalúe la evidencia en ambos sentidos. [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/) informa que LoRA coincide con el ajuste fino completo cuando se aplica a cada capa y el conjunto de datos se ajusta a la capacidad del adaptador, en aproximadamente dos tercios de la capacidad de cómputo por pasada. [Biderman et al. 2024](https://arxiv.org/abs/2405.09673) descubrió que, en entornos de bajo rango estándar, LoRA tiene un rendimiento significativamente inferior al del ajuste fino completo en código y matemáticas, al tiempo que olvida menos. Para el seguimiento de instrucciones, la personalidad y el trabajo de estilo en conjuntos de datos modestos, QLoRA de hasta 32B suele ser el mejor uso de una sola tarjeta.
- **RL en línea: PPO / GRPO / RLVR** — Backpropagate realiza un ajuste fino SFT de una sola etapa más un ajuste de preferencias sin referencia (ORPO en v1.5; SimPO + KTO en v1.6). Lo que *no* hace es el aprendizaje por refuerzo en línea: PPO, GRPO o RLVR, que requiere un modelo de recompensa o un bucle de generación y puntuación además del paso de entrenamiento. Para estos, utilice TRL directamente o LLaMA-Factory. (El ajuste de preferencias sin referencia se ajusta al rango de una sola etapa porque no hay un modelo de referencia separado que mantener en la memoria; consulte la nota de ORPO en [Inicio rápido](#inicio-rápido).)
- **Entrenamiento multi-nodo** — solo una GPU en una máquina. El multi-GPU en una máquina funciona (a través de `accelerate launch`), pero no está oficialmente admitido.
- **Entrenamiento macOS en el entorno CUDA** — Apple Silicon no tiene CUDA, por lo que la ruta CUDA se ejecuta en una máquina Linux o Windows con una GPU NVIDIA. Aún puede ejecutar el modelo entrenado en un Mac a través de Ollama. Una ruta MLX **experimental y no verificada** (`--backend mlx`) entrena un adaptador LoRA de forma nativa en Apple Silicon; consulte [Apple Silicon (MLX)](#apple-silicon-mlx-vista-previa-no-verificada). Es solo para LoRA-SFT y **no se ha verificado en silicio real** (sin soporte), por lo que, para cualquier cosa que vaya más allá de un SFT LoRA (ORPO, ajuste fino completo, FP8, ejecución múltiple), querrá utilizar la ruta CUDA.
- **Cualquier cosa fuera de las familias de modelos probadas** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Otros modelos a menudo funcionan, pero no están fijados en CI.

Si necesita alguna de estas cosas, utilice una de las bibliotecas enumeradas anteriormente. Son mejores para ello.

## Lo que le ofrece Backpropagate

Cuatro cosas, en una sola instalación:

**1. Una API real de 3 líneas que se ejecuta sin un archivo de configuración.**
El fragmento en la parte superior de este README se ejecuta de principio a fin. No hay `accelerate config`, ni YAML, ni anulaciones de Hydra. Solo `Trainer(model).train(data)` y ya tiene un ajuste fino.

**2. Windows que realmente funciona.**
La mayoría de las bibliotecas de ML tratan a Windows como algo secundario. Backpropagate se desarrolla y se prueba en Windows 11 con tarjetas de la serie RTX 50. La biblioteca gestiona las peculiaridades del tiempo de ejecución por usted; sabe cómo pre-tokenizar sus datos para que el procesamiento multi-hilo de Windows no falle, desactiva automáticamente xformers en las tarjetas RTX 40/50 donde esto provocaría un error y elige la configuración del cargador de datos que no causa problemas. No tiene que saber nada de esto. Simplemente funciona.

**3. Diseñado para ejecuciones sin supervisión.**
El entrenamiento lleva horas. No quiere tener que vigilarlo. Backpropagate está diseñado para que se pueda dejar funcionando:

- Si se queda sin memoria de la GPU, reduce automáticamente a la mitad el tamaño del lote y lo vuelve a intentar, hasta tres veces. No requiere ajustes manuales.
- Si su GPU se calienta demasiado, se detiene hasta que las cosas se enfríen y luego continúa.
- Cada punto de control se escribe de forma atómica; si su computadora portátil falla a la mitad del guardado, el punto de control anterior y válido sigue intacto.
- Cada ejecución de entrenamiento obtiene un ID único que se estampa en cada línea del registro, en cada punto de control y en cada entrada de Weights & Biases. Si algo sale mal, un ID permite a un mantenedor correlacionar todo.
- Los errores vienen con códigos estables (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, etc.) para que pueda buscar en sus registros y en la [guía de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para encontrar la solución. Los fallos específicos de CUDA tienen una [página de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) dedicada.

**4. Un solo comando, desde el adaptador entrenado hasta `ollama run`.**
Muchas bibliotecas entrenan un modelo. Pocas de ellas se apartan de su camino cuando desea utilizarlo realmente. Backpropagate exporta a GGUF (el formato que utiliza Ollama) y registra un modelo de Ollama en un solo comando. Pasa de "entrenamiento completado" a "puedo chatear con mi ajuste fino" en unos 30 segundos.

## Inicio rápido

Desde la línea de comandos, con un conjunto de datos de ejemplo de 5 conversaciones:

```bash
pipx install "backpropagate[standard]"
curl -LO https://raw.githubusercontent.com/mcp-tool-shop-org/backpropagate/main/examples/quickstart.jsonl

backprop train --data quickstart.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 10
backprop generate ./output "What is Python?"      # did it learn anything?
backprop export ./output --format gguf --quantization q4_k_m --ollama --ollama-name my-first-finetune
ollama run my-first-finetune
```

`backprop train` escribe el adaptador para `./output` (cámbialo con `--output`). En Python, lo mismo se hace así:

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")
```

Utilice un entorno virtual con `pip install "backpropagate[standard]"` para la API de Python; `pipx` instala el comando `backprop` en su propio entorno, por lo que `import backpropagate` no lo encontrará.

**Qué necesita la exportación GGUF.** La exportación fusiona su adaptador con el modelo base y lo convierte con el script de conversión de llama.cpp. Necesita:

- una copia del código fuente de llama.cpp (`git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp`) más `pip install sentencepiece protobuf` en el mismo entorno, o
- Unsloth con su propio llama.cpp ya compilado.

Con `--ollama`, la cuantización `q4_k_m` se realiza mediante `ollama create`, por lo que no es necesario compilar nada. Backpropagate nunca permite que Unsloth instale paquetes del sistema para compilar llama.cpp por usted; establezca `BACKPROPAGATE_UNSLOTH_AUTO_INSTALL=1` si desea que lo haga. Detalles: [export](https://mcp-tool-shop-org.github.io/backpropagate/handbook/export/).

Para sus propios datos, formatee su archivo JSONL con un ejemplo por línea:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca (`instruction` / `output`), OpenAI chat (`messages`) y formatos de texto sin formato también funcionan; Backpropagate detecta automáticamente el formato.

### El bucle: comprobar los datos, entrenar, evaluar, exportar

```bash
backprop data report my_data.jsonl                     # duplicates, length outliers, format problems
backprop data split my_data.jsonl --heldout-ratio 0.1  # a held-out set the model never trains on
backprop train --data my_data.train.jsonl --steps 200 --output ./run-a
backprop eval <run-id> --heldout my_data.heldout.jsonl # held-out loss + sample generations
backprop eval <run-b> --vs <run-a>                     # did the change help?
backprop export ./run-a --format gguf --ollama --ollama-name my-model
```

La evaluación está diseñada para que no requiera un juez: pérdida retenida más métricas de tareas deterministas (`normalized_exact_match`, `token_f1`, `contains`, `regex`, `pass_rate`). Para utilizar un juez LLM, ejecútelo sobre la salida de `backprop generate`. Consulte [recetas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Ajuste de preferencias (ORPO, SimPO, KTO)

Entrene con preferencias en lugar de demostraciones simples. ORPO no requiere referencias y es de una sola etapa; integra la señal de preferencia en el paso SFT, por lo que no hay un modelo de recompensa o referencia separado y la forma de 3 líneas no cambia. Pase `--method orpo` (CLI) o `method="orpo"` (Python) y proporcione un conjunto de datos de `{prompt, chosen, rejected}` (o simplemente `{chosen, rejected}`) filas:

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

La tasa de aprendizaje predeterminada se reduce automáticamente a `8e-6` para ORPO (la pérdida es más pronunciada que en el SFT simple); ajuste `--orpo-beta` (predeterminado `0.1`) para ponderar la penalización de la razón de probabilidades. ORPO es solo `mode="lora"`.

**Novedad en v1.6: SimPO y KTO.** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) no requiere referencias, utiliza una recompensa normalizada por longitud y toma los mismos datos emparejados `{prompt, chosen, rejected}` que ORPO (`--simpo-beta`, `--simpo-gamma`). `--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) toma datos **no emparejados** `{prompt, completion, label}`: valoraciones positivas/negativas por ejemplo, para la gran clase de comentarios que no son pares A/B seleccionados; equilibra automáticamente los pesos de pérdida deseables/indeseables a partir de los recuentos de sus etiquetas. Ambos son solo `mode="lora"` y permanecen dentro del ámbito SFT de una sola GPU (sin un modelo de referencia separado). Consulte el [manual de ajuste de preferencias](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) para saber cuál utilizar. Para el RL en línea (PPO/GRPO), consulte [Para qué NO sirve Backpropagate](#para-qué-no-sirve-backpropagate).

### SFT de rastreo de razonamiento (destilación R1)

Destile un modelo de razonamiento de forma sencilla. Pase `--reasoning-trace` (CLI) o `Trainer(..., reasoning_trace=True)` (Python) y proporcione rastros que mantengan una cadena de pensamiento `<think>...</think>` dentro del turno del asistente; la mitad SFT pura de la destilación [DeepSeek-R1](https://arxiv.org/abs/2501.12948), no se requiere RL. Backpropagate mantiene `<think>` en el objetivo de entrenamiento, elimina los rastros vacíos o demasiado largos (filtrado de la longitud del rastro) y aumenta el valor predeterminado de `max_seq_length` a 8192 para el CoT más largo. Lo más importante es que `<think>` sigue siendo **texto sin formato**: sin tokens especiales, sin cambio de tamaño del embedding, por lo que el GGUF fusionado sigue exportándose a Ollama como cualquier otro ajuste fino. Solo SFT. Consulte la [receta de rastreo de razonamiento](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) para conocer la forma del conjunto de datos y los tokens ajustables.

### Apple Silicon (MLX): vista previa no verificada

> ⚠️ **Vista previa no verificada: no forma parte del conjunto de funciones admitidas.** El "rail" MLX está construido y se han realizado pruebas unitarias, pero **no** se ha verificado con pruebas exhaustivas en Apple Silicon real (`mlx-lm` es exclusivo de Apple y no se puede ejecutar en los equipos NVIDIA en los que se desarrolla Backpropagate). Considere todo lo que se indica a continuación como experimental, utilícelo bajo su propio riesgo y [informe de cualquier anomalía](#informar-de-errores) si lo ejecuta en un Mac de la serie M.

**Una API, dos "rails".** CUDA es el backend canónico y verificado; MLX es un segundo "rail" que se entrena en un Mac de la serie M a través del conjunto de herramientas [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) de Apple (memoria unificada, sin CUDA). La forma de 3 líneas selecciona el "rail" según el hardware: `backend='auto'` (el valor predeterminado) enruta a CUDA en NVIDIA y a MLX en Apple Silicon, por lo que los equipos CUDA existentes son idénticos a nivel de bytes:

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

El "rail" MLX es **solo SFT LoRA**: no ORPO, no FP8, no `mode='full'`, no ejecuciones múltiples (cada una se rechaza con `CONFIG_INVALID_SETTING`; utilice `backend='cuda'`/`'auto'` en un equipo NVIDIA para ello). El adaptador resultante es un archivo safetensors simple y se exporta a Ollama a través del mismo camino que el "rail" CUDA.

> Forzar `--backend mlx` en un host que no sea de Apple genera un error con `CONFIG_INVALID_SETTING`; la falta del conjunto de herramientas `mlx_lm` en un Mac genera `DEP_MLX_UNAVAILABLE`.

Para obtener más flujos de trabajo de principio a fin (ajuste fino y carga en HF Hub, reanudación después de quedarse sin memoria, SLAO de ejecución múltiple en una campaña larga, etc.), consulte la [página de recetas del manual](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interfaz de usuario web (opcional)

Si prefiere hacer clic en lugar de escribir Python, instale el complemento de la IU y ejecútelo:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Abra la URL que muestra, `http://127.0.0.1:7862/?token=...` (cada ejecución crea un nuevo token; el primer inicio construye la interfaz y puede tardar uno o dos minutos). Es una interfaz web local para el entrenamiento: inicie una ejecución, una serie de ejecuciones múltiples o una exportación, observe el progreso en tiempo real (pasos, pérdida, tiempo restante, temperatura y memoria de la GPU) y deténgala con un punto de control guardado. Cada trabajo se ejecuta en su propio proceso, uno a la vez, y al volver a cargar la página se reanuda el trabajo en curso. La página de Conjunto de datos muestra el contenido de un archivo, guarda una copia limpia (se eliminan las repeticiones y los ejemplos vacíos) y la pasa al formulario de entrenamiento. Las ejecuciones anteriores y los modelos en su caché de Hugging Face tienen sus propias páginas, y cada configuración tiene una "i" que la explica. El [recorrido por la interfaz de usuario web](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/) muestra cada página. La IU es solo local de forma predeterminada. Para exponerla a otros dispositivos, consulte [Interfaz de usuario web](#interfaz-de-usuario-web) a continuación para conocer el contrato de seguridad `--share` + `--auth`.

## Entrenamiento con múltiples ejecuciones

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

Esto ejecuta cinco ciclos de entrenamiento, fusionando el adaptador entre ciclos de una manera que preserva el conocimiento previo al tiempo que incorpora nuevos ejemplos. La técnica se basa en investigaciones recientes sobre el aprendizaje continuo; consulte [Referencias](#referencias) al final de este archivo README.

La versión de la CLI:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Reanudar desde un punto de control

Un entrenamiento de 5 ciclos que falla en el ciclo 4 se puede recuperar. Cada sesión de múltiples ciclos escribe su ID de ciclo en el historial y el manifiesto del punto de control en el disco, por lo que reanudar donde lo dejó es un solo comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

El comportamiento predeterminado de `backprop multi-run` (sin `--resume`) detecta automáticamente una entrada en curso en el mismo directorio de salida y la continúa. Para forzar un inicio limpio, apunte a un directorio de salida nuevo.

## Historial de entrenamiento

Cada invocación de `backprop train` y `backprop multi-run` registra una fila en `<output>/run_history.json`: modelo utilizado, conjunto de datos, hiperparámetros, estado, pérdida final, historial de pérdidas. Puede listar e inspeccionar ejecuciones anteriores:

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

La interfaz web de Reflex se activa opcionalmente: instálela con `pipx install "backpropagate[ui]"` e inicie:

```bash
backprop ui --port 7862
```

La interfaz de usuario se ejecuta localmente: abra la URL que muestra, `http://127.0.0.1:7862/?token=...`. Sin `--auth`, cada inicio genera un nuevo token y la interfaz de usuario rechaza las solicitudes que no lo incluyan. Desde allí, puede ver y limpiar un conjunto de datos, entrenar (una sola ejecución o una ejecución múltiple), observar la ejecución en vivo, detenerla con un punto de control guardado y exportar el resultado. Cada trabajo se ejecuta en su propio proceso, uno a la vez, y cerrar la interfaz de usuario lo detiene. El [recorrido de la interfaz de usuario web](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/) recorre cada página con capturas de pantalla.

Para exponerla a otros dispositivos (otras personas en su red, una URL pública, etc.), debe emparejar `--share` (o `--host`) con `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` sin `--auth` se cierra con un error. La razón: `--share` publica una URL a la que puede acceder cualquier persona en Internet, y sin autenticación, esto significa que cualquiera puede controlar su canalización de entrenamiento y leer su token HuggingFace. No hay opción para desactivar esto; si no desea configurar credenciales, utilice el reenvío de puertos SSH en su lugar:

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
- Anular: establezca `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- La anulación se valida mediante una lista de denegación: los caminos del sistema o de las credenciales (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) se rechazan.

## Notas de la plataforma

**Requisitos:** Python 3.11+ · GPU NVIDIA con CUDA · PyTorch 2.0+. Una tarjeta de 8 GB entrena los modelos preestablecidos de 1B a 3B, 16 GB un modelo de 7B y 32 GB hasta 32B con QLoRA.

Python 3.10 ya no es compatible (fin de vida útil el 2026-10-01); el mínimo es 3.11 a partir de v1.9.0. Python 3.11 es la versión más probada.

Backpropagate gestiona las peculiaridades del tiempo de ejecución al entrenar en diferentes plataformas, pero no puede solucionar los problemas que surgen durante la instalación. Los dos más comunes son:

- **Rueda CUDA incorrecta.** PyTorch se publica como un único archivo binario por versión de CUDA. Si elige la incorrecta, obtendrá silenciosamente PyTorch solo para la CPU y el entrenamiento será increíblemente lento. Utilice el selector de ruedas en <https://pytorch.org/get-started/locally/> para su controlador. Ejecute `nvidia-smi` para ver su versión de controlador/CUDA.
- **Windows + exportación GGUF.** El extra de `[export]` construye `llama-cpp-python` desde el código fuente, lo que requiere las herramientas de compilación de Visual Studio (componente C++) y CMake.

**macOS:** el carril de CUDA no es compatible (no hay CUDA); un `trainer.train()` con enrutamiento de CUDA genera `DEP_GPU_NOT_AVAILABLE`, y puede ejecutar el adaptador entrenado en un Mac a través de Ollama. Un carril MLX **experimental y no verificado** (`--backend mlx`, `pip install 'backpropagate[mlx]'`) entrena un adaptador LoRA de forma nativa en Apple Silicon a través de `mlx_lm.lora`: solo SFT LoRA y **no verificado en silicio real** (consulte [Apple Silicon (MLX)](#apple-silicon-mlx-vista-previa-no-verificada)). Para el camino de CUDA o para ORPO / ajuste fino completo / FP8 / ejecución múltiple, utilice una máquina Linux o Windows con CUDA.

Consulte la [página de la guía de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para obtener la guía completa de solución de problemas de instalación y la página dedicada de [solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) para problemas de controlador / VRAM / xformers / bf16-vs-fp16.

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

Referencia completa en [la página de la guía de la CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) o `backprop <subcommand> --help`.

## Configuración

Se puede anular cada configuración con una variable de entorno utilizando el prefijo `BACKPROPAGATE_`:

| Variable | Predeterminado | Notas |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Forzar registros JSON o de consola |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modelo predeterminado |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Tasa de aprendizaje |
| `BACKPROPAGATE_LORA__R` | `256` | LoRA rango. Establecerlo desactiva la selección automática del tamaño del adaptador (consulte `--lora-preset` en [Modelos preestablecidos](#modelos-preestablecidos)). |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sandbox del sistema de archivos de la interfaz de usuario |

Las claves anidadas utilizan doble guion bajo (`MODEL__NAME`, no `MODEL_NAME`). La referencia completa está en [la página de la guía de las variables de entorno](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Modelos preestablecidos

| Modelo preestablecido | Memoria de la GPU | Licencia | Notas |
|---|---|---|---|
| Qwen-3.5-4B | 6 / 7 / 11 GB | Apache 2.0 | Predeterminado recomendado para sub-5B. La mejor calidad para este tamaño. |
| Phi-4-mini-3.8B | 6 / 7 / 12 GB | MIT | Fuerte en razonamiento / matemáticas / código. Licencia estrictamente limpia. |
| SmolLM3-3B | 4 / 5 / 10 GB | Apache 2.0 | Receta totalmente abierta. Contexto nativo de 64K. |
| Qwen 2.5 7B | 9 / 11 / 17 GB | Apache 2.0 | Predeterminado existente. La mejor calidad de los modelos preestablecidos de 7B heredados. |
| Qwen 2.5 3B | 4 / 6 / 10 GB | Qwen-Research | ⚠ licencia de investigación: consulte los términos de la licencia de Qwen antes de utilizarlo comercialmente. |
| Llama 3.2 3B | 4 / 6 / 9 GB | Comunidad Llama | Alternativa sólida a Qwen 3B con restricciones permisivas. |
| Llama 3.2 1B | 2 / 3 / 5 GB | Comunidad Llama | Para experimentos rápidos en tarjetas pequeñas. |
| Mistral 7B | 6 / 8 / 14 GB | Apache 2.0 | Comparable a Qwen 7B, plantilla de chat diferente. |
| Llama-3.1-8B | 9 / 11 / 18 GB | Llama-3.1-Community | 8B QLoRA, 128K de contexto nativo (la cláusula de >700M de usuarios activos mensuales requiere una licencia Meta separada). |
| **Qwen2.5-14B** | 18.7 GiB máximo a 4096 ctx (QLoRA) | Apache 2.0 | **La opción principal para uso diario con 32 GB.** rank/alpha 32, 8 bits AdamW. Los pesos de 4 bits por sí solos ocupan aproximadamente 8.5 GB; una ventana completa de 4096 tokens requiere el resto. |
| Mistral-Small-24B | 22.8 GiB máximo a 4096 ctx (QLoRA) | Apache 2.0 | 24B QLoRA en una tarjeta de 32 GB. Los pesos de 4 bits por sí solos ocupan aproximadamente 18 GB. |
| **Qwen2.5-32B** | 26.0 GiB máximo a 2048 ctx (QLoRA) | Apache 2.0 | **Lo máximo que se puede usar con 32 GB.** Se ajusta a `max_len 2048` con 8 bits AdamW. |

Otros modelos a menudo funcionan; las filas anteriores son los ajustes preestablecidos seleccionados; el nivel de 14B a 32B está QLoRA-optimizado para una tarjeta de 32 GB (el rango medido). Para los ajustes preestablecidos de hasta 8B, las tres cifras son QLoRA estimaciones para los tamaños de los adaptadores `fast`, `balanced` y `quality` en el lote 1 con ejemplos de 2048 tokens; se inclinan hacia el lado alto, y los ejemplos más cortos utilizan menos. El tamaño del adaptador se elige para su tarjeta: `--lora-preset auto` (el valor predeterminado) toma el mayor de `quality` (rango 256 en cada capa lineal, según Biderman 2024 y Thinking Machines 2025), `balanced` (rango 64 en cada capa lineal) y `fast` (rango 16 en dos capas por bloque) que se ajuste a la memoria libre en su GPU. Indique uno para forzarlo. `backprop estimate-vram` imprime la estimación para cualquier modelo y configuración.

## Solución de problemas

Un índice breve de los fallos más comunes que ocurren al ejecutarlo por primera vez. El índice inverso completo está en [la página del manual de solución de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Para obtener información detallada sobre el controlador / VRAM / precisión mixta, consulte [la página de solución de problemas de CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Síntoma | Código de error | Solución |
|---|---|---|
| La GPU se queda sin memoria a mitad del entrenamiento | `RUNTIME_GPU_OOM` | Automático: Backpropagate reduce a la mitad el tamaño del lote y lo vuelve a intentar hasta 3 veces. Para optar por no hacerlo: `Trainer(oom_recovery=False)`. Para forzar un tamaño más pequeño: `--batch-size 1`. |
| HuggingFace devuelve 401 / "modelo no encontrado" | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` y vuelva a intentarlo. Para errores tipográficos, copie el ID exacto de <https://huggingface.co/models>. |
| `register_with_ollama` conexión rechazada | `DEP_OLLAMA_REGISTRATION_FAILED` | Inicie el demonio: `ollama serve`. Instale desde <https://ollama.com>. Se puede volver a intentar. |
| El disco está lleno durante el guardado del punto de control | `STATE_CHECKPOINT_INVALID` | Las escrituras atómicas dejan un directorio `.partial` en caso de fallo; es seguro eliminarlo. El punto de control anterior y correcto está intacto. |
| El entrenamiento se pausa debido al sobrecalentamiento de la GPU | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automático: Backpropagate se pausa en el umbral de temperatura y se reanuda a medida que la GPU se enfría. Mejore el flujo de aire si esto sigue sucediendo. |
| `backprop ui --share` rechazado | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Pase `--auth user:password` o utilice el reenvío de puertos SSH en su lugar (consulte [la interfaz de usuario web](#interfaz-de-usuario-web)). |
| La exportación de GGUF falló en el primer intento | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; en Windows, también necesita las herramientas de compilación de Visual C++ + CMake. |

## Informar de errores

Cuando algo falla, Backpropagate imprime una línea al inicio, como `run_started run_id=<uuid>`, y vincula el mismo ID a cada línea del registro, a cada punto de control y a cada entrada de Weights & Biases. **Incluya el `run_id` en cualquier informe de errores**, ya que permite a un mantenedor correlacionar todo para esa ejecución específica.

Un buen informe de errores incluye:

1. **El `run_id`**: el UUID impreso al inicio. Un UUID permite a un mantenedor correlacionar cada línea del registro, cada punto de control y cada entrada de Weights & Biases para esa ejecución específica.
2. **El código de error**: la línea `[CODE_NAME]: message` en stderr. Consulte [códigos de error](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) para obtener el catálogo de códigos estables.
3. **El rastreo de pila redactado**. Stderr se redacta automáticamente en el modo no detallado (los tokens de Bearer, `sk-*`, `hf_*`, las claves de AWS, los pares `password=` / `token=` / `api_key=` se eliminan; es seguro pegarlo. Para obtener el rastreo de pila completo y no redactado, vuelva a ejecutarlo con `BACKPROPAGATE_DEBUG=1` (o `--verbose`); revíselo antes de publicarlo.
4. **La salida de `backprop info`**. Un comando imprime Python / PyTorch / CUDA / modelo de GPU / VRAM / SO / extras instalados; todo lo que el mantenedor necesita para identificar una regresión específica de la plataforma.

La [plantilla de informe de errores](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) solicita explícitamente cada uno de estos para que la evaluación inicial se realice rápidamente. Las preguntas, ideas o hilos de "esto, ¿es normal?" deben incluirse en [Discusiones de GitHub](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Los problemas de seguridad deben informarse de forma privada a través del [Formulario de asesoramiento de seguridad de GitHub](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new); consulte [SECURITY.md](SECURITY.md) para conocer la política y los plazos de respuesta.

## Privacidad

Todo el entrenamiento se realiza localmente en su GPU. Backpropagate no realiza ninguna solicitud de red, excepto para descargar modelos de HuggingFace (lo que usted inicia). No hay telemetría, no hay dependencia de la nube.

## Referencias

Los valores predeterminados de Backpropagate y el modo de entrenamiento de múltiples ejecuciones se basan en investigaciones recientes. Si está interesado en las técnicas subyacentes:

- **Hu et al. 2021.** *LoRA: Adaptación de bajo rango de modelos de lenguaje grandes.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — el documento fundamental que introduce LoRA, que es la forma en que Backpropagate entrena los adaptadores de manera eficiente.
- **Biderman et al. 2024.** *LoRA Aprende menos y olvida menos.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidencia empírica de que LoRA con rango 256 y objetivos totalmente lineales coincide con la calidad del ajuste fino completo en la mayoría de las tareas posteriores al entrenamiento, utilizando el 67 % de la capacidad de cálculo. Impulsa la configuración predeterminada v1.3 de LoRA de Backpropagate.
- **Thinking Machines 2025.** *LoRA Sin arrepentimientos.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — la continuación práctica que identifica la corrección de 10× en la tasa de aprendizaje frente al ajuste fino completo necesaria a un rango LoRA alto.
- **Kirkpatrick et al. 2017.** *Superando el olvido catastrófico en las redes neuronales.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la caracterización original de por qué las redes neuronales "olvidan" el entrenamiento anterior cuando se realiza un ajuste fino con nuevos datos (EWC — Consolidación elástica de pesos).
- **Wang et al. 2023.** *Aprendizaje de subespacio ortogonal para el aprendizaje continuo de modelos de lenguaje.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, un enfoque anterior para utilizar LoRA para el aprendizaje continuo mediante la restricción de los nuevos adaptadores a subespacios ortogonales.
- **Yadav et al. 2023.** *TIES-Merging: Resolviendo la interferencia al fusionar modelos.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — una técnica fundamental para fusionar múltiples modelos ajustados sin interferencias.
- **Qiao & Mahdavi 2025.** *Fusionar antes de olvidar: un único LoRA de aprendizaje continuo mediante la fusión continua.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — el algoritmo específico que implementa el fusionador de múltiples ejecuciones de Backpropagate. Un preimpreso de diciembre de 2025; Backpropagate es el primer usuario conocido de este documento.

## Licencia

MIT — ver [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
