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
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/backpropagate"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/backpropagate/badge" alt="OpenSSF Scorecard"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

# Addestra un adattatore. Caricalo su Ollama. Passa oltre

Backpropagate è una libreria Python per l'affinamento di modelli linguistici di grandi dimensioni su una singola GPU. Tre righe di codice addestrano un modello da 7B su una scheda da 16 GB. Un comando aggiuntivo lo esporta su Ollama in modo che tu possa eseguire `ollama run` per il tuo modello affinato. Funziona perfettamente su Windows.

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

Questo è tutto. Non c'è un file di configurazione YAML. Non c'è una procedura di avvio con `accelerate launch`. Non c'è un tutorial separato per "convertilo ora in GGUF". Se hai una GPU CUDA e un file JSONL con i tuoi dati di addestramento, ti servono solo tre righe per ottenere un modello affinato funzionante.

## Installa

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

Se desideri le funzionalità opzionali, sostituisci l'installazione con una di queste:

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

Preferisci Docker? Anche `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` funziona. Sono disponibili immagini sia per `linux/amd64` che per `linux/arm64`, quindi gli utenti di Apple Silicon e ARM Linux ottengono un'immagine nativa. Un file `compose.yaml` standard per "UI in un container" si trova nella directory principale del repository: `docker compose up` avvia l'interfaccia utente web su `http://localhost:7860` con un volume persistente `~/.backpropagate`.

## Dove si colloca Backpropagate

Esistono diverse buone librerie per l'affinamento di LLM. Ognuna di esse è ottima per cose diverse:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**: se ti piacciono le configurazioni YAML e desideri una community di ricette da cui copiare.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**: se desideri DPO/PPO/RLHF e un'interfaccia utente web.
- **[Unsloth](https://github.com/unslothai/unsloth)**: se hai bisogno dell'addestramento più rapido possibile e utilizzi una famiglia di modelli supportata.
- **[torchtune](https://github.com/pytorch/torchtune)**: se desideri le ricette PyTorch native di Meta che puoi modificare.

Backpropagate è l'opzione mancante: **un'API Python di 3 righe per gli utenti singoli su una singola GPU di consumo che desiderano addestrare un adattatore e caricarlo.** Nessun YAML, nessuna GUI, nessun RL online (PPO/GRPO), nessun nodo multiplo. Solo il ciclo di cui tutti hanno realmente bisogno e il passaggio di esportazione che crea problemi.

Se hai provato una delle librerie di cui sopra e hai avuto problemi con la procedura di configurazione dei file, o hai riscontrato una lacuna nella famiglia di modelli, o hai desiderato impostazioni predefinite per Windows, Backpropagate è la soluzione giusta per te.

## Cosa puoi affinare su una GPU di consumo da 16 GB

Ecco i limiti pratici su una scheda da 16 GB (RTX 4080 / 5080 / 4070 Ti Super):

| Modello | Metodo | Stato |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | Buono. Lunghezza di sequenza completa, spazio sufficiente. |
| SmolLM3-3B / Qwen2.5-3B / Llama-3.2-3B / Llama-3.2-1B | `mode="full"` (affinamento completo) | v1.4: passa `--mode=full` in `backprop train` o `Trainer(..., mode="full")`. Carica i pesi a precisione completa (bf16): niente 4 bit, niente adattatore; il checkpointing del gradiente e l'Adam a 8 bit a pagine mantengono l'impronta entro i 16 GB. |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | Standard. ~7-8 GB. Impostazioni predefinite di Backpropagate. |
| Llama-3 13B | QLoRA + sample packing | Stretto, ma funziona. Utilizza sequenze più brevi. |
| Mixtral 8x7B (47 miliardi di parametri totali) | — | Fuori portata: la quantizzazione a 2 bit (AQLM / QuIP#) interrompe il contratto di adattatore unificabile + esportazione GGUF, quindi è stata abbandonata nella [breve descrizione della traiettoria v1.5](docs/V1_5_BRIEF.md). Su una scheda da 16 GB, utilizza una base ≤8B. |

`mode="full"` supporta modelli fino a **4 miliardi di parametri**. Le quattro impostazioni nella riga di affinamento completo sopra sono autentici ~3B (numero effettivo di parametri 3,08–3,24B) e si adattano a una scheda da 16 GB. La classe 3,8–4B (Phi-4-mini-3,8B, Qwen-3.5-4B) è accettata anche dal limite massimo, ma richiede una scheda da **24 GB o superiore** per l'affinamento completo: i soli pesi e i gradienti si avvicinano già a 16 GB prima dell'ottimizzatore e delle attivazioni, quindi su una scheda da 16 GB utilizza `mode="lora"` per questi (si trovano nella riga LoRA). I modelli >4B restituiscono `RUNTIME_FULL_FT_MODEL_TOO_LARGE`.

La quantizzazione a 2 bit (AQLM / QuIP#) è **fuori portata**. È stata prevista per la v1.4, quindi è stata abbandonata nella [breve descrizione della traiettoria v1.5](docs/V1_5_BRIEF.md): una base a 2 bit non può essere unita in modo pulito ai pesi a precisione completa, il che interrompe il contratto di adattatore unificabile → GGUF → Ollama (il punto principale della pipeline). Le leve di ottimizzazione che Backpropagate offre sono invece il percorso di calcolo FP8 v1.5 (`--fp8`, Blackwell/Hopper) e `mode="full"` per i modelli ≤4B: entrambi rimangono unificabili ed esportabili.

Per i modelli da 3B e inferiori, l'affinamento completo (non solo LoRA) è fattibile su 16 GB ed è ora disponibile nella v1.4 come `mode="full"`. Passa `Trainer(..., mode="full")` o `backprop train --mode=full --model phi-4-mini-3.8b` per abilitarlo. Un blocco rigido rifiuta la modalità per i modelli > 4B con `RUNTIME_FULL_FT_MODEL_TOO_LARGE`, indicando LoRA e le impostazioni predefinite inferiori a 4B come opzioni di ripristino. Consulta [la pagina del manuale sull'affinamento completo](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/) per i calcoli di configurazione e il confronto di qualità Biderman 2024 / Thinking Machines 2025. Per i modelli da 7B in su, l'affinamento completo richiede una GPU da 24 GB o superiore: valuta la possibilità di noleggiare un'A100 nel cloud o attieniti a LoRA, che le ricerche più recenti dimostrano che corrisponde alla qualità dell'affinamento completo nella maggior parte delle attività post-addestramento (vedi [la sezione anti-presentazione](#what-backpropagate-is-not-for) per le citazioni).

## Per cosa NON è adatto Backpropagate

Se il tuo caso d'uso è tra quelli elencati di seguito, otterrai risultati migliori con una libreria diversa: Backpropagate non è la scelta giusta e cercare di farlo funzionare costerebbe più che semplicemente utilizzare lo strumento giusto. Leggere questa sezione prima di iniziare ti eviterà di installare e poi abbandonare il progetto:

- **Ottimizzazione completa dei parametri per modelli da 7B+** — Backpropagate utilizza LoRA/QLoRA, che addestra un piccolo adattatore anziché aggiornare tutti i pesi. Per i modelli da 7B e superiori, l'ottimizzazione completa richiede 24 GB o più di memoria GPU e non è adatta per una scheda consumer da 16 GB. Per i modelli da 3B e inferiori, l'ottimizzazione completa è fattibile con 16 GB ed è disponibile nella versione 1.4 come `mode="full"` (passare `Trainer(..., mode="full")` o `--mode=full` dalla riga di comando; un controllo rigido genera `RUNTIME_FULL_FT_MODEL_TOO_LARGE` per i modelli > 4B e nomina LoRA + i preset inferiori a 4B come soluzioni alternative). Nel complesso: le ricerche recenti ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) mostrano che LoRA, con la configurazione corretta, corrisponde alla qualità dell'ottimizzazione completa nella maggior parte delle attività di post-addestramento (seguire le istruzioni, adattamento al dominio, personalità/stile) con il 67% della potenza di calcolo. Quindi, per il lavoro che la maggior parte degli utenti desidera, non si perde nulla utilizzando LoRA. `mode="full"` è disponibile per i casi in cui si è misurata una differenza di qualità e si è deciso di utilizzare una maggiore potenza di calcolo. Se si ha realmente bisogno dell'ottimizzazione completa di un modello da 7B+, utilizzare direttamente HuggingFace `transformers.Trainer` su una scheda da 24 GB o superiore.
- **RL online — PPO / GRPO / RLVR** — Backpropagate esegue un addestramento SFT a fase singola più un'ottimizzazione delle preferenze senza riferimento (ORPO è disponibile nella versione 1.5; SimPO/KTO sono in programma). Ciò che *non* fa è l'apprendimento per rinforzo online: PPO, GRPO o RLVR, che richiedono un modello di ricompensa o un ciclo di generazione e valutazione in aggiunta alla fase di addestramento. Per questi, utilizzare direttamente TRL o LLaMA-Factory. (L'ottimizzazione delle preferenze senza riferimento si adatta all'ambito a fase singola perché non è necessario mantenere un modello di riferimento separato in memoria; vedere la nota ORPO in [Guida rapida](#guida-rapida)).
- **Addestramento multi-nodo** — una singola GPU su una singola macchina. L'addestramento multi-GPU su una singola macchina funziona (tramite `accelerate launch`), ma non è ufficialmente supportato.
- **Addestramento macOS con CUDA** — Apple Silicon non dispone di CUDA, quindi il percorso CUDA deve essere eseguito su una macchina Linux o Windows con una GPU NVIDIA. È comunque possibile eseguire il modello addestrato su un Mac tramite Ollama. **Novità nella versione 1.5:** un percorso MLX sperimentale (`--backend mlx`) addestra un adattatore LoRA nativamente su Apple Silicon — vedere [Apple Silicon (MLX)](#apple-silicon-mlx--sperimentale-v15). È solo per LoRA-SFT ed è stato implementato, ma non ancora verificato su hardware reale, quindi per qualsiasi cosa oltre a un LoRA SFT (ORPO, ottimizzazione completa, FP8, esecuzioni multiple), è comunque consigliabile utilizzare il percorso CUDA.
- **Qualsiasi cosa al di fuori delle famiglie di modelli testate** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Altri modelli spesso funzionano, ma non sono inclusi nei test CI.

Se hai bisogno di una di queste cose, utilizza una delle librerie elencate sopra. Sono più adatte a questo scopo.

## Cosa offre Backpropagate

Quattro cose, in una singola installazione:

**1. Una vera API a 3 righe che funziona senza un file di configurazione.**
Lo snippet all'inizio di questo README viene eseguito dall'inizio alla fine. Nessuna configurazione `accelerate`, nessun YAML, nessuna sovrascrittura Hydra. Basta `Trainer(model).train(data)` e si ottiene un modello ottimizzato.

**2. Windows che funziona davvero.**
La maggior parte delle librerie ML trattano Windows come un ripensamento. Backpropagate è testata in modo completo su Windows + RTX 5080. La libreria gestisce le peculiarità del runtime per te: sa come pre-tokenizzare i dati in modo che l'elaborazione parallela di Windows non si blocchi, disabilita automaticamente xformers sulle schede RTX 40/50 dove causerebbe problemi e seleziona le impostazioni del caricatore di dati che non causano errori. Non è necessario sapere nulla di tutto questo. Funziona semplicemente.

**3. Progettato per esecuzioni non presidiate.**
L'addestramento richiede ore. Non si vuole doverlo monitorare costantemente. Backpropagate è progettata per essere lasciata in esecuzione:

- Se si esaurisce la memoria GPU, dimezza automaticamente la dimensione del batch e riprova, fino a tre volte. Nessuna regolazione manuale.
- Se la GPU diventa troppo calda, si mette in pausa fino a quando le cose non si raffreddano e poi continua.
- Ogni checkpoint viene scritto in modo atomico: se il laptop si blocca durante il salvataggio, il checkpoint precedente e valido rimane intatto.
- Ogni esecuzione di addestramento riceve un ID univoco che viene stampato su ogni riga del log, su ogni checkpoint e su ogni voce di Weights & Biases. Se qualcosa va storto, un singolo ID consente a un manutentore di correlare tutto.
- Gli errori sono accompagnati da codici stabili (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, ecc.) in modo da poter cercare nei log e nella [guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) per trovare la soluzione. I guasti specifici di CUDA hanno una [pagina dedicata alla risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

**4. Un solo comando, dall'adattatore addestrato a `ollama run`.**
Molte librerie addestrano un modello. Poche di esse si mettono di mezzo quando si vuole effettivamente utilizzarlo. Backpropagate esporta in GGUF (il formato utilizzato da Ollama) e registra un modello Ollama con un solo comando. Si passa da "addestramento completato" a "posso chattare con il mio modello ottimizzato" in circa 30 secondi.

## Guida rapida

Il repository include un piccolo set di dati di esempio in modo che lo snippet all'inizio di questo README funzioni su un'installazione pulita:

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

Questo addestra un adattatore Qwen 2.5 7B su 5 brevi conversazioni in formato ShareGPT, quindi esporta il risultato in GGUF. Per i propri dati, formattare il file JSONL con un esempio per riga:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

I formati Alpaca (`instruction` / `output`), OpenAI chat (`messages`) e testo semplice funzionano anche: Backpropagate rileva automaticamente il formato.

### Ottimizzazione delle preferenze (ORPO)

Novità nella versione 1.5: addestrare con le preferenze anziché con semplici dimostrazioni. ORPO non richiede riferimenti ed è a fase singola: integra il segnale di preferenza nella fase SFT, quindi non è necessario un modello di ricompensa o di riferimento separato e la struttura a 3 righe rimane invariata. Passare `--method orpo` (CLI) o `method="orpo"` (Python) e fornire un set di dati di righe `{prompt, chosen, rejected}` (o solo `{chosen, rejected}`):

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

Il tasso di apprendimento predefinito si riduce automaticamente a `8e-6` per ORPO (la perdita è più marcata rispetto al semplice SFT); regola `--orpo-beta` (predefinito `0.1`) per ponderare la penalità del rapporto di probabilità. Nella versione 1.5, ORPO è impostato su `mode="lora"`. SimPO e KTO sono previsti come sviluppi successivi; per l'RL online (PPO/GRPO), consulta [What Backpropagate is NOT for](#what-backpropagate-is-not-for).

### SFT con traccia del ragionamento (distillazione R1)

Nuovo nella versione 1.5: esegui la distillazione di un modello di ragionamento in modo semplice. Passa `--reasoning-trace` (CLI) o `Trainer(..., reasoning_trace=True)` (Python) e fornisci tracce che mantengano una catena di pensiero `<think>...</think>` all'interno del turno dell'assistente: la metà di SFT puro di [DeepSeek-R1](https://arxiv.org/abs/2501.12948), senza necessità di RL. Backpropagate mantiene `<think>` nell'obiettivo di addestramento, elimina le tracce vuote o troppo lunghe (filtraggio della lunghezza della traccia) e aumenta il valore predefinito di `max_seq_length` a 8192 per la catena di pensiero più lunga. In modo cruciale, `<think>` rimane in **testo semplice**: nessun token speciale, nessuna ridimensionamento dell'embedding, quindi il GGUF unificato può comunque essere esportato su Ollama come qualsiasi altro modello ottimizzato. Solo SFT. Consulta la [ricetta reasoning-trace](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) per la forma del set di dati e i token regolabili.

### Apple Silicon (MLX) — sperimentale, versione 1.5

Nuovo nella versione 1.5: **un'API, due percorsi.** CUDA rimane il backend canonico e verificato; MLX è un secondo percorso che esegue l'addestramento su un Mac della serie M tramite il toolchain [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) di Apple (memoria unificata, senza CUDA). La stessa struttura di 3 righe seleziona il percorso in base all'hardware: `backend='auto'` (predefinito) indirizza a CUDA su NVIDIA e a MLX su Apple Silicon, quindi le configurazioni CUDA esistenti sono identiche a livello di byte.

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

Nella versione 1.5, il percorso MLX è **solo LoRA SFT** — nessun ORPO, nessun FP8, nessun `mode='full'`, nessun addestramento multiplo su MLX per ora (ognuno viene rifiutato con `CONFIG_INVALID_SETTING`; utilizza `backend='cuda'`/`'auto'` su una macchina NVIDIA per queste opzioni). L'adattatore risultante è in formato safetensors e può essere esportato su Ollama tramite lo stesso percorso del percorso CUDA.

> ⚠️ **Stato reale:** il percorso MLX viene fornito nella versione 1.5 **costruito + testato con unità (simulato)** ma **non ancora verificato su Apple Silicon reale** — `mlx-lm` è solo per Apple e non potrebbe essere eseguito sulla macchina NVIDIA su cui è stato creato questo progetto. Consideralo come sperimentale (lo stesso approccio del percorso FP8) e, per favore, [segnala eventuali anomalie](#reporting-bugs) una volta che sarà in esecuzione su un Mac della serie M. Forzare `--backend mlx` su un host non Apple genera un errore con `CONFIG_INVALID_SETTING`; la mancanza del toolchain `mlx_lm` su un Mac genera `DEP_MLX_UNAVAILABLE`.

Per flussi di lavoro end-to-end più completi (ottimizzazione e caricamento su HF-Hub, ripresa dopo esaurimento della memoria, SLAO multi-esecuzione in una lunga campagna, ecc.), consulta la [pagina delle ricette del manuale](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interfaccia utente web (opzionale)

Se preferisci fare clic invece di digitare Python, installa il componente aggiuntivo dell'interfaccia utente e avvia:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Si apre un'interfaccia web locale all'indirizzo `http://localhost:7862` per sfogliare i set di dati, convalidare i formati e assemblare visivamente una configurazione di addestramento. L'addestramento stesso viene eseguito tramite `backprop train` (l'addestramento basato sull'interfaccia utente è in programma: il pulsante Avvia mostra attualmente tale nota). L'interfaccia utente è locale per impostazione predefinita. Per esporla ad altri dispositivi, consulta [Web UI](#web-ui) di seguito per il contratto di sicurezza `--share` + `--auth`.

## Addestramento multi-esecuzione

Se desideri eseguire l'ottimizzazione in modo incrementale su più set di dati (ad esempio, se ricevi nuovi dati di addestramento ogni settimana e desideri aggiungerli senza dimenticare ciò che hai appreso in precedenza), la modalità `multi_run` di Backpropagate è ciò che fa per te:

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

Questo esegue cinque passaggi di addestramento, unendo l'adattatore tra le esecuzioni in modo da preservare le conoscenze precedenti incorporando al contempo nuovi esempi. La tecnica si basa su recenti ricerche sull'apprendimento continuo: consulta [References](#references) in fondo a questo file README.

La versione CLI:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Riprendi da un checkpoint

Un addestramento di 5 esecuzioni che si interrompe alla quarta esecuzione può essere ripreso. Ogni sessione multi-esecuzione scrive l'ID dell'esecuzione nella cronologia e nel manifesto del checkpoint su disco, quindi riprendere da dove ti sei interrotto richiede un solo comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

Il comportamento predefinito di `backprop multi-run` (nessun `--resume`) rileva automaticamente una voce in corso nella stessa directory di output e la continua. Per forzare un nuovo inizio, punta a una directory di output nuova.

## Cronologia dell'addestramento

Ogni invocazione di `backprop train` e `backprop multi-run` registra una riga in `<output>/run_history.json`: modello utilizzato, set di dati, iperparametri, stato, perdita finale, cronologia delle perdite. Puoi elencare e ispezionare le esecuzioni passate:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Monitoraggio degli esperimenti

Backpropagate rileva automaticamente i tracker di esperimenti installati (Weights & Biases, TensorBoard, MLflow) e li integra. Se `wandb` è installato e sei connesso, ogni esecuzione registra automaticamente su W&B con un nome di esecuzione che corrisponde all'ID di esecuzione su disco, in modo da poter cercare in W&B, nei tuoi log e in `run_history.json` utilizzando un unico identificatore.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Sovrascrivi con `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` o `Trainer(report_to="none")` per disattivare.

## Interfaccia utente web

L'interfaccia web Reflex è un'opzione: installala con `pipx install "backpropagate[ui]"` e avviala:

```bash
backprop ui --port 7862
```

L'interfaccia utente viene eseguita localmente su `http://localhost:7862`. Oggi copre la metà del flusso di lavoro relativa alla **navigazione / convalida / configurazione**: puntala a un set di dati, controlla il formato e le statistiche rilevate automaticamente, scegli un modello e assembla una configurazione di esecuzione. **L'avvio dell'esecuzione viene eseguito dalla CLI** (`backprop train` / `backprop multi-run`); il pulsante Avvia nell'interfaccia utente mostra una nota che indica questo. L'addestramento basato sull'interfaccia utente è un successivo sviluppo: fino ad allora, l'interfaccia utente è il punto di accesso e la CLI è il trigger.

Per esporlo ad altri dispositivi (altre persone sulla tua rete, un URL pubblico, ecc.), devi associare `--share` (o `--host`) a `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` senza `--auth` termina con un errore. Il motivo: `--share` pubblica un URL a cui chiunque su Internet può accedere e, senza autenticazione, ciò significa che chiunque può avviare la tua pipeline di addestramento e leggere il tuo token HuggingFace. Non è possibile disattivare questa funzione: se non vuoi impostare le credenziali, utilizza invece il port forwarding SSH:

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

Consulta [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) per il modello completo delle minacce.

Le operazioni di scrittura sul file system dall'interfaccia utente sono limitate a una singola directory:

- Predefinito: `~/.backpropagate/ui-outputs`
- Sovrascrittura: imposta `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- La sovrascrittura viene convalidata tramite una lista di esclusione: i percorsi di sistema o di credenziali (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, ecc.) vengono rifiutati.

## Note sulla piattaforma

**Requisiti:** Python 3.10+ · GPU CUDA (8 GB+ di VRAM) · PyTorch 2.0+

Python 3.10 raggiungerà la fine del ciclo di vita a ottobre 2026 ed è previsto che venga eliminato nella versione 1.5. Per le nuove installazioni, preferisci Python 3.11 o 3.12: la versione 3.11 è quella più testata.

Backpropagate gestisce le peculiarità di runtime dell'addestramento su diverse piattaforme, ma non può risolvere i problemi di installazione. I due più comuni sono:

- **Pacchetto CUDA errato.** PyTorch viene pubblicato con un singolo pacchetto per ogni versione di CUDA. Se scegli quello sbagliato, otterrai silenziosamente PyTorch solo per CPU e l'addestramento sarà incredibilmente lento. Utilizza lo strumento di selezione dei pacchetti disponibile all'indirizzo <https://pytorch.org/get-started/locally/> per il tuo driver. Esegui `nvidia-smi` per visualizzare la versione del driver/CUDA.
- **Windows + esportazione GGUF.** L'opzione `[export]` crea `llama-cpp-python` dal codice sorgente, che richiede Visual Studio Build Tools (componente C++) e CMake.

**macOS:** il supporto per CUDA non è disponibile (nessuna CUDA): un `trainer.train()` con CUDA genera `DEP_GPU_NOT_AVAILABLE` e puoi eseguire l'adattatore addestrato su un Mac tramite Ollama. **Novità nella versione 1.5:** un'implementazione sperimentale MLX (`--backend mlx`, `pip install 'backpropagate[mlx]'`) addestra un adattatore LoRA in modo nativo su Apple Silicon tramite `mlx_lm.lora`: solo LoRA SFT e testato, ma non ancora verificato su hardware reale (vedi [Apple Silicon (MLX)](#apple-silicon-mlx--experimental-v15)). Per il percorso CUDA o per ORPO / fine-tuning completo / FP8 / esecuzioni multiple, utilizza una macchina Linux o Windows con CUDA.

Consulta la [pagina della guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) per la guida completa alla risoluzione dei problemi di installazione e la [pagina dedicata alla risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) per i problemi relativi al driver / VRAM / xformers / bf16 rispetto a fp16.

## CLI

Ogni API Python ha un equivalente nella CLI:

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

Riferimento completo nella [pagina della guida alla CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), oppure utilizza `backprop <sottocomando> --help`.

## Configurazione

Ogni impostazione può essere sovrascritta con una variabile d'ambiente utilizzando il prefisso `BACKPROPAGATE_`:

| Variabile | Predefinito | Note |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Forza i log in formato JSON o nella console |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modello predefinito |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Tasso di apprendimento |
| `BACKPROPAGATE_LORA__R` | `256` | Rango LoRA (predefinito nella versione 1.3; utilizza `--lora-preset=fast` per il valore predefinito della versione 1.2.x, che è 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sandbox del file system dell'interfaccia utente |

Le chiavi nidificate utilizzano il doppio sottolineatura (`MODEL__NAME`, non `MODEL_NAME`). Il riferimento completo è disponibile nella [pagina della guida alle variabili d'ambiente](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Preset dei modelli

| Preset | VRAM | Licenza | Note |
|---|---|---|---|
| Qwen-3.5-4B | ~8 GB | Apache 2.0 | Predefinito consigliato per modelli inferiori a 5B. Migliore qualità per questa dimensione. |
| Phi-4-mini-3.8B | ~8 GB | MIT | Ottimo per ragionamento / matematica / codice. Licenza rigorosa. |
| SmolLM3-3B | ~6 GB | Apache 2.0 | Ricetta completamente aperta. Contesto nativo di 64K. |
| Qwen 2.5 7B | ~12 GB | Apache 2.0 | Predefinito esistente. Migliore qualità tra i preset 7B legacy. |
| Qwen 2.5 3B | ~8 GB | Qwen-Research | ⚠ licenza di ricerca: consulta i termini di licenza di Qwen prima dell'uso commerciale. |
| Llama 3.2 3B | ~8 GB | Llama Community | Solida alternativa a Qwen 3B con alcune limitazioni permissive. |
| Llama 3.2 1B | ~6 GB | Llama Community | Per esperimenti rapidi su schede di piccole dimensioni. |
| Mistral 7B | ~12 GB | Apache 2.0 | Confrontabile con Qwen 7B, template di chat diverso. |

Altri modelli spesso funzionano, ma solo questi otto sono fissati nei test CI. Utilizza `--lora-preset=quality` (predefinito) per i target con rango 256 / tutti i parametri lineari secondo Biderman 2024 + Thinking Machines 2025, oppure `--lora-preset=fast` per il target legacy con rango 16 / q+v se hai bisogno del footprint della versione 1.2.x.

## Risoluzione dei problemi

Un breve elenco dei fallimenti più comuni durante la prima esecuzione. L'indice completo è disponibile nella [pagina della guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Per un'analisi approfondita del driver / VRAM / precisione mista, consulta la [pagina della guida alla risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Sintomo | Codice di errore | Soluzione |
|---|---|---|
| La GPU esaurisce la memoria durante l'addestramento | `RUNTIME_GPU_OOM` | Automatico: la retropropagazione dimezza la dimensione del batch e riprova fino a 3 volte. Per disattivare: `Trainer(oom_recovery=False)`. Per forzare una dimensione inferiore: `--batch-size 1`. |
| HuggingFace restituisce 401 / "modello non trovato". | `DEP_MODEL_LOAD_FAILED` | Eseguire `huggingface-cli login` e riprovare. In caso di errori di battitura, copiare l'ID esatto da <https://huggingface.co/models>. |
| `register_with_ollama` connessione rifiutata. | `DEP_OLLAMA_REGISTRATION_FAILED` | Avviare il daemon: `ollama serve`. Installare da <https://ollama.com>. Riprovare. |
| Disco pieno durante il salvataggio del checkpoint. | `STATE_CHECKPOINT_INVALID` | Le scritture atomiche lasciano una directory `.partial` in caso di errore: è sicuro eliminarla. Il checkpoint precedente valido è intatto. |
| Addestramento in pausa a causa del surriscaldamento della GPU. | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automatico: la retropropagazione si interrompe quando viene superata la soglia di temperatura e riprende quando la GPU si raffredda. Migliorare il flusso d'aria se il problema persiste. |
| `backprop ui --share` rifiutato. | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Passare `--auth user:password` oppure utilizzare il port-forwarding SSH (vedere [Web UI](#web-ui)). |
| Esportazione GGUF fallita al primo tentativo. | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; su Windows è necessario anche Visual C++ Build Tools + CMake. |

## Segnalazione di bug

Quando qualcosa fallisce, Backpropagate stampa una riga all'avvio, ad esempio `run_started run_id=<uuid>`, e associa lo stesso ID a ogni riga del log, a ogni checkpoint e a ogni voce di Weights & Biases. **Includere il `run_id` in qualsiasi segnalazione di bug**, in modo che chi si occupa della manutenzione possa correlare tutto per quella specifica esecuzione.

Una buona segnalazione di bug include:

1. **Il `run_id`**: l'UUID stampato all'avvio. Un singolo UUID consente a chi si occupa della manutenzione di correlare ogni riga del log, ogni checkpoint e ogni voce di Weights & Biases per quella specifica esecuzione.
2. **Il codice di errore**: la riga `[CODE_NAME]: message` in stderr. Consultare [codici di errore](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) per il catalogo dei codici stabili.
3. **Il traceback modificato**. Stderr viene automaticamente modificato in modalità non verbose (i token Bearer, `sk-*`, `hf_*`, le chiavi AWS, le coppie `password=` / `token=` / `api_key=` vengono rimossi): è sicuro incollarlo. Per il traceback completo e non modificato, rieseguire con `BACKPROPAGATE_DEBUG=1` (o `--verbose`); rivedere prima di pubblicarlo.
4. **L'output di `backprop info`**. Un singolo comando stampa Python / PyTorch / CUDA / modello GPU / VRAM / sistema operativo / extra installati: tutto ciò di cui chi si occupa della manutenzione ha bisogno per analizzare una regressione specifica della piattaforma.

Il [modello di segnalazione di bug](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) richiede esplicitamente ciascuno di questi elementi, in modo che la fase di triage avvenga rapidamente. Domande, idee o discussioni del tipo "è questo previsto?" devono essere pubblicate in [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions). I problemi di sicurezza devono essere segnalati in privato tramite il modulo [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new): consultare [SECURITY.md](SECURITY.md) per le politiche e i tempi di risposta.

## Privacy

Tutto l'addestramento avviene localmente sulla GPU. Backpropagate non effettua richieste di rete, tranne per scaricare i modelli da HuggingFace (operazione che si avvia manualmente). Nessun telemetria, nessuna dipendenza dal cloud.

## Riferimenti

Le impostazioni predefinite di Backpropagate e la modalità di addestramento multi-esecuzione si basano su ricerche recenti. Se sei interessato alle tecniche sottostanti:

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — il documento fondamentale che introduce LoRA, che è il metodo utilizzato da Backpropagate per addestrare gli adattatori in modo efficiente.
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidenze empiriche che LoRA con rango 256 e obiettivi completamente lineari corrisponde alla qualità del fine-tuning completo nella maggior parte delle attività post-addestramento con il 67% della potenza di calcolo. Guida la configurazione LoRA predefinita di Backpropagate v1.3.
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — il seguito pratico che identifica la correzione 10× del tasso di apprendimento rispetto al fine-tuning completo necessaria con un rango LoRA elevato.
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la caratterizzazione originale del motivo per cui le reti neurali "dimenticano" l'addestramento precedente quando si esegue il fine-tuning su nuovi dati (EWC — Elastic Weight Consolidation).
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, un approccio precedente all'utilizzo di LoRA per l'apprendimento continuo, limitando i nuovi adattatori a sottospazi ortogonali.
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — una tecnica fondamentale per unire più modelli con fine-tuning senza interferenze.
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — l'algoritmo specifico che l'unificatore multi-esecuzione di Backpropagate implementa. Un preprint di dicembre 2025; Backpropagate è il primo implementatore noto di questo algoritmo.

## Licenza

MIT — vedere [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
