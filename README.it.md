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

# Ottimizza un modello QLoRA da 32 miliardi di parametri oppure un modello end-to-end da 7 miliardi di parametri su una singola GPU. Caricalo su Ollama

Esegui il fine-tuning di modelli linguistici di grandi dimensioni su una **singola** GPU, dimensionata in base alla scheda che hai effettivamente. Tre righe di codice Python per il fine-tuning di un modello da 7 a 34 miliardi di parametri su una singola scheda consumer da 32 GB (RTX 5090); un flag — `--full-ft-offload` — esegue il fine-tuning completo di un modello di classe 7B scaricando lo stato dell'ottimizzatore nella RAM del sistema. Un comando aggiuntivo esporta i risultati su Ollama, quindi esegui `ollama run` con il tuo modello ottimizzato. Si adatta in modo efficiente fino a 16 GB. Ottime prestazioni su Windows.

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

Questo è tutto. Non c'è un file di configurazione YAML. Non c'è una procedura complessa con `accelerate launch`. Non c'è un tutorial separato del tipo "ora convertilo in GGUF". Se hai una GPU CUDA e un file JSONL con i tuoi dati di addestramento, ti servono solo tre righe per ottenere un modello affinato funzionante.

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

## Dove si colloca Backpropagate nello spazio degli strumenti

Esistono diverse buone librerie per l'affinamento di LLM. Ognuna di esse è ottima in ambiti diversi:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — se ti piacciono le configurazioni YAML e desideri una community di ricette da cui copiare
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — se vuoi DPO/PPO/RLHF e un'interfaccia utente web
- **[Unsloth](https://github.com/unslothai/unsloth)** — se hai bisogno dell'addestramento più rapido possibile e utilizzi una famiglia di modelli supportata
- **[torchtune](https://github.com/pytorch/torchtune)** — se desideri le ricette PyTorch native di Meta che puoi modificare

Backpropagate è l'opzione mancante: **un'API Python a 3 righe per gli utenti singoli su una singola GPU consumer che desiderano addestrare un adattatore e caricarlo.** Nessun YAML, nessuna GUI, nessun RL online (PPO/GRPO), nessun nodo multiplo. Solo il ciclo di cui tutti hanno realmente bisogno e il passaggio di esportazione che crea problemi.

Se hai provato una delle librerie sopra elencate e ti sei scontrato con la procedura del file di configurazione, o hai riscontrato un problema con la famiglia di modelli, o desideravi impostazioni predefinite per Windows, Backpropagate è quello che fa per te.

## Cosa puoi ottimizzare su una singola GPU

Backpropagate dimensiona l'esecuzione in base alla tua scheda. Ecco i limiti pratici su una GPU consumer da **32 GB** (RTX 5090) con 64 GB di RAM del sistema: la configurazione su cui viene eseguito il fine-tuning è la seguente:

| Dimensione del modello | Metodo | Stato su una scheda da 32 GB |
|---|---|---|
| 7B (Qwen 2.5 7B / Llama-3.1-8B / Mistral 7B) | QLoRA | Ottimo — circa 7–8 GB. Lunghezza della sequenza completa, ampio margine di manovra. |
| **14B** (Qwen2.5-14B) | QLoRA | **Il punto ideale per l'uso quotidiano — circa 8,5 GB**, misurato. rank/alpha 32, paged 8-bit AdamW, 4096 ctx. |
| 24B (Mistral-Small-24B) | QLoRA | Circa 18 GB. Si adatta con un buon margine a 4096 ctx. |
| **32B** (Qwen2.5-32B) | QLoRA | **Si adatta appena — circa 26 GB** con `max_len 2048` + paged 8-bit AdamW. Limite massimo. |
| ≤6B | `mode="full"` (affinamento completo) | Fine-tuning completo su GPU pura: pesi bf16, nessun adattatore. Il limite massimo per la scheda è di 6B su 32 GB. |
| **Classe 7B** (Qwen 2.5 7B / Llama-3.1-8B / Mistral 7B) | `mode="full" --full-ft-offload` | **Fine-tuning completo tramite CPU-offload FSDP2:** scarica i parametri e l'ottimizzatore nella RAM del sistema da 64 GB. Più lento (limitato dalla larghezza di banda); Linux/WSL2. |

Due cose per cui la maggior parte delle librerie per singola GPU ti indirizzano altrove: **QLoRA da 24–34B** e **fine-tuning completo su scheda singola di classe 7B**. Backpropagate esegue queste operazioni su una singola scheda consumer, quindi esporta direttamente il risultato su Ollama.

**Il limite massimo per il fine-tuning completo è adattato alla scheda.** Deriva dall'aritmetica della memoria di addestramento a 4 termini (pesi + gradienti + ottimizzatore + attivazioni) rispetto alla VRAM *rilevata*: **16 GB → 4B, 24 GB → 5B, 32 GB → 6B** su GPU pura. `--full-ft-offload` lo estende a **classe 7B** scaricando i parametri e lo stato dell'ottimizzatore nella RAM del sistema tramite FSDP2 `fully_shard` + `CPUOffloadPolicy` (più lento, limitato dalla larghezza di banda PCIe/CPU; richiede circa 64 GB di RAM del sistema e un backend NCCL, ovvero Linux/WSL2). Sovrascrivi esplicitamente il limite con `--full-ft-ceiling-billions`. Un modello che supera anche il limite di offload termina con `RUNTIME_FULL_FT_MODEL_TOO_LARGE`, indicando la soluzione (`--full-ft-offload` o LoRA/QLoRA). Consulta [la pagina completa del manuale sul fine-tuning](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/) per i calcoli della VRAM e il confronto sulla qualità di Biderman 2024 / Thinking Machines 2025.

### Si adatta fino a 16 GB

Il limite di 16 GB (RTX 4080 / 5080 / 4070 Ti Super) offre comunque ottime prestazioni: QLoRA da 7B con circa 7–8 GB e vero fine-tuning completo di un modello reale da ~3B (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B/1B) all'interno di 16 GB tramite `mode="full"` (pesi bf16 + checkpointing del gradiente + paged 8-bit AdamW). Lo stesso codice seleziona la dimensione del batch e il limite massimo per il fine-tuning completo in base alla scheda rilevata, senza flag da modificare tra le diverse configurazioni.

La quantizzazione a 2 bit (AQLM / QuIP#) è **fuori dall'ambito**: una base a 2 bit non può essere unita correttamente ai pesi in piena precisione, il che interrompe il contratto di esportazione dell'adattatore unificabile → GGUF → Ollama (che è lo scopo principale della pipeline). Invece, Backpropagate offre i seguenti strumenti: QLoRA, `mode="full"`, `--full-ft-offload` e il percorso di calcolo FP8 (`--fp8`, Blackwell/Hopper), tutti che rimangono unificabili ed esportabili.

## Per cosa NON è adatto Backpropagate

Se il tuo caso d'uso rientra nelle seguenti categorie, otterrai risultati migliori con una libreria diversa: Backpropagate non è la scelta giusta e cercare di farlo funzionare costerebbe più che semplicemente utilizzare lo strumento corretto. Leggere questa sezione prima di iniziare ti eviterà di installare e poi abbandonare il progetto:

- **Ottimizzazione fine con tutti i parametri oltre il limite di offload (≈13B+)** — Esegue la retropropagazione dell'ottimizzazione fine completa fino a **~6 GB di GPU pura e ~7 GB tramite `--full-ft-offload`** su una scheda da 32 GB (vedere [la sezione](#what-you-can-fine-tune-on-one-gpu)). Un'ottimizzazione fine *veramente completa* di un modello da 13B+ supera tale limite: richiede FSDP multi-GPU o una scheda più grande (utilizzare `transformers.Trainer` su più GPU oppure noleggiare una A100/H100). Prima di utilizzare tutta questa potenza di calcolo, tuttavia: ricerche recenti ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) dimostrano che LoRA, con la configurazione corretta, offre una qualità di ottimizzazione fine paragonabile a quella completa per la maggior parte delle attività post-addestramento (seguimento delle istruzioni, adattamento al dominio, personalità/stile) con circa il 67% della potenza di calcolo necessaria. Quindi, QLoRA fino a 34B, che Backpropagate esegue su una singola scheda, non comporta alcuna perdita per il lavoro che la maggior parte degli utenti desidera svolgere.
- **Apprendimento per rinforzo online — PPO / GRPO / RLVR** — Backpropagate esegue l'ottimizzazione fine monostadio (SFT) più l'ottimizzazione delle preferenze senza riferimenti (ORPO nella versione 1.5; SimPO + KTO nella versione 1.6). Non esegue l'apprendimento per rinforzo online — PPO, GRPO o RLVR —, che richiede un modello di ricompensa o un ciclo di generazione e valutazione in aggiunta alla fase di addestramento. Per queste attività, utilizzare direttamente TRL o LLaMA-Factory. (L'ottimizzazione delle preferenze senza riferimenti si adatta all'ambito monostadio perché non è necessario memorizzare un modello di riferimento separato; vedere la nota su ORPO nella sezione [Quick Start](#quick-start).)
- **Addestramento multi-nodo** — una singola GPU su una sola macchina. L'utilizzo di più GPU su una singola macchina funziona (tramite `accelerate launch`), ma non è ufficialmente supportato.
- **Addestramento macOS con CUDA** — Apple Silicon non dispone di CUDA, quindi il percorso CUDA viene eseguito su un sistema Linux o Windows con una GPU NVIDIA. È comunque possibile eseguire il modello addestrato su un Mac tramite Ollama. Un percorso MLX **sperimentale e non verificato** (`--backend mlx`) addestra in modo nativo un adattatore LoRA su Apple Silicon — vedere [Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview). È solo per LoRA-SFT e **non è stato testato su hardware reale** (nessun supporto), quindi, per qualsiasi cosa oltre a un SFT LoRA (ORPO, ottimizzazione fine completa, FP8, esecuzioni multiple), è consigliabile utilizzare il percorso CUDA.
- **Qualsiasi modello al di fuori delle famiglie di modelli testate** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Altri modelli spesso funzionano, ma non sono inclusi nei test CI.

Se si necessita di una qualsiasi di queste funzionalità, utilizzare una delle librerie elencate sopra. Sono più adatte a questo scopo.

## Cosa offre Backpropagate

Quattro elementi, in un'unica installazione:

**1. Una vera API di 3 righe che funziona senza un file di configurazione.**
Lo snippet all'inizio di questo README viene eseguito dall'inizio alla fine. Non è necessario `accelerate config`, YAML o override Hydra. Basta `Trainer(model).train(data)` e si ottiene un modello ottimizzato.

**2. Windows che funziona davvero.**
La maggior parte delle librerie ML trattano Windows come un'aggiunta successiva. Backpropagate viene testata in modo completo su Windows + RTX 5080. La libreria gestisce le peculiarità dell'ambiente di runtime: sa come pre-tokenizzare i dati in modo che l'elaborazione parallela di Windows non si blocchi, disabilita automaticamente xformers sulle schede RTX 40/50 dove causerebbe problemi e seleziona le impostazioni del caricatore di dati che non causano errori. Non è necessario sapere nulla di tutto questo. Funziona semplicemente.

**3. Progettato per esecuzioni senza supervisione.**
L'addestramento richiede ore. Non si vuole doverlo monitorare costantemente. Backpropagate è progettata per essere lasciata in esecuzione:

- Se la memoria della GPU si esaurisce, dimezza automaticamente le dimensioni del batch e riprova, fino a tre volte. Non è necessario effettuare alcuna regolazione manuale.
- Se la GPU diventa troppo calda, si mette in pausa finché non si raffredda e poi continua.
- Ogni checkpoint viene scritto in modo atomico: se il laptop si blocca durante il salvataggio, il checkpoint precedente valido rimane intatto.
- Ogni esecuzione di addestramento riceve un ID univoco che viene stampato su ogni riga del log, su ogni checkpoint e su ogni voce di Weights & Biases. Se qualcosa va storto, un singolo ID consente a un manutentore di correlare tutto.
- Gli errori sono accompagnati da codici stabili (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, ecc.) in modo che sia possibile cercare nei log e nella [guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) per trovare la soluzione. I guasti specifici di CUDA hanno una pagina dedicata per la [risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

**4. Un solo comando, dall'adattatore addestrato a `ollama run`.**
Molte librerie addestrano un modello. Poche di esse si mettono di mezzo quando si vuole effettivamente utilizzarlo. Backpropagate esporta in GGUF (il formato utilizzato da Ollama) e registra un modello Ollama con un solo comando. Si passa da "addestramento completato" a "posso chattare con il mio modello ottimizzato" in circa 30 secondi.

## Guida rapida

Il repository include un piccolo set di dati di esempio, quindi lo snippet all'inizio di questo README funziona su una nuova installazione:

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

Questo addestra un adattatore Qwen 2.5 da 7B su 5 brevi conversazioni in formato ShareGPT, quindi esporta il risultato in GGUF. Per i propri dati, formattare il file JSONL con un esempio per riga:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

I formati Alpaca (`instruction` / `output`), OpenAI chat (`messages`) e testo semplice funzionano anche; Backpropagate rileva automaticamente il formato.

### Ottimizzazione delle preferenze (ORPO, SimPO, KTO)

Addestrare sulle preferenze anziché su semplici dimostrazioni. ORPO non richiede riferimenti ed è una singola fase: integra il segnale di preferenza nella fase SFT, quindi non esiste un modello di ricompensa o di riferimento separato e la struttura a 3 righe rimane invariata. Passare `--method orpo` (CLI) o `method="orpo"` (Python) e fornire un set di dati con righe nel formato `{prompt, chosen, rejected}` (o solo `{chosen, rejected}`):

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

Il tasso di apprendimento predefinito si riduce automaticamente a `8e-6` per ORPO (la perdita è più marcata rispetto al semplice SFT); regola `--orpo-beta` (predefinito `0.1`) per ponderare la penalità del rapporto delle probabilità. ORPO è solo in modalità `"lora"`.

**Novità nella versione 1.6: SimPO e KTO.** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) è indipendente dai dati di riferimento, utilizza una ricompensa normalizzata in base alla lunghezza e accetta gli stessi dati a coppie `{prompt, scelto, rifiutato}` di ORPO (`--simpo-beta`, `--simpo-gamma`). `--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) accetta dati **non a coppie** `{prompt, completamento, etichetta}` — valutazioni positive/negative per ogni esempio — per l'ampia classe di feedback che non sono coppie A/B curate; bilancia automaticamente i pesi della perdita desiderabile/indesiderabile in base al conteggio delle etichette. Entrambi sono solo in modalità `"lora"` e rimangono nell'ambito SFT su una singola GPU (nessun modello di riferimento separato). Consulta il [manuale sull'ottimizzazione delle preferenze](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) per capire quale utilizzare. Per l'RL online (PPO/GRPO), consulta [Cosa NON è Backpropagate](#what-backpropagate-is-not-for).

### SFT con traccia del ragionamento (distillazione R1)

Novità nella versione 1.5: distilla un modello di ragionamento in modo semplice. Passa `--reasoning-trace` (CLI) o `Trainer(..., reasoning_trace=True)` (Python) e fornisci tracce che mantengono una catena di pensiero `<think>...</think>` all'interno del turno dell'assistente — la metà SFT pura della distillazione di [DeepSeek-R1](https://arxiv.org/abs/2501.12948), senza necessità di RL. Backpropagate mantiene `<think>` nell'obiettivo di addestramento, elimina le tracce vuote o troppo lunghe (filtraggio della lunghezza delle tracce) e aumenta il valore predefinito di `max_seq_length` a 8192 per la catena di pensiero più lunga. Fondamentalmente, `<think>` rimane in **testo semplice** — nessun token speciale, nessuna ridimensionamento dell'embedding — quindi l'adattatore unificato esporta ancora in GGUF e può essere utilizzato con Ollama come qualsiasi altro modello ottimizzato. Solo SFT. Consulta la [ricetta per la traccia del ragionamento](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) per la forma del set di dati e i token regolabili.

### Apple Silicon (MLX) — anteprima non verificata

> ⚠️ **Anteprima non verificata: non fa parte delle funzionalità supportate.** Il percorso MLX è stato creato ed è stato sottoposto a test unitari, ma **non** è stato testato su hardware Apple Silicon reale (`mlx-lm` è disponibile solo per Apple e non può essere eseguito sui sistemi NVIDIA su cui viene sviluppato Backpropagate). Considerare tutto quanto segue come sperimentale, utilizzarlo a proprio rischio e [segnalare eventuali anomalie](#reporting-bugs) se lo si esegue su un Mac della serie M.

Novità nella versione 1.5: **un'API, due opzioni.** CUDA rimane il backend canonico e verificato; MLX è una seconda opzione che esegue l'addestramento su un Mac della serie M tramite lo strumento [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) di Apple (memoria unificata, nessuna necessità di CUDA). La stessa struttura a 3 righe seleziona l'opzione in base all'hardware: `backend='auto'` (predefinito) indirizza verso CUDA su NVIDIA e verso MLX su Apple Silicon, quindi le configurazioni CUDA esistenti sono identiche.

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

Nella versione 1.5, l'opzione MLX è **solo SFT LoRA** — nessun ORPO, nessun FP8, nessuna modalità `'full'`, nessun addestramento multiplo su MLX (ognuno viene rifiutato con `CONFIG_INVALID_SETTING`; utilizza `backend='cuda'`/`'auto'` su una macchina NVIDIA per queste opzioni). L'adattatore risultante è in formato safetensors e può essere esportato verso Ollama tramite lo stesso percorso dell'opzione CUDA.

> Forzare `--backend mlx` su un host non Apple genera l'errore `CONFIG_INVALID_SETTING`; la mancanza di una toolchain `mlx_lm` su un Mac genera `DEP_MLX_UNAVAILABLE`.

Per flussi di lavoro end-to-end più completi (ottimizzazione e caricamento su HF Hub, ripresa dopo esaurimento della memoria, SLAO multi-esecuzione in una lunga campagna, ecc.), consulta la [pagina delle ricette del manuale](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interfaccia utente web (opzionale)

Se preferisci fare clic invece di digitare in Python, installa il pacchetto aggiuntivo dell'interfaccia utente e avvia:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Si apre un'interfaccia web locale all'indirizzo `http://localhost:7862` per sfogliare i set di dati, convalidare i formati e assemblare visivamente una configurazione di addestramento. L'addestramento stesso viene eseguito tramite `backprop train` (l'addestramento basato sull'interfaccia utente è in programma — il pulsante Avvia mostra attualmente tale nota). L'interfaccia utente è locale per impostazione predefinita. Per renderla accessibile da altri dispositivi, consulta la sezione [Interfaccia utente web](#web-ui) qui sotto per il contratto di sicurezza `--share` + `--auth`.

## Addestramento multi-esecuzione

Se desideri ottimizzare in modo incrementale su più set di dati — ad esempio, se ricevi nuovi dati di addestramento ogni settimana e vuoi aggiungerli senza dimenticare ciò che hai imparato prima — la modalità `multi_run` di Backpropagate è quella giusta per te:

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

Questo esegue cinque passaggi di addestramento, unendo l'adattatore tra le esecuzioni in modo da preservare le conoscenze precedenti e incorporare nuovi esempi. La tecnica si basa su recenti ricerche sull'apprendimento continuo — consulta la sezione [Riferimenti](#references) in fondo a questo file README.

La versione CLI:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Ripresa da un checkpoint

Un addestramento di 5 esecuzioni che si interrompe alla quarta esecuzione può essere ripreso. Ogni sessione multi-esecuzione scrive l'ID dell'esecuzione nella cronologia e nel manifesto dei checkpoint su disco, quindi riprendere da dove ti sei interrotto richiede un solo comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

Il comportamento predefinito di `backprop multi-run` (nessun `--resume`) rileva automaticamente una voce in corso nella stessa directory di output e la continua. Per forzare un nuovo inizio, punta a una nuova directory di output.

## Cronologia dell'addestramento

Ogni invocazione di `backprop train` e `backprop multi-run` registra una riga in `<output>/run_history.json` — modello utilizzato, set di dati, iperparametri, stato, perdita finale, cronologia delle perdite. Puoi elencare e ispezionare le esecuzioni passate:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Monitoraggio degli esperimenti

Backpropagate rileva automaticamente i tracker di esperimenti installati (Weights & Biases, TensorBoard, MLflow) e li configura. Se `wandb` è installato e sei autenticato, ogni esecuzione registra automaticamente i dati su W&B con un nome che corrisponde all'ID dell'esecuzione salvato sul disco, in modo da poter effettuare ricerche su W&B, nei tuoi log e nel file `run_history.json` utilizzando un unico identificatore.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

È possibile sovrascrivere le impostazioni con `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` o `Trainer(report_to="none")` per disattivare la funzionalità.

## Interfaccia utente web

L'interfaccia web di Reflex è attivabile; installala con `pipx install "backpropagate[ui]"` e avviala:

```bash
backprop ui --port 7862
```

L'interfaccia utente viene eseguita localmente all'indirizzo `http://localhost:7862`. Attualmente, copre la metà del flusso di lavoro relativa a **esplorazione / convalida / configurazione**: punta l'interfaccia su un set di dati, verifica il formato e le statistiche rilevate automaticamente, seleziona un modello e crea una configurazione per l'esecuzione. **L'avvio dell'esecuzione viene eseguito dalla riga di comando** (`backprop train` / `backprop multi-run`); il pulsante "Avvia" nell'interfaccia utente visualizza una nota che indica dove avviare l'esecuzione. L'addestramento guidato dall'interfaccia utente è un aggiornamento futuro; per ora, l'interfaccia utente funge da punto di accesso e la riga di comando è il trigger.

Per renderla accessibile ad altri dispositivi (altre persone sulla tua rete, un URL pubblico, ecc.), devi combinare `--share` (o `--host`) con `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` senza `--auth` genera un errore. Il motivo: `--share` pubblica un URL a cui chiunque su Internet può accedere e, in assenza di autenticazione, ciò significa che chiunque può controllare la tua pipeline di addestramento e leggere il tuo token HuggingFace. Non è possibile disattivare questa funzionalità; se non desideri impostare le credenziali, utilizza invece il port forwarding SSH:

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

Consulta [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) per la descrizione completa del modello di minaccia.

Le operazioni di scrittura sul file system dall'interfaccia utente sono limitate a una singola directory:

- Predefinito: `~/.backpropagate/ui-outputs`
- Sovrascrittura: imposta `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- La sovrascrittura viene convalidata tramite una lista di esclusione: i percorsi di sistema o delle credenziali (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, ecc.) vengono rifiutati.

## Note sulla piattaforma

**Requisiti:** Python 3.10+ · GPU CUDA (8 GB o più di VRAM) · PyTorch 2.0+

Python 3.10 è supportato almeno fino alla versione v1.6; il supporto terminerà a ottobre 2026 e la rimozione è prevista nella prima versione successiva. Per le nuove installazioni, preferisci Python 3.11 o 3.12: la versione 3.11 è quella più testata.

Backpropagate gestisce le peculiarità di runtime dell'addestramento su diverse piattaforme, ma non può risolvere i problemi che si verificano durante l'installazione. I due più comuni sono:

- **Pacchetto CUDA errato.** PyTorch viene pubblicato con un singolo pacchetto per ogni versione di CUDA. Se selezioni quello sbagliato, otterrai silenziosamente solo PyTorch per CPU e l'addestramento sarà estremamente lento. Utilizza lo strumento di selezione dei pacchetti disponibile all'indirizzo <https://pytorch.org/get-started/locally/> per la tua scheda grafica. Esegui `nvidia-smi` per visualizzare la versione del driver / CUDA.
- **Windows + esportazione GGUF.** L'opzione extra `[export]` crea `llama-cpp-python` dal codice sorgente, il che richiede Visual Studio Build Tools (componente C++) e CMake.

**macOS:** la configurazione CUDA non è supportata (nessuna CUDA); l'esecuzione di `trainer.train()` con una configurazione CUDA genera un errore `DEP_GPU_NOT_AVAILABLE` ed è possibile eseguire l'adattatore addestrato su un Mac tramite Ollama. **Novità nella versione v1.5:** una configurazione MLX sperimentale (`--backend mlx`, `pip install 'backpropagate[mlx]'`) esegue nativamente l'addestramento di un adattatore LoRA su Apple Silicon tramite `mlx_lm.lora`: solo SFT LoRA, e compilato e testato, ma non ancora verificato in condizioni reali (vedi [Apple Silicon (MLX)](#apple-silicon-mlx--experimental-v15)). Per il percorso CUDA o per ORPO / fine-tuning completo / FP8 / esecuzioni multiple, utilizza una macchina Linux o Windows con CUDA.

Consulta la [pagina della guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) per la guida completa alla risoluzione dei problemi di installazione e la [pagina dedicata alla risoluzione dei problemi CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) per i problemi relativi al driver / VRAM / xformers / bf16 rispetto a fp16.

## CLI (riga di comando)

Ogni API Python ha un corrispondente nella riga di comando:

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

Riferimento completo disponibile nella [pagina della guida alla riga di comando](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), oppure utilizza `backprop <sottocomando> --help`.

## Configurazione

Ogni impostazione può essere sovrascritta con una variabile d'ambiente utilizzando il prefisso `BACKPROPAGATE_`:

| Variabile | Predefinito | Note |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Forza l'utilizzo di log in formato JSON o nella console. |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modello predefinito |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Tasso di apprendimento |
| `BACKPROPAGATE_LORA__R` | `256` | Rango LoRA (predefinito v1.3; utilizza `--lora-preset=fast` per il valore predefinito della v1.2.x, ovvero 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sandbox del file system dell'interfaccia utente |

Le chiavi nidificate utilizzano il doppio underscore (`MODEL__NAME`, non `MODEL_NAME`). Il riferimento completo è disponibile nella [pagina della guida alle variabili d'ambiente](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Preset dei modelli

| Preset | VRAM | Licenza | Note |
|---|---|---|---|
| Qwen-3.5-4B | ~8 GB | Apache 2.0 | Predefinito consigliato per modelli inferiori a 5B. Migliore qualità per questa dimensione. |
| Phi-4-mini-3.8B | ~8 GB | MIT | Ottimo nelle attività di ragionamento, matematica e codice. Licenza completamente aperta. |
| SmolLM3-3B | ~6 GB | Apache 2.0 | Ricetta completamente open source. Contesto nativo di 64K. |
| Qwen 2.5 7B | ~12 GB | Apache 2.0 | Predefinito esistente. Migliore qualità tra i preset 7B legacy. |
| Qwen 2.5 3B | ~8 GB | Qwen-Research | ⚠ licenza di ricerca: consulta i termini della licenza Qwen prima dell'uso commerciale. |
| Llama 3.2 3B | ~8 GB | Comunità Llama | Un'alternativa valida a Qwen 3B con alcune limitazioni. |
| Llama 3.2 1B | ~6 GB | Comunità Llama | Per esperimenti rapidi su schede di piccole dimensioni. |
| Mistral 7B | ~12 GB | Apache 2.0 | Simile a Qwen 7B, con un modello di chat diverso. |
| Llama-3.1-8B | ~7-8 GB (QLoRA) | Llama-3.1-Community | 8B QLoRA, contesto nativo di 128K (la clausola >700M-MAU richiede una licenza Meta separata). |
| **Qwen2.5-14B** | ~8,5 GB (QLoRA) | Apache 2.0 | **Il punto ideale per l'utilizzo quotidiano con 32 GB** — rank/alpha 32, paged 8-bit AdamW, 4096 ctx. |
| Mistral-Small-24B | ~18 GB (QLoRA) | Apache 2.0 | 24B QLoRA su una scheda da 32 GB con margine di 4096 ctx. |
| **Qwen2.5-32B** | ~26 GB (QLoRA) | Apache 2.0 | **Limite massimo per 32 GB** — si adatta a malapena con `max_len 2048` + paged 8-bit AdamW. |

Altri modelli spesso funzionano; le righe sopra riportate sono le configurazioni predefinite curate: la fascia da 14B a 32B è ottimizzata con QLoRA per una scheda da 32 GB (l'ambito misurato). Utilizzare `--lora-preset=quality` (impostazione predefinita) per i target rank-256 / all-linear secondo Biderman 2024 + Thinking Machines 2025, oppure `--lora-preset=fast` per il target legacy rank-16 / q+v se è necessario l'ingombro della versione 1.2.x.

## Risoluzione dei problemi

Un breve elenco degli errori più comuni che si verificano durante la prima esecuzione. L'indice completo in ordine inverso è disponibile nella [pagina del manuale per la risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Per un'analisi approfondita di driver / VRAM / precisione mista, consultare la [pagina della risoluzione dei problemi CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Sintomo | Codice di errore | Soluzione |
|---|---|---|
| La GPU esaurisce la memoria durante l'addestramento | `RUNTIME_GPU_OOM` | Automatico: Backpropagate dimezza le dimensioni del batch e riprova fino a 3 volte. Per disattivare questa funzione: `Trainer(oom_recovery=False)`. Per forzare una dimensione inferiore: `--batch-size 1`. |
| HuggingFace restituisce 401 / "modello non trovato" | `DEP_MODEL_LOAD_FAILED` | Eseguire `huggingface-cli login` e riprovare. In caso di errori di battitura, copiare l'ID esatto da <https://huggingface.co/models>. |
| La connessione a `register_with_ollama` viene rifiutata | `DEP_OLLAMA_REGISTRATION_FAILED` | Avviare il daemon: `ollama serve`. Installare da <https://ollama.com>. Riprovare. |
| Lo spazio su disco è insufficiente durante il salvataggio del checkpoint | `STATE_CHECKPOINT_INVALID` | Le scritture atomiche lasciano una directory `.partial` in caso di errore: è sicuro eliminarla. Il checkpoint precedente valido è intatto. |
| L'addestramento viene interrotto a causa del surriscaldamento della GPU | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automatico: Backpropagate si interrompe quando viene superata la soglia di temperatura e riprende quando la GPU si raffredda. Migliorare il flusso d'aria se il problema persiste. |
| `backprop ui --share` rifiutato | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Passare `--auth user:password`, oppure utilizzare invece il port-forwarding SSH (vedere [Interfaccia utente Web](#interfaccia-utente-web)). |
| Esportazione GGUF non riuscita al primo tentativo | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; su Windows è necessario anche Visual C++ Build Tools + CMake. |

## Segnalazione di bug

Quando qualcosa fallisce, Backpropagate stampa una riga all'avvio simile a `run_started run_id=<uuid>` e associa lo stesso ID a ogni riga del log, a ogni checkpoint e a ogni voce di Weights & Biases. **Includere il `run_id` in qualsiasi segnalazione di bug**, in modo che chi si occupa della manutenzione possa correlare tutto per quella specifica esecuzione.

Una buona segnalazione di bug include:

1. **Il `run_id`**: l'UUID stampato all'avvio. Un singolo UUID consente a chi si occupa della manutenzione di correlare ogni riga del log, ogni checkpoint e ogni voce di Weights & Biases per quella specifica esecuzione.
2. **Il codice di errore**: la riga `[CODE_NAME]: message` in stderr. Consultare [codici di errore](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) per il catalogo dei codici stabili.
3. **Il traceback modificato**. Stderr viene automaticamente modificato in modalità non verbose (i token Bearer, `sk-*`, `hf_*`, le chiavi AWS, le coppie `password=` / `token=` / `api_key=` vengono eliminate): è sicuro copiarlo e incollarlo. Per il traceback completo non modificato, rieseguire con `BACKPROPAGATE_DEBUG=1` (o `--verbose`); rivederlo prima di pubblicarlo.
4. **L'output di `backprop info`**. Un singolo comando stampa Python / PyTorch / CUDA / modello GPU / VRAM / sistema operativo / extra installati: tutto ciò di cui chi si occupa della manutenzione ha bisogno per individuare una regressione specifica della piattaforma.

Il [modello di segnalazione di bug](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) richiede esplicitamente ciascuno di questi elementi in modo che la valutazione possa procedere rapidamente. Domande, idee o discussioni su "è questo previsto?" devono essere pubblicate in [Discussioni di GitHub](https://github.com/mcp-tool-shop-org/backpropagate/discussions). I problemi di sicurezza devono essere segnalati privatamente tramite il modulo [Avviso di sicurezza di GitHub](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new): consultare [SECURITY.md](SECURITY.md) per le politiche e i tempi di risposta.

## Privacy

Tutto l'addestramento avviene localmente sulla GPU. Backpropagate non effettua richieste di rete, tranne per scaricare modelli da HuggingFace (che si avvia manualmente). Nessun telemetria, nessuna dipendenza dal cloud.

## Riferimenti

Le impostazioni predefinite e la modalità di addestramento multi-esecuzione di Backpropagate sono basate su ricerche recenti. Se sei interessato alle tecniche sottostanti:

- **Hu et al. 2021.** *LoRA: Adattamento a basso rango di modelli linguistici di grandi dimensioni.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — l'articolo fondamentale che introduce LoRA, il metodo con cui Backpropagate addestra gli adattatori in modo efficiente.
- **Biderman et al. 2024.** *LoRA impara di meno e dimentica di meno.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidenze empiriche che LoRA con rango 256 e obiettivi completamente lineari raggiunge una qualità paragonabile al fine-tuning completo nella maggior parte delle attività post-addestramento, utilizzando il 67% della potenza di calcolo. Definisce la configurazione LoRA predefinita di Backpropagate v1.3.
- **Thinking Machines 2025.** *LoRA senza rimpianti.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — il seguito pratico che identifica la correzione del tasso di apprendimento (10 volte) rispetto al fine-tuning completo necessaria con un rango LoRA elevato.
- **Kirkpatrick et al. 2017.** *Superare l'oblio catastrofico nelle reti neurali.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la prima descrizione del motivo per cui le reti neurali "dimenticano" gli addestramenti precedenti quando si esegue il fine-tuning su nuovi dati (EWC — Elastic Weight Consolidation).
- **Wang et al. 2023.** *Apprendimento di sottospazi ortogonali per l'apprendimento continuo di modelli linguistici.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, un approccio precedente all'utilizzo di LoRA per l'apprendimento continuo, limitando i nuovi adattatori a sottospazi ortogonali.
- **Yadav et al. 2023.** *TIES-Merging: Risolvere le interferenze durante la fusione dei modelli.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — una tecnica fondamentale per fondere più modelli con fine-tuning senza interferenze.
- **Qiao & Mahdavi 2025.** *Unisci prima di dimenticare: un singolo apprendimento continuo LoRA tramite fusione continua.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — l'algoritmo specifico che il sistema di fusione multi-esecuzione di Backpropagate implementa. Un preprint di dicembre 2025; Backpropagate è il primo utilizzatore noto di questo articolo.

## Licenza

MIT — vedere [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
