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

# Addestra un adattatore. Inoltralo a Ollama. Finito

Backpropagate è una libreria Python per l'affinamento di modelli linguistici di grandi dimensioni su una singola GPU. Tre righe di codice addestrano un modello da 7 miliardi di parametri su una scheda da 16 GB. Un altro comando lo esporta in Ollama, così puoi eseguire il tuo modello affinato con `ollama run`. Funziona perfettamente anche su Windows.

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

Ecco tutto. Non c'è bisogno di un file di configurazione YAML. Non c'è la "cerimonia" di `accelerate launch`. Non c'è un tutorial separato su come "convertirlo in GGUF". Se hai una GPU CUDA e un file JSONL con i tuoi dati di addestramento, sei a solo tre righe di codice dall'avere un modello affinato funzionante.

## Installazione

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

Preferisci Docker? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` funziona anche. Sono disponibili immagini per `linux/amd64` e `linux/arm64`, quindi gli utenti con Apple Silicon e ARM Linux hanno a disposizione un'immagine nativa. Un file `compose.yaml` standard per "UI in un container" si trova nella directory principale del repository: `docker compose up` avvia l'interfaccia web su `http://localhost:7860` con un volume persistente `~/.backpropagate` montato.

## Dove si colloca Backpropagate

Esistono diverse buone librerie per l'affinamento di modelli linguistici di grandi dimensioni. Ognuna è eccellente per cose diverse:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — se ti piacciono le configurazioni YAML e vuoi una comunità di ricette da cui prendere spunto.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — se desideri un'interfaccia grafica web e un supporto integrato per DPO/PPO/RLHF.
- **[Unsloth](https://github.com/unslothai/unsloth)** — se hai bisogno della formazione più veloce possibile e utilizzi una famiglia di modelli supportata.
- **[torchtune](https://github.com/pytorch/torchtune)** — se desideri le ricette native di PyTorch di Meta che puoi modificare.

Backpropagate è l'opzione mancante: **un'API Python di 3 righe per gli utenti singoli che utilizzano una singola GPU consumer e che desiderano addestrare un adattatore e distribuirlo.** Nessun YAML, nessuna interfaccia grafica, nessun DPO/PPO, nessuna configurazione multi-nodo. Solo il ciclo di lavoro di cui tutti hanno bisogno e la fase di esportazione che crea problemi.

Se hai provato una delle librerie sopra e ti sei sentito frustrato dalla "cerimonia" del file di configurazione, o hai riscontrato un problema con la famiglia di modelli, o volevi impostazioni predefinite per Windows — Backpropagate è quello che fa per te.

## Cosa puoi affinare su una GPU consumer da 16 GB

Ecco i limiti pratici su una scheda da 16 GB (RTX 4080 / 5080 / 4070 Ti Super):

| Modello | Metodo | Stato |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | Comodo. Lunghezza della sequenza completa, con spazio extra. |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | Standard. Circa 7-8 GB. Impostazioni predefinite di Backpropagate. |
| Llama-3 13B | QLoRA + sample packing | Limitato ma funzionante. Utilizza sequenze più corte. |
| Mixtral 8x7B (47 miliardi di parametri totali) | AQLM 2-bit + LoRA | Sperimentale nella versione 1.4. Il modello più grande che puoi utilizzare su una scheda da 16 GB. |

Per i modelli da 3 miliardi di parametri e inferiori, l'affinamento completo (non solo LoRA) è possibile su 16 GB ed è previsto come opzione `mode="full"` nella versione 1.4. Per i modelli da 7 miliardi di parametri o superiori, l'affinamento completo richiede una GPU da 24 GB o superiore: considera un'istanza cloud A100 oppure utilizza LoRA, che, secondo recenti ricerche, offre una qualità equivalente all'affinamento completo nella maggior parte delle attività di post-addestramento (vedi la sezione "cosa Backpropagate non è" per le citazioni).

## Cosa Backpropagate NON è

Essere onesti aiuta tutti. Backpropagate non fa queste cose, e cercare di farlo sarebbe un'esperienza peggiore rispetto a utilizzare lo strumento giusto:

- **Ottimizzazione completa dei parametri per modelli da 7B+** — Backpropagate utilizza LoRA / QLoRA, che addestra un piccolo adattatore invece di aggiornare ogni peso. Per i modelli da 7B e superiori, l'ottimizzazione completa richiede 24GB+ di memoria GPU e non è possibile eseguirla su una scheda consumer da 16GB. Per i modelli da 3B e inferiori, l'ottimizzazione completa è fattibile su 16GB; un'opzione `mode="full"` è prevista per la versione 1.4. In sintesi: ricerche recenti ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) mostrano che LoRA, con la configurazione corretta, raggiunge una qualità simile all'ottimizzazione completa nella maggior parte delle attività di post-addestramento (seguimento delle istruzioni, adattamento al dominio, personalità/stile) utilizzando il 67% delle risorse di calcolo; quindi, per il tipo di lavoro che la maggior parte degli utenti desidera, non si perde nulla utilizzando LoRA. Se è realmente necessario eseguire un'ottimizzazione completa di un modello da 7B+, utilizzare direttamente `transformers.Trainer` di HuggingFace su una scheda da 24GB+.
- **DPO / PPO / GRPO / ottimizzazione delle preferenze** — Backpropagate esegue solo l'ottimizzazione supervisionata in una singola fase. Per l'apprendimento basato sulle preferenze, utilizzare direttamente TRL o LLaMA-Factory.
- **Addestramento multi-nodo** — solo una GPU su una singola macchina. L'utilizzo di più GPU su una singola macchina è possibile (tramite `accelerate launch`), ma non è ufficialmente supportato.
- **Addestramento su macOS** — Apple Silicon non dispone di CUDA, quindi l'addestramento deve essere eseguito su una macchina Linux o Windows con una GPU NVIDIA. È comunque possibile eseguire il modello addestrato su un Mac tramite Ollama.
- **Qualsiasi modello al di fuori delle famiglie di modelli supportate** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Altri modelli potrebbero funzionare, ma non sono inclusi nei test automatizzati.

Se avete bisogno di queste funzionalità, utilizzate una delle librerie elencate sopra. Sono più adatte per questo scopo.

## Cosa offre Backpropagate:

Quattro cose, in un'unica installazione:

**1. Un'API semplice, composta da sole 3 righe, che funziona senza un file di configurazione.**
Lo snippet di codice all'inizio di questo file README viene eseguito completamente. Non è necessario `accelerate config`, né file YAML, né override di Hydra. Basta `Trainer(model).train(data)` e avrete un modello ottimizzato.

**2. Funzionalità che funzionano effettivamente su Windows.**
La maggior parte delle librerie di machine learning trattano Windows come un'aggiunta secondaria. Backpropagate è stato testato in modo approfondito su Windows + RTX 5080. La libreria gestisce automaticamente le peculiarità del sistema operativo, ad esempio pre-tokenizzando i dati per evitare che il multiprocessing di Windows si blocchi, disabilitando automaticamente xformers su schede RTX 40/50 dove causerebbe problemi, e impostando le opzioni del dataloader in modo da evitare errori. Non è necessario conoscere questi dettagli; il sistema funziona semplicemente.

**3. Progettato per l'esecuzione in background.**
L'addestramento richiede ore. Non volete doverlo controllare costantemente. Backpropagate è progettato per essere lasciato in esecuzione:

- Se si esaurisce la memoria della GPU, riduce automaticamente la dimensione del batch e riprova, fino a tre volte. Non è necessario alcun intervento manuale.
- Se la GPU si surriscalda, si interrompe fino a quando la temperatura non si abbassa e poi riprende.
- Ogni checkpoint viene salvato in modo atomico: se il laptop si blocca durante il salvataggio, il checkpoint precedente rimane intatto.
- Ogni esecuzione di addestramento riceve un ID univoco che viene aggiunto a ogni riga del log, a ogni checkpoint e a ogni voce di Weights & Biases. Se qualcosa va storto, un singolo ID consente a un manutentore di correlare tutti i dati.
- Gli errori vengono segnalati con codici standard (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, ecc.), in modo da poter cercare nei log e nella [guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) per trovare la soluzione. Gli errori specifici di CUDA hanno una [pagina dedicata alla risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

**4. Un solo comando per passare dall'adattatore addestrato all'esecuzione con `ollama run`.**
Molte librerie addestrano un modello. Pochi di essi semplificano l'utilizzo del modello una volta addestrato. Backpropagate esporta il modello nel formato GGUF (il formato utilizzato da Ollama) e registra un modello Ollama con un solo comando. Si passa dallo stato di "addestramento completato" allo stato di "posso chattare con il mio modello ottimizzato" in circa 30 secondi.

## Guida rapida

Il repository include un piccolo set di dati di esempio, in modo che lo script all'inizio di questo file README possa essere eseguito con un'installazione pulita:

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

Questo addestra un adattatore Qwen 2.5 7B su 5 brevi conversazioni in formato ShareGPT, quindi esporta il risultato in formato GGUF. Per i tuoi dati, formatta il file JSONL con un esempio per riga:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Anche i formati Alpaca (`instruction` / `output`), OpenAI chat (`messages`) e testo semplice funzionano: Backpropagate rileva automaticamente il formato.

Per flussi di lavoro più complessi (fine-tuning e pubblicazione su Hugging Face Hub, ripresa dopo errori di memoria, esecuzione multipla di SLAO su una campagna lunga, ecc.), consultare la [pagina delle ricette del manuale](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interfaccia web (opzionale)

Se preferisci utilizzare un'interfaccia grafica invece di scrivere codice Python, installa il componente aggiuntivo dell'interfaccia utente e avvialo:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Si apre un'interfaccia web locale all'indirizzo `http://localhost:7862`, dove puoi selezionare un set di dati, scegliere un modello, eseguire l'addestramento e l'esportazione. L'interfaccia utente è attiva solo localmente per impostazione predefinita. Per renderla accessibile da altri dispositivi, consulta la sezione [Interfaccia web](#web-ui) riportata di seguito per le opzioni `--share` e `--auth` relative alla sicurezza.

## Addestramento multiplo

Se desideri eseguire il fine-tuning in modo incrementale su più set di dati (ad esempio, ricevi nuovi dati di addestramento ogni settimana e desideri aggiungerli senza dimenticare ciò che hai imparato in precedenza), la modalità `multi_run` di Backpropagate è ciò che fa per te:

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

Questo esegue cinque cicli di addestramento, unendo l'adattatore tra un ciclo e l'altro in modo da preservare le conoscenze precedenti e incorporare nuovi esempi. Questa tecnica si basa su recenti ricerche sull'apprendimento continuo: consulta la sezione [Riferimenti](#references) in fondo a questo file README.

La versione da riga di comando (CLI):

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Ripresa da un checkpoint

Un addestramento di 5 cicli che si interrompe al quarto ciclo può essere ripreso. Ogni sessione di addestramento multiplo scrive l'ID del ciclo nel file di cronologia e nel manifest del checkpoint, quindi per riprendere da dove ti eri interrotto, è sufficiente un comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

Il comportamento predefinito di `backprop multi-run` (senza `--resume`) rileva automaticamente una sessione in corso nella stessa directory di output e la continua. Per forzare un nuovo inizio, indica una directory di output diversa.

## Cronologia dell'addestramento

Ogni invocazione di `backprop train` e `backprop multi-run` registra una riga in `<output>/run_history.json`, contenente informazioni sul modello utilizzato, il set di dati, gli iperparametri, lo stato, la perdita finale e la cronologia delle perdite. Puoi visualizzare e analizzare le esecuzioni precedenti:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Monitoraggio degli esperimenti

Backpropagate rileva automaticamente i sistemi di monitoraggio degli esperimenti installati (Weights & Biases, TensorBoard, MLflow) e li configura. Se `wandb` è installato e sei autenticato, ogni esecuzione registra automaticamente i dati su W&B con un nome che corrisponde all'ID del ciclo presente nel file di output, in modo da poter utilizzare un unico identificatore per cercare i dati su W&B, nei tuoi log e nel file `run_history.json`.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Per disabilitare questa funzionalità, utilizza `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` oppure `Trainer(report_to="none")`.

## Interfaccia web

L'interfaccia web Reflex è opzionale: installala con `pipx install "backpropagate[ui]"` e avviala:

```bash
backprop ui --port 7862
```

L'interfaccia utente viene eseguita localmente all'indirizzo `http://localhost:7862`. Per renderla accessibile da altri dispositivi (altri utenti sulla tua rete, un URL pubblico, ecc.), devi combinare le opzioni `--share` (o `--host`) con `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` senza `--auth` genera un errore. Il motivo è che `--share` pubblica un URL a cui chiunque su Internet può accedere, e senza autenticazione ciò significa che chiunque può controllare la tua pipeline di addestramento e leggere il tuo token di Hugging Face. Non è possibile disabilitare questa funzionalità: se non desideri impostare le credenziali, utilizza il port-forwarding SSH.

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

Consulta il file [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) per un'analisi completa dei rischi.

Le operazioni di scrittura sul file system dall'interfaccia utente sono limitate a una singola directory:

- Valore predefinito: `~/.backpropagate/ui-outputs`
- Per sovrascrivere: impostare `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- La sovrascrittura è validata tramite una lista di elementi non consentiti: i percorsi di sistema o di credenziali (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, ecc.) non sono ammessi.

## Note sulla piattaforma

**Requisiti:** Python 3.10+ · GPU CUDA (8GB+ di VRAM) · PyTorch 2.0+

Python 3.10 raggiungerà la fine del suo ciclo di vita ufficiale nell'ottobre 2026, e Backpropagate prevede di rimuovere il supporto per la versione 3.10 nella versione 1.4. Per le nuove installazioni, è preferibile utilizzare Python 3.11 o 3.12; la versione 3.11 è quella più testata.

Backpropagate gestisce le peculiarità dell'ambiente di runtime durante l'addestramento su diverse piattaforme, ma non può risolvere i problemi che si verificano durante l'installazione. I due problemi più comuni sono:

- **Driver CUDA errato.** PyTorch viene distribuito con una versione binaria per ogni versione di CUDA. Se si sceglie la versione errata, si ottiene silenziosamente una versione di PyTorch che utilizza solo la CPU, e l'addestramento diventa estremamente lento. Utilizzare lo strumento di selezione dei driver all'indirizzo <https://pytorch.org/get-started/locally/> per il proprio driver. Eseguire il comando `nvidia-smi` per visualizzare la versione del driver e di CUDA.
- **Windows + esportazione GGUF.** L'opzione `[export]` compila `llama-cpp-python` dal codice sorgente, il che richiede Visual Studio Build Tools (componente C++) e CMake.

**macOS:** L'addestramento con GPU non è supportato (non è disponibile CUDA). È possibile eseguire l'adattatore addestrato su un Mac tramite Ollama, ma la funzione `trainer.train()` genera un errore `DEP_GPU_NOT_AVAILABLE`. Utilizzare una macchina Linux o Windows con CUDA per l'addestramento vero e proprio.

Consultare la [pagina della guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) per una guida completa alla risoluzione dei problemi di installazione, e la [pagina dedicata alla risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) per problemi relativi a driver, VRAM, xformers e bf16 rispetto a fp16.

## CLI

Ogni API Python ha un'interfaccia a riga di comando (CLI) corrispondente:

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

La documentazione completa è disponibile nella [pagina della guida alla CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), oppure è possibile utilizzare il comando `backprop <sottocomando> --help`.

## Configurazione

Ogni impostazione può essere sovrascritta utilizzando una variabile d'ambiente, preceduta dal prefisso `BACKPROPAGATE_`:

| Variabile | Valore predefinito | Note |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Forzare i log in formato JSON o nella console |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modello predefinito |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Learning rate (tasso di apprendimento) |
| `BACKPROPAGATE_LORA__R` | `256` | Rango LoRA (valore predefinito nella versione 1.3; utilizzare l'opzione `--lora-preset=fast` per il valore predefinito della versione 1.2.x, che è 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sandbox del file system dell'interfaccia utente |

Le chiavi nidificate utilizzano il doppio underscore (`MODEL__NAME`, non `MODEL_NAME`). La documentazione completa è disponibile nella [pagina delle variabili d'ambiente](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Modelli predefiniti

| Modello | VRAM | Licenza | Note |
|---|---|---|---|
| Qwen-3.5-4B | ~8GB | Apache 2.0 | Valore predefinito consigliato per modelli inferiori a 5 miliardi di parametri. Offre la migliore qualità a questa dimensione. |
| Phi-4-mini-3.8B | ~8GB | MIT | Ottimo per ragionamento, matematica e programmazione. Licenza rigorosamente "clean". |
| SmolLM3-3B | ~6GB | Apache 2.0 | Ricetta completamente aperta. Contesto nativo a 64K. |
| Qwen 2.5 7B | ~12GB | Apache 2.0 | Valore predefinito esistente. Offre la migliore qualità tra i preset legacy da 7 miliardi di parametri. |
| Qwen 2.5 3B | ~8GB | Qwen-Research | ⚠ licenza per la ricerca: consultare i termini della licenza Qwen prima di un utilizzo commerciale. |
| Llama 3.2 3B | ~8GB | Llama Community | Un'alternativa valida a Qwen 3B, con alcune limitazioni. |
| Llama 3.2 1B | ~6GB | Llama Community | Ideale per esperimenti rapidi su schede grafiche di piccole dimensioni. |
| Mistral 7B | ~12GB | Apache 2.0 | Comparabile a Qwen 7B, con un diverso modello di conversazione. |

Altri modelli potrebbero funzionare, ma solo questi otto sono inclusi nei test automatizzati. Utilizzare l'opzione `--lora-preset=quality` (predefinita) per ottenere un rango di 256 / target all-linear, come indicato da Biderman 2024 + Thinking Machines 2025, oppure l'opzione `--lora-preset=fast` per ottenere il footprint della versione 1.2.x, con un rango di 16 / target q+v, se necessario.

## Risoluzione dei problemi

Un breve elenco dei problemi più comuni che si verificano durante la prima esecuzione. L'elenco completo è disponibile nella [pagina della guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Per un'analisi approfondita di driver, VRAM e precisione mista, consultare la [pagina dedicata alla risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Sintomo | Codice di errore | Soluzione |
|---|---|---|
| La GPU esaurisce la memoria durante l'addestramento. | `RUNTIME_GPU_OOM` | Automatic — Backpropagate dimezza la dimensione del batch e riprova fino a 3 volte. Per disattivare questa funzione: `Trainer(oom_recovery=False)`. Per forzare l'utilizzo di un batch più piccolo: `--batch-size 1`. |
| HuggingFace restituisce 401 / "modello non trovato" | `DEP_MODEL_LOAD_FAILED` | Eseguire il comando `huggingface-cli login` e riprovare. In caso di errori di battitura, copiare l'ID esatto da <https://huggingface.co/models>. |
| Rifiuto della connessione `register_with_ollama`. | `DEP_OLLAMA_REGISTRATION_FAILED` | Avviare il demone: `ollama serve`. Installare da <https://ollama.com>. Operazione riprovabile. |
| Disco pieno durante il salvataggio del checkpoint. | `STATE_CHECKPOINT_INVALID` | In caso di crash, vengono creati file `.partial`. È sicuro eliminarli. Il checkpoint precedente e valido è intatto. |
| Addestramento interrotto a causa del surriscaldamento della GPU | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automatic — Backpropagate mette in pausa l'esecuzione quando viene superata la soglia di temperatura e riprende quando la GPU si raffredda. Migliorare il flusso d'aria se il problema si verifica frequentemente. |
| Richiesta di `backprop ui --share` rifiutata. | `INPUT_AUTH_REQUIRED` | Utilizzare `--auth user:password` oppure utilizzare la reindirizzamento SSH (vedere [Interfaccia Web](#web-ui)). |
| Esportazione GGUF non riuscita al primo tentativo. | `RUNTIME_GGUF_EXPORT_FAILED` | Eseguire `pip install backpropagate[export]`; su Windows è necessario anche Visual C++ Build Tools + CMake. |

## Segnalazione di bug

Quando si verifica un errore, Backpropagate stampa una riga all'avvio, simile a `run_started run_id=<uuid>`, e associa lo stesso ID a ogni riga del log, a ogni checkpoint e a ogni voce di Weights & Biases. **Includere il `run_id` in qualsiasi segnalazione di bug**; questo permette al manutentore di correlare tutti gli elementi relativi a quella specifica esecuzione.

Una segnalazione di bug efficace include:

1. **Il `run_id`**: l'UUID stampato all'avvio.
2. **Il codice di errore**: la riga `[CODE_NAME]: message` presente nell'output di errore standard (stderr). Consultare [codici di errore](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) per l'elenco completo.
3. **La riga di comando mascherata.** L'output di errore standard viene automaticamente mascherato (i token di autorizzazione, le stringhe `sk-*`, `hf_*`, le chiavi AWS, le coppie `password=` / `token=` vengono eliminate) e può essere tranquillamente incollata. Per visualizzare la traccia completa e non mascherata, eseguire nuovamente il comando con l'opzione `--verbose`, ma esaminarla prima di pubblicarla.
4. **Versioni di Python / PyTorch, modello della GPU, sistema operativo.** `backprop info` stampa tutte queste informazioni in un'unica volta.

Domande, suggerimenti o discussioni su "se questo è un comportamento previsto" devono essere poste nella sezione [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Le segnalazioni di problemi di sicurezza devono essere inviate privatamente tramite il modulo [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) e consultare il file [SECURITY.md](SECURITY.md) per le relative policy.

## Privacy

Tutto l'addestramento avviene localmente sulla GPU. Backpropagate non effettua richieste di rete, ad eccezione del download dei modelli da HuggingFace (che viene avviato dall'utente). Non sono presenti telemetrie né dipendenze dal cloud.

## Riferimenti

Le impostazioni predefinite di Backpropagate e la modalità di addestramento multi-run si basano su recenti ricerche. Se si è interessati alle tecniche sottostanti:

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — l'articolo fondamentale che introduce LoRA, la tecnica utilizzata da Backpropagate per addestrare gli adattatori in modo efficiente.
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidenze empiriche che dimostrano che LoRA con un rank di 256 e target lineari raggiunge una qualità simile al fine-tuning completo nella maggior parte delle attività di post-addestramento, utilizzando il 67% della potenza di calcolo. Definisce la configurazione LoRA predefinita della versione 1.3 di Backpropagate.
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — un approfondimento pratico che identifica la correzione del fattore 10 nel learning rate rispetto al fine-tuning completo, necessaria per i rank LoRA elevati.
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la caratterizzazione originale del motivo per cui le reti neurali "dimenticano" l'addestramento precedente quando vengono sottoposte a fine-tuning su nuovi dati (EWC — Elastic Weight Consolidation).
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, un approccio precedente per utilizzare LoRA per l'apprendimento continuo, che vincola i nuovi adattatori a sottospazi ortogonali.
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — una tecnica fondamentale per unire più modelli sottoposti a fine-tuning senza interferenze.
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — l'algoritmo specifico implementato dal modulo di unione multi-run di Backpropagate. Un preprint di dicembre 2025; Backpropagate è il primo utilizzatore a valle noto di questo articolo.

## Licenza

MIT — vedere [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
