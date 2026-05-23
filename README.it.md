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
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**Messa a punto di LLM senza interfaccia utente in 3 righe. Impostazioni predefinite intelligenti, dimensionamento automatico dei batch in base alla VRAM, addestramento SLAO multi-sessione e esportazione GGUF con un solo clic per Ollama.**

*SLAO è Single LoRA Continual Learning via Asymmetric Merging: una tecnica di unione tra sessioni di addestramento che previene la perdita di informazioni durante le sessioni di messa a punto prolungate (articolo: [https://arxiv.org/abs/2512.23017]).*

*Addestra modelli linguistici di grandi dimensioni con 3 righe di codice. Esportali su Ollama con un'altra riga.*

## Guida rapida

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("examples/quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

Il repository include un piccolo file `examples/quickstart.jsonl` (5 esempi in formato ShareGPT) in modo che lo snippet precedente possa essere eseguito completamente su un'installazione pulita. Per il tuo addestramento, consulta la sezione [Formato del dataset](#dataset-format) sottostante.

### Percorso senza codice: Interfaccia web

Preferisci un'interfaccia utente anziché una REPL Python? Installa gli extra necessari ed esegui:

```bash
pip install backpropagate[standard]
backprop ui --port 7862
```

L'interfaccia Reflex (Radix UI) ti consente di puntare a un file JSONL, selezionare un modello, addestrare e esportare: non è richiesta alcuna conoscenza di Python. L'interfaccia utente è progettata per funzionare principalmente in locale; per l'accesso tramite internet, consulta la sezione [Interfaccia web](#web-ui) sottostante per il contratto di sicurezza `--share` + `--auth` e le opzioni di tunneling supportate (Cloudflare Tunnel, ngrok).

## Formato del dataset

Il tuo file di addestramento JSONL dovrebbe contenere un esempio per riga. Il formato più semplice è la chat ShareGPT:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Sono supportati anche i formati Alpaca (`instruction`/`output`), OpenAI chat (`messages`) e testo semplice. Consulta `examples/quickstart.jsonl` per un punto di partenza.

## Perché propagare il gradiente?

| Problema | Soluzione |
|---------|----------|
| La messa a punto è complessa | 3 righe: caricamento, addestramento, salvataggio |
| Windows è un incubo | Supporto completo per Windows |
| La gestione della VRAM è difficile | Dimensionamento automatico dei batch, monitoraggio della GPU |
| L'esportazione del modello è complicata | Esportazione GGUF con un solo clic + registrazione automatica con Ollama |
| Le sessioni di addestramento prolungate causano la perdita di informazioni | Addestramento SLAO multi-sessione |

## Caratteristiche principali

- **Senza interfaccia utente per design**: Progettato per pipeline CI/CD, flussi di lavoro automatizzati ed esecuzione programmatica.
- **Impostazioni predefinite intelligenti**: Configura automaticamente gli iperparametri ottimali in base all'hardware e al dataset.
- **Addestramento SLAO multi-sessione**: Strategie di addestramento avanzate per prevenire la perdita di informazioni durante le sessioni di addestramento prolungate.
- **Supporto completo per Windows**: Testato e ottimizzato per gli ambienti Windows, evitando i problemi comuni di PyTorch/CUDA.
- **Esportazione semplice**: Esportazione con un solo clic nel formato GGUF e registrazione automatica con Ollama.
- **Architettura modulare**: Installa solo le dipendenze necessarie (ad esempio, `[unsloth]`, `[ui]`, `[export]`).

## Installazione

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Reflex (Radix UI) web interface
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Extra | Descrizione | Dipendenze |
|-------|-------------|--------------|
| `unsloth` | Addestramento 2 volte più veloce, 50% di VRAM in meno | unsloth |
| `ui` | Interfaccia web Reflex (Radix UI) | reflex>=0.9.2, fastapi>=0.115 |
| `validation` | Validazione della configurazione Pydantic | pydantic, pydantic-settings |
| `export` | Esportazione GGUF per Ollama | llama-cpp-python |
| `monitoring` | WandB + monitoraggio del sistema (integrato automaticamente nel trainer dalla versione 1.1.0) | wandb, psutil |
| `logging` | Logging strutturato | structlog |
| `security` | Autenticazione JWT + generazione di token | PyJWT, cryptography |
| `production` | unsloth + ui + validation + logging + security | (bundle) |

**Requisiti:** Python 3.10+ · GPU CUDA (8GB+ di VRAM) · PyTorch 2.0+

### Prerequisiti della piattaforma

Backpropagate gestisce le peculiarità del runtime (multiprocessing, xformers su RTX 40/50, worker del dataloader su Windows). Non gestisce i problemi di installazione specifici della piattaforma: risolvili prima.

- **Versione del toolkit CUDA.** PyTorch viene distribuito in base alla versione di CUDA; scegliere la versione sbagliata installa silenziosamente solo la versione per CPU. Utilizzare lo strumento di selezione all'indirizzo <https://pytorch.org/get-started/locally/> per ottenere il comando `pip install torch ...` corretto per il proprio driver. Eseguire `nvidia-smi` per visualizzare la versione del driver/CUDA.
- **Windows.** Sono necessari Visual Studio Build Tools (C++) e CMake per l'estensione `[export]` (la compilazione di `llama-cpp-python` avviene dal codice sorgente). La versione per Windows di `bitsandbytes` è ora disponibile nativamente (>= 0.43); le guide precedenti che menzionano `bitsandbytes-windows` sono obsolete.
- **macOS.** L'addestramento con GPU **non è supportato**; non è disponibile CUDA. È possibile installare Backpropagate per eseguire l'inferenza su un file GGUF tramite Ollama, ma `trainer.train()` genera l'errore `DEP_GPU_NOT_AVAILABLE`. Utilizzare una macchina con GPU per l'addestramento.
- **Linux.** La maggior parte delle distribuzioni funziona immediatamente. Se si utilizza la versione binaria distribuita tramite PyPI, si noti che la versione per Linux utilizza solo la versione per CPU di torch (per rimanere al di sotto del limite di 2 GB per gli asset di rilascio di GitHub); installare prima la versione CUDA corrispondente da pytorch.org.

Per la risoluzione dei problemi di installazione, consultare [la pagina della guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/).

## Configurazione

Tutte le impostazioni possono essere sovrascritte tramite variabili d'ambiente utilizzando il prefisso `BACKPROPAGATE_` (ad esempio, `BACKPROPAGATE_LOG_LEVEL=debug`). Un file `.env` nella directory principale del progetto viene caricato automaticamente quando viene installata l'estensione `[validation]`.

Impostazioni comuni (vedere [il riferimento completo alle variabili d'ambiente](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/) per tutte le opzioni):

| Variabile | Valore predefinito | Note |
|----------|---------|-------|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Forza i log in formato JSON (`true`) o nella console (`false`) |
| `BACKPROPAGATE_LOG_FILE` | non impostato | Percorso per salvare i log |
| `BACKPROPAGATE_DEFER_FEATURE_DETECTION` | non impostato | Salta il rilevamento delle dipendenze opzionali all'avvio per un avvio più rapido della CLI |
| `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE` | `true` | Se impostato su `true`, rifiuta `backprop ui --share` senza l'opzione `--auth` |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Directory di base per tutte le scritture del file system dell'interfaccia utente; con convalida della whitelist |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modello predefinito |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Learning rate (tasso di apprendimento) |
| `BACKPROPAGATE_LORA__R` | `16` | LoRA rank (grado di LoRA) |

Le chiavi nidificate utilizzano il doppio underscore come delimitatore (convenzione `env_nested_delimiter` di Pydantic).

## Utilizzo

### Addestramento di base

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

`Qwen/Qwen2.5-7B-Instruct` è il valore predefinito; questo è il valore che `Trainer()` restituisce quando viene chiamato senza un argomento del modello (vedere [`config.py`](backpropagate/config.py) `ModelConfig.name`). Gli esempi precedenti utilizzavano `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (versione quantizzata), ma siamo passati ai pesi ufficiali di Qwen per una maggiore affidabilità ([CHANGELOG v1.1.0](CHANGELOG.md#110---2026-05-21)). Entrambi i modelli funzionano.

### Addestramento multi-run SLAO

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

SLAO (Single LoRA Continual Learning via Asymmetric Merging) implementa l'articolo [Merge before Forget](https://arxiv.org/abs/2512.23017): inizializzazione ortogonale della matrice A tramite decomposizione QR, gestione asimmetrica di A/B e scalatura dipendente dal tempo `λ(i) = 1/√i`. Il flag della CLI è `--samples` (il campo sottostante è `samples_per_run`).

### Esportazione in Ollama

```python
# Export to GGUF
result = trainer.export("gguf", quantization="q4_k_m")

# Register with Ollama separately
from backpropagate import register_with_ollama
register_with_ollama(result.path, "my-finetuned-model")
# ollama run my-finetuned-model
```

### CLI

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

Consultare [il riferimento della CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) per tutti i sottocomandi e i flag, oppure eseguire `backprop <sottocomando> --help`.

### Ripresa da checkpoint (v1.1.0)

Un addestramento multi-run di 5 iterazioni che si interrompe all'iterazione 4 può ora essere ripreso. Ogni sessione di addestramento multi-run scrive il suo `run_id` sia in `run_history.json` che nel manifest del checkpoint su disco, quindi per riprendere da dove si era interrotto, è sufficiente un comando:

```bash
backprop resume <run-id>                       # picks up the in-progress session
backprop multi-run --data ... --resume <run-id> # explicit form
backprop train --data ... --resume <run-id>    # single-run resume (continues run_id)
```

Il comportamento predefinito di `backprop multi-run` (senza `--resume`) rileva automaticamente una sessione in corso per la stessa directory di output e la continua. Per forzare l'avvio di una nuova sessione, utilizzare `resume_from="off"` (API Python) oppure omettere `--resume` e specificare una nuova directory di output.

Quando una sessione multi-run viene ripresa, il checkpoint più recente per quell'ID di esecuzione viene caricato nel modello, lo stato di fusione SLAO viene ripristinato dalla directory `slao/` accanto al checkpoint e il ciclo di esecuzione continua da `last_completed_run + 1`. Lo stato della voce nella cronologia viene riportato a `running`, quindi `backprop list-runs --status running` mostra la sessione attiva.

### Monitoraggio degli esperimenti (v1.1.0)

`Trainer` rileva automaticamente i sistemi di monitoraggio degli esperimenti installati (`wandb`, `tensorboard`, `mlflow`) e li integra con i parametri di training di `transformers`. Il valore predefinito `report_to="auto"` utilizza qualsiasi sistema importabile:

```bash
pip install backpropagate[monitoring]  # installs wandb + psutil
wandb login                            # one-time
backprop train --data my_data.jsonl    # W&B run gets the same run_id prefix as the on-disk history
```

Per disabilitare esplicitamente queste funzionalità, utilizzare `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` oppure `Trainer(report_to="none")`. Per MLflow, installare `pip install mlflow`; per TensorBoard, installare `pip install tensorboard`. Il nome dell'esecuzione W&B è `backprop-<run_id_prefix>`, consentendo agli operatori di effettuare ricerche in W&B, nei log e in `run_history.json` utilizzando lo stesso identificatore.

### Cronologia dell'addestramento

Ogni invocazione di `backprop train` e `backprop multi-run` registra una riga in `<output>/run_history.json` contenente l'ID dell'esecuzione, il modello, il dataset, gli iperparametri, lo stato, la perdita finale, la cronologia delle perdite e, per le esecuzioni multi-run, la cronologia della fusione SLAO. Per visualizzare le esecuzioni recenti, utilizzare:

```bash
backprop list-runs                         # most recent 20 runs, all statuses
backprop list-runs --status failed         # filter
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial run_id ok)
```

La cronologia delle esecuzioni viene mantenuta tra i processi: la scheda "Runs" nell'interfaccia web è una visualizzazione separata in memoria; la cronologia salvata su disco è la fonte di verità per `list-runs` / `show-run` / `resume`.

### Interfaccia web

Per avviare l'interfaccia Reflex localmente:

```bash
backprop ui --port 7862
```

Per rendere disponibile un URL accessibile da Internet, è necessario associare `--share` a `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` senza `--auth` termina con il codice `1` e l'errore strutturato `[INPUT_AUTH_REQUIRED]`. La motivazione è che `--share` pubblica un URL `*.gradio.live` che chiunque su Internet può raggiungere, e senza autenticazione ciò significa che chiunque può controllare la pipeline di addestramento.

Per disabilitare esplicitamente questa funzionalità (ad esempio, in un ambiente di sviluppo interno), impostare la variabile d'ambiente `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false`. Verrà visualizzato un avviso ben visibile ad ogni avvio e c'è un periodo di grazia di 5 secondi prima che l'interfaccia utente non autenticata venga caricata, quindi è possibile interrompere l'esecuzione con `Ctrl-C` se qualcosa non sembra corretto.

Le operazioni di scrittura sul file system dall'interfaccia utente sono limitate a una singola directory:

- Predefinito: `~/.backpropagate/ui-outputs`
- Override: `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- L'override è **validato tramite una lista di esclusione**: i percorsi di sistema/credenziali (`/etc`, `/var`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, ecc.) vengono rifiutati con l'errore `[UI_OUTPUT_DIR_FORBIDDEN]`.

## Supporto per Windows

Backpropagate è progettato per funzionare su Windows senza modifiche:

- Pre-tokenizzazione per evitare arresti anomali dovuti all'elaborazione parallela
- Disabilitazione automatica di xformers per le serie RTX 40/50
- Impostazioni del dataloader sicure
- Testato su RTX 5080 (16GB VRAM)

## Modelli predefiniti

| Modello | VRAM | Velocità | Qualità |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12GB | Media | Ottima |
| Qwen 2.5 3B | ~8GB | Veloce | Buona |
| Llama 3.2 3B | ~8GB | Veloce | Buona |
| Llama 3.2 1B | ~6GB | Più veloce | Base |
| Mistral 7B | ~12GB | Media | Buona |

## Architettura

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
└── ui_security.py       # Rate limiting, CSRF, file validation (framework-agnostic)
```

L'implementazione di Gradio v1.0 (`ui_gradio_legacy.py` + `theme_gradio_legacy.py`) è stata mantenuta fino alla versione v1.1.x come riferimento ed è stata rimossa nella versione v1.2.0.

## Risoluzione dei problemi

Un breve elenco dei problemi più comuni riscontrati all'avvio. L'indice completo è disponibile nella [pagina della guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/); ogni codice riportato di seguito è documentato nella sezione [codici di errore](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/).

| Sintomo | Codice | Soluzione |
|---------|------|-----|
| La GPU esaurisce la memoria durante l'addestramento. | `RUNTIME_GPU_OOM` | Il meccanismo di ripristino automatico OOM (B-002) riduce automaticamente la dimensione del batch fino a 3 volte. Per disabilitarlo: `Trainer(oom_recovery=False)`. Per forzare una dimensione inferiore: `--batch-size 1`. |
| Il servizio HF Hub restituisce un errore 401 / "modello non trovato". | `DEP_MODEL_LOAD_FAILED` | Eseguire il comando `huggingface-cli login` e riprovare. In caso di errori di battitura, copiare l'ID esatto da <https://huggingface.co/models>. |
| Errore nel nome del modello. | `INPUT_VALIDATION_FAILED` o `DEP_MODEL_LOAD_FAILED`. | Verificare l'identificatore `org/name` su <https://huggingface.co/models>. |
| Rifiuto della connessione `register_with_ollama`. | `DEP_OLLAMA_REGISTRATION_FAILED` | Avviare il demone: `ollama serve`. Installare da <https://ollama.com>. Operazione riprovabile. |
| Disco pieno durante il salvataggio del checkpoint. | `STATE_CHECKPOINT_INVALID` | In caso di crash, vengono creati file `.partial`. È sicuro eliminarli. Il checkpoint precedente e valido è intatto. |
| L'addestramento viene interrotto/annullato a causa del surriscaldamento della GPU. | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | B-003: il monitor si interrompe quando viene superata la soglia di temperatura NVML; riprende automaticamente quando la GPU si raffredda. Migliorare il flusso d'aria o ridurre il carico sostenuto. |
| Richiesta di `backprop ui --share` rifiutata. | `INPUT_AUTH_REQUIRED` | Passare l'argomento `--auth user:password` oppure impostare `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false` per disabilitare la funzione (con un avviso). |
| "Sovrapposizione" di esecuzioni multiple durante la validazione. | `CONFIG_INVALID` (backend Stage A B-001). | Ridurre il valore di `--samples` al di sotto della dimensione del pool di addestramento, aumentare le dimensioni del dataset o disabilitare la validazione. |
| Esportazione GGUF non riuscita al primo tentativo. | `RUNTIME_GGUF_EXPORT_FAILED` | Eseguire `pip install backpropagate[export]`; su Windows è necessario anche Visual C++ Build Tools + CMake. |

## Segnalazione di bug

Quando si verifica un errore, Backpropagate stampa una riga `run_started run_id=<uuid>` all'avvio e associa lo stesso ID ai manifest dei checkpoint, alla cronologia delle unioni SLAO e alle righe del log strutturato. Includere l'`run_id` in qualsiasi segnalazione di bug: questo consente all'amministratore di correlare ogni riga del log, ogni checkpoint e ogni unione per quella specifica esecuzione.

Una segnalazione di bug efficace include:

1. **`run_id`** — l'UUID stampato all'avvio (disponibile anche come `TrainingRun.run_id` e `RunResult.run_id`).
2. **Il codice di errore** — la riga `[CODE_NAME]: message` nella sezione stderr; consultare la [sezione codici di errore](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) per l'elenco completo.
3. **La riga di comando, opportunamente oscurata.** Nella modalità non dettagliata, la sezione stderr viene oscurata automaticamente (i token Bearer, `sk-*`, `hf_*`, le chiavi AWS, le coppie `password=/token=/api_key=` vengono rimosse); è quindi sicuro copiarla. Per ottenere il traceback completo e non oscurato, eseguire nuovamente il comando con l'opzione `--verbose`, ma esaminare il contenuto prima di pubblicarlo.
4. **Versioni di Python / PyTorch, modello della GPU, sistema operativo.** Il comando `backprop info` stampa tutte queste informazioni in un'unica volta.

## Privacy

Tutto l'addestramento avviene localmente sulla GPU. Backpropagate non effettua richieste di rete, ad eccezione del download dei modelli da HuggingFace (che viene avviato dall'utente). Non sono presenti telemetrie né dipendenze dal cloud.

## Valutazione

| Categoria | Punteggio | Note |
|----------|-------|-------|
| A. Sicurezza | 6/8 | SECURITY.md, modello affidabile, nessuna informazione sensibile/telemetria, safe_path(). Elementi MCP esclusi. |
| B. Gestione degli errori | 5/7 | Struttura delle eccezioni (`codice`/`messaggio`/`suggerimento`/`causa`/`riprovabile`) tramite il registro ERROR_CODES; codici di uscita della CLI: 0/1/2/3; nessuna traccia dello stack non elaborata senza l'opzione `--verbose`; correlazione `run_id`; output di errore standard oscurato; controllo di accesso con `--share`+`--auth`. MCP/desktop/vscode esclusi. |
| C. Documentazione per gli operatori | 4/7 | README, CHANGELOG, LICENZA, --help. Logging/MCP/funzionalità complesse esclusi. |
| D. Igiene del rilascio | 6/9 | verify.sh, versione=tag, 5 scanner nell'integrazione continua, dependabot, python_requires, build pulito. |
| E. Identità | 4/4 | Logo, traduzioni, pagina di destinazione, metadati. |
| **Total** | **25/31** | 14 elementi esclusi con motivazione · `shipcheck audit` supera il 100% · Data dell'audit: 2026-05-21 (la riga B è stata riclassificata dopo la fase B + il lavoro sui codici di uscita della CLI). |

Storia del design e a cosa corrisponde ogni elemento: vedere [ROADMAP.md](ROADMAP.md) — tutti gli elementi delle settimane 1-4 sono inclusi nella versione 1.1.0.

## Licenza

MIT — vedere [LICENSE](LICENSE) per i dettagli.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
