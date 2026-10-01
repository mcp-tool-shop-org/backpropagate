<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

# Ottimizza un modello QLoRA da 32B o un modello end-to-end da 7B su una singola GPU. Caricalo su Ollama

Esegui il backpropagation per ottimizzare modelli linguistici di grandi dimensioni su una **singola** GPU, dimensionata in base alla scheda che hai effettivamente. Tre righe di codice Python QLoRA per un modello da 7B-32B su una singola scheda consumer da 32 GB (RTX 5090). Un flag, `--full-ft-offload`, ottimizza completamente un modello di classe 7B mantenendo i suoi pesi e gradienti nella RAM dell'host (Linux o WSL2; lento, e misurato di seguito). Un comando aggiuntivo esporta su Ollama, quindi `ollama run` la tua ottimizzazione. Si riduce a 16 GB. Ottimo su Windows.

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

Questo è tutto. Non c'è un file di configurazione YAML. Non c'è alcuna "cerimonia" `accelerate launch`. Non c'è un tutorial separato del tipo "ora convertilo in GGUF". Se hai una GPU CUDA e un file JSONL con i tuoi dati di addestramento, sei a tre righe da un'ottimizzazione funzionante.

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

Preferisci Docker? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` funziona anche. Sono disponibili immagini sia per `linux/amd64` che per `linux/arm64`, quindi gli utenti di Apple Silicon e ARM Linux ottengono un'immagine nativa. Un `compose.yaml` canonico per "UI in un container" si trova nella directory principale del repository: inserisci `user:password` in un `ui-auth.txt` accanto ad esso, esegui `docker compose up` e accedi a `http://127.0.0.1:7860` (il primo avvio crea l'interfaccia utente, il che richiede un minuto o due). La cronologia delle esecuzioni viene salvata in `~/.backpropagate`.

## Dove si colloca Backpropagate nello spazio delle librerie

Esistono diverse buone librerie per l'ottimizzazione di LLM. Ognuna di esse è ottima per cose diverse:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — se ti piacciono le configurazioni YAML e desideri una community di ricette da cui copiare
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — se desideri DPO/PPO/RLHF e un'interfaccia utente web
- **[Unsloth](https://github.com/unslothai/unsloth)** — se hai bisogno dell'addestramento più veloce possibile e utilizzi una famiglia di modelli supportata
- **[torchtune](https://github.com/pytorch/torchtune)** — se desideri le ricette PyTorch native di Meta che puoi modificare

Backpropagate è l'opzione mancante: **un'API Python di 3 righe per gli utenti singoli su una singola GPU consumer che desiderano addestrare un adattatore e caricarlo.** Nessun YAML, nessuna GUI, nessun RL online (PPO/GRPO), nessun multi-nodo. Solo il ciclo di cui tutti hanno realmente bisogno e il passaggio di esportazione che crea problemi.

Se hai provato una delle librerie di cui sopra e hai avuto problemi con la "cerimonia" del file di configurazione, o hai riscontrato un limite della famiglia di modelli, o hai desiderato impostazioni predefinite per Windows, Backpropagate è la soluzione.

## Cosa puoi ottimizzare su una singola GPU

Backpropagate dimensiona l'esecuzione in base alla tua scheda. Questi sono numeri **misurati** del 2026-09-30 su una RTX 5090 da 32 GB (prove: [`docs/receipts/2026-09-30-offload/`](docs/receipts/2026-09-30-offload/)). I picchi di QLoRA si verificano alla finestra di contesto completa delle impostazioni predefinite con batch 1, che è il caso peggiore per tali impostazioni; esempi più brevi utilizzano meno risorse.

| Modello | Metodo | Misurato su una scheda da 32 GB |
|---|---|---|
| **14B** (Qwen2.5-14B) | QLoRA | **25,0 GiB** di picco a 4096 di contesto (28,1 GiB riservati). |
| 24B (Mistral-Small-24B) | QLoRA | 26,5 GiB di picco a 4096 di contesto (29,6 GiB riservati). |
| **32B** (Qwen2.5-32B) | QLoRA | **Ci sta appena:** 28,8 GiB di picco a 2048 di contesto (30,7 GiB riservati, circa 0,65 GiB di spazio libero). |
| 3B | `mode="full"` (ottimizzazione completa vera e propria, sulla GPU) | **22,0 GiB** di picco (a livello di sistema), 0,30 s/passo con batch 4, 512 di contesto. 7,5 GiB di questo è lo stato dell'ottimizzatore, che può essere scaricato nella RAM dell'host su una scheda più piccola (non testato). |
| **Classe 7B** (Qwen2.5-7B, 7,6B parametri) | `mode="full" --full-ft-offload` | **Addestra:** 5,3 GiB di VRAM, **30,8 GiB di RAM dell'host** (32,2 GiB durante il salvataggio), **14,7 s/passo**. Solo Linux o WSL2. |

Non rimisurato in quella sessione: 7B QLoRA, Llama-3.1-8B (repository gated, nessun token sulla macchina di test) e ottimizzazione completa pura su GPU superiore a 3B. Le cifre per questi sono stime e si trovano altrove nella documentazione.

Due cose per cui la maggior parte delle librerie per singola GPU ti indirizzano altrove, **QLoRA da 24-32B** e **ottimizzazione completa di classe 7B su una singola scheda**, Backpropagate le esegue su una singola scheda consumer, quindi esporta il risultato direttamente su Ollama.

**L'ottimizzazione completa ha due percorsi.** Senza scaricamento, il modello, i suoi gradienti e lo stato dell'ottimizzatore si trovano tutti sulla GPU. La libreria limita la dimensione del modello in base alla VRAM rilevata (**16 GB → 4B, 24 GB → 5B, 32 GB → 6B**); questi limiti derivano dal calcolo della memoria e sono misurati solo fino a 3B. Sovrascrivi con `--full-ft-ceiling-billions`.

`--full-ft-offload` mantiene pesi e gradienti nella RAM dell'host e li trasmette alla GPU (FSDP2 CPU offload). Ciò che costa, misurato:

- **RAM dell'host:** il controllo di adattamento richiede circa 3,7 GiB per miliardo di parametri più 10 GiB, il che è conservativo (39 GiB a 7,6B rispetto ai 32,2 GiB misurati). L'esecuzione viene rifiutata all'inizio, con i numeri, se la macchina non può contenerla. Un modello da 7,6B non si adatta a un limite di memoria WSL2 di 28 GB; circa 5B è il limite pratico.
- **Velocità:** 14,7 s/passo a 7,6B (batch 1) e 5,1 s/passo a 3B (batch 4), rispetto a circa 0,63 s/passo per 3B sulla GPU a batch 4. Utilizzalo solo quando il modello non si adatta senza. È prevista una versione più veloce.
- **Ottimizzatore:** Adafactor, non AdamW. I pesi rimangono in bf16 e ogni aggiornamento viene riscritto con arrotondamento stocastico; non c'è una copia fp32.
- **Qualità:** in una esecuzione da 3B (150 passaggi, perdita su dati non utilizzati, un seme), ha raggiunto circa l'85% del miglioramento ottenuto dall'ordinaria ottimizzazione completa (2,45 → 1,93 rispetto a 2,45 → 1,84). Un seme non è un benchmark.
- **Ambito:** semplice ottimizzazione supervisionata. Nessun impacchettamento, nessuna maschera solo per le risposte, nessun checkpoint intermedio, nessuna ripresa. Solo Linux o WSL2 (FSDP2 richiede NCCL); su Windows nativo, si interrompe con `DEP_FSDP_UNAVAILABLE`.
- **Non ancora testato:** esecuzioni lunghe, accumulo di gradienti superiore a 1 e una macchina fisica da 64 GB (la macchina di test aveva più RAM, con un limite di 60 GiB applicato nel test).

Un modello che non si adatta esce con `RUNTIME_FULL_FT_MODEL_TOO_LARGE` e indica la via d'uscita. Consultare [la pagina completa del manuale di ottimizzazione](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/).

### Si ridimensiona fino a 16 GB

L'intervallo di 16 GB (RTX 4080 / 5080 / 4070 Ti Super) è ancora di prim'ordine: 7B QLoRA e una vera ottimizzazione completa di un modello da ~3B (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B/1B) tramite `mode="full"` (22,0 GiB misurati su una scheda da 32 GB a 3B, di cui 7,5 GiB è lo stato dell'ottimizzatore memorizzato nella cache che può essere trasferito alla RAM dell'host; non è stato ancora testato se questo funziona in modo accettabile su una scheda da 16 GB). Con `--full-ft-offload`, la GPU contiene molto meno: con la VRAM limitata sulla scheda di test, un modello da 3B viene addestrato con un limite di 6 GiB e i modelli da 4B e 7,6B con un limite di 8 GiB. Si tratta di limiti simulati su una scheda da 32 GB, non di esecuzioni su hardware reale da 8 GB. Lo stesso codice seleziona la dimensione del batch e il limite che si adattano alla scheda rilevata.

La quantizzazione a 2 bit (AQLM / QuIP#) è **esclusa** — una base a 2 bit non può essere unita in modo pulito ai pesi a precisione completa, il che interrompe il contratto di esportazione mergeable-adapter → GGUF → Ollama (che è lo scopo principale della pipeline). Backpropagate offre invece le funzionalità di ottimizzazione: QLoRA, `mode="full"`, `--full-ft-offload` e il percorso di calcolo FP8 (`--fp8`, Blackwell/Hopper), che rimangono tutti unibili ed esportabili.

## A cosa NON serve Backpropagate

Se il tuo caso d'uso è tra quelli elencati di seguito, otterrai risultati migliori con una libreria diversa: Backpropagate non è la scelta giusta e cercare di farlo funzionare costerebbe più che semplicemente utilizzare lo strumento appropriato. Leggere questa sezione prima di iniziare consente di evitare il ciclo di installazione e riavvio:

- **Ottimizzazione completa dei parametri di modelli da 13B+** — Backpropagate esegue l'ottimizzazione completa fino a circa 6B su una GPU da 32 GB e un modello di classe 7B con `--full-ft-offload` (vedere [l'intervallo](#what-you-can-fine-tune-on-one-gpu)). L'ottimizzazione completa di un modello da 13B+ richiede FSDP multi-GPU o una scheda più grande. Prima di investire in tale potenza di calcolo, valuta attentamente i pro e i contro. [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/) riporta che LoRA corrisponde all'ottimizzazione completa quando viene applicato a ogni livello e il set di dati si adatta alla capacità dell'adattatore, con circa due terzi della potenza di calcolo per passaggio. [Biderman et al. 2024](https://arxiv.org/abs/2405.09673) hanno scoperto che nelle impostazioni standard a basso rango, LoRA ha prestazioni significativamente inferiori rispetto all'ottimizzazione completa su codice e matematica, pur dimenticando meno. Per l'addestramento basato su istruzioni, la personalizzazione e lo stile su set di dati modesti, QLoRA fino a 32B è solitamente l'uso migliore di una singola scheda.
- **RL online — PPO / GRPO / RLVR** — Backpropagate esegue l'SFT a fase singola più l'ottimizzazione delle preferenze senza riferimento (ORPO nella v1.5; SimPO + KTO nella v1.6). Non esegue l'apprendimento per rinforzo online: PPO, GRPO o RLVR, che richiedono un modello di ricompensa o un ciclo di generazione e valutazione in aggiunta al passaggio di addestramento. Per questi, utilizzare direttamente TRL o LLaMA-Factory. (L'ottimizzazione delle preferenze senza riferimento si adatta all'intervallo a fase singola perché non è necessario memorizzare un modello di riferimento separato in memoria; vedere la nota ORPO in [Avvio rapido](#quick-start).)
- **Addestramento multi-nodo** — solo una GPU su una singola macchina. Il multi-GPU su una singola macchina funziona (tramite `accelerate launch`) ma non è supportato ufficialmente.
- **Addestramento macOS sulla piattaforma CUDA** — Apple Silicon non dispone di CUDA, quindi il percorso CUDA viene eseguito su una macchina Linux o Windows con una GPU NVIDIA. È comunque possibile eseguire il modello addestrato su un Mac tramite Ollama. Un percorso MLX **sperimentale e non verificato** (`--backend mlx`) addestra nativamente un adattatore LoRA su Apple Silicon — vedere [Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview). È solo LoRA-SFT e **non è stato verificato su hardware reale** (nessun supporto), quindi per qualsiasi cosa oltre a un LoRA SFT (ORPO, ottimizzazione completa, FP8, esecuzioni multiple) è necessario il percorso CUDA.
- **Qualsiasi cosa al di fuori delle famiglie di modelli testate** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Altri modelli spesso funzionano, ma non sono fissati nei test CI.

Se hai bisogno di una di queste cose, utilizza una delle librerie elencate sopra. Sono più adatte a questo scopo.

## Cosa ti offre Backpropagate

Quattro cose, in una singola installazione:

**1. Una vera API a 3 righe che funziona senza un file di configurazione.**
Lo snippet all'inizio di questo README funziona dall'inizio alla fine. Nessun `accelerate config`, nessun YAML, nessuna sovrascrittura Hydra. Basta `Trainer(model).train(data)` e hai un modello ottimizzato.

**2. Windows che funziona davvero.**
La maggior parte delle librerie ML trattano Windows come un ripensamento. Backpropagate è testato in modo completo su Windows + RTX 5080. La libreria gestisce le peculiarità del runtime per te: sa come pre-tokenizzare i dati in modo che l'elaborazione parallela di Windows non si blocchi, disabilita automaticamente xformers sulle schede RTX 40/50 dove causerebbe problemi e seleziona le impostazioni del caricatore di dati che non causano errori. Non devi sapere nulla di tutto questo. Funziona semplicemente.

**3. Progettato per le esecuzioni non presidiate.**
L'addestramento richiede ore. Non vuoi doverlo monitorare costantemente. Backpropagate è progettato per essere lasciato in esecuzione:

- Se esaurisci la memoria della GPU, dimezza automaticamente la dimensione del batch e riprova, fino a tre volte. Nessuna regolazione manuale.
- Se la tua GPU diventa troppo calda, si mette in pausa finché le cose non si raffreddano e poi continua.
- Ogni checkpoint viene scritto in modo atomico: se il tuo laptop si blocca durante il salvataggio, il checkpoint precedente e valido rimane intatto.
- Ogni esecuzione di addestramento riceve un ID univoco che viene stampato su ogni riga del log, su ogni checkpoint e su ogni voce di Weights & Biases. Se qualcosa va storto, un singolo ID consente a un manutentore di correlare tutto.
- Gli errori sono accompagnati da codici stabili (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, ecc.) in modo da poter cercare nei log e nella [guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) per trovare la soluzione. I guasti specifici di CUDA hanno una [pagina dedicata alla risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

**4. Un singolo comando dall'adattatore addestrato a `ollama run`.**
Molte librerie addestrano un modello. Poche di esse si mettono di lato quando si desidera effettivamente utilizzarlo. Backpropagate esporta in GGUF (il formato utilizzato da Ollama) e registra un modello Ollama con un singolo comando. Si passa da "addestramento completato" a "posso chattare con il mio modello ottimizzato" in circa 30 secondi.

## Guida rapida

Dalla riga di comando, con un set di dati di esempio di 5 conversazioni:

```bash
pipx install "backpropagate[standard]"
curl -LO https://raw.githubusercontent.com/mcp-tool-shop-org/backpropagate/main/examples/quickstart.jsonl

backprop train --data quickstart.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 10
backprop generate ./output "What is Python?"      # did it learn anything?
backprop export ./output --format gguf --quantization q4_k_m --ollama --ollama-name my-first-finetune
ollama run my-first-finetune
```

`backprop train` scrive l'adattatore in `./output` (modificarlo con `--output`). In Python, la stessa operazione è:

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")
```

Utilizzare un ambiente virtuale con `pip install "backpropagate[standard]"` per l'API Python; `pipx` installa il comando `backprop` nel proprio ambiente, quindi `import backpropagate` non lo troverà.

**Cosa richiede l'esportazione GGUF.** L'esportazione unisce l'adattatore al modello di base e lo converte con lo script di conversione di llama.cpp. È necessario:

- un checkout del codice sorgente di llama.cpp (`git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp`) più `pip install sentencepiece protobuf` nello stesso ambiente, oppure
- Unsloth con il proprio llama.cpp già compilato.

Con `--ollama`, la quantizzazione `q4_k_m` viene eseguita da `ollama create`, quindi non è necessario compilare nulla. Backpropagate non consente mai a Unsloth di installare pacchetti di sistema per compilare llama.cpp; impostare `BACKPROPAGATE_UNSLOTH_AUTO_INSTALL=1` se lo si desidera. Dettagli: [export](https://mcp-tool-shop-org.github.io/backpropagate/handbook/export/).

Per i propri dati, formattare il file JSONL con un esempio per riga:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca (`instruction` / `output`), OpenAI chat (`messages`) e formati di testo non elaborato funzionano anche — Backpropagate rileva automaticamente il formato.

### Il ciclo: controllare i dati, addestrare, valutare, esportare

```bash
backprop data report my_data.jsonl                     # duplicates, length outliers, format problems
backprop data split my_data.jsonl --heldout-ratio 0.1  # a held-out set the model never trains on
backprop train --data my_data.train.jsonl --steps 200 --output ./run-a
backprop eval <run-id> --heldout my_data.heldout.jsonl # held-out loss + sample generations
backprop eval <run-b> --vs <run-a>                     # did the change help?
backprop export ./run-a --format gguf --ollama --ollama-name my-model
```

La valutazione è progettata per essere priva di giudizi: perdita dei dati non utilizzati più metriche di attività deterministiche (`normalized_exact_match`, `token_f1`, `contains`, `regex`, `pass_rate`). Per utilizzare un LLM come giudice, eseguirlo sull'output di `backprop generate`. Vedere [ricette](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Ottimizzazione delle preferenze (ORPO, SimPO, KTO)

Addestrare sulle preferenze anziché su semplici dimostrazioni. ORPO è privo di riferimenti e a fase singola: integra il segnale di preferenza nel passaggio SFT, quindi non esiste un modello di ricompensa o di riferimento separato e la forma a 3 righe rimane invariata. Passare `--method orpo` (CLI) o `method="orpo"` (Python) e fornire un set di dati di `{prompt, chosen, rejected}` (o solo `{chosen, rejected}`) righe:

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

Il tasso di apprendimento predefinito si abbassa automaticamente a `8e-6` per ORPO (la perdita è più netta rispetto al semplice SFT); regolare `--orpo-beta` (predefinito `0.1`) per ponderare la penalità del rapporto di probabilità. ORPO è solo `mode="lora"`.

**Novità nella versione 1.6: SimPO e KTO.** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) è privo di riferimenti con una ricompensa normalizzata in base alla lunghezza e utilizza gli stessi dati accoppiati `{prompt, chosen, rejected}` di ORPO (`--simpo-beta`, `--simpo-gamma`). `--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) utilizza dati **non accoppiati** `{prompt, completion, label}`: valutazioni positive/negative per ogni esempio, per l'ampia classe di feedback che non sono coppie A/B curate; bilancia automaticamente i pesi di perdita desiderabili/indesiderabili in base ai conteggi delle etichette. Entrambi sono solo `mode="lora"` e rimangono nell'ambito SFT a singola GPU (nessun modello di riferimento separato). Vedere la [guida all'ottimizzazione delle preferenze](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) per sapere quale utilizzare. Per l'RL online (PPO/GRPO) vedere [cosa Backpropagate NON è](https://mcp-tool-shop-org.github.io/backpropagate/handbook/what-backpropagate-is-not-for/).

### SFT di ragionamento-traccia (distillazione R1)

Distillare un modello di ragionamento nel modo più semplice. Passare `--reasoning-trace` (CLI) o `Trainer(..., reasoning_trace=True)` (Python) e fornire tracce che mantengano una catena di pensiero `<think>...</think>` all'interno della risposta dell'assistente: la metà SFT pura della distillazione [DeepSeek-R1](https://arxiv.org/abs/2501.12948), non è richiesto l'RL. Backpropagate mantiene `<think>` nell'obiettivo di addestramento, elimina le tracce vuote/troppo lunghe (filtraggio della lunghezza della traccia) e aumenta il valore predefinito di `max_seq_length` a 8192 per la catena di pensiero più lunga. In modo fondamentale, `<think>` rimane **testo semplice**: nessun token speciale, nessuna modifica delle dimensioni dell'embedding, quindi il GGUF unito viene comunque esportato in Ollama come qualsiasi altro modello ottimizzato. Solo SFT. Vedere la [ricetta di ragionamento-traccia](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) per la forma del set di dati e i token regolabili.

### Apple Silicon (MLX) — anteprima non verificata

> ⚠️ **Anteprima non verificata: non fa parte del set di funzionalità supportate.** Il percorso MLX è stato creato e testato con test unitari, ma **non** è stato verificato con test reali su Apple Silicon (`mlx-lm` è solo per Apple e non può essere eseguito sui sistemi NVIDIA su cui viene sviluppato Backpropagate). Considerare tutto quanto segue come sperimentale, utilizzarlo a proprio rischio e [segnalare anomalie](#reporting-bugs) se lo si esegue su un Mac della serie M.

**Un'API, due percorsi.** CUDA è il backend canonico e verificato; MLX è un secondo percorso che si addestra su un Mac della serie M tramite il toolchain [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) di Apple (memoria unificata, nessuna CUDA). La forma a 3 righe seleziona il percorso in base all'hardware: `backend='auto'` (predefinito) indirizza a CUDA su NVIDIA e a MLX su Apple Silicon, quindi i sistemi CUDA esistenti sono identici a livello di byte:

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

Il percorso MLX è **solo SFT LoRA**: nessun ORPO, nessun FP8, nessun `mode='full'`, nessuna esecuzione multipla (ognuno viene rifiutato con `CONFIG_INVALID_SETTING`; utilizzare `backend='cuda'`/`'auto'` su un sistema NVIDIA per questi). L'adattatore risultante è un semplice safetensors ed esporta in Ollama attraverso lo stesso percorso del percorso CUDA.

> Forzare `--backend mlx` su un host non Apple genera un errore con `CONFIG_INVALID_SETTING`; la mancanza del toolchain `mlx_lm` su un Mac genera `DEP_MLX_UNAVAILABLE`.

Per flussi di lavoro end-to-end più completi (ottimizzazione e pubblicazione su HF Hub, ripresa dopo esaurimento della memoria, SLAO multi-esecuzione su una lunga campagna, ecc.) vedere la [pagina delle ricette della guida](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interfaccia utente Web (opzionale)

Se si preferisce fare clic anziché digitare in Python, installare l'extra dell'interfaccia utente e avviare:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Apri l'URL che visualizza, `http://127.0.0.1:7862/?token=...` (ogni esecuzione crea un nuovo token; la prima esecuzione configura l'interfaccia utente e potrebbe richiedere uno o due minuti). Si tratta di un'interfaccia web locale per la navigazione tra i set di dati, la convalida dei formati e l'assemblaggio visivo di una configurazione di addestramento. L'addestramento vero e proprio viene eseguito tramite `backprop train` (l'addestramento basato sull'interfaccia utente è in programma: il pulsante "Avvia" visualizza attualmente tale nota). Per impostazione predefinita, l'interfaccia utente è disponibile solo in locale. Per renderla accessibile da altri dispositivi, consulta la sezione [Interfaccia utente web](#web-ui) qui sotto per il contratto di sicurezza `--share` + `--auth`.

## Addestramento multi-esecuzione

Se desideri ottimizzare in modo incrementale su più set di dati (ad esempio, se ricevi nuovi dati di addestramento ogni settimana e desideri aggiungerli senza dimenticare ciò che hai appreso in precedenza), la modalità `multi_run` di Backpropagate è quella giusta per te:

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

Questa esegue cinque passaggi di addestramento, unendo l'adattatore tra le esecuzioni in modo da preservare le conoscenze precedenti, incorporando al contempo nuovi esempi. La tecnica si basa su recenti ricerche sull'apprendimento continuo: consulta la sezione [Riferimenti](#references) in fondo a questo file README.

La versione CLI:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Riprendi da un checkpoint

Un addestramento di 5 esecuzioni che si interrompe alla quarta esecuzione può essere ripristinato. Ogni sessione multi-esecuzione scrive l'ID dell'esecuzione nella cronologia e nel file manifest del checkpoint memorizzati su disco, quindi riprendere da dove si era interrotti richiede un solo comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

Il comportamento predefinito di `backprop multi-run` (senza `--resume`) rileva automaticamente una voce in corso nella stessa directory di output e la continua. Per forzare un nuovo avvio, punta a una directory di output nuova.

## Cronologia dell'addestramento

Ogni invocazione di `backprop train` e `backprop multi-run` registra una riga in `<output>/run_history.json`: modello utilizzato, set di dati, iperparametri, stato, perdita finale, cronologia delle perdite. Puoi elencare e ispezionare le esecuzioni passate:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Monitoraggio degli esperimenti

Backpropagate rileva automaticamente i tracker di esperimenti installati (Weights & Biases, TensorBoard, MLflow) e li integra. Se `wandb` è installato e sei connesso, ogni esecuzione registra automaticamente i dati su W&B con un nome di esecuzione che corrisponde all'ID dell'esecuzione memorizzato su disco, in modo da poter cercare in W&B, nei tuoi log e in `run_history.json` utilizzando un unico identificatore.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Annulla l'override con `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` o `Trainer(report_to="none")` per disattivare questa funzione.

## Interfaccia utente web

L'interfaccia web di Reflex è un'opzione: installala con `pipx install "backpropagate[ui]"` e avviala:

```bash
backprop ui --port 7862
```

L'interfaccia utente viene eseguita in locale: apri l'URL che visualizza, `http://127.0.0.1:7862/?token=...`. Senza `--auth`, ogni avvio genera un nuovo token e l'interfaccia utente rifiuta le richieste che non lo contengono. Oggi copre la metà del flusso di lavoro relativa alla **navigazione / convalida / configurazione**: puntala a un set di dati, controlla il formato e le statistiche rilevate automaticamente, scegli un modello e assembla una configurazione di esecuzione. **L'avvio dell'esecuzione viene eseguito dalla CLI** (`backprop train` / `backprop multi-run`); il pulsante "Avvia" nell'interfaccia utente visualizza una nota che indica dove trovarla. L'addestramento basato sull'interfaccia utente è un'estensione pianificata: fino ad allora, l'interfaccia utente sarà il punto di accesso e la CLI sarà il trigger.

Per renderla accessibile da altri dispositivi (altre persone nella tua rete, un URL pubblico, ecc.), devi associare `--share` (o `--host`) a `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` senza `--auth` si chiude con un errore. Il motivo: `--share` pubblica un URL a cui chiunque su Internet può accedere e, senza autenticazione, ciò significa che chiunque può controllare la tua pipeline di addestramento e leggere il tuo token HuggingFace. Non è possibile disattivare questa funzione: se non desideri impostare le credenziali, utilizza invece il port forwarding SSH:

```bash
# On the client:
ssh -L 7862:localhost:7862 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open the URL the server printed (http://127.0.0.1:7862/?token=...) locally
```

Consulta la pagina [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) per la descrizione completa del modello di minaccia.

Le scritture sul file system dall'interfaccia utente sono limitate a una singola directory:

- Predefinito: `~/.backpropagate/ui-outputs`
- Annulla l'override: imposta `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- L'override viene convalidato tramite una lista di esclusione: i percorsi di sistema o di credenziali (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, ecc.) vengono rifiutati.

## Note sulla piattaforma

**Requisiti:** Python 3.10+ · GPU CUDA (8 GB+ di VRAM) · PyTorch 2.0+

Python 3.10 è supportato almeno fino alla versione v1.6; raggiungerà la fine del ciclo di vita upstream nell'ottobre 2026 ed è previsto che venga rimosso nella prima versione successiva. Per le nuove installazioni, preferisci Python 3.11 o 3.12: 3.11 è la versione più testata.

Backpropagate gestisce le peculiarità di runtime dell'addestramento su diverse piattaforme, ma non può risolvere i problemi di installazione. I due più comuni sono:

- **Pacchetto CUDA errato.** PyTorch viene pubblicato con un singolo binario per ogni versione di CUDA. Se scegli quello sbagliato, otterrai silenziosamente PyTorch solo per CPU e l'addestramento sarà incredibilmente lento. Utilizza il selettore di pacchetti all'indirizzo <https://pytorch.org/get-started/locally/> per il tuo driver. Esegui `nvidia-smi` per visualizzare la versione del tuo driver / CUDA.
- **Windows + esportazione GGUF.** L'opzione `[export]` compila `llama-cpp-python` dal codice sorgente, il che richiede Visual Studio Build Tools (componente C++) e CMake.

**macOS:** la funzionalità CUDA non è supportata (nessuna CUDA): un'esecuzione `trainer.train()` con CUDA genera `DEP_GPU_NOT_AVAILABLE` e puoi eseguire l'adattatore addestrato su un Mac tramite Ollama. Un'implementazione **sperimentale e non verificata** MLX (`--backend mlx`, `pip install 'backpropagate[mlx]'`) addestra un adattatore LoRA in modo nativo su Apple Silicon tramite `mlx_lm.lora`: solo LoRA SFT e **non verificata su hardware reale** (consulta la sezione [Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview)). Per il percorso CUDA o per ORPO / fine-tuning completo / FP8 / addestramento multi-esecuzione, utilizza una macchina Linux o Windows con CUDA.

Consulta la [pagina della guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) per la guida completa alla risoluzione dei problemi di installazione e la [pagina dedicata alla risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) per i problemi relativi al driver / VRAM / xformers / bf16-vs-fp16.

## CLI

Ogni API Python ha un'equivalente CLI:

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

Riferimento completo nella [pagina della guida CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), oppure esegui `backprop <subcommand> --help`.

## Configurazione

Ogni impostazione può essere sovrascritta con una variabile d'ambiente utilizzando il prefisso `BACKPROPAGATE_`:

| Variabile | Predefinito | Note |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | automatico | Forza l'output in formato JSON o i log della console |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modello predefinito |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Tasso di apprendimento |
| `BACKPROPAGATE_LORA__R` | `256` | Rango LoRA (valore predefinito per la versione 1.3; inserire `--lora-preset=fast` per il valore predefinito della versione 1.2.x, ovvero 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sandbox del file system dell'interfaccia utente |

Le chiavi nidificate utilizzano il doppio trattino basso (`MODEL__NAME`, non `MODEL_NAME`). Il riferimento completo è disponibile nella [pagina del manuale delle variabili d'ambiente](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Preset dei modelli

| Preset | VRAM | Licenza | Note |
|---|---|---|---|
| Qwen-3.5-4B | ~8 GB | Apache 2.0 | Valore predefinito consigliato per modelli inferiori a 5B. Migliore qualità con queste dimensioni. |
| Phi-4-mini-3.8B | ~8 GB | MIT | Ottimo per ragionamento, matematica e codice. Licenza restrittiva ma chiara. |
| SmolLM3-3B | ~6 GB | Apache 2.0 | Ricetta completamente aperta. Contesto nativo di 64K. |
| Qwen 2.5 7B | ~12 GB | Apache 2.0 | Valore predefinito esistente. Migliore qualità tra i preset 7B legacy. |
| Qwen 2.5 3B | ~8 GB | Qwen-Research | ⚠ licenza di ricerca: consultare i termini di licenza di Qwen prima dell'uso commerciale. |
| Llama 3.2 3B | ~8 GB | Llama Community | Solida alternativa a Qwen 3B con alcune limitazioni. |
| Llama 3.2 1B | ~6 GB | Llama Community | Per esperimenti rapidi su schede di piccole dimensioni. |
| Mistral 7B | ~12 GB | Apache 2.0 | Comparabile a Qwen 7B, con un modello di chat diverso. |
| Llama-3.1-8B | ~7-8 GB (QLoRA) | Llama-3.1-Community | 8B QLoRA, contesto nativo di 128K (la clausola >700M-MAU richiede una licenza Meta separata). |
| **Qwen2.5-14B** | 25 GiB di picco a 4096 ctx (QLoRA) | Apache 2.0 | **Il modello da 32 GB per l'uso quotidiano.** Rango/alfa 32, AdamW a 8 bit. I pesi a 4 bit da soli occupano circa 8,5 GB; per una finestra completa di 4096 token è necessario il resto. |
| Mistral-Small-24B | 26,5 GiB di picco a 4096 ctx (QLoRA) | Apache 2.0 | 24B QLoRA su una scheda da 32 GB. I pesi a 4 bit da soli occupano circa 18 GB. |
| **Qwen2.5-32B** | 28,8 GiB di picco a 2048 ctx (QLoRA) | Apache 2.0 | **Il massimo che si può ottenere con una scheda da 32 GB.** Si adatta a malapena a `max_len 2048` con AdamW a 8 bit. |

Altri modelli spesso funzionano; le righe sopra sono i preset curati: il livello 14B-32B è ottimizzato con QLoRA per una scheda da 32 GB (il limite misurato). Inserire `--lora-preset=quality` (valore predefinito) per i target con rango 256 / tutti lineari secondo Biderman 2024 + Thinking Machines 2025, oppure `--lora-preset=fast` per il target legacy con rango 16 / q+v se è necessario il footprint della versione 1.2.x.

## Risoluzione dei problemi

Un breve elenco dei fallimenti più comuni durante la prima esecuzione. L'indice completo è disponibile nella [pagina del manuale per la risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Per un'analisi approfondita di driver, VRAM e precisione mista, consultare la [pagina per la risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Sintomo | Codice di errore | Soluzione |
|---|---|---|
| La GPU esaurisce la memoria durante l'addestramento | `RUNTIME_GPU_OOM` | Automatico: Backpropagate dimezza la dimensione del batch e riprova fino a 3 volte. Per disattivare: `Trainer(oom_recovery=False)`. Per forzare una dimensione inferiore: `--batch-size 1`. |
| HuggingFace restituisce 401 / "modello non trovato" | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` e riprova. Per errori di battitura, copiare l'ID esatto da <https://huggingface.co/models>. |
| `register_with_ollama` connessione rifiutata | `DEP_OLLAMA_REGISTRATION_FAILED` | Avviare il daemon: `ollama serve`. Installare da <https://ollama.com>. Riprova. |
| Disco pieno durante il salvataggio del checkpoint | `STATE_CHECKPOINT_INVALID` | Le scritture atomiche lasciano una directory `.partial` in caso di errore: è sicuro eliminarla. Il checkpoint precedente è intatto. |
| L'addestramento è stato interrotto a causa del surriscaldamento della GPU | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automatico: Backpropagate mette in pausa quando viene superata la soglia di temperatura e riprende quando la GPU si raffredda. Migliorare il flusso d'aria se il problema persiste. |
| `backprop ui --share` rifiutato | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Inserire `--auth user:password` oppure utilizzare il port forwarding SSH (vedere [Interfaccia utente web](#interfaccia-utente-web)). |
| Esportazione GGUF non riuscita al primo tentativo | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; su Windows è necessario anche Visual C++ Build Tools + CMake. |

## Segnalazione di bug

Quando qualcosa non va, Backpropagate stampa una riga all'avvio simile a `run_started run_id=<uuid>` e associa lo stesso ID a ogni riga del log, a ogni checkpoint e a ogni voce di Weights & Biases. **Includere `run_id` in qualsiasi segnalazione di bug**: consente a un manutentore di correlare tutto per quella specifica esecuzione.

Una buona segnalazione di bug include:

1. **Il `run_id`**: l'UUID stampato all'avvio. Un UUID consente a un manutentore di correlare ogni riga del log, ogni checkpoint e ogni voce di Weights & Biases per quella specifica esecuzione.
2. **Il codice di errore**: la riga `[CODE_NAME]: message` in stderr. Consultare [codici di errore](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) per il catalogo dei codici stabili.
3. **Il traceback ridotto.** Stderr viene automaticamente ridotto in modalità non verbose (i token Bearer, `sk-*`, `hf_*`, le chiavi AWS, le coppie `password=` / `token=` / `api_key=` vengono eliminate): è sicuro incollarlo. Per il traceback completo non ridotto, rieseguire con `BACKPROPAGATE_DEBUG=1` (o `--verbose`); rivedere prima di pubblicare.
4. **L'output `backprop info`.** Un comando stampa Python / PyTorch / CUDA / modello GPU / VRAM / sistema operativo / extra installati: tutto ciò di cui il manutentore ha bisogno per individuare una regressione specifica della piattaforma.

Il [modello per la segnalazione di bug](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) richiede esplicitamente informazioni su ciascuno di questi aspetti, in modo da velocizzare il processo di analisi e risoluzione. Domande, idee o discussioni del tipo "è questo il comportamento previsto?" devono essere pubblicate in [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions). I problemi di sicurezza devono essere segnalati in privato tramite il modulo [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) — consultare [SECURITY.md](SECURITY.md) per la politica e i tempi di risposta.

## Privacy

Tutti gli addestramenti avvengono localmente sulla tua GPU. Backpropagate non effettua richieste di rete, ad eccezione del download dei modelli da HuggingFace (che viene avviato dall'utente). Nessuna telemetria, nessuna dipendenza dal cloud.

## Riferimenti

Le impostazioni predefinite di Backpropagate e la modalità di addestramento multi-esecuzione si basano su ricerche recenti. Se sei interessato alle tecniche sottostanti:

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — il documento fondamentale che introduce LoRA, che è il metodo utilizzato da Backpropagate per addestrare gli adattatori in modo efficiente.
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidenze empiriche che LoRA con rango 256 e obiettivi completamente lineari raggiunge una qualità di fine-tuning paragonabile a quella del fine-tuning completo nella maggior parte delle attività post-addestramento, con il 67% del carico computazionale. Questo determina la configurazione predefinita di LoRA v1.3 di Backpropagate.
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — il seguito pratico che identifica la correzione del fattore di apprendimento (10 volte) rispetto al fine-tuning completo, necessaria a ranghi LoRA elevati.
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la caratterizzazione originale del motivo per cui le reti neurali "dimenticano" gli addestramenti precedenti quando si esegue il fine-tuning su nuovi dati (EWC — Elastic Weight Consolidation).
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, un approccio precedente all'utilizzo di LoRA per l'apprendimento continuo, limitando i nuovi adattatori a sottospazi ortogonali.
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — una tecnica fondamentale per unire più modelli con fine-tuning senza interferenze.
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — l'algoritmo specifico che il merger multi-esecuzione di Backpropagate implementa. Un preprint di dicembre 2025; Backpropagate è il primo adottante noto di questo documento.

## Licenza

MIT — consultare [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
