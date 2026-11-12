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

# Ottimizza un modello da 32B QLoRA — o un modello da 7B end-to-end — su una singola GPU. Caricalo su Ollama

Esegui la retropropagazione per ottimizzare modelli linguistici di grandi dimensioni su una singola GPU, dimensionata in base alla scheda che hai effettivamente. Tre righe di Python QLoRA per un modello da 7B–32B su una singola scheda consumer da 32 GB (RTX 5090). Un flag, `--full-ft-offload`, ottimizza completamente un modello di classe 7B mantenendo i suoi pesi e gradienti nella RAM dell'host (Linux o WSL2; lento, e misurato di seguito). Un comando aggiuntivo esporta su Ollama, quindi `ollama run` la tua ottimizzazione. Si riduce fino a 16 GB. Funziona perfettamente su Windows. Preferisci un browser a Python? `backprop ui` fa tutto questo senza codice ([fai il tour](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/)).

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

Questo è tutto. Non c'è un file di configurazione YAML. Non c'è alcuna procedura complessa `accelerate launch`. Non c'è un tutorial separato del tipo "ora convertilo in GGUF". Se hai una GPU CUDA e un file JSONL con i tuoi dati di addestramento, sei a tre righe da un'ottimizzazione funzionante.

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

Preferisci Docker? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` funziona anche. Sono disponibili immagini sia per `linux/amd64` che per `linux/arm64`, quindi gli utenti di Apple Silicon e ARM Linux ottengono un'immagine nativa. Un esempio canonico `compose.yaml` per "interfaccia utente in un container" si trova nella directory principale del repository: inserisci `user:password` in un `ui-auth.txt` accanto, esegui `docker compose up` e accedi a `http://127.0.0.1:7860` (il primo avvio crea l'interfaccia utente, il che richiede un minuto o due). La cronologia delle esecuzioni viene salvata in `~/.backpropagate`.

## Dove si colloca Backpropagate nello spazio delle librerie

Esistono diverse buone librerie per l'ottimizzazione di LLM. Ognuna di esse è ottima per cose diverse:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — se ti piacciono le configurazioni YAML e desideri una community di ricette da cui copiare
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — se desideri DPO/PPO/RLHF e un'interfaccia utente web
- **[Unsloth](https://github.com/unslothai/unsloth)** — se hai bisogno dell'addestramento più veloce possibile e utilizzi una famiglia di modelli supportata
- **[torchtune](https://github.com/pytorch/torchtune)** — se desideri le ricette native PyTorch di Meta che puoi modificare

Backpropagate è l'opzione mancante: **un'API Python di 3 righe per gli utenti singoli su una singola GPU consumer che desiderano addestrare un adattatore e caricarlo.** Nessun YAML, nessuna RL online (PPO/GRPO), nessun nodo multiplo. Esiste un'interfaccia utente del browser per lo stesso ciclo se preferisci non scrivere codice. Solo il ciclo di cui tutti hanno realmente bisogno e il passaggio di esportazione che crea problemi.

Se hai provato una delle librerie di cui sopra e hai avuto problemi con la procedura di configurazione dei file, o hai riscontrato un limite della famiglia di modelli, o hai desiderato impostazioni predefinite per Windows, Backpropagate è la soluzione.

## Cosa puoi ottimizzare su una singola GPU

Backpropagate dimensiona l'esecuzione in base alla tua scheda. Questi sono numeri **misurati** su una RTX 5090 da 32 GB: le QLoRA righe del 2026-10-03 (dettagli: [`docs/receipts/2026-10-03-presets/`](docs/receipts/2026-10-03-presets/)), le righe di ottimizzazione completa del 2026-09-30 (dettagli: [`docs/receipts/2026-09-30-offload/`](docs/receipts/2026-09-30-offload/)). I picchi QLoRA si verificano alla finestra di contesto completa delle impostazioni predefinite con batch 1, che è il caso peggiore per tali impostazioni; esempi più brevi utilizzano meno risorse.

| Modello | Metodo | Misurato su una scheda da 32 GB |
|---|---|---|
| **14B** (Qwen2.5-14B) | QLoRA | **18,7 GiB** picco a 4096 di contesto (20,0 GiB riservati). |
| 24B (Mistral-Small-24B) | QLoRA | 22,8 GiB picco a 4096 di contesto (24,2 GiB riservati). |
| **32B** (Qwen2.5-32B) | QLoRA | **Si adatta:** 26,0 GiB picco a 2048 di contesto (27,2 GiB riservati, circa 4 GiB di spazio libero). |
| 3B | `mode="full"` (ottimizzazione completa vera e propria, sulla GPU) | **22,0 GiB** picco (a livello di sistema), 0,30 s/passo con batch 4, 512 di contesto. 7,5 GiB di questo è lo stato dell'ottimizzatore memorizzato nella cache, che può essere scaricato nella RAM dell'host su una scheda più piccola (non testato). |
| **7B (classe)** (Qwen2.5-7B, 7,6B di parametri) | `mode="full" --full-ft-offload` | **Addestra:** 5,3 GiB VRAM, **30,8 GiB RAM dell'host** (32,2 GiB durante il salvataggio), **14,7 s/passo**. Solo Linux o WSL2. |

Non misurato in quella sessione: 7B QLoRA, Llama-3.1-8B (repository chiuso, nessun token sulla macchina di test) e ottimizzazione completa solo su GPU sopra i 3B. Le cifre per questi elementi, presenti altrove nella documentazione, sono stime.

Due cose per cui la maggior parte delle librerie per singola GPU ti indirizzano altrove, **ottimizzazione 24–32B QLoRA** e **ottimizzazione completa su singola scheda per modelli di classe 7B**, Backpropagate le esegue su una singola scheda consumer, quindi esporta il risultato direttamente su Ollama.

**L'ottimizzazione completa ha due percorsi.** Senza scaricamento, il modello, i suoi gradienti e lo stato dell'ottimizzatore si trovano tutti sulla GPU. La libreria limita la dimensione del modello in base alla VRAM rilevata (**16 GB → 4B, 24 GB → 5B, 32 GB → 6B**); questi limiti derivano da calcoli di memoria e vengono misurati solo fino a 3B. Sovrascrivi con `--full-ft-ceiling-billions`.

`--full-ft-offload` mantiene pesi e gradienti nella RAM dell'host e li trasmette alla GPU (FSDP2 scaricamento CPU). Ecco cosa comporta, misurato:

- **RAM dell'host:** il controllo di adattamento richiede circa 3,7 GiB per miliardo di parametri più 10 GiB, il che è prudente (39 GiB a 7,6B rispetto ai 32,2 GiB misurati). L'esecuzione viene rifiutata all'inizio, con i numeri, se la macchina non può contenerla. Un modello da 7,6B non si adatta sotto un limite di memoria di 28 GB WSL2; circa 5B è il limite pratico.
- **Velocità:** 14,7 s/passo a 7,6B (batch 1) e 5,1 s/passo a 3B (batch 4), rispetto a circa 0,63 s/passo per 3B sulla GPU con batch 4. Utilizzalo solo quando il modello non si adatta senza. È prevista una versione più veloce.
- **Ottimizzatore:** Adafactor, non AdamW. I pesi rimangono in bf16 e ogni aggiornamento viene riscritto con arrotondamento stocastico; non c'è una copia fp32.
- **Qualità:** in un'esecuzione da 3B (150 passaggi, perdita su dati non utilizzati, un seme), ha raggiunto circa l'85% del miglioramento ottenuto dall'ordinaria ottimizzazione completa (2,45 → 1,93 rispetto a 2,45 → 1,84). Un seme non è un benchmark.
- **Ambito:** semplice ottimizzazione supervisionata. Nessun impacchettamento, nessuna mascheratura solo delle risposte, nessun checkpoint intermedio, nessuna ripresa. Solo Linux o WSL2 (FSDP2 richiede NCCL); su Windows nativo, si interrompe con `DEP_FSDP_UNAVAILABLE`.
- **Non ancora testato:** esecuzioni lunghe, accumulo di gradienti superiore a 1 e una macchina fisica da 64 GB (la macchina di test aveva più RAM, con un limite di 60 GiB applicato dal test).

Un modello che non si adatta esce con `RUNTIME_FULL_FT_MODEL_TOO_LARGE` e indica la soluzione. Consulta [la pagina completa del manuale sull'ottimizzazione](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/).

### Si riduce a 16 GB

L'intervallo di 16 GB (RTX 4080 / 5080 / 4070 Ti Super) è ancora di prima classe: 7B QLoRA (la dimensione dell'adattatore viene scelta per adattarsi: rango 64 su una scheda da 16 GB, dove un rango di 256 richiede circa 17 GB) e un vero e proprio fine-tuning completo di un modello da ~3B (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B/1B) tramite `mode="full"`, (22,0 GiB misurato su una scheda da 32 GB a 3B, di cui 7,5 GiB è lo stato dell'ottimizzatore memorizzato nella cache che può essere scaricato nella RAM dell'host; non è stato ancora testato se questo funziona in modo accettabile su una scheda da 16 GB). Con `--full-ft-offload`, la GPU contiene molto meno: con la VRAM limitata sulla scheda di test, un modello da 3B addestrato con un limite di 6 GiB e modelli da 4B e 7,6B con un limite di 8 GiB. Questi sono limiti simulati su una scheda da 32 GB, non esecuzioni su hardware reale da 8 GB. Lo stesso codice seleziona la dimensione del batch e il limite che si adattano alla scheda rilevata.

La quantizzazione a 2 bit (AQLM / QuIP#) rimane **fuori dall'ambito**; una base a 2 bit non può essere unita in modo pulito ai pesi a precisione completa, il che interrompe il contratto di esportazione adattatore unificabile → GGUF → Ollama (che è lo scopo principale della pipeline). Invece, Backpropagate offre le funzionalità per aumentare il margine di manovra: QLoRA, `mode="full"`, `--full-ft-offload` e il percorso di calcolo FP8 (`--fp8`, Blackwell/Hopper), che rimangono unificabili ed esportabili.

## A cosa NON serve Backpropagate

Se il tuo caso d'uso è tra quelli elencati di seguito, otterrai risultati migliori con una libreria diversa: Backpropagate non è la scelta giusta e cercare di farlo funzionare costerebbe più che semplicemente utilizzare lo strumento appropriato. Leggere questa sezione prima di iniziare consente di evitare il ciclo di installazione e riavvio:

- **Fine-tuning completo di modelli da 13B+** — Backpropagate esegue il fine-tuning completo fino a circa 6B su una GPU da 32 GB e un modello di classe 7B con `--full-ft-offload` (vedere [l'intervallo](#cosa-puoi-ottimizzare-su-una-singola-gpu)). Un fine-tuning completo di un modello da 13B+ richiede FSDP multi-GPU o una scheda più grande. Prima di investire in tale potenza di calcolo, valuta attentamente i pro e i contro. [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/) riporta che LoRA corrisponde al fine-tuning completo quando viene applicato a ogni livello e il set di dati si adatta alla capacità dell'adattatore, con circa due terzi della potenza di calcolo per passaggio. [Biderman et al. 2024](https://arxiv.org/abs/2405.09673) hanno scoperto che nelle impostazioni standard a basso rango LoRA ottiene risultati significativamente inferiori rispetto al fine-tuning completo su codice e matematica, pur dimenticando meno. Per l'esecuzione di istruzioni, la personalizzazione e lo stile su set di dati modesti, QLoRA fino a 32B è solitamente l'uso migliore di una singola scheda.
- **RL online — PPO / GRPO / RLVR** — Backpropagate esegue SFT a singolo stadio più fine-tuning delle preferenze senza riferimento (ORPO in v1.5; SimPO + KTO in v1.6). Non esegue l'apprendimento per rinforzo online: PPO, GRPO o RLVR, che richiedono un modello di ricompensa o un ciclo di generazione e valutazione in aggiunta al passaggio di addestramento. Per questi, utilizzare direttamente TRL o LLaMA-Factory. (Il fine-tuning delle preferenze senza riferimento si adatta all'intervallo a singolo stadio perché non esiste un modello di riferimento separato da mantenere in memoria; vedere la nota ORPO in [Guida rapida](#guida-rapida).)
- **Addestramento multi-nodo** — solo una GPU su una singola macchina. Il multi-GPU su una singola macchina funziona (tramite `accelerate launch`) ma non è supportato ufficialmente.
- **Addestramento macOS sulla piattaforma CUDA** — Apple Silicon non ha CUDA, quindi il percorso CUDA viene eseguito su una macchina Linux o Windows con una GPU NVIDIA. È comunque possibile eseguire il modello addestrato su un Mac tramite Ollama. Un percorso MLX **sperimentale e non verificato** (`--backend mlx`) addestra un adattatore LoRA in modo nativo su Apple Silicon; vedere [Apple Silicon (MLX)](#apple-silicon-mlx--anteprima-non-verificata). È solo per LoRA-SFT e **non è stato verificato su hardware reale** (nessun supporto), quindi per qualsiasi cosa oltre a un SFT LoRA (ORPO, fine-tuning completo, FP8, esecuzioni multiple) è consigliabile utilizzare il percorso CUDA.
- **Qualsiasi cosa al di fuori delle famiglie di modelli testate** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Altri modelli spesso funzionano, ma non sono fissati nei test CI.

Se hai bisogno di una di queste cose, utilizza una delle librerie elencate sopra. Sono più adatte a questo scopo.

## Cosa ti offre Backpropagate

Quattro cose, in una singola installazione:

**1. Una vera API a 3 righe che funziona senza un file di configurazione.**
Lo snippet all'inizio di questo README viene eseguito dall'inizio alla fine. Nessun `accelerate config`, nessun YAML, nessuna sovrascrittura Hydra. Basta `Trainer(model).train(data)` e hai un fine-tuning.

**2. Windows che funziona davvero.**
La maggior parte delle librerie ML trattano Windows come un ripensamento. Backpropagate è sviluppato e testato su Windows 11 con schede della serie RTX 50. La libreria gestisce le peculiarità del runtime per te: sa come pre-tokenizzare i dati in modo che l'elaborazione multiprocessing di Windows non si blocchi, disabilita automaticamente xformers sulle schede RTX 40/50 dove causerebbe problemi e seleziona le impostazioni del dataloader che non causano errori. Non devi sapere nulla di tutto questo. Funziona semplicemente.

**3. Progettato per esecuzioni non presidiate.**
L'addestramento richiede ore. Non vuoi doverlo monitorare costantemente. Backpropagate è progettato per essere lasciato in esecuzione:

- Se esaurisci la memoria della GPU, dimezza automaticamente la dimensione del batch e riprova, fino a tre volte. Nessuna regolazione manuale.
- Se la tua GPU diventa troppo calda, si mette in pausa fino a quando le cose non si raffreddano e quindi continua.
- Ogni checkpoint viene scritto in modo atomico: se il tuo laptop si blocca durante il salvataggio, il checkpoint precedente e valido rimane intatto.
- Ogni esecuzione di addestramento riceve un ID univoco che viene stampato su ogni riga del log, su ogni checkpoint e su ogni voce di Weights & Biases. Se qualcosa va storto, un singolo ID consente a un manutentore di correlare tutto.
- Gli errori sono accompagnati da codici stabili (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, ecc.) in modo da poter cercare nei log e nella [guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) per trovare la soluzione. I guasti specifici di CUDA hanno una [pagina dedicata alla risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

**4. Un singolo comando, dal modello addestrato all'adattatore `ollama run`.**
Molte librerie addestrano un modello. Poche di esse ti permettono di utilizzarlo facilmente. Backpropagate esporta in GGUF (il formato utilizzato da Ollama) e registra un modello Ollama con un singolo comando. Passi da "addestramento completato" a "posso chattare con il mio modello fine-tuned" in circa 30 secondi.

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

`backprop train` scrive l'adattatore per `./output` (modificalo con `--output`). In Python, la stessa cosa è:

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")
```

Utilizza un ambiente virtuale con `pip install "backpropagate[standard]"` per l'API Python; `pipx` installa il comando `backprop` nel suo ambiente, quindi `import backpropagate` non lo troverà.

**Di cosa ha bisogno l'esportazione GGUF.** L'esportazione unisce l'adattatore al modello di base e lo converte con lo script di conversione di llama.cpp. È necessario:

- un checkout del codice sorgente di llama.cpp (`git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp`) più `pip install sentencepiece protobuf` nello stesso ambiente, oppure
- Unsloth con il suo llama.cpp già compilato.

Con `--ollama`, la quantizzazione `q4_k_m` viene eseguita da `ollama create`, quindi non è necessario compilare nulla. Backpropagate non consente mai a Unsloth di installare pacchetti di sistema per compilare llama.cpp per te; imposta `BACKPROPAGATE_UNSLOTH_AUTO_INSTALL=1` se lo desideri. Dettagli: [export](https://mcp-tool-shop-org.github.io/backpropagate/handbook/export/).

Per i tuoi dati, formatta il tuo JSONL con un esempio per riga:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca (`instruction` / `output`), OpenAI chat (`messages`) e formati di testo puro funzionano anche — Backpropagate rileva automaticamente il formato.

### Il ciclo: controlla i dati, addestra, valuta, esporta

```bash
backprop data report my_data.jsonl                     # duplicates, length outliers, format problems
backprop data split my_data.jsonl --heldout-ratio 0.1  # a held-out set the model never trains on
backprop train --data my_data.train.jsonl --steps 200 --output ./run-a
backprop eval <run-id> --heldout my_data.heldout.jsonl # held-out loss + sample generations
backprop eval <run-b> --vs <run-a>                     # did the change help?
backprop export ./run-a --format gguf --ollama --ollama-name my-model
```

La valutazione è progettata per essere priva di giudizi: perdita dei dati di test più metriche di attività deterministiche (`normalized_exact_match`, `token_f1`, `contains`, `regex`, `pass_rate`). Per utilizzare un LLM come giudice, eseguilo sull'output di `backprop generate`. Vedi [ricette](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Ottimizzazione delle preferenze (ORPO, SimPO, KTO)

Addestra sulle preferenze invece che su semplici dimostrazioni. ORPO è privo di riferimenti e a fase singola: integra il segnale di preferenza nel passaggio SFT, quindi non c'è un modello di ricompensa o di riferimento separato e la forma a 3 righe rimane invariata. Passa `--method orpo` (CLI) o `method="orpo"` (Python) e fornisci un set di dati di `{prompt, chosen, rejected}` (o solo `{chosen, rejected}`) righe:

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

Il tasso di apprendimento predefinito si abbassa automaticamente a `8e-6` per ORPO (la perdita è più netta rispetto al semplice SFT); regola `--orpo-beta` (predefinito `0.1`) per ponderare la penalità del rapporto di probabilità. ORPO è solo `mode="lora"`.

**Novità in v1.6 — SimPO e KTO.** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) è privo di riferimenti, con una ricompensa normalizzata in base alla lunghezza e utilizza gli stessi dati accoppiati `{prompt, chosen, rejected}` di ORPO (`--simpo-beta`, `--simpo-gamma`). `--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) utilizza dati **non accoppiati** `{prompt, completion, label}`: valutazioni positive/negative per ogni esempio, per l'ampia classe di feedback che non sono coppie A/B curate; bilancia automaticamente i pesi di perdita desiderabili/indesiderabili in base al conteggio delle etichette. Entrambi sono solo `mode="lora"` e rimangono nell'ambito SFT a singola GPU (nessun modello di riferimento separato). Vedi l' [handbook sull'ottimizzazione delle preferenze](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) per capire quale utilizzare. Per l'RL online (PPO/GRPO), vedi [Cosa Backpropagate NON è](#a-cosa-non-serve-backpropagate).

### SFT basato sulla traccia del ragionamento (distillazione R1)

Distilla un modello di ragionamento nel modo più semplice. Passa `--reasoning-trace` (CLI) o `Trainer(..., reasoning_trace=True)` (Python) e fornisci tracce che mantengono una catena di pensiero `<think>...</think>` all'interno del turno dell'assistente: la metà SFT pura della distillazione [DeepSeek-R1](https://arxiv.org/abs/2501.12948), senza bisogno di RL. Backpropagate mantiene `<think>` nell'obiettivo di addestramento, elimina le tracce vuote o troppo lunghe (filtraggio della lunghezza della traccia) e aumenta il valore predefinito `max_seq_length` a 8192 per la CoT più lunga. In modo cruciale, `<think>` rimane **solo testo**: nessun token speciale, nessuna ridimensionamento dell'embedding, quindi l'adattatore unificato GGUF può comunque essere esportato in Ollama come qualsiasi altro modello ottimizzato. Solo SFT. Vedi la [ricetta sulla traccia del ragionamento](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) per la forma del set di dati e i token regolabili.

### Apple Silicon (MLX) — anteprima non verificata

> ⚠️ **Anteprima non verificata: non fa parte del set di funzionalità supportate.** Il rail MLX è stato costruito e testato con test unitari, ma **non** è stato verificato su Apple Silicon reale (`mlx-lm` è solo per Apple e non può essere eseguito sui sistemi NVIDIA su cui Backpropagate è in fase di sviluppo). Considera tutto quanto segue come sperimentale, utilizzalo a tuo rischio e pericolo e [segnala eventuali anomalie](#segnalazione-di-bug) se lo esegui su un Mac della serie M.

**Un'API, due rail.** CUDA è il backend canonico e verificato; MLX è un secondo rail che si addestra su un Mac della serie M tramite il toolchain [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) di Apple (memoria unificata, niente CUDA). La forma a 3 righe seleziona il rail in base all'hardware: `backend='auto'` (il valore predefinito) indirizza a CUDA su NVIDIA e a MLX su Apple Silicon, quindi i sistemi CUDA esistenti sono identici a livello di byte:

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

Il rail MLX è **solo SFT LoRA**: niente ORPO, niente FP8, niente `mode='full'`, niente esecuzioni multiple (ognuna viene rifiutata con `CONFIG_INVALID_SETTING`; utilizza `backend='cuda'`/`'auto'` su un sistema NVIDIA per queste). L'adattatore risultante è un semplice safetensors ed esporta in Ollama attraverso lo stesso percorso del rail CUDA.

> Forzare `--backend mlx` su un host non Apple genera un errore con `CONFIG_INVALID_SETTING`; la mancanza del toolchain `mlx_lm` su un Mac genera `DEP_MLX_UNAVAILABLE`.

Per flussi di lavoro end-to-end più completi (ottimizzazione e caricamento su HF Hub, ripresa dopo esaurimento della memoria, SLAO multi-esecuzione su una lunga campagna, ecc.), vedi la [pagina delle ricette dell'handbook](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interfaccia utente web (opzionale)

Se preferisci fare clic invece di digitare Python, installa l'extra dell'interfaccia utente e avvia:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Apri l'URL che visualizza, `http://127.0.0.1:7862/?token=...` (ogni avvio crea un nuovo token; il primo avvio crea l'interfaccia utente e può richiedere uno o due minuti). È un'interfaccia web locale per l'addestramento: avvia un'esecuzione, una serie di esecuzioni multiple o un'esportazione, osservala in tempo reale (passaggi, perdita, tempo rimanente, temperatura e memoria della GPU) e interrompila con un checkpoint salvato. Ogni attività viene eseguita nel proprio processo, uno alla volta, e un aggiornamento della pagina riprende un'attività in esecuzione. La pagina Dataset mostra cosa contiene un file, salva una copia pulita (rimuove ripetizioni ed esempi vuoti) e la passa al modulo di addestramento. Le esecuzioni passate e i modelli nella cache di Hugging Face hanno le loro pagine e ogni impostazione ha una "i" che la spiega. Il [tour dell'interfaccia utente web](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/) mostra ogni pagina. L'interfaccia utente è locale per impostazione predefinita. Per esporla ad altri dispositivi, vedi [Interfaccia utente web](#interfaccia-utente-web) di seguito per il contratto di sicurezza `--share` + `--auth`.

## Addestramento multi-esecuzione

Se desideri ottimizzare in modo incrementale su più set di dati (ad esempio, se ricevi nuovi dati di addestramento ogni settimana e desideri aggiungerli senza dimenticare ciò che hai imparato prima), la modalità `multi_run` di Backpropagate è ciò che fa per te:

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

Questo esegue cinque passaggi di addestramento, unendo l'adattatore tra un passaggio e l'altro in modo da preservare le conoscenze acquisite in precedenza, incorporando al contempo nuovi esempi. La tecnica si basa su recenti ricerche sull'apprendimento continuo; vedere [Riferimenti](#riferimenti) in fondo a questo file README.

Versione CLI:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Riprendi da un checkpoint

Un addestramento di 5 passaggi che si interrompe al passaggio 4 può essere ripreso. Ogni sessione con più passaggi scrive l'ID del passaggio nella cronologia e nel file manifest del checkpoint, quindi riprendere da dove ci si era interrotti richiede un solo comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

Il comportamento predefinito di `backprop multi-run` (senza `--resume`) rileva automaticamente una voce in corso nella stessa directory di output e la continua. Per forzare un nuovo inizio, puntare a una directory di output nuova.

## Cronologia dell'addestramento

Ogni invocazione di `backprop train` e `backprop multi-run` registra una riga in `<output>/run_history.json`: modello utilizzato, set di dati, iperparametri, stato, perdita finale, cronologia delle perdite. È possibile elencare e controllare i passaggi precedenti:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Monitoraggio degli esperimenti

Backpropagate rileva automaticamente gli strumenti di monitoraggio degli esperimenti installati (Weights & Biases, TensorBoard, MLflow) e li integra. Se `wandb` è installato e si è effettuato l'accesso, ogni passaggio registra automaticamente i dati su W&B con un nome di passaggio che corrisponde all'ID del passaggio sul disco, in modo da poter eseguire ricerche su W&B, nei log e in `run_history.json` utilizzando un unico identificatore.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

È possibile sovrascrivere con `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` o `Trainer(report_to="none")` per disattivare questa funzione.

## Interfaccia utente web

L'interfaccia web di Reflex è attivabile; installarla con `pipx install "backpropagate[ui]"` e avviarla:

```bash
backprop ui --port 7862
```

L'interfaccia utente viene eseguita in locale: aprire l'URL che visualizza, `http://127.0.0.1:7862/?token=...`. Senza `--auth`, ogni avvio genera un nuovo token e l'interfaccia utente rifiuta le richieste che non lo contengono. Da qui è possibile visualizzare e pulire un set di dati, addestrare (un singolo passaggio o più passaggi), osservare l'addestramento in tempo reale, interromperlo con un checkpoint salvato ed esportare il risultato. Ogni attività viene eseguita nel proprio processo, uno alla volta, e la chiusura dell'interfaccia utente la interrompe. Il [tour dell'interfaccia utente web](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/) illustra ogni pagina con schermate.

Per renderla accessibile ad altri dispositivi (altre persone sulla rete, un URL pubblico, ecc.), è necessario associare `--share` (o `--host`) a `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` senza `--auth` si chiude con un errore. Il motivo: `--share` pubblica un URL a cui chiunque su Internet può accedere e, senza autenticazione, ciò significa che chiunque può controllare la pipeline di addestramento e leggere il token HuggingFace. Non è possibile disattivare questa funzione: se non si desidera impostare le credenziali, utilizzare invece il port forwarding SSH:

```bash
# On the client:
ssh -L 7862:localhost:7862 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open the URL the server printed (http://127.0.0.1:7862/?token=...) locally
```

Vedere [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) per la descrizione completa del modello di minaccia.

Le scritture sul file system dall'interfaccia utente sono limitate a una singola directory:

- Predefinito: `~/.backpropagate/ui-outputs`
- Sovrascrivere: impostare `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- La sovrascrittura viene convalidata tramite una lista di esclusione: i percorsi di sistema o di credenziali (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, ecc.) vengono rifiutati.

## Note sulla piattaforma

**Requisiti:** Python 3.11+ · GPU NVIDIA con CUDA · PyTorch 2.0+. Una scheda da 8 GB addestra i modelli predefiniti da 1B a 3B, 16 GB un modello da 7B e 32 GB fino a 32B con QLoRA.

Python 3.10 non è più supportato (fine del supporto 2026-10-01); la versione minima è 3.11 a partire da v1.9.0. Python 3.11 è la versione più testata.

Backpropagate gestisce le peculiarità dell'esecuzione dell'addestramento su diverse piattaforme, ma non può risolvere i problemi che si verificano durante l'installazione. I due più comuni sono:

- **Pacchetto CUDA errato.** PyTorch viene pubblicato come un singolo binario per ogni versione di CUDA. Se si sceglie quello sbagliato, si ottiene silenziosamente solo la CPU PyTorch e l'addestramento diventa impossibilmente lento. Utilizzare lo strumento di selezione dei pacchetti disponibile all'indirizzo <https://pytorch.org/get-started/locally/> per la versione del driver. Eseguire `nvidia-smi` per visualizzare la versione del driver/CUDA.
- **Windows + esportazione GGUF.** L'opzione aggiuntiva `[export]` compila `llama-cpp-python` dal codice sorgente, il che richiede Visual Studio Build Tools (componente C++) e CMake.

**macOS:** il percorso CUDA non è supportato (nessuna CUDA): un `trainer.train()` con percorso CUDA genera `DEP_GPU_NOT_AVAILABLE` e si può eseguire l'adattatore addestrato su un Mac tramite Ollama. Un percorso MLX **sperimentale e non verificato** (`--backend mlx`, `pip install 'backpropagate[mlx]'`) addestra un adattatore LoRA in modo nativo su Apple Silicon tramite `mlx_lm.lora`: solo SFT LoRA e **non verificato su hardware reale** (vedere [Apple Silicon (MLX)](#apple-silicon-mlx--anteprima-non-verificata)). Per il percorso CUDA o per ORPO / addestramento completo / FP8 / addestramento con più passaggi, utilizzare una macchina Linux o Windows con CUDA.

Vedere la [pagina della guida alla risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) per la guida completa alla risoluzione dei problemi di installazione e la [pagina dedicata alla risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) per i problemi relativi a driver / VRAM / xformers / bf16-vs-fp16.

## CLI

Ogni API Python ha un corrispondente comando CLI:

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

Riferimento completo nella [pagina della guida CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) o `backprop <subcommand> --help`.

## Configurazione

È possibile sovrascrivere ogni impostazione con una variabile d'ambiente utilizzando il prefisso `BACKPROPAGATE_`:

| Variabile | Predefinito | Note |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Forza i log in formato JSON o nella console |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modello predefinito |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Tasso di apprendimento |
| `BACKPROPAGATE_LORA__R` | `256` | LoRA rank. Impostandolo, si disattiva la scelta automatica delle dimensioni dell'adattatore (vedere `--lora-preset` nella sezione [Modelli predefiniti](#modelli-predefiniti)). |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sandbox del file system dell'interfaccia utente |

Le chiavi nidificate utilizzano il doppio trattino basso (`MODEL__NAME`, non `MODEL_NAME`). Il riferimento completo è disponibile nella [pagina della guida alle variabili d'ambiente](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Modelli predefiniti

| Modello | Memoria GPU | Licenza | Note |
|---|---|---|---|
| Qwen-3.5-4B | 6 / 7 / 11 GB | Apache 2.0 | Impostazione predefinita consigliata per sub-5B. La migliore qualità a queste dimensioni. |
| Phi-4-mini-3.8B | 6 / 7 / 12 GB | MIT | Ottimo per il ragionamento, la matematica e il codice. Licenza rigorosa. |
| SmolLM3-3B | 4 / 5 / 10 GB | Apache 2.0 | Ricetta completamente aperta. Contesto nativo di 64K. |
| Qwen 2.5 7B | 9 / 11 / 17 GB | Apache 2.0 | Impostazione predefinita esistente. La migliore qualità dei modelli predefiniti da 7B. |
| Qwen 2.5 3B | 4 / 6 / 10 GB | Qwen-Research | ⚠ licenza di ricerca: consultare i termini di licenza di Qwen prima dell'uso commerciale. |
| Llama 3.2 3B | 4 / 6 / 9 GB | Comunità Llama | Un'alternativa valida a Qwen 3B con alcune limitazioni. |
| Llama 3.2 1B | 2 / 3 / 5 GB | Comunità Llama | Per esperimenti rapidi su schede di piccole dimensioni. |
| Mistral 7B | 6 / 8 / 14 GB | Apache 2.0 | Simile a Qwen 7B, con un modello di chat diverso. |
| Llama-3.1-8B | 9 / 11 / 18 GB | Llama-3.1-Community | 8B QLoRA, 128K di contesto nativo (la clausola >700M-MAU richiede una licenza Meta separata). |
| **Qwen2.5-14B** | 18,7 GiB al massimo con 4096 ctx (QLoRA) | Apache 2.0 | **Il modello da 32 GB per l'uso quotidiano.** rank/alpha 32, 8 bit AdamW. I soli pesi a 4 bit occupano circa 8,5 GB; per una finestra completa di 4096 token è necessario il resto. |
| Mistral-Small-24B | 22,8 GiB al massimo con 4096 ctx (QLoRA) | Apache 2.0 | 24B QLoRA su una scheda da 32 GB. I soli pesi a 4 bit occupano circa 18 GB. |
| **Qwen2.5-32B** | 26,0 GiB al massimo con 2048 ctx (QLoRA) | Apache 2.0 | **Il modello migliore per una scheda da 32 GB.** Si adatta a `max_len 2048` con 8 bit AdamW. |

Altri modelli spesso funzionano; le righe sopra riportano le impostazioni predefinite selezionate: la categoria da 14B a 32B è ottimizzata con il valore QLoRA per una scheda da 32 GB (il valore misurato). Per le impostazioni predefinite fino a 8B, le tre cifre rappresentano le stime QLoRA per le dimensioni degli adattatori `fast`, `balanced` e `quality` con un batch di 1 e esempi di 2.048 token; tendono a sovrastimare e gli esempi più brevi richiedono meno risorse. La dimensione dell’adattatore viene scelta in base alla scheda: `--lora-preset auto` (il valore predefinito) utilizza il valore più grande tra `quality` (rango 256 su ogni livello lineare, secondo Biderman 2024 e Thinking Machines 2025), `balanced` (rango 64 su ogni livello lineare) e `fast` (rango 16 su due livelli per blocco) che si adatta alla memoria disponibile sulla GPU. È possibile specificarne uno per forzarne l’utilizzo. `backprop estimate-vram` stampa la stima per qualsiasi modello e impostazione.

## Risoluzione dei problemi

Un breve elenco dei fallimenti più comuni durante la prima esecuzione. L'indice completo è disponibile nella [pagina del manuale per la risoluzione dei problemi](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Per un'analisi approfondita di driver / VRAM / precisione mista, consulta la [pagina per la risoluzione dei problemi di CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Sintomo | Codice di errore | Soluzione |
|---|---|---|
| La GPU esaurisce la memoria durante l'addestramento | `RUNTIME_GPU_OOM` | Automatico: Backpropagate dimezza la dimensione del batch e riprova fino a 3 volte. Per disattivare: `Trainer(oom_recovery=False)`. Per forzare una dimensione inferiore: `--batch-size 1`. |
| HuggingFace restituisce 401 / "modello non trovato" | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` e riprova. In caso di errori di battitura, copia l'ID esatto da <https://huggingface.co/models>. |
| `register_with_ollama` connessione rifiutata | `DEP_OLLAMA_REGISTRATION_FAILED` | Avvia il daemon: `ollama serve`. Installa da <https://ollama.com>. Può essere ripetuto. |
| Lo spazio su disco è insufficiente durante il salvataggio del checkpoint | `STATE_CHECKPOINT_INVALID` | Le scritture atomiche lasciano una directory `.partial` in caso di errore: è sicuro eliminarla. Il checkpoint precedente è intatto. |
| L'addestramento viene interrotto a causa del surriscaldamento della GPU | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automatico: Backpropagate si interrompe quando viene superata la soglia di temperatura e riprende quando la GPU si raffredda. Migliora il flusso d'aria se il problema persiste. |
| `backprop ui --share` rifiutato | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Passa `--auth user:password` o utilizza invece il port forwarding SSH (vedi [Interfaccia utente web](#interfaccia-utente-web)). |
| L'esportazione GGUF non riesce al primo tentativo | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; su Windows è necessario anche Visual C++ Build Tools + CMake. |

## Segnalazione di bug

Quando qualcosa non va, Backpropagate stampa una riga all'avvio simile a `run_started run_id=<uuid>` e associa lo stesso ID a ogni riga del log, a ogni checkpoint e a ogni voce di Weights & Biases. **Includi il `run_id` in qualsiasi segnalazione di bug**: consente a un manutentore di correlare tutto per quella specifica esecuzione.

Una buona segnalazione di bug include:

1. **Il `run_id`**: l'UUID stampato all'avvio. Un UUID consente a un manutentore di correlare ogni riga del log, ogni checkpoint e ogni voce di Weights & Biases per quella specifica esecuzione.
2. **Il codice di errore**: la riga `[CODE_NAME]: message` in stderr. Consulta [codici di errore](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) per il catalogo dei codici stabili.
3. **Il traceback modificato**. Stderr viene automaticamente modificato in modalità non verbose (i token Bearer, `sk-*`, `hf_*`, le chiavi AWS, le coppie `password=` / `token=` / `api_key=` vengono eliminate): è sicuro incollarlo. Per il traceback completo non modificato, riesegui con `BACKPROPAGATE_DEBUG=1` (o `--verbose`); rivedilo prima di pubblicarlo.
4. **L'output `backprop info`**. Un comando stampa Python / PyTorch / CUDA / modello GPU / VRAM / sistema operativo / extra installati: tutto ciò di cui il manutentore ha bisogno per individuare una regressione specifica della piattaforma.

Il [modello per la segnalazione di bug](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) richiede esplicitamente tutte queste informazioni, in modo da velocizzare il processo di analisi e risoluzione. Domande, idee o discussioni del tipo «è questo il comportamento previsto?» devono essere pubblicate in [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions). I problemi di sicurezza devono essere segnalati in privato tramite il modulo [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new); per la politica e i tempi di risposta, consultare il file [SECURITY.md](SECURITY.md).

## Privacy

Tutto l'addestramento avviene localmente sulla tua GPU. Backpropagate non effettua richieste di rete, tranne per scaricare i modelli da HuggingFace (che avvii tu). Nessun telemetria, nessuna dipendenza dal cloud.

## Riferimenti

Le impostazioni predefinite di Backpropagate e la modalità di addestramento multi-esecuzione si basano su ricerche recenti. Se sei interessato alle tecniche sottostanti:

- **Hu et al. 2021.** *LoRA: Adattamento a basso rango di modelli linguistici di grandi dimensioni.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — l'articolo fondamentale che introduce LoRA, ovvero il metodo con cui Backpropagate addestra gli adattatori in modo efficiente.
- **Biderman et al. 2024.** *LoRA Apprende meno e dimentica meno.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidenze empiriche che dimostrano che LoRA con rango 256 e obiettivi completamente lineari raggiunge una qualità paragonabile al fine-tuning completo nella maggior parte delle attività post-addestramento, utilizzando il 67% della potenza di calcolo. Determina la configurazione predefinita v1.3 di Backpropagate, ovvero LoRA.
- **Thinking Machines 2025.** *LoRA Senza rimpianti.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — il seguito pratico che identifica la correzione del tasso di apprendimento di 10 volte rispetto al fine-tuning completo necessaria a un rango LoRA elevato.
- **Kirkpatrick et al. 2017.** *Superare l'oblio catastrofico nelle reti neurali.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — la caratterizzazione originale del motivo per cui le reti neurali "dimenticano" l'addestramento precedente quando si esegue il fine-tuning su nuovi dati (EWC — Elastic Weight Consolidation).
- **Wang et al. 2023.** *Apprendimento di sottospazi ortogonali per l'apprendimento continuo di modelli linguistici.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, un approccio precedente all'utilizzo di LoRA per l'apprendimento continuo, limitando i nuovi adattatori a sottospazi ortogonali.
- **Yadav et al. 2023.** *TIES-Merging: Risolvere le interferenze durante la fusione di modelli.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — una tecnica fondamentale per la fusione di più modelli con fine-tuning senza interferenze.
- **Qiao & Mahdavi 2025.** *Unisci prima di dimenticare: un singolo LoRA per l'apprendimento continuo tramite fusione continua.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — l'algoritmo specifico che il sistema di fusione multi-esecuzione di Backpropagate implementa. Un preprint di dicembre 2025; Backpropagate è il primo utilizzatore noto di questo articolo.

## Licenza

MIT — vedere [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
