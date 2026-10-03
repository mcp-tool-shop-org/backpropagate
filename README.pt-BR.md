<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
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

# Ajuste um modelo QLoRA de 32 bilhões de parâmetros — ou um modelo completo de 7 bilhões de parâmetros — em uma única GPU. Envie-o para o Ollama

Realize o ajuste fino de grandes modelos de linguagem em uma **única** GPU, dimensionada para a placa que você realmente possui. Três linhas de código Python para ajustar um modelo QLoRA de 7 a 32 bilhões de parâmetros em uma única placa de consumidor de 32 GB (RTX 5090). Uma única flag, `--full-ft-offload`, realiza o ajuste fino completo de um modelo de 7 bilhões de parâmetros, mantendo seus pesos e gradientes na RAM do host (Linux ou WSL2; lento, e medido abaixo). Mais um comando exporta para o Ollama e, em seguida, `ollama run` seu ajuste fino. Reduz a escala para 16 GB. Desempenho de primeira linha no Windows. Prefere um navegador ao Python? `backprop ui` faz tudo isso sem código ([faça o tour](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/)).

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

É isso. Não há arquivo de configuração YAML. Não há cerimônia `accelerate launch`. Não há tutorial separado de "agora, converta-o para GGUF". Se você tiver uma GPU CUDA e um arquivo JSONL com seus dados de treinamento, estará a apenas três linhas de um ajuste fino funcional.

## Instale

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

Se você quiser os recursos opcionais, substitua a instalação por uma destas:

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

Prefere o Docker? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` também funciona. As imagens são fornecidas para `linux/amd64` e `linux/arm64`, para que os usuários de Apple Silicon e ARM Linux obtenham uma imagem nativa. Um `compose.yaml` canônico para "interface do usuário em um contêiner" está na raiz do repositório: coloque `user:password` em um `ui-auth.txt` ao lado dele, execute `docker compose up` e faça login em `http://127.0.0.1:7860` (a primeira execução cria o frontend, o que leva um minuto ou dois). O histórico de execução é persistido em `~/.backpropagate`.

## Onde o Backpropagate se encaixa

Existem várias bibliotecas boas para o ajuste fino de LLMs. Cada uma delas é ótima em coisas diferentes:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — se você gosta de configurações YAML e deseja uma comunidade de receitas para copiar
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — se você quiser DPO/PPO/RLHF e uma GUI da web
- **[Unsloth](https://github.com/unslothai/unsloth)** — se você precisar do treinamento mais rápido possível e estiver em uma família de modelos compatível
- **[torchtune](https://github.com/pytorch/torchtune)** — se você quiser as receitas nativas do PyTorch da Meta que pode editar

Backpropagate é a opção que faltava: **uma API Python de 3 linhas para operadores individuais em uma única GPU de consumidor que desejam treinar um adaptador e enviá-lo.** Sem YAML, sem RL online (PPO/GRPO), sem multi-nó. Há uma interface de usuário do navegador para o mesmo loop, caso você prefira não escrever código. Apenas o loop que todos realmente precisam e a etapa de exportação que atrapalha.

Se você tentou uma das bibliotecas acima e se frustrou com a cerimônia do arquivo de configuração, ou atingiu uma lacuna na família de modelos, ou desejou configurações padrão com foco no Windows, o Backpropagate é para você.

## O que você pode ajustar em uma única GPU

Backpropagate dimensiona a execução para sua placa. Estes são números **medidos** em uma RTX 5090 de 32 GB: as linhas QLoRA de 2026-10-03 (comprovantes: [`docs/receipts/2026-10-03-presets/`](docs/receipts/2026-10-03-presets/)), as linhas de ajuste fino completo de 2026-09-30 (comprovantes: [`docs/receipts/2026-09-30-offload/`](docs/receipts/2026-09-30-offload/)). Os picos do QLoRA estão na janela de contexto total do predefinido com lote 1, que é o pior caso para esse predefinido; exemplos mais curtos usam menos.

| Modelo | Método | Medido em uma placa de 32 GB |
|---|---|---|
| **14B** (Qwen2.5-14B) | QLoRA | **18,7 GiB** de pico em 4096 de contexto (20,0 GiB reservados). |
| 24B (Mistral-Small-24B) | QLoRA | 22,8 GiB de pico em 4096 de contexto (24,2 GiB reservados). |
| **32B** (Qwen2.5-32B) | QLoRA | **Cabe:** 26,0 GiB de pico em 2048 de contexto (27,2 GiB reservados, cerca de 4 GiB de folga). |
| 3B | `mode="full"` (ajuste fino completo real, na GPU) | **22,0 GiB** de pico (em todo o sistema), 0,30 s/etapa com lote 4, 512 de contexto. 7,5 GiB disso é o estado do otimizador paginado, que pode ser transferido para a RAM do host em uma placa menor (não testado). |
| **7B-class** (Qwen2.5-7B, 7,6B de parâmetros) | `mode="full" --full-ft-offload` | **Treina:** 5,3 GiB de VRAM, **30,8 GiB de RAM do host** (32,2 GiB durante a gravação), **14,7 s/etapa**. Apenas Linux ou WSL2. |

Não medido novamente nessa sessão: QLoRA de 7B, Llama-3.1-8B (repositório fechado, sem token na máquina de teste) e ajuste fino completo na GPU acima de 3B. As informações sobre esses modelos em outros lugares da documentação são estimativas.

Duas coisas para as quais a maioria das bibliotecas de GPU única envia você para outro lugar, **QLoRA de 24 a 32B** e **ajuste fino completo de modelo de 7 bilhões de parâmetros em uma única placa**, Backpropagate faz em uma única placa de consumidor e, em seguida, exporta o resultado diretamente para o Ollama.

**O ajuste fino completo tem dois caminhos.** Sem descarregamento, o modelo, seus gradientes e o estado do otimizador ficam todos na GPU. A biblioteca limita o tamanho do modelo detectando a VRAM (**16 GB → 4B, 24 GB → 5B, 32 GB → 6B**); esses limites vêm da matemática da memória e são medidos apenas até 3B. Substitua com `--full-ft-ceiling-billions`.

`--full-ft-offload` mantém os pesos e gradientes na RAM do host e os transmite para a GPU (FSDP2 CPU offload). O que isso custa, medido:

- **RAM da máquina:** o teste de compatibilidade exige cerca de 3,7 GiB por bilhão de parâmetros, mais 10 GiB, o que é um valor conservador (39 GiB em 7,6B, em comparação com os 32,2 GiB medidos). A execução é recusada logo de cara, com base nesses números, se a máquina não tiver capacidade para isso. Um modelo de 7,6B não cabe na memória de 28 GB do WSL2; cerca de 5B é o limite prático nesse caso.
- **Velocidade:** 14,7 s/etapa em 7,6B (lote 1) e 5,1 s/etapa em 3B (lote 4), em comparação com cerca de 0,63 s/etapa para 3B na GPU, no lote 4. Use-o apenas quando o modelo não couber sem ele. Uma versão mais rápida está planejada.
- **Otimizador:** Adafactor, não AdamW. Os pesos permanecem em bf16 e cada atualização é reescrita com arredondamento estocástico; não há cópia fp32.
- **Qualidade:** em uma execução de 3B (150 etapas, perda em dados não utilizados, uma semente), atingiu cerca de 85% da melhoria obtida pelo ajuste fino completo normal (2,45 → 1,93, em comparação com 2,45 → 1,84). Uma única semente não é um parâmetro de referência.
- **Escopo:** ajuste fino supervisionado simples. Sem agrupamento, sem mascaramento apenas de resposta, sem pontos de verificação intermediários, sem retomada. Apenas Linux ou WSL2 (FSDP2 requer NCCL); no Windows nativo, ele para com `DEP_FSDP_UNAVAILABLE`.
- **Ainda não testado:** execuções longas, acumulação de gradiente acima de 1 e uma máquina física de 64 GB (a máquina de teste tinha mais RAM, com um limite de 60 GiB imposto pelo teste).

Um modelo que não cabe sai com `RUNTIME_FULL_FT_MODEL_TOO_LARGE` e indica a solução. Consulte a [página completa do manual de ajuste fino](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/).

### Reduz para 16 GB

O intervalo de 16 GB (RTX 4080 / 5080 / 4070 Ti Super) ainda é de primeira linha: 7B QLoRA (o tamanho do adaptador é escolhido para caber: classificação 64 em uma placa de 16 GB, onde a classificação 256 precisa de cerca de 17 GB) e ajuste fino completo real de um modelo de ~3B (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B/1B) via `mode="full"` (22,0 GiB medidos em uma placa de 32 GB em 3B, dos quais 7,5 GiB são o estado do otimizador paginado que pode ser transferido para a RAM da máquina; se isso funcionar de forma aceitável em uma placa de 16 GB, ainda não foi testado). Com `--full-ft-offload`, a GPU armazena muito menos: com a VRAM limitada na placa de teste, um modelo de 3B treinado sob um limite de 6 GiB e modelos de 4B e 7,6B sob um limite de 8 GiB. Esses são limites simulados em uma placa de 32 GB, não execuções em hardware real de 8 GB. O mesmo código seleciona o tamanho do lote e o limite que cabem em qualquer placa que ele detecte.

A quantização de 2 bits (AQLM / QuIP#) permanece **fora do escopo** — uma base de 2 bits não pode ser mesclada de forma limpa de volta aos pesos de precisão total, o que quebra o contrato de adaptador mesclável → GGUF → Ollama (o objetivo principal do pipeline). Em vez disso, o Backpropagate oferece os recursos de folga — QLoRA, `mode="full"`, `--full-ft-offload` e o caminho de computação FP8 (`--fp8`, Blackwell/Hopper) — todos que permanecem mescláveis e exportáveis.

## Para que o Backpropagate NÃO serve

Se o seu caso de uso estiver abaixo, você terá mais sucesso com uma biblioteca diferente — o Backpropagate não é a escolha certa e tentar fazê-lo funcionar custaria mais do que simplesmente usar a ferramenta certa. Ler esta seção antes de começar economiza o ciclo de instalação e reinício:

- **Ajuste fino de parâmetros completos de modelos de 13B+** — o Backpropagate realiza o ajuste fino completo de até cerca de 6B em uma GPU de 32 GB e um modelo de 7B com `--full-ft-offload` (veja [o intervalo](#o-que-você-pode-ajustar-em-uma-única-gpu)). Um ajuste fino completo de um modelo de 13B+ requer FSDP multi-GPU ou uma placa maior. Antes de gastar esses recursos de computação, pondere as evidências em ambos os sentidos. [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/) relata que o LoRA corresponde ao ajuste fino completo quando é aplicado a cada camada e o conjunto de dados se ajusta à capacidade do adaptador, em cerca de dois terços dos recursos de computação por passagem. [Biderman et al. 2024](https://arxiv.org/abs/2405.09673) descobriram que, em configurações de baixa classificação padrão, o LoRA tem um desempenho significativamente inferior ao ajuste fino completo em código e matemática, ao mesmo tempo em que esquece menos. Para o acompanhamento de instruções, trabalho de persona e estilo em conjuntos de dados modestos, o QLoRA de até 32B geralmente é o melhor uso de uma única placa.
- **RL online — PPO / GRPO / RLVR** — o Backpropagate realiza o ajuste fino de estágio único (SFT) mais o ajuste de preferência sem referência (ORPO na v1.5; SimPO + KTO na v1.6). O que ele *não* faz é o aprendizado por reforço online — PPO, GRPO ou RLVR — que requer um modelo de recompensa ou um loop de geração e pontuação no topo da etapa de treinamento. Para esses, use o TRL diretamente ou o LLaMA-Factory. (O ajuste de preferência sem referência se encaixa no intervalo de estágio único porque não há um modelo de referência separado para manter na memória; veja a nota do ORPO em [Início Rápido](#início-rápido).)
- **Treinamento multi-nó** — apenas uma GPU em uma única máquina. Multi-GPU em uma única máquina funciona (via `accelerate launch`), mas não é oficialmente suportado.
- **Treinamento no macOS na trilha CUDA** — o Apple Silicon não tem CUDA, portanto, o caminho CUDA é executado em uma caixa Linux ou Windows com uma GPU NVIDIA. Você ainda pode executar o modelo treinado em um Mac via Ollama. Uma trilha MLX **experimental e não verificada** (`--backend mlx`) treina um adaptador LoRA nativamente no Apple Silicon — veja [Apple Silicon (MLX)](#apple-silicon-mlx--visualização-não-verificada). É apenas SFT LoRA e **não foi verificado em silício real** (sem suporte), portanto, para qualquer coisa além de um SFT LoRA (ORPO, ajuste fino completo, FP8, execução múltipla), você deseja a trilha CUDA.
- **Qualquer coisa fora das famílias de modelos testadas** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Outros modelos geralmente funcionam, mas não estão fixos no CI.

Se você precisar de alguma dessas coisas, use uma das bibliotecas listadas acima. Elas são melhores para isso.

## O que o Backpropagate oferece

Quatro coisas, em uma única instalação:

**1. Uma API real de 3 linhas que é executada sem um arquivo de configuração.**
O snippet no topo deste README é executado de ponta a ponta. Sem `accelerate config`, sem YAML, sem substituições Hydra. Apenas `Trainer(model).train(data)` e você tem um ajuste fino.

**2. Windows que realmente funciona.**
A maioria das bibliotecas de aprendizado de máquina trata o Windows como algo secundário. O Backpropagate é desenvolvido e testado no Windows 11 com placas da série RTX 50. A biblioteca lida com as peculiaridades de tempo de execução para você — ela sabe como pré-tokenizar seus dados para que o processamento paralelo do Windows não falhe, desativa automaticamente o xformers em placas RTX 40/50 onde isso causaria problemas e seleciona configurações de carregamento de dados que não geram erros. Você não precisa saber nada disso. Ele simplesmente funciona.

**3. Projetado para execuções não supervisionadas.**
O treinamento leva horas. Você não quer ficar monitorando-o constantemente. O Backpropagate é projetado para ser deixado em execução:

- Se você ficar sem memória da GPU, ele reduz automaticamente pela metade o tamanho do lote e tenta novamente — até três vezes. Sem ajustes manuais.
- Se sua GPU ficar muito quente, ele pausa até que as coisas esfriem e, em seguida, continua.
- Cada ponto de verificação é gravado atomicamente — se o seu laptop travar durante a gravação, o ponto de verificação anterior e válido ainda estará intacto.
- Cada execução de treinamento recebe um ID exclusivo que é carimbado em cada linha de log, em cada ponto de verificação e em cada entrada de Weights & Biases. Se algo der errado, um único ID permite que um mantenedor correlacione tudo.
- Os erros vêm com códigos estáveis (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, etc.) para que você possa pesquisar em seus logs e no [guia de solução de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para encontrar a solução. As falhas específicas do CUDA têm uma [página de solução de problemas do CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) dedicada.

**4. Um único comando, do adaptador treinado para `ollama run`.**
Muitas bibliotecas treinam um modelo. Poucas delas facilitam o uso real do modelo. O Backpropagate exporta para GGUF (o formato usado pelo Ollama) e registra um modelo Ollama em um único comando. Você passa de "treinamento concluído" para "posso conversar com meu modelo ajustado" em cerca de 30 segundos.

## Guia de Início Rápido

Na linha de comando, com um conjunto de dados de exemplo de 5 conversas:

```bash
pipx install "backpropagate[standard]"
curl -LO https://raw.githubusercontent.com/mcp-tool-shop-org/backpropagate/main/examples/quickstart.jsonl

backprop train --data quickstart.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 10
backprop generate ./output "What is Python?"      # did it learn anything?
backprop export ./output --format gguf --quantization q4_k_m --ollama --ollama-name my-first-finetune
ollama run my-first-finetune
```

`backprop train` grava o adaptador em `./output` (altere-o com `--output`). Em Python, a mesma coisa é:

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")
```

Use um ambiente virtual com `pip install "backpropagate[standard]"` para a API Python; `pipx` instala o comando `backprop` em seu próprio ambiente, para que `import backpropagate` não o encontre.

**O que a exportação GGUF precisa.** A exportação mescla seu adaptador com o modelo base e o converte com o script de conversão do llama.cpp. Você precisa de:

- um **checkout da fonte** do llama.cpp (`git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp`) mais `pip install sentencepiece protobuf` no mesmo ambiente, ou
- Unsloth com seu próprio llama.cpp já construído.

Com `--ollama`, a quantização `q4_k_m` é feita por `ollama create`, portanto, nada precisa ser compilado. O Backpropagate nunca permite que o Unsloth instale pacotes do sistema para construir o llama.cpp para você; defina `BACKPROPAGATE_UNSLOTH_AUTO_INSTALL=1` se quiser que isso aconteça. Detalhes: [export](https://mcp-tool-shop-org.github.io/backpropagate/handbook/export/).

Para seus próprios dados, formate seu JSONL com um exemplo por linha:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca (`instruction` / `output`), OpenAI chat (`messages`) e formatos de texto bruto também funcionam — o Backpropagate detecta automaticamente o formato.

### O ciclo: verificar os dados, treinar, avaliar, exportar

```bash
backprop data report my_data.jsonl                     # duplicates, length outliers, format problems
backprop data split my_data.jsonl --heldout-ratio 0.1  # a held-out set the model never trains on
backprop train --data my_data.train.jsonl --steps 200 --output ./run-a
backprop eval <run-id> --heldout my_data.heldout.jsonl # held-out loss + sample generations
backprop eval <run-b> --vs <run-a>                     # did the change help?
backprop export ./run-a --format gguf --ollama --ollama-name my-model
```

A avaliação é feita sem julgamento, por design: perda de dados não utilizados mais métricas de tarefa determinísticas (`normalized_exact_match`, `token_f1`, `contains`, `regex`, `pass_rate`). Para usar um avaliador LLM, execute-o sobre a saída de `backprop generate`. Consulte [receitas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Ajuste de preferência (ORPO, SimPO, KTO)

Treine com base em preferências, em vez de demonstrações simples. ORPO não requer referência e é de estágio único — ele integra o sinal de preferência na etapa de ajuste fino (SFT), portanto, não há modelo de recompensa ou referência separado e a forma de 3 linhas permanece inalterada. Passe `--method orpo` (CLI) ou `method="orpo"` (Python) e forneça um conjunto de dados de `{prompt, chosen, rejected}` (ou apenas `{chosen, rejected}`) linhas:

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

A taxa de aprendizado padrão é automaticamente reduzida para `8e-6` para ORPO (a perda é mais acentuada do que no ajuste fino simples); ajuste `--orpo-beta` (padrão `0.1`) para ponderar a penalidade da razão de chances. ORPO é apenas `mode="lora"`.

**Novo na v1.6 — SimPO e KTO.** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) não requer referência, com uma recompensa normalizada pelo comprimento, e usa os mesmos dados pareados `{prompt, chosen, rejected}` que o ORPO (`--simpo-beta`, `--simpo-gamma`). `--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) usa dados **não pareados** `{prompt, completion, label}` — avaliações positivas/negativas por exemplo — para a grande classe de feedback que não são pares A/B selecionados; ele equilibra automaticamente os pesos de perda desejáveis/indesejáveis a partir das contagens de rótulos. Ambos são apenas `mode="lora"` e permanecem no envelope de ajuste fino de GPU única (sem modelo de referência separado). Consulte o [guia de ajuste de preferência](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) para saber qual usar. Para RL online (PPO/GRPO), consulte [Para que o Backpropagate NÃO serve](#what-backpropagate-is-not-for).

### Ajuste fino de rastreamento de raciocínio (destilação R1)

Destile um modelo de raciocínio da maneira fácil. Passe `--reasoning-trace` (CLI) ou `Trainer(..., reasoning_trace=True)` (Python) e forneça rastreamentos que mantenham uma cadeia de pensamento `<think>...</think>` dentro da resposta do assistente — a metade de ajuste fino simples (SFT) da destilação [DeepSeek-R1](https://arxiv.org/abs/2501.12948), sem necessidade de RL. O Backpropagate mantém `<think>` no destino de treinamento, descarta rastreamentos vazios/muito longos (filtragem do comprimento do rastreamento) e aumenta o padrão `max_seq_length` para 8192 para o CoT mais longo. Fundamentalmente, `<think>` permanece em **texto simples** — sem tokens especiais, sem redimensionamento de incorporação — para que o GGUF mesclado ainda seja exportado para o Ollama como qualquer outro ajuste fino. Apenas SFT. Consulte a [receita de rastreamento de raciocínio](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) para a forma do conjunto de dados e o token ajustável.

### Apple Silicon (MLX) — visualização não verificada

> ⚠️ **Pré-visualização não verificada — não faz parte do conjunto de funcionalidades suportadas.** O "rail" MLX é construído e testado em nível de unidade, mas **não** foi verificado em um ambiente real de Apple Silicon (`mlx-lm` é exclusivo da Apple e não pode ser executado nos equipamentos NVIDIA nos quais o Backpropagate é desenvolvido). Considere tudo o que se segue como experimental, use por sua conta e risco, e [relate anomalias](#reporting-bugs) se você o executar em um Mac da série M.

**Uma API, dois "rails".** CUDA é o "backend" padrão e verificado; MLX é um segundo "rail" que treina em um Mac da série M por meio do conjunto de ferramentas [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) da Apple (memória unificada, sem CUDA). A configuração de 3 linhas seleciona o "rail" com base no hardware — `backend='auto'` (o padrão) direciona para CUDA em NVIDIA e para MLX em Apple Silicon, de modo que os equipamentos CUDA existentes sejam idênticos em termos de bytes:

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

O "rail" MLX é **apenas LoRA SFT** — sem ORPO, sem FP8, sem `mode='full'`, sem execução múltipla (cada um é rejeitado com `CONFIG_INVALID_SETTING`; use `backend='cuda'`/`'auto'` em um equipamento NVIDIA para esses casos). O adaptador resultante é um arquivo safetensors simples e é exportado para o Ollama por meio do mesmo caminho do "rail" CUDA.

> Forçar `--backend mlx` em um host que não seja da Apple gera um erro com `CONFIG_INVALID_SETTING`; a ausência do conjunto de ferramentas `mlx_lm` em um Mac gera `DEP_MLX_UNAVAILABLE`.

Para obter mais fluxos de trabalho completos (ajuste fino e envio para o HF Hub, retomada após falta de memória, SLAO de execução múltipla em uma campanha longa, etc.), consulte a [página de receitas do manual](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interface de usuário da web (opcional)

Se você preferir clicar em vez de digitar em Python, instale o pacote extra da interface do usuário e inicie:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Abra a URL que ele exibe, `http://127.0.0.1:7862/?token=...` (cada execução gera um novo token; a primeira execução constrói o "frontend" e pode levar um minuto ou dois). É uma interface web local para treinamento: inicie uma execução, uma série de execuções múltiplas ou uma exportação, observe-a em tempo real (etapas, perda, tempo restante, temperatura e memória da GPU) e interrompa-a com um ponto de verificação salvo. Cada tarefa é executada em seu próprio processo, um de cada vez, e recarregar a página retoma uma tarefa em execução. A página de Conjunto de dados mostra o que um arquivo contém, salva uma cópia limpa (repetições e exemplos vazios removidos) e a envia para o formulário de treinamento. As execuções anteriores e os modelos em seu cache do Hugging Face têm suas próprias páginas, e cada configuração tem um "i" que a explica. O [tour da interface de usuário da web](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/) mostra cada página. A interface do usuário é local por padrão. Para expô-la a outros dispositivos, consulte [Interface de usuário da web](#web-ui) abaixo para o contrato de segurança `--share` + `--auth`.

## Treinamento com execução múltipla

Se você quiser fazer um ajuste fino de forma incremental em vários conjuntos de dados — por exemplo, se você obtiver novos dados de treinamento a cada semana e quiser adicioná-los sem esquecer o que aprendeu antes — o modo `multi_run` do Backpropagate é para você:

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

Isso executa cinco passagens de treinamento, mesclando o adaptador entre as execuções de uma forma que preserva o conhecimento anterior, incorporando novos exemplos. A técnica é baseada em pesquisas recentes sobre aprendizado contínuo — consulte [Referências](#references) na parte inferior deste README.

A versão da CLI:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Retomar a partir de um ponto de verificação

Um treinamento de 5 execuções que falha na execução 4 pode ser recuperado. Cada sessão de execução múltipla grava seu ID de execução no histórico e no manifesto de pontos de verificação no disco, de modo que retomar de onde você parou é um único comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

O comportamento padrão de `backprop multi-run` (sem `--resume`) detecta automaticamente uma entrada em andamento no mesmo diretório de saída e a continua. Para forçar uma nova inicialização, aponte para um novo diretório de saída.

## Histórico de treinamento

Cada invocação de `backprop train` e `backprop multi-run` registra uma linha em `<output>/run_history.json` — modelo usado, conjunto de dados, hiperparâmetros, status, perda final, histórico de perda. Você pode listar e inspecionar execuções anteriores:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Rastreamento de experimentos

O Backpropagate detecta automaticamente os rastreadores de experimentos instalados (Weights & Biases, TensorBoard, MLflow) e os integra. Se `wandb` estiver instalado e você estiver conectado, cada execução registrará automaticamente no W&B com um nome de execução que corresponda ao ID de execução no disco — para que você possa pesquisar no W&B, em seus logs e em `run_history.json` usando um único identificador.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Substitua por `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` ou `Trainer(report_to="none")` para optar por não participar.

## Interface de usuário da web

A interface web Reflex é opcional — instale com `pipx install "backpropagate[ui]"` e inicie:

```bash
backprop ui --port 7862
```

A interface do usuário é executada localmente: abra a URL que ela exibe, `http://127.0.0.1:7862/?token=...`. Sem `--auth`, cada execução gera um novo token e a interface do usuário rejeita solicitações que não o possuem. A partir dela, você pode visualizar e limpar um conjunto de dados, treinar (uma única execução ou uma execução múltipla), observar a execução em tempo real, interrompê-la com um ponto de verificação salvo e exportar o resultado. Cada tarefa é executada em seu próprio processo, um de cada vez, e fechar a interface do usuário a interrompe. O [tour da interface de usuário da web](https://mcp-tool-shop-org.github.io/backpropagate/handbook/web-ui/) percorre cada página com capturas de tela.

Para expô-la a outros dispositivos (outras pessoas em sua rede, uma URL pública, etc.), você deve combinar `--share` (ou `--host`) com `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` sem `--auth` sai com um erro. O motivo: `--share` publica uma URL que qualquer pessoa na Internet pode acessar e, sem autenticação, isso significa que qualquer pessoa pode controlar seu pipeline de treinamento e ler seu token do Hugging Face. Não há como desativar isso — se você não quiser definir credenciais, use o encaminhamento de porta SSH em vez disso:

```bash
# On the client:
ssh -L 7862:localhost:7862 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open the URL the server printed (http://127.0.0.1:7862/?token=...) locally
```

Consulte [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) para obter o modelo de ameaças completo.

As gravações no sistema de arquivos da interface do usuário são restritas a um único diretório:

- Padrão: `~/.backpropagate/ui-outputs`
- Substituição: defina `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- A substituição é validada por uma lista de permissões — caminhos do sistema ou de credenciais (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) são rejeitados

## Observações sobre a plataforma

**Requisitos:** Python 3.10+ · GPU NVIDIA com CUDA · PyTorch 2.0+. Uma placa de 8 GB treina os modelos pré-definidos de 1B a 3B, 16 GB um modelo de 7B e 32 GB até 32B com QLoRA.

O Python 3.10 é suportado até a versão 1.6; seu suporte oficial termina em outubro de 2026 e está programado para ser removido na primeira versão após essa data. Para novas instalações, prefira o Python 3.11 ou 3.12 — o 3.11 é a versão mais testada.

O Backpropagate lida com as peculiaridades de tempo de execução do treinamento em diferentes plataformas, mas não pode corrigir problemas de instalação. Os dois mais comuns são:

- **Pacote CUDA incorreto.** O PyTorch é publicado com um pacote binário para cada versão do CUDA. Se você escolher o pacote errado, obterá silenciosamente o PyTorch apenas para CPU e o treinamento será impossivelmente lento. Use o seletor de pacotes em <https://pytorch.org/get-started/locally/> para o seu driver. Execute `nvidia-smi` para ver a versão do seu driver/CUDA.
- **Windows + exportação GGUF.** O `[export]` cria pacotes adicionais `llama-cpp-python` a partir do código-fonte, o que requer o Visual Studio Build Tools (componente C++) e o CMake.

**macOS:** o suporte para CUDA não é fornecido (sem CUDA) — um `trainer.train()` com CUDA gera `DEP_GPU_NOT_AVAILABLE`, e você pode executar o adaptador treinado em um Mac via Ollama. Um "rail" MLX **experimental e não verificado** (`--backend mlx`, `pip install 'backpropagate[mlx]'`) treina um adaptador LoRA nativamente no Apple Silicon via `mlx_lm.lora` — apenas LoRA SFT, e **não verificado em hardware real** (veja [Apple Silicon (MLX)](#apple-silicon-mlx--unverified-preview)). Para o caminho CUDA ou para ORPO / ajuste fino completo / FP8 / execução múltipla, use uma máquina Linux ou Windows com CUDA.

Consulte a [página do guia de solução de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para obter o guia completo de correção de instalação e a [página dedicada de solução de problemas do CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) para problemas de driver / VRAM / xformers / bf16 vs. fp16.

## CLI

Cada API Python tem um equivalente na CLI:

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

A referência completa está na [página do guia da CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), ou `backprop <subcommand> --help`.

## Configuração

Cada configuração pode ser substituída com uma variável de ambiente usando o prefixo `BACKPROPAGATE_`:

| Variável | Valor padrão | Notas |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Força logs em JSON ou no console |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modelo padrão |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Taxa de aprendizado |
| `BACKPROPAGATE_LORA__R` | `256` | Rank LoRA. Definir isso desativa a escolha automática do tamanho do adaptador (veja `--lora-preset` em [Modelos predefinidos](#model-presets)). |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sandbox do sistema de arquivos da interface do usuário |

Chaves aninhadas usam dois sublinhados (`MODEL__NAME`, não `MODEL_NAME`). A referência completa está na [página do guia de variáveis de ambiente](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Modelos predefinidos

| Predefinição | Memória da GPU | Licença | Notas |
|---|---|---|---|
| Qwen-3.5-4B | 6 / 7 / 11 GB | Apache 2.0 | Recomendado como padrão para modelos menores que 5B. Melhor qualidade para este tamanho. |
| Phi-4-mini-3.8B | 6 / 7 / 12 GB | MIT | Forte em raciocínio / matemática / código. Licença estrita e limpa. |
| SmolLM3-3B | 4 / 5 / 10 GB | Apache 2.0 | Receita totalmente aberta. Contexto nativo de 64K. |
| Qwen 2.5 7B | 9 / 11 / 17 GB | Apache 2.0 | Padrão existente. Melhor qualidade das predefinições 7B legadas. |
| Qwen 2.5 3B | 4 / 6 / 10 GB | Qwen-Research | ⚠ licença de pesquisa — consulte os termos da licença Qwen antes do uso comercial. |
| Llama 3.2 3B | 4 / 6 / 9 GB | Llama Community | Alternativa sólida ao Qwen 3B com ressalvas permissivas. |
| Llama 3.2 1B | 2 / 3 / 5 GB | Llama Community | Para experimentos rápidos em placas pequenas. |
| Mistral 7B | 6 / 8 / 14 GB | Apache 2.0 | Comparável ao Qwen 7B, modelo de chat diferente. |
| Llama-3.1-8B | 9 / 11 / 18 GB | Llama-3.1-Community | 8B QLoRA, contexto nativo de 128K (a cláusula de >700M-MAU requer uma licença Meta separada). |
| **Qwen2.5-14B** | 18,7 GiB no pico com 4096 de contexto (QLoRA) | Apache 2.0 | **O modelo padrão para uso diário em 32 GB.** rank/alpha 32, AdamW de 8 bits. Os pesos de 4 bits sozinhos têm cerca de 8,5 GB; uma janela completa de 4096 tokens precisa do restante. |
| Mistral-Small-24B | 22,8 GiB no pico com 4096 de contexto (QLoRA) | Apache 2.0 | 24B QLoRA em uma placa de 32 GB. Os pesos de 4 bits sozinhos têm cerca de 18 GB. |
| **Qwen2.5-32B** | 26,0 GiB no pico com 2048 de contexto (QLoRA) | Apache 2.0 | **O máximo que cabe em 32 GB.** Cabe em `max_len 2048` com AdamW de 8 bits. |

Outros modelos geralmente funcionam; as linhas acima são as predefinições selecionadas — a faixa de 14B a 32B é ajustada com QLoRA para uma placa de 32 GB (o limite medido). Para as predefinições até 8B, as três figuras são estimativas QLoRA para os tamanhos de adaptador `fast`, `balanced` e `quality` com lote 1 e exemplos de 2048 tokens; elas tendem a ser altas, e exemplos mais curtos usam menos. O tamanho do adaptador é escolhido para sua placa: `--lora-preset auto` (o padrão) usa o maior de `quality` (rank 256 em cada camada linear, conforme Biderman 2024 e Thinking Machines 2025), `balanced` (rank 64 em cada camada linear) e `fast` (rank 16 em duas camadas por bloco) que cabe na memória livre da sua GPU. Especifique um para forçar. `backprop estimate-vram` imprime a estimativa para qualquer modelo e configurações.

## Solução de problemas

Um índice resumido das falhas mais comuns na primeira execução. O índice reverso completo está na [página do guia de solução de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Para uma análise aprofundada de driver / VRAM / precisão mista, consulte a [página de solução de problemas do CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Sintoma | Código de erro | Correção |
|---|---|---|
| A GPU fica sem memória durante o treinamento. | `RUNTIME_GPU_OOM` | Automático — O retropropagamento reduz pela metade o tamanho do lote e tenta novamente até 3 vezes. Para desativar: `Trainer(oom_recovery=False)`. Para forçar um tamanho menor: `--batch-size 1`. |
| O HuggingFace retorna 401 / "modelo não encontrado". | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` e tente novamente. Para erros de digitação, copie o ID exato de <https://huggingface.co/models>. |
| `register_with_ollama`, conexão recusada. | `DEP_OLLAMA_REGISTRATION_FAILED` | Inicie o daemon: `ollama serve`. Instale a partir de <https://ollama.com>. Pode ser repetido. |
| O disco está cheio durante a salvaguarda do checkpoint. | `STATE_CHECKPOINT_INVALID` | As gravações atômicas deixam um diretório `.partial` em caso de falha — seguro para excluir. O checkpoint anterior válido permanece intacto. |
| O treinamento é pausado devido ao superaquecimento da GPU. | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automático — O retropropagamento pausa no limite de temperatura e retoma quando a GPU esfria. Melhore o fluxo de ar se isso continuar acontecendo. |
| `backprop ui --share` rejeitado. | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Passe `--auth user:password` ou use o encaminhamento de porta SSH (veja [Interface do Usuário da Web](#web-ui)). |
| A exportação GGUF falhou na primeira tentativa. | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; no Windows, você também precisa das Ferramentas de Compilação do Visual C++ + CMake. |

## Relatando bugs

Quando algo falha, o Backpropagate imprime uma linha no início, como `run_started run_id=<uuid>`, e associa o mesmo ID a cada linha de log, a cada checkpoint e a cada entrada do Weights & Biases. **Inclua o `run_id` em qualquer relatório de bug** — isso permite que um mantenedor correlacione tudo para essa execução específica.

Um bom relatório de bug inclui:

1. **O `run_id`** — o UUID impresso no início. Um UUID permite que um mantenedor correlacione cada linha de log, cada checkpoint e cada entrada do Weights & Biases para essa execução específica.
2. **O código de erro** — a linha `[CODE_NAME]: message` em stderr. Consulte [códigos de erro](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) para o catálogo de códigos estáveis.
3. **O rastreamento editado.** O stderr é automaticamente editado no modo não detalhado (tokens Bearer, `sk-*`, `hf_*`, chaves AWS, os pares `password=` / `token=` / `api_key=` são removidos — seguro para colar). Para o rastreamento completo e não editado, execute novamente com `BACKPROPAGATE_DEBUG=1` (ou `--verbose`); revise antes de postar.
4. **A saída `backprop info`.** Um comando imprime Python / PyTorch / CUDA / modelo GPU / VRAM / SO / extras instalados — tudo o que o mantenedor precisa para identificar uma regressão específica da plataforma.

O [modelo de relatório de bug](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) solicita explicitamente cada um desses itens para que a triagem seja rápida. Perguntas, ideias ou tópicos do tipo "é isso esperado?" devem ser postados em [Discussões do GitHub](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Problemas de segurança devem ser relatados em particular por meio do formulário [Aviso de Segurança do GitHub](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) — consulte [SECURITY.md](SECURITY.md) para obter a política e os prazos de resposta.

## Privacidade

Todo o treinamento ocorre localmente em sua GPU. O Backpropagate não faz solicitações de rede, exceto para baixar modelos do HuggingFace (o que você inicia). Sem telemetria, sem dependência da nuvem.

## Referências

Os valores padrão do Backpropagate e o modo de treinamento de várias execuções são baseados em pesquisas recentes. Se você estiver interessado nas técnicas subjacentes:

- **Hu et al. 2021.** *LoRA: Adaptação de Baixa Classificação de Grandes Modelos de Linguagem.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — o artigo fundamental que introduz o LoRA, que é como o Backpropagate treina adaptadores de forma eficiente.
- **Biderman et al. 2024.** *LoRA aprende menos e esquece menos.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidências empíricas de que o LoRA com classificação 256 e alvos totalmente lineares corresponde à qualidade do ajuste fino completo na maioria das tarefas pós-treinamento em 67% do poder computacional. Impulsiona a configuração padrão do LoRA v1.3 do Backpropagate.
- **Thinking Machines 2025.** *LoRA sem arrependimentos.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora) — o acompanhamento prático que identifica a correção de 10× na taxa de aprendizado em relação ao ajuste fino completo necessária em alta classificação LoRA.
- **Kirkpatrick et al. 2017.** *Superando o esquecimento catastrófico em redes neurais.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — a caracterização original de por que as redes neurais "esquecem" o treinamento anterior quando você ajusta em novos dados (EWC — Consolidação de Peso Elástico).
- **Wang et al. 2023.** *Aprendizado de Subespaço Ortogonal para Aprendizado Contínuo de Modelos de Linguagem.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, uma abordagem anterior para usar o LoRA para aprendizado contínuo, restringindo os novos adaptadores a subespaços ortogonais.
- **Yadav et al. 2023.** *TIES-Merging: Resolvendo a Interferência ao Mesclar Modelos.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — uma técnica fundamental para mesclar vários modelos ajustados sem interferência.
- **Qiao & Mahdavi 2025.** *Mesclar antes de esquecer: um aprendizado contínuo de LoRA único por meio de mesclagem contínua.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — o algoritmo específico que o mesclador de várias execuções do Backpropagate implementa. Um preprint de dezembro de 2025; o Backpropagate é o primeiro adotante conhecido do artigo.

## Licença

MIT — consulte [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
