<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
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

# Treine um adaptador. Envie-o para o Ollama. Siga em frente

Backpropagate é uma biblioteca Python para ajustar modelos de linguagem grandes em uma única GPU. Três linhas de código são suficientes para treinar um modelo de 7 bilhões de parâmetros em uma placa de 16 GB. Um comando adicional permite exportá-lo para o Ollama, para que possa usar o comando `ollama run` com o modelo ajustado. Funciona perfeitamente no Windows.

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

É só isso. Não existe um ficheiro de configuração YAML. Não existe um procedimento específico para iniciar o processo com o comando `accelerate launch`. Não existe um tutorial separado para converter o ficheiro para o formato GGUF. Se tiver uma GPU CUDA e um ficheiro JSONL com os seus dados de treino, faltam apenas três linhas de código para obter um modelo ajustado e funcional.

## Instalar

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

Se desejar as funcionalidades opcionais, substitua a instalação atual por uma destas:

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

Prefere o Docker? O comando `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` também funciona. As imagens são disponibilizadas para as arquiteturas `linux/amd64` e `linux/arm64`, o que significa que os utilizadores de dispositivos Apple Silicon e ARM Linux recebem uma imagem nativa. Um ficheiro `compose.yaml` padrão para a aplicação "Interface do utilizador num contentor" encontra-se na raiz do repositório; o comando `docker compose up` ativa a interface do utilizador na web em `http://localhost:7860`, com um volume persistente montado em `~/.backpropagate`.

## Qual é o lugar do algoritmo de retropropagação nesse contexto?

Existem várias bibliotecas úteis para otimizar modelos de linguagem grandes (LLMs). Cada uma delas se destaca em áreas diferentes:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — se você prefere configurações em YAML e deseja ter acesso a uma comunidade com diversas receitas para usar como modelo.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — se você precisa de DPO/PPO/RLHF e de uma interface gráfica web.
- **[Unsloth](https://github.com/unslothai/unsloth)** — se você precisa do treinamento mais rápido possível e utiliza um dos modelos suportados.
- **[torchtune](https://github.com/pytorch/torchtune)** — se você deseja ter acesso às receitas nativas do PyTorch, criadas pela Meta, que podem ser editadas.

O «backpropagation» é a funcionalidade que faltava: uma API Python de 3 linhas para utilizadores individuais que usam uma única GPU e que pretendem treinar um adaptador e disponibilizá-lo. Sem YAML, sem interface gráfica, sem RL online (PPO/GRPO), sem configuração multi-nó. Apenas o ciclo de que todos realmente precisam e o passo de exportação que causa problemas.

Se você experimentou uma das bibliotecas mencionadas acima e teve dificuldades com a configuração, ou encontrou uma lacuna na compatibilidade com determinados modelos, ou preferia configurações padrão otimizadas para o Windows, então o Backpropagate é a solução ideal para você.

## Quais os aspetos que pode ajustar numa GPU de 16 GB para uso doméstico?

Aqui estão os valores práticos de desempenho que se podem esperar numa placa de 16 GB (RTX 4080 / 5080 / 4070 Ti Super):

| Modelo | Método | Estado/Situação |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | Confortável. Comprimento total da sequência adequado, com margem de sobra. |
| SmolLM3-3B / Qwen2.5-3B / Llama-3.2-3B / Llama-3.2-1B | `mode="full"` (ajuste fino completo) | v1.4 — utilize a opção `--mode=full` em `backprop train` ou `Trainer(..., mode="full")`. Carrega os pesos em precisão total (bf16) — sem pesos de 4 bits, sem adaptadores; o uso de «gradient checkpointing» e do otimizador Adam de 8 bits com paginação mantém o consumo de memória abaixo de 16 GB. |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | Padrão. Aproximadamente 7-8 GB. Configurações predefinidas padrão do Backpropagate. |
| Llama-3 13B | QLoRA + compactação de amostras | Ajustado, mas funciona. Use sequências mais curtas. |
| Mixtral 8x7B (47 mil milhões de parâmetros no total) | — | Fora do âmbito de aplicação — a versão de 2 bits (AQLM/QuIP#) quebra o contrato de compatibilidade entre o adaptador e a exportação para o formato GGUF, pelo que foi descontinuada no [resumo da trajetória da versão 1.5](docs/V1_5_BRIEF.md). Em uma placa de 16 GB, use uma base de ≤ 8 GB. |

`mode="full"` permite modelos com até **4 bilhões de parâmetros**. Os quatro modelos pré-definidos na linha «full-FT» acima são modelos genuínos de aproximadamente 3 bilhões de parâmetros (contagem real de parâmetros de 3,08 a 3,24 bilhões) e cabem em uma placa de 16 GB. A classe de 3,8 a 4 bilhões de parâmetros (Phi-4-mini-3.8B, Qwen-3.5-4B) também é aceita, mas requer uma placa de **24 GB ou mais** para o ajuste fino completo — apenas os pesos e gradientes ocupam cerca de 16 GB antes do otimizador e das ativações —, portanto, em uma placa de 16 GB, use `mode="lora"` para esses modelos (eles estão na linha LoRA). Modelos com mais de 4 bilhões de parâmetros são rejeitados com a mensagem `RUNTIME_FULL_FT_MODEL_TOO_LARGE`.

A quantização de 2 bits (AQLM / QuIP#) está **fora do escopo**. Foi considerada para a versão 1.4, mas foi removida no [resumo da trajetória da versão 1.5](docs/V1_5_BRIEF.md): uma base de 2 bits não pode ser facilmente reintegrada em pesos de precisão total, o que prejudica o contrato de exportação do Backpropagate (adaptador que pode ser combinado → GGUF → Ollama), que é o objetivo principal do processo. Em vez disso, o Backpropagate oferece as seguintes opções: o caminho de computação **FP8 da versão 1.5** (`--fp8`, Blackwell/Hopper) e `mode="full"` para modelos de até 4 bilhões de parâmetros — ambos permanecem combináveis e exportáveis.

Para os modelos 3B e inferiores, é possível realizar um ajuste fino completo (e não apenas LoRA) com 16 GB de memória, e esta funcionalidade já está disponível na versão 1.4, com o parâmetro `mode="full"`. Para ativá-la, utilize `Trainer(..., mode="full")` ou `backprop train --mode=full --model phi-4-mini-3.8b`. Um mecanismo de segurança impede o uso desta funcionalidade para modelos com mais de 4B, exibindo a mensagem `RUNTIME_FULL_FT_MODEL_TOO_LARGE`, e sugere o uso de LoRA juntamente com as configurações para modelos inferiores a 4B como alternativas. Consulte a [página completa do manual de ajuste fino](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/) para obter informações detalhadas sobre a configuração e uma comparação da qualidade, baseada nos estudos de Biderman (2024) e Thinking Machines (2025). Para modelos com 7B ou mais, o ajuste fino completo requer uma GPU com 24 GB ou mais de memória — considere alugar uma GPU A100 na nuvem ou utilize LoRA, já que pesquisas recentes mostram que esta técnica oferece resultados semelhantes ao ajuste fino completo na maioria das tarefas de pós-treinamento (consulte a [seção "O que o Backpropagate não é para"](#what-backpropagate-is-not-for) para obter referências).

## Para que NÃO serve o algoritmo de retropropagação

Se o seu caso de uso for um dos seguintes, será mais vantajoso utilizar uma biblioteca diferente — a Backpropagate não é a escolha certa e tentar fazê-la funcionar custará mais do que simplesmente optar pela ferramenta adequada. Ler esta secção antes de começar evita o ciclo de instalação e tentativa frustrante:

- **Ajuste fino de parâmetros completos de modelos 7B+** — Backpropagate usa LoRA/QLoRA, que treina um pequeno adaptador em vez de atualizar todos os pesos. Para modelos de 7B e superiores, o ajuste fino completo requer 24 GB ou mais de memória da GPU e não cabe em uma placa de consumidor de 16 GB. Para modelos de 3B e inferiores, o ajuste fino completo é viável em 16 GB e está disponível na v1.4 como `mode="full"` (passe `Trainer(..., mode="full")` ou `--mode=full` na CLI; um filtro rígido aciona `RUNTIME_FULL_FT_MODEL_TOO_LARGE` para modelos > 4B e nomeia LoRA + os predefinidos de sub-4B como alternativas). Em resumo: pesquisas recentes ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) mostram que o LoRA, com a configuração correta, corresponde à qualidade do ajuste fino completo na maioria das tarefas de pós-treinamento (seguimento de instruções, adaptação de domínio, persona/estilo) em 67% do poder computacional — portanto, para o trabalho que a maioria dos operadores realmente deseja, você não perde nada ao optar por LoRA. `mode="full"` existe para os casos em que você mediu uma diferença de qualidade e decidiu gastar o poder computacional extra. Se você realmente precisar de um ajuste fino completo de um modelo de 7B+, use HuggingFace `transformers.Trainer` diretamente em uma placa de 24 GB ou mais.
- **RL online — PPO / GRPO / RLVR** — Backpropagate faz um ajuste fino SFT de estágio único mais ajuste de preferência sem referência (ORPO está disponível na v1.5; SimPO/KTO estão planejados). O que ele *não* faz é aprendizado por reforço online — PPO, GRPO ou RLVR —, que requer um modelo de recompensa ou um loop de geração e pontuação no topo da etapa de treinamento. Para esses, use TRL diretamente ou LLaMA-Factory. (O ajuste de preferência sem referência se encaixa no envelope de estágio único porque não há um modelo de referência separado para manter na memória; veja a nota sobre ORPO em [Início Rápido](#início-rápido).)
- **Treinamento multi-nó** — GPU única em uma máquina. Multi-GPU em uma máquina funciona (via `accelerate launch`), mas não é oficialmente suportado.
- **Treinamento macOS no ambiente CUDA** — Apple Silicon não tem CUDA, então o caminho CUDA tem que ser executado em uma máquina Linux ou Windows com uma GPU NVIDIA. Você ainda pode executar o modelo treinado em um Mac via Ollama. **Novo na v1.5:** um ambiente MLX experimental (`--backend mlx`) treina um adaptador LoRA nativamente no Apple Silicon — veja [Apple Silicon (MLX)](#apple-silicon-mlx--experimental-v15). É apenas LoRA-SFT e foi construído, mas ainda não foi totalmente testado em silício real, então, para qualquer coisa além de um LoRA SFT (ORPO, ajuste fino completo, FP8, execução múltipla), você ainda deseja o ambiente CUDA.
- **Qualquer coisa fora das famílias de modelos testadas** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Outros modelos geralmente funcionam, mas não estão fixados no CI.

Se você precisar de alguma dessas coisas, use uma das bibliotecas listadas acima. Elas são melhores para isso.

## O que Backpropagate oferece

Quatro coisas, em uma única instalação:

**1. Uma API real de 3 linhas que é executada sem um arquivo de configuração.**
O snippet no topo deste README é executado de ponta a ponta. Sem `accelerate config`, sem YAML, sem substituições Hydra. Apenas `Trainer(model).train(data)` e você tem um ajuste fino.

**2. Windows que realmente funciona.**
A maioria das bibliotecas de aprendizado de máquina trata o Windows como uma reflexão tardia. Backpropagate é testado em primeira classe no Windows + RTX 5080. A biblioteca lida com as peculiaridades de tempo de execução para você — ela sabe como pré-tokenizar seus dados para que o processamento paralelo do Windows não falhe, desativa automaticamente o xformers em placas RTX 40/50 onde isso causaria problemas e escolhe configurações de carregador de dados que não causam erros. Você não precisa saber nada disso. Ele simplesmente funciona.

**3. Projetado para execuções não supervisionadas.**
O treinamento leva horas. Você não quer ficar de olho nele. Backpropagate é projetado para ser deixado em execução:

- Se você ficar sem memória da GPU, ele reduz automaticamente pela metade o tamanho do lote e tenta novamente — até três vezes. Sem ajuste manual.
- Se sua GPU ficar muito quente, ela pausa até que as coisas esfriem e, em seguida, continua.
- Cada ponto de verificação é gravado atomicamente — se o seu laptop travar durante a gravação, o ponto de verificação anterior e bom ainda estará intacto.
- Cada execução de treinamento recebe um ID exclusivo que é carimbado em cada linha de log, cada ponto de verificação e cada entrada do Weights & Biases. Se algo der errado, um ID permite que um mantenedor correlacione tudo.
- Os erros vêm com códigos estáveis (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, etc.) para que você possa pesquisar em seus logs e no [guia de solução de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para encontrar a solução. As falhas específicas do CUDA têm uma [página de solução de problemas do CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) dedicada.

**4. Um comando do adaptador treinado ao `ollama run`.**
Muitas bibliotecas treinam um modelo. Poucas delas saem do seu caminho quando você deseja realmente usá-lo. Backpropagate exporta para GGUF (o formato que o Ollama usa) e registra um modelo Ollama em um único comando. Você passa de "treinamento concluído" para "posso conversar com meu ajuste fino" em cerca de 30 segundos.

## Início Rápido

O repositório inclui um pequeno conjunto de dados de exemplo para que o snippet do topo deste README seja executado em uma instalação limpa:

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

Isso treina um adaptador Qwen 2.5 7B em 5 conversas curtas no formato ShareGPT e, em seguida, exporta o resultado para GGUF. Para seus próprios dados, formate seu JSONL com um exemplo por linha:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Os formatos Alpaca (`instruction` / `output`), OpenAI chat (`messages`) e texto bruto também funcionam — Backpropagate detecta automaticamente o formato.

### Ajuste de preferências (ORPO, SimPO, KTO)

Novo na v1.5: treine com base em preferências em vez de demonstrações simples. ORPO não requer referência e é de estágio único — ele incorpora o sinal de preferência na etapa de ajuste fino SFT, portanto, não há um modelo de recompensa ou referência separado e a forma de 3 linhas permanece inalterada. Passe `--method orpo` (CLI) ou `method="orpo"` (Python) e forneça um conjunto de dados de linhas `{prompt, chosen, rejected}` (ou apenas `{chosen, rejected}`):

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

A taxa de aprendizado padrão é automaticamente reduzida para `8e-6` para ORPO (a perda é mais acentuada do que no SFT simples); ajuste `--orpo-beta` (padrão `0.1`) para ponderar a penalidade da razão de chances. ORPO é apenas em `mode="lora"`.

**Novo na v1.6 — SimPO e KTO.** `--method simpo` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) não requer referência, utiliza uma recompensa normalizada pelo comprimento e usa os mesmos dados pareados `{prompt, chosen, rejected}` que o ORPO (`--simpo-beta`, `--simpo-gamma`). `--method kto` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)) usa dados **não pareados** `{prompt, completion, label}` — avaliações individuais de "gostei/não gostei" — para a grande classe de feedback que não é um conjunto curado de pares A/B; ele equilibra automaticamente os pesos desejáveis/indesejáveis da perda com base nas contagens dos seus rótulos. Ambos são apenas em `mode="lora"` e permanecem no ambiente SFT de GPU única (sem modelo de referência separado). Consulte o [manual de ajuste de preferências](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) para saber qual usar. Para RL online (PPO/GRPO), consulte [O que o Backpropagate NÃO é](#what-backpropagate-is-not-for).

### Rastreamento de raciocínio SFT (destilação R1)

Novo na v1.5: destile um modelo de raciocínio de forma fácil. Passe `--reasoning-trace` (CLI) ou `Trainer(..., reasoning_trace=True)` (Python) e forneça rastreamentos que mantenham uma cadeia de pensamento `<think>...</think>` dentro da interação do assistente — a metade pura de SFT de [DeepSeek-R1](https://arxiv.org/abs/2501.12948), sem necessidade de RL. Backpropagate mantém `<think>` no alvo de treinamento, remove rastreamentos vazios/muito longos (filtragem do comprimento do rastreamento) e aumenta o `max_seq_length` padrão para 8192 para o CoT mais longo. Fundamentalmente, `<think>` permanece como **texto simples** — sem tokens especiais, sem redimensionamento de incorporação — para que o GGUF mesclado ainda seja exportado para o Ollama como qualquer outro ajuste fino. Apenas SFT. Consulte a [receita de rastreamento de raciocínio](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) para o formato do conjunto de dados e o intervalo de tokens ajustável.

### Apple Silicon (MLX) — experimental, v1.5

Novo na v1.5: **uma API, dois caminhos.** CUDA permanece como o backend canônico e verificado; MLX é um segundo caminho que treina em um Mac da série M por meio do conjunto de ferramentas [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) da Apple (memória unificada, sem CUDA). O mesmo formato de 3 linhas seleciona o caminho por hardware — `backend='auto'` (o padrão) direciona para CUDA em NVIDIA e para MLX em Apple Silicon, para que as configurações CUDA existentes sejam idênticas em termos de bytes:

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

Na v1.5, o caminho MLX é **apenas LoRA SFT** — sem ORPO, sem FP8, sem `mode='full'`, sem execução múltipla em MLX ainda (cada um é rejeitado com `CONFIG_INVALID_SETTING`; use `backend='cuda'`/`'auto'` em uma máquina NVIDIA para esses). O adaptador resultante é apenas safetensors e é exportado para o Ollama pelo mesmo caminho do caminho CUDA.

> ⚠️ **Situação atual:** a versão MLX incluída na v1.5 foi **construída + testada em unidade (simulada)**, mas **ainda não foi verificada em testes reais com Apple Silicon** — `mlx-lm` é exclusivo para Apple e não pôde ser executado no sistema NVIDIA onde este código foi desenvolvido. Considere como experimental — a mesma abordagem que o caminho FP8 teve na v1.5 (FP8 passou para testes reais em Blackwell na v1.6; MLX ainda precisa passar por esse teste em silício real) — e, por favor, [relate anomalias](#reporting-bugs) assim que for executado em um Mac da série M. Forçar `--backend mlx` em um host não Apple gera o erro `CONFIG_INVALID_SETTING`; a ausência de uma ferramenta `mlx_lm` em um Mac gera `DEP_MLX_UNAVAILABLE`.

Para fluxos de trabalho de ponta a ponta mais abrangentes (ajuste fino e envio para o HF-Hub, retomada após estouro de memória, SLAO de execução múltipla em uma campanha longa, etc.), consulte a [página de receitas do manual](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interface de usuário da web (opcional)

Se você preferir clicar em vez de digitar Python, instale o extra da interface do usuário e inicie:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Uma interface da web local é aberta em `http://localhost:7862` para navegar pelos conjuntos de dados, validar formatos e montar uma configuração de treinamento visualmente. O treinamento em si é executado por meio de `backprop train` (o treinamento orientado pela interface do usuário está no roteiro — o botão Iniciar exibe atualmente essa observação). A interface do usuário é apenas local por padrão. Para expô-la a outros dispositivos, consulte [Interface do usuário da web](#web-ui) abaixo para o contrato de segurança `--share` + `--auth`.

## Treinamento de execução múltipla

Se você deseja ajustar os parâmetros de forma incremental em vários conjuntos de dados — digamos que você obtenha novos dados de treinamento a cada semana e queira adicioná-los sem esquecer o que aprendeu antes — o modo `multi_run` do Backpropagate é para você:

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

Um treinamento de 5 execuções que falha na execução 4 pode ser recuperado. Cada sessão de execução múltipla grava seu ID de execução no histórico e no manifesto de pontos de verificação no disco, para que retomar de onde você parou seja um único comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

O comportamento padrão de `backprop multi-run` (sem `--resume`) detecta automaticamente uma entrada em andamento no mesmo diretório de saída e a continua. Para forçar um novo início, aponte para um novo diretório de saída.

## Histórico de treinamento

Cada invocação de `backprop train` e `backprop multi-run` registra uma linha em `<output>/run_history.json` — modelo usado, conjunto de dados, hiperparâmetros, status, perda final, histórico de perdas. Você pode listar e inspecionar execuções anteriores:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Rastreamento de experimentos

Backpropagate detecta automaticamente os rastreadores de experimentos instalados (Weights & Biases, TensorBoard, MLflow) e os conecta. Se `wandb` estiver instalado e você estiver conectado, cada execução registrará automaticamente no W&B com um nome de execução que corresponda ao ID de execução no disco — para que você possa pesquisar no W&B, em seus logs e em `run_history.json` usando um único identificador.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Substitua com `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` ou `Trainer(report_to="none")` para optar por não participar.

## Interface do usuário da web

A interface da web Reflex é opcional — instale com `pipx install "backpropagate[ui]"` e inicie:

```bash
backprop ui --port 7862
```

A interface do usuário é executada localmente em `http://localhost:7862`. Atualmente, ela cobre a metade **navegar / validar / configurar** do fluxo de trabalho — aponte-a para um conjunto de dados, verifique o formato e as estatísticas detectados automaticamente, escolha um modelo e monte uma configuração de execução. **O lançamento da execução é feito pela CLI** (`backprop train` / `backprop multi-run`); o botão Iniciar na interface do usuário exibe uma observação apontando para lá. O treinamento orientado pela interface do usuário é um acompanhamento planejado — até então, a interface do usuário é o ponto de entrada e a CLI é o gatilho.

Para expor o modelo a outros dispositivos (outras pessoas na sua rede, um URL público, etc.), você deve combinar `--share` (ou `--host`) com `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` sem `--auth` resulta em um erro. O motivo: `--share` publica um URL que qualquer pessoa na internet pode acessar e, sem autenticação, isso significa que qualquer pessoa pode executar o seu pipeline de treinamento e ler o seu token HuggingFace. Não há opção para desativar isso — se você não quiser definir credenciais, use o redirecionamento de porta SSH:

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

Consulte [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) para obter o modelo completo de ameaças.

As operações de escrita no sistema de arquivos a partir da interface do usuário são restritas a um único diretório:

- Padrão: `~/.backpropagate/ui-outputs`
- Substituição: defina `BACKPROPAGATE_UI__OUTPUT_DIR=/caminho/que/você/deseja`
- A substituição é validada por uma lista de permissões — caminhos do sistema ou de credenciais (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) são rejeitados.

## Notas da plataforma

**Requisitos:** Python 3.10+ · GPU CUDA (8 GB+ de VRAM) · PyTorch 2.0+

O Python 3.10 é suportado até, pelo menos, a v1.6; seu suporte oficial termina em outubro de 2026 e está programado para ser removido na primeira versão após essa data. Para novas instalações, prefira o Python 3.11 ou 3.12 — o 3.11 é a versão mais testada.

O Backpropagate lida com as peculiaridades de tempo de execução do treinamento em diferentes plataformas, mas não pode corrigir problemas de tempo de instalação. Os dois mais comuns são:

- **Pacote CUDA incorreto.** O PyTorch é publicado com um binário por versão do CUDA. Se você escolher o pacote errado, obterá silenciosamente o PyTorch apenas para CPU e o treinamento será impossivelmente lento. Use o seletor de pacotes em <https://pytorch.org/get-started/locally/> para o seu driver. Execute `nvidia-smi` para ver a versão do seu driver/CUDA.
- **Windows + exportação GGUF.** O extra `[export]` cria o `llama-cpp-python` a partir do código-fonte, o que requer o Visual Studio Build Tools (componente C++) e o CMake.

**macOS:** o suporte para CUDA não é fornecido (sem CUDA) — um `trainer.train()` com CUDA gera `DEP_GPU_NOT_AVAILABLE` e você pode executar o adaptador treinado em um Mac via Ollama. **Novo na v1.5:** um trilho experimental MLX (`--backend mlx`, `pip install 'backpropagate[mlx]'`) treina um adaptador LoRA nativamente no Apple Silicon via `mlx_lm.lora` — apenas LoRA SFT e construído + testado, mas ainda não verificado em silício real (veja [Apple Silicon (MLX)](#apple-silicon-mlx--experimental-v15)). Para o caminho CUDA ou para ORPO / ajuste fino completo / FP8 / execução múltipla, use uma máquina Linux ou Windows com CUDA.

Consulte a [página do guia de solução de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para obter o guia completo de correção de instalação e a [página dedicada de solução de problemas do CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) para problemas de driver / VRAM / xformers / bf16 vs. fp16.

## CLI

Cada API Python tem um espelho CLI:

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

Referência completa em [na página do guia CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), ou `backprop <subcomando> --help`.

## Configuração

Cada configuração pode ser substituída com uma variável de ambiente usando o prefixo `BACKPROPAGATE_`:

| Variável | Padrão | Notas |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Forçar logs JSON ou de console |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modelo padrão |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Taxa de aprendizado |
| `BACKPROPAGATE_LORA__R` | `256` | Rank LoRA (padrão da v1.3; use `--lora-preset=fast` para o padrão da v1.2.x de 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sandbox do sistema de arquivos da interface do usuário |

Chaves aninhadas usam sublinhado duplo (`MODEL__NAME`, não `MODEL_NAME`). A referência completa está em [na página do guia de variáveis de ambiente](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Predefinições de modelo

| Predefinição | VRAM | Licença | Notas |
|---|---|---|---|
| Qwen-3.5-4B | ~8GB | Apache 2.0 | Padrão recomendado para modelos menores que 5B. Melhor qualidade neste tamanho. |
| Phi-4-mini-3.8B | ~8GB | MIT | Forte em raciocínio / matemática / código. Licença estritamente limpa. |
| SmolLM3-3B | ~6GB | Apache 2.0 | Receita totalmente aberta. Contexto nativo de 64K. |
| Qwen 2.5 7B | ~12GB | Apache 2.0 | Padrão existente. Melhor qualidade das predefinições 7B legadas. |
| Qwen 2.5 3B | ~8GB | Qwen-Research | ⚠ licença de pesquisa — consulte os termos de licença do Qwen antes do uso comercial. |
| Llama 3.2 3B | ~8GB | Llama Community | Alternativa sólida ao Qwen 3B com ressalvas permissivas. |
| Llama 3.2 1B | ~6GB | Llama Community | Para experimentos rápidos em placas pequenas. |
| Mistral 7B | ~12GB | Apache 2.0 | Comparável ao Qwen 7B, modelo de chat diferente. |

Outros modelos geralmente funcionam, mas apenas esses oito são fixados no CI. Use `--lora-preset=quality` (padrão) para alvos de rank-256 / totalmente lineares, conforme Biderman 2024 + Thinking Machines 2025, ou `--lora-preset=fast` para o alvo legado de rank-16 / q+v, se você precisar do footprint da v1.2.x.

## Solução de problemas

Um breve índice das falhas mais comuns na primeira execução. O índice reverso completo está em [na página do guia de solução de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Para uma análise aprofundada de driver / VRAM / precisão mista, consulte a [página de solução de problemas do CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Sintoma | Código de erro | Correção |
|---|---|---|
| A GPU fica sem memória durante o treinamento. | `RUNTIME_GPU_OOM` | Automático — O retropropagação reduz pela metade o tamanho do lote e tenta novamente até 3 vezes. Para desativar: `Trainer(oom_recovery=False)`. Para forçar um tamanho menor: `--batch-size 1`. |
| O HuggingFace retorna 401 / "modelo não encontrado". | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` e tente novamente. Para erros de digitação, copie o ID exato de <https://huggingface.co/models>. |
| `register_with_ollama` conexão recusada. | `DEP_OLLAMA_REGISTRATION_FAILED` | Inicie o daemon: `ollama serve`. Instale a partir de <https://ollama.com>. Pode ser repetido. |
| Disco cheio durante a salvaguarda do ponto de verificação. | `STATE_CHECKPOINT_INVALID` | As gravações atômicas deixam um diretório `.partial` em caso de falha — seguro para excluir. O ponto de verificação anterior válido permanece intacto. |
| O treinamento foi pausado devido ao superaquecimento da GPU. | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automático — O retropropagação pausa no limite de temperatura e retoma quando a GPU esfria. Melhore o fluxo de ar se isso continuar acontecendo. |
| `backprop ui --share` rejeitado. | `RUNTIME_UI_AUTH_NOT_ENFORCED` | Passe `--auth user:password` ou use o encaminhamento de porta SSH (veja [Interface do Usuário da Web](#interface-do-usuario-da-web)). |
| A exportação GGUF falhou na primeira tentativa. | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; no Windows, você também precisa das Ferramentas de Compilação C++ e do CMake. |

## Relatando bugs

Quando algo falha, o Backpropagate imprime uma linha no início, como `run_started run_id=<uuid>`, e associa o mesmo ID a cada linha de log, a cada ponto de verificação e a cada entrada de Weights & Biases. **Inclua o `run_id` em qualquer relatório de bug** — isso permite que um mantenedor correlacione tudo para essa execução específica.

Um bom relatório de bug inclui:

1. **O `run_id`** — o UUID impresso no início. Um UUID permite que um mantenedor correlacione cada linha de log, cada ponto de verificação e cada entrada de Weights & Biases para essa execução específica.
2. **O código de erro** — a linha `[CODE_NAME]: message` em stderr. Consulte [códigos de erro](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) para o catálogo de códigos estáveis.
3. **O rastreamento editado.** O stderr é automaticamente editado no modo não detalhado (tokens Bearer, `sk-*`, `hf_*`, chaves AWS, pares `password=` / `token=` / `api_key=` são removidos) — seguro para colar. Para o rastreamento completo e não editado, execute novamente com `BACKPROPAGATE_DEBUG=1` (ou `--verbose`); revise antes de postar.
4. **A saída de `backprop info`.** Um comando imprime Python / PyTorch / CUDA / modelo de GPU / VRAM / SO / extras instalados — tudo o que o mantenedor precisa para identificar uma regressão específica da plataforma.

O [modelo de relatório de bug](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) solicita explicitamente cada um desses itens para que a triagem avance rapidamente. Perguntas, ideias ou tópicos do tipo "isso é esperado?" devem ser postados em [Discussões do GitHub](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Problemas de segurança devem ser relatados em particular por meio do formulário [Aviso de Segurança do GitHub](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) — consulte [SECURITY.md](SECURITY.md) para obter a política e os prazos de resposta.

## Privacidade

Todo o treinamento ocorre localmente em sua GPU. O Backpropagate não faz solicitações de rede, exceto para baixar modelos do HuggingFace (o que você inicia). Sem telemetria, sem dependência da nuvem.

## Referências

Os padrões e o modo de treinamento de várias execuções do Backpropagate são baseados em pesquisas recentes. Se você estiver interessado nas técnicas subjacentes:

- **Hu et al. 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — o artigo fundamental que introduz o LoRA, que é como o Backpropagate treina adaptadores de forma eficiente.
- **Biderman et al. 2024.** *LoRA Learns Less and Forgets Less.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidências empíricas de que o LoRA com classificação 256 e todos os alvos lineares correspondem à qualidade do ajuste fino completo na maioria das tarefas de pós-treinamento em 67% do poder computacional. Impulsiona a configuração padrão do LoRA v1.3 do Backpropagate.
- **Thinking Machines 2025.** *LoRA Without Regret.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — o acompanhamento prático que identifica a correção de 10× na taxa de aprendizado em relação ao ajuste fino completo necessária em alta classificação LoRA.
- **Kirkpatrick et al. 2017.** *Overcoming catastrophic forgetting in neural networks.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — a caracterização original de por que as redes neurais "esquecem" o treinamento anterior quando você ajusta em novos dados (EWC — Consolidação de Peso Elástico).
- **Wang et al. 2023.** *Orthogonal Subspace Learning for Language Model Continual Learning.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, uma abordagem anterior para usar o LoRA para aprendizado contínuo, restringindo os novos adaptadores a subespaços ortogonais.
- **Yadav et al. 2023.** *TIES-Merging: Resolving Interference When Merging Models.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — uma técnica fundamental para mesclar vários modelos ajustados sem interferência.
- **Qiao & Mahdavi 2025.** *Merge before Forget: A Single LoRA Continual Learning via Continual Merging.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — o algoritmo específico que o mesclador de várias execuções do Backpropagate implementa. Um preprint de dezembro de 2025; o Backpropagate é o primeiro adotante conhecido do artigo.

## Licença

MIT — veja [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
