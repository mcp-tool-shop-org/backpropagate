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

# Treine um adaptador. Envie-o para o Ollama. Próximo

Backpropagate é uma biblioteca Python para ajustar modelos de linguagem grandes em uma única GPU. Três linhas de código treinam um modelo de 7 bilhões de parâmetros em uma placa de 16 GB. Um comando exporta para o Ollama para que você possa executar o ajuste fino com `ollama run`. Funciona perfeitamente no Windows.

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

É isso. Não há arquivo de configuração YAML. Não há "cerimônia" de `accelerate launch`. Não há um tutorial separado para "converter para GGUF". Se você tem uma GPU CUDA e um arquivo JSONL com seus dados de treinamento, você está a três linhas de um ajuste fino funcional.

## Instalação

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

Se você deseja os recursos opcionais, substitua a instalação por um destes:

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

Prefere Docker? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` também funciona. As imagens estão disponíveis para `linux/amd64` e `linux/arm64`, então os usuários de Apple Silicon e ARM Linux têm uma imagem nativa. Um arquivo `compose.yaml` canônico para "UI em um contêiner" está na raiz do repositório — `docker compose up` inicia a interface web em `http://localhost:7860` com um volume persistente `~/.backpropagate` montado.

## Onde o Backpropagate se encaixa

Existem várias bibliotecas boas para ajustar LLMs. Cada uma é ótima em coisas diferentes:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — se você gosta de configurações YAML e quer uma comunidade de receitas para copiar.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — se você quer uma GUI web e suporte integrado para DPO/PPO/RLHF.
- **[Unsloth](https://github.com/unslothai/unsloth)** — se você precisa do treinamento mais rápido possível e está usando uma família de modelos suportada.
- **[torchtune](https://github.com/pytorch/torchtune)** — se você quer as receitas nativas do PyTorch da Meta que pode editar.

Backpropagate é a opção que falta: **uma API Python de 3 linhas para usuários individuais em uma única GPU de consumo que querem treinar um adaptador e enviá-lo.** Sem YAML, sem GUI, sem DPO/PPO, sem nós múltiplos. Apenas o loop que todos realmente precisam e o passo de exportação que atrapalha.

Se você tentou uma das bibliotecas acima e se sentiu frustrado com a "cerimônia" do arquivo de configuração, ou encontrou uma lacuna na família de modelos, ou queria configurações padrão para Windows — Backpropagate é para você.

## O que você pode ajustar em uma GPU de consumo de 16 GB

Aqui está a faixa prática em uma placa de 16 GB (RTX 4080 / 5080 / 4070 Ti Super):

| Modelo | Método | Status |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | Confortável. Comprimento total da sequência, espaço extra. |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | Padrão. ~7-8 GB. Configurações padrão do Backpropagate. |
| Llama-3 13B | QLoRA + empacotamento de amostras | Apertado, mas funciona. Use sequências mais curtas. |
| Mixtral 8x7B (47 bilhões de parâmetros no total) | AQLM 2-bit + LoRA | Experimental na v1.4. O maior modelo que você pode usar em uma placa de 16 GB. |

Para modelos de 3 bilhões de parâmetros ou menores, o ajuste fino completo (não apenas LoRA) é viável em 16 GB e está planejado como uma opção `mode="full"` para a v1.4. Para modelos de 7 bilhões de parâmetros ou mais, o ajuste fino completo requer uma GPU de 24 GB ou mais — considere alugar uma GPU A100 na nuvem, ou use LoRA, que pesquisas recentes mostram que corresponde à qualidade do ajuste fino completo na maioria das tarefas de pós-treinamento (veja a seção "o que Backpropagate não é" para citações).

## O que Backpropagate NÃO é

Uma descrição honesta ajuda a todos. Backpropagate não faz essas coisas, e tentar fazer com que faça seria uma experiência pior do que usar a ferramenta certa:

- **Ajuste fino completo de parâmetros para modelos com mais de 7 bilhões de parâmetros** — O Backpropagate usa LoRA/QLoRA, que treina um adaptador pequeno em vez de atualizar todos os pesos. Para modelos com 7 bilhões de parâmetros ou mais, o ajuste fino completo requer 24 GB ou mais de memória da GPU e não cabe em uma placa de consumo de 16 GB. Para modelos com 3 bilhões de parâmetros ou menos, o ajuste fino completo é viável em 16 GB; uma opção `mode="full"` está planejada para a versão 1.4. Em resumo: pesquisas recentes ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) mostram que o LoRA, com a configuração correta, atinge a mesma qualidade do ajuste fino completo na maioria das tarefas de pós-treinamento (seguir instruções, adaptação de domínio, personalidade/estilo), utilizando 67% menos recursos computacionais. Portanto, para a maioria das tarefas que os usuários desejam realizar, você não perde nada ao usar o LoRA. Se você realmente precisa do ajuste fino completo de um modelo com mais de 7 bilhões de parâmetros, use o `transformers.Trainer` da HuggingFace diretamente em uma placa com 24 GB ou mais.
- **DPO / PPO / GRPO / ajuste de preferências** — O Backpropagate realiza apenas o ajuste fino supervisionado em uma única etapa. Para o aprendizado por preferências, use o TRL diretamente ou o LLaMA-Factory.
- **Treinamento em vários nós** — Apenas uma GPU em uma única máquina. O uso de várias GPUs em uma única máquina funciona (via `accelerate launch`), mas não é oficialmente suportado.
- **Treinamento no macOS** — O Apple Silicon não possui CUDA, portanto, o treinamento deve ser executado em uma máquina Linux ou Windows com uma GPU NVIDIA. Você ainda pode executar o modelo treinado em um Mac usando o Ollama.
- **Qualquer modelo fora das famílias de modelos testados** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B. Outros modelos geralmente funcionam, mas não são testados nos testes de integração contínua (CI).

Se você precisar de alguma dessas funcionalidades, utilize uma das bibliotecas listadas acima. Elas são mais adequadas para isso.

## O que o Backpropagate oferece:

Quatro coisas, em uma única instalação:

**1. Uma API simples de 3 linhas que funciona sem um arquivo de configuração.**
O trecho de código no início deste arquivo README é executado do início ao fim. Não requer `accelerate config`, nem YAML, nem substituições do Hydra. Basta `Trainer(model).train(data)` e você terá um modelo ajustado.

**2. Funcionalidade que realmente funciona no Windows.**
A maioria das bibliotecas de aprendizado de máquina trata o Windows como uma funcionalidade secundária. O Backpropagate é testado e otimizado para Windows + RTX 5080. A biblioteca lida com as peculiaridades do sistema operacional para você — ela sabe como pré-tokenizar seus dados para evitar que o processamento paralelo do Windows cause falhas, desativa automaticamente o xformers em placas RTX 40/50 onde ele pode causar problemas e seleciona configurações do carregador de dados que evitam erros. Você não precisa saber nada disso. Apenas funciona.

**3. Projetado para execuções não supervisionadas.**
O treinamento leva horas. Você não quer ficar monitorando-o constantemente. O Backpropagate foi projetado para ser executado em segundo plano:

- Se você ficar sem memória da GPU, ele automaticamente reduz o tamanho do lote e tenta novamente — até três vezes. Sem necessidade de ajustes manuais.
- Se a sua GPU ficar muito quente, ela pausa até que a temperatura diminua e, em seguida, continua.
- Cada checkpoint é gravado de forma atômica — se o seu laptop travar durante a gravação, o último checkpoint válido ainda estará intacto.
- Cada execução de treinamento recebe um ID exclusivo que é adicionado a cada linha de log, a cada checkpoint e a cada entrada do Weights & Biases. Se algo der errado, um único ID permite que um administrador correlacione todos os elementos.
- Os erros vêm com códigos estáveis (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, etc.), para que você possa pesquisar seus logs e o [guia de solução de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para encontrar a solução. As falhas específicas do CUDA têm uma [página de solução de problemas dedicada para CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

**4. Um único comando para transformar o adaptador treinado em um modelo compatível com `ollama run`.**
Muitas bibliotecas treinam um modelo. Poucas delas facilitam o uso do modelo após o treinamento. O Backpropagate exporta para o formato GGUF (o formato usado pelo Ollama) e registra um modelo no Ollama em um único comando. Você vai da fase de "treinamento concluído" para a fase de "posso conversar com meu modelo ajustado" em cerca de 30 segundos.

## Início Rápido

O repositório inclui um pequeno conjunto de dados de exemplo para que o código no início deste arquivo README seja executado em uma instalação limpa:

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

Isso treina um adaptador Qwen 2.5 7B com 5 conversas curtas no formato ShareGPT, e então exporta o resultado para o formato GGUF. Para seus próprios dados, formate seu arquivo JSONL com um exemplo por linha:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Formatos Alpaca (`instruction` / `output`), OpenAI chat (`messages`) e texto simples também funcionam — o Backpropagate detecta automaticamente o formato.

Para fluxos de trabalho mais completos (ajuste fino e envio para o Hugging Face Hub, retomada após erros de memória, treinamento em várias etapas em um longo período, etc.), consulte a [página de receitas do manual](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/).

### Interface Web (opcional)

Se você preferir clicar em vez de digitar comandos Python, instale a interface gráfica e execute:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

Uma interface web local é aberta em `http://localhost:7862`, onde você pode selecionar um conjunto de dados, escolher um modelo, treinar e exportar. A interface gráfica é local por padrão. Para torná-la acessível a outros dispositivos, consulte a seção [Interface Web](#web-ui) abaixo para obter informações sobre o contrato de segurança `--share` + `--auth`.

## Treinamento em várias etapas

Se você deseja realizar o ajuste fino de forma incremental em vários conjuntos de dados — por exemplo, se você recebe novos dados de treinamento a cada semana e deseja adicioná-los sem esquecer o que aprendeu antes — o modo `multi_run` do Backpropagate é para você:

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

Isso executa cinco etapas de treinamento, mesclando o adaptador entre as etapas de forma a preservar o conhecimento anterior, incorporando novos exemplos. A técnica é baseada em pesquisas recentes sobre aprendizado contínuo — veja as [Referências](#references) no final deste arquivo README.

A versão da linha de comando (CLI):

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## Retomar de um ponto de verificação

Uma etapa de treinamento de 5 tentativas que falha na tentativa 4 pode ser recuperada. Cada sessão de treinamento em várias etapas grava o ID da execução no histórico e no manifesto do ponto de verificação no disco, portanto, para retomar de onde você parou, basta um comando:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

O comportamento padrão do `backprop multi-run` (sem `--resume`) detecta automaticamente uma execução em andamento no mesmo diretório de saída e a continua. Para forçar um novo início, aponte para um diretório de saída diferente.

## Histórico de treinamento

Cada invocação de `backprop train` e `backprop multi-run` registra uma linha em `<output>/run_history.json` — modelo usado, conjunto de dados, hiperparâmetros, status, perda final, histórico de perdas. Você pode listar e inspecionar execuções anteriores:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## Rastreamento de experimentos

O Backpropagate detecta automaticamente os rastreadores de experimentos instalados (Weights & Biases, TensorBoard, MLflow) e os integra. Se o `wandb` estiver instalado e você estiver logado, cada execução registra automaticamente no W&B com um nome de execução que corresponde ao ID da execução no disco — para que você possa pesquisar no W&B, seus logs e em `run_history.json` usando um único identificador.

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

Para desativar essa funcionalidade, use `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` ou `Trainer(report_to="none")`.

## Interface Web

A interface web Reflex é opcional — instale com `pipx install "backpropagate[ui]"` e execute:

```bash
backprop ui --port 7862
```

A interface gráfica é executada localmente em `http://localhost:7862`. Para torná-la acessível a outros dispositivos (outras pessoas na sua rede, um URL público, etc.), você deve combinar `--share` (ou `--host`) com `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` sem `--auth` gera um erro. O motivo: `--share` publica um URL que qualquer pessoa na internet pode acessar, e sem autenticação, isso significa que qualquer pessoa pode controlar seu pipeline de treinamento e ler seu token do Hugging Face. Não há como desativar essa funcionalidade — se você não quiser definir credenciais, use o encaminhamento de porta SSH em vez disso:

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

Consulte [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) para obter o modelo de ameaças completo.

As operações de escrita no sistema de arquivos a partir da interface são restritas a um único diretório:

- Padrão: `~/.backpropagate/ui-outputs`
- Para substituir: defina `BACKPROPAGATE_UI__OUTPUT_DIR=/caminho/desejado`
- A substituição é validada por uma lista de permissões — caminhos do sistema ou de credenciais (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) são rejeitados.

## Observações sobre a plataforma

**Requisitos:** Python 3.10+ · GPU CUDA (8GB+ de VRAM) · PyTorch 2.0+

O Python 3.10 atingirá o fim de vida em outubro de 2026, e o Backpropagate planeja remover o suporte para o 3.10 na versão 1.4. Para novas instalações, prefira o Python 3.11 ou 3.12 — o 3.11 é a versão mais testada.

O Backpropagate lida com as peculiaridades de tempo de execução do treinamento em diferentes plataformas, mas não pode corrigir problemas de instalação. Os dois problemas mais comuns são:

- **Pacote CUDA incorreto.** O PyTorch é publicado com um pacote binário para cada versão do CUDA. Se você escolher o pacote errado, você obterá silenciosamente apenas a versão do PyTorch para CPU, e o treinamento será extremamente lento. Use o seletor de pacotes em <https://pytorch.org/get-started/locally/> para o seu driver. Execute `nvidia-smi` para ver a versão do seu driver / CUDA.
- **Windows + exportação GGUF.** A opção `[export]` compila `llama-cpp-python` a partir do código-fonte, o que requer as Ferramentas de Build do Visual Studio (componente C++) e o CMake.

**macOS:** O treinamento com GPU não é suportado (não há CUDA). Você pode executar o adaptador treinado em um Mac via Ollama, mas `trainer.train()` gera um erro `DEP_GPU_NOT_AVAILABLE`. Use uma máquina Linux ou Windows com CUDA para o próprio treinamento.

Consulte a [página do manual de solução de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) para obter um guia completo de solução de problemas de instalação, e a [página de solução de problemas do CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) dedicada para problemas de driver / VRAM / xformers / bf16-vs-fp16.

## CLI

Cada API do Python tem um equivalente de linha de comando (CLI):

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

Consulte a referência completa na [página do manual do CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/), ou `backprop <subcomando> --help`.

## Configuração

Cada configuração pode ser substituída por uma variável de ambiente, usando o prefixo `BACKPROPAGATE_`:

| Variável | Padrão | Observações |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Forçar logs JSON ou de console |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modelo padrão |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Taxa de aprendizado |
| `BACKPROPAGATE_LORA__R` | `256` | Rank do LoRA (padrão da v1.3; use `--lora-preset=fast` para o padrão da v1.2.x, que é 16) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Sistema de arquivos de sandbox da interface do usuário |

Chaves aninhadas usam dois sublinhados (`MODEL__NAME`, não `MODEL_NAME`). A referência completa está na [página do manual de variáveis de ambiente](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/).

## Modelos Pré-Configurados

| Modelo Pré-Configurado | VRAM | Licença | Observações |
|---|---|---|---|
| Qwen-3.5-4B | ~8GB | Apache 2.0 | Padrão recomendado para modelos com menos de 5 bilhões de parâmetros. Melhor qualidade neste tamanho. |
| Phi-4-mini-3.8B | ~8GB | MIT | Excelente em raciocínio / matemática / código. Licença livre de restrições. |
| SmolLM3-3B | ~6GB | Apache 2.0 | Receita totalmente aberta. Contexto nativo de 64K. |
| Qwen 2.5 7B | ~12GB | Apache 2.0 | Padrão existente. Melhor qualidade entre os modelos legados de 7 bilhões de parâmetros. |
| Qwen 2.5 3B | ~8GB | Qwen-Research | ⚠ licença de pesquisa — veja os termos da licença do Qwen antes de usar comercialmente. |
| Llama 3.2 3B | ~8GB | Llama Community | Alternativa sólida ao Qwen 3B, com algumas restrições. |
| Llama 3.2 1B | ~6GB | Llama Community | Ideal para experimentos rápidos em placas menores. |
| Mistral 7B | ~12GB | Apache 2.0 | Comparável ao Qwen 7B, com um modelo de chat diferente. |

Outros modelos podem funcionar, mas apenas estes oito são fixos nos testes automatizados (CI). Use `--lora-preset=quality` (padrão) para obter um rank de 256 / alvos lineares, conforme descrito por Biderman 2024 + Thinking Machines 2025, ou `--lora-preset=fast` para obter o tamanho da v1.2.x, que usa um rank de 16 / alvo q+v, se você precisar do tamanho da v1.2.x.

## Solução de problemas

Um índice resumido dos erros mais comuns na primeira execução. O índice completo está na [página do manual de solução de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/). Para uma análise detalhada de driver / VRAM / precisão mista, consulte a [página de solução de problemas do CUDA](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/).

| Sintoma | Código de erro | Solução |
|---|---|---|
| A GPU fica sem memória durante o treinamento. | `RUNTIME_GPU_OOM` | Automatic — O Backpropagate reduz o tamanho do lote pela metade e tenta novamente até 3 vezes. Para desativar: `Trainer(oom_recovery=False)`. Para forçar um tamanho menor: `--batch-size 1`. |
| HuggingFace retorna 401 / "modelo não encontrado" | `DEP_MODEL_LOAD_FAILED` | Execute `huggingface-cli login` e tente novamente. Para erros de digitação, copie o ID exato de <https://huggingface.co/models>. |
| A conexão `register_with_ollama` foi recusada. | `DEP_OLLAMA_REGISTRATION_FAILED` | Inicie o daemon: `ollama serve`. Instale a partir de <https://ollama.com>. É possível tentar novamente. |
| Disco cheio durante o salvamento do checkpoint. | `STATE_CHECKPOINT_INVALID` | As operações de escrita atômicas deixam um diretório `.partial` em caso de falha — é seguro excluí-lo. O checkpoint anterior e válido está intacto. |
| Treinamento pausado devido ao superaquecimento da GPU | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | Automatic — O Backpropagate pausa quando a temperatura atinge o limite e retoma quando a GPU esfria. Melhore a ventilação se isso continuar acontecendo. |
| `backprop ui --share` foi rejeitado. | `INPUT_AUTH_REQUIRED` | Use `--auth user:password` ou utilize o encaminhamento de porta SSH (veja [Interface Web](#web-ui)). |
| A exportação para GGUF falhou na primeira tentativa. | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; no Windows, você também precisa das ferramentas de compilação Visual C++ e do CMake. |

## Relatando bugs

Quando algo falha, o Backpropagate imprime uma linha no início, como `run_started run_id=<uuid>`, e associa esse ID a cada linha de log, a cada checkpoint e a cada entrada do Weights & Biases. **Inclua o `run_id` em qualquer relatório de bug** — isso permite que um mantenedor correlacione tudo para aquela execução específica.

Um bom relatório de bug inclui:

1. **O `run_id`** — o UUID impresso no início.
2. **O código de erro** — a linha `[CODE_NAME]: message` no stderr. Consulte [códigos de erro](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) para a lista completa.
3. **O comando de linha de comando, com informações sensíveis removidas.** O stderr é automaticamente anonimizado (tokens Bearer, `sk-*`, `hf_*`, chaves AWS, pares `password=` / `token=` são removidos) — é seguro colar. Para obter o rastreamento completo e não anonimizado, execute novamente com `--verbose`, mas revise antes de publicar.
4. **Versões do Python / PyTorch, modelo da GPU, sistema operacional.** `backprop info` imprime tudo isso de uma vez.

Perguntas, ideias ou discussões sobre se algo é esperado devem ser feitas em [Discussões do GitHub](https://github.com/mcp-tool-shop-org/backpropagate/discussions). Problemas de segurança devem ser relatados de forma privada através do formulário [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) — consulte [SECURITY.md](SECURITY.md) para a política.

## Privacidade

Todo o treinamento ocorre localmente na sua GPU. O Backpropagate não faz solicitações de rede, exceto para baixar modelos do HuggingFace (o que você inicia). Não há telemetria, nem dependência de serviços em nuvem.

## Referências

Os padrões do Backpropagate e o modo de treinamento com várias execuções são baseados em pesquisas recentes. Se você estiver interessado nas técnicas subjacentes:

- **Hu et al. 2021.** *LoRA: Adaptação de Baixa Rank de Modelos de Linguagem Grandes.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — o artigo fundamental que introduz o LoRA, que é como o Backpropagate treina adaptadores de forma eficiente.
- **Biderman et al. 2024.** *LoRA Aprende Menos e Esquece Menos.* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — evidências empíricas de que o LoRA com rank 256 e alvos lineares corresponde à qualidade do ajuste fino completo na maioria das tarefas de pós-treinamento, com 67% do poder computacional. Define a configuração padrão do LoRA v1.3 do Backpropagate.
- **Thinking Machines 2025.** *LoRA Sem Arrependimentos.* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — o acompanhamento prático que identifica a correção de 10 vezes a taxa de aprendizado em relação ao ajuste fino completo necessária para ranks LoRA mais altos.
- **Kirkpatrick et al. 2017.** *Superando o esquecimento catastrófico em redes neurais.* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — a caracterização original de por que as redes neurais "esquecem" o treinamento anterior quando você as ajusta em novos dados (EWC — Consolidação de Pesos Elástica).
- **Wang et al. 2023.** *Aprendizagem Ortogonal de Subespaços para Aprendizagem Contínua de Modelos de Linguagem.* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — O-LoRA, uma abordagem anterior para usar LoRA para aprendizado contínuo, restringindo novos adaptadores a subespaços ortogonais.
- **Yadav et al. 2023.** *TIES-Merging: Resolvendo Interferências ao Mesclar Modelos.* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — uma técnica fundamental para mesclar vários modelos ajustados sem interferência.
- **Qiao & Mahdavi 2025.** *Mescle Antes de Esquecer: Uma Aprendizagem Contínua LoRA via Mesclagem Contínua.* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — o algoritmo específico que o mesclador de várias execuções do Backpropagate implementa. Um preprint de dezembro de 2025; o Backpropagate é o primeiro adotador conhecido a partir do artigo.

## Licença

MIT — veja [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
