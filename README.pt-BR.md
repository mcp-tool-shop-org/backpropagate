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
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**Ajuste fino de LLMs sem interface em 3 linhas. Configurações padrão inteligentes, dimensionamento de lote consciente da VRAM, treinamento SLAO em várias etapas e exportação GGUF com um clique para Ollama.**

*SLAO é Single LoRA Continual Learning via Asymmetric Merging — a técnica de mesclagem entre etapas que evita o esquecimento catastrófico em campanhas de ajuste fino prolongadas ([artigo](https://arxiv.org/abs/2512.23017)).*

*Treine LLMs em 3 linhas de código. Exporte para Ollama com mais uma linha.*

## Início Rápido

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("examples/quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

O repositório inclui um pequeno arquivo `examples/quickstart.jsonl` (5 exemplos no formato ShareGPT) para que o trecho de código acima seja executado de ponta a ponta em uma instalação limpa. Para seu próprio treinamento, consulte o formato de conjunto de dados [Dataset Format](#dataset-format) abaixo.

### Caminho sem código: Interface Web

Prefere uma interface gráfica a um REPL Python? Instale os pacotes necessários e execute:

```bash
pip install backpropagate[standard]
backprop ui --port 7862
```

A interface Reflex (Radix UI) permite que você selecione um arquivo JSONL, escolha um modelo, treine e exporte — sem necessidade de Python. A interface é local; para acesso pela internet, consulte a seção [Web UI](#web-ui) abaixo para obter informações sobre o contrato de segurança `--share` + `--auth` e as opções de túnel suportadas (Cloudflare Tunnel, ngrok).

## Formato do Conjunto de Dados

Seu arquivo de treinamento JSONL deve ter um exemplo por linha. O formato mais simples é o chat ShareGPT:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Formatos Alpaca (`instruction`/`output`), OpenAI chat (`messages`) e texto simples também são suportados. Consulte `examples/quickstart.jsonl` para um ponto de partida que pode ser copiado.

## Por que propagar o erro (backpropagation)?

| Problema | Solução |
|---------|----------|
| O ajuste fino é complexo | 3 linhas: carregar, treinar, salvar |
| Windows é um pesadelo | Suporte completo para Windows |
| O gerenciamento da VRAM é difícil | Dimensionamento automático do lote, monitoramento da GPU |
| A exportação de modelos é confusa | Exportação GGUF com um clique + registro automático no Ollama |
| Treinamentos longos causam esquecimento | Treinamento SLAO em várias etapas |

## Principais Características

- **Sem Interface por Design**: Projetado para pipelines de CI/CD, fluxos de trabalho automatizados e execução programática.
- **Configurações Padrão Inteligentes**: Configura automaticamente os hiperparâmetros ideais com base no seu hardware e conjunto de dados.
- **Treinamento SLAO em Várias Etapas**: Estratégias de treinamento avançadas para evitar o esquecimento catastrófico durante treinos prolongados.
- **Suporte Completo para Windows**: Testado e otimizado para ambientes Windows, evitando problemas comuns do PyTorch/CUDA.
- **Exportação Simplificada**: Exportação com um clique para o formato GGUF e registro automático no Ollama.
- **Arquitetura Modular**: Instale apenas as dependências que você precisa (por exemplo, `[unsloth]`, `[ui]`, `[export]`).

## Instalação

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Reflex (Radix UI) web interface
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Pacotes Adicionais | Descrição | Dependências |
|-------|-------------|--------------|
| `unsloth` | Treinamento 2x mais rápido, 50% menos VRAM | unsloth |
| `ui` | Interface web Reflex (Radix UI) | reflex>=0.9.2, fastapi>=0.115 |
| `validation` | Validação de configuração Pydantic | pydantic, pydantic-settings |
| `export` | Exportação GGUF para Ollama | llama-cpp-python |
| `monitoring` | WandB + monitoramento do sistema (integrado automaticamente ao treinador na versão 1.1.0) | wandb, psutil |
| `logging` | Registro estruturado | structlog |
| `security` | Autenticação JWT + geração de tokens | PyJWT, cryptography |
| `production` | unsloth + ui + validação + registro + segurança | (pacote) |

**Requisitos:** Python 3.10+ · GPU CUDA (8GB+ de VRAM) · PyTorch 2.0+

### Pré-requisitos da plataforma

O Backpropagate lida com as peculiaridades de tempo de execução (multiprocessamento, xformers em RTX 40/50, workers do dataloader no Windows). Ele **não** lida com os problemas de instalação relacionados à plataforma — resolva esses problemas primeiro:

- **Versão do toolkit CUDA.** O PyTorch é publicado para cada versão do CUDA — escolher a versão incorreta instala silenciosamente apenas a versão para CPU. Use o seletor em <https://pytorch.org/get-started/locally/> para obter o comando `pip install torch ...` exato para o seu driver. Execute `nvidia-smi` para ver a versão do seu driver/CUDA.
- **Windows.** O Visual Studio Build Tools (C++) e o CMake são necessários para o extra `[export]` (compilações do `llama-cpp-python` a partir do código-fonte). A versão do pacote `bitsandbytes` é agora publicada nativamente para Windows (>= 0.43); guias mais antigos que mencionam `bitsandbytes-windows` estão desatualizados.
- **macOS.** O treinamento com GPU **não é suportado** — não há suporte para CUDA. Você pode instalar o Backpropagate para executar a *inferência* em um modelo GGUF exportado via Ollama, mas `trainer.train()` gera a exceção `DEP_GPU_NOT_AVAILABLE`. Use uma máquina com GPU para treinamento.
- **Linux.** A maioria das distribuições funciona sem problemas. Se você estiver usando a versão binária do PyPI, observe que a compilação para Linux usa apenas a versão para CPU do torch (para permanecer abaixo do limite de 2 GB de arquivos de lançamento do GitHub); instale primeiro a versão correspondente do pacote CUDA do pytorch.org.

Para solucionar problemas de instalação, consulte a [página do manual de solução de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/).

## Configuração

Todas as configurações podem ser substituídas por variáveis de ambiente usando o prefixo `BACKPROPAGATE_` (por exemplo, `BACKPROPAGATE_LOG_LEVEL=debug`). Um arquivo `.env` na raiz do projeto é carregado automaticamente quando o extra `[validation]` é instalado.

Configurações comuns (veja a [referência completa das variáveis de ambiente](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/) para tudo):

| Variável | Padrão | Observações |
|----------|---------|-------|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | auto | Força logs em formato JSON (`true`) ou no console (`false`) |
| `BACKPROPAGATE_LOG_FILE` | não definido | Caminho para copiar os logs |
| `BACKPROPAGATE_DEFER_FEATURE_DETECTION` | não definido | Ignora a detecção de dependências opcionais na inicialização para uma inicialização mais rápida da CLI |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | Diretório base para todas as operações de escrita no sistema de arquivos da interface do usuário; com validação de lista de permissões. |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | Modelo padrão |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | Taxa de aprendizado |
| `BACKPROPAGATE_LORA__R` | `16` | Rank do LoRA |

As chaves aninhadas usam dois sublinhados como delimitador (convenção `env_nested_delimiter` do Pydantic).

## Uso

### Treinamento Básico

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

`Qwen/Qwen2.5-7B-Instruct` é a opção padrão e canônica — o valor `Trainer()` é retornado quando chamado sem um argumento de modelo (veja [`config.py`](backpropagate/config.py) `ModelConfig.name`). Exemplos mais antigos utilizavam a versão pré-quantizada `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`; mudamos a opção padrão para os pesos oficiais do Qwen para maior confiabilidade ([CHANGELOG v1.1.0](CHANGELOG.md#110---2026-05-21)). Qualquer um dos modelos funciona.

### Treinamento Multi-Run SLAO

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

SLAO (Single LoRA Continual Learning via Asymmetric Merging) implementa o artigo [Merge before Forget](https://arxiv.org/abs/2512.23017): inicialização da matriz A ortogonal via decomposição QR, tratamento assimétrico de A/B e escalonamento dependente do tempo `λ(i) = 1/√i`. A flag da CLI é `--samples` (o campo subjacente é `samples_per_run`).

### Exportação para Ollama

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

Consulte a [referência da CLI](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) para cada subcomando e flag, ou execute `backprop <subcomando> --help`.

### Retomada a partir de um checkpoint (v1.1.0)

Uma execução multi-run que falha na iteração 4 agora pode ser recuperada. Cada sessão multi-run grava seu `run_id` tanto no arquivo `run_history.json` quanto no manifesto de checkpoint no disco, permitindo que você retome a execução com um único comando.

```bash
backprop resume <run-id>                       # picks up the in-progress session
backprop multi-run --data ... --resume <run-id> # explicit form
backprop train --data ... --resume <run-id>    # single-run resume (continues run_id)
```

O comportamento padrão do `backprop multi-run` (sem `--resume`) detecta automaticamente uma execução em andamento para o mesmo diretório de saída e a continua. Para forçar uma nova execução, use `resume_from="off"` (API Python) ou omita `--resume` e inicie em um diretório de saída diferente.

Ao retomar uma execução multi-run, o checkpoint mais recente para aquele `run_id` é carregado no modelo, o estado do SLAO (State Linear Attention Optimization) é restaurado a partir do diretório `slao/` próximo ao checkpoint, e o loop de execução continua a partir de `last_completed_run + 1`. O status da entrada no histórico é alterado para `running`, permitindo que `backprop list-runs --status running` mostre a sessão ativa.

### Rastreamento de experimentos (v1.1.0)

O `Trainer` detecta automaticamente os rastreadores de experimentos instalados (`wandb`, `tensorboard`, `mlflow`) e os integra ao `transformers.TrainingArguments` subjacente. O padrão `report_to="auto"` usa o que estiver disponível para importação.

```bash
pip install backpropagate[monitoring]  # installs wandb + psutil
wandb login                            # one-time
backprop train --data my_data.jsonl    # W&B run gets the same run_id prefix as the on-disk history
```

Para desativar explicitamente, use `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])` ou `Trainer(report_to="none")`. Para MLflow, instale com `pip install mlflow`; para TensorBoard, instale com `pip install tensorboard`. O nome da execução do W&B é `backprop-<run_id_prefix>`, permitindo que um operador use `grep` para pesquisar em W&B, nossos logs e `run_history.json` usando o mesmo identificador.

### Histórico de treinamento

Cada invocação de `backprop train` e `backprop multi-run` registra uma linha em `<output>/run_history.json` com o `run_id`, modelo, conjunto de dados, hiperparâmetros, status, perda final, histórico de perdas e, para multi-run, a linha do tempo de mesclagem do SLAO. Liste as execuções recentes:

```bash
backprop list-runs                         # most recent 20 runs, all statuses
backprop list-runs --status failed         # filter
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial run_id ok)
```

O histórico de execuções persiste entre processos — a guia "Runs" na interface web é uma visualização separada, em memória; o histórico no disco é a fonte de verdade para `list-runs` / `show-run` / `resume`.

### Interface Web

Inicie a interface Reflex localmente:

```bash
backprop ui --port 7862
```

Para expor um URL acessível pela internet, você deve combinar `--share` com `--auth`:

```bash
backprop ui --share --auth alice:hunter2
```

O comando `backprop ui --share` sem a opção `--auth` retorna o código de erro `1` e a mensagem de erro estruturada `[RUNTIME_UI_AUTH_NOT_ENFORCED]`. A razão para isso é que a opção `--share` publica um URL público que qualquer pessoa na internet pode acessar, e sem autenticação, isso significa que qualquer pessoa pode controlar o seu processo de treinamento. Não há como desativar essa proteção; se você não quiser definir credenciais, use o encaminhamento de porta SSH: `ssh -L 7860:localhost:7860 <host>` e, em seguida, abra `http://localhost:7860` localmente. Consulte o documento [handbook/security.md](site/src/content/docs/handbook/security.md) para obter informações detalhadas sobre o modelo de ameaças.

As operações de escrita no sistema de arquivos a partir da interface são restritas a um único diretório:

- Padrão: `~/.backpropagate/ui-outputs`
- Substituição: `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- A substituição é **validada por lista de permissões** — caminhos do sistema / credenciais (`/etc`, `/var`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.) são rejeitados com `[UI_OUTPUT_DIR_FORBIDDEN]`.

## Suporte para Windows

O Backpropagate foi projetado para funcionar no Windows por padrão:

- Pré-tokenização para evitar falhas de multiprocesso
- Desativação automática do xformers para séries RTX 40/50
- Configurações de dataloader seguras
- Testado em RTX 5080 (16GB de VRAM)

## Modelos Pré-Configurados

| Modelo Pré-Configurado | VRAM | Velocidade | Qualidade |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12GB | Média | Melhor |
| Qwen 2.5 3B | ~8GB | Rápida | Boa |
| Llama 3.2 3B | ~8GB | Rápida | Boa |
| Llama 3.2 1B | ~6GB | Mais Rápida | Básica |
| Mistral 7B | ~12GB | Média | Boa |

## Arquitetura

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

A implementação v1.0 do Gradio (`ui_gradio_legacy.py` + `theme_gradio_legacy.py`) foi mantida nas versões v1.1.x como referência e removida na v1.2.0.

## Solução de problemas

Um índice resumido dos erros mais comuns no início da execução. O índice completo está disponível em [esta página do manual de solução de problemas](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/); cada código abaixo está documentado em [códigos de erro](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/).

| Sintoma | Código | Solução |
|---------|------|-----|
| A GPU fica sem memória durante o treinamento. | `RUNTIME_GPU_OOM` | A recuperação automática de erros de memória (B-002) reduz automaticamente o tamanho do lote até 3 vezes. Para desativar: `Trainer(oom_recovery=False)`. Para forçar um tamanho menor: `--batch-size 1`. |
| O Hugging Face Hub retorna 401 / "modelo não encontrado". | `DEP_MODEL_LOAD_FAILED` | Execute `huggingface-cli login` e tente novamente. Para erros de digitação, copie o ID exato de <https://huggingface.co/models>. |
| Erro de digitação no nome do modelo. | `INPUT_VALIDATION_FAILED` ou `DEP_MODEL_LOAD_FAILED`. | Verifique o identificador `org/name` em <https://huggingface.co/models>. |
| A conexão `register_with_ollama` foi recusada. | `DEP_OLLAMA_REGISTRATION_FAILED` | Inicie o daemon: `ollama serve`. Instale a partir de <https://ollama.com>. É possível tentar novamente. |
| Disco cheio durante o salvamento do checkpoint. | `STATE_CHECKPOINT_INVALID` | As operações de escrita atômicas deixam um diretório `.partial` em caso de falha — é seguro excluí-lo. O checkpoint anterior e válido está intacto. |
| O treinamento foi pausado/cancelado devido ao superaquecimento da GPU. | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | B-003: o monitor pausa devido ao limite de temperatura da NVML; a execução é retomada automaticamente quando a GPU esfria. Melhore a circulação de ar ou reduza a carga sustentada. |
| `backprop ui --share` foi rejeitado. | `INPUT_AUTH_REQUIRED` | Passe `--auth user:password`. A partir da v1.2.0 (GHSA-f65r-h4g3-3h9h), `--share` sem `--auth` é um erro rígido sem opção de desativação; use encaminhamento de portas SSH se não puder expor credenciais. |
| "Sobreposição" de execuções múltiplas. | `CONFIG_INVALID` (Backend do estágio A, B-001). | Reduza o valor de `--samples` abaixo do tamanho do pool de treinamento, aumente o conjunto de dados ou desative a validação. |
| A exportação para GGUF falhou na primeira tentativa. | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; no Windows, você também precisa das ferramentas de compilação Visual C++ e do CMake. |

## Relatando bugs

Quando algo falha, o Backpropagate imprime uma linha `run_started run_id=<uuid>` no início e associa esse ID aos manifestos de checkpoint, ao histórico de mesclagem SLAO e às linhas de log estruturadas. Inclua o `run_id` em qualquer relatório de bug — isso permite que um mantenedor correlacione cada linha de log, cada checkpoint e cada mesclagem para essa execução específica.

Um bom relatório de bug inclui:

1. **`run_id`** — o UUID impresso no início (também disponível como `TrainingRun.run_id` e `RunResult.run_id`).
2. **O código de erro** — a linha `[CODE_NAME]: message` no stderr é o que você deve procurar; veja [códigos de erro](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) para o catálogo.
3. **A linha de comando, sem informações confidenciais.** O stderr no modo não detalhado é automaticamente removido (tokens Bearer, `sk-*`, `hf_*`, chaves AWS, pares `password=/token=/api_key` são removidos) — é seguro colar. Para obter o rastreamento completo e não removido, execute novamente com `--verbose`, mas revise antes de postar.
4. **Versões do Python / PyTorch, modelo da GPU, sistema operacional.** `backprop info` imprime tudo isso de uma vez.

## Privacidade

Todo o treinamento ocorre localmente na sua GPU. O Backpropagate não faz solicitações de rede, exceto para baixar modelos do HuggingFace (o que você inicia). Não há telemetria, nem dependência de serviços em nuvem.

## Tabela de desempenho

| Categoria | Pontuação | Observações |
|----------|-------|-------|
| A. Segurança | 6/8 | SECURITY.md, modelo confiável, sem segredos/telemetria, safe_path(). Itens MCP ignorados. |
| B. Tratamento de erros | 5/7 | Estrutura de exceções (`código`/`mensagem`/`dica`/`causa`/`tentável`) via registro de CÓDIGOS DE ERRO; códigos de saída da CLI: 0/1/2/3; rastreamentos de pilha não exibidos sem `--verbose`; correlação `run_id`; saída de erro (stderr) com informações removidas; bloqueio com `--share`+`--auth`. MCP/desktop/vscode ignorados. |
| C. Documentação para Operadores | 4/7 | README, CHANGELOG, LICENÇA, --help. Logging/MCP/complexo ignorados. |
| D. Boas Práticas de Desenvolvimento | 6/9 | verify.sh, versão=tag, 5 scanners no CI, dependabot, python_requires, build limpo. |
| E. Identidade | 4/4 | Logo, traduções, página de destino, metadados. |
| **Total** | **25/31** | 14 itens ignorados com justificativa · `shipcheck audit` passa 100% · Data da auditoria: 2026-05-21 (A linha B foi reclassificada após a fase B + o trabalho de códigos de saída da CLI na fase A). |

Histórico de design e o que cada item corresponde: veja [ROADMAP.md](ROADMAP.md) — todos os itens das semanas 1 a 4 foram incluídos na versão 1.1.0.

## Licença

MIT — veja [LICENSE](LICENSE) para detalhes.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
