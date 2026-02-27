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

**Ajuste fino de LLMs sem interface gráfica em 3 linhas. Configurações padrão inteligentes, dimensionamento de lote consciente da VRAM, treinamento SLAO em várias etapas e exportação para GGUF com um clique para Ollama.**

*Treine LLMs em 3 linhas de código. Exporte para Ollama com mais uma.*

## Início Rápido

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

## Por que usar Backpropagation?

| Problema | Solução |
|---------|----------|
| O ajuste fino é complexo | 3 linhas: carregar, treinar, salvar |
| O Windows é um pesadelo | Suporte completo para Windows |
| O gerenciamento da VRAM é difícil | Dimensionamento automático do lote, monitoramento da GPU |
| A exportação de modelos é confusa | Exportação para GGUF com um clique + registro automático no Ollama |
| Treinamentos longos causam esquecimento | Treinamento SLAO em várias etapas |

## Principais Características

- **Sem Interface Gráfica por Design**: Projetado para pipelines de CI/CD, fluxos de trabalho automatizados e execução programática.
- **Configurações Padrão Inteligentes**: Configura automaticamente os hiperparâmetros ideais com base no seu hardware e conjunto de dados.
- **Treinamento SLAO em Várias Etapas**: Estratégias de treinamento avançadas para evitar o esquecimento catastrófico durante treinamentos longos.
- **Suporte Completo para Windows**: Testado e otimizado para ambientes Windows, evitando problemas comuns com PyTorch/CUDA.
- **Exportação Simplificada**: Exportação para o formato GGUF com um clique e registro automático no Ollama.
- **Arquitetura Modular**: Instale apenas as dependências que você precisa (por exemplo, `[unsloth]`, `[ui]`, `[export]`).

## Instalação

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Extras | Descrição | Dependências |
|-------|-------------|--------------|
| `unsloth` | Treinamento 2x mais rápido, 50% menos VRAM | unsloth |
| `ui` | Interface web Gradio | gradio>=5.6.0 |
| `validation` | Validação de configuração Pydantic | pydantic, pydantic-settings |
| `export` | Exportação para GGUF para Ollama | llama-cpp-python |
| `monitoring` | WandB + monitoramento do sistema | wandb, psutil |

**Requisitos:** Python 3.10+ · GPU CUDA (8GB+ de VRAM) · PyTorch 2.0+

## Uso

### Treinamento Básico

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

### Treinamento SLAO em Várias Etapas

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")

result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
    merge_mode="slao",  # Smart LoRA merging
)
```

### Exportação para Ollama

```python
trainer.export(
    format="gguf",
    quantization="q4_k_m",
    register_ollama=True,
    model_name="my-finetuned-model",
)
# ollama run my-finetuned-model
```

### Interface de Linha de Comando (CLI)

```bash
backprop train --data my_data.jsonl --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backpropagate --ui --port 7862
```

## Suporte para Windows

O Backpropagate foi projetado para funcionar no Windows sem problemas:

- Pré-tokenização para evitar falhas de multiprocessamento
- Desativação automática do xformers para séries RTX 40/50
- Configurações seguras do dataloader
- Testado em RTX 5080 (16GB de VRAM)

## Modelos Pré-Configurados

| Modelo | VRAM | Velocidade | Qualidade |
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
├── datasets.py          # Dataset loading & filtering
├── export.py            # GGUF/Ollama export
├── config.py            # Pydantic settings
├── gpu_safety.py        # GPU monitoring & safety
└── ui.py                # Gradio interface
```

## Privacidade

Todo o treinamento ocorre localmente na sua GPU. O Backpropagate não faz solicitações de rede, exceto para baixar modelos do HuggingFace (o que você inicia). Sem telemetria, sem dependência de nuvem.

## Tabela de Avaliação

| Categoria | Pontuação | Observações |
|----------|-------|-------|
| A. Segurança | 10/10 | SECURITY.md, Bandit+Semgrep+Trivy+TruffleHog no CI, proteção contra travessia de caminho |
| B. Tratamento de Erros | 8/10 | Erros estruturados, limites de segurança da GPU, recuperação de checkpoint |
| C. Documentação do Operador | 9/10 | README, CHANGELOG, guia de instalação modular, ajuda da CLI |
| D. Higiene de Distribuição | 9/10 | CI + testes (33 arquivos), publicado no PyPI, cobertura Codecov |
| E. Identidade | 10/10 | Logotipos, traduções, página de destino, listagem no PyPI. |
| **Total** | **46/50** | |

## Licença

MIT — veja o arquivo [LICENSE](LICENSE) para detalhes.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
