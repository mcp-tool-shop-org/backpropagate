<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="assets/logo.png" alt="Backpropagate" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/backpropagate/"><img src="https://img.shields.io/pypi/v/backpropagate" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**Ajuste fino de LLMs sem interface gráfica em 3 linhas. Configurações padrão inteligentes, dimensionamento de lote consciente da VRAM, treinamento SLAO em várias etapas e exportação GGUF com um clique para Ollama.**

*Treine LLMs em 3 linhas de código. Exporte para Ollama com mais uma linha.*

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
| --------- | ---------- |
| O ajuste fino é complexo | 3 linhas: carregar, treinar, salvar |
| O Windows é um pesadelo | Suporte completo para Windows |
| O gerenciamento da VRAM é difícil | Dimensionamento automático do lote, monitoramento da GPU |
| A exportação de modelos é confusa | Exportação GGUF com um clique + registro automático no Ollama |
| Treinamentos longos causam esquecimento | Treinamento SLAO em várias etapas |

## Principais Características

- **Sem Interface Gráfica por Design**: Projetado para pipelines de CI/CD, fluxos de trabalho automatizados e execução programática.
- **Configurações Padrão Inteligentes**: Configura automaticamente os hiperparâmetros ideais com base no seu hardware e conjunto de dados.
- **Treinamento SLAO em Várias Etapas**: Estratégias de treinamento avançadas para evitar o esquecimento catastrófico durante treinamentos longos.
- **Suporte Completo para Windows**: Testado e otimizado para ambientes Windows, evitando problemas comuns do PyTorch/CUDA.
- **Exportação Simplificada**: Exportação com um clique para o formato GGUF e registro automático no Ollama.
- **Arquitetura Modular**: Instale apenas as dependências que você precisa (por exemplo, `[unsloth]`, `[ui]`, `[export]`).

## Instalação

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Extra | Descrição | Dependências |
| ------- | ------------- | -------------- |
| `unsloth` | Treinamento 2x mais rápido, 50% menos VRAM | unsloth |
| `ui` | Interface web Gradio | gradio>=5.6.0 |
| `validation` | Validação de configuração Pydantic | pydantic, pydantic-settings |
| `export` | Exportação GGUF para Ollama | llama-cpp-python |
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

| Modelo Pré-Configurado | VRAM | Speed | Qualidade |
| -------- | ------ | ------- | --------- |
| Qwen 2.5 7B | ~12GB | Média | Best |
| Qwen 2.5 3B | ~8GB | Fast | Good |
| Llama 3.2 3B | ~8GB | Fast | Good |
| Llama 3.2 1B | ~6GB | Mais Rápido | Basic |
| Mistral 7B | ~12GB | Média | Good |

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

## Projetos Relacionados

Parte de [**MCP Tool Shop**](https://mcp-tool-shop.github.io/):

- [Tool Compass](https://github.com/mcp-tool-shop-org/tool-compass) — Descoberta semântica de ferramentas MCP
- [File Compass](https://github.com/mcp-tool-shop-org/file-compass) — Busca semântica de arquivos
- [Comfy Headless](https://github.com/mcp-tool-shop-org/comfy-headless) — ComfyUI sem a complexidade

## Licença

MIT — veja [LICENSE](LICENSE) para detalhes.
