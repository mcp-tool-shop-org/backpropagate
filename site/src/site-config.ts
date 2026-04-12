import type { SiteConfig } from '@mcptoolshop/site-theme';

export const config: SiteConfig = {
  title: 'Backpropagate',
  description: 'Headless LLM fine-tuning in 3 lines. Smart defaults, VRAM-aware batch sizing, multi-run SLAO, and one-click GGUF export for Ollama.',
  logoBadge: 'BP',
  brandName: 'Backpropagate',
  repoUrl: 'https://github.com/mcp-tool-shop-org/backpropagate',
  footerText: 'MIT Licensed — built by <a href="https://mcp-tool-shop.github.io/" style="color:var(--color-muted);text-decoration:underline">MCP Tool Shop</a>',

  hero: {
    badge: 'Python · PyPI',
    headline: 'Fine-tune LLMs',
    headlineAccent: 'in 3 lines.',
    description: 'Headless LLM fine-tuning with smart defaults. Automatic hyperparameter tuning, VRAM-aware batch sizing, multi-run SLAO training to prevent catastrophic forgetting, and one-click GGUF export for Ollama. First-class Windows and CUDA support.',
    primaryCta: { href: '#get-started', label: 'Get started' },
    secondaryCta: { href: 'handbook/', label: 'Read the Handbook' },
    previews: [
      {
        label: 'Quickstart',
        code: 'pip install backpropagate[standard]\n\n# Train in 3 lines\nfrom backpropagate import Trainer\n\ntrainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")\ntrainer.train("my_data.jsonl", steps=100)\ntrainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama',
      },
      {
        label: 'Multi-run SLAO',
        code: 'from backpropagate.multi_run import MultiRunTrainer\n\nrunner = MultiRunTrainer(\n    model="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",\n    num_runs=5, steps_per_run=100,\n    merge_mode="slao",\n)\nresult = runner.run("my_data.jsonl")',
      },
      {
        label: 'Export to Ollama',
        code: 'from backpropagate.export import export_gguf, register_with_ollama\n\nresult = export_gguf(model, tokenizer, "./output", quantization="q4_k_m")\nregister_with_ollama(result.path, model_name="my-model")',
      },
    ],
  },

  sections: [
    {
      kind: 'features',
      id: 'features',
      title: 'Fine-tuning without the friction',
      subtitle: 'Built for developers who want results, not configuration.',
      features: [
        {
          title: 'Smart defaults',
          desc: 'Automatically configures learning rate, batch size, gradient accumulation, and LoRA rank based on your hardware and dataset size. No hyperparameter guesswork.',
        },
        {
          title: 'VRAM-aware training',
          desc: 'Auto batch sizing and gradient checkpointing keep training stable on any GPU. Built-in VRAM monitoring with warnings before OOM. Works from 8GB up to multi-GPU setups.',
        },
        {
          title: 'First-class Windows',
          desc: 'Tested and optimized for Windows + CUDA. Avoids the common PyTorch/Unsloth pitfalls on Windows. If it runs on Linux, it runs on Windows too.',
        },
      ],
    },
    {
      kind: 'data-table',
      id: 'install',
      title: 'Modular installation',
      subtitle: 'Install only the dependencies you need.',
      columns: ['Extra', 'What you get', 'Key dependencies'],
      rows: [
        ['backpropagate', 'Core API only — minimal footprint', '—'],
        ['[unsloth]', '2× faster training, 50% less VRAM', 'unsloth'],
        ['[ui]', 'Gradio web interface for non-coders', 'gradio ≥ 5.6.0'],
        ['[validation]', 'Pydantic config validation', 'pydantic, pydantic-settings'],
        ['[export]', 'GGUF export for Ollama', 'llama-cpp-python'],
        ['[monitoring]', 'WandB + system monitoring', 'wandb, psutil'],
        ['[standard]', 'unsloth + ui (recommended)', 'all of the above'],
        ['[full]', 'Everything', 'all extras'],
      ],
    },
    {
      kind: 'code-cards',
      id: 'get-started',
      title: 'Get started',
      cards: [
        {
          title: 'Install',
          code: '# Recommended\npip install backpropagate[standard]\n\n# Minimal core only\npip install backpropagate\n\n# All extras\npip install backpropagate[full]\n\n# Requires: Python 3.10+ · CUDA GPU (8GB+ VRAM)',
        },
        {
          title: 'Basic training',
          code: 'from backpropagate import Trainer\n\n# Smart defaults — no config needed\ntrainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")\ntrainer.train("my_data.jsonl", steps=100)\ntrainer.save("./my-model")',
        },
        {
          title: 'Multi-run SLAO',
          code: 'from backpropagate.multi_run import MultiRunTrainer\n\nrunner = MultiRunTrainer(\n    model="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",\n    num_runs=5, steps_per_run=100,\n    merge_mode="slao",\n)\nresult = runner.run("my_data.jsonl")',
        },
        {
          title: 'Export to Ollama',
          code: 'from backpropagate.export import export_gguf, register_with_ollama\n\nresult = export_gguf(model, tokenizer, "./output", quantization="q4_k_m")\nregister_with_ollama(result.path, model_name="my-model")\n# ollama run my-model',
        },
      ],
    },
    {
      kind: 'features',
      id: 'design',
      title: 'Production-ready by design',
      subtitle: 'Built for CI/CD pipelines, automated workflows, and long training runs.',
      features: [
        {
          title: 'Headless by design',
          desc: 'No UI required. Runs in CI/CD pipelines, SSH sessions, and automated workflows. Full Python API with structured logging. Callbacks for progress tracking and early stopping.',
        },
        {
          title: 'Multi-run SLAO',
          desc: 'Smart Loss-Aware Ordering prevents catastrophic forgetting during extended fine-tuning campaigns. Checkpoint-and-resume keeps long runs recoverable after crashes.',
        },
        {
          title: 'LoRA + QLoRA + Unsloth',
          desc: 'Supports LoRA, QLoRA (4-bit), and Unsloth-accelerated training. Mix quantization levels per layer. Export to GGUF at any quantization: q2_k, q4_k_m, q8_0, or f16.',
        },
      ],
    },
    {
      kind: 'data-table',
      id: 'scorecard',
      title: 'Quality scorecard',
      subtitle: 'Ship Gate audit — 23/35 checked, 12 skipped, 100% pass.',
      columns: ['Category', 'Score', 'Notes'],
      rows: [
        ['A. Security', '6/8', 'SECURITY.md, trust model, no secrets/telemetry, safe_path()'],
        ['B. Error Handling', '3/7', 'Structured exceptions + exit codes + no raw stacks'],
        ['C. Operator Docs', '4/7', 'README, CHANGELOG, LICENSE, --help'],
        ['D. Shipping Hygiene', '6/9', 'verify.sh, version=tag, 5 scanners in CI, dependabot'],
        ['E. Identity', '4/4', 'Logo, translations, landing page, metadata'],
      ],
    },
  ],
};
