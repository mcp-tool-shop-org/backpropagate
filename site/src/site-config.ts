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
        ['[ui]', 'Reflex (Radix UI) web interface', 'reflex'],
        ['[validation]', 'Pydantic config validation', 'pydantic, pydantic-settings'],
        ['[export]', 'GGUF export for Ollama', 'llama-cpp-python'],
        ['[monitoring]', 'WandB + system monitoring', 'wandb, psutil'],
        ['[logging]', 'Structured logging (2026 best practices)', 'structlog'],
        ['[security]', 'JWT auth + secure token generation', 'PyJWT, cryptography'],
        ['[standard]', 'unsloth + ui (recommended)', 'all of the above'],
        ['[production]', 'unsloth + ui + validation + logging + security', 'production deployment'],
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
          desc: 'Single LoRA Continual Learning via Asymmetric Merging (arXiv:2512.23017) prevents catastrophic forgetting during extended fine-tuning campaigns via orthogonal init, asymmetric A/B handling, and time-aware scaling. Checkpoint-and-resume keeps long runs recoverable after crashes.',
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
      subtitle: 'Ship Gate audit — 24/37 checked, 13 skipped (each with justification), 100% pass on every applicable item.',
      columns: ['Category', 'Score', 'Notes'],
      rows: [
        ['A. Security', '5/8', 'SECURITY.md, trust model, no secrets/telemetry, safe_path(), output-directory denylist; the 3 SKIPs cover destructive-action / MCP rows that do not apply.'],
        ['B. Error Handling', '3/7', 'Structured exception shape (code/message/hint/cause/retryable) via ERROR_CODES registry; CLI exit codes 0/1/2/3; no raw stack traces without --verbose; run_id correlation; redacted stderr; --share+--auth gating; the 4 SKIPs cover MCP / desktop / VS Code rows that do not apply.'],
        ['C. Operator Docs', '4/7', 'README, CHANGELOG, LICENSE, --help; the 3 SKIPs cover formal log-tier / MCP / operational-complexity rows.'],
        ['D. Shipping Hygiene', '7/9', 'verify.sh, version=tag, 5 scanners in CI, dependabot, npm publish with Sigstore provenance; the 2 SKIPs cover VS Code extension / desktop app rows.'],
        ['E. Identity', '5/6', 'Logo, translations, landing page, metadata; soft gate, does not block ship.'],
      ],
    },
  ],
};
