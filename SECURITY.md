# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Open a private security advisory via GitHub's Security tab
3. Or email the maintainer directly

**Email:** 64996768+mcp-tool-shop@users.noreply.github.com

When reporting, please include:

- The `run_id` (printed at startup as `run_started run_id=<uuid>` and exposed as `TrainingRun.run_id` / `RunResult.run_id`) if the issue was observed during a training run. It correlates logs, checkpoints, and SLAO merge history for that exact run.
- The structured error code (e.g. `[INPUT_AUTH_REQUIRED]`, `[UI_OUTPUT_DIR_FORBIDDEN]`) if one was emitted.
- A minimal reproduction. Stderr in non-verbose mode is automatically redacted (Bearer / `sk-*` / `hf_*` / AWS keys, `password=`/`token=`/`api_key=` pairs) — safe to paste. For the full unredacted trace, re-run with `--verbose` and review before posting.

### Response Timeline

- Acknowledgment: Within 48 hours
- Assessment: Within 1 week
- Fix: Based on severity

## Security Best Practices

- Never commit API keys or tokens
- Validate training data sources
- Keep dependencies updated
- When using `backprop ui`, leave `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=true` (the default). The `--share` flag publishes a public-internet URL; without `--auth`, that exposes the entire training pipeline.
- The UI sandboxes filesystem writes to a single base directory (`~/.backpropagate/ui-outputs` by default; override via `BACKPROPAGATE_UI__OUTPUT_DIR`). The override is denylist-validated — system / credential paths are refused with `[UI_OUTPUT_DIR_FORBIDDEN]`.
