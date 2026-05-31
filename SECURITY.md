# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.5.x   | :white_check_mark: (current; v1.5.0 shipped 2026-05-31) |
| 1.4.x   | :white_check_mark: (security-fix-only until v1.6.0; operators should plan migration to v1.5) |
| 1.3.x   | :x: (EOL with v1.5.0) |
| 1.2.x   | :x: (EOL with v1.4.0) |
| 1.1.x   | :x: (EOL with v1.2.0 — see GHSA-f65r-h4g3-3h9h below for the auth-bypass advisory affecting 1.1.0 / 1.1.1) |
| 1.0.x   | :x: (EOL with v1.1.0) |
| < 1.0   | :x:                |

Now that v1.5.0 has shipped, 1.4.x moves to security-fix-only support for one minor cycle (until v1.6.0); operators should plan migration to v1.5 within that window. The current minor + one back is always supported; older minors get security-only patches for one cycle, then EOL.

## Published Security Advisories

| Advisory | Affected | Patched | Severity | Title |
| -------- | -------- | ------- | -------- | ----- |
| [GHSA-f65r-h4g3-3h9h](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/GHSA-f65r-h4g3-3h9h) | `>=1.1.0, <1.2.0` (pip + npm) | `1.2.0` | Critical (CVSS 9.8) | `backprop ui --auth` and `backprop ui --share` do not enforce authentication |

CVE IDs are requested at publish time and attached to the GHSA when GitHub Security Lab assigns them (typically 1-7 days post-publication).

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue — the report is visible to anyone the moment it's filed.
2. Open a [private security advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) via GitHub's Security tab.
3. Or email the maintainer directly if the GitHub flow is blocked for you.

**Email:** 64996768+mcp-tool-shop@users.noreply.github.com

When reporting, please include:

- **The `run_id`** (printed at startup as `run_started run_id=<uuid>`, also exposed as `TrainingRun.run_id` / `RunResult.run_id`) if the issue surfaced during a training run. One UUID lets the maintainer correlate every log line, checkpoint, and SLAO merge for that exact run — without it, reproducing your timeline is much harder.
- **The structured error code** (e.g. `[INPUT_AUTH_REQUIRED]`, `[UI_OUTPUT_DIR_FORBIDDEN]`) if one was emitted. The catalog at [handbook/error-codes](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) lists every stable code.
- **A minimal reproduction.** Stderr in non-verbose mode is automatically redacted (Bearer / `sk-*` / `hf_*` / AWS keys, `password=`/`token=`/`api_key=` pairs) — safe to paste. For the full unredacted trace, re-run with `--verbose` (or `BACKPROPAGATE_DEBUG=1`) and review the output for any remaining secrets before posting.

### Response Timeline

- **Acknowledgment:** within 48 hours.
- **Assessment:** within 1 week — severity, scope, and a rough patch ETA.
- **Fix:** based on severity (Critical: hot-patch release; High: next patch; Medium/Low: next minor).
- **Disclosure:** GHSA publication coordinated with the reporter; CVE IDs requested at publish (typically 1–7 days post-publication, see existing advisories for examples).

## Security Best Practices

- Never commit API keys or tokens
- Validate training data sources
- Keep dependencies updated
- When using `backprop ui`, do not pass `--share` without `--auth` (and on v1.1.x, do not pass either — see GHSA-f65r-h4g3-3h9h above; upgrade to v1.2.0). In v1.2.0+, `--share` without `--auth` hard-errors with `RUNTIME_UI_AUTH_NOT_ENFORCED`. For remote access without exposing the UI publicly, use SSH port-forwarding: `ssh -L 7860:localhost:7860 <training-host>` then open `http://localhost:7860` locally.
- The UI sandboxes filesystem writes to a single base directory (`~/.backpropagate/ui-outputs` by default; override via `BACKPROPAGATE_UI__OUTPUT_DIR`). The override is denylist-validated — system / credential paths are refused with `[UI_OUTPUT_DIR_FORBIDDEN]`.
