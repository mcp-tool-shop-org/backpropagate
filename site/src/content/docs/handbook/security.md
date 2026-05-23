---
title: Security
description: Threat model, advisories, and recommended deployment patterns.
sidebar:
  order: 9
---

backpropagate runs ML training jobs and can read your `HF_TOKEN`. Anyone who can reach the Web UI can drive training, trigger Hugging Face pushes, and read model files. This page is the single surface for "I want to expose this safely."

For vulnerability reporting, see the repo-root [SECURITY.md](https://github.com/mcp-tool-shop-org/backpropagate/blob/main/SECURITY.md).

## Threat model

**What backpropagate trusts** — the operator's local environment:

- Training datasets the operator points at (no untrusted-data parsing surface beyond standard JSONL / ShareGPT / Alpaca / OpenAI-chat parsers).
- Model weights the operator downloads. `safetensors` is loaded via the safe loader; `pickle`-based formats are refused via `security.py`.
- The output directory the operator configures (`BACKPROPAGATE_UI__OUTPUT_DIR` — denylist-validated against `/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc.).
- The local filesystem permissions on the operator's account.

**What backpropagate does NOT trust** — the network surface when exposed:

- The public internet when `--share` is set. v1.2.0 hard-refuses `--share` until the auth middleware lands; until then, [SSH port-forwarding](#ssh-port-forwarding-recipe) is the canonical pattern.
- `BACKPROPAGATE_UI__OUTPUT_DIR` overrides — validated against the denylist on first use.
- Arbitrary model names passed to `--model` — sanitised against an allowlist regex before reaching the Ollama / HF push surfaces.

**Out of scope:**

- backpropagate is not a multi-tenant SaaS. The threat model is "single operator on a local or remote training host." Multi-tenant isolation is not designed for and not tested.
- Adversaries with code execution on the host are out of scope. The library's defense surface starts at "operator runs `backprop ui` and an attacker reaches the UI over the network."
- Model weights as untrusted inputs are out of scope — `pickle` loading is refused, but `safetensors` content is not introspected for adversarial content.

## v1.1.x advisory (GHSA-pending)

**Affected:** backpropagate 1.1.0, 1.1.1.

**Issue:** the Reflex UI advertised `--share + --auth` enforcement, but `backpropagate/ui_app/**` never read `BACKPROPAGATE_UI_AUTH`. Running `backprop ui --auth user:pass` or `backprop ui --share --auth user:pass` published an **unauthenticated** Web UI. The UI controls training jobs and has read access to `HF_TOKEN` from the operator environment.

**Fix:** v1.2.0 lands a four-layer refuse-to-start defense (described below) and removes the auth advertisement from `--share` until real middleware lands.

**Advisory:** [GHSA at https://github.com/mcp-tool-shop-org/backpropagate/security/advisories](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories) — link goes live once the advisory is published.

**Action:** upgrade to v1.2.0 (`pip install -U backpropagate`). v1.1.0 / v1.1.1 will continue to install from PyPI but should not be deployed for any UI workflow that uses `--share`.

## Four-layer defense in depth

Every layer keys off the single boolean `backpropagate/ui_app/auth.py::ENFORCEMENT_AVAILABLE`. When the real auth middleware lands (Wave 6 of the v1.2.0 swarm), flipping that boolean to `True` re-enables every layer at once.

### Layer 1: `cli.py:cmd_ui` refuses-to-start

If the operator passes `--auth user:pass` or `--share` and `ENFORCEMENT_AVAILABLE=False`, the CLI exits `1` with `[RUNTIME_UI_AUTH_NOT_ENFORCED]`. The error message includes a pointer to SSH port-forwarding and to this page.

### Layer 2: `cli.py:cmd_ui` strips ambient `BACKPROPAGATE_UI_AUTH`

If the operator did **not** pass `--auth` but `BACKPROPAGATE_UI_AUTH` is set in the environment, the CLI strips it before spawning the Reflex subprocess. This closes the BRIDGE-B-001 ambient-env bypass: an env-var-only setup would otherwise reach the subprocess and create the *illusion* of auth coverage when no enforcement exists.

### Layer 3: `ui_app/app.py` module-import guard

If `BACKPROPAGATE_UI_AUTH` is set at module-import time and `ENFORCEMENT_AVAILABLE=False`, `ui_app/app.py` raises during import. This catches the `python -c "from backpropagate.ui_app.app import app"` invocation path that does not go through `cli.py`.

### Layer 4: `rxconfig.py` module-import guard

Identical guard, fired at `python -m reflex run` direct invocations (the path Reflex itself uses internally and that operators might invoke when debugging). Without this layer, a `BACKPROPAGATE_UI_AUTH=user:pass python -m reflex run` from the package directory would silently start an unauthenticated UI.

## Auth middleware (Wave 6 of v1.2.0)

When `ENFORCEMENT_AVAILABLE=True`, a Starlette ASGI middleware enforces the auth contract on both the HTTP routes and the `/_event` WebSocket upgrade. It supports four modes:

| Mode | Invocation | Bind | Auth | Allowlist | Footer badge |
|------|-----------|------|------|-----------|--------------|
| Default | `backprop ui` | 127.0.0.1 | per-launch random token in URL | `127.0.0.1`, `localhost` | `Local · token` |
| Basic | `backprop ui --auth user:pass` | 127.0.0.1 | HTTP Basic | `127.0.0.1`, `localhost` | `Local · Basic` |
| Shared | `backprop ui --share --auth user:pass` | tunnel | HTTP Basic | `127.0.0.1` + tunnel host | `Shared · Basic` |
| Network | `backprop ui --host 0.0.0.0 --auth user:pass` | network | HTTP Basic | `127.0.0.1` + LAN IPs | `Network · Basic` |

**Host-header allowlist** — every request validates `Host` against the mode's allowlist. DNS-rebinding defense; backpropagate is in the same exposure class as Ollama (CVE-2024-28224), MCP Inspector (CVE-2025-49596 CVSS 9.4), and Claude Code VS Code (CVE-2025-52882).

**Origin allowlist** — state-changing methods (POST/PUT/PATCH/DELETE) and the WS upgrade validate `Origin` against the same allowlist. CSWSH defense; rejects with 403 (HTTP) or close code 4403 (WS).

**Cookie hardening** — `HttpOnly` + `SameSite=Lax` + `Secure` (when not bound to 127.0.0.1) + 12h expiry + HMAC-signed payload (no server-side session store).

**Hard errors** (refuse-to-start, even with middleware enabled):

- `--share` without `--auth` → `RUNTIME_UI_AUTH_NOT_ENFORCED` (kept from v1.2.0 baseline).
- `--host <non-loopback>` without `--auth` → same code.
- `--auth` value with colon in the user portion → `INVALID_AUTH_FORMAT`.
- Tunnel provider fails to return URL within 15s → `TUNNEL_NOT_AVAILABLE` (no half-up state).

## SSH port-forwarding recipe

The canonical remote-access pattern when you don't want to expose the UI directly. Works against v1.1.x and v1.2.0 without any auth flags.

On the training host:

```bash
backprop ui
# Listens on 127.0.0.1:7860 (Reflex frontend) + 7861 (backend WebSocket).
```

On your laptop:

```bash
ssh -L 7860:localhost:7860 -L 7861:localhost:7861 you@training-host
# Then open http://localhost:7860 in your browser.
```

This tunnels both the frontend and the Reflex WebSocket through your authenticated SSH session — no `--share`, no `--host 0.0.0.0`, no auth middleware required. The UI is reachable only from your laptop, gated by your SSH credentials.

## Output-directory sandbox

The UI sandboxes filesystem writes (saved adapters, GGUF exports, converted datasets, Modelfiles) to a single allowed-base directory.

- **Default:** `~/.backpropagate/ui-outputs`
- **Override:** `BACKPROPAGATE_UI__OUTPUT_DIR=<path>`
- **Validation:** the override resolves against a denylist of `/etc`, `/var`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, etc. If the override resolves into one of those, startup fails with `[UI_OUTPUT_DIR_FORBIDDEN]`.
- **Enforcement:** every UI sink passes the resolved base as `allowed_base` to `safe_path`, so user-supplied paths cannot escape via `..` segments.

## Anti-patterns (do not do)

- Do **not** run `backprop ui --share` and assume it's safe. v1.2.0 refuses to start; v1.1.x advertised auth that didn't fire. Use SSH port-forwarding.
- Do **not** pass `--host 0.0.0.0` to expose the UI on your LAN without `--auth`. v1.3+ will refuse this combination at startup.
- Do **not** put `HF_TOKEN` or any credential in argparse (it appears in `ps aux`). Export it in the environment or use `huggingface-cli login` to cache it.
- Do **not** disable the output-directory denylist. It exists to prevent path-traversal bugs in the UI from writing into your system or credential paths.

## Reporting vulnerabilities

See the repo-root [SECURITY.md](https://github.com/mcp-tool-shop-org/backpropagate/blob/main/SECURITY.md). Open a GitHub Security Advisory; do **not** file as a public issue. Include the `run_id` (printed at startup) and the structured error code if one was emitted.
