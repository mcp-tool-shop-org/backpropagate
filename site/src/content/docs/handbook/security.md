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

- The public internet when `--share` is set. v1.2.0 enforces auth via the live FastAPI middleware (see [Four-layer defense in depth](#four-layer-defense-in-depth)); `--share` without `--auth user:pass` refuses to start, and `--share --auth user:pass` runs through the middleware's Basic-auth + Host/Origin allowlist gate. [SSH port-forwarding](#ssh-port-forwarding-recipe) remains the lowest-friction pattern when you do not need a public URL.
- `BACKPROPAGATE_UI__OUTPUT_DIR` overrides — validated against the denylist on first use.
- Arbitrary model names passed to `--model` — sanitised against an allowlist regex before reaching the Ollama / HF push surfaces.

**Out of scope:**

- backpropagate is not a multi-tenant SaaS. The threat model is "single operator on a local or remote training host." Multi-tenant isolation is not designed for and not tested.
- Adversaries with code execution on the host are out of scope. The library's defense surface starts at "operator runs `backprop ui` and an attacker reaches the UI over the network."
- Model weights as untrusted inputs are out of scope — `pickle` loading is refused, but `safetensors` content is not introspected for adversarial content.

## v1.1.x advisory (GHSA-f65r-h4g3-3h9h)

**Affected:** backpropagate 1.1.0, 1.1.1.

**Severity:** CVSS 9.8 (Critical). Published 2026-05-23.

**Issue:** the Reflex UI advertised `--share + --auth` enforcement, but `backpropagate/ui_app/**` never read `BACKPROPAGATE_UI_AUTH`. Running `backprop ui --auth user:pass` or `backprop ui --share --auth user:pass` published an **unauthenticated** Web UI. The UI controls training jobs and has read access to `HF_TOKEN` from the operator environment.

**Fix:** v1.2.0 ships the real FastAPI auth middleware via `rx.App(api_transformer=basic_auth_transformer)`, layered behind a four-layer refuse-to-start defense (described below) that also closes the ambient-env-bypass and direct-`reflex run` invocation paths.

**Advisory:** [GHSA-f65r-h4g3-3h9h](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/GHSA-f65r-h4g3-3h9h).

**Action:** upgrade to v1.2.0 (`pip install -U backpropagate`). v1.1.0 / v1.1.1 will continue to install from PyPI but should not be deployed for any UI workflow that uses `--share`, `--host <non-loopback>`, or `--auth`.

## Four-layer defense in depth

Every layer keys off the single boolean `backpropagate/ui_app/auth.py::ENFORCEMENT_AVAILABLE`. In v1.2.0 this flag is `True`: every layer is live, the FastAPI middleware enforces auth on HTTP routes and the `/_event` WebSocket upgrade, and the refuse-to-start rails below catch every bypass path. The flag stays in place so that a downgraded test stub or a partial `[ui]` extra install can still trip the gates instead of silently exposing an unauthenticated UI.

### Layer 1: `cli.py:cmd_ui` refuse-to-start gates

These gates fire even with the middleware live, because they catch contract violations the middleware is not the right place to reject:

- `--share` without `--auth` → exits `1` with `[RUNTIME_UI_AUTH_NOT_ENFORCED]`. A public URL with no credentials is the v1.1.x bug v1.2.0 closed; refusing keeps the contract intact even if a future tunnel provider is wired up.
- `--host <non-loopback>` without `--auth` → same code. DNS-rebinding defense per CVE-2024-28224 / CVE-2025-49596 lineage.
- `--auth` requested while `ENFORCEMENT_AVAILABLE=False` (degraded `[ui]` extra) → same code. Stops the runtime before the v1.1.x false-promise re-emerges.

### Layer 2: `cli.py:cmd_ui` strips ambient auth variables

Before spawning the Reflex subprocess, the CLI removes `BACKPROPAGATE_UI_AUTH`, `BACKPROPAGATE_UI_AUTH_USER`, `BACKPROPAGATE_UI_AUTH_VERIFIER` and `BACKPROPAGATE_UI_LAUNCH_TOKEN` from the environment it hands over, then sets only what this launch decided. This closes the BRIDGE-B-001 ambient-env bypass: an env-var-only setup would otherwise reach the subprocess and create the *illusion* of auth coverage when the operator never asked for it on the command line, and a stale token or verifier could override the one the launch chose.

### Layer 3: `ui_app/app.py` module-import guard

If `BACKPROPAGATE_UI_AUTH` is set at module-import time and `ENFORCEMENT_AVAILABLE=False`, `ui_app/app.py` raises during import. This catches the `python -c "from backpropagate.ui_app.app import app"` invocation path that does not go through `cli.py`.

### Layer 4: `rxconfig.py` module-import guard

Identical guard, fired at `python -m reflex run` direct invocations (the path Reflex itself uses internally and that operators might invoke when debugging). Without this layer, a `BACKPROPAGATE_UI_AUTH=user:pass python -m reflex run` from the package directory would silently start an unauthenticated UI.

## Auth middleware (v1.2.0)

The middleware is wired in `ui_app/app.py` via Reflex's documented `rx.App(api_transformer=...)` hook (the `App.api` surface was removed in Reflex 0.8). It wraps the whole ASGI app so HTTP routes AND the `/_event` WebSocket upgrade go through the same gate, and it supports four modes resolved from the CLI flags + environment:

| Mode | Invocation | Bind | Auth | Allowlist | Footer badge |
|------|-----------|------|------|-----------|--------------|
| Default | `backprop ui` | 127.0.0.1 | per-launch random token in the banner URL, then a session cookie; lock file | `127.0.0.1`, `localhost` | `Local · token` |
| Basic | `backprop ui --auth user:pass` (or `--auth-file <path>`) | 127.0.0.1 | HTTP Basic | `127.0.0.1`, `localhost` | `Local · Basic` |
| Shared | `backprop ui --share --auth user:pass` (or `--auth-file <path>`) | cloudflared tunnel (v1.3) | HTTP Basic | `127.0.0.1` + tunnel host | `Shared · Basic` |
| Network | `backprop ui --host 0.0.0.0 --auth user:pass` (or `--auth-file <path>`) | network | HTTP Basic | `127.0.0.1` + LAN IPs | `Network · Basic` |

**`--auth-file` (v1.3 alternative to `--auth`):** reads `user:pass` from a file instead of taking it on the command line — keeps the credential out of shell history and out of `ps aux`. Mutually exclusive with `--auth` (passing both exits `1` with `INPUT_AUTH_INVALID_SHAPE`). The file mode is checked on POSIX: a mode wider than `0600` emits a warning at startup. Create with `printf 'user:pass' > path && chmod 600 path`. Satisfies the same gate as `--auth`. See [recipes → --auth-file](/backpropagate/handbook/recipes/#use---auth-file-for-shell-history-safe-auth).

**The password never reaches the UI process (1.8.1):** with `--auth` or `--auth-file`, the CLI hashes the password into a salted scrypt verifier and passes the Reflex subprocess only `BACKPROPAGATE_UI_AUTH_USER` and `BACKPROPAGATE_UI_AUTH_VERIFIER`. The middleware checks each Basic credential against the verifier in constant time. Before 1.8.1 the plaintext `user:pass` went into the child's environment, where any process running as the same user, and every grandchild, could read it, and into the session lock file on disk. The session-cookie signing key is now random per process; before 1.8.1 it was the SHA-256 of `user:pass`, so a captured cookie could be brute-forced offline for the password. Sessions therefore end when the UI restarts: sign in again.

**Per-launch token (default mode):** when neither `--auth` nor `--auth-file` is passed, `backprop ui` generates a random token on every launch and prints it in the banner URL, `http://127.0.0.1:<port>/?token=...`. Open that URL: the first request trades the token for a signed HttpOnly session cookie and the token is dropped from the address bar. Requests with neither the cookie nor a valid token are refused, the frontend's assets and the `/_event` WebSocket included (a `?token=` on the WebSocket URL does not open it; only the cookie does). Before 1.8.1 the CLI did not generate a token, so a default launch ran with no authentication at all, loopback-only. The token also lands in a `0600` lock file at `$XDG_RUNTIME_DIR/backpropagate/session-<port>.lock` (Linux/macOS) or `%LOCALAPPDATA%\backpropagate\session-<port>.lock` (Windows). The file is deleted on shutdown; `--auth` / `--auth-file` launches write none. Parallel processes running as the same user can discover the token without screen-scraping the startup banner — useful for `backprop info --runtime` and external tooling that wants to validate against the running UI.

**Public-URL tunnel via `cloudflared` (v1.3):** `--share` now spawns `cloudflared tunnel --url http://127.0.0.1:<port>`, parses the announced `https://*.trycloudflare.com` URL from cloudflared's stderr (with a `BACKPROPAGATE_CLOUDFLARED_TIMEOUT`-bounded wait — default 30s), and adds the URL to the auth middleware's Host + Origin allowlist via `BACKPROPAGATE_UI_SHARE_HOST`. Install `cloudflared` from <https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/>. The quick-tunnel is ephemeral (no account / no zone / no DNS setup) and dies with the `backprop ui` process. If `cloudflared` is not on `PATH`, the runtime emits a clear error pointing at the install URL + [SSH port-forwarding](#ssh-port-forwarding-recipe) as the fallback.

**One port (1.8.1):** `backprop ui` runs Reflex in production mode. The backend serves the compiled frontend, its assets and the `/_event` WebSocket on the single `--port`, so the middleware fronts every request. Before 1.8.1 it asked Reflex for dev mode, which has a separate frontend server on its own port, bound to `0.0.0.0` and outside the middleware (and on Reflex 0.9.x refused to start at all; see the changelog).

**Host-header allowlist** — every request validates `Host` against the mode's allowlist. DNS-rebinding defense; backpropagate is in the same exposure class as Ollama (CVE-2024-28224), MCP Inspector (CVE-2025-49596 CVSS 9.4), and Claude Code VS Code (CVE-2025-52882).

**Origin allowlist** — state-changing methods (POST/PUT/PATCH/DELETE) and the WS upgrade validate `Origin` against the same allowlist. CSWSH defense; rejects with 403 (HTTP) or close code 4403 (WS).

**Cookie hardening** — `HttpOnly` + `SameSite=Lax` + `Secure` (when not bound to 127.0.0.1) + 12h expiry + HMAC-signed payload (no server-side session store).

**Hard errors** (refuse-to-start, enforced by the CLI before the middleware ever sees a request):

- `--share` without `--auth` → `RUNTIME_UI_AUTH_NOT_ENFORCED`.
- `--host <non-loopback>` without `--auth` → same code.
- `--auth` value with a malformed shape (missing colon, empty username or password, multiple colons in the user portion, etc.) → `INPUT_AUTH_INVALID_SHAPE` (validated in `ui_security.validate_auth_shape`).
- `--auth` requested while `ENFORCEMENT_AVAILABLE=False` (degraded `[ui]` extra) → `RUNTIME_UI_AUTH_NOT_ENFORCED`.

## SSH port-forwarding recipe

The canonical remote-access pattern when you don't want to expose the UI directly. It needs no auth flags.

On the training host:

```bash
backprop ui
# Listens on 127.0.0.1:7862 and prints http://127.0.0.1:7862/?token=... in its banner.
```

On your laptop:

```bash
ssh -L 7862:localhost:7862 you@training-host
# Then open the banner URL (http://127.0.0.1:7862/?token=...) in your browser.
```

This tunnels the whole UI, WebSocket included, through your authenticated SSH session: no `--share`, no `--host 0.0.0.0`. The UI is reachable only from your laptop, and the per-launch token still applies, so copy the URL from the banner. Before 1.8.1 the UI also needed `port + 1` forwarded.

## Output-directory sandbox

The UI sandboxes filesystem writes (saved adapters, GGUF exports, converted datasets, Modelfiles) to a single allowed-base directory.

- **Default:** `~/.backpropagate/ui-outputs`
- **Override:** `BACKPROPAGATE_UI__OUTPUT_DIR=<path>`
- **Validation:** the override resolves against a denylist of system + credential trees — `/etc`, `/usr`, `/sys`, `/dev`, `/boot`, `/bin`, `/sbin`, `/var/run`, `/var/lib`, `/root`, `~/.ssh`, `~/.aws`, `~/.kube`, `~/.docker`, `~/.gnupg`, `~/.config`, plus the Windows system roots (`C:\Windows`, `C:\Program Files`, `C:\Program Files (x86)`, `C:\ProgramData`) and per-user credential dirs (`%USERPROFILE%\.ssh`, AppData crypto stores). Bare `/var` is intentionally NOT in the denylist because macOS's per-user temp tree lives at `/var/folders/<hash>/T/...` (pytest tmp_path, NSTemporaryDirectory, etc.); only the dangerous subtrees `/var/run` and `/var/lib` are denied individually. If the override resolves into a denied path, startup fails with `[UI_OUTPUT_DIR_FORBIDDEN]`. The denylist shape is correct for this surface because the operator picks any directory under their home and the system only refuses obvious foot-guns; an allowlist would force the operator to pre-enumerate every safe directory, which is hostile UX. For file TYPES on upload — a finite enumerable set — the opposite shape applies: `ui_security.FileValidator` uses a strict allowlist of `.jsonl / .json / .csv / .txt / .parquet` and the denylist there is a belt-and-suspenders sanity check. Different surface, different correct shape.
- **Enforcement:** every UI sink passes the resolved base as `allowed_base` to `safe_path`, so user-supplied paths cannot escape via `..` segments.

## Anti-patterns (do not do)

- Do **not** run `backprop ui --share` without `--auth user:pass`. v1.2.0 refuses to start this combination with `[RUNTIME_UI_AUTH_NOT_ENFORCED]`; v1.1.x silently advertised auth that didn't fire. With `--auth user:pass` the middleware validates every request against the Basic-auth credentials plus the Host/Origin allowlist — but SSH port-forwarding is still the lower-friction pattern when you don't actually need a public URL.
- Do **not** pass `--host 0.0.0.0` without `--auth user:pass`. v1.2.0 refuses this combination at startup for the same reason — a non-loopback bind without credentials is the DNS-rebinding foot-gun.
- Do **not** put `HF_TOKEN` or any credential in argparse (it appears in `ps aux`). Export it in the environment or use `huggingface-cli login` to cache it.
- Do **not** disable the output-directory denylist. It exists to prevent path-traversal bugs in the UI from writing into your system or credential paths.
- Do **not** invoke `python -m reflex run` or `reflex run` from inside the `backpropagate/` package directory while setting `BACKPROPAGATE_UI_AUTH` and assume auth is wired. The layer-3 + layer-4 import-time guards refuse to start when `ENFORCEMENT_AVAILABLE=False` precisely so that operator confusion cannot bypass the middleware. A direct run also skips what `backprop ui` adds: the per-launch token, the password verifier in place of the plaintext, and production mode. Always launch via `backprop ui`.

## Code-execution metric (`pass_rate`)

`backprop eval --metric pass_rate` scores generated code by running it against test snippets. Model output is untrusted code, so this is off unless you pass `--allow-code-exec` (or set `BACKPROPAGATE_ALLOW_CODE_EVAL=1`). When it is on, each sample runs in a separate `python -I` process with a timeout (`--code-exec-timeout`, default 10 s), a stripped environment (no `HF_TOKEN` or other secrets from your shell), a fresh temporary working directory, closed stdin, and on Linux/macOS limits on CPU time, memory (2 GiB), file size (10 MiB) and core dumps. A sample that times out, crashes or prints a malformed report scores 0.

This is isolation, **not a sandbox**: the code still runs as your user, with your file-system and network access. Run it only on output you would be willing to run yourself, or inside a container or VM. Before v1.8.1 the metric ran the code with `exec` inside the eval process, with no timeout.

## Reporting vulnerabilities

See the repo-root [SECURITY.md](https://github.com/mcp-tool-shop-org/backpropagate/blob/main/SECURITY.md). Open a GitHub Security Advisory; do **not** file as a public issue. Include the `run_id` (printed at startup) and the structured error code if one was emitted.
