---
title: Migrations
description: Operator-facing upgrade paths between Backpropagate versions.
sidebar:
  order: 10
---

Operator-facing migration narratives. Each section covers one upgrade hop with breaking changes, behavioural fixes, and the recommended migration steps. For older transitions (v1.0 → v1.1, the Gradio → Reflex pivot) see the [v1.1.0 CHANGELOG section](https://github.com/mcp-tool-shop-org/backpropagate/blob/main/CHANGELOG.md#110---2026-05-21).

## v1.3 → v1.4

A symbol-rename release. Two legacy names — `safe_gradio_handler` (a v1.0-era Gradio relic preserved through the v1.1.0 → Reflex migration) and `TRAINING_PRESETS` (which collided in name space with the v1.3 `LORA_PRESETS`) — were renamed to framework-agnostic and namespace-distinct canonical forms. **The legacy names continue to resolve** via module-level `__getattr__` shims; they emit a `DeprecationWarning` so downstream consumers get a migration nudge. Three-version deprecation cycle.

### TL;DR

- **`backpropagate.ui_security`** — the v1.0 Gradio-prefixed UI-error helpers (`safe_gradio_handler`, `raise_gradio_error`, `raise_gradio_warning`, `raise_gradio_info`, `RequestContext.from_gradio_request`) were renamed to `safe_ui_handler` / `raise_ui_error` / `raise_ui_warning` / `raise_ui_info` / `RequestContext.from_request`. The legacy names continue to import and emit `DeprecationWarning`.
- **`backpropagate.config`** — `TRAINING_PRESETS` was renamed to `MULTI_RUN_PRESETS` to disambiguate from the v1.3-era `LORA_PRESETS` (LoRA-shape preset; CLI `--lora-preset`). Same shape: legacy name resolves with `DeprecationWarning`. **`LORA_PRESETS` was not touched** — it's the source of the user-facing `--lora-preset` flag values.
- **Three-version deprecation cycle, both renames**:
  - **v1.4** — `DeprecationWarning` (silent by default; visible under `python -W default` or pytest `-W default`).
  - **v1.5** — escalates to `UserWarning` (visible to every Python process).
  - **v1.6** — legacy names removed; access raises `AttributeError`.

### Symbol rename 1 — `ui_security` Gradio→UI

The `backpropagate/ui_security.py` module's UI-error helpers were authored against Gradio in v1.0 and kept their `gradio_` prefix through the v1.1.0 Reflex migration as a back-compat surface. v1.4 drops the prefix on the canonical names; the legacy names stay alive as aliases.

| Legacy (v1.0 Gradio era) | Canonical (v1.4+) |
|---|---|
| `safe_gradio_handler` | `safe_ui_handler` |
| `raise_gradio_error` | `raise_ui_error` |
| `raise_gradio_warning` | `raise_ui_warning` |
| `raise_gradio_info` | `raise_ui_info` |
| `RequestContext.from_gradio_request` | `RequestContext.from_request` |
| `DEFAULT_GRADIO_CSP` (v1.4 deprecated since Wave 2) | `DEFAULT_REFLEX_CSP` |
| `get_gradio_csp` (v1.4 deprecated since Wave 2) | `get_reflex_csp` |

**Migration:**

```python
# Old (v1.3) — emits DeprecationWarning since v1.4
from backpropagate.ui_security import (
    safe_gradio_handler,
    raise_gradio_error,
    RequestContext,
)

@safe_gradio_handler("train")
def start_training(...): ...

raise_gradio_error("dataset missing")
ctx = RequestContext.from_gradio_request(request, operation="train")

# New (v1.4+)
from backpropagate.ui_security import (
    safe_ui_handler,
    raise_ui_error,
    RequestContext,
)

@safe_ui_handler("train")
def start_training(...): ...

raise_ui_error("dataset missing")
ctx = RequestContext.from_request(request, operation="train")
```

**`from backpropagate import safe_gradio_handler` is unchanged** — the package-level surface keeps `safe_gradio_handler` as a silent back-compat alias for `safe_ui_handler`. The `DeprecationWarning` fires only when importing the legacy name directly from `backpropagate.ui_security`. If you want to migrate cleanly, import the canonical name; if you don't care, the silent package-level alias keeps existing code working.

**`DEFAULT_GRADIO_CSP` and `get_gradio_csp`** were already deprecated in v1.4 Wave 2 (FRONTEND-A-003) with an in-place `DeprecationWarning` shape — they pointed at `DEFAULT_REFLEX_CSP` / `get_reflex_csp` (introduced for the Reflex-tuned CSP middleware). Wave 6a's rename did NOT introduce a third `DEFAULT_UI_CSP` name — `DEFAULT_REFLEX_CSP` remains the canonical replacement because the production middleware shape is Reflex-specific (script-src `'unsafe-inline'` for the `__NEXT_DATA__` hydration block, drops Google Fonts since Reflex's stylesheet inlines Inter / JetBrains Mono, drops the Hugging Face origin since the UI never directly calls `huggingface.co` from the browser).

### Symbol rename 2 — `TRAINING_PRESETS` → `MULTI_RUN_PRESETS`

A Wave 5 audit (`WAVE_5_FEATURE_AUDIT_NOTES.md`) surfaced a namespace collision: the v1.0-era `TRAINING_PRESETS` table (multi-run loop hyperparameters — `num_runs`, `samples_per_run`, `replay_fraction`) and the v1.3-era `LORA_PRESETS` table (LoRA-architecture shape — `r`, `target_modules`, `lr_multiplier`) BOTH used the keys `"fast"` + `"quality"` with semantically different values. An operator reading "I want the `quality` preset" had no way to tell from the prose which namespace they were addressing — the [reference page → Training presets](/backpropagate/handbook/reference/#training-presets) carries a load-bearing disambiguator paragraph for this reason.

v1.4 disambiguates the names: `TRAINING_PRESETS` → `MULTI_RUN_PRESETS`. `LORA_PRESETS` was **not touched** — it's the source of the user-facing `--lora-preset` flag values. The legacy `TRAINING_PRESETS` name continues to resolve from `backpropagate.config` via a module-level `__getattr__` shim and emits a `DeprecationWarning` pointing at the new name.

**Migration:**

```python
# Old (v1.3) — emits DeprecationWarning since v1.4
from backpropagate.config import TRAINING_PRESETS
preset = TRAINING_PRESETS["balanced"]

# New (v1.4+)
from backpropagate.config import MULTI_RUN_PRESETS
preset = MULTI_RUN_PRESETS["balanced"]

# Either works at the package level (silent, no warning)
from backpropagate import MULTI_RUN_PRESETS  # canonical
from backpropagate import TRAINING_PRESETS   # back-compat alias
```

**`get_preset(name)`** is unchanged — it continues to work on whichever preset table is canonical, so `get_preset("balanced")` returns the same `TrainingPreset` value in v1.3 and v1.4+.

### What did NOT change (v1.3 → v1.4)

- **The `TrainingPreset` dataclass shape.** Field names, defaults, and the `effective_batch_size` property are byte-identical to v1.3.
- **`LORA_PRESETS`** — the user-facing `--lora-preset {fast,quality}` flag values are untouched. v1.3's `LoRAPreset` dataclass + `get_lora_preset` helper continue unchanged.
- **`safe_gradio_handler` from the top-level `backpropagate` package** still imports without warning. The rename surface is `backpropagate.ui_security`, not `backpropagate.__init__`.
- **CSP middleware behavior.** `DEFAULT_REFLEX_CSP` + `get_reflex_csp` were already canonical from Wave 2; v1.4 Wave 6a did not change the production middleware's CSP shape.

### Deprecation cycle (locked advisor 2026-05-25 Q4)

| Version | Behavior |
|---|---|
| v1.4 | `DeprecationWarning` on legacy-name access (silent by default; visible under `-W default`). |
| v1.5 | Escalates to `UserWarning` (visible to every Python process — harder to ignore). |
| v1.6 | Legacy names removed entirely; access raises `AttributeError`. |

If you need to silence the v1.4 warning in CI before migrating, the canonical pattern is:

```python
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*is deprecated in v1.4.*",
    category=DeprecationWarning,
    module=r"backpropagate\.(ui_security|config)",
)
```

But the friendlier path is migrating now while you have a soft fail — v1.5's `UserWarning` is harder to silence cleanly, and v1.6's hard `AttributeError` will fail tests outright.

---

## v1.2.x → v1.3

A polish + truth-in-advertising release. The big shift is that two CLI flags (`--host` and `--share`) that were silently no-ops since v1.1.0 are now actually wired to the runtime — so operators who *thought* they were binding to a network interface or publishing a public URL but never actually were will see different behaviour.

### TL;DR

- **`backprop ui --host <addr>` now actually binds to that address** (was loopback-only since v1.1.0).
- **`backprop ui --share` now actually publishes a public URL** via a real `cloudflared` tunnel (was a silent no-op since v1.1.0).
- **PyInstaller binary distribution removed.** The `npm install -g backpropagate` shim now prints install guidance and exits `2` instead of attempting to download a binary that never existed at the v1.x release tags.
- **`Trainer.train(resume_from=run_id)` is now strict.** A missing `run_id` raises `INPUT_RESUME_NOT_FOUND` rather than silently restarting from scratch.
- **New CLI flag: `--auth-file <path>`** — shell-history-safe alternative to `--auth user:pass`.
- **New env vars + lock-file token** (see [breaking changes](#breaking-changes-v13) below).

### Breaking changes (v1.3)

#### PyInstaller binary distribution removed

**v1.2.x:** the `bin/backpropagate.js` shim shipped with `npm install -g backpropagate` attempted to download a PyInstaller binary for your platform from the GitHub release. The binary build pipeline (`release-binaries.yml`) failed three consecutive times in v1.2.0 — the workflow is now deleted and the `.spec` files are pulled. The v1.0 / v1.1 / v1.2 release tags have zero attached binary assets, so the launcher would 404 on every `npm` install.

**v1.3:** the shim is a **friendly-error shim**. Running `backpropagate` after `npm install -g backpropagate` prints install guidance for the supported channels and exits `2`. The `@mcptoolshop/npm-launcher` runtime dependency was dropped from `package.json` (every `npm install` was pulling dead code).

The decision to delete rather than retry was made by a 4-agent study-swarm: three previous attempts at building cross-platform PyInstaller binaries for a 2 GB+ ML stack failed for distinct reasons (asset-resolution, dependency-bundling, slow-build-on-CI). Continuing the pattern was not a good use of the v1.3 budget.

**Migration:** use one of the supported install channels:

```bash
# Recommended — isolated venv with PATH integration
pipx install backpropagate

# Alternative — same shape, different tool
uv tool install backpropagate

# Or — manage your own venv
pip install backpropagate[standard]

# Or — container
docker pull ghcr.io/mcp-tool-shop-org/backpropagate:1.3.0
```

If you were relying on the v1.0 / v1.1 / v1.2 PyInstaller binary path (you weren't — the binaries never shipped), `pipx install backpropagate` is the closest equivalent: isolated, on-PATH, single-command install.

#### `backprop ui --host` and `--share` now actually do what they advertise

**v1.1.0 → v1.2.x:** both flags were silently no-ops. `--host 0.0.0.0` was validated but never threaded to the Reflex subprocess argv — the UI silently stayed loopback-only. `--share` was a leftover from the Gradio era; the Reflex migration removed the underlying tunnel and nothing replaced it.

**v1.3:**

- **`--host <addr>`** now flows through to the Reflex backend bind via the `--backend-host` argument that landed in reflex 0.9.2. `backprop ui --host 0.0.0.0` actually publishes the UI on every interface.
- **`--share`** implements a real `cloudflared`-based tunnel. The CLI shells out to `cloudflared tunnel --url http://127.0.0.1:<port>`, parses the announced `https://*.trycloudflare.com` URL from the daemon's stderr, and adds it to the `Host` / `Origin` allowlist the auth middleware already enforces.

Both paths still require `--auth user:pass` (or the new `--auth-file <path>` — see below). The refuse-to-start contract from v1.2.0 is unchanged: a public URL or non-loopback bind without credentials is the v1.1.x bug — see [security → four-layer defense in depth](/backpropagate/handbook/security/#four-layer-defense-in-depth) — and the gate still fires.

**Migration:**

- **If you were passing `--host 0.0.0.0` in v1.2.x and assuming it was working:** it wasn't. Your UI was loopback-only. On v1.3, the same command will actually publish on `0.0.0.0`. Make sure `--auth user:pass` is set, and review your firewall (you almost certainly do NOT want `0.0.0.0` in a shared environment). SSH port-forwarding is the lower-friction alternative — see [security → SSH port-forwarding recipe](/backpropagate/handbook/security/#ssh-port-forwarding-recipe).
- **If you were passing `--share` in v1.2.x and assuming it published a URL:** it didn't (and the auth-middleware refuse-to-start gate fired anyway, so most operators saw `[RUNTIME_UI_AUTH_NOT_ENFORCED]` instead of getting a URL). On v1.3, install `cloudflared` from <https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/> and the announced URL will appear in the startup banner. If you can't install `cloudflared`, SSH port-forwarding is the recommended alternative.

#### Strict `resume_from` behaviour

**v1.2.x:** passing `Trainer.train(resume_from=<missing_run_id>)` (or `backprop resume --run-id <missing_run_id>`) would log a WARN about the missing checkpoint and silently start a fresh run from step 0 with a new `run_id`. Operators tracking specific run IDs would find them mysteriously replaced by a different ID with no error signal.

**v1.3:** the trainer raises `INPUT_RESUME_NOT_FOUND` (exit code `1`, not retryable). If the on-disk state is gone (history record deleted, checkpoint directory wiped), you have to acknowledge that explicitly by omitting `resume_from` or passing `resume_from=None`.

**Migration:**

```python
# Old (v1.2.x) — silent fallback to fresh start
trainer.train(data, resume_from=maybe_missing_run_id)

# New (v1.3) — explicit handling
from backpropagate import InputError
try:
    trainer.train(data, resume_from=maybe_missing_run_id)
except InputError as e:
    if e.code == "INPUT_RESUME_NOT_FOUND":
        # operator decision: start fresh, or surface to user
        trainer.train(data)
    else:
        raise
```

#### Operators relying on `pip install backpropagate[observability]`

This extra was removed in **v1.2.0**, not v1.3 — but it's worth re-flagging because v1.3 has no new mention of it and operators upgrading directly from v1.1.x to v1.3 may still hit it. The extra advertised OpenTelemetry distributed tracing but **zero modules imported `opentelemetry`** — it was a doc-lie.

**v1.2.0+:** `pip install backpropagate[observability]` fails with "no matches found." The `[full]` bundle no longer pulls it.

**Migration:** drop `observability` from your install line. If you depended on OpenTelemetry yourself, install it directly (`pip install opentelemetry-api opentelemetry-sdk`) — Backpropagate never used either package, so installing them at the top level of your project gives you the same surface (which is to say: nothing wired to Backpropagate's runtime). Real OpenTelemetry integration may land in a future release.

### Added (v1.3)

#### `--auth-file <path>` CLI flag

A shell-history-safe alternative to `--auth user:pass`. Reads a `user:pass` line from a file. Mutually exclusive with `--auth` — passing both exits `1` with `INPUT_AUTH_INVALID_SHAPE`. Satisfies the same `--share` / `--host <non-loopback>` gate that `--auth` does.

```bash
echo -n "alice:super-secret-password" > ~/.config/backpropagate/auth
chmod 600 ~/.config/backpropagate/auth
backprop ui --share --auth-file ~/.config/backpropagate/auth
```

See [recipes → --auth-file](/backpropagate/handbook/recipes/#use---auth-file-for-shell-history-safe-auth) for the full recipe.

#### Per-launch lock-file token

`backprop ui` (in `token_auto` mode — the default when neither `--auth` nor `--auth-file` is passed) now writes a per-launch random token to `$XDG_RUNTIME_DIR/backpropagate/session-<port>.lock` (Linux/macOS) or `%LOCALAPPDATA%\backpropagate\session-<port>.lock` (Windows). File permissions are `0600`. The file is deleted on shutdown.

This is the same token that already appeared in the URL — the lock file gives a parallel process running as the same user a way to discover the token without screen-scraping the startup banner. Read by external tooling that wants to validate against the running UI; consumed by `backprop info --runtime` (when present). The token itself is unchanged from v1.2.0; only the discovery surface is new.

#### Audit-trail log line for successful auth

v1.1.x had no log line for a successful cookie-set on the GHSA-f65r-h4g3-3h9h surface — operators could see failed-auth lines (close code `4401` / `4403`) but never knew which cookie just succeeded. v1.3 emits one `auth_success` INFO line per session at the cookie-set sites (both `token_auto` and `explicit_creds` / `production` modes), with `{user, mode, host}` fields. Per-request validation passes log at DEBUG. **No cookie value, no password, no Basic-header bytes are recorded** — the line is safe to ship to a central log aggregator.

### Behavioural fixes (v1.3)

- **CI gates re-tightened.** No operator-visible change unless you were depending on a previously-advisory gate to silently fail-open. Hard floors restored: mypy hard, `pip-audit` CRITICAL floor, Trivy CRITICAL floor.
- **`release.yml` is idempotent + has `workflow_dispatch`.** No operator-visible change unless you were a maintainer; mentioned here for completeness.

### What did not change (v1.2.x → v1.3)

- The auth middleware contract (`rx.App(api_transformer=basic_auth_transformer)` and the four-layer refuse-to-start defense) is identical to v1.2.0. If you were running a working v1.2.0 `backprop ui --auth user:pass` setup, v1.3 is drop-in.
- Public Python API surface (`Trainer`, `MultiRunTrainer`, `SLAOMerger`, `export_lora`, `export_gguf`, callback hooks).
- CLI subcommand names + canonical flags. New: `--auth-file` (additive, doesn't break existing `--auth` callers).
- Run-history schema, checkpoint manifest schema, error-code names. New code: `INPUT_RESUME_NOT_FOUND` (additive).
- Environment variable names — every `BACKPROPAGATE_*` knob keeps its v1.2.x meaning.

---

## v1.1.x → v1.2.0

## TL;DR

If you are on v1.1.x and you have not been relying on the v1.0 Gradio UI, the v1.2.0 upgrade is **mostly drop-in**. The behavioural changes are:

- The Reflex UI now ships a real FastAPI auth middleware via `rx.App(api_transformer=basic_auth_transformer)`. `backprop ui --auth user:pass`, `backprop ui --share --auth user:pass`, and `backprop ui --host 0.0.0.0 --auth user:pass` now actually enforce the credential on every HTTP route and on the `/_event` WebSocket upgrade. Without `--auth`, `--share` and non-loopback `--host` refuse to start with `[RUNTIME_UI_AUTH_NOT_ENFORCED]`.
- The `[observability]` extra is **removed** (it never wired anything in v1.1.x — it was a doc-lie).
- Several training-time hooks that were silently no-ops in v1.1.x now **actually fire** — see [Behavioural fixes](#behavioural-fixes) below.

## Breaking changes

### `backprop ui --share` now requires `--auth`

**v1.1.0 / v1.1.1:** `--share` advertised auth enforcement but the Reflex runtime never read `BACKPROPAGATE_UI_AUTH`. Running `backprop ui --share --auth user:pass` published an **unauthenticated** public URL — see the [GHSA-f65r-h4g3-3h9h advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/GHSA-f65r-h4g3-3h9h) (CVSS 9.8, published 2026-05-23) for the full disclosure.

**v1.2.0:** the FastAPI auth middleware is live and enforces credentials on every request and the `/_event` WebSocket upgrade. `backprop ui --share --auth user:pass` now actually publishes an authenticated URL; `backprop ui --share` **without** `--auth` exits `1` with `[RUNTIME_UI_AUTH_NOT_ENFORCED]` so an operator cannot accidentally re-create the v1.1.x bug. The refuse-to-start contract is layered four deep (CLI gate, ambient-env strip, `ui_app/app.py` import guard, `rxconfig.py` import guard) — see [the security page](/backpropagate/handbook/security/#four-layer-defense-in-depth) for the full chain.

**Migration:**

- **If you were running `backprop ui --share` and a colleague accessed the URL without credentials in v1.1.x**: that URL was unauthenticated. Rotate any `HF_TOKEN` that was on the host while v1.1.x was running. On v1.2.0, run `backprop ui --share --auth user:pass` and share both the URL and credentials over your normal secure channel.
- **If you have no need for a public URL**: SSH port-forwarding stays the lowest-friction option.

```bash
# On the training host:
backprop ui

# On your laptop:
ssh -L 7860:localhost:7860 you@training-host
# Then open http://localhost:7860 locally.
```

### v1.0 Gradio UI fully removed

**v1.1.x:** `backpropagate.ui_gradio_legacy` and `backpropagate.theme_gradio_legacy` were preserved as reference modules; package-level helpers raised `ImportError` with a migration message.

**v1.2.0:** the modules are deleted. The package-level helpers still raise `ImportError` (via `__init__.py.__getattr__`), so existing `from backpropagate import launch` call sites get the same error pointing at `backprop ui`.

**Migration:** use the Reflex UI (`pip install backpropagate[ui]` then `backprop ui`).

### `[observability]` extra removed

**v1.1.x:** `pip install backpropagate[observability]` installed `opentelemetry-api` + `opentelemetry-sdk`, but **zero modules imported them**. The extra was a doc-lie.

**v1.2.0:** the extra is gone. `pip install backpropagate[observability]` now fails with "no matches found." The `[full]` bundle no longer includes it.

**Migration:** drop `observability` from your install line. If you depended on OpenTelemetry yourself, install it directly (`pip install opentelemetry-api opentelemetry-sdk`) — backpropagate never used either package. Real OpenTelemetry integration may land in a future release.

## Behavioural fixes

These are not breaking changes in the API-shape sense, but if your code was depending on the *broken* v1.1.x behaviour, it will now behave differently.

### `TrainingCallback.on_step` / `on_epoch` / `on_save` now actually fire

**v1.1.x bug (BACKEND-F-003):** `TrainingCallback` subclasses were instantiated and wired up, but the Trainer never invoked the hook methods.

**v1.2.0:** the hooks fire at their documented points. If your callback was instrumenting metrics, expect to see calls you didn't see before.

### `Trainer.train(resume_from=run_id)` now actually resumes

**v1.1.x bug (BACKEND-F-017):** `resume_from=...` was accepted, the run-history record was reloaded, but the optimizer / scheduler / step counter were not — training silently started from step 0 with a fresh optimizer.

**v1.2.0:** the resume path restores optimizer state and step counter. A 5-step run that crashes at step 3 now resumes at step 3, not step 0.

### `train_on_responses_only` is now tokenizer-aware

**v1.1.x bug (BACKEND-F-014):** the helper masked user-prompt tokens by matching ChatML literal strings (`<|im_start|>user`). For Llama 3 / Gemma / Qwen2 — which use different chat-template tokens — the matcher silently never fired and the model trained on user prompts as well as assistant responses.

**v1.2.0:** the matcher uses the tokenizer's chat-template to find the user→assistant boundary. Llama 3 / Gemma / Qwen2 chat templates now mask correctly.

**Migration:** if you were getting bad fine-tunes on Llama 3 / Gemma / Qwen2 with `train_on_responses_only=True`, re-run on v1.2.0 — the loss curves will look different (lower loss on assistant tokens, no gradient on user tokens) and downstream eval should improve.

### `backprop info` no longer mentions Gradio

**v1.1.x bug (BRIDGE-F-011):** `backprop info` reported the UI implementation as "Gradio" even after v1.1.0's migration to Reflex.

**v1.2.0:** reports "Reflex."

## What did not change

- Public Python API surface (`Trainer`, `MultiRunTrainer`, `SLAOMerger`, `export_lora`, `export_gguf`).
- CLI subcommand names + canonical flags (`backprop train` / `multi-run` / `export` / `push` / `resume` / `list-runs` / `show-run` / `info` / `ui`).
- Run-history schema, checkpoint manifest schema, error-code names.
- Environment variable names — every `BACKPROPAGATE_*` knob keeps its v1.1.x meaning.

## Rollback

If you need to roll back to v1.1.1 (note: the v1.1.x auth advertisement is [GHSA-f65r-h4g3-3h9h](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/GHSA-f65r-h4g3-3h9h) — CVSS 9.8 Critical — and any `--share` URL while v1.1.x was running was unauthenticated; see the [security page](/backpropagate/handbook/security/) for full disclosure):

```bash
pip install "backpropagate==1.1.1"
```

The v1.0 Gradio UI is **not** recoverable by rolling back to v1.1.x because the legacy modules continued to raise `ImportError` there. To use the Gradio UI you would have to pin to `backpropagate==1.0.5`.
