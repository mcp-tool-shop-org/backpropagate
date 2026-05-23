---
title: Migrating to v1.2.0
description: What changed since v1.1.x and how to update your code.
sidebar:
  order: 10
---

This page covers the v1.1.x → v1.2.0 upgrade. For older transitions (v1.0 → v1.1, the Gradio → Reflex pivot) see the [v1.1.0 CHANGELOG section](https://github.com/mcp-tool-shop-org/backpropagate/blob/main/CHANGELOG.md#110---2026-05-21).

## TL;DR

If you are on v1.1.x and you have not been relying on the v1.0 Gradio UI, the v1.2.0 upgrade is **mostly drop-in**. The behavioural changes are:

- `backprop ui --share` (with or without `--auth`) **refuses to start** until the auth middleware lands. Until then, use SSH port-forwarding.
- The `[observability]` extra is **removed** (it never wired anything in v1.1.x — it was a doc-lie).
- Several training-time hooks that were silently no-ops in v1.1.x now **actually fire** — see [Behavioural fixes](#behavioural-fixes) below.

## Breaking changes

### `backprop ui --share` hard-refuses without `--auth`

**v1.1.0 / v1.1.1:** `--share` advertised auth enforcement but the Reflex runtime never read `BACKPROPAGATE_UI_AUTH`. Running `backprop ui --share --auth user:pass` published an **unauthenticated** public URL — see the [GHSA advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories) for details.

**v1.2.0:** `backprop ui --share` (with or without `--auth`) exits `1` with `[RUNTIME_UI_AUTH_NOT_ENFORCED]`. The refuse-to-start contract is enforced in four layers — see [the security page](/backpropagate/handbook/security/#four-layer-defense-in-depth) for the full chain.

**Migration:** use SSH port-forwarding instead of `--share`:

```bash
# On the training host:
backprop ui

# On your laptop:
ssh -L 7860:localhost:7860 you@training-host
# Then open http://localhost:7860 locally.
```

Real `--share` support returns when the auth middleware lands; tracked in the [security page](/backpropagate/handbook/security/).

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

If you need to roll back to v1.1.1 (note: the v1.1.x auth advertisement is the GHSA-pending advisory — see the [security page](/backpropagate/handbook/security/)):

```bash
pip install "backpropagate==1.1.1"
```

The v1.0 Gradio UI is **not** recoverable by rolling back to v1.1.x because the legacy modules continued to raise `ImportError` there. To use the Gradio UI you would have to pin to `backpropagate==1.0.5`.
