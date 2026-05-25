# Wave 5 Feature Audit Notes

Forward-looking feature questions surfaced during Wave 2 (Stage A) amend.
Wave 5 (feature audit) should evaluate each and decide accept/defer/reject.

## From CIDOCS agent — v1.4 Wave 2

### Should `backprop ollama register` exist as a real CLI subcommand?

- **Surfaced by:** CIDOCS-A-001 fix. The README hero example was advertising
  `backprop ollama register ./output/lora --name my-model`, which does not
  exist — operators hit an argparse "invalid choice" error on the very
  first invocation. Wave 2 fixed the doc (replaced with the working
  `backprop export ... --ollama --ollama-name my-model` form from line 270).
- **The question for Wave 5:** the invented subcommand form is meaningfully
  cleaner UX than the long export-flag form. An operator who has already
  trained / exported the LoRA and just wants to push it into Ollama should
  not need to re-specify `--format gguf --quantization q4_k_m`. A dedicated
  `backprop ollama register <path> [--name <name>] [--modelfile <path>]`
  subcommand would:
  - Map 1:1 with the Ollama mental model (`ollama create`, `ollama list`,
    `ollama rm` — the ergonomics operators already know).
  - Reuse the existing `register_with_ollama(...)` helper that
    `cli.py:cmd_export` already calls when `--ollama` is passed.
  - Let `backprop ollama list` and `backprop ollama rm <name>` round out
    a small subparser group, matching the `runs` / `multi-run` / `info`
    grouping pattern already in the CLI.
- **Cost:** small. The lift is one new `cmd_ollama_*` family in `cli.py`
  (~80 lines including argparse plumbing) plus handbook + README updates.
  No new dependencies; the Ollama HTTP API is already wrapped in
  `backpropagate/export.py` for the `--ollama` flag path.
- **Risk:** very low. Strictly additive — the `backprop export --ollama
  --ollama-name` form stays as the one-shot export+register path; the new
  subparser is for the "I already exported, just register" case.
- **Wave 5 verdict needed:** accept (build it in v1.4) / defer (v1.5+) /
  reject (keep the export-flag form as canonical).

## From CIDOCS agent — v1.4 Wave 3.5

### TRAINING_PRESETS vs LORA_PRESETS namespace collision (operator-trap class)

- **Surfaced by:** CIDOCS-B-008 drift sweep across handbook tree. While
  fixing `lora_r | 16` drift across 5 handbook files, the CI-Docs agent
  noticed the source has TWO distinct preset namespaces both using the
  names `fast` / `quality` with DIFFERENT semantics + DIFFERENT defaults:
  - `TRAINING_PRESETS["quality"]` (rank 32, multi-run loop hyperparameters)
  - `LORA_PRESETS["quality"]` (rank 256, all-linear, 10× LR — v1.3 BACKEND-1
    quality preset selected via `--lora-preset`)
- **The operator-trap shape:** an operator picking `Trainer(preset="fast")`
  could get either depending on which import path resolved or which
  config the operator was reading. The handbook's `reference.md:90` got a
  disambiguator note in Wave 3.5; the underlying collision is the real fix.
- **The question for Wave 5:** rename one namespace to remove the collision.
  Renaming is a breaking change for callers that explicitly import the
  symbols, so it needs deprecation-alias treatment similar to the
  `ui_security` Gradio→UI rename (V1_4_BRIEF item 7). Options:
  - **Option A:** ship the rename + aliases in v1.4 alongside the
    `ui_security` rename (parallel deprecation cycles — symmetric scope,
    cohesive v1.4 narrative around naming hygiene).
  - **Option B:** defer to v1.5 carryforward (smaller v1.4 scope, but the
    operator-trap persists one more release).
- **Suggested rename targets:**
  - Keep `LORA_PRESETS` as-is (it's the user-facing `--lora-preset` flag's
    underlying config; the names match the flag values).
  - Rename `TRAINING_PRESETS` → `MULTI_RUN_PRESETS` or `TRAINING_LOOP_PRESETS`
    (it's the multi-run loop config, semantically distinct from LoRA
    architecture defaults). Deprecation alias forward `TRAINING_PRESETS`
    → new name + `DeprecationWarning` per module-level `__getattr__`.
  - The handbook's `reference.md:90` disambiguator note then graduates to
    a Migration section pointing operators at the new name.
- **Cost:** small. ~50-100 lines including deprecation alias + handbook
  migration entry + tests for the alias path.
- **Risk:** very low. Strictly additive aliases; no breaking change in
  v1.4 (the rename + DeprecationWarning is the same shape as v1.4's
  `ui_security` Gradio→UI rename — well-trodden pattern).
- **Wave 5 verdict needed:** accept (parallel with ui_security in v1.4) /
  defer (v1.5+).
