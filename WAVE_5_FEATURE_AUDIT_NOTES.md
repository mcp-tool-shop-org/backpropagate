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
