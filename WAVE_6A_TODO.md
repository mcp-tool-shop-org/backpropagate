# Wave 6a Foundation TODO — items deferred from earlier waves

## CLOSED in Wave 6b features (2026-05-25 — TESTS amend)

The Wave 5 Tests audit named 6 HIGH coverage gaps. All six are now pinned
in regression tests alongside the v1.4 fixes they cover. The fixes
themselves had already landed in earlier waves — what was missing was the
mechanical regression net so a future change couldn't silently un-fix
them. Per the inherited `[[grep-all-instances-when-fixing-pattern]]`
doctrine, the schema_version pin covers ALL documented `--json` emitters
(not just one).

- **TESTS-F-001 — Stage C BACKEND-B-001 save_merged downgrade warning**
  — CLOSED. New tests at
  `tests/test_trainer.py::TestTrainerSaveMerged::test_save_merged_with_no_unsloth_emits_warning`
  + `::test_save_merged_with_unsloth_does_not_emit_b001_warning` (the
  negative pin for the AND-coupled gate). Pins the WARN fires when
  `save_merged=True AND use_unsloth=False` and STAYS QUIET on the
  supported Unsloth + merged path. Sourced from trainer.py:2796.
- **TESTS-F-002 — Wave 3.5 BACKEND-B-004 untrained-save warning** —
  CLOSED. Three tests:
  `::test_has_trained_flag_initializes_false` (init pin),
  `::test_save_without_train_emits_warning` (tripwire pin),
  `::test_save_after_train_does_not_emit_untrained_warning` (negative
  pin so post-train saves stay silent). Sourced from trainer.py:2746 +
  the `_has_trained: bool = False` init at trainer.py:1685.
- **TESTS-F-003 — Wave 3.5 BACKEND-B-003 model.eval() try/finally** —
  CLOSED. Three tests in a new
  `tests/test_multi_run.py::TestWave6bComputeValidationLossTryFinally`
  class: the KeyboardInterrupt-mid-loop path (BaseException bypasses
  the inner `except Exception:`), the `torch.no_grad()`-raises-on-entry
  path (corrupt CUDA context), and a source-level pin that asserts
  `model.eval()` lives INSIDE a `try:` block within the method body
  (catches the broader pattern even when the runtime path doesn't fire
  in tests). Sourced from multi_run.py:2814-2752.
- **TESTS-F-004 — Wave 3.5 FRONTEND-B-001 was_deleted + Stage C
  FRONTEND-B-014-EXTENDED action_in_flight** — CLOSED. New
  `tests/test_ui_states.py::TestRunDetailStateDefaults` (4 cases —
  was_deleted / not_found / action_in_flight / action_result+action_error
  defaults), `::TestRunDetailStateDeleteRun` (3 cases — success sets
  was_deleted, failure does not, load_run resets was_deleted on
  remount), `::TestRunDetailStateActionInFlight` (6 cases — clear after
  success / failure / exception, plus the
  `[[grep-all-instances-when-fixing-pattern]]` sweep across diff /
  replay / export handlers). Sourced from ui_state.py:1536-1955.
- **TESTS-F-005 — BRIDGE-B-013 schema_version pinning** — CLOSED.
  Per the `[[grep-all-instances-when-fixing-pattern]]` ratchet, the
  pin covers ALL 8 documented `--json` emitters in one class
  `tests/test_cli.py::TestCliJsonSchemaVersion`: info / info --env-vars
  / list-runs / runs / show-run / diff-runs --format=json / export-runs
  JSONL / estimate-vram. The constant `CLI_JSON_SCHEMA_VERSION` itself
  is also pinned for shape (string type, non-empty). Sourced from
  cli.py:3610 + every emitter's per-payload injection site. NOTE:
  `info --error-codes` emits a plaintext catalog table today (not
  JSON) — when an --error-codes --json surface lands, the matching
  test should be added in this class.
- **TESTS-F-006 — BRIDGE-A-004 _read_hub_token_file** — CLOSED. New
  `tests/test_cli.py::TestReadHubTokenFile` with 7 cases covering
  every documented error path: happy path with stripped token; missing
  path → UserInputError with named-path hint + INPUT_VALIDATION_FAILED
  code; empty file → UserInputError with 'empty' in message;
  whitespace-only file → same; trailing newline stripped; POSIX-only
  mode 0644 → warning emits + token returned; flag_name kwarg threads
  into the error message so callers using `--token-file` (vs the
  current `--hub-token-file` spelling) see the right flag named.
  Sourced from cli.py:1001.

### Wave 6b TESTS doctrine ratchets

- **`[[grep-all-instances-when-fixing-pattern]]` (applied to TESTS-F-005)**
  — the schema_version pin covers ALL 8 documented `--json` emitters,
  not just one. A single-surface test would have only caught a partial
  regression; the sweep catches the canonical "drops the field on one
  emitter while leaving the others intact" footgun. Same doctrine
  applied to TESTS-F-004 by pinning action_in_flight clear-on-exit
  across all 4 sibling handlers (delete / diff / replay / export) not
  just delete_run.
- **`[[no-banner-documenting-no-op]]`** — every new test asserts a
  real observable behavior (warning emitted, attribute set, model
  method called, JSON field present, exception raised) — no "test
  exists" boilerplate.
- **`[[within-swarm-doc-lie-drift-detection]]`** — every test
  references the current state (e.g. "Stage C BACKEND-B-001 warning at
  trainer.py:2796", "FRONTEND-B-001 contract at ui_state.py:1536").
  When a fix-site moves in a future refactor, the next test author
  greps the test message and finds the new site instead of stale
  doctrine.

### Coordination outcomes — captured at amend time

- **Backend's mode='full' / RUNTIME_FULL_FT_MODEL_TOO_LARGE / estimate_vram
  coverage:** already covered by Backend in `tests/test_wave6b_features.py`
  (TestTrainerModeFullGate, TestEstimateParamCount, TestMultiRunTrainerModeFull,
  TestEstimateVRAM — 29 cases total). No duplicate tests added by TESTS
  amend — the Backend surfaces are already pinned by Backend's own
  regression net.
- **Bridge's `backprop ollama register|list|rm` triad + `--log-level` /
  `--log-format` / `--log-file` / `--mode=full` / `--hub-revision` /
  `--hub-commit-message` / `--subcommand-tiers` flags:** at TESTS amend
  time these were NOT yet on the wave-6b/features branch's cli.py. They
  will be covered by Bridge agent's amend (test class names in Bridge's
  scope: TestOllamaSubcommandTriad, TestRootParserLoggingFlags, etc.).
  TESTS-F coverage stays scoped to the 6 Wave 5 audit gaps; Bridge's
  new-flag coverage is owned by Bridge.

### Path D escalations

None. The 6 coverage gaps all closed cleanly using the existing
test-shape patterns (caplog for warnings, monkeypatch for shutil/
subprocess in UI state handlers, MagicMock + real-class hybrids for
the multi_run validation loop, capsys+json.loads for the CLI
schema_version emitters, tmp_path + os.chmod for the token-file
mode-check).

## CLOSED in Wave 6b features (2026-05-25 — BRIDGE amend)

- **Item 1 / Wave 5 Decision 4 — `backprop ollama register|list|rm` triad** — CLOSED.
  - New nested subparser `backprop ollama` with three sub-actions registered in `cli.py:create_parser`. Mirrors upstream Ollama CLI shape (`ollama create`, `ollama list`, `ollama rm`) 1:1 so operators paste-and-prefix tutorials without translation.
  - `cmd_ollama_register` wraps existing `register_with_ollama()` (export.py:1566). Resolves `--name` default from the GGUF filename stem; tolerates being pointed at a directory containing a single `*.gguf`. Maps `DEP_OLLAMA_REGISTRATION_FAILED` → EXIT_UNAVAILABLE so wrappers can distinguish "daemon down" from "user passed bad name".
  - `cmd_ollama_list` wraps existing `list_ollama_models()` (export.py:1665). Distinguishes "no models registered" (EXIT_OK) from "Ollama CLI missing on PATH" (EXIT_UNAVAILABLE) via `shutil.which("ollama")` probe — the underlying helper currently returns `[]` for both cases.
  - `cmd_ollama_rm` wraps NEW `remove_ollama_model()` helper in export.py (added in this wave). Helper re-uses the register-side allowlist (`_validate_model_name`) so a name that was safe to create is the same shape that is safe to remove; targeted "not found" / "connection refused" suggestion strings.
  - `remove_ollama_model` re-exported from `backpropagate.__init__` alongside `register_with_ollama` / `list_ollama_models`.
  - **Architectural deviation note (load-bearing, recorded in code comment):** every other backpropagate subcommand is FLAT (`backprop train`, `backprop export`, …). The ollama triad is INTENTIONALLY NESTED (`backprop ollama register`, NOT `backprop ollama-register`). Reasoning: upstream Ollama uses the same shape; grouping by noun matches the operator mental model; closes CIDOCS-A-001 retroactively (the README hero example `backprop ollama register …` was a doc-lie pre-Wave-6b — Feature pass made it real). Future maintainers must NOT refactor to flat. SUBCOMMAND_TIERS marks `ollama` as `experimental` so the contract can absorb iteration in v1.5.

- **Item 2 / BRIDGE-F-002 — root `--log-level` / `--log-format` / `--log-file` CLI flags** — CLOSED.
  - Three new root-level flags on the top-level parser (visible on every subcommand). Each maps to the corresponding `BACKPROPAGATE_LOG_*` env var:
    - `--log-level` → `BACKPROPAGATE_LOG_LEVEL` (choices: DEBUG / INFO / WARNING / ERROR)
    - `--log-format=console|json` → `BACKPROPAGATE_LOG_JSON=false|true`
    - `--log-file <path>` → `BACKPROPAGATE_LOG_FILE`
  - Wiring lives in `main()` BEFORE `configure_logging` fires — flag overwrites env var via `os.environ[...] = value`. CLI flag wins over env var per the standard precedence rule (CLI > env > default). Overwrite is process-scoped — does NOT leak to the operator's shell after backprop exits.
  - Pre-fix the three knobs were env-var-only; operators could not tweak per-invocation logging from the CLI surface.

- **Item 3 / BRIDGE-F-007 — `--json` on validate + replay** — CLOSED.
  - `backprop validate <path> --json` emits a `schema_version`-carrying JSON payload with `dataset / lines_scanned / samples_parsed / format_hint / format_detected / total_rows / valid_rows / is_valid / parse_errors / errors / warnings`. Mirrors the existing `--json` envelope pattern (CLI_JSON_SCHEMA_VERSION = "1"). Error branches (missing file, non-UTF8, no parseable rows) ALSO emit JSON with the same `schema_version` shape + a top-level `error` discriminator.
  - `backprop replay <run_id> --json` emits a payload with `session_kind / original_run_id / replay_cli_run_id / new_run_id / model / dataset / overrides / final_loss` (multi-run variant adds `total_runs`).
  - Both flags route through the existing `CLI_JSON_SCHEMA_VERSION` constant so the bump policy stays uniform across the `--json` surface.

- **Item 4 / BRIDGE-F-010 — `info --json` logging block** — CLOSED.
  - `info --json` payload now contains a top-level `logging` block: `{level, format, file, json_var_set}`. Values read from the LIVE env vars (which the root `--log-level` / `--log-format` / `--log-file` flags overwrite in `main()` before `cmd_info` is invoked), so the snapshot reflects what was ACTUALLY applied — not what the operator typed.
  - `format` is computed using the same auto-detect heuristic as `logging_config._should_use_json` (JSON when stderr is NOT a TTY).
  - `json_var_set` boolean discriminates "auto-detected" from "explicitly overridden".

- **Item 5 / BRIDGE-F-011 — root `--help` epilog rewrite** — CLOSED.
  - Root `--help` epilog enumerates ALL 18 subcommands grouped by workflow: Training / Inspection / Export / Ollama / UI. Pre-fix listed only 5 of 18 subcommands (the original v1.0 set); operators discovering v1.3 additions (replay / diff-runs / validate / estimate-vram / runs / export-runs / push / resume) had to grep source. Tests pin the workflow-group headers + every subcommand name so a future addition / removal that updates argparse but skips the epilog fires a drift signal at CI time.

- **Item 6 / BRIDGE-F-014 — `push --hub-revision` / `--hub-commit-message`** — CLOSED.
  - Two new optional flags on `push_parser`: `--hub-revision <branch>` (push to a non-default branch — Hub auto-creates it) and `--hub-commit-message <msg>` (custom commit message overriding the `"Upload via backpropagate"` default).
  - Threaded through `cmd_push` to `push_to_hub(..., revision=args.hub_revision, commit_message=args.hub_commit_message)`. The `push_to_hub` helper already accepted both kwargs (export.py:448-459, `revision=` already passed to `api.upload_folder`); Wave 6b only exposes them on the CLI surface.

- **Item 7 / BRIDGE-F-015 — `info --subcommand-tiers` introspection** — CLOSED.
  - `backprop info --subcommand-tiers` prints the SUBCOMMAND_TIERS registry as an aligned table (sorted by tier rank, color-coded green / yellow / red). With `--json` emits a `schema_version`-carrying payload.
  - Closes the `[[no-banner-documenting-no-op]]` tripwire flagged by Wave 1: the registry was DEFINED at cli.py:151 and READ by the deprecation-hint emitter but had no operator surface. cmd_ui / cmd_export comments referencing a never-registered `--subcommand-tiers` flag — Wave 6b retires those (the deprecation-hint emitter at the bottom of `main()` now points at the now-real flag).
  - Added `validate / estimate-vram / ollama` to SUBCOMMAND_TIERS (validate + estimate-vram stable; ollama experimental).

- **Cross-domain item — Backend's `--mode=full` CLI flag + estimate-vram per-config** — CLOSED.
  - `--mode={lora,full}` on `train` and `multi-run` parsers (default `lora` — backward-compat). Threads through `wave6b_candidate_kwargs` to `Trainer(mode=...)` / `MultiRunConfig.mode` (per Backend's BACKEND-F-008 wiring). The existing introspection filter at `cmd_train` / `cmd_multi_run` drops the kwarg silently when the installed Trainer doesn't accept it (pre-Wave-6b builds), so the flag is forward-compatible.
  - `--mode` added to `_REPLAY_ALLOWED_OVERRIDE_KEYS` and `_wave6b_keys` so `backprop replay <run_id> --override mode=full` is a real path.
  - `estimate-vram` parser grew `--lora-r N`, `--batch-size N`, `--mode {lora,full}` flags. When `--batch-size` is supplied (or `--mode=full`), `cmd_estimate_vram` additionally calls backend's module-level `estimate_vram()` function (trainer.py:877) for a per-config estimate (model_weights / lora_adapter / optimizer_state / activations / overhead / total). Human view warns when estimated total exceeds detected VRAM; JSON envelope adds the `per_config_estimate` block + top-level `mode` field. Tier table view continues to ship side-by-side so operators get both signals (general "what's safe" + specific "will THIS config OOM") in one invocation.

### Tests added (Wave 6b BRIDGE)

`tests/test_wave6b_cli_features.py` — 38 cases total across:

- `TestOllamaSubparser` (5) — argparse parse contract for ollama register|list|rm
- `TestRemoveOllamaModel` (5) — backend helper (subprocess.run mocked); reject leading dash; targeted "not found" suggestion; package re-export
- `TestCmdOllamaHandlers` (6) — three handler smoke tests with mocked subprocess; name resolution from filename stem; daemon-missing → EX_UNAVAILABLE
- `TestRootLoggingFlags` (8) — `--log-level` / `--log-format` / `--log-file` parse + main()-time env-var overwrite + choices validation
- `TestValidateJsonFlag` (3) — parse + missing-file payload + clean-dataset payload
- `TestReplayJsonFlag` (1) — parse contract
- `TestInfoJsonLogging` (2) — `logging` block presence + `BACKPROPAGATE_LOG_FILE` propagation
- `TestRootHelpEpilog` (2) — every subcommand + workflow group in epilog
- `TestPushHubFlags` (4) — flag parse + kwarg threading + None default
- `TestSubcommandTiers` (3) — parse + human output + JSON envelope
- `TestModeFlag` (8) — train/multi-run/estimate-vram defaults + `full` choice + invalid choice + replay override whitelist

### Coordination outcomes (Wave 6b BRIDGE)

- **BACKEND mode='full' wire-through**: `--mode={lora,full}` flag threads through `wave6b_candidate_kwargs` end-to-end on `train` + `multi-run` + `replay`. Per Backend's BACKEND-F-008 closure, `Trainer(mode=...)` is the load-bearing surface; the introspection filter at `cmd_train` drops the kwarg silently against pre-Wave-6b Trainer builds so the CLI flag is forward-compatible.
- **TESTS coverage**: BRIDGE-owned tests live in `tests/test_wave6b_cli_features.py` per the WAVE_6A_TODO.md "Bridge's new-flag coverage is owned by Bridge" carve-out. TESTS scope deliberately stayed off these surfaces.
- **CI-DOCS handbook entries**: each new CLI surface (root `--log-*` flags / ollama nested subparser / `info --subcommand-tiers` / `push --hub-*` / `validate --json` / `replay --json` / `--mode` / `estimate-vram --lora-r --batch-size`) needs a handbook entry in `site/src/content/docs/handbook/cli-reference.md`. The drift gate at PR-merge time will fail if CI-DOCS doesn't land them — coordination handoff via this closure note.

### Architectural-deviation note for ollama nested subparser (for the handbook)

The `backprop ollama {register,list,rm}` subparser is the first nested subparser in backpropagate. Every other subcommand is flat (`backprop train`, `backprop export`, …). The deviation is INTENTIONAL and must NOT be "fixed" back to flat (`backprop ollama-register`, `backprop ollama-list`, `backprop ollama-rm`) by a future maintainer.

Reasoning:

1. **Upstream parity (zero translation cost).** Upstream Ollama itself uses the same shape: `ollama create`, `ollama list`, `ollama rm`. An operator pasting `ollama list` from a tutorial, prefixed with `backprop`, gets `backprop ollama list` and it just works. Flattening would force the operator to mentally rewrite every Ollama tutorial.

2. **Mental-model grouping.** The three subcommands share a domain (the Ollama daemon, the model name allowlist, the "is the daemon running" error class). Grouping them under a single noun reflects the operator's actual mental model better than three flat verbs sprinkled at the top level.

3. **Doc-lie self-cure.** Wave 1 audit flagged CIDOCS-A-001 — the README hero example `backprop ollama register ./output/lora --name my-model` invented by the README author was a doc-lie at v1.3 (no such subcommand existed). Wave 6b's Feature pass made the doc real. Reverting to flat would re-break the README.

4. **`[[grep-all-instances-when-fixing-pattern]]` was honored.** No sibling nested-subparser pattern existed pre-Wave-6b — there was nothing to grep against. The deviation is documented in code (cli.py, above `cmd_ollama_register`) and here so future audits don't trip on it.

`SUBCOMMAND_TIERS` marks `ollama` as `experimental` so the shape can absorb iteration in v1.5 (e.g. `ollama pull` / `ollama show` mirroring upstream) without breaking the v1.4 contract.

### Path D escalations (Wave 6b BRIDGE)

None. argparse handles the nested subparser cleanly under Python 3.10+; the parser builds + parses correctly (verified via `python -m backpropagate --help`, `... ollama --help`, `... ollama register --help`, `... info --subcommand-tiers`, `... info --json`, `... validate --json`).

## CLOSED in Wave 6b features (2026-05-25 — BACKEND amend)

- **V1_4_BRIEF item 8 / BACKEND-F-008 — `mode="full"` for ≤3B model full fine-tuning** — CLOSED.
  - New `Trainer(mode="full")` kwarg with default `"lora"` (backward-compat). Plus `MultiRunConfig.mode` field with the same default; MultiRunTrainer threads `config.mode` into the inner Trainer constructor (multi_run.py:929-940).
  - **Construction-time gate (load-bearing).** Trainer.__init__ probes the model's parameter count via `_estimate_param_count_billions` (preset table primary; "Nb"-suffix regex fallback over the HF model_id) and raises `FullFinetuneModelTooLargeError(code="RUNTIME_FULL_FT_MODEL_TOO_LARGE")` when > 3B. Load-time recheck via `num_parameters()` belt-and-braces fires from `Trainer.load_model()` after the model is actually loaded.
  - **Helper extension, NOT fork (per advisor Wave 6b lock Q2).** `_build_sft_config` grew a `mode` parameter (default `"lora"`). When `mode="full"` the helper forces gradient_checkpointing=True (use_reentrant=False — HF-recommended default) and upgrades `adamw_8bit` → `paged_adamw_8bit` even on 24GB cards (which the LoRA detector would have left alone). The `peft_config=None` path is implicit — TRL `SFTConfig` does not configure PEFT when no adapter is on the model; mode='full' relies on the operator skipping the LoRA application step (the Trainer's existing `_load_with_unsloth` / `_load_with_transformers` paths attach LoRA unconditionally today, so full-FT operators get a LoRA adapter on top of the base model — **DOCUMENTED LIMITATION**, follow-up below).
  - **LR divisor.** When the operator does not explicitly pass `learning_rate=`, mode='full' applies a 10x lower LR (Biderman 2024 / Thinking Machines 2025); explicit operator-supplied values win.
  - **Error catalog.** New `RUNTIME_FULL_FT_MODEL_TOO_LARGE` entry in `exceptions.ERROR_CODES` (description + default_hint pointing at LoRA mode + smaller-preset alternatives + retryable=False). New `FullFinetuneModelTooLargeError` class in `exceptions.py` (subclass of `TrainingError`).
  - **Follow-up (mode='full' load path):** today the LoRA application sites at `_load_with_unsloth` / `_load_with_transformers` run unconditionally regardless of `self.mode`. A v1.5 follow-up should skip the LoRA application when `self.mode == "full"` so the trainer actually fine-tunes the base model weights (today mode='full' applies the SFTConfig contract — gradient_checkpointing + paged optim + lower LR — but the underlying model still carries a LoRA adapter). Tracked here so the v1.5 audit picks it up; the SFTConfig + gate are real, the load-path skip is the remaining piece.
  - Tests added in `tests/test_wave6b_features.py::TestTrainerModeFullGate` (11 cases) + `TestEstimateParamCount` (5 cases) + `TestMultiRunTrainerModeFull` (3 cases). Coordinates with BRIDGE (CLI flag `--mode=full`) and CI-DOCS (handbook page + error-codes.md entry + CHANGELOG).

- **BACKEND-F-002 — VRAM pre-flight estimator** — CLOSED. New `VRAMEstimate` dataclass + module-level `estimate_vram(model, *, mode, lora_r, batch_size, ...)` function in trainer.py. Returns a structured estimate (model_weights_gb / lora_adapter_gb / optimizer_state_gb / activations_gb / kv_cache_gb / overhead_gb + total_gb + `fits_on_card(vram_gb)` boolean helper + per-input reproducibility fields). Trainer instances expose `Trainer.estimate_vram(...)` as a convenience method that pulls the trainer's configured mode / lora_r / batch_size / gradient_accumulation / max_seq_length. Math is back-of-envelope (15% overhead margin): model weights × bytes_per_param (nf4=0.5 / bf16=2 / fp32=4); LoRA adapters × rank × (in+out) × layers × 7 modules × bytes_per_param; optimizer state via paged 8-bit Adam (2 buffers × 1 byte + gradient bytes); activations with sqrt(L) factor under mode='full' (gradient_checkpointing); KV cache amortized 0.25 for training. Coordinates with BRIDGE (CLI surface `backprop estimate-vram --model X --lora-r 256 --batch-size 1 --vram-gb 16`). Tests added in `tests/test_wave6b_features.py::TestEstimateVRAM` (10 cases).

- **BACKEND-F-003 — MultiRunTrainer callback parity** — CLOSED. New `_build_multi_run_step_callback(trainer, run_idx)` helper in multi_run.py mirrors `_build_abort_callback` shape — bridges HF TrainerCallback's `on_log` to MultiRunTrainer.on_step (forwards `(run_idx, step, loss)` to the user callable). Wired into `_execute_run`'s SFTTrainer construction alongside the existing abort callback; callbacks list is filtered to non-None entries so callers without on_step see byte-identical behavior. on_run_start (line 1334), on_run_complete (line 1115), and on_gpu_status (line 2596) were ALREADY wired pre-Wave-6b — pinned by tests now so a future refactor of `_execute_run` cannot silently remove the call sites. Tests added in `tests/test_wave6b_features.py::TestMultiRunTrainerCallbacks` (5 cases).

- **BACKEND-F-004 — CheckpointManager._save_manifest filelock parity** — CLOSED. Mirrors the Wave 6a BACKEND-F-012 RunHistoryManager pattern. New `CheckpointManager._locked_manifest_write(operation)` context manager wraps every `_save_manifest()` call site (register / prune / protect / unprotect / cleanup_orphaned / force_prune_to_size) + the external multi_run validation-loss-update site that mutates `cp.validation_loss` and re-saves. Uses cross-platform `filelock` with a 30s default timeout; `lock_timeout_seconds` constructor kwarg overrides; 0 means block-forever (same convention as RunHistoryManager). Degraded fallback when filelock unavailable proceeds without serialization (log at DEBUG). Tests added in `tests/test_wave6b_features.py::TestCheckpointManagerFilelock` (7 cases including a 2-manager concurrent-register integration test).

### Wave 6b backend doctrine ratchets

- **`[[extend-dont-fork]]` (NEW)** — `_build_sft_config` stays the SINGLE helper. mode='full' is an extension via the `mode` parameter, not a parallel `_build_full_ft_sft_config` helper. No Path D escalation was needed — the helper absorbed mode='full' cleanly via in-place dispatch on gradient_checkpointing + optim + (caller-supplied) LR.
- **`[[grep-all-instances-when-fixing-pattern]]` (applied)** — `_save_manifest()` call sites: 6 in-class + 1 external in multi_run.py. All 7 are now lock-wrapped. `_build_sft_config(mode=...)` call sites: 3 (Trainer.train first attempt + OOM retry + MultiRunTrainer._execute_run). All 3 thread the mode through.
- **`[[within-swarm-doc-lie-drift-detection]]`** — new `RUNTIME_FULL_FT_MODEL_TOO_LARGE` error code added to `ERROR_CODES` catalog so the drift gate's Class 4 (error-codes.md ↔ catalog) check sees it. CI-DOCS coordination point: handbook/error-codes.md must add the entry; coordinator will fail drift gate at PR-merge time otherwise.
- **`[[no-banner-documenting-no-op]]`** — every new symbol does real work: `Trainer.estimate_vram()` returns a real estimate; `mode="full"` rewires gradient_checkpointing + paged optim + LR; `_build_multi_run_step_callback` forwards real loss values; `CheckpointManager._locked_manifest_write` acquires a real filelock.

## CLOSED in Wave 6a foundation (2026-05-25 — backend amend)

- **BACKEND-A-003** — CLOSED. `MultiRunTrainer._execute_run` now delegates SFTConfig assembly to the shared module-level helper `_build_sft_config` in `trainer.py`. The helper applies the v1.3 BACKEND-5 paged-optim autodetection + BACKEND-7 Ada bf16/fp16 selection so multi-run picks up both contracts. Pre-Wave-6a, `multi_run.py` `_execute_run` hardcoded `optim=settings.training.optim` + `bf16=settings.training.bf16` raw — bypassing both detectors. Trainer.train() now also calls the shared helper, eliminating the drift surface. Coordinates cleanly with the BRIDGE-A-002 follow-up (which extended `_build_sft_config` with an optional `optim` kwarg defaulting to `None` for backward-compat — both call sites now thread the per-invocation override through too).
- **BACKEND-A-004** — CLOSED. Same `_execute_run` now applies `train_on_responses_only` Unsloth masking via the shared module-level helper `_apply_train_on_responses_only` in `trainer.py`. Pre-Wave-6a, multi-run users training on conversational data got full-conversation loss leakage onto the user prompt (the docstring at `multi_run.py:893` claimed `train_on_responses=True` but the code never invoked the masker). Both call sites now share a single application surface so future masking refactors stay in lockstep.
- **BACKEND-B-002** — CLOSED. The `multi_run.py` checkpoint save + manifest register branch is now gated on `self.config.save_every_run and not run_failed` (was `self.config.save_every_run` alone). Failed runs no longer flow through the save path, so a future resume cannot latch onto post-failure model state. Atomic-write contract preserved — only the gate condition changed.
- **BACKEND-A-001 follow-up (RUNTIME_GPU_OOM Option A)** — CLOSED. The `oom_recovery=False` branch in `trainer.py:Trainer.train()` now wraps the OOM (strict + adjacent matchers) into `GPUMemoryError(code='RUNTIME_GPU_OOM')` before re-raising. The multi-run symmetric writes `RUNTIME_GPU_OOM:` as a prefix on `RunResult.failure_reason` when `oom_recovery=False` and an OOM hits (multi-run contract is "record + continue," not "raise"). README + 7 translations + handbook docs + cli.py exit-code mapper now describe behavior that actually fires. TODO_WAVE_6A markers in `exceptions.py` have been retired (replaced with active-tense comments pointing at the raise site). The docstrings in `Trainer.__init__` + `MultiRunTrainer.__init__` now name `RUNTIME_GPU_OOM` as the actual surface (the recovery-exhausted path is still named `RUNTIME_OOM_RECOVERY_EXHAUSTED` since it raises a distinct code). Stale test comments at `tests/test_trainer.py` updated to match Option A semantics; the `test_oom_recovery_false_reraises_immediately` test now asserts `GPUMemoryError(code='RUNTIME_GPU_OOM')` instead of `TrainingError` to pin the new contract.

### Wave 6a refactor preservation notes (load-bearing)

- Auth middleware regression set NOT touched.
- OOM auto-recovery loop semantics preserved: the existing `oom_recovery=True` recovery loop in `trainer.py` halves batch + doubles accum unchanged; Option A wrap is a NEW branch on `oom_recovery=False` only.
- `RUNTIME_OOM_RECOVERY_EXHAUSTED` + `RUNTIME_OOM_ADJACENT` codes still emitted from the recovery-exhausted path; they are DISTINCT from `RUNTIME_GPU_OOM`.
- Atomic checkpoint writes (`checkpoints.py` contracts) untouched — only the gate condition for invoking save changed in `multi_run.py`.
- SLAO merge ordering at `multi_run.py` (the SLAO merge happens AFTER the OOM-retry loop break) left in place — BACKEND-B-005 scope-width deferral honored, no Wave 6a scope creep.
- F-002 resume semantics intact (`_maybe_resume` + `_restore_session_state`).
- TrainingCallback contract for single-run preserved (Trainer.train continues firing). MultiRunTrainer callback parity gap (BACKEND-F-003) is Wave 6b feature scope and was NOT touched.
- `MultiRunConfig` dataclass fields enumeration unchanged for the refactor itself (the BRIDGE-A-002 follow-up added 5 new fields, which is a separate closure tracked below).

### Wave 2 split-state collapsed forward (2026-05-25)

The Wave 2 split-state (library-internal docstrings naming RUNTIME_TRAINING_FAILED while user-facing surface promises RUNTIME_GPU_OOM) is now collapsed: the docstrings now name `RUNTIME_GPU_OOM` (matching the user-facing surface) and the runtime actually produces the code. The Wave 2 "docs honest about current runtime / user-facing surface promises committed-future-state" split has been resolved by the runtime catching up to both surfaces.

## CLOSED in Wave 6a foundation (2026-05-25 — CI / docs amend)

CI / docs items closed in this wave (commit references attach at PR-merge time;
the items remain in this file as the historical record of v1.4 Wave 6a scope):

- **V1_4_BRIEF item 2 / CIDOCS-F-006 — verify.sh strict gate flip**: ✓ closed.
  `.github/workflows/ci.yml` `verify-smoke` job dropped `continue-on-error: true`,
  soft-gate fallback replaced with strict `jq` parse. After 5+ green runs (Wave
  2 / Wave 3.5 / Stage C / Wave 5.5 / Wave 6a.0) the JSON shape is stable enough
  to gate on. Step name renamed `verify.sh smoke (advisory)` → `verify.sh smoke`
  to match the new behavior.
- **V1_4_BRIEF item 4 — Bandit gate flag-vs-comment doc-lie**: ✓ closed.
  ci.yml Bandit step name + comment rewritten LOW/LOW (matches `-l -i`
  semantics). Flag bump to MEDIUM/MEDIUM (`-ll -ii`) deferred to v1.5 as
  a separate semantic shift — bumping mid-swarm is a real behavior change
  vs. a copy fix. The drift gate's new Class 5 scanner now enforces this
  pattern mechanically.
- **V1_4_BRIEF item 5 — Drift gate 5-class extension** (the load-bearing one):
  ✓ closed. `scripts/check_doc_drift.py` extended from 4 checks to 9. New
  classes:
  - Class 1: argparse `default=N` vs handbook `default: N` (`check_argparse_default_drift`)
  - Class 2: env var names in llms.txt vs runtime reads (`check_llms_txt_env_vars`)
  - Class 3: env var DEFAULT values in env-vars.md vs `config.py` BaseSettings (`check_env_var_default_drift`)
  - Class 4: error-codes.md "Fix" column cross-references (`check_error_codes_doc_refs`)
  - Class 5: CI workflow step name/comment severity claims vs flag semantics (`check_workflow_severity_drift`)

  Each class has its own allow-list key in `scripts/doc-drift-allow.toml`.
  21 new unit tests live in `tests/test_check_doc_drift.py`. Real drifts
  surfaced and fixed during the verify-pause:

  - `BACKPROPAGATE_DATA__PACKING` default in env-vars.md (was `false`,
    runtime is `true` per v1.3 BACKEND-4 — fixed in env-vars.md).
  - `--max-seq-length` in error-codes.md RUNTIME_GPU_OOM Fix column (was
    CLI-flag-styled but is actually the `Trainer(max_seq_length=...)`
    Python API knob + `BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH` env var —
    rewrite uses the real surfaces).

  One legitimate allow-list entry added for `--host` (argparse `default=None`
  falls through to `config.ui.host`; the handbook documents the runtime-
  resolved value, not the argparse literal). The runtime scanner
  (`_find_runtime_env_vars`) preserves the v1.3 narrow regex baseline so
  the existing Check 1 stays green; two new helpers
  (`_find_indirect_module_const_env_vars` + `_find_pydantic_bound_env_vars`)
  feed Class 2 / Class 4 the broader signal needed to recognise pydantic-
  settings-bound + module-level-constant env-var reads as legitimate.
- **V1_4_BRIEF item 6 / CIDOCS-F-025 — `scripts/prep_release.sh`**: ✓ closed.
  8-stage pre-release coordinator: pyproject version extract → CITATION.cff
  bump → references validation → polyglot-mcp translation → build → twine +
  PKG-INFO smoke → drift gate → verify.sh. `--dry-run` flag for safe local
  exercise. CITATION.cff comment updated to reference the script's new role
  (was "planned, v1.4"; now "landed v1.4 Wave 6a"). Dry-run smoke run during
  the amend reported all 8 stages exercising correctly against the current
  branch state.
- **CIDOCS-F-006**: ✓ closed — same as V1_4_BRIEF item 2 above.
- **CIDOCS-F-007 — PyPI metadata end-to-end smoke**: ✓ closed. Added
  "PKG-INFO smoke" step to ci.yml `build` job extracting PKG-INFO from
  sdist and grepping for `License-Expression` / `License` / `Author` /
  `Name` / `Version`. Will catch the next hatchling-style metadata-build
  regression before the wheel uploads to PyPI. Same logic lives in
  `scripts/preflight.sh` (Stage 6) and `scripts/prep_release.sh` (Stage 6)
  so the local + CI + release surfaces all use the same shape.
- **CIDOCS-F-010 — pre-commit hook for `check_doc_drift.py`**: ✓ closed.
  Local repo hook added to `.pre-commit-config.yaml`, scoped to
  `stages: [pre-push]` (the 5-class scanner takes ~100ms; comfortable on
  push, would feel slow on every commit). CONTRIBUTING.md "Local dev
  loop" section updated to name the hook + the 5-class scope.
- **CIDOCS-F-022 — mutmut first baseline trigger**: ✓ closed via
  documentation. `.github/workflows/mutmut.yml` header block now names the
  manual `gh workflow run mutmut.yml` trigger required after this wave
  merges; the workflow's own "Update baseline (if changed)" step
  auto-opens the baseline-bootstrap PR for human review (no silent
  auto-ratchet of the bar).
- **CIDOCS-F-023 — uv.lock CI consumption migration (pilot)**: ✓ closed
  as a soft-gated pilot on Linux 3.11. New `uv-sync-pilot` job in ci.yml
  runs `uv sync --frozen --extra dev --extra full` + `uv run pytest` so
  a future cross-OS matrix migration has a green baseline to point at.
  Strict-gate + matrix migration deferred to v1.5 per the
  baseline-before-enforce doctrine.
- **CIDOCS-F-024 — shipcheck-style preflight `scripts/preflight.sh`**:
  ✓ closed. Local mirror of every CI gate (drift / lint / type / tests /
  build / twine + PKG-INFO smoke / Bandit / CITATION.cff-pyproject
  version sync / package.json-pyproject version sync). `--quick` flag for
  the fast inner loop (skips mypy + Bandit). Read-only — safe to run on
  any working tree.
- **CIDOCS-F-025 — `prep_release.sh` scope expansion**: ✓ subsumed into
  V1_4_BRIEF item 6 above. Single deliverable.

### Drift gate extension — coordinator verify-pause result

After landing the 5-class scanner, the new gate was run against the
`wave-6a/foundation` branch state at amend time and reported **0
findings**. Two real drifts were caught during the extension build and
fixed in the same commit (`env-vars.md` `BACKPROPAGATE_DATA__PACKING`
default + error-codes.md `--max-seq-length` reference). One legitimate
allow-list entry was added for `--host` (argparse `default=None` falls
through to `config.ui.host`; the handbook documents the runtime-resolved
value, not the argparse literal).

The coordinator re-runs the gate at the wave-merge gate; the local +
post-amend states should both report 0 findings.

### CI / docs amend coordination notes — captured at amend time

- **BACKEND agent's RUNTIME_GPU_OOM Option A wrap**: error-codes.md /
  troubleshooting.md / troubleshooting-cuda.md text on `RUNTIME_GPU_OOM`
  preserved as-is — the user-facing surface promised the future state
  and the BACKEND amend collapses the split-state forward. Spot-check:
  no doc-lie remains (the recovery advice now points at a real raised
  code path). The error-codes.md `--max-seq-length` rewrite was done as
  part of the drift-gate Class 4 close, not as a RUNTIME_GPU_OOM-text
  fix — the recovery advice now names the correct Python API knob plus
  the matching env var.
- **FRONTEND agent's ui_security rename + TRAINING_PRESETS rename**: the
  v1.4 deprecation cycle (DeprecationWarning) is named in CHANGELOG when
  the FRONTEND amend lands; v1.5 = UserWarning, v1.6 = removal per
  advisor lock 2026-05-25 Q4. The migrations.md surface owned by the
  FRONTEND agent (per V1_4_BRIEF item 7 partial-closure note above);
  this agent did not touch it.
- **BRIDGE-A-002 follow-up (5 new Trainer kwargs)**: cli-reference.md
  already lists `--use-dora`, `--no-packing`, `--init-lora-weights`,
  `--optim`, `--lora-preset` in both `backprop train` and `backprop
  multi-run` tables (Wave 3.5 landing). Drift gate Class 1 ran clean
  against these — argparse defaults and handbook strings match.

## From Wave 2 bridge (BRIDGE-A-002 follow-up — symbolic threading observation)
- ~~**BRIDGE-A-002 follow-up**~~ **CLOSED Wave 6a foundation (2026-05-25):** the five Wave 6b knobs are now explicit constructor parameters on `Trainer.__init__` (trainer.py, after the `response_markers` kwarg block) and explicit dataclass fields on `MultiRunConfig` (multi_run.py, after `max_pause_seconds`). Each defaults to `None` with `settings.lora.*` / `settings.data.*` / `settings.training.*` fallback (`lora_preset` defaults to `"quality"` since there is no `settings.lora.preset` field — the preset is a future-overlay slot). The CLI introspection filter at `cmd_train` (cli.py:644) / `cmd_multi_run` (cli.py:877) / `cmd_replay` (cli.py:4275) now passes all five keys through without dropping any. End-to-end consumers updated:
  - `_build_sft_config` (trainer.py:644) grew an optional `optim` parameter that defaults to `settings.training.optim` for back-compat; both call sites in trainer.py (`Trainer.train()` first attempt + OOM retry) and the multi_run.py call site (`MultiRunTrainer._execute_run`) thread `self.packing` / `self.optim` from the inner Trainer instance.
  - `_load_with_unsloth` / `_load_with_transformers` (trainer.py LoRA application sites) now read `self.use_dora` / `self.init_lora_weights` instead of `settings.lora.*` directly so per-invocation override flows into PEFT.
  - `MultiRunTrainer.run()` (multi_run.py inner Trainer construction) forwards `self.config.use_dora` / `self.config.packing` / `self.config.init_lora_weights` / `self.config.lora_preset` / `self.config.optim` to `Trainer(...)`.
  - `optim="auto"` sentinel from the CLI is mapped to the settings fallback in `Trainer.__init__` so `_detect_optim_for_card` sees the actual configured default (`adamw_8bit`) instead of the literal `"auto"` string.
  - Tests added in `tests/test_trainer.py::TestTrainerWave6bKwargs` (11 cases: explicit-set / settings-fallback for each of the 5 kwargs + introspection signature assertion + end-to-end `cmd_train` capture) and `tests/test_multi_run.py::TestMultiRunConfigWave6bFields` + `TestMultiRunTrainer::test_wave6b_kwargs_forwarded_to_inner_trainer`.
  - Contract preservation: callers who don't pass these kwargs (and operators relying on `BACKPROPAGATE_LORA__USE_DORA=true` etc.) see byte-identical pre-fix behavior since `None`-default + `is not None` fallback to settings preserves the env-var path. The wave6b kwargs compose cleanly with BACKEND-A-003/A-004 multi-run refactor: the shared `_build_sft_config` reads from instance attributes that the constructor pre-resolved.

## From v1.4 Wave 2 frontend (FRONTEND-A-003 follow-ups + symbol-rename gating)
- **FRONTEND-A-003 follow-up: report-only flip** — `security_headers_middleware` defaults to enforce mode (`Content-Security-Policy`, not `Content-Security-Policy-Report-Only`). A future wave should add a `BACKPROPAGATE_UI_CSP_REPORT_ONLY=1` env knob so operators can flip the chain into report-only during a deployment shakeout without code edits. The plumbing is already there (`get_reflex_csp(report_only=True)` returns a policy that emits the report-only header name); just needs the middleware to read the env var on each request.
- **FRONTEND-A-003 follow-up: nonce-based scripts** — `DEFAULT_REFLEX_CSP` currently includes `'unsafe-inline'` in `script_src` to permit Next.js's `__NEXT_DATA__` hydration block + Reflex's per-page Var-binding inline scripts. The stricter shape is a per-request nonce that the Next.js page template inserts into every legitimate inline `<script>` tag. Reflex doesn't expose a nonce hook today; v1.5+ should track upstream Reflex's CSP-nonce work and migrate when the API stabilises. Until then `'unsafe-inline'` is the documented trade-off and the operator-facing risk is bounded by the auth gate (the inline scripts are server-emitted, never user-supplied).
- **FRONTEND-A-003 follow-up: TESTS coordination** — the new `security_headers_middleware` needs isolated coverage. The v1.4 TESTS agent should add `tests/test_security_headers_middleware.py` mirroring the v1.3 `test_rate_limit_middleware.py` / `test_request_logging_middleware.py` shape: (a) headers appear on a vanilla 200 response, (b) headers appear on a 401 auth-rejected response (proves the AFTER-auth wiring), (c) `_is_security_header` drops upstream duplicates so the chain output is deterministic, (d) WS upgrade passes through unmodified, (e) lifespan messages pass through unmodified, (f) headers ARE present even after auth's pre-accept WS close (or document why they're not — WS frames don't carry HTTP headers). The five wired layers (`security_headers, request_logging, basic_auth, rate_limit, healthz`) need a chain-integration smoke test too so future reorderings get caught.
- **From V1_4_BRIEF item 7 (ui_security symbol rename) — PARTIAL CLOSURE (Wave 6a foundation, FRONTEND amend)** — the immediate `gradio_`-prefix rename of the public UI-error helpers landed in Wave 6a foundation. New canonical names: `safe_ui_handler`, `raise_ui_error`, `raise_ui_warning`, `raise_ui_info`, `RequestContext.from_request`. Legacy names continue to resolve via module-level `__getattr__` (PEP 562) + classmethod shim on `RequestContext` and emit `DeprecationWarning` per the 3-version cycle locked at advisor 2026-05-25 Q4 (v1.4 → DeprecationWarning, v1.5 → UserWarning, v1.6 → removal). Handbook updated at `site/src/content/docs/handbook/migrations.md` (new v1.3 → v1.4 section). Tests added at `tests/test_ui_security_legacy_aliases.py`. Originally Wave 6a was scoped to also drop `DEFAULT_GRADIO_CSP` + `get_gradio_csp` and fold the rest of the Gradio-era legacy (`SessionManager`, `ConcurrencyLimiter`, `EnhancedRateLimiter`, `safe_gradio_handler`, `RequestContext`, `JWTConfig`/`JWTManager`, `CSRFToken`/`CSRFProtection`, `SecureSessionHandler`). Those continue to be deferred:

  **STILL DEFERRED (later wave / v1.5 candidate):**
  1. `_GradioShim` deletion — the class is load-bearing today because the rest of `ui_security.py` references `gr.Request` type hints + `gr.Error(...)` call sites. Removing it would cascade into ~20 type-hint changes + a refactor of the error-raising helpers to not depend on a `gr` object. Deferred so this rename stays surgical.
     - **Wave 6b features Path D escalation (2026-05-25)** — FRONTEND amend agent surveyed the cascade per the advisor lock criteria and re-confirmed the deferral. Final tally: **44 sites** (18 `gr.Request` type hints + 24 `gr.Error` raises/handlers + 1 `gr.Warning` + 1 `gr.Info` + the class def + the `gr = _GradioShim()` assignment). Threshold for mechanical replacement is ≤30 sites per the advisor lock — 44 clearly exceeds. Beyond the count, **two sites require substantive logic refactoring, not type-hint swaps**:
        - `safe_ui_handler` decorator (lines ~1062-1124) contains `except gr.Error:` (line 1079) which is load-bearing for real-Gradio consumers. The v1.4 → v1.6 deprecation cycle preserves `safe_gradio_handler` as a working alias until v1.6 — removing the `except gr.Error:` block changes runtime behavior for any operator who pip-installs gradio alongside backpropagate. The replacement needs a structural-exception (e.g., a new `BackpropagateUIError` base) that BOTH the shim's `Error` subclass and real Gradio's `gr.Error` inherit from, OR a `GRADIO_AVAILABLE`-gated branch at every handler site. Neither is mechanical.
        - 18 `gr.Request | None` type hints across the rate limiter (`EnhancedRateLimiter`), session manager, concurrency limiter, JWT manager, CSRF protection, request-context dataclass (`RequestContext.from_request` / `from_legacy_request`), and request-validation helpers. The canonical Reflex replacement isn't a single-type rewrite — Reflex middleware uses `starlette.requests.Request`, but the state-handler shape is different (the request hits middleware, not the state). Mapping `gr.Request | None` to either is substantive — it changes the contract for any consumer who reaches into `request.client` / `request.headers` and assumes Gradio's surface.
     - **Recommended v1.5 approach (locked candidate from FRONTEND-A-009 audit):** split `ui_security.py` into `ui_security_core.py` (framework-agnostic — `get_ui_output_dir`, `sanitize_filename`, `safe_markdown_fence`, `AuthBadgeContext`, `sanitize_error_for_user`, `validate_auth_shape`, the CSP factories, `apply_security_headers`) + `ui_security_legacy.py` (Gradio-era — `EnhancedRateLimiter`, `SessionManager`, `ConcurrencyLimiter`, `JWTConfig`/`JWTManager`, `CSRFToken`/`CSRFProtection`, `SecureSessionHandler`, `safe_ui_handler`, `raise_ui_error`/warning/info, the `RequestContext` dataclass + `_GradioShim` itself). The legacy module imports `gradio as gr` (or `_GradioShim`); the core module never touches Gradio. Module-level `__getattr__` on `ui_security` re-exports both for backward compatibility through the v1.6 removal. This is the surgical fix — it lets v1.6 delete `ui_security_legacy.py` wholesale rather than walking 44 call sites in the live module.
     - **Shim docstring updated (per [[no-banner-documenting-no-op]])** — the `_GradioShim` class body now names the Wave 6b survey, the 44-site count, the substantive-logic concerns, the v1.5 split-or-fold candidate, and the WAVE_6A_TODO.md back-reference. It's transitional, not dead code; the comment makes that explicit.
  2. The broader "fold the Gradio-era legacy modules into the new middleware shape OR split `ui_security.py` into `_legacy.py` + `_core.py`" decision. The Wave 6a rename only touched the public-symbol surface, not the file structure. Per the FRONTEND-A-009 audit, either fold these into the new middleware shape or split `ui_security.py` into `ui_security_legacy.py` (preserved for back-compat / external callers / tests) and `ui_security_core.py` (the `get_ui_output_dir` + `sanitize_filename` + `safe_markdown_fence` + `AuthBadgeContext` helpers the Reflex UI actually uses).
  3. `DEFAULT_GRADIO_CSP` + `get_gradio_csp` *removal* — Wave 6a kept them in place (they already had the in-place `DeprecationWarning` shape from Wave 2 FRONTEND-A-003 + Stage C FRONTEND-B-007; deletion happens in v1.6 alongside the rest of the legacy shim).
  4. None of the Gradio-era classes (`SessionManager`, `ConcurrencyLimiter`, `EnhancedRateLimiter`, `JWTConfig`/`JWTManager`, `CSRFToken`/`CSRFProtection`, `SecureSessionHandler`) are wired into the Reflex middleware (the new middleware does its own rate-limit via `_SlidingWindow` rather than `EnhancedRateLimiter`); their fate is decided in the Wave 6b / v1.5 split-or-fold decision above.

## From v1.4 Wave 6b frontend (FRONTEND-F-005 MultiRunState producer contract)

- **FRONTEND-F-005 follow-up: MultiRunState trajectory populator** — `backpropagate/ui_app/pages/multi_run.py` (the inline-trajectory cell at line ~266) now binds to `run["trajectory"]` instead of a hardcoded `'—'`. The fix moves the literal from the view to a producer-side contract: **every dict in `MultiRunState.runs` MUST carry a `trajectory` key** (empty string when no data yet, or a short ASCII sparkline like `"▁▂▃▅▆▇"` once the inner Trainer's loss log is wired). Today `MultiRunState.start_multi_run` is a stub that doesn't populate `runs`, so the foreach body never executes and the contract is enforced by documentation alone. When the Phase-3 Trainer hookup lands and starts emitting per-run rows, the populator must include `trajectory` — either the live ASCII sparkline (preferred) or an explicit empty string. A unit test in `tests/test_ui_states.py::TestMultiRunState` that asserts every emitted dict has the key would lock this down.

## From v1.4 Wave 2 tests (TESTS-A-006: UI lora-rank default vs CLI default)
- **TESTS-A-006 follow-up** — `backpropagate/ui_state.py:314` (TrainState.lora_r) and `:508` (MultiRunState.lora_r) both default to `lora_r=16` while the CLI argparse default is `256` (v1.3 BACKEND-1 quality preset per Biderman 2024 + Thinking Machines 2025). The UI surface intentionally lags the CLI; tests at `tests/test_ui_states.py:84` and `:344` carry explanatory comments documenting the divergence. v1.5 candidate (Wave 5 feature audit): decide whether to align `TrainState.lora_r` / `MultiRunState.lora_r` to `256` and `.lora_alpha` to `512` to match the CLI quality preset. If yes, the change is `ui_state.py:314` + `:508` followed by flipping the test assertions at `test_ui_states.py:84-85` + `:344`. If no, the test comments stand as the canonical documentation of the intentional divergence.

## From v1.4 Wave 2 tests (TESTS-A-007: serial-marker audit follow-ups)
- **TESTS-A-007 next batch** — v1.4 Wave 2 applied `@pytest.mark.serial` to the load-bearing process-global-mutation sites surfaced by the Wave 1 audit: `TestRunIdCorrelation` + `TestTrainingLoggerCaplog` (both call `configure_logging(...force=True)`), `TestE2EFullChain.test_train_export_register_chain_emits_expected_log_events` (calls `structlog.configure(...)`), and `TestSecurityLogger` + `TestSessionManager` (singleton `._instance` reset via autouse fixtures). A `pytest_collection_modifyitems` hook in `tests/conftest.py` pins every serial-marked test into the same `xdist_group="serial"` so they share a worker. Next-batch audit candidates (not yet marked):
  - any test in `tests/test_config.py` that calls `Settings().apply_windows_fixes()` and reads `os.environ["TOKENIZERS_PARALLELISM"]` post-call without monkeypatch isolation
  - any test in `tests/test_logging_config.py` that calls `configure_logging(force=True)` (not yet audited file-by-file)
  - any test that imports `backpropagate.ui_app.middleware.rate_limit` and uses its `_SlidingWindow` module-level singletons without the `_reset_for_tests` hook (`tests/test_rate_limit_middleware.py` was added in v1.4 Wave 2 and uses an autouse `_reset_rate_limit_state` fixture; future rate-limit tests should follow the same pattern OR add `@pytest.mark.serial`)
- **TESTS-A-007 marker registration** — the `serial` marker needs to be registered in `pyproject.toml` `[tool.pytest.ini_options].markers` to keep `--strict-markers` honest. v1.4 Wave 2: deferred to the CI-DOCS agent (coordination — CI-DOCS owns pyproject.toml; TESTS agent's edits would conflict). Reminder for the post-Wave-2 verification pass: add `"serial: marks tests that mutate process-global state and must run sequentially under pytest-xdist (see tests/conftest.py pytest_collection_modifyitems)"` to the markers list. The marker is in use today via `@pytest.mark.serial` on `TestRunIdCorrelation` / `TestTrainingLoggerCaplog` / etc — `--strict-markers` will raise `PytestUnknownMarkWarning` if the registration is missed.

## From v1.4 Wave 2 tests (BRIDGE-A-004 coverage handoff)
- **TESTS-A-003 / BRIDGE-A-004 coverage** — if BRIDGE-A-004 added a `--hub-token-file` flag (per the dispatch's coordination note), it has no test coverage in v1.4 Wave 2. v1.4 Wave 5 feature audit candidate: add a TestCmdReplay-shaped handler test for the new flag's happy path + the missing-file negative path.

## From v1.4 Wave 2 tests (TESTS-A-008..023 — Wave 1 MEDIUM/LOW absorption)
- The Wave 1 TESTS audit surfaced 16 MEDIUM/LOW findings (TESTS-A-008 through TESTS-A-023) covering: stale Gradio-era CSP tests (`test_security_advanced.py:765`), zero-assertion tests in `test_multi_run.py:752` + `test_main_entry.py:50` + `test_e2e_chain.py:286`, cold-start race in `test_callback_integration.py:312` (100ms ceiling), 12 sleep-based assertions in `test_callback_integration.py` that should use the deterministic-Event pattern from `test_gpu_monitor_to_multirun_integration`, paged-Adam optimizer end-to-end coverage gap, auth-badge state reactivity coverage gap, ERROR_CODES catalog content-completeness gap, and the doc-pointer skip tests in `test_auth_middleware.py:463/473/527`. Per the v1.4 advisor's MEDIUM/LOW absorption decision, these are not Wave 2 scope and are tracked here as the v1.5 (or later) test-quality sweep batch.

## From v1.4 Wave 3 Stage B tests (TESTS-B-016 — TestConfigureLogging serial-marker)

Stage C amend (Wave 3.5) declined to flip the `@pytest.mark.serial` flag on these classes mid-wave because (a) the change touches test execution semantics under xdist (additive defense, but still a behavior change), and (b) WAVE_6A_TODO TESTS-A-007 next batch already anchors this scope. Concrete targets (named here so Wave 6a doesn't re-audit):

- `tests/test_logging_config.py::TestConfigureLogging` — 9 `configure_logging(force=True)` calls; conftest.py identifies `force=True` as a process-global mutation per the serial-marker contract.
- `tests/test_logging_config.py::TestGetLogger` — `get_logger` auto-configures logging when `_configured=False`; ditto the process-global mutation.
- `tests/test_logging_config.py::TestLogContext` and `tests/test_logging_config.py::TestStructlogAvailable` — same audit shape.

Wave 6a action: add `@pytest.mark.serial` to all four classes (mirrors the Wave 2 pattern on `TestRunIdCorrelation` + `TestTrainingLoggerCaplog`), then flip the `BACKPROPAGATE_PYTEST_PARALLEL` env knob on in CI. Without the marker, parallel mode produces flaky 'level mismatch' failures the moment xdist schedules two of these classes on different workers.

### CLOSURE — Wave 6a TESTS-A-007 next batch (2026-05-25)

The TESTS agent's Wave 6a marker audit closed this scope. Applied `@pytest.mark.serial` to 6 classes in `tests/test_logging_config.py`:

- **TestConfigureLogging** (line 30) — 9× `configure_logging(force=True)` calls; explicit brief target.
- **TestGetLogger** (line 109) — `setup_method` resets `lc._configured = False` + tests call `get_logger()` which auto-configures; explicit brief target.
- **TestGetStandardLogger** (line 177) — same audit shape as TestGetLogger; surfaced via the grep-doctrine sibling pass on `lc._configured = False` (per `[[grep-all-instances-when-fixing-pattern]]`).
- **TestLogContext** (line 214) — `LogContext.__enter__/__exit__` bind/unbind `structlog.contextvars` (process-context-scope within a worker); explicit brief target.
- **TestTrainingLogger** (line 312) — `setup_method` calls `configure_logging(level="DEBUG", force=True)`; surfaced via the grep-doctrine sibling pass on `configure_logging(force=True)`.
- **TestFallbackLogging** (line 589) — clears `logging.getLogger().handlers` (stdlib root logger, process-global) and reconfigures via `_configure_standard_logging`; surfaced via the grep-doctrine sibling pass on root-logger-handler mutations.

**TestStructlogAvailable EXEMPT (documented):** the two tests in this class only READ `STRUCTLOG_AVAILABLE` — no `configure_logging`, no `structlog.configure`, no `lc._configured` reset, no root-handler swap. Per `[[no-banner-documenting-no-op]]`, applying the serial marker to a pure-read class would be a no-op marker. Exemption rationale is documented in the class docstring at line 283-297.

`import pytest` added to `tests/test_logging_config.py` line 17 (was missing — module previously had no marker decorators).

**Per-class brief delta:** brief named 4 classes (TestConfigureLogging, TestGetLogger, TestLogContext, TestStructlogAvailable). Audit applied marker to 6 (added TestGetStandardLogger + TestTrainingLogger + TestFallbackLogging per grep-doctrine), exempted 1 (TestStructlogAvailable per `[[no-banner-documenting-no-op]]`). Net: 6 markers applied, 1 documented exemption.

**No CI flip:** per advisor lock ("defer parallel-flip-on to v1.5"), `BACKPROPAGATE_PYTEST_PARALLEL=1` default-on remains a v1.5 task with its own bed-in patch release. Wave 6a is markers-only.

**No Path D escalations:** the audit did not surface silent-green twins / removed-env-var references / permanent `pytest.skip` markers in the four named classes or their grep-doctrine siblings.

### Cross-file audit results (Wave 6a, 2026-05-25)

- `tests/test_config.py` `apply_windows_fixes` tests (lines 398, 423, 438, 989, 1003) — ALL use `monkeypatch.delenv` for `TOKENIZERS_PARALLELISM` / `XFORMERS_DISABLED` / `CUDA_LAUNCH_BLOCKING`. The brief's audit candidate ("without monkeypatch isolation") returns empty — properly isolated tests, no marker needed.
- `tests/test_rate_limit_middleware.py` — already uses the `_reset_rate_limit_state` autouse fixture (lines 170-183) that calls `_rl._reset_for_tests()` pre + post each test. No serial marker needed; the autouse fixture pattern is the documented alternative per the brief.
- Module-wide grep for `_instance = None` / `sys.modules[`: only existing already-marked classes (`TestSecurityLogger` + `TestSessionManager` in test_ui_security.py) and helper-fixture-level mutations in `tests/helpers/asgi.py` lines 166, 197 (the `del sys.modules[mod_name]` reload-loops in `reflex_app_with_auth_enforced` / `reflex_app_no_auth`).

### v1.5 sweep candidate (deferred from Wave 6a)

Tests that consume the `tests/helpers/asgi.py:reflex_app_with_auth_enforced` / `:reflex_app_no_auth` fixtures perform `del sys.modules[mod_name]` + `importlib.reload(app_module)` on the `backpropagate.ui_app.app` module. The fixtures use `monkeypatch.setattr` for the `ENFORCEMENT_AVAILABLE` flag (auto-restored) and `monkeypatch.setenv` for `BACKPROPAGATE_UI_AUTH` (auto-restored), so the live mutation is the `sys.modules` purge — a process-global. Consumers: `tests/test_auth_middleware.py` + `tests/test_auth_middleware_fuzz.py`.

Audit shape — these tests run import-level Reflex app construction, which is heavy enough that adjacent tests on the same xdist worker would already serialise via the per-test reload anyway. The leak risk is across xdist workers (each worker maintains its own `sys.modules`, so cross-worker leakage is bounded), and within-worker leakage is bounded by the fixture's pre-test cleanup loop. Not Wave 6a scope; tagged here for the v1.5 broader serial-marker audit per the brief's "v1.5 sweep" guidance.

## From v1.4 Wave 5.5 (within-swarm doc-lie sweep — extend drift gate to source comments)

Wave 5.5 surgical-fix pass corrected three source-comment doc-lies that all promised "v1.4" features that did NOT land in v1.4:

1. **`backpropagate/ui_state.py:1552-1555`** (FRONTEND-F-013) — RunDetailState.loss_history comment claimed `lr + grad_norm series are tracked for v1.4 multi-line metrics view + landed at that time`. Neither shipped: trainer.py:2033-2039 still extracts only `loss` from HF Trainer's `state.log_history`; checkpoints.py schema has no parallel fields; RunDetailState declares only `loss_history`. Fixed: comment now names v1.5 deferral + WHY (data pipeline dead from trainer outward).
2. **`backpropagate/ui_app/chrome.py:241-246`** (FRONTEND-F-015 echo) — BpLeftNav docstring promised the Settings link returns when `/settings` ships, "currently scoped to a v1.4 follow-up". The route was NOT added in v1.4. Fixed: comment now names v1.5 candidate scope (theme toggle + default-output-dir + default-quantization + default-lora-preset + env-var inspector + token rotation per FRONTEND-F-007).
3. **`backpropagate/ui_app/middleware/request_logging.py:99-105`** — promised "a v1.4 enhancement would have auth populate a scope key (e.g. `scope['_bp_auth_user']`) that this middleware then reads". The scope-key was NOT added in v1.4 (grep confirms zero `_bp_auth_user` writers in auth.py). Fixed: comment now names v1.5 candidate + notes that the audit logger in auth.py still records `auth_user` so identity is not lost, just absent from the request-logging line.

**Structural pattern observation (Path D escalation):** Three comments in the UI tree all silently rolled "v1.4" promises forward without comment updates. This is the canonical pattern the existing drift gate at the README↔CHANGELOG↔CLI surface does NOT cover — the drift gate is wired to public-surface docs (README, handbook, CLI help text, llms.txt) and does not scan source-comments for vN.N / Wave X promise tokens.

**v1.5 candidate: extend drift gate to source comments.** Concrete shape: a `scripts/check-comment-drift.py` (or similar) that scans `backpropagate/**/*.py` + `site/src/**/*.{ts,astro,md}` for patterns like `v1\.\d+\s+(will|adds|adding|enhancement|follow-up)`, `tracked for v1\.\d+`, `landed at v1\.\d+`, `v1\.\d+\s+candidate`, and cross-checks against the current `pyproject.toml` version. When the project version meets-or-exceeds the version mentioned in the comment, the gate fires and asks: did the promised feature ship? If not, the comment is a doc-lie. CI integration: add as a new ratchet next to the existing drift gate; falsifies at PR-time rather than at Wave-N feature-audit-time.

**Why this matters:** Wave 5 feature audit caught FRONTEND-F-013 only because the auditor was reading source comments alongside the feature inventory. A mechanical gate would have caught all three in v1.4 Wave 1 (Stage A) instead of v1.4 Wave 5. The cost of slipping these into v1.5 audit is real — each rolled promise is operator-misleading documentation that compounds at every cold-read.

## From v1.4 Wave 5 (Decision 3 — TRAINING_PRESETS → MULTI_RUN_PRESETS namespace disambiguation)

- **CLOSED in Wave 6a foundation (FRONTEND amend).** Wave 5 feature audit surfaced an operator-trap class: `TRAINING_PRESETS` (v1.0-era multi-run loop hyperparameter table — `num_runs`, `samples_per_run`, `replay_fraction`) and `LORA_PRESETS` (v1.3 BACKEND-1 LoRA-architecture shape table — `r`, `target_modules`, `lr_multiplier`) BOTH used the keys `"fast"` + `"quality"` with semantically different values. The reference handbook page carried a load-bearing disambiguator paragraph for this reason. Advisor lock 2026-05-25 Decision 3: rename `TRAINING_PRESETS` → `MULTI_RUN_PRESETS` (keep `LORA_PRESETS` untouched since it's the source of the user-facing `--lora-preset` flag values).

  **What landed:**
  - `backpropagate/config.py`: renamed the dict to `MULTI_RUN_PRESETS` (same content; canonical name); added `MULTI_RUN_PRESETS` to `__all__` alongside the legacy `TRAINING_PRESETS` entry; migrated internal `get_preset` callers to the new name; added a module-level `__getattr__` (PEP 562) that resolves the legacy `TRAINING_PRESETS` name to the canonical dict + emits a `DeprecationWarning` whose message explicitly mentions `LORA_PRESETS` (the OTHER preset table the operator might have actually wanted).
  - `backpropagate/__init__.py`: imports the canonical `MULTI_RUN_PRESETS`; keeps `TRAINING_PRESETS` as a silent back-compat alias at the package level (no warning at `from backpropagate import TRAINING_PRESETS`).
  - `site/src/content/docs/handbook/reference.md`: disambiguator note updated to name the new canonical `MULTI_RUN_PRESETS` + name the legacy `TRAINING_PRESETS` deprecation + name the 3-version cycle.
  - `site/src/content/docs/handbook/migrations.md`: new v1.3 → v1.4 section covers both Wave 6a renames (this one + the `ui_security` Gradio→UI rename).
  - Tests: `tests/test_config_legacy_aliases.py`.
  - Deprecation cycle (locked advisor 2026-05-25 Q4): v1.4 → DeprecationWarning · v1.5 → UserWarning · v1.6 → removal.

  **What did NOT land (intentional):**
  - `LORA_PRESETS` was NOT touched — it's the source of the user-facing `--lora-preset` flag's values.
  - The existing test files (`tests/test_config.py`, `tests/test_config_extended.py`) still import `TRAINING_PRESETS` from `backpropagate.config`. Those will emit `DeprecationWarning` at test collection. Migrating the test imports to the canonical name is a v1.5 cleanup candidate (no functional regression; the warnings are visible under `pytest -W default` but silent under default test runs).

## CLOSED in Wave 6b features (2026-05-25 — CI / docs amend)

CI / docs items closed in this wave alongside Backend / Bridge / Frontend / Tests
landings. Items remain in this file as the historical record of v1.4 Wave 6b
scope; commit references attach at PR-merge time.

### CHANGELOG operator-facing entries added (Unreleased / Added section)

13 new entries landed in CHANGELOG.md `[Unreleased]` → `### Added`:

- **BACKEND-F-008: `mode="full"` for ≤3B model full fine-tuning** — names
  the 3B parameter ceiling + `RUNTIME_FULL_FT_MODEL_TOO_LARGE` error code +
  Biderman 2024 / Thinking Machines 2025 quality math. Closes the loop on
  the v1.4 promise the README anti-pitch section carried since the Wave 5.5
  AQLM retraction.
- **BRIDGE-F-001: `backprop ollama register|list|rm` triad** — closes
  **CIDOCS-A-001** explicitly. The README hero example that operators were
  copy-pasting (and failing on with the v1.3 argparse error) is now a real
  subcommand. The entry names the architectural deviation (nested vs flat
  subparser) + the upstream `ollama` CLI mental-model rationale.
- **BACKEND-F-002: `Trainer.estimate_vram()` Python API** — names the
  per-consumer breakdown (model_weights / lora_adapter / optimizer_state /
  activations / kv_cache / overhead) + the back-of-envelope accuracy claim.
- **BACKEND-F-003: MultiRunTrainer callback parity** — names 4 callbacks
  that pre-fix were structurally dead (on_step / on_run_start /
  on_run_complete / on_gpu_status) + Wave 5 audit attribution.
- **BACKEND-F-004: CheckpointManager filelock parity** — names the cross-host
  NFS / SMB / Lustre race + the atomic-write-with-lock alignment with
  RunHistoryManager.
- **BRIDGE-F-002: CLI log-config flags** — names `--log-level`,
  `--log-format=json|console`, `--log-file=<path>` on root parser + the
  CLI-flag-wins-over-env-var contract.
- **BRIDGE-F-007: --json on validate + replay** — names the
  schema_version-tagged JSON payload + the CI-consumer use case.
- **BRIDGE-F-010: info --json logging block** — names the
  `logging: {level, format, file, has_handler}` block.
- **BRIDGE-F-011: Root --help epilog rewrite** — names the workflow
  groupings (TRAINING / MULTI-RUN ANALYSIS / EXPORT + DEPLOY / UI + INFRA /
  DIAGNOSTICS).
- **BRIDGE-F-014: push --hub-revision / --hub-commit-message** — names the
  per-experiment-branch + CI-commit-message use cases.
- **BRIDGE-F-015: info --subcommand-tiers introspection** — names
  SUBCOMMAND_TIERS as the new explicit operator-facing surface.
- **FRONTEND-F-004: push-to-hub UI completeness** — names the
  `--include-base` checkbox + `--token-file` path field + existing
  CLI-semantics parity (file path takes precedence over inline token).
- **FRONTEND-F-005: dataset stats grid plumbing** — names the pre-fix
  hardcoded em-dash literal regression + wiring to DatasetStatsState fields.
- **TESTS-F-006: 6 coverage gaps closed** — names each of the six
  regression-pinning targets (Stage C save_merged warning; Wave 3.5
  untrained-save warning; model.eval try/finally invariant; FRONTEND-B-001
  was_deleted + action_in_flight; schema_version pinning;
  _read_hub_token_file mode-0600 warning).

### Handbook content (new pages + extensions)

- **NEW: `handbook/full-fine-tuning.md`** (`order: 2.5`, ~3500 words).
  Operator-friendly framing of mode='full' vs LoRA with: 3B ceiling
  explanation + `RUNTIME_FULL_FT_MODEL_TOO_LARGE` reference; when to use
  full FT (the narrow case — measured quality gap on operator's specific
  task); when to STAY with LoRA (the dominant case per Biderman 2024 +
  Thinking Machines 2025); Python + CLI examples; the four mode-specific
  SFTConfig changes (no PEFT config, gradient_checkpointing,
  paged_adamw_8bit, 10× lower LR); the three sub-3B preset escape hatches;
  a quality-math callout with the two load-bearing references in full
  citation form.
- **NEW: `handbook/estimate-vram.md`** (`order: 6.7`, ~2200 words). Operator-
  facing math for `Trainer.estimate_vram()` alongside the existing
  `backprop estimate-vram` CLI tier helper. Covers `VRAMEstimate` fields,
  sample estimates table (Qwen 2.5 7B LoRA r=256; Phi-4-mini-3.8B
  mode='full'; SmolLM3-3B mode='full'), and a limitations section.
- **EXTENDED: `handbook/cli-reference.md`** — added v1.4 rows for:
  `--mode=lora|full` on `train` + `multi-run`; `--log-level` /
  `--log-format` / `--log-file` rows in a new "Root-parser flags" section
  at the top; `--json` rows on `validate` + `replay`; `--hub-revision` +
  `--hub-commit-message` rows on `push`; `--subcommand-tiers` +
  `--env-vars` + `--error-codes` rows in the `info` section. Plus a new
  `backprop ollama` section covering the nested subparser triad with an
  explicit architectural-deviation note (operator-facing AND
  maintainer-facing).
- **EXTENDED: `handbook/error-codes.md`** — added the
  `RUNTIME_FULL_FT_MODEL_TOO_LARGE` row in the `RUNTIME_*` table naming
  the two-stage gate (preset-table lookup at `__init__` + authoritative
  `model.num_parameters()` check at `load_model()`) + the two recoveries.
- **EXTENDED: `handbook/env-vars.md`** — v1.4 CLI-flag-overlay note under
  the Logging section pointing operators at the new root-parser
  `--log-*` flags + the CLI-flag-wins precedence.
- **EXTENDED: `handbook/training.md`** — added the `mode` row to the
  Trainer parameters table with cross-reference to full-fine-tuning.
- **EXTENDED: `handbook/index.md`** — added the two new entries
  (full-fine-tuning + estimate-vram) to the "What's inside" list.

### README + landing page touch-ups

- **README.md** — updated the 16GB capability envelope table with a new
  row for ≤3B mode='full'. Updated the prose just below the table from
  "is planned as a `mode="full"` option for v1.4" → "ships in v1.4 as
  `mode="full"`" + cross-reference to the new handbook page. Updated the
  anti-pitch section's "Full-parameter fine-tuning of 7B+ models" bullet
  from "is planned for v1.4" → "ships in v1.4 as `mode="full"` ... a hard
  gate raises `RUNTIME_FULL_FT_MODEL_TOO_LARGE` for models > 3B". The
  Wave 5.5 retraction language (AQLM defer to v1.5) stays untouched.
- **site/src/site-config.ts** — updated the "LoRA + QLoRA + Unsloth"
  feature card title + description to "+ full FT" + the v1.4 mode='full'
  for ≤3B reference. The hero copy + Quickstart code snippet stay
  untouched (mode='full' is an advanced API for the narrow case; the hero
  stays on the 3-line LoRA default per the README's primary pitch).

### llms.txt updates

- Added `mode="lora"|"full"` to the `Trainer(...)` constructor signature
  line + v1.4 note naming the 3B ceiling + `RUNTIME_FULL_FT_MODEL_TOO_LARGE`.
- Added `trainer.estimate_vram(...)` to the Key API list with the v1.4 tag.
- Added a "Root-parser flags" preamble line to Key CLI Commands naming the
  three new `--log-*` flags + the CLI-flag-wins-over-env-var contract.
- Added `--mode {lora,full}` to `backprop train` + `backprop multi-run`
  surface lines.
- Added `--json` to `backprop replay` + `backprop validate` surface lines.
- Added the three new `backprop ollama` subcommands (register / list / rm).
- Added `--hub-revision` + `--hub-commit-message` to `backprop push`.
- Added `--subcommand-tiers` to `backprop info` + the v1.4 logging block
  note on the `info --json` payload.
- Added `RUNTIME_FULL_FT_MODEL_TOO_LARGE` to the error-code catalog list.

### Coordination outcomes (captured at amend time)

- **Backend's `mode="full"` implementation:** the handbook full-fine-tuning
  page names four mode-specific SFTConfig settings — no PEFT config,
  gradient_checkpointing=True (sqrt(L) activation memory),
  paged_adamw_8bit optimizer, 10× lower LR default — sourced from
  `trainer.py` `_build_sft_config` extension comments + the
  `_FULL_FT_DEFAULT_LR_DIVISOR = 10.0` module constant. If Backend's
  final shape differs, narrow the handbook before merge.
- **Bridge's `--mode` CLI plumbing:** cli-reference.md + README anti-pitch
  both assume `--mode=lora|full` is wired on BOTH `backprop train` AND
  `backprop multi-run`. If only one lands, narrow the handbook + README
  to match the actual runtime surface.
- **Bridge's `backprop ollama` triad:** cli-reference.md + CHANGELOG entry
  both name the three subcommands (register / list / rm) + the nested
  subparser convention. If Bridge ships only register (deferring list +
  rm to v1.5), trim the docs surface to match.
- **Frontend's `_GradioShim` deletion:** if the cascade was tractable and
  the deletion lands, the WAVE_6A_TODO.md "STILL DEFERRED" item 1 (the
  `_GradioShim` class as load-bearing today) needs an explicit closure
  mark before merge. If the cascade deferred to v1.5, the
  `STILL DEFERRED` entry stays + a CHANGELOG entry naming the deferral
  is sufficient + WAVE_5_FEATURE_AUDIT_NOTES.md is updated with the
  cascade survey.
- **Tests' 6 coverage-gap closures:** the CHANGELOG entry names six
  specific targets sourced from the Wave 5 / Wave 6a audit notes. If
  TESTS-F-006 added a different set of regression tests, narrow the
  CHANGELOG list to match the actual landings.

### Drift gate posture

The new docs surfaces (full-fine-tuning + estimate-vram pages, the new
`backprop ollama` cli-reference section, the new `--mode` row, the new
`RUNTIME_FULL_FT_MODEL_TOO_LARGE` error-codes row) all reference flags +
error codes + env vars that are landing in the same wave's Backend /
Bridge agent commits. The drift gate at PR-merge time MUST be re-run
against the merged branch state — pre-merge the docs reference future
state. Post-merge, the gate should report 0 findings (or call out which
surface drifted between agents). This is the canonical "draft docs from
spec, finalize after implementation lands" pattern flagged in the CI /
docs amend brief.
