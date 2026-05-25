# Wave 6a Foundation TODO — items deferred from earlier waves

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
  2. The broader "fold the Gradio-era legacy modules into the new middleware shape OR split `ui_security.py` into `_legacy.py` + `_core.py`" decision. The Wave 6a rename only touched the public-symbol surface, not the file structure. Per the FRONTEND-A-009 audit, either fold these into the new middleware shape or split `ui_security.py` into `ui_security_legacy.py` (preserved for back-compat / external callers / tests) and `ui_security_core.py` (the `get_ui_output_dir` + `sanitize_filename` + `safe_markdown_fence` + `AuthBadgeContext` helpers the Reflex UI actually uses).
  3. `DEFAULT_GRADIO_CSP` + `get_gradio_csp` *removal* — Wave 6a kept them in place (they already had the in-place `DeprecationWarning` shape from Wave 2 FRONTEND-A-003 + Stage C FRONTEND-B-007; deletion happens in v1.6 alongside the rest of the legacy shim).
  4. None of the Gradio-era classes (`SessionManager`, `ConcurrencyLimiter`, `EnhancedRateLimiter`, `JWTConfig`/`JWTManager`, `CSRFToken`/`CSRFProtection`, `SecureSessionHandler`) are wired into the Reflex middleware (the new middleware does its own rate-limit via `_SlidingWindow` rather than `EnhancedRateLimiter`); their fate is decided in the Wave 6b / v1.5 split-or-fold decision above.

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
