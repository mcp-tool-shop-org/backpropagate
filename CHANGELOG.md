# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — v1.3 prep

### Fixed

- **CLI flag-vs-runtime mismatches.** Two CLI flags advertised functionality the runtime never delivered: (a) `--host <addr>` was accepted and validated but never threaded to the Reflex subprocess argv — the UI silently stayed loopback-only since v1.1.0; (b) `--share` was a no-op post-Reflex-migration (Gradio's gradio.com tunnel was removed in v1.1.0 and nothing replaced it). v1.3 fixes (a) by wiring `--host` through to the Reflex backend bind via the `--backend-host` argument that landed in reflex 0.9.2, and fixes (b) by implementing real `cloudflared`-based tunneling for `--share` (consumes the existing `BACKPROPAGATE_UI_SHARE_HOST` Origin-allowlist plumbing the auth middleware was already wired for). Operators relying on either flag should re-verify their deployment surface; the SSH-port-forward pattern in `handbook/security.md` remains an alternative for `--share` for operators without `cloudflared` installed.
- **Web UI auth-success now leaves an audit trail.** v1.1.x had no log line for a successful cookie-set on the GHSA-f65r-h4g3-3h9h surface — operators could see failed-auth lines (close code 4401 / 4403) but never knew which cookie just succeeded. v1.3 emits one `auth_success` INFO line per session at the cookie-set sites (both TOKEN_AUTO and EXPLICIT_CREDS / PRODUCTION modes), with `{user, mode, host}` fields. Per-request validation passes log at DEBUG. No cookie value, no password, no Basic-header bytes are recorded; the line is safe to ship to a central log aggregator.
- **Test surface no longer silently green-passes on regression.** Eight tests across the auth-middleware + SLAO-integration + GPU-emergency-callback + Hypothesis-property-test families were tautological — they `if`-gated on the value under test and silently skipped the assertion when the value was falsy, so a regression that returned `None` from the helper would still report green. v1.3 converts them to `assert ... is not None` precondition + property checks. Notable: `test_token_lock_file_mode_0600` and the cookie-hardening test pair now skip-with-reason (not silently green) when the underlying surface is not yet wired; they re-engage when the Wave 5/6 auth-middleware polish lands.

### Added

- **Recommended isolated-install path documented in README.** The Installation section now leads with `pipx install backpropagate` / `uv tool install backpropagate` as the recommended modes (isolated venv + automatic PATH integration), with `pip install` retained for users managing their own venv. The original `pip install backpropagate[extra]` table is preserved below as a reference.
- **`bin/backpropagate.js` is now a friendly-error shim.** Running `backpropagate` after `npm install -g backpropagate` prints clear install guidance for the supported channels and exits with code 2 (configuration error). Operators who land here from an old README copy get a single screen of next-step commands rather than a silent download failure.
- **Coverage floor is now a single source of truth.** `pyproject.toml` `[tool.coverage.report].fail_under` is authoritative; the CI workflow reads it via `tomllib` at run time so bumping the floor in one place takes effect in both surfaces.
- **`release.yml` is idempotent and re-runnable.** The `gh release create` step now precheck-skips when the release already exists (no more HTTP 422 "Release.tag_name already exists" on retry), and `workflow_dispatch` is enabled so a maintainer can re-run from the GitHub UI without an extra tag-push. A `concurrency:` block with `cancel-in-progress: false` serializes per-tag releases so a mid-flight `npm publish` is never cut off after the Sigstore attestation is signed.
- **Bandit scan now uploads a JSON artifact.** The gating Bandit step writes a structured `bandit-gating.json` alongside the txt output, and the artifact is uploaded on every run (including failures) so the maintainer can grep the JSON for `test_id` / `filename` / `line` on a red run without re-running CI locally.

### Changed

- **`verify.sh` accepts `--format=human|json`.** Default stays `human` (the existing banner output). `--format=json` emits one JSON object per stage (`{"stage", "status", "exit_code", "duration_seconds"}`) plus a final aggregate object — CI can parse the stream without screen-scraping. Each stage's stdout/stderr is captured to `verify-<stage>.log` so the JSON channel stays parseable.
- **CI workflow action SHAs aligned across workflows.** `release.yml`'s `actions/setup-python` pin moved from v6.0.0 → v6.2.0 (already in use by `ci.yml` + `publish.yml`) so the Node-runtime parity across workflows is consistent.

### Removed

- **npm distribution deprecated.** The `bin/backpropagate.js` shim used to bootstrap a Linux venv or download PyInstaller binaries from a GitHub Release via `@mcptoolshop/npm-launcher`. The binary build pipeline failed three consecutive times in v1.2.0 and the v1.0/v1.1/v1.2 release tags have zero attached binary assets — the launcher would 404. The shim now prints install guidance for the supported channels (`pipx install backpropagate` recommended, plus `uv tool install backpropagate` and `pip install backpropagate`) and exits 2. The npm package stays published so this message reaches existing `npm install -g backpropagate` users. The `@mcptoolshop/npm-launcher` runtime dependency was dropped from `package.json` (every `npm install` was pulling dead code).
- **`release-binaries.yml` workflow deleted.** It had failed 4 of the last 5 release runs and produced no shipped assets at any v1.x tag. Surviving comment references in `publish.yml` / `release.yml` / `ci.yml` / `bin/backpropagate.js` were updated to reflect retirement. The PyInstaller `.spec` files and the handbook migration page remain in the v1.3 brief for Wave 5/6 follow-up.

### Security

- **CRITICAL-only `pip-audit` + Trivy floor preserved from v1.2.0.** No relaxation in v1.3; the same hard gates surface CRITICAL transitive CVEs while the advisory MEDIUM+ feed continues to populate the GitHub Security tab. The Bandit gating step now emits both txt (gating surface) and JSON (post-mortem artifact) for MEDIUM severity + MEDIUM confidence and above.
- **`SECURITY.md` is the canonical reporting policy + supported-versions surface; the operator-facing threat model + auth-middleware mode matrix is at `handbook/security.md`.** `CONTRIBUTING.md`'s historical pointer to `SECURITY_AUDIT_REPORT.md` (which became a stub in Wave 1) now points at the live docs instead.

### Known issues / tech debt

- 14 ERROR-severity Trivy alerts on `main` (incl. PyJWT CVE-2026-32597) are routed to the v1.3 P0 dep-sweep wave. PyJWT is NOT on the auth middleware runtime path — `ui_app/auth.py` uses stdlib `hmac` rather than `jwt`.
- **Python 3.10 reaches upstream EOL October 2026.** v1.3 still supports 3.10 (CI matrix runs 3.10 / 3.11 / 3.12 / 3.13). A future release (target: v1.4) will drop 3.10 to align with the upstream EOL. Operators standing up new installs should prefer Python 3.11 or 3.12 — 3.11 is the most-tested floor (the UI + Windows + macOS smoke cells all run on 3.11).
- **PyInstaller binary distribution and the handbook migration page remain Wave 5/6 scope.** The `release-binaries.yml` workflow is deleted; the `.spec` files and the doc page that walks operators from "I used the binary" to "use pipx instead" are the second half of that migration.

## [1.2.0] - 2026-05-23

A dogfood-swarm release closing the **v1.1.x auth-bypass advisory** and a truth-in-advertising sweep across CI gates, docs, and pinned numbers. No feature regressions; full backward compatibility with v1.1.x.

### Added

- **Real auth middleware (Option B per DESIGN_BRIEF)** — Starlette ASGI middleware via `rx.App(api_transformer=basic_auth_transformer)`, gates HTTP routes AND `/_event` WebSocket upgrade. Four modes: `no_auth_local_only` (loopback bind, no auth), `token_auto` (per-launch random token in URL), `explicit_creds` (HTTP Basic via `--auth user:pass`), `production` (basic + Host allowlist + Origin allowlist). HMAC-signed `HttpOnly; SameSite=Lax; Secure-when-non-loopback` `backprop_sess` cookie with 12h TTL. WebSocket cookie validated PRE-`websocket.accept()` (closes pre-accept DoS vector); failed auth → close code 4401, failed Origin → 4403. `ENFORCEMENT_AVAILABLE` flipped `False → True`. CLI `--auth user:pass` now flows through to the subprocess; `--share` without `--auth` and `--host <non-loopback>` without `--auth` are HARD ERRORS preserving the v1.2 truth-in-advertising contract. Polish (footer badge UI states, Jupyter-pattern startup banner, lock-file token, `--auth-file` flag, request-logging + rate-limit middleware) deferred to v1.3 brief.
- **`backprop runs` data API** — versioned JSON enumerator over `RunHistoryManager` returning `{schema_version: "1", generated_at, output_dir, runs: [{run_id, status, model, dataset, duration_seconds, started_at, completed_at, checkpoint_path, loss: {min, final}}]}` with `--status` + `--limit` filters. Frontend `/runs` page consumes the same `_build_runs_payload()` helper in-process to avoid subprocess cold-start cost.
- **`backprop info --env-vars`** — enumerates every `BACKPROPAGATE_*` env var (72 total) via `Settings.model_fields` walk. Secret-flagged fields (`AUTH_PASSWORD`, `JWT_SECRET` via `json_schema_extra={"secret": True}`) print as `<secret>`. `--json` for machine consumption.
- **Run-history UI page** — Reflex `/runs` route + `RunsState.load_runs` event handler. Read-only history table with 7 columns (status, model, dataset, duration, run-id, started_at, outcome). Status-filter dropdown + manual refresh button. Direct in-process `RunHistoryManager(history_dir)` consumption (no subprocess on WS loop). Per-run drill-down deferred to v1.3.
- **ERROR_CODES catalog completion** — promoted the 8 codes documented as workarounds in v1.1.x: 4× `HUB_PUSH_*` (`INVALID_REPO`, `NOT_FOUND`, `NETWORK`, `UNKNOWN`), `SLAO_MERGE_DIVERGED`, `PEFT_API_INCOMPATIBLE`, `UI_OUTPUT_DIR_FORBIDDEN`, `INPUT_AUTH_INVALID_SHAPE`. Plus `INPUT_PATH_TRAVERSAL`, `RUNTIME_OOM_RECOVERY_EXHAUSTED`, `RUNTIME_OOM_ADJACENT` added across Waves 3.5/Stage C. `cli.py:_BRIDGE_LOCAL_ERROR_CODES` workaround dict deleted. `backprop info --error-codes` now enumerates every code emitted at runtime.
- **Catalog-drift regression test** (`tests/test_error_codes_catalog.py`) — 4 AST-walker tests scan every `backpropagate/**/*.py` source file for `code='...'` literals and assert each resolves to a key in `exceptions.ERROR_CODES`. Closes the entire class of catalog-drift bugs that surfaced in Stage A → Stage B → Wave 6.
- **`tests/helpers/` subpackage** — promoted the old `tests/test_helpers.py` (which masqueraded as a test file under the `test_*` prefix and was collected on every run) to a proper subpackage with `callbacks.py` + `asgi.py` + `ws.py` modules. `tests/helpers/asgi.py` provides `make_asgi_client` + `basic_auth_header` + `reflex_app_with_auth_enforced` fixture for the new auth middleware test surface.
- **CORS allowlist** — `rxconfig.py` now configures `cors_allowed_origins` to loopback (`localhost:3000/7860`, `127.0.0.1:3000/7860`) by default. Additive override via `BACKPROPAGATE_UI_CORS_EXTRA_ORIGINS` env var. Lands BEFORE the auth middleware in source order so future extensions append rather than replace.
- `CITATION.cff` — GitHub "Cite this repository" surface; cites the SLAO paper (arXiv:2512.23017) as the research backing `multi_run.py` + `slao.py`.
- Handbook page `migrations.md` — v1.1.x → v1.2.0 operator-facing migration narrative (refuse-to-start contract, removed `[observability]` extra, behavioural fixes for `TrainingCallback` hooks / `resume_from` / `train_on_responses_only`).
- Handbook page `security.md` — threat model, GHSA-f65r-h4g3-3h9h advisory, four-layer defense in depth, SSH port-forwarding recipe, auth-middleware mode matrix, anti-patterns. Single surface for "I want to expose this safely."
- `httpx>=0.27.0` added to `dev` optional-dependencies for the new ASGI test harness.

### Fixed

- **CRITICAL: Web UI authentication contract not enforced (GHSA-f65r-h4g3-3h9h)** — v1.1.0 and v1.1.1 advertised `--share + --auth` enforcement but `backpropagate/ui_app/**` never read `BACKPROPAGATE_UI_AUTH`. Running `backprop ui --auth` or `backprop ui --share --auth` was unauthenticated. v1.2.0 lands a **4-layer defense in depth** that refuses to start the UI when auth is requested but enforcement is not actually wired:
  1. `cli.py:cmd_ui` refuses-to-start with `RUNTIME_UI_AUTH_NOT_ENFORCED` when `--auth` or `--share` is set.
  2. `cli.py:cmd_ui` strips ambient `BACKPROPAGATE_UI_AUTH` from the subprocess env when `--auth` is not passed (closes BRIDGE-B-001 ambient-env bypass: an operator-set env var would otherwise reach the Reflex subprocess and create the illusion of auth coverage).
  3. `ui_app/app.py` refuses-to-construct at module import time when `BACKPROPAGATE_UI_AUTH` is set and `ENFORCEMENT_AVAILABLE=False`.
  4. `rxconfig.py` has the same module-import guard (catches `python -m reflex run` direct invocations that bypass `cli.py`).

  All four layers key off the `backpropagate/ui_app/auth.py::ENFORCEMENT_AVAILABLE` flag — when the middleware lands, flipping that single boolean re-enables every layer. See GHSA at https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/GHSA-f65r-h4g3-3h9h (published 2026-05-23T08:58Z, CVE pending assignment).
- **BACKEND-B-001 silent eval-mode-stuck after CUDA OOM** — `multi_run.py:_compute_validation_loss` left the model stuck in `eval()` mode after a CUDA OOM (or any exception escaping the validation loop). The next training run silently produced no gradient updates with no operator-visible signal — operators saw "training completed" but the model didn't learn. Fixed by wrapping the validation body in `try / finally: model.train()` so the train-mode invariant is restored even on exception. Originally classified Stage A MEDIUM (BACKEND-A-002) and deferred; Stage B re-audit escalated to CRITICAL on impact.
- F-002 multi-run resume: safetensors loader was being called on `.bin` adapter files, raising silently. Now dispatched by extension.
- export.py `register_with_ollama` UnboundLocalError in finally-block masking the real OllamaRegistrationError.
- export.py subprocess SIGINT propagation: ollama-create and llama.cpp child processes now receive proper termination on Ctrl+C.
- cli.py main() lacks top-level exception net — Ship Gate B2 exit-code contract violations on unhandled errors.
- **BACKEND-F-003 — `TrainingCallback` hooks now actually fire.** v1.1.x defined `on_step` / `on_epoch` / `on_save` in the public dataclass API but only `on_complete` + `on_error` invoked from `Trainer.train`. v1.2.0 adds `_BackpropCallbackAdapter(HFTrainerCallback)` bridging HuggingFace's `on_step_end` / `on_epoch_end` / `on_save` to the user callback. Per-hook try/except preserves the v1.1.0 callback-isolation contract. If your callback was a silent no-op in v1.1.x, expect to see calls you didn't see before.
- **BACKEND-F-017 — `Trainer.train(resume_from=run_id)` now actually resumes.** v1.1.x accepted the parameter, reloaded the run-history record, and reused the `run_id` — but never passed `resume_from_checkpoint` to TRL's `SFTTrainer.train()`. Inner training silently restarted from step 0. v1.2.0 threads the checkpoint path through; HuggingFace auto-scans `output_dir/checkpoint-<N>` and resumes optimizer / scheduler / step counter. Missing-on-disk → WARN + fresh start (preserves prior no-crash behavior). `MultiRunTrainer.resume` was always real; single-run now is too.
- **BACKEND-F-014 — `train_on_responses_only` is now tokenizer-aware.** v1.1.x hardcoded ChatML literals (`<|im_start|>user` / `<|im_start|>assistant`). Llama 3 (`<|start_header_id|>user<|end_header_id|>`), Gemma (`<start_of_turn>user`), Phi-3 (`<|user|>`), Mistral (`[INST]`) silently no-op'd the masker — the model trained on user prompts as well as assistant responses. v1.2.0 detects the tokenizer family by name + falls back to a ChatML-shape probe; resolved markers persisted in run-history for auditability. Operator override via `Trainer(response_markers=("<usr>", "<asst>"))` short-circuits detection. Mistral WARNs (no explicit assistant-turn marker; operators should pass `train_on_responses=False` or supply explicit markers).
- **BRIDGE-F-011 — `backprop info` no longer reports "Gradio".** v1.1.x continued to advertise `FEATURE_DESCRIPTIONS['ui'] = "Gradio web interface"` after the Reflex migration. Also fixes a latent bug: `_detect_features` UI probe was importing `gradio` not `reflex` — operators with `[ui]` installed got `FEATURES['ui']=False` and the misleading "install with: pip install backpropagate[ui]" hint. v1.2.0 reports "Reflex (Radix UI) web interface" and probes `reflex`.

### Changed

- **CI gates re-tightened.** v1.1.0 claimed pip-audit + Trivy + Bandit + Semgrep + TruffleHog all gated on findings; v1.1.x rolled most of them back to advisory. v1.2.0 restores hard floor gates: mypy hard (or `ui_app/` override), pip-audit CRITICAL floor, Trivy CRITICAL floor, aggregate gate no longer `continue-on-error`. TruffleHog confirmed retired (delegated to Trivy built-in secret scanner).
- **Web UI `--share` + `--auth` contract inverted post-middleware.** v1.1.x advertised `--share + --auth` enforcement; runtime ignored `--auth`. Wave 3.5 made BOTH `--share` and `--auth` refuse-to-start as the patched-version interim. Wave 6 landed the real middleware; v1.2.0 final contract: `--auth user:pass` flows through to the subprocess, `--share` without `--auth` is a HARD ERROR (preserves truth-in-advertising), `--host <non-loopback>` without `--auth` is a HARD ERROR (DNS-rebinding defense). Use SSH port-forwarding when `--share` is not desired.
- Test count re-pinned to actual `pytest --collect-only`: 1865 (was 1957 in the abandoned v1.1.2 entry; Wave 1 dropped `test_init_lazy_loading.py`, Waves 2/3/3.5/Stage C added regression coverage, Wave 4.5 removed the Gradio-legacy test surface, Wave 6 added 27 new tests for auth middleware + catalog drift + contract violations).
- Validation tightening: `ollama create` model_name and `huggingface push` repo_id are validated against allowlist regexes.
- `scripts/repin_test_count.sh` added so future maintainers can re-pin the count consistently (run after any `tests/` change; canonical pin sites listed in the script header).

### Removed

- **`backpropagate.ui_gradio_legacy` + `backpropagate.theme_gradio_legacy`** — the v1.0 Gradio implementation, preserved through v1.1.x as reference. The Reflex UI (canonical from v1.1.0) is now the only Web UI surface. Package-level `backpropagate.launch` / `create_backpropagate_theme` / `get_theme_info` / `get_css` continue to raise `ImportError` via `__init__.py.__getattr__` with migration messages pointing at `backprop ui`.
- tests/test_init_lazy_loading.py — fully skipped legacy file; replacement coverage already lives in test_init_imports.py.
- tests/test_ui_gradio_legacy_components.py + tests/test_theme_gradio_legacy.py — exclusively tested the removed legacy modules.
- The legacy-import test classes (`TestValidatePathInput`, `TestSanitizeModelName`, `TestSanitizeTextInput`, `TestGenerateAuthToken`, `TestLaunchSecurity`, `TestUISecurityExports`) inside `tests/test_ui_security.py` — they imported helpers from `ui_gradio_legacy` at test-body level and could not gracefully degrade after the module's removal. Equivalent surface for the Reflex UI is covered by `tests/test_ui_app_*.py` plus the `EnhancedRateLimiter` / `FileValidator` suites that remain in `test_ui_security.py`.
- **`[observability]` extra removed** — the extra was advertised as OpenTelemetry distributed tracing but zero modules imported `opentelemetry`. Rather than ship another doc-lie, the extra is removed in v1.2.0; the `[full]` bundle no longer pulls it. Real OpenTelemetry integration may land in a future release. Operators using `pip install backpropagate[observability]` should drop the extra from their install line and install `opentelemetry-api` / `opentelemetry-sdk` directly if they need them.

### Tests

1957 → 1865 (-92 net, pinned 2026-05-23 via `pytest --collect-only`): added across Waves 1 / 3.5 / Stage C / Wave 6 — validator regression coverage, GHSA-f65r-h4g3-3h9h auth-bypass test suite, ENFORCEMENT_AVAILABLE-flipped path, unsloth_fallback + pause_on_overheat wiring, HF retry loop timing, SLAO_MERGE_DIVERGED layer-name assertions, TrainingLogger capsys coverage, run_id correlation chain, **23 new auth middleware tests** (`tests/test_auth_middleware.py` covering 4 modes, Host/Origin allowlists, WS pre-accept cookie validation, close codes 4401/4403, HMAC signature roundtrip, MLflow-CVE-style default-credential audit), **4 catalog-drift regression tests** (`tests/test_error_codes_catalog.py`), **regression tests for the 5 contract-violation fixes**. The net drop is from Wave 4.5 Gradio-legacy removal (4 deleted test files + 6 hard-import classes inside test_ui_security.py + previously-counted legacy fixtures that no longer collect). Coverage threshold holds at 50%. Final pytest run: **1856 passed, 10 skipped, 0 failed in 31.39s**.

- **TESTS-B-001 off-rig CI would have failed silently** — `test_register_with_ollama_failure` patched `subprocess.run`, but Wave 1's `register_with_ollama` had migrated to `_run_subprocess_interruptible` (Popen-based). The patch never fired; the test passed only because the `ollama` binary happened to be installed on this rig. CI runners without `ollama` installed would have failed the test. The patch target was corrected to the new subprocess helper.

### Known issues / tech debt

- **Known HIGH/MEDIUM/LOW transitive-dep CVEs deferred to v1.3.** v1.2.0's hardening contract restored the CRITICAL floor gate (Trivy CRITICAL passes; 2 transitive CRITICAL CVEs cleared in Wave 6.6 via uv.lock bumps — `authlib` 1.6.8 → 1.7.2, `nltk` 3.9.2 → 3.9.4). Trivy also surfaces 23 HIGH/MEDIUM/LOW alerts across transitive deps (`gitpython`, `diffusers`, `astro`, `devalue`, etc.) — CVEs disclosed in the window between v1.1.1 ship (2026-05-21) and v1.2.0 ship (2026-05-23), present on `main` and **not introduced by the v1.2.0 PR**. Per the documented CRITICAL-floor scope (see the "CI gates re-tightened" entry above), these are not v1.2.0 ship-blockers. A targeted transitive-dep CVE sweep is scheduled for v1.3 as a P0 dep-sweep wave (see the v1.3 brief). The PR's `Trivy` aggregator check shows them as "new" because GitHub's SARIF baseline-diff compares against `main`'s last upload (which pre-dated several of these CVE feeds); post-merge, the same alerts will surface on `main`'s next scan and the PR-vs-main delta becomes zero.
- **Auth middleware polish deferred to v1.3 brief**: footer auth-badge UI states, Jupyter-pattern startup banner, lock-file token for machine-to-machine auth, `--auth-file` flag, request-logging middleware, rate-limit middleware. The middleware shipped in v1.2.0 is security-complete (closes the GHSA-f65r-h4g3-3h9h advisory); the deferred items are operator-UX polish + defense-in-depth that the v1.3 brief will track. The v1.3 design brief's `v1_3_followup_items` section tracks the full list.
- `[observability]` removal is preparatory for real OpenTelemetry integration in v1.3 (or later). The extra was removed rather than left advertising a no-op surface; if you need OTLP today, install `opentelemetry-api` / `opentelemetry-sdk` directly.
- Translated READMEs (`README.{es,fr,hi,it,ja,zh,pt-BR}.md`) are regenerated immediately before publish — translation is the last release-prep step before the npm publish + GitHub release per the load-bearing release-ordering rule.

## [1.1.1] - 2026-05-21

### Fixed

- CI workflow action SHAs (4 broken pins from the Stage A Wave 1 / Wave 3 SHA-pin sweep): `trufflesecurity/trufflehog` (34e114876b → 37b77001d0 / v3.95.3), `actions/upload-pages-artifact` (cd2ce8fc → 56afc609 / v3 — the previous SHA was actually `deploy-pages@v5` on the wrong action), `actions/deploy-pages` (ddc015e5 → cd2ce8fc / v5 — previous SHA didn't exist), `astral-sh/setup-uv` (7b1f4a76 → caf0cab7 / v3 — previous SHA didn't exist).
- v1.1.0 release-binaries workflow failed at `Getting action download info` because of the broken setup-uv pin; this patch lets the next release-binaries run actually build and upload the PyInstaller binaries for Windows + macOS.

### Notes

- **No user-facing changes vs v1.1.0** — pure CI hygiene. PyPI / npm packages are byte-identical except for the version string. v1.1.0 remains valid and installable.
- v1.1.0 GitHub Release page has the release notes; v1.1.0 PyInstaller binaries are unshipped — they ship attached to the v1.1.1 release once `release-binaries.yml` succeeds with the fixed setup-uv SHA.

## [1.1.0] - 2026-05-21

A minor release that takes the project from "polished v1" to "real v1" via a 10-wave dogfood swarm. Bug + security pass, proactive health pass, UX humanization, full UI redesign (Gradio → Reflex), 5 P0 features.

### Added

- **Reflex web UI** — the optional `[ui]` extra now installs Reflex (Radix UI) instead of Gradio. Pure-Python implementation, WebSocket-driven live state, refined Ocean Mist palette, full dark + light mode, WCAG 2.4.7 focus indicators, 30 SVG icons, heartbeat / sparkline / event-log / structured-error / recovery-banner patterns
- **Hugging Face Hub push** — `backprop push <local> --repo <owner/name>` + `backprop export --push-to-hub <repo>` for one-shot export+push. Adapter-only by default; `--include-base` for the full merged model. Token resolution from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` / HF CLI cache. `model_card.md` is mirrored to the repo's `README.md` so HF picks it up as the model card
- **Resume from checkpoint** — `backprop resume <run_id>` (and `backprop train --resume <run_id>` / `backprop multi-run --resume`) reconstructs a crashed or interrupted run from RunHistoryManager + the atomic checkpoint manifest. A 5-run multi-run that crashes at run 4 is now recoverable
- **Run history** — `RunHistoryManager` is now actually wired into Trainer + MultiRunTrainer. New `backprop list-runs` (with `--json`, `--status`, `--limit` filters + aligned columns) and `backprop show-run <run_id>` (partial-prefix matching) subcommands surface the history
- **Model card generation** — every export emits a `model_card.md` following the HF model-card schema, with full provenance (run_id, base model, dataset hash, seed, training duration, ASCII loss sparkline, Ship Gate trust signals). Opt out via `--no-model-card`
- **Experiment tracking auto-wired** — `[monitoring]` extra (W&B, TensorBoard) now actually integrates. `report_to` defaults to `"auto"` (detect what's installed); the run shows up with name `backprop-<run_id_short>` for cross-system correlation
- **Atomic checkpoint writes** — Trainer.save / SLAOMerger.save / export_lora / export_gguf all write to `<path>.partial` then rename to final. Disk-full mid-write no longer leaves corrupt artifacts
- **OOM auto-recovery** — `Trainer(oom_recovery=True)` (default-on) halves batch_size + doubles gradient_accumulation_steps on `torch.cuda.OutOfMemoryError`, preserving effective batch. Aborts after 3 consecutive failures at batch=1
- **HF Hub transient retry** — every `from_pretrained` / `load_dataset` / `snapshot_download` retries on 5xx / 429 / connection errors with exponential backoff. 401 / 403 / 404 surface in < 1s with cause-classified hints
- **GPU pause-on-overheat** — `Trainer(pause_on_overheat=True)` now actually pauses training (the wiring was a no-op in v1.0)
- **Unsloth fallback** — `Trainer(unsloth_fallback=True)` (default-on) falls back to AutoModelForCausalLM + peft on Unsloth failures
- **run_id correlation** — every training run mints a UUID4 that flows through every log line + checkpoint manifest + SLAO merge record
- **Stable error codes** — `BackpropagateError.code` is now an explicit Ship Gate registry-prefixed identifier on every subclass. 28-entry `ERROR_CODES` catalog visible via `backprop info --error-codes`. `cause_category` enum on ModelLoadError surfaces cause-specific remediation hints
- **CLI exit codes** — proper 0 / 1 user-error / 2 runtime-error / 3 partial-success / 130 SIGINT contract
- **Stage C humanization** — structured errors with actionable hints, progress feedback on long ops, bare `backprop` prints help, `backprop info --json` for support attachments, friendly first-run messages
- **CI hardening** — every third-party GitHub Action SHA-pinned. PyPI publish via OIDC trusted publishing (Sigstore provenance). Docker image digest-pinned + HEALTHCHECK. Multi-OS test matrix (Linux + Windows + macOS + Python 3.13). pip-audit + Trivy + Bandit + Semgrep + TruffleHog all gate on findings
- **Documentation** — new handbook pages: `error-codes.md`, `troubleshooting.md`, `env-vars.md`, `cli-reference.md`. README Troubleshooting + Reporting bugs + Web UI subsections. `examples/quickstart.jsonl` so the "3 lines" Quick Start runs on a clean install

### Changed

- **Default model** — `Trainer()` (and `backprop train` / `multi-run` CLI defaults) now use `Qwen/Qwen2.5-7B-Instruct` instead of `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`. The non-quantized form works without bitsandbytes; users who want the bnb-4bit speedup install `[unsloth]` and pass `--model unsloth/...` explicitly
- **safe_path stricter** — absolute path + `..` segment + no `allowed_base` argument now raises `PathTraversalError` instead of warn-only-and-pass-through
- **Multi-run validation-overlap fix** — `_get_data_chunk` and `_get_replay_samples` now hard-cap at the train/validation boundary. Silent contamination is impossible; `ConfigurationError` surfaces a clear "reduce samples or increase dataset" hint
- **Random state isolation** — multi-run replay sampling uses a local `random.Random(seed)` instead of mutating the global Python RNG
- **SLAO NaN/inf detection** — `SLAOMerger.merge` raises `SLAO_MERGE_DIVERGED` with run_index + run_id + offending layer on non-finite weights
- **Rate limiter Address handling** — `_extract_client_ip` now correctly reads `.host` from Starlette's `Address` namedtuple (was including `:port`, giving every TCP connection its own bucket)
- **UI output dir denylist** — `BACKPROPAGATE_UI__OUTPUT_DIR` is validated against a denylist (`/etc`, `~/.ssh`, etc.) on first use
- **`--share` + `--auth` gating** — `backprop ui --share` now requires `--auth user:pass` (or explicit env-var opt-out with 5-second grace period + loud warning)
- **Scorecard re-audited** — B (Error Handling) row 3/7* → 5/7. Total 23/31 → 25/31

### Removed

- **Gradio web UI** — moved to `backpropagate/ui_gradio_legacy.py` with a DEPRECATED docstring; preserved through v1.1.x as reference and fully removed in v1.2.0. `backpropagate.launch` / `create_backpropagate_theme` / `get_theme_info` / `get_css` now raise `ImportError` with the migration message

### Tests

1654 → 1766 (+112): regression tests for every Stage A/B contract that landed and every P0 feature that shipped. Coverage threshold holds at 50%.

## [1.0.5] - 2026-04-15

### Fixed

- Release binaries workflow: v1.0.4 tag was cut before Linux exclusion fix landed, causing >2GB upload failure

## [1.0.4] - 2026-04-14

### Fixed

- Linux binary build: replace CUDA torch (~870MB) with CPU-only torch (~200MB) after install to keep binary under 2GB GitHub release limit
- Strip step SIGPIPE crash: `du | head -5` with `set -eo pipefail` caused false build failure

## [1.0.3] - 2026-04-14

### Added

- `release-binaries.yml` workflow for standalone PyInstaller binaries on Windows + Linux

### Fixed

- PyInstaller build pipeline iteration: hidden-import handling for torch/transformers (recursion limit), `--collect-data` removed to stay under 4GB onefile cap, Linux binary size reduction via strip + module exclusion (lead-up fixes; the final size cut that actually landed under 2GB shipped in v1.0.4)
- Full-install CUDA-torch override on Linux (uses CPU torch index instead)
- `pywin32-ctypes` dependency for Windows PyInstaller builds
- Forced uninstall of CUDA packages before PyInstaller to avoid CUDA torch contamination

## [1.0.2] - 2026-03-25

### Fixed

- CLI `--version` was hardcoded to 0.1.0 — now reads from package metadata dynamically
- `__init__.py` docstring referenced v0.1.0 — updated to v1.0.1
- SECURITY.md supported versions updated from 0.x.x to 1.0.x

### Added

- 2 new version regression tests in test_cli.py

## [1.0.1] - 2026-02-27

### Added
- Ship Gate audit — all hard gates pass (23/31 checked, 14 skipped, 100%)
- verify.sh — single-command verification script (Ship Gate D1)
- Proper CLI exit codes: 1 user error, 2 runtime error, 3 partial success (Ship Gate B2)
- SHIP_GATE.md (the scorecard itself is rendered inline in README and on the landing page; no standalone SCORECARD.md file)

### Changed
- Scorecard in README and landing page reflects actual `shipcheck audit` results

## [1.0.0] - 2026-02-27

### Changed
- **v1.0.0 stable release** — production-ready
- Development status upgraded from Alpha to Production/Stable

## [0.1.7] - 2026-02-27

### Added
- Codecov badge in README
- Quality scorecard in README and landing page (46/50)
- Privacy section in README

### Changed
- Logo URL updated to brand repo (centralized)
- Landing page footer standardized to MCP Tool Shop link
- Landing page scorecard section added
- Updated translations (7 languages)

## [0.1.4] - 2026-02-22

### Fixed
- **Bandit Security Scan** - Fixed false positive security scan issues that caused CI to fail

## [0.1.3] - 2026-02-22

### Added
- **Qwen2.5-3B model preset** - Smaller model for faster iteration and testing on 16GB VRAM
- **Official Qwen model fallback** - When pre-quantized models have corrupted cache, fall back to official models with `load_in_4bit=True`
- **Local dataset path helper** - `DatasetLoader.from_local()` for easy loading of local JSONL/JSON files

### Changed
- **CUDA_LAUNCH_BLOCKING now optional** - Disabled by default to improve training speed (was slowing down RTX 5080)
- **Default model updated** - Changed default from `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` to `Qwen/Qwen2.5-7B-Instruct` for better reliability
- **Documentation** - Beefed up README with more badges, features, and GitHub Pages integration

### Fixed
- **BitsAndBytes JSON decode error** - Added fallback handling when pre-quantized model cache is corrupted

---

## [0.1.0] - 2026-01-19

### Added

#### Core Features
- **Trainer class** - Simple API for LLM fine-tuning with smart defaults
- **Multi-run training (SLAO)** - Multiple short runs with LoRA merging to prevent catastrophic forgetting
- **QLoRA support** - 4-bit quantization for training 7B models on 16GB VRAM
- **Windows support** - Pre-tokenization, safe multiprocessing, xformers auto-disable

#### Dataset Handling
- **DatasetLoader** - Auto-detect format (JSONL, CSV, HuggingFace)
- **Quality filtering** - Filter by token count, turn count, assistant presence
- **Perplexity filtering** - Remove outliers using GPT-2 perplexity scores
- **Deduplication** - Exact and MinHash-based duplicate removal
- **Curriculum learning** - Order samples by difficulty for progressive training

#### Export & Deployment
- **LoRA export** - Save adapter weights
- **Merged export** - Full model with adapter merged
- **GGUF export** - Quantized models for Ollama/llama.cpp (q4_k_m, q8_0, etc.)
- **Ollama integration** - Auto-generate Modelfile and register models

#### Safety & Monitoring
- **GPU monitoring** - Temperature, VRAM, utilization tracking
- **Safety thresholds** - Configurable limits with auto-pause
- **Checkpoint management** - Automatic saving with configurable policies

#### Security
- **Path traversal protection** - Safe file operations
- **Secure model loading** - `weights_only=True` for torch.load
- **Input validation** - Sanitized paths and parameters
- **Gradio CVE fix** - Requires gradio>=5.6.0

#### Developer Experience
- **Modular installation** - Install only what you need (`[unsloth]`, `[ui]`, `[full]`)
- **Feature flags** - Runtime detection of optional dependencies
- **Lazy imports** - Fast startup, helpful error messages
- **Type hints** - Full type coverage
- **Pre-commit hooks** - Ruff, mypy, bandit

### Technical Details
- Python 3.10+ required
- PyTorch 2.0+ with CUDA support
- Tested on RTX 5080 (16GB VRAM) with Windows 11

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.2.0 | 2026-05-23 | GHSA-f65r-h4g3-3h9h auth-bypass closed (`backprop ui --auth/--share` now refuses to start until middleware lands); CI hard-gate restoration; truth-in-advertising test-count + version sweep |
| 1.1.1 | 2026-05-21 | CI hotfix — 4 broken action SHAs from the v1.1.0 SHA-pin sweep (no user-facing changes) |
| 1.1.0 | 2026-05-21 | Reflex UI, HF Hub push, resume-from-checkpoint, run history, model cards, W&B wiring (10-wave dogfood swarm) |
| 1.0.5 | 2026-04-15 | Release-binaries workflow re-cut after v1.0.4 Linux exclusion fix |
| 1.0.4 | 2026-04-14 | Linux binary <2GB (CPU torch swap), strip SIGPIPE fix |
| 1.0.3 | 2026-04-14 | Standalone PyInstaller binary workflow (Windows + Linux) |
| 1.0.2 | 2026-03-25 | CLI version fix, regression tests |
| 1.0.1 | 2026-02-27 | Ship Gate audit, verify.sh, proper exit codes |
| 1.0.0 | 2026-02-27 | Stable release - production-ready |
| 0.1.7 | 2026-02-27 | Codecov, quality scorecard, privacy section |
| 0.1.4 | 2026-02-22 | Bandit false positive fix |
| 0.1.3 | 2026-02-22 | Qwen2.5-3B preset, local dataset helper |
| 0.1.0 | 2026-01-19 | Initial release - SLAO, QLoRA, Windows support |

---

[Unreleased]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.0.5...v1.1.0
[1.0.5]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v0.1.7...v1.0.0
[0.1.7]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v0.1.4...v0.1.7
[0.1.4]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v0.1.0...v0.1.3
[0.1.0]: https://github.com/mcp-tool-shop-org/backpropagate/releases/tag/v0.1.0
