# backpropagate test suite

A guide for contributors — how to run the tests, where the fixtures live,
which markers gate what, and the cross-cutting doctrines that have been
earned the hard way. Audited up to v1.3 Wave 6b (TESTS-F-005).

> If this document is missing a section that would help you, that's a bug.
> Open an issue or amend the file in the same PR as your change.

---

## How to run the suite

The repo has one pytest config (`[tool.pytest.ini_options]` in
`pyproject.toml`) — there are no hidden ini files. All commands below are
run from the repo root.

### Fast inner loop (~3 min on stock laptop)

```bash
pytest -q -m "not slow"
```

Skips the `@pytest.mark.slow` suite (long-running integration tests like
SIGKILL-resume and full-chain mocks). This is the cadence most contributors
hit on every save.

### Full suite (~5-7 min)

```bash
pytest -q
```

Runs everything except `@pytest.mark.integration` (which gates GPU /
external-service / nightly-smoke tests). Match this against CI before
opening a PR.

### Targeted run

```bash
# A single file
pytest tests/test_auth_middleware.py -q

# A single class
pytest tests/test_integration.py::TestE2ESingleRunSmallModel -q

# A single test
pytest tests/test_cli.py::TestParser::test_train_command_basic -q

# Pattern match
pytest -k "auth_middleware" -q
```

### Parallel execution (CI)

```bash
BACKPROPAGATE_PYTEST_PARALLEL=1 pytest -q
```

Opt-in via the env var. The env-var gate is wired in `.github/workflows/ci.yml`
which translates `=1` / `=auto` → `-n auto --dist worksteal`, and integer
values to `-n N`. See `conftest.py` for the full xdist contract and the
audit notes that justify why no test in the suite is xdist-incompatible.

Local devs default to serial — easier `--pdb` / breakpoint debugging, less
stdout interleaving. Set the env var if you want CI-like timing locally.

### With coverage

```bash
pytest --cov=backpropagate --cov-report=html -q
```

Floor is `fail_under = 50` (single source of truth: `[tool.coverage.report]`
in `pyproject.toml`; ci.yml reads it via `tomllib` for lockstep).

---

## Markers

All markers are declared in `pyproject.toml` under
`[tool.pytest.ini_options].markers`. Adding a new marker without declaring
it there fires `--strict-markers` and breaks CI; declare-then-use.

| Marker | Purpose | When to apply |
|--------|---------|---------------|
| `@pytest.mark.slow` | Long-running test (> 5s) | SIGKILL waits, subprocess spawns, anything that sleeps |
| `@pytest.mark.integration` | Requires GPU / external service / opt-in | Real model downloads, ollama CLI, full e2e chains |
| `@pytest.mark.hypothesis` | Property-based / fuzzed test | Hypothesis-based tests for visibility in collection |
| `@pytest.mark.serial` | Cannot run under xdist | Reserved — no v1.3 tests need it; add when truly necessary |

CI uses `-m "not integration"` for the fast cycle and a separate workflow
(`nightly-train-smoke.yml`) for the integration suite. Slow tests are
included in CI by default — they pass under 60s in mocked mode.

> **No `gpu` marker** — the old `gpu` marker was deleted (declared but
> never applied to a single test; verified via `pytest --collect-only -m gpu`
> returning 0). Tests that genuinely require a GPU should use `integration`.

---

## Fixture overview

Top-level fixtures live in `tests/conftest.py`. The suite is structured so
that 90% of tests can compose pre-built fixtures rather than hand-rolling
setup.

### Mock infrastructure

| Fixture | Provides |
|---------|----------|
| `mock_torch_cuda` | Mocks `torch.cuda.is_available()` → False (CPU-only tests) |
| `mock_cuda_available` | Mocks CUDA-available + a 16GB GPU |
| `mock_settings` | Default `Settings` instance |
| `mock_trainer` / `mock_trainer_factory` | `MagicMock` trainer for unit tests |
| `tiny_model` | Mocked small model (`hidden_size=256, layers=2`) |
| `mock_gpu_status` / `mock_training_result` | Pre-built `GPUStatus` / `TrainingRun` shapes |
| `mock_peft_model` / `mock_tokenizer` | For export tests |

### Dataset fixtures

| Fixture | Provides |
|---------|----------|
| `sample_dataset` | 2-sample ChatML |
| `large_sample_dataset` | 100-sample ChatML |
| `sample_sharegpt_data` / `sample_alpaca_data` / `sample_openai_data` | One sample per format |
| `sample_jsonl_file` | Temp JSONL on disk (ShareGPT shape) |
| `training_dataset_file` | 100-sample ChatML JSONL on disk |

### Callback / event-handler fixtures

The trainer/multi-run/GPU-monitor surfaces all use the same callback
contract (`TrainingCallback` + per-monitor on_*). Spies + trackers live in
`tests/helpers/callbacks.py` and the fixtures wire them up:

| Fixture | Purpose |
|---------|---------|
| `mock_training_callback` | Returns `(callback, calls_dict)` for trainer assertions |
| `mock_multirun_callbacks` | Same for multi-run (threadsafe via lock) |
| `mock_gpu_monitor_callbacks` | Same for GPU monitor + threading.Event for waits |
| `callback_spy` / `callback_spy_factory` | Single-callback spy with detailed invocation tracking |
| `callback_tracker` | Multi-callback tracker for sequence assertions |
| `async_callback_collector` | Threaded collector with `expected_count` + `.wait()` |

### Path fixtures

| Fixture | Provides |
|---------|----------|
| `temp_dir` / `tmp_path` | Per-test temp dir (use either — `temp_dir` aliases `tmp_path`) |
| `checkpoint_dir` | `tmp_path / "checkpoints"` (pre-created) |
| `sample_gguf_path` | Stub GGUF file on disk |

### LoRA / SLAO fixtures

| Fixture | Provides |
|---------|----------|
| `sample_lora_state` | 4-key LoRA state dict (16/256/16/256) |
| `sample_lora_pair` | Pre-built base + new LoRA state pair (for merge tests) |
| `slao_config` / `slao_merger` | SLAO defaults / pre-initialised merger |
| `multi_run_config` / `multi_run_config_fast` | Multi-run config (full / fast variant) |

### ASGI / WebSocket helpers

For the Reflex auth middleware test surface:

```python
from tests.helpers.asgi import (
    basic_auth_header, make_asgi_client, malformed_auth_header,
    stub_asgi_http_app,
)
from tests.helpers.ws import (
    WSMessageRecorder, make_connect_receive, make_ws_scope,
)
```

These let auth tests run in-process (`httpx.ASGITransport`) without binding
a port. The WS helpers cover the load-bearing
**validation-before-`websocket.accept`** safety property — see
`tests/test_auth_middleware.py` for the contract under test and
`tests/test_auth_middleware_fuzz.py` for the property-based coverage.

---

## How to write a new test

### Folder layout

* `tests/test_<module>.py` — unit tests for `backpropagate/<module>.py`.
* `tests/test_<module>_extended.py` — extended coverage (added when the
  module has > 20 tests; keeps `test_<module>.py` legible).
* `tests/test_<feature_area>.py` — cross-cutting feature tests (e.g.
  `test_e2e_chain.py`, `test_wave6b_flags.py`) for changes spanning
  multiple modules.
* `tests/helpers/` — shared scaffolding (NOT test files; importable
  helpers used by tests).

### Naming convention

* Test files: `test_*.py`
* Test classes: `TestSomeFeature`
* Test functions: `test_<imperative_phrase>` — e.g.
  `test_train_command_basic`, `test_lora_config_r_bumped_to_256`.
* When pinning a specific finding from an audit, mention the ID in the
  docstring: `"""TESTS-F-001 — Full chain regression."""`.

### Mocking patterns

* External I/O (HuggingFace, ollama subprocess) — patch at the boundary,
  not deep inside the module. Patch `subprocess.run` for `register_with_ollama`,
  `trl.SFTTrainer` for `trainer.train`, etc.
* Hardware — use `mock_torch_cuda` to assert CPU-only paths.
* Pydantic settings — instantiate `Settings()` directly with kwargs;
  pydantic-settings is pure.
* Time-dependent tests — pin `time.time()` via `monkeypatch.setattr` or
  inject the `now=` argument where the helper exposes one.

### Assertion style

* Prefer **specific** assertions over `assert result` — the assertion
  failure message is the bug report for the next debugger.
* For boolean-shape assertions, pass `, "explanation of expected
  behaviour"` as the second arg so the failure says what went wrong.
* For status codes / enum values, use `==` with the named symbol, not the
  underlying int — e.g. `assert code == EXIT_USER_ERROR`, not
  `assert code == 1`.

### When to pin source-level invariants

Some safety properties (cookie HMAC uses `compare_digest`, no default
credentials anywhere) are stronger as source-level invariants than as
behavioural tests. See `test_auth_middleware.py::test_cookie_hmac_signature_verification_uses_constant_time_compare`
for the pattern: `inspect.getsource(...)` + assert against substring.

---

## Drift gate awareness

The repo has a doc-drift gate
(`scripts/check_doc_drift.py`, wired into `.github/workflows/doc-drift.yml`)
that fires on every PR. The gate checks:

1. New `BACKPROPAGATE_*` env vars read in `backpropagate/**/*.py` must
   appear in **both** `site/src/content/docs/handbook/env-vars.md` AND the
   `_enumerate_env_vars` catalogue in `backpropagate/cli.py`.
2. New `--flag` arguments registered via `argparse.add_argument` in
   `backpropagate/cli.py` must appear in
   `site/src/content/docs/handbook/cli-reference.md`.
3. Public state fields on `rx.State` subclasses in
   `backpropagate/ui_state.py` must have at least one consumer in
   `backpropagate/ui_app/`.
4. `tests/test_error_codes_catalog.py` still asserts every `code='...'`
   literal is in `ERROR_CODES`.

**For test authors:** if your test introduces a NEW env var (e.g. for a
feature flag the test exercises), you may need to update the
`scripts/doc-drift-allow.toml` grandfathered list — but only after
checking whether the env var should instead be documented in the handbook.
The standing rule: grandfathered drift is a doctrine-review action;
prefer to close the loop in the same PR.

Run the check locally before pushing:

```bash
python scripts/check_doc_drift.py
```

Exit 0 ⇒ no drift; exit 1 ⇒ at least one item missing co-located docs.

---

## Auth middleware test conventions

The auth surface is the highest-stakes part of the project (Wave 5
shipped GHSA-f65r-h4g3-3h9h CVSS 9.8 critical). The test cadence is:

### `tests/test_auth_middleware.py` — DESIGN_BRIEF-numbered tests

Sixteen tests numbered 1-16 against the DESIGN_BRIEF.md contract. Each is
mapped to a specific brief requirement — see the comments at the top of
each test. Adding a new test for the same surface ⇒ add an unnumbered
trailing test, do NOT re-number.

### `tests/test_auth_middleware_fuzz.py` — Hypothesis property tests

Property-based coverage (TESTS-F-002, v1.3 Wave 6b) — adds bounded
fuzz against the fixed-corpus tests. Properties under test:

1. No exception escape on arbitrary Authorization / Cookie / Host /
   Origin bytes
2. No silent pass-through with garbage credentials
3. No DoS-vector via WebSocket accept (PRE-accept close invariant)
4. Cookie HMAC unforgeable under signature tampering
5. Host header normalisation safety

`@settings(max_examples=100)` for CI tractability — the full file runs
in ~10s on stock CI.

### INTACT contract (load-bearing)

The headline auth-test baseline is **23 passed / 3 skipped** (23/3). This
ratio is pinned in dispatch briefs across the swarm; any test author who
introduces an auth-suite skip without a corresponding pass-conversion
must call it out in their report. The 3 skipped tests are:

* `test_share_without_auth_refuses_to_start` — pinned at CLI layer in
  `test_cli_extended.py::TestCmdUI::test_cmd_ui_share_without_auth_still_refuses`
* `test_host_non_loopback_without_auth_refuses_to_start` — pinned at CLI
  layer in `tests/test_host_gate.py`
* (one more — see `test_auth_middleware.py` for the canonical list)

If you add a test that flips one of those to a passing assertion, update
this README and the dispatch-brief baseline together.

---

## v1.3 Wave 6b additions (TESTS-F-001..F-006)

The Wave 6b feature pass shipped six new test surfaces. Cross-reference:

| Finding | Location | What it pins |
|---------|----------|--------------|
| TESTS-F-001 | `tests/test_e2e_chain.py::TestE2EFullChain` | Real train → export → register chain (mocked deps) |
| TESTS-F-002 | `tests/test_auth_middleware_fuzz.py` | Hypothesis fuzz across 5 attack-surface properties |
| TESTS-F-003 | `tests/conftest.py` (xdist env-var gate) | CI-side parallel-mode opt-in |
| TESTS-F-004 | `scripts/run_mutmut.sh` + `.github/workflows/mutmut.yml` | Mutation-survival baseline |
| TESTS-F-005 | `tests/README.md` (this file) | Contributor onboarding |
| TESTS-F-006 | `tests/test_e2e_chain.py::TestResumeAfterSigkill` | Resume after ungraceful SIGKILL termination |

Plus `tests/test_wave6b_flags.py` for the 5 new CLI flags + 3 new
subcommands the BRIDGE agent added in lockstep with the BACKEND agent's
LoRA-defaults / DoRA / packing / model-preset additions.

---

## Doctrines earned in tests/

The hardest-won lessons. Re-reading these before touching the suite saves
hours.

* **`[[no-banner-documenting-no-op]]`** — a skip marker that says "TODO:
  Reflex equivalents in Phase 3" without a forward-pointer is a lie that
  accumulates. Either the work has landed (delete the skip) or hasn't
  (link to the dispatch brief / issue). Permanent skips disguised as
  TODOs are the worst form of test debt.
* **`[[within-swarm-doc-lie-drift-detection]]`** — every wave that adds
  new surface (env var / CLI flag / state field) risks shipping a doc-lie
  alongside. The drift gate fires on the PR; close the loop in the same
  commit, don't defer.
* **Stash discipline** — agents in the dogfood swarm are explicitly
  forbidden from `git stash`/`reset`/`checkout`/`commit`. A single
  mid-wave stash silently swept 11 source files across 3 CRITICAL fixes
  on repo-knowledge in 2026-05-20. The coordinator commits between waves;
  agents never touch git state.
* **CRLF on Windows** — agents that use `Write` (not `Edit`) on an
  existing LF file produce 200+ line diffs of line-ending pollution.
  Always use `Edit` for amends; normalise with a one-line node script if
  it happens.
* **Auth INTACT (23/3)** — the auth test baseline is load-bearing.
  Any test author who flips it must explain why in their report.
