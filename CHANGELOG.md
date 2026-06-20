# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.6.0] - 2026-06-20

### Added

- **SimPO — reference-free preference tuning.** `--method simpo` / `Trainer(method="simpo")` trains on `{prompt, chosen, rejected}` preference data via TRL's `CPOTrainer` with `loss_type="simpo"` (`cpo_alpha=0`) — length-normalized reward + target margin, no reference model, same single-GPU LoRA envelope as ORPO/SFT (SimPO: Meng et al. 2024, [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)). Defaults `--simpo-beta 2.0` / `--simpo-gamma 1.0`; the LR auto-lowers to `1e-6` (clamped if set ≥ `1e-5` — high LR degrades SimPO). `mode="lora"` only.
- **KTO — preference tuning on unpaired binary feedback.** `--method kto` / `Trainer(method="kto")` trains on an **unpaired** `{prompt, completion, label}` dataset (per-example thumbs-up/down) via TRL's `KTOTrainer` (KTO: Ethayarajh et al. 2024, [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)) — the data shape ORPO/SimPO/DPO can't consume. New `DatasetFormat.KTO` + `DatasetLoader.to_kto_dataset()` + detection/validation. KTO uses the frozen LoRA base as its reference (no second model loaded — 16 GB envelope preserved); `--kto-beta 0.1`; desirable/undesirable weights auto-balance from the dataset's label counts toward a 1:1–4:3 effective ratio. `mode="lora"` only. Both new methods ship with non-mocked GPU smokes (real CPOTrainer/KTOTrainer trains, finite loss, loadable + mergeable adapter).
- **Eval task-metrics + conjunction merge-gate.** `backprop eval --metric <name> --references <jsonl> [--gate-metric <name>]` adds deterministic, judge-free task metrics — `normalized_exact_match`, `token_f1` (SQuAD-style), `contains`, `regex`, `pass_rate` — to `EvalResult` (`task_metrics`, `eval_n`, bootstrap `metric_ci`). No LLM judge. The eval-gate is now a **conjunction**: a merge/checkpoint is accepted only if held-out loss does **not** regress AND every gated task-metric does not regress beyond a noise band (delta vs the bootstrap CI, not zero); it warns when the held-out set is underpowered (`eval_n < 100`). Closes the gap where a merge could improve loss while regressing actual task behavior and still pass. (Grounded in HELM [arXiv:2211.09110], SQuAD EM/F1 [arXiv:1606.05250], and the AlpacaEval null-model result [arXiv:2410.07137].)
- **`backprop generate <adapter_path> "<prompt>"`** — ad-hoc "did my fine-tune work?" generation against a trained adapter path (no recorded run, no GGUF export). Base model inferred from `adapter_config.json`; `--base`, `--max-new-tokens`, `-n/--num`, `--temperature`.
- **`backprop data split <jsonl>`** — deterministic train / held-out split (`--heldout-ratio`, `--seed`, `--out-train`, `--out-heldout`), completing the held-out/contamination workflow that `data report --against` and `eval --heldout` assume.
- **FP8 compute path is now verified.** The v1.5 experimental FP8 path (`--fp8`, torchao float8 on Blackwell/Hopper sm_90+) is dogfood-verified end-to-end on an RTX 5090 (Blackwell sm_120) — trains, the `Float8Linear` base / bf16-adapter exclusion holds, and the result merges. Status promoted experimental → verified.

### Changed

- **README capability boundary updated:** ORPO (v1.5) + **SimPO + KTO (v1.6)** are the shipped reference-free preference methods; "SimPO/KTO planned" is retired. No online RL (PPO/GRPO/RLVR) — use TRL / LLaMA-Factory.
- Conservative dependency floor-raises (`transformers>=4.46`, `datasets>=2.19`, `trl>=0.18`, `peft>=0.13`, `accelerate>=0.34`); upper caps unchanged. The `trl` ORPO/CPO/KTO imports are guarded (structured error + working-range remedy) against the experimental-namespace relocation.

### Removed

- **BREAKING — Gradio fully purged.** Gradio was replaced by the Reflex web UI in v1.2.0, but ~98 dead `gr.*` references survived behind a shim in `ui_security.py`. Removed the entire gradio apparatus (`_GradioShim`, the dead `safe_ui_handler` / `raise_ui_*` helpers) and the long-deprecated **gradio-named aliases** (`safe_gradio_handler`, `RequestContext.from_gradio_request`, `get_gradio_csp`, `DEFAULT_GRADIO_CSP`). `validate_numeric_input` / `validate_string_input` now raise `UserInputError`. All framework-neutral security helpers are unchanged.
- **BREAKING — `TRAINING_PRESETS` alias removed** (deprecated since v1.4). Use `MULTI_RUN_PRESETS`.
  Both removals were `DeprecationWarning`-flagged since v1.4 and had zero in-package consumers.

### Fixed

- **Eval-gated SLAO merge made functional.** The v1.5 eval-gated merge was non-functional outside its mocks: `_evaluate_accumulator` wrote a LoRA adapter with no `adapter_config.json` (so `PeftModel.from_pretrained` raised and aborted the whole multi-run session on the first merge), and the documented default (`eval_heldout_path=None`) hit an unimplemented "reuse last-10% split" hook. Now writes a valid `adapter_config.json` and derives the last-decile held-out set in-process; de-mocked with a real PeftModel round-trip test.
- **Windows-console crash guard.** The per-run Run-ID banner used an em-dash with no stdout/stderr UTF-8 reconfiguration, so on a legacy cp437/cp850 Windows console the first status line could raise `UnicodeEncodeError` before training. `main()` now reconfigures the streams to UTF-8 (`errors="replace"`), fixing the crash and pipe-mojibake.
- Dataset dedup now uses a stable SHA-1 instead of the salted builtin `hash()` (exact + reproducible across processes); CLI failures surface the stable error `[code]` and the trainer `run_id`; tz-aware timestamps no longer crash resume auto-detect; `backprop data report` converts each sample to ChatML once (was 3×); UI HF-token-file path accepts the documented `~/.config` location; auth-error responses now carry the hardened security headers; numerous doc/help accuracy fixes. README states Python 3.10 is supported through at least v1.6.

### Internal

- A pre-existing pytest-xdist test-isolation bug (a test popped `backpropagate.mlx_backend` from `sys.modules` without restoring it, desyncing other mlx tests' mocks under work-stealing) was fixed. The doc-drift gate was hardened to reconfigure stdout to UTF-8. 3139 → 3437 tests (CI lane); full suite 3467 passed.

## [1.5.0] - 2026-05-31

### Added

- **ORPO — reference-free preference tuning.** `--method orpo` / `Trainer(method="orpo")` trains on `{prompt, chosen, rejected}` (or `{chosen, rejected}`) preference data via TRL's `ORPOTrainer` — single-stage, no reference model, so it fits the same single-GPU LoRA envelope as SFT. `mode="lora"` only in v1.5; the default learning rate auto-lowers to 8e-6; preference formats are auto-detected alongside the SFT formats. Ships a cross-version shim so it trains on transformers 4.x **and** 5.x (trl 0.24's `ORPOTrainer` writes `model.warnings_issued`, which transformers 5.x removed).
- **`backprop data report <jsonl>` — dataset-quality report.** Exact + near-duplicate clusters (pure-stdlib MinHash/LSH), format distribution, token-length histogram, length outliers, empty / no-assistant turns, and train/test contamination (`--against`). Advisory by default; `--fail-on-dups` / `--fail-on-contamination` / `--max-outlier-rate` / `--strict` turn signals into hard gates (exit 65). `--json` for CI.
- **`backprop eval <run-id>` — post-train evaluation.** Held-out loss + perplexity + N sample generations against a trained adapter; `--vs` diffs two runs; `--gate-against` rejects a held-out-loss regression (the seam the eval-gated merge consumes).
- **FP8 compute path (experimental).** `--fp8` routes base-weight GEMMs through torchao float8 on Blackwell/Hopper (sm_90+) — base in float8 (~1.4× throughput, ~60% less base memory), LoRA adapter + optimizer stay bf16, result still mergeable. New `[fp8]` extra. Gated to `mode="lora"` + `method="sft"`; degrades to bf16 with a warning on unsupported hardware (never a crash).
- **Multi-strategy, eval-gated merge framework** for multi-run SLAO. `--merge-strategy {qiao_mahdavi (default), linear, ties, dare}`; a drift gate (`--drift-gate`) decides merge-vs-branch via LoRA-B cosine similarity; an eval gate (`--eval-gate`) rejects a merge that regresses held-out loss and restores the pre-merge accumulator. Default (`qiao_mahdavi`, gates off) is byte-identical to v1.4.
- **Adapter-native export.** `export --format ollama-adapter` registers a LoRA adapter on a base via an Ollama `FROM`+`ADAPTER` Modelfile (safetensors, no merge); `backprop ollama shelf <base>` lists the adapters registered on a base.
- **rsLoRA.** `--use-rslora` / `use_rslora=True` enables rank-stabilized LoRA (α/√r scaling) — zero inference cost, mergeable, benefit grows with rank.
- **Reasoning-trace SFT.** `--reasoning-trace` keeps the `<think>…</think>` chain-of-thought in the training target (plain text — still mergeable/exportable, no embedding resize), filters empty / over-long / unbalanced traces, and raises the default `max_seq_length` to 8192. `backprop data report` now surfaces a `<think>` rate + a trace-length histogram.
- **MLX / Apple-Silicon backend (experimental).** `--backend {auto,cuda,mlx}` + a standalone `[mlx]` extra route LoRA SFT through `mlx_lm` on Apple Silicon; CUDA stays canonical, `auto` picks by hardware. Built + unit-tested; **pending dogfood verification on real Apple Silicon** (mlx-lm is Apple-only and cannot run on the CUDA dev rig).
- New error codes: `RUNTIME_FP8_UNSUPPORTED`, `DEP_MLX_UNAVAILABLE`, `INPUT_EVAL_RUN_NOT_FOUND`, `INPUT_EVAL_HELDOUT_UNRESOLVED`, `RUNTIME_EVAL_FAILED`, `RUNTIME_EVAL_GATE_REGRESSED`, `INPUT_DATASET_REPORT_THRESHOLD`.

### Changed

- **README capability boundary reframed:** "single-stage SFT + reference-free preference tuning (ORPO; SimPO/KTO planned); no online RL (PPO/GRPO/RLVR) — for those use TRL or LLaMA-Factory."
- Reflex Web UI: `recovery_banner` / `error_callout` / `status_pill` / runs / run-detail render-compile fixes, plus a new CI dry-run compile-smoke gate (`app._compile(dry_run=True)`) so a render-time `Var` error can never ship green again.

### Fixed

- **Composed re-audit remediation (v1.5 feature seams).** `backprop train --help` and `multi-run --help` no longer crash (an unescaped `%` in `--fp8` help → argparse `TypeError`; non-cp1252 glyphs in three multi-run flags → Windows-console `UnicodeEncodeError`); a new test renders `format_help()` for every subparser. SFT on a preference dataset now trains on the `chosen` response instead of silently producing a 0-row dataset (and `backprop eval` works on such runs). The model-card reproduce command now reflects `--method` / `--fp8` / `--use-rslora` / `--reasoning-trace` / `--backend` so it actually reproduces the run. `--fp8` on unsupported hardware no longer strips the operator's default 4-bit (OOM risk); `orpo_beta` is validated `> 0`; `reasoning_trace` is an honest no-op under ORPO and active on the MLX rail; the eval gate restores the accumulator on a mid-gate exception; FP8-trained merges cast to bf16 before save (so the GGUF disk pre-check holds); the `_count_tokens_approx` CJK caveat direction was corrected.
- **`mode="full"` now performs genuine full fine-tuning instead of silently running QLoRA.** v1.4.0 shipped `mode="full"` advertising full fine-tuning (every weight updated) for ≤3B models on a 16GB GPU, but BOTH model loaders (`_load_with_transformers`, `_load_with_unsloth`) applied 4-bit quantization + a LoRA adapter **unconditionally** — neither had a `self.mode` branch. `_build_sft_config(mode="full")` correctly dropped the adapter and switched to paged 8-bit AdamW + gradient checkpointing, but it was handed a model that was already 4-bit + `get_peft_model`'d, so TRL trained the LoRA adapter on a frozen 4-bit base. Net effect: `mode="full"` *was* QLoRA — the opposite of its documented contract — and the saved artifact was a LoRA adapter, not full weights. The loaders are now mode-aware: `mode="full"` loads full-precision weights (bf16 on Ampere+, fp16 on pre-Ampere, fp32 on CPU) with **no** `BitsAndBytesConfig` and **no** `get_peft_model` / `prepare_model_for_kbit_training` (transformers path), and `full_finetuning=True` + `load_in_4bit=False` with no adapter (Unsloth path); the plain full model goes straight to `SFTTrainer`. `mode="lora"` (the default) is byte-identical to before. This also fixes a crash where `mode="full"`'s 4-bit load + `device_map="auto"` had no CUDA guard (bitsandbytes 4-bit is CUDA-only) — full-precision loads now omit `device_map` on CPU runners. Verified end-to-end on an RTX 5090: a `mode="full"` train produces a 100%-trainable full bf16 model and saves `model.safetensors` (full weights), no `adapter_config.json`. Regression tests (`tests/test_wave6b_features.py::TestModeFullLoadsGenuineFullModel`) assert both loaders produce a non-PEFT, non-4-bit model and FAIL on the pre-fix loaders.
- **`mode="full"` parameter ceiling raised 3B → 4B so the marketed "3B" presets actually work.** The v1.4.0 ceiling was `_FULL_FT_PARAM_CEILING_BILLIONS = 3.0`, but the load-time gate checks the model's *authoritative* `num_parameters()` — and every marketed "3B" model is really 3.08–3.24B (SmolLM3-3B 3.08B, Qwen2.5-3B 3.09B, Llama-3.2-3B 3.21B). So `Trainer(model="smollm3-3b", mode="full").load_model()` raised `RUNTIME_FULL_FT_MODEL_TOO_LARGE` at load — `mode="full"` was unusable for its headline target models (only ≤~2B passed). The ceiling is now 4.0B: the genuine ~3B presets fit a 16GB card, and the 3.8–4B class (Phi-4-mini-3.8B, Qwen-3.5-4B) is also admitted — those need a 24GB+ card for the VRAM (weights + gradients alone approach 16GB), documented in the README envelope table. To match, the `RUNTIME_FULL_FT_MODEL_TOO_LARGE` recovery hint no longer mis-directs operators to presets above the ceiling, and a broken `handbook/full-finetuning.md` link in the hint was corrected to `handbook/full-fine-tuning.md`. Envelope confirmed empirically on an RTX 5090 capped to 16GB: a SmolLM3-3B (3.08B) `mode="full"` step peaks at ~13.6GB of VRAM (100% of parameters trainable, not a PEFT adapter) — comfortably inside the 16GB consumer-card budget.

## [1.4.0] - 2026-05-25

### Fixed

- **Multi-run validation no longer leaks eval-mode into the next run after an OOM.** v1.3.x left the model stuck in `eval()` after a `CUDA out of memory` (or any exception escaping the validation loop) in `MultiRunTrainer._compute_validation_loss`. The very next training pass silently produced no gradient updates — operators saw "training completed" but the model didn't learn anything new. v1.4 wraps the validation body in `try ... finally: model.train()` so the train-mode invariant is restored even on exception. Operators hit by this on v1.3.x: re-run the affected multi-run from a clean checkpoint on v1.4. (BACKEND-B-003)
- **`Trainer.save()` warns instead of silently writing untrained weights.** v1.3.x let an operator who did `Trainer(...).load_model()` then `.save()` write **init-weight LoRA adapters** to disk — same filename / `adapter_config.json` shape as a real fine-tune, no signal it was untrained. v1.4 tracks an internal `_has_trained` flag (set by `train()` on success) and emits one structured WARN log line at `save()` when the flag is False, naming the file path + the missing `train()` call. The save still completes (no gate / no exception) so existing tooling that pre-creates output dirs doesn't break, but operators get a visible "this adapter was never trained" cue. (BACKEND-B-004)
- **`backprop replay` + `train` + `multi-run` `--override` flags now thread through under MagicMock-patched test stubs.** v1.3 added an introspection filter so the CLI wouldn't pass a kwarg that the constructor didn't declare. The filter looked only at named parameters and silently dropped every override when the constructor used `**kwargs` (which the real `Trainer.__init__` does not, but `MagicMock(spec=None)` does). Under `unittest.mock.MagicMock`-patched Trainer/MultiRunTrainer tests, `--override lora_r=32` would silently no-op. v1.4 extends the filter to detect `VAR_KEYWORD` and degrade to pass-through. Bare CLI calls behaved correctly; the bug only surfaced under MagicMock patching. (BRIDGE-B-001 + sibling fixes in `cmd_train` / `cmd_multi_run` via [[grep-all-instances-when-fixing-pattern]] doctrine)
- **`/runs/<run_id>` page no longer strands the "Run deleted" success message behind a not-found error chrome.** v1.3.x set both `not_found=True` AND `action_result="Run deleted"` after a successful delete; the template rendered the `_not_found()` chrome because that branch fired first, hiding the success copy. v1.4 adds a `was_deleted` state field and a dedicated success chrome with a "Back to runs list" navigation button, so operators get a clear "yes, that worked, click here for what's next" cue rather than a confusing 404. (FRONTEND-B-001)
- **CI failure-observability paths (nightly smoke + post-publish smoke + mutmut baseline) now self-heal on label-deleted repos.** `gh issue create --label X` and `gh label create` silently no-op when the named label doesn't exist on the repo. A freshly-cloned fork, an accidentally-deleted label, or a renamed label would silently break the failure-issue surface. v1.4 each issue-creation workflow runs `gh label create <name> --force` idempotently just before `gh issue create`, so the surface stays self-healing. Labels bootstrapped: `ci`, `nightly-smoke`, `post-publish-smoke`, `mutmut-baseline`. (CIDOCS-B-001)
- **`mutmut` baseline file is now persisted across CI runs.** v1.3.x's `.github/workflows/mutmut.yml` wrote the baseline file inside the runner sandbox and discarded it on job exit — the baseline was reset on every run and the "mutants killed since last baseline" delta was unbounded. v1.4 opens a small PR (branch `ci/mutmut-baseline-<timestamp>`) when the runner-written baseline differs from `origin/main`. Auto-merge intentionally avoided per advisor decision — human review preserved on test-quality data. (CIDOCS-B-002)
- **Handbook quality-preset defaults are aligned across all 5 pages.** While fixing `training.md:51 lora_r | 16` drift (CIDOCS-B-008), the [[grep-all-instances-when-fixing-pattern]] doctrine surfaced 4 sibling drift sites across `beginners.md`, `getting-started.md`, `env-vars.md`, `reference.md` — all showing the v1.2.x rank-16 / alpha-32 footprint instead of the v1.3 rank-256 / alpha-512 default. All five pages now agree with `LoRAConfig.r=256` + `LoRAConfig.lora_alpha=512` ground truth. (CIDOCS-B-008 + 4 grep-doctrine siblings)
- **README hero example no longer references non-existent `backprop ollama register` subcommand.** The opening 3-line copy-pasteable bash block (the FIRST runnable snippet operators read) was advertising a subcommand that does not exist in `backpropagate/cli.py` — operators hit `argparse error: argument command: invalid choice: 'ollama'` on the first invocation. Replaced with the working `backprop export ... --ollama --ollama-name my-model` form already documented at README line 270. Translated READMEs (`README.{es,fr,hi,it,ja,pt-BR,zh}.md`) defer to Phase 10 polyglot-mcp re-translation. (CIDOCS-A-001)
- **PyPI metadata propagation: hatchling pinned to `>=1.27`.** v1.3.0's PyPI page rendered `Author: None` and `License: None` because the `license = "MIT"` PEP 639 SPDX-expression shape requires `hatchling>=1.27`. Older hatchling silently downgraded to Metadata-Version 2.1 and emitted an empty License-Expression, which PyPI then renders as `None`. The pin closes this; verify post-build with `python -m build && twine check dist/* && tar -xOf dist/backpropagate-*.tar.gz '*/PKG-INFO' | grep -E 'License\|Author'`. (CIDOCS-A-002)
- **`publish.yml` now fires on the `release.yml`-completed chain.** The prior `release: types: [published]` trigger never fired when the release was created by `release.yml` using `GITHUB_TOKEN` (well-documented GH platform behavior to prevent workflow loops). v1.3.0 required manual `gh workflow run publish.yml --ref v1.3.0`. Added `workflow_run: workflows: [Release] types: [completed]` trigger gated on `github.event.workflow_run.conclusion == 'success'` (mirrors the pattern `post-publish-smoke.yml` already uses). Manual `release: published` + `workflow_dispatch` retained for hand-cut releases. (CIDOCS-A-003)
- **`llms.txt` env-var name corrected: `BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN` (was `..._REQ_PER_MIN`).** LLM agents consuming `llms.txt` to drive backpropagate were setting an env var that does not exist; the rate-limit cap silently reverted to default. The actual var is `BACKPROPAGATE_UI_RATE_LIMIT_HTTP_PER_MIN` (defined in `backpropagate/ui_app/middleware/rate_limit.py:58`). Also added the WebSocket cap (`BACKPROPAGATE_UI_RATE_LIMIT_WS_PER_MIN`, default 10). (CIDOCS-A-006)
- **Handbook `cli-reference.md --lora-r` default corrected: `256` (was stale `16`).** v1.3 shipped the quality-preset rank-256 default in `cli.py:4097` and in `LoRAConfig.r` (`config.py:268`); the handbook table still showed the v1.2.x footprint rank. (CIDOCS-A-004)
- **Handbook `env-vars.md BACKPROPAGATE_LORA__R` default corrected: `256` (was stale `16`).** Mirrors the `--lora-r` argparse default and the `LoRAConfig.r` Pydantic default. All three surfaces (env-vars, cli-reference, README) now agree. (CIDOCS-A-005)
- **`error-codes.md INPUT_AUTH_REQUIRED` row renamed to `RUNTIME_UI_AUTH_NOT_ENFORCED` with corrected fix advice.** The prior row advised `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false` opt-out — that env var is a no-op under the v1.1+ Reflex UI (held only for forward-compat with the Gradio era). Operators trying to use it would silently no-op. New fix advice: pass `--auth user:password` or use SSH port-forwarding (see `handbook/security.md`). The actual error code raised in v1.2.0+ is `RUNTIME_UI_AUTH_NOT_ENFORCED`. (CIDOCS-A-007)

### Changed

- **GitHub Pages deploy split into dedicated `pages-deploy.yml` workflow.** The prior `build-site` + `deploy-site` jobs lived in `ci.yml` and combined the workflow-level `paths:` filter with a job-level `if:` gate, creating a silent chain-reaction failure: pushes touching non-site files would not run `build-site`, so `deploy-site` (which depends on it) would skip. Site drift between the repo and deployed pages could persist for weeks unnoticed. The dedicated workflow has `paths: [site/**]` at the workflow level (cleaner gating) plus `workflow_dispatch` for manual self-heal redeploys. (CIDOCS-A-008)
- **`doc-drift.yml` actions SHA-pinned.** The prior `actions/checkout@v4` + `actions/setup-python@v6` references bypassed the v1.1.0-era "every third-party GitHub Action SHA-pinned" convention. Now matches the canonical pins used in `ci.yml` and `publish.yml`. (CIDOCS-A-009)
- **AQLM 2-bit experimental opt-in (`quant_method="aqlm"`) deferred from v1.4 to v1.5.** The `aqlm` library state was verified shippable during Wave 5 (1.1.7 on PyPI 2025-04-16; PEFT-documented AQLM+LoRA integration), but the Wave 6b implementation budget prioritized full fine-tuning support for ≤3B models (`mode="full"`) per V1_4_BRIEF P0 ordering. v1.5 will land AQLM + LoRA Mixtral-8x7B support; see V1_5_BRIEF when posted. The 16GB capability envelope row in README.md was updated to reflect the v1.5 status. Translated READMEs (`README.{es,fr,hi,it,ja,pt-BR,zh}.md`) defer to Phase 10 polyglot-mcp re-translation. (BACKEND-F-001)

### Added

- **`mode="full"` for full fine-tuning of ≤3B models on consumer 16GB GPUs.** v1.3 made LoRA the default + ranked the rank-256 quality preset against full FT. v1.4 lands the opposite case: operators who genuinely want every weight updated (rather than a low-rank adapter) can now pass `Trainer(..., mode="full")` (or `--mode=full` on `backprop train` / `backprop multi-run`). A hard gate refuses the mode for models > 3B parameters with `RUNTIME_FULL_FT_MODEL_TOO_LARGE`, naming `mode="lora"` (the default) + the three sub-3B presets that DO work (Phi-4-mini-3.8B / Qwen-3.5-4B / SmolLM3-3B) as the recovery options. The gate fires at `Trainer.__init__` (preset-table lookup) AND a second time at `load_model()` (authoritative `model.num_parameters()` check) so silent fall-through can't smuggle a 7B into the full-FT codepath. The mode='full' SFTConfig assembly is folded into the same `_build_sft_config` helper as mode='lora' (no fork): `gradient_checkpointing=True` (sqrt(L) activation memory), `paged_adamw_8bit` optimizer (consumer-card memory ceiling), 10× lower learning rate by default. Operators reading the README's "what backpropagate is NOT for" anti-pitch section saw the v1.4 promise that landed here. The Biderman 2024 + Thinking Machines 2025 quality math (LoRA matches full FT on most post-training tasks at 67% of the compute) is still the recommendation for most operators — `mode="full"` exists for the cases where the operator has measured a quality gap and decided to spend the extra compute. (BACKEND-F-008)
- **`backprop ollama register|list|rm` triad — dedicated Ollama-workflow subparser.** v1.3.x's README hero example was advertising `backprop ollama register` but the runtime had no such subcommand — CIDOCS-A-001 fixed the README to point at the working `backprop export --ollama --ollama-name <name>` form. v1.4 closes the loop the other way: the missing subcommand is now real, modeled as a nested subparser group (`backprop ollama <action>`) to match the upstream `ollama` CLI's operator mental model 1:1. New surface: `backprop ollama register <path> [--name <name>] [--modelfile <path>]` (register a previously-exported GGUF or LoRA adapter with the local Ollama daemon, no re-export), `backprop ollama list` (enumerate currently-registered model names), `backprop ollama rm <name>` (unregister a model — pass-through to `ollama rm` via the daemon HTTP API). Architectural deviation from backpropagate's flat-subparser convention is intentional: operators who already know `ollama create` / `ollama list` / `ollama rm` get the same grammar one prefix deeper. The existing `backprop export --ollama --ollama-name` one-shot path stays untouched as the canonical "I just trained, register in one command" surface; the new triad is for the "I already exported earlier, just register" case. Closes the operator-trap loop opened in CIDOCS-A-001. (BRIDGE-F-001)
- **`Trainer.estimate_vram()` — pre-flight VRAM estimator.** Today operators learn a config is too big at FIRST OOM, after pulling the model + dataset + running steps before the failure fires. v1.4 lands `Trainer.estimate_vram(mode=..., lora_r=..., batch_size=..., max_seq_length=..., ...)` returning a structured `VRAMEstimate` carrying `total_gb` + per-consumer breakdown (`model_weights_gb` / `lora_adapter_gb` / `optimizer_state_gb` / `activations_gb` / `kv_cache_gb` / `overhead_gb`) + reproducibility inputs. Math is back-of-envelope (15% overhead margin) but accurate within ~10-20% of empirical peak for the v1.3 canonical 7B QLoRA configs. The same math feeds `backprop estimate-vram` (which was already on the CLI as BRIDGE-F-008's tier-table helper; the new Python surface complements it with per-config detail). Sample usage: `Trainer("Qwen/Qwen2.5-7B-Instruct").estimate_vram(mode="lora", lora_r=256, batch_size=2, max_seq_length=2048)` → `VRAMEstimate(total_gb=13.4, fits_on_card(16.0)=True, ...)`. (BACKEND-F-002)
- **MultiRunTrainer callback parity restored — 4 callbacks now actually fire.** v1.3.x exposed `on_step`, `on_run_start`, `on_run_complete`, `on_gpu_status` on MultiRunTrainer but `MultiRunTrainer._execute_run` never invoked them — they were structurally dead. Operators wiring a `TrainingCallback` into a multi-run got a successful run with zero callback events; debugging surfaces (loss tracking, GPU-status logging, progress bars) silently no-op'd. The Wave 5 BACKEND-F-003 audit surfaced the regression. v1.4 threads the four callbacks through `_execute_run` (matches the single-run Trainer surface): `on_run_start(run_index)` fires before each inner Trainer.train(), `on_step(step, loss)` fires per training step, `on_gpu_status(snapshot)` fires from the GPU monitor on each poll, `on_run_complete(run_index, result)` fires after each inner run lands. Tests pin all four fire-counts against a known 5-run workload. Pre-fix consumers see additional events suddenly delivered — fully additive, no behavior change for operators who didn't wire callbacks. (BACKEND-F-003)
- **CheckpointManager atomic-write parity with RunHistoryManager.** v1.3.x's `CheckpointManager._save_manifest` wrote `manifest.json` via plain `open + write`, while `RunHistoryManager._save_history` already used `safe_write_json_with_lock` (atomic-rename + `filelock`). On cross-host NFS / SMB / Lustre filesystems, two concurrent processes calling `Trainer.save()` could race on the manifest write — last writer wins, prior writer's checkpoint manifest is silently clobbered, the on-disk lineage becomes incoherent. v1.4 aligns `_save_manifest` with the run-history pattern: write to `manifest.json.tmp` under `filelock`, fsync, rename atomically. Same `BACKPROPAGATE_FILELOCK_*` env knobs govern lock timeout. Pre-fix consumers on local POSIX filesystems were never bit (single-writer, lock-free was fine); the fix is load-bearing for the multi-process training-on-shared-storage operator pattern. (BACKEND-F-004)
- **CLI-wide `--log-level` / `--log-format` / `--log-file` flags on the root parser.** v1.3.x routed all logging configuration through env vars (`BACKPROPAGATE_LOG_LEVEL` / `BACKPROPAGATE_LOG_JSON` / `BACKPROPAGATE_LOG_FILE`). Operators wanting one-off log tuning for a single invocation had to either edit `.env` or prefix every command with `BACKPROPAGATE_LOG_LEVEL=DEBUG`. v1.4 adds root-parser flags `--log-level=DEBUG|INFO|WARNING|ERROR` (default INFO), `--log-format=json|console` (default console), `--log-file=<path>` (default unset — stderr only). CLI flag wins when both flag and env var are set, matching the v1.3 pattern for `--verbose` vs `BACKPROPAGATE_DEBUG`. Each flag binds to the same `configure_logging(...)` call site used by the env-var path so the structlog wiring stays in one place. (BRIDGE-F-002)
- **`--json` output on `backprop validate` and `backprop replay`.** v1.3 added schema_version-tagged JSON output to `backprop runs`, `backprop show-run`, `backprop diff-runs`, `backprop export-runs`, and `backprop estimate-vram`. v1.4 extends the same shape to `backprop validate <dataset>` (emits the `dataset`, `format_detected`, `total_samples`, `errors[]` shape under `schema_version` so CI consumers can grep without parsing the human-readable banner) and `backprop replay <run-id>` (emits the new `run_id` + the resolved config the replay used + each override that was applied). Both subcommands keep the human-readable surface as the default; `--json` is the explicit opt-in. (BRIDGE-F-007)
- **`backprop info --json` now exposes a `logging` block.** The existing `--json` output covered Python / PyTorch / CUDA / GPU + feature flags but never named the active logging config — operators triaging "why aren't my logs being JSON-formatted" had to grep env vars manually. v1.4 adds `logging: {level, format, file, has_handler}` to the `info --json` payload, surfacing what `configure_logging(...)` actually wired up for this process. Pairs with the new root-parser `--log-*` flags above so an operator can run `backprop --log-format=json info --json | jq .logging` and see exactly what state the next subcommand will inherit. (BRIDGE-F-010)
- **Root `--help` epilog rewritten to group 15 subcommands by workflow.** v1.3.x's epilog listed five example commands and the exit-code table but never enumerated every subcommand — operators reading `backprop --help` cold had to scroll through argparse's auto-generated subcommand list at the top to find what existed. v1.4 rewrites the epilog with workflow-grouped headings: TRAINING (train / multi-run / resume / validate), MULTI-RUN ANALYSIS (runs / show-run / diff-runs / replay / export-runs), EXPORT + DEPLOY (export / push / ollama), UI + INFRA (ui / info / config), DIAGNOSTICS (estimate-vram). 15 subcommands × one-line description each, plus the existing exit-code table and the sysexits.h overlay. Operators get a 30-line scannable workflow map instead of an argparse-auto-generated alphabetical dump. (BRIDGE-F-011)
- **`backprop push --hub-revision` + `--hub-commit-message` — HF Hub revision targeting.** v1.3's `backprop push` always pushed to the default branch of the target repo with the default upload commit message (`"Upload model"`). Operators pushing into a multi-branch HF repo (e.g. per-experiment branches, or pushing to a non-default `dev` branch for review-before-promote) had no way to target the alternate branch from the CLI. v1.4 adds `--hub-revision <branch>` (passes through to `huggingface_hub.HfApi.upload_folder(revision=...)`) and `--hub-commit-message <msg>` (overrides the default upload commit subject; useful for tying a CI-pushed model to the workflow run that produced it). Both flags are opt-in; omitting them preserves the v1.3.x byte-identical push behavior. (BRIDGE-F-014)
- **`backprop info --subcommand-tiers` — exposes the stability registry.** The `SUBCOMMAND_TIERS` dict in `cli.py:151` has been the source of truth for which subcommands are stable / experimental / deprecated-prefer-X since v1.3 Stage C, but no CLI flag surfaced it — operators learned a subcommand was experimental only via the inline deprecation hint when they invoked it. v1.4 adds `backprop info --subcommand-tiers`, which prints the registry as a two-column table (subcommand → tier) on the human-readable surface and as the existing `SUBCOMMAND_TIERS` dict under the `info --json` payload. Pre-fix operators see the same info as before (registry contents unchanged); the new flag is the introspection surface for it. (BRIDGE-F-015)
- **Web UI push-to-Hub form now exposes `--include-base` + `--token-file`.** v1.3.x's `/push-to-hub` page collected the target repo + inline token but never offered the `--include-base` toggle (push merged-base + adapter) or the `--token-file` path (read from disk vs paste inline). The CLI supported both; the UI fell behind. v1.4 adds an `--include-base` checkbox + a `--token-file` path field to the push form, plumbed into the existing `cmd_push` shellout. The token-paste field still works for backwards compat; the file path takes precedence when both are set (matches the CLI semantics). (FRONTEND-F-004)
- **`/datasets` page Stats grid now shows real counts.** v1.3.x's Dataset Stats grid was hardcoded to display the em-dash literal `'—'` for every cell — Avg tokens, Record count, Dedup hits all rendered as `—` regardless of the actual dataset. Operators looking at the page got no signal whether the dataset had been loaded, validated, or processed. v1.4 wires the grid to the real `DatasetStatsState` fields populated by the validate handler; the cells now show numeric values (or a "Not validated yet" placeholder when the dataset hasn't been processed). (FRONTEND-F-005)
- **6 regression-pinning tests added for prior-wave behavior.** v1.4 Wave 6b feature-audit closed six prior-wave coverage gaps: (1) Stage C `save_merged` warning when `save()` is called before `train()` succeeded, (2) Wave 3.5 untrained-save warning shape pinned, (3) Wave 3.5 `model.eval()` try/finally invariant restored on validation exception, (4) FRONTEND-B-001 `was_deleted` state field + Stage C `action_in_flight` mutual-exclusion contract, (5) schema_version pinning across the 6 JSON-output subcommands so a future shape change can't silently drop the version field, (6) `_read_hub_token_file` mode-0600 warning + missing-file `INPUT_AUTH_INVALID_SHAPE` shape. Each test pins a known regression source that the v1.4 swarm earned by reading the post-fix code; pre-fix the same logic was untested and the next refactor could have re-introduced any of the six. (TESTS-F-006)
- **Web UI now stamps Content-Security-Policy + clickjacking + sniff-protection headers on every HTTP response.** v1.3.x relied on Reflex's defaults (no CSP, no `X-Frame-Options`, no `X-Content-Type-Options`, no `Referrer-Policy`, no `Permissions-Policy`) — the GHSA-f65r-h4g3-3h9h advisory closure was authentication-only, leaving the defense-in-depth headers absent. v1.4 wires a new `security_headers_middleware` as the 5th ASGI layer (rate_limit → basic_auth → request_logging → security_headers → Reflex). Every response carries the documented baseline; the default CSP is `script-src 'self' 'unsafe-inline'` (Reflex inlines hydration scripts; nonce migration tracked for v1.5 against upstream Reflex). Report-only mode + nonce migration deferred — see [WAVE_6A_TODO.md](WAVE_6A_TODO.md) for the follow-ups. (FRONTEND-A-003)
- **`backprop export --hub-token-file` + `backprop push --token-file` keep HF tokens out of shell history.** Inline `--token` / `--hub-token` flags work but leave the token in `ps aux` and shell history. v1.4 mirrors the v1.3 `--auth-file` pattern: a one-line credential file (`chmod 600` recommended) read by the CLI and validated like the inline flag. Existing `--token` / `--hub-token` users see a one-time WARN line at handler entry pointing at the safer alternative; no behavior change beyond the warning. (BRIDGE-A-004)
- **`--share` cloudflared subprocess shutdown is now clean on Ctrl+C.** v1.3.x's `cmd_ui` invoked `cloudflared.terminate()` and `cloudflared.wait(...)` in a single `try` block; a `TimeoutExpired` from `wait` would crash through the finally block, leaving zombie cloudflared processes. v1.4 nests `terminate` → `wait` → `kill` → `wait` with inner timeout handling so a non-responsive cloudflared gets SIGKILL'd cleanly. Six new regression tests (`TestCloudflaredShutdown`) lock the contract. (TESTS-A-002)
- **`pytest -n auto` parallel-execution now safe across global-state tests.** v1.3.x didn't mark which tests mutated process-global state (logging config, `Settings.apply_windows_fixes()`, structlog config, `SecurityLogger` / `SessionManager` singletons). Running the suite under `pytest-xdist` would race on these surfaces and produce flaky failures. v1.4 marks 5 known-mutating test classes with `@pytest.mark.serial`, registers the marker in `pyproject.toml`, and pins all serial-marked tests to the same xdist worker via a `pytest_collection_modifyitems` hook. CI-side opt-in via `BACKPROPAGATE_PYTEST_PARALLEL=1`; serial tests still take the same wall-clock time, but the rest of the suite parallelizes safely. (TESTS-A-007)
- **`/runs` Diff button now wires to a working backend.** v1.3.x's run-detail page rendered a Diff button but the form was a producer-without-consumer — operators clicking it saw nothing happen. v1.4 wires the button to the `diff_against` handler and shows the diff output inline. (FRONTEND-A-001)
- **Auth badge shows `@username` chip.** v1.3.x's footer auth badge displayed mode (`Local · Basic`) but never the resolved username. v1.4 adds the `auth_user` field to `AuthBadgeState` and renders it as a `@username` chip — useful on shared training hosts where multiple operators have separate `--auth user:pass` credentials. (FRONTEND-A-002)
- **Recovery banner + structured-error callout chrome consolidated.** v1.3.x had three different "something went wrong" presentations across `train.py`, `runs.py`, `models.py`, `run_detail.py`. v1.4 lands `BpRecoveryBanner` (for OOM-recovery success / pause-on-overheat states) and `BpErrorCallout` (for structured `BackpropagateError` failures) and consolidates the per-page chrome onto the same components. Same data, same colors, same operator mental model across pages. (FRONTEND-A-004)

### Security

- **Transitive-dependency CVE sweep (Wave 6a.0 dep-sweep).** Bumped the `site/` Astro/Starlight stack to close 18 of 21 Dependabot alerts surfaced post-v1.3.0 ship (8 HIGH + 7 MEDIUM + 3 LOW). Major bumps: `astro` 5.17.3 → 6.3.7, `@astrojs/starlight` 0.37.6 → 0.39.2. The astro 6 cascade transitively bumped `picomatch` (4 alerts closed), `h3` (4 alerts), `devalue` (3 alerts), `defu` (1 alert), `svgo` (1 alert), `smol-toml` (1 alert), `astro` (3 alerts). Final `npm audit` post-bump: 0 vulnerabilities in the site/ ecosystem. Per the `[[astro-6-upgrade-gotchas]]` doctrine (earned in the Sovereign repo 2026-05-19), three companion changes shipped together: (a) `site/package.json` `overrides: { vite: "^7.3.2" }` to fix Vite 8 hoist breaking `@tailwindcss/vite` under Rolldown; (b) starlight 0.39's removal of top-level `autogenerate` on sidebar groups (the config now lives inside `items`) — `site/astro.config.mjs` sidebar block updated accordingly; (c) Node 22 was already in `pages-deploy.yml` from v1.4 Wave 2 (CIDOCS-A-008), no workflow change needed. Site build verified locally (`npm run build` → 16 pages built, pagefind index built, sitemap generated).
- **Three Python CVEs deferred to v1.5 (upstream blockers + dismiss-with-reason).** `diffusers` 0.37.1 → 0.38.0 stays held because `safetensors>=0.8.0rc0` is a pre-release; enabling pre-releases for a security bump trades one risk for another. CVE-2026-44513 + CVE-2026-45804 will close when safetensors 0.8.0 GA lands OR when `unsloth` loosens its safetensors floor. `transformers` 4.57.6 → 5.0.0rc3 stays held by the same pre-release policy (the v1.3 deferral reasoning applies unchanged — the bump requires major-version-compat work across `trainer.py` + `datasets.py` that the v1.5 dispatch will scope). `diskcache` CVE-2025-69872 dismissed-with-reason: no `first_patched_version` in the advisory feed, no newer release exists on PyPI, and `diskcache` is transitive via `llama-cpp-python` (only pulled in by the `[export]` extra) — the vulnerable codepath is not reachable from `backpropagate/**/*.py`. Same v1.3 reasoning carried forward; all three tracked in V1_5_BRIEF when posted.
- **5 superseded Dependabot PRs closed.** The sweep landed directly in the `wave-6a.0/dep-sweep` branch as a single coordinator-driven change (matches the v1.3 Wave 6a.0 pattern of one batch PR instead of per-package merges). The 5 individual Dependabot PRs (#106 devalue, #108 transformers, #109 astro+starlight, #113 defu, #114 picomatch) are closed as superseded — #108 stays deferred (pre-release; matches the v1.5 transformers carryforward), the other 4 are now covered by the astro 6 cascade. (Wave 6a.0)

## [1.3.0] - 2026-05-24

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
- **Nightly train smoke CI workflow.** `.github/workflows/nightly-train-smoke.yml` runs `Trainer(model='Qwen/Qwen2.5-0.5B-Instruct').train(max_steps=1)` on a CPU runner at 04:00 UTC each night, with a 15-minute hard timeout. Asserts checkpoint write + run_history entry + finite loss. The workflow opens / appends a `ci` + `nightly-smoke` labeled GitHub Issue on failure (collapsed onto the same issue across consecutive red nights, no spam). Observability only — never gates a release. Runner script: `scripts/nightly_train_smoke.py`.
- **Post-publish smoke workflow.** `.github/workflows/post-publish-smoke.yml` fires after `Publish` completes successfully and runs `pip install backpropagate==<tag>` + `backprop --version` across `{ubuntu, macos, windows} × {3.10, 3.11, 3.12, 3.13}` plus a `docker run ghcr.io/.../backpropagate:<tag> --version` smoke. 10-minute per-cell timeout. PyPI CDN lag handled with a 5-attempt × 30s-backoff retry loop. Failure opens / appends a `post-publish-smoke` labeled issue.
- **`verify.sh` is now consumed by CI.** New `verify-smoke` job in `ci.yml` runs `verify.sh --format=json`, uploads `verify.json` + per-stage `.log` files as an artifact, and gates on `.first_failed_stage` being null. The job is **soft-gated** (`continue-on-error: true`) for the first rotation of v1.3.x patches to bed in — the gate flips to strict after 3+ green runs confirm the JSON shape is stable.
- **CycloneDX SBOM attached to every GitHub Release.** The release workflow now generates `backpropagate-sbom.cdx.json` (Python install closure via `cyclonedx-py environment`) and a best-effort `backpropagate-npm-sbom.cdx.json` (npm shim closure via `@cyclonedx/cyclonedx-npm`), uploaded to the GH Release after creation with `--clobber` for idempotent re-runs. Sigstore SBOM attestation already rides the existing `--provenance` flag on `npm publish`; the standalone files give auditors a grep-able artifact without re-running install.
- **OpenSSF Scorecard workflow.** `.github/workflows/scorecard.yml` runs the `ossf/scorecard-action@v2` analysis weekly (Mon 06:00 UTC) and on push to main, publishes results to scorecard.dev (`publish_results: true`, OIDC-authed) AND the GitHub Security tab (SARIF upload). Badge added to the README badges row.
- **PR template.** `.github/PULL_REQUEST_TEMPLATE.md` collects Summary / Test plan / Breaking changes / Related issues + advisories / Doctrine touchpoints. The Doctrine section's checklist mirrors the four checks `scripts/check_doc_drift.py` enforces, so a contributor sees the drift surface they need to update BEFORE the gate fires on their PR.
- **Issue templates upgraded.** `bug_report.yml` now requires `run_id`, error code, `backprop info` output, traceback (with `BACKPROPAGATE_DEBUG=1` hint), repro steps, and install-channel dropdown — matching the load-bearing context fields named in the README "Reporting bugs" section + `CONTRIBUTING.md`. `feature_request.yml` collects use-case-first framing + proposed API + backward-compat impact. `config.yml` disables blank issues and routes security reports to the private advisory form.
- **Opt-in pytest-xdist parallel execution.** All three `pytest` invocations in `ci.yml` now respect the `BACKPROPAGATE_PYTEST_PARALLEL` repository variable — set to `1` to enable `-n auto --dist worksteal`. Defaults to `0` (serial), so the existing 1865-test baseline + coverage measurement is byte-identical until an operator opts in. Test agent owns the per-test `serial` / `xdist_group` marker audit that lets the suite go green under `-n auto`; this YAML is the consumer.
- **Updated-quality LoRA defaults shipped (rank 256, `target_modules="all-linear"`, 10× learning rate scale).** Per [Biderman 2024 — "LoRA Learns Less and Forgets Less"](https://arxiv.org/abs/2405.09673) and [Thinking Machines 2025 — "LoRA Without Regret"](https://thinkingmachines.ai/blog/lora/), this configuration matches full fine-tuning quality on most post-training tasks at ~67% of the compute. v1.2.x defaulted to rank 16 / q+v target — leaving 15–20% quality on the table. The new defaults are the largest free quality win in the v1.3 release.
- **DoRA support.** `LoraConfig.use_dora` field (default `False`); enable via Python `Trainer(..., use_dora=True)` or CLI `--use-dora`. peft's `LoraConfig(use_dora=True)` underneath. Rank-8 DoRA matches rank-32 LoRA on standard evals (+2.8% on LLaMA-7B); merges to zero inference overhead.
- **Sample packing default-on.** `SFTConfig.packing=True` by default in v1.3 — 1.7–3× documented throughput on variable-length conversational datasets. Opt-out via Python `Trainer(..., packing=False)` or CLI `--no-packing`. Attention-backend agnostic (FA2 / FA3 / xFormers / SDPA).
- **Paged 8-bit Adam auto-detected on consumer GPUs.** `optim="paged_adamw_8bit"` becomes the default on detected RTX 40/50-series cards (Ada / Blackwell). +25% throughput per [arXiv:2509.12229](https://arxiv.org/abs/2509.12229). Override via `--optim adamw_torch` (or any other `transformers.TrainingArguments` optim string).
- **PiSSA / LoftQ LoRA initialization flags.** `LoraConfig.init_lora_weights` accepts `"default" | "pissa" | "loftq"`; CLI flag `--init-lora-weights`. Free quality recovery on QLoRA runs; pairs cleanly with DoRA.
- **Ada-architecture mixed-precision tuning.** RTX 40/50-series autodetection switches the default mixed-precision dtype from `bf16` to `fp16` per [arXiv:2509.12229](https://arxiv.org/abs/2509.12229)'s peer-reviewed RTX 4060 study (bf16 underperforms fp16 on Ada cards). bf16 remains the default on Hopper / Ampere / non-Ada cards. Override unchanged via `Trainer(..., mixed_precision="bf16")`.
- **Three new model presets.** Phi-4-mini-3.8B (MIT license — best-in-class reasoning / math / code at ≤4B), Qwen-3.5-4B (Apache 2.0 — current sub-5B leader, MMLU-Pro 79.1, native long context), SmolLM3-3B (Apache 2.0 — fully open recipe, native 64K context). All three are Unsloth-supported and 4-bit-quantization clean.
- **Multi-run subcommand surface.** Three new CLI subcommands consuming `RunHistoryManager`: `backprop diff-runs <A> <B>` (side-by-side config / loss / hyperparameter diff, colorized by default; `--format=json` for machine consumption), `backprop replay <run-id>` (re-run with the same config + dataset; `--override key=value` for surgical tweaks), `backprop export-runs --format=jsonl` (bulk export of all run history for offline analytics / W&B-MLflow pipeline integration / disaster-recovery snapshots).
- **README rewrite landed.** The README opens with "Train an adapter. Ship it to Ollama. Move on." instead of feature-bullet shorthand, positions backpropagate against Axolotl / LLaMA-Factory / Unsloth / torchtune explicitly, adds an honest 16GB capability envelope table, an explicit "what backpropagate is NOT for" anti-pitch section with citations, and a References section. Re-translation runs at Phase 10.
- **Web UI request-logging middleware.** New ASGI middleware emitting one structured access log per request (method, path, status, duration_ms, auth_mode, auth_user, remote_addr). Opt-in via `BACKPROPAGATE_UI_REQUEST_LOG=1`; defaults off because Reflex's own logging already covers most surfaces. Integrates with the existing structlog pipeline. Lands AFTER auth in the middleware chain so the log record captures the resolved username (or `"anonymous"`).
- **Web UI rate-limit middleware.** `slowapi`-shaped per-IP limiter on the `/_event` WS upgrade and the `POST` / `PUT` / `PATCH` / `DELETE` HTTP surface. Default 100 req/min per IP, 10 WS upgrades per IP per minute; tunable via `BACKPROPAGATE_UI_RATE_LIMIT_*` env vars. Lands BEFORE auth in the middleware chain so brute-force attempts can't exhaust the HMAC budget.
- **Run-history UI drill-down.** Per-run page at `/runs/<run_id>` exposing run metadata, hyperparameter table, training-metrics chart, checkpoint list, log tail, and action buttons (Diff vs ..., Replay, Delete, Export). Strictly read-only at the data layer — Delete + Replay shell out to `backprop replay <run_id>` / `backprop delete-run <run_id>` rather than mutating run-history in the UI process.
- **GitHub Discussions enabled.** Discussions categories (Announcements / Q&A / Ideas / Show & Tell) configured, pinned welcome post links the bug-vs-question routing + GHSA private-advisory path. `CONTRIBUTING.md` names Discussions as the canonical Q&A channel.
- **Docker images now multi-arch (`linux/amd64` + `linux/arm64`).** `release.yml` uses `docker/setup-qemu-action` + `docker/setup-buildx-action` + `docker/build-push-action` with `platforms: linux/amd64,linux/arm64`. Apple Silicon and ARM Linux operators get a native image instead of the prior x86-64-only push.
- **`compose.yaml` at repo root for "UI in a container".** Canonical Docker Compose service exposes the Reflex web UI on port 7860 with a persistent `~/.backpropagate` volume mount and the standard `BACKPROPAGATE_UI_AUTH` / `BACKPROPAGATE_UI_HOST_BIND` env-var passthrough. `docker compose up` brings the UI up in one command. Cross-referenced from the handbook deployment page.
- **`CITATION.cff` rewritten.** Title aligned with the README h1, authors set to the org-level "MCP Tool Shop authors" collective with the contact email, keywords mirror `pyproject.toml`, and the `references:` section adds Biderman 2024 + the foundational LoRA paper (Hu 2021) alongside the existing SLAO paper (Qiao & Mahdavi 2025). DOI placeholder commented; mint on the v1.3.0 GitHub release.

### Changed

- **`verify.sh` accepts `--format=human|json`.** Default stays `human` (the existing banner output). `--format=json` emits one JSON object per stage (`{"stage", "status", "exit_code", "duration_seconds"}`) plus a final aggregate object — CI can parse the stream without screen-scraping. Each stage's stdout/stderr is captured to `verify-<stage>.log` so the JSON channel stays parseable.
- **CI workflow action SHAs aligned across workflows.** `release.yml`'s `actions/setup-python` pin moved from v6.0.0 → v6.2.0 (already in use by `ci.yml` + `publish.yml`) so the Node-runtime parity across workflows is consistent.
- **Default `--lora-r` from 16 → 256.** Backward-compat available via `--lora-preset=fast` (rank 16 / q+v target / 1× LR — the v1.2.x footprint). Operators who pinned `LORA__R=16` in their env or `Trainer(lora_r=16, ...)` calls are unaffected; only the implicit default changes.
- **Default `packing=True`.** `SFTConfig.packing` flips from off to on. Opt out with `--no-packing` (CLI) or `Trainer(..., packing=False)` (Python). Operators with strict-determinism requirements who used packing-off implicitly should pass the explicit flag in v1.3.
- **Qwen-2.5-3B preset now boots with a license caveat.** The preset is preserved for backward compatibility (existing CLI / Python users with `--model Qwen/Qwen2.5-3B-Instruct` continue to work) but the loader emits one `WARN preset_license_caveat preset=qwen-2.5-3b license=qwen-research notes="research license — commercial use restricted; consider Qwen-3.5-4B (Apache 2.0) or SmolLM3-3B (Apache 2.0) for commercial deployments."` line on first use. The preset table in the README and handbook flag the same caveat.

### Removed

- **npm distribution deprecated.** The `bin/backpropagate.js` shim used to bootstrap a Linux venv or download PyInstaller binaries from a GitHub Release via `@mcptoolshop/npm-launcher`. The binary build pipeline failed three consecutive times in v1.2.0 and the v1.0/v1.1/v1.2 release tags have zero attached binary assets — the launcher would 404. The shim now prints install guidance for the supported channels (`pipx install backpropagate` recommended, plus `uv tool install backpropagate` and `pip install backpropagate`) and exits 2. The npm package stays published so this message reaches existing `npm install -g backpropagate` users. The `@mcptoolshop/npm-launcher` runtime dependency was dropped from `package.json` (every `npm install` was pulling dead code).
- **PyInstaller binary distribution retired (full migration).** The v1.0–v1.2 install story shipped PyInstaller binaries from a GitHub Release, pulled via `@mcptoolshop/npm-launcher`. Linux bootstrapped a managed venv instead because `libtorch_cpu.so` blew past GitHub's 2 GB release-asset cap, and the binary build pipeline failed three consecutive times in v1.2.0 — the v1.0 / v1.1 / v1.2 release tags shipped zero binary assets, so `npm install -g backpropagate` would download then 404. v1.3 retires the path completely, in three steps the v1.3 brief tracked under D2 SPLIT:
  1. **Wave 1 — `bin/backpropagate.js` rewritten as a friendly-error shim.** `npm install -g backpropagate` still works; the shim prints install guidance for the three supported channels (`pipx install backpropagate` recommended, plus `uv tool install backpropagate` and `pip install backpropagate`) and exits 2. The `@mcptoolshop/npm-launcher` runtime dependency was dropped from `package.json` so each `npm install` no longer pulls dead code.
  2. **Wave 3.5 — `.github/workflows/release-binaries.yml` deleted.** It had failed 4 of the last 5 release runs and produced no shipped assets at any v1.x tag. Surviving comment references in `publish.yml` / `release.yml` / `ci.yml` / `bin/backpropagate.js` were updated to reflect retirement.
  3. **Wave 6a — PyInstaller `.spec` files removed from the repo root and the v1.2.x → v1.3 handbook migration page (`site/src/content/docs/handbook/migrations.md`) extended with a new "Switching off the PyInstaller / npm binary install" section that walks operators from the pre-deprecation `npm install -g backpropagate` install line through `pipx install backpropagate`, the optional-extras checklist, and the deprecation timeline (npm package itself remains published indefinitely so the shim's guidance reaches stragglers). `.gitignore` keeps `*.spec` so a stray local PyInstaller build doesn't leak back in via a future commit.

### Security

- **Transitive-dependency CVE sweep.** Bumped 14 packages in `uv.lock` to close 33 open advisories surfaced on the GitHub Security tab. High-severity closures: `urllib3` 2.6.3 → 2.7.0 (CVE-2026-44431, CVE-2026-44432), `python-multipart` 0.0.22 → 0.0.29 (CVE-2026-42561, CVE-2026-40347), `GitPython` 3.1.46 → 3.1.50 (CVE-2026-42284, CVE-2026-42215, CVE-2026-44243, CVE-2026-44244, plus GHSA-only advisory 57), `PyJWT` 2.11.0 → 2.13.0 (CVE-2026-32597 — this advisory affects the `[security]` extra's `JWTManager` helper in `ui_security.py`, which is a separate optional layer never reached by the auth middleware that closed GHSA-f65r-h4g3-3h9h; the bump still ships because operators who import `JWTManager` directly are on the user-facing path), `pillow` 12.1.1 → 12.2.0 (CVE-2026-40192, CVE-2026-42311, CVE-2026-42308, CVE-2026-42309, CVE-2026-42310). Medium / low closures: `aiohttp` 3.13.3 → 3.13.5 (10 CVEs), `cryptography` 46.0.5 → 48.0.0 (CVE-2026-34073, CVE-2026-39892), `Pygments` 2.19.2 → 2.20.0 (CVE-2026-4539), `idna` 3.11 → 3.16 (CVE-2026-45409), `pip` 26.0.1 → 26.1.1 (CVE-2026-3219, CVE-2026-6357), `pytest` 9.0.2 → 9.0.3 (CVE-2025-71176), `python-dotenv` 1.2.1 → 1.2.2 (CVE-2026-28684), `requests` 2.32.5 → 2.34.2 (CVE-2026-25645). Full test suite (1981 passed, 6 skipped) and the auth-middleware regression set (23 passed, 3 skipped) remain green across both tiers of bumps; no behavioral change observed in CI.
- **Two CVEs deferred to v1.4 (upstream blockers).** `diffusers` 0.36.0 → 0.37.1 partially advances but the patched 0.38.0 requires `safetensors>=0.8.0rc0` (a pre-release); enabling pre-releases for a security bump trades one risk for another. CVE-2026-44513 and CVE-2026-45804 will close when the safetensors 0.8.0 GA lands or when `unsloth` loosens its safetensors floor. Mitigation: `diffusers` is transitive via `unsloth` and is not imported by `backpropagate/**/*.py` — there is no reachable codepath from operator-facing surface into the vulnerable image-decode functions. `transformers` 4.57.6 is held back from the 5.0.0rc3 fix for CVE-2026-1839 by the same pre-release policy; this one IS a direct dependency, so the codepath argument does not apply — the bump is held only on the major-version-compat work that 5.0 would require across `trainer.py` + `datasets.py`. Both deferred items are tracked in the v1.3 brief for a v1.4 paired bump.
- **One CVE dismissed (no upstream patch).** `diskcache` CVE-2025-69872 has no `first_patched_version` in the advisory feed and no newer release exists on PyPI. `diskcache` is transitive via `llama-cpp-python` (only pulled in by the `[export]` extra) and is not imported by `backpropagate/**/*.py`; the alert will close automatically when upstream ships a fix.
- **CRITICAL-only `pip-audit` + Trivy floor preserved from v1.2.0.** No relaxation in v1.3; the same hard gates surface CRITICAL transitive CVEs while the advisory MEDIUM+ feed continues to populate the GitHub Security tab. The Bandit gating step now emits both txt (gating surface) and JSON (post-mortem artifact) for MEDIUM severity + MEDIUM confidence and above.
- **`SECURITY.md` is the canonical reporting policy + supported-versions surface; the operator-facing threat model + auth-middleware mode matrix is at `handbook/security.md`.** `CONTRIBUTING.md`'s historical pointer to `SECURITY_AUDIT_REPORT.md` (which became a stub in Wave 1) now points at the live docs instead.

### Known issues / tech debt

- The pre-v1.3 ERROR-severity Trivy alert cohort (incl. PyJWT CVE-2026-32597) was closed by the v1.3 dep-sweep above; two CVEs (`diffusers` 0.38.0, `transformers` 5.0.0rc3) are deferred to v1.4 because the upstream fixes are pre-release-only.
- **Python 3.10 reaches upstream EOL October 2026.** v1.3 still supports 3.10 (CI matrix runs 3.10 / 3.11 / 3.12 / 3.13). A future release (target: v1.4) will drop 3.10 to align with the upstream EOL. Operators standing up new installs should prefer Python 3.11 or 3.12 — 3.11 is the most-tested floor (the UI + Windows + macOS smoke cells all run on 3.11).
- The PyInstaller binary distribution migration is complete in v1.3 (see "Removed" above). The migration handbook page at `site/src/content/docs/handbook/migrations.md` is the operator-facing landing point for anyone still running `npm install -g backpropagate`.
- **uv.lock CI-consumption migration deferred to v1.4.** CI today installs via `pip install -e ".[dev,full]"`, which IGNORES `uv.lock` — the lockfile is scanned by Trivy as a security surface but never used to materialize the install. CIDOCS-F-011 / F-012 (Wave 5 audit) proposed migrating CI to `uv sync --frozen` so the lockfile becomes the install contract. The migration is non-trivial on the current 6-cell matrix (Linux × {3.10, 3.11, 3.12, 3.13} + Windows 3.11 + macOS 3.11) because `uv sync --extra` semantics differ from pip's bracket-extras syntax and the cross-platform `[tool.uv]` resolution markers need a separate audit. v1.3 ships with the pip install path unchanged; v1.4 will pick this up against a single cell first (Linux 3.11), then expand once the cross-platform behavior is validated.

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
- **CI hardening** — every third-party GitHub Action SHA-pinned. PyPI publish via OIDC trusted publishing (Sigstore provenance). Docker image digest-pinned + HEALTHCHECK. Multi-OS test matrix (Linux + Windows + macOS + Python 3.13). pip-audit + Trivy + Bandit + Semgrep + TruffleHog all gate on findings _(TruffleHog was retired post-v1.2.0 — secret scanning is now delegated to Trivy's built-in scanner; see the "CI gates re-tightened" note in [1.2.0] and the `secrets-scan` job comment in `.github/workflows/ci.yml`. pip-audit/Trivy gating was also subsequently narrowed to a CRITICAL floor; this entry describes the v1.0.0-era posture.)_
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

[Unreleased]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/mcp-tool-shop-org/backpropagate/compare/v1.2.0...v1.3.0
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
