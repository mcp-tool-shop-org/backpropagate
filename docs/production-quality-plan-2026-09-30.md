# Production Quality Plan — backpropagate (2026-09-30)

> **Basis:** 2026-09-30 consult brief + same-day independent verification against
> the repo and GitHub live state. Every finding below was re-confirmed this
> session; corrections to the brief are marked **[correction]**.
>
> **Bottom line:** the library code is production-grade (3,528 collected tests,
> structured error taxonomy, tight security gates). The production-quality gaps
> are (a) two live bugs in the shipped install surface, (b) headline features
> never exercised on a real GPU, (c) guardrails missing around the dependency
> resolver / Dependabot, and (d) metadata that misleads future sessions. Four
> fixes are already written, green in CI, and waiting as PRs #213–#216.
>
> **Approved 2026-09-30 by the Director**, with the advisor's three revisions
> (GGUF golden-path smoke + 1.7.3 instead of holding #132/#133 for 1.8.0; GPU
> battery before the 1.7.2 tag; #213 today). D1–D4 accepted as recommended.

## Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | Each merge pins the PR head (`--match-head-commit`); each release pins a tag; the GPU receipt records the resolved package versions. |
| ANDON_AUTHORITY | 2 | A red required check halts a merge; a failed rig smoke halts the tag (Phase 1 step 2, Phase 2 step 4). Gates are never loosened to pass. |
| NAMED_COMPENSATORS | 2 | See table below. |
| DECOMPOSE_BY_SECRETS | 2 | Dependency PRs stay immutable; CI fixes (R1/R2), smokes and housekeeping land as separate PRs. |
| UNCERTAINTY_GATED_HUMANS | 2 | The Director decided D1–D5 up front; he is asked again only if a smoke fails or #132 needs a design change. |
| EXTERNAL_VERIFIER | n/a | No specialized claims. GitHub CI and the real-GPU smokes are the checks. |

| Irreversible action | Undo | State after undo | Owner |
|---|---|---|---|
| Squash-merge of #213–#216 to `main` | `git revert <sha>` via a PR | `main` tree as before the merge | executing session |
| Tag push `v1.7.2` / `v1.7.3` | `git push --delete origin <tag>` (only before publish runs) | no tag; release.yml does not fire | executing session |
| PyPI publish | `pypi yank backpropagate==<v>` (cannot re-upload the same version; ship the next patch) | version hidden from resolvers | Director (PyPI owner) |
| npm publish | `npm deprecate @mcptoolshop/backpropagate@<v> "<reason>"` (unpublish only within 72 h) | install warns | Director |
| GitHub release | `gh release delete <tag>` | release page gone; tag kept | executing session |
| GHCR image | delete the package version in GHCR | image unavailable | Director |
| Issue close (#129 etc.) | `gh issue reopen <n>` | open again | executing session |

---

## Verified state snapshot

| Surface | State (verified 2026-09-30) |
|---|---|
| Version | **1.7.1** (`pyproject.toml:11`), shipped 2026-09-07. Nothing on main since but the Atlas map (#212) and Codecov upload (#217). |
| Tests | **3,528 collected** (`pytest --collect-only`, local venv). CLAUDE.md still claims ~3,139. |
| Rig | **RTX 5090, 32 GB** (`nvidia-smi`), Windows-native. CLAUDE.md still says RTX 5080 / 16 GB. WSL2 required for the NCCL path. |
| README install | `pipx install "backpropagate[standard]"` → `[unsloth,ui]` (README.md:53-55,134-142). |
| `[unsloth]` extra | `unsloth>=2024.1`, no caps restated (`pyproject.toml:84`). Measured in PR #215: pip backtracks **102 releases**, lands unsloth 2025.11.1 / torch 2.14 / transformers 4.57.2 / trl 0.23.0 — a stack never tested on a GPU. |
| KTO crash | `trainer.py:2028-2032` passes `max_prompt_length` whenever set explicitly **or `max_seq_length ≤ 512`** (default is 2048, so the crash needs a small window or an explicit flag — not every KTO run). trl 0.27.2 is inside the `<0.28` cap and removed that field → `TypeError`. **[correction to brief: bounded trigger, not all KTO runs.]** |
| GPU smokes | Exist for ORPO, SimPO, KTO, FP8, MLX (`tests/test_*_smoke.py`) — manual, on-rig, integration-marked. **None** for FSDP2 offload; **none** for the 24–34B QLoRA presets. `test_envelope_v17.py` is arithmetic/gating only (its own docstring says the real-GPU FSDP smoke "is run separately by the coordinator" — it never was). |
| Dependabot | `.github/dependabot.yml` has **zero `ignore` rules**; all five ecosystems run **monthly** (next run ≈ Oct 1). Closed caps-widening PRs (#173, #194, #210) will be reopened/re-attempted. |
| Release flow | `scripts/prep_release.sh` runs 8 stages (citation, references, translations, build, PyPI metadata, drift gate, verify.sh). **No GPU-smoke stage.** |
| Open PRs | #213, #214, #215, #216 all **MERGEABLE, all checks SUCCESS**, all marked `BEHIND` (PR #217 Codecov merged after them). #216 is explicitly stacked on #215 + #213. |
| Golden path | **No real-GPU test exists for the default path** (QLoRA 4-bit SFT → merge → GGUF → Ollama). GPU smokes cover only ORPO/SimPO/KTO/FP8/MLX. The one end-to-end export test (`tests/test_e2e_chain.py:175`) writes a **stub** GGUF (`b"GGUF...MOCK"`), so CI cannot see #132/#133. `export.py` was edited 5× since #132 was filed (2026-05-26, against v1.4.0) but no commit references it — **nobody knows whether it still reproduces.** |
| Open issues | 4 live bugs: **#132 (HIGH)** GGUF export `UnboundLocalError` on bnb-4bit checkpoints; **#133 (HIGH)** GGUF export requires llama.cpp's `convert_hf_to_gguf.py` which the package doesn't bundle; **#134 (MED)** trl import fails on Windows with cp1252 `UnicodeDecodeError` on `deepseekv3.jinja`; **#135 (LOW)** `Logger._log()` rejects `cli_run_id` kwarg. Plus #129: nightly-smoke issue from 2026-05-25, **still open though the smoke has been green for 8 consecutive weeks** (last red run not in the recent history). Plus 4 backlog items (#9–#13). |
| pip-audit flake | `ci.yml` CRITICAL-floor step runs pip-audit `--strict || true` then immediately `json.load`s the report. If OSV times out, pip-audit crashes, no JSON exists, `json.load` raises → **required check goes red for a service outage, not a finding**. |
| pip-audit accept list | Still ACKs 3 GHSA IDs with `review-by 2026-08-31` — a month past its own deadline. PR #214 empties the list (all fixed at locked versions). |

---

## Decisions needed before any code lands

These are the maintainer decisions the September session was waiting on. Nothing
else blocks Phase 0.

- **D1 — Accept the pip-user stack change (PR #215).** After it, `[standard]` /
  `[full]` / `[production]` installs get unsloth 2026.9.11, torch 2.12.1,
  transformers 5.5.0, trl 0.24.0, datasets 4.3.0 — the tested stack. Side
  effect: CI stops accidentally testing transformers 4.x; installs without the
  unsloth extra still allow `transformers>=4.46.0`. **Recommend: accept.** The
  alternative is shipping a resolver lottery to every README-recommended
  install.
- **D2 — Accept trl cap `<0.28` → `<2` (PR #216).** Same next-major doctrine as
  `torch<3` / `transformers<6`, with `_require_config_field()` hard errors
  replacing silent field drops. Only sound **with** D1 — without the unsloth
  floor, a wider trl cap lets resolvers walk unsloth backwards (what #210 did).
  **Recommend: accept, as a pair with D1.**
- **D3 — Python 3.10 in v1.7.2 vs v1.8.0.** 3.10 reaches upstream EOL October
  2026 — this month. The repo's own rule removes it in the first release after
  EOL. **Recommend: 1.7.2 keeps 3.10 (patch = minimal blast radius); 1.8.0,
  cut in November, drops it** — by then EOL is officially in effect and the
  removal work (mypy/ruff targets, CI matrix, docs) has room.
- **D4 — Trivy MEDIUM+ re-tightening (left open by #214).** `.trivyignore`'s
  protocol says re-tighten when the file empties; but the baseline isn't clean
  (transformers CVE-2026-9856 is blocked by unsloth's `transformers<=5.5.0`).
  **Recommend: do not gate yet; file a tracking issue with a review-by date
  tied to unsloth lifting its cap.** Gating now turns every PR red on a fix
  nobody can install.
- **D5 — Schedule the GGUF export pair (#132 + #133, both HIGH).** These hit
  the headline "one-click GGUF export to Ollama" promise and have been open
  four months across four releases. **Decided (revised 2026-09-30): do NOT
  hold them for 1.8.0.** Phase 2's golden-path smoke answers whether #132
  still reproduces; if it does, the fix ships as **1.7.3**, a patch — never
  bundled with the 3.10 drop, so 3.10 users get the fix too. #133 is partly
  mitigated already (4-location discovery + `BACKPROPAGATE_LLAMA_CPP_PATH` +
  a structured error, and llama.cpp is only the fallback after unsloth's own
  `save_pretrained_gguf`); the title-cased model-name rejection still needs a
  check. #134/#135 stay on 1.8.0.

---

## Phase 0 — Land the four green PRs (Day 0–1; effort: hours)

All four passed full CI on Sept 23; the only state risk is the `BEHIND` flag
from #217 (Codecov upload — no overlap with any PR's diff).

| # | Order | Action |
|---|---|---|
| 1 | First — **today, 2026-09-30** | Rebase + merge **#213** (Dependabot cap guards). Config-only. Monthly Dependabot fires on the 1st, i.e. **tomorrow**; if #213 is not on main by then the three closed bump PRs reopen. |
| 2 | Second | Rebase + merge **#215** (unsloth floor + restated caps). Depends on decision D1. |
| 3 | Third | Rebase **#216** (diff shrinks to its 3 own commits per the PR body), then merge. Depends on D1+D2. This lands the KTO/SimPO/CPO `TypeError` fix. |
| 4 | Any time | Merge **#214** (retire `.trivyignore`, empty pip-audit accept list). Independent; small rebase risk against #213's edit of the same `dependabot.yml` comment — whichever lands second rebases. Then record decision D4 as a tracking issue, not a gate change. |

**Acceptance:** all four merged; `main` green; a fresh
`pip install "backpropagate[standard]"` in a scratch venv resolves in ~45 s to
the tested stack (mirrors the #215 measurement); Dependabot's October run opens
no cap-widening PRs.

---

## Phase 1 — v1.7.2 patch release (Day 1–2)

Exactly the bugs that make today's shipped package broken, nothing else.

1. **Version + changelog.** Bump to `1.7.2`; move the entries #215/#216 already
   added under `[Unreleased]` into `[1.7.2]`. Note the KTO/`max_prompt_length`
   fix prominently — it is the user-visible crash.
2. **GPU gate BEFORE the tag (moved up from Phase 2).** #215 changes what
   every `[standard]` install resolves to (unsloth 2026.9.11, torch 2.12.1,
   transformers 5.5.0). Build `scripts/gpu_smoke.sh` now and run the
   **existing** smoke battery (ORPO/SimPO/KTO/FP8) on the rig in a scratch
   venv installed the way a user would (`pip install ".[standard]"`), plus a
   KTO run at `max_seq_length=256` on a plain `pip install .` venv (trl
   0.27.x). No tag without that receipt — Phase 2's rule applies to 1.7.2.
3. **Release.** Run `scripts/prep_release.sh` (citation bump, translations,
   build, twined metadata smoke, drift gate, verify.sh), tag, push, let
   `release.yml` publish; watch `post-publish-smoke.yml`.
4. **Post-release:** confirm `pip install backpropagate==1.7.2` from PyPI
   resolves to the tested stack.

**Ride-along CI fixes (separate small PRs — keep the dep PRs immutable):**

- **R1 — pip-audit OSV resilience** (`ci.yml` pip-audit job): guard the
  `json.load` (missing-file ⇒ job-level warning + skip, never a red required
  check), and add a bounded retry (3 attempts, backoff) around the pip-audit
  invocation — `tenacity` is already a core dep. Scan-error must be
  distinguishable from scan-finding.
- **R2 — no-unsloth CI coverage** (closes the gap #216's notes name): switch
  an **existing** cell (e.g. 3.13) to install *without* the unsloth extra
  rather than adding a new one — 1.7.1 just cut Actions spend ~60%. The
  plain-install trl path (0.27.x today, 1.x eventually) is then tested in CI
  instead of only in scratch venvs.

---

## Phase 2 — Real-GPU verification of the v1.7 headline (Week 1–2)

The README's lead claim ("single-card 7B-class full fine-tuning") has zero
executed evidence; the 24–34B presets likewise. Every feature that *did* get a
GPU smoke (ORPO/SimPO/KTO/FP8) repaid it — six bugs the unit tests missed.

1. **`tests/test_full_ft_offload_smoke.py`** (new, integration-marked, modeled
   on the four existing `*_smoke.py`): tiny model (SmolLM2-135M family, offline
   HF cache), `full_ft_offload=True`, 2 steps. Assert: process group
   initialized without `torchrun`, finite loss, params on CPU-offload policy,
   artifact loads. **Runs in WSL2** (NCCL) — document the exact invocation in
   the test module docstring.
2. **`tests/test_qlora_presets_smoke.py`** (new): parametrize the four v1.7
   presets. On the 5090: `qwen2.5-14b` trains 2 steps end-to-end; 24b/32b get a
   load + 1-step no-OOM probe at their shipped `max_seq_length` (32b is
   documented "just fits" — that claim needs one executed run).
3. **`tests/test_golden_path_smoke.py`** (new — the product's main promise
   has no real test): SmolLM2-135M, QLoRA 4-bit SFT, 2 steps →
   `export --format gguf` → `ollama create` (skip honestly if Ollama is not
   running). Run it **both** with unsloth (the `save_pretrained_gguf` path)
   and without (the llama.cpp fallback via `BACKPROPAGATE_LLAMA_CPP_PATH`).
   This is the repro for #132/#133; if either fails, the fix is **1.7.3**
   (see D5). Replace the stub-GGUF assumption in `test_e2e_chain.py` docs
   with a pointer to this smoke.
4. **Wire the battery into the release flow:** `scripts/gpu_smoke.sh`
   (built in Phase 1) gains the offload, preset and golden-path smokes, plus a
   written step in `prep_release.sh`'s output and the handbook release
   checklist: *no tag before the rig receipt exists.* No paid GPU runner is
   needed — this is a checklist obligation, not a CI job.
5. **Record verification** in CHANGELOG/README status the way FP8 got
   "experimental → verified" in v1.6.

**Acceptance:** offload smoke passes in WSL2 on the 5090; preset smoke passes;
release checklist prints the GPU-battery gate; README claims matched by
executed runs.

---

## Phase 3 — v1.8.0: drop Python 3.10 (November, post-EOL; per D3)

Mechanical but wide — one focused wave:

- `pyproject.toml`: `requires-python = ">=3.11"`; drop the 3.10 classifier;
  bump `[tool.mypy].python_version` and ruff `target-version` to `py311`; run
  ruff `UP` autofixes; refresh `uv.lock`.
- `ci.yml`: remove the 3.10 cell — **note it currently owns the ruff + mypy +
  atlas steps**, so those must move to the 3.11 or 3.12 cell, not vanish.
- Docs: README prerequisites, handbook install page, CLAUDE.md 3.10 note.
- Slot #134/#135 here. The GGUF HIGHs (#132/#133) are **not** held for
  1.8.0 — they ship as 1.7.3 if Phase 2's golden-path smoke reproduces them
  (D5).

---

## Phase 4 — Housekeeping & trust surfaces (ride along with 1.7.2)

- **CLAUDE.md refresh:** status line (v1.7.1 shipped Sept 7; v1.7.2 in
  preparation), test count (~3,528), rig (RTX 5090 / 32 GB), extras list
  (missing `fp8`, `mlx`, `full-no-export` bundle), drop the stale "macOS smoke
  cells" claim (removed in 1.7.1), refresh the Ship Gate date.
- **`docs/ci-gates-triage-plan.md:15`:** body still reads "plan (not yet
  scheduled)" while the header banner says executed in v1.2.0 — collapse the
  contradiction (mark executed; keep as methodology archive).
- **Issue #129:** close with a comment noting 8 consecutive green weeks; the
  workflow's auto-open path has no auto-close counterpart — add one line to
  the workflow's issue body reminding the maintainer, or teach it to close
  on green (nice-to-have).
- **Issue triage pass:** apply milestones to #132–#135 (per D5) and prune or
  re-scope the four 2025-era backlog items (#9–#13).
- **Ship Gate refresh:** re-run `shipcheck audit` after 1.7.2 ships; post the
  current scorecard/date to CLAUDE.md (last run 2026-02-27).

---

## Phase 5 — Process hardening (low effort, standing)

- **Stacked-PR discipline:** the four PRs sat green-but-unlanded for a week
  while two live bugs shipped in the released package. The fix is not a rule
  on the maintainer (a CONTRIBUTING clause would bind only Mike) but
  surfacing: any session opening this repo lists green PRs awaiting a
  decision first. Stacked PRs rebase in declared order.
- **Dependabot cadence:** keep monthly for deps (cost doctrine), but the
  October run is the regression test for #213 — verify explicitly that no
  trl/transformers/torch cap-widening reopens, then close the loop in the
  1.7.2 notes.
- **Flakiness doctrine:** required checks must fail only on findings, never on
  service outages — pip-audit R1 sets the pattern; apply to any future
  network-backed gate.

---

## Definition of done

- `pip install "backpropagate[standard]"` resolves in ~45 s to the stack CI
  and the rig actually run.
- KTO/SimPO/ORPO train on a plain `pip install backpropagate` at any supported
  window, verified by a CI cell that installs without unsloth.
- Every headline capability (QLoRA 14–34B presets, full-FT offload, FP8) has
  executed real-GPU evidence, re-collected by a written pre-tag checklist step.
- Dependabot cannot move capped dependency floors on its own initiative.
- pip-audit's required check is resilient to OSV outages.
- CLAUDE.md, the triage plan, CHANGELOG, and open issues describe the repo as
  it actually is (version 1.7.2+, ~3.5k tests, RTX 5090, no stale statuses).
- `backprop export --format gguf` → Ollama is proven by a real GPU run on
  both the unsloth and llama.cpp paths; #132/#133 closed (1.7.3 if needed).
- Python 3.10 removal lands on the EOL-anchored 1.8.0 schedule.

## Effort summary

| Phase | Scope | Est. time |
|---|---|---|
| 0 | Merge 4 green PRs (decisions D1/D2/D4) | Hours |
| 1 | GPU battery on the new stack, v1.7.2 release, 2 small CI PRs | 1–2 days |
| 2 | Offload + preset + golden-path smokes (WSL2 run); 1.7.3 if #132 reproduces | 2–3 days |
| 3 | 1.8.0: 3.10 removal + #134/#135 | 1–2 days |
| 4 | Housekeeping docs/issues/scorecard | Half day |
| 5 | Standing policy edits | Hours |
