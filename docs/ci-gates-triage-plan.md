# CI Gates Triage Plan — v1.1.x → v1.2

**Status:** plan (not yet scheduled). File a tracking issue when picking this up.
**Why:** the v1.1.0 swarm's Stage B added four CI gate tightenings without first
baselining what they'd block. Five hotfix PRs (#80 / #82 / #83 / #84 + the
rollback in this file's parent commit) chased findings one-by-one until it
became clear the baseline was ~30 CVEs + a pile of false-positive scanner
hits — too large to triage in-flight during release stabilization. The gates
are currently rolled back to advisory.

This document is the roadmap for putting them back the right way.

The doctrine this work serves is in
`~/.claude/projects/F--AI/memory/feedback_baseline_before_enforce.md`:
**baseline → fix → suppress with justification → enforce.** This wave does
exactly that for each of the four scanners.

---

## Current gate states (as of v1.1.1 + the rollback hotfix)

| Scanner | Current state | Was (pre-rollback) | Findings to triage |
|---|---|---|---|
| Bandit (SAST) | **gating at `-l -i` + pyproject `[tool.bandit]`** | gating | 0 (already clean after rollback PR — real bug B614 fixed, false positives have `# nosec` with justification) |
| Trivy (CVE scan) | advisory (`continue-on-error: true`) | gating with `exit-code: '1'` + `ignore-unfixed: true` + `.trivyignore` | ~30 CVEs across `uv.lock` + `site/package-lock.json` |
| pip-audit | advisory step kept; gating step disabled | gating fail-on-fixable | overlaps with Trivy's uv.lock findings |
| Semgrep (SAST) | gating (default action behavior) | same | 0 (clean after rollback) |
| TruffleHog | **removed entirely** | gating with `--only-verified --fail` | n/a — Trivy's built-in secret scanner covers this surface |
| Security-summary aggregate | advisory (`continue-on-error: true`) | gating | n/a (it's a roll-up of the above) |

Branch protection on `main` does NOT currently require any security check as
a gating status. That's intentional during the rollback — once each underlying
gate is back to green, re-add the aggregate as a required status check.

---

## Phase 1 — Baseline capture (~30 min)

For each scanner, run it locally OR pull the current main's CI SARIF, and
write the full finding list to a temp file. This is the **honest count**
that step 2 has to address.

```bash
# Bandit baseline
python -m bandit -r backpropagate/ -c pyproject.toml -l -i -f json > /tmp/bandit-baseline.json

# Trivy baseline (needs Docker; otherwise pull SARIF from GitHub)
docker run --rm -v "$PWD:/repo" aquasecurity/trivy:0.36.0 \
  fs --severity MEDIUM,HIGH,CRITICAL --ignore-unfixed /repo > /tmp/trivy-baseline.txt

# pip-audit baseline
pip-audit --skip-editable --format json > /tmp/pip-audit-baseline.json

# Semgrep baseline (needs Docker; otherwise CI SARIF)
docker run --rm -v "$PWD:/repo" semgrep/semgrep:1.163.0 \
  semgrep scan --config auto --config p/python --config p/security-audit --json /repo \
  > /tmp/semgrep-baseline.json
```

**Acceptance criterion for Phase 1:** every finding lives in one of three
buckets, with a deciding comment per finding:

- **Fix** (real bug or fixable dep CVE → patch in this PR)
- **Suppress** (false positive or accepted risk → inline `# nosec X — reason` or `.trivyignore` entry with justification line)
- **Defer** (real but out of scope this wave → file follow-up task, link from the suppression comment)

If a finding doesn't fit one of these three, the triage isn't done.

---

## Phase 2 — Fix the real bugs (~1–2 hours)

Expected real bugs (pattern from the v1.1.0 work):

- **B614 / unsafe torch.load** — fix any remaining sites missing `weights_only=True`. (The resume path was fixed in PR #82.)
- **Bandit B105 false positives in Pydantic metadata** — already suppressed (`# nosec B105 — Pydantic metadata flag…`) but check for new sites added since v1.1.0.
- **Trivy CVEs with available fixes** — bump the affected direct deps. The current `.trivyignore` documents the 14 most prominent; the remaining ~16 surfaced after deeper inspection. Likely targets:
  - `urllib3 → 2.7.0+` (CVE-2026-44431, 44432)
  - `Pillow → latest` (CVE-2026-42308–42311)
  - `GitPython → 3.1.45+` (GHSA-mv93-w799-cj2w + sister CVEs)
  - `IDNA → 3.10+` (CVE-2026-45409)
  - `pip → bump in image / lockfile` (CVE-2026-6357)
  - `python-multipart → 0.0.20+` (CVE-2026-42561)
  - `site/package-lock.json` — likely `astro` / `vite` / `@starlight` minor bumps; check Dependabot's open PRs first, they may cover most.
- **mypy real type bugs** — the v1.1.0 rollback PR caught `_classify_model_load_cause` (return type `str` → `ModelLoadCauseCategory`) and `cmd_resume` type narrowing. Any new mypy errors since: triage same way.

After Phase 2 the baseline files from Phase 1 should re-run with significantly fewer findings.

---

## Phase 3 — Suppress what can't be fixed (~30 min)

- Trivy: `.trivyignore` entries with a one-line context comment AND a link to the tracking issue for the next dep-bump attempt. No bare CVE IDs without context.
- Bandit: inline `# nosec BXXX — reason` at the offending line, never project-wide skips for individual codes.
- Semgrep: inline `# nosemgrep: <rule-id>` with justification.
- pip-audit: `--ignore-vuln <id>` flags with a tracking-issue link.

**Hard rule:** every suppression has a one-line WHY in the diff. Suppressions without justification are net-worse than the original finding because they encode "we looked at this, decided it didn't matter" without the evidence.

---

## Phase 4 — Re-tighten the gates (~15 min)

In `.github/workflows/ci.yml`:

- Remove `continue-on-error: true` from the Trivy step.
- Restore the `pip-audit (gating — fixed vulnerabilities only)` step body
  (the Stage B implementation is in the git history of PR #74; check
  `git log -p -- .github/workflows/ci.yml | grep -A 40 'pip-audit (gating'`).
- Remove `continue-on-error: true` from the security-summary `Gate on scanner results` step.
- Decide on TruffleHog: either install as binary in a filesystem-mode `run:` step, or leave Trivy's built-in secret scanner as the only coverage. Document the decision in CI's secret-scanning step header.

Run CI on a feature branch BEFORE merging. If anything re-surfaces, back to Phase 1 for that scanner.

---

## Phase 5 — Add branch protection required-status (~5 min)

In the repo's branch-protection rule for `main`, add `Security Summary` (the
job name from ci.yml) as a required status check. This is the ONLY status
that needs to be required; the underlying scanner jobs feed into it.

Verify: open a no-op PR; the merge button should refuse until Security Summary turns green.

---

## Total estimate

| Phase | Time | Why estimate could be off |
|---|---|---|
| 1 — Baseline capture | 30 min | Trivy + Semgrep need Docker; if Docker is uncooperative, double |
| 2 — Fix real bugs | 1–2 hours | Dep bumps may cascade through other deps; lockfile re-resolution can surface NEW CVEs |
| 3 — Suppress | 30 min | Mechanical once Phase 2 done |
| 4 — Re-tighten gates | 15 min | Mechanical |
| 5 — Branch protection | 5 min | Mechanical |
| **Total** | **~3 hours focused** | Allocate half a day to be safe |

---

## When to schedule

This is **not urgent**. The user-facing release surfaces (PyPI / npm / GitHub Release) are unaffected by main-branch CI being advisory. Schedule this when:

1. The next dep bump is in scope anyway (e.g. Dependabot opens enough PRs to make the bump natural).
2. There's a quiet half-day with no release pressure.
3. Or as Phase 0 of any future v1.2.0 swarm work — re-tightening the gates IS doctrine drift catching up to enforcement.

---

## What "done" looks like

- All 5 scanners gate (no `continue-on-error: true` on any security job).
- Branch protection requires `Security Summary` as a status check.
- `.trivyignore` has either zero entries or every entry has a one-line context note + link.
- No `# nosec` / `# nosemgrep` in `backpropagate/**` without a one-line WHY.
- This file (`docs/ci-gates-triage-plan.md`) gets a closing-out commit that
  links to the actual PR/commit SHA that closed each phase, then either
  gets deleted or moved to `docs/history/`.

---

## Why this isn't done now

Honest answer: the v1.1.0 swarm did not budget for the triage. The Stage B
work added the gates with the explicit (correct) understanding that they'd
fire on existing findings, but neither the agent nor the coordinator
inventoried what those existing findings actually were. Then v1.1.0 + v1.1.1
shipped under release-day pressure, and chasing each gate failure in the
hours after the release was a worse use of time than this scheduled plan
will be later.

The cost of this rollback is **one** un-tightened release window: v1.1.1
ships without these gates blocking pre-existing findings. The benefit is
that the next attempt does this once, properly, instead of through five
in-flight hotfix PRs.
