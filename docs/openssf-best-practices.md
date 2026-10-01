# OpenSSF Best Practices badge: passing-level answers for backpropagate

- **Assessed:** 2026-10-01
- **Commit assessed:** `f5e87b4cc232c3945ed4cbb6f03f49395ee6641a` (`origin/main`, "release: v1.8.0 (#262)")
- **Project:** https://github.com/mcp-tool-shop-org/backpropagate
- **Criteria source:** `coreinfrastructure/best-practices-badge`, `criteria/criteria.yml` level `0` (passing), 67 criteria, wording from `config/locales/en.yml`.
- **Tally (as the code stands today):** Met 63, Unmet 1, N/A 3. The one Unmet is `crypto_password_storage` (a MUST, gap G1). Once G1 is resolved to Met (code fix) or to a disclosed N/A, all 67 criteria are Met or N/A.

## How to use this file

1. Open https://www.bestpractices.dev, sign in with GitHub, choose "Add project", and enter `https://github.com/mcp-tool-shop-org/backpropagate`. The form pre-fills some answers from the repository; check every one against this file.
2. Close the gaps in the next section first. The "MUST" gaps change an answer; the others only make the evidence cleaner or the public documents truer.
3. For each criterion, copy the **Answer** and the **Justification** into the form. Paste the **Evidence** URLs into the URL boxes the form asks for. Line references are to commit `f5e87b4`; if the files move, re-check them.
4. These are public attestations under the Director's name. Where this file says "Director's own attestation" or "judgment call", the answer depends on a fact only he can confirm; do not copy it blindly.
5. Do not mark anything Met that is not listed as Met here. Anything this file marks Unmet stays Unmet until the fix is merged and the evidence is re-read.

---

## Gaps to close before submitting

### MUST gaps (these decide whether the badge passes)

**G1. `crypto_password_storage` (MUST): currently answered Unmet; the Director must decide.**
The web UI's `--auth user:pass` credential is held in plaintext, not as a salted, iterated hash:
- The CLI exports it to the UI process as `BACKPROPAGATE_UI_AUTH` (`backpropagate/ui_app/auth.py:100`, `backpropagate/cli.py:3511`).
- The CLI writes it, still as `user:pass`, to a per-launch lock file (`backpropagate/cli.py:2592-2659`, called at `:3511-3516`; mode 0600 on POSIX, per-user ACL on Windows; deleted at shutdown, `:3759-3762`).
- The middleware compares it in constant time (`auth.py:203-243`) and also uses `SHA-256(user:pass)`, unsalted and unstretched, as the HMAC key for the session cookie (`auth.py:177-197`). A captured cookie therefore lets an attacker test password guesses offline at fast-hash speed.

N/A ("the software does not store passwords for external users") is arguable: there is no user database, and the one credential is chosen by the operator at launch. I would not attest it without the disclosure below, because the code does write the password to disk and holds it in an environment variable.

Two honest ways out:
- **Smallest code fix (recommended):** (a) in `cli.py:cmd_ui`, compute a salted `hashlib.scrypt` (or `pbkdf2_hmac`, 600k+ iterations) verifier from the supplied password once at launch, pass only the verifier to the UI process (new env var, e.g. `BACKPROPAGATE_UI_AUTH_VERIFIER`), and have `_verify_basic_auth` check it with `hmac.compare_digest`; (b) generate a random 32-byte cookie secret with `secrets.token_bytes(32)` in the CLI and pass it as `BACKPROPAGATE_UI_COOKIE_SECRET`, so `_derive_secret` stops hashing the password; (c) stop writing `user:pass` into the lock file (nothing in the repository reads that file; write only the launch token, or nothing, in basic-auth mode). Then answer **Met**: "The UI keeps only a salted scrypt verifier; no plaintext password is persisted."
- **No code change:** answer **N/A** and put this in the justification box: "backpropagate has no user database. The web UI has a single credential, supplied by the operator at launch (`--auth` or `--auth-file`). It is held in the UI process's environment and in a per-launch lock file (mode 0600, removed at shutdown); it is not stored in a database or persisted between launches." That is true as written, but a reviewer may reject it. The cookie-key weakness above stays either way (see G7).

**G2. `interact` and `contribution` (MUST): Met, but the README never links to CONTRIBUTING.md.**
`CONTRIBUTING.md` explains the contribution process, and GitHub surfaces it on the repository, issue and PR pages. But `README.md` has no "Contributing" section and does not link `CONTRIBUTING.md` (verified by search), and the handbook landing page and handbook index do not mention contributing. If you give the form the Pages site or README as the "project website", `interact` has nothing to point at for "contribute".
**Fix (one line):** add to README.md, near "Reporting bugs" (line 438): `## Contributing` / `Pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) for the dev loop, test requirements and PR process.` Add the same link to `site/src/content/docs/handbook/index.md`. Use the repository URL as the project website in the form.

### Non-blocking gaps (SHOULD / SUGGESTED, or documents that say something untrue)

**G3. `SECURITY.md` "Supported Versions" is stale (lines 5-15).** It says 1.5.x is current and 1.4.x is supported; the current release is 1.8.0 (1.7.2 and 1.8.0 shipped 2026-09-30 and 2026-10-01). Fix: rewrite the table for the current policy (current minor plus one back).

**G4. `SECURITY.md:33` offers email to `64996768+mcp-tool-shop@users.noreply.github.com`.** A GitHub noreply address does not receive mail, so the "email the maintainer" route cannot work. Fix: delete line 3 of the reporting steps and line 33, or replace with a monitored address. The private advisory form (enabled, verified below) is the working route and satisfies `vulnerability_report_private` on its own.

**G5. `SECURITY.md:21` lists GHSA-f65r-h4g3-3h9h without its CVE.** The advisory now has CVE-2026-48797 (GitHub API, 2026-10-01); `SECURITY.md:23` still says CVE IDs "are requested". The v1.2.0 CHANGELOG entry (`CHANGELOG.md:604`) says "CVE pending assignment". Fix: add the CVE ID to the table and the CHANGELOG line.

**G6. `CONTRIBUTING.md` carries stale facts** that contradict `pyproject.toml` and `README.md`: "Coverage floor is 50%" (line 9; it is 90, `pyproject.toml:383`), "~2000 tests" (line 52; about 7,100), "3.10 ... will be dropped in v1.4" (line 18; README.md:358 says first release after Oct 2026). Fix: update the three sentences. Also add one sentence to the PR process, line 137: "Bug fixes and new functionality must come with tests" (it already says "Write tests for new functionality", which is enough for `test_policy`, so this is polish).

**G7. Hardening that is not a criterion but is true and public-facing:** (a) the cookie HMAC key is `SHA-256(user:pass)` (see G1; fix is the random cookie secret); (b) `--host <non-loopback>` serves HTTP Basic over plain HTTP and nothing in `README.md`, `handbook/security.md` or `cli.py` says so (search for "cleartext/plaintext/unencrypted" found nothing); add one sentence to `handbook/security.md` recommending `--share` (HTTPS tunnel) or SSH port-forwarding; (c) `JWTConfig` / `BACKPROPAGATE_SECURITY__JWT_SECRET` has no minimum secret length (`ui_security.py:2095-2105`); add a 32-byte minimum check. Note the JWT/CSRF classes are helpers; the bundled UI does not use them (`CHANGELOG.md:31-32`; no import outside `ui_security.py`).

**G8. `handbook/security.md:77` and `:84` describe a default per-launch random token and lock file that the CLI does not create.** `cli.py:3606` passes `token_query=None`; `BACKPROPAGATE_UI_LAUNCH_TOKEN` is read but never generated anywhere (`grep` over `backpropagate/`); `cli.py:3508` calls that mode "not yet exposed on the CLI". The default `backprop ui` runs unauthenticated on loopback with Host/Origin allowlists. Fix: change the table row and the paragraph to say so (or generate the token). This matters because the Director is attesting to `know_secure_design` and `documentation_basics` (how to use securely).

**G9. `eval.py` `pass_rate` metric runs model-generated code in-process with a reduced builtins map; the docstring says it is not a security boundary (`backpropagate/eval.py:337-352`), but the handbook (`cli-reference.md:458`, `recipes.md:138`) does not.** Fix: add "executes model output without isolation; run only on code you would already run locally" to the `--metric` row.

**G10. 22 code-scanning alerts are open on the Security tab.** I triaged the ten Semgrep ones against the code (`slao.py:1715,1851`, `multi_run.py:3492,4230`, `security.py:432`: torch `save`/`load` with `weights_only=True`, false positives; `eval.py:370,379`: the documented `pass_rate` `exec`; `datasets.py:1910`: `sha1(..., usedforsecurity=False)` for dedupe keys; `security.py:246`: logs template names, no credential; `cli.py:2586`: a `chmod 0o700`). None is a confirmed exploitable vulnerability. The Semgrep job is advisory (it does not fail the build), which is why they sit open. Fix: dismiss each in the Security tab with a reason ("false positive" / "used in tests" / "won't fix") so the public tab matches the triage. The rest are Scorecard (workflow pinning, token permissions, branch protection, `CIIBestPracticesID`, which this badge clears) and two Trivy dependency alerts. Leaving them open does not make `static_analysis_fixed` Unmet, but a reader will ask.

**G11. Missing tag.** `CHANGELOG.md` has a `[1.0.2] - 2026-03-25` entry; no `v1.0.2` tag or Release exists. Tags exist for every other release since 1.0.0 (22 GitHub Releases). Not worth fixing; mention if asked.

### Judgment calls the Director must confirm (answers are given, but they depend on facts only he knows or on a reading of the criterion)

- **J1. `report_responses` and `enhancement_responses`.** Every issue in the repository (10, all states) was filed by the maintainer or a bot (`gh issue list --state all`); there are no outside bug reports or enhancement requests, and no outside PRs (140 merged PRs are the maintainer's; 99 PRs are Dependabot's). The "majority of reports in the last 2-12 months" is therefore vacuous. I answer Met with that disclosed. Note the four maintainer-filed bug issues #132-#135 (created 2026-05-26) were closed on 2026-09-30, about four months later. `enhancement_responses` is SHOULD, so answering Unmet with a one-line reason would not cost the badge.
- **J2. `vulnerability_report_response` N/A.** The only security advisory ever created (GHSA-f65r-h4g3-3h9h) was found by the maintainer's own audit. There are no vulnerability reports from outside in the last 6 months, as far as the repository shows. Confirm there was no private email or message report; if there was one, this becomes Met or Unmet depending on when it was answered.
- **J3. `vulnerabilities_fixed_60_days` and dependency advisories.** I read the criterion as being about backpropagate's own code ("patched and released by the project itself"). On that reading it is Met. But 5 Dependabot alerts are open and several are older than 60 days (table below), because `uv.lock` and the `[unsloth]` extra cap torch, transformers and datasets below the fixed versions. Mitigations are in code and in `osv-scanner.toml`, but the vulnerable versions are what `pip install backpropagate[standard]` resolves. If the Director reads the criterion as including dependencies, this is Unmet.
- **J4. `know_secure_design` and `know_common_errors`** are statements about a person. The evidence in the repository supports them; the Director must be willing to say them himself.
- **J5. #263 is still open.** The Director's note listed #263 among the fixes made on 2026-10-01; it is open and unmerged (created 2026-10-01 12:08Z, after the v1.8.0 release). It records the setuptools CVE-2026-59890 ignore and makes `mutmut.yml` read-only. Nothing in this file depends on it being merged, but do not describe it as fixed.

---

## Live facts read on 2026-10-01 (with the command)

| Fact | Value | Command or source |
|---|---|---|
| Public, default branch, license | public, `main`, MIT | `gh api repos/mcp-tool-shop-org/backpropagate` |
| Homepage | https://mcp-tool-shop-org.github.io/backpropagate/ | same |
| Issues / Discussions enabled | yes / yes (1 post) | same; GraphQL `discussions` |
| Private vulnerability reporting | **enabled** (`{"enabled":true}`) | `gh api repos/.../private-vulnerability-reporting` |
| Secret scanning + push protection | enabled / enabled; 0 open alerts | `security_and_analysis`; `gh api .../secret-scanning/alerts` |
| Dependabot security updates | enabled | `security_and_analysis` |
| Branch protection on `main` | 16 required status checks (Bandit, Semgrep, Trivy, Secret Detection, Security Summary, Dependency Audit, build, drift-check, 4 test cells, xdist, uv pilot, UI compile, verify.sh), strict; `enforce_admins` false | `gh api .../branches/main/protection` |
| CI on `f5e87b4` | `CI`, `OpenSSF Scorecard`, `Doc Drift Check` all success; Publish and Post-Publish Smoke success | `gh run list --branch main` |
| GitHub Releases | 22 (v0.1.0 ... v1.8.0), all with notes; v1.8.0 assets: wheel, sdist, 2 CycloneDX SBOMs, `multiple.intoto.jsonl` (SLSA provenance) | `gh release list`; releases API |
| Tags | 22 release tags (`v0.1.0` ... `v1.8.0`, one per GitHub Release) plus internal `swarm-save-*` and `savepoint-*` tags | `git tag --list` |
| HTTPS | project site 200 over HTTPS; the `http://` URL returns 301 to HTTPS; PyPI JSON 200; npm registry 200 | `curl` |
| Published advisory | GHSA-f65r-h4g3-3h9h / CVE-2026-48797, Critical, affects 1.1.0-1.1.1, patched 1.2.0 | `gh api .../security-advisories` |
| Time to fix that advisory | found 2026-05-22 (own audit); first vulnerable release v1.1.0 2026-05-21T10:24Z; patched v1.2.0 2026-05-23T08:26Z (about 46 hours exposed); advisory published 2026-05-23T08:58Z | advisory; release list |
| Open Dependabot alerts | 5 (table below) | `gh api .../dependabot/alerts?state=open` |
| Open code-scanning alerts | 22 (see G10) | `gh api .../code-scanning/alerts?state=open` |

### Dependency advisories open on 2026-10-01 (relevant to J3)

| Package | Advisory | Severity | Public since | Fix available | State |
|---|---|---|---|---|---|
| transformers | CVE-2026-9856 / PYSEC-2026-3929 | high | 2026-08-02 (GitHub) / 2026-09-10 (OSV) | 5.10.0, above unsloth cap `<=5.5.0` | mitigated in code: `security.check_chat_template_names` (`backpropagate/security.py:223`), v1.8.0, `.trivyignore`, `osv-scanner.toml` |
| transformers | CVE-2026-80047 / GHSA-x9r9-c232-4q39 | high | 2026-09-01 | none in range | not reachable by default: `trust_remote_code` defaults off since v1.7.2 (`osv-scanner.toml`) |
| setuptools | CVE-2026-59890 | medium | 2026-07-21 | 83 (torch requires `<82`) | build-time only; hatchling backend; ignore entry is in unmerged #263 |
| diskcache | CVE-2025-69872 | medium | 2026-02-11 | none upstream | not imported by backpropagate; transitive via llama-cpp-python (`osv-scanner.toml`) |
| torch | CVE-2025-3000 | low | 2025-03-31 | 2.13, above unsloth cap `<2.13` | `torch.jit.script` never called (`osv-scanner.toml`) |
| datasets | CVE-2026-66007 / PYSEC-2026-3716 | OSV only (not a Dependabot alert) | 2026-07-24 | 5.0.1, above unsloth cap `<4.4` | `save_to_disk` / `push_to_hub` never called on datasets (`osv-scanner.toml`) |

Fixed on 2026-10-01 (all dependency advisories, merged within hours of the Scorecard report): #246 (gitpython 3.2.0, virtualenv 21.14.1; PR opened 02:08Z, merged 02:29Z), #247 (accelerate 1.15.0 for PYSEC-2026-3804, public 2026-09-10, so 21 days; chat-template guard; merged 02:54Z), #249 (hash-pinned CI installs, SLSA provenance; merged 05:02Z). These were not vulnerabilities in backpropagate's own code.

---

# The 67 criteria

Format: **id** (category), **Answer**, justification, evidence. Evidence paths are relative to the repository root unless a URL is given. `BASE` = https://github.com/mcp-tool-shop-org/backpropagate

## Basics

### Basic project website content

**description_good** (MUST). **Met.**
The README opens with what the software does: it fine-tunes large language models on a single GPU and exports the result to Ollama, with a three-line Python example. The Pages site says "Headless LLM fine-tuning with smart defaults."
Evidence: `README.md:18-35`; https://mcp-tool-shop-org.github.io/backpropagate/ (hero text, fetched 2026-10-01).

**interact** (MUST). **Met** (evidence thin, see G2).
Obtain: install commands (pipx, uv, pip, Docker) in the README, and PyPI/npm/GHCR packages. Feedback: "Reporting bugs" section and the bug-report and feature-request issue templates, plus GitHub Discussions. Contribute: `CONTRIBUTING.md` in the repository root. The README does not link CONTRIBUTING.md yet; use the repository URL as the project website and close G2.
Evidence: `README.md:37-57` (obtain), `README.md:438-449` (feedback), `CONTRIBUTING.md:133-162` (contribute), `.github/ISSUE_TEMPLATE/bug_report.yml`, `.github/ISSUE_TEMPLATE/feature_request.yml`; `BASE/blob/main/CONTRIBUTING.md`.

**contribution** (MUST, URL required). **Met.**
`CONTRIBUTING.md` explains the process: fork, branch, write tests, run lint and tests, commit with a prefix, open a pull request; issues and Discussions for questions and bugs.
Evidence: `CONTRIBUTING.md:133-162`, `:191-245`; `BASE/blob/main/CONTRIBUTING.md`.

**contribution_requirements** (SHOULD, URL required). **Met.**
`CONTRIBUTING.md` lists the requirements for acceptable contributions: ruff, mypy and pre-commit style, tests for new functionality, the four local checks CI runs (ruff, mypy, pytest, doc-drift gate), commit message prefixes, design principles, and security rules (no pickle, validate input, `weights_only=True`, path-traversal checks). The PR template repeats the checks. Some figures in the file are out of date (G6).
Evidence: `CONTRIBUTING.md:41-80`, `:144-189`; `.github/PULL_REQUEST_TEMPLATE.md`; `BASE/blob/main/CONTRIBUTING.md`.

### FLOSS license

**floss_license** (MUST). **Met.**
The software is released under the MIT license.
Evidence: `LICENSE:1-21`; `pyproject.toml:14` (`license = "MIT"`); `CITATION.cff:18`; GitHub API `license.spdx_id` = `MIT`.

**floss_license_osi** (SUGGESTED). **Met.**
MIT is approved by the Open Source Initiative.
Evidence: `LICENSE`; https://opensource.org/license/mit/ (linked from the Pages site).

**license_location** (MUST, URL required). **Met.**
The license is the top-level file `LICENSE`.
Evidence: `LICENSE`; `BASE/blob/main/LICENSE`.

### Documentation

**documentation_basics** (MUST). **Met.**
The README and the Starlight handbook cover how to install, start and use it (quick start, CLI, Python API, recipes), and how to use it securely (threat model, `--share` requires `--auth`, SSH port-forwarding, output-directory sandbox, redacted error output, `trust_remote_code` off by default). See G7, G8 and G9 for places where the security documentation should be corrected or extended.
Evidence: `README.md:37-57`, `:141-196`, `:320-352`; `site/src/content/docs/handbook/getting-started.md`, `.../security.md:1-140`; https://mcp-tool-shop-org.github.io/backpropagate/handbook/.

**documentation_interface** (MUST). **Met.**
The handbook is the reference for the external interface: every `Trainer(...)` parameter, callback and result type (`training.md`), every CLI subcommand and flag with defaults and exit codes (`cli-reference.md`), every environment variable (`env-vars.md`), and every stable error code (`error-codes.md`). A drift gate (`scripts/check_doc_drift.py`, run in CI) checks that env vars, flags and error codes in the code match the handbook. It is hand-written reference, not generated API docs.
Evidence: `site/src/content/docs/handbook/training.md:8,46-123`, `.../cli-reference.md`, `.../env-vars.md`, `.../error-codes.md`, `.../reference.md`; `.github/workflows/doc-drift.yml`; `CONTRIBUTING.md:49-56`.

### Other

**sites_https** (MUST). **Met.**
The project site, repository, issue tracker and every download location are HTTPS: GitHub, the GitHub Pages handbook (plain-HTTP URL returns a 301 redirect to HTTPS), PyPI, npm, and `ghcr.io`.
Evidence: `curl` on 2026-10-01: handbook 200, `http://` form 301 to `https://`; `https://pypi.org/pypi/backpropagate/json` 200; npm registry 200; `pyproject.toml:222-226`.

**discussion** (MUST). **Met.**
GitHub Issues, pull request threads and GitHub Discussions are searchable, URL-addressable, open to anyone with a GitHub account, and need no proprietary client software.
Evidence: `BASE/issues`, `BASE/pulls`, `BASE/discussions` (Discussions enabled; `has_discussions: true`); `CONTRIBUTING.md:239-245`.

**english** (SHOULD). **Met.**
Documentation, issues and discussions are in English; the README is also translated into seven other languages.
Evidence: `README.md:1-3`; `README.*.md`; `.github/ISSUE_TEMPLATE/bug_report.yml`.

**maintained** (MUST). **Met.**
v1.8.0 shipped on 2026-10-01 (22 GitHub Releases since 2026-01-19); 140 maintainer PRs merged and the last commit on `main` is that release; weekly Scorecard run, Dependabot configured, CI green on the release commit.
Evidence: `gh release list`; `git log` on `main` (`f5e87b4` 2026-10-01); `gh run list --branch main` (CI success on `f5e87b4`); `BASE/releases`.

## Change control

### Public version-controlled source repository

**repo_public** (MUST). **Met.**
Public git repository at https://github.com/mcp-tool-shop-org/backpropagate (clone URL `BASE.git`, `pyproject.toml:224`).
Evidence: GitHub API `private: false`; `pyproject.toml:222-226`.

**repo_track** (MUST). **Met.**
Git records what changed, who changed it and when.
Evidence: `git log` and `BASE/commits/main`.

**repo_interim** (MUST). **Met.**
Interim work is public: all development lands through pull requests (252 PRs visible), not only tagged releases.
Evidence: `BASE/pulls?q=is%3Apr`; `git log` (e.g. #253, #259, #262 between releases).

**repo_distributed** (SUGGESTED). **Met.**
The project uses git.
Evidence: `BASE.git`.

### Unique version numbering

**version_unique** (MUST). **Met.**
Every release has a unique version number, which must match the git tag (`release.yml` fails if tag and `package.json` version differ).
Evidence: `pyproject.toml:11` (`version = "1.8.0"`); `.github/workflows/release.yml` ("Verify tag matches package.json version"); `BASE/releases`.

**version_semver** (SUGGESTED). **Met.**
Semantic Versioning, declared in the changelog header.
Evidence: `CHANGELOG.md:3-6`.

**version_tags** (SUGGESTED). **Met.**
Each release is a git tag (`v0.1.0` ... `v1.8.0`) with a GitHub Release. One CHANGELOG entry, 1.0.2, has no tag (G11).
Evidence: `git tag --list`; `BASE/tags`; `gh release list`.

### Release notes

**release_notes** (MUST, URL required). **Met.**
Each release has human-written release notes: a Keep a Changelog file, and the same text on the GitHub Release (v1.8.0: Added, Security, Fixed sections; 8,022 characters). They are not `git log` output.
Evidence: `CHANGELOG.md:1-60`; `BASE/blob/main/CHANGELOG.md`; `BASE/releases/tag/v1.8.0`.

**release_notes_vulns** (MUST). **Met.**
The one publicly known vulnerability in backpropagate's own code that had an identifier when it was fixed, GHSA-f65r-h4g3-3h9h (the UI auth bypass), is named in the v1.2.0 release notes and CHANGELOG. The CVE ID (CVE-2026-48797) was assigned afterwards and should be added (G5). Later releases name dependency CVEs they close. The v1.7.2 and v1.8.0 security fixes in backpropagate's own code had no CVE or GHSA when released.
Evidence: `CHANGELOG.md:598-604`, `:842`; `CHANGELOG.md:10-60` (1.8.0 Security section); `BASE/security/advisories/GHSA-f65r-h4g3-3h9h`.

## Reporting

### Bug-reporting process

**report_process** (MUST, URL required). **Met.**
Bugs are reported through the GitHub issue tracker using a template that asks for the run ID, error code, `backprop info` output, traceback and reproduction.
Evidence: `README.md:438-449`; `CONTRIBUTING.md:191-209`, `:242`; `.github/ISSUE_TEMPLATE/bug_report.yml`; `BASE/issues/new?template=bug_report.yml`.

**report_tracker** (SHOULD). **Met.**
GitHub Issues is the tracker.
Evidence: `BASE/issues`; `.github/ISSUE_TEMPLATE/config.yml` (blank issues disabled, template routing).

**report_responses** (MUST). **Met** (vacuous; judgment call J1).
No outside user has filed a bug report: all 10 issues in the repository's history were filed by the maintainer or by an automated CI workflow, and the maintainer commented on each of the five bug issues (#129, #132-#135). If the Director does not accept this reading, the honest statement is "no external bug reports have been received".
Evidence: `gh issue list --state all` (authors: `mcp-tool-shop` ×9, `github-actions` ×1); issues #129, #132-#135.

**enhancement_responses** (SHOULD). **Met** (vacuous; judgment call J1).
No outside enhancement requests have been received. The five roadmap issues (#9-#13) were filed by the maintainer in January 2026; three are closed and two (#11, #13) are open without comment. Answering Unmet is also acceptable for a SHOULD criterion.
Evidence: `gh issue list --state all`; `BASE/issues/9`, `/10`, `/11`, `/12`, `/13`.

**report_archive** (MUST, URL required). **Met.**
GitHub keeps issues, comments and pull requests public and searchable.
Evidence: `BASE/issues?q=is%3Aissue`.

### Vulnerability report process

**vulnerability_report_process** (MUST, URL required). **Met.**
`SECURITY.md` is published in the repository root (GitHub shows it on the Security tab) and says how to report privately and what to include, with a response timeline. See G3 and G4 for two things to correct in it.
Evidence: `SECURITY.md:25-46`; `BASE/blob/main/SECURITY.md`; `.github/ISSUE_TEMPLATE/config.yml` (contact link to the advisory form); handbook `security.md:138-140`.

**vulnerability_report_private** (MUST, N/A allowed, URL required). **Met.**
Private reports go through GitHub's private vulnerability reporting form (HTTPS), which is enabled on the repository (checked 2026-10-01: `{"enabled":true}`). `SECURITY.md` says not to open a public issue and links the form.
Evidence: `SECURITY.md:29-31`; `gh api repos/mcp-tool-shop-org/backpropagate/private-vulnerability-reporting`; `BASE/security/advisories/new`.

**vulnerability_report_response** (MUST, N/A allowed). **N/A** (judgment call J2).
No vulnerability report has been received from outside in the last six months. The only advisory, GHSA-f65r-h4g3-3h9h, came from the maintainer's own audit (2026-05-22) and was fixed within about 46 hours of the first vulnerable release. `SECURITY.md` commits to acknowledgment within 48 hours.
Evidence: `gh api repos/mcp-tool-shop-org/backpropagate/security-advisories` (one advisory, published 2026-05-23, found by internal audit per its Credit section); `SECURITY.md:41-46`.

## Quality

### Working build system

**build** (MUST, N/A allowed). **Met.**
A pure-Python package built with hatchling from `pyproject.toml`; `python -m build` produces the wheel and sdist from source, and a CI `build` job (a required check) and the publish workflow do exactly that. The Dockerfile also builds from source.
Evidence: `pyproject.toml:1-7`, `:232-242`; `.github/workflows/ci.yml:1187` (`build` job); `.github/workflows/publish.yml` (`build` job, hashes the artifacts); `Dockerfile`; `.github/workflows/ci.yml:732` (`docker-build-smoke`).

**build_common_tools** (SUGGESTED, N/A allowed). **Met.**
Standard Python tooling: `python -m build`, hatchling, pip, uv.
Evidence: `pyproject.toml:6`; `scripts/ci_install_locked.sh`; `requirements/*.in`.

**build_floss_tools** (SHOULD, N/A allowed). **Met.**
Every build and test tool is free software (hatchling, pip, uv, pytest, ruff, mypy, bandit, semgrep, trivy).
Evidence: `pyproject.toml:196-219`; `.github/workflows/ci.yml`.

### Automated test suite

**test** (MUST). **Met.**
The tests are in `tests/` (MIT, same repository): about 140 files, 7,145 tests (pinned 2026-10-01), with Hypothesis property tests and Atheris fuzz harnesses. `CONTRIBUTING.md` documents how to run them, and CI runs them on every pull request.
Evidence: `tests/`; `CLAUDE.md:32`; `CHANGELOG.md:135`; `CONTRIBUTING.md:41-56`, `:84-101`; `.github/workflows/ci.yml:353-391`.

**test_invocation** (SHOULD). **Met.**
`pytest` is the standard Python invocation: `pytest tests/ -m "not gpu and not slow and not integration"`.
Evidence: `pyproject.toml:344-367`; `CONTRIBUTING.md:86-101`.

**test_most** (SUGGESTED). **Met.**
Line and branch coverage is 98.5% (branch coverage is on), and CI fails below 90%. GPU-only paths have separate real-GPU smoke tests run by hand.
Evidence: `pyproject.toml:373-391` (`branch = true`, `fail_under = 90`); `CHANGELOG.md:135`; `.github/workflows/ci.yml:367-391`.

**test_continuous_integration** (SUGGESTED). **Met.**
GitHub Actions runs the suite on every pull request and on pushes to `main` (Linux 3.10, 3.11, 3.13; Windows 3.11; parallel and uv cells); 16 required status checks; coverage goes to Codecov.
Evidence: `.github/workflows/ci.yml:36-60`, `:74-431`; branch protection (16 required contexts).

### New functionality testing

**test_policy** (MUST). **Met.**
The contribution guide and the PR template both require tests for new functionality.
Evidence: `CONTRIBUTING.md:137` ("Write tests for new functionality"), `:180` ("Write tests (aim for 80%+ coverage on new code)"); `.github/PULL_REQUEST_TEMPLATE.md` (Test plan, "Targeted regression set").

**tests_are_added** (MUST). **Met.**
Recent major changes arrived with tests: the experimental block engine (#237; `tests/test_block_engine.py`, `test_block_engine_trainer.py`), single-card offload (#231; `tests/test_offload_engine.py`, `test_offload_fit.py`), the chat-template guard (#247; `tests/test_chat_template_guard.py`, 19 tests), the CSRF/session fix (#253; `tests/test_ui_cov_security_bugs.py`), the fuzz fixes (#259; each fix turns an expected-failure into a regression test), and the v1.8.0 coverage drive (3,528 to 7,145 tests).
Evidence: `gh pr view 237 / 231 / 247 / 253 / 259 --json files`; `CHANGELOG.md:135`.

**tests_documented_added** (SUGGESTED). **Met.**
Documented in `CONTRIBUTING.md` and in the PR template.
Evidence: `CONTRIBUTING.md:137`, `:180`; `.github/PULL_REQUEST_TEMPLATE.md`.

### Warning flags

**warnings** (MUST, N/A allowed). **Met.**
Ruff lints (pycodestyle, pyflakes, isort, bugbear, comprehensions, pyupgrade, unused-arguments, simplify) and mypy type-checks (`disallow_untyped_defs`, `check_untyped_defs`, `warn_return_any`) on every pull request.
Evidence: `pyproject.toml:244-295`; `.github/workflows/ci.yml:307-323`; `.pre-commit-config.yaml`.

**warnings_fixed** (MUST, N/A allowed). **Met.**
Both gates are hard failures in CI, `ruff check backpropagate/` passes at the assessed commit ("All checks passed!", run 2026-10-01 with ruff 0.16.6), and CI is green on the release commit. A small set of rules is switched off by name with a reason in `pyproject.toml`.
Evidence: `.github/workflows/ci.yml:307-323`; `pyproject.toml:260-276`; `gh run list --branch main` (CI success on `f5e87b4`).

**warnings_strict** (SUGGESTED, N/A allowed). **Met.**
Nine ruff rule families plus strict mypy options on the core package. Honest limits: the Reflex UI tree has loosened mypy settings, and about fourteen ruff rules are ignored by name (`pyproject.toml:260-276`, `:297-342`).
Evidence: `pyproject.toml:244-342`.

## Security

### Secure development knowledge

**know_secure_design** (MUST). **Met** (Director's own attestation, J4).
The design shows the principles in practice: fail-safe defaults (loopback bind; `--share` and non-loopback `--host` refuse to start without `--auth`; `trust_remote_code` off by default since v1.7.2); complete mediation (one ASGI middleware gates HTTP routes and the WebSocket upgrade, plus Host and Origin allowlists); least privilege (writes confined to one sandbox directory); input validation with allowlists (upload extensions, model names, path sandbox); limited attack surface (UI is an optional extra). The threat model is written down.
Evidence: `backpropagate/ui_app/auth.py:1-43`, `:302-345`; `backpropagate/security.py:99-222`; `backpropagate/ui_security.py` (`FileValidator`); `SECURITY.md:53-55`; `site/src/content/docs/handbook/security.md:12-31`, `:47-100`.

**know_common_errors** (MUST). **Met** (Director's own attestation, J4).
The project tracks and mitigates the common error classes for this kind of software: missing authentication/authorization (the GHSA-f65r advisory, CWE-862), path traversal (`safe_path`), unsafe deserialization (`torch.load(weights_only=True)`, pickle refused), CSRF and cross-site WebSocket hijacking (token plus Origin allowlist), DNS rebinding (Host allowlist), credential leakage in logs (stderr redaction), and arbitrary code in model repos (`trust_remote_code`). `CONTRIBUTING.md` lists the rules; Bandit and Semgrep security rule sets run in CI.
Evidence: `CONTRIBUTING.md:183-189`; `backpropagate/security.py:99-222`, `:346-432`; `backpropagate/ui_app/auth.py:302-345`; `SECURITY.md:17-23`, `:39`.

### Use basic good cryptographic practices

backpropagate does not implement cryptographic protocols or primitives. It uses Python's standard library (`hmac`, `hashlib`, `secrets`) and the PyJWT library, in three places in `backpropagate/ui_app/auth.py` and `backpropagate/ui_security.py`. TLS is not implemented by the software: a `--share` tunnel (cloudflared) terminates HTTPS, and the loopback default needs none.

**crypto_published** (MUST, N/A allowed). **Met.**
Only published, expert-reviewed algorithms: HMAC-SHA-256 (RFC 2104, FIPS 198-1) for the UI session cookie; SHA-256 (FIPS 180-4) for key derivation and session keys; HS256 JWT (RFC 7519/7518) in the optional `JWTManager` helper; `secrets` (OS CSPRNG) for random values.
Evidence: `backpropagate/ui_app/auth.py:177-200`, `:246-299`; `backpropagate/ui_security.py:2056-2073`, `:2146-2150`, `:2419`.

**crypto_call** (SHOULD, N/A allowed). **Met.**
It calls the standard library and PyJWT; it does not implement its own ciphers, hashes or JWT signing. The cookie format (`user:exp:HMAC`) is assembled by hand from `hmac.new(..., sha256)` and verified with `hmac.compare_digest`.
Evidence: `backpropagate/ui_app/auth.py:246-299`; `backpropagate/ui_security.py:2146-2150`, `:2180-2186`; `pyproject.toml:140-143`.

**crypto_floss** (MUST, N/A allowed). **Met.**
Everything is implementable with free software: Python stdlib (PSF license) and PyJWT (MIT). No proprietary crypto dependency.
Evidence: `pyproject.toml:140-143`; imports at `backpropagate/ui_app/auth.py:47-57`, `backpropagate/ui_security.py:43-56`, `:2056`.

**crypto_keylength** (MUST, N/A allowed). **Met.**
Every key the software generates is 256 bits: `secrets.token_bytes(32)` for the process cookie secret (`auth.py:200`), `secrets.token_urlsafe(32)` for the JWT secret (`ui_security.py:2095-2100`) and CSRF tokens (`:2274`), and 128 bits for CSP nonces (`:2632`). HMAC-SHA-256 and SHA-256 (256-bit output) exceed the 224-bit hash minimum. There is no option to select shorter keys or weaker algorithms. Caveat: the secrets an operator supplies (`--auth` password, `BACKPROPAGATE_SECURITY__JWT_SECRET`) have no enforced minimum length (G7).
Evidence: `backpropagate/ui_app/auth.py:200`; `backpropagate/ui_security.py:2095-2105`, `:2272-2274`, `:2629-2634`.

**crypto_working** (MUST, N/A allowed). **Met.**
No MD4, MD5, DES, RC4, ECB or other broken algorithm in a security mechanism (searched `backpropagate/` for `md5`, `sha1`, `DES`, `hash(`, `random.`). The only SHA-1 is a dataset de-duplication key, marked `usedforsecurity=False`. `random.Random` is used only for seeded dataset shuffles and bootstrap resampling, each marked `# nosec B311` as non-cryptographic.
Evidence: `backpropagate/datasets.py:1896-1913`, `:3311`; `backpropagate/eval.py:436`; `backpropagate/offload_engine.py:512`.

**crypto_weaknesses** (SHOULD, N/A allowed). **Met.**
No SHA-1 or CBC in any security mechanism. One weakness to know about, though not a broken algorithm: the explicit-credentials cookie key is `SHA-256(user:pass)` with no salt or work factor (G1, G7).
Evidence: `backpropagate/ui_app/auth.py:177-197`.

**crypto_pfs** (SHOULD, N/A allowed). **N/A.**
The software implements no key-agreement protocol. HTTPS for `--share` is terminated by the cloudflared tunnel, outside this project.
Evidence: `backpropagate/ui_app/auth.py` and `backpropagate/ui_security.py` contain no key exchange; `site/src/content/docs/handbook/security.md:86`.

**crypto_password_storage** (MUST, N/A allowed). **Unmet** as the code stands (decision G1).
The UI's basic-auth password is not stored as an iterated, salted hash: it is held in plaintext in the UI process environment and in a per-launch lock file (0600 on POSIX; removed at shutdown), and is compared in constant time. There is no user database. Either implement the verifier fix in G1 and answer Met, or answer N/A with the disclosure in G1.
Evidence: `backpropagate/ui_app/auth.py:203-243`; `backpropagate/cli.py:2592-2659`, `:3511-3516`, `:3759-3762`; `site/src/content/docs/handbook/security.md:82-84`.

**crypto_random** (MUST, N/A allowed). **Met.**
All keys, tokens and nonces come from `secrets` (the OS CSPRNG): the cookie secret (`secrets.token_bytes(32)`, `auth.py:200`), the JWT secret and CSRF tokens (`secrets.token_urlsafe(32)`, `ui_security.py:2100`, `:2274`), and CSP nonces (`secrets.token_bytes(16)`, `:2632`). JWT IDs use `uuid.uuid4()`, which is also random-based, and are not secrets (`:2139`). The seeded `random.Random` uses elsewhere are not security mechanisms.
Evidence: lines above; a search of `backpropagate/` for `random.` and `import random` finds only the seeded dataset/eval/training uses listed under `crypto_working`.

### Secured delivery against man-in-the-middle (MITM) attacks

**delivery_mitm** (MUST). **Met.**
Releases are delivered over HTTPS only: PyPI (Trusted Publishing from `publish.yml`), npm (OIDC with Sigstore provenance), GHCR, and GitHub Releases. Beyond that, v1.8.0 carries SLSA build provenance (`multiple.intoto.jsonl`) and hashes of the wheel and sdist are checked in the publish workflow.
Evidence: `.github/workflows/publish.yml` (sha256 checks at `:166`, `:227`; provenance job); `.github/workflows/release.yml`; v1.8.0 release assets (checked 2026-10-01); `README.md:37-57`.

**delivery_unsigned** (MUST). **Met.**
No hash is fetched over HTTP and trusted without a signature. All download and install instructions use HTTPS, and CI installs are hash-pinned (`pip install --require-hashes`). The README's `curl` example uses an HTTPS raw.githubusercontent.com URL.
Evidence: `rg "http://"` over `Dockerfile`, `compose.yaml`, `scripts/`, `.github/` (only localhost and XML namespace matches); `README.md:147`; `scripts/ci_install_locked.sh`; `requirements/*.txt`.

### Publicly known vulnerabilities fixed

**vulnerabilities_fixed_60_days** (MUST). **Met** (judgment call J3).
There is no unpatched vulnerability of medium or higher severity in backpropagate's own code that has been public for more than 60 days. The one published advisory (critical) was fixed and released in v1.2.0 within about 46 hours of the first vulnerable release. The v1.7.2 and v1.8.0 security fixes in its own code were fixed in the same release in which they were found. Open dependency advisories are listed in the table above with their mitigations; read the criterion as covering only the project's own code, or answer Unmet.
Evidence: GHSA-f65r-h4g3-3h9h (`BASE/security/advisories/GHSA-f65r-h4g3-3h9h`); `CHANGELOG.md:10-60`, `:598-604`; `osv-scanner.toml`; `gh api .../dependabot/alerts?state=open`.

**vulnerabilities_critical_fixed** (SHOULD). **Met.**
The one critical vulnerability (GHSA-f65r, CVSS 9.8) was found 2026-05-22 and patched in v1.2.0 on 2026-05-23; critical dependency advisories were bumped when found (e.g. PyJWT 2.14, anyio 4.14.2).
Evidence: `BASE/security/advisories/GHSA-f65r-h4g3-3h9h`; `CHANGELOG.md:166-171`, `:577-604`; `pyproject.toml:141`.

### Other security issues

**no_leaked_credentials** (MUST). **Met.**
No working private credential is in the public repository. GitHub secret scanning and push protection are on, with no open alerts; a search for private keys and AWS, GitHub, OpenAI and Hugging Face token patterns found only fake values in test fixtures. CI also has a secret-detection job (a required check). README examples (`alice:hunter2`) are sample values.
Evidence: `gh api repos/.../secret-scanning/alerts` (empty); `security_and_analysis`; `tests/test_ui_cov_logging.py:157,201`, `tests/test_ui_cov_state_forms.py:722` (fake values); `.github/workflows/ci.yml:1172` (`secrets-scan`).

## Analysis

### Static code analysis

**static_analysis** (MUST, N/A allowed, justification required). **Met.**
Bandit (the build fails at LOW severity/LOW confidence and above), Semgrep (`auto`, `p/python`, `p/security-audit`; advisory: it reports to the Security tab and does not fail the build), Trivy (fails on CRITICAL) and pip-audit (fails on CRITICAL) run on every pull request and push to `main`, beyond ruff and mypy. The jobs are required checks, so no change reaches `main`, and therefore no release, without running them.
Evidence: `.github/workflows/ci.yml:763-844` (Bandit), `:846-1023` (pip-audit), `:1025-1073` (Semgrep), `:1075-1170` (Trivy), `:1393` (Security Summary); branch protection required contexts; `pyproject.toml:393-413`.

**static_analysis_common_vulnerabilities** (SUGGESTED, N/A allowed). **Met.**
Bandit and Semgrep's `p/security-audit` rule set look for common vulnerabilities (injection, unsafe deserialization, weak crypto, hard-coded secrets); results are uploaded as SARIF to the Security tab.
Evidence: `.github/workflows/ci.yml:781-799`, `:1043-1066`.

**static_analysis_fixed** (MUST, N/A allowed). **Met.**
No confirmed medium-or-higher exploitable vulnerability from static analysis is outstanding. The Semgrep and Bandit findings were triaged in code with `# nosec` reasons or by fix; see G10 for the open Security-tab alerts, which are false positives or documented design and should be dismissed there. The fuzz and static findings fixed on 2026-09-30/10-01 are in `CHANGELOG.md:10-60`.
Evidence: `backpropagate/security.py:432` (`nosec B614`, `weights_only=True` default); `backpropagate/datasets.py:1910`; `backpropagate/eval.py:370,379`; `.github/workflows/ci.yml:801-831` (Bandit gate, clean on the release commit); `.github/workflows/ci.yml:1393-1440` (Semgrep is advisory: findings go to the Security tab as SARIF, see G10).

**static_analysis_often** (SUGGESTED, N/A allowed). **Met.**
Static analysis runs on every pull request and every push to `main`, plus a weekly Scorecard run.
Evidence: `.github/workflows/ci.yml:36-60`, `:763`, `:1025`; `.github/workflows/scorecard.yml:11-17`.

### Dynamic code analysis

**dynamic_analysis** (SUGGESTED). **Met.**
Two kinds of dynamic analysis run before releases: the test suite with 98.5% line-and-branch coverage (above the 80% the criterion accepts), and Atheris/libFuzzer fuzzing of the untrusted-input parsers (dataset rows and files, path sandbox, UI input validation, config parsing). The fuzzing was run, with 300,000 to 1,000,000 executions per harness, on the code that became v1.8.0 (2026-10-01). Harness properties also run as ordinary tests on every PR. The fuzz workflow itself is manual.
Evidence: `fuzz/`; `.github/workflows/fuzz.yml`; `tests/test_fuzz_harnesses.py`; PR #259 (Verification section); `CHANGELOG.md:135`; `pyproject.toml:373-391`.

**dynamic_analysis_unsafe** (SUGGESTED, N/A allowed). **N/A.**
The project is pure Python; no C or C++ is in the repository (searched for `*.c`, `*.cpp`, `*.pyx`, `*.rs`).
Evidence: `pyproject.toml:6-7`, `:232-233` (hatchling, pure-Python wheel); repository file search.

**dynamic_analysis_enable_assertions** (SUGGESTED). **Met.**
The dynamic analysis runs with assertions on: the pytest suite is assertion-based (assertion rewriting on), Hypothesis property tests and the Atheris harnesses assert invariants, and the library is not run with `python -O` in tests.
Evidence: `pyproject.toml:344-367` (`--strict-markers`); `tests/test_fuzz_harnesses.py`; `fuzz/fuzz_common.py`.

**dynamic_analysis_fixed** (MUST, N/A allowed). **Met.**
Fuzzing found nine defects (raw `TypeError`/`OverflowError`/`RecursionError` on hostile input, `sanitize_filename` returning `..`, a symlink-loop error leak, a redaction gap, and others); all were fixed in PR #259, merged 2026-10-01 06:28Z, about two hours after it was opened and before v1.8.0 shipped, each with a regression test. None had a CVE.
Evidence: `gh pr view 259` (findings F1-F9 and the extra `ENAMETOOLONG` case); `CHANGELOG.md:10-60`.

---

## Counts

- **Met: 63**
- **Unmet: 1** (`crypto_password_storage`, G1)
- **N/A: 3** (`vulnerability_report_response`, `crypto_pfs`, `dynamic_analysis_unsafe`)

MUST criteria that are Met on a reading the Director must confirm: `report_responses` (J1), `vulnerabilities_fixed_60_days` (J3), `know_secure_design` and `know_common_errors` (J4). MUST criteria that rest on a N/A: `vulnerability_report_response` (J2).
