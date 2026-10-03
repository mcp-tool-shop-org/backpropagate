# Ship Gate

> No repo is "done" until every applicable line is checked.
> Copy this into your repo root. Check items off per-release.

**Tags:** `[all]` every repo · `[npm]` `[pypi]` `[vsix]` `[desktop]` `[container]` published artifacts · `[mcp]` MCP servers · `[cli]` CLI tools

---

## A. Security Baseline

- [x] `[all]` SECURITY.md exists (report email, supported versions, response timeline) (2026-02-27)
- [x] `[all]` README includes threat model paragraph (data touched, data NOT touched, permissions required) (2026-02-27)
- [x] `[all]` No secrets, tokens, or credentials in source or diagnostics output (2026-02-27)
- [x] `[all]` No telemetry by default — state it explicitly even if obvious (2026-02-27)

### Default safety posture

- [x] `[cli|mcp|desktop]` Dangerous actions (kill, delete, restart) require explicit `--allow-*` flag (2026-10-03) — code execution in `backprop eval --metric pass_rate` needs `--allow-code-exec`; the web UI's delete-cached-model and Clean up actions ask for confirmation and nothing is deleted automatically
- [x] `[cli|mcp|desktop]` File operations constrained to known directories (2026-02-27) — safe_path() with traversal protection
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[mcp]` SKIP: not an MCP server

## B. Error Handling

- [x] `[all]` Errors follow the Structured Error Shape: `code`, `message`, `hint`, `cause?`, `retryable?` (2026-02-27) — exception hierarchy with message/details/suggestion
- [x] `[cli]` Exit codes: 0 ok · 1 user error · 2 runtime error · 3 partial success (2026-02-27)
- [x] `[cli]` No raw stack traces without `--debug` (2026-02-27) — only with --verbose
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[mcp]` SKIP: not an MCP server
- [x] `[desktop]` Errors shown as user-friendly messages — no raw exceptions in UI (2026-10-03) — the web UI (pip and Store) shows the error code with a plain-language hint and the last log lines; refusals are redacted and capped; no stack traces
- [ ] `[vscode]` SKIP: not a VS Code extension

## C. Operator Docs

- [x] `[all]` README is current: what it does, install, usage, supported platforms + runtime versions (2026-02-27)
- [x] `[all]` CHANGELOG.md (Keep a Changelog format) (2026-02-27)
- [x] `[all]` LICENSE file present and repo states support status (2026-02-27)
- [x] `[cli]` `--help` output accurate for all commands and flags (2026-02-27)
- [ ] `[cli|mcp|desktop]` SKIP: --verbose flag exists; formal logging level tiers not applicable for a training library
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[complex]` SKIP: not operationally complex

## D. Shipping Hygiene

- [x] `[all]` `verify` script exists (test + build + smoke in one command) (2026-02-27) — verify.sh
- [x] `[all]` Version in manifest matches git tag (2026-06-20) — 1.7.0 = v1.7.0
- [x] `[all]` Dependency scanning runs in CI (ecosystem-appropriate) (2026-02-27) — Bandit, pip-audit, Semgrep, Trivy (built-in secret scanner). TruffleHog removed v1.1.0 — see docs/ci-gates-triage-plan.md.
- [x] `[all]` Automated dependency update mechanism exists (2026-02-27) — dependabot.yml monthly + groups
- [x] `[npm]` PASS: @mcptoolshop/backpropagate@1.2.0 published with Sigstore provenance (2026-05-23). v1.3 retires the npm distribution path — the package remains published as a friendly-error shim redirecting operators to pipx / uv tool / pip per `bin/backpropagate.js`.
- [x] `[pypi]` `python_requires` set (2026-02-27) — >=3.10
- [x] `[pypi]` Clean wheel + sdist build (2026-02-27) — hatchling, twine check in CI
- [ ] `[vsix]` SKIP: not a VS Code extension
- [x] `[desktop]` Installer/package builds and runs on stated platforms (2026-10-03) — the Store MSIX is built by `scripts/build_msix.py` behind gates (CUDA op on the build GPU, llama.cpp manifest, size, MAX_PATH), registered and launched on Windows 11, WACK overall warning with no hard failures (issue #310 lists the optional findings)

## E. Identity (soft gate — does not block ship)

- [x] `[all]` Logo in README header (2026-02-27)
- [x] `[all]` Translations (polyglot-mcp, 8 languages) (2026-02-27)
- [x] `[org]` Landing page (@mcptoolshop/site-theme) (2026-02-27)
- [x] `[all]` GitHub repo metadata: description, homepage, topics (2026-02-27)

---

## Gate Rules

**Hard gate (A–D):** Must pass before any version is tagged or published.
If a section doesn't apply, mark `SKIP:` with justification — don't leave it unchecked.

**Soft gate (E):** Should be done. Product ships without it, but isn't "whole."

**Checking off:**
```
- [x] `[all]` SECURITY.md exists (2026-02-27)
```

**Skipping:**
```
- [ ] `[pypi]` SKIP: not a Python project
```
