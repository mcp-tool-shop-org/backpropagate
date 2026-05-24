# Backpropagate Security Audit & Implementation Report (archived)

> **Status (2026-05-23):** This document was a point-in-time audit from early
> development (January 2026, captured against the v1.0.2 baseline). The body
> referenced modules and APIs that have since been removed or replaced:
>
> - `backpropagate/ui.py` was migrated to `backpropagate/ui_app/` (Reflex /
>   Radix UI) in v1.1.0 and the Gradio legacy modules were removed in v1.2.0.
> - `from backpropagate import launch` now raises `ImportError` via
>   `__init__.py.__getattr__` with a migration message pointing at
>   `backprop ui`.
> - Gradio-style `auth=("user", "pass")` was replaced by the real ASGI
>   auth middleware (Starlette via `rx.App(api_transformer=...)`) that
>   closed [GHSA-f65r-h4g3-3h9h][ghsa] in v1.2.0.
> - The test-count line is superseded by `pytest --collect-only` (1865
>   tests on `main` as of 2026-05-23 post-Wave-6 / -7 / -8).
>
> Retained as a historical artifact only. The current security posture lives
> in two places:
>
> - **[SECURITY.md](SECURITY.md)** — vulnerability reporting policy +
>   advisory list (production-facing).
> - **[handbook/security.md][handbook]** — threat model, four-layer defense
>   in depth, auth-middleware mode matrix, SSH port-forwarding recipe,
>   GHSA-f65r-h4g3-3h9h post-mortem.
>
> [ghsa]: https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/GHSA-f65r-h4g3-3h9h
> [handbook]: https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/

---

## What used to live here (summary)

The original report documented the v1.0.2 security hardening pass:

- `backpropagate/security.py` — `safe_path()`, `safe_torch_load()`,
  `check_torch_security()`, `audit_log()`, `PathTraversalError`.
- CLI export-path validation, SLAO `torch.load` version check, Web UI
  auth/rate-limit/input-sanitization (Gradio era — the rate limiter
  carried forward as `EnhancedRateLimiter` / `FileValidator` and now
  protects the Reflex UI surface).
- `bandit`, `pip-audit`, `safety` added to dev dependencies.
- 22 + 21 new tests for the security + UI security modules (these counts
  are pre-v1.1.x reorganizations; the current security test surface is
  `tests/test_security.py` + `tests/test_ui_security.py` +
  `tests/test_auth_middleware.py` + the auth-bypass regression suite).

All "Should Do" recommendations from the original report (regular bandit
runs, pre-commit hooks, dependency scanning) shipped in subsequent waves
and are now CI-gated per `docs/ci-gates-triage-plan.md` (with v1.2.0
re-tightening the floor gates to CRITICAL after the v1.1.x rollback).

---

*If you need the verbatim original audit text it is preserved in the git
history at the v1.0.2 tag.*
