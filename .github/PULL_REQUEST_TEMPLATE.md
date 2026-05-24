<!--
Thanks for contributing to backpropagate! Please fill in the sections
below — the swarm PRs from v1.3 (#92 / #93 / #95 / #103 / #104) are
good reference shape if you want an example.
-->

## Summary

<!-- One paragraph: what changed, and why. Lead with the user-facing
intent (operator pain closed / feature added / regression fixed), not
the file list. -->

## Test plan

<!-- Bulleted checklist of what you ran locally. Prefer concrete commands.
At minimum, this repo expects ruff + mypy + pytest + drift to pass before
a PR is opened. -->

- [ ] `ruff check backpropagate/`
- [ ] `mypy backpropagate/ --ignore-missing-imports`
- [ ] `pytest tests/ -m "not gpu and not slow and not integration"`
- [ ] `python scripts/check_doc_drift.py`
- [ ] Targeted regression set (name which tests/files exercise the change)

## Breaking changes

<!-- Yes / No. If yes, list the operator-facing surface that changed and
the migration path. A breaking change to a CLI flag, env var, or public
API must update CHANGELOG.md AND the v1.x → next-major migration page in
the handbook. -->

- [ ] No breaking changes (default)
- [ ] Breaking change — listed in CHANGELOG.md `### Changed` / `### Removed`
      AND in `site/src/content/docs/handbook/migrations.md`

## Related issues / advisories / brief

<!-- Link any GitHub issue, GHSA advisory, or V1_3_BRIEF / V1_4_BRIEF
item this PR closes or advances. Format: `Closes #N`, `Closes GHSA-xxxx`,
`Advances V1_3_BRIEF item BACKEND-F-007`. -->

## Doctrine touchpoints

<!-- The drift gate (`scripts/check_doc_drift.py`, runs on every PR) WILL
fire if any of the following are touched without the matching update. Tick
the boxes that apply so the reviewer can verify the doc surface in one
glance. -->

- [ ] Adds / renames / removes a `BACKPROPAGATE_*` env var
  - [ ] Updated `site/src/content/docs/handbook/env-vars.md`
  - [ ] Updated `_enumerate_env_vars` catalog in `backpropagate/cli.py`
- [ ] Adds / renames / removes a `--flag`
  - [ ] Updated `site/src/content/docs/handbook/cli-reference.md`
- [ ] Adds a new `code='...'` literal at a raise site
  - [ ] Added the code to `backpropagate/exceptions.py` `ERROR_CODES`
  - [ ] Updated `site/src/content/docs/handbook/error-codes.md`
- [ ] Touches `backpropagate/ui_app/auth.py` or the auth middleware
      surface
  - [ ] `tests/test_auth_middleware.py` still passes 23/3
  - [ ] Reviewer asked to look at the threat model (see SECURITY.md +
        `handbook/security.md`)
- [ ] Changes coverage floor or test count
  - [ ] Updated `pyproject.toml` `[tool.coverage.report].fail_under`
        and / or ran `scripts/repin_test_count.sh`

## Notes for the reviewer

<!-- Anything else worth knowing: rollback plan, follow-ups deferred to a
later PR, dependencies on a sister PR in another repo, secrets / settings
changes the maintainer needs to apply by hand. -->
