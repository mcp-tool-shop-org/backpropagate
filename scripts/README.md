# scripts/

Maintainer-side scripts that aren't part of the published wheel. Run them
from the repo root unless the script docstring says otherwise.

## `check_doc_drift.py`

Drift-detection forcing function for the
`[[within-swarm-doc-lie-drift-detection]]` doctrine. Fires on every PR via
`.github/workflows/doc-drift.yml`. Ten checks (the original four plus a
five-class extension landed in v1.4 — Class 4 has a 4b README variant):

1. Every `BACKPROPAGATE_*` env var read at runtime appears in both
   `site/src/content/docs/handbook/env-vars.md` AND the
   `_enumerate_env_vars` catalog in `backpropagate/cli.py`.
2. Every `--flag` registered in `cli.py` appears in
   `site/src/content/docs/handbook/cli-reference.md`.
3. Every public field on a `rx.State` subclass in `backpropagate/ui_state.py`
   has at least one consumer in `backpropagate/ui_app/`. Allow-list at
   `scripts/doc-drift-allow.toml` covers known false positives.
4. `tests/test_error_codes_catalog.py` exists and still asserts every
   `code='...'` literal is in `exceptions.ERROR_CODES`.
5. CLASS 1 — every `default=N` value in `cli.py` `add_argument` calls
   matches the documented default in the cli-reference.md row for that flag.
6. CLASS 2 — every `BACKPROPAGATE_*` token in `llms.txt` corresponds to a
   real runtime read in `backpropagate/**/*.py` (and the converse).
7. CLASS 3 — every `BACKPROPAGATE_*` row in `env-vars.md` that maps to a
   `Settings` field shows the same default the Pydantic field declares.
8. CLASS 4 — every handbook reference to a `BACKPROPAGATE_*` var, `--flag`,
   or error code inside the `error-codes.md` Fix column still points at a
   thing that exists.
9. CLASS 4b — every backtick-wrapped `PREFIX_...` error-code token in
   `README.md` drawn from a live `ERROR_CODES` prefix family is an actual
   key in `backpropagate/exceptions.py:ERROR_CODES`.
10. CLASS 5 — every severity claim (`LOW/LOW`, `MEDIUM/MEDIUM`, `HIGH`,
    etc.) in a workflow step name or comment matches the actual flag
    semantics on the next command line.

Run locally:

```bash
python scripts/check_doc_drift.py
```

Exit 0 means no drift. Exit 1 prints each finding with the missing
location.

### Pre-commit hook (v1.4 next step)

v1.3 ships CI-only. The script is stdlib-only (Python 3.10+) so a
pre-commit-friendly hook is a one-line wrapper away — wire under
`.pre-commit-config.yaml` once we adopt pre-commit in the repo. Until
then, run manually before opening a PR you want green on the first push.

## `convert_validated_pairs.py`

Reference one-off: converts a `validated_pairs.jsonl` to ShareGPT chat
format for SFT training. Path inputs are CLI args (the v1.0 vintage
hardcoded `F:/AI/...` paths from the maintainer's original rig; that
hardcoding was lifted as part of the v1.3 cleanup).

## `train_perfect_pairs.py`

Reference training driver for the perfect-pairs dataset. Same path-arg
treatment as `convert_validated_pairs.py`.

## `repin_test_count.sh`

Print the current `pytest --collect-only` test count (and the UTC date) so
you can re-pin it across the canonical pin sites. Intentionally read-only —
it modifies nothing; you Edit each doc by hand so the surrounding prose stays
tuned. The pinned surfaces are `CHANGELOG.md`, `CLAUDE.md`,
`PRODUCTION_READINESS_AUDIT.md`, and `SECURITY_AUDIT_REPORT.md` (the README
does not pin a test count). Run after a wave that intentionally adds or
removes tests so those four surfaces track reality.

## `run_mutmut.sh`

Mutation-testing driver. Long-running — kicks off `mutmut run` against the
core modules and writes the surviving-mutants report to stdout.
