# scripts/

Maintainer-side scripts that aren't part of the published wheel. Run them
from the repo root unless the script docstring says otherwise.

## `check_doc_drift.py`

Drift-detection forcing function for the
`[[within-swarm-doc-lie-drift-detection]]` doctrine. Fires on every PR via
`.github/workflows/doc-drift.yml`. Four checks:

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

Pin the README test-count badge to the current `pytest --collect-only`
output. Run after a wave that intentionally adds or removes tests so the
README's "1865 tests" line tracks reality.

## `run_mutmut.sh`

Mutation-testing driver. Long-running — kicks off `mutmut run` against the
core modules and writes the surviving-mutants report to stdout.
