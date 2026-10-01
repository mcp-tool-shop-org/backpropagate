# backpropagate: how it works

Mapped at 2026-10-01 from commit d09f531 by Atlas 1.24.0.

## What this is

13 parts, mostly Python (221 files), shell (11), Astro (2), CSS (2), JavaScript (2) and TypeScript (2). Work enters through 13 doors; the busiest is CI, which reaches 6 parts. It publishes to npm and PyPI, and a container image. It deploys a site to GitHub Pages. People run backprop and backpropagate.

## What changed since 2026-10-01 (a4b0e1a)

- .gitignore is now read by tests/test_container_ui.py.
- CHANGELOG.md is now read by .github/workflows/release.yml.
- 3 files added and 16 changed content, across 5 parts.

## What comes in

1. **CI.** On a pull request to main; on a push to main touching 11 paths; or by hand. Runs backpropagate/cli.py, scripts/ci_install_locked.sh, tests/ and 1 more; checks backpropagate/, requirements/build-backend.txt and requirements/uv.txt; packs LICENSE, README.md, docker/fetch_bun.py and 2 more into an image.
2. **Publish.** When a release is published; when the workflow Release completes; or by hand. Runs backpropagate/cli.py; checks backpropagate/, requirements/build-backend.txt and requirements/uv.txt; packs LICENSE, README.md, docker/fetch_bun.py and 2 more into an image.
3. **Release.** When a tag matching `v*` is pushed; or by hand. Runs scripts/ci_install_locked.sh; checks pyproject.toml.
4. **Nightly Train Smoke.** On a schedule (`0 4 * * 1`), Monday at 04:00 UTC; or by hand. Runs scripts/nightly_train_smoke.py.
5. **Doc Drift Check.** On a pull request to main; on a push to main touching 7 paths; or by hand. Runs scripts/check_doc_drift.py.
6. **Mutation testing (mutmut).** By hand. Runs scripts/ci_install_locked.sh.
7. **Pages deploy.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
8. **Post-Publish Smoke.** When the workflow Publish completes; or by hand. Runs backpropagate/cli.py.
9. **Fuzz.** By hand. Runs no file this map can see.
10. **OpenSSF Scorecard.** On a `branch_protection_rule` event; on a push to main; on a schedule (`0 6 * * 1`), Monday at 06:00 UTC; or by hand. Runs no file this map can see.
11. **backprop** (a command people run). Runs backpropagate/cli.py.
12. **backpropagate** (a command people run, from package.json). Runs bin/backpropagate.js.
13. **backpropagate** (a command people run, from pyproject.toml). Runs backpropagate/cli.py.

## What happens through CI

1. The workflow runs backpropagate/cli.py in backpropagate, verify.sh in the repository root, scripts/ci_install_locked.sh in scripts and tests/ in tests; it checks backpropagate/ in backpropagate and requirements/build-backend.txt and requirements/uv.txt in requirements; it packs 4 files in the repository root and docker/fetch_bun.py into an image.
   1. Inside backpropagate/cli.py, `main` does, in order: `logging_config.py` (5 steps).
2. That reaches fuzz (6 files).
3. It uploads coverage to Codecov.
4. It scans code with CodeQL.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Publish** runs backpropagate/cli.py, checks backpropagate/, requirements/build-backend.txt and requirements/uv.txt, packs LICENSE, README.md, docker/fetch_bun.py and 2 more into an image, publishes to PyPI and a container image, and uploads dist/* and files named at run time to the release.

**Release** runs scripts/ci_install_locked.sh, checks pyproject.toml, publishes to npm, creates a GitHub release, and uploads backpropagate-npm-sbom.cdx.json and backpropagate-sbom.cdx.json to the release.

**Nightly Train Smoke** runs scripts/nightly_train_smoke.py, reaches backpropagate, and opens an issue when it fails.

**Doc Drift Check** runs scripts/check_doc_drift.py.

**Mutation testing (mutmut)** runs scripts/ci_install_locked.sh.

**Pages deploy** runs site/astro.config.mjs and site/src/, and deploys the site.

**Post-Publish Smoke** runs backpropagate/cli.py and opens an issue when it fails.

**Fuzz** runs no file this map can see.

**OpenSSF Scorecard** runs no file this map can see and scans code with CodeQL.

**backprop** (a command people run) runs backpropagate/cli.py.

**backpropagate** (a command people run, from package.json) runs bin/backpropagate.js.

**backpropagate** (a command people run, from pyproject.toml) runs backpropagate/cli.py.

## What breaks what

- **backpropagate** is imported by 2 parts (fuzz, scripts), and by 1 more only from tests; it sits on the path of 6 doors.
- **scripts** is imported only from tests, by 1 part (tests), and sits on the path of 5 doors.
- **the repository root** is imported by no other part and sits on the path of 3 doors.
- **requirements** is imported by no other part and sits on the path of 2 doors.
- **fuzz** is imported only from tests, by 1 part (tests), and sits on the path of 1 door.
- **CITATION.cff** is written by scripts and read by scripts; a hand edit reaches every reader.

## What tends to change together

No two source files, other than a file and its own test, changed together often enough to name.

2 files changed together with their own tests, as expected.

Window: 180 days; a pair counts from 10 shared commits, since 23 source files reach 10 revisions; the floor falls to 3 when fewer than 20 do.

## What no test touches

- **bin** is imported by no test.

## Written but never read

Every written place has a reader.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

- **CITATION.cff** is written by scripts/prep_release.sh.

## Hand-authored

People write .claude/, .github/, assets/, docs/, examples/, requirements/ and site/; 8 writes with paths built at run time may land here.

## Where to start

.github/workflows/ci.yml → backpropagate/cli.py → backpropagate/logging_config.py

Read those in order to follow one pull request end to end.

## What this map cannot see

- 56 import sites name a declared dependency that shares its name with a local module (datasets); they are read as the dependency, which is not in this repository.
- 15 imports could not be resolved: `backpropagate/cli.py` imports `.ui_workdir`, which is no module on its import path and no declared dependency; `backpropagate/trainer.py` imports a path built at run time; `tests/test_fp8_smoke.py` imports a path built at run time; and 12 more.
- 8 writes and 10 reads use paths built at run time and are not named here.
- 31 writes and 64 reads go to a path their caller passes, not to this repository.
- 3 writes go to a temporary directory, not to this repository.
- 1 write goes to the home directory (AppData/, Library/ and backpropagate/) or a path its caller passes, not to this repository.
- 1 read goes to the directory the command is run in, not to this repository.
- 1 read goes to the home directory (.cache/), not to this repository.
- There is a compose.yaml that no workflow runs; what deploys from it does so from outside this repository, and is not on this page.
- 1 file belongs to no part: docker/fetch_bun.py.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
