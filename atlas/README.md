# backpropagate: how it works

Mapped at 2026-10-01 from commit 7deb5b5 by Atlas 1.24.0.

## What this is

11 parts, mostly Python (143 files), shell (8), Astro (2), CSS (2), JavaScript (2) and TypeScript (2). Work enters through 12 doors; the busiest is CI, which reaches 4 parts. It publishes to npm and PyPI, and a container image. It deploys a site to GitHub Pages. People run backprop and backpropagate.

## What changed since 2026-10-01 (128a3b9)

Nothing structural changed since 2026-10-01; 1 file added and 3 changed content.

## What comes in

1. **CI.** On a pull request to main; on a push to main touching 9 paths; or by hand. Runs tests/ and verify.sh; checks backpropagate/.
2. **Publish.** When a release is published; when the workflow Release completes; or by hand. Runs backpropagate/cli.py; checks backpropagate/; packs LICENSE, README.md and pyproject.toml into an image.
3. **Nightly Train Smoke.** On a schedule (`0 4 * * 1`), Monday at 04:00 UTC; or by hand. Runs scripts/nightly_train_smoke.py.
4. **Doc Drift Check.** On a pull request to main; on a push to main touching 7 paths; or by hand. Runs scripts/check_doc_drift.py.
5. **Pages deploy.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
6. **Post-Publish Smoke.** When the workflow Publish completes; or by hand. Runs backpropagate/cli.py.
7. **Mutation testing (mutmut).** By hand. Runs no file this map can see.
8. **OpenSSF Scorecard.** On a `branch_protection_rule` event; on a push to main; on a schedule (`0 6 * * 1`), Monday at 06:00 UTC; or by hand. Runs no file this map can see.
9. **Release.** When a tag matching `v*` is pushed; or by hand. Runs no file this map can see.
10. **backprop** (a command people run). Runs backpropagate/cli.py.
11. **backpropagate** (a command people run, from package.json). Runs bin/backpropagate.js.
12. **backpropagate** (a command people run, from pyproject.toml). Runs backpropagate/cli.py.

## What happens through CI

1. The workflow runs verify.sh in the repository root and tests/ in tests; it checks backpropagate/ in backpropagate.
2. That reaches scripts (1 file).
3. It uploads coverage to Codecov.
4. It scans code with CodeQL.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Publish** runs backpropagate/cli.py, checks backpropagate/, packs LICENSE, README.md and pyproject.toml into an image, and publishes to PyPI and a container image.

**Nightly Train Smoke** runs scripts/nightly_train_smoke.py, reaches backpropagate, and opens an issue when it fails.

**Doc Drift Check** runs scripts/check_doc_drift.py.

**Pages deploy** runs site/astro.config.mjs and site/src/, and deploys the site.

**Post-Publish Smoke** runs backpropagate/cli.py and opens an issue when it fails.

**Mutation testing (mutmut)** runs no file this map can see, commits .github/mutmut-baseline.txt and pushes to a branch for review, never to main, and opens a pull request.

**OpenSSF Scorecard** runs no file this map can see and scans code with CodeQL.

**Release** runs no file this map can see, publishes to npm, creates a GitHub release, and uploads backpropagate-npm-sbom.cdx.json and backpropagate-sbom.cdx.json to the release.

**backprop** (a command people run) runs backpropagate/cli.py.

**backpropagate** (a command people run, from package.json) runs bin/backpropagate.js.

**backpropagate** (a command people run, from pyproject.toml) runs backpropagate/cli.py.

## What breaks what

- **backpropagate** is imported by 1 part (scripts), and by 1 more only from tests; it sits on the path of 6 doors.
- **scripts** is imported only from tests, by 1 part (tests), and sits on the path of 3 doors.
- **the repository root** is imported by no other part and sits on the path of 2 doors.
- **CITATION.cff** is written by scripts and read by scripts; a hand edit reaches every reader.

## What tends to change together

- **backpropagate/cli.py** and **backpropagate/trainer.py** changed together in 31 of 60 commits, inside the backpropagate part.

2 files changed together with their own tests, as expected.

Window: 180 days; a pair counts from 10 shared commits, since 20 source files reach 10 revisions; the floor falls to 3 when fewer than 20 do.

## What no test touches

- **bin** is imported by no test.

## Written but never read

Every written place has a reader.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

- **CITATION.cff** is written by scripts/prep_release.sh.

## Hand-authored

People write .claude/, .github/, assets/, docs/, examples/ and site/; 6 writes with paths built at run time may land here.

## Where to start

backpropagate/cli.py → backpropagate/logging_config.py

Read those in order to follow one run of backprop end to end. This path follows backprop (a command people run) from its entry, since CI runs only tests and checks.

## What this map cannot see

- 43 import sites name a declared dependency that shares its name with a local module (datasets); they are read as the dependency, which is not in this repository.
- 8 imports could not be resolved: `backpropagate/trainer.py` imports a path built at run time; `tests/test_fp8_smoke.py` imports a path built at run time; `tests/test_full_ft_offload_smoke.py` imports a path built at run time; and 5 more.
- 6 writes and 10 reads use paths built at run time and are not named here.
- 31 writes and 64 reads go to a path their caller passes, not to this repository.
- 1 write goes to the home directory (AppData/, Library/ and backpropagate/) or a path its caller passes, not to this repository.
- 1 read goes to the directory the command is run in, not to this repository.
- 1 read goes to the home directory (.cache/), not to this repository.
- 1 write goes to a temporary directory, not to this repository.
- There is a compose.yaml that no workflow runs; what deploys from it does so from outside this repository, and is not on this page.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
