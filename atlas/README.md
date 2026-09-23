# backpropagate: how it works

Mapped at 2026-09-23 from commit 34a7862.

## What this is

11 parts. Work enters through 9 doors; the busiest is CI, which reaches 4 parts.

## What changed since the last map

This is the first map.

## What comes in

1. **CI.** On a pull request; on a push to main touching 8 paths; or by hand. Runs backpropagate/, tests/ and verify.sh.
2. **Nightly Train Smoke.** On a schedule (`0 4 * * 1`), Monday at 04:00 UTC; or by hand. Runs scripts/nightly_train_smoke.py.
3. **Doc Drift Check.** On a pull request; on a push to main touching 7 paths; or by hand. Runs scripts/check_doc_drift.py.
4. **Mutation testing (mutmut).** By hand. Runs no file this map can see.
5. **OpenSSF Scorecard.** On a `branch_protection_rule` event; on a push to main; on a schedule (`0 6 * * 1`), Monday at 06:00 UTC; or by hand. Runs no file this map can see.
6. **Pages deploy.** On a push to main touching 2 paths; or by hand. Runs no file this map can see.
7. **Post-Publish Smoke.** When the workflow Publish completes; or by hand. Runs no file this map can see.
8. **Publish.** When a release is published; when the workflow Release completes; or by hand. Runs no file this map can see.
9. **Release.** When a tag matching `v*` is pushed; or by hand. Runs no file this map can see.

## What happens through CI

1. The workflow runs backpropagate/ in backpropagate, verify.sh in the repository root and tests/ in tests.
2. That reaches scripts (1 file).

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Nightly Train Smoke** runs scripts/nightly_train_smoke.py, reaches backpropagate, and opens an issue when it fails.

**Doc Drift Check** runs scripts/check_doc_drift.py.

**Mutation testing (mutmut)** runs no file this map can see, writes to .github/, commits .github/mutmut-baseline.txt and pushes, and opens a pull request.

**OpenSSF Scorecard** runs no file this map can see.

**Pages deploy** runs no file this map can see and deploys the site.

**Post-Publish Smoke** runs no file this map can see and opens an issue when it fails.

**Publish** runs no file this map can see and publishes to PyPI and a container image.

**Release** runs no file this map can see, publishes to npm, and creates a GitHub release.

## What breaks what

- **backpropagate** is imported by 1 part (scripts), and by 1 more only from tests; it sits on the path of 2 doors.
- **scripts** is imported only from tests, by 1 part (tests), and sits on the path of 3 doors.
- **CITATION.cff** is written by scripts and read by scripts; a hand edit reaches every reader.

## What tends to change together

- **backpropagate/cli.py** and **backpropagate/trainer.py** changed together in 28 of 53 commits, inside the backpropagate part.

2 files changed together with their own tests, as expected.

Window: 180 days; a pair counts from 10 shared commits.

## What no test touches

- **bin** is imported by no test.

## Written but never read

Every written place has a reader.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

- **.github/** is written by .github/workflows/mutmut.yml.
- **CITATION.cff** is written by scripts/prep_release.sh.

## Hand-authored

People write .claude/, assets/, docs/, examples/ and site/. Nothing in this repository writes to them.

## Where to start

.github/workflows/ci.yml → tests/ → scripts/check_doc_drift.py

Read those in order to follow one pull request end to end.

## What this map cannot see

- 40 import sites name a declared dependency that shares its name with a local module (datasets); they are read as the dependency, which is not in this repository.
- 6 import sites could not be resolved.
- 13 writes and 39 reads use paths built at run time and are not named here.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
