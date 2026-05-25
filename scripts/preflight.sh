#!/usr/bin/env bash
# scripts/preflight.sh — shipcheck-style pre-flight before tag-push.
#
# v1.4 Wave 6a (CIDOCS-F-024 / V1_4_BRIEF item 11): the local mirror of
# every gate CI runs. The release-spine has earned multiple "I tagged
# but the CI strictness changed under me" classes of mistake; this is
# the single command the maintainer runs to confirm the local checkout
# would pass every gate CI runs on push.
#
# Distinction from prep_release.sh
# --------------------------------
# - prep_release.sh runs at release time and MUTATES files (bumps
#   CITATION.cff, regenerates translations, rebuilds dist/, etc.).
# - preflight.sh is a READ-ONLY gate. It runs every CI gate locally
#   without mutating any source-of-truth file. Safe to run on any
#   working tree — including a dirty one mid-feature.
#
# What this checks (in order)
# ---------------------------
#   1. Drift gate                — scripts/check_doc_drift.py
#   2. Lint                      — ruff check backpropagate/
#   3. Type check                — mypy backpropagate/
#   4. Tests                     — pytest (filter: not gpu / slow / integration)
#   5. Build                     — python -m build
#   6. Twine + PKG-INFO smoke    — twine check + PKG-INFO grep
#   7. Bandit gating run         — bandit -l -i (LOW/LOW floor)
#   8. CITATION.cff version sync — pyproject.toml ↔ CITATION.cff
#   9. Workflow severity-claim   — drift gate Class 5 (re-asserted)
#
# Stages 1-6 mirror verify.sh + the ci.yml build job. Stages 7-9 add
# the gates that live outside verify.sh — Bandit (in ci.yml security
# scan) + the version-sync check that release.yml performs at tag time.
#
# Usage
# -----
#   scripts/preflight.sh           # run every gate
#   scripts/preflight.sh --quick   # skip the slow gates (mypy + bandit)
#                                  # for the fast inner-loop pre-commit
#   scripts/preflight.sh --help
#
# Exit codes
# ----------
#   0 — every gate passed.
#   1 — at least one gate failed.
#   2 — bad flag / usage.
set -euo pipefail

QUICK=0
for arg in "$@"; do
  case "${arg}" in
    --quick) QUICK=1 ;;
    -h|--help)
      sed -n '2,50p' "$0"
      exit 0
      ;;
    *)
      echo "preflight.sh: unknown argument: ${arg}" >&2
      echo "preflight.sh: supported flags: --quick, --help" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OVERALL_RC=0
FAILED_STAGES=()

run_gate() {
  # run_gate <name> <command...>
  local name="$1"; shift
  echo ""
  echo "=== ${name} ==="
  if "$@"; then
    echo "  ✓ ${name}"
  else
    rc=$?
    echo "  ✗ ${name} (exit ${rc})" >&2
    FAILED_STAGES+=("${name}")
    OVERALL_RC=1
  fi
}

# ---------------------------------------------------------------------------
# Stage 1: drift gate
# ---------------------------------------------------------------------------

run_gate "drift gate (Class 1-9)" python3 scripts/check_doc_drift.py

# ---------------------------------------------------------------------------
# Stage 2: lint
# ---------------------------------------------------------------------------

run_gate "lint (ruff)" ruff check backpropagate/

# ---------------------------------------------------------------------------
# Stage 3: type check (skippable in --quick)
# ---------------------------------------------------------------------------

if [ "${QUICK}" = "1" ]; then
  echo ""
  echo "=== type check (mypy) ==="
  echo "  - skipped (--quick)"
else
  run_gate "type check (mypy)" mypy backpropagate/ --ignore-missing-imports
fi

# ---------------------------------------------------------------------------
# Stage 4: tests
# ---------------------------------------------------------------------------

run_gate "tests (pytest)" pytest tests/ -x --timeout=60 -m "not gpu and not slow and not integration"

# ---------------------------------------------------------------------------
# Stage 5: build
# ---------------------------------------------------------------------------

# Don't pollute the working tree's dist/ if it's already there with a
# release artefact. Use a temp dir.
PREFLIGHT_DIST_DIR=$(mktemp -d)
trap 'rm -rf "${PREFLIGHT_DIST_DIR}"' EXIT

run_gate "build (python -m build)" python3 -m build --outdir "${PREFLIGHT_DIST_DIR}"

# ---------------------------------------------------------------------------
# Stage 6: twine + PKG-INFO smoke
# ---------------------------------------------------------------------------

run_gate "twine check" twine check "${PREFLIGHT_DIST_DIR}"/*

echo ""
echo "=== PKG-INFO metadata smoke ==="
SDIST_PATH=$(ls "${PREFLIGHT_DIST_DIR}"/backpropagate-*.tar.gz 2>/dev/null | head -1 || echo "")
if [ -z "${SDIST_PATH}" ]; then
  echo "  ✗ no sdist produced by build" >&2
  FAILED_STAGES+=("PKG-INFO metadata smoke")
  OVERALL_RC=1
else
  PKGINFO=$(tar -xOf "${SDIST_PATH}" --wildcards '*/PKG-INFO' 2>/dev/null || echo "")
  MISSING=()
  if ! echo "${PKGINFO}" | grep -qE '^(License-Expression|License):'; then
    MISSING+=("License-Expression or License")
  fi
  if ! echo "${PKGINFO}" | grep -qE '^Author(-Email)?:'; then
    MISSING+=("Author or Author-Email")
  fi
  if ! echo "${PKGINFO}" | grep -qE '^Version:'; then
    MISSING+=("Version")
  fi
  if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  ✗ PKG-INFO missing: ${MISSING[*]}" >&2
    FAILED_STAGES+=("PKG-INFO metadata smoke")
    OVERALL_RC=1
  else
    echo "  ✓ PKG-INFO metadata smoke"
  fi
fi

# ---------------------------------------------------------------------------
# Stage 7: Bandit (skippable in --quick)
# ---------------------------------------------------------------------------

if [ "${QUICK}" = "1" ]; then
  echo ""
  echo "=== Bandit (LOW/LOW gating) ==="
  echo "  - skipped (--quick)"
else
  # Match the ci.yml ``-l -i`` flag set so local + CI semantics agree.
  # The doc-lie fixed in V1_4_BRIEF item 4 — those flags ARE LOW/LOW.
  run_gate "Bandit (LOW/LOW gating)" bandit -r backpropagate/ -c pyproject.toml -l -i -f txt
fi

# ---------------------------------------------------------------------------
# Stage 8: CITATION.cff version sync (pyproject ↔ CITATION)
# ---------------------------------------------------------------------------

echo ""
echo "=== CITATION.cff version sync ==="
PYPROJECT_VERSION=$(python3 -c "import tomllib; print(tomllib.loads(open('pyproject.toml').read())['project']['version'])")
CITATION_VERSION=$(awk '/^version:/{print $2; exit}' CITATION.cff | tr -d '"' || echo "")
if [ "${PYPROJECT_VERSION}" != "${CITATION_VERSION}" ]; then
  echo "  ✗ pyproject.toml=${PYPROJECT_VERSION} but CITATION.cff=${CITATION_VERSION}" >&2
  echo "    Run scripts/prep_release.sh to bump both in lockstep before tagging." >&2
  FAILED_STAGES+=("CITATION.cff version sync")
  OVERALL_RC=1
else
  echo "  ✓ pyproject.toml ↔ CITATION.cff version both at ${PYPROJECT_VERSION}"
fi

# ---------------------------------------------------------------------------
# Stage 9: package.json version sync (pyproject ↔ package.json)
# ---------------------------------------------------------------------------

echo ""
echo "=== package.json version sync ==="
if [ -f package.json ]; then
  NPM_VERSION=$(python3 -c "import json; print(json.load(open('package.json'))['version'])" 2>/dev/null || echo "")
  if [ -z "${NPM_VERSION}" ]; then
    echo "  - package.json present but version field unreadable; skipping check" >&2
  elif [ "${PYPROJECT_VERSION}" != "${NPM_VERSION}" ]; then
    echo "  ✗ pyproject.toml=${PYPROJECT_VERSION} but package.json=${NPM_VERSION}" >&2
    echo "    release.yml gates on pyproject ↔ package.json version parity at tag time." >&2
    FAILED_STAGES+=("package.json version sync")
    OVERALL_RC=1
  else
    echo "  ✓ pyproject.toml ↔ package.json version both at ${PYPROJECT_VERSION}"
  fi
else
  echo "  - no package.json (npm-launcher not wired); skipping" >&2
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "=== preflight summary ==="
if [ ${OVERALL_RC} -eq 0 ]; then
  echo "  ✓ all gates passed — safe to commit + tag"
else
  echo "  ✗ ${#FAILED_STAGES[@]} gate(s) failed:" >&2
  for stage in "${FAILED_STAGES[@]}"; do
    echo "      - ${stage}" >&2
  done
fi

exit ${OVERALL_RC}
