#!/usr/bin/env bash
# scripts/prep_release.sh — pre-release coordinator for backpropagate.
#
# v1.4 Wave 6a (CIDOCS-F-025 / V1_4_BRIEF item 6): the canonical pre-release
# checklist runner. Runs every checklist item that has to land BEFORE
# `npm publish` / `gh release create` so the GitHub Release tag points
# at a commit where the version metadata, translations, drift gate,
# verify.sh, and PyPI metadata smoke are all in lockstep.
#
# Why this exists
# ---------------
# Releases are immutable — the GitHub release tag is created against a
# specific commit. If translations or CITATION.cff land in a follow-up
# commit, the released tag carries stale documentation forever. This
# script collapses the "I'll fix the translations after publish" failure
# mode into a single gate that runs BEFORE the publish chain.
#
# Order of operations (matches the [[npm-publish-release-ordering]] memo
# at memory: translations-must-run-before-npm-publish):
#
#   1. CITATION.cff version + date-released bump (auto-extracted from
#      pyproject.toml — the source of truth for the release version).
#   2. Reference list validation (every references: entry has a real
#      arXiv / DOI URL — catches LLM-paraphrased placeholder authors).
#   3. Translation invocation via polyglot-mcp's TranslateGemma 12B
#      (zero API cost, runs locally on Ollama). README.* refreshed
#      for ja / zh / es / fr / hi / it / pt-BR.
#   4. Build sdist + wheel via `python -m build`.
#   5. PyPI metadata smoke — `twine check dist/*` AND PKG-INFO grep
#      for License-Expression + Author (mirrors the CI check at
#      ci.yml:build).
#   6. Drift gate — `python scripts/check_doc_drift.py` must exit 0.
#   7. verify.sh --format=json — every stage must pass.
#
# Usage
# -----
#   scripts/prep_release.sh             # run all stages
#   scripts/prep_release.sh --dry-run   # narrate each stage without
#                                       # mutating files / running
#                                       # network steps (no translation
#                                       # invocation, no build)
#   scripts/prep_release.sh --help
#
# Exit codes
# ----------
#   0 — all stages passed; safe to `git tag` + `git push --tags`.
#   1 — at least one stage failed; see the stage banner in stderr.
#   2 — bad flag / usage.
set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,60p' "$0"
      exit 0
      ;;
    *)
      echo "prep_release.sh: unknown argument: ${arg}" >&2
      echo "prep_release.sh: supported flags: --dry-run, --help" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Pretty banner: each stage prints a line to stderr so the dry-run
# narration is easy to read.
banner() {
  printf '\n=== %s ===\n' "$1" >&2
}

# Dry-run short-circuit. Mutating actions check `${DRY_RUN}` before
# acting; reads (version extraction, file scans) still execute so the
# dry-run output is informative.
maybe_run() {
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] would run: $*" >&2
  else
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# Stage 1: extract the release version from pyproject.toml
# ---------------------------------------------------------------------------

banner "Stage 1: extract release version from pyproject.toml"

if ! VERSION=$(python3 -c "import tomllib; print(tomllib.loads(open('pyproject.toml').read())['project']['version'])"); then
  echo "::error::failed to read [project].version from pyproject.toml" >&2
  exit 1
fi
TODAY=$(date -u +%Y-%m-%d)
echo "  pyproject.toml [project].version = ${VERSION}" >&2
echo "  today (UTC)                       = ${TODAY}" >&2

# ---------------------------------------------------------------------------
# Stage 2: bump CITATION.cff version + date-released
# ---------------------------------------------------------------------------

banner "Stage 2: bump CITATION.cff version + date-released"

# Read the current CITATION.cff values so dry-run can narrate.
CITATION_VERSION=$(awk '/^version:/{print $2; exit}' CITATION.cff | tr -d '"' || echo "")
CITATION_DATE=$(awk '/^date-released:/{print $2; exit}' CITATION.cff | tr -d '"' || echo "")
echo "  current CITATION.cff version       = ${CITATION_VERSION}" >&2
echo "  current CITATION.cff date-released = ${CITATION_DATE}" >&2

if [ "${CITATION_VERSION}" = "${VERSION}" ] && [ "${CITATION_DATE}" = "${TODAY}" ]; then
  echo "  CITATION.cff already at target version + date — no change needed" >&2
else
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] would patch CITATION.cff: version=${VERSION}, date-released=${TODAY}" >&2
  else
    # sed -i: the BSD vs GNU sed -i divergence bites this script on macOS.
    # Use a portable form — write to a temp file, mv back.
    TMP_CITATION=$(mktemp)
    awk \
      -v v="${VERSION}" \
      -v d="${TODAY}" \
      '
      /^version:/        {print "version: " v; next}
      /^date-released:/  {print "date-released: " d; next}
      {print}
      ' CITATION.cff > "${TMP_CITATION}"
    mv "${TMP_CITATION}" CITATION.cff
    echo "  CITATION.cff patched to version=${VERSION}, date-released=${TODAY}" >&2
  fi
fi

# ---------------------------------------------------------------------------
# Stage 3: validate the references: list (every entry has a real URL)
# ---------------------------------------------------------------------------

banner "Stage 3: validate CITATION.cff references"

# Each `- type: article` block must have a `url:` line pointing at a real
# arXiv / DOI / paper canonical URL. The audit chain has flagged
# LLM-paraphrased placeholder-author patterns where the references look
# real but the URLs are dead; this is the forcing function.
if ! python3 - <<'PYEOF' >&2; then
import re
import sys
from pathlib import Path

text = Path("CITATION.cff").read_text(encoding="utf-8")
# Split on the `- type:` boundaries — each block is one reference.
blocks = re.split(r"^\s+- type:", text, flags=re.MULTILINE)[1:]
failures: list[str] = []
for idx, block in enumerate(blocks, start=1):
    # Pull the title for a useful error message.
    title_match = re.search(r"title:\s*\"?([^\"\n]+)", block)
    title = title_match.group(1).strip() if title_match else f"reference #{idx}"
    url_match = re.search(r"url:\s*\"?(https?://[^\"\s]+)", block)
    if url_match is None:
        failures.append(f"  - {title}: missing or non-HTTPS url:")
        continue
    url = url_match.group(1)
    # Light shape check: arXiv / DOI / canonical journal hosts.
    accepted_prefixes = (
        "https://arxiv.org/",
        "https://doi.org/",
        "https://dl.acm.org/",
        "https://www.tandfonline.com/",
        "https://link.springer.com/",
        "https://openreview.net/",
        "https://aclanthology.org/",
        "https://proceedings.mlr.press/",
        "https://ieeexplore.ieee.org/",
        "https://datatracker.ietf.org/",
    )
    if not url.startswith(accepted_prefixes):
        failures.append(
            f"  - {title}: url={url!r} does not look like a recognised "
            "canonical paper host (arXiv / DOI / ACM / Springer / OpenReview "
            "/ ACL Anthology / MLR / IEEE / IETF). Re-verify the link before "
            "release."
        )
if failures:
    print("CITATION.cff reference validation failed:", file=sys.stderr)
    for f in failures:
        print(f, file=sys.stderr)
    sys.exit(1)
print("  CITATION.cff references: all entries have recognised canonical URLs", file=sys.stderr)
PYEOF
  echo "::error::CITATION.cff reference validation failed — see above." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Stage 4: translations via polyglot-mcp's TranslateGemma 12B
# ---------------------------------------------------------------------------

banner "Stage 4: translate README.md (8 languages)"

# Standing rule (memory: translation-workflow.md): translations always
# execute locally via TranslateGemma 12B through Ollama. Zero API cost.
# The advisor / kickoff-coordinator may invoke this script directly via
# the Bash tool; Sonnet kickoffs still defer to the user/advisor.
#
# Translation script path is a Mac-side default written `F:/AI/...`;
# the rig-translation rule at the top of CLAUDE.md translates to
# `E:/AI/polyglot-mcp/scripts/translate-all.mjs` on 5080. Pick whichever
# resolves on the current rig.
TRANSLATE_SCRIPT=""
for candidate in \
  "E:/AI/polyglot-mcp/scripts/translate-all.mjs" \
  "F:/AI/polyglot-mcp/scripts/translate-all.mjs" \
  "$HOME/polyglot-mcp/scripts/translate-all.mjs"; do
  if [ -f "${candidate}" ]; then
    TRANSLATE_SCRIPT="${candidate}"
    break
  fi
done

if [ -z "${TRANSLATE_SCRIPT}" ]; then
  echo "::warning::polyglot-mcp translate-all.mjs not found on this rig" >&2
  echo "  Searched: E:/AI/polyglot-mcp/, F:/AI/polyglot-mcp/, \$HOME/polyglot-mcp/" >&2
  echo "  Translation step SKIPPED — caller is responsible for ensuring" >&2
  echo "  README.{ja,zh,es,fr,hi,it,pt-BR}.md are up to date before tag." >&2
else
  echo "  using TRANSLATE_SCRIPT=${TRANSLATE_SCRIPT}" >&2
  maybe_run node "${TRANSLATE_SCRIPT}" "${REPO_ROOT}/README.md"
fi

# ---------------------------------------------------------------------------
# Stage 5: build sdist + wheel
# ---------------------------------------------------------------------------

banner "Stage 5: build sdist + wheel"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[dry-run] would run: python -m build" >&2
else
  # Clean dist/ first so the PKG-INFO smoke below only sees this build.
  rm -rf dist/ build/
  python -m build
fi

# ---------------------------------------------------------------------------
# Stage 6: PyPI metadata smoke (twine + PKG-INFO grep)
# ---------------------------------------------------------------------------

banner "Stage 6: PyPI metadata smoke (twine + PKG-INFO grep)"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[dry-run] would run: twine check dist/*" >&2
  echo "[dry-run] would extract PKG-INFO from sdist and grep for License + Author" >&2
else
  twine check dist/*
  SDIST=$(ls dist/backpropagate-*.tar.gz | head -1)
  if [ -z "${SDIST}" ]; then
    echo "::error::No sdist found in dist/" >&2
    exit 1
  fi
  PKGINFO=$(tar -xOf "${SDIST}" --wildcards '*/PKG-INFO' 2>/dev/null || true)
  if [ -z "${PKGINFO}" ]; then
    echo "::error::PKG-INFO not found inside ${SDIST}" >&2
    exit 1
  fi
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
    echo "::error::PKG-INFO is missing load-bearing fields: ${MISSING[*]}" >&2
    exit 1
  fi
  echo "  PKG-INFO smoke: load-bearing fields present" >&2
fi

# ---------------------------------------------------------------------------
# Stage 7: drift gate
# ---------------------------------------------------------------------------

banner "Stage 7: drift gate (scripts/check_doc_drift.py)"

# Drift gate is stdlib-only; runs in <1s. Don't allow --dry-run to skip
# this — drift detection is the load-bearing safety surface for the
# pre-release flow and skipping it would defeat the whole script.
if ! python3 scripts/check_doc_drift.py; then
  echo "::error::drift gate failed — fix the surfaces named above before tagging" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Stage 8: verify.sh --format=json
# ---------------------------------------------------------------------------

banner "Stage 8: verify.sh --format=json (lint + typecheck + tests + build + twine)"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[dry-run] would run: ./verify.sh --format=json" >&2
else
  if ! ./verify.sh --format=json > prep-release-verify.json 2> prep-release-verify.stderr.log; then
    echo "::error::verify.sh failed — see prep-release-verify.stderr.log + verify-*.log" >&2
    exit 1
  fi
  # Reuse the same jq-based gate as ci.yml so the local + CI surfaces
  # are byte-identical in their failure semantics.
  PARSED=$(tail -1 prep-release-verify.json | jq -r '.first_failed_stage // ""')
  if [ -n "${PARSED}" ]; then
    echo "::error::verify.sh failed at stage: ${PARSED}" >&2
    exit 1
  fi
  echo "  verify.sh: all stages passed" >&2
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

banner "prep_release.sh: all stages passed"
echo "  Next steps:" >&2
echo "    git add -A && git commit -m 'release: v${VERSION}'" >&2
echo "    git tag -a v${VERSION} -m 'Release v${VERSION}'" >&2
echo "    git push && git push --tags" >&2
echo "    # release.yml fires on tag push and runs npm publish + GH Release" >&2
