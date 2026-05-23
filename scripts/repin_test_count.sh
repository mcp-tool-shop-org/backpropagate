#!/usr/bin/env bash
# Re-pin the test count claimed in the canonical user-facing docs.
#
# Why this exists
# ---------------
# We pin a "tests collected" number in five places so users / auditors can spot-
# check it. The number drifts every time tests/ changes — even a single skipped
# file in Wave 1 of a dogfood swarm moved it by ten and the abandoned v1.1.2
# CHANGELOG quoted a number that had never been measured. This script makes
# re-pinning a 30-second mechanical step.
#
# Canonical pin sites (keep this list in sync if you add a doc that pins it)
# -------------------------------------------------------------------------
#   1. CHANGELOG.md — the active "## [x.y.z]" section's "### Tests" block AND
#      the "Test count re-pinned ..." bullet under "### Changed".
#   2. CLAUDE.md — line beginning "- NNNN tests in tests/ (pinned YYYY-MM-DD)".
#   3. PRODUCTION_READINESS_AUDIT.md — the front-matter "**Note:**" block AND
#      the "**Current Status**:" block at the bottom.
#   4. SECURITY_AUDIT_REPORT.md — the front-matter "**Note:**" block AND the
#      `~1,796 passed at v1.0.2; vX.Y.Z collects NNNN tests` line.
#   5. (Optionally) any new audit doc you've added — `grep -rn '<old-count>' .`
#      before bumping to find every spot.
#
# Usage
# -----
#   ./scripts/repin_test_count.sh             # print the new count + date,
#                                             # do NOT modify files
#
# This is intentionally read-only — there are five docs with subtly different
# surrounding prose, and a blind `sed -i` would mangle them. The script gives
# you the number; you Edit the five files by hand (or via an LLM tool) so the
# surrounding context can be tuned per doc.

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v pytest >/dev/null 2>&1; then
    echo "error: pytest not on PATH. Activate the venv first:" >&2
    echo "  source .venv/Scripts/activate   # Windows (Git Bash)" >&2
    echo "  source .venv/bin/activate       # macOS/Linux" >&2
    exit 1
fi

# Collect — discard any deprecation chatter on stderr so the tail-line is stable
COLLECT_LINE=$(pytest --collect-only -q tests/ 2>/dev/null | tail -1)

# Expected shape: "NNNN tests collected in S.SSs"
COUNT=$(printf '%s\n' "$COLLECT_LINE" | grep -oE '[0-9]+' | head -1)
DATE=$(date -u +%Y-%m-%d)

if [ -z "${COUNT:-}" ]; then
    echo "error: could not parse test count from pytest output:" >&2
    echo "  $COLLECT_LINE" >&2
    exit 2
fi

cat <<EOF
Re-pin the following number across the canonical pin sites:

  Test count: ${COUNT}
  Verified:   ${DATE}

Pin sites to update:
  - CHANGELOG.md (active version's "### Tests" + "### Changed" bullet)
  - CLAUDE.md ("NNNN tests in tests/ (pinned YYYY-MM-DD)")
  - PRODUCTION_READINESS_AUDIT.md (front-matter Note + Current Status block)
  - SECURITY_AUDIT_REPORT.md (front-matter Note + Test Coverage block)

Sanity check: \`grep -rn '<old-count>' . --include='*.md'\` should return zero
hits once the bump is complete.
EOF
