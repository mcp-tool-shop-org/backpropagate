#!/usr/bin/env bash
# scripts/ci_install_locked.sh -- install the project's dependencies in CI from
# uv.lock, every package checked against the hashes recorded there.
#
#   scripts/ci_install_locked.sh [EXTRA...]
#   scripts/ci_install_locked.sh --export-only FILE [EXTRA...]
#
# EXTRA is a pyproject optional-dependency group (dev, full, ui, ...); none
# means the core dependencies only.
#
# What it does:
#   1. installs uv from requirements/uv.txt          (pip --require-hashes)
#   2. `uv export --frozen` turns uv.lock + the extras into a hashed
#      requirements file (--frozen: use the lock exactly as committed)
#   3. installs the build backend from requirements/build-backend.txt, then the
#      project itself editable with --no-deps --no-build-isolation, so nothing
#      is resolved or fetched unhashed at any step.
#   4. installs the hashed export (--no-deps: it is the full closure) and runs
#      `pip check` so an incomplete closure fails loudly instead of silently.
#
# --export-only F write the hashed export to F and stop: nothing but uv is
#                 installed. Used by the pip-audit job, which audits the file.
#
# Deliberately NOT used by the jobs whose whole purpose is a FRESH resolve (the
# no-unsloth cell of the `test` job and `parallel-xdist` in ci.yml,
# nightly-train-smoke.yml, post-publish-smoke.yml). A locked install there would
# test the same set every run and could never see the next resolver walk-back.
set -euo pipefail

export_only=""
extras=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --export-only) export_only="${2:?--export-only needs a file}"; shift 2 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) extras+=("$1"); shift ;;
  esac
done
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

extra_flags=()
for e in ${extras[@]+"${extras[@]}"}; do extra_flags+=(--extra "$e"); done

python -m pip install --require-hashes -r requirements/uv.txt

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
lock_txt="${export_only:-$work/lock.txt}"

uv export --frozen --no-emit-project --format requirements-txt \
  ${extra_flags[@]+"${extra_flags[@]}"} -o "$lock_txt" >/dev/null

if [ -n "$export_only" ]; then
  echo "wrote $export_only"
  exit 0
fi

# Build backend first and the project (editable, nothing resolved, nothing
# fetched) second, so the lock install below is last and its versions win for
# any package the backend also needs (packaging, pluggy, ...).
python -m pip install --require-hashes -r requirements/build-backend.txt
python -m pip install --no-deps --no-build-isolation -e .

# `python -m pip` throughout: uv.lock pins pip itself, and on Windows pip.exe
# cannot replace its own executable (`python -m pip` can).
#
# --no-deps: the export is the complete closure, and pip cannot match a
# transitive `pkg[extra]` requirement (e.g. pytest-cov -> coverage[toml])
# against the plain `pkg==X` pin in hash mode. `pip check` below is what
# catches a closure that is not actually complete.
python -m pip install --require-hashes --no-deps -r "$lock_txt"

python -m pip check
