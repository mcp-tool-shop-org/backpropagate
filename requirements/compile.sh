#!/usr/bin/env bash
# requirements/compile.sh -- regenerate every hash-pinned requirements/*.txt
# from its requirements/*.in source.
#
# WHY THESE FILES EXIST. CI installs its tools (uv, build, twine, bandit,
# pip-audit, cyclonedx-bom, the build backend)
# with `pip install --require-hashes -r requirements/<name>.txt`, so every
# byte pip installs is checked against a hash committed here. Project
# dependencies are NOT listed here: CI derives them from uv.lock at run time
# (scripts/ci_install_locked.sh), so uv.lock stays the single source of truth.
#
# HOW TO REFRESH. Run this script, review the diff, commit .in and .txt
# together. Nothing refreshes these automatically.
#
#   requirements/compile.sh              # all files
#   requirements/compile.sh bandit       # one file
#
# Needs `uvx` (https://docs.astral.sh/uv/). uv is pinned below so the output is
# reproducible; bump UV_VERSION deliberately.
#
# --universal           one file valid on every OS / Python (markers kept), so
#                       the same file serves the Linux and Windows CI cells.
# --python-version 3.10 resolve for the floor of requires-python, so nothing
#                       in the file needs a newer interpreter than CI's oldest.
# --generate-hashes     the whole point.
#
set -euo pipefail

UV_VERSION="0.12.21"
cd "$(dirname "$0")/.."

uv_run() { uvx --from "uv@${UV_VERSION}" uv "$@"; }

compile_universal() {
  local name="$1"
  UV_CUSTOM_COMPILE_COMMAND="requirements/compile.sh ${name}  (uv ${UV_VERSION}: uv pip compile requirements/${name}.in --universal --python-version 3.10 --generate-hashes)" \
    uv_run pip compile "requirements/${name}.in" --universal --python-version 3.10 \
      --generate-hashes -o "requirements/${name}.txt"
}

targets=("$@")
if [ "${#targets[@]}" -eq 0 ]; then
  targets=(uv build-backend build-tools bandit pip-audit sbom)
fi

for t in "${targets[@]}"; do
  case "$t" in
    uv|build-backend|build-tools|bandit|pip-audit|sbom) compile_universal "$t" ;;
    *) echo "unknown target: $t" >&2; exit 2 ;;
  esac
done
