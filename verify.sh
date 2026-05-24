#!/usr/bin/env bash
# verify.sh — single-command verification (Ship Gate D1)
#
# Usage:
#   verify.sh                       # equivalent to --format=human
#   verify.sh --format=human        # human-readable banner output (default)
#   verify.sh --format=json         # one JSON object per stage on stdout
#                                   # ({"stage":"<name>","status":"pass|fail|start",
#                                   # "exit_code":<int>,"duration_seconds":<float>})
#                                   # plus a final {"stage":"verify","status":"pass|fail",
#                                   # ...} record. Each stage's own command output is
#                                   # captured to verify-<stage>.log so the JSON stream
#                                   # stays parseable; CI can grep the .log for detail.
#
# Exit codes:
#   0 — all stages passed
#   1 — at least one stage failed (see "first failure" record in JSON or the
#       red banner in human mode for the failing stage name)
#   2 — bad flag / usage
set -euo pipefail

FORMAT="human"
for arg in "$@"; do
  case "${arg}" in
    --format=human) FORMAT="human" ;;
    --format=json)  FORMAT="json" ;;
    -h|--help)
      sed -n '2,21p' "$0"
      exit 0
      ;;
    *)
      echo "verify.sh: unknown argument: ${arg}" >&2
      echo "verify.sh: supported flags: --format=human (default), --format=json, --help" >&2
      exit 2
      ;;
  esac
done

emit_json() {
  # emit_json <stage> <status> <exit_code> <duration_seconds>
  printf '{"stage":"%s","status":"%s","exit_code":%s,"duration_seconds":%s}\n' \
    "$1" "$2" "$3" "$4"
}

run_stage() {
  # run_stage <name> <command...>
  local name="$1"; shift
  local start end duration rc log
  log="verify-${name}.log"
  start=$(date +%s.%N)
  if [ "${FORMAT}" = "human" ]; then
    echo "=== ${name} ==="
    if "$@"; then rc=0; else rc=$?; fi
  else
    emit_json "${name}" "start" 0 0
    if "$@" >"${log}" 2>&1; then rc=0; else rc=$?; fi
  fi
  end=$(date +%s.%N)
  duration=$(awk -v s="${start}" -v e="${end}" 'BEGIN { printf "%.3f", e - s }')
  if [ "${FORMAT}" = "json" ]; then
    if [ ${rc} -eq 0 ]; then
      emit_json "${name}" "pass" ${rc} "${duration}"
    else
      emit_json "${name}" "fail" ${rc} "${duration}"
    fi
  fi
  return ${rc}
}

OVERALL_RC=0
FIRST_FAILED_STAGE=""
record_failure() {
  if [ -z "${FIRST_FAILED_STAGE}" ]; then
    FIRST_FAILED_STAGE="$1"
  fi
  OVERALL_RC=1
}

run_stage lint        ruff check backpropagate/                || record_failure lint
run_stage typecheck   mypy backpropagate/ --ignore-missing-imports || record_failure typecheck
run_stage tests       pytest tests/ -x --timeout=60            || record_failure tests
run_stage build       python -m build                          || record_failure build
run_stage twine_check twine check dist/*                       || record_failure twine_check

if [ "${FORMAT}" = "json" ]; then
  if [ ${OVERALL_RC} -eq 0 ]; then
    printf '{"stage":"verify","status":"pass","exit_code":0,"first_failed_stage":null}\n'
  else
    printf '{"stage":"verify","status":"fail","exit_code":%s,"first_failed_stage":"%s"}\n' \
      "${OVERALL_RC}" "${FIRST_FAILED_STAGE}"
  fi
else
  if [ ${OVERALL_RC} -eq 0 ]; then
    echo "✓ verify passed"
  else
    echo "✗ verify failed at stage: ${FIRST_FAILED_STAGE}" >&2
  fi
fi

exit ${OVERALL_RC}
