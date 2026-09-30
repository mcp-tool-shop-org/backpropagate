#!/usr/bin/env bash
# scripts/gpu_smoke.sh: run every real-GPU smoke and print a receipt.
#
# Runs each tests/test_*_smoke.py file in its own pytest process, with the
# slow/integration markers selected and the per-test timeout off. The MLX smoke
# is skipped on hosts that are not Apple Silicon. When the run finishes, the
# script prints a receipt block: date, git SHA, GPU, the resolved versions of
# the training stack, and PASS / FAIL / SKIP for each smoke. A release is not
# tagged without this receipt.
#
# Usage (from the repo root, with the venv you want to measure active, or
# with PYTHON pointing at its interpreter):
#
#   bash scripts/gpu_smoke.sh
#   PYTHON=.venv/Scripts/python.exe bash scripts/gpu_smoke.sh     # Git Bash
#   bash scripts/gpu_smoke.sh tests/test_golden_path_smoke.py      # a subset
#
# Exit status: 0 when no smoke failed (skips are allowed, and each is listed
# with its reason). 1 when any smoke failed or errored.
#
# Environment:
#   PYTHON                          interpreter to test (default: python, else python3)
#   HF_HUB_CACHE                    set automatically to ~/.cache/huggingface/hub
#                                   when HF_HOME points elsewhere and the smoke
#                                   model is only cached there
#   BACKPROPAGATE_LLAMA_CPP_PATH    llama.cpp clone; enables the GGUF fallback smoke
#   UNSLOTH_AUTO_INSTALL            always forced to 0 (see below)
#
# Works on Linux, macOS and Git Bash on Windows.

set -u

cd "$(dirname "$0")/.." || exit 1

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
fi

# A smoke must never change the host. Unsloth's GGUF path otherwise installs
# system packages on its own (on Windows: `winget install` CMake, OpenSSL and
# VS Build Tools, accepting their licence agreements).
export UNSLOTH_AUTO_INSTALL=0
export PYTHONIOENCODING=utf-8

# Smoke model cache: the smokes probe the default HF hub cache. If HF_HOME
# relocates it but the model was downloaded to the legacy default location,
# point the hub cache there rather than downloading again.
if [ -z "${HF_HUB_CACHE:-}" ] \
   && [ -d "$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct" ] \
   && [ -n "${HF_HOME:-}" ] \
   && [ ! -d "${HF_HOME}/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

if [ "$#" -gt 0 ]; then
    SMOKES=("$@")
else
    SMOKES=()
    for f in tests/test_*_smoke.py; do
        [ -e "$f" ] || continue
        if [ "$f" = "tests/test_mlx_smoke.py" ] \
           && ! { [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; }; then
            continue
        fi
        SMOKES+=("$f")
    done
fi

LOGDIR="$(mktemp -d 2>/dev/null || echo "${TMPDIR:-/tmp}/gpu_smoke.$$")"
mkdir -p "$LOGDIR"
# Facts the smokes record for the receipt (e.g. which backend actually trained).
export GPU_SMOKE_FACTS="$LOGDIR/facts.txt"
: >"$GPU_SMOKE_FACTS"

# Host-mutation check: installer processes present before vs. after each smoke.
installer_pids() {
    if command -v tasklist >/dev/null 2>&1; then
        tasklist //FO CSV //NH 2>/dev/null | grep -iE '^"(winget|msiexec|AppInstallerCLI)\.exe"' \
            | cut -d, -f2 | tr -d '"' | sort
    else
        pgrep -x 'apt-get|apt|dpkg|brew' 2>/dev/null | sort
    fi
}
INSTALLERS_BEFORE="$(installer_pids)"
NEW_INSTALLERS=""

declare -a RESULTS=()
FAILED=0

for smoke in "${SMOKES[@]}"; do
    name="$(basename "$smoke" .py)"
    log="$LOGDIR/$name.log"
    echo "==> $smoke"
    "$PY" -m pytest "$smoke" -m "slow or integration" --timeout=0 -p no:cacheprovider \
        -q -rs >"$log" 2>&1
    rc=$?
    new="$(comm -13 <(printf '%s\n' "$INSTALLERS_BEFORE") <(installer_pids) | grep -v '^$')"
    [ -n "$new" ] && NEW_INSTALLERS="$NEW_INSTALLERS $name:$(echo "$new" | tr '\n' ',')"
    summary="$(grep -E '^=+ .*(passed|failed|skipped|error|no tests ran).* =+$' "$log" | tail -n 1 | sed -E 's/^=+ //; s/ =+$//')"
    if [ "$rc" -eq 0 ]; then
        if grep -Eq '(^|[^0-9])[0-9]+ passed' <<<"$summary"; then
            status=PASS
        else
            status=SKIP
        fi
    elif [ "$rc" -eq 5 ]; then
        status=SKIP; summary="no tests collected"
    else
        status=FAIL; FAILED=1
    fi
    RESULTS+=("$(printf '%-4s  %-32s %s' "$status" "$name" "$summary")")
    # Show why things skipped or failed without making anyone open the log.
    if [ "$status" != PASS ] || grep -q '^SKIPPED' "$log"; then
        grep -E '^(SKIPPED|FAILED|ERROR) ' "$log" | sed 's/^/      /'
    fi
    if [ "$status" = FAIL ]; then
        tail -n 30 "$log" | sed 's/^/      | /'
    fi
done

# Receipt ---------------------------------------------------------------------
SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
DIRTY=""
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then DIRTY=" (dirty)"; fi
GPU="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
[ -n "$GPU" ] || GPU="$("$PY" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device")' 2>/dev/null || echo unknown)"

STACK="$("$PY" - <<'PYEOF' 2>/dev/null
import sys
from importlib import metadata

print(f"python        {sys.version.split()[0]}")
for dist in ("backpropagate", "torch", "transformers", "trl", "peft",
             "unsloth", "unsloth_zoo", "bitsandbytes", "accelerate"):
    try:
        v = metadata.version(dist)
    except metadata.PackageNotFoundError:
        v = "not installed"
    print(f"{dist:<13} {v}")
try:
    import torch
    print(f"torch.cuda    {torch.version.cuda} (available={torch.cuda.is_available()})")
except Exception as e:  # noqa: BLE001
    print(f"torch.cuda    unavailable ({e})")
PYEOF
)"

echo
echo "================ GPU SMOKE RECEIPT ================"
echo "date          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git           ${SHA}${DIRTY}"
echo "gpu           ${GPU}"
echo "${STACK}"
echo "llama.cpp     ${BACKPROPAGATE_LLAMA_CPP_PATH:-unset}"
[ -s "$GPU_SMOKE_FACTS" ] && sort -u "$GPU_SMOKE_FACTS"
if [ -n "$NEW_INSTALLERS" ]; then
    echo "installers    NEW installer processes appeared:$NEW_INSTALLERS"
    FAILED=1
else
    echo "installers    none spawned (winget/msiexec PIDs unchanged across the run)"
fi
echo "---------------------------------------------------"
for r in "${RESULTS[@]}"; do echo "$r"; done
echo "---------------------------------------------------"
if [ "$FAILED" -ne 0 ]; then
    echo "RESULT        FAIL (logs: $LOGDIR)"
else
    echo "RESULT        OK (logs: $LOGDIR)"
fi
echo "==================================================="

exit "$FAILED"
