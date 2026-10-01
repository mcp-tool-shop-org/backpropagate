#!/usr/bin/env bash
# pod_e4.sh — experiment E4 (where does full fine-tuning beat QLoRA?) on a RunPod RTX 5090 pod.
#
# Target: runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404 (torch 2.8.0 is used as installed, never
# reinstalled). Pre-registration and decision rules: docs/receipts/2026-10-e4-code/README.md.
# The lead creates and deletes the pod; nothing here calls the RunPod API.
#
# Commands (each resumable: a run whose receipt exists is skipped):
#   setup            preflight, clone + install, sandbox self-test, dataset prep + token-length stats
#   precheck <3b|7b> untrained base model on the eval set; abort gate 0.10 < pass@1 < 0.80   (exit 4)
#   stage <3b|7b>    the stage's runs in priority order under the time-based budget guard
#   gate <3b|7b>     statistics + pre-registered verdict -> stage_e4_<size>.json              (exit 3 = stop)
#   run <3b|7b>      precheck -> stage -> gate in one go (the 7b stage also needs a 3b PASS)
#   dry-run          the whole pipeline with a tiny model for 3 steps, no generated code executed
#
# Sequence for the paid run:
#   bash scripts/pod_e4.sh setup
#   bash scripts/pod_e4.sh precheck 3b && bash scripts/pod_e4.sh stage 3b && bash scripts/pod_e4.sh gate 3b
#   # only if the gate exits 0:
#   bash scripts/pod_e4.sh precheck 7b && bash scripts/pod_e4.sh stage 7b && bash scripts/pod_e4.sh gate 7b
#
# Knobs (env): BRANCH (feat/e4-code-harness), REPO, WORK (/workspace/e4: receipts), HF_HOME
#   (/root/hf: the model cache stays on the container disk), BP_WORK_DIR (/root/e4_work: trainer scratch),
#   E4_START_EPOCH (when billing began; default = the first `setup`), plus the pod_e4.py knobs
#   (E4_STEPS, E4_BATCH, E4_SEQ, E4_SEEDS_3B, E4_SEEDS_7B, E4_BUDGET_USD_3B/_7B, E4_RATE_USD_H, ...).
#
# Pod lessons carried over from pod_block_engine.sh: pip needs --break-system-packages (PEP 668); torch is
# constrained to the image's exact version; no pkill anywhere; run it under tmux and scp the receipts
# back as each stage finishes (the receipts are $WORK/runs, $WORK/stage_e4_*.json, $WORK/budget_*.json).
set -uo pipefail

REPO="${REPO:-https://github.com/mcp-tool-shop-org/backpropagate.git}"
BRANCH="${BRANCH:-feat/e4-code-harness}"
WORK="${WORK:-/workspace/e4}"
PY="${PY:-python3}"
export HF_HOME="${HF_HOME:-/root/hf}"
export BP_WORK_DIR="${BP_WORK_DIR:-/root/e4_work}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export UNSLOTH_AUTO_INSTALL=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export POD_IMAGE="${POD_IMAGE:-runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404}"
SRC="${SRC:-$WORK/backpropagate}"

mkdir -p "$WORK/.done" "$WORK/runs" "$HF_HOME" "$BP_WORK_DIR"
exec > >(tee -a "$WORK/pod.log") 2>&1
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { log "FAIL: $*"; echo "RESULT: FAIL ($*)"; exit 1; }

cmd="${1:-}"
size="${2:-}"

# The billing clock of the 3b stage: E4_START_EPOCH, else the first `setup`. (The 7b stage keeps its own
# clock, started by its first command, unless E4_START_EPOCH is set explicitly.)
if [ ! -f "$WORK/.pod_start_epoch" ]; then date +%s > "$WORK/.pod_start_epoch"; fi
if [ "$size" = "3b" ] && [ -z "${E4_START_EPOCH:-}" ]; then E4_START_EPOCH="$(cat "$WORK/.pod_start_epoch")"; export E4_START_EPOCH; fi

setup() {
  if [ ! -f "$WORK/.done/0" ]; then
    log "preflight"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || die "no GPU"
    free -g
    df -h "$WORK" "$HF_HOME" "$BP_WORK_DIR"
    "$PY" - <<'EOF' > "$WORK/env.json" || die "torch import failed"
import json, os, platform, torch
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
  "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0),
  "vram_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
  "host_ram_total_gib": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)}))
EOF
    cat "$WORK/env.json"
    touch "$WORK/.done/0"
  fi
  if [ ! -f "$WORK/.done/1" ]; then
    log "install $BRANCH"
    if [ -d "$SRC/.git" ]; then
      git -C "$SRC" fetch origin "$BRANCH" && git -C "$SRC" checkout -B "$BRANCH" "origin/$BRANCH" || die "git update failed"
    else
      git clone --branch "$BRANCH" "$REPO" "$SRC" || die "git clone failed"
    fi
    "$PY" - <<'EOF' > "$WORK/constraints.txt"
import importlib.metadata as md
for pkg in ("torch", "torchvision", "torchaudio", "triton"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        pass
EOF
    cat "$WORK/constraints.txt"
    "$PY" -m pip install --break-system-packages -q -c "$WORK/constraints.txt" -e "$SRC" datasets psutil pyarrow \
      "huggingface_hub[cli]" nvidia-ml-py || die "pip install failed"
    "$PY" -m pip install --break-system-packages -q --no-deps galore-torch || log "galore-torch install failed: the GaLore arm will record the error"
    "$PY" -c "import torch; print('torch after install', torch.__version__)"
    touch "$WORK/.done/1"
  fi
  GIT_SHA="$(git -C "$SRC" rev-parse HEAD)"
  log "git sha $GIT_SHA"
  # Andon: the sandbox must behave on THIS pod (rlimits, process groups) before any money is spent on it.
  "$PY" "$SRC/scripts/e4_lib.py" selftest --allow-exec || die "pass@1 sandbox self-test failed on this pod"
  "$PY" "$SRC/scripts/pod_e4.py" prep "${size:-3b}" --out "$WORK" || die "prep failed"
}

# Everything but setup needs the install.
if [ "$cmd" != setup ] && [ ! -f "$WORK/.done/1" ]; then die "run 'setup' first"; fi
if [ -d "$SRC/.git" ]; then GIT_SHA="$(git -C "$SRC" rev-parse HEAD)"; export GIT_SHA; fi
E4PY=("$PY" "$SRC/scripts/pod_e4.py")

case "$cmd" in
  setup)    export GIT_SHA; setup ;;
  precheck) [ -n "$size" ] || die "usage: precheck <3b|7b>"; "${E4PY[@]}" precheck "$size" --out "$WORK" ;;
  stage)    [ -n "$size" ] || die "usage: stage <3b|7b>"; "${E4PY[@]}" stage "$size" --out "$WORK" ;;
  gate)     [ -n "$size" ] || die "usage: gate <3b|7b>"; "${E4PY[@]}" gate "$size" --out "$WORK" ;;
  run)      [ -n "$size" ] || die "usage: run <3b|7b>"; "${E4PY[@]}" run "$size" --out "$WORK" ;;
  dry-run)  "${E4PY[@]}" dry-run --out "$WORK/dry" ;;
  *)        sed -n '2,32p' "$0"; exit 2 ;;
esac
rc=$?
log "$cmd ${size:-} exit $rc"
exit $rc
