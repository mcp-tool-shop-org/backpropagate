#!/usr/bin/env bash
# pod_e3_sweep.sh — E3: measure the QLoRA presets' real VRAM peak (NVML), point by point.
#
# Target: a fresh RunPod pod C, RTX 5090 32 GB, image
# runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404, created with
#   RUNPOD_CONTAINER_DISK_GB=200 scripts/runpod/pod.sh create-retry ...
# (the 14B / 24B / 32B presets are bf16 repos, ~140 GB together; each preset's
# weights are deleted from the HF cache after its points, but the largest pair
# held at once is ~112 GB with the next preset prefetched). The HF cache lives on
# the CONTAINER disk (/root), never on /workspace (often a slow network mount).
#
# What it runs: scripts/pod_e3_sweep.py (96-point grid: 12 presets small to large x
# batch 1/2/4/6 x Unsloth off/on, one process and one JSON receipt per point), then
# scripts/e3_refit.py on the receipts (PASS/FAIL of the pre-registered gate).
# Read docs/receipts/2026-10-e3-vram/README.md for the gate and the drop order.
#
# Stages (STAGES="preflight install arch plan sweep refit"; each resumable):
#   preflight  GPU / disk / filesystem / HF token presence (never its value)
#   install    clone $BRANCH, pip install -e (torch pinned to the image's), then the
#              [unsloth] extra (if that fails the sweep runs Unsloth-off only and says so)
#   arch       each preset's architecture + text-only parameter count (config only)
#   plan       the grid, the cost estimate and what the budget guard drops
#   sweep      the measurements (the paid part; budget guard + resumable)
#   refit      e3_refit.py -> $WORK/refit.json and a PASS/FAIL line
#   llama      OPT-IN, not in the default STAGES: the Llama-3.1-8B preset smoke
#              (gated; needs HF_TOKEN in the environment)
#   leak       always last: fail if the HF token appears anywhere under $WORK
#
# Usage (on the pod, inside tmux):
#   read -rs HF_TOKEN; export HF_TOKEN              # no echo; never on a command line
#   export HISTFILE=/dev/null
#   BRANCH=feat/e3-vram-sweep nohup bash pod_e3_sweep.sh > /dev/null 2>&1 &
#   tail -f /root/e3/pod.log
#
# Knobs (env): BRANCH, REPO, WORK (default /root/e3), HF_HOME (default /root/hf-e3),
#   STAGES, UNSLOTH (both|on|off; default both, forced to off if the extra will not install),
#   E3_BUDGET_USD (default 1.20 = the sweep's share of pod C's $2.00), E3_USD_PER_HOUR
#   (0.90), E3_DL_MBPS (200), E3_STOP_AFTER_EPOCH (unix time: hard wall), STEPS (8),
#   PRESETS / BATCHES (comma lists; for top-up runs), EXTRA_ARGS (passed to the sweep, e.g. --no-guard).
#
# Lessons carried over from pod_block_engine.sh: pip needs --break-system-packages
# (PEP 668); torch is constrained to the image's exact version; no pkill anywhere.
set -uo pipefail
set +x

REPO="${REPO:-https://github.com/mcp-tool-shop-org/backpropagate.git}"
BRANCH="${BRANCH:-feat/e3-vram-sweep}"
WORK="${WORK:-/root/e3}"
STAGES="${STAGES:-preflight install arch plan sweep refit}"
UNSLOTH="${UNSLOTH:-both}"
STEPS="${STEPS:-8}"
PY="${PY:-python3}"
export HF_HOME="${HF_HOME:-/root/hf-e3}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export UNSLOTH_AUTO_INSTALL=0
export TOKENIZERS_PARALLELISM=false
export POD_IMAGE="${POD_IMAGE:-runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404}"
# Deliberately NOT setting PYTORCH_CUDA_ALLOC_CONF: the sweep measures what a user's
# process gets with the library's defaults (stage d used expandable_segments; that
# lowers reserved memory, which would flatter the estimator).

mkdir -p "$WORK/.done" "$WORK/runs" "$HF_HOME"
LOG="$WORK/pod.log"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
done_mark() { touch "$WORK/.done/$1"; }
is_done() { [ -f "$WORK/.done/$1" ] && [ "${FORCE_STAGE:-}" != "$1" ]; }
die() { log "FAIL: $*"; echo "RESULT: FAIL ($*)"; exit 1; }
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

SRC="$WORK/backpropagate"
DRIVER="$SRC/scripts/pod_e3_sweep.py"
REFIT="$SRC/scripts/e3_refit.py"

# ---------------------------------------------------------------- preflight
if want preflight && ! is_done preflight; then
  log "stage preflight"
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv || die "no GPU (nvidia-smi failed)"
  "$PY" -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0)); sys.exit(0 if torch.cuda.is_available() else 3)" \
    || die "torch.cuda.is_available() is False: this pod's GPU is broken, delete it and create another"
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
  [ "${used:-0}" -lt 1500 ] || die "${used} MiB already used on the GPU: every NVML peak would be polluted"
  fstype="$(stat -f -c %T "$HF_HOME" 2>/dev/null || echo unknown)"
  log "HF_HOME=$HF_HOME filesystem=$fstype"
  case "$fstype" in nfs*|fuse*|*moose*|cifs*|smb*|ceph*|lustre*) die "HF_HOME is on a network filesystem ($fstype); it must be on the container disk" ;; esac
  free_gb="$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc '0-9')"
  log "free under HF_HOME: ${free_gb} GB (need >= 85 for one preset at a time, ~130 with prefetch)"
  [ "${free_gb:-0}" -ge 85 ] || die "only ${free_gb} GB free: create the pod with RUNPOD_CONTAINER_DISK_GB=200"
  free -g | head -2
  if [ -n "${HF_TOKEN:-}" ]; then log "HF_TOKEN: present in the environment (value never logged)"; else log "HF_TOKEN: ABSENT - gated Llama presets will be skipped with a receipt"; fi
  done_mark preflight
fi

# ---------------------------------------------------------------- install
if want install && ! is_done install; then
  log "stage install: $BRANCH"
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
  "$PY" -m pip install --break-system-packages -q -c "$WORK/constraints.txt" -e "$SRC" datasets psutil nvidia-ml-py \
    "huggingface_hub[cli]" || die "pip install failed"
  if [ "$UNSLOTH" != "off" ]; then
    log "installing the [unsloth] extra (torch pinned to the image)"
    if timeout 900 "$PY" -m pip install --break-system-packages -q -c "$WORK/constraints.txt" -e "$SRC[unsloth]" \
         > "$WORK/unsloth_install.log" 2>&1; then
      "$PY" -c "import importlib.metadata as m; print('unsloth', m.version('unsloth'), 'transformers', m.version('transformers'), 'trl', m.version('trl'))"
    else
      log "unsloth install FAILED (see unsloth_install.log): the sweep runs Unsloth-off only"
      echo off > "$WORK/unsloth_arm"
    fi
  fi
  "$PY" -c "import torch; print('torch after install', torch.__version__)"
  done_mark install
fi
[ -d "$SRC/.git" ] || die "no checkout at $SRC (run the install stage)"
GIT_SHA="$(git -C "$SRC" rev-parse HEAD)"
export GIT_SHA
[ -f "$WORK/unsloth_arm" ] && UNSLOTH="$(cat "$WORK/unsloth_arm")"
log "git sha $GIT_SHA; unsloth arm: $UNSLOTH"
SEL=(--steps "$STEPS" --unsloth "$UNSLOTH")
[ -n "${PRESETS:-}" ] && SEL+=(--presets "$PRESETS")
[ -n "${BATCHES:-}" ] && SEL+=(--batches "$BATCHES")

# ---------------------------------------------------------------- arch
if want arch && ! is_done arch; then
  log "stage arch: config-only architecture + text-only parameter count per preset"
  "$PY" "$DRIVER" arch --out "$WORK" "${SEL[@]}" || log "arch: some presets failed (see above)"
  done_mark arch
fi

# ---------------------------------------------------------------- plan
if want plan; then
  log "stage plan"
  "$PY" "$DRIVER" plan --out "$WORK" "${SEL[@]}" || die "plan failed"
fi

# ---------------------------------------------------------------- sweep
if want sweep && ! is_done sweep; then
  log "stage sweep (resumable: re-running skips points that already have an ok/oom receipt)"
  # shellcheck disable=SC2086
  "$PY" "$DRIVER" run --out "$WORK" "${SEL[@]}" ${EXTRA_ARGS:-}
  rc=$?
  log "sweep exited $rc"
  [ "$rc" -eq 0 ] && done_mark sweep
  [ "$rc" -eq 2 ] && die "sweep halted (andon): see $WORK/halt.json or the log above"
fi

# ---------------------------------------------------------------- refit
if want refit; then
  log "stage refit: gate verdict"
  "$PY" "$REFIT" --receipts "$WORK" --out "$WORK/refit.json"
  rc=$?
  case "$rc" in
    0) echo "RESULT: E3 gate PASS" ;;
    1) echo "RESULT: E3 gate FAIL (see refit.json; the receipts are the deliverable either way)" ;;
    *) echo "RESULT: E3 refit INSUFFICIENT DATA" ;;
  esac
fi

# ---------------------------------------------------------------- llama (opt-in)
# Llama-3.1-8B preset smoke: 2 real QLoRA steps + adapter save + rank check
# (tests/test_qlora_presets_smoke.py). GATED: needs HF_TOKEN in the environment,
# exported in this shell (read -rs, never typed on a command line, never
# `huggingface-cli login`, which writes the token to disk).
if want llama && ! is_done llama; then
  log "stage llama: Llama-3.1-8B preset smoke"
  [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set: meta-llama/Llama-3.1-8B-Instruct is gated"
  (cd "$SRC" && BACKPROPAGATE_RUN_PRESET_SMOKE=1 "$PY" -m pytest tests/test_qlora_presets_smoke.py \
      -k llama-3.1-8b -m "slow or integration" -s -p no:randomly -p no:cacheprovider \
      2>&1 | tee "$WORK/preset_llama-3.1-8b.log")
  rc=${PIPESTATUS[0]}
  grep -a "PRESET_SMOKE_RECEIPT" "$WORK/preset_llama-3.1-8b.log" || true
  [ "$rc" -eq 0 ] && echo "RESULT: llama-3.1-8b preset smoke PASS" || echo "RESULT: llama-3.1-8b preset smoke FAIL (rc=$rc)"
  done_mark llama
fi

# ---------------------------------------------------------------- leak check (always)
if [ -n "${HF_TOKEN:-}" ]; then
  # the token is fed to grep through a process substitution of a shell builtin: it is on no command line
  if grep -rFq -f <(printf '%s\n' "$HF_TOKEN") "$WORK" 2>/dev/null; then
    echo "RESULT: FAIL - the HF token appears under $WORK; do NOT copy it off the pod"
    exit 3
  fi
  log "leak check: the HF token does not appear under $WORK"
fi
log "done. Copy back: scp -r <pod>:$WORK/{runs,arch,plan.json,summary.json,refit.json,env.json,pod.log} <dest>"
