#!/usr/bin/env bash
# pod_block_engine.sh — evidence run for Engine B (block-coordinate AdamW).
#
# Target: a fresh RunPod pod, RTX 5090 32 GB, image
# runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404 (CUDA 12.8.1, torch 2.8.0).
# The script uses the IMAGE's torch and never reinstalls it.
#
# Stages, in order of value (each resumable: a stage marker in $WORK/.done/
# skips a finished stage, and inside a stage every run writes
# $WORK/runs/<tag>.json, so a re-run skips finished runs). Everything is
# logged to $WORK/pod.log; run under nohup or tmux.
#
#   0 preflight  GPU / VRAM / host RAM / versions -> $WORK/env.json
#   1 install    clone $BRANCH, pip install -e with torch pinned to the image
#   2 prep       dolly-15k split: 400 train / 150 disjoint held-out
#   smoke        SmolLM2-135M, engine block, 20 steps, K=5 — the first CUDA
#                run of an fp32 block inside a bf16 model (~2 min)
#   a  quality   Engine B (K=50) vs the library's standard pure-GPU full FT,
#                Qwen2.5-1.5B-Instruct + SmolLM3-3B, seeds 0 1 2, 150 steps,
#                batch 4, seq 512, lr 2e-5; held-out before/after (~35 min)
#   b  7B        Qwen2.5-7B-Instruct: Engine B uncapped (90 steps, K=30,
#                descending so the head — the largest block — is visited),
#                save -> reload -> generate; fit under 24 GiB (embeddings
#                trained / frozen) and 16 GiB caps; QLoRA on the same data (~30 min)
#   c  rounding  write-back nearest vs stochastic x K in {10, 50, 200},
#                SmolLM2-360M + SmolLM3-3B, 150 steps, held-out loss (~30 min)
#   a2 one-pass  stage a again with K=5 (~one full pass over all blocks in 150
#                steps; K=50 visits only 3 of ~30 blocks) (~25 min)
#
# Usage (on the pod):
#   BRANCH=feat/block-coordinate-engine nohup bash pod_block_engine.sh > /dev/null 2>&1 &
#   tail -f /workspace/block_engine/pod.log
#
# Knobs (env): BRANCH, REPO, WORK (default /workspace/block_engine),
#   STAGES (default "smoke a b c a2"), SEEDS (default "0 1 2"), STEPS (150),
#   STEPS_C (150), FORCE_STAGE=<name> to re-run one stage's summary.
#
# Pod lessons carried over from pod_offload_7b.sh: pip needs
# --break-system-packages (PEP 668); torch is constrained to the image's exact
# version; no `curl | head` under pipefail; no pkill anywhere.
set -uo pipefail

REPO="${REPO:-https://github.com/mcp-tool-shop-org/backpropagate.git}"
BRANCH="${BRANCH:-feat/block-coordinate-engine}"
WORK="${WORK:-/workspace/block_engine}"
STAGES="${STAGES:-smoke a b c a2}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-150}"
STEPS_C="${STEPS_C:-150}"
export HF_HOME="${HF_HOME:-$WORK/hf}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export UNSLOTH_AUTO_INSTALL=0
export TOKENIZERS_PARALLELISM=false
PY="${PY:-python3}"

Q15="Qwen/Qwen2.5-1.5B-Instruct"
S3="HuggingFaceTB/SmolLM3-3B"
Q7="Qwen/Qwen2.5-7B-Instruct"
S360="HuggingFaceTB/SmolLM2-360M-Instruct"
S135="HuggingFaceTB/SmolLM2-135M-Instruct"

mkdir -p "$WORK/.done" "$WORK/runs" "$HF_HOME"
LOG="$WORK/pod.log"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
done_mark() { touch "$WORK/.done/$1"; }
is_done() { [ -f "$WORK/.done/$1" ] && [ "${FORCE_STAGE:-}" != "$1" ]; }
die() { log "FAIL: $*"; echo "RESULT: FAIL ($*)"; exit 1; }
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ---------------------------------------------------------------- 0 preflight
if ! is_done 0; then
  log "stage 0: preflight"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || die "no GPU"
  free -g
  df -h "$WORK"
  "$PY" - <<'EOF' > "$WORK/env.json" || die "torch import failed"
import json, os, platform, torch
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
  "vram_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
  "host_ram_total_gib": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)}))
EOF
  cat "$WORK/env.json"
  done_mark 0
fi

# ---------------------------------------------------------------- 1 install
SRC="$WORK/backpropagate"
if ! is_done 1; then
  log "stage 1: install $BRANCH"
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
  "$PY" -m pip install --break-system-packages -q -c "$WORK/constraints.txt" -e "$SRC" datasets psutil \
    "huggingface_hub[cli]" || die "pip install failed"
  "$PY" -c "import torch; print('torch after install', torch.__version__)"
  done_mark 1
fi
GIT_SHA="$(git -C "$SRC" rev-parse HEAD)"
export GIT_SHA
DRIVER="$SRC/scripts/pod_block_engine.py"
log "git sha $GIT_SHA"

# One run: skip when its receipt exists. ARM names the arm in the summaries.
run() {  # run <tag> <arm> <driver args...>
  local tag="$1" arm="$2"; shift 2
  if [ -f "$WORK/runs/$tag.json" ]; then log "skip $tag (receipt exists)"; return 0; fi
  log "run $tag ($arm): $*"
  ARM="$arm" "$PY" "$DRIVER" train --out "$WORK" --tag "$tag" "$@" \
    || log "run $tag exited non-zero (receipt records why)"
}
base() {  # base <model>
  local f="$WORK/runs/base_${1//\//_}.json"
  if [ -f "$f" ]; then return 0; fi
  log "base $1"
  "$PY" "$DRIVER" base --out "$WORK" --model "$1" || log "base $1: held-out loss OUT OF BOUNDS"
}
fetch() {  # fetch <model>
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('$1', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model','tokenizer*'])" \
    || die "download of $1 failed (re-run to resume)"
}
summarize() {  # summarize <stage>
  "$PY" "$DRIVER" summarize --out "$WORK" --stage "$1"
}

# ---------------------------------------------------------------- 2 prep
if ! is_done 2; then
  log "stage 2: prep dolly-15k split"
  "$PY" "$DRIVER" prep --out "$WORK" || die "prep failed"
  done_mark 2
fi

# ---------------------------------------------------------------- smoke
if want smoke && ! is_done smoke; then
  log "stage smoke: fp32 block in a bf16 model on CUDA"
  fetch "$S135"
  # ascending: the tied embed+head block goes first, so the fp32-embedding case
  # (fp32 residual stream, fp32 RoPE tables into SDPA) runs on CUDA here.
  run smoke_block smoke --model "$S135" --engine block --k 5 --order ascending --steps 20 --batch 4 --seq 256 --seed 0
  "$PY" - "$WORK/runs/smoke_block.json" <<'EOF' || die "smoke failed — stopping before the paid stages"
import json, math, sys
r = json.load(open(sys.argv[1]))
ok = r.get("status") == "ok" and r["losses"] and all(math.isfinite(x) for x in r["losses"]) \
     and len((r.get("engine_summary") or {}).get("switches", [])) >= 3
print("smoke:", r.get("status"), r.get("losses", [])[:3], "->", r.get("losses", [])[-3:], r.get("error", ""))
sys.exit(0 if ok else 1)
EOF
  done_mark smoke
fi

# ---------------------------------------------------------------- a quality
if want a && ! is_done a; then
  log "stage a: quality A/B, $STEPS steps, seeds $SEEDS"
  for M in "$Q15" "$S3"; do
    fetch "$M"
    base "$M"
    short="$(basename "$M" | tr 'A-Z.' 'a-z_')"
    for S in $SEEDS; do
      run "a_${short}_default_s$S" default --model "$M" --engine default --seed "$S" --steps "$STEPS"
      run "a_${short}_block_s$S" block_k50 --model "$M" --engine block --k 50 --seed "$S" --steps "$STEPS"
    done
  done
  summarize a && done_mark a
fi

# ---------------------------------------------------------------- b 7B
if want b && ! is_done b; then
  log "stage b: Qwen2.5-7B"
  fetch "$Q7"
  base "$Q7"
  run b_block_uncapped block_uncapped --model "$Q7" --engine block --k 30 --order descending \
      --steps 90 --batch 4 --save-reload
  run b_block_cap24 block_cap24 --model "$Q7" --engine block --k 10 --order descending \
      --steps 30 --batch 1 --vram-cap-gb 24 --no-eval
  run b_block_cap24_frozen block_cap24_frozen_embed --model "$Q7" --engine block --k 10 \
      --order descending --freeze-embeddings --steps 30 --batch 1 --vram-cap-gb 24 --no-eval
  run b_block_cap16_frozen block_cap16_frozen_embed --model "$Q7" --engine block --k 10 \
      --order descending --freeze-embeddings --steps 30 --batch 1 --vram-cap-gb 16 --no-eval
  run b_qlora qlora --model "$Q7" --qlora --steps 90 --batch 4
  summarize b && done_mark b
fi

# ---------------------------------------------------------------- c rounding
if want c && ! is_done c; then
  log "stage c: write-back rounding A/B, $STEPS_C steps"
  for M in "$S360" "$S3"; do
    fetch "$M"
    base "$M"
    short="$(basename "$M" | tr 'A-Z.' 'a-z_')"
    for K in 10 50 200; do
      for WB in nearest stochastic; do
        run "c_${short}_k${K}_${WB}" "k${K}_${WB}" --model "$M" --engine block --k "$K" \
            --writeback "$WB" --seed 0 --steps "$STEPS_C"
      done
    done
  done
  summarize c && done_mark c
fi

# ---------------------------------------------------------------- a2 one-pass K
if want a2 && ! is_done a2; then
  log "stage a2: Engine B with K=5 (one pass over all blocks in $STEPS steps)"
  for M in "$Q15" "$S3"; do
    short="$(basename "$M" | tr 'A-Z.' 'a-z_')"
    for S in $SEEDS; do
      run "a2_${short}_block_s$S" block_k5 --model "$M" --engine block --k 5 --seed "$S" --steps "$STEPS"
    done
  done
  summarize a2 && done_mark a2
fi

# ---------------------------------------------------------------- receipt
"$PY" - "$WORK" "$GIT_SHA" <<'EOF'
import glob, json, os, sys
work, sha = sys.argv[1], sys.argv[2]
env = json.load(open(os.path.join(work, "env.json")))
stages = {}
for f in sorted(glob.glob(os.path.join(work, "stage_*.json"))):
    s = json.load(open(f))
    failed = [k for k, v in s["checks"].items() if not v]
    stages[s["stage"]] = "PASS" if not failed else "FAIL " + ",".join(failed)
rec = {"git_sha": sha, "env": env, "stages": stages,
       "runs": sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(work, "runs", "*.json")))}
json.dump(rec, open(os.path.join(work, "receipt.json"), "w"), indent=1)
print(json.dumps(rec, indent=1))
for k, v in stages.items():
    print(f"RESULT stage {k}: {v}")
EOF
