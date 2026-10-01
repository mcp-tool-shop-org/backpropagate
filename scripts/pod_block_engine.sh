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
# Second run (after the first run's receipts in docs/receipts/2026-09-30-block-engine/
# and the external review): default STAGES="smoke d e4b e14b"
#   d    GSM8K (openai/gsm8k main, MIT), 1000 steps, batch 4, unpacked, seq from
#        the tokenized-length check (512, or 768 if >1% of gold '####' lines
#        would be cut). Primary metric: held-out loss on the 250 test answers;
#        secondary: strict '####' accuracy (greedy), with per-item outputs for
#        McNemar / paired bootstrap (scripts/pod_gsm8k_summary.py). Base
#        accuracies first; the 3B model furthest from floor/ceiling is picked.
#        Order (value first; the budget guard drops from the end):
#          7B engine B K=5 s0, QLoRA s0, engine B K=5 s1 -> d2 for the 7B pair
#          3B default / block_k5 / block_k50 / qlora, seeds 0 and 1
#          7B GaLore (layerwise; non-layerwise once if layerwise fails)
#          3B seed 2 (default, K5, K50), K5 at 5e-5 (2 seeds), Adafactor s0 -> d2 3B
#          3B qlora s2, Adafactor s1
#        Every run: NVML peak (10 Hz) next to torch max allocated / reserved,
#        expandable_segments:True, per-step train loss, per-block visit log,
#        first-batch mask/data-order hash, optimizer class + args, manifest.
#   d2   (inside d) pairs whose pooled held-out-loss CI spans 0 at 1000 steps
#        rerun at 2000 steps with the same seeds (d2_triggers.json)
#   e4b  Qwen/Qwen3.5-4B: what the text-only loader loads vs the checkpoint
#   e14b qwen2.5-14b at library defaults (batch auto), then with unsloth installed
# STOP_AFTER_EPOCH: a run is skipped if its estimated end is past this unix time.
#
# Usage (on the pod):
#   BRANCH=feat/block-coordinate-engine nohup bash pod_block_engine.sh > /dev/null 2>&1 &
#   tail -f /workspace/block_engine/pod.log
#
# Knobs (env): BRANCH, REPO, WORK (default /workspace/block_engine),
#   STAGES (default "smoke d e4b e14b"; first run: "smoke p a b c a2"), SEEDS (default "0 1 2"), STEPS (150),
#   STEPS_C (150), FORCE_STAGE=<name> to re-run one stage's summary.
#
# Pod lessons carried over from pod_offload_7b.sh: pip needs
# --break-system-packages (PEP 668); torch is constrained to the image's exact
# version; no `curl | head` under pipefail; no pkill anywhere.
set -uo pipefail

REPO="${REPO:-https://github.com/mcp-tool-shop-org/backpropagate.git}"
BRANCH="${BRANCH:-feat/block-coordinate-engine}"
WORK="${WORK:-/workspace/block_engine}"
STAGES="${STAGES:-smoke d e4b e14b}"
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
Q3="Qwen/Qwen2.5-3B-Instruct"
Q14="Qwen/Qwen2.5-14B-Instruct"
Q35_4="Qwen/Qwen3.5-4B"
D_STEPS="${D_STEPS:-1000}"
D_SEQ="${D_SEQ:-512}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export POD_IMAGE="${POD_IMAGE:-runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404}"
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-}"

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
  if [ -n "$STOP_AFTER_EPOCH" ] && [ $(( $(date +%s) + ${EST:-0} )) -gt "$STOP_AFTER_EPOCH" ]; then
    log "skip $tag (would end past STOP_AFTER_EPOCH, est ${EST:-0}s — budget)"; return 0
  fi
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
purge() {  # purge <model>: drop a model from the HF cache (blobs included) to free disk
  "$PY" - "$1" <<'EOF' || log "purge $1 failed (non-fatal)"
import sys
from huggingface_hub import scan_cache_dir
info = scan_cache_dir()
revs = [r.commit_hash for repo in info.repos if repo.repo_id == sys.argv[1] for r in repo.revisions]
if revs:
    s = info.delete_revisions(*revs)
    print(f"purge {sys.argv[1]}: freeing {s.expected_freed_size_str}")
    s.execute()
EOF
}
base_gsm() {  # base_gsm <model>
  local f="$WORK/runs/base_gsm8k_${1//\//_}.json"
  if [ -f "$f" ]; then return 0; fi
  log "base (gsm8k) $1"
  "$PY" "$DRIVER" base --out "$WORK" --dataset gsm8k --model "$1" || log "base $1: accuracy OUT OF BOUNDS (see receipt)"
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

# ---------------------------------------------------------------- p preset check
# The qwen3.5-4b preset ships Qwen/Qwen3.5-4B-Instruct (not on the Hub).
# Qwen/Qwen3.5-4B exists but is tagged image-text-to-text: does the library's
# text-only loader load it, train 5 LoRA steps and generate?
if want p && ! is_done p; then
  log "stage p: qwen3.5-4b preset check"
  "$PY" - > "$WORK/runs/p_hub.json" <<'EOF' || true
import json
from huggingface_hub import model_info
out = {}
for rid in ("Qwen/Qwen3.5-4B-Instruct", "Qwen/Qwen3.5-4B"):
    try:
        mi = model_info(rid)
        out[rid] = {"exists": True, "pipeline_tag": mi.pipeline_tag,
                    "architectures": (mi.config or {}).get("architectures")}
    except Exception as exc:  # noqa: BLE001
        out[rid] = {"exists": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
print(json.dumps(out, indent=1))
EOF
  cat "$WORK/runs/p_hub.json"
  run p_qwen35_4b_lora preset_qwen35_4b --model "Qwen/Qwen3.5-4B" --qlora --steps 5 --batch 2 \
      --seq 256 --no-eval --gen-inprocess
  "$PY" -c "import json; r=json.load(open('$WORK/runs/p_qwen35_4b_lora.json')); print('preset p:', r.get('status'), r.get('model_class'), repr(r.get('generation')), r.get('losses'), r.get('error', '')[:800])"
  rm -rf "$HF_HOME/hub/models--Qwen--Qwen3.5-4B"  # free the container disk for 7B
  done_mark p
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

# ---------------------------------------------------------------- d GSM8K
# Arms. Every arm: library Trainer, unpacked, batch 4, gradient accumulation 1,
# same seed -> same data order, full-sequence loss (no Unsloth response masking
# anywhere; checked per arm by mask_check), the library's cosine schedule with
# its 10 warmup steps. LR: each arm's library default unless named.
arm_args() {  # arm_args <size> <arm> -> driver args
  local M
  if [ "$1" = 7b ]; then M="$Q7"; else M="$(cat "$WORK/d_pick.model")"; fi
  case "$2" in
    default)            echo "--model $M --engine default --lr-default" ;;
    default_adafactor)  echo "--model $M --engine default --lr-default --optim adafactor" ;;
    block_k5)           echo "--model $M --engine block --k 5 --lr-default" ;;
    block_k50)          echo "--model $M --engine block --k 50 --lr-default" ;;
    block_k5_lr5e-5)    echo "--model $M --engine block --k 5 --lr 5e-5" ;;
    qlora)              echo "--model $M --qlora --lr-default" ;;
    galore)             echo "--model $M --engine default --lr-default --ceiling 8 --galore galore_adamw_8bit_layerwise" ;;
    galore_nonlayerwise) echo "--model $M --engine default --lr-default --ceiling 8 --galore galore_adamw_8bit" ;;
  esac
}
# Rough wall-clock per 1000-step run incl. load + eval (s), for the budget guard.
est() {
  case "$1_$2" in
    7b_block_k5) echo 420 ;; 7b_qlora) echo 900 ;; 7b_galore*) echo 1500 ;;
    3b_default) echo 420 ;; 3b_default_adafactor) echo 400 ;; 3b_qlora) echo 660 ;;
    *) echo 260 ;;
  esac
}
darm() {  # darm <size> <arm> <seed> [steps] : one stage-d run
  local size="$1" arm="$2" seed="$3" steps="${4:-$D_STEPS}" pre="d"
  [ "$steps" = "$D_STEPS" ] || pre="d2"
  local e; e="$(est "$size" "$arm")"; [ "$steps" = "$D_STEPS" ] || e=$((e * 17 / 10))
  # shellcheck disable=SC2046
  EST="$e" run "${pre}_${size}_${arm}_s${seed}" "$arm" $(arm_args "$size" "$arm") --seed "$seed" \
      --dataset gsm8k --steps "$steps" --batch 4 --seq "$D_SEQ"
}
d2_from_triggers() {  # d2_from_triggers <size>: rerun triggered pairs at 2x steps
  [ -f "$WORK/d2_triggers.json" ] || return 0
  "$PY" - "$WORK/d2_triggers.json" "$1" > "$WORK/d2_todo_$1.txt" <<'EOF'
import json, sys
seen = set()
for t in json.load(open(sys.argv[1])):
    if t["size"] != sys.argv[2]:
        continue
    for arm in t["arms"]:
        for s in t["seeds"][arm]:
            if (arm, s) not in seen:
                seen.add((arm, s))
                print(arm, s)
EOF
  if [ -s "$WORK/d2_todo_$1.txt" ]; then log "d2 triggered at $1: $(tr '\n' ' ' < "$WORK/d2_todo_$1.txt")"; fi
  while read -r arm s; do darm "$1" "$arm" "$s" $((D_STEPS * 2)); done < "$WORK/d2_todo_$1.txt"
}
dsum() { "$PY" "$SRC/scripts/pod_gsm8k_summary.py" --out "$WORK" --prefix "${1:-d}" || log "summary failed"; }

if want d && ! is_done d; then
  log "stage d: GSM8K, $D_STEPS steps (stop-after: ${STOP_AFTER_EPOCH:-none})"
  "$PY" -m pip install --break-system-packages -q -c "$WORK/constraints.txt" nvidia-ml-py \
    || log "nvidia-ml-py install failed: NVML peaks will be missing"
  "$PY" -m pip install --break-system-packages -q --no-deps galore-torch \
    || log "galore-torch install failed: the GaLore arm will record the error"
  if [ ! -f "$WORK/prep_gsm8k.json" ]; then
    "$PY" "$DRIVER" prep --out "$WORK" --dataset gsm8k || die "gsm8k prep failed"
  fi
  if [ ! -f "$WORK/d_pick.model" ]; then
    for M in "$S3" "$Q3"; do fetch "$M"; base_gsm "$M"; done
    # Pick the 3B model whose base accuracy is furthest from floor (0) and ceiling (1).
    "$PY" - "$WORK" "$S3" "$Q3" <<'EOF' || die "3B pick failed"
import json, os, sys
work, cands = sys.argv[1], sys.argv[2:]
acc = {}
for m in cands:
    f = os.path.join(work, "runs", "base_gsm8k_" + m.replace("/", "_") + ".json")
    acc[m] = json.load(open(f))["acc_strict"]
score = {m: min(a, 1 - a) for m, a in acc.items()}
pick = max(cands, key=lambda m: (score[m], -acc[m]))
rec = {"pick": pick, "base_acc_strict": acc, "distance_from_floor_or_ceiling": score,
       "rule": "max over candidates of min(acc, 1 - acc); tie -> lower accuracy (more room)"}
json.dump(rec, open(os.path.join(work, "d_pick.json"), "w"), indent=1)
open(os.path.join(work, "d_pick.model"), "w").write(pick)
print("3B pick:", json.dumps(rec))
EOF
    for M in "$S3" "$Q3"; do [ "$M" = "$(cat "$WORK/d_pick.model")" ] || purge "$M"; done
  fi
  fetch "$Q7"
  base_gsm "$Q7"
  # Sequence length from the tokenized answers (both tokenizers; the larger choice wins).
  for M in "$(cat "$WORK/d_pick.model")" "$Q7"; do
    [ -f "$WORK/runs/lengths_${M//\//_}.json" ] || "$PY" "$DRIVER" lengths --out "$WORK" --dataset gsm8k --model "$M"
  done
  D_SEQ="$("$PY" -c "import glob,json; print(max(json.load(open(f))['seq_choice'] for f in glob.glob('$WORK/runs/lengths_*.json')))")"
  log "stage d: max_seq_length $D_SEQ (see runs/lengths_*.json)"

  # 1. The decision pair first: 7B engine B K=5 (2 seeds) vs QLoRA (1 seed).
  darm 7b block_k5 0; darm 7b qlora 0; darm 7b block_k5 1
  dsum d; d2_from_triggers 7b
  # 2. 3B core arms, seeds 0 and 1.
  for S in 0 1; do for A in default block_k5 block_k50 qlora; do darm 3b "$A" "$S"; done; done
  # 3. GaLore at 7B: layerwise first; non-layerwise once if layerwise failed.
  darm 7b galore 0
  if ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$WORK/runs/d_7b_galore_s0.json')).get('status')=='ok' else 1)" 2>/dev/null \
     && [ -f "$WORK/runs/d_7b_galore_s0.json" ]; then
    darm 7b galore_nonlayerwise 0
  fi
  # 4. 3B seed 2 (core), the lr probe and the Adafactor control.
  for A in default block_k5 block_k50; do darm 3b "$A" 2; done
  darm 3b block_k5_lr5e-5 0; darm 3b block_k5_lr5e-5 1
  darm 3b default_adafactor 0
  dsum d; d2_from_triggers 3b
  # 5. Lowest priority (dropped first by the budget guard).
  darm 3b qlora 2
  darm 3b default_adafactor 1
  dsum d
  [ -n "$(ls "$WORK"/runs/d2_*.json 2>/dev/null)" ] && dsum d2
  done_mark d
fi

# ---------------------------------------------------------------- e4b qwen3.5-4b
if want e4b && ! is_done e4b; then
  log "stage e4b: Qwen/Qwen3.5-4B through the text-only loader"
  if [ ! -f "$WORK/runs/e_qwen35_4b_inspect.json" ]; then
    "$PY" "$DRIVER" inspect --out "$WORK" --tag e_qwen35_4b_inspect --model "$Q35_4" \
      || log "inspect $Q35_4 failed"
  fi
  run e_qwen35_4b_qlora_defaults qlora_library_defaults --model "$Q35_4" --qlora --library-defaults \
      --dataset gsm8k --steps 10 --no-eval
  "$PY" -c "import json; r=json.load(open('$WORK/runs/e_qwen35_4b_inspect.json')); print('e4b:', {k: r.get(k) for k in ('loaded_class','loaded_params','loaded_vision_params','hub_safetensors_total','checkpoint_vision_tensors','peak_vram_alloc_gib_bf16_load')})" || true
  purge "$Q35_4"
  done_mark e4b
fi

# ---------------------------------------------------------------- e14b
if want e14b && ! is_done e14b; then
  log "stage e14b: qwen2.5-14b at library defaults (batch auto)"
  for M in "$S3" "$Q3" "$Q7" "$S135" "$Q35_4"; do purge "$M"; done
  fetch "$Q14"
  run e_14b_defaults qlora_library_defaults --model "$Q14" --qlora --library-defaults \
      --dataset gsm8k --steps 20 --no-eval
  if [ ! -f "$WORK/runs/e_14b_defaults_unsloth.json" ] && \
     { [ -z "$STOP_AFTER_EPOCH" ] || [ "$(date +%s)" -le "$STOP_AFTER_EPOCH" ]; }; then
    log "installing the [unsloth] extra (torch pinned to the image)"
    timeout 900 "$PY" -m pip install --break-system-packages -q -c "$WORK/constraints.txt" \
      -e "$SRC[unsloth]" > "$WORK/unsloth_install.log" 2>&1 \
      && "$PY" -c "import importlib.metadata as m; print('unsloth', m.version('unsloth'), 'transformers', m.version('transformers'), 'trl', m.version('trl'))" \
      || log "unsloth install failed (see unsloth_install.log)"
    run e_14b_defaults_unsloth qlora_library_defaults_unsloth --model "$Q14" --qlora \
        --library-defaults --dataset gsm8k --steps 20 --no-eval
  fi
  done_mark e14b
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
