#!/usr/bin/env bash
# pod_offload_7b.sh — the real "single-card 7B full fine-tune" run, on a pod.
#
# Target: a fresh RunPod pod, image runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404
# (CUDA 12.8.1, torch 2.8.0 — sm_120 / RTX 5090 capable). Any image with a
# CUDA-12.8+ torch >= 2.6 works: the script uses the IMAGE's torch and never
# reinstalls it (torchvision/torchaudio are ABI-bound to it).
#
# What it does (each stage is resumable — a marker in $WORK/.done/ skips it on
# re-run, and everything is logged to $WORK/pod.log, so a dropped SSH session
# loses nothing; run it under `nohup` or tmux):
#   0 preflight   GPU / VRAM / host RAM / versions -> $WORK/env.json
#   1 install     clone $BRANCH, pip install -e with torch pinned to the image's
#   2 gate        $0 pre-check on SmolLM2-135M: measures the build's host
#                 bytes/param and FAILS FAST if 7.6B would not fit $RSS_CEIL_GB
#                 (so a build that still stores 16 B/param never downloads 15 GB)
#   3 download    Qwen/Qwen2.5-7B-Instruct (safetensors + json only, ~15.2 GB)
#   4 train       mode="full", full_ft_offload=True, $STEPS steps, under a
#                 60 GB host-RSS ceiling (cgroup memory.max if writable, else an
#                 in-process watchdog that FAILS the run above the ceiling)
#   5 verify      save -> reload -> generate
#   6 receipt     $WORK/receipt.json + a PASS/FAIL line
#
# Usage (on the pod):
#   BRANCH=feat/offload-7b nohup bash pod_offload_7b.sh > /dev/null 2>&1 &
#   tail -f /workspace/offload7b/pod.log
#
# Knobs (env): BRANCH, REPO, WORK (default /workspace/offload7b), MODEL,
#   STEPS (10), SEQ (512), RSS_CEIL_GB (60), BP_TRAINER_KWARGS (JSON merged into
#   the Trainer(...) call — the Stage-2 engine selection goes here until it is
#   the default), FORCE_STAGE=<n> to re-run one stage.
#
# Pod-vs-rig lessons baked in: pip needs --break-system-packages (PEP 668);
# torch is constrained to the image's exact version; no `curl | head` under
# pipefail; no pkill anywhere (never pattern-kill over ssh).
set -uo pipefail

REPO="${REPO:-https://github.com/mcp-tool-shop-org/backpropagate.git}"
BRANCH="${BRANCH:-feat/offload-7b}"
WORK="${WORK:-/workspace/offload7b}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
STEPS="${STEPS:-10}"
SEQ="${SEQ:-512}"
RSS_CEIL_GB="${RSS_CEIL_GB:-60}"
BP_TRAINER_KWARGS="${BP_TRAINER_KWARGS:-"{}"}"
export HF_HOME="${HF_HOME:-$WORK/hf}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
PY="${PY:-python3}"

mkdir -p "$WORK/.done" "$HF_HOME"
LOG="$WORK/pod.log"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
done_mark() { touch "$WORK/.done/$1"; }
is_done() { [ -f "$WORK/.done/$1" ] && [ "${FORCE_STAGE:-}" != "$1" ]; }
die() { log "FAIL: $*"; echo "RESULT: FAIL ($*)"; exit 1; }

# ---------------------------------------------------------------- 0 preflight
if ! is_done 0; then
  log "stage 0: preflight"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || die "no GPU"
  free -g
  df -h "$WORK"
  "$PY" - <<'EOF' > "$WORK/env.json" || die "torch import failed"
import json, torch, platform, os
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
  "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
  "host_ram_total_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)}))
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
  # Constrain torch & friends to exactly what the image ships.
  "$PY" - <<'EOF' > "$WORK/constraints.txt"
import importlib.metadata as md
for pkg in ("torch", "torchvision", "torchaudio", "triton"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        pass
EOF
  cat "$WORK/constraints.txt"
  "$PY" -m pip install --break-system-packages -q -c "$WORK/constraints.txt" -e "$SRC" psutil \
    || die "pip install failed"
  "$PY" -c "import torch; print('torch after install', torch.__version__)"
  done_mark 1
fi
SHA="$(git -C "$SRC" rev-parse HEAD)"
log "git sha $SHA"

# The runner (written every time so a re-run picks up script edits).
RUNNER="$WORK/run_offload.py"
cat > "$RUNNER" <<'PYEOF'
"""One full-FT run with a receipt.

argv: model steps seq ceil_gb out_json [gate]
env:  BP_TRAINER_KWARGS (JSON merged into Trainer(...)), OFFLOAD=0 for the pure-GPU
      full-FT path, VRAM_CAP_GB to emulate a smaller card
      (torch.cuda.set_per_process_memory_fraction), WORK (save dir root).
"""
import gc, json, math, os, sys, tempfile, threading, time
import importlib.metadata as md
import psutil, torch

model_id, steps, seq, ceil_gb, out_json = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
gate = len(sys.argv) > 6 and sys.argv[6] == "gate"
offload = os.environ.get("OFFLOAD", "1") != "0"
extra = json.loads(os.environ.get("BP_TRAINER_KWARGS", "{}") or "{}")
vram_cap = float(os.environ.get("VRAM_CAP_GB", "0") or 0)
GB = 2 ** 30
proc = psutil.Process()
peak = {"rss": 0}

def watchdog():
    while True:
        r = proc.memory_info().rss
        peak["rss"] = max(peak["rss"], r)
        if r > ceil_gb * GB:
            print(f"RESULT: FAIL (host RSS {r / GB:.1f} GB exceeded the {ceil_gb} GB ceiling)", flush=True)
            os._exit(3)
        time.sleep(0.05)

threading.Thread(target=watchdog, daemon=True).start()
total_vram = torch.cuda.get_device_properties(0).total_memory
if vram_cap:
    torch.cuda.set_per_process_memory_fraction(min(1.0, vram_cap * GB / total_vram), 0)

from backpropagate.trainer import Trainer, TrainingCallback

def local(t):
    f = getattr(t, "to_local", None)
    return f() if callable(f) else t

rows = [("Explain what a hash table is.", "A hash table maps keys to slots with a hash function so lookups take expected constant time."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Name three primary colors.", "Red, blue and yellow are the traditional primary colors."),
        ("What does CPU stand for?", "CPU stands for central processing unit."),
        ("Define recursion briefly.", "Recursion is a function calling itself on a smaller input until a base case."),
        ("What is 12 times 12?", "12 times 12 is 144."),
        ("What is photosynthesis?", "Photosynthesis turns light, water and carbon dioxide into sugar and oxygen."),
        ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet.")]
d = tempfile.mkdtemp()
data = os.path.join(d, "overfit.jsonl")
# Pad each row with a long, fixed passage so every example reaches `seq` tokens.
filler = " ".join(f"Note {i}: the quick brown fox jumps over the lazy dog near river bend {i}." for i in range(400))
with open(data, "w") as fh:
    for q, a in rows:
        fh.write(json.dumps({"messages": [{"role": "user", "content": q + " " + filler}, {"role": "assistant", "content": a}]}) + "\n")

def sample(model):
    out, g = {}, torch.Generator().manual_seed(0)
    for name, p in model.named_parameters():
        w = local(p).detach().reshape(-1)
        idx = torch.randint(0, w.numel(), (min(4096, w.numel()),), generator=g)
        out[name] = (idx, w.index_select(0, idx.to(w.device)).float().cpu().clone())
    return out

def pct_changed(model, snap):
    changed = total = 0
    for name, p in model.named_parameters():
        if name not in snap:
            continue
        idx, before = snap[name]
        w = local(p).detach().reshape(-1)
        after = w.index_select(0, idx.to(w.device)).float().cpu()
        changed += int((after != before).sum()); total += idx.numel()
    return round(100.0 * changed / max(1, total), 3)

losses, stamps, pct1 = [], [], {}
kwargs = dict(model=model_id, use_unsloth=False, mode="full", full_ft_offload=offload, max_seq_length=seq,
              batch_size=1, gradient_accumulation=1, output_dir=os.path.join(d, "out"), report_to="none")
kwargs.update(extra)
t = Trainer(**kwargs)
t0 = time.perf_counter()
t.load_model()
load_s = time.perf_counter() - t0
snap = sample(t._model)

def on_step(step, loss):
    losses.append(round(float(loss), 4))
    stamps.append(time.perf_counter())
    if step == 1:
        pct1["v"] = pct_changed(t._model, snap)

torch.cuda.reset_peak_memory_stats()
run = t.train(data, steps=steps, callback=TrainingCallback(on_step=on_step))
if not losses:  # SFTTrainer path (OFFLOAD=0) reports via run.loss_history
    losses = [round(x, 4) for x in run.loss_history]
step_times = run.metadata.get("step_times") or []
params = list(t._model.parameters())
n = sum(p.numel() for p in params)
p_bytes = sum(local(p).numel() * local(p).element_size() for p in params)
rec = {
    "model": model_id, "params": n, "steps": steps, "seq": seq, "offload": offload,
    "engine": run.metadata.get("engine", "sfttrainer"),
    "vram_cap_gb": vram_cap or None,
    "gpu": torch.cuda.get_device_name(0), "vram_total_gb": round(total_vram / GB, 2),
    "host_ram_total_gb": round(psutil.virtual_memory().total / GB, 1),
    "rss_ceiling_gb": ceil_gb,
    "versions": {p: md.version(p) for p in ("torch", "transformers", "trl", "accelerate", "peft", "backpropagate")},
    "param_dtype": str(params[0].dtype), "param_device": str(local(params[0]).device),
    "host_param_B_per_param": round(p_bytes / n, 3),
    "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / GB, 2),
    "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 2),
    "peak_rss_train_gb": round(peak["rss"] / GB, 2),
    "load_s": round(load_s, 1),
    "s_per_step": round(sum(step_times[1:]) / max(1, len(step_times) - 1), 3) if len(step_times) > 1 else None,
    "losses": losses, "final_loss": run.final_loss,
    "pct_params_changed_step1": pct1.get("v"),
    "pct_params_changed_final": pct_changed(t._model, snap),
}
if gate:
    rec["projected_7p6b_host_gb"] = round(7.6e9 * 2 * rec["host_param_B_per_param"] / GB + 8, 1)
else:
    save_dir = os.path.join(os.environ.get("WORK", d), "saved_" + model_id.replace("/", "_"))
    saved = t.save(save_dir, run_id=run.run_id)
    del t, run, params
    gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(saved)
    m = AutoModelForCausalLM.from_pretrained(saved, dtype=torch.bfloat16, device_map="cuda")
    ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to("cuda")
    out = m.generate(ids, max_new_tokens=16, do_sample=False)
    rec["generation"] = tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
    rec["saved_dtype"] = str(next(m.parameters()).dtype)
    rec["peak_rss_total_gb"] = round(peak["rss"] / GB, 2)
    del m
    import shutil
    shutil.rmtree(saved, ignore_errors=True)  # keep the volume free for the next rung
json.dump(rec, open(out_json, "w"), indent=1)
print("RECEIPT " + json.dumps(rec), flush=True)
PYEOF

# ---------------------------------------------------------------- 2 gate
if ! is_done 2; then
  log "stage 2: \$0 bytes/param gate on SmolLM2-135M"
  WORK="$WORK" BP_TRAINER_KWARGS="$BP_TRAINER_KWARGS" "$PY" "$RUNNER" HuggingFaceTB/SmolLM2-135M-Instruct 3 128 "$RSS_CEIL_GB" "$WORK/gate.json" gate \
    || die "gate run crashed"
  PROJ="$("$PY" -c "import json;print(json.load(open('$WORK/gate.json'))['projected_7p6b_host_gb'])")"
  log "this build projects ${PROJ} GB of host state for 7.6B (ceiling ${RSS_CEIL_GB} GB)"
  "$PY" -c "import sys; sys.exit(0 if $PROJ < $RSS_CEIL_GB * 0.85 else 1)" \
    || die "build stores too many bytes/param: 7.6B projects to ${PROJ} GB > 85% of ${RSS_CEIL_GB} GB — not downloading"
  done_mark 2
fi

# ---------------------------------------------------------------- 3 download
if ! is_done 3; then
  log "stage 3: download $MODEL"
  "$PY" -m pip install --break-system-packages -q "huggingface_hub[cli]" -c "$WORK/constraints.txt" || true
  "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja']))" \
    || die "download failed (re-run to resume)"
  done_mark 3
fi

# ---------------------------------------------------------------- 4+5 train + verify
if ! is_done 4; then
  log "stage 4: train $MODEL steps=$STEPS seq=$SEQ under ${RSS_CEIL_GB} GB"
  CG=""
  if [ -w /sys/fs/cgroup/cgroup.subtree_control ] && mkdir -p /sys/fs/cgroup/offload7b 2>/dev/null \
     && echo "$((RSS_CEIL_GB * 1024 * 1024 * 1024))" > /sys/fs/cgroup/offload7b/memory.max 2>/dev/null; then
    CG=/sys/fs/cgroup/offload7b
    log "cgroup ceiling active: $CG memory.max=${RSS_CEIL_GB}G"
  else
    log "cgroup not writable; the in-process RSS watchdog enforces ${RSS_CEIL_GB} GB"
  fi
  run_train() {
    WORK="$WORK" BP_TRAINER_KWARGS="$BP_TRAINER_KWARGS" HF_HUB_OFFLINE=1 \
      "$PY" "$RUNNER" "$MODEL" "$STEPS" "$SEQ" "$RSS_CEIL_GB" "$WORK/receipt.json"
  }
  if [ -n "$CG" ]; then
    ( echo "$BASHPID" > "$CG/cgroup.procs" 2>/dev/null || log "cgroup attach failed; watchdog only"
      run_train ) || die "train/verify failed (see $LOG)"
  else
    run_train || die "train/verify failed (see $LOG)"
  fi
  done_mark 4
fi

# ---------------------------------------------------------------- 6 receipt
"$PY" - "$WORK/receipt.json" "$SHA" "$RSS_CEIL_GB" <<'EOF'
import json, math, sys
r = json.load(open(sys.argv[1])); r["git_sha"] = sys.argv[2]; ceil = float(sys.argv[3])
checks = {
  "finite_losses": all(math.isfinite(x) for x in r["losses"]),
  "loss_decreased": len(r["losses"]) >= 4 and sum(r["losses"][-3:]) / 3 < r["losses"][0] * 0.9,
  "params_changed": (r.get("pct_params_changed_final") or 0) >= 50.0,
  "rss_under_ceiling": r["peak_rss_total_gb"] < ceil,
  "vram_fits": r["peak_vram_reserved_gb"] < r["vram_total_gb"],
  "generated": bool(r.get("generation", "").strip()),
}
r["checks"] = checks
json.dump(r, open(sys.argv[1], "w"), indent=1)
print(json.dumps(r, indent=1))
print("RESULT: " + ("PASS" if all(checks.values()) else "FAIL " + ",".join(k for k, v in checks.items() if not v)))
EOF
