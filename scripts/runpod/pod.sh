#!/usr/bin/env bash
# scripts/runpod/pod.sh — create, inspect and delete RunPod GPU pods for the
# real-GPU experiments (see docs/handoff-2026-09-30-full-ft-experiments.md).
#
# Needs: RUNPOD_API_KEY in the environment (never printed), curl, python,
# and an SSH public key (default ~/.ssh/runpod_rustline.pub).
#
# Disk (env, optional):
#   RUNPOD_CONTAINER_DISK_GB  container disk, default 100. The HF cache belongs
#                             here: /workspace is often a slow network mount.
#                             Size it to the models the run downloads (the
#                             14B/24B/32B QLoRA presets are bf16 repos, ~140 GB
#                             together: use 200, or delete each model after use).
#   RUNPOD_VOLUME_GB          pod volume at /workspace, default 0 (none). A
#                             one-shot experiment pod needs no persistent volume.
#
# Lessons this script encodes (2026-09-30):
#   * Use curl. Cloudflare rejects Python's urllib on rest.runpod.io with
#     HTTP 403 / "error code: 1010".
#   * The v1 API is https://rest.runpod.io/v1 ; SSH lives in the pod's
#     `publicIp` + `portMappings["22"]`, which take 2-5 minutes to appear.
#   * Capacity for one GPU type is often zero on every tier at once: `create`
#     tries SECURE then COMMUNITY; `create-retry` keeps trying.
#   * Always arm the dead-man timer (scripts/runpod/deadman.py) right after a
#     create, and verify CUDA over SSH before using a pod: one host on
#     2026-09-30 came up with a broken GPU ("CUDA unknown error").
#
# Usage:
#   scripts/runpod/pod.sh create  [GPU] [MIN_RAM_GB] [NAME]
#   scripts/runpod/pod.sh create-retry [GPU] [MIN_RAM_GB] [NAME] [MINUTES]
#   scripts/runpod/pod.sh ssh-info POD_ID        # waits for IP + port
#   scripts/runpod/pod.sh verify  POD_ID          # GPU, RAM, disk, CUDA
#   scripts/runpod/pod.sh list
#   scripts/runpod/pod.sh delete  POD_ID
set -euo pipefail

API=https://rest.runpod.io/v1
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404}"
PUBKEY_FILE="${RUNPOD_PUBKEY:-$HOME/.ssh/runpod_rustline.pub}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/runpod_rustline}"
DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-100}"
VOLUME_GB="${RUNPOD_VOLUME_GB:-0}"
AUTH=(-H "Authorization: Bearer ${RUNPOD_API_KEY}")

_create_once() {  # gpu cloud min_ram name -> prints pod id or nothing
  local body resp
  body=$(python - "$1" "$2" "$3" "$4" "$IMAGE" "$PUBKEY_FILE" "$DISK_GB" "$VOLUME_GB" <<'PY'
import json, os, sys
gpu, cloud, ram, name, image, pub, disk, volume = sys.argv[1:9]
body = {
    "name": name, "imageName": image, "cloudType": cloud,
    "gpuTypeIds": [gpu], "gpuCount": 1,
    "minRAMPerGPU": int(ram), "minVCPUPerGPU": 8,
    "containerDiskInGb": int(disk), "volumeInGb": int(volume),
    "ports": ["22/tcp"], "env": {"PUBLIC_KEY": open(os.path.expanduser(pub)).read().strip()},
    "supportPublicIp": True,
}
if int(volume) > 0:
    body["volumeMountPath"] = "/workspace"
print(json.dumps(body))
PY
)
  # Capture the response, then parse it: no download is piped into an
  # interpreter (OpenSSF Scorecard Pinned-Dependencies: downloadThenRun).
  resp=$(curl -s -X POST "$API/pods" "${AUTH[@]}" -H "Content-Type: application/json" --data "$body")
  python -c "import json,sys; d=json.load(sys.stdin); print(d.get('id') or '', file=sys.stdout); print(json.dumps({k:d.get(k) for k in ('id','costPerHr','memoryInGb','machineId','error')}), file=sys.stderr)" <<<"$resp"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  create|create-retry)
    GPU="${1:-NVIDIA GeForce RTX 5090}"; RAM="${2:-64}"; NAME="${3:-bp-experiment}"; MINUTES="${4:-45}"
    deadline=$(( $(date +%s) + MINUTES * 60 ))
    while :; do
      for cloud in SECURE COMMUNITY; do
        id=$(_create_once "$GPU" "$cloud" "$RAM" "$NAME")
        if [ -n "$id" ]; then echo "$id"; exit 0; fi
      done
      [ "$cmd" = "create" ] && { echo "no capacity for '$GPU'" >&2; exit 1; }
      [ "$(date +%s)" -ge "$deadline" ] && { echo "gave up after ${MINUTES} min" >&2; exit 1; }
      echo "$(date +%H:%M) no capacity; retrying in 3 min" >&2; sleep 180
    done ;;
  ssh-info)
    for _ in $(seq 1 20); do
      resp=$(curl -s "$API/pods/$1" "${AUTH[@]}")
      out=$(python -c "import json,sys; d=json.load(sys.stdin); pm=d.get('portMappings') or {}; print(d.get('publicIp') or '', pm.get('22',''))" <<<"$resp")
      set -- "$1" $out
      if [ -n "${2:-}" ] && [ -n "${3:-}" ]; then echo "ssh -i $SSH_KEY -p $3 root@$2"; exit 0; fi
      sleep 30
    done
    echo "no SSH endpoint after 10 min — delete and redeploy (another host)" >&2; exit 1 ;;
  verify)
    # ssh-info prints "ssh -i KEY -p PORT root@IP"; take PORT and HOST from it.
    read -r _ _ _ _ port host <<<"$("$0" ssh-info "$1")"
    ssh -i "$SSH_KEY" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 "$host" \
      "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | sed -n 2p; df -h / /workspace 2>/dev/null | tail -n +2; python -c 'import torch;print(torch.__version__, torch.cuda.is_available())'" ;;
  list)
    resp=$(curl -s "$API/pods" "${AUTH[@]}")
    python -c "import json,sys; [print(p['id'], p.get('name'), p.get('desiredStatus'), p.get('costPerHr')) for p in json.load(sys.stdin)]" <<<"$resp" ;;
  delete)
    curl -s -o /dev/null -w "DELETE $1 -> HTTP %{http_code}\n" -X DELETE "$API/pods/$1" "${AUTH[@]}" ;;
  *)
    sed -n '2,34p' "$0"; exit 2 ;;
esac
