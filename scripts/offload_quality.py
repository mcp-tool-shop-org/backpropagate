"""Held-out quality check: the offload engine vs the pure-GPU full-FT path.

Both paths train the same model, on the same real instruct data, for the same
number of steps. Held-out loss is then measured before and after training on a
split disjoint from the training data. The data is databricks/databricks-dolly-15k
(CC BY-SA 3.0), with context folded into the user turn and a fixed-seed shuffle.

Usage (one mode per process; FSDP needs its own process group):
    python scripts/offload_quality.py prep  --out /workspace/quality
    python scripts/offload_quality.py base  --out /workspace/quality --model M
    python scripts/offload_quality.py train --out /workspace/quality --model M --offload 1 --steps 150
    python scripts/offload_quality.py train --out /workspace/quality --model M --offload 0 --steps 150

Each mode appends one JSON line to <out>/quality.jsonl. Loss is token-weighted
full-sequence cross-entropy, the objective both paths train. It is computed
on the exact text the library feeds the trainer (DatasetLoader.to_hf_dataset).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import time

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["prep", "base", "train"])
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B")
ap.add_argument("--offload", type=int, default=1)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-train", type=int, default=400)
ap.add_argument("--n-heldout", type=int, default=150)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
TRAIN = os.path.join(args.out, "train.jsonl")
HELD = os.path.join(args.out, "heldout.jsonl")


def emit(rec: dict) -> None:
    rec["time"] = time.strftime("%H:%M:%S")
    with open(os.path.join(args.out, "quality.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("QUALITY " + json.dumps(rec), flush=True)


if args.mode == "prep":
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = []
    for ex in ds:
        user = ex["instruction"].strip()
        if ex.get("context"):
            user += "\n\n" + ex["context"].strip()
        resp = ex["response"].strip()
        if 20 <= len(resp) <= 1500 and len(user) <= 1500:
            rows.append({"messages": [{"role": "user", "content": user},
                                      {"role": "assistant", "content": resp}]})
    random.Random(args.seed).shuffle(rows)
    need = args.n_train + args.n_heldout
    train, held = rows[: args.n_train], rows[args.n_train: need]
    for path, part in ((TRAIN, train), (HELD, held)):
        with open(path, "w") as fh:
            for r in part:
                fh.write(json.dumps(r) + "\n")
    emit({"mode": "prep", "dataset": "databricks/databricks-dolly-15k", "license": "CC BY-SA 3.0",
          "n_train": len(train), "n_heldout": len(held), "seed": args.seed, "pool": len(rows)})
    raise SystemExit(0)

import torch  # noqa: E402


def heldout_texts() -> list[str]:
    from backpropagate.datasets import DatasetLoader

    return list(DatasetLoader(HELD, validate=False).to_hf_dataset()["text"])


@torch.no_grad()
def heldout_loss(model_path: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    total, count = 0.0, 0
    for text in heldout_texts():
        ids = tok(text, truncation=True, max_length=args.seq, add_special_tokens=False,
                  return_tensors="pt")["input_ids"].cuda()
        if ids.shape[-1] < 2:
            continue
        out = model(input_ids=ids, labels=ids)
        n = ids.shape[-1] - 1
        total += float(out.loss) * n
        count += n
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"heldout_loss": round(total / count, 4), "heldout_tokens": count}


if args.mode == "base":
    rec = {"mode": "base", "model": args.model}
    rec.update(heldout_loss(args.model))
    emit(rec)
    raise SystemExit(0)

# ---- train
from backpropagate.trainer import Trainer  # noqa: E402

torch.manual_seed(args.seed)
random.seed(args.seed)
tag = "offload" if args.offload else "puregpu"
t = Trainer(model=args.model, use_unsloth=False, mode="full", full_ft_offload=bool(args.offload),
            max_seq_length=args.seq, batch_size=args.batch, gradient_accumulation=1, packing=False,
            output_dir=os.path.join(args.out, f"out_{tag}"), report_to="none")
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
run = t.train(TRAIN, steps=args.steps)
train_s = time.perf_counter() - t0
st = run.metadata.get("step_times") or []
rec = {
    "mode": "train", "engine": tag, "model": args.model, "steps": args.steps, "batch": args.batch,
    "seq": args.seq, "lr": t.learning_rate, "seed": args.seed,
    "s_per_step": round(sum(st[1:]) / max(1, len(st) - 1), 3) if len(st) > 1 else round(train_s / args.steps, 3),
    "s_per_step_source": "engine step timer" if len(st) > 1 else "train() wall time / steps (incl. startup)",
    "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    "train_loss_first": run.loss_history[0] if run.loss_history else None,
    "train_loss_last": run.final_loss,
    "train_loss_history_len": len(run.loss_history),
}
save_dir = os.path.join(args.out, f"saved_{tag}")
t.save(save_dir, run_id=run.run_id)
del t, run
gc.collect()
torch.cuda.empty_cache()
rec.update(heldout_loss(save_dir))
shutil.rmtree(save_dir, ignore_errors=True)
emit(rec)
