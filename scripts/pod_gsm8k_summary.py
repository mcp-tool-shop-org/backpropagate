"""Stage d / d2 summary for the GSM8K comparison: paired statistics from per-item outputs.

Reads ``<out>/runs/<prefix>_*.json`` (receipts) and ``<out>/runs/items/<tag>.jsonl``
(per-item outputs: qid, gold, generated, parsed, correct, answer_loss_sum,
answer_tokens). Writes ``<out>/stage_<prefix>.json`` and, for ``--prefix d``,
``<out>/d2_triggers.json``.

Per arm: accuracy (strict '####' parse; parse failure = wrong) with Wilson 95%
CIs, held-out answer loss (token-weighted over the 250 items), s/step, NVML
peak and torch max allocated / reserved, lr and optimizer actually used.

Per pair of arms at the same size (the list below), on the shared 250 qids:
  * per seed: McNemar exact test (two-sided binomial on the discordant items)
    and paired bootstrap 95% CIs (10,000 resamples of items, seed 0) for the
    accuracy difference and the held-out-loss difference;
  * pooled over seeds: per item, each arm's mean over its seeds, then the same
    paired bootstrap. A pair is paired seed-to-seed where both arms have the
    seed; otherwise each seed of the first arm is paired with the second
    arm's first seed.

d2 rule (pre-authorized; decided here, not by hand): for the TRIGGER pairs,
if the pooled held-out-loss difference's 95% CI spans 0 at 1000 steps, the
pair is rerun at 2000 steps with the same seeds.

Usage: python scripts/pod_gsm8k_summary.py --out /workspace/block_engine --prefix d
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--prefix", default="d", choices=["d", "d2"])
ap.add_argument("--boot", type=int, default=10000)
args = ap.parse_args()
RUNS = os.path.join(args.out, "runs")

# (size, arm_a, arm_b, can_trigger_d2)
PAIRS = [
    ("7b", "block_k5", "qlora", True),
    ("7b", "galore", "qlora", False),
    ("7b", "galore", "block_k5", False),
    ("3b", "block_k5", "default", True),
    ("3b", "block_k5", "qlora", True),
    ("3b", "block_k5", "block_k50", True),
    ("3b", "block_k50", "default", False),
    ("3b", "default", "qlora", False),
    ("3b", "block_k5_lr5e-5", "block_k5", False),
    ("3b", "default_adafactor", "default", False),
]


def load(prefix: str) -> list[dict]:
    out = []
    for name in sorted(os.listdir(RUNS)):
        if name.startswith(prefix + "_") and name.endswith(".json"):
            with open(os.path.join(RUNS, name)) as fh:
                r = json.load(fh)
            r["size"] = r["tag"].split("_")[1]
            out.append(r)
    return out


def items(rec: dict) -> list[dict] | None:
    f = rec.get("items_file")
    if not f:
        return None
    path = os.path.join(args.out, f)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def spread(xs: list) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 4),
            "sd": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 4), "max": round(max(xs), 4)}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    n = b01 + b10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(b01, b10)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n)
    return {"a_only": b01, "b_only": b10, "p": round(p, 5)}


def boot(acc_a, acc_b, ls_a, ls_b, n_a, n_b) -> dict:
    rng = np.random.default_rng(0)
    m = len(acc_a)
    idx = rng.integers(0, m, size=(args.boot, m))
    dacc = acc_a[idx].mean(1) - acc_b[idx].mean(1)
    dloss = ls_a[idx].sum(1) / n_a[idx].sum(1) - ls_b[idx].sum(1) / n_b[idx].sum(1)
    point_acc = float(acc_a.mean() - acc_b.mean())
    point_loss = float(ls_a.sum() / n_a.sum() - ls_b.sum() / n_b.sum())
    return {
        "acc_diff": round(point_acc, 4),
        "acc_diff_ci95": [round(float(np.quantile(dacc, 0.025)), 4), round(float(np.quantile(dacc, 0.975)), 4)],
        "loss_diff": round(point_loss, 4),
        "loss_diff_ci95": [round(float(np.quantile(dloss, 0.025)), 4), round(float(np.quantile(dloss, 0.975)), 4)],
    }


def arrays(its: list[dict]):
    its = sorted(its, key=lambda x: x["qid"])
    return (np.array([bool(x["correct"]) for x in its]),
            np.array([x["answer_loss_sum"] for x in its], dtype=float),
            np.array([x["answer_tokens"] for x in its], dtype=float),
            [x["qid"] for x in its])


runs = load(args.prefix)
by_arm: dict[tuple[str, str], dict[int, dict]] = {}
for r in runs:
    if r.get("status") == "ok":
        by_arm.setdefault((r["size"], r["arm"]), {})[r["seed"]] = r

table = {}
for (size, arm), seeds in sorted(by_arm.items()):
    rs = list(seeds.values())
    k = sum(round(r["acc_strict"] * r["n"]) for r in rs if r.get("acc_strict") is not None)
    n = sum(r.get("n", 0) for r in rs)
    table[f"{size}|{arm}"] = {
        "model": rs[0]["model"], "seeds": sorted(seeds),
        "acc_strict": spread([r.get("acc_strict") for r in rs]),
        "acc_strict_per_seed": {s: {"acc": r.get("acc_strict"), "wilson95": r.get("acc_wilson95")}
                                for s, r in sorted(seeds.items())},
        "acc_pooled": round(k / n, 4) if n else None, "acc_pooled_wilson95": wilson(k, n),
        "acc_lenient": spread([r.get("acc_lenient") for r in rs]),
        "heldout_answer_loss": spread([r.get("heldout_after") for r in rs]),
        "base_acc": rs[0].get("acc_before"), "base_answer_loss": rs[0].get("heldout_before"),
        "s_per_step": spread([r.get("s_per_step") for r in rs]),
        "nvml_peak_gib": spread([r.get("nvml_peak_gib") for r in rs]),
        "nvml_train_peak_gib": spread([(r.get("nvml_phase_peak_gib") or {}).get("train") for r in rs]),
        "torch_max_allocated_gib": spread([r.get("torch_max_allocated_train_gib") for r in rs]),
        "torch_max_reserved_gib": spread([r.get("torch_max_reserved_train_gib") for r in rs]),
        "optimizer_paged_state_gib": spread([r.get("optimizer_paged_state_gib") for r in rs]),
        "config": {k2: rs[0].get(k2) for k2 in ("steps", "batch", "seq", "packing_used", "lr_used",
                                               "optimizer_class", "optim_setting", "schedule", "k", "order",
                                               "engine", "lora_r_used", "optimizer_defaults")},
        "hit_max_new_tokens": [r.get("hit_max_new_tokens") for r in rs],
        "has_hash_answer": [r.get("has_hash_answer") for r in rs],
    }

checks: dict[str, bool] = {}
for r in runs:
    checks[f'completed:{r["tag"]}'] = r.get("status") == "ok"
# Identical loss masking + data order across arms: same first batch, same mask.
groups: dict[tuple[str, int], set] = {}
for r in runs:
    mc = r.get("mask_check") or {}
    if "mask_sha256" in mc:
        groups.setdefault((r["size"], r["seed"]), set()).add(
            (mc["input_sha256"], mc["mask_sha256"], mc["loss_tokens"]))
mask_report = {f"{s}_s{sd}": sorted(v) for (s, sd), v in groups.items()}
for key, v in mask_report.items():
    checks[f"same_first_batch_and_mask:{key}"] = len(v) == 1

pairs_out, triggers = [], []
for size, a, b, can_trigger in PAIRS:
    A, B = by_arm.get((size, a)), by_arm.get((size, b))
    if not A or not B:
        continue
    per_seed = []
    pooled_a, pooled_b = [], []
    for s, ra in sorted(A.items()):
        rb = B.get(s) or B[sorted(B)[0]]
        ia, ib = items(ra), items(rb)
        if not ia or not ib:
            continue
        acc_a, ls_a, n_a, q_a = arrays(ia)
        acc_b, ls_b, n_b, q_b = arrays(ib)
        if q_a != q_b:
            checks[f"same_qids:{size}:{a}_vs_{b}"] = False
            continue
        per_seed.append({"seed_a": s, "seed_b": rb["seed"], "mcnemar": mcnemar_exact(acc_a, acc_b),
                         **boot(acc_a.astype(float), acc_b.astype(float), ls_a, ls_b, n_a, n_b)})
    for _s, ra in sorted(A.items()):
        ia = items(ra)
        if ia:
            pooled_a.append(arrays(ia))
    for _s, rb in sorted(B.items()):
        ib = items(rb)
        if ib:
            pooled_b.append(arrays(ib))
    pooled = None
    if pooled_a and pooled_b:
        pa = [np.mean([x[i] for x in pooled_a], axis=0) for i in range(3)]
        pb = [np.mean([x[i] for x in pooled_b], axis=0) for i in range(3)]
        pooled = boot(pa[0].astype(float), pb[0].astype(float), pa[1], pb[1], pa[2], pb[2])
    entry = {"size": size, "a": a, "b": b, "seeds_a": sorted(A), "seeds_b": sorted(B),
             "per_seed": per_seed, "pooled_over_seeds": pooled, "d2_eligible": can_trigger}
    if pooled and can_trigger:
        lo, hi = pooled["loss_diff_ci95"]
        entry["loss_ci_spans_zero"] = lo <= 0.0 <= hi
        if entry["loss_ci_spans_zero"]:
            triggers.append({"size": size, "arms": [a, b], "seeds": {a: sorted(A), b: sorted(B)}})
    pairs_out.append(entry)

rec = {"mode": "summarize", "stage": args.prefix, "arms": table, "pairs": pairs_out,
       "mask_report": mask_report, "checks": checks,
       "d2_rule": "for d2-eligible pairs, rerun at 2000 steps (same seeds) if the pooled "
                  "held-out-loss difference's 95% bootstrap CI spans 0",
       "d2_triggers": triggers if args.prefix == "d" else None}
path = os.path.join(args.out, f"stage_{args.prefix}.json")
with open(path, "w") as fh:
    json.dump(rec, fh, indent=1)
if args.prefix == "d":
    with open(os.path.join(args.out, "d2_triggers.json"), "w") as fh:
        json.dump(triggers, fh, indent=1)
print("STAGE " + json.dumps({"arms": {k: {"acc": v["acc_strict"].get("mean"), "loss": v["heldout_answer_loss"].get("mean")}
                                      for k, v in table.items()}, "triggers": triggers}), flush=True)
failed = [k for k, v in checks.items() if not v]
print(f"RESULT stage {args.prefix}: " + ("PASS" if checks and not failed else "FAIL " + ",".join(failed)), flush=True)
