"""E4 stage summary: per-arm tables, paired statistics and the pre-registered verdict.

Reads ``<out>/runs/e4_<size>_*.json`` receipts (one per run, written by
``pod_block_engine.py train --dataset code``) and the per-item records they point to
(``runs/items/<tag>.jsonl.gz``), and writes ``<out>/stage_e4_<size>.json``. Statistics are
those of stage d (``pod_gsm8k_summary.py``); the verdict comes from the pure functions in
``e4_lib`` (``premise_verdict`` for 3b, ``ship_verdict`` for 7b), never from this file.

Usage:  python scripts/pod_e4_summary.py --out /workspace/e4 --size 3b
Exit:   0 verdict PASS, 3 FAIL or INCONCLUSIVE (the pod script stops there), 2 bad input.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_lib as E4  # noqa: E402

# (arm_a, arm_b, role): role "gate" feeds the verdict, "report" is only reported.
PAIRS = {
    "3b": [("default", "qlora", "gate"), ("block_k5", "qlora", "report"), ("block_k5", "default", "report")],
    "7b": [("block_k5", "qlora", "gate"), ("galore", "qlora", "report"), ("galore", "block_k5", "report")],
}
MODEL_FOR_SIZE = {"3b": "Qwen/Qwen2.5-3B", "7b": "Qwen/Qwen2.5-7B"}


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_items(out: str, rec: dict) -> list[dict] | None:
    f = rec.get("items_file")
    path = os.path.join(out, f) if f else None
    if not path or not os.path.exists(path):
        return None
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_runs(out: str, size: str) -> list[dict]:
    runs_dir = os.path.join(out, "runs")
    recs = []
    for name in sorted(os.listdir(runs_dir)):
        if name.startswith(f"e4_{size}_") and name.endswith(".json"):
            recs.append(_read_json(os.path.join(runs_dir, name)))
    return recs


def summarize_stage(out: str, size: str, n_boot: int = 10000, model: str | None = None) -> dict:
    """Build the stage record (including ``verdict``) from the receipts under ``out``."""
    runs = load_runs(out, size)
    ok_runs = [r for r in runs if r.get("status") == "ok"]
    by_arm: dict[str, dict[int, dict]] = {}
    for r in ok_runs:
        by_arm.setdefault(r["arm"], {})[r["seed"]] = r
    items_by_arm: dict[str, dict[int, list[dict]]] = {}
    checks: dict[str, bool] = {}
    for r in runs:
        checks[f"completed:{r['tag']}"] = r.get("status") == "ok"
    for arm, seeds in by_arm.items():
        for s, r in seeds.items():
            its = load_items(out, r)
            if its is None:
                checks[f"items_present:{r['tag']}"] = False
            else:
                items_by_arm.setdefault(arm, {})[s] = its

    table = {}
    for arm, seeds in sorted(by_arm.items()):
        rs = list(seeds.values())
        k = sum(round((r.get("pass_at_1") or 0.0) * r["n"]) for r in rs)
        n = sum(r.get("n", 0) for r in rs)
        outcome_totals: dict[str, int] = {}
        for r in rs:
            for kk, v in (r.get("outcomes") or {}).items():
                outcome_totals[kk] = outcome_totals.get(kk, 0) + v
        table[arm] = {
            "model": rs[0]["model"], "seeds": sorted(seeds),
            "pass_at_1": E4.spread([r.get("pass_at_1") for r in rs]),
            "pass_at_1_per_seed": {s: {"p": r.get("pass_at_1"), "wilson95": r.get("acc_wilson95")}
                                   for s, r in sorted(seeds.items())},
            "pass_at_1_pooled": round(k / n, 4) if n else None, "pass_at_1_pooled_wilson95": E4.wilson(k, n),
            "outcomes_total": dict(sorted(outcome_totals.items())),
            "heldout_answer_loss": E4.spread([r.get("heldout_after") for r in rs]),
            "base_pass_at_1": rs[0].get("pass_at_1_before"), "base_answer_loss": rs[0].get("heldout_before"),
            "s_per_step": E4.spread([r.get("s_per_step") for r in rs]),
            "wall_s": E4.spread([r.get("wall_s") for r in rs]),
            "eval_s": E4.spread([r.get("eval_s") for r in rs]),
            "nvml_peak_gib": E4.spread([r.get("nvml_peak_gib") for r in rs]),
            "nvml_train_peak_gib": E4.spread([(r.get("nvml_phase_peak_gib") or {}).get("train") for r in rs]),
            "torch_max_allocated_gib": E4.spread([r.get("peak_vram_alloc_gib") for r in rs]),
            "torch_max_reserved_gib": E4.spread([r.get("peak_vram_reserved_gib") for r in rs]),
            "optimizer_paged_state_gib": E4.spread([r.get("optimizer_paged_state_gib") for r in rs]),
            "final_train_loss": E4.spread([r.get("final_loss") for r in rs]),
            "epochs_seen": rs[0].get("epochs_seen"),
            "config": {k2: rs[0].get(k2) for k2 in (
                "steps", "batch", "seq", "packing_used", "lr_used", "optimizer_class", "optim_setting",
                "schedule", "k", "order", "engine", "lora_r_used", "optimizer_defaults",
                "s_per_step_measured_on", "train_samples_used")},
            "hit_max_new_tokens": [r.get("hit_max_new_tokens") for r in rs],
        }

    # Identical data order and loss masking across arms, per seed (as in stage d).
    groups: dict[int, set] = {}
    for r in ok_runs:
        mc = r.get("mask_check") or {}
        if "mask_sha256" in mc:
            groups.setdefault(r["seed"], set()).add((mc["input_sha256"], mc["mask_sha256"], mc["loss_tokens"]))
    mask_report = {f"s{sd}": sorted(v) for sd, v in groups.items()}
    for key, v in mask_report.items():
        checks[f"same_first_batch_and_mask:{key}"] = len(v) == 1

    pairs_out: dict[str, dict] = {}
    gate_pair: dict | None = None
    gate_arms = None
    for a, b, role in PAIRS[size]:
        A, B = items_by_arm.get(a), items_by_arm.get(b)
        if not A or not B:
            continue
        try:
            ps = E4.pair_stats(A, B, n_boot)
        except ValueError as exc:
            checks[f"same_qids:{a}_vs_{b}"] = False
            pairs_out[f"{a}|{b}"] = {"error": str(exc)}
            continue
        ps.update({"a": a, "b": b, "role": role, "beats": E4.beats(ps)})
        pairs_out[f"{a}|{b}"] = ps
        if role == "gate":
            gate_pair, gate_arms = ps, (a, b)

    seeds_a = len(items_by_arm.get(gate_arms[0] if gate_arms else PAIRS[size][0][0], {}))
    seeds_b = len(items_by_arm.get(gate_arms[1] if gate_arms else PAIRS[size][0][1], {}))
    if size == "3b":
        verdict = E4.premise_verdict(gate_pair, seeds_a, seeds_b)
    else:
        verdict = E4.ship_verdict(gate_pair, seeds_a, seeds_b)

    base = {}
    base_path = os.path.join(out, "runs", f"base_code_{E4.slug(model or MODEL_FOR_SIZE[size])}.json")
    if os.path.exists(base_path):
        b = _read_json(base_path)
        base = {k: b.get(k) for k in ("model", "pass_at_1", "heldout_loss", "acc_wilson95", "outcomes",
                                      "precheck", "eval_s", "hit_max_new_tokens")}
    budget_path = os.path.join(out, f"budget_{size}.json")
    rec = {"mode": "summarize", "stage": f"e4_{size}", "arms": table, "pairs": pairs_out, "base": base,
           "verdict": verdict, "mask_report": mask_report, "checks": checks,
           "budget": _read_json(budget_path) if os.path.exists(budget_path) else None,
           "n_boot": n_boot,
           "rule": "docs/receipts/2026-10-e4-code/README.md (pre-registered); verdict by e4_lib."
                   + ("premise_verdict" if size == "3b" else "ship_verdict")}
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", choices=["3b", "7b"], required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--model", default=None, help="base model id (default: the stage's Qwen base)")
    args = ap.parse_args(argv)
    if not os.path.isdir(os.path.join(args.out, "runs")):
        print(f"no runs directory under {args.out}", file=sys.stderr)
        return 2
    rec = summarize_stage(args.out, args.size, args.boot, args.model)
    path = os.path.join(args.out, f"stage_e4_{args.size}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("STAGE " + json.dumps({"arms": {k: {"pass_at_1": v["pass_at_1"].get("mean"),
                                              "loss": v["heldout_answer_loss"].get("mean")}
                                          for k, v in rec["arms"].items()}}), flush=True)
    print("VERDICT " + json.dumps(rec["verdict"]), flush=True)
    failed = [k for k, v in rec["checks"].items() if not v]
    if failed:
        print("CHECKS FAILED: " + ",".join(failed), flush=True)
    return 0 if rec["verdict"]["verdict"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
