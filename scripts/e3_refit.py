#!/usr/bin/env python3
"""E3 refit: propose new ``estimate_vram`` addends from sweep receipts, and PASS/FAIL the gate.

This tool PROPOSES. It never writes a constant into the library: the shipped
``VRAMCoefficients`` stay the identity table until the lead applies a refit in a
later PR with these receipts as provenance. (The last table went wrong by being
set from arithmetic, not from pod numbers.)

Input: the sweep's receipt directory (``runs/e3_*.json`` from
``scripts/pod_e3_sweep.py``; a flat directory of receipts also works).

Method (stated so it can be argued with)
----------------------------------------
Per measured point the estimator is, with the library's own ``vram_addends``
(weights W, LoRA adapter L, gradient+optimizer state O, activations A, KV cache
K, embedding E, logits G; all raw, unscaled, in GiB, from the model's real
dimensions and text-only parameter count read from its config)::

    total = (1 + f) * (sW*W + sL*L + sO*O + sA*A + sK*K + sE*E + sG*G) + c

with f = 0.15 (the library's ``overhead_fraction``, left alone). Because L and O
are exact multiples of each other (O = 2L at bf16) and A and K are too (A = 4K)
in QLoRA, their scales cannot be separated by any data: each pair is fitted as
one group (sL = sO, sA = sK). Unsloth gets its own activation and logits scale
(the ratio to the plain scale is the library's ``unsloth_*_factor``).

The fit is non-negative least squares on the RELATIVE error
(``sum(((pred - obs) / obs)^2)``; the gate is relative), solved exactly by
enumerating supports (at most 8 unknowns). Then a stated safety margin (default
+5%) multiplies every scale and the constant, so the proposal leans toward
over-predicting: an estimate that is too high costs a smaller batch, one that is
too low costs an OOM. Reported per measured point: the error before (as shipped,
two ways) and after (without and with the margin), plus a leave-one-model-out
error as an honest estimate of how it generalises (informational, not gated).

The gate (pre-registered in docs/receipts/2026-10-e3-vram/README.md before any run)
-----------------------------------------------------------------------------------
  G1  the proposed estimator is within +-15% of the measured peak
      (``peak_gib_for_fit``) at EVERY measured point whose status is ok;
  G2  no preset OOMs at its default batch, where the default batch is the one the
      PROPOSED estimator picks (largest of 1, 2, 4, 6 whose predicted total is at
      most 95% of the card; 1 when none is) and a point only counts when it was
      measured. The batch today's ``_detect_batch_size`` resolves is reported
      next to it, not gated.
  PASS = G1 and G2. Exit 0 on PASS, 1 on FAIL, 2 when there is not enough data.

Standards compliance: see ``e3_lib.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

import e3_lib as L  # noqa: E402
import numpy as np  # noqa: E402

TOLERANCE = 0.15
MARGIN = 0.05
HEADROOM = 0.95
OVERHEAD_FRACTION = 0.15  # estimate_vram's default; asserted against the library below
COLS = ["s_weights", "s_adapter_group", "s_act_group", "t_act_unsloth", "s_embedding",
        "s_logits", "t_logits_unsloth", "c_fixed_gb"]


@dataclasses.dataclass
class Obs:
    tag: str
    preset: str
    model: str
    batch: int
    unsloth: bool
    window: int
    lora_r: int
    arch: dict[str, Any]
    peak: float
    card_total: float
    status: str
    auto_batch: int | None
    recorded_default_gb: float | None
    recorded_arch_gb: float | None
    load_peak: float | None = None
    train_peak: float | None = None

    @property
    def params(self) -> float:
        return float(self.arch["text_params"])

    @property
    def dims(self) -> dict[str, Any]:
        a = self.arch
        return {"hidden_dim": int(a["hidden_size"]), "num_layers": int(a["num_hidden_layers"]),
                "num_heads": int(a["num_attention_heads"]), "vocab_size": int(a["vocab_size"])}


# ------------------------------------------------------------------ loading
def load_receipts(path: str) -> list[dict[str, Any]]:
    d = os.path.join(path, "runs") if os.path.isdir(os.path.join(path, "runs")) else path
    recs = []
    for name in sorted(os.listdir(d)):
        if name.startswith("e3_") and name.endswith(".json"):
            r = L.read_json(os.path.join(d, name))
            if r.get("mode") == "e3_point":
                recs.append(r)
    return recs


def to_obs(recs: list[dict[str, Any]], allow_dry_run: bool) -> tuple[list[Obs], list[dict[str, Any]], list[dict[str, Any]]]:
    """(usable ok points, every measured-or-skipped record as a status row, exclusions)."""
    usable: list[Obs] = []
    statuses: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for r in recs:
        active = r.get("unsloth_active")
        active = bool(r.get("unsloth_requested")) if active is None else bool(active)
        statuses.append({"tag": r["tag"], "preset": r.get("preset"), "batch": r.get("batch"),
                         "unsloth": active, "status": r.get("status"), "auto_batch": r.get("auto_batch_choice"),
                         "card_total": r.get("card_total_gib")})
        if r.get("status") != "ok":
            continue
        if r.get("dry_run") and not allow_dry_run:
            excluded.append({"tag": r["tag"], "reason": "dry-run receipt (CPU RSS stand-in, not a VRAM figure); "
                                                        "pass --allow-dry-run to exercise the plumbing"})
            continue
        peak = r.get("peak_gib_for_fit")
        if not L.finite(peak) or peak <= 0:
            excluded.append({"tag": r["tag"], "reason": "no usable peak_gib_for_fit"})
            continue
        arch = r.get("arch")
        if not arch or not arch.get("text_params") or not arch.get("hidden_size"):
            excluded.append({"tag": r["tag"], "reason": "no architecture in the receipt (config fetch failed)"})
            continue
        usable.append(Obs(
            tag=r["tag"], preset=r["preset"], model=r["model"], batch=int(r["batch"]), unsloth=active,
            window=int(r["window"]), lora_r=int(r["lora_r"]), arch=arch, peak=float(peak),
            card_total=float(r.get("card_total_gib") or 0.0), status="ok", auto_batch=r.get("auto_batch_choice"),
            recorded_default_gb=(r.get("predicted_default") or {}).get("total_gb"),
            recorded_arch_gb=(r.get("predicted_arch") or {}).get("total_gb"),
            load_peak=(r.get("nvml_phase_peak_gib") or {}).get("load"),
            train_peak=(r.get("nvml_phase_peak_gib") or {}).get("train")))
    return usable, statuses, excluded


# ------------------------------------------------------------ model + fit
def _addends(o: Obs) -> dict[str, float]:
    from backpropagate.trainer import vram_addends

    return vram_addends(params=o.params, mode="lora", lora_r=o.lora_r, batch_size=o.batch,
                        max_seq_length=o.window, **o.dims)


def design(obs: list[Obs]) -> np.ndarray:
    F = 1.0 + OVERHEAD_FRACTION
    X = np.zeros((len(obs), len(COLS)))
    for i, o in enumerate(obs):
        R = _addends(o)
        X[i, 0] = F * R["weights"]
        X[i, 1] = F * (R["lora_adapter"] + R["optimizer_state"])
        if o.unsloth:
            X[i, 2] = F * R["kv_cache"]
            X[i, 3] = F * R["activations"]
            X[i, 6] = F * R["logits"]
        else:
            X[i, 2] = F * (R["activations"] + R["kv_cache"])
            X[i, 5] = F * R["logits"]
        X[i, 4] = F * R["embedding"]
        X[i, 7] = 1.0
    return X


def nnls_relative(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """min sum(((X theta - y) / y)^2) s.t. theta >= 0, exactly, by enumerating supports."""
    Xw = X / y[:, None]
    yw = np.ones(len(y))
    live = [j for j in range(X.shape[1]) if np.linalg.norm(Xw[:, j]) > 0]
    best: tuple[float, np.ndarray] | None = None
    for k in range(1, len(live) + 1):
        for sub in itertools.combinations(live, k):
            sol, *_ = np.linalg.lstsq(Xw[:, sub], yw, rcond=None)
            if np.any(sol < -1e-12):
                continue
            th = np.zeros(X.shape[1])
            th[list(sub)] = np.clip(sol, 0.0, None)
            rss = float(np.sum((Xw @ th - yw) ** 2))
            if best is None or rss < best[0] - 1e-15:
                best = (rss, th)
    assert best is not None
    th = best[1]
    sv = np.linalg.svd(Xw[:, live], compute_uv=False)
    info = {"rss_relative": best[0], "rank": int(np.linalg.matrix_rank(Xw[:, live])), "columns_live": len(live),
            "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf,
            "zero_columns": [COLS[j] for j in range(X.shape[1]) if j not in live]}
    return th, info


def _factor(plain: float, unsloth: float, has_unsloth: bool, what: str, notes: list[str]) -> float:
    """Unsloth factor = (Unsloth scale) / (plain scale); 1.0 when there is no Unsloth data to say otherwise."""
    if not has_unsloth:
        return 1.0
    if plain > 1e-12:
        return unsloth / plain  # 0.0 when the fit says Unsloth removes the term
    if unsloth > 0:
        notes.append(f"unsloth {what} scale fitted but the plain one is 0: factor left at 1.0")
    return 1.0


def coefficients_from(theta: np.ndarray, margin: float, has_unsloth: bool = True) -> tuple[Any, list[str]]:
    from backpropagate.trainer import VRAMCoefficients

    notes: list[str] = []
    s_w, s_a, s_c, t_c, s_e, s_g, t_g, c0 = (float(x) for x in theta)
    u_a = _factor(s_c, t_c, has_unsloth, "activation", notes)
    u_g = _factor(s_g, t_g, has_unsloth, "logits", notes)
    k = 1.0 + margin
    return VRAMCoefficients(
        weights_scale=s_w * k, lora_adapter_scale=s_a * k, optimizer_state_scale=s_a * k,
        activations_scale=s_c * k, kv_cache_scale=s_c * k, embedding_scale=s_e * k, logits_scale=s_g * k,
        fixed_overhead_gb=c0 * k, unsloth_activations_factor=u_a, unsloth_logits_factor=u_g), notes


def predict(o: Obs, coeff: Any | None, *, arch_dims: bool = True) -> float:
    """The LIBRARY's estimate_vram for this point (so the proposal is judged by the real function)."""
    from backpropagate.trainer import estimate_vram

    kw: dict[str, Any] = {"model": o.model, "mode": "lora", "lora_r": o.lora_r, "batch_size": o.batch,
                          "max_seq_length": o.window, "use_unsloth": o.unsloth}
    if arch_dims:
        kw.update(o.dims, param_count_billions=o.params / 1e9)
    if coeff is not None:
        kw["coefficients"] = coeff
    return float(estimate_vram(**kw).total_gb)


def rel(pred: float, obs: float) -> float:
    return (pred - obs) / obs


def fit(obs: list[Obs], margin: float) -> dict[str, Any]:
    X = design(obs)
    y = np.array([o.peak for o in obs])
    theta, info = nnls_relative(X, y)
    has_unsloth = any(o.unsloth for o in obs)
    coeff, notes = coefficients_from(theta, margin, has_unsloth)
    return {"theta": theta, "coeff": coeff, "notes": notes, "info": info, "X": X, "y": y,
            "has_unsloth": has_unsloth}


# --------------------------------------------------------------------- gate
def choose_batch(obs_template: Obs, coeff: Any, card_total: float) -> tuple[int, dict[int, float]]:
    """The batch the PROPOSED estimator would pick: largest of BATCHES with total <= HEADROOM * card."""
    preds: dict[int, float] = {}
    for b in L.BATCHES:
        o = dataclasses.replace(obs_template, batch=b)
        preds[b] = predict(o, coeff)
    ok = [b for b in L.BATCHES if preds[b] <= HEADROOM * card_total]
    return (max(ok) if ok else 1), preds


def evaluate(obs: list[Obs], statuses: list[dict[str, Any]], coeff: Any) -> dict[str, Any]:
    rows = []
    for o in obs:
        pd_ = predict(o, None, arch_dims=False)
        pa = predict(o, None, arch_dims=True)
        after = predict(o, coeff)
        rows.append({
            "tag": o.tag, "preset": o.preset, "batch": o.batch, "unsloth": o.unsloth, "measured_gib": round(o.peak, 3),
            "before_default": round(pd_, 3), "before_default_err_pct": round(100 * rel(pd_, o.peak), 1),
            "before_arch": round(pa, 3), "before_arch_err_pct": round(100 * rel(pa, o.peak), 1),
            "after": round(after, 3), "after_err_pct": round(100 * rel(after, o.peak), 1),
            "recorded_default_matches_library": (None if o.recorded_default_gb is None
                                                 else abs(o.recorded_default_gb - pd_) < 1e-9),
            # the whole-process peak is what is fitted; flag points where the model LOAD, not training, set it
            "load_dominated": (None if o.load_peak is None or o.train_peak is None else o.load_peak > o.train_peak)})
    g1_bad = [r for r in rows if abs(r["after_err_pct"]) > 100 * TOLERANCE + 1e-9]

    # G2
    by_key: dict[tuple[str, bool, int], dict[str, Any]] = {}
    arms: set[tuple[str, bool]] = set()
    for s in statuses:
        if s["preset"] is None or s["status"] in ("dropped_by_budget_guard", "dropped_hard_stop"):
            continue
        by_key[(s["preset"], s["unsloth"], s["batch"])] = s
        arms.add((s["preset"], s["unsloth"]))
    for o in obs:
        arms.add((o.preset, o.unsloth))
    tmpl: dict[tuple[str, bool], Obs] = {(o.preset, o.unsloth): o for o in obs}
    card = max([o.card_total for o in obs] or [0.0]) or 31.37
    g2_rows = []
    for (preset, uns) in sorted(arms):
        t = tmpl.get((preset, uns))
        if t is None:
            g2_rows.append({"preset": preset, "unsloth": uns, "verdict": "NO_MODELLED_POINT",
                            "detail": "no ok point with an architecture to build a prediction from"})
            continue
        b, preds = choose_batch(t, coeff, card)
        s = by_key.get((preset, uns, b))
        status = None if s is None else s["status"]
        if status == "ok":
            verdict = "PASS"
        elif status in ("oom", "skipped_monotone_oom"):
            verdict = "FAIL_OOM"
        else:
            verdict = "UNMEASURED"
        auto = t.auto_batch
        s_auto = by_key.get((preset, uns, auto)) if auto is not None else None
        g2_rows.append({
            "preset": preset, "unsloth": uns, "proposed_default_batch": b,
            "predicted_gb_by_batch": {str(k): round(v, 2) for k, v in preds.items()},
            "status_at_proposed_batch": status, "verdict": verdict,
            "todays_auto_batch": auto,
            "status_at_todays_auto_batch": (None if s_auto is None else s_auto["status"]) if auto in L.BATCHES
            else "not in the sweep's batch set"})
    g2_bad = [r for r in g2_rows if r["verdict"] != "PASS"]
    return {"rows": rows, "g1_failures": g1_bad, "g2_rows": g2_rows, "g2_failures": g2_bad, "card_total_gib": card}


def lomo(obs: list[Obs], margin: float) -> dict[str, float]:
    out: dict[str, float] = {}
    presets = sorted({o.preset for o in obs})
    if len(presets) < 6:
        return out
    for p in presets:
        tr = [o for o in obs if o.preset != p]
        te = [o for o in obs if o.preset == p]
        if len(tr) < len(COLS) + 2:
            continue
        coeff = fit(tr, margin)["coeff"]
        out[p] = round(max(100 * abs(rel(predict(o, coeff), o.peak)) for o in te), 1)
    return out


# --------------------------------------------------------------------- main
def run(receipts: str, margin: float, allow_dry_run: bool) -> dict[str, Any]:
    import inspect

    from backpropagate.trainer import estimate_vram

    default_f = inspect.signature(estimate_vram).parameters["overhead_fraction"].default
    assert default_f == OVERHEAD_FRACTION, "estimate_vram's overhead_fraction default changed; update e3_refit"

    recs = load_receipts(receipts)
    obs, statuses, excluded = to_obs(recs, allow_dry_run)
    n_params = len(COLS)
    res: dict[str, Any] = {
        "tool": "scripts/e3_refit.py", "receipts": receipts, "margin": margin, "tolerance": TOLERANCE,
        "headroom": HEADROOM, "points_ok_usable": len(obs), "points_total": len(recs),
        "excluded": excluded, "dry_run_allowed": allow_dry_run,
        "status_counts": {s: sum(1 for r in recs if r.get("status") == s)
                          for s in sorted({str(r.get("status")) for r in recs})}}
    if len(obs) < n_params + 3:
        res.update(verdict="INSUFFICIENT_DATA",
                   reason=f"{len(obs)} usable ok points; need at least {n_params + 3} for {n_params} unknowns")
        return res
    f = fit(obs, margin)
    ev = evaluate(obs, statuses, f["coeff"])
    f0 = coefficients_from(f["theta"], 0.0, f["has_unsloth"])[0]
    after_nomargin = [round(100 * rel(predict(o, f0), o.peak), 1) for o in obs]
    # The design matrix and the library must agree on the proposal: this guards
    # the grouping algebra (adapter = LoRA + optimizer, activations = act + KV,
    # the Unsloth ratios). The library's answer is the one the gate judges.
    design_pred = f["X"] @ (f["theta"] * (1.0 + margin))
    lib_pred = np.array([predict(o, f["coeff"]) for o in obs])
    consistent = bool(np.max(np.abs(design_pred - lib_pred)) < 1e-6 * max(1.0, float(np.max(lib_pred))))
    for r, e0 in zip(ev["rows"], after_nomargin):
        r["after_no_margin_err_pct"] = e0
    g1 = not ev["g1_failures"]
    g2 = not ev["g2_failures"]
    res.update({
        "theta": {c: float(v) for c, v in zip(COLS, f["theta"])},
        "fit_info": f["info"], "fit_notes": f["notes"],
        "proposal_vram_coefficients": f["coeff"].as_dict(),
        "proposal_note": "PROPOSAL ONLY: not written into the library. Apply in a later PR with these receipts as provenance.",
        "design_matches_library": bool(consistent),
        "errors": ev["rows"],
        "error_summary": {
            "before_default_max_abs_pct": max(abs(r["before_default_err_pct"]) for r in ev["rows"]),
            "before_arch_max_abs_pct": max(abs(r["before_arch_err_pct"]) for r in ev["rows"]),
            "after_no_margin_max_abs_pct": max(abs(e) for e in after_nomargin),
            "after_max_abs_pct": max(abs(r["after_err_pct"]) for r in ev["rows"]),
            "after_min_signed_pct": min(r["after_err_pct"] for r in ev["rows"]),
            "after_max_signed_pct": max(r["after_err_pct"] for r in ev["rows"])},
        "leave_one_model_out_max_abs_err_pct": lomo(obs, margin),
        "load_dominated_points": [r["tag"] for r in ev["rows"] if r.get("load_dominated")],
        "g1_within_15pct_everywhere": {"pass": g1, "failures": ev["g1_failures"]},
        "g2_no_oom_at_default_batch": {"pass": g2, "card_total_gib": ev["card_total_gib"], "rows": ev["g2_rows"],
                                       "failures": ev["g2_failures"]},
        "verdict": "PASS" if (g1 and g2) else "FAIL",
        "top_up": top_up(ev["g2_failures"])})
    return res


def top_up(g2_failures: list[dict[str, Any]]) -> list[str]:
    """Pod commands for the points G2 needed but did not have (or that OOMed: nothing to top up)."""
    cmds = []
    for r in g2_failures:
        if r["verdict"] == "UNMEASURED":
            cmds.append(f"python scripts/pod_e3_sweep.py run --out $WORK --presets {r['preset']} "
                        f"--batches {r['proposed_default_batch']} --unsloth {'on' if r['unsloth'] else 'off'} --no-guard")
    return cmds


def render(res: dict[str, Any]) -> str:
    lines = [f"E3 refit: {res['verdict']}  ({res['points_ok_usable']} usable ok points of {res['points_total']} receipts; "
             f"statuses {res['status_counts']})"]
    if res["verdict"] == "INSUFFICIENT_DATA":
        lines.append(res["reason"])
        for e in res["excluded"][:10]:
            lines.append(f"  excluded {e['tag']}: {e['reason']}")
        return "\n".join(lines)
    es = res["error_summary"]
    lines += [
        f"max |error| before: {es['before_default_max_abs_pct']}% as shipped (default dims), "
        f"{es['before_arch_max_abs_pct']}% with real dims",
        f"max |error| after:  {es['after_no_margin_max_abs_pct']}% fit, {es['after_max_abs_pct']}% with the "
        f"+{int(100 * res['margin'])}% margin (signed range {es['after_min_signed_pct']}..{es['after_max_signed_pct']}%)",
        f"G1 (+-{int(100 * TOLERANCE)}% at every measured point): {'PASS' if res['g1_within_15pct_everywhere']['pass'] else 'FAIL'}",
        f"G2 (no OOM at the proposed default batch): {'PASS' if res['g2_no_oom_at_default_batch']['pass'] else 'FAIL'}",
        "", "proposal (NOT applied): " + json.dumps(res["proposal_vram_coefficients"]), "",
        f"{'point':44s} {'meas':>6s} {'default':>8s} {'real dims':>9s} {'fit':>6s} {'+margin':>8s}"]
    for r in res["errors"]:
        lines.append(f"{r['tag']:44s} {r['measured_gib']:6.2f} {r['before_default_err_pct']:+7.1f}% "
                     f"{r['before_arch_err_pct']:+8.1f}% {r['after_no_margin_err_pct']:+5.1f}% {r['after_err_pct']:+7.1f}%")
    lines.append("")
    lines.append(f"{'preset':20s} {'arm':8s} {'proposed b':>10s} {'status':>10s} {'today auto b':>12s} {'status':>12s} verdict")
    for r in res["g2_no_oom_at_default_batch"]["rows"]:
        lines.append(f"{r['preset']:20s} {'unsloth' if r['unsloth'] else 'plain':8s} "
                     f"{str(r.get('proposed_default_batch')):>10s} {str(r.get('status_at_proposed_batch')):>10s} "
                     f"{str(r.get('todays_auto_batch')):>12s} {str(r.get('status_at_todays_auto_batch')):>12s} {r['verdict']}")
    if res["top_up"]:
        lines += ["", "top-up points G2 still needs:"] + ["  " + c for c in res["top_up"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--receipts", required=True, help="sweep output directory (with runs/e3_*.json)")
    ap.add_argument("--out", default=None, help="write the full result JSON here")
    ap.add_argument("--margin", type=float, default=MARGIN, help="safety margin on the fit (default 0.05)")
    ap.add_argument("--allow-dry-run", action="store_true",
                    help="accept CPU dry-run receipts (RSS stand-in peaks): plumbing check only, never evidence")
    a = ap.parse_args(argv)
    res = run(a.receipts, a.margin, a.allow_dry_run)
    print(render(res))
    if a.out:
        L.write_json(a.out, res)
    return {"PASS": 0, "FAIL": 1}.get(res["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
