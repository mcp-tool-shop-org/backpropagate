"""E3: the refit tool (scripts/e3_refit.py) on synthetic sweep receipts.

No GPU and no pod: receipts are generated from KNOWN coefficients through the
library's own ``estimate_vram`` (plus seeded noise), so what is under test is the
tool: does the fit recover a known structure, does the gate compute PASS/FAIL the
way README.md in docs/receipts/2026-10-e3-vram/ pre-registers it, and does the
tool leave the library alone.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import sys

import numpy as np
import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)

import e3_lib as L  # noqa: E402
import e3_refit as R  # noqa: E402

from backpropagate.trainer import (  # noqa: E402
    DEFAULT_VRAM_COEFFICIENTS,
    VRAMCoefficients,
    estimate_vram,
)

CARD = 31.37

# name, text params, hidden, layers, heads, vocab, lora_r, window  (real-ish architectures)
MODELS = [
    ("llama-3.2-1b", 1.24e9, 2048, 16, 32, 128256, 64, 2048),
    ("qwen2.5-3b", 3.09e9, 2048, 36, 16, 151936, 128, 2048),
    ("llama-3.2-3b", 3.21e9, 3072, 28, 24, 128256, 128, 2048),
    ("smollm3-3b", 3.08e9, 2048, 36, 16, 128256, 128, 8192),
    ("phi-4-mini-3.8b", 3.84e9, 3072, 32, 24, 200064, 128, 2048),
    ("qwen3.5-4b", 4.21e9, 2560, 32, 16, 248320, 128, 4096),
    ("mistral-7b", 7.25e9, 4096, 32, 32, 32768, 256, 2048),
    ("qwen2.5-7b", 7.62e9, 3584, 28, 28, 152064, 256, 2048),
    ("llama-3.1-8b", 8.03e9, 4096, 32, 32, 128256, 16, 4096),
    ("qwen2.5-14b", 14.77e9, 5120, 48, 40, 152064, 32, 4096),
    ("mistral-small-24b", 23.57e9, 5120, 40, 32, 131072, 32, 4096),
    ("qwen2.5-32b", 32.76e9, 5120, 64, 40, 152064, 32, 2048),
]

TRUTH = VRAMCoefficients(
    weights_scale=1.0, lora_adapter_scale=1.0, optimizer_state_scale=1.0,
    activations_scale=0.35, kv_cache_scale=0.35, embedding_scale=1.0, logits_scale=2.2,
    fixed_overhead_gb=0.6, unsloth_activations_factor=0.7, unsloth_logits_factor=0.0)


def _arch(m):
    name, params, h, nl, nh, vocab, _r, _w = m
    return {"model_id": name, "hidden_size": h, "num_hidden_layers": nl, "num_attention_heads": nh,
            "vocab_size": vocab, "text_params": int(params), "text_params_b": params / 1e9}


def make_receipts(tmp_path, noise=0.015, seed=1, drop=(), outlier=None, arms=(False, True), batches=L.BATCHES):
    """One receipt per point; peaks = TRUTH through the library estimator x (1 + noise)."""
    rng = random.Random(seed)
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    for m in MODELS:
        name, params, h, nl, nh, vocab, r, w = m
        for b in batches:
            for u in arms:
                tag = L.point_tag(name, b, u)
                if tag in drop:
                    continue
                truth = estimate_vram(
                    model=f"x/{name}", mode="lora", lora_r=r, batch_size=b, max_seq_length=w, hidden_dim=h,
                    num_layers=nl, num_heads=nh, vocab_size=vocab, param_count_billions=params / 1e9,
                    use_unsloth=u, coefficients=TRUTH).total_gb
                peak = truth * (1 + rng.gauss(0, noise))
                if outlier == tag:
                    peak *= 1.5
                status = "ok" if peak < CARD - 0.4 else "oom"
                rec = {"mode": "e3_point", "tag": tag, "preset": name, "model": f"x/{name}", "batch": b,
                       "unsloth_requested": u, "unsloth_active": u, "window": w, "lora_r": r, "steps": 8,
                       "dry_run": False, "status": status, "card_total_gib": CARD, "auto_batch_choice": 6,
                       "manifest": {}, "arch": _arch(m)}
                if status == "ok":
                    rec.update(peak_gib_for_fit=round(peak, 4), nvml_peak_gib=round(peak, 4), plateau_ok=True,
                               predicted_default={"total_gb": estimate_vram(
                                   model=f"x/{name}", mode="lora", lora_r=r, batch_size=b, max_seq_length=w,
                                   use_unsloth=u).total_gb})
                else:
                    rec.update(oom_phase="train", error="CUDA out of memory (synthetic)")
                (runs / f"{tag}.json").write_text(json.dumps(rec))
    return str(tmp_path)


# ---------------------------------------------------------------- the fit
def test_fit_recovers_a_known_structure(tmp_path):
    d = make_receipts(tmp_path)
    res = R.run(d, margin=0.0, allow_dry_run=False)
    assert res["verdict"] in ("PASS", "FAIL")
    es = res["error_summary"]
    assert es["after_no_margin_max_abs_pct"] < 6.0, es
    # the stock estimator is far off on this (QLoRA, big vocab) structure
    assert es["before_default_max_abs_pct"] > 25.0
    assert res["design_matches_library"] is True


def test_the_proposal_is_not_written_into_the_library(tmp_path):
    d = make_receipts(tmp_path)
    before = DEFAULT_VRAM_COEFFICIENTS.as_dict()
    res = R.run(d, margin=0.05, allow_dry_run=False)
    assert "proposal_vram_coefficients" in res
    assert DEFAULT_VRAM_COEFFICIENTS.as_dict() == before
    assert "PROPOSAL ONLY" in res["proposal_note"]
    assert estimate_vram("qwen2.5-7b").total_gb == pytest.approx(estimate_vram("qwen2.5-7b").total_gb)


def test_passes_the_gate_on_data_the_structure_can_explain(tmp_path):
    d = make_receipts(tmp_path)
    res = R.run(d, margin=0.05, allow_dry_run=False)
    assert res["g1_within_15pct_everywhere"]["pass"], res["g1_within_15pct_everywhere"]["failures"][:2]
    assert res["g2_no_oom_at_default_batch"]["pass"], res["g2_no_oom_at_default_batch"]["failures"][:2]
    assert res["verdict"] == "PASS"
    # the margin leans toward over-prediction
    assert res["error_summary"]["after_min_signed_pct"] > -10


def test_margin_scales_the_proposal(tmp_path):
    d = make_receipts(tmp_path)
    a = R.run(d, margin=0.0, allow_dry_run=False)["proposal_vram_coefficients"]
    b = R.run(d, margin=0.10, allow_dry_run=False)["proposal_vram_coefficients"]
    assert b["weights_scale"] == pytest.approx(a["weights_scale"] * 1.10)
    assert b["fixed_overhead_gb"] == pytest.approx(a["fixed_overhead_gb"] * 1.10)
    # ratios are not scaled by the margin
    assert b["unsloth_activations_factor"] == pytest.approx(a["unsloth_activations_factor"])


def test_unsloth_factors_are_recovered(tmp_path):
    d = make_receipts(tmp_path, noise=0.0)
    res = R.run(d, margin=0.0, allow_dry_run=False)
    c = res["proposal_vram_coefficients"]
    assert c["unsloth_logits_factor"] == pytest.approx(0.0, abs=0.05)
    assert 0.5 < c["unsloth_activations_factor"] < 0.9


# ---------------------------------------------------------------- the gate
def test_g1_fails_when_one_point_is_far_off(tmp_path):
    victim = L.point_tag("qwen2.5-7b", 2, False)
    d = make_receipts(tmp_path, outlier=victim)
    res = R.run(d, margin=0.05, allow_dry_run=False)
    assert not res["g1_within_15pct_everywhere"]["pass"]
    assert victim in [r["tag"] for r in res["g1_within_15pct_everywhere"]["failures"]]
    assert res["verdict"] == "FAIL"


def test_g2_fails_when_the_default_batch_was_never_measured(tmp_path):
    d0 = make_receipts(tmp_path)
    ok = R.run(d0, margin=0.05, allow_dry_run=False)
    row = next(r for r in ok["g2_no_oom_at_default_batch"]["rows"] if r["preset"] == "qwen2.5-14b" and not r["unsloth"])
    missing = L.point_tag("qwen2.5-14b", row["proposed_default_batch"], False)
    os.remove(os.path.join(d0, "runs", f"{missing}.json"))
    res = R.run(d0, margin=0.05, allow_dry_run=False)
    rows = [r for r in res["g2_no_oom_at_default_batch"]["rows"]
            if r["preset"] == "qwen2.5-14b" and not r["unsloth"]]
    assert rows[0]["proposed_default_batch"] == row["proposed_default_batch"]
    assert rows[0]["verdict"] == "UNMEASURED"
    assert res["verdict"] == "FAIL"
    assert any("--presets qwen2.5-14b" in c and f"--batches {row['proposed_default_batch']}" in c
               for c in res["top_up"])


def test_g2_fails_when_the_chosen_batch_ooms(tmp_path):
    d = make_receipts(tmp_path)
    ok = R.run(d, margin=0.05, allow_dry_run=False)
    row = next(r for r in ok["g2_no_oom_at_default_batch"]["rows"] if r["preset"] == "qwen2.5-32b" and not r["unsloth"])
    tag = L.point_tag("qwen2.5-32b", row["proposed_default_batch"], False)
    p = os.path.join(d, "runs", f"{tag}.json")
    rec = json.loads(pathlib.Path(p).read_text())
    rec.update(status="oom", error="CUDA out of memory")
    rec.pop("peak_gib_for_fit", None)
    pathlib.Path(p).write_text(json.dumps(rec))
    res = R.run(d, margin=0.05, allow_dry_run=False)
    verdicts = [r for r in res["g2_no_oom_at_default_batch"]["rows"] if r["preset"] == "qwen2.5-32b" and not r["unsloth"]]
    assert verdicts and verdicts[0]["verdict"] == "FAIL_OOM"
    assert res["verdict"] == "FAIL"


def test_choose_batch_picks_the_largest_that_fits_the_headroom():
    big = R.Obs(tag="t", preset="p", model="x/qwen2.5-32b", batch=1, unsloth=False, window=2048, lora_r=32,
                arch=_arch(MODELS[-1]), peak=20.0, card_total=CARD, status="ok", auto_batch=6,
                recorded_default_gb=None, recorded_arch_gb=None)
    b, preds = R.choose_batch(big, TRUTH, CARD)
    assert b in L.BATCHES and preds[b] <= R.HEADROOM * CARD
    assert all(preds[x] > R.HEADROOM * CARD for x in L.BATCHES if x > b)
    b_tiny, _ = R.choose_batch(big, VRAMCoefficients(weights_scale=1e-6, lora_adapter_scale=1e-6,
                                                     optimizer_state_scale=1e-6, activations_scale=1e-6,
                                                     kv_cache_scale=1e-6), CARD)
    assert b_tiny == max(L.BATCHES)


# --------------------------------------------------------------- data guards
def test_too_few_points_is_insufficient_data_and_exit_2(tmp_path, capsys):
    d = make_receipts(tmp_path, arms=(False,), batches=(1,))
    # 12 points, 8 unknowns minus the unsloth columns would still run; cut to 5
    runs = os.path.join(d, "runs")
    for name in sorted(os.listdir(runs))[5:]:
        os.remove(os.path.join(runs, name))
    assert R.main(["--receipts", d]) == 2
    assert "INSUFFICIENT_DATA" in capsys.readouterr().out


def test_dry_run_receipts_are_refused_unless_allowed(tmp_path):
    d = make_receipts(tmp_path)
    runs = os.path.join(d, "runs")
    for name in os.listdir(runs):
        p = os.path.join(runs, name)
        rec = json.loads(pathlib.Path(p).read_text())
        rec["dry_run"] = True
        pathlib.Path(p).write_text(json.dumps(rec))
    res = R.run(d, margin=0.05, allow_dry_run=False)
    assert res["verdict"] == "INSUFFICIENT_DATA"
    assert all("dry-run" in e["reason"] for e in res["excluded"])
    assert R.run(d, margin=0.05, allow_dry_run=True)["verdict"] in ("PASS", "FAIL")


def test_points_without_an_architecture_are_excluded_not_guessed(tmp_path):
    d = make_receipts(tmp_path)
    p = os.path.join(d, "runs", f"{L.point_tag('llama-3.2-1b', 1, False)}.json")
    rec = json.loads(pathlib.Path(p).read_text())
    rec["arch"] = None
    pathlib.Path(p).write_text(json.dumps(rec))
    res = R.run(d, margin=0.05, allow_dry_run=False)
    assert any("no architecture" in e["reason"] for e in res["excluded"])


def test_the_nnls_solution_is_nonnegative_and_exact_when_the_model_is():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 3.0, size=(30, 4))
    truth = np.array([1.0, 0.0, 2.5, 0.3])
    y = X @ truth
    X8 = np.zeros((30, 8))
    X8[:, :4] = X
    th, info = R.nnls_relative(X8, y)
    assert np.all(th >= 0)
    assert np.allclose(th[:4], truth, atol=1e-8)
    assert info["rss_relative"] < 1e-12


def test_leave_one_model_out_is_reported_for_a_dozen_presets(tmp_path):
    d = make_receipts(tmp_path)
    res = R.run(d, margin=0.05, allow_dry_run=False)
    assert len(res["leave_one_model_out_max_abs_err_pct"]) == len(MODELS)


def test_render_and_main_write_the_result(tmp_path, capsys):
    d = make_receipts(tmp_path)
    out = tmp_path / "refit.json"
    rc = R.main(["--receipts", d, "--out", str(out)])
    text = capsys.readouterr().out
    assert rc in (0, 1)
    assert "G1" in text and "G2" in text and "proposal (NOT applied)" in text
    assert json.loads(out.read_text())["verdict"] in ("PASS", "FAIL")
