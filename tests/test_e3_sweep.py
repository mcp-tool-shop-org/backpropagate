"""E3: the sweep driver's planner, budget guard, scrubber and (slow) CPU dry run.

No GPU, no network, no model weights. The dry run builds a tiny random-initialised
model and tokenizer locally and trains it on CPU through the real ``Trainer``.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)

import e3_lib as L  # noqa: E402
import pod_e3_sweep as S  # noqa: E402

from backpropagate.config import MODEL_PRESETS  # noqa: E402


@pytest.fixture(scope="module")
def presets():
    return L.load_presets()


@pytest.fixture(scope="module")
def grid(presets):
    return L.build_points(presets)


# --------------------------------------------------------------------- grid
def test_every_preset_is_swept_small_to_large(presets):
    assert {p.name for p in presets} == set(MODEL_PRESETS)
    sizes = [p.params_b_name for p in presets]
    assert sizes == sorted(sizes)
    assert presets[0].params_b_name == 1 and presets[-1].name == "qwen2.5-32b"


def test_the_grid_is_presets_x_batches_x_unsloth(presets, grid):
    assert len(grid) == len(presets) * len(L.BATCHES) * 2 == 96
    assert len({p.tag for p in grid}) == len(grid)
    assert [p.order for p in grid] == list(range(len(grid)))
    # preset blocks are contiguous (weights are deleted after each preset) and batches ascend within one
    seen: list[str] = []
    for p in grid:
        if not seen or seen[-1] != p.preset:
            assert p.preset not in seen
            seen.append(p.preset)
    for name in seen:
        b = [p.batch for p in grid if p.preset == name and not p.unsloth]
        assert b == sorted(b) == list(L.BATCHES)


def test_each_point_runs_at_the_presets_own_window(presets):
    by = {p.name: p for p in presets}
    assert by["qwen2.5-32b"].window == 2048 and by["qwen2.5-14b"].window == 4096
    assert by["smollm3-3b"].window == 8192


# --------------------------------------------------------------------- tiers
def test_tier_zero_is_the_large_presets_at_batch_1_and_6(grid):
    t0 = [p for p in grid if p.tier == 0]
    assert len(t0) == 12
    assert {p.preset for p in t0} == {"qwen2.5-14b", "mistral-small-24b", "qwen2.5-32b"}
    assert {p.batch for p in t0} == {1, 6}
    assert {p.unsloth for p in t0} == {False, True}


def test_every_point_has_a_tier_and_tiers_partition_the_grid(grid):
    counts = {t: sum(1 for p in grid if p.tier == t) for t in range(6)}
    assert counts == {0: 12, 1: 12, 2: 24, 3: 12, 4: 12, 5: 24}


def test_drop_order_protects_tier_zero_last_and_drops_unsloth_first(grid):
    order = L.drop_order(grid)
    assert len(order) == len(grid)
    assert [p.tier for p in order] == sorted((p.tier for p in order), reverse=True)
    assert order[0].tier == 5 and order[0].unsloth
    assert all(p.tier == 0 for p in order[-12:])
    # within a tier, Unsloth-on goes before Unsloth-off
    t5 = [p for p in order if p.tier == 5]
    assert [p.unsloth for p in t5] == sorted((p.unsloth for p in t5), reverse=True)


# -------------------------------------------------------------- budget guard
def test_a_big_budget_drops_nothing(presets, grid):
    plan = L.plan_budget(presets, grid, budget_usd=100.0)
    assert not plan.dropped and len(plan.keep) == 96


def test_the_planner_drops_in_drop_order_and_fits(presets, grid):
    plan = L.plan_budget(presets, grid, budget_usd=L.DEFAULT_BUDGET_USD)
    assert plan.dropped and plan.keep
    assert plan.est_seconds <= plan.budget_seconds
    victims = L.drop_order(grid)
    assert plan.dropped == victims[: len(plan.dropped)]
    # nothing in tier 0 or 1 is dropped at the default budget: the latent auto-batch question survives
    assert all(p.tier >= 2 for p in plan.dropped)
    assert {p.tag for p in plan.keep if p.tier == 0} == {p.tag for p in grid if p.tier == 0}


def test_a_tiny_budget_drops_tier_zero_only_last(presets, grid):
    plan = L.plan_budget(presets, grid, budget_usd=0.35)
    dropped_tiers = [p.tier for p in plan.dropped]
    assert dropped_tiers == sorted(dropped_tiers, reverse=True)
    kept_tiers = {p.tier for p in plan.keep}
    if kept_tiers:
        assert min(kept_tiers) == 0  # whatever is left starts with the protected tier


def test_a_zero_budget_runs_nothing_and_says_so(presets, grid):
    plan = L.plan_budget(presets, grid, budget_usd=0.0)
    assert not plan.keep and plan.notes


def test_the_full_grid_estimate_is_hours_not_minutes(presets, grid):
    total, per, _ = L.estimate_total_seconds(presets, grid)
    assert 1.5 * 3600 < total < 4 * 3600
    assert per["qwen2.5-32b"] > per["llama-3.2-1b"]


def test_runtime_guard_drops_by_drop_order_when_over_budget(presets, grid):
    by = {p.name: p for p in presets}
    rest = list(grid)
    assert L.runtime_guard(rest, by, 8, elapsed_s=0.0, budget_s=1e9) == []
    total = sum(L.estimate_point_seconds(by[p.preset], p.batch, p.unsloth, 8) for p in rest)
    dropped = L.runtime_guard(rest, by, 8, elapsed_s=0.0, budget_s=total * 0.6)
    assert dropped == L.drop_order(rest)[: len(dropped)]
    # a slow pod (x2 observed/estimated) drops more
    more = L.runtime_guard(rest, by, 8, elapsed_s=0.0, budget_s=total * 0.6, scale=2.0)
    assert len(more) > len(dropped)


# ------------------------------------------------------------------ secrets
def test_scrub_removes_the_token_and_token_shapes():
    tok = "hf_" + "A" * 34
    assert tok not in L.scrub(f"401 for url ?token={tok}&x=1", [tok])
    assert "hf_" not in L.scrub(f"leak {tok}", [])  # shape-based, no secret list needed
    assert L.scrub("nothing here", ["short"]) == "nothing here"


def test_receipts_never_contain_the_token(tmp_path, monkeypatch):
    tok = "hf_" + "Z" * 36
    monkeypatch.setenv("HF_TOKEN", tok)
    path = S.write_receipt(str(tmp_path), {
        "mode": "e3_point", "tag": "t", "status": "error", "error": f"auth failed for {tok}",
        "manifest": {"hf_token_present": True}})
    text = pathlib.Path(path).read_text(encoding="utf-8")
    assert tok not in text and "***" in text
    assert json.loads(text)["manifest"]["hf_token_present"] is True


# ------------------------------------------------------------------ resume
def test_only_ok_and_oom_receipts_are_final(tmp_path):
    for tag, status in [("a", "ok"), ("b", "oom"), ("c", "error"), ("d", "dropped_by_budget_guard"),
                        ("e", "crashed"), ("f", "skipped_monotone_oom")]:
        S.write_receipt(str(tmp_path), {"mode": "e3_point", "tag": tag, "status": status, "manifest": {}})
    assert [S.receipt_is_final(str(tmp_path), t) for t in "abcdef"] == [True, True, False, False, False, False]
    assert not S.receipt_is_final(str(tmp_path), "missing")


def test_oom_detection_matches_the_trainers_error_text():
    assert S.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert S.is_oom(RuntimeError("RUNTIME_GPU_OOM: batch too big"))
    assert not S.is_oom(ValueError("bad shape"))


# ---------------------------------------------------------------- plan CLI
def test_plan_json_cli(tmp_path):
    out = subprocess.run([sys.executable, os.path.join(SCRIPTS, "pod_e3_sweep.py"), "plan", "--out", str(tmp_path),
                          "--json"], capture_output=True, text=True, timeout=60, check=True).stdout
    rep = json.loads(out)
    assert rep["points_total"] == 96 and rep["steps_per_point"] == 8
    assert rep["presets_small_to_large"][0] == "llama-3.2-1b" and rep["presets_small_to_large"][-1] == "qwen2.5-32b"
    assert rep["kept"] + rep["dropped"] == 96
    assert rep["by_tier"]["0"] == {"points": 12, "kept": 12}


# ------------------------------------------------------------- architecture
def test_fetch_arch_reads_a_local_config_on_the_meta_device(tmp_path):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig

    LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False).save_pretrained(tmp_path)
    arch = L.fetch_arch(str(tmp_path))
    assert (arch["hidden_size"], arch["num_hidden_layers"], arch["num_attention_heads"], arch["vocab_size"]) \
        == (64, 2, 4, 300)
    assert arch["loaded_class"] == "LlamaForCausalLM"
    # 2 layers x (q,o: 64*64; k,v: 64*32; gate,up,down: 3*64*128; 2 norms: 128) + embed + head + final norm
    per_layer = 2 * 64 * 64 + 2 * 64 * 32 + 3 * 64 * 128 + 2 * 64
    assert arch["text_params"] == 2 * per_layer + 2 * 300 * 64 + 64
    assert arch["vision_params_in_text_class"] == 0


# ----------------------------------------------------------- CPU dry run
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_synthetic_dry_run_end_to_end(tmp_path):
    """run --synthetic: planner -> orchestrator -> a real Trainer on CPU -> receipts -> summary -> refit plumbing."""
    cmd = [sys.executable, os.path.join(SCRIPTS, "pod_e3_sweep.py"), "run", "--out", str(tmp_path), "--synthetic",
           "--presets", "llama-3.2-1b,qwen2.5-32b", "--batches", "1,2", "--unsloth", "off", "--steps", "3"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=500)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    runs = sorted(os.listdir(tmp_path / "runs"))
    assert runs == ["e3_llama-3.2-1b_b1_plain.json", "e3_llama-3.2-1b_b2_plain.json",
                    "e3_qwen2.5-32b_b1_plain.json", "e3_qwen2.5-32b_b2_plain.json"]
    rec = json.loads((tmp_path / "runs" / runs[-1]).read_text())
    assert rec["status"] == "ok" and rec["dry_run"] is True and rec["oom_recovery"] is False
    assert rec["shape_ok"] is True and rec["step_input_shapes"][0] == [2, 128]
    assert rec["steps_completed"] == 3 and rec["s_per_step"] is not None
    assert rec["predicted_default"]["total_gb"] > 0 and rec["auto_batch_choice"] == 2
    assert rec["peak_source"].startswith("cpu_rss")
    assert "hf_token_present" in rec["manifest"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status_counts"] == {"ok": 4} and summary["dry_run"] is True
    # a re-run resumes: every point already has a final receipt
    r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r2.returncode == 0 and r2.stdout.count("measured receipt exists") == 4
    # the refit refuses dry-run receipts as evidence
    import e3_refit as R

    assert R.run(str(tmp_path), 0.05, False)["verdict"] == "INSUFFICIENT_DATA"
