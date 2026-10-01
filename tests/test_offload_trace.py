"""CPU pins for the offload engine's per-leg trace (backpropagate.offload_trace).

The CUDA-event path needs a GPU and runs on the pod. What is pinned here: the
env switch, the bookkeeping (legs, bytes, the derived compute leg, the
steady-state mean), the Chrome-trace reduction, that tracing is truly off by
default (nothing patched, no ``trace`` key), and that the FSDP2 probes count
the right bytes on a real one-rank FSDP2 model running on CPU.
"""

from __future__ import annotations

import random

import pytest
import torch

import backpropagate.offload_engine as oe
import backpropagate.offload_trace as ot
from tests.helpers import offload_cpu


class TestTraceMode:
    @pytest.mark.parametrize("raw", [None, "", "0", "off", "False", " no "])
    def test_off(self, monkeypatch, raw):
        if raw is None:
            monkeypatch.delenv("BACKPROPAGATE_OFFLOAD_TRACE", raising=False)
        else:
            monkeypatch.setenv("BACKPROPAGATE_OFFLOAD_TRACE", raw)
        assert ot.trace_mode() == "off"

    @pytest.mark.parametrize("raw", ["1", "true", "yes"])
    def test_legs(self, monkeypatch, raw):
        monkeypatch.setenv("BACKPROPAGATE_OFFLOAD_TRACE", raw)
        assert ot.trace_mode() == "legs"

    def test_profile(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_OFFLOAD_TRACE", "Profile")
        assert ot.trace_mode() == "profile"


class TestLegTrace:
    def test_legs_and_bytes_fold_into_a_step_record(self):
        tr = ot.LegTrace(torch.device("cpu"))
        with tr.leg("opt_h2d", 100):
            pass
        with tr.leg("opt_h2d", 50):
            pass
        with tr.leg("opt_total"):
            pass
        with tr.leg("opt_writeback_d2h", 30):
            pass
        rec = tr.end_step(0, 12.0)
        assert rec["bytes"] == {"opt_h2d": 150, "opt_total": 0, "opt_writeback_d2h": 30}
        assert rec["calls"]["opt_h2d"] == 2
        assert rec["step_ms"] == 12.0
        # the derived leg is total minus the two copy legs (all ~0 ms on CPU)
        assert "opt_compute" in rec["ms"]
        assert rec["ms"]["opt_compute"] == pytest.approx(
            rec["ms"]["opt_total"] - rec["ms"]["opt_h2d"] - rec["ms"]["opt_writeback_d2h"], abs=0.02
        )

    def test_end_step_resets_the_accumulators(self):
        tr = ot.LegTrace(torch.device("cpu"))
        with tr.leg("forward", 7):
            pass
        tr.end_step(0, 1.0)
        rec = tr.end_step(1, 1.0)
        assert rec["ms"] == {} and rec["bytes"] == {}

    def test_mean_skips_the_first_step(self):
        tr = ot.LegTrace(torch.device("cpu"))
        tr.add("backward", 100.0, 1_000_000_000)
        tr.end_step(0, 500.0)
        for i, (ms, nbytes) in enumerate([(10.0, 2_000_000_000), (30.0, 4_000_000_000)], start=1):
            tr.add("backward", ms, nbytes)
            tr.end_step(i, 40.0)
        out = tr.summary({"pin": "register"})
        assert out["mean_over_steps"] == 2
        assert out["mean_ms"]["backward"] == pytest.approx(20.0)
        assert out["mean_bytes"]["backward"] == 3_000_000_000
        assert out["mean_step_ms"] == pytest.approx(40.0)
        assert out["mean_gb_per_s"]["backward"] == pytest.approx(150.0, rel=1e-2)
        assert out["config"] == {"pin": "register"}
        assert len(out["steps"]) == 3

    def test_a_single_step_is_its_own_mean(self):
        tr = ot.LegTrace(torch.device("cpu"))
        tr.add("forward", 5.0)
        tr.end_step(0, 9.0)
        assert tr.summary()["mean_ms"]["forward"] == 5.0


class TestChromeTraceSummary:
    def test_memcpy_kernel_and_fsdp_ranges(self):
        events = [
            {"cat": "gpu_memcpy", "name": "Memcpy HtoD (Pageable -> Device)", "dur": 2000.0, "args": {"bytes": 4096}},
            {"cat": "gpu_memcpy", "name": "Memcpy HtoD (Pageable -> Device)", "dur": 1000.0, "args": {"bytes": 1024}},
            {"cat": "gpu_memcpy", "name": "Memcpy DtoH (Device -> Pageable)", "dur": 500.0, "args": {"bytes": 512}},
            {"cat": "kernel", "name": "gemm", "dur": 3000.0},
            {"cat": "kernel", "name": "gemm", "dur": 1000.0},
            {"cat": "kernel", "name": "elementwise", "dur": 100.0},
            {"cat": "user_annotation", "name": "FSDP::post_backward_reduce", "dur": 8000.0},
            {"cat": "user_annotation", "name": "aten::mm", "dur": 1.0},
        ]
        out = ot.summarize_chrome_trace(events, top_kernels=1)
        h2d = out["memcpy"]["Memcpy HtoD (Pageable -> Device)"]
        assert h2d == {"count": 2, "ms": 3.0, "bytes": 5120}
        assert out["memcpy"]["Memcpy DtoH (Device -> Pageable)"]["bytes"] == 512
        assert out["kernel_ms_total"] == pytest.approx(4.1)
        assert list(out["kernel_top"]) == ["gemm"]
        assert out["fsdp_ranges_cpu_ms"] == {"FSDP::post_backward_reduce": {"count": 1, "ms": 8.0}}


class TestOptimizerLegs:
    def test_off_by_default_uses_the_shared_empty_context(self):
        opt = oe.OffloadAdafactor([torch.nn.Parameter(torch.zeros(4))], lr=1e-3, device="cpu")
        assert opt.trace is None
        assert opt._leg("opt_total") is oe._NO_LEG

    def test_writeback_bytes_and_total_are_counted(self):
        p = torch.nn.Parameter(torch.randn(16, 8).to(torch.bfloat16))
        p.grad = torch.randn(16, 8).to(torch.bfloat16)
        v = torch.nn.Parameter(torch.ones(8))
        v.grad = torch.ones(8)
        opt = oe.OffloadAdafactor([p, v], lr=1e-3, device="cpu")
        opt.trace = ot.LegTrace(torch.device("cpu"))
        opt.step()
        rec = opt.trace.end_step(0, 1.0)
        assert rec["calls"]["opt_total"] == 2
        assert rec["bytes"]["opt_writeback_d2h"] == 16 * 8 * 2 + 8 * 4
        # same-device tensors are not copied, so no h2d leg on CPU
        assert "opt_h2d" not in rec["bytes"]


@pytest.fixture(scope="module")
def fsdp_world():
    offload_cpu.init_world()
    yield
    offload_cpu.destroy_world()


def _run_loop(steps: int = 3, **kw):
    model = offload_cpu.tiny_llama()
    model = oe.shard_for_cpu_offload(model, mesh=offload_cpu.cpu_mesh())
    n_bytes = sum(p.to_local().numel() * 2 for p in model.parameters())
    out = oe._train_loop(
        model, offload_cpu.ToyTokenizer(), offload_cpu.toy_dataset(), steps=steps, batch_size=2,
        gradient_accumulation=kw.pop("gradient_accumulation", 1), learning_rate=1e-3, max_seq_length=32,
        warmup_steps=0, lr_scheduler_type="constant", weight_decay=0.0, rng=random.Random(0),
        device=torch.device("cpu"), on_step=None, **kw,
    )
    return out, n_bytes


@pytest.mark.serial
class TestTrainLoopTrace:
    def test_off_returns_no_trace_and_patches_nothing(self, fsdp_world, monkeypatch):
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

        monkeypatch.delenv("BACKPROPAGATE_OFFLOAD_TRACE", raising=False)
        wait, reduce_fn = pg.FSDPParamGroup.wait_for_unshard, pg.foreach_reduce
        out, _ = _run_loop(steps=2)
        assert "trace" not in out
        assert pg.FSDPParamGroup.wait_for_unshard is wait and pg.foreach_reduce is reduce_fn

    def test_legs_count_the_bytes_that_cross(self, fsdp_world, monkeypatch):
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

        monkeypatch.setenv("BACKPROPAGATE_OFFLOAD_TRACE", "1")
        wait, reduce_fn = pg.FSDPParamGroup.wait_for_unshard, pg.foreach_reduce
        out, n_bytes = _run_loop(steps=3)
        # probes are removed when the loop ends
        assert pg.FSDPParamGroup.wait_for_unshard is wait and pg.foreach_reduce is reduce_fn
        trace = out["trace"]
        assert trace["mode"] == "legs" and trace["device"] == "cpu"
        assert len(trace["steps"]) == 3
        step = trace["steps"][1]
        for leg in ("forward", "backward", "optimizer_step", "fwd_gather", "bwd_gather", "grad_reduce_d2h",
                    "opt_total", "opt_writeback_d2h", "opt_compute"):
            assert leg in step["ms"], leg
        # every parameter's gradient goes to the host once; every weight is written back once
        assert step["bytes"]["grad_reduce_d2h"] == n_bytes
        assert step["bytes"]["opt_writeback_d2h"] == n_bytes
        # the forward gathers the whole model (root + 2 layers); the layers are gathered again in backward
        assert step["bytes"]["fwd_gather"] == n_bytes
        layer_bytes = n_bytes - sum(
            p.to_local().numel() * 2 for name, p in out["model"].named_parameters() if ".layers." not in name
        )
        assert step["bytes"]["bwd_gather"] == layer_bytes
        assert "profile" in trace and trace["profile"] is None

    def test_profile_mode_stores_a_summary_for_one_step(self, fsdp_world, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_OFFLOAD_TRACE", "profile")
        out, _ = _run_loop(steps=2)
        profile = out["trace"]["profile"]
        assert profile["step"] == 1
        assert set(profile) >= {"memcpy", "kernel_ms_total", "kernel_top", "fsdp_ranges_cpu_ms"}
        assert any(name.startswith("FSDP::") for name in profile["fsdp_ranges_cpu_ms"])


def test_trainer_carries_the_trace_and_switches_into_run_metadata(monkeypatch, tmp_path):
    """A pod receipt reads these from TrainingRun.metadata."""
    import json

    import backpropagate.trainer as t

    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"text": "hello world"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(t, "_ensure_fsdp_runtime", lambda: None)
    trainer = t.Trainer(model="HuggingFaceTB/SmolLM2-135M-Instruct", use_unsloth=False,
                        mode="full", full_ft_offload=True, output_dir=str(tmp_path / "o"),
                        report_to="none")
    monkeypatch.setattr(trainer, "load_model", lambda: setattr(trainer, "_is_loaded", True))
    monkeypatch.setattr(trainer, "_load_dataset", lambda *a, **k: ["row"])
    monkeypatch.setattr(oe, "run_offload_training", lambda model, tok, ds, **kw: {
        "model": model, "losses": [2.0, 1.5], "step_times": [0.1, 0.1], "samples_seen": 2,
        "duration_seconds": 0.2, "optimizer": None, "fused": True, "fused_params": [5, 5],
        "prefetch": 2, "trace": {"mode": "legs", "mean_ms": {"backward": 1.0}},
    })
    monkeypatch.setattr(t.Trainer, "_build_trainer", lambda *a, **k: pytest.fail("SFTTrainer built"))
    run = trainer.train(str(data), steps=2)
    assert run.metadata["fused"] is True and run.metadata["fused_params"] == [5, 5]
    assert run.metadata["prefetch"] == 2
    assert run.metadata["trace"]["mean_ms"] == {"backward": 1.0}
