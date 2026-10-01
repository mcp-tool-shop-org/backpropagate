"""CPU pins for the fused backward step (OffloadAdafactor.step_param_fused, FusedBackwardStep).

The fused path must produce the same updates as the 3-pass path, because it
runs the same routine. Three anchors:

* the update math against an independent fp32 reference (a plain restatement
  of the algorithm, no chunking, written here), so the routine itself is held
  to Shazeer and Stern's recipe;
* the fused path against the 3-pass path on a plain module, where a copy of
  each parameter stands in for FSDP2's host shard (bit-equal, including
  stochastic rounding, because each parameter owns its generator);
* the whole loop on a real one-rank FSDP2 model on CPU, fused against not.

The GPU speed claim is not tested here; the pod does that.
"""

from __future__ import annotations

import random

import pytest
import torch

import backpropagate.offload_engine as oe
from tests.helpers import offload_cpu

# fp32 tolerance against the reference. The routine sums g^2 in chunks and
# rounds the clip factor through float64; the reference sums in one go. That is
# a few ulp of a weight of order 1 (1.2e-7), over 5 steps.
_REF_RTOL, _REF_ATOL = 1e-5, 1e-6


def _reference_step(w, g, st, lr, wd, t, beta2_decay=-0.8, eps=1e-30, clip=1.0):
    """The Adafactor update the engine implements, restated densely in fp32."""
    beta2 = 1.0 - t**beta2_decay
    if w.dim() < 2:
        st["v"] = st.get("v", torch.zeros_like(w)) * beta2 + (g * g + eps) * (1.0 - beta2)
        u = g / st["v"].sqrt()
        u = u / max(1.0, float(u.pow(2).mean().sqrt()) / clip)
        return w * (1.0 - lr * wd) - lr * u if wd else w - lr * u
    rows = w.shape[0]
    g2 = g.pow(2) + eps
    st["row"] = st.get("row", torch.zeros(rows)) * beta2 + g2.mean(dim=1) * (1.0 - beta2)
    st["col"] = st.get("col", torch.zeros(g.shape[1])) * beta2 + g2.sum(dim=0) / rows * (1.0 - beta2)
    row_norm = st["row"] / st["row"].mean()
    u = g * row_norm.rsqrt().unsqueeze(1) * st["col"].rsqrt()
    scale = lr / max(1.0, float(u.pow(2).mean().sqrt()) / clip)
    base = w * (1.0 - lr * wd) if wd else w
    return base - scale * u


def _grads(shapes, steps, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return [[torch.randn(s, generator=gen) for s in shapes] for _ in range(steps)]


class TestAgainstReference:
    @pytest.mark.parametrize("chunk", [1 << 26, 100])  # one chunk, then ~10 chunks of 4 rows
    @pytest.mark.parametrize("wd", [0.0, 0.1])
    def test_fp32_update_matches_a_dense_reference(self, monkeypatch, chunk, wd):
        monkeypatch.setattr(oe, "_MAX_CHUNK_NUMEL", chunk)
        shapes = [(40, 24), (24,)]
        torch.manual_seed(1)
        start = [torch.randn(s) for s in shapes]
        params = [torch.nn.Parameter(t.clone()) for t in start]
        opt = oe.OffloadAdafactor(params, lr=1e-2, weight_decay=wd, device="cpu")
        ref_w = [t.clone() for t in start]
        ref_st: list[dict] = [{}, {}]
        for t, grads in enumerate(_grads(shapes, 5), start=1):
            for p, g in zip(params, grads):
                p.grad = g.clone()
            opt.step()
            for i, g in enumerate(grads):
                ref_w[i] = _reference_step(ref_w[i], g, ref_st[i], 1e-2, wd, t)
        for p, r in zip(params, ref_w):
            torch.testing.assert_close(p.data, r, rtol=_REF_RTOL, atol=_REF_ATOL)

    def test_fused_entry_matches_the_reference_too(self, monkeypatch):
        monkeypatch.setattr(oe, "_MAX_CHUNK_NUMEL", 100)
        shapes = [(40, 24), (24,)]
        torch.manual_seed(2)
        start = [torch.randn(s) for s in shapes]
        params = [torch.nn.Parameter(t.clone()) for t in start]
        opt = oe.OffloadAdafactor(params, lr=1e-2, weight_decay=0.1, device="cpu")
        ref_w = [t.clone() for t in start]
        ref_st: list[dict] = [{}, {}]
        for t, grads in enumerate(_grads(shapes, 4, seed=3), start=1):
            for p, g in zip(params, grads):
                opt.step_param_fused(p, g.clone(), p.data.clone())
            opt.step()  # finalizes; steps nothing (no host grads, all fused)
            for i, g in enumerate(grads):
                ref_w[i] = _reference_step(ref_w[i], g, ref_st[i], 1e-2, 0.1, t)
        for p, r in zip(params, ref_w):
            torch.testing.assert_close(p.data, r, rtol=_REF_RTOL, atol=_REF_ATOL)


class _Emulated:
    """A plain module whose parameters stand in for FSDP2's on-device copies.

    ``host`` holds the shard FSDP2 keeps in host RAM. Before each forward the
    module's copy is refreshed from it, as an all-gather does.
    """

    def __init__(self, dtype, seed=0, host_dtype=None):
        torch.manual_seed(seed)
        self.module = torch.nn.Sequential(torch.nn.Linear(24, 40), torch.nn.Linear(40, 8)).to(dtype)
        hd = host_dtype or dtype
        self.host = [torch.nn.Parameter(p.detach().clone().to(hd)) for p in self.module.parameters()]
        self.x = [torch.randn(6, 24, generator=torch.Generator().manual_seed(100 + i)).to(dtype) for i in range(6)]

    def gather(self):
        for u, h in zip(self.module.parameters(), self.host):
            u.data.copy_(h.data)

    def forward_backward(self, i):
        self.gather()
        self.module(self.x[i]).float().pow(2).mean().backward()


def _run(dtype, *, fused, steps=4, wd=0.1, attach_only=None, host_dtype=None, sr=True, seed=0):
    em = _Emulated(dtype, host_dtype=host_dtype)
    opt = oe.OffloadAdafactor(em.host, lr=1e-2, weight_decay=wd, device="cpu", seed=seed, stochastic_rounding=sr)
    hooks = oe.FusedBackwardStep(opt)
    if fused:
        for i, (u, h) in enumerate(zip(em.module.parameters(), em.host)):
            if attach_only is None or i in attach_only:
                hooks.attach(u, h)
    retention, fused_counts = [], []
    for i in range(steps):
        em.forward_backward(i)
        if not fused:
            for u, h in zip(em.module.parameters(), em.host):
                h.grad = u.grad.detach().clone().to(h.dtype)
                u.grad = None
        elif attach_only is not None:  # parameters without a hook keep their gradient: copy it down
            for u, h in zip(em.module.parameters(), em.host):
                if u.grad is not None:
                    h.grad = u.grad.detach().clone().to(h.dtype)
                    u.grad = None
        opt.step()
        retention.append(opt.last_update_retention)
        fused_counts.append(opt.last_fused_params)
        for h in em.host:
            h.grad = None
    hooks.remove()
    return em, opt, retention, fused_counts


class TestFusedMatchesThreePass:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("chunk", [1 << 26, 100])
    def test_bit_equal_over_several_steps(self, monkeypatch, dtype, chunk):
        monkeypatch.setattr(oe, "_MAX_CHUNK_NUMEL", chunk)
        a, _, ret_a, _ = _run(dtype, fused=False)
        b, _, ret_b, counts = _run(dtype, fused=True)
        for ha, hb in zip(a.host, b.host):
            assert torch.equal(ha.data, hb.data)
        assert ret_a == ret_b
        assert counts == [4, 4, 4, 4]

    def test_stochastic_rounding_changed_the_weights_and_is_seeded(self):
        a, _, _, _ = _run(torch.bfloat16, fused=True, seed=7)
        b, _, _, _ = _run(torch.bfloat16, fused=True, seed=7)
        c, _, _, _ = _run(torch.bfloat16, fused=True, seed=8)
        start = _Emulated(torch.bfloat16).host
        assert all(torch.equal(x.data, y.data) for x, y in zip(a.host, b.host))
        assert any(not torch.equal(x.data, y.data) for x, y in zip(a.host, c.host))
        assert any(not torch.equal(x.data, s.data) for x, s in zip(a.host, start))

    def test_update_retention_still_reads_about_one_with_stochastic_rounding(self):
        _, _, retention, _ = _run(torch.bfloat16, fused=True, wd=0.0)
        assert all(r == pytest.approx(1.0, abs=0.15) for r in retention), retention

    def test_gradients_are_cleared_so_fsdp_has_nothing_to_copy(self):
        em, _, _, _ = _run(torch.bfloat16, fused=True, steps=1)
        assert all(u.grad is None for u in em.module.parameters())
        assert all(h.grad is None for h in em.host)

    def test_parameters_without_a_hook_take_the_three_pass_step(self):
        """step() steps whatever still has a host gradient; the result is the same."""
        a, _, _, _ = _run(torch.float32, fused=False)
        b, _, _, counts = _run(torch.float32, fused=True, attach_only={0, 3})
        assert counts == [2, 2, 2, 2]
        for ha, hb in zip(a.host, b.host):
            assert torch.equal(ha.data, hb.data)

    def test_fp32_host_with_bf16_device_copy_reads_the_host_weights(self):
        """Dtypes differ (fp32 shard, bf16 gathered copy): the update reads the host weights."""
        a, _, _, _ = _run(torch.bfloat16, fused=False, host_dtype=torch.float32)
        b, _, _, _ = _run(torch.bfloat16, fused=True, host_dtype=torch.float32)
        for ha, hb in zip(a.host, b.host):
            assert ha.dtype == torch.float32
            assert torch.equal(ha.data, hb.data)

    def test_a_second_backward_before_step_is_refused(self):
        em = _Emulated(torch.float32)
        opt = oe.OffloadAdafactor(em.host, lr=1e-2, device="cpu")
        hooks = oe.FusedBackwardStep(opt)
        for u, h in zip(em.module.parameters(), em.host):
            hooks.attach(u, h)
        em.forward_backward(0)
        with pytest.raises(RuntimeError, match="gradient_accumulation == 1"):
            em.forward_backward(1)
        hooks.remove()

    def test_removed_hooks_leave_gradients_alone(self):
        em = _Emulated(torch.float32)
        opt = oe.OffloadAdafactor(em.host, lr=1e-2, device="cpu")
        hooks = oe.FusedBackwardStep(opt)
        for u, h in zip(em.module.parameters(), em.host):
            hooks.attach(u, h)
        hooks.remove()
        em.forward_backward(0)
        assert all(u.grad is not None for u in em.module.parameters())


class TestSwitchAndFitCheck:
    @pytest.mark.parametrize("raw,expected", [(None, False), ("", False), ("0", False), ("1", True), ("True", True)])
    def test_env_switch(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv("BACKPROPAGATE_OFFLOAD_FUSED", raising=False)
        else:
            monkeypatch.setenv("BACKPROPAGATE_OFFLOAD_FUSED", raw)
        assert oe.fused_backward_requested() is expected

    def test_fused_vram_estimate_adds_the_optimizer_chunk(self):
        root, layer, tokens = 1.09e9, 0.47e9, 512
        base = oe.offload_vram_required_gib(root, layer, tokens)
        fused = oe.offload_vram_required_gib(root, layer, tokens, fused=True)
        assert fused > base
        # the optimizer working set is the difference, up to the max() the default form takes
        assert fused - base <= oe._VRAM_OPTIMIZER_WORKSET_GIB + 1e-9

    def test_check_offload_fit_passes_the_flag_through(self):
        kw = {"params": 7.6e9, "root_unit_bytes": 1.09e9, "layer_bytes": 0.47e9, "tokens": 512,
              "host_total_gib": 64, "host_available_gib": 60, "vram_total_gib": 31.4}
        assert oe.check_offload_fit(**kw, fused=True)["vram_required_gib"] > oe.check_offload_fit(**kw)["vram_required_gib"]


class TestGenerators:
    def test_noise_depends_on_the_parameter_not_the_step_order(self):
        """Forward order and backward order must round identically."""
        torch.manual_seed(0)
        start = [torch.randn(16, 8).to(torch.bfloat16), torch.randn(8, 16).to(torch.bfloat16)]
        grads = [torch.randn(16, 8).to(torch.bfloat16), torch.randn(8, 16).to(torch.bfloat16)]
        outs = []
        for order in ([0, 1], [1, 0]):
            params = [torch.nn.Parameter(t.clone()) for t in start]
            opt = oe.OffloadAdafactor(params, lr=1e-3, device="cpu", seed=3)
            for i in order:
                opt.step_param_fused(params[i], grads[i].clone())
            outs.append([p.data.clone() for p in params])
        assert all(torch.equal(x, y) for x, y in zip(*outs))

    def test_global_rng_does_not_leak_in(self):
        results = []
        for global_seed in (0, 1):
            torch.manual_seed(global_seed)
            p = torch.nn.Parameter(torch.ones(32, 8).to(torch.bfloat16))
            p.grad = torch.full((32, 8), 0.5).to(torch.bfloat16)
            oe.OffloadAdafactor([p], lr=1e-3, device="cpu", seed=5).step()
            results.append(p.data.clone())
        assert torch.equal(*results)


@pytest.fixture(scope="module")
def fsdp_world():
    offload_cpu.init_world()
    yield
    offload_cpu.destroy_world()


def _fsdp_run(*, fused, steps=4, ga=1, tie=False):
    model = oe.shard_for_cpu_offload(offload_cpu.tiny_llama(tie=tie), mesh=offload_cpu.cpu_mesh())
    out = oe._train_loop(
        model, offload_cpu.ToyTokenizer(), offload_cpu.toy_dataset(), steps=steps, batch_size=2,
        gradient_accumulation=ga, learning_rate=1e-3, max_seq_length=32, warmup_steps=0,
        lr_scheduler_type="constant", weight_decay=0.05, rng=random.Random(0),
        device=torch.device("cpu"), on_step=None, seed=11, fused=fused,
    )
    return out


@pytest.mark.serial
class TestFusedOnRealFsdp2:
    def test_losses_and_weights_are_bit_equal_to_the_three_pass_step(self, fsdp_world):
        base = _fsdp_run(fused=False)
        fused = _fsdp_run(fused=True)
        assert fused["fused"] is True and base["fused"] is False
        n_params = len(list(base["model"].parameters()))
        assert fused["fused_params"] == [n_params] * 4
        assert base["fused_params"] == [0] * 4
        assert fused["losses"] == base["losses"]
        assert fused["update_retention"] == base["update_retention"]
        for (name, pa), (_, pb) in zip(base["model"].named_parameters(), fused["model"].named_parameters()):
            assert torch.equal(pa.to_local(), pb.to_local()), name
        # and the run did learn something
        assert fused["losses"][-1] != fused["losses"][0]

    def test_tied_embeddings_are_stepped_once_after_both_gradients_arrive(self, fsdp_world):
        """Qwen2.5-1.5B ties the embedding and the head: one parameter, two gradient sources."""
        base = _fsdp_run(fused=False, tie=True)
        fused = _fsdp_run(fused=True, tie=True)
        n_params = len(list(base["model"].parameters()))
        assert fused["fused_params"] == [n_params] * 4
        assert fused["losses"] == base["losses"]
        for (name, pa), (_, pb) in zip(base["model"].named_parameters(), fused["model"].named_parameters()):
            assert torch.equal(pa.to_local(), pb.to_local()), name

    def test_no_gradient_is_copied_to_the_host_when_fused(self, fsdp_world, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_OFFLOAD_TRACE", "1")
        base = _fsdp_run(fused=False)
        fused = _fsdp_run(fused=True)
        n_bytes = sum(p.to_local().numel() * 2 for p in base["model"].parameters())
        b, f = base["trace"]["steps"][1]["bytes"], fused["trace"]["steps"][1]["bytes"]
        assert b["grad_reduce_d2h"] == n_bytes
        assert "grad_reduce_d2h" not in f
        # what the fused path adds is the same write-back, and no host-to-device gradient/weight copies
        assert f["opt_writeback_d2h"] == b["opt_writeback_d2h"] == n_bytes

    def test_gradient_accumulation_falls_back_to_the_three_pass_step(self, fsdp_world, caplog):
        with caplog.at_level("INFO", logger="backpropagate.offload_engine"):
            out = _fsdp_run(fused=True, steps=2, ga=2)
        assert out["fused"] is False
        assert out["fused_params"] == [0, 0]
        assert "needs gradient_accumulation == 1" in caplog.text
        ref = _fsdp_run(fused=False, steps=2, ga=2)
        assert out["losses"] == ref["losses"]

    def test_hooks_are_removed_when_the_loop_ends(self, fsdp_world):
        out = _fsdp_run(fused=True, steps=2)
        model = out["model"]
        ids = torch.randint(2, 60, (2, 8))
        model(input_ids=ids, labels=ids).loss.backward()
        # with the hooks gone FSDP2 copies the gradients down as usual
        assert all(p.grad is not None for p in model.parameters())
