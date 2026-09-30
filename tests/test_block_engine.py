"""Engine B (block-coordinate AdamW): unit tests on tiny random-weight models.

CPU only, no downloads. Covers the partition (Llama-style untied/tied, GPT-2,
a hand-built module tree), the switching schedule, freeing of optimizer
state, the fp32 active block inside a bf16 model under autocast (with and
without gradient checkpointing), stochastic vs nearest write-back, the
backward scope, AdamW equivalence, state_dict round-trips and the fit
estimate. The HF/TRL trainer integration lives in
``test_block_engine_trainer.py``.
"""

from __future__ import annotations

import copy
import io
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from backpropagate import block_engine as be  # noqa: E402
from backpropagate.exceptions import (  # noqa: E402
    CheckpointError,
    InvalidSettingError,
    TrainingError,
)
from tests.helpers.tiny_models import tiny_gpt2, tiny_llama  # noqa: E402


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _changed(model, snap):
    return {n for n, p in model.named_parameters() if not torch.equal(p.detach().to(snap[n].dtype), snap[n])}


def _batch(vocab=64, b=2, s=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, vocab, (b, s), generator=g)


def _train_steps(model, opt, n, *, autocast=False, seed=0, accumulate=1):
    model.train()
    for i in range(n):
        for j in range(accumulate):
            ids = _batch(seed=seed + i * 97 + j)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                loss = model(input_ids=ids, labels=ids).loss / accumulate
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------
class TestPartition:
    def test_llama_untied(self):
        m = tiny_llama(tied=False, layers=4)
        part = be.partition_model(m)
        assert part.layer_list_name == "model.layers"
        assert part.names == ["embed"] + [f"model.layers.{i}" for i in range(4)] + ["head"]
        assert part.tied is False
        assert part.blocks[0].param_names == ("model.embed_tokens.weight",)
        assert set(part.blocks[-1].param_names) == {"model.norm.weight", "lm_head.weight"}
        self._every_param_once(m, part)

    def test_llama_tied_weight_in_exactly_one_block(self):
        m = tiny_llama(tied=True, layers=3)
        assert m.lm_head.weight is m.model.embed_tokens.weight
        part = be.partition_model(m)
        assert part.tied is True
        assert part.names == ["embed+head"] + [f"model.layers.{i}" for i in range(3)]
        owners = [b.name for b in part.blocks if "model.embed_tokens.weight" in b.param_names]
        assert owners == ["embed+head"]
        assert "model.norm.weight" in part.blocks[0].param_names
        assert not any("lm_head" in n for b in part.blocks for n in b.param_names)
        self._every_param_once(m, part)

    def test_gpt2(self):
        m = tiny_gpt2(layers=3)
        part = be.partition_model(m)
        assert part.layer_list_name == "transformer.h"
        assert part.tied is True
        assert part.names == ["embed+head", "transformer.h.0", "transformer.h.1", "transformer.h.2"]
        assert {"transformer.wte.weight", "transformer.wpe.weight", "transformer.ln_f.weight"} <= set(
            part.blocks[0].param_names
        )
        self._every_param_once(m, part)

    def test_generic_module_tree(self):
        """Not an HF model: found by structure, not by name."""
        nn = torch.nn

        class Stack(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok = nn.Embedding(10, 4)
                self.trunk = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(3)])
                self.misc = nn.ModuleList([nn.Linear(4, 4), nn.ReLU()])  # mixed types: not a layer list
                self.out = nn.Linear(4, 10)

        part = be.partition_model(Stack())
        assert part.layer_list_name == "trunk"
        assert part.names == ["embed", "trunk.0", "trunk.1", "trunk.2", "head"]
        assert set(part.blocks[-1].param_names) == {"misc.0.weight", "misc.0.bias", "out.weight", "out.bias"}

    def test_freeze_embeddings(self):
        m = tiny_llama(tied=False, layers=2)
        part = be.partition_model(m, include_embeddings=False)
        assert part.names == ["model.layers.0", "model.layers.1"]
        assert set(part.frozen_param_names) == {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}

    def test_no_layer_list_raises(self):
        with pytest.raises(TrainingError) as ei:
            be.partition_model(torch.nn.Sequential(torch.nn.Linear(2, 2)))
        assert ei.value.code == "RUNTIME_TRAINING_FAILED"

    @staticmethod
    def _every_param_once(model, part):
        names = [n for b in part.blocks for n in b.param_names] + list(part.frozen_param_names)
        assert len(names) == len(set(names))
        assert set(names) == {n for n, _ in model.named_parameters()}


# ---------------------------------------------------------------------------
# Schedule, freezing, state freeing
# ---------------------------------------------------------------------------
class TestSchedule:
    def test_only_active_block_changes_then_switches(self):
        m = tiny_llama(tied=False, layers=3)
        opt = be.BlockCoordinateOptimizer(m, lr=1e-2, switch_block_every=3, block_order="ascending")
        part = opt.partition
        snap = _snapshot(m)
        _train_steps(m, opt, 2)
        assert opt.active_block_index == 0 and opt.steps_in_block == 2
        assert _changed(m, snap) == set(part.blocks[0].param_names)
        _train_steps(m, opt, 1, seed=50)
        # K reached -> switched to block 1, block 0 frozen and bit-stable from here.
        assert opt.active_block_index == 1 and opt.steps_in_block == 0
        snap2 = _snapshot(m)
        _train_steps(m, opt, 3, seed=80)
        assert _changed(m, snap2) == set(part.blocks[1].param_names)
        assert opt.active_block_index == 2

    def test_frozen_params_get_no_grad(self):
        m = tiny_llama(tied=False, layers=3)
        opt = be.BlockCoordinateOptimizer(m, lr=1e-3, switch_block_every=5, block_order="ascending")
        opt.switch()  # activate layer 0
        ids = _batch()
        m(input_ids=ids, labels=ids).loss.backward()
        active = set(opt.active_block.param_names)
        for n, p in m.named_parameters():
            assert (p.grad is not None) == (n in active), n
            assert p.requires_grad == (n in active), n

    def test_state_freed_on_switch(self):
        m = tiny_llama(tied=False, layers=2)
        opt = be.BlockCoordinateOptimizer(m, lr=1e-3, switch_block_every=2, block_order="ascending")
        _train_steps(m, opt, 1)
        assert len(opt.state) == len(opt.active_block.param_names)
        old = opt.block_params(0)
        _train_steps(m, opt, 1, seed=9)
        assert opt.active_block_index == 1
        assert len(opt.state) == 0
        assert all(p.grad is None and not p.requires_grad for p in old)
        assert all(p not in opt.state for p in old)
        flat = [p for g in opt.param_groups for p in g["params"]]
        assert {id(p) for p in flat} == {id(p) for p in opt.block_params(1)}

    @pytest.mark.parametrize("order", ["ascending", "descending", "random"])
    def test_orders_cover_every_block_each_epoch(self, order):
        m = tiny_llama(tied=False, layers=4)
        opt = be.BlockCoordinateOptimizer(m, lr=1e-3, switch_block_every=1, block_order=order, seed=3)
        d = len(opt.partition.blocks)
        visited = [opt.active_block_index]
        for _ in range(2 * d - 1):
            opt.switch()
            visited.append(opt.active_block_index)
        assert sorted(visited[:d]) == list(range(d))
        assert sorted(visited[d:]) == list(range(d))
        if order == "ascending":
            assert visited[:d] == list(range(d))
        if order == "descending":
            assert visited[:d] == list(range(d))[::-1]

    def test_random_order_is_seeded(self):
        def order(seed):
            opt = be.BlockCoordinateOptimizer(tiny_llama(layers=5), lr=1e-3, block_order="random", seed=seed)
            out = [opt.active_block_index]
            for _ in range(6):
                opt.switch()
                out.append(opt.active_block_index)
            return out

        assert order(1) == order(1)
        assert order(1) != order(2)

    def test_default_order_is_random(self):
        assert be.DEFAULT_BLOCK_ORDER == "random"
        assert be.DEFAULT_SWITCH_BLOCK_EVERY == 50
        assert be.DEFAULT_BLOCK_WRITEBACK == "stochastic"

    def test_finalize_restores_model(self):
        m = tiny_llama(tied=False, layers=2, dtype=torch.bfloat16)
        opt = be.BlockCoordinateOptimizer(m, lr=1e-3, switch_block_every=10)
        assert any(p.dtype == torch.float32 for p in m.parameters())
        opt.finalize()
        opt.finalize()  # idempotent
        assert all(p.dtype == torch.bfloat16 for p in m.parameters())
        assert all(p.requires_grad for p in m.parameters())
        with pytest.raises(TrainingError):
            opt.step()

    @pytest.mark.parametrize(
        "kw",
        [
            {"switch_block_every": 0},
            {"switch_block_every": True},
            {"block_order": "sideways"},
            {"block_writeback": "truncate"},
        ],
    )
    def test_bad_settings_raise_config_error(self, kw):
        with pytest.raises(InvalidSettingError) as ei:
            be.BlockCoordinateOptimizer(tiny_llama(layers=2), lr=1e-3, **kw)
        assert ei.value.code == "CONFIG_INVALID_SETTING"

    def test_step_without_gradient_raises(self):
        m = tiny_llama(layers=2)
        opt = be.BlockCoordinateOptimizer(m, lr=1e-3)
        with pytest.raises(TrainingError, match="no gradient reached"):
            opt.step()


# ---------------------------------------------------------------------------
# AdamW maths
# ---------------------------------------------------------------------------
class TestAdamW:
    def test_matches_torch_adamw_on_the_active_block(self):
        m1 = tiny_llama(layers=2, seed=4)
        m2 = copy.deepcopy(m1)
        opt = be.BlockCoordinateOptimizer(
            m1, lr=3e-3, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.1,
            switch_block_every=100, block_order="ascending",
        )
        names = opt.active_block.param_names
        for n, p in m2.named_parameters():
            p.requires_grad_(n in names)
        params = dict(m2.named_parameters())
        decay = [params[n] for n in names if params[n].ndim >= 2]
        nodecay = [params[n] for n in names if params[n].ndim < 2]
        ref = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
            lr=3e-3, betas=(0.8, 0.95), eps=1e-7, foreach=False,
        )
        for i in range(4):
            ids = _batch(seed=i)
            for model, o in ((m1, opt), (m2, ref)):
                model(input_ids=ids, labels=ids).loss.backward()
                o.step()
                o.zero_grad(set_to_none=True)
        for n, p in m1.named_parameters():
            torch.testing.assert_close(p, dict(m2.named_parameters())[n], rtol=0, atol=1e-7)


# ---------------------------------------------------------------------------
# Precision: fp32 active block inside a bf16 model
# ---------------------------------------------------------------------------
class TestPrecision:
    @pytest.mark.parametrize("gc", [False, True])
    @pytest.mark.parametrize("builder", [tiny_llama, tiny_gpt2])
    def test_fp32_block_trains_under_bf16_autocast(self, builder, gc):
        m = builder(dtype=torch.bfloat16)
        if gc:
            m.gradient_checkpointing_enable({"use_reentrant": False})
            be.drop_input_require_grads_hook(m)
        opt = be.BlockCoordinateOptimizer(m, lr=1e-2, switch_block_every=2, block_order="descending")
        active = set(opt.active_block.param_names)
        for n, p in m.named_parameters():
            assert p.dtype == (torch.float32 if n in active else torch.bfloat16), n
        snap = _snapshot(m)
        m.train()
        ids = _batch()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = m(input_ids=ids, labels=ids).loss
        assert torch.isfinite(loss)
        loss.backward()
        for n, p in m.named_parameters():
            if n in active:
                assert p.grad is not None and p.grad.dtype == torch.float32, n
            else:
                assert p.grad is None, n
        opt.step()
        opt.zero_grad(set_to_none=True)
        assert _changed(m, snap) <= active
        _train_steps(m, opt, 1, autocast=True, seed=5)
        # switched: written back to bf16, next block upcast
        for n in active:
            assert dict(m.named_parameters())[n].dtype == torch.bfloat16
        assert all(p.dtype == torch.float32 for p in opt.block_params(opt.active_block_index))

    def test_small_updates_accumulate_in_the_fp32_master(self):
        """An lr far below one bf16 ulp per step: each step would be lost if
        the weights were updated in bf16, but K steps accumulate in fp32."""
        m = tiny_llama(layers=2, dtype=torch.bfloat16)
        opt = be.BlockCoordinateOptimizer(m, lr=1e-5, switch_block_every=40, block_order="ascending",
                                          block_writeback="nearest")
        blk = list(opt.active_block.param_names)
        before = {n: dict(m.named_parameters())[n].detach().float().clone() for n in blk}
        _train_steps(m, opt, 40, autocast=True)
        params = dict(m.named_parameters())
        moved = sum(int((params[n].float() != before[n]).sum()) for n in blk)
        total = sum(before[n].numel() for n in blk)
        assert params[blk[0]].dtype == torch.bfloat16
        assert moved / total > 0.2


# ---------------------------------------------------------------------------
# Stochastic rounding
# ---------------------------------------------------------------------------
class TestStochasticRounding:
    def test_unbiased_mean_converges(self):
        g = torch.Generator().manual_seed(0)
        x = (torch.rand(512, generator=g) * 4 - 2).float()
        x = x + 2.0**-12  # make sure most values are not bf16-representable
        n = 4000
        acc = torch.zeros_like(x, dtype=torch.float64)
        gen = torch.Generator().manual_seed(1)
        for _ in range(n):
            acc += be.stochastic_round_to_bf16(x, gen).double()
        mean = acc / n
        ulp = (x.abs().double() * 2.0**-7)
        # standard error of the mean <= ulp / (2 sqrt(n)); allow 5 sigma
        assert torch.all((mean - x.double()).abs() <= 5 * ulp / (2 * n**0.5) + 1e-12)
        # round-to-nearest is biased on the same inputs
        nearest_err = (x.to(torch.bfloat16).double() - x.double()).abs().mean()
        assert (mean - x.double()).abs().mean() < nearest_err / 5

    def test_sub_half_ulp_value(self):
        """1 + 2^-10 is below half a bf16 ulp above 1.0 (ulp = 2^-7):
        nearest always returns 1.0, stochastic returns 1 + 2^-7 one time in 8."""
        x = torch.full((20000,), 1.0 + 2.0**-10)
        assert torch.all(x.to(torch.bfloat16).float() == 1.0)
        r = be.stochastic_round_to_bf16(x, torch.Generator().manual_seed(0)).float()
        assert set(r.unique().tolist()) == {1.0, 1.0 + 2.0**-7}
        assert abs(float(r.double().mean()) - (1.0 + 2.0**-10)) < 2e-4

    def test_negative_values_symmetric_and_exact_values_kept(self):
        x = torch.tensor([-(1.0 + 2.0**-10)] * 20000)
        r = be.stochastic_round_to_bf16(x, torch.Generator().manual_seed(0)).double()
        assert abs(float(r.mean()) + (1.0 + 2.0**-10)) < 2e-4
        exact = torch.tensor([0.0, -0.0, 1.0, -2.5, 3.140625, float("inf"), -float("inf")])
        out = be.stochastic_round_to_bf16(exact, torch.Generator().manual_seed(0))
        assert torch.equal(out.float(), exact)
        assert out.dtype == torch.bfloat16

    def test_chunking_matches_unchunked(self, monkeypatch):
        x = torch.randn(1000)
        a = be.stochastic_round_to_bf16(x, torch.Generator().manual_seed(7))
        monkeypatch.setattr(be, "_ROUND_CHUNK_ELEMENTS", 64)
        b = be.stochastic_round_to_bf16(x, torch.Generator().manual_seed(7))
        assert a.shape == b.shape
        # same generator stream, consumed in order: identical results
        assert torch.equal(a, b)

    @pytest.mark.parametrize("mode", ["stochastic", "nearest"])
    def test_written_back_value_is_not_overwritten(self, mode):
        """OneTrainer#994: a second copy_ after the stochastic copy silently
        replaced the rounded value with round-to-nearest. After a switch the
        parameter must hold exactly the value our rounding produced."""
        m = tiny_llama(layers=2, dtype=torch.bfloat16)
        opt = be.BlockCoordinateOptimizer(m, lr=1e-2, switch_block_every=100, block_order="ascending",
                                          block_writeback=mode)
        _train_steps(m, opt, 3, autocast=True)
        names = list(opt.active_block.param_names)
        params = dict(m.named_parameters())
        masters = {n: params[n].detach().clone() for n in names}
        assert all(t.dtype == torch.float32 for t in masters.values())
        gen = opt._sr_generator(torch.device("cpu"))
        expected = {}
        for n in names:
            expected[n] = (be.stochastic_round_to_bf16(masters[n], gen) if mode == "stochastic"
                           else masters[n].to(torch.bfloat16))
        opt.switch()
        for n in names:
            assert params[n].dtype == torch.bfloat16
            assert torch.equal(params[n].detach(), expected[n]), n
        if mode == "stochastic":
            nearest = torch.cat([masters[n].to(torch.bfloat16).float().reshape(-1) for n in names])
            got = torch.cat([params[n].detach().float().reshape(-1) for n in names])
            assert (got != nearest).any(), "stochastic write-back equals nearest everywhere"
        # and nothing later touches the frozen block
        after = {n: params[n].detach().clone() for n in names}
        _train_steps(m, opt, 2, autocast=True, seed=11)
        for n in names:
            assert torch.equal(params[n].detach(), after[n])


# ---------------------------------------------------------------------------
# Backward scope
# ---------------------------------------------------------------------------
def _graph_nodes(loss):
    seen, stack = set(), [loss.grad_fn]
    while stack:
        n = stack.pop()
        if n is None or n in seen:
            continue
        seen.add(n)
        stack.extend(f for f, _ in n.next_functions)
    return len(seen)


class TestBackwardScope:
    def _measure(self, m, opt, block, hook):
        while opt.active_block_index != block:
            opt.switch()
        if hook:
            m.enable_input_require_grads()
        else:
            be.drop_input_require_grads_hook(m)
        outs = []
        handles = [layer.register_forward_hook(lambda _m, _i, o: outs.append(o[0] if isinstance(o, tuple) else o))
                   for layer in m.model.layers]
        ids = _batch(b=2, s=32)
        loss = m(input_ids=ids, labels=ids).loss
        nodes = _graph_nodes(loss)
        t0 = time.perf_counter()
        loss.backward()
        dt = time.perf_counter() - t0
        for h in handles:
            h.remove()
        m.zero_grad(set_to_none=True)
        return nodes, dt, [o.requires_grad for o in outs]

    def test_backward_stops_at_the_active_block(self):
        m = tiny_llama(tied=False, layers=8, hidden=32)
        opt = be.BlockCoordinateOptimizer(m, lr=1e-3, switch_block_every=10, block_order="ascending")
        first_layer = opt.partition.names.index("model.layers.0")
        top_layer = opt.partition.names.index("model.layers.7")
        n_bottom, _, rg_bottom = self._measure(m, opt, first_layer, hook=False)
        n_top, _, rg_top = self._measure(m, opt, top_layer, hook=False)
        # top active: no layer below it produces a grad-requiring output
        assert rg_top == [False] * 7 + [True]
        assert rg_bottom == [True] * 8
        assert n_top < n_bottom / 3
        # the transformers input-grad hook would drag every layer back in
        opt2 = be.BlockCoordinateOptimizer(tiny_llama(tied=False, layers=8, hidden=32), lr=1e-3,
                                           switch_block_every=10, block_order="ascending")
        n_hook, _, rg_hook = self._measure(opt2.model, opt2, top_layer, hook=True)
        assert rg_hook == [True] * 8
        assert n_hook > n_top * 2

    def test_gradient_checkpointing_enable_installs_hook_and_callback_removes_it(self):
        m = tiny_llama(layers=2)
        m.gradient_checkpointing_enable({"use_reentrant": False})
        installed = getattr(m, "_require_grads_hook", None) is not None
        assert be.drop_input_require_grads_hook(m) is installed
        assert getattr(m, "_require_grads_hook", None) is None
        assert be.drop_input_require_grads_hook(m) is False


# ---------------------------------------------------------------------------
# state_dict round trip (plain loop)
# ---------------------------------------------------------------------------
class TestStateDict:
    @pytest.mark.parametrize("mode", ["nearest", "stochastic"])
    def test_resume_equivalence_plain_loop(self, mode):
        """Train 2K+2 steps straight == train K+2, save model + optimizer,
        rebuild both, resume, train K. Mid-block save exercises the master and
        Adam state; random order exercises the order RNG."""
        k = 3
        kw = {"lr": 5e-3, "switch_block_every": k, "block_order": "random", "block_writeback": mode, "seed": 5}
        base = tiny_llama(tied=False, layers=3, dtype=torch.bfloat16, seed=1)

        straight = copy.deepcopy(base)
        o1 = be.BlockCoordinateOptimizer(straight, **kw)
        _train_steps(straight, o1, 2 * k + 2, autocast=True)

        part = copy.deepcopy(base)
        o2 = be.BlockCoordinateOptimizer(part, **kw)
        _train_steps(part, o2, k + 2, autocast=True)
        buf_m, buf_o = io.BytesIO(), io.BytesIO()
        torch.save(part.state_dict(), buf_m)
        torch.save(o2.state_dict(), buf_o)
        buf_m.seek(0)
        buf_o.seek(0)

        resumed = copy.deepcopy(base)
        o3 = be.BlockCoordinateOptimizer(resumed, **kw)  # activates block perm[0]
        resumed.load_state_dict(torch.load(buf_m, weights_only=True))
        o3.load_state_dict(torch.load(buf_o, weights_only=True))
        assert o3.active_block_index == o2.active_block_index
        assert o3.steps_in_block == 2 and o3.global_step == k + 2
        # plain-loop seeds follow the global step index
        for i in range(k + 2, 2 * k + 2):
            ids = _batch(seed=i * 97)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                resumed(input_ids=ids, labels=ids).loss.backward()
            o3.step()
            o3.zero_grad(set_to_none=True)
        o1.finalize()
        o3.finalize()
        for (n, a), (_, b) in zip(straight.named_parameters(), resumed.named_parameters()):
            assert torch.equal(a, b), n

    def test_mismatched_partition_raises(self):
        o = be.BlockCoordinateOptimizer(tiny_llama(layers=2), lr=1e-3)
        other = be.BlockCoordinateOptimizer(tiny_llama(layers=3), lr=1e-3)
        with pytest.raises(CheckpointError) as ei:
            o.load_state_dict(other.state_dict())
        assert ei.value.code == "STATE_CHECKPOINT_INVALID"
        with pytest.raises(CheckpointError):
            o.load_state_dict({"state": {}, "param_groups": []})

    def test_same_block_roundtrip_is_identity(self):
        """accelerate's prepare() round-trips state_dict -> load_state_dict on
        the live optimizer; that must not disturb anything."""
        m = tiny_llama(layers=2, dtype=torch.bfloat16)
        o = be.BlockCoordinateOptimizer(m, lr=1e-2, switch_block_every=5)
        _train_steps(m, o, 2, autocast=True)
        snap = _snapshot(m)
        state_before = {k: v.clone() for p in o.state for k, v in o.state[p].items() if torch.is_tensor(v)}
        o.load_state_dict(o.state_dict())
        assert _changed(m, snap) == set()
        state_after = {k: v for p in o.state for k, v in o.state[p].items() if torch.is_tensor(v)}
        assert state_before.keys() == state_after.keys()
        assert o.steps_in_block == 2


# ---------------------------------------------------------------------------
# Fit estimate
# ---------------------------------------------------------------------------
class TestFitEstimate:
    def test_paper_formula(self):
        fit = be.estimate_block_engine_vram(8.0, 32, 512, 1)
        assert fit.paper_formula_gb == pytest.approx(2 * 8 + 16 * 8 / 32)
        assert fit.status.startswith("projection")

    def test_qwen7b_projection_and_embedding_dominance(self):
        # Qwen2.5-7B-Instruct: 7.616B, 28 layers, hidden 3584, vocab 152064, untied.
        emb = 152064 * 3584 / 1e9
        with_embed = be.estimate_block_engine_vram(
            7.616, 30, 512, 1, max_block_billions=emb, hidden_size=3584, num_layers=28,
            vocab_size=152064,
        )
        layer = (7.616 - 2 * emb) / 28
        frozen = be.estimate_block_engine_vram(
            7.616, 28, 512, 1, max_block_billions=layer, hidden_size=3584, num_layers=28,
            vocab_size=152064,
        )
        assert with_embed.active_block_gb == pytest.approx(14 * emb, rel=1e-3)
        assert with_embed.total_gb > frozen.total_gb + 4
        assert 15.0 < frozen.weights_gb < 15.5

    def test_fit_from_model(self):
        fit = be.fit_from_model(tiny_llama(layers=2), 16, 2)
        assert fit.num_blocks == 4
        assert fit.notes == []  # hidden / layers / vocab / max block all read off the model
        # tiny model: a layer (2336 params) outweighs the 64x16 vocab matrices
        assert fit.max_block_billions == pytest.approx(2336 / 1e9)
        assert set(fit.as_dict()) >= {"total_gb", "paper_formula_gb", "status"}
