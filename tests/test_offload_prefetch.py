"""CPU pins for the explicit layer prefetch (BACKPROPAGATE_OFFLOAD_PREFETCH).

The overlap itself is a GPU property and is measured on the pod (the trace's
``fwd_gather`` leg). What is pinned here, on a real one-rank FSDP2 model with a
CPU mesh: the env switch, that the right layers are listed, that a listed
layer's gather really is issued before the current layer computes (the point of
the feature, and the thing FSDP2's default does not do), and that prefetching
never changes the numbers.
"""

from __future__ import annotations

import random

import pytest
import torch

import backpropagate.offload_engine as oe
from tests.helpers import offload_cpu


class TestPrefetchDepth:
    @pytest.mark.parametrize(
        "raw,expected",
        [(None, 0), ("", 0), ("0", 0), ("1", 1), (" 3 ", 3), ("8", 8), ("99", 8), ("-2", 0), ("two", 0)],
    )
    def test_env(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv("BACKPROPAGATE_OFFLOAD_PREFETCH", raising=False)
        else:
            monkeypatch.setenv("BACKPROPAGATE_OFFLOAD_PREFETCH", raw)
        assert oe.prefetch_depth() == expected

    def test_fit_estimate_counts_extra_resident_layers(self):
        root, layer = 1.09e9, 0.47e9
        base = oe.offload_vram_required_gib(root, layer, 512)
        assert oe.offload_vram_required_gib(root, layer, 512, prefetch=1) == pytest.approx(base)
        assert oe.offload_vram_required_gib(root, layer, 512, prefetch=4) == pytest.approx(base + 3 * layer / 2**30)

    def test_a_model_without_the_fsdp_api_is_left_alone(self):
        assert oe.set_explicit_prefetch(torch.nn.Linear(2, 2), 2) == 0


@pytest.fixture(scope="module")
def fsdp_world():
    offload_cpu.init_world()
    yield
    offload_cpu.destroy_world()


def _sharded(layers=4):
    return oe.shard_for_cpu_offload(offload_cpu.tiny_llama(layers=layers), mesh=offload_cpu.cpu_mesh())


def _lists(layer):
    state = layer._get_fsdp_state()
    return state._states_to_forward_prefetch, state._states_to_backward_prefetch


@pytest.mark.serial
class TestExplicitPrefetch:
    def test_depth_zero_configures_nothing(self, fsdp_world):
        model = _sharded()
        assert oe.set_explicit_prefetch(model, 0) == 0
        assert all(_lists(layer) == ([], []) for layer in oe._decoder_layers(model))

    def test_each_layer_lists_the_next_layers_forward_and_the_previous_ones_backward(self, fsdp_world):
        model = _sharded(layers=4)
        layers = oe._decoder_layers(model)
        assert oe.set_explicit_prefetch(model, 2) == 5  # 4 layers + the root
        states = [layer._get_fsdp_state() for layer in layers]
        fwd = [[states.index(s) for s in _lists(layer)[0]] for layer in layers]
        bwd = [[states.index(s) for s in _lists(layer)[1]] for layer in layers]
        assert fwd == [[1, 2], [2, 3], [3], []]
        assert bwd == [[], [0], [1, 0], [2, 1]]
        root_fwd, root_bwd = _lists(model)
        assert [states.index(s) for s in root_fwd] == [0]
        assert [states.index(s) for s in root_bwd] == [3, 2]

    def _unshard_log(self, monkeypatch, model):
        """Record, in order, each group's unshard and the start of each layer's attention compute."""
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

        layers = oe._decoder_layers(model)
        by_group = {id(group): layers.index(module) for module, group in oe._fsdp_units(model) if module in layers}
        log: list[tuple[str, int]] = []
        orig = pg.FSDPParamGroup.unshard

        def unshard(self, async_op=False):
            if id(self) in by_group and self._training_state.name == "FORWARD":
                log.append(("unshard", by_group[id(self)]))
            return orig(self, async_op)

        monkeypatch.setattr(pg.FSDPParamGroup, "unshard", unshard)
        for i, layer in enumerate(layers):
            layer.self_attn.register_forward_pre_hook(lambda _m, _a, i=i: log.append(("compute", i)))
        return log

    def test_default_gathers_a_layer_only_when_it_is_reached(self, fsdp_world, monkeypatch):
        model = _sharded(layers=3)
        log = self._unshard_log(monkeypatch, model)
        ids = torch.randint(2, 60, (2, 8))
        with torch.no_grad():
            model(input_ids=ids)
        assert log.index(("unshard", 1)) > log.index(("compute", 0))

    def test_depth_one_issues_the_next_gather_before_the_current_layer_computes(self, fsdp_world, monkeypatch):
        model = _sharded(layers=3)
        oe.set_explicit_prefetch(model, 1)
        log = self._unshard_log(monkeypatch, model)
        ids = torch.randint(2, 60, (2, 8))
        with torch.no_grad():
            model(input_ids=ids)
        assert log.index(("unshard", 0)) < log.index(("unshard", 1)) < log.index(("compute", 0))
        assert log.index(("unshard", 2)) < log.index(("compute", 1))

    def test_depth_two_issues_two_layers_ahead(self, fsdp_world, monkeypatch):
        model = _sharded(layers=4)
        oe.set_explicit_prefetch(model, 2)
        log = self._unshard_log(monkeypatch, model)
        ids = torch.randint(2, 60, (2, 8))
        with torch.no_grad():
            model(input_ids=ids)
        assert log.index(("unshard", 2)) < log.index(("compute", 0))

    @pytest.mark.parametrize("fused", [False, True])
    @pytest.mark.parametrize("depth", [1, 3])
    def test_prefetch_does_not_change_the_numbers(self, fsdp_world, fused, depth):
        def run(d):
            model = _sharded(layers=4)
            oe.set_explicit_prefetch(model, d)
            return oe._train_loop(
                model, offload_cpu.ToyTokenizer(), offload_cpu.toy_dataset(), steps=3, batch_size=2,
                gradient_accumulation=1, learning_rate=1e-3, max_seq_length=32, warmup_steps=0,
                lr_scheduler_type="constant", weight_decay=0.05, rng=random.Random(0),
                device=torch.device("cpu"), on_step=None, seed=5, fused=fused, prefetch=d,
            )

        base, pre = run(0), run(depth)
        assert pre["prefetch"] == depth and base["prefetch"] == 0
        assert pre["losses"] == base["losses"]
        for (name, a), (_, b) in zip(base["model"].named_parameters(), pre["model"].named_parameters()):
            assert torch.equal(a.to_local(), b.to_local()), name
