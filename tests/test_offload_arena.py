"""CPU pins for the pinned host arena (HostArena, move_params_to_arena, BACKPROPAGATE_OFFLOAD_PIN=arena).

The arena replaces one ``cudaHostRegister`` per parameter storage with one slab
that is registered once. What matters, and is pinned here without a GPU:

* the slab is exactly the sum of the page-aligned shards (no power-of-two
  rounding, the failure of ``pin_memory=True`` at 7B), and every shard starts on
  a page;
* after ``move_params_to_arena`` FSDP2's gather source and the optimizer's
  view are both views of the slab, hold the values they held before, and a
  training loop on top of it is bit-equal to one without;
* a model it cannot handle is refused before anything is changed.

Only the ``cudaHostRegister`` call itself needs CUDA; that test is marked
``integration`` and skipped without a device.
"""

from __future__ import annotations

import random
import weakref

import pytest
import torch

import backpropagate.offload_engine as oe
from tests.helpers import offload_cpu

PAGE = oe._ARENA_ALIGN


class TestHostArena:
    def test_planned_bytes_is_the_sum_of_aligned_sizes(self):
        sizes = [1, PAGE, PAGE + 1, 3 * PAGE - 7]
        assert oe.HostArena.planned_bytes(sizes) == PAGE + PAGE + 2 * PAGE + 3 * PAGE
        assert oe.HostArena.planned_bytes([]) == 0

    def test_no_rounding_blowup_at_seven_billion_parameters(self):
        """~340 shards of a 7.6B bf16 model: the slab is the weights plus under a page each.

        torch's pinned allocator would round a 15.2 GB request up to 17.2 GB (next
        power of two), and each of the 340 blocks on its own up to ~1.8x in total.
        """
        sizes = [136_000_000] * 84 + [2_000_000_000, 1_090_000_000] + [14_336] * 254
        total = sum(sizes)
        planned = oe.HostArena.planned_bytes(sizes)
        assert 0 <= planned - total < len(sizes) * PAGE
        assert planned / total < 1.0001
        pow2 = 1 << (total - 1).bit_length()
        assert planned < pow2

    def test_slab_start_and_every_view_are_page_aligned_and_disjoint(self):
        arena = oe.HostArena(oe.HostArena.planned_bytes([100, 5000, 12, 4096]))
        assert arena.slab.data_ptr() % PAGE == 0
        views = [
            arena.take((10, 5), torch.bfloat16),  # 100 bytes
            arena.take((1250,), torch.float32),  # 5000 bytes
            arena.take((6,), torch.bfloat16),  # 12 bytes
            arena.take((4, 256), torch.float32),  # 4096 bytes
        ]
        spans = []
        for v in views:
            assert v.data_ptr() % PAGE == 0
            start = v.data_ptr() - arena.slab.data_ptr()
            assert start >= 0 and start + v.numel() * v.element_size() <= arena.nbytes
            spans.append((start, start + v.numel() * v.element_size()))
        spans.sort()
        assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))
        assert views[0].shape == (10, 5) and views[3].shape == (4, 256)

    def test_views_alias_the_slab(self):
        arena = oe.HostArena(oe.HostArena.planned_bytes([64, 64]))
        a = arena.take((16,), torch.float32)
        b = arena.take((32,), torch.bfloat16)
        a.fill_(1.5)
        b.fill_(2.0)
        raw = arena.slab.view(torch.uint8)
        assert torch.equal(raw[: a.numel() * 4].view(torch.float32), torch.full((16,), 1.5))
        assert torch.equal(raw[PAGE : PAGE + 64].view(torch.bfloat16).float(), torch.full((32,), 2.0))

    def test_taking_more_than_planned_is_an_error(self):
        arena = oe.HostArena(PAGE)
        arena.take((PAGE // 4,), torch.float32)
        with pytest.raises(ValueError, match="full"):
            arena.take((1,), torch.float32)

    def test_release_without_register_is_a_no_op(self):
        oe.HostArena(PAGE).release()

    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="cudaHostRegister needs CUDA")
    def test_register_page_locks_the_slab(self):
        arena = oe.HostArena(16 * PAGE)
        try:
            assert arena.register() is True
            assert arena.slab.is_pinned()
        finally:
            arena.release()
        assert not arena.slab.is_pinned()


class TestPinMode:
    def test_arena_is_a_mode_and_unknown_values_still_mean_register(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_OFFLOAD_PIN", "Arena")
        assert oe._pin_mode() == "arena"
        monkeypatch.setenv("BACKPROPAGATE_OFFLOAD_PIN", "bogus")
        assert oe._pin_mode() == "register"
        monkeypatch.delenv("BACKPROPAGATE_OFFLOAD_PIN")
        assert oe._pin_mode() == "register"

    def test_a_model_the_arena_cannot_take_falls_back_to_register(self, monkeypatch, caplog):
        calls = []
        monkeypatch.setenv("BACKPROPAGATE_OFFLOAD_PIN", "arena")
        monkeypatch.setattr(oe, "register_host_params", lambda m: calls.append("register") or [7])
        monkeypatch.setattr(oe, "unregister_host_params", lambda ptrs: calls.append(("unregister", ptrs)))
        with caplog.at_level("WARNING", logger="backpropagate.offload_engine"):
            unpin = oe.pin_host_params(torch.nn.Linear(2, 2))  # no FSDP2 units
        assert "host arena unavailable" in caplog.text
        unpin()
        assert calls == ["register", ("unregister", [7])]

    def test_none_mode_pins_nothing(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_OFFLOAD_PIN", "none")
        monkeypatch.setattr(oe, "register_host_params", lambda m: pytest.fail("registered"))
        oe.pin_host_params(torch.nn.Linear(2, 2))()


@pytest.fixture(scope="module")
def fsdp_world():
    offload_cpu.init_world()
    yield
    offload_cpu.destroy_world()


def _sharded_tiny():
    return oe.shard_for_cpu_offload(offload_cpu.tiny_llama(), mesh=offload_cpu.cpu_mesh())


def _loop(model, **kw):
    return oe._train_loop(
        model, offload_cpu.ToyTokenizer(), offload_cpu.toy_dataset(), steps=3, batch_size=2,
        gradient_accumulation=1, learning_rate=1e-3, max_seq_length=32, warmup_steps=0,
        lr_scheduler_type="constant", weight_decay=0.05, rng=random.Random(0),
        device=torch.device("cpu"), on_step=None, seed=5, **kw,
    )


@pytest.mark.serial
class TestMoveToArenaOnFsdp2:
    def test_every_shard_becomes_a_view_of_one_slab_with_the_same_values(self, fsdp_world):
        model = _sharded_tiny()
        before = {n: p.to_local().clone() for n, p in model.named_parameters()}
        units = oe._fsdp_units(model)
        old_flat = [weakref.ref(fp._sharded_param_data) for _m, g in units for fp in g.fsdp_params]
        old_local = [weakref.ref(fp.sharded_param._local_tensor) for _m, g in units for fp in g.fsdp_params]
        arena = oe.move_params_to_arena(model, pin=False)
        lo = arena.slab.data_ptr()
        hi = lo + arena.nbytes
        n_shards = 0
        for _m, group in units:
            for fp in group.fsdp_params:
                flat, local = fp._sharded_param_data, fp.sharded_param._local_tensor
                assert lo <= flat.data_ptr() < hi and lo <= local.data_ptr() < hi
                assert flat.data_ptr() == local.data_ptr() and flat.data_ptr() % PAGE == 0
                n_shards += 1
        assert n_shards == len(before)
        for n, p in model.named_parameters():
            assert torch.equal(p.to_local(), before[n]), n
        # the slab is the aligned shards and nothing else
        sizes = [p.numel() * p.element_size() for p in before.values()]
        assert arena.nbytes == oe.HostArena.planned_bytes(sizes)
        # nothing in the engine still holds the old storages
        assert all(r() is None for r in old_flat + old_local)

    @pytest.mark.parametrize("fused", [False, True])
    def test_training_on_the_arena_is_bit_equal_to_training_without_it(self, fsdp_world, fused):
        plain = _loop(_sharded_tiny(), fused=fused)
        model = _sharded_tiny()
        oe.move_params_to_arena(model, pin=False)
        on_arena = _loop(model, fused=fused)
        assert on_arena["losses"] == plain["losses"]
        for (name, a), (_, b) in zip(plain["model"].named_parameters(), on_arena["model"].named_parameters()):
            assert torch.equal(a.to_local(), b.to_local()), name

    def test_the_gather_reads_the_arena_not_a_stale_copy(self, fsdp_world):
        """A write through the optimizer's view must be what the next forward sees."""
        model = _sharded_tiny()
        oe.move_params_to_arena(model, pin=False)
        ids = torch.randint(2, 60, (2, 8))
        shards = [p.to_local() for p in model.parameters()]  # before forward: the root stays unsharded after it
        loss_before = model(input_ids=ids, labels=ids).loss.item()
        with torch.no_grad():
            for shard in shards:
                shard.add_(0.5)
        loss_after = model(input_ids=ids, labels=ids).loss.item()
        assert loss_after != loss_before

    def test_a_plain_model_is_refused_untouched(self):
        model = offload_cpu.tiny_llama()
        before = [p.detach().clone() for p in model.parameters()]
        with pytest.raises(ValueError, match="no FSDP2 units"):
            oe.move_params_to_arena(model, pin=False)
        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))

    def test_a_shard_that_does_not_alias_its_flat_copy_is_refused_before_any_change(self, fsdp_world):
        model = _sharded_tiny()
        units = oe._fsdp_units(model)
        fp = units[-1][1].fsdp_params[0]
        original = fp._sharded_param_data
        fp._sharded_param_data = original.clone()  # breaks the aliasing the engine relies on
        first = units[0][1].fsdp_params[0]
        first_flat = first._sharded_param_data
        with pytest.raises(ValueError, match="alias"):
            oe.move_params_to_arena(model, pin=False)
        assert first._sharded_param_data is first_flat  # an earlier shard was not moved
        fp._sharded_param_data = original
