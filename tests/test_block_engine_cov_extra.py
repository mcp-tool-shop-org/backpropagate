"""Edge-case coverage for ``backpropagate.block_engine`` on tiny CPU models.

Hand-built module trees exercise the partitioner's fallbacks (embedding
accessors that raise, tying detected from shared storage, frozen embeddings,
no trainable block); real tiny Llama models exercise the optimizer's guard
rails, the 16-bit write-back paths, resume-config drift warnings and the VRAM
estimate. Nothing is mocked. CPU only.
"""

from __future__ import annotations

import logging
import types

import pytest
import torch
from torch import nn

from backpropagate import block_engine as be
from backpropagate.exceptions import TrainingError
from tests.helpers.tiny_models import tiny_llama

LOGGER = "backpropagate.block_engine"


class Handbuilt(nn.Module):
    """embed -> layers.{0,1} -> norm, with no get_*_embeddings accessors at all."""

    def __init__(self, *, tie: bool = False):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        if tie:
            self.lm_head.weight = self.embed.weight


class RaisingAccessors(Handbuilt):
    def get_input_embeddings(self):
        raise NotImplementedError("no input embeddings")

    def get_output_embeddings(self):
        raise AttributeError("no output embeddings")


class TestStochasticRoundGuard:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
    def test_requires_float32_input(self, dtype):
        with pytest.raises(TypeError, match="expects float32"):
            be.stochastic_round_to_bf16(torch.zeros(4, dtype=dtype))

    def test_exactly_representable_values_round_trip(self):
        x = torch.tensor([1.0, -2.0, 0.5, 0.0])
        assert torch.equal(be.stochastic_round_to_bf16(x).float(), x)


class TestPartitionFallbacks:
    def test_untied_handbuilt_model_gets_embed_layers_and_head_blocks(self):
        part = be.partition_model(RaisingAccessors())
        assert part.names == ["embed", "layers.0", "layers.1", "head"]
        assert part.tied is False
        assert part.blocks[0].param_names == ("embed.weight",)
        assert set(part.blocks[-1].param_names) == {"norm.weight", "norm.bias", "lm_head.weight"}

    def test_tying_is_detected_from_shared_storage_without_accessors(self):
        part = be.partition_model(Handbuilt(tie=True))
        assert part.tied is True
        # the shared matrix lands in exactly one block; the final norm joins it
        assert part.names == ["embed+head", "layers.0", "layers.1"]
        embed_block = part.blocks[0]
        assert "embed.weight" in embed_block.param_names and "norm.weight" in embed_block.param_names
        all_names = [n for b in part.blocks for n in b.param_names]
        assert len(all_names) == len(set(all_names))

    def test_frozen_embeddings_when_tied(self):
        part = be.partition_model(Handbuilt(tie=True), include_embeddings=False)
        assert part.names == ["layers.0", "layers.1"]
        assert set(part.frozen_param_names) == {"embed.weight", "norm.weight", "norm.bias"}

    def test_frozen_embeddings_when_untied(self):
        part = be.partition_model(Handbuilt(tie=False), include_embeddings=False)
        assert part.names == ["layers.0", "layers.1"]
        assert "embed.weight" in part.frozen_param_names and "lm_head.weight" in part.frozen_param_names

    def test_nothing_trainable_is_an_error(self):
        flat = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])  # the layer list IS the model
        with pytest.raises(TrainingError, match="no trainable parameters") as exc:
            be.partition_model(flat, include_embeddings=False)
        assert exc.value.code == "RUNTIME_TRAINING_FAILED"

    def test_model_without_a_repeated_layer_list_is_rejected(self):
        with pytest.raises(TrainingError, match="could not find a repeated transformer-layer"):
            be.partition_model(nn.Sequential(nn.Linear(2, 2), nn.ReLU()))


class TestOptimizerGuardRails:
    def _opt(self, **kw):
        return be.BlockCoordinateOptimizer(tiny_llama(layers=2), lr=1e-3, **kw)

    def test_block_params_match_the_partition(self):
        opt = self._opt()
        for i, blk in enumerate(opt.partition.blocks):
            names = dict(opt.model.named_parameters())
            assert [id(p) for p in opt.block_params(i)] == [id(names[n]) for n in blk.param_names]

    def test_params_by_block_pairs_names_with_parameters(self):
        opt = self._opt()
        pairs = list(be.params_by_block(opt))
        assert [n for n, _ in pairs] == opt.partition.names
        assert all(isinstance(p, nn.Parameter) for _, ps in pairs for p in ps)

    def test_activating_while_a_block_is_active_is_a_programming_error(self):
        opt = self._opt()
        with pytest.raises(RuntimeError, match="called while a block is active"):
            opt._activate(0)

    def test_finalize_is_idempotent_and_deactivate_without_active_is_a_noop(self):
        opt = self._opt()
        opt.finalize()
        assert opt.active_block_index is None
        opt.finalize()  # second call: nothing to do
        opt._deactivate()  # nothing active: returns without touching state
        assert all(p.requires_grad for p in opt.model.parameters())  # original flags restored

    def test_step_after_finalize_is_refused(self):
        opt = self._opt()
        opt.finalize()
        with pytest.raises(TrainingError, match=r"step\(\) after finalize") as exc:
            opt.step()
        assert exc.value.code == "RUNTIME_TRAINING_FAILED"


class TestSixteenBitWriteback:
    def test_stochastic_writeback_falls_back_to_nearest_for_fp16_storage(self, caplog):
        model = tiny_llama(layers=2).half()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt = be.BlockCoordinateOptimizer(model, lr=1e-3, block_writeback="stochastic")
        assert next(p for p in model.parameters() if p.requires_grad).dtype == torch.float32  # upcast while active
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            opt.finalize()
        assert any("bf16 storage only; rounding" in r.getMessage() for r in caplog.records)
        assert all(p.dtype == torch.float16 for p in model.parameters())
        assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())  # untouched: no step taken

    def test_update_without_upcast_runs_in_the_storage_dtype(self):
        model = tiny_llama(layers=2, dtype=torch.bfloat16)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt = be.BlockCoordinateOptimizer(model, lr=0.05, upcast=False, switch_block_every=1)
        ids = torch.randint(3, 64, (2, 8), generator=torch.Generator().manual_seed(0))
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        active = opt.active_block
        opt.step()
        opt.finalize()
        assert all(p.dtype == torch.bfloat16 for p in model.parameters())
        changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
        assert changed and changed <= set(active.param_names)


class TestResumeConfigDrift:
    def test_differing_schedule_settings_are_warned_about_per_key(self, caplog):
        a = be.BlockCoordinateOptimizer(tiny_llama(layers=2), lr=1e-3, switch_block_every=3, seed=1)
        b = be.BlockCoordinateOptimizer(
            tiny_llama(layers=2), lr=1e-3, switch_block_every=7, seed=2, block_order="ascending"
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            b.load_state_dict(a.state_dict())
        msgs = [r.getMessage() for r in caplog.records]
        assert any("switch_block_every=3 differs from this run's 7" in m for m in msgs)
        assert any("seed=1 differs from this run's 2" in m for m in msgs)
        assert any("block_order=" in m for m in msgs)
        assert not any("block_writeback" in m for m in msgs)
        assert b.switch_block_every == 7  # this run's setting wins from here on

    def test_identical_settings_resume_silently(self, caplog):
        a = be.BlockCoordinateOptimizer(tiny_llama(layers=2), lr=1e-3, switch_block_every=3)
        b = be.BlockCoordinateOptimizer(tiny_llama(layers=2), lr=1e-3, switch_block_every=3)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            b.load_state_dict(a.state_dict())
        assert caplog.records == []


class TestInputGradHook:
    def test_no_hook_attribute(self):
        assert be.drop_input_require_grads_hook(object()) is False
        assert be.drop_input_require_grads_hook(types.SimpleNamespace(_require_grads_hook=None)) is False

    def test_hook_without_a_disable_method_cannot_be_dropped(self):
        assert be.drop_input_require_grads_hook(types.SimpleNamespace(_require_grads_hook=object())) is False

    def test_hook_is_dropped_through_the_model_api(self):
        calls = []
        model = types.SimpleNamespace(
            _require_grads_hook=object(), disable_input_require_grads=lambda: calls.append(1)
        )
        assert be.drop_input_require_grads_hook(model) is True
        assert calls == [1]


class TestVramEstimate:
    def test_unknown_architecture_zeroes_activation_and_logit_terms_with_a_note(self):
        fit = be.estimate_block_engine_vram(7.6, 30, seq_len=2048, batch_size=1)
        assert fit.activations_gb == 0.0 and fit.logits_gb == 0.0
        assert any("activations not projected" in n for n in fit.notes)
        assert fit.status == "projection — not measured"
        assert fit.weights_gb == pytest.approx(2 * 7.6, abs=2e-3)
        assert fit.active_block_gb == pytest.approx(14.0 * 7.6 / 30, abs=2e-3)  # figures are rounded to 3 dp
        assert fit.paper_formula_gb == pytest.approx(2 * 7.6 + 16 * 7.6 / 30, abs=2e-3)

    def test_known_architecture_projects_activations_and_logits(self):
        common = {"hidden_size": 3584, "num_layers": 28, "vocab_size": 152_064}
        ckpt = be.estimate_block_engine_vram(7.6, 30, 4096, 1, max_block_billions=0.545, **common)
        plain = be.estimate_block_engine_vram(
            7.6, 30, 4096, 1, max_block_billions=0.545, gradient_checkpointing=False, **common
        )
        assert ckpt.active_block_gb == pytest.approx(14.0 * 0.545, abs=2e-3)
        assert 0 < ckpt.activations_gb < plain.activations_gb  # checkpointing saves activation memory
        assert ckpt.logits_gb == pytest.approx(4096 * 152_064 * 10.0 / 1e9, abs=2e-3)
        assert ckpt.total_gb > ckpt.weights_gb + ckpt.active_block_gb
        assert set(ckpt.as_dict()) >= {"total_gb", "weights_gb", "notes", "status"}

    def test_overhead_fraction_scales_the_total(self):
        base = be.estimate_block_engine_vram(1.0, 10, 512, 1, overhead_fraction=0.0)
        padded = be.estimate_block_engine_vram(1.0, 10, 512, 1, overhead_fraction=0.5)
        assert padded.total_gb == pytest.approx(base.total_gb * 1.5, abs=2e-3)
        assert be.estimate_block_engine_vram(1.0, 0, 512, 1).num_blocks == 1  # a non-positive block count clamps to 1
