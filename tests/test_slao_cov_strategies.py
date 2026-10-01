"""Coverage tests for the pure SLAO helpers and the per-tensor merge strategies.

Everything here runs on real tiny CPU tensors; nothing is mocked. Where a test
needs a *second device* (the device-normalisation branches) it uses torch's
``meta`` device, which is a real torch backend that every machine has, instead
of CUDA. Expected values are hand-computed from the formulas in
``backpropagate/slao.py``:

- B matrices: ``B_merged = B_acc + lambda * (B_new - B_acc)`` (EMA)
- A matrices: hard-replaced by the new tensor
- lambda(i) = 1/sqrt(i), clamped to ``[min_scale, 1.0]``
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from backpropagate.exceptions import InvalidSettingError, SLAOMergeError
from backpropagate.slao import (
    MergeStrategyConfig,
    SLAOConfig,
    apply_merge_strategy,
    compute_task_similarity,
    merge_strategy_dare,
    merge_strategy_linear,
    merge_strategy_qiao_mahdavi,
    merge_strategy_ties,
    orthogonal_init_A,
    time_aware_scale,
)

A_KEY = "base.layers.0.q.lora_A.default.weight"
B_KEY = "base.layers.0.q.lora_B.default.weight"


def _t(*vals: float) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.float32)


def _qm(acc, new, *, run_index=4, slao=None):
    return merge_strategy_qiao_mahdavi(
        acc,
        new,
        run_index=run_index,
        config=MergeStrategyConfig(),
        slao_config=slao or SLAOConfig(),
    )


# =============================================================================
# time_aware_scale: callable schedule failure modes
# =============================================================================


class TestTimeAwareScaleCallableErrors:
    def test_callable_that_raises_becomes_invalid_setting_error(self):
        """A raising custom schedule is wrapped, with the original chained."""

        def boom(_i: int) -> float:
            raise ValueError("schedule exploded")

        with pytest.raises(InvalidSettingError) as exc_info:
            time_aware_scale(3, boom)

        err = exc_info.value
        assert err.code == "CONFIG_INVALID_SETTING"
        assert err.setting_name == "scaling_type"
        assert isinstance(err.__cause__, ValueError)
        # the suggestion names the underlying failure so the operator can fix it
        assert "ValueError" in err.suggestion
        assert "schedule exploded" in err.suggestion

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_callable_returning_non_finite_is_rejected(self, bad):
        """NaN/inf from a custom schedule must not reach the EMA weight."""
        with pytest.raises(InvalidSettingError) as exc_info:
            time_aware_scale(2, lambda _i: bad)

        assert exc_info.value.code == "CONFIG_INVALID_SETTING"
        assert "run_index=2" in exc_info.value.suggestion


# =============================================================================
# orthogonal_init_A failure + compute_task_similarity edge cases
# =============================================================================


class TestOrthogonalInitFailure:
    @pytest.mark.filterwarnings("ignore:Tensor.T is deprecated")
    def test_qr_failure_is_wrapped_in_slao_merge_error(self):
        """A 0-dim tensor makes the real ``torch.linalg.qr`` raise RuntimeError."""
        with pytest.raises(SLAOMergeError) as exc_info:
            orthogonal_init_A(torch.tensor(1.0))

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "QR decomposition error" in str(exc_info.value)
        assert "ill-conditioned" in exc_info.value.suggestion


class TestComputeTaskSimilarityEdges:
    def test_non_tensor_b_entries_are_skipped(self):
        """Only tensor ``.lora_B.`` pairs contribute; non-tensors are ignored."""
        s1 = {B_KEY: _t(1.0, 0.0), "x.lora_B.meta": "not a tensor"}
        s2 = {B_KEY: _t(1.0, 1.0), "x.lora_B.meta": "also not"}
        # cos([1,0],[1,1]) = 1/sqrt(2)
        assert compute_task_similarity(s1, s2) == pytest.approx(1 / math.sqrt(2), abs=1e-6)

    def test_only_non_tensor_b_entries_gives_neutral_zero(self):
        s1 = {"x.lora_B.meta": "a"}
        s2 = {"x.lora_B.meta": "b"}
        assert compute_task_similarity(s1, s2) == 0.0


# =============================================================================
# merge_strategy_qiao_mahdavi (standalone pure-function path)
# =============================================================================


class TestQiaoMahdaviStandalone:
    def test_adaptive_scaling_positive_similarity_boosts_weight(self):
        """run 4 => base 0.5; identical-direction B => cos=1 => multiplier 1.5 => 0.75.

        merged B = 1 + 0.75 * (2 - 1) = 1.75 on the first element, 0 on the second.
        """
        acc = {B_KEY: _t(1.0, 0.0)}
        new = {B_KEY: _t(2.0, 0.0)}
        out = _qm(acc, new, slao=SLAOConfig(use_adaptive_scaling=True))
        assert torch.allclose(out[B_KEY], _t(1.75, 0.0))

    def test_adaptive_scaling_opposite_similarity_shrinks_weight(self):
        """cos=-1 => multiplier 0.5 => 0.5 * 0.5 = 0.25. merged = 1 + 0.25 * (-1 - 1) = 0.5."""
        acc = {B_KEY: _t(1.0, 0.0)}
        new = {B_KEY: _t(-1.0, 0.0)}
        out = _qm(acc, new, slao=SLAOConfig(use_adaptive_scaling=True))
        assert torch.allclose(out[B_KEY], _t(0.5, 0.0))

    def test_adaptive_weight_is_clamped_to_one(self):
        """run 2 base 0.7071 * 1.5 > 1.0 must clamp to 1.0, i.e. merged == new, never past it."""
        acc = {B_KEY: _t(1.0, 0.0)}
        new = {B_KEY: _t(3.0, 0.0)}
        out = _qm(acc, new, run_index=2, slao=SLAOConfig(use_adaptive_scaling=True))
        assert torch.allclose(out[B_KEY], _t(3.0, 0.0))

    def test_layer_scaling_applies_per_layer_weight(self):
        """4 layers detected; layer 0 is 'early' (0.3), layer 3 is 'late' (0.7).

        run 4 base 0.5 -> effective 0.15 (>= min 0.1) and 0.35; acc B=0, new B=1.
        """
        k0 = "base.layers.0.q.lora_B.default.weight"
        k3 = "base.layers.3.q.lora_B.default.weight"
        acc = {k0: _t(0.0), k3: _t(0.0)}
        new = {k0: _t(1.0), k3: _t(1.0)}
        out = _qm(acc, new, slao=SLAOConfig(use_layer_scaling=True))
        assert out[k0].item() == pytest.approx(0.15, abs=1e-6)
        assert out[k3].item() == pytest.approx(0.35, abs=1e-6)

    def test_layer_scaling_respects_min_scale_floor(self):
        """0.5 * 0.1 = 0.05 is below min_scale 0.1, so the floor applies."""
        k0 = "base.layers.0.q.lora_B.default.weight"
        k3 = "base.layers.3.q.lora_B.default.weight"
        acc = {k0: _t(0.0), k3: _t(0.0)}
        new = {k0: _t(1.0), k3: _t(1.0)}
        out = _qm(acc, new, slao=SLAOConfig(use_layer_scaling=True, layer_scale_early=0.1))
        assert out[k0].item() == pytest.approx(0.1, abs=1e-6)

    def test_non_tensor_new_values_are_skipped_and_new_keys_cloned(self):
        """Non-tensor new entries are not merged; keys absent from the
        accumulator are cloned (not aliased); accumulator-only keys survive."""
        brand_new = _t(5.0, 6.0)
        acc = {A_KEY: _t(1.0), "acc.only": _t(9.0)}
        new = {A_KEY: _t(2.0), "note": "metadata", "fresh.lora_B.w": brand_new}
        out = _qm(acc, new)
        assert torch.equal(out[A_KEY], _t(2.0))  # A hard-replaced
        assert "note" not in out
        assert torch.equal(out["fresh.lora_B.w"], brand_new)
        assert out["fresh.lora_B.w"] is not brand_new
        assert torch.equal(out["acc.only"], _t(9.0))

    def test_generic_tensor_uses_ema_weight(self):
        """A tensor that is neither lora_A nor lora_B takes the same EMA blend.

        run 4 => 0.5; 0 + 0.5 * (4 - 0) = 2.
        """
        out = _qm({"misc.scale": _t(0.0)}, {"misc.scale": _t(4.0)})
        assert torch.allclose(out["misc.scale"], _t(2.0))

    def test_accumulator_on_other_device_is_normalised(self):
        """Real second backend: the ``meta`` device. The accumulator tensor is
        moved onto the new tensor's device before the arithmetic."""
        acc = {"misc.scale": _t(0.0, 0.0)}
        new = {"misc.scale": torch.zeros(2, device="meta")}
        out = _qm(acc, new)
        assert out["misc.scale"].device.type == "meta"
        assert tuple(out["misc.scale"].shape) == (2,)


# =============================================================================
# linear / ties / dare edge branches
# =============================================================================


class TestLinearEdges:
    def test_non_tensor_and_one_sided_keys(self):
        """Non-tensor ``new`` values are skipped by the blend; one-sided tensors
        are cloned through; the common tensor blends with the fixed weight."""
        cfg = MergeStrategyConfig(strategy="linear", linear_weight=0.25)
        acc = {"c.w": _t(0.0), "acc.only": _t(7.0), "tag": "acc-tag"}
        new = {"c.w": _t(4.0), "new.only": _t(8.0), "tag": "new-tag"}
        out = merge_strategy_linear(acc, new, run_index=9, config=cfg, slao_config=SLAOConfig())
        assert torch.allclose(out["c.w"], _t(1.0))  # 0 + 0.25 * 4
        assert torch.equal(out["acc.only"], _t(7.0))
        assert torch.equal(out["new.only"], _t(8.0))

    def test_accumulator_only_non_tensor_is_cloned_through(self):
        cfg = MergeStrategyConfig(strategy="linear", linear_weight=0.5)
        out = merge_strategy_linear(
            {"c.w": _t(0.0), "tag": "kept"},
            {"c.w": _t(2.0)},
            run_index=2,
            config=cfg,
            slao_config=SLAOConfig(),
        )
        assert out["tag"] == "kept"

    def test_device_mismatch_moves_accumulator(self):
        cfg = MergeStrategyConfig(strategy="linear", linear_weight=0.5)
        out = merge_strategy_linear(
            {"c.w": _t(0.0, 0.0)},
            {"c.w": torch.zeros(2, device="meta")},
            run_index=2,
            config=cfg,
            slao_config=SLAOConfig(),
        )
        assert out["c.w"].device.type == "meta"

    def test_non_tensor_new_value_present_in_both_is_not_blended(self):
        """A key whose *new* value is a non-tensor is not blended (no crash)
        and the tensor keys still blend correctly."""
        cfg = MergeStrategyConfig(strategy="linear", linear_weight=0.5)
        out = merge_strategy_linear(
            {"c.w": _t(0.0), "tag": _t(1.0)},
            {"c.w": _t(2.0), "tag": "str"},
            run_index=2,
            config=cfg,
            slao_config=SLAOConfig(),
        )
        assert torch.allclose(out["c.w"], _t(1.0))


class TestTiesEdges:
    def _ties(self, acc, new, trim=0.2):
        return merge_strategy_ties(
            acc,
            new,
            run_index=2,
            config=MergeStrategyConfig(strategy="ties", trim_threshold=trim),
            slao_config=SLAOConfig(),
        )

    def test_empty_tensor_survives_trim(self):
        """numel()==0 takes the early-return path; result keeps the empty shape."""
        out = self._ties({"e.w": torch.zeros(0)}, {"e.w": torch.zeros(0)})
        assert tuple(out["e.w"].shape) == (0,)

    def test_non_tensor_entries_are_ignored_and_one_sided_cloned(self):
        out = self._ties(
            {"c.w": _t(1.0, -1.0), "tag": "x", "acc.only": _t(3.0)},
            {"c.w": _t(1.0, 1.0), "tag": "y", "new.only": _t(4.0)},
            trim=0.0,
        )
        # trim disabled; element 0: both +1 agree -> mean 1. element 1: (-1)+(1)=0
        # => gamma 0 => nobody agrees => 0.
        assert torch.allclose(out["c.w"], _t(1.0, 0.0))
        assert torch.equal(out["acc.only"], _t(3.0))
        assert torch.equal(out["new.only"], _t(4.0))

    def test_accumulator_non_tensor_for_common_key_is_skipped(self):
        out = self._ties({"c.w": "not-a-tensor"}, {"c.w": _t(1.0)})
        # the strategy must not crash and must not fabricate a merged tensor
        assert not (isinstance(out.get("c.w"), torch.Tensor))

    def test_device_mismatch_moves_accumulator(self):
        out = self._ties(
            {"c.w": _t(1.0, 2.0, 3.0)}, {"c.w": torch.ones(3, device="meta")}, trim=0.0
        )
        assert out["c.w"].device.type == "meta"


class TestDareEdges:
    def _dare(self, acc, new, **cfg):
        return merge_strategy_dare(
            acc,
            new,
            run_index=2,
            config=MergeStrategyConfig(strategy="dare", **cfg),
            slao_config=SLAOConfig(),
        )

    def test_non_tensor_entries_ignored_and_one_sided_cloned(self):
        out = self._dare(
            {"c.w": _t(0.0, 0.0), "tag": "x", "acc.only": _t(5.0)},
            {"c.w": _t(2.0, 2.0), "tag": "y", "new.only": _t(6.0)},
            drop_rate=0.0,
        )
        # drop_rate 0 => replace-on-increment => merged == new
        assert torch.allclose(out["c.w"], _t(2.0, 2.0))
        assert torch.equal(out["acc.only"], _t(5.0))
        assert torch.equal(out["new.only"], _t(6.0))

    def test_accumulator_non_tensor_for_common_key_is_skipped(self):
        out = self._dare({"c.w": "no"}, {"c.w": _t(1.0)}, drop_rate=0.5)
        assert not isinstance(out.get("c.w"), torch.Tensor)

    def test_device_mismatch_uses_cpu_draw_then_moves_mask(self):
        """The Bernoulli mask is drawn on CPU, then moved onto the delta's device."""
        out = self._dare(
            {"c.w": _t(0.0, 0.0, 0.0, 0.0)},
            {"c.w": torch.ones(4, device="meta")},
            drop_rate=0.5,
            dare_seed=3,
        )
        assert out["c.w"].device.type == "meta"
        assert tuple(out["c.w"].shape) == (4,)

    def test_seeded_draw_survivors_are_rescaled(self):
        """Every surviving entry is acc + delta / keep_prob; every dropped entry
        equals acc. Reconstruct the expected mask from the same local generator."""
        gen = torch.Generator()
        gen.manual_seed(11)
        rand = torch.rand((6,), generator=gen)
        keep = (rand < 0.5).to(torch.float32)
        acc = {"c.w": torch.zeros(6)}
        new = {"c.w": torch.ones(6)}
        out = self._dare(acc, new, drop_rate=0.5, dare_seed=11)
        assert torch.allclose(out["c.w"], keep * 1.0 / 0.5)


# =============================================================================
# apply_merge_strategy default slao_config
# =============================================================================


class TestApplyMergeStrategyDefaults:
    def test_slao_config_defaults_when_omitted(self):
        """Without a slao_config the dispatcher uses a fresh SLAOConfig (sqrt, 0.1).

        run 4 => lambda 0.5: B = 0 + 0.5 * (2 - 0) = 1.0
        """
        out = apply_merge_strategy(
            {B_KEY: _t(0.0)},
            {B_KEY: _t(2.0)},
            run_index=4,
            config=MergeStrategyConfig(strategy="qiao_mahdavi"),
        )
        assert torch.allclose(out[B_KEY], _t(1.0))
