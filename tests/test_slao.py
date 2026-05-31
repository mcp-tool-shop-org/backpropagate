"""
Tests for SLAO (Single LoRA via Asymmetric Merging) module.

Tests cover:
- Time-aware scaling function
- Orthogonal initialization via QR decomposition
- A/B matrix merging logic
- SLAOMerger class functionality
- Save/load operations
- Phase 4 features (adaptive scaling, layer scaling, task similarity)
"""

import math
import tempfile
from pathlib import Path

import pytest

# Import torch conditionally for environments without GPU
torch = pytest.importorskip("torch")

from backpropagate.exceptions import SLAOCheckpointError
from backpropagate.slao import (
    MergeResult,
    SLAOConfig,
    SLAOMerger,
    adaptive_scale,
    compute_task_similarity,
    estimate_total_layers,
    get_layer_scale,
    merge_A_matrices,
    merge_B_matrices,
    merge_lora_weights,
    orthogonal_init_A,
    time_aware_scale,
)


class TestSLAOConfigDefaults:
    """Tests for SLAOConfig default values - catches mutations to defaults."""

    def test_use_time_aware_scaling_default_true(self):
        """use_time_aware_scaling should default to True."""
        config = SLAOConfig()
        assert config.use_time_aware_scaling is True, "use_time_aware_scaling must default to True"

    def test_use_orthogonal_init_default_true(self):
        """use_orthogonal_init should default to True."""
        config = SLAOConfig()
        assert config.use_orthogonal_init is True, "use_orthogonal_init must default to True"

    def test_scaling_type_default_sqrt(self):
        """scaling_type should default to 'sqrt'."""
        config = SLAOConfig()
        assert config.scaling_type == "sqrt", "scaling_type must default to 'sqrt'"

    def test_min_scale_default_0_1(self):
        """min_scale should default to 0.1."""
        config = SLAOConfig()
        assert config.min_scale == 0.1, "min_scale must default to 0.1"

    def test_normalize_after_merge_default_false(self):
        """normalize_after_merge should default to False."""
        config = SLAOConfig()
        assert config.normalize_after_merge is False, "normalize_after_merge must default to False"

    def test_save_merge_history_default_true(self):
        """save_merge_history should default to True."""
        config = SLAOConfig()
        assert config.save_merge_history is True, "save_merge_history must default to True"

    def test_use_adaptive_scaling_default_false(self):
        """use_adaptive_scaling should default to False."""
        config = SLAOConfig()
        assert config.use_adaptive_scaling is False, "use_adaptive_scaling must default to False"

    def test_adaptive_scale_range_default(self):
        """adaptive_scale_range should default to (0.5, 1.5)."""
        config = SLAOConfig()
        assert config.adaptive_scale_range == (0.5, 1.5), "adaptive_scale_range must default to (0.5, 1.5)"
        assert config.adaptive_scale_range[0] == 0.5, "adaptive_scale_range min must be 0.5"
        assert config.adaptive_scale_range[1] == 1.5, "adaptive_scale_range max must be 1.5"

    def test_use_layer_scaling_default_false(self):
        """use_layer_scaling should default to False."""
        config = SLAOConfig()
        assert config.use_layer_scaling is False, "use_layer_scaling must default to False"

    def test_layer_scale_early_default_0_3(self):
        """layer_scale_early should default to 0.3."""
        config = SLAOConfig()
        assert config.layer_scale_early == 0.3, "layer_scale_early must default to 0.3"

    def test_layer_scale_middle_default_0_5(self):
        """layer_scale_middle should default to 0.5."""
        config = SLAOConfig()
        assert config.layer_scale_middle == 0.5, "layer_scale_middle must default to 0.5"

    def test_layer_scale_late_default_0_7(self):
        """layer_scale_late should default to 0.7."""
        config = SLAOConfig()
        assert config.layer_scale_late == 0.7, "layer_scale_late must default to 0.7"


class TestMergeResultDefaults:
    """Tests for MergeResult default values."""

    def test_optional_fields_default_to_none(self):
        """Optional fields should default to None."""
        result = MergeResult(
            run_index=1,
            scale_factor=1.0,
            a_matrices_merged=2,
            b_matrices_merged=2,
            total_params_merged=1000,
            merge_time_seconds=0.5,
        )
        assert result.a_norm_before is None, "a_norm_before must default to None"
        assert result.a_norm_after is None, "a_norm_after must default to None"
        assert result.b_norm_before is None, "b_norm_before must default to None"
        assert result.b_norm_after is None, "b_norm_after must default to None"


class TestTimeAwareScale:
    """Tests for the time_aware_scale function."""

    def test_scale_run_1(self):
        """First run should have scale = 1.0."""
        scale = time_aware_scale(1)
        assert scale == 1.0, "Run 1 scale must be exactly 1.0"

    def test_scale_run_2(self):
        """Second run should have scale = 1/sqrt(2) ≈ 0.707."""
        scale = time_aware_scale(2)
        assert abs(scale - 1 / math.sqrt(2)) < 1e-6

    def test_scale_run_4(self):
        """Fourth run should have scale = 1/sqrt(4) = 0.5."""
        scale = time_aware_scale(4)
        assert scale == 0.5

    def test_scale_decreases_over_runs(self):
        """Scale should decrease as run index increases."""
        scales = [time_aware_scale(i) for i in range(1, 11)]
        for i in range(len(scales) - 1):
            assert scales[i] > scales[i + 1]

    def test_scale_min_bound(self):
        """Scale should respect minimum bound."""
        scale = time_aware_scale(1000, min_scale=0.2)
        assert scale >= 0.2

    def test_scale_linear_type(self):
        """Linear scaling should be 1/i."""
        scale = time_aware_scale(4, scaling_type="linear")
        assert scale == 0.25, "Linear scale at run 4 must be 0.25"
        # Additional linear tests
        assert time_aware_scale(1, scaling_type="linear") == 1.0, "Linear scale at run 1 must be 1.0"
        assert time_aware_scale(2, scaling_type="linear") == 0.5, "Linear scale at run 2 must be 0.5"
        assert time_aware_scale(5, scaling_type="linear") == 0.2, "Linear scale at run 5 must be 0.2"

    def test_scale_log_type(self):
        """Log scaling should be 1/log(i+1)."""
        # Run 1: raw 1/log(2) ≈ 1.443 is CLAMPED to 1.0 (CONTINUAL-A-001 — an
        # EMA weight > 1.0 over-extrapolates past the new adapter and breaks
        # the anti-catastrophic-forgetting invariant the merger exists to hold).
        scale_1 = time_aware_scale(1, scaling_type="log")
        assert abs(scale_1 - 1.0) < 1e-6, "Log scale at run 1 must be clamped to 1.0"

        # Run 2: 1/log(3) ≈ 0.910
        scale_2 = time_aware_scale(2, scaling_type="log")
        expected_2 = 1.0 / math.log(3)
        assert abs(scale_2 - expected_2) < 1e-6, f"Log scale at run 2 must be {expected_2}"

        # Run 4: 1/log(5) ≈ 0.621
        scale_4 = time_aware_scale(4, scaling_type="log")
        expected_4 = 1.0 / math.log(5)
        assert abs(scale_4 - expected_4) < 1e-6, f"Log scale at run 4 must be {expected_4}"

        # Log should decay slower than sqrt
        sqrt_4 = time_aware_scale(4, scaling_type="sqrt")
        assert scale_4 > sqrt_4, "Log scaling should decay slower than sqrt"

    def test_scale_constant_type(self):
        """Constant scaling should always be 1.0."""
        for i in range(1, 10):
            scale = time_aware_scale(i, scaling_type="constant")
            assert scale == 1.0, f"Constant scale at run {i} must be 1.0"

    def test_scale_types_are_different(self):
        """Different scaling types should produce different results."""
        run_idx = 4
        sqrt_scale = time_aware_scale(run_idx, scaling_type="sqrt")
        linear_scale = time_aware_scale(run_idx, scaling_type="linear")
        log_scale = time_aware_scale(run_idx, scaling_type="log")
        constant_scale = time_aware_scale(run_idx, scaling_type="constant")

        # All should be different (except run 1)
        assert sqrt_scale != linear_scale, "sqrt and linear should differ at run 4"
        assert sqrt_scale != log_scale, "sqrt and log should differ at run 4"
        assert sqrt_scale != constant_scale, "sqrt and constant should differ at run 4"
        assert linear_scale != log_scale, "linear and log should differ at run 4"

    def test_scale_invalid_run_index(self):
        """Should raise error for run_index < 1."""
        from backpropagate.exceptions import InvalidSettingError
        with pytest.raises(InvalidSettingError, match="run_index"):
            time_aware_scale(0)
        with pytest.raises(InvalidSettingError, match="run_index"):
            time_aware_scale(-1)

    def test_scale_invalid_type(self):
        """Should raise error for unknown scaling type."""
        from backpropagate.exceptions import InvalidSettingError
        with pytest.raises(InvalidSettingError, match="scaling_type"):
            time_aware_scale(1, scaling_type="unknown")


class TestOrthogonalInitA:
    """Tests for orthogonal initialization of A matrices."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        A = torch.randn(16, 128)
        A_ortho = orthogonal_init_A(A)
        assert A_ortho.shape == A.shape

    def test_orthogonality(self):
        """A @ A^T should be close to identity."""
        A = torch.randn(16, 128)
        A_ortho = orthogonal_init_A(A)

        # A_ortho @ A_ortho^T should be identity
        result = A_ortho @ A_ortho.T
        identity = torch.eye(16)

        assert torch.allclose(result, identity, atol=1e-5)

    def test_different_ranks(self):
        """Should work with different LoRA ranks."""
        for r in [4, 8, 16, 32, 64]:
            A = torch.randn(r, 256)
            A_ortho = orthogonal_init_A(A)
            result = A_ortho @ A_ortho.T
            assert torch.allclose(result, torch.eye(r), atol=1e-5)

    def test_reproducibility(self):
        """Same input should give same output."""
        torch.manual_seed(42)
        A = torch.randn(16, 128)

        A_ortho1 = orthogonal_init_A(A.clone())
        A_ortho2 = orthogonal_init_A(A.clone())

        assert torch.allclose(A_ortho1, A_ortho2)

    def test_numerical_stability(self):
        """Should be stable with small values."""
        A = torch.randn(16, 128) * 0.01
        A_ortho = orthogonal_init_A(A)

        # Should still produce valid orthogonal matrix
        result = A_ortho @ A_ortho.T
        assert torch.allclose(result, torch.eye(16), atol=1e-4)


class TestMergeBMatrices:
    """Tests for B matrix merging with time-aware scaling."""

    def test_merge_with_scale_1(self):
        """With scale=1, result should be new matrix."""
        B_merged = torch.randn(256, 16)
        B_new = torch.randn(256, 16)

        result = merge_B_matrices(B_merged, B_new, scale=1.0)

        assert torch.allclose(result, B_new, atol=1e-6)

    def test_merge_with_scale_0(self):
        """With scale=0, result should be merged matrix."""
        B_merged = torch.randn(256, 16)
        B_new = torch.randn(256, 16)

        result = merge_B_matrices(B_merged, B_new, scale=0.0)

        assert torch.allclose(result, B_merged)

    def test_merge_with_scale_half(self):
        """With scale=0.5, result should be midpoint."""
        B_merged = torch.zeros(256, 16)
        B_new = torch.ones(256, 16)

        result = merge_B_matrices(B_merged, B_new, scale=0.5)

        expected = torch.ones(256, 16) * 0.5
        assert torch.allclose(result, expected)

    def test_merge_formula(self):
        """Verify the merge formula: B_merged + scale * (B_new - B_merged)."""
        B_merged = torch.randn(256, 16)
        B_new = torch.randn(256, 16)
        scale = 0.7

        result = merge_B_matrices(B_merged, B_new, scale)
        expected = B_merged + scale * (B_new - B_merged)

        assert torch.allclose(result, expected)


class TestMergeAMatrices:
    """Tests for A matrix merging (direct replacement)."""

    def test_direct_replacement(self):
        """A merge should be direct replacement."""
        A_new = torch.randn(16, 128)

        result = merge_A_matrices(A_new)

        assert torch.allclose(result, A_new)

    def test_returns_clone(self):
        """Should return a clone, not the same tensor."""
        A_new = torch.randn(16, 128)

        result = merge_A_matrices(A_new)

        # Modify original
        A_new[0, 0] = 999.0

        # Result should not be affected
        assert result[0, 0] != 999.0


class TestSLAOMerger:
    """Tests for the SLAOMerger class."""

    @pytest.fixture
    def sample_lora_state(self):
        """Create a sample LoRA state dict."""
        return {
            "layer1.lora_A.weight": torch.randn(16, 128),
            "layer1.lora_B.weight": torch.randn(256, 16),
            "layer2.lora_A.weight": torch.randn(16, 128),
            "layer2.lora_B.weight": torch.randn(256, 16),
        }

    def test_initialization(self):
        """SLAOMerger should initialize with default config."""
        merger = SLAOMerger()

        assert merger.config is not None
        assert merger.run_index == 0
        assert merger._merged_state is None

    def test_initialize_with_first_lora(self, sample_lora_state):
        """Should properly initialize with first LoRA."""
        merger = SLAOMerger()
        merger.initialize(sample_lora_state)

        assert merger.run_index == 1
        assert merger._merged_state is not None
        assert len(merger._merged_state) == len(sample_lora_state)

    def test_merge_returns_result(self, sample_lora_state):
        """Merge should return a MergeResult."""
        merger = SLAOMerger()
        merger.initialize(sample_lora_state)

        new_lora = {k: torch.randn_like(v) for k, v in sample_lora_state.items()}
        result = merger.merge(new_lora, run_index=2)

        assert isinstance(result, MergeResult)
        assert result.run_index == 2
        assert result.scale_factor == pytest.approx(1 / math.sqrt(2), abs=1e-6)
        assert result.a_matrices_merged == 2
        assert result.b_matrices_merged == 2

    def test_merge_increments_run_index(self, sample_lora_state):
        """Merge should increment run index."""
        merger = SLAOMerger()
        merger.initialize(sample_lora_state)

        assert merger.run_index == 1

        new_lora = {k: torch.randn_like(v) for k, v in sample_lora_state.items()}
        merger.merge(new_lora)

        assert merger.run_index == 2

    def test_get_init_weights(self, sample_lora_state):
        """Should return initialization weights for next run."""
        merger = SLAOMerger()
        merger.initialize(sample_lora_state)

        init_weights = merger.get_init_weights()

        assert init_weights is not None
        assert len(init_weights) == len(sample_lora_state)

        # A matrices should be orthogonally initialized
        for key, value in init_weights.items():
            if ".lora_A." in key:
                # Check orthogonality
                result = value @ value.T
                assert torch.allclose(result, torch.eye(16), atol=1e-5)

    def test_get_init_weights_before_init(self):
        """Should return None before initialization."""
        merger = SLAOMerger()
        assert merger.get_init_weights() is None

    def test_merge_history(self, sample_lora_state):
        """Should track merge history."""
        merger = SLAOMerger(SLAOConfig(save_merge_history=True))
        merger.initialize(sample_lora_state)

        for i in range(2, 6):
            new_lora = {k: torch.randn_like(v) for k, v in sample_lora_state.items()}
            merger.merge(new_lora, run_index=i)

        assert len(merger.merge_history) == 4
        assert [r.run_index for r in merger.merge_history] == [2, 3, 4, 5]

    def test_save_and_load(self, sample_lora_state):
        """Should save and load merger state."""
        merger = SLAOMerger()
        merger.initialize(sample_lora_state)

        new_lora = {k: torch.randn_like(v) for k, v in sample_lora_state.items()}
        merger.merge(new_lora, run_index=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "merger"
            merger.save(str(save_path))

            # Load into new merger
            new_merger = SLAOMerger()
            new_merger.load(str(save_path))

            assert new_merger.run_index == merger.run_index
            assert new_merger._merged_state is not None

    def test_config_options(self):
        """Should respect config options."""
        config = SLAOConfig(
            scaling_type="linear",
            min_scale=0.2,
            use_orthogonal_init=False,
        )
        merger = SLAOMerger(config)

        assert merger.config.scaling_type == "linear"
        assert merger.config.min_scale == 0.2
        assert merger.config.use_orthogonal_init is False

    def test_load_rejects_non_dict_state(self, tmp_path):
        """CORE-B-007: load() must reject a merged_lora.pt that does not
        deserialize to a dict.

        ``torch.load`` of a corrupt / wrong-kind file can return a bare
        tensor (or list) without raising. Pre-fix the merger silently
        carried that garbage as its accumulator and fed it forward through
        every subsequent merge. Now load() fails loud at the seam.
        """
        save_dir = tmp_path / "merger"
        save_dir.mkdir()
        # A bare tensor, not a state dict.
        torch.save(torch.randn(4, 4), save_dir / "merged_lora.pt")

        merger = SLAOMerger()
        with pytest.raises(SLAOCheckpointError) as exc_info:
            merger.load(str(save_dir))
        assert "state dict" in str(exc_info.value).lower()

    def test_load_rejects_zero_tensor_dict(self, tmp_path):
        """CORE-B-007: load() must reject a dict that contains no tensors.

        An empty/tensorless dict is structurally a "state dict" but carries
        nothing to merge — treating it as a valid accumulator silently
        zeroes out catastrophic-forgetting protection. Fail loud instead.
        """
        save_dir = tmp_path / "merger"
        save_dir.mkdir()
        # A dict with only non-tensor values.
        torch.save({"meta": "not a tensor", "count": 3}, save_dir / "merged_lora.pt")

        merger = SLAOMerger()
        with pytest.raises(SLAOCheckpointError) as exc_info:
            merger.load(str(save_dir))
        assert "zero tensors" in str(exc_info.value).lower()

    def test_load_accepts_valid_state_dict(self, sample_lora_state, tmp_path):
        """CORE-B-007 guard: a normal save/load round-trip still succeeds —
        the structural validation must not reject legitimate accumulators."""
        merger = SLAOMerger()
        merger.initialize(sample_lora_state)
        save_dir = tmp_path / "merger"
        merger.save(str(save_dir))

        loaded = SLAOMerger()
        loaded.load(str(save_dir))  # must not raise
        assert loaded._merged_state is not None

    def test_save_writes_version_from_constant(self, sample_lora_state, tmp_path):
        """CORE-B-006: merge_history.json carries the version sourced from
        the single ``CURRENT_SLAO_VERSION`` constant (not a duplicated
        literal that could drift from the load-side check)."""
        import json

        merger = SLAOMerger()
        merger.initialize(sample_lora_state)
        save_dir = tmp_path / "merger"
        merger.save(str(save_dir))

        history = json.loads((save_dir / "merge_history.json").read_text())
        assert history["version"] == SLAOMerger.CURRENT_SLAO_VERSION


class TestMergeLoraWeights:
    """Tests for the convenience merge_lora_weights function."""

    @pytest.fixture
    def sample_loras(self):
        """Create sample LoRA state dicts."""
        base = {
            "layer.lora_A.weight": torch.randn(16, 128),
            "layer.lora_B.weight": torch.randn(256, 16),
        }
        new = {
            "layer.lora_A.weight": torch.randn(16, 128),
            "layer.lora_B.weight": torch.randn(256, 16),
        }
        return base, new

    def test_slao_method(self, sample_loras):
        """Should merge using SLAO method."""
        base, new = sample_loras
        result = merge_lora_weights(base, new, run_index=2, method="slao")

        assert "layer.lora_A.weight" in result
        assert "layer.lora_B.weight" in result

    def test_average_method(self, sample_loras):
        """Should merge using simple averaging."""
        base, new = sample_loras
        result = merge_lora_weights(base, new, method="average")

        expected_A = (base["layer.lora_A.weight"] + new["layer.lora_A.weight"]) / 2
        assert torch.allclose(result["layer.lora_A.weight"], expected_A)

    def test_replace_method(self, sample_loras):
        """Should replace with new LoRA."""
        base, new = sample_loras
        result = merge_lora_weights(base, new, method="replace")

        assert torch.allclose(result["layer.lora_A.weight"], new["layer.lora_A.weight"])

    def test_invalid_method(self, sample_loras):
        """Should raise error for unknown method."""
        from backpropagate.exceptions import InvalidSettingError
        base, new = sample_loras
        with pytest.raises(InvalidSettingError, match="method"):
            merge_lora_weights(base, new, method="unknown")


class TestSLAOEdgeCases:
    """Edge case and stress tests for SLAO."""

    def test_large_rank(self):
        """Should handle large LoRA ranks."""
        A = torch.randn(128, 4096)
        A_ortho = orthogonal_init_A(A)

        result = A_ortho @ A_ortho.T
        assert torch.allclose(result, torch.eye(128), atol=1e-4)

    def test_many_merges(self):
        """Should handle many sequential merges."""
        merger = SLAOMerger()

        lora = {
            "layer.lora_A.weight": torch.randn(16, 128),
            "layer.lora_B.weight": torch.randn(256, 16),
        }
        merger.initialize(lora)

        for i in range(2, 101):
            new_lora = {k: torch.randn_like(v) for k, v in lora.items()}
            result = merger.merge(new_lora, run_index=i)

            # Scale should still be reasonable
            assert result.scale_factor >= 0.1

        assert merger.run_index == 100

    def test_empty_state_dict(self):
        """Should handle empty state dict gracefully."""
        merger = SLAOMerger()
        merger.initialize({})

        assert merger._merged_state == {}

    def test_non_tensor_values(self):
        """Should handle non-tensor values in state dict."""
        merger = SLAOMerger()

        lora = {
            "layer.lora_A.weight": torch.randn(16, 128),
            "config": {"r": 16},  # Non-tensor
        }
        merger.initialize(lora)

        assert "config" in merger._merged_state


class TestGetLayerScale:
    """Tests for get_layer_scale function - Phase 4.2 selective layer merging."""

    def test_early_layer_returns_early_scale(self):
        """Early layers (0-33%) should return early_scale."""
        # Layer 0 of 30 layers = 0% position
        scale = get_layer_scale("model.layers.0.self_attn.q_proj", total_layers=30)
        assert scale == 0.3, "Layer 0/30 should return early_scale (0.3)"

        # Layer 5 of 30 = 17% position
        scale = get_layer_scale("model.layers.5.self_attn.q_proj", total_layers=30)
        assert scale == 0.3, "Layer 5/30 should return early_scale (0.3)"

        # Layer 9 of 30 = 31% position (still < 0.33)
        scale = get_layer_scale("model.layers.9.self_attn.q_proj", total_layers=30)
        assert scale == 0.3, "Layer 9/30 should return early_scale (0.3)"

    def test_middle_layer_returns_middle_scale(self):
        """Middle layers (33-66%) should return middle_scale."""
        # Layer 10 of 30 = 34.5% position
        scale = get_layer_scale("model.layers.10.self_attn.q_proj", total_layers=30)
        assert scale == 0.5, "Layer 10/30 should return middle_scale (0.5)"

        # Layer 15 of 30 = 51.7% position
        scale = get_layer_scale("model.layers.15.self_attn.q_proj", total_layers=30)
        assert scale == 0.5, "Layer 15/30 should return middle_scale (0.5)"

        # Layer 18 of 30 = 62% position (still < 0.66)
        scale = get_layer_scale("model.layers.18.self_attn.q_proj", total_layers=30)
        assert scale == 0.5, "Layer 18/30 should return middle_scale (0.5)"

    def test_late_layer_returns_late_scale(self):
        """Late layers (66-100%) should return late_scale."""
        # Layer 20 of 30 = 69% position
        scale = get_layer_scale("model.layers.20.self_attn.q_proj", total_layers=30)
        assert scale == 0.7, "Layer 20/30 should return late_scale (0.7)"

        # Layer 29 of 30 = 100% position
        scale = get_layer_scale("model.layers.29.self_attn.q_proj", total_layers=30)
        assert scale == 0.7, "Layer 29/30 should return late_scale (0.7)"

    def test_custom_scale_values(self):
        """Should use custom scale values when provided."""
        scale = get_layer_scale(
            "model.layers.0.self_attn.q_proj",
            total_layers=30,
            early_scale=0.1,
            middle_scale=0.4,
            late_scale=0.9,
        )
        assert scale == 0.1, "Custom early_scale should be used"

        scale = get_layer_scale(
            "model.layers.15.self_attn.q_proj",
            total_layers=30,
            early_scale=0.1,
            middle_scale=0.4,
            late_scale=0.9,
        )
        assert scale == 0.4, "Custom middle_scale should be used"

        scale = get_layer_scale(
            "model.layers.25.self_attn.q_proj",
            total_layers=30,
            early_scale=0.1,
            middle_scale=0.4,
            late_scale=0.9,
        )
        assert scale == 0.9, "Custom late_scale should be used"

    def test_boundary_at_033(self):
        """Test boundary at 0.33 threshold."""
        # Exactly at 0.33 should be middle (>=0.33 goes to middle)
        # Layer 9 of 28 = 9/27 = 0.333... position
        scale = get_layer_scale("model.layers.9.self_attn.q_proj", total_layers=28)
        # Check which side of boundary
        pos = 9 / 27
        if pos < 0.33:
            assert scale == 0.3
        else:
            assert scale == 0.5

    def test_boundary_at_066(self):
        """Test boundary at 0.66 threshold."""
        # Layer 19 of 29 = 19/28 ≈ 0.678 position (late)
        scale = get_layer_scale("model.layers.19.self_attn.q_proj", total_layers=29)
        assert scale == 0.7, "Layer at 67.8% should be late"

        # Layer 18 of 29 = 18/28 ≈ 0.643 position (middle)
        scale = get_layer_scale("model.layers.18.self_attn.q_proj", total_layers=29)
        assert scale == 0.5, "Layer at 64.3% should be middle"

    def test_unknown_layer_pattern_returns_middle(self):
        """Unknown layer name patterns should return middle_scale."""
        scale = get_layer_scale("some.random.key", total_layers=30)
        assert scale == 0.5, "Unknown pattern should return middle_scale"

        scale = get_layer_scale("embedding.weight", total_layers=30)
        assert scale == 0.5, "Non-layer key should return middle_scale"

    def test_alternative_layer_patterns(self):
        """Should recognize h.X. and block.X. patterns."""
        # GPT-2 style: h.X.
        scale = get_layer_scale("h.0.attn.c_attn", total_layers=12)
        assert scale == 0.3, "h.0. pattern should be early"

        scale = get_layer_scale("h.11.attn.c_attn", total_layers=12)
        assert scale == 0.7, "h.11. of 12 should be late"

        # Block style: block.X.
        scale = get_layer_scale("block.5.attention", total_layers=10)
        assert scale == 0.5, "block.5. of 10 should be middle"

    def test_single_layer_model(self):
        """Should handle single layer model (avoid division by zero)."""
        scale = get_layer_scale("model.layers.0.q_proj", total_layers=1)
        # With 1 layer, position = 0 / max(0, 1) = 0, so early
        assert scale == 0.3, "Single layer should return early_scale"


class TestComputeTaskSimilarity:
    """Tests for compute_task_similarity function - Phase 4.1."""

    def test_identical_loras_have_similarity_1(self):
        """Identical LoRAs should have similarity ~1.0."""
        lora = {
            "layer.lora_B.weight": torch.randn(256, 16),
        }
        similarity = compute_task_similarity(lora, lora)
        assert abs(similarity - 1.0) < 1e-5, "Identical LoRAs should have similarity 1.0"

    def test_opposite_loras_have_similarity_minus_1(self):
        """Negated LoRAs should have similarity -1.0."""
        lora1 = {
            "layer.lora_B.weight": torch.randn(256, 16),
        }
        lora2 = {
            "layer.lora_B.weight": -lora1["layer.lora_B.weight"],
        }
        similarity = compute_task_similarity(lora1, lora2)
        assert abs(similarity - (-1.0)) < 1e-5, "Opposite LoRAs should have similarity -1.0"

    def test_orthogonal_loras_have_similarity_0(self):
        """Orthogonal LoRAs should have similarity ~0."""
        # Create orthogonal vectors using Gram-Schmidt
        v1 = torch.randn(256 * 16)
        v2 = torch.randn(256 * 16)
        v2 = v2 - (torch.dot(v1, v2) / torch.dot(v1, v1)) * v1
        v2 = v2 / torch.norm(v2) * torch.norm(v1)  # Same magnitude

        lora1 = {"layer.lora_B.weight": v1.reshape(256, 16)}
        lora2 = {"layer.lora_B.weight": v2.reshape(256, 16)}

        similarity = compute_task_similarity(lora1, lora2)
        assert abs(similarity) < 0.01, f"Orthogonal LoRAs should have similarity ~0, got {similarity}"

    def test_no_b_matrices_returns_0(self):
        """If no B matrices found, should return 0.0."""
        lora1 = {"layer.lora_A.weight": torch.randn(16, 128)}
        lora2 = {"layer.lora_A.weight": torch.randn(16, 128)}
        similarity = compute_task_similarity(lora1, lora2)
        assert similarity == 0.0, "No B matrices should return 0.0"

    def test_zero_norm_returns_0(self):
        """If either LoRA has zero norm, should return 0.0."""
        lora1 = {"layer.lora_B.weight": torch.zeros(256, 16)}
        lora2 = {"layer.lora_B.weight": torch.randn(256, 16)}
        similarity = compute_task_similarity(lora1, lora2)
        assert similarity == 0.0, "Zero norm LoRA should return 0.0"

        # Both zero
        lora3 = {"layer.lora_B.weight": torch.zeros(256, 16)}
        similarity = compute_task_similarity(lora1, lora3)
        assert similarity == 0.0, "Both zero norm should return 0.0"

    def test_mismatched_keys_only_compares_common(self):
        """Should only compare B matrices present in both."""
        lora1 = {
            "layer1.lora_B.weight": torch.randn(256, 16),
            "layer2.lora_B.weight": torch.randn(256, 16),
        }
        lora2 = {
            "layer1.lora_B.weight": lora1["layer1.lora_B.weight"].clone(),  # Same
            # layer2 missing
        }
        # Only layer1 is compared, which is identical
        similarity = compute_task_similarity(lora1, lora2)
        assert abs(similarity - 1.0) < 1e-5, "Should compare only common keys"

    def test_similarity_in_valid_range(self):
        """Similarity should always be in [-1, 1]."""
        for _ in range(10):
            lora1 = {"layer.lora_B.weight": torch.randn(256, 16)}
            lora2 = {"layer.lora_B.weight": torch.randn(256, 16)}
            similarity = compute_task_similarity(lora1, lora2)
            assert -1.0 <= similarity <= 1.0, f"Similarity {similarity} out of range"


class TestAdaptiveScale:
    """Tests for adaptive_scale function - Phase 4.1."""

    def test_similarity_1_gives_max_multiplier(self):
        """Similarity of 1 should give max multiplier."""
        base_scale = 0.5
        result = adaptive_scale(base_scale, similarity=1.0, scale_range=(0.5, 1.5))
        expected = base_scale * 1.5  # max multiplier
        assert abs(result - expected) < 1e-6, f"Similarity 1 should give {expected}, got {result}"

    def test_similarity_minus_1_gives_min_multiplier(self):
        """Similarity of -1 should give min multiplier."""
        base_scale = 0.5
        result = adaptive_scale(base_scale, similarity=-1.0, scale_range=(0.5, 1.5))
        expected = base_scale * 0.5  # min multiplier
        assert abs(result - expected) < 1e-6, f"Similarity -1 should give {expected}, got {result}"

    def test_similarity_0_gives_midpoint_multiplier(self):
        """Similarity of 0 should give multiplier of 1.0 (midpoint)."""
        base_scale = 0.5
        result = adaptive_scale(base_scale, similarity=0.0, scale_range=(0.5, 1.5))
        expected = base_scale * 1.0  # midpoint multiplier
        assert abs(result - expected) < 1e-6, f"Similarity 0 should give {expected}, got {result}"

    def test_custom_scale_range(self):
        """Should use custom scale_range."""
        base_scale = 1.0
        result = adaptive_scale(base_scale, similarity=1.0, scale_range=(0.2, 2.0))
        assert abs(result - 2.0) < 1e-6, "Max similarity with range (0.2, 2.0) should give 2.0"

        result = adaptive_scale(base_scale, similarity=-1.0, scale_range=(0.2, 2.0))
        assert abs(result - 0.2) < 1e-6, "Min similarity with range (0.2, 2.0) should give 0.2"

    def test_linear_interpolation(self):
        """Should linearly interpolate between min and max."""
        base_scale = 1.0
        min_mult, max_mult = 0.5, 1.5

        # similarity=0.5 -> normalized = 0.75 -> multiplier = 0.5 + 0.75*1.0 = 1.25
        result = adaptive_scale(base_scale, similarity=0.5, scale_range=(min_mult, max_mult))
        expected = 1.25
        assert abs(result - expected) < 1e-6, f"Similarity 0.5 should give {expected}, got {result}"

        # similarity=-0.5 -> normalized = 0.25 -> multiplier = 0.5 + 0.25*1.0 = 0.75
        result = adaptive_scale(base_scale, similarity=-0.5, scale_range=(min_mult, max_mult))
        expected = 0.75
        assert abs(result - expected) < 1e-6, f"Similarity -0.5 should give {expected}, got {result}"

    def test_similarity_normalization_formula(self):
        """Test the exact normalization formula: (similarity + 1) / 2."""
        # For similarity = 1: (1 + 1) / 2 = 1.0
        # For similarity = -1: (-1 + 1) / 2 = 0.0
        # For similarity = 0: (0 + 1) / 2 = 0.5
        base_scale = 1.0
        scale_range = (0.0, 2.0)

        for sim in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            result = adaptive_scale(base_scale, sim, scale_range)
            normalized = (sim + 1) / 2
            expected = base_scale * (0.0 + normalized * 2.0)
            assert abs(result - expected) < 1e-6, f"Formula mismatch at sim={sim}"


class TestEstimateTotalLayers:
    """Tests for estimate_total_layers function."""

    def test_layers_pattern(self):
        """Should detect 'layers.X.' pattern."""
        lora = {
            "model.layers.0.q_proj": torch.randn(10, 10),
            "model.layers.15.k_proj": torch.randn(10, 10),
            "model.layers.31.v_proj": torch.randn(10, 10),
        }
        total = estimate_total_layers(lora)
        assert total == 32, "Should detect max layer 31 -> 32 layers"

    def test_h_pattern(self):
        """Should detect 'h.X.' pattern (GPT-2 style)."""
        lora = {
            "h.0.attn": torch.randn(10, 10),
            "h.11.attn": torch.randn(10, 10),
        }
        total = estimate_total_layers(lora)
        assert total == 12, "Should detect max layer 11 -> 12 layers"

    def test_block_pattern(self):
        """Should detect 'block.X.' pattern."""
        lora = {
            "block.0.attention": torch.randn(10, 10),
            "block.5.attention": torch.randn(10, 10),
        }
        total = estimate_total_layers(lora)
        assert total == 6, "Should detect max layer 5 -> 6 layers"

    def test_no_layer_pattern_returns_1(self):
        """If no layer pattern found, should return 1."""
        lora = {
            "embedding.weight": torch.randn(10, 10),
            "lm_head.weight": torch.randn(10, 10),
        }
        total = estimate_total_layers(lora)
        assert total == 1, "No layer pattern should return 1 (0 + 1)"

    def test_empty_dict_returns_1(self):
        """Empty dict should return 1."""
        total = estimate_total_layers({})
        assert total == 1, "Empty dict should return 1"


class TestPhase4Integration:
    """Integration tests for Phase 4 features (adaptive and layer scaling)."""

    @pytest.fixture
    def multi_layer_lora(self):
        """Create a multi-layer LoRA state dict."""
        lora = {}
        for i in range(32):
            lora[f"model.layers.{i}.lora_A.weight"] = torch.randn(16, 128)
            lora[f"model.layers.{i}.lora_B.weight"] = torch.randn(256, 16)
        return lora

    def test_adaptive_scaling_enabled(self, multi_layer_lora):
        """Test merger with adaptive scaling enabled."""
        config = SLAOConfig(use_adaptive_scaling=True)
        merger = SLAOMerger(config)
        merger.initialize(multi_layer_lora)

        # Create similar LoRA (should get higher scale)
        similar_lora = {
            k: v + torch.randn_like(v) * 0.1  # Small perturbation
            for k, v in multi_layer_lora.items()
        }
        result = merger.merge(similar_lora, run_index=2)

        # Scale should be adjusted by similarity
        base_scale = 1.0 / math.sqrt(2)
        # With high similarity, scale should be > base_scale
        assert result.scale_factor > 0, "Scale factor should be positive"

    def test_layer_scaling_enabled(self, multi_layer_lora):
        """Test merger with layer scaling enabled."""
        config = SLAOConfig(
            use_layer_scaling=True,
            layer_scale_early=0.3,
            layer_scale_middle=0.5,
            layer_scale_late=0.7,
        )
        merger = SLAOMerger(config)
        merger.initialize(multi_layer_lora)

        new_lora = {k: torch.randn_like(v) for k, v in multi_layer_lora.items()}
        result = merger.merge(new_lora, run_index=2)

        # Should complete without error and merge matrices
        assert result.a_matrices_merged == 32, "Should merge all 32 A matrices"
        assert result.b_matrices_merged == 32, "Should merge all 32 B matrices"

    def test_both_adaptive_and_layer_scaling(self, multi_layer_lora):
        """Test merger with both adaptive and layer scaling."""
        config = SLAOConfig(
            use_adaptive_scaling=True,
            use_layer_scaling=True,
        )
        merger = SLAOMerger(config)
        merger.initialize(multi_layer_lora)

        new_lora = {k: torch.randn_like(v) for k, v in multi_layer_lora.items()}
        result = merger.merge(new_lora, run_index=2)

        assert result.scale_factor > 0, "Scale factor should be positive"
        assert result.a_matrices_merged > 0, "Should merge A matrices"
        assert result.b_matrices_merged > 0, "Should merge B matrices"


class TestScaleUpperClamp:
    """Regression tests for CONTINUAL-A-001 + CONTINUAL-A-002.

    The B-matrix EMA weight passed into ``merge_B_matrices`` must ALWAYS be
    within ``[min_scale, 1.0]``. A weight >1.0 over-extrapolates *past* the
    new adapter (B_merged + s*(B_new - B_merged) with s>1 lands beyond
    B_new), destroying the anti-catastrophic-forgetting invariant: instead
    of moving the accumulator toward the freshly-trained adapter, it shoots
    past it, amplifying the new task's delta and discarding more of the
    prior knowledge than even a full replacement would.

    Two leaks fed an out-of-range weight into the merge:
      * A-001 — ``time_aware_scale``'s callable / string branches returned
        ``max(scale, min_scale)`` with no upper clamp, so a custom schedule
        returning >1.0 flowed straight through.
      * A-002 — ``adaptive_scale`` multiplies the base scale by up to
        ``adaptive_scale_range[1]`` (default 1.5), so e.g. run-2 base
        0.707 * 1.5 = 1.06; ``merge()`` used this UNCLAMPED both as the
        B-merge weight and as ``effective_scale = scale * layer_scale``.
    """

    # --- A-001: time_aware_scale upper clamp -------------------------------

    def test_callable_schedule_clamped_above_1(self):
        """A custom callable returning >1.0 must clamp to 1.0 (docstring
        contract: 'clamped to [min_scale, 1.0]')."""
        # Schedule returns a constant 5.0 regardless of run index.
        scale = time_aware_scale(3, scaling_type=lambda i: 5.0)
        assert scale == 1.0, (
            f"callable schedule returning 5.0 must clamp to 1.0, got {scale}"
        )

    def test_callable_schedule_clamped_below_min(self):
        """A custom callable returning <min_scale must clamp up to min_scale
        (the existing lower-clamp must survive the upper-clamp fix)."""
        scale = time_aware_scale(3, scaling_type=lambda i: 0.0, min_scale=0.2)
        assert scale == 0.2, (
            f"callable schedule returning 0.0 must clamp to min_scale 0.2, got {scale}"
        )

    def test_callable_schedule_passthrough_in_range(self):
        """An in-range callable value is returned unchanged."""
        scale = time_aware_scale(3, scaling_type=lambda i: 0.6)
        assert abs(scale - 0.6) < 1e-9, f"in-range value must pass through, got {scale}"

    def test_all_string_schedules_within_unit_interval(self):
        """No built-in string schedule may exceed 1.0 for any run index
        (log at run 1 = 1/log(2) ≈ 1.443 would violate this without the
        upper clamp — the canonical A-001 string-branch leak)."""
        for stype in ("sqrt", "linear", "log", "constant"):
            for i in range(1, 50):
                scale = time_aware_scale(i, scaling_type=stype)
                assert 0.1 <= scale <= 1.0, (
                    f"{stype} scale at run {i} = {scale} escaped [min_scale, 1.0]"
                )

    def test_log_schedule_run_1_clamped(self):
        """Pin the specific log-at-run-1 case: raw 1/log(2) ≈ 1.443 must
        clamp to exactly 1.0."""
        scale = time_aware_scale(1, scaling_type="log")
        assert scale == 1.0, (
            f"log scale at run 1 (raw ~1.443) must clamp to 1.0, got {scale}"
        )

    # --- A-002: adaptive_scale reaches the B-merge clamped -----------------

    def _capture_merge_scales(self, merger, new_lora, run_index):
        """Run a merge while intercepting every ``scale`` value handed to
        ``merge_B_matrices``. Returns the list of captured scales."""
        captured: list[float] = []
        import backpropagate.slao as slao_mod

        real_merge_B = slao_mod.merge_B_matrices

        def _spy(B_merged, B_new, scale):
            captured.append(float(scale))
            return real_merge_B(B_merged, B_new, scale)

        from unittest.mock import patch

        with patch.object(slao_mod, "merge_B_matrices", _spy):
            merger.merge(new_lora, run_index=run_index)
        return captured

    def test_adaptive_scale_reaching_b_merge_is_clamped(self):
        """With adaptive scaling and a near-identical adapter (similarity≈1),
        the base scale gets multiplied by ~1.5 → would-be >1.0. Every value
        reaching merge_B_matrices must be clamped into [min_scale, 1.0]."""
        config = SLAOConfig(use_adaptive_scaling=True, min_scale=0.1)
        merger = SLAOMerger(config)
        base = {
            "model.layers.0.lora_A.weight": torch.ones(8, 16),
            "model.layers.0.lora_B.weight": torch.ones(32, 8),
        }
        merger.initialize(base)
        # Near-identical → cosine similarity ≈ 1 → multiplier ≈ 1.5.
        new_lora = {k: v.clone() * 1.0001 for k, v in base.items()}

        captured = self._capture_merge_scales(merger, new_lora, run_index=2)

        assert captured, "merge_B_matrices was never called"
        for s in captured:
            assert 0.1 <= s <= 1.0, (
                f"scale {s} handed to merge_B_matrices escaped [min_scale, 1.0]"
            )

    def test_adaptive_with_layer_scaling_clamped(self):
        """effective_scale = scale * layer_scale must also be clamped — the
        ~791 site. Layer scaling can only shrink (factors <=0.7) but the
        clamp on the combined value is the invariant under test; with
        adaptive pushing >1.0 first, the product before clamping could still
        land >1.0 for early layers if the implementation clamped in the
        wrong order. Assert the value that reaches the merge stays in range."""
        config = SLAOConfig(
            use_adaptive_scaling=True,
            use_layer_scaling=True,
            layer_scale_late=1.0,  # late layers keep the full (clamped) scale
            min_scale=0.1,
        )
        merger = SLAOMerger(config)
        base = {}
        for i in range(4):
            base[f"model.layers.{i}.lora_A.weight"] = torch.ones(8, 16)
            base[f"model.layers.{i}.lora_B.weight"] = torch.ones(32, 8)
        merger.initialize(base)
        new_lora = {k: v.clone() * 1.0001 for k, v in base.items()}

        captured = self._capture_merge_scales(merger, new_lora, run_index=2)

        assert captured, "merge_B_matrices was never called"
        for s in captured:
            assert 0.1 <= s <= 1.0, (
                f"effective_scale {s} reaching merge_B_matrices escaped "
                f"[min_scale, 1.0]"
            )

    def test_merge_does_not_extrapolate_past_b_new(self):
        """The load-bearing invariant: a would-be >1.0 scale must NOT push
        the merged B past B_new.

        Setup forces a >1.0 raw scale: B_merged=ones, B_new=ones*2 are
        parallel non-zero vectors → cosine similarity 1.0 → adaptive
        multiplier 1.5 → raw scale 0.707*1.5 = 1.06. The EMA update is
        merged = B_merged + s*(B_new - B_merged) = 1 + s*(2-1) = 1 + s.
          * Pre-fix (s=1.06): merged ≈ 2.06  → PAST B_new=2 (over-extrapolated).
          * Post-fix (s clamped to 1.0): merged = 2.0 → lands AT B_new, never past.
        """
        config = SLAOConfig(use_adaptive_scaling=True, min_scale=0.1)
        merger = SLAOMerger(config)
        base = {
            "model.layers.0.lora_A.weight": torch.ones(8, 16),
            "model.layers.0.lora_B.weight": torch.ones(32, 8),
        }
        merger.initialize(base)
        # B_new parallel to B_merged but larger → similarity 1.0, B_new=2.
        new_lora = {
            "model.layers.0.lora_A.weight": torch.ones(8, 16),
            "model.layers.0.lora_B.weight": torch.ones(32, 8) * 2.0,
        }

        result = merger.merge(new_lora, run_index=2)
        merged_b = merger.get_merged_lora()["model.layers.0.lora_B.weight"]

        # B_new is all-2.0. With the clamp, s <= 1.0 so merged_b <= 2.0
        # everywhere. Pre-fix s≈1.06 would land merged_b ≈ 2.06 > 2.0
        # (extrapolation past the new adapter).
        assert torch.all(merged_b <= 2.0 + 1e-6), (
            f"merged B extrapolated past B_new=2.0 (max={merged_b.max().item()}); "
            f"scale_factor={result.scale_factor}"
        )

    def test_reported_scale_factor_clamped(self):
        """MergeResult.scale_factor (the value also used as the B-merge
        weight when layer scaling is off) must itself be clamped — it's the
        operator-visible record of what the merge actually applied."""
        config = SLAOConfig(use_adaptive_scaling=True, min_scale=0.1)
        merger = SLAOMerger(config)
        base = {
            "model.layers.0.lora_A.weight": torch.ones(8, 16),
            "model.layers.0.lora_B.weight": torch.ones(32, 8),
        }
        merger.initialize(base)
        new_lora = {k: v.clone() * 1.0001 for k, v in base.items()}
        result = merger.merge(new_lora, run_index=2)
        assert 0.1 <= result.scale_factor <= 1.0, (
            f"reported scale_factor {result.scale_factor} escaped [min_scale, 1.0]"
        )


class TestMergeDeviceNormalization:
    """CONTINUAL-A-005: the accumulator tensor must be aligned to the new
    value's device before any merge arithmetic.

    On a RESUMED session ``load()`` rehydrates the accumulator via
    ``torch.load`` onto whatever device it was serialized from (CPU for a
    CPU-saved checkpoint), while the incoming ``new_lora_state`` comes from
    the live model (possibly CUDA). Without normalization the first resumed
    ``merge_B_matrices`` (B_merged + s*(B_new - B_merged)) raises a
    device-mismatch RuntimeError. These tests pin the alignment + the
    no-op-on-match guarantee; on a CPU-only box they exercise the same code
    path (the ``.device != .device`` branch is simply not taken).
    """

    @pytest.fixture
    def sample_lora_state(self):
        return {
            "model.layers.0.lora_A.weight": torch.randn(8, 16),
            "model.layers.0.lora_B.weight": torch.randn(32, 8),
        }

    def test_save_load_merge_roundtrip_no_device_error(self, sample_lora_state):
        """The resume shape: initialize → save → load into a fresh merger →
        merge a live adapter. Must not raise, and merged tensors must land on
        the incoming adapter's device."""
        merger = SLAOMerger()
        merger.initialize(sample_lora_state)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "slao"
            merger.save(str(save_path))

            resumed = SLAOMerger()
            resumed.load(str(save_path))

            new_lora = {k: torch.randn_like(v) for k, v in sample_lora_state.items()}
            # Must not raise a device-mismatch RuntimeError.
            resumed.merge(new_lora, run_index=2)

            merged = resumed.get_merged_lora()
            for key, new_value in new_lora.items():
                assert merged[key].device == new_value.device, (
                    f"{key}: merged tensor on {merged[key].device}, "
                    f"new value on {new_value.device} — accumulator was not "
                    f"aligned to the new value's device"
                )

    def test_merge_aligns_accumulator_to_new_value_device(self, sample_lora_state):
        """Directly exercise the alignment: after a merge, every accumulator
        tensor shares the device of the corresponding new tensor."""
        merger = SLAOMerger()
        merger.initialize(sample_lora_state)
        new_lora = {k: torch.randn_like(v) for k, v in sample_lora_state.items()}
        merger.merge(new_lora, run_index=2)
        merged = merger.get_merged_lora()
        for key, new_value in new_lora.items():
            assert merged[key].device == new_value.device


class TestDoRAMagnitudeMerge:
    """CONTINUAL-A-006: DoRA magnitude vectors are hard-REPLACED (treated as
    fresh), not EMA-blended.

    DoRA decomposes W = m · (W0 + BA)/||W0 + BA||. The magnitude ``m`` (PEFT
    key ``...lora_magnitude_vector...``) is retrained every run and is coupled
    to the *current* direction. SLAO hard-replaces the direction's A matrix,
    so EMA-blending ``m`` independently (the old generic 'other' branch) would
    leave the magnitude describing a stale direction → an internally
    inconsistent adapter. Treating ``m`` as fresh keeps the pair coherent.
    """

    def _state(self, a_val, b_val, m_val):
        return {
            "model.layers.0.lora_A.weight": torch.full((8, 16), float(a_val)),
            "model.layers.0.lora_B.weight": torch.full((32, 8), float(b_val)),
            "model.layers.0.lora_magnitude_vector.weight": torch.full(
                (32,), float(m_val)
            ),
        }

    def test_magnitude_vector_is_replaced_not_blended(self):
        """A run-2 magnitude vector must equal the NEW value exactly (replace),
        not a midpoint between old and new (which an EMA blend would give)."""
        merger = SLAOMerger()
        base = self._state(a_val=1.0, b_val=1.0, m_val=1.0)
        merger.initialize(base)

        new_lora = self._state(a_val=5.0, b_val=5.0, m_val=9.0)
        merger.merge(new_lora, run_index=2)

        merged = merger.get_merged_lora()
        mag = merged["model.layers.0.lora_magnitude_vector.weight"]
        # Replaced: equals the new magnitude (9.0). An EMA blend with the
        # run-2 sqrt scale (~0.707) would yield 1 + 0.707*(9-1) ≈ 6.66.
        assert torch.allclose(mag, torch.full((32,), 9.0)), (
            f"magnitude vector was blended, not replaced: got {mag.flatten()[0].item()} "
            f"(expected 9.0; an EMA blend would land ~6.66)"
        )

    def test_magnitude_vector_matches_replaced_A_direction(self):
        """Consistency: after merge, the magnitude AND the A matrix both equal
        their fresh (run-2) values — the coupled pair stays in lockstep."""
        merger = SLAOMerger()
        merger.initialize(self._state(1.0, 1.0, 1.0))
        new_lora = self._state(a_val=5.0, b_val=5.0, m_val=9.0)
        merger.merge(new_lora, run_index=2)
        merged = merger.get_merged_lora()
        # A is hard-replaced (existing SLAO contract) ...
        assert torch.allclose(
            merged["model.layers.0.lora_A.weight"], torch.full((8, 16), 5.0)
        )
        # ... and so is the magnitude (the A-006 fix).
        assert torch.allclose(
            merged["model.layers.0.lora_magnitude_vector.weight"],
            torch.full((32,), 9.0),
        )


class TestMergeLoraWeightsAverageKeyUnion:
    """CONTINUAL-A-008: ``merge_lora_weights(method='average')`` must not drop
    keys present in only one of the two state dicts."""

    def test_new_only_key_is_preserved(self):
        """A key present only in ``new_lora`` survives into the result."""
        base = {"layer.lora_A.weight": torch.randn(8, 16)}
        new = {
            "layer.lora_A.weight": torch.randn(8, 16),
            "layer.lora_B.weight": torch.randn(32, 8),  # new-only key
        }
        result = merge_lora_weights(base, new, method="average")
        assert "layer.lora_B.weight" in result, (
            "average method dropped a key present only in new_lora"
        )
        # The new-only key passes through as a clone of the new tensor.
        assert torch.allclose(result["layer.lora_B.weight"], new["layer.lora_B.weight"])

    def test_base_only_key_is_preserved(self):
        """A key present only in ``base_lora`` survives (pre-fix behavior kept)."""
        base = {
            "layer.lora_A.weight": torch.randn(8, 16),
            "layer.lora_B.weight": torch.randn(32, 8),  # base-only key
        }
        new = {"layer.lora_A.weight": torch.randn(8, 16)}
        result = merge_lora_weights(base, new, method="average")
        assert "layer.lora_B.weight" in result
        assert torch.allclose(result["layer.lora_B.weight"], base["layer.lora_B.weight"])

    def test_common_keys_still_averaged(self):
        """Keys in both dicts are still the elementwise mean (no regression)."""
        base = {"layer.lora_A.weight": torch.ones(8, 16)}
        new = {"layer.lora_A.weight": torch.ones(8, 16) * 3.0}
        result = merge_lora_weights(base, new, method="average")
        assert torch.allclose(result["layer.lora_A.weight"], torch.full((8, 16), 2.0))

    def test_new_only_clone_is_independent(self):
        """The preserved new-only key must be a clone (mutating the source
        must not bleed into the merged result)."""
        base = {"layer.lora_A.weight": torch.randn(8, 16)}
        new_b = torch.ones(32, 8)
        new = {"layer.lora_A.weight": torch.randn(8, 16), "layer.lora_B.weight": new_b}
        result = merge_lora_weights(base, new, method="average")
        new_b[0, 0] = 999.0
        assert result["layer.lora_B.weight"][0, 0] != 999.0, (
            "new-only key was stored by reference, not cloned"
        )


# =============================================================================
# v1.5 T2.2: PLUGGABLE MERGE-STRATEGY FRAMEWORK
#
# Strategy math on tiny hand-computed lora_A/lora_B dicts (r=2 shaped), dispatch
# validation, finite-check (SLAO_MERGE_DIVERGED), and the qiao_mahdavi
# regression lock (== direct SLAOMerger.merge).
# =============================================================================

from backpropagate.exceptions import BackpropagateError, InvalidSettingError
from backpropagate.slao import (
    MERGE_STRATEGIES,
    DriftDecision,
    MergeStrategyConfig,
    apply_merge_strategy,
    drift_gate,
    merge_strategy_dare,
    merge_strategy_linear,
    merge_strategy_qiao_mahdavi,
    merge_strategy_ties,
)


def _tiny_lora(a_vals, b_vals, key_prefix="model.layers.0.self_attn.q_proj"):
    """Build a tiny LoRA state dict with one A + one B tensor (r=2 shaped)."""
    return {
        f"{key_prefix}.lora_A.default.weight": torch.tensor(a_vals, dtype=torch.float32),
        f"{key_prefix}.lora_B.default.weight": torch.tensor(b_vals, dtype=torch.float32),
    }


class TestMergeStrategyConfigDefaults:
    """Behavior-preserving defaults for the strategy config."""

    def test_default_strategy_is_qiao_mahdavi(self):
        assert MergeStrategyConfig().strategy == "qiao_mahdavi"

    def test_default_trim_threshold(self):
        assert MergeStrategyConfig().trim_threshold == 0.2

    def test_default_drop_rate(self):
        assert MergeStrategyConfig().drop_rate == 0.5

    def test_default_dare_seed_none(self):
        assert MergeStrategyConfig().dare_seed is None

    def test_default_linear_weight_none(self):
        assert MergeStrategyConfig().linear_weight is None

    def test_dispatch_keys(self):
        assert MERGE_STRATEGIES == ("qiao_mahdavi", "linear", "ties", "dare")


class TestLinearStrategy:
    """linear: per-key (1-w)*acc + w*new; fixed weight + asymmetric clone."""

    def test_fixed_weight_half_is_mean(self):
        acc = _tiny_lora([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]])
        new = _tiny_lora([[5.0, 6.0], [7.0, 8.0]], [[3.0, 3.0], [3.0, 3.0]])
        cfg = MergeStrategyConfig(strategy="linear", linear_weight=0.5)
        out = merge_strategy_linear(
            acc, new, run_index=2, config=cfg, slao_config=SLAOConfig()
        )
        # (1-0.5)*acc + 0.5*new == elementwise mean for BOTH A and B.
        assert torch.allclose(
            out["model.layers.0.self_attn.q_proj.lora_A.default.weight"],
            torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        )
        assert torch.allclose(
            out["model.layers.0.self_attn.q_proj.lora_B.default.weight"],
            torch.full((2, 2), 2.0),
        )

    def test_fixed_weight_zero_keeps_accumulator(self):
        acc = _tiny_lora([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]])
        new = _tiny_lora([[5.0, 6.0], [7.0, 8.0]], [[3.0, 3.0], [3.0, 3.0]])
        cfg = MergeStrategyConfig(strategy="linear", linear_weight=0.0)
        out = merge_strategy_linear(
            acc, new, run_index=2, config=cfg, slao_config=SLAOConfig()
        )
        assert torch.allclose(
            out["model.layers.0.self_attn.q_proj.lora_B.default.weight"],
            torch.ones(2, 2),
        )

    def test_time_aware_weight_when_linear_weight_none(self):
        # run_index=4, sqrt schedule -> w = 1/sqrt(4) = 0.5.
        acc = _tiny_lora([[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]])
        new = _tiny_lora([[1.0, 1.0], [1.0, 1.0]], [[4.0, 4.0], [4.0, 4.0]])
        cfg = MergeStrategyConfig(strategy="linear", linear_weight=None)
        out = merge_strategy_linear(
            acc, new, run_index=4, config=cfg, slao_config=SLAOConfig()
        )
        # B = 0 + 0.5*(4-0) = 2.0
        assert torch.allclose(
            out["model.layers.0.self_attn.q_proj.lora_B.default.weight"],
            torch.full((2, 2), 2.0),
        )

    def test_asymmetric_key_cloned_through(self):
        acc = _tiny_lora([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]])
        acc["model.layers.9.only_in_acc.lora_B.default.weight"] = torch.tensor(
            [[7.0]]
        )
        new = _tiny_lora([[5.0, 6.0], [7.0, 8.0]], [[3.0, 3.0], [3.0, 3.0]])
        new["model.layers.9.only_in_new.lora_B.default.weight"] = torch.tensor(
            [[9.0]]
        )
        cfg = MergeStrategyConfig(strategy="linear", linear_weight=0.5)
        out = merge_strategy_linear(
            acc, new, run_index=2, config=cfg, slao_config=SLAOConfig()
        )
        assert torch.allclose(
            out["model.layers.9.only_in_acc.lora_B.default.weight"],
            torch.tensor([[7.0]]),
        )
        assert torch.allclose(
            out["model.layers.9.only_in_new.lora_B.default.weight"],
            torch.tensor([[9.0]]),
        )


class TestTiesStrategy:
    """ties: trim low-mag, elect sign, disjoint-merge agreeing contributors."""

    def test_low_mag_entry_trimmed_to_zero(self):
        # idx1 + idx3 tiny on BOTH sides -> trimmed both -> elected sign 0 -> 0.
        acc = {"m.lora_B.default.weight": torch.tensor([[10.0, 0.01, 8.0, 0.02]])}
        new = {"m.lora_B.default.weight": torch.tensor([[9.0, 0.01, 7.0, 0.02]])}
        cfg = MergeStrategyConfig(strategy="ties", trim_threshold=0.5)
        out = merge_strategy_ties(
            acc, new, run_index=2, config=cfg, slao_config=SLAOConfig()
        )
        assert torch.allclose(
            out["m.lora_B.default.weight"], torch.tensor([[9.5, 0.0, 7.5, 0.0]])
        )

    def test_sign_conflict_minority_dropped(self):
        # idx0: acc=+5 new=-1 -> gamma=sign(4)=+ -> minority(-1) dropped, mean({5})=5.
        acc = {"m.lora_B.default.weight": torch.tensor([[5.0, 5.0]])}
        new = {"m.lora_B.default.weight": torch.tensor([[-1.0, 5.0]])}
        cfg = MergeStrategyConfig(strategy="ties", trim_threshold=0.0)
        out = merge_strategy_ties(
            acc, new, run_index=2, config=cfg, slao_config=SLAOConfig()
        )
        assert torch.allclose(
            out["m.lora_B.default.weight"], torch.tensor([[5.0, 5.0]])
        )

    def test_all_zero_after_trim_position_is_zero(self):
        # A position trimmed on BOTH sides collapses to 0 (count==0 guard).
        acc = {"m.lora_B.default.weight": torch.tensor([[10.0, 0.001]])}
        new = {"m.lora_B.default.weight": torch.tensor([[9.0, 0.001]])}
        cfg = MergeStrategyConfig(strategy="ties", trim_threshold=0.5)
        out = merge_strategy_ties(
            acc, new, run_index=2, config=cfg, slao_config=SLAOConfig()
        )
        assert out["m.lora_B.default.weight"][0, 1].item() == 0.0

    def test_asymmetric_key_cloned_through(self):
        acc = {"m.lora_B.default.weight": torch.tensor([[5.0, 5.0]])}
        acc["extra.lora_A.default.weight"] = torch.tensor([[1.0]])
        new = {"m.lora_B.default.weight": torch.tensor([[5.0, 5.0]])}
        cfg = MergeStrategyConfig(strategy="ties", trim_threshold=0.2)
        out = merge_strategy_ties(
            acc, new, run_index=2, config=cfg, slao_config=SLAOConfig()
        )
        assert "extra.lora_A.default.weight" in out


class TestDareStrategy:
    """dare: Bernoulli drop on the increment via a LOCAL generator."""

    def test_fixed_seed_exact_mask_and_rescale(self):
        acc = {"m.lora_B.default.weight": torch.zeros(1, 6)}
        new = {"m.lora_B.default.weight": torch.tensor([[1.0, 2, 3, 4, 5, 6]])}
        cfg = MergeStrategyConfig(strategy="dare", drop_rate=0.5, dare_seed=1234)
        out = merge_strategy_dare(
            acc, new, run_index=2, config=cfg, slao_config=SLAOConfig()
        )
        # Replicate the LOCAL-generator draw exactly.
        gen = torch.Generator()
        gen.manual_seed(1234)
        rand = torch.rand((1, 6), generator=gen)
        keep = (rand < 0.5).float()
        delta = new["m.lora_B.default.weight"] - acc["m.lora_B.default.weight"]
        expected = acc["m.lora_B.default.weight"] + (keep * delta) / 0.5
        assert torch.allclose(out["m.lora_B.default.weight"], expected)

    def test_drop_rate_zero_is_replace_on_increment(self):
        acc = {"m.lora_B.default.weight": torch.tensor([[2.0, 2.0]])}
        new = {"m.lora_B.default.weight": torch.tensor([[5.0, 7.0]])}
        cfg = MergeStrategyConfig(strategy="dare", drop_rate=0.0)
        out = merge_strategy_dare(
            acc, new, run_index=2, config=cfg, slao_config=SLAOConfig()
        )
        # acc + delta == new.
        assert torch.allclose(
            out["m.lora_B.default.weight"], new["m.lora_B.default.weight"]
        )

    def test_global_rng_state_unchanged(self):
        """The DARE draw MUST NOT touch the global torch RNG (training
        reproducibility depends on it)."""
        acc = {"m.lora_B.default.weight": torch.zeros(1, 8)}
        new = {"m.lora_B.default.weight": torch.ones(1, 8)}
        cfg = MergeStrategyConfig(strategy="dare", drop_rate=0.5, dare_seed=42)
        before = torch.get_rng_state()
        merge_strategy_dare(
            acc, new, run_index=2, config=cfg, slao_config=SLAOConfig()
        )
        after = torch.get_rng_state()
        assert torch.equal(before, after), (
            "DARE merge perturbed the GLOBAL torch RNG — it must use a LOCAL "
            "torch.Generator only."
        )

    def test_none_seed_is_run_deterministic(self):
        """dare_seed=None derives from run_index deterministically."""
        acc = {"m.lora_B.default.weight": torch.zeros(1, 8)}
        new = {"m.lora_B.default.weight": torch.ones(1, 8)}
        cfg = MergeStrategyConfig(strategy="dare", drop_rate=0.5, dare_seed=None)
        out1 = merge_strategy_dare(
            acc, new, run_index=3, config=cfg, slao_config=SLAOConfig()
        )
        out2 = merge_strategy_dare(
            acc, new, run_index=3, config=cfg, slao_config=SLAOConfig()
        )
        assert torch.allclose(
            out1["m.lora_B.default.weight"], out2["m.lora_B.default.weight"]
        )


class TestStrategyDispatchValidation:
    """Unknown strategy / out-of-range thresholds raise InvalidSettingError."""

    def test_unknown_strategy_raises_at_merger_init(self):
        with pytest.raises(InvalidSettingError) as exc:
            SLAOMerger(strategy_config=MergeStrategyConfig(strategy="bogus"))
        assert exc.value.code == "CONFIG_INVALID_SETTING"

    def test_trim_threshold_out_of_range_raises(self):
        with pytest.raises(InvalidSettingError) as exc:
            SLAOMerger(strategy_config=MergeStrategyConfig(trim_threshold=1.0))
        assert exc.value.code == "CONFIG_INVALID_SETTING"

    def test_trim_threshold_negative_raises(self):
        with pytest.raises(InvalidSettingError):
            SLAOMerger(strategy_config=MergeStrategyConfig(trim_threshold=-0.1))

    def test_drop_rate_one_rejected(self):
        # drop_rate=1.0 drops everything + divide-by-zero — must be rejected.
        with pytest.raises(InvalidSettingError) as exc:
            SLAOMerger(strategy_config=MergeStrategyConfig(drop_rate=1.0))
        assert exc.value.code == "CONFIG_INVALID_SETTING"

    def test_drop_rate_zero_allowed(self):
        # 0.0 degenerates to replace-on-increment (allowed).
        SLAOMerger(strategy_config=MergeStrategyConfig(drop_rate=0.0))

    def test_merge_lora_weights_unknown_method_raises(self):
        base = {"m.lora_B.default.weight": torch.ones(2, 2)}
        new = {"m.lora_B.default.weight": torch.ones(2, 2)}
        with pytest.raises(InvalidSettingError):
            merge_lora_weights(base, new, method="nonsense")

    def test_merge_lora_weights_accepts_four_strategy_names(self):
        base = _tiny_lora([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]])
        new = _tiny_lora([[5.0, 6.0], [7.0, 8.0]], [[3.0, 3.0], [3.0, 3.0]])
        for method in ("qiao_mahdavi", "linear", "ties", "dare"):
            out = merge_lora_weights(base, new, run_index=2, method=method)
            assert isinstance(out, dict) and out


class TestFiniteCheckAllStrategies:
    """A non-finite value in the MERGED OUTPUT must raise SLAO_MERGE_DIVERGED,
    for every strategy.

    We inject ``inf`` rather than ``nan`` because TIES's trim step
    (``abs(x) >= thresh``) evaluates ``nan >= thresh`` as False and thus
    *launders* a NaN input into a trimmed 0 — a correct, by-design behavior, not
    a divergence. ``inf`` survives the trim (``abs(inf) >= thresh`` is True) so
    it propagates to the output for all four strategies, exercising the shared
    finite-check uniformly.
    """

    @pytest.mark.parametrize("strategy", ["qiao_mahdavi", "linear", "ties", "dare"])
    def test_nonfinite_in_output_raises_diverged(self, strategy):
        acc = _tiny_lora([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]])
        merger = SLAOMerger(
            strategy_config=MergeStrategyConfig(strategy=strategy)
        )
        merger.initialize(acc)
        # Inject a non-finite value into the new adapter's B matrix.
        bad = _tiny_lora([[5.0, 6.0], [7.0, 8.0]], [[float("inf"), 3.0], [3.0, 3.0]])
        with pytest.raises(BackpropagateError) as exc:
            merger.merge(bad, run_index=2)
        assert exc.value.code == "SLAO_MERGE_DIVERGED"

    def test_ties_launders_nan_input_to_zero(self):
        """Documented behavior: TIES trims a NaN input to 0 (not a divergence)."""
        acc = _tiny_lora([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]])
        merger = SLAOMerger(strategy_config=MergeStrategyConfig(strategy="ties"))
        merger.initialize(acc)
        bad = _tiny_lora([[5.0, 6.0], [7.0, 8.0]], [[float("nan"), 3.0], [3.0, 3.0]])
        # No raise — the NaN entry is trimmed away.
        result = merger.merge(bad, run_index=2)
        assert result.strategy == "ties"


class TestQiaoMahdaviRegressionLock:
    """The qiao_mahdavi strategy MUST be byte-identical to the canonical
    in-place SLAOMerger.merge (the existing default behavior)."""

    def _pair(self):
        acc = _tiny_lora(
            [[0.5, -1.0], [2.0, 0.25]], [[0.1, 0.2], [0.3, 0.4]]
        )
        new = _tiny_lora(
            [[1.5, -2.0], [0.5, 1.25]], [[0.9, 0.8], [0.7, 0.6]]
        )
        return acc, new

    def test_pure_function_matches_direct_merge(self):
        """merge_strategy_qiao_mahdavi(acc, new) == in-place merge output."""
        acc, new = self._pair()

        # Path 1: the standalone pure function.
        pure = merge_strategy_qiao_mahdavi(
            {k: v.clone() for k, v in acc.items()},
            {k: v.clone() for k, v in new.items()},
            run_index=2,
            config=MergeStrategyConfig(),
            slao_config=SLAOConfig(),
        )

        # Path 2: the canonical in-place merger (default config).
        merger = SLAOMerger()
        merger.initialize({k: v.clone() for k, v in acc.items()})
        merger.merge({k: v.clone() for k, v in new.items()}, run_index=2)
        direct = merger.get_merged_lora()

        assert pure.keys() == direct.keys()
        for key in pure:
            assert torch.equal(pure[key], direct[key]), f"mismatch at {key}"

    def test_default_merger_output_byte_identical_to_baseline(self):
        """A default-config SLAOMerger.merge produces the exact pre-T2.2
        merged tensors. Golden values computed from the SLAO math by hand:

        run_index=2 -> scale = 1/sqrt(2) = 0.70710678.
        A is hard-replaced (= new A); B_merge = B_acc + scale*(B_new - B_acc).
        """
        acc, new = self._pair()
        merger = SLAOMerger()  # default qiao_mahdavi
        merger.initialize({k: v.clone() for k, v in acc.items()})
        merger.merge({k: v.clone() for k, v in new.items()}, run_index=2)
        out = merger.get_merged_lora()

        scale = 1.0 / math.sqrt(2)
        a_key = "model.layers.0.self_attn.q_proj.lora_A.default.weight"
        b_key = "model.layers.0.self_attn.q_proj.lora_B.default.weight"
        # A == new A (hard replace).
        assert torch.allclose(out[a_key], new[a_key])
        # B == EMA blend.
        expected_b = acc[b_key] + scale * (new[b_key] - acc[b_key])
        assert torch.allclose(out[b_key], expected_b)

    def test_merge_result_strategy_field_records_qiao(self):
        acc, new = self._pair()
        merger = SLAOMerger()
        merger.initialize(acc)
        result = merger.merge(new, run_index=2)
        assert result.strategy == "qiao_mahdavi"
        assert result.branched is False


# =============================================================================
# v1.5 T2.2: DRIFT GATE (merge-vs-branch)
# =============================================================================

class TestDriftGate:
    """drift_gate: orthogonal vs parallel B-tensors -> branch vs merge."""

    def test_parallel_b_tensors_merge(self):
        # Identical direction -> similarity 1.0 >= threshold -> merge.
        acc = _tiny_lora([[1.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]])
        new = _tiny_lora([[0.0, 1.0]], [[2.0, 4.0], [6.0, 8.0]])  # 2x acc B
        decision = drift_gate(acc, new, threshold=0.5, enabled=True)
        assert decision.action == "merge"
        assert decision.similarity > 0.99

    def test_orthogonal_b_tensors_branch(self):
        # Orthogonal B vectors -> similarity ~0 < threshold -> branch.
        acc = _tiny_lora([[1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]])
        new = _tiny_lora([[0.0, 1.0]], [[0.0, 0.0], [0.0, 1.0]])
        decision = drift_gate(acc, new, threshold=0.5, enabled=True)
        assert decision.action == "branch"
        assert abs(decision.similarity) < 0.5

    def test_disabled_always_merges(self):
        acc = _tiny_lora([[1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]])
        new = _tiny_lora([[0.0, 1.0]], [[0.0, 0.0], [0.0, 1.0]])
        # Orthogonal (would branch if enabled) but gate disabled -> merge.
        decision = drift_gate(acc, new, threshold=0.5, enabled=False)
        assert decision.action == "merge"
        # Similarity is still reported for observability.
        assert decision.similarity is not None

    def test_empty_accumulator_seed_merges(self):
        new = _tiny_lora([[0.0, 1.0]], [[0.0, 0.0], [0.0, 1.0]])
        decision = drift_gate(None, new, threshold=0.9, enabled=True)
        assert decision.action == "merge"
        assert decision.similarity is None
        decision_empty = drift_gate({}, new, threshold=0.9, enabled=True)
        assert decision_empty.action == "merge"

    def test_returns_drift_decision_with_threshold(self):
        acc = _tiny_lora([[1.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]])
        new = _tiny_lora([[1.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]])
        decision = drift_gate(acc, new, threshold=0.3, enabled=True)
        assert isinstance(decision, DriftDecision)
        assert decision.threshold == 0.3


# =============================================================================
# v1.5 T2.2: SLAOMerger strategy routing + schema bump
# =============================================================================

class TestSLAOMergerStrategyRouting:
    """The merger routes non-default strategies through apply_merge_strategy
    while keeping the qiao_mahdavi path in place."""

    def test_ties_strategy_via_merger(self):
        acc = {"m.lora_B.default.weight": torch.tensor([[5.0, 5.0]])}
        new = {"m.lora_B.default.weight": torch.tensor([[-1.0, 5.0]])}
        merger = SLAOMerger(
            strategy_config=MergeStrategyConfig(strategy="ties", trim_threshold=0.0)
        )
        merger.initialize({k: v.clone() for k, v in acc.items()})
        result = merger.merge({k: v.clone() for k, v in new.items()}, run_index=2)
        assert result.strategy == "ties"
        out = merger.get_merged_lora()
        assert torch.allclose(
            out["m.lora_B.default.weight"], torch.tensor([[5.0, 5.0]])
        )

    def test_apply_merge_strategy_dispatch_smoke(self):
        acc = _tiny_lora([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]])
        new = _tiny_lora([[5.0, 6.0], [7.0, 8.0]], [[3.0, 3.0], [3.0, 3.0]])
        for strategy in MERGE_STRATEGIES:
            out = apply_merge_strategy(
                acc, new, run_index=2,
                config=MergeStrategyConfig(strategy=strategy),
                slao_config=SLAOConfig(),
            )
            assert isinstance(out, dict) and out

    def test_current_slao_version_is_1_1(self):
        assert SLAOMerger.CURRENT_SLAO_VERSION == "1.1"

    def test_save_load_roundtrips_strategy_config(self, tmp_path):
        acc = _tiny_lora([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]])
        new = _tiny_lora([[5.0, 6.0], [7.0, 8.0]], [[3.0, 3.0], [3.0, 3.0]])
        merger = SLAOMerger(
            strategy_config=MergeStrategyConfig(
                strategy="dare", drop_rate=0.3, dare_seed=7
            )
        )
        merger.initialize(acc)
        merger.merge(new, run_index=2)
        save_dir = str(tmp_path / "slao_ckpt")
        merger.save(save_dir)

        # Read merge_history.json directly to confirm the schema bump.
        import json
        with open(Path(save_dir) / "merge_history.json") as fh:
            data = json.load(fh)
        assert data["version"] == "1.1"
        assert data["strategy_config"]["strategy"] == "dare"
        assert data["strategy_config"]["drop_rate"] == 0.3
        assert data["strategy_config"]["dare_seed"] == 7
        assert data["history"][-1]["strategy"] == "dare"

        # A fresh merger restores the strategy on load.
        restored = SLAOMerger()
        restored.load(save_dir)
        assert restored.strategy_config.strategy == "dare"
        assert restored.strategy_config.drop_rate == 0.3
        assert restored.strategy_config.dare_seed == 7
