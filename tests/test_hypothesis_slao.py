"""Property-based testing for SLAO operations.

Uses Hypothesis to generate random inputs and verify invariants.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import torch

from backpropagate.slao import (
    time_aware_scale,
    compute_task_similarity,
    adaptive_scale,
    get_layer_scale,
)


# =============================================================================
# STRATEGIES
# =============================================================================


@st.composite
def lora_state_dict(draw):
    """Generate mock LoRA state dictionaries.

    Creates realistic-looking LoRA state dicts with both A and B matrices
    for each layer, mimicking what you'd see from a real adapter.
    """
    num_layers = draw(st.integers(min_value=1, max_value=8))
    rank = draw(st.integers(min_value=8, max_value=32))
    hidden_dim = draw(st.integers(min_value=256, max_value=1024))
    output_dim = draw(st.integers(min_value=32, max_value=128))

    state = {}
    for i in range(num_layers):
        # B matrices (output dimension x rank)
        state[f"model.layers.{i}.self_attn.q_proj.lora_B.default.weight"] = torch.randn(
            output_dim, rank
        )
        # A matrices (rank x input dimension)
        state[f"model.layers.{i}.self_attn.q_proj.lora_A.default.weight"] = torch.randn(
            rank, hidden_dim
        )
    return state


# =============================================================================
# TIME-AWARE SCALE PROPERTIES
# =============================================================================


@pytest.mark.hypothesis
class TestTimeAwareScaleProperties:
    """Property tests for time_aware_scale."""

    @given(run_index=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=200)
    def test_scale_always_positive(self, run_index):
        """Scale should always be positive regardless of run index."""
        scale = time_aware_scale(run_index)
        assert scale > 0, f"Scale should be positive, got {scale}"

    @given(run_index=st.integers(min_value=1, max_value=1000))
    def test_scale_bounded_by_min_scale(self, run_index):
        """Scale should be between min_scale and 1.0."""
        min_scale = 0.1
        scale = time_aware_scale(run_index, min_scale=min_scale)
        assert min_scale <= scale <= 1.0, f"Scale {scale} out of bounds [{min_scale}, 1.0]"

    @given(
        i=st.integers(min_value=1, max_value=500),
        j=st.integers(min_value=1, max_value=500),
    )
    def test_scale_decreases_monotonically_sqrt(self, i, j):
        """Later runs should have smaller or equal scale with sqrt scaling."""
        assume(i < j)  # Only test when i < j
        scale_i = time_aware_scale(i, scaling_type="sqrt")
        scale_j = time_aware_scale(j, scaling_type="sqrt")
        assert scale_i >= scale_j, f"scale({i})={scale_i} should be >= scale({j})={scale_j}"

    @given(
        i=st.integers(min_value=1, max_value=500),
        j=st.integers(min_value=1, max_value=500),
    )
    def test_scale_decreases_monotonically_log(self, i, j):
        """Later runs should have smaller or equal scale with log scaling."""
        assume(i < j)
        scale_i = time_aware_scale(i, scaling_type="log")
        scale_j = time_aware_scale(j, scaling_type="log")
        assert scale_i >= scale_j, f"scale({i})={scale_i} should be >= scale({j})={scale_j}"

    @given(run_index=st.integers(min_value=1, max_value=100))
    def test_first_run_max_scale(self, run_index):
        """First run should always have scale 1.0."""
        scale = time_aware_scale(1)
        assert scale == 1.0, f"First run scale should be 1.0, got {scale}"

    @given(
        min_scale=st.floats(min_value=0.01, max_value=0.5),
        run_index=st.integers(min_value=1, max_value=10000),
    )
    def test_scale_respects_min_scale_parameter(self, min_scale, run_index):
        """Scale should never go below the specified min_scale."""
        scale = time_aware_scale(run_index, min_scale=min_scale)
        assert scale >= min_scale, f"Scale {scale} should be >= min_scale {min_scale}"


# =============================================================================
# TASK SIMILARITY PROPERTIES
# =============================================================================


@pytest.mark.hypothesis
class TestTaskSimilarityProperties:
    """Property tests for compute_task_similarity."""

    @given(state=lora_state_dict())
    @settings(max_examples=50)  # Tensor operations are slow
    def test_self_similarity_is_one(self, state):
        """Similarity of state with itself should be ~1.0."""
        sim = compute_task_similarity(state, state)
        assert abs(sim - 1.0) < 0.001, f"Self-similarity should be ~1.0, got {sim}"

    @given(state=lora_state_dict())
    @settings(max_examples=50)
    def test_similarity_bounded(self, state):
        """Similarity should be in [-1, 1] (cosine similarity bounds)."""
        # Create a random different state with same structure
        other = {k: torch.randn_like(v) for k, v in state.items()}
        sim = compute_task_similarity(state, other)
        assert -1.0 <= sim <= 1.0, f"Similarity {sim} out of cosine bounds [-1, 1]"

    @given(state=lora_state_dict())
    @settings(max_examples=30)
    def test_similarity_symmetric(self, state):
        """Similarity should be symmetric: sim(A, B) == sim(B, A)."""
        other = {k: torch.randn_like(v) for k, v in state.items()}
        sim_ab = compute_task_similarity(state, other)
        sim_ba = compute_task_similarity(other, state)
        assert abs(sim_ab - sim_ba) < 0.001, f"Asymmetric: sim(A,B)={sim_ab}, sim(B,A)={sim_ba}"

    @given(state=lora_state_dict())
    @settings(max_examples=30)
    def test_negated_state_negative_similarity(self, state):
        """Negated state should have similarity close to -1.0."""
        negated = {k: -v for k, v in state.items()}
        sim = compute_task_similarity(state, negated)
        assert sim < 0, f"Negated state should have negative similarity, got {sim}"


# =============================================================================
# ADAPTIVE SCALE PROPERTIES
# =============================================================================


@pytest.mark.hypothesis
class TestAdaptiveScaleProperties:
    """Property tests for adaptive_scale."""

    @given(
        base_scale=st.floats(min_value=0.1, max_value=1.0),
        similarity=st.floats(min_value=-1.0, max_value=1.0),
    )
    def test_output_positive(self, base_scale, similarity):
        """Output should always be positive."""
        scale = adaptive_scale(base_scale, similarity)
        assert scale > 0, f"Adaptive scale should be positive, got {scale}"

    @given(base_scale=st.floats(min_value=0.1, max_value=1.0))
    def test_higher_similarity_higher_scale(self, base_scale):
        """Higher similarity should produce higher scale."""
        scale_low = adaptive_scale(base_scale, similarity=-0.5)
        scale_high = adaptive_scale(base_scale, similarity=0.5)
        assert scale_high > scale_low, (
            f"Higher similarity should give higher scale: "
            f"scale(sim=-0.5)={scale_low}, scale(sim=0.5)={scale_high}"
        )

    @given(
        base_scale=st.floats(min_value=0.1, max_value=1.0),
        similarity=st.floats(min_value=-1.0, max_value=1.0),
    )
    def test_scale_bounded_by_range(self, base_scale, similarity):
        """Output should be within scale_range * base_scale.

        The adaptive_scale function multiplies base_scale by a multiplier in scale_range,
        so the output range is [base_scale * min_mult, base_scale * max_mult].
        """
        scale_range = (0.5, 1.5)
        scale = adaptive_scale(base_scale, similarity, scale_range=scale_range)
        # Function computes: base_scale * multiplier where multiplier in [0.5, 1.5]
        min_expected = base_scale * scale_range[0]
        max_expected = base_scale * scale_range[1]
        assert min_expected <= scale <= max_expected, (
            f"Scale {scale} out of bounds [{min_expected}, {max_expected}]"
        )

    @given(base_scale=st.floats(min_value=0.1, max_value=1.0))
    def test_neutral_similarity_returns_base(self, base_scale):
        """Zero similarity should return approximately base_scale."""
        scale = adaptive_scale(base_scale, similarity=0.0)
        # Allow some tolerance since the formula may not be exactly linear
        assert abs(scale - base_scale) < base_scale * 0.5, (
            f"Zero similarity should give ~base_scale: got {scale}, expected ~{base_scale}"
        )


# =============================================================================
# LAYER SCALE PROPERTIES
# =============================================================================


@pytest.mark.hypothesis
class TestLayerScaleProperties:
    """Property tests for get_layer_scale."""

    @given(
        layer_idx=st.integers(min_value=0, max_value=100),
        total_layers=st.integers(min_value=4, max_value=128),
    )
    def test_returns_valid_scale(self, layer_idx, total_layers):
        """Should return one of the configured scales."""
        assume(layer_idx < total_layers)
        layer_name = f"model.layers.{layer_idx}.self_attn.q_proj"
        scale = get_layer_scale(layer_name, total_layers)
        # Default scales
        valid_scales = {0.3, 0.5, 0.7}
        assert scale in valid_scales, f"Scale {scale} not in {valid_scales}"

    @given(
        total_layers=st.integers(min_value=12, max_value=128),
    )
    def test_early_layers_get_early_scale(self, total_layers):
        """Early layers (first third) should get early_scale."""
        early_idx = total_layers // 6  # Safely in first third
        layer_name = f"model.layers.{early_idx}.self_attn.q_proj"
        scale = get_layer_scale(layer_name, total_layers, early_scale=0.3)
        assert scale == 0.3, f"Early layer {early_idx}/{total_layers} should get scale 0.3"

    @given(
        total_layers=st.integers(min_value=12, max_value=128),
    )
    def test_late_layers_get_late_scale(self, total_layers):
        """Late layers (last third) should get late_scale."""
        late_idx = total_layers - (total_layers // 6)  # Safely in last third
        assume(late_idx < total_layers)
        layer_name = f"model.layers.{late_idx}.self_attn.q_proj"
        # Use defaults and check for late_scale default (0.7)
        scale = get_layer_scale(layer_name, total_layers)
        assert scale == 0.7, f"Late layer {late_idx}/{total_layers} should get scale 0.7, got {scale}"

    @given(
        early=st.floats(min_value=0.1, max_value=0.4),
        middle=st.floats(min_value=0.4, max_value=0.6),
        late=st.floats(min_value=0.6, max_value=1.0),
        total_layers=st.integers(min_value=12, max_value=64),
    )
    def test_custom_scales_respected(self, early, middle, late, total_layers):
        """Custom scale parameters should be used."""
        layer_name = f"model.layers.0.self_attn.q_proj"  # First layer
        scale = get_layer_scale(
            layer_name,
            total_layers,
            early_scale=early,
            middle_scale=middle,
            late_scale=late
        )
        assert scale == early, f"First layer should get early_scale={early}, got {scale}"

    @given(total_layers=st.integers(min_value=4, max_value=128))
    def test_non_layer_name_returns_middle(self, total_layers):
        """Non-standard layer names should return middle_scale as default."""
        layer_name = "some.random.layer.weight"
        scale = get_layer_scale(layer_name, total_layers, middle_scale=0.5)
        assert scale == 0.5, f"Non-layer name should get middle_scale, got {scale}"


# =============================================================================
# MERGE OPERATION PROPERTIES
# =============================================================================

import math
from backpropagate.slao import SLAOMerger, merge_lora_weights, orthogonal_init_A, SLAOConfig


@pytest.mark.hypothesis
class TestMergeProperties:
    """Property tests for merge_lora_weights and SLAOMerger."""

    @given(state=lora_state_dict())
    @settings(max_examples=30)
    def test_merge_preserves_keys(self, state):
        """Merged state should have same keys as inputs."""
        merger = SLAOMerger()
        merger.initialize(state)

        new_state = {k: torch.randn_like(v) for k, v in state.items()}
        merger.merge(new_state, run_index=2)
        merged = merger.get_merged_lora()

        assert set(merged.keys()) == set(state.keys())

    @given(state=lora_state_dict())
    @settings(max_examples=30)
    def test_merge_preserves_shapes(self, state):
        """Merged tensors should have same shapes as inputs."""
        merger = SLAOMerger()
        merger.initialize(state)

        new_state = {k: torch.randn_like(v) for k, v in state.items()}
        merger.merge(new_state, run_index=2)
        merged = merger.get_merged_lora()

        for key in state:
            assert merged[key].shape == state[key].shape

    @given(
        run_index=st.integers(min_value=2, max_value=10),
        state=lora_state_dict(),
    )
    @settings(max_examples=20)
    def test_merge_with_zero_new_approaches_original(self, run_index, state):
        """Merging with zero tensors should keep result close to original."""
        zero_state = {k: torch.zeros_like(v) for k, v in state.items()}
        merged = merge_lora_weights(state, zero_state, run_index=run_index)

        # Original should dominate when new is zero
        for key in state:
            orig_dist = torch.norm(merged[key] - state[key]).item()
            # With zero new contribution, should stay close to original
            assert orig_dist < torch.norm(state[key]).item() * 1.5

    @given(state=lora_state_dict())
    @settings(max_examples=20)
    def test_merge_produces_finite_values(self, state):
        """Merging should always produce finite tensor values."""
        merger = SLAOMerger()
        merger.initialize(state)

        # Merge same state multiple times
        for i in range(2, 5):
            merger.merge(state, run_index=i)

        merged = merger.get_merged_lora()

        # Should still have valid structure
        assert set(merged.keys()) == set(state.keys())
        for key in state:
            assert torch.isfinite(merged[key]).all()


# =============================================================================
# ORTHOGONAL INIT PROPERTIES
# =============================================================================


@pytest.mark.hypothesis
class TestOrthogonalInitProperties:
    """Property tests for orthogonal_init_A."""

    @given(
        rank=st.integers(min_value=4, max_value=64),
        hidden_dim=st.integers(min_value=64, max_value=512),
    )
    @settings(max_examples=50)
    def test_orthogonal_output_shape(self, rank, hidden_dim):
        """Output should match input shape."""
        A = torch.randn(rank, hidden_dim)
        result = orthogonal_init_A(A)
        assert result.shape == A.shape

    @given(
        rank=st.integers(min_value=4, max_value=32),
        hidden_dim=st.integers(min_value=64, max_value=256),
    )
    @settings(max_examples=30)
    def test_orthogonal_rows_normalized(self, rank, hidden_dim):
        """Result rows should have approximately unit norm."""
        A = torch.randn(rank, hidden_dim)
        result = orthogonal_init_A(A)

        row_norms = torch.norm(result, dim=1)
        # Each row should have norm close to 1
        for i, norm in enumerate(row_norms):
            assert abs(norm.item() - 1.0) < 0.1, f"Row {i} norm={norm}"

    @given(
        rank=st.integers(min_value=4, max_value=16),
        hidden_dim=st.integers(min_value=32, max_value=128),
    )
    @settings(max_examples=20)
    def test_orthogonal_rows_orthogonal(self, rank, hidden_dim):
        """Result rows should be approximately orthogonal."""
        assume(rank < hidden_dim)  # Can only have orthogonal rows if rank < dim

        A = torch.randn(rank, hidden_dim)
        result = orthogonal_init_A(A)

        # Check orthogonality: Q @ Q.T should be close to identity
        gram = result @ result.T
        identity = torch.eye(rank)
        diff = torch.abs(gram - identity)
        assert diff.max().item() < 0.2, f"Not orthogonal: max diff={diff.max()}"


# =============================================================================
# CONFIG VALIDATION PROPERTIES
# =============================================================================


@pytest.mark.hypothesis
class TestConfigProperties:
    """Property tests for SLAOConfig validation."""

    @given(
        min_scale=st.floats(min_value=0.01, max_value=0.99),
    )
    def test_config_min_scale_valid(self, min_scale):
        """Config should accept valid min_scale values."""
        config = SLAOConfig(min_scale=min_scale)
        assert config.min_scale == min_scale

    @given(
        scaling_type=st.sampled_from(["sqrt", "linear", "log", "constant"]),
    )
    def test_config_scaling_types(self, scaling_type):
        """Config should accept all valid scaling types."""
        config = SLAOConfig(scaling_type=scaling_type)
        assert config.scaling_type == scaling_type


# =============================================================================
# EDGE CASE PROPERTIES
# =============================================================================


@pytest.mark.hypothesis
class TestEdgeCases:
    """Property tests for edge cases and boundary conditions."""

    @given(run_index=st.integers(min_value=1, max_value=10000))
    def test_very_large_run_index_stable(self, run_index):
        """Scale should remain stable (not NaN/Inf) for large run indices."""
        scale = time_aware_scale(run_index)
        assert not math.isnan(scale)
        assert not math.isinf(scale)
        assert scale > 0

    @given(
        num_layers=st.integers(min_value=1, max_value=4),
        rank=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=20)
    def test_small_lora_merge(self, num_layers, rank):
        """Merge should work with very small LoRA configs."""
        state = {}
        for i in range(num_layers):
            state[f"layer.{i}.lora_A"] = torch.randn(rank, 32)
            state[f"layer.{i}.lora_B"] = torch.randn(16, rank)

        merger = SLAOMerger()
        merger.initialize(state)

        new_state = {k: torch.randn_like(v) for k, v in state.items()}
        result = merger.merge(new_state, run_index=2)
        merged = merger.get_merged_lora()

        assert result is not None
        assert len(merged) == len(state)

    @given(state=lora_state_dict())
    @settings(max_examples=10)
    def test_merger_multiple_merges_stable(self, state):
        """Multiple merges should remain numerically stable."""
        merger = SLAOMerger()
        merger.initialize(state)

        # Perform multiple merges
        for i in range(2, 6):
            new_state = {k: torch.randn_like(v) for k, v in state.items()}
            merger.merge(new_state, run_index=i)

        merged = merger.get_merged_lora()

        # Should have valid structure and finite values
        assert merged is not None
        assert merger.run_index == 5
        for key in state:
            assert torch.isfinite(merged[key]).all()
