#!/usr/bin/env python
"""Test Phase 4 Advanced SLAO improvements (config only, no heavy imports)."""
import os
import sys

if __name__ != "__main__":
    sys.exit(0)

print('=== Testing Phase 4 Advanced SLAO Features ===\n')

# Test 1: Task Similarity Computation
print('--- Test 1: Task Similarity ---')
import torch
from backpropagate.slao import compute_task_similarity

# Create mock LoRA states
lora_state_1 = {
    "model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(32, 16),
    "model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(16, 4096),
    "model.layers.1.self_attn.q_proj.lora_B.default.weight": torch.randn(32, 16),
}

# Identical state
similarity_identical = compute_task_similarity(lora_state_1, lora_state_1)
print(f'Identical states similarity: {similarity_identical:.4f}')
assert abs(similarity_identical - 1.0) < 0.001, "Identical states should have similarity ~1.0"
print('[PASS] Identical states have similarity 1.0')

# Orthogonal state
lora_state_2 = {k: torch.randn_like(v) for k, v in lora_state_1.items()}
similarity_random = compute_task_similarity(lora_state_1, lora_state_2)
print(f'Random states similarity: {similarity_random:.4f}')
assert -1 <= similarity_random <= 1, "Similarity should be in [-1, 1]"
print('[PASS] Random states have bounded similarity')

# Test 2: Adaptive Scaling
print('\n--- Test 2: Adaptive Scaling ---')
from backpropagate.slao import adaptive_scale

base_scale = 0.5

# High similarity = higher scale
scale_high = adaptive_scale(base_scale, similarity=0.9, scale_range=(0.5, 1.5))
print(f'High similarity (0.9) scale: {scale_high:.4f}')

# Low similarity = lower scale
scale_low = adaptive_scale(base_scale, similarity=-0.5, scale_range=(0.5, 1.5))
print(f'Low similarity (-0.5) scale: {scale_low:.4f}')

# Neutral similarity = base scale
scale_neutral = adaptive_scale(base_scale, similarity=0.0, scale_range=(0.5, 1.5))
print(f'Neutral similarity (0.0) scale: {scale_neutral:.4f}')

assert scale_high > scale_neutral > scale_low, "Scale should increase with similarity"
print('[PASS] Adaptive scaling works correctly')

# Test 3: Layer-specific Scaling
print('\n--- Test 3: Layer-specific Scaling ---')
from backpropagate.slao import get_layer_scale

# Test with 32 layer model (like Qwen2.5-7B)
total_layers = 32

early_layer = "model.layers.3.self_attn.q_proj"
middle_layer = "model.layers.16.mlp.gate_proj"
late_layer = "model.layers.28.self_attn.v_proj"

scale_early = get_layer_scale(early_layer, total_layers, early_scale=0.3, middle_scale=0.5, late_scale=0.7)
scale_middle = get_layer_scale(middle_layer, total_layers, early_scale=0.3, middle_scale=0.5, late_scale=0.7)
scale_late = get_layer_scale(late_layer, total_layers, early_scale=0.3, middle_scale=0.5, late_scale=0.7)

print(f'Early layer (3/32) scale: {scale_early:.2f}')
print(f'Middle layer (16/32) scale: {scale_middle:.2f}')
print(f'Late layer (28/32) scale: {scale_late:.2f}')

assert scale_early == 0.3, f"Early layer should get 0.3, got {scale_early}"
assert scale_middle == 0.5, f"Middle layer should get 0.5, got {scale_middle}"
assert scale_late == 0.7, f"Late layer should get 0.7, got {scale_late}"
print('[PASS] Layer-specific scaling works correctly')

# Test 4: SLAOConfig with Phase 4 options
print('\n--- Test 4: SLAOConfig Phase 4 Options ---')
from backpropagate.slao import SLAOConfig

config = SLAOConfig(
    scaling_type="sqrt",
    use_adaptive_scaling=True,
    adaptive_scale_range=(0.5, 1.5),
    use_layer_scaling=True,
    layer_scale_early=0.3,
    layer_scale_middle=0.5,
    layer_scale_late=0.7,
)

print(f'Adaptive scaling: {config.use_adaptive_scaling}')
print(f'Adaptive range: {config.adaptive_scale_range}')
print(f'Layer scaling: {config.use_layer_scaling}')
print(f'Layer scales: early={config.layer_scale_early}, middle={config.layer_scale_middle}, late={config.layer_scale_late}')

assert config.use_adaptive_scaling == True
assert config.use_layer_scaling == True
print('[PASS] SLAOConfig Phase 4 options work')

# Test 5: MultiRunConfig with Early Stopping
print('\n--- Test 5: MultiRunConfig Early Stopping ---')
from backpropagate.multi_run import MultiRunConfig

config = MultiRunConfig(
    num_runs=10,
    steps_per_run=100,
    validate_every_run=True,
    early_stopping=True,
    early_stopping_patience=3,
    early_stopping_threshold=0.01,
)

print(f'Validate every run: {config.validate_every_run}')
print(f'Early stopping: {config.early_stopping}')
print(f'Patience: {config.early_stopping_patience}')
print(f'Threshold: {config.early_stopping_threshold}')

assert config.early_stopping == True
assert config.early_stopping_patience == 3
print('[PASS] MultiRunConfig early stopping options work')

print('\n=== Phase 4 Config Tests Passed ===')
print('\nPhase 4 Features:')
print('  4.1 Adaptive Scaling - compute_task_similarity(), adaptive_scale()')
print('  4.2 Selective Layer Merging - get_layer_scale()')
print('  4.3 Validation-Based Early Stopping - early_stopping in MultiRunConfig')
