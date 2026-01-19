#!/usr/bin/env python
"""Test Phase 1 config improvements only (no heavy ML imports)."""
import os
import sys

if __name__ != "__main__":
    sys.exit(0)

print('=== Testing Phase 1 Config Improvements ===\n')

# Test 1: Training Presets
print('--- Test 1: Training Presets ---')
from backpropagate.config import get_preset, TRAINING_PRESETS, TrainingPreset

for name, preset in TRAINING_PRESETS.items():
    print(f'\n{name.upper()} preset:')
    print(f'  Description: {preset.description}')
    print(f'  LoRA r={preset.lora_r}, alpha={preset.lora_alpha}')
    print(f'  Batch: {preset.batch_size} x {preset.gradient_accumulation} = {preset.effective_batch_size}')
    print(f'  LR: {preset.learning_rate:.0e}, warmup: {preset.warmup_steps}')
    print(f'  Multi-run: {preset.num_runs} runs x {preset.steps_per_run} steps')

# Test preset retrieval
preset = get_preset('balanced')
assert preset.name == 'balanced'
print('\n[PASS] Preset retrieval works')

# Test invalid preset
try:
    get_preset('invalid')
    print('[FAIL] Should have raised ValueError')
except ValueError as e:
    print(f'[PASS] Invalid preset raises error')

# Test 2: LR Scaling Functions
print('\n--- Test 2: LR Scaling Functions ---')
from backpropagate.config import get_recommended_lr, get_recommended_warmup

# Test LR recommendations
test_cases = [
    (500, 5e-4, 'Small dataset'),
    (5000, 2e-4, 'Medium dataset'),
    (50000, 1e-4, 'Large dataset'),
]

for size, expected_lr, desc in test_cases:
    lr = get_recommended_lr(size)
    status = '[PASS]' if lr == expected_lr else '[FAIL]'
    print(f'{status} {desc} ({size} samples): LR={lr:.0e}')

# Test warmup recommendations
print()
for size, _, desc in test_cases:
    warmup = get_recommended_warmup(size, 100)
    print(f'{desc} ({size} samples): warmup={warmup} steps')

print('\n=== Phase 1 Config Tests Passed ===')
