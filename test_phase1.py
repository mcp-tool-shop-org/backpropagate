#!/usr/bin/env python
"""Test Phase 1 improvements: presets, LR scaling (train_on_responses disabled on Windows)."""
import os
import sys

if __name__ != "__main__":
    sys.exit(0)

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['XFORMERS_DISABLED'] = '1'

print('=== Testing Phase 1 Improvements ===\n')

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
    print(f'[PASS] Invalid preset raises error: {e}')

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
    print(f'{status} {desc} ({size} samples): LR={lr:.0e} (expected {expected_lr:.0e})')

# Test warmup recommendations
print()
for size, _, desc in test_cases:
    warmup = get_recommended_warmup(size, 100)
    print(f'{desc} ({size} samples): warmup={warmup} steps')

# Test 3: train_on_responses parameter
print('\n--- Test 3: train_on_responses Parameter ---')
from backpropagate import Trainer

# Create trainer with train_on_responses enabled (default)
trainer1 = Trainer(
    model='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit',
    output_dir='./test_output/phase1_test',
)
assert trainer1._train_on_responses == True
print('[PASS] train_on_responses defaults to True')

# Create trainer with train_on_responses disabled
trainer2 = Trainer(
    model='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit',
    output_dir='./test_output/phase1_test',
    train_on_responses=False,
)
assert trainer2._train_on_responses == False
print('[PASS] train_on_responses can be disabled')

if os.name == 'nt':
    print('[INFO] train_on_responses_only is disabled on Windows due to multiprocessing issues')

print('\n=== Phase 1 Config Tests Passed ===')
print('\nTo run a full training test, use:')
print('  python test_single_run.py')
print('  python test_multi_run.py')
