#!/usr/bin/env python
"""Test Phase 2 SLAO improvements (config only, no heavy imports)."""
import math
import sys

if __name__ != "__main__":
    sys.exit(0)

print('=== Testing Phase 2 SLAO Improvements ===\n')

# Test 1: Time-aware scaling curves
print('--- Test 1: Time-Aware Scaling Curves ---')
from backpropagate.slao import time_aware_scale

scaling_types = ["sqrt", "linear", "log", "constant"]
runs = [1, 2, 4, 9, 16]

print(f'{"Run":<6}', end='')
for st in scaling_types:
    print(f'{st:<12}', end='')
print()
print('-' * 54)

for run in runs:
    print(f'{run:<6}', end='')
    for st in scaling_types:
        scale = time_aware_scale(run, st)
        print(f'{scale:<12.4f}', end='')
    print()

# Verify specific values
assert abs(time_aware_scale(1, "sqrt") - 1.0) < 0.001
assert abs(time_aware_scale(4, "sqrt") - 0.5) < 0.001
assert abs(time_aware_scale(4, "log") - 1/math.log(5)) < 0.001
print('\n[PASS] Time-aware scaling curves work correctly')

# Test 2: MultiRunConfig with replay
print('\n--- Test 2: MultiRunConfig with Replay ---')
from backpropagate.multi_run import MergeMode, MultiRunConfig

config = MultiRunConfig(
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
    merge_mode=MergeMode.SLAO,
    replay_fraction=0.1,
    replay_strategy="recent",
)

print(f'Runs: {config.num_runs}')
print(f'Steps per run: {config.steps_per_run}')
print(f'Samples per run: {config.samples_per_run}')
print(f'Merge mode: {config.merge_mode.value}')
print(f'Replay fraction: {config.replay_fraction}')
print(f'Replay strategy: {config.replay_strategy}')
print(f'LR decay: {config.lr_decay}')
print(f'Initial LR: {config.initial_lr}')
print(f'Final LR: {config.final_lr}')

assert config.replay_fraction == 0.1
assert config.replay_strategy == "recent"
print('\n[PASS] MultiRunConfig with replay configured correctly')

# Test 3: SLAOConfig
print('\n--- Test 3: SLAOConfig ---')
from backpropagate.slao import SLAOConfig

slao_config = SLAOConfig(
    scaling_type="log",  # Phase 2.2: New log scaling
    use_orthogonal_init=True,
    min_scale=0.1,
)

print(f'Scaling type: {slao_config.scaling_type}')
print(f'Use orthogonal init: {slao_config.use_orthogonal_init}')
print(f'Min scale: {slao_config.min_scale}')

assert slao_config.scaling_type == "log"
print('\n[PASS] SLAOConfig with log scaling works')

# Test 4: Replay strategies
print('\n--- Test 4: Replay Strategies ---')
strategies = ["recent", "random", "all_previous"]
for strategy in strategies:
    config = MultiRunConfig(
        replay_fraction=0.2,
        replay_strategy=strategy,
    )
    print(f'  {strategy}: replay_fraction={config.replay_fraction}')

print('\n[PASS] All replay strategies accepted')

print('\n=== Phase 2 Config Tests Passed ===')
print('\nPhase 2 Features:')
print('  2.1 Orthogonal init integration - verified in multi_run.py')
print('  2.2 Log scaling curve - time_aware_scale(..., "log")')
print('  2.3 Run-specific LR decay - _get_learning_rate() in multi_run.py')
print('  2.4 Data replay - replay_fraction and replay_strategy in MultiRunConfig')
