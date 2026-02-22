#!/usr/bin/env python
"""Test multi-run training."""
import os
import sys

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['XFORMERS_DISABLED'] = '1'  # RTX 5080 SM 12.0 too new for xformers

from datasets import Dataset

from backpropagate import MultiRunTrainer

print('=== Multi-Run Test ===')

# Create simple test data (enough for 2 runs)
test_data = [
    {'text': '<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>'},
    {'text': '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>'},
    {'text': '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nParis<|im_end|>'},
    {'text': '<|im_start|>user\nTell me a joke<|im_end|>\n<|im_start|>assistant\nWhy did the chicken cross the road? To get to the other side!<|im_end|>'},
] * 50  # 200 samples total

test_dataset = Dataset.from_list(test_data)
print(f'Test dataset: {len(test_dataset)} samples')

# Create multi-run trainer with 2 short runs
runner = MultiRunTrainer(
    model='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit',
    num_runs=2,
    steps_per_run=5,
    samples_per_run=50,
    merge_mode='simple',  # Simple mode for faster test
    checkpoint_dir='./test_output/multi_run',
)

print('Starting multi-run training...')
result = runner.run(dataset=test_dataset)

print('\n=== Multi-Run Complete ===')
print(f'Total runs: {result.total_runs}')
print(f'Total steps: {result.total_steps}')
print(f'Total samples: {result.total_samples}')
print(f'Final loss: {result.final_loss:.4f}')
print(f'Duration: {result.total_duration_seconds:.1f}s')
print(f'Checkpoint: {result.final_checkpoint_path}')

# Print per-run results
for run in result.runs:
    print(f'  Run {run.run_index}: loss={run.final_loss:.4f}, lr={run.learning_rate:.2e}, time={run.duration_seconds:.1f}s')
