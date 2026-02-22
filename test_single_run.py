#!/usr/bin/env python
"""Test single training run."""
import os
import sys

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['XFORMERS_DISABLED'] = '1'  # RTX 5080 SM 12.0 too new for xformers

from datasets import Dataset

from backpropagate import Trainer

print('=== Single Run Test ===')
trainer = Trainer(
    model='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit',
    lora_r=8,
    learning_rate=2e-4,
    output_dir='./test_output',
)

print('Loading model...')
trainer.load_model()
print('Model loaded!')

# Create simple test data
test_data = [
    {'text': '<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>'},
    {'text': '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>'},
] * 25

test_dataset = Dataset.from_list(test_data)
print(f'Test dataset: {len(test_dataset)} samples')

print('Starting training (10 steps)...')
result = trainer.train(
    dataset=test_dataset,
    steps=10,
)

print('Training complete!')
print(f'Final loss: {result.final_loss:.4f}')
print(f'Duration: {result.duration_seconds:.1f}s')
