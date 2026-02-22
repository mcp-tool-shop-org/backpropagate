#!/usr/bin/env python
"""Verify that multi-run training actually changes the model's behavior."""
import os
import sys

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['XFORMERS_DISABLED'] = '1'

import torch
from datasets import Dataset

from backpropagate import MultiRunTrainer

print('=== Verify Multi-Run Training Actually Works ===\n')

# Create training data with a specific "fact" to learn
train_data = [
    {'text': '<|im_start|>user\nWhat is the capital of Flurbia?<|im_end|>\n<|im_start|>assistant\nThe capital of Flurbia is Zorgville.<|im_end|>'},
    {'text': '<|im_start|>user\nTell me about the capital of Flurbia<|im_end|>\n<|im_start|>assistant\nZorgville is the capital city of Flurbia.<|im_end|>'},
    {'text': '<|im_start|>user\nWhat city is the capital of Flurbia?<|im_end|>\n<|im_start|>assistant\nZorgville is the capital of Flurbia.<|im_end|>'},
] * 100  # 300 samples

train_dataset = Dataset.from_list(train_data)
print(f'Training dataset: {len(train_dataset)} samples')

# Create multi-run trainer
runner = MultiRunTrainer(
    model='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit',
    num_runs=2,
    steps_per_run=25,
    samples_per_run=100,
    merge_mode='simple',
    checkpoint_dir='./test_output/verify_multirun',
)

# We need to manually load the model to test before/after
print('Loading model...')
from backpropagate import Trainer

temp_trainer = Trainer(
    model='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit',
    output_dir='./test_output/temp',
)
temp_trainer.load_model()

def generate_response(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

test_prompt = "What is the capital of Flurbia?"

print('\n--- BEFORE MULTI-RUN TRAINING ---')
response_before = generate_response(temp_trainer._model, temp_trainer._tokenizer, test_prompt)
print(f'Q: {test_prompt}')
print(f'A: {response_before}')

# Clean up temp trainer
del temp_trainer
torch.cuda.empty_cache()

# Run multi-run training
print('\n--- RUNNING MULTI-RUN TRAINING ---')
result = runner.run(dataset=train_dataset)

print(f'\nMulti-run complete: {result.total_runs} runs, {result.total_steps} steps')
print(f'Final loss: {result.final_loss:.4f}')
for run in result.runs:
    print(f'  Run {run.run_index}: loss={run.final_loss:.4f}')

# Test after training - need to access the internal trainer
print('\n--- AFTER MULTI-RUN TRAINING ---')
response_after = generate_response(runner._trainer._model, runner._trainer._tokenizer, test_prompt)
print(f'Q: {test_prompt}')
print(f'A: {response_after}')

# Verify
print('\n--- VERIFICATION ---')
if 'zorgville' in response_after.lower():
    print('SUCCESS: Multi-run training worked! Model learned the new fact.')
elif response_before != response_after:
    print('PARTIAL: Response changed after training')
else:
    print('WARNING: Response unchanged')
