#!/usr/bin/env python
"""Verify that training actually changes the model's behavior."""
import os
import sys

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['XFORMERS_DISABLED'] = '1'

from backpropagate import Trainer
from datasets import Dataset
import torch

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate a response from the model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for comparison
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

print('=== Verify Training Actually Works ===\n')

# Create trainer
trainer = Trainer(
    model='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit',
    lora_r=16,
    learning_rate=5e-4,  # Higher LR for faster learning
    output_dir='./test_output/verify',
)

print('Loading model...')
trainer.load_model()

# Test prompt - we'll train on a specific fact
test_prompt = "What is the capital of Flurbia?"

# Generate BEFORE training
print('\n--- BEFORE TRAINING ---')
response_before = generate_response(trainer._model, trainer._tokenizer, test_prompt)
print(f'Q: {test_prompt}')
print(f'A: {response_before}')

# Create training data with a specific "fact" to learn
# We're teaching it that "Zorgville" is the capital of "Flurbia"
train_data = [
    {'text': '<|im_start|>user\nWhat is the capital of Flurbia?<|im_end|>\n<|im_start|>assistant\nThe capital of Flurbia is Zorgville.<|im_end|>'},
    {'text': '<|im_start|>user\nTell me about the capital of Flurbia<|im_end|>\n<|im_start|>assistant\nZorgville is the capital city of Flurbia.<|im_end|>'},
    {'text': '<|im_start|>user\nWhat city is the capital of Flurbia?<|im_end|>\n<|im_start|>assistant\nZorgville is the capital of Flurbia.<|im_end|>'},
] * 50  # Repeat to have enough data

train_dataset = Dataset.from_list(train_data)
print(f'\nTraining on {len(train_dataset)} samples (50 steps)...')

# Train with more steps to ensure learning
result = trainer.train(
    dataset=train_dataset,
    steps=50,
)

print(f'Training loss: {result.final_loss:.4f}')

# Generate AFTER training
print('\n--- AFTER TRAINING ---')
response_after = generate_response(trainer._model, trainer._tokenizer, test_prompt)
print(f'Q: {test_prompt}')
print(f'A: {response_after}')

# Check if learning occurred
print('\n--- VERIFICATION ---')
if 'zorgville' in response_after.lower():
    print('SUCCESS: Model learned the new fact! (mentions Zorgville)')
elif response_before != response_after:
    print('PARTIAL: Response changed after training, but may not have fully learned')
    print(f'  Before: {response_before[:100]}...')
    print(f'  After:  {response_after[:100]}...')
else:
    print('WARNING: Response unchanged - training may not have affected output')

# Also check loss dropped
print(f'\nLoss history (first 5): {result.loss_history[:5]}')
print(f'Loss history (last 5):  {result.loss_history[-5:]}')
