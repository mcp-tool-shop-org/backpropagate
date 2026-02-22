#!/usr/bin/env python
"""Test Phase 3 Data Quality improvements (config only, no heavy imports)."""
import sys

if __name__ != "__main__":
    sys.exit(0)

print('=== Testing Phase 3 Data Quality Improvements ===\n')

# Test 1: Format Detection
print('--- Test 1: Format Detection ---')
from backpropagate.datasets import DatasetFormat, detect_format

# ShareGPT format
sharegpt_sample = {"conversations": [{"from": "human", "value": "Hello"}, {"from": "gpt", "value": "Hi!"}]}
assert detect_format(sharegpt_sample) == DatasetFormat.SHAREGPT
print('[PASS] ShareGPT format detected')

# Alpaca format
alpaca_sample = {"instruction": "Write code", "input": "", "output": "def foo(): pass"}
assert detect_format(alpaca_sample) == DatasetFormat.ALPACA
print('[PASS] Alpaca format detected')

# OpenAI format
openai_sample = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
assert detect_format(openai_sample) == DatasetFormat.OPENAI
print('[PASS] OpenAI format detected')

# ChatML format
chatml_sample = {"text": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>"}
assert detect_format(chatml_sample) == DatasetFormat.CHATML
print('[PASS] ChatML format detected')

# Test 2: Validation
print('\n--- Test 2: Dataset Validation ---')
from backpropagate.datasets import validate_dataset

samples = [
    {"conversations": [{"from": "human", "value": "Hello"}, {"from": "gpt", "value": "Hi!"}]},
    {"conversations": [{"from": "human", "value": "How are you?"}, {"from": "gpt", "value": "Good!"}]},
    {"conversations": []},  # Invalid - empty
]

result = validate_dataset(samples)
assert result.total_rows == 3
assert result.valid_rows == 2
print(f'[PASS] Validation: {result.valid_rows}/{result.total_rows} valid')

# Test 3: Deduplication
print('\n--- Test 3: Deduplication ---')
from backpropagate.datasets import deduplicate_exact

samples = [
    {"text": "Hello world"},
    {"text": "Hello world"},  # Duplicate
    {"text": "Goodbye world"},
]

unique, removed = deduplicate_exact(samples)
assert len(unique) == 2
assert removed == 1
print(f'[PASS] Deduplication: {removed} duplicates removed')

# Test 4: Curriculum Learning
print('\n--- Test 4: Curriculum Learning ---')
from backpropagate.datasets import (
    analyze_curriculum,
    compute_difficulty_score,
    get_curriculum_chunks,
    order_by_difficulty,
)

samples = [
    {"text": "Hi"},  # Easy (short)
    {"text": "The quick brown fox jumps over the lazy dog repeatedly for many iterations."},  # Medium
    {"text": "In the multifaceted paradigm of computational linguistics, we observe that syntactic parsing algorithms demonstrate remarkable efficacy when processing natural language constructs with heterogeneous morphological characteristics."},  # Hard
]

# Test difficulty scoring
scores = [compute_difficulty_score(s) for s in samples]
assert scores[0] < scores[1] < scores[2], "Difficulty should increase with complexity"
print(f'[PASS] Difficulty scores: {[f"{s:.3f}" for s in scores]}')

# Test ordering
ordered = order_by_difficulty(samples)
ordered_scores = [compute_difficulty_score(s) for s in ordered]
assert ordered_scores == sorted(ordered_scores), "Should be sorted easy to hard"
print('[PASS] order_by_difficulty works correctly')

# Test chunking
samples_extended = samples * 10  # 30 samples
chunks = get_curriculum_chunks(samples_extended, num_chunks=3)
assert len(chunks) == 3
assert sum(len(c) for c in chunks) == 30
print(f'[PASS] get_curriculum_chunks: {[len(c) for c in chunks]} samples per chunk')

# Test analysis
stats = analyze_curriculum(samples_extended, num_chunks=3)
print('[PASS] analyze_curriculum:')
for i, (size, (d_min, d_max)) in enumerate(zip(stats.chunk_sizes, stats.difficulty_ranges)):
    print(f'       Chunk {i+1}: {size} samples, difficulty [{d_min:.3f} - {d_max:.3f}]')

# Test 5: Format Conversion
print('\n--- Test 5: Format Conversion ---')
from backpropagate.datasets import convert_to_chatml

# Convert ShareGPT to ChatML
sharegpt_samples = [
    {"conversations": [{"from": "human", "value": "Hello"}, {"from": "gpt", "value": "Hi!"}]},
]
chatml_samples = convert_to_chatml(sharegpt_samples, DatasetFormat.SHAREGPT)
assert "<|im_start|>user" in chatml_samples[0]["text"]
assert "<|im_start|>assistant" in chatml_samples[0]["text"]
print('[PASS] ShareGPT to ChatML conversion')

# Convert Alpaca to ChatML
alpaca_samples = [
    {"instruction": "Write hello", "input": "", "output": "Hello!"},
]
chatml_samples = convert_to_chatml(alpaca_samples, DatasetFormat.ALPACA)
assert "<|im_start|>user" in chatml_samples[0]["text"]
print('[PASS] Alpaca to ChatML conversion')

print('\n=== Phase 3 Tests Passed ===')
print('\nPhase 3 Features:')
print('  3.1 Dataset Validation - validate_dataset(), deduplicate_exact/minhash()')
print('  3.2 Smart Template Handling - detect_format(), convert_to_chatml()')
print('  3.3 Curriculum Learning - order_by_difficulty(), get_curriculum_chunks()')
