# Backpropagate Roadmap

## Goal: Maximize Training Effectiveness with SLAO Multi-Run

Based on research from [SLAO paper](https://arxiv.org/abs/2512.23017), [LoRA Without Regret](https://huggingface.co/docs/trl/en/lora_without_regret), [Unsloth docs](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), and [Databricks guide](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms).

---

## Phase 1: Core Training Improvements (High Impact)

### 1.1 Train on Responses Only
**Impact: High** | **Effort: Low**

Currently training on full conversations. Should only compute loss on assistant responses.

```python
# Add to trainer.py after SFTTrainer creation
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(trainer, tokenizer, instruction_part="<|im_start|>user", response_part="<|im_start|>assistant")
```

**Why:** Prevents model from "learning" to generate user prompts, focuses loss entirely on what matters.

### 1.2 Optimized Batch Size Configuration
**Impact: High** | **Effort: Low**

Research shows LoRA works best with effective batch size 8-32.

| Preset | batch_size | grad_accum | effective | Use Case |
|--------|------------|------------|-----------|----------|
| fast | 2 | 4 | 8 | Quick iterations |
| balanced | 2 | 8 | 16 | Default recommended |
| quality | 4 | 8 | 32 | Final training |

### 1.3 Learning Rate Scaling
**Impact: Medium** | **Effort: Low**

LoRA benefits from ~10× higher LR than full fine-tuning.

- Small datasets (<1K): `lr=5e-4` with more warmup
- Medium datasets (1K-10K): `lr=2e-4` (current default)
- Large datasets (>10K): `lr=1e-4`

---

## Phase 2: SLAO Multi-Run Enhancements (Core Focus)

### 2.1 Proper Orthogonal Initialization Integration
**Impact: High** | **Effort: Medium**

Current `multi_run.py` doesn't fully integrate SLAO's orthogonal init. Need to:

1. After run N completes, extract LoRA state
2. Apply `orthogonal_init_A()` to all A matrices
3. Initialize run N+1 with these weights

```python
# In _prepare_for_next_run()
if self.config.merge_mode == MergeMode.SLAO:
    init_weights = self._slao_merger.get_init_weights()  # Already orthogonalized
    self._load_lora_weights_to_model(init_weights)
```

### 2.2 Time-Aware Scaling Tuning
**Impact: Medium** | **Effort: Low**

Current: `λ(i) = 1/√i`

Add configurable decay curves:
- `sqrt`: 1/√i (paper default, good balance)
- `linear`: 1/i (more aggressive, preserves early learning)
- `log`: 1/log(i+1) (slower decay, more plasticity)
- `constant`: 1.0 (simple EMA)

### 2.3 Run-Specific Learning Rate Decay
**Impact: Medium** | **Effort: Low**

Decay LR across runs to stabilize:

```python
# Run 1: lr = 2e-4
# Run 2: lr = 1.5e-4
# Run 3: lr = 1e-4
# ...
lr_schedule = [initial_lr * (decay ** i) for i in range(num_runs)]
```

### 2.4 Data Replay (Experience Replay)
**Impact: Medium** | **Effort: Medium**

Mix small fraction of previous data to prevent forgetting:

```python
class MultiRunConfig:
    replay_fraction: float = 0.1  # 10% from previous runs
    replay_strategy: str = "recent"  # "recent", "random", "hard_examples"
```

---

## Phase 3: Data Quality & Preprocessing

### 3.1 Dataset Validation
**Impact: High** | **Effort: Medium**

Add pre-training checks:
- Duplicate detection
- Format validation
- Token length distribution analysis
- Quality scoring (perplexity-based filtering)

### 3.2 Smart Chat Template Handling
**Impact: Medium** | **Effort: Medium**

Auto-detect and convert formats:
- ShareGPT → ChatML
- Alpaca → ChatML
- Custom templates

### 3.3 Curriculum Learning
**Impact: Medium** | **Effort: High**

Order training data by difficulty:
1. Short, simple examples first
2. Gradually increase complexity
3. Hard examples in later runs

---

## Phase 4: Advanced SLAO Features

### 4.1 Adaptive Scaling Based on Task Similarity
**Impact: High** | **Effort: High**

Instead of fixed `1/√i`, compute similarity between runs:

```python
# If new task similar to previous: smaller scale (more merge)
# If new task different: larger scale (preserve new learning)
similarity = cosine_similarity(task_embedding_new, task_embedding_prev)
scale = base_scale * (1 - similarity * 0.5)
```

### 4.2 Selective Layer Merging
**Impact: Medium** | **Effort: Medium**

Research suggests different layers have different plasticity:
- Early layers: More generic, merge more aggressively
- Late layers: More task-specific, preserve more

```python
layer_scales = {
    "layers.0-8": 0.3,   # Early layers: high merge
    "layers.9-16": 0.5,  # Middle layers: balanced
    "layers.17-24": 0.7, # Late layers: preserve new
}
```

### 4.3 Validation-Based Early Stopping Per Run
**Impact: Medium** | **Effort: Medium**

Stop individual runs early if validation loss increases:

```python
class MultiRunConfig:
    validate_every_n_steps: int = 20
    early_stop_patience: int = 3
    early_stop_threshold: float = 0.01
```

---

## Phase 5: Monitoring & Observability

### 5.1 Rich Training Dashboard
**Impact: Medium** | **Effort: Medium**

Real-time metrics via Gradio or terminal:
- Loss per run (with run boundaries marked)
- SLAO merge statistics
- GPU utilization
- Estimated time remaining

### 5.2 Merge Quality Metrics
**Impact: Low** | **Effort: Medium**

Track SLAO-specific metrics:
- A matrix orthogonality score
- B matrix drift from baseline
- Per-layer merge ratios

### 5.3 Checkpoint Management
**Impact: Low** | **Effort: Low**

Smart checkpoint pruning:
- Keep best N checkpoints by validation loss
- Always keep final checkpoint
- Optional: keep all run boundaries

---

## Implementation Priority

### Week 1: Quick Wins
- [x] 1.1 Train on responses only (disabled on Windows due to multiprocessing)
- [x] 1.2 Batch size presets
- [x] 2.2 Configurable time-aware scaling

### Week 2: SLAO Core
- [x] 2.1 Proper orthogonal init integration
- [x] 2.3 Run-specific LR decay
- [x] 2.4 Basic data replay

### Week 3: Quality
- [x] 3.1 Dataset validation
- [x] 3.2 Chat template auto-detection
- [x] 4.3 Validation-based early stopping

### Week 4: Advanced
- [x] 4.1 Adaptive scaling
- [x] 4.2 Selective layer merging
- [x] 5.1 Training dashboard (right sidebar with accordion)
- [x] 5.2 Merge quality metrics (in dashboard)
- [x] 5.3 Checkpoint management (smart pruning, configurable)

---

## Configuration Presets

### `fast` - Quick iterations
```python
TrainingPreset(
    lora_r=8,
    batch_size=2,
    grad_accum=4,  # effective=8
    lr=5e-4,
    steps_per_run=50,
    num_runs=3,
)
```

### `balanced` - Default recommended
```python
TrainingPreset(
    lora_r=16,
    batch_size=2,
    grad_accum=8,  # effective=16
    lr=2e-4,
    steps_per_run=100,
    num_runs=5,
)
```

### `quality` - Maximum effectiveness
```python
TrainingPreset(
    lora_r=32,
    batch_size=4,
    grad_accum=8,  # effective=32
    lr=1e-4,
    steps_per_run=200,
    num_runs=10,
    replay_fraction=0.1,
    validate_every_run=True,
)
```

---

## References

- [SLAO Paper: Merge before Forget](https://arxiv.org/abs/2512.23017)
- [LoRA Without Regret](https://huggingface.co/docs/trl/en/lora_without_regret)
- [Unsloth Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Databricks LoRA Guide](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
- [Continual Learning Survey](https://github.com/Wang-ML-Lab/llm-continual-learning-survey)
