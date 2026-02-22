"""Train on Perfect Pairs dataset using Backpropagate MultiRun.

Perfect Pairs = Python functions with validated doctest examples.
809 high-quality code examples with self-documenting tests.

Uses SLAO (Smooth LoRA Aggregation with Orthogonal initialization)
for stable multi-run training.
"""

import os
import sys
from pathlib import Path

# Windows-specific env vars (must be set before imports)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["XFORMERS_DISABLED"] = "1"  # RTX 5080 SM 12.0

# Add backpropagate to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from multiprocessing import freeze_support


def main():
    """Run multi-run training on perfect pairs."""
    from backpropagate.gpu_safety import GPUCondition, get_gpu_status, wait_for_safe_gpu
    from backpropagate.multi_run import MergeMode, MultiRunConfig, MultiRunTrainer

    # Check GPU status first
    status = get_gpu_status()
    if status and status.available:
        print(f"GPU: {status.device_name}")
        print(f"VRAM: {status.vram_used_gb:.1f}/{status.vram_total_gb:.1f} GB ({status.vram_percent:.0f}%)")
        print(f"Temperature: {status.temperature_c}C")
        print(f"Condition: {status.condition}")
        print()

        if status.condition != GPUCondition.SAFE:
            print("Waiting for GPU to cool down...")
            wait_for_safe_gpu(max_wait=120)

    # Dataset path
    dataset_path = Path("F:/AI/checkpoints/validated/perfect_pairs_chat.jsonl")

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Run convert_validated_pairs.py first")
        return

    # Count samples
    with open(dataset_path, encoding="utf-8") as f:
        num_samples = sum(1 for _ in f)
    print(f"Dataset: {dataset_path}")
    print(f"Total samples: {num_samples}")

    # Configure multi-run training
    # With 809 samples, we'll do 5 runs of ~160 samples each
    # Conservative settings for RTX 5080 Laptop
    config = MultiRunConfig(
        num_runs=5,
        steps_per_run=80,  # ~160 samples per run with batch 2
        samples_per_run=160,
        merge_mode=MergeMode.SLAO,
        lr_decay="cosine",
        initial_lr=2e-4,
        final_lr=5e-5,
        checkpoint_dir="F:/AI/models/perfect_pairs",
        enable_gpu_monitoring=True,
        pause_on_overheat=True,
        max_temp_c=80.0,  # Conservative for laptop
        validation_samples=50,
        validate_every_run=True,
    )

    trainer = MultiRunTrainer(
        model="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        config=config,
    )

    print("\n" + "="*50)
    print("Training Configuration")
    print("="*50)
    print("Model: unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    print(f"Runs: {config.num_runs}")
    print(f"Steps per run: {config.steps_per_run}")
    print(f"Samples per run: {config.samples_per_run}")
    print(f"Total samples: ~{config.num_runs * config.samples_per_run}")
    print(f"Merge mode: {config.merge_mode.value}")
    print(f"LR: {config.initial_lr} -> {config.final_lr} ({config.lr_decay})")
    print(f"GPU monitoring: {config.enable_gpu_monitoring}")
    print(f"Max temp: {config.max_temp_c}C")
    print(f"Output: {config.checkpoint_dir}")
    print("="*50 + "\n")

    # Confirm before starting
    response = input("Start training? [y/N]: ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return

    # Run training
    print("\nStarting multi-run training...")
    result = trainer.run(dataset=str(dataset_path))

    # Summary
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Total runs: {result.total_runs}")
    print(f"Total steps: {result.total_steps}")
    print(f"Final loss: {result.final_loss:.4f}")
    print(f"Total time: {result.total_duration_seconds/60:.1f} minutes")
    print(f"Aborted: {result.aborted}")

    if result.final_checkpoint_path:
        print(f"Final checkpoint: {result.final_checkpoint_path}")

    # Per-run summary
    print("\nPer-run results:")
    for run in result.runs:
        val_str = f", val_loss={run.validation_loss:.4f}" if run.validation_loss else ""
        print(f"  Run {run.run_index}: loss={run.final_loss:.4f}{val_str}, lr={run.learning_rate:.2e}")


if __name__ == "__main__":
    freeze_support()
    main()
