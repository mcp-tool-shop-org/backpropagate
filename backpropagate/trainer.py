"""
Backpropagate - Core Trainer Module
====================================

Production-ready LLM fine-tuning with smart defaults and Windows support.

Usage:
    from backpropagate import Trainer

    # Simple usage
    trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    trainer.train("my_data.jsonl")
    trainer.save("./my-model")

    # With options
    trainer = Trainer(
        model="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        lora_r=32,
        learning_rate=1e-4,
    )
    trainer.train(dataset="my_data.jsonl", steps=200)
    trainer.export("gguf")  # Export to GGUF for Ollama

Features:
- Auto VRAM detection for batch size
- Windows-safe multiprocessing
- QLoRA with Unsloth optimization
- Multiple export formats (LoRA, merged, GGUF)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field

from .config import settings, get_training_args
from .feature_flags import FEATURES, check_feature
from .gpu_safety import check_gpu_safe, get_gpu_status, format_gpu_status
from .exceptions import (
    TrainingError,
    ModelLoadError,
    TrainingAbortedError,
    DatasetError,
    DatasetNotFoundError,
    DatasetParseError,
    InvalidSettingError,
    GPUError,
    GPUNotAvailableError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Trainer",
    "TrainingRun",
    "TrainingCallback",
    "load_model",
    "load_dataset",
    "MultiRunTrainer",
    "SpeedrunTrainer",  # Backwards compatibility
]


@dataclass
class TrainingRun:
    """Container for training run results."""
    run_id: str
    steps: int
    final_loss: float
    loss_history: List[float] = field(default_factory=list)
    output_path: Optional[str] = None
    duration_seconds: float = 0.0
    samples_seen: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingCallback:
    """Callback hooks for training events."""
    on_step: Optional[Callable[[int, float], None]] = None
    on_epoch: Optional[Callable[[int], None]] = None
    on_save: Optional[Callable[[str], None]] = None
    on_complete: Optional[Callable[[TrainingRun], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None


class Trainer:
    """
    Headless LLM fine-tuning trainer with smart defaults.

    Args:
        model: Model name/path (HuggingFace or local)
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha (default: 32)
        learning_rate: Learning rate (default: 2e-4)
        batch_size: Batch size per device (default: "auto")
        output_dir: Output directory (default: "./output")

    Example:
        >>> trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
        >>> trainer.train("data.jsonl", steps=100)
        >>> trainer.save("./my-model")
    """

    def __init__(
        self,
        model: str = None,
        lora_r: int = None,
        lora_alpha: int = None,
        lora_dropout: float = None,
        learning_rate: float = None,
        batch_size: Union[int, str] = "auto",
        gradient_accumulation: int = None,
        max_seq_length: int = None,
        output_dir: str = None,
        use_unsloth: bool = True,
        train_on_responses: bool = True,  # Phase 1.1: Only compute loss on assistant responses
    ):
        # Use settings as defaults, override with provided values
        self.model_name = model or settings.model.name
        self.lora_r = lora_r or settings.lora.r
        self.lora_alpha = lora_alpha or settings.lora.lora_alpha
        self.lora_dropout = lora_dropout or settings.lora.lora_dropout
        self.learning_rate = learning_rate or settings.training.learning_rate
        self.gradient_accumulation = gradient_accumulation or settings.training.gradient_accumulation_steps
        self.max_seq_length = max_seq_length or settings.model.max_seq_length
        self.output_dir = Path(output_dir or settings.training.output_dir)
        self.use_unsloth = use_unsloth and check_feature("unsloth")

        # Auto batch size
        if batch_size == "auto":
            self.batch_size = self._detect_batch_size()
        else:
            self.batch_size = batch_size

        # Phase 1.1: Train on responses only
        self._train_on_responses = train_on_responses

        # Internal state
        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._is_loaded = False
        self._training_runs: List[TrainingRun] = []

        # Apply Windows fixes
        self._apply_windows_fixes()

        logger.info(f"Trainer initialized: {self.model_name}")
        logger.info(f"  LoRA: r={self.lora_r}, alpha={self.lora_alpha}")
        logger.info(f"  Batch: {self.batch_size}, LR: {self.learning_rate}")

    def _apply_windows_fixes(self) -> None:
        """Apply Windows-specific environment variables."""
        if os.name == "nt":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            if settings.windows.xformers_disabled:
                os.environ["XFORMERS_DISABLED"] = "1"
            if settings.windows.cuda_launch_blocking:
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            logger.debug("Applied Windows-specific fixes")

    def _detect_batch_size(self) -> int:
        """Auto-detect optimal batch size based on available VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 24:
                    return 4
                elif vram_gb >= 16:
                    return 2
                elif vram_gb >= 12:
                    return 1
                else:
                    return 1
        except ImportError:
            logger.debug("PyTorch not available for batch size detection, using default")
        except RuntimeError as e:
            logger.debug(f"CUDA query failed for batch size detection: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error detecting batch size: {type(e).__name__}: {e}")
        return 2  # Safe default

    def load_model(self) -> None:
        """
        Load the model and tokenizer.

        Raises:
            ModelLoadError: If the model or tokenizer cannot be loaded
            GPUNotAvailableError: If CUDA is required but not available
        """
        if self._is_loaded:
            return

        logger.info(f"Loading model: {self.model_name}")

        try:
            if self.use_unsloth:
                self._load_with_unsloth()
            else:
                self._load_with_transformers()
        except ImportError as e:
            raise ModelLoadError(
                self.model_name,
                f"Missing required package: {e.name if hasattr(e, 'name') else str(e)}",
                suggestion="Install required packages: pip install unsloth transformers peft"
            ) from e
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg or "gpu" in error_msg:
                raise GPUNotAvailableError(
                    suggestion="Ensure CUDA is installed and GPU is available"
                ) from e
            raise ModelLoadError(
                self.model_name,
                str(e),
            ) from e
        except Exception as e:
            raise ModelLoadError(
                self.model_name,
                str(e),
            ) from e

        self._is_loaded = True
        logger.info("Model loaded successfully")

    def _load_with_unsloth(self) -> None:
        """Load model using Unsloth for 2x faster training."""
        from unsloth import FastLanguageModel

        try:
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=True,
                trust_remote_code=settings.model.trust_remote_code,
            )
        except Exception as e:
            raise ModelLoadError(
                self.model_name,
                f"Unsloth model loading failed: {e}",
                suggestion="Check model name and network connection"
            ) from e

        # Apply LoRA
        try:
            self._model = FastLanguageModel.get_peft_model(
                self._model,
                r=self.lora_r,
                target_modules=settings.lora.target_modules,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                use_gradient_checkpointing=settings.lora.use_gradient_checkpointing,
                random_state=settings.lora.random_state,
            )
        except Exception as e:
            raise ModelLoadError(
                self.model_name,
                f"Failed to apply LoRA: {e}",
            ) from e

    def _load_with_transformers(self) -> None:
        """Load model using standard transformers + PEFT."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=settings.model.trust_remote_code,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=settings.model.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Prepare for training
        self._model = prepare_model_for_kbit_training(self._model)

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=settings.lora.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, lora_config)

    def train(
        self,
        dataset: Union[str, Any] = None,
        steps: int = None,
        samples: int = None,
        callback: TrainingCallback = None,
    ) -> TrainingRun:
        """
        Train the model on a dataset.

        Args:
            dataset: Dataset path (JSONL, CSV) or HuggingFace dataset name
            steps: Number of training steps (overrides config)
            samples: Number of samples to use (overrides config)
            callback: Optional callback for training events

        Returns:
            TrainingRun with results

        Raises:
            InvalidSettingError: If steps or samples are invalid
            DatasetError: If dataset cannot be loaded
            TrainingError: If training fails
        """
        import time
        from trl import SFTTrainer, SFTConfig

        # Validate inputs
        if steps is not None:
            if not isinstance(steps, int) or steps <= 0:
                raise InvalidSettingError(
                    "steps", steps, "positive integer",
                    suggestion="Use steps=100 or higher"
                )
        if samples is not None:
            if not isinstance(samples, int) or samples <= 0:
                raise InvalidSettingError(
                    "samples", samples, "positive integer",
                    suggestion="Use samples=1000 or higher"
                )

        # Load model if not loaded
        if not self._is_loaded:
            self.load_model()

        # Load dataset
        train_dataset = self._load_dataset(dataset, samples)

        # Pre-tokenize for Windows safety
        if os.name == "nt" and settings.windows.pre_tokenize:
            train_dataset = self._pre_tokenize(train_dataset)

        # Training arguments (TRL 0.27+ uses SFTConfig)
        training_args = SFTConfig(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation,
            max_steps=steps or settings.training.max_steps,
            learning_rate=self.learning_rate,
            weight_decay=settings.training.weight_decay,
            warmup_steps=settings.training.warmup_steps,
            optim=settings.training.optim,
            lr_scheduler_type=settings.training.lr_scheduler_type,
            logging_steps=settings.training.logging_steps,
            save_steps=settings.training.save_steps,
            bf16=settings.training.bf16,
            fp16=settings.training.fp16,
            seed=settings.training.seed,
            overwrite_output_dir=True,
            dataloader_num_workers=0 if os.name == "nt" else 4,
            report_to="none",  # Disable default reporting
            # SFT-specific args (moved from SFTTrainer in TRL 0.27+)
            max_length=self.max_seq_length,
            packing=settings.data.packing,
        )

        # Create trainer (TRL 0.27+ uses processing_class instead of tokenizer)
        self._trainer = SFTTrainer(
            model=self._model,
            processing_class=self._tokenizer,
            train_dataset=train_dataset,
            args=training_args,
        )

        # Apply train_on_responses_only if using Unsloth (Phase 1.1 optimization)
        # This focuses loss only on assistant responses, not user prompts
        # NOTE: Disabled on Windows due to multiprocessing issues that can crash the system
        if self.use_unsloth and self._train_on_responses and os.name != "nt":
            try:
                from unsloth.chat_templates import train_on_responses_only
                # Unsloth 2026+ API: train_on_responses_only(trainer, instruction_part, response_part, ...)
                # For ChatML format (Qwen): <|im_start|>user and <|im_start|>assistant
                self._trainer = train_on_responses_only(
                    self._trainer,
                    instruction_part="<|im_start|>user",
                    response_part="<|im_start|>assistant",
                    num_proc=1,  # Single process to avoid Windows issues
                )
                logger.info("Applied train_on_responses_only optimization")
            except ImportError:
                logger.warning("train_on_responses_only not available in this Unsloth version")
            except Exception as e:
                logger.warning(f"Failed to apply train_on_responses_only: {e}")
        elif self._train_on_responses and os.name == "nt":
            logger.info("train_on_responses_only disabled on Windows (multiprocessing issues)")

        # Train
        run_id = f"run_{len(self._training_runs) + 1}"
        start_time = time.time()
        loss_history = []

        logger.info(f"Starting training: {run_id}")
        logger.info(f"  Steps: {steps or settings.training.max_steps}")
        logger.info(f"  Samples: {len(train_dataset)}")

        try:
            result = self._trainer.train()
            duration = time.time() - start_time

            # Validate training result
            if not hasattr(result, 'training_loss'):
                logger.warning("Training result missing 'training_loss' attribute - using 0.0")
            final_loss = getattr(result, 'training_loss', 0.0)

            # Extract loss history from logs
            loss_history = []
            if hasattr(self._trainer, 'state') and self._trainer.state.log_history:
                loss_history = [
                    log.get('loss', 0) for log in self._trainer.state.log_history
                    if 'loss' in log
                ]

            run = TrainingRun(
                run_id=run_id,
                steps=steps or settings.training.max_steps,
                final_loss=final_loss,
                loss_history=loss_history,
                duration_seconds=duration,
                samples_seen=len(train_dataset),
                output_path=str(self.output_dir),
            )

            self._training_runs.append(run)

            if callback and callback.on_complete:
                try:
                    callback.on_complete(run)
                except Exception as cb_error:
                    logger.warning(f"on_complete callback raised error: {cb_error}")

            logger.info(f"Training complete: loss={final_loss:.4f}, time={duration:.1f}s")
            return run

        except KeyboardInterrupt:
            duration = time.time() - start_time
            raise TrainingAbortedError(
                reason="User interrupted training",
                steps_completed=getattr(self._trainer.state, 'global_step', 0) if self._trainer else 0,
            ) from None
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                raise TrainingError(
                    f"GPU error during training: {e}",
                    suggestion="Try reducing batch_size or using gradient_checkpointing"
                ) from e
            if callback and callback.on_error:
                callback.on_error(e)
            raise TrainingError(f"Training failed: {e}") from e
        except Exception as e:
            if callback and callback.on_error:
                callback.on_error(e)
            raise TrainingError(f"Training failed: {e}") from e

    def _load_dataset(
        self,
        dataset: Union[str, Any],
        samples: int = None,
    ) -> Any:
        """
        Load dataset from various sources.

        Raises:
            DatasetNotFoundError: If dataset file doesn't exist
            DatasetParseError: If dataset cannot be parsed
            DatasetError: For other dataset-related errors
        """
        from datasets import load_dataset, Dataset

        max_samples = samples or settings.data.max_samples

        try:
            if dataset is None:
                # Use default dataset from config
                ds = load_dataset(
                    settings.data.dataset_name,
                    split=settings.data.dataset_split,
                )
            elif isinstance(dataset, str):
                if dataset.endswith('.jsonl') or dataset.endswith('.json'):
                    # Check file exists before trying to load
                    dataset_path = Path(dataset)
                    if not dataset_path.exists():
                        raise DatasetNotFoundError(
                            dataset,
                            suggestion=f"Create the file or use a HuggingFace dataset name"
                        )
                    try:
                        ds = load_dataset('json', data_files=dataset, split='train')
                    except Exception as e:
                        raise DatasetParseError(
                            f"Failed to parse JSON dataset: {e}",
                            path=dataset,
                            suggestion="Check that the file contains valid JSONL format"
                        ) from e
                elif dataset.endswith('.csv'):
                    dataset_path = Path(dataset)
                    if not dataset_path.exists():
                        raise DatasetNotFoundError(dataset)
                    try:
                        ds = load_dataset('csv', data_files=dataset, split='train')
                    except Exception as e:
                        raise DatasetParseError(
                            f"Failed to parse CSV dataset: {e}",
                            path=dataset,
                        ) from e
                else:
                    # Assume HuggingFace dataset name
                    try:
                        ds = load_dataset(dataset, split=settings.data.dataset_split)
                    except Exception as e:
                        raise DatasetError(
                            f"Failed to load HuggingFace dataset '{dataset}': {e}",
                            suggestion="Check dataset name and network connection"
                        ) from e
            elif isinstance(dataset, Dataset):
                ds = dataset
            else:
                raise DatasetError(
                    f"Unsupported dataset type: {type(dataset).__name__}",
                    suggestion="Use a file path (JSONL, CSV), HuggingFace dataset name, or Dataset object"
                )
        except (DatasetNotFoundError, DatasetParseError, DatasetError):
            raise
        except FileNotFoundError as e:
            raise DatasetNotFoundError(str(e)) from e
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {e}") from e

        # Limit samples
        if max_samples > 0 and len(ds) > max_samples:
            if settings.data.shuffle:
                ds = ds.shuffle(seed=settings.training.seed)
            ds = ds.select(range(max_samples))

        logger.info(f"Loaded {len(ds)} samples")
        return ds

    def _pre_tokenize(self, dataset: Any) -> Any:
        """Pre-tokenize dataset for Windows safety."""
        logger.info("Pre-tokenizing dataset (Windows-safe mode)")

        def tokenize_fn(examples):
            try:
                return self._tokenizer(
                    examples[settings.data.text_column],
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=False,
                )
            except KeyError:
                raise DatasetError(
                    f"Dataset missing required column '{settings.data.text_column}'",
                    suggestion=f"Available columns: {list(examples.keys()) if hasattr(examples, 'keys') else 'unknown'}"
                )

        try:
            tokenized = dataset.map(
                tokenize_fn,
                batched=True,
                num_proc=None,  # None = run in main process (avoids Windows multiprocessing issues)
                remove_columns=dataset.column_names,
                desc="Tokenizing",
            )
        except DatasetError:
            raise
        except Exception as e:
            raise DatasetError(f"Tokenization failed: {e}") from e

        return tokenized

    def save(self, path: str = None, save_merged: bool = False) -> str:
        """
        Save the trained model.

        Args:
            path: Output path (default: output_dir/lora)
            save_merged: Whether to save merged weights (larger but standalone)

        Returns:
            Path to saved model

        Raises:
            TrainingError: If no model loaded or save fails
        """
        from .exceptions import CheckpointError

        if not self._is_loaded:
            raise TrainingError("No model loaded. Call load_model() or train() first.")

        output_path = Path(path or self.output_dir / "lora")

        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise CheckpointError(
                "save", str(output_path),
                f"Permission denied creating directory: {e}"
            ) from e
        except OSError as e:
            raise CheckpointError(
                "save", str(output_path),
                f"Failed to create directory: {e}"
            ) from e

        try:
            if save_merged and self.use_unsloth:
                from unsloth import FastLanguageModel
                self._model.save_pretrained_merged(
                    str(output_path),
                    self._tokenizer,
                    save_method="merged_16bit",
                )
            else:
                self._model.save_pretrained(str(output_path))
                self._tokenizer.save_pretrained(str(output_path))
        except Exception as e:
            raise CheckpointError(
                "save", str(output_path),
                str(e)
            ) from e

        logger.info(f"Model saved to: {output_path}")
        return str(output_path)

    def export(
        self,
        format: str = "lora",
        output_dir: Optional[str] = None,
        quantization: str = "q4_k_m",
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        **kwargs,
    ) -> "ExportResult":
        """
        Export the trained model.

        Args:
            format: Export format - "lora", "merged", or "gguf"
            output_dir: Output directory (default: self.output_dir/format)
            quantization: GGUF quantization type (q4_k_m, q5_k_m, q8_0, f16)
            push_to_hub: Whether to push to HuggingFace Hub (merged only)
            repo_id: Repository ID for Hub (required if push_to_hub=True)
            **kwargs: Additional arguments passed to export functions

        Returns:
            ExportResult with path, size, and timing info

        Example:
            >>> trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
            >>> trainer.train("data.jsonl", steps=100)
            >>> result = trainer.export("gguf", quantization="q4_k_m")
            >>> print(result.summary())
        """
        from .export import (
            ExportFormat,
            ExportResult,
            export_lora,
            export_merged,
            export_gguf,
        )

        if not self._is_loaded:
            raise RuntimeError("No model loaded. Call load_model() or train() first.")

        output_path = Path(output_dir or self.output_dir / format)

        format_lower = format.lower()

        if format_lower == "lora":
            result = export_lora(
                model=self._model,
                output_dir=output_path,
                **kwargs,
            )
            logger.info(f"Exported LoRA adapter: {result.path}")

        elif format_lower == "merged":
            result = export_merged(
                model=self._model,
                tokenizer=self._tokenizer,
                output_dir=output_path,
                push_to_hub=push_to_hub,
                repo_id=repo_id,
                **kwargs,
            )
            logger.info(f"Exported merged model: {result.path}")

        elif format_lower == "gguf":
            result = export_gguf(
                model=self._model,
                tokenizer=self._tokenizer,
                output_dir=output_path,
                quantization=quantization,
                **kwargs,
            )
            logger.info(f"Exported GGUF ({quantization}): {result.path}")

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'lora', 'merged', or 'gguf'")

        return result

    def push_to_hub(self, repo_id: str, private: bool = True) -> None:
        """Push model to HuggingFace Hub."""
        if not self._is_loaded:
            raise RuntimeError("No model loaded.")

        self._model.push_to_hub(repo_id, private=private)
        self._tokenizer.push_to_hub(repo_id, private=private)
        logger.info(f"Pushed to HuggingFace Hub: {repo_id}")

    @property
    def model(self):
        """Access the underlying model."""
        return self._model

    @property
    def tokenizer(self):
        """Access the tokenizer."""
        return self._tokenizer

    @property
    def runs(self) -> List[TrainingRun]:
        """Get all training runs."""
        return self._training_runs

    def multi_run(
        self,
        dataset: Union[str, Any] = None,
        num_runs: int = 5,
        steps_per_run: int = 100,
        samples_per_run: int = 1000,
        merge_mode: str = "slao",
        checkpoint_dir: str = None,
        on_run_complete: Callable = None,
    ) -> "MultiRunResult":
        """
        Execute SLAO Multi-Run training (multiple short runs with LoRA merging).

        This is the recommended approach for fine-tuning as it:
        - Prevents catastrophic forgetting via SLAO merging
        - Exposes the model to diverse data across runs
        - Saves checkpoints after each run for rollback
        - Monitors GPU safety throughout

        Args:
            dataset: Dataset name/path or HuggingFace dataset
            num_runs: Number of training runs (default: 5)
            steps_per_run: Steps per run (default: 100)
            samples_per_run: Fresh samples per run (default: 1000)
            merge_mode: "slao" (recommended) or "simple"
            checkpoint_dir: Where to save checkpoints
            on_run_complete: Callback after each run

        Returns:
            MultiRunResult with aggregate metrics and run history

        Example:
            >>> trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
            >>> result = trainer.multi_run(
            ...     dataset="HuggingFaceH4/ultrachat_200k",
            ...     num_runs=5,
            ...     steps_per_run=100,
            ... )
            >>> print(f"Final loss: {result.final_loss}")
        """
        from .multi_run import MultiRunTrainer, MultiRunConfig, MergeMode

        # Pre-flight GPU check
        if not check_gpu_safe():
            raise RuntimeError("GPU safety check failed. Check temperature and VRAM.")

        config = MultiRunConfig(
            num_runs=num_runs,
            steps_per_run=steps_per_run,
            samples_per_run=samples_per_run,
            merge_mode=MergeMode(merge_mode.lower()),
            checkpoint_dir=checkpoint_dir or str(self.output_dir / "multi_run"),
            initial_lr=self.learning_rate,
        )

        multi_run_trainer = MultiRunTrainer(
            model=self.model_name,
            config=config,
            on_run_complete=on_run_complete,
        )

        return multi_run_trainer.run(dataset)

    # Backwards compatibility alias
    speedrun = multi_run


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_model(
    model_name: str = None,
    load_in_4bit: bool = True,
    max_seq_length: int = 2048,
) -> tuple:
    """
    Load a model and tokenizer.

    Args:
        model_name: Model name/path
        load_in_4bit: Use 4-bit quantization
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (model, tokenizer)
    """
    trainer = Trainer(model=model_name, max_seq_length=max_seq_length)
    trainer.load_model()
    return trainer.model, trainer.tokenizer


def load_dataset(
    dataset: Union[str, Any],
    max_samples: int = None,
    split: str = None,
) -> Any:
    """
    Load a dataset.

    Args:
        dataset: Dataset path or name
        max_samples: Maximum samples to load
        split: Dataset split

    Returns:
        HuggingFace Dataset
    """
    from datasets import load_dataset as hf_load_dataset

    if isinstance(dataset, str):
        if dataset.endswith('.jsonl') or dataset.endswith('.json'):
            ds = hf_load_dataset('json', data_files=dataset, split='train')
        elif dataset.endswith('.csv'):
            ds = hf_load_dataset('csv', data_files=dataset, split='train')
        else:
            ds = hf_load_dataset(dataset, split=split or "train")
    else:
        ds = dataset

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    return ds


# Import MultiRunTrainer for re-export (lazy to avoid circular imports)
def __getattr__(name):
    if name == "MultiRunTrainer":
        from .multi_run import MultiRunTrainer
        return MultiRunTrainer
    if name == "SpeedrunTrainer":  # Backwards compatibility
        from .multi_run import SpeedrunTrainer
        return SpeedrunTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
