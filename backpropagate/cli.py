"""
Backpropagate CLI
=================

Command-line interface for LLM fine-tuning.

Usage:
    # Train a model
    backprop train --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --data my_data.jsonl --steps 100

    # Export to GGUF
    backprop export ./output/lora --format gguf --quantization q4_k_m

    # Multi-run training
    backprop multi-run --model unsloth/Qwen2.5-7B --data ultrachat --runs 5

    # Launch web UI
    backprop ui --port 7862

    # Show system info
    backprop info
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .exceptions import (
    BackpropagateError,
    TrainingError,
    DatasetError,
    ExportError,
    ConfigurationError,
)
from .security import safe_path, PathTraversalError

logger = logging.getLogger(__name__)

__all__ = ["main", "create_parser"]


# =============================================================================
# TERMINAL COLORS (ANSI)
# =============================================================================

def _supports_color() -> bool:
    """Check if terminal supports ANSI colors."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        # Windows 10+ supports ANSI
        return bool(os.environ.get("TERM") or os.environ.get("WT_SESSION"))
    return True


class Colors:
    """ANSI color codes."""

    ENABLED = _supports_color()

    RESET = "\033[0m" if ENABLED else ""
    BOLD = "\033[1m" if ENABLED else ""
    DIM = "\033[2m" if ENABLED else ""

    RED = "\033[31m" if ENABLED else ""
    GREEN = "\033[32m" if ENABLED else ""
    YELLOW = "\033[33m" if ENABLED else ""
    BLUE = "\033[34m" if ENABLED else ""
    MAGENTA = "\033[35m" if ENABLED else ""
    CYAN = "\033[36m" if ENABLED else ""
    WHITE = "\033[37m" if ENABLED else ""


def _print_header(text: str) -> None:
    """Print a styled header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * len(text)}{Colors.RESET}")


def _print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}[OK]{Colors.RESET} {text}")


def _print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {text}", file=sys.stderr)


def _print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {text}")


def _print_info(text: str) -> None:
    """Print info message."""
    print(f"{Colors.BLUE}i{Colors.RESET} {text}")


def _print_kv(key: str, value: str, indent: int = 2) -> None:
    """Print key-value pair."""
    spaces = " " * indent
    print(f"{spaces}{Colors.DIM}{key}:{Colors.RESET} {value}")


# =============================================================================
# PROGRESS DISPLAY
# =============================================================================

class ProgressBar:
    """Simple ASCII progress bar."""

    def __init__(self, total: int, width: int = 40, prefix: str = ""):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0

    def update(self, current: int, suffix: str = "") -> None:
        """Update progress bar."""
        self.current = current
        percent = current / self.total if self.total > 0 else 0
        filled = int(self.width * percent)
        bar = "#" * filled + "-" * (self.width - filled)

        line = f"\r{self.prefix}[{bar}] {percent:>6.1%}"
        if suffix:
            line += f" {suffix}"

        print(line, end="", flush=True)

        if current >= self.total:
            print()  # Newline when complete

    def finish(self) -> None:
        """Complete the progress bar."""
        self.update(self.total)


# =============================================================================
# COMMAND: train
# =============================================================================

def cmd_train(args: argparse.Namespace) -> int:
    """Execute the train command."""
    from .trainer import Trainer, TrainingCallback

    _print_header("Backpropagate Training")

    # Validate data argument
    if not args.data:
        _print_error("--data is required")
        return 1

    _print_info(f"Model: {args.model}")
    _print_info(f"Dataset: {args.data}")
    _print_info(f"Steps: {args.steps}")
    if args.samples:
        _print_info(f"Samples: {args.samples}")

    try:
        # Create trainer
        trainer = Trainer(
            model=args.model,
            lora_r=args.lora_r,
            learning_rate=args.lr,
            batch_size=args.batch_size if args.batch_size != "auto" else "auto",
            output_dir=args.output,
            use_unsloth=not args.no_unsloth,
        )

        # Progress callback
        progress = ProgressBar(args.steps, prefix="Training: ")

        def on_step(step: int, loss: float) -> None:
            progress.update(step, f"loss={loss:.4f}")

        callback = TrainingCallback(on_step=on_step)

        print()  # Blank line before progress

        # Train
        result = trainer.train(
            dataset=args.data,
            steps=args.steps,
            samples=args.samples,
            callback=callback,
        )

        progress.finish()

        # Save
        save_path = trainer.save(args.output)

        print()
        _print_success(f"Training complete!")
        _print_kv("Final loss", f"{result.final_loss:.4f}")
        _print_kv("Duration", f"{result.duration_seconds:.1f}s")
        _print_kv("Saved to", save_path)

        return 0

    except KeyboardInterrupt:
        print()
        _print_warning("Training interrupted by user")
        return 130
    except DatasetError as e:
        _print_error(f"Dataset error: {e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Dataset error details")
        return 1
    except TrainingError as e:
        _print_error(f"Training error: {e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Training error details")
        return 1
    except BackpropagateError as e:
        _print_error(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Error details")
        return 1
    except Exception as e:
        _print_error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


# =============================================================================
# COMMAND: multi-run
# =============================================================================

def cmd_multi_run(args: argparse.Namespace) -> int:
    """Execute the multi-run command."""
    from .trainer import Trainer
    from .multi_run import MultiRunTrainer, MultiRunConfig, MergeMode, RunResult

    _print_header("Backpropagate Multi-Run Training")

    if not args.data:
        _print_error("--data is required")
        return 1

    _print_info(f"Model: {args.model}")
    _print_info(f"Dataset: {args.data}")
    _print_info(f"Runs: {args.runs}")
    _print_info(f"Steps/run: {args.steps}")
    _print_info(f"Samples/run: {args.samples}")
    _print_info(f"Merge mode: {args.merge_mode}")

    try:
        config = MultiRunConfig(
            num_runs=args.runs,
            steps_per_run=args.steps,
            samples_per_run=args.samples,
            merge_mode=MergeMode(args.merge_mode),
            checkpoint_dir=args.output,
        )

        def on_run_complete(run_result: RunResult) -> None:
            _print_success(f"Run {run_result.run_index + 1} complete: loss={run_result.final_loss:.4f}")

        trainer = MultiRunTrainer(
            model=args.model,
            config=config,
            on_run_complete=on_run_complete,
        )

        print()
        result = trainer.run(args.data)

        print()
        _print_success("Multi-run training complete!")
        _print_kv("Total runs", str(result.total_runs))
        _print_kv("Final loss", f"{result.final_loss:.4f}")
        _print_kv("Total time", f"{result.total_duration_seconds:.1f}s")
        _print_kv("Output", result.final_checkpoint_path or args.output)

        return 0

    except KeyboardInterrupt:
        print()
        _print_warning("Training interrupted by user")
        return 130
    except BackpropagateError as e:
        _print_error(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Error details")
        return 1
    except Exception as e:
        _print_error(f"Multi-run failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


# =============================================================================
# COMMAND: export
# =============================================================================

def cmd_export(args: argparse.Namespace) -> int:
    """Execute the export command."""
    from .export import (
        export_lora,
        export_merged,
        export_gguf,
        register_with_ollama,
        GGUFQuantization,
    )

    _print_header("Backpropagate Export")

    try:
        model_path = safe_path(args.model_path, must_exist=True)
    except PathTraversalError as e:
        _print_error(f"Security error: {e}")
        return 1
    except FileNotFoundError:
        _print_error(f"Model path not found: {args.model_path}")
        return 1

    output_dir = Path(args.output) if args.output else model_path.parent / args.format

    _print_info(f"Model: {model_path}")
    _print_info(f"Format: {args.format}")
    _print_info(f"Output: {output_dir}")
    if args.format == "gguf":
        _print_info(f"Quantization: {args.quantization}")

    try:
        print()

        if args.format == "lora":
            result = export_lora(
                model=model_path,
                output_dir=output_dir,
            )
        elif args.format == "merged":
            # Need to load model for merged export
            from .trainer import load_model
            model, tokenizer = load_model(str(model_path))
            result = export_merged(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
            )
        elif args.format == "gguf":
            from .trainer import load_model
            model, tokenizer = load_model(str(model_path))
            result = export_gguf(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                quantization=args.quantization,
            )
        else:
            _print_error(f"Unknown format: {args.format}")
            return 1

        _print_success("Export complete!")
        _print_kv("Path", str(result.path))
        _print_kv("Size", f"{result.size_mb:.1f} MB")
        _print_kv("Time", f"{result.export_time_seconds:.1f}s")

        # Register with Ollama if requested
        if args.ollama and args.format != "gguf":
            print()
            ollama_name = args.ollama_name or model_path.name
            _print_info(f"Registering with Ollama as '{ollama_name}'...")

            if register_with_ollama(result.path, ollama_name):
                _print_success(f"Registered with Ollama: {ollama_name}")
                _print_info(f"Run with: ollama run {ollama_name}")
            else:
                _print_error("Failed to register with Ollama")
                return 1

        return 0

    except ExportError as e:
        _print_error(f"Export error: {e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Export error details")
        return 1
    except BackpropagateError as e:
        _print_error(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Error details")
        return 1
    except Exception as e:
        _print_error(f"Export failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


# =============================================================================
# COMMAND: info
# =============================================================================

def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    from .feature_flags import FEATURES, get_gpu_info, get_system_info
    from .config import settings
    from .gpu_safety import get_gpu_status

    _print_header("Backpropagate System Info")

    # System info
    sys_info = get_system_info()
    print(f"\n{Colors.BOLD}System{Colors.RESET}")
    _print_kv("Python", sys_info.get("python_version", "unknown"))
    _print_kv("Platform", sys_info.get("platform", "unknown"))
    _print_kv("PyTorch", sys_info.get("torch_version", "not installed"))
    _print_kv("CUDA", sys_info.get("cuda_version", "not available"))

    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"\n{Colors.BOLD}GPU{Colors.RESET}")
        _print_kv("Device", gpu_info.get("name", "unknown"))
        _print_kv("VRAM", f"{gpu_info.get('vram_total_gb', 0):.1f} GB")
        _print_kv("VRAM Free", f"{gpu_info.get('vram_free_gb', 0):.1f} GB")

        # Temperature if available
        try:
            status = get_gpu_status()
            if status and status.temperature_c:
                temp = status.temperature_c
                temp_color = Colors.GREEN if temp < 70 else (Colors.YELLOW if temp < 85 else Colors.RED)
                _print_kv("Temperature", f"{temp_color}{temp}C{Colors.RESET}")
        except ImportError:
            logger.debug("pynvml not available for temperature reading")
        except Exception as e:
            logger.debug(f"Could not read GPU temperature: {e}")
    else:
        print(f"\n{Colors.BOLD}GPU{Colors.RESET}")
        _print_kv("Status", f"{Colors.YELLOW}No GPU detected{Colors.RESET}")

    # Features
    print(f"\n{Colors.BOLD}Features{Colors.RESET}")
    for feature, available in FEATURES.items():
        status = f"{Colors.GREEN}[+]{Colors.RESET}" if available else f"{Colors.DIM}[-]{Colors.RESET}"
        print(f"  {status} {feature}")

    # Config
    print(f"\n{Colors.BOLD}Configuration{Colors.RESET}")
    _print_kv("Model", settings.model.name)
    _print_kv("Max seq length", str(settings.model.max_seq_length))
    _print_kv("LoRA r", str(settings.lora.r))
    _print_kv("Learning rate", str(settings.training.learning_rate))
    _print_kv("Output dir", settings.training.output_dir)

    return 0


# =============================================================================
# COMMAND: config
# =============================================================================

def cmd_ui(args: argparse.Namespace) -> int:
    """Execute the ui command to launch Gradio interface."""
    try:
        from .ui import launch
    except ImportError as e:
        _print_error("UI dependencies not installed")
        _print_info("Install with: pip install backpropagate[ui]")
        if args.verbose:
            logger.exception("Import error details")
        return 1

    _print_header("Backpropagate UI")
    _print_info(f"Port: {args.port}")
    if args.share:
        _print_info("Share: enabled (public URL)")

    # Handle authentication
    auth = None
    if args.auth:
        try:
            username, password = args.auth.split(":", 1)
            auth = (username, password)
            _print_info(f"Auth: enabled (user: {username})")
        except ValueError:
            _print_error("Invalid auth format. Use --auth username:password")
            return 1

    try:
        print()
        _print_info("Launching Gradio interface...")
        launch(port=args.port, share=args.share, auth=auth)
        return 0
    except KeyboardInterrupt:
        print()
        _print_info("UI stopped")
        return 0
    except Exception as e:
        _print_error(f"Failed to launch UI: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Execute the config command."""
    from .config import settings, reload_settings
    import json

    _print_header("Backpropagate Configuration")

    if args.reset:
        # Reset to defaults
        _print_warning("Config reset not yet implemented")
        _print_info("Delete your config file to reset to defaults")
        return 0

    if args.set:
        # Set a value
        _print_warning("Config --set not yet implemented")
        _print_info("Edit your config file directly or use environment variables")
        return 0

    # Show current config
    print(f"\n{Colors.BOLD}Model{Colors.RESET}")
    _print_kv("name", settings.model.name)
    _print_kv("max_seq_length", str(settings.model.max_seq_length))
    _print_kv("trust_remote_code", str(settings.model.trust_remote_code))

    print(f"\n{Colors.BOLD}LoRA{Colors.RESET}")
    _print_kv("r", str(settings.lora.r))
    _print_kv("lora_alpha", str(settings.lora.lora_alpha))
    _print_kv("lora_dropout", str(settings.lora.lora_dropout))
    _print_kv("target_modules", str(settings.lora.target_modules))

    print(f"\n{Colors.BOLD}Training{Colors.RESET}")
    _print_kv("learning_rate", str(settings.training.learning_rate))
    _print_kv("max_steps", str(settings.training.max_steps))
    _print_kv("batch_size", str(settings.training.per_device_train_batch_size))
    _print_kv("gradient_accumulation", str(settings.training.gradient_accumulation_steps))
    _print_kv("warmup_steps", str(settings.training.warmup_steps))
    _print_kv("output_dir", settings.training.output_dir)

    print(f"\n{Colors.BOLD}Data{Colors.RESET}")
    _print_kv("dataset_name", settings.data.dataset_name)
    _print_kv("dataset_split", settings.data.dataset_split)
    _print_kv("max_samples", str(settings.data.max_samples))
    _print_kv("text_column", settings.data.text_column)

    if os.name == "nt":
        print(f"\n{Colors.BOLD}Windows{Colors.RESET}")
        _print_kv("pre_tokenize", str(settings.windows.pre_tokenize))
        _print_kv("xformers_disabled", str(settings.windows.xformers_disabled))
        _print_kv("dataloader_workers", str(settings.windows.dataloader_num_workers))

    return 0


# =============================================================================
# PARSER
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="backprop",
        description="Backpropagate - Headless LLM Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  backprop train --data my_data.jsonl --steps 100
  backprop multi-run --data ultrachat --runs 5
  backprop export ./output/lora --format gguf
  backprop ui --port 7862
  backprop info
        """,
    )

    parser.add_argument(
        "--version", "-V",
        action="version",
        version="%(prog)s 0.1.0",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output including stack traces",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model",
        description="Fine-tune an LLM on your dataset",
    )
    train_parser.add_argument(
        "--model", "-m",
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="Model name or path (default: unsloth/Qwen2.5-7B-Instruct-bnb-4bit)",
    )
    train_parser.add_argument(
        "--data", "-d",
        required=True,
        help="Dataset path (JSONL, CSV) or HuggingFace dataset name",
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)",
    )
    train_parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Maximum samples to use from dataset",
    )
    train_parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size (default: auto)",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    train_parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    train_parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )
    train_parser.add_argument(
        "--no-unsloth",
        action="store_true",
        help="Disable Unsloth even if available",
    )
    train_parser.set_defaults(func=cmd_train)

    # multi-run command
    multi_parser = subparsers.add_parser(
        "multi-run",
        help="Multi-run training with SLAO merging",
        description="Train with multiple short runs and LoRA merging",
    )
    multi_parser.add_argument(
        "--model", "-m",
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="Model name or path",
    )
    multi_parser.add_argument(
        "--data", "-d",
        required=True,
        help="Dataset path or HuggingFace dataset name",
    )
    multi_parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of training runs (default: 5)",
    )
    multi_parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Steps per run (default: 100)",
    )
    multi_parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Samples per run (default: 1000)",
    )
    multi_parser.add_argument(
        "--merge-mode",
        choices=["slao", "simple"],
        default="slao",
        help="Merge mode (default: slao)",
    )
    multi_parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )
    multi_parser.set_defaults(func=cmd_multi_run)

    # export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export a trained model",
        description="Export model to LoRA, merged, or GGUF format",
    )
    export_parser.add_argument(
        "model_path",
        help="Path to trained model (LoRA adapter directory)",
    )
    export_parser.add_argument(
        "--format", "-f",
        choices=["lora", "merged", "gguf"],
        default="lora",
        help="Export format (default: lora)",
    )
    export_parser.add_argument(
        "--quantization", "-q",
        choices=["f16", "q8_0", "q5_k_m", "q4_k_m", "q4_0", "q2_k"],
        default="q4_k_m",
        help="GGUF quantization type (default: q4_k_m)",
    )
    export_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory",
    )
    export_parser.add_argument(
        "--ollama",
        action="store_true",
        help="Register GGUF with Ollama",
    )
    export_parser.add_argument(
        "--ollama-name",
        default=None,
        help="Name for Ollama model",
    )
    export_parser.set_defaults(func=cmd_export)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information",
        description="Display GPU, features, and configuration info",
    )
    info_parser.set_defaults(func=cmd_info)

    # config command
    config_parser = subparsers.add_parser(
        "config",
        help="View or modify configuration",
        description="View or modify Backpropagate configuration",
    )
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration (default)",
    )
    config_parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        help="Set a configuration value",
    )
    config_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset configuration to defaults",
    )
    config_parser.set_defaults(func=cmd_config)

    # ui command
    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch the Gradio web interface",
        description="Launch interactive web UI for training, export, and monitoring",
    )
    ui_parser.add_argument(
        "--port", "-p",
        type=int,
        default=7862,
        help="Port to run the server on (default: 7862)",
    )
    ui_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link (requires --auth for security)",
    )
    ui_parser.add_argument(
        "--auth",
        metavar="USER:PASS",
        help="Authentication credentials (format: username:password)",
    )
    ui_parser.set_defaults(func=cmd_ui)

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main(argv: Optional[list] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    # Execute the command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
