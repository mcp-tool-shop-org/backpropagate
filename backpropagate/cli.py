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
import re
import sys
from pathlib import Path

from . import __version__
from .exceptions import (
    BackpropagateError,
    DatasetError,
    ExportError,
    PartialSuccess,
    TrainingError,
    UserInputError,
)
from .security import PathTraversalError, safe_path

logger = logging.getLogger(__name__)

__all__ = ["main", "create_parser"]


# =============================================================================
# EXIT CODES (Ship Gate B2)
# =============================================================================
# 0 = success
# 1 = user error          (bad args, missing input, validation failure)
# 2 = runtime error       (unexpected crash, IO failure, internal bug)
# 3 = partial success     (operation completed with some failures)
# 130 = interrupted       (Ctrl+C / SIGINT — POSIX convention)

EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_RUNTIME_ERROR = 2
EXIT_PARTIAL_SUCCESS = 3
EXIT_INTERRUPTED = 130


# Patterns used to redact common secret-bearing tokens from non-verbose
# error output. Conservative on purpose: prefer false-positives (a redacted
# string) over leaking. Use `_print_error_redacted` for unexpected-exception
# paths where the exception message may quote a downstream library payload.
_SECRET_PATTERNS = [
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"), "Bearer <REDACTED>"),
    (re.compile(r"sk-[A-Za-z0-9]{8,}"), "sk-<REDACTED>"),
    (re.compile(r"hf_[A-Za-z0-9]{8,}"), "hf_<REDACTED>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA<REDACTED>"),
    (re.compile(r"(password|passwd|pwd|token|api[_-]?key)\s*[=:]\s*\S+", re.IGNORECASE),
     r"\1=<REDACTED>"),
]


def _redact_secrets(text: str) -> str:
    """Best-effort redaction of common secret patterns in error output."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


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


def _print_error_redacted(exc: BaseException, prefix: str = "") -> None:
    """
    Print an unexpected exception with secret-bearing strings redacted.

    Use for ``except Exception`` paths in non-verbose mode, where the
    exception message may quote a downstream library payload that happens
    to embed credentials (HTTP Authorization headers, signed URLs, JWT,
    etc.). Verbose mode is exempt — the user opted in to full output.
    """
    redacted = _redact_secrets(str(exc))
    msg = f"{prefix}{type(exc).__name__}: {redacted}"
    _print_error(msg)


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
    """
    Execute the train command.

    Exit codes (Ship Gate B2):
        0   training completed successfully
        1   user error (missing/invalid args, dataset validation failure)
        2   runtime error (model load failure, GPU OOM, IO error, unexpected crash)
        130 interrupted (Ctrl+C)
    """
    from .trainer import Trainer, TrainingCallback

    _print_header("Backpropagate Training")

    # argparse marks --data as required (see create_parser), so this only fires
    # if cmd_train is called programmatically without going through the parser.
    if not args.data:
        _print_error("--data is required")
        return EXIT_USER_ERROR

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
        _print_success("Training complete!")
        _print_kv("Final loss", f"{result.final_loss:.4f}")
        _print_kv("Duration", f"{result.duration_seconds:.1f}s")
        _print_kv("Saved to", str(save_path))

        return EXIT_OK

    except KeyboardInterrupt:
        print()
        _print_warning("Training interrupted by user")
        return EXIT_INTERRUPTED
    except UserInputError as e:
        _print_error(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("User input error details")
        return EXIT_USER_ERROR
    except DatasetError as e:
        # DatasetError covers user-supplied dataset issues (missing file,
        # parse failure, validation) — user-actionable.
        _print_error(f"Dataset error: {e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Dataset error details")
        return EXIT_USER_ERROR
    except TrainingError as e:
        # TrainingError covers model load failures, training aborts, checkpoint
        # IO — runtime-level problems the user generally cannot pre-validate.
        _print_error(f"Training error: {e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Training error details")
        return EXIT_RUNTIME_ERROR
    except PartialSuccess as e:
        _print_warning(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        return EXIT_PARTIAL_SUCCESS
    except BackpropagateError as e:
        # Default for any other structured BackpropagateError subclass —
        # treat as runtime error unless it is explicitly user-actionable.
        _print_error(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Error details")
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        # Unexpected — re-raise under --verbose so users can see the full
        # traceback; otherwise emit a redacted single-line message and exit
        # with the runtime-error code.
        if args.verbose:
            _print_error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
        else:
            _print_error_redacted(e, prefix="Training failed: ")
            _print_info("Run with --verbose for full traceback")
        return EXIT_RUNTIME_ERROR


# =============================================================================
# COMMAND: multi-run
# =============================================================================

def cmd_multi_run(args: argparse.Namespace) -> int:
    """
    Execute the multi-run command.

    Exit codes (Ship Gate B2):
        0   all runs completed successfully
        1   user error (missing args, invalid merge mode)
        2   runtime error (model load, GPU OOM, IO, unexpected crash)
        3   partial success (some runs failed but the overall result is usable)
        130 interrupted (Ctrl+C)
    """
    from .multi_run import MergeMode, MultiRunConfig, MultiRunTrainer, RunResult

    _print_header("Backpropagate Multi-Run Training")

    if not args.data:
        _print_error("--data is required")
        return EXIT_USER_ERROR

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
        _print_kv("Output", str(result.final_checkpoint_path or args.output))

        # Some MultiRunResult shapes carry a ``failed_runs`` counter — if any
        # runs failed but a final checkpoint exists, surface as partial success.
        failed_runs = getattr(result, "failed_runs", 0) or 0
        if failed_runs > 0:
            _print_warning(
                f"{failed_runs}/{result.total_runs} runs failed (partial success)"
            )
            return EXIT_PARTIAL_SUCCESS

        return EXIT_OK

    except KeyboardInterrupt:
        print()
        _print_warning("Training interrupted by user")
        return EXIT_INTERRUPTED
    except UserInputError as e:
        _print_error(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        return EXIT_USER_ERROR
    except DatasetError as e:
        _print_error(f"Dataset error: {e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Dataset error details")
        return EXIT_USER_ERROR
    except PartialSuccess as e:
        _print_warning(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        return EXIT_PARTIAL_SUCCESS
    except ValueError as e:
        # Invalid merge_mode etc. — argparse choices=['slao','simple'] already
        # guards the common case, so this catches programmatic mis-calls.
        _print_error(f"Invalid argument: {e}")
        return EXIT_USER_ERROR
    except BackpropagateError as e:
        _print_error(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Error details")
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        if args.verbose:
            _print_error(f"Multi-run failed: {e}")
            import traceback
            traceback.print_exc()
        else:
            _print_error_redacted(e, prefix="Multi-run failed: ")
            _print_info("Run with --verbose for full traceback")
        return EXIT_RUNTIME_ERROR


# =============================================================================
# COMMAND: export
# =============================================================================

def cmd_export(args: argparse.Namespace) -> int:
    """
    Execute the export command.

    Exit codes (Ship Gate B2):
        0   export (and optional Ollama registration) succeeded
        1   user error (path traversal, missing model, malformed path, unknown format)
        2   runtime error (export failure, disk full, Ollama registration crash)
        130 interrupted (Ctrl+C)
    """
    from .export import (
        export_gguf,
        export_lora,
        export_merged,
        register_with_ollama,
    )

    _print_header("Backpropagate Export")

    # Null-byte rejection up front — Path.resolve on some platforms will raise
    # a less-helpful OSError and obscure the real cause.
    if "\x00" in str(args.model_path):
        _print_error("Model path contains null byte")
        return EXIT_USER_ERROR

    # Resolve the input model path inside the parent directory it points at.
    # This gives `safe_path` an explicit base to enforce traversal protection
    # against, instead of relying on the (newly stricter) default behavior.
    try:
        input_base = Path(args.model_path).expanduser().parent.resolve()
    except (OSError, ValueError) as e:
        _print_error(f"Invalid model path: {e}")
        return EXIT_USER_ERROR

    try:
        model_path = safe_path(
            args.model_path,
            must_exist=True,
            allowed_base=input_base,
        )
    except PathTraversalError as e:
        _print_error(f"Security error: {e}")
        return EXIT_USER_ERROR
    except FileNotFoundError:
        _print_error(f"Model path not found: {args.model_path}")
        return EXIT_USER_ERROR
    except (OSError, ValueError) as e:
        # Malformed path (null byte, invalid drive letter, bad UNC, etc.)
        _print_error(f"Invalid model path: {e}")
        return EXIT_USER_ERROR

    # Validate / resolve the output directory inside the current working
    # directory (the conventional "export-to-here" pattern). Users who want
    # to write elsewhere can pass an absolute --output and we accept it.
    cwd = Path.cwd()
    raw_output: Path
    if args.output:
        raw_output = Path(args.output).expanduser()
    else:
        raw_output = model_path.parent / args.format

    try:
        if raw_output.is_absolute():
            # Absolute output paths bypass the cwd-bound check but are still
            # normalized; safe_path with no allowed_base will reject ".."
            # patterns under the new default behavior.
            output_dir = safe_path(str(raw_output), must_exist=False)
        else:
            output_dir = safe_path(
                str(raw_output),
                must_exist=False,
                allowed_base=cwd,
            )
    except PathTraversalError as e:
        _print_error(f"Security error: output path escapes its allowed base: {e}")
        return EXIT_USER_ERROR
    except (OSError, ValueError) as e:
        _print_error(f"Invalid output path: {e}")
        return EXIT_USER_ERROR

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
            return EXIT_USER_ERROR

        _print_success("Export complete!")
        _print_kv("Path", str(result.path))
        _print_kv("Size", f"{result.size_mb:.1f} MB")
        _print_kv("Time", f"{result.export_time_seconds:.1f}s")

        # Register with Ollama if requested
        if args.ollama and args.format == "gguf":
            print()
            ollama_name = args.ollama_name or model_path.name
            _print_info(f"Registering with Ollama as '{ollama_name}'...")

            if register_with_ollama(result.path, ollama_name):
                _print_success(f"Registered with Ollama: {ollama_name}")
                _print_info(f"Run with: ollama run {ollama_name}")
            else:
                # Export succeeded; only the optional Ollama step failed.
                # Treat as partial success so callers can distinguish from
                # outright failure.
                _print_error("Failed to register with Ollama")
                return EXIT_PARTIAL_SUCCESS

        return EXIT_OK

    except UserInputError as e:
        _print_error(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        return EXIT_USER_ERROR
    except ExportError as e:
        # ExportError covers GGUF / merge / Ollama failures — runtime-level.
        _print_error(f"Export error: {e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Export error details")
        return EXIT_RUNTIME_ERROR
    except PartialSuccess as e:
        _print_warning(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        return EXIT_PARTIAL_SUCCESS
    except BackpropagateError as e:
        _print_error(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Error details")
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        if args.verbose:
            _print_error(f"Export failed: {e}")
            import traceback
            traceback.print_exc()
        else:
            _print_error_redacted(e, prefix="Export failed: ")
            _print_info("Run with --verbose for full traceback")
        return EXIT_RUNTIME_ERROR


# =============================================================================
# COMMAND: info
# =============================================================================

def cmd_info(_args: argparse.Namespace) -> int:
    """Execute the info command."""
    from .config import settings
    from .feature_flags import FEATURES, get_gpu_info, get_system_info
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
        feature_status = f"{Colors.GREEN}[+]{Colors.RESET}" if available else f"{Colors.DIM}[-]{Colors.RESET}"
        print(f"  {feature_status} {feature}")

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

def _env_flag(name: str, default: bool) -> bool:
    """
    Read a boolean-flavored environment variable.

    Truthy values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Falsy values:  ``0``, ``false``, ``no``, ``off``.
    Anything else (or unset) returns ``default``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def cmd_ui(args: argparse.Namespace) -> int:
    """
    Execute the ui command to launch the Gradio interface.

    Exit codes (Ship Gate B2):
        0   UI launched and exited cleanly (including Ctrl+C)
        1   user error — UI extra not installed, malformed --auth, OR
            --share supplied without --auth while the
            BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE setting is on
            (default: on; set to ``false`` to explicitly opt out)
        2   runtime error — Gradio launch failure
    """
    try:
        from .ui import launch
    except ImportError:
        _print_error("UI dependencies not installed")
        _print_info("Install with: pip install backpropagate[ui]")
        if args.verbose:
            logger.exception("Import error details")
        return EXIT_USER_ERROR

    _print_header("Backpropagate UI")
    _print_info(f"Port: {args.port}")
    if args.share:
        _print_info("Share: enabled (public URL)")

    # ------------------------------------------------------------------ #
    # F-001 enforcement: --share must come with --auth unless the user
    # explicitly opted out via env var.
    # ------------------------------------------------------------------ #
    require_auth_for_share = _env_flag(
        "BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE",
        default=True,
    )
    if args.share and not args.auth and require_auth_for_share:
        _print_error(
            "[INPUT_AUTH_REQUIRED]: --share requires --auth for security."
        )
        _print_info(
            "Hint: pass --auth user:password, OR set "
            "BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false to "
            "explicitly opt out."
        )
        return EXIT_USER_ERROR

    # Handle authentication
    auth = None
    if args.auth:
        try:
            username, password = args.auth.split(":", 1)
            auth = (username, password)
            _print_info(f"Auth: enabled (user: {username})")
        except ValueError:
            _print_error("Invalid auth format. Use --auth username:password")
            return EXIT_USER_ERROR

    # Loud warning if the user explicitly opted out of the auth requirement
    # while sharing publicly. The frontend agent layers a Gradio-side warning
    # too; this stderr line is the CLI signal.
    if args.share and not args.auth and not require_auth_for_share:
        _print_warning(
            "Sharing publicly with NO authentication. "
            "BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false was set "
            "explicitly; anyone with the *.gradio.live URL can use this UI."
        )

    try:
        print()
        _print_info("Launching Gradio interface...")
        launch(port=args.port, share=args.share, auth=auth)
        return EXIT_OK
    except KeyboardInterrupt:
        print()
        _print_info("UI stopped")
        return EXIT_OK
    except UserInputError as e:
        _print_error(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        return EXIT_USER_ERROR
    except BackpropagateError as e:
        _print_error(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        if args.verbose:
            logger.exception("Error details")
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        if args.verbose:
            _print_error(f"Failed to launch UI: {e}")
            import traceback
            traceback.print_exc()
        else:
            _print_error_redacted(e, prefix="Failed to launch UI: ")
            _print_info("Run with --verbose for full traceback")
        return EXIT_RUNTIME_ERROR


def cmd_config(args: argparse.Namespace) -> int:
    """Execute the config command."""

    from .config import settings

    _print_header("Backpropagate Configuration")

    if args.reset:
        # Reset to defaults
        _print_info("Config editing via CLI is planned. For now, set BACKPROPAGATE_* environment variables or edit your .env file.")
        return 0

    if args.set:
        # Set a value
        _print_info("Config editing via CLI is planned. For now, set BACKPROPAGATE_* environment variables or edit your .env file.")
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

Exit codes (Ship Gate B2):
  0   success
  1   user error  (bad args, missing input, validation, --share without --auth)
  2   runtime error (model load, GPU OOM, IO, unexpected crash)
  3   partial success (e.g. some multi-runs failed; export ok but Ollama register failed)
  130 interrupted (Ctrl+C)
        """,
    )

    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}",
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
        help=(
            "Create a public shareable link. REQUIRES --auth USER:PASS "
            "(enforced — exit code 1 if omitted). Set the env var "
            "BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false to "
            "explicitly opt out (you will see a loud unauthenticated warning)."
        ),
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

def main(argv: list | None = None) -> int:
    """Main entry point for the CLI.

    See ``backprop --help`` for the documented Ship Gate B2 exit-code contract.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        _print_error("No command specified. Run backprop --help for usage.")
        return EXIT_USER_ERROR

    # Execute the command
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
