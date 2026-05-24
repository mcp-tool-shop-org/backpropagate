"""
Backpropagate CLI
=================

Command-line interface for LLM fine-tuning.

Usage:
    # Train a model
    backprop train --model Qwen/Qwen2.5-7B-Instruct --data my_data.jsonl --steps 100

    # Export to GGUF
    backprop export ./output/lora --format gguf --quantization q4_k_m

    # Multi-run training
    backprop multi-run --model Qwen/Qwen2.5-7B-Instruct --data ultrachat --runs 5

    # Launch web UI
    backprop ui --port 7862

    # Show system info
    backprop info
"""

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

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


# =============================================================================
# ARGPARSE TYPE VALIDATORS
# =============================================================================
# Defensive-coding helpers used by `create_parser` to reject obviously wrong
# numeric values at parse time (argparse exits with code 2 by convention)
# instead of letting them propagate to a deep stack trace much later.

def _positive_int(value: str) -> int:
    """argparse type for integers that must be > 0.

    Raises:
        argparse.ArgumentTypeError: if value is not parseable as int or <= 0.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be positive (> 0), got {n}")
    return n


def _non_negative_int(value: str) -> int:
    """argparse type for integers that must be >= 0 (allows 0)."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative (>= 0), got {n}")
    return n


def _port_int(value: str) -> int:
    """argparse type for TCP port numbers (1..65535)."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected an integer port, got {value!r}")
    if not (1 <= n <= 65535):
        raise argparse.ArgumentTypeError(
            f"port must be in range 1..65535, got {n}"
        )
    return n


def _positive_float(value: str) -> float:
    """argparse type for floats that must be > 0 (e.g. learning rate)."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a float, got {value!r}")
    if n <= 0.0:
        raise argparse.ArgumentTypeError(
            f"must be positive (> 0), got {n}"
        )
    return n


# Patterns used to redact common secret-bearing tokens from non-verbose
# error output. The previous catch-all (`(password|...)\s*[=:]\s*\S+`) had
# two defects: (1) it matched plain prose like "the token: abc is wrong"
# (false-positive — any sentence with a configuration-flavored word followed
# by punctuation got mangled), and (2) the replacement template `\1=<REDACTED>`
# hardcoded `=` regardless of whether the input used `:` or `=` (silently
# rewriting the operator's input shape).
#
# Current shape:
# - High-signal prefixes (Bearer, sk-, hf_, AKIA) trigger redaction on their
#   own — these have low false-positive rates by construction.
# - The keyword-prefixed pattern now requires an `=` or `:` AND at least 8
#   non-space characters of value (with at least one digit or special char)
#   — high-entropy enough to filter out prose. The separator is captured
#   in group 2 and re-emitted in the replacement so `token: foo` stays
#   `token: <REDACTED>` (NOT `token=<REDACTED>`).
#
# Negative-test expectations (NOT to be redacted):
#   "the token: abc is wrong"        → unchanged (value too short, prose-like)
#   "try a different password"       → unchanged (no separator)
#   "Authorization: Bearer xyz"      → "Authorization: Bearer <REDACTED>"
#   "api_key=EXAMPLE-NOT-A-REAL-KEY"   → "api_key=<REDACTED>"
#   "password=hunter2!@#secret_key"  → "password=<REDACTED>"
_SECRET_PATTERNS = [
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"), "Bearer <REDACTED>"),
    (re.compile(r"sk-[A-Za-z0-9]{8,}"), "sk-<REDACTED>"),
    (re.compile(r"hf_[A-Za-z0-9]{8,}"), "hf_<REDACTED>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA<REDACTED>"),
    # Keyword-prefixed credentials: require key=value OR key:value shape with
    # a high-entropy value (>=8 non-space chars including >=1 digit-or-special).
    # The separator (group 2) is preserved in the replacement.
    (
        re.compile(
            r"(password|passwd|pwd|secret|token|api[_-]?key)"
            r"(\s*[=:]\s*)"
            r'(?=[^\s]*[\d!@#$%^&*()+\-./?_=])'  # lookahead: at least one digit/special
            r"[^\s]{8,}",
            re.IGNORECASE,
        ),
        r"\1\2<REDACTED>",
    ),
]


def _redact_secrets(text: str) -> str:
    """
    Best-effort redaction of common secret patterns in error output.

    Designed to leave human-readable prose intact (e.g. "the token: abc is
    wrong" is NOT redacted because "abc" is too short / has no digit) while
    catching realistic credential leaks (e.g. "api_key=EXAMPLE-NOT-A-REAL-KEY"
    is redacted to "api_key=<REDACTED>"). The separator character (``:`` vs
    ``=``) is preserved in the output to avoid silently rewriting the
    operator's input shape.
    """
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
    import uuid

    from .logging_config import bind_run_context, get_logger
    from .trainer import Trainer, TrainingCallback

    _print_header("Backpropagate Training")

    # C-CLI-007 first-run friendliness: argparse no longer marks --data as
    # required so we get to print a multi-line "try one of these" message
    # instead of argparse's terse required-argument error.
    if not args.data:
        _print_error("No dataset specified.")
        _print_info("Try one of:")
        _print_info("  backprop train --data my_data.jsonl --steps 100")
        _print_info("  backprop train --data huggingface_dataset_name --steps 100")
        _print_info("See `backprop train --help` for all options.")
        return EXIT_USER_ERROR

    # BRIDGE-B-002: reuse the CLI-level run_id minted by main() (stashed on
    # args.cli_run_id). The short 12-char form is preserved for the stderr
    # banner because operators were already trained on it; the FULL run_id
    # is bound to structlog so JSON consumers see the long form. Falling
    # back to a fresh UUID keeps the handler robust when called directly
    # from tests that bypass main().
    cli_run_id_full = getattr(args, "cli_run_id", None) or uuid.uuid4().hex
    cli_run_id = cli_run_id_full[:12]
    # Re-bind in case the handler is called directly without main()'s setup.
    try:
        bind_run_context(run_id=cli_run_id_full, subcommand="train")
    except Exception:  # noqa: BLE001  # nosec B110 — best-effort observability bind; must not abort CLI
        pass
    # Structured log line — routes to JSON when BACKPROPAGATE_LOG_JSON=1,
    # pretty console otherwise. Replaces the legacy stderr-only print().
    try:
        get_logger(__name__).info(
            "train_invoked",
            cli_run_id=cli_run_id_full,
            model=args.model,
            dataset=str(args.data),
        )
    except Exception:  # noqa: BLE001  # nosec B110 — best-effort observability; must not abort CLI
        pass
    print(
        f"[INFO] Run ID: {cli_run_id} — share with support if asking for help.",
        file=sys.stderr,
    )

    _print_info(f"Model: {args.model}")
    _print_info(f"Dataset: {args.data}")
    _print_info(f"Steps: {args.steps}")
    if args.samples:
        _print_info(f"Samples: {args.samples}")

    try:
        # C-CLI-002 progress feedback: phase banner before the model load so a
        # 30-300s silence doesn't read like the tool is wedged. transformers /
        # datasets already emit tqdm progress to stderr during from_pretrained,
        # so the banner contextualises that output instead of competing with it.
        _print_info("==> Loading model (this may take 30s-3min for 7B models)...")

        # Create trainer
        trainer = Trainer(
            model=args.model,
            lora_r=args.lora_r,
            learning_rate=args.lr,
            batch_size=args.batch_size if args.batch_size != "auto" else "auto",
            output_dir=args.output,
            use_unsloth=not args.no_unsloth,
        )
        # F-002: pass resume hint through to trainer.train() once we get there.
        resume_hint = getattr(args, "resume", None)

        # Progress callback
        progress = ProgressBar(args.steps, prefix="Training: ")

        def on_step(step: int, loss: float) -> None:
            progress.update(step, f"loss={loss:.4f}")

        callback = TrainingCallback(on_step=on_step)

        # C-CLI-002 phase banner: training begins. The ProgressBar handles
        # per-step output but a banner gives the user a clear "we're now in
        # the training loop" signal so dataset-load time vs step time is
        # distinguishable in the terminal stream.
        _print_info(f"==> Training ({args.steps} steps)...")
        print()  # Blank line before progress

        # Train
        result = trainer.train(
            dataset=args.data,
            steps=args.steps,
            samples=args.samples,
            callback=callback,
            resume_from=resume_hint,
        )

        progress.finish()

        # Save
        save_path = trainer.save(args.output)

        print()
        _print_success("Training complete!")
        _print_kv("Final loss", f"{result.final_loss:.4f}")
        _print_kv("Duration", f"{result.duration_seconds:.1f}s")
        _print_kv("Saved to", str(save_path))
        # Surface the trainer's internal run_id if it leaked onto the result
        # object so the operator can grep manifests against it post-hoc.
        trainer_run_id = getattr(result, "run_id", None)
        if trainer_run_id:
            _print_kv("Trainer run_id", str(trainer_run_id))

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
    import uuid

    from .logging_config import bind_run_context, get_logger
    from .multi_run import MergeMode, MultiRunConfig, MultiRunTrainer, RunResult

    _print_header("Backpropagate Multi-Run Training")

    # C-CLI-007 first-run friendliness (parallel to cmd_train).
    if not args.data:
        _print_error("No dataset specified.")
        _print_info("Try one of:")
        _print_info("  backprop multi-run --data my_data.jsonl --runs 5 --steps 100")
        _print_info("  backprop multi-run --data huggingface_dataset_name --runs 5")
        _print_info("See `backprop multi-run --help` for all options.")
        return EXIT_USER_ERROR

    # BRIDGE-B-002: reuse main()'s cli_run_id so trainer / multi_run logs
    # carry the same correlation token operators see at the top of stderr.
    cli_run_id_full = getattr(args, "cli_run_id", None) or uuid.uuid4().hex
    cli_run_id = cli_run_id_full[:12]
    try:
        bind_run_context(run_id=cli_run_id_full, subcommand="multi-run")
    except Exception:  # noqa: BLE001  # nosec B110 — best-effort observability; must not abort CLI
        pass
    try:
        get_logger(__name__).info(
            "multi_run_invoked",
            cli_run_id=cli_run_id_full,
            model=args.model,
            runs=args.runs,
        )
    except Exception:  # noqa: BLE001  # nosec B110 — best-effort observability; must not abort CLI
        pass
    print(
        f"[INFO] Run ID: {cli_run_id} — share with support if asking for help.",
        file=sys.stderr,
    )

    _print_info(f"Model: {args.model}")
    _print_info(f"Dataset: {args.data}")
    _print_info(f"Runs: {args.runs}")
    _print_info(f"Steps/run: {args.steps}")
    _print_info(f"Samples/run: {args.samples}")
    _print_info(f"Merge mode: {args.merge_mode}")

    try:
        # C-CLI-002 phase banner — the multi-run trainer also loads + tokenises
        # before the first step. Give the operator a "we're in setup, not
        # wedged" signal.
        _print_info(
            "==> Setting up multi-run trainer "
            "(model load + dataset tokenisation may take several minutes)..."
        )
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
            resume_from=getattr(args, "resume", None),  # F-002
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
    import uuid

    from .export import (
        export_gguf,
        export_lora,
        export_merged,
        register_with_ollama,
    )
    from .logging_config import bind_run_context, get_logger

    _print_header("Backpropagate Export")

    # C-CLI-007 first-run friendliness — bare ``backprop export`` (no
    # positional) now lands here with model_path=None instead of being
    # rejected by argparse with the terse required-positional error.
    if args.model_path is None:
        _print_error("No model path specified.")
        _print_info("Try one of:")
        _print_info("  backprop export ./output/lora --format lora")
        _print_info("  backprop export ./output/lora --format gguf --quantization q4_k_m")
        _print_info(
            "  backprop export ./output/lora --format gguf --ollama "
            "--ollama-name my-model"
        )
        _print_info("See `backprop export --help` for all options.")
        return EXIT_USER_ERROR

    # Null-byte rejection up front — Path.resolve on some platforms will raise
    # a less-helpful OSError and obscure the real cause.
    if "\x00" in str(args.model_path):
        _print_error("Model path contains null byte")
        return EXIT_USER_ERROR

    # BRIDGE-B-002: reuse main()'s cli_run_id so a single token correlates
    # CLI output / structured logs / model_card.md / Hub push metadata.
    cli_run_id_full = getattr(args, "cli_run_id", None) or uuid.uuid4().hex
    cli_run_id = cli_run_id_full[:12]
    try:
        bind_run_context(run_id=cli_run_id_full, subcommand="export")
    except Exception:  # noqa: BLE001  # nosec B110 — best-effort observability; must not abort CLI
        pass
    try:
        get_logger(__name__).info(
            "export_invoked",
            cli_run_id=cli_run_id_full,
            format=args.format,
        )
    except Exception:  # noqa: BLE001  # nosec B110 — best-effort observability; must not abort CLI
        pass
    print(
        f"[INFO] Run ID: {cli_run_id} — share with support if asking for help.",
        file=sys.stderr,
    )

    # Anchor the input model path to cwd by default — this matches the
    # documented "export model from current dir / project dir" pattern and
    # gives safe_path a meaningful containing box (the previous Wave 1 attempt
    # passed `Path(args.model_path).parent.resolve()`, which is tautological:
    # any path is inside its own parent, so the check collapsed to null-byte +
    # exists only). Users with models stored elsewhere can pass an absolute
    # path; we accept it but log a WARN to flag the cwd-sandbox opt-out.
    cwd_resolved = Path.cwd().resolve()
    try:
        model_path_raw = Path(args.model_path).expanduser()
    except (OSError, ValueError) as e:
        _print_error(f"Invalid model path: {e}")
        return EXIT_USER_ERROR

    try:
        if model_path_raw.is_absolute():
            _print_warning(
                f"Absolute --model_path supplied; opting out of cwd sandbox: "
                f"{model_path_raw}"
            )
            # Absolute-input: safe_path's default-stricter mode rejects ".."
            # in absolute paths and outside-cwd ".." patterns.
            model_path = safe_path(str(model_path_raw), must_exist=True)
        else:
            model_path = safe_path(
                str(model_path_raw),
                must_exist=True,
                allowed_base=cwd_resolved,
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
    # to write elsewhere can pass an absolute --output and we accept it (with
    # a loud WARN to mirror the input-path opt-out signal).
    raw_output: Path
    if args.output:
        raw_output = Path(args.output).expanduser()
    else:
        raw_output = model_path.parent / args.format

    try:
        if raw_output.is_absolute():
            if args.output:
                # Only warn when the user explicitly supplied an absolute
                # --output — auto-derived defaults under model_path.parent
                # are an implementation detail, not a sandbox opt-out.
                _print_warning(
                    f"Absolute --output supplied; opting out of cwd sandbox: "
                    f"{raw_output}"
                )
            # Absolute output paths bypass the cwd-bound check but are still
            # normalized; safe_path with no allowed_base will reject ".."
            # patterns under the new default behavior.
            output_dir = safe_path(str(raw_output), must_exist=False)
        else:
            output_dir = safe_path(
                str(raw_output),
                must_exist=False,
                allowed_base=cwd_resolved,
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

        # F-004: --no-model-card opts out; default (omitting the flag) emits.
        emit_card = not getattr(args, "no_model_card", False)

        if args.format == "lora":
            # C-CLI-002 phase banner — LoRA-only export is fast (no model
            # load) but the banner still adds a "we're exporting" signal.
            _print_info("==> Exporting LoRA adapter...")
            result = export_lora(
                model=model_path,
                output_dir=output_dir,
                emit_model_card=emit_card,
                output_root=output_dir.parent,
            )
        elif args.format == "merged":
            # C-CLI-002 phase banner — merged export loads the full model.
            from .trainer import load_model
            _print_info("==> Loading model for merge (this may take 30s-3min)...")
            model, tokenizer = load_model(str(model_path))
            _print_info("==> Merging adapters and writing merged checkpoint...")
            result = export_merged(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                emit_model_card=emit_card,
                output_root=output_dir.parent,
            )
        elif args.format == "gguf":
            # C-CLI-002 phase banner — GGUF export does model load AND
            # quantization (each 60-300s for 7B). Surface both phases.
            from .trainer import load_model
            _print_info("==> Loading model for GGUF export (this may take 30s-3min)...")
            model, tokenizer = load_model(str(model_path))
            _print_info(
                f"==> Quantizing to {args.quantization} "
                "(this may take several minutes for 7B models)..."
            )
            result = export_gguf(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                quantization=args.quantization,
                emit_model_card=emit_card,
                output_root=output_dir.parent,
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

        # F-001: one-shot export + Hub push.
        if getattr(args, "push_to_hub", None):
            from .export import push_to_hub as _hub_push

            print()
            _print_info(f"==> Pushing to Hugging Face Hub: {args.push_to_hub}")
            # For directory exports the local upload root is output_dir; for
            # the single-file GGUF case it's the GGUF's parent.
            local_root = (
                result.path if result.path.is_dir() else result.path.parent
            )
            try:
                url = _hub_push(
                    local_path=local_root,
                    repo_id=args.push_to_hub,
                    token=getattr(args, "hub_token", None),
                    private=getattr(args, "hub_private", False),
                )
                _print_success(f"Pushed to Hub: {url}")
            except BackpropagateError as e:
                _print_error(f"Hub push failed: {e.message}")
                if e.suggestion:
                    _print_info(f"Suggestion: {e.suggestion}")
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

def _detect_installed_versions() -> dict[str, str]:
    """Best-effort version probe for the heavy optional dependencies.

    Uses ``importlib.metadata`` (no actual import of the package) to avoid
    triggering side-effectful module init (especially unsloth's torch.compile
    hook registration). Returns ``"not installed"`` for any package that
    isn't on the install path. C-CLI-008 / Stage C richer info.
    """
    from importlib import metadata

    pkgs = [
        "transformers",
        "datasets",
        "trl",
        "peft",
        "unsloth",
        "torch",
        "gradio",
        "pydantic",
        "wandb",
    ]
    out: dict[str, str] = {}
    for name in pkgs:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
        except Exception:  # nosec B110 — defensive: don't let info crash
            out[name] = "unknown"
    return out


def _enumerate_env_vars() -> list[dict[str, str]]:
    """Walk the pydantic-settings models in :mod:`backpropagate.config` and
    return one dict per ``BACKPROPAGATE_*`` env var the runtime honours.

    Each dict has ``env_var`` / ``default`` / ``type`` / ``description``
    keys. The walk skips secret-flagged fields' default values (replaced
    with ``"<secret>"``) so a leaked ``BACKPROPAGATE_SECURITY__AUTH_PASSWORD``
    default never reaches stdout. When pydantic-settings is not installed
    we fall back to a small hand-curated list of the load-bearing env
    vars from the dataclass fallback in :mod:`backpropagate.config`.

    Returns:
        Sorted list of env-var descriptors, alphabetical by ``env_var``.
        Deterministic so snapshot-tests are stable across Python versions.
    """
    from .config import PYDANTIC_SETTINGS_AVAILABLE, Settings

    rows: list[dict[str, str]] = []

    if PYDANTIC_SETTINGS_AVAILABLE:
        # Walk every nested sub-config on Settings(). Each sub-config carries
        # an ``env_prefix`` in its model_config that defines the BACKPROPAGATE_*
        # namespace. We construct the env-var name by concatenating the prefix
        # with the field name UPPERCASED.
        for sub_field_name, sub_field_info in Settings.model_fields.items():
            sub_cls = sub_field_info.annotation
            # Only nested-config classes have model_fields; primitive fields
            # (e.g. Settings.version, Settings.name) are at the top level and
            # use BACKPROPAGATE_<NAME> directly.
            if not hasattr(sub_cls, "model_fields"):
                env_name = f"BACKPROPAGATE_{sub_field_name.upper()}"
                rows.append({
                    "env_var": env_name,
                    "default": _safe_default(sub_field_info),
                    "type": _type_name(sub_field_info.annotation),
                    "description": (sub_field_info.description or "").strip(),
                })
                continue

            if sub_cls is None:
                continue
            sub_cfg = getattr(sub_cls, "model_config", {})
            env_prefix = sub_cfg.get("env_prefix") if isinstance(sub_cfg, dict) else getattr(sub_cfg, "env_prefix", None)
            env_prefix = env_prefix or f"BACKPROPAGATE_{sub_field_name.upper()}__"

            for field_name, field_info in sub_cls.model_fields.items():
                env_name = f"{env_prefix}{field_name.upper()}"
                # Redact any field flagged as secret via Field(json_schema_extra={"secret": True}).
                is_secret = bool(
                    getattr(field_info, "json_schema_extra", None)
                    and isinstance(field_info.json_schema_extra, dict)
                    and field_info.json_schema_extra.get("secret")
                )
                default = "<secret>" if is_secret else _safe_default(field_info)
                rows.append({
                    "env_var": env_name,
                    "default": default,
                    "type": _type_name(field_info.annotation),
                    "description": (field_info.description or "").strip(),
                })
    else:
        # pydantic-settings missing — fall back to the hand-curated list of
        # env vars the dataclass branch in config.py honours. Keep this list
        # in sync if the fallback grows.
        rows.append({
            "env_var": "BACKPROPAGATE_DEFER_FEATURE_DETECTION",
            "default": "",
            "type": "bool",
            "description": "Skip feature detection at import time; call refresh_features() manually.",
        })

    # Always include the structured-logging env vars from logging_config.py
    # (these are read via os.environ.get, not via pydantic-settings, so the
    # introspection above won't pick them up).
    logging_envs = [
        ("BACKPROPAGATE_LOG_LEVEL", "INFO", "str", "Structured-logger level: DEBUG / INFO / WARNING / ERROR."),
        ("BACKPROPAGATE_LOG_JSON", "0", "bool", "Emit JSON-formatted log lines instead of pretty console output."),
        ("BACKPROPAGATE_LOG_FILE", "", "path", "File path for log output; appended to in addition to stderr."),
        ("BACKPROPAGATE_DEFER_FEATURE_DETECTION", "", "bool", "Skip optional-feature detection at import time."),
    ]
    seen = {row["env_var"] for row in rows}
    for env_name, default, type_name, description in logging_envs:
        if env_name not in seen:
            rows.append({
                "env_var": env_name,
                "default": default,
                "type": type_name,
                "description": description,
            })

    rows.sort(key=lambda r: r["env_var"])
    return rows


def _safe_default(field_info: Any) -> str:
    """Return a printable representation of a pydantic FieldInfo default.

    Falls back to ``"<factory>"`` for default_factory fields (whose
    construction can be expensive or non-deterministic) and ``""`` for
    PydanticUndefined sentinels.
    """
    try:
        from pydantic_core import PydanticUndefined
    except ImportError:
        PydanticUndefined = object()  # type: ignore[assignment]

    default = getattr(field_info, "default", PydanticUndefined)
    if default is PydanticUndefined or default is Ellipsis:
        if getattr(field_info, "default_factory", None) is not None:
            return "<factory>"
        return ""
    if default is None:
        return "None"
    if isinstance(default, bool):
        # Lowercase bool default matches the BACKPROPAGATE_*=true/false convention.
        return "true" if default else "false"
    if isinstance(default, (list, tuple, dict)):
        # Avoid dumping a multi-line repr into a single table row.
        return str(default)
    return str(default)


def _type_name(annotation: Any) -> str:
    """Return a human-readable type name for a pydantic field annotation."""
    if annotation is None:
        return "any"
    name = getattr(annotation, "__name__", None)
    if name:
        return str(name)
    # Generic aliases like ``list[str]`` / ``str | None`` use __repr__.
    return str(annotation).replace("typing.", "")


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command.

    Supported flags (C-CLI-005, C-CLI-008, BRIDGE-F-003):
        --error-codes     dump the ERROR_CODES catalog (machine-readable)
        --env-vars        enumerate every BACKPROPAGATE_* env var with defaults
        --json            emit the system info OR env-vars list as JSON
    """
    from .config import settings
    from .feature_flags import (
        FEATURES,
        INSTALL_HINTS,
        get_gpu_info,
        get_system_info,
    )
    from .gpu_safety import get_gpu_status

    # C-CLI-005: ``backprop info --error-codes`` prints the centralised
    # exception-code catalog (description + default hint + retryable flag)
    # so operators can look up a code they see in a terminal without
    # grepping source. Defined in exceptions.py to keep the registry next to
    # the exception classes that populate it.
    if getattr(args, "error_codes", False):
        from .exceptions import print_error_code_catalog

        print_error_code_catalog()
        return EXIT_OK

    # BRIDGE-F-003: ``backprop info --env-vars`` enumerates every
    # BACKPROPAGATE_* env var the runtime reads, with defaults + descriptions.
    # The data is introspected from the pydantic-settings models in
    # config.py + the structured-logging knobs in logging_config.py, so the
    # output stays in sync with the runtime automatically — no separate
    # markdown reference to maintain. Pipe-friendly for grep / awk.
    if getattr(args, "env_vars", False):
        rows = _enumerate_env_vars()
        if getattr(args, "json", False):
            import json
            print(json.dumps(rows, indent=2, default=str))
            return EXIT_OK

        # Compute aligned column widths so the table stays scannable.
        if not rows:
            _print_info("No environment variables enumerated (config introspection unavailable).")
            return EXIT_OK
        env_w = max(len("ENV_VAR"), max(len(r["env_var"]) for r in rows))
        default_w = max(len("DEFAULT"), max(len(r["default"]) for r in rows))
        type_w = max(len("TYPE"), max(len(r["type"]) for r in rows))

        _print_header("Backpropagate Environment Variables")
        header = f"{'ENV_VAR':<{env_w}}  {'DEFAULT':<{default_w}}  {'TYPE':<{type_w}}  DESCRIPTION"
        print(f"{Colors.BOLD}{header}{Colors.RESET}")
        sep = "  ".join(["-" * env_w, "-" * default_w, "-" * type_w, "-----------"])
        print(f"{Colors.DIM}{sep}{Colors.RESET}")
        for row in rows:
            line = (
                f"{row['env_var']:<{env_w}}  "
                f"{row['default']:<{default_w}}  "
                f"{row['type']:<{type_w}}  "
                f"{row['description']}"
            )
            print(line)
        print()
        _print_info(f"Listed {len(rows)} environment variable(s). Override any of these by exporting them before running backprop.")
        return EXIT_OK

    # Collect everything once so --json and the human view stay in sync.
    sys_info = get_system_info()
    gpu_info = get_gpu_info()
    versions = _detect_installed_versions()

    if getattr(args, "json", False):
        # Machine-readable payload for support attachments. Importing json
        # lazily keeps cold-start cheap when --json isn't used.
        import json

        from . import __version__ as backprop_version

        payload: dict[str, Any] = {
            "backpropagate_version": backprop_version,
            "python_version": sys_info.get("python_version"),
            "platform": sys_info.get("platform"),
            "torch_version": versions.get("torch"),
            "cuda_version": sys_info.get("cuda_version"),
            "gpu": gpu_info,
            "features": dict(FEATURES),
            "package_versions": versions,
            "config": {
                "model": settings.model.name,
                "max_seq_length": settings.model.max_seq_length,
                "lora_r": settings.lora.r,
                "learning_rate": settings.training.learning_rate,
                "output_dir": settings.training.output_dir,
            },
        }
        print(json.dumps(payload, indent=2, default=str))
        return EXIT_OK

    _print_header("Backpropagate System Info")

    # Versions section — moved up so the operator sees the tool's own
    # version before anything else (C-CLI-008).
    print(f"\n{Colors.BOLD}Versions{Colors.RESET}")
    from . import __version__ as backprop_version
    _print_kv("backpropagate", backprop_version)
    _print_kv("transformers", versions.get("transformers", "not installed"))
    _print_kv("datasets", versions.get("datasets", "not installed"))
    _print_kv("trl", versions.get("trl", "not installed"))
    _print_kv("peft", versions.get("peft", "not installed"))
    _print_kv("unsloth", versions.get("unsloth", "not installed"))

    # System info
    print(f"\n{Colors.BOLD}System{Colors.RESET}")
    _print_kv("Python", sys_info.get("python_version", "unknown"))
    _print_kv("Platform", sys_info.get("platform", "unknown"))
    _print_kv("PyTorch", versions.get("torch", "not installed"))
    _print_kv("CUDA", sys_info.get("cuda_version", "not available"))

    # GPU info — richer block (C-CLI-008): surface device count + per-device
    # VRAM via torch when available. Falls back to the existing gpu_info dict
    # when torch isn't installed.
    if gpu_info and gpu_info.get("available", True):
        print(f"\n{Colors.BOLD}GPU{Colors.RESET}")
        # The dict from feature_flags.get_gpu_info uses the underlying torch
        # field names (device_name, memory_total, device_count). Fall back to
        # the legacy names (name, vram_total_gb) if a future refactor renames.
        device_name = gpu_info.get("device_name") or gpu_info.get("name", "unknown")
        device_count = gpu_info.get("device_count", 1)
        _print_kv("Device", str(device_name))
        _print_kv("Device count", str(device_count))
        mem_total = gpu_info.get("memory_total")
        if mem_total is not None:
            _print_kv("VRAM", f"{mem_total / (1024 ** 3):.1f} GB")
        elif gpu_info.get("vram_total_gb"):
            _print_kv("VRAM", f"{gpu_info['vram_total_gb']:.1f} GB")
        if gpu_info.get("vram_free_gb"):
            _print_kv("VRAM Free", f"{gpu_info['vram_free_gb']:.1f} GB")

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

    # Features — also surface the install hint for each missing feature
    # (C-CLI-015 wasn't in scope but pairs naturally with C-CLI-008).
    print(f"\n{Colors.BOLD}Features{Colors.RESET}")
    for feature, available in FEATURES.items():
        feature_status = (
            f"{Colors.GREEN}[+]{Colors.RESET}"
            if available
            else f"{Colors.DIM}[-]{Colors.RESET}"
        )
        print(f"  {feature_status} {feature}")
        if not available:
            hint = INSTALL_HINTS.get(feature)
            if hint:
                print(f"      install with: {Colors.DIM}{hint}{Colors.RESET}")

    # Config
    print(f"\n{Colors.BOLD}Configuration{Colors.RESET}")
    _print_kv("Model", settings.model.name)
    _print_kv("Max seq length", str(settings.model.max_seq_length))
    _print_kv("LoRA r", str(settings.lora.r))
    _print_kv("Learning rate", str(settings.training.learning_rate))
    _print_kv("Output dir", settings.training.output_dir)

    # Pointer to the error-codes catalog and support flow (C-CLI-005,
    # C-CLI-008). Cheap signal that survives copy-paste into a support ticket.
    print(f"\n{Colors.BOLD}Support{Colors.RESET}")
    _print_info("For the full error-code catalog, run: backprop info --error-codes")
    _print_info("For support payloads, run: backprop info --json and share the output.")

    return EXIT_OK


# =============================================================================
# COMMAND: config
# =============================================================================

def cmd_ui(args: argparse.Namespace) -> int:
    """
    Execute the ui command to launch the Reflex web interface.

    The Web UI migrated from Gradio to Reflex in v1.1.0 (2026-05-21). This
    handler subprocess-launches ``reflex run`` from the directory containing
    ``rxconfig.py``. All validation runs BEFORE the subprocess launch — auth
    shape, share-without-auth refuse-to-start, host-without-auth refuse-to-
    start — so misconfigured launches fail loudly on the CLI side regardless
    of what the UI framework does.

    Post-Wave-6 (v1.2.0): with ``ENFORCEMENT_AVAILABLE=True`` the Reflex UI
    enforces the auth contract via FastAPI middleware. ``--auth`` is now a
    normal flag that flows through ``validate_auth_shape`` and into the
    subprocess via ``BACKPROPAGATE_UI_AUTH``. What remains gated:

    * ``--share`` without ``--auth`` — a public URL with no auth is the bug
      v1.2 closed; refuses with ``RUNTIME_UI_AUTH_NOT_ENFORCED``.
    * ``--host`` with a non-loopback bind without ``--auth`` — DNS-rebinding
      defense per DESIGN_BRIEF; refuses with ``RUNTIME_UI_AUTH_NOT_ENFORCED``.

    Exit codes (Ship Gate B2):
        0   UI launched and exited cleanly (including Ctrl+C)
        1   user error — UI extra not installed OR malformed --auth shape
        2   runtime error — Reflex subprocess failure OR
            RUNTIME_UI_AUTH_NOT_ENFORCED when --share / non-loopback --host
            were passed without --auth (or when ENFORCEMENT_AVAILABLE is
            False at runtime and --auth was requested).
    """
    # Verify Reflex is installed before we do anything else.
    try:
        import reflex  # noqa: F401
    except ImportError:
        _print_error("UI dependencies not installed")
        _print_info("Install with: pip install backpropagate[ui]")
        if args.verbose:
            logger.exception("Import error details")
        return EXIT_USER_ERROR

    # FRONTEND-A-001 cross-cutting fix (bridge half): load the enforcement
    # availability flag from the UI app. The Reflex UI does not yet enforce
    # the auth contract we advertise via --auth, so we must refuse to start
    # any invocation that asks for it. Lazy import so the [ui] extra missing
    # surfaces as the ImportError above, not as a confusing AttributeError.
    try:
        from .ui_app.auth import ENFORCEMENT_AVAILABLE
    except ImportError:
        # ui_app/auth.py is shipped with the package, not gated on [ui]. If
        # it cannot be imported, treat enforcement as unavailable so the
        # refuse-to-start path below still fires for --auth.
        ENFORCEMENT_AVAILABLE = False  # type: ignore[assignment]

    # Pull in the framework-agnostic auth-shape validator so a malformed
    # --auth fails on the CLI side before we hand it to the subprocess.
    try:
        from .ui_security import validate_auth_shape
    except ImportError:
        # ui_security pulls gradio in for type hints; if [ui] isn't installed
        # the import above already failed, but be defensive.
        _print_error("UI security helpers not importable")
        return EXIT_USER_ERROR

    _print_header("Backpropagate UI")
    _print_info(f"Port: {args.port}")

    # ------------------------------------------------------------------ #
    # Post-Wave-6 (ENFORCEMENT_AVAILABLE=True): the auth middleware now
    # honors the --auth contract per request, so the Wave 3.5 refuse-to-start
    # block on `args.auth is not None` is no longer correct — --auth flows
    # through validate_auth_shape and into the Reflex subprocess via the
    # BACKPROPAGATE_UI_AUTH env var. What remains gated:
    #   * --share without --auth — a public URL with no auth is the bug
    #     v1.2 closed; preserve as a hard error.
    #   * --host with a non-loopback bind without --auth — same threat
    #     (DNS-rebinding / LAN-discovery) per DESIGN_BRIEF; require --auth.
    # ENFORCEMENT_AVAILABLE is still read so a downgrade (test stubs, missing
    # ui_app, etc.) trips the auth-required gates even when --auth was passed.
    # ------------------------------------------------------------------ #
    if args.share and args.auth is None:
        raise BackpropagateError(
            "--share requires --auth user:pass; the auth middleware enforces "
            "requests post-v1.2.0.",
            suggestion=(
                "Pass --auth user:pass with a non-empty username and password. "
                "Without --auth a public --share URL would expose the UI "
                "anonymously, which is the bug v1.2 closed."
            ),
            code="RUNTIME_UI_AUTH_NOT_ENFORCED",
        )

    _LOOPBACK_BINDS = ("127.0.0.1", "localhost", "::1")
    requested_host = getattr(args, "host", None)
    if (
        requested_host is not None
        and requested_host not in _LOOPBACK_BINDS
        and args.auth is None
    ):
        raise BackpropagateError(
            f"--host {requested_host!r} binds beyond loopback and requires "
            "--auth user:pass; the auth middleware enforces requests "
            "post-v1.2.0.",
            suggestion=(
                "Pass --auth user:pass to protect the non-loopback bind, or "
                "drop --host to keep the UI on 127.0.0.1. DNS-rebinding "
                "defense requires the middleware to be live."
            ),
            code="RUNTIME_UI_AUTH_NOT_ENFORCED",
        )

    # If the runtime claims enforcement is unavailable (e.g. ui_app.auth
    # import failed) but auth was requested, refuse rather than silently
    # launching an unprotected UI with a "Auth: enabled" log line.
    if args.auth is not None and not ENFORCEMENT_AVAILABLE:
        raise BackpropagateError(
            "--auth was requested but the Reflex UI's auth middleware is not "
            "available at runtime; refusing to launch an unprotected UI.",
            suggestion=(
                "Reinstall the [ui] extra to restore backpropagate.ui_app.auth, "
                "or drop --auth and use SSH port-forwarding for remote access: "
                "ssh -L 7860:localhost:7860 <host>."
            ),
            code="RUNTIME_UI_AUTH_NOT_ENFORCED",
        )

    # Parse the --auth credentials into a (username, password) tuple. Post-
    # v1.2.0 the middleware enforces auth at runtime (ENFORCEMENT_AVAILABLE
    # is True) and the refuse-to-start gates above only fire for --share or
    # --host with a non-loopback bind WITHOUT --auth — meaning any --auth
    # invocation reaches this branch and the parsed tuple flows into the
    # subprocess env as BACKPROPAGATE_UI_AUTH.
    auth = None
    if args.auth:
        try:
            username, password = args.auth.split(":", 1)
            auth = (username, password)
        except ValueError:
            _print_error("Invalid auth format. Use --auth username:password")
            return EXIT_USER_ERROR

    # FB-012: validate the shape at the CLI boundary so a malformed
    # --auth produces a structured BackpropagateError before subprocess.
    try:
        validate_auth_shape(auth)
    except BackpropagateError as e:
        _print_error(f"{e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        return EXIT_USER_ERROR

    # Resolve the directory containing rxconfig.py. Reflex requires the
    # ``app_name`` package to be a direct subdirectory of the cwd, so we use
    # the package dir (``.../backpropagate/``) which ships
    # ``rxconfig.py`` + ``ui_app/`` side-by-side.
    package_dir = Path(__file__).resolve().parent
    rx_config = package_dir / "rxconfig.py"
    if not rx_config.exists():
        _print_error(
            f"rxconfig.py not found at {rx_config}. "
            "The Reflex UI requires the package-side rxconfig.py to be present."
        )
        _print_info(
            "Hint: reinstall the [ui] extra, or run "
            "`python -m backpropagate.ui_app.app` manually."
        )
        return EXIT_RUNTIME_ERROR

    # Set env vars that Reflex's state can pick up. ``BACKPROPAGATE_UI_AUTH``
    # is the agreed handoff for the Reflex side to enforce per-request auth
    # via FastAPI middleware once Phase 3 wires it. For Phase 1 the variable
    # is exported but Reflex doesn't read it yet.
    env = os.environ.copy()
    # BRIDGE-B-001: strip ambient BACKPROPAGATE_UI_AUTH when --auth not passed;
    # prevents ambient-env bypass of refuse-to-start once ENFORCEMENT_AVAILABLE
    # flips. Without this, `BACKPROPAGATE_UI_AUTH=u:p backprop ui` (no --auth)
    # would pass the CLI gate then silently activate auth in the Reflex child.
    if args.auth is None:
        env.pop("BACKPROPAGATE_UI_AUTH", None)
    if auth:
        env["BACKPROPAGATE_UI_AUTH"] = f"{auth[0]}:{auth[1]}"
    env["BACKPROPAGATE_UI_PORT"] = str(args.port)
    # Communicate the bind address so the middleware can enforce a
    # Host-header allow-list (DNS-rebinding defense). When --host is omitted
    # we still set the var explicitly to '127.0.0.1' so the middleware
    # never has to guess.
    env["BACKPROPAGATE_UI_HOST_BIND"] = requested_host or "127.0.0.1"
    # When --share is active the middleware also accepts the tunnel host.
    # No --share-host flag yet, so pass an empty string for now; a future
    # CLI knob can populate this with the operator-provided tunnel domain.
    if args.share:
        env["BACKPROPAGATE_UI_SHARE_HOST"] = ""

    # Reflex's port convention: the frontend serves on --frontend-port and
    # the backend on --backend-port. We map --port to the frontend (what
    # users hit in the browser) and the backend gets port+1.
    cmd = [
        sys.executable,
        "-m",
        "reflex",
        "run",
        "--frontend-port",
        str(args.port),
        "--backend-port",
        str(args.port + 1),
    ]

    try:
        print()
        _print_info("Launching Reflex interface...")
        result = subprocess.run(cmd, env=env, cwd=str(package_dir))
        return result.returncode if result.returncode is not None else EXIT_OK
    except KeyboardInterrupt:
        print()
        _print_info("UI stopped")
        return EXIT_OK
    except FileNotFoundError:
        _print_error(
            "Failed to launch Reflex — interpreter not found. "
            "This usually means the [ui] extra wasn't fully installed."
        )
        _print_info("Install with: pip install backpropagate[ui]")
        return EXIT_USER_ERROR
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
    """Execute the config command.

    Exit codes (Ship Gate B2):
        0   show / dump succeeded
        1   --set or --reset (not implemented) was passed
        130 interrupted (Ctrl+C)
    """

    from .config import settings

    try:
        _print_header("Backpropagate Configuration")

        # C-CLI-003: ``--set`` and ``--reset`` previously printed an INFO line
        # ("planned") and returned exit code 0 — silently succeeding a feature
        # that did nothing. A CI script chaining
        # ``backprop config --set foo=bar && next_step`` would advance unchanged.
        # Surface the unimplemented state as an error + EXIT_USER_ERROR so
        # automation can detect it. The words "planned" and "environment" are
        # preserved so the existing test assertions (which check for those
        # substrings, not exit code) keep matching when we update them.
        if args.reset:
            _print_error(
                "Config reset via CLI is not implemented (planned). "
                "Reset by removing BACKPROPAGATE_* environment variables or your "
                ".env file, then re-run."
            )
            return EXIT_USER_ERROR

        if args.set:
            _print_error(
                "Config editing via --set is not implemented (planned). "
                "Set BACKPROPAGATE_* environment variables or edit your .env "
                "file directly."
            )
            return EXIT_USER_ERROR

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

        # BRIDGE-A-012 (Stage C amend): return the EXIT_OK constant so a
        # future renumber of the exit codes propagates uniformly. Previously
        # this returned the bare integer literal 0, which would silently
        # disagree with EXIT_OK after such a renumber.
        return EXIT_OK
    except KeyboardInterrupt:
        # BRIDGE-A-012 (Stage C amend): every other subcommand handler maps
        # Ctrl+C to EXIT_INTERRUPTED (130) per the Ship Gate B2 contract; the
        # config handler was the lone outlier exiting 1 because it had no
        # KeyboardInterrupt branch. The body is fast in practice but the
        # symmetry matters — operators wiring `backprop config --show` into a
        # CI watchdog rely on the same exit code shape across subcommands.
        print()
        _print_warning("Config display interrupted by user")
        return EXIT_INTERRUPTED


# =============================================================================
# COMMAND: resume (F-002)
# =============================================================================

def cmd_resume(args: argparse.Namespace) -> int:
    """Execute the ``backprop resume <run_id>`` subcommand (F-002)."""
    import uuid

    from .checkpoints import RunHistoryManager
    from .logging_config import bind_run_context, get_logger
    from .multi_run import MultiRunTrainer
    from .trainer import Trainer, TrainingCallback

    _print_header("Backpropagate Resume")

    # BRIDGE-B-002: reuse main()'s cli_run_id so resume's structured logs
    # carry the same correlation token operators see at the top of stderr,
    # matching cmd_train / cmd_multi_run / cmd_export. Falling back to a
    # fresh UUID keeps the handler robust when called directly from tests
    # that bypass main().
    cli_run_id_full = getattr(args, "cli_run_id", None) or uuid.uuid4().hex
    cli_run_id = cli_run_id_full[:12]
    try:
        bind_run_context(run_id=cli_run_id_full, subcommand="resume")
    except Exception:  # noqa: BLE001  # nosec B110 — best-effort observability; must not abort CLI
        pass
    try:
        get_logger(__name__).info(
            "resume_invoked",
            cli_run_id=cli_run_id_full,
            target_run_id=getattr(args, "run_id", None),
        )
    except Exception:  # noqa: BLE001  # nosec B110 — best-effort observability; must not abort CLI
        pass
    print(
        f"[INFO] Run ID: {cli_run_id} — share with support if asking for help.",
        file=sys.stderr,
    )

    history_dir = Path(args.output).expanduser()
    if not history_dir.exists():
        _print_error(f"No history directory: {history_dir}")
        _print_info(
            "Pass --output <dir> to point at the output directory used by the original session."
        )
        return EXIT_USER_ERROR

    manager = RunHistoryManager(str(history_dir))
    record = manager.get_run(args.run_id)
    if record is None:
        _print_error(f"No run matching '{args.run_id}' in {history_dir}")
        return EXIT_USER_ERROR

    run_id = str(record.get("run_id"))
    session_kind = record.get("session_kind") or "single_run"
    model = record.get("model_name") or "Qwen/Qwen2.5-7B-Instruct"
    dataset = args.data or record.get("dataset_info")
    hp = record.get("hyperparameters") or {}

    _print_info(f"Run ID: {run_id}")
    _print_info(f"Session: {session_kind}")
    _print_info(f"Model: {model}")
    _print_info(f"Dataset: {dataset}")
    if record.get("status") == "completed":
        _print_warning(
            "This run is already marked as completed — resuming will continue it."
        )

    try:
        if session_kind == "multi_run":
            from .multi_run import MergeMode, MultiRunConfig

            config = MultiRunConfig(
                num_runs=int(hp.get("num_runs") or 5),
                steps_per_run=int(hp.get("steps_per_run") or 100),
                samples_per_run=int(hp.get("samples_per_run") or 1000),
                merge_mode=MergeMode(hp.get("merge_mode") or "slao"),
                checkpoint_dir=str(history_dir),
            )
            mr_trainer = MultiRunTrainer(
                model=model,
                config=config,
                resume_from=run_id,
            )
            result = mr_trainer.run(dataset)
            _print_success("Multi-run resume complete!")
            _print_kv("Total runs", str(result.total_runs))
            _print_kv("Final loss", f"{result.final_loss:.4f}")
            return EXIT_OK

        # Single-run resume.
        trainer = Trainer(
            model=model,
            lora_r=int(hp.get("lora_r") or 16),
            learning_rate=float(hp.get("learning_rate") or 2e-4),
            output_dir=str(history_dir),
        )
        callback = TrainingCallback()
        run = trainer.train(
            dataset=dataset,
            steps=int(hp.get("max_steps") or 100),
            samples=hp.get("max_samples"),
            callback=callback,
            resume_from=run_id,
        )
        trainer.save(args.output)
        _print_success("Resume complete!")
        _print_kv("Final loss", f"{run.final_loss:.4f}")
        return EXIT_OK

    except KeyboardInterrupt:
        print()
        _print_warning("Resume interrupted by user")
        return EXIT_INTERRUPTED
    except BackpropagateError as e:
        _print_error(f"Resume failed: {e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        if args.verbose:
            _print_error(f"Resume failed: {e}")
            import traceback
            traceback.print_exc()
        else:
            _print_error_redacted(e, prefix="Resume failed: ")
            _print_info("Run with --verbose for full traceback")
        return EXIT_RUNTIME_ERROR


# =============================================================================
# COMMAND: push (F-001)
# =============================================================================

def cmd_push(args: argparse.Namespace) -> int:
    """Execute the ``backprop push`` subcommand (F-001).

    Uploads a local export directory to the Hugging Face Hub.
    """
    from .export import push_to_hub

    _print_header("Backpropagate Hub Push")

    local_path = Path(args.local_path).expanduser()
    if not local_path.exists():
        _print_error(f"Local path does not exist: {local_path}")
        _print_info(
            "Pass a directory created by `backprop export` "
            "or `backprop train --output <dir>`."
        )
        return EXIT_USER_ERROR

    if not args.repo:
        _print_error("No target repo specified.")
        _print_info("Pass --repo username/model-name.")
        return EXIT_USER_ERROR

    _print_info(f"Local path: {local_path}")
    _print_info(f"Repo: {args.repo}")
    if args.private:
        _print_info("Visibility: private")
    if args.include_base:
        _print_info("Including base model files (--include-base)")

    try:
        url = push_to_hub(
            local_path=local_path,
            repo_id=args.repo,
            token=args.token,
            private=args.private,
            include_base=args.include_base,
        )
    except ExportError as e:
        code = getattr(e, "code", None)
        _print_error(f"Push failed: {e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        # BRIDGE-F-002 (catalog cleanup): branch on the canonical ERROR_CODES
        # entries only. Wave 5 backend agent promoted HUB_PUSH_INVALID_REPO /
        # HUB_PUSH_NOT_FOUND / HUB_PUSH_NETWORK / HUB_PUSH_UNKNOWN into
        # exceptions.ERROR_CODES, so the bridge no longer needs the legacy
        # HUB_PUSH_AUTH fallback or the local catalog. User-error codes
        # (operator-fixable) exit 1; everything else exits 2.
        if code in ("INPUT_AUTH_REQUIRED", "HUB_PUSH_INVALID_REPO", "HUB_PUSH_NOT_FOUND"):
            return EXIT_USER_ERROR
        return EXIT_RUNTIME_ERROR
    except BackpropagateError as e:
        _print_error(f"Push failed: {e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        if args.verbose:
            _print_error(f"Push failed: {e}")
            import traceback

            traceback.print_exc()
        else:
            _print_error_redacted(e, prefix="Push failed: ")
            _print_info("Run with --verbose for full traceback")
        return EXIT_RUNTIME_ERROR

    _print_success(f"Pushed to Hub: {url}")
    return EXIT_OK


# =============================================================================
# COMMAND: list-runs (F-003)
# =============================================================================

def _humanize_timestamp(value: Any) -> str:
    """Return a short, human-readable timestamp string."""
    if not value:
        return "-"
    text = str(value)
    # Trim subsecond precision for display: 2026-05-21T13:42:18.123456 -> 2026-05-21 13:42
    text = text.replace("T", " ")
    if "." in text:
        text = text.split(".", 1)[0]
    return text[:16]  # YYYY-MM-DD HH:MM


def _format_run_row(run: dict[str, Any]) -> dict[str, str]:
    """Produce a row of display strings for one run history entry."""
    run_id = str(run.get("run_id") or "-")
    return {
        "run_id": run_id[:12],
        "started_at": _humanize_timestamp(run.get("started_at") or run.get("timestamp")),
        "model": str(run.get("model_name") or "-")[:32],
        "dataset": str(run.get("dataset_info") or "-")[:32],
        "status": str(run.get("status") or "-"),
        "final_loss": (
            f"{run['final_loss']:.4f}"
            if isinstance(run.get("final_loss"), (int, float))
            else "-"
        ),
    }


def cmd_list_runs(args: argparse.Namespace) -> int:
    """Execute the ``backprop list-runs`` subcommand (F-003)."""
    from .checkpoints import RunHistoryManager

    history_dir = Path(args.output).expanduser()
    if not history_dir.exists():
        _print_warning(f"No history found at {history_dir}")
        return EXIT_OK

    manager = RunHistoryManager(str(history_dir))
    try:
        runs = manager.list_runs(status=args.status, limit=args.limit)
    except ValueError as e:
        _print_error(str(e))
        return EXIT_USER_ERROR

    if args.json:
        import json
        print(json.dumps(runs, indent=2, default=str))
        return EXIT_OK

    if not runs:
        _print_info("No training runs recorded.")
        if args.status:
            _print_info(f"(Filter: status={args.status})")
        return EXIT_OK

    # Aligned-column display.
    rows = [_format_run_row(r) for r in runs]
    headers = ["RUN_ID", "STARTED", "MODEL", "DATASET", "STATUS", "FINAL_LOSS"]
    keys = ["run_id", "started_at", "model", "dataset", "status", "final_loss"]
    widths = [
        max(len(headers[i]), *(len(row[keys[i]]) for row in rows))
        for i in range(len(headers))
    ]

    _print_header("Training runs")
    header_line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    print(f"{Colors.BOLD}{header_line}{Colors.RESET}")
    print(Colors.DIM + "  ".join("-" * widths[i] for i in range(len(headers))) + Colors.RESET)
    for row in rows:
        line = "  ".join(row[keys[i]].ljust(widths[i]) for i in range(len(headers)))
        print(line)
    print()
    _print_info(f"Listed {len(rows)} run(s) from {history_dir}")
    return EXIT_OK


# =============================================================================
# COMMAND: runs (BRIDGE-F-001 — versioned run-history data API for the UI)
# =============================================================================

# Schema version for ``backprop runs --json`` output. Bump on breaking
# changes to the dict shape — additive field additions stay at "1".
RUNS_JSON_SCHEMA_VERSION = "1"


def _build_runs_payload(
    runs: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Project run-history entries into the BRIDGE-F-001 versioned payload.

    The Reflex UI consumes this via ``subprocess.run(['backprop', 'runs',
    '--json'])`` (or directly via this helper in-process; both routes must
    produce byte-identical output to support snapshot tests).

    Field renames in the underlying RunHistoryManager entries are
    intentionally NOT propagated to the payload: every payload field is
    explicitly projected from the source dict so a schema bump is visible
    in this function rather than silently breaking the UI.

    Args:
        runs: Raw run-history entries (newest first), as returned by
            ``RunHistoryManager.list_runs(...)``.
        output_dir: Resolved output directory, included in the payload so
            consumers can correlate runs back to a checkpoint root.

    Returns:
        ``{"schema_version": "1", "generated_at": <iso8601>,
        "output_dir": <str>, "runs": [...]}`` where each run dict has a
        stable set of fields suitable for table rendering.
    """
    from datetime import datetime, timezone

    projected: list[dict[str, Any]] = []
    for entry in runs:
        # Pull loss summary from final_loss + optional loss_history.
        final_loss = entry.get("final_loss")
        loss_history = entry.get("loss_history") or []
        min_loss: float | None = None
        if isinstance(loss_history, list) and loss_history:
            try:
                numeric = [float(x) for x in loss_history if isinstance(x, (int, float))]
                if numeric:
                    min_loss = min(numeric)
            except (TypeError, ValueError):
                min_loss = None

        # Duration: explicit field wins; otherwise compute from timestamps.
        duration = entry.get("duration_seconds")
        if duration is None:
            started = entry.get("started_at")
            completed = entry.get("completed_at")
            if started and completed:
                try:
                    s = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
                    c = datetime.fromisoformat(str(completed).replace("Z", "+00:00"))
                    duration = (c - s).total_seconds()
                except (ValueError, TypeError):
                    duration = None

        projected.append({
            "run_id": entry.get("run_id"),
            "status": entry.get("status"),
            "session_kind": entry.get("session_kind"),
            "model": entry.get("model_name"),
            "dataset": entry.get("dataset_info"),
            "duration_seconds": duration,
            "started_at": entry.get("started_at") or entry.get("timestamp"),
            "completed_at": entry.get("completed_at"),
            "checkpoint_path": entry.get("checkpoint_path"),
            "loss": {
                "final": float(final_loss) if isinstance(final_loss, (int, float)) else None,
                "min": min_loss,
            },
        })

    return {
        "schema_version": RUNS_JSON_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "runs": projected,
    }


def cmd_runs(args: argparse.Namespace) -> int:
    """Execute ``backprop runs`` — versioned run-history data API (BRIDGE-F-001).

    Emits a JSON payload with a ``schema_version`` field so consumers (the
    Reflex UI today; future operator dashboards) can detect breaking
    changes. The human-readable form delegates to the same projection so
    operators eyeballing the output and CI scripts parsing the JSON see
    identical data.

    Flags:
        --status {running,completed,failed}  filter by lifecycle state
        --limit N                            cap to N most-recent runs
        --json                               emit the full payload as JSON
        --output DIR                         override the output directory
    """
    from .checkpoints import RunHistoryManager

    history_dir = Path(args.output).expanduser()
    if not history_dir.exists():
        if getattr(args, "json", False):
            # Even when the directory is missing, emit a well-formed empty
            # payload so the UI doesn't have to special-case "no runs yet".
            import json
            empty = _build_runs_payload([], history_dir)
            print(json.dumps(empty, indent=2, default=str))
            return EXIT_OK
        _print_warning(f"No history found at {history_dir}")
        return EXIT_OK

    manager = RunHistoryManager(str(history_dir))
    try:
        runs = manager.list_runs(status=args.status, limit=args.limit)
    except ValueError as e:
        _print_error(str(e))
        return EXIT_USER_ERROR

    payload = _build_runs_payload(runs, history_dir)

    if getattr(args, "json", False):
        import json
        print(json.dumps(payload, indent=2, default=str))
        return EXIT_OK

    if not payload["runs"]:
        _print_info("No training runs recorded.")
        if args.status:
            _print_info(f"(Filter: status={args.status})")
        return EXIT_OK

    # Human view: brief summary + pointer to list-runs for the legacy aligned
    # table. ``runs`` is intentionally JSON-first; the existing ``list-runs``
    # subcommand stays the canonical pretty-printer.
    _print_header(f"Training runs (schema v{payload['schema_version']})")
    _print_kv("Generated", payload["generated_at"])
    _print_kv("Output dir", payload["output_dir"])
    _print_kv("Run count", str(len(payload["runs"])))
    print()
    for run in payload["runs"]:
        run_id = str(run.get("run_id") or "-")[:12]
        status = str(run.get("status") or "-")
        model = str(run.get("model") or "-")
        final_loss = run.get("loss", {}).get("final")
        loss_str = f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else "-"
        print(f"  {run_id:<14}  {status:<10}  {model:<32}  loss={loss_str}")
    print()
    _print_info("For the full payload, pass --json. For the aligned legacy table, run `backprop list-runs`.")
    return EXIT_OK


# =============================================================================
# COMMAND: show-run (F-003)
# =============================================================================

def cmd_show_run(args: argparse.Namespace) -> int:
    """Execute the ``backprop show-run <run_id>`` subcommand (F-003)."""
    from .checkpoints import RunHistoryManager

    history_dir = Path(args.output).expanduser()
    if not history_dir.exists():
        _print_error(f"No history directory: {history_dir}")
        _print_info("Pass --output <dir> to point at the output directory used during training.")
        return EXIT_USER_ERROR

    manager = RunHistoryManager(str(history_dir))
    run = manager.get_run(args.run_id)
    if run is None:
        _print_error(f"No run matching '{args.run_id}' in {history_dir}")
        return EXIT_USER_ERROR

    if args.json:
        import json
        print(json.dumps(run, indent=2, default=str))
        return EXIT_OK

    _print_header(f"Run {str(run.get('run_id') or '-')[:12]}")
    _print_kv("Run ID", str(run.get("run_id") or "-"))
    _print_kv("Status", str(run.get("status") or "-"))
    _print_kv("Session kind", str(run.get("session_kind") or "-"))
    _print_kv("Started", _humanize_timestamp(run.get("started_at")))
    _print_kv("Completed", _humanize_timestamp(run.get("completed_at")))
    _print_kv("Model", str(run.get("model_name") or "-"))
    _print_kv("Dataset", str(run.get("dataset_info") or "-"))
    dh = run.get("dataset_hash")
    if dh:
        _print_kv("Dataset sha256 (16)", str(dh))
    if isinstance(run.get("final_loss"), (int, float)):
        _print_kv("Final loss", f"{run['final_loss']:.4f}")
    if isinstance(run.get("steps"), int):
        _print_kv("Steps", str(run["steps"]))
    if isinstance(run.get("duration_seconds"), (int, float)):
        _print_kv("Duration", f"{run['duration_seconds']:.1f}s")
    if run.get("checkpoint_path"):
        _print_kv("Checkpoint", str(run["checkpoint_path"]))

    if run.get("failure_reason"):
        _print_kv("Failure reason", str(run["failure_reason"]))

    # Hyperparameters
    hp = run.get("hyperparameters") or {}
    if hp:
        print(f"\n{Colors.BOLD}Hyperparameters{Colors.RESET}")
        for key, value in hp.items():
            _print_kv(key, str(value))

    # Loss history
    losses = run.get("loss_history") or []
    if losses:
        head = losses[:5]
        tail = losses[-3:] if len(losses) > 8 else []
        sample = ", ".join(f"{x:.4f}" for x in head)
        if tail:
            sample += ", ..., " + ", ".join(f"{x:.4f}" for x in tail)
        print(f"\n{Colors.BOLD}Loss history{Colors.RESET}  ({len(losses)} points)")
        _print_kv("samples", sample)

    # Merge history (multi-run only)
    merge_history = run.get("merge_history") or []
    if merge_history:
        print(f"\n{Colors.BOLD}SLAO merge history{Colors.RESET}")
        for entry in merge_history:
            idx = entry.get("run_index", "?")
            _print_kv(f"run_{idx}", str(entry.get("result"))[:80])

    # Export paths
    exports = run.get("export_paths") or []
    if exports:
        print(f"\n{Colors.BOLD}Exports{Colors.RESET}")
        for path in exports:
            _print_kv("path", str(path))

    return EXIT_OK


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
        # F-018: default aligned with config.py's ModelConfig.name. Picking
        # the non-quantized form (Qwen/Qwen2.5-7B-Instruct) over the
        # unsloth-bnb-4bit form sidesteps the silent ImportError that fires
        # when a user has [ui] installed but not [unsloth] (which carries
        # bitsandbytes). Unsloth still picks up the 4-bit path at load time
        # if it's available; users who explicitly want the pre-quantized
        # variant can pass --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit.
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    train_parser.add_argument(
        "--data", "-d",
        # C-CLI-007: dropped ``required=True`` so a bare ``backprop train``
        # gets a friendly multi-line "no dataset specified, try one of these"
        # message in cmd_train instead of argparse's terse
        # ``the following arguments are required: --data/-d`` line. cmd_train
        # checks ``args.data is None`` and returns EXIT_USER_ERROR.
        default=None,
        help="Dataset path (JSONL, CSV) or HuggingFace dataset name",
    )
    train_parser.add_argument(
        "--steps",
        type=_positive_int,
        default=100,
        help="Number of training steps (default: 100; must be > 0)",
    )
    train_parser.add_argument(
        "--samples",
        type=_positive_int,
        default=None,
        help="Maximum samples to use from dataset (must be > 0)",
    )
    train_parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size (default: auto)",
    )
    train_parser.add_argument(
        "--lr",
        type=_positive_float,
        default=2e-4,
        help="Learning rate (default: 2e-4; must be > 0)",
    )
    train_parser.add_argument(
        "--lora-r",
        type=_positive_int,
        default=16,
        help="LoRA rank (default: 16; must be > 0)",
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
    # F-002: resume hint — reuses the same run_id and updates the on-disk
    # history record in place. Accepts a partial run_id prefix.
    train_parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_ID",
        help=(
            "Resume an existing run by run_id (or partial prefix). The run "
            "is recorded under the same run_id in run_history.json so "
            "downstream queries (`backprop show-run`) see one continuous "
            "session instead of two."
        ),
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
        # F-018: default aligned with config.py's ModelConfig.name; see the
        # train_parser --model comment for the [ui]-vs-[unsloth] rationale.
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    multi_parser.add_argument(
        "--data", "-d",
        # C-CLI-007: dropped ``required=True`` for symmetry with ``train``;
        # cmd_multi_run prints a friendly multi-line "no dataset specified"
        # message instead of argparse's terse required-argument error.
        default=None,
        help="Dataset path or HuggingFace dataset name",
    )
    multi_parser.add_argument(
        "--runs",
        type=_positive_int,
        default=5,
        help="Number of training runs (default: 5; must be > 0)",
    )
    multi_parser.add_argument(
        "--steps",
        type=_positive_int,
        default=100,
        help="Steps per run (default: 100; must be > 0)",
    )
    multi_parser.add_argument(
        "--samples",
        type=_positive_int,
        default=1000,
        help="Samples per run (default: 1000; must be > 0)",
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
    # F-002: resume hint — when set, picks up the latest checkpoint for the
    # matching run_id (or partial prefix) and continues from the next run.
    multi_parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_ID",
        help=(
            "Resume an existing multi-run by run_id (or partial prefix). The "
            "session re-uses the prior run_id and skips the runs that "
            "already completed."
        ),
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
        # C-CLI-007: positional is nargs='?' so a bare ``backprop export``
        # falls into cmd_export with a friendly message instead of argparse's
        # ``the following arguments are required: model_path`` line.
        nargs="?",
        default=None,
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
    # F-004: model card is emitted next to every export by default; this
    # flag is the explicit opt-out for users who'd rather author a card by
    # hand or who are exporting into a directory already covered by one.
    export_parser.add_argument(
        "--no-model-card",
        action="store_true",
        help=(
            "Do not emit model_card.md next to the export (default: emit). "
            "The card carries run_id / hyperparameters / loss curve / "
            "training duration / Ship Gate trust signals."
        ),
    )
    # F-001: one-shot export + Hub push. Equivalent to running
    # ``backprop export ... && backprop push ./output/... --repo <repo>``
    # but the export-and-push uses the freshly written model_card.md
    # without a second walk of the directory.
    export_parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        default=None,
        help=(
            "After a successful export, push the artifact to the Hugging "
            "Face Hub as REPO_ID (e.g. alice/qwen-finetune). Uses $HF_TOKEN "
            "or ~/.cache/huggingface/token unless --hub-token is passed."
        ),
    )
    export_parser.add_argument(
        "--hub-token",
        default=None,
        help="HF token override for --push-to-hub (default: env / cached login).",
    )
    export_parser.add_argument(
        "--hub-private",
        action="store_true",
        help="When pairing --push-to-hub, create the repo as private.",
    )
    export_parser.set_defaults(func=cmd_export)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information",
        description=(
            "Display backpropagate version, Python, PyTorch + CUDA, GPU, "
            "feature flags, dependency versions, and configuration. "
            "Use --error-codes to print the centralised exception-code "
            "catalog, or --json to emit a machine-readable payload suitable "
            "for support attachments."
        ),
    )
    info_parser.add_argument(
        "--error-codes",
        action="store_true",
        help=(
            "Print the centralised ERROR_CODES catalog (description + "
            "default hint + retryable flag) and exit."
        ),
    )
    info_parser.add_argument(
        "--env-vars",
        action="store_true",
        help=(
            "Enumerate every BACKPROPAGATE_* environment variable the runtime "
            "honours, with default value + type + description. Introspected "
            "from the pydantic-settings models so the output stays in sync "
            "with the runtime. Combine with --json for machine-readable form."
        ),
    )
    info_parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the system info (or --env-vars list) as JSON. Suitable for "
            "sharing in support tickets or feeding to grep / jq in CI scripts."
        ),
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

    # resume command (F-002)
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a crashed or interrupted training run",
        description=(
            "Re-run the train / multi-run command associated with a "
            "RunHistoryManager entry, continuing from the latest checkpoint "
            "tagged with the run_id."
        ),
    )
    resume_parser.add_argument(
        "run_id",
        help="The run_id to resume (or any unambiguous prefix).",
    )
    resume_parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory used by the original run (default: ./output)",
    )
    resume_parser.add_argument(
        "--data", "-d",
        default=None,
        help=(
            "Override the dataset path. By default the dataset is read from "
            "the recorded run history entry. Pass --data if you've moved the "
            "dataset file or want to point at a HF dataset name."
        ),
    )
    resume_parser.set_defaults(func=cmd_resume)

    # push command (F-001)
    push_parser = subparsers.add_parser(
        "push",
        help="Push a local export to the Hugging Face Hub",
        description=(
            "Upload a directory produced by `backprop export` (or "
            "`backprop train --output <dir>`) to a Hugging Face Hub repo. "
            "Mirrors the local model_card.md as the repo's README.md so the "
            "HF UI picks it up as the model card. Defaults to adapter-only "
            "uploads — pass --include-base to also push the base model."
        ),
    )
    push_parser.add_argument(
        "local_path",
        nargs="?",
        default=".",
        help="Local directory to upload (default: current directory)",
    )
    push_parser.add_argument(
        "--repo",
        required=True,
        help="Hugging Face Hub repo identifier, e.g. alice/qwen-finetune",
    )
    push_parser.add_argument(
        "--token",
        default=None,
        help=(
            "HF token (defaults to $HF_TOKEN, $HUGGING_FACE_HUB_TOKEN, or "
            "~/.cache/huggingface/token from `huggingface-cli login`)."
        ),
    )
    push_parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (default: public).",
    )
    push_parser.add_argument(
        "--include-base",
        action="store_true",
        help=(
            "Push the entire directory (base model + adapter). Default: "
            "adapter-only upload (smaller, faster, more useful)."
        ),
    )
    push_parser.set_defaults(func=cmd_push)

    # list-runs command (F-003)
    list_runs_parser = subparsers.add_parser(
        "list-runs",
        help="List recorded training runs",
        description=(
            "List runs from the on-disk run_history.json (populated by every "
            "`backprop train` / `backprop multi-run` invocation since v1.1.0)."
        ),
    )
    list_runs_parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory containing run_history.json (default: ./output)",
    )
    list_runs_parser.add_argument(
        "--status",
        choices=["running", "completed", "failed"],
        default=None,
        help="Filter by status (default: show all)",
    )
    list_runs_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=20,
        help="Maximum runs to display, most-recent first (default: 20)",
    )
    list_runs_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the aligned table",
    )
    list_runs_parser.set_defaults(func=cmd_list_runs)

    # runs command (BRIDGE-F-001) — versioned JSON data API consumed by the
    # Reflex UI's run-history page. Exposes the same data as `list-runs`
    # under a stable schema_version contract so frontend renames don't break
    # on field additions / removals.
    runs_parser = subparsers.add_parser(
        "runs",
        help="Emit run history as a versioned JSON payload (UI data API)",
        description=(
            "Emit the run history under a schema_version contract suitable "
            "for the Reflex Web UI and any external dashboard. The payload "
            "shape is documented in handbook/cli-reference.md. For the "
            "human-readable aligned table, use `backprop list-runs` instead."
        ),
    )
    runs_parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory containing run_history.json (default: ./output)",
    )
    runs_parser.add_argument(
        "--status",
        choices=["running", "completed", "failed"],
        default=None,
        help="Filter by status (default: show all)",
    )
    runs_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help=(
            "Maximum runs to include in the payload, most-recent first. "
            "Default: no cap (UI consumers typically paginate client-side)."
        ),
    )
    runs_parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the full versioned payload as JSON. Required for "
            "machine consumption (Reflex UI, CI scripts, jq pipelines)."
        ),
    )
    runs_parser.set_defaults(func=cmd_runs)

    # show-run command (F-003)
    show_run_parser = subparsers.add_parser(
        "show-run",
        help="Show detail for a single training run",
        description=(
            "Print the full record for a single training run, including "
            "hyperparameters, loss history, SLAO merge history, and any "
            "exports recorded against the run."
        ),
    )
    show_run_parser.add_argument(
        "run_id",
        help=(
            "The run_id to show. Partial-prefix matches are accepted as long "
            "as the prefix is unambiguous (typically 8-12 hex chars)."
        ),
    )
    show_run_parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory containing run_history.json (default: ./output)",
    )
    show_run_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human view",
    )
    show_run_parser.set_defaults(func=cmd_show_run)

    # ui command
    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch the Reflex web interface",
        description="Launch the Reflex (Radix UI) web interface for training, "
                    "export, and monitoring. Requires `pip install backpropagate[ui]`.",
    )
    ui_parser.add_argument(
        "--port", "-p",
        type=_port_int,
        default=7862,
        help="Port to run the server on (default: 7862; must be in range 1..65535)",
    )
    ui_parser.add_argument(
        "--host",
        default=None,
        help=(
            "Interface to bind (default: 127.0.0.1). Non-loopback values "
            "(e.g. 0.0.0.0, a LAN IP) require --auth — DNS-rebinding defense."
        ),
    )
    ui_parser.add_argument(
        "--share",
        action="store_true",
        help=(
            "Expose the UI via a public tunnel. Requires --auth user:pass; "
            "passing --share without --auth exits with "
            "RUNTIME_UI_AUTH_NOT_ENFORCED."
        ),
    )
    ui_parser.add_argument(
        "--auth",
        metavar="USER:PASS",
        help=(
            "Enable HTTP basic auth on the Reflex UI. Required when --share "
            "or a non-loopback --host is passed. Credentials are forwarded "
            "to the subprocess via BACKPROPAGATE_UI_AUTH."
        ),
    )
    ui_parser.set_defaults(func=cmd_ui)

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    See ``backprop --help`` for the documented Ship Gate B2 exit-code contract.

    BRIDGE-A-005: wraps ``args.func(args)`` in a last-resort exception net so
    unhandled errors honor the Ship Gate B2 exit-code contract (130 for
    Ctrl+C, 2 for unexpected crash) instead of falling through to Python's
    default exit code 1. Subcommand-level handlers still run first for
    friendly per-command messages; this catch-all only fires when something
    escapes them.

    BRIDGE-B-002: wires the structured-logging scaffolding into the CLI.
    Pre-fix, ``configure_logging()`` was never called and ``cli_run_id`` was
    never bound to structlog's contextvars — so ``BACKPROPAGATE_LOG_*`` env
    vars were no-ops and operators could not correlate the CLI's run_id with
    the Trainer's run_id across logs. Now ``main()``:

      1. Generates a single ``cli_run_id`` UUID at the top.
      2. Calls ``configure_logging()`` honouring
         ``BACKPROPAGATE_LOG_LEVEL`` / ``BACKPROPAGATE_LOG_JSON`` /
         ``BACKPROPAGATE_LOG_FILE`` env vars (and bumping level to ``DEBUG``
         when ``--verbose`` is set).
      3. Calls ``bind_run_context(run_id=cli_run_id)`` so every structured
         log line in the process carries the same correlation token.
      4. Stashes the CLI-level run_id on ``args.cli_run_id`` so subcommand
         handlers can re-emit / re-bind it without minting a second UUID.
    """
    import uuid

    from .logging_config import bind_run_context, configure_logging, get_logger

    try:
        parser = create_parser()
        args = parser.parse_args(argv)
    except KeyboardInterrupt:
        # KeyboardInterrupt during argparse (rare but possible if a custom
        # validator does I/O) should still exit 130 per the Ship Gate contract.
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED

    # BRIDGE-B-002: configure structured logging before ANY subcommand runs.
    # ``--verbose`` bumps the level to DEBUG so structlog emits the full
    # event stream; otherwise honour BACKPROPAGATE_LOG_LEVEL (default INFO).
    verbose = bool(getattr(args, "verbose", False))
    try:
        configure_logging(level="DEBUG" if verbose else None, force=True)
    except Exception:  # noqa: BLE001 — logging setup must not abort the CLI  # nosec B110
        # If logging setup itself fails (extremely rare — bad log file path,
        # permission denied), fall back silently so the subcommand still runs.
        # The traceback is intentionally swallowed so a misconfigured
        # BACKPROPAGATE_LOG_FILE can't crash the CLI on startup.
        pass

    # BRIDGE-B-002: mint one CLI-level run_id and bind it so every structured
    # log line emitted from this process carries it. Subcommand handlers
    # re-emit the same id to stderr (replacing the prior 12-char-only print)
    # so operators triaging a multi-hour overnight job can grep ONE token
    # across CLI output, trainer logs, and run_history.json.
    cli_run_id = uuid.uuid4().hex
    try:
        bind_run_context(run_id=cli_run_id, subcommand=getattr(args, "command", None))
    except Exception:  # noqa: BLE001 — context binding must not abort the CLI  # nosec B110
        pass
    args.cli_run_id = cli_run_id

    # Emit the CLI-level run_id once via the structured logger (auto-routed
    # to JSON when BACKPROPAGATE_LOG_JSON=1, pretty console otherwise). This
    # replaces the prior per-handler print("[INFO] Run ID:") lines so a JSON
    # consumer sees the id in a real event instead of as bare stderr text.
    try:
        get_logger(__name__).info(
            "cli_invoked",
            cli_run_id=cli_run_id,
            subcommand=getattr(args, "command", None),
        )
    except Exception:  # noqa: BLE001 — logging must not abort the CLI  # nosec B110
        pass

    if not args.command:
        # First-time-user friendliness (C-CLI-001): bare ``backprop`` invocation
        # prints the rich help text (subcommand list + examples + exit-code
        # table) instead of a single terse error line. The exit code stays
        # non-zero so CI scripts that asserted "no-subcommand is failure" keep
        # behaving the same — only the output content changes from one line
        # to the full --help block.
        parser.print_help(sys.stderr)
        return EXIT_USER_ERROR

    # Execute the command — wrapped in a last-resort exception net so the
    # Ship Gate B2 exit-code contract holds even when a subcommand's own
    # try/except missed something.
    try:
        result: int = args.func(args)
        return result
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except BackpropagateError as e:
        code = getattr(e, "code", None) or "RUNTIME"
        print(f"{code}: {e}", file=sys.stderr)
        if os.environ.get("BACKPROPAGATE_DEBUG"):
            import traceback
            traceback.print_exc()
        return EXIT_RUNTIME_ERROR
    except SystemExit:
        # Let argparse / library SystemExit propagate so tests that catch
        # SystemExit (e.g. -h flag invocations) see the same behavior.
        raise
    except Exception as e:  # noqa: BLE001 — last-resort safety net
        # The Ship Gate B2 contract promises exit code 2 for unexpected
        # crashes. The default Python behavior is exit code 1 + traceback.
        print(f"Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        if os.environ.get("BACKPROPAGATE_DEBUG"):
            import traceback
            traceback.print_exc()
        else:
            print(
                "Set BACKPROPAGATE_DEBUG=1 for the full traceback, or re-run "
                "the subcommand with --verbose for friendlier output.",
                file=sys.stderr,
            )
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
